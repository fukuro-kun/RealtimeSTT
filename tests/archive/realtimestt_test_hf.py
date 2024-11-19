"""
Echtzeit-Spracherkennung mit HuggingFace Whisper
===============================================

Diese Implementierung nutzt das HuggingFace Whisper-Modell für hochqualitative
Echtzeit-Spracherkennung in deutscher Sprache. Die Hauptmerkmale sind:

- Optimierte Spracherkennung mit dem deutschen Whisper-Modell
- Echtzeitfähige Audio-Verarbeitung mit PyAudio
- Robuste Sprach-/Pause-Erkennung mit WebRTC VAD
- Effiziente Audio-Resampling-Pipeline
- Umfassende Fehlerbehandlung und Logging
"""

# ====== Grundlegende System- und Utility-Imports ======
import os              # Betriebssystem-Interaktion, Umgebungsvariablen
import sys            # System-spezifische Parameter und Funktionen
import time           # Zeitbezogene Funktionen
import torch          # PyTorch für ML-Operationen
import ctypes         # Für Low-Level-Interaktion mit System-Bibliotheken
import logging        # Strukturiertes Logging
import warnings       # Warnung-Behandlung
from contextlib import contextmanager  # Für sauberes Resource-Management
import io             # Input/Output-Operationen
import warnings
from transformers import logging as transformers_logging

# ====== Audio-Processing-Imports ======
import pyaudio        # Audio I/O, Mikrofon-Zugriff
import numpy as np    # Numerische Operationen, Audio-Array-Verarbeitung
import webrtcvad      # Voice Activity Detection (VAD)
from scipy import signal  # Signalverarbeitung, Audio-Resampling

# ====== ML-Model-Imports ======
from transformers import pipeline  # HuggingFace Transformers Pipeline

# ====== Warnung-Unterdrückung ======
# Unterdrücke verschiedene Warnungen, die für den Betrieb nicht relevant sind
warnings.filterwarnings("ignore", category=UserWarning)     # Allgemeine Warnungen
warnings.filterwarnings("ignore", category=FutureWarning)   # Zukunfts-Warnungen
# Spezifische Whisper-Warnungen unterdrücken
warnings.filterwarnings("ignore", message=".*transcription using a multilingual Whisper.*")
warnings.filterwarnings("ignore", message=".*past_key_values.*")
warnings.filterwarnings('ignore', message='.*past_key_values.*', category=FutureWarning)
transformers_logging.set_verbosity_error()  # Nur Fehler anzeigen, keine Warnungen
# HuggingFace Transformers Warnungen deaktivieren
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'

# ====== Audio-System-Konfiguration ======
# Unterdrücke ALSA/JACK-Fehlermeldungen für sauberere Ausgabe
os.environ['ALSA_CARD'] = 'Generic'                    # Generische ALSA-Karte
os.environ['JACK_NO_AUDIO_RESERVATION'] = '1'          # Keine Audio-Reservierung
os.environ['JACK_NO_START_SERVER'] = '1'               # Kein Auto-Start des JACK-Servers

# ====== Stderr-Umleitung ======
@contextmanager
def stderr_redirected(to=os.devnull):
    """
    Kontextmanager zum temporären Umleiten von stderr.
    
    Diese Funktion ist wichtig für die saubere Unterdrückung von Low-Level-
    Fehlermeldungen, besonders von Audio-Bibliotheken wie ALSA/JACK.
    
    Args:
        to (str): Zieldatei für die Umleitung (Standard: os.devnull)
        
    Yields:
        None: Wird als Kontextmanager verwendet
        
    Beispiel:
        with stderr_redirected():
            # Code der möglicherweise stderr-Ausgaben produziert
            pass
    """
    fd = sys.stderr.fileno()
    stderr_copy = os.dup(fd)
    
    try:
        with open(to, 'wb') as devnull:
            os.dup2(devnull.fileno(), fd)
            yield
    finally:
        os.dup2(stderr_copy, fd)
        os.close(stderr_copy)

# ====== Konfigurationskonstanten ======
# Debug-Konfiguration
EXTENDED_LOGGING = False  # Aktiviert zusätzliche Debug-Ausgaben wenn True

# Modell-Konfiguration
MODEL_ID = "primeline/whisper-large-v3-turbo-german"  # Optimiertes deutsches Whisper-Modell
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # GPU wenn verfügbar, sonst CPU
COMPUTE_TYPE = "float16" if DEVICE == "cuda" else "float32"  # float16 für GPU-Optimierung

# Audio-Aufnahme-Konfiguration
DEVICE_RATE = 44100   # Hardware-Samplerate (Hz) - Standard für viele USB-Mikrofone
MODEL_RATE = 16000    # Whisper-Modell erwartet 16kHz
CHANNELS = 1          # Mono-Audio für bessere Verarbeitung
FORMAT = pyaudio.paFloat32  # 32-bit Float für bessere Audioqualität
CHUNK_SIZE = 480      # 30ms bei 16kHz - optimal für VAD

# Voice Activity Detection (VAD) Parameter
VAD_FRAME_MS = 30     # Frame-Länge in Millisekunden für VAD
MIN_SPEECH_DURATION = 0.8  # Minimale Sprachdauer in Sekunden
POST_SPEECH_SILENCE = 0.8  # Nachbearbeitungszeit in Sekunden

# ====== Logging-Konfiguration ======
# Konfiguriere strukturiertes Logging für bessere Nachvollziehbarkeit
logging.basicConfig(
    level=logging.INFO,                                # Log-Level: INFO für wichtige Events
    format='%(asctime)s - %(levelname)s - %(message)s',  # Zeitstempel - Level - Nachricht
    datefmt='%H:%M:%S',                               # Kurzes Zeitformat für Übersichtlichkeit
    handlers=[
        logging.StreamHandler(sys.stdout)              # Logging auf stdout statt stderr
    ]
)

class AudioTranscriber:
    """
    Hauptklasse für die Echtzeit-Spracherkennung.
    
    Diese Klasse kombiniert mehrere Komponenten für eine robuste Echtzeit-Spracherkennung:
    1. Audio-Aufnahme über PyAudio
    2. Voice Activity Detection (VAD) mit WebRTC
    3. Audio-Resampling von 44.1kHz auf 16kHz
    4. Spracherkennung mit HuggingFace Whisper
    
    Die Klasse ist optimiert für:
    - Geringe Latenz durch effiziente Pufferverarbeitung
    - Robuste Spracherkennung durch VAD
    - Speichereffiziente Verarbeitung durch Streaming
    - Deutsche Sprache durch spezialisiertes Modell
    """
    
    def __init__(self):
        """
        Initialisiert die Echtzeit-Spracherkennung.
        
        Dieser Konstruktor richtet alle notwendigen Komponenten ein:
        1. Status-Variablen für die Aufnahme
        2. Audio-Puffer und Transkriptionspuffer
        3. VAD-Parameter und Konfiguration
        4. Whisper-Modell und Pipeline
        5. Audio-Interface und Stream
        
        Raises:
            Exception: Bei Initialisierungsfehlern (z.B. kein Mikrofon gefunden)
        """
        try:
            # Komponenten initialisieren
            print("Initialisiere Spracherkennung...")
            
            # Status-Variablen für Aufnahme und Verarbeitung
            self.is_recording = False          # Aktiver Aufnahmestatus
            self.audio_buffer = []             # Puffer für Audio-Chunks
            self.transcription_buffer = ""     # Puffer für erkannten Text
            self.silence_frames = 0            # Zähler für Stille-Frames
            self.last_text = ""                # Letzter erkannter Text (Deduplizierung)
            
            # VAD-Parameter für Sprach-/Pause-Erkennung
            self.min_recording_length = int(DEVICE_RATE * MIN_SPEECH_DURATION)  # Min. Samples für Sprache
            self.post_speech_silence = int(DEVICE_RATE * POST_SPEECH_SILENCE)   # Samples für Nachbearbeitung
            self.vad_frame_duration = VAD_FRAME_MS                              # VAD Frame-Länge
            self.vad_frame_size = int(MODEL_RATE * 0.03)                       # Samples pro VAD-Frame
            
            # Audio-Processing-Komponenten
            self.resampler = signal.resample_poly  # Effizientes Resampling
            self.vad = webrtcvad.Vad(2)           # VAD mit mittlerer Aggressivität
            logging.info("Audio-Processing eingerichtet")
            
            # Whisper-Modell laden und konfigurieren
            logging.info("Lade Whisper-Modell...")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")  # Unterdrücke Modell-Warnungen
                self.pipe = pipeline(
                    task="automatic-speech-recognition",
                    model=MODEL_ID,
                    device=DEVICE,
                    torch_dtype=torch.float16 if COMPUTE_TYPE == "float16" else torch.float32,
                    model_kwargs={"forced_decoder_ids": [[1, None], [2, 50360]]}  # Erzwinge Deutsch
                )
            logging.info("Whisper-Modell geladen")
            
            # Audio-Interface initialisieren
            self.setup_audio()
            
        except Exception as e:
            logging.error(f"Initialisierungsfehler: {e}")
            raise
            
    def setup_audio(self):
        """Initialisiert das Audio-Interface und konfiguriert den Audio-Stream.
        
        Diese Methode richtet die Audio-Aufnahme mit PyAudio ein:
        1. Initialisiert PyAudio mit stderr Unterdrückung für sauberere Ausgabe
        2. Wählt das USB-Mikrofon mit Index 11 als Eingabegerät
        3. Konfiguriert einen Audio-Stream mit folgenden Parametern:
           - FORMAT: 32-bit Float für hohe Audioqualität
           - CHANNELS: Mono für effiziente Verarbeitung
           - RATE: 44.1kHz Hardware-Samplerate
           - CHUNK_SIZE: 480 Samples (30ms) für optimale Latenz
        4. Setzt audio_callback als Stream-Callback für Echtzeit-Verarbeitung
        
        Die Methode verwendet stderr_redirected() um PyAudio Warnungen zu unterdrücken,
        die bei normaler Nutzung nicht relevant sind.
        
        Raises:
            PyAudio.Error: Bei Problemen mit der Audio-Initialisierung
            IOError: Wenn das spezifische USB-Mikrofon nicht gefunden wird
        """
        try:
            # PyAudio initialisieren
            with stderr_redirected():
                self.audio = pyaudio.PyAudio()
            
            # Spezifisches USB-Mikrofon (Index 11) verwenden
            device_index = 11
            
            try:
                device_info = self.audio.get_device_info_by_index(device_index)
                print(f"Verwende Eingabegerät: {device_info.get('name')}")
            except Exception as e:
                logging.error(f"Gerät mit Index {device_index} nicht gefunden: {e}")
                raise
            
            # Audio-Stream konfigurieren
            with stderr_redirected():
                self.stream = self.audio.open(
                    input_device_index=device_index,
                    format=FORMAT,
                    channels=CHANNELS,
                    rate=DEVICE_RATE,
                    input=True,
                    frames_per_buffer=CHUNK_SIZE,
                    stream_callback=self.audio_callback
                )
            
            print(f"Audio-Stream initialisiert: {self.stream.get_input_latency()*1000:.2f}ms Latenz")
            print("\nEchtzeit-Spracherkennung gestartet. Sprechen Sie etwas...")
            
        except Exception as e:
            logging.error(f"Fehler bei Audio-Setup: {e}")
            raise
            
    def is_speech(self, audio_chunk):
        """Analysiert einen Audio-Chunk auf Sprachaktivität mittels WebRTC VAD.
        
        Diese Methode verarbeitet die rohen Audio-Daten in mehreren Schritten:
        1. Berechnet den durchschnittlichen Audiopegel zur Grundanalyse
        2. Resampled das Audio von 44.1kHz auf 16kHz für VAD-Kompatibilität
        3. Konvertiert Float32 zu Int16 für VAD-Verarbeitung
        4. Teilt Audio in 30ms Frames für optimale VAD-Erkennung
        5. Prüft jeden Frame auf Sprachaktivität
        
        Die WebRTC VAD ist auf Aggressivitätslevel 2 eingestellt, was einen guten
        Kompromiss zwischen Empfindlichkeit und Robustheit bietet.
        
        Args:
            audio_chunk (numpy.ndarray): Float32 Audio-Daten vom PyAudio Stream
            
        Returns:
            bool: True wenn Sprache erkannt wurde, False wenn nicht oder bei Fehlern
            
        Technische Details:
        - Frame-Größe: 30ms (optimal für WebRTC VAD)
        - Resampling: 44.1kHz -> 16kHz (VAD Anforderung)
        - Amplitude: Float32 -> Int16 (Skalierung: * 32767)
        """
        try:
            # Pegel prüfen
            level = np.abs(audio_chunk).mean()
            
            # Resampling für VAD
            resampled = self.resampler(audio_chunk, DEVICE_RATE, MODEL_RATE)
            
            # VAD
            frame = (resampled * 32767).astype(np.int16)
            frames = [frame[i:i + self.vad_frame_size] for i in range(0, len(frame) - self.vad_frame_size + 1, self.vad_frame_size)]
            
            if frames:
                is_speech = any(self.vad.is_speech(f.tobytes(), MODEL_RATE) for f in frames)
            else:
                is_speech = False
                
            if EXTENDED_LOGGING:
                logging.debug(f"Sprachaktivität erkannt (VAD: {is_speech}, Level: {level:.4f})")
                
            return is_speech
            
        except Exception as e:
            logging.error(f"Fehler bei Spracherkennung: {e}")
            return False
            
    def audio_callback(self, in_data, frame_count, time_info, status):
        """PyAudio Callback-Funktion für die Echtzeit-Audioverarbeitung.
        
        Diese Callback-Methode implementiert die Kernlogik der Spracherkennung:
        1. Konvertiert die rohen Audiodaten in ein numpy-Array
        2. Prüft bei aktiver Aufnahme auf Sprachende (Stille)
        3. Prüft bei inaktiver Aufnahme auf Sprachbeginn
        4. Puffert Audiodaten während der Aufnahme
        
        Die Methode verwendet einen adaptiven Mechanismus zur Sprach/Pause-Erkennung:
        - Minimum Aufnahmelänge: 0.8 Sekunden (self.min_recording_length)
        - Stille nach Sprache: 0.8 Sekunden (self.post_speech_silence)
        - Chunk-basierte Verarbeitung für geringe Latenz
        
        Args:
            in_data (bytes): Rohe Audiodaten vom Stream
            frame_count (int): Anzahl der Frames im Chunk
            time_info (dict): Stream-Timing Information (unused)
            status (int): Stream-Status Flags (unused)
            
        Returns:
            tuple: (in_data, pyaudio.paContinue) für kontinuierliche Aufnahme
        """
        try:
            # Audio-Daten in numpy-Array umwandeln
            audio_chunk = np.frombuffer(in_data, dtype=np.float32)
            
            # Spracherkennung
            if self.is_recording:
                self.audio_buffer.append(audio_chunk)
                
                # Prüfen ob Sprachende
                if not self.is_speech(audio_chunk):
                    self.silence_frames += 1
                    buffer_duration = len(self.audio_buffer) * CHUNK_SIZE / DEVICE_RATE
                    
                    # Nach Stille Audio verarbeiten
                    if self.silence_frames * CHUNK_SIZE > self.post_speech_silence:  # 0.8s Stille
                        if len(self.audio_buffer) * CHUNK_SIZE > self.min_recording_length:  # 0.8s Minimum
                            logging.info(f"Verarbeite Audio-Buffer ({buffer_duration:.2f}s)")
                            self.process_audio()
                        else:
                            logging.debug(f"Buffer zu kurz ({buffer_duration:.2f}s)")
                        self.is_recording = False
                        self.audio_buffer = []
                        self.silence_frames = 0
                        logging.info("Sprachpause erkannt")
                else:
                    self.silence_frames = 0
            else:
                # Prüfen ob Sprachbeginn
                if self.is_speech(audio_chunk):
                    self.is_recording = True
                    self.audio_buffer = [audio_chunk]
                    logging.info("Sprachaktivität erkannt")
                    
            return (in_data, pyaudio.paContinue)
            
        except Exception as e:
            logging.error(f"Fehler im Audio-Callback: {str(e)}")
            logging.error(traceback.format_exc())
            return (in_data, pyaudio.paContinue)
            
    def process_audio(self):
        """Verarbeitet den aufgenommenen Audio-Buffer zur Transkription.
        
        Diese Methode verarbeitet die gepufferten Audiodaten und führt die
        Whisper-Transkription durch. Der Prozess umfasst:
        1. Zusammenführen der gepufferten Audio-Chunks
        2. Berechnung des Audio-Pegels für Rauschunterdrückung
        3. Resampling auf die Whisper-Modell-Rate (16kHz)
        4. Durchführung der Whisper-Transkription
        5. Ausgabe des erkannten Texts
        
        Technische Details:
        - Rausch-Schwellwert: 0.002 (mittlerer Absolutpegel)
        - Resampling: 44.1kHz -> 16kHz
        - Whisper-Pipeline: Optimiert für Deutsche Sprache
        - Logging: Detaillierte Prozess-Informationen
        
        Die Methode ignoriert zu leise Audiodaten, um Fehlerkennungen
        durch Hintergrundrauschen zu vermeiden.
        """
        if not self.audio_buffer:
            return
            
        try:
            # Audio-Daten zusammenführen und in numpy-Array konvertieren
            audio_data = b''.join(self.audio_buffer)
            audio_np = np.frombuffer(audio_data, dtype=np.float32)
            
            # Berechne Audio-Level
            audio_level = np.abs(audio_np).mean()
            logging.info(f"Verarbeite Audio: {len(audio_np)/DEVICE_RATE:.2f}s, Level: {audio_level:.4f}")
            
            # Ignoriere zu leise Audio (wahrscheinlich Rauschen)
            if audio_level < 0.002:  # Erhöhter Schwellenwert für Rauschunterdrückung
                logging.info("Audio zu leise - überspringe Transkription")
                return ""
                
            # Resampling von 44.1kHz auf 16kHz
            resampled_audio = self.resampler(
                audio_np, 
                up=MODEL_RATE,
                down=DEVICE_RATE
            )
            
            # Transkription mit Whisper
            logging.info("Starte Whisper-Transkription...")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = self.pipe(
                    inputs=resampled_audio,
                    return_timestamps=True
                )
            
            # Text verarbeiten und anzeigen
            text = result["text"].strip()
            if text:  # Nur ausgeben wenn Text erkannt wurde
                print(f"\rErkannt: {text}", flush=True)
                logging.info(f"Erkannt: {text}")
            else:
                logging.warning("Keine Transkription erhalten")
            
            # Buffer zurücksetzen
            self.audio_buffer = []
            
        except Exception as e:
            logging.error(f"Fehler bei Audio-Verarbeitung: {e}")
            import traceback
            logging.error(traceback.format_exc())
            
    def run(self):
        """Startet die Echtzeit-Spracherkennung.
        
        Diese Methode initiiert die kontinuierliche Audio-Aufnahme und
        Verarbeitung. Sie:
        1. Startet den PyAudio Stream
        2. Hält das Programm am Laufen
        3. Fängt Keyboard-Interrupts (Ctrl+C) ab
        4. Führt sauberes Cleanup beim Beenden durch
        
        Die Methode blockiert den Hauptthread und läuft bis zum
        Programmabbruch durch den Benutzer (Ctrl+C) oder bis ein
        Fehler auftritt.
        
        Raises:
            Exception: Bei Fehlern während der Audio-Verarbeitung
        """
        try:
            self.stream.start_stream()
            print("\nDrücken Sie Ctrl+C zum Beenden...")
            
            while self.stream.is_active():
                time.sleep(0.1)
                
        except KeyboardInterrupt:
            print("\nBeende Programm...")
        finally:
            self.cleanup()
            
    def cleanup(self):
        """Führt sauberes Herunterfahren der Audio-Ressourcen durch.
        
        Diese Methode stellt sicher, dass alle Audio-Ressourcen
        ordnungsgemäß freigegeben werden:
        1. Stoppt den aktiven Audio-Stream
        2. Schließt den Stream
        3. Beendet die PyAudio-Instanz
        4. Gibt GPU-Ressourcen frei (falls verwendet)
        
        Die Methode wird automatisch beim Beenden des Programms
        aufgerufen und kann auch manuell ausgeführt werden.
        Sie ist idempotent, d.h. mehrfaches Aufrufen ist sicher.
        """
        try:
            if self.stream is not None:
                if self.stream.is_active():
                    self.stream.stop_stream()
                self.stream.close()
                
            if self.audio is not None:
                self.audio.terminate()
                
            # Speicher freigeben
            self.stream = None
            self.audio = None
            
        except Exception as e:
            logging.error(f"Fehler beim Cleanup: {e}")
            
if __name__ == "__main__":
    try:
        transcriber = AudioTranscriber()
        transcriber.run()
    finally:
        # Cleanup durchführen
        if 'transcriber' in locals():
            transcriber.cleanup()
