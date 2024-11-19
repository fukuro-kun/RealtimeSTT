"""
Echtzeit-Spracherkennung mit HuggingFace Whisper
===============================================

Optimierte Version mit:
- Mehrsprachunterstützung (Deutsch/Englisch)
- Effiziente Audio-Verarbeitung
- Robuste VAD-Integration
"""

import os
import sys
import time
import torch
import logging
import warnings
import numpy as np
import pyaudio
import webrtcvad
from scipy import signal as scipy_signal
import signal as system_signal
from transformers import pipeline
from dataclasses import dataclass
from contextlib import contextmanager
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.text import Text
from typing import Optional
from pynput import keyboard
import threading
import signal
from transformers import logging as transformers_logging

# Warnung-Unterdrückung
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*transcription using a multilingual Whisper.*")
transformers_logging.set_verbosity_error()
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'
os.environ['ALSA_CARD'] = 'Generic'
os.environ['JACK_NO_AUDIO_RESERVATION'] = '1'
os.environ['JACK_NO_START_SERVER'] = '1'

# Debug-Konfiguration
DEBUG = True
EXTENDED_LOGGING = True

# Audio-Konfiguration
DEVICE_RATE = 44100
MODEL_RATE = 16000
CHANNELS = 1
CHUNK_SIZE = 1323      # 1323 samples at 44.1kHz = 480 samples at 16kHz (30ms)
FORMAT = pyaudio.paFloat32

# VAD Parameter
VAD_FRAME_MS = 30
MIN_SPEECH_DURATION = 0.6    # Erhöht auf 800ms für noch längere minimale Sprachdauer
POST_SPEECH_SILENCE = 1.0    # Erhöht auf 1s für längere Pausen
VAD_MODE = 2                 # Erhöht auf 3 für aggressivste Filterung von Nicht-Sprache

# Modell-Konfiguration
MODELS = {
    'de': "primeline/whisper-large-v3-turbo-german",
    'en': "openai/whisper-small"
}

@contextmanager
def stderr_redirected(to=os.devnull):
    """Kontextmanager zum temporären Umleiten von stderr."""
    fd = sys.stderr.fileno()
    stderr_copy = os.dup(fd)
    try:
        with open(to, 'wb') as devnull:
            os.dup2(devnull.fileno(), fd)
            yield
    finally:
        os.dup2(stderr_copy, fd)
        os.close(stderr_copy)

@dataclass
class AudioDevice:
    """Audio-Gerät Informationen"""
    index: int
    name: str
    max_input_channels: int
    default_sample_rate: float

class AudioLevelMeter:
    def __init__(self, width=40):
        self.width = width
        self.last_level = 0
        self.peak_level = 0
        self.decay = 0.1

    def update(self, level):
        """Aktualisiert den Level-Meter"""
        self.peak_level = max(self.peak_level * (1 - self.decay), level)
        self.last_level = level
        
        # dB-Wert berechnen (mit -60dB minimum)
        db = 20 * np.log10(max(level, 1e-6))
        db = max(db, -60)
        
        return f"{db:.1f}"

class TerminalOutput:
    """Klasse für formatierte Terminal-Ausgabe"""
    def __init__(self):
        self.console = Console()
        
        self.live = Live(
            console=self.console,
            refresh_per_second=4,
            auto_refresh=False,
            screen=True
        )
        self.live.start()
        
        self.text = ""
        self.meter = "-60.0"  # Initialisiere mit Stille
        self.language = "de"
        self.current_text = ""
        self._update_display()

    def update(self, text=None, language=None, meter=None):
        if text is not None:
            if text.strip() in ['.', '!', '?', '...']:
                return
            text = text.strip().lstrip('.!?')
            if not any(c.isalnum() for c in text):
                return
            if self.current_text and not self.current_text.endswith(('!', '.', '?', '...')):
                self.current_text += " "
            self.current_text += text
            self.text = Text(self.current_text, style="green")
            
        if language is not None:
            self.language = language
        if meter is not None:
            self.meter = meter
        self._update_display()

    def _update_display(self):
        # Erstelle einen Container für den Haupttext
        content = Text()
        
        # Haupttext hinzufügen
        if self.text:
            content.append(self.text)
        else:
            content.append(Text("Warte auf Spracheingabe...", style="italic"))
        
        # Erstelle den Titel mit allen Informationen
        title = Text()
        
        # Spracherkennung und Sprachcode
        title.append("Spracherkennung (", style="green")
        title.append(self.language.upper(), style="white")
        title.append(")", style="green")
        
        # Steuerelemente
        title.append(" • ")
        title.append("Tab", style="white")
        title.append(" ")
        title.append("Sprache wechseln", style="cyan")
        title.append(" • ")
        title.append("Esc", style="white")
        title.append("/")
        title.append("Strg+C", style="white")
        title.append(" ")
        title.append("Beenden", style="cyan")
        
        # dB-Wert - immer anzeigen
        title.append(" • ")
        # Farbe basierend auf Lautstärke
        db_value = float(self.meter)
        if db_value > -12:
            style = "red"
        elif db_value > -24:
            style = "yellow"
        else:
            style = "blue"
        title.append(f"{self.meter} dB", style=style)
        
        panel = Panel(
            content,
            title=title,
            border_style="green",
            padding=(1, 2)
        )
        
        # Update das Display
        self.live.update(panel)
        self.live.refresh()

    def cleanup(self):
        self.live.stop()

class EnhancedAudioRecorder:
    """Hauptklasse für Audio-Aufnahme und Spracherkennung"""
    def __init__(self):
        self.pa = pyaudio.PyAudio()
        self.device = self._find_audio_device()
        self.current_language = 'de'
        self.output_handler = None
        self.is_recording = False
        self.audio_buffer = []
        self.silence_frames = 0
        self.post_speech_silence_duration = POST_SPEECH_SILENCE  # Standardwert
        self._running = True
        self._audio_thread = None
        
        # VAD Setup
        self.vad = webrtcvad.Vad(VAD_MODE)
        self.min_speech_duration = int(DEVICE_RATE * MIN_SPEECH_DURATION)
        self.post_speech_silence = int(self.post_speech_silence_duration * DEVICE_RATE / CHUNK_SIZE)
        
        # Modelle laden
        self._load_models()
        
        # Level-Meter initialisieren
        self.level_meter = AudioLevelMeter()
        
        if DEBUG:
            logging.debug(f"VAD Parameter initialisiert:")
            logging.debug(f"- Frame MS: {VAD_FRAME_MS}")
            logging.debug(f"- Min Speech Duration: {MIN_SPEECH_DURATION}s")
            logging.debug(f"- Post Speech Silence: {self.post_speech_silence_duration}s")
            logging.debug(f"- VAD Mode: {VAD_MODE}")
        
    def _load_models(self):
        """Lädt beide Sprachmodelle"""
        print("Lade Sprachmodelle...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        compute_type = "float16" if device == "cuda" else "float32"
        
        self.models = {}
        for lang, model_id in MODELS.items():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.models[lang] = pipeline(
                    "automatic-speech-recognition",
                    model=model_id,
                    device=device,
                    torch_dtype=torch.float16 if compute_type == "float16" else torch.float32,
                    model_kwargs={"forced_decoder_ids": [[1, None], [2, 50360]] if lang == 'de' else None}
                )
        print("Modelle geladen!")

    def _find_audio_device(self) -> AudioDevice:
        """Findet das beste verfügbare Audio-Eingabegerät"""
        print("\nVerfügbare Audio-Geräte:")
        usb_device = None
        
        for i in range(self.pa.get_device_count()):
            dev_info = self.pa.get_device_info_by_index(i)
            if dev_info['maxInputChannels'] > 0:
                print(f"Gefunden: [{i}] {dev_info['name']}")
                if 'USB' in dev_info['name']:
                    usb_device = AudioDevice(
                        index=i,
                        name=dev_info['name'],
                        max_input_channels=dev_info['maxInputChannels'],
                        default_sample_rate=dev_info['defaultSampleRate']
                    )
        
        if not usb_device:
            # Fallback auf Standard-Eingabegerät
            info = self.pa.get_default_input_device_info()
            usb_device = AudioDevice(
                index=info['index'],
                name=info['name'],
                max_input_channels=info['maxInputChannels'],
                default_sample_rate=info['defaultSampleRate']
            )
        
        print(f"\nVerwende Gerät: [{usb_device.index}] {usb_device.name}\n")
        return usb_device

    def resample_audio(self, audio_data, src_rate, target_rate):
        """Resampled Audio auf die Ziel-Samplerate"""
        if src_rate == target_rate:
            return audio_data
        
        # Berechne das exakte Verhältnis
        ratio = target_rate / src_rate
        out_samples = int(len(audio_data) * ratio)
        
        try:
            resampled = scipy_signal.resample(audio_data, out_samples)
            if DEBUG:
                logging.debug(f"Resampling: {len(audio_data)} -> {len(resampled)} samples ({src_rate}Hz -> {target_rate}Hz)")
            return resampled
        except Exception as e:
            logging.error(f"Fehler beim Resampling: {e}")
            return audio_data

    def is_speech(self, audio_data):
        """Erkennt ob der Audio-Frame Sprache enthält"""
        try:
            # Konvertiere zu NumPy Array wenn nötig
            if isinstance(audio_data, bytes):
                audio_data = np.frombuffer(audio_data, dtype=np.float32)
            
            # Prüfe Audio-Level
            audio_level = np.max(np.abs(audio_data))
            if DEBUG:
                logging.debug(f"Audio Level: {audio_level:.6f}")
            
            # Strengerer Schwellwert für Audio-Level
            if audio_level < 0.05:  # Basierend auf silero_sensitivity
                return False
            
            # Resample auf 16kHz für VAD
            vad_rate = 16000  # WebRTC VAD erwartet 16kHz
            resampled = self.resample_audio(audio_data, DEVICE_RATE, vad_rate)
            
            # Sanfte Normalisierung für VAD
            max_level = np.max(np.abs(resampled))
            if max_level > 0.05:  # Nur normalisieren wenn Signal stark genug
                resampled = resampled / max_level * 0.95  # Leichte Dämpfung
            
            # Konvertiere zu int16 für VAD
            resampled_int16 = (resampled * 32767).astype(np.int16)
            
            # Teile in 30ms VAD-Frames (480 samples bei 16kHz)
            frame_size = int(vad_rate * 0.03)  # 30ms bei 16kHz = 480 samples
            frame_bytes = frame_size * 2  # 2 bytes pro sample für int16
            
            frames = []
            for i in range(0, len(resampled_int16) - frame_size + 1, frame_size):
                frame = resampled_int16[i:i + frame_size].tobytes()
                if len(frame) == frame_bytes:  # Nur vollständige Frames verwenden
                    frames.append(frame)
            
            # Prüfe VAD für jeden Frame
            speech_frames = 0
            total_frames = len(frames)
            
            for frame_idx, frame in enumerate(frames):
                try:
                    is_speech_frame = self.vad.is_speech(frame, vad_rate)
                    if DEBUG and is_speech_frame:
                        logging.debug(f"🎤 Frame {frame_idx+1}/{total_frames}: Sprache erkannt")
                    if is_speech_frame:
                        speech_frames += 1
                except Exception as e:
                    logging.error(f"VAD Fehler bei Frame {frame_idx}: {e}")
            
            speech_ratio = speech_frames / total_frames if total_frames > 0 else 0
            if DEBUG:
                logging.debug(f"Speech Ratio: {speech_ratio:.2f} ({speech_frames}/{total_frames} frames)")
            
            # Strengerer Schwellwert für Speech Ratio
            is_speech = speech_ratio > 0.5  # Mindestens 50% der Frames müssen Sprache sein
            if is_speech and DEBUG:
                logging.info("🎙️ Sprache erkannt!")
            
            return is_speech
            
        except Exception as e:
            logging.error(f"Fehler in is_speech: {e}")
            return False

    def format_output_text(self, text):
        """Formatiert den Text für die Ausgabe, ohne die VAD-Logik zu beeinflussen"""
        if not text:
            return text
            
        # Entferne überflüssige Leerzeichen
        text = ' '.join(text.split())
        
        # Füge Leerzeichen nach Satzzeichen ein (nur für die Anzeige)
        formatted = text
        for punct in ['.', '!', '?', '。']:
            # Berücksichtige Fälle wie "..." oder ".."
            if f"{punct}.." not in formatted and f"{punct}." not in formatted:
                formatted = formatted.replace(f"{punct}", f"{punct} ")
        
        return formatted.strip()

    def process_text(self, text):
        """Verarbeitet den erkannten Text und passt die VAD-Parameter an."""
        if not text:
            return text
            
        text = text.strip()
        
        # Pausenlänge basierend auf Satzende anpassen
        if text.endswith("..."):
            self.post_speech_silence_duration = 2.0    # Mitten im Satz
        elif text and text[-1] in ['.', '!', '?', '。']:
            self.post_speech_silence_duration = 0.45   # Satzende
        else:
            self.post_speech_silence_duration = 0.7    # Unbekannt
            
        # Aktualisiere die Frame-Anzahl für die Stille
        self.post_speech_silence = int(self.post_speech_silence_duration * DEVICE_RATE / CHUNK_SIZE)
        
        if DEBUG:
            logging.debug(f"Neue Pause eingestellt: {self.post_speech_silence_duration}s ({self.post_speech_silence} frames)")
            
        # Text bereinigen und formatieren
        if text.endswith("..."):
            text = text[:-3].rstrip()
        
        # Leerzeichen nach Satzzeichen einfügen
        for punct in ['.', '!', '?', '。']:
            text = text.replace(f"{punct}", f"{punct} ")
        text = text.strip()
            
        return text

    def start_stream(self):
        """Startet den Audio-Stream"""
        if DEBUG:
            logging.debug("Starte Audio-Stream...")
        
        self._running = True
        self.stream = self.pa.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=DEVICE_RATE,
            input=True,
            input_device_index=self.device.index,
            frames_per_buffer=CHUNK_SIZE,
            stream_callback=self.audio_callback
        )
        self.stream.start_stream()

    def stop_stream(self):
        """Stoppt den Audio-Stream"""
        if DEBUG:
            logging.debug("Stoppe Audio-Stream...")
        
        self._running = False
        if hasattr(self, 'stream') and self.stream:
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None
            
        if self._audio_thread and self._audio_thread.is_alive():
            self._audio_thread.join(timeout=1.0)
            if DEBUG:
                logging.debug("Audio-Thread beendet")

    def audio_callback(self, in_data, frame_count, time_info, status):
        """Callback für Audio-Verarbeitung"""
        if not self._running:
            if DEBUG:
                logging.debug("Audio-Callback: Stream wird beendet")
            return (None, pyaudio.paComplete)
            
        try:
            if DEBUG:
                logging.debug(f"Audio Callback: {frame_count} frames")
            
            # Audio-Level berechnen und anzeigen
            audio_data = np.frombuffer(in_data, dtype=np.float32)
            audio_level = np.abs(audio_data).mean()
            
            if self.output_handler:
                meter = self.level_meter.update(audio_level)
                self.output_handler.update(meter=meter)
            
            if DEBUG:
                logging.debug(f"Audio Level: {audio_level:.6f}")
            
            # Sprache erkennen
            is_speech = self.is_speech(in_data)
            if DEBUG:
                logging.debug(f"Speech Detected: {is_speech}")
            
            if self.is_recording:
                if DEBUG:
                    logging.debug("Nehme auf...")
                self.audio_buffer.append(in_data)
                if not is_speech:
                    self.silence_frames += 1
                    if DEBUG:
                        logging.debug(f"Silence Frames: {self.silence_frames}/{self.post_speech_silence}")
                    
                    if self.silence_frames >= self.post_speech_silence:
                        if DEBUG:
                            logging.debug("⚡ Starte Audio-Verarbeitung")
                        self._audio_thread = threading.Thread(target=self.process_audio)
                        self._audio_thread.start()
                        self.is_recording = False
                        self.audio_buffer = []
                        self.silence_frames = 0
                else:
                    self.silence_frames = 0
            else:
                if is_speech:
                    if DEBUG:
                        logging.debug("🎙️ Starte neue Aufnahme")
                    self.is_recording = True
                    self.audio_buffer = [in_data]
                    self.silence_frames = 0
                    
        except Exception as e:
            logging.error(f"Fehler im Audio-Callback: {e}")
            import traceback
            logging.error(traceback.format_exc())
        
        return (in_data, pyaudio.paContinue if self._running else pyaudio.paComplete)

    def process_audio(self):
        """Verarbeitet aufgenommenes Audio"""
        if not self.audio_buffer:
            if DEBUG:
                logging.debug("🚫 Leerer Audio-Buffer - überspringe Verarbeitung")
            return
            
        try:
            # Audio-Daten zusammenführen und konvertieren
            audio_data = b''.join(self.audio_buffer)
            audio_np = np.frombuffer(audio_data, dtype=np.float32)
            
            # Audio-Level prüfen
            audio_level = np.abs(audio_np).mean()
            if DEBUG:
                logging.debug(f"🔊 Aufnahme-Level: {audio_level:.6f}")
            
            if audio_level < 0.0005:  
                logging.info("🔇 Audio zu leise - überspringe Transkription")
                return
            
            # Resampling für Modell
            resampled = self.resample_audio(audio_np, DEVICE_RATE, MODEL_RATE)
            if DEBUG:
                logging.debug(f"⚙️ Resampled Shape: {resampled.shape}")
                logging.debug(f"⚙️ Resampled Mean: {np.mean(resampled):.6f}")
                logging.debug(f"⚙️ Resampled Max: {np.max(np.abs(resampled)):.6f}")
            
            # Normalisiere Audio
            max_val = np.max(np.abs(resampled))
            if max_val > 0:
                resampled = resampled / max_val * 0.95
                if DEBUG:
                    logging.debug(f"📊 Audio normalisiert, neuer Max: {np.max(np.abs(resampled)):.6f}")
            
            # Transkription mit angepassten Parametern
            if DEBUG:
                logging.debug(f"🎯 Starte Transkription ({self.current_language})")
                
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = self.models[self.current_language](
                    inputs=resampled,
                    return_timestamps=False,
                    chunk_length_s=30,
                    batch_size=1
                )
            
            if DEBUG:
                logging.debug(f"📝 Rohe Modell-Ausgabe: {result}")
            
            # Text verarbeiten
            if result and "text" in result and result["text"]:
                # Erst VAD-Parameter anpassen mit Original-Text
                text = result["text"].strip()
                self.process_text(text)
                
                # Dann Text für Ausgabe formatieren
                formatted_text = self.format_output_text(text)
                
                if DEBUG:
                    logging.debug(f"✨ Erkannter Text: '{formatted_text}'")
                    
                if self.output_handler:
                    if DEBUG:
                        logging.debug("📤 Sende Text an Terminal-Output")
                    self.output_handler.update(text=formatted_text)
                    if DEBUG:
                        logging.debug("✅ Terminal-Output aktualisiert")
                else:
                    logging.warning("❌ Kein Output-Handler verfügbar!")
            else:
                logging.warning("❌ Keine Transkription erhalten")
                if DEBUG:
                    logging.debug(f"❌ Leere Modell-Ausgabe: {result}")
            
        except Exception as e:
            logging.error(f"❌ Fehler in process_audio: {e}")
            import traceback
            logging.error(traceback.format_exc())

def setup_logging():
    """Konfiguriert das Logging-System"""
    log_file = 'realtimestt_debug.log'
    
    # Lösche altes Log-File
    if os.path.exists(log_file):
        os.remove(log_file)
    
    logging.basicConfig(
        level=logging.DEBUG if DEBUG else logging.INFO,
        format='%(asctime)s.%(msecs)03d - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

if __name__ == "__main__":
    setup_logging()
    
    logging.info("=== Neue Aufnahme-Session gestartet ===")
    
    # Globale Variablen
    running = True
    recorder = None
    terminal_output = None
    
    def cleanup():
        """Säubert alle Ressourcen"""
        global running, recorder, terminal_output
        
        try:
            logging.info("Cleanup gestartet...")
            running = False
            
            if recorder:
                logging.debug("Stoppe Audio-Stream...")
                recorder.stop_stream()
                logging.debug("Beende PyAudio...")
                recorder.pa.terminate()
                
            if terminal_output:
                logging.debug("Beende Terminal-Ausgabe...")
                terminal_output.cleanup()
                
            print("\nProgramm wird beendet...")
            
        except Exception as e:
            logging.error(f"Fehler beim Cleanup: {e}")
            import traceback
            logging.error(traceback.format_exc())
        finally:
            # Stellen sicher, dass das Programm beendet wird
            os._exit(0)
            
    def signal_handler(signum, frame):
        """Handler für System-Signale (CTRL+C)"""
        logging.info(f"Signal empfangen: {signum}")
        cleanup()
        
    def on_press(key):
        """Keyboard Event Handler"""
        global running, recorder, terminal_output
        try:
            if DEBUG:
                logging.debug(f"Taste gedrückt: {key}")
            
            if key == keyboard.Key.tab:
                if recorder.current_language == 'de':
                    recorder.current_language = 'en'
                else:
                    recorder.current_language = 'de'
                logging.info(f"Sprache gewechselt zu: {recorder.current_language}")
                terminal_output.update(language=recorder.current_language)
            elif key == keyboard.Key.esc:
                logging.info("ESC gedrückt - beende Programm")
                cleanup()
                return False
        except AttributeError as e:
            logging.error(f"Fehler bei Tastendruck: {e}")
    
    try:
        # Signal Handler registrieren
        system_signal.signal(system_signal.SIGINT, signal_handler)
        system_signal.signal(system_signal.SIGTERM, signal_handler)
        
        # Recorder mit Terminal-Output initialisieren
        with stderr_redirected():
            recorder = EnhancedAudioRecorder()
            terminal_output = TerminalOutput()
            recorder.output_handler = terminal_output
            
            # Keyboard Listener in separatem Thread starten
            listener = keyboard.Listener(on_press=on_press)
            listener.start()
            
            # Audio-Stream in Hauptthread starten
            recorder.start_stream()
            print(f"\nAufnahme gestartet mit {recorder.device.name}")
            print(f"Aktuelle Sprache: {'🇩🇪 Deutsch' if recorder.current_language == 'de' else '🇬🇧 English'}")
            print("Steuerung: [Tab] Sprache wechseln, [Esc] oder [Strg+C] zum Beenden\n")
            
            # Hauptschleife
            while running:
                time.sleep(0.1)
                
    except Exception as e:
        print(f"\nFehler: {e}")
    finally:
        cleanup()