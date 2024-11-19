# RealtimeSTT Technisches Entwicklungsprotokoll

Dieses Dokument dient als technisches Gedächtnis des RealtimeSTT-Projekts. Es ist ein lebendes Dokument, das kontinuierlich aktualisiert wird und folgende Aspekte festhält:

## Zweck
- Dokumentation kritischer technischer Parameter und Konfigurationen
- Archivierung von Implementierungsentscheidungen und deren Begründungen
- Sammlung von Problemlösungen und technischen Erkenntnissen
- Nachvollziehbarkeit der Entwicklungshistorie

## Verwendung
- Nachschlagewerk für exakte technische Spezifikationen
- Referenz für Debugging und Problemlösung
- Vermeidung bereits getesteter, nicht funktionierender Ansätze
- Wissensspeicher für neue Projektmitarbeiter

## Inhaltsverzeichnis

### 1. ⚙️ Kritische Konfiguration
- Audio-Verarbeitung
- Voice Activity Detection (VAD)
- Sprachmodelle und KI
- Thread-Management
- X11-Integration

### 2. 💡 Technische Erkenntnisse
- Erfolgreiche Implementierungen
- Verworfene Ansätze
- Kritische Berechnungen
- Performance-Optimierungen

### 3. 🚧 Aktuelle Entwicklung
- Bekannte Probleme
- Work in Progress
- Nächste Schritte
- Aktuelle Experimente

## 🎯 Vision
Entwicklung eines robusten, multilingualen Echtzeit-Spracherkennungssystems mit Fokus auf:
- Präzise Spracherkennung für Deutsch und Englisch
- Minimale Latenz
- Adaptive Sprachpausenerkennung
- Flexible Ausgabemodi

### 1. ⚙️ Kritische Konfiguration

### Audio-Verarbeitung

#### Grundlegende Parameter
```python
DEVICE_RATE = 44100    # Hardware-Abtastrate
MODEL_RATE = 16000     # Erforderliche Rate für VAD und Whisper
CHANNELS = 1           # Mono-Audio
CHUNK_SIZE = 1323      # Berechnet für exakte 30ms Frames nach Resampling
FORMAT = paFloat32     # 32-bit Float Audio-Format
```

#### Frame-Verarbeitung
- Exakte Frame-Größe: 30ms bei 16kHz = 480 Samples
- Frame-Größe hat größeren Einfluss auf VAD-Qualität als Abtastrate
- Jeder Frame muss EXAKT diese Größe haben
- Zu kleine oder große Frames werden von WebRTC VAD abgelehnt
- Frame-Länge muss VOR der VAD-Verarbeitung geprüft werden
- Unvollständige Frames am Ende des Audio-Chunks überspringen

#### Byte-Konvertierung und Normalisierung
- Audio-Samples müssen als 16-bit Integer vorliegen
- 2 Bytes pro Sample (int16) = 960 Bytes pro Frame
- Wichtig: Korrekte Normalisierung auf [-32768, 32767]
- Normalisierung VOR Resampling verbessert Audioqualität
- Schwellwert bei 0.001 für Audio-Pegel

#### Hardware-Timing und Puffer
- USB-Mikrofon-spezifische Verzögerungen
- Präzise Frame-Synchronisation notwendig
- Kompensation von Hardware-Jitter
- Überlappende Audio-Fenster trotz höherem Ressourcenverbrauch nötig
- 25-Sekunden-Limit verhindert überraschenderweise Speicherlecks
- Mehr Puffer-Kopien können Performance verbessern (Cache-Alignment)

### Sprachmodelle und KI

#### Deutsch (primeline/whisper-large-v3-turbo-german)
- Accuracy: 98.2%
- Optimiert für deutsche Dialekte
- Performance-Monitoring für Ladezeiten
- Thread-Status-Tracking bei Modellwechsel
- Schnelle Inferenz durch Turbo-Optimierung

#### Englisch (openai/whisper-small)
- Accuracy: 99.18%
- Geringer Ressourcenverbrauch
- Effiziente GPU-Nutzung
- Kontinuierliches Performance-Profiling
- Ideal für Echtzeit-Verarbeitung

#### Modell-Management
- Vorladestrategie statt Lazy Loading (schneller trotz höherem initialen Speicherverbrauch)
- Permanenter Modell-Cache
- Optimierte CUDA-Nutzung
- Float16-Präzision wo möglich
- Beide Modelle beim Start laden
- Permanent im Speicher halten
- Schnelles Switching ohne Ladezeiten
- Keine dynamische Nachladelogik

### Voice Activity Detection (VAD)

#### Grundeinstellungen
```python
VAD_FRAME_MS = 30          # Frame-Dauer in Millisekunden
MIN_SPEECH_DURATION = 0.3   # Minimale Dauer für Spracherkennung
POST_SPEECH_SILENCE = {
    "incomplete": 2.0,    # Für "..." am Ende
    "complete": 0.45,     # Für ".", "!", "?"
    "unclear": 0.7       # Für unklare Enden
}
VAD_MODE = 2               # Mittlere Aggressivität (0-3)
```

#### Adaptive Pausenerkennung
- **Kurze Pause**: 0.45s nach Satzende (".", "!", "?")
- **Mittlere Pause**: 0.7s bei unklarem Ende
- **Lange Pause**: 2.0s bei Unterbrechungen ("...")

#### Optimierte Konfiguration
- Aggressivitätslevel 2 für beste Balance
- Präzise Frame-Ausrichtung für zuverlässige Erkennung
- Schwellwertbasierte Aktivierung
- Reduzierte False-Positives durch angepasste Parameter

### Thread-Management
- Thread-sichere Queues für Audio-Verarbeitung
- Robustes Exception-Handling
- Saubere Ressourcen-Freigabe
- Automatische Buffer-Bereinigung

#### Verworfene Threading-Modelle
1. **Single-Thread mit Callback**
   - Zu hohe Latenz
   - Audio-Dropouts bei Modellwechsel
   - Blockierung der UI

2. **Thread-Pool Ansatz**
   - Overhead durch Task-Scheduling
   - Komplexe Synchronisation
   - Ressourcenverschwendung

3. **Async/Await Modell**
   - Nicht kompatibel mit PortAudio
   - Komplexe Error-Propagation
   - Schwierige Ressourcenverwaltung

#### Aktuelles Modell
- Dedizierte Threads für:
  * Audio-Capture (realtime-priority)
  * VAD-Processing
  * Whisper-Inference
  * UI-Updates
- Ringpuffer für Thread-Kommunikation
- Explizite Synchronisationspunkte
- Robuste Exception-Propagation

### X11-Integration
#### Grundkonfiguration
- xdotool für Cursor-Steuerung
- X11-Display-Environment erforderlich
- Root-Rechte nicht notwendig

#### Kritische Parameter
- Cursor-Update-Rate: 75ms (experimentell ermittelt)
- X11-Display-Buffer-Size: 8192 bytes
- Event-Queue-Size: 256 events

#### Bekannte Einschränkungen
- Funktioniert nur unter X11, nicht Wayland
- Root-Window muss zugänglich sein
- Multi-Monitor Setup erfordert spezielle Konfiguration

#### Cursor-Management
- Präzise Positionierung via xdotool
- Koordinaten-Mapping zwischen Screens
- Event-basierte Position-Updates
- Throttling zur Vermeidung von Screen-Tearing

#### Error-Handling
- X11-Connection-Loss Recovery
- Display-Buffer-Overflow Protection
- Auto-Reconnect bei Connection-Drop
- Graceful Fallback bei fehlenden Berechtigungen

## 2. 💡 Technische Erkenntnisse

### Implementierungs-Evolution
#### Standard Version (Original Whisper)
- Wichtige Erkenntnisse:
  * Direkte Whisper-Integration zu langsam für Echtzeit
  * Hardware-Anforderungen nicht praktikabel
  * Wertvolle Basis für Optimierungsstrategien

#### USB Version (realtimestt_test_usb.py)
- Kritische Learnings:
  * Überlappende Audio-Fenster notwendig gegen Wortverluste
  * 25-Sekunden-Limit pro Block verhindert Memory-Leaks
  * Hardware-Timing kritisch für USB-Mikrofone
  * Adaptive Pausenerkennung verbessert Nutzererlebnis

#### HF Version (realtimestt_test_hf.py)
- Wichtige Erkenntnisse:
  * Vorladestrategie besser als Lazy Loading
  * Terminal-GUI erschwert Debugging erheblich
  * Lösung: Separates Logging-System implementiert
  * Modell-Wechsel erzeugt Audio-Dropouts

### Architektur-Evolution
- **Ursprünglicher Ansatz**: 
  * Batch-basierte Verarbeitung
  * Feste Puffergröße
  * Statische Pausenerkennung

- **Neue Architektur** (basierend auf realtimestt_test_usb.py):
  * Kontinuierlicher Audio-Stream
  * Chunk-basierte Echtzeit-Verarbeitung
  * Dynamische Puffer-Anpassung
  * Adaptive VAD-Integration
  * Verbesserte Thread-Synchronisation

#### Vorteile der neuen Architektur
1. **Performance**:
   * Geringere Latenz durch Stream-Verarbeitung
   * Effizientere Ressourcennutzung
   * Bessere CPU-Auslastung

2. **Zuverlässigkeit**:
   * Stabilere USB-Mikrofon-Erkennung
   * Robustere Audio-Capture
   * Verbesserte Fehlerbehandlung

3. **Flexibilität**:
   * Einfachere Integration neuer Features
   * Modulare Komponenten
   * Bessere Testbarkeit

#### Implementierungsdetails
```python
# Neue Stream-basierte Verarbeitung
def audio_callback(self, in_data, frame_count, time_info, status):
    """Callback für kontinuierliche Audio-Verarbeitung"""
    if not self._running:
        return (None, pyaudio.paComplete)
    
    # Echtzeit-Verarbeitung statt Batch
    audio_data = np.frombuffer(in_data, dtype=np.float32)
    is_speech = self.is_speech(in_data)
    
    if is_speech:
        self.audio_buffer.append(in_data)
    else:
        # Adaptive Pausenerkennung
        self.process_buffer()
    
    return (in_data, pyaudio.paContinue)
```

#### Kritische Aspekte
- Thread-Safety in der Audio-Verarbeitung
- Korrekte Buffer-Synchronisation
- Präzise Timing-Kontrolle
- Speichermanagement bei kontinuierlichem Streaming

### Frame-Verarbeitung
#### Grundlegende Parameter
```python
DEVICE_RATE = 44100    # Hardware-Abtastrate
MODEL_RATE = 16000     # Erforderliche Rate für VAD und Whisper
CHANNELS = 1           # Mono-Audio
CHUNK_SIZE = 1323      # Berechnet für exakte 30ms Frames nach Resampling
FORMAT = paFloat32     # 32-bit Float Audio-Format
```

#### Frame-Größen-Berechnung
Der wichtigste Aspekt ist die Einhaltung exakter 30ms Frame-Größen für WebRTC VAD:

1. WebRTC VAD benötigt exakt 480 Samples pro Frame bei 16kHz (30ms)
2. Um dies nach dem Resampling zu erreichen, brauchen wir:
   ```
   480 * (44100/16000) = 1323 Samples bei 44.1kHz
   ```
3. Nach dem Resampling von 1323 Samples bei 44.1kHz erhalten wir exakt 480 Samples bei 16kHz

#### Audio-Format-Spezifikationen
- **Eingabeformat**: Float32 (-1.0 bis 1.0)
- **VAD-Format**: Int16 (-32768 bis 32767)
- **Normalisierung**: Schwellwert bei 0.001 für Audio-Pegel

#### Frame-Verarbeitung
1. **Frame-Größen-Präzision**
   - Exakte Frame-Größen sind kritisch für VAD
   - Falsche Resampling-Verhältnisse führen zu Frame-Fehlausrichtung
   - Puffergrößen müssen das Resampling-Verhältnis berücksichtigen
   - Frame-Länge muss VOR der VAD-Verarbeitung geprüft werden
   - Unvollständige Frames am Ende des Audio-Chunks überspringen

2. **Audio-Format-Behandlung**
   - Eingang: Float32 (-1.0 bis 1.0)
   - VAD: Int16 (-32768 bis 32767)
   - Korrekte Normalisierung ist essentiell
   - Wichtig: Korrekte Normalisierung auf [-32768, 32767]

3. **Performance-Optimierungen**
   - Minimierung von Pufferkopien
   - Effizientes Numpy-Array-Slicing für Frame-Extraktion
   - Vorherige Normalisierung der Audio-Daten
   - Reduzierte False-Positives durch angepasste Schwellwerte
   - Effizientes Resampling
   - Echtzeitverarbeitungs-Anforderungen

4. **Kontraintuitive Optimierungen**
   - Frame-Größe hat größeren Einfluss auf VAD-Qualität als Abtastrate
   - Normalisierung VOR Resampling verbessert Audioqualität
   - Mehr Puffer-Kopien können Performance verbessern (Cache-Alignment)
   - Speech-Ratio muss über ALLE gültigen Frames berechnet werden
   - Überlappende Audio-Fenster trotz höherem Ressourcenverbrauch nötig
   - 25-Sekunden-Limit verhindert überraschenderweise Speicherlecks
   - Vorladestrategie schneller als Lazy Loading trotz höherem initialen Speicherverbrauch

### Audio-Stream-Management
#### Kritische Erkenntnisse
- Chunk-basierte Verarbeitung mit präzisem Timing
- Adaptive Pausenerkennung für natürliche Segmentierung
- Überlappende Fenster gegen Wortverluste
- Buffer-Management mit 25-Sekunden-Limit

#### Hardware-Timing
- USB-Mikrofon-spezifische Verzögerungen
- Präzise Frame-Synchronisation notwendig
- Kompensation von Hardware-Jitter
- Robuste Error-Handling für Timing-Probleme

### Debugging und Problemlösung
#### Debugging-Tipps
1. Überwachung der Audio-Pegel in Debug-Logs
2. Überprüfung der Frame-Größen nach Resampling
3. Kontrolle der VAD-Frame-Ausrichtung
4. Validierung der Abtastraten-Konvertierungen

#### Häufige Probleme und Lösungen
1. **Frame-Fehlausrichtung**
   - Symptom: Keine Spracherkennung
   - Ursache: Falsche Chunk-Größe
   - Lösung: Exakte Chunk-Größe (1323 Samples)

2. **Audio-Pegel-Probleme**
   - Symptom: Falsch-Negative
   - Ursache: Fehlerhafte Normalisierung
   - Lösung: Korrekte Float32-zu-Int16-Konvertierung

3. **Resampling-Artefakte**
   - Symptom: Schlechte Erkennungsqualität
   - Ursache: Ungenaues Resampling
   - Lösung: Korrektes Verhältnis und Puffergröße

### Implementierungserfolge
- Erfolgreiche Tests mit USB-Mikrofon
- Präzise Spracherkennung für Deutsch
- Robuste Frame-Verarbeitung
- Minimale Latenz
- Zuverlässige Sprachsegmentierung

## 3. 🚧 Aktuelle Entwicklung

### Bekannte Probleme
1. **Terminal-Fenster-Anpassung**
   - Problem: Terminalausgabe überschreibt sich bei mehrzeiligem Text
   - Getestete Ansätze:
     * ANSI-Escape-Sequenzen für Zeilenlöschung (unzuverlässig)
     * tput-Befehle für Fensteranpassung (funktioniert nicht überall)
     * Curses-Library (vielversprechend, aber komplexere Implementation)
   - Aktueller Stand: Curses-basierte Lösung in Entwicklung

2. **Audio-Processing**
   - ALSA-Warnungen bei Programmstart
   - Gelegentliche Verzögerungen bei Sprachmoduswechsel
   - Speichernutzung bei langen Aufnahmen
   - Transkriptionen werden nicht angezeigt
     * Vermutung: Problem in der Verarbeitungskette
     * Logging zeigt erfolgreiche Erkennung, aber keine Anzeige

### Work in Progress
1. **Audio-Qualitätskontrolle**
   - Implementierung von Audio-Level-Schwellwerten
   - Optimierung der Normalisierung
   - Validierung der Resampling-Qualität

2. **Whisper-Integration**
   - Validierung der Modell-Ausgabe
   - Fehlerbehandlung bei der Transkription
   - Performance-Monitoring

3. **Textformatierung**
   - Getrennte Zuständigkeiten:
     * `process_text()`: VAD-Parameter-Steuerung
     * `format_output_text()`: Anzeigeformatierung
   - Wichtig: Original-Text für VAD-Logik beibehalten
   - Formatierung erst nach VAD-Verarbeitung
   - Sorgfältige Behandlung von Sonderzeichen ("...")

### Nächste Schritte
1. **Kurzfristig**
   - Feinjustierung der VAD-Parameter
   - Implementation der adaptiven Pausenerkennung
   - Debug-Logging in der Verarbeitungskette

2. **Mittelfristig**
   - Audio-Level-Meter
   - Detaillierte Audio-Diagnostik
   - Offline-Test-Modus
   - Performance-Profiling

3. **Langfristig**
   - Wayland Support evaluieren
   - Performance-Optimierungen
   - Weitere Sprach-Modelle

### Aktuelle Experimente
- Audio-Level-Meter Tests
- VAD-Parameter Optimierung
- Modell-Performance-Tests

## 4. 🔄 Git und Versionierung

#### Sprachrichtlinien
- **Code und Commits**
  - Variablen, Funktionen und Klassen auf Englisch
  - Commit-Messages auf Englisch
  - Verbessert Universalität und erleichtert potenzielle Pull Requests

- **Dokumentation**
  - README.md: Zweisprachig (Deutsch/Englisch)
  - Technische Dokumentation (DEVELOPMENT.md, README_Benchmark.md): Primär Deutsch
  - Code-Kommentare: Deutsch (lokale Entwicklung)

- **Pull Request Vorbereitung**
  - Betroffene Dokumentation ins Englische übersetzen
  - Code-Kommentare der betroffenen Dateien ins Englische übersetzen
  - Nur für Dateien, die Teil des Pull Requests sind

Diese Richtlinien ermöglichen:
- Effiziente lokale Entwicklung in Deutsch
- Universelle Code-Verständlichkeit
- Einfache Integration von Pull Requests
- Zugänglichkeit für internationale Entwickler
