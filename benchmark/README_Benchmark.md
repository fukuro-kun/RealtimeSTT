# Whisper Modell Benchmark-System

Dieses Repository enthält ein umfassendes Benchmark-System für Whisper-Spracherkennungsmodelle, das entwickelt wurde, um verschiedene Modelle systematisch zu vergleichen und ihre Leistung in Deutsch und Englisch zu evaluieren.

## 🎯 Überblick

Das System besteht aus drei Hauptkomponenten:
1. **Benchmark-Skript** (`whisper_model_benchmark.py`): Führt die eigentlichen Tests durch
2. **Visualisierungs-Server** (`benchmark_server.py`): Stellt die Ergebnisse bereit
3. **Visualisierungs-Interface** (`visualize_benchmark.html`): Zeigt die Ergebnisse grafisch an

## 📋 Installation

### 1. Conda-Umgebung erstellen und aktivieren
```bash
conda create -n realtimestt python=3.10
conda activate realtimestt
```

### 2. CUDA Libraries via pip installieren
```bash
pip install nvidia-cublas-cu12 nvidia-cudnn-cu12==9.*
```

### 3. PyTorch mit CUDA 12 installieren
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 4. Whisper und Dependencies installieren
```bash
pip install faster-whisper
pip install transformers
pip install soundfile
pip install psutil
pip install tqdm
```

### 5. Zusätzliche Dependencies
```bash
pip install 'accelerate>=0.26.0'
pip install python-Levenshtein
```

### 6. LD_LIBRARY_PATH Setup in conda
```bash
mkdir -p /home/fukuro/miniconda3/envs/realtimestt/etc/conda/activate.d

cat > /home/fukuro/miniconda3/envs/realtimestt/etc/conda/activate.d/cuda_path.sh << 'EOF'
#!/bin/bash
export LD_LIBRARY_PATH=/home/fukuro/miniconda3/envs/realtimestt/lib/python3.10/site-packages/nvidia/cublas/lib:/home/fukuro/miniconda3/envs/realtimestt/lib/python3.10/site-packages/nvidia/cudnn/lib
EOF

chmod +x /home/fukuro/miniconda3/envs/realtimestt/etc/conda/activate.d/cuda_path.sh
```

### 7. .env Konfiguration
- Kopieren Sie `.env.example` zu `.env`:
  ```bash
  cp .env.example .env
  ```
- Passen Sie die Werte in `.env` an:
  - `BENCHMARK_AUDIO_EN`: Pfad zur englischen Audio-Testdatei
  - `BENCHMARK_AUDIO_DE`: Pfad zur deutschen Audio-Testdatei
  - `BENCHMARK_TEXT_EN`: Englischer Referenztext für Genauigkeitsvergleich
  - `BENCHMARK_TEXT_DE`: Deutscher Referenztext für Genauigkeitsvergleich
  - `BENCHMARK_OUTPUT_EN`: Ausgabedatei für englische Benchmark-Ergebnisse
  - `BENCHMARK_OUTPUT_DE`: Ausgabedatei für deutsche Benchmark-Ergebnisse

## 🚀 Verwendung

### Benchmark-Skript

Das Skript unterstützt zwei Hauptverwendungsmodi:

1. **Vereinfachter Modus** (verwendet Standardwerte):
```bash
# Für Deutsch
python whisper_model_benchmark.py --language de

# Für Englisch
python whisper_model_benchmark.py --language en
```

2. **Erweiterter Modus** (mit expliziten Parametern):
```bash
python whisper_model_benchmark.py --audio /pfad/zur/datei.wav --language de --output ergebnis.json --device cuda
```

#### Wichtige Parameter:
- `--language`: Sprachcode (de/en)
- `--audio`: Pfad zur Audio-Datei (optional)
- `--output`: Ausgabedatei (optional)
- `--device`: Rechengerät (Standard: cuda)
- `--compute-type`: Berechnungstyp (Standard: float16)
- `--num-runs`: Anzahl der Durchläufe (Standard: 3)

### Visualisierungs-Server

Der Visualisierungs-Server (`benchmark_server.py`) ist eine wichtige Komponente des Benchmark-Systems:

#### Features
- Leichtgewichtiger HTTP-Server (basiert auf Python's SimpleHTTPRequestHandler)
- CORS-Unterstützung für lokale Entwicklung
- Automatisches Auflisten verfügbarer Benchmark-Dateien
- Statische Datei-Bereitstellung für die Visualisierung

#### Starten des Servers
```bash
cd tests  # Wichtig: Server muss im tests-Verzeichnis gestartet werden
python benchmark_server.py
```

Der Server startet standardmäßig auf Port 8000. Die Visualisierung ist dann verfügbar unter:
- http://localhost:8000/visualize_benchmark.html

#### API-Endpunkte
- `/list-benchmarks`: Listet alle verfügbaren Benchmark-JSON-Dateien auf
- `/*.json`: Direkter Zugriff auf Benchmark-Ergebnisdateien
- `/visualize_benchmark.html`: Hauptvisualisierungsinterface

#### Verwendung mit der Visualisierung
1. Server starten
2. Browser öffnen und zu http://localhost:8000/visualize_benchmark.html navigieren
3. Benchmark-Datei aus der Dropdown-Liste wählen
4. Ergebnisse werden automatisch geladen und visualisiert

#### Beenden des Servers
- Drücken Sie `Ctrl+C` im Terminal, um den Server zu beenden

### Visualisierungs-Interface

Öffnen Sie `visualize_benchmark.html` in einem modernen Webbrowser. Die Visualisierung zeigt:
- Box-Plots für Genauigkeit, Verarbeitungszeit, Textlänge und Segmentanzahl
- Detaillierte Modell-Karten mit Statistiken
- Vergleichende Analysen

## 📊 Benchmark-Ergebnisse

### Getestete Modelle
1. Basis-Modelle:
   - tiny, base, small, medium
   - large-v1, large-v2, large-v3

2. Spezialisierte Modelle:
   - Deutsch: primeline/whisper-large-v3-turbo-german
   - Englisch: distil-whisper/distil-large-v3, openai/whisper-large-v3

### Leistungsvergleich

#### Deutsch
- **Beste Genauigkeit**: primeline/whisper-large-v3-turbo-german (98.2%)
- **Beste Geschwindigkeit**: distil-whisper/distil-large-v3
- **Empfehlung**: primeline/whisper-large-v3-turbo-german
  - Optimiert für deutsche Sprache
  - Beste Balance aus Genauigkeit und Geschwindigkeit
  - Besonders gut bei Dialekten und Akzenten

#### Englisch
- **Beste Genauigkeit**: whisper-small (99.18%)
- **Beste Geschwindigkeit**: whisper-small (0.81s)
- **Empfehlung**: whisper-small oder whisper-medium
  - Hervorragendes Geschwindigkeits-/Genauigkeitsverhältnis
  - whisper-small: 99.18% Genauigkeit, 0.81s durchschnittliche Verarbeitungszeit
  - whisper-medium: 98.57% Genauigkeit, 1.42s durchschnittliche Verarbeitungszeit
  - Deutlich geringerer Speicherverbrauch (small: 1GB, medium: 2.5GB VRAM)
  - whisper-small bietet beste Balance aus Geschwindigkeit und Genauigkeit

## 🔧 Technische Details

### Benchmark-Skript
- Unterstützt verschiedene Whisper-Implementierungen
- Misst Genauigkeit, Geschwindigkeit und Ressourcenverbrauch
- Speichert detaillierte Ergebnisse im JSON-Format

### Visualisierungs-Server
- Leichtgewichtiger Python-Server
- Verarbeitet und serviert Benchmark-Ergebnisse
- Unterstützt CORS für lokale Entwicklung

### Visualisierungs-Interface
- Responsive Design mit Bootstrap 5.1.3
- Interaktive Plots mit Plotly.js
- Detaillierte Modell-Karten mit Statistiken

## 📈 Visualisierungen

Das Interface bietet vier Hauptvisualisierungen:
1. **Genauigkeits-Plot**: Vergleicht die Erkennungsgenauigkeit
2. **Zeit-Plot**: Zeigt die Verarbeitungsgeschwindigkeit
3. **Textlängen-Plot**: Analysiert die Ausgabemenge
4. **Segment-Plot**: Vergleicht die Segmentierung

## 🎯 Empfehlungen für die Produktion

### Für deutsche Sprache
```python
from faster_whisper import WhisperModel

# Empfohlen: Beste Performance für Deutsch
model = WhisperModel(
    model_size_or_path="large-v3",
    device="cuda",
    compute_type="float16",
    download_root="/pfad/zu/model/cache",  # Optional: Spezifiziere Download-Verzeichnis
    local_files_only=True,  # Optional: Nur lokale Dateien verwenden
)

# Inferenz
segments, info = model.transcribe(
    "audio.wav",
    language="de",
    beam_size=5,
    vad_filter=True,
    vad_parameters=dict(min_silence_duration_ms=500)
)
```

### Für englische Sprache
```python
from faster_whisper import WhisperModel

# Empfohlen: Beste Balance aus Geschwindigkeit und Genauigkeit
model = WhisperModel(
    model_size_or_path="small",
    device="cuda",
    compute_type="float16",
    download_root="/pfad/zu/model/cache",  # Optional: Spezifiziere Download-Verzeichnis
    local_files_only=True,  # Optional: Nur lokale Dateien verwenden
)

# Alternative: Ähnliche Genauigkeit, etwas langsamer
model = WhisperModel(
    model_size_or_path="medium",
    device="cuda",
    compute_type="float16"
)

# Inferenz
segments, info = model.transcribe(
    "audio.wav",
    language="en",
    beam_size=5,
    vad_filter=True,
    vad_parameters=dict(min_silence_duration_ms=500)
)

# Ergebnisse verarbeiten
for segment in segments:
    print(f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}")
```

## ⚠️ Bekannte Einschränkungen
- GPU mit mindestens 8GB VRAM empfohlen
- Benchmark-Ergebnisse können je nach Hardware variieren
- Speicherverbrauch steigt mit Modelgröße

## 🔄 Updates & Wartung
- Regelmäßige Updates der Modelle empfohlen
- Benchmark-Tests nach größeren Updates durchführen
- Ergebnisse regelmäßig überprüfen und vergleichen
