#!/usr/bin/env python3
"""
Whisper Modell Benchmark-Tool
============================

Dieses Tool führt umfassende Leistungstests für verschiedene Whisper-Spracherkennungsmodelle durch.
Es unterstützt sowohl Standard Whisper-Modelle als auch spezialisierte Modelle von Hugging Face.

Funktionen:
-----------
- Verarbeitung von Audio-Dateien in verschiedenen Formaten
- Unterstützung für GPU (CUDA) und CPU-Verarbeitung
- Flexible Modellauswahl (Standard und sprachspezifisch)
- Detaillierte Leistungsmetriken
- JSON-Export der Ergebnisse

Verwendung:
----------
python whisper_model_benchmark.py --audio pfad/zur/audio.wav [optionen]

Wichtige Parameter:
-----------------
--audio:        Pfad zur Audio-Datei (optional, Standard basierend auf Sprache)
--output:       Ausgabepfad für JSON-Ergebnisse (optional, Standard basierend auf Sprache)
--device:       Rechengerät (cuda/auto/cpu/mps, Standard: cuda)
--compute-type: Berechnungstyp (float16/float32/int8)
--num-runs:     Anzahl der Testdurchläufe
--language:     Sprachcode (z.B. 'de', 'en')
--models:       Spezifische Modelle zum Testen

Beispiel:
--------
python whisper_model_benchmark.py --audio speech.wav --language de --device cuda
"""

# Standard-Bibliotheken
import os
import json
import time
import logging
import argparse
from typing import Dict, List, Optional, Union
from datetime import datetime
from pathlib import Path
import gc
import Levenshtein  # Für String-Ähnlichkeitsberechnung
from dotenv import load_dotenv
from pathlib import Path

# Lade Umgebungsvariablen aus .env im Hauptverzeichnis
env_path = Path(__file__).resolve().parent.parent / '.env'
load_dotenv(env_path)

# Audio-Verarbeitung
import soundfile
import torchaudio

# Machine Learning
import torch
from faster_whisper import WhisperModel
from transformers import pipeline as transformers_pipeline
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

# ====== Logging-Konfiguration ======
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WhisperModelBenchmark:
    """
    Benchmark-System für Whisper Spracherkennungsmodelle.
    
    Diese Klasse führt systematische Leistungstests für verschiedene Whisper-Modelle durch.
    Sie unterstützt sowohl Standard Whisper-Modelle als auch spezialisierte Hugging Face Modelle
    und kann auf verschiedenen Rechengeräten (CPU/GPU) ausgeführt werden.
    """
    # Basis-Modelle (Generalisten)
    STANDARD_MODELS = [
        "tiny",    # Schnellstes Modell, geringste Genauigkeit
        "base",    # Guter Kompromiss für einfache Anwendungen (präferiert für englische Sprache)
        "small",   # Ausgewogenes Modell                       (präferiert für englische Sprache)
        "medium",  # Hohe Genauigkeit
        "large-v1",# Erste Version des großen Modells
        "large-v2",# Verbesserte Version
        "large-v3" # Neueste Version, beste Genauigkeit
    ]

    # Sprachspezifische Modelle
    LANGUAGE_MODELS = {
        "de": [
            "primeline/whisper-large-v3-turbo-german"  # Optimiert für Deutsch (präferiert)
        ],
        "en": [
            "distil-whisper/distil-large-v3",  # neues, effizientes destilliertes Modell
            "openai/whisper-large-v3"          # OpenAI's neuestes Modell
        ]
    }

    # Ground Truth Texte für Genauigkeitsvergleich
    GROUND_TRUTH = {
        "en": os.getenv("BENCHMARK_TEXT_EN"),
        "de": os.getenv("BENCHMARK_TEXT_DE")
    }

    # Standard Audio-Dateien und Ausgabepfade für verschiedene Sprachen
    DEFAULT_PATHS = {
        "en": {
            "audio": os.getenv("BENCHMARK_AUDIO_EN"),
            "output": os.getenv("BENCHMARK_OUTPUT_EN")
        },
        "de": {
            "audio": os.getenv("BENCHMARK_AUDIO_DE"),
            "output": os.getenv("BENCHMARK_OUTPUT_DE")
        }
    }

    def __init__(self, compute_type="float16", device="cuda", num_runs=3):
        """
        Initialisiert das Benchmark-Objekt.
        :param compute_type: Der Typ der Berechnung (z.B. float16).
        :param device: Das zu verwendende Gerät (Standard: cuda).
        :param num_runs: Anzahl der Durchläufe pro Modell.
        """
        self.compute_type = compute_type
        self.device = self._determine_device() if device == "auto" else device
        self.num_runs = num_runs
        self.results = {}
        self.current_model = None
        
    def _determine_device(self) -> str:
        """
        Ermittelt das optimale Rechengerät für die Modellausführung.
        
        Prüft die Verfügbarkeit von:
        1. CUDA (NVIDIA GPU)
        2. MPS (Apple Silicon)
        3. CPU (Fallback)
        
        Returns:
            str: Name des zu verwendenden Geräts ('cuda', 'mps' oder 'cpu')
        """
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _load_audio(self, audio_path: str):
        """
        Lädt und verarbeitet eine Audio-Datei für die Whisper-Modelle.
        
        Features:
        - Automatisches Resampling auf 16kHz
        - Konvertierung zu float32
        - Unterstützung verschiedener Audio-Formate
        
        Args:
            audio_path (str): Pfad zur Audio-Datei
            
        Returns:
            dict: Audio-Daten und Sample-Rate
        """
        # Audio laden
        audio_input = soundfile.read(audio_path)
        
        # Resampling wenn nötig
        sampling_rate = audio_input[1]
        if sampling_rate != 16000:
            # Konvertiere zu float32 auf CPU
            waveform = torch.from_numpy(audio_input[0]).to(torch.float32)
            resampler = torchaudio.transforms.Resample(orig_freq=sampling_rate, new_freq=16000)
            audio_input = resampler(waveform).numpy()
        else:
            audio_input = audio_input[0]
        return {"array": audio_input, "sampling_rate": 16000}

    def _initialize_huggingface_pipeline(self, model_name: str):
        """
        Erstellt eine optimierte Hugging Face Pipeline für Spracherkennung.
        
        Optimierungen:
        - Automatische Geräteauswahl (GPU/CPU)
        - Optimierte Speichernutzung
        - Batch-Verarbeitung
        - Timestamp-Unterstützung
        
        Args:
            model_name (str): Name oder Pfad des Hugging Face Modells
            
        Returns:
            Pipeline: Konfigurierte Hugging Face Pipeline
        """
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        # Lade Modell mit optimierten Einstellungen
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True
        )
        model.to(device)

        # Lade Processor
        processor = AutoProcessor.from_pretrained(model_name)

        # Erstelle optimierte Pipeline
        pipe = transformers_pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            max_new_tokens=128,
            chunk_length_s=30,
            batch_size=16,
            return_timestamps=True,
            torch_dtype=torch_dtype,
            device=device,
        )

        return pipe

    def _is_huggingface_model(self, model_name: str) -> bool:
        """
        Überprüft, ob es sich um eine Hugging Face Modell-ID handelt.
        """
        return "/" in model_name

    def calculate_accuracy(self, transcribed_text: str, language: str) -> float:
        """
        Berechnet die Genauigkeit der Transkription im Vergleich zum Ground Truth Text.
        
        Args:
            transcribed_text (str): Die vom Modell erzeugte Transkription
            language (str): Sprache des Texts ('en' oder 'de')
            
        Returns:
            float: Genauigkeit als Prozentsatz (0-100)
        """
        if language not in self.GROUND_TRUTH:
            return 0.0
            
        ground_truth = self.GROUND_TRUTH[language]
        
        # Normalisiere Texte für besseren Vergleich
        transcribed_text = transcribed_text.lower().strip()
        ground_truth = ground_truth.lower().strip()
        
        # Berechne Levenshtein-Distanz
        distance = Levenshtein.distance(transcribed_text, ground_truth)
        max_length = max(len(transcribed_text), len(ground_truth))
        
        # Berechne Genauigkeit in Prozent
        accuracy = (1 - distance / max_length) * 100
        return round(accuracy, 2)

    def benchmark_model(self, model_name: str, audio_path: str, language: str = "de", 
                       num_runs: int = 3, device: str = "cuda", 
                       compute_type: str = "float16") -> Dict:
        """
        Führt einen vollständigen Benchmark-Test für ein einzelnes Modell durch.
        
        Prozess:
        1. Modell-Initialisierung (Hugging Face oder Whisper)
        2. Audio-Verarbeitung
        3. Mehrfache Transkriptionsdurchläufe
        4. Erfassung von Metriken (Zeit, Speicher)
        5. Cleanup und Speicherfreigabe
        
        Args:
            model_name (str): Name des zu testenden Modells
            audio_path (str): Pfad zur Test-Audio-Datei
            language (str, optional): Sprachcode für die Transkription
            num_runs (int, optional): Anzahl der Durchläufe
            device (str, optional): Rechengerät
            compute_type (str, optional): Berechnungstyp
            
        Returns:
            Dict: Benchmark-Ergebnisse mit allen Metriken
        """
        logger.info(f"Teste Modell: {model_name}")
        self.current_model = model_name
        
        results = {
            "model_name": model_name,
            "device": device,
            "compute_type": compute_type,
            "runs": []
        }

        try:
            # Audio laden
            audio_input = self._load_audio(audio_path)

            if self._is_huggingface_model(model_name):
                # Nutze optimierte Hugging Face Pipeline
                pipe = self._initialize_huggingface_pipeline(model_name)
                
                for i in range(num_runs):
                    start_time = time.time()
                    
                    # Verarbeite Audio direkt mit der Pipeline
                    result = pipe(
                        audio_input["array"],
                        return_timestamps=True
                    )
                    
                    # Extrahiere Text und Segmente aus dem Hugging Face Format
                    text = result["text"] if isinstance(result, dict) else result
                    chunks = result["chunks"] if isinstance(result, dict) and "chunks" in result else []
                    segments = [
                        {
                            "start": chunk["timestamp"][0] if isinstance(chunk["timestamp"], (list, tuple)) else 0.0,
                            "end": chunk["timestamp"][1] if isinstance(chunk["timestamp"], (list, tuple)) else 0.0,
                            "text": chunk["text"]
                        }
                        for chunk in chunks
                    ] if chunks else [{"start": 0.0, "end": 0.0, "text": text}]
                    
                    end_time = time.time()
                    
                    # Berechne Genauigkeit
                    accuracy = self.calculate_accuracy(text, language)
                    
                    run_result = {
                        "run": i + 1,
                        "time": end_time - start_time,
                        "text": text,
                        "segments": segments,
                        "accuracy": accuracy
                    }
                    results["runs"].append(run_result)
                    logger.info(f"Durchlauf {i + 1} abgeschlossen: {run_result['time']:.2f}s")

            else:
                # Für Standard Whisper Modelle
                model = WhisperModel(
                    model_name,
                    device=device,
                    compute_type=compute_type
                )
                
                for i in range(num_runs):
                    start_time = time.time()
                    
                    # Transkribiere mit faster-whisper
                    segments, info = model.transcribe(
                        audio_path,
                        language=language,
                        beam_size=5
                    )
                    
                    # Konvertiere Generator zu Liste
                    segments = list(segments)
                    
                    end_time = time.time()
                    
                    # Vollständiges Transkript erstellen
                    full_transcript = " ".join([seg.text for seg in segments])
                    
                    # Berechne Genauigkeit
                    accuracy = self.calculate_accuracy(full_transcript, language)
                    
                    run_result = {
                        "run": i + 1,
                        "time": end_time - start_time,
                        "text": full_transcript,
                        "segments": [
                            {
                                "start": seg.start,
                                "end": seg.end,
                                "text": seg.text
                            } for seg in segments
                        ],
                        "accuracy": accuracy
                    }
                    results["runs"].append(run_result)
                    logger.info(f"Durchlauf {i + 1} abgeschlossen: {run_result['time']:.2f}s")

        except Exception as e:
            logger.error(f"Fehler beim Testen von {model_name}: {str(e)}")
            results["error"] = str(e)
            
        finally:
            # Aufräumen
            if 'pipe' in locals():
                del pipe
            if 'model' in locals():
                del model
            
            gc.collect()
            if device == "cuda":
                torch.cuda.empty_cache()

        return results

    def run_benchmarks(self, audio_path: str, models: List[str] = None, language: str = "de") -> Dict:
        """
        Führt Benchmark-Tests für mehrere Modelle durch.
        
        Features:
        - Automatische Modellauswahl basierend auf Sprache
        - Zeitmessung für Gesamtdurchlauf
        - Detaillierte Metadaten-Erfassung
        
        Args:
            audio_path (str): Pfad zur Test-Audio-Datei
            models (List[str], optional): Liste spezifischer Modelle
            language (str, optional): Sprachcode für die Tests
            
        Returns:
            Dict: Gesamtergebnisse aller Modell-Tests
        """
        # Startzeit des Benchmarks
        start_time = datetime.now()
        
        if not models:
            models = self.get_available_models(language)
        
        for model_name in models:
            self.results[model_name] = self.benchmark_model(model_name, audio_path, language)
        
        # Endzeit des Benchmarks
        end_time = datetime.now()
        
        # Füge Metadaten hinzu
        self.results["_metadata"] = {
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "duration": str(end_time - start_time),
            "audio_file": audio_path,
            "language": language
        }
        
        return self.results

    def save_results(self, output_path: str):
        """
        Speichert die Benchmark-Ergebnisse im JSON-Format.
        
        Die Ergebnisse enthalten:
        - Zeitmessungen pro Durchlauf
        - Transkriptionsergebnisse
        - Metadaten zum Benchmark
        
        Args:
            output_path (str): Pfad für die JSON-Ausgabedatei
        """
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2)
        logger.info(f"Ergebnisse wurden in {output_path} gespeichert")

    def get_available_models(self, language: str = "de") -> List[str]:
        """
        Ermittelt verfügbare Modelle basierend auf der gewählten Sprache.
        
        Logik:
        1. Beginnt mit Standard-Modellen (sprachunabhängig)
        2. Fügt sprachspezifische Modelle hinzu, falls Sprache angegeben
        
        Args:
            language (str, optional): Sprachcode (z.B. 'de', 'en')
            
        Returns:
            List[str]: Liste verfügbarer Modellnamen
        """
        models = self.STANDARD_MODELS.copy()  # Start mit Basis-Modellen

        # Füge sprachspezifische Modelle hinzu, falls eine Sprache angegeben wurde
        if language and language in self.LANGUAGE_MODELS:
            models.extend(self.LANGUAGE_MODELS[language])

        return models

def main():
    """
    Hauptfunktion des Benchmark-Tools.
    
    Ablauf:
    1. Parsen der Kommandozeilenargumente
    2. Initialisierung des Benchmark-Systems
    3. Ausführung der Tests
    4. Speichern der Ergebnisse
    
    Kommandozeilenoptionen:
    --audio:        Pfad zur Audio-Datei (optional, Standard basierend auf Sprache)
    --output:       JSON-Ausgabedatei (optional, Standard basierend auf Sprache)
    --device:       Rechengerät (cuda/auto/cpu/mps, Standard: cuda)
    --compute-type: Berechnungstyp
    --num-runs:     Anzahl der Durchläufe
    --language:     Sprachcode
    --models:       Spezifische Modelle
    """
    parser = argparse.ArgumentParser(description="Benchmark-Tool für verschiedene Whisper-Modelle")
    
    # Optionale Parameter
    parser.add_argument("--language", default="de",
                       help="Sprachcode (z.B. 'de' für Deutsch)")
    parser.add_argument("--audio", 
                       help="Pfad zur Audio-Datei für die Transkription")
    parser.add_argument("--output",  
                       help="Ausgabedatei für Ergebnisse")
    parser.add_argument("--device", default="cuda", 
                       choices=["auto", "cuda", "cpu", "mps"], 
                       help="Rechengerät (Standard: cuda)")
    parser.add_argument("--compute-type", default="float16",
                       choices=["float16", "float32", "int8"],
                       help="Berechnungstyp")
    parser.add_argument("--num-runs", type=int, default=3,
                       help="Anzahl der Durchläufe pro Modell")
    parser.add_argument("--models", nargs="+",
                       help="Liste spezifischer zu testender Modelle")

    args = parser.parse_args()
    
    # Setze Standardwerte basierend auf der Sprache
    if args.language not in WhisperModelBenchmark.DEFAULT_PATHS:
        raise ValueError(f"Sprache '{args.language}' wird nicht unterstützt. "
                       f"Verfügbare Sprachen: {list(WhisperModelBenchmark.DEFAULT_PATHS.keys())}")
            
    defaults = WhisperModelBenchmark.DEFAULT_PATHS[args.language]
    audio_path = args.audio or defaults["audio"]
    output_path = args.output or defaults["output"]
        
    # Überprüfe, ob die Audio-Datei existiert
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio-Datei nicht gefunden: {audio_path}")
        
    # Benchmark-System initialisieren und ausführen
    benchmark = WhisperModelBenchmark(
        compute_type=args.compute_type,
        device=args.device,
        num_runs=args.num_runs
    )
        
    results = benchmark.run_benchmarks(
        audio_path=audio_path,
        models=args.models,
        language=args.language
    )
        
    # Ergebnisse speichern
    benchmark.save_results(output_path)

if __name__ == "__main__":
    main()