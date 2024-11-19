#!/usr/bin/env python3
"""
Echtzeit-Spracherkennung mit USB-Mikrofon
========================================

Dieses Skript ermöglicht die Echtzeit-Transkription von Sprache über ein USB-Mikrofon
mit Unterstützung für die deutsche Sprache.

Features:
- Echtzeit-Spracherkennung mit Whisper
- Automatische Pausenerkennung
- Farbige Echtzeitanzeige
- Optionale Tastatureingabe
"""

# ====== Konfiguration ======
EXTENDED_LOGGING = False

# Tastatureingabe-Intervall (0 = deaktiviert)
# Niedrigere Werte (0.002) für schnellere Eingabe, höhere (0.05) bei Problemen
WRITE_TO_KEYBOARD_INTERVAL = 0.002

# ====== Hilfsfunktionen für Textverarbeitung ======
def preprocess_text(text):
    """Bereitet den erkannten Text für die Anzeige vor."""
    # Entferne führende Leerzeichen und Auslassungspunkte
    text = text.lstrip()
    if text.startswith("..."):
        text = text[3:]
    text = text.lstrip()
    
    # Ersten Buchstaben groß schreiben
    if text:
        text = text[0].upper() + text[1:]
    return text

def text_detected(text):
    """Callback für die Echtzeit-Texterkennung."""
    global prev_text, displayed_text, rich_text_stored, recorder, update_counter

    text = preprocess_text(text)

    # Pausenlänge basierend auf Satzende anpassen
    if text.endswith("..."):
        recorder.post_speech_silence_duration = 2.0    # Mitten im Satz
    elif text and text[-1] in ['.', '!', '?', '。']:
        recorder.post_speech_silence_duration = 0.45   # Satzende
    else:
        recorder.post_speech_silence_duration = 0.7    # Unbekannt

    prev_text = text

    # Rich Text mit alternierenden Farben erstellen
    rich_text = Text()
    for i, sentence in enumerate(full_sentences):
        style = "yellow" if i % 2 == 0 else "cyan"
        rich_text += Text(sentence, style=style) + Text(" ")
    
    if text:
        rich_text += Text(text, style="bold yellow")

    new_displayed_text = rich_text.plain
    
    # Anzeige nur bei Änderungen aktualisieren (nicht zu häufig)
    update_counter += 1
    if new_displayed_text != displayed_text and update_counter >= 2:
        displayed_text = new_displayed_text
        panel = Panel(
            rich_text,
            title="[bold green]Live Transkription[/bold green]",
            border_style="bold green",
            padding=(1, 2)
        )
        live.update(panel)
        live.refresh()
        rich_text_stored = rich_text
        update_counter = 0

def process_text(text):
    """Verarbeitet den finalen erkannten Text."""
    global recorder, full_sentences, prev_text

    text = preprocess_text(text)
    text = text.rstrip()
    if text.endswith("..."):
        text = text[:-2]
            
    if not text:
        return

    full_sentences.append(text)
    prev_text = ""
    text_detected("")

    # Optional: Text per Tastatur ausgeben
    if WRITE_TO_KEYBOARD_INTERVAL:
        pyautogui.write(f"{text} ", interval=WRITE_TO_KEYBOARD_INTERVAL)

# ====== Hauptprogramm ======
if __name__ == '__main__':
    # Erforderliche Bibliotheken importieren
    import os
    import sys
    import ctypes
    from RealtimeSTT import AudioToTextRecorder
    from rich.console import Console
    from rich.live import Live
    from rich.text import Text
    from rich.panel import Panel
    import pyautogui
    import logging

    # ALSA Fehlermeldungen unterdrücken
    ERROR_HANDLER_FUNC = ctypes.CFUNCTYPE(None, ctypes.c_char_p, ctypes.c_int,
                                        ctypes.c_char_p, ctypes.c_int,
                                        ctypes.c_char_p)
    def py_error_handler(filename, line, function, err, fmt):
        pass
    try:
        c_error_handler = ERROR_HANDLER_FUNC(py_error_handler)
        asound = ctypes.CDLL('libasound.so.2')
        asound.snd_lib_error_set_handler(c_error_handler)
    except:
        pass

    # Rich Console für schöne Ausgabe initialisieren
    console = Console()
    console.print("System wird initialisiert...")

    live = Live(
        console=console,
        refresh_per_second=4,    # Reduzierte Rate gegen Flackern
        screen=True,             # Vollbild-Modus
        auto_refresh=False       # Manuelle Aktualisierung
    )
    live.start()

    # Globale Variablen für Textverarbeitung initialisieren
    full_sentences = []         # Liste aller erkannten Sätze
    rich_text_stored = ""       # Gespeicherter formatierter Text
    recorder = None             # Aufnahme-Objekt
    displayed_text = ""         # Aktuell angezeigter Text
    prev_text = ""             # Vorheriger Text
    update_counter = 0          # Zähler für Aktualisierungsrate

    # Recorder-Konfiguration mit optimierten Parametern
    recorder_config = {
        # Modell-Einstellungen
        'model': 'large-v2',
        'realtime_model_type': 'tiny',
        'language': 'de',
        'device': 'cuda',
        
        # Audio-Einstellungen
        'input_device_index': 11,  # USB-Mikrofon
        
        # VAD-Einstellungen (Voice Activity Detection)
        'silero_sensitivity': 0.05,
        'webrtc_sensitivity': 3,
        'silero_deactivity_detection': True,
        
        # Aufnahme-Einstellungen
        'post_speech_silence_duration': 0.7,
        'min_length_of_recording': 1.1,
        
        # Echtzeit-Transkription
        'enable_realtime_transcription': True,
        'realtime_processing_pause': 0.02,
        'on_realtime_transcription_update': text_detected,
        
        # Modell-Parameter
        'beam_size': 5,
        'beam_size_realtime': 3,
        
        # Sonstige Einstellungen
        'no_log_file': True,
        'initial_prompt': (
            "End incomplete sentences with ellipses.\n"
            "Examples:\n"
            "Complete: Der Himmel ist blau.\n"
            "Incomplete: Wenn der Himmel...\n"
            "Complete: Sie ging nach Hause.\n"
            "Incomplete: Weil er...\n"
        )
    }

    # Debug-Logging aktivieren falls gewünscht
    if EXTENDED_LOGGING:
        recorder_config['level'] = logging.DEBUG

    # Recorder initialisieren und starten
    recorder = AudioToTextRecorder(**recorder_config)
    
    # Startbildschirm anzeigen
    initial_text = Panel(
        Text("Sprechen Sie etwas...", style="cyan bold"),
        title="[bold yellow]Warte auf Eingabe[/bold yellow]",
        border_style="bold yellow",
        padding=(1, 2)
    )
    live.update(initial_text)
    live.refresh()

    # Hauptschleife für die Transkription
    try:
        while True:
            recorder.text(process_text)
    except KeyboardInterrupt:
        live.stop()
        console.print("[bold red]Transkription vom Benutzer gestoppt. Beende...[/bold red]")
        exit(0)
