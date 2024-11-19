#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Whisper Benchmark Visualisierungs-Server

Dieser Server stellt eine einfache HTTP-Schnittstelle bereit, um die Whisper Benchmark-Ergebnisse
im Browser zu visualisieren. Er bietet zwei Hauptfunktionen:

1. Bereitstellung der statischen Visualisierungs-HTML-Datei
2. API-Endpunkt zum Auflisten verfügbarer Benchmark-Dateien

Der Server unterstützt CORS (Cross-Origin Resource Sharing) für die Entwicklung
und kann die Benchmark-Ergebnisse im JSON-Format an den Client senden.

Typische Verwendung:
    python benchmark_server.py

Dies startet den Server auf Port 8000 und macht die Visualisierung unter
http://localhost:8000/visualize_benchmark.html verfügbar.
"""

from http.server import HTTPServer, SimpleHTTPRequestHandler
import json
from pathlib import Path
import os

class BenchmarkHandler(SimpleHTTPRequestHandler):
    """
    Spezialisierter HTTP-Handler für Benchmark-Visualisierungen.
    
    Erweitert den SimpleHTTPRequestHandler um zusätzliche Funktionalität:
    - API-Endpunkt zum Auflisten von Benchmark-Dateien
    - CORS-Unterstützung für Cross-Origin-Anfragen
    - Statische Datei-Bereitstellung für die Visualisierungs-HTML
    """
    
    def do_GET(self):
        """
        Behandelt eingehende GET-Anfragen.
        
        Spezielle Routen:
        - /list-benchmarks: Listet alle verfügbaren Benchmark-JSON-Dateien auf
        - Alle anderen Pfade: Werden als statische Dateianfragen behandelt
        
        Returns:
            None. Sendet die HTTP-Antwort direkt an den Client.
        """
        if self.path == '/list-benchmarks':
            # Liste alle JSON-Dateien im aktuellen Verzeichnis auf
            benchmark_files = list(Path('.').glob('whisper_benchmark_results_*.json'))
            
            # Sende die Liste als JSON-Antwort
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps([f.name for f in benchmark_files]).encode())
            return
        
        # Für alle anderen Pfade: Standard-Verhalten des SimpleHTTPRequestHandler
        return super().do_GET()

    def end_headers(self):
        """
        Erweitert die Standard-Header um CORS-Unterstützung.
        
        Fügt den 'Access-Control-Allow-Origin: *' Header hinzu, um
        Cross-Origin-Anfragen von allen Domains zu erlauben.
        """
        self.send_header('Access-Control-Allow-Origin', '*')
        super().end_headers()

def run_server(port=8000):
    """
    Startet den HTTP-Server auf dem angegebenen Port.
    
    Args:
        port (int): Der Port, auf dem der Server laufen soll (Standard: 8000)
    
    Der Server läuft unbegrenzt, bis er durch Ctrl+C unterbrochen wird.
    """
    server_address = ('', port)
    httpd = HTTPServer(server_address, BenchmarkHandler)
    print(f'Server läuft auf Port {port}...')
    print(f'Öffne http://localhost:{port}/visualize_benchmark.html im Browser')
    httpd.serve_forever()

if __name__ == '__main__':
    # Wechsel ins Verzeichnis mit den Benchmark-Dateien
    # Dies ist wichtig, damit die relativen Pfade korrekt aufgelöst werden
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    # Starte den Server mit Standardeinstellungen
    run_server()
