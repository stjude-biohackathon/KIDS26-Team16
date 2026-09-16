#!/usr/bin/env python3
"""
Lightweight Zero-Dependency HTTP Server for the SCOGS Results Dashboard.
Usage:
    python3 dashboard/server.py [port]
"""

import http.server
import socketserver
import os
import sys
import json
import glob

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8080
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

class DashboardRequestHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=BASE_DIR, **kwargs)

    def do_GET(self):
        # API route: list available runs in results/
        if self.path == "/api/runs":
            self.handle_api_runs()
        # API route: get a specific run file content
        elif self.path.startswith("/api/run/"):
            filename = self.path.replace("/api/run/", "").split("?")[0]
            self.handle_api_get_run(filename)
        else:
            super().do_GET()

    def handle_api_runs(self):
        files = []
        for path in glob.glob(os.path.join(RESULTS_DIR, "**/*"), recursive=True):
            if os.path.isfile(path) and (path.endswith(".json") or path.endswith(".csv")):
                rel_path = os.path.relpath(path, RESULTS_DIR)
                files.append({
                    "name": os.path.basename(path),
                    "relative_path": rel_path,
                    "size_bytes": os.path.getsize(path),
                    "is_json": path.endswith(".json"),
                    "is_csv": path.endswith(".csv")
                })
        
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps({"runs": files}, indent=2).encode("utf-8"))

    def handle_api_get_run(self, filename):
        target_path = os.path.abspath(os.path.join(RESULTS_DIR, filename))
        if not target_path.startswith(os.path.abspath(RESULTS_DIR)) or not os.path.isfile(target_path):
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b"File not found")
            return

        self.send_response(200)
        if target_path.endswith(".json"):
            self.send_header("Content-Type", "application/json")
        else:
            self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        
        with open(target_path, "rb") as f:
            self.wfile.write(f.read())


def run_server():
    socketserver.TCPServer.allow_reuse_address = True
    for port_attempt in range(PORT, PORT + 20):
        try:
            with socketserver.TCPServer(("", port_attempt), DashboardRequestHandler) as httpd:
                print(f"\n" + "=" * 60)
                print(f"🚀 SCOGS & MedGemma Results Dashboard is live!")
                print(f"👉 Local URL: http://localhost:{port_attempt}")
                print(f"📂 Serving dashboard from: {BASE_DIR}")
                print(f"📊 Reading results from:  {RESULTS_DIR}")
                print("=" * 60 + "\n")
                httpd.serve_forever()
                break
        except OSError as e:
            if "Address already in use" in str(e):
                continue
            raise e

if __name__ == "__main__":
    run_server()
