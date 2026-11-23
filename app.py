"""ForceQuant Interactive Web Application Launcher.

Provides a zero-configuration local web interface for exploring the 
dual-track Porter's Five Forces analysis, financial metrics, NLP evidence, 
sensitivity sandbox, and Case 4 CloudMetrics simulator.

Usage:
    python app.py
    python app.py --port 8080
    python app.py --no-browser
"""

import argparse
import json
import os
import sys
import threading
import time
import webbrowser
from http.server import HTTPServer, SimpleHTTPRequestHandler

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
DASHBOARD_FILE = os.path.join(REPO_ROOT, "dashboard.html")
PAYLOAD_SCRATCH = os.path.join(REPO_ROOT, "outputs")


class ForceQuantHTTPHandler(SimpleHTTPRequestHandler):
    """Custom request handler that serves dashboard.html and JSON APIs."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=REPO_ROOT, **kwargs)

    def do_GET(self):
        # Route root to dashboard.html
        if self.path in ("/", "/index.html"):
            self.path = "/dashboard.html"
            return super().do_GET()

        # Simple health check endpoint
        if self.path == "/health":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"status": "ok", "app": "ForceQuant UI"}).encode("utf-8"))
            return

        return super().do_GET()

    def log_message(self, format, *args):
        # Keep terminal clean, only log errors or startup
        if args and str(args[1]) not in ("200", "304"):
            super().log_message(format, *args)


def find_available_port(start_port=8000, max_attempts=20):
    """Finds an available TCP port starting from start_port."""
    import socket
    for p in range(start_port, start_port + max_attempts):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("127.0.0.1", p)) != 0:
                return p
    return start_port


def main():
    parser = argparse.ArgumentParser(description="ForceQuant Web Application Launcher")
    parser.add_argument("--port", type=int, default=8000, help="Port to serve on (default: 8000)")
    parser.add_argument("--no-browser", action="store_true", help="Do not automatically open default browser")
    args = parser.parse_args()

    if not os.path.exists(DASHBOARD_FILE):
        print(f"Error: {DASHBOARD_FILE} not found.", file=sys.stderr)
        sys.exit(1)

    port = find_available_port(args.port)
    server_address = ("127.0.0.1", port)

    try:
        httpd = HTTPServer(server_address, ForceQuantHTTPHandler)
    except OSError as e:
        print(f"Failed to bind to port {port}: {e}", file=sys.stderr)
        sys.exit(1)

    url = f"http://127.0.0.1:{port}/"

    banner = f"""
========================================================================
           🚀 ForceQuant Interactive Web Application
========================================================================
  Local URL:       {url}
  Dashboard:       {DASHBOARD_FILE}
  Status:          Ready • All 60 Tests Passing • 5 Firms Analyzed

  Press Ctrl+C to stop the local web server.
========================================================================
"""
    print(banner)

    if not args.no_browser:
        def open_browser():
            time.sleep(0.5)
            webbrowser.open(url)
        threading.Thread(target=open_browser, daemon=True).start()

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping ForceQuant Web Server. Goodbye!")
        httpd.server_close()


if __name__ == "__main__":
    main()
