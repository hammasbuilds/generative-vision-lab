"""Generative Vision Lab viewer: a small local web page over the lab's committed outputs.

    python app/server.py [--port 8843] [--no-browser]

Standard-library HTTP server; Pillow and NumPy only for the per-image PSNR.
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

sys.path.insert(0, str(Path(__file__).resolve().parent))
import catalogue  # noqa: E402

STATIC = Path(__file__).resolve().parent / "static"
DEFAULT_PORT = 8843


class Handler(BaseHTTPRequestHandler):
    server_version = "GenerativeVisionLab/1.0"

    def log_message(self, fmt, *args):  # keep the console quiet
        pass

    def _send(self, code: int, body: bytes, ctype: str, cache: bool = False) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "max-age=3600" if cache else "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        if self.command != "HEAD":
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                pass  # the browser moved on before the image finished; nothing to do

    def _json(self, code: int, obj) -> None:
        self._send(code, json.dumps(obj).encode("utf-8"), "application/json; charset=utf-8")

    def do_HEAD(self):
        self.do_GET()

    def do_GET(self):
        url = urlparse(self.path)
        q = parse_qs(url.query)
        path = url.path
        if path in ("/", "/index.html"):
            return self._static("index.html")
        if path.startswith("/static/"):
            return self._static(path[len("/static/"):])
        if path == "/api/health":
            return self._json(200, {"ok": True})
        if path == "/api/catalogue":
            return self._json(200, catalogue.catalogue())
        if path == "/api/scores":
            task = (q.get("task") or [""])[0]
            sample = (q.get("sample") or [""])[0]
            if task not in catalogue.BY_ID:
                return self._json(404, {"error": f"unknown task {task!r}"})
            scores = catalogue.sample_scores(task, sample)
            if not scores:
                return self._json(404, {"error": f"no scored sample {sample!r} in {task}"})
            return self._json(200, scores)
        if path == "/thumb":
            f = catalogue.public_file((q.get("p") or [""])[0])
            if f is None or f.suffix.lower() != ".png":
                return self._json(404, {"error": "not a committed sample image"})
            return self._send(200, catalogue.thumbnail(f), "image/jpeg", cache=True)
        if path == "/file":
            f = catalogue.public_file((q.get("p") or [""])[0])
            if f is None:
                return self._json(404, {"error": "not a committed sample file"})
            ctype = mimetypes.guess_type(f.name)[0] or "application/octet-stream"
            return self._send(200, f.read_bytes(), ctype, cache=True)
        return self._json(404, {"error": "not found"})

    def _static(self, name: str) -> None:
        f = (STATIC / name).resolve()
        if f.parent != STATIC.resolve() or not f.is_file():
            return self._json(404, {"error": "not found"})
        ctype = mimetypes.guess_type(f.name)[0] or "application/octet-stream"
        if ctype.startswith("text/") or ctype.endswith("javascript"):
            ctype += "; charset=utf-8"
        return self._send(200, f.read_bytes(), ctype)


class Server(ThreadingHTTPServer):
    daemon_threads = True
    request_queue_size = 64  # a page asks for up to a dozen multi-megabyte PNGs at once


def make_server(port: int, host: str = "127.0.0.1") -> ThreadingHTTPServer:
    return Server((host, port), Handler)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--no-browser", action="store_true", help="do not open a browser tab")
    args = ap.parse_args(argv)
    srv = make_server(args.port)
    url = f"http://127.0.0.1:{args.port}/"
    print(f"Generative Vision Lab viewer on {url}  (Ctrl+C to stop)", flush=True)
    if not args.no_browser:
        threading.Timer(0.5, webbrowser.open, (url,)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()


if __name__ == "__main__":
    main()
