"""Serve the paper-trading dashboard.

    python -m tradebot.run.dashboard                 # http://127.0.0.1:8060
    python -m tradebot.run.dashboard --host 0.0.0.0  # for Docker / a server

Read-only: it renders what is in state/paper/ and never writes a byte of run
state. Stdlib only — no web framework, no CDN — so it runs anywhere the
harness runs, including an offline box.
"""

from __future__ import annotations

import argparse
import json
from functools import partial
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources

from ..config import Config
from ..paper.report import all_runs
from ._common import load_cfg


def _page() -> bytes:
    return (resources.files("tradebot.paper") / "dashboard.html").read_bytes()


class Handler(BaseHTTPRequestHandler):
    cfg: Config = None  # injected via partial

    def do_GET(self) -> None:  # noqa: N802 (stdlib naming)
        if self.path == "/" or self.path == "/index.html":
            body = _page()
            self._reply(200, "text/html; charset=utf-8", body)
        elif self.path == "/api/data":
            body = json.dumps(all_runs(self.cfg.paper.state_dir)).encode()
            self._reply(200, "application/json", body)
        else:
            self._reply(404, "text/plain", b"not found")

    def _reply(self, code: int, ctype: str, body: bytes) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt: str, *args) -> None:
        pass  # one request every 30s per viewer; silence beats noise


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="config/default.yaml")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8060)
    args = p.parse_args()
    cfg = load_cfg(args)

    handler = partial(_Bound, cfg)
    server = ThreadingHTTPServer((args.host, args.port), handler)
    print(f"dashboard: http://{args.host}:{args.port}  (reading {cfg.paper.state_dir}/paper/)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


class _Bound(Handler):
    def __init__(self, cfg: Config, *a, **kw):
        self.cfg = cfg
        super().__init__(*a, **kw)


if __name__ == "__main__":
    main()
