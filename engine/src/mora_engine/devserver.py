"""A localhost bridge for frontend development only.

The shipped app talks to the sidecar over stdin and stdout, with no socket
open anywhere. During frontend work the browser cannot speak stdio, so this
wraps the same Sidecar object in a loopback HTTP server. It binds to
127.0.0.1, it is never started by the packaged application, and the
frontend uses it only when Vite is running.

    python -m mora_engine.devserver --port 8756
"""

from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .sidecar import Sidecar

SIDECAR = Sidecar()


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def _send(self, payload: dict, status: int = 200) -> None:
        body = json.dumps(payload, default=str).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self) -> None:  # noqa: N802
        self._send({}, 204)

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length", 0))
        try:
            message = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            self._send({"ok": False,
                        "error": {"message": "That request was not valid "
                                             "JSON.",
                                  "action": "Send one JSON object."}}, 400)
            return
        self._send(SIDECAR.handle(message))

    def log_message(self, *args) -> None:
        return


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8756)
    args = parser.parse_args()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"MoRa engine listening on http://127.0.0.1:{args.port} "
          f"(development only, loopback only)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
