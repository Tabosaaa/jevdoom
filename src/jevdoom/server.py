"""The full UI: game + Jev + controls on one page (ui/index.html). A local server
using only the standard library: the game runs on a thread (session.py) and every
event — screen frame, decision, objectives, end of episode — goes to the browser over
Server-Sent Events. The browser picks the mode, scenario, map, difficulty and pace,
sends keyboard/mouse in human mode and reopens games recorded in runs/.

    PYTHONPATH=src python -m jevdoom.server        # opens http://127.0.0.1:8765
"""

import argparse
import sys
import json
import os
import queue
import re
import threading
import webbrowser
from dataclasses import asdict
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from jevdoom import engine, session, wad

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
UI_FILE = os.path.join(ROOT, "ui", "index.html")
RUNS_DIR = os.path.join(ROOT, session.RUNS_DIR)
SCENARIOS = {
    "freedoom2": "Freedoom 2",
    "freedoom1": "Freedoom 1",
    "defend_the_center": "Arena · center",
    "defend_the_line": "Arena · line",
}
MAX_BODY = 64 * 1024
QUEUE_SIZE = 256
RUN_NAME = re.compile(r"^\d{8}-\d{6}(-\d{3})?\.jsonl$")


class Hub:
    """Fans events out to the connected browsers. Frames are disposable: a client
    that falls behind loses frames, never decisions."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._clients: list[queue.Queue] = []
        self._replay: dict[str, dict] = {}  # latest event of each kind, for late joiners

    def subscribe(self) -> queue.Queue:
        q: queue.Queue = queue.Queue(QUEUE_SIZE)
        with self._lock:
            self._clients.append(q)
            for kind in ("status", "start", "objectives", "results"):
                if kind in self._replay:
                    q.put_nowait((kind, self._replay[kind]))
        return q

    def unsubscribe(self, q: queue.Queue) -> None:
        with self._lock:
            if q in self._clients:
                self._clients.remove(q)

    def publish(self, kind: str, data: dict) -> None:
        with self._lock:
            if kind in ("status", "start", "objectives", "results"):
                self._replay[kind] = data
            for q in list(self._clients):
                try:
                    q.put_nowait((kind, data))
                except queue.Full:
                    if kind != "frame":  # decisions are not lost: make room by dropping the oldest
                        try:
                            q.get_nowait()
                            q.put_nowait((kind, data))
                        except (queue.Empty, queue.Full):
                            pass


class Runner:
    """One game at a time, on a thread; starting another stops the previous one."""

    def __init__(self, hub: Hub) -> None:
        self.hub = hub
        self.human = session.HumanInput()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._results: list[dict] = []

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self, cfg: session.SessionConfig) -> None:
        cfg = cfg.validated()
        self.stop()
        self._stop = threading.Event()
        self._results = []
        self._thread = threading.Thread(target=self._run, args=(cfg, self._stop), daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=10)

    def _run(self, cfg: session.SessionConfig, stop: threading.Event) -> None:
        self.hub.publish("status", {"running": True, "config": asdict(cfg)})

        def emit(kind: str, data: dict) -> None:
            if kind == "episode":
                self._results.append(data)
                self.hub.publish("results", {"results": list(self._results)})
            self.hub.publish(kind, data)

        try:
            session.Session(cfg, emit, stop.is_set, self.human).run()
        except Exception as error:  # the UI shows it; the server stays up
            self.hub.publish("error", {"message": f"{type(error).__name__}: {error}"})
        finally:
            self.hub.publish("status", {"running": False, "config": asdict(cfg)})


def options() -> dict:
    scenarios = []
    for name, label in SCENARIOS.items():
        if not os.path.exists(os.path.join(engine.SCENARIOS_DIR, f"{name}.cfg")):
            continue
        iwad = engine.iwad_path(name)
        scenarios.append({"id": name, "label": label, "maps": wad.map_names(iwad) if iwad else []})
    return {"scenarios": scenarios, "modes": list(session.MODES),
            "defaults": asdict(session.SessionConfig(frames=True, realtime=True))}


def config_from(body: dict) -> session.SessionConfig:
    """Only the known fields, with the right types; range checks belong to SessionConfig."""
    defaults = session.SessionConfig()
    try:
        return session.SessionConfig(
            scenario=str(body.get("scenario", defaults.scenario)),
            doom_map=(str(body["doom_map"]) if body.get("doom_map") else None),
            mode=str(body.get("mode", defaults.mode)),
            tics=int(body.get("tics", defaults.tics)),
            episodes=int(body.get("episodes", 1)),
            max_steps=int(body["max_steps"]) if body.get("max_steps") else None,
            skill=int(body.get("skill", defaults.skill)),
            speed=float(body.get("speed", 1.0)),
            realtime=bool(body.get("realtime", True)),
            frames=True,
        )
    except (TypeError, ValueError) as error:
        raise ValueError(f"invalid configuration: {error}") from error


def make_handler(hub: Hub, runner: Runner):
    class Handler(BaseHTTPRequestHandler):
        server_version = "jevDoom"

        def log_message(self, fmt, *args) -> None:  # quiet: the stream is continuous
            pass

        def _send(self, status: int, payload, content_type: str = "application/json") -> None:
            body = payload if isinstance(payload, bytes) else json.dumps(payload, ensure_ascii=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _body(self) -> dict:
            length = int(self.headers.get("Content-Length") or 0)
            if length > MAX_BODY:
                raise ValueError("request body too large")
            data = json.loads(self.rfile.read(length) or b"{}")
            if not isinstance(data, dict):
                raise ValueError("expected a JSON object")
            return data

        def do_GET(self) -> None:
            path = self.path.split("?", 1)[0]
            if path in ("/", "/index.html"):
                with open(UI_FILE, "rb") as f:
                    self._send(HTTPStatus.OK, f.read(), "text/html; charset=utf-8")
            elif path == "/api/options":
                self._send(HTTPStatus.OK, options())
            elif path == "/api/map":
                query = dict(part.split("=", 1) for part in self.path.partition("?")[2].split("&") if "=" in part)
                scenario, name = query.get("scenario", ""), query.get("map", "")
                if scenario not in SCENARIOS or not re.fullmatch(r"(MAP\d\d|E\dM\d)", name.upper()):
                    self._send(HTTPStatus.BAD_REQUEST, {"error": "invalid scenario or map"})
                    return
                lv = session.load_level(scenario, name)
                self._send(HTTPStatus.OK, {"map": session.minimap(lv) if lv is not None else None})
            elif path == "/api/runs":
                names = sorted((n for n in os.listdir(RUNS_DIR) if RUN_NAME.match(n)), reverse=True) \
                    if os.path.isdir(RUNS_DIR) else []
                self._send(HTTPStatus.OK, {"runs": names[:200]})
            elif path.startswith("/api/runs/"):
                name = path.rsplit("/", 1)[-1]
                file = os.path.join(RUNS_DIR, name)
                if not RUN_NAME.match(name) or not os.path.isfile(file):
                    self._send(HTTPStatus.NOT_FOUND, {"error": "game not found"})
                    return
                with open(file, "rb") as f:
                    self._send(HTTPStatus.OK, f.read(), "application/x-ndjson")
            elif path == "/api/stream":
                self._stream()
            else:
                self._send(HTTPStatus.NOT_FOUND, {"error": "not found"})

        def do_POST(self) -> None:
            try:
                body = self._body()
                if self.path == "/api/start":
                    runner.start(config_from(body))
                    self._send(HTTPStatus.OK, {"ok": True})
                elif self.path == "/api/stop":
                    runner.stop()
                    self._send(HTTPStatus.OK, {"ok": True})
                elif self.path == "/api/input":
                    keys = body.get("keys", [])
                    if not isinstance(keys, list) or len(keys) > 16:
                        raise ValueError("invalid keys")
                    runner.human.set(keys[:16], max(-500.0, min(500.0, float(body.get("dx", 0.0)))))
                    self._send(HTTPStatus.OK, {"ok": True})
                else:
                    self._send(HTTPStatus.NOT_FOUND, {"error": "not found"})
            except (ValueError, json.JSONDecodeError) as error:
                self._send(HTTPStatus.BAD_REQUEST, {"error": str(error)})

        def _stream(self) -> None:
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Connection", "keep-alive")
            self.end_headers()
            q = hub.subscribe()
            try:
                while True:
                    try:
                        kind, data = q.get(timeout=15)
                        chunk = f"event: {kind}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"
                    except queue.Empty:
                        chunk = ": ping\n\n"  # keeps the connection alive
                    self.wfile.write(chunk.encode())
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass
            finally:
                hub.unsubscribe(q)

    return Handler


class Server(ThreadingHTTPServer):
    daemon_threads = True

    def handle_error(self, request, client_address) -> None:
        """A browser closing the tab mid-stream is not a server error."""
        if not isinstance(sys.exc_info()[1], (ConnectionResetError, BrokenPipeError)):
            super().handle_error(request, client_address)


def main() -> None:
    p = argparse.ArgumentParser(description="jevDoom — full UI")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--no-browser", action="store_true")
    args = p.parse_args()
    os.chdir(ROOT)  # session writes runs/ relative to the project root
    hub = Hub()
    runner = Runner(hub)
    httpd = Server(("127.0.0.1", args.port), make_handler(hub, runner))
    url = f"http://127.0.0.1:{args.port}"
    print(f"jevDoom UI at {url}  (Ctrl+C to quit)")
    if not args.no_browser:
        threading.Timer(0.8, webbrowser.open, args=(url,)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        runner.stop()
        httpd.server_close()


if __name__ == "__main__":
    main()
