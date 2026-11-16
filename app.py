"""Alaala as a Mac app window.

Double-click Alaala.app (built by `python make_app.py`), or run `python app.py`.
Everything runs on your Mac: a tiny local server on 127.0.0.1 plus a native window.
"""
import json
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from common import DATA, ROOT

THUMBS = DATA / "thumbs"
EXAMPLES = ["lolo", "halo-halo", "lola sa beach", "pamilya", "lechon", "ube", "tita at the beach"]
UUID_RE = re.compile(r"^[A-Za-z0-9-]{1,64}$")  # no slashes or dots, so no path tricks

state = {"s": None, "error": None}
model_lock = threading.Lock()


def load_model():
    try:
        from search import Searcher

        state["s"] = Searcher()
    except Exception as e:  # shown in the window instead of crashing
        state["error"] = f"{type(e).__name__}: {e}"


def make_thumb(uuid: str, size: int = 480) -> bytes | None:
    """Thumbnails come from data/thumbs (built by make_app.py from Terminal, which
    can read the Photos library). Falls back to the original if it's readable."""
    cached = THUMBS / f"{uuid}.jpg"
    if cached.exists():
        return cached.read_bytes()
    s = state["s"]
    if s is None:
        return None
    hit = s.meta[s.meta.uuid == uuid]
    if hit.empty:
        return None
    from search import thumb_data_uri
    import base64

    uri = thumb_data_uri(hit.iloc[0].image_path, size=size)
    if not uri:
        return None
    data = base64.b64decode(uri.split(",", 1)[1])
    THUMBS.mkdir(parents=True, exist_ok=True)
    cached.write_bytes(data)
    return data


def rows(df):
    return [
        {"uuid": r.uuid, "score": round(float(r.score), 3), "date": f"{r.date:%b %d, %Y}",
         "persons": [str(p) for p in r.persons], "video": bool(r.is_video)}
        for r in df.itertuples()
    ]


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def _send(self, body: bytes, ctype: str, code: int = 200):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store" if ctype.startswith("application/json") else "max-age=3600")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code=200):
        self._send(json.dumps(obj).encode(), "application/json", code)

    def do_GET(self):
        url = urlparse(self.path)
        q = parse_qs(url.query)
        if url.path == "/":
            return self._send(PAGE.encode(), "text/html; charset=utf-8")
        if url.path == "/api/status":
            s = state["s"]
            return self._json({
                "ready": s is not None, "error": state["error"], "examples": EXAMPLES,
                "count": int(len(s.meta)) if s else 0,
                "people": sorted(s.names) if s else [],
            })
        if url.path == "/api/search":
            s = state["s"]
            if s is None:
                return self._json({"error": "Still loading"}, 503)
            query = (q.get("q") or [""])[0].strip()
            if not query:
                return self._json({"error": "Empty search"}, 400)
            compare = (q.get("compare") or ["0"])[0] == "1"
            with model_lock:
                res, parsed = s.search(query, k=48)
                base = s.search(query, k=48, baseline=True)[0] if compare else None
                verdict = s.assess(query, parsed)
            return self._json({
                "query": query, "notes": parsed.notes, "variants": parsed.variants,
                "terms": parsed.terms, "results": rows(res), **verdict,
                "baseline": rows(base) if base is not None else None,
            })
        if url.path.startswith("/thumb/"):
            uuid = url.path.rsplit("/", 1)[-1]
            data = make_thumb(uuid) if UUID_RE.match(uuid) else None
            if data is None:
                return self._send(b"", "image/jpeg", 404)
            return self._send(data, "image/jpeg")
        self._send(b"Not found", "text/plain", 404)

    def do_POST(self):
        url = urlparse(self.path)
        try:
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        except json.JSONDecodeError:
            return self._json({"error": "Bad request"}, 400)
        try:
            import tag

            if url.path == "/api/tag":
                kw = str(body.get("keyword", "")).strip()
                ids = [u for u in body.get("uuids", []) if UUID_RE.match(str(u))]
                if not kw or not ids:
                    return self._json({"error": "Pick photos and a keyword"}, 400)
                tagged, missing = tag.write_tags({kw: ids}, progress=False)
                return self._json({"tagged": tagged, "missing": missing, "keyword": kw})
            if url.path == "/api/undo":
                return self._json({"removed": tag.remove_tags()})
            if url.path == "/api/show":
                uuid = str(body.get("uuid", ""))
                if UUID_RE.match(uuid):
                    import photoscript

                    photoscript.Photo(uuid).spotlight()
                    return self._json({"ok": True})
                return self._json({"error": "Bad photo id"}, 400)
        except ValueError:
            return self._json({"error": "That photo isn't in the library Photos has open. "
                                        "Open the same library you indexed."}, 409)
        except Exception as e:
            return self._json({"error": f"Couldn't talk to Photos ({type(e).__name__}). "
                                        "Check System Settings > Privacy & Security > Automation."}, 500)
        self._json({"error": "Not found"}, 404)


def serve() -> str:
    srv = ThreadingHTTPServer(("127.0.0.1", 0), Handler)  # local only, any free port
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return f"http://127.0.0.1:{srv.server_address[1]}/"


def main():
    threading.Thread(target=load_model, daemon=True).start()
    url = serve()
    try:
        import webview
    except ImportError:
        import webbrowser

        print(f"Alaala is running at {url} (install pywebview for a native window). Ctrl+C to quit.")
        webbrowser.open(url)
        try:
            threading.Event().wait()
        except KeyboardInterrupt:
            pass
        return
    webview.create_window("Alaala", url, width=1320, height=900, min_size=(900, 620))
    webview.start()


PAGE = (ROOT / "ui.html").read_text() if (ROOT / "ui.html").exists() else "<h1>ui.html is missing</h1>"

if __name__ == "__main__":
    main()
