"""The ethikit console: a local web UI that streams a run as it happens.

    python -m ethikit serve            # http://127.0.0.1:8765

Standard library only (http.server), so it runs anywhere ethikit's core does.
The browser builds a run config (see ethikit/session.py), POSTs it to
/api/run, and reads back newline-delimited JSON events -- run_start,
model_fit_*, tool_start / tool_done (with results) -- while the sweep is still
going; that stream drives the live flow graph and the log. The final `result`
event carries the scored profile, Pareto front, raw records and every tool's
trace (formula + intermediate tables), all of which the browser keeps.

Binds to 127.0.0.1 by default: uploading a pre-trained model means unpickling
it, which runs code, so don't expose this server to a network you don't trust.
"""
from __future__ import annotations

import io
import json
import mimetypes
import os
import tempfile
import threading
import traceback
import uuid
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict
from urllib.parse import parse_qs, unquote, urlparse

import pandas as pd

from . import checklists as checklists_mod
from . import session
from .benchmark.models import load_fitted_model
from .benchmark.runner import jsonable
from .report import model_card_html, report_html

WEB = Path(__file__).resolve().parent / "web"
# The docs live beside the package in a source checkout; served at /docs/ so the whole
# project (console + docs) works offline with no hosting. Absent in a plain wheel install.
DOCS = Path(__file__).resolve().parent.parent / "docs"
MAX_UPLOAD = 200 * 1024 * 1024

_LOCK = threading.Lock()
UPLOADS: Dict[str, pd.DataFrame] = {}
MODELS: Dict[str, Any] = {}
RUNS: Dict[str, dict] = {}


def describe_frame(df: pd.DataFrame) -> dict:
    cols = []
    for c in df.columns:
        s = df[c]
        cols.append({"name": str(c), "dtype": str(s.dtype), "nunique": int(s.nunique()),
                     "missing": int(s.isna().sum()),
                     "sample": [str(v) for v in s.dropna().unique()[:5]]})
    # heuristics for a sensible default target / sensitive pick
    lowered = {str(c).lower(): str(c) for c in df.columns}
    target = next((lowered[k] for k in ("target", "label", "class", "y", "outcome", "income")
                   if k in lowered), str(df.columns[-1]))
    sensitive = next((lowered[k] for k in ("sex", "gender", "race", "age", "group", "ethnicity")
                      if k in lowered), None)
    return {"n_rows": int(len(df)), "columns": cols,
            "suggest": {"target": target, "sensitive": sensitive}}


class Handler(BaseHTTPRequestHandler):
    server_version = "ethikit-console"

    def log_message(self, fmt, *args):  # keep the terminal for ethikit's own output
        pass

    # ---- helpers -----------------------------------------------------------
    def _send(self, code: int, body: bytes, ctype: str = "application/json",
              extra: Dict[str, str] = None):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj: Any, code: int = 200):
        self._send(code, json.dumps(jsonable(obj)).encode())

    def _error(self, exc: Exception, code: int = 400):
        self._json({"error": f"{type(exc).__name__}: {exc}"}, code)

    def _body(self) -> bytes:
        n = int(self.headers.get("Content-Length") or 0)
        if n > MAX_UPLOAD:
            raise ValueError(f"upload too large ({n} bytes)")
        return self.rfile.read(n)

    # ---- routes ------------------------------------------------------------
    def do_GET(self):
        url = urlparse(self.path)
        path = unquote(url.path)
        try:
            if path in ("/", "/index.html"):
                return self._static("index.html")
            if path == "/api/meta":
                return self._json({**session.catalogue(), "live": True})
            if path.startswith("/api/card/"):
                _, _, _, run_id, model = path.split("/", 4)
                return self._card(run_id, model)
            if path.startswith("/api/report/"):
                run = RUNS[path.rsplit("/", 1)[1]]
                s = run["summary"]
                html = report_html(s["dataset"], run["objective"], s["pareto"])
                return self._send(200, html.encode(), "text/html; charset=utf-8")
            if path.startswith("/api/checklist-md/"):
                q = parse_qs(url.query)
                run = RUNS.get(path.rsplit("/", 1)[1]) or {}
                cl = run.get("checklist") or checklists_mod.blank_answers(q.get("id", ["deon"])[0])
                md = checklists_mod.to_markdown(cl["checklist"], cl.get("answers"))
                return self._send(200, md.encode(), "text/markdown; charset=utf-8",
                                  {"Content-Disposition": f'attachment; filename="{cl["checklist"]}.md"'})
            if path == "/docs" or path == "/docs/":
                return self._static("index.html", DOCS)
            if path.startswith("/docs/"):
                rel = path[len("/docs/"):]
                return self._static(rel + "index.html" if rel.endswith("/") else rel, DOCS)
            return self._static(path.lstrip("/"))
        except KeyError as exc:
            return self._error(exc, 404)
        except Exception as exc:
            return self._error(exc, 500)

    def do_POST(self):
        url = urlparse(self.path)
        try:
            if url.path == "/api/upload":
                df = pd.read_csv(io.BytesIO(self._body()))
                uid = uuid.uuid4().hex[:10]
                with _LOCK:
                    UPLOADS[uid] = df
                name = parse_qs(url.query).get("name", ["uploaded.csv"])[0]
                return self._json({"upload_id": uid, "name": os.path.splitext(name)[0],
                                   **describe_frame(df)})
            if url.path == "/api/upload-model":
                data = self._body()
                with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as fh:
                    fh.write(data)
                try:
                    model = load_fitted_model(fh.name)
                finally:
                    os.unlink(fh.name)
                if not hasattr(model, "predict"):
                    raise TypeError(f"{type(model).__name__} has no predict() method")
                mid = uuid.uuid4().hex[:10]
                with _LOCK:
                    MODELS[mid] = model
                return self._json({"upload_model_id": mid, "class": type(model).__name__,
                                   "n_features_in": getattr(model, "n_features_in_", None)})
            if url.path == "/api/run":
                return self._run(json.loads(self._body() or b"{}"))
            if url.path == "/api/domain-run":
                return self._domain_run(json.loads(self._body() or b"{}"))
            if url.path.startswith("/api/checklist/"):
                run_id = url.path.rsplit("/", 1)[1]
                data = json.loads(self._body() or b"{}")
                if data.get("checklist") not in checklists_mod.CHECKLISTS:
                    raise KeyError(f"unknown checklist {data.get('checklist')!r}")
                RUNS[run_id]["checklist"] = data
                summary = checklists_mod.principle_summary(data["checklist"], data.get("answers", {}))
                return self._json({"ok": True, "summary": summary})
            return self._json({"error": "not found"}, 404)
        except Exception as exc:
            return self._error(exc)

    def _static(self, rel: str, root: Path = WEB):
        target = (root / rel).resolve()
        if root.resolve() not in target.parents or not target.is_file():
            return self._json({"error": f"not found: {rel}"}, 404)
        ctype = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
        if ctype.startswith("text/") or ctype.endswith("javascript"):
            ctype += "; charset=utf-8"
        self._send(200, target.read_bytes(), ctype)

    def _card(self, run_id: str, model: str):
        run = RUNS[run_id]
        s = run["summary"]
        html = model_card_html(model, s["dataset"], run["raw"], run["objective"], s["pareto"],
                               sensitive_name=s["sensitive"],
                               hyperparameters=run["prepared"].hyperparameters.get(model),
                               origin=run["prepared"].origins.get(model),
                               checklist=run.get("checklist"))
        self._send(200, html.encode(), "text/html; charset=utf-8")

    def _domain_run(self, cfg: dict):
        """Stream a reference-suite run: dataset_start / dataset_done / dataset_skip events,
        then a final `domain_result` (family evidence + shortlist)."""
        from .domains import get_domains, run_domain, shortlist
        from .domains.runner import DomainError
        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "close")
        self.end_headers()

        def emit(event: dict):
            self.wfile.write((json.dumps(jsonable(event)) + "\n").encode())
            self.wfile.flush()
        try:
            states = tuple(s for s in (cfg.get("states") or "").replace(" ", "").split(",") if s)
            domains = get_domains(*([states] if states else []))
            if cfg.get("domain") not in domains:
                raise DomainError(f"unknown domain {cfg.get('domain')!r}; choose from {', '.join(domains)}")
            cv = cfg.get("cv")
            res = run_domain(domains[cfg["domain"]], models=cfg.get("models") or None,
                             tools=cfg.get("tools") or None,
                             max_datasets=int(cfg["max_datasets"]) if cfg.get("max_datasets") else None,
                             max_rows=int(cfg.get("max_rows") or 5000), seed=int(cfg.get("seed") or 0),
                             cv=(int(cv[0]), int(cv[1])) if cv else None, on_event=emit)
            emit({"type": "domain_result", **res.to_dict(), "shortlist": shortlist(res, cfg.get("weights"))})
        except (BrokenPipeError, ConnectionResetError):
            return
        except Exception as exc:
            try:
                emit({"type": "error", "error": f"{type(exc).__name__}: {exc}"})
            except (BrokenPipeError, ConnectionResetError):
                return
        self.close_connection = True

    def _run(self, config: dict):
        """Stream NDJSON events while the sweep runs, then a final `result`."""
        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson")
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Accel-Buffering", "no")
        self.send_header("Connection", "close")
        self.end_headers()

        def emit(event: dict):
            self.wfile.write((json.dumps(jsonable(event)) + "\n").encode())
            self.wfile.flush()

        run_id = uuid.uuid4().hex[:10]
        try:
            emit({"type": "prepare", "run_id": run_id})
            prepared = session.prepare(config, UPLOADS, MODELS)
            for note in prepared.notes:
                emit({"type": "note", "message": note})
            out = session.execute(prepared, on_event=emit)
            RUNS[run_id] = {"prepared": prepared, "raw": out["raw"], "objective": out["objective"],
                            "summary": out["summary"], "checklist": prepared.checklist,
                            "config": config}
            emit({"type": "result", "run_id": run_id, **out["summary"],
                  "hyperparameters": {m: {k: repr(v) for k, v in p.items()}
                                      for m, p in prepared.hyperparameters.items()}})
        except (BrokenPipeError, ConnectionResetError):
            return
        except Exception as exc:
            try:
                emit({"type": "error", "error": f"{type(exc).__name__}: {exc}",
                      "trace": traceback.format_exc(limit=3)})
            except (BrokenPipeError, ConnectionResetError):
                return
        self.close_connection = True


def serve(host: str = "127.0.0.1", port: int = 8765, open_browser: bool = False) -> None:
    httpd = ThreadingHTTPServer((host, port), Handler)
    url = f"http://{host}:{port}/"
    print(f"ethikit console running at {url}  (Ctrl+C to stop)")
    if open_browser:
        import webbrowser
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped.")
    finally:
        httpd.server_close()
