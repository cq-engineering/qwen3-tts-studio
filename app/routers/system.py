"""System endpoints: model unload, process shutdown, and clean restart."""
from __future__ import annotations

import time
from typing import Optional

from fastapi import APIRouter, HTTPException

from app.config import BASE_DIR, MODELS
from app.errlog import _log
from app.jobs import _jobs, _jobs_lock, _unload_models
from app.writers import _ollama_unload_all

router = APIRouter()


@router.post("/api/models/unload")
def models_unload(model: Optional[str] = None):
    """Purge loaded weights from memory. Safe any time: an in-flight inference
    holds its own reference, so nothing crashes — RAM frees when it returns."""
    if model and model not in MODELS:
        raise HTTPException(400, f"Unknown model '{model}'.")
    return _unload_models([model] if model else None)


@router.post("/api/server/shutdown")
def server_shutdown():
    """Full stop from the UI Exit button: cancel jobs, unload TTS weights
    (reports freed MB), queue Ollama-model unloads, then terminate this
    process. UI shows the offline banner; relaunch from the desktop icon
    or ./start.sh."""
    import threading as _th
    try:
        with _jobs_lock:
            for job in _jobs.values():
                if job["status"] in ("queued", "running"):
                    job["cancel_requested"] = True
    except Exception:
        pass
    freed = _unload_models()

    def _do():
        try:
            _ollama_unload_all()
        except Exception:
            pass
        time.sleep(1.0)  # let the 200 response flush before we vanish
        import os as _os
        _os._exit(0)
    _th.Thread(target=_do, daemon=True).start()
    return {"stopping": True,
            "tts_unloaded": freed["unloaded"], "tts_freed_mb": freed["freed_mb"],
            "note": "Writer models unloading, process terminating. Relaunch via desktop icon."}


@router.post("/api/server/restart")
def server_restart():
    """Clean restart on the SAME port, triggered from the UI. Responds 200
    first, then exec's the process image into a fresh server: the old
    listener closes on exec (CLOEXEC) and the new process rebinds :8000 with
    SO_REUSEADDR — no port conflicts, no stale state. In-flight jobs are
    marked cancelled first (fresh boot drops workers, cache, and queues)."""
    import threading as _th

    def _do():
        try:
            with _jobs_lock:
                for job in _jobs.values():
                    if job["status"] in ("queued", "running"):
                        job["cancel_requested"] = True
        except Exception:
            pass
        time.sleep(1.0)  # let the 200 response flush before we vanish
        try:
            import os as _os
            import sys as _sys
            host = _os.environ.get("BIND_HOST", "0.0.0.0")
            port = str(_os.environ.get("PORT", "8000"))
            _os.execv(_sys.executable, [_sys.executable, "-m", "uvicorn",
                                        "server:app", "--app-dir", str(BASE_DIR),
                                        "--host", host, "--port", port])
        except Exception:
            _log.exception("server restart exec failed")
            import os as _os2
            _os2._exit(0)
    _th.Thread(target=_do, daemon=True).start()
    return {"restarting": True, "port": 8000,
            "note": "Exec'ing into a fresh server on the same port. Poll GET /api/health until ok:true."}
