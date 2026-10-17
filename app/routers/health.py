"""Health, catalog, client-log, favicon and output-history endpoints."""
from __future__ import annotations

import shutil
import time

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, JSONResponse

from app.boot import BOOT_INFO
from app.config import (BASE_DIR, ERROR_LOG, LANGUAGES, MODELS, OUTPUTS_DIR,
                        VOICES)
from app.engine import _backend_info, _model_cache
from app.errlog import _log
from app.schemas import ClientError

router = APIRouter()


@router.get("/api/health")
def health():
    return {
        "ok": True,
        "backend": _backend_info,
        "loaded_models": list(_model_cache.keys()),
        "disk_outputs": len(list(OUTPUTS_DIR.glob("*.wav"))),
        "free_gb": round(shutil.disk_usage(str(BASE_DIR)).free / 1e9, 1),
        "boot": BOOT_INFO,
    }


@router.get("/api/voices")
def voices():
    return {"voices": VOICES, "languages": LANGUAGES, "models": MODELS}


@router.post("/api/log/client")
def log_client(e: ClientError):
    _log.error("CLIENT %s | %s | %s", e.url[:200], e.line, e.msg[:500])
    return {"ok": True}


@router.get("/api/logs/errors")
def logs_errors(limit: int = 50):
    """Tail of the persistent error log for future diagnosis."""
    limit = max(1, min(limit, 200))
    try:
        lines = ERROR_LOG.read_text(encoding="utf-8", errors="replace").splitlines()
    except FileNotFoundError:
        return {"lines": [], "note": "no errors logged yet"}
    return {"lines": lines[-limit:], "total": len(lines)}


@router.get("/favicon.ico")
def favicon():
    p = BASE_DIR / "assets" / "icon.ico"
    if p.exists():
        return FileResponse(str(p), media_type="image/x-icon")
    return JSONResponse({"detail": "no icon"}, status_code=404)


@router.get("/api/history")
def history(limit: int = 30):
    files = sorted(OUTPUTS_DIR.glob("*.wav"), key=lambda p: p.stat().st_mtime, reverse=True)[: max(1, min(limit, 100))]
    items = [{"filename": f.name, "url": f"/outputs/{f.name}",
              "size_kb": round(f.stat().st_size / 1024, 1),
              "created": time.strftime("%Y-%m-%d %H:%M", time.localtime(f.stat().st_mtime))} for f in files]
    return {"items": items}


@router.delete("/api/history/{filename}")
def delete_item(filename: str):
    p = (OUTPUTS_DIR / filename).resolve()
    if p.parent != OUTPUTS_DIR.resolve() or not p.exists():
        raise HTTPException(404, "Not found")
    p.unlink()
    return {"ok": True}
