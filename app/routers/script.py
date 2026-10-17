"""Script-writer endpoints: writer catalog, cloud key save/forget, and the
async script-draft job."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.calibration import _calib_lock
from app.config import (SCENARIOS, SCRIPT_CLOUD, SCRIPT_LENGTHS, SCRIPT_MODELS,
                        SCRIPT_STEPS)
from app.jobs import _job_enqueue
from app.schemas import KeyRequest
from app.writers import (_cloud_key, _cloud_verify, _keys, _keys_save,
                         _ollama_tags, _ollama_up)

router = APIRouter()


@router.post("/api/script/keys")
def script_keys(req: KeyRequest):
    if req.provider not in SCRIPT_CLOUD:
        raise HTTPException(400, "Unknown provider")
    key = (req.key or "").strip()
    if len(key) < 8:
        raise HTTPException(400, "That key looks too short — paste the full key.")
    ok, why = _cloud_verify(req.provider, key)
    if not ok:
        raise HTTPException(400, f"Key rejected by {req.provider}: {why}")
    with _calib_lock:
        _keys[req.provider] = key
    _keys_save()
    return {"saved": True, "provider": req.provider}


@router.delete("/api/script/keys/{provider}")
def script_keys_forget(provider: str):
    with _calib_lock:
        _keys.pop(provider, None)
    _keys_save()
    return {"saved": False, "provider": provider}


@router.get("/api/script/models")
def script_models():
    """Writer models (availability probed live), scenarios, and lengths."""
    up = _ollama_up()
    tags = _ollama_tags() if up else {}
    try:
        import psutil
        avail = round(psutil.virtual_memory().available / 1e9, 1)
    except Exception:
        avail = None
    def _size_gb(tag):
        return round(tags[tag].get("size", 0) / 1e9, 1) if tag in tags else None
    return {
        "ollama_up": up,
        "free_gb_ram": avail,
        "llm_models": [{"id": k, "tag": m["tag"], "label": m["label"], "note": m["note"],
                        "available": m["tag"] in tags,
                        "size_gb": _size_gb(m["tag"]),
                        "fits_ram": (_size_gb(m["tag"]) < (avail or 0) * 0.85)
                        if (_size_gb(m["tag"]) is not None and avail) else None}
                       for k, m in SCRIPT_MODELS.items()],
        "scenarios": [{"id": k, **v} for k, v in SCENARIOS.items()],
        "lengths": [{"id": k, "label": v["label"]} for k, v in SCRIPT_LENGTHS.items()],
        "providers": [{"id": k, "label": v["label"], "note": v["note"], "key_url": v["key_url"],
                       "default": v["default"], "models": v["models"],
                       "configured": bool(_cloud_key(k))} for k, v in SCRIPT_CLOUD.items()],
    }


@router.post("/api/jobs/script", status_code=202)
def job_script(payload: dict):
    topic = (payload.get("topic") or "").strip()
    if not topic:
        raise HTTPException(400, "Topic is required")
    if payload.get("scenario") not in SCENARIOS:
        raise HTTPException(400, "Unknown scenario")
    provider = payload.get("provider") or "local"
    if provider == "local":
        if payload.get("model", "best") not in SCRIPT_MODELS:
            raise HTTPException(400, "Unknown writer model")
        model = payload.get("model", "best")
    elif provider in SCRIPT_CLOUD:
        model = payload.get("model") or SCRIPT_CLOUD[provider]["default"]
    else:
        raise HTTPException(400, "Unknown provider")
    if payload.get("length", "standard") not in SCRIPT_LENGTHS:
        raise HTTPException(400, "Unknown length")
    p = {"topic": topic, "scenario": payload["scenario"],
         "tone": (payload.get("tone") or "").strip(),
         "length": payload.get("length", "standard"),
         "model": model, "provider": provider}
    return {"job_id": _job_enqueue("script", p, steps=SCRIPT_STEPS), "status": "queued"}
