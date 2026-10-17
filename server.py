"""
Qwen3-TTS Studio — FastAPI backend
Wraps official `qwen-tts` package (Qwen/Qwen3-TTS-12Hz-*) with a clean REST API
+ serves the sleek dark frontend in ./static

Two ways to synthesize:
  * POST /api/synthesize/*  — blocking, returns WAV when done (for scripts)
  * POST /api/jobs/*        — async, returns job_id immediately; poll
                              GET /api/jobs/{id} for live phase/progress.
                              The UI uses jobs so the status bar is truthful.
Jobs run on a SINGLE background worker so concurrent generations can never
OOM the box (1.7B bf16 ≈ 3.5 GB resident).
"""
from __future__ import annotations
import time
import json as _json  # boot-times file (lifespan) + misc server-side encoding
BOOT_T0 = time.time()  # process boot start; the lifespan handler completes the measurement
import uuid
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# filesystem + static config now live in app.config (single source of truth)
from app.config import (BASE_DIR, STATIC_DIR, OUTPUTS_DIR, UPLOADS_DIR, ERROR_LOG,
                        CALIB_PATH, KEYS_PATH, VOICES, LANGUAGES, MODELS,
                        PHASE_CAP, PHASE_PROGRESS, JOB_STEPS, SCRIPT_STEPS,
                        SCRIPT_MODELS, SCRIPT_LENGTHS, SCENARIOS, SCRIPT_CLOUD)

from app.errlog import _log  # rotating errors.log handler

# voices / LANGUAGES / MODELS -> app.config (imported above)

from app.engine import (_model_cache, _backend_info, _detect_backend, _load_model,
                        _save_wav, _canon_speaker, _gen_kwargs, _apply_seed,
                        _synth_custom_voice, _synth_voice_design, _synth_voice_clone)
from app.calibration import (_calib, _calib_lock, _calib_load, _calib_save,
                             _CALIB_DEFAULT_SPC, _job_model_key, _job_text_len,
                             _predict_total)

from app.jobs import (_job_enqueue, _job_public, _job_set, _jobs, _jobs_lock,
                     _pending, _check_cancel, _unload_models, _finalize_cancel,
                     _split_sentences, _parse_dialogue, _stitch_parts)


from app.writers import (_keys, _keys_save, _cloud_key, _cloud_verify,
                           _draft_cloud, _ollama_up, _ollama_tags,
                           _ollama_unload_all, _draft_script)
from app.errors import _Cancelled  # raised mid-draft / mid-job on cancel

# ------------------------------------------------------------- FastAPI ---
from contextlib import asynccontextmanager
BOOT_FILE = BASE_DIR / "boot_times.json"
BOOT_INFO: Dict[str, Any] = {"actual_sec": None, "average_sec": None,
                             "estimated_sec": None, "samples": 0}

@asynccontextmanager
async def lifespan(app):
    actual = round(time.time() - BOOT_T0, 1)
    try:
        samples = _json.loads(BOOT_FILE.read_text()) if BOOT_FILE.exists() else []
        samples = [float(x) for x in samples if isinstance(x, (int, float))][-19:] + [actual]
        BOOT_FILE.write_text(_json.dumps(samples))
    except Exception:
        samples = [actual]
    import statistics
    BOOT_INFO.update({"actual_sec": actual,
                      "average_sec": round(statistics.mean(samples), 1),
                      "estimated_sec": round(statistics.mean(samples[-3:]), 1),
                      "samples": len(samples)})
    _log.info("boot complete in %ss (avg %s over %d runs)",
              actual, BOOT_INFO["average_sec"], len(samples))
    yield

app = FastAPI(title="Qwen3-TTS Studio", version="1.1.0", lifespan=lifespan)
_log.info("server boot (backend=%s)", _backend_info)

class KeyRequest(BaseModel):
    provider: str
    key: str = Field(max_length=500)

@app.post("/api/script/keys")
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

@app.delete("/api/script/keys/{provider}")
def script_keys_forget(provider: str):
    with _calib_lock:
        _keys.pop(provider, None)
    _keys_save()
    return {"saved": False, "provider": provider}

class CustomVoiceRequest(BaseModel):
    text: str = Field(min_length=1, max_length=8000)
    language: str = "English"
    speaker: str = "Aiden"
    instruct: Optional[str] = ""
    model: str = "custom-1.7B"
    max_new_tokens: int = Field(2048, ge=128, le=4096)
    temperature: Optional[float] = Field(None, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(None, ge=0.0, le=1.0)
    top_k: Optional[int] = Field(None, ge=1, le=500)
    seed: Optional[int] = None

class VoiceDesignRequest(BaseModel):
    text: str = Field(min_length=1, max_length=8000)
    language: str = "English"
    instruct: str = Field(min_length=3, max_length=2000)
    max_new_tokens: int = Field(2048, ge=128, le=4096)
    temperature: Optional[float] = Field(None, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(None, ge=0.0, le=1.0)
    top_k: Optional[int] = Field(None, ge=1, le=500)
    seed: Optional[int] = None

@app.get("/api/health")
def health():
    import shutil
    return {
        "ok": True,
        "backend": _backend_info,
        "loaded_models": list(_model_cache.keys()),
        "disk_outputs": len(list(OUTPUTS_DIR.glob("*.wav"))),
        "free_gb": round(shutil.disk_usage(str(BASE_DIR)).free / 1e9, 1),
        "boot": BOOT_INFO,
    }

@app.get("/api/voices")
def voices():
    return {"voices": VOICES, "languages": LANGUAGES, "models": MODELS}

# ------------------------- blocking endpoints (for scripts) ---
@app.post("/api/synthesize/custom-voice")
def synth_custom(req: CustomVoiceRequest):
    try:
        wavs, sr, info = _synth_custom_voice(req.model_dump())
        fname = _save_wav(wavs[0], int(sr), info["speaker"].lower())
        return {"ok": True, "file": f"/outputs/{fname}", "filename": fname,
                "sample_rate": int(sr), "duration_sec": round(len(wavs[0]) / float(sr), 2), **info}
    except HTTPException:
        raise
    except Exception as e:
        _log.exception("synth custom-voice failed (speaker=%s)",
                       getattr(req, "speaker", "?"))
        raise HTTPException(500, f"{type(e).__name__}: {e}")

@app.post("/api/synthesize/voice-design")
def synth_design(req: VoiceDesignRequest):
    try:
        wavs, sr, info = _synth_voice_design(req.model_dump())
        fname = _save_wav(wavs[0], int(sr), "designed")
        return {"ok": True, "file": f"/outputs/{fname}", "filename": fname,
                "sample_rate": int(sr), "duration_sec": round(len(wavs[0]) / float(sr), 2), **info}
    except HTTPException:
        raise
    except Exception as e:
        _log.exception("synth voice-design failed")
        raise HTTPException(500, f"{type(e).__name__}: {e}")

@app.post("/api/synthesize/voice-clone")
async def synth_clone(
    target_text: str = Form(...),
    language: str = Form("English"),
    ref_text: str = Form(""),
    model: str = Form("base-1.7B"),
    x_vector_only: bool = Form(False),
    max_new_tokens: int = Form(2048, ge=128, le=4096),
    ref_audio: UploadFile = File(...),
):
    raw = await ref_audio.read()
    if len(raw) > 25 * 1024 * 1024:
        raise HTTPException(400, "Reference audio must be under 25 MB")
    upath = UPLOADS_DIR / f"ref_{uuid.uuid4().hex[:8]}_{Path(ref_audio.filename or 'clip.wav').name}"
    upath.write_bytes(raw)
    try:
        wavs, sr, info = _synth_voice_clone({
            "target_text": target_text, "language": language, "ref_text": ref_text,
            "model": model, "x_vector_only": x_vector_only,
            "max_new_tokens": max_new_tokens, "ref_audio_path": str(upath)})
        fname = _save_wav(wavs[0], int(sr), "clone")
        return {"ok": True, "file": f"/outputs/{fname}", "filename": fname,
                "sample_rate": int(sr), "duration_sec": round(len(wavs[0]) / float(sr), 2), **info}
    except HTTPException:
        raise
    except Exception as e:
        _log.exception("synth voice-clone failed")
        raise HTTPException(500, f"{type(e).__name__}: {e}")

# ------------------------- async jobs (the UI uses these) ---
@app.post("/api/jobs/custom-voice", status_code=202)
def job_custom(req: CustomVoiceRequest):
    return {"job_id": _job_enqueue("custom-voice", req.model_dump()), "status": "queued"}

@app.post("/api/jobs/chunked", status_code=202)
def job_chunked(req: CustomVoiceRequest):
    """Stream mode: sentence-split synthesis with per-part playback + real
    parts-done progress. Single-sentence input falls back to a normal job."""
    if len(_split_sentences(req.text)) <= 1:
        return {"job_id": _job_enqueue("custom-voice", req.model_dump()), "status": "queued"}
    return {"job_id": _job_enqueue("chunked", req.model_dump()), "status": "queued"}

@app.post("/api/jobs/voice-design", status_code=202)
def job_design(req: VoiceDesignRequest):
    return {"job_id": _job_enqueue("voice-design", req.model_dump()), "status": "queued"}

@app.post("/api/jobs/voice-clone", status_code=202)
async def job_clone(
    target_text: str = Form(...),
    language: str = Form("English"),
    ref_text: str = Form(""),
    model: str = Form("base-1.7B"),
    x_vector_only: bool = Form(False),
    max_new_tokens: int = Form(2048, ge=128, le=4096),
    ref_audio: UploadFile = File(...),
):
    if not (target_text or "").strip():
        raise HTTPException(400, "Target text is required")
    raw = await ref_audio.read()
    if len(raw) > 25 * 1024 * 1024:
        raise HTTPException(400, "Reference audio must be under 25 MB")
    upath = UPLOADS_DIR / f"ref_{uuid.uuid4().hex[:8]}_{Path(ref_audio.filename or 'clip.wav').name}"
    upath.write_bytes(raw)
    return {"job_id": _job_enqueue("voice-clone", {
        "target_text": target_text, "language": language, "ref_text": ref_text,
        "model": model, "x_vector_only": x_vector_only,
        "max_new_tokens": max_new_tokens, "ref_audio_path": str(upath)}), "status": "queued"}

@app.get("/api/script/models")
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

@app.post("/api/jobs/script", status_code=202)
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

class DialogueRequest(BaseModel):
    text: str = Field(min_length=1, max_length=8000)
    speaker_a: str = "Aiden"
    speaker_b: str = "Ryan"
    language: str = "English"
    model: str = "custom-1.7B"
    instruct: Optional[str] = ""
    max_new_tokens: int = Field(2048, ge=128, le=4096)

@app.post("/api/jobs/dialogue", status_code=202)
def job_dialogue(req: DialogueRequest):
    a = _canon_speaker(req.speaker_a)
    b = _canon_speaker(req.speaker_b)
    segs = _parse_dialogue(req.text, a, b)
    if not segs:
        raise HTTPException(400, "No speakable lines — prefix lines with A: or B:")
    if len(segs) > 24:
        raise HTTPException(400, f"Too many lines ({len(segs)}). Max 24 per duet.")
    if req.model not in ("custom-0.6B", "custom-1.7B"):
        raise HTTPException(400, "model must be custom-0.6B or custom-1.7B")
    p = req.model_dump()
    p["speaker_a"] = a
    p["speaker_b"] = b
    return {"job_id": _job_enqueue("dialogue", p), "status": "queued", "lines": len(segs)}

class StitchRequest(BaseModel):
    filenames: List[str]
    prefix: str = Field("stitch", max_length=20)

@app.post("/api/stitch")
def stitch(req: StitchRequest):
    if not req.filenames or len(req.filenames) > 20:
        raise HTTPException(400, "Pick 1–20 files.")
    safe_prefix = "".join(c for c in (req.prefix or "stitch") if c.isalnum() or c in "-_") or "stitch"
    cfname, dur, sr = _stitch_parts(req.filenames, safe_prefix)
    return {"ok": True, "file": f"/outputs/{cfname}", "filename": cfname,
            "sample_rate": int(sr), "duration_sec": dur, "parts": len(req.filenames)}

@app.get("/api/jobs/{job_id}")
def job_status(job_id: str):
    return _job_public(job_id)

@app.post("/api/jobs/{job_id}/cancel")
def job_cancel(job_id: str, unload: bool = True):
    """Cancel a job. Queued jobs drop instantly; running ones stop at the next
    phase boundary (torch calls can't be preempted mid-forward). With
    unload=true (default) the model is also purged from memory."""
    with _jobs_lock:
        job = _jobs.get(job_id)
        if not job:
            raise HTTPException(404, "Unknown job id")
        if job["status"] in ("done", "error", "cancelled"):
            raise HTTPException(409, f"Job already {job['status']}")
        if job["status"] == "queued":
            if job_id in _pending:
                _pending.remove(job_id)
            job["cancel_requested"] = True
            job["cancel_unload"] = unload
        else:
            job["cancel_requested"] = True
            job["cancel_unload"] = unload
            return {"cancelled": True, "mode": "running — will stop at the next phase boundary",
                    "phase": job.get("phase"), "unload_on_stop": unload}
    info = _finalize_cancel(job_id)
    with _jobs_lock:
        detail = _jobs[job_id]["detail"]
    return {"cancelled": True, "mode": "was queued — dropped instantly",
            "detail": detail, "unload_info": info}

@app.post("/api/models/unload")
def models_unload(model: Optional[str] = None):
    """Purge loaded weights from memory. Safe any time: an in-flight inference
    holds its own reference, so nothing crashes — RAM frees when it returns."""
    if model and model not in MODELS:
        raise HTTPException(400, f"Unknown model '{model}'.")
    return _unload_models([model] if model else None)

@app.post("/api/server/shutdown")
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

@app.post("/api/server/restart")
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

class ClientError(BaseModel):
    msg: str = Field(max_length=2000, default="")
    url: str = Field(max_length=500, default="")
    line: str = Field(max_length=50, default="")

@app.post("/api/log/client")
def log_client(e: ClientError):
    _log.error("CLIENT %s | %s | %s", e.url[:200], e.line, e.msg[:500])
    return {"ok": True}

@app.get("/api/logs/errors")
def logs_errors(limit: int = 50):
    """Tail of the persistent error log for future diagnosis."""
    limit = max(1, min(limit, 200))
    try:
        lines = ERROR_LOG.read_text(encoding="utf-8", errors="replace").splitlines()
    except FileNotFoundError:
        return {"lines": [], "note": "no errors logged yet"}
    return {"lines": lines[-limit:], "total": len(lines)}

@app.get("/favicon.ico")
def favicon():
    from fastapi.responses import FileResponse
    p = BASE_DIR / "assets" / "icon.ico"
    if p.exists():
        return FileResponse(str(p), media_type="image/x-icon")
    return JSONResponse({"detail": "no icon"}, status_code=404)

@app.get("/api/history")
def history(limit: int = 30):
    files = sorted(OUTPUTS_DIR.glob("*.wav"), key=lambda p: p.stat().st_mtime, reverse=True)[: max(1, min(limit, 100))]
    items = [{"filename": f.name, "url": f"/outputs/{f.name}",
              "size_kb": round(f.stat().st_size / 1024, 1),
              "created": time.strftime("%Y-%m-%d %H:%M", time.localtime(f.stat().st_mtime))} for f in files]
    return {"items": items}

@app.delete("/api/history/{filename}")
def delete_item(filename: str):
    p = (OUTPUTS_DIR / filename).resolve()
    if p.parent != OUTPUTS_DIR.resolve() or not p.exists():
        raise HTTPException(404, "Not found")
    p.unlink()
    return {"ok": True}

# static + outputs
app.mount("/outputs", StaticFiles(directory=str(OUTPUTS_DIR)), name="outputs")
app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")
