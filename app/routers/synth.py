"""Synthesis endpoints: blocking (for scripts), async jobs (the UI path),
two-speaker dialogue, the library stitcher, and job status/cancel."""
from __future__ import annotations

import uuid
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from app.config import MODELS, UPLOADS_DIR
from app.engine import (_canon_speaker, _save_wav, _synth_custom_voice,
                        _synth_voice_clone, _synth_voice_design)
from app.errlog import _log
from app.jobs import (_finalize_cancel, _job_enqueue, _job_public, _jobs,
                      _jobs_lock, _parse_dialogue, _pending, _split_sentences,
                      _stitch_parts)
from app.schemas import (CustomVoiceRequest, DialogueRequest, StitchRequest,
                         VoiceDesignRequest)

router = APIRouter()


# ------------------------- blocking endpoints (for scripts) ---
@router.post("/api/synthesize/custom-voice")
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


@router.post("/api/synthesize/voice-design")
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


@router.post("/api/synthesize/voice-clone")
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
@router.post("/api/jobs/custom-voice", status_code=202)
def job_custom(req: CustomVoiceRequest):
    return {"job_id": _job_enqueue("custom-voice", req.model_dump()), "status": "queued"}


@router.post("/api/jobs/chunked", status_code=202)
def job_chunked(req: CustomVoiceRequest):
    """Stream mode: sentence-split synthesis with per-part playback + real
    parts-done progress. Single-sentence input falls back to a normal job."""
    if len(_split_sentences(req.text)) <= 1:
        return {"job_id": _job_enqueue("custom-voice", req.model_dump()), "status": "queued"}
    return {"job_id": _job_enqueue("chunked", req.model_dump()), "status": "queued"}


@router.post("/api/jobs/voice-design", status_code=202)
def job_design(req: VoiceDesignRequest):
    return {"job_id": _job_enqueue("voice-design", req.model_dump()), "status": "queued"}


@router.post("/api/jobs/voice-clone", status_code=202)
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


@router.post("/api/jobs/dialogue", status_code=202)
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


@router.post("/api/stitch")
def stitch(req: StitchRequest):
    if not req.filenames or len(req.filenames) > 20:
        raise HTTPException(400, "Pick 1–20 files.")
    safe_prefix = "".join(c for c in (req.prefix or "stitch") if c.isalnum() or c in "-_") or "stitch"
    cfname, dur, sr = _stitch_parts(req.filenames, safe_prefix)
    return {"ok": True, "file": f"/outputs/{cfname}", "filename": cfname,
            "sample_rate": int(sr), "duration_sec": dur, "parts": len(req.filenames)}


@router.get("/api/jobs/{job_id}")
def job_status(job_id: str):
    return _job_public(job_id)


@router.post("/api/jobs/{job_id}/cancel")
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
