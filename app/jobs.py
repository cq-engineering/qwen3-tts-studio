"""Job store, single background worker, stitching/dialogue helpers, and
graceful cancel / model unload.

Single worker (``max_workers=1``) so concurrent generations can never OOM the
box (1.7B bf16 ~3.5 GB resident). Phases are REAL backend state:
queued -> loading_model -> synthesizing -> saving -> done|error. The status
bar polls ``GET /api/jobs/{id}`` and renders exactly this.
"""
from __future__ import annotations

import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List

from fastapi import HTTPException

from app.calibration import (_CALIB_DEFAULT_SPC, _calib, _calib_lock, _calib_save,
                             _job_model_key, _job_text_len, _predict_total)
from app.config import (JOB_STEPS, MODELS, OUTPUTS_DIR, PHASE_CAP, PHASE_PROGRESS,
                        SCENARIOS, SCRIPT_LENGTHS)
from app.engine import (_load_model, _model_cache, _save_wav, _synth_custom_voice,
                        _synth_voice_clone, _synth_voice_design)
from app.errlog import _log
from app.errors import _Cancelled
from app.writers import _cloud_key, _draft_cloud, _draft_script

_executor = ThreadPoolExecutor(max_workers=1)
_jobs: Dict[str, Dict[str, Any]] = {}
_jobs_lock = threading.Lock()
_pending: List[str] = []


def _job_set(jid: str, **kw) -> None:
    with _jobs_lock:
        job = _jobs.get(jid)
        if not job:
            return
        job.update(kw)
        job["updated_at"] = time.time()
        if "phase" in kw and "progress" not in kw:
            job["progress"] = PHASE_PROGRESS.get(kw["phase"], job.get("progress", 0.04))


def _split_sentences(text: str, max_chars: int = 220, max_parts: int = 15):
    """Split text into speakable chunks (sentence-aware, hard-capped)."""
    import re
    raw = [s.strip() for s in re.split(r"(?<=[.!?…\n])\s+", (text or "").strip()) if s.strip()]
    parts, cur = [], ""
    for s in raw:
        if len(cur) + len(s) + 1 <= max_chars:
            cur = (cur + " " + s).strip()
        else:
            if cur:
                parts.append(cur)
            cur = s
            while len(cur) > max_chars:
                parts.append(cur[:max_chars])
                cur = cur[max_chars:]
    if cur:
        parts.append(cur)
    while len(parts) > max_parts:
        parts[-2] = parts[-2] + " " + parts.pop()
    return [p for p in parts if p]


def _job_label(kind: str, p: dict) -> str:
    if kind in ("custom-voice", "chunked"):
        return f"{p.get('speaker', '?')} · {p.get('model', '')}"
    if kind == "voice-design":
        return "VoiceDesign 1.7B"
    if kind == "script":
        return f"Script · {(SCENARIOS.get(p.get('scenario') or '', {}) or {}).get('label', '?')}"
    if kind == "dialogue":
        return f"Duet {p.get('speaker_a', '?')} + {p.get('speaker_b', '?')}"
    return f"Clone · {p.get('model', '')}"


def _job_enqueue(kind: str, p: dict, steps=None) -> str:
    jid = uuid.uuid4().hex[:12]
    now = time.time()
    with _jobs_lock:
        _jobs[jid] = {"id": jid, "kind": kind, "label": _job_label(kind, p),
                      "params": p, "status": "queued", "phase": "queued",
                      "detail": "Waiting for the worker…", "progress": 0.04,
                      "created_at": now, "updated_at": now,
                      "text_len": _job_text_len(kind, p),
                      "steps": steps or [s[:] for s in JOB_STEPS],
                      "started_at": None, "finished_at": None,
                      "result": None, "error": None}
        _pending.append(jid)
        # keep memory bounded: drop oldest finished jobs beyond 50
        finished = [k for k, v in _jobs.items() if v["status"] in ("done", "error")]
        for k in sorted(finished, key=lambda k: _jobs[k].get("finished_at", 0))[:-50]:
            _jobs.pop(k, None)
    _executor.submit(_run_job, jid)
    return jid


def _stitch_parts(names: list, prefix: str):
    """Stitch saved WAVs (by filename) into one file. Validates containment
    + suffix + uniform sample rate. Returns (filename, duration, sr)."""
    import soundfile as sf
    import numpy as np
    if not names or len(names) > 20:
        raise HTTPException(400, "Pick 1–20 files to stitch.")
    arrays, sr = [], None
    for nm in names:
        base = Path(nm).name
        if not base.endswith(".wav"):
            raise HTTPException(400, f"Not a WAV: {base}")
        f = (OUTPUTS_DIR / base).resolve()
        if f.parent != OUTPUTS_DIR.resolve() or not f.exists():
            raise HTTPException(404, f"Not found: {base}")
        data, sri = sf.read(str(f), dtype="float32", always_2d=False)
        if sr is None:
            sr = int(sri)
        elif int(sri) != sr:
            raise HTTPException(400, "All parts need the same sample rate.")
        arrays.append(np.asarray(data, dtype=np.float32).reshape(-1))
    combo = np.concatenate(arrays)
    cfname = f"{prefix}_{int(time.time())}_{uuid.uuid4().hex[:6]}.wav"
    sf.write(str(OUTPUTS_DIR / cfname), combo, sr)
    return cfname, round(len(combo) / float(sr), 2), sr


def _parse_dialogue(text: str, speaker_a: str, speaker_b: str):
    """[(who, line, speaker_id)] — 'A:'/'B:' prefixed lines; bare lines
    continue the current speaker (default A)."""
    import re
    out, cur = [], "A"
    for raw in (text or "").splitlines():
        s = raw.strip()
        if not s:
            continue
        m = re.match(r"^([ABab])\s*[:\-–)]\s*(.+)$", s)
        if m:
            cur = m.group(1).upper()
            line = m.group(2).strip()
        else:
            line = s
        if line:
            out.append((cur, line, speaker_a if cur == "A" else speaker_b))
    return out


def _run_job(jid: str) -> None:
    with _jobs_lock:
        job = _jobs.get(jid)
        if not job:
            return
        if jid in _pending:
            _pending.remove(jid)
        p = job["params"]
        kind = job["kind"]
    if (job or {}).get("cancel_requested"):  # cancelled while queued
        _finalize_cancel(jid)
        return
    _job_set(jid, status="running", started_at=time.time())
    try:
        if kind == "script":
            provider = (p.get("provider") or "local")
            last_push = [0.0]

            def _on_chunk(txt):
                now = time.time()
                if now - last_push[0] >= 2.0:
                    last_push[0] = now
                    _job_set(jid, partial=txt,
                             detail=f"Drafting… {len(txt.split())} words so far")
            if provider == "local":
                _job_set(jid, phase="loading_model",
                         detail="Parking TTS voices to free RAM + waking the writer…")
                freed = _unload_models()
                _job_set(jid, phase="synthesizing",
                         detail="Drafting your script — watch it appear live below…")
                try:
                    res = _draft_script(p, on_chunk=_on_chunk,
                                        should_cancel=lambda: _check_cancel(jid))
                except _Cancelled:
                    _finalize_cancel(jid)
                    return
                if freed["unloaded"]:
                    res["tts_parked"] = (f"Parked {', '.join(freed['unloaded'])} "
                                         f"(~{freed['freed_mb']} MB) to fit the writer — they reload on your next Generate.")
            else:
                _job_set(jid, phase="loading_model",
                         detail=f"Contacting free {provider} cloud writer — zero RAM touched…")
                if not _cloud_key(provider):
                    _job_set(jid, status="error", phase="error",
                             detail=f"No {provider} key saved yet — add your free key in the Script tab (60 seconds, link provided), then retry. Nothing was drafted or charged.",
                             error=f"No {provider} key saved yet — add your free key in the Script tab (60 seconds, link provided), then retry.",
                             finished_at=time.time())
                    return
                _job_set(jid, phase="synthesizing",
                         detail="Drafting at datacenter speed — seconds, not minutes…")
                try:
                    res = _draft_cloud(provider, p)
                    _job_set(jid, partial=res["script"])
                except HTTPException as e:
                    _job_set(jid, phase="synthesizing",
                             detail=f"Cloud hiccup ({e.detail}) — falling back to the free local writer…")
                    freed = _unload_models()
                    fb = dict(p)
                    fb["provider"] = "local"
                    fb["model"] = "darkidol"
                    try:
                        res = _draft_script(fb, on_chunk=_on_chunk,
                                            should_cancel=lambda: _check_cancel(jid))
                    except _Cancelled:
                        _finalize_cancel(jid)
                        return
                    res["fallback"] = (f"{provider} failed ({e.detail}) — drafted locally "
                                       f"with Dark Idol instead, still free.")
                    if freed["unloaded"]:
                        res["tts_parked"] = (f"Parked {', '.join(freed['unloaded'])} "
                                             f"(~{freed['freed_mb']} MB) to fit the writer — they reload on your next Generate.")
                if _check_cancel(jid):
                    _finalize_cancel(jid)
                    return
            _job_set(jid, phase="saving", detail="Polishing…")
            _job_set(jid, status="done", phase="done",
                     detail=f"Done — {res['words']} words in {res['took_sec']}s ✓",
                     result=res, finished_at=time.time(), progress=1.0)
            return
        if kind == "chunked":
            chunks = _split_sentences(p.get("text") or "")
            if not chunks:
                _job_set(jid, status="error", phase="error", detail="Text is empty",
                         error="Text is empty", finished_at=time.time())
                return
            n = len(chunks)
            key = p.get("model") or "custom-1.7B"
            _job_set(jid, phase="loading_model",
                     detail=f"Warming voice for {n} parts…" if key in _model_cache
                     else f"Loading {MODELS[key]['size']} weights (~{'4' if '1.7B' in key else '2'} GB, first run only)…",
                     total_parts=n, parts_done=[])
            _load_model(key)
            if _check_cancel(jid):
                _finalize_cancel(jid)
                return
            parts, sr, info = [], 24000, {}
            for i, ch in enumerate(chunks, 1):
                if _check_cancel(jid):
                    _finalize_cancel(jid)
                    return
                _job_set(jid, phase="synthesizing",
                         detail=f"Part {i}/{n} — earlier parts already playable below" if parts
                         else f"Part {i}/{n} — first audio arrives here, keep watching",
                         progress=round(12 + 78 * (i - 1) / n, 1),
                         parts_done=[f["filename"] for f in parts], total_parts=n)
                lining = dict(p)
                lining["text"] = ch
                wavs, sr, info = _synth_custom_voice(lining)
                fname = _save_wav(wavs[0], int(sr), f"{info['speaker'].lower()}_p{i}")
                parts.append({"filename": fname,
                              "duration_sec": round(len(wavs[0]) / float(sr), 2)})
                _job_set(jid, parts_done=[f["filename"] for f in parts],
                         progress=round(12 + 78 * i / n, 1))
            if _check_cancel(jid):
                _finalize_cancel(jid)
                return
            _job_set(jid, phase="saving", detail="Stitching parts into one file…", progress=96)
            cfname, dur, sr = _stitch_parts([f["filename"] for f in parts],
                                            info['speaker'].lower() + "_full")
            result = {"ok": True, "file": f"/outputs/{cfname}", "filename": cfname,
                      "sample_rate": int(sr), "duration_sec": dur,
                      "parts": [dict(f, url=f"/outputs/{f['filename']}") for f in parts],
                      **info}
            _job_set(jid, status="done", phase="done",
                     detail=f"Done — {n} parts, {dur}s total ✓",
                     result=result, finished_at=time.time(), progress=1.0)
            return
        if kind == "dialogue":
            segs = _parse_dialogue(p.get("text") or "", p.get("speaker_a") or "?",
                                   p.get("speaker_b") or "?")
            if not segs:
                _job_set(jid, status="error", phase="error",
                         detail="No speakable lines — prefix lines with A: or B:",
                         error="No speakable lines — prefix lines with A: or B:",
                         finished_at=time.time())
                return
            key = p.get("model") or "custom-1.7B"
            if key not in ("custom-0.6B", "custom-1.7B"):
                _job_set(jid, status="error", phase="error",
                         detail="model must be custom-0.6B or custom-1.7B",
                         error="model must be custom-0.6B or custom-1.7B",
                         finished_at=time.time())
                return
            m = len(segs)
            _job_set(jid, phase="loading_model",
                     detail=f"Warming voices for a {m}-line duet…" if key in _model_cache
                     else f"Loading {MODELS[key]['size']} weights…",
                     total_parts=m, parts_done=[])
            _load_model(key)
            if _check_cancel(jid):
                _finalize_cancel(jid)
                return
            lang = (p.get("language") or "").strip() or "Auto"
            parts, sr = [], 24000
            for i, (who, line, spk) in enumerate(segs, 1):
                if _check_cancel(jid):
                    _finalize_cancel(jid)
                    return
                _job_set(jid, phase="synthesizing",
                         detail=f"Line {i}/{m} ({spk})…" + (" — earlier lines playable in Library" if parts else ""),
                         progress=round(12 + 78 * (i - 1) / m, 1),
                         parts_done=[f["filename"] for f in parts], total_parts=m)
                lining = {"text": line, "language": lang, "speaker": spk,
                          "instruct": (p.get("instruct") or "").strip() or None,
                          "model": key, "max_new_tokens": p.get("max_new_tokens") or 2048}
                wavs, sr, info = _synth_custom_voice(lining)
                fname = _save_wav(wavs[0], int(sr), f"duet_{who.lower()}{i}")
                parts.append({"filename": fname, "speaker": spk, "who": who, "text": line,
                              "duration_sec": round(len(wavs[0]) / float(sr), 2)})
                _job_set(jid, parts_done=[f["filename"] for f in parts],
                         progress=round(12 + 78 * i / m, 1))
            if _check_cancel(jid):
                _finalize_cancel(jid)
                return
            _job_set(jid, phase="saving", detail="Stitching the duet…", progress=96)
            cfname, dur, sr = _stitch_parts([f["filename"] for f in parts], "duet_full")
            result = {"ok": True, "file": f"/outputs/{cfname}", "filename": cfname,
                      "sample_rate": int(sr), "duration_sec": dur,
                      "segments": [dict(s, url=f"/outputs/{s['filename']}") for s in parts],
                      "model": key, "language": lang}
            _job_set(jid, status="done", phase="done",
                     detail=f"Done — duet of {m} lines, {dur}s ✓",
                     result=result, finished_at=time.time(), progress=1.0)
            return
        model_key = _job_model_key(kind, p)
        if model_key in _model_cache:
            _job_set(jid, phase="loading_model", detail="Model already warm — skipping load ✓")
        else:
            size = MODELS[model_key]["size"]
            _job_set(jid, phase="loading_model",
                     detail=f"Loading {size} weights (~{'4' if '1.7B' in size else '2'} GB, first run only)…")
            _load_model(model_key)
        if _check_cancel(jid):
            _finalize_cancel(jid)
            return
        _job_set(jid, phase="synthesizing",
                 detail="Running the model — on CPU a sentence typically takes ~2 min. The bar moves when phases change; elapsed time proves it's alive.")
        if kind == "custom-voice":
            wavs, sr, info = _synth_custom_voice(p)
            prefix = info["speaker"].lower()
        elif kind == "voice-design":
            wavs, sr, info = _synth_voice_design(p)
            prefix = "designed"
        else:
            wavs, sr, info = _synth_voice_clone(p)
            prefix = "clone"
        if _check_cancel(jid):  # discard partial work, never save it
            _finalize_cancel(jid)
            return
        _job_set(jid, phase="saving", detail="Writing 24 kHz WAV…")
        fname = _save_wav(wavs[0], int(sr), prefix)
        dur = round(len(wavs[0]) / float(sr), 2)
        result = {"ok": True, "file": f"/outputs/{fname}", "filename": fname,
                  "sample_rate": int(sr), "duration_sec": dur, **info}
        with _calib_lock:  # learn real throughput for the next estimate
            n = _job_text_len(kind, p)
            wall = max(1.0, time.time() - (job.get("started_at") or time.time()))
            sample = min(8.0, max(0.05, wall / max(1, n)))
            old = float(_calib["sec_per_char"].get(model_key, _CALIB_DEFAULT_SPC.get(model_key, 1.2)))
            runs = int(_calib["runs"].get(model_key, 0))
            alpha = 1.0 / min(runs + 1, 8)
            _calib["sec_per_char"][model_key] = round(old + (sample - old) * alpha, 3)
            _calib["runs"][model_key] = runs + 1
        _calib_save()
        _job_set(jid, status="done", phase="done", detail=f"Done — {dur}s of audio ✓",
                 result=result, finished_at=time.time(), progress=1.0)
    except HTTPException as e:
        _log.warning("job %s rejected: %s", jid, e.detail)
        _job_set(jid, status="error", phase="error", detail=str(e.detail),
                 error=str(e.detail), finished_at=time.time())
    except Exception as e:
        try:
            _jb = _jobs.get(jid, {})
            _log.exception("job %s failed (kind=%s label=%s)", jid, _jb.get("kind"), _jb.get("label"))
        except Exception:
            _log.exception("job %s failed", jid)
        _job_set(jid, status="error", phase="error",
                 detail=f"{type(e).__name__}: {e}", error=f"{type(e).__name__}: {e}",
                 finished_at=time.time())


def _job_public(jid: str) -> dict:
    with _jobs_lock:
        job = _jobs.get(jid)
        if not job:
            raise HTTPException(404, "Unknown job id")
        out = {k: v for k, v in job.items() if k != "params"}
        try:
            out["queue_position"] = _pending.index(jid) if job["status"] == "queued" else 0
        except ValueError:
            out["queue_position"] = 0
        now = time.time()
        t0 = job.get("started_at") or job.get("created_at") or now
        t1 = job.get("finished_at") or now
        out["elapsed_sec"] = round(t1 - t0, 1)
        out["partial"] = (job.get("partial") or "")[-4000:]
        try:
            if job["kind"] == "script":
                mk = "script"
                pp = job["params"] or {}
                pred = (25.0 if pp.get("provider") not in (None, "local")
                        else SCRIPT_LENGTHS[pp.get("length", "standard")]["eta"])
            else:
                mk = _job_model_key(job["kind"], job["params"])
                pred = _predict_total(job["kind"], job["params"], mk)
        except Exception:
            pred, mk = 120.0, "custom-1.7B"
        cap = PHASE_CAP.get(job["phase"], 99)
        pct = 100.0 * (t1 - t0) / max(1.0, pred)
        out["percent"] = 100 if job["status"] == "done" else round(min(cap, max(0.0, pct)), 1)
        out["predicted_sec"] = round(pred, 1)
        out["overtime"] = bool(job["status"] == "running" and (t1 - t0) > pred * 1.15)
        with _calib_lock:
            out["calib_runs"] = int(_calib["runs"].get(mk, 0))
        return out


# ---------------------------------------------------- cancel + unload ---
# Cancel is GRACEFUL, never a thread kill (torch forward passes can't be
# preempted safely): the flag takes effect at the next phase boundary, queued
# jobs drop instantly, and partial work is always discarded, never saved.
# Unload drops the model reference + gc + cuda cache and reports REAL freed
# MB measured from process RSS before/after. The running inference holds its
# own reference, so unloading mid-synthesis is safe (memory frees when the
# abandoned call returns and is discarded).


def _check_cancel(jid: str) -> bool:
    with _jobs_lock:
        job = _jobs.get(jid)
        return bool(job and job.get("cancel_requested"))


def _unload_models(keys=None) -> dict:
    import gc
    rss0 = None
    try:
        import psutil, os
        rss0 = psutil.Process(os.getpid()).memory_info().rss
    except Exception:
        pass
    unloaded = []
    for k in (keys or list(_model_cache.keys())):
        m = _model_cache.pop(k, None)
        if m is not None:
            try:
                del m
            except Exception:
                pass
            unloaded.append(k)
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    gc.collect()
    rss1 = None
    try:
        import psutil, os
        rss1 = psutil.Process(os.getpid()).memory_info().rss
    except Exception:
        pass
    freed = round((rss0 - rss1) / 1e6, 1) if rss0 and rss1 else None
    if freed is not None and freed < 0:
        freed = 0.0  # RSS noise when little was actually resident
    return {"unloaded": unloaded, "freed_mb": freed,
            "rss_mb": round(rss1 / 1e6, 1) if rss1 else None}


def _finalize_cancel(jid: str) -> str:
    with _jobs_lock:
        job = _jobs.get(jid)
        unload = bool(job and job.get("cancel_unload", True))
        kind, p = (job["kind"], job["params"]) if job else (None, {})
    for stale in [p.get("ref_audio_path")] if isinstance(p, dict) else []:
        try:
            if stale:
                Path(stale).unlink(missing_ok=True)  # cancelled clones leave no orphans
        except Exception:
            pass
    info = ""
    if unload and kind:
        try:
            r = _unload_models([_job_model_key(kind, p)])
            if r["unloaded"]:
                info = (f" · model unloaded, freed ~{r['freed_mb']} MB"
                        if r["freed_mb"] is not None else " · model unloaded")
            else:
                info = " · model was already unloaded"
        except Exception as e:
            info = f" · unload hiccup: {e}"
    _job_set(jid, status="cancelled", phase="cancelled",
             detail="Cancelled — partial work discarded." + info,
             finished_at=time.time())
    return info
