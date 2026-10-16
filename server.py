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

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
OUTPUTS_DIR = BASE_DIR / "outputs"
UPLOADS_DIR = BASE_DIR / "uploads"
OUTPUTS_DIR.mkdir(exist_ok=True)
UPLOADS_DIR.mkdir(exist_ok=True)

# ------------------------------------------------------------- error log ---
# Persistent, timestamped, rotation-capped error reporting (errors.log).
# Why: tracebacks previously went only to stdout (server.log), which is
# TRUNCATED on every start — so post-crash diagnosis was impossible. Every
# 500, job failure, and browser-side error lands here with context.
import logging
from logging.handlers import RotatingFileHandler
ERROR_LOG = BASE_DIR / "errors.log"
_log = logging.getLogger("qwen3-tts_studio")
_log.setLevel(logging.INFO)
if not _log.handlers:
    _fh = RotatingFileHandler(ERROR_LOG, maxBytes=1_000_000, backupCount=3, encoding="utf-8")
    _fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    _log.addHandler(_fh)

# ---------------------------------------------------------------- voices ---
# From official README (Qwen/Qwen3-TTS-12Hz-*-CustomVoice)
VOICES: List[Dict[str, Any]] = [
    {
        "id": "Aiden", "label": "Aiden", "gender": "Male", "age": "Young Adult",
        "native_language": "English", "region": "American",
        "description": "Sunny American male voice with a clear midrange.",
        "tags": ["american", "male", "english", "narration", "podcast", "high-quality"],
        "featured": True, "flag": "🇺🇸",
        "best_for": "Podcasts, narration, ads, YouTube — flagship US voice",
    },
    {
        "id": "Ryan", "label": "Ryan", "gender": "Male", "age": "Young Adult",
        "native_language": "English", "region": "American",
        "description": "Dynamic male voice with strong rhythmic drive.",
        "tags": ["american", "male", "english", "energetic", "promo", "high-quality"],
        "featured": True, "flag": "🇺🇸",
        "best_for": "Trailers, promos, energetic reads, storytelling",
    },
    {
        "id": "Vivian", "label": "Vivian", "gender": "Female", "age": "Young",
        "native_language": "Chinese", "region": "Chinese",
        "description": "Bright, slightly edgy young female voice.",
        "tags": ["female", "chinese", "bright", "young"],
        "featured": False, "flag": "🇨🇳",
        "best_for": "Multilingual content, character work",
    },
    {
        "id": "Serena", "label": "Serena", "gender": "Female", "age": "Young",
        "native_language": "Chinese", "region": "Chinese",
        "description": "Warm, gentle young female voice.",
        "tags": ["female", "chinese", "warm", "gentle"],
        "featured": False, "flag": "🇨🇳",
        "best_for": "Audiobooks, calm narration",
    },
    {
        "id": "Uncle_Fu", "label": "Uncle Fu", "gender": "Male", "age": "Middle-aged",
        "native_language": "Chinese", "region": "Chinese",
        "description": "Seasoned male voice with a low, mellow timbre.",
        "tags": ["male", "chinese", "deep", "mellow", "documentary"],
        "featured": False, "flag": "🇨🇳",
        "best_for": "Documentary, deep narration",
    },
    {
        "id": "Dylan", "label": "Dylan", "gender": "Male", "age": "Young",
        "native_language": "Chinese (Beijing Dialect)", "region": "Chinese",
        "description": "Youthful Beijing male voice with a clear, natural timbre.",
        "tags": ["male", "chinese", "beijing", "dialect"],
        "featured": False, "flag": "🇨🇳",
        "best_for": "Beijing dialect content",
    },
    {
        "id": "Eric", "label": "Eric", "gender": "Male", "age": "Young",
        "native_language": "Chinese (Sichuan Dialect)", "region": "Chinese",
        "description": "Lively Chengdu male voice with a slightly husky brightness.",
        "tags": ["male", "chinese", "sichuan", "dialect", "lively"],
        "featured": False, "flag": "🇨🇳",
        "best_for": "Sichuan dialect content",
    },
    {
        "id": "Ono_Anna", "label": "Ono Anna", "gender": "Female", "age": "Young",
        "native_language": "Japanese", "region": "Japanese",
        "description": "Playful Japanese female voice with a light, nimble timbre.",
        "tags": ["female", "japanese", "playful", "anime"],
        "featured": False, "flag": "🇯🇵",
        "best_for": "Anime, playful characters",
    },
    {
        "id": "Sohee", "label": "Sohee", "gender": "Female", "age": "Young",
        "native_language": "Korean", "region": "Korean",
        "description": "Warm Korean female voice with rich emotion.",
        "tags": ["female", "korean", "warm", "emotional"],
        "featured": False, "flag": "🇰🇷",
        "best_for": "K-drama style, emotional reads",
    },
]

LANGUAGES = ["Auto", "English", "Chinese", "Japanese", "Korean", "French",
             "German", "Spanish", "Portuguese", "Russian", "Italian"]

MODELS = {
    "custom-0.6B": {"hf_id": "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice", "label": "CustomVoice 0.6B — Fast", "size": "0.6B", "kind": "CustomVoice", "instruct": False, "quality": "Fast / lighter"},
    "custom-1.7B": {"hf_id": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice", "label": "CustomVoice 1.7B — High Quality ⭐", "size": "1.7B", "kind": "CustomVoice", "instruct": True, "quality": "Best quality"},
    "design-1.7B": {"hf_id": "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign", "label": "VoiceDesign 1.7B — describe any voice", "size": "1.7B", "kind": "VoiceDesign", "instruct": True, "quality": "Creative"},
    "base-0.6B": {"hf_id": "Qwen/Qwen3-TTS-12Hz-0.6B-Base", "label": "Base 0.6B — Voice Clone (fast)", "size": "0.6B", "kind": "Base", "instruct": False, "quality": "Fast clone"},
    "base-1.7B": {"hf_id": "Qwen/Qwen3-TTS-12Hz-1.7B-Base", "label": "Base 1.7B — Voice Clone (best)", "size": "1.7B", "kind": "Base", "instruct": False, "quality": "Best clone"},
}

_model_cache: Dict[str, Any] = {}
_backend_info: Dict[str, Any] = {"torch": None, "cuda": False, "qwen_tts": None, "device": "cpu"}

def _detect_backend():
    try:
        import torch
        _backend_info["torch"] = torch.__version__
        try:
            _backend_info["cuda"] = torch.cuda.is_available()
        except Exception:
            _backend_info["cuda"] = False
        _backend_info["device"] = "cuda" if _backend_info["cuda"] else "cpu"
    except Exception as e:
        _backend_info["torch"] = f"missing ({e})"
    try:
        import qwen_tts  # noqa
        import importlib.metadata as md
        try:
            _backend_info["qwen_tts"] = md.version("qwen-tts")
        except Exception:
            _backend_info["qwen_tts"] = "installed"
    except Exception as e:
        _backend_info["qwen_tts"] = f"missing ({e})"

_detect_backend()

def _load_model(key: str):
    if key in _model_cache:
        return _model_cache[key]
    if key not in MODELS:
        raise HTTPException(400, f"Unknown model '{key}'. Choose from {list(MODELS)}")
    try:
        import torch
        from qwen_tts import Qwen3TTSModel
    except Exception as e:
        raise HTTPException(500, f"TTS backend not installed yet: {e}. Run: pip install -r requirements.txt (+ torch CPU).")
    hf_id = MODELS[key]["hf_id"]
    use_cuda = _backend_info.get("cuda", False)
    # bfloat16 halves RAM vs float32 (~3.5 GB for 1.7B) and works on modern CPUs;
    # fall back to float32 if the kernel lacks bf16 support.
    kwargs: Dict[str, Any] = dict(device_map="cuda" if use_cuda else "cpu")
    kwargs["attn_implementation"] = "sdpa"
    print(f"[studio] loading {hf_id} on {kwargs['device_map']} ...", flush=True)
    t0load = time.time()
    try:
        import torch
        model = Qwen3TTSModel.from_pretrained(hf_id, dtype=torch.bfloat16, **kwargs)
    except Exception as e:
        print(f"[studio] bfloat16 load failed ({e}), retrying float32 ...", flush=True)
        import torch
        model = Qwen3TTSModel.from_pretrained(hf_id, dtype=torch.float32, **kwargs)
    _model_cache[key] = model
    with _calib_lock:
        _calib["load_sec"][key] = round(time.time() - t0load, 1)
    _calib_save()
    print(f"[studio] loaded {hf_id}", flush=True)
    return model

def _save_wav(wav, sr: int, prefix: str) -> str:
    import soundfile as sf
    import numpy as np
    arr = np.asarray(wav)
    if arr.dtype != "float32":
        arr = arr.astype("float32")
    fname = f"{prefix}_{int(time.time())}_{uuid.uuid4().hex[:6]}.wav"
    path = OUTPUTS_DIR / fname
    sf.write(str(path), arr, sr)
    return fname

# ------------------------------------------------------ synth cores ---
# Pure logic shared by the blocking endpoints and the background jobs.
# Each returns (wavs, sample_rate, info). Raises HTTPException on bad input.

def _canon_speaker(name: str) -> str:
    canon = {v["id"].lower(): v["id"] for v in VOICES}
    key = (name or "").strip().replace(" ", "_").lower()
    if key not in canon:
        raise HTTPException(400, f"Unknown speaker '{name}'.")
    return canon[key]

def _gen_kwargs(p: dict) -> Dict[str, Any]:
    kw: Dict[str, Any] = {"max_new_tokens": int(p.get("max_new_tokens") or 2048)}
    if p.get("temperature") is not None:
        kw["temperature"] = float(p["temperature"])
    if p.get("top_p") is not None:
        kw["top_p"] = float(p["top_p"])
    if p.get("top_k") is not None:
        kw["top_k"] = int(p["top_k"])
    return kw

def _apply_seed(seed) -> None:
    if seed is None or seed == "":
        return
    try:
        import torch, random, numpy as np
        seed = int(seed)
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed % (2**32 - 1))
    except Exception:
        pass

def _synth_custom_voice(p: dict) -> Tuple[Any, int, dict]:
    text = (p.get("text") or "").strip()
    if not text:
        raise HTTPException(400, "Text is empty")
    speaker = _canon_speaker(p.get("speaker", ""))
    model_key = p.get("model") or "custom-1.7B"
    if model_key not in ("custom-0.6B", "custom-1.7B"):
        raise HTTPException(400, "model must be custom-0.6B or custom-1.7B")
    language = (p.get("language") or "").strip() or "Auto"
    instruct = (p.get("instruct") or "").strip() or None
    if instruct and not MODELS[model_key]["instruct"]:
        instruct = None  # 0.6B CustomVoice has no instruction control
    _apply_seed(p.get("seed"))
    model = _load_model(model_key)
    wavs, sr = model.generate_custom_voice(
        text=text, language=language, speaker=speaker,
        instruct=instruct, non_streaming_mode=True, **_gen_kwargs(p),
    )
    return wavs, int(sr), {"speaker": speaker, "language": language,
                           "model": model_key, "instruct_applied": bool(instruct)}

def _synth_voice_design(p: dict) -> Tuple[Any, int, dict]:
    text = (p.get("text") or "").strip()
    instruct = (p.get("instruct") or "").strip()
    if not text or not instruct:
        raise HTTPException(400, "Text and voice description are required")
    _apply_seed(p.get("seed"))
    model = _load_model("design-1.7B")
    wavs, sr = model.generate_voice_design(
        text=text, language=(p.get("language") or "").strip() or "Auto",
        instruct=instruct, non_streaming_mode=True, **_gen_kwargs(p),
    )
    return wavs, int(sr), {"language": (p.get("language") or "Auto").strip() or "Auto"}

def _synth_voice_clone(p: dict) -> Tuple[Any, int, dict]:
    target_text = (p.get("target_text") or "").strip()
    if not target_text:
        raise HTTPException(400, "Target text is required")
    model_key = p.get("model") or "base-1.7B"
    if model_key not in ("base-0.6B", "base-1.7B"):
        raise HTTPException(400, "model must be base-0.6B or base-1.7B")
    xvec = bool(p.get("x_vector_only"))
    ref_text = (p.get("ref_text") or "").strip()
    if not xvec and not ref_text:
        raise HTTPException(400, "Reference transcript is required (or enable x-vector-only mode)")
    ref_path = p.get("ref_audio_path")
    if not ref_path:
        raise HTTPException(400, "Reference audio is missing")
    try:
        import soundfile as sf
        data, sr_in = sf.read(str(ref_path), dtype="float32", always_2d=False)
        if getattr(data, "ndim", 1) > 1:
            data = data.mean(axis=-1)
        ref_arg = (data, int(sr_in))
    except Exception:
        ref_arg = str(ref_path)
    tts = _load_model(model_key)
    wavs, sr = tts.generate_voice_clone(
        text=target_text, language=(p.get("language") or "").strip() or "Auto",
        ref_audio=ref_arg, ref_text=(ref_text or None),
        x_vector_only_mode=xvec,
        max_new_tokens=int(p.get("max_new_tokens") or 2048),
    )
    return wavs, int(sr), {"model": model_key}

# ------------------------------------------------- live % calibration ---
# True per-token progress isn't exposed by qwen-tts (one monolithic generate
# call, no stream/callback hooks), so the % is a SELF-CALIBRATING live
# estimate: wall-seconds per input char, learned from real completed jobs on
# THIS machine and persisted in calib.json. It moves every poll, converges
# with use, and always lands on exactly 100 at completion. Phases stay 100%
# real backend state; only the within-phase % is estimated (labeled as such).
import json as _json
_calib_lock = threading.Lock()
CALIB_PATH = BASE_DIR / "calib.json"
_calib = {"sec_per_char": {}, "runs": {}, "load_sec": {}}
_CALIB_DEFAULT_SPC = {"custom-1.7B": 1.5, "custom-0.6B": 0.5, "design-1.7B": 1.5,
                      "base-1.7B": 1.5, "base-0.6B": 0.5}
def _calib_load():
    try:
        data = _json.loads(CALIB_PATH.read_text())
        for k in ("sec_per_char", "runs", "load_sec"):
            if isinstance(data.get(k), dict):
                _calib[k].update({kk: float(v) for kk, v in data[k].items()})
    except Exception:
        pass
def _calib_save():
    try:
        CALIB_PATH.write_text(_json.dumps(_calib))
    except Exception:
        pass
_calib_load()

def _job_model_key(kind: str, p: dict) -> str:
    return {"custom-voice": p.get("model") or "custom-1.7B",
            "chunked": p.get("model") or "custom-1.7B",
            "voice-design": "design-1.7B",
            "voice-clone": p.get("model") or "base-1.7B"}[kind]

def _job_text_len(kind: str, p: dict) -> int:
    t = p.get("text") if kind != "voice-clone" else p.get("target_text")
    return max(1, len((t or "").strip()))

def _predict_total(kind: str, p: dict, model_key: str) -> float:
    with _calib_lock:
        spc = float(_calib["sec_per_char"].get(model_key, _CALIB_DEFAULT_SPC.get(model_key, 1.2)))
        load = float(_calib["load_sec"].get(model_key, 40.0))
    total = _job_text_len(kind, p) * spc + (0.0 if model_key in _model_cache else load) + 6.0
    return max(10.0, total)

PHASE_CAP = {"queued": 5, "loading_model": 22, "synthesizing": 99, "saving": 99,
             "done": 100, "error": 99, "cancelled": 100}

# ------------------------------------------------------------- jobs ---
# Single background worker. Phases are REAL backend state (not estimates):
# queued -> loading_model -> synthesizing -> saving -> done | error.
# The status bar polls GET /api/jobs/{id} and renders exactly this.

_executor = ThreadPoolExecutor(max_workers=1)
_jobs: Dict[str, Dict[str, Any]] = {}
_jobs_lock = threading.Lock()
_pending: List[str] = []

PHASE_PROGRESS = {"queued": 0.04, "loading_model": 0.12, "synthesizing": 0.40,
                  "saving": 0.92, "done": 1.0, "error": 1.0}

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
            import soundfile as sf
            import numpy as np
            arrays = [np.asarray(sf.read(str(OUTPUTS_DIR / f["filename"]),
                                         dtype="float32", always_2d=False)[0],
                                 dtype=np.float32).reshape(-1) for f in parts]
            combo = np.concatenate(arrays)
            cfname = f"{info['speaker'].lower()}_full_{int(time.time())}_{uuid.uuid4().hex[:6]}.wav"
            sf.write(str(OUTPUTS_DIR / cfname), combo, sr)
            dur = round(len(combo) / float(sr), 2)
            result = {"ok": True, "file": f"/outputs/{cfname}", "filename": cfname,
                      "sample_rate": int(sr), "duration_sec": dur,
                      "parts": [dict(f, url=f"/outputs/{f['filename']}") for f in parts],
                      **info}
            _job_set(jid, status="done", phase="done",
                     detail=f"Done — {n} parts, {dur}s total ✓",
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

# ------------------------------------------------- script writer ---
# FREE local-LLM scripts via Ollama (no cloud, no key, no content filters on
# the abliterated models). TTS weights are parked first so the writer fits in
# this box's 7.8 GB RAM, and they reload on the next audio Generate.
import urllib.request as _urlreq

SCRIPT_MODELS = {
    "best": {"tag": "huihui_ai/qwen3.5-abliterated:9b", "label": "Best — 9B abliterated ⭐",
             "note": "Sharpest prose, unfiltered. Parks TTS voices to fit RAM."},
    "balanced": {"tag": "hermes3:8b", "label": "Balanced — 8B",
                 "note": "Strong quality, lighter footprint."},
    "darkidol": {"tag": "crown/darkidol:latest", "label": "Dark Idol — uncensored",
                 "note": "Roleplay-strong creative writer, unfiltered."},
    "fast": {"tag": "hermes3:3b", "label": "Fast — 3B",
             "note": "Drafts in seconds, simpler prose."},
}
SCRIPT_LENGTHS = {
    "hook": {"label": "Hook ~60 words", "words": 60, "tokens": 160, "eta": 40.0},
    "standard": {"label": "Standard ~180 words", "words": 180, "tokens": 420, "eta": 90.0},
    "extended": {"label": "Extended ~400 words", "words": 400, "tokens": 900, "eta": 180.0},
}
SCENARIOS = {
    "ad": {"label": "📢 Ad Read", "desc": "Punchy 30-second style spot",
           "brief": "Punchy direct-response advertising: magnetic hook, two vivid benefits, hard call to action."},
    "podcast": {"label": "🎧 Podcast Intro", "desc": "Warm host welcome + tease",
                "brief": "Warm podcast host welcoming listeners back, teasing today's topic, setting an inviting tone."},
    "story": {"label": "📖 Short Story", "desc": "Micro-fiction with a twist",
              "brief": "Vivid micro-fiction with a twist ending and rich sensory detail."},
    "dialogue": {"label": "💬 Dialogue", "desc": "Two voices, labeled A:/B:",
                 "brief": "A two-character dialogue with lines labeled A: and B:, distinct voices, natural interruptions and subtext."},
    "monologue": {"label": "🎭 Monologue", "desc": "One raw theatrical moment",
                  "brief": "One character's raw inner moment, theatrical and confessional, building to a killer final line."},
    "horror": {"label": "👻 Horror Scene", "desc": "Genuinely unsettling",
               "brief": "Genuinely unsettling horror: dread, gore and darkness allowed, no cozy resolutions."},
    "noir": {"label": "🕵️ Noir", "desc": "Hardboiled detective yarn",
             "brief": "Hardboiled detective narration dripping with cynical metaphors, rain and neon."},
    "romance": {"label": "💘 Romance", "desc": "Adult tension, tasteful",
                "brief": "Adult romantic tension, sensual but tasteful, all longing and payoff."},
    "comedy": {"label": "😂 Comedy Sketch", "desc": "Fast jokes, absurdity",
               "brief": "Fast-paced comedy sketch with escalating absurdity and quotable punchlines."},
    "motivational": {"label": "🔥 Motivational", "desc": "High-voltage coach speech",
                     "brief": "High-voltage motivational speech: short hammer sentences, fire, and a call to action."},
    "bedtime": {"label": "🌙 Bedtime", "desc": "Slow, soft, drowsy-safe",
                "brief": "Slow bedtime tale: soft drowsy rhythm, gentle imagery, a sleepy ending, nothing frightening."},
    "epic": {"label": "⚔️ Epic Narration", "desc": "Mythic trailer-scale",
             "brief": "Mythic trailer-scale narration: weighty, dramatic, larger than life."},
}
SCRIPT_STEPS = [["queued", "Queued"], ["loading_model", "Waking writer"],
                ["synthesizing", "Drafting"], ["saving", "Polishing"]]
JOB_STEPS = [["queued", "Queued"], ["loading_model", "Loading model"],
             ["synthesizing", "Synthesizing"], ["saving", "Saving"]]

# ------------------------------------------------- cloud writers ---
# OPTIONAL free-tier cloud backends (Groq / Gemini): 10–100x faster than local
# CPU inference with 70B-class quality, still $0 on their free tiers.
# Keys are BYO (60-second signup, links in the UI), stored chmod-600 in
# keys.json — values NEVER leave the server (GET only reports configured
# true/false) — and GROQ_API_KEY / GEMINI_API_KEY env vars also work.
# Cloud failures fall back to the free local writer instead of erroring.
import os as _os
KEYS_PATH = BASE_DIR / "keys.json"
_keys: Dict[str, str] = {}

def _keys_load():
    try:
        d = _json.loads(KEYS_PATH.read_text())
        if isinstance(d, dict):
            _keys.update({k: str(v) for k, v in d.items() if v})
    except Exception:
        pass

def _keys_save():
    try:
        KEYS_PATH.write_text(_json.dumps(_keys))
        _os.chmod(KEYS_PATH, 0o600)
    except Exception:
        pass

_keys_load()

SCRIPT_CLOUD = {
    "groq": {"label": "Groq ⚡", "key_url": "https://console.groq.com/keys",
             "note": "Free tier, ~2000 tok/s. 70B quality in seconds.",
             "default": "llama-3.3-70b-versatile",
             "models": [{"id": "llama-3.3-70b-versatile", "label": "70B Versatile — best quality"},
                        {"id": "llama-3.1-8b-instant", "label": "8B Instant — fastest"}]},
    "gemini": {"label": "Gemini ✨", "key_url": "https://aistudio.google.com/apikey",
               "note": "Free tier, generous quotas, fast.",
               "default": "gemini-2.0-flash",
               "models": [{"id": "gemini-2.0-flash", "label": "2.0 Flash — fast + sharp"}]},
}
# NOTE (Oct 2026): providers rename models occasionally — if a cloud draft
# starts failing with model-not-found, check their docs and edit the IDs here.

def _cloud_key(provider: str) -> str:
    env = {"groq": "GROQ_API_KEY", "gemini": "GEMINI_API_KEY"}.get(provider, "")
    k = (_os.environ.get(env, "") or "").strip() if env else ""
    if not k:
        with _calib_lock:
            k = str(_keys.get(provider, "") or "").strip()
    return k

def _cloud_verify(provider: str, key: str):
    try:
        if provider == "groq":
            req = _urlreq.Request("https://api.groq.com/openai/v1/models",
                                  headers={"Authorization": f"Bearer {key}"})
            with _urlreq.urlopen(req, timeout=20) as r:
                return (r.status == 200), ("ok" if r.status == 200 else f"HTTP {r.status}")
        if provider == "gemini":
            req = _urlreq.Request(
                f"https://generativelanguage.googleapis.com/v1beta/models?key={key}")
            with _urlreq.urlopen(req, timeout=20) as r:
                return (r.status == 200), ("ok" if r.status == 200 else f"HTTP {r.status}")
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"
    return False, "unknown provider"

def _draft_cloud(provider: str, p: dict) -> dict:
    topic = (p.get("topic") or "").strip()
    sc = SCENARIOS.get(p.get("scenario") or "")
    if not topic or not sc:
        raise HTTPException(400, "Topic and scenario are required")
    cfg = SCRIPT_CLOUD.get(provider)
    if not cfg:
        raise HTTPException(400, "Unknown provider")
    mid = p.get("model") or cfg["default"]
    if mid not in {m["id"] for m in cfg["models"]}:
        mid = cfg["default"]
    key = _cloud_key(provider)
    if not key:
        raise HTTPException(400, f"No {provider} key saved — add your free key in the Script tab (60 seconds, link provided).")
    length = p.get("length") if p.get("length") in SCRIPT_LENGTHS else "standard"
    words, toks = SCRIPT_LENGTHS[length]["words"], min(SCRIPT_LENGTHS[length]["tokens"], 600)
    system, prompt = _script_prompt(topic, sc, (p.get("tone") or "").strip(), words)
    t0 = time.time()
    try:
        if provider == "groq":
            body = _json.dumps({"model": mid,
                                "messages": [{"role": "system", "content": system},
                                             {"role": "user", "content": prompt}],
                                "temperature": 0.95, "top_p": 0.9,
                                "max_tokens": toks}).encode()
            req = _urlreq.Request("https://api.groq.com/openai/v1/chat/completions",
                                  data=body, headers={"Content-Type": "application/json",
                                                      "Authorization": f"Bearer {key}"})
            with _urlreq.urlopen(req, timeout=120) as r:
                d = _json.loads(r.read().decode())
            text = d["choices"][0]["message"]["content"]
        else:
            body = _json.dumps({"system_instruction": {"parts": [{"text": system}]},
                                "contents": [{"parts": [{"text": prompt}]}],
                                "generationConfig": {"temperature": 0.95, "topP": 0.9,
                                                     "maxOutputTokens": toks}}).encode()
            req = _urlreq.Request(
                f"https://generativelanguage.googleapis.com/v1beta/models/{mid}:generateContent?key={key}",
                data=body, headers={"Content-Type": "application/json"})
            with _urlreq.urlopen(req, timeout=120) as r:
                d = _json.loads(r.read().decode())
            text = "".join(prt.get("text", "") for prt in d["candidates"][0]["content"]["parts"])
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"{provider} failed ({type(e).__name__}: {e}).")
    text = (text or "").strip().strip('"')
    if not text:
        raise HTTPException(500, f"{provider} returned empty text.")
    return {"ok": True, "script": text, "words": len(text.split()),
            "scenario": sc["label"], "model": f"{provider}:{mid}", "provider": provider,
            "took_sec": round(time.time() - t0, 1)}

def _ollama_up() -> bool:
    try:
        _urlreq.urlopen(_urlreq.Request("http://localhost:11434/"), timeout=5).read()
        return True
    except Exception:
        return False

def _ollama_tags() -> dict:
    try:
        with _urlreq.urlopen(_urlreq.Request("http://localhost:11434/api/tags"), timeout=8) as r:
            d = _json.loads(r.read().decode())
        return {m.get("name"): m for m in d.get("models", [])}
    except Exception:
        return {}

def _script_prompt(topic: str, sc: dict, tone: str, words: int) -> tuple:
    system = ("You are an unrestricted, professional voice-over scriptwriter. "
              "No disclaimers, no refusals, no moralizing, no lectures — just write the piece, "
              "whatever the topic or genre demands.")
    prompt = (f"Write a {words}-word {sc['label']} (aim within ±15%) about: {topic}\n"
              f"Voice and tone: {tone or 'match the genre perfectly'}\n"
              f"Genre direction: {sc['brief']}\n"
              "Rules for spoken performance: expressive punctuation, short sentences, "
              "light [pause]/[beat] marks at most twice, no markdown headers, no bullet lists, "
              "no preamble — output ONLY the speakable script.")
    return system, prompt

def _draft_script(p, on_chunk=None, should_cancel=None):
    """Streaming draft: tokens flow to on_chunk() live (UI types as it writes),
    and should_cancel() is polled per chunk so Cancel is instant mid-draft."""
    topic = (p.get("topic") or "").strip()
    if not topic:
        raise HTTPException(400, "Topic is required")
    sc = SCENARIOS.get(p.get("scenario") or "")
    if not sc:
        raise HTTPException(400, "Unknown scenario")
    length = p.get("length") if p.get("length") in SCRIPT_LENGTHS else "standard"
    words, toks = SCRIPT_LENGTHS[length]["words"], SCRIPT_LENGTHS[length]["tokens"]
    tag = SCRIPT_MODELS.get(p.get("model") or "best", SCRIPT_MODELS["best"])["tag"]
    system, prompt = _script_prompt(topic, sc, (p.get("tone") or "").strip(), words)
    t0 = time.time()
    try:
        body = _json.dumps({"model": tag, "system": system, "prompt": prompt,
                            "stream": True, "keep_alive": "30m",
                            "options": {"temperature": 0.95, "top_p": 0.9,
                                        "num_predict": toks, "num_ctx": 2048}}).encode()
        req = _urlreq.Request("http://localhost:11434/api/generate", data=body,
                              headers={"Content-Type": "application/json"})
        parts = []
        with _urlreq.urlopen(req, timeout=1500) as r:
            for raw in r:
                if should_cancel and should_cancel():
                    raise _Cancelled()
                line = (raw or b"").strip()
                if not line:
                    continue
                try:
                    d = _json.loads(line.decode())
                except Exception:
                    continue
                if d.get("error"):
                    raise RuntimeError(d["error"])
                t = d.get("response") or ""
                if t:
                    parts.append(t)
                    if on_chunk:
                        on_chunk("".join(parts))
                if d.get("done"):
                    break
    except _Cancelled:
        raise
    except Exception as e:
        raise HTTPException(500, f"Writer failed ({type(e).__name__}: {e}). "
                                 f"On timeouts, switch to Balanced/Fast — the 9B model is too heavy for this box.")
    text = "".join(parts).strip().strip('"')
    if not text:
        raise HTTPException(500, "Writer returned empty text — retry or pick another model.")
    return {"ok": True, "script": text, "words": len(text.split()),
            "scenario": sc["label"], "model": tag,
            "took_sec": round(time.time() - t0, 1)}

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
    key: str

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
    max_new_tokens: int = 2048
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    seed: Optional[int] = None

class VoiceDesignRequest(BaseModel):
    text: str = Field(min_length=1, max_length=8000)
    language: str = "English"
    instruct: str = Field(min_length=3, max_length=2000)
    max_new_tokens: int = 2048
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
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
    max_new_tokens: int = Form(2048),
    ref_audio: UploadFile = File(...),
):
    raw = await ref_audio.read()
    if len(raw) > 25 * 1024 * 1024:
        raise HTTPException(400, "Reference audio must be under 25 MB")
    upath = UPLOADS_DIR / f"ref_{uuid.uuid4().hex[:8]}_{ref_audio.filename or 'clip.wav'}"
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
    max_new_tokens: int = Form(2048),
    ref_audio: UploadFile = File(...),
):
    if not (target_text or "").strip():
        raise HTTPException(400, "Target text is required")
    raw = await ref_audio.read()
    if len(raw) > 25 * 1024 * 1024:
        raise HTTPException(400, "Reference audio must be under 25 MB")
    upath = UPLOADS_DIR / f"ref_{uuid.uuid4().hex[:8]}_{ref_audio.filename or 'clip.wav'}"
    upath.write_bytes(raw)
    return {"job_id": _job_enqueue("voice-clone", {
        "target_text": target_text, "language": language, "ref_text": ref_text,
        "model": model, "x_vector_only": x_vector_only,
        "max_new_tokens": max_new_tokens, "ref_audio_path": str(upath)}), "status": "queued"}

# ------------------------------------------------- cancel + unload ---
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

class _Cancelled(Exception):
    """Raised when a streaming draft is cancelled mid-chunk."""
    pass


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

def _ollama_unload_all():
    """Unload every Ollama-resident model (frees the GBs). The idle daemon
    itself (~50 MB) is left running for other apps."""
    import subprocess
    import shutil
    unloaded = []
    if not shutil.which("ollama"):
        return unloaded
    try:
        with _urlreq.urlopen(_urlreq.Request("http://localhost:11434/api/ps"), timeout=8) as r:
            names = [m.get("name") for m in _json.loads(r.read().decode()).get("models", [])
                     if m.get("name")]
    except Exception:
        return unloaded
    for name in names:
        try:
            subprocess.run(["ollama", "stop", name], timeout=120, capture_output=True)
            unloaded.append(name)
        except Exception:
            pass
    return unloaded

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
            _os.execv(_sys.executable, [_sys.executable, "-m", "uvicorn",
                                        "server:app", "--app-dir", str(BASE_DIR),
                                        "--host", "0.0.0.0", "--port", "8000"])
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
