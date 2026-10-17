"""TTS engine: backend detection, model cache/loading, WAV saving and the
three synth cores shared by the blocking endpoints and the background jobs.

Each synth core returns ``(wavs, sample_rate, info)`` and raises
``HTTPException`` on bad input.
"""
from __future__ import annotations

import time
import uuid
from pathlib import Path
from typing import Any, Dict, Tuple

from fastapi import HTTPException

from app.config import MODELS, OUTPUTS_DIR, VOICES

# ------------------------------------------------------------ backend ---
_model_cache: Dict[str, Any] = {}
_backend_info: Dict[str, Any] = {"torch": None, "cuda": False, "qwen_tts": None, "device": "cpu"}


def _detect_backend() -> None:
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
    # record load time for the self-calibrating progress estimate
    from app.calibration import _calib, _calib_lock, _calib_save
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


# ---------------------------------------------------------- synth cores ---
def _canon_speaker(name: str) -> str:
    canon = {v["id"].lower(): v["id"] for v in VOICES}
    key = (name or "").strip().replace(" ", "_").lower()
    if key not in canon:
        raise HTTPException(400, f"Unknown speaker '{name}'.")
    return canon[key]


def _gen_kwargs(p: dict) -> Dict[str, Any]:
    kw: Dict[str, Any] = {"max_new_tokens": min(4096, max(128, int(p.get("max_new_tokens") or 2048)))}
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
            max_new_tokens=min(4096, max(128, int(p.get("max_new_tokens") or 2048))),
        )
        return wavs, int(sr), {"model": model_key}
    finally:
        try:
            Path(ref_path).unlink(missing_ok=True)  # never hoard uploads
        except Exception:
            pass
