"""Self-calibrating progress estimation.

qwen-tts exposes no per-token hooks, so the live % is an estimate: wall-seconds
per input char, learned from real completed jobs on THIS machine and persisted
in calib.json. Phases are real backend state; only the within-phase % is
estimated. Also holds the per-kind helpers used to size an estimate.
"""
from __future__ import annotations

import json as _json
import threading

from app.config import CALIB_PATH
from app.engine import _model_cache

_calib_lock = threading.Lock()
_calib = {"sec_per_char": {}, "runs": {}, "load_sec": {}}
_CALIB_DEFAULT_SPC = {"custom-1.7B": 1.5, "custom-0.6B": 0.5, "design-1.7B": 1.5,
                      "base-1.7B": 1.5, "base-0.6B": 0.5}


def _calib_load() -> None:
    try:
        data = _json.loads(CALIB_PATH.read_text())
        for k in ("sec_per_char", "runs", "load_sec"):
            if isinstance(data.get(k), dict):
                _calib[k].update({kk: float(v) for kk, v in data[k].items()})
    except Exception:
        pass


def _calib_save() -> None:
    try:
        CALIB_PATH.write_text(_json.dumps(_calib))
    except Exception:
        pass


_calib_load()


def _job_model_key(kind: str, p: dict) -> str:
    return {"custom-voice": p.get("model") or "custom-1.7B",
            "chunked": p.get("model") or "custom-1.7B",
            "dialogue": p.get("model") or "custom-1.7B",
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
