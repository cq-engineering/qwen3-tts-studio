"""Process boot timing: server import -> application startup complete.

BOOT_T0 is captured at import of this module (imported early by server.py, before
the heavy engine import triggers torch), and ``mark_boot`` completes the
measurement in the lifespan handler, persisting a rolling sample in
boot_times.json.
"""
from __future__ import annotations

import time
from typing import Any, Dict

from app.config import BASE_DIR

BOOT_T0 = time.time()
BOOT_FILE = BASE_DIR / "boot_times.json"
BOOT_INFO: Dict[str, Any] = {"actual_sec": None, "average_sec": None,
                             "estimated_sec": None, "samples": 0}


def mark_boot() -> float:
    import json as _json
    import statistics
    actual = round(time.time() - BOOT_T0, 1)
    try:
        samples = _json.loads(BOOT_FILE.read_text()) if BOOT_FILE.exists() else []
        samples = [float(x) for x in samples if isinstance(x, (int, float))][-19:] + [actual]
        BOOT_FILE.write_text(_json.dumps(samples))
    except Exception:
        samples = [actual]
    BOOT_INFO.update({"actual_sec": actual,
                      "average_sec": round(statistics.mean(samples), 1),
                      "estimated_sec": round(statistics.mean(samples[-3:]), 1),
                      "samples": len(samples)})
    return actual
