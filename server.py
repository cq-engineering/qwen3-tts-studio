"""
Qwen3-TTS Studio — FastAPI backend (thin entrypoint).

The implementation now lives in the ``app`` package (single-responsibility
modules); this file only wires it together and keeps the ``uvicorn server:app``
contract used by start.sh, the Windows launcher, and the UI restart endpoint.

Two ways to synthesize:
  * POST /api/synthesize/*  — blocking, returns WAV when done (for scripts)
  * POST /api/jobs/*        — async, returns job_id immediately; poll
                              GET /api/jobs/{id} for live phase/progress.
Jobs run on a SINGLE background worker so concurrent generations can never
OOM the box (1.7B bf16 ≈ 3.5 GB resident).

Modules: app.config (paths/constants) · app.errlog · app.engine (model+synth)
· app.calibration · app.writers (Ollama + cloud) · app.jobs (queue/worker)
· app.schemas · app.routers.* (endpoints) · app.boot (boot timing).
"""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app import boot  # captures BOOT_T0 early, before the heavy engine import

from app.config import OUTPUTS_DIR, STATIC_DIR  # noqa: E402
from app.engine import _backend_info  # noqa: E402
from app.errlog import _log  # noqa: E402
from app.routers import health, script, synth, system  # noqa: E402

# Back-compat re-exports (some tooling / docs reference these via `import server`)
from app.config import (LANGUAGES, MODELS, SCENARIOS, SCRIPT_CLOUD,  # noqa: E402,F401
                        SCRIPT_LENGTHS, SCRIPT_MODELS, VOICES)
from app.jobs import _job_enqueue, _job_public, _jobs  # noqa: E402,F401


@asynccontextmanager
async def lifespan(app):
    actual = boot.mark_boot()
    _log.info("boot complete in %ss (avg %s over %d runs)",
              actual, boot.BOOT_INFO["average_sec"], boot.BOOT_INFO["samples"])
    yield


app = FastAPI(title="Qwen3-TTS Studio", version="1.1.0", lifespan=lifespan)
_log.info("server boot (backend=%s)", _backend_info)

app.include_router(health.router)
app.include_router(synth.router)
app.include_router(script.router)
app.include_router(system.router)

# static + outputs
app.mount("/outputs", StaticFiles(directory=str(OUTPUTS_DIR)), name="outputs")
app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")
