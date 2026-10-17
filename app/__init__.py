"""Qwen3-TTS Studio backend package.

Modules
-------
config       paths + all static constants (voices, models, phases, script tables)
errlog       rotating error-log setup (errors.log)
engine       backend detection, model cache/loading, WAV save, synth cores
calibration  self-calibrating progress estimates (calib.json)
writers      script writer: local Ollama + cloud (Groq/Gemini/Grok/Claude)
jobs         job store, single worker, stitch, dialogue, cancel/unload
schemas      Pydantic request models
routers      FastAPI route modules

`server.py` is the thin entrypoint that wires these together; it keeps the
`uvicorn server:app` contract and re-exports the public names (so
`import server; server.SCENARIOS` etc. keep working).
"""
