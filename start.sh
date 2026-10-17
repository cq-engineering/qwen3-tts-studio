#!/bin/bash
# Qwen3-TTS Studio launcher
export PYTHONUNBUFFERED=1  # unbuffered logs: crashes leave evidence in server.log instead of dying silent
cd "$(dirname "$0")"
if [ -d ".venv" ]; then source .venv/bin/activate; fi
python -c "import qwen_tts" 2>/dev/null || pip install -r requirements.txt
# torch CPU (skip if already present / GPU present)
python -c "import torch" 2>/dev/null || pip install torch --index-url https://download.pytorch.org/whl/cpu
exec uvicorn server:app --host "${BIND_HOST:-0.0.0.0}" --port "${PORT:-8000}"
