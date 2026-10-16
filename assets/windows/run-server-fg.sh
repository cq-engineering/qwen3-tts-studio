#!/bin/bash
# Foreground server entry for the Windows launcher (which keeps this console
# alive - that living parent is what keeps the server alive, see launcher.bat).
# Run via: wsl.exe -d Ubuntu -- ~/qwen3-tts_studio/assets/windows/run-server-fg.sh
# NOTE: quote-free invocation only. Never background here - foreground on purpose.
cd ~/qwen3-tts_studio || exit 1
exec ./start.sh
