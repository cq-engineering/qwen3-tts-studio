@echo off
rem Qwen3-TTSI launcher - starts Ollama + backend, then opens Edge.
rem Each service runs FOREGROUND inside its own persistent minimized console:
rem a live parent console is what keeps WSL processes alive (background
rem daemons spawned from one-shot wsl.exe calls die silently with it).
rem Close a service console to stop that service. Logs to launcher.log.
rem The health wait lives in wait-health.ps1, which returns the moment the
rem endpoint answers 200 - so Edge opens as soon as the API is really up
rem instead of spinning a fixed batch loop for up to 200 seconds.
rem Batch rules observed here: no goto, no timeout/ping, no parens in echos.
setlocal EnableDelayedExpansion
set URL=http://localhost:8000
set LOG=%USERPROFILE%\qwen3-tts_studio\launcher.log
set WAIT=%USERPROFILE%\qwen3-tts_studio\wait-health.ps1
set OK_OLLAMA=1
set OK_BACKEND=1
echo [%date% %time%] launcher started >> "%LOG%"
curl -s -m 3 http://127.0.0.1:11434/ >> "%LOG%" 2>&1
if !errorlevel! neq 0 (
  set OK_OLLAMA=0
  echo Starting Ollama writer service in background window...
  echo [%date% %time%] ollama down, starting... >> "%LOG%"
  start "Qwen3-TTSI Ollama" /min cmd /c wsl.exe -d Ubuntu -- ollama serve
  powershell.exe -NoProfile -ExecutionPolicy Bypass -File "!WAIT!" http://127.0.0.1:11434/ 90 >nul 2>&1
  if !errorlevel! equ 0 set OK_OLLAMA=1
  if !OK_OLLAMA! equ 0 echo [%date% %time%] WARNING ollama still down, continuing >> "%LOG%"
)
curl -s -m 3 %URL%/api/health >> "%LOG%" 2>&1
if !errorlevel! neq 0 (
  set OK_BACKEND=0
  echo Starting Qwen3-TTSI backend - first boot can take about 60 seconds...
  echo [%date% %time%] backend down, starting... >> "%LOG%"
  start "Qwen3-TTSI Server" /min cmd /c wsl.exe -d Ubuntu -- ~/qwen3-tts_studio/assets/windows/run-server-fg.sh
  powershell.exe -NoProfile -ExecutionPolicy Bypass -File "!WAIT!" %URL%/api/health 180 >nul 2>&1
  if !errorlevel! equ 0 set OK_BACKEND=1
)
if !OK_BACKEND! equ 0 (
  echo [%date% %time%] FAILED - see launcher.log >> "%LOG%"
  echo Server did not start. Details in launcher.log in the qwen3-tts_studio folder.
  pause
  exit /b 1
)
echo [%date% %time%] opening Edge >> "%LOG%"
if exist "C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe" (
  "C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe" "%URL%/"
) else (
  if exist "C:\Program Files\Microsoft\Edge\Application\msedge.exe" (
    "C:\Program Files\Microsoft\Edge\Application\msedge.exe" "%URL%/"
  ) else (
    start "" "%URL%/"
  )
)
exit /b 0
