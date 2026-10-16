# AGENTS.md — qwen3-tts_studio

**Path:** `/home/stigg/qwen3-tts_studio/` · **Port 8000 is permanent** (hardcoded in `start.sh`, `server.py` restart endpoint, Windows launcher URL, docs — change all or none). Not a git repo.

## Quickstart

```bash
cd qwen3-tts_studio
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu   # CPU machines only
./start.sh          # uvicorn on http://localhost:8000
```

- GPU machine? Skip the torch line — CUDA is auto-detected.
- First `Generate` warms the model (~2 min CPU). Subsequent calls use the cache.
- Weights auto-download from HuggingFace on first use per `hf_id` — needs network + several GB disk; a stall on first Generate usually means downloading, check `server.log`.

## Layout (what's real)

| Path | Purpose |
|---|---|
| `server.py` | Sole backend (~54KB single file): FastAPI, lazy model loading, job queue, static hosting |
| `static/index.html` | Entire frontend (~85KB single file, 4 tabs: Studio / Design / Clone / Script) |
| `assets/` | `make_logo.py` → `logo.png`/`icon.ico`; `assets/windows/` = Edge launcher `.bat` + shortcut `.ps1` (installed copies: `C:\Users\stigg\qwen3-tts_studio\`, `qwen3-tts Studio.lnk` on Desktop) |
| Script writer backends | Local Ollama default (`SCRIPT_MODELS`, UI default `darkidol`) + optional free-tier Groq/Gemini (`SCRIPT_CLOUD`). Cloud keys: `keys.json` (chmod 600, values never served) or `GROQ_API_KEY`/`GEMINI_API_KEY` env. Cloud failures auto-fall-back to local Dark Idol. |
| `outputs/` / `uploads/` | Served WAVs (`/outputs/*.wav`) / temp reference-audio uploads |
| Repo hygiene | `LICENSE` (Apache-2.0), `.gitignore` (excludes `keys.json`, WAVs, logs, runtime JSON), `smoke.sh` (endpoint smoke test — run with server up) |
| `calib.json` | Self-tuning synth-time estimates (auto-written, learn from real jobs) |
| `server.log` | Uvicorn log (tail for OOM / load errors) |

No tests, no lint, no CI in this project. Verify by hand: `curl /api/health`, submit a job, check `server.log`.

## API (verified against `server.py` routes)

- `GET /api/health` · `GET /api/voices` · `GET /api/history` · `DELETE /api/history/{f}` (health includes `boot`: actual/average/estimated sec + sample count, persisted in `boot_times.json`)
- Blocking (scripts): `POST /api/synthesize/custom-voice|voice-design|voice-clone`
- Async (UI path): `POST /api/jobs/*` → 202 `{job_id}`, poll `GET /api/jobs/{id}` for phases `queued → loading_model → synthesizing → saving → done|error|cancelled` + `percent`, `elapsed_sec`, `queue_position`, script `partial`, chunked `parts_done`/`total_parts`. **Single worker (`max_workers=1`) — jobs serialize, never concurrent.** `POST /api/jobs/chunked` = Stream mode (sentence-split, per-part playback, real parts-done progress, stitched WAV; single-sentence input auto-falls-back to a normal job). Past-estimate runs get `overtime:true` + UI note instead of a frozen 99%.
- `POST /api/jobs/{id}/cancel?unload=true` — graceful (phase-boundary stop, partial discarded). Queued drops instantly.
- `POST /api/models/unload[?model=]` — purges weights, reports real `freed_mb` from RSS. Safe mid-synthesis (in-flight calls hold their own ref).
- `POST /api/server/restart` — 200s first, then `execv`s a fresh process rebinding `:8000`. Old jobs vanish; UI auto-reconnects.
- `GET /api/script/models` (Ollama models + `fits_ram` + `ollama_up`, scenarios, cloud `providers` with key URLs) · `POST /api/jobs/script`
- `POST /api/script/keys` + `DELETE /api/script/keys/{provider}` — cloud key save/forget (values never returned).

## Gotchas earned the hard way

- **Port 8000 is shared with `repocast`.** Run one at a time — `ss -tlnp | grep 8000` before starting either, or you get "address in use".
- **Never `pkill -f "uvicorn server:app"`** with that literal string anywhere in your own command — it matches your own shell and SIGTERMs you. Use `P='[u]vicorn'; pkill -f "$P server:app"` or kill by PID.
- **venv is not relocatable**: moving the project dir breaks every `.venv/bin/*` shebang + `activate`. Fix with `sed -i 's|OLD|NEW|g'` across `.venv/bin/` (uvicorn's shebang especially), then verify `./.venv/bin/uvicorn --version`.
- **Detached server dies silently between sessions.** Check `curl /api/health` before assuming it's up. Start: `setsid nohup ./start.sh > server.log 2>&1 < /dev/null & disown`.
- **0.6B CustomVoice ignores `instruct`** — UI disables Style on 0.6B. Don't "fix" this; it matches the official model table.
- **CPU inference ≈ 0.04× realtime** (1.7B ≈ 2 min/sentence). 9B Ollama drafts time out on 7.8 GB RAM — default writer is Dark Idol; don't switch default back to 9B on this box.
- **SoX missing warning on import is benign.** Do not install it.
- **qwen-tts has no streaming/token hooks** (`non_streaming_mode` is simulated) — the `%` is a calibrated estimate from `calib.json`, phases are the real state. Don't try to wire per-token progress.
- Script jobs **park TTS weights first** (RAM fit), so `loaded_models` empties after drafting — expected, they reload on next Generate.
- Cloud model IDs go stale (providers rename them; current IDs date to Oct 2026, flagged in `server.py`). A sudden cloud failure with model-not-found means check docs and edit `SCRIPT_CLOUD`.
- Frontend JS can be syntax-checked: extract last `<script>` block, `node --check`. Balance check alone once missed a bad edit.
- `.ps1`/`.bat` for Windows must be **ASCII + CRLF** — PowerShell 5.1 chokes on UTF-8 em-dashes.
- `assets/windows/launcher.bat` rules (all learned from real failures): NO `goto` (jumping out of blocks corrupts the parser), NO `timeout`/`ping` waits (return instantly with no stdin — use PowerShell `Start-Sleep`), NO parens inside echo text in blocks, launch Edge by **full path** (`start msedge` → Access denied, bare `msedge` unresolvable in scripts). Services run FOREGROUND in persistent consoles — never daemonize through one-shot `wsl.exe` calls, and never pass quoted compound commands to `wsl.exe` (quotes get stripped); ship a script file (`run-server-fg.sh`) and invoke it bare.
- Empty `text` → instant 422 (Pydantic `min_length`); missing clone `ref_text` without x-vector → 400. Both surface in the status bar, no toast-only paths.
- `edit` tool fails on curly-quote/Unicode mismatches — fall back to python `re.sub` for those lines.
