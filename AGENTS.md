# AGENTS.md — qwen3-tts_studio

**Path:** `/home/stigg/qwen3-tts_studio/` · **Port 8000 is permanent** (hardcoded in `start.sh`, `server.py` restart endpoint, Windows launcher URL, docs — change all or none). **Git repo** (branch `master`); tag **`pre-refactor`** = the last single-file version before the modular split — `git reset --hard pre-refactor` rolls the refactor back.

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
| `server.py` | Thin entrypoint (~57 lines): creates the `FastAPI` app, wires the routers + static mounts, keeps the `uvicorn server:app` contract |
| `app/` | Backend package — `config` (constants) · `errlog` · `errors` · `engine` (models+synth) · `calibration` · `writers` (Ollama+cloud) · `jobs` (queue/worker) · `schemas` · `boot` · `routers/{health,synth,script,system}` |
| `static/index.html` | Markup only (~375 lines, 5 tabs: Studio / Design / Clone / Script / Duet) |
| `static/js/` | Frontend logic split into ordered classic scripts: `core · jobs · script · duet · media · boot` (shared global scope — inline `onclick` handlers still work; `boot.js` calls `init()` last) |
| `assets/` | `make_logo.py` → `logo.png`/`icon.ico`; `assets/windows/` = Edge launcher `launcher.bat` + health waiter `wait-health.ps1` + shortcut `install-shortcut.ps1` (installed copies: `C:\Users\stigg\qwen3-tts_studio\`, `qwen3-tts Studio.lnk` on Desktop) |
| Script writer backends | Local Ollama default (`SCRIPT_MODELS`, UI default `darkidol`) + optional cloud `SCRIPT_CLOUD` providers: **groq**, **gemini**, **grok** (xAI, OpenAI-compatible), **claude** (Anthropic Messages API, `x-api-key` + `anthropic-version` header). Cloud keys: `keys.json` (chmod 600, values never served) or `GROQ_API_KEY`/`GEMINI_API_KEY`/`XAI_API_KEY`/`ANTHROPIC_API_KEY` env. Cloud failures auto-fall-back to local Dark Idol. |
| `outputs/` / `uploads/` | Served WAVs (`/outputs/*.wav`) / temp reference-audio uploads |
| Repo hygiene | `LICENSE` (Apache-2.0), `.gitignore` (excludes `keys.json`, WAVs, logs, runtime JSON), `smoke.sh` (endpoint smoke test — run with server up) |
| `calib.json` | Self-tuning synth-time estimates (auto-written, learn from real jobs) |
| `server.log` | Uvicorn log (tail for OOM / load errors) |

No tests, no lint, no CI in this project. Verify by hand: `curl /api/health`, submit a job, check `server.log`.

## API (verified against `app/routers/*.py`)

- `GET /api/health` · `GET /api/voices` · `GET /api/history` · `DELETE /api/history/{f}` (health includes `boot`: actual/average/estimated sec + sample count, persisted in `boot_times.json`)
- Blocking (scripts): `POST /api/synthesize/custom-voice|voice-design|voice-clone`
- Async (UI path): `POST /api/jobs/*` → 202 `{job_id}`, poll `GET /api/jobs/{id}` for phases `queued → loading_model → synthesizing → saving → done|error|cancelled` + `percent`, `elapsed_sec`, `queue_position`, script `partial`, chunked `parts_done`/`total_parts`. **Single worker (`max_workers=1`) — jobs serialize, never concurrent.** `POST /api/jobs/chunked` = Stream mode (sentence-split, per-part playback, real parts-done progress, stitched WAV; single-sentence input auto-falls-back to a normal job). Past-estimate runs get `overtime:true` + UI note instead of a frozen 99%.
- `POST /api/jobs/{id}/cancel?unload=true` — graceful (phase-boundary stop, partial discarded). Queued drops instantly.
- `POST /api/models/unload[?model=]` — purges weights, reports real `freed_mb` from RSS. Safe mid-synthesis (in-flight calls hold their own ref).
- `POST /api/server/restart` — 200s first, then `execv`s a fresh process rebinding `:8000`. Old jobs vanish; UI auto-reconnects.
- `GET /api/script/models` (Ollama models + `fits_ram` + `ollama_up`, scenarios, cloud `providers` with key URLs) · `POST /api/jobs/script`
- `POST /api/jobs/dialogue` (two-speaker duet: `A:`/`B:` lines → per-line synth + stitched file + playable segments, max 24 lines) · `POST /api/stitch` (combine up to 20 library WAVs, same-rate only)
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
- Cloud model IDs go stale (providers rename them; current IDs date to Oct 2026, flagged in `app/config.py`). A sudden cloud failure with model-not-found means check docs and edit `SCRIPT_CLOUD`.
- Frontend JS is split across `static/js/*.js` (classic scripts, ordered, shared global scope) — syntax-check each with `node --check static/js/*.js`, or concatenate in DOM order and check. Balance check alone once missed a bad edit.
- `.ps1`/`.bat` for Windows must be **ASCII + CRLF** — PowerShell 5.1 chokes on UTF-8 em-dashes.
- `assets/windows/launcher.bat` rules (all learned from real failures): NO `goto` (jumping out of blocks corrupts the parser), NO `timeout`/`ping` waits (return instantly with no stdin — use PowerShell `Start-Sleep`), NO parens inside echo text in blocks, launch Edge by **full path** (`start msedge` → Access denied, bare `msedge` unresolvable in scripts). Services run FOREGROUND in persistent consoles — never daemonize through one-shot `wsl.exe` calls, and never pass quoted compound commands to `wsl.exe` (quotes get stripped); ship a script file (`run-server-fg.sh`) and invoke it bare. The Ollama/backend wait is delegated to `wait-health.ps1` (poll the endpoint until HTTP 200 → `exit 0`, else `exit 1` after a timeout) so Edge opens the instant the API is really up — no fixed batch loop.
- Empty `text` → instant 422 (Pydantic `min_length`); missing clone `ref_text` without x-vector → 400. Both surface in the status bar, no toast-only paths.
- Uploads: 25 MB cap, client filenames reduced to a safe basename (`Path(name).name`), reference clips deleted in a `finally` after synthesis — `uploads/` can't fill or escape.
- Sampling params are range-validated (`max_new_tokens` 128–4096, temperature 0–2, top-p 0–1, top-k 1–500) — the UI can't push the worker into hour-long runs.
- No auth by design (single-user local app): restart/shutdown/unload are open endpoints. Never expose to the internet; lock down with `BIND_HOST=127.0.0.1` (`start.sh` + restart endpoint honor `BIND_HOST`/`PORT` env).
- `edit` tool fails on curly-quote/Unicode mismatches — fall back to python `re.sub` for those lines.
