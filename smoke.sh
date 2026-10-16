#!/bin/bash
# Smoke test for qwen3-tts_studio — needs the server running on :8000.
# Usage: ./smoke.sh
set -u
B=http://127.0.0.1:8000
fail() { echo "SMOKE FAIL: $1"; exit 1; }
curl -sf $B/api/health | python3 -c "import json,sys; j=json.load(sys.stdin); assert j['ok'], j" || fail "health"
curl -sf $B/api/voices | python3 -c "import json,sys; j=json.load(sys.stdin); assert len(j['voices'])==9, j; assert 'custom-1.7B' in j['models']" || fail "voices"
curl -sf $B/ | grep -q "voiceSelect" || fail "UI missing voice dropdown"
curl -sf $B/favicon.ico -o /dev/null || fail "favicon"
curl -sf "$B/api/logs/errors?limit=1" | python3 -c "import json,sys; assert 'lines' in json.load(sys.stdin)" || fail "error log endpoint"
curl -sf $B/api/script/models | python3 -c "import json,sys; d=json.load(sys.stdin); assert 'scenarios' in d and 'providers' in d" || fail "script models"
echo "SMOKE OK — all endpoints live"
