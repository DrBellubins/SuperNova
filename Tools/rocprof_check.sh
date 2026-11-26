#!/usr/bin/env bash
# Cross-check the launch/kernel counts with rocprofv2 instead of the LD_PRELOAD shim.
#
# Two passes over the same prompt: pass 1 warms the KV cache, pass 2 therefore has *no prefill*
# and its dispatches are pure decode. The parser (Tools/rocprof_parse.py) finds the idle gap
# between the two requests and counts what follows it, divided by the decode tokens requested.
set -uo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
FORK=${FORK:-/home/justin/Documents/Github/llama.cpp-opensensenova}
SERVER=${SERVER:-$FORK/build/bin/llama-server}
MODEL=${MODEL:-/home/justin/Documents/AI/Models/SenseLoop/SenseNova-Loop-9B-Q4_K_M.gguf}
OUT=${OUT:-/tmp/rp2}
PORT=${PORT:-8123}
CTXS=${CTXS:-4096}
NDEC=${NDEC:-32}
PROMPT_FILE=${PROMPT_FILE:-$HERE/phase0_prompt.txt}
RP=${RP:-/opt/rocm/bin/rocprofv2}
# TRACE=api counts HIP API calls (cross-check of Tools/hip_count.c); TRACE=kernel counts actual GPU
# dispatches and their names/durations. rocprofv2 here silently drops --kernel-trace when combined
# with --hip-api, so pick one per run.
TRACE=${TRACE:-api}
case "$TRACE" in
    api)   FLAGS="--hip-api" ;;
    kernel) FLAGS="--kernel-trace" ;;
    *)     echo "TRACE must be api or kernel"; exit 1 ;;
esac

command -v "$RP" >/dev/null || { echo "rocprofv2 not found ($RP)"; exit 1; }
# A stale server holding the port makes the profiled one exit immediately, and the requests then
# silently hit the *unprofiled* process.
if ps -C "$(basename "$SERVER")" > /dev/null; then
    echo "a $(basename "$SERVER") is already running - stop it first (it would steal the port)"
    exit 1
fi
rm -rf "$OUT"; mkdir -p "$OUT"

PROMPT=$(cat "$PROMPT_FILE")

echo "== profiling $SERVER under $(basename "$RP") $FLAGS"
# rocprofv2 does not accept a `--` separator: the executable follows the options directly.
"$RP" $FLAGS -d "$OUT" -o supernova \
    "$SERVER" -m "$MODEL" -ngl 99 -c "$CTXS" --host 127.0.0.1 --port "$PORT" -t 1 \
    > "$OUT/server.log" 2>&1 &
RP_PID=$!

# rocprofv2 timestamps come from a different clock than the shell's, but they tick at the same rate,
# so the parser anchors on "first record == process start" and uses these markers for the window.
MONO() { python3 -c 'import time; print(int(time.clock_gettime(time.CLOCK_MONOTONIC)*1e9))'; }
MONA() { echo "$1 $(MONO)"; }
: > "$OUT/markers.txt"
MONA spawn

for i in $(seq 1 240); do
    code=$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:$PORT/health" || echo 000)
    [ "$code" = 200 ] && break
    sleep 1
done
[ "$i" = 240 ] && { echo "server did not come up; see $OUT/server.log"; kill $RP_PID; exit 1; }
echo "   up after ${i}s"

req() { # $1 = tokens to generate, $2 = tag
    curl -s "http://127.0.0.1:$PORT/completion" \
        -H 'Content-Type: application/json' \
        -d "{\"prompt\": $(python3 -c 'import json,sys; print(json.dumps(sys.stdin.read()))' < "$PROMPT_FILE"),
             \"n_predict\": $1, \"temperature\": 0, \"seed\": 42}" \
        -o "$OUT/req-$2.json"
    python3 - "$OUT/req-$2.json" <<'PY'
import json,sys
d=json.load(open(sys.argv[1]))
t=d.get("timings",{})
print(f"   warm={t.get('cache_n',0)} prompt={t.get('prompt_n',0)} pred={t.get('predicted_n',0)} "
      f"pp={t.get('prompt_per_second',0):.1f} t/s tg={t.get('predicted_per_second',0):.2f} t/s")
PY
}

echo "== request 1 (warm/prefill)"; req 4 warm
sleep 2
echo "== request 2 ($NDEC decode tokens, prompt cached)"
MONA req2_start
req "$NDEC" meas
MONA req2_end
sleep 2
kill -INT $RP_PID
for i in $(seq 1 60); do kill -0 $RP_PID 2>/dev/null || break; sleep 1; done
kill -9 $RP_PID 2>/dev/null
ls -la "$OUT"
python3 "$HERE/rocprof_parse.py" "$OUT" "$NDEC" "$OUT/markers.txt"
