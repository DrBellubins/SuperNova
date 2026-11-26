#!/usr/bin/env bash
# phase0.sh — Phase 0 measurement recipe (Docs/design-plan.md §4.4) run against the reference fork.
#
# Passes (PASS=launch|cold|fill|all):
#   launch : one warm prompt, then a measured *decode-only* window (prompt fully cached), run under
#            Tools/hip_count.c (LD_PRELOAD) to count real HIP launches / graph launches / memcpys,
#            plus sched_reserve node count, HIP-graph status, CPU time of the server process and
#            VRAM at n_ctx in {1024, 8192, 32768}.
#   cold   : prompt cache disabled, one unrelated warm-up prompt, then the measured prompt cold ->
#            honest prefill (pp) and decode (tg) token/s.
#   fill   : single n_ctx run whose prompt fills the context (long prompt) -> pp/tg at depth.
#
# Every number it prints is reproducible from the command lines echoed into the run log.
#
# Usage: Tools/phase0.sh [output-dir]        env: CTXS PASS NGL DECODE_TOKENS PROMPT_FILE PORT0
set -u

MODEL=${MODEL:-/home/justin/Documents/AI/Models/SenseLoop/SenseNova-Loop-9B-Q4_K_M.gguf}
FORK=${FORK:-/home/justin/Documents/Github/llama.cpp-opensensenova}
SERVER=${SERVER:-$FORK/build/bin/llama-server}
OUT=${1:-/tmp/phase0}
NGL=${NGL:-99}
DECODE_TOKENS=${DECODE_TOKENS:-32}
PORT0=${PORT0:-8901}
PASS=${PASS:-all}
HERE=$(cd "$(dirname "$0")" && pwd)
PROMPT=${PROMPT_FILE:-$HERE/phase0_prompt.txt}
WARM_PROMPT=$OUT/.warm.txt
mkdir -p "$OUT"
printf 'Explain, in one paragraph, what a memory bandwidth roofline is.\n' > "$WARM_PROMPT"
[ -f "$PROMPT" ] || { echo "missing prompt file $PROMPT" >&2; exit 2; }

if [ ! -f "$OUT/libhipcount.so" ]; then
  gcc -O2 -fPIC -shared -o "$OUT/libhipcount.so" "$HERE/hip_count.c" -ldl || exit 3
fi

req() { # $1 port, $2 outfile, $3 prompt-file, $4 n_predict
  curl -s -H "Content-Type: application/json" -d "$(python3 - "$3" "$4" <<'PY'
import json,sys
print(json.dumps({"prompt": open(sys.argv[1]).read(), "n_predict": int(sys.argv[2]), "temperature": 0, "seed": 42}))
PY
)" "http://127.0.0.1:$1/completion" > "$2"
}

show_timing() { # $1 json
  python3 - "$1" <<'PY'
import json,sys
d=json.load(open(sys.argv[1])); t=d["timings"]
print("TIMINGS prompt_n=%d cached_n=%d pred_n=%d pp=%.2f t/s tg=%.2f t/s ms/token=%.2f"
      % (t["prompt_n"], t["cache_n"], t["predicted_n"], t["prompt_per_second"],
         t["predicted_per_second"], t["predicted_per_token_ms"]))
PY
}

wait_ready() { # $1 port, $2 log
  local port=$1 log=$2 i
  for i in $(seq 1 300); do
    grep -q "listening on http" "$log" 2>/dev/null && return 0
    grep -qiE "failed to (load|initialize)|aborting" "$log" 2>/dev/null && return 1
    sleep 1
  done
  return 1
}

cpu_clocks() { awk '{print $14+$15}' /proc/$1/stat; }   # utime+stime in clock ticks
CLK=$(getconf CLK_TCK)

vram_bytes() {
  /opt/rocm/bin/rocm-smi --showmeminfo vram 2>/dev/null |
    awk '/VRAM Total Used Memory \(B\)/{print $NF}' | head -1
}
gpu_busy() { /opt/rocm/bin/rocm-smi --showuse 2>/dev/null | awk '/GPU use \(%\)/{print $NF}' | head -1; }

instrumented_grep() { # $1 log
  echo "--- sched_reserve / offload / graph status ---"
  grep -hE "sched_reserve: max_nodes|offloaded [0-9]+/[0-9]+ layers|- ROCm[0-9]" "$1" | head -8
  echo "GRAPH_REUSED_LINES=$(grep -c 'CUDA Graph id [0-9]* reused' "$1")"
  echo "GRAPH_WARMUP_COMPLETE=$(grep -c 'CUDA graph warmup complete' "$1")"
  echo "GRAPH_DISABLED_DUE_TO_NODE=$(grep -c 'disabling CUDA graphs' "$1")"
  echo "GRAPH_REINSTANTIATED=$(grep -c 're-instantiating CUDA graph' "$1")"
  echo "GRAPH_UPDATE_FAILED=$(grep -c 'CUDA graph update failed' "$1")"
}

start() { # $1 ctx, $2 port, $3 log, $4 extra args...
  local ctx=$1 port=$2 log=$3; shift 3
  echo "\$ GGML_CUDA_GRAPH_DEBUG=1${LD_PRELOAD:+ LD_PRELOAD=$LD_PRELOAD} $SERVER -m $MODEL -ngl $NGL -c $ctx --port $port --host 127.0.0.1 --temp 0 -s 42 --log-prefix -v $*" >> "$OUT/commands.txt"
  GGML_CUDA_GRAPH_DEBUG=1 "$SERVER" -m "$MODEL" -ngl "$NGL" -c "$ctx" --port "$port" --host 127.0.0.1 \
      --temp 0 -s 42 --log-prefix -v "$@" > "$log" 2>&1 &
  SRV_PID=$!
}

stop() { kill -TERM "${SRV_PID:-0}" 2>/dev/null; wait "${SRV_PID:-0}" 2>/dev/null; sleep 3; SRV_PID=; }

run_pass() { # $1 pass, $2 ctx, $3 prompt-file
  local pass=$1 ctx=$2 pf=$3
  local port=$((PORT0 + ctx % 97))
  local log extra=()
  log="$OUT/$pass-ctx${ctx}.log"
  echo; echo "################ PASS=$pass n_ctx=$ctx prompt=$(wc -w < "$pf") words ################"
  [ "$pass" = cold ] && extra=(--no-cache-prompt)
  [ "$pass" = launch ] && export LD_PRELOAD="$OUT/libhipcount.so" || unset LD_PRELOAD
  start "$ctx" "$port" "$log" "${extra[@]+"${extra[@]}"}"
  if ! wait_ready "$port" "$log"; then
    echo "SERVER FAILED to start for n_ctx=$ctx (see $log)"; stop; return 1
  fi
  echo "--- VRAM after load (bytes) ---"; vram_bytes
  if [ "$pass" = launch ]; then
    req "$port" "$OUT/$pass-ctx${ctx}-warm.json" "$PROMPT" 8          # fills prompt cache + graphs
    sleep 2
    grep -c "HIPCOUNTER" "$log" >/dev/null
    kill -USR1 "$SRV_PID"; sleep 1
    local c0 c1 t0 t1
    c0=$(cpu_clocks "$SRV_PID")
    req "$port" "$OUT/$pass-ctx${ctx}-meas.json" "$PROMPT" "$DECODE_TOKENS"
    c1=$(cpu_clocks "$SRV_PID")
    kill -USR1 "$SRV_PID"; sleep 1
    echo "--- decode-window CPU of server process ---"
    echo "SERVER_CPU_TICKS=$((c1-c0)) over clock=$CLK  (core-seconds=$(python3 -c "print(($c1-$c0)/$CLK)"))"
    show_timing "$OUT/$pass-ctx${ctx}-meas.json"
    echo "--- HIPCOUNTER lines (1st = reset before window, 2nd = end of window) ---"
    grep "HIPCOUNTER" "$log" | tail -3
  else
    req "$port" "$OUT/$pass-ctx${ctx}-warm.json" "$WARM_PROMPT" 4     # unrelated warm-up
    sleep 1
    local c0 c1
    c0=$(cpu_clocks "$SRV_PID")
    req "$port" "$OUT/$pass-ctx${ctx}-meas.json" "$pf" "$DECODE_TOKENS"
    c1=$(cpu_clocks "$SRV_PID")
    echo "--- measured-window CPU of server process ---"
    echo "SERVER_CPU_TICKS=$((c1-c0)) over clock=$CLK  (core-seconds=$(python3 -c "print(($c1-$c0)/$CLK)"))"
    show_timing "$OUT/$pass-ctx${ctx}-meas.json"
  fi
  echo "--- VRAM after request (bytes) / GPU busy % ---"; vram_bytes; gpu_busy
  instrumented_grep "$log"
  unset LD_PRELOAD
  stop
}

FILL_CTX=${FILL_CTX:-8192}
FILL_PROMPT=${FILL_PROMPT:-$HERE/phase0_prompt_8k.txt}

for c in ${CTXS:-1024 8192 32768}; do
  case "$PASS" in
    all)   run_pass launch "$c" "$PROMPT"; run_pass cold "$c" "$PROMPT" ;;
    launch) run_pass launch "$c" "$PROMPT" ;;
    cold)   run_pass cold "$c" "$PROMPT" ;;
    fill)   ;;
  esac
done
if [ "$PASS" = all ] || [ "$PASS" = fill ]; then
  [ -f "$FILL_PROMPT" ] && run_pass fill "$FILL_CTX" "$FILL_PROMPT" || echo "fill prompt missing: $FILL_PROMPT"
fi
echo "DONE. artifacts in $OUT"
