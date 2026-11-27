#!/usr/bin/env bash
# Tools/ask.sh — text in, text out. The three steps of one interactive run, chained.
#
#   Tools/ask.sh "The capital of France is" [-n 24] [--ctx 2048] [--raw|--fused] [--kv f16|q8_0|q4_0]
#   Tools/ask.sh -m other.gguf --quiet "1, 2, 3, 4, 5,"
#
# SuperNova has no tokenizer of its own yet (Phase 10, Docs/design-plan.md §7), so this is a thin
# pipe over the parts that do exist:
#
#   Tools/tok    text -> ids      (the checkpoint's own BPE, through the fork's libllama)
#   sn run       ids  -> ids      (prefill + decode on the GPU, greedy)
#   Tools/detok  ids -> text      (the reading half of Tools/tok)
#
# The engine's own report (device, program size, t/s, cache state) goes to stderr, so stdout is
# only the continuation: `Tools/ask.sh ... 2>/dev/null`, or --quiet, to drop it.
#
# Model defaults to $SN_MODEL, then the SenseNova-Loop-9B-Q4_K_M.gguf path in Docs/bench.md.
# Needs the same prerequisites as any Phase 6 run: `cargo build --release`, Tools/{tok,detok},
# and `./Tools/build-kernels.sh` (the kernel image in target/hip/).
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
MODEL=${SN_MODEL:-/home/justin/Documents/AI/Models/SenseLoop/SenseNova-Loop-9B-Q4_K_M.gguf}
SN=$ROOT/target/release/sn
QUIET=0
PROMPT=""
EXTRA=()

while [ $# -gt 0 ]; do
    case "$1" in
        -m | --model) MODEL=${2:?-m needs a path}; shift 2 ;;
        --quiet)      QUIET=1; shift ;;
        --)           shift; while [ $# -gt 0 ]; do PROMPT+="${PROMPT:+ }$1"; shift; done ;;
        --raw | --fused)
                      EXTRA+=("$1"); shift ;;          # boolean switches for sn's fusion ladder
        -*)           EXTRA+=("$1")                    # every other flag forwards its value too
                      if [ $# -gt 1 ] && [ "${2#-}" = "$2" ]; then EXTRA+=("$2"); shift; fi
                      shift ;;
        *)            PROMPT+="${PROMPT:+ }$1"; shift ;;
    esac
done

[ -n "$PROMPT" ] || { echo "usage: Tools/ask.sh \"your prompt\" [-n N] [--ctx C] [--raw|--fused] [--kv K]" >&2; exit 2; }
for b in "$SN" "$ROOT/Tools/tok" "$ROOT/Tools/detok"; do
    [ -x "$b" ] || { echo "missing $b (cargo build --release; gcc Tools/{tok,detok}.c)" >&2; exit 1; }
done
[ -f "$MODEL" ] || { echo "no model at $MODEL (set SN_MODEL)" >&2; exit 1; }

tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT

# one id per line: `sn run --ids` reads whitespace-separated ids. Both tokenizer tools load the
# checkpoint just to read its vocab, and libllama narrates that at length -> keep it in the logs.
"$ROOT/Tools/tok" "$MODEL" <(printf '%s' "$PROMPT") > "$tmp/prompt.ids" 2>"$tmp/tok.log"
n=$(grep -c . "$tmp/prompt.ids")
echo "ask: ${n} prompt tokens via $(basename "$MODEL")" >&2

# the ids go in a file so nothing has to be quoted twice.
"$SN" run --model "$MODEL" --ids "$tmp/prompt.ids" ${EXTRA[@]+"${EXTRA[@]}"} > "$tmp/run.txt" 2>&1
[ "$QUIET" = 1 ] || cat "$tmp/run.txt" >&2

# only the continuation line, so the report's numbers never reach the tokenizer
grep -m1 '^continuation:' "$tmp/run.txt" | cut -d' ' -f2- | "$ROOT/Tools/detok" "$MODEL" 2>"$tmp/detok.log"
