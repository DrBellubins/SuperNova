#!/usr/bin/env bash
# One-branch-at-a-time differential probe: keep a single branch's output projection,
# zero all the others, and ask the reference what that branch contributes.
#   Tools/probe-branch.sh blk.0.ssm_out.weight blk.3.attn_output.weight ...
set -e
M=${MODEL:-/home/justin/Documents/AI/Models/SenseLoop/SenseNova-Loop-9B-Q4_K_M.gguf}
SN=${SN:-Engine/target/release/sn}
TOKENS=${TOKENS:-Docs/golden/tokens/p1-capital.ids}
PROMPT=${PROMPT:-p1-capital}
OUT=${OUT:-.abl}
PORT=${PORT:-8140}
ROLES="attn_output.weight,ssm_out.weight,ffn_down.weight"
for keep in "$@"; do
  tag=$(echo "$keep" | tr '.' '_')
  gg="$OUT/probe-$tag.gguf"
  python3 Tools/ablate.py --model "$M" --out "$gg" --zero "$ROLES" --keep "$keep" >/dev/null
  PORT=$((PORT+1))
  python3 Tools/golden-embd.py --model "$gg" --out-dir "$OUT/probe-embd-$tag" --prompts "$PROMPT" --port "$PORT" >/dev/null 2>&1
  "$SN" ref --model "$gg" --tokens-file "$TOKENS" --dump "$OUT/probe-sn-$tag" --dump-only h_final --json >/dev/null 2>&1
  python3 - "$OUT" "$tag" <<'PY'
import json, gzip, math, sys
out, tag = sys.argv[1], sys.argv[2]
def rd(p):
    with gzip.open(p) as f: return json.load(f)
def sn(p):
    return [float(l) for l in open(p) if not l.startswith("#") and l.strip()]
r = rd(f'{out}/probe-embd-{tag}/{ "p1-capital" }.json.gz')
print(f"### only branch live: {tag.replace('_','.')[:-6]}")
for t in range(len(r['embd'])):
    a = r['embd'][t]; b = sn(f'{out}/probe-sn-{tag}/h_final.{t}.txt')
    num = sum(x*y for x, y in zip(a, b)); da = math.sqrt(sum(x*x for x in a)); db = math.sqrt(sum(y*y for y in b))
    print(f"  pos {t}: cos {num/(da*db):+.5f}  max|d| {max(abs(x-y) for x,y in zip(a,b)):.4f}")
PY
  rm -f "$gg"
done
