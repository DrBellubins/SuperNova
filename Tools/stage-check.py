#!/usr/bin/env python3
"""Compare the reference forward's hidden states against the reference server's.

The reference server exposes the trunk hidden state (the vector the output head reads) through
`/embedding` with `--pooling none --embd-normalize -1` - Tools/golden-embd.py captures that.
`sn ref --dump <dir> --dump-only h_final` writes the same vector per position. Comparing them
localises a numeric problem long before the logits do: a logit difference of 0.1 is meaningless on
its own, but the per-position relative error of the hidden state says which part of the forward is
off, and position 0 is the useful one - at position 0 rope is the identity, attention is a
single-token passthrough and the recurrent state is empty, so the difference is entirely per-token.

  Tools/golden-embd.py --out-dir Docs/golden/embd        # GPU reference (f32 KV)
  Tools/golden-embd.py --ngl 0 --out-dir Docs/golden/embd-cpu   # CPU reference
  Engine/target/release/sn ref ... --dump /tmp/hf --dump-only h_final
  Tools/stage-check.py --ref Docs/golden/embd/p1-capital.json.gz --sn-dir /tmp/hf
  Tools/stage-check.py --ref A.json.gz --ref2 B.json.gz --sn-dir /tmp/hf   # A vs B = the noise floor

Exits non-zero when the worst relative error against --ref exceeds --max-rel (default 1e-2).
"""
import argparse
import gzip
import json
import math
import os
import sys


def read_vals(path):
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            out.append(float(line))
    return out


def stats(a, b):
    n = min(len(a), len(b))
    if n == 0:
        return 0.0, 0.0, 0.0, 0.0
    dot = sum(a[i] * b[i] for i in range(n))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    diff = math.sqrt(sum((a[i] - b[i]) ** 2 for i in range(n)))
    cos = dot / (na * nb) if na and nb else 0.0
    return cos, diff / nb if nb else 0.0, na, max(abs(a[i] - b[i]) for i in range(n))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ref", required=True, help="Docs/golden/embd/<id>.json.gz")
    ap.add_argument("--ref2", help="second reference capture, to measure the reference's own spread")
    ap.add_argument("--sn-dir", help="directory with h_final.<t>.txt from `sn ref --dump`")
    ap.add_argument("--prefix", default="h_final", help="stage dump prefix (default h_final)")
    ap.add_argument("--max-rel", type=float, default=1e-2, help="per-position relative error budget")
    args = ap.parse_args()

    with gzip.open(args.ref) as f:
        ref = json.load(f)
    rows = ref["embd"]
    if args.ref2:
        with gzip.open(args.ref2) as f:
            rows2 = json.load(f)["embd"]
    else:
        rows2 = None

    print(f"positions={len(rows)} width={len(rows[0])} prompt={ref.get('tokens') and len(ref['tokens'])} tokens")
    print(f"{'pos':>4}  {'cos':>10}  {'rel_l2':>9}  {'rms':>9}  {'max|d|':>8}"
          + ("   reference-vs-reference" if rows2 else ""))
    worst = 0.0
    for pos, row in enumerate(rows):
        cols = f"{pos:>4}"
        if args.sn_dir:
            path = os.path.join(args.sn_dir, f"{args.prefix}.{pos}.txt")
            if not os.path.exists(path):
                print(f"{cols}  {path} missing")
                continue
            mine = read_vals(path)
            cos, rel, rms, md = stats(mine, row)
            worst = max(worst, rel)
            cols += f"  {cos:>10.6f}  {rel:>9.5f}  {rms:>9.3f}  {md:>8.4f}"
        if rows2:
            cos, rel, _, md = stats(rows2[pos], row)
            cols += f"   cos {cos:.6f} rel {rel:.5f} max|d| {md:.4f}"
        print(cols)

    if args.sn_dir and worst > args.max_rel:
        print(f"\nFAIL worst relative error {worst:.5f} > {args.max_rel}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
