#!/usr/bin/env python3
"""Per-position agreement between `sn ref` and the reference engine, with weights out of the way.

`Tools/prefix-check.py` finds *which block* the two engines stop agreeing in. This finds *how the
agreement behaves along the sequence*, which is what exposed the rope pair-layout defect
(Docs/numerics.md §3): with the position-0 comparison everyone passes, and the disagreement only
appears at positions >= 1 inside a full-attention block.

It builds a prefix-truncated checkpoint (Tools/truncate.py), converts the block weights to f32
(Tools/def32.py) so weight quantisation and the reference's q8_0 activation requantisation stop
being variables, feeds N tokens through both engines, and compares `result_norm` per position.

    Tools/position-check.py --blocks 5 --tokens 12 --prompt p5-loop-pressure \
        --variants "",--rope-adjacent,--attn-f16

`--variants` rows are extra `sn ref` flags; the first column is the plain reference forward.
Run `--keep` to reuse the (large) truncated f32 model for a second pass.
"""
import argparse
import os
import subprocess
import sys

DEF_MODEL = "/home/justin/Documents/AI/Models/SenseLoop/SenseNova-Loop-9B-Q4_K_M.gguf"
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def vec(path):
    return [float(l) for l in open(path) if not l.startswith("#") and l.strip()]


def score(a, b):
    num = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(x * x for x in b) ** 0.5
    return (num / (na * nb) if na and nb else float("nan"),
            max(abs(x - y) for x, y in zip(a, b)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEF_MODEL)
    ap.add_argument("--prompt", default="p5-loop-pressure", help="id under --tokens-dir")
    ap.add_argument("--tokens-dir", default="Docs/golden/tokens")
    ap.add_argument("--tokens", type=int, default=8, help="how many of them to feed")
    ap.add_argument("--blocks", type=int, default=5, help="prefix length (truncate.py --blocks)")
    ap.add_argument("--variants", default="", help="comma list of extra `sn ref` flag sets")
    ap.add_argument("--work", default="/dev/shm/poscheck")
    ap.add_argument("--oracle", default=os.path.join(HERE, "sn-oracle"))
    ap.add_argument("--sn", default="target/release/sn")
    ap.add_argument("--quantised", action="store_true", help="skip def32 (keep quantised weights)")
    ap.add_argument("--keep", action="store_true", help="keep the generated model")
    args = ap.parse_args()

    ids = open(os.path.join(args.tokens_dir, args.prompt + ".ids")).read().replace(",", " ").split()
    ids = ids[: args.tokens]
    os.makedirs(args.work, exist_ok=True)
    feed = os.path.join(args.work, "feed.ids")
    open(feed, "w").write(",".join(ids) + "\n")

    gg = os.path.join(args.work, f"p{args.blocks}.gguf")
    if not os.path.exists(gg):
        cmd = [sys.executable, os.path.join(HERE, "truncate.py"), "--model", args.model,
               "--out", gg, "--blocks", str(args.blocks)]
        subprocess.run(cmd, check=True, capture_output=True)
        if not args.quantised:
            f32 = gg + ".f32.gguf"
            if not os.path.exists(f32):
                subprocess.run([sys.executable, os.path.join(HERE, "def32.py"), "--model", gg,
                                "--out", f32], check=True, capture_output=True)
            os.remove(gg)
            gg = f32
    if args.quantised and not gg.endswith(".gguf"):
        pass

    orc = os.path.join(args.work, "oracle")
    os.makedirs(orc, exist_ok=True)
    for stale in os.listdir(orc):
        os.remove(os.path.join(orc, stale))
    r = subprocess.run([args.oracle, "--model", gg, "--layers", "", "--no-logits", "--out", orc, feed],
                       capture_output=True)
    if r.returncode:
        print("reference failed:", r.stderr.decode().strip().splitlines()[-1:])
        return 1

    # an empty leading entry means "no extra flags" (the plain reference forward)
    variants = args.variants.split(",") if args.variants else [""]

    print(f"# model={os.path.basename(gg)} blocks={args.blocks} tokens={len(ids)} "
          f"{'(quantised)' if args.quantised else '(f32 block weights)'}")
    head = f"{'pos':>4}" + "".join(f"{('cos[' + (v or 'plain') + ']'):>26}" for v in variants)
    print(head)
    worst = {}
    for t in range(len(ids)):
        ref = vec(os.path.join(orc, f"result_norm.p{t}.txt"))
        row = f"{t:>4}"
        for v in variants:
            out = os.path.join(args.work, f"sn-{v or 'plain'}")
            os.makedirs(out, exist_ok=True)
            for stale in os.listdir(out):
                os.remove(os.path.join(out, stale))
            cmd = [args.sn, "ref", "--model", gg, "--tokens-file", feed, "--dump", out,
                   "--dump-only", "h_final", "--json"] + (v.split() if v else [])
            r = subprocess.run(cmd, capture_output=True)
            if r.returncode:
                row += f"{'sn ref failed':>26}"
                continue
            c, d = score(ref, vec(os.path.join(out, f"h_final.{t}.txt")))
            worst[v or "plain"] = min(worst.get(v or "plain", 2.0), c)
            row += f"{c:>18.10f}{d:>8.1e}"
        print(row)
    print("# worst cos per variant: " + ", ".join(f"{k}={v:.10f}" for k, v in worst.items()))
    if not args.keep:
        os.remove(gg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
