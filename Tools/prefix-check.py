#!/usr/bin/env python3
"""Where do `sn ref` and the reference forward stop agreeing? Sweep prefix-truncated checkpoints.

Docs/design-plan.md §1.1: the trunk is an execution list over physical blocks and every cache is per
execution, so the hidden state after execution K-1 depends on nothing beyond blocks 0..K-1. For each
K, Tools/truncate.py writes a model that *is* that prefix, and this script compares the two engines'
final hidden states (`result_norm` from the reference via Tools/sn-oracle, `h_final` from `sn ref`).

    cos(K) ~ 1 up to some K and ~0 above it  -> the defect is inside block K
    cos(K) ~ 1 for K <= 28 and K = 52        -> the defect is in blocks 28..31 (the loop exit)

    Tools/prefix-check.py --tokens Docs/golden/tokens/p1-capital.ids --ks 5-28,52
    Tools/prefix-check.py --ks 12-14 --prompt p1-capital   # (reads the corpus token files)
"""
import argparse
import math
import os
import subprocess
import sys

DEF_MODEL = "/home/justin/Documents/AI/Models/SenseLoop/SenseNova-Loop-9B-Q4_K_M.gguf"
HERE = os.path.dirname(os.path.abspath(__file__))


def vec(path):
    with open(path) as f:
        return [float(l) for l in f if not l.startswith("#") and l.strip()]


def score(a, b):
    num = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return (num / (na * nb) if na and nb else float("nan"),
            na / math.sqrt(len(a)), nb / math.sqrt(len(b)),
            max(abs(x - y) for x, y in zip(a, b)))


def expand(spec):
    out = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-")
            out.extend(range(int(lo), int(hi) + 1))
        else:
            out.append(int(part))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEF_MODEL)
    ap.add_argument("--tokens", default="Docs/golden/tokens/p1-capital.ids")
    ap.add_argument("--ks", default="5,8,12,16,20,24,28,52")
    ap.add_argument("--work", default="/dev/shm/prefix")
    ap.add_argument("--oracle", default=os.path.join(HERE, "sn-oracle"))
    ap.add_argument("--sn", default="Engine/target/release/sn")
    ap.add_argument("--keep-gguf", action="store_true")
    ap.add_argument("--sn-flags", default="", help="extra flags for `sn ref` (e.g. --attn-f16)")
    ap.add_argument("--all-tokens", action="store_true",
                    help="feed the whole token list instead of only the first id, and report every "
                         "position. Position 0 runs with empty caches, so anything that only shows "
                         "up once a cache is non-empty (GDN decay, conv window, KV) is invisible "
                         "there - this flag makes it visible.")
    ap.add_argument("--tokens-per", type=int, default=0, help="truncate the token list")
    args = ap.parse_args()

    toks = open(args.tokens).read()
    os.makedirs(args.work, exist_ok=True)
    ids = toks.replace(",", " ").split()
    if not args.all_tokens:
        # only the first token matters for the localisation (position 0 runs with empty caches),
        # and one token keeps every run cheap
        ids = ids[:1]
    if args.tokens_per:
        ids = ids[: args.tokens_per]
    single = os.path.join(args.work, "feed.ids")
    open(single, "w").write(",".join(ids) + "\n")

    print(f"# K  cos(ref,sn)      rms_ref   rms_sn    max|d|   note")
    bad = None
    for k in expand(args.ks):
        gg = os.path.join(args.work, f"p{k}.gguf")
        cmd = [sys.executable, os.path.join(HERE, "truncate.py"), "--model", args.model,
               "--out", gg, "--blocks", str(k)]
        if k == 52:
            cmd += ["--loop-count", "2"]
        if subprocess.run(cmd, capture_output=True).returncode:
            print(f"{k:>4}  truncate failed"); continue
        o = os.path.join(args.work, f"orc{k}")
        s = os.path.join(args.work, f"sn{k}")
        os.makedirs(o, exist_ok=True)
        os.makedirs(s, exist_ok=True)
        for d in (o, s):
            for stale in os.listdir(d):
                os.remove(os.path.join(d, stale))
        rc = subprocess.run([args.oracle, "--model", gg, "--layers", "", "--no-logits", "--out", o, single],
                            capture_output=True)
        if rc.returncode:
            print(f"{k:>4}  reference failed: {rc.stderr.decode().strip().splitlines()[-1:]}")
            continue
        rc = subprocess.run([args.sn, "ref", "--model", gg, "--tokens-file", single,
                             "--dump", s, "--dump-only", "h_final", "--json"]
                            + args.sn_flags.split(),
                            capture_output=True)
        if rc.returncode:
            print(f"{k:>4}  sn ref failed: {rc.stderr.decode().strip().splitlines()[-1:]}")
            continue
        a = vec(os.path.join(o, "result_norm.p0.txt"))
        if args.all_tokens:
            # the reference writes one file per position; report the worst agreement of the run
            worst = (0, 2.0)
            for t in range(len(ids)):
                c, _, _, _ = score(vec(os.path.join(o, f"result_norm.p{t}.txt")),
                                   vec(os.path.join(s, f"h_final.{t}.txt")))
                if c < worst[1]:
                    worst = (t, c)
            print(f"{k:>4}  worst cos {worst[1]:+.6f} at position {worst[0]} of {len(ids)}"
                  f"   {'ok' if worst[1] > 0.99 else 'DIVERGES'}")
            if worst[1] <= 0.99 and bad is None:
                bad = k
            continue
        b = vec(os.path.join(s, "h_final.0.txt"))
        cos, ra, rb, md = score(a, b)
        note = "ok" if cos > 0.99 else ("DIVERGES" if bad is None else "diverges")
        if cos <= 0.99 and bad is None:
            bad = k
        print(f"{k:>4}  {cos:+.6f}  {ra:8.4f} {rb:8.4f} {md:8.4f}   {note}")
        if not args.keep_gguf:
            os.remove(gg)
    print(f"# first diverging prefix: {bad}")
    return 0 if bad is None else 1


if __name__ == "__main__":
    sys.exit(main())
