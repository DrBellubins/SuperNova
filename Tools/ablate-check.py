#!/usr/bin/env python3
"""Ablation bisect: which part of the forward is wrong?

For each named set of tensor roles, zero them in a copy of the checkpoint, capture the
reference's per-position hidden states from that model, run `sn ref` on the same model
and prompt, and report the cosine similarity. With a part of the network silenced, a
cosine of 1.0 proves the rest of the pipeline matches the reference exactly, which
narrows a numeric bug to the silenced part. The mHC skeleton was proven this way on
2026-11-26 (zeroing attn_output+ssm_out+ffn_down reproduced the reference to
max|diff| = 0.0).

    Tools/ablate-check.py --model M.gguf --prompt p1-capital \
        --cases attn_output.weight,ssm_out.weight:ffn_down.weight \
                ffn_down.weight:branches --port 8130
"""
import argparse, gzip, json, os, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))


def run(cmd, **kw):
    r = subprocess.run(cmd, capture_output=True, text=True, **kw)
    if r.returncode != 0:
        sys.exit(f"{' '.join(cmd)}\n{r.stdout}\n{r.stderr}")
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompt", default="p1-capital")
    ap.add_argument("--corpus", default="Docs/golden/corpus.json")
    ap.add_argument("--sn", default="target/release/sn")
    ap.add_argument("--server", default="/home/justin/Documents/Github/llama.cpp-opensensenova/build/bin/llama-server")
    ap.add_argument("--port", type=int, default=8130)
    ap.add_argument("--cases", nargs="+", required=True,
                    help="zero_roles:label pairs (roles comma-separated, no spaces)")
    ap.add_argument("--dir", default=".abl")
    args = ap.parse_args()

    os.makedirs(args.dir, exist_ok=True)
    rows = []
    for spec in args.cases:
        roles, _, label = spec.partition(":")
        label = label or roles.replace(".", "_")
        out = os.path.join(args.dir, f"ab-{label}.gguf")
        run([sys.executable, os.path.join(HERE, "ablate.py"), "--model", args.model,
             "--out", out, "--zero", roles])
        run([sys.executable, os.path.join(HERE, "golden-embd.py"), "--model", out,
             "--out-dir", os.path.join(args.dir, f"embd-{label}"), "--prompts", args.prompt,
             "--port", str(args.port), "--corpus", args.corpus])
        run([args.sn, "ref", "--model", out,
             "--tokens-file", f"Docs/golden/tokens/{args.prompt}.ids",
             "--dump", os.path.join(args.dir, f"sn-{label}"), "--dump-only", "h_final"])
        ref = json.loads(gzip.open(os.path.join(args.dir, f"embd-{label}", f"{args.prompt}.json.gz")).read())
        import numpy as np
        cos = []
        for t in range(len(ref["embd"])):
            hv = np.array(ref["embd"][t], np.float32)
            p = os.path.join(args.dir, f"sn-{label}", f"h_final.{t}.txt")
            mv = np.array([float(l) for l in open(p) if not l.startswith("#")], np.float32)
            cos.append(float((hv @ mv) / (np.linalg.norm(hv) * np.linalg.norm(mv))))
        rows.append((label, roles, cos))
        os.remove(out)
        print(f"{label:24s} cos per position: {[round(c, 6) for c in cos]}")

    print("\nverdict: cos = 1.0 means everything *except* the zeroed roles matches the reference")
    for label, roles, cos in rows:
        worst = min(cos)
        print(f"  {label:24s} worst cos {worst:.6f}  {'MATCH' if worst > 0.999 else 'DIFFERS'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
