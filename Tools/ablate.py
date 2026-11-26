#!/usr/bin/env python3
"""Ablation oracle: write a copy of the checkpoint with whole tensor roles zeroed, so
the *reference* server can tell us what a part of the model contributes.

    Tools/ablate.py --model M.gguf --out /tmp/ablate-branches.gguf \
        --zero attn_output,ssm_out,ffn_down

Run the reference (Tools/golden-embd.py --model <out>) and `sn ref --model <out>` on
the same prompt and compare: the difference isolates that part of the network without
trusting either implementation's internals.
"""
import argparse
import struct
import shutil
import sys

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from dequant_ref import read_header  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--zero", help="comma-separated name suffixes to zero")
    ap.add_argument("--one", help="comma-separated name suffixes to set to 1.0 (must be f32)")
    ap.add_argument("--keep", help="comma-separated exact tensor names exempt from --zero")
    ap.add_argument("--onehot", help="suffix:index - write a one-hot along each tensor's fastest "
                                    "dimension, e.g. ssm_conv1d.weight:3 leaves only the current "
                                    "token's conv tap, which removes the window from the model")
    args = ap.parse_args()

    zpats = [p.strip() for p in (args.zero or "").split(",") if p.strip()]
    opats = [p.strip() for p in (args.one or "").split(",") if p.strip()]
    oh = {}
    for spec in (args.onehot or "").split(","):
        if spec.strip():
            suffix, idx = spec.rsplit(":", 1)
            oh[suffix.strip()] = int(idx)
    if not zpats and not opats and not oh:
        raise SystemExit("give --zero and/or --one and/or --onehot")
    _ver, info, header_end = read_header(args.model)
    data_start = (header_end + 31) // 32 * 32
    total = max(o + s for (_ne, _dt, o) in info.values() for s in [0])  # placeholder

    # Tensor sizes: byte distance to the next tensor, in offset order.
    order = sorted(info.items(), key=lambda kv: kv[1][2])
    ends = []
    for i, (_name, (_ne, _dt, off)) in enumerate(order):
        nxt = order[i + 1][1][2] if i + 1 < len(order) else None
        ends.append((off, nxt))
    import os
    file_len = os.path.getsize(args.model)

    keep = {k.strip() for k in (args.keep or "").split(",") if k.strip()}
    zeroed = [n for n in info if any(n.endswith(p) for p in zpats) and n not in keep]
    ones = [n for n in info if any(n.endswith(p) for p in opats)]
    ohmap = {n: idx for n, idx in
             ((n, idx) for n in info for p, idx in oh.items() if n.endswith(p))}
    if not zeroed and not ones and not ohmap:
        raise SystemExit(f"no tensor matches {zpats + opats + list(oh)}")
    for n in ones:
        ne, dt, _ = info[n]
        if dt != 0:
            raise SystemExit(f"{n}: --one needs an f32 tensor, got dtype {dt}")
    print(f"zeroing {len(zeroed)} tensors, setting 1.0 on {len(ones)}, one-hot on {len(ohmap)}")

    shutil.copyfile(args.model, args.out)
    with open(args.out, "r+b") as f:
        for i, (name, (ne, dt, off)) in enumerate(order):
            end = order[i + 1][1][2] if i + 1 < len(order) else file_len - data_start
            if name in ones:
                n_elts = 1
                for d in ne:
                    n_elts *= d
                f.seek(data_start + off)
                f.write(struct.pack("<%df" % n_elts, *([1.0] * n_elts)))
            elif name in ohmap:
                ne, dt, _ = info[name]
                if dt != 0:
                    raise SystemExit(f"{name}: --onehot needs an f32 tensor, got dtype {dt}")
                n0, n_elts = ne[0], 1
                for dim in ne:
                    n_elts *= dim
                idx = ohmap[name]
                f.seek(data_start + off)
                f.write(struct.pack("<%df" % n_elts,
                                    *[1.0 if i % n0 == idx else 0.0 for i in range(n_elts)]))
            elif name in zeroed:
                f.seek(data_start + off)
                f.write(b"\0" * (end - off))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
