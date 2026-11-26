#!/usr/bin/env python3
"""Write a copy of a (prefix) checkpoint with quantised tensors converted to f32.

The Phase 2 gate compares an f32 reference forward against a *quantised* reference engine, and the
CPU path of that engine requantises every activation row to q8_0 before a Q4_K/Q6_K dot product
(`ggml_vec_dot_q8_0_q4_K`), i.e. ~0.3 % relative error per matmul. That is ~1e4 times larger than
anything an f32 implementation can be blamed for, and it accumulates through 42 recurrent states.
To tell "SuperNova disagrees with the reference" from "the reference's operands disagree with f32",
convert the weights and compare again: with `--match blk.` the block weights are exact f32 in both
engines, so the only variables left are the arithmetic and the code.

    Tools/def32.py --model /tmp/prefix/p5.gguf --out /tmp/p5-f32.gguf --match 'blk.'

Keeps `token_embd` / `output` quantised on purpose: both engines dequantise the same bytes to the
same f32 values (verified row-by-row in Docs/design-plan.md §2.9), and f32 versions of those two
tensors alone would cost 8.2 GB.
"""
import argparse
import os
import struct
import sys

import numpy as np  # noqa: E402  (the dequant helpers below are vectorised with it)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dequant_ref import (Q4K_BYTES, Q6K_BYTES, deq_q4k_np, deq_q6k_np)  # noqa: E402
from truncate import ALIGN, enc_str, enc_val, read_gguf  # noqa: E402

F32, Q4_K, Q6_K = 0, 12, 14
TYPES = {F32: "f32", 1: "f16", Q4_K: "q4_k", Q6_K: "q6_k"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--match", default="blk.", help="substring of the tensor names to convert")
    ap.add_argument("--skip", default="token_embd,output", help="comma list never converted")
    args = ap.parse_args()

    skip = {s.strip() for s in args.skip.split(",") if s.strip()}
    ver, kv, tens, header_end = read_gguf(args.model)
    order = sorted(tens, key=lambda t: t[3])
    file_size = os.path.getsize(args.model)
    data_start = (header_end + ALIGN - 1) // ALIGN * ALIGN
    sizes = {}
    for i, (name, _ne, _dt, off) in enumerate(order):
        nxt = order[i + 1][3] if i + 1 < len(order) else file_size - data_start
        sizes[name] = nxt - off

    out = bytearray()
    new_info = []
    done = 0
    with open(args.model, "rb") as src:
        for name, ne, dt, off in order:
            while len(out) % ALIGN:
                out += b"\0"
            src.seek(data_start + off)
            raw = src.read(sizes[name])
            if args.match in name and not any(name.startswith(s) or f".{s}" in name for s in skip):
                if dt in (Q4_K, Q6_K):
                    n_in, n_rows = ne[0], (ne[1] if len(ne) > 1 else 1)
                    fn = deq_q4k_np if dt == Q4_K else deq_q6k_np
                    vals = fn(raw, n_rows, n_in).astype(np.float32)
                    raw = vals.reshape(-1).tobytes()
                    dt = F32
                    done += 1
            new_info.append((name, ne, dt, len(out)))
            out += raw

    hdr = bytearray()
    hdr += b"GGUF" + struct.pack("<I", ver)
    hdr += struct.pack("<Q", len(new_info)) + struct.pack("<Q", len(kv))
    for key, t, v in kv:
        hdr += enc_str(key) + struct.pack("<I", t) + enc_val(t, v)
    for name, ne, dt, off in new_info:
        hdr += enc_str(name) + struct.pack("<I", len(ne))
        hdr += b"".join(struct.pack("<Q", d) for d in ne)
        hdr += struct.pack("<I", dt) + struct.pack("<Q", off)
    hdr += b"\0" * ((-len(hdr)) % ALIGN)
    with open(args.out, "wb") as f:
        f.write(hdr)
        f.write(out)
    print(f"{args.out}: {done} tensors converted to f32, "
          f"{os.path.getsize(args.out) / 1e6:.1f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
