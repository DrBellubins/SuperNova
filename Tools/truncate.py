#!/usr/bin/env python3
"""Write a prefix copy of the checkpoint: the first K physical blocks and nothing else.

This is the localisation oracle for the Phase 2 forward. `Docs/design-plan.md` §1.1 makes the trunk
an *execution* list over physical blocks, and every cache (KV, GDN state, conv window) is per
execution, so the hidden state after execution K-1 depends on nothing beyond blocks 0..K-1. Cutting
the checkpoint at K therefore gives a model whose final hidden state is *exactly* the reference's
state at that point in the real model - which is what `sn ref --dump h_final` on the same K must
match. Comparing full 56-execution outputs says "something is wrong"; comparing prefixes says where.

The prefix model has to satisfy the fork's own validations (`src/models/sensenova-loop.cpp:20-39`),
so the loop and mHC ranges are rewritten to the truncated block range with `loop.count = 1`: for
`loop.count == 1`, `physical_layer(il) == il`, so the execution order of the prefix is unchanged.
`nextn_predict_layers` goes to 0 because its validation requires the last block to be full
attention, and the MTP tensors (`blk.56.*`) are not part of any prefix.

    Tools/truncate.py --model M.gguf --out /tmp/prefix-12.gguf --blocks 12

usage then:
    Tools/sn-oracle --model /tmp/prefix-12.gguf --layers "" --out /tmp/orc12 <tokens>   # result_norm + logits
    target/debug/sn ref --model /tmp/prefix-12.gguf --tokens-file <tokens> --dump /tmp/sn12
"""
import argparse
import os
import struct
import sys

ARCH = "sensenova_loop"
ALIGN = 32


def read_gguf(path):
    """(kv list in file order, tensor list in file order, header end)."""
    kv = []
    tens = []
    with open(path, "rb") as f:
        assert f.read(4) == b"GGUF", "not a GGUF"
        ver = struct.unpack("<I", f.read(4))[0]

        def rstr():
            return f.read(struct.unpack("<Q", f.read(8))[0])

        n_tensors = struct.unpack("<Q", f.read(8))[0]
        n_kv = struct.unpack("<Q", f.read(8))[0]
        scalar = {0: ("<B", 1), 1: ("<b", 1), 2: ("<H", 2), 3: ("<h", 2), 4: ("<I", 4),
                  5: ("<i", 4), 6: ("<f", 4), 7: ("<?", 1), 10: ("<Q", 8), 11: ("<q", 8),
                  12: ("<d", 8)}

        def rval(t):
            if t == 8:
                return ("s", rstr())
            if t == 9:
                et = struct.unpack("<I", f.read(4))[0]
                n = struct.unpack("<Q", f.read(8))[0]
                return ("a", et, [rval(et) for _ in range(n)])
            fmt, size = scalar[t]
            return ("v", struct.unpack(fmt, f.read(size))[0])

        for _ in range(n_kv):
            key = rstr().decode()
            t = struct.unpack("<I", f.read(4))[0]
            kv.append((key, t, rval(t)))
        for _ in range(n_tensors):
            name = rstr().decode()
            nd = struct.unpack("<I", f.read(4))[0]
            ne = [struct.unpack("<Q", f.read(8))[0] for _ in range(nd)]
            dt = struct.unpack("<I", f.read(4))[0]
            off = struct.unpack("<Q", f.read(8))[0]
            tens.append((name, ne, dt, off))
        return ver, kv, tens, f.tell()


def enc_str(s):
    b = s if isinstance(s, bytes) else s.encode()
    return struct.pack("<Q", len(b)) + b


def enc_val(t, v):
    """Re-encode a value read by read_gguf ('s', bytes) / ('a', et, items) / ('v', scalar)."""
    if v[0] == "s":
        return enc_str(v[1])
    if v[0] == "a":
        et, items = v[1], v[2]
        out = struct.pack("<I", et) + struct.pack("<Q", len(items))
        for it in items:
            out += enc_val(et, it)
        return out
    fmt = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?",
           10: "<Q", 11: "<q", 12: "<d"}[t]
    return struct.pack(fmt, v[1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--blocks", type=int, required=True, help="keep physical blocks 0..K-1")
    ap.add_argument("--loop-count", type=int, default=1,
                    help="1 (default): run each kept block once - a prefix of the real trunk for "
                         "K <= 28. 2: keep the checkpoint's loop, i.e. blocks 0..27 then 4..27 "
                         "again, which is the real trunk minus blocks 28..31 at K = 52.")
    args = ap.parse_args()
    k = args.blocks
    lc = args.loop_count
    if k < 5:
        raise SystemExit("K < 5 cannot carry the mHC range (block 4 is the first mHC block)")
    if lc not in (1, 2):
        raise SystemExit("--loop-count is 1 or 2 (the checkpoint stores 2)")
    if lc == 2 and k != 52:
        raise SystemExit("loop.count=2 needs block_count = 2*loop.end = 52")

    ver, kv, tens, header_end = read_gguf(args.model)
    src = {name: (ne, dt, off) for name, ne, dt, off in tens}
    order = sorted(tens, key=lambda t: t[3])
    file_size = os.path.getsize(args.model)
    data_start = (header_end + ALIGN - 1) // ALIGN * ALIGN
    sizes = {}
    for i, (name, _ne, _dt, off) in enumerate(order):
        nxt = order[i + 1][3] if i + 1 < len(order) else file_size - data_start
        sizes[name] = nxt - off

    def block_of(name):
        if not name.startswith("blk."):
            return None
        return int(name.split(".", 1)[1].split(".", 1)[0])

    src_max = k if lc == 1 else 28
    keep = [t for t in tens if (block_of(t[0]) is None or block_of(t[0]) < src_max)]
    dropped = len(tens) - len(keep)

    new_kv = []
    rec = None
    for key, t, v in kv:
        if key == f"{ARCH}.block_count":
            new_kv.append((key, t, ("v", k)))
        elif key == f"{ARCH}.nextn_predict_layers":
            new_kv.append((key, t, ("v", 0)))
        elif key == f"{ARCH}.loop.end":
            new_kv.append((key, t, ("v", k if lc == 1 else 28)))
        elif key == f"{ARCH}.loop.count":
            new_kv.append((key, t, ("v", lc)))
        elif key == f"{ARCH}.mhc.end":
            new_kv.append((key, t, ("v", k if lc == 1 else 28)))
        elif key == f"{ARCH}.attention.recurrent_layers":
            rec = v
            new_kv.append((key, t, ("a", v[1], v[2][:k])))
        else:
            new_kv.append((key, t, v))
    for want, key in [(f"{ARCH}.block_count", "block_count"), (f"{ARCH}.loop.end", "loop.end"),
                      (f"{ARCH}.mhc.end", "mhc.end"), (f"{ARCH}.loop.count", "loop.count"),
                      (f"{ARCH}.nextn_predict_layers", "nextn_predict_layers"),
                      (f"{ARCH}.attention.recurrent_layers", "recurrent_layers")]:
        if not any(key == want for key, _t, _v in new_kv):
            raise SystemExit(f"{key} missing from the source metadata")
    if len(rec[2]) < k:
        raise SystemExit("recurrent_layers array shorter than block_count")

    # repack tensor data, preserving file order, each tensor on its own alignment boundary
    out = bytearray()
    new_info = []
    for name, ne, dt, off in keep:
        while len(out) % ALIGN:
            out += b"\0"
        with open(args.model, "rb") as f:
            f.seek(data_start + off)
            out += f.read(sizes[name])
        new_info.append((name, ne, dt, len(out) - sizes[name]))

    hdr = bytearray()
    hdr += b"GGUF" + struct.pack("<I", ver)
    hdr += struct.pack("<Q", len(new_info)) + struct.pack("<Q", len(new_kv))
    for key, t, v in new_kv:
        hdr += enc_str(key) + struct.pack("<I", t) + enc_val(t, v)
    for name, ne, dt, off in new_info:
        hdr += enc_str(name) + struct.pack("<I", len(ne))
        hdr += b"".join(struct.pack("<Q", d) for d in ne)
        hdr += struct.pack("<I", dt) + struct.pack("<Q", off)
    pad = (-len(hdr)) % ALIGN
    hdr += b"\0" * pad

    with open(args.out, "wb") as f:
        f.write(hdr)
        f.write(out)
    print(f"{args.out}: {len(new_info)} tensors ({dropped} dropped), blocks 0..{k - 1}, "
          f"loop[4,{k if lc == 1 else 28}) x{lc}, mhc[4,{k if lc == 1 else 28}), nextn 0, "
          f"{os.path.getsize(args.out) / 1e6:.1f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
