#!/usr/bin/env python3
"""Per-tensor census of a GGUF: exact byte sizes from the element type, cross-checked against the
offset gaps, grouped by role. This is the authoritative "which quant types does this checkpoint
actually contain" tool - the class listing in gguf_dump.py is a convenience view, not this.

Usage: tensor_census.py <model.gguf> [--json] [--filter SUBSTR]
"""
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gguf_dump import Reader  # noqa: E402

# GGML element types: id -> (name, elems_per_block, bytes_per_block)
TYPES = {
    0: ("f32", 1, 4), 1: ("f16", 1, 2), 2: ("q4_0", 32, 18), 3: ("q4_1", 32, 20),
    6: ("q5_0", 32, 22), 7: ("q5_1", 32, 24), 8: ("q8_0", 32, 34), 9: ("q8_1", 32, 36),
    10: ("q2_k", 256, 84), 11: ("q3_k", 256, 110), 12: ("q4_k", 256, 144),
    13: ("q5_k", 256, 176), 14: ("q6_k", 256, 210), 15: ("q8_k", 256, 260),
    24: ("i8", 1, 1), 25: ("i16", 1, 2), 26: ("i32", 1, 4), 27: ("i64", 1, 8),
    28: ("f64", 1, 8), 30: ("bf16", 1, 2),
}


def main():
    path = sys.argv[1]
    filt = None
    if "--filter" in sys.argv:
        filt = sys.argv[sys.argv.index("--filter") + 1]
    f = open(path, "rb")
    r = Reader(f)
    assert r.read(4) == b"GGUF"
    ver = r.u32()
    n_tensors = r.u64()
    n_kv = r.u64()
    md = {}
    for _ in range(n_kv):
        k = r.string()
        md[k] = r.value(r.u32())
    infos = []
    for _ in range(n_tensors):
        name = r.string()
        nd = r.u32()
        dims = [r.u64() for _ in range(nd)]
        dt = r.u32()
        off = r.u64()
        infos.append((name, dims, dt, off))
    align = md.get("general.alignment", 32)
    data_start = (f.tell() + align - 1) // align * align
    order = sorted(range(len(infos)), key=lambda i: infos[i][3])
    gap = {}
    for j, i in enumerate(order):
        nxt = infos[order[j + 1]][3] if j + 1 < len(order) else os.path.getsize(path) - data_start
        gap[infos[i][0]] = nxt - infos[i][3]

    rows, bad = [], []
    by_type, by_role = {}, {}
    total_type = total_gap = 0
    for name, dims, dt, off in infos:
        tn, bs, tl = TYPES[dt]
        n = 1
        for d in dims:
            n *= d
        size = -(-n // bs) * tl
        g = gap[name]
        pad = g - size
        if pad < 0 or pad >= align:
            bad.append((name, size, g))
        total_type += size
        total_gap += g
        by_type.setdefault(tn, [0, 0])
        by_type[tn][0] += 1
        by_type[tn][1] += size
        role = re.sub(r"^blk\.\d+\.", "blk.N.", name)
        role = re.sub(r"\d+", "#", role)
        by_role.setdefault((role, tn, tuple(dims)), [0, 0])
        by_role[(role, tn, tuple(dims))][0] += 1
        by_role[(role, tn, tuple(dims))][1] += size
        if not filt or filt in name:
            rows.append((name, dims, tn, size, g))

    print(f"file             : {path}")
    print(f"gguf             : v{ver}  tensors={n_tensors}  data_start={data_start:,}  align={align}")
    print(f"bytes by type    : {total_type:,}   bytes by offset-gap: {total_gap:,}   (gap >= size, pad < align)")
    print(f"gap anomalies    : {len(bad)}")
    print()
    print(f"{'type':<8}{'count':>7}{'bytes':>18}{'b/elem':>9}")
    for tn, (c, nb) in sorted(by_type.items(), key=lambda kv: -kv[1][1]):
        ne = sum(1 for _ in [])  # placeholder
        print(f"{tn:<8}{c:>7}{nb:>18,}{bytes_per_elem(by_role, tn):>9.4f}")
    print()
    print(f"{'role':<40}{'n':>4}{'type':<8}{'dims':<22}{'bytes':>16}")
    for (role, tn, dims), (c, nb) in sorted(by_role.items()):
        print(f"{role:<40}{c:>4}{tn:<8}{str(list(dims)):<22}{nb:>16,}")
    if bad:
        print()
        print("GAP ANOMALIES:")
        for name, size, g in bad[:20]:
            print(f"  {name}: size={size} gap={g}")
    if "--json" in sys.argv:
        out = {
            "file": path, "data_start": data_start, "align": align,
            "total_bytes_by_type": total_type, "total_bytes_by_gap": total_gap,
            "tensors": [{"name": n, "dims": d, "type": TYPES[t][0], "size": s, "gap": gap[n]}
                        for n, d, t, s, _ in rows],
        }
        json.dump(out, sys.stdout, indent=1)


def bytes_per_elem(by_role, tn):
    for (role, t, dims), (c, nb) in by_role.items():
        if t == tn:
            ne = 1
            for d in dims:
                ne *= d
            return nb / (ne * c)
    return 0.0


if __name__ == "__main__":
    main()
