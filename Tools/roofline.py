#!/usr/bin/env python3
"""Exact weight-traffic roofline for SenseNova-Loop-9B (and any GGUF).

Computes real per-tensor byte sizes from GGUF offsets, then multiplies by the
number of times each block is *executed* per decode token, so we get the true
bytes/token that any decoder must stream from VRAM.

Usage: roofline.py <model.gguf> [--bw GBPS] [--json]

--json emits the per-tensor accounting (name, stored bytes, executions per token) plus totals,
which is what Tools/plan-check.py diffs against `sn dump --plan --json`.
"""
import os
import re
import struct
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gguf_dump import Reader  # noqa: E402


def load(path):
    f = open(path, "rb")
    r = Reader(f)
    assert r.read(4) == b"GGUF"
    r.u32()
    n_tensors = r.u64()
    n_kv = r.u64()
    md = {}
    for _ in range(n_kv):
        k = r.string()
        t = r.u32()
        md[k] = r.value(t)
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
    # exact size = next offset - this offset (tensors are packed in offset order)
    order = sorted(range(len(infos)), key=lambda i: infos[i][3])
    size = {}
    for j, i in enumerate(order):
        nxt = infos[order[j + 1]][3] if j + 1 < len(order) else os.path.getsize(path) - data_start
        size[infos[i][0]] = nxt - infos[i][3]
    return md, infos, size


def dtn(dt):
    # GGML element type ids (ggml.h): only what this checkpoint uses needs a name.
    names = {0: "f32", 1: "f16", 8: "q8_0", 10: "q2_k", 11: "q3_k", 12: "q4_k",
             13: "q5_k", 14: "q6_k", 30: "bf16"}
    return names.get(dt, f"type#{dt}")


def main():
    path = sys.argv[1]
    bw = 240.0
    if "--bw" in sys.argv:
        bw = float(sys.argv[sys.argv.index("--bw") + 1])
    md, infos, size = load(path)
    p = md["general.architecture"]
    loop_s, loop_e, loop_c = (md[f"{p}.loop.start"], md[f"{p}.loop.end"], md[f"{p}.loop.count"])
    n_all = md[f"{p}.block_count"]
    n_nextn = md.get(f"{p}.nextn_predict_layers", 0)
    n_layer = n_all - n_nextn
    extra = (loop_c - 1) * (loop_e - loop_s)

    def phys(il):
        if il < loop_s:
            return il
        if il < loop_e + extra:
            return loop_s + (il - loop_s) % (loop_e - loop_s)
        return il - extra

    # how many times is each *GGUF block index* executed per forward?
    nexec = {}
    per_tensor = []
    for il in range(n_layer):
        ph = phys(il)
        # block index used in tensor names (see conversion: physical -> first_visit)
        bid = ph if ph < loop_e else il
        nexec[bid] = nexec.get(bid, 0) + 1
    for k in range(n_layer, n_all):
        nexec[k] = 0  # MTP: not executed by the trunk

    tot = 0
    by_class = {}
    for name, dims, dt, off in infos:
        m = re.match(r"blk\.(\d+)\.", name)
        n = size[name]
        if m:
            e = nexec.get(int(m.group(1)), 1)
            cls = re.sub(r"^blk\.\d+\.", "blk.N.", name)
            cls = re.sub(r"\d+", "#", cls)
        else:
            e = 1
            cls = name
        by_class.setdefault((cls, e), [0, 0])
        by_class[(cls, e)][0] += n
        by_class[(cls, e)][1] += 1
        tot += n * e
        per_tensor.append({"name": name, "class": cls, "type": dtn(dt), "bytes": n,
                           "visits": e, "bytes_per_token": n * e})

    if "--json" in sys.argv:
        import json
        print(json.dumps({
            "model": md.get("general.name"), "arch": p,
            "n_layer": n_layer, "n_physical": n_layer - extra,
            "loop": [loop_s, loop_e, loop_c], "extra_visits": extra,
            "mhc": [md.get(f"{p}.mhc.start"), md.get(f"{p}.mhc.end")],
            "mhc_executions": sum(1 for il in range(n_layer)
                                  if md.get(f"{p}.mhc.start", 0) <= phys(il) < md.get(f"{p}.mhc.end", 0)),
            "stored_bytes": sum(size.values()), "bytes_per_token": tot,
            "tensors": per_tensor,
        }))
        return

    print(f"model            : {md.get('general.name')}")
    print(f"n_layer (exec)   : {n_layer}   physical: {n_layer - extra}   loop: [{loop_s},{loop_e}) x{loop_c}   MTP: {n_nextn}")
    print(f"stored weights   : {sum(size.values())/1e9:.3f} GB")
    print(f"executed / token : {tot/1e9:.3f} GB   (weight re-reads from the loop: x{tot/sum(size.values()):.2f})")
    print()
    print(f"{'class':<34}{'n':>4}{'stored MB':>12}{'exec':>6}{'MB/token':>10}")
    for (cls, e), (nb, n) in sorted(by_class.items(), key=lambda kv: -kv[1][0] * kv[1][1]):
        if nb * e / 1e6 < 1:
            continue
        print(f"{cls:<34}{n:>4}{nb/1e6:>12.1f}{e:>6}{nb*e/1e6:>10.1f}")
    print()
    for b in (288.0, 240.0, 180.0):
        print(f"  decode roofline @{b:>5.0f} GB/s : {tot/1e9/(b*0.85)*1000:7.1f} ms/token  = {b*0.85/(tot/1e9):6.1f} t/s (85% eff.)")


if __name__ == "__main__":
    main()
