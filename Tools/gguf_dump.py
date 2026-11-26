#!/usr/bin/env python3
"""Dump GGUF metadata + tensor info. Dependency-free (stdlib only).

Usage:
  gguf_dump.py <model.gguf> [--tensors] [--filter SUBSTR] [--json]
"""
import struct
import sys
import json
import os

GGUF_MAGIC = b"GGUF"

GGUF_TYPE_UINT8   = 0
GGUF_TYPE_INT8    = 1
GGUF_TYPE_UINT16  = 2
GGUF_TYPE_INT16   = 3
GGUF_TYPE_UINT32  = 4
GGUF_TYPE_INT32   = 5
GGUF_TYPE_FLOAT32 = 6
GGUF_TYPE_BOOL    = 7
GGUF_TYPE_STRING  = 8
GGUF_TYPE_ARRAY   = 9
GGUF_TYPE_UINT64  = 10
GGUF_TYPE_INT64   = 11
GGUF_TYPE_FLOAT64 = 12

TYPE_FMT = {
    GGUF_TYPE_UINT8: "<B", GGUF_TYPE_INT8: "<b",
    GGUF_TYPE_UINT16: "<H", GGUF_TYPE_INT16: "<h",
    GGUF_TYPE_UINT32: "<I", GGUF_TYPE_INT32: "<i",
    GGUF_TYPE_FLOAT32: "<f", GGUF_TYPE_BOOL: "<B",
    GGUF_TYPE_UINT64: "<Q", GGUF_TYPE_INT64: "<q",
    GGUF_TYPE_FLOAT64: "<d",
}
TYPE_NAME = {v: k for k, v in {
    0: "u8", 1: "i8", 2: "u16", 3: "i16", 4: "u32", 5: "i32", 6: "f32",
    7: "bool", 8: "str", 9: "arr", 10: "u64", 11: "i64", 12: "f64"}.items()}

QUANTIZED = {
    "q4_0": 128, "q4_1": 144, "q5_0": 160, "q5_1": 176, "q8_0": 132,
    "q2_0": 96, "q3_0": 112, "q4_k": 144, "q5_k": 160, "q6_k": 168, "q8_k": 136,
    "iq2_xxs": 98, "iq2_xs": 98, "iq2_s": 96, "iq3_xxs": 132, "iq3_s": 132,
    "iq1_s": 40, "iq1_s_m": 48, "iq2_s": 96, "iq4_nl": 72, "iq4_xs": 72,
}


class Reader:
    def __init__(self, f):
        self.f = f

    def read(self, n):
        d = self.f.read(n)
        if len(d) != n:
            raise EOFError(f"short read {len(d)}/{n}")
        return d

    def u32(self):
        return struct.unpack("<I", self.read(4))[0]

    def u64(self):
        return struct.unpack("<Q", self.read(8))[0]

    def string(self):
        n = self.u64()
        return self.read(n).decode("utf-8", "replace")

    def scalar(self, t):
        fmt = TYPE_FMT[t]
        return struct.unpack(fmt, self.read(struct.calcsize(fmt)))[0]

    def value(self, t):
        if t == GGUF_TYPE_ARRAY:
            et = self.u32()
            n = self.u64()
            if et == GGUF_TYPE_STRING:
                return [self.string() for _ in range(n)]
            if n > 4096:
                # don't materialise giant arrays (vocab / merges)
                step = {GGUF_TYPE_UINT32: 4, GGUF_TYPE_INT32: 4, GGUF_TYPE_FLOAT32: 4,
                        GGUF_TYPE_UINT8: 1, GGUF_TYPE_INT8: 1, GGUF_TYPE_UINT16: 2,
                        GGUF_TYPE_INT16: 2, GGUF_TYPE_BOOL: 1, GGUF_TYPE_UINT64: 8,
                        GGUF_TYPE_INT64: 8, GGUF_TYPE_FLOAT64: 8}.get(et)
                self.f.seek(step * n, 1)
                return f"<{n} x {TYPE_NAME.get(et, et)} skipped>"
            return [self.scalar(et) for _ in range(n)]
        if t == GGUF_TYPE_STRING:
            return self.string()
        return self.scalar(t)


def main():
    path = sys.argv[1]
    want_tensors = "--tensors" in sys.argv
    filt = None
    if "--filter" in sys.argv:
        filt = sys.argv[sys.argv.index("--filter") + 1]
    as_json = "--json" in sys.argv

    with open(path, "rb") as f:
        r = Reader(f)
        magic = r.read(4)
        assert magic == GGUF_MAGIC, magic
        version = r.u32()
        n_tensors = r.u64()
        n_kv = r.u64()
        md = {}
        for _ in range(n_kv):
            k = r.string()
            t = r.u32()
            md[k] = r.value(t)
        info = {"version": version, "n_tensors": n_tensors, "n_kv": n_kv}

        # tensor info section starts here
        alignment = md.get("general.alignment", 32)
        tinfos = []
        for _ in range(n_tensors):
            name = r.string()
            nd = r.u32()
            dims = [r.u64() for _ in range(nd)]
            dt = r.u32()
            off = r.u64()
            tinfos.append((name, dims, dt, off))
        if tinfos:
            data_start = (r.f.tell() + alignment - 1) // alignment * alignment
        else:
            data_start = r.f.tell()

    if filt:
        md = {k: v for k, v in md.items() if filt.lower() in k.lower()}

    if as_json:
        print(json.dumps({"info": info, "metadata": md,
                          "tensors": [{"name": n, "dims": d, "dtype": t, "offset": o}
                                      for n, d, t, o in tinfos]}, indent=2, default=str))
        return

    print(f"=== {path} ===")
    sz = os.path.getsize(path)
    print(f"size         : {sz/1e9:.3f} GB ({sz:,} bytes)")
    print(f"gguf version : {info['version']}")
    print(f"tensors      : {info['n_tensors']}")
    print(f"kv pairs     : {info['n_kv']}")
    print()
    print("--- METADATA ---")
    for k in sorted(md):
        v = md[k]
        if isinstance(v, list) and len(v) > 24:
            print(f"  {k:55s} = [{len(v)} items]")
        else:
            print(f"  {k:55s} = {v}")

    if want_tensors:
        print()
        print("--- TENSORS ---")
        dtnames = {0: "f32", 1: "f16", 2: "q4_0", 3: "q4_1", 6: "q5_0", 7: "q5_1",
                   8: "q8_0", 10: "q2_0", 11: "q3_0", 12: "q2_k", 13: "q3_k",
                   14: "q4_k", 15: "q5_k", 16: "q6_k", 17: "q8_k",
                   22: "bf16", 23: "q4_k", 24: "iq4_nl", 25: "iq2_s", 26: "iq3_s",
                   28: "iq1_s", 29: "iq1_s_m", 30: "iq2_s", 31: "iq2_xxs", 32: "iq2_xs",
                   34: "iq3_xxs", 35: "iq3_s", 36: "iq4_nl", 37: "iq4_xs", 38: "f32"}
        by_pattern = {}
        total = 0
        for name, dims, dt, off in tinfos:
            import re
            pat = re.sub(r"\d+", "#", name)
            e = by_pattern.setdefault(pat, {"n": 0, "dims": dims, "dt": dt, "bytes": 0})
            e["n"] += 1
            nelem = 1
            for d in dims:
                nelem *= d
            bq = QUANTIZED.get(dtnames.get(dt, ""), None)
            nbytes = (nelem // 16 * bq) if bq and nelem % 16 == 0 else nelem * {0: 4, 1: 2, 22: 2}.get(dt, 4)
            e["bytes"] += nbytes
            total += nbytes
            e["dt"] = dtnames.get(dt, dt)
        for pat in sorted(by_pattern):
            e = by_pattern[pat]
            print(f"  {pat:60s} x{e['n']:<4} {str(e['dims']):24s} {e['dt']:8s} {e['bytes']/1e6:9.2f} MB")
        print(f"\n  tensor bytes total: {total/1e9:.3f} GB")
        print(f"  data section starts at: {data_start:,}")
        # list a few concrete tensor names for layer 0
        print("\n--- layer 0 tensor names ---")
        for name, dims, dt, off in tinfos:
            if ".0." in name or name.startswith("token") or name.startswith("output") or "norm" in name and ".0." not in name:
                if int(os.environ.get("SHOW_ALL", "0")) or ".0." in name or "token" in name or "output" in name:
                    print(f"  {name:55s} {str(dims):22s} {dtnames.get(dt, dt)}")


if __name__ == "__main__":
    main()
