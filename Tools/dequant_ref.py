#!/usr/bin/env python3
"""Independent Q4_K / Q6_K / f16 dequantiser, transliterated from GGML's reference C.

Source (read-only reference implementation, llama.cpp-opensensenova):
  ggml/src/ggml-quants.c   dequantize_row_q4_K, get_scale_min_k4, dequantize_row_q6_K
  ggml/src/ggml-common.h   block field order, which is what the kernels read:
                           q4_K: d @0 (f16), dmin @2 (f16), scales[12] @4, qs[128] @16 = 144 B
                           q6_K: ql[128] @0, qh[64] @128, sc[16] (signed) @192, d @208 = 210 B

This exists to be a *second* implementation: `sn deq` (crates/sn-core/src/dq.rs) is compared against
it on real tensors from the checkpoint, which a hand-built block test cannot do - a wrong nibble or
scale mapping inside a 256-element super-block preserves every row norm and still destroys the model.

Usage:
  Tools/dequant_ref.py <model.gguf> --tensor blk.4.attn_qkv_gate.weight --rows 0,1 --check-sn
  Tools/dequant_ref.py <model.gguf> --matrix        # the regression matrix below
"""
import argparse, json, math, os, struct, subprocess, sys
import numpy as np

GGUF_TYPE_F32, GGUF_TYPE_F16, GGUF_TYPE_Q4_K, GGUF_TYPE_Q6_K = 0, 1, 12, 14
QK_K, Q4K_BYTES, Q6K_BYTES = 256, 144, 210


def f16(h):
    """IEEE-754 half, via struct's native '<e' (this file's only job is to be a
    second opinion, so do not hand-roll the bias: an earlier version used the
    f32 bias here and quietly rescaled every Q4_K super-block by 2^-102)."""
    return struct.unpack("<e", struct.pack("<H", h))[0]


def get_scale_min_k4(j, sc_bytes):
    """GGML get_scale_min_k4, byte-for-byte."""
    if j < 4:
        return sc_bytes[j] & 63, sc_bytes[j + 4] & 63
    return ((sc_bytes[j + 4] & 0xF) | ((sc_bytes[j - 4] >> 6) << 4),
            (sc_bytes[j + 4] >> 4) | ((sc_bytes[j] >> 6) << 4))


def dequant_q4_K(blk):
    """256 values from a 144-byte block."""
    d = f16(struct.unpack_from("<H", blk, 0)[0])
    dmin = f16(struct.unpack_from("<H", blk, 2)[0])
    scales = blk[4:16]
    qs = blk[16:144]
    y = []
    q = 0
    is_ = 0
    for _ in range(0, QK_K, 64):
        sc0, m0 = get_scale_min_k4(is_ + 0, scales)
        sc1, m1 = get_scale_min_k4(is_ + 1, scales)
        d1, m1v = d * sc0, dmin * m0
        d2, m2v = d * sc1, dmin * m1
        for l in range(32):
            y.append(d1 * (qs[q + l] & 0xF) - m1v)
        for l in range(32):
            y.append(d2 * (qs[q + l] >> 4) - m2v)
        q += 32
        is_ += 2
    return y


def dequant_q6_K(blk):
    """256 values from a 210-byte block.

    Two 128-value half-blocks; each advances ql by 64, qh by 32 and sc by 8, so
    the slices must be re-taken per half-block, and the four values produced for
    one `l` land at l, l+32, l+64, l+96 (not next to each other).
    """
    d = f16(struct.unpack_from("<H", blk, 208)[0])
    y = [0.0] * QK_K
    for n in range(0, QK_K, 128):
        hb = n // 128
        ql = blk[64 * hb:64 * hb + 64]
        qh = blk[128 + 32 * hb:128 + 32 * hb + 32]
        sc = struct.unpack_from("<8b", blk, 192 + 8 * hb)
        for l in range(32):
            i2 = l // 16
            q1 = ((ql[l + 0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32
            q2 = ((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32
            q3 = ((ql[l + 0] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32
            q4 = ((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32
            y[n + l + 0] = d * sc[i2 + 0] * q1
            y[n + l + 32] = d * sc[i2 + 2] * q2
            y[n + l + 64] = d * sc[i2 + 4] * q3
            y[n + l + 96] = d * sc[i2 + 6] * q4
    return y


SCALAR = {0: ("<B", 1), 1: ("<b", 1), 2: ("<H", 2), 3: ("<h", 2), 4: ("<I", 4), 5: ("<i", 4),
          6: ("<f", 4), 7: ("<B", 1), 10: ("<Q", 8), 11: ("<q", 8), 12: ("<d", 8)}


def f16_np(h):
    """Vectorised half -> float32 (numpy has IEEE half natively)."""
    return np.asarray(h, dtype=np.uint16).view("<f2").astype(np.float32)


def deq_q6k_np(buf, n_rows, n_in):
    """Vectorised Q6_K -> (n_rows, n_in) f32; same equations as dequant_q6_K."""
    nb = n_in // QK_K
    b = np.frombuffer(buf[: n_rows * nb * Q6K_BYTES], dtype=np.uint8).reshape(-1, Q6K_BYTES)
    dv = f16_np(np.frombuffer(b[:, 208:210].tobytes(), dtype=np.uint16))[:, None]
    out = np.empty((b.shape[0], QK_K), np.float32)
    for n in (0, 128):
        hb = n // 128
        qlh = b[:, 64 * hb:64 * hb + 64]
        qhh = b[:, 128 + 32 * hb:128 + 32 * hb + 32]
        sch = b[:, 192 + 8 * hb:192 + 8 * hb + 8].astype(np.int8)
        for l in range(32):
            i2 = l // 16
            for g in range(4):
                nib = (qlh[:, l + 32 * (g % 2)] >> (4 * (g // 2))) & 0xF
                hi = (qhh[:, l] >> (2 * g)) & 3
                q = ((hi << 4) | nib).astype(np.float32) - 32.0
                out[:, n + l + 32 * g] = dv[:, 0] * sch[:, i2 + 2 * g] * q
    return out.reshape(n_rows, n_in)


def deq_q4k_np(buf, n_rows, n_in):
    """Vectorised Q4_K -> (n_rows, n_in) f32; same equations as dequant_q4_K."""
    nb = n_in // QK_K
    b = np.frombuffer(buf[: n_rows * nb * Q4K_BYTES], dtype=np.uint8).reshape(-1, Q4K_BYTES)
    dv = f16_np(np.frombuffer(b[:, 0:2].tobytes(), dtype=np.uint16))[:, None]
    mn = f16_np(np.frombuffer(b[:, 2:4].tobytes(), dtype=np.uint16))[:, None]
    s = b[:, 4:16]
    sc = np.empty((b.shape[0], 8), np.float32)
    mm = np.empty_like(sc)
    for j in range(8):
        if j < 4:
            sc[:, j] = s[:, j] & 63
            mm[:, j] = s[:, j + 4] & 63
        else:
            sc[:, j] = (s[:, j + 4] & 0xF) | ((s[:, j - 4] >> 6) << 4)
            mm[:, j] = (s[:, j + 4] >> 4) | ((s[:, j] >> 6) << 4)
    qs = b[:, 16:144]
    lo, hi = qs & 0xF, qs >> 4
    out = np.empty((b.shape[0], QK_K), np.float32)
    for j in range(8):
        q = (lo if j % 2 == 0 else hi)[:, (j // 2) * 32:(j // 2) * 32 + 32]
        out[:, j * 32:(j + 1) * 32] = dv * sc[:, j:j + 1] * q - mn * mm[:, j:j + 1]
    return out.reshape(n_rows, n_in)


def read_header(path):
    """(version, {name: (ne, dtype, offset)}, header_end)."""
    import struct as S
    end = 0
    out = {}
    with open(path, "rb") as f:
        assert f.read(4) == b"GGUF"
        ver = S.unpack("<I", f.read(4))[0]
        n_tensors = S.unpack("<Q", f.read(8))[0]
        n_kv = S.unpack("<Q", f.read(8))[0]

        def val(t):
            if t == 8:
                return f.read(S.unpack("<Q", f.read(8))[0])
            if t == 9:
                et = S.unpack("<I", f.read(4))[0]
                n = S.unpack("<Q", f.read(8))[0]
                return [val(et) for _ in range(n)]
            fmt, size = SCALAR[t]
            return S.unpack(fmt, f.read(size))[0]

        for _ in range(n_kv):
            k = f.read(S.unpack("<Q", f.read(8))[0])
            val(S.unpack("<I", f.read(4))[0])
        for _ in range(n_tensors):
            name = f.read(S.unpack("<Q", f.read(8))[0]).decode()
            nd = S.unpack("<I", f.read(4))[0]
            ne = list(S.unpack(f"<{nd}Q", f.read(8 * nd)))
            dt = S.unpack("<I", f.read(4))[0]
            off = S.unpack("<Q", f.read(8))[0]
            out[name] = (ne, dt, off)
        end = f.tell()
    return ver, out, end

def row_bytes(path, info, base, row, n_out):
    ne, dt, off = info
    off += base  # GGUF tensor offsets are relative to the start of the data section
    n_in = ne[0]
    if dt == GGUF_TYPE_F32:
        with open(path, "rb") as f:
            f.seek(off + row * n_in * 4)
            return list(struct.unpack(f"<{n_in}f", f.read(n_in * 4)))
    if dt == GGUF_TYPE_F16:
        with open(path, "rb") as f:
            f.seek(off + row * n_in * 2)
            return [f16(h) for h in struct.unpack(f"<{n_in}H", f.read(n_in * 2))]
    per = Q4K_BYTES if dt == GGUF_TYPE_Q4_K else Q6K_BYTES
    nblocks = n_in // QK_K
    with open(path, "rb") as f:
        f.seek(off + row * nblocks * per)
        raw = f.read(nblocks * per)
    out = []
    fn = dequant_q4_K if dt == GGUF_TYPE_Q4_K else dequant_q6_K
    for i in range(nblocks):
        out.extend(fn(raw[i * per:(i + 1) * per]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--tensor", required=True)
    ap.add_argument("--rows", default="0")
    ap.add_argument("--check-sn", action="store_true", help="run `sn deq` and compare")
    ap.add_argument("--sn", default="target/release/sn")
    ap.add_argument("--stats", action="store_true")
    args = ap.parse_args()

    ver, tensors, header_end = read_header(args.model)
    info = tensors.get(args.tensor)
    if info is None:
        sys.exit(f"{args.tensor} not in {args.model}")
    ne, dt, off = info
    align = 32
    data_start = (header_end + align - 1) // align * align
    n_in = ne[0]
    n_out = ne[1] if len(ne) > 1 else 1
    print(f"{args.tensor}: ne={ne} type={dt} offset={off} data_start={data_start}")

    worst = 0.0
    for row in [int(r) for r in args.rows.split(",")]:
        ref = row_bytes(args.model, (ne, dt, off), data_start, row, n_out)
        assert len(ref) == n_in, (len(ref), n_in)
        if args.stats:
            rms = math.sqrt(sum(x * x for x in ref) / len(ref))
            print(f"  row {row}: rms {rms:.6f} min {min(ref):.4f} max {max(ref):.4f} "
                  f"mean {sum(ref)/len(ref):+.6f}")
        if args.check_sn:
            tmp = f"/tmp/deq-{args.tensor}-{row}.txt"
            r = subprocess.run([args.sn, "deq", "--model", args.model, "--tensor", args.tensor,
                                "--row", str(row), "--out", tmp], capture_output=True, text=True)
            if r.returncode != 0:
                sys.exit(f"sn deq failed: {r.stdout}{r.stderr}")
            mine = [float(l) for l in open(tmp) if not l.startswith("#")]
            if len(mine) != len(ref):
                sys.exit(f"length mismatch: sn {len(mine)} vs reference {len(ref)}")
            d = max(abs(a - b) for a, b in zip(mine, ref))
            scale = max(abs(x) for x in ref) or 1.0
            worst = max(worst, d / scale)
            # `sn deq` prints %.9e, so agreement is bounded by ~5e-10 relative;
            # anything larger is a real numeric difference.
            tol = 1e-6 * max(1.0, scale)
            print(f"  row {row}: max|diff| {d:.6e} (relative {d/scale:.3e}) "
                  f"{'OK' if d <= tol else 'MISMATCH'}")
            if d > tol and len(ref) > 8:
                k = max(range(len(ref)), key=lambda i: abs(mine[i] - ref[i]))
                print(f"    worst at element {k}: sn {mine[k]:.6f} vs ref {ref[k]:.6f}")
    if args.check_sn:
        print("match" if worst <= 1e-6 else f"worst relative difference {worst:.3e}")
    return 0 if worst <= 1e-6 else 1


# One model open per line, but every quantised type in the file is exercised:
# q4_k (attn_qkv_gate, ffn_gate, ssm_out, ssm_beta_alpha, attn_q, nextn.eh_proj,
# token_embd), q6_k (output, ffn_down), f32 (routers, conv taps).
MATRIX = """blk.4.attn_qkv_gate.weight 0,1
blk.0.ffn_gate.weight 0,3
output.weight 0,7
blk.52.ssm_out.weight 2
blk.4.ssm_beta_alpha.weight 5
blk.3.attn_q.weight 0,4095
blk.0.ffn_down.weight 0,12287
blk.4.hc_attn_fn.weight 0,7
blk.4.ssm_conv1d.weight 0,3
blk.56.nextn.eh_proj.weight 0
"""


def matrix_main(argv):
    if not argv:
        print("usage: dequant_ref.py <model> --matrix")
        return 2
    rc = 0
    for line in MATRIX.strip().splitlines():
        name, rows = line.split()
        print(f"=== {name} rows {rows}")
        rc |= subprocess.call([sys.executable, __file__, argv[0], "--tensor", name,
                               "--rows", rows, "--check-sn"])
    return rc


if __name__ == "__main__":
    if "--matrix" in sys.argv:
        sys.argv.remove("--matrix")
        sys.exit(matrix_main(sys.argv[1:]))
    sys.exit(main())
