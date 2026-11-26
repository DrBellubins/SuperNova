#!/usr/bin/env python3
"""pyref.py — an independent NumPy implementation of Docs/design-plan.md §2.

Not a test: a second implementation, written from the spec rather than from
crates/sn-core/src/forward.rs, used to localise disagreements between the Rust
reference and the fork. Weights are dequantised per tensor on demand (peak
memory is one matrix, ~200 MB) and stage states are dumped as .npy files that
`--compare-sn DIR` scores against `sn ref --dump` output.

  Tools/pyref.py --model M.gguf --tokens-file Docs/golden/tokens/p1-capital.ids \
      --dump-dir /tmp/pyref --limit-exec 9
  Tools/pyref.py --model M.gguf --tokens-file … --compare-embd Docs/golden/embd/p1-capital.json.gz

A full token costs tens of seconds (it streams 8.8 GB of weights), so keep probe
prompts short.
"""
import argparse, gzip, json, math, os, sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dequant_ref import read_header, f16, QK_K  # noqa: E402

F32, Q4_K, Q6_K = 0, 12, 14
MAGIC = b"GGUF"


# Dequant comes from the verified reference implementation (Tools/dequant_ref.py,
# transliterated from ggml-quants.c and cross-checked against `sn deq` row by row).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dequant_ref as dqref  # noqa: E402


def deq_q4k_np(buf, n_in, n_out):
    return dqref.deq_q4k_np(buf, n_out, n_in)


def deq_q6k_np(buf, n_in, n_out):
    return dqref.deq_q6k_np(buf, n_out, n_in)


class Gguf:
    def __init__(self, path):
        self.path = path
        ver, self.info, header_end = read_header(path)
        self.align = 32
        try:
            import struct
            with open(path, "rb") as f:
                f.seek(0)
                pass
        except Exception:
            pass
        self.data_start = (header_end + self.align - 1) // self.align * self.align
        self.f = open(path, "rb")
        self._cache = {}

    def matrix(self, name):
        """(ne1, ne0) f32, rows = outputs. Cached for the run."""
        if name in self._cache:
            return self._cache[name]
        ne, dt, off = self.info[name]
        if len(ne) == 1:
            ne = [ne[0], 1]  # 1-D vectors (norms, router bias, ssm_a) load as one row
        n_out, n_in = ne[1], ne[0]
        self.f.seek(self.data_start + off)
        if dt == F32:
            m = np.frombuffer(self.f.read(n_in * n_out * 4), dtype=np.float32).reshape(n_out, n_in)
        elif dt == Q4_K:
            m = deq_q4k_np(self.f.read(n_in // QK_K * 144 * n_out), n_in, n_out)
        elif dt == Q6_K:
            m = deq_q6k_np(self.f.read(n_in // QK_K * 210 * n_out), n_in, n_out)
        else:
            raise SystemExit(f"{name}: element type {dt} not supported by the probe")
        self._cache[name] = m
        return m

    def vec(self, name):
        return self.matrix(name).reshape(-1)

    def evict(self):
        """Drop cached weights: a token streams 5.8 GB of them and keeping them as
        f32 (23 GB) pushes the box into swap and triples the probe's runtime. Each
        tensor is used once per execution, so only the loop's second visit re-reads."""
        self._cache.clear()


# ------------------------------------------------------------------ math -----
def rms(x, eps):
    return x * np.float32(1.0 / math.sqrt(float(np.dot(x, x)) / x.size + eps))


def rms_rows(X, eps):
    return X * (1.0 / np.sqrt((X * X).sum(axis=-1, keepdims=True) / X.shape[-1] + eps)).astype(np.float32)


def softplus(x):
    x = np.asarray(x, dtype=np.float32)
    return np.where(x > 20.0, x, np.log1p(np.exp(np.clip(x, -80.0, 20.0))))


def l2n(x, eps):
    return x / np.float32(math.sqrt(float(np.dot(x, x)) + eps))


PAIRS = ((0, 1), (2, 3), (0, 2), (1, 3))


def sinkhorn(mix, sweeps):
    mix = list(mix)
    for _ in range(sweeps):
        for i, j in PAIRS:
            d = mix[i] - mix[j]
            mix[i] = float(-softplus(np.float32(-d)))
            mix[j] = float(-softplus(np.float32(d)))
            # (softplus clips its exponent internally; see softplus above)
    return np.array(mix, dtype=np.float32)


class Rope:
    def __init__(self, n_rot, base, factor, ctx_orig, beta_fast=32.0, beta_slow=1.0):
        self.n_pairs = n_rot // 2
        self.theta_scale = float(base) ** (-2.0 / n_rot)
        self.freq_scale = 1.0 / factor
        n_dims = n_rot
        start = n_dims * math.log(ctx_orig / (beta_fast * 2 * math.pi)) / (2 * math.log(base))
        end = n_dims * math.log(ctx_orig / (beta_slow * 2 * math.pi)) / (2 * math.log(base))
        self.corr = (max(0.0, start), min(n_dims - 1.0, end))
        self.mscale = 1.0 + 0.1 * math.log(1.0 / self.freq_scale)

    def table(self, pos):
        cos = np.empty(self.n_pairs, dtype=np.float32)
        sin = np.empty(self.n_pairs, dtype=np.float32)
        th = float(pos)
        for p in range(self.n_pairs):
            lo, hi = self.corr
            y = (p - lo) / max(0.001, hi - lo)
            ramp = 1.0 - min(1.0, max(0.0, y))
            ti = self.freq_scale * th
            t = ti * (1.0 - ramp) + th * ramp        # ext_factor = 1
            cos[p] = math.cos(t) * self.mscale
            sin[p] = math.sin(t) * self.mscale
            th *= self.theta_scale
        return cos, sin


def rotate(head, cos, sin):
    h = head.astype(np.float32).copy()
    v = h[: 2 * cos.size].reshape(-1, 2)
    out = np.empty_like(v)
    out[:, 0] = v[:, 0] * cos - v[:, 1] * sin
    out[:, 1] = v[:, 1] * cos + v[:, 0] * sin
    h[: 2 * cos.size] = out.reshape(-1)
    return h


# ----------------------------------------------------------------- model -----
class Ref:
    def __init__(self, g, hp, threads=0):
        self.g = g
        self.hp = hp
        self.rope = Rope(hp["n_rot"], hp["base"], hp["yarn_factor"], hp["ctx_orig"])
        self.kv = [(np.zeros((0, hp["n_kv"] * hp["d_head"]), np.float32),
                    np.zeros((0, hp["n_kv"] * hp["d_head"]), np.float32))
                   for _ in range(hp["n_exec"])]
        self.state = [np.zeros((hp["n_vh"], hp["d_v"], hp["d_k"]), np.float32)
                      for _ in range(hp["n_exec"])]
        self.conv = [np.zeros((hp["d_conv"] - 1, hp["qkv_w"]), np.float32)
                     for _ in range(hp["n_exec"])]
        self.dumps = {}

    def t(self, name):
        """2-D weights as (out, in) matrices; 1-D tensors (norms, biases, ssm_a) flat."""
        m = self.g.matrix(name)
        return m.reshape(-1) if len(self.g.info[name][0]) == 1 else m

    def route(self, a, b, blk, role, tag):
        hp = self.hp
        w = self.t(f"blk.{blk}.hc_{role}_fn.weight")
        bias = self.t(f"blk.{blk}.hc_{role}_fn.bias")
        flat = np.concatenate([a, b])
        h = rms(flat, 1e-6)
        lg = w @ h + bias
        mix = lg[4:8] if not hp["fixed_identity"] else np.ones(4, np.float32)
        if not hp["fixed_identity"]:
            mix = np.exp(sinkhorn(mix, hp["iters"]))
        sig = (1.0 / (1.0 + np.exp(-np.clip(lg, -80.0, 80.0)))).astype(np.float32)
        inp = (a * sig[0] + b * sig[1]) * np.float32(hp["pre_scale"])
        post = np.array([2.0 * float(sig[2]), 2.0 * float(sig[3])], np.float32)
        self.dumps[f"{tag}.mix"] = mix
        self.dumps[f"{tag}.post"] = post
        return inp, post, mix

    def merge(self, a, b, branch, mix, post):
        na = a * mix[0] + b * mix[1]
        nb = a * mix[2] + b * mix[3]
        return na + branch * post[0], nb + branch * post[1]

    def attention(self, ex, x, pos):
        hp, blk = self.hp, ex["blk"]
        n_h, n_kvh, dh = hp["n_h"], hp["n_kv"], hp["d_head"]
        qg = self.t(f"blk.{blk}.attn_q.weight") @ x            # [n_h * 2 * dh]
        k = self.t(f"blk.{blk}.attn_k.weight") @ x
        v = self.t(f"blk.{blk}.attn_v.weight") @ x
        qn = self.t(f"blk.{blk}.attn_q_norm.weight")
        kn = self.t(f"blk.{blk}.attn_k_norm.weight")
        cos, sin = self.rope.table(pos)
        qs, ks = [], []
        for h in range(n_h):
            seg = rms_rows(qg[h * 2 * dh:h * 2 * dh + dh][None, :], hp["rms_eps"])[0] * qn
            qs.append(rotate(seg, cos, sin))
        for h in range(n_kvh):
            seg = rms_rows(k[h * dh:(h + 1) * dh][None, :], hp["rms_eps"])[0] * kn
            ks.append(rotate(seg, cos, sin))
        k = np.concatenate(ks)
        v2 = v.reshape(n_kvh, dh)
        ki, vi = self.kv[ex["id"]]
        ki = np.concatenate([ki, k.reshape(1, -1)])
        vi = np.concatenate([vi, v.reshape(1, -1)])
        self.kv[ex["id"]] = (ki, vi)
        scale = np.float32(1.0 / math.sqrt(dh))
        out = np.empty(n_h * dh, np.float32)
        for h in range(n_h):
            kh = (h * n_kvh) // n_h
            scores = ki.reshape(-1, n_kvh, dh)[:, kh, :] @ qs[h] * scale
            p = np.exp(scores - scores.max())
            p /= p.sum()
            out[h * dh:(h + 1) * dh] = p @ vi.reshape(-1, n_kvh, dh)[:, kh, :]
        gate = (1.0 / (1.0 + np.exp(-np.clip(
            np.array([qg[h * 2 * dh + dh + i] for h in range(n_h) for i in range(dh)], np.float32),
            -80.0, 80.0)))).astype(np.float32)
        out = out * gate
        self.dumps[f"attn.{ex['il']}.out"] = out
        return self.t(f"blk.{blk}.attn_output.weight") @ out

    def gdn(self, ex, x, pos):
        hp, blk = self.hp, ex["blk"]
        n_kh, n_vh, d_k, d_v = hp["n_kh"], hp["n_vh"], hp["d_k"], hp["d_v"]
        qkv_w = hp["qkv_w"]
        z = None
        qkv = self.t(f"blk.{blk}.attn_qkv_gate.weight") @ x
        z = qkv[qkv_w:qkv_w + hp["d_inner"]]
        qkv = qkv[:qkv_w]
        ba = self.t(f"blk.{blk}.ssm_beta_alpha.weight") @ x
        beta = (1.0 / (1.0 + np.exp(-np.clip(ba[:n_vh], -80.0, 80.0)))).astype(np.float32)
        alpha = ba[n_vh:2 * n_vh]
        a_log = self.t(f"blk.{blk}.ssm_a")
        dt = self.t(f"blk.{blk}.ssm_dt.bias")
        gate = (softplus(alpha + dt) * a_log).astype(np.float32)
        # conv: window of the last d_conv-1 pre-conv rows, then the current one
        win = self.conv[ex["id"]]
        full = np.concatenate([win, qkv[None, :]])
        self.conv[ex["id"]] = full[-(hp["d_conv"] - 1):].copy()
        kern = self.t(f"blk.{blk}.ssm_conv1d.weight")            # [qkv_w, d_conv]
        cc = np.zeros(qkv_w, np.float32)
        for t in range(hp["d_conv"]):
            cc += full[t] * kern[:, hp["d_conv"] - 1 - t]
        cc = (cc / (1.0 + np.exp(-np.clip(cc, -80.0, 80.0)))).astype(np.float32)
        self.dumps[f"gdn.{ex['il']}.conv"] = cc
        q = cc[0:n_kh * d_k].reshape(n_kh, d_k)
        k = cc[n_kh * d_k:2 * n_kh * d_k].reshape(n_kh, d_k)
        v = cc[2 * n_kh * d_k:].reshape(n_vh, d_v)
        q = np.stack([l2n(q[h], hp["rms_eps"]) for h in range(n_kh)])
        k = np.stack([l2n(k[h], hp["rms_eps"]) for h in range(n_kh)])
        S = self.state[ex["id"]]
        out = np.empty(n_vh * d_v, np.float32)
        scale = np.float32(1.0 / math.sqrt(d_v))
        for h in range(n_vh):
            kh = h % n_kh
            S[h] *= np.float32(math.exp(float(gate[h])))
            pred = S[h] @ k[kh]
            delta = (v[h] - pred) * beta[h]
            S[h] += delta[:, None] * k[kh][None, :]
            out[h * d_v:(h + 1) * d_v] = (S[h] @ q[kh]) * scale
        self.state[ex["id"]] = S
        norm = self.t(f"blk.{blk}.ssm_norm.weight")
        gated = np.empty_like(out)
        d_v = hp["d_v"]
        for h in range(n_vh):
            seg = out[h * d_v:(h + 1) * d_v]
            # ssm_norm is [head_v_dim]: one weight per key slot, shared by all v-heads
            seg = rms(seg, 1e-6) * norm[:d_v]
            gated[h * d_v:(h + 1) * d_v] = seg * (z[h * d_v:(h + 1) * d_v] /
                                                  (1.0 + np.exp(-z[h * d_v:(h + 1) * d_v])))
        return self.t(f"blk.{blk}.ssm_out.weight") @ gated

    def ffn(self, ex, x):
        blk = ex["blk"]
        gate = self.t(f"blk.{blk}.ffn_gate.weight") @ x
        up = self.t(f"blk.{blk}.ffn_up.weight") @ x
        act = (gate / (1.0 + np.exp(-np.clip(gate, -80.0, 80.0)))) * up
        return self.t(f"blk.{blk}.ffn_down.weight") @ act

    def run(self, tokens, dump_dir=None, limit_exec=None):
        hp = self.hp
        embd = self.t("token_embd.weight")
        h = None
        for t, tok in enumerate(tokens):
            cur = embd[tok].copy()
            a = b = None
            for il in range(hp["n_exec"]):
                if limit_exec is not None and il >= limit_exec:
                    break
                ex = hp["execs"][il]
                blk = ex["blk"]
                self.g.evict()
                if ex["mhc"] and ex["physical"] == hp["mhc_start"]:
                    a = b = cur.copy()
                if ex["mhc"]:
                    inp, post, mix = self.route(a, b, blk, "attn", f"mhc.{il}")
                    cur = inp
                else:
                    inp = None
                h_pre = rms(cur, hp["rms_eps"]) * self.t(f"blk.{blk}.attn_norm.weight")
                branch = self.gdn(ex, h_pre, t) if ex["recr"] else self.attention(ex, h_pre, t)
                if ex["mhc"]:
                    a, b = self.merge(a, b, branch, mix, post)
                    inp2, post2, mix2 = self.route(a, b, blk, "ffn", f"mhc.{il}.ffn")
                    cur2 = rms(inp2, hp["rms_eps"]) * self.t(f"blk.{blk}.post_attention_norm.weight")
                    f = self.ffn(ex, cur2)
                    a, b = self.merge(a, b, f, mix2, post2)
                    cur = (a + b) * np.float32(0.5)
                else:
                    cur = cur + branch
                    cur2 = rms(cur, hp["rms_eps"]) * self.t(f"blk.{blk}.post_attention_norm.weight")
                    cur = self.ffn(ex, cur2) + cur
                self.dumps[f"l_out.{il}"] = cur
            if ex.get("mtp", 0) == 0:
                pass
            last = rms(cur, hp["rms_eps"]) * self.t("output_norm.weight")
            self.dumps[f"h_final.{t}"] = last
            h = self.t("output.weight") @ last
        if dump_dir:
            os.makedirs(dump_dir, exist_ok=True)
            for name, arr in self.dumps.items():
                np.save(os.path.join(dump_dir, name.replace(".", "_") + ".npy"), arr)
        return h


def read_kv(path):
    """GGUF KV metadata, iteratively (the token-array entry is ~1 MB)."""
    import struct
    def u64(f): return struct.unpack("<Q", f.read(8))[0]
    def u32(f): return struct.unpack("<I", f.read(4))[0]
    STR, ARR = 8, 9
    SCAL = {0: ("<B", 1), 1: ("<b", 1), 2: ("<H", 2), 3: ("<h", 2), 4: ("<I", 4), 5: ("<i", 4),
            6: ("<f", 4), 7: ("<B", 1), 10: ("<Q", 8), 11: ("<q", 8), 12: ("<d", 8)}
    def val(f, t):
        if t == STR:
            return f.read(u64(f)).decode("utf-8", "replace")
        if t == ARR:
            et, n = u32(f), u64(f)
            return [val(f, et) for _ in range(n)]
        fmt, size = SCAL[t]
        return struct.unpack(fmt, f.read(size))[0]
    out = {}
    with open(path, "rb") as f:
        assert f.read(4) == MAGIC
        f.read(4)          # version
        u64(f)             # n_tensors (the tensor info records follow the KV pairs)
        for _ in range(u64(f)):
            k = f.read(u64(f)).decode("utf-8", "replace")
            out[k] = val(f, u32(f))
    return out


def build_plan(model_path):
    """The execution plan, read from the metadata (independent of plan.rs on purpose)."""
    md = read_kv(model_path)
    P = "sensenova_loop."
    def get(k, d=None):
        return md[k] if k in md else d
    n_layer_all = int(get(P + "block_count"))
    loop_start = int(get(P + "loop.start"))
    loop_end = int(get(P + "loop.end"))
    loop_count = int(get(P + "loop.count"))
    mhc_start = int(get(P + "mhc.start"))
    mhc_end = int(get(P + "mhc.end"))
    iters = int(get(P + "mhc.iterations"))
    n_nextn = int(get(P + "nextn_predict_layers", 0))
    n_layer = n_layer_all - n_nextn
    width = loop_end - loop_start
    extra = (loop_count - 1) * width
    rec = get(P + "attention.recurrent_layers")
    is_recr = [bool(x) for x in rec] if rec else [((i + 1) % 4 != 0) for i in range(n_layer_all)]
    execs = []
    for il in range(n_layer + n_nextn):
        if il < loop_start:
            physical = il
        elif il < loop_end + extra:
            physical = loop_start + (il - loop_start) % width
        else:
            physical = il - extra
        execs.append({
            "id": il, "il": il, "physical": physical,
            "blk": physical if physical < loop_end else il,
            "mhc": mhc_start <= physical < mhc_end,
            "recr": bool(is_recr[il]),
            "mtp": il - (n_layer - 1) if il >= n_layer else 0,
        })
    d_state = int(get(P + "ssm.state_size"))
    d_inner = int(get(P + "ssm.inner_size"))
    hp = {
        "rms_eps": float(get(P + "attention.layer_norm_rms_epsilon", 1e-6)),
        "base": float(get(P + "rope.freq_base", 1e7)),
        "n_rot": int(get(P + "rope.dimension_count")),
        "yarn_factor": float(get(P + "rope.scaling.factor", 1.0)),
        "ctx_orig": int(get(P + "rope.scaling.original_context_length", 262144)),
        "n_h": int(get(P + "attention.head_count")),
        "n_kv": int(get(P + "attention.head_count_kv")),
        "d_head": int(get(P + "attention.key_length")),
        "n_kh": int(get(P + "ssm.group_count")),
        "n_vh": d_inner // d_state,
        "d_k": d_state, "d_v": d_state,
        "d_inner": d_inner,
        "qkv_w": 2 * d_inner,
        "d_conv": int(get(P + "ssm.conv_kernel")),
        "n_embd": int(get(P + "embedding_length")),
        "mhc_start": mhc_start, "iters": iters,
        "pre_scale": float(get(P + "mhc.pre_scale", 1.0)),
        "fixed_identity": bool(get(P + "mhc.fixed_identity", 0)),
        "n_exec": len(execs), "execs": execs,
    }
    return hp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tokens-file", required=True)
    ap.add_argument("--dump-dir")
    ap.add_argument("--limit-exec", type=int)
    ap.add_argument("--iters", type=int)
    ap.add_argument("--compare-embd")
    ap.add_argument("--compare-sn")
    args = ap.parse_args()

    hp = build_plan(args.model)
    if args.iters:
        hp["iters"] = args.iters
    tokens = [int(x) for x in open(args.tokens_file).read().replace(",", " ").split()]
    g = Gguf(args.model)
    ref = Ref(g, hp)
    logits = ref.run(tokens, args.dump_dir, args.limit_exec)
    top = np.argsort(-logits)[:5]
    print("top-5:", [(int(i), round(float(logits[i]), 4)) for i in top])

    if args.compare_embd:
        d = json.loads(gzip.open(args.compare_embd).read())
        embd = d["embd"] if "embd" in d else d["embedding"]
        for t in range(len(embd)):
            mine = ref.dumps.get(f"h_final.{t}")
            if mine is None:
                continue
            r = np.array(embd[t], dtype=np.float32)
            cos = float(np.dot(mine, r) / (np.linalg.norm(mine) * np.linalg.norm(r)))
            print(f"pos {t}: cos {cos:+.5f} rms_ref {np.sqrt((r*r).mean()):.4f} "
                  f"rms_sn {np.sqrt((mine*mine).mean()):.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
