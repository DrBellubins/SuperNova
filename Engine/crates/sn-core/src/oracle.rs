//! Scalar oracles for the Phase 4 fused kernels (`Engine/crates/sn-hip/kernels/phase4.hip`).
//!
//! Every kernel in Phase 4 has exactly one of these next to it, and every kernel test compares the
//! two on identical inputs. That is design-plan §6 P4 ("a scalar CPU reference always exists and is
//! authoritative") applied per kernel rather than end to end.
//!
//! Two conventions make a kernel and its oracle comparable:
//!
//! * **f32 accumulate, f16 operands where the kernel says so.** An f16 oracle takes values that
//!   already live in an f16 buffer (so they are exactly representable) and rounds its output with
//!   [`crate::dq::f16_round`], which is the same round-to-nearest-even the device does. Comparing an
//!   f32 oracle against an f16 kernel would measure the format, not the code.
//! * **the arithmetic order is spelled out**, because these are the operations whose rounding is
//!   load-bearing across 48 loop executions (see `Docs/numerics.md`): the mHC joint RMS is one
//!   unweighted norm over the *concatenated* streams with `eps = 1e-6` (not the model's epsilon),
//!   the balancing is exactly `mhc.iterations` sweeps in the pair order `(0,1),(2,3),(0,2),(1,3)`,
//!   and the merge uses the *pre-merge* streams.
//!
//! `nmse` / `max_abs` are the gate metrics (design-plan §7 Phase 4: NMSE <= 1e-5 per kernel,
//! <= 1e-6 on `mhc_route`'s `mix`).

use crate::dq::f16_round;
use crate::num::{rms_norm, sigmoid, silu, sinkhorn_pair, RopePairing};

/// The mHC route's RMS epsilon is hard-coded in the reference, not the model's `rms_eps` (§2.7).
pub const MHC_EPS: f32 = 1e-6;

/// Mean squared error of `a` against `b`, normalised by the energy of `b`.
///
/// This is the metric the plan gates on. It is relative, so a kernel that is right to 1e-6 relative
/// precision reports NMSE ~1e-12; the plan's 1e-5 therefore allows ~3e-3 relative error, which is
/// f16-operand territory (see [`f16_rel`]).
pub fn nmse(a: &[f32], b: &[f32]) -> f64 {
    assert_eq!(a.len(), b.len(), "nmse needs equal lengths");
    let mut num = 0f64;
    let mut den = 0f64;
    for (x, y) in a.iter().zip(b.iter()) {
        let d = (*x - *y) as f64;
        num += d * d;
        den += (*y) as f64 * (*y) as f64;
    }
    if den <= 0.0 {
        return num;
    }
    num / den
}

pub fn max_abs(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

/// The relative error an f16 operand/operand-pair introduces: half the machine epsilon of the
/// format, i.e. the *floor* below which an f16 kernel cannot match an f32 oracle. NMSE of a perfect
/// f16 kernel is therefore about `f16_rel()^2` (~5.9e-8), which is what "NMSE <= 1e-5" is really
/// allowing.
pub fn f16_rel() -> f64 {
    4.882_812_5e-4 // 2^-11, half ulp of f16
}

/// Read one stage dump written by `forward::Dumper`: a `#` header line, then one value per line.
/// This is the Phase 2 golden format, so a kernel check and a stage check read the same files
/// (`Tools/golden-check.py` reads the reference side of the same pair).
pub fn read_stage(path: &str) -> crate::error::Result<Vec<f32>> {
    let text = std::fs::read_to_string(path)
        .map_err(|e| crate::error::Error::msg(format!("{path}: {e}")))?;
    let mut out = Vec::new();
    for line in text.lines() {
        let l = line.trim();
        if l.is_empty() || l.starts_with('#') {
            continue;
        }
        out.push(l.parse::<f32>().map_err(|e| crate::error::Error::msg(format!("{path}: '{l}': {e}")))?);
    }
    Ok(out)
}

/// Round-trip a vector through f16 (what an f16 activation buffer holds).
pub fn f16(v: &[f32]) -> Vec<f32> {
    v.iter().map(|&x| f16_round(x)).collect()
}

/// Deterministic inputs for the kernel tests. Same values every run, so a measured NMSE is a
/// property of the kernel and not of the draw; the distribution is deliberately mixed (small and
/// large magnitudes, both signs, exact zeros) because a norm's error is dominated by whichever
/// term the accumulator meets first.
pub fn pattern(seed: u64, n: usize) -> Vec<f32> {
    let mut s = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        // xorshift64*: cheap, exact in u64, and no dependency on a crate.
        s ^= s >> 12;
        s = s.wrapping_mul(2560484252874878803);
        s ^= s >> 32;
        let u = ((s >> 11) & 0xffff_ffff) as f32 / 4294967296.0; // [0,1)
        let x = (u - 0.5) * 2.0; // [-1,1)
        // Every 64th element is a special case; the rest span two magnitudes.
        out.push(match i % 64 {
            0 => 0.0,
            1 => x * 1e-3,
            2 => x * 6e2,
            3 => x.signum(),
            _ => x * (1.0 + (i % 7) as f32),
        });
    }
    out
}

// ---------------------------------------------------------------------------------------------
// rmsnorm
// ---------------------------------------------------------------------------------------------

/// `y = x * rsqrtf(mean(x^2) + eps) [* w]`, one row per call.
///
/// `w == None` is the unweighted form (the mHC route's joint norm). The checkpoint's norms are
/// `rms(x) * w` with **no** `(1+w)` (§2.1).
pub fn rmsnorm(x: &[f32], w: Option<&[f32]>, eps: f32) -> Vec<f32> {
    let mut y = vec![0.0f32; x.len()];
    rms_norm(x, eps, &mut y);
    if let Some(w) = w {
        assert_eq!(w.len(), x.len());
        for (o, ww) in y.iter_mut().zip(w.iter()) {
            *o *= *ww;
        }
    }
    y
}

/// The f16 in / f16 out form the plan specifies: operands are read out of an f16 buffer, the
/// accumulation is f32, the store rounds back to f16.
pub fn rmsnorm_f16(x: &[f32], w: Option<&[f32]>, eps: f32) -> Vec<f32> {
    f16(&rmsnorm(&f16(x), w, eps))
}

// ---------------------------------------------------------------------------------------------
// elementwise
// ---------------------------------------------------------------------------------------------

/// Residual add (`a += b`, reported as a new vector).
pub fn add(a: &[f32], b: &[f32]) -> Vec<f32> {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b.iter()).map(|(x, y)| x + y).collect()
}

pub fn add_f16(a: &[f32], b: &[f32]) -> Vec<f32> {
    f16(&add(&f16(a), &f16(b)))
}

/// The FFN intermediate: `out = silu(g) * u`, SiLU on the gate branch (§2.5).
pub fn silu_mul(g: &[f32], u: &[f32]) -> Vec<f32> {
    assert_eq!(g.len(), u.len());
    g.iter().zip(u.iter()).map(|(a, b)| silu(*a) * *b).collect()
}

pub fn silu_mul_f16(g: &[f32], u: &[f32]) -> Vec<f32> {
    f16(&silu_mul(&f16(g), &f16(u)))
}

/// The same op read from **one** buffer holding `gate` then `up` (what a fused `ffn_gate/up` GEMM
/// would write): `out[i] = silu(x[i]) * x[n+i]`.
pub fn swiglu(gu: &[f32]) -> Vec<f32> {
    assert_eq!(gu.len() % 2, 0);
    let n = gu.len() / 2;
    (0..n).map(|i| silu(gu[i]) * gu[n + i]).collect()
}

pub fn swiglu_f16(gu: &[f32]) -> Vec<f32> {
    f16(&swiglu(&f16(gu)))
}

/// Elementwise sigmoid (the GDN `beta`, and anything else that wants it as its own node).
pub fn sigmoid_v(x: &[f32]) -> Vec<f32> {
    x.iter().map(|&v| sigmoid(v)).collect()
}

pub fn sigmoid_v_f16(x: &[f32]) -> Vec<f32> {
    f16(&sigmoid_v(&f16(x)))
}

// ---------------------------------------------------------------------------------------------
// rope_inplace
// ---------------------------------------------------------------------------------------------

/// Rotate the first `n_rot` dimensions of `n_heads` head vectors in place.
///
/// `rows` is a flat activation buffer; head `h` of token `t` starts at `h * head_stride +
/// t * token_stride`, so the kernel's grid maps directly onto the offsets. `cos`/`sin` are the
/// per-position rows of the `Rope` table (§2.6), `n_pairs = cos.len() = n_rot/2`.
///
/// With [`RopePairing::HalfOffset`] (what `sensenova_loop` uses) pair `p` is
/// `(p, p + n_rot/2)`; dimensions `n_rot..head_dim` are untouched, which is why the kernel is
/// in-place and only touches `n_rot` elements per head.
// The rope geometry is six numbers that all have to agree with the plan; packing them into a
// struct would only move the same list to the caller.
#[allow(clippy::too_many_arguments)]
pub fn rope_inplace(
    rows: &mut [f32],
    base: usize,
    n_tokens: usize,
    n_heads: usize,
    head_stride: usize,
    token_stride: usize,
    n_rot: usize,
    cos: &[f32],
    sin: &[f32],
    pairing: RopePairing,
) {
    assert_eq!(cos.len(), sin.len());
    assert_eq!(cos.len() * 2, n_rot, "the table is one entry per rotary pair");
    for t in 0..n_tokens {
        for h in 0..n_heads {
            let start = base + h * head_stride + t * token_stride;
            let head = &mut rows[start..start + n_rot];
            crate::num::apply_rope(head, cos, sin, pairing);
        }
    }
}

// ---------------------------------------------------------------------------------------------
// attn_gate_norm
// ---------------------------------------------------------------------------------------------

/// `out[i] = o[i] * sigmoid(gate[i])` with the gate read **strided** out of the fused `attn_q`
/// projection (§2.3): head `h`'s gate starts at `h * head_stride + gate_off` inside `qg`, so the
/// `[q | gate]` interleave is never materialised.
pub fn attn_gate_norm(o: &[f32], qg: &[f32], n_heads: usize, d_head: usize, head_stride: usize, gate_off: usize) -> Vec<f32> {
    assert_eq!(o.len() % (n_heads * d_head), 0);
    let mut out = Vec::with_capacity(o.len());
    for chunk in o.chunks(n_heads * d_head) {
        for (i, &v) in chunk.iter().enumerate() {
            let (h, c) = (i / d_head, i % d_head);
            let g = qg[h * head_stride + gate_off + c];
            out.push(v * sigmoid(g));
        }
    }
    out
}

pub fn attn_gate_norm_f16(o: &[f32], qg: &[f32], n_heads: usize, d_head: usize, head_stride: usize, gate_off: usize) -> Vec<f32> {
    f16(&attn_gate_norm(&f16(o), &f16(qg), n_heads, d_head, head_stride, gate_off))
}

// ---------------------------------------------------------------------------------------------
// mhc_route / mhc_merge
// ---------------------------------------------------------------------------------------------

/// The parameters `mhc_route` bakes in from `Hyperparams`.
#[derive(Clone, Copy, Debug)]
pub struct RouteParams {
    pub eps: f32,
    pub pre_scale: f32,
    pub sweeps: usize,
    pub fixed_identity: bool,
}

/// Everything one route application produces. `logits` is written by the kernel too: it is the
/// stage Phase 2 dumps (`mhc.<il>.logits`), so the gate can compare it.
#[derive(Clone, Debug, PartialEq)]
pub struct RouteOut {
    pub logits: Vec<f32>,
    pub input: Vec<f32>,
    pub post: [f32; 2],
    pub mix: [f32; 4],
}

/// The part of a route that runs *after* the logits exist: `in`, `post`, `mix`.
///
/// This is shared with `forward::RefModel::route`, which gets its logits from the dequantising
/// matmul path instead of the dense loop below — the two must not drift, because a route that
/// differs between the reference forward and the kernel oracle would make every later comparison
/// meaningless.
pub fn route_derive(a: &[f32], b: &[f32], logits: &[f32], p: &RouteParams) -> (Vec<f32>, [f32; 2], [f32; 4]) {
    // `plan::Dims::n_route` is 4 rows when fixed_identity is set (logits 4..7 do not exist for
    // that checkpoint shape) and 8 otherwise.
    assert!(logits.len() >= 4, "a route needs at least 4 logits, got {}", logits.len());
    assert!(
        p.fixed_identity || logits.len() >= 8,
        "a live mix needs 8 logits, got {}",
        logits.len()
    );
    let d = a.len();
    assert_eq!(b.len(), d);
    let (s0, s1) = (sigmoid(logits[0]), sigmoid(logits[1]));
    let input = (0..d).map(|i| (a[i] * s0 + b[i] * s1) * p.pre_scale).collect();
    let post = [sigmoid(logits[2]) * 2.0, sigmoid(logits[3]) * 2.0];
    let mix = if p.fixed_identity {
        // `mhc.fixed_identity == 1`: the mix matrix is the identity and logits 4..8 are unread.
        [1.0f32, 0.0, 0.0, 1.0]
    } else {
        let mut m = [logits[4], logits[5], logits[6], logits[7]];
        for _ in 0..p.sweeps {
            for (i, j) in [(0usize, 1usize), (2, 3), (0, 2), (1, 3)] {
                sinkhorn_pair(&mut m, i, j);
            }
        }
        m.map(|v| v.exp())
    };
    (input, post, mix)
}

/// One mHC route, exactly §2.7:
///
/// ```text
/// f  = rms([a ; b], 1e-6)                   one unweighted norm over the concatenation
/// l  = W f + bias                           W is [n_route, 2d], k-contiguous
/// in = (a * sigmoid(l0) + b * sigmoid(l1)) * pre_scale
/// post = [2 * sigmoid(l2), 2 * sigmoid(l3)]
/// mix = exp(sinkhorn_sweeps(l[4..8], sweeps))   (or [1,0,0,1] when fixed_identity)
/// ```
///
/// The device kernel defers the `f` scale factor into the dot (`scale * sum(w*v)` instead of
/// `sum(w * (scale*v))`); that is algebraically identical and inside every gate, but it is the one
/// place where the oracle and the kernel are not bit-identical by construction, so it is stated
/// here rather than discovered later.
pub fn mhc_route(a: &[f32], b: &[f32], w: &[f32], bias: &[f32], p: &RouteParams) -> RouteOut {
    assert_eq!(a.len(), b.len());
    let d = a.len();
    let n_route = bias.len();
    assert_eq!(w.len(), n_route * 2 * d, "router must be [n_route][2d], k-contiguous");

    let mut flat = Vec::with_capacity(2 * d);
    flat.extend_from_slice(a);
    flat.extend_from_slice(b);
    let f = rmsnorm(&flat, None, p.eps);
    let mut logits = vec![0.0f32; n_route];
    for r in 0..n_route {
        let row = &w[r * 2 * d..(r + 1) * 2 * d];
        let mut acc = 0.0f32;
        for (fv, wv) in f.iter().zip(row.iter()) {
            acc += *fv * *wv;
        }
        logits[r] = acc + bias[r];
    }

    let (input, post, mix) = route_derive(a, b, &logits, p);
    RouteOut { logits, input, post, mix }
}

/// The merge after one branch: both lines read the **pre-merge** streams (§2.7).
pub fn mhc_merge(a: &[f32], b: &[f32], branch: &[f32], mix: &[f32; 4], post: &[f32; 2]) -> (Vec<f32>, Vec<f32>) {
    assert_eq!(a.len(), b.len());
    assert_eq!(branch.len(), a.len());
    let na: Vec<f32> = (0..a.len()).map(|i| a[i] * mix[0] + b[i] * mix[1] + branch[i] * post[0]).collect();
    let nb: Vec<f32> = (0..a.len()).map(|i| a[i] * mix[2] + b[i] * mix[3] + branch[i] * post[1]).collect();
    (na, nb)
}

/// The value the trunk exposes while inside the mHC range: `0.5 * (a + b)` (§2.7).
///
/// This is what the *last* mHC execution hands to the next one when that next execution is outside
/// the mHC range (and, if the trunk ended inside it, to the output head). The exit blocks of this
/// checkpoint are outside the range, so a program that exposes only at the end of the trunk feeds
/// the head the loop's streams instead of block 55's output.
pub fn mhc_expose(a: &[f32], b: &[f32]) -> Vec<f32> {
    a.iter().zip(b.iter()).map(|(x, y)| (x + y) * 0.5).collect()
}

/// The other end of the mHC region: the streams do not exist until the first mHC execution, where
/// they are seeded from the hidden state that reaches it (`a = b = cur`, §2.7). `forward.rs` does
/// this with `clone`; the kernel exists so the decode program has an op that does it too - without
/// one, the first route reads buffers nothing ever wrote.
pub fn mhc_seed(cur: &[f32]) -> (Vec<f32>, Vec<f32>) {
    (cur.to_vec(), cur.to_vec())
}

// ============================================================================================
// Phase 6 oracles: the caches and the two recurrent inner loops.
//
// One per kernel in `Engine/crates/sn-hip/kernels/phase6.hip`, same operand order, so `sn gpu phase6`
// measures the thing the engine launches. `forward.rs` already contains this math inside the whole
// model; these are the same formulas with the smallest possible context around them, because a
// kernel-level failure has to point at one kernel.
// ============================================================================================

/// One row of a Q4_K matrix (the embedding table's type), via the same field walk `dq` uses.
pub fn row_gather_q4k(bytes: &[u8], k: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; k];
    for b in 0..k / 256 {
        crate::dq::q4k_dequant(&bytes[b * 144..b * 144 + 144], &mut out[b * 256..b * 256 + 256]);
    }
    out
}

/// Per-head Q/K RMS norm with the checkpoint's gains (§2.3). Returns both tensors.
#[allow(clippy::too_many_arguments)]
pub fn qk_head_norm(
    qg: &[f32],
    qw: &[f32],
    kx: &[f32],
    kw: &[f32],
    n_qh: usize,
    n_kh: usize,
    d: usize,
    q_stride: usize,
    eps: f32,
) -> (Vec<f32>, Vec<f32>) {
    let mut q = qg.to_vec();
    let mut kk = kx.to_vec();
    for h in 0..n_qh {
        let base = h * q_stride;
        let mut out = vec![0.0f32; d];
        crate::num::rms_norm_w(&q[base..base + d], eps, qw, &mut out);
        q[base..base + d].copy_from_slice(&out);
    }
    for h in 0..n_kh {
        let base = h * d;
        let mut out = vec![0.0f32; d];
        crate::num::rms_norm_w(&kk[base..base + d], eps, kw, &mut out);
        kk[base..base + d].copy_from_slice(&out);
    }
    (q, kk)
}

/// Pack one KV row the way `cache::KvKind` stores it (quant array, then f32 scales).
pub fn kv_pack(kind: &str, src: &[f32]) -> Vec<u8> {
    let nb = src.len().div_ceil(32);
    match kind {
        "f16" => src.iter().flat_map(|v| crate::dq::f32_to_f16(*v).to_le_bytes()).collect(),
        "q8_0" => {
            let mut out = vec![0u8; nb * 32];
            for b in 0..nb {
                let blk = &src[b * 32..(b + 1) * 32].to_vec();
                let amax = blk.iter().fold(0.0f32, |a, v| a.max(v.abs()));
                let sc = amax / 127.0;
                out.extend_from_slice(&sc.to_le_bytes());
                for (j, v) in blk.iter().enumerate() {
                    let q = (*v / sc).round() as i32;
                    out[b * 32 + j] = (q.clamp(-127, 127) as i8) as u8;
                }
            }
            out
        }
        "q4_0" => {
            let mut qs = vec![0u8; nb * 16];
            let mut sc = Vec::new();
            for b in 0..nb {
                let blk = &src[b * 32..(b + 1) * 32].to_vec();
                let amax = blk.iter().fold(0.0f32, |a, v| a.max(v.abs()));
                let s = amax / 7.0;
                sc.extend_from_slice(&s.to_le_bytes());
                for j in 0..16 {
                    let lo = (blk[2 * j] / s).round() as i32 + 8;
                    let hi = (blk[2 * j + 1] / s).round() as i32 + 8;
                    qs[b * 16 + j] = lo.clamp(0, 15) as u8 | ((hi.clamp(0, 15) as u8) << 4);
                }
            }
            qs.extend(sc);
            qs
        }
        other => panic!("no KV packing for {other}"),
    }
}

/// Read a packed row back (the oracle for `sn_attn_decode_*`, which does the same unpacking).
pub fn kv_unpack(kind: &str, row: &[u8], values: usize) -> Vec<f32> {
    match kind {
        "f16" => row[..values * 2]
            .chunks_exact(2)
            .map(|c| crate::dq::f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
            .collect(),
        "q8_0" => {
            let nb = values.div_ceil(32);
            let mut out = Vec::with_capacity(values);
            for b in 0..nb {
                let s = f32::from_le_bytes([
                    row[nb * 32 + b * 4],
                    row[nb * 32 + b * 4 + 1],
                    row[nb * 32 + b * 4 + 2],
                    row[nb * 32 + b * 4 + 3],
                ]);
                for j in 0..32 {
                    out.push((row[b * 32 + j] as i8) as f32 * s);
                }
            }
            out.truncate(values);
            out
        }
        "q4_0" => {
            let nb = values.div_ceil(32);
            let mut out = Vec::with_capacity(values);
            for b in 0..nb {
                let s = f32::from_le_bytes([
                    row[nb * 16 + b * 4],
                    row[nb * 16 + b * 4 + 1],
                    row[nb * 16 + b * 4 + 2],
                    row[nb * 16 + b * 4 + 3],
                ]);
                for j in 0..16 {
                    let byte = row[b * 16 + j];
                    out.push(((byte & 0xF) as f32 - 8.0) * s);
                    out.push((((byte >> 4) & 0xF) as f32 - 8.0) * s);
                }
            }
            out.truncate(values);
            out
        }
        other => panic!("no KV packing for {other}"),
    }
}

/// Causal attention for one query over the K/V rows of its own cache (§2.3), two-pass softmax as
/// the reference forward does it.
pub fn attn_decode(q: &[f32], ks: &[&[f32]], vs: &[&[f32]], scale: f32) -> Vec<f32> {
    let d = q.len();
    let mut p = vec![0.0f32; ks.len()];
    let mut max = f32::NEG_INFINITY;
    for (t, k) in ks.iter().enumerate() {
        let s: f32 = q.iter().zip(k.iter()).map(|(a, b)| a * b).sum();
        p[t] = s * scale;
        if p[t] > max {
            max = p[t];
        }
    }
    let mut sum = 0.0f32;
    for v in p.iter_mut() {
        *v = (*v - max).exp();
        sum += *v;
    }
    let mut out = vec![0.0f32; d];
    for (t, p) in p.iter().enumerate() {
        let w = p / sum;
        for i in 0..d {
            out[i] += w * vs[t][i];
        }
    }
    out
}

/// Causal depthwise conv over the raw qkv plus SiLU (§2.4). Returns the output and the new window.
pub fn gdn_conv_silu(qkv: &[f32], cw: &[f32], window: &[f32], d_conv: usize) -> (Vec<f32>, Vec<f32>) {
    let ch = qkv.len();
    let rows = d_conv - 1;
    let mut out = vec![0.0f32; ch];
    let mut next = vec![0.0f32; rows * ch];
    for c in 0..ch {
        let mut s = cw[c * d_conv + rows] * qkv[c];
        for t in 0..rows {
            s += cw[c * d_conv + t] * window[t * ch + c];
            next[t * ch + c] = if t + 1 < rows { window[(t + 1) * ch + c] } else { qkv[c] };
        }
        out[c] = crate::num::silu(s);
    }
    (out, next)
}

/// L2-normalise q/k head blocks (§2.4).
pub fn gdn_l2norm(x: &[f32], dk: usize, eps: f32) -> Vec<f32> {
    let mut out = x.to_vec();
    for h in 0..(x.len() / dk) {
        crate::num::l2_norm(&mut out[h * dk..h * dk + dk], eps);
    }
    out
}

/// One token of the delta rule (§2.4), decay -> predict -> write -> read.
#[allow(clippy::too_many_arguments)]
pub fn gdn_decode_step(
    conv: &[f32],
    ba: &[f32],
    ssm_a: &[f32],
    dt_bias: &[f32],
    state: &[f32],
    n_v: usize,
    d_v: usize,
    d_k: usize,
    n_k: usize,
) -> (Vec<f32>, Vec<f32>) {
    let mut st = state.to_vec();
    let mut out = vec![0.0f32; n_v * d_v];
    for h in 0..n_v {
        let kh = h % n_k;
        let qt = &conv[kh * d_k..kh * d_k + d_k];
        let kt = &conv[n_k * d_k + kh * d_k..n_k * d_k + kh * d_k + d_k];
        let vt = &conv[2 * n_k * d_k + h * d_v..2 * n_k * d_k + h * d_v + d_v];
        let gate = ssm_a[h] * crate::num::softplus(ba[n_v + h] + dt_bias[h]);
        let decay = gate.exp();
        let beta = ba[h];
        let base = h * d_v * d_k;
        for i in 0..d_v {
            let row = &mut st[base + i * d_k..base + (i + 1) * d_k];
            let mut pred = 0.0f32;
            for j in 0..d_k {
                row[j] *= decay;
                pred += row[j] * kt[j];
            }
            let delta = (vt[i] - pred) * beta;
            for j in 0..d_k {
                row[j] += kt[j] * delta;
            }
            out[h * d_v + i] = row.iter().zip(qt.iter()).map(|(a, b)| a * b).sum::<f32>() / (d_k as f32).sqrt();
        }
    }
    (st, out)
}

/// The gated per-head norm on the GDN output (§2.4).
pub fn gdn_gate_norm(core: &[f32], w: &[f32], z: &[f32], d_v: usize, eps: f32) -> Vec<f32> {
    let mut out = vec![0.0f32; core.len()];
    for h in 0..core.len() / d_v {
        let base = h * d_v;
        let mut normed = vec![0.0f32; d_v];
        crate::num::rms_norm_w(&core[base..base + d_v], eps, w, &mut normed);
        for i in 0..d_v {
            out[base + i] = normed[i] * crate::num::silu(z[base + i]);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::num::{rms_norm, sinkhorn_sweeps};

    #[test]
    fn nmse_is_relative_and_zero_for_identical() {
        let a = [1.0f32, -2.0, 3.0];
        assert_eq!(nmse(&a, &a), 0.0);
        // 1% on every element of a vector with energy 14 -> 0.0001 * 14 / 14
        let b = [1.01, -2.02, 3.03];
        assert!((nmse(&b, &a) - 1.0e-4).abs() < 1e-6, "{}", nmse(&b, &a));
        assert!((max_abs(&b, &a) - 0.03).abs() < 1e-6);
        // A zero reference must not silently report 0.
        assert!(nmse(&[1.0, 1.0], &[0.0, 0.0]) > 0.0);
    }

    #[test]
    fn f16_operands_put_the_kernel_vs_oracle_gap_at_f16_rel() {
        // The point of the f16 oracles: an f16 kernel compared against an f32 oracle measures the
        // format. With an f16 oracle the same data sits at the format's floor.
        let x = pattern(7, 4096);
        let w: Vec<f32> = (0..x.len()).map(|i| 1.0 + 0.25 * ((i % 5) as f32)).collect();
        let exact = rmsnorm(&x, Some(&w), 1e-6);
        let as_f16 = rmsnorm_f16(&x, Some(&w), 1e-6);
        let n = nmse(&as_f16, &exact);
        assert!(n > 0.0 && n < (2.0 * f16_rel()).powi(2), "f16 norm NMSE {n:e}");
        assert!(n > (0.3 * f16_rel()).powi(2), "NMSE {n:e} is suspiciously small - is the f16 round a no-op?");
    }

    #[test]
    fn rmsnorm_matches_num_and_the_weighted_form_has_no_one_plus_w() {
        let x = [3.0f32, -4.0, 12.0, 0.0];
        let w = [1.0f32, 2.0, 0.5, 7.0];
        let mut want = vec![0.0f32; 4];
        rms_norm(&x, 1e-6, &mut want);
        let unweighted = rmsnorm(&x, None, 1e-6);
        assert_eq!(unweighted, want);
        let scaled: Vec<f32> = want.iter().zip(w.iter()).map(|(a, b)| a * b).collect();
        assert_eq!(rmsnorm(&x, Some(&w), 1e-6), scaled);
    }

    #[test]
    fn elementwise_oracles_match_num_element_by_element() {
        let g = pattern(3, 128);
        let u = pattern(4, 128);
        let sm = silu_mul(&g, &u);
        let sw = swiglu(&[g.clone(), u.clone()].concat());
        for i in 0..128 {
            assert_eq!(sm[i], sw[i], "silu_mul and swiglu must agree on the same data");
        }
        let sv = sigmoid_v(&g);
        assert!(sv.iter().all(|v| (0.0f32..=1.0).contains(v)));
        let a = pattern(5, 128);
        let b = pattern(6, 128);
        assert_eq!(add(&a, &b)[7], a[7] + b[7]);
    }

    /// The rope oracle is `num::apply_rope` per head; the kernel does the same with a grid, so this
    /// pins the offset arithmetic (head stride vs token stride) that is easy to invert.
    #[test]
    fn rope_oracle_rotates_only_the_rotated_span_at_the_right_offsets() {
        let (n_heads, d_head, n_rot) = (4usize, 8usize, 4usize);
        let token_stride = n_heads * d_head;
        let mut rows = vec![0.0f32; token_stride];
        for (i, r) in rows.iter_mut().enumerate() {
            *r = i as f32;
        }
        let before = rows.clone();
        let cos = [0.0f32, 1.0];
        let sin = [1.0f32, 0.0];
        rope_inplace(&mut rows, 0, 1, n_heads, d_head, token_stride, n_rot, &cos, &sin, RopePairing::HalfOffset);
        // head h rotates elements h*d_head + 0 with h*d_head + 2 (half offset, n_rot/2 == 2).
        for h in 0..n_heads {
            let s = h * d_head;
            assert_eq!(rows[s], -before[s + 2]);
            assert_eq!(rows[s + 2], before[s]);
            // dims 4..8 of the head are untouched (partial rope)
            assert_eq!(&rows[s + 4..s + d_head], &before[s + 4..s + d_head]);
        }
        // A base offset shifts every row by the same amount.
        let mut shifted = vec![0.0f32; token_stride + d_head];
        shifted[d_head..].copy_from_slice(&before);
        rope_inplace(&mut shifted, d_head, 1, n_heads, d_head, token_stride, n_rot, &cos, &sin, RopePairing::HalfOffset);
        assert_eq!(&shifted[d_head..], &rows[..]);
    }

    #[test]
    fn attn_gate_norm_reads_the_interleaved_gate() {
        // head stride 2*d_head, gate at offset d_head: the [q | gate] layout of attn_q.
        let (n_heads, d) = (2usize, 3usize);
        let o: Vec<f32> = (0..n_heads * d).map(|i| (i as f32) + 1.0).collect();
        let mut qg = vec![0.0f32; n_heads * 2 * d];
        for (i, v) in qg.iter_mut().enumerate() {
            *v = (i as f32) * 0.5;
        }
        let out = attn_gate_norm(&o, &qg, n_heads, d, 2 * d, d);
        for (i, &got) in out.iter().enumerate() {
            let (h, c) = (i / d, i % d);
            let want = o[i] * sigmoid(qg[h * 2 * d + d + c]);
            assert!((got - want).abs() < 1e-9, "{i}: {got} vs {want}");
        }
    }

    /// The route oracle must agree with the Phase 2 reference path, which is the same math in
    /// `forward::RefModel::route` but driven through `num::` helpers directly.
    #[test]
    fn mhc_route_matches_the_reference_primitives() {
        let d = 8;
        let a = pattern(11, d);
        let b = pattern(12, d);
        let w: Vec<f32> = (0..8 * 2 * d).map(|i| ((i % 13) as f32) * 0.01 - 0.06).collect();
        let bias = [0.5f32, -0.25, 1.0, -1.0, 0.7, -0.7, 0.1, 0.0];
        let p = RouteParams { eps: MHC_EPS, pre_scale: 1.0, sweeps: 20, fixed_identity: false };
        let got = mhc_route(&a, &b, &w, &bias, &p);

        // logits, recomputed the long way
        let mut flat = Vec::new();
        flat.extend_from_slice(&a);
        flat.extend_from_slice(&b);
        let mut f = vec![0.0f32; 2 * d];
        rms_norm(&flat, MHC_EPS, &mut f);
        for r in 0..8 {
            let mut acc = 0.0f32;
            for k in 0..2 * d {
                acc += f[k] * w[r * 2 * d + k];
            }
            assert!((got.logits[r] - (acc + bias[r])).abs() < 1e-6, "logit {r}");
        }
        // mix: exp of the 20-sweep iterate of logits[4..8]
        let mut m = [got.logits[4], got.logits[5], got.logits[6], got.logits[7]];
        sinkhorn_sweeps(&mut m, 20);
        for (i, &mi) in m.iter().enumerate() {
            assert!((got.mix[i] - mi.exp()).abs() < 1e-9, "mix[{i}]");
        }
        // the mixing matrix is doubly stochastic in probability space: rows and columns sum to 1
        // up to the 20-sweep residual, and every entry is in (0, 1].
        let s = got.mix[0] + got.mix[1] + got.mix[2] + got.mix[3];
        assert!((s - 2.0).abs() < 5e-2, "sum of mixes {s}");
        assert!(got.mix.iter().all(|&v| v > 0.0 && v <= 1.0));
        // post and input
        assert!((got.post[0] - 2.0 * sigmoid(got.logits[2])).abs() < 1e-9);
        let s0 = sigmoid(got.logits[0]);
        let s1 = sigmoid(got.logits[1]);
        for i in 0..d {
            assert!((got.input[i] - (a[i] * s0 + b[i] * s1)).abs() < 1e-9);
        }
    }

    #[test]
    fn mhc_route_honours_fixed_identity_and_pre_scale() {
        let d = 4;
        let a = pattern(13, d);
        let b = pattern(14, d);
        let w: Vec<f32> = (0..8 * 2 * d).map(|i| (i as f32) * 1e-3).collect();
        let bias = [0.0f32; 8];
        let fixed = mhc_route(
            &a,
            &b,
            &w,
            &bias,
            &RouteParams { eps: MHC_EPS, pre_scale: 1.0, sweeps: 20, fixed_identity: true },
        );
        assert_eq!(fixed.mix, [1.0f32, 0.0, 0.0, 1.0]);
        let scaled = mhc_route(
            &a,
            &b,
            &w,
            &bias,
            &RouteParams { eps: MHC_EPS, pre_scale: 0.25, sweeps: 20, fixed_identity: false },
        );
        let unit = mhc_route(
            &a,
            &b,
            &w,
            &bias,
            &RouteParams { eps: MHC_EPS, pre_scale: 1.0, sweeps: 20, fixed_identity: false },
        );
        for i in 0..d {
            assert!((scaled.input[i] - unit.input[i] * 0.25).abs() < 1e-9);
        }
    }

    #[test]
    fn mhc_merge_uses_the_pre_merge_streams() {
        let a = [1.0f32, 2.0];
        let b = [3.0f32, 4.0];
        let br = [10.0f32, 20.0];
        let mix = [0.5f32, 0.5, 0.5, 0.5];
        let post = [1.0f32, -1.0];
        let (na, nb) = mhc_merge(&a, &b, &br, &mix, &post);
        assert_eq!(na, [0.5 * 1.0 + 0.5 * 3.0 + 10.0, 0.5 * 2.0 + 0.5 * 4.0 + 20.0]);
        assert_eq!(nb, [0.5 * 1.0 + 0.5 * 3.0 - 10.0, 0.5 * 2.0 + 0.5 * 4.0 - 20.0]);
        // The identity mix leaves the streams alone apart from the branch.
        let (ia, ib) = mhc_merge(&a, &b, &br, &[1.0, 0.0, 0.0, 1.0], &[1.0, 1.0]);
        assert_eq!(ia, [11.0, 22.0]);
        assert_eq!(ib, [13.0, 24.0]);
    }

    /// The two ends of the mHC region, pinned so the decode program's seed/expose ops have a
    /// spec to be compared against (design-plan P4: every kernel lands with an oracle).
    #[test]
    fn mhc_seed_and_expose_are_the_two_ends_of_the_region() {
        let cur = pattern(21, 16);
        let (a, b) = mhc_seed(&cur);
        assert_eq!(a, cur);
        assert_eq!(b, cur);
        // Exposing freshly seeded streams is the identity, which is why seeding at all is what
        // decides whether execution `mhc_start` sees the entry blocks' state or uninitialised memory.
        assert_eq!(mhc_expose(&a, &b), cur);
        // After a merge the exposure is the mix-weighted average, not either stream.
        let br = pattern(22, 16);
        let (na, nb) = mhc_merge(&a, &b, &br, &[0.5, 0.5, 0.5, 0.5], &[1.0, 1.0]);
        let ex = mhc_expose(&na, &nb);
        assert_ne!(ex, cur);
        // Seeded streams are equal, so 0.5*(a',b') = cur + br here: the branch, once, not twice.
        for i in 0..cur.len() {
            assert!((ex[i] - (cur[i] + br[i])).abs() < 1e-6, "{i}");
        }
        assert_eq!(na, nb);
    }

    #[test]
    fn pattern_is_deterministic_and_well_ranged() {
        let a = pattern(9, 257);
        let b = pattern(9, 257);
        assert_eq!(a, b);
        assert_ne!(a, pattern(10, 257));
        assert_eq!(a[0], 0.0);
        assert!(a.iter().all(|v| v.is_finite()));
        assert!(a.iter().any(|v| *v > 10.0) && a.iter().any(|v| *v < -10.0));
        assert!(a.iter().any(|v| v.abs() > 0.0 && v.abs() < 1e-2));
        // Different seeds must not produce a permutation of the same values.
        assert_ne!(a[8], pattern(10, 257)[8]);
    }
}

