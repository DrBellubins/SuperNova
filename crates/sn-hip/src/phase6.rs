//! Phase 6: registration and correctness for the cache and recurrence kernels.
//!
//! Same shape as [`crate::gemm`] for Phase 5: a table of the symbols in
//! `kernels/phase6.hip`, a typed view for the decode compiler and the check harness, and one
//! function that scores every kernel against its `sn-core::oracle` twin. The *gate* this phase
//! ships is `phase6_checks` plus the end-to-end forward in [`crate::driver`]; the numbers land in
//! `Docs/bench.md` §4.

use sn_core::error::{Error, Result};
use sn_core::oracle;

use crate::check::{Check, Harness};
use crate::hip::Device;
use crate::kernels::{DType, Kernel, KernelRegistry};

/// One thread per state row of the delta rule (`d_v` = 128 in this checkpoint). The kernel needs no
/// cross-lane reduction, so the workgroup size is the state's row count, not `ops::BLOCK`.
pub const GDN_DECODE_BLOCK: u32 = 128;

/// The attention decode workgroup: 8 wavefronts, each owning one token of an 8-token tile, with
/// `d_head / 32` accumulator registers per lane. `d_head` is 256 here, which is also why the
/// shared tiles are sized 8 x 256.
pub const ATTN_BLOCK: u32 = 256;

/// Tokens per K/V tile in `sn_attn_decode_f32`; the .hip's `SN_KV_TILE`.
pub const KV_TILE: usize = 8;

/// Everything in `phase6.hip` that uses `SN_BLOCK` for its shared-memory reduction.
pub const PHASE6_BLOCK: u32 = crate::ops::BLOCK;

pub const PHASE6_SPECS: &[(&str, DType, u32)] = &[
    ("sn_row_gather_q4_k", DType::Any, PHASE6_BLOCK),
    ("sn_qk_head_norm_f32", DType::F32, PHASE6_BLOCK),
    ("sn_kv_append", DType::Any, PHASE6_BLOCK),
    ("sn_attn_decode_f32", DType::F32, ATTN_BLOCK),
    ("sn_gdn_conv_silu_f32", DType::F32, PHASE6_BLOCK),
    ("sn_gdn_l2norm_f32", DType::F32, PHASE6_BLOCK),
    ("sn_gdn_decode_f32", DType::F32, GDN_DECODE_BLOCK),
    ("sn_gdn_gate_norm_f32", DType::F32, PHASE6_BLOCK),
    ("sn_gdn_snapshot", DType::F32, PHASE6_BLOCK),
];

pub fn register_phase6(dev: &Device, reg: &mut KernelRegistry) -> Result<usize> {
    let mut n = 0;
    for (name, dtype, block) in PHASE6_SPECS {
        reg.register(
            dev,
            &crate::kernels::Key { name: (*name).to_string(), shape: vec![], dtype: *dtype },
            name,
            [*block, 1, 1],
        )?;
        n += 1;
    }
    Ok(n)
}

#[derive(Clone, Copy, Debug)]
pub struct Phase6 {
    pub row_gather_q4_k: Kernel,
    pub qk_head_norm: Kernel,
    pub kv_append: Kernel,
    pub attn_decode: Kernel,
    pub gdn_conv: Kernel,
    pub gdn_l2norm: Kernel,
    pub gdn_decode: Kernel,
    pub gdn_gate_norm: Kernel,
    pub snapshot: Kernel,
}

impl Phase6 {
    pub fn resolve(reg: &KernelRegistry) -> Result<Phase6> {
        let g = |name: &str| -> Result<Kernel> {
            reg.find(name, &[], DType::F32)
                .or_else(|| reg.find(name, &[], DType::Any))
                .copied()
                .ok_or_else(|| Error::msg(format!("{name} is not registered (run register_phase6)")))
        };
        Ok(Phase6 {
            row_gather_q4_k: g("sn_row_gather_q4_k")?,
            qk_head_norm: g("sn_qk_head_norm_f32")?,
            kv_append: g("sn_kv_append")?,
            attn_decode: g("sn_attn_decode_f32")?,
            gdn_conv: g("sn_gdn_conv_silu_f32")?,
            gdn_l2norm: g("sn_gdn_l2norm_f32")?,
            gdn_decode: g("sn_gdn_decode_f32")?,
            gdn_gate_norm: g("sn_gdn_gate_norm_f32")?,
            snapshot: g("sn_gdn_snapshot")?,
        })
    }
}

fn i(v: usize) -> crate::program::Arg {
    crate::program::Arg::I32(v as i32)
}
fn f(v: f32) -> crate::program::Arg {
    crate::program::Arg::F32(v)
}
fn le32(v: i32) -> [u8; 4] {
    v.to_le_bytes()
}

fn score(name: &str, got: &[f32], want: &[f32], gate: f64) -> Check {
    Check {
        name: name.to_string(),
        against: "oracle",
        n: got.len(),
        nmse: oracle::nmse(got, want),
        max_abs: oracle::max_abs(got, want),
        gate,
    }
}

/// Inputs with a fixed seed and a fixed shape, so the numbers in `Docs/bench.md` §4 reproduce.
fn pattern(seed: u64, n: usize) -> Vec<f32> {
    oracle::pattern(seed, n)
}

/// Every Phase 6 kernel against its oracle, on the checkpoint's real geometry.
pub fn phase6_checks(p6: &Phase6, h: &mut Harness, dims: &sn_core::plan::Dims, d_conv: usize, rms_eps: f32) -> Result<Vec<Check>> {
    let mut out = Vec::new();

    // ---- embedding row gather ------------------------------------------------------------
    {
        let k = 4096usize;
        // A decodable Q4_K row: 16 super-blocks x 144 B. The `d`/`dmin` fields are written as real
        // f16 values first - arbitrary bytes there are inf/NaN, and an NMSE of NaN says nothing
        // about the kernel.
        let rb = 16 * 144;
        let bytes = pattern(0x51, rb);
        let mut raw: Vec<u8> = bytes.iter().map(|v| (*v as i32 & 0xFF) as u8).collect();
        for b in 0..16 {
            let (d, dm) = (0x2E66u16, 0x2B28u16); // 0.01 and 0.004 in f16
            raw[b * 144] = (d & 0xFF) as u8;
            raw[b * 144 + 1] = (d >> 8) as u8;
            raw[b * 144 + 2] = (dm & 0xFF) as u8;
            raw[b * 144 + 3] = (dm >> 8) as u8;
        }
        let w = h.alloc(rb)?;
        h.put_bytes(w, &raw)?;
        let y = h.alloc(k * 4)?;
        let id = h.alloc(4)?;
        h.put_bytes(id, &le32(0i32))?;
        h.run("row_gather", &p6.row_gather_q4_k, [1, 1, 1], 0, &[h.addr(w), h.addr(id), h.addr(y), i(k)])?;
        let got = h.get_f32(y, k)?;
        let want = oracle::row_gather_q4k(&raw, k);
        out.push(score("row_gather_q4_k embd row", &got, &want, 1e-10));
    }

    // ---- per-head Q/K norm ---------------------------------------------------------------
    {
        let (nq, nkv, d) = (dims.attn_n_q_heads, dims.attn_n_kv_heads, dims.attn_d_head);
        let qs = 2 * d;
        let qg = h.alloc(nq * qs * 4)?;
        let kx = h.alloc(nkv * d * 4)?;
        let qw = h.alloc(d * 4)?;
        let kw = h.alloc(d * 4)?;
        let qin = pattern(7, nq * qs);
        let kin = pattern(11, nkv * d);
        let qwn = pattern(13, d);
        let kwn = pattern(17, d);
        h.put_f32(qg, &qin)?;
        h.put_f32(kx, &kin)?;
        h.put_f32(qw, &qwn)?;
        h.put_f32(kw, &kwn)?;
        h.run(
            "qk_head_norm",
            &p6.qk_head_norm,
            [(nq + nkv) as u32, 1, 1],
            0,
            &[h.addr(qg), h.addr(qw), h.addr(kx), h.addr(kw), i(nq), i(nkv), i(d), i(qs), f(rms_eps)],
        )?;
        let (want_q, want_k) = oracle::qk_head_norm(&qin, &qwn, &kin, &kwn, nq, nkv, d, qs, rms_eps);
        let got_q = h.get_f32(qg, nq * qs)?;
        let got_k = h.get_f32(kx, nkv * d)?;
        // Only the first `d` values of each q head block are the query; the gate half must not move.
        let mut gate_ok = true;
        for hh in 0..nq {
            for j in d..qs {
                if got_q[hh * qs + j] != qin[hh * qs + j] {
                    gate_ok = false;
                }
            }
        }
        out.push(score("qk_head_norm q", &got_q, &want_q, 1e-10));
        out.push(score("qk_head_norm k", &got_k, &want_k, 1e-10));
        out.push(Check {
            name: "qk_head_norm leaves the gate half untouched".into(),
            against: "oracle",
            n: nq * (qs - d),
            nmse: if gate_ok { 0.0 } else { 1.0 },
            max_abs: 0.0,
            gate: 0.0,
        });
    }

    // ---- KV append + attention decode, for every cache type ------------------------------
    {
        let (nq, nkv, d) = (dims.attn_n_q_heads, dims.attn_n_kv_heads, dims.attn_d_head);
        let values = dims.n_kv; // n_kv_heads * d_head
        let ntok = 20usize; // not a multiple of the tile, on purpose
        let qs = 2 * d;
        for kind in ["f16", "q8_0", "q4_0"] {
            let kind_enum = crate::cache::KvKind::parse(kind)?;
            let row_bytes = kind_enum.row_bytes(values);
            let rows = ntok;
            let region = h.alloc(rows * 2 * row_bytes + 64)?;
            h.zero()?;
            let lenp = h.alloc(4)?;
            let kx = h.alloc(values * 4)?;
            let vx = h.alloc(values * 4)?;
            let qg = h.alloc(nq * qs * 4)?;
            let ao = h.alloc(nq * d * 4)?;
            // Re-fill the per-token inputs through the same buffers each append writes.
            let qin = pattern(23, nq * qs);
            h.put_f32(qg, &qin)?;
            h.put_bytes(lenp, &le32(rows as i32))?;
            let mut want_k: Vec<Vec<f32>> = Vec::new();
            let mut want_v: Vec<Vec<f32>> = Vec::new();
            for t in 0..rows {
                h.put_bytes(lenp, &le32(t as i32 + 1))?;
                let ksrc = pattern(101 + t as u64, values);
                let vsrc = pattern(201 + t as u64, values);
                h.put_f32(kx, &ksrc)?;
                h.put_f32(vx, &vsrc)?;
                h.run(
                    "kv_append",
                    &p6.kv_append,
                    [1, 1, 1],
                    0,
                    &[h.addr(kx), h.addr(vx), h.addr(region), h.addr(lenp), i(values), i(row_bytes), i(2 * row_bytes), i(kind_enum_code(kind))],
                )?;
                want_k.push(oracle::kv_unpack(kind, &oracle::kv_pack(kind, &ksrc), values));
                want_v.push(oracle::kv_unpack(kind, &oracle::kv_pack(kind, &vsrc), values));
            }
            let mut worst = 0.0f64;
            let mut worst_h = 0usize;
            for hh in 0..nq {
                let kh = hh * nkv / nq;
                h.run(
                    "attn_decode",
                    &p6.attn_decode,
                    [nq as u32, 1, 1],
                    0,
                    &[
                        h.addr(qg),
                        h.addr(region),
                        h.addr(ao),
                        h.addr(lenp),
                        i(nq),
                        i(nkv),
                        i(d),
                        i(qs),
                        i(values),
                        i(row_bytes),
                        i(2 * row_bytes),
                        i(kind_enum_code(kind)),
                        f(1.0 / (d as f32).sqrt()),
                    ],
                )?;
                let got = h.get_f32(ao + hh * d * 4, d)?;
                let ks: Vec<&[f32]> = want_k.iter().map(|r| &r[kh * d..kh * d + d]).collect();
                let vs: Vec<&[f32]> = want_v.iter().map(|r| &r[kh * d..kh * d + d]).collect();
                let want = oracle::attn_decode(&qin[hh * qs..hh * qs + d], &ks, &vs, 1.0 / (d as f32).sqrt());
                let c = score(&format!("attn_decode[{kind}] head {hh}"), &got, &want, 1e-4);
                if c.nmse > worst {
                    worst = c.nmse;
                    worst_h = hh;
                }
                out.push(c);
            }
            let _ = worst_h;
        }
    }

    // ---- one-token attention: softmax over a single row is exactly that row's V -------------
    {
        let (nq, nkv, d) = (dims.attn_n_q_heads, dims.attn_n_kv_heads, dims.attn_d_head);
        let values = dims.n_kv;
        let row_bytes = crate::cache::KvKind::parse("f16")?.row_bytes(values);
        let region = h.alloc(2 * row_bytes + 64)?;
        let kx = h.alloc(values * 4)?;
        let vx = h.alloc(values * 4)?;
        let qg = h.alloc(nq * 2 * d * 4)?;
        let lenp = h.alloc(4)?;
        let ao = h.alloc(nq * d * 4)?;
        h.put_bytes(lenp, &le32(1i32))?;
        let vs = pattern(97, values);
        h.put_f32(kx, &pattern(89, values))?;
        h.put_f32(vx, &vs)?;
        h.put_f32(qg, &pattern(91, nq * 2 * d))?;
        h.run(
            "kv_append one",
            &p6.kv_append,
            [1, 1, 1],
            0,
            &[h.addr(kx), h.addr(vx), h.addr(region), h.addr(lenp), i(values), i(row_bytes), i(2 * row_bytes), i(0)],
        )?;
        h.run(
            "attn_decode one",
            &p6.attn_decode,
            [nq as u32, 1, 1],
            0,
            &[h.addr(qg), h.addr(region), h.addr(ao), h.addr(lenp), i(nq), i(nkv), i(d), i(2 * d), i(values), i(row_bytes), i(2 * row_bytes), i(0), f(1.0 / (d as f32).sqrt())],
        )?;
        let got = h.get_f32(ao, nq * d)?;
        let want = oracle::kv_unpack("f16", &oracle::kv_pack("f16", &vs), values);
        let mut head_want: Vec<f32> = Vec::new();
        for hh in 0..nq {
            head_want.extend_from_slice(&want[(hh * nkv / nq) * d..(hh * nkv / nq) * d + d]);
        }
        out.push(score("attn_decode single token = that token's V", &got, &head_want, 1e-6));
    }

    // ---- GDN conv ------------------------------------------------------------------------
    {
        let ch = dims.qkv_width;
        let rows = d_conv - 1;
        let qkv = h.alloc(ch * 4)?;
        let cw = h.alloc(ch * d_conv * 4)?;
        let st0 = h.alloc(rows * ch * 4)?;
        let co = h.alloc(ch * 4)?;
        let qv = pattern(31, ch);
        let cwv = pattern(37, ch * d_conv);
        let sv = pattern(41, rows * ch);
        h.put_f32(qkv, &qv)?;
        h.put_f32(cw, &cwv)?;
        h.put_f32(st0, &sv)?;
        let grid = [((ch / PHASE6_BLOCK as usize) as u32).max(1), 1, 1];
        h.run("gdn_conv", &p6.gdn_conv, grid, 0, &[h.addr(qkv), h.addr(cw), h.addr(co), h.addr(st0), i(ch), i(d_conv)])?;
        let (want, next) = oracle::gdn_conv_silu(&qv, &cwv, &sv, d_conv);
        out.push(score("gdn_conv_silu out", &h.get_f32(co, ch)?, &want, 1e-10));
        out.push(score("gdn_conv_silu state slides", &h.get_f32(st0, rows * ch)?, &next, 0.0));
    }

    // ---- GDN l2norm ----------------------------------------------------------------------
    {
        let (nk, dk) = (dims.n_k_heads, dims.d_k);
        let n = 2 * nk * dk;
        let x = h.alloc(n * 4)?;
        let xv = pattern(43, n);
        h.put_f32(x, &xv)?;
        h.run("gdn_l2norm", &p6.gdn_l2norm, [(2 * nk) as u32, 1, 1], 0, &[h.addr(x), h.addr(x), i(dk), f(rms_eps)])?;
        let want = oracle::gdn_l2norm(&xv, dk, rms_eps);
        out.push(score("gdn_l2norm", &h.get_f32(x, n)?, &want, 1e-10));
    }

    // ---- GDN delta rule ------------------------------------------------------------------
    {
        let (nv, dv, dk, nk) = (dims.n_v_heads, dims.d_v, dims.d_k, dims.n_k_heads);
        let conv = h.alloc(dims.qkv_width * 4)?;
        let ba = h.alloc(2 * nv * 4)?;
        let sa = h.alloc(nv * 4)?;
        let dt = h.alloc(nv * 4)?;
        let st = h.alloc(nv * dv * dk * 4)?;
        let core = h.alloc(nv * dv * 4)?;
        // A normalised q/k, as the pipeline hands them over after the L2 norm.
        let mut cv = pattern(47, dims.qkv_width);
        for h_ in 0..(2 * nk) {
            let b = h_ * dk;
            let nn = cv[b..b + dk].iter().map(|v| v * v).sum::<f32>().sqrt();
            for v in cv[b..b + dk].iter_mut() {
                *v /= nn;
            }
        }
        let bv = pattern(53, 2 * nv);
        // `ssm_a` is stored as -exp(A_log): every value is negative.
        let sav: Vec<f32> = (0..nv).map(|x| -((x as f32 + 1.0) * 0.13).exp()).collect();
        let dtv = pattern(59, nv);
        let sv = pattern(61, nv * dv * dk);
        h.put_f32(conv, &cv)?;
        h.put_f32(ba, &bv)?;
        h.put_f32(sa, &sav)?;
        h.put_f32(dt, &dtv)?;
        h.put_f32(st, &sv)?;
        h.run(
            "gdn_decode",
            &p6.gdn_decode,
            [nv as u32, 1, 1],
            0,
            &[h.addr(conv), h.addr(ba), h.addr(sa), h.addr(dt), h.addr(st), h.addr(core), i(nv), i(dv), i(dk), i(nk)],
        )?;
        let (want_st, want_core) = oracle::gdn_decode_step(&cv, &bv, &sav, &dtv, &sv, nv, dv, dk, nk);
        out.push(score("gdn_decode state", &h.get_f32(st, nv * dv * dk)?, &want_st, 1e-10));
        out.push(score("gdn_decode out", &h.get_f32(core, nv * dv)?, &want_core, 1e-10));

        // ---- the gated norm on its output -------------------------------------------------
        let w = h.alloc(dv * 4)?;
        let z = h.alloc(nv * dv * 4)?;
        let wv = pattern(67, dv);
        let zv = pattern(71, nv * dv);
        h.put_f32(w, &wv)?;
        h.put_f32(z, &zv)?;
        h.run("gdn_gate_norm", &p6.gdn_gate_norm, [nv as u32, 1, 1], 0, &[h.addr(core), h.addr(w), h.addr(z), h.addr(core), i(dv), f(rms_eps)])?;
        let got = h.get_f32(core, nv * dv)?;
        let want = oracle::gdn_gate_norm(&want_core, &wv, &zv, dv, rms_eps);
        out.push(score("gdn_gate_norm", &got, &want, 1e-10));
    }

    // ---- snapshot / restore --------------------------------------------------------------
    {
        let (nv, dv, dk) = (dims.n_v_heads, dims.d_v, dims.d_k);
        let execs = 2usize; // two executions of the same shape is enough to catch a stride bug
        let sv = nv * dv * dk;
        let cv = 3 * dims.qkv_width;
        let live_s = h.alloc(execs * sv * 4)?;
        let live_c = h.alloc(execs * cv * 4)?;
        let snap_s = h.alloc(execs * sv * 4)?;
        let snap_c = h.alloc(execs * cv * 4)?;
        let s0 = pattern(73, execs * sv);
        let c0 = pattern(79, execs * cv);
        h.put_f32(live_s, &s0)?;
        h.put_f32(live_c, &c0)?;
        let grid = [64, 1, 1];
        let args = |to_snap: i32| -> Vec<crate::program::Arg> {
            vec![h.addr(live_s), h.addr(live_c), h.addr(snap_s), h.addr(snap_c), i(sv), i(cv), i(execs), i(sv), i(cv), i(to_snap as usize)]
        };
        h.run("snapshot", &p6.snapshot, grid, 0, &args(1))?;
        // Change the live state, then restore: the snapshot must bring back exactly what was there.
        let s1 = pattern(83, execs * sv);
        h.put_f32(live_s, &s1)?;
        h.put_f32(live_c, &pattern(89, execs * cv))?;
        h.run("restore", &p6.snapshot, grid, 0, &args(0))?;
        out.push(score("gdn_snapshot restore state", &h.get_f32(live_s, execs * sv)?, &s0, 0.0));
        out.push(score("gdn_snapshot restore conv", &h.get_f32(live_c, execs * cv)?, &c0, 0.0));
    }

    Ok(out)
}

fn kind_enum_code(kind: &str) -> usize {
    match kind {
        "f16" => 0,
        "q8_0" => 1,
        "q4_0" => 2,
        _ => 0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn phase6_specs_are_unique_and_launchable() {
        let names: Vec<&str> = PHASE6_SPECS.iter().map(|s| s.0).collect();
        assert_eq!(names.len(), names.iter().collect::<std::collections::BTreeSet<_>>().len());
        for (name, _, block) in PHASE6_SPECS {
            assert!(*block > 0 && *block <= 1024, "{name} workgroup {block}");
        }
        // The .hip sizes its shared tiles from these two constants.
        assert_eq!(ATTN_BLOCK % 32, 0);
        assert_eq!(KV_TILE * 256 * 2 * 4, 16384, "two 8x256 f32 tiles");
        let shared = KV_TILE * 256 * 2 * 4 + 512 * 4 + 8 * 256 * 4;
        assert!(shared <= 65536, "{} bytes, over the per-block shared limit", shared);
    }
}
