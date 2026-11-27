//! The Phase 4 acceptance harness: every fused kernel measured against its scalar oracle, the
//! fused route measured against the Phase 2 golden stage dumps, and the fused route measured
//! against the unfused node sequence it replaces.
//!
//! Three things live here because they share the plumbing (upload inputs, launch one program,
//! read back, score with `sn_core::oracle::nmse`):
//!
//! * [`kernel_checks`] - each kernel vs `sn-core/src/oracle.rs` on identical deterministic inputs.
//!   This is design-plan P4 applied per kernel, and it is the only thing that can catch a wrong
//!   offset in the `[q | gate]` stride or the half-offset rope pairing before a forward exists.
//! * [`route_vs_golden`] - the mHC route and merge fed the *reference's own* streams (`mhc.<il>.a`,
//!   `mhc.<il>.b` from `sn ref --dump`) with the checkpoint's real router weights, scored against
//!   `mhc.<il>.logits` / `.post` / `.mix`, `l.<il>.h` / `.h2`, and the post-merge
//!   `mhc.<il>.a` / `.b`. That is Phase 4's gate ("NMSE <= 1e-5 vs Phase 2 stage dumps, <= 1e-6 on
//!   `mix`") in the literal sense: the same files `Tools/golden-check.py` reads.
//! * [`fusion_gate`] - the same route written the way the fork writes it (486 launches) against
//!   the fused kernel (1 launch), in device time, both with and without a graph.
//!
//! Everything here is measurement: nothing outside `sn-hip` calls HIP, and no number in
//! `Docs/bench.md` was typed by hand.

use std::ffi::c_void;

use sn_core::dq::{f16_to_f32, f32_to_f16};
use sn_core::error::{Error, Result};
use sn_core::gguf::Gguf;
use sn_core::num::{Rope, RopePairing};
use sn_core::oracle::{self, RouteParams};
use sn_core::plan::Plan;

use crate::hip::check as hipcheck;
use crate::kernels::{DType, Kernel, KernelRegistry};
use crate::ops::{self, Kernels, BLOCK};
use crate::program::{Arg, Graph, Program, ProgramBuilder, Timer};
use crate::{Slab, Substrate};

/// One measured comparison.
#[derive(Clone, Debug)]
pub struct Check {
    pub name: String,
    /// What the comparison says the output is compared against ("oracle", "golden").
    pub against: &'static str,
    pub n: usize,
    pub nmse: f64,
    pub max_abs: f32,
    /// The gate from design-plan §7 Phase 4.
    pub gate: f64,
}

impl Check {
    pub fn ok(&self) -> bool {
        self.nmse <= self.gate && self.nmse.is_finite()
    }
}

/// Device scratch for a check run: one slab, bump-allocated, offsets stable per call order.
pub struct Harness<'a> {
    sub: &'a mut Substrate,
    k: Kernels,
    base: u64,
    cursor: usize,
    len: usize,
}

/// Enough arena for the widest case (three 12288-element FFN buffers, an 8192-element stream pair,
/// an 8x8192 router, and a two-token rope layout with a 512-wide head stride). Phase 6 adds the
/// GDN state of a whole execution (32 x 128 x 128 f32 = 2 MiB) and a snapshot pair next to it, so
/// the ceiling is sized for that.
pub const ARENA_BYTES: usize = 32 << 20;

impl<'a> Harness<'a> {
    /// Register the Phase 4 kernels, resolve the typed handles, and allocate the scratch slab.
    pub fn open(sub: &'a mut Substrate, reg: &mut KernelRegistry) -> Result<Harness<'a>> {
        ops::register_phase4(&sub.dev, reg)?;
        let k = Kernels::resolve(reg)?;
        sub.alloc_activations(ARENA_BYTES)?;
        let base = sub.activations()?.addr();
        Ok(Harness { sub, k, base, cursor: 0, len: ARENA_BYTES })
    }

    /// A copy, deliberately: holding a `&Kernels` across a call would keep `h` immutably borrowed
    /// and make the next `alloc` (which needs `&mut`) a borrow error.
    pub fn kernels(&self) -> Kernels {
        self.k
    }

    /// Reserve `len` bytes of scratch. Deterministic, so a failure reproduces at the same offset.
    pub fn alloc(&mut self, len: usize) -> Result<usize> {
        let off = self.cursor.div_ceil(256) * 256;
        if off + len > self.len {
            return Err(Error::msg(format!(
                "check arena is {len} bytes short (increase ARENA_BYTES beyond {})",
                self.len
            )));
        }
        self.cursor = off + len;
        Ok(off)
    }

    pub fn addr(&self, off: usize) -> Arg {
        Arg::Ptr(self.base + off as u64)
    }

    fn put(&self, off: usize, bytes: &[u8]) -> Result<()> {
        self.sub.activations()?.write(&self.sub.dev.api, off, bytes)
    }

    pub fn put_f32(&self, off: usize, v: &[f32]) -> Result<()> {
        let bytes: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
        self.put(off, &bytes)
    }

    /// Write real f16 storage: the buffer holds f16 bits, so the kernel sees the same operand
    /// rounding it will see in the engine.
    pub fn put_f16(&self, off: usize, v: &[f32]) -> Result<()> {
        let bytes: Vec<u8> = v.iter().flat_map(|x| f32_to_f16(*x).to_le_bytes()).collect();
        self.put(off, &bytes)
    }

    /// Raw device bytes. The KV cache checks need them: a packed row is only meaningful as bytes
    /// until it goes through `oracle::kv_unpack`.
    pub fn put_bytes(&self, off: usize, bytes: &[u8]) -> Result<()> {
        self.put(off, bytes)
    }

    pub fn get_bytes(&self, off: usize, n: usize) -> Result<Vec<u8>> {
        let mut raw = vec![0u8; n];
        hipcheck(
            "check readback bytes",
            self.sub.dev.api.hipMemcpy.unwrap()(
                raw.as_mut_ptr() as *mut c_void,
                (self.base + off as u64) as *const c_void,
                raw.len(),
                crate::hip::MEMCPY_DEVICE_TO_HOST,
            ),
            &self.sub.dev.api,
        )?;
        Ok(raw)
    }

    pub fn get_f32(&self, off: usize, n: usize) -> Result<Vec<f32>> {
        let mut raw = vec![0u8; n * 4];
        hipcheck(
            "check readback",
            self.sub.dev.api.hipMemcpy.unwrap()(
                raw.as_mut_ptr() as *mut c_void,
                (self.base + off as u64) as *const c_void,
                raw.len(),
                crate::hip::MEMCPY_DEVICE_TO_HOST,
            ),
            &self.sub.dev.api,
        )?;
        Ok(raw
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect())
    }

    pub fn get_f16(&self, off: usize, n: usize) -> Result<Vec<f32>> {
        let mut raw = vec![0u8; n * 2];
        hipcheck(
            "check readback f16",
            self.sub.dev.api.hipMemcpy.unwrap()(
                raw.as_mut_ptr() as *mut c_void,
                (self.base + off as u64) as *const c_void,
                raw.len(),
                crate::hip::MEMCPY_DEVICE_TO_HOST,
            ),
            &self.sub.dev.api,
        )?;
        Ok(raw
            .chunks_exact(2)
            .map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
            .collect())
    }

    /// Launch one kernel through the same `Program` path the engine uses - the checked thing had
    /// better be the launched thing.
    pub fn run(&self, label: &str, k: &Kernel, grid: [u32; 3], shared: u32, args: &[Arg]) -> Result<()> {
        let mut b = ProgramBuilder::new(&self.sub.dev);
        b.kernel(label, k, grid, shared, args)?;
        let p = b.finish();
        p.launch(&self.sub.dev, self.sub.stream)?;
        self.sub.sync()
    }

    /// Zero the whole scratch slab (kernels that accumulate or update in place need a known start).
    pub fn zero(&mut self) -> Result<()> {
        self.sub.activations()?.zero(&self.sub.dev.api)
    }

    pub fn dev(&self) -> &crate::hip::Device {
        &self.sub.dev
    }

    pub fn stream(&self) -> *mut c_void {
        self.sub.stream
    }
}

/// Workgroups that cover  elements at  threads each, as a  grid dimension.
fn u(n: usize) -> u32 {
    ((n as u64).div_ceil(BLOCK as u64)).clamp(1, 65535) as u32
}

fn score(name: &str, against: &'static str, got: &[f32], want: &[f32], gate: f64) -> Check {
    Check { name: name.to_string(), against, n: got.len(), nmse: oracle::nmse(got, want), max_abs: oracle::max_abs(got, want), gate }
}

/// Every Phase 4 kernel against its oracle. Deterministic inputs, so the numbers in
/// `Docs/bench.md` are reproducible on any gfx1102.
pub fn kernel_checks(h: &mut Harness) -> Result<Vec<Check>> {
    let mut out = Vec::new();
    let k = h.kernels();
    let d = 4096usize; // n_embd of the model this engine targets

    // ---- rmsnorm ---------------------------------------------------------------------
    {
        let cols = d;
        let rows = 3usize;
        let x = h.alloc(rows * cols * 4)?;
        let w = h.alloc(cols * 4)?;
        let y = h.alloc(rows * cols * 4)?;
        let xf = oracle::pattern(101, rows * cols);
        let wf: Vec<f32> = (0..cols).map(|i| 1.0 + 0.01 * ((i % 97) as f32) - 0.5 * ((i % 7) as f32 - 3.0) * 0.02).collect();
        h.put_f32(x, &xf)?;
        h.put_f32(w, &wf)?;
        h.run("rmsnorm f32", &k.rmsnorm_f32, [rows as u32, 1, 1], 0, &[h.addr(x), h.addr(w), h.addr(y), Arg::I32(cols as i32), Arg::F32(1e-6), Arg::I32(1)])?;
        let got = h.get_f32(y, rows * cols)?;
        let mut want = Vec::new();
        for r in 0..rows {
            want.extend(oracle::rmsnorm(&xf[r * cols..(r + 1) * cols], Some(&wf), 1e-6));
        }
        out.push(score("rmsnorm_f32 (4096, weighted)", "oracle", &got, &want, 1e-5));

        // the unweighted form the mHC route uses internally
        h.run("rmsnorm f32 unweighted", &k.rmsnorm_f32, [1, 1, 1], 0, &[h.addr(x), Arg::U64(0), h.addr(y), Arg::I32((2 * cols) as i32), Arg::F32(1e-6), Arg::I32(0)])?;
        let got = h.get_f32(y, 2 * cols)?;
        let want = oracle::rmsnorm(&xf[..2 * cols], None, 1e-6);
        out.push(score("rmsnorm_f32 (8192, unweighted)", "oracle", &got, &want, 1e-5));

        // f16 in / f16 out, f32 accumulate: the plan's spec for activations
        let x16 = h.alloc(rows * cols * 2)?;
        let y16 = h.alloc(rows * cols * 2)?;
        h.put_f16(x16, &xf)?;
        h.run("rmsnorm f16", &k.rmsnorm_f16, [rows as u32, 1, 1], 0, &[h.addr(x16), h.addr(w), h.addr(y16), Arg::I32(cols as i32), Arg::F32(1e-6), Arg::I32(1)])?;
        let got = h.get_f16(y16, rows * cols)?;
        let mut want = Vec::new();
        for r in 0..rows {
            want.extend(oracle::rmsnorm_f16(&xf[r * cols..(r + 1) * cols], Some(&wf), 1e-6));
        }
        out.push(score("rmsnorm_f16 (4096, weighted)", "oracle", &got, &want, 1e-5));
    }

    // ---- add -------------------------------------------------------------------------
    {
        let n = 3 * d;
        let a = h.alloc(n * 4)?;
        let b = h.alloc(n * 4)?;
        let av = oracle::pattern(201, n);
        let bv = oracle::pattern(202, n);
        h.put_f32(a, &av)?;
        h.put_f32(b, &bv)?;
        h.run("add f32", &k.add_f32, ops::grid1d(n), 0, &[h.addr(a), h.addr(b), Arg::U64(n as u64)])?;
        let got = h.get_f32(a, n)?;
        out.push(score("add_f32", "oracle", &got, &oracle::add(&av, &bv), 1e-5));

        let a16 = h.alloc(n * 2)?;
        let b16 = h.alloc(n * 2)?;
        h.put_f16(a16, &av)?;
        h.put_f16(b16, &bv)?;
        h.run("add f16", &k.add_f16, ops::grid1d(n), 0, &[h.addr(a16), h.addr(b16), Arg::U64(n as u64)])?;
        let got = h.get_f16(a16, n)?;
        out.push(score("add_f16", "oracle", &got, &oracle::add_f16(&av, &bv), 1e-5));
    }

    // ---- silu_mul / swiglu / sigmoid ---------------------------------------------------
    {
        let n = 12288; // n_ffn
        let g = h.alloc(n * 4)?;
        let u = h.alloc(n * 4)?;
        let o = h.alloc(n * 4)?;
        // A product of two `pattern` draws overflows f16 (the pattern reaches 6e2). Real FFN
        // activations do not, so the fixture is clipped to a range an activation actually occupies;
        // an unclipped fixture measures the format's overflow, not the kernel.
        let gv: Vec<f32> = oracle::pattern(301, n).iter().map(|x| x.clamp(-16.0, 16.0)).collect();
        let uv: Vec<f32> = oracle::pattern(302, n).iter().map(|x| x.clamp(-16.0, 16.0)).collect();
        h.put_f32(g, &gv)?;
        h.put_f32(u, &uv)?;
        h.run("silu_mul f32", &k.silu_mul_f32, ops::grid1d(n), 0, &[h.addr(g), h.addr(u), h.addr(o), Arg::U64(n as u64)])?;
        let got = h.get_f32(o, n)?;
        out.push(score("silu_mul_f32 (12288)", "oracle", &got, &oracle::silu_mul(&gv, &uv), 1e-5));

        // the same data laid out as one [gate | up] buffer must give the same answer
        h.run("swiglu f32", &k.swiglu_f32, ops::grid1d(n), 0, &[h.addr(g), h.addr(o), Arg::U64(n as u64)])?;
        let got2 = h.get_f32(o, n)?;
        out.push(score("swiglu_f32 (12288)", "oracle", &got2, &oracle::swiglu(&[gv.clone(), uv.clone()].concat()), 1e-5));

        let g16 = h.alloc(n * 2)?;
        let u16 = h.alloc(n * 2)?;
        let o16 = h.alloc(n * 2)?;
        h.put_f16(g16, &gv)?;
        h.put_f16(u16, &uv)?;
        h.run("silu_mul f16", &k.silu_mul_f16, ops::grid1d(n), 0, &[h.addr(g16), h.addr(u16), h.addr(o16), Arg::U64(n as u64)])?;
        let got = h.get_f16(o16, n)?;
        out.push(score("silu_mul_f16 (12288)", "oracle", &got, &oracle::silu_mul_f16(&gv, &uv), 1e-5));

        let m = 2048;
        let sx = h.alloc(m * 4)?;
        let sy = h.alloc(m * 4)?;
        let xv = oracle::pattern(303, m);
        h.put_f32(sx, &xv)?;
        h.run("sigmoid f32", &k.sigmoid_f32, ops::grid1d(m), 0, &[h.addr(sx), h.addr(sy), Arg::U64(m as u64)])?;
        let got = h.get_f32(sy, m)?;
        out.push(score("sigmoid_f32 (2048)", "oracle", &got, &oracle::sigmoid_v(&xv), 1e-5));
    }

    // ---- rope_inplace -------------------------------------------------------------------
    // The real geometry: attn_q's output is 16 heads of [q(256) | gate(256)], rope rotates the
    // first 64 dims of the q half in half-offset pairs, and the token stride is 8192.
    {
        let rope = Rope::new(64, 1e7, &[11, 11, 10, 0], 2.0, 262144);
        let pos = 977usize; // anything but 0: at position 0 both pairings are the identity
        let (cos, sin) = rope.table(pos);
        let np = rope.n_pairs();
        for (label, heads, head_stride, token_stride, f16) in [
            ("q", 16usize, 512usize, 8192usize, false),
            ("k", 4usize, 256usize, 1024usize, false),
            ("q f16", 16usize, 512usize, 8192usize, true),
        ] {
            let n = 2 * token_stride; // two tokens' worth of buffer, one table applied to both
            let buf = h.alloc(n * if f16 { 2 } else { 4 })?;
            // The table is indexed `[position][pair]`, exactly as the engine precomputes it for
            // n_ctx rows, so the fixture has to carry the row this test asks for.
            let rows = pos + 1;
            let ctab = h.alloc(rows * np * 4)?;
            let stab = h.alloc(rows * np * 4)?;
            let ptab = h.alloc(8)?;
            let v = oracle::pattern(401 + heads as u64, n);
            if f16 {
                h.put_f16(buf, &v)?;
            } else {
                h.put_f32(buf, &v)?;
            }
            let mut ctabv = vec![0.0f32; rows * np];
            let mut stabv = vec![0.0f32; rows * np];
            ctabv[pos * np..pos * np + np].copy_from_slice(&cos);
            stabv[pos * np..pos * np + np].copy_from_slice(&sin);
            h.put_f32(ctab, &ctabv)?;
            h.put_f32(stab, &stabv)?;
            h.put_f32(ptab, &[0.0f32; 2])?; // the position array is u32; write it as bytes below
            let posbytes = [pos as u32, pos as u32];
            let pb: Vec<u8> = posbytes.iter().flat_map(|x| x.to_le_bytes()).collect();
            h.put(ptab, &pb)?;
            let kn: &Kernel = if f16 { &k.rope_inplace_f16 } else { &k.rope_inplace_f32 };
            h.run(
                &format!("rope {label}"),
                kn,
                [heads as u32, 2, 1],
                0,
                &[
                    h.addr(buf),
                    h.addr(ctab),
                    h.addr(stab),
                    h.addr(ptab),
                    Arg::I32(np as i32),
                    Arg::I32(64),
                    Arg::I32(head_stride as i32),
                    Arg::U64(token_stride as u64),
                    Arg::I32(1), // HalfOffset: what sensenova_loop uses
                ],
            )?;
            let mut want = v.clone();
            oracle::rope_inplace(&mut want, 0, 2, heads, head_stride, token_stride, 64, &cos, &sin, RopePairing::HalfOffset);
            let want = if f16 { oracle::f16(&want) } else { want };
            let got = if f16 { h.get_f16(buf, n)? } else { h.get_f32(buf, n)? };
            out.push(score(&format!("rope_inplace_{label} (stride {head_stride}/{token_stride}, pos {pos})"), "oracle", &got, &want, 1e-5));
        }
    }

    // ---- attn_gate_norm ------------------------------------------------------------------
    {
        let (nh, dh) = (16usize, 256usize);
        let n = nh * dh;
        let o = h.alloc(n * 4)?;
        let qg = h.alloc(2 * n * 4)?; // [q | gate] per head
        let y = h.alloc(n * 4)?;
        let ov = oracle::pattern(501, n);
        let qgv = oracle::pattern(502, 2 * n);
        h.put_f32(o, &ov)?;
        h.put_f32(qg, &qgv)?;
        h.run(
            "attn gate norm",
            &k.attn_gate_norm_f32,
            [ops::grid1d(n)[0], 1, 1],
            0,
            &[
                h.addr(o),
                h.addr(qg),
                h.addr(y),
                Arg::I32(dh as i32),
                Arg::I32(2 * dh as i32),
                Arg::I32(dh as i32),
                Arg::U64(n as u64),
                Arg::U64(n as u64),
                Arg::U64(2 * n as u64),
                Arg::U64(n as u64),
            ],
        )?;
        let got = h.get_f32(y, n)?;
        let want = oracle::attn_gate_norm(&ov, &qgv, nh, dh, 2 * dh, dh);
        out.push(score("attn_gate_norm_f32 (16x256, gate stride 512)", "oracle", &got, &want, 1e-5));
    }

    // ---- mhc_route / mhc_merge --------------------------------------------------------------
    {
        let a = h.alloc(d * 4)?;
        let b = h.alloc(d * 4)?;
        let w = h.alloc(8 * 2 * d * 4)?;
        let bias = h.alloc(8 * 4)?;
        let yin = h.alloc(d * 4)?;
        let lg = h.alloc(8 * 4)?;
        let post = h.alloc(2 * 4)?;
        let mix = h.alloc(4 * 4)?;
        let av = oracle::pattern(601, d);
        let bv = oracle::pattern(602, d);
        // A router worth testing: the logits it produces straddle the softplus threshold and the
        // near-degenerate case where 20 sweeps have not converged (num::tests pins that shape).
        let wv: Vec<f32> = (0..8 * 2 * d).map(|i| 4.0e-4 * ((i % 251) as f32 - 125.0) / 125.0).collect();
        let biasv = [1.2f32, -0.4, 2.5, -2.5, 6.0, -6.0, 1.0, -1.0];
        h.put_f32(a, &av)?;
        h.put_f32(b, &bv)?;
        h.put_f32(w, &wv)?;
        h.put_f32(bias, &biasv)?;
        let p = RouteParams { eps: oracle::MHC_EPS, pre_scale: 1.0, sweeps: 20, fixed_identity: false };
        let want = oracle::mhc_route(&av, &bv, &wv, &biasv, &p);

        let route = |h: &Harness, sweeps: i32| -> Result<()> {
            h.run(
                "mhc route",
                &k.mhc_route_f32,
                [1, 1, 1],
                0, // the kernel declares its scratch static; see `route_static_shared`
                // (passing it as dynamic shared trashed the results - Docs/bench.md)
                &[
                    h.addr(a),
                    h.addr(b),
                    h.addr(w),
                    h.addr(bias),
                    h.addr(yin),
                    h.addr(lg),
                    h.addr(post),
                    h.addr(mix),
                    Arg::I32(d as i32),
                    Arg::I32(8),
                    Arg::U64(d as u64),
                    Arg::U64(d as u64),
                    Arg::U64(d as u64),
                    Arg::U64(8),
                    Arg::U64(2),
                    Arg::U64(4),
                    Arg::F32(1e-6),
                    Arg::F32(1.0),
                    Arg::I32(sweeps),
                    Arg::I32(0),
                ],
            )
        };
        route(h, 20)?;
        out.push(score("mhc_route_f32 logits", "oracle", &h.get_f32(lg, 8)?, &want.logits, 1e-5));
        out.push(score("mhc_route_f32 input", "oracle", &h.get_f32(yin, d)?, &want.input, 1e-5));
        out.push(score("mhc_route_f32 post", "oracle", &h.get_f32(post, 2)?, &want.post, 1e-5));
        out.push(score("mhc_route_f32 mix", "oracle", &h.get_f32(mix, 4)?, &want.mix, 1e-6));

        // The 20-sweep contract, as a sensitivity control rather than an assumption: the same
        // kernel with 6 sweeps must miss `mix` by far more than the gate, the way `sn ref --iters
        // 6` misses the reference's logits (Phase 2's gate).
        route(h, 6)?;
        let six = h.get_f32(mix, 4)?;
        let nmse6 = oracle::nmse(&six, &want.mix);
        out.push(Check {
            name: format!("mhc_route_f32 mix with 6 sweeps (must FAIL the gate: {nmse6:.3e})"),
            against: "oracle",
            n: 4,
            nmse: nmse6,
            max_abs: oracle::max_abs(&six, &want.mix),
            gate: -1.0, // deliberately unsatisfiable: this check documents sensitivity
        });

        // fixed_identity: mix is the identity and logits 4..7 are never read.
        h.run(
            "mhc route fixed identity",
            &k.mhc_route_f32,
            [1, 1, 1],
            0, // static scratch, no dynamic shared (see `route_static_shared`)
            &[
                h.addr(a),
                h.addr(b),
                h.addr(w),
                h.addr(bias),
                h.addr(yin),
                h.addr(lg),
                h.addr(post),
                h.addr(mix),
                Arg::I32(d as i32),
                Arg::I32(8),
                Arg::U64(d as u64),
                Arg::U64(d as u64),
                Arg::U64(d as u64),
                Arg::U64(8),
                Arg::U64(2),
                Arg::U64(4),
                Arg::F32(1e-6),
                Arg::F32(0.5),
                Arg::I32(20),
                Arg::I32(1),
            ],
        )?;
        let pf = RouteParams { eps: oracle::MHC_EPS, pre_scale: 0.5, sweeps: 20, fixed_identity: true };
        let wantf = oracle::mhc_route(&av, &bv, &wv, &biasv, &pf);
        out.push(score("mhc_route_f32 mix (fixed_identity)", "oracle", &h.get_f32(mix, 4)?, &wantf.mix, 1e-9));
        out.push(score("mhc_route_f32 input (pre_scale 0.5)", "oracle", &h.get_f32(yin, d)?, &wantf.input, 1e-5));

        // merge: in-place on both streams, from the pre-merge values
        let br = h.alloc(d * 4)?;
        let brv = oracle::pattern(603, d);
        h.put_f32(br, &brv)?;
        let a2 = h.alloc(d * 4)?;
        let b2 = h.alloc(d * 4)?;
        h.put_f32(a2, &av)?;
        h.put_f32(b2, &bv)?;
        let mixv = [0.3f32, 0.7, 0.6, 0.4];
        let postv = [1.5f32, -0.5];
        h.put_f32(mix, &mixv)?;
        h.put_f32(post, &postv)?;
        h.run(
            "mhc merge",
            &k.mhc_merge_f32,
            [u(d), 1, 1],
            0,
            &[
                h.addr(a2),
                h.addr(b2),
                h.addr(br),
                h.addr(mix),
                h.addr(post),
                Arg::I32(d as i32),
                Arg::U64(d as u64),
                Arg::U64(d as u64),
                Arg::U64(d as u64),
                Arg::U64(4),
                Arg::U64(2),
            ],
        )?;
        let (wa, wb) = oracle::mhc_merge(&av, &bv, &brv, &mixv, &postv);
        out.push(score("mhc_merge_f32 a", "oracle", &h.get_f32(a2, d)?, &wa, 1e-5));
        out.push(score("mhc_merge_f32 b", "oracle", &h.get_f32(b2, d)?, &wb, 1e-5));

        // The two ends of the mHC region (§2.7). `sn_mhc_seed_f32` is the only thing that writes
        // `a`/`b` before the first route reads them, and `sn_mhc_expose_f32` is what the trunk reads
        // back when the next execution is outside the region - the decode program places both, so
        // both are gated like every other kernel here.
        let cur = h.alloc(d * 4)?;
        let curv = oracle::pattern(604, d);
        h.put_f32(cur, &curv)?;
        h.run(
            "mhc seed",
            &k.mhc_seed_f32,
            [u(d), 1, 1],
            0,
            &[
                h.addr(a2),
                h.addr(b2),
                h.addr(cur),
                Arg::I32(d as i32),
                Arg::U64(d as u64),
                Arg::U64(d as u64),
                Arg::U64(d as u64),
            ],
        )?;
        let (sa, sb) = oracle::mhc_seed(&curv);
        out.push(score("mhc_seed_f32 a", "oracle", &h.get_f32(a2, d)?, &sa, 1e-12));
        out.push(score("mhc_seed_f32 b", "oracle", &h.get_f32(b2, d)?, &sb, 1e-12));
        // Different values in the two streams, so an expose that read only one of them is visible.
        h.put_f32(a2, &av)?;
        h.put_f32(b2, &bv)?;
        h.run(
            "mhc expose",
            &k.mhc_expose_f32,
            [u(d), 1, 1],
            0,
            &[
                h.addr(a2),
                h.addr(b2),
                h.addr(cur),
                Arg::I32(d as i32),
                Arg::U64(d as u64),
                Arg::U64(d as u64),
                Arg::U64(d as u64),
            ],
        )?;
        let wex = oracle::mhc_expose(&av, &bv);
        out.push(score("mhc_expose_f32 cur", "oracle", &h.get_f32(cur, d)?, &wex, 1e-12));
    }

    Ok(out)
}

/// One mHC execution replayed against the reference's own stage dumps.
#[derive(Clone, Debug)]
pub struct GoldenRun {
    pub exec: usize,
    pub checks: Vec<Check>,
}

/// Replay the mHC executions in `execs` through `sn_mhc_route_f32` + `sn_mhc_merge_f32` +
/// `sn_rmsnorm_f32`, starting from the reference's streams, and score every output the reference
/// dumped.
///
/// The inputs come from `sn ref --dump` (`mhc.<il>.a` / `mhc.<il>.b`, which the reference writes at
/// the *end* of every mHC execution, so execution `il` starts from execution `il-1`'s output), the
/// weights come from the checkpoint, and the branch comes from `gdn.<il>.branch` /
/// `attn.<il>.branch`. Nothing is synthesised, which is what makes this Phase 4's gate rather than
/// a self-consistency check.
pub fn route_vs_golden(h: &mut Harness, g: &Gguf, plan: &Plan, dump: &str, execs: &[usize]) -> Result<Vec<GoldenRun>> {
    let k = h.kernels();
    let d = plan.hp.n_embd;
    let dm = &plan.hp.dims;

    // The router and norm weights are f32 and tiny; copy just those tensors instead of uploading
    // the 5.8 GB slab, so the gate runs on a card that is also holding something else.
    let mut names: Vec<String> = Vec::new();
    for &il in execs {
        let blk = plan.exec(il).tensor_block;
        for role in [
            "hc_attn_fn.weight",
            "hc_attn_fn.bias",
            "hc_ffn_fn.weight",
            "hc_ffn_fn.bias",
            "attn_norm.weight",
            "post_attention_norm.weight",
        ] {
            names.push(format!("blk.{blk}.{role}"));
        }
    }
    names.sort();
    names.dedup();
    let mut off: Vec<usize> = Vec::new();
    let mut total = 0usize;
    for n in &names {
        let t = g.tensor(n)?;
        total = total.div_ceil(256) * 256;
        off.push(total);
        total += t.size as usize;
    }
    let wslab = Slab::new(&h.sub.dev.api, total.max(16), "route weights")?;
    for (i, n) in names.iter().enumerate() {
        let t = g.tensor(n)?;
        // Each tensor starts on a 256-byte boundary: `sn_mhc_route_f32` reads the router as float4,
        // and the GGUF's own per-tensor alignment is not enough to guarantee that after packing.
        if off[i] % 256 != 0 {
            return Err(Error::msg(format!("{n}: check-slab offset {} is not 256-aligned", off[i])));
        }
        wslab.write(&h.sub.dev.api, off[i], g.tensor_data(t))?;
    }
    let wbase = wslab.addr();
    let waddr = |name: &str| -> Result<Arg> {
        let i = names
            .iter()
            .position(|n| n == name)
            .ok_or_else(|| Error::msg(format!("{name} was not uploaded for the golden route check")))?;
        Ok(Arg::Ptr(wbase + off[i] as u64))
    };

    let stage = |name: String| -> Result<Vec<f32>> {
        oracle::read_stage(&format!("{dump}/{name}.txt")).map_err(|e| {
            Error::msg(format!("{e} - run `sn ref --dump {dump} --dump-only mhc.,l.,l_out,gdn.,ffn.,attn.` first"))
        })
    };

    let mut runs = Vec::new();
    let a = h.alloc(d * 4)?;
    let b = h.alloc(d * 4)?;
    let yin = h.alloc(d * 4)?;
    let lg = h.alloc(8 * 4)?;
    let post = h.alloc(4 * 4)?;
    let mix = h.alloc(4 * 4)?;
    let hbuf = h.alloc(d * 4)?;
    let brbuf = h.alloc(d * 4)?;
    let pa = h.alloc(d * 4)?;
    let pb = h.alloc(d * 4)?;

    for &il in execs {
        let ex = plan.exec(il);
        let blk = ex.tensor_block;
        // Always the previous execution: for the first mHC one that is the execution that produced
        // `l_out`, which is what `a = b = cur` was seeded from.
        let prev = il - 1;
        let mut checks = Vec::new();
        // streams as the reference left them at the end of the previous mHC execution (for the
        // first one, the exposed residual `l_out.<il-1>`, since a = b = cur there)
        let src = if il == plan.hp.mhc_start {
            stage(format!("l_out.{prev}"))?
        } else {
            stage(format!("mhc.{prev}.a"))?
        };
        let srcb = if il == plan.hp.mhc_start {
            src.clone()
        } else {
            stage(format!("mhc.{prev}.b"))?
        };
        h.put_f32(a, &src)?;
        h.put_f32(b, &srcb)?;

        let route = |h: &Harness, which: &str| -> Result<()> {
            let w = waddr(&format!("blk.{blk}.hc_{which}_fn.weight"))?;
            let bias = waddr(&format!("blk.{blk}.hc_{which}_fn.bias"))?;
            h.run(
                &format!("route {which} blk.{blk}"),
                &k.mhc_route_f32,
                [1, 1, 1],
                0, // the kernel declares its scratch static; see `route_static_shared`
                // (passing it as dynamic shared trashed the results - Docs/bench.md)
                &[
                    h.addr(a),
                    h.addr(b),
                    w,
                    bias,
                    h.addr(yin),
                    h.addr(lg),
                    h.addr(post),
                    h.addr(mix),
                    Arg::I32(d as i32),
                    Arg::I32(dm.n_route as i32),
                    Arg::U64(d as u64),
                    Arg::U64(d as u64),
                    Arg::U64(d as u64),
                    Arg::U64(dm.n_route as u64),
                    Arg::U64(2),
                    Arg::U64(4),
                    Arg::F32(oracle::MHC_EPS),
                    Arg::F32(plan.hp.mhc_pre_scale),
                    Arg::I32(plan.hp.mhc_iters as i32),
                    Arg::I32(if plan.hp.mhc_fixed_identity { 1 } else { 0 }),
                ],
            )
        };

        // Diagnostic first, on the host: the *oracle* against the same dumps. If this fails too,
        // the harness is reading the wrong stage files (or the contract changed); if it passes and
        // the kernel fails, the defect is in the kernel with exactly these inputs.
        let host_w = g.tensor_data(g.tensor(&format!("blk.{blk}.hc_attn_fn.weight"))?).chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect::<Vec<f32>>();
        let host_b = g.tensor_data(g.tensor(&format!("blk.{blk}.hc_attn_fn.bias"))?).chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect::<Vec<f32>>();
        let want_route = oracle::mhc_route(
            &src,
            &srcb,
            &host_w,
            &host_b,
            &RouteParams {
                eps: oracle::MHC_EPS,
                pre_scale: plan.hp.mhc_pre_scale,
                sweeps: plan.hp.mhc_iters,
                fixed_identity: plan.hp.mhc_fixed_identity,
            },
        );

        // ---- attention route ---------------------------------------------------------
        route(h, "attn")?;
        let tag = il.to_string();
        checks.push(score(&format!("mhc.{il}.logits (scalar oracle, host)"), "golden", &want_route.logits, &stage(format!("mhc.{tag}.logits"))?, 1e-5));
checks.push(score(&format!("mhc.{il}.logits"), "golden", &h.get_f32(lg, dm.n_route)?, &stage(format!("mhc.{tag}.logits"))?, 1e-5));
        checks.push(score(&format!("mhc.{il}.post"), "golden", &h.get_f32(post, 2)?, &stage(format!("mhc.{tag}.post"))?, 1e-5));
        checks.push(score(&format!("mhc.{il}.mix"), "golden", &h.get_f32(mix, 4)?, &stage(format!("mhc.{tag}.mix"))?, 1e-6));
        // `in` itself is not dumped; its consumer is. rmsnorm(in) * attn_norm must equal l.<il>.h.
        h.run(
            &format!("rmsnorm attn blk.{blk}"),
            &k.rmsnorm_f32,
            [1, 1, 1],
            0,
            &[h.addr(yin), waddr(&format!("blk.{blk}.attn_norm.weight"))?, h.addr(hbuf), Arg::I32(d as i32), Arg::F32(plan.hp.rms_eps), Arg::I32(1)],
        )?;
        checks.push(score(&format!("l.{il}.h (via route input)"), "golden", &h.get_f32(hbuf, d)?, &stage(format!("l.{il}.h"))?, 1e-5));

        // ---- merge with the reference's own branch ------------------------------------
        let branch = stage(format!("{}.{}.branch", if ex.recurrent { "gdn" } else { "attn" }, il))?;
        h.put_f32(brbuf, &branch)?;
        h.put_f32(pa, &src)?;
        h.put_f32(pb, &srcb)?;
        let merge = |h: &Harness| -> Result<()> {
            h.run(
                &format!("merge attn blk.{blk}"),
                &k.mhc_merge_f32,
                [(d as u32).div_ceil(BLOCK), 1, 1],
                0,
                &[
                    h.addr(pa),
                    h.addr(pb),
                    h.addr(brbuf),
                    h.addr(mix),
                    h.addr(post),
                    Arg::I32(d as i32),
                    Arg::U64(d as u64),
                    Arg::U64(d as u64),
                    Arg::U64(d as u64),
                    Arg::U64(4),
                    Arg::U64(2),
                ],
            )
        };
        merge(h)?;

        // ---- ffn route + merge --------------------------------------------------------
        h.put_f32(a, &h.get_f32(pa, d)?)?;
        h.put_f32(b, &h.get_f32(pb, d)?)?;
        route(h, "ffn")?;
        let ftag = format!("{il}.ffn");
        checks.push(score(&format!("mhc.{il}.ffn.logits"), "golden", &h.get_f32(lg, dm.n_route)?, &stage(format!("mhc.{ftag}.logits"))?, 1e-5));
        checks.push(score(&format!("mhc.{il}.ffn.mix"), "golden", &h.get_f32(mix, 4)?, &stage(format!("mhc.{ftag}.mix"))?, 1e-6));
        h.run(
            &format!("rmsnorm post blk.{blk}"),
            &k.rmsnorm_f32,
            [1, 1, 1],
            0,
            &[
                h.addr(yin),
                waddr(&format!("blk.{blk}.post_attention_norm.weight"))?,
                h.addr(hbuf),
                Arg::I32(d as i32),
                Arg::F32(plan.hp.rms_eps),
                Arg::I32(1),
            ],
        )?;
        checks.push(score(&format!("l.{il}.h2 (via ffn route input)"), "golden", &h.get_f32(hbuf, d)?, &stage(format!("l.{il}.h2"))?, 1e-5));
        h.put_f32(brbuf, &stage(format!("ffn.{il}.branch"))?)?;
        merge(h)?;
        // The streams the reference wrote at the end of this execution.
        checks.push(score(&format!("mhc.{il}.a (after both merges)"), "golden", &h.get_f32(pa, d)?, &stage(format!("mhc.{il}.a"))?, 1e-5));
        checks.push(score(&format!("mhc.{il}.b (after both merges)"), "golden", &h.get_f32(pb, d)?, &stage(format!("mhc.{il}.b"))?, 1e-5));

        runs.push(GoldenRun { exec: il, checks });
    }
    Ok(runs)
}

/// The result of the fusion micro-benchmark: the same route, fused and unfused.
#[derive(Clone, Debug)]
pub struct FusionGate {
    pub fused_launches: usize,
    pub unfused_launches: usize,
    pub sweeps: usize,
    pub reps: usize,
    /// Timing trials per number; the minimum is kept (GPU clock states are bimodal).
    pub trials: usize,
    /// ms per route, launched one node at a time.
    pub flat_fused_ms: f64,
    pub flat_unfused_ms: f64,
    /// ms per route, as one captured graph each.
    pub graph_fused_ms: f64,
    pub graph_unfused_ms: f64,
}

impl FusionGate {
    fn ratio(fused: f64, unfused: f64) -> f64 {
        if fused <= 0.0 {
            return 0.0;
        }
        unfused / fused
    }
    /// `(what, ok, detail, gating)`. The last entry is reported but does not gate: launching a
    /// 1024-thread / 32 KB-shared workgroup node-by-node costs ~20 us of submission latency on this
    /// device (measured 25.9 us as a graph vs 46.7 us launched, both minima), so the flat number is
    /// bimodal (85x .. 191x run to run) and is not how the engine runs - Phase 3 replays captured
    /// graphs, which is why the graph leg is the one the plan's >= 100x is tested against.
    pub fn passes(&self) -> Vec<(String, bool, String, bool)> {
        let f = Self::ratio(self.flat_fused_ms, self.flat_unfused_ms);
        let g = Self::ratio(self.graph_fused_ms, self.graph_unfused_ms);
        vec![
            (
                "fused route >= 100x the unfused node sequence (captured graph - how the engine runs)"
                    .into(),
                g >= 100.0,
                format!("{g:.1}x"),
                true,
            ),
            (
                "same sequence launched node-by-node".into(),
                true,
                format!("{f:.1}x"),
                false,
            ),
            (
                "the unfused route really is the fork's 480-node balancing".into(),
                self.unfused_launches >= 480,
                format!("{} launches", self.unfused_launches),
                true,
            ),
        ]
    }
    pub fn report(&self) -> String {
        let f = Self::ratio(self.flat_fused_ms, self.flat_unfused_ms);
        let g = Self::ratio(self.graph_fused_ms, self.graph_unfused_ms);
        let mut s = format!(
            "mhc_route fusion, one route x {} reps x {} trials (min), 4096-wide streams, 8-route router, {} sweeps\n  \
               unfused: {:>5} launches   flat {:>9.4} ms   graph {:>9.4} ms\n  \
               fused:   {:>5} launch     flat {:>9.4} ms   graph {:>9.4} ms\n  \
               speedup: flat {f:.1}x, graph {g:.1}x\n",
            self.reps,
            self.trials,
            self.sweeps,
            self.unfused_launches,
            self.flat_unfused_ms,
            self.graph_unfused_ms,
            self.fused_launches,
            self.flat_fused_ms,
            self.graph_fused_ms,
        );
        for (what, ok, detail, gating) in self.passes() {
            let tag = if !gating { "INFO" } else if ok { "PASS" } else { "FAIL" };
            s += &format!("  [{tag}] {what} {detail}\n");
        }
        s
    }
}

/// Build the unfused route the way `build_route()` in the fork writes it: `sub`, `neg`, `softplus`,
/// `neg`, `softplus`, `neg` per pair update, four pair updates per sweep, `sweeps` sweeps, on a
/// lane buffer - plus the surrounding nodes in their *best* shape (one node for the whole router
/// matmul, one for `in`), so the measurement is not flattered by a strawman.
// One slot per operand of the baseline node graph; they are all arena offsets of the same run.
#[allow(clippy::too_many_arguments)]
fn unfused_route_program(
    h: &Harness,
    a: usize,
    b: usize,
    w: usize,
    bias: usize,
    f: usize,
    lg: usize,
    s: usize,
    post: usize,
    mixlanes: usize,
    scratch: usize,
    yin: usize,
    d: usize,
    sweeps: usize,
) -> Result<Program> {
    let k = h.kernels();
    let mut p = ProgramBuilder::new(h.dev());
    let d2 = 2 * d;
    // rmsnorm over the concatenation, then the router, bias, sigmoids.
    p.kernel("rmsnorm", &k.rmsnorm_f32, [1, 1, 1], 0, &[h.addr(f), Arg::U64(0), h.addr(f), Arg::I32(d2 as i32), Arg::F32(1e-6), Arg::I32(0)])?;
    p.kernel("gemv 8", &k.ub_gemv8, [1, 1, 1], 0, &[h.addr(w), h.addr(f), h.addr(lg), Arg::I32(d2 as i32), Arg::I32(8)])?;
    p.kernel("bias", &k.ub_bias_add, [1, 1, 1], 0, &[h.addr(lg), h.addr(bias), Arg::I32(8)])?;
    p.kernel("sigmoid+post", &k.ub_sigmoid_post, [1, 1, 1], 0, &[h.addr(lg), h.addr(s), h.addr(post)])?;
    // The four log-mixes start in the lane buffer (stride 1 here, 8 in the fork's 8-wide logits).
    for sw in 0..sweeps {
        for (q, (i, j)) in [(0usize, (0usize, 1usize)), (1, (2, 3)), (2, (0, 2)), (3, (1, 3))] {
            let _ = q;
            // 1. sub: d = mix[i] - mix[j]
            p.kernel(
                "sub",
                &k.ub_sub,
                [1, 1, 1],
                0,
                &[h.addr(scratch), h.addr(mixlanes), Arg::U64(i as u64), Arg::U64(j as u64), Arg::U64(1)],
            )?;
            // 2. neg: -d
            p.kernel("neg", &k.ub_neg, [1, 1, 1], 0, &[h.addr_off(scratch + 4), h.addr(scratch), Arg::U64(0)])?;
            // 3. softplus(-d)
            p.kernel("softplus", &k.ub_softplus, [1, 1, 1], 0, &[h.addr_off(scratch + 8), h.addr_off(scratch + 4), Arg::U64(0)])?;
            // 4. neg -> mix[i] = -softplus(-d)
            p.kernel("neg", &k.ub_neg, [1, 1, 1], 0, &[h.addr_off(mixlanes + 4 * i), h.addr_off(scratch + 8), Arg::U64(0)])?;
            // 5. softplus(d)
            p.kernel("softplus", &k.ub_softplus, [1, 1, 1], 0, &[h.addr_off(scratch + 12), h.addr(scratch), Arg::U64(0)])?;
            // 6. neg -> mix[j] = -softplus(d)
            p.kernel("neg", &k.ub_neg, [1, 1, 1], 0, &[h.addr_off(mixlanes + 4 * j), h.addr_off(scratch + 12), Arg::U64(0)])?;
        }
        let _ = sw;
    }
    p.kernel("exp4", &k.ub_exp4, [1, 1, 1], 0, &[h.addr(mixlanes), h.addr(mixlanes)])?;
    p.kernel("in", &k.ub_mix_ab, [((d as u32).div_ceil(BLOCK)), 1, 1], 0, &[h.addr(a), h.addr(b), h.addr(s), h.addr(yin), Arg::I32(d as i32), Arg::F32(1.0)])?;
    Ok(p.finish())
}

impl<'a> Harness<'a> {
    /// `addr` with a byte offset, for the scratch slots the unfused route's temps need.
    pub fn addr_off(&self, off: usize) -> Arg {
        Arg::Ptr(self.base + off as u64)
    }
}

/// The Phase 4 fusion gate: the same route computed by one kernel and by the fork's node sequence,
/// timed in device time both ways, with and without a graph.
pub fn fusion_gate(h: &mut Harness, sweeps: usize, reps: usize, trials: usize) -> Result<FusionGate> {
    let k = h.kernels();
    let d = 4096usize;
    let a = h.alloc(d * 4)?;
    let b = h.alloc(d * 4)?;
    let w = h.alloc(8 * 2 * d * 4)?;
    let bias = h.alloc(8 * 4)?;
    let f = h.alloc(2 * d * 4)?;
    let lg = h.alloc(8 * 4)?;
    let s = h.alloc(4 * 4)?;
    let post = h.alloc(2 * 4)?;
    let mix = h.alloc(4 * 4)?;
    let lanes = h.alloc(8 * 4)?;
    let scratch = h.alloc(16)?;
    let yin = h.alloc(d * 4)?;

    h.put_f32(a, &oracle::pattern(701, d))?;
    h.put_f32(b, &oracle::pattern(702, d))?;
    let wv: Vec<f32> = (0..8 * 2 * d).map(|i| 4.0e-4 * ((i % 251) as f32 - 125.0) / 125.0).collect();
    h.put_f32(w, &wv)?;
    h.put_f32(bias, &[0.5f32, -0.5, 1.0, -1.0, 0.7, -0.7, 0.2, -0.2])?;
    h.put_f32(lanes, &[0.7f32, -0.3, 0.4, -0.9, 0.0, 0.0, 0.0, 0.0])?;

    let fused = {
        let mut p = ProgramBuilder::new(h.dev());
        p.kernel(
            "sn_mhc_route_f32",
            &k.mhc_route_f32,
            [1, 1, 1],
            0, // static scratch, no dynamic shared (see `route_static_shared`)
            &[
                h.addr(a),
                h.addr(b),
                h.addr(w),
                h.addr(bias),
                h.addr(yin),
                h.addr(lg),
                h.addr(post),
                h.addr(mix),
                Arg::I32(d as i32),
                Arg::I32(8),
                Arg::U64(d as u64),
                Arg::U64(d as u64),
                Arg::U64(d as u64),
                Arg::U64(8),
                Arg::U64(2),
                Arg::U64(4),
                Arg::F32(1e-6),
                Arg::F32(1.0),
                Arg::I32(sweeps as i32),
                Arg::I32(0),
            ],
        )?;
        p.finish()
    };
    let unfused = unfused_route_program(h, a, b, w, bias, f, lg, s, post, lanes, scratch, yin, d, sweeps)?;

    // The GPU clocks down between launches, and one timing run of this kernel lands in either clock
    // state (the flat ratio measured 92x and 191x on back-to-back runs). The minimum over `trials`
    // is the only stable statistic: a slower trial can only mean the clocks dropped, never that the
    // kernel got faster.
    let trials = trials.max(1);
    let flat_once = |p: &Program| -> Result<f64> {
        // warm the clocks, then time `reps` whole routes in device time
        p.launch(h.dev(), h.stream())?;
        h.sub.sync()?;
        let t = Timer::new(h.dev())?;
        t.begin(h.dev(), h.stream())?;
        for _ in 0..reps {
            p.launch(h.dev(), h.stream())?;
        }
        let ms = t.end_ms(h.dev(), h.stream())? / reps as f64;
        Ok(ms)
    };
    let flat = |p: &Program| -> Result<f64> {
        let mut best = f64::INFINITY;
        for _ in 0..trials {
            best = best.min(flat_once(p)?);
        }
        Ok(best)
    };
    let flat_fused_ms = flat(&fused)?;
    let flat_unfused_ms = flat(&unfused)?;

    let graph_once = |p: &Program| -> Result<f64> {
        let cap = h.sub.new_stream()?;
        let g = match Graph::capture(h.dev(), cap, p) {
            Ok(g) => g,
            Err(e) => {
                h.sub.destroy_stream(cap);
                return Err(e);
            }
        };
        h.sub.destroy_stream(cap);
        g.replay(h.dev(), h.stream())?;
        h.sub.sync()?;
        let t = Timer::new(h.dev())?;
        t.begin(h.dev(), h.stream())?;
        for _ in 0..reps {
            g.replay(h.dev(), h.stream())?;
        }
        let ms = t.end_ms(h.dev(), h.stream())? / reps as f64;
        Ok(ms)
    };
    let graph_ms = |p: &Program| -> Result<f64> {
        let mut best = f64::INFINITY;
        for _ in 0..trials {
            best = best.min(graph_once(p)?);
        }
        Ok(best)
    };
    let graph_fused_ms = graph_ms(&fused)?;
    let graph_unfused_ms = graph_ms(&unfused)?;

    Ok(FusionGate {
        fused_launches: fused.len(),
        unfused_launches: unfused.len(),
        sweeps,
        reps,
        trials,
        flat_fused_ms,
        flat_unfused_ms,
        graph_fused_ms,
        graph_unfused_ms,
    })
}

/// A text report over a list of checks, with the plan's gate as the header.
pub fn report(checks: &[Check]) -> String {
    let mut s = format!(
        "{:<58} {:>9} {:>10} {:>11} {:>9} {}   (NMSE vs {gate})",
        "",
        "n",
        "NMSE",
        "max|d|",
        "gate",
        "",
        gate = "gate"
    );
    s.push('\n');
    for c in checks {
        s += &format!(
            "{:<58} {:>9} {:>10.3e} {:>11.3e} {:>9.0e} {}\n",
            c.name,
            c.n,
            c.nmse,
            c.max_abs,
            c.gate,
            if c.gate < 0.0 {
                "SENSITIVITY".to_string()
            } else if c.ok() {
                "PASS".to_string()
            } else {
                "FAIL".to_string()
            }
        );
    }
    let failed: Vec<&Check> = checks.iter().filter(|c| c.gate >= 0.0 && !c.ok()).collect();
    s += &format!("{} checks, {} failing\n", checks.len(), failed.len());
    s
}

/// The number of Phase 4 kernels the registry should hold, for `sn hip info`.
pub fn dtype_name(d: DType) -> &'static str {
    match d {
        DType::F32 => "f32",
        DType::F16 => "f16",
        DType::Bf16 => "bf16",
        DType::Q4K => "q4_k",
        DType::Q6K => "q6_k",
        DType::Any => "any",
    }
}
