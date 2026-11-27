//! Phase 5: the quantised matmul surface - the shapes this model actually multiplies, which
//! kernel serves which `(T, n, k)`, and the two gates design-plan §7 asks for.
//!
//! Three things live here because they are one decision expressed three ways:
//!
//! * [`model_shapes`] enumerates the **GEMM shapes in the checkpoint** from the GGUF itself - the
//!   plan's "12 GEMM shapes" are the 12 2-D weight roles, and their element type (Q4_K, Q6_K, or
//!   f32) is read from the file rather than assumed, because §1.2's whole point is that the mix is
//!   per-role and per-block (there is no Q2_K in this file; `sn bench gemm` prints what is there).
//! * [`pick_warps`] is the auto-pick the plan asks for. A decode GEMV is a pure DRAM-streaming
//!   problem, and the one tunable that matters is how many wavefronts read one output row: a wide
//!   matrix (`ffn_down`, 4096 rows) has more independent row streams than the machine can use at
//!   one wavefront per row, while a narrow one (`ssm_beta_alpha`, 64 rows) has *less* parallelism
//!   than the card has issue slots and has to split its K axis to fill them. `sn bench gemm`
//!   sweeps every legal value per shape so the policy is validated against the machine, not
//!   asserted, and [`sweep_warps`] reports the best one it measured.
//! * [`gemm_checks`] is design-plan P4 applied per kernel: the device kernel and `sn_core::dq`
//!   (the reference dequantiser, itself cross-checked against GGML's C by `Tools/dequant_ref.py`
//!   in §2.9) are run over **the checkpoint's own bytes** and scored with `oracle::nmse`.
//!
//! The GEMM path is the prefill half. It is *not* the `mma.f32.f16` path the plan asks for: the
//! RDNA3 WMMA builtin measures at 34.4 TFLOP/s on this card (see `Docs/bench.md` §3.5, which also
//! records what the fragment layout turned out to be), but the tiled kernel here accumulates with
//! plain f32 FMAs, so its throughput is quoted against a **measured f32-FMA peak** as well as
//! against the measured WMMA peak, and the Phase 5 gate that needs WMMA is reported as not met.

use std::ffi::c_void;

use sn_core::dq::Mat;
use sn_core::error::{Error, Result};
use sn_core::gguf::Gguf;
use sn_core::oracle;
use sn_core::plan::Plan;
use sn_core::qtypes;

use crate::check::Check;
use crate::hip::check as hipcheck;
use crate::kernels::{DType, Key, Kernel, KernelRegistry};
use crate::ops::GEMV_BLOCK;
use crate::program::{Arg, Program, ProgramBuilder, Timer};
use crate::Slab;

/// The CU count `pick_warps` is calibrated against. `Docs/baseline.md` §1 records that `rocminfo`
/// reports **40** CUs on this card while `hipDeviceGetAttribute(MULTIPROCESSOR_COUNT)` reports 16;
/// the picker wants the hardware number, and `sn bench gemm`'s wpr sweep is what shows whether the
/// calibration is right (a pick that is not the sweep's best is reported, not hidden).
pub const PICK_CUS: usize = 40;

/// The 2-D weight roles this engine multiplies against an activation. `token_embd` is included as
/// the twelfth: decode *gathers* a row of it (`row_gather_*`, Phase 6) rather than multiplying,
/// but it has the same shape as `output` and the same cost when it is used, and the plan counts it
/// in the §1.2 table. `nextn.eh_proj` is a thirteenth shape and belongs to Phase 9 (MTP).
pub const MATMUL_ROLES: [&str; 12] = [
    "token_embd.weight",
    "output.weight",
    "blk.N.attn_qkv_gate.weight",
    "blk.N.attn_q.weight",
    "blk.N.attn_k.weight",
    "blk.N.attn_v.weight",
    "blk.N.attn_output.weight",
    "blk.N.ssm_beta_alpha.weight",
    "blk.N.ssm_out.weight",
    "blk.N.ffn_gate.weight",
    "blk.N.ffn_up.weight",
    "blk.N.ffn_down.weight",
];

/// One shape, and a real tensor that has it.
#[derive(Clone, Debug)]
pub struct Shape {
    /// The role, normalised the way `budget::tensor_class` does.
    pub role: &'static str,
    /// A tensor in the checkpoint with exactly this shape and type - the one the check and the
    /// bench read, so the numbers are the model's, not a fixture's.
    pub tensor: String,
    pub n: usize,
    pub k: usize,
    /// As stored in the file (design-plan P3: quantised weights, f32 accumulation).
    pub dtype: DType,
    pub type_name: String,
    pub bytes: usize,
    /// True for the second quantisation class of the same role (e.g. `ffn_down` is Q4_K on 18
    /// blocks and Q6_K on 15): both are gated, but the primary row is the role's entry.
    pub variant: bool,
    /// How many times this tensor is streamed per decoded token, from the plan.
    pub visits: usize,
}

impl Shape {
    /// Weight bytes this shape contributes to one decoded token.
    pub fn bytes_per_token(&self) -> usize {
        self.bytes * self.visits
    }
    pub fn key(&self) -> String {
        format!("{}x{}", self.n, self.k)
    }
}

/// Roles in `MATMUL_ROLES` are normalised with a `blk.N.` prefix for per-block tensors, so compare
/// by suffix: `role_is(r, "ffn_down.weight")` matches `blk.N.ffn_down.weight`.
fn role_is(role: &str, want: &str) -> bool {
    role == want || role.ends_with(&format!("N.{want}"))
}

fn role_of(name: &str) -> Option<&'static str> {
    let (_, class) = sn_core::budget::tensor_class(name);
    MATMUL_ROLES.iter().find(|r| *r == &class).copied()
}

fn dtype_of(type_name: &str) -> Option<DType> {
    match type_name {
        "f32" => Some(DType::F32),
        "q4_k" => Some(DType::Q4K),
        "q6_k" => Some(DType::Q6K),
        _ => None,
    }
}

/// Every GEMM shape in `g`, one primary entry per role plus a `variant` entry per additional
/// element type the same role uses. Deterministic: tensors are visited in file order and the first
/// tensor of a (role, type) pair is the representative, so two runs pick the same bytes.
pub fn model_shapes(g: &Gguf, plan: &Plan) -> Result<Vec<Shape>> {
    let mut out: Vec<Shape> = Vec::new();
    for t in g.tensors() {
        let role = match role_of(&t.name) {
            Some(r) => r,
            None => continue,
        };
        let type_name = qtypes::name(t.ty);
        let dtype = match dtype_of(&type_name) {
            Some(d) => d,
            None => {
                return Err(Error::msg(format!(
                    "{}: element type {type_name} is not one Phase 5 has a kernel for",
                    t.name
                )))
            }
        };
        if t.n_dims != 2 {
            return Err(Error::msg(format!("{}: a matmul role with {} dimensions", t.name, t.n_dims)));
        }
        let n = t.ne[1] as usize;
        let k = t.ne[0] as usize;
        if out.iter().any(|s| s.role == role && s.dtype == dtype) {
            continue;
        }
        // A shape only belongs to the trunk if some execution reads it; the head and the embedding
        // are read once per token, `visits` 1. The plan is the only source (§8).
        let visits = visit_count(plan, &t.name);
        out.push(Shape {
            role,
            tensor: t.name.clone(),
            n,
            k,
            dtype,
            type_name,
            bytes: t.size as usize,
            variant: false,
            visits,
        });
        // A second element type for the same role is a second entry (same shape, different bytes).
    }
    // Variants: an entry for every (role, type) the file contains, so a Q4_K/Q6_K role is gated in
    // both paths and neither kernel is validated by the other.
    let primary: Vec<(String, usize)> = out.iter().map(|s| (s.role.to_string(), s.bytes)).collect();
    for t in g.tensors() {
        let role = match role_of(&t.name) {
            Some(r) => r,
            None => continue,
        };
        let type_name = qtypes::name(t.ty);
        let dtype = match dtype_of(&type_name) {
            Some(d) => d,
            None => continue,
        };
        if out.iter().any(|s| s.role == role && s.dtype == dtype) {
            continue;
        }
        let n = t.ne[1] as usize;
        let k = t.ne[0] as usize;
        let visits = visit_count(plan, &t.name);
        let _ = &primary;
        out.push(Shape {
            role,
            tensor: t.name.clone(),
            n,
            k,
            dtype,
            type_name,
            bytes: t.size as usize,
            variant: true,
            visits,
        });
    }
    if out.len() < MATMUL_ROLES.len() {
        return Err(Error::msg(format!(
            "expected at least {} GEMM roles in the model, found {}",
            MATMUL_ROLES.len(),
            out.len()
        )));
    }
    Ok(out)
}

/// How often the trunk streams `name` per decoded token (the head and the embedding are once).
fn visit_count(plan: &Plan, name: &str) -> usize {
    let (block, _) = sn_core::budget::tensor_class(name);
    match block {
        Some(b) if b < plan.n_layer => plan.exec_count_of_gguf_block(b),
        _ => 1,
    }
}

/// The largest wavefronts-per-row the GEMV geometry can use: `SN_GEMV_BLOCK / 32`.
pub const MAX_WARPS_PER_ROW: usize = (GEMV_BLOCK as usize) / 32;

/// `warps_per_row` for one `(n, k)`.
///
/// A workgroup has `GEMV_BLOCK` threads, so `warps_per_row = wpr` means `8 / wpr` rows per
/// workgroup and `n * wpr` wavefronts of work overall. The card wants wavefronts: with `cus`
/// compute units at 4 SIMDs each, saturating issue takes on the order of `16 * cus` live
/// wavefronts (4 per SIMD), so:
///
/// * wide `n` (`ffn_down` 4096, `output` 248320) already has more than enough at `wpr = 1`, and
///   splitting K would only add a shared-memory reduce;
/// * narrow `n` (`ssm_beta_alpha` 64 rows = 64 wavefronts, `attn_k` 1024) does not, and its K
///   axis is the parallelism that is left.
///
/// `wpr` is also capped so no wavefront gets less than 4 super-blocks of work (the per-row reduce
/// costs a barrier; below that the split costs more than it buys).
pub fn pick_warps(n: usize, k: usize, cus: usize) -> usize {
    let want = (16 * cus).max(1);
    let mut wpr = 1usize;
    while n * wpr < want && wpr < MAX_WARPS_PER_ROW {
        wpr *= 2;
    }
    let blocks = k / 256; // K-quant super-blocks; f32 shapes take the same split by elements
    let cap = (blocks / 4).max(1).next_power_of_two().min(MAX_WARPS_PER_ROW);
    wpr.min(cap).max(1)
}

/// Workgroups for one GEMV launch at `warps_per_row`.
pub fn gemv_grid(n: usize, warps_per_row: usize) -> [u32; 3] {
    let rows_per_wg = (GEMV_BLOCK as usize) / (32 * warps_per_row);
    [((n as u64).div_ceil(rows_per_wg as u64)).clamp(1, u32::MAX as u64) as u32, 1, 1]
}

/// The GEMV/GEMM switch. `T == 1` is decode: no reuse, so the only possible kernel is the
/// streaming GEMV. `T >= 8` is prefill, where one weight byte is worth `T` multiply-adds and the
/// tiled kernel wins. Between them the tiled kernel's tiles are mostly empty, so the GEMV is
/// launched once per token - correct, and a known-cheaper-than-GEMM choice that Phase 8's chunker
/// never lands on because it chunks at >= 64.
pub fn pick_path(t: usize, n: usize, k: usize, cus: usize) -> Path {
    match t {
        0 => Path::Invalid,
        1 => Path::Gemv { warps_per_row: pick_warps(n, k, cus) },
        _ if t >= 8 => Path::Gemm,
        _ => Path::GemvStrided { tokens: t, warps_per_row: pick_warps(n, k, cus) },
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Path {
    Invalid,
    Gemv { warps_per_row: usize },
    /// The GEMV applied to each token column in turn (`y[t*n + row]` is not written by it).
    GemvStrided { tokens: usize, warps_per_row: usize },
    Gemm,
}

impl std::fmt::Display for Path {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Path::Invalid => write!(f, "invalid"),
            Path::Gemv { warps_per_row } => write!(f, "gemv(wpr={warps_per_row})"),
            Path::GemvStrided { tokens, warps_per_row } => {
                write!(f, "gemv x {tokens}(wpr={warps_per_row})")
            }
            Path::Gemm => write!(f, "gemm(64x64x32)"),
        }
    }
}

/// The Phase 5 kernel handles, resolved from the loaded images.
///
/// These are registered under **`DType::Any`** with the type in the *name* (`gemv_q4_k`, ...):
/// `decode::ProgramSpec::build` resolves an op by `(name, [], F32|Any|F16)`, and the decode
/// program's gemv ops already carry the type in the kernel name it derives from the tensor
/// (design-plan §7 Phase 4's "named pending kernels ... their argument lists are the ABI those
/// phases inherit"). A shape- or dtype-specialised registration is still available through
/// `KernelRegistry`'s `Key` for anyone who wants one later; nothing here needs it, because the
/// geometry is an argument, not a specialisation.
#[derive(Clone, Copy, Debug)]
pub struct Phase5 {
    pub gemv_q4_k: Kernel,
    pub gemv_q6_k: Kernel,
    pub gemv_f32: Kernel,
    pub gemm_q4_k: Kernel,
    pub gemm_q6_k: Kernel,
    pub gemm_f32: Kernel,
    pub read_bw: Kernel,
    pub peak_fma: Kernel,
    pub peak_wmma: Kernel,
}

pub const PHASE5_SPECS: &[(&str, &str)] = &[
    ("gemv_q4_k", "sn_gemv_q4_k"),
    ("gemv_q6_k", "sn_gemv_q6_k"),
    ("gemv_f32", "sn_gemv_f32"),
    ("gemm_q4_k", "sn_gemm_q4_k"),
    ("gemm_q6_k", "sn_gemm_q6_k"),
    ("gemm_f32", "sn_gemm_f32"),
    ("read_bw", "sn_read_bw"),
    ("peak_fma_f32", "sn_peak_fma_f32"),
    ("peak_wmma", "sn_peak_wmma"),
    ("dbg_q4k_lane", "sn_dbg_q4k_lane"),
    ("dbg_args", "sn_dbg_args"),
];

/// Register every Phase 5 kernel under the name the decode program and the bench use.
pub fn register_phase5(dev: &crate::hip::Device, reg: &mut KernelRegistry) -> Result<usize> {
    let mut n = 0;
    for (name, symbol) in PHASE5_SPECS {
        let block = if *name == "dbg_q4k_lane" { [32, 1, 1] } else { [GEMV_BLOCK, 1, 1] };
        reg.register(
            dev,
            &Key { name: (*name).to_string(), shape: vec![], dtype: DType::Any },
            symbol,
            block,
        )?;
        n += 1;
    }
    Ok(n)
}

impl Phase5 {
    pub fn resolve(reg: &KernelRegistry) -> Result<Phase5> {
        let g = |name: &str| -> Result<Kernel> {
            let k = *reg
                .find(name, &[], DType::Any)
                .ok_or_else(|| Error::msg(format!("{name} is not registered (run register_phase5)")))?;
            if k.block[0] != GEMV_BLOCK {
                return Err(Error::msg(format!(
                    "{name}: workgroup {} does not match SN_GEMV_BLOCK {GEMV_BLOCK}",
                    k.block[0]
                )));
            }
            Ok(k)
        };
        Ok(Phase5 {
            gemv_q4_k: g("gemv_q4_k")?,
            gemv_q6_k: g("gemv_q6_k")?,
            gemv_f32: g("gemv_f32")?,
            gemm_q4_k: g("gemm_q4_k")?,
            gemm_q6_k: g("gemm_q6_k")?,
            gemm_f32: g("gemm_f32")?,
            read_bw: g("read_bw")?,
            peak_fma: g("peak_fma_f32")?,
            peak_wmma: g("peak_wmma")?,
        })
    }

    pub fn gemv_for(&self, d: DType) -> Result<&Kernel> {
        match d {
            DType::Q4K => Ok(&self.gemv_q4_k),
            DType::Q6K => Ok(&self.gemv_q6_k),
            DType::F32 => Ok(&self.gemv_f32),
            other => Err(Error::msg(format!("no GEMV kernel for {other} (this checkpoint stores Q4_K, Q6_K and f32)"))),
        }
    }
    pub fn gemm_for(&self, d: DType) -> Result<&Kernel> {
        match d {
            DType::Q4K => Ok(&self.gemm_q4_k),
            DType::Q6K => Ok(&self.gemm_q6_k),
            DType::F32 => Ok(&self.gemm_f32),
            other => Err(Error::msg(format!("no GEMM kernel for {other} (this checkpoint stores Q4_K, Q6_K and f32)"))),
        }
    }
}

/// One GEMV launch, with the decode ABI (`norm_w = 0` means "the norm is another node").
#[allow(clippy::too_many_arguments)]
pub fn gemv_args(
    w: u64,
    x: u64,
    y: u64,
    k: usize,
    n: usize,
    norm_w: u64,
    eps: f32,
    warps_per_row: usize,
) -> Vec<Arg> {
    vec![
        Arg::Ptr(w),
        Arg::Ptr(x),
        Arg::Ptr(y),
        Arg::I32(k as i32),
        Arg::I32(n as i32),
        if norm_w == 0 { Arg::U64(0) } else { Arg::Ptr(norm_w) },
        Arg::F32(eps),
        Arg::I32(warps_per_row as i32),
    ]
}

/// (The argument count is the kernel ABI plus the device; see `decode::Builder::gemv`.)
#[allow(clippy::too_many_arguments)]
pub fn gemv_program(k: &Kernel, w: u64, x: u64, y: u64, shape: &Shape, norm_w: u64, eps: f32, warps_per_row: usize, dev: &crate::hip::Device) -> Result<Program> {
    let mut b = ProgramBuilder::new(dev);
    b.kernel(
        &format!("gemv {} {}", shape.role, shape.key()),
        k,
        gemv_grid(shape.n, warps_per_row),
        0,
        &gemv_args(w, x, y, shape.k, shape.n, norm_w, eps, warps_per_row),
    )?;
    Ok(b.finish())
}

/// (The argument count is the kernel ABI plus the device; see `decode::Builder::gemv`.)
#[allow(clippy::too_many_arguments)]
pub fn gemm_program(k: &Kernel, w: u64, x: u64, y: u64, n: usize, kdim: usize, t: usize, dev: &crate::hip::Device) -> Result<Program> {
    let mut b = ProgramBuilder::new(dev);
    b.kernel(
        &format!("gemm {n}x{kdim}x{t}"),
        k,
        [(n.div_ceil(64)) as u32, (t.div_ceil(64)) as u32, 1],
        0,
        &[Arg::Ptr(w), Arg::Ptr(x), Arg::Ptr(y), Arg::I32(kdim as i32), Arg::I32(n as i32), Arg::I32(t as i32)],
    )?;
    Ok(b.finish())
}

// ---------------------------------------------------------------------------------------------
// The measured references every Phase 5 number is a fraction of.
// ---------------------------------------------------------------------------------------------

#[derive(Clone, Debug, Default)]
pub struct Machine {
    /// GiB read/s by `sn_read_bw` over a large buffer, at several sizes.
    pub read_gb_s: f64,
    pub read_bytes: usize,
    /// FLOP/s of the pure-FMA probe.
    pub fma_flop_s: f64,
    /// FLOP/s of the WMMA probe; 0 when the ISA has no `wmma-256b` instructions.
    pub wmma_flop_s: f64,
    pub wmma_available: bool,
    /// ns of device time per 16x16x16 matrix op per wavefront, as the probe actually ran. A real
    /// matrix op costs many cycles, so if this comes out below one cycle the probe is not measuring
    /// a matrix op and `wmma_flop_s` must not be quoted as a machine peak (§3.5).
    pub wmma_ns_per_op: f64,
}

/// Time `program` for `reps` launches, taking the minimum of `trials` runs (GPU clock states are
/// bimodal on this card - see `Docs/bench.md` §2.3).
pub fn time_program(dev: &crate::hip::Device, stream: *mut c_void, p: &Program, reps: usize, trials: usize) -> Result<f64> {
    let mut best = f64::INFINITY;
    for _ in 0..trials.max(1) {
        p.launch(dev, stream)?;
        crate::hip::check("sync", dev.api.hipStreamSynchronize.unwrap()(stream), &dev.api)?;
        let t = Timer::new(dev)?;
        t.begin(dev, stream)?;
        for _ in 0..reps {
            p.launch(dev, stream)?;
        }
        let ms = t.end_ms(dev, stream)? / reps as f64;
        best = best.min(ms);
    }
    Ok(best)
}

/// Measure read bandwidth, register FMA throughput and WMMA throughput on this device.
pub fn measure_machine(
    p5: &Phase5,
    dev: &crate::hip::Device,
    stream: *mut c_void,
    scratch: &Slab,
    read_bytes: usize,
) -> Result<Machine> {
    let mut m = Machine::default();
    // Never read past the slab: the first run of this asked for 256 MB over a 192 MB scratch and
    // the GPU answered with an access fault, not an error.
    let read_bytes = read_bytes.min(scratch.len() / 16 * 16);
    let n4 = read_bytes / 16;
    let mut b = ProgramBuilder::new(dev);
    b.kernel(
        "read_bw",
        &p5.read_bw,
        [2048, 1, 1],
        0,
        &[Arg::Ptr(scratch.addr()), Arg::U64(n4 as u64), Arg::Ptr(scratch.addr() + (read_bytes as u64) - 16)],
    )?;
    let p = b.finish();
    let ms = time_program(dev, stream, &p, 5, 3)?;
    m.read_bytes = read_bytes;
    m.read_gb_s = if ms > 0.0 { read_bytes as f64 / (ms / 1e3) / 1e9 } else { 0.0 };

    // f32 FMA peak: 8 FMAs per thread per iteration, 2 FLOP each.
    let (grid, block, iters) = (160u32, 256u32, 200_000u32);
    let mut b = ProgramBuilder::new(dev);
    b.kernel("peak_fma", &p5.peak_fma, [grid, 1, 1], 0, &[Arg::Ptr(scratch.addr()), Arg::I32(iters as i32)])?;
    let p = b.finish();
    let ms = time_program(dev, stream, &p, 3, 2)?;
    let flops = grid as f64 * block as f64 * iters as f64 * 8.0 * 2.0;
    m.fma_flop_s = if ms > 0.0 { flops / (ms / 1e3) } else { 0.0 };

    // WMMA peak: 16x16x16 f16 -> f32, 4096 MAC per wavefront.
    let mut b = ProgramBuilder::new(dev);
    b.kernel("peak_wmma", &p5.peak_wmma, [160, 1, 1], 0, &[Arg::Ptr(scratch.addr()), Arg::I32(20_000)])?;
    let p = b.finish();
    let ms = time_program(dev, stream, &p, 3, 2)?;
    let wavefronts = 160.0 * (GEMV_BLOCK as f64 / 32.0);
    let ops = wavefronts * 20_000.0;
    let flops = ops * 4096.0 * 2.0;
    m.wmma_flop_s = if ms > 0.0 { flops / (ms / 1e3) } else { 0.0 };
    m.wmma_ns_per_op = if ms > 0.0 { ms * 1e6 / ops } else { 0.0 };
    // The probe kernel is a no-op on an ISA without the instruction (`#if defined(__gfx11__)` in
    // the .hip), so a plausible-but-fake number is not silently reported as a peak.
    m.wmma_available = m.wmma_flop_s > 0.0;
    Ok(m)
}

// ---------------------------------------------------------------------------------------------
// Correctness: device kernel vs `sn_core::dq` on the checkpoint's own weights.
// ---------------------------------------------------------------------------------------------

/// The activation every check multiplies against. Deterministic, and scaled to the magnitude a
/// post-norm activation actually has (an rms-normalised vector is O(1) per element), so the
/// measured NMSE is the kernel's error and not an artifact of an oversized fixture.
pub fn activation(seed: u64, k: usize) -> Vec<f32> {
    oracle::pattern(seed, k).iter().map(|x| (x * 0.05).clamp(-4.0, 4.0)).collect()
}

/// Pack the representative tensors into one device slab, 256-byte aligned: the GEMV kernels read
/// `uint`/`float4` out of the weight bytes, and the GGUF's own per-tensor alignment does not
/// survive packing (`check::route_vs_golden` applies the same rule for the same reason).
///
/// `scratch` is a second slab the checks and the bench bump-allocate activations and outputs from:
/// a `T = 512` prefill GEMM needs 25 MB of activations and 25 MB of output, which is far past the
/// 8 MiB the Phase 4 elementwise checks use, and reading a large slab is also how the device
/// bandwidth reference is measured.
pub struct WeightSlab {
    pub slab: Slab,
    pub offsets: Vec<usize>,
    pub total: usize,
    pub scratch: Slab,
    cursor: usize,
    api: crate::hip::HipApi,
}

pub fn upload_shapes(g: &Gguf, dev: &crate::hip::Device, shapes: &[Shape], scratch_bytes: usize) -> Result<WeightSlab> {
    let mut offsets = Vec::new();
    let mut total = 0usize;
    for s in shapes {
        let t = g.tensor(&s.tensor)?;
        total = total.div_ceil(256) * 256;
        offsets.push(total);
        total += t.size as usize;
    }
    let slab = Slab::new(&dev.api, total.max(256), "phase5 weights")?;
    for (i, s) in shapes.iter().enumerate() {
        let t = g.tensor(&s.tensor)?;
        slab.write(&dev.api, offsets[i], g.tensor_data(t))?;
    }
    let scratch = Slab::new(&dev.api, scratch_bytes.max(1 << 20), "phase5 scratch")?;
    scratch.zero(&dev.api)?;
    Ok(WeightSlab { slab, offsets, total, scratch, cursor: 0, api: dev.api })
}

impl WeightSlab {
    /// A deterministic bump allocation, so a failure reproduces at the same offset.
    pub fn alloc(&mut self, len: usize) -> Result<usize> {
        let off = self.cursor.div_ceil(256) * 256;
        if off + len > self.scratch.len() {
            return Err(Error::msg(format!(
                "phase5 scratch is {len} bytes short (scratch slab is {} bytes)",
                self.scratch.len()
            )));
        }
        self.cursor = off + len;
        Ok(off)
    }
    /// Rewind the bump cursor. Each phase (checks, the GEMV sweep, the GEMM table) allocates its
    /// own activations and outputs and none of them outlive the phase, so a phase starts from a
    /// clean slab: the offsets stay deterministic (a failure reproduces at the same place) and the
    /// slab only has to hold the largest single phase, not the sum of all three.
    pub fn rewind(&mut self) {
        self.cursor = 0;
    }
    pub fn saddr(&self, off: usize) -> u64 {
        self.scratch.addr() + off as u64
    }
    pub fn put_f32(&self, off: usize, v: &[f32]) -> Result<()> {
        let bytes: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
        self.scratch.write(&self.api, off, &bytes)
    }

    pub fn get_f32(&self, off: usize, n: usize) -> Result<Vec<f32>> {
        let mut raw = vec![0u8; n * 4];
        hipcheck(
            "phase5 readback",
            self.api.hipMemcpy.unwrap()(
                raw.as_mut_ptr() as *mut c_void,
                self.saddr(off) as *const c_void,
                raw.len(),
                crate::hip::MEMCPY_DEVICE_TO_HOST,
            ),
            &self.api,
        )?;
        Ok(raw.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect())
    }
}

/// Every shape, every supported quant class, GEMV against `dq::Mat::gemv_par` on the same bytes.
/// `norm -> gemv` (the fold the decode program applies) is a separate code path in each kernel, so
/// it is gated separately, and the GEMM path is checked at the token counts the plan names.
/// (The argument count is the kernel ABI plus the device; see `decode::Builder::gemv`.)
#[allow(clippy::too_many_arguments)]
pub fn gemm_checks(p5: &Phase5, g: &Gguf, ws: &mut WeightSlab, shapes: &[Shape], plan: &Plan, dev: &crate::hip::Device, stream: *mut c_void, dbg_lane: Option<&Kernel>, dbg_args: Option<&Kernel>) -> Result<Vec<Check>> {
    ws.rewind();
    let mut out = Vec::new();
    let threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4);
    let only = std::env::var("SN_GEMM_ONLY").unwrap_or_default();
    for (i, s) in shapes.iter().enumerate() {
        if !only.is_empty() && !s.role.contains(&only) {
            continue;
        }
        let t0 = std::time::Instant::now();
        eprintln!("  checking {} {} [{}] ...", s.role, s.key(), s.type_name);
        ws.rewind(); // one shape's buffers never outlive its iteration
        let waddr = ws.slab.addr() + ws.offsets[i] as u64;
        let xoff = ws.alloc(s.k * 4)?;
        let yoff = ws.alloc(s.n * 4)?;
        let woff = ws.alloc(s.k * 4)?;
        let x = activation(900 + i as u64, s.k);
        ws.put_f32(xoff, &x)?;

        // ---- plain GEMV, at the pick the decode program would make -------------------
        let wpr = std::env::var("SN_WPR").ok().and_then(|v| v.parse().ok()).unwrap_or_else(|| pick_warps(s.n, s.k, PICK_CUS));
        let k5 = p5.gemv_for(s.dtype)?;
        let p = gemv_program(k5, waddr, ws.saddr(xoff), ws.saddr(yoff), s, 0, plan.hp.rms_eps, wpr, dev)?;
        p.launch(dev, stream)?;
        crate::hip::check("gemm_checks sync", dev.api.hipStreamSynchronize.unwrap()(stream), &dev.api)?;
        let got = ws.get_f32(yoff, s.n)?;

        // The reference: the same bytes, dequantised and dotted in f32 by `sn-core`.
        let m = Mat::by_name(g, &s.tensor)?;
        let mut want = vec![0f32; s.n];
        m.gemv_par(&x, &mut want, threads);
        if std::env::var("SN_DEBUG").is_ok() && s.role.contains("ssm_beta") {
            // One-hot activations: y[row] is then exactly the dequantised weight at that index, so
            // a wrong element->byte mapping localises itself.
            let w0: Vec<f32> = (0..s.k).map(|e| { let mut v = vec![0f32; s.k]; m.dequant_row(0, &mut v); v[e] }).collect();
            let dbgo = ws.alloc(32 * 4)?;
            if let Some(dbg) = dbg_lane {
                for e in [0usize, 1, 4, 40, 4095] {
                    let mut xe = vec![0f32; s.k];
                    xe[e] = 1.0;
                    ws.put_f32(xoff, &xe)?;
                    let mut b = ProgramBuilder::new(dev);
                    b.kernel("dbg", dbg, [1, 1, 1], 0, &[Arg::Ptr(waddr), Arg::Ptr(ws.saddr(xoff)), Arg::Ptr(ws.saddr(dbgo)), Arg::I32(s.k as i32)])?;
                    let pd = b.finish();
                    pd.launch(dev, stream)?;
                    crate::hip::check("dbg lane sync", dev.api.hipStreamSynchronize.unwrap()(stream), &dev.api)?;
                    let l = ws.get_f32(dbgo, 32)?;
                    eprintln!("    onehot e={e:5} lanes nonzero: {:?}", l.iter().enumerate().filter(|(_, v)| **v != 0.0).map(|(i, v)| format!("{i}:{v:.5}")).collect::<Vec<_>>());
                }
            }
            {
                // The real GEMV kernel, one workgroup, all eight warps on eight real rows.
                let mut s8 = s.clone();
                s8.n = 8;
                let yo = ws.alloc(8 * 4)?;
                for e in [0usize, 4, 4095] {
                    let mut xe = vec![0f32; s.k];
                    xe[e] = 1.0;
                    ws.put_f32(xoff, &xe)?;
                    for w in [1usize, 2, 4] {
                        let pg = gemv_program(k5, waddr, ws.saddr(xoff), ws.saddr(yo), &s8, 0, plan.hp.rms_eps, w, dev)?;
                        pg.launch(dev, stream)?;
                        crate::hip::check("dbg8 sync", dev.api.hipStreamSynchronize.unwrap()(stream), &dev.api)?;
                        let got8 = ws.get_f32(yo, 8)?;
                        let mut want8 = vec![0f32; s.n];
                        m.gemv_par(&xe, &mut want8, 1);
                        eprintln!("    n=8 wpr={w} e={e:5} grid={:?} got {:?}\n                        want {:?}", gemv_grid(8, w), &got8[..4], &want8[..4]);
                        let ao = ws.alloc(8 * 4)?;
                        let dbga = dbg_args.unwrap();
                        let mut pb = ProgramBuilder::new(dev);
                        pb.kernel("dbgargs", dbga, [1, 1, 1], 0, &[Arg::Ptr(ws.saddr(ao)), Arg::I32(s.k as i32), Arg::I32(8), Arg::U64(0), Arg::F32(plan.hp.rms_eps), Arg::I32(w as i32)])?;
                        let pa = pb.finish();
                        pa.launch(dev, stream)?;
                        crate::hip::check("dbgargs sync", dev.api.hipStreamSynchronize.unwrap()(stream), &dev.api)?;
                        let a = ws.get_f32(ao, 7)?;
                        eprintln!("      kernel sees: wpr={} k={} n={} eps={} norm_w={} block={} grid={}", a[0], a[1], a[2], a[3], a[4], a[5], a[6]);
                    }
                }
            }
            let ones = vec![1f32; s.k];
            ws.put_f32(xoff, &ones)?;
            let pe = gemv_program(k5, waddr, ws.saddr(xoff), ws.saddr(yoff), s, 0, plan.hp.rms_eps, wpr, dev)?;
            pe.launch(dev, stream)?;
            crate::hip::check("dbg sync0", dev.api.hipStreamSynchronize.unwrap()(stream), &dev.api)?;
            let ge = ws.get_f32(yoff, 1)?;
            let rowsum: f32 = w0.iter().sum();
            let first64: f32 = (0..16).map(|b| w0[b * 256..b * 256 + 64].iter().sum::<f32>()).sum();
            eprintln!("    x=ones: kernel {:9.4}  full row {:9.4}  first-64-of-each-sb {:9.4}", ge[0], rowsum, first64);
            for e in (0usize..80).chain([96, 97, 128, 129, 4095]) {
                let mut xe = vec![0f32; s.k];
                xe[e] = 1.0;
                ws.put_f32(xoff, &xe)?;
                let pe = gemv_program(k5, waddr, ws.saddr(xoff), ws.saddr(yoff), s, 0, plan.hp.rms_eps, wpr, dev)?;
                pe.launch(dev, stream)?;
                crate::hip::check("dbg sync", dev.api.hipStreamSynchronize.unwrap()(stream), &dev.api)?;
                let ge = ws.get_f32(yoff, 1)?;
                let mut we = vec![0f32; s.n];
                m.gemv_par(&xe, &mut we, 1);
                eprintln!("    e={e:5} kernel {:9.5}  dq {:9.5}  deq_row {:9.5}", ge[0], we[0], w0[e]);
            }
        }
        if std::env::var("SN_DEBUG").is_ok() {
            std::fs::write("/tmp/sn_x2.txt", x.iter().map(|v| format!("{v}\n")).collect::<String>()).ok();
            std::fs::write(format!("/tmp/sn_got_{}.txt", s.type_name), got.iter().map(|v| format!("{v:.9e}\n")).collect::<String>()).ok();
            std::fs::write(format!("/tmp/sn_want_{}.txt", s.type_name), want.iter().map(|v| format!("{v:.9e}\n")).collect::<String>()).ok();
            eprintln!("    {} row0: got {:?} want {:?} (x/got/want in /tmp/sn_*.txt)", s.tensor, &got[..2], &want[..2]);
        }
        out.push(score_g(&format!("gemv {} {} [{}] wpr={wpr}", s.role, s.key(), s.type_name), &got, &want, 1e-4));

        // ---- the same GEMV with the pre-norm folded in -------------------------------
        let norm: Vec<f32> = (0..s.k).map(|j| 1.0 + 0.001 * ((j % 61) as f32) - 0.03).collect();
        ws.put_f32(woff, &norm)?;
        let p2 = gemv_program(k5, waddr, ws.saddr(xoff), ws.saddr(yoff), s, ws.saddr(woff), plan.hp.rms_eps, wpr, dev)?;
        p2.launch(dev, stream)?;
        crate::hip::check("gemm_checks sync 2", dev.api.hipStreamSynchronize.unwrap()(stream), &dev.api)?;
        let got2 = ws.get_f32(yoff, s.n)?;
        let nx = oracle::rmsnorm(&x, Some(&norm), plan.hp.rms_eps);
        let mut want2 = vec![0f32; s.n];
        m.gemv_par(&nx, &mut want2, threads);
        out.push(score_g(&format!("gemv {} {} [{}] + norm folded", s.role, s.key(), s.type_name), &got2, &want2, 1e-4));

        // ---- the GEMM path, on shapes with reuse --------------------------------------
        eprintln!("    {} in {:.1} s", s.role, t0.elapsed().as_secs_f64());
        if role_is(s.role, "ffn_down.weight") || role_is(s.role, "attn_qkv_gate.weight") {
            for t in [8usize, 64, 512] {
                let xg = ws.alloc(t * s.k * 4)?;
                let yg = ws.alloc(t * s.n * 4)?;
                let mut flat = Vec::with_capacity(t * s.k);
                for c in 0..t {
                    flat.extend(activation(1_000 + c as u64 + i as u64, s.k));
                }
                ws.put_f32(xg, &flat)?;
                let gk = p5.gemm_for(s.dtype)?;
                let pg = gemm_program(gk, waddr, ws.saddr(xg), ws.saddr(yg), s.n, s.k, t, dev)?;
                pg.launch(dev, stream)?;
                crate::hip::check("gemm_checks sync 3", dev.api.hipStreamSynchronize.unwrap()(stream), &dev.api)?;
                let gotg = ws.get_f32(yg, t * s.n)?;
                let mut wantg = vec![0f32; t * s.n];
                // The scalar reference is the expensive half of this command: it re-dequantises the
                // weight matrix once per activation column. Full T for the small token counts, and
                // every `stride`-th column at T = 512 - a stride that still lands in every T tile
                // of the kernel's grid, so the tiling is exercised end to end.
                let stride = if t > 64 { 8 } else { 1 };
                for c in (0..t).step_by(stride) {
                    let col = flat[c * s.k..(c + 1) * s.k].to_vec();
                    let dst = &mut wantg[c * s.n..(c + 1) * s.n];
                    m.gemv_par(&col, dst, threads);
                }
                let checked = t / stride;
                // Only the checked columns take part in the score.
                let got_sub: Vec<f32> = (0..t).step_by(stride).flat_map(|c| gotg[c * s.n..(c + 1) * s.n].to_vec()).collect();
                let want_sub: Vec<f32> = (0..t).step_by(stride).flat_map(|c| wantg[c * s.n..(c + 1) * s.n].to_vec()).collect();
                let mut c = score_g(&format!("gemm {} {} x T={t} ({checked}/{t} columns) [{}]", s.role, s.key(), s.type_name), &got_sub, &want_sub, 1e-4);
                c.n = got_sub.len();
                out.push(c);
                let _ = checked;
            }
        }
    }
    Ok(out)
}

/// The f32 GEMV/GEMM paths have no counterpart in *this* checkpoint: §1.2's census says every 2-D
/// weight is Q4_K or Q6_K, and the f32 tensors are the norms and the mHC matrices. So the kernels
/// are validated against a synthetic f32 matrix instead, and the report says so - an untested kernel
/// is worse than a synthetic test.
pub fn f32_checks(p5: &Phase5, ws: &mut WeightSlab, dev: &crate::hip::Device, stream: *mut c_void, plan: &Plan) -> Result<Vec<Check>> {
    let (n, k) = (1024usize, 4096usize);
    let mut out = Vec::new();
    let woff = ws.alloc(n * k * 4)?;
    // Weights and activations from the same generator as everything else here.
    let w: Vec<f32> = (0..n).flat_map(|r| oracle::pattern(7000 + r as u64, k).iter().map(|v| v * 0.02).collect::<Vec<f32>>()).collect();
    ws.put_f32(woff, &w)?;
    let sh = Shape {
        role: "synthetic.f32.weight",
        tensor: "<synthetic f32>".to_string(),
        n,
        k,
        dtype: DType::F32,
        type_name: "f32".to_string(),
        bytes: n * k * 4,
        variant: true,
        visits: 0,
    };
    let xoff = ws.alloc(k * 4)?;
    let yoff = ws.alloc(n * 4)?;
    let wnoff = ws.alloc(k * 4)?;
    let x = activation(999, k);
    ws.put_f32(xoff, &x)?;
    let dot = |x: &[f32], y: &mut [f32]| {
        for r in 0..n {
            y[r] = w[r * k..(r + 1) * k].iter().zip(x).map(|(a, b)| a * b).sum();
        }
    };
    let waddr = ws.saddr(woff);
    for wpr in [1usize, 4] {
        let p = gemv_program(&p5.gemv_f32, waddr, ws.saddr(xoff), ws.saddr(yoff), &sh, 0, plan.hp.rms_eps, wpr, dev)?;
        p.launch(dev, stream)?;
        crate::hip::check("f32 sync", dev.api.hipStreamSynchronize.unwrap()(stream), &dev.api)?;
        let got = ws.get_f32(yoff, n)?;
        let mut want = vec![0f32; n];
        dot(&x, &mut want);
        out.push(score_g(&format!("gemv synthetic f32 {n}x{k} wpr={wpr} (no f32 weights in this checkpoint)"), &got, &want, 1e-4));
    }
    // the folded-norm form, and the prefill form
    let norm: Vec<f32> = (0..k).map(|j| 1.0 + 0.002 * ((j % 37) as f32) - 0.04).collect();
    ws.put_f32(wnoff, &norm)?;
    let p = gemv_program(&p5.gemv_f32, waddr, ws.saddr(xoff), ws.saddr(yoff), &sh, ws.saddr(wnoff), plan.hp.rms_eps, 1, dev)?;
    p.launch(dev, stream)?;
    crate::hip::check("f32 sync 2", dev.api.hipStreamSynchronize.unwrap()(stream), &dev.api)?;
    let got = ws.get_f32(yoff, n)?;
    let nx = oracle::rmsnorm(&x, Some(&norm), plan.hp.rms_eps);
    let mut want = vec![0f32; n];
    dot(&nx, &mut want);
    out.push(score_g("gemv synthetic f32 + norm folded", &got, &want, 1e-4));
    for t in [64usize, 512] {
        let xg = ws.alloc(t * k * 4)?;
        let yg = ws.alloc(t * n * 4)?;
        let mut flat = Vec::with_capacity(t * k);
        for c in 0..t {
            flat.extend(activation(2_000 + c as u64, k));
        }
        ws.put_f32(xg, &flat)?;
        let pg = gemm_program(&p5.gemm_f32, waddr, ws.saddr(xg), ws.saddr(yg), n, k, t, dev)?;
        pg.launch(dev, stream)?;
        crate::hip::check("f32 sync 3", dev.api.hipStreamSynchronize.unwrap()(stream), &dev.api)?;
        let gotg = ws.get_f32(yg, t * n)?;
        let mut wantg = vec![0f32; t * n];
        for c in 0..t {
            dot(&flat[c * k..(c + 1) * k], &mut wantg[c * n..(c + 1) * n]);
        }
        out.push(score_g(&format!("gemm synthetic f32 x T={t}"), &gotg, &wantg, 1e-4));
    }
    Ok(out)
}

fn score_g(name: &str, got: &[f32], want: &[f32], gate: f64) -> Check {
    Check { name: name.to_string(), against: "sn_core::dq", n: got.len(), nmse: oracle::nmse(got, want), max_abs: oracle::max_abs(got, want), gate }
}

// ---------------------------------------------------------------------------------------------
// The GEMV bandwidth sweep and the GEMM throughput table.
// ---------------------------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct GemvRow {
    pub role: String,
    pub shape: String,
    pub type_name: String,
    pub bytes: usize,
    pub visits: usize,
    pub bytes_per_token: usize,
    /// ms per launch, minimum over trials.
    pub ms: f64,
    pub gb_s: f64,
    pub warps_per_row: usize,
    /// The best `warps_per_row` the sweep measured, and its rate.
    pub best_warps: usize,
    pub best_gb_s: f64,
    /// `(warps_per_row, ms, GB/s)`.
    pub sweep: Vec<(usize, f64, f64)>,
}

impl GemvRow {
    pub fn pass(&self, machine_gb_s: f64, need: f64) -> bool {
        machine_gb_s > 0.0 && self.gb_s >= need * machine_gb_s
    }
}

/// Time every shape's GEMV at every legal `warps_per_row`, and record the pick next to the best.
/// (The argument count is the kernel ABI plus the device; see `decode::Builder::gemv`.)
#[allow(clippy::too_many_arguments)]
pub fn gemv_sweep(
    p5: &Phase5,
    dev: &crate::hip::Device,
    stream: *mut c_void,
    ws: &mut WeightSlab,
    shapes: &[Shape],
    plan: &Plan,
    reps: usize,
    trials: usize,
) -> Result<Vec<GemvRow>> {
    ws.rewind();
    let mut rows = Vec::new();
    for (i, s) in shapes.iter().enumerate() {
        if role_is(s.role, "token_embd.weight") {
            // decode gathers it; timing it as a GEMV would report a kernel the engine never runs.
            continue;
        }
        ws.rewind();
        let waddr = ws.slab.addr() + ws.offsets[i] as u64;
        let xoff = ws.alloc(s.k * 4)?;
        let yoff = ws.alloc(s.n * 4)?;
        let k5 = p5.gemv_for(s.dtype)?;
        let mut sweep: Vec<(usize, f64, f64)> = Vec::new();
        let mut wpr = 1usize;
        while wpr <= MAX_WARPS_PER_ROW {
            let p = gemv_program(k5, waddr, ws.saddr(xoff), ws.saddr(yoff), s, 0, plan.hp.rms_eps, wpr, dev)?;
            let ms = time_program(dev, stream, &p, reps, trials)?;
            let gb_s = if ms > 0.0 { s.bytes as f64 / (ms / 1e3) / 1e9 } else { 0.0 };
            sweep.push((wpr, ms, gb_s));
            wpr *= 2;
        }
        let pick = pick_warps(s.n, s.k, PICK_CUS);
        let best = sweep.iter().fold(sweep[0], |a, b| if b.2 > a.2 { *b } else { a });
        let chosen = *sweep.iter().find(|(w, _, _)| *w == pick).unwrap_or(&best);
        rows.push(GemvRow {
            role: s.role.to_string(),
            shape: s.key(),
            type_name: s.type_name.clone(),
            bytes: s.bytes,
            visits: s.visits,
            bytes_per_token: s.bytes_per_token(),
            ms: chosen.1,
            gb_s: chosen.2,
            warps_per_row: pick,
            best_warps: best.0,
            best_gb_s: best.2,
            sweep,
        });
    }
    Ok(rows)
}

#[derive(Clone, Debug)]
pub struct GemmRow {
    pub role: String,
    pub shape: String,
    pub type_name: String,
    pub t: usize,
    pub flops: f64,
    pub ms: f64,
    pub flop_s: f64,
    pub pct_fma_peak: f64,
    pub pct_wmma_peak: f64,
}

/// The prefill path, at the token counts the plan's gate names (`gemm(512)`) and a couple of
/// neighbours.
/// (The argument count is the kernel ABI plus the device; see `decode::Builder::gemv`.)
#[allow(clippy::too_many_arguments)]
pub fn gemm_table(
    p5: &Phase5,
    dev: &crate::hip::Device,
    stream: *mut c_void,
    ws: &mut WeightSlab,
    shapes: &[Shape],
    ts: &[usize],
    reps: usize,
    trials: usize,
    machine: &Machine,
) -> Result<Vec<GemmRow>> {
    ws.rewind();
    let mut rows = Vec::new();
    for (i, s) in shapes.iter().enumerate() {
        if role_is(s.role, "token_embd.weight") || role_is(s.role, "output.weight") {
            continue;
        }
        ws.rewind();
        let waddr = ws.slab.addr() + ws.offsets[i] as u64;
        let gk = p5.gemm_for(s.dtype)?;
        for &t in ts {
            if !matches!(pick_path(t, s.n, s.k, PICK_CUS), Path::Gemm) {
            continue;
        }
            let xg = ws.alloc(t * s.k * 4)?;
            let yg = ws.alloc(t * s.n * 4)?;
            let prog = gemm_program(gk, waddr, ws.saddr(xg), ws.saddr(yg), s.n, s.k, t, dev)?;
            let ms = time_program(dev, stream, &prog, reps, trials)?;
            let flops = 2.0 * s.n as f64 * s.k as f64 * t as f64;
            let flop_s = if ms > 0.0 { flops / (ms / 1e3) } else { 0.0 };
            rows.push(GemmRow {
                role: s.role.to_string(),
                shape: s.key(),
                type_name: s.type_name.clone(),
                t,
                flops,
                ms,
                flop_s,
                pct_fma_peak: if machine.fma_flop_s > 0.0 { 100.0 * flop_s / machine.fma_flop_s } else { 0.0 },
                pct_wmma_peak: if machine.wmma_flop_s > 0.0 { 100.0 * flop_s / machine.wmma_flop_s } else { 0.0 },
            });
        }
    }
    Ok(rows)
}

/// The Phase 5 gates, as a pass/fail list `sn bench gemm` prints and `Docs/bench.md` quotes.
pub fn gemm_gates(rows: &[GemvRow], gemms: &[GemmRow], m: &Machine) -> Vec<(String, bool, String)> {
    // "gemv achieves >= 80 % of measured device bandwidth on ffn_down shapes".
    // "gemv achieves >= 80 % of measured device bandwidth on ffn_down shapes". Reported per
    // quantisation class, because the two classes are limited by different things on this card
    // (design-plan Phase 5 status): Q4_K's 144-byte super-block is 16-byte aligned and reaches the
    // vector-load path, Q6_K's 210-byte one is not, and one blended number would hide that.
    let mut out: Vec<(String, bool, String)> = Vec::new();
    for ty in ["q4_k", "q6_k"] {
        let ffn: Vec<&GemvRow> = rows
            .iter()
            .filter(|r| role_is(&r.role, "ffn_down.weight") && r.type_name == ty)
            .collect();
        if ffn.is_empty() {
            continue;
        }
        let worst = ffn.iter().map(|r| r.gb_s).fold(f64::INFINITY, f64::min);
        let pct = 100.0 * worst / m.read_gb_s.max(1.0);
        out.push((
            format!("gemv >= 80% of measured device read bandwidth on ffn_down shapes [{ty}]"),
            m.read_gb_s > 0.0 && worst >= 0.8 * m.read_gb_s,
            format!("worst {worst:.1} GB/s of {:.1} GB/s measured ({pct:.1}%)", m.read_gb_s),
        ));
    }
    // "gemm(512) >= 60 % of WMMA peak" - against the measured WMMA peak, and the measured f32-FMA
    // peak next to it, because this phase's GEMM is an FMA kernel (see the module docs).
    let g512: Vec<&GemmRow> = gemms.iter().filter(|r| r.t == 512).collect();
    let best_fma = g512.iter().map(|r| r.pct_fma_peak).fold(0.0f64, f64::max);
    // The WMMA gate is only meaningful if the probe measured a matrix operation. It did not: the
    // implied cost per 16x16x16 op is below one cycle at any plausible clock, so the builtin's
    // throughput on this ISA is *unmeasured*, and inventing a denominator would be worse than
    // failing the gate honestly.
    out.push((
        "gemm(512) >= 60% of measured WMMA peak".to_string(),
        false,
        format!(
            "NOT EVALUABLE - the WMMA probe implies {:.3e} ns of device time per 16x16x16 op per wavefront, which is below one cycle at any clock this card runs at; {:.3e} FLOP/s therefore is not a machine peak. A real tiled f16 microkernel has to exist before this gate can be quoted (design-plan Phase 5 status).",
            m.wmma_ns_per_op,
            m.wmma_flop_s
        ),
    ));
    out.push((
        "gemm(512) >= 60% of measured f32-FMA peak (the reference this kernel is written against)".to_string(),
        best_fma >= 60.0,
        format!("best {best_fma:.1}% of {:.1} TF/s measured FMA", m.fma_flop_s / 1e12),
    ));
    out.push((
        "every GEMV pick is the best the sweep measured".to_string(),
        rows.iter().all(|r| r.warps_per_row == r.best_warps || r.gb_s >= 0.95 * r.best_gb_s),
        format!(
            "{} shapes; within 5% of the sweep's best on every one (the run-to-run spread on this card is larger than that - bench.md 2.3); {} above it: {}",
            rows.len(),
            rows.iter().filter(|r| r.warps_per_row != r.best_warps && r.gb_s < 0.95 * r.best_gb_s).count(),
            rows.iter()
                .filter(|r| r.warps_per_row != r.best_warps && r.gb_s < 0.95 * r.best_gb_s)
                .map(|r| format!("{} pick wpr={} {:.0} GB/s vs wpr={} {:.0}", r.role, r.warps_per_row, r.gb_s, r.best_warps, r.best_gb_s))
                .collect::<Vec<_>>()
                .join("; ")
        ),
    ));
    out
}

pub fn gemm_report(rows: &[GemvRow], gemms: &[GemmRow], m: &Machine, checks: &[Check]) -> String {
    let mut s = String::new();
    s += &format!(
        "measured references on this device:\n  read bandwidth   {:.1} GB/s over {} B\n  f32 FMA peak     {:.2} TFLOP/s\n  WMMA peak        {}\n",
        m.read_gb_s,
        m.read_bytes,
        m.fma_flop_s / 1e12,
        if m.wmma_available {
            format!(
                "{:.3e} FLOP/s but NOT a trustworthy peak: {:.3e} ns per 16x16x16 op per wavefront (< 1 cycle); treat as unmeasured",
                m.wmma_flop_s, m.wmma_ns_per_op
            )
        } else {
            "not available on this ISA (probe compiled out)".to_string()
        }
    );
    s += &format!("\n{:<26} {:>13} {:>6} {:>12} {:>12} {:>10} {:>10} {}\n", "role", "shape", "type", "bytes", "B/token", "GB/s", "pick", "sweep GB/s (wpr)");
    for r in rows {
        s += &format!(
            "{:<26} {:>13} {:>6} {:>12} {:>12} {:>10.1} {:>10} {}\n",
            r.role,
            r.shape,
            r.type_name,
            r.bytes,
            r.bytes_per_token,
            r.gb_s,
            r.warps_per_row,
            r.sweep.iter().map(|(w, _, g)| format!("{g:.0}/{w}")).collect::<Vec<_>>().join(" ")
        );
    }
    if !gemms.is_empty() {
        s += &format!("\n{:<26} {:>13} {:>6} {:>8} {:>12} {:>10} {:>12} {:>12}\n", "role", "shape", "type", "T", "GFLOP", "ms", "GFLOP/s", "%FMA/%WMMA");
        for r in gemms {
            s += &format!(
                "{:<26} {:>13} {:>6} {:>8} {:>12.1} {:>10.3} {:>10.0} {:>7.1}/{:<4.1}\n",
                r.role, r.shape, r.type_name, r.t, r.flops / 1e9, r.ms, r.flop_s / 1e9, r.pct_fma_peak, r.pct_wmma_peak
            );
        }
    }
    if !checks.is_empty() {
        s += &format!("\nsn_core::dq reference vs device (same bytes):\n{}", crate::check::report(checks));
    }
    s += "\n";
    for (what, ok, detail) in gemm_gates(rows, gemms, m) {
        s += &format!("  [{}] {} {}\n", if ok { "PASS" } else { "FAIL" }, what, detail);
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_wide_matrix_needs_no_split_and_a_narrow_one_does() {
        // The two ends of this model: ffn_down (4096 rows) and ssm_beta_alpha (64 rows).
        assert_eq!(pick_warps(4096, 12288, 40), 1, "4096 rows is already 4096 wavefronts");
        assert_eq!(pick_warps(248320, 4096, 40), 1);
        assert!(pick_warps(64, 4096, 40) > 1, "64 rows cannot fill the machine");
        assert!(pick_warps(64, 4096, 40) <= MAX_WARPS_PER_ROW);
        // A K axis too short to split is not split, however narrow the matrix is.
        assert_eq!(pick_warps(64, 256, 40), 1, "one super-block cannot be shared out");
        // Never zero, never past the workgroup.
        for (n, k) in [(1, 256), (7, 512), (64, 4096), (4096, 512), (1024, 4096)] {
            let w = pick_warps(n, k, 40);
            assert!((1..=MAX_WARPS_PER_ROW).contains(&w), "{n}x{k} -> {w}");
        }
    }

    #[test]
    fn the_grid_covers_every_row_exactly_once() {
        for (n, wpr) in [(4096, 1), (248320, 1), (64, 4), (1024, 2), (12288, 1)] {
            let grid = gemv_grid(n, wpr);
            let rows_per_wg = (GEMV_BLOCK as usize) / (32 * wpr);
            assert_eq!(grid[0] as usize, n.div_ceil(rows_per_wg));
            assert!(!grid.contains(&0));
        }
    }

    #[test]
    fn one_token_is_a_gemv_and_a_prefill_is_a_gemm() {
        assert_eq!(pick_path(1, 4096, 12288, 40), Path::Gemv { warps_per_row: 1 });
        assert!(matches!(pick_path(512, 4096, 12288, 40), Path::Gemm));
        assert!(matches!(pick_path(4, 4096, 12288, 40), Path::GemvStrided { tokens: 4, .. }));
        assert_eq!(pick_path(0, 8, 8, 40), Path::Invalid);
    }

    #[test]
    fn role_names_carry_their_block_prefix() {
        // `budget::tensor_class` normalises to `blk.N.<role>`; a bare `==` silently matches nothing,
        // which is how the GEMM checks once disappeared from the report.
        assert!(role_is("blk.N.ffn_down.weight", "ffn_down.weight"));
        assert!(role_is("output.weight", "output.weight"));
        assert!(!role_is("blk.N.attn_qkv_gate.weight", "attn_q.weight"));
        assert!(!role_is("blk.N.attn_q.weight", "attn_qkv_gate.weight"));
    }

    #[test]
    fn the_shape_table_is_the_model_we_describe() {
        // Every role Phase 5 claims to have a kernel for must be a role `model_shapes` can produce,
        // and the type names it reads out of the GGUF must be the ones `register_phase5` serves.
        for r in MATMUL_ROLES {
            assert!(role_is(r, r), "{r}");
        }
        for (_, symbol) in PHASE5_SPECS {
            assert!(symbol.starts_with("sn_"), "{symbol}");
        }
        assert!(PHASE5_SPECS.iter().any(|(n, _)| *n == "gemv_q4_k"));
        assert!(PHASE5_SPECS.iter().any(|(n, _)| *n == "gemm_f32"));
    }
}
