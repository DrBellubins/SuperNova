//! Compile **one decode step** into the flat op list the P1 design replays, and count it.
//!
//! Phase 4's gate asks for "node/op count for a full decode step <= 300". Answering that needs a
//! builder, not an estimate: this module walks [`sn_core::plan::Plan`] (the single source for the
//! loop/mHC structure, per §8 "never re-derive the loop/mHC/plan logic inside a kernel") and emits
//! one [`OpSpec`] per kernel launch, with every pointer resolved from arena offsets plus slab
//! bases. No HIP is needed to compile or count it - that is the point: a decode program is data.
//!
//! The list contains two kinds of entry:
//!
//! * **real Phase 4 kernels** (`sn_rmsnorm_f32`, `sn_silu_mul_f32`, `sn_sigmoid_f32`,
//!   `sn_rope_inplace_f32`, `sn_attn_gate_norm_f32`, `sn_mhc_seed_f32`, `sn_mhc_route_f32`,
//!   `sn_mhc_merge_f32`, `sn_mhc_expose_f32`, `sn_add_f32`) - registered, launchable today;
//! * **named pending kernels** for whoever owns them: `gemv_q4_k`/`gemv_q6_k`/`gemv_f32` (Phase 5),
//!   `kv_append_f16` / `attn_decode_f16` / `gdn_conv_silu` / `gdn_l2norm` / `gdn_decode` /
//!   `gdn_gate_norm` (Phase 6), `row_gather_*` (the embedding). They are in
//!   the count, because the count is what the gate is about, and [`ProgramSpec::build`] reports
//!   them as pending rather than pretending they exist. Their argument lists are the ABI those
//!   phases inherit, which is the other half of why this lives here.
//!
//! [`Fusion`] spells the §6 P2 fusions as flags so the count can be reported as a ladder
//! (raw -> fused) rather than one number; a fused launch drops the producer and passes its inputs
//! to the consumer instead (the consumer's argument list is fixed, with `norm_w = 0` meaning
//! "not folded").

use sn_core::error::{Error, Result};
use sn_core::plan::Plan;

use crate::arena::Arena;
use crate::cache::{CacheLayout, KvKind, META_LEN};
use crate::kernels::{DType, KernelRegistry};
use crate::ops::{grid1d, BLOCK};
use crate::program::{Arg, ProgramBuilder};

/// Which §6 P2 fusions the compiled program applies. Each removes launches; they are independent,
/// so the report can attribute the win to each one.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Fusion {
    /// `mhc_merge` and the `mhc_route` that follows it in one kernel.
    pub route_with_merge: bool,
    /// The pre-norm `rms(x) * w` folded into the GEMV that reads the result.
    pub norm_into_gemv: bool,
    /// `silu(gate) * up` folded into the `ffn_down` GEMV, reading both projections.
    pub silu_into_gemv: bool,
    /// RoPE folded into the Q/K producer's epilogue (plan Phase 7).
    pub rope_into_gemv: bool,
    /// The `sigmoid(g) * attn_out` gate folded into `attn_output`'s GEMV prologue (plan Phase 7).
    pub gate_into_gemv: bool,
    /// Weight matrices that share one input launched together: `ffn_gate`+`ffn_up` (Phase 7:
    /// `sn_gemv_q4_k_pair`, a real kernel). Kept separate from `multi_qkv` because the two have
    /// different feasibility: gate/up are the same type and shape in every block, q/k/v are not
    /// (`attn_v` is Q6_K in five blocks), so a single "batch the GEMVs" switch would claim a
    /// fusion the kernels cannot make.
    pub multi_gemv: bool,
    /// The `q`+`k`+`v` triple in one launch. Still a pending name: see `multi_gemv`.
    pub multi_qkv: bool,
}

impl Fusion {
    /// Nothing fused: one kernel per logical step of §2. The honest floor for "what a naive port
    /// of the reference forward would launch".
    pub const RAW: Fusion = Fusion {
        route_with_merge: false,
        norm_into_gemv: false,
        silu_into_gemv: false,
        gate_into_gemv: false,
        rope_into_gemv: false,
        multi_gemv: false,
        multi_qkv: false,
    };
    /// The §6 P2 target.
    pub const FUSED: Fusion = Fusion {
        route_with_merge: true,
        norm_into_gemv: true,
        silu_into_gemv: true,
        gate_into_gemv: true,
        rope_into_gemv: true,
        multi_gemv: true,
        multi_qkv: true,
    };

    /// The same preset with one fusion turned off, for the ladder.
    pub fn without(self, which: FusionFlag) -> Fusion {
        let mut f = self;
        match which {
            FusionFlag::RouteMerge => f.route_with_merge = false,
            FusionFlag::Norm => f.norm_into_gemv = false,
            FusionFlag::Silu => f.silu_into_gemv = false,
            FusionFlag::Rope => f.rope_into_gemv = false,
            FusionFlag::Gate => f.gate_into_gemv = false,
            FusionFlag::Multi => f.multi_gemv = false,
            FusionFlag::MultiQkv => f.multi_qkv = false,
        }
        f
    }

    pub fn flags(&self) -> [(&'static str, bool); 7] {
        [
            ("route+merge", self.route_with_merge),
            ("norm->gemv", self.norm_into_gemv),
            ("silu->gemv", self.silu_into_gemv),
            ("rope->gemv", self.rope_into_gemv),
            ("gate->gemv", self.gate_into_gemv),
            ("multi-gemv pair", self.multi_gemv),
            ("multi-gemv qkv", self.multi_qkv),
        ]
    }
}

/// One fusion, nameable for the ladder.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FusionFlag {
    RouteMerge,
    Norm,
    Silu,
    Rope,
    Gate,
    Multi,
    MultiQkv,
}

/// One planned launch.
#[derive(Clone, Debug)]
pub struct OpSpec {
    pub label: String,
    /// Coarse class for the report: `route`, `merge`, `norm`, `gemv`, `rope`, `gate`, `silu`,
    /// `recur` (attention/GDN inner kernels), `add`, `head`, `input`.
    pub class: &'static str,
    /// The kernel symbol, or the name the owning phase will register.
    pub kernel: String,
    /// True for kernels this phase cannot launch yet.
    pub pending: bool,
    pub grid: [u32; 3],
    pub block: u32,
    pub shared: u32,
    pub args: Vec<Arg>,
}

/// A compiled decode step: the launches, in order, with baked arguments.
pub struct ProgramSpec {
    pub ops: Vec<OpSpec>,
    /// Pinned-host -> device copies the program starts with. These are `Op::H2DPinned` nodes, not
    /// kernels, and they are the only host writes in a decode step (P1's "write a token id and a
    /// position, launch one graph"). `(label, device address, pinned host address, length)`; a host
    /// address of 0 means the program was compiled for counting only and cannot be launched.
    pub inputs: Vec<(String, u64, u64, usize)>,
    pub fusion: Fusion,
    /// Bytes of activation arena the program addresses, i.e. the arena `Slots` reserved.
    pub arena_bytes: usize,
    /// `(execution, exclusive op index)` for every trunk execution, in order. This is what lets a
    /// debug run stop the program after execution *N* and compare the arena against the Phase 2
    /// stage dump for *N* — the bisect tool for "the forward is wrong, which block is wrong".
    /// Without it, finding the first wrong block means reading 900 labels by eye.
    pub exec_ends: Vec<(usize, usize)>,
}

impl ProgramSpec {
    pub fn len(&self) -> usize {
        self.ops.len()
    }
    pub fn is_empty(&self) -> bool {
        self.ops.is_empty()
    }
    /// Launches backed by a kernel that exists today.
    pub fn launchable(&self) -> usize {
        self.ops.iter().filter(|o| !o.pending).count()
    }
    pub fn pending(&self) -> usize {
        self.ops.iter().filter(|o| o.pending).count()
    }

    /// `class -> count`, biggest first.
    pub fn counts(&self) -> Vec<(&'static str, usize)> {
        let mut out: Vec<(&'static str, usize)> = Vec::new();
        for o in &self.ops {
            match out.iter_mut().find(|(c, _)| *c == o.class) {
                Some((_, n)) => *n += 1,
                None => out.push((o.class, 1)),
            }
        }
        out.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(b.0)));
        out
    }

    /// How much of the program a fusion preset is worth, in launches: `FUSED` vs `RAW`.
    pub fn saved_vs_raw(&self) -> usize {
        0 // filled in by the caller; kept out of ProgramSpec so counts stay a pure function
    }

    pub fn report(&self) -> String {
        let mut s = format!(
            "decode step: {} planned launches ({} launchable today, {} pending) + {} pinned copies, arena {}\n",
            self.len(),
            self.launchable(),
            self.pending(),
            self.inputs.len(),
            crate::human(self.arena_bytes)
        );
        for (class, n) in self.counts() {
            s += &format!("  {:<8} {:>5}\n", class, n);
        }
        s
    }

    /// Append the pinned input copies and the launchable ops to a [`ProgramBuilder`]. Pending
    /// kernels are skipped unless `strict`, which is what Phase 7 will use to prove the program is
    /// complete. The input copies come first: they are how a token id, a position and a cache
    /// length reach a program that is never rebuilt (P1).
    pub fn build<'b>(&self, reg: &KernelRegistry, b: &mut ProgramBuilder<'b>, strict: bool) -> Result<usize> {
        let mut n = 0;
        for (label, dst, host, len) in &self.inputs {
            if *host == 0 {
                if strict {
                    return Err(Error::msg(format!("{label}: compiled without a pinned host base, cannot launch")));
                }
                continue;
            }
            b.h2d_pinned(label, *dst, *host, *len);
        }
        for o in &self.ops {
            let k = reg
                .find(&o.kernel, &[], DType::F32)
                .or_else(|| reg.find(&o.kernel, &[], DType::Any))
                .or_else(|| reg.find(&o.kernel, &[], DType::F16));
            let k = match k {
                Some(k) => k,
                None if o.pending && !strict => continue,
                None => {
                    return Err(Error::msg(format!("{}: kernel {} is not registered", o.label, o.kernel)))
                }
            };
            // The ABI guard: the image records how many parameters each entry point takes, and a
            // launch that passes a different number is a wrong-number bug waiting to happen, not a
            // fault. HIP itself never notices.
            if let Some(n) = reg.arity(&o.kernel) {
                if n != o.args.len() {
                    return Err(Error::msg(format!(
                        "{}: {} takes {n} arguments but this launch passes {} - the kernel image and \
                         decode.rs disagree about the ABI (run Tools/build-kernels.sh if the image is stale)",
                        o.label,
                        o.kernel,
                        o.args.len()
                    )));
                }
            }
            b.kernel(&o.label, k, o.grid, o.shared, &o.args)?;
            n += 1;
        }
        Ok(n)
    }
}

/// Every activation slot a decode step needs, offsets resolved. One set only: inside an execution
/// the data flow is a line, so nothing is live across executions except `cur`/`a`/`b` and the rope
/// table (Phase 7's "3-slot ring" is the same idea at coarser grain).
#[derive(Clone, Debug)]
pub struct Slots {
    pub d: usize,
    pub pos: usize,
    pub token: usize,
    pub cos: usize,
    pub sin: usize,
    pub cur: usize,
    pub a: usize,
    pub b: usize,
    pub route_in: usize,
    pub h: usize,
    pub branch: usize,
    pub logits: usize,
    pub post: usize,
    pub mix: usize,
    pub qg: usize,
    pub k: usize,
    pub v: usize,
    pub attn_out: usize,
    pub attn_gated: usize,
    pub qkvz: usize,
    pub ba: usize,
    pub conv_out: usize,
    pub gdn_core: usize,
    pub ffn_g: usize,
    pub ffn_u: usize,
    pub ffn_act: usize,
    pub head: usize,
}

/// f32 activations throughout Phase 4; the f16 kernel variants exist so Phase 7 can move them
/// without touching a single call site's shape.
fn f32s(n: usize) -> usize {
    n * 4
}

impl Slots {
    pub fn reserve(act: &mut Arena, plan: &Plan, n_ctx: usize) -> Result<(Slots, usize)> {
        let hp = &plan.hp;
        let dm = &hp.dims;
        let d = hp.n_embd;
        let np = hp.n_rot / 2;
        let s = Slots {
            d,
            pos: act.reserve_aligned("pos", 4)?,
            token: act.reserve_aligned("token_id", 4)?,
            cos: act.reserve_aligned("rope_cos", f32s(n_ctx.max(1) * np))?,
            sin: act.reserve_aligned("rope_sin", f32s(n_ctx.max(1) * np))?,
            cur: act.reserve_aligned("cur", f32s(d))?,
            a: act.reserve_aligned("stream_a", f32s(d))?,
            b: act.reserve_aligned("stream_b", f32s(d))?,
            route_in: act.reserve_aligned("route_in", f32s(d))?,
            h: act.reserve_aligned("h", f32s(d))?,
            branch: act.reserve_aligned("branch", f32s(d))?,
            logits: act.reserve_aligned("route_logits", f32s(dm.n_route))?,
            post: act.reserve_aligned("route_post", f32s(2))?,
            mix: act.reserve_aligned("route_mix", f32s(4))?,
            qg: act.reserve_aligned("attn_qg", f32s(dm.n_q * 2))?,
            k: act.reserve_aligned("attn_k", f32s(dm.n_kv))?,
            v: act.reserve_aligned("attn_v", f32s(dm.n_kv))?,
            attn_out: act.reserve_aligned("attn_out", f32s(dm.n_q))?,
            attn_gated: act.reserve_aligned("attn_gated", f32s(dm.n_q))?,
            qkvz: act.reserve_aligned("gdn_qkvz", f32s(dm.qkv_width + d))?,
            ba: act.reserve_aligned("gdn_beta_alpha", f32s(2 * dm.n_v_heads))?,
            conv_out: act.reserve_aligned("gdn_conv", f32s(dm.qkv_width))?,
            gdn_core: act.reserve_aligned("gdn_core", f32s(dm.n_v_heads * dm.d_v))?,
            ffn_g: act.reserve_aligned("ffn_gate", f32s(hp.n_ffn))?,
            ffn_u: act.reserve_aligned("ffn_up", f32s(hp.n_ffn))?,
            ffn_act: act.reserve_aligned("ffn_act", f32s(hp.n_ffn))?,
            head: act.reserve_aligned("logits", f32s(hp.vocab))?,
        };
        Ok((s, act.total()))
    }
}

/// The bases a compiled program bakes into its pointers.
#[derive(Clone, Copy, Debug, Default)]
pub struct Bases {
    pub activations: u64,
    pub weights: u64,
    /// The cache slab (`Substrate::caches`), or 0 when the program has no caches.
    pub caches: u64,
    /// The pinned host slot, or 0 when the program is only being counted.
    pub pinned: u64,
}

/// Which cached state a compiled step belongs to. Every cache address the program bakes is
/// `caches + CacheLayout::...`, so a second sequence is a second `compile`, not a graph update.
#[derive(Clone, Copy, Debug)]
pub struct CacheArgs<'a> {
    pub layout: &'a CacheLayout,
    pub seq: usize,
    pub kind: KvKind,
}

impl CacheArgs<'_> {
    fn kind_code(&self) -> usize {
        match self.kind {
            KvKind::F16 => 0,
            KvKind::Q8_0 => 1,
            KvKind::Q4_0 => 2,
        }
    }
}

/// The pinned host slot layout. One block, one H2D copy per step: the token id, the position, and
/// the per-sequence meta rows (`cache::Sessions::meta_bytes`), which is where the attention kernel
/// reads the current cache length from. A captured graph takes new inputs without being rebuilt.
pub const PIN_POS: usize = 0;
pub const PIN_TOKEN: usize = 4;
/// Where the program's own argmax writes the token id it sampled (see `sn_argmax_f32`): the host
/// reads four bytes instead of copying back the vocabulary.
pub const PIN_SAMPLE: usize = 128;
pub const PIN_META: usize = 256;
pub const PIN_BYTES: usize = 4096;

/// Shared memory `sn_mhc_route_f32` needs: the wavefront partials, the 8 x workgroup reduction
/// scratch, and the eight finals. Matches the `__shared__` declarations in `kernels/phase4.hip`.
/// The routing kernel's **static** shared footprint. It is *not* a dynamic-shared size: passing it
/// as `sharedBytes` to `hipModuleLaunchKernel` reserves a second 32 KB region on top of the one the
/// kernel already declares, which on gfx1102 pushes the workgroup over the 64 KB per-block limit and
/// silently corrupts the reduction scratch (measured: every router logit but the first, and the
/// corruption moved with the size argument). Every launch site in this crate passes 0 instead; this
/// function exists so the footprint stays asserted in one place.
pub fn route_static_shared() -> u32 {
    ((crate::ops::ROUTE_BLOCK / 32) + 8 * crate::ops::ROUTE_BLOCK + 8) * 4
}

fn i(v: usize) -> Arg {
    Arg::I32(v as i32)
}
fn u(v: usize) -> Arg {
    Arg::U64(v as u64)
}

/// Compile one decode step (`T = 1`) for `plan`.
///
/// `weights` resolves a tensor name to its byte offset in the weight slab (the GGUF `offset`,
/// because the slab is the data section uploaded verbatim) and to its type name (`q4_k`, `q6_k`,
/// `f32`), which decides which GEMV the tensor needs. Taking them as closures keeps this module
/// independent of `Gguf`, so the counts are testable without a model file.
pub fn compile(
    plan: &Plan,
    slots: &Slots,
    bases: Bases,
    arena_bytes: usize,
    fusion: Fusion,
    weights: &dyn Fn(&str) -> Result<(u64, String)>,
) -> Result<ProgramSpec> {
    compile_inner(plan, slots, bases, arena_bytes, fusion, weights, None)
}

/// The Phase 6 form: the same walk, with the caches of one sequence resolved, so the recurrent
/// kernels (`kv_append`, `attn_decode`, the GDN set) and the embedding gather are real launches
/// rather than names in a count.
pub fn compile_with_caches(
    plan: &Plan,
    slots: &Slots,
    bases: Bases,
    arena_bytes: usize,
    fusion: Fusion,
    weights: &dyn Fn(&str) -> Result<(u64, String)>,
    cache: &CacheArgs,
) -> Result<ProgramSpec> {
    compile_inner(plan, slots, bases, arena_bytes, fusion, weights, Some(cache))
}

fn compile_inner(
    plan: &Plan,
    slots: &Slots,
    bases: Bases,
    arena_bytes: usize,
    fusion: Fusion,
    weights: &dyn Fn(&str) -> Result<(u64, String)>,
    cache: Option<&CacheArgs>,
) -> Result<ProgramSpec> {
    let hp = &plan.hp;
    let dm = &hp.dims;
    let d = hp.n_embd;
    let mut b = Builder {
        ops: Vec::new(),
        inputs: Vec::new(),
        exec_ends: Vec::new(),
        cache,
        bases,
        slots: slots.clone(),
        fusion,
        weights,
        n_ffn: hp.n_ffn,
        n_route: dm.n_route,
        rms_eps: hp.rms_eps,
        mhc_iters: hp.mhc_iters,
        mhc_pre_scale: hp.mhc_pre_scale,
        mhc_fixed_identity: hp.mhc_fixed_identity,
    };

    // ---- input staging --------------------------------------------------------------
    // The token id and the position arrive in pinned host memory; the graph's first two nodes copy
    // them into the arena. No hipMemcpy in the loop, no allocation, no graph update (P1).
    // The device side of each copy is the arena slot the kernels read; the host side is the pinned
    // slot. Writing to `activations + PIN_POS` instead of `activations + slots.pos` is the kind of
    // mistake that shows up as a garbage embedding row, not an error.
    b.input_from("input:pos", PIN_POS, slots.pos, 4);
    b.input_from("input:token", PIN_TOKEN, slots.token, 4);
    if let Some(c) = cache {
        // The per-sequence meta rows, including the current cache length the attention kernels read.
        // One node for every sequence, so a multi-sequence engine still has one host write.
        let n_seq = c.layout.geom.as_ref().map(|g| g.n_seq).unwrap_or(1);
        b.input_to("input:seq_meta", PIN_META, bases.caches + c.layout.meta as u64, n_seq * 16);
    }

    // ---- embedding ------------------------------------------------------------------
    // One row of a quantised table, gathered rather than multiplied against a one-hot vector.
    let (emb_off, emb_ty) = weights("token_embd.weight")?;
    // Phase 6 has the gather for the type this checkpoint stores; anything else stays an honest
    // pending name rather than a wrong-typed launch.
    let have_gather = cache.is_some() && emb_ty == "q4_k";
    let gather_name = if have_gather { "sn_row_gather_q4_k".to_string() } else { format!("row_gather_{emb_ty}") };
    b.push(
        "embd row",
        "input",
        &gather_name,
        !have_gather,
        [1, 1, 1],
        BLOCK,
        0,
        vec![b.w_off(emb_off), b.a(slots.token), b.a(slots.cur), i(d)],
    );

    // ---- the trunk ------------------------------------------------------------------
    // The trunk in execution order with MTP excluded, so §2.7's two boundaries - the first mHC
    // execution, where the streams come into existence, and the last one, where the trunk reads
    // them back - come from the plan instead of being guessed from the mHC range.
    let trunk: Vec<usize> = (0..plan.n_exec()).filter(|&il| plan.exec(il).mtp == 0).collect();
    for (n, &il) in trunk.iter().enumerate() {
        let ex = *plan.exec(il);
        let blk = ex.tensor_block;
        let t = |role: &str| format!("blk.{blk}.{role}");
        let mhc = ex.mhc;
        let prev_mhc = n > 0 && plan.exec(trunk[n - 1]).mhc;
        let next_mhc = n + 1 < trunk.len() && plan.exec(trunk[n + 1]).mhc;

        // ---- route + pre-norm --------------------------------------------------------
        // Inside the mHC range the branch reads the route's `in`; outside it, `cur` (§2.2, §2.7).
        // Nothing outside the range has written `a`/`b`, so they are seeded here, from the hidden
        // state that reaches the first mHC execution - the same `a = b = cur` `forward.rs` does.
        if mhc && ex.physical == hp.mhc_start {
            // §2.7: the streams come into existence at the *first* execution of every loop pass,
            // not once per trunk. The reference re-seeds whenever `physical == mhc.start` (logical 4
            // and logical 28 here), and what it seeds from is the hidden state the previous
            // execution exposed, i.e. `0.5*(a+b)`. Skipping that is silent: the second pass simply
            // continues the first pass's streams, which is a plausible-looking forward whose logits
            // are 3e-2 off (measured 2026-11-27, `sn gpu stages`).
            if prev_mhc {
                b.expose_streams();
            }
            b.seed_streams(blk);
        }
        let x_src = if mhc {
            b.route("attn", &t("hc_attn_fn.weight"), &t("hc_attn_fn.bias"), blk);
            slots.route_in
        } else {
            slots.cur
        };
        let h_src = if b.fusion.norm_into_gemv {
            x_src
        } else {
            b.rmsnorm(&format!("rmsnorm attn blk.{blk}"), x_src, &t("attn_norm.weight"), slots.h, blk)?;
            slots.h
        };

        // ---- the branch --------------------------------------------------------------
        if ex.recurrent {
            let norm_name = t("attn_norm.weight");
            let qkv_name = t("attn_qkv_gate.weight");
            b.gemv(
                &format!("gemv qkv_gate blk.{blk}"),
                &qkv_name,
                h_src,
                slots.qkvz,
                d,
                dm.qkv_width + d,
                if b.fusion.norm_into_gemv { Some(norm_name.as_str()) } else { None },
                blk,
            )?;
            // The *second* GEMV that reads this execution's normalised input. With `norm_into_gemv`
            // there is no buffer holding the normalised vector, so each GEMV that reads one has to
            // fold the norm itself: leaving it off here made beta/alpha come out of the *un*normalised
            // hidden state - a plausible wrong gate, not a fault (found 2026-11-27 by `sn run
            // --raw`, which matches the reference to 3e-12 and is 6.8e-3 off with the fold on).
            b.gemv(
                &format!("gemv beta_alpha blk.{blk}"),
                &t("ssm_beta_alpha.weight"),
                h_src,
                slots.ba,
                d,
                2 * dm.n_v_heads,
                if b.fusion.norm_into_gemv { Some(norm_name.as_str()) } else { None },
                blk,
            )?;
            // beta = sigmoid(beta_raw), in place over the first half of the fused projection.
            b.push(
                &format!("sigmoid beta blk.{blk}"),
                "silu",
                "sn_sigmoid_f32",
                false,
                grid1d(dm.n_v_heads),
                BLOCK,
                0,
                vec![b.a(slots.ba), b.a(slots.ba), u(dm.n_v_heads)],
            );
            // The GDN recurrent kernels are Phase 6's; named, counted, reported pending.
            let g = b.gdn_opt(il);
            let (state_ptr, conv_ptr) = match g {
                Some((st, cv)) => (Some(b.ca_off(st)), Some(b.ca_off(cv))),
                None => (None, None),
            };
            let have_gdn = g.is_some();
            b.push(
                &format!("gdn conv blk.{blk}"),
                "recur",
                if have_gdn { "sn_gdn_conv_silu_f32" } else { "gdn_conv_silu" },
                !have_gdn,
                [((dm.qkv_width / BLOCK as usize) as u32).max(1), 1, 1],
                BLOCK,
                0,
                vec![
                    b.a(slots.qkvz),
                    b.w(&t("ssm_conv1d.weight"))?,
                    b.a(slots.conv_out),
                    match conv_ptr { Some(p) => p, None => Arg::U64(0) },
                    i(dm.qkv_width),
                    i(hp.d_conv),
                ],
            );
            b.push(
                &format!("gdn l2norm blk.{blk}"),
                "recur",
                if have_gdn { "sn_gdn_l2norm_f32" } else { "gdn_l2norm" },
                !have_gdn,
                [2 * dm.n_k_heads as u32, 1, 1],
                BLOCK,
                0,
                vec![b.a(slots.conv_out), b.a(slots.conv_out), i(dm.d_k), Arg::F32(hp.rms_eps)],
            );
            // The Phase 7 form: the delta rule and the head's gated norm in one launch. The state
            // row lives in registers, the head's outputs stay in shared memory, and the separate
            // `sn_gdn_gate_norm_f32` launch (42 per token) plus its round trip through `gdn_core`
            // are gone. The argument order is the kernel's: geometry after the buffers, and `eps`
            // last - `ProgramSpec::build` checks the count against the image.
            b.push(
                &format!("gdn step+gate_norm blk.{blk}"),
                "recur",
                if have_gdn { "sn_gdn_decode_norm_f32" } else { "gdn_decode_norm" },
                !have_gdn,
                [dm.n_v_heads as u32, 1, 1],
                crate::phase6::GDN_DECODE_NORM_BLOCK,
                0,
                vec![
                    b.a(slots.conv_out),
                    b.a(slots.ba),
                    b.w(&t("ssm_a"))?,
                    b.w(&t("ssm_dt.bias"))?,
                    match state_ptr { Some(p) => p, None => Arg::U64(0) },
                    b.w(&t("ssm_norm.weight"))?,
                    b.a_off(slots.qkvz + f32s(dm.qkv_width)),
                    b.a(slots.gdn_core),
                    i(dm.d_v),
                    i(dm.d_k),
                    i(dm.n_k_heads),
                    i(dm.n_v_heads),
                    Arg::F32(hp.rms_eps),
                ],
            );
            b.gemv(
                &format!("gemv ssm_out blk.{blk}"),
                &t("ssm_out.weight"),
                slots.gdn_core,
                slots.branch,
                dm.n_v_heads * dm.d_v,
                d,
                None,
                blk,
            )?;
        } else {
            // q (interleaved [q | gate]), k, v
            if b.fusion.multi_qkv {
                b.push(
                    &format!("gemv qkv blk.{blk}"),
                    "gemv",
                    "gemv_multi_qkv",
                    true,
                    [1, 1, 1],
                    BLOCK,
                    0,
                    vec![
                        b.w(&t("attn_q.weight"))?,
                        b.w(&t("attn_k.weight"))?,
                        b.w(&t("attn_v.weight"))?,
                        b.a(h_src),
                        b.a(slots.qg),
                        b.a(slots.k),
                        b.a(slots.v),
                        i(d),
                        i(dm.n_q * 2),
                        i(dm.n_kv),
                        if b.fusion.norm_into_gemv { b.w(&t("attn_norm.weight"))? } else { Arg::U64(0) },
                        Arg::F32(hp.rms_eps),
                    ],
                );
            } else {
                for (role, slot, rows) in
                    [("attn_q.weight", slots.qg, dm.n_q * 2), ("attn_k.weight", slots.k, dm.n_kv), ("attn_v.weight", slots.v, dm.n_kv)]
                {
                    let name = t(role);
                    let norm_name = t("attn_norm.weight");
                    b.gemv(
                        &format!("gemv {role} blk.{blk}"),
                        &name,
                        h_src,
                        slot,
                        d,
                        rows,
                        if b.fusion.norm_into_gemv { Some(norm_name.as_str()) } else { None },
                        blk,
                    )?;
                }
            }
            // Per-head Q/K norm with `attn_q_norm` / `attn_k_norm`, before rope (§2.3). The Phase 4
            // count never emitted this; the reference forward does, and a missing norm is exactly the
            // kind of silent gap the structural assertions below exist to catch.
            if b.cache.is_some() {
                b.push(
                    &format!("qk head norm blk.{blk}"),
                    "norm",
                    "sn_qk_head_norm_f32",
                    false,
                    [dm.attn_n_q_heads as u32 + dm.attn_n_kv_heads as u32, 1, 1],
                    BLOCK,
                    0,
                    vec![
                        b.a(slots.qg),
                        b.w(&t("attn_q_norm.weight"))?,
                        b.a(slots.k),
                        b.w(&t("attn_k_norm.weight"))?,
                        i(dm.attn_n_q_heads),
                        i(dm.attn_n_kv_heads),
                        i(dm.attn_d_head),
                        i(2 * dm.attn_d_head),
                        Arg::F32(hp.rms_eps),
                    ],
                );
            }
            // rope: q and k are different buffers, so two launches (16 + 4 heads).
            if !b.fusion.rope_into_gemv {
                let pairing = if hp.arch == "sensenova_loop" { 1 } else { 0 };
                for (label, slot, head_stride, token_stride, heads) in [
                    ("q", slots.qg, 2 * dm.attn_d_head, dm.n_q, dm.attn_n_q_heads),
                    ("k", slots.k, dm.attn_d_head, dm.n_kv, dm.attn_n_kv_heads),
                ] {
                    b.push(
                        &format!("rope {label} blk.{blk}"),
                        "rope",
                        "sn_rope_inplace_f32",
                        false,
                        [heads as u32, 1, 1],
                        (hp.n_rot / 2) as u32,
                        0,
                        vec![
                            b.a(slot),
                            b.a(slots.cos),
                            b.a(slots.sin),
                            b.a(slots.pos),
                            i(hp.n_rot / 2),
                            i(hp.n_rot),
                            i(head_stride),
                            u(token_stride),
                            i(pairing as usize),
                        ],
                    );
                }
            }
            let g = b.kv_opt(il);
            let have_kv = g.is_some();
            let (kv_ptr, row_bytes, kind_code) = match g {
                Some(kv) => (Some(b.ca_off(kv)), b.kv_row_bytes(), b.cache.unwrap().kind_code()),
                None => (None, 0, 0),
            };
            let values = dm.n_kv;
            let len_arg = b.len_arg();
            b.push(
                &format!("kv append blk.{blk}"),
                "recur",
                if have_kv { "sn_kv_append" } else { "kv_append_f16" },
                !have_kv,
                [1, 1, 1],
                BLOCK,
                0,
                vec![
                    b.a(slots.k),
                    b.a(slots.v),
                    match kv_ptr { Some(p) => p, None => Arg::U64(0) },
                    // The row this token writes is `*len_ptr - 1`: the meta copy above carries the
                    // length *after* this append, which is also what attention reads. Keeping the
                    // index on the device is what makes one captured graph serve every token.
                    len_arg,
                    i(values),
                    i(row_bytes),
                    i(2 * row_bytes),
                    i(kind_code),
                ],
            );
            b.push(
                &format!("attn decode blk.{blk}"),
                "recur",
                if have_kv { "sn_attn_decode_f32" } else { "attn_decode_f16" },
                !have_kv,
                [dm.attn_n_q_heads as u32, 1, 1],
                crate::phase6::ATTN_BLOCK,
                0,
                vec![
                    b.a(slots.qg),
                    match kv_ptr { Some(p) => p, None => Arg::U64(0) },
                    b.a(slots.attn_out),
                    len_arg,
                    i(dm.attn_n_q_heads),
                    i(dm.attn_n_kv_heads),
                    i(dm.attn_d_head),
                    i(2 * dm.attn_d_head),
                    i(values),
                    i(row_bytes),
                    i(2 * row_bytes),
                    i(kind_code),
                    Arg::F32(1.0 / (dm.attn_d_head as f32).sqrt()),
                ],
            );
            let (gated, out_slot) = if b.fusion.gate_into_gemv {
                (slots.attn_out, slots.attn_out)
            } else {
                b.push(
                    &format!("gate norm blk.{blk}"),
                    "gate",
                    "sn_attn_gate_norm_f32",
                    false,
                    grid1d(dm.n_q),
                    BLOCK,
                    0,
                    vec![
                        b.a(slots.attn_out),
                        b.a(slots.qg),
                        b.a(slots.attn_gated),
                        i(dm.attn_d_head),
                        i(2 * dm.attn_d_head),
                        i(dm.attn_d_head),
                        u(dm.n_q),
                        u(dm.n_q),
                        u(2 * dm.n_q),
                        u(dm.n_q),
                    ],
                );
                (slots.attn_gated, slots.attn_gated)
            };
            b.gemv(&format!("gemv attn_output blk.{blk}"), &t("attn_output.weight"), gated, slots.branch, dm.n_q, d, None, blk)?;
            let _ = out_slot;
        }

        // ---- merge, then the FFN half -------------------------------------------------
        if mhc {
            b.merge("attn", blk);
            b.ffn(blk)?;
            b.merge("ffn", blk);
        } else {
            // Outside the mHC range a branch is a plain residual add (§2.2).
            b.push(
                &format!("add residual attn blk.{blk}"),
                "add",
                "sn_add_f32",
                false,
                grid1d(d),
                BLOCK,
                0,
                vec![b.a(slots.cur), b.a(slots.branch), u(d)],
            );
            // The same fold as the mHC path below: with `norm_into_gemv` the pre-norm belongs to the
            // GEMV that reads the result, so launching it here too would apply it twice (once into
            // `slots.h`, once again inside the gate/up GEMV).
            let h2 = if b.fusion.norm_into_gemv {
                slots.cur
            } else {
                b.rmsnorm(&format!("rmsnorm post blk.{blk}"), slots.cur, &t("post_attention_norm.weight"), slots.h, blk)?;
                slots.h
            };
            b.ffn_body(blk, h2)?;
            b.push(
                &format!("add residual ffn blk.{blk}"),
                "add",
                "sn_add_f32",
                false,
                grid1d(d),
                BLOCK,
                0,
                vec![b.a(slots.cur), b.a(slots.branch), u(d)],
            );
        }

        // ---- streams out --------------------------------------------------------------
        // §2.7 exposes `cur = 0.5*(a+b)` *at the end of an mHC execution*, and the value is read by
        // whichever execution comes next. Every mHC execution except the last is followed by another
        // mHC execution, which ignores `cur` and re-derives everything from the streams, so the only
        // exposure that matters is the one at the mHC -> non-mHC boundary. For this checkpoint the
        // exit blocks (physical 28..31, stored as `blk.52..55`) are outside the range: an exposure
        // placed in the head instead would leave those blocks reading a `cur` last written at
        // execution 3, and would feed the output head the loop's streams instead of block 55's
        // output. If a trunk did end inside the range, `next_mhc` is false for its last execution
        // and this is exactly the exposure the head needs.
        if mhc && !next_mhc {
            b.expose_streams();
        }
        // The exclusive end of this execution's ops: `sn gpu stages` replays the program up to here
        // and compares the arena with the reference's `l_out.<il>` stage.
        b.exec_ends.push((il, b.ops.len()));
    }

    // ---- the head -------------------------------------------------------------------
    // `cur` is whatever the last trunk execution left there: the exposure above already ran at the
    // boundary when the mHC region does not reach the end of the trunk.
    b.rmsnorm("rmsnorm output", slots.cur, "output_norm.weight", slots.h, 0)?;
    b.gemv("gemv output head", "output.weight", slots.h, slots.head, d, hp.vocab, None, 0)?;
    // Greedy sampling, in the program: the id lands in the pinned slot the host is already holding,
    // so the decode loop never copies the vocabulary back (Phase 7's "no hipMemcpy in the loop").
    // Only emitted for a program that can actually launch it (a pinned slot exists and the kernel is
    // registered), because a counted-only program must not claim a launch it cannot make.
    if bases.pinned != 0 {
        b.push(
            "sample argmax",
            "sample",
            "sn_argmax_f32",
            false,
            [1, 1, 1],
            1024,
            0,
            vec![b.a(slots.head), i(hp.vocab), b.u_pin(bases.pinned + crate::decode::PIN_SAMPLE as u64)],
        );
    }

    Ok(ProgramSpec {
        ops: b.ops,
        inputs: b.inputs,
        fusion,
        arena_bytes,
        exec_ends: b.exec_ends,
    })
}

/// The emission state. A struct rather than closures so the borrow checker and the reader both
/// stay happy: `push` is the only place an `OpSpec` is created.
struct Builder<'a> {
    ops: Vec<OpSpec>,
    inputs: Vec<(String, u64, u64, usize)>,
    exec_ends: Vec<(usize, usize)>,
    cache: Option<&'a CacheArgs<'a>>,
    bases: Bases,
    slots: Slots,
    fusion: Fusion,
    weights: &'a dyn Fn(&str) -> Result<(u64, String)>,
    n_ffn: usize,
    n_route: usize,
    rms_eps: f32,
    mhc_iters: usize,
    mhc_pre_scale: f32,
    mhc_fixed_identity: bool,
}

impl<'a> Builder<'a> {
    fn u_pin(&self, addr: u64) -> Arg {
        Arg::Ptr(addr)
    }
    fn a(&self, off: usize) -> Arg {
        Arg::Ptr(self.bases.activations + off as u64)
    }
    fn a_off(&self, off: usize) -> Arg {
        Arg::Ptr(self.bases.activations + off as u64)
    }
    fn w(&self, name: &str) -> Result<Arg> {
        Ok(Arg::Ptr(self.bases.weights + (self.weights)(name)?.0))
    }
    fn w_off(&self, off: u64) -> Arg {
        Arg::Ptr(self.bases.weights + off)
    }
    fn input_from(&mut self, label: &str, pin_off: usize, act_off: usize, len: usize) {
        self.input_to(label, pin_off, self.bases.activations + act_off as u64, len);
    }

    fn input_to(&mut self, label: &str, pin_off: usize, dst: u64, len: usize) {
        let host = if self.bases.pinned == 0 { 0 } else { self.bases.pinned + pin_off as u64 };
        self.inputs.push((label.to_string(), dst, host, len));
    }

    /// The device address a kernel should read the sequence length from, or a null pointer for a
    /// program that was only compiled for counting. Resolved before `push`, because the argument
    /// list is an argument of `push`.
    fn len_arg(&self) -> Arg {
        match self.kv_len_ptr() {
            Ok(p) => Arg::Ptr(p),
            Err(_) => Arg::U64(0),
        }
    }

    fn ca_off(&self, off: u64) -> Arg {
        Arg::Ptr(self.bases.caches + off)
    }

    /// The KV region for this execution, or `None` for a GDN execution (which has no KV cache) and
    /// for a program compiled without caches.
    fn kv_opt(&self, exec: usize) -> Option<u64> {
        let c = self.cache?;
        if !c.layout.kv_exec.contains(&exec) {
            return None;
        }
        c.layout.kv(exec, c.seq).ok().map(|o| o as u64)
    }

    /// The GDN state and conv regions for this execution, or `None` for an attention execution.
    fn gdn_opt(&self, exec: usize) -> Option<(u64, u64)> {
        let c = self.cache?;
        if !c.layout.gdn_exec.contains(&exec) {
            return None;
        }
        Some((c.layout.state(exec, c.seq).ok()? as u64, c.layout.conv(exec, c.seq).ok()? as u64))
    }

    fn kv_row_bytes(&self) -> usize {
        self.cache.map(|c| c.layout.kv_row_bytes().unwrap_or(0)).unwrap_or(0)
    }

    fn kv_len_ptr(&self) -> Result<u64> {
        let c = match self.cache {
            Some(c) => c,
            None => return Err(Error::msg("this program was compiled without caches")),
        };
        Ok(self.bases.caches + (c.layout.meta_row(c.seq)? + META_LEN * 4) as u64)
    }
    #[allow(clippy::too_many_arguments)]
    fn push(&mut self, label: &str, class: &'static str, kernel: &str, pending: bool, grid: [u32; 3], block: u32, shared: u32, args: Vec<Arg>) {
        self.ops.push(OpSpec {
            label: label.to_string(),
            class,
            kernel: kernel.to_string(),
            pending,
            grid,
            block,
            shared,
            args,
        });
    }

    fn rmsnorm(&mut self, label: &str, x: usize, norm: &str, y: usize, blk: usize) -> Result<()> {
        let _ = blk;
        self.push(
            label,
            "norm",
            "sn_rmsnorm_f32",
            false,
            [1, 1, 1],
            BLOCK,
            0,
            vec![self.a(x), self.w(norm)?, self.a(y), i(self.slots.d), Arg::F32(self.rms_eps), i(1)],
        );
        Ok(())
    }

    /// One weight matrix, one launch. The trailing arguments are the folded-in pre-norm
    /// (`norm_w = 0` when the norm is a separate launch), its epsilon, and `warps_per_row`: the
    /// Phase 5 GEMV ABI, with the geometry the plan says is a per-shape tuning decision.
    ///
    /// Phase 5 owns `gemv_q4_k` / `gemv_q6_k` / `gemv_f32`, so a tensor of one of those three
    /// types is no longer reported pending: `ProgramSpec::build` launches it. Any other element
    /// type stays pending, which is the honest answer for a checkpoint this engine has not been
    /// built for (§1.2: this file stores Q4_K, Q6_K and f32 only).
    #[allow(clippy::too_many_arguments)]
    fn gemv(&mut self, label: &str, tensor: &str, x: usize, y: usize, k: usize, n: usize, fold_norm: Option<&str>, blk: usize) -> Result<()> {
        let _ = blk;
        let (off, ty) = (self.weights)(tensor)?;
        let norm_w = match fold_norm {
            Some(name) => self.w(name)?,
            None => Arg::U64(0),
        };
        let wpr = crate::gemm::pick_warps(n, k, crate::gemm::PICK_CUS);
        let have = matches!(ty.as_str(), "q4_k" | "q6_k" | "f32");
        self.push(
            label,
            "gemv",
            &format!("gemv_{ty}"),
            !have,
            crate::gemm::gemv_grid(n, wpr),
            crate::ops::GEMV_BLOCK,
            0,
            vec![self.w_off(off), self.a(x), self.a(y), i(k), i(n), norm_w, Arg::F32(self.rms_eps), i(wpr)],
        );
        Ok(())
    }

    fn route(&mut self, which: &str, w: &str, bias: &str, blk: usize) {
        debug_assert!(self.slots.d % 4 == 0, "sn_mhc_route_f32 reads float4");
        let _ = blk;
        // The strides are the lengths themselves because a decode step is one token (T = 1); prefill
        // (Phase 8) re-compiles with T > 1 and real token strides.
        self.push(
            &format!("route {which} blk.{blk}"),
            "route",
            "sn_mhc_route_f32",
            false,
            [1, 1, 1],
            BLOCK,
            0, // sn_mhc_route_f32's scratch is static shared - see `route_static_shared`
            vec![
                self.a(self.slots.a),
                self.a(self.slots.b),
                self.w_off((self.weights)(w).map(|o| o.0).unwrap_or(0)),
                self.w_off((self.weights)(bias).map(|o| o.0).unwrap_or(0)),
                self.a(self.slots.route_in),
                self.a(self.slots.logits),
                self.a(self.slots.post),
                self.a(self.slots.mix),
                i(self.slots.d),
                i(self.n_route),
                u(self.slots.d),
                u(self.slots.d),
                u(self.slots.d),
                u(self.n_route),
                u(2),
                u(4),
                Arg::F32(1e-6),
                Arg::F32(self.mhc_pre_scale),
                i(self.mhc_iters),
                i(if self.mhc_fixed_identity { 1 } else { 0 }),
            ],
        );
    }

    /// Seed both streams from `cur` (`a = b = cur`, §2.7). One launch writes both: this is the only
    /// point in a decode step where the streams are created, and without it the first `mhc_route`
    /// reads buffers nothing wrote.
    fn seed_streams(&mut self, blk: usize) {
        let d = self.slots.d;
        self.push(
            &format!("seed streams blk.{blk}"),
            "merge",
            "sn_mhc_seed_f32",
            false,
            grid1d(d),
            BLOCK,
            0,
            // (a, b, cur, d, sa, sb, scur): the strides are the widths themselves at T = 1.
            vec![self.a(self.slots.a), self.a(self.slots.b), self.a(self.slots.cur), i(d), u(d), u(d), u(d)],
        );
    }

    /// `cur = 0.5*(a+b)`, emitted at the mHC -> non-mHC boundary (see the call site).
    fn expose_streams(&mut self) {
        let d = self.slots.d;
        self.push(
            "mhc expose 0.5*(a+b)",
            "merge",
            "sn_mhc_expose_f32",
            false,
            grid1d(d),
            BLOCK,
            0,
            vec![self.a(self.slots.a), self.a(self.slots.b), self.a(self.slots.cur), i(d), u(d), u(d), u(d)],
        );
    }

    fn merge(&mut self, which: &str, blk: usize) {
        let fused = self.fusion.route_with_merge;
        self.push(
            &format!("merge {which} blk.{blk}"),
            "merge",
            if fused { "sn_mhc_merge_route_f32" } else { "sn_mhc_merge_f32" },
            fused,
            [grid1d(self.slots.d)[0], 1, 1],
            BLOCK,
            0, // as above: the fused route's scratch is static shared
            vec![
                self.a(self.slots.a),
                self.a(self.slots.b),
                self.a(self.slots.branch),
                self.a(self.slots.mix),
                self.a(self.slots.post),
                i(self.slots.d),
                u(self.slots.d),
                u(self.slots.d),
                u(self.slots.d),
                u(4),
                u(2),
            ],
        );
    }

    /// The FFN half inside the mHC range: route, folded/unfolded norm, gate/up, SiLU-mul, down.
    fn ffn(&mut self, blk: usize) -> Result<()> {
        let x = if self.fusion.route_with_merge {
            self.slots.route_in
        } else {
            self.route("ffn", &format!("blk.{blk}.hc_ffn_fn.weight"), &format!("blk.{blk}.hc_ffn_fn.bias"), blk);
            self.slots.route_in
        };
        let h2 = if self.fusion.norm_into_gemv {
            x
        } else {
            self.rmsnorm(
                &format!("rmsnorm post blk.{blk}"),
                x,
                &format!("blk.{blk}.post_attention_norm.weight"),
                self.slots.h,
                blk,
            )?;
            self.slots.h
        };
        self.ffn_body(blk, h2)
    }

    /// gate/up, the SwiGLU intermediate, and down. `h` is the slot holding the normalised input.
    fn ffn_body(&mut self, blk: usize, h: usize) -> Result<()> {
        let d = self.slots.d;
        let ffn = self.n_ffn;
        if self.fusion.multi_gemv {
            // Phase 7: this is no longer a name. `ffn_gate` and `ffn_up` are the one pair in this
            // checkpoint that always shares an input *and* has identical shape and type in every
            // block (`Tools/tensor_census.py`: 33 x q4_k each), which is what `sn_gemv_q4_k_pair`
            // requires. If a file ever breaks that, fall back to two single launches rather than to
            // a launch the kernel cannot serve.
            let gate = format!("blk.{blk}.ffn_gate.weight");
            let up = format!("blk.{blk}.ffn_up.weight");
            let (g1, t1) = (self.weights)(&gate)?;
            let (g2, t2) = (self.weights)(&up)?;
            if t1 == "q4_k" && t2 == "q4_k" {
                let wpr = crate::gemm::pick_warps(ffn, d, crate::gemm::PICK_CUS);
                let norm_w = if self.fusion.norm_into_gemv {
                    self.w(&format!("blk.{blk}.post_attention_norm.weight"))?
                } else {
                    Arg::U64(0)
                };
                self.push(
                    &format!("gemv ffn_gate+up blk.{blk}"),
                    "gemv",
                    "gemv_q4_k_pair",
                    false,
                    crate::gemm::gemv_grid(2 * ffn, wpr),
                    crate::ops::GEMV_BLOCK,
                    0,
                    vec![
                        self.w_off(g1),
                        self.w_off(g2),
                        self.a(h),
                        self.a(self.slots.ffn_g),
                        self.a(self.slots.ffn_u),
                        i(d),
                        i(ffn),
                        norm_w,
                        Arg::F32(1e-6),
                        i(wpr),
                    ],
                );
                let act = if self.fusion.silu_into_gemv { self.slots.ffn_g } else { self.slots.ffn_act };
                if !self.fusion.silu_into_gemv {
                    self.push(
                        &format!("silu_mul blk.{blk}"),
                        "silu",
                        "sn_silu_mul_f32",
                        false,
                        grid1d(ffn),
                        BLOCK,
                        0,
                        vec![self.a(self.slots.ffn_g), self.a(self.slots.ffn_u), self.a(act), u(ffn)],
                    );
                }
                return self.gemv(
                    &format!("gemv ffn_down blk.{blk}"),
                    &format!("blk.{blk}.ffn_down.weight"),
                    act,
                    self.slots.branch,
                    ffn,
                    d,
                    None,
                    blk,
                );
            }
            self.push(
                &format!("gemv ffn_gate_up blk.{blk}"),
                "gemv",
                "gemv_multi_gate_up",
                true,
                [1, 1, 1],
                BLOCK,
                0,
                vec![
                    self.w(&format!("blk.{blk}.ffn_gate.weight"))?,
                    self.w(&format!("blk.{blk}.ffn_up.weight"))?,
                    self.a(h),
                    self.a(self.slots.ffn_g),
                    self.a(self.slots.ffn_u),
                    i(d),
                    i(ffn),
                    if self.fusion.norm_into_gemv { self.w(&format!("blk.{blk}.post_attention_norm.weight"))? } else { Arg::U64(0) },
                    Arg::F32(1e-6),
                ],
            );
        } else {
            for (role, slot) in [("ffn_gate.weight", self.slots.ffn_g), ("ffn_up.weight", self.slots.ffn_u)] {
                let name = format!("blk.{blk}.{role}");
                let norm_name = format!("blk.{blk}.post_attention_norm.weight");
                self.gemv(
                    &format!("gemv {role} blk.{blk}"),
                    &name,
                    h,
                    slot,
                    d,
                    ffn,
                    if self.fusion.norm_into_gemv { Some(norm_name.as_str()) } else { None },
                    blk,
                )?;
            }
        }
        let act = if self.fusion.silu_into_gemv { self.slots.ffn_g } else { self.slots.ffn_act };
        if !self.fusion.silu_into_gemv {
            self.push(
                &format!("silu_mul blk.{blk}"),
                "silu",
                "sn_silu_mul_f32",
                false,
                grid1d(ffn),
                BLOCK,
                0,
                vec![self.a(self.slots.ffn_g), self.a(self.slots.ffn_u), self.a(act), u(ffn)],
            );
        }
        self.gemv(
            &format!("gemv ffn_down blk.{blk}"),
            &format!("blk.{blk}.ffn_down.weight"),
            act,
            self.slots.branch,
            ffn,
            d,
            None,
            blk,
        )
    }

    // `n_ffn` and the mHC scalars are copied into the builder from the plan, so this module never
    // needs a `Gguf` (and the op counts stay testable without a model file).
}

#[cfg(test)]
mod tests {
    use crate::ops::ROUTE_BLOCK;
    use super::*;

    #[test]
    fn fusion_presets_differ_by_exactly_the_seven_flags() {
        assert_ne!(Fusion::RAW, Fusion::FUSED);
        assert_eq!(Fusion::FUSED.flags().iter().filter(|(_, on)| *on).count(), 7);
        assert_eq!(Fusion::RAW.flags().iter().filter(|(_, on)| *on).count(), 0);
        let f = Fusion::FUSED.without(FusionFlag::Multi);
        assert!(!f.multi_gemv && f.norm_into_gemv);
    }

    #[test]
    fn counts_sort_by_size_and_add_up() {
        let mk = |class: &'static str, pending: bool| OpSpec {
            label: "x".into(),
            class,
            kernel: String::new(),
            pending,
            grid: [1, 1, 1],
            block: 1,
            shared: 0,
            args: vec![],
        };
        let spec = ProgramSpec {
            ops: vec![mk("gemv", true), mk("gemv", true), mk("route", false)],
            inputs: vec![],
            fusion: Fusion::RAW,
            arena_bytes: 0,
            exec_ends: vec![(0, 3)],
        };
        assert_eq!(spec.len(), 3);
        assert_eq!(spec.pending(), 2);
        assert_eq!(spec.launchable(), 1);
        assert_eq!(spec.counts()[0], ("gemv", 2));
    }

    /// Phase 7, part 1: the stream hand-off at the loop boundary, asserted as emitted structure.
    ///
    /// The reference seeds the mHC streams at the first execution of **every pass through the loop**
    /// (`physical == mhc.start`), and what it seeds from is what the previous pass exposed,
    /// `0.5*(a+b)`. Before this was known the second pass simply continued the first pass's streams:
    /// no fault, no NaN, logits 3e-2 off, and the first execution that diverged was invisible
    /// without `sn gpu stages`. So the invariant is asserted where it is produced.
    #[test]
    fn the_loop_re_seeds_the_streams_from_what_the_previous_pass_exposed() {
        let plan = sn_core::plan::fixture_plan().expect("fixture plan");
        let mut act = crate::arena::Arena::new("a");
        let (slots, bytes) = Slots::reserve(&mut act, &plan, 8).unwrap();
        let bases = Bases {
            activations: 0x1_0000,
            weights: 0x2_0000,
            pinned: 0x3_0000,
            ..Default::default()
        };
        let spec = compile(&plan, &slots, bases, bytes, Fusion::RAW, &|_| Ok((0, "q4_k".to_string())))
            .expect("the fixture plan compiles");

        let passes = (0..plan.n_exec())
            .filter(|&il| plan.exec(il).mtp == 0 && plan.exec(il).mhc && plan.exec(il).physical == plan.hp.mhc_start)
            .count();
        assert!(passes >= 2, "the fixture loops, so there are {passes} stream seeds to assert");
        let seed = spec.ops.iter().filter(|o| o.kernel == "sn_mhc_seed_f32").count();
        assert_eq!(seed, passes, "one seed per pass, not one per trunk");

        // Every seed after the first is preceded immediately by the exposure it reads.
        // The order the two hand-off kernels appear in, and nothing else.
        let seq: Vec<&str> = spec
            .ops
            .iter()
            .filter(|o| o.kernel == "sn_mhc_seed_f32" || o.kernel == "sn_mhc_expose_f32")
            .map(|o| o.kernel.as_str())
            .collect();
        let mut seen_seed = 0;
        for (p, k) in seq.iter().enumerate() {
            if *k == "sn_mhc_seed_f32" {
                if seen_seed > 0 {
                    assert_eq!(seq[p - 1], "sn_mhc_expose_f32", "the {seen_seed}-nd seed has no exposure before it");
                }
                seen_seed += 1;
            }
        }
        assert_eq!(seen_seed, passes);
        // The pass boundary is a *second* hand-off, distinct from the mHC -> non-mHC exposure before
        // the exit blocks; both must exist.
        let expose = spec.ops.iter().filter(|o| o.kernel == "sn_mhc_expose_f32").count();
        assert!(expose >= passes, "a re-seed exposure per pass plus the boundary exposure: got {expose}");
    }

    /// Phase 7, part 2: the two GDN launches became one. The op count is the gate's currency, and a
    /// fusion that only exists in a comment is worth nothing, so the count is asserted structurally
    /// (same style as `tests/phase5.rs`).
    #[test]
    fn the_gdn_branch_emits_one_recurrent_launch_per_execution() {
        let plan = sn_core::plan::fixture_plan().expect("fixture plan");
        let mut act = crate::arena::Arena::new("a");
        let (slots, bytes) = Slots::reserve(&mut act, &plan, 8).unwrap();
        let spec = compile(&plan, &slots, Bases::default(), bytes, Fusion::RAW, &|_| Ok((0, "q4_k".to_string())))
            .expect("the fixture plan compiles");
        let gdn_execs = (0..plan.n_exec()).filter(|&il| plan.exec(il).mtp == 0 && plan.exec(il).recurrent).count();
        let fused = spec.ops.iter().filter(|o| o.kernel == "gdn_decode_norm").count();
        assert_eq!(fused, gdn_execs, "one fused delta-rule launch per GDN execution");
        for gone in ["gdn_decode", "gdn_gate_norm"] {
            assert!(spec.ops.iter().all(|o| o.kernel != gone), "{gone} is still emitted; the fused kernel replaced it");
        }
        // And the sampler is part of the program when the program can be launched at all.
        let spec2 = compile(
            &plan,
            &slots,
            Bases { activations: 1, weights: 2, pinned: 3, ..Default::default() },
            bytes,
            Fusion::RAW,
            &|_| Ok((0, "q4_k".to_string())),
        )
        .expect("compiles with a pinned slot");
        assert_eq!(spec2.ops.iter().filter(|o| o.kernel == "sn_argmax_f32").count(), 1, "greedy sampling lives in the program");
        assert_eq!(spec.ops.iter().filter(|o| o.kernel == "sn_argmax_f32").count(), 0, "a counting-only program does not claim it");
    }

    #[test]
    fn route_shared_matches_the_kernels_shared_memory() {
        // part[8] + red[8*256] + out8[8] floats, matching kernels/phase4.hip's __shared__ blocks.
        assert_eq!(route_static_shared(), ((ROUTE_BLOCK / 32) + 8 * ROUTE_BLOCK + 8) * 4);
        // The device's default static shared limit: exceeding it would need hipFuncSetAttribute.
        assert!(route_static_shared() < 48 * 1024, "{} bytes", route_static_shared());
        assert!(route_static_shared() <= 65536, "within the 64 KB per-block shared memory here");
    }
}
