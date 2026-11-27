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
    /// Weight matrices that share one input launched together: `q`+`k`+`v`, `ffn_gate`+`ffn_up`.
    pub multi_gemv: bool,
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
    };
    /// The §6 P2 target.
    pub const FUSED: Fusion = Fusion {
        route_with_merge: true,
        norm_into_gemv: true,
        silu_into_gemv: true,
        gate_into_gemv: true,
        rope_into_gemv: true,
        multi_gemv: true,
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
        }
        f
    }

    pub fn flags(&self) -> [(&'static str, bool); 6] {
        [
            ("route+merge", self.route_with_merge),
            ("norm->gemv", self.norm_into_gemv),
            ("silu->gemv", self.silu_into_gemv),
            ("rope->gemv", self.rope_into_gemv),
            ("gate->gemv", self.gate_into_gemv),
            ("multi-gemv", self.multi_gemv),
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
    /// position, launch one graph").
    pub inputs: Vec<(String, u64, usize)>,
    pub fusion: Fusion,
    /// Bytes of activation arena the program addresses, i.e. the arena `Slots` reserved.
    pub arena_bytes: usize,
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

    /// Append the launchable ops to a [`ProgramBuilder`]. Pending kernels are skipped unless
    /// `strict`, which is what Phase 7 will use to prove the program is complete.
    pub fn build<'b>(&self, reg: &KernelRegistry, b: &mut ProgramBuilder<'b>, strict: bool) -> Result<usize> {
        let mut n = 0;
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
#[derive(Clone, Copy, Debug)]
pub struct Bases {
    pub activations: u64,
    pub weights: u64,
}

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
    let hp = &plan.hp;
    let dm = &hp.dims;
    let d = hp.n_embd;
    let mut b = Builder {
        ops: Vec::new(),
        inputs: Vec::new(),
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
    b.input("input:pos", slots.pos, 4);
    b.input("input:token", slots.token, 4);

    // ---- embedding ------------------------------------------------------------------
    // One row of a quantised table, gathered rather than multiplied against a one-hot vector.
    let (emb_off, emb_ty) = weights("token_embd.weight")?;
    b.push(
        "embd row",
        "input",
        &format!("row_gather_{emb_ty}"),
        true,
        grid1d(d),
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
        if mhc && !prev_mhc {
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
            b.gemv(
                &format!("gemv beta_alpha blk.{blk}"),
                &t("ssm_beta_alpha.weight"),
                h_src,
                slots.ba,
                d,
                2 * dm.n_v_heads,
                None,
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
            b.push(
                &format!("gdn conv blk.{blk}"),
                "recur",
                "gdn_conv_silu",
                true,
                grid1d(dm.qkv_width),
                BLOCK,
                0,
                vec![b.a(slots.qkvz), b.w(&t("ssm_conv1d.weight"))?, b.a(slots.conv_out), i(dm.qkv_width), i(hp.d_conv)],
            );
            b.push(
                &format!("gdn l2norm blk.{blk}"),
                "recur",
                "gdn_l2norm",
                true,
                [dm.n_k_heads as u32, 1, 1],
                dm.d_k as u32,
                0,
                vec![b.a(slots.conv_out), b.a(slots.conv_out), i(dm.n_k_heads), i(dm.d_k), Arg::F32(hp.rms_eps)],
            );
            b.push(
                &format!("gdn step blk.{blk}"),
                "recur",
                "gdn_decode",
                true,
                [dm.n_v_heads as u32, 1, 1],
                BLOCK,
                0,
                vec![b.a(slots.conv_out), b.a(slots.ba), b.a(slots.gdn_core), Arg::U64(0), i(dm.n_v_heads), i(dm.d_v), i(dm.d_k)],
            );
            // out = (rms(core) * ssm_norm) * silu(z): one fused gated norm (§2.4).
            b.push(
                &format!("gdn gate_norm blk.{blk}"),
                "gate",
                "gdn_gate_norm",
                true,
                grid1d(dm.n_v_heads * dm.d_v),
                BLOCK,
                0,
                vec![
                    b.a(slots.gdn_core),
                    b.w(&t("ssm_norm.weight"))?,
                    b.a_off(slots.qkvz + f32s(dm.qkv_width)),
                    b.a(slots.gdn_core),
                    i(dm.n_v_heads),
                    i(dm.d_v),
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
            if b.fusion.multi_gemv {
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
            b.push(
                &format!("kv append blk.{blk}"),
                "recur",
                "kv_append_f16",
                true,
                grid1d(dm.n_kv),
                BLOCK,
                0,
                vec![b.a(slots.k), b.a(slots.v), Arg::U64(0), i(dm.n_kv), i(dm.attn_d_head)],
            );
            b.push(
                &format!("attn decode blk.{blk}"),
                "recur",
                "attn_decode_f16",
                true,
                [dm.attn_n_q_heads as u32, 1, 1],
                BLOCK,
                0,
                vec![
                    b.a(slots.qg),
                    Arg::U64(0),
                    b.a(slots.attn_out),
                    i(dm.attn_n_q_heads),
                    i(dm.attn_n_kv_heads),
                    i(dm.attn_d_head),
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
    }

    // ---- the head -------------------------------------------------------------------
    // `cur` is whatever the last trunk execution left there: the exposure above already ran at the
    // boundary when the mHC region does not reach the end of the trunk.
    b.rmsnorm("rmsnorm output", slots.cur, "output_norm.weight", slots.h, 0)?;
    b.gemv("gemv output head", "output.weight", slots.h, slots.head, d, hp.vocab, None, 0)?;

    Ok(ProgramSpec {
        ops: b.ops,
        inputs: b.inputs,
        fusion,
        arena_bytes,
    })
}

/// The emission state. A struct rather than closures so the borrow checker and the reader both
/// stay happy: `push` is the only place an `OpSpec` is created.
struct Builder<'a> {
    ops: Vec<OpSpec>,
    inputs: Vec<(String, u64, usize)>,
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
    fn input(&mut self, label: &str, off: usize, len: usize) {
        self.inputs.push((label.to_string(), self.bases.activations + off as u64, len));
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

    /// One weight matrix, one launch. The trailing two arguments are the folded-in pre-norm
    /// (`norm_w = 0` when the norm is a separate launch) and its epsilon: the Phase 5 GEMV ABI.
    #[allow(clippy::too_many_arguments)]
    fn gemv(&mut self, label: &str, tensor: &str, x: usize, y: usize, k: usize, n: usize, fold_norm: Option<&str>, blk: usize) -> Result<()> {
        let _ = blk;
        let (off, ty) = (self.weights)(tensor)?;
        let norm_w = match fold_norm {
            Some(name) => self.w(name)?,
            None => Arg::U64(0),
        };
        self.push(
            label,
            "gemv",
            &format!("gemv_{ty}"),
            true,
            [1, 1, 1],
            BLOCK,
            0,
            vec![self.w_off(off), self.a(x), self.a(y), i(k), i(n), norm_w, Arg::F32(self.rms_eps)],
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
    fn fusion_presets_differ_by_exactly_the_six_flags() {
        assert_ne!(Fusion::RAW, Fusion::FUSED);
        assert_eq!(Fusion::FUSED.flags().iter().filter(|(_, on)| *on).count(), 6);
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
        };
        assert_eq!(spec.len(), 3);
        assert_eq!(spec.pending(), 2);
        assert_eq!(spec.launchable(), 1);
        assert_eq!(spec.counts()[0], ("gemv", 2));
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
