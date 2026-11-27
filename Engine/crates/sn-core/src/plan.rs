//! The execution plan: which GGUF block each logical layer reads weights from, and what kind of
//! layer it is.
//!
//! The checkpoint is a *looped* transformer. GGUF block indices are **first-visit logical**, not
//! physical: blocks `0..loop_end` are stored once, and the blocks re-executed by the loop
//! (`loop_end..loop_end+extra`) are stored under `blk.(loop_end+extra)..` instead. For this
//! checkpoint that means `blk.52..55` hold physical blocks `28..31`. So a tensor name is a
//! function of the visit, not of the physical block, and every layer index in the engine must go
//! through this module before it touches a weight.
//!
//! The three maps are:
//!
//! ```text
//!   logical il        0 .. n_layer_all                visit index, what a trace says
//!   physical(il)      0 .. n_physical                 unique block index (the topology)
//!   tensor_block(il)  GGUF block index whose weights this visit reads
//! ```
//!
//! All validation lives in [`Plan::build`], and the failure messages deliberately match the
//! reference implementation's (`llama.cpp-opensensenova`, src/models/sensenova-loop.cpp:19-36) so a
//! bad checkpoint fails the same way in both engines. See `Docs/design-plan.md` §1.1.

use crate::gguf::Gguf;
use crate::require;
use crate::{Error, Result};

/// Unique block index for a visit index. `width` is `loop_end - loop_start`, `extra` the number of
/// duplicated visits, `n_layer` the trunk length excluding the MTP block(s).
pub fn physical(
    il: usize,
    n_layer: usize,
    loop_start: usize,
    loop_end: usize,
    extra: usize,
) -> usize {
    if il >= n_layer {
        return il - extra; // MTP block
    }
    let width = loop_end - loop_start;
    if il < loop_start {
        return il;
    }
    if il < loop_end + extra {
        return loop_start + (il - loop_start) % width;
    }
    il - extra
}

/// What one execution of one block does.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Exec {
    /// Visit index (`0..n_layer_all`).
    pub logical: usize,
    /// Unique block index.
    pub physical: usize,
    /// GGUF block index whose tensors this visit reads.
    pub tensor_block: usize,
    /// mHC router + pre/post-merge runs around this execution.
    pub mhc: bool,
    /// GDN (gated delta net) instead of full attention.
    pub recurrent: bool,
    /// 0 = trunk, 1.. = MTP (multi-token prediction) block.
    pub mtp: usize,
}

/// Derived dimensions, all computed from the checkpoint rather than assumed.
#[derive(Debug, Clone, Copy)]
pub struct Dims {
    /// Width of the q/k/v projections: `head_count * key_length` and `head_count_kv * key_length`.
    /// These are widths, not head counts — the head counts are `attn_n_q_heads` / `attn_n_kv_heads`.
    pub n_q: usize,
    pub n_kv: usize,
    /// Full-attention head counts and head width (`attention.key_length`, 256 in this checkpoint).
    pub attn_n_q_heads: usize,
    pub attn_n_kv_heads: usize,
    pub attn_d_head: usize,
    /// GDN key heads = `ssm.group_count`.
    pub n_k_heads: usize,
    /// GDN value heads = `d_inner / d_state` (== `ssm.time_step_rank`).
    pub n_v_heads: usize,
    /// GDN key/value head width = `ssm.state_size` (128 here). This is *not* the attention head
    /// width, which is `attn_d_head` (256 here) — the two are easy to confuse and are different.
    pub d_k: usize,
    pub d_v: usize,
    /// Width of the fused `attn_qkv_gate` qkv part: `2 * d_inner`.
    pub qkv_width: usize,
    /// Width of the mHC stream vector: `mhc_streams * n_embd`.
    pub hc_width: usize,
    /// Rows of the mHC router logit vector: `fixed_identity ? 4 : 8`.
    pub n_route: usize,
}

#[derive(Debug, Clone)]
pub struct Hyperparams {
    pub arch: String,
    pub name: String,
    /// `general.file_type` rendered as a quant name (Q4_K_M), for reports only.
    pub quant_label: String,

    pub n_layer_all: usize,
    pub n_layer_nextn: usize,
    pub n_embd: usize,
    pub n_ffn: usize,
    pub n_head: usize,
    pub n_head_kv: usize,
    pub head_dim: usize,
    /// Rope'd dimensions per head (`rope.dimension_count`).
    pub n_rot: usize,
    pub rms_eps: f32,
    pub vocab: usize,
    pub ctx_train: usize,

    pub rope_base: f32,
    pub rope_type: String,
    pub yarn_factor: f32,
    pub ctx_orig: usize,
    /// Per-axis pair counts, e.g. `[11, 11, 10, 0]` for (height, width, temporal, text).
    pub rope_sections: Vec<u32>,

    pub d_conv: usize,
    pub d_state: usize,
    pub n_group: usize,
    pub dt_rank: usize,
    pub d_inner: usize,

    pub loop_start: usize,
    pub loop_end: usize,
    pub loop_count: usize,
    pub mhc_start: usize,
    pub mhc_end: usize,
    pub mhc_streams: usize,
    pub mhc_iters: usize,
    pub mhc_pre_scale: f32,
    pub mhc_fixed_identity: bool,

    /// Per *logical* layer: true = GDN, false = full attention (`attention.recurrent_layers`).
    pub is_recr: Vec<bool>,
    pub dims: Dims,
}

impl Hyperparams {
    /// Reads every key the engine needs. Keys that the reference implementation treats as optional
    /// (rope scaling, nextn) are defaulted the same way it does.
    pub fn from_gguf(g: &Gguf) -> Result<Hyperparams> {
        let arch = g.arch()?;
        require!(
            arch == "sensenova_loop",
            Error::msg(format!(
                "unsupported architecture '{arch}' - this engine implements sensenova_loop only"
            ))
        );
        let p = format!("{arch}.");
        let get = |k: &str| -> Result<u32> {
            g.opt_u32(&format!("{p}{k}")).ok_or_else(|| {
                Error::msg(format!("missing required metadata key {p}{k}"))
            })
        };

        let n_layer_all = get("block_count")? as usize;
        let n_layer_nextn = g
            .opt_u32(&format!("{p}nextn_predict_layers"))
            .unwrap_or(0) as usize;
        let n_embd = get("embedding_length")? as usize;
        let n_ffn = get("feed_forward_length")? as usize;
        let n_head = get("attention.head_count")? as usize;
        let n_head_kv = get("attention.head_count_kv")? as usize;
        // `unwrap_or(get(..))` would evaluate the fallback eagerly and fail on files that do have
        // the primary key, so chain Options instead.
        let head_dim = g
            .opt_u32(&format!("{p}attention.key_length"))
            .or_else(|| g.opt_u32(&format!("{p}attention.head_dim")))
            .ok_or_else(|| Error::msg(format!("missing {p}attention.key_length")))
            ? as usize;
        let rms_eps = g
            .opt_f32(&format!("{p}attention.layer_norm_rms_epsilon"))
            .unwrap_or(1e-6);
        let vocab = g
            .arr_len("tokenizer.ggml.tokens")
            .or_else(|| g.opt_u32(&format!("{p}vocab_size")).map(|v| v as usize))
            .ok_or_else(|| Error::msg("cannot determine vocabulary size"))?;
        let ctx_train = g.opt_u32(&format!("{p}context_length")).unwrap_or(0) as usize;

        let n_rot = g.opt_u32(&format!("{p}rope.dimension_count"));
        let rope_base = g.opt_f32(&format!("{p}rope.freq_base")).unwrap_or(1e4);
        let rope_type = g
            .opt_str(&format!("{p}rope.scaling.type"))
            .unwrap_or("none")
            .to_string();
        let yarn_factor = g.opt_f32(&format!("{p}rope.scaling.factor")).unwrap_or(1.0);
        let ctx_orig = g
            .opt_u32(&format!("{p}rope.scaling.original_context_length"))
            .unwrap_or(ctx_train as u32) as usize;
        let rope_sections = g
            .opt_arr_u32(&format!("{p}rope.dimension_sections"))
            .unwrap_or_default();

        let d_conv = get("ssm.conv_kernel")? as usize;
        let d_state = get("ssm.state_size")? as usize;
        let n_group = get("ssm.group_count")? as usize;
        let dt_rank = get("ssm.time_step_rank")? as usize;
        let d_inner = get("ssm.inner_size")? as usize;

        let loop_start = get("loop.start")? as usize;
        let loop_end = get("loop.end")? as usize;
        let loop_count = get("loop.count")? as usize;
        let mhc_start = get("mhc.start")? as usize;
        let mhc_end = get("mhc.end")? as usize;
        let mhc_streams = get("mhc.streams")? as usize;
        let mhc_iters = get("mhc.iterations")? as usize;
        let mhc_pre_scale = g
            .opt_f32(&format!("{p}mhc.pre_scale"))
            .unwrap_or(1.0);
        let mhc_fixed_identity = g.opt_bool(&format!("{p}mhc.fixed_identity")).unwrap_or(false);

        // attention.recurrent_layers, or derived from full_attention_interval if absent (same
        // fallback the reference uses).
        let is_recr = match g.opt_arr_u32(&format!("{p}attention.recurrent_layers")) {
            Some(arr) => {
                require!(
                    arr.len() == n_layer_all,
                    Error::msg(format!(
                        "attention.recurrent_layers has {} entries, expected {}",
                        arr.len(),
                        n_layer_all
                    ))
                );
                arr.iter().map(|v| *v != 0).collect()
            }
            None => {
                let interval = g
                    .opt_u32(&format!("{p}full_attention_interval"))
                    .unwrap_or(0) as usize;
                require!(
                    interval > 1,
                    Error::msg("no attention.recurrent_layers and no usable full_attention_interval")
                );
                (0..n_layer_all)
                    .map(|i| (i + 1) % interval != 0)
                    .collect()
            }
        };

        let n_q = n_head
            .checked_mul(head_dim)
            .ok_or_else(|| Error::msg("n_head * head_dim overflows"))?;
        let n_kv = n_head_kv
            .checked_mul(head_dim)
            .ok_or_else(|| Error::msg("n_head_kv * head_dim overflows"))?;
        require!(
            d_inner % d_state == 0,
            Error::msg(format!(
                "ssm.inner_size {} is not a multiple of ssm.state_size {}",
                d_inner, d_state
            ))
        );
        let dims = Dims {
            n_q,
            n_kv,
            attn_n_q_heads: n_head,
            attn_n_kv_heads: n_head_kv,
            attn_d_head: head_dim,
            n_k_heads: n_group,
            n_v_heads: d_inner / d_state,
            d_k: d_state,
            d_v: d_state,
            qkv_width: d_inner * 2,
            hc_width: mhc_streams * n_embd,
            n_route: if mhc_fixed_identity { 4 } else { 8 },
        };
        // `compute_dims` is the definition; this is a straight-line copy for the common path.

        let hp = Hyperparams {
            name: g.str("general.name").unwrap_or("unknown").to_string(),
            quant_label: g.file_type_name(),
            arch: arch.to_string(),
            n_layer_all,
            n_layer_nextn,
            n_embd,
            n_ffn,
            n_head,
            n_head_kv,
            head_dim,
            n_rot: n_rot.unwrap_or(head_dim as u32) as usize,
            rms_eps,
            vocab,
            ctx_train,
            rope_base,
            rope_type,
            yarn_factor,
            ctx_orig,
            rope_sections,
            d_conv,
            d_state,
            n_group,
            dt_rank,
            d_inner,
            loop_start,
            loop_end,
            loop_count,
            mhc_start,
            mhc_end,
            mhc_streams,
            mhc_iters,
            mhc_pre_scale,
            mhc_fixed_identity,
            is_recr,
            dims,
        };
        check_dims(&hp)?;
        Ok(hp)
    }
}

impl Hyperparams {
    /// Derived widths, from the declared ones. `Plan::build` calls this so a hand-edited
    /// [`Hyperparams`] can never carry stale widths into the forward.
    pub fn compute_dims(&self) -> Dims {
        Dims {
            n_q: self.n_head * self.head_dim,
            n_kv: self.n_head_kv * self.head_dim,
            attn_n_q_heads: self.n_head,
            attn_n_kv_heads: self.n_head_kv,
            attn_d_head: self.head_dim,
            n_k_heads: self.n_group,
            n_v_heads: self.d_inner / self.d_state,
            d_k: self.d_state,
            d_v: self.d_state,
            qkv_width: 2 * self.d_inner,
            hc_width: self.mhc_streams * self.n_embd,
            n_route: if self.mhc_fixed_identity { 4 } else { 8 },
        }
    }
}

/// Shape consistency checks that are not about the loop/mHC ranges.
fn check_dims(hp: &Hyperparams) -> Result<()> {
    require!(
        hp.dims.n_q == hp.n_embd,
        Error::msg(format!(
            "attention.head_count * attention.key_length ({}) != embedding_length ({})",
            hp.dims.n_q, hp.n_embd
        ))
    );
    require!(
        hp.dims.n_kv <= hp.n_embd && hp.n_embd % hp.dims.n_kv == 0,
        Error::msg("attention.key_length * head_count_kv is not a divisor of embedding_length")
    );
    require!(
        hp.dt_rank == hp.dims.n_v_heads,
        Error::msg(format!(
            "ssm.time_step_rank ({}) != ssm.inner_size / ssm.state_size ({})",
            hp.dt_rank, hp.dims.n_v_heads
        ))
    );
    require!(
        hp.n_embd % hp.n_group == 0,
        Error::msg("ssm.group_count does not divide embedding_length")
    );
    require!(
        hp.n_rot % 2 == 0 && hp.n_rot <= hp.head_dim,
        Error::msg(format!(
            "rope.dimension_count ({}) must be even and <= head_dim ({})",
            hp.n_rot, hp.head_dim
        ))
    );
    if !hp.rope_sections.is_empty() {
        let sum: u32 = hp.rope_sections.iter().sum();
        require!(
            sum as usize * 2 == hp.n_rot,
            Error::msg(format!(
                "rope.dimension_sections {:?} sums to {} pairs, expected {} for dimension_count {}",
                hp.rope_sections,
                sum,
                hp.n_rot / 2,
                hp.n_rot
            ))
        );
    }
    require!(
        hp.mhc_streams == 2,
        Error::msg("this engine implements 2-stream mHC only")
    );
    Ok(())
}

/// The validated plan. Built once at load time; every later phase reads it instead of
/// recomputing indices.
#[derive(Debug, Clone)]
pub struct Plan {
    pub hp: Hyperparams,
    /// Trunk length excluding MTP blocks.
    pub n_layer: usize,
    /// Duplicated visits introduced by the loop: `(loop_count - 1) * (loop_end - loop_start)`.
    pub extra: usize,
    /// Unique blocks, **including** the MTP block. `n_physical - n_layer_nextn` is the trunk-only
    /// figure that `Tools/roofline.py` reports.
    pub n_physical: usize,
    pub width: usize,
    /// One entry per execution, in execution order: trunk visits first (ascending logical index),
    /// then the MTP blocks.
    pub execs: Vec<Exec>,
    /// logical layer -> index into `execs`.
    logical_exec: Vec<usize>,
    /// GGUF block index -> the `execs` indices that read its weights.
    by_gguf_block: Vec<Vec<usize>>,
}

impl Plan {
    pub fn build(mut hp: Hyperparams) -> Result<Plan> {
        hp.dims = hp.compute_dims();
        validate(&hp)?;
        let n_layer = hp.n_layer_all - hp.n_layer_nextn;
        let width = hp.loop_end - hp.loop_start;
        let extra = (hp.loop_count - 1) * width;
        let n_physical = hp.n_layer_all - extra;

        let mut execs = Vec::with_capacity(hp.n_layer_all);
        let mut logical_exec = vec![usize::MAX; hp.n_layer_all];
        // Indexed by logical layer on purpose: `il` also indexes `hp.is_recr` and the diagnostic
        // messages, so the range loop reads better here than an enumeration would.
        #[allow(clippy::needless_range_loop)]
        for il in 0..n_layer {
            let ph = physical(il, n_layer, hp.loop_start, hp.loop_end, extra);
            let tb = if ph < hp.loop_end { ph } else { il };
            require!(
                ph < n_physical && tb < hp.n_layer_all,
                Error::msg(format!("internal: layer {il} maps out of range ({ph}/{tb})"))
            );
            logical_exec[il] = execs.len();
            execs.push(Exec {
                logical: il,
                physical: ph,
                tensor_block: tb,
                mhc: ph >= hp.mhc_start && ph < hp.mhc_end,
                recurrent: hp.is_recr[il],
                mtp: 0,
            });
        }
        for k in 0..hp.n_layer_nextn {
            let il = n_layer + k;
            let ph = il - extra;
            logical_exec[il] = execs.len();
            execs.push(Exec {
                logical: il,
                physical: ph,
                tensor_block: il,
                mhc: false,
                recurrent: hp.is_recr[il],
                mtp: k + 1,
            });
        }

        let mut by_gguf_block = vec![Vec::new(); hp.n_layer_all];
        for (i, e) in execs.iter().enumerate() {
            by_gguf_block[e.tensor_block].push(i);
        }

        Ok(Plan {
            hp,
            n_layer,
            extra,
            n_physical,
            width,
            execs,
            logical_exec,
            by_gguf_block,
        })
    }

    /// Number of executions that touch the trunk + MTP (i.e. weights streamed per token).
    pub fn n_exec(&self) -> usize {
        self.execs.len()
    }
    pub fn exec(&self, logical: usize) -> &Exec {
        &self.execs[self.logical_exec[logical]]
    }
    pub fn physical(&self, il: usize) -> usize {
        physical(
            il,
            self.n_layer,
            self.hp.loop_start,
            self.hp.loop_end,
            self.extra,
        )
    }
    /// GGUF block index whose weights the visit `il` reads.
    pub fn tensor_block(&self, il: usize) -> usize {
        self.exec(il).tensor_block
    }
    pub fn is_mhc(&self, il: usize) -> bool {
        self.exec(il).mhc
    }
    pub fn is_recr(&self, il: usize) -> bool {
        self.exec(il).recurrent
    }
    pub fn is_mtp(&self, il: usize) -> bool {
        self.exec(il).mtp != 0
    }
    /// How many times the mHC router runs per token (the source of the 46,080-node problem).
    pub fn n_mhc_executions(&self) -> usize {
        self.execs.iter().filter(|e| e.mhc).count()
    }
    pub fn n_gdn_executions(&self) -> usize {
        self.execs.iter().filter(|e| e.recurrent).count()
    }
    pub fn n_attn_executions(&self) -> usize {
        self.execs.iter().filter(|e| !e.recurrent).count()
    }
    /// Executions that read `blk.<gguf_block>.*`, in execution order.
    pub fn execs_of_gguf_block(&self, b: usize) -> &[usize] {
        self.by_gguf_block.get(b).map(|v| v.as_slice()).unwrap_or(&[])
    }
    /// Executions that read `blk.<gguf_block>.*` as a count (`0` for blocks whose weights are
    /// never streamed, e.g. none here, and for MTP blocks during trunk decode).
    pub fn exec_count_of_gguf_block(&self, b: usize) -> usize {
        self.execs_of_gguf_block(b).len()
    }
    /// Tensor name for a role, e.g. `block_tensor(blk, "attn_q", "weight")`.
    pub fn tensor_name(&self, gguf_block: usize, role: &str, kind: &str) -> String {
        format!("blk.{gguf_block}.{role}.{kind}")
    }
    pub fn token_embd(&self) -> String {
        "token_embd.weight".into()
    }
    pub fn output(&self) -> String {
        "output.weight".into()
    }
}

impl std::ops::Index<usize> for Plan {
    type Output = Exec;
    fn index(&self, logical: usize) -> &Exec {
        self.exec(logical)
    }
}

/// Every range check, with the reference implementation's messages.
fn validate(hp: &Hyperparams) -> Result<()> {
    require!(
        hp.n_layer_all > hp.n_layer_nextn,
        Error::msg("invalid layer count")
    );
    let n_layer = hp.n_layer_all - hp.n_layer_nextn;
    require!(
        hp.n_layer_nextn <= 1,
        Error::msg("SenseNova Loop MTP: only one MTP block is supported")
    );
    require!(
        hp.is_recr.len() == hp.n_layer_all,
        Error::msg("recurrent layer mask length mismatch")
    );
    require!(
        hp.n_layer_nextn == 0 || !hp.is_recr[n_layer],
        Error::msg("SenseNova Loop MTP requires a full-attention layer")
    );
    require!(
        hp.loop_count != 0 && hp.loop_start < hp.loop_end && hp.loop_end <= n_layer,
        Error::msg("invalid SenseNova Loop layer range or count")
    );
    let width = hp.loop_end - hp.loop_start;
    let extra = (hp.loop_count - 1) * width;
    require!(
        extra <= n_layer - hp.loop_end,
        Error::msg("SenseNova Loop execution plan exceeds the block count")
    );
    let n_physical = hp.n_layer_all - extra;
    require!(
        hp.mhc_start < hp.mhc_end
            && hp.mhc_end <= n_physical
            && hp.mhc_streams == 2
            && hp.mhc_iters != 0
            && hp.mhc_iters <= 20
            && hp.mhc_pre_scale.is_finite()
            && hp.mhc_pre_scale > 0.0,
        Error::msg("unsupported SenseNova mHC configuration")
    );
    require!(
        hp.mhc_end <= hp.loop_start
            || (hp.mhc_start >= hp.loop_start && hp.mhc_end <= hp.loop_end)
            || hp.mhc_start >= hp.loop_end,
        Error::msg("SenseNova mHC range crosses a Loop boundary")
    );
    // Every visit of a physical block must agree with the first visit about attention type -
    // the reference implementation checks the same thing ("SenseNova Loop visits have different
    // attention types") because it decides per visit which builder to call.
    for il in 0..n_layer {
        let ph = physical(il, n_layer, hp.loop_start, hp.loop_end, extra);
        require!(
            ph == il || hp.is_recr[il] == hp.is_recr[ph],
            Error::msg("SenseNova Loop visits have different attention types")
        );
    }
    Ok(())
}

/// The checkpoint under development, expressed without a GGUF on disk. The same fixture
/// `Engine/crates/sn-core/tests/plan_gates.rs` uses (values from SenseNova-Loop-9B-Q4_K_M.gguf, §1), and
/// `sn-hip`'s Phase 6 tests build cache layouts from it: a unit test should not need a 5.8 GB file
/// to find out whether the index arithmetic is right.
pub fn fixture_hyperparams() -> Hyperparams {
    let n_layer_all = 57;
    let is_recr = (0..n_layer_all).map(|i| (i + 1) % 4 != 0 && i != 56).collect();
    let mut hp = Hyperparams {
        arch: "sensenova_loop".into(),
        name: "fixture".into(),
        quant_label: "Q4_K_M".into(),
        n_layer_all,
        n_layer_nextn: 1,
        n_embd: 4096,
        n_ffn: 12288,
        n_head: 16,
        n_head_kv: 4,
        head_dim: 256,
        n_rot: 64,
        rms_eps: 1e-6,
        vocab: 248320,
        ctx_train: 524288,
        rope_base: 1e7,
        rope_type: "yarn".into(),
        yarn_factor: 2.0,
        ctx_orig: 262144,
        rope_sections: vec![11, 11, 10, 0],
        d_conv: 4,
        d_state: 128,
        n_group: 16,
        dt_rank: 32,
        d_inner: 4096,
        loop_start: 4,
        loop_end: 28,
        loop_count: 2,
        mhc_start: 4,
        mhc_end: 28,
        mhc_streams: 2,
        mhc_iters: 20,
        mhc_pre_scale: 1.0,
        mhc_fixed_identity: false,
        is_recr,
        dims: Dims {
            n_q: 0,
            n_kv: 0,
            attn_n_q_heads: 0,
            attn_n_kv_heads: 0,
            attn_d_head: 0,
            n_k_heads: 0,
            n_v_heads: 0,
            d_k: 0,
            d_v: 0,
            qkv_width: 0,
            hc_width: 0,
            n_route: 0,
        },
    };
    hp.dims = hp.compute_dims();
    hp
}

pub fn fixture_plan() -> Result<Plan> {
    Plan::build(fixture_hyperparams())
}
