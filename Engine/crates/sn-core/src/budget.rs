//! Weight-traffic budget: how many bytes each tensor contributes **per decoded token**.
//!
//! The stored size of a GGUF is not the cost of a token. A block that the loop re-executes streams
//! its weights again on every visit, so the real figure is `stored_bytes * visits`, summed over
//! visits. This module computes that from [`Plan`] plus the exact tensor sizes from [`Gguf`], and is
//! the single source of truth for both `sn dump --plan` and the design plan's roofline arithmetic
//! (`Tools/roofline.py` implements the same algorithm in Python; the Phase 1 gate requires the two
//! to agree byte-for-byte on every tensor).
//!
//! See `Docs/design-plan.md` §3 and §4.1.

use std::collections::BTreeMap;

use crate::gguf::{Gguf, TensorInfo};
use crate::plan::Plan;
use crate::require;
use crate::{Error, Result};

/// One tensor, as the budget sees it.
#[derive(Debug, Clone)]
pub struct WeightRow {
    pub name: String,
    /// `blk.<n>.role.kind` for block tensors, otherwise the name itself.
    pub class: String,
    pub dims: [u64; 4],
    pub n_dims: usize,
    pub type_name: String,
    pub type_id: u32,
    pub n_elts: u64,
    /// Bytes the checkpoint stores for this tensor (from the element type; validated against the
    /// offset gap at open time).
    pub stored_bytes: u64,
    /// GGUF block index, `usize::MAX` for non-block tensors (token_embd, output, ...).
    pub gguf_block: usize,
    /// How many times this tensor is streamed per decoded token.
    pub visits: usize,
    pub bytes_per_token: u64,
}

/// Aggregated per class (`blk.N.attn_qkv_gate.weight` etc.), in the layout `Tools/roofline.py`
/// prints.
#[derive(Debug, Clone)]
pub struct ClassRow {
    pub class: String,
    pub n_tensors: usize,
    pub stored_bytes: u64,
    pub visits: usize,
    pub bytes_per_token: u64,
    /// The GGUF blocks this class lives in (for cross-checking the loop expansion).
    pub blocks: Vec<usize>,
}

#[derive(Debug, Default, Clone, Copy)]
pub struct Totals {
    pub n_tensors: usize,
    pub stored_bytes: u64,
    pub bytes_per_token: u64,
    /// `bytes_per_token / stored_bytes` - the price of the loop, 1.0 for a plain transformer.
    pub re_read_factor: f64,
    /// Bytes of activations + caches that a token touches but that are not weights (reported by the
    /// caller, kept here so the CLI can print one coherent table).
    pub kv_bytes_per_token: u64,
}

#[derive(Debug, Clone)]
pub struct Budget {
    pub rows: Vec<WeightRow>,
    pub classes: Vec<ClassRow>,
    pub totals: Totals,
}

/// Normalise a tensor name into its class: `blk.23.attn_q.weight` -> `blk.N.attn_q.weight`, then
/// any remaining digits -> `#` (so `ssm_conv1d` and per-head variants fold together). Same rule as
/// `Tools/roofline.py`.
pub fn tensor_class(name: &str) -> (Option<usize>, String) {
    let mut block = None;
    let mut cls = name.to_string();
    if let Some(rest) = name.strip_prefix("blk.") {
        if let Some(dot) = rest.find('.') {
            if let Ok(b) = rest[..dot].parse::<usize>() {
                block = Some(b);
                cls = format!("blk.N.{}", &rest[dot + 1..]);
            }
        }
    }
    // Digit-fold the remainder (conv kernel indices, MTP sub-indices, ...).
    let folded: String = cls
        .chars()
        .map(|c| if c.is_ascii_digit() { '#' } else { c })
        .collect();
    (block, folded)
}

/// Build the budget. Every tensor in the file must be attributable to a visit or be a known
/// trunk-independent tensor (`token_embd`, `output`, `output_norm`, MTP block weights); anything
/// else is an error, because a silently-unattributed tensor is a silently wrong roofline.
pub fn build(plan: &Plan, g: &Gguf) -> Result<Budget> {
    let mut rows: Vec<WeightRow> = Vec::with_capacity(g.tensors().len());
    let mut unattributed = Vec::new();

    for t in g.tensors() {
        let (block, class) = tensor_class(&t.name);
        let visits = match block {
            Some(b) => {
                // A block index above the trunk count belongs to the MTP block, which the trunk
                // does not stream; it is executed once per speculated token instead.
                if b >= plan.n_layer {
                    if b < plan.hp.n_layer_all {
                        0
                    } else {
                        unattributed.push(t.name.clone());
                        continue;
                    }
                } else {
                    plan.exec_count_of_gguf_block(b)
                }
            }
            None => {
                if t.name == "token_embd.weight" || t.name == "output.weight" || t.name == "output_norm.weight" {
                    1
                } else {
                    unattributed.push(t.name.clone());
                    continue;
                }
            }
        };
        rows.push(WeightRow {
            name: t.name.clone(),
            class,
            dims: t.ne,
            n_dims: t.n_dims as usize,
            type_name: crate::qtypes::name(t.ty).to_string(),
            type_id: t.ty,
            n_elts: t.n_elts(),
            stored_bytes: t.size,
            gguf_block: block.unwrap_or(usize::MAX),
            visits,
            bytes_per_token: t.size * visits as u64,
        });
    }

    if !unattributed.is_empty() {
        return Err(Error::msg(format!(
            "{} tensor(s) not attributable to any execution: {}",
            unattributed.len(),
            unattributed[..unattributed.len().min(5)].join(", ")
        )));
    }

    // Cross-check: every block tensor that exists must be reachable from the plan, and every
    // execution must find the tensors it needs (missing-tensor audit).
    audit(plan, g)?;

    let mut by_class: BTreeMap<(String, usize), ClassRow> = BTreeMap::new();
    let mut stored = 0u64;
    let mut per_token = 0u64;
    for r in &rows {
        stored += r.stored_bytes;
        per_token += r.bytes_per_token;
        // Keyed by (class, visits) rather than class alone: the same role lives in blocks with
        // different visit counts in one model (an mHC/loop block and a pre-loop block, or an MTP
        // block that the trunk never streams). `Tools/roofline.py` keys the same way.
        let e = by_class
            .entry((r.class.clone(), r.visits))
            .or_insert_with(|| ClassRow {
                class: r.class.clone(),
                n_tensors: 0,
                stored_bytes: 0,
                visits: r.visits,
                bytes_per_token: 0,
                blocks: Vec::new(),
            });
        e.n_tensors += 1;
        e.stored_bytes += r.stored_bytes;
        e.bytes_per_token += r.bytes_per_token;
        if r.gguf_block != usize::MAX && !e.blocks.contains(&r.gguf_block) {
            e.blocks.push(r.gguf_block);
        }
    }

    let mut classes: Vec<ClassRow> = by_class.into_values().collect();
    classes.sort_by(|a, b| {
        b.bytes_per_token
            .cmp(&a.bytes_per_token)
            .then_with(|| a.class.cmp(&b.class))
    });

    let n_tensors = rows.len();
    Ok(Budget {
        rows,
        classes,
        totals: Totals {
            n_tensors,
            stored_bytes: stored,
            bytes_per_token: per_token,
            re_read_factor: per_token as f64 / stored as f64,
            kv_bytes_per_token: 0,
        },
    })
}

/// Check that the plan's executions all have the weights they will ask for, and report the block
/// inventory. Cheap, and it turns a truncated or partially requantised checkpoint into a load-time
/// error instead of a NaN at inference time.
pub fn audit(plan: &Plan, g: &Gguf) -> Result<()> {
    let hp = &plan.hp;
    let has = |b: usize, role: &str| g.has_tensor(&format!("blk.{b}.{role}.weight"));
    let has_or = |b: usize, roles: &[&str]| roles.iter().any(|r| has(b, r));

    for e in &plan.execs {
        let b = e.tensor_block;
        let want_norms = ["attn_norm", "post_attention_norm"];
        for n in want_norms {
            require!(
                has(b, n),
                Error::msg(format!("exec {:?} missing blk.{b}.{n}", e))
            );
        }
        require!(
            has_or(b, &["ffn_gate", "ffn_up", "ffn_down"]),
            Error::msg(format!("exec {:?} missing FFN weights at blk.{b}", e))
        );
        if e.recurrent {
            require!(
                has(b, "attn_qkv_gate")
                    && has(b, "ssm_out")
                    && has(b, "ssm_conv1d")
                    && has(b, "ssm_beta_alpha")
                    && has(b, "ssm_norm"),
                Error::msg(format!("exec {:?} missing GDN weights at blk.{b}", e))
            );
        } else {
            require!(
                has(b, "attn_q") && has(b, "attn_k") && has(b, "attn_v") && has(b, "attn_output"),
                Error::msg(format!("exec {:?} missing attention weights at blk.{b}", e))
            );
        }
        if e.mhc {
            require!(
                has(b, "hc_attn_fn") && has(b, "hc_ffn_fn"),
                Error::msg(format!("exec {:?} is mHC but has no router at blk.{b}", e))
            );
        }
        if e.mtp != 0 {
            require!(
                g.has_tensor(&format!("blk.{b}.nextn.eh_proj.weight")),
                Error::msg(format!("MTP execution {:?} has no eh_proj at blk.{b}", e))
            );
        }
    }
    for name in ["token_embd.weight", "output.weight", "output_norm.weight"] {
        require!(g.has_tensor(name), Error::msg(format!("checkpoint has no {name}")));
    }
    let _ = (hp.n_layer_all, hp.loop_end);
    Ok(())
}

/// KV-cache bytes added per token of context.
///
/// One KV buffer per **full-attention execution**, not per unique block: two visits of the same
/// physical block project *different* hidden states, so their K/V are different tensors and cannot
/// be shared. `Docs/design-plan.md` §3 counts them the same way, and the Phase 0 slope agrees: the
/// reference implementation's VRAM grows 58.5 KB/token between `n_ctx` 1024 and 8192, which is this
/// figure (14 full-attention visits x 2 x 1024 x f16 = 57,344 B), not the unique-block one.
pub fn kv_bytes_per_token(plan: &Plan, kv_bytes_per_elem: usize) -> u64 {
    n_kv_layers(plan) as u64 * 2 * plan.hp.dims.n_kv as u64 * kv_bytes_per_elem as u64
}

/// Full-attention caches per token (the number the memory table uses).
pub fn n_kv_layers(plan: &Plan) -> usize {
    (0..plan.n_layer)
        .filter(|&il| !plan.is_recr(il))
        .count()
}

/// The same figure if caches were shared between visits of one block — a diagnostic only; sharing
/// them would be *wrong*, since the visits hold different states.
pub fn kv_bytes_per_token_unique(plan: &Plan, kv_bytes_per_elem: usize) -> u64 {
    let n = (0..plan.n_layer)
        .filter(|&il| !plan.is_recr(il))
        .map(|il| plan.exec(il).tensor_block)
        .collect::<std::collections::HashSet<usize>>()
        .len();
    n as u64 * 2 * plan.hp.dims.n_kv as u64 * kv_bytes_per_elem as u64
}

/// GDN recurrent-state bytes per sequence: the delta-rule state (`d_inner * d_state` f32) plus the
/// causal conv window over the *qkv* projection ((`d_conv`-1) x `2*d_inner` channels), per GDN
/// **execution** - same reasoning as the KV cache: a looped block's second visit carries its own
/// state. Matches the reference allocation and `Docs/design-plan.md` §3.
pub fn gdn_state_bytes(plan: &Plan) -> u64 {
    gdn_state_per_layer(plan) * plan.n_gdn_executions() as u64
}

/// State bytes for one GDN layer.
pub fn gdn_state_per_layer(plan: &Plan) -> u64 {
    let hp = &plan.hp;
    hp.d_inner as u64 * hp.d_state as u64 * 4 + (hp.d_conv as u64 - 1) * hp.dims.qkv_width as u64 * 4
}

/// The same figure counting unique blocks only (diagnostic).
pub fn gdn_state_bytes_unique(plan: &Plan) -> u64 {
    let n = (0..plan.n_layer)
        .filter(|&il| plan.is_recr(il))
        .map(|il| plan.exec(il).tensor_block)
        .collect::<std::collections::HashSet<usize>>()
        .len();
    gdn_state_per_layer(plan) * n as u64
}

impl WeightRow {
    pub fn as_mat(&self) -> Option<(u64, u64)> {
        if self.n_dims == 2 {
            Some((self.dims[0], self.dims[1]))
        } else {
            None
        }
    }
}

/// Convenience for the CLI: look up one tensor row.
impl Budget {
    pub fn row(&self, name: &str) -> Option<&WeightRow> {
        self.rows.iter().find(|r| r.name == name)
    }
    pub fn stored_total(&self) -> u64 {
        self.totals.stored_bytes
    }
}

/// Re-exported so callers can name the type without importing gguf.
pub type Tensor = TensorInfo;
