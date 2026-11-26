//! The scalar f32 reference forward: Docs/design-plan.md §2 as running code.
//!
//! Deliberately slow and obvious - this is what every later phase is measured against (the golden
//! corpus, the `--iters 6` sensitivity check, and eventually the HIP kernels). Weights are
//! dequantised row by row as they are used and never materialised; caches are kept per *execution*,
//! because two visits of one physical block hold different states.

use std::fs::File;
use std::io::Write;

use crate::dq::{f16_round, Mat};
use crate::error::{Error, Result};
use crate::gguf::Gguf;
use crate::num::{
    apply_rope, l2_norm, rms_norm, rms_norm_w, silu, sigmoid, softplus, sinkhorn_sweeps, Rope,
};
use crate::plan::{Exec, Plan};
use crate::qtypes;
use crate::require;

/// Tensor name for a role inside a GGUF block. The role carries its own suffix because the
/// checkpoint is not uniform: `attn_q.weight`, `ssm_dt.bias`, and a bare `ssm_a`.
fn tname(block: usize, role: &str) -> String {
    format!("blk.{block}.{role}")
}

/// Stage dumps, one text file per stage in the dump directory. `Tools/golden-check.py` reads this
/// format: a `#` header line, then one value per line.
///
/// `allow` is a prefix filter. Dumping everything costs more than the forward itself (a GDN state is
/// 524 288 floats of text per token), so the useful mode is one stage at a time, e.g.
/// `--dump-only h_final` to compare hidden states against the reference's `/embedding` output.
pub struct Dumper {
    dir: Option<std::path::PathBuf>,
    allow: Vec<String>,
}

impl Dumper {
    pub fn new(dir: Option<std::path::PathBuf>) -> Result<Dumper> {
        Dumper::with_filter(dir, &[])
    }

    pub fn with_filter(dir: Option<std::path::PathBuf>, allow: &[String]) -> Result<Dumper> {
        if let Some(d) = &dir {
            std::fs::create_dir_all(d)?;
        }
        Ok(Dumper { dir, allow: allow.to_vec() })
    }

    pub fn none() -> Result<Dumper> {
        Dumper::new(None)
    }

    pub fn dumping(&self, stage: &str) -> bool {
        self.dir.is_some() && (self.allow.is_empty() || self.allow.iter().any(|p| stage.starts_with(p)))
    }

    pub fn dump(&self, stage: &str, shape: &str, v: &[f32]) -> Result<()> {
        if !self.dumping(stage) {
            return Ok(());
        }
        let dir = self.dir.as_ref().unwrap();
        let name = stage.replace('/', "_");
        let mut f = File::create(dir.join(format!("{name}.txt")))?;
        writeln!(f, "# stage {stage} shape {shape} n {}", v.len())?;
        for x in v {
            writeln!(f, "{x:.9e}")?;
        }
        Ok(())
    }
}

/// The eight mHC router outputs plus the derived `input`/`post`/`mix` factors.
pub struct Route {
    pub logits: Vec<f32>,
    pub input: Vec<f32>,
    pub post: [f32; 2],
    pub mix: [f32; 4],
}

/// Everything the reference reads from the checkpoint, plus the caches it carries across tokens.
pub struct RefModel<'a> {
    plan: &'a Plan,
    g: &'a Gguf,
    rope: Rope,
    /// KV caches by execution: `[token][kv_head*d_k + i]`.
    kv: Vec<(Vec<f32>, Vec<f32>)>,
    /// GDN states by execution: `[head][v][k]`, i.e. the reference's transposed layout, in which the
    /// row a dot needs is contiguous.
    state: Vec<Vec<f32>>,
    /// Conv windows by execution: `[t][channel]`, oldest first, `d_conv - 1` rows.
    conv: Vec<Vec<f32>>,
    threads: usize,
    /// `mhc.iterations` override for the sensitivity check; `None` = the checkpoint's value.
    iters: Option<usize>,
    limit_exec: Option<usize>,
    /// Diagnostic: round the weight / activation operands of every projection to f16 (the operand
    /// precision the reference's MMA kernels use). See `set_f16_operands`.
    f16w: bool,
    f16a: bool,
    /// Store the attention K/V cache in f16 (rounded on write) as the reference's default
    /// `cache_type_k/v` does. See `set_kv_f16`.
    kv_f16: bool,
}

impl<'a> RefModel<'a> {
    pub fn new(plan: &'a Plan, g: &'a Gguf) -> Result<RefModel<'a>> {
        let hp = &plan.hp;
        if hp.rope_type != "yarn" && hp.rope_type != "none" {
            return Err(Error::msg(format!(
                "the reference implements rope.type yarn/none (text-only); found {:?}",
                hp.rope_type
            )));
        }
        let rope = Rope::new(
            hp.n_rot,
            hp.rope_base,
            &hp.rope_sections,
            hp.yarn_factor,
            hp.ctx_orig,
        );
        let n = plan.n_exec();
        let d = &hp.dims;
        Ok(RefModel {
            plan,
            g,
            rope,
            kv: vec![(Vec::new(), Vec::new()); n],
            state: vec![vec![0.0f32; d.n_v_heads * d.d_v * hp.d_state]; n],
            conv: vec![vec![0.0f32; (hp.d_conv - 1) * d.qkv_width]; n],
            threads: std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(1),
            iters: None,
            limit_exec: None,
            f16w: false,
            f16a: false,
            kv_f16: false,
        })
    }

    pub fn set_threads(&mut self, n: usize) {
        self.threads = n.max(1);
    }

    /// Override `mhc.iterations` (Phase 2 gate: `--iters 6` must be visibly different).
    pub fn set_mhc_iters(&mut self, n: Option<usize>) {
        self.iters = n;
    }

    /// Debug aid: run only the first `n` executions of each token, to bisect a numerical problem
    /// without paying for the whole 57-execution forward.
    pub fn set_limit_exec(&mut self, n: Option<usize>) {
        self.limit_exec = n;
    }

    /// Round the operands of every projection to f16. The reference dequantises into f16 tiles for
    /// its MMA units, so an f32-only forward cannot be expected to match it to better than that
    /// operand error; this switch says which of the two the residual disagreement is.
    pub fn set_f16_operands(&mut self, weights: bool, acts: bool) {
        self.f16w = weights;
        self.f16a = acts;
    }

    /// Write the K/V cache through an f16 round-trip. Not part of §2's math - the plan keeps all
    /// accumulations in f32 - but the reference's KV cache is f16 by default (that is the type the
    /// §3 VRAM budget is priced in), and its rounding is the one operand-precision difference that
    /// grows with the number of cached tokens. `Tools/golden-check.py` measures both ways.
    pub fn set_kv_f16(&mut self, on: bool) {
        self.kv_f16 = on;
    }

    pub fn clear(&mut self) {
        for (k, v) in self.kv.iter_mut() {
            k.clear();
            v.clear();
        }
        for s in self.state.iter_mut() {
            s.fill(0.0);
        }
        for c in self.conv.iter_mut() {
            c.fill(0.0);
        }
    }

    fn mat(&self, name: &str) -> Result<Mat<'a>> {
        Mat::by_name(self.g, name)
    }

    /// `y = W @ x`, threaded for the wide projections.
    fn proj(&self, m: &Mat, x: &[f32], y: &mut [f32]) {
        assert_eq!(m.ne0, x.len(), "{}: {} inputs, expected {}", m.name, x.len(), m.ne0);
        assert_eq!(m.ne1, y.len(), "{}: {} outputs, expected {}", m.name, y.len(), m.ne1);
        if self.f16w || self.f16a {
            // Diagnostic path: materialise each row and round the operands, the way the reference's
            // kernels do. Slower (it writes 4 bytes per weight), and only used for measurement.
            let xr: Vec<f32> = if self.f16a {
                x.iter().map(|&v| f16_round(v)).collect()
            } else {
                x.to_vec()
            };
            let chunk = m.ne1.div_ceil(self.threads.max(1)).max(1);
            std::thread::scope(|s| {
                for (i, yb) in y.chunks_mut(chunk).enumerate() {
                    let xr = &xr;
                    s.spawn(move || {
                        let mut row = Vec::new();
                        for (k, out) in yb.iter_mut().enumerate() {
                            *out = m.dot_row_round(i * chunk + k, xr, &mut row);
                        }
                    });
                }
            });
            return;
        }
        if m.ne1 * m.ne0 >= 1 << 18 && self.threads > 1 {
            m.gemv_par(x, y, self.threads);
        } else {
            m.gemv(x, y);
        }
    }

    fn row(&self, m: &Mat, j: usize) -> Vec<f32> {
        let mut v = vec![0.0f32; m.ne0];
        m.dequant_row(j, &mut v);
        v
    }

    /// A weight vector: `norm` gains, router biases, `ssm_a`, `ssm_dt.bias`. These are 1-D, so they
    /// never go through the 2-D `Mat` path, and in this checkpoint every one of them is stored f32
    /// (Tools/tensor_census.py) so there is nothing to dequantise here.
    fn gain(&self, name: &str) -> Result<Vec<f32>> {
        let t = self.g.tensor(name)?;
        if t.n_dims != 1 || t.ty != qtypes::ty::F32 {
            return Err(Error::msg(format!(
                "{name}: expected a 1-D f32 vector, got {} dimensions of type {}",
                t.n_dims,
                qtypes::name(t.ty)
            )));
        }
        Ok(self.g.tensor_f32s(t)?.to_vec())
    }

    /// The mHC route: one unweighted RMS over the concatenated streams, the low-rank router, then
    /// `post`/`mix` with exactly `mhc.iterations` Sinkhorn sweeps.
    fn route(&self, a: &[f32], b: &[f32], block: usize, fn_: &str, tag: &str, d: &Dumper) -> Result<Route> {
        let hp = &self.plan.hp;
        let dm = &hp.dims;
        assert_eq!(a.len(), b.len());
        let mut flat = Vec::with_capacity(a.len() + b.len());
        flat.extend_from_slice(a);
        flat.extend_from_slice(b);
        let mut normed = vec![0.0f32; flat.len()];
        // 1e-6 is hard-coded in the reference for this norm, not the model's rms epsilon.
        rms_norm(&flat, 1e-6, &mut normed);
        let w = self.mat(&tname(block, &format!("{fn_}.weight")))?;
        let bias = self.gain(&tname(block, &format!("{fn_}.bias")))?;
        let n_route = dm.n_route;
        assert_eq!(w.ne1, n_route);
        let mut logits = vec![0.0f32; n_route];
        self.proj(&w, &normed, &mut logits);
        for i in 0..n_route {
            logits[i] += bias[i];
        }
        d.dump(&format!("mhc.{tag}.logits"), &format!("{n_route}"), &logits)?;
        let (s0, s1) = (sigmoid(logits[0]), sigmoid(logits[1]));
        let input = (0..a.len())
            .map(|i| (a[i] * s0 + b[i] * s1) * hp.mhc_pre_scale)
            .collect();
        let post = [sigmoid(logits[2]) * 2.0, sigmoid(logits[3]) * 2.0];
        let mut mix = if hp.mhc_fixed_identity {
            [1.0f32; 4]
        } else {
            [logits[4], logits[5], logits[6], logits[7]]
        };
        sinkhorn_sweeps(&mut mix, self.iters.unwrap_or(hp.mhc_iters));
        // The sweeps run in log space; the reference exponentiates after the last one, so `mix` is
        // the doubly stochastic mixing matrix by the time `merge` uses it (§2.7). Without this the
        // merge multiplies the streams by negative log-mix values and the residual blows up.
        for m in mix.iter_mut() {
            *m = m.exp();
        }
        d.dump(&format!("mhc.{tag}.post"), "2", &post)?;
        d.dump(&format!("mhc.{tag}.mix"), "4", &mix)?;
        Ok(Route {
            logits,
            input,
            post,
            mix,
        })
    }

    /// `a' = a*mix0 + b*mix1 + branch*post0` and the mirror for `b`, both from the pre-merge streams.
    fn merge(a: &mut [f32], b: &mut [f32], branch: &[f32], r: &Route) {
        for i in 0..a.len() {
            let (ai, bi) = (a[i], b[i]);
            a[i] = ai * r.mix[0] + bi * r.mix[1] + branch[i] * r.post[0];
            b[i] = ai * r.mix[2] + bi * r.mix[3] + branch[i] * r.post[1];
        }
    }

    /// Full attention for one token against the execution's own KV cache (§2.3).
    fn attention(&mut self, ex: &Exec, x: &[f32], pos: usize, d: &Dumper) -> Result<Vec<f32>> {
        let hp = &self.plan.hp;
        let dm = &hp.dims;
        let blk = ex.tensor_block;
        let eps = hp.rms_eps;
        let qn = self.gain(&tname(blk, "attn_q_norm.weight"))?;
        let kn = self.gain(&tname(blk, "attn_k_norm.weight"))?;
        // Widths come from the checkpoint's own matrices, never from a assumed head geometry
        // (`proj` re-checks both sides). `attn_q` is [n_embd, n_q_heads * 2 * d_head]: each head
        // block is [q | gate], which is why the head stride is twice the head width.
        let wq = self.mat(&tname(blk, "attn_q.weight"))?;
        let wk = self.mat(&tname(blk, "attn_k.weight"))?;
        let wv = self.mat(&tname(blk, "attn_v.weight"))?;
        let n_h = dm.attn_n_q_heads;
        let n_kvh = dm.attn_n_kv_heads;
        let dh = dm.attn_d_head;
        let hstride = wq.ne1 / n_h;
        require!(
            hstride == 2 * dh,
            Error::msg(format!(
                "{}: head block is {} wide, expected 2 * key_length ({})",
                wq.name,
                hstride,
                2 * dh
            ))
        );
        require!(
            qn.len() == dh && kn.len() == dh,
            Error::msg(format!(
                "{}: head norm is {} wide, {} is {} wide, expected key_length {}",
                tname(blk, "attn_q_norm.weight"),
                qn.len(),
                tname(blk, "attn_k_norm.weight"),
                kn.len(),
                dh
            ))
        );
        let mut qg = vec![0.0f32; wq.ne1];
        self.proj(&wq, x, &mut qg);
        let mut k = vec![0.0f32; wk.ne1];
        self.proj(&wk, x, &mut k);
        let mut v = vec![0.0f32; wv.ne1];
        self.proj(&wv, x, &mut v);

        // Per-head Q/K norm, then rope over the first `n_rot` dimensions only. The second half of
        // each head block is the output gate and is never rotated.
        let (cos, sin) = self.rope.table(pos);
        let mut q = vec![0.0f32; qg.len()];
        for h in 0..n_h {
            let src = &qg[h * hstride..h * hstride + dh];
            let dst = &mut q[h * hstride..h * hstride + dh];
            rms_norm_w(src, eps, &qn, dst);
            apply_rope(dst, &cos, &sin);
        }
        for h in 0..n_kvh {
            let range = h * dh..h * dh + dh;
            let hk = &k[range.clone()];
            let mut tmp = vec![0.0f32; hk.len()];
            rms_norm_w(hk, eps, &kn, &mut tmp);
            apply_rope(&mut tmp, &cos, &sin);
            k[range].copy_from_slice(&tmp);
        }

        let (kc, vc) = &mut self.kv[ex.logical];
        if self.kv_f16 {
            // round-trip on write, the way an f16 cache stores what it is given
            kc.extend(k.iter().map(|&v| f16_round(v)));
            vc.extend(v.iter().map(|&v| f16_round(v)));
        } else {
            kc.extend_from_slice(&k);
            vc.extend_from_slice(&v);
        }
        let kv_w = n_kvh * dh;
        let ntok = kc.len() / kv_w;
        let scale = 1.0f32 / (dh as f32).sqrt();
        let kv_head = |h: usize| h * n_kvh / n_h;

        // `o` is the packed attention output (n_h * dh), *not* the [q | gate] layout of `qg`: the
        // gate is read out of `qg` and applied here, and `attn_output` takes n_head*key_length in.
        let mut o = vec![0.0f32; n_h * dh];
        let mut p = vec![0.0f32; ntok];
        for h in 0..n_h {
            let qh = &q[h * hstride..h * hstride + dh];
            let kh = kv_head(h);
            let mut max = f32::NEG_INFINITY;
            for t in 0..ntok {
                let kr = &kc[t * kv_w + kh * dh..][..dh];
                let mut s = 0.0f32;
                for i in 0..dh {
                    s += qh[i] * kr[i];
                }
                p[t] = s * scale;
                if p[t] > max {
                    max = p[t];
                }
            }
            let mut sum = 0.0f32;
            for pr in p.iter_mut() {
                *pr = (*pr - max).exp();
                sum += *pr;
            }
            let inv = 1.0f32 / sum;
            let mut acc = vec![0.0f32; dh];
            for t in 0..ntok {
                let pt = p[t] * inv;
                let vr = &vc[t * kv_w + kh * dh..][..dh];
                for i in 0..dh {
                    acc[i] += pt * vr[i];
                }
            }
            let dst = &mut o[h * dh..h * dh + dh];
            for i in 0..dh {
                // The gate is applied after attention and before wo.
                dst[i] = acc[i] * sigmoid(qg[h * hstride + dh + i]);
            }
        }
        d.dump(&format!("attn.{}.out", ex.logical), &format!("{}x{}", n_h, dh), &o)?;
        let mut out = vec![0.0f32; hp.n_embd];
        self.proj(&self.mat(&tname(blk, "attn_output.weight"))?, &o, &mut out);
        d.dump(&format!("attn.{}.branch", ex.logical), &hp.n_embd.to_string(), &out)?;
        Ok(out)
    }

    /// Gated DeltaNet for one token (§2.4): fused projections, causal depthwise conv, then the
    /// delta rule with the state carried per execution.
    fn gdn(&mut self, ex: &Exec, x: &[f32], d: &Dumper) -> Result<Vec<f32>> {
        let hp = &self.plan.hp;
        let dm = &hp.dims;
        let blk = ex.tensor_block;
        let eps = hp.rms_eps;
        let (qkv_w, d_state, d_v) = (dm.qkv_width, hp.d_state, dm.d_v);

        let mut qkvz = vec![0.0f32; qkv_w + hp.n_embd];
        self.proj(
            &self.mat(&tname(blk, "attn_qkv_gate.weight"))?,
            x,
            &mut qkvz,
        );
        let qkv = qkvz[..qkv_w].to_vec();
        let z = qkvz[qkv_w..].to_vec();
        d.dump(&format!("gdn.{}.z", ex.logical), &format!("{}x{}", dm.n_v_heads, d_v), &z)?;

        let mut ba = vec![0.0f32; 2 * dm.n_v_heads];
        self.proj(&self.mat(&tname(blk, "ssm_beta_alpha.weight"))?, x, &mut ba);
        let beta: Vec<f32> = ba[..dm.n_v_heads].iter().map(|v| sigmoid(*v)).collect();
        let ssm_a = self.gain(&tname(blk, "ssm_a"))?;
        let dt_bias = self.gain(&tname(blk, "ssm_dt.bias"))?;
        // `ssm_a` is stored as `-exp(A_log)`, so `gate` is a negative log-decay.
        let gate: Vec<f32> = (0..dm.n_v_heads)
            .map(|h| ssm_a[h] * softplus(ba[dm.n_v_heads + h] + dt_bias[h]))
            .collect();

        // Causal depthwise conv over the raw qkv. Tap `d_conv-1` is the current token, tap 0 the
        // oldest cached one.
        let rows = hp.d_conv - 1;
        let cw = self.mat(&tname(blk, "ssm_conv1d.weight"))?;
        let win = &mut self.conv[ex.logical];
        d.dump(&format!("gdn.{}.conv_state.in", ex.logical), &format!("{rows}x{qkv_w}"), win)?;
        let mut conv_out = vec![0.0f32; qkv_w];
        for c in 0..qkv_w {
            let w = cw.row_f32(c).ok_or_else(|| {
                Error::msg(format!("{}: conv taps must be f32", cw.name))
            })?;
            let mut s = w[rows] * qkv[c];
            for k in 0..rows {
                s += w[k] * win[k * qkv_w + c];
            }
            conv_out[c] = silu(s);
        }
        // Slide the window: drop the oldest row, append this token's raw qkv.
        win.copy_within(qkv_w..rows * qkv_w, 0);
        win[(rows - 1) * qkv_w..].copy_from_slice(&qkv);
        d.dump(&format!("gdn.{}.conv_state", ex.logical), &format!("{rows}x{qkv_w}"), win)?;

        // Split into heads; q/k are L2-normalised, v is not. The eps here is the model's rms epsilon
        // (`build_gdn_l2_norm` = `rms(x, eps/n) / sqrt(n)` = `x / sqrt(sum(x^2) + eps)`).
        let qk_w = dm.n_k_heads * d_state;
        let mut cq = vec![0.0f32; qk_w];
        let mut ck = vec![0.0f32; qk_w];
        let mut cv = vec![0.0f32; dm.n_v_heads * d_v];
        cq.copy_from_slice(&conv_out[..qk_w]);
        ck.copy_from_slice(&conv_out[qk_w..2 * qk_w]);
        cv.copy_from_slice(&conv_out[2 * qk_w..]);
        for h in 0..dm.n_k_heads {
            l2_norm(&mut cq[h * d_state..h * d_state + d_state], eps);
            l2_norm(&mut ck[h * d_state..h * d_state + d_state], eps);
        }
        d.dump(&format!("gdn.{}.q_t", ex.logical), &format!("{}x{}", dm.n_k_heads, d_state), &cq)?;
        d.dump(&format!("gdn.{}.k_t", ex.logical), &format!("{}x{}", dm.n_k_heads, d_state), &ck)?;
        d.dump(&format!("gdn.{}.v_t", ex.logical), &format!("{}x{}", dm.n_v_heads, d_v), &cv)?;

        let sn = self.gain(&tname(blk, "ssm_norm.weight"))?;
        let out_scale = 1.0f32 / (d_v as f32).sqrt();
        let mut core = vec![0.0f32; hp.n_embd];
        let st = &mut self.state[ex.logical];
        let mut delta = vec![0.0f32; d_v];
        let mut normed = vec![0.0f32; d_v];
        for h in 0..dm.n_v_heads {
            let kh = h % dm.n_k_heads;
            let qh = &cq[kh * d_state..kh * d_state + d_state];
            let kk = &ck[kh * d_state..kh * d_state + d_state];
            let base = h * d_v * d_state;
            // decay -> predict -> write -> read.
            let decay = gate[h].exp();
            for i in 0..d_v {
                let row = &mut st[base + i * d_state..base + (i + 1) * d_state];
                for s in row.iter_mut() {
                    *s *= decay;
                }
            }
            for i in 0..d_v {
                let row = &st[base + i * d_state..base + (i + 1) * d_state];
                let mut pred = 0.0f32;
                for j in 0..d_state {
                    pred += row[j] * kk[j];
                }
                delta[i] = (cv[h * d_v + i] - pred) * beta[h];
            }
            for i in 0..d_v {
                let row = &mut st[base + i * d_state..base + (i + 1) * d_state];
                for j in 0..d_state {
                    row[j] += kk[j] * delta[i];
                }
            }
            let head = &mut core[h * d_v..h * d_v + d_v];
            for i in 0..d_v {
                let row = &st[base + i * d_state..base + (i + 1) * d_state];
                let mut acc = 0.0f32;
                for j in 0..d_state {
                    acc += row[j] * qh[j];
                }
                head[i] = acc * out_scale;
            }
            // Gated RMS norm per head: rms(core) * ssm_norm, then * silu(z).
            rms_norm_w(head, eps, &sn, &mut normed);
            for i in 0..d_v {
                head[i] = normed[i] * silu(z[h * d_v + i]);
            }
        }
        d.dump(&format!("gdn.{}.state", ex.logical), &format!("{}x{d_v}x{d_state}", dm.n_v_heads), st)?;
        let mut out = vec![0.0f32; hp.n_embd];
        self.proj(&self.mat(&tname(blk, "ssm_out.weight"))?, &core, &mut out);
        d.dump(&format!("gdn.{}.branch", ex.logical), &hp.n_embd.to_string(), &out)?;
        Ok(out)
    }

    fn ffn(&self, ex: &Exec, x: &[f32], d: &Dumper) -> Result<Vec<f32>> {
        let hp = &self.plan.hp;
        let blk = ex.tensor_block;
        let mut g = vec![0.0f32; hp.n_ffn];
        let mut u = vec![0.0f32; hp.n_ffn];
        self.proj(&self.mat(&tname(blk, "ffn_gate.weight"))?, x, &mut g);
        self.proj(&self.mat(&tname(blk, "ffn_up.weight"))?, x, &mut u);
        let f: Vec<f32> = (0..hp.n_ffn).map(|i| silu(g[i]) * u[i]).collect();
        let mut out = vec![0.0f32; hp.n_embd];
        self.proj(&self.mat(&tname(blk, "ffn_down.weight"))?, &f, &mut out);
        d.dump(&format!("ffn.{}.branch", ex.logical), &hp.n_embd.to_string(), &out)?;
        Ok(out)
    }

    /// Run the trunk over `tokens` and return the logits of the last position.
    ///
    /// `iters` overrides `mhc.iterations`; the Phase 2 gate runs `6` to prove the 20-sweep contract
    /// is exercised rather than assumed.
    pub fn run(&mut self, tokens: &[u32], iters: Option<usize>, dump: &Dumper) -> Result<Vec<f32>> {
        self.set_mhc_iters(iters);
        let hp = &self.plan.hp;
        let eps = hp.rms_eps;
        let embd = self.mat("token_embd.weight")?;
        let mut logits = Vec::new();
        for (t, &id) in tokens.iter().enumerate() {
            if id as usize >= embd.ne1 {
                return Err(Error::msg(format!(
                    "token id {id} is outside the embedding table ({} rows)",
                    embd.ne1
                )));
            }
            let mut cur = self.row(&embd, id as usize);
            dump.dump("embd", &hp.n_embd.to_string(), &cur)?;
            let mut a: Vec<f32> = Vec::new();
            let mut b: Vec<f32> = Vec::new();
            for il in 0..self.plan.n_exec() {
                if self.limit_exec.is_some_and(|n| il >= n) {
                    break;
                }
                let ex = *self.plan.exec(il);
                if ex.mtp > 0 {
                    // The MTP block is a separate graph in the reference too; `sn ref` scores the trunk.
                    continue;
                }
                let mhc = ex.mhc;
                let mut attn_route: Option<Route> = None;
                if mhc {
                    if ex.physical == hp.mhc_start {
                        a = cur.clone();
                        b = cur.clone();
                    }
                    let r = self.route(&a, &b, ex.tensor_block, "hc_attn_fn", &il.to_string(), dump)?;
                    cur = r.input.clone();
                    attn_route = Some(r);
                }
                let w = self.gain(&tname(ex.tensor_block, "attn_norm.weight"))?;
                let mut h = vec![0.0f32; hp.n_embd];
                rms_norm_w(&cur, eps, &w, &mut h);
                dump.dump(&format!("l.{il}.h"), &hp.n_embd.to_string(), &h)?;
                let out = if ex.recurrent {
                    self.gdn(&ex, &h, dump)?
                } else {
                    self.attention(&ex, &h, t, dump)?
                };
                if let Some(r) = attn_route {
                    Self::merge(&mut a, &mut b, &out, &r);
                    let r2 = self.route(&a, &b, ex.tensor_block, "hc_ffn_fn", &format!("{il}.ffn"), dump)?;
                    let w2 = self.gain(&tname(ex.tensor_block, "post_attention_norm.weight"))?;
                    let mut h2 = vec![0.0f32; hp.n_embd];
                    rms_norm_w(&r2.input, eps, &w2, &mut h2);
                    dump.dump(&format!("l.{il}.h2"), &hp.n_embd.to_string(), &h2)?;
                    let f = self.ffn(&ex, &h2, dump)?;
                    Self::merge(&mut a, &mut b, &f, &r2);
                    cur = (0..hp.n_embd).map(|i| (a[i] + b[i]) * 0.5).collect();
                    dump.dump(&format!("mhc.{il}.a"), &hp.n_embd.to_string(), &a)?;
                    dump.dump(&format!("mhc.{il}.b"), &hp.n_embd.to_string(), &b)?;
                } else {
                    for i in 0..hp.n_embd {
                        cur[i] += out[i];
                    }
                    let w2 = self.gain(&tname(ex.tensor_block, "post_attention_norm.weight"))?;
                    let mut h2 = vec![0.0f32; hp.n_embd];
                    rms_norm_w(&cur, eps, &w2, &mut h2);
                    dump.dump(&format!("l.{il}.h2"), &hp.n_embd.to_string(), &h2)?;
                    let f = self.ffn(&ex, &h2, dump)?;
                    for i in 0..hp.n_embd {
                        cur[i] += f[i];
                    }
                }
                dump.dump(&format!("l_out.{il}"), &hp.n_embd.to_string(), &cur)?;
            }
            let w = self.gain("output_norm.weight")?;
            let mut h = vec![0.0f32; hp.n_embd];
            rms_norm_w(&cur, eps, &w, &mut h);
            // The trunk's per-position hidden state, named so it can be compared against the
            // reference server's `/embedding` output with pooling none (see Tools/golden-embd.py).
            dump.dump(&format!("h_final.{t}"), &hp.n_embd.to_string(), &h)?;
            let wo = self.mat("output.weight")?;
            logits = vec![0.0f32; wo.ne1];
            self.proj(&wo, &h, &mut logits);
            if t + 1 == tokens.len() {
                dump.dump("logits", &wo.ne1.to_string(), &logits)?;
                dump.dump("final_norm", &hp.n_embd.to_string(), &h)?;
            }
        }
        Ok(logits)
    }
}
