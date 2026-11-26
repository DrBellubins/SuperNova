//! The numerics contract (Docs/design-plan.md §2) as tested code.
//!
//! Every function here is the reference the HIP kernels are later compared against, so the details
//! that "should not matter" are pinned deliberately: the softplus threshold, the order of the mHC
//! sweeps, the two separate `1/sqrt(128)` factors in the GDN step, and `rms(x) * w` with no `(1+w)`.

/// `softplus` switches to the linear branch at exactly this argument (the reference's `20.0f`).
pub const SOFTPLUS_THRESHOLD: f32 = 20.0;

#[inline]
pub fn softplus(x: f32) -> f32 {
    // log1p(exp(x)) with the same guard the reference uses; without the guard exp(x) overflows
    // around x = 88 and the result loses all precision well before that.
    if x > SOFTPLUS_THRESHOLD {
        x
    } else {
        x.exp().ln_1p() // ln(1 + e^x); ln_1p already adds the 1
    }
}

#[inline]
pub fn sigmoid(x: f32) -> f32 {
    1.0f32 / (1.0f32 + (-x).exp())
}

#[inline]
pub fn silu(x: f32) -> f32 {
    x * sigmoid(x)
}

/// `sum(x^2)`, four accumulators.
#[inline]
pub fn sum_sq(x: &[f32]) -> f32 {
    let mut a = [0f32; 4];
    for (i, v) in x.iter().enumerate() {
        a[i & 3] += *v * *v;
    }
    (a[0] + a[1]) + (a[2] + a[3])
}

/// RMS normalisation: `out = x * rsqrt(mean(x^2) + eps)`.
#[inline]
pub fn rms_norm(x: &[f32], eps: f32, out: &mut [f32]) {
    assert_eq!(x.len(), out.len());
    let scale = 1.0f32 / (sum_sq(x) / x.len() as f32 + eps).sqrt();
    for (o, v) in out.iter_mut().zip(x.iter()) {
        *o = *v * scale;
    }
}

/// RMS normalisation followed by the checkpoint's weight vector: `out = rms(x) * w`.
///
/// Note there is **no** `(1+w)` factor in this checkpoint: the norms are plain `x * scale * w`.
#[inline]
pub fn rms_norm_w(x: &[f32], eps: f32, w: &[f32], out: &mut [f32]) {
    assert_eq!(x.len(), out.len());
    assert_eq!(w.len(), x.len());
    let scale = 1.0f32 / (sum_sq(x) / x.len() as f32 + eps).sqrt();
    for ((o, v), ww) in out.iter_mut().zip(x.iter()).zip(w.iter()) {
        *o = *v * scale * *ww;
    }
}

/// L2 normalisation of a head vector: `x / sqrt(sum(x^2) + eps)`.
#[inline]
pub fn l2_norm(x: &mut [f32], eps: f32) {
    let scale = 1.0f32 / (sum_sq(x) + eps).sqrt();
    for v in x.iter_mut() {
        *v *= scale;
    }
}

/// In-place softmax over the full slice (attention scores for one head, one query).
pub fn softmax(x: &mut [f32]) {
    let max = x.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let mut sum = 0f32;
    for v in x.iter_mut() {
        *v = (*v - max).exp();
        sum += *v;
    }
    let inv = 1.0f32 / sum;
    for v in x.iter_mut() {
        *v *= inv;
    }
}

/// One mHC Sinkhorn sweep over the four log-mixes: pairs (0,1), (2,3), (0,2), (1,3).
///
/// Each pair update is an exact renormalisation of that pair: with `d = x-y`,
/// `exp(-softplus(-d)) = e^x/(e^x+e^y)` and `exp(-softplus(d)) = e^y/(e^x+e^y)`. The order of the
/// four pairs inside a sweep, and the number of sweeps, are both part of the contract: the reference
/// runs exactly `mhc.iterations` (20 for this checkpoint), and it only converges as O(1/n).
pub const SINKHORN_PAIRS: [(usize, usize); 4] = [(0, 1), (2, 3), (0, 2), (1, 3)];

/// One pair update; after it `exp(mix[i]) + exp(mix[j]) == 1`.
#[inline]
pub fn sinkhorn_pair(mix: &mut [f32; 4], i: usize, j: usize) {
    let d = mix[i] - mix[j];
    mix[i] = -softplus(-d);
    mix[j] = -softplus(d);
}

pub fn sinkhorn_sweeps(mix: &mut [f32; 4], sweeps: usize) {
    for _ in 0..sweeps {
        for (i, j) in SINKHORN_PAIRS {
            sinkhorn_pair(mix, i, j);
        }
    }
}

/// Paired (height,width) or single interleaved rope: rotate adjacent pairs `(2p, 2p+1)`.
///
/// Only the first `2 * cos.len()` dimensions of `head` are touched; the rest of the head is copied
/// through unchanged by the caller (this function does not touch them).
pub fn apply_rope(head: &mut [f32], cos: &[f32], sin: &[f32]) {
    for (p, (c, s)) in cos.iter().zip(sin.iter()).enumerate() {
        let (a, b) = (head[2 * p], head[2 * p + 1]);
        head[2 * p] = a * *c - b * *s;
        head[2 * p + 1] = b * *c + a * *s;
    }
}

/// Partial YaRN rope for this checkpoint (`rope.type = imrope`, sections `[11,11,10,0]`).
///
/// For text-only input every section carries the same position, so `imrope` collapses to plain rope
/// over pairs `p = 0..n_rot/2` with `theta_p = pos * base^(-p / (n_rot/2))`: the position is chosen
/// per pair by `p % 3`, but the frequency index stays the global pair index. The table below is
/// built once (`inv` below is `theta_scale^p` after YaRN interpolation) and evaluated per position.
pub struct Rope {
    /// `theta_scale = base^(-2/n_rot)`, the per-pair multiplier.
    theta_scale: f32,
    freq_scale: f32,
    ext_factor: f32,
    corr: [f32; 2],
    n_pairs: usize,
    mscale: f32,
}

impl Rope {
    /// Build from the checkpoint's rope hyperparameters. `sections` are the per-axis pair counts;
    /// they do not change the text-only table but are validated so a wrong model is not silently
    /// rope'd wrongly.
    pub fn new(
        n_rot: usize,
        freq_base: f32,
        sections: &[u32],
        yarn_factor: f32,
        ctx_orig: usize,
    ) -> Rope {
        assert!(n_rot % 2 == 0);
        // `rope.dimension_count = 0` means "do not rotate" (the reference takes the same path: the
        // rope kernel rotates the first `n_dims` dimensions and copies the rest). It is useful as a
        // differential probe, because every rope error is invisible at position 0.
        if n_rot == 0 {
            return Rope {
                theta_scale: 0.0,
                freq_scale: 1.0,
                ext_factor: 0.0,
                corr: [0.0, 0.0],
                n_pairs: 0,
                mscale: 1.0,
            };
        }
        // The section pair counts must cover the rope'd width; they select *which position* a pair
        // would use, and every position is equal for text, so the table itself is section-free.
        debug_assert!(sections.iter().sum::<u32>() as usize * 2 == n_rot);
        let n_pairs = n_rot / 2;
        // The reference's yarn parameters: freq_scale = 1/factor, ext_factor is enabled by the
        // scaling type being yarn, attn_factor 1.0, beta_fast 32, beta_slow 1.
        let freq_scale = 1.0f32 / yarn_factor;
        let ext_factor = 1.0f32;
        let attn_factor = 1.0f32;
        let (beta_fast, beta_slow) = (32.0f32, 1.0f32);
        let corr_dim = |n_rot_val: f32| -> f32 {
            (n_rot as f32) * (ctx_orig as f32 / (n_rot_val * 2.0 * std::f32::consts::PI)).ln()
                / (2.0 * freq_base.ln())
        };
        // `ggml_rope_yarn_corr_dims` (ggml/src/ggml.c:4377) truncates the two correlation dims
        // before clamping - `floorf` the start, `ceilf` the end - and the rope node keeps truncate on
        // unless something calls `ggml_rope_set_truncate(false)`, which nothing in this fork does
        // (`ggml_rope_get_truncate` is `op_params[3] == 0`). For this checkpoint that is the
        // difference between corr = [14.241, 21.122] and the [14, 22] actually used: pairs 14..22
        // rotate at a different rate, so the error is invisible at position 0 (theta = 0) and grows
        // with every position after it.
        let corr = [
            corr_dim(beta_fast).floor().max(0.0),
            (n_rot as f32 - 1.0).min(corr_dim(beta_slow).ceil()),
        ];
        // The reference seeds mscale with attn_factor and then scales it by the YaRN magnitude term,
        // and applies the product to **both** cos and sin.
        let mscale = attn_factor
            * if ext_factor != 0.0 {
                1.0f32 + 0.1 * (1.0f32 / freq_scale).ln()
            } else {
                1.0
            };
        Rope {
            theta_scale: freq_base.powf(-2.0 / n_rot as f32),
            freq_scale,
            ext_factor,
            corr,
            n_pairs,
            mscale,
        }
    }

    pub fn n_pairs(&self) -> usize {
        self.n_pairs
    }

    /// The ramp used to interpolate between the scaled and unscaled frequencies, at pair `p`.
    #[inline]
    fn ramp(&self, p: usize) -> f32 {
        // The reference passes the *dimension* index and divides it by 2 internally, so the
        // correlation dimensions are compared against pair indices as-is.
        let y = (p as f32 - self.corr[0]) / (self.corr[1] - self.corr[0]).max(0.001);
        1.0 - 1.0f32.min(y.max(0.0))
    }

    /// `cos`/`sin` tables (length `n_pairs`) for one position, including the YaRN `mscale` that the
    /// reference applies to **both** components.
    pub fn table(&self, pos: usize) -> (Vec<f32>, Vec<f32>) {
        let mut cos = vec![0.0f32; self.n_pairs];
        let mut sin = vec![0.0f32; self.n_pairs];
        // Mirrors the reference's accumulation: theta starts at the position and is multiplied by
        // theta_scale once per pair, in f32, so the rounding matches step for step.
        let mut theta = pos as f32;
        for p in 0..self.n_pairs {
            let theta_extrap = theta;
            let theta_interp = self.freq_scale * theta_extrap;
            let t = if self.ext_factor != 0.0 {
                let ramp_mix = self.ramp(p) * self.ext_factor;
                theta_interp * (1.0 - ramp_mix) + theta_extrap * ramp_mix
            } else {
                theta_interp
            };
            cos[p] = t.cos() * self.mscale;
            sin[p] = t.sin() * self.mscale;
            theta *= self.theta_scale;
        }
        (cos, sin)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn softplus_matches_log1p_exp_and_switches_at_20() {
        assert_eq!(softplus(20.0), 20.0f32.exp().ln_1p());
        assert_eq!(softplus(20.1), 20.1);
        assert_eq!(softplus(-30.0), (-30.0f32).exp().ln_1p());
        assert!((softplus(0.0) - std::f32::consts::LN_2).abs() < 1e-6);
        // softplus(x) - softplus(-x) == x, away from the linear branch.
        for x in [-8.0f32, -1.0, 0.5, 3.0, 12.0] {
            let d = softplus(x) - softplus(-x);
            assert!((d - x).abs() < 1e-4, "{x}: {d}");
        }
        // Large arguments must not overflow or lose the mantissa.
        assert_eq!(softplus(1e4), 1e4);
    }

    #[test]
    fn sigmoid_and_silu_values() {
        assert_eq!(sigmoid(0.0), 0.5);
        assert!((sigmoid(10.0) - 0.9999546).abs() < 1e-6);
        assert!((sigmoid(-10.0) - 4.539_787e-5).abs() < 1e-10);
        assert_eq!(silu(0.0), 0.0);
        assert!((silu(2.0) - 1.7615942).abs() < 1e-6);
        assert!((silu(-2.0) - -0.23840584).abs() < 1e-6);
    }

    #[test]
    fn rms_norm_is_unweighted_and_then_weighted() {
        let x = [3.0f32, -4.0, 12.0, 0.0];
        let mut out = [0.0f32; 4];
        rms_norm(&x, 1e-6, &mut out);
        let scale = 1.0f32 / ((9.0f32 + 16.0 + 144.0) / 4.0 + 1e-6).sqrt();
        assert_eq!(out, [3.0 * scale, -4.0 * scale, 12.0 * scale, 0.0]);
        let w = [1.0f32, 2.0, 0.5, 7.0];
        rms_norm_w(&x, 1e-6, &w, &mut out);
        assert_eq!(out, [3.0 * scale, -8.0 * scale, 6.0 * scale, 0.0]);
        // eps keeps a zero vector finite (the mHC router can see an all-zero stream pair).
        rms_norm(&[0.0f32; 4], 1e-6, &mut out);
        assert_eq!(out, [0.0; 4]);
    }

    #[test]
    fn l2_norm_uses_the_documented_epsilon_scale() {
        let mut x = [3.0f32, 4.0];
        l2_norm(&mut x, 0.0);
        assert!((x[0] - 0.6).abs() < 1e-6 && (x[1] - 0.8).abs() < 1e-6);
        // eps/128 for a 128-wide head is tiny but must keep the all-zero case finite.
        let mut z = [0.0f32; 4];
        l2_norm(&mut z, 1e-6 / 128.0);
        assert!(z.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn softmax_is_shift_stable_and_normalized() {
        let mut s = [1000.0f32, 1001.0, 999.0, 1002.5];
        softmax(&mut s);
        let sum: f32 = s.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6, "{sum}");
        assert!(s[3] > s[1] && s[1] > s[0] && s[0] > s[2]);
    }

    /// Each pair update renormalises exactly that pair; the sweep order is what makes the result
    /// non-commutative, so this test pins both.
    #[test]
    fn sinkhorn_pair_update_renormalises_that_pair() {
        let mut m = [0.2f32, -1.3, 4.0, 0.7];
        let (x, y) = (m[0], m[1]);
        let d = x - y;
        let nx = -softplus(-d);
        let ny = -softplus(d);
        let (a, b) = (x.exp(), y.exp());
        assert!((nx.exp() - a / (a + b)).abs() < 1e-6);
        assert!((ny.exp() - b / (a + b)).abs() < 1e-6);
        sinkhorn_pair(&mut m, 0, 1);
        sinkhorn_pair(&mut m, 2, 3);
        assert!((m[0].exp() + m[1].exp() - 1.0).abs() < 1e-5);
        assert!((m[2].exp() + m[3].exp() - 1.0).abs() < 1e-5);
        // The next two pairs touch the same lanes again, which is what makes the order matter.
        sinkhorn_pair(&mut m, 0, 2);
        sinkhorn_pair(&mut m, 1, 3);
        assert!((m[0].exp() + m[2].exp() - 1.0).abs() < 1e-5);
        assert!((m[1].exp() + m[3].exp() - 1.0).abs() < 1e-5);
        let mut s = [0.2f32, -1.3, 4.0, 0.7];
        sinkhorn_sweeps(&mut s, 1);
        assert_eq!(s, m);
    }

    /// Regression pins for the 20-sweep iterate the checkpoint was trained against, plus the fact
    /// that a shorter loop is *visibly* different (Phase 2 gate: `--iters 6` must differ).
    #[test]
    fn twenty_sweeps_are_pinned_and_six_are_different() {
        // Two shapes matter. A typical router output converges to machine precision in a handful of
        // sweeps; a near-degenerate one (two lanes strongly preferred on the same side) converges only
        // slowly, geometrically, and is still visibly off at 20 sweeps. Both are pinned here because
        // the second is why `mHC.iterations = 20` is part of the model, not an accuracy knob.
        let diff = |x: &[f32; 4], y: &[f32; 4]| -> f32 {
            (0..4).map(|i| (x[i] - y[i]).abs()).fold(0.0, f32::max)
        };
        let residual = |m: &[f32; 4]| -> f32 {
            let p = m.map(|v| v.exp());
            [
                (p[0] + p[1] - 1.0).abs(),
                (p[2] + p[3] - 1.0).abs(),
                (p[0] + p[2] - 1.0).abs(),
                (p[1] + p[3] - 1.0).abs(),
            ]
            .into_iter()
            .fold(0.0f32, f32::max)
        };
        let run = |start: [f32; 4], n| {
            let mut m = start;
            sinkhorn_sweeps(&mut m, n);
            m
        };

        // 1. Ordinary router logits: converged by 20 sweeps.
        let typical = run([1.1f32, -0.4, 0.25, 2.0], 20);
        let want = [-0.17974462f32, -1.8047445, -1.8047447, -0.17974465];
        for i in 0..4 {
            assert!(
                (typical[i] - want[i]).abs() < 1e-5,
                "mix[{i}] = {} expected {}",
                typical[i],
                want[i]
            );
        }
        assert!(residual(&typical) < 1e-6, "{}", residual(&typical));

        // 2. Near-degenerate router logits: still 2.4e-2 off after 20 sweeps, and 6 sweeps is a
        //    clearly different model. The reference runs exactly 20, so these are the numbers the
        //    engine has to reproduce - converging further changes the checkpoint's behaviour.
        let hard = [6.0f32, -6.0, 1.0, -1.0];
        let m20 = run(hard, 20);
        let want = [-0.025805635f32, -6.3575053, -3.6700375, -0.001735191];
        for i in 0..4 {
            assert!(
                (m20[i] - want[i]).abs() < 1e-5,
                "mix[{i}] = {} expected {}",
                m20[i],
                want[i]
            );
        }
        let r6 = residual(&run(hard, 6));
        let r20 = residual(&m20);
        let r200 = residual(&run(hard, 200));
        assert!((r20 - 0.0237).abs() < 2e-3, "20-sweep residual {r20}");
        assert!(r6 > r20 && r20 > r200, "{r6} {r20} {r200}");
        assert!(r200 < 1e-3, "200 sweeps must be close to a fixed point: {r200}");
        // 6 vs 20 sweeps changes the mixes by ~0.13 in log space, which is a different routing.
        assert!(diff(&run(hard, 6), &m20) > 0.05, "6 sweeps must differ from 20");
        // And 20 vs "iterate to convergence" is not a rounding difference either.
        assert!(diff(&m20, &run(hard, 200)) > 1.0);
    }

    #[test]
    fn rope_parameters_resolve_to_the_checkpoints_yarn_values() {
        let r = Rope::new(64, 1e7, &[11, 11, 10, 0], 2.0, 262144);
        assert_eq!(r.n_pairs(), 32);
        // The raw correlation dims are [14.241, 21.122], but the reference floors the start and
        // ceils the end before clamping (`ggml_rope_yarn_corr_dims`, truncate is on by default), so
        // the pair range it actually ramps over is [14, 22]. mscale = 1 + 0.1 ln 2.
        assert!((r.corr[0] - 14.0).abs() < 1e-6, "{}", r.corr[0]);
        assert!((r.corr[1] - 22.0).abs() < 1e-6, "{}", r.corr[1]);
        assert!((r.mscale - 1.0693155).abs() < 1e-5, "{}", r.mscale);
        assert!((r.freq_scale - 0.5).abs() < 1e-9);
        // theta_scale = 1e7^(-1/32)
        assert!((r.theta_scale - 1e7f32.powf(-1.0 / 32.0)).abs() < 1e-9);
        // Pairs below corr[0]/2 keep the unscaled frequency (ramp 1), above corr[1]/2 the scaled one.
        assert!((r.ramp(0) - 1.0).abs() < 1e-6);
        assert!((r.ramp(13) - 1.0).abs() < 1e-6);
        assert!((r.ramp(14) - 1.0).abs() < 1e-6);
        assert!((r.ramp(22) - 0.0).abs() < 1e-6);
        // Pair 18 is the exact midpoint of the truncated range, so the ramp is exactly a half.
        assert!((r.ramp(18) - 0.5).abs() < 1e-6);
        let (cos, sin) = r.table(0);
        // pos 0: cos = mscale, sin = 0 for every pair.
        assert!(cos.iter().all(|v| (*v - r.mscale).abs() < 1e-5));
        assert!(sin.iter().all(|v| v.abs() < 1e-5));
    }

    #[test]
    fn rope_table_matches_the_closed_form_per_pair() {
        let r = Rope::new(64, 1e7, &[11, 11, 10, 0], 2.0, 262144);
        let pos = 137usize;
        let (cos, sin) = r.table(pos);
        let base = 1e7f32;
        for p in 0..32 {
            let theta_extrap = (pos as f32) * base.powf(-(p as f32) / 32.0);
            let ramp = {
                let y = (p as f32 - r.corr[0]) / (r.corr[1] - r.corr[0]);
                1.0 - 1.0f32.min(y.max(0.0))
            };
            let t = 0.5f32 * theta_extrap * (1.0 - ramp) + theta_extrap * ramp;
            assert!(
                (cos[p] - t.cos() * r.mscale).abs() < 1e-4,
                "pair {p}: {} vs {}",
                cos[p],
                t.cos() * r.mscale
            );
            assert!((sin[p] - t.sin() * r.mscale).abs() < 1e-4);
        }
        // Rotation preserves the norm of every rotated pair.
        let mut head = vec![0.0f32; 64];
        for (i, h) in head.iter_mut().enumerate() {
            *h = (i as f32) * 0.1 - 3.0;
        }
        let before: f32 = (0..32).map(|p| head[2 * p].powi(2) + head[2 * p + 1].powi(2)).sum();
        apply_rope(&mut head, &cos, &sin);
        let after: f32 = (0..32)
            .map(|p| head[2 * p].powi(2) + head[2 * p + 1].powi(2))
            .sum();
        // mscale scales the whole table, so the pair norms scale by mscale^2.
        assert!((after / before - r.mscale.powi(2)).abs() < 1e-3, "{after} {before}");
    }
}
