# Numerics contract, as tested code

`Docs/design-plan.md` §2 states the math. This file states **how each rule in §2 is pinned by
running code**, and **what the measured agreement is** between `sn ref` (this repository's scalar f32
forward: `Engine/crates/sn-core/src/{num,dq,forward}.rs`) and the reference engine
(`llama.cpp-opensensenova`, read-only). Phase 3+ diffs against these numbers, so anything here is a
measurement with a command line behind it, not an assertion.

## 1. Rules and where each one is enforced

| §2 rule | code | pinned by |
|---|---|---|
| `rms(x,ε) = x·rsqrt(mean(x²)+ε)`, then plain `*w` (**no `1+w`**) | `num::rms_norm`, `rms_norm_w` | `num::tests::rms_norm_is_unweighted_and_then_weighted`; §2.9 row checks |
| norm ε: attention/FFN/GDN use `attention.layer_norm_rms_epsilon` = 1e‑6; the **mHC** norm uses 1e‑6 **hard-coded** (`build_route`, `models/../sensenova-loop.cpp:108`) | `forward::route` | full-model prefix agreement, `prefix-check.py` K=5 |
| `softplus(x) = x if x > 20 else log1pf(exp x)`, threshold exactly 20.0 | `num::softplus` | `num::tests::softplus_matches_log1p_exp_and_switches_at_20` |
| GDN L2 norm is `x / sqrt(Σx² + ε)` with ε = `f_norm_rms_eps` (equals `build_gdn_l2_norm` = `rms(x, ε/n)·1/sqrt(n)`) | `num::l2_norm` | `num::tests::l2_norm_uses_the_documented_epsilon_scale`; block-by-block agreement K=5 |
| GDN step order **decay → predict → write → read**, two `1/sqrt(128)` factors (L2 norm and the read), v-head h → k-head `h % n_k_heads` | `forward::gdn` | matches `ggml_compute_forward_gated_delta_net_one_chunk` (`ggml/src/ggml-cpu/ops.cpp:11051`) term for term; agreement with f32 weights is exact (see §3) |
| conv: causal depthwise, taps aligned oldest→`w[0]` … current→`w[d_conv-1]`, cache holds the last `d_conv-1` rows oldest-first | `forward::gdn` | `ggml_compute_forward_ssm_conv_f32` (`ops.cpp:~5900`) + `build_conv_state_store` (`llama-graph.cpp:4096`, stores columns `[n_seq_tokens, +3)`) |
| beta = `sigmoid(ba[:, 0:n_v])`, gate = `ssm_a[h] · softplus(ba[:, n_v:2n_v] + ssm_dt[h])`, `ssm_a` stored as `-exp(A_log)` | `forward::gdn` | the `beta`/`alpha` view offsets in `qwen35.cpp:build_layer_attn_linear`; K=5 agreement |
| attention gate `sigmoid(g)` applied **after** softmax, **before** `wo`; head block is `[q|gate]` stride `2·d_head`; scale `1/sqrt(d_head)` | `forward::attention` | `qwen35.cpp:build_layer_attn` (`attn_pregate` → `gate_sigmoid` → `wo`) |
| mHC route: **one** unweighted RMS over the concatenated `2d`, 8 logits, `in = a·σ(ℓ0)+b·σ(ℓ1)` scaled by `pre_scale`, `post = 2σ(ℓ2),2σ(ℓ3)`, `mix = ℓ4..ℓ7` | `forward::route` | `sensenova-loop.cpp:105-146` |
| balancing: **exactly `mhc.iterations` (20)** sweeps of pairs (0,1),(2,3),(0,2),(1,3), pair update `δ=x−y; x←−softplus(−δ); y←−softplus(δ)`, then **`exp()` after the last sweep** | `num::sinkhorn_sweeps`, `forward::route` | `num::tests::twenty_sweeps_are_pinned_and_six_are_different`; the merged mix matrices are doubly stochastic after 20 sweeps |
| merge uses the **pre-merge** a,b; `cur = 0.5·(a+b)` is only the exposed value; streams re-seed at every `physical == mhc_start` (i.e. once per loop pass — `sensenova-loop.cpp:174`) | `forward::merge`, `forward::run` | K=5/K=52 agreement |
| RoPE: partial (first `n_rot=64` of 256), pairs **(p, p + n_rot/2)** — half-offset, not adjacent, YaRN with `freq_scale=1/factor`, `ext_factor=1` because type is yarn, `mscale = attn_factor·(1+0.1·ln(1/freq_scale))` applied to **cos and sin**, IMROPE sections select *which position* multiplies θ while the **frequency index stays the global pair index** (⇒ text collapses to plain RoPE) | `num::Rope` | `rope_yarn` + `ggml_rope_cache_init` + `ggml_mrope_cache_init` (`ggml/src/ggml-cpu/ops.cpp:6122,6147,~6180`); `num::tests::rope_*` |
| **correlation dims are truncated**: `corr[0]=floor(...)`, `corr[1]=ceil(...)` before clamping (`ggml_rope_yarn_corr_dims`, ggml.c:4377; `ggml_rope_get_truncate` is true unless someone calls `ggml_rope_set_truncate(false)`, which nothing in this fork does) → this checkpoint uses **[14, 22]**, not the raw [14.241, 21.122] §2.6 quotes | `num::Rope::new` | `num::tests::rope_parameters_resolve_to_the_checkpoints_yarn_values` (fixed 2026‑11‑26: the old test pinned the raw values, which is how the bug survived) |
| weight formats q4_K / q6_k / f32 read byte-exactly | `dq.rs`, `qtypes.rs` | §2.9 (vs `Tools/dequant_ref.py`), `dq::tests::*`, `Tools/stage-check.py` |
| GGUF block indices are first-visit-logical (`blk.52..55` = physical 28..31) | `plan.rs` | `tests/plan_gates.rs` |

## 2. The noise floor of the oracle itself

The golden reference is a *quantised* engine, and its CPU path requantises every activation row to
q8_0 before a Q4_K/Q6_K dot (`ggml_vec_dot_q8_0_q4_K`): ~3e‑4 relative per matmul, ~1e4× larger
than anything an f32 implementation can be blamed for. Two captures of the *same* checkpoint through
two of the fork's own backends (ROCm-host vs CPU, `Docs/golden/ref` vs `Docs/golden/ref-cpu`, and
`Docs/golden/embd` vs `Docs/golden/embd-cpu`) measure that spread directly:

| prompt (tokens) | centred logit NMSE, reference vs reference | per-position hidden-state cos, reference vs reference |
|---|---|---|
| p1-capital (5) | 3.6e‑3 | 0.9969 … 0.9997 |
| p3-code (19) | 1.05e‑2 | — |
| p5-loop-pressure (35) | 8.2e‑3 | 0.9978 … 0.99998 (flat in position) |

So the plan's original "logit NMSE ≤ 1e‑4" is not reachable against this oracle; `Tools/golden-check.py`
gates on `max(1e‑4, margin × measured floor)` instead, and the honest target is *agreement at the
level the reference agrees with itself*.

## 3. Measured agreement of `sn ref` against the reference (2026‑11‑26)

Differential tools built for Phase 2 (all under `Tools/`):

* `sn-oracle.c` — per-layer hidden states, per-position `result_norm` and logits straight from
  libllama (`llama_set_embeddings_layer_inp`, `llama.h:1182`). Force `--` CPU with the default
  `--cpu`: with the ROCm split graph, `extract_layer_inputs` aborts on `backend != nullptr`
  (`llama-context.cpp:2807`) for several layer ids. Build (the binary is git-ignored):

  ```sh
  gcc -O1 -w -o Tools/sn-oracle Tools/sn-oracle.c \
    -I <fork>/include -I <fork>/ggml/include \
    -L <fork>/build/bin -lllama -lggml -lggml-base \
    -Wl,-rpath,<fork>/build/bin -lm
  # <fork> = /home/justin/Documents/Github/llama.cpp-opensensenova
  # usage: Tools/sn-oracle --model M.gguf --layers 1-5 --tokens-file IDS  # or positional ids
  ```
* `truncate.py` / `prefix-check.py` — a model whose trunk *is* the first K executions (§1.1 makes
  that exact, because every cache is per execution). Sweeps cos(K) to find the block that breaks.
* `def32.py` — same model with the block weights converted to f32, so the weights stop being a
  variable. `ablate.py` — zero / one / one-hot whole tensor roles in both engines at once.

Results:

1. **Weight reading**: exact (§2.9, 1e‑11).
2. **Single token (position 0), full model**: `h_final` cos **0.99995**, worst across all 56
   executions; prefix truncations K = 5,6,8,…,28 and K = 52 (blocks 0..27 twice) all agree at
   **cos ≥ 0.9996**, and with `def32` weights cos = **1.00000000**, i.e. the whole non-recursive
   trunk, the mHC routes/merges, the norms, the dequantisers and the block-index remap are exact.
3. **Multi token — the open item of 2026-11-26 is closed: the rope *pair layout*.**
   `Tools/position-check.py` (prefix-truncate → `def32` → compare `result_norm` **per position**)
   settles it. Same 5-execution prefix, f32 block weights, 12 tokens of `p5-loop-pressure`:

   | position | `sn ref` (fixed) | `--rope-adjacent` (the defect) | `--attn-f16` |
   |---|---|---|---|
   | 0 | 0.9999999995 | 0.9999999995 | 1.0000000000 |
   | 1 | 0.9999999988 | 0.9999787001 | 0.9999999991 |
   | 4 | 0.9999999982 | 0.9999228928 | 0.9999999996 |
   | 8 | 0.9999999971 | 0.9996215263 | 0.9999999998 |
   | 11 | 0.9999999960 | 0.9991539387 | 0.9999999988 |

   ```sh
   ./Tools/build-kernels.sh   # not needed here; see Docs/bench.md for Phase 3
   python3 Tools/position-check.py --blocks 5 --tokens 12 --prompt p5-loop-pressure \
       --variants ",--rope-adjacent,--attn-f16"
   ```

   **The defect.** `num::apply_rope` rotated adjacent pairs `(2p, 2p+1)`; this checkpoint rotates
   `(p, p + n_rot/2)`. The pair layout is chosen by **architecture**, not by GGUF metadata:
   `llama_model_rope_type` returns `LLAMA_ROPE_TYPE_IMROPE` for `general.architecture =
   sensenova_loop` (`src/llama-model.cpp:3867`), and the rope op's dispatch
   (`ggml/src/ggml-cpu/ops.cpp:6379-6386`) calls `rotate_pairs(n_dims, n_dims/2, cache, ...)`
   (`ops.cpp:6236`), i.e. `x[p]` pairs with `x[p + n_rot/2]` while `GGML_ROPE_TYPE_NORMAL`
   (stride 1) is the adjacent form. The two are **identical at position 0** — there `theta = 0`, so
   `sin = 0` and `cos = mscale` for every pair, and a rotation is a scale whatever pairing it
   walks. That is exactly why every §3.2 measurement agreed to `1.00000000` while positions ≥ 1
   drifted, and it is why "position 0 is bit-exact" is a weak assertion about rope: it constrains
   `mscale`, nothing else. Pinned by `num::tests::rope_pairing_half_offset_rotates_across_the_half`
   and `rope_pairings_are_indistinguishable_at_position_zero`; `sn ref --rope-adjacent` reproduces
   the defect so this measurement stays repeatable.

   **What the two earlier candidates turned out to be.** Neither:
   * *IMROPE position sections for text* — no effect, as §2.6 already says. `ggml_mrope_cache_init`
     (`ops.cpp:6164`) selects `theta_t/h/w/e` per pair from `sections = [11,11,10,0]`
     (`sector = (i0/2) % 32`; every pair 0..31 lands in t/h/w, never in the `e` section), and for a
     text batch all four position rows are the same value (`p_t = pos[i2]`, `p_h = pos[i2 + ne2]`,
     …), so it collapses to plain rope with the **global pair index** — which is what we implement.
     The correlation-dimension ramp *is* evaluated on the pair index: `rope_yarn_ramp` does
     `(i0 / 2 - low) / (high - low)` (`ops.cpp`), i.e. the reference divides the dimension index by
     2 itself.
   * *GQA head→KV pairing* — identical. Ours is `kv_head(h) = h * n_kv / n_head`; the CPU FATT uses
     `ik2 = iq2 / rk2` with `rk2 = n_head / n_kv` (`ops.cpp:8845-8860`). Also, a wrong pairing is
     visible at position 0 (each head would read a different `v`), and position 0 was exact.
   * *The reference's operand precision in the score path* is real but is **not** the gap, and is now
     measured: flash attention is on by default (`llama_context_default_params().flash_attn_type =
     LLAMA_FLASH_ATTN_TYPE_AUTO`, and `src/llama-context.cpp:322` maps AUTO to
     `cparams.flash_attn = true`), so the score path is `ggml_flash_attn_ext(q_f32, k_f16, v_f16,
     mask_f16)` — the mask must be f16 (`ggml.c:5434` asserts it), and the CPU kernel converts Q to
     K's `vec_dot_type` (f16 for an f16 cache) at `ops.cpp:8850` before `kq_vec_dot`.
     `Tools/attn-probe.c` builds that exact op with synthetic inputs against the same op with f32
     K/V: the whole-row cos is 0.99999997 at neutral (unit-Gaussian) q/k, 0.99999990 at a score std
     of ~16², and never worse than 0.9999995 — one to two orders of magnitude too small to explain
     what was seen. `sn ref --attn-f16` reproduces those operands when a closer-than-f32 comparison
     is wanted (it is the last column above).

   Consequence for the rest of the plan: **f16 attention operands are inside the oracle's own noise
   budget** (≤1e-7 at the model level here), so Phase 6/8 may compute QK in f16 without moving
   away from the reference; only the f32 *accumulation* is load-bearing. What is *not* optional is
   the pair layout, the `1/sqrt(d_head)` scale, and the rope frequency table.

4. **Gate as run today** (`Tools/golden-check.py --floor-dir Docs/golden/ref-cpu/ref`): top‑1
   matches the reference for **5/5** corpus prompts; see `Docs/golden/` and the run log.

## 4. What Phase 3+ may assume

* Everything in §1 is safe to implement as kernels; the mHC route/merge, the GDN recurrence, the
  norms, the conv and the dequantisers are confirmed identical to the reference at f32.
* Comparisons against the *fork* must carry the §2 floor: expect ~1e‑2 centred logit NMSE on long
  prompts, and use per-position hidden-state cos (`sn ref --dump`, `Tools/oracle-check`/`sn-oracle`)
  rather than logits to find a defect — logits amplify.
* Do not "optimise" the 20 sweeps: they are the trained iterate (`num::tests` pins the difference
  against 6 and against convergence).

## 5. Phase 4: the same rules, now in HIP (2026‑11‑27)

The kernels do not get their own numerics. They get the §1 rules, a scalar oracle in
`Engine/crates/sn-core/src/oracle.rs` that states each rule once, and a comparison of the two — plus a comparison
against the Phase 2 dumps, because agreement with my own oracle proves only that the kernel implements
*something* consistently.

* **One implementation of routing, three consumers.** `oracle::route_derive` / `mhc_route` / `mhc_merge`
  are what `forward::RefModel::route`/`merge` call, what the checks compare against, and what the
  kernels were written from. `forward.rs` no longer contains a second copy, so the reference, the oracle
  and the kernel cannot drift independently. (`golden-check` reproduces the Phase 2 closure numbers
  exactly after the move: 1.79e‑3 / 9.57e‑3 / 8.87e‑3 / 2.37e‑3 / 4.85e‑2, top‑1 5/5.)
* **Measured:** kernel vs oracle, 25 checks, worst f32 NMSE 1.69e‑13; `mix` 6.17e‑13 against its 1e‑6
  gate. Kernel vs Phase 2 dumps on real weights (executions 4..11): worst `mix` 4.5e‑15, worst other
  stage 8.65e‑13. Details and commands: `Docs/bench.md` §2.1–2.2.
* **The mHC region has two ends, and both are now ops.** §2.7 creates the streams at the first mHC
  execution (`a = b = cur`) and hands the trunk `0.5·(a+b)` at the last one — the value the *next*
  execution reads, which for this checkpoint is the first exit block (`blk.52`), not the output head.
  `oracle::mhc_seed` / `oracle::mhc_expose` are the spec, `sn_mhc_seed_f32` / `sn_mhc_expose_f32` the
  kernels (NMSE 0), and `decode::compile` places both by walking the plan's mHC boundaries. Seeding
  then exposing is the identity — which is exactly why an unseeded program reports plausible numbers
  instead of an error: it scores whatever the previous token left in the stream slots.
* **The sweep rewrite is an identity, not a truncation.** `sn_mhc_route_f32` computes the pair update as
  `m_i' = m_i − lse(m_i, m_j)`, `m_j' = m_j − lse(...)`, which equals the reference's
  `m_i' = −softplus(−(m_i−m_j))`, `m_j' = −softplus(m_i−m_j)` by definition of `lse`; it exists because
  `log1pf` compiles to a software fallback on GCN and 80 serial calls per thread dominated the kernel.
  `logf(1 + exp(−|d|))` differs from `log1p(exp(−|d|))` by at most `exp(−|d|)/2` in absolute terms — for
  `|d| > 17` that is below the f32 resolution of the result, and every sweep renormalises anyway. Measured
  against the oracle: mix NMSE 6.17e‑13 (the pre-rewrite form was 2.85e‑17 — the difference is the
  float4 accumulation order, not the sweeps). The *unfused baseline* keeps `log1pf` verbatim: it has to
  look like the fork's nodes, and its speed is not the point.
* **`fixed_identity` dumps differently now.** With `mhc.fixed_identity = true` the mix is now
  `[1, 0, 0, 1]` (the identity, as the flag says) where the Phase 2 dump code wrote `[1, 1, 1, 1]`.
  Both are dead code for this checkpoint (`fixed_identity false`, pinned by the `mhc_route_f32 mix
  (fixed_identity)` check), but a future dump of a model that sets the flag will not be comparable to an
  old one.
* **f16 in / f16 out is a store, not a computation.** The f16 kernels accumulate in f32 and round once,
  so their floor is the f16 quantisation step (relative 4.88e‑4, `oracle::f16_rel`) and *not* 1e‑5; the
  checks say so with a separate gate. `sn_from<T>`/`sn_to<T>` are explicit template specialisations
  because an `sn_h(float) -> __half` overload quietly routed every f32 path through f16 (measured:
  `rmsnorm_f32` NMSE 4e‑8, `silu_mul_f32` → inf).
* **A kernel can be wrong without any API noticing.** The reduction scratch of `sn_mhc_route_f32` is
  sized by a `#define` in the `.hip` while the workgroup size is chosen in Rust; with the two disagreeing
  the reduction reads outside its array and returns wrong logits for every row but the first — silently,
  deterministically, and differently whenever the launch's `sharedBytes` changes. `Engine/target/hip/*.meta`
  (written by `Tools/build-kernels.sh`) and `KernelRegistry::add_image` (which refuses the mismatch) are
  the guard; `decode::route_static_shared()` documents why the dynamic shared size must be 0.
