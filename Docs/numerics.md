# Numerics contract, as tested code

`Docs/design-plan.md` §2 states the math. This file states **how each rule in §2 is pinned by
running code**, and **what the measured agreement is** between `sn ref` (this repository's scalar f32
forward: `crates/sn-core/src/{num,dq,forward}.rs`) and the reference engine
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
| RoPE: partial (first `n_rot=64` of 256), pairs adjacent, YaRN with `freq_scale=1/factor`, `ext_factor=1` because type is yarn, `mscale = attn_factor·(1+0.1·ln(1/freq_scale))` applied to **cos and sin**, IMROPE sections select *which position* multiplies θ while the **frequency index stays the global pair index** (⇒ text collapses to plain RoPE) | `num::Rope` | `rope_yarn` + `ggml_rope_cache_init` + `ggml_mrope_cache_init` (`ggml/src/ggml-cpu/ops.cpp:6122,6147,~6180`); `num::tests::rope_*` |
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
3. **Multi token**: agreement degrades with position. With f32 weights (`def32.py`, K=5) the per
   layer-input comparison is:

   | layer input | pos 0 | pos 1 | pos 2 |
   |---|---|---|---|
   | into block 1 (after GDN 0) | 1.00000000 | 1.00000000 | 1.00000000 |
   | into block 2, 3 | 1.00000000 | 1.00000000 | 1.00000000 |
   | **into block 4 (after full-attention 3)** | 1.00000000 | **0.99999212** | **0.99998816** |

   so the recurrent path (state, decay, conv window, delta rule) is exact across positions, and the
   first difference appears **inside the full-attention block once it has more than one token in the
   cache**. Ruled out as the cause by measurement: KV storage (`--kv-f16` changes the worst cos by
   1e‑6), weight quantisation (`def32` gives the same curve), the conv window (`--onehot
   ssm_conv1d.weight:3` moves it the wrong way), and float rounding (position 0 is bit-exact).
   Still open; the remaining candidates are the score path (how the four IMROPE position sections
   are filled for text, `GGML_ROPE_TYPE_IMROPE` + `ggml_mrope_cache_init`) and the GQA head→KV
   pairing under the multi-token mask. Note `rope.dimension_count = 0` is not a usable probe: the
   reference aborts ("tensor buffer not set").
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
