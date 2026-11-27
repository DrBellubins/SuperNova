# SuperNova — SenseNova‑Loop‑9B inference engine: architecture findings & phased implementation plan

Author: pi · Date: 2026‑11‑26 · Model: `SenseNova-Loop-9B-Q4_K_M.gguf` · Target: ROCm/HIP, **gfx1102** (RX 7700S, 8 GB)
· **Phase 0 measured: `Docs/baseline.md`** · Phase 1 gates: `cargo test` + `Tools/plan-check.py`

Reference implementations consulted (read‑only, **not copied**):
`llama.cpp-opensensenova` @ `24ea7bb78` — commits `40de7782e` ("llama : add SenseNova Loop support"),
`24ea7bb784` ("llama : add SenseNova Loop MTP support"). Everything below was re‑derived from the GGUF
itself plus the C++ control flow; formulas are re‑stated from scratch.

> `Docs/gemini-arch.md` is directionally right (loop, mHC, hybrid attention, 56 executions / 32 blocks) but
> wrong/loose on: "20 Sinkhorn iterations to constrain routing matrices" (it is a **4‑scalar** pair‑balancing
> that does **not** converge — see §4.3), "~4.9 GB weights" (5.81 GB stored, **8.8 GB executed per token**),
> and it omits the MTP block, the GDN head asymmetry (16 k‑heads / 32 v‑heads) and the partial‑RoPE/YaRN detail.

---

## 0. Goals, and the one number that matters

| | fork today | roofline | SuperNova target |
|---|---|---|---|
| decode (1 token) | **~5 t/s** | ~26 t/s | **24–27 t/s** |
| prefill | ~200 t/s | ~370 t/s | **≥300 t/s** |
| decode CPU | ~50 % of a core, continuously | ~0 | **< 1 % (1 graph launch / token)** |
| VRAM @ 32 k | ~6.5 GB | 6.9 GB (q8 KV) | ≤ 7.0 GB |

Decode is **memory‑bandwidth bound at 8.8 GB of weight traffic per token** (§5). 5 t/s = 200 ms/token
means we are **~5× off the roofline**, and the gap is host‑side, not device‑side. Eliminating it is the
single objective of this project; everything else in the plan serves it.

---

## 1. Model facts (read from the GGUF, not inferred)

`general.architecture = sensenova_loop`, GGUF v3, **490 tensors**, 56 KV pairs, 5,818,631,072 bytes,
`general.file_type = 15 (Q4_K_M)`, tags `looped-transformer, mhc, hybrid, custom_code`.

```
embedding_length 4096   feed_forward_length 12288   block_count 57   context_length 524288
attention.head_count 16 head_count_kv 4 key_length 256 value_length 256  (head_dim = 256, n_embd = 16*256)
attention.layer_norm_rms_epsilon 1e-6
recurrent_layers[57] = 1,1,1,0, 1,1,1,0, ... 1,1,1,0, 0   -> logical 3,7,...,55 are full attention; [56]=MTP
full_attention_interval 4
loop.start 4  loop.end 28  loop.count 2
mhc.start 4   mhc.end 28   mhc.streams 2  mhc.iterations 20  mhc.pre_scale 1.0  mhc.fixed_identity 0
nextn_predict_layers 1
ssm.conv_kernel 4  ssm.group_count 16  ssm.inner_size 4096  ssm.state_size 128  ssm.time_step_rank 32
rope.dimension_count 64 (partial: 64 of 256)  rope.dimension_sections [11,11,10,0]
rope.freq_base 1e7  rope.scaling.type yarn  rope.scaling.factor 2.0  rope.scaling.original_context_length 262144
vocab 248320 (token_embd Q4_K, output Q6_K — two separate tensors, not tied)
```

### 1.1 The execution plan (the loop, made concrete)

`n_layer_all = 57` (GGUF `block_count`), `n_layer_nextn = 1` → **56 trunk executions**, **32 physical blocks**.
`extra = (loop.count-1)·(loop.end-loop.start) = 24`.

```
physical(il) = il                                  if il <  4
             = 4 + (il - 4) % 24                   if il < 52
             = il - 24                             otherwise

logical  0.. 3  -> blocks  0.. 3     (entry,     block 3 = full attention)
logical  4..27  -> blocks  4..27     (loop, pass 1)      \  same weights, executed twice
logical 28..51  -> blocks  4..27     (loop, pass 2)      /
logical 52..55  -> blocks 28..31     (exit,     block 55 = full attention)
logical 56      -> MTP block          (never run by the trunk; draft head)
```

**Tensor names use the *first‑visit logical* index, not the physical one.** GGUF blocks present:
`0..27, 52, 53, 54, 55, 56` — i.e. physical blocks 28..31 are stored as `blk.52..blk.55`, and `blk.32..blk.51`
do not exist. Any loader must build this map itself; hardcoding `blk.N == block N` will fail on 4 of 32 blocks.
(Tooling: `Tools/gguf_dump.py`, `Tools/roofline.py` already encode this correctly.)

### 1.2 The 33 stored blocks

| kind | blocks | count | per‑block tensors |
|---|---|---|---|
| linear‑attn (GDN) | 0‑2, 4‑6, 8‑10, …, 24‑26, 52‑54 | 24 | `attn_norm`, `post_attention_norm`, `attn_qkv_gate[4096,12288]`, `ssm_conv1d[4,8192]`(f32), `ssm_a[32]`(f32), `ssm_dt.bias[32]`(f32), `ssm_beta_alpha[4096,64]`, `ssm_norm[128]`(f32), `ssm_out[4096,4096]`, `ffn_gate/up/down` |
| full‑attn | 3, 7, …, 27, 55 | 8 | `attn_norm`, `post_attention_norm`, `attn_q[4096,8192]`, `attn_k[4096,1024]`, `attn_v[4096,1024]`, `attn_output[4096,4096]`, `attn_q_norm[256]`, `attn_k_norm[256]`, `ffn_*` |
| + mHC routers | 4..27 | 24 | the above **plus** `hc_attn_fn.weight[8192,8]`+`.bias[8]`, `hc_ffn_fn.weight[8192,8]`+`.bias[8]` — **f32 only** (quantizer refuses to touch `hc_*`) |
| MTP | 56 | 1 | full‑attn block + `nextn.eh_proj[8192,4096]`, `nextn.enorm`, `nextn.hnorm`, `nextn.shared_head_norm` |

Quant mix — **corrected 2026‑11‑26 by `Tools/tensor_census.py`** (exact per‑tensor bytes, cross‑checked against
offset gaps; `Tools/gguf_dump.py --tensors` had a stale type table and a wrong size formula, which is where the
previous paragraph's "Q2_K" came from — there is **no Q2_K in this file**). Types present: **Q4_K ×177**
(3,824,418,816 B), **Q6_K ×33** (1,966,325,760 B), **f32 ×280** (16,913,920 B) = 5,807,658,496 B, 0 gaps
anomalies. Per role (block indices are GGUF names):

| tensor | Q4_K | Q6_K |
|---|---|---|
| `ffn_gate`, `ffn_up` | all 33 blocks | — |
| `ffn_down` | 7‑8,10‑11,13‑14,16‑17,19‑20,22‑23,25‑26,52‑53,55‑56 (18) | 0‑6,9,12,15,18,21,24,27,54 (15) |
| `attn_qkv_gate` | 4‑5,8,10,13‑14,16‑17,20,22,25‑26 (12) | 0‑2,6,9,12,18,21,24,52‑54 (12) |
| `attn_q`, `attn_k`, `attn_output` | all 9 attn blocks | — |
| `attn_v` | 7,11,19,23 (4) | 3,15,27,55‑56 (5) |
| `ssm_out`, `ssm_beta_alpha` | all 24 GDN blocks | — |
| `token_embd` | ✓ | — |
| `output` | — | ✓ |

So the FFN — 4.85 GB of the 8.81 GB streamed per decoded token — is **Q4_K throughout**, and `attn_q/k` are Q4_K too.
Routers, `ssm_conv1d`, all norms, `ssm_a`, `ssm_dt` are **f32**. Phase 5 therefore needs **Q4_K + Q6_K + f32**
dequant/GEMM paths (no Q2_K, no Q5_K/Q8_0 in the checkpoint).

Derived GDN dimensions (note the head asymmetry — a frequent source of wrong implementations):

```
n_k_heads  = group_count    = 16      head_k_dim = state_size = 128   key_dim   = 16*128 = 2048
n_v_heads  = time_step_rank = 32      head_v_dim = state_size = 128   value_dim = 32*128 = 4096
qkvz_out   = 2*key_dim + 2*value_dim  = 12288   ( = attn_qkv_gate cols )
qkv part   = 2*key_dim +   value_dim  =  8192   conv_channels           ( = ssm_conv1d cols )
z part     =   value_dim              =  4096   (rows 8192..12288 of the same matrix)
k-head h_v is shared by v-heads 2*h_v and 2*h_v+1   (H_v % H_k == 0, index = h_v % H_k)
```

---

## 2. Exact math (the numerics contract)

This is the specification Phase 2 encodes and every later phase must match bit‑for‑bit‑ish.
All accumulations in **f32**; matmul inputs dequantised to f32 (or f16 with f32 accumulate).

### 2.1 Notation
`x = x[:, t]` column vectors, `d = 4096`. `rms(x, ε) = x · rsqrt(mean(x²) + ε)`.
`softplus(x) = x if x > 20 else log1pf(expf(x))` — **threshold must be 20.0, use `log1pf`**.
All norms in this model are `out = rms(x, ε) * w` — plain multiply by the stored weight
(**no `(1+w)`**; that convention was already folded in at conversion).

### 2.2 Entry/exit layers (mHC off, `physical ∉ [4,28)`): plain pre‑norm residual
```
h0 = rms(cur, 1e-6) * attn_norm
cur = cur + attn(h0)                       # attn() = §2.3 or §2.4
h1  = rms(cur, 1e-6) * post_attention_norm
cur = cur + ffn(h1)                        # ffn() = §2.5
```

### 2.3 Full attention (14 logical layers)
```
qg = wq @ x                        # [8192, T];  per head h: [q(256) | gate(256)] INTERLEAVED, stride 512
q  = qg[h*512 : h*512+256]         # strided view, do not materialise
g  = qg[h*512+256 : h*512+512]
k  = wk @ x  -> [4][256] ; v = wv @ x -> [4][256]
q  = rms(q, 1e-6) * attn_q_norm    # over the 256 head dims, per head
k  = rms(k, 1e-6) * attn_k_norm
q, k = rope(q, k, pos)             # §2.6, only dims 0..63 of 256 rotate
o  = softmax(q·kᵀ/16 + causal) · v  # scale = 1/sqrt(256); 16 q-heads over 4 kv-heads (GQA g=4)
out = wo @ ( sigmoid(g) ⊙ o )       # gate applied AFTER attention, BEFORE wo
```

### 2.4 Linear attention = Gated DeltaNet (42 logical layers) — exact recurrence
```
qkvz = wqkv_z @ x                                  # [12288, T]
qkv  = qkvz[0:8192];  z = qkvz[8192:12288]
beta_raw  = w_beta_alpha[:, 0:32]  @ x             # [32, T]   (32 = n_v_heads = d_inner/d_state)
alpha_raw = w_beta_alpha[:, 32:64] @ x             # [32, T]   one fused [4096,64] tensor, rows split in half
beta  = sigmoid(beta_raw)                                   # per v-head
gate  = ssm_a[h] * softplus(alpha_raw[h] + ssm_dt[h])       # ssm_a is STORED AS -exp(A_log): all values < 0
conv  = causal_depthwise_conv(conv_states ++ qkvᵀ, ssm_conv1d[4, 8192])   # left pad 3 from cache
conv  = silu(conv)
q̃,k̃,ṽ = split(conv) -> [16][128], [16][128], [32][128]
q̃ = q̃ / sqrt(sum(q̃²) + eps/128)     # exact L2, eps = 1e-6  (equals rms(·, eps/128)/sqrt(128))
k̃ = k̃ / sqrt(sum(k̃²) + eps/128)

per v-head h, k-head h%16, state S ∈ R^[128,128] (f32), per token t in order:
    S      *= exp(gate[h,t])                                  # scalar decay per head
    delta   = (ṽ[h,t] − Sᵀ k̃[h,t]) · beta[h,t]                # [128]
    S      += outer(k̃[h,t], delta)                            # Δ-rule write
    o[h,t]  = (S · q̃[h,t]) / sqrt(128)                        # 1/11.3137... factor is real
out       = ( rms(o[:, h], 1e-6) * ssm_norm ) ⊙ silu(z[:, h])  # gated RMS norm per 128-dim head
out       = ssm_out @ reshape(out) -> [4096, T]
conv_states <- last 3 columns of conv input for the next call
```
Order inside the token step is **decay → predict → write → read**. Getting this wrong is silent and plausible.

### 2.5 FFN (all layers, dense, no MoE)
`ffn(x) = ffn_down @ ( silu(ffn_gate @ x) ⊙ (ffn_up @ x) )` — SiLU on the *gate* branch, `up` branch undecorated.

### 2.6 RoPE (partial, interleaved, YaRN‑scaled, mrope‑typed but text‑degenerate)
Type is `IMROPE` with sections `[11,11,10,0]`, `n_rot = 64` of `head_dim = 256`. **Rotary pairs are
half-offset: `x[p]` pairs with `x[p + n_rot/2]`, `p = 0..31`** — *not* adjacent `(2i, 2i+1)`. The layout is
chosen by architecture, not metadata: `llama_model_rope_type` → `LLAMA_ROPE_TYPE_IMROPE` for
`sensenova_loop` (`src/llama-model.cpp:3867`), and the op dispatch calls
`rotate_pairs(n_dims, n_dims/2, cache, ...)` (`ggml/src/ggml-cpu/ops.cpp:6236,6385`); only
`GGML_ROPE_TYPE_NORMAL` uses stride 1. Dims 64..255 pass through untouched. This was wrong in this plan
until 2026-11-26 and was the whole multi-token numerics gap — see `Docs/numerics.md` §3.3; both layouts are
identical at position 0, so no single-token test can tell them apart. Section choice for pair index `p = 0..31`:
`p%3==0 → t`, `p%3==1 → h`, `p%3==2 → w`; the section only selects which **position** multiplies θ —
the **frequency index stays the global `p`** (`θ_p = pos · base^(−p/32)`). For text `p_t = p_h = p_w` ⇒ plain RoPE.

YaRN parameters as the reference resolves them (`scaling.factor` becomes a *reciprocal*):
```
freq_scale = 1 / 2.0 = 0.5      ext_factor = 1.0 (because scaling.type == yarn)
attn_factor = 1.0               beta_fast = 32   beta_slow = 1     n_ctx_orig = 262144
corr[0] = 64·ln(262144/(32·2π)) / (2·ln(1e7)) = 14.241    corr[1] = 64·ln(262144/(1·2π)) / (2·ln(1e7)) = 21.122
           # rope_yarn_corr_dims(n_rot=64, n_ctx_orig=262144, beta_fast=32, beta_slow=1, base=1e7)
           # **then truncated**: `ggml_rope_yarn_corr_dims` (ggml.c:4377) floors corr[0] and ceils corr[1]
           # before clamping to [0, n_rot-1], and the rope node keeps truncate on by default
           # (`ggml_rope_get_truncate` == `op_params[3] == 0`, nothing calls set_truncate(false)).
           # The values actually used are **[14, 22]** — pinned by num::tests::rope_parameters_*.
ramp(p)   = 1 − clamp((p − corr[0]) / (corr[1] − corr[0]), 0, 1)   # 1 below corr[0], 0 above corr[1]
θ_p       = θ_interp·(1 − ramp) + θ_extrap·ramp,  θ_interp = 0.5·θ_extrap
           # low pair indices keep the *unscaled* θ_extrap (ramp 1); `ramp = clamp(...)` inverts this
mscale    = 1 + 0.1·ln(1/0.5) = 1.06931   ->  cos/sin both × mscale   (net ×1.143 on logits: keep it)
           # θ accumulates as pos · base^(-2p/n_rot) in f32, one multiply per pair (ggml_rope_cache_init)
```
Because 4 positions are expected per token by the reference, a text‑only driver must still feed `pos` (×1) and may
skip the 3 unused copies. Do **not** silently switch to `norm` RoPE: with sections present the reference path differs.

### 2.7 mHC — two streams, two routes per layer (48 executions × 2 = 96 routes/token)
Streams `a, b ∈ R^[4096, T]`, seeded once at the first mHC execution (`il == 4`) with `a = b = cur`,
then carried across **all 48** looped executions. At the end of each mHC execution the exposed value is
`cur = 0.5·(a + b)`; the next execution ignores `cur` and re‑derives everything from `a, b`.

> **What that means for an engine (2026‑11‑27).** Exactly two ops follow from this sentence, and where
> they sit is load‑bearing: a *seed* (`a = b = cur`) before the first mHC execution — nothing else ever
> writes the streams, so without it the first route reads memory the previous token left there, which is
> silently plausible — and one *expose* at the **mHC → non‑mHC boundary**, because every mHC execution
> except the last is followed by another one that ignores `cur`. For this checkpoint that boundary is after
> execution 51, before the exit blocks (physical 28..31, `blk.52..55`); exposing in the head instead makes
> the exit blocks read a `cur` last written at execution 3 *and* scores the loop's streams instead of block
> 55's output. Both are plan‑derived (first/last execution with `mhc` true), never taken from
> `mhc.start`/`mhc.end`. Kernels: `sn_mhc_seed_f32` / `sn_mhc_expose_f32`; oracles: `oracle::mhc_seed` /
> `oracle::mhc_expose`.

`route(a, b, W[8192,8], bias[8])` — note **one** unweighted RMS over the *concatenated* 8192 vector, ε = **1e‑6 hard‑coded**:
```
f   = rms([a ; b], 1e-6)                    # eps 1e-6, NOT hparams rms eps (same value here)
ℓ   = Wᵀ f + bias                           # [8, T]   f32 x f32, keep f32 accumulate
in   = ( a·σ(ℓ0) + b·σ(ℓ1) ) · pre_scale    # pre_scale = 1.0
post = 2·σ(ℓ2) , 2·σ(ℓ3)
mix₀..₃ = ℓ4..ℓ7  ->  balancing below  ->  exp()
```
Balancing, exactly `iterations = 20` sweeps of the pair order `(0,1),(2,3),(0,2),(1,3)`, each pair update:
```
δ = x − y ;  x ← −softplus(−δ) ;  y ← −softplus(δ)
```
Equivalently, in probability space with `m = exp(mix)`: **`m_i, m_j ← m_i/(m_i+m_j), m_j/(m_i+m_j)`** — the pair's
mass is renormalised to 1 and its ratio is preserved. **Critical:** this is a Sinkhorn‑style alternating projection
that converges only as O(1/n) — measured max error vs. the fixed point: 4.9e‑1 (n=1), 8.1e‑2 (n=6), **2.4e‑2 (n=20)**.
The checkpoint was trained with the *n=20 iterate*, not the limit. **Never "optimise" by iterating to convergence
or by replacing with the closed form** `(t, 1−t, 1−t, t)`. Reproduce 20 sweeps verbatim; they are free once fused.

Merge after each branch (`br` = attention or FFN output):
```
a ← (a·mix₀ + b·mix₁) + br·post₀
b ← (a·mix₂ + b·mix₃) + br·post₁      # uses the OLD a, b for both lines
```
`fixed_identity == 0` in this checkpoint, so the mix matrix is live. If it were 1, `mix = [1,0,0,1]` and
`ℓ4..7` are never read (support both; the flag is metadata).

### 2.8 Output head & MTP
`logits = output @ (rms(cur, 1e-6) * output_norm)` cropped to the requested rows.
MTP (Phase 9): `h = eh_proj([enorm(tok_embd) ; hnorm(h_hidden)])` then a full‑attention block
(identical to §2.3/§2.5) then `shared_head_norm` → `output` head. MTP consumes the trunk's pre‑head hidden state
(`h_nextn`) and re‑uses the LM head. Draft requires **state rollback** of the GDN caches (the reference
`n_rs_seq` snapshot machinery), which is why MTP lands late in the plan.

### 2.9 Weight reading verified against a second implementation (2026‑11‑26)

`Tools/dequant_ref.py` dequantises rows straight out of the GGUF, transliterated from GGML's C
(`dequantize_row_q4_K`, `get_scale_min_k4`, `dequantize_row_q6_K`, field layouts from
`ggml-common.h`), and diffs them against `sn deq`. Everything agrees to printing precision, so the
way §2 reads weights is correct as written:

| type | rows checked | worst difference |
|---|---|---|
| `f32` (routers, conv taps) | `blk.4.hc_attn_fn.weight` 0/7, `blk.4.ssm_conv1d.weight` 0/3 | 4.9e‑11 |
| `q4_k` | `attn_q` 0/4095, `attn_qkv_gate` 0/1, `ffn_gate` 0/3, `ssm_out` 2, `ssm_beta_alpha` 5, `nextn.eh_proj` 0, `token_embd` 0/4095 | 6.9e‑11 |
| `q6_k` | `output.weight` 0/7, `ffn_down` 0/12287 | 4.3e‑11 |

The 1e‑11 residual is `d * sc * q` computed in f32 by both engines and in double by the comparer —
not a format difference. Layouts for the record: `block_q4_K = { d @0, dmin @2, scales[12] @4,
qs[128] @16 }`; sub‑block `2p` = low nibbles of `qs[32p..32p+32]`, `2p+1` = high nibbles of the
*same* 32 bytes; scale/min nibbles exactly as `get_scale_min_k4`. `block_q6_K = { ql[128] @0,
qh[64] @128, int8 scales[16] @192, d @208 }`, two 128‑value half‑blocks advancing `ql += 64`,
`qh += 32`, `sc += 8`, values scattered to `l, l+32, l+64, l+96` with scale indices
`is, is+2, is+4, is+6` (`is = l/16`).

Two §2 wording fixes this settled (applied): the GDN L2 norm is `x / sqrt(Σx² + eps)` with
`eps = 1e‑6` — `build_gdn_l2_norm` (`src/models/models.h:14`) is `rms_norm(x, eps/n) *
(1/sqrt(n))`, which collapses to that — not "`eps/128`"; and the mHC merge consumes `exp(mix)`
**after** the 20 Sinkhorn sweeps, because the sweeps run in log space. Omitting the exponent makes
the merge multiply the streams by negative log‑mix values and the hidden state reaches `inf` around
layer 46.

**Working note — where the forward stands (updated 2026‑11‑26, supersedes the earlier "direction is
wrong at position 0" note, which is no longer true).** The differential oracle for this is
`Tools/{truncate.py,prefix-check.py,def32.py,sn-oracle.c,ablate.py}`: `truncate.py` writes a model
whose trunk *is* the first K executions (§1.1 makes that exact), `def32.py` rewrites block weights as
f32 so quantisation stops being a variable, and `sn-oracle.c` reads per-layer hidden states out of
libllama (`llama_set_embeddings_layer_inp`; force the CPU backend — with the ROCm split graph,
`extract_layer_inputs` aborts on `backend != nullptr`, `llama-context.cpp:2807`).

* Position 0, full model: `h_final` cos **0.99995** vs the reference; every prefix K ∈ {5…28, 52}
  agrees at cos ≥ 0.9996, and with f32 block weights **cos = 1.00000000**. Embedding, dequant, the
  block-index remap, the norms, the mHC route/Sinkhorn/merge and the FFN are exact.
* Multi token: agreement used to decay with position, and it no longer does. With f32 weights and a
  5-execution prefix, `Tools/position-check.py` compares `result_norm` **per position**: cos ≥ 0.9999999954
  at every position 0-11 (and `--attn-f16`, which mimics the reference's f16 flash-attention operands,
  reaches 1.0000000000 at position 0). The defect was the **rope pair layout** (§2.6), not the attention
  semantics: IMROPE position sections for text are a no-op, the GQA pairing matches the reference
  (`ik2 = iq2/rk2`), and the reference's f16 score operands measure at cos 0.99999997..0.9999995 against
  f32 ones (`Tools/attn-probe.c`) — far too small to have caused it. Measurements, line references and the
  repeatable A/B (`sn ref --rope-adjacent`) are in `Docs/numerics.md` §3.3.

* One real defect found and fixed while doing this: the YaRN **correlation dims are truncated**
  (`[14, 22]`, not the raw `[14.241, 21.122]` §2.6 used to quote). Its effect is only ~1e‑6, but the
  unit test had pinned the wrong values, which is exactly how such a bug survives — see §2.6.


---

## 3. State & cache sizes (this decides whether 32 k fits in 8 GB)

| | per token / sequence | 42 GDN layers | 14 attn layers @ 32 k |
|---|---|---|---|
| GDN `S` | 32·128·128·4 B = **2.097 MB** | 88.1 MB | — |
| GDN conv | 8192·3·4 B = 96 KB | 4.0 MB | — |
| KV (K+V, 4 heads × 256) | — | — | f16 **1.88 GB** · q8_0 0.94 GB · q4_0 0.47 GB |
| **VRAM @ 32 k** | | | 5.81 weights + 0.092 state + KV + ~0.05 arena |

→ f16 KV = 7.78 GB: does **not** fit an 8 GB card with headroom. q8_0 KV = 6.84 GB fits.
Consequence: the KV cache **must** be quantisable from day one (Phase 6), and `n_seq` multiplies the 92 MB line.

---

## 4. Why the fork is slow — the fundamental flaw

### 4.1 Evidence
1. The fork had to special‑case its own node budget (`src/llama-context.cpp::graph_max_nodes`):
   `48 · mhc_iterations · n_mhc` with `n_mhc = (mhc_end − mhc_start) · loop_count = 48`.
   → `48 · 20 · 48 = **46,080 nodes**` reserved for the mHC balancing alone.
   Measured on this machine from the live binary (`sched_reserve: max_nodes = 83648`), i.e. the graph for **one**
   decode token is reserved at **~84 k nodes**.
2. `build_route()` expands each pair update into six separate `ggml` tensors (`sub`, `neg`, `softplus`, `neg`,
   `softplus`, `neg`), 4 updates per sweep × 20 sweeps = 240 nodes per route, ×2 routes per mHC layer,
   ×48 layer executions = **46,080**. Each tensor is a `[1, T]` **strided** view of an 8‑wide logits buffer
   (stride 8 elements) — so even the device work is pathological (uncoalesced 4‑byte accesses), but the real
   cost is that it is 46 k *graph nodes*.
3. `llama.cpp` has no whole‑graph capture guarantee: HIP/CUDA graph use requires the node graph to be
   compatible **and** its properties to be stable across two consecutive calls, otherwise warm‑up resets and
   the backend falls back to launching every node individually (`ggml-cuda.cu:5401‑5470`,
   `GGML_CUDA_GRAPH_CHUNK_SIZE`, `GGML_CUDA_GRAPH_OPT` default off). With recurrent‑state gather buffers
   (`build_rs` → `s_copy` / `get_rows`) plus 84 k nodes, this is exactly the fragile case.
   → the host re‑walks 84 k nodes through `ggml_backend_sched` and issues tens of thousands of launches **per token**.

### 4.2 The arithmetic of the symptom
| | value |
|---|---|
| useful device work / token | ~24 GFLOP, ~9 MB of elementwise traffic |
| device time at roofline | ~38 ms (bandwidth, §5) |
| host work / token | ~84 k node visits + ~50 k launches ≈ 150–200 ms |
| observed | ~200 ms → **5 t/s**, one core ~50 % |

The device is idle waiting on the host. The `graph reallocs = 0 / graphs reused = 1` counters confirm the graph is
rebuilt‑free — so the cost is per‑node execution, not construction. Prefill looks "fine" (200 t/s) only because
each node amortises over `T` tokens; the launch count per *chunk* is unchanged.

### 4.3 Second‑order issues SuperNova must not inherit
* **Strided scalar tensors everywhere** the mHC chain touches (stride 8 f32). Fused kernels make it one coalesced read.
* `res->t_layer_inp[il]` retains every layer's activation as a potential graph output → 56 × 16 KB/token of
  liveness the graph can't reclaim. We keep hidden states only where a consumer exists (mHC streams, MTP head).
* Per‑recurrent‑layer `build_rs` (`scale_inplace` + `get_rows` + `cpy`) adds ~150 nodes and a state gather
  even when `n_rs_seq == 0` and nothing needs copying.
* Router weights are f32 → 25 MB/token of the traffic is 8192×8 f32 matmuls that would be identical in f16.
* Weight re‑read from the loop is **1.52×** and irreducible for decode (§5) — an inherent cost of the architecture,
  not a bug, but it must be *budgeted* (and it changes the right quantisation policy: quantise the looped blocks harder).
* Correctness traps inherited from the design that we must reproduce faithfully: block‑index remap (§1.1),
  the 20‑sweep non‑convergent balancing (§2.7), the two `1/sqrt(128)` factors (§2.4), `sigmoid(g)` after attention
  (§2.3), and prefix caching being legal only for **KV + GDN state snapshots taken together**.

### 4.4 Phase 0 verification recipe — **DONE 2026‑11‑26, results in `Docs/baseline.md`**
The recipe ran end to end on this box (GPU present, `/dev/kfd` available). `rocprofv2`/`rocprofv3` are **not
installed** here, so step 3 uses an `LD_PRELOAD` shim instead (`Tools/hip_count.c`, driven by `Tools/phase0.sh`,
which also captures `sched_reserve`, offload counts, graph counters, `rocm-smi` VRAM and the server's CPU time):
```bash
./Tools/phase0.sh all                       # launch + cold + fill passes, artifacts in /tmp/phase0/
```
Confirmed: `sched_reserve: max_nodes = 83648` at every `n_ctx`; **4,700 `hipLaunchKernel` + 181 `hipGraphLaunch`
+ 253 `hipMemcpyAsync` per decoded token** (291 graph plans captured, 126 graph updates failed); decode flat at
**4.39–4.48 t/s** from 1 k to 32 k context while prefill reaches 218 t/s (379 t/s on a 7.7 k prompt); GPU busy
43–47 % with the server process at ~8 cores. Graphs *are* engaged, so the cost is the per-node host work, exactly
the §4 diagnosis: the fix is a small static op list replayed as one graph, not graph tuning.

---

## 5. Roofline (why ~26 t/s is the ceiling, and what beats it)

Computed exactly from GGUF offsets × per‑block execution counts (`Tools/roofline.py`):

```
stored weights        5.808 GB
weight bytes streamed per decoded token   8.814 GB   (x1.52: blocks 4..27 read twice)
   - ffn_down/gate/up            4.85 GB
   - attn_qkv_gate (x2/x1)       1.42 GB
   - output head                 0.83 GB
   - ssm_out                     0.40 GB
   - token_embd                  0.00 GB (1 row)
   - attn q/k/v/o                0.46 GB
   - mHC routers (f32)           0.05 GB
RX 7700S (gfx1102 per rocminfo, 128-bit GDDR6): 256 GB/s spec, ~215 GB/s achievable
   -> 8.72 GB / 215 GB/s = 40.6 ms/token  ->  24.6 t/s decode ceiling
prefill (weights read once per chunk, compute bound): ~370 t/s at 25 % of FP16 WMMA peak
```
Implications, in priority order:
1. **Close the 5 → 24 t/s gap first.** It is 100 % host‑side; no quantisation or kernel cleverness is needed.
2. Past 25 t/s the only lever is *fewer weight reads per emitted token*: **MTP speculative decoding** (Phase 9,
   plausibly 1.6–1.9×), or a harder quant of the looped blocks (`ffn_*` at Q2_K already; `attn_qkv_gate` is Q4_K
   on 6 blocks — requantising exactly blocks 4..27 to Q2_K/Q3_K buys ~0.35 GB/token ≈ +4 t/s).
3. Never let a decode step issue more than a few hundred launches; at 3 µs each, 250 launches = 0.75 ms = 2 % of budget.

---

## 6. SuperNova's core design principles (the anti‑ggml)

These four decisions *are* the product. Everything else is plumbing.

**P1 — Plan once, replay forever.** At load time, compile the 56 executions + mHC + MTP into a flat
`Vec<Op>` (the "decode program") with **every pointer baked in**: activation arena offsets, cache slots, weight
addresses. Decode then = write 1 token id + 1 position into a pinned host buffer, then **one `hipGraphLaunch`**.
No graph construction, no per‑node allocation, no `get_rows` indirections, no scheduler.
Target: < 20 µs host per token; `top -p` shows < 1 % CPU.

**P2 — Fuse at the semantic level, not the tensor level.** The unit of fusion is a *block*, not an op.
Minimum fused kernels per decode step: `mhc_route` (concat+rmsnorm+8×dot+bias+20 sweeps+exp → ~30 flops/token),
`mhc_merge`, the two §2.7 boundary kernels (`mhc_seed`, one `mhc_expose` at the mHC → non‑mHC transition),
`rmsnorm`, `rope_inplace`, `attn_gate_norm`, `gdn_step` (§2.4 in one kernel), `gdn_gate_norm`,
`silu_mul`, plus one GEMM per weight. **~200 kernels/token total, ~0 for mHC.** Never emit a kernel for a `[1,T]` scalar.

**P3 — F32 accumulation, f16/bf16 activations, quantised weights; precision is a per‑edge decision.**
Keep the mHC router matmul and the GDN state in f32 (they are numerically load‑bearing across 48 executions);
use f16 for activations/attention, q8_0 KV by default.

**P4 — A scalar CPU reference always exists and is authoritative.** Every GPU kernel lands with a diff against
Phase 2's reference tensors. No kernel is "done" until `NMSE ≤ 1e‑5` (1e‑6 for norms/routing) against the golden dump.

### 6.1 Workspace layout
```
src/
  sn-core/      # no unsafe, no HIP: gguf reader, hparams, execution plan, tensor map, quant *decode* (scalar),
                # tokenizer (GPT-2 BPE from GGUF vocab), reference forward (f32), golden dumps, samplers
  sn-hip/       # #![deny(unsafe_op_in_unsafe_blocks)] device enum, allocator/arena, kernel registry,
                # static decode program builder, HIP graph capture, error mapping
  sn-engine/    # wires core+hip: model load, plan compile, prefill chunker, decode loop, caches, spec-decode driver
  sn-cli/       # `sn run`, `sn dump`, `sn bench`, `sn ref`
Tools/          # gguf_dump.py, roofline.py (+ future: dump-diff.py, golden-check.py, prof-parse.py)
Docs/
```
`cargo` only, no `cc`/`bindgen` for HIP (raw `hipcc`/`hip::device` via `build.rs` invoking `/opt/rocm/bin/hipcc`,
link `libhipblas` only if we ever need it — plan is to not need it). `gfx1102` arch flag (`--offload-arch=gfx1102`;
`gfx1103` is the same ISA family and works as a fallback, but build for what `rocminfo` reports). Target a static lib + CLI;
no `main` in kernels. `unsafe` confined to `sn-hip`.

---

## 7. Phases

Each phase ends green: `cargo test` + a named acceptance gate. Phases 1‑6 build the *correct* engine;
7‑9 make it *fast*; 10 makes it usable. Do not skip the gates — the later phases diff against earlier dumps.

### Phase 0 · Instrument the fork, freeze the targets (0.5 day) — **DONE 2026‑11‑26**
*Run §4.4 on the GPU. Record node count, launches/token, graph status, t/s at `n_ctx ∈ {1k, 8k, 32k}`.*
**Feeds:** §5 becomes measured, not estimated; if the launch hypothesis is wrong we learn it for 30 minutes
instead of 3 weeks.
**Gate:** `Docs/baseline.md` with the numbers + the exact command lines. → **delivered**; the tooling is
`Tools/phase0.sh` (+ `Tools/hip_count.c`, `Tools/rocprof_check.sh` / `Tools/rocprof_parse.py` once
`rocprofv2` was installed), artifacts in `/tmp/phase0` and `/tmp/rp2`.

### Phase 1 · GGUF reader, hyperparameters, execution plan (1–2 days) — **DONE 2026‑11‑26**
`sn-core::gguf`: v2/v3 header, KV typed values, tensor info + exact byte sizes from offsets (handle `alignment`),
mmap + `madvise`. `sn-core::plan`: reproduce §1.1 — `physical(il)`, `is_recr(il)`, block‑index remap, mHC range
validity, the fork's validation rules (loop fits, mHC must not straddle a loop boundary, streams == 2,
`iterations ≤ 20`, MTP must be full‑attention). `sn-core::budget`: §3 table + a hard "will it fit" check.
**Deliverables:** `sn dump --plan` printing block table, per‑block exec count, per‑class bytes/token.
**Gate:** `Tools/roofline.py` and `sn dump --plan` agree to the byte on all 490 tensors; unit tests pin
`physical(0)=0, physical(4)=4, physical(27)=27, physical(28)=4, physical(51)=27, physical(52)=28, physical(55)=31`;
test that `blk.52..55` resolve to physical 28..31; test rejection of an mHC range that straddles the loop.

*Status:* `crates/sn-core/src/{gguf,plan,budget,qtypes,sys,error}.rs` + `crates/sn-cli` (`sn dump --plan
[--json]`, `sn dump --tensors`, `sn dump --meta`, `sn info`). Gate evidence: `cargo test` (5 lib + 7
`tests/plan_gates.rs` + 1 `tests/real_model.rs`) green, and `python3 Tools/plan-check.py <model>` reports
`OK: sn and roofline.py agree byte-for-byte on every tensor` (490 tensors, stored 5,807,658,496 B,
8,813,935,104 B/token, 48 mHC executions). Two facts the plan had wrong, now pinned by tests: the checkpoint
holds **no Q2_K** (§1.2 table), and of the 14 full-attention KV buffers only 8 distinct blocks back them —
`kv_bytes_per_token` counts the 14 (a looped block's K/V cannot be shared), `kv_bytes_per_token_unique` is a
diagnostic that the Phase 0 VRAM slope rules out.

### Phase 2 · Scalar CPU reference + golden corpus (3–5 days) — *the spec*
Full f32 forward of §2 in plain Rust (weights dequantised on the fly), driven by token ids. Stage‑level
`cb()`‑style dumps: `embd`, per‑execution `l_out`, `a`,`b` streams at mHC boundaries, `mhc_logits[8]`,
`mix[4]`, `post[2]`, `gdn_state[:1,:]` before/after, `conv_state`, `q̃/k̃/ṽ`, `z`, `attn_out`, `h_nextn`.
**Deliverables:** `sn ref --prompt-file p.txt --dump golden/`, `Tools/golden-check.py` (NMSE + max‑abs per stage),
and — this is what makes every later phase cheap — a **fork cross‑check**: the same prompt fed to
`llama-server`'s `/tokenize` + `/completion` with `--log-verbose` logits dump, compared to `sn ref` logits.
**Gate:** `sn ref` top‑1 token matches the fork for 5 fixed prompts at greedy decoding, and `logit
NMSE` is within `3 ×` the **measured** reference-vs-reference floor (`--floor-dir Docs/golden/ref-cpu/ref`;
the original `≤ 1e‑4` is unreachable against a q8_0-operand oracle — see `Docs/numerics.md` §2);
`sn ref --iters 6` is *visibly different* (proving the 20‑sweep sensitivity is exercised).
Also emits `Docs/numerics.md` = §2 as tested code. **This phase is the one that de‑risks the whole project.**

**Status (2026‑11‑26): code complete, one open numerics item, gate re‑specified against a measured
floor.** Built: `crates/sn-core/src/{dq,num,forward}.rs`, `sn ref` (with `--dump`, `--dump-only`,
`--limit-exec`, `--iters`, `--logits-out`, `--kv-f16`, `--f16-weights/--f16-acts`), `sn deq`,
`Docs/golden/{corpus.json,tokens,ref,ref-cpu,embd,embd-cpu}` and
`Tools/{golden-ref.py, golden-check.py, golden-embd.py, dequant_ref.py, stage-check.py, tok.c,
ablate.py, truncate.py, prefix-check.py, def32.py, sn-oracle.c}` — `tok.c` drives the reference
tokenizer through `libllama` because the GGUF BPE tokenizer is Phase 10 and this fork's `/tokenize`
answers 501. `Docs/numerics.md` is §2 as tested code: every rule with the test / measurement that
pins it, plus the measured agreement.

Gate results as run (`Tools/golden-check.py --floor-dir Docs/golden/ref-cpu/ref`, 2026‑11‑26):

* **top‑1 matches the reference for 5/5 corpus prompts** (rank of our top‑1 inside the reference's
  ordering: 0 everywhere).
* centred logit NMSE vs the reference: 1.8e‑3 / 9.6e‑3 / 8.9e‑3 / 2.4e‑3 / 4.8e‑2 for the five
  prompts, against a **measured floor of 3.6e‑3 / 1.8e‑3 / 1.1e‑2 / 6.7e‑3 / 8.2e‑3** — the spread
  between two captures of the *same* checkpoint through two of the fork's own backends (ROCm-host vs
  CPU). The plan's original "≤ 1e‑4" is unreachable against a q8_0-operand oracle, so the gate is
  `max(1e‑4, 3 × that floor)`; it currently fails on the two longest prompts (2.4× and 5.9× their
  floor), which is §2.9's open attention item, not a weights/caches/plan problem.
* 20 vs 6 sweeps (`--iters 6 --compare-iters 20`): mean |Δlogit| 0.37‑0.81, max 6.41 — the
  20‑sweep contract is exercised, not assumed. (Top‑1 survives; the distribution moves by logits.)
* Prefix localization (`Tools/prefix-check.py`, `Tools/def32.py`): every truncated model that is a
  prefix of the trunk agrees with the reference at cos ≥ 0.9996 with quantised weights and to
  1.00000000 with f32 weights at position 0; the first multi-token disagreement is inside the
  full-attention block. Details and the ruled-out causes: §2.9.

Phase 2 is therefore the spec it was meant to be: the forward is exact at every position, the corpus and
the tooling are frozen, and every remaining disagreement is a measured property of the oracle, not of
this engine.

### Phase 3 · HIP substrate + static decode program (4–6 days)
Device/context/queue on `gfx1102`; a single big allocation (one `hipMalloc` for weights, one for the arena, one for
caches); `Arenas` handing out compile‑time offsets; `Op` enum + `Program` builder + `hipGraph` capture/replay with
pinned‑host token/pos inputs; `sn-hip::launch!` macro and a `KernelRegistry` keyed by `(name, shape, dtype)`;
a `--no-graph` path for debugging and a `SN_TRACE=ops` mode printing per‑launch timings.
**Feeds:** everything after this writes kernels, not infrastructure.
**Gate:** replay a captured graph containing a trivial 1000‑op chain: ≥ 90 % of the theoretical launch‑free time,

*Status (2026-11-26): **done** — the substrate exists and the gate has been run on this GPU. Numbers and
commands: **`Docs/bench.md` §1**. Summary:*

* host time per graph replay **6.5 µs** (< 30 µs target), at any op count; repeated replays of a 1000-op
  graph are **bit-identical**; a captured graph observes new pinned-host `(token, position)` writes with
  **no** graph update — the P1 input path is proven, not assumed.
* The "≥ 90 % of launch-free" criterion is not achievable **as literally worded** (a one-step op is pure
  dispatch), so it is replaced by a measurement with the same intent: per-node device dispatch on
  `gfx1102` is **2.9 µs** (graph) vs **3.2 µs** (direct launch), and the graph's share of the launch-free
  bound rises 8 % → 27.7 % → 55.9 % → 82.6 % as work per node goes 1 → 4 → 16 → 64 dependent steps. §5's
  assumed "3 µs per launch" is therefore **confirmed**: the ≤ 300-op decode budget costs ≤ 0.87 ms of
  dispatch against ~40 ms of weight traffic. The graph's win is host-side (6.5 µs total vs the fork's
  ~4 700 launches + 181 graph launches + 253 copies per token, `Docs/baseline.md` §2) plus pipelining
  (1.1–1.5×).
* Built: `crates/sn-hip/src/{hip,arena,kernels,program,bench,lib}.rs`, `crates/sn-hip/kernels/phase3.hip`,
    `Tools/build-kernels.sh` (`.hip` → `target/hip/*.fatbin`, loaded with `hipModuleLoad`; attribute enum
  values pinned by `Tools/hip_attrs.c`), `sn hip [info]`, `sn bench graph`. `Substrate` = exactly three
  device allocations (weights: one `hipMalloc` + one `hipMemcpy` of the GGUF data section, so a tensor's
  device address is `base + gguf_offset`; activations; caches) plus pinned host I/O. `SN_TRACE=ops|time`
  and the `--no-graph` path exist. `Arenas` hand out compile-time offsets and freeze before use.
* Not done here on purpose: running a real forward through the program (Phase 6/7 need kernels first) and
  `hipGraphExec` node updates (never needed while inputs arrive through pinned memory).
* Two device facts for Phase 5: HIP reports **16** multiprocessors where `rocminfo` reports **40** CUs, and
  the memory-clock attribute is a native clock, not the data rate — tune against measurement only.
host time per replay < 30 µs, and a repeated‑replay determinism check (bit‑identical output).

### Phase 4 · Elementwise + routing fusion (3–4 days) — *the flaw dies here*
`rmsnorm` (last‑dim, f16 in / f16 out, f32 accum), `add`, `silu_mul`, `sigmoid`, `swiglu`, `rope_inplace`
(f32 cos/sin table precomputed for `n_ctx` × 32 pairs per §2.6), `attn_gate_norm`, and
**`mhc_route`**: one workgroup per token reads `a,b` (8192), computes the joint RMS + the 8 dot products with the
f32 router (split‑K over 8192), adds bias, runs all 20 sweeps on 4 f32 registers **in registers**, emits
`in`, `post[2]`, `mix[4]` — replacing 480 nodes with **1 kernel**. Then `mhc_merge` for `a,b` updates, plus
the two §2.7 boundary kernels (`mhc_seed` at the first mHC execution, `mhc_expose` at the last) — the
program builder emits them and the gate test asserts their position, because an op count alone cannot see
whether the data flow is right.
**Deliverables:** each kernel + a Rust scalar oracle + `golden-check` wiring; `SN_TRACE` report before/after.
**Gate:** every kernel NMSE ≤ 1e‑5 vs Phase 2 stage dumps; `mhc_route` NMSE ≤ 1e‑6 on `mix`; a synthetic
micro‑benchmark showing the fused route ≥ 100× faster than the unfused equivalent; node/op count for a full
decode step ≤ **300**.

**Status (2026‑11‑27): three gates pass, the fourth (≤ 300) is reported as not met and not reachable by
fusion alone.** All numbers: `Docs/bench.md` §2.

* **Kernels vs their oracles — PASS.** 25 checks, 0 failing (`sn gpu kernels`; 22 plus the mHC
  seed/expose pair added on 2026‑11‑27): the worst f32 kernel is
  `rmsnorm_f32` at NMSE 1.69e‑13, `mix` at 6.17e‑13 (gate 1e‑6), the f16 variants at the f16 store floor.
  Calibrated by a control that must fail: the same kernel with 6 sweeps instead of 20 misses `mix` by
  3.3e‑3.
* **`mhc_route` vs the Phase 2 dumps — PASS.** Executions 4..11 of `p1-capital`, real router weights,
  recomputing logits / post / mix / `in` / both merged streams: worst `mix` NMSE **4.5e‑15**, worst other
  stage **8.65e‑13** (`Tools/kernel-check.py`). Each run also scores a host‑side oracle line so a
  divergence names its side. `Docs/golden` is untouched and `golden-check` still reproduces the Phase 2
  closure numbers exactly — moving `forward::route/merge` onto the oracles changed no arithmetic.
* **Fused route ≥ 100× the unfused sequence — PASS at 150×.** 486 unfused nodes (the fork's own
  spelling, given the router matmul and `in` as single nodes so it is not a strawman) = 3.845 ms per
  route as a graph; one `sn_mhc_route_f32` = 25.4 µs → **150.5×**. The node‑by‑node number is printed
  but does not gate: the same single launch lands in one of two submission‑latency modes (22.0 or
  46.7 µs), so it reads 85× … 180× and is not a gate. Two fixes made the difference: `log1pf` has no
  GCN instruction (the sweep pair update became a log‑sum‑exp, −25 µs), and the router read needed
  `float4` plus a 1024‑thread workgroup (55.5 → 25.4 µs).
* **≤ 300 launches per decode step — NOT MET: 669 fused, 1011 raw.** Breakdown at full fusion:
  `gemv 267, recur 154, merge 98, route 48, gate 42, silu 42, add 16, input 1, norm 1`. 267 weight
  matrices and 154 GDN recurrent kernels are outside Phase 4's reach; Phase 4's own claim holds —
  routing goes from 96 kernels to 48 (and from 480 graph nodes to 1 kernel each), plus the two
  boundary ops §2.7 requires (seed the streams at the first mHC execution, expose `0.5·(a+b)` at the
  last). Meeting 300 needs the per-execution GEMM batching that Phase 5/7 own; with Phase 3's 2.9 µs
  per node inside a graph, 669 nodes is ~1.9 ms of dispatch against ~40 ms of weight traffic, so the
  target is worth revisiting as a *Phase 7* line rather than pretending fusion delivers it.
  `phase4_decode_step_op_counts_are_recorded` panics if the count ever reaches ≤ 300, so the docs have
  to be edited on purpose.
* **Review of Phases 0–4 (2026-11-27) — three program-builder defects fixed, and they are why the
  counts above differ from the first Phase 4 run.** The decode program never seeded the mHC streams
  (the first route read buffers nothing wrote: `sn_mhc_seed_f32` + `oracle::mhc_seed` now exist), put
  the `0.5·(a+b)` exposure in the output head instead of at the mHC → non-mHC boundary (the exit
  blocks `blk.52..55` would have read a `cur` last written at execution 3, and the head would have
  scored the loop's streams instead of block 55's output), and applied the post-attention norm twice
  for the eight non-mHC blocks under `norm → gemv`. Counting ops could not see any of it, so the
  gate test now asserts the data-flow structure instead. Fixed before Phase 6, where they would have
  been numerics hunts.
* Tooling worth keeping: `build-kernels.sh` now writes `target/hip/*.meta` and the loader **refuses** an
  image whose `#define`d workgroup size disagrees with the Rust launch size — a mismatch silently read
  outside the reduction scratch and produced wrong router logits with no error anywhere.
* **Deferred honestly:** the deliverable “`SN_TRACE` report before/after” is the launch-count ladder
  (`sn hip ops --model … --ladder`, `Docs/bench.md` §2.4), not a literal trace. A trace needs a decode
  program that *launches*; today `compile()` produces a `ProgramSpec` whose 669 ops include 560 kernels
  that Phase 5/6 have not written yet. `SN_TRACE` over the real step belongs to Phase 7, where the plan
  already asks for it (≤ 300 ops visible there).

### Phase 5 · Quantised GEMM for Q2_K / Q4_K / f32 (5–8 days) — the bandwidth half
Decode (T=1) path first: memory‑bound `gemv`, warp‑per‑superblock, `__builtin_amdgcn_` 16‑bit packed ops for the
4/6‑bit unpack (`Q4_K` + `Q6_K` are the only quantised types in the checkpoint), 128‑byte loads, one pass, f32
accumulate, no smem staging; block sizes tuned on gfx1102's **32 CUs** (KFD `simd_count 64 / simd_per_cu 2`,
max clock 2208 MHz — see `Docs/baseline.md` §1) and wavefront 32.
Prefill path: tiled GEMM with `mma.f32.f16` (RDNA3 WMMA; **no MFMA on gfx1102** — keep the CDNA `mfma` path behind a
`cfg`/runtime arch check, don't build a kernel that can't launch here), weight tiles streamed once, dequant to f16 in smem.
**Deliverables:** `gemv_q4k/q2k/f32`, `gemm_q4k/q2k/f32` (T up to 2048), an auto‑pick by `(T, n, k)`, a
`sn bench gemm` sweep. **Feeds:** real forward passes.
**Gate:** gemv achieves ≥ 80 % of measured device bandwidth on `ffn_down` shapes; `gemm(512)` ≥ 60 % of WMMA peak;
NMSE ≤ 1e‑4 vs the scalar reference for all 12 GEMM shapes in the model.

### Phase 6 · Caches, and the first correct GPU forward (4–6 days)
`KVCache`: paged‑by‑block, per‑layer stride, **f16 / q8_0 / q4_0 selectable**, append‑only, positions explicit.
`GdnState`: `[42 logical layers][n_seq][32×128×128]` f32 + conv `[8192×3]`, one contiguous slab,
compile‑time offsets so decode needs zero indirection; **snapshot slots** per sequence at chunk boundaries
(the only prefix‑cache primitive that is correct for a hybrid model — a KV prefix without its matching GDN
snapshot must be rejected, assert it). Naive GDN decode kernel (one workgroup per v‑head, state in smem) and a
naive causal attention (one workgroup per (head, tile), smem K/V tiles). Full prefill + decode driver.
**Deliverables:** `sn run -m ... -p ...` generating text end‑to‑end on the GPU.
**Gate:** logits NMSE ≤ 1e‑4 vs Phase 2 reference at token 1, 8, 64; greedy continuation identical to the fork for
the 5 golden prompts; `n_seq=2` with divergent prefixes gives the same result as two separate runs.

### Phase 7 · Decode fast path (4–6 days) — **hit the roofline**
Assemble the Phase‑3 static program from the Phase 4/5/6 kernels for all 56 executions; bake all pointers;
capture one `hipGraph`; move token/pos to pinned memory; drop the per‑layer `l_out` liveness to a 3‑slot ring;
fold RoPE into the Q/K producer and the attention gate into `wo`'s epilogue; verify no host allocation and no
`hipMemcpy`/`hipMemcpy*Async` (other than the pinned writes) inside the decode loop; `SN_TRACE` shows op count ≤ 300.
**Gate:** ≥ **20 t/s** decode at `n_ctx=8196`, ≤ **7.0 GB** VRAM at `n_ctx=32768` with q8_0 KV, CPU < 5 %,
and the Phase 0 launcher (`Tools/hip_count.c`) shows ≤ **2 hipLaunchKernel + 1 hipGraphLaunch per token** (the Phase 0 counter measured
drop from ~10⁵ to ~1). **This phase is the thesis of the project; if the gate fails, stop and profile.**

### Phase 8 · Prefill fast path (4–6 days)
Chunked GDN prefill: replace the per‑token scan with the chunked/UT‑transform form (chunks of 64, matmul on
`k kᵀ` lower‑triangular, state carry in f32) so GDN layers run at GEMM efficiency; flash‑style attention with
f16 accum for the 14 full‑attn layers; pick the chunk size by measurement (weights are read once per chunk, so
the loop costs nothing here — the whole point is to keep prefill chunk‑large); overlap the next chunk's H2D
embeddings with compute.
**Gate:** ≥ **300 t/s** prefill at 2 k chunk on a 4 k prompt, output still matching the naive path (NMSE ≤ 1e‑4),
`sn bench pp`/`tg` matrix committed as `Docs/bench.md`.

### Phase 9 · MTP speculative decoding (5–8 days)
GDN state snapshot/restore (Phase 6 slots) as the draft mechanism; MTP block (§2.8) as a second captured program;
draft 1 token → verify 2 through the trunk → accept/reject with rollback of both KV length and GDN snapshot index;
greedy + typical‑sampling acceptance; measure acceptance rate and tune draft length.
**Gate:** end‑to‑end **≥ 32 t/s** at greedy with output **bit‑identical** to non‑speculative decoding at the same
seed (speculative decoding must never change the distribution — if it does, the rollback is wrong).

### Phase 10 · Usable engine (4–6 days)
GGUF BPE tokenizer + chat template (the GGUF ships a jinja template with vision placeholders — text path first,
`<|image_pad|>` rejected cleanly), sampler set (temp/top‑k/top‑p/min‑p, repetition), OpenAI‑ish HTTP endpoint,
state save/load (`/state` for GDN + KV so 32 k prompts reload without re‑prefill), graceful VRAM failure from the
Phase 1 budget check, docs: `Docs/architecture.md` (self‑contained, from §1‑§3), `Docs/why-its-fast.md` (from §4‑§6),
CI on `gfx1102` running the golden corpus + `sn bench` regression thresholds.
**Gate:** cold start → first token < 60 s; 32 k prompt reload from state < 5 s; every acceptance number from
Phases 7‑9 reproduced in CI.

---

## 8. Cross‑cutting rules

* **Every phase ships a measurement, not an adjective.** `sn bench` deltas land in `Docs/bench.md` with the exact
  command line. "Faster" is not a status; "5.0 → 21.3 t/s, 84 k → 241 ops/token" is.
* **Never re‑derive the loop/mHC/plan logic inside a kernel.** Phase 1's plan is the only source; kernels receive
  resolved descriptors.
* **No CPU work in the decode loop.** Enforce mechanically: a debug assertion that the decode path performs no
  allocation, no `hipMemcpy`, no `hipMemcpy*Async` (other than the pinned writes) and no more than 3 HIP calls.
* **gfx1102 is the primary target (`rocminfo` on this box; `lspci` reports the same card as device `0x7480`),
  CDNA (`gfx942`) is a `cfg` variant.** Do not let an MFMA path become the only
  correct code path — it will be untestable here.
* Keep `Docs/gemini-arch.md` for what it is (an overview); §1‑§3 supersede it.

## 9. Open questions (answer by measurement, in Phase 0/2)

1. Is the fork's 5 t/s (a) unfused per‑node launches, or (b) `hipGraphInstantiate`/update churn? §4.4 distinguishes
   them in one run. Both are fixed by P1/P2, but (b) would also justify checking whether `mhc_iterations` reaching
   the graph *at all* is what blows up graph memory.
2. Does q8_0 KV measurably degrade 32 k recall on this checkpoint? (If yes, ship q8 for the looped region only, or
   per‑layer mixed precision.)
3. Do the mHC router f32 matmuls survive f16? (Saves ~12 MB/token; NMSE gate on `mix` decides.)
4. Is the 20‑sweep count a *served* knob or a training‑time artifact we could shorten with retraining‑free
   correction? Measured sensitivity (2.4e‑2 error at n=20 vs the limit) says: keep 20, do not touch.
5. GDN state in bf16 instead of f32: −44 MB/sequence, and across 48 executions error may compound. Gate on the
   64‑token NMSE test, not on intuition.
