# SuperNova — SenseNova‑Loop‑9B inference engine: architecture findings & phased implementation plan

Author: pi · Date: 2026‑11‑26 · Model: `SenseNova-Loop-9B-Q4_K_M.gguf` · Target: ROCm/HIP, gfx1103 (RX 7700S, 8 GB)

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
vocab 248320 (token_embd Q2_K, output Q4_K — two separate tensors, not tied)
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

Quant mix (exact, from offsets): FFN is mostly **Q2_K**, `attn_qkv_gate` is **Q4_K** for blocks 0‑2/28‑31 and
**Q2_K** for most looped blocks, `attn_q/k` Q2_K, `attn_v`/`ffn_down` (some) Q4_K, `output` Q4_K, `token_embd` Q2_K.
Routers, `ssm_conv1d`, all norms, `ssm_a`, `ssm_dt` are **f32**.

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
beta_raw  = w_beta_alpha[0:64]     @ x             # [32, T]
alpha_raw = w_beta_alpha[64:128]   @ x             # [32, T]     (one fused [4096,64] tensor)
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
Type is `IMROPE` with sections `[11,11,10,0]`, `n_rot = 64` of `head_dim = 256`, pairs are adjacent `(2i, 2i+1)`,
dims 64..255 pass through untouched. Section choice for pair index `p = 0..31`:
`p%3==0 → t`, `p%3==1 → h`, `p%3==2 → w`; the section only selects which **position** multiplies θ —
the **frequency index stays the global `p`** (`θ_p = pos · base^(−p/32)`). For text `p_t = p_h = p_w` ⇒ plain RoPE.

YaRN parameters as the reference resolves them (`scaling.factor` becomes a *reciprocal*):
```
freq_scale = 1 / 2.0 = 0.5      ext_factor = 1.0 (because scaling.type == yarn)
attn_factor = 1.0               beta_fast = 32   beta_slow = 1     n_ctx_orig = 262144
corr[0] = 64·ln(262144/(32·2π)) / (2·ln(1e7)) = 14.241    corr[1] = 64·ln(262144/(1·2π)) / (2·ln(1e7)) = 21.122
           # rope_yarn_corr_dims(n_rot=64, n_ctx_orig=262144, beta_fast=32, beta_slow=1, base=1e7)
ramp(p)   = clamp((p − corr[0]) / (corr[1] − corr[0]), 0, 1)
θ_p       = θ_interp·(1 − ramp) + θ_extrap·ramp,  θ_interp = 0.5·θ_extrap
mscale    = 1 + 0.1·ln(1/0.5) = 1.06931   ->  cos/sin both × mscale   (net ×1.143 on logits: keep it)
```
Because 4 positions are expected per token by the reference, a text‑only driver must still feed `pos` (×1) and may
skip the 3 unused copies. Do **not** silently switch to `norm` RoPE: with sections present the reference path differs.

### 2.7 mHC — two streams, two routes per layer (48 executions × 2 = 96 routes/token)
Streams `a, b ∈ R^[4096, T]`, seeded once at the first mHC execution (`il == 4`) with `a = b = cur`,
then carried across **all 48** looped executions. At the end of each mHC execution the exposed value is
`cur = 0.5·(a + b)`; the next execution ignores `cur` and re‑derives everything from `a, b`.

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

### 4.4 Phase 0 verification recipe (5 min, on the user's GPU — do this before writing kernels)
```bash
# 1) how many nodes does one decode token really cost?
llama-server -m .../SenseNova-Loop-9B-Q4_K_M.gguf -ngl 99 -c 4096 -v 2>&1 | grep sched_reserve
# 2) is HIP graph capture actually engaged?
GGML_CUDA_GRAPH_DEBUG=1 llama-server ... 2>&1 | grep -iE "cuda graph|warmup"
# 3) count real kernel launches for 32 decode tokens (expect ~1.5–2.7 million if the flaw is live)
rocprofv2 --hip-api -o /tmp/prof -- llama-cli -m ... -n 32 -p "hi" -ngl 99 -c 4096
grep -c "hipLaunchKernel" /tmp/prof/*.csv
```
If (3) shows ≳ 10⁵ launches/token, the diagnosis is confirmed and SuperNova's core design (below) is justified.
If (2) shows graphs *are* engaged and decode is still 5 t/s, the flaw is graph instantiate/update cost instead —
same conclusion, still requires a static small op list.

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
RX 7700S (gfx1103, 128-bit GDDR6): 256 GB/s spec, ~215 GB/s achievable
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
`mhc_merge`, `rmsnorm`, `rope_inplace`, `attn_gate_norm`, `gdn_step` (§2.4 in one kernel), `gdn_gate_norm`,
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
link `libhipblas` only if we ever need it — plan is to not need it). `gfx1103` arch flag. Target a static lib + CLI;
no `main` in kernels. `unsafe` confined to `sn-hip`.

---

## 7. Phases

Each phase ends green: `cargo test` + a named acceptance gate. Phases 1‑6 build the *correct* engine;
7‑9 make it *fast*; 10 makes it usable. Do not skip the gates — the later phases diff against earlier dumps.

### Phase 0 · Instrument the fork, freeze the targets (0.5 day)
*Run §4.4 on the GPU. Record node count, launches/token, graph status, t/s at `n_ctx ∈ {1k, 8k, 32k}`.*
**Feeds:** §5 becomes measured, not estimated; if the launch hypothesis is wrong we learn it for 30 minutes
instead of 3 weeks.
**Gate:** `Docs/baseline.md` with the numbers + the exact command lines.

### Phase 1 · GGUF reader, hyperparameters, execution plan (1–2 days)
`sn-core::gguf`: v2/v3 header, KV typed values, tensor info + exact byte sizes from offsets (handle `alignment`),
mmap + `madvise`. `sn-core::plan`: reproduce §1.1 — `physical(il)`, `is_recr(il)`, block‑index remap, mHC range
validity, the fork's validation rules (loop fits, mHC must not straddle a loop boundary, streams == 2,
`iterations ≤ 20`, MTP must be full‑attention). `sn-core::budget`: §3 table + a hard "will it fit" check.
**Deliverables:** `sn dump --plan` printing block table, per‑block exec count, per‑class bytes/token.
**Gate:** `Tools/roofline.py` and `sn dump --plan` agree to the byte on all 490 tensors; unit tests pin
`physical(0)=0, physical(4)=4, physical(27)=27, physical(28)=4, physical(51)=27, physical(52)=28, physical(55)=31`;
test that `blk.52..55` resolve to physical 28..31; test rejection of an mHC range that straddles the loop.

### Phase 2 · Scalar CPU reference + golden corpus (3–5 days) — *the spec*
Full f32 forward of §2 in plain Rust (weights dequantised on the fly), driven by token ids. Stage‑level
`cb()`‑style dumps: `embd`, per‑execution `l_out`, `a`,`b` streams at mHC boundaries, `mhc_logits[8]`,
`mix[4]`, `post[2]`, `gdn_state[:1,:]` before/after, `conv_state`, `q̃/k̃/ṽ`, `z`, `attn_out`, `h_nextn`.
**Deliverables:** `sn ref --prompt-file p.txt --dump golden/`, `Tools/golden-check.py` (NMSE + max‑abs per stage),
and — this is what makes every later phase cheap — a **fork cross‑check**: the same prompt fed to
`llama-server`'s `/tokenize` + `/completion` with `--log-verbose` logits dump, compared to `sn ref` logits.
**Gate:** `sn ref` top‑1 token matches the fork for 5 fixed prompts at greedy decoding, and `logit NMSE ≤ 1e‑4`
after the full prompt; `sn ref --iters 6` is *visibly different* (proving the 20‑sweep sensitivity is exercised).
Also emits `Docs/numerics.md` = §2 as tested code. **This phase is the one that de‑risks the whole project.**

### Phase 3 · HIP substrate + static decode program (4–6 days)
Device/context/queue on `gfx1103`; a single big allocation (one `hipMalloc` for weights, one for the arena, one for
caches); `Arenas` handing out compile‑time offsets; `Op` enum + `Program` builder + `hipGraph` capture/replay with
pinned‑host token/pos inputs; `sn-hip::launch!` macro and a `KernelRegistry` keyed by `(name, shape, dtype)`;
a `--no-graph` path for debugging and a `SN_TRACE=ops` mode printing per‑launch timings.
**Feeds:** everything after this writes kernels, not infrastructure.
**Gate:** replay a captured graph containing a trivial 1000‑op chain: ≥ 90 % of the theoretical launch‑free time,
host time per replay < 30 µs, and a repeated‑replay determinism check (bit‑identical output).

### Phase 4 · Elementwise + routing fusion (3–4 days) — *the flaw dies here*
`rmsnorm` (last‑dim, f16 in / f16 out, f32 accum), `add`, `silu_mul`, `sigmoid`, `swiglu`, `rope_inplace`
(f32 cos/sin table precomputed for `n_ctx` × 32 pairs per §2.6), `attn_gate_norm`, and
**`mhc_route`**: one workgroup per token reads `a,b` (8192), computes the joint RMS + the 8 dot products with the
f32 router (split‑K over 8192), adds bias, runs all 20 sweeps on 4 f32 registers **in registers**, emits
`in`, `post[2]`, `mix[4]` — replacing 480 nodes with **1 kernel**. Then `mhc_merge` for `a,b` updates.
**Deliverables:** each kernel + a Rust scalar oracle + `golden-check` wiring; `SN_TRACE` report before/after.
**Gate:** every kernel NMSE ≤ 1e‑5 vs Phase 2 stage dumps; `mhc_route` NMSE ≤ 1e‑6 on `mix`; a synthetic
micro‑benchmark showing the fused route ≥ 100× faster than the unfused equivalent; node/op count for a full
decode step ≤ **300**.

### Phase 5 · Quantised GEMM for Q2_K / Q4_K / f32 (5–8 days) — the bandwidth half
Decode (T=1) path first: memory‑bound `gemv`, warp‑per‑superblock, `__builtin_amdgcn_` 16‑bit packed ops for the
4/2‑bit unpack, 128‑byte loads, one pass, f32 accumulate, no smem staging; block sizes tuned on gfx1103's 32 CUs.
Prefill path: tiled GEMM with `mma.f32.f16` (RDNA3 WMMA; **no MFMA on gfx1103** — keep the CDNA `mfma` path behind a
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
and `rocprofv2 --hip-api` shows ≤ **2 hipLaunchKernel + 1 hipGraphLaunch per token** (the Phase 0 counter must
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
CI on `gfx1103` running the golden corpus + `sn bench` regression thresholds.
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
* **gfx1103 is the primary target, CDNA (`gfx942`) is a `cfg` variant.** Do not let an MFMA path become the only
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
