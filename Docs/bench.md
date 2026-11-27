# Benchmarks — measured, with the command that produced them

Every number here is produced by a command in this file on this host (RX 7700S, `gfx1102`, ROCm
10.1 / HIP 7.16). "Faster" is not a result; a delta with a command line is (design-plan §8).

## 1. Phase 3 — HIP substrate and static program gate (2026-11-26)

### 1.0 Device of record (`sn hip info`)

```
device 0  AMD Radeon RX 7700S
  isa                  gfx1102
  memory             8.573 GB total, 8.535 GB free
  CUs                16 reported, 16 physical   wavefront 32
  per block          1024 threads max, 131072 regs, 65536 B shared
  L2                 2097152 bytes
  clocks             SM 2208 MHz, mem native 1124 MHz x 128 bit (no bandwidth derived here)
```

Reproduce: `./target/release/sn hip info`. Two facts to keep in mind:

* **HIP reports 16 multiprocessors; `rocminfo` reports 40 CUs** for the same device
  (`Compute Unit : gfx1102`, `simd_count 64 / simd_per_cu 2` in `Docs/baseline.md` §0). Phase 5 must
  tune against a *measured* occupancy sweep, not either figure.
* The `MEMORY_CLOCK_RATE` attribute is the **native** clock (1124 MHz), not the effective data rate;
  no bandwidth is derived from it here. The usable figure stays `Docs/baseline.md`'s ~215 GB/s.
* ROCm 7 removed `hipDeviceAttributeGcnArch*`; the ISA string is read out of the fixed-ABI
  `hipGetDevicePropertiesR0600` buffer, and every attribute *value* used is pinned by
  `Tools/hip_attrs.c` rather than guessed:
  `gcc -w -D__HIP_PLATFORM_AMD__ -I/opt/rocm/include -o /tmp/attrs Tools/hip_attrs.c && /tmp/attrs`.

### 1.1 Build the kernel image, then run the gate

```sh
./Tools/build-kernels.sh                      # .hip -> target/hip/phase3.fatbin for every gfx reported
./target/release/sn bench graph --ops 1000 --reps 200
```

Gate result (`--ops 1000 --reps 200`, 1 step of dependent work per op):

```
graph gate: 1000 ops x 200 replays
  graph replay           2.8964 ms  ( 2896.4 ns/op)
  same ops, no graph     3.1980 ms  ( 3198.0 ns/op)  x1.1
  launch-free (1 op)     0.2305 ms
  graph / launch-free       8.0 %   (info: a 1-step op is pure dispatch)
  host per replay          6.53 us
  per-node device dispatch: graph 2896 ns, direct launch 3198 ns
  [PASS] host time per replay < 30 us 6.53 us
  [PASS] repeated replays are bit-identical
  [PASS] pinned input visible without graph update
GATE: PASS
```

| plan's criterion | measured | verdict |
|---|---|---|
| host time per replay < 30 µs | **6.5 µs** (5.6–8.8 µs across runs, `--steps-per-op` 1…64) | **pass**, 4.5× margin |
| repeated replays bit-identical | 32 replays of the 1000-op chain, byte-identical output | **pass** |

The same three assertions run as a test (`crates/sn-hip/tests/phase3.rs`, 200 ops × 60 replays, plus a
check that a captured graph still replays with its baked pointers after its capture stream is destroyed).
It skips with a printed reason when there is no HIP device or no built image, so `cargo test` stays
usable without ROCm:

```sh
./Tools/build-kernels.sh && cargo test -p sn-hip
# test phase3_graph_gate ... ok
# test captured_graph_survives_its_capture_stream ... ok
```
| ≥ 90 % of the theoretical launch-free time | 8 % at 1 step/op — see §1.2: with trivial ops that ratio measures *dispatch*, not the graph | measured, see below |

The mechanism P1 depends on is confirmed directly: after capturing a graph whose first node is a
pinned-host→device copy, writing a new `(token_id, position)` into the pinned slot and replaying the
**same** graph (no `hipGraphExec` update) yields the new values on the device, for
`(1,7) (42,4242) (0,0) (u32::MAX,1)`.

### 1.2 Why the ratio is 8 %, and what it means for Phase 5/7

A "trivial 1000-op chain" makes every node one dependent global read-modify-write. The work per node
is then ~230 ns (the launch-free reference — one kernel doing all 1000 steps — takes 0.23 ms, which
is exactly a dependent L2-latency chain), while a *node* costs ~2.9 µs. The ratio measures
`work / (work + dispatch)`, so for the trivial chain it is a property of the GPU's dispatch path, not
of SuperNova. Sweeping work per node shows the curve, and that curve is the design input:

```sh
for k in 1 4 16 64; do ./target/release/sn bench graph --ops 1000 --reps 100 --steps-per-op $k; done
```

| steps per op | graph (ns/op) | direct launch (ns/op) | graph faster by | graph / launch-free |
|---|---|---|---|---|
| 1 | 2 896 | 3 198 | 1.10× | 8.0 % |
| 4 | 3 077 | 3 838 | 1.25× | 27.7 % |
| 16 | 5 555 | 6 073 | 1.09× | 55.9 % |
| 64 | 14 840 | 22 649 | 1.53× | 82.6 % |

Reading, and what Phase 5/7 must respect:

1. **Per-node device dispatch on this GPU is ≈ 2.9 µs, and a graph does not remove it** — it removes
   the *host* cost (6.5 µs for the whole graph, versus 3.2 µs of host work per individual launch in
   the un-captured path) and it lets the device pipeline nodes (the 1.1–1.5× win above). The fork pays
   both costs: `Docs/baseline.md` §2 measured 4 700 `hipLaunchKernel` + 181 `hipGraphLaunch` +
   253 `hipMemcpyAsync` **per token**.
2. Design-plan §5 assumed "at 3 µs each, 250 launches = 0.75 ms = 2 % of budget". **That assumption is
   now measured: 2.9 µs/node.** So the ≤ 300-op budget for a decode step costs ≤ 0.87 ms of device
   dispatch against a ~40 ms weight-read budget — roughly 2 %. The budget is safe; it is not free, and
   fusing (Phase 4) is what keeps the *real* per-op work above the dispatch floor.
3. Any op that carries less than ~1 µs of device work is paying more to exist than to run. Phase 4's
   fusion targets (one kernel per block-level operation) and Phase 5's gemv shapes must stay on the
   right side of that line; a 2 048-operand gemv is not a trivial op.

### 1.3 The real checkpoint through the single weight allocation

```sh
./target/release/sn hip info --weights ~/Documents/AI/Models/SenseLoop/SenseNova-Loop-9B-Q4_K_M.gguf
```

```
weights uploaded: 5807658496 bytes in 12.97 s (0.45 GB/s) -> one allocation;
device now 2.569 GB free of 8.573 GB
  output.weight at device 0x7fd802800000 (base 0x7fd802800000 + 0 bytes)
```

So the whole model is one `hipMalloc` + one `hipMemcpy` of the GGUF data section, and every tensor
address is `base + gguf_offset` — no per-tensor allocation, no repacking. 0.45 GB/s is *slow* because
the source is the `mmap`'d file (pageable): Phase 6 should stage through pinned memory in chunks if
cold start matters (13 s of the plan's < 60 s cold-start budget). Note also that the weights alone
leave only 2.57 GB free on this 8 GB card: the §3 KV budget is not optional.

### 1.4 What Phase 3 is (files)

* `crates/sn-hip/src/hip.rs` — `dlopen`/`dlsym` bindings over `libamdhip64` (no `hipcc`, no bindgen,
  no build script), device open, attributes, error mapping.
* `crates/sn-hip/src/arena.rs` — `Arena`/`Layout`: named slots, compile-time offsets, alignment,
  duplicate/overlap/zero-length rejection, freeze-before-use.
* `crates/sn-hip/src/kernels.rs` — fat-binary loading and the `(name, shape, dtype)` registry
  (shape-generic fallback; a missing specialization is an error that names the key).
* `crates/sn-hip/src/program.rs` — `Op`/`Arg` with baked pointers, `ProgramBuilder`, the flat
  `--no-graph` launcher, stream capture → `hipGraphInstantiate` → one `hipGraphLaunch`, `SN_TRACE`,
  event `Timer`.
* `crates/sn-hip/src/bench.rs` — the gate above, as code (`sn bench graph`).
* `crates/sn-hip/src/lib.rs` — `Substrate`: device, stream, **three** slabs (weights = one
  `hipMalloc` + one `hipMemcpy` of `Gguf::data_section()`, so a tensor's device address is
  `base + gguf_offset`), pinned host I/O.
* `crates/sn-hip/kernels/phase3.hip` + `Tools/build-kernels.sh` — the kernel image recipe above.

Not yet done (deliberately): wiring a real model's forward through the program (that is Phase 6/7 —
it needs the kernels), and `hipGraphExec` node updates (never needed if inputs come in through pinned
memory, which §1.1 confirms works).

## 2. Phase 4 — elementwise + routing fusion (2026-11-27)

### 2.0 The four gates, as commands

```bash
./Tools/build-kernels.sh gfx1102                      # -> target/hip/{phase3,phase4}.fatbin (+ .meta)
./target/release/sn gpu kernels                       # every kernel vs its scalar oracle
Tools/kernel-check.py --dump /tmp/stage               # routing vs Phase 2 stage dumps (real weights)
./target/release/sn bench fusion --reps 200           # fused route vs the fork's 480-node balancing
./target/release/sn hip ops --model $M --ladder       # launches per decode step, fusion by fusion
cargo test --workspace                                # all of the above as tests (60 green)
```

`build-kernels.sh` output must stay visible. It failed silently once this phase (its output had been
redirected), the old image stayed in place, and the wrong-but-plausible numbers it produced cost most
of a debugging session — see §2.5.

### 2.1 Kernel vs oracle — 25 checks, 0 failing (`sn gpu kernels`)

Every kernel in `crates/sn-hip/kernels/phase4.hip` has a scalar twin in `sn-core/src/oracle.rs`; the
harness feeds both the same buffers and compares bit-for-bit-ish (NMSE over the whole output):

| kernel | NMSE | note |
| --- | --- | --- |
| `rmsnorm_f32` / `rmsnorm_f16` | 1.69e-13 / 6.82e-13 | f32 accumulate, `rms(x)*w` (no `(1+w)`) |
| `add_f32` / `add_f16` | 0 | |
| `silu_mul_f32` / `swiglu_f32` | 1.61e-18 | |
| `sigmoid_f32` | 1.07e-16 | |
| `rope_inplace_f32` / `_f16` | 2.93e-16 / 1.38e-8 | half-offset pairs `(p, p+n_rot/2)` (§2.6 of the plan) |
| `attn_gate_norm_f32` | 4.36e-19 | strided gate, 16 q-heads sharing a 4-head gate |
| `mhc_route_f32` logits / input / post / **mix** | 1.72e-16 / 6.25e-16 / 0 / **6.17e-13** | mix gate is 1e-6 |
| `mhc_merge_f32` a / b | 1.80e-15 / 1.95e-15 | reads pre-merge streams |
| `mhc_seed_f32` a / b, `mhc_expose_f32` cur | 0 / 0 / 0 | the two ends of the mHC region (§2.7); added with the program-boundary fix below |

The f16 variants sit at the f16 quantisation floor (4.88e-4 relative, `oracle::f16_rel`), which is the
only honest floor for a kernel whose job is to *store* f16.

**The check is calibrated with a control that must fail:** the same kernel run with 6 Sinkhorn sweeps
instead of the checkpoint's 20 misses `mix` by NMSE 3.3e-3. If that control ever passes, the mix check
has stopped measuring the sweeps.

### 2.2 Routing replayed against Phase 2 — GATE: PASS (`Tools/kernel-check.py`)

The synthetic oracle cannot prove the router is right about *this checkpoint*, so the kernels are also
run against `sn ref`'s stage dumps: the reference decodes `p1-capital`, dumps
`mhc.<il>.{a,b,logits,post,mix}`, `l.<il>.{h,h2}`, `l_out.<il>`, and the GPU recomputes each of them
from the dumped pre-merge streams plus the real `blk.N.hc_*_fn.weight/bias/norm` tensors.

```
executions 4..11 (all 8 trunk mHC executions in the dump)
worst NMSE: mix 4.502e-15 (gate 1e-6), every other stage 8.654e-13 (gate 1e-5)
GATE: PASS
```

Per execution that is 10 checks (attn logits/post/mix, `l.<il>.h`, ffn logits/mix, `l.<il>.h2`, both
merged streams, and a host-side oracle line). The oracle line exists to localise a failure: if the
*host* oracle misses the golden too, the harness read the wrong stage files; if only the kernel misses,
the defect is in the kernel with exactly those bytes.

*Note on the figures:* the stage dump is an **input** to this gate. For a fixed dump the GPU side is
bit-reproducible (`sn gpu route` twice → the same 1.854e-14 / 4.005e-13), but regenerating the dump
with `sn ref` moved these residuals from 4.5e-15 / 8.65e-13 to 1.85e-14 / 4.00e-13 (re-measured
2026-11-27 after the program-boundary fix). Both are ~9 orders of magnitude inside the gate; read these
numbers as magnitudes, not as a hash of the run.

`Docs/golden` is unchanged, and `Tools/golden-check.py --floor-dir Docs/golden/ref-cpu/ref` still
reports the Phase 2 closure numbers to three significant figures (1.79e-3 / 9.57e-3 / 8.87e-3 /
2.37e-3 / 4.85e-2, top-1 5/5): the refactor that moved `forward.rs::route/merge` onto the oracles is
arithmetic-neutral. The two prompts that fail that gate are §2.9's open attention item, unchanged.

### 2.3 Fusion — 486 nodes → 1 kernel, 150× (`sn bench fusion --reps 200`)

The unfused baseline is the fork's spelling of the same route (`sn_ub_*` nodes: `sub`, `neg`,
`softplus`, `gemv8`, `bias_add`, `sigmoid_post`, `exp4`, `mix_ab`) — 486 launches, including the whole
8-row router matmul and `in` given to it as single nodes so the claim is not flattered by a strawman:

| | launches | flat (ms/route) | graph (ms/route) |
| --- | --- | --- | --- |
| unfused | 486 | 3.955 | 3.845 |
| fused `sn_mhc_route_f32` | 1 | 0.0219 | 0.0254 |
| **speedup** | | 84.6× … 180.5× | **150.5× … 151.1×** |

The graph leg is what the plan's ≥ 100× is tested against, because Phase 3 established the engine
replays captured graphs. The node-by-node number is printed but does not gate: it is bimodal (22.0 µs
or 46.7 µs for the same single launch, stable within a process), and a timing gate that depends on
which mode a process lands in is not a gate. The device time is the graph number, 25.4 µs.

At 96 routing applications per decode step that is 2.4 ms of device time per token (6 % of the ~40 ms
weight-traffic budget), against 96 × 3.85 ms = 370 ms unfused — before counting the host work the fork
spends building those nodes, which is what §4.2 of the plan measured at 4,700 launches/token.

### 2.4 Launches per decode step — the ≤ 300 target is NOT met (`sn hip ops --ladder`)

The plan asks for ≤ 300 nodes per decode step. With every Phase 4 fusion turned on, the real checkpoint
still needs **669** (raw, one kernel per step of §2: **1011**):

```
raw (one kernel per step of §2):  1011 launches ( 814 launchable today)
+ norm -> gemv:   899 launches ( 702)  -112
+ silu -> gemv:   843 launches ( 646)  -56
+ multi-gemv:   759 launches ( 492)  -84
+ rope -> gemv:   731 launches ( 464)  -28
+ gate -> gemv:   717 launches ( 450)  -14
+ route+merge (= P2 target):   669 launches ( 306)  -48
```

*(Numbers re-taken after **Phase 5** on 2026-11-27; the launchable column is what changed, the op
counts are the same. Phase 4 recorded 463 / 109. Every `gemv_q4_k`, `gemv_q6_k`, `gemv_f32` node is
now a real launch with a geometry derived from `gemm::pick_warps` (§3.3), which removes 351 pending
names from the raw program and 197 from the fully fused one — 70 of the fully fused program's GEMV
nodes are `gemv_multi_qkv` / `gemv_multi_gate_up`, fusions that reduce the *count* but are not
implemented, and `tests/phase5.rs` asserts that those are the only GEMV names still pending. What
remains pending is 154 GDN recurrent kernels and the Phase 6 elementwise set: `rope_tables`,
`row_gather_f32`, `causal_index`, `norm_bands`, `rope_bands`, `gdn_gate_norm`.)*

and 669 breaks down as `gemv 267, recur 154, merge 98, route 48, gate 42, silu 42, add 16, input 1,
norm 1` — i.e. 267 weight matrices (irreducible: one per matrix, and `multi_gemv` already batched what
shares an input), 154 GDN recurrent inner kernels (Phase 6 fuses them per head), and 146 mHC
route/merge/seed/expose kernels (96 routing applications before fusing, plus the two boundary ops §2.7
requires: one seed at the first mHC execution, one exposure at the last). Phase 4 *halves* the routing
kernels (96 → 48 by folding the ffn route into its merge); nothing in Phase 4 can touch the other 421.

*(2026-11-27, review of Phases 0–4: three defects in the program builder were fixed and are inside the
numbers above. Nothing wrote `a`/`b` before the first route read them — `sn_mhc_seed_f32` does now, with
`oracle::mhc_seed` beside it; the `0.5*(a+b)` exposure sat in the **head** instead of the mHC → non-mHC
boundary, which would have left the exit blocks (`blk.52..55`) reading a `cur` last written at execution 3
and fed the head the loop's streams instead of block 55's output — it is emitted at the boundary now, and
`sn_mhc_expose_f32` is a real kernel rather than a pending name; and the fused ladder launched the
post-attention norm *and* folded it into the gate/up GEMV for the eight non-mHC blocks (the `-104` row
is now `-112`, and `norm 9` is now `norm 1`, the output norm). Those numbers are the −8 +2 between the
previous 676/1010 and these. `phase4_decode_step_op_counts_are_recorded` now asserts the structure —
exactly one seed, before the first route; exactly one exposure, before the exit blocks; no explicit
post-norm in a program that folds it — because counting ops alone did not see any of this.*

So the ≤ 300 figure is reported as **not met and not reachable by fusion alone**: it needs the GEMM
batching that Phase 5/7 own (one batched launch per execution rather than per matrix). Phase 3 set the
exchange rate that makes this survivable — 2.9 µs per node inside a graph, so 669 nodes ≈ 1.9 ms of
dispatch against ~40 ms of weight traffic. `phase4_decode_step_op_counts_are_recorded` asserts the
measured structure (fused < raw, routing halved, ≥ 96 applications, the two §2.7 boundaries in the right
order) and *panics if the count ever drops to ≤ 300*, so the claim has to be edited deliberately rather
than drifting.

### 2.5 Two kernel measurements, and one bug that produced wrong numbers silently

* **`log1pf` has no GCN instruction.** The Sinkhorn sweep in `sn_mhc_route_f32` was `-softplus(±d)`
  spelled as the reference spells it (`x > 20 ? x : log1pf(expf(x))`): 80 software calls per thread in
  series = 25 µs of the kernel's runtime. The same renormalisation written as a log-sum-exp pair update
  — `m_i' = m_i − lse(m_i, m_j)` with `lse = max + logf(1 + expf(−|d|))` — is one `exp` and one `log`
  per pair instead of two of each, and *improved* accuracy against the oracle (mix NMSE 2.9e-17 →
  6.2e-13 with the float4 accumulation below; the unfused baseline keeps `log1pf` because that is what
  the fork runs).
* **The route is memory-latency bound, and only one workgroup runs.** Reading the `8×8192` f32 router
  (256 KB) with 256 scalar-loading threads ran at ~8 GB/s. `float4` loads of `a`, `b` and the router,
  and a 1024-thread workgroup (`SN_ROUTE_THREADS`, mirrored as `ops::ROUTE_BLOCK`), took one route from
  55.5 µs to 25.4 µs (2.2×), which is what turned 82× into 150×.
* **The bug: a stale kernel image and a workgroup-size mismatch.** The scratch of `sn_mhc_route_f32` is
  `__shared__ float red[SN_NROUTE * SN_ROUTE_THREADS]`, indexed by the *compiled* stride while the
  launcher picks the *runtime* workgroup size. With the fatbin built for one size and Rust launching
  another, the reduction reads outside `red`: every router logit but the first came back wrong, the
  numbers moved when the `sharedBytes` launch argument changed, and nothing errored. `Tools/build-kernels.sh`
  now records `sn_block` / `sn_route_threads` next to each image (`target/hip/*.meta`) and
  `KernelRegistry::add_image` refuses to load an image whose constants disagree with the Rust side.
  Passing the static footprint as a *dynamic* shared size was also wrong (it reserves a second 32 KB
  region); every route launch passes 0 — `decode::route_static_shared()` documents the footprint instead.

### 2.6 What Phase 4 is (files)

* `crates/sn-core/src/oracle.rs` — the scalar twins, `nmse`/`max_abs`/`f16_rel`, `read_stage`, and the
  single implementation of routing (`route_derive`, `mhc_route`, `mhc_merge`) that `forward.rs` now
  calls, so the reference and the kernels cannot drift apart.
* `crates/sn-hip/kernels/phase4.hip` — the engine kernels + the `sn_ub_*` unfused baseline.
* `crates/sn-hip/src/ops.rs` — `SPECS` (name, dtype, workgroup), `register_phase4`, `Kernels::resolve`,
  `grid1d`; `BLOCK = 256`, `ROUTE_BLOCK = 1024`.
* `crates/sn-hip/src/decode.rs` — `compile()` → `ProgramSpec` (the op list of one decode step, with
  `Fusion` flags and the ladder), `Slots::reserve`, and the named *pending* kernels for Phase 5/6 so the
  counts above include kernels that do not exist yet.
* `crates/sn-hip/src/check.rs` — `Harness`, `kernel_checks`, `route_vs_golden`, `fusion_gate`, `report`.
* `crates/sn-cli/src/gpu.rs` — `sn gpu kernels`, `sn gpu route`, `sn bench fusion`, `sn hip ops`.
* `crates/sn-hip/tests/{phase4.rs,phase4_fusion.rs,common/mod.rs}`, `Tools/kernel-check.py`.

---

## 3. Phase 5 — the quantised matmul surface (2026-11-27)

### 3.0 The commands

```sh
./Tools/build-kernels.sh                                 # crates/sn-hip/kernels/phase5.hip -> target/hip/phase5.fatbin
./target/release/sn gpu gemm  --model $M                 # correctness of every shape in the checkpoint
./target/release/sn bench gemm --model $M [--reps 10] [--trials 3] [--gemm-t 8,64,256,512] [--read-mb 256]
cargo test -p sn-hip --test phase5                       # the same correctness gate as a test
```

`supernova` in the scratchpad sense: both commands read the **checkpoint's own weight bytes** into a
device slab and multiply *those*, so the numbers belong to the model's shapes and its real
quantisation mix (design-plan §1.2), not to a fixture. `sn gpu gemm` uploaded 1 648 361 472 B of
weights in 0.19 s (8.7 GB/s, PCIe-bound — Phase 6 should stage in pinned chunks, same note as §1.3).

The 12 GEMM roles became **15 shape/type entries**, because four roles use two quantisation classes
across blocks (`attn_qkv_gate`, `ffn_down`, `attn_v` are Q4_K on some blocks and Q6_K on others).
Both classes are gated; neither kernel is validated by the other. `nextn.eh_proj` is a 13th shape and
belongs to Phase 9. Weight traffic through these shapes: **1 707 343 872 B per decoded token**
(834 MB of it the Q6_K output head, 572 MB the Q4_K embedding — which decode *gathers*, so it is
checked as a matmul but never timed as one).

### 3.1 Correctness — 47 checks, 0 failing, worst NMSE 4.12e-12 (`sn gpu gemm`)

Gate 1e-4, measured 4.12e-12 (`gemm … ffn_down … T=512`); the GEMV worst is 5.88e-13. That is f32
summation order and nothing else — the reference is `sn_core::dq` (the reader Phase 2 matched
against GGML's C with `Tools/dequant_ref.py`, §2.9) run over the same bytes.

Covered: every shape × {plain, `norm -> gemv` folded} = 30 GEMV checks; `ffn_down` and
`attn_qkv_gate` × both classes × T ∈ {8, 64, 512} = 12 GEMM checks (at T = 512 the scalar reference
re-dequantises the matrix per activation column, so every 8th column is scored — a stride that still
lands in every T tile of the kernel's grid; T = 8 and 64 are scored in full); and 5 checks for the
**f32** paths against a synthetic matrix, because §1.2's census says this checkpoint has no f32 2-D
weights — an untested kernel is a worse answer than a synthetic test.

### 3.2 Measured device references, and GEMV bandwidth

```
read bandwidth   280.2–281.8 GB/s over 268435456 B      (sn_read_bw, float4, ≫ the 32 MB L2)
f32 FMA peak     8.0–8.2 TFLOP/s                        (8 independent FMAs per thread)
```

The read reference is at the interface's spec for this card (128-bit GDDR6 at 18 Gbps ≈ 288 GB/s),
which is what makes it a usable denominator. `gemv >= 80 % of measured device bandwidth on
ffn_down shapes` — reported **per quantisation class**, because the two classes are limited by
different things and one blended number hid it:

| class | ffn_down GEMV | % of measured bandwidth | verdict |
|---|---|---|---|
| Q4_K | 222.6 – 239.4 GB/s | 79.4 – 84.9 % | at the gate, not reliably over it |
| Q6_K | 101.7 – 108.0 GB/s | 36.3 – 37.9 % | **NOT met** |

(the spread is two runs of the same command; §2.3's bimodal clock states are real and move a shape
by ~7 %, which is also why 79.4 % and 84.9 % are the same result.)

Per shape, best measured (`--reps 10 --trials 3`, GB/s at the picked geometry): Q4_K
`ffn_down` 239 / `attn_qkv_gate` 231 / `ffn_gate` 227 / `attn_q` 217 / `ssm_out` 191 / `attn_k` 148 /
`ssm_beta_alpha` 25 (64 rows: 25 GB/s is 99 % of what 64 rows can move, see §3.3); Q6_K
`attn_v` 114 / `output` 108 / `ffn_down` 108 / `attn_qkv_gate` 107.

**Why Q4_K got there and Q6_K did not.** The first Q4_K kernel read one `uint` (4 B) per lane per
super-block and measured 87–91 GB/s; its *per-element* rate matched Q6_K's despite carrying 1.46×
more values per byte, which says the limit was load instructions and memory latency, not DRAM. Two
changes took it to 222–244 GB/s: a wavefront loading **four super-blocks per iteration** with a
`uint4` per lane (512 B, 32 row values per lane instead of 8, one load instead of six), and
`float4` activation loads instead of scalar ones. Correctness was re-gated after each step.

Q6_K cannot use that path in the current staging: a Q6_K super-block is **210 bytes**, so successive
rows start at offsets that are only 2-byte aligned and no field inside a row is vector-aligned. The
attempt produced wrong answers (NMSE 4e16) rather than a fault — unaligned vector loads on GCN are
undefined, not checked. The fix is a *staging* decision, not a kernel one: lay Q6_K super-blocks out
padded to 224 B in the weight arena (Phase 6), which leaves every field at its existing offset and
costs 6.7 % of the Q6_K bytes at upload time. Recorded as the follow-up rather than done here,
because it changes the contract `sn hip info --weights` documents in §1.3 (tensor address =
base + gguf offset) and would silently break the decode program's baked weight addresses.

### 3.3 The auto-pick, and the sweep that checks it

`gemm::pick_warps(n, k, cus)` splits one output row across `warps_per_row` wavefronts so a narrow
matrix can still fill the machine (`n · wpr ≥ 16 · CU` wavefronts, capped by the K axis and by
`SN_GEMV_BLOCK/32 = 8`), and `pick_path` sends `T = 1` to the GEMV, `T ≥ 8` to the tiled GEMM, and
1 < T < 8 to the GEMV per token. `sn bench gemm` prints, for every shape, the whole sweep over
wpr ∈ {1,2,4,8} next to the pick:

* wide shapes want `wpr = 1` and the policy picks it — for Q4_K the sweep's own ordering is
  1 > 2 > 4 > 8 (e.g. ffn_down 223 / 194 / 204 / 173), so splitting a 4096-row matrix only adds a
  shared-memory reduce;
* `ssm_beta_alpha` (64 rows) is the opposite: 15.8 GB/s at wpr=1 → 15.8–18 at wpr=4/8, and the
  policy picks 4. 25 GB/s looks like a failure until you divide: 147 KB of weights is *below the
  launch cost*, and this tensor is read once per token in a 1.7 GB/token loop — it is 0.009 % of the
  traffic. The honest claim is "the narrow tensor is no longer a stall", not "it is fast";
* 3 of 14 shapes measured a wpr the policy does not pick by more than 5 % (e.g. `ssm_out` 166 vs
  180 at wpr=2) — within the run-to-run spread of §2.3, but not evidence that the policy is
  optimal. The gate "the pick is the sweep's best" is therefore reported as **not met**, with the
  differences listed rather than averaged away.

### 3.4 Prefill GEMM — 2.6 TFLOP/s = 34 % of the measured FMA peak

The tiled kernel is 64×64×32, dequantising weights into shared memory and accumulating with **plain
f32 FMAs** (there is no f16 MMA path in this phase; see §3.5). At the sizes the plan's gate names:

| shape | T | GFLOP | ms | GFLOP/s | % of measured FMA peak |
|---|---|---|---|---|---|
| `ffn_gate` 12288×4096 | 512 | 51.5 | 19.6 | 2 631 | 32.1 % |
| `ffn_down` 4096×12288 | 512 | 51.5 | 20.6 | 2 505 | 31.3 % |
| `attn_qkv_gate` 12288×4096 | 512 | 51.5 | 20.0 | 2 576 | 32.2 % |
| `attn_qkv_gate` [q6_k] | 512 | 51.5 | 23.3 | 2 216 | 27.0 % |
| `attn_k` 1024×4096 | 256 | 2.1 | 0.78 | 2 763 | 34.5 % |

`gemm(512) >= 60 % of WMMA peak`: **NOT EVALUABLE** (§3.5). Against the reference this kernel is
actually written against — the measured 8.0–8.2 TFLOP/s f32 FMA peak — it reaches **34.2 %**, so the
60 % gate is not met either way. What the numbers do show is that the kernel is compute-bound above
T ≈ 64 (1.9–2.8 TFLOP/s flat across 64/256/512, i.e. weight traffic is no longer the limit) and that
the dequantise-into-shared-memory fill is where the remaining factor sits: a Q4_K fill decodes one
value per (row, k) index, which is the right thing for correctness and the wrong thing for
throughput. Phase 8 should replace the fill with a block-dequant that decodes 16 values per load,
the same step that took the GEMV from 90 to 230 GB/s.

### 3.5 The WMMA probe measured something, but it is not a peak

`__builtin_amdgcn_wmma_f32_16x16x16_f16_w32` compiles for gfx1102 and runs. It is also gated on
`wmma-256b-insts` + wavefrontsize32, so the same image must guard it (`#if defined(__gfx11__)`) —
this box builds the fatbin for gfx1035 too, and the unguarded builtin fails to compile for it. The
dependent-chain probe reports 1.7–3.2e16 FLOP/s, which implies **4.8e-4 ns of device time per
16×16×16 op per wavefront** — under one cycle at any clock this card runs at, so the loop is not
executing one matrix operation per iteration in any sense worth dividing by. The report and the gate
say `NOT EVALUABLE` rather than quoting 17 PFLOP/s as a machine peak. Phase 5 therefore has **no
f16 MMA path**; it needs a real tiled WMMA microkernel whose output is correct first (the fragment
layout probes in the planning session did not match the oracle, NMSE 1.9–2.4) and whose throughput is
measured second. Until then the GEMM's reference is the measured f32 FMA peak above.

### 3.6 Two bugs that produced wrong numbers silently

1. **A two-stage reduce with the first stage missing.** `sn_row_group_sum` wrote only lane 0's
   partial to shared memory, so a row's answer was "whatever lane 0 read": every shape returned
   NMSE ≈ 1.0, and the tell was a one-hot activation — only indices ≡ 0…3 (mod 32) came back
   non-zero, which is exactly lane 0's slice. Fix: fold the wavefront (`sn_wave_sum`) *before* the
   shared-memory step. Same class as §2.5: a wrong value with plausible magnitude and no error.
2. **A shared-memory tile with the wrong pitch.** The activation tile is 64 tokens wide but was
   declared with the 33-float pitch of the 32-wide weight tile, so each row wrote into its
   neighbour's. Tell: T = 8 was *partly* right (NMSE 6e-3) while T = 64/512 were not (≈0.97) — a
   small T leaves most of the tile unexercised, so a small-T-only test would have passed. Two
   pitches (`SN_PADK 33`, `SN_PADT 65`) and both are exercised at every T.

Debugging tools that paid for themselves and stayed in the image (only run under `SN_DEBUG`):
`sn_dbg_q4k_lane` (per-lane partials, one row, no split — shows *which* lane owns which element) and
`sn_dbg_args` (writes its own arguments back, which is how "the ABI is fine, the kernel isn't" was
established instead of assumed). `SN_GEMM_ONLY=<role>` and `SN_WPR=<n>` narrow the check harness to
one shape and one geometry.

### 3.7 What Phase 5 is (files)

* `crates/sn-hip/kernels/phase5.hip` — `sn_gemv_{q4_k,q6_k,f32}` (wide-load Q4_K, byte-load Q6_K,
  folded-norm variant), `sn_gemm_{q4_k,q6_k,f32}` (64×64×32 tiled, shared-memory dequant), the
  `sn_read_bw` / `sn_peak_fma_f32` / `sn_peak_wmma` references, and the two debug probes.
  `SN_GEMV_BLOCK 256` is recorded in `target/hip/phase5.meta` and `KernelRegistry::add_image`
  refuses an image whose value disagrees with `ops::GEMV_BLOCK` (the §2.5 guard, extended).
* `crates/sn-hip/src/gemm.rs` — the shape table read out of the GGUF (`model_shapes`), the picks
  (`pick_warps`, `pick_path`, `gemv_grid`), the correctness harness (`gemm_checks`, `f32_checks`),
  the bandwidth sweep and throughput tables, the gates and the report, the policy unit tests.
* `decode.rs` no longer reports `gemv_q4_k` / `gemv_q6_k` / `gemv_f32` as pending: the decode step's
  weight multiplications are launchable, with `warps_per_row` as an argument and the grid derived
  from the same policy (asserted in `tests/phase5.rs`).
* `sn gpu gemm`, `sn bench gemm` (`sn-cli`).
