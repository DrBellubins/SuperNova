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
raw (one kernel per step of §2):  1011 launches ( 463 launchable today)
+ norm -> gemv:   899 launches ( 351)  -112
+ silu -> gemv:   843 launches ( 295)  -56
+ multi-gemv:   759 launches ( 295)  -84
+ rope -> gemv:   731 launches ( 267)  -28
+ gate -> gemv:   717 launches ( 253)  -14
+ route+merge (= P2 target):   669 launches ( 109)  -48
```

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
