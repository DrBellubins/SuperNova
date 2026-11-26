# Phase 0 baseline — the reference engine, measured on this host

Purpose (Docs/design-plan.md §4.4): establish, on this machine, what the llama.cpp fork actually costs per
decoded token, so the Phase 8/10 gates ("≤ 2 `hipLaunchKernel`/token", "≥ 20 t/s", "graph nodes 46,080 → 0")
compare against measured facts rather than prose. Everything below was produced by `Tools/phase0.sh` on
2026-11-26; the exact command lines are in the tool's own output (`$OUT/commands.txt`), and every number is
reproducible from them.

## 0. Environment of record

| Item | Measured value |
|---|---|
| GPU (compute) | `AMD Radeon RX 7700S`, PCI `1002:7480` (Navi 33), **8176 MiB** total / 8140 MiB free at idle |
| ISA | **`gfx1102`** per `/opt/rocm/bin/rocminfo` (`Compute Unit : gfx1102`, 40 CUs, wave64, 64 KB LDG per CUI, 128 MB L2) — the design plan said `gfx1103`; see §7 |
| Second GPU | `AMD Radeon 680M` iGPU (7607 MiB, UMA) — unused, `device 1` |
| ROCm | 6.10.x toolchain under `/opt/rocm` (`hipcc`, `rocminfo`, `rocm-smi`); **no `rocprofv2`/`rocprofv3`/`rocprof`/`roctx` installed** — hence the LD_PRELOAD shim in §2 |
| Kernel driver | `/dev/kfd` present and world-accessible → ROCm compute works from the pi sandbox (this was not true when the plan was written) |
| CPU / RAM | 16 logical cores, 15.2 GiB RAM |
| Reference build | `llama.cpp-opensensenova` build **12467 @ 24ea7bb78**, GNU 16.2.1; built with `GGML_HIP=ON`, `GGML_HIP_GRAPHS=ON` (→ `USE_CUDA_GRAPH`), `GGML_HIP_NO_VMM=ON`, `GGML_HIP_MMQ_MFMA=ON`; only `bin/llama-server` is built |
| Offload | `offloaded 58/58 layers to GPU` (ROCm0), i.e. full offload works |

Model: `/home/justin/Documents/AI/Models/SenseLoop/SenseNova-Loop-9B-Q4_K_M.gguf` (5,818,631,072 B, 490 tensors).
Model load to GPU: ~30 s (mmap + upload); `llama-server` ready ~90 s after exec.

## 1. Node/graph budget (the flaw, confirmed)

At **every** context size tested, with `-ngl 99`:

```
sched_reserve: max_nodes = 83648
```

This is the fork reserving graph space for **one decode token**: 46,080 of those nodes are the mHC Sinkhorn
(`48 × mhc_iterations(20) × n_mhc_executions(48)` — the special case in `llama-context.cpp` `graph_reserve`),
the remaining ~37.6 k everything else. Matches design-plan §4.1 exactly.
Note it is *independent of n_ctx*: the node storm is per token, not per token-of-context, which is exactly why
prefill hides it and decode does not.

## 2. Launch counts per decoded token

Method: `LD_PRELOAD` shim (`Tools/hip_count.c`) that counts the real HIP calls the fork's CUDA-named
wrappers make; counters dumped and reset by `SIGUSR1` immediately before and after a measured
decode-only window (prompt fully cached, `prompt_n=4`, `n_predict=32`, so every counted call belongs to decode).
The window is identical at n_ctx 1024 / 8192 / 32768 and reproduced bit-for-bit across runs:

| Counter | Measured for 32 decoded tokens | Per token |
|---|---|---|
| `hipLaunchKernel` | 150,497 | **≈ 4,700** |
| `hipGraphLaunch` | 5,790 | ≈ 181 |
| `hipMemcpyAsync` | 8,104 | ≈ 253 |
| `hipLaunchKernelExC` | 0 | 0 |
| `hipMemcpy` | 0 | 0 |

SuperNova's Phase 8 target for the first three columns combined: **≤ 2**.

HIP graphs are *not* disabled and are *not* skipped for exceeding a node budget — the fork captures and replays
them (`GRAPH_DISABLED_DUE_TO_NODE=0`, `GRAPH_REINSTANTIATED=0`), building **291** distinct graph plans
("CUDA graph warmup complete" ×291, ids 3179…4541, `CUDA Graph id N reused` 3,201–3,589 lines) and
**126** `CUDA graph update failed` fallbacks. Even so the per-token stream is ≈4.7 k direct launches plus
181 graph replays plus 253 async copies: the node storm is not absorbed by graph capture, it is replayed.

## 3. Throughput

Same 692-token prose prompt (`Tools/phase0_prompt.txt`), `--temp 0 -s 42`, 32 decoded tokens, one slot.

| Pass | n_ctx | prefill t/s | **decode t/s** | ms/token |
|---|---|---|---|---|
| cold (uncached prompt) | 1024 | 218.3 | **4.44** | 225.1 |
| cold | 8192 | 190.7 | **4.39** | 227.6 |
| cold | 32768 | 211.5 | **4.43** | 225.8 |
| fill (7,724-token prompt) | 8192 | **379.3** | **4.97** | 201.1 |
| decode-only window | all | — (cache hit) | 4.41 / 4.44 / 4.39 | 225–228 |

Reading: decode speed is **flat in context length** (≈4.4 t/s at 1 k, 8 k and 32 k) — the cost is the
per-token host/node storm, not KV traffic; and prefill is ~45–85× faster than decode in t/s, and gets *faster*
with bigger batches, which is why the node storm is invisible in prefill benchmarks. The plan's "≈5 t/s on this
model today" is confirmed: measured **4.4 t/s**, i.e. **5.6× below the §3 bandwidth roofline of ~24.6 t/s**.

## 4. Memory at each context size (f16 KV, `n_parallel=4`)

| n_ctx | VRAM after load | VRAM after request | Δ vs n_ctx=1024 load |
|---|---|---|---|
| 1024 | 5.941 GB | 6.177 GB | — |
| 8192 | 6.360 GB | 6.596 GB | +0.419 GB |
| 32768 | **7.791 GB** | **8.027 GB** | +1.850 GB |
| 8192 (fill pass) | 6.360 GB | 6.600 GB | +0.419 GB |

Device total is 8.176 GB, so f16 KV at 32 k leaves **≈150 MB headroom** — the plan's "32 k does not fit
comfortably with f16 KV, use q8_0" is confirmed empirically.
KV scaling measured: (7.791 − 5.941) GB / 31,744 extra tokens ≈ **57 KiB per token**. The plan's arithmetic
(40 logical full-attn layers × 1024 elements × 2 B) says 80 KiB/token; the gap suggests the fork does **not**
allocate KV per logical Loop visit (the 4 physical in-loop full-attn blocks appear to share), which is worth
pinning down before Phase 7 sizing. The ≈0.236 GB step between "after load" and "after request" at every ctx is
the per-slot hybrid-state allocation (GDN recurrent + conv state for `n_parallel=4`).

## 5. Host cost

Server-process CPU (`/proc/<pid>/stat` utime+stime over the measured window):

| Window | Core-seconds consumed | Wall | ≈ cores busy |
|---|---|---|---|
| decode-only, 32 tokens, n_ctx=1024 | 57.5 | ~7.3 s | ~7.9 |
| decode-only, n_ctx=8192 / 32768 | 51.9 / 57.1 | ~7.3 s | ~7–8 |
| cold 692-token prompt + 32 tokens | 66.7 | ~13 s | ~5 |
| fill 7,724-token prompt + 32 tokens | 147.8 | ~33 s | ~4.5 |

`rocm-smi --showuse` sampled at the end of the cold windows: **43 %, 47 %, 76 %** device utilisation —
the GPU is idle roughly half the time even at 4.4 t/s. Combined with ~7 cores of host work per token, this is
the direct evidence for the plan's central claim: **decode is host/graph-management bound, not bandwidth bound.**
(The core-seconds figure is process-wide and includes the 15 HTTP threads and the CPU backend's share; it is a
bound on host cost, not an attribution per component.)

## 6. What Phase 0 confirms, and what it changes

1. **Confirmed**: the mHC unroll really costs 46,080 ggml nodes/token (reserved total 83,648) and ≈4.7 k real
   launches/token; graphs do not save it; decode is 5.6× under roofline with the device ~half idle.
   SuperNova's standing gates (≤300 ops/token, ≤2 launches/token, ≥20 t/s decode, <5 % CPU) stay as written.
2. **Confirmed**: 32 k with f16 KV is marginal on 8 GB → q8_0 KV remains the plan of record (design-plan §3).
3. **Corrected**: this GPU is **`gfx1102`**, not `gfx1103` (§7).
4. **Corrected**: the checkpoint contains **no Q2_K tensors** (only `f32`, `Q4_K`, `Q6_K`) — design-plan §2.1
   and §5 Phase 5 updated accordingly; the required dequant/GEMM paths are Q4_K, Q6_K, f32.
5. **New info**: the fork's KV grows ≈57 KiB/token, not 80 — recheck before Phase 7 sizing.
6. **Not established** (deliberately out of scope here): whether `hipGraphLaunch` actually replays on this
   driver. The counter proves the *call*, not the replay efficiency. SuperNova's Phase 8 asserts its own replay
   by the decode-time launch counter, per design-plan §4.3.

## 7. ISA correction: `gfx1102`, not `gfx1103`

```
$ /opt/rocm/bin/rocminfo | grep -A3 " gfx"
  Name:                                                            gfx1102
  Compute Unit : gfx1102
```
`lspci` reports `Device 7480` = Navi 33 `[RX 7700S]`. The distinction matters only in that Phase 3 must compile
for `gfx1102` (`--offload-arch=gfx1102` or `-mgpu-gfx1102`); the design plan's claims about the RDNA3 **1100
series ISA** (`v_wmma_*`, no `v_mfma_*`, wave32 build, no `cuPerRS` on RDNA3) are unaffected — `gfx1102` is in
that family. The §3 roofline (215 GB/s) is unchanged.

## 8. Reproducing this table

```sh
gcc -O2 -fPIC -shared -o /tmp/phase0/libhipcount.so Tools/hip_count.c -ldl
PASS=all CTXS="1024 8192 32768" NGL=99 DECODE_TOKENS=32 \
  FILL_CTX=8192 FILL_PROMPT=Tools/phase0_prompt_8k.txt \
  bash Tools/phase0.sh /tmp/phase0 | tee /tmp/phase0/run.log
```
Passes: `launch` (graph/node/launch counters, decode-only window), `cold` (`--no-cache-prompt`, honest pp/tg),
`fill` (7.7 k-token prompt to measure prefill at depth). ~15 min wall for the whole sweep; each pass starts its
own server so nothing is shared between configurations.
Caveat when reading the raw log: during the `launch` pass `LD_PRELOAD` is exported to the shell's children too,
so `HIPCOUNTER EXIT …` lines from `curl`/`rocm-smi` interleave with the VRAM figures — the per-pass
`*.log`/`*.json` artifacts are the clean source.

## 9. Still to measure at Phase 10

The plan's comparison harness (`llama-bench -pg 1`, n_ctx 8196, pp512/tg128 at n_batch 512, 1/8/32 streams,
plus the launch counter for the same shape) is not part of Phase 0: this build has only `llama-server`, and
`llama-cli`/`llama-bench` are absent from the fork's CMake target list (design-plan R9). Revisit in Phase 10.
