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
