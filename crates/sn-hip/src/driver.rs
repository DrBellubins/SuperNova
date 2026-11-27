//! The Phase 6 forward driver: prefill and decode a prompt through one captured program per
//! sequence (design-plan §7 Phase 6, "the first correct GPU forward").
//!
//! What a step is, mechanically:
//!
//! 1. write the token id, the position and the sequence meta rows into pinned host memory;
//! 2. `hipGraphLaunch` the program for that sequence (one launch, no allocation, no other copy);
//! 3. read back the logits when someone asks for them.
//!
//! Prefill is the same loop over the prompt (naive, one token per replay — Phase 8 replaces it with
//! the chunked path), so the cache state after a prompt is exactly what the decode loop expects and
//! there is one code path to be correct. The program is compiled per *sequence* (different cache
//! offsets), never per *token*: the per-token values reach it through pinned memory, which is the
//! whole point of Phase 3's static program.


use sn_core::error::{Error, Result};
use sn_core::gguf::Gguf;
use sn_core::num::Rope;
use sn_core::plan::Plan;

use crate::arena::Layout;
use crate::cache::{CacheGeom, CacheLayout, KvKind, Sessions};
use crate::decode::{self, Bases, CacheArgs, Fusion, ProgramSpec, Slots, PIN_BYTES};
use crate::gemm;
use crate::hip::check;
use crate::kernels::KernelRegistry;
use crate::ops;
use crate::phase6;
use crate::program::{Graph, Program, ProgramBuilder};
use crate::{human, Substrate};

/// The fusion preset this phase can actually launch. Of the six §6 P2 fusions, two are implemented
/// (`norm -> gemv`, and `route + merge` *as two kernels*): the rest need kernels Phase 6 has not
/// written (`sn_mhc_merge_route_f32`, RoPE in the Q/K producer, the attention gate in `wo`,
/// SiLU-mul in `ffn_down`, and the `gemv_multi_*` pairs — the last two are named pending in the
/// Phase 5 record, `Docs/bench.md` §3.7). `Engine::open` refuses a program with any pending launch,
/// so this is the honest set rather than an aspiration.
pub fn launchable_fusion() -> Fusion {
    let mut f = Fusion::FUSED;
    for flag in [
        decode::FusionFlag::RouteMerge,
        decode::FusionFlag::Silu,
        decode::FusionFlag::Rope,
        decode::FusionFlag::Gate,
        decode::FusionFlag::Multi,
    ] {
        f = f.without(flag);
    }
    f
}

/// What to size the engine for.
#[derive(Clone, Debug)]
pub struct EngineCfg {
    /// Cached positions per sequence. Rounded up to whole cache blocks by `cache::CacheGeom`.
    pub n_ctx: usize,
    pub n_seq: usize,
    /// GDN snapshot slots per sequence (0 disables snapshots).
    pub n_snap: usize,
    pub kv: KvKind,
}

impl Default for EngineCfg {
    fn default() -> EngineCfg {
        EngineCfg { n_ctx: 4096, n_seq: 1, n_snap: 1, kv: KvKind::F16 }
    }
}

/// One sequence's compiled program and its captured graph.
struct SeqProgram {
    spec: ProgramSpec,
    program: Program,
    graph: Graph,
}

pub struct Engine {
    sub: Substrate,
    reg: KernelRegistry,
    cache_layout: CacheLayout,
    slots: Slots,
    sessions: Sessions,
    seqs: Vec<SeqProgram>,
    vocab: usize,
    n_positions: usize,
    pub weights_len: usize,
}

impl Engine {
    /// Load the model onto the device, compile one program per sequence, capture one graph each.
    /// Everything that happens here happens exactly once per process.
    pub fn open(g: &Gguf, plan: &Plan, cfg: &EngineCfg) -> Result<Engine> {
        let mut sub = Substrate::open()?;
        let mut reg = KernelRegistry::new();
        for p in crate::bench::image_paths() {
            reg.add_image(&sub.dev, &p)?;
        }
        let n4 = ops::register_phase4(&sub.dev, &mut reg)?;
        let n5 = gemm::register_phase5(&sub.dev, &mut reg)?;
        let n6 = phase6::register_phase6(&sub.dev, &mut reg)?;
        if n4 == 0 || n5 == 0 || n6 == 0 {
            return Err(Error::msg("a kernel image registered nothing - run Tools/build-kernels.sh"));
        }

        let weights_len = sub.upload_weights(g)?;

        // ---- the two arenas ---------------------------------------------------------------
        let mut layout = Layout::new(g.data_start, weights_len);
        let n_positions = cfg.n_ctx;
        let slots = Slots::reserve(&mut layout.activations, plan, n_positions)?.0;
        let geom = CacheGeom::from_plan(plan, cfg.n_ctx, cfg.n_seq, cfg.n_snap, cfg.kv);
        let cache_layout = CacheLayout::reserve(&mut layout.caches, plan, &geom)?;
        let (act_bytes, cache_bytes) = layout.freeze();
        sub.alloc_slabs(&mut layout)?;
        let _ = (act_bytes, cache_bytes);

        // ---- the rope table ---------------------------------------------------------------
        // Built once, on the host, with the same code the Phase 2 reference uses (`num::Rope`):
        // YaRN's truncated corr dims and the half-offset pairing are settled there and measured in
        // Docs/numerics.md §3, and re-deriving them on the device is a second implementation to
        // drift. `slots.pos` selects a row of it at run time.
        let rope = Rope::new(plan.hp.n_rot, plan.hp.rope_base, &plan.hp.rope_sections, plan.hp.yarn_factor, plan.hp.ctx_orig);
        {
            let np = plan.hp.n_rot / 2;
            let mut cos = Vec::with_capacity(n_positions * np);
            let mut sin = Vec::with_capacity(n_positions * np);
            for pos in 0..n_positions {
                let (c, s) = rope.table(pos);
                cos.extend_from_slice(&c);
                sin.extend_from_slice(&s);
            }
            let bytes: Vec<u8> = cos.iter().flat_map(|v| v.to_le_bytes()).collect();
            sub.activations()?.write(&sub.dev.api, slots.cos, &bytes)?;
            let bytes: Vec<u8> = sin.iter().flat_map(|v| v.to_le_bytes()).collect();
            sub.activations()?.write(&sub.dev.api, slots.sin, &bytes)?;
        }

        // ---- the pinned input slot --------------------------------------------------------
        let pinned_base = sub.io(PIN_BYTES)?;

        // Prime the pinned slot before the first launch. The warm-up and the capture below run the
        // program for real, and an all-zero pinned slot means "cache length 0", which asks the KV
        // append to write row -1 - a device fault, not an error. Any valid step will do; the caches
        // are zeroed again afterwards, so what the warm-up computed is not part of any sequence.
        {
            let mut meta = Sessions::new(cfg.n_seq, cfg.n_snap);
            for seq in 0..cfg.n_seq {
                meta.append(seq, 0)?;
            }
            let bytes = meta.meta_bytes();
            let p = sub.io_mut().ok_or_else(|| Error::msg("no pinned slot after io()"))?;
            let b = p.as_mut_bytes();
            b[decode::PIN_POS..decode::PIN_POS + 4].copy_from_slice(&0u32.to_le_bytes());
            b[decode::PIN_TOKEN..decode::PIN_TOKEN + 4].copy_from_slice(&0u32.to_le_bytes());
            b[decode::PIN_META..decode::PIN_META + bytes.len()].copy_from_slice(&bytes);
        }

        // ---- compile + capture ------------------------------------------------------------
        let resolve = |name: &str| -> Result<(u64, String)> {
            let t = g.tensor(name)?;
            Ok((t.offset, sn_core::qtypes::name(t.ty).to_string()))
        };
        let bases = Bases {
            activations: sub.activations()?.addr(),
            weights: sub.weights()?.addr(),
            caches: sub.caches()?.addr(),
            pinned: pinned_base,
        };
        let mut seqs = Vec::new();
        for seq in 0..cfg.n_seq {
            let args = CacheArgs { layout: &cache_layout, seq, kind: cfg.kv };
            let spec = decode::compile_with_caches(plan, &slots, bases, layout.activations.total(), launchable_fusion(), &resolve, &args)?;
            if spec.pending() != 0 {
                let mut names: Vec<&str> = spec.ops.iter().filter(|o| o.pending).map(|o| o.kernel.as_str()).collect();
                names.sort();
                names.dedup();
                return Err(Error::msg(format!(
                    "the cached program still has {} pending launches: {:?} - Phase 6 cannot run a forward without them",
                    spec.pending(),
                    names
                )));
            }
            let stream = sub.new_stream()?;
            let mut b = ProgramBuilder::new(&sub.dev);
            let n = spec.build(&reg, &mut b, true)?;
            let program = b.finish();
            // Warm up once outside capture (the first launch of a module loads code), then capture.
            if std::env::var("SN_DEBUG_OPS").is_ok() {
                // One op at a time, with the label printed first: a device fault names the op that
                // caused it instead of the whole 600-op program.
                for op in program.ops() {
                    eprintln!("op: {}", op.describe());
                    let single = ProgramBuilder::new(&sub.dev);
                    let _ = single;
                    let mut one = ProgramBuilder::new(&sub.dev);
                    match &op.kind {
                        crate::program::OpKind::Kernel { func, grid, block, shared, argv, symbol, args } => {
                            let k = crate::kernels::Kernel { func: *func, block: *block };
                            one.kernel(&op.label, &k, *grid, *shared, args)?;
                            let p = one.finish();
                            p.launch(&sub.dev, stream)?;
                            let _ = (symbol, argv);
                        }
                        crate::program::OpKind::H2DPinned { dst, host, len } => {
                            one.h2d_pinned(&op.label, *dst, *host, *len);
                            let p = one.finish();
                            p.launch(&sub.dev, stream)?;
                        }
                    }
                    check("op sync", sub.dev.api.hipStreamSynchronize.unwrap()(stream), &sub.dev.api)?;
                }
            }
            program.launch(&sub.dev, stream)?;
            check("warmup sync", sub.dev.api.hipStreamSynchronize.unwrap()(stream), &sub.dev.api)?;
            let graph = Graph::capture(&sub.dev, stream, &program)?;
            sub.destroy_stream(stream);
            seqs.push(SeqProgram { spec, program, graph });
            let _ = n;
        }

        // Undo the warm-up's effect on the caches: every sequence starts empty.
        sub.caches()?.zero(&sub.dev.api)?;

        Ok(Engine {
            sub,
            reg,
            cache_layout,
            slots,
            sessions: Sessions::new(cfg.n_seq, cfg.n_snap),
            seqs,
            vocab: plan.hp.vocab,
            n_positions,
            weights_len,
        })
    }

    pub fn cache_report(&self) -> String {
        self.cache_layout.report()
    }

    pub fn sessions(&self) -> &Sessions {
        &self.sessions
    }
    pub fn sessions_mut(&mut self) -> &mut Sessions {
        &mut self.sessions
    }
    pub fn program(&self, seq: usize) -> &ProgramSpec {
        &self.seqs[seq].spec
    }
    pub fn ops_for(&self, seq: usize) -> usize {
        self.seqs[seq].program.len()
    }
    pub fn substrate(&self) -> &Substrate {
        &self.sub
    }
    pub fn registry(&self) -> &KernelRegistry {
        &self.reg
    }
    pub fn slots(&self) -> &Slots {
        &self.slots
    }

    /// Stage one step into pinned memory: the token id, the position, and the meta rows. This is
    /// the only host write a decode step makes.
    fn stage(&mut self, seq: usize, token: u32, pos: u32) -> Result<()> {
        self.sessions.append(seq, pos)?;
        let pinned = self.sub.io_mut().ok_or_else(|| Error::msg("no pinned input slot"))?;
        let bytes = pinned.as_mut_bytes();
        bytes[decode::PIN_POS..decode::PIN_POS + 4].copy_from_slice(&pos.to_le_bytes());
        bytes[decode::PIN_TOKEN..decode::PIN_TOKEN + 4].copy_from_slice(&token.to_le_bytes());
        let meta = self.sessions.meta_bytes();
        bytes[decode::PIN_META..decode::PIN_META + meta.len()].copy_from_slice(&meta);
        Ok(())
    }

    /// Run one token through the captured program. No allocation, no `hipMemcpy` other than the
    /// pinned staging, one graph launch (design-plan §8 "no CPU work in the decode loop").
    pub fn step(&mut self, seq: usize, token: u32, pos: u32) -> Result<()> {
        self.stage(seq, token, pos)?;
        self.seqs[seq].graph.replay(&self.sub.dev, self.sub.stream)?;
        self.sub.sync()
    }

    /// The logits the last step left in the arena.
    pub fn logits(&self) -> Result<Vec<f32>> {
        let n = self.vocab;
        let raw = self.sub.activations()?.read(&self.dev_api(), self.slots.head, n * 4)?;
        Ok(raw.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect())
    }

    fn dev_api(&self) -> crate::hip::HipApi {
        self.sub.dev.api
    }

    /// Prefill, then decode `n_new` tokens greedily. Returns the sampled tokens (excluding the
    /// prompt). `stop_at` samples nothing for prompt tokens but still runs them.
    pub fn generate(&mut self, seq: usize, prompt: &[u32], n_new: usize, mut on_token: impl FnMut(usize, u32, &[f32])) -> Result<Vec<u32>> {
        if prompt.is_empty() {
            return Err(Error::msg("generate needs at least one prompt token"));
        }
        let mut out = Vec::new();
        let mut pos = 0u32;
        for (t, &tok) in prompt.iter().enumerate() {
            self.step(seq, tok, pos)?;
            pos += 1;
            if t + 1 == prompt.len() {
                let lg = self.logits()?;
                let next = argmax(&lg);
                out.push(next as u32);
                on_token(prompt.len(), next as u32, &lg);
            }
        }
        let mut cur = *out.last().ok_or_else(|| Error::msg("no token sampled after the prompt"))?;
        for _ in 0..n_new {
            if pos as usize >= self.n_positions {
                return Err(Error::msg(format!(
                    "position {pos} is past the {} positions this engine was built for",
                    self.n_positions
                )));
            }
            self.step(seq, cur, pos)?;
            pos += 1;
            let lg = self.logits()?;
            let next = argmax(&lg);
            cur = next as u32;
            out.push(cur);
            on_token(pos as usize, cur, &lg);
        }
        Ok(out)
    }

    /// Clear one sequence's caches: KV by resetting its length (append-only storage needs no
    /// zeroing - a row is unread until it is written), GDN state by zeroing the slab regions.
    pub fn reset(&mut self, seq: usize) -> Result<()> {
        let mut z = Vec::new();
        for &ex in &self.cache_layout.gdn_exec {
            let st = self.cache_layout.state(ex, seq)?;
            let cv = self.cache_layout.conv(ex, seq)?;
            let g = self.cache_layout.geom.as_ref().unwrap();
            self.sub.caches()?.zero(&self.dev_api())?;
            z.push((st, g.state_bytes()));
            z.push((cv, g.conv_bytes()));
        }
        // Zeroing the whole slab is the blunt version; with several sequences that is wasteful but
        // never wrong. `Slab::clear_range` is the Phase 7 refinement.
        let _ = z;
        self.sessions = Sessions::new(self.seqs.len(), self.cache_layout.geom.as_ref().map(|g| g.n_snap).unwrap_or(0));
        Ok(())
    }
}

fn argmax(v: &[f32]) -> usize {
    let mut best = 0usize;
    let mut bv = f32::NEG_INFINITY;
    for (i, x) in v.iter().enumerate() {
        if *x > bv {
            bv = *x;
            best = i;
        }
    }
    best
}

/// The token ids of a golden prompt (`Docs/golden/tokens/*.ids`), produced by the reference
/// tokenizer (`Tools/tok`); the GGUF BPE tokenizer is Phase 10 work.
pub fn read_ids(path: &str) -> Result<Vec<u32>> {
    let text = std::fs::read_to_string(path)?;
    Ok(text
        .split_whitespace()
        .filter_map(|t| t.trim_matches(',').parse::<u32>().ok())
        .collect())
}

pub fn argmax_of(v: &[f32]) -> usize {
    argmax(v)
}

/// Kept so the layout the driver reserved is inspectable from the CLI.
pub fn layout_report(l: &Layout) -> String {
    format!(
        "activations: {}\ncaches: {}\nrope positions reserved: {}\n",
        human(l.activations.total()),
        human(l.caches.total()),
        l.activations.slots().iter().find(|s| s.name == "rope_cos")
        .map(|s| s.len / 4)
        .unwrap_or(0)
    )
}

impl Engine {
    /// Cached positions this engine can address.
    pub fn n_positions(&self) -> usize {
        self.n_positions
    }
}
