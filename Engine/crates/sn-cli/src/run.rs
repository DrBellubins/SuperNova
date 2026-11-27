//! `sn run`: the Phase 6 deliverable — prefill and decode a prompt end-to-end on the GPU.
//!
//! Token ids in, token ids out. The GGUF BPE tokenizer is Phase 10 work (design-plan §7), so the
//! prompt comes from `Docs/golden/tokens/*.ids` (produced by `Tools/tok`, the reference tokenizer)
//! or from `-t`. `--expect <golden-id>` scores the first sampled token against the fork's own
//! top-1 from `Docs/golden/expected.tsv`, and `--ref-tokens N` runs the Phase 2 scalar reference
//! over the same prompt and reports the logits NMSE — the two halves of the Phase 6 gate.

use sn_core::error::{Error, Result};
use sn_core::gguf::Gguf;
use sn_core::plan::{Hyperparams, Plan};
use sn_hip::cache::KvKind;
use sn_hip::driver::{Engine, EngineCfg};

/// One row of `Docs/golden/expected.tsv`.
pub struct Expected {
    pub id: String,
    pub prompt: Vec<u32>,
    pub top1: u32,
    pub top8: Vec<(u32, f32)>,
}

/// The frozen corpus is kept at the repository root (`Docs/`), outside this workspace, so its
/// default location is found by search rather than assumed to be under the working directory —
/// `Engine/target/release/sn` and `cargo test` run from different directories. `--expected <path>`
/// is always used verbatim; this only decides what happens when it is not given.
pub const DEFAULT_EXPECTED: &str = "Docs/golden/expected.tsv";

pub fn expected_path(explicit: Option<String>) -> String {
    explicit.unwrap_or_else(|| {
        sn_core::paths::resolve_file(DEFAULT_EXPECTED)
            .map(|p| p.to_string_lossy().into_owned())
            // Not in this tree: report the path the caller would have had to pass.
            .unwrap_or_else(|| DEFAULT_EXPECTED.to_string())
    })
}

pub fn read_expected(path: &str, id: &str) -> Result<Expected> {
    let text = std::fs::read_to_string(path)?;
    for line in text.lines() {
        if line.starts_with('#') || line.trim().is_empty() {
            continue;
        }
        let mut f = line.split('\t');
        let (Some(gid), Some(ids), Some(top1)) = (f.next(), f.next(), f.next()) else { continue };
        if gid != id {
            continue;
        }
        let prompt = ids.split(',').filter_map(|t| t.trim().parse::<u32>().ok()).collect();
        let top1 = top1
            .trim()
            .parse::<u32>()
            .map_err(|_| Error::msg(format!("{path}: bad top1 for {id}")))?;
        let top8 = f
            .next()
            .unwrap_or("")
            .split_whitespace()
            .filter_map(|s| {
                let (a, b) = s.split_once(':')?;
                Some((a.parse().ok()?, b.parse().ok()?))
            })
            .collect();
        return Ok(Expected { id: gid.to_string(), prompt, top1, top8 });
    }
    Err(Error::msg(format!("{path}: no golden prompt named {id}")))
}

pub fn cmd_run(positional: &[String], opt_val: &dyn Fn(&str) -> Option<String>) -> Result<String> {
    let flag = |n: &str| positional.iter().any(|a| a == n) || std::env::args().any(|a| a == n);
    // `--golden p1-capital` puts the golden id in `positional` too, so a positional model path is
    // only accepted if it is not that id.
    let golden = opt_val("--golden").or_else(|| opt_val("--expect"));
    let model = opt_val("--model")
        .or_else(|| opt_val("-m"))
        .or_else(|| positional.get(1).cloned().filter(|p| Some(p.as_str()) != golden.as_deref()))
        .ok_or_else(|| Error::msg("sn run --model <gguf> (-t ids | --golden id) [-n N]"))?;
    let g = Gguf::open(&model)?;
    let plan = Plan::build(Hyperparams::from_gguf(&g)?)?;

    // ---- the prompt ---------------------------------------------------------------------
    let expected_path = expected_path(opt_val("--expected"));
    let mut expect: Option<Expected> = None;
    let prompt: Vec<u32> = if let Some(id) = &golden {
        let e = read_expected(&expected_path, id)?;
        let out = e.prompt.clone();
        expect = Some(e);
        out
    } else if let Some(path) = opt_val("--ids") {
        sn_hip::driver::read_ids(&path)?
    } else if let Some(list) = opt_val("-t").or_else(|| opt_val("--tokens")) {
        list.split(',').filter_map(|t| t.trim().parse::<u32>().ok()).collect()
    } else if let Some(text) = opt_val("-p").or_else(|| opt_val("--prompt")) {
        return Err(Error::msg(format!(
            "the GGUF tokenizer is Phase 10 work; tokenise {text:?} with Tools/tok and pass it with --ids or -t\n\
             (the golden prompts are already tokenised: try --golden p1-capital)"
        )));
    } else {
        return Err(Error::msg("sn run needs --golden <id>, --ids <file> or -t 1,2,3"));
    };
    if prompt.is_empty() {
        return Err(Error::msg("the prompt has no tokens"));
    }

    let n_new: usize = opt_val("-n").or_else(|| opt_val("--max-new")).unwrap_or("8".into()).parse().unwrap_or(8);
    let n_ctx: usize = opt_val("--ctx").unwrap_or("2048".into()).parse().unwrap_or(2048);
    let n_seq: usize = opt_val("--seq").unwrap_or("1".into()).parse().unwrap_or(1);
    let n_snap: usize = opt_val("--snap").unwrap_or("1".into()).parse().unwrap_or(1);
    let kv = match opt_val("--kv") { Some(v) => KvKind::parse(&v)?, None => KvKind::F16 };
    // `--raw` compiles the program with nothing fused: every step of §2 is a separate launch. It
    // exists to answer "is this difference the fusion or the maths?" without rebuilding anything.
    let fusion = if flag("--raw") {
        sn_hip::decode::Fusion::RAW
    } else if flag("--fused") {
        sn_hip::decode::Fusion::FUSED
    } else {
        sn_hip::driver::launchable_fusion()
    };
    let cfg = EngineCfg {
        n_ctx,
        n_seq,
        n_snap,
        kv,
        fusion,
        drop_classes: opt_val("--drop")
            .map(|v| v.split(',').map(|s| s.trim().to_string()).filter(|s| !s.is_empty()).collect())
            .unwrap_or_default(),
        drop_match: opt_val("--drop-match"),
    };

    let t0 = std::time::Instant::now();
    let mut eng = Engine::open(&g, &plan, &cfg)?;
    let open_secs = t0.elapsed().as_secs_f64();

    let mut out = String::new();
    out += &format!("{}\n", eng.substrate().dev.info);
    out += &format!("model: {model}  plan: {} executions, {} physical blocks\n", plan.n_exec(), plan.n_physical);
    out += &format!(
        "engine: {} sequences, {} token regions, KV type {} — weights {} in {:.1} s\n",
        n_seq,
        n_ctx,
        kv.name(),
        sn_hip::human(eng.weights_len),
        open_secs
    );
    out += &eng.cache_report();
    // Phase 7's VRAM half of the gate is a number about the *device*, not about the layout: what is
    // actually left after the weight slab and both arenas are allocated.
    let (free, total) = eng.substrate().free_device_memory()?;
    out += &format!(
        "vram: {} of {} device memory in use (weights + arenas), {} free\n",
        sn_hip::human(total - free),
        sn_hip::human(total),
        sn_hip::human(free)
    );
    let spec = eng.program(0);
    out += &format!(
        "program: {} ops per token ({} launches), {} ops including pinned copies\n",
        spec.len(),
        eng.ops_for(0),
        spec.len() + spec.inputs.len()
    );
    for (class, n) in spec.counts() {
        out += &format!("  {class:<8} {n:>5}\n");
    }

    // ---- run ----------------------------------------------------------------------------
    let mut log = Vec::new();
    let run0 = std::time::Instant::now();
    let sampled = eng.generate(0, &prompt, n_new, |pos, tok, _| {
        log.push(format!("  pos {pos:>4} -> {}", tok));
    })?;
    let run_secs = run0.elapsed().as_secs_f64();
    let total_steps = prompt.len() + n_new;
    out += &format!(
        "\nprompt: {} tokens, decoded {} ({} steps in {:.2} s = {:.2} t/s)\n",
        prompt.len(),
        sampled.len() - 1 + n_new,
        total_steps,
        run_secs,
        total_steps as f64 / run_secs
    );
    out += &format!("first sample after prompt: {}\n", sampled[0]);
    out += &format!("continuation: {}\n", sampled.iter().map(|t| t.to_string()).collect::<Vec<_>>().join(" "));
    out += &log.join("\n");
    out += &format!("\ncaches after the run:\n{}", eng.sessions_mut().report());

    // ---- the gate halves ----------------------------------------------------------------
    // `--bench N`: N graph replays with device events and nothing else in the loop. This is the
    // Phase 7 number (t/s at a given n_ctx); with `--drop <class>` the difference between two runs
    // is what that class costs, which is the no-profiler version of "stop and profile".
    if let Some(n) = opt_val("--bench").and_then(|v| v.parse::<usize>().ok()).filter(|n| *n > 0) {
        let ms = eng.bench_replay(0, n)?;
        out += &format!(
            "\nbench: {} replays of a {}-launch program in {:.1} ms = {:.3} ms/token = {:.2} t/s{}\n",
            n,
            eng.ops_for(0),
            ms,
            ms / n as f64,
            n as f64 / (ms / 1e3),
            if cfg.drop_classes.is_empty() {
                String::new()
            } else {
                format!(" (classes dropped: {:?})", cfg.drop_classes)
            }
        );
    }

    if let Some(e) = &expect {
        let got = sampled[0];
        let ok = got == e.top1;
        out += &format!(
            "\ngolden {}: engine top-1 {} vs fork top-1 {} -> {}\n",
            e.id,
            got,
            e.top1,
            if ok { "MATCH" } else { "DIFFER" }
        );
        if let Some((id2, lp)) = e.top8.first() {
            out += &format!("  fork's top-1 was {id2} at logprob {lp:.4}\n");
        }
    }

    if let Some(n) = opt_val("--ref-tokens").map(|v| v.parse::<usize>().unwrap_or(0)).filter(|n| *n > 0) {
        let take = n.min(prompt.len());
        let mut lmse = 0.0f64;
        let n_runs = take.clamp(1, 8);
        for k in 1..=n_runs {
            let mut rm = sn_core::forward::RefModel::new(&plan, &g)?;
            rm.set_kv_f16(true); // the engine's cache type; the rounding is the same on both sides
            rm.set_threads(8);
            let want = rm.run(&prompt[..k], None, &sn_core::forward::Dumper::none()?)?;
            // Re-run the engine over the same prefix on a fresh sequence's state.
            eng.reset(0)?;
            for (p, &tok) in prompt[..k].iter().enumerate() {
                eng.step(0, tok, p as u32)?;
            }
            let got = eng.logits()?;
            let nmse = sn_core::oracle::nmse(&got, &want);
            lmse = lmse.max(nmse);
            out += &format!("\nlogits vs Phase 2 reference at token {k}: NMSE {nmse:.3e} (gate 1e-4)\n");
        }
        out += &format!("\nworst logits NMSE over {n_runs} positions: {lmse:.3e}\nGATE (logits): {}\n", if lmse <= 1e-4 { "PASS" } else { "NOT MET" });
    }

    Ok(out)
}
