//! `sn gpu` and the Phase 4 bench/plan commands: the numbers Phase 4's gate is made of.
//!
//! Every subcommand prints the measurement and the gate verdict, so `Docs/bench.md` can quote it
//! without anyone retyping a figure.

use sn_core::error::Result;
use sn_core::gguf::Gguf;
use sn_core::plan::{Hyperparams, Plan};
use sn_hip::check::{self, Check};
use sn_hip::decode::{self, Bases, Fusion};
use sn_hip::{KernelRegistry, Substrate};

/// Parse `4-11`, `4`, or `4,5,6` into an execution list.
fn exec_range(spec: &str, plan: &Plan) -> Result<Vec<usize>> {
    let mut out = Vec::new();
    let push = |a: usize, b: usize| -> Vec<usize> {
        (a..=b).filter(|&il| il < plan.n_exec() && plan.exec(il).mhc && plan.exec(il).mtp == 0).collect()
    };
    for part in spec.split(',') {
        let part = part.trim();
        if let Some((a, b)) = part.split_once('-') {
            let a: usize = a.trim().parse().map_err(|_| sn_core::Error::msg(format!("bad range {part:?}")))?;
            let b: usize = b.trim().parse().map_err(|_| sn_core::Error::msg(format!("bad range {part:?}")))?;
            out.extend(push(a, b));
        } else if !part.is_empty() {
            let a: usize = part.parse().map_err(|_| sn_core::Error::msg(format!("bad execution {part:?}")))?;
            out.extend(push(a, a));
        }
    }
    if out.is_empty() {
        // The default: the first eight mHC executions, which is what `sn ref --limit-exec 12` dumps
        // cheaply, plus enough of the loop to see the two routes of each.
        let first = plan.hp.mhc_start;
        out.extend(push(first, first + 7));
    }
    Ok(out)
}

/// `sn gpu kernels`: every Phase 4 kernel against its scalar oracle.
pub fn cmd_kernels(opt_val: &dyn Fn(&str) -> Option<String>) -> Result<String> {
    let _ = opt_val;
    let mut sub = Substrate::open()?;
    let mut reg = KernelRegistry::new();
    for p in sn_hip::bench::image_paths() {
        reg.add_image(&sub.dev, &p)?;
    }
    let mut h = check::Harness::open(&mut sub, &mut reg)?;
    let checks = check::kernel_checks(&mut h)?;
    let mut out = format!("{}\n", sub.dev.info);
    out += "kernel vs sn_core::oracle (identical deterministic inputs)\n";
    out += &check::report(&checks);
    let _ = opt_val;
    Ok(out)
}

/// `sn gpu route`: the fused route and merge replayed on the reference's own stage dumps.
pub fn cmd_route(positional: &[String], opt_val: &dyn Fn(&str) -> Option<String>) -> Result<String> {
    let model = opt_val("--model")
        .or_else(|| positional.get(2).cloned())
        .ok_or_else(|| sn_core::Error::msg("sn gpu route --model <gguf> --dump <dir> [--execs 4-11]"))?;
    let dump = opt_val("--dump").ok_or_else(|| sn_core::Error::msg("sn gpu route needs --dump <stage-dump-dir>"))?;
    let g = Gguf::open(&model)?;
    let plan = Plan::build(Hyperparams::from_gguf(&g)?)?;
    let execs = exec_range(&opt_val("--execs").unwrap_or_default(), &plan)?;

    let mut sub = Substrate::open()?;
    let mut reg = KernelRegistry::new();
    for p in sn_hip::bench::image_paths() {
        reg.add_image(&sub.dev, &p)?;
    }
    let mut h = check::Harness::open(&mut sub, &mut reg)?;
    let runs = check::route_vs_golden(&mut h, &g, &plan, &dump, &execs)?;

    let mut out = format!("{}\n", sub.dev.info);
    out += &format!("mHC route/merge replayed on Phase 2 stage dumps from {dump}\n");
    out += &format!("executions: {} ({})\n", execs.len(), execs.iter().map(|e| e.to_string()).collect::<Vec<_>>().join(","));
    let mut all: Vec<Check> = Vec::new();
    for r in &runs {
        out += &format!("\n--- execution {} (block {}, {}) ---\n", r.exec, plan.exec(r.exec).tensor_block, if plan.exec(r.exec).recurrent { "GDN" } else { "full attention" });
        out += &check::report(&r.checks);
        all.extend(r.checks.iter().cloned());
    }
    let mix: Vec<&Check> = all.iter().filter(|c| c.name.ends_with(".mix")).collect();
    let worst_mix = mix.iter().map(|c| c.nmse).fold(0.0f64, f64::max);
    let worst_other = all.iter().filter(|c| !c.name.ends_with(".mix")).map(|c| c.nmse).fold(0.0f64, f64::max);
    let failing: Vec<&Check> = all.iter().filter(|c| !c.ok()).collect();
    out += &format!(
        "\nworst NMSE: mix {worst_mix:.3e} (gate 1e-6), every other stage {worst_other:.3e} (gate 1e-5)\nGATE: {}\n",
        if failing.is_empty() { "PASS" } else { "FAIL" }
    );
    for c in &failing {
        out += &format!("  {} NMSE {:.3e} > {:.0e}\n", c.name, c.nmse, c.gate);
    }
    Ok(out)
}

/// Read one Phase 2 stage dump (`forward.rs`'s text format: a `# stage …` header, then one value
/// per line).
fn read_stage(dir: &std::path::Path, stage: &str) -> Option<Vec<f32>> {
    let text = std::fs::read_to_string(dir.join(format!("{stage}.txt"))).ok()?;
    let mut v = Vec::new();
    for line in text.lines() {
        let l = line.trim();
        if l.is_empty() || l.starts_with('#') {
            continue;
        }
        v.push(l.parse::<f32>().ok()?);
    }
    Some(v)
}

/// The worst element-wise difference, for a failing stage (NMSE alone hides whether one value is
/// wrong or all of them are slightly wrong).
fn worst_diff(a: &[f32], b: &[f32]) -> (usize, f32, f32) {
    let mut w = 0usize;
    let mut wd = 0.0f32;
    for i in 0..a.len().min(b.len()) {
        let d = (a[i] - b[i]).abs();
        if d > wd {
            wd = d;
            w = i;
        }
    }
    (w, a[w], b[w])
}

/// `sn gpu stages`: replay the decode program execution by execution and compare the arena with the
/// Phase 2 reference's own stage dumps. This is the bisect for "the forward is wrong": it names the
/// first execution whose output leaves the reference, instead of reporting one NMSE on the logits.
///
/// It compiles the program with `Fusion::RAW` on purpose: every step of §2 is then a separate
/// launch writing a separate buffer, so there is something to read back at every boundary.
pub fn cmd_stages(positional: &[String], opt_val: &dyn Fn(&str) -> Option<String>) -> Result<String> {
    use sn_core::forward::{Dumper, RefModel};
    use sn_hip::decode::Fusion as Fus;
    use sn_hip::driver::{Engine, EngineCfg};

    let model = opt_val("--model")
        .or_else(|| positional.get(2).cloned())
        .ok_or_else(|| sn_core::Error::msg("sn gpu stages --model <gguf> --golden <id> [--step 1]"))?;
    let gid = opt_val("--golden").ok_or_else(|| sn_core::Error::msg("stages needs --golden <id> (a tokenised golden prompt)"))?;
    let step = opt_val("--step").and_then(|v| v.parse::<usize>().ok()).unwrap_or(1).max(1);
    // `--deep N`: stop after execution N and compare every internal buffer that execution writes.
    let deep: Option<usize> = opt_val("--deep").and_then(|v| v.parse().ok());
    let g = Gguf::open(&model)?;
    let plan = Plan::build(Hyperparams::from_gguf(&g)?)?;
    let e = crate::run::read_expected(&opt_val("--expected").unwrap_or("Docs/golden/expected.tsv".into()), &gid)?;
    let prompt = e.prompt.clone();
    let ntok = step.min(prompt.len());

    // 1. the reference, over the same tokens, with every stage dumped.
    let dir = std::env::temp_dir().join(format!("sn-stages-{}", std::process::id()));
    std::fs::create_dir_all(&dir)?;
    let allow: Vec<String> = ["embd", "l.", "l_out.", "gdn.", "attn.", "ffn.", "mhc.", "logits", "final_norm"]
        .iter()
        .map(|s| s.to_string())
        .collect();
    let mut rm = RefModel::new(&plan, &g)?;
    rm.set_kv_f16(true);
    rm.set_threads(8);
    let rl = rm.run(&prompt[..ntok], None, &Dumper::with_filter(Some(dir.clone()), &allow)?)?;

    // 2. the engine, same tokens, nothing fused, one execution at a time.
    let cfg = EngineCfg {
        n_ctx: (ntok + 8).next_power_of_two(),
        n_seq: 1,
        n_snap: 0,
        kv: sn_hip::cache::KvKind::F16,
        fusion: Fus::RAW,
        ..Default::default()
    };
    let mut eng = Engine::open(&g, &plan, &cfg)?;
    // Everything the loop needs from the compiled program, taken before the loop borrows `eng`
    // mutably: the execution boundaries, the label of each boundary's last op, and the op count.
    let ends: Vec<(usize, usize, String)> = {
        let spec = eng.program(0);
        spec.exec_ends
            .iter()
            .map(|&(il, end)| (il, end, spec.ops[end.saturating_sub(1)].label.clone()))
            .collect()
    };
    let nsched = eng.program(0).len();
    let nops = eng.ops_for(0);
    // `ProgramSpec::build` prepends the pinned-input copies, so a `spec.exec_ends` index is an index
    // into `spec.ops`, and the program the driver holds has `inputs.len()` nodes in front of it.
    let ibase = eng.program(0).inputs.len();
    let slots = eng.slots().clone();
    let n = plan.hp.n_embd;
    eng.stage(0, prompt[0], 0)?;

    let mut out = format!("{}\nmodel: {model}\ngolden {gid}: comparing {} engine execution(s) against the Phase 2 reference at token {ntok}\n", sub_info(&eng), nops);
    out += &format!("reference stage dumps: {}\n\n", dir.display());
    out += &format!("program: {} scheduled nodes = {} launches; spec: {} ops + {} pinned copies; last execution ends at spec op {}\n", nops, eng.ops_for(0), nsched, ibase, ends.last().map(|e| e.1).unwrap_or(0));
    out += "  il blk kind  hidden a/b or cur         ffn branch            last op\n";
    let mut prev = 0usize;
    let mut first_bad: Option<usize> = None;
    for entry in &ends {
        let il = entry.0;
        let end = entry.1;
        let last = &entry.2;
        // The first range starts at 0 so the pinned-input nodes run once; after that only the
        // execution's own ops are added to the stream.
        eng.launch_ops(0, if prev == 0 { 0 } else { prev + ibase }, (end + ibase).min(nops))?;
        prev = end;
        let ex = *plan.exec(il);
        let kind = if ex.mtp != 0 { "mtp" } else if ex.recurrent { "gdn" } else { "attn" };
        // Inside the mHC range the state of the trunk is the two streams; outside it, `cur`.
        let (stage, got) = if ex.mhc {
            (format!("mhc.{il}.a"), eng.read_f32(slots.a, n)?)
        } else {
            (format!("l_out.{il}"), eng.read_f32(slots.cur, n)?)
        };
        let want = read_stage(&dir, &stage);
        let (hn, hd) = match &want {
            Some(w) => (sn_core::oracle::nmse(&got, w), worst_diff(&got, w)),
            None => (f64::NAN, (0, 0.0, 0.0)),
        };
        let bn = if ex.mtp != 0 { "-".to_string() } else {
            // At the boundary the `branch` slot holds the FFN's output (every branch shares it).
            let stage = format!("ffn.{il}.branch");
            let got = eng.read_f32(slots.branch, n)?;
            match read_stage(&dir, &stage) {
                Some(w) => format!("{:>9.3e}", sn_core::oracle::nmse(&got, &w)),
                None => "     n/a".to_string(),
            }
        };
        if hn > 1e-3 && first_bad.is_none() {
            first_bad = Some(il);
        }
        out += &format!(
            "{:>4} {:>3} {:<5} {:>9.3e} {:>7} {:>9} {}\n",
            il,
            ex.tensor_block,
            kind,
            hn,
            if want.is_some() { "" } else { "MISSING" },
            bn,
            last,
        );
        if want.is_none() {
            out += &format!("       (no {} in the dump)\n", stage);
        } else if hn > 1e-3 {
            out += &format!("       worst element index {w}, engine {eg}, reference {rf} (reference stage {stage})\n", w = hd.0, eg = hd.1, rf = hd.2);
        }
        if deep == Some(il) {
            out += &format!("\ndeep comparison for execution {il} (block {})\n", plan.exec(il).tensor_block);
            out += &stage_deep(&mut eng, &plan, &dir, il, &slots)?;
            break;
        }
    }
    // The head: the ops after the last execution (final norm + output projection).
    eng.launch_ops(0, (prev + ibase).min(nops), nops)?;
    let got = eng.read_f32(slots.head, plan.hp.vocab)?;
    let nm = sn_core::oracle::nmse(&got, &rl);
    out += &format!(
        "\nlogits: NMSE {nm:.3e}  engine top-1 {}  reference top-1 {}  fork top-1 {}\n",
        sn_hip::driver::argmax_of(&got),
        sn_hip::driver::argmax_of(&rl),
        e.top1
    );
    // The stage dumps are hundreds of MB (the GDN state is 0.5 M values per execution); they are a
    // scratch artifact, not an output of the command.
    let _ = std::fs::remove_dir_all(&dir);
    match first_bad {
        Some(il) => out += &format!("\nfirst execution that leaves the reference: {il} (block {})\n", plan.exec(il).tensor_block),
        None => out += "\nno execution diverged beyond 1e-3\n",
    }
    Ok(out)
}

fn sub_info(eng: &sn_hip::driver::Engine) -> String {
    eng.substrate().dev.info.to_string()
}

/// Every internal buffer one execution writes, against the reference's stage dump for it. This is
/// the level where a wrong GDN argument order (q vs k, tap order in the conv, the z offset) shows
/// up as one line instead of as "the logits are wrong".
fn stage_deep(
    eng: &mut sn_hip::driver::Engine,
    plan: &Plan,
    dir: &std::path::Path,
    il: usize,
    slots: &sn_hip::decode::Slots,
) -> Result<String> {
    let dm = &plan.hp.dims;
    let n = plan.hp.n_embd;
    let ex = *plan.exec(il);
    let mut out = String::new();
    let cmp = |s: &mut String, stage: &str, got: &[f32]| {
        match read_stage(dir, stage) {
            Some(w) => {
                let nm = sn_core::oracle::nmse(got, &w);
                let wd = worst_diff(got, &w);
                *s += &format!(
                    "  {:<18} NMSE {:>9.3e}  worst idx {:>5}  engine {:>13.6}  reference {:>13.6}\n",
                    stage, nm, wd.0, wd.1, wd.2
                );
            }
            None => *s += &format!("  {:<18} (not dumped)\n", stage),
        }
    };
    let h = eng.read_f32(slots.h, n)?;
    // Whichever `h` the execution leaves in the slot: outside the mHC range that is the
    // post-attention norm (`l.<il>.h2`), because the same slot is reused for both halves.
    cmp(&mut out, &format!("l.{il}.h2"), &h);
    cmp(&mut out, &format!("l.{il}.h"), &h);
    if ex.recurrent {
        let z = eng.read_f32(slots.qkvz + dm.qkv_width * 4, dm.n_v_heads * dm.d_v)?;
        cmp(&mut out, &format!("gdn.{il}.z"), &z);
        let conv = eng.read_f32(slots.conv_out, dm.qkv_width)?;
        let qk = dm.n_k_heads * dm.d_k;
        cmp(&mut out, &format!("gdn.{il}.q_t"), &conv[..qk]);
        cmp(&mut out, &format!("gdn.{il}.k_t"), &conv[qk..2 * qk]);
        cmp(&mut out, &format!("gdn.{il}.v_t"), &conv[2 * qk..]);
        let (st, _) = eng.gdn_regions(il, 0)?;
        let state = eng.read_cache_f32(st, dm.n_v_heads * dm.d_v * dm.d_k)?;
        cmp(&mut out, &format!("gdn.{il}.state"), &state);
        let br = eng.read_f32(slots.branch, n)?;
        cmp(&mut out, &format!("gdn.{il}.branch"), &br);
    } else {
        let ao = eng.read_f32(slots.attn_out, dm.n_q)?;
        cmp(&mut out, &format!("attn.{il}.out"), &ao);
        let br = eng.read_f32(slots.branch, n)?;
        cmp(&mut out, &format!("attn.{il}.branch"), &br);
    }
    // `branch` is the slot every branch writes, so at the boundary it holds the FFN's output.
    let br = eng.read_f32(slots.branch, n)?;
    cmp(&mut out, &format!("ffn.{il}.branch"), &br);
    let cur = eng.read_f32(slots.cur, n)?;
    cmp(&mut out, &format!("l_out.{il}"), &cur);
    Ok(out)
}

/// `sn gpu gemm` / `sn bench gemm`: the Phase 5 matmul surface.
///
/// One command does both halves because they share the only expensive setup - copying the
/// checkpoint's own weight bytes into a device slab, so the numbers are the model's shapes and the
/// model's quantisation mix rather than a fixture's. `sn gpu gemm` is the correctness gate alone
/// (what a test runs); `sn bench gemm` additionally measures the device references (read
/// bandwidth, f32-FMA peak, WMMA peak), sweeps every legal `warps_per_row` for every shape, and
/// times the prefill GEMM - the two Phase 5 gates and the auto-pick evidence in one run.
pub fn cmd_gemm(opt_val: &impl Fn(&str) -> Option<String>, bench: bool) -> Result<String> {
    let num = |name: &str, d: usize| -> Result<usize> {
        match opt_val(name) {
            None => Ok(d),
            Some(v) => v.parse::<usize>().map_err(|e| sn_core::Error::msg(format!("{name}: {e}"))),
        }
    };
    let path = opt_val("--model").ok_or_else(|| sn_core::Error::msg("gemm needs --model <gguf>"))?;
    let g = Gguf::open(&path)?;
    let plan = Plan::build(Hyperparams::from_gguf(&g)?)?;
    let sub = Substrate::open()?;
    let mut reg = KernelRegistry::new();
    for p in sn_hip::bench::image_paths() {
        reg.add_image(&sub.dev, &p)?;
    }
    let n5 = sn_hip::gemm::register_phase5(&sub.dev, &mut reg)?;
    let p5 = sn_hip::gemm::Phase5::resolve(&reg)?;

    let shapes = sn_hip::gemm::model_shapes(&g, &plan)?;
    let roles = shapes.iter().filter(|s| !s.variant).count();
    let mut out = format!(
        "{}\nmodel: {}\nplan: {} blocks, n_embd = {}\n{} Phase 5 kernels registered\n{} GEMM roles in the model ({} entries: a role that uses two quant classes is gated in both)\n",
        sub.dev.info,
        path,
        plan.n_layer,
        plan.hp.n_embd,
        n5,
        roles,
        shapes.len()
    );
    let mut tok_bytes = 0usize;
    for s in &shapes {
        out += &format!(
            "  {:<26} {:>13} {:>6} {:>12} B x{:>2} = {:>12} B/token{}\n",
            s.role,
            s.key(),
            s.type_name,
            s.bytes,
            s.visits,
            s.bytes_per_token(),
            if s.variant { "  (variant)" } else { "" }
        );
        tok_bytes += s.bytes_per_token();
    }
    out += &format!("  weight bytes per decoded token (these shapes): {tok_bytes}\n");

    let tmax = num("--gemm-max-t", 512)?;
    // Enough for a T = tmax activation and output of the widest model shape, and at least 256 MB so
    // the read-bandwidth reference runs over something far bigger than the 32 MB L2 (`--read-mb`
    // lowers it; `measure_machine` clamps to what is actually allocated).
    let scratch = (2 * tmax * 12288 * 4).max(256 << 20);
    let weight_bytes: usize = shapes.iter().map(|s| s.bytes).sum();
    out += &format!("\nuploading {weight_bytes} B of the model's own weights, {scratch} B scratch ...\n");
    let t = sn_hip::program::Timer::new(&sub.dev)?;
    t.begin(&sub.dev, sub.stream)?;
    let mut ws = sn_hip::gemm::upload_shapes(&g, &sub.dev, &shapes, scratch)?;
    out += &format!("  uploaded in {:.2} s\n", t.end_ms(&sub.dev, sub.stream)? / 1e3);

    let d1 = reg.find("dbg_q4k_lane", &[], sn_hip::kernels::DType::Any).cloned();
    let d2 = reg.find("dbg_args", &[], sn_hip::kernels::DType::Any).cloned();
    let mut checks = sn_hip::gemm::gemm_checks(&p5, &g, &mut ws, &shapes, &plan, &sub.dev, sub.stream, d1.as_ref(), d2.as_ref())?;
    // The f32 kernels are gated too, against a synthetic matrix: this checkpoint has no f32 2-D
    // weights, and an untested kernel is a worse answer than a synthetic test.
    checks.extend(sn_hip::gemm::f32_checks(&p5, &mut ws, &sub.dev, sub.stream, &plan)?);
    out += "\ncorrectness against sn_core::dq (the reference Phase 2 matched GGML's C with), same bytes\n";
    out += &check::report(&checks);
    let fails = checks.iter().filter(|c| !c.ok()).count();
    if !bench {
        return Ok(format!("{out}\n{} checks, {fails} failing\n", checks.len()));
    }

    let reps = num("--reps", 20)?.max(1);
    let trials = num("--trials", 3)?.max(1);
    let read_bytes = num("--read-mb", 256)?.max(16) << 20;
    let ts: Vec<usize> = match opt_val("--gemm-t") {
        Some(v) => v.split(',').filter_map(|x| x.trim().parse::<usize>().ok()).filter(|x| *x > 0).collect(),
        None => vec![8, 64, 256, 512],
    };
    let machine = sn_hip::gemm::measure_machine(&p5, &sub.dev, sub.stream, &ws.scratch, read_bytes)?;
    let rows = sn_hip::gemm::gemv_sweep(&p5, &sub.dev, sub.stream, &mut ws, &shapes, &plan, reps, trials)?;
    let gemms = sn_hip::gemm::gemm_table(&p5, &sub.dev, sub.stream, &mut ws, &shapes, &ts, reps, trials, &machine)?;
    out += "\n";
    out += &sn_hip::gemm::gemm_report(&rows, &gemms, &machine, &[]);
    Ok(format!("{out}\ncorrectness: {} checks, {fails} failing\n", checks.len()))
}

/// `sn bench fusion`: the fused route against the fork's node sequence.
pub fn cmd_fusion(opt_val: &dyn Fn(&str) -> Option<String>) -> Result<String> {
    let num = |name: &str, d: usize| -> Result<usize> {
        match opt_val(name) {
            None => Ok(d),
            Some(v) => v.parse::<usize>().map_err(|e| sn_core::Error::msg(format!("{name}: {e}"))),
        }
    };
    let reps = num("--reps", 50)?;
    let default_sweeps = plan_sweeps(opt_val("--model").as_deref())?;
    let sweeps = num("--sweeps", default_sweeps)?;
    let trials = num("--trials", 5)?;
    let mut sub = Substrate::open()?;
    let mut reg = KernelRegistry::new();
    for p in sn_hip::bench::image_paths() {
        reg.add_image(&sub.dev, &p)?;
    }
    let mut h = check::Harness::open(&mut sub, &mut reg)?;
    let gate = check::fusion_gate(&mut h, sweeps, reps, trials)?;
    let mut out = format!("{}\n", sub.dev.info);
    out += &gate.report();
    let failed: Vec<_> = gate.passes().into_iter().filter(|(_, ok, _, gating)| *gating && !*ok).collect();
    out += if failed.is_empty() { "GATE: PASS\n" } else { "GATE: FAIL\n" };
    Ok(out)
}

/// `mhc.iterations` from the checkpoint when a model is given, else the value this checkpoint uses.
fn plan_sweeps(model: Option<&str>) -> Result<usize> {
    match model {
        Some(m) => {
            let g = Gguf::open(m)?;
            Ok(Hyperparams::from_gguf(&g)?.mhc_iters)
        }
        None => Ok(20),
    }
}

/// `sn hip ops --model <gguf> [--fused]`: the decode step's op count, by class and by fusion.
pub fn cmd_ops(
    positional: &[String],
    opt_val: &dyn Fn(&str) -> Option<String>,
    flag: &dyn Fn(&str) -> bool,
) -> Result<String> {
    let model = opt_val("--model")
        .or_else(|| positional.get(2).cloned())
        .ok_or_else(|| sn_core::Error::msg("sn hip ops --model <gguf> [--fused] [--ladder]"))?;
    let g = Gguf::open(&model)?;
    let plan = Plan::build(Hyperparams::from_gguf(&g)?)?;
    let n_ctx = opt_val("--ctx").and_then(|v| v.parse::<usize>().ok()).unwrap_or(8192);

    let mut act = sn_hip::arena::Arena::new("decode-activations");
    let (slots, bytes) = decode::Slots::reserve(&mut act, &plan, n_ctx)?;
    act.freeze();
    let bases = Bases { activations: 0x1_0000_0000, weights: 0x8_0000_0000, ..Default::default() }; // illustrative, never launched
    let resolve = |name: &str| -> sn_core::Result<(u64, String)> {
        let t = g.tensor(name)?;
        Ok((t.offset, sn_core::qtypes::name(t.ty).to_string()))
    };

    let mut out = format!("{}: {} executions ({} in the trunk), arena {}\n", g.path.display(), plan.n_exec(), trunk(&plan), sn_hip::human(bytes));
    let fused = flag("--fused");
    let ladder = flag("--ladder");
    if ladder {
        // The fusion ladder: each row turns one more §6 P2 fusion on, so the count is attributable.
        let presets: [(&str, Fusion); 7] = [
            ("raw (one kernel per step of §2)", Fusion::RAW),
            ("+ norm -> gemv", Fusion { norm_into_gemv: true, ..Fusion::RAW }),
            ("+ silu -> gemv", Fusion { norm_into_gemv: true, silu_into_gemv: true, ..Fusion::RAW }),
            ("+ multi-gemv", Fusion { norm_into_gemv: true, silu_into_gemv: true, multi_gemv: true, ..Fusion::RAW }),
            ("+ rope -> gemv", Fusion { norm_into_gemv: true, silu_into_gemv: true, multi_gemv: true, rope_into_gemv: true, ..Fusion::RAW }),
            ("+ gate -> gemv", Fusion { norm_into_gemv: true, silu_into_gemv: true, multi_gemv: true, rope_into_gemv: true, gate_into_gemv: true, ..Fusion::RAW }),
            ("+ route+merge (= P2 target)", Fusion::FUSED),
        ];
        let mut prev: Option<usize> = None;
        for (label, f) in presets {
            let spec = decode::compile(&plan, &slots, bases, bytes, f, &resolve)?;
            let delta = prev.map(|p| spec.len() as i64 - p as i64);
            out += &format!("{label}: {:>5} launches ({:>4} launchable today)", spec.len(), spec.launchable());
            if let Some(dd) = delta {
                out += &format!("  {dd:+}");
            }
            out += "\n";
            prev = Some(spec.len());
        }
        return Ok(out);
    }
    let f = if fused { Fusion::FUSED } else { Fusion::RAW };
    let spec = decode::compile(&plan, &slots, bases, bytes, f, &resolve)?;
    out += &format!("\nfusion: {:?}\n", f.flags().iter().filter(|(_, on)| *on).map(|(n, _)| *n).collect::<Vec<_>>());
    out += &spec.report();
    if flag("--print") {
        for o in spec.ops.iter().take(80) {
            out += &format!("  {:<44} {:<20} {}\n", o.label, o.kernel, if o.pending { "pending" } else { "ok" });
        }
        if spec.ops.len() > 80 {
            out += &format!("  ... {} more\n", spec.ops.len() - 80);
        }
    }
    Ok(out)
}

fn trunk(plan: &Plan) -> usize {
    (0..plan.n_exec()).filter(|&il| plan.exec(il).mtp == 0).count()
}

/// `sn gpu phase6`: every Phase 6 kernel against its scalar oracle, on the checkpoint's geometry.
pub fn cmd_phase6(opt_val: &dyn Fn(&str) -> Option<String>) -> Result<String> {
    let path = opt_val("--model").ok_or_else(|| sn_core::Error::msg("phase6 needs --model <gguf>"))?;
    let g = Gguf::open(&path)?;
    let plan = Plan::build(Hyperparams::from_gguf(&g)?)?;
    let mut sub = Substrate::open()?;
    let mut reg = KernelRegistry::new();
    for p in sn_hip::bench::image_paths() {
        reg.add_image(&sub.dev, &p)?;
    }
    let n4 = sn_hip::ops::register_phase4(&sub.dev, &mut reg)?;
    let n5 = sn_hip::gemm::register_phase5(&sub.dev, &mut reg)?;
    let n6 = sn_hip::phase6::register_phase6(&sub.dev, &mut reg)?;
    let p6 = sn_hip::phase6::Phase6::resolve(&reg)?;
    let mut h = check::Harness::open(&mut sub, &mut reg)?;
    let checks = sn_hip::phase6::phase6_checks(&p6, &mut h, &plan.hp.dims, plan.hp.d_conv, plan.hp.rms_eps)?;
    let failing: Vec<&Check> = checks.iter().filter(|c| !c.ok()).collect();
    let mut out = format!("{}\n", sub.dev.info);
    out += &format!("model: {path}\n{} Phase 4 + {} Phase 5 + {} Phase 6 kernels registered\n", n4, n5, n6);
    out += "Phase 6 kernels vs sn_core::oracle (deterministic inputs, real geometry)\n";
    out += &check::report(&checks);
    out += &format!("\nGATE: {} ({} checks, {} failing)\n", if failing.is_empty() { "PASS" } else { "FAIL" }, checks.len(), failing.len());
    for c in &failing {
        out += &format!("  {} NMSE {:.3e} > {:.0e}\n", c.name, c.nmse, c.gate);
    }
    Ok(out)
}
