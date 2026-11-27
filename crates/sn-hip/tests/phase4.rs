//! Phase 4 correctness gates: kernel-vs-oracle, and the routing kernels replayed against the
//! Phase 2 reference stages (design-plan: every phase ends green with `cargo test` + a named gate).
//! The >= 100x fusion timing gate is in `phase4_fusion.rs`.
//!
//! Skip convention as in `phase3.rs` / `sn-core/tests/real_model.rs`: no ROCm device, or no kernel
//! image, prints a skip line instead of failing. Build the images first with
//! `./Tools/build-kernels.sh` (do **not** send its output to /dev/null: a failed compile leaves the
//! previous image in place - the loader refuses a stale one now, but only if the image is there).
//!
//! The golden test needs two things a plain checkout does not have, so it skips unless both exist:
//!   * the checkpoint (`SUPERNOVA_MODEL`, default path as in `real_model.rs`), and
//!   * Phase 2 stage dumps in `SN_STAGE_DUMP`, produced by
//!     `sn ref --model <gguf> --tokens-file Docs/golden/tokens/p1-capital.ids --dump <dir>
//!      --dump-only mhc.,l.,l_out.,gdn.,ffn.,attn. --limit-exec 12` - which is exactly what
//!     `Tools/kernel-check.py` does, followed by the same check on the GPU.

mod common;
use common::{model_path, with_harness};

use sn_core::gguf::Gguf;
use sn_core::plan::{Hyperparams, Plan};
use sn_hip::check;
use sn_hip::decode::{self, Bases, Fusion};


/// Gate: every Phase 4 kernel matches its scalar oracle in `sn-core/src/oracle.rs` (NMSE <= 1e-5,
/// and <= 1e-6 for the routing mix, which every downstream stage inherits).
#[test]
fn phase4_kernels_match_their_oracles() {
    let Some((report, checks)) = with_harness("kernel/oracle gate", |h| {
        let checks = check::kernel_checks(h).expect("kernel checks ran");
        (check::report(&checks), checks)
    }) else {
        return;
    };
    eprint!("{report}");
    let gated: Vec<&check::Check> = checks.iter().filter(|c| c.gate >= 0.0).collect();
    assert!(gated.len() >= 20, "expected the full Phase 4 kernel set to be checked, got {}", gated.len());
    let failed: Vec<&check::Check> = gated.iter().filter(|c| !c.ok()).cloned().collect();
    assert!(failed.is_empty(), "{} kernel(s) disagree with their oracle: {:?}", failed.len(), failed.iter().map(|c| (c.name.clone(), c.nmse)).collect::<Vec<_>>());
    // The sensitivity control has to stay sensitive: with 6 sweeps instead of the checkpoint's 20,
    // the mix must miss its gate by orders of magnitude, or the check proves nothing.
    let control = checks.iter().find(|c| c.name.contains("6 sweeps")).expect("sensitivity control present");
    assert!(!control.ok(), "the 6-sweep sensitivity control passed - the mix check is not measuring the sweeps");
}

/// Gate: the routing kernels reproduce the Phase 2 reference stage-for-stage on real checkpoint
/// weights and real decoded activations - the part of Phase 4 that a synthetic oracle cannot prove.
#[test]
fn phase4_route_replays_the_phase2_golden_stages() {
    let Some(model) = model_path() else {
        eprintln!("skip golden route gate: no checkpoint (set SUPERNOVA_MODEL)");
        return;
    };
    let dump = match std::env::var("SN_STAGE_DUMP") {
        Ok(d) if std::path::Path::new(&d).exists() => d,
        _ => {
            eprintln!("skip golden route gate: no Phase 2 stage dump (set SN_STAGE_DUMP; see this file's header)");
            return;
        }
    };
    let g = Gguf::open(&model).expect("checkpoint opens");
    let plan = Plan::build(Hyperparams::from_gguf(&g).expect("hyperparams")).expect("plan");
    // The first trunk execution and the last one: everything between re-reads the same blocks, so
    // this pins down both the weights-per-execution mapping and the stream carry.
    // Whatever the dump actually covers, first / middle / last of it: the documented `sn ref`
    // recipe dumps the first mHC executions of the trunk, and every one of them re-reads a
    // different block, so this pins the weights-per-execution mapping and the stream carry.
    let have: Vec<usize> = (plan.hp.mhc_start..plan.n_exec())
        .filter(|&il| plan.exec(il).mhc && plan.exec(il).mtp == 0)
        .filter(|&il| std::path::Path::new(&format!("{dump}/mhc.{il}.logits.txt")).exists())
        .collect();
    if have.is_empty() {
        eprintln!("skip golden route gate: {dump} has no mhc.<il>.logits.txt (regenerate it with the recipe in this file's header)");
        return;
    }
    let execs = vec![have[0], have[have.len() / 2], *have.last().expect("one execution")];
    eprintln!("replaying executions {} from {dump}", execs.iter().map(|e| e.to_string()).collect::<Vec<_>>().join(", "));
    let runs = with_harness("golden route gate", |h| {
        check::route_vs_golden(h, &g, &plan, &dump, &execs).expect("replay ran")
    });
    let Some(runs) = runs else { return };
    let mut worst_mix = 0.0f64;
    let mut worst_other = 0.0f64;
    let mut failed = Vec::new();
    for r in &runs {
        eprint!("{}", check::report(&r.checks));
        for c in &r.checks {
            if c.name.ends_with(".mix") {
                worst_mix = worst_mix.max(c.nmse);
            } else {
                worst_other = worst_other.max(c.nmse);
            }
            if !c.ok() {
                failed.push((c.name.clone(), c.nmse));
            }
        }
    }
    eprintln!("worst mix NMSE {worst_mix:.3e} (gate 1e-6), worst other {worst_other:.3e} (gate 1e-5)");
    assert!(failed.is_empty(), "routing diverged from the Phase 2 reference: {failed:?}");
}

/// Gate: what a decode step costs in launches, with and without the Phase 4 fusions. The plan's
/// "<= 300 nodes" target is NOT met and is reported as such (Docs/bench.md): the trunk alone has
/// ~350 weight matrices and 96 routing applications, so the honest claim is the measured ladder.
#[test]
fn phase4_decode_step_op_counts_are_recorded() {
    let Some(model) = model_path() else {
        eprintln!("skip op-count gate: no checkpoint (set SUPERNOVA_MODEL)");
        return;
    };
    let g = Gguf::open(&model).expect("checkpoint opens");
    let plan = Plan::build(Hyperparams::from_gguf(&g).expect("hyperparams")).expect("plan");
    let mut act = sn_hip::arena::Arena::new("decode-activations");
    let (slots, bytes) = decode::Slots::reserve(&mut act, &plan, 8192).expect("arena reserved");
    act.freeze();
    let bases = Bases { activations: 0x1_0000_0000, weights: 0x8_0000_0000, ..Default::default() }; // never launched
    let resolve = |name: &str| -> sn_core::Result<(u64, String)> {
        let t = g.tensor(name)?;
        Ok((t.offset, sn_core::qtypes::name(t.ty).to_string()))
    };
    let raw = decode::compile(&plan, &slots, bases, bytes, Fusion::RAW, &resolve).expect("raw compiles");
    let fused = decode::compile(&plan, &slots, bases, bytes, Fusion::FUSED, &resolve).expect("fused compiles");
    eprintln!("{}", fused.report());
    let route = |s: &decode::ProgramSpec, kind: &str| -> usize {
        s.counts().into_iter().find(|(k, _)| *k == kind).map(|(_, v)| v).unwrap_or(0)
    };
    // What Phase 4 actually claims: the fusions reduce the launch count, and routing becomes one
    // kernel per route instead of the fork's 480-node balancing.
    assert!(fused.len() < raw.len(), "fusion did not reduce the decode step ({} -> {})", raw.len(), fused.len());
    let (r_raw, r_fused) = (route(&raw, "route"), route(&fused, "route"));
    assert!(r_raw >= 96, "expected >= 96 routing applications (48 routers x attn+ffn), got {r_raw}");
    assert_eq!(r_raw, 2 * r_fused, "route+merge fusion should halve the routing kernels ({r_raw} -> {r_fused})");
    // The two boundaries of §2.7, asserted as *structure*. These were wrong once: nothing wrote
    // `a`/`b` before the first route read them, and the `0.5*(a+b)` exposure sat in the head, which
    // would have left the exit blocks reading a stale `cur` and fed the head the loop's streams.
    let at = |name: &str| -> Vec<usize> {
        fused.ops.iter().enumerate().filter(|(_, o)| o.kernel == name).map(|(k, _)| k).collect()
    };
    let seeds = at("sn_mhc_seed_f32");
    let exposes = at("sn_mhc_expose_f32");
    assert_eq!(seeds.len(), 1, "the streams are seeded exactly once, at the first mHC execution");
    assert_eq!(exposes.len(), 1, "the exposure appears exactly once, at the mHC -> non-mHC boundary");
    let first_route = fused.ops.iter().position(|o| o.kernel == "sn_mhc_route_f32").expect("a route op exists");
    assert!(seeds[0] < first_route, "the seed must precede the first route");
    let trunk: Vec<usize> = (0..plan.n_exec()).filter(|&k| plan.exec(k).mtp == 0).collect();
    let last_mhc = *trunk.iter().rev().find(|&&k| plan.exec(k).mhc).expect("the trunk has mHC executions");
    match trunk.iter().find(|&&k| k > last_mhc && !plan.exec(k).mhc).map(|&k| plan.exec(k).tensor_block) {
        Some(exit_block) => {
            let first_exit_op = fused
                .ops
                .iter()
                .position(|o| o.label.contains(&format!("blk.{exit_block}")))
                .expect("the exit blocks emit ops");
            assert!(
                exposes[0] < first_exit_op,
                "the exposure must precede block {exit_block}: outside the mHC range a branch reads `cur`"
            );
        }
        None => assert_eq!(exposes[0], fused.ops.len() - 3, "a trunk ending inside the mHC region exposes before the head"),
    }
    // A folded pre-norm and an explicit one are alternatives, never both: applying the norm twice is
    // what the `norm` row of the fused ladder used to count.
    let folded = fused.ops.iter().filter(|o| o.class == "gemv").count();
    let explicit_post = fused.ops.iter().filter(|o| o.kernel == "sn_rmsnorm_f32" && o.label.contains("post")).count();
    assert!(folded > 0 && explicit_post == 0, "fused program still launches {explicit_post} post-norm(s) it also folds");
    let raw_post = raw.ops.iter().filter(|o| o.kernel == "sn_rmsnorm_f32" && o.label.contains("post")).count();
    assert!(raw_post > 0, "the unfused ladder has to keep spelling the norm as its own kernel");

    // The plan's target, asserted as NOT met so this test has to be edited when it becomes true.
    if fused.len() <= 300 {
        panic!("the decode step is now {} launches: the plan's <= 300 target is met, update Docs/bench.md and Docs/design-plan.md", fused.len());
    }
    eprintln!("NOTE: {} launches fused ({} raw) - the plan's <= 300 target is not met; see Docs/bench.md Phase 4", fused.len(), raw.len());
}
