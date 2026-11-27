//! Phase 5 gates: the quantised matmul surface.
//!
//! The correctness gate is design-plan P4 applied per kernel - device kernel vs `sn_core::dq` (the
//! reference dequantiser that Phase 2 cross-checked against GGML's C with `Tools/dequant_ref.py`)
//! over **the checkpoint's own weight bytes**, not a fixture - and the auto-pick claim is checked
//! against the sweep the same command measures. The throughput gates (GEMV bandwidth, GEMM
//! throughput) are numbers `Docs/bench.md` §3 quotes from `sn bench gemm`; they are deliberately
//! not asserted here, because a bandwidth gate fails for machine reasons and a test should not go
//! red over a slower card without saying so.
//!
//! Skip convention as in `phase4.rs`: no device, no kernel image, or no checkpoint prints a reason.

mod common;
use common::{images, model_path};

use sn_core::gguf::Gguf;
use sn_core::plan::{Hyperparams, Plan};
use sn_hip::gemm;
use sn_hip::kernels::KernelRegistry;
use sn_hip::Substrate;
use sn_hip::decode;

/// Device + every kernel image + the Phase 5 handles. `None` always prints why.
fn with_phase5<T>(what: &str, f: impl FnOnce(&Substrate, &KernelRegistry) -> T) -> Option<T> {
    let imgs = images();
    if imgs.is_empty() {
        eprintln!("skip {what}: no kernel image under Engine/target/hip (run ./Tools/build-kernels.sh)");
        return None;
    }
    let sub = match Substrate::open() {
        Ok(s) => s,
        Err(e) => {
            eprintln!("skip {what}: no HIP device ({e})");
            return None;
        }
    };
    let mut reg = KernelRegistry::new();
    for p in &imgs {
        if let Err(e) = reg.add_image(&sub.dev, p) {
            eprintln!("skip {what}: cannot load {p} ({e})");
            return None;
        }
    }
    if let Err(e) = gemm::register_phase5(&sub.dev, &mut reg) {
        eprintln!("skip {what}: the Phase 5 image does not provide its kernels ({e})");
        return None;
    }
    Some(f(&sub, &reg))
}

/// Gate: every GEMV and GEMM shape in the model, in every quantisation class the model actually
/// uses, plus the `norm -> gemv` form and the f32 paths against a synthetic matrix. NMSE <= 1e-4;
/// what this measures is the kernel's own error, and it has come out at 1e-12..1e-13 - f32
/// summation order and nothing else.
#[test]
fn phase5_matmul_matches_the_scalar_reference() {
    let Some(model) = model_path() else {
        eprintln!("skip matmul gate: no checkpoint (set SUPERNOVA_MODEL)");
        return;
    };
    let g = Gguf::open(&model).expect("checkpoint opens");
    let plan = Plan::build(Hyperparams::from_gguf(&g).expect("plan from the checkpoint")).expect("plan builds");
    let result = with_phase5("matmul correctness gate", |sub, reg| {
        let p5 = gemm::Phase5::resolve(reg).expect("phase 5 kernels resolve");
        let shapes = gemm::model_shapes(&g, &plan).expect("the model's GEMM shapes are enumerable");
        // Every role in the plan's table, and both quantisation classes where a role uses two.
        let roles = shapes.iter().filter(|s| !s.variant).count();
        assert!(roles >= 12, "expected the 12 GEMM roles of design-plan §1.2, found {roles}");
        assert!(
            shapes.iter().any(|s| s.role.ends_with("ffn_down.weight") && s.type_name == "q4_k"),
            "the shape table must cover both quantisation classes of ffn_down"
        );
        // Big enough for a T = 512 activation and output of the widest shape.
        let mut ws = gemm::upload_shapes(&g, &sub.dev, &shapes, 256 << 20).expect("weights upload");
        let mut checks =
            gemm::gemm_checks(&p5, &g, &mut ws, &shapes, &plan, &sub.dev, sub.stream, None, None).expect("gemv/gemm checks run");
        checks.extend(gemm::f32_checks(&p5, &mut ws, &sub.dev, sub.stream, &plan).expect("f32 checks run"));
        (sn_hip::check::report(&checks), checks)
    });
    let Some((report, checks)) = result else { return };
    eprint!("{report}");
    // 15 shape/type entries x 2 (plain + folded norm) = 30 GEMV, 4 x 3 = 12 GEMM, 5 synthetic f32.
    assert!(checks.len() >= 47, "expected the whole Phase 5 surface to be checked, got {}", checks.len());
    assert!(checks.iter().all(|c| c.gate > 0.0), "every Phase 5 check carries a gate");
    let failed: Vec<(_, _)> = checks
        .iter()
        .filter(|c| !c.ok())
        .map(|c| (c.name.clone(), c.nmse))
        .collect();
    assert!(failed.is_empty(), "{} matmul kernel(s) disagree with sn-core's reference: {:?}", failed.len(), failed);
}

/// Gate: the decode program's GEMV nodes are launchable, and their launch geometry comes from the
/// same policy the bench validates. Phase 4 recorded these as pending by name; Phase 5 owns the
/// names, so a program that still reports them pending would mean the two halves disagree about the
/// ABI - and a baked-in grid would mean the pick is not the pick.
#[test]
fn phase5_decode_gemv_nodes_are_launchable_with_the_picked_geometry() {
    let Some(model) = model_path() else {
        eprintln!("skip decode gemv ABI check: no checkpoint (set SUPERNOVA_MODEL)");
        return;
    };
    let g = Gguf::open(&model).expect("checkpoint opens");
    let plan = Plan::build(Hyperparams::from_gguf(&g).expect("hyperparams")).expect("plan");
    let mut act = sn_hip::arena::Arena::new("decode-activations");
    let (slots, bytes) = decode::Slots::reserve(&mut act, &plan, 8192).expect("arena reserved");
    act.freeze();
    let bases = decode::Bases { activations: 0x1_0000_0000, weights: 0x8_0000_0000, ..Default::default() }; // never launched
    let resolve = |name: &str| -> sn_core::Result<(u64, String)> {
        let t = g.tensor(name)?;
        Ok((t.offset, sn_core::qtypes::name(t.ty).to_string()))
    };
    let spec = decode::compile(&plan, &slots, bases, bytes, decode::Fusion::FUSED, &resolve).expect("fused compiles");
    eprintln!("{}", spec.report());
    let gemvs: Vec<&decode::OpSpec> = spec.ops.iter().filter(|o| o.class == "gemv").collect();
    assert_eq!(gemvs.len(), 267, "the decode step multiplies 267 weight matrices");
    // The per-tensor GEMVs are all launchable, and so is the one multi-GEMV fusion whose operands
    // always match (`ffn_gate` + `ffn_up`: same type, same shape, in every block - Phase 7's
    // `sn_gemv_q4_k_pair`). What may stay pending is the `q+k+v` triple, because `attn_v` is Q6_K in
    // five blocks while q/k are Q4_K, so one entry point cannot serve all three. Naming the rest
    // keeps "what is missing is countable" true.
    let still_pending: Vec<String> = gemvs.iter().filter(|o| o.pending).map(|o| o.kernel.clone()).collect();
    assert!(
        still_pending.iter().all(|k| k.starts_with("gemv_multi_")),
        "only the multi-GEMV fusions may remain pending, got {:?}",
        {
            let mut u: Vec<String> = still_pending.clone();
            u.sort();
            u.dedup();
            u
        }
    );
    assert_eq!(still_pending.len(), 14, "one qkv triple per full-attention block, still unimplemented (the gate/up pair is a real kernel since Phase 7)");
    let pairs: Vec<&decode::OpSpec> = gemvs.iter().copied().filter(|o| o.kernel == "gemv_q4_k_pair").collect();
    assert_eq!(pairs.len(), 56, "one gate/up pair launch per FFN execution");
    for op in &pairs {
        // (W1, W2, x, y1, y2, k, n, norm_w, eps, warps_per_row)
        assert_eq!(op.args.len(), 10, "the pair GEMV ABI");
        let (k, n) = match (&op.args[5], &op.args[6]) {
            (sn_hip::program::Arg::I32(k), sn_hip::program::Arg::I32(n)) => (*k as usize, *n as usize),
            _ => panic!("k and n must be ints"),
        };
        let wpr = match op.args[9] {
            sn_hip::program::Arg::I32(v) => v as usize,
            _ => panic!("warps_per_row must be an int"),
        };
        assert_eq!(wpr, gemm::pick_warps(n, k, gemm::PICK_CUS), "the pair's split is the policy's for one matrix");
        assert_eq!(op.grid, gemm::gemv_grid(2 * n, wpr), "the pair's grid covers both matrices");
    }
    for probe in ["gemv_q4_k", "gemv_q6_k", "gemv_f32"] {
        let op = gemvs.iter().find(|o| o.kernel == probe);
        match (probe, op) {
            ("gemv_f32", None) => continue, // no f32 2-D weight in this file (design-plan §1.2)
            (_, Some(op)) => {
                assert_eq!(op.args.len(), 8, "the GEMV ABI is (W, x, y, k, n, norm_w, eps, warps_per_row)");
                let k = match op.args[3] {
                    sn_hip::program::Arg::I32(v) => v as usize,
                    _ => panic!("k must be an int"),
                };
                let n = match op.args[4] {
                    sn_hip::program::Arg::I32(v) => v as usize,
                    _ => panic!("n must be an int"),
                };
                let wpr = match op.args[7] {
                    sn_hip::program::Arg::I32(v) => v as usize,
                    _ => panic!("warps_per_row must be an int"),
                };
                assert_eq!(op.block, sn_hip::GEMV_BLOCK, "the GEMV workgroup is SN_GEMV_BLOCK");
                assert_eq!(wpr, gemm::pick_warps(n, k, gemm::PICK_CUS), "{probe}: the node's split must be the policy's");
                assert_eq!(op.grid, gemm::gemv_grid(n, wpr), "{probe}: the grid must follow from the pick");
            }
            (_, None) => panic!("a {probe} node is expected in this checkpoint's program"),
        }
    }
}
