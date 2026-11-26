//! The Phase 3 gate as a test, so `cargo test` alone proves the substrate still works
//! (design-plan: every phase ends green with `cargo test` + a named gate).
//!
//! Same convention as `sn-core/tests/real_model.rs`: when the machine has no ROCm device or the
//! kernel image has not been built, the test prints a skip line instead of failing. Run
//! `./Tools/build-kernels.sh` first to make it real; `SN_KERNELS` overrides the image path.

use sn_hip::bench::{image_path, register_probes, run_graph_gate};
use sn_hip::kernels::KernelRegistry;
use sn_hip::Substrate;

fn substrate() -> Option<(Substrate, KernelRegistry)> {
    let image = image_path();
    if !std::path::Path::new(&image).exists() {
        eprintln!("skip: kernel image {image} is not built (run ./Tools/build-kernels.sh)");
        return None;
    }
    let sub = match Substrate::open() {
        Ok(s) => s,
        Err(e) => {
            eprintln!("skip: no HIP device ({e})");
            return None;
        }
    };
    let mut reg = match KernelRegistry::load_image(&sub.dev, &image) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("skip: cannot load {image} ({e})");
            return None;
        }
    };
    register_probes(&sub, &mut reg).expect("probe kernels register");
    Some((sub, reg))
}

/// `Docs/design-plan.md` Phase 3 gate, at a size that runs in well under a second.
#[test]
fn phase3_graph_gate() {
    let Some((mut sub, reg)) = substrate() else { return };
    let gate = run_graph_gate(&mut sub, &reg, 200, 60, 1).expect("gate measured");
    eprintln!("{}", gate.report());

    // P1: the host must not be the bottleneck. One submission per replay, at any op count.
    assert!(
        gate.host_us_per_replay < 30.0,
        "host cost per graph replay is {:.1} us (target < 30)",
        gate.host_us_per_replay
    );
    // Replay must be pure replay: no re-initialisation, no re-ordering, no drift.
    assert!(gate.bit_identical, "replaying the same graph changed its output");
    // The input path the decode program depends on: a captured graph sees new pinned-host writes
    // without any hipGraphExec update.
    assert!(
        gate.pinned_input_visible,
        "a replayed graph did not observe a new pinned-host input"
    );
}

/// A captured graph is a durable object: it replays after the stream it was captured on is gone,
/// and its nodes keep the device pointers baked into them at capture time.
#[test]
fn captured_graph_survives_its_capture_stream() {
    let Some((mut sub, reg)) = substrate() else { return };
    sub.alloc_activations(64).expect("activation slab");
    let base = sub.activations().expect("slab").addr();
    let step = reg
        .require("sn_chain_step", &[], sn_hip::kernels::DType::Any)
        .expect("chain kernel");

    let mut b = sn_hip::program::ProgramBuilder::new(&sub.dev);
    for i in 0..8 {
        b.kernel(&format!("step {i}"), step, [1, 1, 1], 0, &[sn_hip::program::Arg::Ptr(base)])
            .expect("node");
    }
    let prog = b.finish();

    let cap = sub.new_stream().expect("capture stream");
    let graph = sn_hip::program::Graph::capture(&sub.dev, cap, &prog).expect("captured");
    sub.destroy_stream(cap);

    let read = |sub: &Substrate| -> f32 {
        let mut bits = [0u8; 4];
        sn_hip::hip::check(
            "readback",
            sub.dev.api.hipMemcpy.unwrap()(
                bits.as_mut_ptr() as *mut std::os::raw::c_void,
                base as *const std::os::raw::c_void,
                4,
                sn_hip::hip::MEMCPY_DEVICE_TO_HOST,
            ),
            &sub.dev.api,
        )
        .expect("read back");
        f32::from_le_bytes(bits)
    };

    let zero = || sub.activations().expect("slab").zero(&sub.dev.api).expect("zero");
    let expected = {
        let mut x: f32 = 0.0;
        for _ in 0..8 {
            x = x * 1.0000001f32 + 1.0;
        }
        x
    };
    for round in 0..3 {
        zero();
        graph.replay(&sub.dev, sub.stream).expect("replay");
        sub.sync().expect("sync");
        let got = read(&sub);
        assert!(
            (got - expected).abs() < 1e-6,
            "round {round}: 8 captured chain steps gave {got}, expected {expected}"
        );
    }
}
