//! Helpers shared by the Phase 4 integration tests (`phase4.rs`, `phase4_fusion.rs`).
//!
//! The timing gate lives in its own test binary on purpose: cargo runs test binaries one at a time
//! but the tests *inside* a binary in parallel, and a concurrent kernel sweep on the same card
//! moves a measured ratio enough to fail a >= 100x gate.

#![allow(dead_code)]

use sn_hip::check;
use sn_hip::kernels::KernelRegistry;
use sn_hip::Substrate;

/// Every kernel image, by absolute path: `bench::image_paths()` globs `target/hip` relative to the
/// working directory, and integration tests run with the crate directory as theirs.
pub fn images() -> Vec<String> {
    if let Ok(v) = std::env::var("SN_KERNELS") {
        let v: Vec<String> = v.split(':').map(|s| s.trim()).filter(|s| !s.is_empty()).map(String::from).collect();
        if !v.is_empty() {
            return v;
        }
    }
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target/hip");
    let mut out: Vec<String> = std::fs::read_dir(&dir)
        .ok()
        .map(|rd| {
            rd.filter_map(|e| e.ok())
                .map(|e| e.path())
                .filter(|p| p.extension().map(|x| x == "fatbin").unwrap_or(false))
                .map(|p| p.to_string_lossy().into_owned())
                .collect()
        })
        .unwrap_or_default();
    out.sort();
    out
}

pub fn model_path() -> Option<String> {
    let p = std::env::var("SUPERNOVA_MODEL")
        .unwrap_or_else(|_| "/home/justin/Documents/AI/Models/SenseLoop/SenseNova-Loop-9B-Q4_K_M.gguf".into());
    std::path::Path::new(&p).exists().then_some(p)
}

/// Open the device, load every kernel image, resolve the Phase 4 set and hand the test a harness.
/// `None` means "this machine cannot run the gate", always with a printed reason.
pub fn with_harness<T>(what: &str, f: impl FnOnce(&mut check::Harness) -> T) -> Option<T> {
    let images = images();
    if images.is_empty() {
        eprintln!("skip {what}: no kernel image under target/hip (run ./Tools/build-kernels.sh)");
        return None;
    }
    let mut sub = match Substrate::open() {
        Ok(s) => s,
        Err(e) => {
            eprintln!("skip {what}: no HIP device ({e})");
            return None;
        }
    };
    let mut reg = KernelRegistry::new();
    for p in &images {
        if let Err(e) = reg.add_image(&sub.dev, p) {
            eprintln!("skip {what}: cannot load {p} ({e})");
            return None;
        }
    }
    let mut h = match check::Harness::open(&mut sub, &mut reg) {
        Ok(h) => h,
        Err(e) => {
            eprintln!("skip {what}: cannot set up the check harness ({e})");
            return None;
        }
    };
    Some(f(&mut h))
}

