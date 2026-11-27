//! `sn hip` and `sn bench graph` - the Phase 3 surface: what device we have, what the substrate
//! allocated, and the graph-replay gate with its numbers.

use sn_core::Result;
use sn_hip::bench::{image_paths, open_registry, register_probes, run_graph_gate};
use sn_hip::KernelRegistry;
use sn_hip::Substrate;

/// `sn hip [info]`: device, ISA, memory, and the kernel image that was built.
pub fn cmd_hip(
    positional: &[String],
    opt_val: impl Fn(&str) -> Option<String>,
    flag: impl Fn(&str) -> bool,
) -> Result<String> {
    let mut sub = Substrate::open()?;
    let mut out = String::new();
    out += &format!("{}\n", sub.dev.info);
    let paths = match opt_val("--kernels") {
        Some(v) => vec![v],
        None => image_paths(),
    };
    let mut reg = KernelRegistry::new();
    let mut loaded = Vec::new();
    for p in &paths {
        match reg.add_image(&sub.dev, p) {
            Ok(()) => loaded.push(p.clone()),
            Err(e) => out += &format!("kernel image {p}: {e}\n"),
        }
    }
    if !loaded.is_empty() {
        // The graph probes plus the Phase 4 elementwise/routing set, so `sn hip info` lists what a
        // decode program can actually launch today.
        register_probes(&sub, &mut reg)?;
        let phase4 = sn_hip::ops::register_phase4(&sub.dev, &mut reg);
        out += &format!("kernel images {}: {} registered\n", loaded.join(" "), reg.len());
        if let Err(e) = phase4 {
            out += &format!("  (Phase 4 kernel set not registered: {e})\n");
        }
        for k in reg.keys() {
            out += &format!("  {k}\n");
        }
    }
    if positional.get(1).map(|s| s.as_str()) == Some("ops") {
        return crate::gpu::cmd_ops(positional, &opt_val, &flag);
    }
    if let Some(model) = opt_val("--weights") {
        // The single-allocation weight path: one hipMalloc + one hipMemcpy of the data section.
        let g = sn_core::gguf::Gguf::open(&model)?;
        let t = std::time::Instant::now();
        let n = sub.upload_weights(&g)?;
        let secs = t.elapsed().as_secs_f64();
        let (free, total) = sub.free_device_memory()?;
        out += &format!(
            "weights uploaded: {} bytes in {:.2} s ({:.2} GB/s) -> one allocation; device now {:.3} GB free of {:.3} GB\n",
            n,
            secs,
            (n as f64 / 1e9) / secs,
            free as f64 / 1e9,
            total as f64 / 1e9
        );
        if let Some(t0) = g.tensors().first() {
            let a = sub.weights_addr(t0.offset)?;
            out += &format!("  {} at device 0x{:x} (base 0x{:x} + {} bytes)\n", t0.name, a, sub.weights()?.addr(), t0.offset);
        }
        return Ok(out);
    }
    let (free, total) = sub.free_device_memory()?;
    out += &format!(
        "device memory after open: {:.3} GB free of {:.3} GB\n",
        free as f64 / 1e9,
        total as f64 / 1e9
    );
    let _ = positional;
    Ok(out)
}

/// `sn bench graph`: the Phase 3 acceptance gate (design-plan §7).
pub fn cmd_bench(positional: &[String], opt_val: impl Fn(&str) -> Option<String>) -> Result<String> {
    match positional.get(1).map(|s| s.as_str()) {
        Some("fusion") => return crate::gpu::cmd_fusion(&opt_val),
        Some("graph") => {}
        _ => {
            return Ok("sn bench graph  [--ops N] [--reps N] [--steps-per-op K] [--kernels PATH]
                      sn bench fusion [--reps N] [--sweeps N] [--model M]
".into())
        }
    }
    let num = |name: &str, d: usize| -> Result<usize> {
        match opt_val(name) {
            None => Ok(d),
            Some(v) => v
                .parse::<usize>()
                .map_err(|e| sn_core::Error::msg(format!("{name}: {e}"))),
        }
    };
    let ops = num("--ops", 1000)?;
    let reps = num("--reps", 200)?;

    let mut sub = Substrate::open()?;
    let mut reg = open_registry(&sub.dev)?;
    register_probes(&sub, &mut reg)?;
    let gate = run_graph_gate(&mut sub, &reg, ops, reps, num("--steps-per-op", 1)?)?;
    let mut out = gate.report();
    let failed: Vec<_> = gate.passes().into_iter().filter(|(_, ok, _)| !*ok).collect();
    if failed.is_empty() {
        out += "GATE: PASS\n";
    } else {
        out += "GATE: FAIL\n";
        for (what, _, _) in failed {
            out += &format!("  {what}\n");
        }
    }
    Ok(out)
}
