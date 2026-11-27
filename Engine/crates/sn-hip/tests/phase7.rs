//! Phase 7 gates: the decode fast path.
//!
//! Two things are asserted here that nothing else in the workspace covers.
//!
//! 1. **The Q6_K repack.** The weight slab is *not* the GGUF data section: every Q6_K tensor is
//!    stored with 224-byte super-blocks so the GEMV can use vector loads (`Docs/bench.md` §6).
//!    That is only safe if every consumer resolves addresses through the same map, and a mistake
//!    there is a wrong-number bug, not a fault (it shows up as NaN logits, which is how it was
//!    found the day it landed). So: the map's arithmetic is checked without a device, and a forward
//!    through the real model is checked *against the single-sequence engine* with a device.
//!
//! 2. **`n_seq = 2` with divergent prefixes** - the Phase 6 gate that was never run through a
//!    forward. The rule it tests: two sequences in one engine produce the tokens each produces on
//!    its own, in either order, and `Engine::reset(seq)` clears that sequence and nothing else.
//!    `reset` is the first user of `Slab::clear_range`, and a GDN state that survives a reset is
//!    not a crash, it is a plausible wrong continuation - which is why this compares token streams
//!    from a second engine rather than trusting that the offsets are right.
//!
//! Skip convention as in `phase5.rs`: no device, no kernel image, or no checkpoint prints a reason.

mod common;
use common::{images, model_path};

/// The repack's arithmetic, without a device: what `Substrate::upload_weights` and
/// `gemm::upload_shapes` both build their layouts from.
#[test]
fn q6k_repack_grows_only_q6_k_and_keeps_every_field_in_place() {
    use sn_core::dq::{Q6K_BLOCK, Q6K_BLOCK_PADDED};
    const _: () = assert!(Q6K_BLOCK_PADDED > Q6K_BLOCK && Q6K_BLOCK_PADDED % 16 == 0, "the whole point is a larger, 16-byte-aligned super-block");

    // A byte pattern that is not all-equal, so a shifted or truncated copy cannot pass.
    let src: Vec<u8> = (0..(Q6K_BLOCK * 3)).map(|i| (i as u32 * 7 + 11) as u8 % 251).collect();
    let mut dst = vec![0u8; Q6K_BLOCK_PADDED * 3];
    let n = sn_core::dq::q6k_pad(&src, &mut dst).unwrap();
    assert_eq!(n, Q6K_BLOCK_PADDED * 3);
    for b in 0..3 {
        let from = &src[b * Q6K_BLOCK..(b + 1) * Q6K_BLOCK];
        let to = &dst[b * Q6K_BLOCK_PADDED..b * Q6K_BLOCK_PADDED + Q6K_BLOCK];
        assert_eq!(from, to, "super-block {b} must be copied verbatim at its own offset");
        assert!(dst[b * Q6K_BLOCK_PADDED + Q6K_BLOCK..][..Q6K_BLOCK_PADDED - Q6K_BLOCK].iter().all(|&v| v == 0), "padding must be zero");
    }
    // Not a whole number of super-blocks is an error, not a silent short copy.
    assert!(sn_core::dq::q6k_pad(&src[..src.len() - 1], &mut dst).is_err());

    // The size rule, over the real checkpoint: Q6_K grows by exactly 224/210, nothing else moves,
    // and every Q6_K tensor has an element count that is a whole number of super-blocks (which is
    // what makes padding super-blocks the same thing as padding rows).
    let Some(model) = model_path() else {
        eprintln!("skip size rule: no checkpoint (set SUPERNOVA_MODEL)");
        return;
    };
    let g = sn_core::gguf::Gguf::open(&model).unwrap();
    let mut grew = 0usize;
    let mut grew_bytes = 0u64;
    for t in g.tensors() {
        let dev = sn_core::dq::device_size(t);
        if t.ty == sn_core::qtypes::ty::Q6_K {
            assert_eq!(t.n_elts() % 256, 0, "{}: Q6_K row is not a whole number of super-blocks", t.name);
            assert_eq!(dev, (t.size as usize / Q6K_BLOCK) * Q6K_BLOCK_PADDED, "{}", t.name);
            grew += 1;
            grew_bytes += dev as u64 - t.size;
        } else {
            assert_eq!(dev, t.size as usize, "{}: a non-Q6_K tensor must not move", t.name);
        }
    }
    assert!(grew >= 4, "this checkpoint has Q6_K tensors ({} seen)", grew);
    eprintln!("repack: {grew} Q6_K tensors, +{grew_bytes} B (super-blocks 210 -> {Q6K_BLOCK_PADDED} B)");
}

/// The Phase 6/7 gate this closes: two sequences, two different prompts, one engine - the same
/// tokens as one sequence at a time, in either order, with `Engine::reset` between.
#[test]
fn two_sequences_with_divergent_prefixes_agree_with_running_apart() {
    let Some(model) = model_path() else {
        eprintln!("skip n_seq gate: no checkpoint (set SUPERNOVA_MODEL)");
        return;
    };
    // `Engine::open` globs `target/hip` relative to the working directory; an integration test runs
    // with the crate directory as its own, so point it at absolute paths.
    let imgs = images();
    if imgs.is_empty() {
        eprintln!("skip n_seq gate: no kernel image under target/hip (run ./Tools/build-kernels.sh)");
        return;
    }
    unsafe { std::env::set_var("SN_KERNELS", imgs.join(":")) };
    let g = sn_core::gguf::Gguf::open(&model).unwrap();
    let plan = match sn_core::plan::Hyperparams::from_gguf(&g).and_then(sn_core::plan::Plan::build) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("skip n_seq gate: {e}");
            return;
        }
    };
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let read = |id: &str| -> Vec<u32> {
        let p = root.join(format!("Docs/golden/tokens/{id}.ids"));
        match sn_hip::driver::read_ids(p.to_str().unwrap()) {
            Ok(v) => v,
            Err(e) => {
                eprintln!("skip n_seq gate: {p:?}: {e}");
                Vec::new()
            }
        }
    };
    let a = read("p1-capital");
    let b = read("p2-arithmetic");
    if a.is_empty() || b.is_empty() {
        eprintln!("skip n_seq gate: no prompt ids read from Docs/golden/tokens");
        return;
    }
    assert_ne!(a, b, "the gate is about *divergent* prefixes");
    let n_new = 4;

    // ---- one engine, two sequences ----------------------------------------------------
    let cfg = sn_hip::driver::EngineCfg { n_ctx: 512, n_seq: 2, n_snap: 0, kv: sn_hip::cache::KvKind::F16, ..Default::default() };
    let mut both = match sn_hip::driver::Engine::open(&g, &plan, &cfg) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("skip n_seq gate: {e}");
            return;
        }
    };
    assert_eq!(both.ops_for(0), both.ops_for(1), "both sequences compile the same program (different cache offsets)");

    // The repack map, through the engine that actually baked it. `decode::compile_*` bakes
    // `Bases.weights + offset`, so the driver must hand it the *device* offset; a mistake there
    // moves every tensor after the first Q6_K one and shows up as NaN logits, not an error.
    {
        let sub = both.substrate();
        assert!(sub.weights_padded(), "the engine's weight slab is the padded layout");
        let map = sub.weight_offsets();
        assert!(!map.is_empty(), "the map must name every tensor the repack moved");
        let base = sub.weights().unwrap().addr();
        let total = sub.weights().unwrap().len();
        let sum: u64 = g.tensors().iter().map(|t| sn_core::dq::device_size(t) as u64).sum();
        assert!((total as u64) >= sum && (total as u64) - sum < 32 * g.tensors().len() as u64, "slab {total} vs the sum of device sizes {sum}");
        // Every tensor resolves inside the slab, in non-overlapping ascending order: the property
        // the whole forward depends on. A repack that put two tensors on top of each other would be
        // a wrong-number bug with no error anywhere.
        let mut prev_end = 0u64;
        let mut moved = 0usize;
        for t in g.tensors() {
            let at = sub.weights_addr(t.offset).unwrap() - base;
            assert!(at >= prev_end, "{}: overlaps the previous tensor", t.name);
            assert!((at as usize) + sn_core::dq::device_size(t) <= total, "{}: past the end of the slab", t.name);
            prev_end = at + sn_core::dq::device_size(t) as u64;
            if map.get(&t.offset).is_some() {
                assert_ne!(at, t.offset, "{} is in the map, so it must have moved", t.name);
                moved += 1;
            }
        }
        assert!(moved > 100, "33 Q6_K tensors move everything after them; only {moved} are mapped");
        // The tensor the file puts first keeps its offset (nothing has grown yet); the one right
        // after it cannot, and that is exactly the case a stale GGUF offset gets wrong.
        for name in ["output.weight", "output_norm.weight", "blk.7.ffn_down.weight", "blk.12.attn_qkv_gate.weight"] {
            let t = g.tensor(name).unwrap();
            let baked = *map.get(&t.offset).unwrap_or(&t.offset);
            assert_eq!(baked, sub.weights_addr(t.offset).unwrap() - base, "{name}: baked offset vs the slab");
        }
        let on = g.tensor("output_norm.weight").unwrap();
        assert_ne!(sub.weights_addr(on.offset).unwrap() - base, on.offset, "every tensor after the first Q6_K one has moved");
    }
    let a_both = both.generate(0, &a, n_new, |_, _, _| {}).unwrap();
    let b_both = both.generate(1, &b, n_new, |_, _, _| {}).unwrap();
    assert_eq!(both.sessions().len(0), a.len() + n_new);
    assert_eq!(both.sessions().len(1), b.len() + n_new);
    assert_ne!(a_both, b_both, "two different prompts must not converge to one continuation (that is what a shared cache region looks like)");

    // Reset sequence 0 only; sequence 1's cache length must survive it.
    both.reset(0).unwrap();
    assert_eq!(both.sessions().len(0), 0, "reset(0) cleared sequence 0");
    assert_eq!(both.sessions().len(1), b.len() + n_new, "reset(0) must not touch sequence 1");

    // ---- the same engine, cleared, with the order swapped -------------------------------
    both.reset(1).unwrap();
    let b_swapped = both.generate(1, &b, n_new, |_, _, _| {}).unwrap();
    let a_swapped = both.generate(0, &a, n_new, |_, _, _| {}).unwrap();
    assert_eq!(b_swapped, b_both, "sequence 1 after a reset must reproduce itself (a GDN state that survived reset(1) would not)");
    assert_eq!(a_swapped, a_both, "sequence 0 likewise");
    drop(both);

    // ---- one sequence at a time, in a second engine --------------------------------------
    let cfg1 = sn_hip::driver::EngineCfg { n_seq: 1, ..cfg.clone() };
    let mut solo = match sn_hip::driver::Engine::open(&g, &plan, &cfg1) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("n_seq gate: cannot open the single-sequence control ({e})");
            return;
        }
    };
    let a_solo = solo.generate(0, &a, n_new, |_, _, _| {}).unwrap();
    solo.reset(0).unwrap();
    let b_solo = solo.generate(0, &b, n_new, |_, _, _| {}).unwrap();
    assert_eq!(a_solo, a_both, "n_seq=2 must give the same continuation as n_seq=1 (prompt A)");
    assert_eq!(b_solo, b_both, "n_seq=2 must give the same continuation as n_seq=1 (prompt B)");
}
