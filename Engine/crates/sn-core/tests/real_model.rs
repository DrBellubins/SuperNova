//! Gates that run against the real checkpoint when it is available.
//!
//! `SUPERNOVA_MODEL` overrides the path; if the file is missing the test prints a skip line
//! rather than failing, so `cargo test` stays usable on machines without the 5.8 GB download.

use sn_core::budget::{
    build, gdn_state_bytes, gdn_state_per_layer, kv_bytes_per_token, kv_bytes_per_token_unique,
    n_kv_layers,
};
use sn_core::gguf::Gguf;
use sn_core::plan::{Hyperparams, Plan};
use std::collections::BTreeMap;

fn model_path() -> Option<std::path::PathBuf> {
    let p = std::env::var("SUPERNOVA_MODEL")
        .unwrap_or_else(|_| "/home/justin/Documents/AI/Models/SenseLoop/SenseNova-Loop-9B-Q4_K_M.gguf".into());
    let path = std::path::PathBuf::from(p);
    path.exists().then_some(path)
}

#[test]
fn checkpoint_census_execution_plan_and_budget() {
    let Some(path) = model_path() else {
        eprintln!("skip: checkpoint not present (set SUPERNOVA_MODEL)");
        return;
    };
    let gguf = Gguf::open(&path).expect("open checkpoint");
    let hp = Hyperparams::from_gguf(&gguf).expect("metadata parses");
    let plan = Plan::build(hp).expect("plan validates");

    // --- execution plan ----------------------------------------------------------------------
    assert_eq!(plan.execs.len(), 57);
    assert_eq!(plan.n_layer, 56);
    assert_eq!(plan.extra, 24);
    assert_eq!(plan.n_physical, 33);
    assert_eq!(plan.width, 24);
    assert_eq!(plan.n_mhc_executions(), 48, "the source of the fork's 46,080 nodes/token");
    assert_eq!(plan.n_gdn_executions(), 42);
    assert_eq!(plan.n_attn_executions(), 15);
    for (il, ph) in [(28, 4usize), (51, 27), (52, 28), (55, 31), (56, 32)] {
        assert_eq!(plan.exec(il).physical, ph);
    }
    assert_eq!(plan.exec(52).tensor_block, 52, "blk.52 holds physical 28");
    assert_eq!(plan.exec_count_of_gguf_block(52), 1);
    assert_eq!(plan.exec_count_of_gguf_block(4), 2);

    // --- metadata the forward depends on ------------------------------------------------------
    assert_eq!(plan.hp.vocab, 248320);
    assert_eq!(plan.hp.n_embd, 4096);
    assert_eq!(plan.hp.n_rot, 64, "partial RoPE over 128 of 256 dims");
    assert_eq!(plan.hp.rope_sections, vec![11, 11, 10, 0]);
    assert_eq!(plan.hp.mhc_iters, 20);
    assert_eq!(gguf.n_tensors(), 490);

    // --- tensor census (Tools/tensor_census.py) ----------------------------------------------
    let b = build(&plan, &gguf).expect("budget");
    let mut census: BTreeMap<String, (u64, u64)> = BTreeMap::new();
    let mut visits = 0u64;
    let mut multi = 0usize;
    let mut never = Vec::new();
    for r in &b.rows {
        let e = census.entry(r.type_name.clone()).or_insert((0, 0));
        e.0 += 1;
        e.1 += r.stored_bytes;
        visits += r.visits as u64;
        if r.visits == 2 {
            multi += 1;
        } else if r.visits == 0 {
            never.push(r.name.clone());
        }
    }
    assert_eq!(
        census.iter().map(|(k, v)| (k.as_str(), v.0, v.1)).collect::<Vec<_>>(),
        vec![
            ("f32", 280, 16_913_920),
            ("q4_k", 177, 3_824_418_816),
            ("q6_k", 33, 1_966_325_760),
        ],
        "only f32, Q4_K and Q6_K are present - no Q2_K"
    );
    assert_eq!(b.rows.len(), 490);
    assert_eq!(b.totals.n_tensors, 490);
    assert_eq!(b.totals.stored_bytes, 5_807_658_496);
    assert_eq!(b.totals.bytes_per_token, 8_813_935_104, "x1.52 loop re-read factor");
    // 378 of the 490 tensors live in loop bodies and are streamed twice; the 15 tensors of the
    // MTP block are never streamed by the trunk (they run once per *speculated* token, Phase 10).
    assert_eq!(visits, 853);
    assert_eq!(multi, 378);
    assert_eq!(never.len(), 15);
    assert!(never.iter().all(|n| n.starts_with("blk.56.")), "{never:?}");
    assert_eq!(b.row("token_embd.weight").unwrap().type_id, 12, "token_embd is Q4_K");
    assert_eq!(b.row("output.weight").unwrap().type_id, 14, "the output head is Q6_K");
    // blk.52 is physical 28, a GDN block outside the loop: read once, never re-streamed.
    assert_eq!(b.row("blk.52.ssm_out.weight").unwrap().visits, 1);
    // blk.4 is the first loop body block: every one of its tensors is streamed twice.
    assert_eq!(b.row("blk.4.attn_qkv_gate.weight").unwrap().visits, 2);
    assert_eq!(b.row("blk.4.attn_norm.weight").unwrap().visits, 2);
    assert_eq!(b.row("blk.7.attn_q.weight").unwrap().visits, 2, "blk.7 is a full-attention loop block");

    // --- caches (design plan §3) -------------------------------------------------------------
    assert_eq!(n_kv_layers(&plan), 14);
    // f16 KV: 2 caches x 14 full-attention executions x 4 kv heads x 256 dims x 2 bytes.
    assert_eq!(kv_bytes_per_token(&plan, 2), 57_344);
    // ... but only 8 *distinct* blocks back those 14 visits (3,7,11,15,19,23,27 and blk.55 =
    // physical 31), which is the figure a cache-sharing engine would use. Diagnostic only.
    assert_eq!(kv_bytes_per_token_unique(&plan, 2), 32_768);
    assert_eq!(kv_bytes_per_token(&plan, 1), 28_672, "q8_0 KV halves it");
    // GDN state: 42 executions x (32 x 128 x 128 x 4 + 3 x 8192 x 4) = 42 x 2,195,456.
    assert_eq!(gdn_state_per_layer(&plan), 2_195_456);
    assert_eq!(gdn_state_bytes(&plan), 92_209_152);
}

/// The dot (`dot_row`) and the read (`dequant_row`) paths are separate code in `dq.rs`, and every
/// projection in the reference forward uses the dot path. A real checkpoint row is the only input
/// that exercises the 6-bit scale packing with the mixed-sign super-blocks a trained model has, so
/// this pins the two against each other per tensor type in the file.
#[test]
fn dot_path_matches_the_dequantised_row_on_real_weights() {
    let Some(path) = model_path() else {
        eprintln!("skip: checkpoint not present (set SUPERNOVA_MODEL)");
        return;
    };
    let gguf = Gguf::open(&path).expect("open checkpoint");
    // deterministic activation, no dependency on the forward
    let mut seed: u64 = 0x243F6A8885A308D3;
    let mut next = move || {
        seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((seed >> 33) as f32 / (1u32 << 31) as f32) - 0.5
    };
    for (name, rows) in [
        ("blk.0.attn_qkv_gate.weight", vec![0usize, 5, 4096, 8191]),
        ("blk.4.attn_qkv_gate.weight", vec![0, 8191]),
        ("blk.0.ffn_gate.weight", vec![0, 12287]),
        ("blk.3.attn_q.weight", vec![0, 4095, 8191]),
        ("blk.52.ssm_out.weight", vec![0, 4095]),
        ("blk.4.hc_attn_fn.weight", vec![0, 7]),
        ("output.weight", vec![0, 7, 248319]),
    ] {
        let m = sn_core::dq::Mat::by_name(&gguf, name).expect("matrix");
        let mut x = Vec::with_capacity(m.ne0);
        for _ in 0..m.ne0 {
            x.push(next());
        }
        let mut v = vec![0.0f32; m.ne0];
        for &j in &rows {
            let dot = m.dot_row(j, &x);
            m.dequant_row(j, &mut v);
            let ref_dot: f64 = v.iter().zip(x.iter()).map(|(a, b)| *a as f64 * *b as f64).sum();
            let scale = ref_dot.abs().max(1.0);
            let err = (dot as f64 - ref_dot).abs() / scale;
            assert!(
                err < 1e-4,
                "{name} row {j}: dot {dot} vs dequantised {ref_dot} (relative {err:.3e})"
            );
        }
    }
}
