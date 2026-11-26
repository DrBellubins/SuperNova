//! Phase 1 acceptance gates from `Docs/design-plan.md` §7:
//!
//! * `physical()` is exact on the boundary cases,
//! * GGUF blocks `52..55` are physical `28..31`,
//! * an mHC range that straddles the loop boundary is rejected.

use sn_core::plan::{physical, Dims, Exec, Hyperparams, Plan};

/// The checkpoint under development, expressed as a fixture so the index arithmetic can be tested
/// without a 5.8 GB file. Values come from SenseNova-Loop-9B-Q4_K_M.gguf (design plan §1).
fn fixture() -> Hyperparams {
    let n_layer_all = 57;
    // attention.recurrent_layers in the checkpoint is 1,1,1,0,... except for the MTP block, which
    // is always full attention (the reference rejects an MTP block that is recurrent).
    let is_recr = (0..n_layer_all)
        .map(|i| (i + 1) % 4 != 0 && i != 56)
        .collect();
    Hyperparams {
        arch: "sensenova_loop".into(),
        name: "fixture".into(),
        quant_label: "Q4_K_M".into(),
        n_layer_all,
        n_layer_nextn: 1,
        n_embd: 4096,
        n_ffn: 12288,
        n_head: 16,
        n_head_kv: 4,
        head_dim: 256,
        n_rot: 64,
        rms_eps: 1e-6,
        vocab: 248320,
        ctx_train: 524288,
        rope_base: 1e7,
        rope_type: "yarn".into(),
        yarn_factor: 2.0,
        ctx_orig: 262144,
        rope_sections: vec![11, 11, 10, 0],
        d_conv: 4,
        d_state: 128,
        n_group: 16,
        dt_rank: 32,
        d_inner: 4096,
        loop_start: 4,
        loop_end: 28,
        loop_count: 2,
        mhc_start: 4,
        mhc_end: 28,
        mhc_streams: 2,
        mhc_iters: 20,
        mhc_pre_scale: 1.0,
        mhc_fixed_identity: false,
        is_recr,
        dims: Dims {
            n_q: 4096,
            n_kv: 1024,
            attn_n_q_heads: 16,
            attn_n_kv_heads: 4,
            attn_d_head: 256,
            n_k_heads: 16,
            n_v_heads: 32,
            d_k: 128,
            d_v: 128,
            qkv_width: 8192,
            hc_width: 8192,
            n_route: 8,
        },
    }
}

fn plan() -> Plan {
    Plan::build(fixture()).expect("fixture must validate")
}

#[test]
fn physical_index_is_exact_on_every_boundary() {
    // The gate table from the design plan, via the free function.
    let f = |il| physical(il, 56, 4, 28, 24);
    assert_eq!(f(0), 0);
    assert_eq!(f(3), 3);
    assert_eq!(f(4), 4);
    assert_eq!(f(27), 27);
    assert_eq!(f(28), 4, "the second loop pass restarts at loop_start");
    assert_eq!(f(51), 27);
    assert_eq!(f(52), 28);
    assert_eq!(f(55), 31);
    assert_eq!(f(56), 32, "the MTP block sits past the trunk");

    // and via the built plan, which is what the forward actually uses.
    let p = plan();
    for il in 0..57 {
        assert_eq!(p.exec(il).physical, f(il), "logical {il}");
    }
}

#[test]
fn gguf_block_52_to_55_hold_physical_28_to_31() {
    let p = plan();
    for (il, ph) in [(52usize, 28usize), (53, 29), (54, 30), (55, 31)] {
        let e = p.exec(il);
        assert_eq!(e.physical, ph);
        assert_eq!(
            e.tensor_block, il,
            "physical {ph} is stored under blk.{il}, not blk.{ph}"
        );
    }
    // The looped visits read the in-loop blocks a second time.
    for il in 28..52 {
        let e = p.exec(il);
        assert_eq!(e.tensor_block, e.physical);
        assert!((4..28).contains(&e.tensor_block));
    }
    assert_eq!(p.exec_count_of_gguf_block(4), 2);
    assert_eq!(p.exec_count_of_gguf_block(27), 2);
    assert_eq!(p.exec_count_of_gguf_block(0), 1);
    assert_eq!(p.exec_count_of_gguf_block(52), 1);
    assert_eq!(p.exec_count_of_gguf_block(56), 1, "the MTP block executes once");
}

#[test]
fn plan_shape_matches_the_checkpoint() {
    let p = plan();
    assert_eq!(p.n_layer, 56);
    assert_eq!(p.extra, 24);
    assert_eq!(p.n_physical, 33, "32 trunk blocks + 1 MTP block");
    assert_eq!(p.width, 24);
    assert_eq!(p.execs.len(), 57);
    assert_eq!(p.n_gdn_executions(), 42);
    assert_eq!(p.n_attn_executions(), 15, "14 trunk full-attention visits + MTP");
    assert_eq!(p.n_mhc_executions(), 48, "24 mHC blocks x 2 passes - the 46,080-node source");
    assert_eq!(p.execs.iter().filter(|e| e.mtp != 0).count(), 1);
    assert!(!p.exec(56).mhc);
    assert!(!p.exec(56).recurrent);
    for e in &p.execs {
        assert_eq!(e.logical, p.exec(e.logical).logical);
        assert!(e.mtp <= 1);
    }
}

#[test]
fn mhc_range_straddling_the_loop_is_rejected() {
    let mut h = fixture();
    h.mhc_start = 2;
    h.mhc_end = 6; // starts before loop_start, ends inside the loop body
    let err = Plan::build(h).unwrap_err().to_string();
    assert_eq!(err, "SenseNova mHC range crosses a Loop boundary");

    let mut h = fixture();
    h.mhc_start = 20;
    h.mhc_end = 30; // starts inside the loop body, ends past loop_end
    assert_eq!(
        Plan::build(h).unwrap_err().to_string(),
        "SenseNova mHC range crosses a Loop boundary"
    );

    // Ranges that stay on one side are fine, including a pre-loop mHC region.
    let mut h = fixture();
    h.mhc_start = 0;
    h.mhc_end = 4;
    let p = Plan::build(h).expect("pre-loop mHC is legal");
    assert_eq!(p.n_mhc_executions(), 4);
    // Post-loop mHC is expressed in *physical* indices (the ranges are checked against
    // n_physical, and blocks 28..31 are the post-loop physical blocks).
    let mut h = fixture();
    h.mhc_start = 28;
    h.mhc_end = 32;
    let p = Plan::build(h).expect("post-loop mHC is legal");
    assert_eq!(p.n_mhc_executions(), 4);
}

#[test]
fn other_reference_validations_are_mirrored() {
    let mut h = fixture();
    h.loop_count = 3;
    assert_eq!(
        Plan::build(h).unwrap_err().to_string(),
        "SenseNova Loop execution plan exceeds the block count"
    );

    let mut h = fixture();
    h.loop_start = 10;
    h.loop_end = 4;
    assert_eq!(
        Plan::build(h).unwrap_err().to_string(),
        "invalid SenseNova Loop layer range or count"
    );

    let mut h = fixture();
    h.mhc_iters = 21;
    assert_eq!(
        Plan::build(h).unwrap_err().to_string(),
        "unsupported SenseNova mHC configuration"
    );
    let mut h = fixture();
    h.mhc_pre_scale = 0.0;
    assert_eq!(
        Plan::build(h).unwrap_err().to_string(),
        "unsupported SenseNova mHC configuration"
    );
    let mut h = fixture();
    h.mhc_streams = 3;
    assert!(Plan::build(h).is_err());

    // Logical layer 56 (the MTP block) is full attention; making it recurrent must be caught.
    let mut h = fixture();
    h.is_recr[56] = true;
    assert_eq!(
        Plan::build(h).unwrap_err().to_string(),
        "SenseNova Loop MTP requires a full-attention layer"
    );

    // A loop body whose visits disagree about attention type is not a loop body.
    let mut h = fixture();
    h.is_recr[5] = false;
    assert_eq!(
        Plan::build(h).unwrap_err().to_string(),
        "SenseNova Loop visits have different attention types"
    );

    // fixed_identity changes the router width, not the graph shape.
    let mut h = fixture();
    h.mhc_fixed_identity = true;
    let p = Plan::build(h).expect("fixed_identity is legal");
    assert_eq!(p.hp.dims.n_route, 4);
}

#[test]
fn attention_mask_and_derived_widths_come_from_the_checkpoint() {
    let p = plan();
    // 1,1,1,0 pattern over the trunk: the 4th layer of every group is full attention.
    for il in 0..56 {
        assert_eq!(p.is_recr(il), (il + 1) % 4 != 0, "logical {il}");
    }
    assert_eq!(p.hp.dims.n_v_heads, p.hp.d_inner / p.hp.d_state);
    assert_eq!(p.hp.dims.qkv_width, 2 * p.hp.d_inner);
    assert_eq!(p.hp.dims.hc_width, p.hp.mhc_streams * p.hp.n_embd);
    // Derived widths must be consistent with the declared ones.
    assert_eq!(p.hp.n_head * p.hp.head_dim, p.hp.n_embd);
}

/// The `Exec` layout is depended on by name in the plan dump; pin its meaning, not its size.
#[test]
fn exec_records_are_self_consistent() {
    let p = plan();
    // logical 30 is the second pass through physical block 6.
    let e: &Exec = p.exec(30);
    assert_eq!(
        *e,
        Exec { logical: 30, physical: 6, tensor_block: 6, mhc: true, recurrent: true, mtp: 0 }
    );
    assert_eq!(e.physical, 6);
    assert!(e.mhc);
    assert!(e.recurrent);
}
