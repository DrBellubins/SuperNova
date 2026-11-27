//! The Phase 4 fusion gate on its own: one fused mHC routing kernel against the fork's 480-node
//! balancing, timed. It is a separate test binary from `phase4.rs` because cargo runs test binaries
//! one after another but their tests in parallel, and the ratio is only stable on an idle GPU.
//!
//! Gate (Docs/design-plan.md section 7): the fused route must be >= 100x the unfused sequence,
//! measured as one captured graph each, which is how the engine runs (Phase 3). Docs/bench.md
//! reports the node-by-node number too. Skips with a printed reason when there is no device or no
//! kernel image - see `tests/common/mod.rs`.

mod common;
use common::with_harness;

use sn_hip::check;

/// Gate: one fused routing kernel beats the unfused node sequence by >= 100x.
#[test]
fn phase4_fused_route_beats_the_unfused_sequence() {
    let Some(gate) = with_harness("fusion gate", |h| check::fusion_gate(h, 20, 60, 5).expect("fusion measured")) else {
        return;
    };
    eprint!("{}", gate.report());
    for (what, ok, detail, gating) in gate.passes() {
        if gating {
            assert!(ok, "{what}: {detail}");
        }
    }
}


