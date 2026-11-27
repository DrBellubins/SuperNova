//! sn-core — the SenseNova-Loop inference core.
//!
//! Dependency-free by design (Docs/design-plan.md §6): `std` only, kernel access through the
//! hand-written declarations in [`sys`], and (from Phase 3 on) HIP access through hand-written
//! bindings in the sibling `sn-hip` crate.
//!
//! Layering:
//! * [`gguf`]   — GGUF v2/v3 reader (Phase 1)
//! * [`plan`]   — hyperparameters + the Loop/mHC execution plan (Phase 1)
//! * [`budget`] — byte accounting per token and the "will it fit" check (Phase 1)
//! * `numerics`, `forward` — the f32 reference forward (Phase 2)
//! * [`oracle`] — scalar oracles for the fused HIP kernels, one per kernel (Phase 4)

pub mod budget;
pub mod dq;
pub mod error;
pub mod forward;
pub mod gguf;
pub mod num;
pub mod oracle;
pub mod plan;
pub mod qtypes;
pub mod sys;

pub use error::{Error, Result};
