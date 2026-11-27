//! The Phase 4 kernel surface: which symbols exist, what workgroup they are built for, and one
//! type-checked handle for each.
//!
//! The registry in [`crate::kernels`] is deliberately dumb (a `(name, shape, dtype)` map, because
//! Phase 5 will need many shapes of the same kernel). This module is the typed view of the
//! elementwise/routing set, so a test or the decode compiler writes `k.rmsnorm_f32` instead of
//! repeating a string and a workgroup size.

use sn_core::error::{Error, Result};

use crate::hip::Device;
use crate::kernels::{DType, Key, Kernel, KernelRegistry};

/// The workgroup size every kernel in `kernels/phase4.hip` is written for. The kernels use it as a
/// compile-time constant for their shared-memory layouts, so a different value here is a bug, not a
/// tuning knob - `register_phase4` pins it and `Kernel::block` is checked against it.
pub const BLOCK: u32 = 256;

/// `sn_rope_inplace_*` is built for `n_pairs` threads (32 for this checkpoint), not `BLOCK`.
pub const ROPE_BLOCK_MAX: u32 = 64;

/// Workgroup size every kernel in `kernels/phase5.hip` is built for. Same reasoning as `BLOCK`:
/// the GEMV kernels size their shared-memory reduce scratch by the *compiled* workgroup width, so
/// `SN_GEMV_BLOCK` in the .hip and this constant must agree, and the loader refuses an image whose
/// recorded value does not.
pub const GEMV_BLOCK: u32 = 256;

/// Threads in one `sn_mhc_route_f32` workgroup: matches `SN_ROUTE_THREADS` in the .hip. A 256-wide
/// workgroup read the 256 KB router at ~8 GB/s; 1024 measured 2.6x faster (Docs/bench.md).
pub const ROUTE_BLOCK: u32 = 1024;

/// One handle per Phase 4 kernel, resolved from the loaded images.
#[derive(Clone, Copy, Debug)]
pub struct Kernels {
    pub rmsnorm_f32: Kernel,
    pub rmsnorm_f16: Kernel,
    pub add_f32: Kernel,
    pub add_f16: Kernel,
    pub silu_mul_f32: Kernel,
    pub silu_mul_f16: Kernel,
    pub swiglu_f32: Kernel,
    pub swiglu_f16: Kernel,
    pub sigmoid_f32: Kernel,
    pub rope_inplace_f32: Kernel,
    pub rope_inplace_f16: Kernel,
    pub attn_gate_norm_f32: Kernel,
    pub attn_gate_norm_f16: Kernel,
    pub mhc_route_f32: Kernel,
    pub mhc_seed_f32: Kernel,
    pub mhc_merge_f32: Kernel,
    pub mhc_expose_f32: Kernel,
    // The unfused baseline of the Phase 4 micro-benchmark (kernels/phase4.hip, "NOT engine code").
    pub ub_sub: Kernel,
    pub ub_neg: Kernel,
    pub ub_softplus: Kernel,
    pub ub_gemv8: Kernel,
    pub ub_bias_add: Kernel,
    pub ub_sigmoid_post: Kernel,
    pub ub_mix_ab: Kernel,
    pub ub_exp4: Kernel,
}

/// The (name, dtype) pairs this module knows about, in registration order. Shape-specialised keys
/// are not used by Phase 4: none of these kernels are specialised per shape, they take their
/// geometry as arguments (which is the point - the decode program bakes the arguments, not the ISA).
pub const SPECS: &[(&str, DType, u32)] = &[
    ("sn_rmsnorm_f32", DType::F32, BLOCK),
    ("sn_rmsnorm_f16", DType::F16, BLOCK),
    ("sn_add_f32", DType::F32, BLOCK),
    ("sn_add_f16", DType::F16, BLOCK),
    ("sn_silu_mul_f32", DType::F32, BLOCK),
    ("sn_silu_mul_f16", DType::F16, BLOCK),
    ("sn_swiglu_f32", DType::F32, BLOCK),
    ("sn_swiglu_f16", DType::F16, BLOCK),
    ("sn_sigmoid_f32", DType::F32, BLOCK),
    ("sn_rope_inplace_f32", DType::F32, ROPE_BLOCK_MAX),
    ("sn_rope_inplace_f16", DType::F16, ROPE_BLOCK_MAX),
    ("sn_attn_gate_norm_f32", DType::F32, BLOCK),
    ("sn_attn_gate_norm_f16", DType::F16, BLOCK),
    ("sn_mhc_route_f32", DType::F32, ROUTE_BLOCK),
    ("sn_mhc_seed_f32", DType::F32, BLOCK),
    ("sn_mhc_merge_f32", DType::F32, BLOCK),
    ("sn_mhc_expose_f32", DType::F32, BLOCK),
    ("sn_ub_sub", DType::F32, 1),
    ("sn_ub_neg", DType::F32, 1),
    ("sn_ub_softplus", DType::F32, 1),
    ("sn_ub_gemv8", DType::Any, BLOCK),
    ("sn_ub_bias_add", DType::Any, 8),
    ("sn_ub_sigmoid_post", DType::Any, 1),
    ("sn_ub_mix_ab", DType::F32, BLOCK),
    ("sn_ub_exp4", DType::Any, 4),
];

/// Register every kernel in [`SPECS`] under its own name. Missing symbols are a hard error naming
/// the image, because a decode program that silently loses a kernel would produce silently wrong
/// numbers instead of failing to build.
pub fn register_phase4(dev: &Device, reg: &mut KernelRegistry) -> Result<usize> {
    let mut n = 0;
    for (name, dtype, block) in SPECS {
        reg.register(
            dev,
            &Key { name: (*name).to_string(), shape: vec![], dtype: *dtype },
            name,
            [*block, 1, 1],
        )?;
        n += 1;
    }
    Ok(n)
}

impl Kernels {
    /// Resolve the typed handles. Call after [`register_phase4`].
    pub fn resolve(reg: &KernelRegistry) -> Result<Kernels> {
        let g = |name: &str, dtype: DType| -> Result<Kernel> {
            let k = *reg
                .find(name, &[], dtype)
                .or_else(|| reg.find(name, &[], DType::Any))
                .ok_or_else(|| Error::msg(format!("{name} is not registered (run register_phase4)")))?;
            if k.block[0] == 0 {
                return Err(Error::msg(format!("{name} has a zero workgroup dimension")));
            }
            Ok(k)
        };
        Ok(Kernels {
            rmsnorm_f32: g("sn_rmsnorm_f32", DType::F32)?,
            rmsnorm_f16: g("sn_rmsnorm_f16", DType::F16)?,
            add_f32: g("sn_add_f32", DType::F32)?,
            add_f16: g("sn_add_f16", DType::F16)?,
            silu_mul_f32: g("sn_silu_mul_f32", DType::F32)?,
            silu_mul_f16: g("sn_silu_mul_f16", DType::F16)?,
            swiglu_f32: g("sn_swiglu_f32", DType::F32)?,
            swiglu_f16: g("sn_swiglu_f16", DType::F16)?,
            sigmoid_f32: g("sn_sigmoid_f32", DType::F32)?,
            rope_inplace_f32: g("sn_rope_inplace_f32", DType::F32)?,
            rope_inplace_f16: g("sn_rope_inplace_f16", DType::F16)?,
            attn_gate_norm_f32: g("sn_attn_gate_norm_f32", DType::F32)?,
            attn_gate_norm_f16: g("sn_attn_gate_norm_f16", DType::F16)?,
            mhc_route_f32: g("sn_mhc_route_f32", DType::F32)?,
            mhc_seed_f32: g("sn_mhc_seed_f32", DType::F32)?,
            mhc_merge_f32: g("sn_mhc_merge_f32", DType::F32)?,
            mhc_expose_f32: g("sn_mhc_expose_f32", DType::F32)?,
            ub_sub: g("sn_ub_sub", DType::F32)?,
            ub_neg: g("sn_ub_neg", DType::F32)?,
            ub_softplus: g("sn_ub_softplus", DType::F32)?,
            ub_gemv8: g("sn_ub_gemv8", DType::Any)?,
            ub_bias_add: g("sn_ub_bias_add", DType::Any)?,
            ub_sigmoid_post: g("sn_ub_sigmoid_post", DType::Any)?,
            ub_mix_ab: g("sn_ub_mix_ab", DType::F32)?,
            ub_exp4: g("sn_ub_exp4", DType::Any)?,
        })
    }
}

/// The grid that covers `n` elements with `BLOCK` threads per workgroup (the elementwise kernels
/// grid-stride, so this is an upper bound rather than an exact cover).
pub fn grid1d(n: usize) -> [u32; 3] {
    [((n as u64).div_ceil(BLOCK as u64)).clamp(1, 65535) as u32, 1, 1]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_registered_kernel_is_also_reachable_through_the_typed_view() {
        // SPECS and Kernels must not drift: a kernel registered but unresolvable is dead, one
        // resolved but unregistered is a runtime error nobody sees until a decode program builds.
        let names: Vec<&str> = SPECS.iter().map(|s| s.0).collect();
        assert_eq!(names.len(), names.iter().collect::<std::collections::BTreeSet<_>>().len(), "duplicate spec");
        for expected in [
            "sn_rmsnorm_f32",
            "sn_rmsnorm_f16",
            "sn_add_f32",
            "sn_add_f16",
            "sn_silu_mul_f32",
            "sn_silu_mul_f16",
            "sn_swiglu_f32",
            "sn_swiglu_f16",
            "sn_sigmoid_f32",
            "sn_rope_inplace_f32",
            "sn_rope_inplace_f16",
            "sn_attn_gate_norm_f32",
            "sn_attn_gate_norm_f16",
            "sn_mhc_route_f32",
            "sn_mhc_seed_f32",
            "sn_mhc_merge_f32",
            "sn_mhc_expose_f32",
        ] {
            assert!(names.contains(&expected), "{expected} is missing from SPECS");
        }
        // Workgroup sizes must be legal for hipModuleLaunchKernel and match the .hip's SN_BLOCK.
        for (name, _, block) in SPECS {
            assert!(*block > 0 && *block <= 1024, "{name} workgroup {block}");
        }
    }

    #[test]
    fn grid1d_covers_without_being_zero_or_absurd() {
        assert_eq!(grid1d(0), [1, 1, 1]);
        assert_eq!(grid1d(1), [1, 1, 1]);
        assert_eq!(grid1d(BLOCK as usize), [1, 1, 1]);
        assert_eq!(grid1d(BLOCK as usize + 1), [2, 1, 1]);
        assert_eq!(grid1d(12288), [48, 1, 1]);
        assert_eq!(grid1d(1 << 40), [65535, 1, 1], "grid-stride keeps a huge n launchable");
    }
}
