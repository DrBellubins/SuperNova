//! GGML tensor element types: numeric ids, block geometry, exact byte sizes.
//!
//! The ids and block layouts are re-derived from the GGUF/GGML wire format (a public format), not
//! copied from any implementation. Only the types this project can actually meet are listed with a
//! size; anything else is reported as unsupported by name.

/// Numeric type ids as they appear in the GGUF tensor-info record.
pub mod ty {
    pub const F32: u32 = 0;
    pub const F16: u32 = 1;
    pub const Q4_0: u32 = 2;
    pub const Q4_1: u32 = 3;
    pub const Q5_0: u32 = 6;
    pub const Q5_1: u32 = 7;
    pub const Q8_0: u32 = 8;
    pub const Q8_1: u32 = 9;
    pub const Q2_K: u32 = 10;
    pub const Q3_K: u32 = 11;
    pub const Q4_K: u32 = 12;
    pub const Q5_K: u32 = 13;
    pub const Q6_K: u32 = 14;
    pub const Q8_K: u32 = 15;
    pub const I8: u32 = 24;
    pub const I16: u32 = 25;
    pub const I32: u32 = 26;
    pub const I64: u32 = 27;
    pub const F64: u32 = 28;
    pub const BF16: u32 = 30;
}

/// `bs` = elements per block, `tl` = bytes per block. For unquantised types `bs == 1`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TypeProps {
    pub name: &'static str,
    pub bs: usize,
    pub tl: usize,
}

const T: &[(&str, u32, usize, usize)] = &[
    ("f32", ty::F32, 1, 4),
    ("f16", ty::F16, 1, 2),
    ("bf16", ty::BF16, 1, 2),
    ("f64", ty::F64, 1, 8),
    ("i8", ty::I8, 1, 1),
    ("i16", ty::I16, 1, 2),
    ("i32", ty::I32, 1, 4),
    ("i64", ty::I64, 1, 8),
    ("q4_0", ty::Q4_0, 32, 18),
    ("q4_1", ty::Q4_1, 32, 20),
    ("q5_0", ty::Q5_0, 32, 22),
    ("q5_1", ty::Q5_1, 32, 24),
    ("q8_0", ty::Q8_0, 32, 34),
    ("q2_k", ty::Q2_K, 256, 84),
    ("q3_k", ty::Q3_K, 256, 110),
    ("q4_k", ty::Q4_K, 256, 144),
    ("q5_k", ty::Q5_K, 256, 176),
    ("q6_k", ty::Q6_K, 256, 210),
    ("q8_k", ty::Q8_K, 256, 260),
];

pub fn props(t: u32) -> Option<TypeProps> {
    T.iter()
        .find(|(_, id, _, _)| *id == t)
        .map(|(name, _, bs, tl)| TypeProps { name, bs: *bs, tl: *tl })
}

pub fn name(t: u32) -> String {
    props(t).map(|p| p.name.to_string()).unwrap_or_else(|| format!("type#{t}"))
}

/// Number of elements described by a `ne` array (missing dims are 1).
pub fn n_elements(ne: &[u64; 4]) -> u64 {
    ne.iter().product()
}

/// The byte count the type implies for `n` elements: whole blocks only, so a partial block still
/// costs a full block. Used to cross-check the size derived from GGUF offsets.
pub fn bytes_for_n(n: u64, t: u32) -> Option<u64> {
    let p = props(t)?;
    let nblocks = n.div_ceil(p.bs as u64);
    Some(nblocks * p.tl as u64)
}

/// Can `dequant` turn this type into f32? (Phase 2 needs exactly what this file contains:
/// f32 / q4_k / q6_k; f16 comes for free because the KV cache and matmul inputs use it.)
pub fn supported_by_reference(t: u32) -> bool {
    matches!(t, ty::F32 | ty::F16 | ty::Q4_K | ty::Q6_K)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bits_per_weight_match_the_documented_k_quants() {
        // These bpw figures are the published ones; they catch a typo in `tl`.
        for (id, want) in [(ty::Q2_K, 2.625), (ty::Q3_K, 3.4375), (ty::Q4_K, 4.5), (ty::Q5_K, 5.5), (ty::Q6_K, 6.5625)] {
            let p = props(id).unwrap();
            let bpw = p.tl as f64 * 8.0 / p.bs as f64;
            assert_eq!(bpw, want, "{}", p.name);
        }
    }

    #[test]
    fn n_elements_and_bytes() {
        assert_eq!(n_elements(&[4096, 12288, 1, 1]), 4096 * 12288);
        // 4096*12288 elements of q4_k = 49_738_240 bytes (this is exactly one attn_qkv_gate).
        assert_eq!(
            bytes_for_n(4096 * 12288, ty::Q4_K),
            Some(4096 * 12288 / 256 * 144)
        );
        assert_eq!(bytes_for_n(8192, ty::F32), Some(32768));
        assert_eq!(bytes_for_n(8192 * 3, ty::F32), Some(98304)); // conv state
        assert_eq!(bytes_for_n(100, ty::Q4_K), Some(144)); // partial block rounds up
        assert_eq!(props(999), None);
    }
}
