//! Weights as the scalar reference reads them: rows of a stored matrix, dequantised to f32 on the
//! fly, or dot-producted against an f32 activation without materialising the row.
//!
//! The K-quant bit layouts are re-derived from the GGUF wire format (see the format notes on each
//! decoder). Nothing here is shared with the reference implementation's kernels — those may be read
//! only to confirm the format, and the layout constants here are the format, not an implementation.

use crate::gguf::{Gguf, TensorInfo};
use crate::qtypes::ty;
use crate::{Error, Result};

/// f16 → f32. Written out because `f16` is not in `core`.
#[inline]
/// Round an f32 to the nearest f16-representable value (round-to-nearest-even).
///
/// The reference's GEMM kernels feed f16 operands to its MMA units, so its projections carry an f16
/// operand error on every weight and activation. `sn ref --f16-weights / --f16-acts` reproduces that
/// to tell operand precision apart from a math error.
pub fn f16_round(x: f32) -> f32 {
    f16_to_f32(f32_to_f16(x))
}

/// Round an f32 to f16 bits (round-to-nearest-even). The kernel tests need the bits, not
/// the rounded f32, because they write real f16 buffers.
pub fn f32_to_f16(x: f32) -> u16 {
    let b = x.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    let exp = ((b >> 23) & 0xFF) as i32 - 127;
    let mant = b & 0x7F_FFFF;
    if exp == 128 {
        return sign | 0x7C00 | if mant != 0 { 0x200 } else { 0 };
    }
    if exp >= -14 {
        let m10 = mant >> 13;
        // the 13 discarded mantissa bits (`>>` binds tighter than `&`, hence the parens)
        let rest = mant & 0x1FFF;
        let round_up = rest > 0x1000 || (rest == 0x1000 && (m10 & 1) != 0);
        let mut e = exp + 15;
        let mut m = m10;
        if round_up {
            m += 1;
            if m == 0x400 {
                m = 0;
                e += 1;
            }
        }
        if e > 30 {
            return sign | 0x7C00; // overflow to infinity
        }
        return sign | (((e << 10) | m as i32) as u16);
    }
    if exp < -25 {
        return sign;
    }
    // subnormal half: value = m_half * 2^-24, so fold the hidden 1 in and shift
    // (m_half = (2^23 | mant) >> (-1 - exp), with exp in -25..=-15)
    let shift = (-1 - exp) as u32;
    let full = mant | (1 << 23);
    let m = full >> shift;
    let rest = full & ((1u32 << shift) - 1);
    let half = 1u32 << (shift - 1);
    let round_up = rest > half || (rest == half && (m & 1) != 0);
    sign | ((m + round_up as u32) as u16)
}

pub fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h >> 15) & 1) as u32;
    let exp = ((h >> 10) & 0x1F) as u32;
    let man = (h & 0x3FF) as u32;
    let bits = if exp == 0 {
        if man == 0 {
            sign << 31
        } else {
            // subnormal: normalise by moving the leading one into the exponent
            let mut e = 0u32;
            let mut m = man;
            while m & 0x400 == 0 {
                m <<= 1;
                e += 1;
            }
            // value = 2^-14 * man/1024 = 2^(-14-e) once `man << e` reaches the hidden bit
            let m = (m & 0x3FF) << 13;
            ((127 - 14 - e) << 23) | m | (sign << 31)
        }
    } else if exp == 0x1F {
        (0xFF << 23) | (man << 13) | (sign << 31)
    } else {
        ((exp + 127 - 15) << 23) | (man << 13) | (sign << 31)
    };
    f32::from_bits(bits)
}

/// Elements per K-quant super-block and bytes per super-block, for the types we can meet.
const QK_K: usize = 256;
const Q4K_BLOCK: usize = 144; // d:f16 dmin:f16 scales:12 qs:128
const Q6K_BLOCK: usize = 210; // ql:128 qh:64 sc:16 d:f16

// Field offsets inside a super-block (the format, not an implementation choice).
const Q4K_D: usize = 0;
const Q4K_DMIN: usize = 2;
const Q4K_SCALES: usize = 4;
const Q4K_QS: usize = 16;
const Q6K_QL: usize = 0;
const Q6K_QH: usize = 128;
const Q6K_SC: usize = 192;
const Q6K_D: usize = 208;

/// A 2-D weight tensor in file layout: `ne0` input features contiguous, `ne1` output rows.
///
/// Row `j` starts at `j * row_bytes()`; for the K-quants a row is a whole number of super-blocks
/// for every weight shape this model contains (`ne0` is always a multiple of 256).
pub struct Mat<'a> {
    pub name: &'a str,
    pub ne0: usize,
    pub ne1: usize,
    pub ty: u32,
    data: &'a [u8],
}

impl<'a> Mat<'a> {
    pub fn of(g: &'a Gguf, t: &'a TensorInfo) -> Result<Mat<'a>> {
        if t.n_dims != 2 {
            return Err(Error::msg(format!(
                "{}: expected a 2-D weight, got {} dimensions",
                t.name, t.n_dims
            )));
        }
        if !crate::qtypes::supported_by_reference(t.ty) {
            return Err(Error::msg(format!(
                "{}: element type {} is not supported by the reference reader",
                t.name,
                crate::qtypes::name(t.ty)
            )));
        }
        let ne0 = t.ne[0] as usize;
        let ne1 = t.ne[1] as usize;
        let m = Mat {
            name: &t.name,
            ne0,
            ne1,
            ty: t.ty,
            data: g.tensor_data(t),
        };
        if m.row_bytes() * ne1 > m.data.len() {
            return Err(Error::msg(format!(
                "{}: row stride {} B x {ne1} rows exceeds the {} B stored",
                t.name,
                m.row_bytes(),
                m.data.len()
            )));
        }
        Ok(m)
    }

    /// Convenience for the forward: resolve a tensor by name and wrap it.
    pub fn by_name(g: &'a Gguf, name: &str) -> Result<Mat<'a>> {
        let t = g.tensor(name)?;
        Mat::of(g, t)
    }

    /// Bytes occupied by one output row.
    #[inline]
    pub fn row_bytes(&self) -> usize {
        match self.ty {
            ty::F32 => self.ne0 * 4,
            ty::F16 => self.ne0 * 2,
            ty::Q4_K => self.ne0 / QK_K * Q4K_BLOCK,
            ty::Q6_K => self.ne0 / QK_K * Q6K_BLOCK,
            _ => 0,
        }
    }

    /// The row as `f32` without copying, when the element type is `f32` (norms, routers, conv taps).
    #[inline]
    pub fn row_f32(&self, j: usize) -> Option<&'a [f32]> {
        if self.ty != ty::F32 {
            return None;
        }
        Some(f32_row(self.row(j)))
    }

    #[inline]
    fn row(&self, j: usize) -> &'a [u8] {
        let s = j * self.row_bytes();
        &self.data[s..s + self.row_bytes()]
    }

    /// `dot(row j, x)`, f32 accumulation, no temporary row.
    pub fn dot_row(&self, j: usize, x: &[f32]) -> f32 {
        assert_eq!(x.len(), self.ne0, "{}: dot with {} values, expected {}", self.name, x.len(), self.ne0);
        let r = self.row(j);
        match self.ty {
            ty::F32 => dot_f32(f32_row(r), x),
            ty::F16 => {
                let mut acc = [0f32; 4];
                for (i, h) in r.chunks_exact(2).enumerate() {
                    let w = f16_to_f32(u16::from_le_bytes([h[0], h[1]]));
                    acc[i & 3] += w * x[i];
                }
                (acc[0] + acc[1]) + (acc[2] + acc[3])
            }
            ty::Q4_K => {
                // Each block covers 256 consecutive inputs: zip the byte blocks with the matching
                // slices of `x`, otherwise every block dots against the first 256 values.
                let mut acc = 0f32;
                for (blk, xs) in r.chunks_exact(Q4K_BLOCK).zip(x.chunks_exact(QK_K)) {
                    acc += q4k_dot(blk, xs);
                }
                acc
            }
            ty::Q6_K => {
                let mut acc = 0f32;
                for (blk, xs) in r.chunks_exact(Q6K_BLOCK).zip(x.chunks_exact(QK_K)) {
                    acc += q6k_dot(blk, xs);
                }
                acc
            }
            t => unreachable!("unsupported element type {t}"),
        }
    }

    /// `out[0..ne0] = row j` dequantised to f32.
    pub fn dequant_row(&self, j: usize, out: &mut [f32]) {
        assert_eq!(out.len(), self.ne0);
        let r = self.row(j);
        match self.ty {
            ty::F32 => out.copy_from_slice(f32_row(r)),
            ty::F16 => {
                for (o, h) in out.iter_mut().zip(r.chunks_exact(2)) {
                    *o = f16_to_f32(u16::from_le_bytes([h[0], h[1]]));
                }
            }
            ty::Q4_K => {
                for (blk, o) in r.chunks_exact(Q4K_BLOCK).zip(out.chunks_exact_mut(QK_K)) {
                    q4k_dequant(blk, o);
                }
            }
            ty::Q6_K => {
                for (blk, o) in r.chunks_exact(Q6K_BLOCK).zip(out.chunks_exact_mut(QK_K)) {
                    q6k_dequant(blk, o);
                }
            }
            t => unreachable!("unsupported element type {t}"),
        }
    }

    /// `y = W x` for a single column. `y` must have `ne1` entries.
    /// Dot a row after rounding every multiplicand to f16. `x` is expected to have been rounded by
    /// the caller (`f16_round` is idempotent). Materialises the row, so this is a diagnostic path.
    pub fn dot_row_round(&self, j: usize, x: &[f32], row: &mut Vec<f32>) -> f32 {
        row.clear();
        row.resize(self.ne0, 0.0);
        self.dequant_row(j, row);
        let mut acc = [0f32; 4];
        for (i, &w) in row.iter().enumerate() {
            acc[i & 3] += f16_round(w) * x[i];
        }
        (acc[0] + acc[1]) + (acc[2] + acc[3])
    }

    pub fn gemv(&self, x: &[f32], y: &mut [f32]) {
        assert_eq!(y.len(), self.ne1, "{}: output length {}", self.name, y.len());
        for (j, out) in y.iter_mut().enumerate() {
            *out = self.dot_row(j, x);
        }
    }

    /// `y += W x`, used for split tensors that are applied as one logical matrix.
    pub fn gemv_add(&self, x: &[f32], y: &mut [f32], row_offset: usize) {
        for j in 0..self.ne1 {
            y[row_offset + j] += self.dot_row(j, x);
        }
    }

    /// `y = W x` with the output rows split across `threads` worker threads.
    ///
    /// The reference forward streams the whole checkpoint per token; a single-threaded scalar loop
    /// costs several seconds per token, which makes the Phase 2 gate take hours. The parallel form
    /// computes the same f32 sums (each worker owns whole output rows, so there is no shared
    /// accumulator and no reassociation of partial sums).
    pub fn gemv_par(&self, x: &[f32], y: &mut [f32], threads: usize) {
        assert_eq!(y.len(), self.ne1);
        let threads = threads.clamp(1, self.ne1.max(1));
        if threads == 1 {
            self.gemv(x, y);
            return;
        }
        let chunk = self.ne1.div_ceil(threads);
        std::thread::scope(|s| {
            // `chunks_mut` hands every worker a disjoint output range, so no two threads touch the
            // same accumulator and the sums are bit-identical to the serial path.
            for (t, ys) in y.chunks_mut(chunk).enumerate() {
                let lo = t * chunk;
                let m = &*self;
                let xs = &*x;
                s.spawn(move || {
                    for (k, out) in ys.iter_mut().enumerate() {
                        *out = m.dot_row(lo + k, xs);
                    }
                });
            }
        });
    }
}

#[inline]
fn f32_row(b: &[u8]) -> &[f32] {
    // SAFETY: tensor data is aligned to `general.alignment` (>= 32, checked by `Gguf::open`) and f32
    // rows are contiguous 4-byte units; length is a multiple of 4.
    unsafe { std::slice::from_raw_parts(b.as_ptr() as *const f32, b.len() / 4) }
}

/// Four accumulators so the adds are independent; the compiler vectorises the body either way.
fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
    let mut acc = [0f32; 8];
    for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
        acc[i & 7] += *x * *y;
    }
    ((acc[0] + acc[1]) + (acc[2] + acc[3])) + ((acc[4] + acc[5]) + (acc[6] + acc[7]))
}

/// Q4_K super-block layout (256 elements, 144 B):
/// `d: f16` at 0, `dmin: f16` at 2, `scales: [u8; 12]` at 4, `qs: [u8; 128]` at 16.
///
/// 8 sub-blocks of 32 elements. Sub-block `2k` holds the **low** nibbles of `qs[32k .. 32k+32]`,
/// sub-block `2k+1` the **high** nibbles of the same bytes. Scale/min for sub-block `j`:
/// `j < 4`: `sc = scales[j] & 63`, `mn = scales[j+4] & 63`;
/// `j >= 4`: `sc = (scales[j+4] & 15) | (scales[j-4] >> 6) << 4`,
///           `mn = (scales[j+4] >> 4) | (scales[j] >> 6) << 4`.
/// Value = `d*sc*q - dmin*mn`.
#[inline]
fn q4k_scale_min(scales: &[u8], j: usize) -> (u32, u32) {
    if j < 4 {
        ((scales[j] & 63) as u32, (scales[j + 4] & 63) as u32)
    } else {
        (
            ((scales[j + 4] & 0xF) | ((scales[j - 4] >> 6) << 4)) as u32,
            ((scales[j + 4] >> 4) | ((scales[j] >> 6) << 4)) as u32,
        )
    }
}

fn q4k_dequant(blk: &[u8], out: &mut [f32]) {
    let d = f16_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
    let dm = f16_to_f32(u16::from_le_bytes([blk[2], blk[3]]));
    let scales = &blk[4..16];
    let qs = &blk[16..144];
    for j in 0..8 {
        let (sc, mn) = q4k_scale_min(scales, j);
        let ds = d * sc as f32;
        let m = dm * mn as f32;
        let base = 32 * j;
        // Sub-blocks 2k and 2k+1 share the 32 bytes at qs[32k .. 32k+32]: 2k is the low nibbles,
        // 2k+1 the high nibbles of the same bytes.
        let bytes = (j / 2) * 32;
        let half = (j & 1) * 4;
        for i in 0..32 {
            let q = (qs[bytes + i] >> half) & 0xF;
            out[base + i] = ds * q as f32 - m;
        }
    }
}

fn q4k_dot(blk: &[u8], x: &[f32]) -> f32 {
    let d = f16_to_f32(u16::from_le_bytes([blk[Q4K_D], blk[Q4K_D + 1]]));
    let dm = f16_to_f32(u16::from_le_bytes([blk[Q4K_DMIN], blk[Q4K_DMIN + 1]]));
    let scales = &blk[Q4K_SCALES..Q4K_SCALES + 12];
    let qs = &blk[Q4K_QS..Q4K_QS + 128];
    let mut acc = 0f32;
    for j in 0..8 {
        let (sc, mn) = q4k_scale_min(scales, j);
        let ds = d * sc as f32;
        let m = dm * mn as f32;
        let base = 32 * j;
        let bytes = (j / 2) * 32;
        let half = (j & 1) * 4;
        let mut part = 0f32;
        for i in 0..32 {
            let q = (qs[bytes + i] >> half) & 0xF;
            part += q as f32 * x[base + i];
        }
        // sum_i (ds*q_i - m)*x_i = ds*sum(q_i*x_i) - m*sum(x_i)
        acc += ds * part - m * x[base..base + 32].iter().sum::<f32>();
    }
    acc
}

/// Q6_K super-block layout (256 elements, 210 B):
/// `ql: [u8; 128]` at 0 (low 6 bits: two nibbles per byte), `qh: [u8; 64]` at 128 (the top 2 bits
/// of each value, four values per byte), `sc: [i8; 16]` at 192, `d: f16` at 208.
///
/// Exactly: for each 128-element half-block, with `l` in 0..32, `is = l / 16`:
/// element `l`      = low nibble of `ql[l]`      | (`qh[l]` bits 0-1  << 4)
/// element `l+32`  = low nibble of `ql[l+32]`   | (`qh[l]` bits 2-3  << 4)
/// element `l+64`  = high nibble of `ql[l]`     | (`qh[l]` bits 4-5  << 4)
/// element `l+96`  = high nibble of `ql[l+32]`  | (`qh[l]` bits 6-7  << 4)
/// The 6-bit value is signed-biased: `q = v - 32`, and `value = d * sc[is + 2k] * q` for the four
/// groups in the order above. The 16 `sc` bytes are **signed** (i8), unlike the Q4_K scales.
/// Half-block 2 follows half-block 1 (`ql += 64, qh += 32, sc += 8`); `d: f16` is at offset 208.
#[inline]
fn q6k_value(ql: &[u8], qh: &[u8], l: usize, group: usize) -> i32 {
    let (byte, nib_hi) = if group == 0 || group == 2 { (l, group == 2) } else { (l + 32, group == 3) };
    let nib = if nib_hi { ql[byte] >> 4 } else { ql[byte] & 0xF };
    let hi = (qh[l] >> (2 * group)) & 0x3;
    (((hi as usize) << 4) | nib as usize) as i32 - 32
}

fn q6k_dequant(blk: &[u8], out: &mut [f32]) {
    let d = f16_to_f32(u16::from_le_bytes([blk[Q6K_D], blk[Q6K_D + 1]]));
    for hb in 0..2 {
        let o = hb * 128;
        let ql = &blk[Q6K_QL + 64 * hb..Q6K_QL + 64 * hb + 64];
        let qh = &blk[Q6K_QH + 32 * hb..Q6K_QH + 32 * hb + 32];
        let sc = &blk[Q6K_SC + 8 * hb..Q6K_SC + 8 * hb + 8];
        for l in 0..32 {
            let is = l / 16;
            for g in 0..4 {
                let q = q6k_value(ql, qh, l, g);
                out[o + l + 32 * g] = d * sc[is + 2 * g] as i8 as f32 * q as f32;
            }
        }
    }
}

fn q6k_dot(blk: &[u8], x: &[f32]) -> f32 {
    let d = f16_to_f32(u16::from_le_bytes([blk[Q6K_D], blk[Q6K_D + 1]]));
    let mut acc = 0f32;
    for hb in 0..2 {
        let o = hb * 128;
        let ql = &blk[Q6K_QL + 64 * hb..Q6K_QL + 64 * hb + 64];
        let qh = &blk[Q6K_QH + 32 * hb..Q6K_QH + 32 * hb + 32];
        let sc = &blk[Q6K_SC + 8 * hb..Q6K_SC + 8 * hb + 8];
        for l in 0..32 {
            let is = l / 16;
            for g in 0..4 {
                let q = q6k_value(ql, qh, l, g);
                acc += d * sc[is + 2 * g] as i8 as f32 * q as f32 * x[o + l + 32 * g];
            }
        }
    }
    acc
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f16_decodes_specials_and_normals() {
        assert_eq!(f16_to_f32(0), 0.0);
        assert_eq!(f16_to_f32(0x8000), -0.0);
        assert_eq!(f16_to_f32(0x3C00), 1.0);
        assert_eq!(f16_to_f32(0xBC00), -1.0);
        assert_eq!(f16_to_f32(0x3800), 0.5);
        assert_eq!(f16_to_f32(0x4000), 2.0);
        assert_eq!(f16_to_f32(0x2B01), 0.054_718_018); // a normal with a fraction: 2^-5 * (1 + 769/1024)
        assert_eq!(f16_to_f32(0x0001), 5.9604645e-8); // smallest subnormal
        assert!(f16_to_f32(0x7C00).is_infinite());
        assert!(f16_to_f32(0x7FFF).is_nan());
    }

    /// Hand-built Q4_K block. The encoder below re-states the format (it is written from the layout
    /// notes on `q4k_dequant`, not from that function), so a mismatch means one of the two has the
    /// nibble order, the sub-block order, or the 6-bit scale/min packing wrong.
    fn encode_q4k(d: f32, dmin: f32, sc: [u32; 8], mn: [u32; 8], nib: [u8; 256]) -> Vec<u8> {
        let mut b = vec![0u8; Q4K_BLOCK];
        let h = |v: f32| -> [u8; 2] {
            // f32 -> f16, exact for the small test values used here
            let bits = v.to_bits();
            let sign = ((bits >> 31) & 1) as u16;
            let exp = ((bits >> 23) & 0xFF) as i32 - 127 + 15;
            let man = (bits & 0x7FFFFF) >> 13;
            ((sign << 15) | ((exp as u16) << 10) | man as u16).to_le_bytes()
        };
        b[0..2].copy_from_slice(&h(d));
        b[2..4].copy_from_slice(&h(dmin));
        // scales[0..4] hold sc[0..3] in their low 6 bits and the top 2 bits of sc[4..7] in bits 6-7;
        // scales[4..8] likewise hold mn[0..3] and the top 2 bits of mn[4..7]; scales[8..12] carry
        // the low nibbles of sc[4..7] and mn[4..7], one sub-block per byte.
        for j in 0..4 {
            b[Q4K_SCALES + j] = (sc[j] & 63) as u8 | (((sc[4 + j] >> 4) as u8 & 3) << 6);
            b[Q4K_SCALES + 4 + j] = (mn[j] & 63) as u8 | (((mn[4 + j] >> 4) as u8 & 3) << 6);
        }
        for k in 0..4 {
            b[Q4K_SCALES + 8 + k] = ((sc[4 + k] & 0xF) as u8) | (((mn[4 + k] & 0xF) as u8) << 4);
        }
        // Element order -> the format's nibble placement: sub-block s = i/32 lives in the bytes at
        // qs[32*(s/2) ..], even s in the low nibbles, odd s in the high nibbles of the same bytes.
        // (Index arithmetic is the point of this loop, so it stays a range loop.)
        #[allow(clippy::needless_range_loop)]
        for i in 0..256 {
            let sb = i / 32;
            let byte = 32 * (sb / 2) + (i % 32);
            b[Q4K_QS + byte] |= (nib[i] & 0xF) << (4 * (sb & 1));
        }
        b
    }

    #[test]
    fn q4k_block_decodes_to_the_format_equation() {
        // sub-block s occupies elements 32s..32s+32; odd sub-blocks are the high nibbles.
        let sc = [1u32, 2, 3, 4, 33, 48, 0, 7];
        let mn = [0u32, 1, 2, 3, 5, 61, 0, 2];
        let mut nib = [0u8; 256];
        for s in 0..8 {
            for i in 0..32 {
                nib[32 * s + i] = (s * 2 + (i % 3)) as u8 % 16;
            }
        }
        let blk = encode_q4k(0.5, 0.25, sc, mn, nib);

        let mut out = [0f32; QK_K];
        q4k_dequant(&blk, &mut out);
        for s in 0..8 {
            for i in 0..32 {
                let q = nib[32 * s + i] as f32;
                let want = 0.5 * sc[s] as f32 * q - 0.25 * mn[s] as f32;
                assert_eq!(
                    out[32 * s + i],
                    want,
                    "sub-block {s} element {i}: got {}, want {want}",
                    out[32 * s + i]
                );
            }
        }
        // The dot path must agree with the materialised one.
        let x: Vec<f32> = (0..QK_K).map(|i| (i as f32 - 128.0) * 0.01).collect();
        let direct = q4k_dot(&blk, &x);
        let want: f32 = out.iter().zip(&x).map(|(a, b)| a * b).sum();
        assert!((direct - want).abs() < 1e-3 * want.abs().max(1.0), "{direct} vs {want}");
    }

    /// Hand-built Q6_K block, same idea: `value = d * sc[is + 2g] * (v - 32)`.
    /// Packs 256 six-bit codes (0..63, before the -32 bias) plus signed super-block scales into the
    /// 210-byte Q6_K layout: two half-blocks of 128 values, `ql` low nibbles, `qh` two high bits per
    /// value, `sc` as i8, `d` as f16 last. Written from the layout notes, so it fails if either the
    /// encoder or `q6k_dequant` has the nibble/high-bit packing wrong.
    fn encode_q6k(d_half: u16, sc: [i8; 16], code: &[u16; QK_K]) -> Vec<u8> {
        let mut blk = vec![0u8; Q6K_BLOCK];
        blk[Q6K_D..Q6K_D + 2].copy_from_slice(&d_half.to_le_bytes());
        for (i, v) in sc.iter().enumerate() {
            blk[Q6K_SC + i] = *v as u8;
        }
        for hb in 0..2 {
            let ql = Q6K_QL + 64 * hb;
            let qh = Q6K_QH + 32 * hb;
            for l in 0..32 {
                for g in 0..4 {
                    let c = code[hb * 128 + l + 32 * g];
                    blk[qh + l] |= ((c >> 4) as u8 & 0x3) << (2 * g);
                    let nib = (c & 0xF) as u8;
                    match g {
                        0 => blk[ql + l] |= nib,
                        1 => blk[ql + l + 32] |= nib,
                        2 => blk[ql + l] |= nib << 4,
                        _ => blk[ql + l + 32] |= nib << 4,
                    }
                }
            }
        }
        blk
    }

    #[test]
    fn q6k_block_decodes_to_the_format_equation() {
        const D_HALF: u16 = 0x3400; // f16 0.25
        let d = 0.25f32;
        let sc: [i8; 16] = [
            1, -1, 2, -2, 3, -3, 4, -4, 5, 5, -6, 7, -8, 9, 10, -11,
        ];
        let mut code = [0u16; QK_K];
        for hb in 0..2 {
            for l in 0..32 {
                for g in 0..4 {
                    code[hb * 128 + l + 32 * g] = ((l * 7 + g * 13 + hb * 5) % 64) as u16;
                }
            }
        }
        let enc = encode_q6k(D_HALF, sc, &code);
        let mut blk = vec![0u8; Q6K_BLOCK];
        blk.copy_from_slice(&enc);
        let mut out = [0f32; QK_K];
        q6k_dequant(&blk, &mut out);
        for hb in 0..2 {
            for l in 0..32 {
                let is = l / 16;
                for g in 0..4 {
                    let idx = hb * 128 + l + 32 * g;
                    let want = d * sc[hb * 8 + is + 2 * g] as f32 * (code[idx] as f32 - 32.0);
                    assert_eq!(out[idx], want, "index {idx}");
                }
            }
        }
        let x: Vec<f32> = (0..QK_K).map(|i| ((i % 17) as f32 - 8.0) * 0.125).collect();
        let direct = q6k_dot(&blk, &x);
        let want: f32 = out.iter().zip(&x).map(|(a, b)| a * b).sum();
        assert!(
            (direct - want).abs() < 1e-3 * want.abs().max(1.0),
            "{direct} vs {want}"
        );
    }

    /// Every 256 elements of a row is a separate block, so the dot path has to advance through the
    /// activation block by block. Both K-quant dot kernels take a *block-sized* slice of `x`; passing
    /// the whole row instead silently dots every block against the first 256 inputs, which every
    /// projection in the forward would then get wrong while single-block tests still passed.
    #[test]
    fn quant_rows_dot_block_by_block_across_the_activation() {
        // Two blocks with deliberately different content, so a stale x slice cannot cancel out.
        let mut q4 = Vec::new();
        q4.extend(encode_q4k(
            0.25,
            0.0625,
            [1, 2, 3, 4, 5, 6, 7, 8],
            [0, 1, 0, 1, 0, 1, 0, 1],
            std::array::from_fn(|i| (i % 16) as u8),
        ));
        q4.extend(encode_q4k(
            0.5,
            0.0,
            [8, 7, 6, 5, 4, 3, 2, 1],
            [0; 8],
            std::array::from_fn(|i| (255 - i % 16) as u8),
        ));
        let mut q6 = Vec::new();
        let sc: [i8; 16] = [1, -1, 2, -2, 3, -3, 4, -4, 5, 5, -6, 7, -8, 9, 10, -11];
        let c0 = std::array::from_fn(|i| (((i as i32) * 7).rem_euclid(64)) as u16);
        let c1 = std::array::from_fn(|i| (63 - (i as i32) * 11).rem_euclid(64) as u16);
        q6.extend(encode_q6k(0x3400, sc, &c0));
        q6.extend(encode_q6k(0x3800, sc.map(|v| -v), &c1));

        for (ty, bytes) in [(ty::Q4_K, q4), (ty::Q6_K, q6)] {
            let m = Mat { name: "two-block-row", ne0: QK_K * 2, ne1: 1, ty, data: &bytes };
            let x: Vec<f32> = (0..QK_K * 2).map(|i| ((i % 23) as f32 - 11.0) * 0.125).collect();
            let mut want_row = vec![0f32; QK_K * 2];
            m.dequant_row(0, &mut want_row);
            let want: f32 = want_row.iter().zip(&x).map(|(a, b)| a * b).sum();
            let got = m.dot_row(0, &x);
            assert!(
                (got - want).abs() < 1e-3 * want.abs().max(1.0),
                "{ty} dot {got} vs dequantised {want}"
            );
            let mut y = [0f32; 1];
            m.gemv(&x, &mut y);
            assert_eq!(y[0], got);
            let mut y2 = [0f32; 1];
            m.gemv_par(&x, &mut y2, 2);
            assert_eq!(y2[0], got, "the parallel path must be bit-identical");
        }
    }

    #[test]
    fn row_strides_are_the_block_geometry() {
        let m = Mat {
            name: "t",
            ne0: 4096,
            ne1: 12288,
            ty: ty::Q4_K,
            data: &[],
        };
        assert_eq!(m.row_bytes(), 4096 * 144 / 256);
        let m6 = Mat { name: "t", ne0: 12288, ne1: 4096, ty: ty::Q6_K, data: &[] };
        assert_eq!(m6.row_bytes(), 12288 * 210 / 256);
        // 4.5 and 6.5625 bits per weight, as documented for the K-quants.
        assert_eq!((4096 * 144 / 256) * 8, 4096 * 9 / 2);
        assert_eq!((12288 * 210 / 256) * 16, 12288 * 105 / 8);
    }

    #[test]
    fn gemv_matches_the_dequantised_row_path() {
        // A small f32 matrix, wrapped without a file: build it from a byte slice.
        let w: Vec<f32> = (0..24).map(|i| i as f32 * 0.1 - 1.0).collect();
        let bytes: Vec<u8> = w.iter().flat_map(|v| v.to_le_bytes()).collect();
        let m = Mat { name: "t", ne0: 6, ne1: 4, ty: ty::F32, data: &bytes };
        let x: Vec<f32> = (0..6).map(|i| 1.0 + i as f32 * 0.5).collect();
        let mut y = [0f32; 4];
        m.gemv(&x, &mut y);
        let mut y2 = [0f32; 4];
        m.gemv_par(&x, &mut y2, 3);
        for j in 0..4 {
            let mut row = [0f32; 6];
            m.dequant_row(j, &mut row);
            let want: f32 = row.iter().zip(&x).map(|(a, b)| a * b).sum();
            assert!((y[j] - want).abs() < 1e-6);
            assert_eq!(y2[j], y[j]);
        }
    }
    #[test]
    fn f16_round_matches_the_reference_operand_precision() {
        // every finite, normal half must survive the round trip
        for h in 0u32..=0xFFFFu32 {
            let h = h as u16;
            if h & 0x7C00 == 0x7C00 {
                continue; // inf / nan
            }
            let x = f16_to_f32(h);
            assert_eq!(f32_to_f16(x), h, "half {h:#06x} = {x} did not round trip");
        }
        assert_eq!(f16_round(1.0 + 2f32.powi(-12)), 1.0); // below half an ulp
        assert!((f16_round(1.0 + 3.0 * 2f32.powi(-12)) - 1.0009_766).abs() < 1e-6);
        assert!(!f16_round(f32::NAN).is_finite());
        assert!(f16_round(70000.0).is_infinite());
        assert_eq!(f16_round(-0.0).to_bits(), (-0.0f32).to_bits());
        let tiny = f16_to_f32(0x0001);
        assert!((tiny - 5.9604645e-8).abs() < 1e-14);
        assert_eq!(f32_to_f16(tiny), 0x0001);
        assert_eq!(f16_round(1e-9), 0.0);
        // idempotence, and never further than half a half-ulp
        let mut s = 0x1234_5678u32;
        for _ in 0..5000 {
            s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let v = ((s >> 8) as f32) / 4096.0;
            let r = f16_round(v);
            assert!((r - v).abs() <= v.abs() * 5e-4 + 6e-8, "{v} -> {r}");
            assert_eq!(f16_round(r).to_bits(), r.to_bits(), "{v} not idempotent");
        }
    }
}
