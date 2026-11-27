//! Compile-time offsets: the arena layout is fixed before the first kernel is recorded, so a decode
//! step never allocates and never looks anything up at run time (design-plan P1).
//!
//! Three slabs, one `hipMalloc` each (`crate::Substrate`):
//!
//! * **weights** - the GGUF data section uploaded verbatim, so a tensor's device address is
//!   `base + (gguf_offset - data_start)` and no per-tensor allocation or repacking exists;
//! * **activations** - the activation arena: hidden states, per-execution buffers, the mHC streams,
//!   the rope table. Every offset is a `Slot` decided here;
//! * **caches** - the KV and GDN recurrent state, one contiguous slab with per-execution strides.

use sn_core::error::{Error, Result};

/// A named region of a slab. `offset` is relative to the slab's device base.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Slot {
    pub name: String,
    pub offset: usize,
    pub len: usize,
    pub align: usize,
}

/// One slab's layout. Reserving is sequential and pure, so the same sequence of reservations always
/// produces the same offsets - which is what lets a captured graph stay valid across replays.
#[derive(Debug, Default)]
pub struct Arena {
    name: String,
    slots: Vec<Slot>,
    cursor: usize,
    frozen: bool,
}

/// Alignment every tensor-ish slot is rounded up to: 256 B keeps every row of a 4-byte activation
/// vector 128-aligned and matches the GPU's preferred allocation granularity.
pub const DEFAULT_ALIGN: usize = 256;

impl Arena {
    pub fn new(name: &str) -> Arena {
        Arena { name: name.to_string(), slots: Vec::new(), cursor: 0, frozen: false }
    }

    /// Reserve `len` bytes and return the offset. The name must be unique: `resolve` is the only
    /// lookup a kernel builder is allowed to do, and it happens at compile time.
    pub fn reserve(&mut self, name: &str, len: usize, align: usize) -> Result<usize> {
        if self.frozen {
            return Err(Error::msg(format!(
                "arena {}: reserving {name:?} after the layout was frozen",
                self.name
            )));
        }
        if len == 0 {
            return Err(Error::msg(format!("arena {name}: zero-length slot {name:?}")));
        }
        if !align.is_power_of_two() || align == 0 || align > 4096 {
            return Err(Error::msg(format!(
                "arena {}: slot {name:?} asks for alignment {align}, which is not a power of two in 1..=4096",
                self.name
            )));
        }
        if self.slots.iter().any(|s| s.name == name) {
            return Err(Error::msg(format!("arena {}: duplicate slot name {name:?}", self.name)));
        }
        let off = self.cursor.div_ceil(align) * align;
        self.slots.push(Slot { name: name.to_string(), offset: off, len, align });
        self.cursor = off + len;
        Ok(off)
    }

    /// Convenience: reserve with the arena's default alignment.
    pub fn reserve_aligned(&mut self, name: &str, len: usize) -> Result<usize> {
        self.reserve(name, len, DEFAULT_ALIGN)
    }

    /// The offset of a reserved slot. Compile-time only - see `freeze`.
    pub fn resolve(&self, name: &str) -> Result<usize> {
        self.slot(name).map(|s| s.offset)
    }

    pub fn slot(&self, name: &str) -> Result<&Slot> {
        self.slots
            .iter()
            .find(|s| s.name == name)
            .ok_or_else(|| Error::msg(format!("arena {}: no slot named {name:?}", self.name)))
    }

    /// Close the layout: after this, `reserve` is an error and `total` cannot move, so the offsets
    /// baked into a program are final.
    pub fn freeze(&mut self) -> usize {
        self.frozen = true;
        self.total()
    }

    pub fn frozen(&self) -> bool {
        self.frozen
    }

    /// Total bytes to allocate for this slab.
    pub fn total(&self) -> usize {
        self.cursor
    }

    pub fn slots(&self) -> &[Slot] {
        &self.slots
    }

    /// `name offset len align` per line, for `sn hip info` and `SN_TRACE=ops`.
    pub fn report(&self) -> String {
        let mut s = format!("arena {} ({} bytes, {} slots)\n", self.name, self.total(), self.slots.len());
        for sl in &self.slots {
            s += &format!("  {:>12}  {:>12}  {:>4}\n", sl.offset, sl.len, sl.name);
        }
        s
    }
}

/// The three slabs together, with the offsets resolved.
#[derive(Debug, Default)]
pub struct Layout {
    /// Byte length of the weight slab (the GGUF data section, uploaded verbatim).
    pub weights_len: usize,
    /// Where the GGUF file's tensor data section starts; a tensor at file offset `o` lives at
    /// `weights_base + (o - data_start)`.
    pub data_start: u64,
    pub activations: Arena,
    pub caches: Arena,
}

impl Layout {
    pub fn new(data_start: u64, weights_len: usize) -> Layout {
        Layout {
            weights_len,
            data_start,
            activations: Arena::new("activations"),
            caches: Arena::new("caches"),
        }
    }

    /// Freeze both arenas and report the sizes to allocate.
    pub fn freeze(&mut self) -> (usize, usize) {
        (self.activations.freeze(), self.caches.freeze())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn offsets_are_deterministic_and_aligned() {
        let mut a = Arena::new("t");
        let a0 = a.reserve_aligned("a", 100).unwrap();
        let b0 = a.reserve("b", 8, 4096).unwrap();
        let c0 = a.reserve_aligned("c", 7).unwrap();
        assert_eq!(a0, 0);
        assert_eq!(b0, 4096);
        assert_eq!(c0, 4096 + 256);
        assert_eq!(a.total(), 4096 + 256 + 7);

        let mut a2 = Arena::new("t");
        a2.reserve_aligned("a", 100).unwrap();
        a2.reserve("b", 8, 4096).unwrap();
        a2.reserve_aligned("c", 7).unwrap();
        assert_eq!(a.slots(), a2.slots());
    }

    #[test]
    fn slots_never_overlap_and_are_named_once() {
        let mut a = Arena::new("t");
        for i in 0..40 {
            a.reserve(&format!("s{i}"), 3 + i * 7, 1 << (i % 5)).unwrap();
        }
        let mut s = a.slots().to_vec();
        s.sort_by_key(|x| x.offset);
        for w in s.windows(2) {
            assert!(w[0].offset + w[0].len <= w[1].offset, "{:?} overlaps {:?}", w[0], w[1]);
        }
        assert!(a.reserve("s3", 4, 1).is_err(), "duplicate names must be rejected");
    }

    #[test]
    fn a_frozen_arena_rejects_new_slots() {
        let mut a = Arena::new("t");
        a.reserve_aligned("x", 16).unwrap();
        a.freeze();
        assert!(a.frozen());
        assert!(a.reserve_aligned("y", 16).is_err());
        assert_eq!(a.resolve("x").unwrap(), 0);
        assert!(a.resolve("nope").is_err());
    }
}
