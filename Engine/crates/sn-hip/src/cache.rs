//! Phase 6 caches: the KV cache and the GDN recurrent state, as one frozen arena.
//!
//! design-plan §7 Phase 6: *"`KVCache`: paged-by-block, per-layer stride, f16 / q8_0 / q4_0
//! selectable, append-only, positions explicit. `GdnState`: `[42 logical layers][n_seq][32×128×128]`
//! f32 + conv `[8192×3]`, one contiguous slab, compile-time offsets so decode needs zero
//! indirection; snapshot slots per sequence at chunk boundaries (the only prefix-cache primitive
//! that is correct for a hybrid model — a KV prefix without its matching GDN snapshot must be
//! rejected, assert it)."*
//!
//! Two decisions worth stating, because they are the ones a reader will ask about:
//!
//! * **"Paged by block" is an *allocation* property here, not a runtime indirection.** Each
//!   sequence owns a contiguous region per cache-bearing execution, sized in whole blocks
//!   ([`BLOCK_TOKENS`]). A vLLM-style block table (logical block → physical block, looked up per
//!   token) exists to fight fragmentation in a shared pool; with fixed per-sequence regions there
//!   is nothing to fragment, and the decode kernel would pay a dependent load per token for it.
//!   Every address a kernel needs is therefore `base + row * row_bytes` — compile-time `base`,
//!   one multiply per row, and `Docs/design-plan.md` §7's "decode needs zero indirection" holds.
//! * **Caches are keyed by *execution*, not by physical block.** Two visits of one physical block
//!   (`il = 4` and `il = 28` both run block 4) hold different recurrent state, and the Phase 2
//!   reference does the same (`forward::RefModel` indexes `kv`/`state`/`conv` by `ex.logical`).
//!   The execution list comes from `sn_core::plan::Plan`, per §8 ("never re-derive the loop/mHC/
//!   plan logic").

use sn_core::error::{Error, Result};
use sn_core::plan::Plan;

use crate::arena::Arena;

/// The allocation granularity of a cache region. Regions start on a block boundary and are a whole
/// number of blocks long, which is what "paged by block" means for a fixed-region layout.
pub const BLOCK_TOKENS: usize = 256;

/// One KV row: all `n_kv_heads` for one token, one cache kind. Sizes come from the GGUF, never
/// from a assumed head geometry.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KvKind {
    F16,
    Q8_0,
    Q4_0,
}

impl KvKind {
    pub fn parse(s: &str) -> Result<KvKind> {
        match s.to_ascii_lowercase().as_str() {
            "f16" | "fp16" | "half" => Ok(KvKind::F16),
            "q8_0" | "q8" => Ok(KvKind::Q8_0),
            "q4_0" | "q4" => Ok(KvKind::Q4_0),
            other => Err(Error::msg(format!(
                "unknown KV type {other:?} (this phase implements f16, q8_0, q4_0)"
            ))),
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            KvKind::F16 => "f16",
            KvKind::Q8_0 => "q8_0",
            KvKind::Q4_0 => "q4_0",
        }
    }

    /// Quantisation blocks per row. f16 has none; q8_0/q4_0 block 32 values with a scale.
    pub fn blocks_per_row(&self, values: usize) -> usize {
        match self {
            KvKind::F16 => 0,
            _ => values.div_ceil(32),
        }
    }

    /// Bytes of the quant array in one row (K or V).
    pub fn qs_bytes(&self, values: usize) -> usize {
        match self {
            KvKind::F16 => values * 2,
            KvKind::Q8_0 => self.blocks_per_row(values) * 32,
            KvKind::Q4_0 => self.blocks_per_row(values) * 16,
        }
    }

    /// Bytes of the scale array, which follows the quant array in the row.
    pub fn scale_bytes(&self, values: usize) -> usize {
        self.blocks_per_row(values) * 4
    }

    /// Bytes for one row (K or V) of `values` values: the quant array, then the scale array.
    ///
    /// This is deliberately **not** GGML's interleaved `block_q8_0` (`qs[32]` then a half, 34 B).
    /// A 34-byte block makes every subsequent block unaligned, which is the Phase 5 Q6_K result in
    /// another place (`Docs/bench.md` §3.3: 210-byte super-blocks, no field vector-aligned, 36 % of
    /// the bandwidth reference). Splitting the row into one contiguous quant run plus a contiguous
    /// f32 scale run costs 4 B per 32 values and keeps every load aligned.
    pub fn row_bytes(&self, values: usize) -> usize {
        self.qs_bytes(values) + self.scale_bytes(values)
    }

    /// Bytes for K *and* V for one token.
    pub fn token_bytes(&self, values: usize) -> usize {
        2 * self.row_bytes(values)
    }
}

/// What to size the cache arena for.
#[derive(Clone, Debug)]
pub struct CacheGeom {
    pub n_ctx: usize,
    pub n_seq: usize,
    /// Snapshot slots per sequence (Phase 9's rollback / MTP mechanism).
    pub n_snap: usize,
    pub kind: KvKind,
    /// Values per KV row (K and V): `n_kv_heads * d_head`.
    pub kv_values: usize,
    /// GDN state per execution: `n_v_heads * d_v * d_k` f32.
    pub state_values: usize,
    /// GDN conv window per execution: `(d_conv - 1) * conv_channels` f32.
    pub conv_values: usize,
}

impl CacheGeom {
    /// Read the geometry out of the plan (which read it out of the GGUF).
    pub fn from_plan(plan: &Plan, n_ctx: usize, n_seq: usize, n_snap: usize, kind: KvKind) -> CacheGeom {
        let dm = &plan.hp.dims;
        CacheGeom {
            n_ctx: n_ctx.max(1),
            n_seq: n_seq.max(1),
            n_snap,
            kind,
            // `Dims::n_kv` is already the *width* `head_count_kv * key_length` (see its own doc
            // comment); multiplying it by `attn_d_head` again is the trap that made a 512-token
            // region 7 GiB. `attn_n_kv_heads * attn_d_head` is the same number, written out.
            kv_values: dm.attn_n_kv_heads * dm.attn_d_head,
            state_values: dm.n_v_heads * dm.d_v * dm.d_k,
            conv_values: (plan.hp.d_conv - 1) * dm.qkv_width,
        }
    }

    /// A sequence region is a whole number of blocks, so `n_ctx` rounds up.
    pub fn blocks(&self) -> usize {
        self.n_ctx.div_ceil(BLOCK_TOKENS)
    }

    pub fn kv_region_bytes(&self) -> usize {
        self.blocks() * BLOCK_TOKENS * self.kind.token_bytes(self.kv_values)
    }

    pub fn state_bytes(&self) -> usize {
        self.state_values * 4
    }

    pub fn conv_bytes(&self) -> usize {
        self.conv_values * 4
    }
}

/// Reserved offsets. Every one of these is baked into the program at compile time.
#[derive(Clone, Debug, Default)]
pub struct CacheLayout {
    pub geom: Option<CacheGeom>,
    /// Execution indices that carry a KV cache / GDN state, in plan order.
    pub kv_exec: Vec<usize>,
    pub gdn_exec: Vec<usize>,
    /// `kv_offset[exec_index][seq]`, same shape for the GDN arrays.
    kv: Vec<Vec<usize>>,
    state: Vec<Vec<usize>>,
    conv: Vec<Vec<usize>>,
    snap_state: Vec<Vec<usize>>,
    snap_conv: Vec<Vec<usize>>,
    /// `pos[seq][row]`: the position each cached row was written at. Append-only, but the engine
    /// never has to *assume* contiguity - `plan` position and cache row are separately recorded.
    pub pos_table: usize,
    pub pos_table_stride: usize,
    /// `meta[seq][field]`: `{ len, n_snap, last_pos, generation }` as i32 pairs.
    pub meta: usize,
    pub meta_stride: usize,
    total: usize,
}

/// Fields of a sequence's meta row.
pub const META_LEN: usize = 0;
pub const META_N_SNAP: usize = 1;
pub const META_LAST_POS: usize = 2;
pub const META_GENERATION: usize = 3;
pub const META_WORDS: usize = 4;

impl CacheLayout {
    pub fn new() -> CacheLayout {
        CacheLayout { geom: None, ..Default::default() }
    }

    /// Reserve every cache region. Names are stable so `Arena::report` is diffable across runs.
    pub fn reserve(caches: &mut Arena, plan: &Plan, geom: &CacheGeom) -> Result<CacheLayout> {
        let mut l = CacheLayout { geom: Some(geom.clone()), ..Default::default() };
        for il in 0..plan.n_exec() {
            let ex = plan.exec(il);
            if ex.mtp != 0 {
                // The MTP block is a second program (Phase 9); its caches are reserved then.
                continue;
            }
            if ex.recurrent {
                l.gdn_exec.push(il);
            } else {
                l.kv_exec.push(il);
            }
        }
        if l.kv_exec.is_empty() {
            return Err(Error::msg("the plan has no full-attention executions: no KV cache to build"));
        }

        // Regions are block-aligned; a 256-token block of f16 KV is 512 KiB, so 256 B alignment is
        // already implied by the region size. Ask for 4096 anyway: kernels use vector loads.
        for (n, ex) in l.kv_exec.iter().enumerate() {
            let mut per_seq = Vec::new();
            for s in 0..geom.n_seq {
                let off = caches.reserve(&format!("kv.e{ex}.s{s}"), geom.kv_region_bytes(), 4096)?;
                per_seq.push(off);
            }
            l.kv.push(per_seq);
            let _ = n;
        }
        for (n, ex) in l.gdn_exec.iter().enumerate() {
            let mut st = Vec::new();
            let mut cv = Vec::new();
            for s in 0..geom.n_seq {
                st.push(caches.reserve(&format!("gdn_state.e{ex}.s{s}"), geom.state_bytes(), 256)?);
                cv.push(caches.reserve(&format!("gdn_conv.e{ex}.s{s}"), geom.conv_bytes(), 256)?);
            }
            l.state.push(st);
            l.conv.push(cv);
            let _ = n;
        }
        for s in 0..geom.n_seq {
            let mut st = Vec::new();
            let mut cv = Vec::new();
            for k in 0..geom.n_snap {
                st.push(caches.reserve(&format!("snap_state.s{s}.{k}"), geom.state_bytes() * l.gdn_exec.len().max(1), 256)?);
                cv.push(caches.reserve(&format!("snap_conv.s{s}.{k}"), geom.conv_bytes() * l.gdn_exec.len().max(1), 256)?);
            }
            l.snap_state.push(st);
            l.snap_conv.push(cv);
        }
        l.pos_table_stride = geom.blocks() * BLOCK_TOKENS;
        l.meta_stride = META_WORDS;
        for s in 0..geom.n_seq {
            caches.reserve(&format!("kv_pos.s{s}"), l.pos_table_stride * 4, 256)?;
        }
        l.pos_table = caches.resolve("kv_pos.s0")?;
        // One contiguous meta block so a single H2D copy can refresh it.
        l.meta = caches.reserve("seq_meta", geom.n_seq * META_WORDS * 4, 256)?;
        l.total = caches.total();
        Ok(l)
    }

    pub fn total(&self) -> usize {
        self.total
    }

    fn geom(&self) -> Result<&CacheGeom> {
        self.geom
            .as_ref()
            .ok_or_else(|| Error::msg("this CacheLayout was never reserved"))
    }

    /// The index of an execution in the reserved lists. Asking for a execution of the wrong kind is
    /// a caller bug (an attention kernel never runs against a GDN execution), and saying so beats
    /// reading whatever the first cache region happens to contain.
    fn exec_index(list: &[usize], exec: usize, what: &str) -> Result<usize> {
        list.iter()
            .position(|&e| e == exec)
            .ok_or_else(|| Error::msg(format!("execution {exec} has no {what} cache (it is not in this plan's {} set)", list.len())))
    }

    pub fn kv(&self, exec: usize, seq: usize) -> Result<usize> {
        let i = Self::exec_index(&self.kv_exec, exec, "KV")?;
        self.row(&self.kv[i], seq, "kv")
    }

    pub fn state(&self, exec: usize, seq: usize) -> Result<usize> {
        let i = Self::exec_index(&self.gdn_exec, exec, "GDN")?;
        self.row(&self.state[i], seq, "gdn state")
    }

    pub fn conv(&self, exec: usize, seq: usize) -> Result<usize> {
        let i = Self::exec_index(&self.gdn_exec, exec, "GDN")?;
        self.row(&self.conv[i], seq, "gdn conv")
    }

    pub fn snapshot_state(&self, seq: usize, slot: usize) -> Result<usize> {
        self.row(self.snap_state.get(seq).map(|v| &v[..]).unwrap_or(&[]), slot, "snapshot state")
    }

    pub fn snapshot_conv(&self, seq: usize, slot: usize) -> Result<usize> {
        self.row(self.snap_conv.get(seq).map(|v| &v[..]).unwrap_or(&[]), slot, "snapshot conv")
    }

    pub fn pos_row(&self, seq: usize) -> Result<usize> {
        if seq >= self.geom()?.n_seq {
            return Err(Error::msg(format!("sequence {seq} is outside a {} sequence layout", self.geom()?.n_seq)));
        }
        Ok(self.pos_table + seq * self.pos_table_stride * 4)
    }

    pub fn meta_row(&self, seq: usize) -> Result<usize> {
        if seq >= self.geom()?.n_seq {
            return Err(Error::msg(format!("sequence {seq} is outside a {} sequence layout", self.geom()?.n_seq)));
        }
        Ok(self.meta + seq * self.meta_stride * 4)
    }

    /// The byte offset of one cached row (K or V half) inside the arena.
    pub fn kv_row(&self, exec: usize, seq: usize, row: usize) -> Result<usize> {
        let g = self.geom()?;
        if row >= g.n_ctx {
            return Err(Error::msg(format!(
                "cached row {row} is past this sequence's {} token region",
                g.n_ctx
            )));
        }
        Ok(self.kv(exec, seq)? + row * g.kind.token_bytes(g.kv_values))
    }

    /// The V half follows the K half at `row_bytes`.
    pub fn kv_v_offset(&self) -> Result<usize> {
        let g = self.geom()?;
        Ok(g.kind.row_bytes(g.kv_values))
    }

    /// Bytes per KV row, for the kernels.
    pub fn kv_row_bytes(&self) -> Result<usize> {
        let g = self.geom()?;
        Ok(g.kind.row_bytes(g.kv_values))
    }

    fn row(&self, list: &[usize], i: usize, what: &str) -> Result<usize> {
        list.get(i).copied().ok_or_else(|| {
            Error::msg(format!("{what}: index {i} is outside a {} entry region", list.len()))
        })
    }

    /// `sn hip caches`: the sizes the §3 table predicts, in the same units.
    pub fn report(&self) -> String {
        let g = match self.geom() {
            Ok(g) => g,
            Err(_) => return "cache layout: not reserved\n".to_string(),
        };
        let mut s = format!(
            "cache layout: {} KV executions, {} GDN executions, {} sequences, {} token regions, KV type {}\n",
            self.kv_exec.len(),
            self.gdn_exec.len(),
            g.n_seq,
            g.n_ctx,
            g.kind.name()
        );
        let mb = |b: usize| b as f64 / (1024.0 * 1024.0);
        s += &format!(
            "  KV per token: {} B/execution  {} B/token total\n",
            g.kind.token_bytes(g.kv_values),
            g.kind.token_bytes(g.kv_values) * self.kv_exec.len()
        );
        s += &format!(
            "  KV per sequence region: {:.1} MiB  ({} executions x {} tokens)\n",
            mb(g.kv_region_bytes() * self.kv_exec.len()),
            self.kv_exec.len(),
            g.n_ctx
        );
        s += &format!(
            "  GDN state per sequence: {:.1} MiB ({} x {:.1} MiB) + conv {:.1} MiB\n",
            mb(g.state_bytes() * self.gdn_exec.len()),
            self.gdn_exec.len(),
            mb(g.state_bytes()),
            mb(g.conv_bytes() * self.gdn_exec.len()),
        );
        s += &format!(
            "  snapshots: {} slots/sequence x {:.1} MiB = {:.1} MiB/sequence\n",
            g.n_snap,
            mb((g.state_bytes() + g.conv_bytes()) * self.gdn_exec.len()),
            mb((g.state_bytes() + g.conv_bytes()) * self.gdn_exec.len() * g.n_snap),
        );
        s += &format!("  total cache arena: {}\n", crate::human(self.total));
        s
    }
}

/// A snapshot slot: a GDN state copy taken at a known cache length.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Snapshot {
    pub slot: usize,
    /// The KV length this snapshot pairs with. **The whole point of the field**: a KV prefix is
    /// only usable together with the GDN state that was current at that same length.
    pub len: usize,
    pub last_pos: u32,
}

/// Host-side bookkeeping for the cached state of each sequence: lengths, positions, snapshots.
/// The device holds the bytes; this holds the truth about what they mean, and every rule about
/// combining them is here so the engine cannot invent its own.
#[derive(Clone, Debug)]
pub struct Sessions {
    n_seq: usize,
    len: Vec<usize>,
    last_pos: Vec<i64>,
    snaps: Vec<Vec<Snapshot>>,
    /// Which snapshot slot the sequence is currently rolled back to, if any.
    active: Vec<Option<usize>>,
    n_snap: usize,
}

impl Sessions {
    pub fn new(n_seq: usize, n_snap: usize) -> Sessions {
        Sessions {
            n_seq,
            len: vec![0; n_seq],
            last_pos: vec![-1; n_seq],
            snaps: vec![Vec::new(); n_seq],
            active: vec![None; n_seq],
            n_snap,
        }
    }

    pub fn n_seq(&self) -> usize {
        self.n_seq
    }
    pub fn len(&self, seq: usize) -> usize {
        self.len[seq]
    }
    pub fn is_empty(&self) -> bool {
        self.len.iter().all(|&n| n == 0)
    }
    pub fn last_pos(&self, seq: usize) -> i64 {
        self.last_pos[seq]
    }
    pub fn snapshots(&self, seq: usize) -> &[Snapshot] {
        &self.snaps[seq]
    }
    pub fn active_snapshot(&self, seq: usize) -> Option<usize> {
        self.active[seq]
    }

    /// Forget one sequence: length back to 0, positions back to -1, every snapshot slot released.
    /// The other sequences are untouched, which is the point - `Engine::reset(seq)` must not reset
    /// the whole batch. The bytes are the caller's business (`Slab::clear_range` over that
    /// sequence's GDN regions); KV rows are append-only and unread until written, so they need
    /// nothing.
    pub fn reset(&mut self, seq: usize) -> Result<()> {
        if seq >= self.n_seq {
            return Err(Error::msg(format!("reset: sequence {seq} of {}", self.n_seq)));
        }
        self.len[seq] = 0;
        self.last_pos[seq] = -1;
        self.snaps[seq].clear();
        self.active[seq] = None;
        Ok(())
    }

    /// Append one token. `pos` is explicit: the engine may skip positions (mrope, gaps after a
    /// rollback), and the cache must not assume that row `n` belongs to position `n`.
    pub fn append(&mut self, seq: usize, pos: u32) -> Result<usize> {
        let g = self.n_seq;
        if seq >= g {
            return Err(Error::msg(format!("sequence {seq} is outside {g} sequences")));
        }
        if pos as i64 <= self.last_pos[seq] {
            return Err(Error::msg(format!(
                "sequence {seq}: appending position {pos} after {} breaks append-only order",
                self.last_pos[seq]
            )));
        }
        self.len[seq] += 1;
        self.last_pos[seq] = pos as i64;
        Ok(self.len[seq] - 1)
    }

    /// Take a snapshot at the current length, in the next free slot (round-robin).
    pub fn snapshot(&mut self, seq: usize) -> Result<Snapshot> {
        if self.n_snap == 0 {
            return Err(Error::msg("snapshots are disabled (n_snap = 0)"));
        }
        let s = &mut self.snaps[seq];
        s.retain(|x| x.slot != self.active[seq].unwrap_or(usize::MAX));
        let slot = if s.len() >= self.n_snap {
            // Recycle the oldest. Phase 9's acceptance loop never needs more than the drafts it
            // has in flight; a longer history is a policy question for Phase 10.
            s.remove(0);
            0
        } else {
            s.iter().map(|x| x.slot + 1).max().unwrap_or(0)
        };
        if slot >= self.n_snap {
            return Err(Error::msg(format!("no free snapshot slot for sequence {seq}")));
        }
        let sn = Snapshot { slot, len: self.len[seq], last_pos: self.last_pos[seq] as u32 };
        s.push(sn);
        Ok(sn)
    }

    /// Roll a sequence back to a snapshot. The KV length and the GDN length are checked together -
    /// this is the assertion design-plan §7 Phase 6 asks for, in the one place that can fail.
    pub fn restore(&mut self, seq: usize, slot: usize) -> Result<Snapshot> {
        let sn = *self
            .snaps[seq]
            .iter()
            .find(|x| x.slot == slot)
            .ok_or_else(|| Error::msg(format!(
                "sequence {seq}: snapshot slot {slot} was never taken (live slots: {:?})",
                self.snaps[seq].iter().map(|x| x.slot).collect::<Vec<_>>()
            )))?;
        // Both halves move back together, or neither does.
        if sn.len > self.len[seq] {
            return Err(Error::msg(format!(
                "sequence {seq}: snapshot {slot} is at length {} but the KV cache is only {} - a snapshot cannot advance a cache",
                sn.len, self.len[seq]
            )));
        }
        self.len[seq] = sn.len;
        self.last_pos[seq] = sn.last_pos as i64;
        self.active[seq] = Some(slot);
        Ok(sn)
    }

    /// Adopt an existing (KV length, GDN length) pair, e.g. loaded from `/state` in Phase 10.
    /// A KV prefix with no matching GDN snapshot is rejected, not silently re-run: a hybrid model's
    /// attention prefix is meaningless without the linear-attention state at the same boundary.
    pub fn adopt(&mut self, seq: usize, kv_len: usize, gdn_len: usize, last_pos: i64) -> Result<Snapshot> {
        if kv_len != gdn_len {
            return Err(Error::msg(format!(
                "sequence {seq}: refusing a KV prefix of {kv_len} tokens with GDN state at {gdn_len} tokens - a hybrid model's prefix cache needs both halves at the same boundary"
            )));
        }
        let sn = *self
            .snaps[seq]
            .iter()
            .find(|x| x.len == gdn_len)
            .ok_or_else(|| Error::msg(format!(
                "sequence {seq}: no GDN snapshot at length {gdn_len}; snapshots are {:?}",
                self.snaps[seq].iter().map(|x| (x.slot, x.len)).collect::<Vec<_>>()
            )))?;
        self.len[seq] = kv_len;
        self.last_pos[seq] = last_pos.max(sn.last_pos as i64);
        self.active[seq] = Some(sn.slot);
        Ok(sn)
    }

    /// The rule itself, exposed so a test and a caller outside this crate can state it.
    pub fn check_pair(kv_len: usize, gdn_len: usize) -> Result<()> {
        if kv_len != gdn_len {
            return Err(Error::msg(format!(
                "KV prefix of {kv_len} tokens with GDN state at {gdn_len}: rejected (design-plan §7 Phase 6)"
            )));
        }
        Ok(())
    }

    /// The meta rows, as the device wants them: `i32` words, one row per sequence.
    pub fn meta_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(self.n_seq * META_WORDS * 4);
        for s in 0..self.n_seq {
            let words = [
                self.len[s] as i32,
                self.snaps[s].len() as i32,
                self.last_pos[s] as i32,
                self.active[s].map(|x| x as i32 + 1).unwrap_or(0),
            ];
            for w in words {
                out.extend_from_slice(&w.to_le_bytes());
            }
        }
        out
    }

    pub fn report(&self) -> String {
        let mut s = String::new();
        for seq in 0..self.n_seq {
            s += &format!(
                "  seq {seq}: len {} last_pos {} snapshots {:?} active {:?}\n",
                self.len[seq],
                self.last_pos[seq],
                self.snaps[seq].iter().map(|x| (x.slot, x.len)).collect::<Vec<_>>(),
                self.active[seq]
            );
        }
        s
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn plan() -> sn_core::plan::Plan {
        sn_core::plan::fixture_plan().unwrap()
    }

    fn geom(n_ctx: usize, n_seq: usize, n_snap: usize, kind: KvKind) -> CacheGeom {
        CacheGeom {
            n_ctx,
            n_seq,
            n_snap,
            kind,
            kv_values: 1024,
            state_values: 32 * 128 * 128,
            conv_values: 3 * 8192,
        }
    }

    #[test]
    fn kv_sizes_match_design_plan_section_3() {
        // §3: 14 KV executions x 32768 tokens x (4 heads x 256 values) x K+V.
        let g = geom(32768, 1, 0, KvKind::F16);
        let per_seq = |kind: KvKind| {
            let mut gg = g.clone();
            gg.kind = kind;
            gg.kv_region_bytes() as f64 * l14() as f64 / (1024.0 * 1024.0 * 1024.0)
        };
        // §3 quotes decimal GB ("f16 1.88 GB"): 1.88e9 B = 1.75 GiB.
        assert!((per_seq(KvKind::F16) - 1.75).abs() < 1e-9, "f16 KV at 32k");
        assert!(
            per_seq(KvKind::Q8_0) < 0.60 * per_seq(KvKind::F16),
            "q8_0 must be the §3-sized escape hatch: {:.2} GiB",
            per_seq(KvKind::Q8_0)
        );
        assert!(
            per_seq(KvKind::Q4_0) < 0.35 * per_seq(KvKind::F16),
            "q4_0: {:.2} GiB",
            per_seq(KvKind::Q4_0)
        );
        // GDN: 42 executions x 2.097 MB = 88.1 MB, conv 42 x 96 KB = 3.9 MB (§3).
        let g2 = geom(1, 1, 0, KvKind::F16);
        assert_eq!(g2.state_bytes(), 2_097_152);
        assert_eq!(g2.conv_bytes(), 98_304);
    }

    fn l14() -> usize {
        14
    }

    #[test]
    fn kv_rows_stay_vector_aligned() {
        for kind in [KvKind::F16, KvKind::Q8_0, KvKind::Q4_0] {
            assert_eq!(kind.row_bytes(1024) % 64, 0, "{:?} rows must stay vector-aligned", kind);
            assert_eq!(kind.qs_bytes(1024) % 64, 0);
            assert_eq!(kind.scale_bytes(1024) % 64, 0);
        }
        assert_eq!(KvKind::F16.row_bytes(1024), 2048);
        assert_eq!(KvKind::Q8_0.row_bytes(1024), 1024 + 128);
        assert_eq!(KvKind::Q4_0.row_bytes(1024), 512 + 128);
    }

    #[test]
    fn kv_kind_parser_rejects_types_the_kernels_do_not_have() {
        assert_eq!(KvKind::parse("q8_0").unwrap(), KvKind::Q8_0);
        assert_eq!(KvKind::parse("F16").unwrap(), KvKind::F16);
        assert!(KvKind::parse("q8_1").is_err());
        assert!(KvKind::parse("q5_k").is_err());
    }

    #[test]
    fn regions_are_block_granular_and_separated() {
        let g = geom(1000, 2, 1, KvKind::F16);
        assert_eq!(g.blocks(), 4, "1000 tokens rounds up to 4 x 256");
        let mut a = Arena::new("caches");
        let plan = plan();
        let l = CacheLayout::reserve(&mut a, &plan, &g).unwrap();
        let (a0, a1) = (l.kv(l.kv_exec[0], 0).unwrap(), l.kv(l.kv_exec[0], 1).unwrap());
        assert_eq!(a1 - a0, g.kv_region_bytes());
        assert_eq!(a0 % 4096, 0);
        assert_eq!(l.kv_row(l.kv_exec[0], 1, 3).unwrap(), a1 + 3 * g.kind.token_bytes(1024));
        assert!(l.kv_row(l.kv_exec[0], 0, 1000).is_err(), "row past the region");
        assert!(l.kv(l.gdn_exec[0], 0).is_err(), "a GDN execution has no KV region");
    }

    #[test]
    fn the_plan_decides_which_executions_carry_which_cache() {
        let plan = plan();
        let g = geom(8, 1, 1, KvKind::F16);
        let mut a = Arena::new("caches");
        let l = CacheLayout::reserve(&mut a, &plan, &g).unwrap();
        for &e in &l.kv_exec {
            assert!(!plan.exec(e).recurrent);
            assert!(plan.exec(e).mtp == 0);
        }
        for &e in &l.gdn_exec {
            assert!(plan.exec(e).recurrent);
        }
        // MTP never gets a region here (Phase 9 owns that program).
        assert!(l.kv_exec.iter().chain(l.gdn_exec.iter()).all(|&e| plan.exec(e).mtp == 0));
    }

    #[test]
    fn snapshots_are_pairwise_and_a_lone_kv_prefix_is_rejected() {
        let mut s = Sessions::new(2, 3);
        for pos in 0..4u32 {
            s.append(0, pos).unwrap();
        }
        let sn = s.snapshot(0).unwrap();
        assert_eq!(sn.len, 4);
        // Two sequences are independent.
        s.append(1, 0).unwrap();
        assert_eq!(s.len(0), 4);
        assert_eq!(s.len(1), 1);
        // Appending out of order is a bug, not a silent overwrite.
        assert!(s.append(0, 4).is_ok());
        assert!(s.append(0, 4).is_err(), "append-only order is checked");
        // Roll back: both halves move together.
        let r = s.restore(0, sn.slot).unwrap();
        assert_eq!(r.len, 4);
        assert_eq!(s.active_snapshot(0), Some(sn.slot));
        assert!(s.restore(0, 9).is_err(), "an unclaimed slot is not a snapshot");
        assert!(s.restore(1, sn.slot).is_err(), "slot 0 of sequence 1 was never taken");
        // The assertion Phase 6 asks for.
        assert!(Sessions::check_pair(4, 4).is_ok());
        assert!(Sessions::check_pair(4, 3).is_err());
        // Adopting a KV prefix needs the GDN snapshot at the same boundary.
        let sn1 = s.snapshot(1).unwrap();
        assert_eq!(sn1.len, 1);
        assert!(s.adopt(1, 1, 1, 0).is_ok());
        assert!(s.adopt(1, 5, 5, 4).is_err(), "no snapshot at length 5");
        assert!(s.adopt(0, 4, 2, 1).is_err(), "a half prefix is rejected before the snapshot lookup");
    }

    #[test]
    fn meta_bytes_match_the_layout_the_device_reads() {
        let mut s = Sessions::new(2, 2);
        s.append(0, 0).unwrap();
        s.append(0, 2).unwrap();
        s.snapshot(0).unwrap();
        let b = s.meta_bytes();
        assert_eq!(b.len(), 2 * META_WORDS * 4);
        let word = |seq: usize, f: usize| {
            i32::from_le_bytes([b[(seq * META_WORDS + f) * 4], b[(seq * META_WORDS + f) * 4 + 1], b[(seq * META_WORDS + f) * 4 + 2], b[(seq * META_WORDS + f) * 4 + 3]])
        };
        assert_eq!(word(0, META_LEN), 2);
        assert_eq!(word(0, META_N_SNAP), 1);
        assert_eq!(word(0, META_LAST_POS), 2);
        assert_eq!(word(1, META_LEN), 0);
    }

    #[test]
    fn snapshot_slots_are_bounded_and_recycled() {
        let mut s = Sessions::new(1, 2);
        s.append(0, 0).unwrap();
        let a = s.snapshot(0).unwrap();
        s.append(0, 1).unwrap();
        let b = s.snapshot(0).unwrap();
        s.append(0, 2).unwrap();
        let c = s.snapshot(0).unwrap();
        assert!(a.slot < 2 && b.slot < 2 && c.slot < 2);
        assert_eq!(s.snapshots(0).len(), 2, "at most n_snap live slots");
        assert!(s.snapshots(0).iter().any(|x| x.len == 3), "the newest survives");
        let mut a2 = Arena::new("c");
        let g = geom(8, 2, 2, KvKind::Q8_0);
        let l = CacheLayout::reserve(&mut a2, &plan(), &g).unwrap();
        assert_ne!(l.snapshot_state(0, 0).unwrap(), l.snapshot_state(0, 1).unwrap());
        assert_ne!(l.snapshot_state(0, 0).unwrap(), l.snapshot_state(1, 0).unwrap());
        assert!(l.snapshot_state(0, 5).is_err());
    }
}
