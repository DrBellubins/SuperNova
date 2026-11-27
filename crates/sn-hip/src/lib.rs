//! `sn-hip` - the ROCm/HIP substrate: one device, three slabs, one static program, one graph.
//!
//! This is Phase 3 of `Docs/design-plan.md` §7. What it owns:
//!
//! * [`hip`] - hand-written bindings over `libamdhip64` (`dlopen`/`dlsym`; no `hipcc`, no bindgen,
//!   no build script) and device discovery.
//! * [`arena`] - compile-time offsets, so no activation address is ever computed at run time.
//! * [`kernels`] - kernel images (offline fat binaries) and the `(name, shape, dtype)` registry.
//! * [`program`] - the `Op` list, the flat `--no-graph` launcher, and graph capture/replay.
//! * [`Substrate`] - the three single allocations (weights / activations / caches) and the pinned
//!   host slot a decode step writes its token id and position into.
//!
//! `unsafe` lives only in this crate (design-plan §6.1: "unsafe confined to sn-hip"), and every
//! `unsafe` block is explicit - the crate denies `unsafe_op_in_unsafe_fn`.

#![deny(unsafe_op_in_unsafe_fn)]
#![allow(clippy::needless_range_loop)]

pub mod arena;
pub mod bench;
pub mod check;
pub mod decode;
pub mod gemm;
pub mod hip;
pub mod kernels;
pub mod ops;
pub mod program;

use std::ffi::c_void;

pub use arena::{Arena, Layout, Slot};
pub use hip::{Device, DeviceInfo, HipApi};
pub use kernels::{DType, Key, Kernel, KernelRegistry};
pub use ops::{Kernels, BLOCK as KERNEL_BLOCK, GEMV_BLOCK};
pub use program::{Arg, Graph, Op, OpKind, Program, ProgramBuilder, Trace};

use sn_core::error::{Error, Result};
use sn_core::gguf::Gguf;

use hip::{check, CAPTURE_MODE_THREAD_LOCAL};

/// Record a kernel launch into a program. The pointers in `args` must already be resolved (from an
/// `arena::Arena` offset plus a slab base) - the whole point of P1 is that nothing here is looked
/// up at replay time.
#[macro_export]
macro_rules! launch {
    ($builder:expr, $label:expr, $kernel:expr, $grid:expr $(, $arg:expr)* $(,)?) => {
        $builder.kernel($label, &$kernel, $grid, 0u32, &[$($arg),*])
    };
}

/// One device allocation. `Drop` frees it; the substrate owns all of them, and there are exactly
/// three of them plus pinned host memory.
pub struct Slab {
    ptr: *mut c_void,
    len: usize,
    what: String,
    api: HipApi,
}

impl Slab {
    fn alloc(api: &HipApi, len: usize, what: &str) -> Result<Slab> {
        if len == 0 {
            return Err(Error::msg(format!("{what}: refusing to allocate a zero-length slab")));
        }
        let mut ptr: *mut c_void = std::ptr::null_mut();
        check(
            &format!("hipMalloc({what})"),
            api.hipMalloc.unwrap()(&mut ptr, len),
            api,
        )
        .map_err(|e| Error::msg(format!("{e} - {what} asked for {} bytes", human(len))))?;
        Ok(Slab { ptr, len, what: what.to_string(), api: *api })
    }

    pub fn ptr(&self) -> *mut c_void {
        self.ptr
    }

    /// Allocate a slab outside the substrate's three. The check harness uses it for the handful of
    /// tensors a stage check needs (the mHC routers) instead of uploading the whole 5.8 GB weight
    /// slab; the engine itself never calls this.
    pub fn new(api: &HipApi, len: usize, what: &str) -> Result<Slab> {
        Slab::alloc(api, len, what)
    }
    pub fn addr(&self) -> u64 {
        self.ptr as u64
    }
    pub fn len(&self) -> usize {
        self.len
    }
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
    pub fn what(&self) -> &str {
        &self.what
    }

    /// `base + offset`, the address an arena `Slot` refers to.
    pub fn at(&self, offset: usize) -> Result<u64> {
        if offset >= self.len {
            return Err(Error::msg(format!(
                "{}: offset {offset} is past the end of a {} byte slab",
                self.what, self.len
            )));
        }
        Ok(self.addr() + offset as u64)
    }

    pub fn zero(&self, api: &HipApi) -> Result<()> {
        check("hipMemset", api.hipMemset.unwrap()(self.ptr, 0, self.len), api)
    }

    pub fn write(&self, api: &HipApi, dst_offset: usize, src: &[u8]) -> Result<()> {
        if dst_offset + src.len() > self.len {
            return Err(Error::msg(format!(
                "{}: writing {} bytes at {dst_offset} overruns a {} byte slab",
                self.what,
                src.len(),
                self.len
            )));
        }
        check(
            &format!("hipMemcpy({})", self.what),
            api.hipMemcpy.unwrap()(
                (self.addr() + dst_offset as u64) as *mut c_void,
                src.as_ptr() as *const c_void,
                src.len(),
                hip::MEMCPY_HOST_TO_DEVICE,
            ),
            api,
        )
    }
}

impl Drop for Slab {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            check("hipFree", self.api.hipFree.unwrap()(self.ptr), &self.api).ok();
            self.ptr = std::ptr::null_mut();
        }
    }
}

/// Page-locked host memory. A decode step writes the next token id and position here; the captured
/// graph's first node copies it to the device, which is why no `hipMemcpy` ever appears in the loop.
pub struct Pinned {
    ptr: *mut c_void,
    len: usize,
    api: HipApi,
}

impl Pinned {
    pub fn alloc(api: &HipApi, len: usize) -> Result<Pinned> {
        let mut ptr: *mut c_void = std::ptr::null_mut();
        check("hipHostMalloc", api.hipHostMalloc.unwrap()(&mut ptr, len, 0), api)?;
        Ok(Pinned { ptr, len, api: *api })
    }
    pub fn addr(&self) -> u64 {
        self.ptr as u64
    }
    pub fn as_mut_bytes(&mut self) -> &mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr as *mut u8, self.len) }
    }
    pub fn len(&self) -> usize {
        self.len
    }
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

impl Drop for Pinned {
    fn drop(&mut self) {
        check("hipHostFree", self.api.hipHostFree.unwrap()(self.ptr), &self.api).ok();
    }
}

/// The device side of the engine: a device, a stream, and the three slabs.
pub struct Substrate {
    pub dev: Device,
    pub stream: *mut c_void,
    weights: Option<Slab>,
    activations: Option<Slab>,
    caches: Option<Slab>,
    io: Option<Pinned>,
    /// Where the GGUF tensor data section starts in the file; weights addresses are relative to it.
    data_start: u64,
}

impl Substrate {
    /// Open the device and create the stream. Never panics without a GPU - it returns an error, so
    /// `sn` keeps working on a machine with no ROCm.
    pub fn open() -> Result<Substrate> {
        let dev = Device::open()?;
        let mut stream: *mut c_void = std::ptr::null_mut();
        check("hipStreamCreate", dev.api.hipStreamCreate.unwrap()(&mut stream), &dev.api)?;
        Ok(Substrate {
            dev,
            stream,
            weights: None,
            activations: None,
            caches: None,
            io: None,
            data_start: 0,
        })
    }

    /// Upload the model: one allocation, one copy, exactly the file's data section. Every tensor
    /// address is then `weights_addr(t.offset)` (design-plan P1: "one `hipMalloc` for weights").
    pub fn upload_weights(&mut self, g: &Gguf) -> Result<usize> {
        let data = g.data_section();
        let slab = Slab::alloc(&self.dev.api, data.len(), "weights")?;
        check(
            "hipMemcpy(weights)",
            self.dev.api.hipMemcpy.unwrap()(
                slab.ptr(),
                data.as_ptr() as *const c_void,
                data.len(),
                hip::MEMCPY_HOST_TO_DEVICE,
            ),
            &self.dev.api,
        )?;
        self.data_start = g.data_start;
        let len = slab.len();
        self.weights = Some(slab);
        Ok(len)
    }

    /// The device address of a tensor given its GGUF `offset` field.
    pub fn weights_addr(&self, offset: u64) -> Result<u64> {
        let w = self.weights()?;
        w.at(offset as usize)
    }

    pub fn weights(&self) -> Result<&Slab> {
        self.weights
            .as_ref()
            .ok_or_else(|| Error::msg("no weights have been uploaded to the device"))
    }

    /// Allocate just the activation slab (the graph gate and the Phase 3 tests use no caches).
    pub fn alloc_activations(&mut self, len: usize) -> Result<()> {
        let a = Slab::alloc(&self.dev.api, len, "activations")?;
        a.zero(&self.dev.api)?;
        self.activations = Some(a);
        Ok(())
    }

    /// Allocate the activation and cache slabs from a frozen layout.
    pub fn alloc_slabs(&mut self, layout: &mut Layout) -> Result<(usize, usize)> {
        let (act, cache) = layout.freeze();
        let a = Slab::alloc(&self.dev.api, act, "activations")?;
        let c = Slab::alloc(&self.dev.api, cache, "caches")?;
        a.zero(&self.dev.api)?;
        c.zero(&self.dev.api)?;
        self.activations = Some(a);
        self.caches = Some(c);
        Ok((act, cache))
    }

    pub fn activations(&self) -> Result<&Slab> {
        self.activations
            .as_ref()
            .ok_or_else(|| Error::msg("the activation slab has not been allocated"))
    }

    pub fn caches(&self) -> Result<&Slab> {
        self.caches
            .as_ref()
            .ok_or_else(|| Error::msg("the cache slab has not been allocated"))
    }

    /// The pinned host slot for token/pos input (`io_bytes` bytes).
    pub fn io(&mut self, io_bytes: usize) -> Result<u64> {
        if self.io.is_none() {
            self.io = Some(Pinned::alloc(&self.dev.api, io_bytes)?);
        }
        Ok(self.io.as_ref().unwrap().addr())
    }

    pub fn io_mut(&mut self) -> Option<&mut Pinned> {
        self.io.as_mut()
    }

    /// Free everything but the device and stream (used by the benchmark's teardown and by tests).
    pub fn release_slabs(&mut self) {
        self.weights = None;
        self.activations = None;
        self.caches = None;
        self.io = None;
    }

    pub fn sync(&self) -> Result<()> {
        check("hipStreamSynchronize", self.dev.api.hipStreamSynchronize.unwrap()(self.stream), &self.dev.api)
    }

    /// A private stream for capture (a capture must not share a stream with unrelated work).
    pub fn new_stream(&self) -> Result<*mut c_void> {
        let mut s: *mut c_void = std::ptr::null_mut();
        check("hipStreamCreate", self.dev.api.hipStreamCreate.unwrap()(&mut s), &self.dev.api)?;
        Ok(s)
    }

    pub fn destroy_stream(&self, stream: *mut c_void) {
        check("hipStreamDestroy", self.dev.api.hipStreamDestroy.unwrap()(stream), &self.dev.api).ok();
    }

    /// Begin/end a capture on `stream` without the `program` helper (used by tests that capture a
    /// hand-built sequence).
    pub fn capture_begin(&self, stream: *mut c_void) -> Result<()> {
        check("hipStreamBeginCapture", self.dev.api.hipStreamBeginCapture.unwrap()(stream, CAPTURE_MODE_THREAD_LOCAL), &self.dev.api)
    }

    pub fn free_device_memory(&self) -> Result<(usize, usize)> {
        let (mut free, mut total) = (0usize, 0usize);
        check("hipMemGetInfo", self.dev.api.hipMemGetInfo.unwrap()(&mut free, &mut total), &self.dev.api)?;
        Ok((free, total))
    }
}

impl Drop for Substrate {
    fn drop(&mut self) {
        if !self.stream.is_null() {
            check("hipStreamDestroy", self.dev.api.hipStreamDestroy.unwrap()(self.stream), &self.dev.api).ok();
            self.stream = std::ptr::null_mut();
        }
    }
}

/// `1.23 GiB` style sizes for reports.
pub fn human(n: usize) -> String {
    format!("{:.3} GiB ({n} B)", n as f64 / (1024.0 * 1024.0 * 1024.0))
}
