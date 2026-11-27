//! Hand-written ROCm/HIP bindings, device open, and error mapping.
//!
//! No `hipcc`, no `bindgen`, no build script: the runtime is `libamdhip64.so`, loaded with
//! `dlopen`/`dlsym` at run time (design-plan §6.1). Only the ~30 entry points SuperNova actually
//! uses are declared. Kernel *code* comes from a separately built fat binary (`hipModuleLoad`), see
//! `crate::kernels` and `Tools/build-kernels.sh`.
//!
//! Everything here is `unsafe` on purpose and confined to this crate; `sn-core` never sees it.

use std::ffi::{c_char, c_int, c_void, CStr, CString};
use std::fmt;
use std::os::raw::c_uchar;

use sn_core::error::{Error, Result};

pub type HipError = c_int;
pub const HIP_SUCCESS: HipError = 0;

pub type HipDevice = c_int;
pub type HipStream = *mut c_void;
pub type HipGraph = *mut c_void;
pub type HipGraphExec = *mut c_void;
pub type HipModule = *mut c_void;
pub type HipFunction = *mut c_void;
pub type HipEvent = *mut c_void;

/// `hipMemcpyKind`.
pub const MEMCPY_HOST_TO_DEVICE: c_int = 1;
pub const MEMCPY_DEVICE_TO_HOST: c_int = 2;
pub const MEMCPY_DEVICE_TO_DEVICE: c_int = 3;

/// `hipStreamCaptureMode`: thread-local so a stray capture never poisons other threads.
pub const CAPTURE_MODE_THREAD_LOCAL: c_int = 0;
// `hipStreamCaptureModeGlobal` etc. are unused; `hipStreamCaptureModeRelaxed` would be 2.

/// `hipDeviceGetAttribute` selectors we need (CUDA-compatible range, `hip_runtime_api.h`).
/// `hipDeviceGetAttribute` selectors. These are enum values, so they are **not** guessed: they are
/// the values the installed header defines, printed by `Tools/hip_attrs.c`:
///
/// ```sh
/// gcc -w -D__HIP_PLATFORM_AMD__ -I/opt/rocm/include -o /tmp/attrs Tools/hip_attrs.c && /tmp/attrs
/// ```
pub mod attr {
    pub const CLOCK_RATE: i32 = 5;
    pub const L2_CACHE_SIZE: i32 = 19;
    pub const MAX_THREADS_PER_BLOCK: i32 = 56;
    pub const MAX_THREADS_PER_MULTI_PROCESSOR: i32 = 57;
    pub const MULTIPROCESSOR_COUNT: i32 = 63;
    pub const WAVEFRONT_WIDTH: i32 = 87;
    pub const MAX_REGISTERS_PER_BLOCK: i32 = 71;
    pub const MAX_SHARED_MEMORY_PER_BLOCK: i32 = 74;
    pub const MEMORY_BUS_WIDTH: i32 = 59;
    pub const MEMORY_CLOCK_RATE: i32 = 60;
    /// AMD-specific range.
    pub const MAX_SHARED_MEMORY_PER_MULTIPROCESSOR: i32 = 10002;
    pub const PHYSICAL_MULTIPROCESSOR_COUNT: i32 = 10015;
}


macro_rules! declare_api {
    ($($name:ident : ($($arg:ty),*) -> $ret:ty),* $(,)?) => {
        /// The loaded subset of the HIP runtime. Every field is a real function pointer; a symbol
        /// that `libamdhip64` does not export is `None` and its wrapper returns a clear error.
        // Field names are the HIP symbols, spelled as the ABI spells them.
        #[allow(non_snake_case)]
        #[derive(Clone, Copy)]
        pub struct HipApi {
            $(pub $name: Option<extern "C" fn($($arg),*) -> $ret>,)*
            /// True when the library was found and every symbol Phase 3 needs was resolved.
            pub complete: bool,
            /// Which library was loaded, for `sn hip info`.
            pub library: &'static str,
        }

        impl HipApi {
            #[allow(clippy::missing_safety_doc)]
            pub fn load() -> HipApi {
                let lib = {
                    let mut best: Option<*mut c_void> = None;
                    for name in LIB_PATHS.iter() {
                        let c = CString::new(*name).unwrap();
                        let h = unsafe { libloading_dlopen(c.as_ptr()) };
                        if !h.is_null() { best = Some(h); break; }
                    }
                    best
                };
                let mut api = HipApi {
                    $($name: None,)*
                    complete: false,
                    library: "none",
                };
                if let Some(h) = lib {
                    api.library = LIB_PATHS[0];
                    $(
                        api.$name = unsafe { symbol::<extern "C" fn($($arg),*) -> $ret>(h, stringify!($name)) };
                    )*
                    api.complete = true;
                    $(
                        if api.$name.is_none() && REQUIRED.contains(&stringify!($name)) {
                            api.complete = false;
                        }
                    )*
                    let _ = h;
                }
                api
            }
        }
    };
}

// dlopen/dlsym without pulling in a crate: liblibc symbols come from the binary itself.
extern "C" {
    fn dlopen(filename: *const c_char, flag: c_int) -> *mut c_void;
    fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
}

unsafe fn libloading_dlopen(name: *const c_char) -> *mut c_void {
    // RTLD_NOW | RTLD_GLOBAL
    unsafe { dlopen(name, 2 | 8) }
}

unsafe fn symbol<T>(handle: *mut c_void, name: &str) -> Option<T> {
    let c = CString::new(name).ok()?;
    let p = unsafe { dlsym(handle, c.as_ptr()) };
    if p.is_null() {
        None
    } else {
        Some(unsafe { std::mem::transmute_copy::<*mut c_void, T>(&p) })
    }
}

/// The library names tried, in order. `libamdhip64.so.7` is what a normal ROCm install ships.
const LIB_PATHS: [&str; 4] = [
    "libamdhip64.so",
    "libamdhip64.so.7",
    "/opt/rocm/lib/libamdhip64.so",
    "/usr/lib/x86_64-linux-gnu/libamdhip64.so",
];

/// Symbols without which the substrate cannot run at all. Everything else degrades with an error.
const REQUIRED: &[&str] = &[
    "hipGetDeviceCount",
    "hipDeviceGet",
    "hipDeviceGetAttribute",
    "hipGetDevicePropertiesR0600",
    "hipMalloc",
    "hipFree",
    "hipMemcpy",
    "hipMemcpyAsync",
    "hipHostMalloc",
    "hipHostFree",
    "hipStreamCreate",
    "hipStreamDestroy",
    "hipStreamSynchronize",
    "hipStreamBeginCapture",
    "hipStreamEndCapture",
    "hipGraphInstantiateWithFlags",
    "hipGraphLaunch",
    "hipGraphExecDestroy",
    "hipModuleLoad",
    "hipModuleGetFunction",
    "hipModuleLaunchKernel",
    "hipGetErrorName",
];

declare_api! {
    hipGetDeviceCount: (*mut c_int) -> HipError,
    hipDeviceGet: (*mut HipDevice, c_int) -> HipError,
    hipDeviceGetAttribute: (*mut c_int, c_int, HipDevice) -> HipError,
    hipGetDevicePropertiesR0600: (*mut c_void, HipDevice) -> HipError,
    hipDeviceTotalMemV2: (*mut usize, HipDevice) -> HipError,
    hipMemGetInfo: (*mut usize, *mut usize) -> HipError,
    hipDeviceSynchronize: () -> HipError,
    hipSetDeviceFlags: (u32) -> HipError,
    hipGetErrorName: (HipError) -> *const c_char,
    hipGetErrorString: (HipError) -> *const c_char,

    hipMalloc: (*mut *mut c_void, usize) -> HipError,
    hipFree: (*mut c_void) -> HipError,
    hipMemcpy: (*mut c_void, *const c_void, usize, c_int) -> HipError,
    hipMemcpyAsync: (*mut c_void, *const c_void, usize, c_int, HipStream) -> HipError,
    hipMemset: (*mut c_void, c_int, usize) -> HipError,
    hipHostMalloc: (*mut *mut c_void, usize, std::os::raw::c_uint) -> HipError,
    hipHostFree: (*mut c_void) -> HipError,

    hipStreamCreate: (*mut HipStream) -> HipError,
    hipStreamDestroy: (HipStream) -> HipError,
    hipStreamSynchronize: (HipStream) -> HipError,
    hipStreamBeginCapture: (HipStream, c_int) -> HipError,
    hipStreamEndCapture: (HipStream, *mut HipGraph) -> HipError,
    hipGraphInstantiateWithFlags: (*mut HipGraphExec, HipGraph, u64) -> HipError,
    hipGraphUpload: (HipGraphExec, HipStream) -> HipError,
    hipGraphLaunch: (HipGraphExec, HipStream) -> HipError,
    hipGraphDestroy: (HipGraph) -> HipError,
    hipGraphExecDestroy: (HipGraphExec) -> HipError,

    hipModuleLoad: (*mut HipModule, *const c_char) -> HipError,
    hipModuleGetFunction: (*mut HipFunction, HipModule, *const c_char) -> HipError,
    hipModuleLaunchKernel: (
        HipFunction, u32, u32, u32, u32, u32, u32, u32, HipStream, *mut *mut c_void, *mut *mut c_void
    ) -> HipError,

    hipEventCreate: (*mut HipEvent) -> HipError,
    hipEventRecord: (HipEvent, HipStream) -> HipError,
    hipEventSynchronize: (HipEvent) -> HipError,
    hipEventElapsedTime: (*mut f32, HipEvent, HipEvent) -> HipError,
    hipEventDestroy: (HipEvent) -> HipError,
}

/// Check a HIP return code. `what` names the call so the message is never just a number.
#[inline]
pub fn check(what: &str, e: HipError, api: &HipApi) -> Result<()> {
    if e == HIP_SUCCESS {
        return Ok(());
    }
    let name = unsafe {
        api.hipGetErrorName
            .map(|f| CStr::from_ptr(f(e)).to_string_lossy().into_owned())
            .unwrap_or_else(|| "?".into())
    };
    let text = unsafe {
        api.hipGetErrorString
            .map(|f| CStr::from_ptr(f(e)).to_string_lossy().into_owned())
            .unwrap_or_else(|| "?".into())
    };
    Err(Error::msg(format!("hip: {what} failed: {name} ({text}) [{e}]")))
}

/// Per-device facts Phase 3 and later phases need. Fields are read from `hipDeviceGetAttribute`,
/// except `gcn_arch`, which is read out of the (ABI-stable) `hipDeviceProp_tR0600` buffer.
#[derive(Clone, Debug, Default)]
pub struct DeviceInfo {
    pub index: usize,
    pub name: String,
    /// `gfx1102` etc. Empty when the props buffer did not yield one (`sn hip info` says so).
    pub gcn_arch: String,
    pub total_mem: usize,
    pub free_mem: usize,
    pub multiprocessors: i32,
    pub physical_multiprocessors: i32,
    pub wavefront_size: i32,
    pub max_threads_per_block: i32,
    pub max_threads_per_mp: i32,
    pub max_shared_mem_per_block: i32,
    pub max_registers_per_block: i32,
    pub l2_cache_bytes: i32,
    pub clock_khz: i32,
    pub mem_clock_khz: i32,
    pub mem_bus_width: i32,
}

impl DeviceInfo {
    /// Peak device memory traffic in bytes/s, from the reported memory clock and bus width. Used by
    /// the roofline check in `sn hip info`; it is a spec figure, so `Docs/baseline.md` stays the
    /// source of the achievable number.
    pub fn spec_bandwidth(&self) -> f64 {
        (self.mem_clock_khz as f64) * 1e3 * (self.mem_bus_width as f64) / 8.0
    }
}

/// One HIP device, opened and ready. The first (and usually only) handle the engine holds.
pub struct Device {
    pub api: HipApi,
    pub raw: HipDevice,
    pub info: DeviceInfo,
}

impl Device {
    /// Load the runtime and open device `index` (0 unless `SN_GPU` says otherwise). Returns an
    /// error, never a panic, when there is no runtime or no device - so `sn` still runs its CPU
    /// paths on a machine without a GPU.
    pub fn open() -> Result<Device> {
        let api = HipApi::load();
        if api.library == "none" {
            return Err(Error::msg(format!(
                "HIP runtime not found (tried {})",
                LIB_PATHS.join(", ")
            )));
        }
        if !api.complete {
            return Err(Error::msg(format!(
                "{} does not export every symbol Phase 3 needs",
                api.library
            )));
        }
        let want: usize = std::env::var("SN_GPU").ok().and_then(|v| v.parse().ok()).unwrap_or(0);
        let mut n = 0;
        check("hipGetDeviceCount", api.hipGetDeviceCount.unwrap()(&mut n), &api)?;
        if n == 0 {
            return Err(Error::msg("the HIP runtime reports zero devices"));
        }
        if want >= n as usize {
            return Err(Error::msg(format!(
                "SN_GPU={want} is out of range: {n} device(s) present"
            )));
        }
        let mut raw: HipDevice = 0;
        check("hipDeviceGet", api.hipDeviceGet.unwrap()(&mut raw, want as c_int), &api)?;
        let info = probe(&api, raw, want)?;
        Ok(Device { api, raw, info })
    }

    pub fn attribute(&self, which: c_int) -> Result<i32> {
        let mut v = 0;
        check(
            "hipDeviceGetAttribute",
            self.api.hipDeviceGetAttribute.unwrap()(&mut v, which, self.raw),
            &self.api,
        )?;
        Ok(v)
    }

    pub fn sync(&self) -> Result<()> {
        check("hipDeviceSynchronize", self.api.hipDeviceSynchronize.unwrap()(), &self.api)
    }
}

fn attr_of(api: &HipApi, dev: HipDevice, which: c_int) -> i32 {
    let mut v = 0;
    match api.hipDeviceGetAttribute {
        Some(f) if f(&mut v, which, dev) == HIP_SUCCESS => v,
        _ => 0,
    }
}

/// `hipGetDevicePropertiesR0600` fills a fixed-ABI struct; rather than restating ~90 fields in
/// Rust (which would silently break on the next ROCm), read it into a byte buffer and take the
/// `gcnArchName` char array by scanning for the `gfx...` string it always contains. Everything
/// numeric comes from `hipDeviceGetAttribute`, whose ABI is genuinely stable.
fn probe(api: &HipApi, dev: HipDevice, index: usize) -> Result<DeviceInfo> {
    let mut info = DeviceInfo {
        index,
        multiprocessors: attr_of(api, dev, attr::MULTIPROCESSOR_COUNT),
        physical_multiprocessors: attr_of(api, dev, attr::PHYSICAL_MULTIPROCESSOR_COUNT),
        wavefront_size: attr_of(api, dev, attr::WAVEFRONT_WIDTH),
        max_threads_per_block: attr_of(api, dev, attr::MAX_THREADS_PER_BLOCK),
        max_threads_per_mp: attr_of(api, dev, attr::MAX_THREADS_PER_MULTI_PROCESSOR),
        max_shared_mem_per_block: attr_of(api, dev, attr::MAX_SHARED_MEMORY_PER_BLOCK),
        max_registers_per_block: attr_of(api, dev, attr::MAX_REGISTERS_PER_BLOCK),
        l2_cache_bytes: attr_of(api, dev, attr::L2_CACHE_SIZE),
        clock_khz: attr_of(api, dev, attr::CLOCK_RATE),
        mem_clock_khz: attr_of(api, dev, attr::MEMORY_CLOCK_RATE),
        mem_bus_width: attr_of(api, dev, attr::MEMORY_BUS_WIDTH),
        ..Default::default()
    };
    let mut total = 0usize;
    if let Some(f) = api.hipDeviceTotalMemV2 {
        if f(&mut total, dev) == HIP_SUCCESS {
            info.total_mem = total;
        }
    }
    let (mut free, mut tot) = (0usize, 0usize);
    if let Some(f) = api.hipMemGetInfo {
        if f(&mut free, &mut tot) == HIP_SUCCESS {
            info.free_mem = free;
            if info.total_mem == 0 {
                info.total_mem = tot;
            }
        }
    }

    // props buffer: 8 KiB is far larger than hipDeviceProp_tR0600 and is zero-filled first.
    let mut buf = vec![0u8; 8192];
    if let Some(f) = api.hipGetDevicePropertiesR0600 {
        if f(buf.as_mut_ptr() as *mut c_void, dev) == HIP_SUCCESS {
            info.gcn_arch = find_arch(&buf);
            info.name = first_ascii(&buf, 4).unwrap_or_default();
        }
    }
    if info.name.is_empty() {
        info.name = format!("device {index}");
    }
    Ok(info)
}

fn find_arch(buf: &[u8]) -> String {
    let hay = b"gfx";
    let mut i = 0;
    while i + hay.len() <= buf.len() {
        if &buf[i..i + 3] == hay {
            let mut j = i;
            while j < buf.len() && (buf[j].is_ascii_alphanumeric() || buf[j] == b'_') {
                j += 1;
            }
            let s = String::from_utf8_lossy(&buf[i..j]).into_owned();
            if s.len() > 3 {
                return s;
            }
        }
        i += 1;
    }
    String::new()
}

/// The first run of printable bytes at least `min` long - in `hipDeviceProp_tR0600` that is `name`.
fn first_ascii(buf: &[u8], min: usize) -> Option<String> {
    let mut i = 0;
    while i < buf.len() {
        let mut j = i;
        while j < buf.len() && (buf[j] as c_uchar).is_ascii_graphic() || j < buf.len() && buf[j] == b' '
        {
            j += 1;
        }
        if j - i >= min {
            return Some(String::from_utf8_lossy(&buf[i..j]).trim().to_string());
        }
        i = j + 1;
    }
    None
}

/// What `sn hip info` prints for one device.
impl fmt::Display for DeviceInfo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "device {}  {}", self.index, self.name)?;
        writeln!(
            f,
            "  isa                  {}",
            if self.gcn_arch.is_empty() { "unknown (props buffer carried no 'gfx' string)" } else { &self.gcn_arch }
        )?;
        writeln!(
            f,
            "  memory             {:.3} GB total, {:.3} GB free",
            self.total_mem as f64 / 1e9,
            self.free_mem as f64 / 1e9
        )?;
        writeln!(
            f,
            "  CUs                {} reported, {} physical   wavefront {}",
            self.multiprocessors, self.physical_multiprocessors, self.wavefront_size
        )?;
        writeln!(
            f,
            "  per block          {} threads max, {} regs, {} B shared",
            self.max_threads_per_block, self.max_registers_per_block, self.max_shared_mem_per_block
        )?;
        writeln!(f, "  L2                 {} bytes", self.l2_cache_bytes)?;
        // The memory *clock* attribute is the native clock, not the effective data rate (GDDR6 is
        // multi-pumped), so no bandwidth is derived here; Docs/baseline.md §0 has measured figures.
        writeln!(
            f,
            "  clocks             SM {} MHz, mem native {} MHz x {} bit (no bandwidth derived here)",
            self.clock_khz / 1000,
            self.mem_clock_khz / 1000,
            self.mem_bus_width
        )
    }
}
