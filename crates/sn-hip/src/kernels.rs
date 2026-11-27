//! Kernel images and the `(name, shape, dtype)` registry.
//!
//! Kernel *source* lives in `crates/sn-hip/kernels/*.hip`; it is compiled offline by
//! `Tools/build-kernels.sh` into a fat binary and loaded here with `hipModuleLoad` - so the Rust
//! build needs no HIP toolchain, and the ISA a kernel was built for is explicit and checkable.

use std::collections::BTreeMap;
use std::ffi::CString;
use std::path::Path;

use sn_core::error::{Error, Result};

use crate::hip::{check, Device, HipFunction};

/// The operand types a kernel specialization can be keyed by. Only types this checkpoint actually
/// stores appear here (design-plan §1.2: Q4_K, Q6_K, f32 - plus f16/bf16 for activations).
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub enum DType {
    F32,
    F16,
    Bf16,
    Q4K,
    Q6K,
    /// Key-independent: the kernel's operands are not type-relevant (index copies, the graph probe).
    Any,
}

impl std::fmt::Display for DType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            DType::F32 => "f32",
            DType::F16 => "f16",
            DType::Bf16 => "bf16",
            DType::Q4K => "q4_k",
            DType::Q6K => "q6_k",
            DType::Any => "any",
        })
    }
}

/// A kernel selection key: what a builder asks for, not what a compiler produced.
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub struct Key {
    pub name: String,
    /// The shape the kernel was specialised for (e.g. `[rows, cols, 1, 1]` for a gemv, or the empty
    /// list for a shape-independent kernel).
    pub shape: Vec<i64>,
    pub dtype: DType,
}

impl std::fmt::Display for Key {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}<{}>[", self.name, self.dtype)?;
        for (i, d) in self.shape.iter().enumerate() {
            if i > 0 {
                write!(f, ",")?;
            }
            write!(f, "{d}")?;
        }
        write!(f, "]")
    }
}

/// One loaded function plus the launch geometry it was validated against.
#[derive(Clone, Copy, Debug)]
pub struct Kernel {
    pub func: HipFunction,
    /// Workgroup size the kernel was built for (Phase 5 tunes this per shape).
    pub block: [u32; 3],
}

/// A loaded kernel image. `Drop` frees it, which invalidates every `Kernel` handed out - in practice
/// the registry outlives every program.
pub struct Module {
    module: *mut std::ffi::c_void,
    path: String,
    funcs: BTreeMap<String, HipFunction>,
}

impl Module {
    pub fn load(dev: &Device, path: &str) -> Result<Module> {
        if !Path::new(path).exists() {
            return Err(Error::msg(format!(
                "kernel image {path} does not exist - build it with Tools/build-kernels.sh \
                 (it runs hipcc for the device's ISA and writes a fat binary)"
            )));
        }
        let c = CString::new(path).map_err(|_| Error::msg("kernel image path contains a NUL"))?;
        let mut module: *mut std::ffi::c_void = std::ptr::null_mut();
        check("hipModuleLoad", dev.api.hipModuleLoad.unwrap()(&mut module, c.as_ptr()), &dev.api)?;
        Ok(Module { module, path: path.to_string(), funcs: BTreeMap::new() })
    }

    pub fn path(&self) -> &str {
        &self.path
    }

    /// Look a symbol up once and remember it. A missing symbol is a hard error naming the key:
    /// there is no silent fallback to a slower kernel.
    pub fn function(&mut self, dev: &Device, symbol: &str) -> Result<HipFunction> {
        if let Some(f) = self.funcs.get(symbol) {
            return Ok(*f);
        }
        let c = CString::new(symbol).map_err(|_| Error::msg("symbol contains a NUL"))?;
        let mut f: HipFunction = std::ptr::null_mut();
        check(
            "hipModuleGetFunction",
            dev.api.hipModuleGetFunction.unwrap()(&mut f, self.module, c.as_ptr()),
            &dev.api,
        )
        .map_err(|e| Error::msg(format!("{e} - {symbol} is not in {}", self.path)))?;
        self.funcs.insert(symbol.to_string(), f);
        Ok(f)
    }

    pub fn symbols(&self) -> impl Iterator<Item = &str> {
        self.funcs.keys().map(|s| s.as_str())
    }
}

impl Drop for Module {
    fn drop(&mut self) {
        // Best effort: unloading a module while a graph still references it is a programming error
        // elsewhere, and hipModuleUnload is not in the loaded symbol set (Phase 3 never unloads).
        let _ = self.module;
    }
}

/// The registry every program builder resolves kernels through.
#[derive(Default)]
pub struct KernelRegistry {
    modules: Vec<Module>,
    by_key: BTreeMap<Key, Kernel>,
}

impl KernelRegistry {
    pub fn new() -> KernelRegistry {
        KernelRegistry::default()
    }

    /// Load one kernel image. Called at model-load time, never in a decode step.
    pub fn load_image(dev: &Device, path: &str) -> Result<KernelRegistry> {
        let mut r = KernelRegistry::new();
        r.add_image(dev, path)?;
        Ok(r)
    }

    pub fn add_image(&mut self, dev: &Device, path: &str) -> Result<()> {
        let meta = std::path::Path::new(path).with_extension("meta");
        if let Ok(text) = std::fs::read_to_string(&meta) {
            for (key, want) in [
                ("sn_block", crate::ops::BLOCK),
                ("sn_route_threads", crate::ops::ROUTE_BLOCK),
            ] {
                if let Some(got) = meta_u32(&text, key) {
                    if got != want {
                        return Err(Error::msg(format!(
                            "{path} was compiled with {key}={got} but this build of sn-hip launches that kernel with {want}: \
                             the kernel image is stale. Run Tools/build-kernels.sh, and do not send \
                             its output to /dev/null: a failed compile leaves the old image in place.")));
                    }
                }
            }
        }
        self.modules.push(Module::load(dev, path)?);
        Ok(())
    }

    /// Resolve `symbol` from the loaded images and register it under `key` with `block`.
    pub fn register(
        &mut self,
        dev: &Device,
        key: &Key,
        symbol: &str,
        block: [u32; 3],
    ) -> Result<Kernel> {
        if block[0] == 0 || block.iter().any(|b| *b > 1024) {
            return Err(Error::msg(format!(
                "{}: workgroup {:?} is not valid (workgroup size must be 1..=1024)",
                key, block
            )));
        }
        let mut last: Option<Error> = None;
        for m in self.modules.iter_mut() {
            match m.function(dev, symbol) {
                Ok(func) => {
                    let k = Kernel { func, block };
                    self.by_key.insert(key.clone(), k);
                    return Ok(k);
                }
                Err(e) => last = Some(e),
            }
        }
        Err(last.unwrap_or_else(|| Error::msg("no kernel images are loaded")))
    }

    pub fn find(&self, name: &str, shape: &[i64], dtype: DType) -> Option<&Kernel> {
        let dt = if self.by_key.keys().any(|k| k.dtype == dtype) { dtype } else { DType::Any };
        self.by_key
            .iter()
            .find(|(k, _)| k.name == name && k.shape == shape && k.dtype == dt)
            .map(|(_, v)| v)
            .or_else(|| {
                // A shape-generic registration is the fallback for an unlisted shape: same name,
                // same dtype, no shape specialization.
                self.by_key
                    .iter()
                    .find(|(k, _)| k.name == name && k.shape.is_empty() && k.dtype == dt)
                    .map(|(_, v)| v)
            })
    }

    pub fn require(&self, name: &str, shape: &[i64], dtype: DType) -> Result<&Kernel> {
        self.find(name, shape, dtype).ok_or_else(|| {
            Error::msg(format!(
                "no kernel registered for {name}<{}>[{}] (registered: {})",
                dtype,
                shape_for_msg(shape),
                self.keys()
                    .map(|k| k.to_string())
                    .collect::<Vec<_>>()
                    .join(", ")
            ))
        })
    }

    pub fn keys(&self) -> impl Iterator<Item = &Key> {
        self.by_key.keys()
    }

    pub fn len(&self) -> usize {
        self.by_key.len()
    }

    pub fn is_empty(&self) -> bool {
        self.by_key.is_empty()
    }
}

fn shape_for_msg(shape: &[i64]) -> String {
    if shape.is_empty() {
        "-".to_string()
    } else {
        shape.iter().map(|d| d.to_string()).collect::<Vec<_>>().join("x")
    }
}

/// `key=value` from a `Tools/build-kernels.sh` sidecar file. Absent, empty, or unparsable reads as
/// absent: the sidecar is a guard, not a required input.
fn meta_u32(text: &str, key: &str) -> Option<u32> {
    text.lines().find_map(|l| {
        let (k, v) = l.split_once('=')?;
        if k.trim() == key { v.trim().parse::<u32>().ok() } else { None }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keys_sort_and_display_deterministically() {
        let a = Key { name: "gemv".into(), shape: vec![12288, 4096], dtype: DType::Q4K };
        let b = Key { name: "gemv".into(), shape: vec![12288, 4096], dtype: DType::Q6K };
        assert!(a < b, "dtype is part of the ordering");
        assert_eq!(a.to_string(), "gemv<q4_k>[12288,4096]");
        assert_eq!(Key { name: "rmsnorm".into(), shape: vec![], dtype: DType::Any }.to_string(), "rmsnorm<any>[]");
    }

    #[test]
    fn shape_generic_registration_is_the_fallback() {
        let mut r = KernelRegistry::new();
        let k = Kernel { func: std::ptr::null_mut(), block: [64, 1, 1] };
        r.by_key.insert(Key { name: "rmsnorm".into(), shape: vec![], dtype: DType::F32 }, k);
        assert!(r.find("rmsnorm", &[4096], DType::F32).is_some());
        assert!(r.find("rmsnorm", &[4096], DType::F16).is_none());
        assert!(r.find("nope", &[1], DType::F32).is_none());
        // an unlisted shape still resolves to the shape-generic registration...
        assert!(r.require("rmsnorm", &[999], DType::F32).is_ok());
        // ...an unregistered name, or an unregistered dtype, is a hard error naming the key.
        let e = r.require("rmsnorm", &[999], DType::Q4K).unwrap_err().to_string();
        assert!(e.contains("rmsnorm<q4_k>"), "{e}");
        assert!(r.require("gemv", &[], DType::F32).is_err());
    }
}
