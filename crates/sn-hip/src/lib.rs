//! sn-hip — ROCm/HIP device access. Not implemented yet.
//!
//! Phase 3 (`Docs/design-plan.md` §7) replaces this stub with hand-written declarations over
//! `libamdhip64` (`dlopen`/`dlsym`, no `hipcc`, no build script) plus the device buffer allocator
//! and one-shot graph capture. Until then the workspace builds without it and nothing links HIP.
