//! Minimal kernel FFI, hand-declared so the core keeps zero dependencies.
//!
//! Only what the loader needs today (read-only file mapping). Phase 3 adds the HIP bindings in the
//! `sn-hip` crate, in the same style.
#![allow(non_camel_case_types)]
use std::fmt;
use std::os::raw::c_void;

const PROT_READ: i32 = 1;
const MAP_PRIVATE: i32 = 0x02;
const MAP_POPULATE: i32 = 0x008000;
pub const MADV_NORMAL: i32 = 0;
pub const MADV_SEQUENTIAL: i32 = 2;
pub const MADV_WILLNEED: i32 = 3;
pub const MADV_HUGEPAGE: i32 = 14;

extern "C" {
    fn mmap(
        addr: *mut c_void,
        len: usize,
        prot: i32,
        flags: i32,
        fd: i32,
        offset: i64,
    ) -> *mut c_void;
    fn munmap(addr: *mut c_void, len: usize) -> i32;
    fn madvise(addr: *mut c_void, len: usize, advice: i32) -> i32;
}

/// A read-only file mapping. `Send + Sync`: the pages are the file's page cache, shared safely.
pub struct Mapping {
    ptr: *mut u8,
    len: usize,
}

unsafe impl Send for Mapping {}
unsafe impl Sync for Mapping {}

impl Mapping {
    /// Map `len` bytes of an open file read-only. `populate` prefaults (fast first pass, slower open).
    pub fn map_file(
        fd: i32,
        len: usize,
        populate: bool,
    ) -> std::result::Result<Mapping, std::io::Error> {
        if len == 0 {
            return Err(std::io::Error::new(std::io::ErrorKind::InvalidInput, "empty mapping"));
        }
        let flags = MAP_PRIVATE | if populate { MAP_POPULATE } else { 0 };
        let p = unsafe { mmap(std::ptr::null_mut(), len, PROT_READ, flags, fd, 0) };
        if p == (-1isize) as *mut c_void {
            return Err(std::io::Error::last_os_error());
        }
        Ok(Mapping { ptr: p as *mut u8, len })
    }

    pub fn as_bytes(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Best-effort hint; errors are ignored on purpose (an advice is never fatal).
    pub fn advise(&self, advice: i32) {
        unsafe {
            madvise(self.ptr as *mut c_void, self.len, advice);
        }
    }
}

impl Drop for Mapping {
    fn drop(&mut self) {
        unsafe {
            munmap(self.ptr as *mut c_void, self.len);
        }
    }
}

impl fmt::Debug for Mapping {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Mapping({:p}, {} bytes)", self.ptr, self.len)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::os::unix::io::AsRawFd;

    #[test]
    fn mmap_round_trip() {
        let f = tempfile();
        let fd = f.as_raw_fd();
        let len = f.metadata().unwrap().len() as usize;
        let m = Mapping::map_file(fd, len, false).unwrap();
        assert_eq!(m.as_bytes(), b"hello supernova\n");
        m.advise(MADV_SEQUENTIAL);
    }

    fn tempfile() -> std::fs::File {
        use std::io::Write;
        let path = std::env::temp_dir().join(format!("sn-sys-test-{}", std::process::id()));
        let mut f = std::fs::File::create(&path).unwrap();
        f.write_all(b"hello supernova\n").unwrap();
        f.flush().unwrap();
        let f = std::fs::File::open(&path).unwrap();
        drop(std::fs::remove_file(&path)); // unlink eagerly; the fd stays valid
        f
    }
}
