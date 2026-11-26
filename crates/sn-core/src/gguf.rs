//! GGUF v2/v3 reader: header, typed KV metadata, tensor info, exact byte sizes, data access.
//!
//! Layout of a GGUF file (this is a public file format; the reader is written from the format, not
//! from any existing implementation):
//!
//! ```text
//! "GGUF" u32=version u64=n_tensors u64=n_kv
//! n_kv *  ( str key, u32 type, value )
//! n_tensors * ( str name, u32 n_dims, u64 ne[n_dims], u32 type, u64 offset )
//! pad to `general.alignment`                              <- data_start
//! tensor data, each tensor at data_start + offset
//! ```
//!
//! Tensor byte sizes come from the type (`qtypes::bytes_for_n`); the gap to the next offset is
//! checked against it, because GGUF aligns every tensor and the difference must be pure padding.

use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::os::unix::io::AsRawFd;

use crate::qtypes::{self, ty, TypeProps};
use crate::sys::Mapping;
use crate::{Error, Result};

pub const GGUF_MAGIC: &[u8; 4] = b"GGUF";
pub const GGUF_V2: u32 = 2;
pub const GGUF_V3: u32 = 3;
pub const DEFAULT_ALIGNMENT: u64 = 32;

/// Metadata value. `Arr` keeps its element values boxed so arrays of any type stay uniform.
#[derive(Clone, Debug, PartialEq)]
pub enum Value {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    F32(f32),
    Bool(bool),
    Str(String),
    U64(u64),
    I64(i64),
    F64(f64),
    Arr(Vec<Value>),
}

impl Value {
    pub fn type_name(&self) -> &'static str {
        match self {
            Value::U8(_) => "u8",
            Value::I8(_) => "i8",
            Value::U16(_) => "u16",
            Value::I16(_) => "i16",
            Value::U32(_) => "u32",
            Value::I32(_) => "i32",
            Value::F32(_) => "f32",
            Value::Bool(_) => "bool",
            Value::Str(_) => "str",
            Value::U64(_) => "u64",
            Value::I64(_) => "i64",
            Value::F64(_) => "f64",
            Value::Arr(_) => "arr",
        }
    }

    fn tag(t: u32) -> Option<&'static str> {
        Some(match t {
            0 => "u8",
            1 => "i8",
            2 => "u16",
            3 => "i16",
            4 => "u32",
            5 => "i32",
            6 => "f32",
            7 => "bool",
            8 => "str",
            9 => "arr",
            10 => "u64",
            11 => "i64",
            12 => "f64",
            _ => return None,
        })
    }

    // Lossy-but-explicit accessors: `plan` asks for the type a key must have and gets a clear error
    // if the file disagrees, instead of a silent conversion.
    pub fn as_u32(&self) -> Option<u32> {
        match self {
            Value::U8(v) => Some(*v as u32),
            Value::U16(v) => Some(*v as u32),
            Value::U32(v) => Some(*v),
            Value::I8(v) => u32::try_from(*v).ok(),
            Value::I16(v) => u32::try_from(*v).ok(),
            Value::I32(v) => u32::try_from(*v).ok(),
            Value::Bool(v) => Some(*v as u32),
            _ => None,
        }
    }
    pub fn as_i32(&self) -> Option<i32> {
        match self {
            Value::I8(v) => Some(*v as i32),
            Value::I16(v) => Some(*v as i32),
            Value::I32(v) => Some(*v),
            Value::U8(v) => Some(*v as i32),
            Value::U16(v) => Some(*v as i32),
            Value::U32(v) => i32::try_from(*v).ok(),
            _ => None,
        }
    }
    pub fn as_f32(&self) -> Option<f32> {
        match self {
            Value::F32(v) => Some(*v),
            Value::F64(v) => Some(*v as f32),
            Value::U32(v) => Some(*v as f32),
            Value::I32(v) => Some(*v as f32),
            _ => None,
        }
    }
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Value::Bool(v) => Some(*v),
            Value::U8(v) => Some(*v != 0),
            Value::U32(v) => Some(*v != 0),
            _ => None,
        }
    }
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Value::Str(s) => Some(s),
            _ => None,
        }
    }
    pub fn as_arr(&self) -> Option<&[Value]> {
        match self {
            Value::Arr(a) => Some(a),
            _ => None,
        }
    }
}

/// Rendering for `sn dump --meta`, matching how `Tools/gguf_dump.py` prints values.
impl std::fmt::Display for Value {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Value::Str(s) => write!(f, "{s}"),
            Value::Bool(b) => write!(f, "{}", *b as u8),
            Value::F32(v) => write!(f, "{v}"),
            Value::F64(v) => write!(f, "{v}"),
            Value::Arr(a) => {
                if a.len() > 24 {
                    return write!(f, "[{} items]", a.len());
                }
                let parts: Vec<String> = a.iter().map(|v| v.to_string()).collect();
                write!(f, "[{}]", parts.join(", "))
            }
            Value::U8(v) => write!(f, "{v}"),
            Value::I8(v) => write!(f, "{v}"),
            Value::U16(v) => write!(f, "{v}"),
            Value::I16(v) => write!(f, "{v}"),
            Value::U32(v) => write!(f, "{v}"),
            Value::I32(v) => write!(f, "{v}"),
            Value::U64(v) => write!(f, "{v}"),
            Value::I64(v) => write!(f, "{v}"),
        }
    }
}

/// One tensor-info record plus the sizes we derived for it.
#[derive(Clone, Debug)]
pub struct TensorInfo {
    pub name: String,
    /// Number of dimensions as written (1..=4).
    pub n_dims: u32,
    /// Extents, padded to 4 with 1s. `ne[0]` is the *contiguous* / input dimension.
    pub ne: [u64; 4],
    pub ty: u32,
    /// Offset from `data_start`.
    pub offset: u64,
    /// Exact bytes implied by the element type and extents.
    pub size: u64,
    /// Bytes to the next tensor's offset (i.e. `size` + alignment padding).
    pub stride: u64,
}

impl TensorInfo {
    pub fn n_elts(&self) -> u64 {
        qtypes::n_elements(&self.ne)
    }
    pub fn props(&self) -> Option<TypeProps> {
        qtypes::props(self.ty)
    }
    /// Matrix view: `[ne[0], ne[1]]` = `[n_in, n_out]`, higher dims ignored.
    pub fn as_mat(&self) -> Option<(u64, u64)> {
        if self.n_dims != 2 {
            return None;
        }
        Some((self.ne[0], self.ne[1]))
    }
}

/// A parsed GGUF file: metadata + tensor table + a read-only mapping of the data.
pub struct Gguf {
    pub version: u32,
    pub alignment: u64,
    pub data_start: u64,
    pub file_len: u64,
    pub path: std::path::PathBuf,
    /// KV pairs in file order.
    pub kvs: Vec<(String, Value)>,
    tensors: Vec<TensorInfo>,
    kv_index: HashMap<String, usize>,
    tensor_index: HashMap<String, usize>,
    map: Mapping,
}

impl Gguf {
    pub fn open<P: AsRef<std::path::Path>>(path: P) -> Result<Gguf> {
        let path = path.as_ref();
        let mut f = File::open(path).map_err(|e| Error::msg(format!("open {}: {e}", path.display())))?;
        let file_len = f.metadata().map_err(Error::Io)?.len();

        // The header holds the vocab, so its size is not bounded by a constant: read it in growing
        // prefixes and re-parse until it is complete. Cheaper than mapping the whole 5.8 GB just to
        // read a few MB.
        let mut cap: usize = 1 << 22;
        let parsed;
        loop {
            let want = cap.min(file_len as usize);
            let mut buf = vec![0u8; want];
            f.seek(SeekFrom::Start(0)).map_err(Error::Io)?;
            f.read_exact(&mut buf).map_err(Error::Io)?;
            match parse_header(&buf) {
                Ok(p) => {
                    parsed = p;
                    break;
                }
                Err(HeaderError::Short) if want < file_len as usize => {
                    cap *= 4;
                }
                Err(HeaderError::Short) => {
                    return Err(Error::msg(format!(
                        "{}: header did not fit in {} bytes (truncated file?)",
                        path.display(),
                        want
                    )));
                }
                Err(HeaderError::Msg(m)) => return Err(Error::msg(m)),
            }
        }
        let align = match parsed.kvs.iter().find(|(k, _)| k == "general.alignment") {
            Some((_, v)) => v.as_u32().map(|a| a as u64).unwrap_or(DEFAULT_ALIGNMENT),
            None => DEFAULT_ALIGNMENT,
        };
        if align == 0 || align.count_ones() != 1 {
            return Err(Error::msg(format!("general.alignment = {align} is not a power of two")));
        }
        let data_start = (parsed.header_end as u64).div_ceil(align) * align;
        if data_start > file_len {
            return Err(Error::msg("tensor data starts past the end of the file"));
        }

        // Strides come from the offset order (the format guarantees offsets are non-decreasing).
        let mut order: Vec<usize> = (0..parsed.tensors.len()).collect();
        order.sort_by_key(|&i| parsed.tensors[i].offset);
        let mut tensors = parsed.tensors;
        for (j, &i) in order.iter().enumerate() {
            let next = if j + 1 < order.len() {
                tensors[order[j + 1]].offset
            } else {
                file_len - data_start
            };
            tensors[i].stride = next - tensors[i].offset;
        }
        // Exact sizes come from the type; the offsets only tell us where things live.
        for t in tensors.iter_mut() {
            let want = qtypes::bytes_for_n(t.n_elts(), t.ty).ok_or_else(|| {
                Error::msg(format!(
                    "tensor {}: unsupported element type {}",
                    t.name,
                    qtypes::name(t.ty)
                ))
            })?;
            t.size = want;
        }
        // Validate sizes/strides: a stride smaller than the type-implied size means the file or our
        // understanding of it is wrong, so fail loudly rather than reading garbage.
        for t in &tensors {
            let want = t.size;
            if t.stride < want {
                return Err(Error::msg(format!(
                    "tensor {}: stride {} < size {} (overlapping tensors)",
                    t.name, t.stride, want
                )));
            }
            if t.stride - want >= align {
                return Err(Error::msg(format!(
                    "tensor {}: gap {} exceeds alignment {align}",
                    t.name,
                    t.stride - want
                )));
            }
            if t.offset.checked_add(t.size).map(|e| data_start + e > file_len) == Some(true) {
                return Err(Error::msg(format!("tensor {} runs past the end of the file", t.name)));
            }
        }

        let kv_index: HashMap<String, usize> =
            parsed.kvs.iter().enumerate().map(|(i, (k, _))| (k.clone(), i)).collect();
        let tensor_index: HashMap<String, usize> =
            tensors.iter().enumerate().map(|(i, t)| (t.name.clone(), i)).collect();

        let map = Mapping::map_file(f.as_raw_fd(), file_len as usize, false)
            .map_err(|e| Error::msg(format!("mmap {}: {e}", path.display())))?;

        Ok(Gguf {
            version: parsed.version,
            alignment: align,
            data_start,
            file_len,
            path: path.to_path_buf(),
            kvs: parsed.kvs,
            tensors,
            kv_index,
            tensor_index,
            map,
        })
    }

    // ---- metadata access ----

    pub fn path(&self) -> &std::path::Path {
        &self.path
    }
    pub fn version(&self) -> u32 {
        self.version
    }
    pub fn file_len(&self) -> u64 {
        self.file_len
    }
    pub fn kv_len(&self) -> usize {
        self.kvs.len()
    }
    pub fn kv(&self) -> &[(String, Value)] {
        &self.kvs
    }

    pub fn value(&self, key: &str) -> Option<&Value> {
        self.kv_index.get(key).map(|&i| &self.kvs[i].1)
    }

    fn need<'a>(&'a self, key: &str, want: &str) -> Result<&'a Value> {
        self.value(key).ok_or_else(|| Error::msg(format!("missing metadata key {key} ({want} expected)")))
    }

    pub fn str(&self, key: &str) -> Result<&str> {
        let v = self.need(key, "str")?;
        v.as_str().ok_or_else(|| Error::msg(format!("{key}: expected str, got {}", v.type_name())))
    }

    pub fn u32(&self, key: &str) -> Result<u32> {
        let v = self.need(key, "u32")?;
        v.as_u32().ok_or_else(|| Error::msg(format!("{key}: expected u32, got {}", v.type_name())))
    }

    pub fn opt_u32(&self, key: &str) -> Option<u32> {
        self.value(key).and_then(|v| v.as_u32())
    }

    pub fn f32(&self, key: &str) -> Result<f32> {
        let v = self.need(key, "f32")?;
        v.as_f32().ok_or_else(|| Error::msg(format!("{key}: expected f32, got {}", v.type_name())))
    }

    pub fn opt_f32(&self, key: &str) -> Option<f32> {
        self.value(key).and_then(|v| v.as_f32())
    }

    pub fn bool(&self, key: &str) -> Result<bool> {
        let v = self.need(key, "bool")?;
        v.as_bool().ok_or_else(|| Error::msg(format!("{key}: expected bool, got {}", v.type_name())))
    }

    /// A metadata array as u32 elements (the usual shape for per-layer flags).
    pub fn arr_u32(&self, key: &str) -> Result<Vec<u32>> {
        let v = self.need(key, "array")?;
        let a = v.as_arr().ok_or_else(|| Error::msg(format!("{key}: expected array, got {}", v.type_name())))?;
        a.iter()
            .map(|e| e.as_u32().ok_or_else(|| Error::msg(format!("{key}: non-integer array element"))))
            .collect()
    }

    pub fn opt_str(&self, key: &str) -> Option<&str> {
        self.value(key).and_then(|v| v.as_str())
    }

    pub fn opt_bool(&self, key: &str) -> Option<bool> {
        self.value(key).and_then(|v| v.as_bool())
    }

    /// Number of elements in a metadata array (any element type), or None if absent.
    pub fn arr_len(&self, key: &str) -> Option<usize> {
        self.value(key).and_then(|v| v.as_arr()).map(|a| a.len())
    }

    /// A metadata array as u32 elements, or None if the key is absent / not an array.
    pub fn opt_arr_u32(&self, key: &str) -> Option<Vec<u32>> {
        self.value(key)
            .and_then(|v| v.as_arr())
            .map(|a| a.iter().filter_map(|e| e.as_u32()).collect())
    }

    pub fn arch(&self) -> Result<&str> {
        self.str("general.architecture")
    }

    /// `general.file_type` rendered the way llama.cpp names it (15 == Q4_K_M). Diagnostics only.
    pub fn file_type_name(&self) -> String {
        const NAMES: &[(u32, &str)] = &[
            (1, "F32"), (2, "F16"), (3, "BF16"), (7, "Q4_0"), (8, "Q4_1"), (9, "Q5_0"),
            (10, "Q5_1"), (11, "Q8_0"), (12, "Q2_K"), (13, "Q3_K"), (14, "Q4_K_S"),
            (15, "Q4_K_M"), (16, "Q5_K_S"), (17, "Q5_K_M"), (18, "Q6_K"), (19, "Q8_0"),
        ];
        match self.opt_u32("general.file_type") {
            Some(n) => NAMES
                .iter()
                .find(|(k, _)| *k == n)
                .map(|(_, s)| (*s).to_string())
                .unwrap_or_else(|| format!("file_type#{n}")),
            None => "unset".to_string(),
        }
    }

    // ---- tensors ----

    pub fn tensors(&self) -> &[TensorInfo] {
        &self.tensors
    }

    pub fn tensor(&self, name: &str) -> Result<&TensorInfo> {
        let i = *self.tensor_index.get(name).ok_or_else(|| {
            Error::msg(format!("no such tensor: {name} ({} tensors in file)", self.tensors.len()))
        })?;
        Ok(&self.tensors[i])
    }

    pub fn has_tensor(&self, name: &str) -> bool {
        self.tensor_index.contains_key(name)
    }

    /// Raw bytes of a tensor inside the mapping.
    pub fn tensor_data(&self, t: &TensorInfo) -> &[u8] {
        let start = self.data_start as usize + t.offset as usize;
        &self.map.as_bytes()[start..start + t.size as usize]
    }

    pub fn tensor_f32s<'a>(&'a self, t: &TensorInfo) -> Result<&'a [f32]> {
        if t.ty != ty::F32 {
            return Err(Error::msg(format!("tensor {}: expected f32, got {}", t.name, qtypes::name(t.ty))));
        }
        let b = self.tensor_data(t);
        if b.len() % 4 != 0 {
            return Err(Error::msg(format!("tensor {}: not f32 aligned", t.name)));
        }
        if (b.as_ptr() as usize) % std::mem::align_of::<f32>() != 0 {
            return Err(Error::msg(format!("tensor {}: data not 4-byte aligned", t.name)));
        }
        // SAFETY: `b` comes from a mapping whose start is aligned to `general.alignment` (a multiple
        // of 32, checked at open time) and every tensor offset is aligned, so `b.as_ptr()` is
        // 4-byte aligned; the length is a multiple of 4; f32 has no invalid bit patterns.
        Ok(unsafe { std::slice::from_raw_parts(b.as_ptr() as *const f32, b.len() / 4) })
    }

    /// Total bytes of all tensor data (sum of exact sizes, not file size).
    pub fn stored_weight_bytes(&self) -> u64 {
        self.tensors.iter().map(|t| t.size).sum()
    }

    pub fn n_tensors(&self) -> usize {
        self.tensors.len()
    }
}

struct ParsedHeader {
    version: u32,
    kvs: Vec<(String, Value)>,
    tensors: Vec<TensorInfo>,
    header_end: usize,
}

#[derive(Debug)]
enum HeaderError {
    Short,
    Msg(String),
}

type HResult<T> = std::result::Result<T, HeaderError>;

fn parse_header(buf: &[u8]) -> HResult<ParsedHeader> {
    let mut c = Cursor::new(buf);
    if c.take(4)? != GGUF_MAGIC {
        return Err(HeaderError::Msg("not a GGUF file (bad magic)".into()));
    }
    let version = c.u32()?;
    if version != GGUF_V2 && version != GGUF_V3 {
        return Err(HeaderError::Msg(format!("unsupported GGUF version {version} (need v2 or v3)")));
    }
    let n_tensors = c.u64()?;
    let n_kv = c.u64()?;
    if n_tensors > 1_000_000 || n_kv > 10_000 {
        return Err(HeaderError::Msg(format!("implausible counts: {n_tensors} tensors, {n_kv} kv")));
    }
    let mut kvs = Vec::with_capacity(n_kv as usize);
    for _ in 0..n_kv {
        let k = c.string()?;
        let t = c.u32()?;
        let v = c.value(t, 0)?;
        kvs.push((k, v));
    }
    let mut tensors = Vec::with_capacity(n_tensors as usize);
    for _ in 0..n_tensors {
        let name = c.string()?;
        let n_dims = c.u32()?;
        if !(1..=4).contains(&n_dims) {
            return Err(HeaderError::Msg(format!("tensor {name}: {n_dims} dimensions (1..=4 expected)")));
        }
        let mut ne = [1u64; 4];
        for e in ne.iter_mut().take(n_dims as usize) {
            *e = c.u64()?;
        }
        let ty = c.u32()?;
        let offset = c.u64()?;
        let size = qtypes::bytes_for_n(qtypes::n_elements(&ne), ty).unwrap_or(0);
        tensors.push(TensorInfo { name, n_dims, ne, ty, offset, size, stride: 0 });
    }
    Ok(ParsedHeader { version, kvs, tensors, header_end: c.pos })
}

struct Cursor<'a> {
    b: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    fn new(b: &'a [u8]) -> Self {
        Cursor { b, pos: 0 }
    }
    fn take(&mut self, n: usize) -> HResult<&'a [u8]> {
        if self.pos + n > self.b.len() {
            return Err(HeaderError::Short);
        }
        let s = &self.b[self.pos..self.pos + n];
        self.pos += n;
        Ok(s)
    }
    fn u16(&mut self) -> HResult<u16> {
        let b = self.take(2)?;
        Ok(u16::from_le_bytes([b[0], b[1]]))
    }
    fn u32(&mut self) -> HResult<u32> {
        let b = self.take(4)?;
        Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }
    fn u64(&mut self) -> HResult<u64> {
        let b = self.take(8)?;
        let mut a = [0u8; 8];
        a.copy_from_slice(b);
        Ok(u64::from_le_bytes(a))
    }
    fn string(&mut self) -> HResult<String> {
        let n = self.u64()?;
        if n > (1 << 28) {
            return Err(HeaderError::Msg(format!("implausible string length {n}")));
        }
        let b = self.take(n as usize)?;
        Ok(String::from_utf8_lossy(b).into_owned())
    }
    fn value(&mut self, t: u32, depth: u32) -> HResult<Value> {
        if depth > 2 {
            return Err(HeaderError::Msg("nested arrays deeper than 2".into()));
        }
        Value::tag(t).ok_or(HeaderError::Msg(format!("unknown GGUF value type {t}")))?;
        Ok(match t {
            0 => Value::U8(self.take(1)?[0]),
            1 => Value::I8(self.take(1)?[0] as i8),
            2 => Value::U16(self.u16()?),
            3 => Value::I16(self.u16()? as i16),
            4 => Value::U32(self.u32()?),
            5 => Value::I32(self.u32()? as i32),
            6 => Value::F32(f32::from_bits(self.u32()?)),
            7 => Value::Bool(self.take(1)?[0] != 0),
            8 => Value::Str(self.string()?),
            9 => {
                let et = self.u32()?;
                let n = self.u64()?;
                if n > (1 << 27) {
                    return Err(HeaderError::Msg(format!("implausible array length {n}")));
                }
                let mut v = Vec::with_capacity(n.min(1 << 20) as usize);
                for _ in 0..n {
                    v.push(self.value(et, depth + 1)?);
                }
                Value::Arr(v)
            }
            10 => Value::U64(self.u64()?),
            11 => Value::I64(self.u64()? as i64),
            12 => Value::F64(f64::from_bits(self.u64()?)),
            // Unreachable: `Value::tag(t)` rejected unknown types above.
            t => unreachable!("value type {t} passed the tag check"),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn kv_str(b: &mut Vec<u8>, key: &str, val: &str) {
        b.extend_from_slice(&(key.len() as u64).to_le_bytes());
        b.extend_from_slice(key.as_bytes());
        b.extend_from_slice(&8u32.to_le_bytes()); // STRING
        b.extend_from_slice(&(val.len() as u64).to_le_bytes());
        b.extend_from_slice(val.as_bytes());
    }

    #[test]
    fn header_parse_matches_type_implied_sizes_on_synthetic_file() {
        // A tiny hand-built GGUF: two KV pairs and one f32 tensor, then padded data.
        let mut b: Vec<u8> = Vec::new();
        b.extend_from_slice(b"GGUF");
        b.extend_from_slice(&3u32.to_le_bytes());
        b.extend_from_slice(&1u64.to_le_bytes()); // n_tensors
        b.extend_from_slice(&2u64.to_le_bytes()); // n_kv
        // "general.architecture" = "test"  (strings are u64 length + bytes)
        kv_str(&mut b, "general.architecture", "test");
        // "n" = 3
        b.extend_from_slice(&1u64.to_le_bytes());
        b.extend_from_slice(b"n");
        b.extend_from_slice(&4u32.to_le_bytes()); // U32
        b.extend_from_slice(&3u32.to_le_bytes());
        // tensor "w" f32 [4,1] - two dims, the shape every weight matrix in the checkpoint has
        b.extend_from_slice(&1u64.to_le_bytes());
        b.extend_from_slice(b"w");
        b.extend_from_slice(&2u32.to_le_bytes());
        b.extend_from_slice(&4u64.to_le_bytes());
        b.extend_from_slice(&1u64.to_le_bytes());
        b.extend_from_slice(&0u32.to_le_bytes()); // f32
        b.extend_from_slice(&0u64.to_le_bytes()); // offset
        while b.len() % 32 != 0 {
            b.push(0);
        }
        let start = b.len();
        for i in 0..4u32 {
            b.extend_from_slice(&(i as f32).to_le_bytes());
        }
        let p = parse_header(&b).unwrap();
        assert_eq!(p.version, 3);
        assert_eq!(p.kvs.len(), 2);
        assert_eq!(p.kvs[1].1, Value::U32(3));
        assert_eq!(p.tensors[0].size, 16);
        assert_eq!(p.tensors[0].as_mat(), Some((4, 1)));
        // The data section starts at the aligned-up end of the tensor-info section.
        assert_eq!(p.header_end, 126);
        assert_eq!(start, p.header_end.div_ceil(32) * 32);
    }

    #[test]
    fn truncated_header_is_reported_as_short() {
        let mut b: Vec<u8> = Vec::new();
        b.extend_from_slice(b"GGUF");
        b.extend_from_slice(&3u32.to_le_bytes());
        b.extend_from_slice(&1u64.to_le_bytes());
        b.extend_from_slice(&1u64.to_le_bytes());
        b.extend_from_slice(&4u64.to_le_bytes());
        b.extend_from_slice(b"name");
        assert!(matches!(parse_header(&b), Err(HeaderError::Short)));
    }
}
