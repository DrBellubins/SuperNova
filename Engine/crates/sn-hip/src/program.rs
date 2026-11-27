//! The static program: a flat list of baked launches, replayed either one by one (debug) or as one
//! captured HIP graph (the P1 design). No scheduler, no allocation, no lookup per token.

use std::ffi::c_void;

use sn_core::error::{Error, Result};

use crate::hip::{check, Device, HipEvent, HipFunction, HipGraph, HipGraphExec, HipStream, MEMCPY_HOST_TO_DEVICE};
use crate::kernels::Kernel;

/// A kernel argument, stored in an 8-byte slot. On a little-endian host a `u32`/`i32`/`f32` argument
/// is read back correctly from the low bytes of its slot, which is what `hipModuleLaunchKernel`'s
/// `kernelParams` expects (a pointer *to the value*).
#[derive(Clone, Copy, Debug)]
pub enum Arg {
    Ptr(u64),
    U32(u32),
    I32(i32),
    F32(f32),
    U64(u64),
}

impl Arg {
    fn bits(&self) -> u64 {
        match self {
            Arg::Ptr(v) | Arg::U64(v) => *v,
            Arg::U32(v) => *v as u64,
            Arg::I32(v) => *v as u32 as u64,
            Arg::F32(v) => v.to_bits() as u64,
        }
    }
}

const _: () = assert!(cfg!(target_endian = "little"), "kernel arguments are packed little-endian");

/// The argument block of one launch: a fixed buffer of slots plus the pointer table HIP walks.
/// `vals` is allocated once, when the program is compiled, and never mutated afterwards, so the
/// pointers stay valid for the lifetime of the program (and of any graph captured from it).
#[derive(Clone, Debug)]
pub struct Argv {
    vals: Vec<u64>,
}

impl Argv {
    pub fn new(args: &[Arg]) -> Argv {
        Argv { vals: args.iter().map(|a| a.bits()).collect() }
    }
    /// The `kernelParams` array: one pointer per argument, into the frozen `vals` buffer.
    fn params(&self) -> Vec<*mut c_void> {
        self.vals.iter().map(|v| v as *const u64 as *mut c_void).collect()
    }
    fn describe(&self, args: &[Arg]) -> String {
        args.iter()
            .map(|a| match a {
                Arg::Ptr(v) | Arg::U64(v) => format!("0x{v:x}"),
                Arg::U32(v) => v.to_string(),
                Arg::I32(v) => v.to_string(),
                Arg::F32(v) => format!("{v}"),
            })
            .collect::<Vec<_>>()
            .join(",")
    }
}

/// One node of the program.
#[derive(Clone, Debug)]
pub struct Op {
    /// Human-readable identity for `SN_TRACE=ops` (`rmsnorm blk.7`, `gemv ffn_down blk.12` ...).
    pub label: String,
    pub kind: OpKind,
}

#[derive(Clone, Debug)]
pub enum OpKind {
    Kernel {
        func: HipFunction,
        symbol: String,
        grid: [u32; 3],
        block: [u32; 3],
        shared: u32,
        argv: Argv,
        args: Vec<Arg>,
    },
    /// Host-to-device from **pinned** memory only: that is what a graph can capture, and the pinned
    /// slot is where the next token id / position is written (design-plan P1).
    H2DPinned { dst: u64, host: u64, len: usize },
}

impl Op {
    pub fn describe(&self) -> String {
        match &self.kind {
            OpKind::Kernel { symbol, grid, block, shared, args, .. } => format!(
                "kernel {symbol} grid={}x{}x{} block={}x{}x{} shared={shared} args={}",
                grid[0],
                grid[1],
                grid[2],
                block[0],
                block[1],
                block[2],
                Argv::new(args).describe(args)
            ),
            OpKind::H2DPinned { dst, host, len } => {
                format!("h2d_pinned dst=0x{dst:x} host=0x{host:x} len={len}")
            }
        }
    }
}

/// Accumulates ops. Compile-time only: after `finish` the program is inert data.
pub struct ProgramBuilder<'a> {
    dev: &'a Device,
    ops: Vec<Op>,
}

impl<'a> ProgramBuilder<'a> {
    pub fn new(dev: &'a Device) -> ProgramBuilder<'a> {
        ProgramBuilder { dev, ops: Vec::new() }
    }

    /// Append one kernel launch with every pointer already resolved.
    pub fn kernel(
        &mut self,
        label: &str,
        k: &Kernel,
        grid: [u32; 3],
        shared: u32,
        args: &[Arg],
    ) -> Result<&mut ProgramBuilder<'a>> {
        if grid.contains(&0) {
            return Err(Error::msg(format!("{label}: grid {grid:?} has a zero dimension")));
        }
        if k.block.contains(&0) {
            return Err(Error::msg(format!("{label}: block {k:?} has a zero dimension")));
        }
        self.ops.push(Op {
            label: label.to_string(),
            kind: OpKind::Kernel {
                func: k.func,
                symbol: label.to_string(),
                grid,
                block: k.block,
                shared,
                argv: Argv::new(args),
                args: args.to_vec(),
            },
        });
        Ok(self)
    }

    pub fn h2d_pinned(&mut self, label: &str, dst: u64, host: u64, len: usize) -> &mut ProgramBuilder<'a> {
        self.ops.push(Op {
            label: label.to_string(),
            kind: OpKind::H2DPinned { dst, host, len },
        });
        self
    }

    pub fn len(&self) -> usize {
        self.ops.len()
    }

    pub fn is_empty(&self) -> bool {
        self.ops.is_empty()
    }

    pub fn finish(self) -> Program {
        Program { ops: self.ops, device: format!("device {}", self.dev.info.index) }
    }
}

/// A compiled program: the ops, in order, with baked arguments.
#[derive(Clone, Debug)]
pub struct Program {
    ops: Vec<Op>,
    device: String,
}

impl Program {
    pub fn ops(&self) -> &[Op] {
        &self.ops
    }

    pub fn len(&self) -> usize {
        self.ops.len()
    }

    pub fn is_empty(&self) -> bool {
        self.ops.is_empty()
    }

    /// Launch every op in order on `stream`, no graph. This is the `--no-graph` debug path and the
    /// baseline the graph gate is measured against.
    pub fn launch(&self, dev: &Device, stream: HipStream) -> Result<()> {
        self.launch_range(dev, stream, 0, self.ops.len())
    }

    /// Launch ops `from..to` in order. The debug path for "which part of the program is wrong (or
    /// slow)": `sn gpu stages` stops after each execution and reads the arena back, and a range
    /// timed with events gives a per-section cost without a profiler.
    pub fn launch_range(&self, dev: &Device, stream: HipStream, from: usize, to: usize) -> Result<()> {
        let a = from.min(self.ops.len());
        let b = to.min(self.ops.len()).max(a);
        for op in self.ops[a..b].iter() {
            match &op.kind {
                OpKind::Kernel { func, grid, block, shared, argv, .. } => {
                    let mut params = argv.params();
                    let f = dev.api.hipModuleLaunchKernel.unwrap();
                    let e = {
                        f(
                            *func,
                            grid[0],
                            grid[1],
                            grid[2],
                            block[0],
                            block[1],
                            block[2],
                            *shared,
                            stream,
                            params.as_mut_ptr(),
                            std::ptr::null_mut(),
                        )
                    };
                    check(&format!("hipModuleLaunchKernel({})", op.label), e, &dev.api)?;
                }
                OpKind::H2DPinned { dst, host, len } => {
                    check(
                        &format!("hipMemcpyAsync({})", op.label),
                        dev.api.hipMemcpyAsync.unwrap()(
                            *dst as *mut c_void,
                            *host as *const c_void,
                            *len,
                            MEMCPY_HOST_TO_DEVICE,
                            stream,
                        ),
                        &dev.api,
                    )?;
                }
            }
        }
        Ok(())
    }

    /// `SN_TRACE=ops`: the whole program, one line per op.
    pub fn report(&self) -> String {
        let mut s = format!("program on {}: {} ops\n", self.device, self.ops.len());
        for (i, op) in self.ops.iter().enumerate() {
            s += &format!("  {i:>4} {:<24} {}\n", op.label, op.describe());
        }
        s
    }
}

/// A captured + instantiated graph. One `hipGraphLaunch` replays the whole program.
pub struct Graph {
    exec: HipGraphExec,
    ops: usize,
}

impl Graph {
    /// Stream-capture the program and instantiate it. The program's arguments are baked, so nothing
    /// here needs updating between replays; inputs arrive through pinned memory (`Op::H2DPinned`).
    pub fn capture(dev: &Device, stream: HipStream, program: &Program) -> Result<Graph> {
        let api = &dev.api;
        if program.is_empty() {
            return Err(Error::msg("cannot capture an empty program"));
        }
        check("hipStreamBeginCapture", api.hipStreamBeginCapture.unwrap()(stream, crate::hip::CAPTURE_MODE_THREAD_LOCAL), api)?;
        let started = program.launch(dev, stream);
        // EndCapture must run even if a node failed to enqueue, or the stream stays in capture mode.
        let mut graph: HipGraph = std::ptr::null_mut();
        let ended = check("hipStreamEndCapture", api.hipStreamEndCapture.unwrap()(stream, &mut graph), api);
        started?;
        ended?;
        if graph.is_null() {
            return Err(Error::msg("hipStreamEndCapture returned a null graph"));
        }
        let mut exec: HipGraphExec = std::ptr::null_mut();
        let r = check(
            "hipGraphInstantiateWithFlags",
            api.hipGraphInstantiateWithFlags.unwrap()(&mut exec, graph, 0),
            api,
        );
        check("hipGraphDestroy", api.hipGraphDestroy.unwrap()(graph), api).ok();
        r?;
        if let Some(up) = api.hipGraphUpload {
            // Push the exec graph to the device once so the first replay does not pay for it.
            check("hipGraphUpload", up(exec, stream), api).ok();
        }
        Ok(Graph { exec, ops: program.len() })
    }

    pub fn ops(&self) -> usize {
        self.ops
    }

    /// The single launch a decode step makes.
    pub fn replay(&self, dev: &Device, stream: HipStream) -> Result<()> {
        check("hipGraphLaunch", dev.api.hipGraphLaunch.unwrap()(self.exec, stream), &dev.api)
    }
}

impl Drop for Graph {
    fn drop(&mut self) {
        // The Device outlives every graph in this crate's API; a leak here would be a bug in the
        // caller, and hipGraphExecDestroy cannot fail usefully in a Drop.
        let _ = self.exec;
    }
}

/// Two `hipEvent`s around a launch sequence: how `SN_TRACE=time` and the Phase 3 gate measure
/// device time (host wall time around a launch is not a device measurement).
pub struct Timer {
    start: HipEvent,
    stop: HipEvent,
}

impl Timer {
    pub fn new(dev: &Device) -> Result<Timer> {
        let (mut a, mut b) = (std::ptr::null_mut(), std::ptr::null_mut());
        check("hipEventCreate", dev.api.hipEventCreate.unwrap()(&mut a), &dev.api)?;
        if let Err(e) = check("hipEventCreate", dev.api.hipEventCreate.unwrap()(&mut b), &dev.api) {
            check("hipEventDestroy", dev.api.hipEventDestroy.unwrap()(a), &dev.api).ok();
            return Err(e);
        }
        Ok(Timer { start: a, stop: b })
    }

    pub fn begin(&self, dev: &Device, stream: HipStream) -> Result<()> {
        check("hipEventRecord", dev.api.hipEventRecord.unwrap()(self.start, stream), &dev.api)
    }

    /// Milliseconds of device time between `begin` and here.
    pub fn end_ms(&self, dev: &Device, stream: HipStream) -> Result<f64> {
        check("hipEventRecord", dev.api.hipEventRecord.unwrap()(self.stop, stream), &dev.api)?;
        check("hipEventSynchronize", dev.api.hipEventSynchronize.unwrap()(self.stop), &dev.api)?;
        let mut ms = 0f32;
        check("hipEventElapsedTime", dev.api.hipEventElapsedTime.unwrap()(&mut ms, self.start, self.stop), &dev.api)?;
        Ok(ms as f64)
    }
}

impl Drop for Timer {
    fn drop(&mut self) {
        let _ = (self.start, self.stop);
    }
}

/// `SN_TRACE` levels. `ops` prints the compiled program; `time` wraps each op in events (which
/// serialises the stream, so it is a debug tool, not a measurement to gate on).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Trace {
    Off,
    Ops,
    Time,
}

pub fn trace_from_env() -> Trace {
    match std::env::var("SN_TRACE").ok().as_deref() {
        Some("ops") | Some("1") => Trace::Ops,
        Some("time") => Trace::Time,
        _ => Trace::Off,
    }
}

/// Time one program on `stream`, optionally per op, and return the total milliseconds.
pub fn timed_launch(dev: &Device, stream: HipStream, program: &Program, trace: Trace) -> Result<f64> {
    match trace {
        Trace::Ops => {
            print!("{}", program.report());
            program.launch(dev, stream)?;
            check("hipStreamSynchronize", dev.api.hipStreamSynchronize.unwrap()(stream), &dev.api)?;
            Ok(0.0)
        }
        Trace::Time => {
            let mut total = 0f64;
            for op in program.ops() {
                let t = Timer::new(dev)?;
                t.begin(dev, stream)?;
                // One op at a time: build a single-op program view by launching it directly.
                launch_one(dev, stream, op)?;
                let ms = t.end_ms(dev, stream)?;
                total += ms;
                println!("{:>10.4} ms  {}", ms, op.label);
            }
            Ok(total)
        }
        Trace::Off => {
            let t = Timer::new(dev)?;
            t.begin(dev, stream)?;
            program.launch(dev, stream)?;
            let ms = t.end_ms(dev, stream)?;
            check("hipStreamSynchronize", dev.api.hipStreamSynchronize.unwrap()(stream), &dev.api)?;
            Ok(ms)
        }
    }
}

fn launch_one(dev: &Device, stream: HipStream, op: &Op) -> Result<()> {
    let single = Program { ops: vec![op.clone()], device: String::new() };
    single.launch(dev, stream)
}
