// SuperNova's Phase 3 kernel image: the two kernels the graph gate is measured with, plus the
// input-staging kernel that proves the pinned-host -> device path works inside a captured graph.
//
// Compiled offline (Tools/build-kernels.sh) into a fat binary and loaded with `hipModuleLoad`; the
// Rust build never invokes a HIP toolchain. Everything is `extern "C"` because the Rust side looks
// these up by symbol name.
//
// The chain kernels are deliberately *dependent* global read-modify-writes: `volatile` stops the
// compiler folding the loop, so `N launches` and `1 launch with an N-step loop` do exactly the same
// device work and the difference between them is dispatch cost - which is what the Phase 3 gate
// measures (design-plan §7: ">= 90 % of the theoretical launch-free time").

#include <hip/hip_runtime.h>

/// One step of the chain: one global load, one FMA, one global store.
extern "C" __global__ void sn_chain_step(float* p) {
    volatile float* v = p;
    *v = *v * 1.0000001f + 1.0f;
}

/// The launch-free equivalent: the same `n` steps inside a single launch.
extern "C" __global__ void sn_chain_n(float* p, int n) {
    volatile float* v = p;
    for (int i = 0; i < n; i++) {
        *v = *v * 1.0000001f + 1.0f;
    }
}

/// Read the staged token id / position (copied from pinned host memory by the graph's first node)
/// and write what a decode step would derive from it. Used to prove that a captured graph observes
/// new inputs with no graph update.
extern "C" __global__ void sn_input_stage(const unsigned int* in, unsigned long long* out) {
    volatile const unsigned int* src = in;
    volatile unsigned long long* dst = out;
    *dst = ((unsigned long long)src[1] << 32) | (unsigned long long)src[0];
}
