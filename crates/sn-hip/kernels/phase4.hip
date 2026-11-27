// SuperNova's Phase 4 kernel image: the elementwise set plus the mHC kernels (seed, route, merge,
// expose).
//
// This is where design-plan §4's flaw dies. The fork runs one mHC route as ~490 graph nodes
// (`build_route()` expands each of the four Sinkhorn pair updates into six `ggml` tensors, four
// updates per sweep, 20 sweeps, on `[1,T]` views that stride 8 f32 through the 8-wide logits
// buffer), twice per mHC execution, 48 executions per token: 46,080 nodes whose entire useful
// output is 4 + 2 + 8 + 4096 floats. Here it is one workgroup.
//
// Compiled offline by Tools/build-kernels.sh into a fat binary and loaded with `hipModuleLoad`;
// the Rust build never invokes a HIP toolchain. Every entry point is `extern "C"` because the Rust
// side resolves symbols by name, and every scalar parameter is baked into the program at compile
// time (P1), which is why none of these kernels take a struct.
//
// The `sn_ub_*` kernels at the bottom are not engine code: they are the *unfused baseline* the
// Phase 4 micro-benchmark has to beat, spelled the way the fork spells it, so the >= 100x claim is
// measured against real nodes instead of an invented number.
//
// Precision contract (design-plan P3, Docs/numerics.md): accumulate in f32; an f16 operand means a
// value that really lives in an f16 buffer, and an f16 store is round-to-nearest-even. Each kernel
// has a scalar oracle in `sn-core/src/oracle.rs`, and the tests compare them.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// Workgroup size every kernel here is registered with. 256 = 8 wavefronts of 32 on gfx1102: wide
// enough to cover the 4096-wide streams in 16 elements per thread, narrow enough that the shared
// memory the reductions use stays small.
#define SN_BLOCK 256

/// Threads in one `sn_mhc_route_f32` workgroup. The route is the only kernel here that is
/// memory-latency bound rather than trivially wide: one workgroup per token has to read the whole
/// `8 x 8192` f32 router (256 KB) plus the `2 x 4096` streams, and at 256 threads that read ran at
/// ~8 GB/s because a single workgroup cannot keep enough loads in flight. 1024 measured 2.6x faster
/// on gfx1102; the shared-memory scratch scales with it (32 KB, under the 48 KB default limit and
/// the device's 64 KB per block).
///
/// `ops::ROUTE_BLOCK` in Rust must equal this: the scratch is indexed by the *compiled* stride while
/// the launcher picks the *runtime* workgroup size, and a mismatch reads outside `red` (it produces
/// wrong router logits, not an error). `Tools/build-kernels.sh` records this value in
/// `target/hip/phase4.meta` and the check harness refuses to run if the two disagree.
#define SN_ROUTE_THREADS 1024

// ---------------------------------------------------------------------------------------------
// scalar helpers
// ---------------------------------------------------------------------------------------------

/// `softplus` with the reference's switch at exactly 20.0f (§2.1). `log1pf`, not `logf(1+x)`:
/// for x < -20 the naive form loses everything below the mantissa.
__device__ __forceinline__ float sn_softplus(float x) {
    return x > 20.0f ? x : log1pf(expf(x));
}

__device__ __forceinline__ float sn_sigmoid(float x) { return 1.0f / (1.0f + expf(-x)); }

__device__ __forceinline__ float sn_silu(float x) { return x * sn_sigmoid(x); }

/// Read a buffer element as f32, and write an f32 result into a buffer of `T`.
///
/// These are function *templates with explicit specialisations*, not overloads by return type: an
/// overload set (`sn_h(float) -> __half`) silently inserted an f16 round-trip into every f32 path,
/// which is exactly the kind of defect an NMSE gate exists to catch (it showed up as 4e-8 instead of
/// 1e-14 on `rmsnorm_f32`, and as `inf` on `silu_mul_f32` where the product exceeds 65504).
template <typename T>
__device__ __forceinline__ float sn_from(T x);
template <>
__device__ __forceinline__ float sn_from<float>(float x) { return x; }
template <>
__device__ __forceinline__ float sn_from<__half>(__half x) { return __half2float(x); }

template <typename T>
__device__ __forceinline__ T sn_to(float x);
template <>
__device__ __forceinline__ float sn_to<float>(float x) { return x; }
template <>
__device__ __forceinline__ __half sn_to<__half>(float x) { return __float2half_rn(x); }

/// Block reduction of one f32: intra-wavefront shuffle, then the wavefront partials in shared
/// memory. `part` must hold ceil(blockDim.x / 32) floats.
__device__ __forceinline__ float sn_block_sum(float v, float* part) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        v += __shfl_down_sync(0xffffffffffffffffull, v, off);
    }
    const int lane = threadIdx.x & 31;
    const int wave = threadIdx.x >> 5;
    if (lane == 0) part[wave] = v;
    __syncthreads();
    const int nwave = ((int)blockDim.x + 31) >> 5;
    float s = 0.0f;
    for (int i = 0; i < nwave; i++) {
        s += part[i];
    }
    return s;
}

/// Reduce eight per-thread accumulators to eight block-wide sums, left in `out8[0..8]`.
///
/// This is the shape the routing needs: one workgroup split-K'd over 8192 router elements, and the
/// eight partial dots have to meet. Layout is [row][thread] so both the scatter and the fold read
/// one bank per thread per access.
/// `stride` is the workgroup size the scratch was sized for: the routing kernel runs with
/// `SN_ROUTE_THREADS` threads, everything else with `SN_BLOCK`.
__device__ __forceinline__ void sn_block_reduce8(const float* acc, float* sm, float* out8, int stride) {
    for (int r = 0; r < 8; r++) {
        sm[r * stride + threadIdx.x] = acc[r];
    }
    __syncthreads();
    for (int s = (int)blockDim.x >> 1; s >= 32; s >>= 1) {
        if (threadIdx.x < s) {
#pragma unroll
            for (int r = 0; r < 8; r++) {
                sm[r * stride + threadIdx.x] += sm[r * stride + threadIdx.x + s];
            }
        }
        __syncthreads();
    }
    if (threadIdx.x < 32) {
#pragma unroll
        for (int r = 0; r < 8; r++) {
            float v = sm[r * stride + threadIdx.x];
#pragma unroll
            for (int off = 16; off > 0; off >>= 1) {
                v += __shfl_down_sync(0xffffffffffffffffull, v, off);
            }
            if (threadIdx.x == 0) out8[r] = v;
        }
    }
    __syncthreads();
}

// ---------------------------------------------------------------------------------------------
// rmsnorm: y = x * rsqrtf(mean(x^2) + eps) [* w], last dimension, one workgroup per row
// ---------------------------------------------------------------------------------------------

template <typename Tin, typename Tout>
__device__ __forceinline__ void sn_rmsnorm_body(const Tin* x, const float* w, Tout* y, int cols, float eps, int weighted) {
    __shared__ float part[SN_BLOCK / 32];
    const size_t base = (size_t)blockIdx.x * (size_t)cols;
    float ss = 0.0f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        const float v = sn_from<Tin>(x[base + i]);
        ss += v * v;
    }
    const float scale = rsqrtf(sn_block_sum(ss, part) / (float)cols + eps);
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        // (x * scale) * w, the order the oracle uses. The checkpoint's norms have no (1+w).
        float v = sn_from<Tin>(x[base + i]) * scale;
        if (weighted) v *= w[i];
        y[base + i] = sn_to<Tout>(v);
    }
}

extern "C" __global__ void sn_rmsnorm_f16(const __half* x, const float* w, __half* y, int cols, float eps, int weighted) {
    sn_rmsnorm_body<__half, __half>(x, w, y, cols, eps, weighted);
}

extern "C" __global__ void sn_rmsnorm_f32(const float* x, const float* w, float* y, int cols, float eps, int weighted) {
    sn_rmsnorm_body<float, float>(x, w, y, cols, eps, weighted);
}

// ---------------------------------------------------------------------------------------------
// elementwise: add / silu_mul / swiglu / sigmoid
// ---------------------------------------------------------------------------------------------

template <typename T>
__device__ __forceinline__ void sn_add_body(T* a, const T* b, long n) {
    const long stride = (long)gridDim.x * (long)blockDim.x;
    for (long i = (long)blockIdx.x * (long)blockDim.x + threadIdx.x; i < n; i += stride) {
        // accumulate in f32, store in the buffer's own type: the residual add is where an f16
        // activation meets an f32 projection, and the rounding must happen once, on the store.
        a[i] = sn_to<T>(sn_from<T>(a[i]) + sn_from<T>(b[i]));
    }
}

extern "C" __global__ void sn_add_f32(float* a, const float* b, long n) { sn_add_body<float>(a, b, n); }
extern "C" __global__ void sn_add_f16(__half* a, const __half* b, long n) { sn_add_body<__half>(a, b, n); }

/// The FFN intermediate, two projection outputs: out = silu(g) * u (§2.5 - SiLU on the gate only).
template <typename T>
__device__ __forceinline__ void sn_silu_mul_body(const T* g, const T* u, T* out, long n) {
    const long stride = (long)gridDim.x * (long)blockDim.x;
    for (long i = (long)blockIdx.x * (long)blockDim.x + threadIdx.x; i < n; i += stride) {
        out[i] = sn_to<T>(sn_silu(sn_from<T>(g[i])) * sn_from<T>(u[i]));
    }
}

extern "C" __global__ void sn_silu_mul_f32(const float* g, const float* u, float* out, long n) {
    sn_silu_mul_body<float>(g, u, out, n);
}
extern "C" __global__ void sn_silu_mul_f16(const __half* g, const __half* u, __half* out, long n) {
    sn_silu_mul_body<__half>(g, u, out, n);
}

/// The same op read from one buffer holding `gate` then `up`, i.e. what a fused `ffn_gate/up` GEMM
/// writes. Two loads and one store instead of three memory ops.
template <typename T>
__device__ __forceinline__ void sn_swiglu_body(const T* gu, T* out, long n) {
    const long stride = (long)gridDim.x * (long)blockDim.x;
    for (long i = (long)blockIdx.x * (long)blockDim.x + threadIdx.x; i < n; i += stride) {
        out[i] = sn_to<T>(sn_silu(sn_from<T>(gu[i])) * sn_from<T>(gu[n + i]));
    }
}

extern "C" __global__ void sn_swiglu_f32(const float* gu, float* out, long n) { sn_swiglu_body<float>(gu, out, n); }
extern "C" __global__ void sn_swiglu_f16(const __half* gu, __half* out, long n) { sn_swiglu_body<__half>(gu, out, n); }

/// Elementwise sigmoid (the GDN `beta`, and anything else that wants it as a node of its own).
extern "C" __global__ void sn_sigmoid_f32(const float* x, float* y, long n) {
    const long stride = (long)gridDim.x * (long)blockDim.x;
    for (long i = (long)blockIdx.x * (long)blockDim.x + threadIdx.x; i < n; i += stride) {
        y[i] = sn_sigmoid(x[i]);
    }
}

// ---------------------------------------------------------------------------------------------
// rope_inplace: partial rope over the first n_rot dims of head vectors, from a cos/sin table
// ---------------------------------------------------------------------------------------------

/// `grid = (n_heads, n_tokens)`, `blockDim.x = n_pairs`.
///
/// Only `n_rot` dimensions per head are touched, which is exactly why this kernel is in-place:
/// dimensions `n_rot..head_dim` of a 256-wide head pass through untouched, and the table (§2.6,
/// `n_ctx x 32` f32 pairs) is precomputed once per position instead of per head.
///
/// `pairing` selects the layout the *architecture* chose, not a knob: `0` is
/// `GGML_ROPE_TYPE_NORMAL`'s adjacent `(2p, 2p+1)`, `1` is the half-offset `(p, p + n_rot/2)` that
/// `sensenova_loop` uses. The two are identical at position 0, so nothing but a table built at
/// position > 0 will ever tell them apart (Docs/numerics.md §3.3).
template <typename T>
__device__ __forceinline__ void sn_rope_body(T* p, const float* ctab, const float* stab, const unsigned int* pos,
                                             int n_pairs, int n_rot, int head_stride, long token_stride, int pairing) {
    const int pr = threadIdx.x;
    if (pr >= n_pairs) return;
    const long t = blockIdx.y;
    // position of this token, from the device-side position array (pinned-written during decode).
    const unsigned int position = pos[t];
    const long tbl = (long)position * (long)n_pairs + pr;
    const float c = ctab[tbl];
    const float s = stab[tbl];
    T* row = p + (long)blockIdx.x * (long)head_stride + t * token_stride;
    int i0, i1;
    if (pairing == 0) {
        i0 = 2 * pr;
        i1 = 2 * pr + 1;
    } else {
        const int half = n_rot >> 1;
        i0 = pr;
        i1 = pr + half;
    }
    const float a = sn_from<T>(row[i0]);
    const float b = sn_from<T>(row[i1]);
    row[i0] = sn_to<T>(a * c - b * s);
    row[i1] = sn_to<T>(a * s + b * c);
}

extern "C" __global__ void sn_rope_inplace_f32(float* p, const float* ctab, const float* stab, const unsigned int* pos,
                                               int n_pairs, int n_rot, int head_stride, long token_stride, int pairing) {
    sn_rope_body<float>(p, ctab, stab, pos, n_pairs, n_rot, head_stride, token_stride, pairing);
}

extern "C" __global__ void sn_rope_inplace_f16(__half* p, const float* ctab, const float* stab, const unsigned int* pos,
                                               int n_pairs, int n_rot, int head_stride, long token_stride, int pairing) {
    sn_rope_body<__half>(p, ctab, stab, pos, n_pairs, n_rot, head_stride, token_stride, pairing);
}

// ---------------------------------------------------------------------------------------------
// attn_gate_norm: out = attn_out * sigmoid(gate), gate read strided out of the fused attn_q
// ---------------------------------------------------------------------------------------------

/// `grid = (ceil(n / SN_BLOCK), n_tokens)`, where `n = n_q_heads * d_head` is one token's packed
/// attention output. The gate lives inside the `[q | gate]` interleave of `attn_q`'s output, at
/// head stride `gstride` and offset `goff`, so nothing materialises it (§2.3).
template <typename Tout, typename TIn>
__device__ __forceinline__ void sn_gate_norm_body(const TIn* o, const float* qg, Tout* out, int d, int gstride, int goff,
                                                  long n, long so, long sg, long sout) {
    const long t = blockIdx.y;
    const long stride = (long)gridDim.x * (long)blockDim.x;
    for (long i = (long)blockIdx.x * (long)blockDim.x + threadIdx.x; i < n; i += stride) {
        const long h = i / d;
        const long c = i - h * d;
        const float g = qg[t * sg + h * gstride + goff + c];
        out[t * sout + i] = sn_to<Tout>(sn_from<TIn>(o[t * so + i]) * sn_sigmoid(g));
    }
}

extern "C" __global__ void sn_attn_gate_norm_f32(const float* o, const float* qg, float* out, int d, int gstride, int goff,
                                                 long n, long so, long sg, long sout) {
    sn_gate_norm_body<float, float>(o, qg, out, d, gstride, goff, n, so, sg, sout);
}

extern "C" __global__ void sn_attn_gate_norm_f16(const __half* o, const float* qg, __half* out, int d, int gstride, int goff,
                                                 long n, long so, long sg, long sout) {
    sn_gate_norm_body<__half, __half>(o, qg, out, d, gstride, goff, n, so, sg, sout);
}

// ---------------------------------------------------------------------------------------------
// mhc_route: the whole route - joint RMS, the 8 router dots, bias, 20 sweeps, exp - in one launch
// ---------------------------------------------------------------------------------------------

/// `grid.x` = token. One workgroup per token, `SN_BLOCK` threads.
///
/// What it replaces (per route): the concatenation's RMS, the `[8,8192] x [8192,1]` router matmul,
/// the bias add, four sigmoids, the 480 nodes of the balancing (`6 * 4 * 20`), the exponentiation,
/// and the weighted sum that forms `in` - ~490 nodes of which the useful output is 8 + 2 + 4 +
/// 4096 floats.
///
/// Structure:
///   1. sum of squares over the concatenated `2d` streams (f32, split across the workgroup);
///   2. the same pass also cannot start before the scale is known, so the 8 router dots are a
///      second pass over `a`,`b` with `f = v * scale` formed in a register: same products the
///      oracle forms from its normalised array, and `a`,`b` come back from L1/L2, not DRAM;
///   3. eight split-K partials reduced once in shared memory (out8[8]);
///   4. every thread then runs the scalar tail redundantly in registers - the 20 sweeps over four
///      log-mixes, the two `2*sigmoid` posts, the four `exp`s. It is ~160 softplus in series, which
///      is microseconds and buys the elimination of a second barrier plus the 480 nodes.
///
/// Requires `d % 4 == 0` (every stream in this engine is 4096 or 8192 wide).
///
/// `flags` bit 0 is `mhc.fixed_identity`: the mix is the identity and logits 4..7 (and the whole
/// balancing) are skipped, which is the other checkpoint shape §2.7 mentions.
///
/// Shared memory: `SN_BLOCK * 8 + 16` floats (~8 KB), so occupancy is not shared-memory limited.
#define SN_NROUTE 8

extern "C" __global__ void sn_mhc_route_f32(const float* a, const float* b, const float* w, const float* bias,
                                            float* y_in, float* logits, float* post, float* mix, int d, int n_route,
                                            long sa, long sb, long sin_, long slog, long spost, long smix, float eps,
                                            float pre_scale, int sweeps, int flags) {
    __shared__ float part[SN_ROUTE_THREADS / 32];
    __shared__ float red[SN_NROUTE * SN_ROUTE_THREADS];
    __shared__ float out8[SN_NROUTE];

    const long t = blockIdx.x;
    const float* va = a + t * sa;
    const float* vb = b + t * sb;
    const int d2 = 2 * d;

    // 1. one unweighted RMS over the concatenation, eps hard-coded at 1e-6 by the reference (§2.7).
    float ss = 0.0f;
    // float4: 16-byte loads are 4x fewer memory instructions and 4x fewer addresses, and every
    // stream in this engine is 256-byte aligned from the arena, so `d % 4 == 0` is the only
    // requirement (n_embd is 4096; the Rust launcher rejects anything else).
    const float4* va4 = reinterpret_cast<const float4*>(va);
    const float4* vb4 = reinterpret_cast<const float4*>(vb);
    const int q = (int)(blockDim.x);
    for (int k = threadIdx.x; k < d / 4; k += q) {
        const float4 x = va4[k];
        const float4 y = vb4[k];
        ss += x.x * x.x + x.y * x.y + x.z * x.z + x.w * x.w;
        ss += y.x * y.x + y.y * y.y + y.z * y.z + y.w * y.w;
    }
    const float scale = rsqrtf(sn_block_sum(ss, part) / (float)d2 + eps);

    // 2. split-K over the router. w is [n_route][2d] with k contiguous, so every load is coalesced;
    // eight accumulators per thread, no atomics and no global scratch.
    float acc[SN_NROUTE];
#pragma unroll
    for (int r = 0; r < SN_NROUTE; r++) acc[r] = 0.0f;
    // The eight row bases are computed once, and both halves of a row are read as float4: the
    // router is [n_route][2d] with k contiguous, so 16-byte loads are naturally coalesced.
    const float4* w4[SN_NROUTE];
#pragma unroll
    for (int r = 0; r < SN_NROUTE; r++) {
        w4[r] = reinterpret_cast<const float4*>(w + (long)r * (long)d2);
    }
    const int dq = d / 4;
    for (int k = threadIdx.x; k < dq; k += q) {
        float4 fa4 = va4[k];
        float4 fb4 = vb4[k];
        fa4.x *= scale; fa4.y *= scale; fa4.z *= scale; fa4.w *= scale;
        fb4.x *= scale; fb4.y *= scale; fb4.z *= scale; fb4.w *= scale;
#pragma unroll
        for (int r = 0; r < SN_NROUTE; r++) {
            if (r < n_route) {
                const float4 wa = w4[r][k];
                const float4 wb = w4[r][dq + k];
                acc[r] += wa.x * fa4.x + wa.y * fa4.y + wa.z * fa4.z + wa.w * fa4.w;
                acc[r] += wb.x * fb4.x + wb.y * fb4.y + wb.z * fb4.z + wb.w * fb4.w;
            }
        }
    }
    sn_block_reduce8(acc, red, out8, SN_ROUTE_THREADS);

    // 3. bias, then the scalar tail. Every thread computes it (registers, no barrier) so that the
    // sigmoids needed for `in` are already in place when the vector pass runs.
    float lg[SN_NROUTE];
#pragma unroll
    for (int r = 0; r < SN_NROUTE; r++) {
        lg[r] = (r < n_route) ? (out8[r] + bias[r]) : 0.0f;
    }
    if (logits != nullptr) {
        for (int r = threadIdx.x; r < n_route; r += blockDim.x) logits[t * slog + r] = lg[r];
    }

    const float s0 = sn_sigmoid(lg[0]);
    const float s1 = sn_sigmoid(lg[1]);
    const float p0 = 2.0f * sn_sigmoid(lg[2]);
    const float p1 = 2.0f * sn_sigmoid(lg[3]);

    float m[4];
    if (flags & 1) {
        m[0] = 1.0f;
        m[1] = 0.0f;
        m[2] = 0.0f;
        m[3] = 1.0f;
    } else {
        m[0] = lg[4];
        m[1] = lg[5];
        m[2] = lg[6];
        m[3] = lg[7];
        // Exactly `mhc.iterations` sweeps in the pair order (0,1),(2,3),(0,2),(1,3). The checkpoint
        // was trained on the 20-sweep iterate, which is only O(1/n) converged: iterating to the
        // fixed point is a *different model* (§2.7), so `sweeps` is data and this loop is verbatim.
        // The pair order is part of the model: (0,1),(2,3),(0,2),(1,3). It is not commutative, so a
        // wrong table is a different model and nothing else here would notice.
        const int pi[4] = {0, 2, 0, 1};
        const int pj[4] = {1, 3, 2, 3};
        // One pair update, spelled as a log-sum-exp instead of two softplus calls:
        //     x' = -softplus(-(x-y)) = x - lse(x,y),  y' = y - lse(x,y)
        // which is the same renormalisation with one exp and one log per pair instead of two of
        // each. `log1pf` is the reason: it has no GCN instruction and compiles to a software
        // fallback, and 80 of them in series dominated this kernel's runtime (measured: 64 us for
        // one route, most of it here). `logf(1 + exp(-|d|))` costs the same as the hardware
        // intrinsics and is accurate to |d| where it matters, i.e. nowhere that survives 20
        // sweeps. The reference's own softplus (used verbatim by the unfused baseline above) keeps
        // `log1pf`, because that is what it does.
        for (int it = 0; it < sweeps; it++) {
#pragma unroll
            for (int q = 0; q < 4; q++) {
                const int i = pi[q];
                const int j = pj[q];
                const float mx = fmaxf(m[i], m[j]);
                const float lse = mx + logf(1.0f + expf(-fabsf(m[i] - m[j])));
                m[i] = m[i] - lse;
                m[j] = m[j] - lse;
            }
        }
#pragma unroll
        for (int r = 0; r < 4; r++) m[r] = expf(m[r]);
    }
    if (post != nullptr && threadIdx.x < 2) post[t * spost + threadIdx.x] = (threadIdx.x == 0) ? p0 : p1;
    if (mix != nullptr && threadIdx.x < 4) mix[t * smix + threadIdx.x] = m[threadIdx.x];

    // 4. `in` = (a * sigmoid(l0) + b * sigmoid(l1)) * pre_scale, the input the branch sees.
    float* yi = y_in + t * sin_;
    float4* yi4 = reinterpret_cast<float4*>(yi);
    for (int k = threadIdx.x; k < d / 4; k += q) {
        float4 x = va4[k];
        float4 y = vb4[k];
        float4 o;
        o.x = (x.x * s0 + y.x * s1) * pre_scale;
        o.y = (x.y * s0 + y.y * s1) * pre_scale;
        o.z = (x.z * s0 + y.z * s1) * pre_scale;
        o.w = (x.w * s0 + y.w * s1) * pre_scale;
        yi4[k] = o;
    }
}

// ---------------------------------------------------------------------------------------------
// mhc_seed / mhc_expose: the two ends of the mHC region (§2.7)
//
// These are not decorations. The streams are *created* at the first mHC execution, from the hidden
// state that reaches it (`a = b = cur`), and the trunk *reads* them back as `0.5*(a+b)` at the last
// one - after which the exit blocks (physical 28..31 here) continue with plain residual adds. A
// program that seeds nothing reads streams nothing wrote, and a program that exposes only at the end
// of the trunk feeds the head the loop's streams instead of the exit blocks' output.
// ---------------------------------------------------------------------------------------------

/// `grid = (ceil(d / SN_BLOCK), n_tokens)`. Seed both streams from `cur`.
extern "C" __global__ void sn_mhc_seed_f32(float* a, float* b, const float* cur, int d, long sa, long sb, long scur) {
    const long t = blockIdx.y;
    const int k = blockIdx.x * SN_BLOCK + threadIdx.x;
    if (k >= d) return;
    const float v = cur[t * scur + k];
    a[t * sa + k] = v;
    b[t * sb + k] = v;
}

/// `grid = (ceil(d / SN_BLOCK), n_tokens)`. The value the trunk exposes while inside the mHC range.
extern "C" __global__ void sn_mhc_expose_f32(const float* a, const float* b, float* cur, int d, long sa, long sb, long scur) {
    const long t = blockIdx.y;
    const int k = blockIdx.x * SN_BLOCK + threadIdx.x;
    if (k >= d) return;
    cur[t * scur + k] = (a[t * sa + k] + b[t * sb + k]) * 0.5f;
}

// ---------------------------------------------------------------------------------------------
// mhc_merge: both streams update from the pre-merge values (§2.7)
// ---------------------------------------------------------------------------------------------

/// `grid = (ceil(d / SN_BLOCK), n_tokens)`. `mix` is post-sweep, post-exponentiation (the mixing
/// matrix in probability space); feeding it log-mixes multiplies the streams by negative numbers
/// and the residual reaches inf around execution 46 (§2.9).
extern "C" __global__ void sn_mhc_merge_f32(float* a, float* b, const float* br, const float* mix, const float* post,
                                            int d, long sa, long sb, long sbr, long smix, long spost) {
    const long t = blockIdx.y;
    const int k = blockIdx.x * SN_BLOCK + threadIdx.x;
    if (k >= d) return;
    const float m0 = mix[t * smix + 0];
    const float m1 = mix[t * smix + 1];
    const float m2 = mix[t * smix + 2];
    const float m3 = mix[t * smix + 3];
    const float p0 = post[t * spost + 0];
    const float p1 = post[t * spost + 1];
    float* pa = a + t * sa;
    float* pb = b + t * sb;
    const float* pbr = br + t * sbr;
    const float ai = pa[k];
    const float bi = pb[k];
    const float x = pbr[k];
    pa[k] = ai * m0 + bi * m1 + x * p0;
    pb[k] = ai * m2 + bi * m3 + x * p1;
}

// ---------------------------------------------------------------------------------------------
// The unfused baseline for the Phase 4 micro-benchmark. NOT engine code.
//
// These mirror the fork's node spelling so `sn bench fusion` measures the thing that is actually
// slow: one ggml node per launch, and the balancing written the way `build_route()` writes it -
// six tensors per pair update (`sub`, `neg`, `softplus`, `neg`, `softplus`, `neg`), four pair
// updates per sweep, 20 sweeps, each node a single thread reading a `[1,T]` view that strides 8
// f32 through the router's logits buffer.
//
// Everything that is *not* the balancing is given to the baseline in its best possible shape (one
// node for the whole router matmul, one node for `in`), so the >= 100x gate is decided by the
// 480 nodes alone rather than by a strawman.
// ---------------------------------------------------------------------------------------------

/// node: `sub` - the fork's `[1,1]` view of logits lane `ia` minus lane `ib`, strided `stride`.
extern "C" __global__ void sn_ub_sub(float* dst, const float* src, long ia, long ib, long stride) {
    dst[0] = src[ia * stride] - src[ib * stride];
}

/// node: `neg` on a scalar view.
extern "C" __global__ void sn_ub_neg(float* dst, const float* src, long i) { dst[i] = -src[i]; }

/// node: `softplus` on a scalar view, same 20.0f threshold.
extern "C" __global__ void sn_ub_softplus(float* dst, const float* src, long i) { dst[i] = sn_softplus(src[i]); }

/// node: the whole router matmul (given to the baseline for free - one node for all 8 logits).
extern "C" __global__ void sn_ub_gemv8(const float* w, const float* x, float* out, int d2, int n_route) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        for (int r = 0; r < n_route; r++) {
            float acc = 0.0f;
            for (int k = 0; k < d2; k++) acc += w[(long)r * d2 + k] * x[k];
            out[r] = acc;
        }
    }
}

/// node: bias add over the 8 logits.
extern "C" __global__ void sn_ub_bias_add(float* y, const float* bias, int n) {
    if (threadIdx.x < n) y[threadIdx.x] += bias[threadIdx.x];
}

/// node: the four sigmoids + two `2*sigmoid` posts.
extern "C" __global__ void sn_ub_sigmoid_post(const float* lg, float* s, float* post) {
    if (threadIdx.x == 0) {
        s[0] = sn_sigmoid(lg[0]);
        s[1] = sn_sigmoid(lg[1]);
        post[0] = 2.0f * sn_sigmoid(lg[2]);
        post[1] = 2.0f * sn_sigmoid(lg[3]);
    }
}

/// node: `in` = (a*s0 + b*s1) * pre_scale over the streams.
extern "C" __global__ void sn_ub_mix_ab(const float* a, const float* b, const float* s, float* out, int d, float pre_scale) {
    const int stride = gridDim.x * blockDim.x;
    for (int k = blockIdx.x * blockDim.x + threadIdx.x; k < d; k += stride) {
        out[k] = (a[k] * s[0] + b[k] * s[1]) * pre_scale;
    }
}

/// node: exp() of the four balanced log-mixes.
extern "C" __global__ void sn_ub_exp4(const float* lg, float* mix) {
    if (threadIdx.x < 4) mix[threadIdx.x] = expf(lg[threadIdx.x]);
}
