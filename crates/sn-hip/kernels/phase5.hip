// SuperNova's Phase 5 kernel image: the quantised matmul set.
//
// Design-plan §7 Phase 5 is "the bandwidth half": decode is 8.8 GB of weight traffic per token
// (§5) and nothing else in the engine can pay for that. Every kernel here is written for one
// thing - read each weight byte exactly once, accumulate in f32, no shared-memory staging of
// weights (§7: "memory-bound `gemv`, ... 128-byte loads, one pass, f32 accumulate, no smem
// staging").
//
// The types are the ones the checkpoint stores (design-plan §1.2, corrected by
// `Tools/tensor_census.py`): **Q4_K, Q6_K and f32**. There is no Q2_K in this file, so the
// plan's `gemv_q2k` does not exist; `sn_gemv_q6_k` is its counterpart here. `Docs/numerics.md`
// and `Docs/bench.md` §3 record the measurements these kernels produce.
//
// The dequantisation is a transliteration of the *reference* dequantiser in
// `crates/sn-core/src/dq.rs` (itself verified against GGML's C by `Tools/dequant_ref.py`,
// design-plan §2.9): the field offsets, the Q4_K 6-bit scale/min packing, and the Q6_K
// `ql`/`qh`/signed-`sc` split are the same ones the scalar reference walks, because the Phase 5
// gate is "NMSE <= 1e-4 vs the scalar reference" against `dq::Mat::gemv` on the checkpoint's own
// tensors, not against a re-derived approximation.
//
// Q4_K's dot uses the same algebraic rearrangement `dq::q4k_dot` uses:
//     sum_i (d*sc*q_i - dm*mn) * x_i  =  d*sc * sum(q_i*x_i) - dm*mn * sum(x_i)
// which halves the multiplies per weight. It is a different *summation order* from an
// elementwise dequant-then-multiply, which is why the gate is 1e-4 and not 1e-12: the measured
// gap between the two orders is ~1e-13 (Docs/bench.md §3.2), i.e. f32 rounding only.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// Workgroup size every GEMV/GEMM here is registered with (8 wavefronts of 32 on gfx1102).
// `ops::GEMV_BLOCK` in Rust must equal this; the row-group indexing below uses the *compiled*
// workgroup size for its shared-memory stride exactly as `SN_ROUTE_THREADS` does in phase4.hip,
// and `Tools/build-kernels.sh` records it in `target/phase5.meta` for the loader to refuse.
#define SN_GEMV_BLOCK 256

// Q4_K / Q6_K super-block geometry, from `dq.rs` (and `ggml-common.h`):
//   block_q4_K = { d:f16 @0, dmin:f16 @2, scales:u8[12] @4, qs:u8[128] @16 } = 144 B / 256 values
//   block_q6_K = { ql:u8[128] @0, qh:u8[64] @128, sc:i8[16] @192, d:f16 @208 } = 210 B / 256 values
#define SN_QK_K 256
#define SN_Q4K_BLOCK 144
#define SN_Q6K_BLOCK 210

// ---------------------------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------------------------

struct SnF4 {
    float x, y, z, w;
};

struct SnU4 {
    unsigned int x, y, z, w;
    __device__ unsigned int& operator[](int i) { return (&x)[i]; }
    __device__ const unsigned int& operator[](int i) const { return (&x)[i]; }
};

__device__ __forceinline__ float sn_ld_f16(unsigned short h) { return __half2float(*reinterpret_cast<const __half*>(&h)); }

__device__ __forceinline__ float sn_ld_f16(const unsigned char* p) {
    // The f16 fields are 2-byte aligned inside a 144- or 210-byte super-block whose base is even,
    // which is what every row start in this model is (ne0 is always a multiple of 256).
    return __half2float(*reinterpret_cast<const __half*>(p));
}

/// One value inside a wavefront: `part` must have room for `blockDim.x / 32` floats.
__device__ __forceinline__ float sn_wave_sum(float v) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        v += __shfl_down_sync(0xffffffffffffffffull, v, off);
    }
    return v;
}

/// Fold the first `n` lanes of a wavefront (n <= 32). Every lane named by the mask has to be
/// active and converged for `__shfl_down_sync` - naming 32 lanes while only `warps_per_row` of
/// them are executing is undefined and **hangs on gfx1102** (it cost a debugging session in
/// `Docs/bench.md` §3). So all 32 lanes take part and the ones past `n` contribute an exact
/// zero: the summation order stays a property of the kernel, and the value is bit-exact.
__device__ __forceinline__ float sn_wave_sum_partial(float v, int n) {
    (void)n;
    return sn_wave_sum(v);
}

/// Reduce the per-thread partials of one row's dot product. `red` is shared memory with
/// `SN_GEMV_BLOCK` floats; row groups are contiguous blocks of `warps_per_row` wavefronts, so
/// only those wavefronts' partials are summed - in a fixed order, which is why a replay is
/// bit-identical (no atomics anywhere in this file).
__device__ __forceinline__ float sn_row_group_sum(float acc, float* red, int warps_per_row) {
    const int warp = (int)threadIdx.x >> 5;
    const int lane = (int)threadIdx.x & 31;
    const int group = warp / warps_per_row;
    const int in_group = warp % warps_per_row;
    // Two stages, both in a fixed order: first the 32 lanes of this wavefront (each lane owns a
    // different slice of the row, so this is where the bulk of the sum is folded), then the
    // wavefronts of this row group through shared memory.
    acc = sn_wave_sum(acc);
    if (lane == 0) red[group * warps_per_row + in_group] = acc;
    __syncthreads();
    // One wavefront per row group finishes the sum (warps_per_row <= 32 by construction). All 32
    // lanes execute the shuffle butterfly; lanes past `warps_per_row` read zero, which is exact, so
    // the result is the same value whatever the split.
    if (in_group == 0) {
        float s = lane < warps_per_row ? red[group * warps_per_row + lane] : 0.0f;
        return sn_wave_sum(s);
    }
    return 0.0f;
}

/// The pre-norm the decode program folds into a GEMV (design-plan §6 P2, `norm -> gemv`):
/// `x_i` is read as `x_i * rsqrtf(mean(x^2) + eps) [* norm_w_i]`. `norm_w == nullptr` means the
/// norm is a separate launch (`sn_rmsnorm_f32`) and the GEMV reads `x` as it is.
/// Computed once per workgroup, cooperatively: every row group reads the same `x`, so this is one
/// pass over an activation vector that stays in L1/L2, not one per row.
__device__ __forceinline__ float sn_norm_scale(const float* x, int k, float eps) {
    __shared__ float part[SN_GEMV_BLOCK / 32];
    __shared__ float scale;
    float ss = 0.0f;
    for (int i = (int)threadIdx.x; i < k; i += SN_GEMV_BLOCK) {
        ss += x[i] * x[i];
    }
    ss = sn_wave_sum(ss);
    const int lane = (int)threadIdx.x & 31;
    const int warp = (int)threadIdx.x >> 5;
    if (lane == 0) part[warp] = ss;
    __syncthreads();
    float tot = 0.0f;
    for (int w = 0; w < SN_GEMV_BLOCK / 32; w++) tot += part[w];
    if (threadIdx.x == 0) scale = rsqrtf(tot / (float)k + eps);
    __syncthreads();
    return scale;
}

/// Read `x` for the dot, applying the folded norm. The values are f32 either way; a folded norm
/// costs one multiply per activation read, and activations are L1-resident, so this is not the
/// path that decides GEMV throughput - weight bytes are.
__device__ __forceinline__ float sn_xv(const float* x, int i, float scale, const float* norm_w) {
    float v = x[i] * scale;
    return norm_w ? v * norm_w[i] : v;
}

// Q4_K's scale/min packing, exactly `q4k_scale_min` in `dq.rs` (6-bit scales for sub-blocks 0..3,
// 6-bit for the mins, and the high 2 bits of each folded into the next 4 bytes).
__device__ __forceinline__ void sn_q4k_sc_min(const unsigned char* s, int j, float* ds, float* dm, float d, float dmin) {
    unsigned int sc, mn;
    if (j < 4) {
        sc = s[j] & 63u;
        mn = s[j + 4] & 63u;
    } else {
        sc = (s[j + 4] & 0xFu) | ((s[j - 4] >> 6) << 4);
        mn = (s[j + 4] >> 4) | ((s[j] >> 6) << 4);
    }
    *ds = d * (float)sc;
    *dm = dmin * (float)mn;
}

// ---------------------------------------------------------------------------------------------
// gemv: y = W x, one output row per (32 * warps_per_row) threads
//
// `grid.x` covers ceil(n / rows_per_wg) row groups; `blockDim.x == SN_GEMV_BLOCK`. A row is
// walked by `warps_per_row` wavefronts, each taking every `warps_per_row`-th super-block, so the
// same kernel serves both shapes in this model:
//
//   * wide output (ffn_down 4096 rows, output 248320 rows): warps_per_row = 1, thousands of
//     independent row streams, and the whole point is saturating DRAM;
//   * narrow output (ssm_beta_alpha 64 rows, attn_k 1024 rows): rows are too few to fill 40 CUs,
//     so the K axis is split across wavefronts of one row group to buy parallelism. That is done
//     inside a workgroup and reduced in shared memory - **not** with atomics, so a replay is
//     bit-identical (the Phase 3 determinism property is not traded away for a few microseconds).
//
// `warps_per_row` is chosen by `gemm::pick` in Rust from (n, k) and baked into the launch.
// ---------------------------------------------------------------------------------------------

/// Q4_K, wide-load form. A wavefront loads **four super-blocks of one row per iteration**: lane
/// `l` takes the `uint4` at `qs + 32p + 16h` where `p = (l>>1)&3` is the sub-block *pair* and
/// `h = l&1` which half of its 32 bytes (`lane & 7` is the `uint4` index inside the 128 B `qs`) - one `qs[0..128)` of
/// super-blocks b, b+1, b+2, b+3 (a row's super-blocks are 144 B apart and 144 % 16 == 0, so
/// every one of these loads is 16-byte aligned).
///
/// That is 32 of the row's values per lane per load, against 8 in the first version (a `uint` per
/// lane, 128 B per wavefront). The first version measured 87-91 GB/s on this card - 31 % of the
/// measured read bandwidth - and the per-element rate it got was the same as Q6_K's despite Q4_K
/// carrying 1.46x more values per byte, which says the limit was load instructions and latency,
/// not DRAM (`Docs/bench.md` §3.2). A lane's f16 `d`/`dmin` pair also comes in one 4-byte load
/// (`d` is at 0, `dmin` at 2), and the two sub-blocks of a pair share the loaded bytes, so a pair
/// costs 1 vector load + 3 scalar header loads instead of 6+ scalar loads per 8 values.
///
/// Determinism is unchanged: the accumulation order inside a lane is fixed, and the cross-lane
/// reduce is the fixed two-stage butterfly in `sn_row_group_sum` (no atomics).
extern "C" __global__ void sn_gemv_q4_k(const unsigned char* __restrict__ W, const float* __restrict__ x,
                                        float* __restrict__ y, int k, int n, const float* __restrict__ norm_w,
                                        float eps, int warps_per_row) {
    __shared__ float red[SN_GEMV_BLOCK];
    const int threads_per_row = 32 * warps_per_row;
    const int rows_per_wg = SN_GEMV_BLOCK / threads_per_row;
    const int row = (int)blockIdx.x * rows_per_wg + (int)threadIdx.x / threads_per_row;
    if (row >= n) return;

    const int in_row = (int)threadIdx.x % threads_per_row;
    const int lane = in_row & 31;
    const int wpr_idx = in_row >> 5;

    const float scale = norm_w ? sn_norm_scale(x, k, eps) : 1.0f;

    const int nb = k / SN_QK_K;
    const unsigned char* wr = W + (size_t)row * (size_t)(nb * SN_Q4K_BLOCK);

    const int p = (lane >> 1) & 3;   // sub-block pair inside this super-block
    const int h = lane & 1;          // which 16 bytes of the pair's 32
    const int sb = lane >> 3;        // which of the four super-blocks this wavefront covers
    float acc = 0.0f;

    // `warps_per_row` wavefronts split the super-blocks of one row; each takes every
    // `warps_per_row`-th group of four.
    for (int b0 = wpr_idx * 4; b0 < nb; b0 += warps_per_row * 4) {
        const int b = b0 + sb;
        if (b >= nb) continue;
        const unsigned char* blk = wr + (size_t)b * SN_Q4K_BLOCK;
        // d @0 and dmin @2 in one load; the pair's two scale entries in two more.
        const unsigned int dm2 = reinterpret_cast<const unsigned int*>(blk)[0];
        const float d = sn_ld_f16((unsigned short)(dm2 & 0xFFFFu));
        const float dmin = sn_ld_f16((unsigned short)(dm2 >> 16));
        const unsigned char* s = blk + 4;
        const SnU4 q = reinterpret_cast<const SnU4*>(blk + 16)[lane & 7];
        float ds0, dm0, ds1, dm1;
        sn_q4k_sc_min(s, 2 * p, &ds0, &dm0, d, dmin);
        sn_q4k_sc_min(s, 2 * p + 1, &ds1, &dm1, d, dmin);

        // Sub-block 2p is the low nibbles of the pair's bytes and 2p+1 the high nibbles *of the
        // same bytes*, but they are different elements of the row: sub-block j covers row elements
        // 32j..32j+31, so the x range differs by 32 even though the weight bytes do not.
        const int e0 = b * SN_QK_K + 64 * p + 16 * h;
        const int e1 = e0 + SN_QK_K / 8;
        float s0 = 0.0f, s1 = 0.0f, a0 = 0.0f, a1 = 0.0f;
#pragma unroll
        for (int i = 0; i < 4; i++) {
            const unsigned int qw = q[i];
            // Both activation vectors come in `float4`s. `x` and `norm_w` are 16-byte aligned (the
            // arena hands out 256-byte-aligned slots) and `e0`/`e1` are multiples of four, so the
            // casts are legal; adding the four elements in register order keeps the sum fixed.
            const SnF4 xa = reinterpret_cast<const SnF4*>(x + e0)[i];
            const SnF4 xb = reinterpret_cast<const SnF4*>(x + e1)[i];
            float wa[4], wb[4];
            if (norm_w) {
                const SnF4 na = reinterpret_cast<const SnF4*>(norm_w + e0)[i];
                const SnF4 nb = reinterpret_cast<const SnF4*>(norm_w + e1)[i];
                // x * rsqrt(mean(x^2)+eps) * norm_w, the same value `sn_xv` computes elementwise.
                wa[0] = xa.x * scale * na.x; wa[1] = xa.y * scale * na.y;
                wa[2] = xa.z * scale * na.z; wa[3] = xa.w * scale * na.w;
                wb[0] = xb.x * scale * nb.x; wb[1] = xb.y * scale * nb.y;
                wb[2] = xb.z * scale * nb.z; wb[3] = xb.w * scale * nb.w;
            } else {
                wa[0] = xa.x * scale; wa[1] = xa.y * scale; wa[2] = xa.z * scale; wa[3] = xa.w * scale;
                wb[0] = xb.x * scale; wb[1] = xb.y * scale; wb[2] = xb.z * scale; wb[3] = xb.w * scale;
            }
            s0 += wa[0] + wa[1] + wa[2] + wa[3];
            s1 += wb[0] + wb[1] + wb[2] + wb[3];
            a0 += (float)((qw >> 0) & 0xFu) * wa[0] + (float)((qw >> 8) & 0xFu) * wa[1] +
                  (float)((qw >> 16) & 0xFu) * wa[2] + (float)((qw >> 24) & 0xFu) * wa[3];
            a1 += (float)((qw >> 4) & 0xFu) * wb[0] + (float)((qw >> 12) & 0xFu) * wb[1] +
                  (float)((qw >> 20) & 0xFu) * wb[2] + (float)((qw >> 28) & 0xFu) * wb[3];
        }
        acc += ds0 * a0 - dm0 * s0;
        acc += ds1 * a1 - dm1 * s1;
    }

    acc = sn_row_group_sum(acc, red, warps_per_row);
    if (wpr_idx == 0 && (threadIdx.x & 31) == 0) y[row] = acc;
}

/// Q6_K. `q6k_value`/`q6k_dequant` in `dq.rs`, one value per (lane, group): lane `l` of a
/// wavefront owns element `32g + l` of each half-block, which is the walk the reference does.
///
/// Unlike Q4_K this kernel cannot use wide loads, and that is worth recording precisely because it
/// is the reason Q6_K sits at ~104 GB/s where Q4_K reaches 231 (`Docs/bench.md` §3.2): a Q6_K
/// super-block is **210 bytes**, so consecutive rows of a matrix start at offsets that are only
/// 2-byte aligned (210 mod 16 = 2) and no `uint4`/`uint` load inside one is aligned. The first
/// version of this kernel tried the `uint4` trick anyway and produced garbage (NMSE 4e16) rather
/// than faulting - unaligned vector loads on GCN are undefined, not checked. The fix is a staging
/// decision, not a kernel one: lay Q6_K super-blocks out padded to 224 bytes in the weight arena
/// (Phase 6), which makes every field aligned at its existing offset and costs 6.7 % of the Q6_K
/// bytes at load time. Until then this path is byte loads, and the loads - not the DRAM - set the
/// rate.
extern "C" __global__ void sn_gemv_q6_k(const unsigned char* __restrict__ W, const float* __restrict__ x,
                                        float* __restrict__ y, int k, int n, const float* __restrict__ norm_w,
                                        float eps, int warps_per_row) {
    __shared__ float red[SN_GEMV_BLOCK];
    const int threads_per_row = 32 * warps_per_row;
    const int rows_per_wg = SN_GEMV_BLOCK / threads_per_row;
    const int row = (int)blockIdx.x * rows_per_wg + (int)threadIdx.x / threads_per_row;
    if (row >= n) return;

    const int in_row = (int)threadIdx.x % threads_per_row;
    const int lane = in_row & 31;
    const int wpr_idx = in_row >> 5;

    const float scale = norm_w ? sn_norm_scale(x, k, eps) : 1.0f;

    const int nb = k / SN_QK_K;
    const unsigned char* wr = W + (size_t)row * (size_t)(nb * SN_Q6K_BLOCK);
    const int is = lane >> 4;   // the `is` in dq's `sc[is + 2k]`

    float acc = 0.0f;
    for (int b = wpr_idx; b < nb; b += warps_per_row) {
        const unsigned char* blk = wr + (size_t)b * SN_Q6K_BLOCK;
        const float d = sn_ld_f16(blk + 208);
#pragma unroll
        for (int hb = 0; hb < 2; hb++) {
            // The top-two-bits byte is the same for all four groups of this half-block: one load.
            const unsigned int qh = blk[128 + 32 * hb + lane];
#pragma unroll
            for (int g = 0; g < 4; g++) {
                const unsigned char qlb = blk[64 * hb + ((g & 1) ? 32 + lane : lane)];
                const unsigned int nib = (g < 2) ? (qlb & 0xFu) : (qlb >> 4);
                const unsigned int bits = (qh >> (2 * g)) & 0x3u;
                const signed char sc = (signed char)blk[192 + 8 * hb + is + 2 * g];
                const int idx = b * SN_QK_K + 128 * hb + 32 * g + lane;
                acc += d * (float)sc * (float)((int)((bits << 4) | nib) - 32) * sn_xv(x, idx, scale, norm_w);
            }
        }
    }

    acc = sn_row_group_sum(acc, red, warps_per_row);
    if (wpr_idx == 0 && (threadIdx.x & 31) == 0) y[row] = acc;
}

extern "C" __global__ void sn_gemv_f32(const float* __restrict__ W, const float* __restrict__ x, float* __restrict__ y,
                                       int k, int n, const float* __restrict__ norm_w, float eps,
                                       int warps_per_row) {
    __shared__ float red[SN_GEMV_BLOCK];
    const int threads_per_row = 32 * warps_per_row;
    const int rows_per_wg = SN_GEMV_BLOCK / threads_per_row;
    const int row = (int)blockIdx.x * rows_per_wg + (int)threadIdx.x / threads_per_row;
    if (row >= n) return;

    const int in_row = (int)threadIdx.x % threads_per_row;
    const int lane = in_row & 31;
    const int wpr_idx = in_row >> 5;

    const float scale = norm_w ? sn_norm_scale(x, k, eps) : 1.0f;

    const int kq = k / 4;
    float acc = 0.0f;
    if (((k & 3) == 0) && (((uintptr_t)W + (size_t)row * (size_t)k * 4u) & 15u) == 0 && (((uintptr_t)x) & 15u) == 0) {
        const float4* wr = reinterpret_cast<const float4*>(W + (size_t)row * (size_t)k);
        const float4* x4 = reinterpret_cast<const float4*>(x);
        const float4* w4 = norm_w ? reinterpret_cast<const float4*>(norm_w) : nullptr;
        for (int i = wpr_idx * 32 + lane; i < kq; i += threads_per_row) {
            const float4 w = wr[i];
            const float4 v = x4[i];
            if (w4) {
                const float4 nw = w4[i];
                acc += w.x * v.x * nw.x + w.y * v.y * nw.y + w.z * v.z * nw.z + w.w * v.w * nw.w;
            } else {
                acc += w.x * v.x + w.y * v.y + w.z * v.z + w.w * v.w;
            }
        }
    } else {
        // A shape this engine does not have (every weight here is 256-aligned in k, and every
        // arena offset is 256-byte aligned), kept so the kernel is correct rather than fast-but-
        // sometimes-wrong: `gemm::pick` refuses to launch it, and `check::gemm_checks` never hits it.
        const float* wr = W + (size_t)row * (size_t)k;
        for (int i = wpr_idx * 32 + lane; i < k; i += threads_per_row) {
            acc += wr[i] * (norm_w ? x[i] * norm_w[i] : x[i]);
        }
    }
    // `scale` is a common factor of every term, so it goes on the sum once.
    acc *= scale;

    acc = sn_row_group_sum(acc, red, warps_per_row);
    if (wpr_idx == 0 && (threadIdx.x & 31) == 0) y[row] = acc;
}

// ---------------------------------------------------------------------------------------------
// gemm: y[t*n + row] = sum_k W[row][k] * x[t*k + k], for T > 1 (the prefill path)
//
// The decode path above never touches this: at T = 1 there is nothing to reuse. For T > 1 a weight
// element is worth T multiply-adds, which is what makes a chunked prefill read the checkpoint once
// per chunk instead of once per token (§5).
//
// Structure: 64 output rows x 64 tokens per workgroup, K walked in steps of 32, the weight tile
// dequantised into shared memory and the activation tile staged there, and plain f32
// multiply-adds (16 per thread per K step). **This is deliberately not the `mma.f32.f16` path the
// plan asks for**: the RDNA3 WMMA fragment layout is measured and written up in Docs/bench.md §3.5
// (the builtin exists and runs at a measured 34.4 TFLOP/s on this card, but its operand mapping is
// not the adjacent-rows-per-lane layout this kernel would need, and nothing here guesses at it).
// The tile is written so the FMA body can be replaced by a WMMA body without touching the
// dequantiser, the staging, or the caller. `sn bench gemm` reports this kernel against a *measured*
// f32-FMA peak and against the measured WMMA peak, and Docs/bench.md §3.4 states which gate it
// meets and which it does not.
//
// `grid = (ceil(n/64), ceil(T/64))`, `blockDim.x == SN_GEMV_BLOCK`.
// ---------------------------------------------------------------------------------------------

#define SN_RT 64
#define SN_TT 64
#define SN_KT 32
// +1 float of padding per row: a 32- or 64-wide row would otherwise map every column to one bank.
// Two different pitches, because the two tiles have different widths - `xs[SN_KT][SN_TT+1]` is 65
// floats per row, and indexing it with a 33-float pitch (the first version did) silently writes
// past the end of each row and reads a neighbour's: every GEMM check failed with NMSE ~0.9.
#define SN_PADK 33
#define SN_PADT 65

/// Dequantise element `kk` of weight row `row` into f32. One value per call, because the tile fill
//  is indexed by (row, k) - the K-quant super-block arithmetic is identical to the GEMV's.
__device__ __forceinline__ float sn_w1_q4k(const unsigned char* W, int row, int k, int kk) {
    const int nb = k / SN_QK_K;
    const unsigned char* blk = W + (size_t)row * (size_t)(nb * SN_Q4K_BLOCK) + (size_t)(kk / SN_QK_K) * SN_Q4K_BLOCK;
    const int j = (kk % SN_QK_K) / 32;
    const int i = kk % 32;
    float ds, dm;
    sn_q4k_sc_min(blk + 4, j, &ds, &dm, sn_ld_f16(blk), sn_ld_f16(blk + 2));
    const unsigned int q = (blk[16 + (j / 2) * 32 + i] >> ((j & 1) * 4)) & 0xFu;
    return ds * (float)q - dm;
}

__device__ __forceinline__ float sn_w1_q6k(const unsigned char* W, int row, int k, int kk) {
    const int nb = k / SN_QK_K;
    const unsigned char* blk = W + (size_t)row * (size_t)(nb * SN_Q6K_BLOCK) + (size_t)(kk / SN_QK_K) * SN_Q6K_BLOCK;
    const int hb = (kk % SN_QK_K) / 128;
    const int l = kk % 32;
    const int g = (kk % 128) / 32;
    const unsigned int nib = (g == 0) ? (blk[64 * hb + l] & 0xFu)
                       : (g == 1)    ? (blk[64 * hb + 32 + l] & 0xFu)
                       : (g == 2)    ? (blk[64 * hb + l] >> 4)
                                     : (blk[64 * hb + 32 + l] >> 4);
    const unsigned int bits = (blk[128 + 32 * hb + l] >> (2 * g)) & 0x3u;
    const int qv = (int)(((bits << 4) | nib)) - 32;
    return sn_ld_f16(blk + 208) * (float)(*((const signed char*)(blk + 192 + 8 * hb) + l / 16 + 2 * g)) * (float)qv;
}

/// The shared tiled body. `LOAD_W(row, kk)` yields the weight value.
template <typename LoadW>
__device__ __forceinline__ void sn_gemm_body(const float* __restrict__ X, float* __restrict__ Y,
                                             int k, int n, int T, LoadW load_w) {
    __shared__ float ws[SN_RT][SN_PADK];
    __shared__ float xs[SN_KT][SN_PADT];

    const int row0 = (int)blockIdx.x * SN_RT;
    const int tok0 = (int)blockIdx.y * SN_TT;
    // 256 threads as 16 x 16, each owning 4 rows x 4 tokens.
    const int tr = ((int)threadIdx.x >> 4) * 4;
    const int tc = ((int)threadIdx.x & 15) * 4;

    float acc[4][4];
#pragma unroll
    for (int a = 0; a < 4; a++)
#pragma unroll
        for (int bq = 0; bq < 4; bq++) acc[a][bq] = 0.0f;

    const int fill_row = (int)threadIdx.x >> 2;
    const int fill_k = ((int)threadIdx.x & 3) * (SN_KT / 4);

    for (int k0 = 0; k0 < k; k0 += SN_KT) {
#pragma unroll
        for (int i = 0; i < SN_KT / 4; i++) {
            const int kk = k0 + fill_k + i;
            const int r = row0 + fill_row;
            ws[fill_row][fill_k + i] = (r < n) ? load_w(r, kk) : 0.0f;
        }
        // Stage the activation tile (SN_KT x SN_TT), `X` being [token][k] in the arena layout the
        // Phase 4 kernels already use (`out[t*sout + i]`), so this reads a column of X.
#pragma unroll
        for (int i = 0; i < (SN_KT * SN_TT) / SN_GEMV_BLOCK; i++) {
            const int idx = (int)threadIdx.x + i * SN_GEMV_BLOCK;
            const int kk = idx / SN_TT;
            const int tt = idx % SN_TT;
            const int tok = tok0 + tt;
            xs[kk][tt] = (k0 + kk < k && tok < T) ? X[(size_t)tok * (size_t)k + (size_t)(k0 + kk)] : 0.0f;
        }
        __syncthreads();

        for (int kk = 0; kk < SN_KT; kk++) {
            float w[4], xv[4];
#pragma unroll
            for (int a = 0; a < 4; a++) w[a] = ws[tr + a][kk];
#pragma unroll
            for (int bq = 0; bq < 4; bq++) xv[bq] = xs[kk][tc + bq];
#pragma unroll
            for (int a = 0; a < 4; a++)
#pragma unroll
                for (int bq = 0; bq < 4; bq++) acc[a][bq] += w[a] * xv[bq];
        }
        __syncthreads();
    }

    for (int a = 0; a < 4; a++) {
        const int row = row0 + tr + a;
        if (row >= n) continue;
        for (int bq = 0; bq < 4; bq++) {
            const int tok = tok0 + tc + bq;
            if (tok < T) Y[(size_t)tok * (size_t)n + row] = acc[a][bq];
        }
    }
}

struct SnLoadQ4K {
    const unsigned char* W;
    int k;
    __device__ float operator()(int row, int kk) const { return sn_w1_q4k(W, row, k, kk); }
};
struct SnLoadQ6K {
    const unsigned char* W;
    int k;
    __device__ float operator()(int row, int kk) const { return sn_w1_q6k(W, row, k, kk); }
};
struct SnLoadF32 {
    const float* W;
    int k;
    __device__ float operator()(int row, int kk) const { return W[(size_t)row * (size_t)k + kk]; }
};

extern "C" __global__ void sn_gemm_q4_k(const unsigned char* __restrict__ W, const float* __restrict__ X,
                                        float* __restrict__ Y, int k, int n, int T) {
    sn_gemm_body(X, Y, k, n, T, SnLoadQ4K{W, k});
}

extern "C" __global__ void sn_gemm_q6_k(const unsigned char* __restrict__ W, const float* __restrict__ X,
                                        float* __restrict__ Y, int k, int n, int T) {
    sn_gemm_body(X, Y, k, n, T, SnLoadQ6K{W, k});
}

extern "C" __global__ void sn_gemm_f32(const float* __restrict__ W, const float* __restrict__ X,
                                       float* __restrict__ Y, int k, int n, int T) {
    sn_gemm_body(X, Y, k, n, T, SnLoadF32{W, k});
}

// ---------------------------------------------------------------------------------------------
// The two references the Phase 5 gates are quoted against. Both are *measured on this device*, so
// `Docs/bench.md` §3 never quotes a datasheet figure (design-plan §8: "every phase ships a
// measurement, not an adjective").
// ---------------------------------------------------------------------------------------------

/// Read-only bandwidth: sum a large buffer with float4 loads and fold the per-thread partials with
/// atomics (the value is not used for anything except keeping the loads live). This is the
/// "measured device bandwidth" the >= 80 % GEMV gate is a fraction of.
extern "C" __global__ void sn_read_bw(const float4* __restrict__ p, size_t n4, float* out) {
    float acc = 0.0f;
    const size_t stride = (size_t)gridDim.x * blockDim.x;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n4; i += stride) {
        const float4 v = p[i];
        acc += v.x + v.y + v.z + v.w;
    }
    if (acc == 1234.5f) atomicAdd(out, acc);
}

/// Pure-register f32 FMA throughput: 8 independent accumulators so the limit is issue rate, not
/// latency. This is the ceiling the tiled GEMM above is measured against.
extern "C" __global__ void sn_peak_fma_f32(float* out, int iters) {
    float a0 = 1.0f, a1 = 1.0f, a2 = 1.0f, a3 = 1.0f, a4 = 1.0f, a5 = 1.0f, a6 = 1.0f, a7 = 1.0f;
    const float s = 1.0000001f;
    for (int i = 0; i < iters; i++) {
        a0 = fmaf(a0, s, 1.0f);
        a1 = fmaf(a1, s, 2.0f);
        a2 = fmaf(a2, s, 3.0f);
        a3 = fmaf(a3, s, 4.0f);
        a4 = fmaf(a4, s, 5.0f);
        a5 = fmaf(a5, s, 6.0f);
        a6 = fmaf(a6, s, 7.0f);
        a7 = fmaf(a7, s, 8.0f);
    }
    if (threadIdx.x == 0) out[blockIdx.x] = a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7;
}

/// The WMMA probe: `__builtin_amdgcn_wmma_f32_16x16x16_f16_w32` exists for gfx1102 and this loop
/// is the measured peak (Docs/bench.md §3.5). A dependent accumulator chain, so it is a lower
/// bound on the machine's WMMA issue rate and an upper bound on what a kernel can hope for.
#if defined(__gfx11__)
// The builtin is gated on `wmma-256b-insts` + wavefrontsize32, i.e. RDNA3 (gfx11) only: the same
// image compiled for gfx1035 (this box reports both ISAs) rejects it, which is exactly the
// "don't build a kernel that cannot launch here" rule in design-plan §8. On a non-gfx11 target the
// probe exists but reports 0, and `sn bench gemm` prints the peak as unavailable.
typedef _Float16 __attribute__((ext_vector_type(16))) sn_h16;
typedef float __attribute__((ext_vector_type(8))) sn_f8;

extern "C" __global__ void sn_peak_wmma(float* out, int iters) {
    sn_h16 a{}, b{};
    sn_f8 c{};
    a[0] = (_Float16)1.0f;
    b[0] = (_Float16)1.0f;
    for (int i = 0; i < iters; i++) {
        c = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, c);
    }
    if (threadIdx.x == 0) {
        float s = 0.0f;
#pragma unroll
        for (int i = 0; i < 8; i++) s += c[i];
        out[blockIdx.x] = s;
    }
}
#else
extern "C" __global__ void sn_peak_wmma(float* out, int iters) {
    if (threadIdx.x == 0) out[blockIdx.x] = 0.0f;
}
#endif

// ---------------------------------------------------------------------------------------------
// debug probe (not part of the engine): one 32-thread workgroup, one output row, no split. It
// writes each lane's accumulator so a wrong element->byte mapping shows up as *which* lane is
// non-zero for a one-hot activation. `sn gpu gemm` runs it only under SN_DEBUG.
// ---------------------------------------------------------------------------------------------
extern "C" __global__ void sn_dbg_q4k_lane(const unsigned char* __restrict__ W, const float* __restrict__ x,
                                           float* __restrict__ out, int k) {
    const int lane = (int)threadIdx.x;
    const int nb = k / SN_QK_K;
    float acc = 0.0f;
    const int p = lane >> 3;
    const int iv = 4 * (lane & 7);
    for (int b = 0; b < nb; b++) {
        const unsigned char* blk = W + (size_t)b * SN_Q4K_BLOCK;
        const float d = sn_ld_f16(blk);
        const float dmin = sn_ld_f16(blk + 2);
        const unsigned char* s = blk + 4;
        const unsigned int q = reinterpret_cast<const unsigned int*>(blk + 16)[lane];
#pragma unroll
        for (int t = 0; t < 2; t++) {
            const int j = 2 * p + t;
            float ds, dm;
            sn_q4k_sc_min(s, j, &ds, &dm, d, dmin);
            const int shift = (j & 1) * 4;
            const int vbase = b * SN_QK_K + 32 * j + iv;
            float sx = 0.0f, sq = 0.0f;
#pragma unroll
            for (int i = 0; i < 4; i++) {
                const float xv = x[vbase + i];
                sx += xv;
                sq += (float)((q >> (shift + 8 * i)) & 0xFu) * xv;
            }
            acc += ds * sq - dm * sx;
        }
    }
    out[lane] = acc;
}

/// Same signature as the GEMV kernels, no work: reports what the arguments actually contain. The
/// Phase 5 debugging session (design-plan Phase 5 status) turned on this, not on reading the code.
extern "C" __global__ void sn_dbg_args(float* __restrict__ out, int k, int n,
                                       const float* __restrict__ norm_w, float eps, int warps_per_row) {
    if (threadIdx.x == 0) {
        out[0] = (float)warps_per_row;
        out[1] = (float)k;
        out[2] = (float)n;
        out[3] = eps;
        out[4] = norm_w ? 1.0f : 0.0f;
        out[5] = (float)blockDim.x;
        out[6] = (float)gridDim.x;
    }
}
