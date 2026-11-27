// SuperNova's Phase 6 kernel image: the caches and the two recurrent inner loops.
//
// design-plan §7 Phase 6 is "the first correct GPU forward": the Phase 5 GEMVs and the Phase 4
// elementwise set already cover every *matmul and pointwise* step of §2, so what is missing is
// everything that touches state - the KV cache (§2.3) and the Gated DeltaNet recurrence (§2.4) -
// plus the embedding gather that starts a forward. Nothing here is fast by construction; each
// kernel is written to be *obviously* the formula in §2, because the gate is "logits NMSE <= 1e-4
// vs the Phase 2 reference", and Phase 8's chunked GDN / flash attention replaces them whole.
//
// The arithmetic follows `crates/sn-core/src/oracle.rs` (one oracle per kernel here; `sn gpu
// phase6` scores every pair against a deterministic input). Where §2 fixes an order it is kept:
// decay -> predict -> write -> read inside the delta rule ("getting this wrong is silent and
// plausible"), and the delta-rule state row is the same `[head][v][k]` layout the reference walks.
//
// KV storage follows `crates/sn-hip/src/cache.rs`: a row is the quant array, then the f32 scale
// array (NOT GGML's interleaved `block_q8_0`). The reason is the Phase 5 Q6_K result
// (Docs/bench.md §3.3): a 34-byte block stride leaves every second block unaligned, and unaligned
// vector loads on GCN are undefined rather than merely slow.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// Workgroup size for the gather / append / norm kernels in this image. Recorded in the .meta by
// Tools/build-kernels.sh; `ops::PHASE6_BLOCK` in Rust must agree (a mismatch silently mis-sizes
// the shared reductions - it has already cost this project a wrong-value bug, see Docs/bench.md).
#define SN_BLOCK 256

// KV cache kinds, matching `cache::KvKind`.
#define SN_KV_F16 0
#define SN_KV_Q8_0 1
#define SN_KV_Q4_0 2
#define SN_QBLOCK 32

// Q4_K super-block geometry, shared with phase5.hip and dq.rs.
#define SN_QK_K 256
#define SN_Q4K_BLOCK 144

// ----------------------------------------------------------------------------------------------
// helpers
// ----------------------------------------------------------------------------------------------

__device__ __forceinline__ float sn6_ld_f16(const unsigned char* p) {
    unsigned short h;
    __builtin_memcpy(&h, p, 2);
    return __half2float(*reinterpret_cast<const __half*>(&h));
}

__device__ __forceinline__ void sn6_st_f16(unsigned char* p, float v) {
    const __half h = __float2half(v);
    unsigned short bits;
    __builtin_memcpy(&bits, &h, 2);
    p[0] = (unsigned char)(bits & 0xFF);
    p[1] = (unsigned char)(bits >> 8);
}

__device__ __forceinline__ float sn6_wave_sum(float v) {
    // 64-bit mask, all lanes: a mask naming inactive lanes hangs gfx1102 rather than returning a
    // wrong value (Docs/bench.md, Phase 5).
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffffffffffffull, v, off);
    return v;
}

__device__ __forceinline__ float sn6_silu(float x) { return x / (1.0f + expf(-x)); }

__device__ __forceinline__ float sn6_softplus(float x) { return x > 20.0f ? x : log1pf(expf(x)); }

/// Quantisation-block scale index of element `d` of one row (`cache.rs` layout: quant array, then
/// one f32 scale per 32-value block).
__device__ __forceinline__ float sn6_kv_value(const unsigned char* q, const float* scales, int d, int kind) {
    if (kind == SN_KV_F16) return sn6_ld_f16(q + 2 * d);
    const int b = d / SN_QBLOCK;
    const int o = d % SN_QBLOCK;
    const float sc = scales[b];
    if (kind == SN_KV_Q8_0) return (float)(int)(signed char)q[b * SN_QBLOCK + o] * sc;
    const unsigned char byte = q[b * (SN_QBLOCK / 2) + (o >> 1)];
    const int nib = (o & 1) ? (byte >> 4) : (byte & 0xF);
    return (float)(nib - 8) * sc; // q4_0 is symmetric around 8
}

// ----------------------------------------------------------------------------------------------
// embedding row gather: cur = embd[id] (§2), not a matmul against a one-hot vector
//
// Q4_K, the type `token_embd` has in this checkpoint (§1.2). The field walk is phase5.hip's and
// dq.rs's - `block_q4_K = { d:f16 @0, dmin:f16 @2, scales:u8[12] @4, qs:u8[128] @16 }`, with the
// element mapping the verified Q4_K GEMV uses (`dq::q4k_dequant`, cross-checked against GGML's C by
// Tools/dequant_ref.py): scale index `j = v/32` with `j` in 0..7, i.e. eight groups of 32 values;
// the bytes of pair `j/2` are `qs[32*(j/2) .. +32]`, sub-block `2k` is their low nibbles and
// `2k+1` the high nibbles of the *same* bytes.
//
// One workgroup per row: thread t writes elements [16t, 16t+16), so 256 threads cover exactly the
// 4096-wide embedding row of this model. A row is read once per token; nothing here is worth
// tuning until Phase 7 asks for it.
// ----------------------------------------------------------------------------------------------

__device__ __forceinline__ void sn6_q4k_sc_min(const unsigned char* s, int j, float* sc, float* mn) {
    unsigned int a, b;
    if (j < 4) {
        a = s[j] & 63u;
        b = s[j + 4] & 63u;
    } else {
        a = (s[j + 4] & 0xFu) | ((s[j - 4] >> 6) << 4);
        b = (s[j + 4] >> 4) | ((s[j] >> 6) << 4);
    }
    *sc = (float)a;
    *mn = (float)b;
}

// The row index comes from device memory (the token id the graph staged from pinned host memory),
// so the program never has to be rebuilt for a different token.
extern "C" __global__ void sn_row_gather_q4_k(const unsigned char* __restrict__ W, const unsigned int* __restrict__ id,
                                              float* __restrict__ y, int k) {
    const long row = (long)*id;
    const int nb = k / SN_QK_K; // super-blocks per row (16 for k = 4096)
    const unsigned char* wr = W + row * (long)nb * SN_Q4K_BLOCK;
    const int lo = threadIdx.x * 16;
    const int hi = lo + 16;
    for (int v = lo; v < hi; v++) {
        if (v >= k) break;
        const int b = v / SN_QK_K;
        const int e = v % SN_QK_K;
        const int j = e / SN_QBLOCK; // 0..7
        const int hi_half = j & 1;
        const int byte = (j / 2) * SN_QBLOCK + (e % SN_QBLOCK);
        const unsigned char* blk = wr + (long)b * SN_Q4K_BLOCK;
        const float d = sn6_ld_f16(blk + 0);
        const float dmin = sn6_ld_f16(blk + 2);
        float sc, mn;
        sn6_q4k_sc_min(blk + 4, j, &sc, &mn);
        const unsigned char q = blk[16 + byte];
        const int nib = hi_half ? (q >> 4) : (q & 0xF);
        y[v] = d * sc * (float)nib - dmin * mn;
    }
}

// ----------------------------------------------------------------------------------------------
// full attention, decode (T = 1)
// ----------------------------------------------------------------------------------------------

/// Per-head Q/K RMS norm with the checkpoint's `attn_q_norm` / `attn_k_norm` gains (§2.3: "q =
/// rms(q,1e-6) * attn_q_norm", over the 256 head dims, per head) - before rope, and the gate half
/// of each q head block is never touched. One launch covers both tensors: 16 q heads + 4 k heads x
/// 14 executions is 280 launches per token if each head got its own.
extern "C" __global__ void sn_qk_head_norm_f32(float* __restrict__ qg, const float* __restrict__ qw,
                                               float* __restrict__ kx, const float* __restrict__ kw,
                                               int n_qh, int n_kh, int d, int q_stride, float eps) {
    // grid: n_qh + n_kh workgroups, block: SN_BLOCK threads, one head per workgroup.
    const int h = blockIdx.x;
    float* p;
    const float* g;
    if (h < n_qh) {
        p = qg + (long)h * q_stride;
        g = qw;
    } else {
        p = kx + (long)(h - n_qh) * d;
        g = kw;
    }
    __shared__ float v[512];
    __shared__ float red[SN_BLOCK / 32];
    for (int i = threadIdx.x; i < d; i += SN_BLOCK) v[i] = p[i];
    __syncthreads();
    float ss = 0.0f;
    for (int i = threadIdx.x; i < d; i += SN_BLOCK) ss += v[i] * v[i];
    ss = sn6_wave_sum(ss);
    if ((threadIdx.x & 31) == 0) red[threadIdx.x / 32] = ss;
    __syncthreads();
    float tot = 0.0f;
#pragma unroll
    for (int w = 0; w < SN_BLOCK / 32; w++) tot += red[w];
    const float inv = rsqrtf(tot / (float)d + eps);
    for (int i = threadIdx.x; i < d; i += SN_BLOCK) p[i] = v[i] * inv * g[i];
}

/// Append one token's K and V. Append-only: nothing in this engine writes a row it did not append,
/// and the position a row belongs to is recorded separately (`cache::Sessions::append`).
__device__ __forceinline__ void sn6_append_f16(const float* __restrict__ src, unsigned char* __restrict__ dst,
                                               int values) {
    for (int i = threadIdx.x; i < values; i += blockDim.x) sn6_st_f16(dst + 2 * i, src[i]);
}

/// One thread per quantisation block, so the block max is computed once and no two threads touch
/// the same byte (the q4_0 nibble packing would otherwise race).
__device__ __forceinline__ void sn6_append_q8(const float* __restrict__ src, unsigned char* __restrict__ qs,
                                              float* __restrict__ scales, int values) {
    const int nb = (values + SN_QBLOCK - 1) / SN_QBLOCK;
    for (int b = threadIdx.x; b < nb; b += blockDim.x) {
        float amax = 0.0f;
        const int first = b * SN_QBLOCK;
        for (int j = 0; j < SN_QBLOCK && first + j < values; j++) amax = fmaxf(amax, fabsf(src[first + j]));
        const float sc = amax / 127.0f;
        scales[b] = sc;
        const float inv = sc != 0.0f ? 1.0f / sc : 0.0f;
        for (int j = 0; j < SN_QBLOCK && first + j < values; j++) {
            const int q = (int)__float2int_rn(src[first + j] * inv);
            qs[b * SN_QBLOCK + j] = (unsigned char)(char)(q < -127 ? -127 : (q > 127 ? 127 : q));
        }
    }
}

__device__ __forceinline__ void sn6_append_q4(const float* __restrict__ src, unsigned char* __restrict__ qs,
                                              float* __restrict__ scales, int values) {
    const int nb = (values + SN_QBLOCK - 1) / SN_QBLOCK;
    for (int b = threadIdx.x; b < nb; b += blockDim.x) {
        float amax = 0.0f;
        const int first = b * SN_QBLOCK;
        for (int j = 0; j < SN_QBLOCK && first + j < values; j++) amax = fmaxf(amax, fabsf(src[first + j]));
        const float sc = amax / 7.0f;
        scales[b] = sc;
        const float inv = sc != 0.0f ? 1.0f / sc : 0.0f;
        for (int j = 0; j < SN_QBLOCK; j += 2) {
            const int v0 = first + j;
            const int v1 = first + j + 1;
            const int q0 = v0 < values ? min(15, max(0, (int)__float2int_rn(src[v0] * inv) + 8)) : 8;
            const int q1 = v1 < values ? min(15, max(0, (int)__float2int_rn(src[v1] * inv) + 8)) : 8;
            qs[b * (SN_QBLOCK / 2) + (j >> 1)] = (unsigned char)(q0 | (q1 << 4));
        }
    }
}

extern "C" __global__ void sn_kv_append(const float* __restrict__ kx, const float* __restrict__ vx,
                                        unsigned char* __restrict__ kv, const int* __restrict__ len,
                                        int values, int row_bytes, int tok_bytes, int kind) {
    // The row index comes from the device-side sequence length (the meta rows the graph staged from
    // pinned memory), which carry the length *after* this append - hence `- 1`. A per-token offset
    // baked into the launch would mean rebuilding the graph for every token.
    const long k_off = (long)(*len - 1) * tok_bytes;
    unsigned char* kd = kv + k_off;
    unsigned char* vd = kv + k_off + row_bytes;
    if (kind == SN_KV_F16) {
        sn6_append_f16(kx, kd, values);
        sn6_append_f16(vx, vd, values);
        return;
    }
    const int nb = (values + SN_QBLOCK - 1) / SN_QBLOCK;
    const int qbytes = (kind == SN_KV_Q8_0) ? values : values / 2;
    if (kind == SN_KV_Q8_0) {
        sn6_append_q8(kx, kd, (float*)(kd + qbytes), values);
        sn6_append_q8(vx, vd, (float*)(vd + qbytes), values);
    } else {
        sn6_append_q4(kx, kd, (float*)(kd + qbytes), values);
        sn6_append_q4(vx, vd, (float*)(vd + qbytes), values);
    }
    (void)nb;
}

/// Causal attention for one query token against this execution's own cache (§2.3, decode shape).
///
/// `tok_bytes` is the distance between one cached token and the next (K row then V row), and
/// `row_bytes` the distance from a token's K row to its V row - `cache::CacheLayout`'s two strides.
///
/// One workgroup per q-head; K and V tiles of `SN_KV_TILE` tokens live in shared memory; the
/// softmax runs online (running max and running sum), so the cache is read once. The reference
/// takes max then sum in two passes; the difference between that and an online softmax is f32
/// summation order, which is what the 1e-4 gate is for. GQA: `kh = h * n_kv / n_q`, the same
/// pairing the reference uses (§2.3, ik2 = iq2/rk2).
#define SN_KV_TILE 8

extern "C" __global__ void sn_attn_decode_f32(const float* __restrict__ qg,
                                              const unsigned char* __restrict__ kv,
                                              float* __restrict__ out, const int* __restrict__ len,
                                              int n_qh, int n_kvh, int d, int q_stride, int values,
                                              int row_bytes, int tok_bytes, int kind, float scale) {
    const int h = blockIdx.x;
    const int kh = (h * n_kvh) / n_qh;
    const int n = *len;
    const int lane = threadIdx.x;
    const int wave = lane / 32;
    const int lid = lane % 32;
    const int nw = (int)(blockDim.x / 32);
    // Per-value and per-head strides inside one row. The scale array follows the quant array.
    const int qs_bytes = (kind == SN_KV_F16) ? values * 2 : (kind == SN_KV_Q8_0 ? values : values / 2);
    const int qs_per_head = (kind == SN_KV_F16) ? d * 2 : (kind == SN_KV_Q8_0 ? d : d / 2);
    const int sc_per_head = (d / SN_QBLOCK) * 4;

    __shared__ float sq[512];
    __shared__ float tk[SN_KV_TILE * 256];
    __shared__ float tv[SN_KV_TILE * 256];
    __shared__ float sc[32];
    __shared__ float red[8 * 256];

    for (int i = lane; i < d; i += blockDim.x) sq[i] = qg[(long)h * q_stride + i];
    float acc[8];
#pragma unroll
    for (int i = 0; i < 8; i++) acc[i] = 0.0f;
    float m = -INFINITY;
    float lsum = 0.0f;
    const int dstep = d / 32; // dims per lane: 8 for d = 256, with 8 wavefronts
    __syncthreads();

    for (int t0 = 0; t0 < n; t0 += SN_KV_TILE) {
        __syncthreads();
        // ---- K tile -----------------------------------------------------------------------
        for (int idx = lane; idx < SN_KV_TILE * d; idx += blockDim.x) {
            const int t = idx / d;
            const int dd = idx % d;
            const int row = t0 + t;
            if (row < n) {
                const unsigned char* r = kv + (long)row * tok_bytes;
                tk[t * 256 + dd] = sn6_kv_value(r + kh * qs_per_head, (const float*)(r + qs_bytes) + (kh * d) / SN_QBLOCK, dd, kind);
            } else {
                tk[t * 256 + dd] = 0.0f;
            }
        }
        __syncthreads();
        // ---- one token per wavefront ------------------------------------------------------
        const int t = wave;
        const int row = t0 + t;
        if (t < SN_KV_TILE) {
            float s = -INFINITY;
            if (row < n) {
                float p = 0.0f;
                for (int i = 0; i < dstep; i++) p += sq[lid * dstep + i] * tk[t * 256 + lid * dstep + i];
                s = sn6_wave_sum(p) * scale;
            }
            if (lid == 0) sc[t] = s;
        }
        __syncthreads();
        // ---- online softmax, computed identically by every thread ---------------------------
        float nm = m;
        float add = 0.0f;
        for (int j = 0; j < SN_KV_TILE && t0 + j < n; j++) nm = fmaxf(nm, sc[j]);
        for (int j = 0; j < SN_KV_TILE && t0 + j < n; j++) add += expf(sc[j] - nm);
        const float corr = (m == -INFINITY) ? 0.0f : expf(m - nm);
        m = nm;
        lsum = lsum * corr + add;
#pragma unroll
        for (int i = 0; i < 8; i++) acc[i] *= corr;
        // ---- V tile -----------------------------------------------------------------------
        for (int idx = lane; idx < SN_KV_TILE * d; idx += blockDim.x) {
            const int tt = idx / d;
            const int dd = idx % d;
            const int r = t0 + tt;
            if (r < n) {
                const unsigned char* p = kv + (long)r * tok_bytes + row_bytes;
                tv[tt * 256 + dd] = sn6_kv_value(p + kh * qs_per_head, (const float*)(p + qs_bytes) + (kh * d) / SN_QBLOCK, dd, kind);
            } else {
                tv[tt * 256 + dd] = 0.0f;
            }
        }
        __syncthreads();
        if (row < n && t < SN_KV_TILE) {
            const float p = expf(sc[t] - m);
            for (int i = 0; i < dstep; i++) acc[i] += p * tv[t * 256 + lid * dstep + i];
        }
    }

    // ---- reduce across wavefronts, then normalise ------------------------------------------
    __syncthreads();
    for (int i = 0; i < dstep; i++) red[wave * 256 + lid * dstep + i] = acc[i];
    __syncthreads();
    const float inv = (lsum > 0.0f) ? 1.0f / lsum : 0.0f;
    for (int i = 0; i < dstep; i++) {
        float tot = 0.0f;
        for (int w = 0; w < nw; w++) tot += red[w * 256 + lid * dstep + i];
        out[(long)h * d + lid * dstep + i] = tot * inv;
    }
    (void)sc_per_head;
}

// ----------------------------------------------------------------------------------------------
// Gated DeltaNet, one token (§2.4)
// ----------------------------------------------------------------------------------------------

/// Causal depthwise conv over the raw qkv, then SiLU. One thread per channel: the window
/// (`(d_conv-1) x channels`, oldest row first) is rewritten in place, which is safe because each
/// thread only ever touches its own column.
extern "C" __global__ void sn_gdn_conv_silu_f32(const float* __restrict__ qkv, const float* __restrict__ cw,
                                                float* __restrict__ out, float* __restrict__ state,
                                                int channels, int d_conv) {
    float win[8];
    const int rows = d_conv - 1;
    for (int c = blockIdx.x * blockDim.x + threadIdx.x; c < channels; c += gridDim.x * blockDim.x) {
        const float* w = cw + (long)c * d_conv;
        float s = w[rows] * qkv[c];
        for (int t = 0; t < rows; t++) {
            win[t] = state[(long)t * channels + c];
            s += w[t] * win[t];
        }
        for (int t = 0; t < rows; t++) {
            state[(long)t * channels + c] = (t + 1 < rows) ? win[t + 1] : qkv[c];
        }
        out[c] = sn6_silu(s);
    }
}

/// L2-normalise q/k head blocks (§2.4: `q / sqrt(sum(q^2) + eps)`). Separate from the conv so the
/// eps and the head geometry stay visible; the caller launches it over the q and k halves in place.
extern "C" __global__ void sn_gdn_l2norm_f32(const float* __restrict__ in, float* __restrict__ out,
                                             int dk, float eps) {
    __shared__ float v[256];
    __shared__ float red[SN_BLOCK / 32];
    const int base = blockIdx.x * dk;
    for (int i = threadIdx.x; i < dk; i += SN_BLOCK) v[i] = in[base + i];
    __syncthreads();
    float ss = 0.0f;
    for (int i = threadIdx.x; i < dk; i += SN_BLOCK) ss += v[i] * v[i];
    ss = sn6_wave_sum(ss);
    if ((threadIdx.x & 31) == 0) red[threadIdx.x / 32] = ss;
    __syncthreads();
    float tot = 0.0f;
#pragma unroll
    for (int w = 0; w < SN_BLOCK / 32; w++) tot += red[w];
    const float inv = rsqrtf(tot + eps); // no /n: this is the L2 form, not an RMS norm
    for (int i = threadIdx.x; i < dk; i += SN_BLOCK) out[base + i] = v[i] * inv;
}

/// The delta rule, one workgroup per v-head, block = d_v threads.
///
/// Each thread owns one *row* of the state, S[i][*], which is exactly the unit every step needs:
///   decay:   S[i][*] *= exp(gate)
///   predict: pred_i = S[i] . k
///   write:   S[i] += k * delta_i, with delta_i = (v_i - pred_i) * beta_i
///   read:    o_i = S[i] . q
/// all row-local, so the loop needs no cross-lane reduction - which is why the plan calls for this
/// shape. `ssm_a` is stored as `-exp(A_log)` (§2.4), so `gate` is a negative log-decay.
extern "C" __global__ void sn_gdn_decode_f32(const float* __restrict__ conv, const float* __restrict__ ba,
                                             const float* __restrict__ ssm_a, const float* __restrict__ dt_bias,
                                             float* __restrict__ state, float* __restrict__ out,
                                             int n_v_heads, int d_v, int dk, int n_k_heads) {
    const int h = blockIdx.x;
    const int i = threadIdx.x;
    if (i >= d_v) return;
    const int kh = h % n_k_heads;
    float* row = state + (long)h * d_v * dk + (long)i * dk;

    const float beta = ba[h];
    const float gate = ssm_a[h] * sn6_softplus(ba[n_v_heads + h] + dt_bias[h]);
    const float decay = expf(gate);

    // conv layout: q | k | v, with n_k_heads blocks of dk for q and k and n_v_heads of d_v for v.
    const float* qt = conv + (long)kh * dk;
    const float* kt = conv + (long)n_k_heads * dk + (long)kh * dk;
    const float* vt = conv + (long)2 * (long)n_k_heads * dk + (long)h * d_v;

    float pred = 0.0f;
    for (int j = 0; j < dk; j++) {
        const float s = row[j] * decay;
        row[j] = s;
        pred += s * kt[j];
    }
    const float delta = (vt[i] - pred) * beta;
    for (int j = 0; j < dk; j++) row[j] += kt[j] * delta;
    float o = 0.0f;
    for (int j = 0; j < dk; j++) o += row[j] * qt[j];
    out[(long)h * d_v + i] = o / sqrtf((float)dk); // §2.4: the 1/sqrt(128) factor is real
}

/// The gated per-head norm on the GDN output (§2.4): `(rms(o_head) * ssm_norm) * silu(z)`.
extern "C" __global__ void sn_gdn_gate_norm_f32(const float* __restrict__ core, const float* __restrict__ w,
                                                const float* __restrict__ z, float* __restrict__ out,
                                                int d_v, float eps) {
    __shared__ float v[256];
    __shared__ float red[SN_BLOCK / 32];
    const int base = blockIdx.x * d_v;
    for (int i = threadIdx.x; i < d_v; i += SN_BLOCK) v[i] = core[base + i];
    __syncthreads();
    float ss = 0.0f;
    for (int i = threadIdx.x; i < d_v; i += SN_BLOCK) ss += v[i] * v[i];
    ss = sn6_wave_sum(ss);
    if ((threadIdx.x & 31) == 0) red[threadIdx.x / 32] = ss;
    __syncthreads();
    float tot = 0.0f;
#pragma unroll
    for (int w2 = 0; w2 < SN_BLOCK / 32; w2++) tot += red[w2];
    const float inv = rsqrtf(tot / (float)d_v + eps);
    for (int i = threadIdx.x; i < d_v; i += SN_BLOCK) {
        out[base + i] = v[i] * inv * w[i] * sn6_silu(z[base + i]);
    }
}

/// Snapshot / restore of the GDN state and conv windows: Phase 6's prefix-cache primitive and
/// Phase 9's rollback. One kernel for both halves, because a snapshot is only ever valid as a pair
/// (`cache::Sessions::check_pair`). `stride_*` are the per-execution strides in the live region, so
/// a snapshot region packs the same rows contiguously.
extern "C" __global__ void sn_gdn_snapshot(float* __restrict__ live_state, float* __restrict__ live_conv,
                                           float* __restrict__ snap_state, float* __restrict__ snap_conv,
                                           int state_values, int conv_values, int execs,
                                           int state_stride, int conv_stride, int to_snapshot) {
    // Two grid-stride loops rather than one index space: the live and snapshot buffers swap roles
    // when restoring, and keeping the src/dst types separate is what makes that readable.
    const long ns = (long)execs * state_values;
    for (long idx = (long)blockIdx.x * blockDim.x + threadIdx.x; idx < ns; idx += (long)gridDim.x * blockDim.x) {
        const long e = idx / state_values;
        const long o = idx % state_values;
        if (to_snapshot) snap_state[e * state_stride + o] = live_state[e * state_stride + o];
        else live_state[e * state_stride + o] = snap_state[e * state_stride + o];
    }
    const long nc = (long)execs * conv_values;
    for (long idx = (long)blockIdx.x * blockDim.x + threadIdx.x; idx < nc; idx += (long)gridDim.x * blockDim.x) {
        const long e = idx / conv_values;
        const long o = idx % conv_values;
        if (to_snapshot) snap_conv[e * conv_stride + o] = live_conv[e * conv_stride + o];
        else live_conv[e * conv_stride + o] = snap_conv[e * conv_stride + o];
    }
}
