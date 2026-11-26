/*
 * attn-probe.c — what precision does the reference's *multi-token score path* actually use?
 *
 * Background (Docs/numerics.md §3 item 3): `sn ref` agrees with the reference engine bit-exactly at
 * position 0 and drifts to cos ~0.99999 at positions >= 1, and the drift starts inside the first
 * full-attention block. Everything upstream (GDN state, conv window, mHC, norms) is exact across
 * positions, so the open question was what the reference does differently once a query row has more
 * than one key to score against.
 *
 * Reading `llama-graph.cpp::build_attn_mha` says: with flash attention on - which is the DEFAULT,
 * because `llama_context_default_params().flash_attn_type == LLAMA_FLASH_ATTN_TYPE_AUTO` and
 * `llama-context.cpp:322` maps AUTO to `cparams.flash_attn = true` - the attention subgraph is
 * `ggml_flash_attn_ext(q_f32, k_f16, v_f16, mask_f16, ...)`:
 *   - K/V are the KV cache, f16 by default (`-ctk f16`),
 *   - the mask tensor is created f16 when flash_attn is on (`llama-graph.cpp:1110,3167`),
 *   - Q is converted to K's `vec_dot_type`, which for an f16 K cache is f16
 *     (`ggml_compute_forward_flash_attn_ext_f16_one_chunk`: `q_to_vec_dot(pq, Q_q, DK)`).
 * So every score the reference forms for a multi-token row comes from f16-rounded operands, with f32
 * accumulation. At position 0 that is invisible: a one-element softmax is exactly 1 whatever the
 * scores are, which is exactly where the agreement is bit-exact.
 *
 * This program measures the size of that effect instead of asserting it. It builds the same op
 * (`ggml_flash_attn_ext`, scale = 1/sqrt(head_dim), causal mask) twice with identical inputs -
 * once with the shapes the reference uses (f16 KV + f16 mask), once with f32 KV + f32 mask - and
 * reports the per-position agreement between them, plus the agreement of an independent f32
 * reimplementation (the loop at the bottom, which mirrors `sn-core::forward::attention`).
 *
 * It needs no model, no GPU, and runs in well under a second:
 *
 *   gcc -O1 -w -o Tools/attn-probe Tools/attn-probe.c \
 *     -I <fork>/ggml/include -L <fork>/build/bin -lggml -lggml-base \
 *     -Wl,-rpath,<fork>/build/bin -lm
 *   Tools/attn-probe                       # default: this checkpoint's attention geometry
 *   Tools/attn-probe --tokens 6 --kv f32 --dump-dir /tmp/attnp
 */
#include <ggml.h>
#include <ggml-cpu.h>

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Same geometry as SenseNova-Loop-9B's full-attention blocks. */
#define D_HEAD 256
#define N_HEAD 16
#define N_KV   4

/* Deterministic inputs. Reproduced independently in `sn attn-probe` (Rust), so the two engines are
 * always fed the same q/k/v/mask. Standard LCG (glibc constants). */
static unsigned long g_seed = 0x853c0e9a9b1df89bUL;
static double frand(void) {
    g_seed = g_seed * 6364136223846793005UL + 1442695040888963407UL;
    return (double)((g_seed >> 11) & 0x1FFFFFFFFFFFFFUL) / 9007199254740992.0; /* [0,1) */
}
static float gauss(void) { /* Box-Muller; q/k after a per-head RMS norm are O(1) like this */
    double u = frand(), v = frand();
    if (u < 1e-12) u = 1e-12;
    return (float)(sqrt(-2.0 * log(u)) * cos(2.0 * M_PI * v));
}

static void store_as(struct ggml_tensor * t, const float * src) {
    const int64_t n = ggml_nelements(t);
    if (t->type == GGML_TYPE_F32) {
        memcpy(t->data, src, sizeof(float) * (size_t) n);
        return;
    }
    ggml_fp16_t * dst = (ggml_fp16_t *) t->data;
    for (int64_t i = 0; i < n; i++) {
        dst[i] = ggml_fp32_to_fp16(src[i]);
    }
}

static void write_bin(const char * dir, const char * name, const float * data, size_t n) {
    if (!dir) return;
    char path[512];
    snprintf(path, sizeof(path), "%s/%s", dir, name);
    FILE * f = fopen(path, "wb");
    if (!f) { fprintf(stderr, "cannot write %s\n", path); return; }
    fwrite(data, sizeof(float), n, f);
    fclose(f);
}

static void round16_inplace(float * x, size_t n) {
    for (size_t i = 0; i < n; i++) x[i] = ggml_fp16_to_fp32(ggml_fp32_to_fp16(x[i]));
}

/* Run one ggml graph: q f32, kv in `kv_type`, mask in `mask_type`. Output is always f32. */
static void run_graph(int n_tok, enum ggml_type kv_type, enum ggml_type mask_type, float * out,
                      const float * q, const float * k, const float * v, const float * mask,
                      const char * dump_dir, const char * tag) {
    const int64_t neq[4] = { D_HEAD, n_tok, N_HEAD, 1 };
    const int64_t nek[4] = { D_HEAD, n_tok, N_KV, 1 };
    const int64_t nem[4] = { n_tok, n_tok, 1, 1 };

    struct ggml_init_params ip = {
        .mem_size   = 512ull * 1024 * 1024,   /* flat budget: plenty for one FATT node */
        .mem_buffer = NULL,
        .no_alloc   = false,
    };

    struct ggml_context * ctx = ggml_init(ip);
    struct ggml_tensor * tq = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, neq);
    struct ggml_tensor * tk = ggml_new_tensor(ctx, kv_type,   4, nek);
    struct ggml_tensor * tv = ggml_new_tensor(ctx, kv_type,   4, nek);
    struct ggml_tensor * tm = ggml_new_tensor(ctx, mask_type, 4, nem);

    memcpy(tq->data, q, sizeof(float) * ggml_nelements(tq));
    store_as(tk, k); store_as(tv, v); store_as(tm, mask);

    struct ggml_tensor * cur = ggml_flash_attn_ext(ctx, tq, tk, tv, tm,
                                                   1.0f / sqrtf((float) D_HEAD), 0.0f, 0.0f);
    ggml_prec_set_acc(cur, GGML_PREC_F32);   /* what build_attn_mha sets */

    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, cur);

    struct ggml_backend * cpu = ggml_backend_cpu_init();
    if (!cpu) { fprintf(stderr, "no CPU backend\n"); exit(1); }
    ggml_backend_cpu_set_n_threads(cpu, 1);
    if (ggml_backend_graph_compute(cpu, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "graph compute failed\n"); exit(1);
    }
    ggml_backend_free(cpu);

    const float * o = (const float *) cur->data;
    memcpy(out, o, sizeof(float) * (size_t) ggml_nelements(cur));
    write_bin(dump_dir, tag, out, (size_t) ggml_nelements(cur));
    ggml_free(ctx);
}

/* Independent f32 attention, same semantics as sn-core::forward::attention (scale, causal mask,
 * max-subtracted softmax, GQA broadcast kv_head = h / (n_head / n_kv)). */
static void attn_f32_ref(int n_tok, const float * q, const float * k, const float * v,
                         const float * mask, float * out) {
    const float scale = 1.0f / sqrtf((float) D_HEAD);
    static float p[4096];
    for (int h = 0; h < N_HEAD; h++) {
        const int kh = h * N_KV / N_HEAD;
        for (int t = 0; t < n_tok; t++) {
            const float * qh = q + ((size_t) t * N_HEAD + h) * D_HEAD;
            const float * khv = k + ((size_t) t * N_KV + kh) * D_HEAD;
            float mx = -INFINITY;
            for (int c = 0; c < n_tok; c++) {
                const float * kc = k + ((size_t) c * N_KV + kh) * D_HEAD;
                float s = 0.0f;
                for (int i = 0; i < D_HEAD; i++) s += qh[i] * kc[i];
                s *= scale;
                float mv = mask[(size_t) t * n_tok + c];
                p[c] = (mv == -INFINITY) ? -INFINITY : s + mv;
                if (p[c] > mx) mx = p[c];
            }
            float sum = 0.0f;
            for (int c = 0; c < n_tok; c++) { p[c] = expf(p[c] - mx); sum += p[c]; }
            float inv = 1.0f / sum;
            for (int i = 0; i < D_HEAD; i++) {
                float acc = 0.0f;
                for (int c = 0; c < n_tok; c++) {
                    acc += p[c] * inv * v[((size_t) c * N_KV + kh) * D_HEAD + i];
                }
                out[((size_t) t * N_HEAD + h) * D_HEAD + i] = acc;
            }
        }
    }
}

static double cos_of(const float * a, const float * b, size_t n) {
    double da = 0, db = 0, dd = 0;
    for (size_t i = 0; i < n; i++) { dd += (double)a[i]*b[i]; da += (double)a[i]*a[i]; db += (double)b[i]*b[i]; }
    if (da <= 0 || db <= 0) return 1.0;
    return dd / sqrt(da * db);
}
static double maxabs(const float * a, const float * b, size_t n) {
    double m = 0;
    for (size_t i = 0; i < n; i++) { double d = fabs((double)a[i] - b[i]); if (d > m) m = d; }
    return m;
}

int main(int argc, char ** argv) {
    int n_tok = 4;
    float gain = 1.0f;
    const char * dump_dir = NULL;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--tokens") && i + 1 < argc) n_tok = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--gain") && i + 1 < argc) gain = (float) atof(argv[++i]);
        else if (!strcmp(argv[i], "--dump-dir") && i + 1 < argc) dump_dir = argv[++i];
        else { fprintf(stderr, "usage: attn-probe [--tokens N] [--gain G] [--dump-dir DIR]\n"); return 2; }
    }
    if (n_tok < 1 || n_tok > 1024) { fprintf(stderr, "bad --tokens\n"); return 2; }

    const size_t nq = (size_t) n_tok * N_HEAD * D_HEAD;
    const size_t nk = (size_t) n_tok * N_KV * D_HEAD;
    const size_t nm = (size_t) n_tok * n_tok;
    float * q = malloc(nq * sizeof(float));
    float * k = malloc(nk * sizeof(float));
    float * v = malloc(nk * sizeof(float));
    float * mask = malloc(nm * sizeof(float));
    /* --gain scales q and k, i.e. how peaked the softmax is. A trained attention layer has
     * logit std of several units; neutral (unit-Gaussian) inputs are the mildest case. */
    for (size_t i = 0; i < nq; i++) q[i] = gauss() * gain;
    for (size_t i = 0; i < nk; i++) k[i] = gauss() * gain;
    for (size_t i = 0; i < nk; i++) v[i] = gauss();
    for (int t = 0; t < n_tok; t++)
        for (int c = 0; c < n_tok; c++)
            mask[(size_t) t * n_tok + c] = (c > t) ? -INFINITY : 0.0f;  /* the fork's decoder mask */

    float * a = malloc(nq * sizeof(float));  /* f16 KV + f16 mask  == what the reference runs */
    float * b = malloc(nq * sizeof(float));  /* f32 KV + f16 mask  == the f32 contract        */
    float * r = malloc(nq * sizeof(float));  /* our own f32 semantics                            */
    float * q16 = malloc(nq * sizeof(float));
    float * k16 = malloc(nk * sizeof(float));
    float * v16 = malloc(nk * sizeof(float));
    float * t1 = malloc(nq * sizeof(float));
    float * t2 = malloc(nq * sizeof(float));

    /* The reference's path. Note the mask is f16 unconditionally - `ggml_flash_attn_ext` asserts
     * `mask->type == GGML_TYPE_F16` (ggml.c:5434), so there is no f32-mask variant to compare. */
    run_graph(n_tok, GGML_TYPE_F16, GGML_TYPE_F16, a, q, k, v, mask, dump_dir, "probe-ref.bin");
    /* The f32 contract: same op, same mask, K/V kept f32 (which build_attn_mha's own
     * `ggml_cast(k, F16)` prevents in the real graph, but this is the number we want). */
    run_graph(n_tok, GGML_TYPE_F32, GGML_TYPE_F16, b, q, k, v, mask, dump_dir, "probe-f32.bin");
    attn_f32_ref(n_tok, q, k, v, mask, r);
    write_bin(dump_dir, "probe-f32-ref-loop.bin", r, nq);

    /* One-factor-at-a-time attribution of the reference's error budget. */
    memcpy(q16, q, nq * sizeof(float)); round16_inplace(q16, nq);
    memcpy(k16, k, nk * sizeof(float)); round16_inplace(k16, nk);
    memcpy(v16, v, nk * sizeof(float)); round16_inplace(v16, nk);
    run_graph(n_tok, GGML_TYPE_F32, GGML_TYPE_F16, t1, q16, k, v, mask, NULL, NULL);
    printf("\nsingle-factor deviation from the f32 contract (whole-row cos):\n");
    printf("  Q rounded to f16   %14.10f   (the CPU FATT does this: q_to_vec_dot, vec_dot_type of an f16 K)\n", cos_of(t1, b, nq));
    run_graph(n_tok, GGML_TYPE_F32, GGML_TYPE_F16, t1, q, k16, v, mask, NULL, NULL);
    printf("  K rounded to f16   %14.10f   (the KV cache type)\n", cos_of(t1, b, nq));
    run_graph(n_tok, GGML_TYPE_F32, GGML_TYPE_F16, t1, q, k, v16, mask, NULL, NULL);
    printf("  V rounded to f16   %14.10f   (the KV cache type)\n", cos_of(t1, b, nq));
    run_graph(n_tok, GGML_TYPE_F16, GGML_TYPE_F16, t2, q16, k16, v16, mask, NULL, NULL);
    printf("  all three (≈ref)   %14.10f\n", cos_of(t2, b, nq));

    if (dump_dir) {
        write_bin(dump_dir, "probe-q.bin", q, nq);
        write_bin(dump_dir, "probe-k.bin", k, nk);
        write_bin(dump_dir, "probe-v.bin", v, nk);
        char path[512]; snprintf(path, sizeof(path), "%s/probe-inputs.txt", dump_dir);
        FILE * f = fopen(path, "w"); if (f) { fprintf(f, "%d\n", n_tok); fclose(f); }
    }

    printf("gain=%.3g (score std ~%.3g)\n", gain, gain*gain); printf("n_tok=%d  head_dim=%d  n_head=%d  n_kv=%d  scale=1/%.4g\n", n_tok, D_HEAD, N_HEAD, N_KV, sqrtf((float) D_HEAD));
    printf("reference path = ggml_flash_attn_ext with f16 Q/K/V operands (flash_attn is on by default)\n");
    printf("f32 contract   = same op, K/V kept f32\n");
    printf("%-8s %16s %14s %16s\n", "query", "cos(ref,f32)", "maxabs", "cos(sn-loop,f32)");
    for (int t = 0; t < n_tok; t++) {
        const float * pa = a + (size_t) t * N_HEAD * D_HEAD;
        const float * pb = b + (size_t) t * N_HEAD * D_HEAD;
        const float * pr = r + (size_t) t * N_HEAD * D_HEAD;
        const size_t n = (size_t) N_HEAD * D_HEAD;
        printf("pos %-4d %14.10f %14.3e %14.12f\n", t, cos_of(pa, pb, n), maxabs(pa, pb, n), cos_of(pr, pb, n));
    }
    printf("whole-row cos(ref,f32) = %.10f\n", cos_of(a, b, nq));
    return 0;
}
