/* Tools/sn-oracle.c — per-layer hidden states from the reference model, offline.
 *
 * Phase 2 compares `sn ref` against the fork, but the server only exposes the *final* hidden state
 * (`--embeddings --pooling none`) and the next-token distribution. When the two forwards disagree,
 * that says "somewhere in 56 executions"; it does not say where. The C API does expose every layer
 * input (`llama_set_embeddings_layer_inp` / `llama_get_embeddings_layer_inp`, llama.h:1182 - the
 * same hook the speculative-decode driver uses), so this tool dumps
 *
 *     layer.<il>.p<t>.txt   the hidden state the reference feeds INTO trunk execution il, token t
 *     result_norm.p<t>.txt  the post-output_norm state (what --embeddings returns)
 *     logits.p<t>.txt       the raw logits for that token
 *
 * in the same one-value-per-line format as `sn ref --dump`, so Tools/oracle-check.py can pair them
 * against our `embd` / `l_out.<il>` / `h_final.<t>` dumps and report the first execution where the
 * two stop agreeing.
 *
 *   gcc -O1 -o Tools/sn-oracle Tools/sn-oracle.c \
 *     -I <fork>/include -L <fork>/build/bin -lllama -Wl,-rpath,<fork>/build/bin -lm
 *
 * usage: sn-oracle [--model M] [--out DIR] <token ids file>     # ids: any whitespace/comma split
 */
#include <llama.h>
#include <ggml.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define LLAMA_ORACLE_MAX_LAYERS 1024

static void dump_vec(const char * path, const char * stage, int n, const float * v) {
    FILE * f = fopen(path, "w");
    if (!f) { fprintf(stderr, "sn-oracle: cannot write %s\n", path); return; }
    fprintf(f, "# stage %s shape %d n %d (reference, llama.cpp fork)\n", stage, n, n);
    for (int i = 0; i < n; i++) {
        fprintf(f, "%.9e\n", (double) v[i]);
    }
    fclose(f);
}

int main(int argc, char ** argv) {
    const char * model_path = "/home/justin/Documents/AI/Models/SenseLoop/SenseNova-Loop-9B-Q4_K_M.gguf";
    const char * out_dir = ".oracle";
    const char * tok_file = NULL;
    int ngl = 0;
    int n_threads = 0;
    int use_cpu = 1;
    int want_logits = 1;
    int layers_all = 1;
    const char * layers_spec = NULL;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--model") && i + 1 < argc)      { model_path = argv[++i]; }
        else if (!strcmp(argv[i], "--out") && i + 1 < argc)   { out_dir = argv[++i]; }
        else if (!strcmp(argv[i], "--ngl") && i + 1 < argc)   { ngl = atoi(argv[++i]); }
        else if (!strcmp(argv[i], "--threads") && i + 1 < argc) { n_threads = atoi(argv[++i]); }
        else if (!strcmp(argv[i], "--layers") && i + 1 < argc) { layers_all = 0; layers_spec = argv[++i]; }
        else if (!strcmp(argv[i], "--gpu")) { use_cpu = 0; }
        else if (!strcmp(argv[i], "--no-logits")) { want_logits = 0; }
        else if (tok_file) { fprintf(stderr, "sn-oracle: too many arguments\n"); return 2; }
        else { tok_file = argv[i]; }
    }
    if (!tok_file) {
        fprintf(stderr, "usage: sn-oracle [--model M] [--out DIR] [--ngl N] [--threads N] <tokens file>\n");
        return 2;
    }

    /* read the token ids (comma or whitespace separated - the corpus .ids files use commas) */
    llama_token toks[512];
    int n = 0;
    FILE * f = fopen(tok_file, "r");
    if (!f) { fprintf(stderr, "sn-oracle: cannot read %s\n", tok_file); return 1; }
    long v;
    while (n < 512 && fscanf(f, " %ld", &v) == 1) {
        toks[n++] = (llama_token) v;
        int c = fgetc(f);
        while (c != EOF && (c == ',' || c == ';')) { c = fgetc(f); }
        if (c != EOF) { ungetc(c, f); }
    }
    fclose(f);
    if (!n) { fprintf(stderr, "sn-oracle: no token ids in %s\n", tok_file); return 1; }

    /* keep logs: the abort reason is the point */

    struct llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = ngl;
    if (use_cpu) {
        /* one backend, so every graph node has an unambiguous owner: the layer-input extraction in
         * llama-context.cpp:2807 asserts on a node the scheduler never assigned, which is what a
         * split ROCm0/ROCm_Host graph does to these pass-through tensors. */
        static ggml_backend_dev_t devs[2];
        devs[0] = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
        devs[1] = NULL;
        if (!devs[0]) { fprintf(stderr, "sn-oracle: no CPU device\n"); return 1; }
        mp.devices = devs;
    }
    struct llama_model * model = llama_model_load_from_file(model_path, mp);
    if (!model) { fprintf(stderr, "sn-oracle: failed to load %s\n", model_path); return 1; }

    const int n_layer = llama_model_n_layer(model);
    const int n_embd  = llama_model_n_embd(model);
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    fprintf(stderr, "sn-oracle: %d tokens, trunk %d layers, n_embd %d, n_vocab %d\n", n, n_layer, n_embd, n_vocab);

    struct llama_context_params cp = llama_context_default_params();
    cp.n_ctx      = (uint32_t) n + 8;
    cp.n_batch    = (uint32_t) n;
    cp.n_ubatch   = (uint32_t) n;
    cp.n_seq_max  = 1;
    cp.n_threads  = n_threads > 0 ? n_threads : cp.n_threads;
    cp.n_threads_batch = cp.n_threads;
    cp.embeddings = true;
    cp.pooling_type = LLAMA_POOLING_TYPE_NONE;

    struct llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) { fprintf(stderr, "sn-oracle: failed to create context\n"); llama_model_free(model); return 1; }

    unsigned char want[LLAMA_ORACLE_MAX_LAYERS];
    memset(want, 0, sizeof want);
    for (int il = 0; il < n_layer; il++) {
        if (layers_all) { want[il] = 1; continue; }
        /* --layers 0-5,8 */
        const char * p = layers_spec;
        while (p && *p) {
            int a = (int) strtol(p, (char **) &p, 10);
            int b = a;
            if (*p == '-') { b = (int) strtol(p + 1, (char **) &p, 10); }
            if (il >= a && il <= b && il < (int) sizeof want) { want[il] = 1; }
            if (*p == ',') { p++; }
        }
    }
    for (int il = 0; il < n_layer; il++) {
        if (want[il]) {
            llama_set_embeddings_layer_inp(ctx, (uint32_t) il, true);
        }
    }

    struct llama_batch batch = llama_batch_init(n, 0, 1);
    for (int i = 0; i < n; i++) {
        batch.token[i]    = toks[i];
        batch.pos[i]      = i;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i]   = 1;  /* every position is an output position */
    }
    batch.n_tokens = n;

    if (llama_decode(ctx, batch) != 0) {
        fprintf(stderr, "sn-oracle: llama_decode failed\n");
        llama_batch_free(batch);
        llama_free(ctx); llama_model_free(model);
        return 1;
    }

    for (int t = 0; t < n; t++) {
        char path[512], stage[96];
        for (int il = 0; il < n_layer; il++) {
            if (!want[il]) { continue; }
            float * p = llama_get_embeddings_layer_inp(ctx, (uint32_t) il);
            if (!p) { fprintf(stderr, "sn-oracle: layer %d has no data\n", il); continue; }
            snprintf(path, sizeof path, "%s/layer.%d.p%d.txt", out_dir, il, t);
            snprintf(stage, sizeof stage, "layer.%d", il);
            dump_vec(path, stage, n_embd, p + (size_t) t * n_embd);
        }
        float * e = llama_get_embeddings_ith(ctx, t);
        if (e) {
            snprintf(path, sizeof path, "%s/result_norm.p%d.txt", out_dir, t);
            dump_vec(path, "result_norm", n_embd, e);
        } else {
            fprintf(stderr, "sn-oracle: no embeddings for token %d\n", t);
        }
        float * lg = want_logits ? llama_get_logits_ith(ctx, t) : NULL;
        if (lg) {
            snprintf(path, sizeof path, "%s/logits.p%d.txt", out_dir, t);
            dump_vec(path, "logits", n_vocab, lg);
        } else if (want_logits) {
            fprintf(stderr, "sn-oracle: no logits for token %d\n", t);
        }
    }

    llama_batch_free(batch);
    llama_free(ctx);
    llama_model_free(model);
    fprintf(stderr, "sn-oracle: wrote %s\n", out_dir);
    return 0;
}
