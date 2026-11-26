/* Tools/tok.c — tokenise text with the checkpoint's own tokenizer, offline.
 *
 * The reference server's /tokenize route answers 501 in this fork's build, and SuperNova's own GGUF
 * BPE tokenizer is Phase 10 work (Docs/design-plan.md §7). So the Phase 2 golden corpus is
 * tokenised through the llama.cpp C API from this small tool: it links the already-built shared
 * libraries and does no inference, so it costs a model load and nothing else.
 *
 *   gcc -O1 -o Tools/tok Tools/tok.c \
 *     -I <fork>/include -L <fork>/build/bin -lllama -Wl,-rpath,<fork>/build/bin -lm
 *
 * usage: Tools/tok <model.gguf> [text file, default stdin]   -> token ids, one per line
 *        Tools/tok --ids <model.gguf> <text file>            -> ids on one line, comma separated
 */
#include <llama.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char ** argv) {
    int i = 1;
    int one_line = 0;
    if (i < argc && strcmp(argv[i], "--ids") == 0) { one_line = 1; i++; }
    if (i + 1 > argc) {
        fprintf(stderr, "usage: tok [--ids] <model.gguf> [text file]\n");
        return 2;
    }
    const char * path = argv[i++];
    if (i >= argc) {
        fprintf(stderr, "tok: no input file given\n");
        return 2;
    }

    struct llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0; /* tokenizer only */
    struct llama_model * model = llama_model_load_from_file(path, mp);
    if (!model) { fprintf(stderr, "failed to load %s\n", path); return 1; }
    const struct llama_vocab * vocab = llama_model_get_vocab(model);

    /* one model load, any number of input files: one output line (or block) per file, in order */
    for (; i < argc; i++) {
        FILE * f = fopen(argv[i], "rb");
        if (!f) { perror(argv[i]); return 1; }
        size_t len = 0, cap = 1 << 16;
        char * text = malloc(cap); /* must be allocated before the first fread */
        if (!text) { fprintf(stderr, "out of memory\n"); return 1; }
        for (;;) {
            if (len == cap) { cap *= 2; text = realloc(text, cap); }
            size_t r = fread(text + len, 1, cap - len, f);
            len += r;
            if (r == 0) break;
        }
        fclose(f);

        int n_max = (int) len * 4 + 1024; /* byte-fallback BPE never needs more than this */
        llama_token * toks = malloc(sizeof(llama_token) * n_max);
        /* add_special = false, parse_special = true: the same call the server makes for a raw
         * prompt (tools/server/server-common.cpp:1100 common_tokenize(vocab, s, false, true)) */
        int n = llama_tokenize(vocab, text, (int) len, toks, n_max, false, true);
        if (n < 0) { fprintf(stderr, "llama_tokenize failed for %s: %d\n", argv[i], n); return 1; }
        for (int j = 0; j < n; j++) {
            if (one_line) printf("%s%d", j ? "," : "", toks[j]);
            else printf("%d\n", toks[j]);
        }
        if (one_line) printf("\n");
        fprintf(stderr, "%s: %d tokens\n", argv[i], n);
        free(toks);
        free(text);
    }
    llama_model_free(model);
    return 0;
}
