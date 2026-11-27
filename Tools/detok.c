/* Tools/detok.c — turn the token ids `sn run` prints back into text.

 * `sn run` emits ids because SuperNova's own GGUF BPE tokenizer is Phase 10 work
 * (Docs/design-plan.md §7); this is the reading-half of Tools/tok.c, and like tok.c it links the
 * fork's shared library and does no inference.
 *
 *   gcc -O1 -o Tools/detok Tools/detok.c \
 *     -I <fork>/include -I <fork>/ggml/include -L <fork>/build/bin -lllama \
 *     -Wl,-rpath,<fork>/build/bin -lm
 *
 * usage: Tools/detok <model.gguf> [ids file, default stdin]   -> the pieces concatenated
 *
 * Feed it either a plain id list (whitespace / comma separated) or the two shapes `sn run` prints:
 * a `continuation: 11751 13 561 ...` line, or `pos 5 -> 11751` trace lines (on a line that has an
 * arrow, only the id to the right of it counts, so the position numbers stay out of the text).
 */
#include <llama.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char ** argv) {
    int i = 1;
    if (i >= argc) {
        fprintf(stderr, "usage: detok <model.gguf> [ids file]\n");
        return 2;
    }
    const char * path = argv[i++];

    struct llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0; /* tokenizer only */
    struct llama_model * model = llama_model_load_from_file(path, mp);
    if (!model) { fprintf(stderr, "failed to load %s\n", path); return 1; }
    const struct llama_vocab * vocab = llama_model_get_vocab(model);

    char line[1 << 16];
    FILE * f = stdin;
    if (i < argc) {
        f = fopen(argv[i], "rb");
        if (!f) { perror(argv[i]); return 1; }
    }
    char piece[256];
    while (fgets(line, sizeof line, f)) {
        /* "pos 5 -> 11751": the sample is everything after the arrow, the rest is bookkeeping. */
        char * arrow = strstr(line, "->");
        char * p = arrow ? arrow + 2 : line;
        for (; *p; ) {
            if (*p < '0' || *p > '9') { p++; continue; }
            long id = strtol(p, &p, 10);
            int n = llama_token_to_piece(vocab, (llama_token) id, piece, sizeof piece, 0, true);
            if (n > 0) fwrite(piece, 1, n, stdout);
        }
    }
    if (f != stdin) fclose(f);
    printf("\n");
    llama_model_free(model);
    return 0;
}
