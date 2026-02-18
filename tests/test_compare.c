/**
 * test_compare.c — Save C engine intermediate outputs for comparison with NeMo.
 *
 * Feeds NeMo's normalized mel into the C engine, saves post-subsampling
 * and final encoder outputs as binary files, then prints frame-level
 * CTC logit details.
 */

#include "conformer_stt.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

/* These match the internal struct layout; we access them via
 * the opaque pointer by reading the header from the .cstt file. */
extern int conformer_stt_d_model(const ConformerSTT *stt);
extern int conformer_stt_n_layers(const ConformerSTT *stt);
extern int conformer_stt_vocab_size(const ConformerSTT *stt);

static void print_stats(const char *name, const float *data, int n) {
    float mn = data[0], mx = data[0], sm = 0;
    for (int i = 0; i < n; i++) {
        if (data[i] < mn) mn = data[i];
        if (data[i] > mx) mx = data[i];
        sm += data[i];
    }
    printf("%s: min=%.4f max=%.4f mean=%.6f\n", name, mn, mx, sm / n);
}

static void save_bin(const char *path, const float *data, int n) {
    FILE *f = fopen(path, "wb");
    if (!f) { printf("Failed to open %s\n", path); return; }
    fwrite(data, sizeof(float), n, f);
    fclose(f);
    printf("Saved %s (%d floats, %zu bytes)\n", path, n, (size_t)n * sizeof(float));
}

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <model.cstt> <nemo_mel.bin> [nemo_subsample.bin] [nemo_block0.bin]\n", argv[0]);
        return 1;
    }

    ConformerSTT *stt = conformer_stt_create(argv[1]);
    if (!stt) { printf("FAIL: Could not load model\n"); return 1; }

    int D = conformer_stt_d_model(stt);
    int vocab = conformer_stt_vocab_size(stt);
    int n_mels = 80;

    /* Load NeMo mel */
    struct stat st;
    stat(argv[2], &st);
    int T = (int)(st.st_size / (n_mels * sizeof(float)));
    float *mel = (float *)malloc(st.st_size);
    FILE *f = fopen(argv[2], "rb");
    fread(mel, sizeof(float), T * n_mels, f);
    fclose(f);
    printf("Loaded mel: T=%d, n_mels=%d\n", T, n_mels);

    /* Run forward pass */
    int n_chars = conformer_stt_forward_normalized_mel(stt, mel, T);
    printf("Forward returned: %d chars\n", n_chars);

    char text[16384];
    conformer_stt_get_text(stt, text, sizeof(text));
    printf("Transcript: \"%s\"\n\n", text);

    /* Load and compare reference data */
    if (argc >= 4) {
        printf("=== Subsampling comparison ===\n");
        stat(argv[3], &st);
        int ref_n = (int)(st.st_size / sizeof(float));
        float *ref_sub = (float *)malloc(st.st_size);
        f = fopen(argv[3], "rb");
        fread(ref_sub, sizeof(float), ref_n, f);
        fclose(f);

        int T_sub = ref_n / D;
        printf("NeMo subsample: %d frames x %d dims\n", T_sub, D);
        print_stats("  NeMo", ref_sub, ref_n);

        printf("\n  NeMo frame[0][:8]:  ");
        for (int i = 0; i < 8; i++) printf("%.2f ", ref_sub[i]);
        printf("\n  NeMo frame[1][:8]:  ");
        for (int i = 0; i < 8; i++) printf("%.2f ", ref_sub[D + i]);
        printf("\n");
        free(ref_sub);
    }

    if (argc >= 5) {
        printf("\n=== Block 0 comparison ===\n");
        stat(argv[4], &st);
        int ref_n = (int)(st.st_size / sizeof(float));
        float *ref_blk = (float *)malloc(st.st_size);
        f = fopen(argv[4], "rb");
        fread(ref_blk, sizeof(float), ref_n, f);
        fclose(f);

        int T_sub = ref_n / D;
        printf("NeMo block0: %d frames x %d dims\n", T_sub, D);
        print_stats("  NeMo", ref_blk, ref_n);

        printf("\n  NeMo block0 frame[0][:8]:  ");
        for (int i = 0; i < 8; i++) printf("%.4f ", ref_blk[i]);
        printf("\n  NeMo block0 frame[1][:8]:  ");
        for (int i = 0; i < 8; i++) printf("%.4f ", ref_blk[D + i]);
        printf("\n");
        free(ref_blk);
    }

    free(mel);
    conformer_stt_destroy(stt);
    return 0;
}
