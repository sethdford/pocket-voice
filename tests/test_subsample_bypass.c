/**
 * test_subsample_bypass.c — Feed NeMo's subsampling output directly
 * into C encoder blocks, bypassing C subsampling entirely.
 *
 * This isolates: are the encoder blocks correct when given exact NeMo input?
 */

#include "conformer_stt.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <model.cstt> <nemo_pre_encode.bin>\n", argv[0]);
        return 1;
    }

    ConformerSTT *stt = conformer_stt_create(argv[1]);
    if (!stt) { printf("FAIL: Could not load model\n"); return 1; }

    int D = conformer_stt_d_model(stt);

    struct stat st;
    stat(argv[2], &st);
    int T_sub = (int)(st.st_size / (D * sizeof(float)));
    float *sub_out = (float *)malloc(st.st_size);
    FILE *f = fopen(argv[2], "rb");
    fread(sub_out, sizeof(float), T_sub * D, f);
    fclose(f);

    printf("NeMo subsampling output: T=%d, D=%d\n", T_sub, D);
    printf("  frame[0][:8]: ");
    for (int i = 0; i < 8; i++) printf("%.2f ", sub_out[i]);
    printf("\n  frame[1][:8]: ");
    for (int i = 0; i < 8; i++) printf("%.2f ", sub_out[D + i]);
    printf("\n");

    float mn = sub_out[0], mx = sub_out[0], sm = 0;
    for (int i = 0; i < T_sub * D; i++) {
        if (sub_out[i] < mn) mn = sub_out[i];
        if (sub_out[i] > mx) mx = sub_out[i];
        sm += sub_out[i];
    }
    printf("  min=%.4f max=%.4f mean=%.6f\n\n", mn, mx, sm / (T_sub * D));

    printf("Running encoder blocks + CTC with NeMo's exact subsampling output...\n");
    int n_chars = conformer_stt_forward_subsample_output(stt, sub_out, T_sub);
    printf("Forward returned: %d chars\n", n_chars);

    char text[16384];
    conformer_stt_get_text(stt, text, sizeof(text));
    printf("\nTranscript: \"%s\"\n", text);
    printf("Expected:   \"The quick brown fox jumps over the lazy dog.\"\n");

    if (strstr(text, "quick") || strstr(text, "brown") || strstr(text, "fox"))
        printf("\nResult: MATCH\n");
    else if (strlen(text) > 0)
        printf("\nResult: PARTIAL (%d chars)\n", (int)strlen(text));
    else
        printf("\nResult: EMPTY (encoder blocks issue)\n");

    free(sub_out);
    conformer_stt_destroy(stt);
    return 0;
}
