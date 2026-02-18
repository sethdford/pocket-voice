/**
 * test_with_nemo_mel.c — Feed NeMo's normalized mel directly into C engine.
 *
 * Tests the encoder + CTC head in isolation, bypassing mel extraction
 * and normalization. This definitively answers: is the encoder correct?
 */

#include "conformer_stt.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <model.cstt> <nemo_mel.bin>\n", argv[0]);
        return 1;
    }

    ConformerSTT *stt = conformer_stt_create(argv[1]);
    if (!stt) { printf("FAIL: Could not load model\n"); return 1; }

    int n_mels = 80;
    struct stat st;
    stat(argv[2], &st);
    int T = (int)(st.st_size / (n_mels * sizeof(float)));
    float *mel = (float *)malloc(st.st_size);
    FILE *f = fopen(argv[2], "rb");
    fread(mel, sizeof(float), T * n_mels, f);
    fclose(f);

    float mn = mel[0], mx = mel[0], sm = 0;
    for (int i = 0; i < T * n_mels; i++) {
        if (mel[i] < mn) mn = mel[i];
        if (mel[i] > mx) mx = mel[i];
        sm += mel[i];
    }
    printf("NeMo mel: T=%d, n_mels=%d\n", T, n_mels);
    printf("  min=%.4f max=%.4f mean=%.6f\n", mn, mx, sm / (T * n_mels));
    printf("  Frame 0 bins 0-9: ");
    for (int i = 0; i < 10; i++) printf("%.4f ", mel[i]);
    printf("\n");

    printf("\nRunning forward pass with NeMo's exact mel features...\n");
    int n_chars = conformer_stt_forward_normalized_mel(stt, mel, T);
    printf("Forward returned: %d chars\n", n_chars);

    char text[16384];
    conformer_stt_get_text(stt, text, sizeof(text));
    printf("\nTranscript: \"%s\"\n", text);

    printf("\nExpected:   \"The quick brown fox jumps over the lazy dog.\"\n");

    if (strstr(text, "quick") || strstr(text, "brown") || strstr(text, "fox"))
        printf("\nResult: PARTIAL MATCH (some correct words detected)\n");
    else if (strlen(text) == 0)
        printf("\nResult: EMPTY (encoder or CTC head issue)\n");

    free(mel);
    conformer_stt_destroy(stt);
    return 0;
}
