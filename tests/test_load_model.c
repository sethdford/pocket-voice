/**
 * test_load_model.c — Smoke test: load a real .cstt model and run inference.
 *
 * Usage: ./test_load_model model.cstt
 */

#include "conformer_stt.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <model.cstt>\n", argv[0]);
        return 1;
    }

    printf("Loading model: %s\n", argv[1]);
    ConformerSTT *stt = conformer_stt_create(argv[1]);
    if (!stt) {
        printf("FAIL: Could not load model\n");
        return 1;
    }

    printf("Model loaded successfully!\n");
    printf("  Sample rate: %d\n", conformer_stt_sample_rate(stt));
    printf("  D model:     %d\n", conformer_stt_d_model(stt));
    printf("  Layers:      %d\n", conformer_stt_n_layers(stt));
    printf("  Vocab size:  %d\n", conformer_stt_vocab_size(stt));

    /* Generate 1 second of silence + a brief tone, process it */
    int sr = conformer_stt_sample_rate(stt);
    int n_samples = sr;
    float *pcm = (float *)calloc(n_samples, sizeof(float));
    if (!pcm) { conformer_stt_destroy(stt); return 1; }

    /* Add a brief 440Hz sine burst (to make it non-trivial) */
    for (int i = 0; i < sr / 4; i++)
        pcm[i] = 0.3f * sinf(2.0f * (float)M_PI * 440.0f * (float)i / (float)sr);

    printf("\nProcessing 1 second of test audio...\n");
    int result = conformer_stt_process(stt, pcm, n_samples);
    printf("  Process returned: %d\n", result);

    /* Flush */
    int flush_result = conformer_stt_flush(stt);
    printf("  Flush returned: %d\n", flush_result);

    /* Get transcript */
    char text[4096];
    int text_len = conformer_stt_get_text(stt, text, sizeof(text));
    printf("  Transcript (%d chars): \"%s\"\n", text_len, text);

    printf("\nPASS: Model loaded and inference ran successfully.\n");

    free(pcm);
    conformer_stt_destroy(stt);
    return 0;
}
