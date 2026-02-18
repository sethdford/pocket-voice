/**
 * test_full_forward.c — Run the full forward pass on entire audio at once.
 *
 * Bypasses chunked processing to test the model with proper per-feature
 * normalization over the complete utterance.
 */

#include "conformer_stt.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

static float *load_pcm(const char *path, int *n_samples) {
    struct stat st;
    if (stat(path, &st) != 0) return NULL;
    *n_samples = (int)(st.st_size / sizeof(float));
    float *buf = (float *)malloc(st.st_size);
    FILE *f = fopen(path, "rb");
    if (!f) { free(buf); return NULL; }
    fread(buf, sizeof(float), *n_samples, f);
    fclose(f);
    return buf;
}

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <model.cstt> <audio.pcm>\n", argv[0]);
        return 1;
    }

    ConformerSTT *stt = conformer_stt_create(argv[1]);
    if (!stt) { printf("FAIL: Could not load model\n"); return 1; }

    int n_samples = 0;
    float *pcm = load_pcm(argv[2], &n_samples);
    if (!pcm) { fprintf(stderr, "Cannot load: %s\n", argv[2]); return 1; }
    printf("Audio: %d samples (%.1fs)\n", n_samples, (float)n_samples / conformer_stt_sample_rate(stt));

    /* Feed ALL audio at once to get proper per-feature normalization */
    int result = conformer_stt_process(stt, pcm, n_samples);
    printf("Process returned: %d chars\n", result);

    int flush_result = conformer_stt_flush(stt);
    printf("Flush returned: %d chars\n", flush_result);

    char text[16384];
    conformer_stt_get_text(stt, text, sizeof(text));
    printf("\nTranscript: \"%s\"\n", text);

    free(pcm);
    conformer_stt_destroy(stt);
    return 0;
}
