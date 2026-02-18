/**
 * test_debug_forward.c — Debug the forward pass by printing intermediate stats.
 *
 * Loads a model, feeds audio, and prints statistics at each stage:
 * mel features, post-subsampling, per-block output, and final logits.
 */

#include "conformer_stt.h"
#include "mel_spectrogram.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static void print_stats(const char *label, const float *data, int n) {
    if (n <= 0) { printf("  %s: (empty)\n", label); return; }
    float mn = data[0], mx = data[0], sum = 0, sum2 = 0;
    for (int i = 0; i < n; i++) {
        if (data[i] < mn) mn = data[i];
        if (data[i] > mx) mx = data[i];
        sum += data[i];
        sum2 += data[i] * data[i];
    }
    float mean = sum / n;
    float var = sum2 / n - mean * mean;
    printf("  %s: min=%.6f max=%.6f mean=%.6f std=%.6f\n",
           label, mn, mx, mean, sqrtf(var > 0 ? var : 0));
}

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
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <model.cstt> [audio.pcm]\n", argv[0]);
        return 1;
    }

    /* Load model */
    ConformerSTT *stt = conformer_stt_create(argv[1]);
    if (!stt) { printf("FAIL: Could not load model\n"); return 1; }

    int sr = conformer_stt_sample_rate(stt);
    int n_mels_cfg = 80;

    /* Load or generate audio */
    float *pcm = NULL;
    int n_samples = 0;

    if (argc >= 3) {
        pcm = load_pcm(argv[2], &n_samples);
        if (!pcm) { fprintf(stderr, "Cannot load: %s\n", argv[2]); return 1; }
    } else {
        n_samples = sr * 2;
        pcm = (float *)calloc(n_samples, sizeof(float));
        for (int i = sr/4; i < sr; i++)
            pcm[i] = 0.5f * sinf(2.0f * M_PI * 300.0f * (float)i / sr);
    }

    printf("Audio: %d samples (%.1fs at %dHz)\n", n_samples, (float)n_samples/sr, sr);
    print_stats("PCM input", pcm, n_samples);

    /* Step 1: Extract mel features manually */
    MelConfig mel_cfg;
    mel_config_default(&mel_cfg);
    mel_cfg.sample_rate = sr;
    mel_cfg.n_mels = n_mels_cfg;
    MelSpectrogram *mel = mel_create(&mel_cfg);

    int max_frames = n_samples / mel_hop_length(mel) + 2;
    float *mel_out = (float *)calloc((size_t)max_frames * n_mels_cfg, sizeof(float));
    int n_frames = mel_process(mel, pcm, n_samples, mel_out, max_frames);

    printf("\nMel spectrogram: %d frames x %d bins\n", n_frames, n_mels_cfg);
    print_stats("Mel features", mel_out, n_frames * n_mels_cfg);

    /* Print first frame */
    if (n_frames > 0) {
        printf("  Frame 0 (first 10 bins): ");
        for (int i = 0; i < 10 && i < n_mels_cfg; i++)
            printf("%.3f ", mel_out[i]);
        printf("...\n");
    }

    mel_destroy(mel);

    /* Step 2: Run through the full STT engine */
    printf("\nRunning full forward pass...\n");
    int result = conformer_stt_process(stt, pcm, n_samples);
    printf("  Process returned: %d new chars\n", result);

    int flush_result = conformer_stt_flush(stt);
    printf("  Flush returned: %d new chars\n", flush_result);

    char text[4096];
    conformer_stt_get_text(stt, text, sizeof(text));
    printf("  Transcript: \"%s\"\n", text);

    printf("\nDone.\n");
    free(pcm);
    free(mel_out);
    conformer_stt_destroy(stt);
    return 0;
}
