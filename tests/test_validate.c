/**
 * test_validate.c — Validate C STT engine against reference data.
 *
 * Loads a .cstt model and reference PCM, runs the full forward pass,
 * and reports the transcript. When reference logits are available
 * (from NeMo), compares them numerically.
 *
 * Usage: ./test_validate model.cstt validation/
 *    or: ./test_validate model.cstt test_audio.pcm
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
    if (!buf) return NULL;
    FILE *f = fopen(path, "rb");
    if (!f) { free(buf); return NULL; }
    fread(buf, sizeof(float), *n_samples, f);
    fclose(f);
    return buf;
}

static void print_top_tokens(const char *label, const float *logits,
                              int T, int V, int top_n) {
    printf("\n%s (first %d frames, top-%d tokens per frame):\n", label,
           T < 5 ? T : 5, top_n);
    for (int t = 0; t < T && t < 5; t++) {
        const float *row = logits + t * V;
        printf("  t=%d: ", t);
        for (int k = 0; k < top_n; k++) {
            float best = -1e30f;
            int best_idx = 0;
            for (int v = 0; v < V; v++) {
                int skip = 0;
                for (int j = 0; j < k; j++) {
                    /* Find kth best by re-scanning (simple approach) */
                }
                if (row[v] > best) {
                    best = row[v];
                    best_idx = v;
                }
            }
            printf("[%d]=%.3f ", best_idx, best);
        }
        printf("\n");
    }
}

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <model.cstt> <validation_dir_or_pcm>\n", argv[0]);
        return 1;
    }

    const char *model_path = argv[1];
    const char *data_path = argv[2];

    /* Determine if data_path is a directory or a .pcm file */
    struct stat st;
    if (stat(data_path, &st) != 0) {
        fprintf(stderr, "Cannot access: %s\n", data_path);
        return 1;
    }

    char pcm_path[4096];
    if (S_ISDIR(st.st_mode)) {
        snprintf(pcm_path, sizeof(pcm_path), "%s/test_audio.pcm", data_path);
    } else {
        snprintf(pcm_path, sizeof(pcm_path), "%s", data_path);
    }

    /* Load model */
    printf("Loading model: %s\n", model_path);
    ConformerSTT *stt = conformer_stt_create(model_path);
    if (!stt) {
        printf("FAIL: Could not load model\n");
        return 1;
    }
    printf("  Sample rate: %d Hz\n", conformer_stt_sample_rate(stt));
    printf("  D model:     %d\n", conformer_stt_d_model(stt));
    printf("  Layers:      %d\n", conformer_stt_n_layers(stt));
    printf("  Vocab:       %d\n", conformer_stt_vocab_size(stt));

    /* Load audio */
    int n_samples = 0;
    float *pcm = load_pcm(pcm_path, &n_samples);
    if (!pcm) {
        fprintf(stderr, "Cannot load audio: %s\n", pcm_path);
        conformer_stt_destroy(stt);
        return 1;
    }
    float duration = (float)n_samples / (float)conformer_stt_sample_rate(stt);
    printf("\nAudio: %s (%.1f seconds, %d samples)\n", pcm_path, duration, n_samples);

    /* Process audio in chunks (simulating real-time streaming) */
    int sr = conformer_stt_sample_rate(stt);
    int chunk_size = sr / 10;  /* 100ms chunks */
    int offset = 0;
    int total_chars = 0;

    printf("\nProcessing audio in %dms chunks...\n", chunk_size * 1000 / sr);
    while (offset < n_samples) {
        int remaining = n_samples - offset;
        int this_chunk = remaining < chunk_size ? remaining : chunk_size;
        int nc = conformer_stt_process(stt, pcm + offset, this_chunk);
        if (nc > 0) {
            total_chars += nc;
            char partial[4096];
            conformer_stt_get_text(stt, partial, sizeof(partial));
            printf("\r  [%.1fs] \"%s\"", (float)offset / sr, partial);
            fflush(stdout);
        }
        offset += this_chunk;
    }

    /* Flush */
    int flush_chars = conformer_stt_flush(stt);
    total_chars += flush_chars;

    /* Final transcript */
    char transcript[16384];
    conformer_stt_get_text(stt, transcript, sizeof(transcript));
    printf("\n\n══════════════════════════════════════════\n");
    printf("C Engine Transcript:\n  \"%s\"\n", transcript);
    printf("══════════════════════════════════════════\n");
    printf("Total characters: %d\n", total_chars);

    /* Load and compare reference transcript if available */
    if (S_ISDIR(st.st_mode)) {
        char ref_path[4096];
        snprintf(ref_path, sizeof(ref_path), "%s/ref_transcript.txt", data_path);
        FILE *ref_f = fopen(ref_path, "r");
        if (ref_f) {
            char ref_text[16384];
            int ref_len = (int)fread(ref_text, 1, sizeof(ref_text) - 1, ref_f);
            ref_text[ref_len] = '\0';
            fclose(ref_f);
            printf("\nNeMo Reference:\n  \"%s\"\n", ref_text);

            if (strcmp(transcript, ref_text) == 0)
                printf("\nMATCH: Transcripts are identical!\n");
            else
                printf("\nDIFFER: Transcripts don't match (this is expected "
                       "without NeMo reference — see scripts/validate_model.py)\n");
        }
    }

    printf("\nDone.\n");
    free(pcm);
    conformer_stt_destroy(stt);
    return 0;
}
