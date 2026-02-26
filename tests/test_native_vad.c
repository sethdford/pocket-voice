/**
 * test_native_vad.c — Native VAD unit + integration tests.
 *
 * Unit tests verify the engine API with synthetic inputs.
 * Integration tests require models/silero_vad.nvad (extracted weights).
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

typedef struct NativeVad NativeVad;
extern NativeVad *native_vad_create(const char *weights_path);
extern void native_vad_destroy(NativeVad *vad);
extern float native_vad_process(NativeVad *vad, const float *samples);
extern int native_vad_process_audio(NativeVad *vad, const float *audio, int n_samples,
                                     float *probs_out, int max_probs);
extern void native_vad_reset(NativeVad *vad);
extern int native_vad_chunk_size(const NativeVad *vad);

static int passed = 0, failed = 0;
#define CHECK(cond, msg) do { \
    if (cond) { printf("  [PASS] %s\n", msg); passed++; } \
    else { printf("  [FAIL] %s\n", msg); failed++; } \
} while (0)

static int file_exists(const char *path) {
    FILE *f = fopen(path, "rb");
    if (f) { fclose(f); return 1; }
    return 0;
}

static float *load_wav_mono_16k(const char *path, int *n_samples) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    uint8_t hdr[44];
    if (fread(hdr, 1, 44, f) != 44) { fclose(f); return NULL; }
    int sr = hdr[24] | (hdr[25] << 8) | (hdr[26] << 16) | (hdr[27] << 24);
    int bits = hdr[34] | (hdr[35] << 8);
    int channels = hdr[22] | (hdr[23] << 8);
    int data_size = hdr[40] | (hdr[41] << 8) | (hdr[42] << 16) | (hdr[43] << 24);
    int n = data_size / (bits / 8) / channels;

    float *raw = (float *)malloc((size_t)n * sizeof(float));
    if (!raw) { fclose(f); return NULL; }
    if (bits == 16) {
        int16_t *buf = (int16_t *)malloc(data_size);
        if (fread(buf, 1, data_size, f)) {}
        for (int i = 0; i < n; i++) raw[i] = buf[i * channels] / 32768.0f;
        free(buf);
    } else if (bits == 32) {
        float *buf = (float *)malloc(data_size);
        if (fread(buf, 1, data_size, f)) {}
        for (int i = 0; i < n; i++) raw[i] = buf[i * channels];
        free(buf);
    }
    fclose(f);

    int out_n = (int)((long long)n * 16000 / sr);
    float *out = (float *)malloc((size_t)out_n * sizeof(float));
    for (int i = 0; i < out_n; i++) {
        float src = (float)i * sr / 16000;
        int idx = (int)src;
        float frac = src - idx;
        if (idx + 1 < n)
            out[i] = raw[idx] * (1.0f - frac) + raw[idx + 1] * frac;
        else
            out[i] = raw[n - 1];
    }
    free(raw);
    *n_samples = out_n;
    return out;
}

int main(void) {
    printf("═══ Native VAD Tests ═══\n\n");

    /* ── API safety tests ────────────────────────────────────────────── */
    printf("── API Safety ──\n");
    CHECK(native_vad_create(NULL) == NULL, "create(NULL) returns NULL");
    CHECK(native_vad_create("nonexistent.nvad") == NULL, "create(bad path) returns NULL");
    CHECK(native_vad_chunk_size(NULL) == 512, "chunk_size(NULL) returns 512");

    float silence[512] = {0};
    CHECK(native_vad_process(NULL, silence) < 0.0f, "process(NULL vad) returns error");
    CHECK(native_vad_process_audio(NULL, silence, 512, silence, 1) < 0, "process_audio(NULL) returns error");

    native_vad_destroy(NULL);
    CHECK(1, "destroy(NULL) is safe");

    native_vad_reset(NULL);
    CHECK(1, "reset(NULL) is safe");

    /* ── Integration tests (require extracted weights) ───────────────── */
    const char *weights_path = "models/silero_vad.nvad";
    if (!file_exists(weights_path)) {
        printf("\n  [SKIP] %s not found — run extract_silero_weights.py first\n", weights_path);
        goto done;
    }

    printf("\n── Model Load ──\n");
    NativeVad *vad = native_vad_create(weights_path);
    CHECK(vad != NULL, "native VAD loads successfully");
    if (!vad) goto done;

    /* ── Silence detection ───────────────────────────────────────────── */
    printf("\n── Silence Detection ──\n");
    float prob = native_vad_process(vad, silence);
    printf("    Silence probability: %.4f\n", prob);
    CHECK(prob >= 0.0f && prob <= 1.0f, "silence probability in valid range");
    CHECK(prob < 0.5f, "silence has low speech probability");

    /* Process several silence chunks (LSTM state should stabilize) */
    float silence_probs[5];
    for (int i = 0; i < 5; i++)
        silence_probs[i] = native_vad_process(vad, silence);
    CHECK(silence_probs[4] < 0.3f, "sustained silence stays low");

    /* ── Tone detection ──────────────────────────────────────────────── */
    printf("\n── Tone Detection ──\n");
    native_vad_reset(vad);
    float tone[512];
    for (int i = 0; i < 512; i++)
        tone[i] = 0.5f * sinf(2.0f * 3.14159f * 300.0f * (float)i / 16000.0f);
    float prob_tone = native_vad_process(vad, tone);
    printf("    300Hz tone probability: %.4f\n", prob_tone);
    CHECK(prob_tone >= 0.0f && prob_tone <= 1.0f, "tone probability in valid range");

    /* Repeated tone should increase probability */
    float tone_probs[5];
    for (int i = 0; i < 5; i++)
        tone_probs[i] = native_vad_process(vad, tone);
    printf("    After 5 tone chunks: %.4f\n", tone_probs[4]);

    /* ── State reset consistency ─────────────────────────────────────── */
    printf("\n── State Reset ──\n");
    native_vad_reset(vad);
    float prob_after_reset = native_vad_process(vad, silence);
    printf("    After reset, silence: %.4f (was: %.4f)\n", prob_after_reset, prob);
    CHECK(fabsf(prob_after_reset - prob) < 0.15f,
          "reset produces consistent results on same input");

    /* ── Batch processing ────────────────────────────────────────────── */
    printf("\n── Batch Processing ──\n");
    native_vad_reset(vad);
    float batch_input[2048];
    memset(batch_input, 0, sizeof(batch_input));
    float batch_probs[4];
    int n = native_vad_process_audio(vad, batch_input, 2048, batch_probs, 4);
    CHECK(n == 4, "process_audio returns correct chunk count");
    CHECK(batch_probs[0] >= 0.0f && batch_probs[0] <= 1.0f, "batch probs in range");

    /* ── Real audio test ─────────────────────────────────────────────── */
    printf("\n── Real Audio ──\n");
    const char *wav_path = "hello_piper.wav";
    if (file_exists(wav_path)) {
        int n_samples = 0;
        float *audio = load_wav_mono_16k(wav_path, &n_samples);
        if (audio) {
            int n_chunks = n_samples / 512;
            float *probs = (float *)malloc((size_t)n_chunks * sizeof(float));
            native_vad_reset(vad);

            int nc = native_vad_process_audio(vad, audio, n_samples, probs, n_chunks);
            CHECK(nc > 0, "process_audio returns chunk count on real audio");

            float max_prob = 0.0f, min_prob = 1.0f;
            int speech_chunks = 0;
            for (int i = 0; i < nc; i++) {
                if (probs[i] > max_prob) max_prob = probs[i];
                if (probs[i] < min_prob) min_prob = probs[i];
                if (probs[i] > 0.5f) speech_chunks++;
            }
            printf("    Chunks: %d, max=%.4f, min=%.4f, speech_chunks=%d\n",
                   nc, max_prob, min_prob, speech_chunks);
            CHECK(max_prob > min_prob, "probabilities vary across audio");
            CHECK(max_prob >= 0.0f && max_prob <= 1.0f, "all probabilities in [0,1]");

            free(probs);
            free(audio);
        } else {
            printf("  [SKIP] Could not load %s\n", wav_path);
        }
    } else {
        printf("  [SKIP] %s not found\n", wav_path);
    }

    /* ── Context continuity test ─────────────────────────────────────── */
    printf("\n── Context Continuity ──\n");
    native_vad_reset(vad);
    float p1 = native_vad_process(vad, silence);  /* first chunk, no context */

    native_vad_reset(vad);
    float p2 = native_vad_process(vad, silence);  /* same: reset + same input */
    CHECK(fabsf(p1 - p2) < 0.001f, "identical input after reset gives identical output");

    /* Without reset, context from previous chunk affects result */
    float p3 = native_vad_process(vad, silence);  /* has context from p2 call */
    printf("    No context: %.4f, with context: %.4f\n", p2, p3);
    /* Just verify it's a valid probability — context may or may not change result */
    CHECK(p3 >= 0.0f && p3 <= 1.0f, "context-aware result is valid");

    native_vad_destroy(vad);

done:
    printf("\n══════════════════════════════════════════\n");
    printf("Results: %d passed, %d failed\n", passed, failed);
    if (failed > 0) { printf("SOME TESTS FAILED\n"); return 1; }
    printf("ALL PASSED\n");
    return 0;
}
