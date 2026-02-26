/**
 * test_real_models.c — Integration tests with real ONNX models.
 *
 * Tests Speaker Encoder and Phonemizer with actual downloaded models
 * and real WAV audio files. Requires:
 *   - models/ecapa_tdnn.onnx
 *   - hello_piper.wav (TTS-generated "Hello" audio)
 *
 * Build: make test-real-models
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

/* ── Speaker Encoder FFI ────────────────────────────────────────────── */

typedef struct SpeakerEncoder SpeakerEncoder;
extern SpeakerEncoder *speaker_encoder_create(const char *model_path);
extern void speaker_encoder_destroy(SpeakerEncoder *enc);
extern int speaker_encoder_extract(SpeakerEncoder *enc, const float *audio,
                                    int n_samples, float *embedding_out);
extern int speaker_encoder_embedding_dim(const SpeakerEncoder *enc);
extern int speaker_encoder_extract_from_wav(SpeakerEncoder *enc,
                                             const char *wav_path,
                                             float *embedding_out);

/* ── Phonemizer FFI ─────────────────────────────────────────────────── */

typedef struct Phonemizer Phonemizer;
extern Phonemizer *phonemizer_create(const char *language);
extern void phonemizer_destroy(Phonemizer *ph);
extern int phonemizer_text_to_ipa(Phonemizer *ph, const char *text,
                                   char *ipa_out, int max_len);

/* ── Test Harness ───────────────────────────────────────────────────── */

static int passed = 0, failed = 0, skipped = 0;
#define CHECK(cond, msg) do { \
    if (cond) { printf("  [PASS] %s\n", msg); passed++; } \
    else { printf("  [FAIL] %s\n", msg); failed++; } \
} while (0)
#define SKIP(msg) do { printf("  [SKIP] %s\n", msg); skipped++; } while (0)

static int file_exists(const char *path) {
    FILE *f = fopen(path, "rb");
    if (f) { fclose(f); return 1; }
    return 0;
}

/* Simple WAV loader: returns float32 mono samples, sets *n_samples and *sample_rate */
static float *load_wav(const char *path, int *n_samples, int *sample_rate) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;

    uint8_t hdr[44];
    if (fread(hdr, 1, 44, f) != 44) { fclose(f); return NULL; }

    if (memcmp(hdr, "RIFF", 4) != 0 || memcmp(hdr + 8, "WAVE", 4) != 0) {
        fclose(f); return NULL;
    }

    int sr = hdr[24] | (hdr[25] << 8) | (hdr[26] << 16) | (hdr[27] << 24);
    int bits = hdr[34] | (hdr[35] << 8);
    int channels = hdr[22] | (hdr[23] << 8);
    int data_size = hdr[40] | (hdr[41] << 8) | (hdr[42] << 16) | (hdr[43] << 24);

    int n = data_size / (bits / 8) / channels;
    float *out = (float *)malloc(n * sizeof(float));
    if (!out) { fclose(f); return NULL; }

    if (bits == 16) {
        int16_t *buf = (int16_t *)malloc(data_size);
        if (!buf) { free(out); fclose(f); return NULL; }
        fread(buf, 1, data_size, f);
        for (int i = 0; i < n; i++)
            out[i] = buf[i * channels] / 32768.0f;
        free(buf);
    } else if (bits == 32) {
        float *buf = (float *)malloc(data_size);
        if (!buf) { free(out); fclose(f); return NULL; }
        fread(buf, 1, data_size, f);
        for (int i = 0; i < n; i++)
            out[i] = buf[i * channels];
        free(buf);
    } else {
        free(out); fclose(f); return NULL;
    }

    fclose(f);
    *n_samples = n;
    *sample_rate = sr;
    return out;
}

/* Simple linear resampling */
static float *resample(const float *in, int in_samples, int in_rate,
                        int out_rate, int *out_samples) {
    if (in_rate == out_rate) {
        float *copy = (float *)malloc(in_samples * sizeof(float));
        memcpy(copy, in, in_samples * sizeof(float));
        *out_samples = in_samples;
        return copy;
    }
    int n = (int)((long long)in_samples * out_rate / in_rate);
    float *out = (float *)malloc(n * sizeof(float));
    if (!out) return NULL;
    for (int i = 0; i < n; i++) {
        float src = (float)i * in_rate / out_rate;
        int idx = (int)src;
        float frac = src - idx;
        if (idx + 1 < in_samples)
            out[i] = in[idx] * (1.0f - frac) + in[idx + 1] * frac;
        else
            out[i] = in[in_samples - 1];
    }
    *out_samples = n;
    return out;
}

/* ── Tests ──────────────────────────────────────────────────────────── */

static void test_speaker_encoder(void) {
    printf("\n═══ Speaker Encoder (Real Model) ═══\n");

    const char *model_path = "models/ecapa_tdnn.onnx";
    if (!file_exists(model_path)) {
        SKIP("ECAPA-TDNN model not found — download to models/ecapa_tdnn.onnx");
        return;
    }

    SpeakerEncoder *enc = speaker_encoder_create(model_path);
    CHECK(enc != NULL, "ECAPA-TDNN loads successfully");
    if (!enc) return;

    int dim = speaker_encoder_embedding_dim(enc);
    printf("    Embedding dimension: %d\n", dim);
    CHECK(dim == 192, "embedding dim is 192 (ECAPA-TDNN standard)");

    /* Extract from WAV file */
    const char *wav1 = "hello_piper.wav";
    const char *wav2 = "cute_piper.wav";

    if (file_exists(wav1)) {
        float emb1[512] = {0};
        int ret = speaker_encoder_extract_from_wav(enc, wav1, emb1);
        CHECK(ret > 0, "embedding extracted from hello_piper.wav");

        if (ret > 0) {
            /* Check embedding is L2-normalized */
            float norm = 0.0f;
            for (int i = 0; i < dim; i++) norm += emb1[i] * emb1[i];
            norm = sqrtf(norm);
            printf("    Embedding L2 norm: %.4f (expect ~1.0)\n", norm);
            CHECK(fabsf(norm - 1.0f) < 0.05f, "embedding is L2-normalized");

            /* Check embedding is non-trivial (not all zeros) */
            float max_val = 0.0f;
            for (int i = 0; i < dim; i++)
                if (fabsf(emb1[i]) > max_val) max_val = fabsf(emb1[i]);
            CHECK(max_val > 0.01f, "embedding has non-trivial values");
        }

        /* Self-similarity test: same file should give same embedding */
        float emb1b[512] = {0};
        int ret2 = speaker_encoder_extract_from_wav(enc, wav1, emb1b);
        if (ret > 0 && ret2 > 0) {
            float cosine = 0.0f;
            for (int i = 0; i < dim; i++) cosine += emb1[i] * emb1b[i];
            printf("    Self-similarity: %.4f (expect ~1.0)\n", cosine);
            CHECK(cosine > 0.99f, "same file gives identical embedding");
        }

        /* Cross-speaker similarity: different files may differ */
        if (file_exists(wav2)) {
            float emb2[512] = {0};
            int ret3 = speaker_encoder_extract_from_wav(enc, wav2, emb2);
            if (ret > 0 && ret3 > 0) {
                float cosine = 0.0f;
                for (int i = 0; i < dim; i++) cosine += emb1[i] * emb2[i];
                printf("    Cross-file similarity: %.4f\n", cosine);
                CHECK(cosine > -1.0f && cosine <= 1.0f, "cross-file cosine in valid range");
            }
        }
    } else {
        SKIP("hello_piper.wav not found for speaker encoder test");
    }

    speaker_encoder_destroy(enc);
}

static void test_phonemizer_real(void) {
    printf("\n═══ Phonemizer (Real espeak-ng) ═══\n");

    Phonemizer *ph = phonemizer_create("en-us");
    CHECK(ph != NULL, "phonemizer creates with en-us");
    if (!ph) return;

    /* Test various text inputs */
    char ipa[512];

    int len = phonemizer_text_to_ipa(ph, "Hello world", ipa, sizeof(ipa));
    printf("    'Hello world' → IPA: %s\n", ipa);
    CHECK(len > 0, "Hello world → IPA succeeds");

    len = phonemizer_text_to_ipa(ph, "The quick brown fox jumps over the lazy dog", ipa, sizeof(ipa));
    printf("    Pangram → IPA: %s\n", ipa);
    CHECK(len > 20, "pangram produces substantial IPA");

    /* Heteronym test: "read" should produce IPA */
    len = phonemizer_text_to_ipa(ph, "I read a book yesterday", ipa, sizeof(ipa));
    printf("    Heteronym → IPA: %s\n", ipa);
    CHECK(len > 0, "heteronym sentence produces IPA");

    /* Numbers and special text */
    len = phonemizer_text_to_ipa(ph, "The year 2025 was exciting", ipa, sizeof(ipa));
    printf("    Numbers → IPA: %s\n", ipa);
    CHECK(len > 0, "numbers in text produce IPA");

    phonemizer_destroy(ph);
}

int main(void) {
    printf("╔══════════════════════════════════════════════════════════╗\n");
    printf("║  Real Model Integration Tests                           ║\n");
    printf("║  Speaker Encoder + Phonemizer                           ║\n");
    printf("╚══════════════════════════════════════════════════════════╝\n");

    test_speaker_encoder();
    test_phonemizer_real();

    printf("\n══════════════════════════════════════════════════════════\n");
    printf("Results: %d passed, %d failed, %d skipped\n", passed, failed, skipped);
    if (failed > 0) { printf("SOME TESTS FAILED\n"); return 1; }
    printf("ALL PASSED\n");
    return 0;
}
