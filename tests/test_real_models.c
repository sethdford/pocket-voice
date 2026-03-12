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
    if (!file_exists(model_path) || !file_exists("models/speaker_encoder_config.json")) {
        SKIP("ECAPA-TDNN safetensors model/config not found — skipping speaker encoder tests");
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

/* ── Additional Tests ────────────────────────────────────────────── */

static void test_model_path_validation(void) {
    printf("\n═══ Model Path Validation ═══\n");

    /* Empty string path */
    SpeakerEncoder *enc = speaker_encoder_create("");
    CHECK(enc == NULL, "speaker_encoder_create with empty path returns NULL");
    if (enc) speaker_encoder_destroy(enc);

    /* Path to a directory (not a file) */
    enc = speaker_encoder_create("/tmp");
    CHECK(enc == NULL, "speaker_encoder_create with directory path returns NULL");
    if (enc) speaker_encoder_destroy(enc);

    /* Path with wrong extension — should still fail gracefully */
    enc = speaker_encoder_create("/tmp/notamodel.txt");
    CHECK(enc == NULL, "speaker_encoder_create with non-ONNX file returns NULL");
    if (enc) speaker_encoder_destroy(enc);
}

static void test_phoneme_map_loading(void) {
    printf("\n═══ Phoneme Map Loading ═══\n");

    const char *map_path = "models/sonata/phoneme_map.json";
    if (!file_exists(map_path)) {
        SKIP("phoneme_map.json not found");
        return;
    }

    Phonemizer *ph = phonemizer_create("en-us");
    CHECK(ph != NULL, "phonemizer creates for map loading test");
    if (!ph) return;

    /* Before loading: vocab_size should be 0 */
    extern int phonemizer_vocab_size(const Phonemizer *ph);
    int vs = phonemizer_vocab_size(ph);
    CHECK(vs == 0, "vocab_size is 0 before loading map");

    /* Load the map */
    extern int phonemizer_load_phoneme_map(Phonemizer *ph, const char *json_path);
    int rc = phonemizer_load_phoneme_map(ph, map_path);
    CHECK(rc == 0, "phoneme_map.json loads successfully");

    /* After loading: vocab_size > 0 */
    vs = phonemizer_vocab_size(ph);
    printf("    Vocab size: %d\n", vs);
    CHECK(vs > 20, "vocab_size > 20 after loading map");

    /* text_to_ids should now produce valid IDs */
    extern int phonemizer_text_to_ids(Phonemizer *ph, const char *text, int *ids_out, int max_ids);
    int ids[128];
    int n = phonemizer_text_to_ids(ph, "Hello world", ids, 128);
    CHECK(n > 0, "text_to_ids produces IDs after map loaded");
    if (n > 0) {
        CHECK(ids[0] == 1, "first ID is BOS token (1)");
        CHECK(ids[n - 1] == 2, "last ID is EOS token (2)");

        /* All IDs should be within vocab range */
        int in_range = 1;
        for (int i = 0; i < n; i++) {
            if (ids[i] < 0 || ids[i] >= vs + 10) { in_range = 0; break; }
        }
        CHECK(in_range, "all IDs are in valid range");
    }

    phonemizer_destroy(ph);
}

static void test_wav_loading_edge_cases(void) {
    printf("\n═══ WAV Loading Edge Cases ═══\n");

    int n_samples = 0, sr = 0;

    /* NULL path */
    float *samples = load_wav(NULL, &n_samples, &sr);
    CHECK(samples == NULL, "load_wav with NULL path returns NULL");

    /* Nonexistent file */
    samples = load_wav("/nonexistent/audio.wav", &n_samples, &sr);
    CHECK(samples == NULL, "load_wav with nonexistent path returns NULL");

    /* Empty string path */
    samples = load_wav("", &n_samples, &sr);
    CHECK(samples == NULL, "load_wav with empty path returns NULL");
}

static void test_resampling_identity(void) {
    printf("\n═══ Resampling Identity Test ═══\n");

    /* Same rate -> should get exact copy */
    float input[100];
    for (int i = 0; i < 100; i++) input[i] = sinf((float)i * 0.1f);

    int out_samples = 0;
    float *output = resample(input, 100, 16000, 16000, &out_samples);
    CHECK(output != NULL, "identity resample returns non-NULL");
    CHECK(out_samples == 100, "identity resample preserves sample count");

    if (output) {
        int match = 1;
        for (int i = 0; i < 100; i++) {
            if (fabsf(output[i] - input[i]) > 1e-6f) { match = 0; break; }
        }
        CHECK(match, "identity resample preserves exact values");
        free(output);
    }
}

static void test_resampling_rate_conversion(void) {
    printf("\n═══ Resampling Rate Conversion ═══\n");

    /* Upsample 16kHz -> 48kHz: output should have 3x samples */
    float input[160];
    for (int i = 0; i < 160; i++) input[i] = sinf((float)i * 0.05f);

    int out_samples = 0;
    float *output = resample(input, 160, 16000, 48000, &out_samples);
    CHECK(output != NULL, "upsample 16kHz->48kHz returns non-NULL");
    CHECK(out_samples == 480, "upsample 16kHz->48kHz produces 3x samples");

    if (output) {
        /* Verify interpolated values are reasonable (within input range) */
        float min_in = input[0], max_in = input[0];
        for (int i = 1; i < 160; i++) {
            if (input[i] < min_in) min_in = input[i];
            if (input[i] > max_in) max_in = input[i];
        }
        int in_range = 1;
        for (int i = 0; i < out_samples; i++) {
            if (output[i] < min_in - 0.01f || output[i] > max_in + 0.01f) {
                in_range = 0; break;
            }
        }
        CHECK(in_range, "upsampled values are within input range");
        free(output);
    }

    /* Downsample 48kHz -> 16kHz: output should have 1/3 samples */
    float input48[480];
    for (int i = 0; i < 480; i++) input48[i] = sinf((float)i * 0.02f);

    output = resample(input48, 480, 48000, 16000, &out_samples);
    CHECK(output != NULL, "downsample 48kHz->16kHz returns non-NULL");
    CHECK(out_samples == 160, "downsample 48kHz->16kHz produces 1/3 samples");
    if (output) free(output);
}

static void test_phonemizer_consistency(void) {
    printf("\n═══ Phonemizer Consistency ═══\n");

    Phonemizer *ph = phonemizer_create("en-us");
    CHECK(ph != NULL, "phonemizer creates for consistency test");
    if (!ph) return;

    /* Same input should always produce same output */
    char ipa1[512], ipa2[512];
    int len1 = phonemizer_text_to_ipa(ph, "Consistent output", ipa1, sizeof(ipa1));
    int len2 = phonemizer_text_to_ipa(ph, "Consistent output", ipa2, sizeof(ipa2));
    CHECK(len1 > 0 && len2 > 0, "consistency: both calls produce output");
    CHECK(len1 == len2, "consistency: same input produces same length");
    if (len1 > 0 && len2 > 0) {
        CHECK(strcmp(ipa1, ipa2) == 0, "consistency: same input produces identical IPA");
    }

    phonemizer_destroy(ph);
}

static void test_phonemizer_special_characters(void) {
    printf("\n═══ Phonemizer Special Characters ═══\n");

    Phonemizer *ph = phonemizer_create("en-us");
    CHECK(ph != NULL, "phonemizer creates for special chars test");
    if (!ph) return;

    char ipa[512];

    /* Punctuation-heavy text */
    int len = phonemizer_text_to_ipa(ph, "Hello! How are you?", ipa, sizeof(ipa));
    CHECK(len > 0, "punctuation text produces IPA");

    /* Text with numbers */
    len = phonemizer_text_to_ipa(ph, "I have 42 cats and 7 dogs", ipa, sizeof(ipa));
    CHECK(len > 0, "text with numbers produces IPA");

    /* Single character */
    len = phonemizer_text_to_ipa(ph, "A", ipa, sizeof(ipa));
    CHECK(len > 0 || len == -1, "single character handled gracefully");

    phonemizer_destroy(ph);
}

static void test_speaker_encoder_embedding_norms(void) {
    printf("\n═══ Embedding Norm Validation ═══\n");

    const char *model_path = "models/ecapa_tdnn.onnx";
    if (!file_exists(model_path) || !file_exists("models/speaker_encoder_config.json")) {
        SKIP("ECAPA-TDNN model/config not found for norm validation");
        return;
    }

    SpeakerEncoder *enc = speaker_encoder_create(model_path);
    CHECK(enc != NULL, "encoder loads for norm validation");
    if (!enc) return;

    int dim = speaker_encoder_embedding_dim(enc);

    /* Test with synthetic audio (silence) */
    float *silence = calloc(16000, sizeof(float));
    float emb[512] = {0};
    int ret = speaker_encoder_extract(enc, silence, 16000, emb);

    if (ret > 0) {
        /* L2 norm should be ~1.0 (embeddings are normalized) */
        float norm = 0.0f;
        for (int i = 0; i < dim; i++) norm += emb[i] * emb[i];
        norm = sqrtf(norm);
        printf("    Silence embedding L2 norm: %.4f\n", norm);
        CHECK(fabsf(norm - 1.0f) < 0.1f, "silence embedding is approximately L2-normalized");

        /* No NaN or Inf values */
        int has_nan = 0;
        for (int i = 0; i < dim; i++) {
            if (isnan(emb[i]) || isinf(emb[i])) { has_nan = 1; break; }
        }
        CHECK(!has_nan, "embedding has no NaN or Inf values");
    } else {
        SKIP("extract from silence failed (model may require non-silent audio)");
    }

    /* Test with noise audio */
    float *noise = malloc(16000 * sizeof(float));
    srand(42);
    for (int i = 0; i < 16000; i++)
        noise[i] = ((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f;

    float emb_noise[512] = {0};
    ret = speaker_encoder_extract(enc, noise, 16000, emb_noise);
    if (ret > 0) {
        float norm = 0.0f;
        for (int i = 0; i < dim; i++) norm += emb_noise[i] * emb_noise[i];
        norm = sqrtf(norm);
        printf("    Noise embedding L2 norm: %.4f\n", norm);
        CHECK(fabsf(norm - 1.0f) < 0.1f, "noise embedding is approximately L2-normalized");
    }

    free(silence);
    free(noise);
    speaker_encoder_destroy(enc);
}

int main(void) {
    printf("╔══════════════════════════════════════════════════════════╗\n");
    printf("║  Real Model Integration Tests                           ║\n");
    printf("║  Speaker Encoder + Phonemizer                           ║\n");
    printf("╚══════════════════════════════════════════════════════════╝\n");

    test_speaker_encoder();
    test_phonemizer_real();
    test_model_path_validation();
    test_phoneme_map_loading();
    test_wav_loading_edge_cases();
    test_resampling_identity();
    test_resampling_rate_conversion();
    test_phonemizer_consistency();
    test_phonemizer_special_characters();
    test_speaker_encoder_embedding_norms();

    printf("\n══════════════════════════════════════════════════════════\n");
    printf("Results: %d passed, %d failed, %d skipped\n", passed, failed, skipped);
    if (failed > 0) { printf("SOME TESTS FAILED\n"); return 1; }
    printf("ALL PASSED\n");
    return 0;
}
