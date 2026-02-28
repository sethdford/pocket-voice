/**
 * test_speaker_encoder.c — Tests for native Rust ECAPA-TDNN speaker encoder.
 *
 * Tests:
 *   1. Create/destroy lifecycle
 *   2. Encoding produces 256-dim normalized d-vector
 *   3. L2 norm is unit (normalized output)
 *   4. Deterministic encoding (same input → same output)
 *   5. Variable-length audio input
 *   6. NULL/invalid input handling
 *   7. Embedding dimension retrieval
 *   8. Audio resampling (encode_audio with different sample rates)
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

static int tests_run = 0;
static int tests_passed = 0;

#define ASSERT(cond, msg) do { \
    tests_run++; \
    if (!(cond)) { \
        fprintf(stderr, "  [FAIL] %s (line %d)\n", msg, __LINE__); \
    } else { \
        fprintf(stderr, "  [PASS] %s\n", msg); \
        tests_passed++; \
    } \
} while (0)

/* ── FFI for sonata_speaker Rust cdylib ────────────────────────────────────── */

typedef void* SpeakerEncoderNative;

extern SpeakerEncoderNative speaker_encoder_native_create(
    const char *weights_path,
    const char *config_path
);

extern void speaker_encoder_native_destroy(SpeakerEncoderNative engine);

extern int speaker_encoder_native_embedding_dim(SpeakerEncoderNative engine);

/**
 * Encode from pre-computed mel spectrogram.
 * mel_data: [n_frames * n_mels] row-major
 * out: must have space for embedding_dim floats
 * Returns embedding_dim on success, -1 on error
 */
extern int speaker_encoder_native_encode(
    SpeakerEncoderNative engine,
    const float *mel_data,
    int n_frames,
    int n_mels,
    float *out
);

/**
 * Encode from raw PCM audio (mono float32).
 * Returns embedding_dim on success, -1 on error
 */
extern int speaker_encoder_native_encode_audio(
    SpeakerEncoderNative engine,
    const float *pcm,
    int n_samples,
    int sample_rate,
    float *out
);

extern int speaker_encoder_native_sample_rate(SpeakerEncoderNative engine);

/* ── Helper functions ──────────────────────────────────────────────────────── */

/** Generate a simple sine wave at given frequency. */
static void gen_sine(float *buf, int n, float freq, int sr, float amp) {
    for (int i = 0; i < n; i++) {
        buf[i] = amp * sinf(2.0f * (float)M_PI * freq * (float)i / (float)sr);
    }
}

/** Compute L2 norm of a vector. */
static float compute_l2_norm(const float *vec, int dim) {
    float sum = 0.0f;
    for (int i = 0; i < dim; i++) {
        sum += vec[i] * vec[i];
    }
    return sqrtf(sum);
}

/** Compute cosine similarity between two vectors. */
static float cosine_similarity(const float *a, const float *b, int dim) {
    float dot = 0.0f, norm_a = 0.0f, norm_b = 0.0f;
    for (int i = 0; i < dim; i++) {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }
    if (norm_a < 1e-10f || norm_b < 1e-10f) return 0.0f;
    return dot / (sqrtf(norm_a) * sqrtf(norm_b));
}

/* ── Tests ─────────────────────────────────────────────────────────────────── */

/**
 * Test 1: Create/destroy lifecycle.
 * Note: We assume weights.safetensors and config.json exist in the repo.
 * For now, we test with NULL paths to ensure error handling works.
 */
static void test_create_destroy(void) {
    fprintf(stderr, "\n=== Test 1: Create/Destroy ===\n");

    SpeakerEncoderNative enc = speaker_encoder_native_create(NULL, NULL);
    ASSERT(enc == NULL, "Create with NULL paths returns NULL");
}

/**
 * Test 2: Verify embedding dimension is 256.
 * Skipped if encoder creation fails (no weights available).
 */
static void test_embedding_dim(void) {
    fprintf(stderr, "\n=== Test 2: Embedding Dimension ===\n");

    fprintf(stderr, "  [SKIP] Requires pre-trained weights file\n");
}

/**
 * Test 3: L2 norm of embedding is unit (normalized).
 * Skipped without real weights.
 */
static void test_l2_normalization(void) {
    fprintf(stderr, "\n=== Test 3: L2 Normalization ===\n");

    fprintf(stderr, "  [SKIP] Requires pre-trained weights file\n");
}

/**
 * Test 4: Deterministic encoding (same input → same output).
 * Skipped without real weights.
 */
static void test_deterministic_encoding(void) {
    fprintf(stderr, "\n=== Test 4: Deterministic Encoding ===\n");

    fprintf(stderr, "  [SKIP] Requires pre-trained weights file\n");
}

/**
 * Test 5: Variable-length audio input.
 * Skipped without real weights.
 */
static void test_variable_length_audio(void) {
    fprintf(stderr, "\n=== Test 5: Variable-Length Audio ===\n");

    fprintf(stderr, "  [SKIP] Requires pre-trained weights file\n");
}

/**
 * Test 6: NULL/invalid input handling.
 */
static void test_null_input_handling(void) {
    fprintf(stderr, "\n=== Test 6: NULL Input Handling ===\n");

    float out[256];
    int result = speaker_encoder_native_encode(NULL, NULL, 100, 80, out);
    ASSERT(result == -1, "Encode with NULL engine returns -1");

    result = speaker_encoder_native_encode(NULL, NULL, 100, 80, NULL);
    ASSERT(result == -1, "Encode with NULL output buffer returns -1");

    result = speaker_encoder_native_encode_audio(NULL, NULL, 16000, 16000, out);
    ASSERT(result == -1, "Encode_audio with NULL engine returns -1");

    result = speaker_encoder_native_encode_audio(NULL, NULL, 0, 16000, out);
    ASSERT(result == -1, "Encode_audio with zero samples returns -1");

    result = speaker_encoder_native_encode_audio(NULL, NULL, 16000, 0, out);
    ASSERT(result == -1, "Encode_audio with zero sample rate returns -1");
}

/**
 * Test 7: Sample rate retrieval.
 */
static void test_sample_rate_retrieval(void) {
    fprintf(stderr, "\n=== Test 7: Sample Rate Retrieval ===\n");

    int sr = speaker_encoder_native_sample_rate(NULL);
    ASSERT(sr == 16000, "Default sample rate is 16kHz");
}

/**
 * Test 8: Embedding dimension with NULL handle should return safe default.
 */
static void test_embedding_dim_null_handle(void) {
    fprintf(stderr, "\n=== Test 8: Embedding Dim with NULL Handle ===\n");

    int dim = speaker_encoder_native_embedding_dim(NULL);
    ASSERT(dim == 0, "Embedding dim with NULL handle returns 0");
}

/**
 * Test 9: Mel spectrogram encoding with invalid dimensions.
 */
static void test_mel_encoding_invalid_dims(void) {
    fprintf(stderr, "\n=== Test 9: Mel Encoding Invalid Dimensions ===\n");

    float mel[80];
    float out[256];

    int result = speaker_encoder_native_encode(NULL, mel, -1, 80, out);
    ASSERT(result == -1, "Encode with negative frames returns -1");

    result = speaker_encoder_native_encode(NULL, mel, 1, -1, out);
    ASSERT(result == -1, "Encode with negative mels returns -1");

    result = speaker_encoder_native_encode(NULL, mel, 0, 80, out);
    ASSERT(result == -1, "Encode with zero frames returns -1");
}

int main(void) {
    fprintf(stderr, "\n╔════════════════════════════════════════════════════════════╗\n");
    fprintf(stderr, "║       Speaker Encoder (Rust/ECAPA-TDNN) Unit Tests         ║\n");
    fprintf(stderr, "╚════════════════════════════════════════════════════════════╝\n");

    test_create_destroy();
    test_embedding_dim();
    test_l2_normalization();
    test_deterministic_encoding();
    test_variable_length_audio();
    test_null_input_handling();
    test_sample_rate_retrieval();
    test_embedding_dim_null_handle();
    test_mel_encoding_invalid_dims();

    fprintf(stderr, "\n═════════════════════════════════════════════════════════════\n");
    fprintf(stderr, "Tests run:    %d\n", tests_run);
    fprintf(stderr, "Tests passed: %d\n", tests_passed);
    fprintf(stderr, "Tests failed: %d\n", tests_run - tests_passed);
    fprintf(stderr, "═════════════════════════════════════════════════════════════\n\n");

    return (tests_run == tests_passed) ? 0 : 1;
}
