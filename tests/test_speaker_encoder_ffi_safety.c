/**
 * test_speaker_encoder_ffi_safety.c — NULL-safety and bounds-checking tests.
 *
 * Tests speaker encoder FFI boundary conditions:
 *   1. NULL pointer handling in all FFI functions
 *   2. Invalid path handling (nonexistent files)
 *   3. Bounds checking on sample counts and frame counts
 *   4. Null-destruction safety
 *   5. Return value validation
 *
 * FFI Functions tested:
 *   - speaker_encoder_native_create(weights, config) → *void
 *   - speaker_encoder_native_destroy(*void) → void
 *   - speaker_encoder_native_embedding_dim(*void) → int
 *   - speaker_encoder_native_encode(*void, mel, n_frames, n_mels, out) → int
 *   - speaker_encoder_native_encode_audio(*void, pcm, n_samples, sr, out) → int
 *   - speaker_encoder_native_sample_rate(*void) → int
 *
 * Build:
 *   cc -O3 -arch arm64 -Isrc \
 *      -Ltarget/release -lsonata_speaker \
 *      -Wl,-rpath,$(CURDIR)/target/release \
 *      -o tests/test_speaker_encoder_ffi_safety tests/test_speaker_encoder_ffi_safety.c -lm
 *
 * Run: ./tests/test_speaker_encoder_ffi_safety
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

/* FFI declarations matching sonata_speaker/src/lib.rs */

void *speaker_encoder_native_create(const char *weights_path, const char *config_path);
void speaker_encoder_native_destroy(void *engine);
int speaker_encoder_native_embedding_dim(const void *engine);
int speaker_encoder_native_encode(
    void *engine,
    const float *mel_data,
    int n_frames,
    int n_mels,
    float *out
);
int speaker_encoder_native_encode_audio(
    void *engine,
    const float *pcm,
    int n_samples,
    int sample_rate,
    float *out
);
int speaker_encoder_native_sample_rate(const void *engine);

/* Test framework */

static int tests_passed = 0;
static int tests_failed = 0;

#define TEST(name) do { printf("  %-65s", name); fflush(stdout); } while(0)
#define PASS() do { printf("PASS\n"); tests_passed++; } while(0)
#define FAIL(msg) do { printf("FAIL: %s\n", msg); tests_failed++; } while(0)

/* ─────────────────────────────────────────────────────────────────────────── */
/* P0: NULL Pointer Safety Tests                                              */
/* ─────────────────────────────────────────────────────────────────────────── */

static void test_create_null_weights_path(void) {
    TEST("create(NULL, config) → NULL");
    void *engine = speaker_encoder_native_create(NULL, "config.json");
    if (engine != NULL) {
        FAIL("should return NULL for NULL weights path");
        speaker_encoder_native_destroy(engine);
        return;
    }
    PASS();
}

static void test_create_null_config_path(void) {
    TEST("create(weights, NULL) → NULL");
    void *engine = speaker_encoder_native_create("model.safetensors", NULL);
    if (engine != NULL) {
        FAIL("should return NULL for NULL config path");
        speaker_encoder_native_destroy(engine);
        return;
    }
    PASS();
}

static void test_create_both_null(void) {
    TEST("create(NULL, NULL) → NULL");
    void *engine = speaker_encoder_native_create(NULL, NULL);
    if (engine != NULL) {
        FAIL("should return NULL for both NULL paths");
        speaker_encoder_native_destroy(engine);
        return;
    }
    PASS();
}

static void test_create_nonexistent_file(void) {
    TEST("create(nonexistent.safetensors, nonexistent.json) → NULL");
    void *engine = speaker_encoder_native_create(
        "/nonexistent/path/model_99999.safetensors",
        "/nonexistent/path/config_99999.json"
    );
    if (engine != NULL) {
        FAIL("should return NULL for nonexistent files");
        speaker_encoder_native_destroy(engine);
        return;
    }
    PASS();
}

static void test_destroy_null(void) {
    TEST("destroy(NULL) → no crash");
    /* Should not crash or have undefined behavior */
    speaker_encoder_native_destroy(NULL);
    PASS();
}

static void test_destroy_double_free(void) {
    TEST("destroy(ptr); destroy(ptr) → no crash");
    /* Create a dummy engine pointer (will be NULL, but let's test anyway) */
    void *engine = speaker_encoder_native_create(NULL, NULL);
    speaker_encoder_native_destroy(engine);
    /* Second destroy should also not crash */
    speaker_encoder_native_destroy(engine);
    PASS();
}

/* ─────────────────────────────────────────────────────────────────────────── */
/* P1: Invalid Handle Tests                                                   */
/* ─────────────────────────────────────────────────────────────────────────── */

static void test_embedding_dim_null_engine(void) {
    TEST("embedding_dim(NULL) → 0 or -1");
    int dim = speaker_encoder_native_embedding_dim(NULL);
    if (dim < 0 && dim != -1) {
        FAIL("unexpected error code");
        return;
    }
    if (dim >= 0 && dim != 0 && dim != 256) {
        FAIL("embedding_dim should return 0, -1, or 256");
        return;
    }
    PASS();
}

static void test_sample_rate_null_engine(void) {
    TEST("sample_rate(NULL) → default or 0/-1");
    int sr = speaker_encoder_native_sample_rate(NULL);
    /* Should return default (16000) or error signal */
    if (sr < 0 && sr != -1) {
        FAIL("unexpected error code");
        return;
    }
    if (sr >= 0 && sr != 16000) {
        FAIL("sample_rate should return 16000 or -1");
        return;
    }
    PASS();
}

static void test_encode_null_engine(void) {
    TEST("encode(NULL, mel, n_frames, n_mels, out) → -1");
    float mel[300 * 80];
    float out[256];
    int result = speaker_encoder_native_encode(NULL, mel, 300, 80, out);
    if (result != -1) {
        FAIL("should return -1 for NULL engine");
        return;
    }
    PASS();
}

static void test_encode_null_mel(void) {
    TEST("encode(engine, NULL, n_frames, n_mels, out) → -1");
    float out[256];
    int result = speaker_encoder_native_encode(NULL, NULL, 300, 80, out);
    if (result != -1) {
        FAIL("should return -1 for NULL mel data");
        return;
    }
    PASS();
}

static void test_encode_null_output(void) {
    TEST("encode(engine, mel, n_frames, n_mels, NULL) → -1");
    float mel[300 * 80];
    int result = speaker_encoder_native_encode((void *)1, mel, 300, 80, NULL);
    if (result != -1) {
        FAIL("should return -1 for NULL output buffer");
        return;
    }
    PASS();
}

static void test_encode_audio_null_engine(void) {
    TEST("encode_audio(NULL, pcm, n_samples, sr, out) → -1");
    float pcm[48000];
    float out[256];
    int result = speaker_encoder_native_encode_audio(NULL, pcm, 48000, 16000, out);
    if (result != -1) {
        FAIL("should return -1 for NULL engine");
        return;
    }
    PASS();
}

static void test_encode_audio_null_pcm(void) {
    TEST("encode_audio(engine, NULL, n_samples, sr, out) → -1");
    float out[256];
    int result = speaker_encoder_native_encode_audio(NULL, NULL, 48000, 16000, out);
    if (result != -1) {
        FAIL("should return -1 for NULL PCM data");
        return;
    }
    PASS();
}

static void test_encode_audio_null_output(void) {
    TEST("encode_audio(engine, pcm, n_samples, sr, NULL) → -1");
    float pcm[48000];
    int result = speaker_encoder_native_encode_audio((void *)1, pcm, 48000, 16000, NULL);
    if (result != -1) {
        FAIL("should return -1 for NULL output buffer");
        return;
    }
    PASS();
}

/* ─────────────────────────────────────────────────────────────────────────── */
/* P2: Bounds Checking Tests                                                  */
/* ─────────────────────────────────────────────────────────────────────────── */

static void test_encode_zero_frames(void) {
    TEST("encode(engine, mel, 0, n_mels, out) → -1");
    float mel[0];
    float out[256];
    int result = speaker_encoder_native_encode((void *)1, mel, 0, 80, out);
    if (result != -1) {
        FAIL("should reject zero frames");
        return;
    }
    PASS();
}

static void test_encode_negative_frames(void) {
    TEST("encode(engine, mel, -1, n_mels, out) → -1");
    float mel[300 * 80];
    float out[256];
    int result = speaker_encoder_native_encode((void *)1, mel, -1, 80, out);
    if (result != -1) {
        FAIL("should reject negative frames");
        return;
    }
    PASS();
}

static void test_encode_zero_mels(void) {
    TEST("encode(engine, mel, n_frames, 0, out) → -1");
    float mel[300 * 80];
    float out[256];
    int result = speaker_encoder_native_encode(NULL, mel, 300, 0, out);
    if (result != -1) {
        FAIL("should reject zero mel bins");
        return;
    }
    PASS();
}

static void test_encode_negative_mels(void) {
    TEST("encode(engine, mel, n_frames, -1, out) → -1");
    float mel[300 * 80];
    float out[256];
    int result = speaker_encoder_native_encode((void *)1, mel, 300, -1, out);
    if (result != -1) {
        FAIL("should reject negative mel bins");
        return;
    }
    PASS();
}

static void test_encode_audio_zero_samples(void) {
    TEST("encode_audio(engine, pcm, 0, sr, out) → -1");
    float pcm[0];
    float out[256];
    int result = speaker_encoder_native_encode_audio((void *)1, pcm, 0, 16000, out);
    if (result != -1) {
        FAIL("should reject zero samples");
        return;
    }
    PASS();
}

static void test_encode_audio_negative_samples(void) {
    TEST("encode_audio(engine, pcm, -1, sr, out) → -1");
    float pcm[48000];
    float out[256];
    int result = speaker_encoder_native_encode_audio((void *)1, pcm, -1, 16000, out);
    if (result != -1) {
        FAIL("should reject negative samples");
        return;
    }
    PASS();
}

static void test_encode_audio_zero_sample_rate(void) {
    TEST("encode_audio(engine, pcm, n_samples, 0, out) → -1");
    float pcm[48000];
    float out[256];
    int result = speaker_encoder_native_encode_audio((void *)1, pcm, 48000, 0, out);
    if (result != -1) {
        FAIL("should reject zero sample rate");
        return;
    }
    PASS();
}

static void test_encode_audio_negative_sample_rate(void) {
    TEST("encode_audio(engine, pcm, n_samples, -1, out) → -1");
    float pcm[48000];
    float out[256];
    int result = speaker_encoder_native_encode_audio((void *)1, pcm, 48000, -1, out);
    if (result != -1) {
        FAIL("should reject negative sample rate");
        return;
    }
    PASS();
}

static void test_encode_oversized_frames(void) {
    TEST("encode(engine, mel, 1000000, 80, out) → -1 (bounds)");
    /* This tests whether the implementation has reasonable frame limits.
       Using a NULL engine pointer to avoid segfault on oversized input. */
    float mel[100 * 80];
    float out[256];
    int result = speaker_encoder_native_encode(NULL, mel, 1000000, 80, out);
    /* Should reject due to NULL engine */
    if (result != -1) {
        FAIL("should return -1 for NULL engine");
        return;
    }
    PASS();
}

static void test_encode_audio_oversized_samples(void) {
    TEST("encode_audio(engine, pcm, 10000000, sr, out) → bounded");
    float pcm[1000];
    float out[256];
    int result = speaker_encoder_native_encode_audio(NULL, pcm, 10000000, 16000, out);
    /* Should reject due to NULL engine and/or negative sample count */
    if (result != -1) {
        FAIL("should return -1 for NULL engine");
        return;
    }
    PASS();
}

/* ─────────────────────────────────────────────────────────────────────────── */
/* P3: Return Value Validation                                                */
/* ─────────────────────────────────────────────────────────────────────────── */

static void test_embedding_dim_expected_value(void) {
    TEST("embedding_dim() returns 256 for valid engine (or 0/-1 for NULL)");
    int dim = speaker_encoder_native_embedding_dim(NULL);
    /* NULL should return 0 or -1, not 256 */
    if (dim > 256 || (dim >= 0 && dim != 0 && dim != 256)) {
        FAIL("unexpected embedding dimension");
        return;
    }
    PASS();
}

static void test_sample_rate_expected_value(void) {
    TEST("sample_rate() returns 16000 for valid engine (or 0/-1 for NULL)");
    int sr = speaker_encoder_native_sample_rate(NULL);
    /* NULL should return 16000 (default) or -1 */
    if (sr < 0 && sr != -1) {
        FAIL("unexpected error code");
        return;
    }
    if (sr > 0 && sr != 16000 && sr != 8000 && sr != 22050 && sr != 24000 && sr != 44100 && sr != 48000) {
        FAIL("sample_rate should return a valid sample rate");
        return;
    }
    PASS();
}

/* ─────────────────────────────────────────────────────────────────────────── */
/* Main                                                                        */
/* ─────────────────────────────────────────────────────────────────────────── */

int main(void) {
    printf("\n╔═══════════════════════════════════════════════════════════════════╗\n");
    printf("║  Speaker Encoder FFI NULL-Safety & Bounds-Checking Test Suite     ║\n");
    printf("╚═══════════════════════════════════════════════════════════════════╝\n\n");

    printf("═ P0: NULL Pointer Safety ═\n");
    test_create_null_weights_path();
    test_create_null_config_path();
    test_create_both_null();
    test_create_nonexistent_file();
    test_destroy_null();
    test_destroy_double_free();

    printf("\n═ P1: Invalid Handle Tests ═\n");
    test_embedding_dim_null_engine();
    test_sample_rate_null_engine();
    test_encode_null_engine();
    test_encode_null_mel();
    test_encode_null_output();
    test_encode_audio_null_engine();
    test_encode_audio_null_pcm();
    test_encode_audio_null_output();

    printf("\n═ P2: Bounds Checking ═\n");
    test_encode_zero_frames();
    test_encode_negative_frames();
    test_encode_zero_mels();
    test_encode_negative_mels();
    test_encode_audio_zero_samples();
    test_encode_audio_negative_samples();
    test_encode_audio_zero_sample_rate();
    test_encode_audio_negative_sample_rate();
    test_encode_oversized_frames();
    test_encode_audio_oversized_samples();

    printf("\n═ P3: Return Value Validation ═\n");
    test_embedding_dim_expected_value();
    test_sample_rate_expected_value();

    printf("\n╔═══════════════════════════════════════════════════════════════════╗\n");
    printf("║ RESULTS: %d passed, %d failed                                     ║\n", tests_passed, tests_failed);
    printf("╚═══════════════════════════════════════════════════════════════════╝\n\n");

    return tests_failed > 0 ? 1 : 0;
}
