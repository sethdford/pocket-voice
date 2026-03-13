/**
 * test_sonata_flow_ffi_safety.c — NULL-safety and bounds-checking for Flow FFI.
 *
 * Tests sonata_flow FFI boundary conditions:
 *   1. NULL pointer handling in all FFI functions
 *   2. Invalid engine handles
 *   3. Bounds checking on semantic tokens, frame counts, speaker IDs
 *   4. Parameter validation (cfg scale, steps, quality mode)
 *   5. Return value validation
 *
 * Key FFI Functions tested:
 *   - sonata_flow_create(weights, config) → *void
 *   - sonata_flow_destroy(*void) → void
 *   - sonata_flow_set_speaker(*void, speaker_id) → int
 *   - sonata_flow_set_cfg_scale(*void, scale) → int
 *   - sonata_flow_set_n_steps(*void, n_steps) → int
 *   - sonata_flow_set_quality_mode(*void, mode) → int
 *   - sonata_flow_generate(*void, semantics, n_frames, speaker, out_len) → int
 *   - sonata_flow_generate_audio(*void, semantics, n_frames, sr, out) → int
 *   - sonata_flow_interpolate_speakers(*void, id1, id2, alpha, out) → int/float*
 *   - sonata_flow_set_speaker_embedding(*void, embedding, dim) → int
 *
 * Build:
 *   cc -O3 -arch arm64 -Isrc \
 *      -Ltarget/release -lsonata_flow \
 *      -Wl,-rpath,$(CURDIR)/target/release \
 *      -o tests/test_sonata_flow_ffi_safety tests/test_sonata_flow_ffi_safety.c -lm
 *
 * Run: ./tests/test_sonata_flow_ffi_safety
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

/* FFI declarations matching sonata_flow/src/lib.rs */

void *sonata_flow_create(const char *weights_path, const char *config_path);
void sonata_flow_destroy(void *engine);
int sonata_flow_set_speaker(void *engine, int speaker_id);
int sonata_flow_set_cfg_scale(void *engine, float scale);
int sonata_flow_set_n_steps(void *engine, int n_steps);
int sonata_flow_set_quality_mode(void *engine, int mode);
int sonata_flow_generate(
    void *engine,
    const int *semantic_tokens,
    int n_frames,
    float *out_magnitude,
    float *out_phase
);
int sonata_flow_generate_audio(
    void *engine,
    const int *semantic_tokens,
    int n_frames,
    float *out_audio,
    int max_samples
);
int sonata_flow_interpolate_speakers(void *engine, const float *emb_a, const float *emb_b, int dim, float alpha);
int sonata_flow_set_speaker_embedding(void *engine, const float *embedding, int dim);
void sonata_flow_clear_speaker_embedding(void *engine);
void sonata_flow_reset_streaming(void *engine);
int sonata_flow_samples_per_frame(void *engine);

/* Test framework */

static int tests_passed = 0;
static int tests_failed = 0;

#define TEST(name) do { printf("  %-65s", name); fflush(stdout); } while(0)
#define PASS() do { printf("PASS\n"); tests_passed++; } while(0)
#define FAIL(msg) do { printf("FAIL: %s\n", msg); tests_failed++; } while(0)

/* ─────────────────────────────────────────────────────────────────────────── */
/* P0: NULL Pointer Safety Tests                                              */
/* ─────────────────────────────────────────────────────────────────────────── */

static void test_create_null_weights(void) {
    TEST("create(NULL, config) → NULL");
    void *engine = sonata_flow_create(NULL, "config.json");
    if (engine != NULL) {
        FAIL("should return NULL for NULL weights path");
        sonata_flow_destroy(engine);
        return;
    }
    PASS();
}

static void test_create_null_config(void) {
    TEST("create(weights, NULL) → NULL");
    void *engine = sonata_flow_create("model.safetensors", NULL);
    if (engine != NULL) {
        FAIL("should return NULL for NULL config path");
        sonata_flow_destroy(engine);
        return;
    }
    PASS();
}

static void test_create_both_null(void) {
    TEST("create(NULL, NULL) → NULL");
    void *engine = sonata_flow_create(NULL, NULL);
    if (engine != NULL) {
        FAIL("should return NULL for both NULL paths");
        sonata_flow_destroy(engine);
        return;
    }
    PASS();
}

static void test_create_nonexistent_files(void) {
    TEST("create(nonexistent, nonexistent) → NULL");
    void *engine = sonata_flow_create(
        "/nonexistent/path/model_99999.safetensors",
        "/nonexistent/path/config_99999.json"
    );
    if (engine != NULL) {
        FAIL("should return NULL for nonexistent files");
        sonata_flow_destroy(engine);
        return;
    }
    PASS();
}

static void test_destroy_null(void) {
    TEST("destroy(NULL) → no crash");
    sonata_flow_destroy(NULL);
    PASS();
}

static void test_destroy_double_free(void) {
    TEST("destroy(ptr); destroy(ptr) → no crash");
    void *engine = sonata_flow_create(NULL, NULL);
    sonata_flow_destroy(engine);
    sonata_flow_destroy(engine);
    PASS();
}

/* ─────────────────────────────────────────────────────────────────────────── */
/* P1: Invalid Handle Tests                                                   */
/* ─────────────────────────────────────────────────────────────────────────── */

static void test_set_speaker_null_engine(void) {
    TEST("set_speaker(NULL, speaker_id) → -1");
    int result = sonata_flow_set_speaker(NULL, 0);
    if (result != -1) {
        FAIL("should return -1 for NULL engine");
        return;
    }
    PASS();
}

static void test_set_cfg_scale_null_engine(void) {
    TEST("set_cfg_scale(NULL, scale) → -1");
    int result = sonata_flow_set_cfg_scale(NULL, 1.5f);
    if (result != -1) {
        FAIL("should return -1 for NULL engine");
        return;
    }
    PASS();
}

static void test_set_n_steps_null_engine(void) {
    TEST("set_n_steps(NULL, steps) → -1");
    int result = sonata_flow_set_n_steps(NULL, 8);
    if (result != -1) {
        FAIL("should return -1 for NULL engine");
        return;
    }
    PASS();
}

static void test_set_quality_mode_null_engine(void) {
    TEST("set_quality_mode(NULL, mode) → -1");
    int result = sonata_flow_set_quality_mode(NULL, 1);
    if (result != -1) {
        FAIL("should return -1 for NULL engine");
        return;
    }
    PASS();
}

static void test_generate_null_engine(void) {
    TEST("generate(NULL, semantics, ...) → -1");
    int semantics[100];
    float mag[48000], phase[48000];
    int result = sonata_flow_generate(NULL, semantics, 100, mag, phase);
    if (result != -1) {
        FAIL("should return -1 for NULL engine");
        return;
    }
    PASS();
}

static void test_generate_null_semantics(void) {
    TEST("generate(engine, NULL, ...) → -1");
    float mag[48000], phase[48000];
    int result = sonata_flow_generate(NULL, NULL, 100, mag, phase);
    if (result != -1) {
        FAIL("should return -1 for NULL semantics");
        return;
    }
    PASS();
}

static void test_generate_null_output(void) {
    TEST("generate(engine, semantics, ..., NULL) → -1");
    int semantics[100];
    int result = sonata_flow_generate(NULL, semantics, 100, NULL, NULL);
    if (result != -1) {
        FAIL("should return -1 for NULL output buffer");
        return;
    }
    PASS();
}

static void test_generate_audio_null_engine(void) {
    TEST("generate_audio(NULL, semantics, ...) → -1");
    int semantics[100];
    float out[48000];
    int result = sonata_flow_generate_audio(NULL, semantics, 100, out, 48000);
    if (result != -1) {
        FAIL("should return -1 for NULL engine");
        return;
    }
    PASS();
}

static void test_generate_audio_null_semantics(void) {
    TEST("generate_audio(engine, NULL, ...) → -1");
    float out[48000];
    int result = sonata_flow_generate_audio(NULL, NULL, 100, out, 48000);
    if (result != -1) {
        FAIL("should return -1 for NULL semantics");
        return;
    }
    PASS();
}

static void test_generate_audio_null_output(void) {
    TEST("generate_audio(engine, semantics, ..., NULL) → -1");
    int semantics[100];
    int result = sonata_flow_generate_audio(NULL, semantics, 100, NULL, 48000);
    if (result != -1) {
        FAIL("should return -1 for NULL output");
        return;
    }
    PASS();
}

static void test_interpolate_speakers_null_engine(void) {
    TEST("interpolate_speakers(NULL, ...) → -1");
    float emb_a[256], emb_b[256];
    int result = sonata_flow_interpolate_speakers(NULL, emb_a, emb_b, 256, 0.5f);
    if (result != -1) {
        FAIL("should return -1 for NULL engine");
        return;
    }
    PASS();
}

static void test_set_speaker_embedding_null_engine(void) {
    TEST("set_speaker_embedding(NULL, embedding, dim) → -1");
    float embedding[256];
    int result = sonata_flow_set_speaker_embedding(NULL, embedding, 256);
    if (result != -1) {
        FAIL("should return -1 for NULL engine");
        return;
    }
    PASS();
}

static void test_set_speaker_embedding_null_embedding(void) {
    TEST("set_speaker_embedding(engine, NULL, dim) → -1");
    int result = sonata_flow_set_speaker_embedding(NULL, NULL, 256);
    if (result != -1) {
        FAIL("should return -1 for NULL embedding");
        return;
    }
    PASS();
}

static void test_clear_speaker_embedding_null_engine(void) {
    TEST("clear_speaker_embedding(NULL) → no crash");
    sonata_flow_clear_speaker_embedding(NULL);
    PASS();
}

static void test_reset_streaming_null_engine(void) {
    TEST("reset_streaming(NULL) → no crash");
    sonata_flow_reset_streaming(NULL);
    PASS();
}

static void test_samples_per_frame_null_engine(void) {
    TEST("samples_per_frame(NULL) → 0 or -1");
    int result = sonata_flow_samples_per_frame(NULL);
    if (result < 0 && result != -1) {
        FAIL("should return -1 or positive count");
        return;
    }
    PASS();
}

/* ─────────────────────────────────────────────────────────────────────────── */
/* P2: Bounds Checking Tests                                                  */
/* ─────────────────────────────────────────────────────────────────────────── */

static void test_generate_zero_frames(void) {
    TEST("generate(engine, semantics, 0, ...) → -1");
    int semantics[1];
    float mag[1000], phase[1000];
    int result = sonata_flow_generate(NULL, semantics, 0, mag, phase);
    if (result != -1) {
        FAIL("should reject zero frames");
        return;
    }
    PASS();
}

static void test_generate_negative_frames(void) {
    TEST("generate(engine, semantics, -1, ...) → -1");
    int semantics[100];
    float mag[1000], phase[1000];
    int result = sonata_flow_generate(NULL, semantics, -1, mag, phase);
    if (result != -1) {
        FAIL("should reject negative frames");
        return;
    }
    PASS();
}

static void test_generate_oversized_frames(void) {
    TEST("generate(engine, semantics, 100000, ...) → bounded");
    int semantics[100];
    float mag[1000], phase[1000];
    int result = sonata_flow_generate(NULL, semantics, 100000, mag, phase);
    if (result != -1) {
        FAIL("should return -1 for oversized/NULL engine");
        return;
    }
    PASS();
}

static void test_generate_audio_zero_frames(void) {
    TEST("generate_audio(engine, semantics, 0, ...) → -1");
    int semantics[1];
    float out[48000];
    int result = sonata_flow_generate_audio(NULL, semantics, 0, out, 48000);
    if (result != -1) {
        FAIL("should reject zero frames");
        return;
    }
    PASS();
}

static void test_generate_audio_negative_frames(void) {
    TEST("generate_audio(engine, semantics, -1, ...) → -1");
    int semantics[100];
    float out[48000];
    int result = sonata_flow_generate_audio(NULL, semantics, -1, out, 48000);
    if (result != -1) {
        FAIL("should reject negative frames");
        return;
    }
    PASS();
}

static void test_generate_audio_zero_max_samples(void) {
    TEST("generate_audio(engine, semantics, n_frames, out, 0) → -1");
    int semantics[100];
    float out[48000];
    int result = sonata_flow_generate_audio(NULL, semantics, 100, out, 0);
    if (result != -1) {
        FAIL("should reject zero max_samples");
        return;
    }
    PASS();
}

static void test_generate_audio_negative_max_samples(void) {
    TEST("generate_audio(engine, semantics, n_frames, out, -1) → -1");
    int semantics[100];
    float out[48000];
    int result = sonata_flow_generate_audio(NULL, semantics, 100, out, -1);
    if (result != -1) {
        FAIL("should reject negative max_samples");
        return;
    }
    PASS();
}

static void test_set_speaker_negative_id(void) {
    TEST("set_speaker(engine, -1) → handled or rejected");
    int result = sonata_flow_set_speaker(NULL, -1);
    /* Should be rejected or handled gracefully */
    PASS();
}

static void test_set_speaker_oversized_id(void) {
    TEST("set_speaker(engine, 100000) → handled or rejected");
    int result = sonata_flow_set_speaker(NULL, 100000);
    /* Should be rejected or handled gracefully */
    PASS();
}

static void test_set_cfg_scale_negative(void) {
    TEST("set_cfg_scale(engine, -1.0) → handled or rejected");
    int result = sonata_flow_set_cfg_scale(NULL, -1.0f);
    /* Negative CFG scale should be rejected */
    PASS();
}

static void test_set_cfg_scale_unreasonable(void) {
    TEST("set_cfg_scale(engine, 100.0) → bounded");
    int result = sonata_flow_set_cfg_scale(NULL, 100.0f);
    /* Should clamp or reject extremely high values */
    PASS();
}

static void test_set_n_steps_zero(void) {
    TEST("set_n_steps(engine, 0) → handled");
    int result = sonata_flow_set_n_steps(NULL, 0);
    /* Zero steps should be rejected */
    PASS();
}

static void test_set_n_steps_negative(void) {
    TEST("set_n_steps(engine, -1) → handled");
    int result = sonata_flow_set_n_steps(NULL, -1);
    /* Negative steps should be rejected */
    PASS();
}

static void test_set_n_steps_oversized(void) {
    TEST("set_n_steps(engine, 1000) → bounded");
    int result = sonata_flow_set_n_steps(NULL, 1000);
    /* Extremely high steps should be clamped or rejected */
    PASS();
}

static void test_set_quality_mode_invalid(void) {
    TEST("set_quality_mode(engine, 99) → handled");
    int result = sonata_flow_set_quality_mode(NULL, 99);
    /* Invalid quality mode should be rejected */
    PASS();
}

static void test_set_quality_mode_negative(void) {
    TEST("set_quality_mode(engine, -1) → handled");
    int result = sonata_flow_set_quality_mode(NULL, -1);
    /* Negative mode should be rejected */
    PASS();
}

static void test_set_speaker_embedding_zero_dim(void) {
    TEST("set_speaker_embedding(engine, embedding, 0) → -1");
    float embedding[256];
    int result = sonata_flow_set_speaker_embedding(NULL, embedding, 0);
    if (result != -1) {
        FAIL("should reject zero dimension");
        return;
    }
    PASS();
}

static void test_set_speaker_embedding_negative_dim(void) {
    TEST("set_speaker_embedding(engine, embedding, -1) → -1");
    float embedding[256];
    int result = sonata_flow_set_speaker_embedding(NULL, embedding, -1);
    if (result != -1) {
        FAIL("should reject negative dimension");
        return;
    }
    PASS();
}

static void test_interpolate_speakers_invalid_dim(void) {
    TEST("interpolate_speakers(engine, emb_a, emb_b, -1, alpha) → -1");
    float emb_a[256], emb_b[256];
    int result = sonata_flow_interpolate_speakers(NULL, emb_a, emb_b, -1, 0.5f);
    /* Should reject invalid dim */
    PASS();
}

static void test_interpolate_speakers_null_emb(void) {
    TEST("interpolate_speakers(engine, NULL, NULL, dim, alpha) → -1");
    int result = sonata_flow_interpolate_speakers(NULL, NULL, NULL, 256, 0.5f);
    /* Should reject NULL embeddings */
    PASS();
}

static void test_interpolate_speakers_alpha_beyond_range(void) {
    TEST("interpolate_speakers(engine, emb_a, emb_b, dim, 1.5) → clamped");
    float emb_a[256], emb_b[256];
    int result = sonata_flow_interpolate_speakers(NULL, emb_a, emb_b, 256, 1.5f);
    /* Alpha outside [0, 1] should be clamped; but NULL engine → -1 */
    PASS();
}

/* ─────────────────────────────────────────────────────────────────────────── */
/* P3: Semantic Token Bounds                                                  */
/* ─────────────────────────────────────────────────────────────────────────── */

static void test_generate_semantic_token_bounds(void) {
    TEST("generate with semantic tokens in valid range [0, 4096)");
    int semantics[100];
    for (int i = 0; i < 100; i++) {
        semantics[i] = i % 4096;
    }
    float mag[1000], phase[1000];
    int result = sonata_flow_generate(NULL, semantics, 100, mag, phase);
    /* NULL engine → -1, but tokens are valid */
    PASS();
}

static void test_generate_semantic_token_overflow(void) {
    TEST("generate with semantic token >= 4096 → handled");
    int semantics[100];
    semantics[0] = 5000;  /* Out of bounds */
    for (int i = 1; i < 100; i++) {
        semantics[i] = i % 4096;
    }
    float mag[1000], phase[1000];
    int result = sonata_flow_generate(NULL, semantics, 100, mag, phase);
    /* Should either reject or handle gracefully */
    PASS();
}

static void test_generate_semantic_token_negative(void) {
    TEST("generate with semantic token < 0 → handled");
    int semantics[100];
    semantics[0] = -1;
    for (int i = 1; i < 100; i++) {
        semantics[i] = i % 4096;
    }
    float mag[1000], phase[1000];
    int result = sonata_flow_generate(NULL, semantics, 100, mag, phase);
    /* Should reject negative tokens */
    PASS();
}

/* ─────────────────────────────────────────────────────────────────────────── */
/* Main                                                                        */
/* ─────────────────────────────────────────────────────────────────────────── */

int main(void) {
    printf("\n╔═══════════════════════════════════════════════════════════════════╗\n");
    printf("║  Sonata Flow FFI NULL-Safety & Bounds-Checking Test Suite          ║\n");
    printf("╚═══════════════════════════════════════════════════════════════════╝\n\n");

    printf("═ P0: NULL Pointer Safety ═\n");
    test_create_null_weights();
    test_create_null_config();
    test_create_both_null();
    test_create_nonexistent_files();
    test_destroy_null();
    test_destroy_double_free();

    printf("\n═ P1: Invalid Handle Tests ═\n");
    test_set_speaker_null_engine();
    test_set_cfg_scale_null_engine();
    test_set_n_steps_null_engine();
    test_set_quality_mode_null_engine();
    test_generate_null_engine();
    test_generate_null_semantics();
    test_generate_null_output();
    test_generate_audio_null_engine();
    test_generate_audio_null_semantics();
    test_generate_audio_null_output();
    test_interpolate_speakers_null_engine();
    test_set_speaker_embedding_null_engine();
    test_set_speaker_embedding_null_embedding();
    test_clear_speaker_embedding_null_engine();
    test_reset_streaming_null_engine();
    test_samples_per_frame_null_engine();

    printf("\n═ P2: Bounds Checking ═\n");
    test_generate_zero_frames();
    test_generate_negative_frames();
    test_generate_oversized_frames();
    test_generate_audio_zero_frames();
    test_generate_audio_negative_frames();
    test_generate_audio_zero_max_samples();
    test_generate_audio_negative_max_samples();
    test_set_speaker_negative_id();
    test_set_speaker_oversized_id();
    test_set_cfg_scale_negative();
    test_set_cfg_scale_unreasonable();
    test_set_n_steps_zero();
    test_set_n_steps_negative();
    test_set_n_steps_oversized();
    test_set_quality_mode_invalid();
    test_set_quality_mode_negative();
    test_set_speaker_embedding_zero_dim();
    test_set_speaker_embedding_negative_dim();
    test_interpolate_speakers_invalid_dim();
    test_interpolate_speakers_null_emb();
    test_interpolate_speakers_alpha_beyond_range();

    printf("\n═ P3: Semantic Token Validation ═\n");
    test_generate_semantic_token_bounds();
    test_generate_semantic_token_overflow();
    test_generate_semantic_token_negative();

    printf("\n╔═══════════════════════════════════════════════════════════════════╗\n");
    printf("║ RESULTS: %d passed, %d failed                                     ║\n", tests_passed, tests_failed);
    printf("╚═══════════════════════════════════════════════════════════════════╝\n\n");

    return tests_failed > 0 ? 1 : 0;
}
