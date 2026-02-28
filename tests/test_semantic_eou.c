/**
 * test_semantic_eou.c — Semantic EOU predictor unit + integration tests.
 *
 * Tests the byte-level LSTM sentence completion predictor and its
 * integration with the fused EOU system as the 5th signal.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

/* ── SemanticEOU API (forward declarations) ─────────────────────────────── */

typedef struct SemanticEOU SemanticEOU;
extern SemanticEOU *semantic_eou_create(void);
extern void semantic_eou_destroy(SemanticEOU *se);
extern int semantic_eou_load_weights(SemanticEOU *se, const char *path);
extern void semantic_eou_init_random(SemanticEOU *se, uint32_t seed);
extern float semantic_eou_process(SemanticEOU *se, const char *text);
extern void semantic_eou_reset(SemanticEOU *se);
extern int semantic_eou_word_count(const char *text);

/* ── FusedEOU API (forward declarations for integration tests) ──────────── */

typedef struct FusedEOU FusedEOU;

typedef struct {
    float energy_signal;
    float mimi_eot_prob;
    float stt_eou_prob;
} EOUSignals;

typedef struct {
    float fused_prob;
    int   triggered;
    int   trigger_source;
    float latency_ms;
    int   consec_frames;
} EOUResult;

#define EOU_SRC_ENERGY   (1 << 0)
#define EOU_SRC_MIMI     (1 << 1)
#define EOU_SRC_STT      (1 << 2)
#define EOU_SRC_FUSED    (1 << 3)
#define EOU_SRC_PROSODY  (1 << 4)
#define EOU_SRC_SEMANTIC (1 << 5)

extern FusedEOU *fused_eou_create(float threshold, int consec_frames, float frame_ms);
extern void fused_eou_destroy(FusedEOU *eou);
extern EOUResult fused_eou_process(FusedEOU *eou, EOUSignals signals);
extern void fused_eou_reset(FusedEOU *eou);
extern void fused_eou_feed_semantic(FusedEOU *eou, float prob);
extern void fused_eou_set_semantic_weight(FusedEOU *eou, float w_semantic);
extern float fused_eou_semantic_prob(const FusedEOU *eou);
extern void fused_eou_set_weights(FusedEOU *eou, float w_energy, float w_mimi, float w_stt);

/* ── Test framework ─────────────────────────────────────────────────────── */

static int passed = 0, failed = 0;
#define CHECK(cond, msg) do { \
    if (cond) { printf("  [PASS] %s\n", msg); passed++; } \
    else { printf("  [FAIL] %s\n", msg); failed++; } \
} while (0)

/* ── SemanticEOU unit tests ─────────────────────────────────────────────── */

static void test_create_destroy(void) {
    printf("── Create/Destroy ──\n");

    SemanticEOU *se = semantic_eou_create();
    CHECK(se != NULL, "create returns non-NULL");

    semantic_eou_destroy(se);
    CHECK(1, "destroy is safe");

    semantic_eou_destroy(NULL);
    CHECK(1, "destroy(NULL) is safe");
}

static void test_null_safety(void) {
    printf("\n── NULL Safety ──\n");

    CHECK(semantic_eou_process(NULL, "hello") == 0.5f,
          "process(NULL, text) returns 0.5");
    CHECK(semantic_eou_process(NULL, NULL) == 0.5f,
          "process(NULL, NULL) returns 0.5");

    semantic_eou_reset(NULL);
    CHECK(1, "reset(NULL) is safe");

    CHECK(semantic_eou_load_weights(NULL, "foo.bin") == -1,
          "load_weights(NULL, path) returns -1");
    CHECK(semantic_eou_load_weights(NULL, NULL) == -1,
          "load_weights(NULL, NULL) returns -1");

    semantic_eou_init_random(NULL, 42);
    CHECK(1, "init_random(NULL) is safe");
}

static void test_empty_input(void) {
    printf("\n── Empty/NULL Input ──\n");

    SemanticEOU *se = semantic_eou_create();
    CHECK(se != NULL, "create for empty input test");
    if (!se) return;

    semantic_eou_init_random(se, 42);

    float p_null = semantic_eou_process(se, NULL);
    CHECK(p_null == 0.5f, "process(NULL text) returns 0.5 (neutral)");

    float p_empty = semantic_eou_process(se, "");
    CHECK(p_empty == 0.5f, "process(empty text) returns 0.5 (neutral)");

    semantic_eou_destroy(se);
}

static void test_random_init_output_range(void) {
    printf("\n── Random Init Output Range ──\n");

    SemanticEOU *se = semantic_eou_create();
    CHECK(se != NULL, "create for range test");
    if (!se) return;

    semantic_eou_init_random(se, 42);

    const char *texts[] = {
        "hello world",
        "Can you tell me about",
        "The answer is yes.",
        "I think that",
        "This is a complete sentence.",
        "um well actually I was going to say",
        ("A very long text input that should be truncated to the last 128 bytes "
        "by the model to ensure we handle long inputs correctly without issues."),
    };
    int n_texts = sizeof(texts) / sizeof(texts[0]);

    for (int i = 0; i < n_texts; i++) {
        float p = semantic_eou_process(se, texts[i]);
        CHECK(p >= 0.0f && p <= 1.0f, "probability in [0, 1] range");
    }

    semantic_eou_destroy(se);
}

static void test_deterministic(void) {
    printf("\n── Deterministic Output ──\n");

    SemanticEOU *se1 = semantic_eou_create();
    SemanticEOU *se2 = semantic_eou_create();
    CHECK(se1 != NULL && se2 != NULL, "create two instances");
    if (!se1 || !se2) { semantic_eou_destroy(se1); semantic_eou_destroy(se2); return; }

    semantic_eou_init_random(se1, 42);
    semantic_eou_init_random(se2, 42);

    float p1 = semantic_eou_process(se1, "The answer is yes");
    float p2 = semantic_eou_process(se2, "The answer is yes");
    CHECK(fabsf(p1 - p2) < 1e-6f, "same seed + same input → same output");

    semantic_eou_destroy(se1);
    semantic_eou_destroy(se2);
}

static void test_different_seeds(void) {
    printf("\n── Different Seeds ──\n");

    SemanticEOU *se1 = semantic_eou_create();
    SemanticEOU *se2 = semantic_eou_create();
    CHECK(se1 != NULL && se2 != NULL, "create two instances");
    if (!se1 || !se2) { semantic_eou_destroy(se1); semantic_eou_destroy(se2); return; }

    semantic_eou_init_random(se1, 42);
    semantic_eou_init_random(se2, 123);

    float p1 = semantic_eou_process(se1, "The answer is yes");
    float p2 = semantic_eou_process(se2, "The answer is yes");
    CHECK(fabsf(p1 - p2) > 1e-6f, "different seeds → different output");

    semantic_eou_destroy(se1);
    semantic_eou_destroy(se2);
}

static void test_reset(void) {
    printf("\n── Reset ──\n");

    SemanticEOU *se = semantic_eou_create();
    CHECK(se != NULL, "create for reset test");
    if (!se) return;

    semantic_eou_init_random(se, 42);

    float p1 = semantic_eou_process(se, "hello world");
    semantic_eou_reset(se);
    float p2 = semantic_eou_process(se, "hello world");
    /* Process resets state internally each call, so these should match */
    CHECK(fabsf(p1 - p2) < 1e-6f, "reset → same input → same output");

    /* Multiple resets are safe */
    semantic_eou_reset(se);
    semantic_eou_reset(se);
    semantic_eou_reset(se);
    float p3 = semantic_eou_process(se, "hello world");
    CHECK(fabsf(p1 - p3) < 1e-6f, "multiple resets are safe");

    semantic_eou_destroy(se);
}

static void test_word_count(void) {
    printf("\n── Word Count ──\n");

    CHECK(semantic_eou_word_count(NULL) == 0, "NULL → 0 words");
    CHECK(semantic_eou_word_count("") == 0, "empty → 0 words");
    CHECK(semantic_eou_word_count("hello") == 1, "one word");
    CHECK(semantic_eou_word_count("hello world") == 2, "two words");
    CHECK(semantic_eou_word_count("  hello   world  ") == 2, "two words with extra spaces");
    CHECK(semantic_eou_word_count("a b c") == 3, "three words");
    CHECK(semantic_eou_word_count("The answer is yes.") == 4, "four words");
    CHECK(semantic_eou_word_count("\thello\nworld\r\n") == 2, "whitespace variants");
}

static void test_long_input(void) {
    printf("\n── Long Input Handling ──\n");

    SemanticEOU *se = semantic_eou_create();
    CHECK(se != NULL, "create for long input test");
    if (!se) return;

    semantic_eou_init_random(se, 42);

    /* Create a very long string (>128 bytes) */
    char long_text[1024];
    memset(long_text, 'a', sizeof(long_text) - 1);
    long_text[sizeof(long_text) - 1] = '\0';
    /* Insert spaces for words */
    for (int i = 10; i < 1000; i += 10) long_text[i] = ' ';

    float p = semantic_eou_process(se, long_text);
    CHECK(p >= 0.0f && p <= 1.0f, "long input produces valid probability");

    semantic_eou_destroy(se);
}

static void test_special_characters(void) {
    printf("\n── Special Characters ──\n");

    SemanticEOU *se = semantic_eou_create();
    CHECK(se != NULL, "create for special char test");
    if (!se) return;

    semantic_eou_init_random(se, 42);

    /* UTF-8 multibyte, punctuation, control chars */
    const char *specials[] = {
        "Hello! How are you?",
        "caf\xc3\xa9 latt\xc3\xa9",  /* UTF-8: café latté */
        "line1\nline2\ttab",
        "price: $99.99 (50% off)",
        "\x01\x02\x03",  /* control characters */
    };
    int n = sizeof(specials) / sizeof(specials[0]);

    for (int i = 0; i < n; i++) {
        float p = semantic_eou_process(se, specials[i]);
        CHECK(p >= 0.0f && p <= 1.0f, "special chars produce valid probability");
    }

    semantic_eou_destroy(se);
}

static void test_load_bad_files(void) {
    printf("\n── Bad Weight Files ──\n");

    SemanticEOU *se = semantic_eou_create();
    CHECK(se != NULL, "create for bad file test");
    if (!se) return;

    CHECK(semantic_eou_load_weights(se, "nonexistent.seou") == -1,
          "load nonexistent file returns -1");
    CHECK(semantic_eou_load_weights(se, "") == -1,
          "load empty path returns -1");
    CHECK(semantic_eou_load_weights(se, NULL) == -1,
          "load NULL path returns -1");

    semantic_eou_destroy(se);
}

/* ── Fused EOU integration tests ────────────────────────────────────────── */

static void test_fused_semantic_disabled(void) {
    printf("\n── Fused: Semantic Disabled (default) ──\n");

    FusedEOU *eou = fused_eou_create(0.6f, 2, 80.0f);
    CHECK(eou != NULL, "fused_eou_create");
    if (!eou) return;

    /* Semantic weight defaults to 0 — feeding it should not affect fusion */
    fused_eou_feed_semantic(eou, 0.99f);
    CHECK(fused_eou_semantic_prob(eou) > 0.98f, "semantic prob stored correctly");

    /* Process with moderate signals — semantic should not contribute */
    EOUSignals sig = { .energy_signal = 0.3f, .mimi_eot_prob = 0.4f, .stt_eou_prob = 0.5f };
    /* Feed speech first so speech_detected = 1 */
    EOUSignals speech = { .energy_signal = 0.0f, .mimi_eot_prob = 0.0f, .stt_eou_prob = 0.0f };
    for (int i = 0; i < 5; i++) {
        speech.energy_signal = 0.1f;  /* Low energy = speech active */
        fused_eou_process(eou, speech);
    }

    EOUResult r1 = fused_eou_process(eou, sig);
    CHECK(r1.fused_prob >= 0.0f && r1.fused_prob <= 1.0f, "fused prob in range");

    /* Without semantic weight, the high semantic prob should not trigger solo */
    CHECK(!(r1.trigger_source & EOU_SRC_SEMANTIC), "semantic not in trigger source when disabled");

    fused_eou_destroy(eou);
}

static void test_fused_semantic_enabled(void) {
    printf("\n── Fused: Semantic Enabled ──\n");

    FusedEOU *eou = fused_eou_create(0.6f, 2, 80.0f);
    CHECK(eou != NULL, "fused_eou_create");
    if (!eou) return;

    fused_eou_set_semantic_weight(eou, 0.15f);

    /* Feed high semantic probability */
    fused_eou_feed_semantic(eou, 0.9f);

    /* Feed enough speech frames to set speech_detected */
    EOUSignals speech = { .energy_signal = 0.1f, .mimi_eot_prob = 0.0f, .stt_eou_prob = 0.0f };
    for (int i = 0; i < 5; i++)
        fused_eou_process(eou, speech);

    /* Now process with moderate signals + high semantic */
    EOUSignals sig = { .energy_signal = 0.5f, .mimi_eot_prob = 0.3f, .stt_eou_prob = 0.6f };
    EOUResult r1 = fused_eou_process(eou, sig);

    /* The semantic signal should contribute to the fused probability */
    CHECK(r1.fused_prob >= 0.0f && r1.fused_prob <= 1.0f, "fused prob in range with semantic");

    fused_eou_destroy(eou);
}

static void test_fused_semantic_solo_trigger(void) {
    printf("\n── Fused: Semantic Solo Trigger ──\n");

    FusedEOU *eou = fused_eou_create(0.6f, 1, 80.0f);
    CHECK(eou != NULL, "fused_eou_create");
    if (!eou) return;

    fused_eou_set_semantic_weight(eou, 0.15f);

    /* Feed very high semantic probability (above solo threshold 0.88) */
    fused_eou_feed_semantic(eou, 0.95f);

    /* Feed speech frames */
    EOUSignals speech = { .energy_signal = 0.1f, .mimi_eot_prob = 0.0f, .stt_eou_prob = 0.0f };
    for (int i = 0; i < 5; i++)
        fused_eou_process(eou, speech);

    /* Process with low other signals — semantic should trigger solo */
    EOUSignals sig = { .energy_signal = 0.3f, .mimi_eot_prob = 0.1f, .stt_eou_prob = 0.2f };
    EOUResult r = fused_eou_process(eou, sig);

    CHECK(r.trigger_source & EOU_SRC_SEMANTIC, "semantic solo trigger fires");

    fused_eou_destroy(eou);
}

static void test_fused_semantic_weight_clamping(void) {
    printf("\n── Fused: Semantic Weight Clamping ──\n");

    FusedEOU *eou = fused_eou_create(0.6f, 2, 80.0f);
    CHECK(eou != NULL, "fused_eou_create");
    if (!eou) return;

    /* Negative weight should clamp to 0 */
    fused_eou_set_semantic_weight(eou, -0.5f);
    CHECK(fused_eou_semantic_prob(eou) == 0.0f, "negative weight → semantic prob not affected");

    /* Excessive weight should clamp to 0.4 */
    fused_eou_set_semantic_weight(eou, 0.8f);
    /* Can't directly read weight, but feeding high prob shouldn't crash */
    fused_eou_feed_semantic(eou, 0.9f);
    CHECK(fused_eou_semantic_prob(eou) > 0.89f, "semantic prob stored after high weight");

    fused_eou_destroy(eou);
}

static void test_fused_semantic_nan_inf(void) {
    printf("\n── Fused: Semantic NaN/Inf Safety ──\n");

    FusedEOU *eou = fused_eou_create(0.6f, 2, 80.0f);
    CHECK(eou != NULL, "fused_eou_create");
    if (!eou) return;

    fused_eou_set_semantic_weight(eou, 0.15f);

    /* Feed NaN — should be sanitized to 0.5 */
    fused_eou_feed_semantic(eou, 0.0f / 0.0f);  /* NaN */
    float p_nan = fused_eou_semantic_prob(eou);
    CHECK(p_nan >= 0.0f && p_nan <= 1.0f, "NaN sanitized to valid range");

    /* Feed +Inf — should be sanitized */
    fused_eou_feed_semantic(eou, 1.0f / 0.0f);  /* +Inf */
    float p_inf = fused_eou_semantic_prob(eou);
    CHECK(p_inf >= 0.0f && p_inf <= 1.0f, "+Inf sanitized to valid range");

    /* Feed -Inf — should be sanitized */
    fused_eou_feed_semantic(eou, -1.0f / 0.0f);  /* -Inf */
    float p_ninf = fused_eou_semantic_prob(eou);
    CHECK(p_ninf >= 0.0f && p_ninf <= 1.0f, "-Inf sanitized to valid range");

    /* NULL safety */
    fused_eou_feed_semantic(NULL, 0.5f);
    CHECK(1, "feed_semantic(NULL) is safe");

    fused_eou_set_semantic_weight(NULL, 0.15f);
    CHECK(1, "set_semantic_weight(NULL) is safe");

    CHECK(fused_eou_semantic_prob(NULL) == 0.0f, "semantic_prob(NULL) returns 0");

    fused_eou_destroy(eou);
}

static void test_fused_semantic_reset(void) {
    printf("\n── Fused: Semantic Reset ──\n");

    FusedEOU *eou = fused_eou_create(0.6f, 2, 80.0f);
    CHECK(eou != NULL, "fused_eou_create");
    if (!eou) return;

    fused_eou_set_semantic_weight(eou, 0.15f);
    fused_eou_feed_semantic(eou, 0.9f);
    CHECK(fused_eou_semantic_prob(eou) > 0.89f, "semantic prob set before reset");

    fused_eou_reset(eou);
    CHECK(fused_eou_semantic_prob(eou) == 0.0f, "semantic prob cleared after reset");

    fused_eou_destroy(eou);
}

/* ── Main ───────────────────────────────────────────────────────────────── */

int main(void) {
    printf("═══ Semantic EOU Tests ═══\n\n");

    /* SemanticEOU unit tests */
    test_create_destroy();
    test_null_safety();
    test_empty_input();
    test_random_init_output_range();
    test_deterministic();
    test_different_seeds();
    test_reset();
    test_word_count();
    test_long_input();
    test_special_characters();
    test_load_bad_files();

    /* Fused EOU integration tests */
    test_fused_semantic_disabled();
    test_fused_semantic_enabled();
    test_fused_semantic_solo_trigger();
    test_fused_semantic_weight_clamping();
    test_fused_semantic_nan_inf();
    test_fused_semantic_reset();

    printf("\n══════════════════════════════════════════\n");
    printf("Results: %d passed, %d failed\n", passed, failed);
    if (failed > 0) { printf("SOME TESTS FAILED\n"); return 1; }
    printf("ALL PASSED\n");
    return 0;
}
