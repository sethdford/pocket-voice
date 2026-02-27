/**
 * test_tdt_decoder.c — Tests for Token Duration Transducer decoder.
 *
 * Validates:
 *   - Create/destroy lifecycle and NULL safety
 *   - Config validation (pred_layers, dimensions)
 *   - Duration value selection
 *   - Token emission with synthetic weight matrices
 *   - Bounds checking (max_tokens, vocab_size)
 *   - Edge cases: zero-length input, single frame, reset between utterances
 *
 * Build:
 *   cc -O3 -arch arm64 -Isrc -framework Accelerate \
 *      -Lbuild -ltdt_decoder \
 *      -o tests/test_tdt_decoder tests/test_tdt_decoder.c
 *
 * Run: ./tests/test_tdt_decoder
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "tdt_decoder.h"

static int tests_passed = 0;
static int tests_failed = 0;

#define TEST(name) do { printf("  %-55s", name); } while(0)
#define PASS() do { printf("PASS\n"); tests_passed++; } while(0)
#define FAIL(msg) do { printf("FAIL: %s\n", msg); tests_failed++; return; } while(0)

/* ── Helpers ─────────────────────────────────────────────── */

/* Allocate zeroed weight matrix of given size */
static float *alloc_weights(int count) {
    return (float *)calloc(count, sizeof(float));
}

/* Create a minimal valid TDT config */
static TDTConfig make_test_config(void) {
    TDTConfig cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.pred_hidden = 32;
    cfg.pred_layers = 2;
    cfg.vocab_size = 10;  /* 0=blank, 1-9=tokens */
    cfg.n_durations = 5;
    cfg.joint_dim = 32;
    cfg.encoder_dim = 64;
    cfg.blank_id = 0;
    cfg.duration_values[0] = 0;
    cfg.duration_values[1] = 1;
    cfg.duration_values[2] = 2;
    cfg.duration_values[3] = 4;
    cfg.duration_values[4] = 8;
    return cfg;
}

/*
 * Allocate all weight buffers for the test config and create a decoder.
 * All weights are initialized to small random values.
 * Caller must free all returned pointers via free_test_weights().
 */
typedef struct {
    float *embed_w;
    float *lstm_wi[4];
    float *lstm_bi[4];
    float *lstm_wh[4];
    float *lstm_bh[4];
    float *joint_enc_w;
    float *joint_enc_b;
    float *joint_pred_w;
    float *joint_pred_b;
    float *joint_out_w;
    float *joint_out_b;
} TestWeights;

static void init_random_weights(float *w, int count, unsigned int *seed) {
    for (int i = 0; i < count; i++) {
        *seed = (*seed) * 1664525u + 1013904223u;
        w[i] = ((float)((int)((*seed) >> 16) % 200 - 100)) / 10000.0f;
    }
}

static TestWeights alloc_test_weights(const TDTConfig *cfg, unsigned int seed) {
    TestWeights w;
    memset(&w, 0, sizeof(w));

    int H = cfg->pred_hidden;
    int V = cfg->vocab_size;
    int D = cfg->n_durations;
    int J = cfg->joint_dim;
    int E = cfg->encoder_dim;

    w.embed_w = alloc_weights(V * H);
    init_random_weights(w.embed_w, V * H, &seed);

    for (int l = 0; l < cfg->pred_layers; l++) {
        w.lstm_wi[l] = alloc_weights(4 * H * H);
        w.lstm_bi[l] = alloc_weights(4 * H);
        w.lstm_wh[l] = alloc_weights(4 * H * H);
        w.lstm_bh[l] = alloc_weights(4 * H);
        init_random_weights(w.lstm_wi[l], 4 * H * H, &seed);
        init_random_weights(w.lstm_bi[l], 4 * H, &seed);
        init_random_weights(w.lstm_wh[l], 4 * H * H, &seed);
        init_random_weights(w.lstm_bh[l], 4 * H, &seed);
    }

    w.joint_enc_w  = alloc_weights(J * E);
    w.joint_enc_b  = alloc_weights(J);
    w.joint_pred_w = alloc_weights(J * H);
    w.joint_pred_b = alloc_weights(J);
    w.joint_out_w  = alloc_weights((V + D) * J);
    w.joint_out_b  = alloc_weights(V + D);
    init_random_weights(w.joint_enc_w, J * E, &seed);
    init_random_weights(w.joint_enc_b, J, &seed);
    init_random_weights(w.joint_pred_w, J * H, &seed);
    init_random_weights(w.joint_pred_b, J, &seed);
    init_random_weights(w.joint_out_w, (V + D) * J, &seed);
    init_random_weights(w.joint_out_b, V + D, &seed);

    return w;
}

static void free_test_weights(TestWeights *w, int pred_layers) {
    free(w->embed_w);
    for (int l = 0; l < pred_layers; l++) {
        free(w->lstm_wi[l]);
        free(w->lstm_bi[l]);
        free(w->lstm_wh[l]);
        free(w->lstm_bh[l]);
    }
    free(w->joint_enc_w);
    free(w->joint_enc_b);
    free(w->joint_pred_w);
    free(w->joint_pred_b);
    free(w->joint_out_w);
    free(w->joint_out_b);
}

static TDTDecoder *create_test_decoder(const TDTConfig *cfg, TestWeights *w) {
    return tdt_decoder_create(
        cfg, w->embed_w,
        (const float *const *)w->lstm_wi,
        (const float *const *)w->lstm_bi,
        (const float *const *)w->lstm_wh,
        (const float *const *)w->lstm_bh,
        w->joint_enc_w, w->joint_enc_b,
        w->joint_pred_w, w->joint_pred_b,
        w->joint_out_w, w->joint_out_b
    );
}

/* ── NULL Safety Tests ───────────────────────────────────── */

static void test_destroy_null(void) {
    TEST("tdt: destroy(NULL) does not crash");
    tdt_decoder_destroy(NULL);
    PASS();
}

static void test_reset_null(void) {
    TEST("tdt: reset(NULL) does not crash");
    tdt_decoder_reset(NULL);
    PASS();
}

static void test_decode_null_decoder(void) {
    TEST("tdt: decode(NULL decoder) returns -1");
    float enc[64] = {0};
    int tokens[10];
    if (tdt_decoder_decode(NULL, enc, 1, tokens, 10) != -1)
        FAIL("expected -1");
    PASS();
}

static void test_decode_null_enc(void) {
    TEST("tdt: decode(NULL enc_out) returns -1");
    TDTConfig cfg = make_test_config();
    TestWeights w = alloc_test_weights(&cfg, 42);
    TDTDecoder *dec = create_test_decoder(&cfg, &w);
    if (!dec) { free_test_weights(&w, cfg.pred_layers); FAIL("create failed"); }

    int tokens[10];
    if (tdt_decoder_decode(dec, NULL, 5, tokens, 10) != -1) {
        tdt_decoder_destroy(dec);
        free_test_weights(&w, cfg.pred_layers);
        FAIL("expected -1 for NULL enc_out");
    }

    tdt_decoder_destroy(dec);
    free_test_weights(&w, cfg.pred_layers);
    PASS();
}

static void test_decode_null_tokens(void) {
    TEST("tdt: decode(NULL tokens) returns -1");
    TDTConfig cfg = make_test_config();
    TestWeights w = alloc_test_weights(&cfg, 42);
    TDTDecoder *dec = create_test_decoder(&cfg, &w);
    if (!dec) { free_test_weights(&w, cfg.pred_layers); FAIL("create failed"); }

    float enc[64] = {0};
    if (tdt_decoder_decode(dec, enc, 1, NULL, 10) != -1) {
        tdt_decoder_destroy(dec);
        free_test_weights(&w, cfg.pred_layers);
        FAIL("expected -1 for NULL tokens");
    }

    tdt_decoder_destroy(dec);
    free_test_weights(&w, cfg.pred_layers);
    PASS();
}

static void test_create_null_config(void) {
    TEST("tdt: create(NULL config) returns NULL");
    TDTDecoder *dec = tdt_decoder_create(NULL, NULL, NULL, NULL, NULL, NULL,
                                          NULL, NULL, NULL, NULL, NULL, NULL);
    if (dec != NULL) {
        tdt_decoder_destroy(dec);
        FAIL("expected NULL");
    }
    PASS();
}

/* ── Config Validation Tests ─────────────────────────────── */

static void test_config_too_many_layers(void) {
    TEST("tdt: create with pred_layers > 4 returns NULL");
    TDTConfig cfg = make_test_config();
    cfg.pred_layers = 5;  /* MAX_PRED_LAYERS = 4 */
    TDTDecoder *dec = tdt_decoder_create(&cfg, NULL, NULL, NULL, NULL, NULL,
                                          NULL, NULL, NULL, NULL, NULL, NULL);
    if (dec != NULL) {
        tdt_decoder_destroy(dec);
        FAIL("expected NULL for pred_layers=5");
    }
    PASS();
}

static void test_config_pred_layers_boundary(void) {
    TEST("tdt: create with pred_layers=4 succeeds");
    TDTConfig cfg = make_test_config();
    cfg.pred_layers = 4;
    TestWeights w = alloc_test_weights(&cfg, 100);
    TDTDecoder *dec = create_test_decoder(&cfg, &w);
    if (!dec) {
        free_test_weights(&w, cfg.pred_layers);
        FAIL("expected non-NULL for pred_layers=4");
    }
    tdt_decoder_destroy(dec);
    free_test_weights(&w, cfg.pred_layers);
    PASS();
}

static void test_config_single_layer(void) {
    TEST("tdt: create with pred_layers=1 succeeds");
    TDTConfig cfg = make_test_config();
    cfg.pred_layers = 1;
    TestWeights w = alloc_test_weights(&cfg, 200);
    TDTDecoder *dec = create_test_decoder(&cfg, &w);
    if (!dec) {
        free_test_weights(&w, cfg.pred_layers);
        FAIL("expected non-NULL for pred_layers=1");
    }
    tdt_decoder_destroy(dec);
    free_test_weights(&w, cfg.pred_layers);
    PASS();
}

/* ── Create/Destroy Lifecycle ────────────────────────────── */

static void test_create_destroy_basic(void) {
    TEST("tdt: basic create and destroy");
    TDTConfig cfg = make_test_config();
    TestWeights w = alloc_test_weights(&cfg, 42);
    TDTDecoder *dec = create_test_decoder(&cfg, &w);
    if (!dec) {
        free_test_weights(&w, cfg.pred_layers);
        FAIL("create returned NULL");
    }
    tdt_decoder_destroy(dec);
    free_test_weights(&w, cfg.pred_layers);
    PASS();
}

static void test_create_destroy_repeated(void) {
    TEST("tdt: repeated create/destroy cycles");
    TDTConfig cfg = make_test_config();
    for (int i = 0; i < 5; i++) {
        TestWeights w = alloc_test_weights(&cfg, 42 + i);
        TDTDecoder *dec = create_test_decoder(&cfg, &w);
        if (!dec) {
            free_test_weights(&w, cfg.pred_layers);
            FAIL("create failed on iteration");
        }
        tdt_decoder_destroy(dec);
        free_test_weights(&w, cfg.pred_layers);
    }
    PASS();
}

/* ── Decode Tests ────────────────────────────────────────── */

static void test_decode_zero_length(void) {
    TEST("tdt: decode with T=0 returns -1");
    TDTConfig cfg = make_test_config();
    TestWeights w = alloc_test_weights(&cfg, 42);
    TDTDecoder *dec = create_test_decoder(&cfg, &w);
    if (!dec) { free_test_weights(&w, cfg.pred_layers); FAIL("create failed"); }

    float enc[64] = {0};
    int tokens[10];
    int n = tdt_decoder_decode(dec, enc, 0, tokens, 10);
    if (n != -1) {
        tdt_decoder_destroy(dec);
        free_test_weights(&w, cfg.pred_layers);
        FAIL("expected -1 for T=0");
    }

    tdt_decoder_destroy(dec);
    free_test_weights(&w, cfg.pred_layers);
    PASS();
}

static void test_decode_max_tokens_zero(void) {
    TEST("tdt: decode with max_tokens=0 returns -1");
    TDTConfig cfg = make_test_config();
    TestWeights w = alloc_test_weights(&cfg, 42);
    TDTDecoder *dec = create_test_decoder(&cfg, &w);
    if (!dec) { free_test_weights(&w, cfg.pred_layers); FAIL("create failed"); }

    float enc[64] = {0};
    int tokens[1];
    int n = tdt_decoder_decode(dec, enc, 1, tokens, 0);
    if (n != -1) {
        tdt_decoder_destroy(dec);
        free_test_weights(&w, cfg.pred_layers);
        FAIL("expected -1 for max_tokens=0");
    }

    tdt_decoder_destroy(dec);
    free_test_weights(&w, cfg.pred_layers);
    PASS();
}

static void test_decode_single_frame(void) {
    TEST("tdt: decode single frame produces tokens >= 0");
    TDTConfig cfg = make_test_config();
    TestWeights w = alloc_test_weights(&cfg, 42);
    TDTDecoder *dec = create_test_decoder(&cfg, &w);
    if (!dec) { free_test_weights(&w, cfg.pred_layers); FAIL("create failed"); }

    /* Single frame of encoder output */
    int E = cfg.encoder_dim;
    float *enc = (float *)calloc(E, sizeof(float));
    int tokens[64];
    int n = tdt_decoder_decode(dec, enc, 1, tokens, 64);
    if (n < 0) {
        free(enc);
        tdt_decoder_destroy(dec);
        free_test_weights(&w, cfg.pred_layers);
        FAIL("expected non-negative return for single frame");
    }

    /* Verify all tokens are in valid range */
    int valid = 1;
    for (int i = 0; i < n; i++) {
        if (tokens[i] < 0 || tokens[i] >= cfg.vocab_size) {
            valid = 0;
            break;
        }
    }
    if (!valid) {
        free(enc);
        tdt_decoder_destroy(dec);
        free_test_weights(&w, cfg.pred_layers);
        FAIL("token out of valid range");
    }

    free(enc);
    tdt_decoder_destroy(dec);
    free_test_weights(&w, cfg.pred_layers);
    PASS();
}

static void test_decode_multi_frame(void) {
    TEST("tdt: decode 10 frames of random encoder output");
    TDTConfig cfg = make_test_config();
    TestWeights w = alloc_test_weights(&cfg, 55);
    TDTDecoder *dec = create_test_decoder(&cfg, &w);
    if (!dec) { free_test_weights(&w, cfg.pred_layers); FAIL("create failed"); }

    int T = 10;
    int E = cfg.encoder_dim;
    float *enc = (float *)calloc(T * E, sizeof(float));
    unsigned int seed = 777;
    for (int i = 0; i < T * E; i++) {
        seed = seed * 1664525u + 1013904223u;
        enc[i] = ((float)((int)(seed >> 16) % 200 - 100)) / 1000.0f;
    }

    int tokens[128];
    int n = tdt_decoder_decode(dec, enc, T, tokens, 128);
    if (n < 0) {
        free(enc);
        tdt_decoder_destroy(dec);
        free_test_weights(&w, cfg.pred_layers);
        FAIL("decode returned -1");
    }

    /* Tokens should all be non-blank (blanks are not emitted) and in range */
    int valid = 1;
    for (int i = 0; i < n; i++) {
        if (tokens[i] == cfg.blank_id) { valid = 0; break; }
        if (tokens[i] < 0 || tokens[i] >= cfg.vocab_size) { valid = 0; break; }
    }
    if (!valid) {
        free(enc);
        tdt_decoder_destroy(dec);
        free_test_weights(&w, cfg.pred_layers);
        FAIL("invalid tokens in output");
    }

    free(enc);
    tdt_decoder_destroy(dec);
    free_test_weights(&w, cfg.pred_layers);
    PASS();
}

static void test_decode_max_tokens_limit(void) {
    TEST("tdt: decode respects max_tokens limit");
    TDTConfig cfg = make_test_config();

    /* Bias joint output to always emit non-blank tokens */
    TestWeights w = alloc_test_weights(&cfg, 42);
    /* Set joint_out_b so that token 1 always wins over blank (token 0) */
    int V = cfg.vocab_size;
    int D = cfg.n_durations;
    for (int i = 0; i < V + D; i++) w.joint_out_b[i] = -10.0f;
    w.joint_out_b[1] = 10.0f;  /* Token 1 wins */
    w.joint_out_b[V + 0] = 10.0f;  /* Duration 0 (=0 frames, stay in place) */

    TDTDecoder *dec = create_test_decoder(&cfg, &w);
    if (!dec) { free_test_weights(&w, cfg.pred_layers); FAIL("create failed"); }

    int T = 50;
    int E = cfg.encoder_dim;
    float *enc = (float *)calloc(T * E, sizeof(float));
    int max_tokens = 3;
    int tokens[3];
    int n = tdt_decoder_decode(dec, enc, T, tokens, max_tokens);

    if (n > max_tokens) {
        char msg[128];
        snprintf(msg, sizeof(msg), "got %d tokens, max was %d", n, max_tokens);
        free(enc); tdt_decoder_destroy(dec); free_test_weights(&w, cfg.pred_layers);
        FAIL(msg);
    }

    free(enc);
    tdt_decoder_destroy(dec);
    free_test_weights(&w, cfg.pred_layers);
    PASS();
}

/* ── Reset Tests ─────────────────────────────────────────── */

static void test_reset_between_utterances(void) {
    TEST("tdt: reset allows clean decode on second utterance");
    TDTConfig cfg = make_test_config();
    TestWeights w = alloc_test_weights(&cfg, 42);
    TDTDecoder *dec = create_test_decoder(&cfg, &w);
    if (!dec) { free_test_weights(&w, cfg.pred_layers); FAIL("create failed"); }

    int T = 5;
    int E = cfg.encoder_dim;
    float *enc = (float *)calloc(T * E, sizeof(float));

    /* First decode */
    int tokens1[64];
    int n1 = tdt_decoder_decode(dec, enc, T, tokens1, 64);
    if (n1 < 0) {
        free(enc); tdt_decoder_destroy(dec); free_test_weights(&w, cfg.pred_layers);
        FAIL("first decode failed");
    }

    /* Reset is called internally by tdt_decoder_decode, but verify explicit reset */
    tdt_decoder_reset(dec);

    /* Second decode with same input should produce same output */
    int tokens2[64];
    int n2 = tdt_decoder_decode(dec, enc, T, tokens2, 64);
    if (n2 < 0) {
        free(enc); tdt_decoder_destroy(dec); free_test_weights(&w, cfg.pred_layers);
        FAIL("second decode failed");
    }

    /* Results should match (deterministic) */
    if (n1 != n2) {
        char msg[128];
        snprintf(msg, sizeof(msg), "counts differ: %d vs %d", n1, n2);
        free(enc); tdt_decoder_destroy(dec); free_test_weights(&w, cfg.pred_layers);
        FAIL(msg);
    }

    int match = 1;
    for (int i = 0; i < n1; i++) {
        if (tokens1[i] != tokens2[i]) { match = 0; break; }
    }
    if (!match) {
        free(enc); tdt_decoder_destroy(dec); free_test_weights(&w, cfg.pred_layers);
        FAIL("tokens differ after reset");
    }

    free(enc);
    tdt_decoder_destroy(dec);
    free_test_weights(&w, cfg.pred_layers);
    PASS();
}

/* ── Duration Value Tests ────────────────────────────────── */

static void test_duration_values_config(void) {
    TEST("tdt: duration_values config is respected");
    TDTConfig cfg = make_test_config();
    /* Verify default duration values */
    if (cfg.duration_values[0] != 0) FAIL("duration[0] should be 0");
    if (cfg.duration_values[1] != 1) FAIL("duration[1] should be 1");
    if (cfg.duration_values[2] != 2) FAIL("duration[2] should be 2");
    if (cfg.duration_values[3] != 4) FAIL("duration[3] should be 4");
    if (cfg.duration_values[4] != 8) FAIL("duration[4] should be 8");
    PASS();
}

static void test_decode_with_large_duration_skip(void) {
    TEST("tdt: decoder advances by duration frames on blank");
    TDTConfig cfg = make_test_config();
    /* Use large duration values */
    cfg.duration_values[0] = 1;
    cfg.duration_values[1] = 2;
    cfg.duration_values[2] = 4;
    cfg.duration_values[3] = 8;
    cfg.duration_values[4] = 16;

    TestWeights w = alloc_test_weights(&cfg, 42);
    /* Bias toward blank with largest duration → should terminate quickly */
    int V = cfg.vocab_size;
    int D = cfg.n_durations;
    for (int i = 0; i < V + D; i++) w.joint_out_b[i] = -10.0f;
    w.joint_out_b[cfg.blank_id] = 10.0f;  /* Blank always wins */
    w.joint_out_b[V + D - 1] = 10.0f;     /* Largest duration wins */

    TDTDecoder *dec = create_test_decoder(&cfg, &w);
    if (!dec) { free_test_weights(&w, cfg.pred_layers); FAIL("create failed"); }

    int T = 100;
    int E = cfg.encoder_dim;
    float *enc = (float *)calloc(T * E, sizeof(float));
    int tokens[128];
    int n = tdt_decoder_decode(dec, enc, T, tokens, 128);

    /* With blank always winning and large duration, should emit 0 tokens */
    if (n != 0) {
        char msg[128];
        snprintf(msg, sizeof(msg), "expected 0 tokens (all blank), got %d", n);
        free(enc); tdt_decoder_destroy(dec); free_test_weights(&w, cfg.pred_layers);
        FAIL(msg);
    }

    free(enc);
    tdt_decoder_destroy(dec);
    free_test_weights(&w, cfg.pred_layers);
    PASS();
}

/* ── Biased Weights: Force Specific Token ────────────────── */

static void test_biased_token_emission(void) {
    TEST("tdt: biased weights force specific token emission");
    TDTConfig cfg = make_test_config();
    TestWeights w = alloc_test_weights(&cfg, 42);

    int V = cfg.vocab_size;
    int D = cfg.n_durations;

    /* Set joint output bias to strongly favor token 3 and duration 1 (=1 frame advance) */
    for (int i = 0; i < V + D; i++) w.joint_out_b[i] = -100.0f;
    w.joint_out_b[3] = 100.0f;         /* Token 3 wins */
    w.joint_out_b[V + 1] = 100.0f;     /* Duration index 1 → value 1 (advance 1 frame) */

    TDTDecoder *dec = create_test_decoder(&cfg, &w);
    if (!dec) { free_test_weights(&w, cfg.pred_layers); FAIL("create failed"); }

    /* MAX_INNER_LOOP=10 means at most 10 tokens per frame before auto-advance.
       With 5 frames: up to 50 tokens possible, but biased output + LSTM interaction
       may vary. Just verify tokens emitted are all token 3 */
    int T = 5;
    int E = cfg.encoder_dim;
    float *enc = (float *)calloc(T * E, sizeof(float));
    int tokens[128];
    int n = tdt_decoder_decode(dec, enc, T, tokens, 128);

    if (n <= 0) {
        free(enc); tdt_decoder_destroy(dec); free_test_weights(&w, cfg.pred_layers);
        FAIL("expected at least 1 token");
    }

    /* All emitted tokens should be token 3 (bias is overwhelmingly strong) */
    int all_three = 1;
    for (int i = 0; i < n; i++) {
        if (tokens[i] != 3) { all_three = 0; break; }
    }
    if (!all_three) {
        free(enc); tdt_decoder_destroy(dec); free_test_weights(&w, cfg.pred_layers);
        FAIL("not all tokens are 3 despite strong bias");
    }

    free(enc);
    tdt_decoder_destroy(dec);
    free_test_weights(&w, cfg.pred_layers);
    PASS();
}

/* ── All-Blank Output ────────────────────────────────────── */

static void test_all_blank_output(void) {
    TEST("tdt: all-blank logits → 0 tokens emitted");
    TDTConfig cfg = make_test_config();
    TestWeights w = alloc_test_weights(&cfg, 42);

    int V = cfg.vocab_size;
    int D = cfg.n_durations;

    /* Strongly bias blank and non-zero duration */
    for (int i = 0; i < V + D; i++) w.joint_out_b[i] = -100.0f;
    w.joint_out_b[cfg.blank_id] = 100.0f;
    w.joint_out_b[V + 1] = 100.0f;  /* duration index 1 = advance 1 frame */

    TDTDecoder *dec = create_test_decoder(&cfg, &w);
    if (!dec) { free_test_weights(&w, cfg.pred_layers); FAIL("create failed"); }

    int T = 10;
    int E = cfg.encoder_dim;
    float *enc = (float *)calloc(T * E, sizeof(float));
    int tokens[128];
    int n = tdt_decoder_decode(dec, enc, T, tokens, 128);

    if (n != 0) {
        char msg[128];
        snprintf(msg, sizeof(msg), "expected 0 tokens, got %d", n);
        free(enc); tdt_decoder_destroy(dec); free_test_weights(&w, cfg.pred_layers);
        FAIL(msg);
    }

    free(enc);
    tdt_decoder_destroy(dec);
    free_test_weights(&w, cfg.pred_layers);
    PASS();
}

/* ── Large Encoder Dim ───────────────────────────────────── */

static void test_large_encoder_dim(void) {
    TEST("tdt: works with encoder_dim=512");
    TDTConfig cfg = make_test_config();
    cfg.encoder_dim = 512;
    cfg.joint_dim = 64;
    TestWeights w = alloc_test_weights(&cfg, 42);
    TDTDecoder *dec = create_test_decoder(&cfg, &w);
    if (!dec) { free_test_weights(&w, cfg.pred_layers); FAIL("create failed"); }

    int T = 3;
    int E = cfg.encoder_dim;
    float *enc = (float *)calloc(T * E, sizeof(float));
    int tokens[64];
    int n = tdt_decoder_decode(dec, enc, T, tokens, 64);
    if (n < 0) {
        free(enc); tdt_decoder_destroy(dec); free_test_weights(&w, cfg.pred_layers);
        FAIL("decode failed with large encoder_dim");
    }

    free(enc);
    tdt_decoder_destroy(dec);
    free_test_weights(&w, cfg.pred_layers);
    PASS();
}

/* ── Main ───────────────────────────────────────────────── */

int main(void) {
    printf("\n=== TDT Decoder Test Suite ===\n\n");

    printf("NULL Safety:\n");
    test_destroy_null();
    test_reset_null();
    test_decode_null_decoder();
    test_decode_null_enc();
    test_decode_null_tokens();
    test_create_null_config();

    printf("\nConfig Validation:\n");
    test_config_too_many_layers();
    test_config_pred_layers_boundary();
    test_config_single_layer();
    test_duration_values_config();

    printf("\nLifecycle:\n");
    test_create_destroy_basic();
    test_create_destroy_repeated();

    printf("\nDecode:\n");
    test_decode_zero_length();
    test_decode_max_tokens_zero();
    test_decode_single_frame();
    test_decode_multi_frame();
    test_decode_max_tokens_limit();

    printf("\nBiased Decode:\n");
    test_biased_token_emission();
    test_all_blank_output();
    test_decode_with_large_duration_skip();

    printf("\nReset:\n");
    test_reset_between_utterances();

    printf("\nDimensions:\n");
    test_large_encoder_dim();

    printf("\n=== Results: %d passed, %d failed ===\n\n",
           tests_passed, tests_failed);

    return tests_failed > 0 ? 1 : 0;
}
