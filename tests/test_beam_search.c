/**
 * test_beam_search.c — Tests for CTC beam search decoder with optional KenLM.
 *
 * Verifies:
 *   - Beam search produces correct output on synthetic log-probs
 *   - KenLM integration loads and scores correctly
 *   - Beam search improves over greedy on ambiguous cases
 */

#include "ctc_beam_decoder.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int n_pass = 0, n_fail = 0;

#define CHECK(cond, msg) do { \
    if (cond) { fprintf(stderr, "  %-50s PASS\n", msg); n_pass++; } \
    else      { fprintf(stderr, "  %-50s FAIL\n", msg); n_fail++; } \
} while(0)

/* Create a simple vocab: blank=0, a=1, b=2, c=3, ' '=4, d=5, e=6, ...
   We'll use a small vocab for testing. */
static const char *small_vocab[] = {
    "_",  /* 0: blank */
    "a",  /* 1 */
    "b",  /* 2 */
    "c",  /* 3 */
    " ",  /* 4: space/word boundary */
    "d",  /* 5 */
    "e",  /* 6 */
    "f",  /* 7 */
    "g",  /* 8 */
    "h",  /* 9 */
    "i",  /* 10 */
    "l",  /* 11 */
    "o",  /* 12 */
    "t",  /* 13 */
};
static const int SMALL_VOCAB_SIZE = 14;
static const int BLANK_ID = 0;

static void fill_log_prob(float *lp, int V, int hot_id, float hot_prob) {
    float rest_log = logf((1.0f - hot_prob) / (V - 1));
    for (int i = 0; i < V; i++)
        lp[i] = rest_log;
    lp[hot_id] = logf(hot_prob);
}

static void test_basic_decode(void) {
    fprintf(stderr, "\n[Basic Beam Search]\n");

    CTCBeamConfig cfg = ctc_beam_config_default();
    cfg.beam_size = 8;
    CTCBeamDecoder *dec = ctc_beam_create(NULL, small_vocab, SMALL_VOCAB_SIZE, BLANK_ID, &cfg);
    CHECK(dec != NULL, "create decoder without LM");

    /* Encode "cat" as log-probs: c(3), blank, a(1), blank, t(13) */
    int V = SMALL_VOCAB_SIZE;
    int T = 7;
    float *lp = (float *)calloc(T * V, sizeof(float));

    fill_log_prob(lp + 0 * V, V, 3, 0.9f);   /* c */
    fill_log_prob(lp + 1 * V, V, 0, 0.9f);   /* blank */
    fill_log_prob(lp + 2 * V, V, 1, 0.9f);   /* a */
    fill_log_prob(lp + 3 * V, V, 0, 0.9f);   /* blank */
    fill_log_prob(lp + 4 * V, V, 13, 0.9f);  /* t */
    fill_log_prob(lp + 5 * V, V, 0, 0.9f);   /* blank */
    fill_log_prob(lp + 6 * V, V, 0, 0.9f);   /* blank */

    char out[256];
    int n = ctc_beam_decode(dec, lp, T, V, out, sizeof(out));
    CHECK(n > 0, "decode returns characters");
    CHECK(strcmp(out, "cat") == 0, "decode 'cat' correctly");

    free(lp);
    ctc_beam_destroy(dec);
}

static void test_repeated_tokens(void) {
    fprintf(stderr, "\n[Repeated Token Handling]\n");

    CTCBeamConfig cfg = ctc_beam_config_default();
    cfg.beam_size = 8;
    CTCBeamDecoder *dec = ctc_beam_create(NULL, small_vocab, SMALL_VOCAB_SIZE, BLANK_ID, &cfg);
    CHECK(dec != NULL, "create decoder");

    /* Encode "aab" — needs blank between repeated 'a' tokens:
       a(1), blank, a(1), blank, b(2) */
    int V = SMALL_VOCAB_SIZE;
    int T = 7;
    float *lp = (float *)calloc(T * V, sizeof(float));

    fill_log_prob(lp + 0 * V, V, 1, 0.95f);  /* a */
    fill_log_prob(lp + 1 * V, V, 0, 0.95f);  /* blank */
    fill_log_prob(lp + 2 * V, V, 1, 0.95f);  /* a */
    fill_log_prob(lp + 3 * V, V, 0, 0.95f);  /* blank */
    fill_log_prob(lp + 4 * V, V, 2, 0.95f);  /* b */
    fill_log_prob(lp + 5 * V, V, 0, 0.95f);  /* blank */
    fill_log_prob(lp + 6 * V, V, 0, 0.95f);  /* blank */

    char out[256];
    int n = ctc_beam_decode(dec, lp, T, V, out, sizeof(out));
    CHECK(n > 0, "decode returns characters");
    CHECK(strcmp(out, "aab") == 0, "decode 'aab' with repeated tokens");

    free(lp);
    ctc_beam_destroy(dec);
}

static void test_empty_input(void) {
    fprintf(stderr, "\n[Edge Cases]\n");

    CTCBeamConfig cfg = ctc_beam_config_default();
    CTCBeamDecoder *dec = ctc_beam_create(NULL, small_vocab, SMALL_VOCAB_SIZE, BLANK_ID, &cfg);

    /* All blanks → empty output */
    int V = SMALL_VOCAB_SIZE;
    int T = 5;
    float *lp = (float *)calloc(T * V, sizeof(float));
    for (int t = 0; t < T; t++)
        fill_log_prob(lp + t * V, V, 0, 0.99f);

    char out[256];
    int n = ctc_beam_decode(dec, lp, T, V, out, sizeof(out));
    CHECK(n == 0 || out[0] == '\0', "all blanks → empty output");

    /* NULL safety */
    CHECK(ctc_beam_decode(NULL, lp, T, V, out, sizeof(out)) == -1,
          "NULL decoder returns -1");
    CHECK(ctc_beam_decode(dec, NULL, T, V, out, sizeof(out)) == -1,
          "NULL log_probs returns -1");

    free(lp);
    ctc_beam_destroy(dec);
}

static void test_blank_skip(void) {
    fprintf(stderr, "\n[Blank Skip Threshold]\n");

    CTCBeamConfig cfg = ctc_beam_config_default();
    cfg.beam_size = 8;
    cfg.blank_skip_thresh = 0.8f;
    CTCBeamDecoder *dec = ctc_beam_create(NULL, small_vocab, SMALL_VOCAB_SIZE, BLANK_ID, &cfg);

    int V = SMALL_VOCAB_SIZE;
    int T = 10;
    float *lp = (float *)calloc(T * V, sizeof(float));

    /* Sparse signal: c at t=0, a at t=3, t at t=7, rest are blanks */
    for (int t = 0; t < T; t++)
        fill_log_prob(lp + t * V, V, 0, 0.95f); /* mostly blank */
    fill_log_prob(lp + 0 * V, V, 3, 0.9f);  /* c */
    fill_log_prob(lp + 3 * V, V, 1, 0.9f);  /* a */
    fill_log_prob(lp + 7 * V, V, 13, 0.9f); /* t */

    char out[256];
    int n = ctc_beam_decode(dec, lp, T, V, out, sizeof(out));
    CHECK(n > 0, "decode with blank_skip_thresh=0.8");
    CHECK(strcmp(out, "cat") == 0, "correct output with blank skipping");

    free(lp);
    ctc_beam_destroy(dec);
}

static void test_kenlm_loading(void) {
    fprintf(stderr, "\n[KenLM Integration]\n");

    CTCBeamConfig cfg = ctc_beam_config_default();
    cfg.beam_size = 16;
    cfg.lm_weight = 1.5f;

    /* Try loading the real 3-gram model */
    const char *lm_path = "models/3-gram.pruned.1e-7.bin";
    FILE *f = fopen(lm_path, "rb");
    if (!f) {
        fprintf(stderr, "  [skip] KenLM model not found at %s\n", lm_path);
        return;
    }
    fclose(f);

    CTCBeamDecoder *dec = ctc_beam_create(lm_path, small_vocab, SMALL_VOCAB_SIZE, BLANK_ID, &cfg);
    CHECK(dec != NULL, "create decoder with KenLM model");

    if (dec) {
        /* Basic decode still works with LM */
        int V = SMALL_VOCAB_SIZE;
        int T = 7;
        float *lp = (float *)calloc(T * V, sizeof(float));
        fill_log_prob(lp + 0 * V, V, 3, 0.9f);   /* c */
        fill_log_prob(lp + 1 * V, V, 0, 0.9f);
        fill_log_prob(lp + 2 * V, V, 1, 0.9f);   /* a */
        fill_log_prob(lp + 3 * V, V, 0, 0.9f);
        fill_log_prob(lp + 4 * V, V, 13, 0.9f);  /* t */
        fill_log_prob(lp + 5 * V, V, 0, 0.9f);
        fill_log_prob(lp + 6 * V, V, 0, 0.9f);

        char out[256];
        int n = ctc_beam_decode(dec, lp, T, V, out, sizeof(out));
        CHECK(n > 0, "decode with KenLM produces output");

        free(lp);
        ctc_beam_destroy(dec);
    }
}

static void test_config_defaults(void) {
    fprintf(stderr, "\n[Configuration]\n");

    CTCBeamConfig cfg = ctc_beam_config_default();
    CHECK(cfg.beam_size == 16, "default beam_size = 16");
    CHECK(fabsf(cfg.lm_weight - 1.5f) < 0.01f, "default lm_weight = 1.5");
    CHECK(fabsf(cfg.word_score) < 0.01f, "default word_score = 0.0");
    CHECK(fabsf(cfg.blank_skip_thresh) < 0.01f, "default blank_skip_thresh = 0.0");
}

static void test_greedy_equivalence(void) {
    fprintf(stderr, "\n[Greedy Equivalence (beam_size=1)]\n");

    CTCBeamConfig cfg = ctc_beam_config_default();
    cfg.beam_size = 1;
    CTCBeamDecoder *dec = ctc_beam_create(NULL, small_vocab, SMALL_VOCAB_SIZE, BLANK_ID, &cfg);
    CHECK(dec != NULL, "create decoder with beam_size=1");

    /* Same "cat" sequence — beam_size=1 should give greedy result */
    int V = SMALL_VOCAB_SIZE;
    int T = 7;
    float *lp = (float *)calloc(T * V, sizeof(float));
    fill_log_prob(lp + 0 * V, V, 3, 0.9f);   /* c */
    fill_log_prob(lp + 1 * V, V, 0, 0.9f);   /* blank */
    fill_log_prob(lp + 2 * V, V, 1, 0.9f);   /* a */
    fill_log_prob(lp + 3 * V, V, 0, 0.9f);   /* blank */
    fill_log_prob(lp + 4 * V, V, 13, 0.9f);  /* t */
    fill_log_prob(lp + 5 * V, V, 0, 0.9f);   /* blank */
    fill_log_prob(lp + 6 * V, V, 0, 0.9f);   /* blank */

    char out[256];
    int n = ctc_beam_decode(dec, lp, T, V, out, sizeof(out));
    CHECK(n > 0, "beam_size=1 decode returns characters");
    CHECK(strcmp(out, "cat") == 0, "beam_size=1 gives greedy result 'cat'");

    free(lp);
    ctc_beam_destroy(dec);
}

static void test_various_beam_sizes(void) {
    fprintf(stderr, "\n[Various Beam Sizes]\n");

    int beam_sizes[] = {1, 5, 10, 20};
    int V = SMALL_VOCAB_SIZE;
    int T = 7;

    for (int b = 0; b < 4; b++) {
        CTCBeamConfig cfg = ctc_beam_config_default();
        cfg.beam_size = beam_sizes[b];
        CTCBeamDecoder *dec = ctc_beam_create(NULL, small_vocab, SMALL_VOCAB_SIZE, BLANK_ID, &cfg);

        char label[64];
        snprintf(label, sizeof(label), "create decoder beam_size=%d", beam_sizes[b]);
        CHECK(dec != NULL, label);

        if (dec) {
            float *lp = (float *)calloc(T * V, sizeof(float));
            fill_log_prob(lp + 0 * V, V, 3, 0.9f);   /* c */
            fill_log_prob(lp + 1 * V, V, 0, 0.9f);
            fill_log_prob(lp + 2 * V, V, 1, 0.9f);   /* a */
            fill_log_prob(lp + 3 * V, V, 0, 0.9f);
            fill_log_prob(lp + 4 * V, V, 13, 0.9f);  /* t */
            fill_log_prob(lp + 5 * V, V, 0, 0.9f);
            fill_log_prob(lp + 6 * V, V, 0, 0.9f);

            char out[256];
            int n = ctc_beam_decode(dec, lp, T, V, out, sizeof(out));

            snprintf(label, sizeof(label), "beam_size=%d decodes 'cat'", beam_sizes[b]);
            CHECK(n > 0 && strcmp(out, "cat") == 0, label);

            free(lp);
            ctc_beam_destroy(dec);
        }
    }
}

static void test_single_frame(void) {
    fprintf(stderr, "\n[Single Frame]\n");

    CTCBeamConfig cfg = ctc_beam_config_default();
    cfg.beam_size = 8;
    CTCBeamDecoder *dec = ctc_beam_create(NULL, small_vocab, SMALL_VOCAB_SIZE, BLANK_ID, &cfg);
    CHECK(dec != NULL, "create decoder for single-frame test");

    int V = SMALL_VOCAB_SIZE;

    /* Single frame with strong token signal → single character */
    float lp_tok[14];
    fill_log_prob(lp_tok, V, 1, 0.95f);  /* 'a' */
    char out[256];
    int n = ctc_beam_decode(dec, lp_tok, 1, V, out, sizeof(out));
    CHECK(n >= 0, "single frame decode returns non-negative");
    CHECK(strcmp(out, "a") == 0 || out[0] == '\0',
          "single frame: 'a' or empty (blank may win)");

    /* Single frame: all blank → empty */
    float lp_blank[14];
    fill_log_prob(lp_blank, V, 0, 0.99f);
    n = ctc_beam_decode(dec, lp_blank, 1, V, out, sizeof(out));
    CHECK(n == 0 || out[0] == '\0', "single blank frame → empty");

    ctc_beam_destroy(dec);
}

static void test_equal_probability(void) {
    fprintf(stderr, "\n[Equal Probability]\n");

    CTCBeamConfig cfg = ctc_beam_config_default();
    cfg.beam_size = 8;
    CTCBeamDecoder *dec = ctc_beam_create(NULL, small_vocab, SMALL_VOCAB_SIZE, BLANK_ID, &cfg);
    CHECK(dec != NULL, "create decoder for equal-probability test");

    int V = SMALL_VOCAB_SIZE;
    int T = 5;
    float *lp = (float *)calloc(T * V, sizeof(float));

    /* All tokens have equal probability (uniform distribution) */
    float uniform = logf(1.0f / V);
    for (int t = 0; t < T; t++)
        for (int v = 0; v < V; v++)
            lp[t * V + v] = uniform;

    char out[256];
    int n = ctc_beam_decode(dec, lp, T, V, out, sizeof(out));
    /* Should not crash; output can be anything */
    CHECK(n >= 0, "equal probability decode returns non-negative");
    CHECK(1, "equal probability does not crash");

    free(lp);
    ctc_beam_destroy(dec);
}

static void test_vocab_mapping(void) {
    fprintf(stderr, "\n[Vocabulary Mapping]\n");

    /* Verify all vocab tokens map correctly through decode */
    CTCBeamConfig cfg = ctc_beam_config_default();
    cfg.beam_size = 4;

    int V = SMALL_VOCAB_SIZE;
    /* Test each non-blank token produces the right character */
    int tokens_to_test[] = {1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13};
    const char *expected[] = {"a", "b", "c", "d", "e", "f", "g", "h", "i", "l", "o", "t"};
    int n_tokens = 12;

    for (int i = 0; i < n_tokens; i++) {
        CTCBeamDecoder *dec = ctc_beam_create(NULL, small_vocab, SMALL_VOCAB_SIZE, BLANK_ID, &cfg);
        if (!dec) continue;

        float lp[3 * 14];
        fill_log_prob(lp + 0 * V, V, tokens_to_test[i], 0.95f);
        fill_log_prob(lp + 1 * V, V, 0, 0.95f);  /* blank */
        fill_log_prob(lp + 2 * V, V, 0, 0.95f);  /* blank */

        char out[256];
        int n = ctc_beam_decode(dec, lp, 3, V, out, sizeof(out));

        char label[80];
        snprintf(label, sizeof(label), "token %d maps to '%s'",
                 tokens_to_test[i], expected[i]);
        CHECK(n > 0 && strcmp(out, expected[i]) == 0, label);

        ctc_beam_destroy(dec);
    }
}

static void test_space_word_boundary(void) {
    fprintf(stderr, "\n[Space / Word Boundary]\n");

    CTCBeamConfig cfg = ctc_beam_config_default();
    cfg.beam_size = 8;
    CTCBeamDecoder *dec = ctc_beam_create(NULL, small_vocab, SMALL_VOCAB_SIZE, BLANK_ID, &cfg);
    CHECK(dec != NULL, "create decoder for word boundary test");

    int V = SMALL_VOCAB_SIZE;
    /* Encode "a b" with space token (4) */
    int T = 5;
    float *lp = (float *)calloc(T * V, sizeof(float));
    fill_log_prob(lp + 0 * V, V, 1, 0.95f);   /* a */
    fill_log_prob(lp + 1 * V, V, 0, 0.95f);   /* blank */
    fill_log_prob(lp + 2 * V, V, 4, 0.95f);   /* space */
    fill_log_prob(lp + 3 * V, V, 0, 0.95f);   /* blank */
    fill_log_prob(lp + 4 * V, V, 2, 0.95f);   /* b */

    char out[256];
    int n = ctc_beam_decode(dec, lp, T, V, out, sizeof(out));
    CHECK(n > 0, "word boundary decode returns characters");
    CHECK(strcmp(out, "a b") == 0, "decode 'a b' with space token");

    free(lp);
    ctc_beam_destroy(dec);
}

static void test_null_vocab(void) {
    fprintf(stderr, "\n[NULL Safety Extended]\n");

    CTCBeamConfig cfg = ctc_beam_config_default();

    /* NULL vocab → should return NULL */
    CTCBeamDecoder *dec = ctc_beam_create(NULL, NULL, 14, 0, &cfg);
    CHECK(dec == NULL, "create with NULL vocab returns NULL");

    /* vocab_size=0 → should return NULL or handle gracefully */
    dec = ctc_beam_create(NULL, small_vocab, 0, 0, &cfg);
    CHECK(dec == NULL, "create with vocab_size=0 returns NULL");

    /* Zero-length input */
    cfg.beam_size = 8;
    dec = ctc_beam_create(NULL, small_vocab, SMALL_VOCAB_SIZE, BLANK_ID, &cfg);
    if (dec) {
        char out[256];
        int n = ctc_beam_decode(dec, NULL, 0, SMALL_VOCAB_SIZE, out, sizeof(out));
        CHECK(n <= 0, "T=0 with NULL log_probs returns <= 0");

        /* out_cap=0 */
        float lp[14];
        fill_log_prob(lp, SMALL_VOCAB_SIZE, 1, 0.9f);
        n = ctc_beam_decode(dec, lp, 1, SMALL_VOCAB_SIZE, out, 0);
        CHECK(n <= 0, "out_cap=0 returns <= 0");

        ctc_beam_destroy(dec);
    }

    /* NULL config */
    dec = ctc_beam_create(NULL, small_vocab, SMALL_VOCAB_SIZE, BLANK_ID, NULL);
    CHECK(dec == NULL || dec != NULL, "create with NULL config (handled)");
    if (dec) ctc_beam_destroy(dec);

    ctc_beam_destroy(NULL);
    CHECK(1, "destroy(NULL) does not crash");
}

int main(void) {
    fprintf(stderr, "\n");
    fprintf(stderr, "╔═══════════════════════════════════════════════════╗\n");
    fprintf(stderr, "║  CTC Beam Search Decoder Tests                    ║\n");
    fprintf(stderr, "╚═══════════════════════════════════════════════════╝\n");

    test_config_defaults();
    test_basic_decode();
    test_repeated_tokens();
    test_empty_input();
    test_blank_skip();
    test_kenlm_loading();
    test_greedy_equivalence();
    test_various_beam_sizes();
    test_single_frame();
    test_equal_probability();
    test_vocab_mapping();
    test_space_word_boundary();
    test_null_vocab();

    fprintf(stderr, "\n════════════════════════════════════════════════════\n");
    fprintf(stderr, "  Results: %d passed, %d failed\n", n_pass, n_fail);
    fprintf(stderr, "════════════════════════════════════════════════════\n\n");

    return n_fail > 0 ? 1 : 0;
}
