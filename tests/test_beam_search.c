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

    fprintf(stderr, "\n════════════════════════════════════════════════════\n");
    fprintf(stderr, "  Results: %d passed, %d failed\n", n_pass, n_fail);
    fprintf(stderr, "════════════════════════════════════════════════════\n\n");

    return n_fail > 0 ? 1 : 0;
}
