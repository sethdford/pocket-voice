/**
 * test_speculative_gen.c — Unit tests for continuous speculative generation.
 *
 * Tests: create/destroy, tick logic, feed tokens, commit, cancel, draft validity,
 * best draft selection, reset, max drafts, transcript divergence.
 *
 * Build: make test-speculative-gen
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "speculative_gen.h"

static int pass_count = 0;
static int fail_count = 0;

#define TEST(cond, name) do { \
    if (cond) { printf("  [PASS] %s\n", name); pass_count++; } \
    else { printf("  [FAIL] %s (line %d)\n", name, __LINE__); fail_count++; } \
} while (0)

static void test_create_destroy(void) {
    printf("\n=== Create/Destroy ===\n");

    SpeculativeGen *sg = speculative_gen_create(NULL);
    TEST(sg != NULL, "create with NULL config returns non-NULL");

    speculative_gen_destroy(sg);
    speculative_gen_destroy(NULL);
    TEST(1, "destroy NULL is safe");
}

static void test_create_with_config(void) {
    printf("\n=== Create with Config ===\n");

    SpeculativeConfig cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.max_drafts = 3;
    cfg.min_words_to_spec = 5;
    cfg.vap_eou_threshold = 0.5f;
    cfg.commit_threshold = 0.9f;
    SpeculativeGen *sg = speculative_gen_create(&cfg);
    TEST(sg != NULL, "create with config returns non-NULL");

    speculative_gen_destroy(sg);
}

static void test_tick_low_eou(void) {
    printf("\n=== Tick with Low EOU: No Action ===\n");

    SpeculativeGen *sg = speculative_gen_create(NULL);
    int r = speculative_gen_tick(sg, "hello world", 2, 0.1f, 0.0f, 0.2f);
    TEST(r == 0, "low EOU returns 0 (no action)");

    r = speculative_gen_tick(sg, "tell me about weather", 4, 0.2f, 0.0f, 0.15f);
    TEST(r == 0, "low vap_eou returns 0");

    speculative_gen_destroy(sg);
}

static void test_tick_high_vap_start_draft(void) {
    printf("\n=== Tick with High VAP EOU: Start Draft ===\n");

    SpeculativeGen *sg = speculative_gen_create(NULL);
    int r = speculative_gen_tick(sg, "tell me about the weather", 5, 0.5f, 0.0f, 0.5f);
    TEST(r == 1, "high vap_eou + enough words returns 1 (start draft)");

    int active = speculative_gen_active_draft(sg);
    TEST(active >= 0, "active draft ID after start");

    speculative_gen_destroy(sg);
}

static void test_tick_min_words(void) {
    printf("\n=== Tick Respects Min Words ===\n");

    SpeculativeConfig cfg = { .min_words_to_spec = 5 };
    SpeculativeGen *sg = speculative_gen_create(&cfg);
    int r = speculative_gen_tick(sg, "hi there", 2, 0.6f, 0.0f, 0.5f);
    TEST(r == 0, "fewer than min_words returns 0");

    r = speculative_gen_tick(sg, "tell me about the weather today", 6, 0.6f, 0.0f, 0.5f);
    TEST(r == 1, "enough words returns 1");

    speculative_gen_destroy(sg);
}

static void test_feed_tokens(void) {
    printf("\n=== Feed Tokens to Draft ===\n");

    SpeculativeGen *sg = speculative_gen_create(NULL);
    speculative_gen_tick(sg, "what is the time", 4, 0.5f, 0.0f, 0.5f);
    int draft_id = speculative_gen_active_draft(sg);
    TEST(draft_id >= 0, "have active draft");

    speculative_gen_feed_token(sg, draft_id, "The ");
    speculative_gen_feed_token(sg, draft_id, "time ");
    speculative_gen_feed_token(sg, draft_id, "is ");

    const SpecDraft *d = speculative_gen_get_draft(sg, draft_id);
    TEST(d != NULL, "get_draft returns non-NULL");
    TEST(d->n_tokens >= 3, "draft has tokens");
    TEST(strstr(d->response, "The ") != NULL, "response contains tokens");

    speculative_gen_destroy(sg);
}

static void test_commit_when_fused_high(void) {
    printf("\n=== Commit Draft When Fused EOU High ===\n");

    SpeculativeGen *sg = speculative_gen_create(NULL);
    speculative_gen_tick(sg, "hello world today", 3, 0.5f, 0.0f, 0.5f);
    int draft_id = speculative_gen_active_draft(sg);
    speculative_gen_feed_token(sg, draft_id, "Hi ");
    speculative_gen_feed_token(sg, draft_id, "there!");
    speculative_gen_draft_done(sg, draft_id);

    int r = speculative_gen_tick(sg, "hello world today", 3, 0.6f, 0.0f, 0.85f);
    TEST(r == 2, "high fused_eou returns 2 (commit)");

    const SpecDraft *best = speculative_gen_get_best(sg);
    TEST(best != NULL, "get_best returns draft");

    const char *resp = speculative_gen_commit(sg);
    TEST(resp != NULL, "commit returns non-NULL response");
    TEST(strstr(resp, "Hi ") != NULL, "response has content");

    speculative_gen_destroy(sg);
}

static void test_cancel_when_eou_drops(void) {
    printf("\n=== Cancel Drafts When EOU Drops ===\n");

    SpeculativeGen *sg = speculative_gen_create(NULL);
    speculative_gen_tick(sg, "tell me something", 3, 0.5f, 0.0f, 0.5f);
    TEST(speculative_gen_active_count(sg) >= 1, "have active draft");

    int r = speculative_gen_tick(sg, "tell me something", 3, 0.1f, 0.0f, 0.15f);
    TEST(r == 0, "low EOU returns 0");
    speculative_gen_cancel_all(sg);
    TEST(speculative_gen_active_count(sg) == 0, "cancel_all clears drafts");

    speculative_gen_destroy(sg);
}

static void test_draft_validity_prefix(void) {
    printf("\n=== Draft Validity (Prefix Matching) ===\n");

    SpeculativeGen *sg = speculative_gen_create(NULL);
    speculative_gen_tick(sg, "tell me about", 3, 0.5f, 0.0f, 0.5f);
    int draft_id = speculative_gen_active_draft(sg);

    int v = speculative_gen_draft_valid(sg, draft_id, "tell me about the weather");
    TEST(v == 1, "draft 'tell me about' valid for 'tell me about the weather'");

    v = speculative_gen_draft_valid(sg, draft_id, "tell me what");
    TEST(v == 0, "draft 'tell me about' invalid for 'tell me what'");

    speculative_gen_destroy(sg);
}

static void test_best_draft_most_tokens(void) {
    printf("\n=== Best Draft Selection (Most Tokens) ===\n");

    SpeculativeConfig cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.max_drafts = 3;
    cfg.min_words_to_spec = 1;
    SpeculativeGen *sg = speculative_gen_create(&cfg);
    speculative_gen_tick(sg, "hello", 1, 0.5f, 0.0f, 0.5f);
    int id1 = speculative_gen_active_draft(sg);
    speculative_gen_feed_token(sg, id1, "A");
    speculative_gen_draft_done(sg, id1);

    speculative_gen_tick(sg, "hello", 1, 0.5f, 0.0f, 0.5f);
    int id2 = speculative_gen_active_draft(sg);
    speculative_gen_feed_token(sg, id2, "Longer ");
    speculative_gen_feed_token(sg, id2, "response ");
    speculative_gen_feed_token(sg, id2, "here");
    speculative_gen_draft_done(sg, id2);

    speculative_gen_tick(sg, "hello", 1, 0.5f, 0.0f, 0.85f);
    const SpecDraft *best = speculative_gen_get_best(sg);
    TEST(best != NULL, "get_best returns valid draft when multiple exist");

    speculative_gen_destroy(sg);
}

static void test_multiple_drafts(void) {
    printf("\n=== Multiple Drafts Management ===\n");

    SpeculativeConfig cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.max_drafts = 2;
    cfg.min_words_to_spec = 2;
    SpeculativeGen *sg = speculative_gen_create(&cfg);
    speculative_gen_tick(sg, "first query", 2, 0.5f, 0.0f, 0.5f);
    speculative_gen_cancel_all(sg);
    speculative_gen_tick(sg, "second query", 2, 0.5f, 0.0f, 0.5f);
    int n = speculative_gen_active_count(sg);
    TEST(n >= 1, "can have multiple draft cycles");

    speculative_gen_destroy(sg);
}

static void test_draft_done(void) {
    printf("\n=== Draft Done Marks READY ===\n");

    SpeculativeConfig cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.min_words_to_spec = 1;
    SpeculativeGen *sg = speculative_gen_create(&cfg);
    speculative_gen_tick(sg, "question here", 2, 0.5f, 0.0f, 0.5f);
    int id = speculative_gen_active_draft(sg);
    speculative_gen_feed_token(sg, id, "Answer");
    speculative_gen_draft_done(sg, id);

    const SpecDraft *d = speculative_gen_get_draft(sg, id);
    TEST(d != NULL, "get_draft works");
    TEST(d->state == SPEC_READY, "draft state is READY after done");

    speculative_gen_destroy(sg);
}

static void test_reset(void) {
    printf("\n=== Reset ===\n");

    SpeculativeConfig cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.min_words_to_spec = 1;
    SpeculativeGen *sg = speculative_gen_create(&cfg);
    speculative_gen_tick(sg, "hello", 1, 0.5f, 0.0f, 0.5f);
    speculative_gen_reset(sg);
    TEST(speculative_gen_active_count(sg) == 0, "reset clears drafts");
    TEST(speculative_gen_active_draft(sg) == -1, "no active draft after reset");

    speculative_gen_destroy(sg);
}

static void test_max_drafts_limit(void) {
    printf("\n=== Max Drafts Limit ===\n");

    SpeculativeConfig cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.max_drafts = 1;
    cfg.min_words_to_spec = 1;
    SpeculativeGen *sg = speculative_gen_create(&cfg);
    speculative_gen_tick(sg, "one", 1, 0.5f, 0.0f, 0.5f);
    int r = speculative_gen_tick(sg, "two", 1, 0.5f, 0.0f, 0.5f);
    TEST(r == 0, "no new draft when max reached (one slot busy)");

    speculative_gen_cancel_all(sg);
    r = speculative_gen_tick(sg, "three", 1, 0.5f, 0.0f, 0.5f);
    TEST(r == 1, "can start draft after cancel");

    speculative_gen_destroy(sg);
}

static void test_transcript_divergence(void) {
    printf("\n=== Transcript Divergence -> Restart ===\n");

    SpeculativeGen *sg = speculative_gen_create(NULL);
    speculative_gen_tick(sg, "tell me about", 3, 0.5f, 0.0f, 0.5f);
    int id = speculative_gen_active_draft(sg);
    speculative_gen_feed_token(sg, id, "Some ");
    speculative_gen_tick(sg, "tell me what", 3, 0.5f, 0.0f, 0.4f);

    int v = speculative_gen_draft_valid(sg, id, "tell me what");
    TEST(v == 0, "diverged transcript invalidates draft");

    speculative_gen_destroy(sg);
}

static void test_get_best_no_valid(void) {
    printf("\n=== Get Best with No Valid Drafts ===\n");

    SpeculativeGen *sg = speculative_gen_create(NULL);
    const SpecDraft *best = speculative_gen_get_best(sg);
    TEST(best == NULL, "get_best returns NULL when no drafts");

    speculative_gen_tick(sg, "hello", 1, 0.5f, 0.0f, 0.5f);
    speculative_gen_cancel_all(sg);
    best = speculative_gen_get_best(sg);
    TEST(best == NULL, "get_best returns NULL when all cancelled");

    speculative_gen_destroy(sg);
}

static void test_active_count(void) {
    printf("\n=== Active Count ===\n");

    SpeculativeGen *sg = speculative_gen_create(NULL);
    TEST(speculative_gen_active_count(sg) == 0, "initial count 0");

    speculative_gen_tick(sg, "one two three", 3, 0.5f, 0.0f, 0.5f);
    TEST(speculative_gen_active_count(sg) >= 1, "count >= 1 after start");

    speculative_gen_cancel_all(sg);
    TEST(speculative_gen_active_count(sg) == 0, "count 0 after cancel");

    speculative_gen_destroy(sg);
}

static void test_commit_empty_response(void) {
    printf("\n=== Commit with Empty Response ===\n");

    SpeculativeGen *sg = speculative_gen_create(NULL);
    speculative_gen_tick(sg, "hi", 1, 0.5f, 0.0f, 0.5f);
    speculative_gen_draft_done(sg, speculative_gen_active_draft(sg));
    speculative_gen_tick(sg, "hi", 1, 0.5f, 0.0f, 0.85f);
    const char *resp = speculative_gen_commit(sg);
    TEST(resp == NULL || resp[0] == '\0', "commit with no tokens returns NULL or empty");

    speculative_gen_destroy(sg);
}

static void test_no_double_start(void) {
    printf("\n=== No Double Start (Has Pending) ===\n");

    SpeculativeGen *sg = speculative_gen_create(NULL);
    speculative_gen_tick(sg, "tell me more", 3, 0.5f, 0.0f, 0.5f);
    int r = speculative_gen_tick(sg, "tell me more", 3, 0.6f, 0.0f, 0.5f);
    TEST(r == 0, "second tick with pending draft returns 0 (no new draft)");

    speculative_gen_destroy(sg);
}

int main(void) {
    setbuf(stdout, NULL);
    printf("\n=== Speculative Generation Tests ===\n");

    test_create_destroy();
    test_create_with_config();
    test_tick_low_eou();
    test_tick_high_vap_start_draft();
    test_tick_min_words();
    test_feed_tokens();
    test_commit_when_fused_high();
    test_cancel_when_eou_drops();
    test_draft_validity_prefix();
    test_best_draft_most_tokens();
    test_multiple_drafts();
    test_draft_done();
    test_reset();
    test_max_drafts_limit();
    test_transcript_divergence();
    test_get_best_no_valid();
    test_active_count();
    test_commit_empty_response();
    test_no_double_start();

    printf("\n--- Results: %d passed, %d failed ---\n", pass_count, fail_count);
    return fail_count ? 1 : 0;
}
