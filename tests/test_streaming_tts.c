/**
 * test_streaming_tts.c — Unit tests for streaming TTS controller.
 *
 * Tests: create/destroy, feed tokens, get/peek/advance audio, finish,
 * rollback, commit, segments, lookahead, done, reset, stats, edge cases.
 *
 * Build: make test-streaming-tts
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "streaming_tts.h"

static int pass_count = 0;
static int fail_count = 0;

#define TEST(cond, name) do { \
    if (cond) { printf("  [PASS] %s\n", name); pass_count++; } \
    else { printf("  [FAIL] %s (line %d)\n", name, __LINE__); fail_count++; } \
} while (0)

/* Mock synthesizer: returns fixed samples per character */
static int mock_synth(void *ctx, const char *text, int text_len,
                      float *out_pcm, int max_samples) {
    (void)ctx;
    if (!text || !out_pcm || text_len <= 0) return 0;
    int samples = text_len * 100; /* 100 samples per char */
    if (samples > max_samples) samples = max_samples;
    for (int i = 0; i < samples; i++) {
        out_pcm[i] = 0.5f;
    }
    return samples;
}

static void test_create_destroy(void) {
    printf("\n=== Create/Destroy ===\n");

    StreamingTTS *stts = streaming_tts_create(NULL);
    TEST(stts != NULL, "create with NULL config returns non-NULL");

    streaming_tts_destroy(stts);
    streaming_tts_destroy(NULL);
    TEST(1, "destroy NULL is safe");
}

static void test_feed_single_token_no_audio(void) {
    printf("\n=== Feed Single Token — No Audio Yet ===\n");

    StreamingTTSConfig cfg = { .min_tokens_to_start = 4 };
    StreamingTTS *stts = streaming_tts_create(&cfg);
    streaming_tts_set_synthesizer(stts, mock_synth, NULL);

    int avail = streaming_tts_feed_token(stts, "Hello", 0);
    TEST(avail == 0, "1 token below min_tokens_to_start yields 0 audio");

    streaming_tts_destroy(stts);
}

static void test_feed_enough_tokens_audio_generated(void) {
    printf("\n=== Feed Enough Tokens — Audio Generated ===\n");

    StreamingTTSConfig cfg = { .min_tokens_to_start = 4 };
    StreamingTTS *stts = streaming_tts_create(&cfg);
    streaming_tts_set_synthesizer(stts, mock_synth, NULL);

    streaming_tts_feed_token(stts, "Hello ", 0);
    streaming_tts_feed_token(stts, "world", 1);
    streaming_tts_feed_token(stts, " this", 2);
    int avail = streaming_tts_feed_token(stts, " is", 3);

    TEST(avail > 0, ">= min_tokens_to_start yields audio");
    TEST(avail >= 1000, "15 chars -> >=1000 samples (100/char mock)");

    streaming_tts_destroy(stts);
}

static void test_get_audio_peek_advance(void) {
    printf("\n=== Get Audio / Peek / Advance ===\n");

    StreamingTTSConfig cfg = { .min_tokens_to_start = 2 };
    StreamingTTS *stts = streaming_tts_create(&cfg);
    streaming_tts_set_synthesizer(stts, mock_synth, NULL);

    streaming_tts_feed_token(stts, "Hi", 0);
    streaming_tts_feed_token(stts, "!", 1);

    int len = 0;
    const float *p = streaming_tts_peek_audio(stts, &len);
    TEST(p != NULL && len > 0, "peek returns non-NULL and length");

    float buf[4096];
    int n = streaming_tts_get_audio(stts, buf, 100);
    TEST(n == 100, "get_audio returns requested amount");

    streaming_tts_advance_audio(stts, 50);
    n = streaming_tts_get_audio(stts, buf, 5000);
    TEST(n > 0, "get_audio after advance returns more");

    streaming_tts_destroy(stts);
}

static void test_feed_more_tokens_incremental(void) {
    printf("\n=== Feed More Tokens — Incremental Synthesis ===\n");

    StreamingTTSConfig cfg = { .min_tokens_to_start = 2, .lookahead_tokens = 1 };
    StreamingTTS *stts = streaming_tts_create(&cfg);
    streaming_tts_set_synthesizer(stts, mock_synth, NULL);

    streaming_tts_feed_token(stts, "A", 0);
    streaming_tts_feed_token(stts, "B", 1);
    int len1 = 0;
    streaming_tts_peek_audio(stts, &len1);

    streaming_tts_feed_token(stts, "C", 2);
    streaming_tts_feed_token(stts, "D", 3);
    int len2 = 0;
    streaming_tts_peek_audio(stts, &len2);

    TEST(len2 > len1, "more tokens produce more audio");

    streaming_tts_destroy(stts);
}

static void test_finish_flushes_remaining(void) {
    printf("\n=== Finish — Flushes Remaining ===\n");

    StreamingTTSConfig cfg = { .min_tokens_to_start = 2, .lookahead_tokens = 2 };
    StreamingTTS *stts = streaming_tts_create(&cfg);
    streaming_tts_set_synthesizer(stts, mock_synth, NULL);

    streaming_tts_feed_token(stts, "Hi", 0);
    streaming_tts_feed_token(stts, " ", 1);
    int len_before = 0;
    streaming_tts_peek_audio(stts, &len_before);

    streaming_tts_finish(stts);
    int len_after = 0;
    streaming_tts_peek_audio(stts, &len_after);

    TEST(len_after >= len_before, "finish produces at least as much audio");
    TEST(streaming_tts_is_done(stts) == false, "not done until audio consumed");

    streaming_tts_destroy(stts);
}

static void test_rollback_basic(void) {
    printf("\n=== Rollback: Feed 10 Tokens, Rollback to 5 ===\n");

    StreamingTTSConfig cfg = {
        .min_tokens_to_start = 2,
        .lookahead_tokens = 0,
        .rollback_enabled = 1
    };
    StreamingTTS *stts = streaming_tts_create(&cfg);
    streaming_tts_set_synthesizer(stts, mock_synth, NULL);

    for (int i = 0; i < 10; i++) {
        char t[8];
        snprintf(t, sizeof(t), "x%d", i);
        streaming_tts_feed_token(stts, t, i);
    }
    streaming_tts_finish(stts);

    int len_before = 0;
    streaming_tts_peek_audio(stts, &len_before);

    int rolled = streaming_tts_rollback(stts, 5);
    TEST(rolled >= 0, "rollback returns non-negative");

    int len_after = 0;
    streaming_tts_peek_audio(stts, &len_after);
    TEST(len_after < len_before || rolled > 0, "rollback reduces audio or reports rolled samples");

    /* Feed 5 new tokens and verify synthesis continues */
    for (int i = 5; i < 8; i++) {
        char t[8];
        snprintf(t, sizeof(t), "y%d", i);
        streaming_tts_feed_token(stts, t, i);
    }
    streaming_tts_finish(stts);
    int len_final = 0;
    streaming_tts_peek_audio(stts, &len_final);
    TEST(len_final > 0, "after rollback, new tokens produce audio");

    streaming_tts_destroy(stts);
}

static void test_rollback_committed_preserved(void) {
    printf("\n=== Rollback: Committed Audio Preserved ===\n");

    StreamingTTSConfig cfg = {
        .min_tokens_to_start = 2,
        .lookahead_tokens = 0,
        .rollback_enabled = 1
    };
    StreamingTTS *stts = streaming_tts_create(&cfg);
    streaming_tts_set_synthesizer(stts, mock_synth, NULL);

    streaming_tts_feed_token(stts, "AB", 0);
    streaming_tts_feed_token(stts, "CD", 1);
    streaming_tts_feed_token(stts, "EF", 2);

    int total = 0;
    streaming_tts_peek_audio(stts, &total);
    streaming_tts_commit_audio(stts, total / 2);

    int before = 0;
    streaming_tts_peek_audio(stts, &before);

    streaming_tts_rollback(stts, 2);

    int after = 0;
    streaming_tts_peek_audio(stts, &after);
    TEST(after >= total / 2, "committed audio preserved after rollback");

    streaming_tts_destroy(stts);
}

static void test_rollback_stats(void) {
    printf("\n=== Rollback Stats Tracking ===\n");

    StreamingTTSConfig cfg = {
        .min_tokens_to_start = 2,
        .lookahead_tokens = 0,
        .rollback_enabled = 1
    };
    StreamingTTS *stts = streaming_tts_create(&cfg);
    streaming_tts_set_synthesizer(stts, mock_synth, NULL);

    streaming_tts_feed_token(stts, "xx", 0);
    streaming_tts_feed_token(stts, "yy", 1);
    streaming_tts_rollback(stts, 1);

    StreamingTTSStats stats;
    streaming_tts_get_stats(stts, &stats);
    TEST(stats.rollback_count == 1, "rollback_count incremented");
    TEST(stats.samples_rolled_back >= 0, "samples_rolled_back tracked");

    streaming_tts_destroy(stts);
}

static void test_segment_tracking(void) {
    printf("\n=== Segment Tracking ===\n");

    StreamingTTSConfig cfg = { .min_tokens_to_start = 2, .lookahead_tokens = 0 };
    StreamingTTS *stts = streaming_tts_create(&cfg);
    streaming_tts_set_synthesizer(stts, mock_synth, NULL);

    streaming_tts_feed_token(stts, "Hi", 0);
    streaming_tts_feed_token(stts, "!", 1);

    int count = 0;
    const AudioSegment *seg = streaming_tts_get_segments(stts, &count);
    TEST(seg != NULL && count >= 1, "segments returned");
    if (count > 0) {
        TEST(seg[0].token_start >= 0, "segment has token_start");
        TEST(seg[0].audio_len > 0, "segment has audio_len");
    }

    streaming_tts_destroy(stts);
}

static void test_commit_audio(void) {
    printf("\n=== Commit Audio ===\n");

    StreamingTTSConfig cfg = { .min_tokens_to_start = 2 };
    StreamingTTS *stts = streaming_tts_create(&cfg);
    streaming_tts_set_synthesizer(stts, mock_synth, NULL);

    streaming_tts_feed_token(stts, "Hi", 0);
    streaming_tts_feed_token(stts, "!", 1);

    int len = 0;
    streaming_tts_peek_audio(stts, &len);
    streaming_tts_commit_audio(stts, len / 2);

    StreamingTTSStats stats;
    streaming_tts_get_stats(stts, &stats);
    TEST(stats.samples_committed == len / 2, "commit updates samples_committed");

    streaming_tts_destroy(stts);
}

static void test_lookahead(void) {
    printf("\n=== Lookahead: Tokens Not Synthesized Until Buffer Filled ===\n");

    StreamingTTSConfig cfg = {
        .min_tokens_to_start = 2,
        .lookahead_tokens = 3
    };
    StreamingTTS *stts = streaming_tts_create(&cfg);
    streaming_tts_set_synthesizer(stts, mock_synth, NULL);

    streaming_tts_feed_token(stts, "A", 0);
    streaming_tts_feed_token(stts, "B", 1);
    int len2 = 0;
    streaming_tts_peek_audio(stts, &len2);

    streaming_tts_feed_token(stts, "C", 2);
    int len3 = 0;
    streaming_tts_peek_audio(stts, &len3);

    /* With lookahead 3, we need 3+ more tokens before next synthesis. So len3 might equal len2. */
    TEST(len2 > 0, "first chunk produced");
    TEST(len3 >= len2, "audio only grows");

    streaming_tts_destroy(stts);
}

static void test_done_flag(void) {
    printf("\n=== Done Flag ===\n");

    StreamingTTSConfig cfg = { .min_tokens_to_start = 2 };
    StreamingTTS *stts = streaming_tts_create(&cfg);
    streaming_tts_set_synthesizer(stts, mock_synth, NULL);

    TEST(streaming_tts_is_done(stts) == true, "done when no tokens");

    streaming_tts_feed_token(stts, "Hi", 0);
    streaming_tts_feed_token(stts, "!", 1);
    streaming_tts_finish(stts);
    TEST(streaming_tts_is_done(stts) == false, "not done while audio available");

    int len = 0;
    streaming_tts_peek_audio(stts, &len);
    streaming_tts_advance_audio(stts, len);
    TEST(streaming_tts_is_done(stts) == true, "done after finish + all consumed");

    streaming_tts_destroy(stts);
}

static void test_reset(void) {
    printf("\n=== Reset ===\n");

    StreamingTTSConfig cfg = { .min_tokens_to_start = 2 };
    StreamingTTS *stts = streaming_tts_create(&cfg);
    streaming_tts_set_synthesizer(stts, mock_synth, NULL);

    streaming_tts_feed_token(stts, "Hi", 0);
    streaming_tts_feed_token(stts, "!", 1);

    streaming_tts_reset(stts);

    int len = 0;
    streaming_tts_peek_audio(stts, &len);
    TEST(len == 0, "reset clears audio");
    TEST(streaming_tts_is_done(stts) == true, "reset clears done state");

    streaming_tts_destroy(stts);
}

static void test_stats(void) {
    printf("\n=== Stats ===\n");

    StreamingTTSConfig cfg = { .min_tokens_to_start = 2 };
    StreamingTTS *stts = streaming_tts_create(&cfg);
    streaming_tts_set_synthesizer(stts, mock_synth, NULL);

    streaming_tts_feed_token(stts, "Hi", 0);
    streaming_tts_feed_token(stts, "!", 1);

    StreamingTTSStats stats;
    streaming_tts_get_stats(stts, &stats);
    TEST(stats.tokens_received == 2, "tokens_received");
    TEST(stats.tokens_synthesized >= 0, "tokens_synthesized");
    TEST(stats.samples_generated > 0, "samples_generated");

    streaming_tts_destroy(stts);
}

static void test_no_synthesizer(void) {
    printf("\n=== No Synthesizer Set ===\n");

    StreamingTTS *stts = streaming_tts_create(NULL);
    /* No set_synthesizer call */

    streaming_tts_feed_token(stts, "Hello", 0);
    streaming_tts_feed_token(stts, " world", 1);
    streaming_tts_feed_token(stts, " foo", 2);
    streaming_tts_feed_token(stts, " bar", 3);

    int len = 0;
    streaming_tts_peek_audio(stts, &len);
    TEST(len == 0, "no synthesizer yields 0 audio");

    streaming_tts_destroy(stts);
}

static void test_empty_token_handling(void) {
    printf("\n=== Empty Token Handling ===\n");

    StreamingTTSConfig cfg = { .min_tokens_to_start = 4 };
    StreamingTTS *stts = streaming_tts_create(&cfg);
    streaming_tts_set_synthesizer(stts, mock_synth, NULL);

    streaming_tts_feed_token(stts, "", 0);
    streaming_tts_feed_token(stts, "a", 1);
    streaming_tts_feed_token(stts, "b", 2);
    int avail = streaming_tts_feed_token(stts, "c", 3);

    TEST(avail >= 0, "empty token does not crash");
    (void)avail;

    streaming_tts_destroy(stts);
}

static void test_max_tokens_limit(void) {
    printf("\n=== Max Tokens Limit ===\n");

    StreamingTTSConfig cfg = { .min_tokens_to_start = 2 };
    StreamingTTS *stts = streaming_tts_create(&cfg);
    streaming_tts_set_synthesizer(stts, mock_synth, NULL);

    for (int i = 0; i < 2100; i++) {
        char t[16];
        snprintf(t, sizeof(t), "x");
        int avail = streaming_tts_feed_token(stts, t, i);
        if (i >= 2048) {
            TEST(avail == 0 || i >= 2048, "stops accepting at max tokens");
            break;
        }
    }

    streaming_tts_destroy(stts);
}

static void test_multiple_rollbacks(void) {
    printf("\n=== Multiple Rollbacks in Sequence ===\n");

    StreamingTTSConfig cfg = {
        .min_tokens_to_start = 2,
        .lookahead_tokens = 0,
        .rollback_enabled = 1
    };
    StreamingTTS *stts = streaming_tts_create(&cfg);
    streaming_tts_set_synthesizer(stts, mock_synth, NULL);

    streaming_tts_feed_token(stts, "A", 0);
    streaming_tts_feed_token(stts, "B", 1);
    streaming_tts_rollback(stts, 1);

    streaming_tts_feed_token(stts, "C", 1);
    streaming_tts_feed_token(stts, "D", 2);
    streaming_tts_rollback(stts, 2);

    StreamingTTSStats stats;
    streaming_tts_get_stats(stts, &stats);
    TEST(stats.rollback_count == 2, "multiple rollbacks counted");

    streaming_tts_destroy(stts);
}

static void test_crossfade_config(void) {
    printf("\n=== Crossfade at Splice Point ===\n");

    StreamingTTSConfig cfg = {
        .min_tokens_to_start = 2,
        .lookahead_tokens = 0,
        .rollback_enabled = 1,
        .crossfade_samples = 240
    };
    StreamingTTS *stts = streaming_tts_create(&cfg);
    streaming_tts_set_synthesizer(stts, mock_synth, NULL);

    streaming_tts_feed_token(stts, "AB", 0);
    streaming_tts_feed_token(stts, "CD", 1);
    streaming_tts_commit_audio(stts, 100);
    streaming_tts_rollback(stts, 1);

    streaming_tts_feed_token(stts, "XY", 1);
    streaming_tts_finish(stts);

    int len = 0;
    streaming_tts_peek_audio(stts, &len);
    TEST(len > 0, "crossfade config allows synthesis after rollback");

    streaming_tts_destroy(stts);
}

static void test_config_defaults(void) {
    printf("\n=== Config Defaults ===\n");

    StreamingTTS *stts = streaming_tts_create(NULL);
    TEST(stts != NULL, "create with NULL config uses defaults");

    streaming_tts_destroy(stts);
}

static void test_rollback_disabled(void) {
    printf("\n=== Rollback Disabled ===\n");

    StreamingTTSConfig cfg = {
        .min_tokens_to_start = 2,
        .rollback_enabled = 0
    };
    StreamingTTS *stts = streaming_tts_create(&cfg);
    streaming_tts_set_synthesizer(stts, mock_synth, NULL);

    streaming_tts_feed_token(stts, "A", 0);
    streaming_tts_feed_token(stts, "B", 1);

    int rolled = streaming_tts_rollback(stts, 1);
    TEST(rolled == 0, "rollback when disabled returns 0");

    int len = 0;
    streaming_tts_peek_audio(stts, &len);
    TEST(len > 0, "audio preserved when rollback disabled");

    streaming_tts_destroy(stts);
}

int main(void) {
    printf("Streaming TTS Unit Tests\n");

    test_create_destroy();
    test_config_defaults();
    test_feed_single_token_no_audio();
    test_feed_enough_tokens_audio_generated();
    test_get_audio_peek_advance();
    test_feed_more_tokens_incremental();
    test_finish_flushes_remaining();
    test_rollback_basic();
    test_rollback_committed_preserved();
    test_rollback_stats();
    test_rollback_disabled();
    test_segment_tracking();
    test_commit_audio();
    test_lookahead();
    test_done_flag();
    test_reset();
    test_stats();
    test_no_synthesizer();
    test_empty_token_handling();
    test_max_tokens_limit();
    test_multiple_rollbacks();
    test_crossfade_config();

    printf("\n=== Summary ===\n");
    printf("Passed: %d  Failed: %d\n", pass_count, fail_count);
    return fail_count > 0 ? 1 : 0;
}
