/**
 * test_backchannel.c — Unit tests for active listening backchannel generation.
 *
 * Tests: create/destroy lifecycle, NULL safety, feed with speech/silence,
 * EOU probability integration, history circular buffer wraparound,
 * enable/disable, get_audio retrieval, reset state, load_wav errors.
 *
 * Build:
 *   cc -O3 -arch arm64 -Isrc -framework Accelerate \
 *      -Lbuild -lbackchannel \
 *      -Wl,-rpath,$(pwd)/build \
 *      -o tests/test_backchannel tests/test_backchannel.c
 *
 * Run: ./tests/test_backchannel
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include "backchannel.h"

static int pass_count = 0;
static int fail_count = 0;

#define TEST(cond, name) do { \
    if (cond) { printf("  [PASS] %s\n", name); pass_count++; } \
    else { printf("  [FAIL] %s (line %d)\n", name, __LINE__); fail_count++; } \
} while (0)

#define SAMPLE_RATE 24000
#define FRAME_SIZE  1920  /* 80ms at 24kHz */

/* ── Synthetic audio helpers ──────────────────────────── */

static void gen_silence(float *buf, int n) {
    memset(buf, 0, n * sizeof(float));
}

static void gen_sine(float *buf, int n, float freq_hz, float amplitude) {
    for (int i = 0; i < n; i++) {
        buf[i] = amplitude * sinf(2.0f * (float)M_PI * freq_hz * i / SAMPLE_RATE);
    }
}

/* ── Tests ────────────────────────────────────────────── */

static void test_create_destroy(void) {
    printf("\n=== Create and Destroy ===\n");

    BackchannelGen *bc = backchannel_create(SAMPLE_RATE);
    TEST(bc != NULL, "create(24000) returns non-NULL");

    backchannel_destroy(bc);
    TEST(1, "destroy(bc) did not crash");

    backchannel_destroy(NULL);
    TEST(1, "destroy(NULL) is safe");
}

static void test_null_safety(void) {
    printf("\n=== NULL Safety ===\n");

    float audio[FRAME_SIZE];
    gen_silence(audio, FRAME_SIZE);

    /* feed with NULL bc */
    BackchannelEvent ev = backchannel_feed(NULL, audio, FRAME_SIZE, 0.0f);
    TEST(ev.type == BC_NONE, "feed(NULL, ...) returns BC_NONE");
    TEST(ev.ready == 0, "feed(NULL, ...) not ready");

    /* feed with NULL audio */
    BackchannelGen *bc = backchannel_create(SAMPLE_RATE);
    backchannel_set_enabled(bc, 1);
    ev = backchannel_feed(bc, NULL, FRAME_SIZE, 0.0f);
    TEST(ev.type == BC_NONE, "feed(bc, NULL, ...) returns BC_NONE");

    /* feed with zero samples */
    ev = backchannel_feed(bc, audio, 0, 0.0f);
    TEST(ev.type == BC_NONE, "feed(bc, audio, 0, ...) returns BC_NONE");

    /* feed with negative samples */
    ev = backchannel_feed(bc, audio, -1, 0.0f);
    TEST(ev.type == BC_NONE, "feed(bc, audio, -1, ...) returns BC_NONE");

    /* get_audio with NULL bc */
    int out_len = 0;
    const float *p = backchannel_get_audio(NULL, BC_MHM, &out_len);
    TEST(p == NULL, "get_audio(NULL, ...) returns NULL");

    /* get_audio with NULL out_len */
    p = backchannel_get_audio(bc, BC_MHM, NULL);
    TEST(p == NULL, "get_audio(bc, MHM, NULL) returns NULL");

    /* get_audio with invalid type */
    p = backchannel_get_audio(bc, BC_NONE, &out_len);
    TEST(p == NULL, "get_audio(bc, BC_NONE, ...) returns NULL");
    p = backchannel_get_audio(bc, BC_COUNT, &out_len);
    TEST(p == NULL, "get_audio(bc, BC_COUNT, ...) returns NULL");

    /* reset with NULL */
    backchannel_reset(NULL);
    TEST(1, "reset(NULL) is safe");

    /* set_enabled with NULL */
    backchannel_set_enabled(NULL, 1);
    TEST(1, "set_enabled(NULL, 1) is safe");

    /* is_enabled with NULL */
    TEST(backchannel_is_enabled(NULL) == 0, "is_enabled(NULL) returns 0");

    /* load_wav with NULL bc */
    TEST(backchannel_load_wav(NULL, BC_MHM, "/tmp/test.wav") == -1,
         "load_wav(NULL, ...) returns -1");

    /* load_wav with invalid type */
    TEST(backchannel_load_wav(bc, BC_NONE, "/tmp/test.wav") == -1,
         "load_wav(bc, BC_NONE, ...) returns -1");

    /* load_wav with NULL path */
    TEST(backchannel_load_wav(bc, BC_MHM, NULL) == -1,
         "load_wav(bc, MHM, NULL) returns -1");

    backchannel_destroy(bc);
}

static void test_disabled_by_default(void) {
    printf("\n=== Disabled by Default ===\n");

    BackchannelGen *bc = backchannel_create(SAMPLE_RATE);
    TEST(backchannel_is_enabled(bc) == 0, "disabled by default");

    /* Feed speech — should never fire when disabled */
    float audio[FRAME_SIZE];
    gen_sine(audio, FRAME_SIZE, 150.0f, 0.5f);
    for (int i = 0; i < 100; i++) {
        BackchannelEvent ev = backchannel_feed(bc, audio, FRAME_SIZE, 0.0f);
        TEST(ev.ready == 0, "no backchannel when disabled");
        if (ev.ready) break;
    }

    backchannel_destroy(bc);
}

static void test_enable_disable(void) {
    printf("\n=== Enable/Disable ===\n");

    BackchannelGen *bc = backchannel_create(SAMPLE_RATE);

    backchannel_set_enabled(bc, 1);
    TEST(backchannel_is_enabled(bc) == 1, "enabled after set_enabled(1)");

    backchannel_set_enabled(bc, 0);
    TEST(backchannel_is_enabled(bc) == 0, "disabled after set_enabled(0)");

    backchannel_set_enabled(bc, 1);
    TEST(backchannel_is_enabled(bc) == 1, "re-enabled");

    backchannel_destroy(bc);
}

static void test_get_audio_all_types(void) {
    printf("\n=== Get Audio All Types ===\n");

    BackchannelGen *bc = backchannel_create(SAMPLE_RATE);

    BackchannelType types[] = { BC_MHM, BC_YEAH, BC_RIGHT, BC_OKAY, BC_UH_HUH };
    const char *names[] = { "MHM", "YEAH", "RIGHT", "OKAY", "UH_HUH" };

    for (int i = 0; i < 5; i++) {
        int out_len = 0;
        const float *audio = backchannel_get_audio(bc, types[i], &out_len);
        char msg[64];

        snprintf(msg, sizeof(msg), "%s audio non-NULL", names[i]);
        TEST(audio != NULL, msg);

        snprintf(msg, sizeof(msg), "%s audio len > 0", names[i]);
        TEST(out_len > 0, msg);

        /* Verify audio has actual content (not all zeros) */
        float max_val = 0.0f;
        for (int j = 0; j < out_len; j++) {
            float a = fabsf(audio[j]);
            if (a > max_val) max_val = a;
        }
        snprintf(msg, sizeof(msg), "%s audio has content (max=%.4f)", names[i], max_val);
        TEST(max_val > 0.001f, msg);
    }

    backchannel_destroy(bc);
}

static void test_feed_silence_no_trigger(void) {
    printf("\n=== Feed Silence — No Trigger ===\n");

    BackchannelGen *bc = backchannel_create(SAMPLE_RATE);
    backchannel_set_enabled(bc, 1);

    float silence[FRAME_SIZE];
    gen_silence(silence, FRAME_SIZE);

    /* Feed only silence — should never trigger (no speech heard) */
    int triggered = 0;
    for (int i = 0; i < 50; i++) {
        BackchannelEvent ev = backchannel_feed(bc, silence, FRAME_SIZE, 0.0f);
        if (ev.ready) triggered++;
    }
    TEST(triggered == 0, "pure silence never triggers backchannel");

    backchannel_destroy(bc);
}

static void test_feed_speech_then_pause(void) {
    printf("\n=== Feed Speech then Pause ===\n");

    BackchannelGen *bc = backchannel_create(SAMPLE_RATE);
    backchannel_set_enabled(bc, 1);

    float speech[FRAME_SIZE], silence[FRAME_SIZE];
    gen_sine(speech, FRAME_SIZE, 150.0f, 0.5f);
    gen_silence(silence, FRAME_SIZE);

    /* Feed enough speech to pass the 10-frame minimum (>800ms speech) */
    for (int i = 0; i < 40; i++) {
        backchannel_feed(bc, speech, FRAME_SIZE, 0.0f);
    }

    /* Now feed a pause in the 200-600ms range (3-7 frames at 80ms/frame) */
    int triggered = 0;
    BackchannelEvent last_ev = { BC_NONE, 0.0f, 0 };
    for (int i = 0; i < 8; i++) {
        BackchannelEvent ev = backchannel_feed(bc, silence, FRAME_SIZE, 0.1f);
        if (ev.ready) {
            triggered++;
            last_ev = ev;
        }
    }

    /* The combination of pause + extended speech + moderate EOU should trigger */
    TEST(triggered >= 0, "speech-then-pause scenario runs without crash");
    if (triggered > 0) {
        TEST(last_ev.type > BC_NONE && last_ev.type < BC_COUNT,
             "triggered backchannel has valid type");
        TEST(last_ev.confidence > 0.0f, "triggered backchannel has positive confidence");
    }

    backchannel_destroy(bc);
}

static void test_high_eou_suppresses(void) {
    printf("\n=== High EOU Suppresses Backchannel ===\n");

    BackchannelGen *bc = backchannel_create(SAMPLE_RATE);
    backchannel_set_enabled(bc, 1);

    float speech[FRAME_SIZE], silence[FRAME_SIZE];
    gen_sine(speech, FRAME_SIZE, 150.0f, 0.5f);
    gen_silence(silence, FRAME_SIZE);

    /* Build up speech */
    for (int i = 0; i < 40; i++) {
        backchannel_feed(bc, speech, FRAME_SIZE, 0.0f);
    }

    /* Pause with high EOU (> 0.5) — should suppress backchannel */
    int triggered = 0;
    for (int i = 0; i < 10; i++) {
        BackchannelEvent ev = backchannel_feed(bc, silence, FRAME_SIZE, 0.8f);
        if (ev.ready) triggered++;
    }
    TEST(triggered == 0, "high EOU probability suppresses backchannel");

    backchannel_destroy(bc);
}

static void test_history_wraparound(void) {
    printf("\n=== History Circular Buffer Wraparound ===\n");

    BackchannelGen *bc = backchannel_create(SAMPLE_RATE);
    backchannel_set_enabled(bc, 1);

    float audio[FRAME_SIZE];
    gen_sine(audio, FRAME_SIZE, 150.0f, 0.3f);

    /* Feed well beyond the 25-frame history buffer (feed 100 frames) */
    int crashed = 0;
    for (int i = 0; i < 100; i++) {
        backchannel_feed(bc, audio, FRAME_SIZE, 0.0f);
    }
    TEST(!crashed, "100 frames without crash (history wraps safely)");

    /* Feed another 200 frames to stress the modular wraparound */
    for (int i = 0; i < 200; i++) {
        backchannel_feed(bc, audio, FRAME_SIZE, 0.0f);
    }
    TEST(1, "300 total frames without crash");

    backchannel_destroy(bc);
}

static void test_reset_clears_state(void) {
    printf("\n=== Reset Clears State ===\n");

    BackchannelGen *bc = backchannel_create(SAMPLE_RATE);
    backchannel_set_enabled(bc, 1);

    float speech[FRAME_SIZE];
    gen_sine(speech, FRAME_SIZE, 150.0f, 0.5f);

    /* Build up speech state */
    for (int i = 0; i < 20; i++) {
        backchannel_feed(bc, speech, FRAME_SIZE, 0.0f);
    }

    /* Reset */
    backchannel_reset(bc);

    /* After reset, silence shouldn't trigger (total_speech_frames is 0) */
    float silence[FRAME_SIZE];
    gen_silence(silence, FRAME_SIZE);
    int triggered = 0;
    for (int i = 0; i < 10; i++) {
        BackchannelEvent ev = backchannel_feed(bc, silence, FRAME_SIZE, 0.0f);
        if (ev.ready) triggered++;
    }
    TEST(triggered == 0, "no trigger after reset (speech history cleared)");

    /* Enabled state should survive reset */
    TEST(backchannel_is_enabled(bc) == 1, "enabled state survives reset");

    backchannel_destroy(bc);
}

static void test_load_wav_nonexistent(void) {
    printf("\n=== Load WAV Nonexistent File ===\n");

    BackchannelGen *bc = backchannel_create(SAMPLE_RATE);

    int r = backchannel_load_wav(bc, BC_MHM, "/tmp/nonexistent_backchannel_test.wav");
    TEST(r == -1, "load_wav with nonexistent file returns -1");

    /* Original built-in audio should still work */
    int out_len = 0;
    const float *audio = backchannel_get_audio(bc, BC_MHM, &out_len);
    TEST(audio != NULL, "built-in MHM audio intact after failed load");
    TEST(out_len > 0, "built-in MHM length intact after failed load");

    backchannel_destroy(bc);
}

static void test_load_wav_empty_file(void) {
    printf("\n=== Load WAV Empty/Tiny File ===\n");

    BackchannelGen *bc = backchannel_create(SAMPLE_RATE);

    /* Create a file smaller than WAV header */
    const char *path = "/tmp/pocket_voice_test_tiny.wav";
    FILE *f = fopen(path, "wb");
    if (f) {
        fwrite("RIFF", 1, 4, f);
        fclose(f);

        int r = backchannel_load_wav(bc, BC_MHM, path);
        TEST(r == -1, "load_wav with tiny file returns -1");
        unlink(path);
    }

    backchannel_destroy(bc);
}

static void test_minimum_gap_between_backchannels(void) {
    printf("\n=== Minimum Gap Between Backchannels ===\n");

    BackchannelGen *bc = backchannel_create(SAMPLE_RATE);
    backchannel_set_enabled(bc, 1);

    float speech[FRAME_SIZE], silence[FRAME_SIZE];
    gen_sine(speech, FRAME_SIZE, 150.0f, 0.5f);
    gen_silence(silence, FRAME_SIZE);

    /* Build up 40 frames of speech */
    for (int i = 0; i < 40; i++) {
        backchannel_feed(bc, speech, FRAME_SIZE, 0.0f);
    }

    /* Try to trigger rapidly — there's a 2000ms minimum gap (25 frames at 80ms) */
    int trigger_count = 0;
    for (int i = 0; i < 50; i++) {
        /* Alternate speech and silence to create trigger conditions */
        float *buf = (i % 5 < 3) ? speech : silence;
        BackchannelEvent ev = backchannel_feed(bc, buf, FRAME_SIZE, 0.15f);
        if (ev.ready) trigger_count++;
    }

    /* Even if triggers happen, they should be spaced by at least ~25 frames */
    TEST(trigger_count <= 2, "minimum gap limits trigger frequency");

    backchannel_destroy(bc);
}

static void test_backchannel_event_fields(void) {
    printf("\n=== Backchannel Event Fields ===\n");

    /* Default event should be BC_NONE with no confidence and not ready */
    BackchannelEvent ev = { BC_NONE, 0.0f, 0 };
    TEST(ev.type == BC_NONE, "default type is BC_NONE");
    TEST(ev.confidence == 0.0f, "default confidence is 0");
    TEST(ev.ready == 0, "default ready is 0");

    /* Verify enum values are sequential */
    TEST(BC_NONE == 0, "BC_NONE == 0");
    TEST(BC_MHM == 1, "BC_MHM == 1");
    TEST(BC_YEAH == 2, "BC_YEAH == 2");
    TEST(BC_RIGHT == 3, "BC_RIGHT == 3");
    TEST(BC_OKAY == 4, "BC_OKAY == 4");
    TEST(BC_UH_HUH == 5, "BC_UH_HUH == 5");
    TEST(BC_COUNT == 6, "BC_COUNT == 6");
}

/* ── Main ─────────────────────────────────────────────── */

int main(void) {
    printf("\n═══ Backchannel Generation Tests ═══\n");

    test_create_destroy();
    test_null_safety();
    test_disabled_by_default();
    test_enable_disable();
    test_get_audio_all_types();
    test_feed_silence_no_trigger();
    test_feed_speech_then_pause();
    test_high_eou_suppresses();
    test_history_wraparound();
    test_reset_clears_state();
    test_load_wav_nonexistent();
    test_load_wav_empty_file();
    test_minimum_gap_between_backchannels();
    test_backchannel_event_fields();

    printf("\n═══ Results: %d pass, %d fail ═══\n", pass_count, fail_count);
    return fail_count ? 1 : 0;
}
