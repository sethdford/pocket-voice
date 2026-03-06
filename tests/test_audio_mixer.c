/**
 * test_audio_mixer.c — Tests for the audio mixer module.
 *
 * Tests: create/destroy, single/multi-channel mix, ducking, crossfade,
 *        flush, overflow, gain, priority, soft limiter.
 *
 * Build: make test-audio-mixer
 * Run: ./build/test-audio-mixer
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>

#include "audio_mixer.h"

static int tests_passed = 0;
static int tests_failed = 0;

#define TEST(name) do { printf("  %-50s", name); } while(0)
#define PASS() do { printf("PASS\n"); tests_passed++; } while(0)
#define FAIL(msg) do { printf("FAIL: %s\n", msg); tests_failed++; } while(0)

#define ASSERT(cond, msg) do { if (!(cond)) { FAIL(msg); return; } } while(0)
#define ASSERT_FLOAT_NEAR(a, b, eps, msg) \
    ASSERT(fabsf((a) - (b)) < (eps), msg)

/* ── Create/Destroy ──────────────────────────────────────────────────────── */

static void test_create_destroy(void) {
    TEST("audio_mixer: create/destroy");
    AudioMixerConfig cfg = {
        .sample_rate = 24000,
        .block_size = 480,
        .ducking_gain = 0.3f,
        .crossfade_samples = 240
    };
    AudioMixer *m = audio_mixer_create(&cfg);
    ASSERT(m != NULL, "create returned NULL");
    audio_mixer_destroy(m);
    audio_mixer_destroy(NULL);  /* NULL-safe */
    PASS();
}

static void test_create_invalid_config(void) {
    TEST("audio_mixer: create rejects invalid config");
    AudioMixerConfig cfg = { .sample_rate = 24000, .block_size = 0 };
    AudioMixer *m = audio_mixer_create(&cfg);
    ASSERT(m == NULL, "should reject block_size 0");
    cfg.block_size = 480;
    cfg.sample_rate = 0;
    m = audio_mixer_create(&cfg);
    ASSERT(m == NULL, "should reject sample_rate 0");
    PASS();
}

/* ── Single Channel Write/Read ───────────────────────────────────────────── */

static void test_single_channel_write_read(void) {
    TEST("audio_mixer: write to single channel, read back");
    AudioMixerConfig cfg = {
        .sample_rate = 24000,
        .block_size = 480,
        .ducking_gain = 0.3f,
        .crossfade_samples = 120
    };
    AudioMixer *m = audio_mixer_create(&cfg);
    ASSERT(m != NULL, "create failed");

    float in[960];
    for (int i = 0; i < 960; i++) in[i] = 0.5f;

    int written = audio_mixer_write(m, MIX_CHANNEL_MAIN, in, 960);
    ASSERT(written == 960, "write should accept all samples");

    float out[960];
    int read = audio_mixer_read(m, out, 960);
    ASSERT(read == 960, "read should return 960 samples");
    /* First block is fade-in (gain 0); second block has full level */
    ASSERT_FLOAT_NEAR(out[500], 0.5f, 0.05f, "sample in second block");
    ASSERT_FLOAT_NEAR(out[959], 0.5f, 0.05f, "last sample");

    audio_mixer_destroy(m);
    PASS();
}

/* ── Multiple Channels Mixing ─────────────────────────────────────────────── */

static void test_multiple_channels_mix(void) {
    TEST("audio_mixer: multiple channels mix correctly");
    AudioMixerConfig cfg = {
        .sample_rate = 24000,
        .block_size = 480,
        .ducking_gain = 0.3f,
        .crossfade_samples = 120
    };
    AudioMixer *m = audio_mixer_create(&cfg);
    ASSERT(m != NULL, "create failed");

    float main_buf[960], bc_buf[960];
    for (int i = 0; i < 960; i++) {
        main_buf[i] = 0.3f;
        bc_buf[i] = 0.2f;
    }
    audio_mixer_write(m, MIX_CHANNEL_MAIN, main_buf, 960);
    audio_mixer_write(m, MIX_CHANNEL_BACKCHANNEL, bc_buf, 960);

    /* Need 2 blocks: first is fade-in (silent), second has mix */
    float out[960];
    int read = audio_mixer_read(m, out, 960);
    ASSERT(read == 960, "should read two blocks");
    /* MAIN priority 10 > BACKCHANNEL 5, so MAIN not ducked. Both contribute. */
    ASSERT(out[500] > 0.35f && out[500] < 0.65f, "mixed sum should be ~0.5");

    audio_mixer_destroy(m);
    PASS();
}

/* ── Ducking ────────────────────────────────────────────────────────────── */

static void test_ducking(void) {
    TEST("audio_mixer: MAIN ducked when BACKCHANNEL active");
    AudioMixerConfig cfg = {
        .sample_rate = 24000,
        .block_size = 480,
        .ducking_gain = 0.3f,
        .crossfade_samples = 10
    };
    AudioMixer *m = audio_mixer_create(&cfg);
    ASSERT(m != NULL, "create failed");

    /* MAIN (priority 10) and BACKCHANNEL (5). MAIN is highest, BACKCHANNEL gets ducked. */
    float main_buf[960], bc_buf[960];
    for (int i = 0; i < 960; i++) {
        main_buf[i] = 0.8f;
        bc_buf[i] = 0.6f;
    }
    audio_mixer_write(m, MIX_CHANNEL_MAIN, main_buf, 960);
    audio_mixer_write(m, MIX_CHANNEL_BACKCHANNEL, bc_buf, 960);

    float out[960];
    audio_mixer_read(m, out, 960);
    /* MAIN full (0.8), BACKCHANNEL ducked 0.3 (0.6*0.3=0.18). Sum ~0.98. First block fade-in. */
    ASSERT(out[500] > 0.8f, "output should reflect MAIN + ducked BC");

    audio_mixer_destroy(m);
    PASS();
}

/* ── Priority Ducking ────────────────────────────────────────────────────── */

static void test_priority_ducking(void) {
    TEST("audio_mixer: higher priority ducks lower");
    AudioMixerConfig cfg = {
        .sample_rate = 24000,
        .block_size = 480,
        .ducking_gain = 0.2f,
        .crossfade_samples = 10
    };
    AudioMixer *m = audio_mixer_create(&cfg);
    ASSERT(m != NULL, "create failed");

    audio_mixer_set_priority(m, MIX_CHANNEL_BACKCHANNEL, 15);  /* BC now highest */
    audio_mixer_set_priority(m, MIX_CHANNEL_MAIN, 5);          /* MAIN lower */

    float main_buf[960], bc_buf[960];
    for (int i = 0; i < 960; i++) {
        main_buf[i] = 0.5f;
        bc_buf[i] = 0.5f;
    }
    audio_mixer_write(m, MIX_CHANNEL_MAIN, main_buf, 960);
    audio_mixer_write(m, MIX_CHANNEL_BACKCHANNEL, bc_buf, 960);

    float out[960];
    audio_mixer_read(m, out, 960);
    /* BACKCHANNEL full (0.5), MAIN ducked (0.5*0.2=0.1). Sum 0.6. First block fade-in. */
    ASSERT(out[500] > 0.5f && out[500] < 0.7f, "priority should affect ducking");

    audio_mixer_destroy(m);
    PASS();
}

/* ── Crossfade ───────────────────────────────────────────────────────────── */

static void test_crossfade_activation(void) {
    TEST("audio_mixer: crossfade on channel activation");
    AudioMixerConfig cfg = {
        .sample_rate = 24000,
        .block_size = 480,
        .ducking_gain = 0.3f,
        .crossfade_samples = 240
    };
    AudioMixer *m = audio_mixer_create(&cfg);
    ASSERT(m != NULL, "create failed");

    float buf[1440];  /* 3 blocks */
    for (int i = 0; i < 1440; i++) buf[i] = 1.0f;
    audio_mixer_write(m, MIX_CHANNEL_MAIN, buf, 1440);

    float out[480];
    audio_mixer_read(m, out, 480);
    /* First block: fade in from 0, so start of block may be quieter */
    ASSERT(out[0] >= 0.0f && out[0] <= 1.0f, "fade-in within range");
    ASSERT(audio_mixer_pending(m, MIX_CHANNEL_MAIN) > 0, "more data pending");

    audio_mixer_destroy(m);
    PASS();
}

/* ── Flush Single Channel ────────────────────────────────────────────────── */

static void test_flush_single_channel(void) {
    TEST("audio_mixer: flush single channel");
    AudioMixerConfig cfg = {
        .sample_rate = 24000,
        .block_size = 480,
        .ducking_gain = 0.3f,
        .crossfade_samples = 120
    };
    AudioMixer *m = audio_mixer_create(&cfg);
    ASSERT(m != NULL, "create failed");

    float buf[960];
    for (int i = 0; i < 960; i++) buf[i] = 0.5f;
    audio_mixer_write(m, MIX_CHANNEL_MAIN, buf, 960);
    audio_mixer_write(m, MIX_CHANNEL_BACKCHANNEL, buf, 480);

    audio_mixer_flush(m, MIX_CHANNEL_MAIN);
    ASSERT(audio_mixer_pending(m, MIX_CHANNEL_MAIN) == 0, "MAIN should be empty");
    ASSERT(audio_mixer_pending(m, MIX_CHANNEL_BACKCHANNEL) == 480, "BC should remain");
    ASSERT(audio_mixer_any_active(m) == 1, "BC still active");

    audio_mixer_destroy(m);
    PASS();
}

/* ── Flush All ───────────────────────────────────────────────────────────── */

static void test_flush_all(void) {
    TEST("audio_mixer: flush all channels");
    AudioMixerConfig cfg = {
        .sample_rate = 24000,
        .block_size = 480,
        .ducking_gain = 0.3f,
        .crossfade_samples = 120
    };
    AudioMixer *m = audio_mixer_create(&cfg);
    ASSERT(m != NULL, "create failed");

    float buf[480];
    for (int i = 0; i < 480; i++) buf[i] = 0.5f;
    audio_mixer_write(m, MIX_CHANNEL_MAIN, buf, 480);
    audio_mixer_write(m, MIX_CHANNEL_BACKCHANNEL, buf, 480);

    audio_mixer_flush_all(m);
    ASSERT(audio_mixer_any_active(m) == 0, "no channel active");
    ASSERT(audio_mixer_pending(m, MIX_CHANNEL_MAIN) == 0, "MAIN empty");
    ASSERT(audio_mixer_pending(m, MIX_CHANNEL_BACKCHANNEL) == 0, "BC empty");

    float out[480];
    int read = audio_mixer_read(m, out, 480);
    ASSERT(read == 0, "read returns 0 when all flushed");

    audio_mixer_destroy(m);
    PASS();
}

/* ── Overflow ────────────────────────────────────────────────────────────── */

static void test_overflow(void) {
    TEST("audio_mixer: overflow drops samples");
    AudioMixerConfig cfg = {
        .sample_rate = 24000,
        .block_size = 480,
        .ducking_gain = 0.3f,
        .crossfade_samples = 120
    };
    AudioMixer *m = audio_mixer_create(&cfg);
    ASSERT(m != NULL, "create failed");

    /* Ring is ~48k samples. Write way more than we can buffer. */
    int total = 100000;
    float *big = malloc((size_t)total * sizeof(float));
    ASSERT(big != NULL, "malloc");
    for (int i = 0; i < total; i++) big[i] = 0.5f;

    int written = audio_mixer_write(m, MIX_CHANNEL_MAIN, big, total);
    ASSERT(written < total, "should drop some samples when buffer full");
    int pending = audio_mixer_pending(m, MIX_CHANNEL_MAIN);
    ASSERT(pending > 0 && pending <= 70000, "reasonable pending after overflow");

    free(big);
    audio_mixer_destroy(m);
    PASS();
}

/* ── Empty Read ──────────────────────────────────────────────────────────── */

static void test_empty_read(void) {
    TEST("audio_mixer: empty read returns 0");
    AudioMixerConfig cfg = {
        .sample_rate = 24000,
        .block_size = 480,
        .ducking_gain = 0.3f,
        .crossfade_samples = 120
    };
    AudioMixer *m = audio_mixer_create(&cfg);
    ASSERT(m != NULL, "create failed");

    float out[960];
    int read = audio_mixer_read(m, out, 960);
    ASSERT(read == 0, "read with no data returns 0");

    audio_mixer_destroy(m);
    PASS();
}

/* ── Gain ────────────────────────────────────────────────────────────────── */

static void test_gain_mute(void) {
    TEST("audio_mixer: gain 0.0 mutes channel");
    AudioMixerConfig cfg = {
        .sample_rate = 24000,
        .block_size = 480,
        .ducking_gain = 0.3f,
        .crossfade_samples = 10
    };
    AudioMixer *m = audio_mixer_create(&cfg);
    ASSERT(m != NULL, "create failed");

    float buf[960];
    for (int i = 0; i < 960; i++) buf[i] = 1.0f;
    audio_mixer_write(m, MIX_CHANNEL_MAIN, buf, 960);
    audio_mixer_set_gain(m, MIX_CHANNEL_MAIN, 0.0f);

    float out[960];
    audio_mixer_read(m, out, 960);
    ASSERT_FLOAT_NEAR(out[500], 0.0f, 0.001f, "muted channel contributes 0");

    audio_mixer_destroy(m);
    PASS();
}

static void test_gain_half(void) {
    TEST("audio_mixer: gain 0.5 halves level");
    AudioMixerConfig cfg = {
        .sample_rate = 24000,
        .block_size = 480,
        .ducking_gain = 0.3f,
        .crossfade_samples = 10
    };
    AudioMixer *m = audio_mixer_create(&cfg);
    ASSERT(m != NULL, "create failed");

    float buf[960];
    for (int i = 0; i < 960; i++) buf[i] = 1.0f;
    audio_mixer_write(m, MIX_CHANNEL_MAIN, buf, 960);
    audio_mixer_set_gain(m, MIX_CHANNEL_MAIN, 0.5f);

    float out[960];
    audio_mixer_read(m, out, 960);
    ASSERT_FLOAT_NEAR(out[500], 0.5f, 0.05f, "gain 0.5 produces ~0.5");

    audio_mixer_destroy(m);
    PASS();
}

/* ── Channel Active / Any Active ─────────────────────────────────────────── */

static void test_channel_active(void) {
    TEST("audio_mixer: channel_active / any_active");
    AudioMixerConfig cfg = {
        .sample_rate = 24000,
        .block_size = 480,
        .ducking_gain = 0.3f,
        .crossfade_samples = 120
    };
    AudioMixer *m = audio_mixer_create(&cfg);
    ASSERT(m != NULL, "create failed");

    ASSERT(audio_mixer_any_active(m) == 0, "empty mixer");
    ASSERT(audio_mixer_channel_active(m, MIX_CHANNEL_MAIN) == 0, "MAIN empty");

    float buf[480];
    for (int i = 0; i < 480; i++) buf[i] = 0.5f;
    audio_mixer_write(m, MIX_CHANNEL_MAIN, buf, 480);

    ASSERT(audio_mixer_any_active(m) == 1, "has data");
    ASSERT(audio_mixer_channel_active(m, MIX_CHANNEL_MAIN) == 1, "MAIN active");

    audio_mixer_read(m, buf, 480);
    ASSERT(audio_mixer_any_active(m) == 0, "after read empty");
    ASSERT(audio_mixer_channel_active(m, MIX_CHANNEL_MAIN) == 0, "MAIN empty after read");

    audio_mixer_destroy(m);
    PASS();
}

/* ── Pending ─────────────────────────────────────────────────────────────── */

static void test_pending(void) {
    TEST("audio_mixer: pending returns correct count");
    AudioMixerConfig cfg = {
        .sample_rate = 24000,
        .block_size = 480,
        .ducking_gain = 0.3f,
        .crossfade_samples = 120
    };
    AudioMixer *m = audio_mixer_create(&cfg);
    ASSERT(m != NULL, "create failed");

    float buf[1000];
    for (int i = 0; i < 1000; i++) buf[i] = 0.5f;
    audio_mixer_write(m, MIX_CHANNEL_MAIN, buf, 1000);

    ASSERT(audio_mixer_pending(m, MIX_CHANNEL_MAIN) == 1000, "pending 1000");
    audio_mixer_read(m, buf, 480);
    ASSERT(audio_mixer_pending(m, MIX_CHANNEL_MAIN) == 520, "pending 520 after read");

    audio_mixer_destroy(m);
    PASS();
}

/* ── Soft Limiter ─────────────────────────────────────────────────────────── */

static void test_soft_limiter(void) {
    TEST("audio_mixer: soft limiter prevents clipping");
    AudioMixerConfig cfg = {
        .sample_rate = 24000,
        .block_size = 480,
        .ducking_gain = 1.0f,
        .crossfade_samples = 10
    };
    AudioMixer *m = audio_mixer_create(&cfg);
    ASSERT(m != NULL, "create failed");

    /* Sum of sources can exceed 1.0 — limiter should clamp */
    float buf[960];
    for (int i = 0; i < 960; i++) buf[i] = 0.8f;
    audio_mixer_write(m, MIX_CHANNEL_MAIN, buf, 960);
    audio_mixer_write(m, MIX_CHANNEL_BACKCHANNEL, buf, 960);
    audio_mixer_set_priority(m, MIX_CHANNEL_MAIN, 10);
    audio_mixer_set_priority(m, MIX_CHANNEL_BACKCHANNEL, 10);  /* Same priority, no ducking */
    audio_mixer_set_gain(m, MIX_CHANNEL_BACKCHANNEL, 1.0f);

    float out[960];
    audio_mixer_read(m, out, 960);
    for (int i = 500; i < 960; i++) {
        ASSERT(out[i] <= 1.01f && out[i] >= -1.01f, "output must not clip");
    }

    audio_mixer_destroy(m);
    PASS();
}

/* ── Reset ───────────────────────────────────────────────────────────────── */

static void test_reset(void) {
    TEST("audio_mixer: reset clears state");
    AudioMixerConfig cfg = {
        .sample_rate = 24000,
        .block_size = 480,
        .ducking_gain = 0.3f,
        .crossfade_samples = 120
    };
    AudioMixer *m = audio_mixer_create(&cfg);
    ASSERT(m != NULL, "create failed");

    float buf[480];
    for (int i = 0; i < 480; i++) buf[i] = 0.5f;
    audio_mixer_write(m, MIX_CHANNEL_MAIN, buf, 480);
    audio_mixer_set_gain(m, MIX_CHANNEL_MAIN, 0.5f);

    audio_mixer_reset(m);
    ASSERT(audio_mixer_pending(m, MIX_CHANNEL_MAIN) == 0, "pending cleared");
    ASSERT(audio_mixer_any_active(m) == 0, "no active after reset");

    audio_mixer_destroy(m);
    PASS();
}

/* ── Write Validation ────────────────────────────────────────────────────── */

static void test_write_validation(void) {
    TEST("audio_mixer: write validates inputs");
    AudioMixerConfig cfg = {
        .sample_rate = 24000,
        .block_size = 480,
        .ducking_gain = 0.3f,
        .crossfade_samples = 120
    };
    AudioMixer *m = audio_mixer_create(&cfg);
    ASSERT(m != NULL, "create failed");

    float buf[100];
    int w = audio_mixer_write(NULL, MIX_CHANNEL_MAIN, buf, 100);
    ASSERT(w == 0, "NULL mixer returns 0");
    w = audio_mixer_write(m, MIX_CHANNEL_MAIN, NULL, 100);
    ASSERT(w == 0, "NULL pcm returns 0");
    w = audio_mixer_write(m, MIX_CHANNEL_MAIN, buf, 0);
    ASSERT(w == 0, "n_samples 0 returns 0");
    w = audio_mixer_write(m, (MixChannel)(MIX_CHANNEL_COUNT + 1), buf, 100);
    ASSERT(w == 0, "invalid channel returns 0");

    audio_mixer_destroy(m);
    PASS();
}

/* ── Multiple Blocks Read ────────────────────────────────────────────────── */

static void test_multiple_blocks_read(void) {
    TEST("audio_mixer: multiple blocks read sequentially");
    AudioMixerConfig cfg = {
        .sample_rate = 24000,
        .block_size = 480,
        .ducking_gain = 0.3f,
        .crossfade_samples = 120
    };
    AudioMixer *m = audio_mixer_create(&cfg);
    ASSERT(m != NULL, "create failed");

    float buf[1440];  /* 3 blocks */
    for (int i = 0; i < 1440; i++) buf[i] = (float)(i % 100) / 100.0f;
    audio_mixer_write(m, MIX_CHANNEL_MAIN, buf, 1440);

    float out[480];
    int r1 = audio_mixer_read(m, out, 480);
    int r2 = audio_mixer_read(m, out, 480);
    int r3 = audio_mixer_read(m, out, 480);
    int r4 = audio_mixer_read(m, out, 480);

    ASSERT(r1 == 480 && r2 == 480 && r3 == 480, "first three blocks full");
    ASSERT(r4 == 0, "fourth block empty (only 3 blocks of data)");
    ASSERT(audio_mixer_pending(m, MIX_CHANNEL_MAIN) == 0, "all consumed");

    audio_mixer_destroy(m);
    PASS();
}

/* ── Main ────────────────────────────────────────────────────────────────── */

int main(void) {
    printf("╔══════════════════════════════════════════════╗\n");
    printf("║  Audio Mixer Tests                          ║\n");
    printf("╚══════════════════════════════════════════════╝\n\n");

    printf("[Create/Destroy]\n");
    test_create_destroy();
    test_create_invalid_config();

    printf("\n[Single/Multi Channel]\n");
    test_single_channel_write_read();
    test_multiple_channels_mix();

    printf("\n[Ducking]\n");
    test_ducking();
    test_priority_ducking();

    printf("\n[Crossfade]\n");
    test_crossfade_activation();

    printf("\n[Flush]\n");
    test_flush_single_channel();
    test_flush_all();

    printf("\n[Edge Cases]\n");
    test_overflow();
    test_empty_read();
    test_write_validation();
    test_multiple_blocks_read();

    printf("\n[Gain]\n");
    test_gain_mute();
    test_gain_half();

    printf("\n[State]\n");
    test_channel_active();
    test_pending();
    test_reset();

    printf("\n[Soft Limiter]\n");
    test_soft_limiter();

    printf("\n════════════════════════════════════════════════\n");
    printf("  Results: %d passed, %d failed\n", tests_passed, tests_failed);
    printf("════════════════════════════════════════════════\n");

    return tests_failed > 0 ? 1 : 0;
}
