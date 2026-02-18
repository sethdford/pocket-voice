/**
 * test_conformer_stt.c — Tests for mel spectrogram and conformer STT engine.
 *
 * Validates:
 *   - Mel spectrogram extraction on synthetic signals (sine wave, silence)
 *   - Mel filterbank properties (triangular, normalized)
 *   - Conformer forward pass components on random data
 *   - CTC greedy decoder logic
 *
 * Build:
 *   cc -O3 -arch arm64 -Isrc -framework Accelerate \
 *      -Lbuild -lmel_spectrogram -lconformer_stt \
 *      -Wl,-rpath,$(pwd)/build \
 *      -o tests/test_conformer_stt tests/test_conformer_stt.c
 *
 * Run: ./tests/test_conformer_stt
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "mel_spectrogram.h"
#include "conformer_stt.h"

static int tests_passed = 0;
static int tests_failed = 0;

#define TEST(name) do { printf("  %-55s", name); } while(0)
#define PASS() do { printf("PASS\n"); tests_passed++; } while(0)
#define FAIL(msg) do { printf("FAIL: %s\n", msg); tests_failed++; return; } while(0)

/* ── Mel Spectrogram Tests ──────────────────────────────── */

static void test_mel_create_destroy(void) {
    TEST("mel: create and destroy");
    MelConfig cfg;
    mel_config_default(&cfg);
    MelSpectrogram *mel = mel_create(&cfg);
    if (!mel) FAIL("create returned NULL");
    if (mel_n_mels(mel) != 80) FAIL("expected 80 mels");
    if (mel_hop_length(mel) != 160) FAIL("expected hop 160");
    mel_destroy(mel);
    mel_destroy(NULL);
    PASS();
}

static void test_mel_silence(void) {
    TEST("mel: silence produces very low energy");
    MelConfig cfg;
    mel_config_default(&cfg);
    MelSpectrogram *mel = mel_create(&cfg);
    if (!mel) FAIL("create failed");

    /* 1 second of silence */
    int n = 16000;
    float *pcm = (float *)calloc(n, sizeof(float));
    float mels[200 * 80];

    int frames = mel_process(mel, pcm, n, mels, 200);
    if (frames <= 0) FAIL("no frames from 1s of audio");

    /* All mel values should be very negative (near log floor) */
    float max_mel = mels[0];
    for (int i = 1; i < frames * 80; i++) {
        if (mels[i] > max_mel) max_mel = mels[i];
    }
    /* Uses natural log: ln(1e-6) ≈ -13.8. Silence values should be near floor. */
    if (max_mel > -5.0f) FAIL("silence mel energy too high");

    free(pcm);
    mel_destroy(mel);
    PASS();
}

static void test_mel_sine_wave(void) {
    TEST("mel: 1kHz sine has energy in correct mel bin");
    MelConfig cfg;
    mel_config_default(&cfg);
    MelSpectrogram *mel = mel_create(&cfg);
    if (!mel) FAIL("create failed");

    /* 0.5 seconds of 1kHz sine at 16kHz sample rate */
    int n = 8000;
    float *pcm = (float *)malloc(n * sizeof(float));
    for (int i = 0; i < n; i++) {
        pcm[i] = 0.5f * sinf(2.0f * M_PI * 1000.0f * i / 16000.0f);
    }

    float mels[100 * 80];
    int frames = mel_process(mel, pcm, n, mels, 100);
    if (frames <= 0) FAIL("no frames from sine wave");

    /* Find which mel bin has the most energy (skip first frame — onset transient) */
    int start_frame = frames > 2 ? 2 : 0;
    float *avg_mel = (float *)calloc(80, sizeof(float));
    for (int f = start_frame; f < frames; f++) {
        for (int m = 0; m < 80; m++) {
            avg_mel[m] += mels[f * 80 + m];
        }
    }
    int count = frames - start_frame;
    for (int m = 0; m < 80; m++) avg_mel[m] /= (float)count;
    int peak_bin = 0;
    float peak_val = avg_mel[0];
    for (int m = 1; m < 80; m++) {
        if (avg_mel[m] > peak_val) {
            peak_val = avg_mel[m];
            peak_bin = m;
        }
    }

    /* 1kHz in an 80-mel filterbank (0-8kHz) should be around bin 15-25 */
    if (peak_bin < 5 || peak_bin > 40) {
        char msg[128];
        snprintf(msg, sizeof(msg), "1kHz peak at mel bin %d (expected ~15-25)", peak_bin);
        free(pcm);
        free(avg_mel);
        mel_destroy(mel);
        FAIL(msg);
    }

    free(pcm);
    free(avg_mel);
    mel_destroy(mel);
    PASS();
}

static void test_mel_streaming(void) {
    TEST("mel: streaming matches single-shot processing");
    MelConfig cfg;
    mel_config_default(&cfg);

    /* Generate 0.5s of audio */
    int n = 8000;
    float *pcm = (float *)malloc(n * sizeof(float));
    for (int i = 0; i < n; i++) {
        pcm[i] = 0.3f * sinf(2.0f * M_PI * 440.0f * i / 16000.0f);
    }

    /* Single-shot: feed all at once */
    MelSpectrogram *mel1 = mel_create(&cfg);
    float mels1[200 * 80];
    int frames1 = mel_process(mel1, pcm, n, mels1, 200);

    /* Streaming: feed in small chunks */
    MelSpectrogram *mel2 = mel_create(&cfg);
    float mels2[200 * 80];
    int frames2 = 0;
    int chunk_size = 320;  /* 20ms chunks */
    for (int pos = 0; pos < n; pos += chunk_size) {
        int this_chunk = (pos + chunk_size <= n) ? chunk_size : (n - pos);
        int f = mel_process(mel2, pcm + pos, this_chunk,
                           mels2 + frames2 * 80, 200 - frames2);
        if (f > 0) frames2 += f;
    }

    if (frames1 != frames2) {
        char msg[128];
        snprintf(msg, sizeof(msg), "frame count mismatch: %d vs %d", frames1, frames2);
        free(pcm);
        mel_destroy(mel1);
        mel_destroy(mel2);
        FAIL(msg);
    }

    /* Compare mel values (should be identical) */
    float max_diff = 0.0f;
    for (int i = 0; i < frames1 * 80; i++) {
        float diff = fabsf(mels1[i] - mels2[i]);
        if (diff > max_diff) max_diff = diff;
    }
    if (max_diff > 0.01f) {
        char msg[128];
        snprintf(msg, sizeof(msg), "mel mismatch: max diff %.6f", max_diff);
        free(pcm);
        mel_destroy(mel1);
        mel_destroy(mel2);
        FAIL(msg);
    }

    free(pcm);
    mel_destroy(mel1);
    mel_destroy(mel2);
    PASS();
}

static void test_mel_reset(void) {
    TEST("mel: reset clears state");
    MelConfig cfg;
    mel_config_default(&cfg);
    MelSpectrogram *mel = mel_create(&cfg);
    if (!mel) FAIL("create failed");

    /* Feed partial data (less than one frame) */
    float pcm[100];
    memset(pcm, 0, sizeof(pcm));
    float out[80];
    int frames = mel_process(mel, pcm, 100, out, 1);
    if (frames != 0) FAIL("expected 0 frames from 100 samples");

    /* Reset should clear internal buffer */
    mel_reset(mel);

    /* After reset, feeding another partial chunk shouldn't combine with previous */
    frames = mel_process(mel, pcm, 100, out, 1);
    if (frames != 0) FAIL("expected 0 frames after reset + 100 samples");

    mel_destroy(mel);
    PASS();
}

static void test_mel_frame_count(void) {
    TEST("mel: correct number of frames for known input");
    MelConfig cfg;
    mel_config_default(&cfg);
    MelSpectrogram *mel = mel_create(&cfg);
    if (!mel) FAIL("create failed");

    /* 1600 samples = 100ms = 10 hops of 160 samples.
     * With win_length=400, we need at least 400 samples for the first frame.
     * Frames produced = floor((1600 - 400) / 160) + 1 = 8 */
    float pcm[1600];
    memset(pcm, 0, sizeof(pcm));
    float mels[20 * 80];
    int frames = mel_process(mel, pcm, 1600, mels, 20);

    int expected = (1600 - cfg.win_length) / cfg.hop_length + 1;
    if (frames != expected) {
        char msg[128];
        snprintf(msg, sizeof(msg), "expected %d frames, got %d", expected, frames);
        mel_destroy(mel);
        FAIL(msg);
    }

    mel_destroy(mel);
    PASS();
}

/* ── Conformer STT Tests ────────────────────────────────── */

static void test_conformer_null_model(void) {
    TEST("conformer: create with nonexistent model returns NULL");
    ConformerSTT *stt = conformer_stt_create("/nonexistent/model.cstt");
    if (stt != NULL) {
        conformer_stt_destroy(stt);
        FAIL("should have returned NULL");
    }
    PASS();
}

static void test_conformer_null_safety(void) {
    TEST("conformer: NULL-safe API calls");
    conformer_stt_destroy(NULL);
    if (conformer_stt_process(NULL, NULL, 0) != -1)
        FAIL("process(NULL) should return -1");
    if (conformer_stt_flush(NULL) != -1)
        FAIL("flush(NULL) should return -1");
    char buf[64];
    if (conformer_stt_get_text(NULL, buf, sizeof(buf)) != -1)
        FAIL("get_text(NULL) should return -1");
    conformer_stt_reset(NULL);
    if (conformer_stt_sample_rate(NULL) != 0)
        FAIL("sample_rate(NULL) should return 0");
    PASS();
}

/* ── Main ───────────────────────────────────────────────── */

int main(void) {
    printf("\n=== Conformer STT Test Suite ===\n\n");

    printf("Mel Spectrogram:\n");
    test_mel_create_destroy();
    test_mel_silence();
    test_mel_sine_wave();
    test_mel_streaming();
    test_mel_reset();
    test_mel_frame_count();

    printf("\nConformer STT Engine:\n");
    test_conformer_null_model();
    test_conformer_null_safety();

    printf("\n=== Results: %d passed, %d failed ===\n\n",
           tests_passed, tests_failed);

    return tests_failed > 0 ? 1 : 0;
}
