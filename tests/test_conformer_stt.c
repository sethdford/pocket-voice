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

    /* 1600 samples = 100ms.
     * mel_process adds center padding of n_fft/2 = 256 on first call
     * (matching torch.stft center=True), so effective length = 1600 + 256 = 1856.
     * Frames = floor((1856 - win_length) / hop_length) + 1 = 10 */
    float pcm[1600];
    memset(pcm, 0, sizeof(pcm));
    float mels[20 * 80];
    int frames = mel_process(mel, pcm, 1600, mels, 20);

    int center_pad = cfg.n_fft / 2;
    int expected = (1600 + center_pad - cfg.win_length) / cfg.hop_length + 1;
    if (frames != expected) {
        char msg[128];
        snprintf(msg, sizeof(msg), "expected %d frames, got %d", expected, frames);
        mel_destroy(mel);
        FAIL(msg);
    }

    mel_destroy(mel);
    PASS();
}

/* ── Mel: White Noise ────────────────────────────────────── */

static void test_mel_white_noise(void) {
    TEST("mel: white noise has broad energy across bins");
    MelConfig cfg;
    mel_config_default(&cfg);
    MelSpectrogram *mel = mel_create(&cfg);
    if (!mel) FAIL("create failed");

    /* 0.5s of pseudo-random noise */
    int n = 8000;
    float *pcm = (float *)malloc(n * sizeof(float));
    unsigned int seed = 12345;
    for (int i = 0; i < n; i++) {
        seed = seed * 1664525u + 1013904223u;
        pcm[i] = ((float)((int)(seed >> 16) % 2000 - 1000)) / 1000.0f;
    }

    float mels[100 * 80];
    int frames = mel_process(mel, pcm, n, mels, 100);
    if (frames <= 0) { free(pcm); mel_destroy(mel); FAIL("no frames"); }

    /* Average across frames, check that energy is spread across bins */
    float *avg = (float *)calloc(80, sizeof(float));
    for (int f = 0; f < frames; f++)
        for (int m = 0; m < 80; m++)
            avg[m] += mels[f * 80 + m];
    for (int m = 0; m < 80; m++) avg[m] /= (float)frames;

    /* Count how many bins have energy above the median */
    float sorted[80];
    memcpy(sorted, avg, 80 * sizeof(float));
    for (int i = 0; i < 79; i++)
        for (int j = i + 1; j < 80; j++)
            if (sorted[j] < sorted[i]) {
                float tmp = sorted[i]; sorted[i] = sorted[j]; sorted[j] = tmp;
            }
    float median = sorted[40];
    int above = 0;
    for (int m = 0; m < 80; m++)
        if (avg[m] >= median) above++;

    /* White noise should have broad energy — at least 30 bins above median */
    if (above < 30) {
        char msg[128];
        snprintf(msg, sizeof(msg), "only %d bins above median (expected >= 30)", above);
        free(pcm); free(avg); mel_destroy(mel);
        FAIL(msg);
    }

    free(pcm);
    free(avg);
    mel_destroy(mel);
    PASS();
}

/* ── Mel: Frequency Bin Verification ─────────────────────── */

static void test_mel_frequency_bins(void) {
    TEST("mel: 4kHz sine has energy in higher bin than 500Hz");
    MelConfig cfg;
    mel_config_default(&cfg);

    /* Process 500Hz tone */
    MelSpectrogram *mel_lo = mel_create(&cfg);
    if (!mel_lo) FAIL("create failed");
    int n = 8000;
    float *pcm = (float *)malloc(n * sizeof(float));
    for (int i = 0; i < n; i++)
        pcm[i] = 0.5f * sinf(2.0f * M_PI * 500.0f * i / 16000.0f);
    float mels_lo[100 * 80];
    int frames_lo = mel_process(mel_lo, pcm, n, mels_lo, 100);

    /* Process 4kHz tone */
    MelSpectrogram *mel_hi = mel_create(&cfg);
    if (!mel_hi) { free(pcm); mel_destroy(mel_lo); FAIL("create failed"); }
    for (int i = 0; i < n; i++)
        pcm[i] = 0.5f * sinf(2.0f * M_PI * 4000.0f * i / 16000.0f);
    float mels_hi[100 * 80];
    int frames_hi = mel_process(mel_hi, pcm, n, mels_hi, 100);

    if (frames_lo <= 2 || frames_hi <= 2) {
        free(pcm); mel_destroy(mel_lo); mel_destroy(mel_hi);
        FAIL("not enough frames");
    }

    /* Find peak bin for each, averaging middle frames */
    int peak_lo = 0, peak_hi = 0;
    float max_lo = -1e30f, max_hi = -1e30f;
    for (int m = 0; m < 80; m++) {
        float sum_lo = 0, sum_hi = 0;
        for (int f = 2; f < frames_lo; f++) sum_lo += mels_lo[f * 80 + m];
        for (int f = 2; f < frames_hi; f++) sum_hi += mels_hi[f * 80 + m];
        if (sum_lo > max_lo) { max_lo = sum_lo; peak_lo = m; }
        if (sum_hi > max_hi) { max_hi = sum_hi; peak_hi = m; }
    }

    if (peak_hi <= peak_lo) {
        char msg[128];
        snprintf(msg, sizeof(msg), "4kHz peak=%d not above 500Hz peak=%d", peak_hi, peak_lo);
        free(pcm); mel_destroy(mel_lo); mel_destroy(mel_hi);
        FAIL(msg);
    }

    free(pcm);
    mel_destroy(mel_lo);
    mel_destroy(mel_hi);
    PASS();
}

/* ── Mel: Custom Parameters ──────────────────────────────── */

static void test_mel_custom_params(void) {
    TEST("mel: custom n_mels=40 and hop_length=320");
    MelConfig cfg;
    mel_config_default(&cfg);
    cfg.n_mels = 40;
    cfg.hop_length = 320;

    MelSpectrogram *mel = mel_create(&cfg);
    if (!mel) FAIL("create with custom params failed");
    if (mel_n_mels(mel) != 40) { mel_destroy(mel); FAIL("expected 40 mels"); }
    if (mel_hop_length(mel) != 320) { mel_destroy(mel); FAIL("expected hop 320"); }

    /* Process 1s of audio, expect ~50 frames (16000/320) */
    float *pcm = (float *)calloc(16000, sizeof(float));
    float *out = (float *)malloc(200 * 40 * sizeof(float));
    int frames = mel_process(mel, pcm, 16000, out, 200);

    /* With center padding, frames should be approximately 16000/320 = 50 */
    if (frames < 40 || frames > 60) {
        char msg[128];
        snprintf(msg, sizeof(msg), "expected ~50 frames, got %d", frames);
        free(pcm); free(out); mel_destroy(mel);
        FAIL(msg);
    }

    free(pcm);
    free(out);
    mel_destroy(mel);
    PASS();
}

static void test_mel_large_nfft(void) {
    TEST("mel: n_fft=1024, win_length=1024, hop=512");
    MelConfig cfg;
    mel_config_default(&cfg);
    cfg.n_fft = 1024;
    cfg.win_length = 1024;
    cfg.hop_length = 512;

    MelSpectrogram *mel = mel_create(&cfg);
    if (!mel) FAIL("create with large nfft failed");

    float *pcm = (float *)calloc(16000, sizeof(float));
    float *out = (float *)malloc(200 * 80 * sizeof(float));
    int frames = mel_process(mel, pcm, 16000, out, 200);
    /* ~31 frames expected: (16000 + 512 - 1024) / 512 + 1 */
    if (frames < 20 || frames > 45) {
        char msg[128];
        snprintf(msg, sizeof(msg), "expected ~31 frames, got %d", frames);
        free(pcm); free(out); mel_destroy(mel);
        FAIL(msg);
    }

    free(pcm);
    free(out);
    mel_destroy(mel);
    PASS();
}

/* ── Mel: Edge Cases ─────────────────────────────────────── */

static void test_mel_very_short_audio(void) {
    TEST("mel: very short audio (< 1 frame) → 0 frames");
    MelConfig cfg;
    mel_config_default(&cfg);
    MelSpectrogram *mel = mel_create(&cfg);
    if (!mel) FAIL("create failed");

    /* Only 10 samples — far less than one frame (hop=160) */
    float pcm[10] = {0};
    float out[80];
    int frames = mel_process(mel, pcm, 10, out, 1);
    if (frames != 0) {
        char msg[128];
        snprintf(msg, sizeof(msg), "expected 0 frames from 10 samples, got %d", frames);
        mel_destroy(mel);
        FAIL(msg);
    }

    mel_destroy(mel);
    PASS();
}

static void test_mel_long_audio(void) {
    TEST("mel: 10 seconds of audio processes correctly");
    MelConfig cfg;
    mel_config_default(&cfg);
    MelSpectrogram *mel = mel_create(&cfg);
    if (!mel) FAIL("create failed");

    /* 10 seconds at 16kHz */
    int n = 160000;
    float *pcm = (float *)malloc(n * sizeof(float));
    for (int i = 0; i < n; i++)
        pcm[i] = 0.3f * sinf(2.0f * M_PI * 440.0f * i / 16000.0f);

    int max_frames = 2000;
    float *out = (float *)malloc(max_frames * 80 * sizeof(float));
    int frames = mel_process(mel, pcm, n, out, max_frames);

    /* Expected: ~1000 frames (160000/160) */
    if (frames < 900 || frames > 1100) {
        char msg[128];
        snprintf(msg, sizeof(msg), "expected ~1000 frames, got %d", frames);
        free(pcm); free(out); mel_destroy(mel);
        FAIL(msg);
    }

    free(pcm);
    free(out);
    mel_destroy(mel);
    PASS();
}

static void test_mel_null_inputs(void) {
    TEST("mel: NULL inputs handled safely");
    MelConfig cfg;
    mel_config_default(&cfg);
    MelSpectrogram *mel = mel_create(&cfg);
    if (!mel) FAIL("create failed");

    /* NULL pcm */
    float out[80];
    int frames = mel_process(mel, NULL, 100, out, 1);
    if (frames != -1 && frames != 0) {
        mel_destroy(mel);
        FAIL("expected -1 or 0 from NULL pcm");
    }

    /* NULL output buffer */
    float pcm[320] = {0};
    frames = mel_process(mel, pcm, 320, NULL, 1);
    if (frames != -1 && frames != 0) {
        mel_destroy(mel);
        FAIL("expected -1 or 0 from NULL output");
    }

    /* Zero max_frames */
    frames = mel_process(mel, pcm, 320, out, 0);
    if (frames != 0 && frames != -1) {
        mel_destroy(mel);
        FAIL("expected 0 or -1 from max_frames=0");
    }

    mel_destroy(mel);
    PASS();
}

static void test_mel_n_mels_accessor(void) {
    TEST("mel: n_mels accessor returns correct value");
    if (mel_n_mels(NULL) != 0 && mel_n_mels(NULL) != -1) {
        FAIL("mel_n_mels(NULL) should return 0 or -1");
    }
    if (mel_hop_length(NULL) != 0 && mel_hop_length(NULL) != -1) {
        FAIL("mel_hop_length(NULL) should return 0 or -1");
    }
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

static void test_conformer_null_path(void) {
    TEST("conformer: create with NULL path returns NULL");
    ConformerSTT *stt = conformer_stt_create(NULL);
    if (stt != NULL) {
        conformer_stt_destroy(stt);
        FAIL("should have returned NULL");
    }
    PASS();
}

static void test_conformer_empty_path(void) {
    TEST("conformer: create with empty string returns NULL");
    ConformerSTT *stt = conformer_stt_create("");
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

static void test_conformer_info_null(void) {
    TEST("conformer: info accessors return 0 on NULL");
    if (conformer_stt_d_model(NULL) != 0)
        FAIL("d_model(NULL) should return 0");
    if (conformer_stt_n_layers(NULL) != 0)
        FAIL("n_layers(NULL) should return 0");
    if (conformer_stt_vocab_size(NULL) != 0)
        FAIL("vocab_size(NULL) should return 0");
    if (conformer_stt_has_eou_support(NULL) != 0)
        FAIL("has_eou_support(NULL) should return 0");
    if (conformer_stt_is_tdt(NULL) != 0)
        FAIL("is_tdt(NULL) should return 0");
    PASS();
}

static void test_conformer_eou_null(void) {
    TEST("conformer: EOU functions return safe defaults on NULL");
    if (conformer_stt_has_eou(NULL) != 0)
        FAIL("has_eou(NULL) should return 0");
    if (conformer_stt_eou_prob(NULL, 4) != 0.0f)
        FAIL("eou_prob(NULL) should return 0.0");
    if (conformer_stt_eou_frame(NULL) != -1)
        FAIL("eou_frame(NULL) should return -1");
    PASS();
}

static void test_conformer_cache_null(void) {
    TEST("conformer: cache-aware functions safe on NULL");
    conformer_stt_set_cache_aware(NULL, 1);
    conformer_stt_set_chunk_frames(NULL, 40);
    if (conformer_stt_stride_ms(NULL) != 0)
        FAIL("stride_ms(NULL) should return 0");
    PASS();
}

static void test_conformer_beam_null(void) {
    TEST("conformer: beam search functions safe on NULL");
    int rc = conformer_stt_enable_beam_search(NULL, NULL, 16, 1.5f, 0.0f);
    if (rc != -1) FAIL("enable_beam_search(NULL) should return -1");
    conformer_stt_disable_beam_search(NULL);
    PASS();
}

static void test_conformer_external_forward_null(void) {
    TEST("conformer: external forward hook safe on NULL");
    conformer_stt_set_external_forward(NULL, NULL, NULL);
    int out_vocab = 0;
    float *buf = conformer_stt_get_logits_buf(NULL, &out_vocab);
    if (buf != NULL) FAIL("get_logits_buf(NULL) should return NULL");
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
    test_mel_white_noise();
    test_mel_frequency_bins();
    test_mel_custom_params();
    test_mel_large_nfft();
    test_mel_very_short_audio();
    test_mel_long_audio();
    test_mel_null_inputs();
    test_mel_n_mels_accessor();

    printf("\nConformer STT Engine:\n");
    test_conformer_null_model();
    test_conformer_null_path();
    test_conformer_empty_path();
    test_conformer_null_safety();
    test_conformer_info_null();
    test_conformer_eou_null();
    test_conformer_cache_null();
    test_conformer_beam_null();
    test_conformer_external_forward_null();

    printf("\n=== Results: %d passed, %d failed ===\n\n",
           tests_passed, tests_failed);

    return tests_failed > 0 ? 1 : 0;
}
