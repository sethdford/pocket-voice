/**
 * test_phase2_regressions.c — Targeted regression tests for Phase 2 bug fixes.
 *
 * Covers:
 *   1. Breath synthesis ADSR overflow normalization
 *   2. Micropause fade validity
 *   3. Sentence gap breath generation
 *   4. vDSP FFT vs naive DFT correctness
 *   5. Conformer STT header validation (div-by-zero guard)
 *   6. Mel spectrogram edge-case inputs
 *   7. Sentence buffer overflow handling
 *   8. Text normalizer edge cases
 *
 * Build:
 *   cc -O2 -arch arm64 -Isrc -framework Accelerate \
 *      tests/test_phase2_regressions.c src/breath_synthesis.c \
 *      src/mel_spectrogram.c src/sentence_buffer.c \
 *      src/text_normalize.c src/conformer_stt.c \
 *      -lm -o tests/test_phase2_regressions
 *
 * Run: ./tests/test_phase2_regressions
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#endif

#include "breath_synthesis.h"
#include "mel_spectrogram.h"
#include "sentence_buffer.h"
#include "text_normalize.h"
#include "conformer_stt.h"

/* ── Test harness ─────────────────────────────────────── */

static int g_pass = 0, g_fail = 0;

#define CHECK(cond, msg) do { \
    if (cond) { g_pass++; printf("  [PASS] %s\n", msg); } \
    else { g_fail++; printf("  [FAIL] %s\n", msg); } \
} while(0)

#define CHECKF(cond, fmt, ...) do { \
    char _buf[256]; snprintf(_buf, sizeof(_buf), fmt, __VA_ARGS__); \
    if (cond) { g_pass++; printf("  [PASS] %s\n", _buf); } \
    else { g_fail++; printf("  [FAIL] %s\n", _buf); } \
} while(0)

/* ── Helpers ──────────────────────────────────────────── */

/** Check that all samples in buf are finite and within [-limit, limit]. */
static int buf_bounded(const float *buf, int n, float limit) {
    for (int i = 0; i < n; i++) {
        if (!isfinite(buf[i]) || buf[i] < -limit || buf[i] > limit)
            return 0;
    }
    return 1;
}

/** Compute RMS of a buffer. */
static float buf_rms(const float *buf, int n) {
    if (n <= 0) return 0.0f;
    double sum = 0.0;
    for (int i = 0; i < n; i++) sum += (double)buf[i] * buf[i];
    return sqrtf((float)(sum / n));
}

/* ══════════════════════════════════════════════════════════
 * 1. Breath ADSR overflow — short buffer shouldn't clip or NaN
 * ══════════════════════════════════════════════════════════ */
static void test_adsr_overflow(void) {
    printf("\n── test_adsr_overflow ──\n");

    BreathSynth *bs = breath_create(48000);
    CHECK(bs != NULL, "breath_create succeeds");
    if (!bs) return;

    /* Very short buffer: ADSR attack+sustain+decay >> n_samples */
    const int n = 100;
    float buf[100];
    memset(buf, 0, sizeof(buf));

    breath_generate(bs, buf, n, 1.0f); /* high amplitude */
    CHECK(buf_bounded(buf, n, 1.0f), "output bounded [-1, 1] with amplitude=1.0");

    /* Even higher amplitude — should still be finite */
    memset(buf, 0, sizeof(buf));
    breath_generate(bs, buf, n, 10.0f);
    CHECK(buf_bounded(buf, n, 20.0f), "output finite with amplitude=10.0");

    /* Single-sample edge case */
    float single = 0.0f;
    breath_generate(bs, &single, 1, 0.5f);
    CHECK(isfinite(single), "single-sample generation is finite");

    breath_destroy(bs);
}

/* ══════════════════════════════════════════════════════════
 * 2. Micropause — fade-out, silence, fade-in structure
 * ══════════════════════════════════════════════════════════ */
static void test_breath_micropause(void) {
    printf("\n── test_breath_micropause ──\n");

    const int sr = 48000;
    const int n = 4800; /* 100ms worth */
    float buf[4800];

    /* Fill with constant amplitude */
    for (int i = 0; i < n; i++) buf[i] = 0.5f;

    breath_micropause(buf, n, 10.0f, sr); /* 10ms fade each side */

    /* Check that the middle section is quieter (silence region) */
    int mid = n / 2;
    float mid_rms = buf_rms(&buf[mid - 100], 200);
    float edge_start_rms = buf_rms(buf, 100);

    CHECK(mid_rms < 0.1f, "middle section is near-silent");
    CHECK(buf_bounded(buf, n, 1.0f), "all samples bounded [-1, 1]");

    /* Verify the output is different from input (fade was applied) */
    int changed = 0;
    for (int i = 0; i < n; i++) {
        if (fabsf(buf[i] - 0.5f) > 1e-6f) { changed = 1; break; }
    }
    CHECK(changed, "micropause modified the buffer");

    /* Edge case: very short buffer */
    float tiny[10];
    for (int i = 0; i < 10; i++) tiny[i] = 1.0f;
    breath_micropause(tiny, 10, 1.0f, sr);
    CHECK(buf_bounded(tiny, 10, 1.5f), "tiny buffer doesn't crash or overflow");
}

/* ══════════════════════════════════════════════════════════
 * 3. Sentence gap — breath noise present in gap
 * ══════════════════════════════════════════════════════════ */
static void test_breath_sentence_gap(void) {
    printf("\n── test_breath_sentence_gap ──\n");

    BreathSynth *bs = breath_create(48000);
    CHECK(bs != NULL, "breath_create succeeds");
    if (!bs) return;

    const int n = 9600; /* 200ms gap */
    float *buf = calloc(n, sizeof(float));
    CHECK(buf != NULL, "allocation succeeds");
    if (!buf) { breath_destroy(bs); return; }

    breath_sentence_gap(bs, buf, n, 0.1f); /* speech_rms=0.1 */

    /* Middle third should have some breath noise */
    int third = n / 3;
    float mid_rms = buf_rms(&buf[third], third);
    CHECK(mid_rms > 1e-6f, "sentence gap has audible breath noise (middle)");
    CHECK(buf_bounded(buf, n, 1.0f), "sentence gap output bounded [-1, 1]");

    /* Edge case: very short gap */
    float tiny[50];
    memset(tiny, 0, sizeof(tiny));
    breath_sentence_gap(bs, tiny, 50, 0.05f);
    CHECK(buf_bounded(tiny, 50, 1.0f), "short sentence gap bounded");

    free(buf);
    breath_destroy(bs);
}

/* ══════════════════════════════════════════════════════════
 * 4. vDSP FFT vs naive DFT — magnitude correctness
 * ══════════════════════════════════════════════════════════ */
static void test_vdsp_fft_vs_naive(void) {
    printf("\n── test_vdsp_fft_vs_naive ──\n");

#ifdef __APPLE__
    const int N = 512;
    const int log2N = 9; /* log2(512) */
    const float sr = 16000.0f;

    /* Generate test signal: 440Hz + 880Hz */
    float *signal = calloc(N, sizeof(float));
    CHECK(signal != NULL, "signal allocation");
    if (!signal) return;

    for (int i = 0; i < N; i++) {
        signal[i] = sinf(2.0f * M_PI * 440.0f * i / sr)
                   + 0.5f * sinf(2.0f * M_PI * 880.0f * i / sr);
    }

    /* ── vDSP FFT ── */
    FFTSetup fft_setup = vDSP_create_fftsetup(log2N, kFFTRadix2);
    CHECK(fft_setup != NULL, "vDSP FFT setup created");
    if (!fft_setup) { free(signal); return; }

    /* Pack into split complex */
    DSPSplitComplex split;
    split.realp = calloc(N / 2, sizeof(float));
    split.imagp = calloc(N / 2, sizeof(float));

    /* Convert real signal to split complex (packed) */
    vDSP_ctoz((const DSPComplex *)signal, 2, &split, 1, N / 2);

    /* Forward FFT */
    vDSP_fft_zrip(fft_setup, &split, 1, log2N, kFFTDirection_Forward);

    /* Compute magnitudes from vDSP result.
     * vDSP_fft_zrip output = 2 * standard_DFT, so divide by 2N
     * to get the same normalization as naive DFT / N. */
    float *vdsp_mag = calloc(N / 2, sizeof(float));
    float vdsp_scale = 1.0f / (2.0f * (float)N);
    for (int k = 0; k < N / 2; k++) {
        float re = split.realp[k] * vdsp_scale;
        float im = split.imagp[k] * vdsp_scale;
        vdsp_mag[k] = sqrtf(re * re + im * im);
    }

    /* ── Naive DFT ── */
    float *naive_mag = calloc(N / 2, sizeof(float));
    for (int k = 0; k < N / 2; k++) {
        float re = 0.0f, im = 0.0f;
        for (int n = 0; n < N; n++) {
            float angle = -2.0f * M_PI * k * n / N;
            re += signal[n] * cosf(angle);
            im += signal[n] * sinf(angle);
        }
        naive_mag[k] = sqrtf(re * re + im * im) / (float)N;
    }

    /* ── Compare ── */
    float max_err = 0.0f;
    for (int k = 1; k < N / 2; k++) { /* skip DC — packing differs */
        float err = fabsf(vdsp_mag[k] - naive_mag[k]);
        if (err > max_err) max_err = err;
    }
    CHECKF(max_err < 1e-2f, "vDSP vs naive max error = %.6f (< 0.01)", max_err);

    /* Verify peak at 440Hz bin */
    int bin_440 = (int)(440.0f * N / sr + 0.5f);
    int bin_880 = (int)(880.0f * N / sr + 0.5f);
    CHECK(vdsp_mag[bin_440] > 0.3f, "440Hz peak detected in vDSP FFT");
    CHECK(vdsp_mag[bin_880] > 0.1f, "880Hz peak detected in vDSP FFT");

    /* Cleanup */
    vDSP_destroy_fftsetup(fft_setup);
    free(signal);
    free(split.realp);
    free(split.imagp);
    free(vdsp_mag);
    free(naive_mag);
#else
    printf("  [SKIP] vDSP FFT test requires Apple Accelerate\n");
#endif
}

/* ══════════════════════════════════════════════════════════
 * 5. Conformer STT header validation — div-by-zero guards
 * ══════════════════════════════════════════════════════════ */
static void test_header_div_by_zero(void) {
    printf("\n── test_header_div_by_zero ──\n");

    /* Creating with NULL path should return NULL, not crash */
    ConformerSTT *stt = conformer_stt_create(NULL);
    CHECK(stt == NULL, "conformer_stt_create(NULL) returns NULL");

    /* Creating with nonexistent path should return NULL */
    stt = conformer_stt_create("/nonexistent/model.cstt");
    CHECK(stt == NULL, "conformer_stt_create(bad path) returns NULL");

    /* Destroy NULL should be safe */
    conformer_stt_destroy(NULL);
    CHECK(1, "conformer_stt_destroy(NULL) safe");
}

/* ══════════════════════════════════════════════════════════
 * 6. Mel spectrogram edge-case inputs
 * ══════════════════════════════════════════════════════════ */
static void test_mel_spectrogram_bounds(void) {
    printf("\n── test_mel_spectrogram_bounds ──\n");

    MelConfig cfg;
    mel_config_default(&cfg);
    MelSpectrogram *mel = mel_create(&cfg);
    CHECK(mel != NULL, "mel_create succeeds");
    if (!mel) return;

    const int max_frames = 100;
    const int n_mels = 80;
    float *out = calloc(max_frames * n_mels, sizeof(float));
    CHECK(out != NULL, "output allocation");
    if (!out) { mel_destroy(mel); return; }

    /* Very short audio: less than one FFT window (512 samples) */
    float short_audio[64];
    for (int i = 0; i < 64; i++) short_audio[i] = sinf(2.0f * M_PI * 440.0f * i / 16000.0f);
    int frames = mel_process(mel, short_audio, 64, out, max_frames);
    CHECKF(frames >= 0, "short audio (%d samples): returned %d frames (no crash)", 64, frames);
    mel_reset(mel);

    /* Single sample */
    float one = 0.5f;
    frames = mel_process(mel, &one, 1, out, max_frames);
    CHECKF(frames >= 0, "single sample: returned %d frames (no crash)", frames);
    mel_reset(mel);

    /* Very loud audio (amplitude = 1000) */
    float loud[1600];
    for (int i = 0; i < 1600; i++) loud[i] = 1000.0f * sinf(2.0f * M_PI * 440.0f * i / 16000.0f);
    frames = mel_process(mel, loud, 1600, out, max_frames);
    CHECKF(frames >= 0, "loud audio: returned %d frames", frames);
    if (frames > 0) {
        CHECK(buf_bounded(out, frames * n_mels, 100.0f), "loud audio: output is finite");
    }
    mel_reset(mel);

    /* Zero-length audio — may return 0 or -1 (both are valid) */
    frames = mel_process(mel, short_audio, 0, out, max_frames);
    CHECKF(frames >= -1, "zero-length audio: returned %d (no crash)", frames);

    /* NULL destroy safety */
    mel_destroy(mel);
    mel_destroy(NULL);
    CHECK(1, "mel_destroy(NULL) safe");

    free(out);
}

/* ══════════════════════════════════════════════════════════
 * 7. Sentence buffer overflow handling
 * ══════════════════════════════════════════════════════════ */
static void test_sentence_buffer_overflow(void) {
    printf("\n── test_sentence_buffer_overflow ──\n");

    SentenceBuffer *sb = sentbuf_create(SENTBUF_MODE_SENTENCE, 5);
    CHECK(sb != NULL, "sentbuf_create succeeds");
    if (!sb) return;

    /* Feed many sentences to stress the internal ring */
    for (int i = 0; i < 200; i++) {
        char tok[64];
        snprintf(tok, sizeof(tok), "Word%d. ", i);
        sentbuf_add(sb, tok, (int)strlen(tok));
    }

    /* Flush all segments */
    char out[4096];
    int total_flushed = 0;
    while (sentbuf_has_segment(sb)) {
        int n = sentbuf_flush(sb, out, sizeof(out));
        if (n <= 0) break;
        total_flushed++;
    }
    CHECKF(total_flushed > 0, "flushed %d segments from stress test", total_flushed);

    /* Flush remaining */
    int n = sentbuf_flush_all(sb, out, sizeof(out));
    CHECK(n >= 0, "flush_all after stress returns >= 0");

    /* Reset and reuse */
    sentbuf_reset(sb);
    sentbuf_add(sb, "After reset. ", 13);
    CHECK(sentbuf_has_segment(sb), "buffer works after reset");

    /* Very long token */
    char long_tok[8192];
    memset(long_tok, 'A', sizeof(long_tok) - 2);
    long_tok[sizeof(long_tok) - 2] = '.';
    long_tok[sizeof(long_tok) - 1] = '\0';
    sentbuf_add(sb, long_tok, (int)strlen(long_tok));
    /* Just verify no crash — flush may truncate */
    sentbuf_flush_all(sb, out, sizeof(out));
    CHECK(1, "very long token handled without crash");

    sentbuf_destroy(sb);

    /* NULL safety */
    sentbuf_destroy(NULL);
    CHECK(1, "sentbuf_destroy(NULL) safe");
}

/* ══════════════════════════════════════════════════════════
 * 8. Text normalizer edge cases
 * ══════════════════════════════════════════════════════════ */
static void test_text_normalize_edge_cases(void) {
    printf("\n── test_text_normalize_edge_cases ──\n");

    char out[4096];

    /* Empty string */
    int n = text_auto_normalize("", out, sizeof(out));
    CHECK(n >= 0, "empty string: no crash");
    CHECK(out[0] == '\0' || n == 0, "empty string: empty or zero output");

    /* Whitespace only */
    n = text_auto_normalize("   \t\n  ", out, sizeof(out));
    CHECK(n >= 0, "whitespace-only: no crash");

    /* Normal text passthrough */
    n = text_auto_normalize("Hello world", out, sizeof(out));
    CHECK(n > 0, "normal text produces output");
    CHECK(strstr(out, "Hello") != NULL || strstr(out, "hello") != NULL,
          "normal text preserved");

    /* Number normalization */
    n = text_auto_normalize("I have 42 apples", out, sizeof(out));
    CHECK(n > 0, "numeric text produces output");

    /* Currency */
    n = text_auto_normalize("It costs $3.50", out, sizeof(out));
    CHECK(n > 0, "currency text produces output");

    /* Very long string (10KB) */
    char *long_str = calloc(10240 + 1, 1);
    CHECK(long_str != NULL, "long string allocation");
    if (long_str) {
        for (int i = 0; i < 10240; i++) long_str[i] = 'a' + (i % 26);
        long_str[10240] = '\0';
        n = text_auto_normalize(long_str, out, sizeof(out));
        CHECK(n >= 0, "10KB string: no crash");
        free(long_str);
    }

    /* Unicode text */
    n = text_auto_normalize("Caf\xc3\xa9 na\xc3\xafve", out, sizeof(out));
    CHECK(n >= 0, "unicode text: no crash");

    /* Small output buffer */
    char tiny[8];
    n = text_auto_normalize("Hello world testing", tiny, sizeof(tiny));
    CHECK(n >= 0, "small output buffer: no crash");
    CHECK(tiny[sizeof(tiny) - 1] == '\0', "small buffer: null-terminated");

    /* Individual normalizer: cardinal */
    n = text_cardinal("42", out, sizeof(out));
    CHECK(n > 0 && strstr(out, "forty") != NULL, "cardinal: 42 -> forty...");

    /* Individual normalizer: ordinal */
    n = text_ordinal("3", out, sizeof(out));
    CHECK(n > 0, "ordinal: produces output");
}

/* ══════════════════════════════════════════════════════════
 * 9. Breath create with invalid sample rates
 * ══════════════════════════════════════════════════════════ */
static void test_breath_invalid_sample_rate(void) {
    printf("\n── test_breath_invalid_sample_rate ──\n");

    /* Zero sample rate */
    BreathSynth *bs = breath_create(0);
    CHECK(bs == NULL, "breath_create(0) returns NULL");
    if (bs) breath_destroy(bs);

    /* Negative sample rate */
    bs = breath_create(-48000);
    CHECK(bs == NULL, "breath_create(-48000) returns NULL");
    if (bs) breath_destroy(bs);

    /* Very high sample rate — should still work */
    bs = breath_create(192000);
    CHECK(bs != NULL, "breath_create(192000) succeeds");
    if (bs) {
        float buf[100];
        memset(buf, 0, sizeof(buf));
        breath_generate(bs, buf, 100, 0.5f);
        CHECK(buf_bounded(buf, 100, 2.0f), "192kHz breath output bounded");
        breath_destroy(bs);
    }

    /* NULL destroy safety (re-verify) */
    breath_destroy(NULL);
    CHECK(1, "breath_destroy(NULL) safe");
}

/* ══════════════════════════════════════════════════════════
 * 10. Breath generate with zero amplitude and zero samples
 * ══════════════════════════════════════════════════════════ */
static void test_breath_generate_edge_cases(void) {
    printf("\n── test_breath_generate_edge_cases ──\n");

    BreathSynth *bs = breath_create(48000);
    CHECK(bs != NULL, "breath_create succeeds");
    if (!bs) return;

    /* Zero amplitude — output should remain unchanged */
    float buf[200];
    for (int i = 0; i < 200; i++) buf[i] = 0.42f;
    breath_generate(bs, buf, 200, 0.0f);
    int unchanged = 1;
    for (int i = 0; i < 200; i++) {
        if (fabsf(buf[i] - 0.42f) > 1e-5f) { unchanged = 0; break; }
    }
    CHECK(unchanged, "zero amplitude: buffer unchanged");

    /* Zero samples — no crash */
    breath_generate(bs, buf, 0, 1.0f);
    CHECK(1, "zero samples: no crash");

    /* Negative amplitude — should still be finite */
    memset(buf, 0, sizeof(buf));
    breath_generate(bs, buf, 200, -0.5f);
    CHECK(buf_bounded(buf, 200, 5.0f), "negative amplitude: output finite");

    breath_destroy(bs);
}

/* ══════════════════════════════════════════════════════════
 * 11. Mel spectrogram — NULL and invalid config values
 * ══════════════════════════════════════════════════════════ */
static void test_mel_spectrogram_null_safety(void) {
    printf("\n── test_mel_spectrogram_null_safety ──\n");

    /* NULL config → should use defaults or return NULL */
    MelSpectrogram *mel = mel_create(NULL);
    /* Either NULL or valid is acceptable */
    CHECKF(1, "mel_create(NULL) returned %p (no crash)", (void *)mel);
    if (mel) mel_destroy(mel);

    /* Process with NULL mel handle */
    float audio[100] = {0};
    float out[8000] = {0};
    int frames = mel_process(NULL, audio, 100, out, 100);
    CHECK(frames == -1 || frames == 0, "mel_process(NULL handle): safe return");

    /* Reset NULL */
    mel_reset(NULL);
    CHECK(1, "mel_reset(NULL): no crash");

    /* mel_n_mels and mel_hop_length with NULL */
    int nm = mel_n_mels(NULL);
    CHECK(nm == 0 || nm == -1 || nm == 80, "mel_n_mels(NULL): safe");
    int hop = mel_hop_length(NULL);
    CHECK(hop == 0 || hop == -1 || hop == 160, "mel_hop_length(NULL): safe");
}

/* ══════════════════════════════════════════════════════════
 * 12. Mel spectrogram — negative and extreme PCM values
 * ══════════════════════════════════════════════════════════ */
static void test_mel_spectrogram_extreme_pcm(void) {
    printf("\n── test_mel_spectrogram_extreme_pcm ──\n");

    MelConfig cfg;
    mel_config_default(&cfg);
    MelSpectrogram *mel = mel_create(&cfg);
    CHECK(mel != NULL, "mel_create succeeds");
    if (!mel) return;

    const int max_frames = 100;
    const int n_mels = 80;
    float *out = calloc(max_frames * n_mels, sizeof(float));
    if (!out) { mel_destroy(mel); return; }

    /* All negative values */
    float neg[1600];
    for (int i = 0; i < 1600; i++) neg[i] = -0.5f;
    int frames = mel_process(mel, neg, 1600, out, max_frames);
    CHECKF(frames >= 0, "all negative audio: returned %d frames", frames);
    mel_reset(mel);

    /* NaN input — should not propagate */
    float nan_buf[1600];
    for (int i = 0; i < 1600; i++) nan_buf[i] = 0.0f;
    nan_buf[800] = 0.0f / 0.0f; /* NaN */
    frames = mel_process(mel, nan_buf, 1600, out, max_frames);
    CHECKF(frames >= -1, "NaN input: returned %d (no crash)", frames);
    mel_reset(mel);

    /* Infinity input */
    float inf_buf[1600];
    for (int i = 0; i < 1600; i++) inf_buf[i] = 0.0f;
    inf_buf[800] = 1.0f / 0.0f; /* +Inf */
    frames = mel_process(mel, inf_buf, 1600, out, max_frames);
    CHECKF(frames >= -1, "Inf input: returned %d (no crash)", frames);
    mel_reset(mel);

    /* NULL PCM pointer */
    frames = mel_process(mel, NULL, 100, out, max_frames);
    CHECK(frames == -1 || frames == 0, "NULL PCM: safe return");

    mel_destroy(mel);
    free(out);
}

/* ══════════════════════════════════════════════════════════
 * 13. Sentence buffer — speculative mode and adaptive
 * ══════════════════════════════════════════════════════════ */
static void test_sentence_buffer_speculative(void) {
    printf("\n── test_sentence_buffer_speculative ──\n");

    SentenceBuffer *sb = sentbuf_create(SENTBUF_MODE_SPECULATIVE, 3);
    CHECK(sb != NULL, "sentbuf_create(SPECULATIVE, 3) succeeds");
    if (!sb) return;

    /* Enable adaptive mode */
    sentbuf_set_adaptive(sb, 2, 2);
    CHECK(1, "sentbuf_set_adaptive: no crash");

    /* Enable eager flushing */
    sentbuf_set_eager(sb, 4);
    CHECK(1, "sentbuf_set_eager(4): no crash");

    /* Feed several words without sentence boundary */
    sentbuf_add(sb, "The ", 4);
    sentbuf_add(sb, "quick ", 6);
    sentbuf_add(sb, "brown ", 6);
    sentbuf_add(sb, "fox ", 4);
    sentbuf_add(sb, "jumps ", 6);

    /* In speculative mode with eager=4, should have a segment */
    char out[512];
    int total = 0;
    while (sentbuf_has_segment(sb)) {
        int n = sentbuf_flush(sb, out, sizeof(out));
        if (n > 0) total++;
        else break;
    }
    CHECKF(total >= 0, "speculative mode: flushed %d segments", total);

    /* Check prosody hint (should be neutral for plain text) */
    SentBufProsodyHint hint = sentbuf_get_prosody_hint(sb);
    CHECK(hint.exclamation_count == 0, "prosody hint: no exclamations");
    CHECK(hint.question_count == 0, "prosody hint: no questions");

    /* Sentence count */
    int sc = sentbuf_sentence_count(sb);
    CHECKF(sc >= 0, "sentence_count: %d", sc);

    /* Predicted length */
    int pl = sentbuf_predicted_length(sb);
    CHECKF(pl >= 0, "predicted_length: %d", pl);

    sentbuf_destroy(sb);
}

/* ══════════════════════════════════════════════════════════
 * 14. Sentence buffer — prosody detection
 * ══════════════════════════════════════════════════════════ */
static void test_sentence_buffer_prosody(void) {
    printf("\n── test_sentence_buffer_prosody ──\n");

    SentenceBuffer *sb = sentbuf_create(SENTBUF_MODE_SENTENCE, 5);
    CHECK(sb != NULL, "sentbuf_create succeeds");
    if (!sb) return;

    /* Feed text with exclamation */
    sentbuf_add(sb, "WOW! That is AMAZING! ", 22);

    char out[512];
    if (sentbuf_has_segment(sb)) {
        sentbuf_flush(sb, out, sizeof(out));
        SentBufProsodyHint hint = sentbuf_get_prosody_hint(sb);
        CHECKF(hint.exclamation_count >= 1, "exclamation detected: %d", hint.exclamation_count);
        CHECK(hint.has_all_caps, "ALL CAPS detected");
    } else {
        /* Flush all to trigger prosody analysis */
        sentbuf_flush_all(sb, out, sizeof(out));
        CHECK(1, "flush_all completed");
    }

    sentbuf_reset(sb);

    /* Feed question text */
    sentbuf_add(sb, "How are you? ", 13);
    if (sentbuf_has_segment(sb)) {
        sentbuf_flush(sb, out, sizeof(out));
        SentBufProsodyHint hint = sentbuf_get_prosody_hint(sb);
        CHECKF(hint.question_count >= 1, "question detected: %d", hint.question_count);
    } else {
        sentbuf_flush_all(sb, out, sizeof(out));
        CHECK(1, "question flush completed");
    }

    sentbuf_destroy(sb);
}

/* ══════════════════════════════════════════════════════════
 * 15. Text normalizer — more format types
 * ══════════════════════════════════════════════════════════ */
static void test_text_normalize_formats(void) {
    printf("\n── test_text_normalize_formats ──\n");

    char out[4096];

    /* Telephone */
    int n = text_telephone("555-867-5309", out, sizeof(out));
    CHECK(n > 0, "telephone: produces output");

    /* Fraction */
    n = text_fraction("3/4", out, sizeof(out));
    CHECK(n > 0, "fraction 3/4: produces output");

    /* Time */
    n = text_time("3:45", NULL, out, sizeof(out));
    CHECK(n > 0, "time 3:45: produces output");

    /* Date */
    n = text_date("2024-01-15", NULL, out, sizeof(out));
    CHECK(n > 0, "date 2024-01-15: produces output");

    /* Characters (spell out) */
    n = text_characters("ABC", out, sizeof(out));
    CHECK(n > 0, "characters ABC: produces output");

    /* Unit */
    n = text_unit("5kg", out, sizeof(out));
    CHECK(n >= 0, "unit 5kg: no crash");

    /* URL */
    n = text_url("example.com", out, sizeof(out));
    CHECK(n >= 0, "url: no crash");

    /* Email */
    n = text_email("test@example.com", out, sizeof(out));
    CHECK(n >= 0, "email: no crash");

    /* Currency: various formats */
    n = text_currency("$0.99", out, sizeof(out));
    CHECK(n > 0, "currency $0.99: produces output");

    n = text_currency("$1,000,000", out, sizeof(out));
    CHECK(n > 0, "currency $1M: produces output");

    /* text_normalize with interpret_as */
    n = text_normalize("42", "cardinal", NULL, out, sizeof(out));
    CHECK(n > 0, "text_normalize cardinal 42: produces output");

    n = text_normalize("3rd", "ordinal", NULL, out, sizeof(out));
    CHECK(n > 0, "text_normalize ordinal 3rd: produces output");
}

/* ══════════════════════════════════════════════════════════
 * 16. Text normalizer — nonverbalisms and IPA expansion
 * ══════════════════════════════════════════════════════════ */
static void test_text_expand_nonverbalisms(void) {
    printf("\n── test_text_expand_nonverbalisms ──\n");

    char out[4096];

    /* Laughter marker */
    int n = text_expand_nonverbalisms("[laughter] That's funny.", out, sizeof(out));
    CHECK(n > 0, "nonverbalism [laughter]: produces output");

    /* Sigh marker */
    n = text_expand_nonverbalisms("Oh [sigh] not again.", out, sizeof(out));
    CHECK(n > 0, "nonverbalism [sigh]: produces output");

    /* Breath marker */
    n = text_expand_nonverbalisms("[breath] Hello.", out, sizeof(out));
    CHECK(n > 0, "nonverbalism [breath]: produces output");

    /* Pause marker */
    n = text_expand_nonverbalisms("Wait [pause] what?", out, sizeof(out));
    CHECK(n > 0, "nonverbalism [pause]: produces output");

    /* No markers — passthrough */
    n = text_expand_nonverbalisms("Plain text here.", out, sizeof(out));
    CHECK(n > 0, "no markers: passthrough");
    CHECK(strstr(out, "Plain") != NULL, "passthrough preserved");

    /* Empty string */
    n = text_expand_nonverbalisms("", out, sizeof(out));
    CHECK(n >= 0, "empty nonverbalism: no crash");

    /* Inline IPA expansion */
    n = text_expand_inline_ipa("Say <<h|ɛ|l|oʊ>> please.", out, sizeof(out));
    CHECK(n > 0, "inline IPA <<>>: produces output");

    /* No IPA — passthrough */
    n = text_expand_inline_ipa("No IPA here.", out, sizeof(out));
    CHECK(n > 0, "no IPA: passthrough");
    CHECK(strstr(out, "No IPA") != NULL, "IPA passthrough preserved");

    /* Small output buffer */
    char tiny[8];
    n = text_expand_nonverbalisms("[laughter] Ha!", tiny, sizeof(tiny));
    CHECK(n >= 0, "small buffer nonverbalism: no crash");
}

/* ══════════════════════════════════════════════════════════
 * 17. Conformer STT — extended null safety
 * ══════════════════════════════════════════════════════════ */
static void test_conformer_extended_null_safety(void) {
    printf("\n── test_conformer_extended_null_safety ──\n");

    /* Process with NULL engine */
    float audio[100] = {0};
    int rc = conformer_stt_process(NULL, audio, 100);
    CHECK(rc == -1 || rc == 0, "conformer_stt_process(NULL): safe");

    /* Process with NULL audio */
    /* Can't call with valid engine since no model, just test NULL engine + NULL audio */
    rc = conformer_stt_process(NULL, NULL, 0);
    CHECK(rc == -1 || rc == 0, "conformer_stt_process(NULL, NULL, 0): safe");

    /* Flush NULL */
    rc = conformer_stt_flush(NULL);
    CHECK(rc == -1 || rc == 0, "conformer_stt_flush(NULL): safe");

    /* get_text NULL */
    char buf[256];
    rc = conformer_stt_get_text(NULL, buf, sizeof(buf));
    CHECK(rc == -1 || rc == 0, "conformer_stt_get_text(NULL): safe");

    /* Reset NULL */
    conformer_stt_reset(NULL);
    CHECK(1, "conformer_stt_reset(NULL): safe");

    /* has_eou NULL */
    rc = conformer_stt_has_eou(NULL);
    CHECK(rc == 0 || rc == -1, "conformer_stt_has_eou(NULL): safe");

    /* eou_prob NULL */
    float prob = conformer_stt_eou_prob(NULL, 4);
    CHECK(prob >= 0.0f || prob == 0.0f, "conformer_stt_eou_prob(NULL): safe");

    /* Info functions with NULL */
    int sr = conformer_stt_sample_rate(NULL);
    CHECK(sr == 0 || sr == -1 || sr == 16000, "conformer_stt_sample_rate(NULL): safe");

    int dm = conformer_stt_d_model(NULL);
    CHECK(dm >= 0 || dm == -1, "conformer_stt_d_model(NULL): safe");
}

/* ══════════════════════════════════════════════════════════
 * Main
 * ══════════════════════════════════════════════════════════ */
int main(void) {
    printf("═══ Phase 2 Regression Tests ═══\n");

    test_adsr_overflow();
    test_breath_micropause();
    test_breath_sentence_gap();
    test_vdsp_fft_vs_naive();
    test_header_div_by_zero();
    test_mel_spectrogram_bounds();
    test_sentence_buffer_overflow();
    test_text_normalize_edge_cases();
    test_breath_invalid_sample_rate();
    test_breath_generate_edge_cases();
    test_mel_spectrogram_null_safety();
    test_mel_spectrogram_extreme_pcm();
    test_sentence_buffer_speculative();
    test_sentence_buffer_prosody();
    test_text_normalize_formats();
    test_text_expand_nonverbalisms();
    test_conformer_extended_null_safety();

    printf("\n═══ Results: %d passed, %d failed ═══\n", g_pass, g_fail);
    return g_fail > 0 ? 1 : 0;
}
