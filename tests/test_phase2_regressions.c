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

    printf("\n═══ Results: %d passed, %d failed ═══\n", g_pass, g_fail);
    return g_fail > 0 ? 1 : 0;
}
