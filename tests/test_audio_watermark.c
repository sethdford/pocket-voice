/**
 * test_audio_watermark.c — Tests for spread-spectrum audio watermarking.
 *
 * Tests:
 *   1. Create/destroy lifecycle
 *   2. Embed + detect round-trip (sine, speech-like, silence)
 *   3. SNR degradation < 0.5 dB (imperceptibility)
 *   4. Robustness: embed → downsample → upsample → detect
 *   5. Payload extraction accuracy
 *   6. No false positives on unwatermarked audio
 *   7. Enable/disable toggle
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include "audio_watermark.h"

static int passed = 0, failed = 0;
#define CHECK(cond, msg) do { \
    if (cond) { printf("  [PASS] %s\n", msg); passed++; } \
    else { printf("  [FAIL] %s\n", msg); failed++; } \
} while (0)

/* ── Signal generators ─────────────────────────────────────────────────── */

/** Generate a sine wave at given frequency. */
static void gen_sine(float *buf, int n, float freq, int sample_rate, float amp) {
    for (int i = 0; i < n; i++) {
        buf[i] = amp * sinf(2.0f * (float)M_PI * freq * (float)i / (float)sample_rate);
    }
}

/** Generate a speech-like signal: sum of formant-like frequencies with noise. */
static void gen_speech_like(float *buf, int n, int sample_rate) {
    /* Fundamental + formants */
    float freqs[] = { 150.0f, 500.0f, 1500.0f, 2500.0f, 3500.0f };
    float amps[]  = { 0.3f,   0.25f,  0.15f,   0.1f,    0.05f   };
    memset(buf, 0, (size_t)n * sizeof(float));

    for (int f = 0; f < 5; f++) {
        for (int i = 0; i < n; i++) {
            buf[i] += amps[f] * sinf(2.0f * (float)M_PI * freqs[f] *
                                     (float)i / (float)sample_rate);
        }
    }

    /* Add slight amplitude modulation (simulates syllables) */
    for (int i = 0; i < n; i++) {
        float env = 0.5f + 0.5f * sinf(2.0f * (float)M_PI * 4.0f *
                                        (float)i / (float)sample_rate);
        buf[i] *= env;
    }
}

/** Compute RMS of a signal. */
static float compute_rms(const float *buf, int n) {
    float sum_sq = 0.0f;
    for (int i = 0; i < n; i++) {
        sum_sq += buf[i] * buf[i];
    }
    return sqrtf(sum_sq / (float)n);
}

/** Compute SNR in dB between original and processed signal. */
static float compute_snr_db(const float *original, const float *processed, int n) {
    float sig_power = 0.0f;
    float noise_power = 0.0f;
    for (int i = 0; i < n; i++) {
        sig_power += original[i] * original[i];
        float diff = processed[i] - original[i];
        noise_power += diff * diff;
    }
    if (noise_power < 1e-20f) return 200.0f; /* Essentially identical */
    return 10.0f * log10f(sig_power / noise_power);
}

/** Simple linear resampling (for robustness test). */
static void linear_resample_test(const float *in, int in_len, float *out,
                                 int out_len) {
    if (out_len <= 0 || in_len <= 0) return;
    for (int i = 0; i < out_len; i++) {
        float pos = (float)i * (float)(in_len - 1) / (float)(out_len - 1);
        int idx = (int)pos;
        float frac = pos - (float)idx;
        if (idx >= in_len - 1) {
            out[i] = in[in_len - 1];
        } else {
            out[i] = in[idx] * (1.0f - frac) + in[idx + 1] * frac;
        }
    }
}

/* ── Tests ─────────────────────────────────────────────────────────────── */

static void test_create_destroy(void) {
    printf("\n=== Create / Destroy ===\n");

    uint8_t key[] = "test-watermark-key-12345";

    /* Valid creation */
    AudioWatermark *wm = audio_watermark_create(24000, 960, key, (int)sizeof(key));
    CHECK(wm != NULL, "Create with valid params");
    if (wm) audio_watermark_destroy(wm);

    /* Invalid: NULL key */
    wm = audio_watermark_create(24000, 960, NULL, 0);
    CHECK(wm == NULL, "Reject NULL key");

    /* Invalid: key too short */
    wm = audio_watermark_create(24000, 960, key, 2);
    CHECK(wm == NULL, "Reject key_len < 4");

    /* Invalid: bad sample rate */
    wm = audio_watermark_create(0, 960, key, (int)sizeof(key));
    CHECK(wm == NULL, "Reject sample_rate=0");

    /* Invalid: bad frame size */
    wm = audio_watermark_create(24000, 0, key, (int)sizeof(key));
    CHECK(wm == NULL, "Reject frame_size=0");

    /* Larger frame size */
    wm = audio_watermark_create(24000, 2048, key, (int)sizeof(key));
    CHECK(wm != NULL, "Create with frame_size=2048");
    if (wm) audio_watermark_destroy(wm);

    /* 48kHz sample rate */
    wm = audio_watermark_create(48000, 960, key, (int)sizeof(key));
    CHECK(wm != NULL, "Create at 48kHz");
    if (wm) audio_watermark_destroy(wm);
}

static void test_embed_detect_sine(void) {
    printf("\n=== Embed + Detect (Sine Wave) ===\n");

    uint8_t key[] = "sine-watermark-secret-key";
    AudioWatermark *wm = audio_watermark_create(24000, 960, key, (int)sizeof(key));
    CHECK(wm != NULL, "Create watermark context");
    if (!wm) return;

    AudioWatermarkPayload payload = { .ai_generated = 1, .timestamp = 1709000000, .model_id = 42 };
    audio_watermark_set_payload(wm, &payload);

    /* Generate 1 second of 440Hz sine */
    const int n = 24000;
    float *signal = malloc((size_t)n * sizeof(float));
    float *original = malloc((size_t)n * sizeof(float));
    CHECK(signal != NULL && original != NULL, "Allocate test buffers");

    gen_sine(signal, n, 440.0f, 24000, 0.5f);
    memcpy(original, signal, (size_t)n * sizeof(float));

    /* Embed */
    int rc = audio_watermark_embed(wm, signal, n);
    CHECK(rc == 0, "Embed returns success");

    /* Detect */
    float score = audio_watermark_detect(wm, signal, n);
    printf("    Detection score (watermarked sine): %.4f\n", score);
    CHECK(score > 0.3f, "Detection score > 0.3 on watermarked signal");

    /* Check SNR */
    float snr = compute_snr_db(original, signal, n);
    printf("    SNR degradation: %.1f dB (SNR = %.1f dB)\n",
           snr < 200.0f ? -snr + 200.0f : 0.0f, snr);
    CHECK(snr > 20.0f, "SNR > 20 dB (imperceptible watermark)");

    free(signal);
    free(original);
    audio_watermark_destroy(wm);
}

static void test_embed_detect_speech(void) {
    printf("\n=== Embed + Detect (Speech-like Signal) ===\n");

    uint8_t key[] = "speech-watermark-key-2024";
    AudioWatermark *wm = audio_watermark_create(24000, 960, key, (int)sizeof(key));
    CHECK(wm != NULL, "Create watermark context");
    if (!wm) return;

    AudioWatermarkPayload payload = { .ai_generated = 1, .timestamp = 1709100000, .model_id = 7 };
    audio_watermark_set_payload(wm, &payload);

    const int n = 24000; /* 1 second */
    float *signal = malloc((size_t)n * sizeof(float));
    float *original = malloc((size_t)n * sizeof(float));

    gen_speech_like(signal, n, 24000);
    memcpy(original, signal, (size_t)n * sizeof(float));

    int rc = audio_watermark_embed(wm, signal, n);
    CHECK(rc == 0, "Embed returns success");

    float score = audio_watermark_detect(wm, signal, n);
    printf("    Detection score (watermarked speech): %.4f\n", score);
    CHECK(score > 0.3f, "Detection score > 0.3 on speech-like signal");

    float snr = compute_snr_db(original, signal, n);
    printf("    SNR: %.1f dB\n", snr);
    CHECK(snr > 20.0f, "SNR > 20 dB on speech-like signal");

    free(signal);
    free(original);
    audio_watermark_destroy(wm);
}

static void test_no_false_positive(void) {
    printf("\n=== No False Positives ===\n");

    uint8_t key[] = "false-positive-test-key";
    AudioWatermark *wm = audio_watermark_create(24000, 960, key, (int)sizeof(key));
    CHECK(wm != NULL, "Create watermark context");
    if (!wm) return;

    const int n = 24000;
    float *signal = malloc((size_t)n * sizeof(float));

    /* Test with unwatermarked sine */
    gen_sine(signal, n, 440.0f, 24000, 0.5f);
    float score = audio_watermark_detect(wm, signal, n);
    printf("    Detection score (unwatermarked sine): %.4f\n", score);
    CHECK(score < 0.3f, "No false positive on unwatermarked sine");

    /* Test with unwatermarked speech */
    gen_speech_like(signal, n, 24000);
    score = audio_watermark_detect(wm, signal, n);
    printf("    Detection score (unwatermarked speech): %.4f\n", score);
    CHECK(score < 0.3f, "No false positive on unwatermarked speech");

    /* Test with silence */
    memset(signal, 0, (size_t)n * sizeof(float));
    score = audio_watermark_detect(wm, signal, n);
    printf("    Detection score (silence): %.4f\n", score);
    CHECK(score < 0.3f, "No false positive on silence");

    /* Test with random noise (deterministic seed for reproducibility) */
    srand(42);
    for (int i = 0; i < n; i++) {
        signal[i] = ((float)rand() / (float)RAND_MAX - 0.5f) * 0.1f;
    }
    score = audio_watermark_detect(wm, signal, n);
    printf("    Detection score (random noise): %.4f\n", score);
    CHECK(score < 0.3f, "No false positive on random noise");

    /* Test with wrong key */
    uint8_t wrong_key[] = "completely-different-key!!";
    AudioWatermark *wm2 = audio_watermark_create(24000, 960, wrong_key, (int)sizeof(wrong_key));
    CHECK(wm2 != NULL, "Create second watermark with different key");
    if (wm2) {
        gen_sine(signal, n, 440.0f, 24000, 0.5f);
        audio_watermark_embed(wm, signal, n);  /* Embed with key 1 */
        score = audio_watermark_detect(wm2, signal, n); /* Detect with key 2 */
        printf("    Detection score (wrong key): %.4f\n", score);
        CHECK(score < 0.3f, "No detection with wrong key");
        audio_watermark_destroy(wm2);
    }

    free(signal);
    audio_watermark_destroy(wm);
}

static void test_robustness_resample(void) {
    printf("\n=== Robustness: Resample Round-trip ===\n");

    uint8_t key[] = "robustness-resample-key";
    AudioWatermark *wm = audio_watermark_create(24000, 960, key, (int)sizeof(key));
    CHECK(wm != NULL, "Create watermark context");
    if (!wm) return;

    AudioWatermarkPayload payload = { .ai_generated = 1, .timestamp = 1234567890, .model_id = 99 };
    audio_watermark_set_payload(wm, &payload);

    const int n = 24000; /* 1 second at 24kHz */
    float *signal = malloc((size_t)n * sizeof(float));
    gen_speech_like(signal, n, 24000);

    /* Embed watermark */
    audio_watermark_embed(wm, signal, n);

    /* Downsample 24kHz → 16kHz */
    int n_down = (int)((float)n * 16000.0f / 24000.0f);
    float *down = malloc((size_t)n_down * sizeof(float));
    linear_resample_test(signal, n, down, n_down);

    /* Upsample 16kHz → 24kHz */
    float *back = malloc((size_t)n * sizeof(float));
    linear_resample_test(down, n_down, back, n);

    /* Detect on resampled signal */
    float score = audio_watermark_detect(wm, back, n);
    printf("    Detection score after 24k→16k→24k resample: %.4f\n", score);
    /* Resampling may degrade watermark; a lower threshold is acceptable */
    CHECK(score > 0.15f, "Watermark partially survives resample (score > 0.15)");

    free(signal);
    free(down);
    free(back);
    audio_watermark_destroy(wm);
}

static void test_payload_extraction(void) {
    printf("\n=== Payload Extraction ===\n");

    uint8_t key[] = "payload-extraction-key-test";
    AudioWatermark *wm = audio_watermark_create(24000, 960, key, (int)sizeof(key));
    CHECK(wm != NULL, "Create watermark context");
    if (!wm) return;

    AudioWatermarkPayload payload = {
        .ai_generated = 1,
        .timestamp = 1709123456,
        .model_id = 255
    };
    audio_watermark_set_payload(wm, &payload);

    /* Generate 8 seconds of speech-like signal for reliable bit extraction.
     * Per-bit SNR improves with sqrt(n_frames) — need enough frames for all
     * 49 payload bits to decode correctly. */
    const int n = 192000;
    float *signal = malloc((size_t)n * sizeof(float));
    gen_speech_like(signal, n, 24000);

    audio_watermark_embed(wm, signal, n);

    /* Extract payload */
    AudioWatermarkPayload extracted = { 0 };
    int rc = audio_watermark_extract(wm, signal, n, &extracted);
    CHECK(rc == 0, "Extraction returns success");

    printf("    Expected:  ai=%d ts=%u model=%u\n",
           payload.ai_generated, payload.timestamp, payload.model_id);
    printf("    Extracted: ai=%d ts=%u model=%u\n",
           extracted.ai_generated, extracted.timestamp, extracted.model_id);

    /* P0-6 fix: verify ALL 49 payload bits, not just ai_generated */
    CHECK(extracted.ai_generated == payload.ai_generated, "AI flag matches");
    CHECK(extracted.timestamp == payload.timestamp, "Timestamp matches (32 bits)");
    CHECK(extracted.model_id == payload.model_id, "Model ID matches (16 bits)");

    /* Test with a different payload to ensure all bit patterns work */
    AudioWatermarkPayload payload2 = {
        .ai_generated = 0,
        .timestamp = 0xDEADBEEF,
        .model_id = 0xCAFE
    };
    audio_watermark_set_payload(wm, &payload2);
    audio_watermark_reset(wm);

    float *signal2 = malloc((size_t)n * sizeof(float));
    gen_speech_like(signal2, n, 24000);
    audio_watermark_embed(wm, signal2, n);

    AudioWatermarkPayload extracted2 = { 0 };
    rc = audio_watermark_extract(wm, signal2, n, &extracted2);
    CHECK(rc == 0, "Second extraction returns success");

    printf("    Expected2:  ai=%d ts=0x%08X model=0x%04X\n",
           payload2.ai_generated, payload2.timestamp, payload2.model_id);
    printf("    Extracted2: ai=%d ts=0x%08X model=0x%04X\n",
           extracted2.ai_generated, extracted2.timestamp, extracted2.model_id);

    CHECK(extracted2.ai_generated == payload2.ai_generated, "AI flag=0 matches");
    CHECK(extracted2.timestamp == payload2.timestamp, "Timestamp 0xDEADBEEF matches");
    CHECK(extracted2.model_id == payload2.model_id, "Model ID 0xCAFE matches");

    free(signal);
    free(signal2);
    audio_watermark_destroy(wm);
}

static void test_codec_survival(void) {
    printf("\n=== Codec Survival (P0-5) ===\n");

    uint8_t key[] = "codec-survival-test-key";
    AudioWatermark *wm = audio_watermark_create(24000, 960, key, (int)sizeof(key));
    CHECK(wm != NULL, "Create watermark context");
    if (!wm) return;

    AudioWatermarkPayload payload = { .ai_generated = 1, .timestamp = 1234567890, .model_id = 42 };
    audio_watermark_set_payload(wm, &payload);

    const int n = 48000; /* 2 seconds */
    float *signal = malloc((size_t)n * sizeof(float));
    gen_speech_like(signal, n, 24000);

    /* Embed watermark */
    audio_watermark_embed(wm, signal, n);

    /* Verify watermark before degradation */
    float score_clean = audio_watermark_detect(wm, signal, n);
    printf("    Detection before degradation: %.4f\n", score_clean);
    CHECK(score_clean > 0.3f, "Watermark detected before degradation");

    /* Simulate 16-bit PCM codec: quantize to int16 and back */
    for (int i = 0; i < n; i++) {
        float clamped = signal[i];
        if (clamped > 1.0f) clamped = 1.0f;
        if (clamped < -1.0f) clamped = -1.0f;
        int16_t q = (int16_t)(clamped * 32767.0f);
        signal[i] = (float)q / 32767.0f;
    }

    float score_quant = audio_watermark_detect(wm, signal, n);
    printf("    Detection after 16-bit quantization: %.4f\n", score_quant);
    CHECK(score_quant > 0.2f, "Watermark survives 16-bit quantization");

    /* Add low-level noise (simulates codec artifacts, ~-60dB) */
    for (int i = 0; i < n; i++) {
        float noise = ((float)rand() / (float)RAND_MAX - 0.5f) * 0.002f;
        signal[i] += noise;
    }

    float score_noisy = audio_watermark_detect(wm, signal, n);
    printf("    Detection after quantization + noise: %.4f\n", score_noisy);
    CHECK(score_noisy > 0.15f, "Watermark partially survives quantization + noise");

    free(signal);
    audio_watermark_destroy(wm);
}

static void test_enable_disable(void) {
    printf("\n=== Enable / Disable ===\n");

    uint8_t key[] = "enable-disable-test-key";
    AudioWatermark *wm = audio_watermark_create(24000, 960, key, (int)sizeof(key));
    CHECK(wm != NULL, "Create watermark context");
    if (!wm) return;

    CHECK(audio_watermark_is_enabled(wm) == 1, "Enabled by default");

    audio_watermark_enable(wm, 0);
    CHECK(audio_watermark_is_enabled(wm) == 0, "Disabled after enable(0)");

    /* Embed with watermark disabled — signal should be unchanged */
    const int n = 24000;
    float *signal = malloc((size_t)n * sizeof(float));
    float *original = malloc((size_t)n * sizeof(float));
    gen_sine(signal, n, 440.0f, 24000, 0.5f);
    memcpy(original, signal, (size_t)n * sizeof(float));

    audio_watermark_embed(wm, signal, n);

    /* Check signal is unchanged */
    int unchanged = 1;
    for (int i = 0; i < n; i++) {
        if (fabsf(signal[i] - original[i]) > 1e-10f) {
            unchanged = 0;
            break;
        }
    }
    CHECK(unchanged, "Signal unchanged when watermark disabled");

    /* Re-enable and verify watermark is applied */
    audio_watermark_enable(wm, 1);
    CHECK(audio_watermark_is_enabled(wm) == 1, "Re-enabled after enable(1)");

    memcpy(signal, original, (size_t)n * sizeof(float));
    audio_watermark_embed(wm, signal, n);

    int changed = 0;
    for (int i = 0; i < n; i++) {
        if (fabsf(signal[i] - original[i]) > 1e-10f) {
            changed = 1;
            break;
        }
    }
    CHECK(changed, "Signal modified when watermark enabled");

    free(signal);
    free(original);
    audio_watermark_destroy(wm);
}

static void test_snr_degradation(void) {
    printf("\n=== SNR Degradation (Imperceptibility) ===\n");

    uint8_t key[] = "snr-measurement-key-test";
    AudioWatermark *wm = audio_watermark_create(24000, 960, key, (int)sizeof(key));
    CHECK(wm != NULL, "Create watermark context");
    if (!wm) return;

    const int n = 48000; /* 2 seconds */
    float *signal = malloc((size_t)n * sizeof(float));
    float *original = malloc((size_t)n * sizeof(float));

    /* Test with speech-like signal (most representative) */
    gen_speech_like(signal, n, 24000);
    memcpy(original, signal, (size_t)n * sizeof(float));

    audio_watermark_embed(wm, signal, n);

    float snr = compute_snr_db(original, signal, n);
    printf("    SNR on speech-like signal: %.1f dB\n", snr);

    /* The watermark is at -40dB below signal, so SNR should be well above 30dB.
     * We require > 20dB as a conservative threshold accounting for
     * windowing and overlap-add artifacts. */
    CHECK(snr > 20.0f, "SNR > 20 dB (well below audibility threshold)");

    /* Test with 1kHz tone (edge of watermark band) */
    gen_sine(signal, n, 1000.0f, 24000, 0.5f);
    memcpy(original, signal, (size_t)n * sizeof(float));
    audio_watermark_embed(wm, signal, n);
    float snr_tone = compute_snr_db(original, signal, n);
    printf("    SNR on 1kHz tone: %.1f dB\n", snr_tone);
    CHECK(snr_tone > 15.0f, "SNR > 15 dB on edge-band tone");

    free(signal);
    free(original);
    audio_watermark_destroy(wm);
}

static void test_silence_handling(void) {
    printf("\n=== Silence Handling ===\n");

    uint8_t key[] = "silence-test-key-12345";
    AudioWatermark *wm = audio_watermark_create(24000, 960, key, (int)sizeof(key));
    CHECK(wm != NULL, "Create watermark context");
    if (!wm) return;

    const int n = 24000;
    float *signal = calloc((size_t)n, sizeof(float)); /* All zeros */
    float *original = calloc((size_t)n, sizeof(float));

    int rc = audio_watermark_embed(wm, signal, n);
    CHECK(rc == 0, "Embed on silence succeeds");

    /* Watermark on silence should be very quiet (near-zero) */
    float rms = compute_rms(signal, n);
    printf("    RMS of watermarked silence: %.8f\n", rms);
    CHECK(rms < 0.001f, "Watermark on silence is near-inaudible");

    free(signal);
    free(original);
    audio_watermark_destroy(wm);
}

static void test_short_buffer(void) {
    printf("\n=== Short Buffer Handling ===\n");

    uint8_t key[] = "short-buffer-test-key";
    AudioWatermark *wm = audio_watermark_create(24000, 960, key, (int)sizeof(key));
    CHECK(wm != NULL, "Create watermark context");
    if (!wm) return;

    /* Buffer shorter than FFT size — should return 0 (no-op) */
    float short_buf[100];
    gen_sine(short_buf, 100, 440.0f, 24000, 0.5f);
    float orig[100];
    memcpy(orig, short_buf, sizeof(short_buf));

    int rc = audio_watermark_embed(wm, short_buf, 100);
    CHECK(rc == 0, "Embed on short buffer returns success (no-op)");

    /* Buffer should be unchanged (too short to process) */
    int unchanged = 1;
    for (int i = 0; i < 100; i++) {
        if (fabsf(short_buf[i] - orig[i]) > 1e-10f) {
            unchanged = 0;
            break;
        }
    }
    CHECK(unchanged, "Short buffer left unmodified");

    /* P0-7 fix: Detection on short buffer returns 0.0 (no frames processed),
     * not just "any non-negative value" which is always true */
    float score = audio_watermark_detect(wm, short_buf, 100);
    CHECK(score == 0.0f, "Detect on short buffer returns exactly 0.0 (no frames)");

    audio_watermark_destroy(wm);
}

/* ── Main ──────────────────────────────────────────────────────────────── */

int main(void) {
    printf("═══ Audio Watermark Tests ═══\n");

    test_create_destroy();
    test_embed_detect_sine();
    test_embed_detect_speech();
    test_no_false_positive();
    test_robustness_resample();
    test_payload_extraction();
    test_codec_survival();
    test_enable_disable();
    test_snr_degradation();
    test_silence_handling();
    test_short_buffer();

    printf("\n═══ Results: %d passed, %d failed ═══\n", passed, failed);
    return failed > 0 ? 1 : 0;
}
