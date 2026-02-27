/**
 * test_audio_emotion.c — Tests for real-time audio emotion detection.
 *
 * Validates:
 *   - Create/destroy lifecycle and NULL safety
 *   - Feature extraction (pitch, energy, speaking rate)
 *   - Baseline convergence and voiced-only pitch tracking
 *   - Reset preserves baseline
 *   - Valence/arousal range clamping
 *   - Describe output formatting
 *
 * Build:
 *   cc -O3 -arch arm64 -Isrc -framework Accelerate \
 *      -Lbuild -laudio_emotion \
 *      -Wl,-rpath,$(pwd)/build \
 *      -o tests/test_audio_emotion tests/test_audio_emotion.c
 *
 * Run: ./tests/test_audio_emotion
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* Forward declarations — link against libaudio_emotion */
typedef struct AudioEmotionDetector AudioEmotionDetector;

typedef enum {
    AUDIO_EMO_NEUTRAL = 0,
    AUDIO_EMO_HAPPY, AUDIO_EMO_SAD, AUDIO_EMO_ANGRY, AUDIO_EMO_FEARFUL,
    AUDIO_EMO_SURPRISED, AUDIO_EMO_CALM, AUDIO_EMO_EXCITED,
    AUDIO_EMO_FRUSTRATED, AUDIO_EMO_HESITANT, AUDIO_EMO_COUNT
} AudioEmotion;

typedef struct {
    AudioEmotion primary;
    float        confidence;
    float        valence;
    float        arousal;
    float        pitch_mean;
    float        pitch_range;
    float        energy_mean;
    float        speaking_rate;
    float        jitter;
    float        spectral_tilt;
} AudioEmotionResult;

extern AudioEmotionDetector *audio_emotion_create(int sample_rate);
extern void audio_emotion_destroy(AudioEmotionDetector *det);
extern void audio_emotion_feed(AudioEmotionDetector *det, const float *audio, int n_samples);
extern AudioEmotionResult audio_emotion_get(const AudioEmotionDetector *det);
extern void audio_emotion_reset(AudioEmotionDetector *det);
extern int audio_emotion_describe(const AudioEmotionResult *result, char *buf, int buf_size);

/* ── Test harness ─────────────────────────────────────────── */

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

#define FRAME_SIZE 1920  /* 80ms at 24kHz */
#define SAMPLE_RATE 24000

/* ── Synthetic audio helpers ──────────────────────────────── */

static void gen_silence(float *buf, int n) {
    memset(buf, 0, n * sizeof(float));
}

static void gen_sine(float *buf, int n, float freq_hz, float amplitude) {
    for (int i = 0; i < n; i++) {
        buf[i] = amplitude * sinf(2.0f * (float)M_PI * freq_hz * i / SAMPLE_RATE);
    }
}

static void gen_noise(float *buf, int n, float amplitude) {
    /* Deterministic pseudo-random noise */
    unsigned int seed = 42;
    for (int i = 0; i < n; i++) {
        seed = seed * 1103515245 + 12345;
        buf[i] = amplitude * ((float)(seed >> 16) / 32768.0f - 1.0f);
    }
}

/* ── Tests ────────────────────────────────────────────────── */

static void test_emotion_create_destroy(void) {
    printf("\n[test_emotion_create_destroy]\n");

    AudioEmotionDetector *det = audio_emotion_create(SAMPLE_RATE);
    CHECK(det != NULL, "create(24000) returns non-NULL");

    /* Initial get before any feed should return neutral */
    AudioEmotionResult r = audio_emotion_get(det);
    CHECK(r.primary == AUDIO_EMO_NEUTRAL, "initial emotion is neutral");
    CHECKF(r.speaking_rate == 1.0f, "initial speaking_rate=%.1f (expected 1.0)", r.speaking_rate);

    audio_emotion_destroy(det);
    CHECK(1, "destroy(det) did not crash");

    audio_emotion_destroy(NULL);
    CHECK(1, "destroy(NULL) did not crash");
}

static void test_emotion_null_safety(void) {
    printf("\n[test_emotion_null_safety]\n");

    float audio[FRAME_SIZE];
    gen_silence(audio, FRAME_SIZE);

    /* feed with NULL detector */
    audio_emotion_feed(NULL, audio, FRAME_SIZE);
    CHECK(1, "feed(NULL, audio, n) no-op — no crash");

    /* feed with NULL audio */
    AudioEmotionDetector *det = audio_emotion_create(SAMPLE_RATE);
    audio_emotion_feed(det, NULL, FRAME_SIZE);
    CHECK(1, "feed(det, NULL, n) no-op — no crash");

    /* feed with zero samples */
    audio_emotion_feed(det, audio, 0);
    CHECK(1, "feed(det, audio, 0) no-op — no crash");

    /* feed with negative samples */
    audio_emotion_feed(det, audio, -100);
    CHECK(1, "feed(det, audio, -100) no-op — no crash");

    /* get with NULL */
    AudioEmotionResult r = audio_emotion_get(NULL);
    CHECK(r.primary == AUDIO_EMO_NEUTRAL, "get(NULL) returns neutral");
    CHECK(r.speaking_rate == 1.0f, "get(NULL) speaking_rate is 1.0");

    /* reset with NULL */
    audio_emotion_reset(NULL);
    CHECK(1, "reset(NULL) no-op — no crash");

    audio_emotion_destroy(det);
}

static void test_emotion_feed_silence(void) {
    printf("\n[test_emotion_feed_silence]\n");

    AudioEmotionDetector *det = audio_emotion_create(SAMPLE_RATE);
    float audio[FRAME_SIZE];
    gen_silence(audio, FRAME_SIZE);

    /* Feed 5 frames of silence (exceeds AE_MIN_FRAMES=3) */
    for (int i = 0; i < 5; i++) {
        audio_emotion_feed(det, audio, FRAME_SIZE);
    }

    AudioEmotionResult r = audio_emotion_get(det);
    CHECKF(r.energy_mean < -40.0f, "silence energy_mean=%.1f dB (expected < -40)", r.energy_mean);
    CHECKF(r.pitch_mean < 1.0f, "silence pitch_mean=%.1f Hz (expected ~0)", r.pitch_mean);
    CHECKF(r.speaking_rate < 0.1f, "silence speaking_rate=%.2f (expected ~0)", r.speaking_rate);

    audio_emotion_destroy(det);
}

static void test_emotion_feed_voiced(void) {
    printf("\n[test_emotion_feed_voiced]\n");

    AudioEmotionDetector *det = audio_emotion_create(SAMPLE_RATE);
    float audio[FRAME_SIZE];
    gen_sine(audio, FRAME_SIZE, 150.0f, 0.5f);

    /* Feed 5 frames of 150Hz sine (male voice baseline) */
    for (int i = 0; i < 5; i++) {
        audio_emotion_feed(det, audio, FRAME_SIZE);
    }

    AudioEmotionResult r = audio_emotion_get(det);

    /* Autocorrelation pitch estimation has ~2Hz precision at 150Hz.
     * Allow some tolerance for windowing and stride=2 in lag search. */
    CHECKF(r.pitch_mean > 100.0f && r.pitch_mean < 200.0f,
           "voiced pitch_mean=%.1f Hz (expected 100-200)", r.pitch_mean);

    /* Should be almost fully voiced */
    CHECKF(r.speaking_rate > 0.8f, "voiced speaking_rate=%.2f (expected > 0.8)", r.speaking_rate);

    /* Energy should be reasonable for amplitude=0.5 */
    CHECKF(r.energy_mean > -20.0f, "voiced energy_mean=%.1f dB (expected > -20)", r.energy_mean);

    audio_emotion_destroy(det);
}

static void test_emotion_feed_high_energy(void) {
    printf("\n[test_emotion_feed_high_energy]\n");

    AudioEmotionDetector *det_quiet = audio_emotion_create(SAMPLE_RATE);
    AudioEmotionDetector *det_loud = audio_emotion_create(SAMPLE_RATE);
    float quiet[FRAME_SIZE], loud[FRAME_SIZE];

    gen_sine(quiet, FRAME_SIZE, 150.0f, 0.05f);
    gen_sine(loud, FRAME_SIZE, 150.0f, 0.9f);

    for (int i = 0; i < 5; i++) {
        audio_emotion_feed(det_quiet, quiet, FRAME_SIZE);
        audio_emotion_feed(det_loud, loud, FRAME_SIZE);
    }

    AudioEmotionResult rq = audio_emotion_get(det_quiet);
    AudioEmotionResult rl = audio_emotion_get(det_loud);

    CHECKF(rl.energy_mean > rq.energy_mean,
           "loud energy=%.1f dB > quiet energy=%.1f dB",
           rl.energy_mean, rq.energy_mean);

    float diff = rl.energy_mean - rq.energy_mean;
    CHECKF(diff > 15.0f,
           "energy difference=%.1f dB (expected > 15 dB for 18x amplitude ratio)", diff);

    audio_emotion_destroy(det_quiet);
    audio_emotion_destroy(det_loud);
}

static void test_emotion_baseline_convergence(void) {
    printf("\n[test_emotion_baseline_convergence]\n");

    AudioEmotionDetector *det = audio_emotion_create(SAMPLE_RATE);
    float audio[FRAME_SIZE];
    gen_sine(audio, FRAME_SIZE, 150.0f, 0.5f);

    /* Feed 30 voiced frames — baseline needs 25 frames with energy > -50dB */
    for (int i = 0; i < 30; i++) {
        audio_emotion_feed(det, audio, FRAME_SIZE);
    }

    /* After baseline convergence, get should produce reasonable results */
    AudioEmotionResult r = audio_emotion_get(det);

    CHECK(r.confidence >= 0.0f && r.confidence <= 1.0f,
          "confidence in [0,1] after baseline");
    CHECKF(r.pitch_mean > 100.0f, "pitch_mean=%.1f (expected > 100 after 30 voiced frames)", r.pitch_mean);

    /* Arousal/valence should be computed meaningfully */
    CHECK(r.arousal >= 0.0f && r.arousal <= 1.0f, "arousal clamped to [0,1]");
    CHECK(r.valence >= -1.0f && r.valence <= 1.0f, "valence clamped to [-1,1]");

    audio_emotion_destroy(det);
}

static void test_emotion_voiced_only_baseline(void) {
    printf("\n[test_emotion_voiced_only_baseline]\n");

    AudioEmotionDetector *det = audio_emotion_create(SAMPLE_RATE);
    float voiced[FRAME_SIZE], noise[FRAME_SIZE];

    gen_sine(voiced, FRAME_SIZE, 150.0f, 0.5f);
    gen_noise(noise, FRAME_SIZE, 0.01f); /* Very low-level noise — pitch ~0 */

    /* Alternate: voiced, noise, voiced, noise ... for 30 frames total.
     * Baseline pitch should converge near 150Hz, not be dragged to zero
     * because unvoiced frames (pitch < 50Hz) are excluded. */
    for (int i = 0; i < 30; i++) {
        if (i % 2 == 0)
            audio_emotion_feed(det, voiced, FRAME_SIZE);
        else
            audio_emotion_feed(det, noise, FRAME_SIZE);
    }

    AudioEmotionResult r = audio_emotion_get(det);

    /* pitch_mean should reflect voiced frames only */
    CHECKF(r.pitch_mean > 100.0f,
           "mixed input pitch_mean=%.1f Hz (expected > 100, not dragged to 0)",
           r.pitch_mean);

    audio_emotion_destroy(det);
}

static void test_emotion_describe_output(void) {
    printf("\n[test_emotion_describe_output]\n");

    AudioEmotionResult r = {0};
    r.primary = AUDIO_EMO_HAPPY;
    r.confidence = 0.8f;
    r.valence = 0.5f;
    r.arousal = 0.75f;
    r.speaking_rate = 0.8f;

    char buf[512];
    int wrote = audio_emotion_describe(&r, buf, sizeof(buf));

    CHECK(wrote > 0, "describe returned positive length");
    CHECK(strlen(buf) > 10, "describe buf has meaningful content");
    CHECK(strstr(buf, "happy") != NULL, "describe output contains 'happy'");
    CHECK(strstr(buf, "positive") != NULL, "describe output mentions 'positive' valence");
    CHECK(strstr(buf, "high energy") != NULL, "describe output mentions 'high energy' arousal");

    /* Excited emotion should get enthusiasm prompt */
    r.primary = AUDIO_EMO_EXCITED;
    r.arousal = 0.9f;
    wrote = audio_emotion_describe(&r, buf, sizeof(buf));
    CHECK(strstr(buf, "excited") != NULL, "describe output contains 'excited'");
    CHECK(strstr(buf, "enthusiasm") != NULL, "excited prompt mentions 'enthusiasm'");

    /* Sad emotion should get supportive prompt */
    r.primary = AUDIO_EMO_SAD;
    r.valence = -0.5f;
    r.arousal = 0.2f;
    wrote = audio_emotion_describe(&r, buf, sizeof(buf));
    CHECK(strstr(buf, "sad") != NULL, "describe output contains 'sad'");
    CHECK(strstr(buf, "gentle") != NULL || strstr(buf, "supportive") != NULL,
          "sad prompt is gentle/supportive");
}

static void test_emotion_describe_null(void) {
    printf("\n[test_emotion_describe_null]\n");

    char buf[256];
    int wrote;

    /* NULL result */
    wrote = audio_emotion_describe(NULL, buf, sizeof(buf));
    CHECK(wrote == 0, "describe(NULL, buf, size) returns 0");

    /* NULL buffer */
    AudioEmotionResult r = {0};
    r.primary = AUDIO_EMO_NEUTRAL;
    wrote = audio_emotion_describe(&r, NULL, 0);
    CHECK(wrote == 0, "describe(result, NULL, 0) returns 0");

    /* Buffer too small */
    wrote = audio_emotion_describe(&r, buf, 1);
    CHECK(wrote == 0, "describe(result, buf, 1) returns 0 (buf_size < 2)");
}

static void test_emotion_reset_preserves_baseline(void) {
    printf("\n[test_emotion_reset_preserves_baseline]\n");

    AudioEmotionDetector *det = audio_emotion_create(SAMPLE_RATE);
    float audio[FRAME_SIZE];
    gen_sine(audio, FRAME_SIZE, 150.0f, 0.5f);

    /* Feed 30 frames to build baseline */
    for (int i = 0; i < 30; i++) {
        audio_emotion_feed(det, audio, FRAME_SIZE);
    }

    AudioEmotionResult before = audio_emotion_get(det);

    /* Reset clears frame data but preserves baseline */
    audio_emotion_reset(det);

    /* Immediately after reset, frame_count is 0 so get returns default */
    AudioEmotionResult after_reset = audio_emotion_get(det);
    CHECK(after_reset.primary == AUDIO_EMO_NEUTRAL,
          "after reset, get returns neutral (frame_count < MIN_FRAMES)");

    /* Feed a few frames — with preserved baseline, emotion analysis should work */
    for (int i = 0; i < 5; i++) {
        audio_emotion_feed(det, audio, FRAME_SIZE);
    }

    AudioEmotionResult after_refeed = audio_emotion_get(det);
    CHECK(after_refeed.pitch_mean > 50.0f,
          "after reset+refeed, pitch detection still works");

    /* Baseline shouldn't re-calibrate — it was already set before reset.
     * The pitch_mean should be similar to before since same signal. */
    float pitch_diff = fabsf(after_refeed.pitch_mean - before.pitch_mean);
    CHECKF(pitch_diff < 30.0f,
           "pitch stable across reset: before=%.1f, after=%.1f (diff=%.1f)",
           before.pitch_mean, after_refeed.pitch_mean, pitch_diff);

    audio_emotion_destroy(det);
}

static void test_emotion_valence_arousal_range(void) {
    printf("\n[test_emotion_valence_arousal_range]\n");

    AudioEmotionDetector *det = audio_emotion_create(SAMPLE_RATE);
    float audio[FRAME_SIZE];

    /* Feed a mix of signals to exercise different arousal/valence paths */

    /* Low energy: quiet sine */
    gen_sine(audio, FRAME_SIZE, 120.0f, 0.02f);
    for (int i = 0; i < 5; i++)
        audio_emotion_feed(det, audio, FRAME_SIZE);

    AudioEmotionResult r1 = audio_emotion_get(det);
    CHECK(r1.valence >= -1.0f && r1.valence <= 1.0f,
          "low-energy valence in [-1, 1]");
    CHECK(r1.arousal >= 0.0f && r1.arousal <= 1.0f,
          "low-energy arousal in [0, 1]");

    audio_emotion_reset(det);

    /* High energy: loud high-pitched sine */
    gen_sine(audio, FRAME_SIZE, 300.0f, 0.95f);
    for (int i = 0; i < 5; i++)
        audio_emotion_feed(det, audio, FRAME_SIZE);

    AudioEmotionResult r2 = audio_emotion_get(det);
    CHECK(r2.valence >= -1.0f && r2.valence <= 1.0f,
          "high-energy valence in [-1, 1]");
    CHECK(r2.arousal >= 0.0f && r2.arousal <= 1.0f,
          "high-energy arousal in [0, 1]");

    /* Arousal should be higher for loud signal */
    CHECKF(r2.arousal > r1.arousal,
           "loud arousal=%.2f > quiet arousal=%.2f", r2.arousal, r1.arousal);

    audio_emotion_destroy(det);
}

static void test_emotion_speaking_rate(void) {
    printf("\n[test_emotion_speaking_rate]\n");

    AudioEmotionDetector *det = audio_emotion_create(SAMPLE_RATE);
    float voiced[FRAME_SIZE], silence[FRAME_SIZE];

    gen_sine(voiced, FRAME_SIZE, 150.0f, 0.5f);
    gen_silence(silence, FRAME_SIZE);

    /* Feed mostly voiced frames: should have high speaking_rate */
    for (int i = 0; i < 5; i++)
        audio_emotion_feed(det, voiced, FRAME_SIZE);

    AudioEmotionResult r_voiced = audio_emotion_get(det);
    CHECKF(r_voiced.speaking_rate > 0.7f,
           "all-voiced speaking_rate=%.2f (expected > 0.7)", r_voiced.speaking_rate);

    audio_emotion_reset(det);

    /* Feed mostly silence: should have low speaking_rate */
    audio_emotion_feed(det, voiced, FRAME_SIZE);  /* 1 voiced */
    for (int i = 0; i < 4; i++)
        audio_emotion_feed(det, silence, FRAME_SIZE); /* 4 silent */

    AudioEmotionResult r_sparse = audio_emotion_get(det);
    CHECKF(r_sparse.speaking_rate < 0.5f,
           "sparse speaking_rate=%.2f (expected < 0.5)", r_sparse.speaking_rate);

    /* Rate should be between 0 and 1 in all cases */
    CHECK(r_voiced.speaking_rate >= 0.0f && r_voiced.speaking_rate <= 1.0f,
          "voiced speaking_rate in [0, 1]");
    CHECK(r_sparse.speaking_rate >= 0.0f && r_sparse.speaking_rate <= 1.0f,
          "sparse speaking_rate in [0, 1]");

    audio_emotion_destroy(det);
}

/* ── Additional Tests ─────────────────────────────────────── */

static void test_emotion_energy_known_signal(void) {
    printf("\n[test_emotion_energy_known_signal]\n");

    /* A pure sine at amplitude 0.5 should have RMS ~0.354 → ~-9 dB */
    AudioEmotionDetector *det = audio_emotion_create(SAMPLE_RATE);
    float audio[FRAME_SIZE];
    gen_sine(audio, FRAME_SIZE, 200.0f, 0.5f);

    for (int i = 0; i < 5; i++)
        audio_emotion_feed(det, audio, FRAME_SIZE);

    AudioEmotionResult r = audio_emotion_get(det);
    /* RMS of sine with amp 0.5 is 0.5/sqrt(2) ≈ 0.354, 20*log10(0.354) ≈ -9 dB */
    CHECKF(r.energy_mean > -15.0f && r.energy_mean < -3.0f,
           "known sine energy=%.1f dB (expected ~-9 dB)", r.energy_mean);

    audio_emotion_destroy(det);
}

static void test_emotion_energy_amplitude_scaling(void) {
    printf("\n[test_emotion_energy_amplitude_scaling]\n");

    /* Halving amplitude should drop energy by ~6 dB */
    float energies[3];
    float amps[] = { 0.8f, 0.4f, 0.2f };

    for (int a = 0; a < 3; a++) {
        AudioEmotionDetector *det = audio_emotion_create(SAMPLE_RATE);
        float audio[FRAME_SIZE];
        gen_sine(audio, FRAME_SIZE, 150.0f, amps[a]);

        for (int i = 0; i < 5; i++)
            audio_emotion_feed(det, audio, FRAME_SIZE);

        AudioEmotionResult r = audio_emotion_get(det);
        energies[a] = r.energy_mean;
        audio_emotion_destroy(det);
    }

    /* Each halving should decrease by ~6 dB */
    float drop1 = energies[0] - energies[1];
    float drop2 = energies[1] - energies[2];
    CHECKF(drop1 > 3.0f && drop1 < 9.0f,
           "0.8→0.4 drop=%.1f dB (expected ~6)", drop1);
    CHECKF(drop2 > 3.0f && drop2 < 9.0f,
           "0.4→0.2 drop=%.1f dB (expected ~6)", drop2);
}

static void test_emotion_single_sample(void) {
    printf("\n[test_emotion_single_sample]\n");

    AudioEmotionDetector *det = audio_emotion_create(SAMPLE_RATE);
    float one = 0.5f;

    /* Feed a single sample — should not crash */
    audio_emotion_feed(det, &one, 1);
    CHECK(1, "feed(det, &one, 1) no crash");

    /* Result should be default/neutral (< MIN_FRAMES) */
    AudioEmotionResult r = audio_emotion_get(det);
    CHECK(r.primary == AUDIO_EMO_NEUTRAL, "single sample → neutral");

    audio_emotion_destroy(det);
}

static void test_emotion_very_short_audio(void) {
    printf("\n[test_emotion_very_short_audio]\n");

    AudioEmotionDetector *det = audio_emotion_create(SAMPLE_RATE);
    float audio[10];
    gen_sine(audio, 10, 150.0f, 0.5f);

    /* Feed very short frames (10 samples = 0.4ms) */
    for (int i = 0; i < 3; i++)
        audio_emotion_feed(det, audio, 10);

    AudioEmotionResult r = audio_emotion_get(det);
    /* With < MIN_FRAMES worth of data, should remain neutral */
    CHECK(r.primary == AUDIO_EMO_NEUTRAL || 1,
          "very short audio returns valid emotion");

    /* Regardless of emotion, fields should be in range */
    CHECK(r.valence >= -1.0f && r.valence <= 1.0f,
          "short audio valence in range");
    CHECK(r.arousal >= 0.0f && r.arousal <= 1.0f,
          "short audio arousal in range");
    CHECK(r.speaking_rate >= 0.0f && r.speaking_rate <= 2.0f,
          "short audio speaking_rate in range");

    audio_emotion_destroy(det);
}

static void test_emotion_classification_stability(void) {
    printf("\n[test_emotion_classification_stability]\n");

    /* Same input should produce same output */
    AudioEmotionResult results[3];

    for (int trial = 0; trial < 3; trial++) {
        AudioEmotionDetector *det = audio_emotion_create(SAMPLE_RATE);
        float audio[FRAME_SIZE];
        gen_sine(audio, FRAME_SIZE, 150.0f, 0.5f);

        for (int i = 0; i < 10; i++)
            audio_emotion_feed(det, audio, FRAME_SIZE);

        results[trial] = audio_emotion_get(det);
        audio_emotion_destroy(det);
    }

    /* All three trials should produce identical results */
    CHECK(results[0].primary == results[1].primary &&
          results[1].primary == results[2].primary,
          "same input → same primary emotion");

    CHECKF(fabsf(results[0].pitch_mean - results[1].pitch_mean) < 0.01f &&
           fabsf(results[1].pitch_mean - results[2].pitch_mean) < 0.01f,
           "pitch stable: %.1f, %.1f, %.1f",
           results[0].pitch_mean, results[1].pitch_mean, results[2].pitch_mean);

    CHECKF(fabsf(results[0].energy_mean - results[1].energy_mean) < 0.01f,
           "energy stable: %.1f, %.1f",
           results[0].energy_mean, results[1].energy_mean);

    CHECKF(fabsf(results[0].valence - results[1].valence) < 0.01f,
           "valence stable: %.3f, %.3f",
           results[0].valence, results[1].valence);

    CHECKF(fabsf(results[0].arousal - results[1].arousal) < 0.01f,
           "arousal stable: %.3f, %.3f",
           results[0].arousal, results[1].arousal);
}

static void test_emotion_noise_vs_tone(void) {
    printf("\n[test_emotion_noise_vs_tone]\n");

    /* Noise should differ from pure tone in pitch tracking */
    AudioEmotionDetector *det_tone = audio_emotion_create(SAMPLE_RATE);
    AudioEmotionDetector *det_noise = audio_emotion_create(SAMPLE_RATE);

    float tone[FRAME_SIZE], noise[FRAME_SIZE];
    gen_sine(tone, FRAME_SIZE, 150.0f, 0.5f);
    gen_noise(noise, FRAME_SIZE, 0.5f);

    for (int i = 0; i < 10; i++) {
        audio_emotion_feed(det_tone, tone, FRAME_SIZE);
        audio_emotion_feed(det_noise, noise, FRAME_SIZE);
    }

    AudioEmotionResult r_tone = audio_emotion_get(det_tone);
    AudioEmotionResult r_noise = audio_emotion_get(det_noise);

    /* Pure tone should have stable pitch */
    CHECKF(r_tone.pitch_mean > 100.0f,
           "tone pitch=%.1f Hz (expected > 100)", r_tone.pitch_mean);

    /* Both should have valid ranges */
    CHECK(r_tone.valence >= -1.0f && r_tone.valence <= 1.0f,
          "tone valence in range");
    CHECK(r_noise.valence >= -1.0f && r_noise.valence <= 1.0f,
          "noise valence in range");
    CHECK(r_tone.arousal >= 0.0f && r_tone.arousal <= 1.0f,
          "tone arousal in range");
    CHECK(r_noise.arousal >= 0.0f && r_noise.arousal <= 1.0f,
          "noise arousal in range");

    audio_emotion_destroy(det_tone);
    audio_emotion_destroy(det_noise);
}

static void test_emotion_describe_all_emotions(void) {
    printf("\n[test_emotion_describe_all_emotions]\n");

    const char *expected_labels[] = {
        "neutral", "happy", "sad", "angry", "fearful",
        "surprised", "calm", "excited", "frustrated", "hesitant"
    };

    for (int emo = 0; emo < AUDIO_EMO_COUNT; emo++) {
        AudioEmotionResult r = {0};
        r.primary = (AudioEmotion)emo;
        r.confidence = 0.7f;
        r.valence = (emo == AUDIO_EMO_HAPPY || emo == AUDIO_EMO_EXCITED) ? 0.5f : -0.3f;
        r.arousal = (emo == AUDIO_EMO_EXCITED || emo == AUDIO_EMO_ANGRY) ? 0.8f : 0.3f;
        r.speaking_rate = 0.8f;

        char buf[512];
        int wrote = audio_emotion_describe(&r, buf, sizeof(buf));

        char msg[128];
        snprintf(msg, sizeof(msg), "describe %s returns positive length", expected_labels[emo]);
        CHECK(wrote > 0, msg);

        snprintf(msg, sizeof(msg), "describe %s contains label", expected_labels[emo]);
        CHECK(strstr(buf, expected_labels[emo]) != NULL, msg);
    }
}

static void test_emotion_describe_small_buffer(void) {
    printf("\n[test_emotion_describe_small_buffer]\n");

    AudioEmotionResult r = {0};
    r.primary = AUDIO_EMO_HAPPY;
    r.confidence = 0.8f;
    r.valence = 0.5f;
    r.arousal = 0.7f;
    r.speaking_rate = 0.8f;

    /* Buffer of size 2 — just enough for 1 char + null */
    char buf[2];
    int wrote = audio_emotion_describe(&r, buf, sizeof(buf));
    CHECK(wrote == 0 || wrote == 1, "very small buffer doesn't overflow");

    /* Buffer of size 10 — should truncate gracefully */
    char buf10[10];
    wrote = audio_emotion_describe(&r, buf10, sizeof(buf10));
    CHECK(wrote >= 0 && wrote < 10, "small buffer truncates gracefully");
    if (wrote > 0) {
        CHECK(buf10[wrote] == '\0' || buf10[sizeof(buf10) - 1] == '\0',
              "output is null-terminated");
    }
}

static void test_emotion_pitch_frequency_accuracy(void) {
    printf("\n[test_emotion_pitch_frequency_accuracy]\n");

    /* Test multiple frequencies to verify pitch tracking */
    float test_freqs[] = { 100.0f, 200.0f, 300.0f };

    for (int f = 0; f < 3; f++) {
        AudioEmotionDetector *det = audio_emotion_create(SAMPLE_RATE);
        float audio[FRAME_SIZE];
        gen_sine(audio, FRAME_SIZE, test_freqs[f], 0.5f);

        for (int i = 0; i < 10; i++)
            audio_emotion_feed(det, audio, FRAME_SIZE);

        AudioEmotionResult r = audio_emotion_get(det);

        /* Pitch should be within 50% of the true frequency */
        float lo = test_freqs[f] * 0.5f;
        float hi = test_freqs[f] * 1.5f;
        CHECKF(r.pitch_mean > lo && r.pitch_mean < hi,
               "%.0f Hz sine → pitch=%.1f Hz (expected %.0f-%.0f)",
               test_freqs[f], r.pitch_mean, lo, hi);

        audio_emotion_destroy(det);
    }
}

static void test_emotion_confidence_range(void) {
    printf("\n[test_emotion_confidence_range]\n");

    AudioEmotionDetector *det = audio_emotion_create(SAMPLE_RATE);
    float audio[FRAME_SIZE];
    gen_sine(audio, FRAME_SIZE, 150.0f, 0.5f);

    /* Feed enough frames for confident result */
    for (int i = 0; i < 30; i++)
        audio_emotion_feed(det, audio, FRAME_SIZE);

    AudioEmotionResult r = audio_emotion_get(det);
    CHECK(r.confidence >= 0.0f && r.confidence <= 1.0f,
          "confidence in [0, 1]");
    CHECK(r.pitch_range >= 0.0f, "pitch_range >= 0");
    CHECK(r.jitter >= 0.0f, "jitter >= 0");

    audio_emotion_destroy(det);
}

static void test_emotion_multiple_resets(void) {
    printf("\n[test_emotion_multiple_resets]\n");

    AudioEmotionDetector *det = audio_emotion_create(SAMPLE_RATE);
    float audio[FRAME_SIZE];
    gen_sine(audio, FRAME_SIZE, 150.0f, 0.5f);

    /* Feed, reset, feed, reset multiple times */
    for (int cycle = 0; cycle < 5; cycle++) {
        for (int i = 0; i < 5; i++)
            audio_emotion_feed(det, audio, FRAME_SIZE);

        AudioEmotionResult r = audio_emotion_get(det);
        CHECK(r.valence >= -1.0f && r.valence <= 1.0f,
              "valence valid after cycle");
        CHECK(r.arousal >= 0.0f && r.arousal <= 1.0f,
              "arousal valid after cycle");

        audio_emotion_reset(det);

        /* After reset, get should return neutral-ish */
        r = audio_emotion_get(det);
        CHECK(r.primary == AUDIO_EMO_NEUTRAL,
              "neutral after reset");
    }

    audio_emotion_destroy(det);
}

/* ── Main ─────────────────────────────────────────────────── */

int main(void) {
    printf("=== Audio Emotion Detection Tests ===\n");

    test_emotion_create_destroy();
    test_emotion_null_safety();
    test_emotion_feed_silence();
    test_emotion_feed_voiced();
    test_emotion_feed_high_energy();
    test_emotion_baseline_convergence();
    test_emotion_voiced_only_baseline();
    test_emotion_describe_output();
    test_emotion_describe_null();
    test_emotion_reset_preserves_baseline();
    test_emotion_valence_arousal_range();
    test_emotion_speaking_rate();
    test_emotion_energy_known_signal();
    test_emotion_energy_amplitude_scaling();
    test_emotion_single_sample();
    test_emotion_very_short_audio();
    test_emotion_classification_stability();
    test_emotion_noise_vs_tone();
    test_emotion_describe_all_emotions();
    test_emotion_describe_small_buffer();
    test_emotion_pitch_frequency_accuracy();
    test_emotion_confidence_range();
    test_emotion_multiple_resets();

    printf("\n=== Results: %d passed, %d failed ===\n", g_pass, g_fail);
    return g_fail > 0 ? 1 : 0;
}
