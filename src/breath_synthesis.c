/**
 * breath_synthesis.c — Human-like breath noise and micropause synthesis.
 *
 * Produces realistic breath sounds using Voss-McCartney pink noise filtered
 * through a 2nd-order Butterworth bandpass (200-2000Hz) to model the vocal
 * tract frequency response. ADSR-shaped envelopes give each breath a natural
 * attack-decay profile.
 *
 * Uses Apple vDSP for vectorized amplitude envelope application and ARM NEON
 * for crossfade operations.
 */

#include "breath_synthesis.h"
#include "neon_audio.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#endif

/* Voss-McCartney pink noise: 16 octaves for high-quality 1/f spectrum */
#define PINK_OCTAVES 16

typedef struct {
    float biquad_state[4]; /* x[n-1], x[n-2], y[n-1], y[n-2] */
    float b0, b1, b2, a1, a2;
} BiquadFilter;

struct BreathSynth {
    int sample_rate;

    /* Voss-McCartney pink noise generator */
    unsigned int pink_counter;
    float pink_octaves[PINK_OCTAVES];
    float pink_running_sum;

    /* Vocal tract bandpass filter (2nd order Butterworth, two cascaded biquads) */
    BiquadFilter bp_lo; /* highpass at 200Hz */
    BiquadFilter bp_hi; /* lowpass at 2000Hz */

    /* PRNG state (xorshift64) */
    uint64_t rng_state;
};

/* ── PRNG ─────────────────────────────────────────────── */

static inline float rng_float(BreathSynth *bs)
{
    uint64_t x = bs->rng_state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    bs->rng_state = x;
    return (float)(x & 0xFFFFFF) / (float)0xFFFFFF * 2.0f - 1.0f;
}

/* ── Voss-McCartney pink noise ────────────────────────── */

static float pink_sample(BreathSynth *bs)
{
    unsigned int last = bs->pink_counter;
    bs->pink_counter++;
    unsigned int changed = last ^ bs->pink_counter;

    for (int i = 0; i < PINK_OCTAVES; i++) {
        if (changed & (1u << i)) {
            bs->pink_running_sum -= bs->pink_octaves[i];
            bs->pink_octaves[i] = rng_float(bs);
            bs->pink_running_sum += bs->pink_octaves[i];
        }
    }
    float white = rng_float(bs);
    return (bs->pink_running_sum + white) / (PINK_OCTAVES + 1);
}

/* ── Biquad filter (Direct Form I) ───────────────────── */

static void biquad_init_highpass(BiquadFilter *f, float fc, float sample_rate)
{
    float w0 = 2.0f * M_PI * fc / sample_rate;
    float cosw = cosf(w0);
    float sinw = sinf(w0);
    float alpha = sinw / (2.0f * 0.7071f); /* Q = 1/sqrt(2) for Butterworth */
    float a0 = 1.0f + alpha;
    f->b0 = ((1.0f + cosw) / 2.0f) / a0;
    f->b1 = -(1.0f + cosw) / a0;
    f->b2 = ((1.0f + cosw) / 2.0f) / a0;
    f->a1 = (-2.0f * cosw) / a0;
    f->a2 = (1.0f - alpha) / a0;
    memset(f->biquad_state, 0, sizeof(f->biquad_state));
}

static void biquad_init_lowpass(BiquadFilter *f, float fc, float sample_rate)
{
    float w0 = 2.0f * M_PI * fc / sample_rate;
    float cosw = cosf(w0);
    float sinw = sinf(w0);
    float alpha = sinw / (2.0f * 0.7071f);
    float a0 = 1.0f + alpha;
    f->b0 = ((1.0f - cosw) / 2.0f) / a0;
    f->b1 = (1.0f - cosw) / a0;
    f->b2 = ((1.0f - cosw) / 2.0f) / a0;
    f->a1 = (-2.0f * cosw) / a0;
    f->a2 = (1.0f - alpha) / a0;
    memset(f->biquad_state, 0, sizeof(f->biquad_state));
}

static inline float biquad_process(BiquadFilter *f, float x)
{
    float y = f->b0 * x + f->b1 * f->biquad_state[0] + f->b2 * f->biquad_state[1]
            - f->a1 * f->biquad_state[2] - f->a2 * f->biquad_state[3];
    f->biquad_state[1] = f->biquad_state[0];
    f->biquad_state[0] = x;
    f->biquad_state[3] = f->biquad_state[2];
    f->biquad_state[2] = y;
    return y;
}

/* ── Public API ───────────────────────────────────────── */

BreathSynth *breath_create(int sample_rate)
{
    BreathSynth *bs = calloc(1, sizeof(BreathSynth));
    if (!bs) return NULL;

    bs->sample_rate = sample_rate;
    bs->rng_state = 0xDEADBEEF12345678ULL;
    bs->pink_counter = 0;
    bs->pink_running_sum = 0.0f;
    memset(bs->pink_octaves, 0, sizeof(bs->pink_octaves));

    biquad_init_highpass(&bs->bp_lo, 200.0f, (float)sample_rate);
    biquad_init_lowpass(&bs->bp_hi, 2000.0f, (float)sample_rate);

    /* Warm up filters to avoid startup transient */
    for (int i = 0; i < 1024; i++) {
        float s = pink_sample(bs);
        s = biquad_process(&bs->bp_lo, s);
        biquad_process(&bs->bp_hi, s);
    }

    return bs;
}

void breath_destroy(BreathSynth *bs)
{
    free(bs);
}

void breath_generate(BreathSynth *bs, float *audio, int n_samples, float amplitude)
{
    if (!bs || !audio || n_samples <= 0) return;

    /* ADSR envelope: 10% attack, 20% sustain, 70% decay */
    int attack  = n_samples / 10;
    int sustain = n_samples / 5;
    int decay   = n_samples - attack - sustain;

    for (int i = 0; i < n_samples; i++) {
        float noise = pink_sample(bs);
        noise = biquad_process(&bs->bp_lo, noise);
        noise = biquad_process(&bs->bp_hi, noise);

        float env;
        if (i < attack) {
            env = (float)i / (float)attack;
            env = env * env; /* quadratic attack for soft onset */
        } else if (i < attack + sustain) {
            env = 1.0f;
        } else {
            float t = (float)(i - attack - sustain) / (float)decay;
            env = 1.0f - t * t; /* quadratic decay */
        }

        audio[i] += noise * amplitude * env;
    }
}

void breath_micropause(float *audio, int n_samples, float fade_ms, int sample_rate)
{
    if (!audio || n_samples <= 0) return;

    int fade_samples = (int)(fade_ms * 0.001f * (float)sample_rate);
    if (fade_samples > n_samples / 2) fade_samples = n_samples / 2;

#ifdef __APPLE__
    /* Use vDSP for vectorized ramp generation */
    if (fade_samples > 0) {
        float *ramp = (float *)malloc((size_t)fade_samples * sizeof(float));
        if (ramp) {
            /* Fade-out ramp: 1.0 → 0.0 */
            float start = 1.0f, step = -1.0f / (float)fade_samples;
            vDSP_vramp(&start, &step, ramp, 1, (vDSP_Length)fade_samples);
            vDSP_vmul(audio, 1, ramp, 1, audio, 1, (vDSP_Length)fade_samples);

            /* Zero the middle */
            int mid = n_samples - 2 * fade_samples;
            if (mid > 0)
                vDSP_vclr(audio + fade_samples, 1, (vDSP_Length)mid);

            /* Fade-in ramp: 0.0 → 1.0 */
            start = 0.0f;
            step = 1.0f / (float)fade_samples;
            vDSP_vramp(&start, &step, ramp, 1, (vDSP_Length)fade_samples);
            int offset = n_samples - fade_samples;
            vDSP_vmul(audio + offset, 1, ramp, 1, audio + offset, 1,
                       (vDSP_Length)fade_samples);

            free(ramp);
            return;
        }
    }
#endif

    /* Scalar fallback */
    for (int i = 0; i < n_samples; i++) {
        float env;
        if (i < fade_samples) {
            env = 1.0f - (float)i / (float)fade_samples;
        } else if (i >= n_samples - fade_samples) {
            env = (float)(i - (n_samples - fade_samples)) / (float)fade_samples;
        } else {
            env = 0.0f;
        }
        audio[i] *= env;
    }
}

void breath_sentence_gap(BreathSynth *bs, float *out, int n_samples, float speech_rms)
{
    if (!bs || !out || n_samples <= 0) return;

    /* Target breath amplitude: -30dB below speech RMS, clamped */
    float amp = speech_rms * 0.031f; /* 10^(-30/20) ≈ 0.031 */
    if (amp < 0.001f) amp = 0.001f;
    if (amp > 0.1f) amp = 0.1f;

    memset(out, 0, (size_t)n_samples * sizeof(float));
    breath_generate(bs, out, n_samples, amp);
}
