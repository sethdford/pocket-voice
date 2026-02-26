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

#ifdef __APPLE__
    /* vDSP-accelerated 2-section biquad for batch filtering */
    vDSP_biquad_Setup bq_setup;
    float bq_delay[2 * 2 + 2]; /* 2 sections: 2*M+2 floats */
#endif

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

#ifdef __APPLE__
    /* Create vDSP 2-section biquad: section 0 = highpass, section 1 = lowpass */
    double coeffs[10] = {
        bs->bp_lo.b0, bs->bp_lo.b1, bs->bp_lo.b2, bs->bp_lo.a1, bs->bp_lo.a2,
        bs->bp_hi.b0, bs->bp_hi.b1, bs->bp_hi.b2, bs->bp_hi.a1, bs->bp_hi.a2,
    };
    bs->bq_setup = vDSP_biquad_CreateSetup(coeffs, 2);
    memset(bs->bq_delay, 0, sizeof(bs->bq_delay));
#endif

    /* Warm up filters to avoid startup transient */
    for (int i = 0; i < 1024; i++) {
        float s = pink_sample(bs);
        s = biquad_process(&bs->bp_lo, s);
        biquad_process(&bs->bp_hi, s);
    }
#ifdef __APPLE__
    /* Warm up vDSP biquad state in parallel */
    {
        float warmup[1024];
        for (int i = 0; i < 1024; i++) warmup[i] = pink_sample(bs);
        if (bs->bq_setup)
            vDSP_biquad(bs->bq_setup, bs->bq_delay, warmup, 1, warmup, 1, 1024);
    }
#endif

    return bs;
}

void breath_destroy(BreathSynth *bs)
{
    if (!bs) return;
#ifdef __APPLE__
    if (bs->bq_setup) vDSP_biquad_DestroySetup(bs->bq_setup);
#endif
    free(bs);
}

void breath_generate(BreathSynth *bs, float *audio, int n_samples, float amplitude)
{
    if (!bs || !audio || n_samples <= 0) return;

    int attack  = n_samples / 10;
    int sustain = n_samples / 5;
    int decay   = n_samples - attack - sustain;
    if (attack < 1) attack = 1;
    if (decay < 1) decay = 0;
    /* Clamp so segments don't exceed buffer: prioritize attack over sustain/decay */
    if (attack > n_samples) { attack = n_samples; sustain = 0; decay = 0; }
    else if (attack + decay > n_samples) { decay = n_samples - attack; sustain = 0; }
    else if (attack + sustain + decay > n_samples) { sustain = n_samples - attack - decay; }

    float *noise = (float *)malloc((size_t)n_samples * sizeof(float));
    float *env   = (float *)malloc((size_t)n_samples * sizeof(float));
    if (!noise || !env) {
        free(noise); free(env);
        return;
    }

    /* 1. Generate pink noise batch (serial — state-dependent PRNG) */
    for (int i = 0; i < n_samples; i++)
        noise[i] = pink_sample(bs);

#ifdef __APPLE__
    /* 2. Batch filter via vDSP 2-section biquad (replaces 2*n scalar calls) */
    if (bs->bq_setup) {
        vDSP_biquad(bs->bq_setup, bs->bq_delay, noise, 1, noise, 1, n_samples);
    } else
#endif
    {
        for (int i = 0; i < n_samples; i++) {
            noise[i] = biquad_process(&bs->bp_lo, noise[i]);
            noise[i] = biquad_process(&bs->bp_hi, noise[i]);
        }
    }

#ifdef __APPLE__
    /* 3. Build envelope with vDSP: quadratic attack, flat sustain, quadratic decay */
    {
        /* Attack: (t/attack)^2 for t in [0, attack) */
        float start = 0.0f, step = 1.0f / (float)attack;
        vDSP_vramp(&start, &step, env, 1, attack);
        vDSP_vsq(env, 1, env, 1, attack);

        /* Sustain: 1.0 */
        float one = 1.0f;
        vDSP_vfill(&one, env + attack, 1, sustain);

        /* Decay: 1 - t^2 */
        start = 0.0f; step = 1.0f / (float)decay;
        float *dec = env + attack + sustain;
        vDSP_vramp(&start, &step, dec, 1, decay);
        vDSP_vsq(dec, 1, dec, 1, decay);
        float neg = -1.0f;
        vDSP_vsmsa(dec, 1, &neg, &one, dec, 1, decay); /* 1 - t^2 */
    }

    /* 4. Combine: audio += noise * amplitude * env */
    vDSP_vmul(noise, 1, env, 1, noise, 1, n_samples);
    vDSP_vsma(noise, 1, &amplitude, audio, 1, audio, 1, n_samples);
#else
    /* Scalar fallback */
    for (int i = 0; i < n_samples; i++) {
        float e;
        if (i < attack) {
            e = (float)i / (float)attack;
            e = e * e;
        } else if (i < attack + sustain) {
            e = 1.0f;
        } else {
            float t = (float)(i - attack - sustain) / (float)decay;
            e = 1.0f - t * t;
        }
        audio[i] += noise[i] * amplitude * e;
    }
#endif

    free(noise);
    free(env);
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
