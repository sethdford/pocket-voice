/**
 * audio_mixer.c — Multi-source audio mixer with priority, ducking, and crossfade.
 *
 * Blends TTS, backchannel, pre-synthesized cache, and cloud audio for
 * full-duplex speaker output. Lock-free SPSC ring buffers per channel.
 * Uses vDSP for vector operations and ARM NEON for the soft limiter.
 */

#include "audio_mixer.h"
#include <stdlib.h>
#include <string.h>
#include <stdatomic.h>
#include <Accelerate/Accelerate.h>

#if defined(__arm64__) || defined(__aarch64__)
#include <arm_neon.h>
#endif

#define MIXER_RING_SIZE 48000  /* 2 seconds at 24kHz, power-of-two */
#define DEFAULT_DUCKING_GAIN 0.3f
#define DEFAULT_CROSSFADE 240  /* 10ms at 24kHz */
#define DEFAULT_BLOCK 480      /* 20ms at 24kHz */

/* Default priorities: higher = dominant, less likely to be ducked */
#define PRIO_MAIN 10
#define PRIO_CLOUD 9
#define PRIO_PRESYNTH 8
#define PRIO_BACKCHANNEL 5

/* Per-channel SPSC ring buffer */
typedef struct {
    float *buffer;
    uint32_t size;
    uint32_t mask;
    _Alignas(64) _Atomic uint64_t head;
    _Alignas(64) _Atomic uint64_t tail;
} SPSCRing;

struct AudioMixer {
    AudioMixerConfig config;
    SPSCRing channels[MIX_CHANNEL_COUNT];
    float gain[MIX_CHANNEL_COUNT];
    int priority[MIX_CHANNEL_COUNT];
    float fade_state[MIX_CHANNEL_COUNT];   /* 0.0 = silent, 1.0 = full */
    float fade_step;                        /* per-sample fade increment */
    /* Pre-allocated work buffers (zero allocations in hot path) */
    float *block_buf;       /* [block_size] per channel */
    float *mix_buf;         /* [block_size] accumulated mix */
    int block_size;
    int crossfade_samples;
    float ducking_gain;
};

static uint32_t spsc_available_read(const SPSCRing *r) {
    uint64_t h = atomic_load_explicit(&r->head, memory_order_acquire);
    uint64_t t = atomic_load_explicit(&r->tail, memory_order_relaxed);
    return (uint32_t)(h - t);
}

static uint32_t spsc_available_write(const SPSCRing *r) {
    return r->size - spsc_available_read(r);
}

/* Read up to `count` samples. Returns actual count read (0 if empty). */
static uint32_t spsc_read(SPSCRing *r, float *out, uint32_t count) {
    uint32_t avail = spsc_available_read(r);
    if (avail == 0) return 0;
    if (count > avail) count = avail;

    uint64_t t = atomic_load_explicit(&r->tail, memory_order_relaxed);
    uint32_t offset = (uint32_t)(t & r->mask);

    if (offset + count <= r->size) {
        memcpy(out, r->buffer + offset, count * sizeof(float));
    } else {
        uint32_t first = r->size - offset;
        memcpy(out, r->buffer + offset, first * sizeof(float));
        memcpy(out + first, r->buffer, (count - first) * sizeof(float));
    }
    atomic_store_explicit(&r->tail, t + count, memory_order_release);
    return count;
}

/* Write samples. Drops if no space. Returns samples written. */
static uint32_t spsc_write(SPSCRing *r, const float *data, uint32_t count) {
    uint32_t space = spsc_available_write(r);
    if (space < count) count = space;
    if (count == 0) return 0;

    uint64_t h = atomic_load_explicit(&r->head, memory_order_relaxed);
    uint32_t offset = (uint32_t)(h & r->mask);

    if (offset + count <= r->size) {
        memcpy(r->buffer + offset, data, count * sizeof(float));
    } else {
        uint32_t first = r->size - offset;
        memcpy(r->buffer + offset, data, first * sizeof(float));
        memcpy(r->buffer, data + first, (count - first) * sizeof(float));
    }
    atomic_store_explicit(&r->head, h + count, memory_order_release);
    return count;
}

static void spsc_flush(SPSCRing *r) {
    uint64_t h = atomic_load_explicit(&r->head, memory_order_acquire);
    atomic_store_explicit(&r->tail, h, memory_order_release);
}

static uint32_t next_pow2(uint32_t x) {
    x--;
    x |= x >> 1; x |= x >> 2; x |= x >> 4; x |= x >> 8; x |= x >> 16;
    return x + 1;
}

static float soft_limit_scalar(float x) {
    if (x > 1.5f) return 1.0f;
    if (x < -1.5f) return -1.0f;
    return x - (x * x * x) / 6.75f;
}

#if defined(__arm64__) || defined(__aarch64__)
/* NEON-accelerated soft limiter: cubic soft clip */
static void soft_limit_neon(float *buf, int n) {
    const float32x4_t one = vdupq_n_f32(1.0f);
    const float32x4_t neg_one = vdupq_n_f32(-1.0f);
    const float32x4_t thresh = vdupq_n_f32(1.5f);
    const float32x4_t neg_thresh = vdupq_n_f32(-1.5f);
    const float32x4_t denom = vdupq_n_f32(6.75f);

    int i = 0;
    for (; i + 8 <= n; i += 8) {
        float32x4_t a = vld1q_f32(buf + i);
        float32x4_t b = vld1q_f32(buf + i + 4);
        /* x3 = x * x * x */
        float32x4_t a3 = vmulq_f32(vmulq_f32(a, a), a);
        float32x4_t b3 = vmulq_f32(vmulq_f32(b, b), b);
        /* x - x3/6.75 */
        float32x4_t out_a = vsubq_f32(a, vdivq_f32(a3, denom));
        float32x4_t out_b = vsubq_f32(b, vdivq_f32(b3, denom));
        /* Hard clip for |x| > 1.5 */
        uint32x4_t gt = vcgtq_f32(a, thresh);
        uint32x4_t lt = vcltq_f32(a, neg_thresh);
        out_a = vbslq_f32(gt, one, out_a);
        out_a = vbslq_f32(lt, neg_one, out_a);
        gt = vcgtq_f32(b, thresh);
        lt = vcltq_f32(b, neg_thresh);
        out_b = vbslq_f32(gt, one, out_b);
        out_b = vbslq_f32(lt, neg_one, out_b);
        vst1q_f32(buf + i, out_a);
        vst1q_f32(buf + i + 4, out_b);
    }
    for (; i + 4 <= n; i += 4) {
        float32x4_t a = vld1q_f32(buf + i);
        float32x4_t a3 = vmulq_f32(vmulq_f32(a, a), a);
        float32x4_t out_a = vsubq_f32(a, vdivq_f32(a3, denom));
        uint32x4_t gt = vcgtq_f32(a, thresh);
        uint32x4_t lt = vcltq_f32(a, neg_thresh);
        out_a = vbslq_f32(gt, one, out_a);
        out_a = vbslq_f32(lt, neg_one, out_a);
        vst1q_f32(buf + i, out_a);
    }
    for (; i < n; i++) buf[i] = soft_limit_scalar(buf[i]);
}
#else
static void soft_limit_neon(float *buf, int n) {
    for (int i = 0; i < n; i++) buf[i] = soft_limit_scalar(buf[i]);
}
#endif

AudioMixer *audio_mixer_create(const AudioMixerConfig *cfg) {
    if (!cfg || cfg->sample_rate <= 0 || cfg->block_size <= 0) return NULL;

    AudioMixer *m = calloc(1, sizeof(AudioMixer));
    if (!m) return NULL;

    m->config = *cfg;
    m->block_size = cfg->block_size > 0 ? cfg->block_size : DEFAULT_BLOCK;
    m->crossfade_samples = cfg->crossfade_samples > 0 ? cfg->crossfade_samples : DEFAULT_CROSSFADE;
    m->ducking_gain = cfg->ducking_gain >= 0.0f ? cfg->ducking_gain : DEFAULT_DUCKING_GAIN;
    m->fade_step = (m->crossfade_samples > 0) ? (1.0f / (float)m->crossfade_samples) : 1.0f;

    uint32_t ring_size = next_pow2((uint32_t)MIXER_RING_SIZE);
    if (ring_size < (uint32_t)m->block_size) ring_size = next_pow2((uint32_t)m->block_size);

    for (int i = 0; i < MIX_CHANNEL_COUNT; i++) {
        m->channels[i].buffer = calloc(ring_size, sizeof(float));
        if (!m->channels[i].buffer) goto fail;
        m->channels[i].size = ring_size;
        m->channels[i].mask = ring_size - 1;
        atomic_store(&m->channels[i].head, 0);
        atomic_store(&m->channels[i].tail, 0);
        m->gain[i] = 1.0f;
        m->fade_state[i] = 0.0f;
    }
    m->priority[MIX_CHANNEL_MAIN] = PRIO_MAIN;
    m->priority[MIX_CHANNEL_BACKCHANNEL] = PRIO_BACKCHANNEL;
    m->priority[MIX_CHANNEL_PRESYNTHESIZED] = PRIO_PRESYNTH;
    m->priority[MIX_CHANNEL_CLOUD_AUDIO] = PRIO_CLOUD;

    /* Work buffers: one block per channel + one mix accumulator */
    m->block_buf = calloc((size_t)MIX_CHANNEL_COUNT * m->block_size, sizeof(float));
    m->mix_buf = calloc((size_t)m->block_size, sizeof(float));
    if (!m->block_buf || !m->mix_buf) goto fail;

    return m;
fail:
    for (int i = 0; i < MIX_CHANNEL_COUNT; i++) free(m->channels[i].buffer);
    free(m->block_buf);
    free(m->mix_buf);
    free(m);
    return NULL;
}

void audio_mixer_destroy(AudioMixer *mixer) {
    if (!mixer) return;
    for (int i = 0; i < MIX_CHANNEL_COUNT; i++) free(mixer->channels[i].buffer);
    free(mixer->block_buf);
    free(mixer->mix_buf);
    free(mixer);
}

int audio_mixer_write(AudioMixer *mixer, MixChannel channel,
                     const float *pcm, int n_samples) {
    if (!mixer || !pcm || n_samples <= 0 || channel < 0 || channel >= MIX_CHANNEL_COUNT)
        return 0;
    uint32_t n = (uint32_t)n_samples;
    uint32_t written = spsc_write(&mixer->channels[channel], pcm, n);
    return (int)written;
}

/* Find highest-priority channel that has pending audio */
static int find_highest_priority_active(const AudioMixer *mixer, int *active_mask) {
    int best = -1;
    int best_prio = -1;
    for (int c = 0; c < MIX_CHANNEL_COUNT; c++) {
        if (!(active_mask[c])) continue;
        if (mixer->priority[c] > best_prio) {
            best_prio = mixer->priority[c];
            best = c;
        }
    }
    return best;
}

/* Mix one block of block_size samples */
static void mix_one_block(AudioMixer *mixer, float *out) {
    const int block = mixer->block_size;
    int active[MIX_CHANNEL_COUNT];

    for (int c = 0; c < MIX_CHANNEL_COUNT; c++) {
        uint32_t avail = spsc_available_read(&mixer->channels[c]);
        active[c] = (avail > 0);
    }

    int highest = find_highest_priority_active(mixer, active);

    memset(mixer->mix_buf, 0, (size_t)block * sizeof(float));

    for (int c = 0; c < MIX_CHANNEL_COUNT; c++) {
        if (!active[c]) {
            /* Fade out when going silent */
            if (mixer->fade_state[c] > 0.0f) {
                mixer->fade_state[c] -= mixer->fade_step * (float)block;
                if (mixer->fade_state[c] < 0.0f) mixer->fade_state[c] = 0.0f;
            }
            continue;
        }

        float *chan_buf = mixer->block_buf + (size_t)c * block;
        uint32_t got = spsc_read(&mixer->channels[c], chan_buf, (uint32_t)block);
        if (got < (uint32_t)block)
            memset(chan_buf + got, 0, (size_t)(block - got) * sizeof(float));

        /* Fade in when becoming active */
        if (mixer->fade_state[c] < 1.0f) {
            mixer->fade_state[c] += mixer->fade_step * (float)got;
            if (mixer->fade_state[c] > 1.0f) mixer->fade_state[c] = 1.0f;
        }

        float effective_gain = mixer->gain[c] * mixer->fade_state[c];

        /* Ducking: if this channel is NOT highest priority and highest is active, reduce gain */
        if (highest >= 0 && c != highest)
            effective_gain *= mixer->ducking_gain;

        vDSP_vsmul(chan_buf, 1, &effective_gain, chan_buf, 1, (vDSP_Length)block);

        /* Add to mix */
        vDSP_vadd(mixer->mix_buf, 1, chan_buf, 1, mixer->mix_buf, 1, (vDSP_Length)block);
    }

    soft_limit_neon(mixer->mix_buf, block);
    memcpy(out, mixer->mix_buf, (size_t)block * sizeof(float));
}

int audio_mixer_read(AudioMixer *mixer, float *out_pcm, int max_samples) {
    if (!mixer || !out_pcm || max_samples <= 0) return 0;

    int block = mixer->block_size;
    int total = 0;

    while (total + block <= max_samples) {
        /* Only produce output if at least one channel has data */
        if (!audio_mixer_any_active(mixer)) break;
        mix_one_block(mixer, out_pcm + total);
        total += block;
    }
    return total;
}

int audio_mixer_channel_active(const AudioMixer *mixer, MixChannel channel) {
    if (!mixer || channel < 0 || channel >= MIX_CHANNEL_COUNT) return 0;
    return spsc_available_read(&mixer->channels[channel]) > 0;
}

int audio_mixer_any_active(const AudioMixer *mixer) {
    if (!mixer) return 0;
    for (int c = 0; c < MIX_CHANNEL_COUNT; c++) {
        if (spsc_available_read(&mixer->channels[c]) > 0) return 1;
    }
    return 0;
}

void audio_mixer_set_gain(AudioMixer *mixer, MixChannel channel, float gain) {
    if (!mixer || channel < 0 || channel >= MIX_CHANNEL_COUNT) return;
    if (gain < 0.0f) gain = 0.0f;
    if (gain > 1.0f) gain = 1.0f;
    mixer->gain[channel] = gain;
}

void audio_mixer_set_priority(AudioMixer *mixer, MixChannel channel, int priority) {
    if (!mixer || channel < 0 || channel >= MIX_CHANNEL_COUNT) return;
    mixer->priority[channel] = priority;
}

void audio_mixer_flush(AudioMixer *mixer, MixChannel channel) {
    if (!mixer || channel < 0 || channel >= MIX_CHANNEL_COUNT) return;
    spsc_flush(&mixer->channels[channel]);
    mixer->fade_state[channel] = 0.0f;
}

void audio_mixer_flush_all(AudioMixer *mixer) {
    if (!mixer) return;
    for (int c = 0; c < MIX_CHANNEL_COUNT; c++) audio_mixer_flush(mixer, (MixChannel)c);
}

void audio_mixer_reset(AudioMixer *mixer) {
    if (!mixer) return;
    audio_mixer_flush_all(mixer);
    for (int c = 0; c < MIX_CHANNEL_COUNT; c++) mixer->gain[c] = 1.0f;
    mixer->priority[MIX_CHANNEL_MAIN] = PRIO_MAIN;
    mixer->priority[MIX_CHANNEL_BACKCHANNEL] = PRIO_BACKCHANNEL;
    mixer->priority[MIX_CHANNEL_PRESYNTHESIZED] = PRIO_PRESYNTH;
    mixer->priority[MIX_CHANNEL_CLOUD_AUDIO] = PRIO_CLOUD;
}

int audio_mixer_pending(const AudioMixer *mixer, MixChannel channel) {
    if (!mixer || channel < 0 || channel >= MIX_CHANNEL_COUNT) return 0;
    return (int)spsc_available_read(&mixer->channels[channel]);
}
