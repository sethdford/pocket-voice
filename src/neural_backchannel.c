/**
 * neural_backchannel.c — Neural backchannel synthesis via Sonata TTS or pink noise fallback.
 *
 * Replaces pre-recorded WAV backchannel clips with neural generation using the
 * Sonata TTS pipeline. When no TTS engine is available, falls back to breath-like
 * pink noise synthesis (similar to breath_synthesis.c).
 *
 * Weak-links to TTS synthesis so it compiles independently without Sonata.
 */

#include "neural_backchannel.h"
#include "breath_synthesis.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

#ifndef ACCELERATE_NEW_LAPACK
#define ACCELERATE_NEW_LAPACK
#endif
#include <Accelerate/Accelerate.h>

/* ── Weak-linked TTS synthesis (pipeline provides strong symbol when linked) ── */

/* Default stub returns -1 (use fallback). Pipeline overrides with strong symbol. */
__attribute__((weak)) int nbc_tts_synthesize(void *engine, const char *text, float *out,
    int max_samples, const float *speaker_emb, int spk_dim, int emotion_id) {
    (void)engine;
    (void)text;
    (void)out;
    (void)max_samples;
    (void)speaker_emb;
    (void)spk_dim;
    (void)emotion_id;
    return -1;
}

/* Text mapping for each backchannel type */
static const char *NBC_TEXTS[NBC_COUNT] = {
    [NBC_MHM]     = "mhm",
    [NBC_YEAH]    = "yeah",
    [NBC_RIGHT]   = "right",
    [NBC_OKAY]    = "okay",
    [NBC_UH_HUH]  = "uh huh",
    [NBC_I_SEE]   = "I see",
    [NBC_SURE]    = "sure",
    [NBC_HMHM]    = "hmhm",
    [NBC_LAUGH]   = "[laugh]",
};

#define NBC_DEFAULT_SR     24000
#define NBC_MAX_SAMPLES    12000   /* 500ms at 24kHz */

struct NeuralBackchannel {
    NBCConfig cfg;
    void     *tts_engine;

    /* Speaker / emotion state */
    float    *speaker_embedding;
    int       speaker_dim;
    int       emotion_id;

    /* WAV override: custom loaded audio per type */
    float    *wav_audio[NBC_COUNT];
    int       wav_len[NBC_COUNT];

    /* Neural/TTS cache */
    float    *cache[NBC_COUNT];
    int       cache_len[NBC_COUNT];

    /* Fallback: breath synth for pink noise when no TTS */
    BreathSynth *breath;
};

/* Breathy "mhm"-like synthesis using pink noise (fallback when no TTS) */
static void synth_fallback_mhm(float *out, int len, int sr) {
    float dt = 1.0f / (float)sr;
    for (int i = 0; i < len; i++) {
        float t = (float)i * dt;
        float env = sinf((float)M_PI * t / (len * dt));
        env *= env;
        float s = 0.6f * sinf(2.0f * (float)M_PI * 270.0f * t)
                + 0.2f * sinf(2.0f * (float)M_PI * 540.0f * t)
                + 0.1f * sinf(2.0f * (float)M_PI * 2000.0f * t);
        float glide = 1.0f - 0.05f * t / (len * dt);
        out[i] = s * glide * env * 0.15f;
    }
}

/* Two-syllable "uh huh" fallback */
static void synth_fallback_uh_huh(float *out, int len, int sr) {
    float dt = 1.0f / (float)sr;
    int half = len / 2;
    for (int i = 0; i < len; i++) {
        float t = (float)i * dt;
        float local_t = (i < half) ? t : (t - half * dt);
        float local_len = (i < half) ? half * dt : (len - half) * dt;
        float env = sinf((float)M_PI * local_t / local_len);
        env *= env;
        float f0 = (i < half) ? 180.0f : 220.0f;
        float s = 0.5f * sinf(2.0f * (float)M_PI * f0 * t)
                + 0.3f * sinf(2.0f * (float)M_PI * f0 * 2.0f * t);
        out[i] = s * env * 0.12f;
    }
}

/* Breath-like pink noise backchannel (generic fallback) */
static void synth_fallback_breathy(NeuralBackchannel *nbc, float *out, int len, NBCType type) {
    if (!nbc->breath || !out || len <= 0) return;
    memset(out, 0, (size_t)len * sizeof(float));
    float amp = 0.08f;
    if (type == NBC_LAUGH) amp = 0.12f;
    breath_generate(nbc->breath, out, len, amp);
}

/* Generate fallback audio for a type (no TTS) */
static int generate_fallback(NeuralBackchannel *nbc, NBCType type, float *out, int max_samples) {
    int sr = nbc->cfg.sample_rate > 0 ? nbc->cfg.sample_rate : NBC_DEFAULT_SR;
    int max_ms = nbc->cfg.max_duration_ms > 0 ? nbc->cfg.max_duration_ms : 500;
    int len = (max_ms * sr) / 1000;
    if (len > max_samples) len = max_samples;

    switch (type) {
        case NBC_MHM:
        case NBC_YEAH:
        case NBC_RIGHT:
        case NBC_HMHM:
            synth_fallback_mhm(out, len, sr);
            if (type == NBC_YEAH) {
                for (int i = 0; i < len; i++) out[i] *= 1.1f;
            } else if (type == NBC_RIGHT) {
                len = len * 3 / 4;
            }
            break;
        case NBC_OKAY:
        case NBC_UH_HUH:
            synth_fallback_uh_huh(out, len, sr);
            break;
        default:
            synth_fallback_breathy(nbc, out, len, type);
            break;
    }
    return len;
}

NeuralBackchannel *nbc_create(const NBCConfig *cfg, void *tts_engine) {
    NeuralBackchannel *nbc = (NeuralBackchannel *)calloc(1, sizeof(NeuralBackchannel));
    if (!nbc) return NULL;

    if (cfg) {
        nbc->cfg = *cfg;
    } else {
        nbc->cfg.sample_rate = NBC_DEFAULT_SR;
        nbc->cfg.max_duration_ms = 500;
        nbc->cfg.cache_enabled = 0;
    }
    if (nbc->cfg.sample_rate <= 0) nbc->cfg.sample_rate = NBC_DEFAULT_SR;
    if (nbc->cfg.max_duration_ms <= 0) nbc->cfg.max_duration_ms = 500;

    nbc->tts_engine = tts_engine;
    nbc->breath = breath_create(nbc->cfg.sample_rate);
    if (!nbc->breath) {
        free(nbc);
        return NULL;
    }

    for (int i = 0; i < NBC_COUNT; i++) {
        nbc->cache[i] = NULL;
        nbc->cache_len[i] = 0;
        nbc->wav_audio[i] = NULL;
        nbc->wav_len[i] = 0;
    }

    return nbc;
}

void nbc_destroy(NeuralBackchannel *nbc) {
    if (!nbc) return;
    breath_destroy(nbc->breath);
    free(nbc->speaker_embedding);
    for (int i = 0; i < NBC_COUNT; i++) {
        free(nbc->cache[i]);
        free(nbc->wav_audio[i]);
    }
    free(nbc);
}

static int do_generate(NeuralBackchannel *nbc, NBCType type, float *out_pcm, int max_samples) {
    if (!nbc || type < 0 || type >= NBC_COUNT || !out_pcm || max_samples <= 0) return -1;

    /* WAV override takes precedence */
    if (nbc->wav_audio[type]) {
        int n = nbc->wav_len[type];
        if (n > max_samples) n = max_samples;
        memcpy(out_pcm, nbc->wav_audio[type], (size_t)n * sizeof(float));
        return n;
    }

    /* Try TTS synthesis if engine available (weak symbol; pipeline overrides) */
    if (nbc->tts_engine) {
        int n = nbc_tts_synthesize(nbc->tts_engine, NBC_TEXTS[type], out_pcm, max_samples,
                                  nbc->speaker_embedding, nbc->speaker_dim, nbc->emotion_id);
        if (n > 0) return n;
    }

    /* Fallback: pink noise / breath synthesis */
    return generate_fallback(nbc, type, out_pcm, max_samples);
}

int nbc_warm_cache(NeuralBackchannel *nbc) {
    if (!nbc) return -1;

    int sr = nbc->cfg.sample_rate;
    int max_samples = (nbc->cfg.max_duration_ms * sr) / 1000;
    if (max_samples <= 0 || max_samples > NBC_MAX_SAMPLES) max_samples = NBC_MAX_SAMPLES;

    for (int i = 0; i < NBC_COUNT; i++) {
        free(nbc->cache[i]);
        nbc->cache[i] = (float *)malloc((size_t)max_samples * sizeof(float));
        if (!nbc->cache[i]) continue;
        int n = do_generate(nbc, (NBCType)i, nbc->cache[i], max_samples);
        nbc->cache_len[i] = (n > 0) ? n : 0;
        if (n <= 0) {
            free(nbc->cache[i]);
            nbc->cache[i] = NULL;
        }
    }
    return 0;
}

const float *nbc_get_cached(NeuralBackchannel *nbc, NBCType type, int *out_len) {
    if (!nbc || type < 0 || type >= NBC_COUNT || !out_len) return NULL;
    *out_len = 0;
    if (nbc->cache[type] && nbc->cache_len[type] > 0) {
        *out_len = nbc->cache_len[type];
        return nbc->cache[type];
    }
    return NULL;
}

int nbc_generate(NeuralBackchannel *nbc, NBCType type, float *out_pcm, int max_samples) {
    return do_generate(nbc, type, out_pcm, max_samples);
}

int nbc_set_speaker(NeuralBackchannel *nbc, const float *embedding, int dim) {
    if (!nbc) return -1;
    free(nbc->speaker_embedding);
    nbc->speaker_embedding = NULL;
    nbc->speaker_dim = 0;
    if (embedding && dim > 0) {
        nbc->speaker_embedding = (float *)malloc((size_t)dim * sizeof(float));
        if (!nbc->speaker_embedding) return -1;
        memcpy(nbc->speaker_embedding, embedding, (size_t)dim * sizeof(float));
        nbc->speaker_dim = dim;
    }
    if (nbc->cfg.cache_enabled)
        nbc_warm_cache(nbc);
    return 0;
}

void nbc_set_emotion(NeuralBackchannel *nbc, int emotion_id) {
    if (nbc) nbc->emotion_id = emotion_id;
}

int nbc_load_wav(NeuralBackchannel *nbc, NBCType type, const char *wav_path) {
    if (!nbc || type < 0 || type >= NBC_COUNT || !wav_path) return -1;

    FILE *f = fopen(wav_path, "rb");
    if (!f) return -1;

    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    fseek(f, 44, SEEK_SET);  /* Skip WAV header */

    long data_bytes = size - 44;
    if (data_bytes <= 0) { fclose(f); return -1; }

    int n_samples = (int)(data_bytes / sizeof(int16_t));
    int16_t *raw = (int16_t *)malloc(data_bytes);
    if (!raw) { fclose(f); return -1; }
    size_t got = fread(raw, 1, data_bytes, f);
    fclose(f);
    if (got != (size_t)data_bytes) { free(raw); return -1; }

    free(nbc->wav_audio[type]);
    nbc->wav_audio[type] = (float *)malloc((size_t)n_samples * sizeof(float));
    if (!nbc->wav_audio[type]) { free(raw); return -1; }

    for (int i = 0; i < n_samples; i++)
        nbc->wav_audio[type][i] = (float)raw[i] / 32768.0f;
    free(raw);
    nbc->wav_len[type] = n_samples;

    /* Invalidate cache for this type so we use WAV */
    free(nbc->cache[type]);
    nbc->cache[type] = NULL;
    nbc->cache_len[type] = 0;

    return 0;
}
