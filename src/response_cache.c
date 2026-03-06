/**
 * response_cache.c — Pre-synthesized TTS response cache for fast-path intents.
 *
 * Caches TTS audio for common responses (greetings, acknowledgments, thanks,
 * goodbyes) so ROUTE_FAST can deliver sub-50ms responses.
 *
 * Zero allocations in get() hot path. Thread-safe reads after warm.
 */

#include "response_cache.h"
#include "intent_router.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_VARIANTS 8
#define MAX_TEXT_LEN 255
#define MAGIC "RCCH"
#define VERSION 1

static const char *RESPONSE_TEXTS[][4] = {
    [FAST_GREETING]    = { "Hey there!", "Hi!", "Hello!", NULL },
    [FAST_ACKNOWLEDGE] = { "Got it.", "Sure thing.", "Okay, got it.", NULL },
    [FAST_THINKING]    = { "Let me think about that.", "Give me a second.", "Hmm, one moment.", NULL },
    [FAST_YES]         = { "Absolutely.", "Yes!", "Sure, absolutely.", NULL },
    [FAST_NO]          = { "I don't think so.", "No, not really.", "Nope.", NULL },
    [FAST_THANKS]      = { "You're welcome!", "Happy to help!", "Of course!", NULL },
    [FAST_GOODBYE]     = { "See you later!", "Bye!", "Take care!", NULL },
};

typedef struct {
    float *audio;
    int    n_samples;
    char   text[MAX_TEXT_LEN + 1];
} CachedVariant;

struct ResponseCache {
    ResponseCacheConfig cfg;
    CachedVariant variants[FAST_COUNT][MAX_VARIANTS];
    int variant_counts[FAST_COUNT];
    float *speaker_embedding;
    int speaker_dim;
    int max_samples_per_variant;
    unsigned int get_counter;  /* Incrementing counter for variant selection */
};

static int count_texts(int type) {
    int n = 0;
    for (int i = 0; i < 4 && RESPONSE_TEXTS[type][i]; i++) n++;
    return n;
}

ResponseCache *response_cache_create(const ResponseCacheConfig *cfg) {
    if (!cfg || cfg->sample_rate <= 0) return NULL;
    ResponseCache *cache = (ResponseCache *)calloc(1, sizeof(ResponseCache));
    if (!cache) return NULL;
    cache->cfg.sample_rate = cfg->sample_rate;
    cache->cfg.max_variants = cfg->max_variants > 0 ? cfg->max_variants : 3;
    if (cache->cfg.max_variants > MAX_VARIANTS)
        cache->cfg.max_variants = MAX_VARIANTS;
    cache->cfg.max_audio_seconds = cfg->max_audio_seconds > 0 ? cfg->max_audio_seconds : 3;
    cache->max_samples_per_variant = cache->cfg.sample_rate * cache->cfg.max_audio_seconds;
    return cache;
}

void response_cache_destroy(ResponseCache *cache) {
    if (!cache) return;
    response_cache_clear(cache);
    free(cache->speaker_embedding);
    free(cache);
}

static void free_variant(CachedVariant *v) {
    if (v->audio) {
        free(v->audio);
        v->audio = NULL;
    }
    v->n_samples = 0;
    v->text[0] = '\0';
}

void response_cache_clear(ResponseCache *cache) {
    if (!cache) return;
    for (int t = 0; t < FAST_COUNT; t++) {
        for (int v = 0; v < MAX_VARIANTS; v++)
            free_variant(&cache->variants[t][v]);
        cache->variant_counts[t] = 0;
    }
}

int response_cache_warm(ResponseCache *cache,
                        int (*tts_synthesize)(void *ctx, const char *text,
                                             float *out_pcm, int max_samples),
                        void *tts_ctx) {
    if (!cache || !tts_synthesize) return 0;
    float *buf = (float *)malloc((size_t)cache->max_samples_per_variant * sizeof(float));
    if (!buf) return 0;
    int success = 0;
    for (int t = 0; t < FAST_COUNT; t++) {
        int n_texts = count_texts(t);
        int to_cache = n_texts < cache->cfg.max_variants ? n_texts : cache->cfg.max_variants;
        for (int v = 0; v < to_cache && RESPONSE_TEXTS[t][v]; v++) {
            const char *text = RESPONSE_TEXTS[t][v];
            int n = tts_synthesize(tts_ctx, text, buf, cache->max_samples_per_variant);
            if (n > 0 && n <= cache->max_samples_per_variant) {
                float *aud = (float *)malloc((size_t)n * sizeof(float));
                if (aud) {
                    memcpy(aud, buf, (size_t)n * sizeof(float));
                    free_variant(&cache->variants[t][v]);
                    cache->variants[t][v].audio = aud;
                    cache->variants[t][v].n_samples = n;
                    strncpy(cache->variants[t][v].text, text, MAX_TEXT_LEN);
                    cache->variants[t][v].text[MAX_TEXT_LEN] = '\0';
                    if (v >= cache->variant_counts[t])
                        cache->variant_counts[t] = v + 1;
                    success++;
                }
            }
        }
    }
    free(buf);
    return success;
}

const float *response_cache_get(ResponseCache *cache, int fast_type, int *out_len) {
    if (!cache || !out_len) return NULL;
    *out_len = 0;
    if (fast_type < 0 || fast_type >= FAST_COUNT) return NULL;
    int n = cache->variant_counts[fast_type];
    if (n <= 0) return NULL;
    unsigned int idx = cache->get_counter++ % (unsigned int)n;
    CachedVariant *v = &cache->variants[fast_type][idx];
    if (!v->audio) return NULL;
    *out_len = v->n_samples;
    return v->audio;
}

const float *response_cache_get_variant(ResponseCache *cache, int fast_type,
                                       int variant_idx, int *out_len) {
    if (!cache || !out_len) return NULL;
    *out_len = 0;
    if (fast_type < 0 || fast_type >= FAST_COUNT) return NULL;
    if (variant_idx < 0 || variant_idx >= cache->variant_counts[fast_type]) return NULL;
    CachedVariant *v = &cache->variants[fast_type][variant_idx];
    if (!v->audio) return NULL;
    *out_len = v->n_samples;
    return v->audio;
}

int response_cache_add(ResponseCache *cache, int fast_type,
                       const char *text, const float *pcm, int n_samples) {
    if (!cache || !text || !pcm || n_samples <= 0) return -1;
    if (fast_type < 0 || fast_type >= FAST_COUNT) return -1;
    if (n_samples > cache->max_samples_per_variant) return -1;
    int slot = cache->variant_counts[fast_type];
    if (slot >= cache->cfg.max_variants) return -1;
    float *aud = (float *)malloc((size_t)n_samples * sizeof(float));
    if (!aud) return -1;
    memcpy(aud, pcm, (size_t)n_samples * sizeof(float));
    free_variant(&cache->variants[fast_type][slot]);
    cache->variants[fast_type][slot].audio = aud;
    cache->variants[fast_type][slot].n_samples = n_samples;
    strncpy(cache->variants[fast_type][slot].text, text, MAX_TEXT_LEN);
    cache->variants[fast_type][slot].text[MAX_TEXT_LEN] = '\0';
    cache->variant_counts[fast_type] = slot + 1;
    return 0;
}

/* Simple linear resample (no Accelerate). */
static int linear_resample(const float *in, int in_len, int in_sr,
                           float *out, int out_cap, int out_sr) {
    if (in_sr == out_sr) {
        int n = in_len < out_cap ? in_len : out_cap;
        memcpy(out, in, (size_t)n * sizeof(float));
        return n;
    }
    double ratio = (double)in_sr / (double)out_sr;
    int out_len = (int)((double)in_len / ratio);
    if (out_len > out_cap) out_len = out_cap;
    if (out_len <= 0) return 0;
    for (int i = 0; i < out_len; i++) {
        double src_pos = (double)i * ratio;
        int a = (int)src_pos;
        int b = a + 1;
        if (b >= in_len) b = in_len - 1;
        double frac = src_pos - (double)a;
        out[i] = (float)((1.0 - frac) * (double)in[a] + frac * (double)in[b]);
    }
    return out_len;
}

/* Load WAV file into float PCM (mono). Returns n_samples or -1. Caller frees *out. */
static int wav_load_simple(const char *path, float **out, int *out_sr) {
    FILE *f = fopen(path, "rb");
    if (!f) return -1;
    char riff[4];
    uint32_t file_size, fmt_size;
    char wave[4], chunk_id[4];
    uint16_t audio_fmt, channels, bits_per_sample;
    uint32_t sample_rate, chunk_size;

    if (fread(riff, 1, 4, f) != 4 || memcmp(riff, "RIFF", 4) != 0) goto fail;
    if (fread(&file_size, 4, 1, f) != 1) goto fail;
    if (fread(wave, 1, 4, f) != 4 || memcmp(wave, "WAVE", 4) != 0) goto fail;

    /* Find fmt chunk */
    if (fread(chunk_id, 1, 4, f) != 4 || memcmp(chunk_id, "fmt ", 4) != 0) goto fail;
    if (fread(&fmt_size, 4, 1, f) != 1 || fmt_size < 16) goto fail;
    if (fread(&audio_fmt, 2, 1, f) != 1) goto fail;
    if (fread(&channels, 2, 1, f) != 1) goto fail;
    if (fread(&sample_rate, 4, 1, f) != 1) goto fail;
    fseek(f, 6, SEEK_CUR);  /* byte_rate(4) + block_align(2) */
    if (fread(&bits_per_sample, 2, 1, f) != 1) goto fail;
    if (fmt_size > 16) fseek(f, (long)(fmt_size - 16), SEEK_CUR);

    /* Find data chunk */
    while (fread(chunk_id, 1, 4, f) == 4 && fread(&chunk_size, 4, 1, f) == 1) {
        if (memcmp(chunk_id, "data", 4) == 0) break;
        fseek(f, (long)(chunk_size + (chunk_size & 1)), SEEK_CUR);
    }
    if (memcmp(chunk_id, "data", 4) != 0) goto fail;

    if (bits_per_sample != 16 && bits_per_sample != 32) goto fail;
    int bytes_per_sample = bits_per_sample / 8;
    int n_samples = (int)(chunk_size / bytes_per_sample / channels);
    if (n_samples <= 0) goto fail;

    float *pcm = (float *)malloc((size_t)n_samples * sizeof(float));
    if (!pcm) goto fail;

    if (bits_per_sample == 16 && audio_fmt == 1) {
        int16_t *raw = (int16_t *)malloc(chunk_size);
        if (!raw) { free(pcm); goto fail; }
        if (fread(raw, 1, chunk_size, f) != (size_t)chunk_size) {
            free(raw); free(pcm); goto fail;
        }
        for (int i = 0; i < n_samples; i++)
            pcm[i] = (float)raw[i * channels] / 32768.0f;
        free(raw);
    } else if (bits_per_sample == 32 && audio_fmt == 3) {
        float *raw = (float *)malloc(chunk_size);
        if (!raw) { free(pcm); goto fail; }
        if (fread(raw, 1, chunk_size, f) != (size_t)chunk_size) {
            free(raw); free(pcm); goto fail;
        }
        for (int i = 0; i < n_samples; i++)
            pcm[i] = raw[i * channels];
        free(raw);
    } else {
        free(pcm);
        goto fail;
    }
    fclose(f);
    *out = pcm;
    *out_sr = (int)sample_rate;
    return n_samples;
fail:
    fclose(f);
    return -1;
}

int response_cache_add_wav(ResponseCache *cache, int fast_type,
                           const char *text, const char *wav_path) {
    if (!cache || !text || !wav_path) return -1;
    if (fast_type < 0 || fast_type >= FAST_COUNT) return -1;
    float *pcm = NULL;
    int sr = 0;
    int n = wav_load_simple(wav_path, &pcm, &sr);
    if (n <= 0) return -1;
    int out_n = n;
    float *resampled = NULL;
    if (sr != cache->cfg.sample_rate) {
        resampled = (float *)malloc((size_t)cache->max_samples_per_variant * sizeof(float));
        if (!resampled) { free(pcm); return -1; }
        out_n = linear_resample(pcm, n, sr, resampled, cache->max_samples_per_variant,
                                cache->cfg.sample_rate);
        free(pcm);
        pcm = resampled;
        resampled = NULL;
    }
    int r = response_cache_add(cache, fast_type, text, pcm, out_n);
    free(pcm);
    return r;
}

int response_cache_has(const ResponseCache *cache, int fast_type) {
    if (!cache || fast_type < 0 || fast_type >= FAST_COUNT) return 0;
    return cache->variant_counts[fast_type] > 0;
}

int response_cache_variant_count(const ResponseCache *cache, int fast_type) {
    if (!cache || fast_type < 0 || fast_type >= FAST_COUNT) return 0;
    return cache->variant_counts[fast_type];
}

void response_cache_set_speaker(ResponseCache *cache,
                                 const float *embedding, int dim) {
    if (!cache) return;
    free(cache->speaker_embedding);
    cache->speaker_embedding = NULL;
    cache->speaker_dim = 0;
    if (embedding && dim > 0) {
        cache->speaker_embedding = (float *)malloc((size_t)dim * sizeof(float));
        if (cache->speaker_embedding) {
            memcpy(cache->speaker_embedding, embedding, (size_t)dim * sizeof(float));
            cache->speaker_dim = dim;
        }
    }
}

void response_cache_stats(const ResponseCache *cache,
                          int *out_entries, float *out_seconds) {
    if (!cache) return;
    int entries = 0;
    int total_samples = 0;
    for (int t = 0; t < FAST_COUNT; t++) {
        for (int v = 0; v < cache->variant_counts[t]; v++) {
            if (cache->variants[t][v].audio) {
                entries++;
                total_samples += cache->variants[t][v].n_samples;
            }
        }
    }
    if (out_entries) *out_entries = entries;
    if (out_seconds) *out_seconds = cache->cfg.sample_rate > 0
        ? (float)total_samples / (float)cache->cfg.sample_rate : 0.0f;
}

/* Binary format: magic(4) version(4) sample_rate(4) max_variants(4) fast_count(4)
 * For each type: variant_count(4)
 *   For each variant: text_len(4) text(bytes) n_samples(4) audio(float32[])
 */
int response_cache_save(const ResponseCache *cache, const char *path) {
    if (!cache || !path) return -1;
    FILE *f = fopen(path, "wb");
    if (!f) return -1;
    fwrite(MAGIC, 1, 4, f);
    uint32_t v = VERSION;
    fwrite(&v, 4, 1, f);
    v = (uint32_t)cache->cfg.sample_rate;
    fwrite(&v, 4, 1, f);
    v = (uint32_t)cache->cfg.max_variants;
    fwrite(&v, 4, 1, f);
    v = (uint32_t)FAST_COUNT;
    fwrite(&v, 4, 1, f);
    for (int t = 0; t < FAST_COUNT; t++) {
        int cnt = cache->variant_counts[t];
        uint32_t u = (uint32_t)cnt;
        fwrite(&u, 4, 1, f);
        for (int v = 0; v < cnt; v++) {
            const CachedVariant *var = &cache->variants[t][v];
            if (!var->audio) continue;
            size_t text_len = strnlen(var->text, MAX_TEXT_LEN);
            uint32_t tl = (uint32_t)text_len;
            fwrite(&tl, 4, 1, f);
            fwrite(var->text, 1, (size_t)tl, f);
            uint32_t ns = (uint32_t)var->n_samples;
            fwrite(&ns, 4, 1, f);
            fwrite(var->audio, sizeof(float), (size_t)var->n_samples, f);
        }
    }
    fclose(f);
    return 0;
}

int response_cache_load(ResponseCache *cache, const char *path) {
    if (!cache || !path) return -1;
    FILE *f = fopen(path, "rb");
    if (!f) return -1;
    char magic[4];
    if (fread(magic, 1, 4, f) != 4 || memcmp(magic, MAGIC, 4) != 0) {
        fclose(f);
        return -1;
    }
    uint32_t ver, sr, maxv, fc;
    if (fread(&ver, 4, 1, f) != 1 || ver != VERSION) { fclose(f); return -1; }
    if (fread(&sr, 4, 1, f) != 1) { fclose(f); return -1; }
    if (fread(&maxv, 4, 1, f) != 1) { fclose(f); return -1; }
    if (fread(&fc, 4, 1, f) != 1 || fc != (uint32_t)FAST_COUNT) { fclose(f); return -1; }
    response_cache_clear(cache);
    for (int t = 0; t < FAST_COUNT && t < (int)fc; t++) {
        uint32_t cnt;
        if (fread(&cnt, 4, 1, f) != 1) { fclose(f); return -1; }
        if (cnt > MAX_VARIANTS) cnt = MAX_VARIANTS;
        for (uint32_t v = 0; v < cnt; v++) {
            uint32_t text_len;
            if (fread(&text_len, 4, 1, f) != 1) { fclose(f); return -1; }
            if (text_len > MAX_TEXT_LEN) text_len = MAX_TEXT_LEN;
            char text_buf[MAX_TEXT_LEN + 1];
            if (text_len > 0 && fread(text_buf, 1, text_len, f) != text_len) {
                fclose(f); return -1;
            }
            text_buf[text_len] = '\0';
            uint32_t n_samples;
            if (fread(&n_samples, 4, 1, f) != 1) { fclose(f); return -1; }
            if (n_samples > (uint32_t)cache->max_samples_per_variant) {
                fseek(f, (long)(n_samples * sizeof(float)), SEEK_CUR);
                continue;
            }
            float *aud = (float *)malloc((size_t)n_samples * sizeof(float));
            if (!aud) { fclose(f); return -1; }
            if (fread(aud, sizeof(float), (size_t)n_samples, f) != (size_t)n_samples) {
                free(aud); fclose(f); return -1;
            }
            free_variant(&cache->variants[t][v]);
            cache->variants[t][v].audio = aud;
            cache->variants[t][v].n_samples = (int)n_samples;
            strncpy(cache->variants[t][v].text, text_buf, MAX_TEXT_LEN);
            cache->variants[t][v].text[MAX_TEXT_LEN] = '\0';
            if ((int)(v + 1) > cache->variant_counts[t])
                cache->variant_counts[t] = (int)(v + 1);
        }
    }
    fclose(f);
    return 0;
}
