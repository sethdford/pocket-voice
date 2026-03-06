#ifndef RESPONSE_CACHE_H
#define RESPONSE_CACHE_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ResponseCache ResponseCache;

typedef struct {
    int   sample_rate;           /* TTS sample rate (24000) */
    int   max_variants;          /* Variants per response type (default: 3) */
    int   max_audio_seconds;     /* Max audio per variant in seconds (default: 3) */
} ResponseCacheConfig;

/* Create response cache. Returns NULL on failure. */
ResponseCache *response_cache_create(const ResponseCacheConfig *cfg);
void response_cache_destroy(ResponseCache *cache);

/* Warm the cache by synthesizing all response variants.
 * tts_synthesize: callback that converts text → PCM audio
 *   int tts_synthesize(void *ctx, const char *text, float *out_pcm, int max_samples);
 *   Returns number of samples written, or -1 on error.
 * tts_ctx: opaque context passed to tts_synthesize.
 * Returns number of responses successfully cached. */
int response_cache_warm(ResponseCache *cache,
                        int (*tts_synthesize)(void *ctx, const char *text,
                                             float *out_pcm, int max_samples),
                        void *tts_ctx);

/* Get pre-synthesized audio for a fast response type.
 * Selects a random variant for naturalness.
 * Returns pointer to internal buffer, sets *out_len.
 * Returns NULL if not cached. */
const float *response_cache_get(ResponseCache *cache, int fast_type, int *out_len);

/* Get a specific variant (for deterministic testing).
 * variant_idx: 0 to max_variants-1.
 * Returns NULL if not cached or variant doesn't exist. */
const float *response_cache_get_variant(ResponseCache *cache, int fast_type,
                                        int variant_idx, int *out_len);

/* Load a pre-built cache from disk (serialized binary format).
 * Returns 0 on success, -1 on error. */
int response_cache_load(ResponseCache *cache, const char *path);

/* Save cache to disk (so we don't re-synthesize on next launch).
 * Format: header + [type][variant][samples] for each entry.
 * Returns 0 on success, -1 on error. */
int response_cache_save(const ResponseCache *cache, const char *path);

/* Add a custom response variant for a type.
 * text: the response text (stored for reference).
 * pcm: audio samples at cache sample_rate.
 * Returns 0 on success, -1 if full. */
int response_cache_add(ResponseCache *cache, int fast_type,
                       const char *text, const float *pcm, int n_samples);

/* Add a custom response from a WAV file. */
int response_cache_add_wav(ResponseCache *cache, int fast_type,
                           const char *text, const char *wav_path);

/* Check if a response type is cached (has at least one variant). */
int response_cache_has(const ResponseCache *cache, int fast_type);

/* Get the number of cached variants for a type. */
int response_cache_variant_count(const ResponseCache *cache, int fast_type);

/* Update speaker embedding — next warm() will generate in this voice. */
void response_cache_set_speaker(ResponseCache *cache,
                                const float *embedding, int dim);

/* Clear all cached audio. */
void response_cache_clear(ResponseCache *cache);

/* Get stats: total cached entries, total audio seconds. */
void response_cache_stats(const ResponseCache *cache,
                          int *out_entries, float *out_seconds);

#ifdef __cplusplus
}
#endif

#endif /* RESPONSE_CACHE_H */
