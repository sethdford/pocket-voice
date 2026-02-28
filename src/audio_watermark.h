/**
 * audio_watermark.h — Spread-spectrum audio watermarking for TTS output.
 *
 * Embeds an imperceptible watermark into audio using a spread-spectrum
 * technique in the frequency domain (1–4kHz bins at ~-40dB below signal).
 *
 * Key features:
 *   - Gold-code PN sequences seeded by a secret key (reproducible)
 *   - Payload: AI-generated flag, timestamp, model ID
 *   - Uses vDSP FFT (Apple Accelerate) for all spectral operations
 *   - Zero allocations after create() — all buffers pre-allocated
 *   - < 1ms per frame processing on Apple Silicon
 *
 * Usage:
 *   AudioWatermark *wm = audio_watermark_create(24000, 960, key, 32);
 *   audio_watermark_set_payload(wm, &payload);
 *   audio_watermark_embed(wm, pcm, n_samples);     // in-place
 *   float score = audio_watermark_detect(wm, pcm, n_samples);
 *   audio_watermark_destroy(wm);
 */

#ifndef AUDIO_WATERMARK_H
#define AUDIO_WATERMARK_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct AudioWatermark AudioWatermark;

/** Watermark payload embedded in each frame. */
typedef struct {
    uint8_t  ai_generated;   /**< 1 = AI-generated audio */
    uint32_t timestamp;      /**< Unix timestamp (seconds, lower 32 bits) */
    uint16_t model_id;       /**< Model identifier (0–65535) */
} AudioWatermarkPayload;

/**
 * Create a watermark context.
 *
 * @param sample_rate   Audio sample rate (e.g. 24000)
 * @param frame_size    Samples per frame (e.g. 960). Rounded up to next
 *                      power of 2 internally for FFT.
 * @param key           Secret key bytes for PN sequence generation
 * @param key_len       Length of key in bytes (>= 4)
 * @return              Handle, or NULL on failure
 */
AudioWatermark *audio_watermark_create(int sample_rate, int frame_size,
                                       const uint8_t *key, int key_len);

/** Destroy and free all resources. */
void audio_watermark_destroy(AudioWatermark *wm);

/**
 * Set the payload to embed in subsequent frames.
 *
 * @param payload  Payload struct (copied internally)
 */
void audio_watermark_set_payload(AudioWatermark *wm,
                                 const AudioWatermarkPayload *payload);

/**
 * Embed watermark into audio in-place.
 *
 * Operates on arbitrary-length buffers by processing frame_size chunks.
 * Remaining samples (< frame_size) are left unmodified.
 *
 * @param pcm       Audio buffer (modified in place)
 * @param n_samples Number of samples
 * @return          0 on success, -1 on error
 */
int audio_watermark_embed(AudioWatermark *wm, float *pcm, int n_samples);

/**
 * Detect watermark presence in audio.
 *
 * Returns a correlation score: > 0.3 indicates watermark present.
 *
 * @param pcm       Audio buffer (not modified)
 * @param n_samples Number of samples
 * @return          Correlation score [0.0, 1.0], or -1.0 on error
 */
float audio_watermark_detect(const AudioWatermark *wm,
                             const float *pcm, int n_samples);

/**
 * Extract payload from watermarked audio.
 *
 * @param pcm       Audio buffer (not modified)
 * @param n_samples Number of samples
 * @param out       Output payload (filled on success)
 * @return          0 on success, -1 if watermark not detected
 */
int audio_watermark_extract(const AudioWatermark *wm,
                            const float *pcm, int n_samples,
                            AudioWatermarkPayload *out);

/** Reset internal state (e.g. between utterances). */
void audio_watermark_reset(AudioWatermark *wm);

/**
 * Enable or disable watermarking.
 *
 * @param enable  1 = embed watermark, 0 = pass-through
 */
void audio_watermark_enable(AudioWatermark *wm, int enable);

/** Return 1 if watermarking is enabled. */
int audio_watermark_is_enabled(const AudioWatermark *wm);

#ifdef __cplusplus
}
#endif

#endif /* AUDIO_WATERMARK_H */
