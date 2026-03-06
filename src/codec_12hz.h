/*
 * codec_12hz.h — Sonata 12.5Hz Neural Audio Codec
 *
 * Low-frame-rate codec (12.5Hz vs 50Hz) for 4x token reduction.
 * Each frame represents 80ms of audio (vs 20ms at 50Hz).
 *
 * Inference path (decoder):
 *   FSQ indices → dequantization → embedding (512-dim)
 *   Semantic codes (4-dim) + acoustic latent (512-dim) → ConvDecoder
 *   ConvDecoder: 5-stage transposed convolution (1920x upsample)
 *   Output: 24 kHz waveform
 *
 * All computations use Apple Accelerate (vDSP/cblas) on AMX.
 */

#ifndef CODEC_12HZ_H
#define CODEC_12HZ_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct Codec12Hz Codec12Hz;

/* Codec configuration (matches train/sonata/config.py Codec12HzConfig) */
typedef struct {
    int sample_rate;              /* 24000 */
    int n_fft;                    /* 4096 */
    int hop_length;               /* 1920 → 12.5 Hz */
    int n_mels;                   /* 160 */

    /* FSQ parameters */
    int fsq_dim;                  /* 4 (8^4 = 4096 codebook) */
    int fsq_codebook_size;        /* 4096 */

    /* Embedding dimension (after FSQ dequant) */
    int fsq_embed_dim;            /* 512 */

    /* Acoustic latent dimension */
    int acoustic_dim;             /* 512 */

    /* ConvDecoder configuration */
    int dec_dim;                  /* 768 */
    int dec_n_layers;             /* 10 */
    int dec_conv_kernel;          /* 7 */
    float dec_ff_mult;            /* 4.0 */

    /* Decoder strides: [3, 4, 5, 8, 4] = 1920x upsample */
    int decoder_strides[5];
} Codec12HzConfig;

/*
 * Create codec decoder from binary model file.
 *
 * Binary format (safetensors-compatible):
 *   - Magic: "CODEC12HZ\x00" (10 bytes)
 *   - Config: codec_12hz_config (uint32 x 16)
 *   - FSQ codebook: 4096 × 512 embeddings (float32)
 *   - Input projection: 768 × 1024 weights + 768 biases
 *   - ConvNeXt backbone: 10 layers of weights
 *   - Upsample blocks: 5 stages of transposed convolutions + residuals
 *   - Output projection: 1 × 768 weights + 1 bias
 *
 * Returns NULL on error (invalid config, malloc failure, etc).
 */
Codec12Hz *codec_12hz_create(const char *model_path);

/*
 * Create codec with explicit config (for testing).
 * Does not load weights — caller must populate via codec_12hz_load_weights().
 */
Codec12Hz *codec_12hz_create_empty(const Codec12HzConfig *cfg);

/*
 * Destroy codec and free all resources.
 */
void codec_12hz_destroy(Codec12Hz *codec);

/*
 * Reset internal state (e.g., between utterances).
 * Clears scratch buffers and ring buffer state if using streaming.
 */
void codec_12hz_reset(Codec12Hz *codec);

/*
 * Enable streaming mode with ring buffer for overlap-add.
 * In streaming mode, decoding processes one frame at a time with
 * O(1) memory bandwidth for overlap-add (vs linear memmove).
 *
 * enable: nonzero to enable streaming, 0 for batch mode.
 */
void codec_12hz_set_streaming(Codec12Hz *codec, int enable);

/*
 * Decode one frame from semantic codes + acoustic latent → audio.
 *
 * semantic_codes: [fsq_dim] indices 0-7 (4-element FSQ indices)
 * acoustic_latent: [acoustic_dim] continuous latent (512-element vector)
 * out_audio: [hop_length] output buffer (1920 samples at 12.5Hz/24kHz)
 *
 * Returns number of samples written (= hop_length), or 0 on error.
 */
int codec_12hz_decode_frame(
    Codec12Hz *codec,
    const uint8_t *semantic_codes,   /* [fsq_dim] */
    const float *acoustic_latent,     /* [acoustic_dim] */
    float *out_audio                  /* [hop_length] */
);

/*
 * Decode a batch of frames.
 *
 * semantic_codes: [n_frames, fsq_dim] indices (row-major)
 * acoustic_latents: [n_frames, acoustic_dim] continuous latents (row-major)
 * out_audio: [n_frames * hop_length] output buffer (row-major)
 *
 * Returns total samples written (= n_frames * hop_length), or 0 on error.
 */
int codec_12hz_decode_batch(
    Codec12Hz *codec,
    const uint8_t *semantic_codes,    /* [n_frames, fsq_dim] */
    const float *acoustic_latents,    /* [n_frames, acoustic_dim] */
    int n_frames,
    float *out_audio                  /* [n_frames * hop_length] */
);

/*
 * Load weights from binary file (internal, called by codec_12hz_create).
 *
 * Returns 0 on success, -1 on error.
 */
int codec_12hz_load_weights(Codec12Hz *codec, const char *model_path);

#ifdef __cplusplus
}
#endif

#endif /* CODEC_12HZ_H */
