/**
 * speaker_encoder.c — Zero-shot voice cloning via speaker encoder.
 *
 * Wraps the Rust ECAPA-TDNN speaker encoder (sonata_speaker).
 * Delegates to native Rust encoder for safetensors loading and forward pass.
 */

#include "speaker_encoder.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <limits.h>

/* ═══════════════════════════════════════════════════════════════════════════
 * Rust FFI (from sonata_speaker native library)
 * ═══════════════════════════════════════════════════════════════════════════ */

typedef void* RustSpeakerEncoder;

extern RustSpeakerEncoder speaker_encoder_native_create(const char *weights_path,
                                                         const char *config_path);
extern void speaker_encoder_native_destroy(RustSpeakerEncoder enc);
extern int speaker_encoder_native_embedding_dim(RustSpeakerEncoder enc);
extern int speaker_encoder_native_sample_rate(RustSpeakerEncoder enc);
extern int speaker_encoder_native_encode(RustSpeakerEncoder enc,
                                          const float *mel_data,
                                          int n_frames, int n_mels,
                                          float *out);
extern int speaker_encoder_native_encode_audio(RustSpeakerEncoder enc,
                                               const float *pcm,
                                               int n_samples,
                                               int sample_rate,
                                               float *out);

/* ═══════════════════════════════════════════════════════════════════════════
 * Speaker Encoder Wrapper
 * ═══════════════════════════════════════════════════════════════════════════ */

struct SpeakerEncoder {
    RustSpeakerEncoder native_encoder;
    int embedding_dim;
    int sample_rate;
};

SpeakerEncoder *speaker_encoder_create(const char *weights_path) {
    if (!weights_path) {
        fprintf(stderr, "[speaker_encoder] weights_path is NULL\n");
        return NULL;
    }

    /* Canonicalize weights path to prevent directory traversal attacks */
    char real_weights[PATH_MAX];
    if (!realpath(weights_path, real_weights)) {
        fprintf(stderr, "[speaker_encoder] Invalid path: %s\n", weights_path);
        return NULL;
    }

    /* Load configuration from same directory as weights */
    /* Assume config is speaker_encoder_config.json in the same dir */
    char config_path[1024] = {0};
    const char *last_slash = strrchr(real_weights, '/');
    if (last_slash) {
        int dir_len = (int)(last_slash - real_weights);
        snprintf(config_path, sizeof(config_path) - 1, "%.*s/speaker_encoder_config.json",
                 dir_len, real_weights);
    } else {
        snprintf(config_path, sizeof(config_path) - 1, "speaker_encoder_config.json");
    }

    SpeakerEncoder *enc = (SpeakerEncoder *)malloc(sizeof(SpeakerEncoder));
    if (!enc) {
        fprintf(stderr, "[speaker_encoder] malloc failed\n");
        return NULL;
    }

    enc->native_encoder = speaker_encoder_native_create(real_weights, config_path);
    if (!enc->native_encoder) {
        fprintf(stderr, "[speaker_encoder] Rust encoder creation failed\n");
        free(enc);
        return NULL;
    }

    enc->embedding_dim = speaker_encoder_native_embedding_dim(enc->native_encoder);
    enc->sample_rate = speaker_encoder_native_sample_rate(enc->native_encoder);

    fprintf(stderr, "[speaker_encoder] Created (dim=%d, sr=%d Hz)\n",
            enc->embedding_dim, enc->sample_rate);

    return enc;
}

void speaker_encoder_destroy(SpeakerEncoder *enc) {
    if (!enc) return;
    if (enc->native_encoder) {
        speaker_encoder_native_destroy(enc->native_encoder);
    }
    free(enc);
}

int speaker_encoder_encode_audio(SpeakerEncoder *enc, const float *pcm,
                                  int n_samples, int sample_rate,
                                  float *out_emb) {
    if (!enc || !enc->native_encoder || !pcm || !out_emb || n_samples <= 0) {
        return -1;
    }

    /* Delegate to native Rust encoder (handles resampling and mel extraction) */
    return speaker_encoder_native_encode_audio(enc->native_encoder, pcm, n_samples,
                                                sample_rate, out_emb);
}

int speaker_encoder_encode_mel(SpeakerEncoder *enc, const float *mel,
                                int n_frames, float *out_emb) {
    if (!enc || !enc->native_encoder || !mel || !out_emb || n_frames <= 0) {
        return -1;
    }

    /* Mel is expected as [n_frames * 80] in row-major (frame-first) */
    return speaker_encoder_native_encode(enc->native_encoder, mel, n_frames, 80, out_emb);
}

int speaker_encoder_embedding_dim(const SpeakerEncoder *enc) {
    return enc ? enc->embedding_dim : -1;
}

int speaker_encoder_sample_rate(const SpeakerEncoder *enc) {
    return enc ? enc->sample_rate : -1;
}
