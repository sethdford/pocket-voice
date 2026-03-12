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

/* Alias: extract from raw PCM at 16kHz (historically used by diarizer and voice_onboard). */
int speaker_encoder_extract(SpeakerEncoder *enc, const float *audio,
                             int n_samples, float *embedding_out) {
    return speaker_encoder_encode_audio(enc, audio, n_samples, 16000, embedding_out);
}

/* Extract embedding from a WAV file (16/24-bit PCM, mono or stereo). */
int speaker_encoder_extract_from_wav(SpeakerEncoder *enc,
                                      const char *wav_path,
                                      float *embedding_out) {
    if (!enc || !wav_path || !embedding_out) return -1;

    FILE *f = fopen(wav_path, "rb");
    if (!f) {
        fprintf(stderr, "[speaker_encoder] Cannot open WAV: %s\n", wav_path);
        return -1;
    }

    /* Parse minimal RIFF/WAV header */
    uint8_t hdr[44];
    if (fread(hdr, 1, 44, f) < 44) { fclose(f); return -1; }

    /* Validate RIFF/WAVE magic */
    if (hdr[0] != 'R' || hdr[1] != 'I' || hdr[2] != 'F' || hdr[3] != 'F' ||
        hdr[8] != 'W' || hdr[9] != 'A' || hdr[10] != 'V' || hdr[11] != 'E') {
        fprintf(stderr, "[speaker_encoder] Not a RIFF/WAVE file: %s\n", wav_path);
        fclose(f); return -1;
    }

    int fmt_tag     = hdr[20] | (hdr[21] << 8);
    int n_channels  = hdr[22] | (hdr[23] << 8);
    int sample_rate = hdr[24] | (hdr[25] << 8) | (hdr[26] << 16) | (hdr[27] << 24);
    int bits        = hdr[34] | (hdr[35] << 8);

    if (fmt_tag != 1 || (bits != 16 && bits != 24 && bits != 32)) {
        fprintf(stderr, "[speaker_encoder] Unsupported WAV format (tag=%d bits=%d)\n",
                fmt_tag, bits);
        fclose(f); return -1;
    }

    /* Scan forward to 'data' chunk */
    uint8_t chunk[8];
    while (fread(chunk, 1, 8, f) == 8) {
        uint32_t chunk_size = chunk[4] | ((uint32_t)chunk[5] << 8) |
                              ((uint32_t)chunk[6] << 16) | ((uint32_t)chunk[7] << 24);
        if (chunk[0] == 'd' && chunk[1] == 'a' && chunk[2] == 't' && chunk[3] == 'a') {
            int bytes_per_sample = bits / 8;
            int n_samples_total = (int)(chunk_size / (bytes_per_sample * n_channels));
            /* Cap at 30s to avoid huge allocations */
            if (n_samples_total > sample_rate * 30) n_samples_total = sample_rate * 30;

            float *pcm = (float *)malloc(n_samples_total * sizeof(float));
            if (!pcm) { fclose(f); return -1; }

            int read_ok = 1;
            for (int i = 0; i < n_samples_total; i++) {
                /* Read first channel only (downmix to mono) */
                float s = 0.0f;
                for (int ch = 0; ch < n_channels; ch++) {
                    uint8_t buf[4] = {0};
                    if (fread(buf, 1, bytes_per_sample, f) < (size_t)bytes_per_sample) {
                        read_ok = 0; break;
                    }
                    if (ch == 0) {
                        if (bits == 16) {
                            int16_t v = buf[0] | ((int16_t)buf[1] << 8);
                            s = v / 32768.0f;
                        } else if (bits == 24) {
                            int32_t v = buf[0] | (buf[1] << 8) | ((int32_t)(int8_t)buf[2] << 16);
                            s = v / 8388608.0f;
                        } else { /* 32-bit float */
                            memcpy(&s, buf, 4);
                        }
                    }
                }
                if (!read_ok) { n_samples_total = i; break; }
                pcm[i] = s;
            }

            fclose(f);
            int ret = speaker_encoder_encode_audio(enc, pcm, n_samples_total,
                                                   sample_rate, embedding_out);
            free(pcm);
            return ret;
        }
        /* Skip non-data chunks */
        fseek(f, (long)chunk_size, SEEK_CUR);
    }

    fclose(f);
    fprintf(stderr, "[speaker_encoder] No data chunk found in WAV: %s\n", wav_path);
    return -1;
}
