/*
 * sonata_stt.h — Sonata STT: CTC streaming speech recognition.
 *
 * Pure C implementation of the Sonata CTC model:
 *   Audio (24kHz) → Mel (80-bin, 50Hz) → Conformer (RoPE) → CTC → Text
 *
 * Features:
 *   - Streaming: feed_chunk / flush API with conv state caching
 *   - Beam search: optional CTC beam decoder with KenLM LM rescoring
 *   - EOU detection: inline <eou> token (id=29) for end-of-utterance
 *   - Supports both base (4L d=256) and large (12L d=512) encoder
 *
 * All matrix ops via cblas_sgemm/sgemv (AMX-accelerated on Apple Silicon).
 *
 * Weight file format (.cstt_sonata):
 *   Header: magic(4) version(4) enc_dim(4) n_layers(4) n_heads(4)
 *           n_mels(4) conv_kernel(4) text_vocab(4) n_weights(4) pad(4)
 *   Weights: float32 array (row-major, PyTorch convention)
 */

#ifndef SONATA_STT_H
#define SONATA_STT_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct SonataSTT SonataSTT;
typedef struct CTCBeamDecoder CTCBeamDecoder;

/* ─── Lifecycle ─────────────────────────────────────────────────────── */

SonataSTT *sonata_stt_create(const char *weights_path);
void sonata_stt_destroy(SonataSTT *stt);
void sonata_stt_reset(SonataSTT *stt);

/* ─── Batch processing (non-streaming) ──────────────────────────────── */

/*
 * Process audio and return transcription (greedy decode).
 * pcm: float32 audio at 24kHz, mono. Returns chars written or -1.
 */
int sonata_stt_process(
    SonataSTT *stt,
    const float *pcm,
    int n_samples,
    char *out_text,
    int max_len
);

/*
 * Get frame-level CTC logits for pipeline integration.
 * out_logits: [max_frames * vocab_size], row-major. Returns frames written or -1.
 */
int sonata_stt_get_logits(
    SonataSTT *stt,
    const float *pcm,
    int n_samples,
    float *out_logits,
    int max_frames
);

/* ─── Streaming API ─────────────────────────────────────────────────── */

/*
 * Start a streaming session. Allocates internal audio accumulation buffer.
 * max_seconds: max audio duration to buffer (default 30).
 */
int sonata_stt_stream_start(SonataSTT *stt, float max_seconds);

/*
 * Feed audio chunk. Appends to internal buffer.
 * pcm: float32 at 24kHz. Returns 0 on success, -1 on overflow.
 */
int sonata_stt_stream_feed(SonataSTT *stt, const float *pcm, int n_samples);

/*
 * Flush: run encoder on accumulated audio and decode.
 * Re-encodes full window each call (growing-window strategy).
 * out_text: output buffer. Returns chars written, or -1.
 */
int sonata_stt_stream_flush(SonataSTT *stt, char *out_text, int max_len);

/* End streaming session and reset buffers. */
void sonata_stt_stream_end(SonataSTT *stt);

/* ─── Beam search ───────────────────────────────────────────────────── */

/*
 * Attach a CTC beam decoder for higher-accuracy transcription.
 * The decoder is NOT owned by sonata_stt (caller manages lifetime).
 * Pass NULL to detach and fall back to greedy decode.
 */
void sonata_stt_set_beam_decoder(SonataSTT *stt, CTCBeamDecoder *beam);

/*
 * Process audio with beam search. Falls back to greedy if no beam decoder set.
 */
int sonata_stt_process_beam(
    SonataSTT *stt,
    const float *pcm,
    int n_samples,
    char *out_text,
    int max_len
);

/* ─── Word-level timestamps (CTC alignment) ──────────────────────────── */

typedef struct {
    char word[64];        /* The word text */
    float start_sec;      /* Start time in seconds */
    float end_sec;        /* End time in seconds */
    float confidence;    /* Geometric mean of CTC token probs for this word */
} SonataSTTWord;

/*
 * Get word-level timestamps from the last sonata_stt_process() call.
 * Returns number of words written, or -1 on error.
 */
int sonata_stt_get_words(const SonataSTT *stt, SonataSTTWord *out, int max_words);

/* ─── EOU detection ─────────────────────────────────────────────────── */

/*
 * Extract per-frame EOU probability from the last processed logits.
 * out_probs: [max_frames] buffer. Returns frames written, or -1.
 * Call after sonata_stt_get_logits() or sonata_stt_stream_flush().
 */
int sonata_stt_eou_probs(SonataSTT *stt, float *out_probs, int max_frames);

/*
 * Get the peak EOU probability from the last N frames of the last decode.
 * window_frames: number of trailing frames to examine (0 = all).
 * Returns max P(eou) in [0,1], or -1.0 on error.
 */
float sonata_stt_eou_peak(SonataSTT *stt, int window_frames);

/* ─── FP16 acceleration ─────────────────────────────────────────────── */

/*
 * Enable FP16 weight storage + AMX half-precision matmul.
 * Reduces memory bandwidth by 2x for ~2x throughput on Apple Silicon.
 * Must be called AFTER create but BEFORE any processing.
 * Returns 0 on success, -1 if conversion fails.
 */
int sonata_stt_enable_fp16(SonataSTT *stt);

/* Query whether FP16 mode is active. */
int sonata_stt_is_fp16(const SonataSTT *stt);

/* ─── Properties ────────────────────────────────────────────────────── */

int sonata_stt_vocab_size(const SonataSTT *stt);
int sonata_stt_enc_dim(const SonataSTT *stt);
int sonata_stt_eou_id(const SonataSTT *stt);

#ifdef __cplusplus
}
#endif

#endif /* SONATA_STT_H */
