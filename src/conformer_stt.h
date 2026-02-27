/**
 * conformer_stt.h — Pure C Conformer CTC speech-to-text for Apple Silicon.
 *
 * FastConformer encoder + CTC greedy/beam decode, running entirely on
 * Apple's AMX (BLAS), vDSP, and optionally Metal. No Python, no Rust,
 * no external ML framework.
 *
 * Compatible with NVIDIA Parakeet TDT, Zipformer, Moonshine, and other
 * CTC-based ASR models exported to the .cstt weight format.
 *
 * API mirrors pocket_stt_* for drop-in replacement in the pipeline.
 */

#ifndef CONFORMER_STT_H
#define CONFORMER_STT_H

#include <stdint.h>

typedef struct ConformerSTT ConformerSTT;

/* ─── Model Configuration ──────────────────────────────────────────────── */

#define CSTT_MAGIC 0x54545343  /* "CSTT" */

/* Subsampling types */
#define CSTT_SUB_CONV1D       0   /* Simple Conv1D stride-2 (generic) */
#define CSTT_SUB_CONV2D       1   /* Conv2D stride-2 (Whisper-style) */
#define CSTT_SUB_DW_STRIDING  2   /* Depthwise-striding (NeMo FastConformer) */

/* Header flags */
#define CSTT_FLAG_HAS_BIAS    (1 << 0)
#define CSTT_FLAG_SLANEY_NORM (1 << 1)
#define CSTT_FLAG_REL_PE      (1 << 2)  /* Relative positional encoding (Shaw-style) */
#define CSTT_FLAG_HAS_EOU     (1 << 3)  /* Vocabulary includes <eou> token for end-of-utterance */
#define CSTT_FLAG_CACHE_AWARE (1 << 4)  /* Activation caching for true streaming */
#define CSTT_FLAG_XSCALING   (1 << 5)  /* Scale encoder input by sqrt(d_model) */
#define CSTT_FLAG_TDT        (1 << 6)  /* TDT transducer decoder appended after encoder */

typedef struct {
    uint32_t magic;
    uint32_t version;
    uint32_t n_layers;          /* Conformer blocks (e.g. 17) */
    uint32_t d_model;           /* Model dimension (e.g. 512) */
    uint32_t n_heads;           /* Attention heads (e.g. 8) */
    uint32_t ff_mult;           /* Feed-forward expansion (e.g. 4) */
    uint32_t conv_kernel;       /* Depthwise conv kernel size (e.g. 9) */
    uint32_t vocab_size;        /* CTC vocabulary including blank (e.g. 1025) */
    uint32_t n_mels;            /* Mel spectrogram bins (e.g. 80) */
    uint32_t sample_rate;       /* Expected input sample rate (e.g. 16000) */
    uint32_t hop_length;        /* Mel hop in samples (e.g. 160) */
    uint32_t win_length;        /* Mel window in samples (e.g. 400) */
    uint32_t n_fft;             /* FFT size (e.g. 512) */
    uint32_t subsample_factor;  /* Time reduction factor (4 or 8) */
    uint32_t dtype;             /* 0 = fp32, 1 = fp16, 2 = int8, 3 = int4 (weight-only) */
    uint32_t flags;             /* Bitfield (CSTT_FLAG_*) */
    uint32_t sub_type;          /* Subsampling type (CSTT_SUB_*) */
    uint32_t n_sub_convs;       /* Number of conv layers in subsampling */
    uint32_t sub_feat_in;       /* Linear input dim after flatten (D*F/sub^2 for conv2d) */
    uint32_t sub_conv_kernel;   /* Subsampling conv kernel size (usually 3) */
    uint32_t reserved[4];       /* Future use, must be 0 */
} CSTTHeader;

/* ─── Engine Lifecycle ─────────────────────────────────────────────────── */

/**
 * Create a Conformer STT engine from a .cstt model file.
 *
 * The model file is mmap'd for zero-copy weight access. Weights stay
 * resident in memory and shared with the GPU via unified memory.
 *
 * @param model_path  Path to .cstt model file
 * @return            Opaque engine pointer, or NULL on failure
 */
ConformerSTT *conformer_stt_create(const char *model_path);

/** Destroy engine, freeing all resources. */
void conformer_stt_destroy(ConformerSTT *stt);

/* ─── Streaming Inference ──────────────────────────────────────────────── */

/**
 * Process a chunk of PCM audio.
 *
 * Audio is accumulated internally. When enough frames are available,
 * the conformer runs a forward pass and CTC decode. Recognized text
 * is appended to the internal transcript.
 *
 * @param stt         Engine instance
 * @param pcm         Float32 mono audio at the model's sample rate
 * @param n_samples   Number of samples in pcm
 * @return            Number of new characters recognized, or -1 on error
 */
int conformer_stt_process(ConformerSTT *stt, const float *pcm, int n_samples);

/**
 * Flush remaining audio at end of utterance.
 *
 * Pads the remaining audio to fill a complete chunk and runs a final
 * forward pass.
 *
 * @return  Number of new characters recognized, or -1 on error
 */
int conformer_stt_flush(ConformerSTT *stt);

/**
 * Get the full transcript from the current utterance.
 *
 * @param stt       Engine instance
 * @param buf       Output buffer (will be null-terminated)
 * @param buf_size  Size of output buffer in bytes
 * @return          Number of bytes written (excluding null), or -1 on error
 */
int conformer_stt_get_text(const ConformerSTT *stt, char *buf, int buf_size);

/** Reset for a new utterance. Clears transcript, audio buffers, and caches. */
void conformer_stt_reset(ConformerSTT *stt);

/* ─── EOU (End-of-Utterance) Detection ─────────────────────────────────── */

/**
 * Check if the model emitted an EOU token in the last CTC decode.
 * Only meaningful when the model was trained with CSTT_FLAG_HAS_EOU.
 *
 * @return  1 if EOU was detected, 0 otherwise
 */
int conformer_stt_has_eou(const ConformerSTT *stt);

/**
 * Get the EOU confidence — the softmax probability of the <eou> token
 * averaged over the trailing frames of the last forward pass.
 *
 * @param horizon  Number of trailing frames to average (e.g. 4)
 * @return         Probability in [0.0, 1.0], or 0.0 if no EOU support
 */
float conformer_stt_eou_prob(const ConformerSTT *stt, int horizon);

/**
 * Get the frame index at which EOU was most recently detected.
 * Returns -1 if no EOU was detected.
 */
int conformer_stt_eou_frame(const ConformerSTT *stt);

/* ─── Cache-Aware Streaming ────────────────────────────────────────────── */

/**
 * Enable or disable activation caching for true streaming.
 * When enabled, the encoder caches K/V projections and depthwise conv
 * states from previous chunks, enabling frame-by-frame streaming
 * without accuracy degradation.
 *
 * @param enable  1 to enable, 0 to disable
 */
void conformer_stt_set_cache_aware(ConformerSTT *stt, int enable);

/**
 * Set the chunk size for the encoder (mel frames per forward pass).
 * Smaller chunks = lower latency but potentially lower accuracy.
 * 0 = use default (8000 = full-sequence).
 * For real-time streaming with cache_aware: 40-80 frames (400-800ms).
 *
 * @param frames  Number of mel frames per chunk
 */
void conformer_stt_set_chunk_frames(ConformerSTT *stt, int frames);

/**
 * Get the per-chunk encoder stride in milliseconds.
 * For cache-aware mode this is the frame time (e.g. 80ms).
 * For batch mode this is the full chunk duration.
 */
int conformer_stt_stride_ms(const ConformerSTT *stt);

/* ─── Beam Search Decoding ─────────────────────────────────────────────── */

/**
 * Enable CTC beam search decoding with optional language model.
 *
 * @param stt        Engine instance
 * @param lm_path    Path to KenLM .arpa or .bin language model (NULL = no LM)
 * @param beam_size  Beam width (0 = default 16)
 * @param lm_weight  LM score weight (negative = default 1.5)
 * @param word_score Per-word insertion bonus (0.0 = neutral)
 * @return           0 on success, -1 on failure
 */
int conformer_stt_enable_beam_search(ConformerSTT *stt,
                                      const char *lm_path,
                                      int beam_size,
                                      float lm_weight,
                                      float word_score);

/**
 * Disable beam search, reverting to greedy CTC decode.
 */
void conformer_stt_disable_beam_search(ConformerSTT *stt);

/* ─── Info ─────────────────────────────────────────────────────────────── */

/** Returns expected sample rate (e.g. 16000). */
int conformer_stt_sample_rate(const ConformerSTT *stt);

/** Returns model dimension. */
int conformer_stt_d_model(const ConformerSTT *stt);

/** Returns number of conformer layers. */
int conformer_stt_n_layers(const ConformerSTT *stt);

/** Returns vocabulary size. */
int conformer_stt_vocab_size(const ConformerSTT *stt);

/** Returns 1 if the model has EOU token support. */
int conformer_stt_has_eou_support(const ConformerSTT *stt);

/** Returns 1 if the model uses TDT (transducer) decoding instead of CTC. */
int conformer_stt_is_tdt(const ConformerSTT *stt);

/* ─── Conmer Mode (Attention-Free Streaming) ──────────────────────────── */

/**
 * Enable or disable Conmer mode (convolution-only Conformer).
 *
 * When enabled, multi-head self-attention is skipped in all encoder blocks.
 * Based on Amazon Conmer research (Interspeech 2023): for streaming short
 * utterances (<10s), removing MHSA yields ~4% WER improvement and ~10%
 * compute savings because the local context from depthwise convolution
 * is sufficient.
 *
 * Block architecture in Conmer mode:
 *   input → FFN½ → Conv → FFN½ → LayerNorm → output
 *
 * The model still loads attention weights (for fallback), but they are
 * not exercised during forward passes while Conmer mode is active.
 *
 * @param stt     Engine instance
 * @param enable  1 to enable Conmer mode, 0 for standard Conformer
 */
void conformer_stt_set_conmer_mode(ConformerSTT *stt, int enable);

/** Returns 1 if Conmer mode is currently enabled. */
int conformer_stt_is_conmer(const ConformerSTT *stt);

/* ─── External Forward-Pass Hook ───────────────────────────────────────── */

/**
 * Callback for replacing the encoder forward pass with an external backend.
 * Called instead of the built-in cblas/vDSP encoder when set.
 *
 * @param user_ctx   Opaque pointer passed through from set_external_forward
 * @param mel_in     Mel spectrogram [T, n_mels], row-major float32
 * @param T          Number of input time frames
 * @param n_mels     Number of mel bins
 * @param logits_out Output logits [T_sub, vocab_size], caller-allocated
 * @param max_T_sub  Maximum output time steps (buffer capacity)
 * @return           Number of output time steps, or -1 to fallback to built-in
 */
typedef int (*conformer_external_forward_fn)(void *user_ctx,
    const float *mel_in, int T, int n_mels,
    float *logits_out, int max_T_sub);

/**
 * Set an external forward-pass hook. When set, the encoder will call this
 * function instead of the built-in conformer blocks. If it returns -1,
 * the built-in path is used as fallback.
 *
 * @param stt        Engine instance
 * @param fn         External forward function, or NULL to disable
 * @param user_ctx   Opaque pointer passed to fn
 */
void conformer_stt_set_external_forward(ConformerSTT *stt,
    conformer_external_forward_fn fn, void *user_ctx);

/**
 * Get a pointer to the logits workspace buffer and its stride.
 * Useful for external forward hooks that need the same output buffer.
 */
float *conformer_stt_get_logits_buf(ConformerSTT *stt, int *out_vocab);

#endif /* CONFORMER_STT_H */
