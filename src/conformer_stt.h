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
    uint32_t dtype;             /* 0 = fp32, 1 = fp16 */
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
 * Get the per-chunk encoder stride in milliseconds.
 * For cache-aware mode this is the frame time (e.g. 80ms).
 * For batch mode this is the full chunk duration.
 */
int conformer_stt_stride_ms(const ConformerSTT *stt);

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

/**
 * Run forward pass on pre-normalized mel features (testing/validation).
 * Mel data should be [T, n_mels] float32, already per-feature normalized.
 * Returns number of new transcript characters, or -1 on error.
 * After calling, use conformer_stt_get_text() to get transcript.
 */
int conformer_stt_forward_normalized_mel(ConformerSTT *stt,
                                          const float *mel, int T);

/**
 * Run forward pass starting from pre-subsampled encoder input (testing).
 * Data should be [T_sub, d_model] float32, already subsampled+projected.
 * Bypasses subsampling entirely — tests encoder blocks + CTC head only.
 */
int conformer_stt_forward_subsample_output(ConformerSTT *stt,
                                            const float *sub_out, int T_sub);

#endif /* CONFORMER_STT_H */
