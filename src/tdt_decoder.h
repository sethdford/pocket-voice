/**
 * tdt_decoder.h — Token Duration Transducer decoder (C-ABI).
 *
 * Implements the TDT (Token Duration Transducer) decoder for ASR.
 * The decoder consists of:
 *   - Prediction network: multi-layer LSTM + token embedding
 *   - Joint network: combines encoder + prediction outputs → token + duration logits
 *   - Greedy decode loop with variable-duration frame skipping
 *
 * Thread safety: A single TDTDecoder instance is NOT thread-safe.
 * Create one per thread if needed.
 */

#ifndef TDT_DECODER_H
#define TDT_DECODER_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct TDTDecoder TDTDecoder;

typedef struct {
    int   pred_hidden;     /* LSTM hidden size (e.g. 640) */
    int   pred_layers;     /* Number of LSTM layers (e.g. 2) */
    int   vocab_size;      /* Token vocabulary size including blank (e.g. 1025) */
    int   n_durations;     /* Number of duration classes (e.g. 5) */
    int   joint_dim;       /* Joint network projection dim (e.g. 640) */
    int   encoder_dim;     /* Encoder output dim (d_model, e.g. 1024) */
    int   blank_id;        /* Index of blank token (last token in NeMo) */
    int   duration_values[16]; /* Mapped duration frame counts (e.g. {0,1,2,4,8}) */
} TDTConfig;

/**
 * Create a TDT decoder. Weights are borrowed (caller keeps them alive).
 *
 * Weight layout expected (all float*, row-major):
 *   embed_w:      [vocab_size, pred_hidden]
 *   For each LSTM layer l:
 *     lstm_wi[l]:  [4*pred_hidden, input_size]  (input_size = pred_hidden)
 *     lstm_bi[l]:  [4*pred_hidden]
 *     lstm_wh[l]:  [4*pred_hidden, pred_hidden]
 *     lstm_bh[l]:  [4*pred_hidden]
 *   joint_enc_w:  [joint_dim, encoder_dim]
 *   joint_enc_b:  [joint_dim]
 *   joint_pred_w: [joint_dim, pred_hidden]
 *   joint_pred_b: [joint_dim]
 *   joint_out_w:  [vocab_size + n_durations, joint_dim]
 *   joint_out_b:  [vocab_size + n_durations]
 */
TDTDecoder *tdt_decoder_create(const TDTConfig *config,
                                const float *embed_w,
                                const float *const *lstm_wi,
                                const float *const *lstm_bi,
                                const float *const *lstm_wh,
                                const float *const *lstm_bh,
                                const float *joint_enc_w,
                                const float *joint_enc_b,
                                const float *joint_pred_w,
                                const float *joint_pred_b,
                                const float *joint_out_w,
                                const float *joint_out_b);

/**
 * Run TDT greedy decoding on encoder output.
 *
 * @param dec       TDT decoder instance
 * @param enc_out   Encoder output [T, encoder_dim], row-major
 * @param T         Number of encoder time steps
 * @param tokens    Output token IDs (caller-allocated, at least T entries)
 * @param max_tokens Maximum number of tokens to output
 * @return          Number of tokens decoded (excluding blank), or -1 on error
 */
int tdt_decoder_decode(TDTDecoder *dec,
                       const float *enc_out, int T,
                       int *tokens, int max_tokens);

/**
 * Reset decoder state (LSTM hidden states). Call between utterances.
 */
void tdt_decoder_reset(TDTDecoder *dec);

/**
 * Destroy decoder and free all working memory.
 */
void tdt_decoder_destroy(TDTDecoder *dec);

#ifdef __cplusplus
}
#endif

#endif /* TDT_DECODER_H */
