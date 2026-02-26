/**
 * ctc_beam_decoder.h — CTC beam search decoder with optional KenLM rescoring.
 *
 * Pure C-ABI interface wrapping a prefix beam search algorithm with n-gram
 * language model support via KenLM. Drop-in alternative to greedy CTC decode.
 */

#ifndef CTC_BEAM_DECODER_H
#define CTC_BEAM_DECODER_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct CTCBeamDecoder CTCBeamDecoder;

typedef struct {
    int    beam_size;       /* Number of beams to keep (default: 16)       */
    float  lm_weight;       /* LM score weight alpha   (default: 1.5)      */
    float  word_score;      /* Per-word insertion bonus (default: 0.0)      */
    float  blank_skip_thresh; /* Skip time steps where P(blank)>thresh (0=off) */
} CTCBeamConfig;

/**
 * Create a beam decoder. If lm_path is NULL, runs without LM (pure CTC).
 *
 * @param lm_path    Path to KenLM binary .arpa or .bin language model (or NULL)
 * @param vocab      Token strings array (index = token id)
 * @param vocab_size Number of tokens including blank
 * @param blank_id   ID of the CTC blank token
 * @param config     Beam search configuration
 * @return           Opaque decoder handle, or NULL on failure
 */
CTCBeamDecoder *ctc_beam_create(const char *lm_path,
                                 const char *const *vocab,
                                 int vocab_size,
                                 int blank_id,
                                 const CTCBeamConfig *config);

/**
 * Decode CTC log-probabilities into text.
 *
 * @param dec        Decoder handle
 * @param log_probs  Log-softmax output, shape [T, vocab_size], row-major
 * @param T          Number of time steps
 * @param vocab_size Vocabulary size (must match creation)
 * @param out        Output text buffer (null-terminated)
 * @param out_cap    Size of output buffer
 * @return           Number of characters written (excluding null), or -1 on error
 */
int ctc_beam_decode(CTCBeamDecoder *dec,
                    const float *log_probs,
                    int T, int vocab_size,
                    char *out, int out_cap);

/**
 * Destroy decoder and free all resources.
 */
void ctc_beam_destroy(CTCBeamDecoder *dec);

/**
 * Return default configuration values.
 */
CTCBeamConfig ctc_beam_config_default(void);

#ifdef __cplusplus
}
#endif

#endif /* CTC_BEAM_DECODER_H */
