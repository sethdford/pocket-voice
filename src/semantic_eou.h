/**
 * semantic_eou.h — Lightweight text-based sentence completion predictor.
 *
 * 5th signal for fused EOU: "Is this sentence linguistically complete?"
 *
 * Architecture: byte-level embedding → LayerNorm → 1-layer LSTM → Linear → sigmoid
 * ~33K params, <1ms inference on Apple Silicon.
 *
 * Examples:
 *   "Can you tell me"          → P(complete) ≈ 0.1  (incomplete)
 *   "The answer is yes"        → P(complete) ≈ 0.9  (complete)
 *   "I think that"             → P(complete) ≈ 0.15 (incomplete)
 *   "That sounds great"        → P(complete) ≈ 0.85 (complete)
 *
 * The model processes the last MAX_SEQ_LEN bytes of the transcript,
 * encoding each byte as a token (vocabulary = 256).
 */

#ifndef SEMANTIC_EOU_H
#define SEMANTIC_EOU_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct SemanticEOU SemanticEOU;

/* Default architecture constants */
#define SEOU_VOCAB_SIZE   256
#define SEOU_EMBED_DIM     32
#define SEOU_HIDDEN_DIM    64
#define SEOU_MAX_SEQ_LEN  128

/**
 * Create a semantic EOU predictor.
 * Allocates all buffers. Weights initialized to zero (must load or init).
 * Returns NULL on allocation failure.
 */
SemanticEOU *semantic_eou_create(void);

/**
 * Destroy and free all resources. Safe to call with NULL.
 */
void semantic_eou_destroy(SemanticEOU *se);

/**
 * Load trained weights from a binary file.
 *
 * Binary format (little-endian):
 *   Header: magic(4) version(4) vocab(4) embed(4) hidden(4) seq_len(4)
 *   Data:   embedding[V×E] norm_w[E] norm_b[E]
 *           lstm_wi[4H×E] lstm_wh[4H×H] lstm_bias[4H]
 *           out_w[H] out_b[1]
 *
 * @return 0 on success, -1 on error
 */
int semantic_eou_load_weights(SemanticEOU *se, const char *path);

/**
 * Initialize with random weights (Xavier). For testing only.
 */
void semantic_eou_init_random(SemanticEOU *se, uint32_t seed);

/**
 * Process a text transcript and return sentence completion probability.
 *
 * Takes the last SEOU_MAX_SEQ_LEN bytes of the text, runs the LSTM
 * over each byte embedding, and outputs P(complete) ∈ [0, 1].
 *
 * @param text  UTF-8 transcript (NULL-safe, returns 0.5)
 * @return Completion probability [0, 1]. 0.5 = neutral/unknown.
 */
float semantic_eou_process(SemanticEOU *se, const char *text);

/**
 * Reset LSTM hidden state. Call between utterances.
 */
void semantic_eou_reset(SemanticEOU *se);

/**
 * Count whitespace-delimited words in text.
 * Used by fusion logic to gate semantic signal (needs 3+ words).
 */
int semantic_eou_word_count(const char *text);

#ifdef __cplusplus
}
#endif

#endif /* SEMANTIC_EOU_H */
