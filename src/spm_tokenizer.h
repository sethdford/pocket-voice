/**
 * spm_tokenizer.h — Minimal SentencePiece unigram tokenizer in C.
 *
 * Reads the vocabulary from a pre-extracted table embedded in the .ctts file
 * (piece string + score pairs). Implements the Viterbi-based unigram model
 * tokenization algorithm without requiring libsentencepiece.
 */

#ifndef SPM_TOKENIZER_H
#define SPM_TOKENIZER_H

#include <stdint.h>

typedef struct SPMTokenizer SPMTokenizer;

/**
 * Create tokenizer from a sentencepiece .model protobuf blob.
 * @param model_data  Raw bytes of the .model file
 * @param model_size  Size in bytes
 * @return Tokenizer handle, or NULL on failure
 */
SPMTokenizer *spm_create(const uint8_t *model_data, uint32_t model_size);

/**
 * Create tokenizer from pre-extracted vocab table.
 * @param pieces   Array of null-terminated piece strings
 * @param scores   Array of float scores (log-probability)
 * @param n_pieces Number of vocabulary entries
 * @return Tokenizer handle, or NULL on failure
 */
SPMTokenizer *spm_create_from_vocab(const char **pieces, const float *scores,
                                     int n_pieces);

void spm_destroy(SPMTokenizer *tok);

/**
 * Encode text to token IDs using Viterbi unigram algorithm.
 * @param tok        Tokenizer
 * @param text       Input UTF-8 text
 * @param out_ids    Output buffer for token IDs
 * @param max_ids    Capacity of out_ids
 * @return Number of tokens written, or -1 on error
 */
int spm_encode(const SPMTokenizer *tok, const char *text,
               int32_t *out_ids, int max_ids);

/**
 * Decode token IDs back to text.
 * @param tok        Tokenizer
 * @param ids        Array of token IDs
 * @param n_ids      Number of token IDs
 * @param out_text   Output buffer
 * @param out_cap    Capacity of output buffer
 * @return Length of decoded text, or -1 on error
 */
int spm_decode(const SPMTokenizer *tok, const int32_t *ids, int n_ids,
               char *out_text, int out_cap);

/** Get vocabulary size. */
int spm_vocab_size(const SPMTokenizer *tok);

#endif /* SPM_TOKENIZER_H */
