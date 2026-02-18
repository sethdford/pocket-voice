/**
 * sentence_buffer.h — Sentence buffer for streaming LLM token accumulation.
 *
 * Accumulates text tokens from streaming LLM responses and flushes at sentence
 * boundaries or clause boundaries (speculative mode). Filters out code blocks
 * and markdown so TTS only speaks prose.
 */

#ifndef SENTENCE_BUFFER_H
#define SENTENCE_BUFFER_H

#ifdef __cplusplus
extern "C" {
#endif

#define SENTBUF_MODE_SENTENCE    0
#define SENTBUF_MODE_SPECULATIVE 1

typedef struct SentenceBuffer SentenceBuffer;

/**
 * Create a sentence buffer.
 * @param mode       SENTBUF_MODE_SENTENCE or SENTBUF_MODE_SPECULATIVE
 * @param min_words  Minimum words before clause flush in speculative mode (default 5)
 * @return Opaque handle, or NULL on allocation failure.
 */
SentenceBuffer *sentbuf_create(int mode, int min_words);

/** Destroy the sentence buffer and free all resources. NULL-safe. */
void sentbuf_destroy(SentenceBuffer *sb);

/** Add a text token/chunk from the LLM stream. */
void sentbuf_add(SentenceBuffer *sb, const char *text, int len);

/** Returns 1 if at least one segment is ready to flush. */
int sentbuf_has_segment(const SentenceBuffer *sb);

/**
 * Pop the next ready segment.
 * Writes null-terminated text to out. Returns bytes written (excluding NUL).
 * Returns 0 if no segments are ready.
 */
int sentbuf_flush(SentenceBuffer *sb, char *out, int out_cap);

/**
 * Flush everything including partial buffer (for end-of-response).
 * Returns all remaining text joined with spaces.
 */
int sentbuf_flush_all(SentenceBuffer *sb, char *out, int out_cap);

/** Reset all state for a new conversation turn. */
void sentbuf_reset(SentenceBuffer *sb);

/**
 * Get predicted sentence length (in chars) based on EMA of recent sentences.
 * Returns 0 if no sentences have been observed yet.
 */
int sentbuf_predicted_length(const SentenceBuffer *sb);

/**
 * Get the number of sentences flushed so far this turn.
 * Can be used to detect "warm-up" phase (first 1-2 sentences should
 * use smaller speculative thresholds for faster first-chunk latency).
 */
int sentbuf_sentence_count(const SentenceBuffer *sb);

/**
 * Enable adaptive mode: first N sentences use aggressive flushing
 * (lower min_words) for faster first-chunk latency, then switch to
 * normal thresholds for steady-state quality.
 *
 * @param sb          Buffer handle
 * @param warmup_n    Number of warmup sentences (default 2)
 * @param warmup_min  Min words during warmup (default 3)
 */
void sentbuf_set_adaptive(SentenceBuffer *sb, int warmup_n, int warmup_min);

#ifdef __cplusplus
}
#endif

#endif /* SENTENCE_BUFFER_H */
