/**
 * sonata_storm.h — SoundStorm parallel semantic token predictor.
 *
 * MaskGIT-style iterative refinement: predicts ALL semantic tokens simultaneously
 * and refines them over multiple rounds. Much faster than autoregressive LM for
 * long sequences.
 *
 * Drop-in replacement for sonata_lm in the Sonata TTS pipeline:
 *   Text → SoundStorm → Semantic Tokens → Flow → iSTFT → Audio
 *
 * Requires: Rust cdylib (src/sonata_storm/) built with Metal support.
 */

#ifndef SONATA_STORM_H
#define SONATA_STORM_H

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Create a SoundStorm engine from safetensors weights + JSON config.
 * Returns opaque engine handle, or NULL on failure.
 * Performs Metal warmup at load time.
 */
void *sonata_storm_create(const char *weights_path, const char *config_path);

/** Destroy engine and free all resources. */
void sonata_storm_destroy(void *engine);

/**
 * Set input text token IDs (from SentencePiece tokenizer).
 * Must be called before generate(). Resets internal state.
 * @return 0 on success, -1 on error
 */
int sonata_storm_set_text(void *engine, const unsigned int *text_ids, int n);

/**
 * Generate semantic tokens in parallel (non-autoregressive).
 * All tokens are produced at once via iterative MaskGIT refinement.
 *
 * @param engine      Engine handle
 * @param out_tokens  Output buffer for semantic token IDs
 * @param max_tokens  Maximum number of tokens to generate
 * @param out_count   Actual number of tokens produced
 * @return            0 on success, -1 on error
 */
int sonata_storm_generate(void *engine, int *out_tokens, int max_tokens, int *out_count);

/**
 * Set generation parameters.
 * @param temperature  Sampling temperature (0 = argmax, default 0.8)
 * @param n_rounds     MaskGIT refinement rounds (default 8, more = better quality)
 * @return             0 on success, -1 on error
 */
int sonata_storm_set_params(void *engine, float temperature, int n_rounds);

/** Reset engine state for a new utterance. */
int sonata_storm_reset(void *engine);

/** Audio sample rate (24000 Hz). */
int sonata_storm_sample_rate(void);

/** Semantic token frame rate (50 Hz). */
int sonata_storm_frame_rate(void);

#ifdef __cplusplus
}
#endif

#endif /* SONATA_STORM_H */
