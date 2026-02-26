/*
 * sonata_refiner.h — Sonata STT Pass 2: Semantic token → text refiner.
 *
 * Encoder-decoder transformer that takes semantic tokens (from CTC/codec)
 * and generates text tokens autoregressively.
 *
 * Architecture:
 *   - Encoder: bidirectional self-attention on semantic tokens
 *   - Decoder: autoregressive with self-attention (RoPE) + cross-attention
 *   - GQA in decoder (n_heads=8, n_kv_heads=4)
 *   - RMSNorm, SiLU FFN
 *
 * Weight file: .cref (see scripts/export_sonata_refiner.py)
 */

#ifndef SONATA_REFINER_H
#define SONATA_REFINER_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct SonataRefiner SonataRefiner;

/* ─── Lifecycle ─────────────────────────────────────────────────────── */

SonataRefiner *sonata_refiner_create(const char *model_path);
void sonata_refiner_destroy(SonataRefiner *ref);
void sonata_refiner_reset(SonataRefiner *ref);

/* ─── Processing ───────────────────────────────────────────────────── */

/*
 * Refine: semantic tokens → text.
 * semantic_ids: array of semantic token IDs (from CTC/codec)
 * n_tokens: number of semantic tokens
 * out_text: output text buffer (null-terminated)
 * max_len: buffer size in bytes
 * Returns number of characters written (excluding null), or -1 on error.
 */
int sonata_refiner_process(SonataRefiner *ref,
                           const int *semantic_ids, int n_tokens,
                           char *out_text, int max_len);

/* ─── Properties ───────────────────────────────────────────────────── */

/* Text vocabulary size (for debug). */
int sonata_refiner_vocab_size(const SonataRefiner *ref);

#ifdef __cplusplus
}
#endif

#endif /* SONATA_REFINER_H */
