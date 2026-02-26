/**
 * bnns_conformer.h — BNNS Graph accelerated Conformer inference (Apple Silicon).
 *
 * Provides an optimized forward pass for the Conformer encoder using Apple's
 * BNNS (Basic Neural Network Subroutines) framework. BNNS automatically
 * dispatches to the best available compute unit (CPU/AMX/ANE) and performs
 * graph-level optimizations including layer fusion and copy elision.
 *
 * This is used as an alternative inference backend when available, falling
 * back to the handwritten cblas_sgemm/vDSP path otherwise.
 */

#ifndef BNNS_CONFORMER_H
#define BNNS_CONFORMER_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct BNNSConformer BNNSConformer;

/**
 * Create a BNNS-accelerated Conformer encoder.
 *
 * @param n_layers   Number of conformer blocks
 * @param d_model    Model dimension
 * @param n_heads    Number of attention heads
 * @param ff_mult    FFN expansion multiplier
 * @param conv_kernel Depthwise convolution kernel size
 * @param vocab_size  Output vocabulary size (for CTC head)
 * @return           Opaque handle, or NULL if BNNS Graph isn't available
 */
BNNSConformer *bnns_conformer_create(int n_layers, int d_model, int n_heads,
                                      int ff_mult, int conv_kernel, int vocab_size);

/**
 * Load weights from the mmap'd .cstt file into BNNS graph tensors.
 * For BNNSGraph path, this is a no-op (weights embedded in mlmodelc).
 */
int bnns_conformer_load_weights(BNNSConformer *bc, const void *weights,
                                 size_t weight_size, int is_fp16);

/**
 * Load a compiled CoreML model (.mlmodelc) for BNNSGraph execution.
 * The .mlmodelc contains all weights and the computation graph.
 *
 * @param bc    BNNS conformer handle
 * @param path  Path to .mlmodelc directory
 * @return      0 on success, -1 on failure
 */
int bnns_conformer_load_mlmodelc(BNNSConformer *bc, const char *path);

/**
 * Run the BNNS-accelerated forward pass.
 *
 * @param bc        BNNS conformer handle
 * @param mel_in    Input mel features [T, n_mels], row-major float32
 * @param T         Number of input time frames
 * @param n_mels    Number of mel bins
 * @param logits_out Output logits [T_sub, vocab_size], caller-allocated
 * @param max_T_sub Maximum output time steps (buffer size)
 * @return          Number of output time steps, or -1 on error
 */
int bnns_conformer_forward(BNNSConformer *bc, const float *mel_in, int T,
                            int n_mels, float *logits_out, int max_T_sub);

/**
 * Destroy and free all BNNS resources.
 */
void bnns_conformer_destroy(BNNSConformer *bc);

/**
 * Check if BNNS Graph is available on this system (macOS 15+).
 * @return 1 if available, 0 otherwise
 */
int bnns_conformer_available(void);

#ifdef __cplusplus
}
#endif

#endif /* BNNS_CONFORMER_H */
