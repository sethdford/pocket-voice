/**
 * bnns_convnext_decoder.h — ANE-accelerated ConvNeXt decoder for Sonata TTS.
 *
 * Offloads the ConvNeXt decoder from Metal GPU to the Apple Neural Engine,
 * freeing the GPU for flow network inference. Uses BNNS Graph for automatic
 * dispatch to ANE with graph-level optimizations (layer fusion, copy elision).
 *
 * Pipeline: Flow (GPU) → ConvNeXt decoder (ANE) → iSTFT (AMX)
 * All three compute units run concurrently.
 */

#ifndef BNNS_CONVNEXT_DECODER_H
#define BNNS_CONVNEXT_DECODER_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct BNNSConvNeXtDecoder BNNSConvNeXtDecoder;

/**
 * Create a BNNS-accelerated ConvNeXt decoder.
 *
 * @param n_layers     Number of ConvNeXt blocks
 * @param dec_dim      Decoder hidden dimension
 * @param conv_kernel  Depthwise conv kernel size
 * @param ff_mult      Feed-forward expansion factor
 * @param input_dim    Input dimension (FSQ codes dim + acoustic latent dim)
 * @param n_fft        FFT size (output magnitude/phase bins = n_fft/2 + 1)
 * @return             Opaque handle, or NULL if BNNS Graph not available
 */
BNNSConvNeXtDecoder *bnns_convnext_create(int n_layers, int dec_dim,
                                            int conv_kernel, float ff_mult,
                                            int input_dim, int n_fft);

/**
 * Load weights from a safetensors file.
 *
 * @param dec   Handle from bnns_convnext_create
 * @param path  Path to safetensors weights file
 * @return      0 on success, -1 on failure
 */
int bnns_convnext_load_weights(BNNSConvNeXtDecoder *dec, const char *path);

/**
 * Load a compiled CoreML model for direct ANE execution.
 *
 * @param dec   Handle from bnns_convnext_create
 * @param path  Path to .mlmodelc directory
 * @return      0 on success, -1 on failure
 */
int bnns_convnext_load_mlmodelc(BNNSConvNeXtDecoder *dec, const char *path);

/**
 * Run decoder forward pass: (semantic_codes, acoustic_latent) → magnitude + inst_freq.
 *
 * @param dec           Decoder handle
 * @param semantic      Semantic FSQ codes [n_frames × fsq_dim], float32
 * @param acoustic      Acoustic latents [n_frames × acoustic_dim], float32
 * @param n_frames      Number of time frames
 * @param out_magnitude Output magnitude [n_frames × n_bins], float32 (caller allocates)
 * @param out_inst_freq Output instantaneous frequency [n_frames × n_bins], float32
 * @return              Number of frequency bins (n_fft/2+1), or 0 on error
 */
int bnns_convnext_forward(BNNSConvNeXtDecoder *dec,
                           const float *semantic, const float *acoustic,
                           int n_frames,
                           float *out_magnitude, float *out_inst_freq);

/**
 * Destroy decoder and free resources.
 */
void bnns_convnext_destroy(BNNSConvNeXtDecoder *dec);

#ifdef __cplusplus
}
#endif

#endif /* BNNS_CONVNEXT_DECODER_H */
