/**
 * bnns_conformer.c — BNNSGraph-accelerated Conformer encoder (macOS 15+).
 *
 * Uses Apple's BNNSGraph API (macOS 15+) to execute a Conformer encoder
 * compiled to .mlmodelc format. BNNSGraph is the low-level C runtime
 * underneath CoreML — it provides:
 *   - Automatic dispatch to CPU/AMX/ANE per operation
 *   - Graph-level fusion (linear+activation, conv+bn, etc.)
 *   - Zero-copy I/O with pre-allocated buffers
 *   - No Objective-C / Foundation dependency
 *
 * Workflow:
 *   1. Python: NeMo Conformer → ONNX → CoreML (.mlpackage) → compile (.mlmodelc)
 *   2. C: BNNSGraphCompileFromFile → BNNSGraphContextMake → BNNSGraphContextExecute
 *
 * Build:
 *   cc -O3 -shared -fPIC -arch arm64 -framework Accelerate \
 *      -install_name @rpath/libbnns_conformer.dylib \
 *      -o libbnns_conformer.dylib bnns_conformer.c
 */

#include <Accelerate/Accelerate.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#include "bnns_conformer.h"

#define MAX_ARGS 16

struct BNNSConformer {
    bnns_graph_t graph;
    bnns_graph_context_t ctx;

    size_t n_inputs;
    size_t n_outputs;
    size_t n_args;

    size_t input_pos;
    size_t output_pos;

    size_t workspace_size;
    char *workspace;

    int d_model;
    int vocab_size;
    int available;
};

int bnns_conformer_available(void) {
    struct stat st;
    return (stat("/System/Library/Frameworks/Accelerate.framework", &st) == 0);
}

BNNSConformer *bnns_conformer_create(int n_layers, int d_model, int n_heads,
                                      int ff_mult, int conv_kernel, int vocab_size) {
    BNNSConformer *bc = calloc(1, sizeof(BNNSConformer));
    if (!bc) return NULL;

    bc->d_model = d_model;
    bc->vocab_size = vocab_size;
    bc->available = 0;

    return bc;
}

/**
 * Load a compiled CoreML model (.mlmodelc) via BNNSGraph.
 * This replaces the traditional weight-loading path with a single
 * compiled graph that includes all weights and operations.
 */
static int load_mlmodelc(BNNSConformer *bc, const char *mlmodelc_path) {
    fprintf(stderr, "[bnns_conformer] Compiling graph from: %s\n", mlmodelc_path);

    bnns_graph_compile_options_t opts = BNNSGraphCompileOptionsMakeDefault();
    BNNSGraphCompileOptionsSetOptimizationPreference(
        opts, BNNSGraphOptimizationPreferencePerformance);

    bc->graph = BNNSGraphCompileFromFile(mlmodelc_path, NULL, opts);
    BNNSGraphCompileOptionsDestroy(opts);

    if (!bc->graph.data) {
        fprintf(stderr, "[bnns_conformer] Failed to compile graph from %s\n", mlmodelc_path);
        return -1;
    }

    bc->n_inputs = BNNSGraphGetInputCount(bc->graph, NULL);
    bc->n_outputs = BNNSGraphGetOutputCount(bc->graph, NULL);
    bc->n_args = BNNSGraphGetArgumentCount(bc->graph, NULL);

    fprintf(stderr, "[bnns_conformer] Graph compiled: %zu inputs, %zu outputs, %zu total args\n",
            bc->n_inputs, bc->n_outputs, bc->n_args);

    const char *names[MAX_ARGS];
    if (bc->n_inputs > 0 && bc->n_inputs <= MAX_ARGS) {
        BNNSGraphGetInputNames(bc->graph, NULL, bc->n_inputs, names);
        for (size_t i = 0; i < bc->n_inputs; i++)
            fprintf(stderr, "[bnns_conformer]   input[%zu]: %s\n", i, names[i] ? names[i] : "(null)");
        bc->input_pos = BNNSGraphGetArgumentPosition(bc->graph, NULL, names[0]);
    }
    if (bc->n_outputs > 0 && bc->n_outputs <= MAX_ARGS) {
        BNNSGraphGetOutputNames(bc->graph, NULL, bc->n_outputs, names);
        for (size_t i = 0; i < bc->n_outputs; i++)
            fprintf(stderr, "[bnns_conformer]   output[%zu]: %s\n", i, names[i] ? names[i] : "(null)");
        bc->output_pos = BNNSGraphGetArgumentPosition(bc->graph, NULL, names[0]);
    }

    bc->ctx = BNNSGraphContextMake(bc->graph);
    if (!bc->ctx.data) {
        fprintf(stderr, "[bnns_conformer] Failed to create execution context\n");
        return -1;
    }

    bc->workspace_size = BNNSGraphContextGetWorkspaceSize(bc->ctx, NULL);
    if (bc->workspace_size > 0 && bc->workspace_size != SIZE_MAX) {
        posix_memalign((void **)&bc->workspace, 4096, bc->workspace_size);
        fprintf(stderr, "[bnns_conformer] Workspace: %zu bytes\n", bc->workspace_size);
    }

    bc->available = 1;
    return 0;
}

int bnns_conformer_load_weights(BNNSConformer *bc, const void *weights,
                                 size_t weight_size, int is_fp16) {
    (void)weights;
    (void)weight_size;
    (void)is_fp16;

    if (!bc) return -1;

    /* For BNNSGraph path, weights are embedded in the .mlmodelc.
     * This function is a no-op — call bnns_conformer_load_mlmodelc instead. */
    fprintf(stderr, "[bnns_conformer] Note: weights embedded in mlmodelc, skipping explicit load\n");
    return 0;
}

/**
 * Public: load a .mlmodelc and prepare for inference.
 */
int bnns_conformer_load_mlmodelc(BNNSConformer *bc, const char *path) {
    if (!bc || !path) return -1;
    return load_mlmodelc(bc, path);
}

int bnns_conformer_forward(BNNSConformer *bc, const float *mel_in, int T,
                            int n_mels, float *logits_out, int max_T_sub) {
    if (!bc || !bc->available || !bc->ctx.data) return -1;
    if (!mel_in || !logits_out || T <= 0) return -1;

    /* Set dynamic shape for variable-length input */
    uint64_t in_shape[] = { (uint64_t)T, (uint64_t)n_mels };
    bnns_graph_shape_t shapes[MAX_ARGS];
    memset(shapes, 0, sizeof(shapes));

    shapes[bc->input_pos].rank = 2;
    shapes[bc->input_pos].shape = in_shape;

    int shape_result = BNNSGraphContextSetDynamicShapes(
        bc->ctx, NULL, bc->n_args, shapes);
    if (shape_result < 0) {
        fprintf(stderr, "[bnns_conformer] Failed to set dynamic shapes (T=%d, mels=%d)\n",
                T, n_mels);
        return -1;
    }

    /* Re-query workspace after shape change */
    size_t new_ws = BNNSGraphContextGetWorkspaceSize(bc->ctx, NULL);
    if (new_ws > bc->workspace_size && new_ws != SIZE_MAX) {
        free(bc->workspace);
        bc->workspace_size = new_ws;
        posix_memalign((void **)&bc->workspace, 4096, new_ws);
    }

    /* Build argument array */
    bnns_graph_argument_t args[MAX_ARGS];
    memset(args, 0, sizeof(args));

    args[bc->input_pos].data_ptr = (void *)mel_in;
    args[bc->input_pos].data_ptr_size = (size_t)T * n_mels * sizeof(float);

    args[bc->output_pos].data_ptr = logits_out;
    args[bc->output_pos].data_ptr_size = (size_t)max_T_sub * bc->vocab_size * sizeof(float);

    int rc = BNNSGraphContextExecute(
        bc->ctx, NULL, bc->n_args, args,
        bc->workspace_size, bc->workspace);

    if (rc != 0) {
        fprintf(stderr, "[bnns_conformer] Execution failed: %d\n", rc);
        return -1;
    }

    /* After subsampling (4x), output T_sub = T / 4 */
    int T_sub = T / 4;
    if (T_sub > max_T_sub) T_sub = max_T_sub;
    return T_sub;
}

void bnns_conformer_destroy(BNNSConformer *bc) {
    if (!bc) return;

    if (bc->ctx.data)
        BNNSGraphContextDestroy(bc->ctx);

    if (bc->graph.data)
        free(bc->graph.data);

    free(bc->workspace);
    free(bc);
}
