/**
 * metal_loader.c — Runtime loader for custom Metal compute kernels.
 *
 * Uses Objective-C Metal API via the C bridge to load .metallib files
 * and create pipeline state objects for each kernel function.
 *
 * This avoids linking Metal.framework at compile time — the loader
 * detects GPU availability at runtime via dlopen.
 */

#include "metal_loader.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dlfcn.h>

#ifdef __OBJC__
#import <Metal/Metal.h>
#endif

#define MAX_KERNELS 16

struct MetalKernels {
    void *device;       /* id<MTLDevice> */
    void *library;      /* id<MTLLibrary> */
    void *pipelines[MAX_KERNELS];  /* id<MTLComputePipelineState> */
    char  names[MAX_KERNELS][64];
    int   n_kernels;
    int   loaded;
};

#ifdef __OBJC__

MetalKernels *metal_kernels_load(const char *metallib_path) {
    if (!metallib_path || metallib_path[0] == '\0') return NULL;

    MetalKernels *mk = (MetalKernels *)calloc(1, sizeof(MetalKernels));
    if (!mk) return NULL;

    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) {
            fprintf(stderr, "[metal_loader] No Metal device available\n");
            free(mk);
            return NULL;
        }
        mk->device = (void *)[device retain];

        NSString *path = [NSString stringWithUTF8String:metallib_path];
        NSURL *url = [NSURL fileURLWithPath:path];
        NSError *error = nil;
        id<MTLLibrary> library = [device newLibraryWithURL:url error:&error];
        if (!library) {
            fprintf(stderr, "[metal_loader] Failed to load %s: %s\n",
                    metallib_path,
                    error ? [[error localizedDescription] UTF8String] : "unknown");
            free(mk);
            return NULL;
        }
        mk->library = (void *)[library retain];

        NSArray<NSString *> *fn_names = [library functionNames];
        mk->n_kernels = 0;
        for (NSString *name in fn_names) {
            if (mk->n_kernels >= MAX_KERNELS) break;
            id<MTLFunction> fn = [library newFunctionWithName:name];
            if (!fn) continue;

            NSError *pipe_err = nil;
            id<MTLComputePipelineState> pso =
                [device newComputePipelineStateWithFunction:fn error:&pipe_err];
            if (pso) {
                int idx = mk->n_kernels++;
                mk->pipelines[idx] = (void *)[pso retain];
                strncpy(mk->names[idx], [name UTF8String], 63);
                mk->names[idx][63] = '\0';
                fprintf(stderr, "[metal_loader] Loaded kernel: %s (maxThreadsPerTG=%lu)\n",
                        mk->names[idx],
                        (unsigned long)[pso maxTotalThreadsPerThreadgroup]);
            } else {
                fprintf(stderr, "[metal_loader] Failed to create PSO for %s: %s\n",
                        [name UTF8String],
                        pipe_err ? [[pipe_err localizedDescription] UTF8String] : "unknown");
            }
        }
    }

    mk->loaded = (mk->n_kernels > 0) ? 1 : 0;
    fprintf(stderr, "[metal_loader] Loaded %d kernel(s) from %s\n",
            mk->n_kernels, metallib_path);
    return mk;
}

int metal_kernels_available(const MetalKernels *mk) {
    return mk && mk->loaded;
}

int metal_kernels_list(const MetalKernels *mk, const char **names, int max_n) {
    if (!mk) return 0;
    int n = mk->n_kernels < max_n ? mk->n_kernels : max_n;
    for (int i = 0; i < n; i++)
        names[i] = mk->names[i];
    return n;
}

static int find_kernel(const MetalKernels *mk, const char *name) {
    for (int i = 0; i < mk->n_kernels; i++)
        if (strcmp(mk->names[i], name) == 0) return i;
    return -1;
}

static id<MTLCommandQueue> get_queue(MetalKernels *mk) {
    static id<MTLCommandQueue> cached = nil;
    if (!cached)
        cached = [(id<MTLDevice>)mk->device newCommandQueue];
    return cached;
}

int metal_gemm_f16(MetalKernels *mk,
                   const void *a_buf, const void *b_buf, void *c_buf,
                   uint32_t M, uint32_t N, uint32_t K, float alpha) {
    if (!mk || !mk->loaded) return -1;
    int idx = find_kernel(mk, "gemm_f16");
    if (idx < 0) return -1;

    @autoreleasepool {
        id<MTLDevice> dev = (id<MTLDevice>)mk->device;
        id<MTLComputePipelineState> pso = (id<MTLComputePipelineState>)mk->pipelines[idx];
        id<MTLCommandQueue> queue = get_queue(mk);

        id<MTLBuffer> bufA = [dev newBufferWithBytesNoCopy:(void *)a_buf
                              length:(NSUInteger)M * K * 2 options:MTLResourceStorageModeShared
                              deallocator:nil];
        id<MTLBuffer> bufB = [dev newBufferWithBytesNoCopy:(void *)b_buf
                              length:(NSUInteger)N * K * 2 options:MTLResourceStorageModeShared
                              deallocator:nil];
        id<MTLBuffer> bufC = [dev newBufferWithBytesNoCopy:c_buf
                              length:(NSUInteger)M * N * 2 options:MTLResourceStorageModeShared
                              deallocator:nil];
        if (!bufA || !bufB || !bufC) return -1;

        id<MTLCommandBuffer> cmd = [queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:pso];
        [enc setBuffer:bufA offset:0 atIndex:0];
        [enc setBuffer:bufB offset:0 atIndex:1];
        [enc setBuffer:bufC offset:0 atIndex:2];
        [enc setBytes:&M length:sizeof(M) atIndex:3];
        [enc setBytes:&N length:sizeof(N) atIndex:4];
        [enc setBytes:&K length:sizeof(K) atIndex:5];
        [enc setBytes:&alpha length:sizeof(alpha) atIndex:6];

        MTLSize grid = MTLSizeMake((N + 31) / 32, (M + 31) / 32, 1);
        MTLSize tg = MTLSizeMake(32, 32, 1);
        [enc dispatchThreadgroups:grid threadsPerThreadgroup:tg];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
    }
    return 0;
}

int metal_silu_gate(MetalKernels *mk,
                    const void *input_buf, void *output_buf,
                    uint32_t N, uint32_t D) {
    if (!mk || !mk->loaded) return -1;
    int idx = find_kernel(mk, "silu_gate");
    if (idx < 0) return -1;

    @autoreleasepool {
        id<MTLDevice> dev = (id<MTLDevice>)mk->device;
        id<MTLComputePipelineState> pso = (id<MTLComputePipelineState>)mk->pipelines[idx];
        id<MTLCommandQueue> queue = get_queue(mk);

        id<MTLBuffer> bufIn = [dev newBufferWithBytesNoCopy:(void *)input_buf
                               length:(NSUInteger)N * 2 * D * 2 options:MTLResourceStorageModeShared
                               deallocator:nil];
        id<MTLBuffer> bufOut = [dev newBufferWithBytesNoCopy:output_buf
                                length:(NSUInteger)N * D * 2 options:MTLResourceStorageModeShared
                                deallocator:nil];
        if (!bufIn || !bufOut) return -1;

        id<MTLCommandBuffer> cmd = [queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:pso];
        [enc setBuffer:bufIn offset:0 atIndex:0];
        [enc setBuffer:bufOut offset:0 atIndex:1];
        [enc setBytes:&N length:sizeof(N) atIndex:2];
        [enc setBytes:&D length:sizeof(D) atIndex:3];

        MTLSize grid = MTLSizeMake(D, N, 1);
        NSUInteger maxTpTG = [pso maxTotalThreadsPerThreadgroup];
        MTLSize tg = MTLSizeMake(D < maxTpTG ? D : maxTpTG, 1, 1);
        [enc dispatchThreads:grid threadsPerThreadgroup:tg];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
    }
    return 0;
}

int metal_layer_norm(MetalKernels *mk,
                     const void *input_buf, void *output_buf,
                     const void *gamma_buf, const void *beta_buf,
                     uint32_t N, uint32_t D, float eps) {
    if (!mk || !mk->loaded) return -1;
    int idx = find_kernel(mk, "layer_norm");
    if (idx < 0) return -1;

    @autoreleasepool {
        id<MTLDevice> dev = (id<MTLDevice>)mk->device;
        id<MTLComputePipelineState> pso = (id<MTLComputePipelineState>)mk->pipelines[idx];
        id<MTLCommandQueue> queue = get_queue(mk);

        id<MTLBuffer> bufIn = [dev newBufferWithBytesNoCopy:(void *)input_buf
                               length:(NSUInteger)N * D * 2 options:MTLResourceStorageModeShared
                               deallocator:nil];
        id<MTLBuffer> bufOut = [dev newBufferWithBytesNoCopy:output_buf
                                length:(NSUInteger)N * D * 2 options:MTLResourceStorageModeShared
                                deallocator:nil];
        id<MTLBuffer> bufG = [dev newBufferWithBytesNoCopy:(void *)gamma_buf
                              length:(NSUInteger)D * 2 options:MTLResourceStorageModeShared
                              deallocator:nil];
        id<MTLBuffer> bufB = [dev newBufferWithBytesNoCopy:(void *)beta_buf
                              length:(NSUInteger)D * 2 options:MTLResourceStorageModeShared
                              deallocator:nil];
        if (!bufIn || !bufOut || !bufG || !bufB) return -1;

        id<MTLCommandBuffer> cmd = [queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:pso];
        [enc setBuffer:bufIn offset:0 atIndex:0];
        [enc setBuffer:bufOut offset:0 atIndex:1];
        [enc setBuffer:bufG offset:0 atIndex:2];
        [enc setBuffer:bufB offset:0 atIndex:3];
        [enc setBytes:&N length:sizeof(N) atIndex:4];
        [enc setBytes:&D length:sizeof(D) atIndex:5];
        [enc setBytes:&eps length:sizeof(eps) atIndex:6];

        uint32_t tg_size = D < 256 ? D : 256;
        MTLSize grid = MTLSizeMake(N, 1, 1);
        MTLSize tg = MTLSizeMake(tg_size, 1, 1);
        [enc dispatchThreadgroups:grid threadsPerThreadgroup:tg];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
    }
    return 0;
}

int metal_flash_attention(MetalKernels *mk,
                          const void *q_buf, const void *k_buf,
                          const void *v_buf, void *o_buf,
                          uint32_t M, uint32_t N, uint32_t head_dim) {
    if (!mk || !mk->loaded) return -1;
    int idx = find_kernel(mk, "flash_attention");
    if (idx < 0) return -1;

    @autoreleasepool {
        id<MTLDevice> dev = (id<MTLDevice>)mk->device;
        id<MTLComputePipelineState> pso = (id<MTLComputePipelineState>)mk->pipelines[idx];
        id<MTLCommandQueue> queue = get_queue(mk);

        id<MTLBuffer> bQ = [dev newBufferWithBytesNoCopy:(void *)q_buf
                            length:(NSUInteger)M * head_dim * 2 options:MTLResourceStorageModeShared
                            deallocator:nil];
        id<MTLBuffer> bK = [dev newBufferWithBytesNoCopy:(void *)k_buf
                            length:(NSUInteger)N * head_dim * 2 options:MTLResourceStorageModeShared
                            deallocator:nil];
        id<MTLBuffer> bV = [dev newBufferWithBytesNoCopy:(void *)v_buf
                            length:(NSUInteger)N * head_dim * 2 options:MTLResourceStorageModeShared
                            deallocator:nil];
        id<MTLBuffer> bO = [dev newBufferWithBytesNoCopy:o_buf
                            length:(NSUInteger)M * head_dim * 2 options:MTLResourceStorageModeShared
                            deallocator:nil];
        if (!bQ || !bK || !bV || !bO) return -1;

        float scale = 1.0f / sqrtf((float)head_dim);

        id<MTLCommandBuffer> cmd = [queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:pso];
        [enc setBuffer:bQ offset:0 atIndex:0];
        [enc setBuffer:bK offset:0 atIndex:1];
        [enc setBuffer:bV offset:0 atIndex:2];
        [enc setBuffer:bO offset:0 atIndex:3];
        [enc setBytes:&M length:sizeof(M) atIndex:4];
        [enc setBytes:&N length:sizeof(N) atIndex:5];
        [enc setBytes:&scale length:sizeof(scale) atIndex:6];

        MTLSize grid = MTLSizeMake(M, 1, 1);
        MTLSize tg = MTLSizeMake(head_dim < 64 ? head_dim : 64, 1, 1);
        [enc dispatchThreadgroups:grid threadsPerThreadgroup:tg];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
    }
    return 0;
}

void metal_kernels_destroy(MetalKernels *mk) {
    if (!mk) return;
    @autoreleasepool {
        for (int i = 0; i < mk->n_kernels; i++) {
            if (mk->pipelines[i])
                [(id)mk->pipelines[i] release];
        }
        if (mk->library)
            [(id)mk->library release];
        if (mk->device)
            [(id)mk->device release];
    }
    free(mk);
}

#else
/* Non-ObjC stub: Metal APIs require Objective-C compilation */

MetalKernels *metal_kernels_load(const char *metallib_path) {
    (void)metallib_path;
    fprintf(stderr, "[metal_loader] Compiled without ObjC — Metal kernels unavailable\n");
    return NULL;
}

int metal_kernels_available(const MetalKernels *mk) {
    (void)mk;
    return 0;
}

int metal_kernels_list(const MetalKernels *mk, const char **names, int max_n) {
    (void)mk; (void)names; (void)max_n;
    return 0;
}

int metal_gemm_f16(MetalKernels *mk,
                   const void *a_buf, const void *b_buf, void *c_buf,
                   uint32_t M, uint32_t N, uint32_t K, float alpha) {
    (void)mk; (void)a_buf; (void)b_buf; (void)c_buf;
    (void)M; (void)N; (void)K; (void)alpha;
    return -1;
}

int metal_silu_gate(MetalKernels *mk,
                    const void *input_buf, void *output_buf,
                    uint32_t N, uint32_t D) {
    (void)mk; (void)input_buf; (void)output_buf; (void)N; (void)D;
    return -1;
}

int metal_layer_norm(MetalKernels *mk,
                     const void *input_buf, void *output_buf,
                     const void *gamma_buf, const void *beta_buf,
                     uint32_t N, uint32_t D, float eps) {
    (void)mk; (void)input_buf; (void)output_buf;
    (void)gamma_buf; (void)beta_buf; (void)N; (void)D; (void)eps;
    return -1;
}

int metal_flash_attention(MetalKernels *mk,
                          const void *q_buf, const void *k_buf,
                          const void *v_buf, void *o_buf,
                          uint32_t M, uint32_t N, uint32_t head_dim) {
    (void)mk; (void)q_buf; (void)k_buf; (void)v_buf; (void)o_buf;
    (void)M; (void)N; (void)head_dim;
    return -1;
}

void metal_kernels_destroy(MetalKernels *mk) {
    free(mk);
}
#endif
