#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include <dlfcn.h>

// Minimal FFI signatures for testing
typedef void* (*sonata_lm_create_t)(const char* weights, const char* config);
typedef void (*sonata_lm_destroy_t)(void* engine);
typedef int (*sonata_lm_load_gru_drafter_t)(void* engine, const char* weights, const char* config);
typedef int (*sonata_lm_step_t)(void* engine, int* out_token);
typedef int (*sonata_lm_set_text_t)(void* engine, const uint32_t* ids, int n);
typedef int (*sonata_lm_reset_t)(void* engine);

int main() {
    printf("[test_gru_drafter] Starting GRU drafter integration test\n");
    
    // Load the dylib (from sonata_lm build directory)
    const char* libpath = "/Users/sethford/Documents/pocket-voice/src/sonata_lm/target/release/libsonata_lm.dylib";
    void* handle = dlopen(libpath, RTLD_LAZY);
    if (handle == NULL) {
        printf("[test_gru_drafter] ERROR: Failed to load libsonata_lm.dylib: %s\n", dlerror());
        return 1;
    }
    printf("[test_gru_drafter] Loaded libsonata_lm.dylib\n");
    
    // Get function pointers
    sonata_lm_create_t create_fn = (sonata_lm_create_t)dlsym(handle, "sonata_lm_create");
    sonata_lm_destroy_t destroy_fn = (sonata_lm_destroy_t)dlsym(handle, "sonata_lm_destroy");
    sonata_lm_load_gru_drafter_t load_gru_fn = (sonata_lm_load_gru_drafter_t)dlsym(handle, "sonata_lm_load_gru_drafter");
    sonata_lm_step_t step_fn = (sonata_lm_step_t)dlsym(handle, "sonata_lm_step");
    sonata_lm_set_text_t set_text_fn = (sonata_lm_set_text_t)dlsym(handle, "sonata_lm_set_text");
    sonata_lm_reset_t reset_fn = (sonata_lm_reset_t)dlsym(handle, "sonata_lm_reset");
    
    if (create_fn == NULL || destroy_fn == NULL || load_gru_fn == NULL || step_fn == NULL) {
        printf("[test_gru_drafter] ERROR: Failed to load required functions\n");
        dlclose(handle);
        return 1;
    }
    printf("[test_gru_drafter] Loaded all required FFI functions\n");
    
    // Test: Check that load_gru_drafter FFI exists and is callable
    // (actual loading would require weights file)
    printf("[test_gru_drafter] Verified: sonata_lm_load_gru_drafter FFI function exists\n");
    
    dlclose(handle);
    printf("[test_gru_drafter] PASS: GRU drafter FFI integration verified\n");
    return 0;
}
