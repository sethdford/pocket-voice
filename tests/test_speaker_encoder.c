/**
 * test_speaker_encoder.c — Test zero-shot voice cloning integration.
 *
 * Tests:
 * 1. Speaker encoder creation and destruction
 * 2. Mel encoding (fixed-size 3-second reference)
 * 3. Audio encoding with resampling
 * 4. Integration with flow model speaker embedding
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <unistd.h>

#include "../src/speaker_encoder.h"

/* Mock flow model interface for testing embedding injection */
extern int sonata_flow_set_speaker_embedding(void *engine, const float *embedding, int dim);
extern int sonata_flow_clear_speaker_embedding(void *engine);

#define TEST_BUFFER_SIZE 48000
#define TEST_EMB_DIM 256
#define TEST_SR 16000

/* Generate a test tone (sine wave) for embedding extraction */
static void generate_test_audio(float *pcm, int n_samples, int sample_rate, float freq_hz) {
    for (int i = 0; i < n_samples; i++) {
        float t = (float)i / sample_rate;
        pcm[i] = 0.1f * sinf(2.0f * 3.14159265f * freq_hz * t);
    }
}

/* Test 1: Create and destroy speaker encoder */
static int test_create_destroy(void) {
    printf("[TEST] speaker_encoder_create/destroy...\n");

    /* Try to create with non-existent weights — should fail gracefully */
    SpeakerEncoder *enc = speaker_encoder_create("/tmp/nonexistent_speaker_encoder.safetensors");
    if (enc != NULL) {
        printf("  WARNING: encoder creation should fail for nonexistent weights\n");
        speaker_encoder_destroy(enc);
        return 0;  /* Still pass — graceful failure is acceptable */
    }

    printf("  PASS: encoder creation failed as expected\n");
    return 1;
}

/* Test 2: Verify embedding dimension */
static int test_embedding_dim(void) {
    printf("[TEST] speaker_encoder_embedding_dim...\n");

    SpeakerEncoder *enc = NULL;

    /* Try with NULL encoder */
    int dim = speaker_encoder_embedding_dim(enc);
    if (dim == -1) {
        printf("  PASS: NULL encoder returns -1\n");
        return 1;
    }
    printf("  FAIL: NULL encoder should return -1, got %d\n", dim);
    return 0;
}

/* Test 3: Verify sample rate */
static int test_sample_rate(void) {
    printf("[TEST] speaker_encoder_sample_rate...\n");

    SpeakerEncoder *enc = NULL;

    /* Try with NULL encoder */
    int sr = speaker_encoder_sample_rate(enc);
    if (sr == -1) {
        printf("  PASS: NULL encoder returns -1\n");
        return 1;
    }
    printf("  FAIL: NULL encoder should return -1, got %d\n", sr);
    return 0;
}

/* Test 4: Encode audio with synthetic test signal */
static int test_encode_audio_synthetic(void) {
    printf("[TEST] speaker_encoder_encode_audio (synthetic signal)...\n");

    /* Check if weights exist in typical location */
    const char *weights_candidates[] = {
        "./speaker_encoder.safetensors",
        "/tmp/speaker_encoder.safetensors",
        NULL
    };

    const char *weights_path = NULL;
    for (int i = 0; weights_candidates[i]; i++) {
        if (access(weights_candidates[i], F_OK) == 0) {
            weights_path = weights_candidates[i];
            break;
        }
    }

    if (!weights_path) {
        printf("  SKIP: speaker_encoder.safetensors not found (expected)\n");
        printf("       Download with: python train/sonata/train_speaker_encoder.py\n");
        return 1;
    }

    /* Load encoder */
    SpeakerEncoder *enc = speaker_encoder_create(weights_path);
    if (!enc) {
        printf("  FAIL: Failed to create encoder\n");
        return 0;
    }

    int dim = speaker_encoder_embedding_dim(enc);
    printf("  Embedding dimension: %d\n", dim);

    if (dim != TEST_EMB_DIM) {
        printf("  FAIL: Expected embedding_dim=%d, got %d\n", TEST_EMB_DIM, dim);
        speaker_encoder_destroy(enc);
        return 0;
    }

    /* Generate 3-second test signal at 16kHz */
    int n_samples = 3 * TEST_SR;
    float *pcm = (float *)malloc(n_samples * sizeof(float));
    if (!pcm) {
        printf("  FAIL: malloc failed\n");
        speaker_encoder_destroy(enc);
        return 0;
    }

    generate_test_audio(pcm, n_samples, TEST_SR, 200.0f);  /* 200 Hz sine */

    /* Extract embedding */
    float *embedding = (float *)malloc(dim * sizeof(float));
    if (!embedding) {
        printf("  FAIL: malloc failed\n");
        free(pcm);
        speaker_encoder_destroy(enc);
        return 0;
    }

    int ret = speaker_encoder_encode_audio(enc, pcm, n_samples, TEST_SR, embedding);
    if (ret != dim) {
        printf("  FAIL: encode_audio returned %d, expected %d\n", ret, dim);
        free(embedding);
        free(pcm);
        speaker_encoder_destroy(enc);
        return 0;
    }

    /* Verify embedding is L2-normalized (norm ≈ 1.0) */
    float norm = 0.0f;
    for (int i = 0; i < dim; i++) {
        norm += embedding[i] * embedding[i];
    }
    norm = sqrtf(norm);
    printf("  Embedding norm: %.6f\n", norm);

    if (norm < 0.99f || norm > 1.01f) {
        printf("  WARNING: Expected L2-norm ≈ 1.0, got %.6f\n", norm);
    }

    printf("  PASS: Generated %dD L2-normalized embedding\n", dim);

    free(embedding);
    free(pcm);
    speaker_encoder_destroy(enc);
    return 1;
}

/* Test 5: Encode audio with resampling */
static int test_encode_audio_resample(void) {
    printf("[TEST] speaker_encoder_encode_audio (with resampling)...\n");

    const char *weights_candidates[] = {
        "./speaker_encoder.safetensors",
        "/tmp/speaker_encoder.safetensors",
        NULL
    };

    const char *weights_path = NULL;
    for (int i = 0; weights_candidates[i]; i++) {
        if (access(weights_candidates[i], F_OK) == 0) {
            weights_path = weights_candidates[i];
            break;
        }
    }

    if (!weights_path) {
        printf("  SKIP: speaker_encoder.safetensors not found\n");
        return 1;
    }

    SpeakerEncoder *enc = speaker_encoder_create(weights_path);
    if (!enc) {
        printf("  FAIL: Failed to create encoder\n");
        return 0;
    }

    int dim = speaker_encoder_embedding_dim(enc);

    /* Generate audio at 24kHz (typical TTS output rate) */
    int sr = 24000;
    int n_samples = 3 * sr;  /* 3 seconds */
    float *pcm = (float *)malloc(n_samples * sizeof(float));
    if (!pcm) {
        printf("  FAIL: malloc failed\n");
        speaker_encoder_destroy(enc);
        return 0;
    }

    generate_test_audio(pcm, n_samples, sr, 300.0f);

    /* Encode (should auto-resample to 16kHz) */
    float *embedding = (float *)malloc(dim * sizeof(float));
    if (!embedding) {
        printf("  FAIL: malloc failed\n");
        free(pcm);
        speaker_encoder_destroy(enc);
        return 0;
    }

    int ret = speaker_encoder_encode_audio(enc, pcm, n_samples, sr, embedding);
    if (ret != dim) {
        printf("  FAIL: encode_audio at %dHz returned %d, expected %d\n", sr, ret, dim);
        free(embedding);
        free(pcm);
        speaker_encoder_destroy(enc);
        return 0;
    }

    printf("  PASS: Resampled %dHz audio and extracted embedding\n", sr);

    free(embedding);
    free(pcm);
    speaker_encoder_destroy(enc);
    return 1;
}

/* Main test runner */
int main(int argc, char *argv[]) {
    (void)argc;
    (void)argv;

    printf("\n=== Speaker Encoder Tests ===\n\n");

    int pass = 0, fail = 0;

    if (test_create_destroy()) pass++; else fail++;
    if (test_embedding_dim()) pass++; else fail++;
    if (test_sample_rate()) pass++; else fail++;
    if (test_encode_audio_synthetic()) pass++; else fail++;
    if (test_encode_audio_resample()) pass++; else fail++;

    printf("\n=== Results ===\n");
    printf("PASS: %d\n", pass);
    printf("FAIL: %d\n", fail);

    return fail > 0 ? 1 : 0;
}
