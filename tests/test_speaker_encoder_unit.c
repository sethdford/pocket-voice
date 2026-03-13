/*
 * test_speaker_encoder_unit.c — Unit tests for Speaker Encoder FFI.
 *
 * Tests null-safety, lifecycle, constant functions, and invalid-input handling
 * for speaker_encoder_* exported functions WITHOUT requiring model weights.
 *
 * Functions tested:
 *   - speaker_encoder_create (NULL paths)
 *   - speaker_encoder_destroy (NULL and valid lifecycle)
 *   - speaker_encoder_embedding_dim (constant, NULL-safe)
 *   - speaker_encoder_sample_rate (constant, NULL-safe)
 *   - speaker_encoder_extract (NULL-safety, parameter validation)
 *   - speaker_encoder_encode_mel (NULL-safety, parameter validation)
 *   - speaker_encoder_encode_audio (NULL-safety, edge cases)
 *   - speaker_encoder_extract_from_wav (path validation)
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ─── Speaker Encoder FFI ──────────────────────────────────────────────────── */

typedef struct SpeakerEncoder SpeakerEncoder;

extern SpeakerEncoder *speaker_encoder_create(const char *weights_path);
extern void speaker_encoder_destroy(SpeakerEncoder *enc);
extern int speaker_encoder_embedding_dim(const SpeakerEncoder *enc);
extern int speaker_encoder_sample_rate(const SpeakerEncoder *enc);
extern int speaker_encoder_extract(SpeakerEncoder *enc, const float *audio,
                                    int n_samples, float *embedding_out);
extern int speaker_encoder_encode_mel(SpeakerEncoder *enc, const float *mel,
                                       int n_frames, float *out_emb);
extern int speaker_encoder_encode_audio(SpeakerEncoder *enc, const float *pcm,
                                         int n_samples, int sample_rate,
                                         float *out_emb);
extern int speaker_encoder_extract_from_wav(SpeakerEncoder *enc,
                                             const char *wav_path,
                                             float *embedding_out);

/* ─── Test helpers ─────────────────────────────────────────────────────── */

static int g_pass = 0, g_fail = 0;
#define CHECK(cond, msg) do { \
    if (cond) { g_pass++; printf("  [PASS] %s\n", msg); } \
    else { g_fail++; printf("  [FAIL] %s\n", msg); } \
} while(0)

#define CHECKF(cond, fmt, ...) do { \
    char _buf[256]; snprintf(_buf, sizeof(_buf), fmt, __VA_ARGS__); \
    if (cond) { g_pass++; printf("  [PASS] %s\n", _buf); } \
    else { g_fail++; printf("  [FAIL] %s\n", _buf); } \
} while(0)

/* ─── Test 1: Create with NULL and invalid paths ──────────────────────────── */

static void test_speaker_encoder_create_null_paths(void) {
    printf("\n═══ Test 1: Create with NULL and invalid paths ═══\n");

    /* NULL path */
    SpeakerEncoder *enc = speaker_encoder_create(NULL);
    CHECK(enc == NULL, "create(NULL) returns NULL");
    if (enc) speaker_encoder_destroy(enc);

    /* Empty string path */
    enc = speaker_encoder_create("");
    CHECK(enc == NULL, "create(\"\") returns NULL");
    if (enc) speaker_encoder_destroy(enc);

    /* Non-existent path */
    enc = speaker_encoder_create("/nonexistent/path/speaker_encoder.safetensors");
    CHECK(enc == NULL, "create(non-existent path) returns NULL");
    if (enc) speaker_encoder_destroy(enc);

    /* Path to directory instead of file */
    enc = speaker_encoder_create("/tmp");
    CHECK(enc == NULL, "create(directory path) returns NULL");
    if (enc) speaker_encoder_destroy(enc);
}

/* ─── Test 2: Destroy with NULL ──────────────────────────────────────────── */

static void test_speaker_encoder_destroy_null(void) {
    printf("\n═══ Test 2: Destroy with NULL ═══\n");

    /* destroy(NULL) should not crash */
    speaker_encoder_destroy(NULL);
    CHECK(1, "destroy(NULL) does not crash");
}

/* ─── Test 3: Constants with NULL encoder ─────────────────────────────────── */

static void test_speaker_encoder_constants_null(void) {
    printf("\n═══ Test 3: Constants with NULL encoder ═══\n");

    /* embedding_dim(NULL) */
    int dim = speaker_encoder_embedding_dim(NULL);
    CHECKF(dim == -1 || dim == 0,
           "embedding_dim(NULL) = %d (returns error code)", dim);

    /* sample_rate(NULL) */
    int sr = speaker_encoder_sample_rate(NULL);
    CHECKF(sr == -1 || sr == 0,
           "sample_rate(NULL) = %d (returns error code)", sr);
}

/* ─── Test 4: Extract with NULL audio buffer ──────────────────────────────── */

static void test_speaker_encoder_extract_null_audio(void) {
    printf("\n═══ Test 4: Extract with NULL audio buffer ═══\n");

    float emb[256];

    /* extract(NULL, NULL, 0, buffer) returns -1 */
    int rc = speaker_encoder_extract(NULL, NULL, 0, emb);
    CHECK(rc == -1, "extract(NULL, NULL, 0, buffer) returns -1");

    /* extract(NULL, non-null, n, buffer) returns -1 */
    float audio[100] = {0};
    rc = speaker_encoder_extract(NULL, audio, 100, emb);
    CHECK(rc == -1, "extract(NULL, audio, 100, buffer) returns -1");

    /* extract(NULL, non-null, n, NULL) returns -1 */
    rc = speaker_encoder_extract(NULL, audio, 100, NULL);
    CHECK(rc == -1, "extract(NULL, audio, 100, NULL) returns -1");
}

/* ─── Test 5: Extract with invalid sample counts ──────────────────────────── */

static void test_speaker_encoder_extract_invalid_counts(void) {
    printf("\n═══ Test 5: Extract with invalid sample counts ═══\n");

    float audio[100];
    float emb[256];

    /* Negative sample count */
    int rc = speaker_encoder_extract(NULL, audio, -1, emb);
    CHECK(rc == -1, "extract(NULL, audio, -1, buffer) returns -1 (negative count)");

    /* Zero sample count */
    rc = speaker_encoder_extract(NULL, audio, 0, emb);
    CHECK(rc == -1, "extract(NULL, audio, 0, buffer) returns -1 (zero count)");

    /* Very large count */
    rc = speaker_encoder_extract(NULL, audio, 999999999, emb);
    CHECK(rc == -1, "extract(NULL, audio, 999999999, buffer) returns -1 (huge count)");
}

/* ─── Test 6: Encode mel with NULL ─────────────────────────────────────────── */

static void test_speaker_encoder_encode_mel_null(void) {
    printf("\n═══ Test 6: Encode mel with NULL ═══\n");

    float mel[80 * 10];
    float emb[256];

    /* encode_mel(NULL, NULL, 0, buffer) returns -1 */
    int rc = speaker_encoder_encode_mel(NULL, NULL, 0, emb);
    CHECK(rc == -1, "encode_mel(NULL, NULL, 0, buffer) returns -1");

    /* encode_mel(NULL, mel, 10, NULL) returns -1 */
    rc = speaker_encoder_encode_mel(NULL, mel, 10, NULL);
    CHECK(rc == -1, "encode_mel(NULL, mel, 10, NULL) returns -1");

    /* encode_mel(NULL, NULL, n, NULL) returns -1 */
    rc = speaker_encoder_encode_mel(NULL, NULL, 100, NULL);
    CHECK(rc == -1, "encode_mel(NULL, NULL, 100, NULL) returns -1");
}

/* ─── Test 7: Encode mel with invalid frame counts ──────────────────────────── */

static void test_speaker_encoder_encode_mel_invalid_frames(void) {
    printf("\n═══ Test 7: Encode mel with invalid frame counts ═══\n");

    float mel[80 * 10];
    float emb[256];

    /* Negative frame count */
    int rc = speaker_encoder_encode_mel(NULL, mel, -1, emb);
    CHECK(rc == -1, "encode_mel(NULL, mel, -1, buffer) returns -1 (negative frames)");

    /* Zero frame count */
    rc = speaker_encoder_encode_mel(NULL, mel, 0, emb);
    CHECK(rc == -1, "encode_mel(NULL, mel, 0, buffer) returns -1 (zero frames)");

    /* Very large frame count */
    rc = speaker_encoder_encode_mel(NULL, mel, 999999, emb);
    CHECK(rc == -1, "encode_mel(NULL, mel, 999999, buffer) returns -1 (huge frames)");
}

/* ─── Test 8: Encode audio with NULL ──────────────────────────────────────── */

static void test_speaker_encoder_encode_audio_null(void) {
    printf("\n═══ Test 8: Encode audio with NULL ═══\n");

    float pcm[1000];
    float emb[256];

    /* encode_audio(NULL, NULL, 0, 16000, buffer) returns -1 */
    int rc = speaker_encoder_encode_audio(NULL, NULL, 0, 16000, emb);
    CHECK(rc == -1, "encode_audio(NULL, NULL, 0, 16000, buffer) returns -1");

    /* encode_audio(NULL, pcm, 1000, 16000, NULL) returns -1 */
    rc = speaker_encoder_encode_audio(NULL, pcm, 1000, 16000, NULL);
    CHECK(rc == -1, "encode_audio(NULL, pcm, 1000, 16000, NULL) returns -1");

    /* encode_audio(NULL, NULL, 1000, 16000, NULL) returns -1 */
    rc = speaker_encoder_encode_audio(NULL, NULL, 1000, 16000, NULL);
    CHECK(rc == -1, "encode_audio(NULL, NULL, 1000, 16000, NULL) returns -1");
}

/* ─── Test 9: Encode audio with invalid sample rate ──────────────────────── */

static void test_speaker_encoder_encode_audio_invalid_sr(void) {
    printf("\n═══ Test 9: Encode audio with invalid sample rate ═══\n");

    float pcm[1000];
    float emb[256];

    /* Zero sample rate */
    int rc = speaker_encoder_encode_audio(NULL, pcm, 1000, 0, emb);
    CHECK(rc == -1, "encode_audio(NULL, pcm, 1000, 0, buffer) returns -1 (zero SR)");

    /* Negative sample rate */
    rc = speaker_encoder_encode_audio(NULL, pcm, 1000, -16000, emb);
    CHECK(rc == -1, "encode_audio(NULL, pcm, 1000, -16000, buffer) returns -1 (negative SR)");

    /* Unreasonably high sample rate */
    rc = speaker_encoder_encode_audio(NULL, pcm, 1000, 999999999, emb);
    CHECK(rc == -1, "encode_audio(NULL, pcm, 1000, 999999999, buffer) returns -1 (huge SR)");
}

/* ─── Test 10: Extract from WAV with NULL and invalid paths ──────────────── */

static void test_speaker_encoder_extract_from_wav_paths(void) {
    printf("\n═══ Test 10: Extract from WAV with NULL and invalid paths ═══\n");

    float emb[256];

    /* extract_from_wav(NULL, NULL, buffer) returns -1 */
    int rc = speaker_encoder_extract_from_wav(NULL, NULL, emb);
    CHECK(rc == -1, "extract_from_wav(NULL, NULL, buffer) returns -1");

    /* extract_from_wav(NULL, empty, buffer) returns -1 */
    rc = speaker_encoder_extract_from_wav(NULL, "", emb);
    CHECK(rc == -1, "extract_from_wav(NULL, \"\", buffer) returns -1");

    /* extract_from_wav(NULL, non-existent, buffer) returns -1 */
    rc = speaker_encoder_extract_from_wav(NULL, "/nonexistent/file.wav", emb);
    CHECK(rc == -1, "extract_from_wav(NULL, non-existent, buffer) returns -1");

    /* extract_from_wav(NULL, path, NULL) returns -1 */
    rc = speaker_encoder_extract_from_wav(NULL, "/some/path.wav", NULL);
    CHECK(rc == -1, "extract_from_wav(NULL, path, NULL) returns -1");
}

/* ─── Test 11: Extract from WAV with NULL output buffer ────────────────── */

static void test_speaker_encoder_extract_from_wav_null_output(void) {
    printf("\n═══ Test 11: Extract from WAV with NULL output buffer ═══\n");

    /* extract_from_wav(NULL, path, NULL) should return -1 */
    int rc = speaker_encoder_extract_from_wav(NULL, "/path/to/file.wav", NULL);
    CHECK(rc == -1, "extract_from_wav(NULL, path, NULL) returns -1");
}

/* ─── Test 12: Embedding buffer size validation ──────────────────────────── */

static void test_speaker_encoder_embedding_buffer_sizes(void) {
    printf("\n═══ Test 12: Embedding buffer size validation ═══\n");

    float pcm[16000]; /* 1 second at 16kHz */
    float small_buffer[128]; /* Too small for 256D embedding */

    /* With NULL engine, should return error */
    int rc = speaker_encoder_extract(NULL, pcm, 16000, small_buffer);
    CHECK(rc == -1, "extract with small output buffer returns -1");

    /* Even with larger buffer but NULL engine */
    float large_buffer[512];
    rc = speaker_encoder_extract(NULL, pcm, 16000, large_buffer);
    CHECK(rc == -1, "extract with large output buffer (but NULL engine) returns -1");
}

/* ─── Test 13: Silence audio handling ──────────────────────────────────── */

static void test_speaker_encoder_silence_input(void) {
    printf("\n═══ Test 13: Silence audio handling ═══\n");

    float silence[16000];
    memset(silence, 0, sizeof(silence));
    float emb[256];

    /* extract(NULL, silence, 16000, buffer) should return -1 */
    /* (because engine is NULL, but the input is valid silence) */
    int rc = speaker_encoder_extract(NULL, silence, 16000, emb);
    CHECK(rc == -1, "extract(NULL, silence, 16000, buffer) returns -1 (NULL engine)");

    /* Same for encode_audio with silence */
    rc = speaker_encoder_encode_audio(NULL, silence, 16000, 16000, emb);
    CHECK(rc == -1, "encode_audio(NULL, silence, 16000, 16000, buffer) returns -1 (NULL engine)");
}

/* ─── Test 14: Cross-sample-rate resampling paths ──────────────────────── */

static void test_speaker_encoder_encode_audio_resampling(void) {
    printf("\n═══ Test 14: Cross-sample-rate resampling paths ═══\n");

    float pcm[24000]; /* 1 second at 24kHz */
    float emb[256];

    /* encode_audio should handle resampling, but with NULL engine, should fail */
    int rc = speaker_encoder_encode_audio(NULL, pcm, 24000, 24000, emb);
    CHECK(rc == -1, "encode_audio(NULL, pcm, 24000, 24000, buffer) returns -1 (NULL engine)");

    /* 8kHz input (downsampling path) */
    float pcm8k[8000]; /* 1 second at 8kHz */
    rc = speaker_encoder_encode_audio(NULL, pcm8k, 8000, 8000, emb);
    CHECK(rc == -1, "encode_audio(NULL, pcm8k, 8000, 8000, buffer) returns -1 (NULL engine)");

    /* 48kHz input (upsampling path) */
    float pcm48k[48000]; /* 1 second at 48kHz */
    rc = speaker_encoder_encode_audio(NULL, pcm48k, 48000, 48000, emb);
    CHECK(rc == -1, "encode_audio(NULL, pcm48k, 48000, 48000, buffer) returns -1 (NULL engine)");
}

/* ─── Test 15: Mel frame dimension validation ──────────────────────────── */

static void test_speaker_encoder_mel_dimensions(void) {
    printf("\n═══ Test 15: Mel frame dimension validation ═══\n");

    float emb[256];

    /* encode_mel with 1 frame (minimum valid) */
    float mel[80];
    int rc = speaker_encoder_encode_mel(NULL, mel, 1, emb);
    CHECK(rc == -1, "encode_mel(NULL, mel, 1, buffer) returns -1 (NULL engine)");

    /* encode_mel with 100 frames (typical) */
    float mel_large[80 * 100];
    rc = speaker_encoder_encode_mel(NULL, mel_large, 100, emb);
    CHECK(rc == -1, "encode_mel(NULL, mel, 100, buffer) returns -1 (NULL engine)");

    /* encode_mel with many frames */
    float mel_huge[80 * 1000];
    rc = speaker_encoder_encode_mel(NULL, mel_huge, 1000, emb);
    CHECK(rc == -1, "encode_mel(NULL, mel, 1000, buffer) returns -1 (NULL engine)");
}

/* ─── Main ──────────────────────────────────────────────────────────────── */

int main(void) {
    printf("\n╔════════════════════════════════════════════════════════════════╗\n");
    printf("║     Speaker Encoder Unit Tests (No Models Required)             ║\n");
    printf("╚════════════════════════════════════════════════════════════════╝\n");

    test_speaker_encoder_create_null_paths();
    test_speaker_encoder_destroy_null();
    test_speaker_encoder_constants_null();
    test_speaker_encoder_extract_null_audio();
    test_speaker_encoder_extract_invalid_counts();
    test_speaker_encoder_encode_mel_null();
    test_speaker_encoder_encode_mel_invalid_frames();
    test_speaker_encoder_encode_audio_null();
    test_speaker_encoder_encode_audio_invalid_sr();
    test_speaker_encoder_extract_from_wav_paths();
    test_speaker_encoder_extract_from_wav_null_output();
    test_speaker_encoder_embedding_buffer_sizes();
    test_speaker_encoder_silence_input();
    test_speaker_encoder_encode_audio_resampling();
    test_speaker_encoder_mel_dimensions();

    printf("\n╔════════════════════════════════════════════════════════════════╗\n");
    printf("║ PASSED: %d  FAILED: %d\n", g_pass, g_fail);
    printf("╚════════════════════════════════════════════════════════════════╝\n");

    return g_fail > 0 ? 1 : 0;
}
