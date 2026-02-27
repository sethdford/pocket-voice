/**
 * test_opus_codec.c — Tests for PocketOpus encoder/decoder.
 *
 * Verifies:
 *   - Lifecycle (create/destroy)
 *   - NULL safety on all API functions
 *   - Encode/decode roundtrip
 *   - Frame size calculation
 *   - Flush behavior
 *   - Various sample rates, bitrates, and application modes
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

typedef struct PocketOpus PocketOpus;

extern PocketOpus *pocket_opus_create(int sample_rate, int channels, int bitrate,
                                       float frame_ms, int application);
extern int pocket_opus_encode(PocketOpus *ctx, const float *pcm, int n_samples,
                               unsigned char *opus_out, int max_out);
extern int pocket_opus_decode(PocketOpus *ctx, const unsigned char *opus_data,
                               int opus_len, float *pcm_out, int max_samples);
extern int pocket_opus_flush(PocketOpus *ctx, unsigned char *opus_out, int max_out);
extern int pocket_opus_frame_size(PocketOpus *ctx);
extern void pocket_opus_destroy(PocketOpus *ctx);

static int passed = 0, failed = 0;
#define CHECK(cond, msg) do { \
    if (cond) { printf("  [PASS] %s\n", msg); passed++; } \
    else { printf("  [FAIL] %s\n", msg); failed++; } \
} while (0)

/* ── Lifecycle tests ─────────────────────────────────────────────── */

static void test_create_destroy_24k(void) {
    printf("\n[Create/Destroy 24kHz]\n");
    PocketOpus *ctx = pocket_opus_create(24000, 1, 24000, 20.0f, 0);
    CHECK(ctx != NULL, "create 24kHz mono VOIP succeeds");
    pocket_opus_destroy(ctx);
    CHECK(1, "destroy does not crash");
}

static void test_create_destroy_48k(void) {
    printf("\n[Create/Destroy 48kHz]\n");
    PocketOpus *ctx = pocket_opus_create(48000, 1, 48000, 20.0f, 1);
    CHECK(ctx != NULL, "create 48kHz mono Audio succeeds");
    pocket_opus_destroy(ctx);
}

static void test_create_restricted_ld(void) {
    printf("\n[Create RestrictedLowDelay]\n");
    PocketOpus *ctx = pocket_opus_create(48000, 1, 32000, 10.0f, 2);
    CHECK(ctx != NULL, "create restricted low-delay succeeds");
    pocket_opus_destroy(ctx);
}

static void test_create_default_app_mode(void) {
    printf("\n[Create Default App Mode]\n");
    PocketOpus *ctx = pocket_opus_create(24000, 1, 24000, 20.0f, 99);
    CHECK(ctx != NULL, "create with invalid app mode defaults to Audio");
    pocket_opus_destroy(ctx);
}

/* ── NULL safety ─────────────────────────────────────────────────── */

static void test_null_safety(void) {
    printf("\n[NULL Safety]\n");
    pocket_opus_destroy(NULL);
    CHECK(1, "destroy(NULL) does not crash");

    int ret = pocket_opus_encode(NULL, NULL, 0, NULL, 0);
    CHECK(ret < 0, "encode(NULL) returns error");

    ret = pocket_opus_decode(NULL, NULL, 0, NULL, 0);
    CHECK(ret < 0, "decode(NULL) returns error");

    ret = pocket_opus_flush(NULL, NULL, 0);
    CHECK(ret == 0, "flush(NULL) returns 0");

    int fs = pocket_opus_frame_size(NULL);
    CHECK(fs == 0, "frame_size(NULL) returns 0");
}

/* ── Frame size ──────────────────────────────────────────────────── */

static void test_frame_size(void) {
    printf("\n[Frame Size]\n");

    PocketOpus *ctx = pocket_opus_create(24000, 1, 24000, 20.0f, 0);
    CHECK(ctx != NULL, "create for frame_size test");
    int fs = pocket_opus_frame_size(ctx);
    CHECK(fs == 480, "24kHz * 20ms = 480 samples");
    pocket_opus_destroy(ctx);

    ctx = pocket_opus_create(48000, 1, 48000, 10.0f, 1);
    CHECK(ctx != NULL, "create 48kHz/10ms for frame_size test");
    fs = pocket_opus_frame_size(ctx);
    CHECK(fs == 480, "48kHz * 10ms = 480 samples");
    pocket_opus_destroy(ctx);
}

/* ── Encode/Decode roundtrip ─────────────────────────────────────── */

static void test_encode_decode_roundtrip(void) {
    printf("\n[Encode/Decode Roundtrip]\n");

    PocketOpus *ctx = pocket_opus_create(24000, 1, 48000, 20.0f, 1);
    CHECK(ctx != NULL, "create for roundtrip test");
    if (!ctx) return;

    int frame_size = pocket_opus_frame_size(ctx);

    /* Generate a 440Hz sine wave, one frame */
    float *pcm_in = (float *)calloc(frame_size, sizeof(float));
    for (int i = 0; i < frame_size; i++) {
        pcm_in[i] = 0.5f * sinf(2.0f * (float)M_PI * 440.0f * i / 24000.0f);
    }

    unsigned char opus_buf[4000];
    int encoded = pocket_opus_encode(ctx, pcm_in, frame_size, opus_buf, sizeof(opus_buf));
    CHECK(encoded > 0, "encode produces bytes");

    float *pcm_out = (float *)calloc(frame_size, sizeof(float));
    int decoded = pocket_opus_decode(ctx, opus_buf, encoded, pcm_out, frame_size);
    CHECK(decoded > 0, "decode produces samples");

    /* Check that decoded signal is correlated with input (lossy codec) */
    float corr = 0, en_in = 0, en_out = 0;
    for (int i = 0; i < decoded && i < frame_size; i++) {
        corr += pcm_in[i] * pcm_out[i];
        en_in += pcm_in[i] * pcm_in[i];
        en_out += pcm_out[i] * pcm_out[i];
    }
    float norm = sqrtf(en_in * en_out);
    float normalized_corr = (norm > 0) ? corr / norm : 0;
    CHECK(normalized_corr > 0.5f, "decoded signal correlates with input (>0.5)");

    free(pcm_in);
    free(pcm_out);
    pocket_opus_destroy(ctx);
}

/* ── Accumulation (partial frames) ────────────────────────────────── */

static void test_partial_frame_accumulation(void) {
    printf("\n[Partial Frame Accumulation]\n");

    PocketOpus *ctx = pocket_opus_create(24000, 1, 24000, 20.0f, 0);
    CHECK(ctx != NULL, "create for accumulation test");
    if (!ctx) return;

    int frame_size = pocket_opus_frame_size(ctx);
    int half = frame_size / 2;

    float *pcm = (float *)calloc(frame_size, sizeof(float));
    for (int i = 0; i < frame_size; i++) {
        pcm[i] = 0.3f * sinf(2.0f * (float)M_PI * 300.0f * i / 24000.0f);
    }

    unsigned char opus_buf[4000];

    /* First half — should not produce output yet */
    int encoded = pocket_opus_encode(ctx, pcm, half, opus_buf, sizeof(opus_buf));
    CHECK(encoded == 0, "half frame produces no output");

    /* Second half — completes the frame */
    encoded = pocket_opus_encode(ctx, pcm + half, half, opus_buf, sizeof(opus_buf));
    CHECK(encoded > 0, "completing frame produces encoded output");

    free(pcm);
    pocket_opus_destroy(ctx);
}

/* ── Flush ───────────────────────────────────────────────────────── */

static void test_flush_with_pending(void) {
    printf("\n[Flush With Pending Samples]\n");

    PocketOpus *ctx = pocket_opus_create(24000, 1, 24000, 20.0f, 0);
    CHECK(ctx != NULL, "create for flush test");
    if (!ctx) return;

    int frame_size = pocket_opus_frame_size(ctx);
    int quarter = frame_size / 4;

    float *pcm = (float *)calloc(quarter, sizeof(float));
    for (int i = 0; i < quarter; i++) pcm[i] = 0.1f;

    unsigned char opus_buf[4000];

    /* Feed partial data */
    int encoded = pocket_opus_encode(ctx, pcm, quarter, opus_buf, sizeof(opus_buf));
    CHECK(encoded == 0, "partial data not yet encoded");

    /* Flush pads with silence and encodes */
    int flushed = pocket_opus_flush(ctx, opus_buf, sizeof(opus_buf));
    CHECK(flushed > 0, "flush produces encoded output");

    /* Second flush with no pending data */
    int flushed2 = pocket_opus_flush(ctx, opus_buf, sizeof(opus_buf));
    CHECK(flushed2 == 0, "flush with no pending data returns 0");

    free(pcm);
    pocket_opus_destroy(ctx);
}

/* ── Multi-frame encode ──────────────────────────────────────────── */

static void test_multi_frame_encode(void) {
    printf("\n[Multi-Frame Encode]\n");

    PocketOpus *ctx = pocket_opus_create(24000, 1, 32000, 20.0f, 1);
    CHECK(ctx != NULL, "create for multi-frame test");
    if (!ctx) return;

    int frame_size = pocket_opus_frame_size(ctx);
    int total = frame_size * 3;  /* 3 full frames */

    float *pcm = (float *)calloc(total, sizeof(float));
    for (int i = 0; i < total; i++) {
        pcm[i] = 0.4f * sinf(2.0f * (float)M_PI * 220.0f * i / 24000.0f);
    }

    unsigned char opus_buf[12000];
    int encoded = pocket_opus_encode(ctx, pcm, total, opus_buf, sizeof(opus_buf));
    CHECK(encoded > 0, "encoding 3 frames at once produces output");

    free(pcm);
    pocket_opus_destroy(ctx);
}

/* ── PLC (Packet Loss Concealment) ────────────────────────────────── */

static void test_plc_decode(void) {
    printf("\n[PLC Decode]\n");

    PocketOpus *ctx = pocket_opus_create(24000, 1, 24000, 20.0f, 0);
    CHECK(ctx != NULL, "create for PLC test");
    if (!ctx) return;

    int frame_size = pocket_opus_frame_size(ctx);

    /* First encode a real frame so decoder has state */
    float *pcm_in = (float *)calloc(frame_size, sizeof(float));
    for (int i = 0; i < frame_size; i++)
        pcm_in[i] = 0.5f * sinf(2.0f * (float)M_PI * 440.0f * i / 24000.0f);

    unsigned char opus_buf[4000];
    int encoded = pocket_opus_encode(ctx, pcm_in, frame_size, opus_buf, sizeof(opus_buf));
    CHECK(encoded > 0, "encode real frame for PLC setup");

    float *pcm_out = (float *)calloc(frame_size, sizeof(float));
    int decoded = pocket_opus_decode(ctx, opus_buf, encoded, pcm_out, frame_size);
    CHECK(decoded > 0, "decode real frame succeeds");

    /* PLC: pass NULL data, 0 length */
    int plc = pocket_opus_decode(ctx, NULL, 0, pcm_out, frame_size);
    CHECK(plc > 0, "PLC decode produces samples");

    free(pcm_in);
    free(pcm_out);
    pocket_opus_destroy(ctx);
}

/* ── Main ────────────────────────────────────────────────────────── */

int main(void) {
    printf("=== Opus Codec Tests ===\n");

    test_create_destroy_24k();
    test_create_destroy_48k();
    test_create_restricted_ld();
    test_create_default_app_mode();
    test_null_safety();
    test_frame_size();
    test_encode_decode_roundtrip();
    test_partial_frame_accumulation();
    test_flush_with_pending();
    test_multi_frame_encode();
    test_plc_decode();

    printf("\n=== Results: %d passed, %d failed ===\n", passed, failed);
    return failed;
}
