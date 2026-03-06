/**
 * test_vap.c — VAP (Voice Activity Projection) model unit tests.
 *
 * Tests creation, feed, KV cache, smoothing, reset, and edge cases.
 */

#include "vap_model.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static int passed = 0, failed = 0;

#define CHECK(cond, msg) do { \
    if (cond) { printf("  [PASS] %s\n", msg); passed++; } \
    else { printf("  [FAIL] %s\n", msg); failed++; } \
} while (0)

#define CHECK_FEQ(a, b, tol, msg) do { \
    if (fabsf((a) - (b)) <= (tol)) { printf("  [PASS] %s\n", msg); passed++; } \
    else { printf("  [FAIL] %s (%.4f vs %.4f)\n", msg, (float)(a), (float)(b)); failed++; } \
} while (0)

static int file_exists(const char *path) {
    FILE *f = fopen(path, "rb");
    if (f) { fclose(f); return 1; }
    return 0;
}

int main(void) {
    printf("═══ VAP Model Tests ═══\n\n");

    /* ── API safety ────────────────────────────────────────────────────── */
    printf("── API Safety ──\n");
    CHECK(vap_create(NULL) == NULL, "create(NULL) returns NULL");
    CHECK(vap_create("nonexistent.vap") == NULL, "create(bad path) returns NULL");
    CHECK(vap_create_config(128, 0, 4, 256) == NULL, "create_config(n_layers=0) returns NULL");
    CHECK(vap_create_config(128, 4, 5, 256) == NULL, "create_config(n_heads not divisible) returns NULL");

    vap_destroy(NULL);
    CHECK(1, "destroy(NULL) is safe");
    vap_reset(NULL);
    CHECK(1, "reset(NULL) is safe");
    vap_set_smoothing(NULL, 0.5f);
    CHECK(1, "set_smoothing(NULL) is safe");

    VAPPrediction p0 = vap_get_smoothed(NULL);
    CHECK(p0.p_user_speaking == 0.0f && p0.p_eou == 0.0f, "get_smoothed(NULL) returns zeros");

    float user_mel[80], system_mel[80];
    memset(user_mel, 0, sizeof(user_mel));
    memset(system_mel, 0, sizeof(system_mel));
    VAPPrediction p1 = vap_feed(NULL, user_mel, system_mel);
    CHECK(p1.p_user_speaking >= 0.0f && p1.p_user_speaking <= 1.0f, "feed(NULL) returns valid range");

    /* ── vap_create_config ─────────────────────────────────────────────── */
    printf("\n── Create Config ──\n");
    VAPModel *vap = vap_create_config(128, 4, 4, 256);
    CHECK(vap != NULL, "create_config(128,4,4,256) succeeds");
    vap_destroy(vap);

    vap = vap_create_config(64, 2, 2, 128);
    CHECK(vap != NULL, "create_config(64,2,2,128) succeeds");
    vap_destroy(vap);

    /* ── vap_feed with synthetic mel ──────────────────────────────────── */
    printf("\n── Feed Synthetic Mel ──\n");
    vap = vap_create_config(128, 4, 4, 256);
    CHECK(vap != NULL, "create for feed test");

    for (int i = 0; i < 80; i++) {
        user_mel[i] = 0.01f * (float)i;
        system_mel[i] = -0.005f * (float)i;
    }

    VAPPrediction pred = vap_feed(vap, user_mel, system_mel);
    CHECK(pred.p_user_speaking >= 0.0f && pred.p_user_speaking <= 1.0f, "p_user_speaking in [0,1]");
    CHECK(pred.p_system_turn >= 0.0f && pred.p_system_turn <= 1.0f, "p_system_turn in [0,1]");
    CHECK(pred.p_backchannel >= 0.0f && pred.p_backchannel <= 1.0f, "p_backchannel in [0,1]");
    CHECK(pred.p_eou >= 0.0f && pred.p_eou <= 1.0f, "p_eou in [0,1]");

    pred = vap_feed(vap, user_mel, NULL);
    CHECK(pred.p_user_speaking >= 0.0f && pred.p_user_speaking <= 1.0f, "feed with NULL system_mel works");

    vap_destroy(vap);

    /* ── KV cache / many frames ─────────────────────────────────────────── */
    printf("\n── KV Cache ──\n");
    vap = vap_create_config(128, 4, 4, 256);
    for (int i = 0; i < 300; i++) {
        for (int j = 0; j < 80; j++) user_mel[j] = 0.1f * (float)(i + j);
        pred = vap_feed(vap, user_mel, system_mel);
    }
    CHECK(pred.p_user_speaking >= 0.0f && pred.p_user_speaking <= 1.0f, "300 frames (KV wrap) produces valid output");
    vap_destroy(vap);

    /* ── Smoothing ─────────────────────────────────────────────────────── */
    printf("\n── Smoothing ──\n");
    vap = vap_create_config(128, 4, 4, 256);
    vap_set_smoothing(vap, 0.0f);
    pred = vap_feed(vap, user_mel, system_mel);
    VAPPrediction smoothed = vap_get_smoothed(vap);
    CHECK_FEQ(pred.p_user_speaking, smoothed.p_user_speaking, 0.01f, "alpha=0: raw ≈ smoothed");

    vap_set_smoothing(vap, 0.9f);
    for (int i = 0; i < 10; i++) vap_feed(vap, user_mel, system_mel);
    smoothed = vap_get_smoothed(vap);
    CHECK(smoothed.p_user_speaking >= 0.0f && smoothed.p_user_speaking <= 1.0f, "alpha=0.9: smoothed valid");
    vap_destroy(vap);

    /* ── Reset ─────────────────────────────────────────────────────────── */
    printf("\n── Reset ──\n");
    vap = vap_create_config(128, 4, 4, 256);
    for (int i = 0; i < 50; i++) vap_feed(vap, user_mel, system_mel);
    vap_reset(vap);
    pred = vap_feed(vap, user_mel, system_mel);
    CHECK(pred.p_user_speaking >= 0.0f && pred.p_user_speaking <= 1.0f, "after reset, feed works");
    vap_destroy(vap);

    /* ── Load from file (optional) ──────────────────────────────────────── */
    printf("\n── Load from File ──\n");
    if (file_exists("models/vap.vap")) {
        vap = vap_create("models/vap.vap");
        CHECK(vap != NULL, "create from models/vap.vap succeeds");
        if (vap) {
            pred = vap_feed(vap, user_mel, system_mel);
            CHECK(pred.p_user_speaking >= 0.0f && pred.p_user_speaking <= 1.0f, "feed with loaded weights works");
            vap_destroy(vap);
        }
    } else {
        printf("  [SKIP] models/vap.vap not found (run train_vap.py --synthetic 100 --export)\n");
    }

    printf("\n═══ VAP Tests: %d passed, %d failed ═══\n", passed, failed);
    return failed ? 1 : 0;
}
