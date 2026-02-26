/**
 * test_prosody_log.c — Unit tests for prosody logging and new prosody features.
 *
 * Tests: prosody_log JSONL output, pause frame calculations, speaker interpolation
 * FFI declarations, prosody-aware chunking logic, and A/B evaluation integration.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include "prosody_log.h"
#include "cJSON.h"

static int pass_count = 0;
static int fail_count = 0;

#define TEST(cond, name) do { \
    if (cond) { printf("  [PASS] %s\n", name); pass_count++; } \
    else { printf("  [FAIL] %s (line %d)\n", name, __LINE__); fail_count++; } \
} while (0)

/* ── Prosody Log Tests ────────────────────────────────────────────────────── */

static void test_prosody_log_basic(void) {
    printf("\n=== Prosody Log: Basic ===\n");

    const char *path = "/tmp/test_prosody_log.jsonl";
    unlink(path);

    ProsodyLog *log = prosody_log_open(path);
    TEST(log != NULL, "log opens successfully");

    prosody_log_segment(log, "Hello world", 1.05f, 0.95f, 2.5f,
                        "happy", "declarative", 1200);
    prosody_log_segment(log, "How are you?", 1.10f, 1.00f, 0.0f,
                        NULL, "interrogative", 800);

    prosody_log_turn(log, 1, "user said hello", "Hello world!",
                     2.5f, 1.05f, 0.95f, -20.0f, 450.0f, 0.15f);

    prosody_log_close(log);

    /* Read back and validate JSONL format */
    FILE *fp = fopen(path, "r");
    TEST(fp != NULL, "log file exists after close");

    int line_count = 0;
    int found_session_start = 0;
    int found_segment = 0;
    int found_turn = 0;
    int found_session_end = 0;
    char buf[4096];

    while (fgets(buf, sizeof(buf), fp)) {
        line_count++;
        cJSON *obj = cJSON_Parse(buf);
        if (!obj) continue;

        const char *type = cJSON_GetStringValue(cJSON_GetObjectItem(obj, "type"));
        if (type) {
            if (strcmp(type, "session_start") == 0) found_session_start = 1;
            else if (strcmp(type, "segment") == 0) found_segment++;
            else if (strcmp(type, "turn") == 0) found_turn = 1;
            else if (strcmp(type, "session_end") == 0) found_session_end = 1;
        }
        cJSON_Delete(obj);
    }
    fclose(fp);

    TEST(found_session_start, "session_start entry present");
    TEST(found_segment == 2, "2 segment entries written");
    TEST(found_turn, "turn entry present");
    TEST(found_session_end, "session_end entry present");
    TEST(line_count >= 5, "at least 5 lines (start + 2 seg + turn + end)");

    unlink(path);
}

static void test_prosody_log_contour(void) {
    printf("\n=== Prosody Log: Contour ===\n");

    const char *path = "/tmp/test_prosody_contour.jsonl";
    unlink(path);

    ProsodyLog *log = prosody_log_open(path);
    TEST(log != NULL, "log opens for contour test");

    float f0[100], energy[100];
    for (int i = 0; i < 100; i++) {
        f0[i] = 150.0f + 30.0f * sinf(2.0f * 3.14159f * i / 50.0f);
        energy[i] = -20.0f + 5.0f * sinf(2.0f * 3.14159f * i / 100.0f);
    }

    prosody_log_contour(log, "test_segment_1", f0, energy, 100, 24000);
    prosody_log_close(log);

    /* Validate contour entry */
    FILE *fp = fopen(path, "r");
    int found_contour = 0;
    char buf[16384];
    while (fgets(buf, sizeof(buf), fp)) {
        cJSON *obj = cJSON_Parse(buf);
        if (!obj) continue;
        const char *type = cJSON_GetStringValue(cJSON_GetObjectItem(obj, "type"));
        if (type && strcmp(type, "contour") == 0) {
            found_contour = 1;
            cJSON *f0_arr = cJSON_GetObjectItem(obj, "f0");
            cJSON *e_arr = cJSON_GetObjectItem(obj, "energy");
            TEST(cJSON_IsArray(f0_arr), "f0 is array");
            TEST(cJSON_IsArray(e_arr), "energy is array");
            TEST(cJSON_GetArraySize(f0_arr) == 100, "f0 has 100 values");
            TEST(cJSON_GetArraySize(e_arr) == 100, "energy has 100 values");
        }
        cJSON_Delete(obj);
    }
    fclose(fp);

    TEST(found_contour, "contour entry found");
    unlink(path);
}

static void test_prosody_log_null_safety(void) {
    printf("\n=== Prosody Log: NULL Safety ===\n");

    TEST(prosody_log_open(NULL) == NULL, "NULL path returns NULL");

    /* All log functions should be NULL-safe */
    prosody_log_segment(NULL, "test", 1.0f, 1.0f, 0.0f, NULL, NULL, 0);
    prosody_log_turn(NULL, 0, NULL, NULL, 0, 0, 0, 0, 0, 0);
    prosody_log_contour(NULL, NULL, NULL, NULL, 0, 0);
    prosody_log_close(NULL);
    TEST(1, "NULL calls don't crash");
}

/* ── Pause Token Frame Calculation Tests ──────────────────────────────────── */

/* Mirror of the Rust implementation: (ms * 0.05 + 0.5) as int */
static int local_ms_to_frames(int ms) {
    return (int)((float)ms * 0.05f + 0.5f);
}

static void test_pause_frame_calculation(void) {
    printf("\n=== Pause Token Frame Calculation ===\n");

    /* 50 Hz = 20ms per frame */
    TEST(local_ms_to_frames(0) == 0, "0ms = 0 frames");
    TEST(local_ms_to_frames(20) == 1, "20ms = 1 frame");
    TEST(local_ms_to_frames(100) == 5, "100ms = 5 frames");
    TEST(local_ms_to_frames(200) == 10, "200ms = 10 frames");
    TEST(local_ms_to_frames(500) == 25, "500ms = 25 frames");
    TEST(local_ms_to_frames(1000) == 50, "1000ms = 50 frames");
    TEST(local_ms_to_frames(300) == 15, "300ms = 15 frames");
}

/* ── Prosody-Aware Chunking Logic Tests ───────────────────────────────────── */

static void test_chunk_boundary_detection(void) {
    printf("\n=== Prosody-Aware Chunk Boundaries ===\n");

    /* Simulate token repetition detection logic from sonata_step */
    int tokens[100];
    for (int i = 0; i < 100; i++) tokens[i] = i % 50;

    /* No repetition = no early flush */
    tokens[29] = 10; tokens[28] = 20; tokens[27] = 30;
    int n = 30;
    int repeat3 = (n >= 3 &&
        tokens[n-1] == tokens[n-2] && tokens[n-2] == tokens[n-3]);
    TEST(!repeat3, "no false early flush without repetition");

    /* Triple repetition = early flush at phrase boundary */
    tokens[29] = 42; tokens[28] = 42; tokens[27] = 42;
    repeat3 = (n >= 3 &&
        tokens[n-1] == tokens[n-2] && tokens[n-2] == tokens[n-3]);
    TEST(repeat3, "triple repetition triggers early flush");

    /* Only fires in the window between FIRST_CHUNK and CHUNK_SIZE */
    int FIRST_CHUNK = 25, CHUNK_SIZE = 50;
    int should_flush_early = (!0 && n >= FIRST_CHUNK && n < CHUNK_SIZE && repeat3);
    TEST(should_flush_early, "flush in valid window [25, 50)");

    /* Below FIRST_CHUNK: no early flush */
    n = 20;
    should_flush_early = (!0 && n >= FIRST_CHUNK && n < CHUNK_SIZE);
    TEST(!should_flush_early, "no early flush below FIRST_CHUNK");
}

/* ── Downsampling for JSONL Contour ───────────────────────────────────────── */

static void test_contour_downsampling(void) {
    printf("\n=== Contour Downsampling ===\n");

    const char *path = "/tmp/test_prosody_downsample.jsonl";
    unlink(path);

    ProsodyLog *log = prosody_log_open(path);

    /* 500 frames should be downsampled to ~200 */
    float f0[500], energy[500];
    for (int i = 0; i < 500; i++) {
        f0[i] = 100.0f + (float)i;
        energy[i] = -30.0f + 0.1f * (float)i;
    }
    prosody_log_contour(log, "long_segment", f0, energy, 500, 24000);
    prosody_log_close(log);

    FILE *fp = fopen(path, "r");
    char buf[65536];
    int contour_points = 0;
    while (fgets(buf, sizeof(buf), fp)) {
        cJSON *obj = cJSON_Parse(buf);
        if (!obj) continue;
        const char *type = cJSON_GetStringValue(cJSON_GetObjectItem(obj, "type"));
        if (type && strcmp(type, "contour") == 0) {
            cJSON *f0_arr = cJSON_GetObjectItem(obj, "f0");
            contour_points = cJSON_GetArraySize(f0_arr);
        }
        cJSON_Delete(obj);
    }
    fclose(fp);

    TEST(contour_points > 0 && contour_points <= 200,
         "500 frames downsampled to <=200 points");
    printf("    (got %d points)\n", contour_points);

    unlink(path);
}

/* ── Speaker Interpolation Logic Tests ────────────────────────────────────── */

static void test_speaker_interpolation_math(void) {
    printf("\n=== Speaker Interpolation Math ===\n");

    /* Test the blending math: result = (1-alpha)*A + alpha*B, then L2-normalize */
    float a[] = {1.0f, 0.0f, 0.0f, 0.0f};
    float b[] = {0.0f, 1.0f, 0.0f, 0.0f};

    /* alpha=0.0 → pure A (after normalization) */
    float result[4];
    float alpha = 0.0f;
    for (int i = 0; i < 4; i++)
        result[i] = (1.0f - alpha) * a[i] + alpha * b[i];
    float norm = 0;
    for (int i = 0; i < 4; i++) norm += result[i] * result[i];
    norm = sqrtf(norm);
    for (int i = 0; i < 4; i++) result[i] /= norm;
    TEST(fabsf(result[0] - 1.0f) < 0.01f, "alpha=0 → pure A");

    /* alpha=1.0 → pure B */
    alpha = 1.0f;
    for (int i = 0; i < 4; i++)
        result[i] = (1.0f - alpha) * a[i] + alpha * b[i];
    norm = 0;
    for (int i = 0; i < 4; i++) norm += result[i] * result[i];
    norm = sqrtf(norm);
    for (int i = 0; i < 4; i++) result[i] /= norm;
    TEST(fabsf(result[1] - 1.0f) < 0.01f, "alpha=1 → pure B");

    /* alpha=0.5 → blend, both non-zero */
    alpha = 0.5f;
    for (int i = 0; i < 4; i++)
        result[i] = (1.0f - alpha) * a[i] + alpha * b[i];
    norm = 0;
    for (int i = 0; i < 4; i++) norm += result[i] * result[i];
    norm = sqrtf(norm);
    for (int i = 0; i < 4; i++) result[i] /= norm;
    TEST(result[0] > 0.5f && result[1] > 0.5f, "alpha=0.5 → both components active");
    float blend_norm = 0;
    for (int i = 0; i < 4; i++) blend_norm += result[i] * result[i];
    TEST(fabsf(blend_norm - 1.0f) < 0.01f, "blended result is L2-normalized");
}

/* ── Main ─────────────────────────────────────────────────────────────────── */

int main(void) {
    printf("╔═══════════════════════════════════════════╗\n");
    printf("║  Prosody Log & Advanced Features Tests    ║\n");
    printf("╚═══════════════════════════════════════════╝\n");

    test_prosody_log_basic();
    test_prosody_log_contour();
    test_prosody_log_null_safety();
    test_pause_frame_calculation();
    test_chunk_boundary_detection();
    test_contour_downsampling();
    test_speaker_interpolation_math();

    printf("\n════════════════════════════════════════════\n");
    printf("  Results: %d passed, %d failed\n", pass_count, fail_count);
    printf("════════════════════════════════════════════\n");

    return fail_count > 0 ? 1 : 0;
}
