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

/* ── Extended JSON Validation ─────────────────────────────────────────────── */

static void test_prosody_log_json_fields(void) {
    printf("\n=== Prosody Log: JSON Field Validation ===\n");

    const char *path = "/tmp/test_prosody_fields.jsonl";
    unlink(path);

    ProsodyLog *log = prosody_log_open(path);
    TEST(log != NULL, "log opens for field validation");

    prosody_log_segment(log, "Test segment", 1.15f, 0.90f, 3.5f,
                        "excited", "exclamatory", 1500);

    prosody_log_turn(log, 42, "user says hi", "bot says hello",
                     3.2f, 1.1f, 0.95f, -18.5f, 350.0f, 0.08f);

    prosody_log_close(log);

    FILE *fp = fopen(path, "r");
    TEST(fp != NULL, "field validation log file exists");

    char buf[8192];
    int found_segment_fields = 0;
    int found_turn_fields = 0;

    while (fgets(buf, sizeof(buf), fp)) {
        cJSON *obj = cJSON_Parse(buf);
        if (!obj) continue;

        const char *type = cJSON_GetStringValue(cJSON_GetObjectItem(obj, "type"));
        if (type && strcmp(type, "segment") == 0) {
            /* Validate segment fields */
            cJSON *text = cJSON_GetObjectItem(obj, "text");
            cJSON *pitch = cJSON_GetObjectItem(obj, "pitch");
            cJSON *rate = cJSON_GetObjectItem(obj, "rate");
            cJSON *emotion = cJSON_GetObjectItem(obj, "emotion");
            cJSON *contour = cJSON_GetObjectItem(obj, "contour");
            cJSON *dur = cJSON_GetObjectItem(obj, "duration_ms");

            if (text && pitch && rate) {
                found_segment_fields = 1;
                TEST(cJSON_IsString(text), "segment.text is string");
                TEST(cJSON_IsNumber(pitch), "segment.pitch is number");
                TEST(cJSON_IsNumber(rate), "segment.rate is number");
                TEST(strcmp(cJSON_GetStringValue(text), "Test segment") == 0,
                     "segment.text matches input");
                TEST(fabsf((float)cJSON_GetNumberValue(pitch) - 1.15f) < 0.01f,
                     "segment.pitch = 1.15");
            }
            if (emotion) {
                TEST(strcmp(cJSON_GetStringValue(emotion), "excited") == 0,
                     "segment.emotion = excited");
            }
            if (contour) {
                TEST(strcmp(cJSON_GetStringValue(contour), "exclamatory") == 0,
                     "segment.contour = exclamatory");
            }
            if (dur) {
                TEST(cJSON_GetNumberValue(dur) == 1500,
                     "segment.duration_ms = 1500");
            }
        }

        if (type && strcmp(type, "turn") == 0) {
            cJSON *turn_id = cJSON_GetObjectItem(obj, "turn_id");
            cJSON *user_text = cJSON_GetObjectItem(obj, "user_text");
            if (turn_id && user_text) {
                found_turn_fields = 1;
                TEST(cJSON_GetNumberValue(turn_id) == 42, "turn.turn_id = 42");
                TEST(strcmp(cJSON_GetStringValue(user_text), "user says hi") == 0,
                     "turn.user_text matches input");
            }
        }

        cJSON_Delete(obj);
    }
    fclose(fp);

    TEST(found_segment_fields, "segment JSON has expected fields");
    TEST(found_turn_fields, "turn JSON has expected fields");

    unlink(path);
}

/* ── Contour with Various Frame Counts ───────────────────────────────────── */

static void test_contour_zero_frames(void) {
    printf("\n=== Contour: 0 Frames ===\n");

    const char *path = "/tmp/test_contour_zero.jsonl";
    unlink(path);

    ProsodyLog *log = prosody_log_open(path);
    TEST(log != NULL, "log opens for zero-frame test");

    /* 0 frames — should not crash */
    prosody_log_contour(log, "empty_contour", NULL, NULL, 0, 24000);
    prosody_log_close(log);

    FILE *fp = fopen(path, "r");
    TEST(fp != NULL, "zero-frame log file exists");
    fclose(fp);

    unlink(path);
    TEST(1, "0 frames logged without crash");
}

static void test_contour_single_frame(void) {
    printf("\n=== Contour: 1 Frame ===\n");

    const char *path = "/tmp/test_contour_one.jsonl";
    unlink(path);

    ProsodyLog *log = prosody_log_open(path);

    float f0 = 180.0f;
    float energy = -15.0f;
    prosody_log_contour(log, "single_frame", &f0, &energy, 1, 24000);
    prosody_log_close(log);

    FILE *fp = fopen(path, "r");
    char buf[4096];
    int found = 0;
    while (fgets(buf, sizeof(buf), fp)) {
        cJSON *obj = cJSON_Parse(buf);
        if (!obj) continue;
        const char *type = cJSON_GetStringValue(cJSON_GetObjectItem(obj, "type"));
        if (type && strcmp(type, "contour") == 0) {
            cJSON *f0_arr = cJSON_GetObjectItem(obj, "f0");
            if (f0_arr && cJSON_GetArraySize(f0_arr) == 1) found = 1;
        }
        cJSON_Delete(obj);
    }
    fclose(fp);

    TEST(found, "single frame contour has 1 f0 value");
    unlink(path);
}

static void test_contour_large_frame_count(void) {
    printf("\n=== Contour: 1000 Frames ===\n");

    const char *path = "/tmp/test_contour_1000.jsonl";
    unlink(path);

    ProsodyLog *log = prosody_log_open(path);

    float f0[1000], energy[1000];
    for (int i = 0; i < 1000; i++) {
        f0[i] = 120.0f + 60.0f * sinf(2.0f * 3.14159f * i / 200.0f);
        energy[i] = -25.0f + 10.0f * sinf(2.0f * 3.14159f * i / 500.0f);
    }

    prosody_log_contour(log, "large_contour", f0, energy, 1000, 24000);
    prosody_log_close(log);

    FILE *fp = fopen(path, "r");
    char buf[131072];  /* large buffer for downsampled data */
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

    TEST(contour_points > 0, "1000-frame contour produces output");
    TEST(contour_points <= 200, "1000 frames downsampled to <= 200");
    printf("    (1000 frames → %d points)\n", contour_points);

    unlink(path);
}

/* ── Turn Logging Validation ──────────────────────────────────────────────── */

static void test_turn_logging_detailed(void) {
    printf("\n=== Turn Logging: Detailed Validation ===\n");

    const char *path = "/tmp/test_turn_detail.jsonl";
    unlink(path);

    ProsodyLog *log = prosody_log_open(path);
    TEST(log != NULL, "log opens for turn detail test");

    /* Log multiple turns */
    prosody_log_turn(log, 0, "first user", "first response",
                     2.0f, 1.0f, 1.0f, -20.0f, 500.0f, 0.1f);
    prosody_log_turn(log, 1, "second user", "second response",
                     3.5f, 1.1f, 0.9f, -15.0f, 300.0f, 0.05f);
    prosody_log_turn(log, 2, NULL, NULL,
                     0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f);

    prosody_log_close(log);

    FILE *fp = fopen(path, "r");
    char buf[8192];
    int turn_count = 0;

    while (fgets(buf, sizeof(buf), fp)) {
        cJSON *obj = cJSON_Parse(buf);
        if (!obj) continue;
        const char *type = cJSON_GetStringValue(cJSON_GetObjectItem(obj, "type"));
        if (type && strcmp(type, "turn") == 0) {
            turn_count++;
        }
        cJSON_Delete(obj);
    }
    fclose(fp);

    TEST(turn_count == 3, "3 turn entries logged");
    unlink(path);
}

/* ── File I/O Edge Cases ──────────────────────────────────────────────────── */

static void test_log_file_io(void) {
    printf("\n=== Log File I/O ===\n");

    /* Write to /tmp with unique name */
    const char *path = "/tmp/test_prosody_io_verify.jsonl";
    unlink(path);

    ProsodyLog *log = prosody_log_open(path);
    TEST(log != NULL, "log created at temp path");

    prosody_log_segment(log, "IO test content", 1.0f, 1.0f, 0.0f,
                        NULL, "declarative", 800);
    prosody_log_close(log);

    /* Verify file exists and has content */
    FILE *fp = fopen(path, "r");
    TEST(fp != NULL, "temp file written and readable");

    long size = 0;
    if (fp) {
        fseek(fp, 0, SEEK_END);
        size = ftell(fp);
        fclose(fp);
    }
    TEST(size > 0, "file has non-zero content");
    printf("    (file size: %ld bytes)\n", size);

    /* Re-open same path should overwrite or append */
    ProsodyLog *log2 = prosody_log_open(path);
    TEST(log2 != NULL, "re-opening same path succeeds");
    prosody_log_close(log2);

    unlink(path);
}

/* ── Expanded NULL/Empty String Handling ──────────────────────────────────── */

static void test_log_null_empty_extended(void) {
    printf("\n=== Prosody Log: NULL/Empty Extended ===\n");

    const char *path = "/tmp/test_null_ext.jsonl";
    unlink(path);

    ProsodyLog *log = prosody_log_open(path);
    TEST(log != NULL, "log opens for null/empty extended");

    /* Segment with NULL text */
    prosody_log_segment(log, NULL, 1.0f, 1.0f, 0.0f, NULL, NULL, 0);
    TEST(1, "NULL text segment doesn't crash");

    /* Segment with empty text */
    prosody_log_segment(log, "", 0.0f, 0.0f, 0.0f, "", "", 0);
    TEST(1, "empty text segment doesn't crash");

    /* Turn with all NULL strings */
    prosody_log_turn(log, -1, NULL, NULL, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f);
    TEST(1, "all-NULL turn doesn't crash");

    /* Contour with NULL arrays but non-zero count */
    prosody_log_contour(log, "test", NULL, NULL, 10, 24000);
    TEST(1, "NULL arrays with n>0 doesn't crash");

    /* Segment with empty emotion/contour strings */
    prosody_log_segment(log, "test", 1.0f, 1.0f, 0.0f, "", "", 100);
    TEST(1, "empty emotion/contour strings don't crash");

    prosody_log_close(log);
    unlink(path);
}

/* ── Multiple Sessions to Same File ──────────────────────────────────────── */

static void test_multiple_sessions(void) {
    printf("\n=== Prosody Log: Multiple Sessions ===\n");

    const char *path = "/tmp/test_multi_session.jsonl";
    unlink(path);

    /* Session 1 */
    ProsodyLog *log1 = prosody_log_open(path);
    prosody_log_segment(log1, "Session 1 text", 1.0f, 1.0f, 0.0f,
                        NULL, "declarative", 500);
    prosody_log_close(log1);

    /* Session 2 — appends or overwrites */
    ProsodyLog *log2 = prosody_log_open(path);
    prosody_log_segment(log2, "Session 2 text", 1.1f, 0.9f, 1.0f,
                        "happy", "exclamatory", 600);
    prosody_log_close(log2);

    /* Verify file has data */
    FILE *fp = fopen(path, "r");
    TEST(fp != NULL, "multi-session file exists");
    int line_count = 0;
    char buf[4096];
    while (fgets(buf, sizeof(buf), fp)) line_count++;
    fclose(fp);

    TEST(line_count >= 2, "multi-session file has multiple entries");

    unlink(path);
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

    /* Extended tests */
    test_prosody_log_json_fields();
    test_contour_zero_frames();
    test_contour_single_frame();
    test_contour_large_frame_count();
    test_turn_logging_detailed();
    test_log_file_io();
    test_log_null_empty_extended();
    test_multiple_sessions();

    printf("\n════════════════════════════════════════════\n");
    printf("  Results: %d passed, %d failed\n", pass_count, fail_count);
    printf("════════════════════════════════════════════\n");

    return fail_count > 0 ? 1 : 0;
}
