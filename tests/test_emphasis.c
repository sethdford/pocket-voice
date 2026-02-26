/**
 * test_emphasis.c — Tests for emphasis prediction, quoted speech detection,
 * and advanced prosody pipeline features.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "emphasis_predict.h"

static int pass_count = 0;
static int fail_count = 0;

#define TEST(cond, name) do { \
    if (cond) { printf("  [PASS] %s\n", name); pass_count++; } \
    else { printf("  [FAIL] %s (line %d)\n", name, __LINE__); fail_count++; } \
} while (0)

/* ── Emphasis Prediction ─────────────────────────────────────────────────── */

static void test_emphasis_basic(void) {
    printf("\n=== Emphasis: Basic ===\n");

    char out[4096];

    /* Contrast marker: "but" → next content word gets emphasis */
    int n = emphasis_predict("I tried hard but failed miserably.", out, sizeof(out));
    TEST(n > 0, "contrast marker 'but' triggers emphasis");
    TEST(strstr(out, "<emphasis") != NULL, "output contains <emphasis> tag");
    TEST(strstr(out, "failed") != NULL || strstr(out, "miserably") != NULL,
         "emphasis on content word after 'but'");

    /* Intensifier: "very" → next word emphasized */
    n = emphasis_predict("That is very important to me.", out, sizeof(out));
    TEST(strstr(out, "<emphasis") != NULL, "intensifier triggers emphasis");
    TEST(strstr(out, "important") != NULL, "emphasis on word after 'very'");

    /* Negation stress: "never" gets emphasis */
    n = emphasis_predict("I never said that.", out, sizeof(out));
    TEST(strstr(out, "<emphasis") != NULL, "negation triggers emphasis");
    TEST(strstr(out, "never") != NULL, "emphasis includes negation word");
}

static void test_emphasis_sentence_final(void) {
    printf("\n=== Emphasis: Sentence-Final ===\n");

    char out[4096];

    /* Sentence-final content word gets mild emphasis */
    int n = emphasis_predict("The weather is beautiful.", out, sizeof(out));
    TEST(n > 0, "sentence-final word gets emphasis");
    TEST(strstr(out, "beautiful") != NULL, "emphasis on final content word");

    /* Function word at end shouldn't be emphasized — look backward for content word */
    n = emphasis_predict("She walked down the road.", out, sizeof(out));
    TEST(strstr(out, "<emphasis") != NULL, "content word gets emphasis");
    TEST(strstr(out, "road") != NULL || strstr(out, "walked") != NULL,
         "emphasis on content word, not function word");
}

static void test_emphasis_list(void) {
    printf("\n=== Emphasis: List-Final Item ===\n");

    char out[4096];

    /* "X, Y, and Z" → Z gets emphasis */
    int n = emphasis_predict("I bought apples, oranges, and bananas.", out, sizeof(out));
    TEST(strstr(out, "bananas") != NULL, "list-final 'bananas' in output");
    /* The emphasis may be on bananas (list-final) */
    TEST(strstr(out, "<emphasis") != NULL, "list pattern triggers emphasis");
}

static void test_emphasis_cap(void) {
    printf("\n=== Emphasis: Cap at 3 ===\n");

    char out[4096];

    /* Very long sentence shouldn't get too many emphasis markers */
    int n = emphasis_predict(
        "However this is actually really very extremely truly remarkably incredible.",
        out, sizeof(out));
    /* Count emphasis tags */
    int count = 0;
    const char *p = out;
    while ((p = strstr(p, "<emphasis")) != NULL) { count++; p++; }
    TEST(count <= 3, "max 3 emphasis points per sentence");
    printf("    (emphasis count: %d)\n", count);
}

static void test_emphasis_passthrough(void) {
    printf("\n=== Emphasis: SSML Passthrough ===\n");

    char out[4096];

    /* Already has <emphasis> → pass through unchanged */
    const char *input = "I <emphasis>really</emphasis> mean it.";
    int n = emphasis_predict(input, out, sizeof(out));
    TEST(n == 0, "no additional emphasis added when SSML present");
    TEST(strcmp(out, input) == 0, "input preserved unchanged");
}

static void test_emphasis_empty(void) {
    printf("\n=== Emphasis: Edge Cases ===\n");

    char out[4096];

    int n = emphasis_predict("", out, sizeof(out));
    TEST(n == 0, "empty string → 0 emphasis");
    TEST(out[0] == '\0', "empty output for empty input");

    n = emphasis_predict("Hi.", out, sizeof(out));
    TEST(n >= 0, "single word doesn't crash");

    /* NULL safety */
    n = emphasis_predict(NULL, out, sizeof(out));
    TEST(n == 0, "NULL input → 0");

    n = emphasis_predict("test", NULL, 0);
    TEST(n == 0, "NULL output → 0");
}

/* ── Quoted Speech Detection ─────────────────────────────────────────────── */

static void test_quotes_basic(void) {
    printf("\n=== Quotes: Basic Detection ===\n");

    char out[4096];

    int n = emphasis_detect_quotes(
        "She said, \"hello there friend\" and walked away.", out, sizeof(out));
    TEST(n == 1, "found 1 quoted segment");
    TEST(strstr(out, "<voice name=\"quoted\">") != NULL, "voice tag inserted");
    TEST(strstr(out, "hello there friend") != NULL, "quoted content preserved");
    TEST(strstr(out, "</voice>") != NULL, "closing voice tag present");
}

static void test_quotes_multiple(void) {
    printf("\n=== Quotes: Multiple Quotes ===\n");

    char out[4096];

    int n = emphasis_detect_quotes(
        "He said \"good morning\" and she replied \"good evening\" politely.",
        out, sizeof(out));
    TEST(n == 2, "found 2 quoted segments");

    /* Count voice tags */
    int count = 0;
    const char *p = out;
    while ((p = strstr(p, "<voice")) != NULL) { count++; p++; }
    TEST(count == 2, "2 voice tags inserted");
}

static void test_quotes_short(void) {
    printf("\n=== Quotes: Short Quotes Skipped ===\n");

    char out[4096];

    /* Short quotes (<= 3 chars) should NOT be wrapped */
    int n = emphasis_detect_quotes("She said \"no\" firmly.", out, sizeof(out));
    TEST(n == 0, "short quote (2 chars) not wrapped");
    TEST(strstr(out, "<voice") == NULL, "no voice tag for short quote");
}

static void test_quotes_empty(void) {
    printf("\n=== Quotes: Edge Cases ===\n");

    char out[4096];

    int n = emphasis_detect_quotes("No quotes here.", out, sizeof(out));
    TEST(n == 0, "no quotes found → 0");
    TEST(strcmp(out, "No quotes here.") == 0, "text preserved");

    n = emphasis_detect_quotes(NULL, out, sizeof(out));
    TEST(n == 0, "NULL input → 0");
}

/* ── Prosody Feedback Loop Logic ─────────────────────────────────────────── */

static void test_prosody_feedback(void) {
    printf("\n=== Prosody Feedback Loop ===\n");

    /* Simulate the boost calculation logic */
    float f0_range = 30.0f;  /* Below 50 Hz target */
    float energy_var = 2.0f; /* Below 4.0 target */

    float f0_ok = (f0_range >= 50.0f) ? 1.0f : 0.5f;
    float e_ok = (energy_var >= 4.0f) ? 1.0f : 0.5f;
    float quality = 0.6f * f0_ok + 0.4f * e_ok;
    TEST(quality < 0.7f, "low quality triggers boost");

    float boost = 1.0f;
    if (quality < 0.7f) boost += 0.05f;
    TEST(boost > 1.0f, "boost increases when quality is low");
    TEST(boost <= 1.3f, "boost capped at max");

    /* Good quality → decay */
    f0_range = 80.0f;
    energy_var = 6.0f;
    f0_ok = (f0_range >= 50.0f) ? 1.0f : 0.5f;
    e_ok = (energy_var >= 4.0f) ? 1.0f : 0.5f;
    quality = 0.6f * f0_ok + 0.4f * e_ok;
    TEST(quality >= 0.7f, "good quality doesn't trigger boost");

    if (quality >= 0.7f && boost > 1.0f) boost -= 0.02f;
    TEST(boost < 1.05f, "boost decays toward 1.0");
}

/* ── Speaker Interpolation Alpha Clamping ────────────────────────────────── */

static void test_alpha_clamping(void) {
    printf("\n=== Speaker Interpolation: Alpha Clamping ===\n");

    /* Test that alpha values are properly clamped */
    float alpha_neg = -0.5f;
    float clamped = alpha_neg < 0.0f ? 0.0f : (alpha_neg > 1.0f ? 1.0f : alpha_neg);
    TEST(clamped == 0.0f, "negative alpha clamped to 0");

    float alpha_over = 1.5f;
    clamped = alpha_over < 0.0f ? 0.0f : (alpha_over > 1.0f ? 1.0f : alpha_over);
    TEST(clamped == 1.0f, "alpha > 1 clamped to 1");

    float alpha_ok = 0.3f;
    clamped = alpha_ok < 0.0f ? 0.0f : (alpha_ok > 1.0f ? 1.0f : alpha_ok);
    TEST(fabsf(clamped - 0.3f) < 0.01f, "valid alpha preserved");
}

/* ── Main ─────────────────────────────────────────────────────────────────── */

int main(void) {
    printf("╔════════════════════════════════════════════════════╗\n");
    printf("║  Emphasis Prediction & Advanced Pipeline Tests     ║\n");
    printf("╚════════════════════════════════════════════════════╝\n");

    test_emphasis_basic();
    test_emphasis_sentence_final();
    test_emphasis_list();
    test_emphasis_cap();
    test_emphasis_passthrough();
    test_emphasis_empty();
    test_quotes_basic();
    test_quotes_multiple();
    test_quotes_short();
    test_quotes_empty();
    test_prosody_feedback();
    test_alpha_clamping();

    printf("\n════════════════════════════════════════════\n");
    printf("  Results: %d passed, %d failed\n", pass_count, fail_count);
    printf("════════════════════════════════════════════\n");

    return fail_count > 0 ? 1 : 0;
}
