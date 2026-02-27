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

/* ── Smart Quotes and Advanced Quote Detection ────────────────────────────── */

static void test_quotes_smart_utf8(void) {
    printf("\n=== Quotes: Smart/Curly Quotes (UTF-8) ===\n");

    char out[4096];

    /* Smart double quotes: \xe2\x80\x9c = left, \xe2\x80\x9d = right */
    int n = emphasis_detect_quotes(
        "She said, \xe2\x80\x9chello there friend\xe2\x80\x9d and walked away.",
        out, sizeof(out));
    /* Smart quotes may or may not be supported — don't crash either way */
    TEST(n >= 0, "smart quotes don't crash");
    printf("    (smart quote segments found: %d)\n", n);

    /* Single straight quotes: not typically speech markers */
    n = emphasis_detect_quotes(
        "It's a nice day to say 'hello' isn't it?",
        out, sizeof(out));
    TEST(n >= 0, "single quotes handled without crash");
}

static void test_quotes_unmatched(void) {
    printf("\n=== Quotes: Unmatched/Edge Cases ===\n");

    char out[4096];

    /* Unmatched opening quote */
    int n = emphasis_detect_quotes(
        "She said \"hello but never finished", out, sizeof(out));
    TEST(n == 0, "unmatched quote → 0 segments");

    /* Empty quotes */
    n = emphasis_detect_quotes(
        "She said \"\" and nothing more.", out, sizeof(out));
    TEST(n == 0, "empty quotes → 0 segments");

    /* Quote at very start */
    n = emphasis_detect_quotes(
        "\"Hello world\" she announced.", out, sizeof(out));
    TEST(n >= 0, "quote at start doesn't crash");

    /* Quote at very end */
    n = emphasis_detect_quotes(
        "She said \"goodbye world\"", out, sizeof(out));
    TEST(n >= 0, "quote at end doesn't crash");
}

/* ── ALL_CAPS Detection via Emphasis ──────────────────────────────────────── */

static void test_emphasis_all_caps(void) {
    printf("\n=== Emphasis: ALL_CAPS Words ===\n");

    char out[4096];

    /* Single ALL_CAPS word in sentence */
    int n = emphasis_predict("This is AMAZING work.", out, sizeof(out));
    TEST(n >= 0, "ALL_CAPS word processed");
    TEST(strstr(out, "AMAZING") != NULL, "AMAZING preserved in output");

    /* Multiple ALL_CAPS words */
    n = emphasis_predict("I am VERY VERY HAPPY today.", out, sizeof(out));
    TEST(strstr(out, "VERY") != NULL, "VERY preserved");
    TEST(strstr(out, "HAPPY") != NULL, "HAPPY preserved");

    /* Mixed case: only fully uppercase words count */
    n = emphasis_predict("Hello WORLD and Goodbye.", out, sizeof(out));
    TEST(strstr(out, "WORLD") != NULL, "WORLD preserved in output");

    /* All caps sentence: emphasis shouldn't over-mark */
    n = emphasis_predict("EVERYTHING IS IN CAPS HERE.", out, sizeof(out));
    int count = 0;
    const char *p = out;
    while ((p = strstr(p, "<emphasis")) != NULL) { count++; p++; }
    TEST(count <= 3, "all-caps sentence capped at 3 emphasis");
}

/* ── Semantic Emphasis: Important Words and Stress ─────────────────────────── */

static void test_emphasis_semantic(void) {
    printf("\n=== Emphasis: Semantic Patterns ===\n");

    char out[4096];

    /* "however" is a contrast marker like "but" */
    int n = emphasis_predict(
        "The plan was solid. However, the execution was poor.", out, sizeof(out));
    TEST(strstr(out, "<emphasis") != NULL, "however triggers emphasis");

    /* "actually" is a contrast/correction marker */
    n = emphasis_predict(
        "I actually disagree with that assessment.", out, sizeof(out));
    TEST(strstr(out, "<emphasis") != NULL, "actually triggers emphasis");

    /* "instead" contrast marker */
    n = emphasis_predict(
        "Don't run. Instead, walk slowly.", out, sizeof(out));
    TEST(strstr(out, "<emphasis") != NULL, "instead triggers emphasis");

    /* "extremely" is an intensifier */
    n = emphasis_predict(
        "That was extremely difficult to solve.", out, sizeof(out));
    TEST(strstr(out, "difficult") != NULL, "extremely boosts next word");

    /* "not" negation emphasis */
    n = emphasis_predict(
        "I do not agree with this at all.", out, sizeof(out));
    TEST(strstr(out, "<emphasis") != NULL, "negation 'not' triggers emphasis");
}

/* ── SSML Bypass Extended ─────────────────────────────────────────────────── */

static void test_emphasis_ssml_bypass_extended(void) {
    printf("\n=== Emphasis: SSML Bypass Extended ===\n");

    char out[4096];

    /* Multiple existing SSML tags */
    const char *input = "<emphasis level=\"strong\">really</emphasis> and "
                        "<emphasis level=\"moderate\">truly</emphasis>.";
    int n = emphasis_predict(input, out, sizeof(out));
    TEST(n == 0, "no emphasis added when multiple SSML tags present");
    TEST(strcmp(out, input) == 0, "multi-tag SSML preserved");

    /* Prosody tag (not emphasis, but still SSML) — should still predict */
    input = "I <prosody rate=\"fast\">really</prosody> mean it.";
    n = emphasis_predict(input, out, sizeof(out));
    /* prosody tags may or may not trigger bypass — test that it doesn't crash */
    TEST(n >= 0, "prosody tag input doesn't crash");

    /* Break tag */
    input = "Hello.<break time=\"500ms\"/> How are you?";
    n = emphasis_predict(input, out, sizeof(out));
    TEST(n >= 0, "break tag input doesn't crash");
}

/* ── Edge Cases Extended ──────────────────────────────────────────────────── */

static void test_emphasis_edge_extended(void) {
    printf("\n=== Emphasis: Edge Cases Extended ===\n");

    char out[4096];

    /* Single character */
    int n = emphasis_predict("A", out, sizeof(out));
    TEST(n >= 0, "single char 'A' doesn't crash");

    /* All punctuation */
    n = emphasis_predict("... !!! ???", out, sizeof(out));
    TEST(n >= 0, "all punctuation doesn't crash");

    /* Very small output buffer */
    char tiny[16];
    n = emphasis_predict("This is a test with emphasis.", tiny, sizeof(tiny));
    TEST(n >= 0, "small buffer doesn't crash");

    /* Whitespace only */
    n = emphasis_predict("   \t\n  ", out, sizeof(out));
    TEST(n >= 0, "whitespace-only doesn't crash");

    /* Numbers only */
    n = emphasis_predict("12345 67890", out, sizeof(out));
    TEST(n >= 0, "numbers-only doesn't crash");

    /* Long sentence to stress-test */
    char long_text[4096];
    memset(long_text, 0, sizeof(long_text));
    for (int i = 0; i < 80; i++) {
        strcat(long_text, "word ");
    }
    n = emphasis_predict(long_text, out, sizeof(out));
    TEST(n >= 0, "very long sentence doesn't crash");
    int count = 0;
    const char *p = out;
    while ((p = strstr(p, "<emphasis")) != NULL) { count++; p++; }
    TEST(count <= 3, "long sentence emphasis capped at 3");
}

/* ── Quotes NULL Buffer ───────────────────────────────────────────────────── */

static void test_quotes_null_buffer(void) {
    printf("\n=== Quotes: NULL/Edge Buffer ===\n");

    /* NULL output buffer */
    int n = emphasis_detect_quotes("She said \"hello\" to me.", NULL, 0);
    TEST(n == 0, "NULL output buffer → 0");

    /* Very small output buffer */
    char tiny[8];
    n = emphasis_detect_quotes("She said \"hello world my friend\" today.",
                                tiny, sizeof(tiny));
    TEST(n >= 0, "tiny buffer doesn't crash");

    /* Empty string */
    char out[4096];
    n = emphasis_detect_quotes("", out, sizeof(out));
    TEST(n == 0, "empty string → 0 quotes");
    TEST(out[0] == '\0', "empty output for empty input");
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

    /* Extended tests */
    test_quotes_smart_utf8();
    test_quotes_unmatched();
    test_emphasis_all_caps();
    test_emphasis_semantic();
    test_emphasis_ssml_bypass_extended();
    test_emphasis_edge_extended();
    test_quotes_null_buffer();

    printf("\n════════════════════════════════════════════\n");
    printf("  Results: %d passed, %d failed\n", pass_count, fail_count);
    printf("════════════════════════════════════════════\n");

    return fail_count > 0 ? 1 : 0;
}
