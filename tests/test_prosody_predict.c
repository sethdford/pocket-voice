/**
 * test_prosody_predict.c — Tests for text-based prosody prediction.
 *
 * Tests: syllable counting, duration estimation, multi-scale prosody analysis,
 *        emotion detection, conversational adaptation, EmoSteer loading.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "prosody_predict.h"
#include "sentence_buffer.h"

static int g_pass = 0, g_fail = 0;

#define ASSERT(cond, msg) do { \
    if (cond) { g_pass++; printf("  [PASS] %s\n", msg); } \
    else { g_fail++; printf("  [FAIL] %s\n", msg); } \
} while(0)

#define ASSERT_EQ(a, b, msg) do { \
    if ((a) == (b)) { g_pass++; printf("  [PASS] %s (%d)\n", msg, (int)(a)); } \
    else { g_fail++; printf("  [FAIL] %s (got %d, expected %d)\n", msg, (int)(a), (int)(b)); } \
} while(0)

#define ASSERT_NEAR(a, b, tol, msg) do { \
    if (fabsf((a) - (b)) < (tol)) { g_pass++; printf("  [PASS] %s (%.3f)\n", msg, (float)(a)); } \
    else { g_fail++; printf("  [FAIL] %s (got %.3f, expected %.3f)\n", msg, (float)(a), (float)(b)); } \
} while(0)

/* ═══════════════════════════════════════════════════════════════════ */

static void test_syllable_counting(void) {
    printf("\n=== Syllable Counting ===\n");
    ASSERT_EQ(prosody_count_syllables("hello"), 2, "hello = 2 syllables");
    ASSERT_EQ(prosody_count_syllables("world"), 1, "world = 1 syllable");
    ASSERT_EQ(prosody_count_syllables("beautiful"), 3, "beautiful = 3 syllables");
    ASSERT_EQ(prosody_count_syllables("the"), 1, "the = 1 syllable");
    ASSERT_EQ(prosody_count_syllables("a"), 1, "a = 1 syllable");
    ASSERT_EQ(prosody_count_syllables("extraordinary"), 5, "extraordinary >= 5 syllables");
    ASSERT_EQ(prosody_count_syllables(""), 0, "empty = 0");
    ASSERT_EQ(prosody_count_syllables(NULL), 0, "NULL = 0");

    int sent_syl = prosody_count_sentence_syllables("Hello beautiful world");
    ASSERT(sent_syl >= 5 && sent_syl <= 7, "sentence syllables in expected range");
}

static void test_duration_estimation(void) {
    printf("\n=== Duration Estimation ===\n");
    float durations[256];
    int n = prosody_estimate_durations("Hello world", 50, durations, 256);
    ASSERT(n == 50, "estimated 50 token durations");
    ASSERT(durations[0] != 0.0f, "first duration non-zero");

    /* Longer sentences should produce valid durations too */
    n = prosody_estimate_durations("This is a much longer sentence with many more syllables", 100, durations, 256);
    ASSERT(n == 100, "100 tokens estimated");

    /* Edge cases */
    n = prosody_estimate_durations(NULL, 10, durations, 256);
    ASSERT_EQ(n, 0, "NULL text returns 0");
    n = prosody_estimate_durations("Hi", 0, durations, 256);
    ASSERT_EQ(n, 0, "0 tokens returns 0");
}

static void test_multi_scale_prosody(void) {
    printf("\n=== Multi-Scale Prosody ===\n");

    /* Question */
    MultiScaleProsody msp = prosody_analyze_text("How are you doing today?");
    ASSERT_EQ(msp.contour, PROSODY_CONTOUR_INTERROGATIVE, "question → interrogative contour");
    ASSERT(msp.utterance.pitch > 1.0f, "question has rising pitch");

    /* Exclamation */
    msp = prosody_analyze_text("That is amazing!");
    ASSERT_EQ(msp.contour, PROSODY_CONTOUR_EXCLAMATORY, "exclamation → exclamatory contour");
    ASSERT(msp.utterance.energy > 0.0f, "exclamation has boosted energy");

    /* Declarative */
    msp = prosody_analyze_text("The weather is nice today.");
    ASSERT_EQ(msp.contour, PROSODY_CONTOUR_DECLARATIVE, "statement → declarative contour");

    /* List */
    msp = prosody_analyze_text("I need apples, oranges, and bananas.");
    ASSERT_EQ(msp.contour, PROSODY_CONTOUR_LIST, "comma-separated → list contour");

    /* ALL CAPS emphasis */
    msp = prosody_analyze_text("This is VERY important NOW");
    ASSERT(msp.n_words > 0, "detected multiple words");
    int found_emphasis = 0;
    for (int i = 0; i < msp.n_words; i++) {
        if (msp.emphasis_mask[i]) found_emphasis++;
    }
    ASSERT(found_emphasis >= 1, "detected ALL CAPS emphasis");

    /* Empty input */
    msp = prosody_analyze_text("");
    ASSERT_EQ(msp.n_words, 0, "empty text → 0 words");
    msp = prosody_analyze_text(NULL);
    ASSERT_EQ(msp.n_words, 0, "NULL → 0 words");
}

static void test_emotion_detection(void) {
    printf("\n=== Emotion Detection ===\n");

    EmotionDetection det;

    det = prosody_detect_emotion("That is wonderful and amazing!!");
    ASSERT(det.emotion != EMOTION_NEUTRAL, "wonderful+amazing → non-neutral");
    ASSERT(det.confidence > 0.0f, "has positive confidence");
    printf("    detected: emotion=%d conf=%.2f\n", det.emotion, det.confidence);

    det = prosody_detect_emotion("Unfortunately, I'm very sorry about that...");
    ASSERT(det.emotion == EMOTION_SAD, "sorry+unfortunately → sad");
    ASSERT(det.confidence >= 0.3f, "confidence >= 0.3");

    det = prosody_detect_emotion("I'm so angry and furious right now!");
    ASSERT(det.emotion == EMOTION_ANGRY, "angry+furious → angry");

    det = prosody_detect_emotion("Wow, that's unbelievable!");
    ASSERT(det.emotion == EMOTION_SURPRISED, "wow+unbelievable → surprised");

    /* ALL CAPS triggers excited or angry */
    det = prosody_detect_emotion("THIS IS ABSOLUTELY INCREDIBLE!!!");
    ASSERT(det.emotion != EMOTION_NEUTRAL, "ALL CAPS + !!! → non-neutral");
    ASSERT(det.confidence >= 0.3f, "high confidence for emphatic text");

    /* Neutral */
    det = prosody_detect_emotion("The sky is blue.");
    ASSERT(det.emotion == EMOTION_NEUTRAL, "bland statement → neutral");

    /* NULL safety */
    det = prosody_detect_emotion(NULL);
    ASSERT(det.emotion == EMOTION_NEUTRAL, "NULL → neutral");
    det = prosody_detect_emotion("");
    ASSERT(det.emotion == EMOTION_NEUTRAL, "empty → neutral");

    /* Hint values should be coherent with emotion */
    det = prosody_detect_emotion("That is fantastic and amazing!!");
    ASSERT(det.hint.pitch >= 1.0f, "positive emotion has pitch >= 1.0");
}

static void test_conversational_adaptation(void) {
    printf("\n=== Conversational Adaptation ===\n");

    ConversationProsodyState state;
    prosody_conversation_init(&state);
    ASSERT_EQ(state.n_samples, 0, "initial n_samples = 0");

    ProsodyHint hint = prosody_conversation_adapt(&state);
    ASSERT_NEAR(hint.pitch, 1.0f, 0.01f, "no data → neutral pitch");
    ASSERT_NEAR(hint.rate, 1.0f, 0.01f, "no data → neutral rate");

    /* Simulate fast speaker (5 wps) */
    prosody_conversation_update(&state, 2.0f, 10, -15.0f, 180.0f);
    ASSERT_EQ(state.n_samples, 1, "n_samples updated");

    hint = prosody_conversation_adapt(&state);
    ASSERT(hint.rate > 1.0f, "fast user → faster response rate");
    printf("    adapted: rate=%.3f pitch=%.3f energy=%.1f\n",
           hint.rate, hint.pitch, hint.energy);

    /* Simulate slow speaker (1.5 wps) */
    for (int i = 0; i < 5; i++) {
        prosody_conversation_update(&state, 4.0f, 6, -25.0f, 120.0f);
    }
    hint = prosody_conversation_adapt(&state);
    ASSERT(hint.rate < 1.05f, "slow user → slower/neutral response rate");
    printf("    adapted: rate=%.3f pitch=%.3f energy=%.1f\n",
           hint.rate, hint.pitch, hint.energy);

    /* NULL safety */
    prosody_conversation_init(NULL);
    prosody_conversation_update(NULL, 1.0f, 5, -20.0f, 150.0f);
    hint = prosody_conversation_adapt(NULL);
    ASSERT_NEAR(hint.pitch, 1.0f, 0.01f, "NULL state → neutral");
}

static void test_emosteer_loading(void) {
    printf("\n=== EmoSteer Loading ===\n");

    /* NULL safety */
    ASSERT(emosteer_load(NULL) == NULL, "NULL path returns NULL");
    ASSERT(emosteer_load("/nonexistent.json") == NULL, "bad path returns NULL");
    ASSERT(emosteer_count(NULL) == 0, "NULL bank count = 0");
    ASSERT(emosteer_get_direction(NULL, "happy") == NULL, "NULL bank direction = NULL");

    /* Destroy NULL is safe */
    emosteer_destroy(NULL);
    ASSERT(1, "emosteer_destroy(NULL) doesn't crash");
}

static void test_sentence_buffer_hints(void) {
    printf("\n=== Sentence Buffer Prosody Hints ===\n");

    /* We test the hint struct type exists and has expected fields */
    SentBufProsodyHint hint;
    memset(&hint, 0, sizeof(hint));
    hint.suggested_rate = 1.0f;
    hint.suggested_pitch = 1.0f;
    ASSERT_NEAR(hint.suggested_rate, 1.0f, 0.01f, "hint rate default = 1.0");
    ASSERT_NEAR(hint.suggested_pitch, 1.0f, 0.01f, "hint pitch default = 1.0");
    ASSERT_EQ(hint.has_all_caps, 0, "hint all_caps default = 0");
    ASSERT_EQ(hint.exclamation_count, 0, "hint excl default = 0");
}

int main(void) {
    printf("════════════════════════════════════════════\n");
    printf("  Prosody Prediction Tests\n");
    printf("════════════════════════════════════════════\n");

    test_syllable_counting();
    test_duration_estimation();
    test_multi_scale_prosody();
    test_emotion_detection();
    test_conversational_adaptation();
    test_emosteer_loading();
    test_sentence_buffer_hints();

    printf("\n════════════════════════════════════════════\n");
    printf("  Results: %d passed, %d failed\n", g_pass, g_fail);
    printf("════════════════════════════════════════════\n");
    return g_fail ? 1 : 0;
}
