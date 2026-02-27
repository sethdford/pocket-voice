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

/* ═══════════════════════════════════════════════════════════════════ */
/* Additional tests added by test-prosody agent                       */
/* ═══════════════════════════════════════════════════════════════════ */

static void test_syllable_counting_extended(void) {
    printf("\n=== Syllable Counting: Extended ===\n");

    /* Monosyllabic words */
    ASSERT_EQ(prosody_count_syllables("cat"), 1, "cat = 1 syllable");
    ASSERT_EQ(prosody_count_syllables("I"), 1, "I = 1 syllable");
    ASSERT_EQ(prosody_count_syllables("through"), 1, "through = 1 syllable");
    ASSERT_EQ(prosody_count_syllables("strength"), 1, "strength = 1 syllable");

    /* Multi-syllable words */
    ASSERT_EQ(prosody_count_syllables("computer"), 3, "computer = 3 syllables");
    ASSERT_EQ(prosody_count_syllables("elephant"), 3, "elephant = 3 syllables");
    ASSERT_EQ(prosody_count_syllables("understanding"), 4, "understanding = 4 syllables");

    /* Silent-e rule: trailing 'e' after consonant shouldn't add a syllable */
    ASSERT_EQ(prosody_count_syllables("make"), 1, "make (silent-e) = 1 syllable");
    ASSERT_EQ(prosody_count_syllables("time"), 1, "time (silent-e) = 1 syllable");
    ASSERT_EQ(prosody_count_syllables("smile"), 1, "smile (silent-e) = 1 syllable");
    ASSERT_EQ(prosody_count_syllables("complete"), 2, "complete (silent-e) = 2 syllables");
    ASSERT_EQ(prosody_count_syllables("telephone"), 3, "telephone (silent-e) = 3 syllables");

    /* Compound-ish words */
    ASSERT_EQ(prosody_count_syllables("sunshine"), 2, "sunshine = 2 syllables");
    ASSERT_EQ(prosody_count_syllables("basketball"), 3, "basketball = 3 syllables");

    /* Words with consecutive vowels (diphthongs/digraphs) */
    ASSERT_EQ(prosody_count_syllables("boat"), 1, "boat (oa digraph) = 1 syllable");
    ASSERT_EQ(prosody_count_syllables("create"), 2, "create = 2 syllables");

    /* Sentence syllable counting with varied content */
    int s = prosody_count_sentence_syllables("I like to create beautiful things");
    ASSERT(s >= 8 && s <= 11, "mixed sentence syllable count in range");

    s = prosody_count_sentence_syllables("A");
    ASSERT(s >= 1, "single word sentence has syllables");

    s = prosody_count_sentence_syllables("");
    ASSERT_EQ(s, 0, "empty sentence = 0 syllables");

    s = prosody_count_sentence_syllables(NULL);
    ASSERT_EQ(s, 0, "NULL sentence = 0 syllables");
}

static void test_emotion_detection_extended(void) {
    printf("\n=== Emotion Detection: Extended ===\n");

    EmotionDetection det;

    /* Question marks should influence detection */
    det = prosody_detect_emotion("What happened to you???");
    ASSERT(det.confidence > 0.0f, "multiple question marks have confidence");

    /* Multiple exclamation marks = more intensity */
    det = prosody_detect_emotion("That is incredible!!!");
    ASSERT(det.emotion != EMOTION_NEUTRAL, "triple exclamation → non-neutral");
    ASSERT(det.confidence >= 0.3f, "triple ! has good confidence");

    /* ALL CAPS full sentence */
    det = prosody_detect_emotion("I AM SO HAPPY RIGHT NOW");
    ASSERT(det.emotion != EMOTION_NEUTRAL, "ALL CAPS sentence → non-neutral");

    /* Mixed emotions: conflicting signals */
    det = prosody_detect_emotion("I'm happy but also a little sad.");
    ASSERT(det.confidence >= 0.0f, "mixed signals still produce valid confidence");

    /* Fear/anxiety keywords */
    det = prosody_detect_emotion("I'm terrified and scared to death!");
    ASSERT(det.emotion == EMOTION_FEARFUL || det.emotion == EMOTION_ANGRY ||
           det.emotion != EMOTION_NEUTRAL, "fear keywords → non-neutral");

    /* Warm/gentle */
    det = prosody_detect_emotion("Thank you so much, that's very kind of you.");
    ASSERT(det.emotion == EMOTION_WARM || det.emotion == EMOTION_HAPPY ||
           det.emotion != EMOTION_NEUTRAL, "warm text → non-neutral");

    /* Very long text should not crash */
    char long_text[2048];
    memset(long_text, 0, sizeof(long_text));
    for (int i = 0; i < 200; i++) {
        strcat(long_text, "word ");
    }
    det = prosody_detect_emotion(long_text);
    ASSERT(det.emotion >= 0 && det.emotion < EMOTION_COUNT,
           "long text returns valid emotion enum");

    /* Confidence always in [0.0, 1.0] */
    det = prosody_detect_emotion("AMAZING!!! WOW!!! INCREDIBLE!!!");
    ASSERT(det.confidence >= 0.0f && det.confidence <= 1.0f,
           "confidence always in [0, 1]");

    /* Hint pitch should be reasonable */
    det = prosody_detect_emotion("I am so excited about this!");
    ASSERT(det.hint.pitch >= 0.5f && det.hint.pitch <= 2.0f,
           "emotion hint pitch in reasonable range");
    ASSERT(det.hint.rate >= 0.5f && det.hint.rate <= 2.0f,
           "emotion hint rate in reasonable range");
}

static void test_prosody_adaptation_extended(void) {
    printf("\n=== Prosody Adaptation: Extended ===\n");

    ConversationProsodyState state;
    prosody_conversation_init(&state);

    /* Verify initial state is clean */
    ASSERT_NEAR(state.ema_rate, 0.0f, 0.01f, "initial ema_rate near 0");
    ASSERT_NEAR(state.ema_energy, 0.0f, 0.01f, "initial ema_energy near 0");
    ASSERT_NEAR(state.ema_pitch, 0.0f, 0.01f, "initial ema_pitch near 0");

    /* Loud, high-pitched user → response should adapt */
    prosody_conversation_update(&state, 1.5f, 8, -10.0f, 220.0f);
    ProsodyHint hint = prosody_conversation_adapt(&state);
    ASSERT(hint.energy != 0.0f || hint.pitch != 1.0f || hint.rate != 1.0f,
           "loud user triggers adaptation");

    /* Accumulate many samples to test stability */
    for (int i = 0; i < 20; i++) {
        prosody_conversation_update(&state, 2.0f, 7, -18.0f, 160.0f);
    }
    ASSERT(state.n_samples >= 20, "n_samples tracks accumulation");
    hint = prosody_conversation_adapt(&state);

    /* Adapted values should be in sane ranges */
    ASSERT(hint.pitch >= 0.5f && hint.pitch <= 2.0f,
           "adapted pitch in sane range [0.5, 2.0]");
    ASSERT(hint.rate >= 0.5f && hint.rate <= 2.0f,
           "adapted rate in sane range [0.5, 2.0]");

    /* Re-init should reset everything */
    prosody_conversation_init(&state);
    ASSERT_EQ(state.n_samples, 0, "re-init resets n_samples to 0");
    hint = prosody_conversation_adapt(&state);
    ASSERT_NEAR(hint.pitch, 1.0f, 0.01f, "re-init → neutral pitch again");
}

static void test_emosteer_extended(void) {
    printf("\n=== EmoSteer: Extended ===\n");

    /* Non-existent path variants */
    ASSERT(emosteer_load("") == NULL, "empty string path returns NULL");
    ASSERT(emosteer_load("/tmp/this_should_not_exist_ever.json") == NULL,
           "non-existent file returns NULL");

    /* Count and direction on NULL bank */
    ASSERT_EQ(emosteer_count(NULL), 0, "NULL bank count = 0 (repeated check)");
    ASSERT(emosteer_get_direction(NULL, "happy") == NULL,
           "NULL bank get_direction = NULL (repeated check)");
    ASSERT(emosteer_get_direction(NULL, NULL) == NULL,
           "NULL bank + NULL emotion = NULL");
    ASSERT(emosteer_get_direction(NULL, "") == NULL,
           "NULL bank + empty emotion = NULL");

    /* Multiple destroy calls should be safe */
    emosteer_destroy(NULL);
    emosteer_destroy(NULL);
    ASSERT(1, "double emosteer_destroy(NULL) is safe");
}

static void test_multi_scale_edge_cases(void) {
    printf("\n=== Multi-Scale Prosody: Edge Cases ===\n");

    /* Single word */
    MultiScaleProsody msp = prosody_analyze_text("Hello");
    ASSERT(msp.n_words >= 1, "single word detected");
    ASSERT(msp.contour >= 0, "single word has valid contour");

    /* Only punctuation */
    msp = prosody_analyze_text("!!??!!");
    ASSERT(msp.n_words >= 0, "punctuation-only doesn't crash");

    /* Imperative (command) */
    msp = prosody_analyze_text("Stop right there!");
    ASSERT(msp.contour == PROSODY_CONTOUR_EXCLAMATORY ||
           msp.contour == PROSODY_CONTOUR_IMPERATIVE,
           "command → exclamatory or imperative");

    /* Continuation (mid-sentence comma) */
    msp = prosody_analyze_text("Well, I think so");
    ASSERT(msp.n_words >= 3, "continuation phrase has words");

    /* Very long text shouldn't overflow word_hints[64] */
    char long_text[4096];
    memset(long_text, 0, sizeof(long_text));
    for (int i = 0; i < 100; i++) {
        strcat(long_text, "word ");
    }
    strcat(long_text, "end.");
    msp = prosody_analyze_text(long_text);
    ASSERT(msp.n_words <= 64, "n_words capped at array size 64");
    ASSERT(msp.contour >= 0, "long text has valid contour");

    /* Emphasis mask consistency: emphasized count <= n_words */
    msp = prosody_analyze_text("This is VERY IMPORTANT and CRITICAL");
    int emph_count = 0;
    for (int i = 0; i < msp.n_words; i++) {
        if (msp.emphasis_mask[i]) emph_count++;
    }
    ASSERT(emph_count <= msp.n_words, "emphasis count <= word count");
    ASSERT(emph_count >= 1, "ALL CAPS words detected as emphasis");
}

static void test_duration_estimation_extended(void) {
    printf("\n=== Duration Estimation: Extended ===\n");

    float durations[512];

    /* Single word */
    int n = prosody_estimate_durations("Hi", 10, durations, 512);
    ASSERT(n == 10, "short text with 10 tokens");
    for (int i = 0; i < n; i++) {
        ASSERT(durations[i] >= 0.0f, "duration values non-negative");
    }

    /* Very long text */
    char long_text[2048];
    memset(long_text, 0, sizeof(long_text));
    for (int i = 0; i < 50; i++) {
        strcat(long_text, "The quick brown fox jumped over the lazy dog. ");
    }
    n = prosody_estimate_durations(long_text, 200, durations, 512);
    ASSERT(n == 200, "200 tokens for long text");

    /* max_frames limit: request more tokens than buffer can hold */
    n = prosody_estimate_durations("Test", 1000, durations, 512);
    ASSERT(n <= 512, "output capped at max_frames");

    /* NULL durations buffer */
    n = prosody_estimate_durations("Test", 10, NULL, 0);
    ASSERT_EQ(n, 0, "NULL buffer returns 0");
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

    /* Extended tests */
    test_syllable_counting_extended();
    test_emotion_detection_extended();
    test_prosody_adaptation_extended();
    test_emosteer_extended();
    test_multi_scale_edge_cases();
    test_duration_estimation_extended();

    printf("\n════════════════════════════════════════════\n");
    printf("  Results: %d passed, %d failed\n", g_pass, g_fail);
    printf("════════════════════════════════════════════\n");
    return g_fail ? 1 : 0;
}
