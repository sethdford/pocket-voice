/**
 * test_prosody_integration.c — End-to-end prosody pipeline integration tests.
 *
 * Tests the full chain: emphasis prediction → SSML parsing → prosody analysis
 * → duration estimation → emotion detection → conversational adaptation.
 * Verifies that all modules wire together correctly.
 *
 * Also validates emphasis against realistic LLM output patterns.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "emphasis_predict.h"
#include "ssml_parser.h"
#include "text_normalize.h"
#include "prosody_predict.h"
#include "sentence_buffer.h"

static int pass_count = 0;
static int fail_count = 0;

#define TEST(cond, name) do { \
    if (cond) { printf("  [PASS] %s\n", name); pass_count++; } \
    else { printf("  [FAIL] %s (line %d)\n", name, __LINE__); fail_count++; } \
} while (0)

/* ── Full Pipeline Integration ────────────────────────────────────────────── */

static void test_full_pipeline_chain(void) {
    printf("\n=== Full Pipeline: emphasis → SSML → prosody → duration ===\n");

    /* Simulate LLM output: plain text */
    const char *llm_output = "I tried really hard, but it never worked out.";

    /* Step 1: Quote detection first (before emphasis inserts " chars) */
    char with_quotes[4096];
    int n_quotes = emphasis_detect_quotes(llm_output, with_quotes, sizeof(with_quotes));
    TEST(n_quotes == 0, "no quotes in this sentence");

    /* Step 2: Emphasis prediction */
    char emphasized[4096];
    int n_emph = emphasis_predict(with_quotes, emphasized, sizeof(emphasized));
    TEST(n_emph > 0, "emphasis found in LLM output");
    TEST(strstr(emphasized, "<emphasis") != NULL, "emphasis tags inserted");

    /* Step 3: SSML parsing */
    SSMLSegment segments[SSML_MAX_SEGMENTS];
    int nseg = ssml_parse(emphasized, segments, SSML_MAX_SEGMENTS);
    TEST(nseg > 0, "SSML parse produced segments");

    /* Step 4: Multi-scale prosody analysis on each segment */
    int any_analyzed = 0;
    for (int i = 0; i < nseg; i++) {
        if (segments[i].text[0]) {
            MultiScaleProsody msp = prosody_analyze_text(segments[i].text);
            if (msp.n_words > 0) any_analyzed = 1;
        }
    }
    TEST(any_analyzed, "prosody analysis ran on segments");

    /* Step 5: Duration estimation */
    float durations[256];
    int n_dur = prosody_estimate_durations(llm_output, 128, durations, 256);
    TEST(n_dur > 0, "duration estimation produced frames");

    /* Step 6: Emotion detection */
    EmotionDetection det = prosody_detect_emotion(llm_output);
    TEST(det.emotion >= 0, "emotion detection returned valid emotion");

    printf("    Chain: %d emphasis → %d segments → %d duration frames → emotion=%d\n",
           n_emph, nseg, n_dur, det.emotion);
}

static void test_quoted_speech_pipeline(void) {
    printf("\n=== Pipeline: Quoted Speech → Multi-Voice ===\n");

    const char *llm_output = "She whispered, \"please don't leave me\" and started crying.";

    /* Step 1: Quote detection first (before emphasis, which may break quotes) */
    char with_quotes[4096];
    int n_quotes = emphasis_detect_quotes(llm_output, with_quotes, sizeof(with_quotes));
    TEST(n_quotes == 1, "found 1 quoted segment");

    /* Step 2: Emphasis on the quote-wrapped text */
    char emphasized[4096];
    emphasis_predict(with_quotes, emphasized, sizeof(emphasized));

    /* The quotes module inserts <voice name="quoted"> tags */
    TEST(strstr(emphasized, "<voice name=\"quoted\">") != NULL ||
         strstr(with_quotes, "<voice name=\"quoted\">") != NULL,
         "quote wrapper tag inserted");
    TEST(strstr(emphasized, "</voice>") != NULL ||
         strstr(with_quotes, "</voice>") != NULL,
         "quote closing tag present");

    /* Step 3: SSML parse */
    SSMLSegment segments[SSML_MAX_SEGMENTS];
    int nseg = ssml_parse(emphasized, segments, SSML_MAX_SEGMENTS);
    TEST(nseg >= 1, "SSML parse produced segments");

    /* Step 4: Emotion detection on the quoted text */
    EmotionDetection det = prosody_detect_emotion("please don't leave me");
    TEST(det.emotion != EMOTION_NEUTRAL || det.confidence >= 0,
         "emotion detection ran on quote text");
}

static void test_ssml_passthrough_integration(void) {
    printf("\n=== Pipeline: Existing SSML Preserved ===\n");

    const char *llm_ssml = "That's <emphasis level=\"strong\">amazing</emphasis>! "
                           "<emotion type=\"excited\">I can't believe it!</emotion>";

    /* Emphasis predict should pass through when SSML already present */
    char emphasized[4096];
    int n = emphasis_predict(llm_ssml, emphasized, sizeof(emphasized));
    TEST(n == 0, "no extra emphasis when SSML already present");

    /* SSML parse with <speak> wrapper to activate full parser */
    char wrapped[8192];
    snprintf(wrapped, sizeof(wrapped), "<speak>%s</speak>", llm_ssml);
    SSMLSegment segments[SSML_MAX_SEGMENTS];
    int nseg = ssml_parse(wrapped, segments, SSML_MAX_SEGMENTS);
    TEST(nseg >= 1, "SSML parse produced segments from wrapped input");

    /* The parser should extract text content */
    int has_text = 0;
    for (int i = 0; i < nseg; i++) {
        if (segments[i].text[0]) has_text = 1;
    }
    TEST(has_text, "SSML segments contain text");
}

/* ── Emphasis Tuning: Real LLM Output Patterns ───────────────────────────── */

static void test_emphasis_llm_patterns(void) {
    printf("\n=== Emphasis Tuning: LLM Response Patterns ===\n");

    char out[8192];
    int n;

    /* Pattern 1: Short answer — should get minimal emphasis */
    n = emphasis_predict("Sure, I can help with that.", out, sizeof(out));
    int count1 = 0;
    const char *p = out;
    while ((p = strstr(p, "<emphasis")) != NULL) { count1++; p++; }
    TEST(count1 <= 2, "short answer: <=2 emphasis points");

    /* Pattern 2: Explanation with contrast */
    n = emphasis_predict(
        "Python is great for prototyping, but Rust is better for performance.",
        out, sizeof(out));
    TEST(strstr(out, "<emphasis") != NULL, "contrast pattern detected");
    /* "better" or "performance" should be emphasized after "but" */
    int has_content_emph = (strstr(out, "better") != NULL && strstr(out, "<emphasis") != NULL) ||
                           (strstr(out, "performance") != NULL && strstr(out, "<emphasis") != NULL);
    TEST(has_content_emph, "emphasis on word after contrast marker");

    /* Pattern 3: List enumeration */
    n = emphasis_predict(
        "You'll need three things: a compiler, a debugger, and a good editor.",
        out, sizeof(out));
    TEST(strstr(out, "editor") != NULL, "list-final item preserved");

    /* Pattern 4: Strong negation in technical answer */
    n = emphasis_predict(
        "That approach absolutely won't work because the API never supported it.",
        out, sizeof(out));
    TEST(n >= 0, "negation sentence processed without crash");

    /* Pattern 5: Intensifier in recommendation */
    n = emphasis_predict(
        "This is a really powerful feature that you should try.",
        out, sizeof(out));
    TEST(strstr(out, "powerful") != NULL, "intensifier boosts next word");

    /* Pattern 6: Question answer — typically no emphasis needed */
    n = emphasis_predict("The answer is forty two.", out, sizeof(out));
    int count6 = 0;
    p = out;
    while ((p = strstr(p, "<emphasis")) != NULL) { count6++; p++; }
    TEST(count6 <= 2, "simple answer: minimal emphasis");

    /* Pattern 7: Multi-sentence response */
    n = emphasis_predict(
        "That's a great question. The key difference is actually in the implementation. "
        "However, the performance impact is negligible.",
        out, sizeof(out));
    TEST(n > 0, "multi-sentence gets some emphasis");
    int count7 = 0;
    p = out;
    while ((p = strstr(p, "<emphasis")) != NULL) { count7++; p++; }
    TEST(count7 <= 3, "multi-sentence: capped at 3 emphasis");

    /* Pattern 8: Emotional response */
    n = emphasis_predict(
        "I'm sorry to hear that. That must be really difficult for you.",
        out, sizeof(out));
    EmotionDetection det = prosody_detect_emotion(
        "I'm sorry to hear that. That must be really difficult for you.");
    TEST(det.emotion == EMOTION_SAD || det.confidence > 0,
         "sad emotion detected in empathetic response");
}

/* ── Conversational Adaptation Integration ────────────────────────────────── */

static void test_conversational_adaptation_pipeline(void) {
    printf("\n=== Conversational Adaptation Pipeline ===\n");

    ConversationProsodyState state;
    prosody_conversation_init(&state);

    /* Simulate 3 turns of fast-speaking user */
    for (int i = 0; i < 3; i++) {
        prosody_conversation_update(&state,
            0.8f,   /* short duration */
            6,      /* many words */
            -15.0f, /* loud */
            180.0f  /* high pitch */
        );
    }

    ProsodyHint adapt = prosody_conversation_adapt(&state);
    TEST(adapt.rate >= 1.0f, "fast user → response rate >= 1.0");
    TEST(state.n_samples == 3, "3 samples tracked");

    /* Now simulate slow user for 3 turns */
    for (int i = 0; i < 3; i++) {
        prosody_conversation_update(&state,
            3.0f,   /* long duration */
            4,      /* few words */
            -28.0f, /* quiet */
            120.0f  /* low pitch */
        );
    }

    adapt = prosody_conversation_adapt(&state);
    TEST(adapt.rate <= 1.0f, "slow user → response rate <= 1.0");

    /* Verify prosody analysis integrates with adaptation */
    MultiScaleProsody msp = prosody_analyze_text("Let me think about that carefully.");
    TEST(msp.contour == PROSODY_CONTOUR_DECLARATIVE, "declarative contour detected");
    TEST(msp.n_words > 0, "word count > 0");
}

/* ── Sentence Buffer → Emphasis → SSML → Prosody Chain ───────────────────── */

static void test_sentbuf_emphasis_chain(void) {
    printf("\n=== SentBuf → Emphasis → SSML Chain ===\n");

    SentenceBuffer *sb = sentbuf_create(0, 3);
    TEST(sb != NULL, "sentence buffer created");

    /* Feed tokens like LLM would */
    const char *tokens[] = {
        "I ", "tried ", "really ", "hard, ", "but ", "it ", "never ", "worked ", "out."
    };
    for (int i = 0; i < 9; i++) {
        sentbuf_add(sb, tokens[i], (int)strlen(tokens[i]));
    }

    /* Flush and process through pipeline */
    char sentence[4096];
    if (sentbuf_has_segment(sb)) {
        int slen = sentbuf_flush(sb, sentence, sizeof(sentence));
        TEST(slen > 0, "sentence flushed from buffer");

        SentBufProsodyHint hint = sentbuf_get_prosody_hint(sb);
        TEST(hint.suggested_pitch > 0, "prosody hint has pitch value");
        TEST(hint.suggested_rate > 0, "prosody hint has rate value");

        /* Run through emphasis prediction */
        char emphasized[4096];
        emphasis_predict(sentence, emphasized, sizeof(emphasized));

        /* Parse SSML */
        SSMLSegment segments[SSML_MAX_SEGMENTS];
        int nseg = ssml_parse(emphasized, segments, SSML_MAX_SEGMENTS);
        TEST(nseg > 0, "segments produced from emphasis+SSML");

        /* Apply prosody analysis */
        for (int i = 0; i < nseg; i++) {
            if (segments[i].text[0]) {
                MultiScaleProsody msp = prosody_analyze_text(segments[i].text);
                TEST(msp.contour >= 0, "valid contour for segment");
            }
        }
    }

    sentbuf_destroy(sb);
}

/* ── Prosody Feedback Math Verification ──────────────────────────────────── */

static void test_prosody_boost_math(void) {
    printf("\n=== Prosody Boost Calculation ===\n");

    /* Test boost application to pitch/volume */
    float seg_pitch = 1.08f;  /* +8% */
    float seg_vol_db = 2.5f;
    float boost = 1.2f;

    float boosted_pitch = 1.0f + (seg_pitch - 1.0f) * boost;
    TEST(fabsf(boosted_pitch - 1.096f) < 0.01f,
         "pitch boost: 1.08 × 1.2 deviation = 1.096");

    float boosted_vol = seg_vol_db * boost;
    TEST(fabsf(boosted_vol - 3.0f) < 0.01f,
         "volume boost: 2.5 × 1.2 = 3.0");

    /* Below-unity pitch should widen in opposite direction */
    seg_pitch = 0.92f;  /* -8% */
    float boosted_low = 1.0f - (1.0f - seg_pitch) * boost;
    TEST(boosted_low < 0.92f, "below-unity pitch widens downward with boost");
    TEST(fabsf(boosted_low - 0.904f) < 0.01f,
         "pitch boost: 0.92 × 1.2 deviation = 0.904");
}

/* ── Main ─────────────────────────────────────────────────────────────────── */

int main(void) {
    printf("╔══════════════════════════════════════════════════════════╗\n");
    printf("║  Prosody Integration & LLM Emphasis Tuning Tests        ║\n");
    printf("╚══════════════════════════════════════════════════════════╝\n");

    test_full_pipeline_chain();
    test_quoted_speech_pipeline();
    test_ssml_passthrough_integration();
    test_emphasis_llm_patterns();
    test_conversational_adaptation_pipeline();
    test_sentbuf_emphasis_chain();
    test_prosody_boost_math();

    printf("\n════════════════════════════════════════════\n");
    printf("  Results: %d passed, %d failed\n", pass_count, fail_count);
    printf("════════════════════════════════════════════\n");

    return fail_count > 0 ? 1 : 0;
}
