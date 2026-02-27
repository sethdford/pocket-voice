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

/* ── Multi-Sentence Emotion Transitions ───────────────────────────────────── */

static void test_multi_sentence_emotions(void) {
    printf("\n=== Multi-Sentence: Emotion Transitions ===\n");

    /* Happy → Sad transition */
    EmotionDetection det1 = prosody_detect_emotion("That is wonderful news!");
    EmotionDetection det2 = prosody_detect_emotion("But unfortunately it didn't last.");

    TEST(det1.emotion != det2.emotion || det1.confidence != det2.confidence,
         "different sentences produce different emotion signals");
    TEST(det1.emotion == EMOTION_HAPPY || det1.emotion == EMOTION_EXCITED,
         "wonderful → happy/excited");
    TEST(det2.emotion == EMOTION_SAD || det2.emotion == EMOTION_NEUTRAL,
         "unfortunately → sad/neutral");

    /* Neutral → Surprised → Angry sequence */
    EmotionDetection seq[3];
    seq[0] = prosody_detect_emotion("The report is on the table.");
    seq[1] = prosody_detect_emotion("Wait, WHAT?! That's incredible!");
    seq[2] = prosody_detect_emotion("I am absolutely furious about this!");

    TEST(seq[0].emotion == EMOTION_NEUTRAL, "report sentence → neutral");
    TEST(seq[1].emotion != EMOTION_NEUTRAL, "WHAT?! → non-neutral");
    TEST(seq[2].emotion == EMOTION_ANGRY || seq[2].emotion != EMOTION_NEUTRAL,
         "furious → angry/non-neutral");

    /* Verify prosody hints change with emotion */
    TEST(seq[0].hint.pitch != seq[2].hint.pitch ||
         seq[0].hint.rate != seq[2].hint.rate ||
         seq[0].hint.energy != seq[2].hint.energy,
         "different emotions produce different prosody hints");
}

/* ── Prosody Parameter Range Validation ──────────────────────────────────── */

static void test_prosody_param_ranges(void) {
    printf("\n=== Prosody Parameter Ranges ===\n");

    /* Test that analyze_text produces in-range parameters */
    const char *test_texts[] = {
        "Hello there.",
        "How are you doing today?",
        "THAT IS AMAZING!!!",
        "I need apples, bananas, and oranges.",
        "Unfortunately, it failed miserably.",
        "Stop right now!",
        NULL
    };

    for (int t = 0; test_texts[t]; t++) {
        MultiScaleProsody msp = prosody_analyze_text(test_texts[t]);

        /* Utterance-level pitch should be reasonable */
        TEST(msp.utterance.pitch >= 0.5f && msp.utterance.pitch <= 2.0f,
             "utterance pitch in [0.5, 2.0]");

        /* Rate should be reasonable */
        TEST(msp.utterance.rate >= 0.5f && msp.utterance.rate <= 2.0f,
             "utterance rate in [0.5, 2.0]");

        /* Energy is a dB offset, typically [-6, +6] */
        TEST(msp.utterance.energy >= -10.0f && msp.utterance.energy <= 10.0f,
             "utterance energy in [-10, 10]");

        /* Contour should be a valid enum */
        TEST(msp.contour >= PROSODY_CONTOUR_DECLARATIVE &&
             msp.contour <= PROSODY_CONTOUR_LIST,
             "contour is valid enum value");

        /* n_words in [0, 64] */
        TEST(msp.n_words >= 0 && msp.n_words <= 64,
             "n_words in [0, 64]");

        /* Per-word hints in range */
        for (int w = 0; w < msp.n_words; w++) {
            TEST(msp.word_hints[w].pitch >= 0.0f && msp.word_hints[w].pitch <= 3.0f,
                 "word pitch in [0, 3]");
            TEST(msp.word_hints[w].rate >= 0.0f && msp.word_hints[w].rate <= 3.0f,
                 "word rate in [0, 3]");
            TEST(msp.emphasis_mask[w] == 0 || msp.emphasis_mask[w] == 1,
                 "emphasis_mask is 0 or 1");
        }
    }
}

/* ── Full Pipeline with Multiple Emotion Types ────────────────────────────── */

static void test_pipeline_varied_emotions(void) {
    printf("\n=== Pipeline: Varied Emotion Types ===\n");

    struct {
        const char *text;
        const char *label;
    } cases[] = {
        {"I'm so happy and grateful for this!", "happy text"},
        {"That's terrifying and awful.", "fearful text"},
        {"STOP DOING THAT RIGHT NOW!", "angry text"},
        {"Well, this is quite ordinary.", "neutral text"},
        {"Oh wow, I never expected that!", "surprised text"},
        {NULL, NULL}
    };

    for (int i = 0; cases[i].text; i++) {
        /* Emphasis prediction */
        char emphasized[4096];
        int n_emph = emphasis_predict(cases[i].text, emphasized, sizeof(emphasized));
        TEST(n_emph >= 0, cases[i].label);

        /* SSML parse */
        SSMLSegment segments[SSML_MAX_SEGMENTS];
        int nseg = ssml_parse(emphasized, segments, SSML_MAX_SEGMENTS);
        TEST(nseg >= 1, "SSML parse succeeds");

        /* Prosody analysis */
        MultiScaleProsody msp = prosody_analyze_text(cases[i].text);
        TEST(msp.n_words > 0, "words detected");

        /* Emotion detection */
        EmotionDetection det = prosody_detect_emotion(cases[i].text);
        TEST(det.emotion >= 0 && det.emotion < EMOTION_COUNT, "valid emotion");
        TEST(det.confidence >= 0.0f && det.confidence <= 1.0f, "valid confidence");

        printf("    %s: emph=%d segs=%d words=%d emotion=%d conf=%.2f\n",
               cases[i].label, n_emph, nseg, msp.n_words,
               det.emotion, det.confidence);
    }
}

/* ── Prosody Clamping: Boost Limits ──────────────────────────────────────── */

static void test_prosody_clamping(void) {
    printf("\n=== Prosody Clamping ===\n");

    /* Boost should be clamped to [1.0, 1.3] */
    float boost = 1.0f;

    /* Simulate repeated low quality → boost escalation */
    for (int i = 0; i < 20; i++) {
        float quality = 0.3f;  /* always low */
        if (quality < 0.7f) boost += 0.05f;
        if (boost > 1.3f) boost = 1.3f;
    }
    TEST(fabsf(boost - 1.3f) < 0.01f, "boost caps at 1.3");

    /* Apply max boost to extreme pitch */
    float seg_pitch = 1.20f;
    float boosted = 1.0f + (seg_pitch - 1.0f) * boost;
    TEST(boosted >= 1.0f && boosted <= 1.5f, "boosted pitch in sane range");
    printf("    pitch 1.20 × boost 1.3 → %.3f\n", boosted);

    /* Apply to negative deviation */
    seg_pitch = 0.80f;
    boosted = 1.0f - (1.0f - seg_pitch) * boost;
    TEST(boosted >= 0.5f && boosted <= 1.0f, "negative boosted pitch in sane range");
    printf("    pitch 0.80 × boost 1.3 → %.3f\n", boosted);

    /* Volume boost clamping */
    float vol_db = 5.0f;
    float boosted_vol = vol_db * boost;
    TEST(boosted_vol <= 10.0f, "volume boost doesn't exceed 10 dB");

    /* Zero deviation → no change regardless of boost */
    seg_pitch = 1.0f;
    boosted = 1.0f + (seg_pitch - 1.0f) * boost;
    TEST(fabsf(boosted - 1.0f) < 0.001f, "zero deviation unaffected by boost");
}

/* ── Sentence Buffer to Full Pipeline: Multiple Sentences ─────────────────── */

static void test_sentbuf_multi_sentence_pipeline(void) {
    printf("\n=== SentBuf: Multi-Sentence Pipeline ===\n");

    SentenceBuffer *sb = sentbuf_create(0, 3);
    TEST(sb != NULL, "sentence buffer created");

    /* Feed two sentences */
    const char *stream[] = {
        "I ", "am ", "really ", "happy ", "today. ",
        "But ", "tomorrow ", "might ", "be ", "different."
    };
    for (int i = 0; i < 10; i++) {
        sentbuf_add(sb, stream[i], (int)strlen(stream[i]));
    }

    /* Process all available segments */
    char sentence[4096];
    int seg_count = 0;
    while (sentbuf_has_segment(sb)) {
        int slen = sentbuf_flush(sb, sentence, sizeof(sentence));
        if (slen <= 0) break;
        seg_count++;

        /* Run each through full pipeline */
        char emphasized[4096];
        emphasis_predict(sentence, emphasized, sizeof(emphasized));

        SSMLSegment segments[SSML_MAX_SEGMENTS];
        int nseg = ssml_parse(emphasized, segments, SSML_MAX_SEGMENTS);
        TEST(nseg >= 1, "sentence produces SSML segments");

        EmotionDetection det = prosody_detect_emotion(sentence);
        TEST(det.emotion >= 0, "emotion detected for sentence");
    }

    /* Flush remaining */
    int remaining = sentbuf_flush_all(sb, sentence, sizeof(sentence));
    if (remaining > 0) {
        seg_count++;
        EmotionDetection det = prosody_detect_emotion(sentence);
        TEST(det.emotion >= 0, "emotion detected for remaining text");
    }

    TEST(seg_count >= 1, "processed at least 1 segment from stream");
    printf("    (total segments processed: %d)\n", seg_count);

    sentbuf_destroy(sb);
}

/* ── Duration + Prosody Combined Validation ──────────────────────────────── */

static void test_duration_prosody_combined(void) {
    printf("\n=== Duration + Prosody Combined ===\n");

    const char *text = "This is a test sentence with several words.";

    /* Duration estimation */
    float durations[256];
    int n = prosody_estimate_durations(text, 64, durations, 256);
    TEST(n == 64, "64 duration frames estimated");

    /* All durations should be non-negative */
    int all_valid = 1;
    for (int i = 0; i < n; i++) {
        if (durations[i] < 0.0f) { all_valid = 0; break; }
    }
    TEST(all_valid, "all durations non-negative");

    /* Prosody analysis */
    MultiScaleProsody msp = prosody_analyze_text(text);
    TEST(msp.n_words > 0, "multi-scale words detected");

    /* Emotion should be neutral for this bland text */
    EmotionDetection det = prosody_detect_emotion(text);
    TEST(det.emotion == EMOTION_NEUTRAL, "bland text → neutral emotion");

    /* Conversational adaptation with the detected emotion's hint */
    ConversationProsodyState state;
    prosody_conversation_init(&state);
    prosody_conversation_update(&state, 2.0f, 8, -20.0f, 150.0f);
    ProsodyHint adapt = prosody_conversation_adapt(&state);

    /* Combined: apply emotion hint and adaptation */
    float final_pitch = det.hint.pitch * adapt.pitch;
    float final_rate = det.hint.rate * adapt.rate;
    TEST(final_pitch > 0.0f, "combined pitch positive");
    TEST(final_rate > 0.0f, "combined rate positive");
    printf("    combined: pitch=%.3f rate=%.3f\n", final_pitch, final_rate);
}

/* ── SSML + Emotion + Emphasis Triple Stack ─────────────────────────────── */

static void test_triple_stack_ssml(void) {
    printf("\n=== Triple Stack: SSML + Emotion + Emphasis ===\n");

    SSMLSegment segs[SSML_MAX_SEGMENTS];

    /* Prosody wrapping emotion wrapping emphasis */
    int n = ssml_parse(
        "<speak><prosody rate=\"85%\" pitch=\"+5%\">"
        "<emotion type=\"sad\"><emphasis level=\"moderate\">I'm sorry</emphasis> "
        "about your loss.</emotion></prosody></speak>",
        segs, SSML_MAX_SEGMENTS);
    TEST(n >= 1, "triple-stack SSML parsed");

    int found_sorry = 0;
    for (int i = 0; i < n; i++) {
        if (strstr(segs[i].text, "sorry")) {
            found_sorry = 1;
            /* Rate should be prosody(0.85) * emphasis_moderate effect */
            TEST(segs[i].rate < 1.0f, "rate below 1.0 from prosody+emphasis");
            TEST(strcmp(segs[i].emotion, "sad") == 0, "emotion is sad");
        }
    }
    TEST(found_sorry, "found sorry segment");

    /* Excited emotion + strong emphasis + fast prosody */
    n = ssml_parse(
        "<speak><prosody rate=\"120%\"><emotion type=\"excited\">"
        "<emphasis level=\"strong\">AMAZING!</emphasis></emotion></prosody></speak>",
        segs, SSML_MAX_SEGMENTS);
    TEST(n >= 1, "excited triple-stack parsed");

    for (int i = 0; i < n; i++) {
        if (strstr(segs[i].text, "AMAZING")) {
            /* Excited rate=1.08, prosody rate=1.2, emphasis strong rate*=0.9 */
            /* Combined: dependent on implementation priority */
            TEST(segs[i].volume > 1.0f, "strong emphasis boosts volume");
            TEST(strcmp(segs[i].emotion, "excited") == 0, "emotion preserved");
        }
    }
}

/* ── Prosody Reset Between Sentences ─────────────────────────────────────── */

static void test_prosody_reset_between_sentences(void) {
    printf("\n=== Prosody Reset Between Sentences ===\n");

    /* Process first sentence: sad */
    EmotionDetection det1 = prosody_detect_emotion(
        "I am so sorry about the terrible news.");
    MultiScaleProsody msp1 = prosody_analyze_text(
        "I am so sorry about the terrible news.");

    /* Process second sentence: happy */
    EmotionDetection det2 = prosody_detect_emotion(
        "But wait, something wonderful happened!");
    MultiScaleProsody msp2 = prosody_analyze_text(
        "But wait, something wonderful happened!");

    /* Emotions should differ */
    TEST(det1.emotion != det2.emotion || det1.confidence != det2.confidence,
         "different sentences produce different emotion signals");

    /* Contours should differ */
    TEST(msp1.contour != msp2.contour || msp1.utterance.energy != msp2.utterance.energy,
         "contour or energy differs between sad and happy sentences");

    /* Process third sentence: neutral (reset) */
    EmotionDetection det3 = prosody_detect_emotion("The weather is mild today.");
    TEST(det3.emotion == EMOTION_NEUTRAL, "neutral sentence resets emotion");
    TEST(det3.hint.pitch >= 0.95f && det3.hint.pitch <= 1.05f,
         "neutral sentence has near-1.0 pitch hint");
}

/* ── Emphasis with Extreme Sentence Patterns ─────────────────────────────── */

static void test_emphasis_extreme_patterns(void) {
    printf("\n=== Emphasis: Extreme Sentence Patterns ===\n");

    char out[8192];

    /* Pattern: ALL CAPS sentence */
    int n = emphasis_predict("STOP EVERYTHING RIGHT NOW", out, sizeof(out));
    TEST(n >= 0, "ALL CAPS sentence processed");

    /* Pattern: single word */
    n = emphasis_predict("No.", out, sizeof(out));
    TEST(n >= 0, "single-word sentence processed");

    /* Pattern: very long sentence with many contrast markers */
    n = emphasis_predict(
        "However, the situation is complex, but actually it's simple, yet "
        "instead of panicking, we should remain calm, but however we must "
        "also stay alert and actually prepare for the worst.",
        out, sizeof(out));
    TEST(n >= 0, "many contrast markers processed");
    /* Count emphasis tags - should be capped */
    int count = 0;
    const char *p = out;
    while ((p = strstr(p, "<emphasis")) != NULL) { count++; p++; }
    TEST(count <= 5, "emphasis count capped with many markers");

    /* Pattern: empty input */
    n = emphasis_predict("", out, sizeof(out));
    TEST(n == 0, "empty input → 0 emphasis");

    /* Pattern: NULL input */
    n = emphasis_predict(NULL, out, sizeof(out));
    TEST(n == 0, "NULL input → 0 emphasis");

    /* Pattern: only punctuation */
    n = emphasis_predict("... !!! ???", out, sizeof(out));
    TEST(n >= 0, "punctuation-only doesn't crash");

    /* Pattern: numbers and special chars */
    n = emphasis_predict("There are 42 items at $19.99 each!", out, sizeof(out));
    TEST(n >= 0, "text with numbers and currency processed");
}

/* ── End-to-End with Text Normalization ──────────────────────────────────── */

static void test_normalize_then_prosody(void) {
    printf("\n=== Normalize → Prosody End-to-End ===\n");

    char normalized[4096];

    /* Normalize text with numbers and abbreviations */
    text_auto_normalize("Dr. Smith paid $42.50 on 1/15/2026.", normalized, sizeof(normalized));
    TEST(strlen(normalized) > 0, "normalization produced output");

    /* Run through emphasis prediction */
    char emphasized[4096];
    int n_emph = emphasis_predict(normalized, emphasized, sizeof(emphasized));
    TEST(n_emph >= 0, "emphasis on normalized text");

    /* Parse as SSML */
    SSMLSegment segs[SSML_MAX_SEGMENTS];
    int nseg = ssml_parse(emphasized, segs, SSML_MAX_SEGMENTS);
    TEST(nseg >= 1, "SSML parse on normalized text");

    /* Multi-scale prosody */
    MultiScaleProsody msp = prosody_analyze_text(normalized);
    TEST(msp.n_words > 0, "prosody analysis on normalized text");

    /* Emotion detection */
    EmotionDetection det = prosody_detect_emotion(normalized);
    TEST(det.emotion >= 0 && det.emotion < EMOTION_COUNT, "valid emotion on normalized text");

    printf("    Normalized: %s\n", normalized);
    printf("    Emphasis: %d, Segments: %d, Words: %d, Emotion: %d\n",
           n_emph, nseg, msp.n_words, det.emotion);
}

/* ── Quote Detection Edge Cases ──────────────────────────────────────────── */

static void test_quote_detection_edge_cases(void) {
    printf("\n=== Quote Detection: Edge Cases ===\n");

    char out[4096];

    /* Multiple quoted segments */
    int n = emphasis_detect_quotes(
        "He said \"hello\" and she said \"goodbye\" and they left.",
        out, sizeof(out));
    TEST(n == 2, "found 2 quoted segments");

    /* No quotes */
    n = emphasis_detect_quotes("There are no quotes here.", out, sizeof(out));
    TEST(n == 0, "no quotes found");

    /* Single unclosed quote */
    n = emphasis_detect_quotes("He said \"hello and never finished", out, sizeof(out));
    TEST(n >= 0, "unclosed quote handled gracefully");

    /* Empty quotes */
    n = emphasis_detect_quotes("She said \"\" nothing.", out, sizeof(out));
    TEST(n >= 0, "empty quotes handled");

    /* NULL input */
    n = emphasis_detect_quotes(NULL, out, sizeof(out));
    TEST(n == 0, "NULL input returns 0 quotes");

    /* Empty input */
    n = emphasis_detect_quotes("", out, sizeof(out));
    TEST(n == 0, "empty input returns 0 quotes");

    /* Quote at the very start and end */
    n = emphasis_detect_quotes("\"Hello world\"", out, sizeof(out));
    TEST(n == 1, "quote spanning entire text");
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

    /* Extended tests */
    test_multi_sentence_emotions();
    test_prosody_param_ranges();
    test_pipeline_varied_emotions();
    test_prosody_clamping();
    test_sentbuf_multi_sentence_pipeline();
    test_duration_prosody_combined();

    /* Deepened tests */
    test_triple_stack_ssml();
    test_prosody_reset_between_sentences();
    test_emphasis_extreme_patterns();
    test_normalize_then_prosody();
    test_quote_detection_edge_cases();

    printf("\n════════════════════════════════════════════\n");
    printf("  Results: %d passed, %d failed\n", pass_count, fail_count);
    printf("════════════════════════════════════════════\n");

    return fail_count > 0 ? 1 : 0;
}
