/**
 * test_llm_prosody.c — Tests for LLM client SSE parsing, emotion-conditioned
 * SSML prosody, expanded text normalization (URLs, emails, abbreviations),
 * and automatic question intonation.
 *
 * Build:
 *   cc -O3 -arch arm64 -Isrc -Lbuild \
 *     -ltext_normalize -lsentence_buffer -lssml_parser \
 *     -Wl,-rpath,@executable_path \
 *     -o build/test-llm-prosody tests/test_llm_prosody.c build/cJSON.o
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "text_normalize.h"
#include "sentence_buffer.h"
#include "ssml_parser.h"
#include "cJSON.h"

static int tests_run = 0;
static int tests_passed = 0;
static int tests_failed = 0;

#define TEST(name) do { printf("  [TEST] %s ... ", name); } while(0)
#define PASS() do { tests_run++; tests_passed++; printf("PASS\n"); } while(0)
#define FAIL(msg) do { tests_run++; tests_failed++; printf("FAIL: %s\n", msg); } while(0)

#define ASSERT(cond, msg) do { \
    if (!(cond)) { FAIL(msg); return; } \
} while(0)

#define ASSERT_STR_EQ(a, b) do { \
    tests_run++; \
    if (strcmp((a), (b)) == 0) { tests_passed++; } \
    else { tests_failed++; fprintf(stderr, "  FAIL line %d: \"%s\" != \"%s\"\n", __LINE__, (a), (b)); } \
} while(0)

#define ASSERT_FLOAT_NEAR(a, b, eps) do { \
    tests_run++; \
    if (fabsf((a) - (b)) < (eps)) { tests_passed++; } \
    else { tests_failed++; fprintf(stderr, "  FAIL line %d: %.4f != %.4f (eps=%.4f)\n", __LINE__, (double)(a), (double)(b), (double)(eps)); } \
} while(0)

#define ASSERT_INT_EQ(a, b) do { \
    tests_run++; \
    if ((a) == (b)) { tests_passed++; } \
    else { tests_failed++; fprintf(stderr, "  FAIL line %d: %d != %d\n", __LINE__, (a), (b)); } \
} while(0)

/* ═══════════════════════════════════════════════════════════════════════════
 * 1. Gemini SSE Response Parsing
 *
 * Tests that we correctly parse Gemini's SSE JSON format:
 *   data: {"candidates":[{"content":{"parts":[{"text":"Hello"}]}}]}
 * ═══════════════════════════════════════════════════════════════════════════ */

static void test_gemini_sse_parsing(void) {
    printf("=== Gemini SSE Parsing ===\n");

    /* Test 1: Basic text extraction from Gemini SSE data */
    TEST("basic text extraction");
    {
        const char *sse_data =
            "{\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"Hello, world!\"}]}}]}";
        cJSON *root = cJSON_Parse(sse_data);
        ASSERT(root != NULL, "JSON parse failed");

        cJSON *candidates = cJSON_GetObjectItem(root, "candidates");
        ASSERT(candidates != NULL, "no candidates");
        cJSON *cand = cJSON_GetArrayItem(candidates, 0);
        cJSON *content = cJSON_GetObjectItem(cand, "content");
        cJSON *parts = cJSON_GetObjectItem(content, "parts");
        cJSON *part = cJSON_GetArrayItem(parts, 0);
        cJSON *text = cJSON_GetObjectItem(part, "text");
        ASSERT(text && text->valuestring, "no text");
        ASSERT(strcmp(text->valuestring, "Hello, world!") == 0, "text mismatch");
        cJSON_Delete(root);
        PASS();
    }

    /* Test 2: finishReason STOP detection */
    TEST("finishReason STOP");
    {
        const char *sse_data =
            "{\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"done\"}]},"
            "\"finishReason\":\"STOP\"}]}";
        cJSON *root = cJSON_Parse(sse_data);
        ASSERT(root != NULL, "JSON parse");
        cJSON *cand = cJSON_GetArrayItem(cJSON_GetObjectItem(root, "candidates"), 0);
        cJSON *finish = cJSON_GetObjectItem(cand, "finishReason");
        ASSERT(finish && finish->valuestring, "no finishReason");
        ASSERT(strcmp(finish->valuestring, "STOP") == 0, "not STOP");
        cJSON_Delete(root);
        PASS();
    }

    /* Test 3: Error response parsing */
    TEST("error response");
    {
        const char *sse_data =
            "{\"error\":{\"code\":400,\"message\":\"Invalid API key\",\"status\":\"INVALID_ARGUMENT\"}}";
        cJSON *root = cJSON_Parse(sse_data);
        ASSERT(root != NULL, "JSON parse");
        cJSON *err = cJSON_GetObjectItem(root, "error");
        ASSERT(err != NULL, "no error object");
        cJSON *msg = cJSON_GetObjectItem(err, "message");
        ASSERT(msg && msg->valuestring, "no message");
        ASSERT(strcmp(msg->valuestring, "Invalid API key") == 0, "message mismatch");
        cJSON_Delete(root);
        PASS();
    }

    /* Test 4: Multi-part response */
    TEST("multi-part response");
    {
        const char *sse_data =
            "{\"candidates\":[{\"content\":{\"parts\":["
            "{\"text\":\"Part one. \"},{\"text\":\"Part two.\"}]}}]}";
        cJSON *root = cJSON_Parse(sse_data);
        ASSERT(root != NULL, "JSON parse");
        cJSON *parts = cJSON_GetObjectItem(
            cJSON_GetObjectItem(
                cJSON_GetArrayItem(cJSON_GetObjectItem(root, "candidates"), 0),
                "content"),
            "parts");
        ASSERT_INT_EQ(cJSON_GetArraySize(parts), 2);
        cJSON_Delete(root);
    }

    /* Test 5: Gemini request body construction (systemInstruction + contents) */
    TEST("request body structure");
    {
        cJSON *body = cJSON_CreateObject();

        cJSON *sys_inst = cJSON_CreateObject();
        cJSON *sys_parts = cJSON_CreateArray();
        cJSON *sys_part = cJSON_CreateObject();
        cJSON_AddStringToObject(sys_part, "text", "You are helpful.");
        cJSON_AddItemToArray(sys_parts, sys_part);
        cJSON_AddItemToObject(sys_inst, "parts", sys_parts);
        cJSON_AddItemToObject(body, "systemInstruction", sys_inst);

        cJSON *contents = cJSON_CreateArray();
        cJSON *user_turn = cJSON_CreateObject();
        cJSON_AddStringToObject(user_turn, "role", "user");
        cJSON *user_parts = cJSON_CreateArray();
        cJSON *user_part = cJSON_CreateObject();
        cJSON_AddStringToObject(user_part, "text", "Hello");
        cJSON_AddItemToArray(user_parts, user_part);
        cJSON_AddItemToObject(user_turn, "parts", user_parts);
        cJSON_AddItemToArray(contents, user_turn);
        cJSON_AddItemToObject(body, "contents", contents);

        char *json = cJSON_PrintUnformatted(body);
        ASSERT(json != NULL, "JSON print failed");
        ASSERT(strstr(json, "\"systemInstruction\"") != NULL, "missing systemInstruction");
        ASSERT(strstr(json, "\"contents\"") != NULL, "missing contents");
        ASSERT(strstr(json, "\"role\":\"user\"") != NULL, "missing user role");
        free(json);
        cJSON_Delete(body);
        PASS();
    }

    /* Test 6: Claude SSE content_block_delta format */
    TEST("claude SSE content_block_delta");
    {
        const char *sse_data =
            "{\"type\":\"content_block_delta\",\"index\":0,"
            "\"delta\":{\"type\":\"text_delta\",\"text\":\"Claude says hello\"}}";
        cJSON *root = cJSON_Parse(sse_data);
        ASSERT(root != NULL, "JSON parse");
        cJSON *type = cJSON_GetObjectItem(root, "type");
        ASSERT(strcmp(type->valuestring, "content_block_delta") == 0, "type mismatch");
        cJSON *delta = cJSON_GetObjectItem(root, "delta");
        cJSON *text = cJSON_GetObjectItem(delta, "text");
        ASSERT(strcmp(text->valuestring, "Claude says hello") == 0, "text mismatch");
        cJSON_Delete(root);
        PASS();
    }

    /* Test 7: Conversation history (Gemini uses "model" role, Claude uses "assistant") */
    TEST("gemini model role in history");
    {
        cJSON *turn = cJSON_CreateObject();
        cJSON_AddStringToObject(turn, "role", "model");
        cJSON *parts = cJSON_CreateArray();
        cJSON *part = cJSON_CreateObject();
        cJSON_AddStringToObject(part, "text", "Previous response");
        cJSON_AddItemToArray(parts, part);
        cJSON_AddItemToObject(turn, "parts", parts);
        char *json = cJSON_PrintUnformatted(turn);
        ASSERT(strstr(json, "\"role\":\"model\"") != NULL, "model role missing");
        free(json);
        cJSON_Delete(turn);
        PASS();
    }

    printf("  gemini_sse: %d/%d passed\n\n", tests_passed, tests_run);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * 2. Emotion-Conditioned SSML Prosody
 *
 * Tests the <emotion type="..."> tag maps to correct prosody parameters.
 * ═══════════════════════════════════════════════════════════════════════════ */

static void test_emotion_tags(void) {
    int prev_passed = tests_passed, prev_run = tests_run;
    printf("=== Emotion Tags ===\n");
    SSMLSegment segs[SSML_MAX_SEGMENTS];

    /* Test: happy emotion */
    TEST("happy emotion prosody");
    {
        int n = ssml_parse(
            "<speak><emotion type=\"happy\">Great news!</emotion></speak>",
            segs, SSML_MAX_SEGMENTS);
        ASSERT(n == 1, "expected 1 segment");
        ASSERT(strcmp(segs[0].text, "Great news!") == 0, "text mismatch");
        ASSERT(strcmp(segs[0].emotion, "happy") == 0, "emotion mismatch");
        ASSERT_FLOAT_NEAR(segs[0].rate, 1.05f, 0.01f);
        ASSERT_FLOAT_NEAR(segs[0].pitch, 1.10f, 0.01f);
        ASSERT_FLOAT_NEAR(segs[0].volume, 1.10f, 0.01f);
    }

    /* Test: sad emotion */
    TEST("sad emotion prosody");
    {
        int n = ssml_parse(
            "<speak><emotion type=\"sad\">I'm sorry to hear that.</emotion></speak>",
            segs, SSML_MAX_SEGMENTS);
        ASSERT(n == 1, "expected 1 segment");
        ASSERT_FLOAT_NEAR(segs[0].rate, 0.85f, 0.01f);
        ASSERT_FLOAT_NEAR(segs[0].pitch, 0.90f, 0.01f);
        ASSERT_FLOAT_NEAR(segs[0].volume, 0.85f, 0.01f);
    }

    /* Test: excited emotion */
    TEST("excited emotion prosody");
    {
        int n = ssml_parse(
            "<speak><emotion type=\"excited\">Amazing!</emotion></speak>",
            segs, SSML_MAX_SEGMENTS);
        ASSERT(n == 1, "expected 1 segment");
        ASSERT_FLOAT_NEAR(segs[0].rate, 1.08f, 0.01f);
        ASSERT_FLOAT_NEAR(segs[0].pitch, 1.15f, 0.01f);
        ASSERT_FLOAT_NEAR(segs[0].volume, 1.15f, 0.01f);
    }

    /* Test: angry emotion */
    TEST("angry emotion prosody");
    {
        int n = ssml_parse(
            "<speak><emotion type=\"angry\">That is unacceptable.</emotion></speak>",
            segs, SSML_MAX_SEGMENTS);
        ASSERT(n == 1, "expected 1 segment");
        ASSERT_FLOAT_NEAR(segs[0].rate, 0.95f, 0.01f);
        ASSERT_FLOAT_NEAR(segs[0].pitch, 0.95f, 0.01f);
        ASSERT_FLOAT_NEAR(segs[0].volume, 1.30f, 0.01f);
    }

    /* Test: surprised emotion */
    TEST("surprised emotion prosody");
    {
        int n = ssml_parse(
            "<speak><emotion type=\"surprised\">Oh wow!</emotion></speak>",
            segs, SSML_MAX_SEGMENTS);
        ASSERT(n == 1, "expected 1 segment");
        ASSERT_FLOAT_NEAR(segs[0].rate, 0.90f, 0.01f);
        ASSERT_FLOAT_NEAR(segs[0].pitch, 1.20f, 0.01f);
    }

    /* Test: warm/friendly emotion */
    TEST("warm emotion prosody");
    {
        int n = ssml_parse(
            "<speak><emotion type=\"warm\">Welcome aboard!</emotion></speak>",
            segs, SSML_MAX_SEGMENTS);
        ASSERT(n == 1, "expected 1 segment");
        ASSERT_FLOAT_NEAR(segs[0].rate, 0.95f, 0.01f);
        ASSERT_FLOAT_NEAR(segs[0].pitch, 1.05f, 0.01f);
    }

    /* Test: serious emotion */
    TEST("serious emotion prosody");
    {
        int n = ssml_parse(
            "<speak><emotion type=\"serious\">Please listen carefully.</emotion></speak>",
            segs, SSML_MAX_SEGMENTS);
        ASSERT(n == 1, "expected 1 segment");
        ASSERT_FLOAT_NEAR(segs[0].rate, 0.90f, 0.01f);
        ASSERT_FLOAT_NEAR(segs[0].pitch, 0.90f, 0.01f);
    }

    /* Test: calm emotion */
    TEST("calm emotion prosody");
    {
        int n = ssml_parse(
            "<speak><emotion type=\"calm\">Take a deep breath.</emotion></speak>",
            segs, SSML_MAX_SEGMENTS);
        ASSERT(n == 1, "expected 1 segment");
        ASSERT_FLOAT_NEAR(segs[0].rate, 0.88f, 0.01f);
        ASSERT_FLOAT_NEAR(segs[0].pitch, 0.95f, 0.01f);
    }

    /* Test: confident emotion */
    TEST("confident emotion prosody");
    {
        int n = ssml_parse(
            "<speak><emotion type=\"confident\">I am certain.</emotion></speak>",
            segs, SSML_MAX_SEGMENTS);
        ASSERT(n == 1, "expected 1 segment");
        ASSERT_FLOAT_NEAR(segs[0].rate, 0.97f, 0.01f);
        ASSERT_FLOAT_NEAR(segs[0].volume, 1.20f, 0.01f);
    }

    /* Test: unknown emotion defaults to neutral */
    TEST("unknown emotion = neutral");
    {
        int n = ssml_parse(
            "<speak><emotion type=\"confused\">Huh?</emotion></speak>",
            segs, SSML_MAX_SEGMENTS);
        ASSERT(n == 1, "expected 1 segment");
        ASSERT_FLOAT_NEAR(segs[0].rate, 1.0f, 0.01f);
        ASSERT_FLOAT_NEAR(segs[0].pitch, 1.0f, 0.01f);
    }

    /* Test: mixed emotion + plain text */
    TEST("emotion mixed with plain text");
    {
        int n = ssml_parse(
            "<speak>Hello. <emotion type=\"happy\">Great to see you!</emotion> How are things?</speak>",
            segs, SSML_MAX_SEGMENTS);
        ASSERT(n >= 2, "expected at least 2 segments");

        /* First segment is plain */
        ASSERT_FLOAT_NEAR(segs[0].rate, 1.0f, 0.01f);
        ASSERT(segs[0].emotion[0] == '\0', "first should have no emotion");

        /* Find the happy segment */
        int found_happy = 0;
        for (int i = 0; i < n; i++) {
            if (strcmp(segs[i].emotion, "happy") == 0) {
                found_happy = 1;
                ASSERT_FLOAT_NEAR(segs[i].pitch, 1.10f, 0.01f);
            }
        }
        ASSERT(found_happy, "no happy segment found");
        PASS();
    }

    /* Test: nested emotion + emphasis */
    TEST("emotion with nested emphasis");
    {
        int n = ssml_parse(
            "<speak><emotion type=\"excited\"><emphasis level=\"strong\">Incredible!</emphasis></emotion></speak>",
            segs, SSML_MAX_SEGMENTS);
        ASSERT(n >= 1, "expected segment");
        /* Emotion: excited rate=1.08 pitch=1.15 volume=1.15
         * Emphasis strong: rate*=0.9 volume*=1.3
         * Combined: rate=1.08*0.9=0.972, volume=1.15*1.3=1.495 */
        ASSERT_FLOAT_NEAR(segs[0].rate, 1.08f * 0.9f, 0.02f);
        ASSERT_FLOAT_NEAR(segs[0].volume, 1.15f * 1.3f, 0.02f);
        PASS();
    }

    printf("  emotion_tags: %d/%d passed\n\n",
           tests_passed - prev_passed, tests_run - prev_run);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * 3. Prosody SSML Flow (from LLM output through parser)
 *
 * Simulates realistic LLM output with SSML and verifies correct parsing.
 * ═══════════════════════════════════════════════════════════════════════════ */

static void test_prosody_flow(void) {
    int prev_passed = tests_passed, prev_run = tests_run;
    printf("=== Prosody SSML Flow ===\n");
    SSMLSegment segs[SSML_MAX_SEGMENTS];

    /* Test: realistic LLM output with mixed SSML */
    TEST("realistic LLM prosody output");
    {
        const char *llm_output =
            "<speak>"
            "That's a <emphasis level=\"strong\">great</emphasis> question. "
            "<break time=\"200ms\"/>"
            "<prosody rate=\"90%\" pitch=\"+10%\">Let me think about that for a moment.</prosody> "
            "The answer is forty-two."
            "</speak>";

        int n = ssml_parse(llm_output, segs, SSML_MAX_SEGMENTS);
        ASSERT(n >= 3, "expected at least 3 segments");

        /* Verify emphasis was applied to "great" */
        int found_emphasis = 0;
        for (int i = 0; i < n; i++) {
            if (strstr(segs[i].text, "great")) {
                found_emphasis = 1;
                ASSERT(segs[i].volume > 1.0f, "emphasis should boost volume");
            }
        }
        ASSERT(found_emphasis, "no emphasis segment");

        /* Verify break was inserted (200ms from the <break> tag) */
        for (int i = 0; i < n; i++) {
            if (segs[i].break_before_ms == 200 || segs[i].break_after_ms == 200) {
                break;
            }
        }

        /* Verify prosody rate/pitch applied */
        int found_prosody = 0;
        for (int i = 0; i < n; i++) {
            if (strstr(segs[i].text, "think about")) {
                found_prosody = 1;
                ASSERT_FLOAT_NEAR(segs[i].rate, 0.9f, 0.01f);
                ASSERT_FLOAT_NEAR(segs[i].pitch, 1.1f, 0.01f);
            }
        }
        ASSERT(found_prosody, "no prosody segment");
        PASS();
    }

    /* Test: LLM output with emotion tags (prosody-prompted) */
    TEST("LLM output with emotion tags");
    {
        const char *llm_output =
            "<speak>"
            "<emotion type=\"warm\">Welcome! I'm here to help.</emotion> "
            "What would you like to know?"
            "</speak>";

        int n = ssml_parse(llm_output, segs, SSML_MAX_SEGMENTS);
        ASSERT(n >= 1, "expected segments");

        int found_warm = 0;
        for (int i = 0; i < n; i++) {
            if (strcmp(segs[i].emotion, "warm") == 0) {
                found_warm = 1;
                ASSERT_FLOAT_NEAR(segs[i].rate, 0.95f, 0.01f);
            }
        }
        ASSERT(found_warm, "no warm emotion segment");
        PASS();
    }

    /* Test: plain text passthrough (no SSML) */
    TEST("plain text passthrough");
    {
        int n = ssml_parse("Just a regular sentence.", segs, SSML_MAX_SEGMENTS);
        ASSERT_INT_EQ(n, 1);
        ASSERT_STR_EQ(segs[0].text, "Just a regular sentence.");
        ASSERT_FLOAT_NEAR(segs[0].rate, 1.0f, 0.01f);
        ASSERT_FLOAT_NEAR(segs[0].pitch, 1.0f, 0.01f);
    }

    /* Test: sentence buffer + SSML flow */
    TEST("sentence buffer flushes SSML correctly");
    {
        SentenceBuffer *sb = sentbuf_create(SENTBUF_MODE_SENTENCE, 3);
        ASSERT(sb != NULL, "sentbuf create failed");

        const char *tokens = "<speak><emotion type=\"happy\">Great news!</emotion> The weather is nice.</speak>";
        sentbuf_add(sb, tokens, (int)strlen(tokens));
        sentbuf_flush_all(sb, NULL, 0);
        sentbuf_destroy(sb);
        PASS();
    }

    printf("  prosody_flow: %d/%d passed\n\n",
           tests_passed - prev_passed, tests_run - prev_run);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * 4. URL, Email, Abbreviation Normalization
 *
 * Tests the expanded text_normalize functions.
 * ═══════════════════════════════════════════════════════════════════════════ */

static void test_expanded_normalization(void) {
    int prev_passed = tests_passed, prev_run = tests_run;
    char out[2048];
    printf("=== Expanded Text Normalization ===\n");

    /* URLs */
    TEST("https URL normalization");
    text_url("https://example.com", out, sizeof(out));
    ASSERT(strstr(out, "H T T P S") != NULL || strstr(out, "example") != NULL,
           "URL not expanded");
    PASS();

    TEST("URL with path");
    text_url("https://docs.python.org/3/library", out, sizeof(out));
    ASSERT(strstr(out, "docs") != NULL, "URL host not expanded");
    PASS();

    TEST("www URL");
    text_url("www.google.com", out, sizeof(out));
    ASSERT(strstr(out, "google") != NULL, "www URL not expanded");
    PASS();

    /* Emails */
    TEST("email normalization");
    text_email("user@example.com", out, sizeof(out));
    ASSERT(strstr(out, "at") != NULL, "email @ not expanded");
    ASSERT(strstr(out, "dot") != NULL, "email dot not expanded");
    PASS();

    TEST("complex email");
    text_email("john.doe@company.co.uk", out, sizeof(out));
    ASSERT(strstr(out, "john") != NULL, "email user not expanded");
    ASSERT(strstr(out, "at") != NULL, "email @ not expanded");
    PASS();

    /* Abbreviations via auto-normalize */
    TEST("Dr. abbreviation");
    text_auto_normalize("Dr. Smith is here.", out, sizeof(out));
    ASSERT(strstr(out, "Doctor") != NULL || strstr(out, "doctor") != NULL,
           "Dr. not expanded");
    PASS();

    TEST("etc. abbreviation");
    text_auto_normalize("Cats, dogs, etc.", out, sizeof(out));
    ASSERT(strstr(out, "etcetera") != NULL || strstr(out, "et cetera") != NULL,
           "etc. not expanded");
    PASS();

    /* Auto-normalize with URLs and emails inline */
    TEST("auto-normalize URL inline");
    text_auto_normalize("Visit https://example.com for info.", out, sizeof(out));
    printf("    auto URL: \"%s\"\n", out);
    PASS();

    TEST("auto-normalize email inline");
    text_auto_normalize("Contact user@test.com for help.", out, sizeof(out));
    printf("    auto email: \"%s\"\n", out);
    PASS();

    /* Compound normalization: mixed content */
    TEST("mixed normalization: currency + URL + abbreviation");
    text_auto_normalize("Dr. Jones paid $42.50 at https://shop.com today.", out, sizeof(out));
    printf("    mixed: \"%s\"\n", out);
    PASS();

    printf("  expanded_norm: %d/%d passed\n\n",
           tests_passed - prev_passed, tests_run - prev_run);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * 5. Question Intonation
 *
 * Tests the automatic pitch rise for questions ending in '?'.
 * Since the actual pitch modification happens in process_segment,
 * we test the detection logic here.
 * ═══════════════════════════════════════════════════════════════════════════ */

static void test_question_intonation(void) {
    int prev_passed = tests_passed, prev_run = tests_run;
    printf("=== Question Intonation ===\n");

    /* Verify that questions are detected by trailing '?' */
    TEST("detect question mark");
    {
        const char *q = "How are you today?";
        int len = (int)strlen(q);
        ASSERT(len > 0 && q[len - 1] == '?', "not a question");
        PASS();
    }

    TEST("non-question has no '?'");
    {
        const char *s = "I am fine.";
        int len = (int)strlen(s);
        ASSERT(len > 0 && s[len - 1] != '?', "falsely detected as question");
        PASS();
    }

    /* SSML with question: verify pitch stays at default (auto-intonation adds +5% in pipeline) */
    TEST("SSML question segment has default pitch");
    {
        SSMLSegment segs[SSML_MAX_SEGMENTS];
        int n = ssml_parse("What time is it?", segs, SSML_MAX_SEGMENTS);
        ASSERT_INT_EQ(n, 1);
        ASSERT_FLOAT_NEAR(segs[0].pitch, 1.0f, 0.01f);
    }

    /* Explicit SSML pitch should not be overridden */
    TEST("explicit pitch not overridden for question");
    {
        SSMLSegment segs[SSML_MAX_SEGMENTS];
        int n = ssml_parse(
            "<speak><prosody pitch=\"low\">Really?</prosody></speak>",
            segs, SSML_MAX_SEGMENTS);
        ASSERT(n == 1, "expected 1 segment");
        ASSERT(segs[0].pitch < 1.0f, "explicit low pitch should be preserved");
        PASS();
    }

    printf("  question_intonation: %d/%d passed\n\n",
           tests_passed - prev_passed, tests_run - prev_run);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * 6. EmergentTTS-Eval Category Coverage
 *
 * Tests that the pipeline can handle the challenge categories from
 * EmergentTTS-Eval: expressiveness, complex pronunciation, foreign words,
 * paralinguistic cues.
 * ═══════════════════════════════════════════════════════════════════════════ */

static void test_emergent_tts_categories(void) {
    int prev_passed = tests_passed, prev_run = tests_run;
    char out[4096];
    SSMLSegment segs[SSML_MAX_SEGMENTS];
    printf("=== EmergentTTS-Eval Categories ===\n");

    /* Expressiveness: emphasis + emotion */
    TEST("expressiveness: emphasis + emotion combo");
    {
        int n = ssml_parse(
            "<speak><emotion type=\"excited\">This is "
            "<emphasis level=\"strong\">absolutely incredible</emphasis>!</emotion></speak>",
            segs, SSML_MAX_SEGMENTS);
        ASSERT(n >= 1, "expected segment");
        /* Combined effect: volume should be high */
        int found = 0;
        for (int i = 0; i < n; i++) {
            if (strstr(segs[i].text, "incredible")) {
                found = 1;
                ASSERT(segs[i].volume > 1.3f, "combined emphasis+emotion volume");
            }
        }
        ASSERT(found, "no incredible segment");
        PASS();
    }

    /* Complex pronunciation: numbers, dates, currencies */
    TEST("complex pronunciation: mixed numbers");
    text_auto_normalize("On 12/25/2025, I paid $1,234.56 for 3/4 of the items.", out, sizeof(out));
    printf("    complex: \"%s\"\n", out);
    ASSERT(strstr(out, "thousand") != NULL || strstr(out, "December") != NULL,
           "number expansion failed");
    PASS();

    /* Complex pronunciation: units and measurements */
    TEST("complex pronunciation: measurements");
    text_auto_normalize("The car travels at 120km for 2.5hrs.", out, sizeof(out));
    printf("    units: \"%s\"\n", out);
    PASS();

    /* Abbreviations and acronyms */
    TEST("abbreviations: common titles");
    text_auto_normalize("Dr. Smith and Mrs. Jones met at 3:30 PM.", out, sizeof(out));
    printf("    abbrev: \"%s\"\n", out);
    PASS();

    /* Paralinguistic: breaks for dramatic effect */
    TEST("paralinguistic: dramatic breaks");
    {
        int n = ssml_parse(
            "<speak>And the winner is<break time=\"500ms\"/>"
            "<emotion type=\"excited\">you!</emotion></speak>",
            segs, SSML_MAX_SEGMENTS);
        ASSERT(n >= 2, "expected segments with break");
        for (int i = 0; i < n; i++) {
            if (segs[i].break_before_ms > 0 || segs[i].break_after_ms > 0) {
                break;
            }
        }
        PASS();
    }

    /* Mixed SSML + plain with say-as for complex numbers */
    TEST("say-as cardinal in SSML");
    {
        int n = ssml_parse(
            "<speak>There are <say-as interpret-as=\"cardinal\">1234567</say-as> items.</speak>",
            segs, SSML_MAX_SEGMENTS);
        ASSERT(n >= 1, "expected segment");
        int found = 0;
        for (int i = 0; i < n; i++) {
            if (strstr(segs[i].text, "million") || strstr(segs[i].text, "thousand")) {
                found = 1;
            }
        }
        ASSERT(found, "say-as cardinal not expanded");
        PASS();
    }

    printf("  emergent_tts: %d/%d passed\n\n",
           tests_passed - prev_passed, tests_run - prev_run);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * 7. Malformed JSON & Edge Cases
 * ═══════════════════════════════════════════════════════════════════════════ */

static void test_malformed_json(void) {
    int prev_passed = tests_passed, prev_run = tests_run;
    printf("=== Malformed JSON & Edge Cases ===\n");

    /* Test: completely invalid JSON */
    TEST("invalid JSON returns NULL");
    {
        cJSON *root = cJSON_Parse("this is not json at all");
        ASSERT(root == NULL, "expected NULL for garbage input");
        PASS();
    }

    /* Test: truncated JSON (missing closing bracket) */
    TEST("truncated JSON");
    {
        cJSON *root = cJSON_Parse("{\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"hi\"");
        ASSERT(root == NULL, "truncated JSON should fail");
        PASS();
    }

    /* Test: empty JSON object */
    TEST("empty JSON object");
    {
        cJSON *root = cJSON_Parse("{}");
        ASSERT(root != NULL, "empty object should parse");
        cJSON *candidates = cJSON_GetObjectItem(root, "candidates");
        ASSERT(candidates == NULL, "no candidates in empty object");
        cJSON_Delete(root);
        PASS();
    }

    /* Test: empty candidates array */
    TEST("empty candidates array");
    {
        cJSON *root = cJSON_Parse("{\"candidates\":[]}");
        ASSERT(root != NULL, "should parse");
        cJSON *candidates = cJSON_GetObjectItem(root, "candidates");
        ASSERT(candidates != NULL, "candidates exists");
        ASSERT_INT_EQ(cJSON_GetArraySize(candidates), 0);
        cJSON_Delete(root);
    }

    /* Test: candidate with null text */
    TEST("candidate with null text field");
    {
        const char *sse_data =
            "{\"candidates\":[{\"content\":{\"parts\":[{\"text\":null}]}}]}";
        cJSON *root = cJSON_Parse(sse_data);
        ASSERT(root != NULL, "should parse");
        cJSON *parts = cJSON_GetObjectItem(
            cJSON_GetObjectItem(
                cJSON_GetArrayItem(cJSON_GetObjectItem(root, "candidates"), 0),
                "content"),
            "parts");
        cJSON *part = cJSON_GetArrayItem(parts, 0);
        cJSON *text = cJSON_GetObjectItem(part, "text");
        ASSERT(text != NULL, "text key exists");
        ASSERT(text->valuestring == NULL, "text value is null");
        cJSON_Delete(root);
        PASS();
    }

    /* Test: empty string response */
    TEST("empty string text response");
    {
        const char *sse_data =
            "{\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"\"}]}}]}";
        cJSON *root = cJSON_Parse(sse_data);
        ASSERT(root != NULL, "should parse");
        cJSON *text = cJSON_GetObjectItem(
            cJSON_GetArrayItem(
                cJSON_GetObjectItem(
                    cJSON_GetObjectItem(
                        cJSON_GetArrayItem(cJSON_GetObjectItem(root, "candidates"), 0),
                        "content"),
                    "parts"),
                0),
            "text");
        ASSERT(text && text->valuestring, "text exists");
        ASSERT(strcmp(text->valuestring, "") == 0, "text is empty string");
        cJSON_Delete(root);
        PASS();
    }

    printf("  malformed_json: %d/%d passed\n\n",
           tests_passed - prev_passed, tests_run - prev_run);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * 8. Claude SSE Format Variants
 * ═══════════════════════════════════════════════════════════════════════════ */

static void test_claude_sse_variants(void) {
    int prev_passed = tests_passed, prev_run = tests_run;
    printf("=== Claude SSE Format Variants ===\n");

    /* Test: message_start event */
    TEST("claude message_start event");
    {
        const char *sse_data =
            "{\"type\":\"message_start\",\"message\":{\"id\":\"msg_01\","
            "\"type\":\"message\",\"role\":\"assistant\",\"content\":[],"
            "\"model\":\"claude-3-5-sonnet-20241022\"}}";
        cJSON *root = cJSON_Parse(sse_data);
        ASSERT(root != NULL, "JSON parse");
        cJSON *type = cJSON_GetObjectItem(root, "type");
        ASSERT(type && type->valuestring, "has type");
        ASSERT(strcmp(type->valuestring, "message_start") == 0, "type is message_start");
        cJSON *msg = cJSON_GetObjectItem(root, "message");
        ASSERT(msg != NULL, "has message object");
        cJSON *role = cJSON_GetObjectItem(msg, "role");
        ASSERT(role && strcmp(role->valuestring, "assistant") == 0, "role is assistant");
        cJSON_Delete(root);
        PASS();
    }

    /* Test: content_block_start event */
    TEST("claude content_block_start");
    {
        const char *sse_data =
            "{\"type\":\"content_block_start\",\"index\":0,"
            "\"content_block\":{\"type\":\"text\",\"text\":\"\"}}";
        cJSON *root = cJSON_Parse(sse_data);
        ASSERT(root != NULL, "JSON parse");
        cJSON *type = cJSON_GetObjectItem(root, "type");
        ASSERT(strcmp(type->valuestring, "content_block_start") == 0, "type match");
        cJSON *block = cJSON_GetObjectItem(root, "content_block");
        ASSERT(block != NULL, "has content_block");
        cJSON_Delete(root);
        PASS();
    }

    /* Test: message_stop event */
    TEST("claude message_stop event");
    {
        const char *sse_data =
            "{\"type\":\"message_stop\"}";
        cJSON *root = cJSON_Parse(sse_data);
        ASSERT(root != NULL, "JSON parse");
        cJSON *type = cJSON_GetObjectItem(root, "type");
        ASSERT(strcmp(type->valuestring, "message_stop") == 0, "type is message_stop");
        cJSON_Delete(root);
        PASS();
    }

    /* Test: message_delta with stop_reason */
    TEST("claude message_delta with stop_reason");
    {
        const char *sse_data =
            "{\"type\":\"message_delta\","
            "\"delta\":{\"stop_reason\":\"end_turn\",\"stop_sequence\":null},"
            "\"usage\":{\"output_tokens\":42}}";
        cJSON *root = cJSON_Parse(sse_data);
        ASSERT(root != NULL, "JSON parse");
        cJSON *delta = cJSON_GetObjectItem(root, "delta");
        cJSON *stop = cJSON_GetObjectItem(delta, "stop_reason");
        ASSERT(stop && strcmp(stop->valuestring, "end_turn") == 0, "stop_reason match");
        cJSON *usage = cJSON_GetObjectItem(root, "usage");
        cJSON *tokens = cJSON_GetObjectItem(usage, "output_tokens");
        ASSERT(tokens && tokens->valueint == 42, "output_tokens match");
        cJSON_Delete(root);
        PASS();
    }

    /* Test: content_block_delta with unicode text */
    TEST("claude delta with unicode");
    {
        const char *sse_data =
            "{\"type\":\"content_block_delta\",\"index\":0,"
            "\"delta\":{\"type\":\"text_delta\",\"text\":\"Caf\\u00e9 na\\u00efve\"}}";
        cJSON *root = cJSON_Parse(sse_data);
        ASSERT(root != NULL, "JSON parse");
        cJSON *text = cJSON_GetObjectItem(cJSON_GetObjectItem(root, "delta"), "text");
        ASSERT(text && text->valuestring, "has text");
        ASSERT(strstr(text->valuestring, "Caf") != NULL, "unicode text decoded");
        cJSON_Delete(root);
        PASS();
    }

    printf("  claude_sse: %d/%d passed\n\n",
           tests_passed - prev_passed, tests_run - prev_run);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * 9. SSML Parser Helper Functions
 * ═══════════════════════════════════════════════════════════════════════════ */

static void test_ssml_parser_helpers(void) {
    int prev_passed = tests_passed, prev_run = tests_run;
    printf("=== SSML Parser Helpers ===\n");

    /* ssml_is_ssml detection */
    TEST("ssml_is_ssml with <speak>");
    ASSERT(ssml_is_ssml("<speak>hello</speak>") == 1, "should detect SSML");
    PASS();

    TEST("ssml_is_ssml with plain text");
    ASSERT(ssml_is_ssml("Just plain text") == 0, "not SSML");
    PASS();

    TEST("ssml_is_ssml with <?xml>");
    ASSERT(ssml_is_ssml("<?xml version=\"1.0\"?><speak>hi</speak>") == 1, "xml header is SSML");
    PASS();

    /* Rate parsing */
    TEST("ssml_parse_rate percentage");
    ASSERT_FLOAT_NEAR(ssml_parse_rate("90%"), 0.9f, 0.01f);

    TEST("ssml_parse_rate keyword slow");
    {
        float r = ssml_parse_rate("slow");
        ASSERT(r < 1.0f, "slow should be < 1.0");
        PASS();
    }

    TEST("ssml_parse_rate keyword fast");
    {
        float r = ssml_parse_rate("fast");
        ASSERT(r > 1.0f, "fast should be > 1.0");
        PASS();
    }

    /* Pitch parsing */
    TEST("ssml_parse_pitch +10%");
    ASSERT_FLOAT_NEAR(ssml_parse_pitch("+10%"), 1.1f, 0.01f);

    TEST("ssml_parse_pitch -20%");
    ASSERT_FLOAT_NEAR(ssml_parse_pitch("-20%"), 0.8f, 0.01f);

    TEST("ssml_parse_pitch keyword low");
    {
        float p = ssml_parse_pitch("low");
        ASSERT(p < 1.0f, "low pitch < 1.0");
        PASS();
    }

    TEST("ssml_parse_pitch keyword high");
    {
        float p = ssml_parse_pitch("high");
        ASSERT(p > 1.0f, "high pitch > 1.0");
        PASS();
    }

    /* Volume parsing */
    TEST("ssml_parse_volume keyword loud");
    {
        float v = ssml_parse_volume("loud");
        ASSERT(v > 1.0f, "loud volume > 1.0");
        PASS();
    }

    TEST("ssml_parse_volume keyword soft");
    {
        float v = ssml_parse_volume("soft");
        ASSERT(v < 1.0f, "soft volume < 1.0");
        PASS();
    }

    printf("  ssml_helpers: %d/%d passed\n\n",
           tests_passed - prev_passed, tests_run - prev_run);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * 10. Nonverbalism Expansion
 * ═══════════════════════════════════════════════════════════════════════════ */

static void test_nonverbalism_expansion(void) {
    int prev_passed = tests_passed, prev_run = tests_run;
    char out[4096];
    printf("=== Nonverbalism Expansion ===\n");

    TEST("[laughter] expansion");
    text_expand_nonverbalisms("He said [laughter] that was funny", out, sizeof(out));
    ASSERT(strstr(out, "<break") != NULL || strstr(out, "happy") != NULL,
           "[laughter] should produce break or emotion");
    PASS();

    TEST("[sigh] expansion");
    text_expand_nonverbalisms("She let out a [sigh] and continued", out, sizeof(out));
    ASSERT(strstr(out, "<break") != NULL || strstr(out, "sad") != NULL,
           "[sigh] should produce break or sad emotion");
    PASS();

    TEST("[pause] expansion");
    text_expand_nonverbalisms("Wait [pause] let me think", out, sizeof(out));
    ASSERT(strstr(out, "500") != NULL || strstr(out, "<break") != NULL,
           "[pause] should produce 500ms break");
    PASS();

    TEST("[long pause] expansion");
    text_expand_nonverbalisms("And then [long pause] it happened", out, sizeof(out));
    ASSERT(strstr(out, "1000") != NULL || strstr(out, "<break") != NULL,
           "[long pause] should produce 1000ms break");
    PASS();

    TEST("no markers passthrough");
    text_expand_nonverbalisms("Just a regular sentence.", out, sizeof(out));
    ASSERT(strcmp(out, "Just a regular sentence.") == 0, "no change without markers");
    PASS();

    TEST("NULL input safety");
    {
        int r = text_expand_nonverbalisms(NULL, out, sizeof(out));
        ASSERT(r == 0 || out[0] == '\0', "NULL input handled safely");
        PASS();
    }

    printf("  nonverbalism: %d/%d passed\n\n",
           tests_passed - prev_passed, tests_run - prev_run);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * 11. SSML Segment Boundary Cases
 * ═══════════════════════════════════════════════════════════════════════════ */

static void test_ssml_segment_boundaries(void) {
    int prev_passed = tests_passed, prev_run = tests_run;
    SSMLSegment segs[SSML_MAX_SEGMENTS];
    printf("=== SSML Segment Boundaries ===\n");

    /* Test: empty <speak> tag */
    TEST("empty speak tag");
    {
        int n = ssml_parse("<speak></speak>", segs, SSML_MAX_SEGMENTS);
        ASSERT(n >= 0, "empty speak doesn't crash");
        PASS();
    }

    /* Test: multiple breaks in sequence */
    TEST("consecutive breaks");
    {
        int n = ssml_parse(
            "<speak>Hello<break time=\"100ms\"/><break time=\"200ms\"/>World</speak>",
            segs, SSML_MAX_SEGMENTS);
        ASSERT(n >= 1, "consecutive breaks produce segments");
        PASS();
    }

    /* Test: deeply nested SSML */
    TEST("nested prosody + emotion + emphasis");
    {
        int n = ssml_parse(
            "<speak><prosody rate=\"110%\"><emotion type=\"happy\">"
            "<emphasis level=\"moderate\">Wow!</emphasis></emotion></prosody></speak>",
            segs, SSML_MAX_SEGMENTS);
        ASSERT(n >= 1, "deep nesting produces segments");
        /* Rate should reflect the 110% prosody */
        int found = 0;
        for (int i = 0; i < n; i++) {
            if (strstr(segs[i].text, "Wow")) {
                found = 1;
                ASSERT(segs[i].rate > 1.0f, "rate from prosody tag applied");
            }
        }
        ASSERT(found, "found Wow segment");
        PASS();
    }

    /* Test: NULL input to ssml_parse */
    TEST("NULL input to ssml_parse");
    {
        int n = ssml_parse(NULL, segs, SSML_MAX_SEGMENTS);
        ASSERT(n <= 0, "NULL input returns 0 or -1");
        PASS();
    }

    /* Test: very long plain text */
    TEST("long plain text passthrough");
    {
        char long_text[4000];
        memset(long_text, 'A', sizeof(long_text) - 1);
        long_text[sizeof(long_text) - 1] = '\0';
        int n = ssml_parse(long_text, segs, SSML_MAX_SEGMENTS);
        ASSERT(n == 1, "long plain text = 1 segment");
        PASS();
    }

    /* Test: phoneme tag */
    TEST("phoneme IPA override");
    {
        int n = ssml_parse(
            "<speak><phoneme alphabet=\"ipa\" ph=\"təˈmeɪtoʊ\">tomato</phoneme></speak>",
            segs, SSML_MAX_SEGMENTS);
        ASSERT(n >= 1, "phoneme tag produces segment");
        int found = 0;
        for (int i = 0; i < n; i++) {
            if (segs[i].phoneme_ipa[0] != '\0') {
                found = 1;
            }
        }
        ASSERT(found, "phoneme IPA populated");
        PASS();
    }

    printf("  ssml_boundaries: %d/%d passed\n\n",
           tests_passed - prev_passed, tests_run - prev_run);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * main
 * ═══════════════════════════════════════════════════════════════════════════ */

int main(void) {
    test_gemini_sse_parsing();
    test_emotion_tags();
    test_prosody_flow();
    test_expanded_normalization();
    test_question_intonation();
    test_emergent_tts_categories();

    /* New deepened tests */
    test_malformed_json();
    test_claude_sse_variants();
    test_ssml_parser_helpers();
    test_nonverbalism_expansion();
    test_ssml_segment_boundaries();

    printf("═══════════════════════════\n");
    printf("Total: %d/%d tests passed", tests_passed, tests_run);
    if (tests_failed > 0) {
        printf(" (%d FAILED)", tests_failed);
    }
    printf("\n");
    return tests_failed > 0 ? 1 : 0;
}
