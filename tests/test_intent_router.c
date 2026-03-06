/**
 * test_intent_router.c — Unit tests for intent router.
 *
 * Tests: create default (heuristic), route greetings->FAST, thanks->FAST,
 * bye->FAST, short->BACKCHANNEL, questions->FULL/MEDIUM, confidence, context.
 *
 * Build: make test-intent-router
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "intent_router.h"

static int pass_count = 0;
static int fail_count = 0;

#define TEST(cond, name) do { \
    if (cond) { printf("  [PASS] %s\n", name); pass_count++; } \
    else { printf("  [FAIL] %s (line %d)\n", name, __LINE__); fail_count++; } \
} while (0)

static void test_create_default(void) {
    printf("\n=== Create Default (Heuristic) ===\n");

    IntentRouter *r = intent_router_create_default();
    TEST(r != NULL, "create_default returns non-NULL");

    intent_router_destroy(r);
    intent_router_destroy(NULL);
    TEST(1, "destroy NULL is safe");
}

static void test_route_greeting(void) {
    printf("\n=== Route Greetings -> FAST ===\n");

    IntentRouter *r = intent_router_create_default();
    RoutingDecision d;

    d = intent_router_route(r, "hi", 1, NULL, NULL);
    TEST(d.route == ROUTE_FAST, "hi -> ROUTE_FAST");
    TEST(d.fast_type == FAST_GREETING, "hi -> FAST_GREETING");

    d = intent_router_route(r, "hey", 1, NULL, NULL);
    TEST(d.route == ROUTE_FAST, "hey -> ROUTE_FAST");

    d = intent_router_route(r, "hello there", 2, NULL, NULL);
    TEST(d.route == ROUTE_FAST, "hello there -> ROUTE_FAST");

    intent_router_destroy(r);
}

static void test_route_thanks(void) {
    printf("\n=== Route Thanks -> FAST ===\n");

    IntentRouter *r = intent_router_create_default();
    RoutingDecision d = intent_router_route(r, "thanks", 1, NULL, NULL);

    TEST(d.route == ROUTE_FAST, "thanks -> ROUTE_FAST");
    TEST(d.fast_type == FAST_THANKS, "thanks -> FAST_THANKS");

    d = intent_router_route(r, "thank you so much", 4, NULL, NULL);
    TEST(d.route == ROUTE_FAST, "thank you -> ROUTE_FAST");

    intent_router_destroy(r);
}

static void test_route_bye(void) {
    printf("\n=== Route Bye -> FAST ===\n");

    IntentRouter *r = intent_router_create_default();
    RoutingDecision d = intent_router_route(r, "bye", 1, NULL, NULL);

    TEST(d.route == ROUTE_FAST, "bye -> ROUTE_FAST");
    TEST(d.fast_type == FAST_GOODBYE, "bye -> FAST_GOODBYE");

    d = intent_router_route(r, "see you later", 3, NULL, NULL);
    TEST(d.route == ROUTE_FAST, "see you later -> ROUTE_FAST");

    intent_router_destroy(r);
}

static void test_route_short_backchannel(void) {
    printf("\n=== Route Short Utterance -> BACKCHANNEL ===\n");

    IntentRouter *r = intent_router_create_default();
    RoutingDecision d = intent_router_route(r, "mhm", 1, NULL, NULL);

    TEST(d.route == ROUTE_BACKCHANNEL, "mhm -> ROUTE_BACKCHANNEL");

    d = intent_router_route(r, "yeah", 1, NULL, NULL);
    TEST(d.route == ROUTE_BACKCHANNEL, "yeah -> ROUTE_BACKCHANNEL");

    intent_router_destroy(r);
}

static void test_route_question(void) {
    printf("\n=== Route Questions ===\n");

    IntentRouter *r = intent_router_create_default();
    RoutingDecision d;

    d = intent_router_route(r, "what is the weather?", 4, NULL, NULL);
    TEST(d.route == ROUTE_MEDIUM, "short question -> ROUTE_MEDIUM");

    d = intent_router_route(r, "tell me a long story about ancient Rome", 8, NULL, NULL);
    TEST(d.route == ROUTE_FULL, "long utterance -> ROUTE_FULL");

    intent_router_destroy(r);
}

static void test_route_complex_question(void) {
    printf("\n=== Route Complex Question -> FULL ===\n");

    IntentRouter *r = intent_router_create_default();
    RoutingDecision d = intent_router_route(r,
        "explain the theory of relativity in detail and how it affects time",
        12, NULL, NULL);

    TEST(d.route == ROUTE_FULL, "complex question -> ROUTE_FULL");

    intent_router_destroy(r);
}

static void test_confidence_values(void) {
    printf("\n=== Confidence Values ===\n");

    IntentRouter *r = intent_router_create_default();
    RoutingDecision d = intent_router_route(r, "hi", 1, NULL, NULL);

    TEST(d.confidence > 0.0f, "confidence > 0");
    TEST(d.confidence <= 1.0f, "confidence <= 1");

    intent_router_destroy(r);
}

static void test_set_context(void) {
    printf("\n=== Set Context ===\n");

    IntentRouter *r = intent_router_create_default();
    intent_router_set_context(r, "User: hello\nAssistant: Hi there!");
    intent_router_set_context(NULL, "test");
    intent_router_set_context(r, NULL);
    TEST(1, "set_context does not crash");
    intent_router_destroy(r);
}

static void test_fast_text(void) {
    printf("\n=== Fast Response Text ===\n");

    TEST(strlen(intent_router_fast_text(FAST_GREETING)) > 0, "FAST_GREETING has text");
    TEST(strlen(intent_router_fast_text(FAST_THANKS)) > 0, "FAST_THANKS has text");
    TEST(strcmp(intent_router_fast_text(FAST_GOODBYE), "See you later!") == 0,
         "FAST_GOODBYE text");
}

static void test_route_null_router(void) {
    printf("\n=== Route with NULL Router ===\n");

    RoutingDecision d = intent_router_route(NULL, "hi", 1, NULL, NULL);
    TEST(d.route == ROUTE_FULL, "NULL router returns default ROUTE_FULL");
}

static void test_create_with_missing_weights(void) {
    printf("\n=== Create with Missing Weights ===\n");

    IntentRouter *r = intent_router_create("/nonexistent/path.router");
    TEST(r == NULL, "create(missing path) returns NULL");
}

static void test_route_auto_word_count(void) {
    printf("\n=== Route Auto Word Count ===\n");

    IntentRouter *r = intent_router_create_default();
    RoutingDecision d = intent_router_route(r, "hello world", -1, NULL, NULL);
    TEST(d.route == ROUTE_FAST, "auto word count works");
    intent_router_destroy(r);
}

int main(void) {
    printf("Intent Router Tests\n");

    test_create_default();
    test_route_greeting();
    test_route_thanks();
    test_route_bye();
    test_route_short_backchannel();
    test_route_question();
    test_route_complex_question();
    test_confidence_values();
    test_set_context();
    test_fast_text();
    test_route_null_router();
    test_create_with_missing_weights();
    test_route_auto_word_count();

    printf("\n--- Result: %d passed, %d failed ---\n", pass_count, fail_count);
    return fail_count ? 1 : 0;
}
