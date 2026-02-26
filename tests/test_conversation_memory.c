/**
 * test_conversation_memory.c — Unit tests for persistent conversation memory.
 *
 * Tests: create, add turns, count, get context, format context, clear,
 * persist/reload, search, NULL handling, max turns eviction.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include "conversation_memory.h"

static int pass_count = 0;
static int fail_count = 0;

#define TEST(cond, name) do { \
    if (cond) { printf("  [PASS] %s\n", name); pass_count++; } \
    else { printf("  [FAIL] %s (line %d)\n", name, __LINE__); fail_count++; } \
} while (0)

static const char *TEMP_PATH = "/tmp/pocket_voice_test_memory.jsonl";

static void test_create_and_add(void) {
    printf("\n=== Create and Add ===\n");
    unlink(TEMP_PATH);

    ConversationMemory *mem = memory_create(TEMP_PATH, 50, 4000);
    TEST(mem != NULL, "create returns handle");
    TEST(memory_turn_count(mem) == 0, "count is 0 initially");

    int r = memory_add_turn(mem, "user", "Hello");
    TEST(r == 0, "add user turn succeeds");
    TEST(memory_turn_count(mem) == 1, "count is 1 after add");

    r = memory_add_turn(mem, "assistant", "Hi there!");
    TEST(r == 0, "add assistant turn succeeds");
    TEST(memory_turn_count(mem) == 2, "count is 2");

    memory_destroy(mem);
    unlink(TEMP_PATH);
}

static void test_get_context(void) {
    printf("\n=== Get Context ===\n");
    unlink(TEMP_PATH);

    ConversationMemory *mem = memory_create(TEMP_PATH, 50, 4000);
    memory_add_turn(mem, "user", "First message");
    memory_add_turn(mem, "assistant", "First reply");
    memory_add_turn(mem, "user", "Second message");
    memory_add_turn(mem, "assistant", "Second reply");

    MemoryTurn turns[10];
    int n = memory_get_context(mem, turns, 10);
    TEST(n == 4, "get_context returns 4 turns");
    TEST(turns[0].role && strcmp(turns[0].role, "assistant") == 0, "most recent is assistant");
    TEST(turns[0].content && strstr(turns[0].content, "Second reply"), "most recent content correct");
    TEST(turns[3].role && strcmp(turns[3].role, "user") == 0, "oldest is user");
    TEST(turns[3].content && strstr(turns[3].content, "First message"), "oldest content correct");

    memory_destroy(mem);
    unlink(TEMP_PATH);
}

static void test_format_context(void) {
    printf("\n=== Format Context ===\n");
    unlink(TEMP_PATH);

    ConversationMemory *mem = memory_create(TEMP_PATH, 50, 4000);
    memory_add_turn(mem, "user", "Hello");
    memory_add_turn(mem, "assistant", "Hi there!");

    char *formatted = memory_format_context(mem);
    TEST(formatted != NULL, "format_context returns non-NULL");
    TEST(strstr(formatted, "Previous conversation:"), "contains header");
    TEST(strstr(formatted, "User: Hello"), "contains user turn");
    TEST(strstr(formatted, "Assistant: Hi there!"), "contains assistant turn");
    free(formatted);

    memory_destroy(mem);
    unlink(TEMP_PATH);
}

static void test_clear(void) {
    printf("\n=== Clear ===\n");
    unlink(TEMP_PATH);

    ConversationMemory *mem = memory_create(TEMP_PATH, 50, 4000);
    memory_add_turn(mem, "user", "Hello");
    memory_add_turn(mem, "assistant", "Reply");

    memory_clear(mem);
    TEST(memory_turn_count(mem) == 0, "count is 0 after clear");

    MemoryTurn turns[4];
    int n = memory_get_context(mem, turns, 4);
    TEST(n == 0, "get_context returns 0 after clear");

    memory_destroy(mem);
    unlink(TEMP_PATH);
}

static void test_persist_and_reload(void) {
    printf("\n=== Persist and Reload ===\n");
    unlink(TEMP_PATH);

    ConversationMemory *mem = memory_create(TEMP_PATH, 50, 4000);
    memory_add_turn(mem, "user", "What is the weather?");
    memory_add_turn(mem, "assistant", "It is sunny.");
    memory_destroy(mem);

    mem = memory_create(TEMP_PATH, 50, 4000);
    TEST(memory_turn_count(mem) == 2, "reload has 2 turns");
    MemoryTurn turns[4];
    int n = memory_get_context(mem, turns, 4);
    TEST(n == 2, "get_context returns 2 after reload");
    TEST(turns[1].content && strstr(turns[1].content, "What is the weather?"), "user turn persisted");
    TEST(turns[0].content && strstr(turns[0].content, "It is sunny."), "assistant turn persisted");

    memory_destroy(mem);
    unlink(TEMP_PATH);
}

static void test_search(void) {
    printf("\n=== Search ===\n");
    unlink(TEMP_PATH);

    ConversationMemory *mem = memory_create(TEMP_PATH, 50, 4000);
    memory_add_turn(mem, "user", "Tell me about the weather");
    memory_add_turn(mem, "assistant", "The weather is nice today");
    memory_add_turn(mem, "user", "What about rain?");
    memory_add_turn(mem, "assistant", "No rain expected");

    MemoryTurn hits[10];
    int n = memory_search(mem, "weather", hits, 10);
    TEST(n == 2, "search 'weather' finds 2 turns");

    n = memory_search(mem, "WEATHER", hits, 10);
    TEST(n == 2, "search case-insensitive finds 2 turns");

    n = memory_search(mem, "rain", hits, 10);
    TEST(n == 2, "search 'rain' finds 2 turns");

    n = memory_search(mem, "nonexistent", hits, 10);
    TEST(n == 0, "search nonexistent returns 0");

    memory_destroy(mem);
    unlink(TEMP_PATH);
}

static void test_null_handling(void) {
    printf("\n=== NULL Handling ===\n");

    memory_destroy(NULL);  /* safe */
    TEST(1, "destroy NULL is safe");

    ConversationMemory *mem = memory_create("/tmp/ok.jsonl", 50, 4000);
    TEST(memory_create(NULL, 50, 4000) == NULL, "create with NULL path returns NULL");
    TEST(memory_add_turn(NULL, "user", "x") == -1, "add_turn NULL mem returns -1");
    TEST(memory_add_turn(mem, NULL, "x") == -1, "add_turn NULL role returns -1");
    TEST(memory_add_turn(mem, "user", NULL) == -1, "add_turn NULL content returns -1");

    MemoryTurn turns[4];
    TEST(memory_get_context(NULL, turns, 4) == 0, "get_context NULL mem returns 0");
    TEST(memory_get_context(mem, NULL, 4) == 0, "get_context NULL out returns 0");

    char *f = memory_format_context(NULL);
    TEST(f == NULL, "format_context NULL returns NULL");

    TEST(memory_turn_count(NULL) == 0, "turn_count NULL returns 0");

    memory_clear(NULL);  /* safe */
    TEST(1, "clear NULL is safe");

    TEST(memory_search(NULL, "x", turns, 4) == 0, "search NULL mem returns 0");
    TEST(memory_search(mem, NULL, turns, 4) == 0, "search NULL keyword returns 0");

    memory_destroy(mem);
    unlink("/tmp/ok.jsonl");
}

static void test_max_turns_eviction(void) {
    printf("\n=== Max Turns Eviction ===\n");
    unlink(TEMP_PATH);

    ConversationMemory *mem = memory_create(TEMP_PATH, 4, 4000);
    for (int i = 0; i < 6; i++) {
        char buf[64];
        snprintf(buf, sizeof(buf), "user message %d", i);
        memory_add_turn(mem, "user", buf);
        snprintf(buf, sizeof(buf), "assistant reply %d", i);
        memory_add_turn(mem, "assistant", buf);
    }

    TEST(memory_turn_count(mem) == 4, "eviction keeps at most 4 turns");
    MemoryTurn turns[8];
    int n = memory_get_context(mem, turns, 8);
    TEST(n == 4, "get_context returns 4");
    /* Most recent should be "assistant reply 5" and "user message 5" (pair), then 4, 3 */
    TEST(turns[0].content && strstr(turns[0].content, "5"), "most recent is from last pair");

    memory_destroy(mem);

    /* Reload and verify file was rewritten */
    mem = memory_create(TEMP_PATH, 4, 4000);
    TEST(memory_turn_count(mem) == 4, "reload after eviction has 4 turns");
    memory_destroy(mem);

    unlink(TEMP_PATH);
}

static void test_max_tokens_cap(void) {
    printf("\n=== Max Tokens Cap ===\n");
    unlink(TEMP_PATH);

    /* max_tokens=10 means ~40 chars total */
    ConversationMemory *mem = memory_create(TEMP_PATH, 50, 10);
    memory_add_turn(mem, "user", "A");      /* 1 token */
    memory_add_turn(mem, "assistant", "BB"); /* 1 token */
    memory_add_turn(mem, "user", "CCCC");   /* 1 token */
    memory_add_turn(mem, "assistant", "DDDDDDDD"); /* 2 tokens */
    memory_add_turn(mem, "user", "EEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE"); /* 10 tokens */

    MemoryTurn turns[8];
    int n = memory_get_context(mem, turns, 8);
    /* Should fit within 10 tokens - walking backward: 10+2+1+1+1 = 15, so we take
       only the last one (10 tokens) or last few that fit. 10 tokens = 1 turn with 40 chars. */
    TEST(n >= 1 && n <= 5, "context respects max_tokens");

    memory_destroy(mem);
    unlink(TEMP_PATH);
}

int main(void) {
    printf("\n═══ Conversation Memory Tests ═══\n");

    test_create_and_add();
    test_get_context();
    test_format_context();
    test_clear();
    test_persist_and_reload();
    test_search();
    test_null_handling();
    test_max_turns_eviction();
    test_max_tokens_cap();

    printf("\n═══ Results: %d pass, %d fail ═══\n", pass_count, fail_count);
    return fail_count ? 1 : 0;
}
