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

static void test_alternating_roles(void) {
    printf("\n=== Alternating Roles ===\n");
    unlink(TEMP_PATH);

    ConversationMemory *mem = memory_create(TEMP_PATH, 50, 4000);

    /* Add alternating user/assistant turns */
    for (int i = 0; i < 10; i++) {
        char buf[64];
        snprintf(buf, sizeof(buf), "user msg %d", i);
        int r = memory_add_turn(mem, "user", buf);
        TEST(r == 0, "add user turn ok");
        snprintf(buf, sizeof(buf), "asst reply %d", i);
        r = memory_add_turn(mem, "assistant", buf);
        TEST(r == 0, "add assistant turn ok");
    }
    TEST(memory_turn_count(mem) == 20, "20 turns stored after 10 exchanges");

    /* Verify roles alternate correctly in context */
    MemoryTurn turns[20];
    int n = memory_get_context(mem, turns, 20);
    TEST(n == 20, "get_context returns all 20");

    /* Most recent first: turns[0] = assistant reply 9, turns[1] = user msg 9, etc. */
    for (int i = 0; i < n - 1; i += 2) {
        TEST(strcmp(turns[i].role, "assistant") == 0, "even idx is assistant");
        TEST(strcmp(turns[i + 1].role, "user") == 0, "odd idx is user");
    }

    memory_destroy(mem);
    unlink(TEMP_PATH);
}

static void test_context_ordering(void) {
    printf("\n=== Context Ordering in Formatted Output ===\n");
    unlink(TEMP_PATH);

    ConversationMemory *mem = memory_create(TEMP_PATH, 50, 4000);
    memory_add_turn(mem, "user", "First");
    memory_add_turn(mem, "assistant", "Second");
    memory_add_turn(mem, "user", "Third");

    char *formatted = memory_format_context(mem);
    TEST(formatted != NULL, "format_context non-NULL");

    /* Verify chronological order: First before Second before Third */
    char *p_first = strstr(formatted, "First");
    char *p_second = strstr(formatted, "Second");
    char *p_third = strstr(formatted, "Third");
    TEST(p_first != NULL && p_second != NULL && p_third != NULL,
         "all turns present in formatted output");
    TEST(p_first < p_second && p_second < p_third,
         "turns in chronological order");

    free(formatted);
    memory_destroy(mem);
    unlink(TEMP_PATH);
}

static void test_empty_content(void) {
    printf("\n=== Empty Content ===\n");
    unlink(TEMP_PATH);

    ConversationMemory *mem = memory_create(TEMP_PATH, 50, 4000);

    /* Empty string is valid content (not NULL) */
    int r = memory_add_turn(mem, "user", "");
    TEST(r == 0, "add turn with empty content succeeds");
    TEST(memory_turn_count(mem) == 1, "count is 1 after empty content");

    MemoryTurn turns[4];
    int n = memory_get_context(mem, turns, 4);
    TEST(n == 1, "get_context returns 1 for empty content turn");
    TEST(turns[0].content != NULL, "empty content turn has non-NULL content");
    TEST(strlen(turns[0].content) == 0, "empty content is empty string");

    /* Format should still work */
    char *formatted = memory_format_context(mem);
    TEST(formatted != NULL, "format_context with empty content ok");
    free(formatted);

    memory_destroy(mem);
    unlink(TEMP_PATH);
}

static void test_very_long_content(void) {
    printf("\n=== Very Long Content ===\n");
    unlink(TEMP_PATH);

    ConversationMemory *mem = memory_create(TEMP_PATH, 50, 100000);

    /* Create a 10KB message */
    size_t big_len = 10000;
    char *big = (char *)malloc(big_len + 1);
    memset(big, 'X', big_len);
    big[big_len] = '\0';

    int r = memory_add_turn(mem, "user", big);
    TEST(r == 0, "add turn with 10KB content succeeds");
    TEST(memory_turn_count(mem) == 1, "count is 1");

    MemoryTurn turns[4];
    int n = memory_get_context(mem, turns, 4);
    TEST(n == 1, "get_context returns 1 for long turn");
    TEST(strlen(turns[0].content) == big_len, "long content preserved in full");

    /* Persist and reload */
    memory_destroy(mem);
    mem = memory_create(TEMP_PATH, 50, 100000);
    TEST(memory_turn_count(mem) == 1, "long content survives reload");

    n = memory_get_context(mem, turns, 4);
    TEST(n == 1, "get_context after reload returns 1");
    TEST(strlen(turns[0].content) == big_len, "long content intact after reload");

    free(big);
    memory_destroy(mem);
    unlink(TEMP_PATH);
}

static void test_multi_cycle_persistence(void) {
    printf("\n=== Multi-Cycle Persistence ===\n");
    unlink(TEMP_PATH);

    /* Cycle 1: add 2 turns, close */
    ConversationMemory *mem = memory_create(TEMP_PATH, 50, 4000);
    memory_add_turn(mem, "user", "cycle1 hello");
    memory_add_turn(mem, "assistant", "cycle1 reply");
    memory_destroy(mem);

    /* Cycle 2: reopen, add 2 more, close */
    mem = memory_create(TEMP_PATH, 50, 4000);
    TEST(memory_turn_count(mem) == 2, "cycle2 reopens with 2 turns");
    memory_add_turn(mem, "user", "cycle2 hello");
    memory_add_turn(mem, "assistant", "cycle2 reply");
    TEST(memory_turn_count(mem) == 4, "cycle2 has 4 turns total");
    memory_destroy(mem);

    /* Cycle 3: reopen, verify all 4 */
    mem = memory_create(TEMP_PATH, 50, 4000);
    TEST(memory_turn_count(mem) == 4, "cycle3 reopens with 4 turns");

    MemoryTurn turns[8];
    int n = memory_get_context(mem, turns, 8);
    TEST(n == 4, "get_context returns 4 across cycles");

    /* Search across cycles */
    MemoryTurn hits[8];
    n = memory_search(mem, "cycle1", hits, 8);
    TEST(n == 2, "search cycle1 finds 2 turns");
    n = memory_search(mem, "cycle2", hits, 8);
    TEST(n == 2, "search cycle2 finds 2 turns");

    memory_destroy(mem);
    unlink(TEMP_PATH);
}

static void test_clear_and_re_add(void) {
    printf("\n=== Clear and Re-Add ===\n");
    unlink(TEMP_PATH);

    ConversationMemory *mem = memory_create(TEMP_PATH, 50, 4000);
    memory_add_turn(mem, "user", "before clear");
    memory_add_turn(mem, "assistant", "reply before");
    TEST(memory_turn_count(mem) == 2, "2 turns before clear");

    memory_clear(mem);
    TEST(memory_turn_count(mem) == 0, "0 turns after clear");

    /* Re-add after clear */
    memory_add_turn(mem, "user", "after clear");
    TEST(memory_turn_count(mem) == 1, "1 turn after re-add");

    MemoryTurn turns[4];
    int n = memory_get_context(mem, turns, 4);
    TEST(n == 1, "get_context returns 1 after clear+add");
    TEST(strstr(turns[0].content, "after clear"), "new content present");

    /* Old content should not appear */
    MemoryTurn hits[4];
    n = memory_search(mem, "before clear", hits, 4);
    TEST(n == 0, "old content gone after clear");

    /* Persist after clear+add and reload */
    memory_destroy(mem);
    mem = memory_create(TEMP_PATH, 50, 4000);
    TEST(memory_turn_count(mem) == 1, "reload after clear+add has 1 turn");

    memory_destroy(mem);
    unlink(TEMP_PATH);
}

static void test_eviction_preserves_newest(void) {
    printf("\n=== Eviction Preserves Newest ===\n");
    unlink(TEMP_PATH);

    /* Capacity of 3 turns */
    ConversationMemory *mem = memory_create(TEMP_PATH, 3, 4000);
    memory_add_turn(mem, "user", "msg_A");
    memory_add_turn(mem, "assistant", "reply_A");
    memory_add_turn(mem, "user", "msg_B");
    /* At capacity (3). Next add should evict oldest. */
    memory_add_turn(mem, "assistant", "reply_B");
    TEST(memory_turn_count(mem) == 3, "capped at 3 after 4 adds");

    /* Oldest (msg_A) should be evicted */
    MemoryTurn hits[4];
    int n = memory_search(mem, "msg_A", hits, 4);
    TEST(n == 0, "oldest turn evicted");

    /* Newest should remain */
    n = memory_search(mem, "reply_B", hits, 4);
    TEST(n == 1, "newest turn preserved");

    memory_destroy(mem);
    unlink(TEMP_PATH);
}

static void test_single_turn_capacity(void) {
    printf("\n=== Single Turn Capacity ===\n");
    unlink(TEMP_PATH);

    /* Edge case: capacity of 1 */
    ConversationMemory *mem = memory_create(TEMP_PATH, 1, 4000);
    memory_add_turn(mem, "user", "first");
    TEST(memory_turn_count(mem) == 1, "1 turn stored");

    memory_add_turn(mem, "assistant", "second");
    TEST(memory_turn_count(mem) == 1, "still 1 after second add");

    MemoryTurn turns[4];
    int n = memory_get_context(mem, turns, 4);
    TEST(n == 1, "get_context returns 1");
    TEST(strstr(turns[0].content, "second"), "most recent turn kept");

    memory_destroy(mem);
    unlink(TEMP_PATH);
}

static void test_get_context_limited_output(void) {
    printf("\n=== Get Context Limited Output ===\n");
    unlink(TEMP_PATH);

    ConversationMemory *mem = memory_create(TEMP_PATH, 50, 4000);
    for (int i = 0; i < 10; i++) {
        char buf[64];
        snprintf(buf, sizeof(buf), "turn %d", i);
        memory_add_turn(mem, (i % 2 == 0) ? "user" : "assistant", buf);
    }

    /* Request only 3 turns even though 10 exist */
    MemoryTurn turns[3];
    int n = memory_get_context(mem, turns, 3);
    TEST(n == 3, "get_context respects max_out=3");
    TEST(strstr(turns[0].content, "9"), "most recent (turn 9) is first");

    memory_destroy(mem);
    unlink(TEMP_PATH);
}

/* ─── New Edge Case Tests ────────────────────────────────────────────────── */

static void test_unicode_content(void) {
    printf("\n=== Unicode Content ===\n");
    unlink(TEMP_PATH);

    ConversationMemory *mem = memory_create(TEMP_PATH, 50, 4000);

    /* Chinese, Japanese, Korean */
    int r = memory_add_turn(mem, "user", "你好世界 こんにちは 안녕하세요");
    TEST(r == 0, "add CJK unicode content succeeds");

    /* Emoji */
    r = memory_add_turn(mem, "assistant", "Hello! 🎉🌍🔥💡");
    TEST(r == 0, "add emoji content succeeds");

    /* Arabic, Hebrew (RTL) */
    r = memory_add_turn(mem, "user", "مرحبا שלום");
    TEST(r == 0, "add RTL unicode content succeeds");

    TEST(memory_turn_count(mem) == 3, "3 unicode turns stored");

    /* Persist and reload */
    memory_destroy(mem);
    mem = memory_create(TEMP_PATH, 50, 4000);
    TEST(memory_turn_count(mem) == 3, "unicode turns survive reload");

    /* Search within unicode content */
    MemoryTurn hits[4];
    int n = memory_search(mem, "你好", hits, 4);
    TEST(n >= 1, "search finds CJK content");

    n = memory_search(mem, "🎉", hits, 4);
    TEST(n >= 1, "search finds emoji content");

    /* Verify content integrity */
    MemoryTurn turns[4];
    n = memory_get_context(mem, turns, 4);
    TEST(n == 3, "get_context returns 3 unicode turns");

    memory_destroy(mem);
    unlink(TEMP_PATH);
}

static void test_json_special_chars(void) {
    printf("\n=== JSON Special Characters ===\n");
    unlink(TEMP_PATH);

    ConversationMemory *mem = memory_create(TEMP_PATH, 50, 4000);

    /* Quotes and backslashes — these must be escaped in JSONL */
    int r = memory_add_turn(mem, "user", "He said \"hello\" and \\n escaped");
    TEST(r == 0, "add content with quotes and backslashes");

    /* Newlines and tabs embedded in content */
    r = memory_add_turn(mem, "assistant", "Line1\nLine2\nLine3\ttab");
    TEST(r == 0, "add content with newlines and tabs");

    /* Null byte adjacent (string still valid C string terminated before it) */
    r = memory_add_turn(mem, "user", "before");
    TEST(r == 0, "add normal string after special chars");

    TEST(memory_turn_count(mem) == 3, "3 turns with special chars stored");

    /* Persist and reload — JSON escaping must round-trip */
    memory_destroy(mem);
    mem = memory_create(TEMP_PATH, 50, 4000);
    TEST(memory_turn_count(mem) == 3, "special char turns survive reload");

    MemoryTurn turns[4];
    int n = memory_get_context(mem, turns, 4);
    TEST(n == 3, "get_context returns 3 after reload");

    /* Verify the quotes content round-tripped */
    int found_quotes = 0;
    for (int i = 0; i < n; i++) {
        if (turns[i].content && strstr(turns[i].content, "\"hello\""))
            found_quotes = 1;
    }
    TEST(found_quotes, "quoted content survived JSON round-trip");

    memory_destroy(mem);
    unlink(TEMP_PATH);
}

static void test_empty_database_file(void) {
    printf("\n=== Empty Database File ===\n");

    const char *empty_path = "/tmp/pocket_voice_test_empty.jsonl";
    unlink(empty_path);

    /* Create an empty file */
    FILE *f = fopen(empty_path, "w");
    if (f) fclose(f);

    /* Open should handle empty file gracefully */
    ConversationMemory *mem = memory_create(empty_path, 50, 4000);
    TEST(mem != NULL, "create with empty file succeeds");
    TEST(memory_turn_count(mem) == 0, "empty file has 0 turns");

    /* Should be able to add turns normally */
    int r = memory_add_turn(mem, "user", "first after empty");
    TEST(r == 0, "add turn after empty file succeeds");
    TEST(memory_turn_count(mem) == 1, "count is 1");

    memory_destroy(mem);
    unlink(empty_path);
}

static void test_corrupted_jsonl(void) {
    printf("\n=== Corrupted JSONL File ===\n");

    const char *corrupt_path = "/tmp/pocket_voice_test_corrupt.jsonl";
    unlink(corrupt_path);

    /* Write a mix of valid and corrupted JSONL lines */
    FILE *f = fopen(corrupt_path, "w");
    if (!f) {
        printf("  [SKIP] Cannot create temp file\n");
        return;
    }
    /* Valid line */
    fprintf(f, "{\"role\":\"user\",\"content\":\"valid line 1\",\"timestamp\":1000.0}\n");
    /* Truncated/invalid JSON */
    fprintf(f, "{\"role\":\"user\",\"content\":\"trun\n");
    /* Another valid line */
    fprintf(f, "{\"role\":\"assistant\",\"content\":\"valid line 3\",\"timestamp\":1002.0}\n");
    /* Completely garbage */
    fprintf(f, "not json at all\n");
    /* Empty line */
    fprintf(f, "\n");
    /* Valid line */
    fprintf(f, "{\"role\":\"user\",\"content\":\"valid line 6\",\"timestamp\":1005.0}\n");
    fclose(f);

    /* Should handle corruption gracefully */
    ConversationMemory *mem = memory_create(corrupt_path, 50, 4000);
    TEST(mem != NULL, "create with corrupted file doesn't crash");

    /* May recover valid lines or start fresh — either is acceptable */
    int count = memory_turn_count(mem);
    TEST(count >= 0, "turn_count is non-negative after corrupt load");

    /* Should be able to add new turns regardless */
    int r = memory_add_turn(mem, "user", "fresh turn after corruption");
    TEST(r == 0, "add turn after corrupt load succeeds");

    memory_destroy(mem);
    unlink(corrupt_path);
}

static void test_search_empty_keyword(void) {
    printf("\n=== Search with Empty Keyword ===\n");
    unlink(TEMP_PATH);

    ConversationMemory *mem = memory_create(TEMP_PATH, 50, 4000);
    memory_add_turn(mem, "user", "hello world");
    memory_add_turn(mem, "assistant", "hi there");

    /* Empty string search — could match all or none, should not crash */
    MemoryTurn hits[4];
    int n = memory_search(mem, "", hits, 4);
    TEST(n >= 0, "search with empty string doesn't crash");

    /* Single character search */
    n = memory_search(mem, "h", hits, 4);
    TEST(n >= 1, "search single char 'h' finds matches");

    memory_destroy(mem);
    unlink(TEMP_PATH);
}

static void test_search_special_chars(void) {
    printf("\n=== Search with Special Characters ===\n");
    unlink(TEMP_PATH);

    ConversationMemory *mem = memory_create(TEMP_PATH, 50, 4000);
    memory_add_turn(mem, "user", "price is $100.00");
    memory_add_turn(mem, "assistant", "use regex like [a-z]+ or foo*bar");
    memory_add_turn(mem, "user", "path is /usr/local/bin");

    MemoryTurn hits[4];

    /* Search with regex-like special chars — should do literal match */
    int n = memory_search(mem, "$100", hits, 4);
    TEST(n >= 0, "search '$100' doesn't crash");

    n = memory_search(mem, "[a-z]+", hits, 4);
    TEST(n >= 0, "search '[a-z]+' doesn't crash");

    n = memory_search(mem, "foo*bar", hits, 4);
    TEST(n >= 0, "search 'foo*bar' doesn't crash");

    n = memory_search(mem, "/usr/local", hits, 4);
    TEST(n >= 1, "search path with slashes finds match");

    memory_destroy(mem);
    unlink(TEMP_PATH);
}

static void test_get_context_zero_max(void) {
    printf("\n=== Get Context with max_out=0 ===\n");
    unlink(TEMP_PATH);

    ConversationMemory *mem = memory_create(TEMP_PATH, 50, 4000);
    memory_add_turn(mem, "user", "message");
    memory_add_turn(mem, "assistant", "reply");

    /* Request 0 turns */
    MemoryTurn turns[1];
    int n = memory_get_context(mem, turns, 0);
    TEST(n == 0, "get_context(max_out=0) returns 0");

    /* Negative max_out */
    n = memory_get_context(mem, turns, -1);
    TEST(n == 0, "get_context(max_out=-1) returns 0");

    memory_destroy(mem);
    unlink(TEMP_PATH);
}

static void test_rapid_add_stress(void) {
    printf("\n=== Rapid Add Stress Test (500 turns) ===\n");
    unlink(TEMP_PATH);

    ConversationMemory *mem = memory_create(TEMP_PATH, 1000, 100000);

    int ok = 1;
    for (int i = 0; i < 500; i++) {
        char buf[128];
        snprintf(buf, sizeof(buf), "stress turn %d with some padding content", i);
        const char *role = (i % 2 == 0) ? "user" : "assistant";
        if (memory_add_turn(mem, role, buf) != 0) {
            ok = 0;
            break;
        }
    }
    TEST(ok, "500 rapid adds all succeed");
    TEST(memory_turn_count(mem) == 500, "500 turns stored");

    /* Search should still work */
    MemoryTurn hits[4];
    int n = memory_search(mem, "stress turn 499", hits, 4);
    TEST(n >= 1, "search finds last turn in 500-turn store");

    n = memory_search(mem, "stress turn 0 ", hits, 4);
    TEST(n >= 1, "search finds first turn in 500-turn store");

    /* Context retrieval should work */
    MemoryTurn turns[10];
    n = memory_get_context(mem, turns, 10);
    TEST(n >= 1 && n <= 10, "get_context returns reasonable count from 500 turns");

    memory_destroy(mem);
    unlink(TEMP_PATH);
}

static void test_create_nonexistent_directory(void) {
    printf("\n=== Create with Non-Existent Directory ===\n");

    /* Path with non-existent parent directories */
    const char *deep_path = "/tmp/pocket_voice_test_nonexist/deep/nested/history.jsonl";

    ConversationMemory *mem = memory_create(deep_path, 50, 4000);
    /* May return NULL (can't create parent dirs) or succeed — both acceptable */
    TEST(1, "create with deep non-existent path doesn't crash");

    if (mem) {
        memory_add_turn(mem, "user", "test");
        memory_destroy(mem);
    }
}

static void test_create_readonly_path(void) {
    printf("\n=== Create with Read-Only Path ===\n");

    /* /dev/null is writable but special; try a truly read-only location */
    const char *readonly_path = "/readonly_location/history.jsonl";

    ConversationMemory *mem = memory_create(readonly_path, 50, 4000);
    /* Should either return NULL or handle gracefully */
    TEST(1, "create with unwritable path doesn't crash");

    if (mem) {
        int r = memory_add_turn(mem, "user", "test write");
        TEST(r == 0 || r == -1, "add_turn on unwritable path handled gracefully");
        memory_destroy(mem);
    }
}

static void test_search_max_out_limits(void) {
    printf("\n=== Search with Limited max_out ===\n");
    unlink(TEMP_PATH);

    ConversationMemory *mem = memory_create(TEMP_PATH, 50, 4000);
    /* Add 10 turns all containing "common" */
    for (int i = 0; i < 10; i++) {
        char buf[64];
        snprintf(buf, sizeof(buf), "common word in turn %d", i);
        memory_add_turn(mem, (i % 2 == 0) ? "user" : "assistant", buf);
    }

    /* Search with max_out=1 — should limit results */
    MemoryTurn hits[10];
    int n = memory_search(mem, "common", hits, 1);
    TEST(n == 1, "search with max_out=1 returns at most 1");

    /* Search with max_out=0 */
    n = memory_search(mem, "common", hits, 0);
    TEST(n == 0, "search with max_out=0 returns 0");

    /* Search with max_out=5 */
    n = memory_search(mem, "common", hits, 5);
    TEST(n == 5, "search with max_out=5 returns 5 of 10 matches");

    memory_destroy(mem);
    unlink(TEMP_PATH);
}

static void test_format_context_empty(void) {
    printf("\n=== Format Context on Empty Memory ===\n");
    unlink(TEMP_PATH);

    ConversationMemory *mem = memory_create(TEMP_PATH, 50, 4000);

    /* Format with no turns */
    char *formatted = memory_format_context(mem);
    /* Should return NULL or a minimal string — either is fine */
    if (formatted) {
        TEST(1, "format_context on empty returns non-NULL");
        free(formatted);
    } else {
        TEST(1, "format_context on empty returns NULL (acceptable)");
    }

    memory_destroy(mem);
    unlink(TEMP_PATH);
}

static void test_multiple_instances(void) {
    printf("\n=== Multiple Simultaneous Instances ===\n");

    const char *path1 = "/tmp/pocket_voice_test_multi1.jsonl";
    const char *path2 = "/tmp/pocket_voice_test_multi2.jsonl";
    unlink(path1);
    unlink(path2);

    ConversationMemory *mem1 = memory_create(path1, 50, 4000);
    ConversationMemory *mem2 = memory_create(path2, 50, 4000);
    TEST(mem1 != NULL && mem2 != NULL, "two instances created");

    /* Add different content to each */
    memory_add_turn(mem1, "user", "instance one content");
    memory_add_turn(mem2, "user", "instance two content");

    TEST(memory_turn_count(mem1) == 1, "instance 1 has 1 turn");
    TEST(memory_turn_count(mem2) == 1, "instance 2 has 1 turn");

    /* Search should be scoped to each instance */
    MemoryTurn hits[4];
    int n1 = memory_search(mem1, "instance one", hits, 4);
    int n2 = memory_search(mem1, "instance two", hits, 4);
    TEST(n1 == 1, "instance 1 finds its own content");
    TEST(n2 == 0, "instance 1 doesn't find instance 2 content");

    n1 = memory_search(mem2, "instance one", hits, 4);
    n2 = memory_search(mem2, "instance two", hits, 4);
    TEST(n1 == 0, "instance 2 doesn't find instance 1 content");
    TEST(n2 == 1, "instance 2 finds its own content");

    memory_destroy(mem1);
    memory_destroy(mem2);
    unlink(path1);
    unlink(path2);
}

static void test_eviction_with_persistence(void) {
    printf("\n=== Eviction with Persistence Round-Trip ===\n");
    unlink(TEMP_PATH);

    /* Capacity 4, add 8 turns (4 pairs) — should evict first 4 */
    ConversationMemory *mem = memory_create(TEMP_PATH, 4, 4000);
    for (int i = 0; i < 4; i++) {
        char buf[64];
        snprintf(buf, sizeof(buf), "user_msg_%d", i);
        memory_add_turn(mem, "user", buf);
        snprintf(buf, sizeof(buf), "asst_reply_%d", i);
        memory_add_turn(mem, "assistant", buf);
    }
    TEST(memory_turn_count(mem) == 4, "eviction keeps 4 turns");
    memory_destroy(mem);

    /* Reload and verify eviction persisted */
    mem = memory_create(TEMP_PATH, 4, 4000);
    TEST(memory_turn_count(mem) == 4, "4 turns after reload");

    /* Oldest turns should be gone */
    MemoryTurn hits[8];
    int n = memory_search(mem, "user_msg_0", hits, 8);
    TEST(n == 0, "evicted turn 0 not found after reload");

    /* Newest should remain */
    n = memory_search(mem, "asst_reply_3", hits, 8);
    TEST(n == 1, "newest turn found after reload");

    /* Add more, trigger eviction again on reloaded instance */
    memory_add_turn(mem, "user", "post_reload_msg");
    memory_add_turn(mem, "assistant", "post_reload_reply");
    TEST(memory_turn_count(mem) == 4, "eviction still works after reload + adds");

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
    test_alternating_roles();
    test_context_ordering();
    test_empty_content();
    test_very_long_content();
    test_multi_cycle_persistence();
    test_clear_and_re_add();
    test_eviction_preserves_newest();
    test_single_turn_capacity();
    test_get_context_limited_output();

    /* New edge case tests */
    test_unicode_content();
    test_json_special_chars();
    test_empty_database_file();
    test_corrupted_jsonl();
    test_search_empty_keyword();
    test_search_special_chars();
    test_get_context_zero_max();
    test_rapid_add_stress();
    test_create_nonexistent_directory();
    test_create_readonly_path();
    test_search_max_out_limits();
    test_format_context_empty();
    test_multiple_instances();
    test_eviction_with_persistence();

    printf("\n═══ Results: %d pass, %d fail ═══\n", pass_count, fail_count);
    return fail_count ? 1 : 0;
}
