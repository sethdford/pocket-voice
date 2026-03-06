/**
 * test_streaming_llm.c — Unit tests for Gemini Live / OpenAI Realtime streaming LLM.
 *
 * Tests do NOT require real API keys. We exercise the public API without
 * establishing WebSocket connections.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "streaming_llm.h"

#define TEST(cond, msg) do { \
    if (!(cond)) { fprintf(stderr, "FAIL %s: %s\n", __func__, msg); return 0; } \
} while (0)
#define OK() return 1

static int test_create_destroy_gemini(void) {
    StreamingLLMConfig cfg = {
        .type = STREAMING_LLM_GEMINI_LIVE,
        .api_key = "test-key",
        .model = "gemini-2.0-flash-live",
        .system_prompt = "You are helpful.",
        .input_sample_rate = 16000,
        .output_sample_rate = 24000,
        .voice = NULL,
        .temperature = 0.8f,
    };
    StreamingLLM *llm = streaming_llm_create(&cfg);
    TEST(llm != NULL, "create");
    streaming_llm_destroy(llm);
    OK();
}

static int test_create_destroy_openai(void) {
    StreamingLLMConfig cfg = {
        .type = STREAMING_LLM_OPENAI_REALTIME,
        .api_key = "sk-test",
        .model = "gpt-4o-realtime-preview",
        .system_prompt = "Assistant.",
        .input_sample_rate = 24000,
        .output_sample_rate = 24000,
        .voice = "alloy",
        .temperature = 0.7f,
    };
    StreamingLLM *llm = streaming_llm_create(&cfg);
    TEST(llm != NULL, "create");
    streaming_llm_destroy(llm);
    OK();
}

static int test_create_null_config(void) {
    StreamingLLM *llm = streaming_llm_create(NULL);
    TEST(llm == NULL, "create with NULL config should return NULL");
    OK();
}

static int test_destroy_null(void) {
    streaming_llm_destroy(NULL);
    OK();
}

static int test_input_output_sample_rates(void) {
    StreamingLLMConfig cfg = {
        .type = STREAMING_LLM_GEMINI_LIVE,
        .api_key = "x",
        .input_sample_rate = 16000,
        .output_sample_rate = 24000,
    };
    StreamingLLM *llm = streaming_llm_create(&cfg);
    TEST(llm != NULL, "create");
    TEST(streaming_llm_input_sample_rate(llm) == 16000, "input rate");
    TEST(streaming_llm_output_sample_rate(llm) == 24000, "output rate");
    streaming_llm_destroy(llm);

    cfg.type = STREAMING_LLM_OPENAI_REALTIME;
    cfg.input_sample_rate = 24000;
    cfg.output_sample_rate = 24000;
    llm = streaming_llm_create(&cfg);
    TEST(llm != NULL, "create");
    TEST(streaming_llm_input_sample_rate(llm) == 24000, "input rate");
    TEST(streaming_llm_output_sample_rate(llm) == 24000, "output rate");
    streaming_llm_destroy(llm);
    OK();
}

static int test_server_vad(void) {
    StreamingLLMConfig cfg = {
        .type = STREAMING_LLM_GEMINI_LIVE,
        .api_key = "x",
    };
    StreamingLLM *llm = streaming_llm_create(&cfg);
    TEST(llm != NULL, "create");
    streaming_llm_set_server_vad(llm, true);
    streaming_llm_set_server_vad(llm, false);
    streaming_llm_destroy(llm);
    OK();
}

static int test_peek_text_empty(void) {
    StreamingLLMConfig cfg = {
        .type = STREAMING_LLM_GEMINI_LIVE,
        .api_key = "x",
    };
    StreamingLLM *llm = streaming_llm_create(&cfg);
    TEST(llm != NULL, "create");
    int len = -1;
    const char *p = streaming_llm_peek_text(llm, &len);
    TEST(p != NULL, "peek non-NULL");
    TEST(len == 0, "len 0 when empty");
    streaming_llm_destroy(llm);
    OK();
}

static int test_consume_text(void) {
    StreamingLLMConfig cfg = {
        .type = STREAMING_LLM_GEMINI_LIVE,
        .api_key = "x",
    };
    StreamingLLM *llm = streaming_llm_create(&cfg);
    TEST(llm != NULL, "create");
    streaming_llm_consume_text(llm, 0);
    streaming_llm_consume_text(llm, 100);
    streaming_llm_destroy(llm);
    OK();
}

static int test_audio_available_empty(void) {
    StreamingLLMConfig cfg = {
        .type = STREAMING_LLM_GEMINI_LIVE,
        .api_key = "x",
    };
    StreamingLLM *llm = streaming_llm_create(&cfg);
    TEST(llm != NULL, "create");
    TEST(streaming_llm_audio_available(llm) == 0, "no audio when empty");
    streaming_llm_destroy(llm);
    OK();
}

static int test_recv_audio_empty(void) {
    StreamingLLMConfig cfg = {
        .type = STREAMING_LLM_GEMINI_LIVE,
        .api_key = "x",
    };
    StreamingLLM *llm = streaming_llm_create(&cfg);
    float buf[128];
    int n = streaming_llm_recv_audio(llm, buf, 128);
    TEST(n == 0, "recv 0 when empty");
    streaming_llm_destroy(llm);
    OK();
}

static int test_send_audio_not_connected(void) {
    StreamingLLMConfig cfg = {
        .type = STREAMING_LLM_GEMINI_LIVE,
        .api_key = "x",
    };
    StreamingLLM *llm = streaming_llm_create(&cfg);
    float pcm[100] = {0};
    int r = streaming_llm_send_audio(llm, pcm, 100);
    TEST(r == -1, "send_audio returns -1 when not connected");
    streaming_llm_destroy(llm);
    OK();
}

static int test_send_text_not_connected(void) {
    StreamingLLMConfig cfg = {
        .type = STREAMING_LLM_GEMINI_LIVE,
        .api_key = "x",
    };
    StreamingLLM *llm = streaming_llm_create(&cfg);
    int r = streaming_llm_send_text(llm, "hello");
    TEST(r == -1, "send_text returns -1 when not connected");
    streaming_llm_destroy(llm);
    OK();
}

static int test_is_done(void) {
    StreamingLLMConfig cfg = {
        .type = STREAMING_LLM_GEMINI_LIVE,
        .api_key = "x",
    };
    StreamingLLM *llm = streaming_llm_create(&cfg);
    TEST(streaming_llm_is_done(llm) == false, "not done initially");
    streaming_llm_destroy(llm);
    OK();
}

static int test_has_error(void) {
    StreamingLLMConfig cfg = {
        .type = STREAMING_LLM_GEMINI_LIVE,
        .api_key = "x",
    };
    StreamingLLM *llm = streaming_llm_create(&cfg);
    TEST(streaming_llm_has_error(llm) == false, "no error initially");
    streaming_llm_destroy(llm);
    OK();
}

static int test_error_message_empty(void) {
    StreamingLLMConfig cfg = {
        .type = STREAMING_LLM_GEMINI_LIVE,
        .api_key = "x",
    };
    StreamingLLM *llm = streaming_llm_create(&cfg);
    const char *msg = streaming_llm_error_message(llm);
    TEST(msg != NULL, "error_message non-NULL");
    streaming_llm_destroy(llm);
    OK();
}

static int test_get_transcript_empty(void) {
    StreamingLLMConfig cfg = {
        .type = STREAMING_LLM_GEMINI_LIVE,
        .api_key = "x",
    };
    StreamingLLM *llm = streaming_llm_create(&cfg);
    const char *t = streaming_llm_get_transcript(llm);
    TEST(t != NULL, "transcript non-NULL");
    streaming_llm_destroy(llm);
    OK();
}

static int test_poll_not_connected(void) {
    StreamingLLMConfig cfg = {
        .type = STREAMING_LLM_GEMINI_LIVE,
        .api_key = "x",
    };
    StreamingLLM *llm = streaming_llm_create(&cfg);
    int n = streaming_llm_poll(llm, 0);
    TEST(n >= 0, "poll does not crash");
    streaming_llm_destroy(llm);
    OK();
}

static int test_cancel(void) {
    StreamingLLMConfig cfg = {
        .type = STREAMING_LLM_GEMINI_LIVE,
        .api_key = "x",
    };
    StreamingLLM *llm = streaming_llm_create(&cfg);
    streaming_llm_cancel(llm);
    streaming_llm_destroy(llm);
    OK();
}

static int test_end_turn(void) {
    StreamingLLMConfig cfg = {
        .type = STREAMING_LLM_OPENAI_REALTIME,
        .api_key = "x",
    };
    StreamingLLM *llm = streaming_llm_create(&cfg);
    streaming_llm_end_turn(llm);
    streaming_llm_destroy(llm);
    OK();
}

static int test_commit_turn(void) {
    StreamingLLMConfig cfg = {
        .type = STREAMING_LLM_GEMINI_LIVE,
        .api_key = "x",
    };
    StreamingLLM *llm = streaming_llm_create(&cfg);
    streaming_llm_commit_turn(llm, "user said this");
    streaming_llm_destroy(llm);
    OK();
}

static int test_set_tools(void) {
    StreamingLLMConfig cfg = {
        .type = STREAMING_LLM_GEMINI_LIVE,
        .api_key = "x",
    };
    StreamingLLM *llm = streaming_llm_create(&cfg);
    int r = streaming_llm_set_tools(llm, "[]");
    TEST(r == 0, "set_tools returns 0");
    r = streaming_llm_set_tools(llm, NULL);
    TEST(r == 0, "set_tools NULL");
    streaming_llm_destroy(llm);
    OK();
}

static int test_disconnect_not_connected(void) {
    StreamingLLMConfig cfg = {
        .type = STREAMING_LLM_GEMINI_LIVE,
        .api_key = "x",
    };
    StreamingLLM *llm = streaming_llm_create(&cfg);
    streaming_llm_disconnect(llm);
    TEST(streaming_llm_is_connected(llm) == 0, "not connected");
    streaming_llm_destroy(llm);
    OK();
}

static int test_null_llm_apis(void) {
    int len;
    TEST(streaming_llm_peek_text(NULL, &len) != NULL, "peek NULL returns non-NULL");
    streaming_llm_consume_text(NULL, 0);
    TEST(streaming_llm_audio_available(NULL) == 0, "audio_available NULL");
    TEST(streaming_llm_recv_audio(NULL, NULL, 0) == 0, "recv NULL");
    TEST(streaming_llm_send_audio(NULL, NULL, 0) == -1, "send_audio NULL");
    TEST(streaming_llm_send_text(NULL, "x") == -1, "send_text NULL");
    TEST(streaming_llm_is_done(NULL) == false, "is_done NULL");
    TEST(streaming_llm_has_error(NULL) == false, "has_error NULL");
    TEST(streaming_llm_error_message(NULL) != NULL, "error_message NULL");
    TEST(streaming_llm_get_transcript(NULL) != NULL, "transcript NULL");
    TEST(streaming_llm_input_sample_rate(NULL) == 16000, "input_rate NULL default");
    TEST(streaming_llm_output_sample_rate(NULL) == 24000, "output_rate NULL default");
    streaming_llm_cancel(NULL);
    streaming_llm_end_turn(NULL);
    streaming_llm_commit_turn(NULL, "x");
    streaming_llm_disconnect(NULL);
    streaming_llm_set_server_vad(NULL, true);
    streaming_llm_set_tools(NULL, "[]");
    streaming_llm_poll(NULL, 0);
    OK();
}

int main(void) {
    static int (*tests[])(void) = {
        test_create_destroy_gemini,
        test_create_destroy_openai,
        test_create_null_config,
        test_destroy_null,
        test_input_output_sample_rates,
        test_server_vad,
        test_peek_text_empty,
        test_consume_text,
        test_audio_available_empty,
        test_recv_audio_empty,
        test_send_audio_not_connected,
        test_send_text_not_connected,
        test_is_done,
        test_has_error,
        test_error_message_empty,
        test_get_transcript_empty,
        test_poll_not_connected,
        test_cancel,
        test_end_turn,
        test_commit_turn,
        test_set_tools,
        test_disconnect_not_connected,
        test_null_llm_apis,
    };
    int n = (int)(sizeof(tests) / sizeof(tests[0]));
    int passed = 0;
    for (int i = 0; i < n; i++) {
        if (tests[i]()) passed++;
    }
    fprintf(stderr, "streaming_llm: %d/%d tests passed\n", passed, n);
    return (passed == n) ? 0 : 1;
}
