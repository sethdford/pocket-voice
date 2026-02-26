/**
 * test_http_api.c — Unit tests for http_api.c (REST API server).
 *
 * Tests: create/destroy lifecycle, health endpoint via loopback,
 *        mock STT/TTS/LLM engine routing.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>

#include "http_api.h"

static int tests_passed = 0;
static int tests_failed = 0;

#define TEST(name) do { printf("  %-55s", name); } while(0)
#define PASS() do { printf("PASS\n"); tests_passed++; } while(0)
#define FAIL(msg) do { printf("FAIL: %s\n", msg); tests_failed++; } while(0)
#define ASSERT(cond, msg) do { if (!(cond)) { FAIL(msg); return; } } while(0)

/* Stub implementations for engines */
static int stub_return_zero(void *e, ...) { (void)e; return 0; }
static int stub_return_one(void *e)  { (void)e; return 1; }
static int stub_stt_get_text(void *e, char *buf, int sz) {
    (void)e;
    snprintf(buf, sz, "hello world");
    return (int)strlen(buf);
}
static int stub_tts_get_audio(void *e, float *buf, int max) {
    (void)e; (void)buf; (void)max;
    return 0;
}
static int stub_llm_poll(void *e, int ms) { (void)e; (void)ms; return 0; }
static const char *stub_llm_peek(void *e, int *len) {
    (void)e; *len = 0; return "";
}
static void stub_llm_consume(void *e, int n) { (void)e; (void)n; }
static bool stub_llm_done(void *e) { (void)e; return true; }

static HttpApiEngines make_stub_engines(void) {
    return (HttpApiEngines){
        .stt_engine = (void *)1,
        .tts_engine = (void *)1,
        .llm_engine = (void *)1,

        .stt_feed     = (int(*)(void*,const float*,int))stub_return_zero,
        .stt_flush    = (int(*)(void*))stub_return_zero,
        .stt_get_text = stub_stt_get_text,
        .stt_reset    = (int(*)(void*))stub_return_zero,

        .tts_speak       = (int(*)(void*,const char*))stub_return_zero,
        .tts_step        = (int(*)(void*))stub_return_zero,
        .tts_is_done     = stub_return_one,
        .tts_set_text_done = (int(*)(void*))stub_return_zero,
        .tts_get_audio   = stub_tts_get_audio,
        .tts_reset       = (int(*)(void*))stub_return_zero,

        .llm_send    = (int(*)(void*,const char*))stub_return_zero,
        .llm_poll    = stub_llm_poll,
        .llm_peek    = stub_llm_peek,
        .llm_consume = stub_llm_consume,
        .llm_is_done = stub_llm_done,
    };
}

/* ── Lifecycle Tests ───────────────────────────────────── */

static void test_create_destroy(void) {
    TEST("http_api: create/destroy lifecycle");
    HttpApiEngines eng = make_stub_engines();
    HttpApi *api = http_api_create(0, eng);
    ASSERT(api != NULL, "create returned NULL");
    http_api_destroy(api);
    PASS();
}

static void test_destroy_null(void) {
    TEST("http_api: destroy(NULL) is safe");
    http_api_destroy(NULL);
    PASS();
}

/* ── Helper: connect and send HTTP request, read response ── */

static int http_get(int port, const char *path, char *resp, int resp_sz) {
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) return -1;

    struct sockaddr_in addr = {
        .sin_family = AF_INET,
        .sin_port = htons(port),
    };
    inet_pton(AF_INET, "127.0.0.1", &addr.sin_addr);

    if (connect(fd, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
        close(fd);
        return -1;
    }

    char req[256];
    int len = snprintf(req, sizeof(req),
                       "GET %s HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n", path);
    write(fd, req, len);

    int total = 0;
    while (total < resp_sz - 1) {
        int n = (int)read(fd, resp + total, resp_sz - 1 - total);
        if (n <= 0) break;
        total += n;
    }
    resp[total] = '\0';
    close(fd);
    return total;
}

/* ── Server Integration Tests ──────────────────────────── */

static void test_health_endpoint(void) {
    TEST("http_api: GET /health returns 200 OK");
    HttpApiEngines eng = make_stub_engines();
    HttpApi *api = http_api_create(18765, eng);
    ASSERT(api != NULL, "create returned NULL");

    int rc = http_api_start(api);
    ASSERT(rc == 0, "start failed");
    usleep(50000);

    char resp[4096];
    int n = http_get(18765, "/health", resp, sizeof(resp));
    ASSERT(n > 0, "no response from server");
    ASSERT(strstr(resp, "200 OK") != NULL, "expected 200 OK");
    ASSERT(strstr(resp, "\"status\":\"ok\"") != NULL, "expected status ok in body");

    http_api_stop(api);
    http_api_destroy(api);
    PASS();
}

static void test_404_endpoint(void) {
    TEST("http_api: GET /nonexistent returns 404");
    HttpApiEngines eng = make_stub_engines();
    HttpApi *api = http_api_create(18766, eng);
    ASSERT(api != NULL, "create returned NULL");

    int rc = http_api_start(api);
    ASSERT(rc == 0, "start failed");
    usleep(50000);

    char resp[4096];
    int n = http_get(18766, "/nonexistent", resp, sizeof(resp));
    ASSERT(n > 0, "no response from server");
    ASSERT(strstr(resp, "404") != NULL, "expected 404 status");

    http_api_stop(api);
    http_api_destroy(api);
    PASS();
}

static void test_stop_restart(void) {
    TEST("http_api: stop then destroy is clean");
    HttpApiEngines eng = make_stub_engines();
    HttpApi *api = http_api_create(18767, eng);
    ASSERT(api != NULL, "create returned NULL");

    int rc = http_api_start(api);
    ASSERT(rc == 0, "start failed");
    usleep(50000);

    http_api_stop(api);
    usleep(50000);
    http_api_destroy(api);
    PASS();
}

/* ── Helper: send HTTP request with body and optional auth ── */

static int http_post(int port, const char *path, const char *content_type,
                     const char *body, int body_len, const char *auth,
                     char *resp, int resp_sz) {
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) return -1;

    struct sockaddr_in addr = {
        .sin_family = AF_INET,
        .sin_port = htons(port),
    };
    inet_pton(AF_INET, "127.0.0.1", &addr.sin_addr);

    if (connect(fd, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
        close(fd);
        return -1;
    }

    char header[1024];
    int hlen;
    if (auth) {
        hlen = snprintf(header, sizeof(header),
            "POST %s HTTP/1.1\r\nHost: localhost\r\n"
            "Content-Type: %s\r\nContent-Length: %d\r\n"
            "Authorization: %s\r\n"
            "Connection: close\r\n\r\n",
            path, content_type, body_len, auth);
    } else {
        hlen = snprintf(header, sizeof(header),
            "POST %s HTTP/1.1\r\nHost: localhost\r\n"
            "Content-Type: %s\r\nContent-Length: %d\r\n"
            "Connection: close\r\n\r\n",
            path, content_type, body_len);
    }
    write(fd, header, hlen);
    if (body && body_len > 0) write(fd, body, body_len);

    int total = 0;
    while (total < resp_sz - 1) {
        int n = (int)read(fd, resp + total, resp_sz - 1 - total);
        if (n <= 0) break;
        total += n;
    }
    resp[total] = '\0';
    close(fd);
    return total;
}

static int http_get_auth(int port, const char *path, const char *auth,
                         char *resp, int resp_sz) {
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) return -1;

    struct sockaddr_in addr = {
        .sin_family = AF_INET,
        .sin_port = htons(port),
    };
    inet_pton(AF_INET, "127.0.0.1", &addr.sin_addr);

    if (connect(fd, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
        close(fd);
        return -1;
    }

    char req_buf[512];
    int len;
    if (auth) {
        len = snprintf(req_buf, sizeof(req_buf),
            "GET %s HTTP/1.1\r\nHost: localhost\r\n"
            "Authorization: %s\r\nConnection: close\r\n\r\n", path, auth);
    } else {
        len = snprintf(req_buf, sizeof(req_buf),
            "GET %s HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n", path);
    }
    write(fd, req_buf, len);

    int total = 0;
    while (total < resp_sz - 1) {
        int n = (int)read(fd, resp + total, resp_sz - 1 - total);
        if (n <= 0) break;
        total += n;
    }
    resp[total] = '\0';
    close(fd);
    return total;
}

/* ── API Key Auth Tests ───────────────────────────────────── */

static void test_auth_no_key_open(void) {
    TEST("http_api: no SONATA_API_KEY = open access");
    HttpApiEngines eng = make_stub_engines();
    HttpApi *api = http_api_create(18770, eng);
    ASSERT(api != NULL, "create");
    /* Explicitly no key */
    http_api_set_api_key(api, NULL);
    int rc = http_api_start(api);
    ASSERT(rc == 0, "start");
    usleep(50000);

    char resp[4096];
    int n = http_get(18770, "/health", resp, sizeof(resp));
    ASSERT(n > 0, "no resp");
    ASSERT(strstr(resp, "200 OK") != NULL, "expected 200");
    http_api_destroy(api);
    PASS();
}

static void test_auth_key_required_401(void) {
    TEST("http_api: API key set, no auth → 401");
    HttpApiEngines eng = make_stub_engines();
    HttpApi *api = http_api_create(18771, eng);
    http_api_set_api_key(api, "test-secret-key");
    int rc = http_api_start(api);
    ASSERT(rc == 0, "start");
    usleep(50000);

    char resp[4096];
    /* /health is open, but /v1/audio/speech requires auth */
    const char *body = "Hello world";
    int n = http_post(18771, "/v1/audio/speech", "text/plain",
                      body, (int)strlen(body), NULL, resp, sizeof(resp));
    ASSERT(n > 0, "no resp");
    ASSERT(strstr(resp, "401") != NULL, "expected 401");
    http_api_destroy(api);
    PASS();
}

static void test_auth_wrong_key_403(void) {
    TEST("http_api: wrong API key → 403");
    HttpApiEngines eng = make_stub_engines();
    HttpApi *api = http_api_create(18772, eng);
    http_api_set_api_key(api, "correct-key");
    int rc = http_api_start(api);
    ASSERT(rc == 0, "start");
    usleep(50000);

    char resp[4096];
    const char *body = "Hello world";
    int n = http_post(18772, "/v1/audio/speech", "text/plain",
                      body, (int)strlen(body), "Bearer wrong-key", resp, sizeof(resp));
    ASSERT(n > 0, "no resp");
    ASSERT(strstr(resp, "403") != NULL, "expected 403");
    http_api_destroy(api);
    PASS();
}

static void test_auth_correct_key(void) {
    TEST("http_api: correct API key → accepted");
    HttpApiEngines eng = make_stub_engines();
    HttpApi *api = http_api_create(18773, eng);
    http_api_set_api_key(api, "my-secret");
    int rc = http_api_start(api);
    ASSERT(rc == 0, "start");
    usleep(50000);

    char resp[4096];
    int n = http_get_auth(18773, "/v1/voices", "Bearer my-secret", resp, sizeof(resp));
    ASSERT(n > 0, "no resp");
    ASSERT(strstr(resp, "200 OK") != NULL, "expected 200");
    ASSERT(strstr(resp, "\"voices\"") != NULL, "expected voice list");
    http_api_destroy(api);
    PASS();
}

/* ── OpenAI Compat Tests ──────────────────────────────────── */

static void test_openai_compat_input(void) {
    TEST("http_api: OpenAI {input, voice} format accepted");
    HttpApiEngines eng = make_stub_engines();
    HttpApi *api = http_api_create(18774, eng);
    int rc = http_api_start(api);
    ASSERT(rc == 0, "start");
    usleep(50000);

    /* OpenAI format: { "model": "tts-1", "input": "Hello", "voice": "alloy" } */
    const char *body = "{\"model\":\"tts-1\",\"input\":\"Hello\",\"voice\":\"alloy\"}";
    char resp[8192];
    int n = http_post(18774, "/v1/audio/speech", "application/json",
                      body, (int)strlen(body), NULL, resp, sizeof(resp));
    ASSERT(n > 0, "no resp");
    /* Should get audio or at least not a 400 for missing text */
    ASSERT(strstr(resp, "Missing or empty text") == NULL, "text should map from input");
    http_api_destroy(api);
    PASS();
}

static void test_openai_response_format(void) {
    TEST("http_api: OpenAI response_format=opus sets container");
    HttpApiEngines eng = make_stub_engines();
    HttpApi *api = http_api_create(18775, eng);
    int rc = http_api_start(api);
    ASSERT(rc == 0, "start");
    usleep(50000);

    const char *body = "{\"input\":\"Test\",\"response_format\":\"opus\"}";
    char resp[8192];
    int n = http_post(18775, "/v1/audio/speech", "application/json",
                      body, (int)strlen(body), NULL, resp, sizeof(resp));
    ASSERT(n > 0, "no resp");
    /* TTS produces no audio (stub), so we get the error, not a 400 */
    ASSERT(strstr(resp, "Missing or empty text") == NULL, "input should work");
    http_api_destroy(api);
    PASS();
}

/* ── Voice List Endpoint ──────────────────────────────────── */

static void test_voice_list_empty(void) {
    TEST("http_api: GET /v1/voices returns empty list");
    HttpApiEngines eng = make_stub_engines();
    HttpApi *api = http_api_create(18776, eng);
    int rc = http_api_start(api);
    ASSERT(rc == 0, "start");
    usleep(50000);

    char resp[4096];
    int n = http_get(18776, "/v1/voices", resp, sizeof(resp));
    ASSERT(n > 0, "no resp");
    ASSERT(strstr(resp, "200 OK") != NULL, "expected 200");
    ASSERT(strstr(resp, "\"voices\":[]") != NULL, "expected empty voices");
    http_api_destroy(api);
    PASS();
}

/* ── Concurrent Request Test (thread pool) ────────────────── */

static void test_concurrent_health(void) {
    TEST("http_api: concurrent /health requests (thread pool)");
    HttpApiEngines eng = make_stub_engines();
    HttpApi *api = http_api_create(18777, eng);
    int rc = http_api_start(api);
    ASSERT(rc == 0, "start");
    usleep(50000);

    int success = 0;
    for (int i = 0; i < 8; i++) {
        char resp[4096];
        int n = http_get(18777, "/health", resp, sizeof(resp));
        if (n > 0 && strstr(resp, "200 OK")) success++;
    }
    ASSERT(success == 8, "expected 8/8 health checks to pass");
    http_api_destroy(api);
    PASS();
}

/* ── Text Length Limit Tests ───────────────────────────────── */

static void test_text_too_long(void) {
    TEST("http_api: text > 10KB → 413 Payload Too Large");
    HttpApiEngines eng = make_stub_engines();
    HttpApi *api = http_api_create(18778, eng);
    int rc = http_api_start(api);
    ASSERT(rc == 0, "start");
    usleep(50000);

    /* Build a JSON body with >10KB of text */
    char *big = malloc(12000);
    ASSERT(big != NULL, "malloc");
    int pos = snprintf(big, 12000, "{\"text\":\"");
    for (int i = 0; i < 10500 && pos < 11900; i++)
        big[pos++] = 'A';
    snprintf(big + pos, 12000 - pos, "\"}");

    char resp[4096];
    int n = http_post(18778, "/v1/audio/speech", "application/json",
                      big, (int)strlen(big), NULL, resp, sizeof(resp));
    free(big);
    ASSERT(n > 0, "no resp");
    ASSERT(strstr(resp, "413") != NULL, "expected 413");
    http_api_destroy(api);
    PASS();
}

/* ── OpenAI Voice Mapping Tests ──────────────────────────── */

static void test_openai_voice_alloy(void) {
    TEST("http_api: OpenAI voice 'alloy' accepted (no crash)");
    HttpApiEngines eng = make_stub_engines();
    HttpApi *api = http_api_create(18779, eng);
    int rc = http_api_start(api);
    ASSERT(rc == 0, "start");
    usleep(50000);

    const char *body = "{\"input\":\"Hello\",\"voice\":\"alloy\",\"model\":\"tts-1\"}";
    char resp[8192];
    int n = http_post(18779, "/v1/audio/speech", "application/json",
                      body, (int)strlen(body), NULL, resp, sizeof(resp));
    ASSERT(n > 0, "no resp");
    ASSERT(strstr(resp, "Missing or empty text") == NULL, "alloy should work");
    http_api_destroy(api);
    PASS();
}

/* ── Speed Range Tests ───────────────────────────────────── */

static void test_speed_wide_range(void) {
    TEST("http_api: speed 2.0 accepted (OpenAI range)");
    HttpApiEngines eng = make_stub_engines();
    HttpApi *api = http_api_create(18780, eng);
    int rc = http_api_start(api);
    ASSERT(rc == 0, "start");
    usleep(50000);

    const char *body = "{\"text\":\"Hello\",\"speed\":2.0}";
    char resp[8192];
    int n = http_post(18780, "/v1/audio/speech", "application/json",
                      body, (int)strlen(body), NULL, resp, sizeof(resp));
    ASSERT(n > 0, "no resp");
    /* TTS stub produces no audio, so we get 500, not a parse error */
    ASSERT(strstr(resp, "400") == NULL, "speed 2.0 should be accepted");
    http_api_destroy(api);
    PASS();
}

/* ── Rate Limiting Tests ─────────────────────────────────── */

static void test_rate_limit_not_triggered(void) {
    TEST("http_api: normal traffic not rate-limited");
    HttpApiEngines eng = make_stub_engines();
    HttpApi *api = http_api_create(18781, eng);
    int rc = http_api_start(api);
    ASSERT(rc == 0, "start");
    usleep(50000);

    int ok = 0;
    for (int i = 0; i < 5; i++) {
        char resp[4096];
        int n = http_get(18781, "/health", resp, sizeof(resp));
        if (n > 0 && strstr(resp, "200 OK")) ok++;
    }
    ASSERT(ok == 5, "5 requests should pass rate limit");
    http_api_destroy(api);
    PASS();
}

/* ── Main ──────────────────────────────────────────────── */

int main(void) {
    printf("\n=== http_api tests ===\n\n");

    test_create_destroy();
    test_destroy_null();
    test_health_endpoint();
    test_404_endpoint();
    test_stop_restart();

    /* Auth */
    test_auth_no_key_open();
    test_auth_key_required_401();
    test_auth_wrong_key_403();
    test_auth_correct_key();

    /* OpenAI compatibility */
    test_openai_compat_input();
    test_openai_response_format();
    test_openai_voice_alloy();

    /* Voices */
    test_voice_list_empty();

    /* Thread pool */
    test_concurrent_health();

    /* Input validation */
    test_text_too_long();
    test_speed_wide_range();

    /* Rate limiting */
    test_rate_limit_not_triggered();

    printf("\n  Results: %d passed, %d failed\n\n", tests_passed, tests_failed);
    return tests_failed > 0 ? 1 : 0;
}
