/**
 * test_web_remote.c — Tests for web_remote.c (WebSocket audio server).
 *
 * Tests: create/destroy lifecycle, port assignment, HTML page generation,
 *        configuration validation (sample rate, port), NULL safety,
 *        connection state tracking, audio send with no client.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>

#include "web_remote.h"

static int tests_passed = 0;
static int tests_failed = 0;

#define TEST(name) do { printf("  %-55s", name); } while(0)
#define PASS() do { printf("PASS\n"); tests_passed++; } while(0)
#define FAIL(msg) do { printf("FAIL: %s\n", msg); tests_failed++; } while(0)
#define ASSERT(cond, msg) do { if (!(cond)) { FAIL(msg); return; } } while(0)

/* Stub audio callback */
static int audio_cb_called = 0;
static void stub_audio_cb(void *ctx, const float *pcm, int n_samples) {
    (void)ctx; (void)pcm; (void)n_samples;
    audio_cb_called++;
}

/* Stub disconnect callback */
static int disconnect_cb_called = 0;
static void stub_disconnect_cb(void *ctx) {
    (void)ctx;
    disconnect_cb_called++;
}

/* ── NULL Safety Tests ─────────────────────────────────── */

static void test_destroy_null(void) {
    TEST("web_remote: destroy(NULL) is safe");
    web_remote_destroy(NULL);
    PASS();
}

static void test_port_null(void) {
    TEST("web_remote: web_remote_port(NULL) returns 0");
    int port = web_remote_port(NULL);
    ASSERT(port == 0, "expected 0 for NULL");
    PASS();
}

static void test_connected_null(void) {
    TEST("web_remote: web_remote_connected(NULL) returns 0");
    int connected = web_remote_connected(NULL);
    ASSERT(connected == 0, "expected 0 for NULL");
    PASS();
}

static void test_send_audio_null(void) {
    TEST("web_remote: send_audio(NULL, ...) returns -1");
    float pcm[160];
    memset(pcm, 0, sizeof(pcm));
    int rc = web_remote_send_audio(NULL, pcm, 160);
    ASSERT(rc == -1, "expected -1 for NULL");
    PASS();
}

/* ── Create/Destroy Lifecycle Tests ────────────────────── */

static void test_create_destroy_basic(void) {
    TEST("web_remote: create and destroy lifecycle");
    WebRemote *wr = web_remote_create(0, 16000, stub_audio_cb, stub_disconnect_cb, NULL);
    ASSERT(wr != NULL, "create returned NULL");

    int port = web_remote_port(wr);
    ASSERT(port > 0, "expected positive port number");

    web_remote_destroy(wr);
    PASS();
}

static void test_create_with_specific_port(void) {
    TEST("web_remote: create with specific port");
    WebRemote *wr = web_remote_create(18990, 16000, stub_audio_cb, NULL, NULL);
    ASSERT(wr != NULL, "create returned NULL");

    int port = web_remote_port(wr);
    ASSERT(port == 18990, "expected port 18990");

    web_remote_destroy(wr);
    PASS();
}

static void test_create_port_zero_assigns(void) {
    TEST("web_remote: port 0 gets assigned ephemeral port");
    WebRemote *wr = web_remote_create(0, 16000, stub_audio_cb, NULL, NULL);
    ASSERT(wr != NULL, "create returned NULL");

    int port = web_remote_port(wr);
    ASSERT(port > 0, "expected ephemeral port > 0");
    ASSERT(port != 0, "should not remain 0");

    web_remote_destroy(wr);
    PASS();
}

static void test_create_null_callbacks(void) {
    TEST("web_remote: create with NULL callbacks succeeds");
    /* NULL disconnect_cb is explicitly allowed */
    WebRemote *wr = web_remote_create(0, 16000, stub_audio_cb, NULL, NULL);
    ASSERT(wr != NULL, "create with NULL disconnect_cb returned NULL");
    web_remote_destroy(wr);
    PASS();
}

static void test_create_with_user_ctx(void) {
    TEST("web_remote: user_ctx passed through");
    int ctx_val = 42;
    WebRemote *wr = web_remote_create(0, 16000, stub_audio_cb, stub_disconnect_cb, &ctx_val);
    ASSERT(wr != NULL, "create returned NULL");
    web_remote_destroy(wr);
    PASS();
}

/* ── Configuration Tests ──────────────────────────────── */

static void test_sample_rate_16000(void) {
    TEST("web_remote: sample_rate 16000 (STT default)");
    WebRemote *wr = web_remote_create(0, 16000, stub_audio_cb, NULL, NULL);
    ASSERT(wr != NULL, "create with 16000 Hz failed");
    web_remote_destroy(wr);
    PASS();
}

static void test_sample_rate_48000(void) {
    TEST("web_remote: sample_rate 48000 (high quality)");
    WebRemote *wr = web_remote_create(0, 48000, stub_audio_cb, NULL, NULL);
    ASSERT(wr != NULL, "create with 48000 Hz failed");
    web_remote_destroy(wr);
    PASS();
}

static void test_sample_rate_8000(void) {
    TEST("web_remote: sample_rate 8000 (telephony)");
    WebRemote *wr = web_remote_create(0, 8000, stub_audio_cb, NULL, NULL);
    ASSERT(wr != NULL, "create with 8000 Hz failed");
    web_remote_destroy(wr);
    PASS();
}

/* ── Connection State Tests ───────────────────────────── */

static void test_not_connected_initially(void) {
    TEST("web_remote: not connected after create");
    WebRemote *wr = web_remote_create(0, 16000, stub_audio_cb, NULL, NULL);
    ASSERT(wr != NULL, "create returned NULL");

    int connected = web_remote_connected(wr);
    ASSERT(connected == 0, "should not be connected initially");

    web_remote_destroy(wr);
    PASS();
}

static void test_send_audio_no_client(void) {
    TEST("web_remote: send_audio with no client returns -1");
    WebRemote *wr = web_remote_create(0, 16000, stub_audio_cb, NULL, NULL);
    ASSERT(wr != NULL, "create returned NULL");

    float pcm[160];
    memset(pcm, 0, sizeof(pcm));
    int rc = web_remote_send_audio(wr, pcm, 160);
    ASSERT(rc == -1, "expected -1 when no client connected");

    web_remote_destroy(wr);
    PASS();
}

/* ── HTML Page Serving Tests ──────────────────────────── */

static int fetch_html(int port, char *buf, int buf_sz) {
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

    const char *req = "GET / HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n";
    write(fd, req, strlen(req));

    int total = 0;
    while (total < buf_sz - 1) {
        int n = (int)read(fd, buf + total, buf_sz - 1 - total);
        if (n <= 0) break;
        total += n;
    }
    buf[total] = '\0';
    close(fd);
    return total;
}

static void test_html_page_served(void) {
    TEST("web_remote: GET / serves HTML page with 200");
    WebRemote *wr = web_remote_create(0, 16000, stub_audio_cb, NULL, NULL);
    ASSERT(wr != NULL, "create returned NULL");

    int port = web_remote_port(wr);
    usleep(50000); /* Let server thread start */

    char buf[65536];
    int n = fetch_html(port, buf, sizeof(buf));
    ASSERT(n > 0, "no response from server");
    ASSERT(strstr(buf, "200 OK") != NULL, "expected 200 OK");
    ASSERT(strstr(buf, "text/html") != NULL, "expected text/html content type");

    web_remote_destroy(wr);
    PASS();
}

static void test_html_contains_doctype(void) {
    TEST("web_remote: HTML contains <!DOCTYPE html>");
    WebRemote *wr = web_remote_create(0, 16000, stub_audio_cb, NULL, NULL);
    ASSERT(wr != NULL, "create returned NULL");

    int port = web_remote_port(wr);
    usleep(50000);

    char buf[65536];
    int n = fetch_html(port, buf, sizeof(buf));
    ASSERT(n > 0, "no response");
    ASSERT(strstr(buf, "<!DOCTYPE html>") != NULL, "expected DOCTYPE");

    web_remote_destroy(wr);
    PASS();
}

static void test_html_contains_javascript(void) {
    TEST("web_remote: HTML contains WebSocket JS code");
    WebRemote *wr = web_remote_create(0, 16000, stub_audio_cb, NULL, NULL);
    ASSERT(wr != NULL, "create returned NULL");

    int port = web_remote_port(wr);
    usleep(50000);

    char buf[65536];
    int n = fetch_html(port, buf, sizeof(buf));
    ASSERT(n > 0, "no response");

    /* Should contain WebSocket connection code */
    ASSERT(strstr(buf, "WebSocket") != NULL, "expected WebSocket JS");
    ASSERT(strstr(buf, "<script>") != NULL, "expected script tag");

    web_remote_destroy(wr);
    PASS();
}

static void test_html_contains_mic_button(void) {
    TEST("web_remote: HTML contains mic button element");
    WebRemote *wr = web_remote_create(0, 16000, stub_audio_cb, NULL, NULL);
    ASSERT(wr != NULL, "create returned NULL");

    int port = web_remote_port(wr);
    usleep(50000);

    char buf[65536];
    int n = fetch_html(port, buf, sizeof(buf));
    ASSERT(n > 0, "no response");
    ASSERT(strstr(buf, "mic-btn") != NULL, "expected mic button element");

    web_remote_destroy(wr);
    PASS();
}

static void test_html_contains_audio_worklet(void) {
    TEST("web_remote: HTML contains AudioWorklet code");
    WebRemote *wr = web_remote_create(0, 16000, stub_audio_cb, NULL, NULL);
    ASSERT(wr != NULL, "create returned NULL");

    int port = web_remote_port(wr);
    usleep(50000);

    char buf[65536];
    int n = fetch_html(port, buf, sizeof(buf));
    ASSERT(n > 0, "no response");
    ASSERT(strstr(buf, "AudioWorklet") != NULL || strstr(buf, "audioWorklet") != NULL,
           "expected AudioWorklet code");

    web_remote_destroy(wr);
    PASS();
}

static void test_html_contains_pocket_voice_title(void) {
    TEST("web_remote: HTML title contains pocket-voice");
    WebRemote *wr = web_remote_create(0, 16000, stub_audio_cb, NULL, NULL);
    ASSERT(wr != NULL, "create returned NULL");

    int port = web_remote_port(wr);
    usleep(50000);

    char buf[65536];
    int n = fetch_html(port, buf, sizeof(buf));
    ASSERT(n > 0, "no response");
    ASSERT(strstr(buf, "pocket-voice") != NULL, "expected pocket-voice in page");

    web_remote_destroy(wr);
    PASS();
}

static void test_html_ws_path(void) {
    TEST("web_remote: HTML references /ws path for WebSocket");
    WebRemote *wr = web_remote_create(0, 16000, stub_audio_cb, NULL, NULL);
    ASSERT(wr != NULL, "create returned NULL");

    int port = web_remote_port(wr);
    usleep(50000);

    char buf[65536];
    int n = fetch_html(port, buf, sizeof(buf));
    ASSERT(n > 0, "no response");
    ASSERT(strstr(buf, "/ws") != NULL, "expected /ws path in JS");

    web_remote_destroy(wr);
    PASS();
}

/* ── Non-WebSocket Request Handling ───────────────────── */

static void test_non_ws_get_closed(void) {
    TEST("web_remote: non-upgrade GET / returns HTML and closes");
    WebRemote *wr = web_remote_create(0, 16000, stub_audio_cb, NULL, NULL);
    ASSERT(wr != NULL, "create returned NULL");

    int port = web_remote_port(wr);
    usleep(50000);

    /* After GET /, the server should close the connection */
    char buf[65536];
    int n = fetch_html(port, buf, sizeof(buf));
    ASSERT(n > 0, "no response");
    ASSERT(strstr(buf, "Connection: close") != NULL, "expected Connection: close");

    /* Server should still be running for new connections */
    char buf2[65536];
    int n2 = fetch_html(port, buf2, sizeof(buf2));
    ASSERT(n2 > 0, "server should still accept connections");

    web_remote_destroy(wr);
    PASS();
}

/* ── Multiple Create/Destroy Cycles ──────────────────── */

static void test_multiple_create_destroy(void) {
    TEST("web_remote: multiple create/destroy cycles");
    for (int i = 0; i < 3; i++) {
        WebRemote *wr = web_remote_create(0, 16000, stub_audio_cb, NULL, NULL);
        ASSERT(wr != NULL, "create returned NULL on iteration");
        ASSERT(web_remote_port(wr) > 0, "invalid port");
        ASSERT(web_remote_connected(wr) == 0, "should start disconnected");
        web_remote_destroy(wr);
    }
    PASS();
}

/* ── Main ──────────────────────────────────────────────── */

int main(void) {
    printf("\n=== web_remote tests ===\n\n");

    printf("[NULL Safety]\n");
    test_destroy_null();
    test_port_null();
    test_connected_null();
    test_send_audio_null();

    printf("\n[Create/Destroy Lifecycle]\n");
    test_create_destroy_basic();
    test_create_with_specific_port();
    test_create_port_zero_assigns();
    test_create_null_callbacks();
    test_create_with_user_ctx();
    test_multiple_create_destroy();

    printf("\n[Configuration]\n");
    test_sample_rate_16000();
    test_sample_rate_48000();
    test_sample_rate_8000();

    printf("\n[Connection State]\n");
    test_not_connected_initially();
    test_send_audio_no_client();

    printf("\n[HTML Page]\n");
    test_html_page_served();
    test_html_contains_doctype();
    test_html_contains_javascript();
    test_html_contains_mic_button();
    test_html_contains_audio_worklet();
    test_html_contains_pocket_voice_title();
    test_html_ws_path();

    printf("\n[Request Handling]\n");
    test_non_ws_get_closed();

    printf("\n  Results: %d passed, %d failed\n\n", tests_passed, tests_failed);
    return tests_failed > 0 ? 1 : 0;
}
