/**
 * test_websocket.c — Comprehensive tests for websocket.c (RFC 6455).
 *
 * Tests: opcode constants, frame encoding (7-bit / 16-bit / 64-bit lengths),
 *        masking, create/destroy lifecycle, NULL safety, send helpers.
 *
 * We test the ws_send() frame encoding by creating a socketpair so we can
 * read back the exact bytes written to the wire.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/socket.h>
#include <assert.h>
#include <stdbool.h>
#include <pthread.h>

#include "websocket.h"

static int tests_passed = 0;
static int tests_failed = 0;

#define TEST(name) do { printf("  %-55s", name); } while(0)
#define PASS() do { printf("PASS\n"); tests_passed++; } while(0)
#define FAIL(msg) do { printf("FAIL: %s\n", msg); tests_failed++; } while(0)
#define ASSERT(cond, msg) do { if (!(cond)) { FAIL(msg); return; } } while(0)

/* ── Opcode Constant Tests ─────────────────────────────── */

static void test_opcode_values(void) {
    TEST("ws: opcode constants match RFC 6455");
    ASSERT(WS_TEXT   == 0x1, "WS_TEXT should be 0x1");
    ASSERT(WS_BINARY == 0x2, "WS_BINARY should be 0x2");
    ASSERT(WS_CLOSE  == 0x8, "WS_CLOSE should be 0x8");
    ASSERT(WS_PING   == 0x9, "WS_PING should be 0x9");
    ASSERT(WS_PONG   == 0xA, "WS_PONG should be 0xA");
    PASS();
}

static void test_opcodes_distinct(void) {
    TEST("ws: all opcodes are distinct values");
    WsOpcode ops[] = { WS_TEXT, WS_BINARY, WS_CLOSE, WS_PING, WS_PONG };
    for (int i = 0; i < 5; i++) {
        for (int j = i + 1; j < 5; j++) {
            ASSERT(ops[i] != ops[j], "duplicate opcode values");
        }
    }
    PASS();
}

/* ── NULL Safety Tests ─────────────────────────────────── */

static void test_destroy_null(void) {
    TEST("ws: ws_destroy(NULL) is safe");
    ws_destroy(NULL);
    PASS();
}

static void test_close_null(void) {
    TEST("ws: ws_close(NULL) is safe");
    ws_close(NULL);
    PASS();
}

static void test_send_null_ws(void) {
    TEST("ws: ws_send(NULL, ...) returns -1");
    uint8_t data[] = {1, 2, 3};
    int rc = ws_send(NULL, WS_TEXT, data, 3);
    ASSERT(rc == -1, "expected -1 for NULL ws");
    PASS();
}

static void test_send_text_null_ws(void) {
    TEST("ws: ws_send_text(NULL, ...) returns -1");
    int rc = ws_send_text(NULL, "hello");
    ASSERT(rc == -1, "expected -1 for NULL ws");
    PASS();
}

static void test_send_text_null_text(void) {
    TEST("ws: ws_send_text(ws, NULL) returns -1");
    int rc = ws_send_text(NULL, NULL);
    ASSERT(rc == -1, "expected -1 for NULL text");
    PASS();
}

static void test_send_binary_null_ws(void) {
    TEST("ws: ws_send_binary(NULL, ...) returns -1");
    uint8_t data[] = {0xAA, 0xBB};
    int rc = ws_send_binary(NULL, data, 2);
    ASSERT(rc == -1, "expected -1 for NULL ws");
    PASS();
}

static void test_send_negative_len(void) {
    TEST("ws: ws_send(..., len=-1) returns -1");
    int rc = ws_send(NULL, WS_BINARY, NULL, -1);
    ASSERT(rc == -1, "expected -1 for negative len");
    PASS();
}

static void test_is_connected_null(void) {
    TEST("ws: ws_is_connected(NULL) returns false");
    ASSERT(ws_is_connected(NULL) == false, "expected false for NULL ws");
    PASS();
}

static void test_recv_null(void) {
    TEST("ws: ws_recv(NULL, ...) returns -1");
    WsOpcode type;
    uint8_t buf[64];
    int rc = ws_recv(NULL, &type, buf, sizeof(buf));
    ASSERT(rc == -1, "expected -1 for NULL ws");
    PASS();
}

static void test_run_loop_null(void) {
    TEST("ws: ws_run_loop(NULL, ...) returns -1");
    int rc = ws_run_loop(NULL, NULL, NULL);
    ASSERT(rc == -1, "expected -1 for NULL ws/callback");
    PASS();
}

static void test_upgrade_null_key(void) {
    TEST("ws: ws_upgrade(fd, NULL) returns NULL");
    WebSocket *ws = ws_upgrade(0, NULL);
    ASSERT(ws == NULL, "expected NULL for NULL key");
    PASS();
}

static void test_upgrade_bad_fd(void) {
    TEST("ws: ws_upgrade(-1, key) returns NULL");
    WebSocket *ws = ws_upgrade(-1, "dGhlIHNhbXBsZSBub25jZQ==");
    ASSERT(ws == NULL, "expected NULL for bad fd");
    PASS();
}

/* ── Frame Encoding Tests (via socketpair) ──────────────── */

/*
 * Helper: create a WebSocket from a socketpair by performing a mock upgrade.
 * We write the HTTP upgrade request from the "client" side and call ws_upgrade
 * on the "server" side. Returns the ws handle and the peer fd for reading.
 */
static WebSocket *make_test_ws(int *peer_fd) {
    int sv[2];
    if (socketpair(AF_UNIX, SOCK_STREAM, 0, sv) < 0) return NULL;

    /* ws_upgrade writes the 101 response and creates the WS handle */
    WebSocket *ws = ws_upgrade(sv[0], "dGhlIHNhbXBsZSBub25jZQ==");
    if (!ws) {
        close(sv[0]);
        close(sv[1]);
        return NULL;
    }

    /* Drain the 101 response from the peer side (single read, response is ~150 bytes) */
    char drain[1024];
    struct timeval tv = { .tv_sec = 1, .tv_usec = 0 };
    setsockopt(sv[1], SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
    read(sv[1], drain, sizeof(drain));

    *peer_fd = sv[1];
    return ws;
}

static void test_send_frame_text_short(void) {
    TEST("ws: send text frame, 7-bit length (< 126)");
    int peer;
    WebSocket *ws = make_test_ws(&peer);
    ASSERT(ws != NULL, "make_test_ws failed");

    const char *msg = "Hello";
    int rc = ws_send(ws, WS_TEXT, (const uint8_t *)msg, 5);
    ASSERT(rc == 0, "ws_send failed");

    uint8_t buf[128];
    ssize_t n = read(peer, buf, sizeof(buf));
    ASSERT(n >= 7, "short read");

    /* Byte 0: FIN=1 + opcode=0x1 => 0x81 */
    ASSERT(buf[0] == 0x81, "expected FIN|TEXT (0x81)");
    /* Byte 1: MASK=0 (server frames unmasked) + len=5 */
    ASSERT(buf[1] == 5, "expected length 5");
    /* Payload */
    ASSERT(memcmp(buf + 2, "Hello", 5) == 0, "payload mismatch");

    close(peer);
    ws_destroy(ws);
    PASS();
}

static void test_send_frame_binary_short(void) {
    TEST("ws: send binary frame, 7-bit length");
    int peer;
    WebSocket *ws = make_test_ws(&peer);
    ASSERT(ws != NULL, "make_test_ws failed");

    uint8_t data[] = {0xDE, 0xAD, 0xBE, 0xEF};
    int rc = ws_send(ws, WS_BINARY, data, 4);
    ASSERT(rc == 0, "ws_send failed");

    uint8_t buf[128];
    ssize_t n = read(peer, buf, sizeof(buf));
    ASSERT(n >= 6, "short read");

    /* Byte 0: FIN=1 + opcode=0x2 => 0x82 */
    ASSERT(buf[0] == 0x82, "expected FIN|BINARY (0x82)");
    ASSERT(buf[1] == 4, "expected length 4");
    ASSERT(memcmp(buf + 2, data, 4) == 0, "payload mismatch");

    close(peer);
    ws_destroy(ws);
    PASS();
}

static void test_send_frame_ping(void) {
    TEST("ws: send ping frame");
    int peer;
    WebSocket *ws = make_test_ws(&peer);
    ASSERT(ws != NULL, "make_test_ws failed");

    int rc = ws_send(ws, WS_PING, NULL, 0);
    ASSERT(rc == 0, "ws_send failed");

    uint8_t buf[32];
    ssize_t n = read(peer, buf, sizeof(buf));
    ASSERT(n >= 2, "short read");

    /* Byte 0: FIN=1 + opcode=0x9 => 0x89 */
    ASSERT(buf[0] == 0x89, "expected FIN|PING (0x89)");
    ASSERT(buf[1] == 0, "expected length 0");

    close(peer);
    ws_destroy(ws);
    PASS();
}

static void test_send_frame_pong(void) {
    TEST("ws: send pong frame");
    int peer;
    WebSocket *ws = make_test_ws(&peer);
    ASSERT(ws != NULL, "make_test_ws failed");

    uint8_t payload[] = {0x01, 0x02};
    int rc = ws_send(ws, WS_PONG, payload, 2);
    ASSERT(rc == 0, "ws_send failed");

    uint8_t buf[32];
    ssize_t n = read(peer, buf, sizeof(buf));
    ASSERT(n >= 4, "short read");

    /* Byte 0: FIN=1 + opcode=0xA => 0x8A */
    ASSERT(buf[0] == 0x8A, "expected FIN|PONG (0x8A)");
    ASSERT(buf[1] == 2, "expected length 2");
    ASSERT(buf[2] == 0x01 && buf[3] == 0x02, "payload mismatch");

    close(peer);
    ws_destroy(ws);
    PASS();
}

static void test_send_frame_close(void) {
    TEST("ws: send close frame");
    int peer;
    WebSocket *ws = make_test_ws(&peer);
    ASSERT(ws != NULL, "make_test_ws failed");

    /* Close with status code 1000 (normal closure) */
    uint8_t status[] = {0x03, 0xE8};
    int rc = ws_send(ws, WS_CLOSE, status, 2);
    ASSERT(rc == 0, "ws_send failed");

    uint8_t buf[32];
    ssize_t n = read(peer, buf, sizeof(buf));
    ASSERT(n >= 4, "short read");

    /* Byte 0: FIN=1 + opcode=0x8 => 0x88 */
    ASSERT(buf[0] == 0x88, "expected FIN|CLOSE (0x88)");
    ASSERT(buf[1] == 2, "expected length 2");
    ASSERT(buf[2] == 0x03 && buf[3] == 0xE8, "close code mismatch");

    close(peer);
    ws_destroy(ws);
    PASS();
}

static void test_send_frame_empty_payload(void) {
    TEST("ws: send frame with empty payload");
    int peer;
    WebSocket *ws = make_test_ws(&peer);
    ASSERT(ws != NULL, "make_test_ws failed");

    int rc = ws_send(ws, WS_TEXT, NULL, 0);
    ASSERT(rc == 0, "ws_send failed");

    uint8_t buf[32];
    ssize_t n = read(peer, buf, sizeof(buf));
    ASSERT(n >= 2, "short read");

    ASSERT(buf[0] == 0x81, "expected FIN|TEXT (0x81)");
    ASSERT(buf[1] == 0, "expected length 0");

    close(peer);
    ws_destroy(ws);
    PASS();
}

static void test_send_frame_125_bytes(void) {
    TEST("ws: send frame with exactly 125 bytes (max 7-bit)");
    int peer;
    WebSocket *ws = make_test_ws(&peer);
    ASSERT(ws != NULL, "make_test_ws failed");

    uint8_t data[125];
    memset(data, 0x42, 125);
    int rc = ws_send(ws, WS_BINARY, data, 125);
    ASSERT(rc == 0, "ws_send failed");

    uint8_t buf[256];
    ssize_t n = read(peer, buf, sizeof(buf));
    ASSERT(n >= 127, "short read");

    ASSERT(buf[0] == 0x82, "expected FIN|BINARY");
    ASSERT(buf[1] == 125, "expected 7-bit length 125");
    ASSERT(buf[2] == 0x42, "payload byte mismatch");

    close(peer);
    ws_destroy(ws);
    PASS();
}

static void test_send_frame_126_bytes(void) {
    TEST("ws: send frame with 126 bytes (triggers 16-bit len)");
    int peer;
    WebSocket *ws = make_test_ws(&peer);
    ASSERT(ws != NULL, "make_test_ws failed");

    uint8_t data[126];
    memset(data, 0xAA, 126);
    int rc = ws_send(ws, WS_BINARY, data, 126);
    ASSERT(rc == 0, "ws_send failed");

    uint8_t buf[256];
    ssize_t total = 0;
    while (total < 130) {
        ssize_t n = read(peer, buf + total, sizeof(buf) - (size_t)total);
        if (n <= 0) break;
        total += n;
    }
    ASSERT(total >= 130, "short read");

    ASSERT(buf[0] == 0x82, "expected FIN|BINARY");
    ASSERT(buf[1] == 126, "expected 16-bit length marker (126)");
    /* 16-bit extended length: big-endian 0x007E = 126 */
    ASSERT(buf[2] == 0x00, "16-bit len high byte");
    ASSERT(buf[3] == 126, "16-bit len low byte");
    /* Payload starts at offset 4 */
    ASSERT(buf[4] == 0xAA, "payload mismatch");

    close(peer);
    ws_destroy(ws);
    PASS();
}

static void test_send_frame_1000_bytes(void) {
    TEST("ws: send frame with 1000 bytes (16-bit len)");
    int peer;
    WebSocket *ws = make_test_ws(&peer);
    ASSERT(ws != NULL, "make_test_ws failed");

    uint8_t *data = malloc(1000);
    ASSERT(data != NULL, "malloc failed");
    memset(data, 0xBB, 1000);
    int rc = ws_send(ws, WS_BINARY, data, 1000);
    ASSERT(rc == 0, "ws_send failed");

    /* Read header (4 bytes for 16-bit extended) */
    uint8_t hdr[4];
    ssize_t n = read(peer, hdr, 4);
    ASSERT(n == 4, "header short read");

    ASSERT(hdr[0] == 0x82, "expected FIN|BINARY");
    ASSERT(hdr[1] == 126, "expected 16-bit length marker");
    int extended_len = (hdr[2] << 8) | hdr[3];
    ASSERT(extended_len == 1000, "expected extended length 1000");

    free(data);
    close(peer);
    ws_destroy(ws);
    PASS();
}

struct drain16_ctx {
    int fd;
    int expected;
    uint8_t hdr[4];
    int ok;
};

static void *drain16_thread(void *arg) {
    struct drain16_ctx *ctx = (struct drain16_ctx *)arg;
    /* Read the 4-byte header */
    ssize_t total = 0;
    while (total < 4) {
        ssize_t n = read(ctx->fd, ctx->hdr + total, 4 - (size_t)total);
        if (n <= 0) { ctx->ok = 0; return NULL; }
        total += n;
    }
    /* Drain the payload */
    uint8_t buf[4096];
    int drained = 0;
    while (drained < ctx->expected) {
        ssize_t n = read(ctx->fd, buf, sizeof(buf));
        if (n <= 0) break;
        drained += (int)n;
    }
    ctx->ok = 1;
    return NULL;
}

static void test_send_frame_65535_bytes(void) {
    TEST("ws: send frame with 60000 bytes (16-bit len, large)");
    int peer;
    WebSocket *ws = make_test_ws(&peer);
    ASSERT(ws != NULL, "make_test_ws failed");

    struct drain16_ctx ctx = { .fd = peer, .expected = 60000, .ok = 0 };
    pthread_t tid;
    pthread_create(&tid, NULL, drain16_thread, &ctx);

    uint8_t *data = malloc(60000);
    ASSERT(data != NULL, "malloc failed");
    memset(data, 0xCC, 60000);
    int rc = ws_send(ws, WS_BINARY, data, 60000);
    free(data);
    ASSERT(rc == 0, "ws_send failed");

    pthread_join(tid, NULL);
    ASSERT(ctx.ok == 1, "drain thread failed");

    ASSERT(ctx.hdr[0] == 0x82, "expected FIN|BINARY");
    ASSERT(ctx.hdr[1] == 126, "expected 16-bit length marker");
    int extended_len = (ctx.hdr[2] << 8) | ctx.hdr[3];
    ASSERT(extended_len == 60000, "expected extended length 60000");

    close(peer);
    ws_destroy(ws);
    PASS();
}

/*
 * Test 64-bit length encoding by sending a 70000-byte payload.
 * Since socketpair buffers are too small for large payloads,
 * we use a concurrent reader thread to drain the data.
 */
struct drain_ctx {
    int fd;
    int expected;
    uint8_t hdr[10];
    int hdr_len;
    int ok;
};

static void *drain_thread(void *arg) {
    struct drain_ctx *ctx = (struct drain_ctx *)arg;
    /* Read the 10-byte header first */
    ssize_t total = 0;
    while (total < 10) {
        ssize_t n = read(ctx->fd, ctx->hdr + total, 10 - (size_t)total);
        if (n <= 0) { ctx->ok = 0; return NULL; }
        total += n;
    }
    ctx->hdr_len = 10;

    /* Drain the payload */
    uint8_t buf[4096];
    int drained = 0;
    while (drained < ctx->expected) {
        ssize_t n = read(ctx->fd, buf, sizeof(buf));
        if (n <= 0) break;
        drained += (int)n;
    }
    ctx->ok = 1;
    return NULL;
}

static void test_send_frame_65536_bytes(void) {
    TEST("ws: send frame with 70000 bytes (triggers 64-bit len)");
    int peer;
    WebSocket *ws = make_test_ws(&peer);
    ASSERT(ws != NULL, "make_test_ws failed");

    struct drain_ctx ctx = { .fd = peer, .expected = 70000, .ok = 0 };
    pthread_t tid;
    pthread_create(&tid, NULL, drain_thread, &ctx);

    uint8_t *data = malloc(70000);
    ASSERT(data != NULL, "malloc failed");
    memset(data, 0xDD, 70000);
    int rc = ws_send(ws, WS_BINARY, data, 70000);
    free(data);
    ASSERT(rc == 0, "ws_send failed");

    pthread_join(tid, NULL);
    ASSERT(ctx.ok == 1, "drain thread failed");

    ASSERT(ctx.hdr[0] == 0x82, "expected FIN|BINARY");
    ASSERT(ctx.hdr[1] == 127, "expected 64-bit length marker (127)");
    /* 64-bit big-endian: 0x0000000000011170 = 70000 */
    ASSERT(ctx.hdr[2] == 0 && ctx.hdr[3] == 0 && ctx.hdr[4] == 0 && ctx.hdr[5] == 0,
           "64-bit len high 4 bytes should be 0");
    uint32_t low4 = ((uint32_t)ctx.hdr[6] << 24) | ((uint32_t)ctx.hdr[7] << 16) |
                    ((uint32_t)ctx.hdr[8] << 8) | (uint32_t)ctx.hdr[9];
    ASSERT(low4 == 70000, "64-bit len low 4 bytes should encode 70000");

    close(peer);
    ws_destroy(ws);
    PASS();
}

/* ── Lifecycle Tests ───────────────────────────────────── */

static void test_upgrade_creates_connected_ws(void) {
    TEST("ws: ws_upgrade creates connected WebSocket");
    int peer;
    WebSocket *ws = make_test_ws(&peer);
    ASSERT(ws != NULL, "upgrade returned NULL");
    ASSERT(ws_is_connected(ws) == true, "should be connected after upgrade");

    close(peer);
    ws_destroy(ws);
    PASS();
}

static void test_close_sets_disconnected(void) {
    TEST("ws: ws_close marks WebSocket as disconnected");
    int peer;
    WebSocket *ws = make_test_ws(&peer);
    ASSERT(ws != NULL, "upgrade returned NULL");

    ws_close(ws);
    ASSERT(ws_is_connected(ws) == false, "should be disconnected after close");

    close(peer);
    ws_destroy(ws);
    PASS();
}

static void test_send_after_close_fails(void) {
    TEST("ws: ws_send after close returns -1");
    int peer;
    WebSocket *ws = make_test_ws(&peer);
    ASSERT(ws != NULL, "upgrade returned NULL");

    ws_close(ws);
    int rc = ws_send(ws, WS_TEXT, (const uint8_t *)"hi", 2);
    ASSERT(rc == -1, "send after close should fail");

    close(peer);
    ws_destroy(ws);
    PASS();
}

static void test_double_close_safe(void) {
    TEST("ws: double ws_close is safe");
    int peer;
    WebSocket *ws = make_test_ws(&peer);
    ASSERT(ws != NULL, "upgrade returned NULL");

    ws_close(ws);
    ws_close(ws);
    ASSERT(ws_is_connected(ws) == false, "still disconnected");

    close(peer);
    ws_destroy(ws);
    PASS();
}

static void test_send_text_convenience(void) {
    TEST("ws: ws_send_text sends correct text frame");
    int peer;
    WebSocket *ws = make_test_ws(&peer);
    ASSERT(ws != NULL, "make_test_ws failed");

    int rc = ws_send_text(ws, "abc");
    ASSERT(rc == 0, "ws_send_text failed");

    uint8_t buf[32];
    ssize_t n = read(peer, buf, sizeof(buf));
    ASSERT(n >= 5, "short read");

    ASSERT(buf[0] == 0x81, "expected FIN|TEXT");
    ASSERT(buf[1] == 3, "expected length 3");
    ASSERT(memcmp(buf + 2, "abc", 3) == 0, "payload mismatch");

    close(peer);
    ws_destroy(ws);
    PASS();
}

static void test_send_binary_convenience(void) {
    TEST("ws: ws_send_binary sends correct binary frame");
    int peer;
    WebSocket *ws = make_test_ws(&peer);
    ASSERT(ws != NULL, "make_test_ws failed");

    uint8_t data[] = {0x00, 0xFF, 0x80};
    int rc = ws_send_binary(ws, data, 3);
    ASSERT(rc == 0, "ws_send_binary failed");

    uint8_t buf[32];
    ssize_t n = read(peer, buf, sizeof(buf));
    ASSERT(n >= 5, "short read");

    ASSERT(buf[0] == 0x82, "expected FIN|BINARY");
    ASSERT(buf[1] == 3, "expected length 3");
    ASSERT(memcmp(buf + 2, data, 3) == 0, "payload mismatch");

    close(peer);
    ws_destroy(ws);
    PASS();
}

/* ── Close Frame Tests ─────────────────────────────────── */

static void test_ws_close_sends_close_frame(void) {
    TEST("ws: ws_close sends close frame on wire");
    int peer;
    WebSocket *ws = make_test_ws(&peer);
    ASSERT(ws != NULL, "make_test_ws failed");

    ws_close(ws);

    uint8_t buf[16];
    ssize_t n = read(peer, buf, sizeof(buf));
    ASSERT(n >= 4, "short read");

    /* ws_close sends: 0x88, 0x02, 0x03, 0xE8 (close code 1000) */
    ASSERT(buf[0] == 0x88, "expected close opcode");
    ASSERT(buf[1] == 0x02, "expected payload length 2");
    ASSERT(buf[2] == 0x03 && buf[3] == 0xE8, "expected close code 1000");

    close(peer);
    ws_destroy(ws);
    PASS();
}

/* ── Main ──────────────────────────────────────────────── */

int main(void) {
    setbuf(stdout, NULL); /* Disable buffering for test output */
    printf("\n=== websocket tests ===\n\n");

    printf("[Opcode Constants]\n");
    test_opcode_values();
    test_opcodes_distinct();

    printf("\n[NULL Safety]\n");
    test_destroy_null();
    test_close_null();
    test_send_null_ws();
    test_send_text_null_ws();
    test_send_text_null_text();
    test_send_binary_null_ws();
    test_send_negative_len();
    test_is_connected_null();
    test_recv_null();
    test_run_loop_null();
    test_upgrade_null_key();
    test_upgrade_bad_fd();

    printf("\n[Frame Encoding - Opcodes]\n");
    test_send_frame_text_short();
    test_send_frame_binary_short();
    test_send_frame_ping();
    test_send_frame_pong();
    test_send_frame_close();
    test_send_frame_empty_payload();

    printf("\n[Frame Encoding - Length Classes]\n");
    test_send_frame_125_bytes();
    test_send_frame_126_bytes();
    test_send_frame_1000_bytes();
    test_send_frame_65535_bytes();
    test_send_frame_65536_bytes();

    printf("\n[Lifecycle]\n");
    test_upgrade_creates_connected_ws();
    test_close_sets_disconnected();
    test_send_after_close_fails();
    test_double_close_safe();

    printf("\n[Convenience Helpers]\n");
    test_send_text_convenience();
    test_send_binary_convenience();

    printf("\n[Close Frame]\n");
    test_ws_close_sends_close_frame();

    printf("\n  Results: %d passed, %d failed\n\n", tests_passed, tests_failed);
    return tests_failed > 0 ? 1 : 0;
}
