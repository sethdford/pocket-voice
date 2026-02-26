#include "websocket.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <errno.h>

#include <CommonCrypto/CommonDigest.h>

#define WS_RECV_BUF_SIZE (64 * 1024)
#define WS_GUID "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"

struct WebSocket {
    int      fd;
    bool     connected;
    uint8_t *recv_buf;
    int      recv_len;
};

static const char b64[] =
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

static int base64_encode(const uint8_t *in, int len, char *out, int out_size) {
    int i = 0, j = 0;
    int need = ((len + 2) / 3) * 4 + 1;
    if (out_size < need) return -1;

    while (i + 2 < len) {
        uint32_t v = (in[i] << 16) | (in[i + 1] << 8) | in[i + 2];
        out[j++] = b64[(v >> 18) & 0x3F];
        out[j++] = b64[(v >> 12) & 0x3F];
        out[j++] = b64[(v >> 6) & 0x3F];
        out[j++] = b64[v & 0x3F];
        i += 3;
    }
    if (i < len) {
        uint32_t v = in[i] << 16;
        if (i + 1 < len) v |= in[i + 1] << 8;
        out[j++] = b64[(v >> 18) & 0x3F];
        out[j++] = b64[(v >> 12) & 0x3F];
        out[j++] = (i + 1 < len) ? b64[(v >> 6) & 0x3F] : '=';
        out[j++] = '=';
    }
    out[j] = '\0';
    return j;
}

WebSocket *ws_upgrade(int fd, const char *sec_websocket_key) {
    if (!sec_websocket_key || fd < 0) return NULL;

    char concat[256];
    int klen = (int)strlen(sec_websocket_key);
    int glen = (int)strlen(WS_GUID);
    if (klen + glen + 1 >= (int)sizeof(concat)) return NULL;

    memcpy(concat, sec_websocket_key, (size_t)klen);
    memcpy(concat + klen, WS_GUID, (size_t)glen + 1);

    uint8_t digest[CC_SHA1_DIGEST_LENGTH];
    CC_SHA1((const unsigned char *)concat, (CC_LONG)(klen + glen), digest);

    char accept_key[64];
    if (base64_encode(digest, CC_SHA1_DIGEST_LENGTH, accept_key, sizeof(accept_key)) < 0)
        return NULL;

    char response[512];
    int rlen = snprintf(response, sizeof(response),
        "HTTP/1.1 101 Switching Protocols\r\n"
        "Upgrade: websocket\r\n"
        "Connection: Upgrade\r\n"
        "Sec-WebSocket-Accept: %s\r\n"
        "\r\n",
        accept_key);

    if (rlen < 0 || rlen >= (int)sizeof(response)) return NULL;

    ssize_t n = write(fd, response, (size_t)rlen);
    if (n != (ssize_t)rlen) return NULL;

    WebSocket *ws = calloc(1, sizeof(WebSocket));
    if (!ws) return NULL;

    ws->recv_buf = malloc(WS_RECV_BUF_SIZE);
    if (!ws->recv_buf) {
        free(ws);
        return NULL;
    }

    ws->fd = fd;
    ws->connected = true;
    ws->recv_len = 0;
    return ws;
}

void ws_close(WebSocket *ws) {
    if (!ws || !ws->connected) return;
    uint8_t close_frame[] = { 0x88, 0x02, 0x03, 0xE8 }; /* status 1000 */
    write(ws->fd, close_frame, sizeof(close_frame));
    ws->connected = false;
}

void ws_destroy(WebSocket *ws) {
    if (!ws) return;
    if (ws->fd >= 0) {
        close(ws->fd);
        ws->fd = -1;
    }
    free(ws->recv_buf);
    ws->recv_buf = NULL;
    ws->connected = false;
    free(ws);
}

static ssize_t ws_read_full(WebSocket *ws, void *buf, size_t len) {
    size_t total = 0;
    while (total < len) {
        ssize_t n = read(ws->fd, (char *)buf + total, len - total);
        if (n <= 0) return n;
        total += (size_t)n;
    }
    return (ssize_t)total;
}

int ws_send(WebSocket *ws, WsOpcode type, const uint8_t *data, int len) {
    if (!ws || !ws->connected) return -1;
    if (len < 0) return -1;

    uint8_t header[14];
    size_t header_len;

    header[0] = 0x80 | (type & 0x0F);
    if (len <= 125) {
        header[1] = (uint8_t)len;
        header_len = 2;
    } else if (len <= 65535) {
        header[1] = 126;
        header[2] = (uint8_t)(len >> 8);
        header[3] = (uint8_t)(len & 0xFF);
        header_len = 4;
    } else {
        uint64_t len64 = (uint64_t)len;
        header[1] = 127;
        header[2] = (uint8_t)((len64 >> 56) & 0xFF);
        header[3] = (uint8_t)((len64 >> 48) & 0xFF);
        header[4] = (uint8_t)((len64 >> 40) & 0xFF);
        header[5] = (uint8_t)((len64 >> 32) & 0xFF);
        header[6] = (uint8_t)((len64 >> 24) & 0xFF);
        header[7] = (uint8_t)((len64 >> 16) & 0xFF);
        header[8] = (uint8_t)((len64 >> 8) & 0xFF);
        header[9] = (uint8_t)(len64 & 0xFF);
        header_len = 10;
    }

    if (write(ws->fd, header, header_len) != (ssize_t)header_len)
        return -1;
    if (len > 0 && data && write(ws->fd, data, (size_t)len) != (ssize_t)len)
        return -1;
    return 0;
}

int ws_send_text(WebSocket *ws, const char *text) {
    if (!text) return -1;
    return ws_send(ws, WS_TEXT, (const uint8_t *)text, (int)strlen(text));
}

int ws_send_binary(WebSocket *ws, const uint8_t *data, int len) {
    return ws_send(ws, WS_BINARY, data, len);
}

int ws_recv(WebSocket *ws, WsOpcode *type_out, uint8_t *buf, int buf_size) {
    if (!ws || !ws->connected || !buf || buf_size <= 0) return -1;

    for (;;) {
        uint8_t hdr[2];
        if (ws_read_full(ws, hdr, 2) != 2) return -1;

        uint8_t opcode = hdr[0] & 0x0F;
        int masked = (hdr[1] >> 7) & 1;
        uint64_t payload_len = hdr[1] & 0x7F;

        if (payload_len == 126) {
            uint8_t ext[2];
            if (ws_read_full(ws, ext, 2) != 2) return -1;
            payload_len = ((uint64_t)ext[0] << 8) | ext[1];
        } else if (payload_len == 127) {
            uint8_t ext[8];
            if (ws_read_full(ws, ext, 8) != 8) return -1;
            payload_len = 0;
            for (int i = 0; i < 8; i++)
                payload_len = (payload_len << 8) | ext[i];
        }

        if (payload_len > (uint64_t)buf_size) return -1;

        uint8_t mask[4];
        if (masked) {
            if (ws_read_full(ws, mask, 4) != 4) return -1;
        }

        if (payload_len > 0) {
            if (ws_read_full(ws, buf, (size_t)payload_len) != (ssize_t)payload_len)
                return -1;
            if (masked) {
                for (uint64_t i = 0; i < payload_len; i++)
                    buf[i] ^= mask[i & 3];
            }
        }

        if (opcode == WS_PING) {
            ws_send(ws, WS_PONG, buf, (int)payload_len);
            continue;
        }
        if (opcode == WS_CLOSE) {
            ws->connected = false;
            uint8_t close_frame[] = { 0x88, 0x00 };
            write(ws->fd, close_frame, 2);
            if (type_out) *type_out = WS_CLOSE;
            return 0;
        }

        if (type_out) *type_out = (WsOpcode)opcode;
        return (int)payload_len;
    }
}

int ws_run_loop(WebSocket *ws, WsMessageCallback callback, void *ctx) {
    if (!ws || !callback) return -1;
    uint8_t buf[65536];
    WsOpcode type;

    while (ws_is_connected(ws)) {
        int n = ws_recv(ws, &type, buf, sizeof(buf));
        if (n < 0) return -1;
        if (n == 0 && type == WS_CLOSE) return 0;
        callback(ws, type, buf, n, ctx);
    }
    return 0;
}

bool ws_is_connected(const WebSocket *ws) {
    return ws && ws->connected;
}
