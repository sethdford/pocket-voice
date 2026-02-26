#ifndef WEBSOCKET_H
#define WEBSOCKET_H

#include <stdint.h>
#include <stdbool.h>

typedef struct WebSocket WebSocket;

/* WebSocket message types */
typedef enum {
    WS_TEXT   = 0x1,
    WS_BINARY = 0x2,
    WS_CLOSE  = 0x8,
    WS_PING   = 0x9,
    WS_PONG   = 0xA
} WsOpcode;

/* Callback for received messages */
typedef void (*WsMessageCallback)(WebSocket *ws, WsOpcode type,
                                    const uint8_t *data, int len, void *ctx);

/**
 * Upgrade an HTTP connection to WebSocket.
 *
 * @param fd              Connected socket file descriptor
 * @param sec_websocket_key  The Sec-WebSocket-Key header value from the HTTP upgrade request
 * @return                WebSocket handle, or NULL on failure
 */
WebSocket *ws_upgrade(int fd, const char *sec_websocket_key);

/** Close WebSocket gracefully with close frame. */
void ws_close(WebSocket *ws);

/** Destroy WebSocket and free resources. */
void ws_destroy(WebSocket *ws);

/**
 * Send a WebSocket frame.
 *
 * @param ws    WebSocket handle
 * @param type  Frame opcode (WS_TEXT, WS_BINARY, etc.)
 * @param data  Payload data
 * @param len   Payload length
 * @return      0 on success, -1 on error
 */
int ws_send(WebSocket *ws, WsOpcode type, const uint8_t *data, int len);

/** Convenience: send text message. */
int ws_send_text(WebSocket *ws, const char *text);

/** Convenience: send binary message (for audio). */
int ws_send_binary(WebSocket *ws, const uint8_t *data, int len);

/**
 * Read next WebSocket frame (blocking).
 *
 * @param ws       WebSocket handle
 * @param type_out Opcode of received frame
 * @param buf      Buffer for payload
 * @param buf_size Size of buffer
 * @return         Payload length, 0 for close, -1 on error
 */
int ws_recv(WebSocket *ws, WsOpcode *type_out, uint8_t *buf, int buf_size);

/**
 * Run a message loop, calling the callback for each received message.
 * Returns when the connection is closed or an error occurs.
 */
int ws_run_loop(WebSocket *ws, WsMessageCallback callback, void *ctx);

/** Check if the WebSocket is still connected. */
bool ws_is_connected(const WebSocket *ws);

#endif /* WEBSOCKET_H */
