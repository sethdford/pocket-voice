/**
 * web_remote.h — WebSocket audio server for remote microphone input.
 *
 * Opens a phone's browser → captures mic via Web Audio API → streams
 * raw PCM over WebSocket → feeds directly into the voice pipeline.
 *
 * Architecture:
 *   Phone Browser ──WebSocket──► web_remote server ──► pipeline capture buffer
 *   Phone Browser ◄──WebSocket── web_remote server ◄── pipeline playback buffer
 *
 * The server runs on a background thread and serves:
 *   GET /           → HTML page with mic capture UI
 *   WebSocket /ws   → Binary audio streaming (16-bit PCM @ 16kHz)
 */

#ifndef WEB_REMOTE_H
#define WEB_REMOTE_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct WebRemote WebRemote;

/** Callback invoked when audio arrives from the phone.
 *  pcm is float32 mono at the configured sample rate. */
typedef void (*web_remote_audio_cb)(void *user_ctx, const float *pcm, int n_samples);

/** Callback invoked when the phone disconnects. */
typedef void (*web_remote_disconnect_cb)(void *user_ctx);

/**
 * Create and start a WebSocket audio server.
 *
 * @param port          TCP port to listen on (e.g. 8088)
 * @param sample_rate   Expected audio sample rate (16000 for STT)
 * @param audio_cb      Called when PCM audio arrives from phone
 * @param disconnect_cb Called when client disconnects (may be NULL)
 * @param user_ctx      Opaque pointer passed to callbacks
 * @return              Server handle, or NULL on failure
 */
WebRemote *web_remote_create(int port, int sample_rate,
                              web_remote_audio_cb audio_cb,
                              web_remote_disconnect_cb disconnect_cb,
                              void *user_ctx);

/** Send audio to the phone for playback (float32 mono).
 *  Encodes as 16-bit PCM WebSocket binary frame. */
int web_remote_send_audio(WebRemote *wr, const float *pcm, int n_samples);

/** Get the port the server is listening on. */
int web_remote_port(const WebRemote *wr);

/** Check if a client is connected. */
int web_remote_connected(const WebRemote *wr);

/** Stop server and free resources. */
void web_remote_destroy(WebRemote *wr);

#ifdef __cplusplus
}
#endif

#endif /* WEB_REMOTE_H */
