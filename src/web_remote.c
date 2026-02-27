/**
 * web_remote.c — Minimal WebSocket server for phone-as-microphone.
 *
 * Implements just enough of HTTP/1.1 Upgrade and RFC 6455 WebSocket to:
 *   1. Serve an HTML page at GET /
 *   2. Accept a WebSocket connection at GET /ws
 *   3. Receive binary audio frames (16-bit PCM) from the phone
 *   4. Send binary audio frames (16-bit PCM) to the phone for playback
 *
 * Zero dependencies beyond POSIX sockets + CommonCrypto (for SHA-1).
 */

#include "web_remote.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <pthread.h>
#include <stdatomic.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <ifaddrs.h>
#include <CommonCrypto/CommonDigest.h>

/* WebSocket constants */
#define WS_MAGIC_GUID "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"
#define WS_OP_TEXT    0x1
#define WS_OP_BINARY  0x2
#define WS_OP_CLOSE   0x8
#define WS_OP_PING    0x9
#define WS_OP_PONG    0xA

/* ═══════════════════════════════════════════════════════════════════════════
 * Embedded HTML page — served at GET /
 * ═══════════════════════════════════════════════════════════════════════════ */

static const char *HTML_PAGE =
"<!DOCTYPE html>\n"
"<html><head>\n"
"<meta charset='utf-8'>\n"
"<meta name='viewport' content='width=device-width,initial-scale=1,user-scalable=no'>\n"
"<title>pocket-voice remote</title>\n"
"<style>\n"
"*{margin:0;padding:0;box-sizing:border-box}\n"
"body{font-family:-apple-system,system-ui,sans-serif;background:#0a0a0a;color:#fff;\n"
"  display:flex;flex-direction:column;align-items:center;justify-content:center;\n"
"  min-height:100vh;padding:20px}\n"
"h1{font-size:1.5em;margin-bottom:8px;font-weight:600}\n"
".sub{color:#888;font-size:0.85em;margin-bottom:32px}\n"
"#mic-btn{width:140px;height:140px;border-radius:50%;border:none;\n"
"  background:linear-gradient(135deg,#1a73e8,#4285f4);color:#fff;\n"
"  font-size:3em;cursor:pointer;transition:all 0.2s;\n"
"  box-shadow:0 4px 20px rgba(66,133,244,0.4);display:flex;\n"
"  align-items:center;justify-content:center}\n"
"#mic-btn:active{transform:scale(0.95)}\n"
"#mic-btn.active{background:linear-gradient(135deg,#ea4335,#ff6659);\n"
"  box-shadow:0 4px 20px rgba(234,67,53,0.5);animation:pulse 1.5s infinite}\n"
"@keyframes pulse{0%,100%{box-shadow:0 4px 20px rgba(234,67,53,0.5)}\n"
"  50%{box-shadow:0 4px 40px rgba(234,67,53,0.8)}}\n"
"#status{margin-top:24px;font-size:0.9em;color:#aaa}\n"
"#level{width:200px;height:6px;background:#222;border-radius:3px;margin-top:16px;overflow:hidden}\n"
"#level-bar{height:100%;width:0%;background:#4285f4;border-radius:3px;transition:width 50ms}\n"
"#level-bar.active{background:#ea4335}\n"
"</style></head><body>\n"
"<h1>pocket-voice</h1>\n"
"<p class='sub'>remote microphone</p>\n"
"<button id='mic-btn' onclick='toggle()'>&#x1F3A4;</button>\n"
"<div id='level'><div id='level-bar'></div></div>\n"
"<p id='status'>tap to connect</p>\n"
"<script>\n"
"let ws,ctx,src,proc,active=false;\n"
"const sr=16000;\n"
"function toggle(){\n"
"  if(active){stop();return}\n"
"  active=true;\n"
"  document.getElementById('mic-btn').classList.add('active');\n"
"  document.getElementById('status').textContent='connecting...';\n"
"  const wsUrl='ws://'+location.hostname+':'+location.port+'/ws';\n"
"  ws=new WebSocket(wsUrl);\n"
"  ws.binaryType='arraybuffer';\n"
"  ws.onopen=()=>startMic();\n"
"  ws.onclose=()=>{document.getElementById('status').textContent='disconnected';stop()};\n"
"  ws.onerror=()=>{document.getElementById('status').textContent='connection failed';stop()};\n"
"  ws.onmessage=(e)=>{\n"
"    if(e.data instanceof ArrayBuffer)playAudio(e.data);\n"
"  };\n"
"}\n"
"async function startMic(){\n"
"  try{\n"
"    const stream=await navigator.mediaDevices.getUserMedia({audio:{sampleRate:sr,channelCount:1,echoCancellation:true,noiseSuppression:true}});\n"
"    ctx=new AudioContext({sampleRate:sr});\n"
"    src=ctx.createMediaStreamSource(stream);\n"
"    await ctx.audioWorklet.addModule('data:text/javascript,'+encodeURIComponent(`\n"
"      class P extends AudioWorkletProcessor{\n"
"        process(inputs){\n"
"          if(inputs[0]&&inputs[0][0])this.port.postMessage(inputs[0][0]);\n"
"          return true;\n"
"        }\n"
"      }\n"
"      registerProcessor(\"p\",P);\n"
"    `));\n"
"    proc=new AudioWorkletNode(ctx,'p');\n"
"    proc.port.onmessage=(e)=>{\n"
"      if(!ws||ws.readyState!==1)return;\n"
"      const f32=e.data;\n"
"      const s16=new Int16Array(f32.length);\n"
"      let sum=0;\n"
"      for(let i=0;i<f32.length;i++){\n"
"        const s=Math.max(-1,Math.min(1,f32[i]));\n"
"        s16[i]=s<0?s*32768:s*32767;\n"
"        sum+=s*s;\n"
"      }\n"
"      ws.send(s16.buffer);\n"
"      const rms=Math.sqrt(sum/f32.length);\n"
"      const pct=Math.min(100,rms*500);\n"
"      const bar=document.getElementById('level-bar');\n"
"      bar.style.width=pct+'%';\n"
"      bar.classList.toggle('active',pct>20);\n"
"    };\n"
"    src.connect(proc);\n"
"    proc.connect(ctx.destination);\n"
"    document.getElementById('status').textContent='streaming audio...';\n"
"  }catch(e){\n"
"    document.getElementById('status').textContent='mic access denied';\n"
"    stop();\n"
"  }\n"
"}\n"
"let playCtx;\n"
"function playAudio(buf){\n"
"  if(!playCtx)playCtx=new AudioContext({sampleRate:48000});\n"
"  const s16=new Int16Array(buf);\n"
"  const f32=new Float32Array(s16.length);\n"
"  for(let i=0;i<s16.length;i++)f32[i]=s16[i]/32768;\n"
"  const ab=playCtx.createBuffer(1,f32.length,48000);\n"
"  ab.getChannelData(0).set(f32);\n"
"  const s=playCtx.createBufferSource();\n"
"  s.buffer=ab;\n"
"  s.connect(playCtx.destination);\n"
"  s.start();\n"
"}\n"
"function stop(){\n"
"  active=false;\n"
"  document.getElementById('mic-btn').classList.remove('active');\n"
"  document.getElementById('level-bar').style.width='0%';\n"
"  if(proc){proc.disconnect();proc=null}\n"
"  if(src){src.disconnect();src=null}\n"
"  if(ctx){ctx.close();ctx=null}\n"
"  if(ws&&ws.readyState<2)ws.close();\n"
"}\n"
"</script></body></html>\n";

/* ═══════════════════════════════════════════════════════════════════════════
 * WebSocket server internals
 * ═══════════════════════════════════════════════════════════════════════════ */

struct WebRemote {
    int listen_fd;
    int client_fd;
    int port;
    int sample_rate;
    _Atomic int connected;
    _Atomic int running;
    pthread_t accept_thread;
    web_remote_audio_cb audio_cb;
    web_remote_disconnect_cb disconnect_cb;
    void *user_ctx;
};

/* Base64 encode for WebSocket accept header */
static const char b64[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

static int base64_encode(const unsigned char *in, int len, char *out, int out_cap)
{
    int i = 0, o = 0;
    while (i < len && o + 4 < out_cap) {
        uint32_t v = (uint32_t)in[i++] << 16;
        if (i < len) v |= (uint32_t)in[i++] << 8;
        if (i < len) v |= (uint32_t)in[i++];
        out[o++] = b64[(v >> 18) & 63];
        out[o++] = b64[(v >> 12) & 63];
        out[o++] = (i > len - 2 + (len % 3 == 1 ? 1 : 0)) ? '=' : b64[(v >> 6) & 63];
        out[o++] = (i > len - 1 + (len % 3 != 0 ? 1 : 0)) ? '=' : b64[v & 63];
    }
    /* Fix padding */
    int pad = (3 - len % 3) % 3;
    for (int p = 0; p < pad && o - p - 1 >= 0; p++)
        out[o - p - 1] = '=';
    out[o] = '\0';
    return o;
}

static void compute_ws_accept(const char *key, char *out, int out_cap)
{
    char concat[256];
    snprintf(concat, sizeof(concat), "%s%s", key, WS_MAGIC_GUID);

    unsigned char sha1[CC_SHA1_DIGEST_LENGTH];
    CC_SHA1(concat, (CC_LONG)strlen(concat), sha1);
    base64_encode(sha1, CC_SHA1_DIGEST_LENGTH, out, out_cap);
}

static char *find_header(const char *headers, const char *name)
{
    static char val[256];
    const char *p = strcasestr(headers, name);
    if (!p) return NULL;
    p += strlen(name);
    while (*p == ' ' || *p == ':') p++;
    int i = 0;
    while (*p && *p != '\r' && *p != '\n' && i < 255)
        val[i++] = *p++;
    val[i] = '\0';
    return val;
}

static int send_all(int fd, const void *buf, size_t len)
{
    const char *p = (const char *)buf;
    size_t sent = 0;
    while (sent < len) {
        ssize_t n = send(fd, p + sent, len - sent, 0);
        if (n <= 0) return -1;
        sent += (size_t)n;
    }
    return 0;
}

/* Get local IP for display */
static void get_local_ip(char *buf, int buf_size)
{
    struct ifaddrs *addrs, *tmp;
    buf[0] = '\0';
    if (getifaddrs(&addrs) != 0) return;
    for (tmp = addrs; tmp; tmp = tmp->ifa_next) {
        if (!tmp->ifa_addr || tmp->ifa_addr->sa_family != AF_INET) continue;
        if (strcmp(tmp->ifa_name, "lo0") == 0) continue;
        struct sockaddr_in *sa = (struct sockaddr_in *)tmp->ifa_addr;
        inet_ntop(AF_INET, &sa->sin_addr, buf, buf_size);
        if (strncmp(buf, "127.", 4) != 0) break;
    }
    freeifaddrs(addrs);
}

/* Handle one HTTP request — either serve HTML or upgrade to WebSocket */
static int handle_request(WebRemote *wr, int fd)
{
    char buf[4096];
    ssize_t n = recv(fd, buf, sizeof(buf) - 1, 0);
    if (n <= 0) return -1;
    buf[n] = '\0';

    /* Check if this is a WebSocket upgrade request */
    char *ws_key = find_header(buf, "Sec-WebSocket-Key");
    if (ws_key && strstr(buf, "GET /ws")) {
        /* Validate Origin header — only allow same-host connections */
        char *origin = find_header(buf, "Origin");
        if (origin) {
            char *host = find_header(buf, "Host");
            if (host && !strstr(origin, host)) {
                const char *forbidden = "HTTP/1.1 403 Forbidden\r\nConnection: close\r\n\r\n";
                send_all(fd, forbidden, strlen(forbidden));
                close(fd);
                return 0;
            }
        }
        /* WebSocket upgrade handshake */
        char accept[64];
        compute_ws_accept(ws_key, accept, sizeof(accept));

        char resp[512];
        int rlen = snprintf(resp, sizeof(resp),
            "HTTP/1.1 101 Switching Protocols\r\n"
            "Upgrade: websocket\r\n"
            "Connection: Upgrade\r\n"
            "Sec-WebSocket-Accept: %s\r\n\r\n", accept);
        send_all(fd, resp, (size_t)rlen);

        wr->client_fd = fd;
        atomic_store(&wr->connected, 1);
        fprintf(stderr, "[web_remote] Phone connected via WebSocket\n");
        return 1; /* WebSocket connection established */
    }

    /* Serve HTML page */
    char resp[65536];
    int html_len = (int)strlen(HTML_PAGE);
    int rlen = snprintf(resp, sizeof(resp),
        "HTTP/1.1 200 OK\r\n"
        "Content-Type: text/html; charset=utf-8\r\n"
        "Content-Length: %d\r\n"
        "Connection: close\r\n"
        "Cache-Control: no-cache\r\n\r\n%s",
        html_len, HTML_PAGE);
    send_all(fd, resp, (size_t)rlen);
    close(fd);
    return 0;
}

/* Read and decode one WebSocket frame. Returns payload length, -1 on error/close. */
static int ws_read_frame(int fd, uint8_t *payload, int max_len, int *opcode)
{
    uint8_t hdr[2];
    if (recv(fd, hdr, 2, MSG_WAITALL) != 2) return -1;

    *opcode = hdr[0] & 0x0F;
    int masked = (hdr[1] & 0x80) != 0;
    uint64_t len = hdr[1] & 0x7F;

    if (len == 126) {
        uint8_t ext[2];
        if (recv(fd, ext, 2, MSG_WAITALL) != 2) return -1;
        len = ((uint64_t)ext[0] << 8) | ext[1];
    } else if (len == 127) {
        uint8_t ext[8];
        if (recv(fd, ext, 8, MSG_WAITALL) != 8) return -1;
        len = 0;
        for (int i = 0; i < 8; i++) len = (len << 8) | ext[i];
    }

    uint8_t mask_key[4] = {0};
    if (masked) {
        if (recv(fd, mask_key, 4, MSG_WAITALL) != 4) return -1;
    }

    if (len > (16 * 1024 * 1024)) return -1;  /* 16 MB frame limit */
    if ((int64_t)len > max_len) return -1;

    int to_read = (int)len;
    int total = 0;
    while (total < to_read) {
        ssize_t n = recv(fd, payload + total, (size_t)(to_read - total), 0);
        if (n <= 0) return -1;
        total += (int)n;
    }

    if (masked) {
        for (int i = 0; i < total; i++)
            payload[i] ^= mask_key[i & 3];
    }

    return total;
}

/* WebSocket message loop — runs on the accept thread after upgrade */
static void ws_message_loop(WebRemote *wr)
{
    uint8_t *frame_buf = malloc(65536);
    float *pcm_buf = malloc(32768 * sizeof(float));

    while (atomic_load(&wr->running) && atomic_load(&wr->connected)) {
        int opcode;
        int len = ws_read_frame(wr->client_fd, frame_buf, 65536, &opcode);
        if (len < 0) break;

        if (opcode == WS_OP_CLOSE) break;

        if (opcode == WS_OP_PING) {
            /* Send pong */
            uint8_t pong[2] = {0x80 | WS_OP_PONG, 0};
            send_all(wr->client_fd, pong, 2);
            continue;
        }

        if (opcode == WS_OP_BINARY && len >= 2 && wr->audio_cb) {
            /* Decode 16-bit PCM to float32 */
            int n_samples = len / 2;
            if (n_samples > 32768) n_samples = 32768;
            const int16_t *s16 = (const int16_t *)frame_buf;
            for (int i = 0; i < n_samples; i++)
                pcm_buf[i] = (float)s16[i] / 32768.0f;

            wr->audio_cb(wr->user_ctx, pcm_buf, n_samples);
        }
    }

    free(frame_buf);
    free(pcm_buf);

    atomic_store(&wr->connected, 0);
    close(wr->client_fd);
    wr->client_fd = -1;
    fprintf(stderr, "[web_remote] Phone disconnected\n");

    if (wr->disconnect_cb)
        wr->disconnect_cb(wr->user_ctx);
}

/* Accept thread — listens for connections */
static void *accept_thread_fn(void *arg)
{
    WebRemote *wr = (WebRemote *)arg;

    while (atomic_load(&wr->running)) {
        struct sockaddr_in client_addr;
        socklen_t addr_len = sizeof(client_addr);
        int fd = accept(wr->listen_fd, (struct sockaddr *)&client_addr, &addr_len);
        if (fd < 0) {
            if (!atomic_load(&wr->running)) break;
            continue;
        }

        int rc = handle_request(wr, fd);
        if (rc == 1) {
            /* WebSocket established — enter message loop */
            ws_message_loop(wr);
        }
    }
    return NULL;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Public API
 * ═══════════════════════════════════════════════════════════════════════════ */

WebRemote *web_remote_create(int port, int sample_rate,
                              web_remote_audio_cb audio_cb,
                              web_remote_disconnect_cb disconnect_cb,
                              void *user_ctx)
{
    WebRemote *wr = calloc(1, sizeof(WebRemote));
    wr->port = port;
    wr->sample_rate = sample_rate;
    wr->audio_cb = audio_cb;
    wr->disconnect_cb = disconnect_cb;
    wr->user_ctx = user_ctx;
    wr->client_fd = -1;

    wr->listen_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (wr->listen_fd < 0) {
        free(wr);
        return NULL;
    }

    int opt = 1;
    setsockopt(wr->listen_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    struct sockaddr_in addr = {0};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port = htons((uint16_t)port);

    if (bind(wr->listen_fd, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
        fprintf(stderr, "[web_remote] bind failed on port %d\n", port);
        close(wr->listen_fd);
        free(wr);
        return NULL;
    }

    if (listen(wr->listen_fd, 2) < 0) {
        close(wr->listen_fd);
        free(wr);
        return NULL;
    }

    /* Get actual port (in case 0 was passed) */
    struct sockaddr_in bound;
    socklen_t blen = sizeof(bound);
    getsockname(wr->listen_fd, (struct sockaddr *)&bound, &blen);
    wr->port = ntohs(bound.sin_port);

    /* Print connection info */
    char ip[64] = "localhost";
    get_local_ip(ip, sizeof(ip));
    fprintf(stderr, "[web_remote] Server listening on port %d\n", wr->port);
    fprintf(stderr, "[web_remote] Open on your phone: http://%s:%d\n", ip, wr->port);

    atomic_store(&wr->running, 1);
    pthread_create(&wr->accept_thread, NULL, accept_thread_fn, wr);

    return wr;
}

int web_remote_send_audio(WebRemote *wr, const float *pcm, int n_samples)
{
    if (!wr || !atomic_load(&wr->connected) || wr->client_fd < 0) return -1;

    /* Convert float32 to int16 */
    int payload_len = n_samples * 2;
    uint8_t *frame = malloc((size_t)(payload_len + 10));

    /* WebSocket frame header (unmasked, server→client) */
    int hdr_len = 0;
    frame[hdr_len++] = 0x80 | WS_OP_BINARY; /* FIN + binary */
    if (payload_len < 126) {
        frame[hdr_len++] = (uint8_t)payload_len;
    } else if (payload_len < 65536) {
        frame[hdr_len++] = 126;
        frame[hdr_len++] = (uint8_t)(payload_len >> 8);
        frame[hdr_len++] = (uint8_t)(payload_len & 0xFF);
    } else {
        frame[hdr_len++] = 127;
        for (int i = 7; i >= 0; i--)
            frame[hdr_len++] = (uint8_t)((payload_len >> (i * 8)) & 0xFF);
    }

    int16_t *s16 = (int16_t *)(frame + hdr_len);
    for (int i = 0; i < n_samples; i++) {
        float s = pcm[i] * 32767.0f;
        if (s > 32767.0f) s = 32767.0f;
        if (s < -32768.0f) s = -32768.0f;
        s16[i] = (int16_t)s;
    }

    int rc = send_all(wr->client_fd, frame, (size_t)(hdr_len + payload_len));
    free(frame);
    return rc;
}

int web_remote_port(const WebRemote *wr)
{
    return wr ? wr->port : 0;
}

int web_remote_connected(const WebRemote *wr)
{
    return wr ? atomic_load(&wr->connected) : 0;
}

void web_remote_destroy(WebRemote *wr)
{
    if (!wr) return;
    atomic_store(&wr->running, 0);
    if (wr->client_fd >= 0) close(wr->client_fd);
    close(wr->listen_fd);
    pthread_join(wr->accept_thread, NULL);
    free(wr);
}
