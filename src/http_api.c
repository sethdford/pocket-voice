#include "http_api.h"
#include "websocket.h"
#include "cJSON.h"
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include <errno.h>
#include <pthread.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <dlfcn.h>
#include <mach/mach_time.h>

/* Opus codec FFI — libpocket_opus provides create/destroy, encode/decode */
typedef struct PocketOpus PocketOpus;
extern PocketOpus *pocket_opus_create(int sample_rate, int channels, int bitrate,
                                       float frame_ms, int application);
extern void pocket_opus_destroy(PocketOpus *ctx);
extern int pocket_opus_encode(PocketOpus *ctx, const float *pcm, int n_samples,
                             unsigned char *opus_out, int max_out);
extern int pocket_opus_decode(PocketOpus *ctx, const unsigned char *opus_data,
                             int opus_len, float *pcm_out, int max_samples);
extern int pocket_opus_flush(PocketOpus *ctx, unsigned char *opus_out, int max_out);

/* Hardware resampler FFI */
typedef struct HWResampler HWResampler;
extern HWResampler *hw_resampler_create(int src_rate, int dst_rate, int channels, int quality);
extern int hw_resample(HWResampler *ctx, const float *input, int in_frames,
                       float *output, int max_out);
extern void hw_resampler_destroy(HWResampler *ctx);

#define MAX_REQ_SIZE  (16 * 1024 * 1024) /* 16 MB max request body (audio) */
#define MAX_RESP_SIZE (32 * 1024 * 1024) /* 32 MB max response */
#define MAX_TEXT_SIZE (10 * 1024)        /* 10 KB max input text */
#define BACKLOG       16
#define THREAD_POOL_SIZE 4
#define CONN_QUEUE_SIZE  64
#define RATE_LIMIT_RPS   60.0   /* requests per second */
#define RATE_LIMIT_BURST 10.0   /* burst capacity */

/* ─── Thread Pool ──────────────────────────────────────────────────────── */

typedef struct {
    int              fds[CONN_QUEUE_SIZE];
    int              head;
    int              tail;
    int              count;
    pthread_mutex_t  mutex;
    pthread_cond_t   not_empty;
    pthread_cond_t   not_full;
} ConnQueue;

static void cq_init(ConnQueue *q) {
    memset(q, 0, sizeof(*q));
    pthread_mutex_init(&q->mutex, NULL);
    pthread_cond_init(&q->not_empty, NULL);
    pthread_cond_init(&q->not_full, NULL);
}

static void cq_destroy(ConnQueue *q) {
    pthread_mutex_destroy(&q->mutex);
    pthread_cond_destroy(&q->not_empty);
    pthread_cond_destroy(&q->not_full);
}

static void cq_push(ConnQueue *q, int fd) {
    pthread_mutex_lock(&q->mutex);
    while (q->count >= CONN_QUEUE_SIZE)
        pthread_cond_wait(&q->not_full, &q->mutex);
    q->fds[q->tail] = fd;
    q->tail = (q->tail + 1) % CONN_QUEUE_SIZE;
    q->count++;
    pthread_cond_signal(&q->not_empty);
    pthread_mutex_unlock(&q->mutex);
}

static int cq_pop(ConnQueue *q, volatile int *running) {
    pthread_mutex_lock(&q->mutex);
    while (q->count == 0 && *running) {
        struct timespec ts;
        clock_gettime(CLOCK_REALTIME, &ts);
        ts.tv_nsec += 100000000; /* 100ms timeout */
        if (ts.tv_nsec >= 1000000000) { ts.tv_sec++; ts.tv_nsec -= 1000000000; }
        pthread_cond_timedwait(&q->not_empty, &q->mutex, &ts);
    }
    if (q->count == 0) {
        pthread_mutex_unlock(&q->mutex);
        return -1;
    }
    int fd = q->fds[q->head];
    q->head = (q->head + 1) % CONN_QUEUE_SIZE;
    q->count--;
    pthread_cond_signal(&q->not_full);
    pthread_mutex_unlock(&q->mutex);
    return fd;
}

/* ─── Token Bucket Rate Limiter ────────────────────────────────────────── */

typedef struct {
    double          tokens;
    double          max_tokens;
    double          refill_rate;
    uint64_t        last_refill_ns;
    pthread_mutex_t mutex;
} RateLimiter;

static uint64_t now_ns(void) {
    static mach_timebase_info_data_t tb;
    if (tb.denom == 0) mach_timebase_info(&tb);
    return mach_absolute_time() * tb.numer / tb.denom;
}

static void rl_init(RateLimiter *rl, double rps, double burst) {
    rl->tokens = burst;
    rl->max_tokens = burst;
    rl->refill_rate = rps;
    rl->last_refill_ns = now_ns();
    pthread_mutex_init(&rl->mutex, NULL);
}

static void rl_destroy(RateLimiter *rl) {
    pthread_mutex_destroy(&rl->mutex);
}

static int rl_allow(RateLimiter *rl) {
    pthread_mutex_lock(&rl->mutex);
    uint64_t t = now_ns();
    double elapsed = (double)(t - rl->last_refill_ns) / 1e9;
    rl->tokens += elapsed * rl->refill_rate;
    if (rl->tokens > rl->max_tokens) rl->tokens = rl->max_tokens;
    rl->last_refill_ns = t;

    if (rl->tokens >= 1.0) {
        rl->tokens -= 1.0;
        pthread_mutex_unlock(&rl->mutex);
        return 1;
    }
    pthread_mutex_unlock(&rl->mutex);
    return 0;
}

struct HttpApi {
    int              port;
    int              server_fd;
    volatile int     running;
    pthread_t        accept_thread;
    int              accept_thread_started;
    pthread_t        workers[THREAD_POOL_SIZE];
    int              worker_count;
    ConnQueue        queue;
    HttpApiEngines   eng;
    char             api_key[256];
    VoiceEntry       voices[VOICE_REGISTRY_MAX];
    pthread_mutex_t  voice_mutex;
    pthread_mutex_t  tts_mutex;
    RateLimiter      rate_limiter;
};

typedef struct {
    char method[8];
    char path[256];
    char content_type[128];
    char authorization[512];
    int  content_length;
    char *body;
    int   body_len;
    bool  ws_upgrade;
    char  ws_key[256];
} HttpRequest;

static void send_response(int fd, int status, const char *content_type,
                           const void *body, int body_len) {
    const char *status_text = (status == 200) ? "OK" :
                              (status == 400) ? "Bad Request" :
                              (status == 401) ? "Unauthorized" :
                              (status == 403) ? "Forbidden" :
                              (status == 404) ? "Not Found" :
                              (status == 413) ? "Payload Too Large" :
                              (status == 429) ? "Too Many Requests" :
                              (status == 500) ? "Internal Server Error" :
                              (status == 501) ? "Not Implemented" : "Unknown";
    char header[512];
    int hlen = snprintf(header, sizeof(header),
        "HTTP/1.1 %d %s\r\n"
        "Content-Type: %s\r\n"
        "Content-Length: %d\r\n"
        "Access-Control-Allow-Origin: *\r\n"
        "Access-Control-Allow-Methods: POST, GET, OPTIONS\r\n"
        "Access-Control-Allow-Headers: Content-Type\r\n"
        "Connection: close\r\n"
        "\r\n",
        status, status_text, content_type, body_len);
    ssize_t wr = write(fd, header, (size_t)hlen);
    if (wr < 0) return;
    if (body && body_len > 0) {
        const uint8_t *p = (const uint8_t *)body;
        int remaining = body_len;
        while (remaining > 0) {
            wr = write(fd, p, (size_t)remaining);
            if (wr <= 0) break;
            p += wr;
            remaining -= (int)wr;
        }
    }
}

static void send_json(int fd, int status, const char *json) {
    send_response(fd, status, "application/json", json, (int)strlen(json));
}

static int parse_request(int fd, HttpRequest *req) {
    memset(req, 0, sizeof(*req));
    char buf[8192];
    ssize_t total = 0;
    ssize_t n;

    while (total < (ssize_t)sizeof(buf) - 1) {
        n = read(fd, buf + total, (size_t)(sizeof(buf) - 1 - (size_t)total));
        if (n <= 0) return -1;
        total += n;
        buf[total] = '\0';
        if (strstr(buf, "\r\n\r\n")) break;
    }

    if (sscanf(buf, "%7s %255s", req->method, req->path) != 2)
        return -1;
    req->method[sizeof(req->method) - 1] = '\0';
    req->path[sizeof(req->path) - 1] = '\0';

    char *cl = strcasestr(buf, "Content-Length:");
    if (cl) {
        const char *val = cl + 15; /* Skip "Content-Length:" (15 chars) */
        while (*val == ' ' || *val == '\t') val++;
        int content_length = atoi(val);
        if (content_length < 0 || content_length > MAX_REQ_SIZE) {
            fprintf(stderr, "[http] Invalid Content-Length: %d (max %d)\n", content_length, MAX_REQ_SIZE);
            return -1;
        }
        req->content_length = content_length;
    }

    char *ct = strcasestr(buf, "Content-Type:");
    if (ct) {
        ct += 13;
        while (*ct == ' ') ct++;
        char *end = strstr(ct, "\r\n");
        if (end) {
            int clen = (int)(end - ct);
            if (clen > 127) clen = 127;
            memcpy(req->content_type, ct, (size_t)clen);
            req->content_type[clen] = '\0';
        }
    }

    char *auth = strcasestr(buf, "Authorization:");
    if (auth) {
        auth += 14;
        while (*auth == ' ') auth++;
        char *ae = strstr(auth, "\r\n");
        if (ae) {
            int alen = (int)(ae - auth);
            if (alen > 511) alen = 511;
            memcpy(req->authorization, auth, (size_t)alen);
            req->authorization[alen] = '\0';
        }
    }

    char *upgrade = strcasestr(buf, "Upgrade:");
    if (upgrade) {
        upgrade += 8;
        while (*upgrade == ' ') upgrade++;
        char *ue = strstr(upgrade, "\r\n");
        char *wp = strcasestr(upgrade, "websocket");
        if (ue && wp && wp < ue)
            req->ws_upgrade = true;
    }

    char *wskey = strcasestr(buf, "Sec-WebSocket-Key:");
    if (wskey) {
        wskey += 18;
        while (*wskey == ' ') wskey++;
        char *ke = strstr(wskey, "\r\n");
        if (ke) {
            int klen = (int)(ke - wskey);
            if (klen > 0 && klen < 255) {
                memcpy(req->ws_key, wskey, (size_t)klen);
                req->ws_key[klen] = '\0';
            }
        }
    }

    char *body_start = strstr(buf, "\r\n\r\n");
    if (!body_start) return -1;
    body_start += 4;
    int header_bytes = (int)(body_start - buf);
    int body_received = (int)(total - header_bytes);

    if (req->content_length > 0 && req->content_length <= MAX_REQ_SIZE) {
        req->body = malloc((size_t)req->content_length + 1);
        if (!req->body) return -1;
        if (body_received > 0) {
            if (body_received > req->content_length)
                body_received = req->content_length;
            memcpy(req->body, body_start, (size_t)body_received);
        }
        while (body_received < req->content_length) {
            n = read(fd, req->body + body_received,
                     (size_t)(req->content_length - body_received));
            if (n <= 0) { free(req->body); req->body = NULL; return -1; }
            body_received += (int)n;
        }
        req->body[req->content_length] = '\0';
        req->body_len = req->content_length;
    }

    return 0;
}

/* Forward declarations */
static int apply_cloned_voice(HttpApi *api, const char *voice_id);
static int json_escape(const char *in, int in_len, char *out, int out_size);

/* ─── TTS Request Cleanup ─────────────────────────────────────────────── */

static void tts_request_cleanup(TtsRequest *req) {
    free(req->text);
    free(req->voice);
    free(req->emotion);
    req->text = req->voice = req->emotion = NULL;
}

/* ─── OpenAI Voice Name → Speaker ID Mapping ─────────────────────────── */

static const struct { const char *name; const char *id; } OPENAI_VOICES[] = {
    {"alloy",   "0"}, {"echo",  "1"}, {"fable", "2"},
    {"onyx",    "3"}, {"nova",  "4"}, {"shimmer", "5"},
    {NULL, NULL}
};

static void map_openai_voice(TtsRequest *req) {
    if (!req->voice) return;
    for (int i = 0; OPENAI_VOICES[i].name; i++) {
        if (strcasecmp(req->voice, OPENAI_VOICES[i].name) == 0) {
            free(req->voice);
            req->voice = strdup(OPENAI_VOICES[i].id);
            return;
        }
    }
}

/* ─── G.711 mu-law / A-law Encoding ──────────────────────────────────────── */

static const int16_t MULAW_BIAS = 0x84;
static const int16_t MULAW_CLIP = 32635;

static uint8_t encode_mulaw(int16_t sample) {
    int sign = (sample >> 8) & 0x80;
    if (sign) sample = -sample;
    if (sample > MULAW_CLIP) sample = MULAW_CLIP;
    sample = (int16_t)(sample + MULAW_BIAS);

    int exponent = 7;
    for (int mask = 0x4000; mask > 0 && !(sample & mask); mask >>= 1)
        exponent--;

    int mantissa = (sample >> (exponent + 3)) & 0x0F;
    uint8_t byte = (uint8_t)(~(sign | (exponent << 4) | mantissa));
    return byte;
}

static uint8_t encode_alaw(int16_t sample) {
    int sign = 0;
    if (sample < 0) { sample = -sample; sign = 0x80; }
    if (sample > 32767) sample = 32767;

    int exponent = 0;
    for (int v = sample >> 4; v > 1; v >>= 1)
        exponent++;

    int mantissa;
    if (exponent > 0)
        mantissa = (sample >> (exponent + 3)) & 0x0F;
    else
        mantissa = (sample >> 4) & 0x0F;

    uint8_t byte = (uint8_t)(sign | (exponent << 4) | mantissa);
    return (uint8_t)(byte ^ 0x55);
}

/* ─── LAME MP3 Encoder (dlopen, optional) ────────────────────────────── */

typedef void *lame_t;
static void *g_lame_lib = NULL;
static int g_lame_tried = 0;

typedef lame_t (*fn_lame_init)(void);
typedef int    (*fn_lame_set_in_samplerate)(lame_t, int);
typedef int    (*fn_lame_set_num_channels)(lame_t, int);
typedef int    (*fn_lame_set_quality)(lame_t, int);
typedef int    (*fn_lame_set_VBR)(lame_t, int);
typedef int    (*fn_lame_set_brate)(lame_t, int);
typedef int    (*fn_lame_init_params)(lame_t);
typedef int    (*fn_lame_encode_buffer_ieee_float)(lame_t, const float *, const float *,
                                                    int, unsigned char *, int);
typedef int    (*fn_lame_encode_flush)(lame_t, unsigned char *, int);
typedef int    (*fn_lame_close)(lame_t);

static struct {
    fn_lame_init                        init;
    fn_lame_set_in_samplerate           set_in_samplerate;
    fn_lame_set_num_channels            set_num_channels;
    fn_lame_set_quality                 set_quality;
    fn_lame_set_VBR                     set_VBR;
    fn_lame_set_brate                   set_brate;
    fn_lame_init_params                 init_params;
    fn_lame_encode_buffer_ieee_float    encode_float;
    fn_lame_encode_flush                encode_flush;
    fn_lame_close                       close;
} lame_fn;

static int lame_load(void) {
    if (g_lame_tried) return g_lame_lib ? 0 : -1;
    g_lame_tried = 1;

    const char *paths[] = {
        "libmp3lame.dylib",
        "/opt/homebrew/lib/libmp3lame.dylib",
        "/usr/local/lib/libmp3lame.dylib",
        NULL
    };

    for (int i = 0; paths[i]; i++) {
        g_lame_lib = dlopen(paths[i], RTLD_LAZY);
        if (g_lame_lib) break;
    }
    if (!g_lame_lib) return -1;

    #define LOAD_SYM(name) lame_fn.name = dlsym(g_lame_lib, "lame_" #name); \
                           if (!lame_fn.name) { dlclose(g_lame_lib); g_lame_lib = NULL; return -1; }

    LOAD_SYM(init)
    LOAD_SYM(set_in_samplerate)
    LOAD_SYM(set_num_channels)
    LOAD_SYM(set_quality)
    LOAD_SYM(set_VBR)
    LOAD_SYM(set_brate)
    LOAD_SYM(init_params)
    lame_fn.encode_float = dlsym(g_lame_lib, "lame_encode_buffer_ieee_float");
    if (!lame_fn.encode_float) { dlclose(g_lame_lib); g_lame_lib = NULL; return -1; }
    LOAD_SYM(encode_flush)
    LOAD_SYM(close)

    #undef LOAD_SYM
    return 0;
}

static int encode_mp3(const float *pcm, int n_samples, int sample_rate,
                       uint8_t *mp3_out, int max_out) {
    if (lame_load() != 0) return -1;

    lame_t lame = lame_fn.init();
    if (!lame) return -1;

    lame_fn.set_in_samplerate(lame, sample_rate);
    lame_fn.set_num_channels(lame, 1);
    lame_fn.set_quality(lame, 2);
    lame_fn.set_VBR(lame, 0); /* CBR */
    lame_fn.set_brate(lame, 128);
    lame_fn.init_params(lame);

    int total = 0;
    int chunk = 4096;
    for (int off = 0; off < n_samples; off += chunk) {
        int n = (n_samples - off < chunk) ? n_samples - off : chunk;
        int ret = lame_fn.encode_float(lame, pcm + off, NULL, n,
                                        mp3_out + total, max_out - total);
        if (ret < 0) break;
        total += ret;
    }

    int flush = lame_fn.encode_flush(lame, mp3_out + total, max_out - total);
    if (flush > 0) total += flush;

    lame_fn.close(lame);
    return total;
}

/* ─── WAV Helpers ─────────────────────────────────────────────────────── */

static int wav_header_ex(uint8_t *buf, int n_samples, int sample_rate,
                          TtsEncoding encoding) {
    int bits_per_sample;
    int format_tag;

    switch (encoding) {
    case TTS_ENC_PCM_F32LE:
        bits_per_sample = 32;
        format_tag = 3; /* IEEE float */
        break;
    case TTS_ENC_PCM_MULAW:
        bits_per_sample = 8;
        format_tag = 7; /* mu-law */
        break;
    case TTS_ENC_PCM_ALAW:
        bits_per_sample = 8;
        format_tag = 6; /* A-law */
        break;
    default: /* PCM_S16LE */
        bits_per_sample = 16;
        format_tag = 1; /* PCM */
        break;
    }

    int block_align = bits_per_sample / 8;
    int byte_rate = sample_rate * block_align;
    int data_size = n_samples * block_align;
    int file_size = 44 + data_size - 8;

    memcpy(buf, "RIFF", 4);
    buf[4]  = (uint8_t)(file_size & 0xFF);
    buf[5]  = (uint8_t)((file_size >> 8) & 0xFF);
    buf[6]  = (uint8_t)((file_size >> 16) & 0xFF);
    buf[7]  = (uint8_t)((file_size >> 24) & 0xFF);
    memcpy(buf + 8, "WAVEfmt ", 8);
    buf[16] = 16; buf[17] = buf[18] = buf[19] = 0;
    buf[20] = (uint8_t)(format_tag & 0xFF);
    buf[21] = (uint8_t)((format_tag >> 8) & 0xFF);
    buf[22] = 1; buf[23] = 0; /* mono */
    buf[24] = (uint8_t)(sample_rate & 0xFF);
    buf[25] = (uint8_t)((sample_rate >> 8) & 0xFF);
    buf[26] = (uint8_t)((sample_rate >> 16) & 0xFF);
    buf[27] = (uint8_t)((sample_rate >> 24) & 0xFF);
    buf[28] = (uint8_t)(byte_rate & 0xFF);
    buf[29] = (uint8_t)((byte_rate >> 8) & 0xFF);
    buf[30] = (uint8_t)((byte_rate >> 16) & 0xFF);
    buf[31] = (uint8_t)((byte_rate >> 24) & 0xFF);
    buf[32] = (uint8_t)(block_align & 0xFF);
    buf[33] = 0;
    buf[34] = (uint8_t)(bits_per_sample & 0xFF);
    buf[35] = 0;
    memcpy(buf + 36, "data", 4);
    buf[40] = (uint8_t)(data_size & 0xFF);
    buf[41] = (uint8_t)((data_size >> 8) & 0xFF);
    buf[42] = (uint8_t)((data_size >> 16) & 0xFF);
    buf[43] = (uint8_t)((data_size >> 24) & 0xFF);
    return 44;
}

static int pcm_from_wav(const uint8_t *wav, int wav_len, float **out_pcm) {
    if (wav_len < 44) return -1;
    if (memcmp(wav, "RIFF", 4) != 0 || memcmp(wav + 8, "WAVE", 4) != 0) return -1;
    int bits = wav[34] | (wav[35] << 8);
    int channels = wav[22] | (wav[23] << 8);
    if (bits != 8 && bits != 16 && bits != 24 && bits != 32) {
        fprintf(stderr, "[http] Invalid WAV bits_per_sample: %d\n", bits);
        return -1;
    }
    if (channels < 1 || channels > 2) {
        fprintf(stderr, "[http] Invalid WAV channels: %d\n", channels);
        return -1;
    }
    int data_offset = 44;
    int data_len = wav_len - data_offset;
    int bytes_per_sample = bits / 8;
    int n_samples = data_len / bytes_per_sample / channels;
    float *pcm = malloc((size_t)n_samples * sizeof(float));
    if (!pcm) return -1;
    if (bits == 16) {
        const int16_t *src = (const int16_t *)(wav + data_offset);
        for (int i = 0; i < n_samples; i++)
            pcm[i] = (float)src[i * channels] / 32768.0f;
    } else if (bits == 32) {
        const float *src = (const float *)(wav + data_offset);
        for (int i = 0; i < n_samples; i++)
            pcm[i] = src[i * channels];
    } else {
        free(pcm);
        return -1;
    }
    *out_pcm = pcm;
    return n_samples;
}

/* ─── TTS Request Parsing ─────────────────────────────────────────────── */

static const int VALID_SAMPLE_RATES[] = { 8000, 16000, 22050, 24000, 44100, 48000, 0 };

static int is_valid_sample_rate(int rate) {
    for (int i = 0; VALID_SAMPLE_RATES[i]; i++)
        if (VALID_SAMPLE_RATES[i] == rate) return 1;
    return 0;
}

static TtsRequest parse_tts_request(const char *body, int body_len) {
    TtsRequest req;
    memset(&req, 0, sizeof(req));
    req.speed = 1.0f;
    req.volume = 1.0f;
    req.sample_rate = 24000;
    req.encoding = TTS_ENC_PCM_S16LE;
    req.container = TTS_CONTAINER_WAV;

    if (!body || body_len == 0) return req;

    const char *trimmed = body;
    while (*trimmed == ' ' || *trimmed == '\t' || *trimmed == '\n' || *trimmed == '\r')
        trimmed++;
    if (*trimmed != '{') {
        req.text = strndup(body, (size_t)body_len);
        return req;
    }

    cJSON *root = cJSON_ParseWithLength(body, (size_t)body_len);
    if (!root) {
        req.text = strndup(body, (size_t)body_len);
        return req;
    }

    cJSON *text = cJSON_GetObjectItemCaseSensitive(root, "text");
    if (cJSON_IsString(text) && text->valuestring)
        req.text = strdup(text->valuestring);

    if (!req.text) {
        cJSON *input = cJSON_GetObjectItemCaseSensitive(root, "input");
        if (cJSON_IsString(input) && input->valuestring)
            req.text = strdup(input->valuestring);
    }

    cJSON *voice = cJSON_GetObjectItemCaseSensitive(root, "voice");
    if (cJSON_IsString(voice) && voice->valuestring)
        req.voice = strdup(voice->valuestring);

    cJSON *emotion = cJSON_GetObjectItemCaseSensitive(root, "emotion");
    if (cJSON_IsString(emotion) && emotion->valuestring)
        req.emotion = strdup(emotion->valuestring);

    cJSON *speed = cJSON_GetObjectItemCaseSensitive(root, "speed");
    if (cJSON_IsNumber(speed)) {
        float s = (float)speed->valuedouble;
        if (s < 0.25f) s = 0.25f;
        if (s > 4.0f) s = 4.0f;
        req.speed = s;
    }

    cJSON *vol = cJSON_GetObjectItemCaseSensitive(root, "volume");
    if (cJSON_IsNumber(vol)) {
        float v = (float)vol->valuedouble;
        if (v >= 0.5f && v <= 2.0f) req.volume = v;
    }

    cJSON *stream_j = cJSON_GetObjectItemCaseSensitive(root, "stream");
    if (cJSON_IsBool(stream_j) && cJSON_IsTrue(stream_j))
        req.stream = 1;

    cJSON *wt = cJSON_GetObjectItemCaseSensitive(root, "word_timestamps");
    if (cJSON_IsBool(wt) && cJSON_IsTrue(wt))
        req.word_timestamps = 1;

    cJSON *rfmt = cJSON_GetObjectItemCaseSensitive(root, "response_format");
    if (cJSON_IsString(rfmt) && rfmt->valuestring) {
        if (strcmp(rfmt->valuestring, "opus") == 0)
            req.container = TTS_CONTAINER_OPUS;
        else if (strcmp(rfmt->valuestring, "mp3") == 0)
            req.container = TTS_CONTAINER_MP3;
        else if (strcmp(rfmt->valuestring, "pcm") == 0)
            req.container = TTS_CONTAINER_RAW;
    }

    cJSON *pron = cJSON_GetObjectItemCaseSensitive(root, "pronunciation_overrides");
    if (cJSON_IsArray(pron)) {
        int n = cJSON_GetArraySize(pron);
        if (n > TTS_MAX_PRON_OVERRIDES) n = TTS_MAX_PRON_OVERRIDES;
        int idx = 0;
        cJSON *entry;
        cJSON_ArrayForEach(entry, pron) {
            if (idx >= n) break;
            cJSON *txt = cJSON_GetObjectItemCaseSensitive(entry, "text");
            cJSON *prn = cJSON_GetObjectItemCaseSensitive(entry, "pronunciation");
            if (!cJSON_IsString(txt) || !cJSON_IsString(prn)) continue;
            if (!txt->valuestring[0] || !prn->valuestring[0]) continue;
            snprintf(req.pron_overrides[idx].word, 64, "%s", txt->valuestring);
            snprintf(req.pron_overrides[idx].pronunciation, 256, "%s", prn->valuestring);
            idx++;
        }
        req.n_pron_overrides = idx;
    }

    cJSON *fmt = cJSON_GetObjectItemCaseSensitive(root, "output_format");
    if (cJSON_IsObject(fmt)) {
        cJSON *sr = cJSON_GetObjectItemCaseSensitive(fmt, "sample_rate");
        if (cJSON_IsNumber(sr) && is_valid_sample_rate(sr->valueint))
            req.sample_rate = sr->valueint;

        cJSON *enc = cJSON_GetObjectItemCaseSensitive(fmt, "encoding");
        if (cJSON_IsString(enc) && enc->valuestring) {
            if (strcmp(enc->valuestring, "pcm_f32le") == 0)
                req.encoding = TTS_ENC_PCM_F32LE;
            else if (strcmp(enc->valuestring, "pcm_mulaw") == 0)
                req.encoding = TTS_ENC_PCM_MULAW;
            else if (strcmp(enc->valuestring, "pcm_alaw") == 0)
                req.encoding = TTS_ENC_PCM_ALAW;
        }

        cJSON *ctr = cJSON_GetObjectItemCaseSensitive(fmt, "container");
        if (cJSON_IsString(ctr) && ctr->valuestring) {
            if (strcmp(ctr->valuestring, "raw") == 0)
                req.container = TTS_CONTAINER_RAW;
            else if (strcmp(ctr->valuestring, "mp3") == 0)
                req.container = TTS_CONTAINER_MP3;
            else if (strcmp(ctr->valuestring, "opus") == 0)
                req.container = TTS_CONTAINER_OPUS;
        }
    }

    cJSON_Delete(root);
    return req;
}

/* Encode float32 PCM to the requested encoding. Returns bytes written. */
static int encode_audio(const float *pcm, int n_samples, TtsEncoding encoding,
                         uint8_t *out) {
    switch (encoding) {
    case TTS_ENC_PCM_F32LE:
        memcpy(out, pcm, (size_t)n_samples * sizeof(float));
        return n_samples * 4;

    case TTS_ENC_PCM_MULAW:
        for (int i = 0; i < n_samples; i++) {
            float s = pcm[i] * 32767.0f;
            if (s > 32767.0f) s = 32767.0f;
            if (s < -32768.0f) s = -32768.0f;
            out[i] = encode_mulaw((int16_t)s);
        }
        return n_samples;

    case TTS_ENC_PCM_ALAW:
        for (int i = 0; i < n_samples; i++) {
            float s = pcm[i] * 32767.0f;
            if (s > 32767.0f) s = 32767.0f;
            if (s < -32768.0f) s = -32768.0f;
            out[i] = encode_alaw((int16_t)s);
        }
        return n_samples;

    default: { /* PCM_S16LE */
        int16_t *dst = (int16_t *)out;
        for (int i = 0; i < n_samples; i++) {
            float s = pcm[i] * 32767.0f;
            if (s > 32767.0f) s = 32767.0f;
            if (s < -32768.0f) s = -32768.0f;
            dst[i] = (int16_t)s;
        }
        return n_samples * 2;
    }
    }
}

/* ─── Ogg Opus Encoder (RFC 7845) ──────────────────────────────────────── */

#define OPUS_APPLICATION_AUDIO 2049
#define OGG_SERIAL 0x534F4E41 /* "SONA" */

static uint32_t ogg_crc_table[256];
static int ogg_crc_inited = 0;

static void ogg_crc_init(void) {
    if (ogg_crc_inited) return;
    for (int i = 0; i < 256; i++) {
        uint32_t r = (uint32_t)i << 24;
        for (int j = 0; j < 8; j++)
            r = (r << 1) ^ ((r & 0x80000000) ? 0x04C11DB7 : 0);
        ogg_crc_table[i] = r;
    }
    ogg_crc_inited = 1;
}

static uint32_t ogg_crc(const uint8_t *data, int len) {
    uint32_t crc = 0;
    for (int i = 0; i < len; i++)
        crc = (crc << 8) ^ ogg_crc_table[((crc >> 24) ^ data[i]) & 0xFF];
    return crc;
}

static void le16(uint8_t *p, uint16_t v) { p[0] = v & 0xFF; p[1] = v >> 8; }
static void le32(uint8_t *p, uint32_t v) {
    p[0] = v & 0xFF; p[1] = (v >> 8) & 0xFF;
    p[2] = (v >> 16) & 0xFF; p[3] = (v >> 24) & 0xFF;
}
static void le64(uint8_t *p, uint64_t v) {
    le32(p, (uint32_t)(v & 0xFFFFFFFF));
    le32(p + 4, (uint32_t)(v >> 32));
}

static int ogg_write_page(uint8_t *out, int max_out, uint8_t type,
                           uint64_t granule, uint32_t serial, uint32_t seq,
                           const uint8_t *data, int data_len) {
    int n_segments = (data_len + 254) / 255;
    if (n_segments < 1) n_segments = 1;
    int header_size = 27 + n_segments;
    int page_size = header_size + data_len;
    if (page_size > max_out) return -1;

    memcpy(out, "OggS", 4);
    out[4] = 0;         /* version */
    out[5] = type;      /* header_type */
    le64(out + 6, granule);
    le32(out + 14, serial);
    le32(out + 18, seq);
    le32(out + 22, 0);  /* CRC placeholder */
    out[26] = (uint8_t)n_segments;

    for (int i = 0; i < n_segments - 1; i++)
        out[27 + i] = 255;
    out[27 + n_segments - 1] = (uint8_t)(data_len - 255 * (n_segments - 1));

    memcpy(out + header_size, data, (size_t)data_len);

    ogg_crc_init();
    uint32_t c = ogg_crc(out, page_size);
    le32(out + 22, c);

    return page_size;
}

static int encode_ogg_opus(const float *pcm, int n_samples, int sample_rate,
                            uint8_t *out, int max_out) {
    uint32_t serial = OGG_SERIAL;
    uint32_t page_seq = 0;
    int pos = 0;

    /* Page 0: OpusHead */
    uint8_t opus_head[19];
    memcpy(opus_head, "OpusHead", 8);
    opus_head[8] = 1;   /* version */
    opus_head[9] = 1;   /* channels */
    le16(opus_head + 10, 312);  /* pre-skip (encoder delay at 48kHz) */
    le32(opus_head + 12, (uint32_t)sample_rate);
    le16(opus_head + 16, 0);    /* output gain */
    opus_head[18] = 0;          /* channel mapping family */

    int w = ogg_write_page(out + pos, max_out - pos, 0x02, 0,
                            serial, page_seq++, opus_head, 19);
    if (w < 0) return -1;
    pos += w;

    /* Page 1: OpusTags */
    uint8_t opus_tags[26];
    memcpy(opus_tags, "OpusTags", 8);
    le32(opus_tags + 8, 6);
    memcpy(opus_tags + 12, "Sonata", 6);
    le32(opus_tags + 18, 0);  /* no user comments */
    /* pad to keep it simple — 22 bytes total */

    w = ogg_write_page(out + pos, max_out - pos, 0x00, 0,
                        serial, page_seq++, opus_tags, 22);
    if (w < 0) return -1;
    pos += w;

    /* Audio pages: encode 20ms Opus frames, pack into Ogg pages */
    PocketOpus *enc = pocket_opus_create(sample_rate, 1, 64000, 20.0f,
                                          OPUS_APPLICATION_AUDIO);
    if (!enc) return pos; /* return what we have (headers only) */

    int frame = sample_rate / 50;
    uint64_t granule = 312; /* start after pre-skip */
    uint8_t page_data[16384];
    uint8_t seg_sizes[255];
    int page_data_len = 0;
    int n_segs = 0;

    for (int off = 0; off < n_samples; off += frame) {
        int n = (n_samples - off < frame) ? n_samples - off : frame;
        uint8_t pkt[4000];
        int pkt_len = pocket_opus_encode(enc, pcm + off, n, pkt, sizeof(pkt));
        if (pkt_len <= 0) continue;

        /* Opus encodes at 48kHz internally regardless of input rate */
        granule += 960; /* 20ms at 48kHz */

        int pkt_segs = (pkt_len + 254) / 255;
        if (n_segs + pkt_segs > 255 || page_data_len + pkt_len > (int)sizeof(page_data)) {
            /* Flush current page */
            w = ogg_write_page(out + pos, max_out - pos, 0x00, granule - 960,
                                serial, page_seq++, page_data, page_data_len);
            if (w < 0) break;
            pos += w;
            page_data_len = 0;
            n_segs = 0;
        }

        memcpy(page_data + page_data_len, pkt, (size_t)pkt_len);
        page_data_len += pkt_len;
        for (int s = 0; s < pkt_segs - 1; s++)
            seg_sizes[n_segs++] = 255;
        seg_sizes[n_segs++] = (uint8_t)(pkt_len - 255 * (pkt_segs - 1));
    }

    pocket_opus_destroy(enc);

    /* Flush final page with EOS flag */
    if (page_data_len > 0) {
        w = ogg_write_page(out + pos, max_out - pos, 0x04, granule,
                            serial, page_seq++, page_data, page_data_len);
        if (w > 0) pos += w;
    }

    return pos;
}

/* ─── Word Timestamp Estimation ───────────────────────────────────────── */

static int estimate_word_timestamps(const char *text, int n_samples,
                                     int sample_rate, WordTimestamp *out,
                                     int max_words) {
    if (!text || !out || max_words <= 0 || n_samples <= 0) return 0;

    /* Tokenize words by spaces */
    char buf[8192];
    snprintf(buf, sizeof(buf), "%s", text);

    char *words[TTS_MAX_WORD_TIMESTAMPS];
    int charlen[TTS_MAX_WORD_TIMESTAMPS];
    int n_words = 0;
    int total_chars = 0;

    char *p = buf;
    while (*p && n_words < max_words && n_words < TTS_MAX_WORD_TIMESTAMPS) {
        while (*p == ' ' || *p == '\t' || *p == '\n') p++;
        if (!*p) break;
        words[n_words] = p;
        int wlen = 0;
        while (*p && *p != ' ' && *p != '\t' && *p != '\n') { p++; wlen++; }
        if (*p) *p++ = '\0';
        charlen[n_words] = wlen;
        total_chars += wlen;
        n_words++;
    }

    if (n_words == 0 || total_chars == 0) return 0;

    float total_duration = (float)n_samples / (float)sample_rate;
    float cursor = 0.0f;

    for (int i = 0; i < n_words; i++) {
        float word_dur = total_duration * ((float)charlen[i] / (float)total_chars);
        snprintf(out[i].word, sizeof(out[i].word), "%s", words[i]);
        out[i].start_s = cursor;
        out[i].end_s = cursor + word_dur;
        cursor += word_dur;
    }

    return n_words;
}

/* ─── Endpoint Handlers ───────────────────────────────────────────────── */

static void handle_health(int fd) {
    send_json(fd, 200, "{\"status\":\"ok\",\"version\":\"pocket-voice 1.0\"}");
}

static void handle_stt(int fd, HttpRequest *req, HttpApiEngines *eng) {
    if (!req->body || req->body_len < 44) {
        send_json(fd, 400, "{\"error\":\"Body must contain WAV audio\"}");
        return;
    }

    float *pcm = NULL;
    int n = pcm_from_wav((const uint8_t *)req->body, req->body_len, &pcm);
    if (n < 0 || !pcm) {
        send_json(fd, 400, "{\"error\":\"Invalid WAV format\"}");
        return;
    }

    eng->stt_reset(eng->stt_engine);

    int chunk = 4800; /* 100ms at 48kHz */
    for (int off = 0; off < n; off += chunk) {
        int len = (n - off < chunk) ? n - off : chunk;
        eng->stt_feed(eng->stt_engine, pcm + off, len);
    }
    eng->stt_flush(eng->stt_engine);

    char text[4096] = {0};
    eng->stt_get_text(eng->stt_engine, text, sizeof(text));

    /* Word timestamps: ?word_timestamps=true or path contains it */
    int want_word_timestamps = (strstr(req->path, "word_timestamps") != NULL);
    int sample_rate = 24000;
    if (req->body_len >= 28)
        sample_rate = (int)(req->body[24] | (req->body[25]<<8) |
                            (req->body[26]<<16) | (req->body[27]<<24));

    if (want_word_timestamps && eng->stt_get_words) {
        WordTimestamp wts[TTS_MAX_WORD_TIMESTAMPS];
        int n_wts = eng->stt_get_words(eng->stt_engine, wts, TTS_MAX_WORD_TIMESTAMPS);
        if (n_wts > 0) {
            char resp[32768];
            int jpos = 0;
            char escaped[256];
            jpos = snprintf(resp, sizeof(resp),
                "{\"text\":\"");
            int tlen = (int)strlen(text);
            if (json_escape(text, tlen, escaped, sizeof(escaped)) >= 0)
                jpos += snprintf(resp + jpos, sizeof(resp) - (size_t)jpos, "%s", escaped);
            else
                jpos += snprintf(resp + jpos, sizeof(resp) - (size_t)jpos, "%s", text);
            jpos += snprintf(resp + jpos, sizeof(resp) - (size_t)jpos,
                "\",\"samples\":%d,\"word_timestamps\":[", n);
            for (int i = 0; i < n_wts && jpos < (int)sizeof(resp) - 256; i++) {
                if (json_escape(wts[i].word, (int)strlen(wts[i].word), escaped, sizeof(escaped)) < 0)
                    snprintf(escaped, sizeof(escaped), "%s", wts[i].word);
                jpos += snprintf(resp + jpos, sizeof(resp) - (size_t)jpos,
                    "%s{\"word\":\"%s\",\"start\":%.3f,\"end\":%.3f}",
                    i > 0 ? "," : "", escaped,
                    (double)wts[i].start_s, (double)wts[i].end_s);
            }
            jpos += snprintf(resp + jpos, sizeof(resp) - (size_t)jpos, "]}");
            send_json(fd, 200, resp);
            free(pcm);
            return;
        }
        /* Fall through to basic response if get_words failed */
    }

    /* Fallback: also use estimated timestamps when stt_get_words unavailable */
    if (want_word_timestamps && !eng->stt_get_words) {
        WordTimestamp wts[TTS_MAX_WORD_TIMESTAMPS];
        int n_wts = estimate_word_timestamps(text, n, sample_rate, wts, TTS_MAX_WORD_TIMESTAMPS);
        if (n_wts > 0) {
            char resp[32768];
            int jpos = 0;
            char escaped[256];
            jpos = snprintf(resp, sizeof(resp), "{\"text\":\"");
            int tlen = (int)strlen(text);
            if (json_escape(text, tlen, escaped, sizeof(escaped)) >= 0)
                jpos += snprintf(resp + jpos, sizeof(resp) - (size_t)jpos, "%s", escaped);
            else
                jpos += snprintf(resp + jpos, sizeof(resp) - (size_t)jpos, "%s", text);
            jpos += snprintf(resp + jpos, sizeof(resp) - (size_t)jpos,
                "\",\"samples\":%d,\"word_timestamps\":[", n);
            for (int i = 0; i < n_wts && jpos < (int)sizeof(resp) - 256; i++) {
                if (json_escape(wts[i].word, (int)strlen(wts[i].word), escaped, sizeof(escaped)) < 0)
                    snprintf(escaped, sizeof(escaped), "%s", wts[i].word);
                jpos += snprintf(resp + jpos, sizeof(resp) - (size_t)jpos,
                    "%s{\"word\":\"%s\",\"start\":%.3f,\"end\":%.3f}",
                    i > 0 ? "," : "", escaped,
                    (double)wts[i].start_s, (double)wts[i].end_s);
            }
            jpos += snprintf(resp + jpos, sizeof(resp) - (size_t)jpos, "]}");
            send_json(fd, 200, resp);
            free(pcm);
            return;
        }
    }

    char resp[8192];
    snprintf(resp, sizeof(resp),
             "{\"text\":\"%s\",\"samples\":%d}", text, n);
    send_json(fd, 200, resp);
    free(pcm);
}

/* ─── Streaming TTS: chunked Transfer-Encoding ────────────────────────── */

static void send_chunked_header(int fd, const char *content_type) {
    char header[512];
    int hlen = snprintf(header, sizeof(header),
        "HTTP/1.1 200 OK\r\n"
        "Content-Type: %s\r\n"
        "Transfer-Encoding: chunked\r\n"
        "Access-Control-Allow-Origin: *\r\n"
        "Connection: close\r\n"
        "\r\n", content_type);
    write(fd, header, (size_t)hlen);
}

static int send_chunk(int fd, const void *data, int len) {
    char size_line[32];
    int slen = snprintf(size_line, sizeof(size_line), "%x\r\n", len);
    if (write(fd, size_line, (size_t)slen) < 0) return -1;
    if (len > 0 && write(fd, data, (size_t)len) < 0) return -1;
    if (write(fd, "\r\n", 2) < 0) return -1;
    return 0;
}

static void send_chunk_end(int fd) {
    write(fd, "0\r\n\r\n", 5);
}

/* Build pronunciation override arrays from TtsRequest for process_text */
static void build_pron_arrays(const TtsRequest *treq,
                               char (*words)[64], char (*repls)[256]) {
    for (int i = 0; i < treq->n_pron_overrides; i++) {
        snprintf(words[i], 64, "%s", treq->pron_overrides[i].word);
        snprintf(repls[i], 256, "%s", treq->pron_overrides[i].pronunciation);
    }
}

static void handle_tts(int fd, HttpRequest *req, HttpApi *api) {
    HttpApiEngines *eng = &api->eng;
    if (!req->body || req->body_len == 0) {
        send_json(fd, 400, "{\"error\":\"Body must contain text or JSON\"}");
        return;
    }

    TtsRequest treq = parse_tts_request(req->body, req->body_len);
    if (!treq.text || !*treq.text) {
        tts_request_cleanup(&treq);
        send_json(fd, 400, "{\"error\":\"Missing or empty text field\"}");
        return;
    }

    if ((int)strlen(treq.text) > MAX_TEXT_SIZE) {
        tts_request_cleanup(&treq);
        send_json(fd, 413, "{\"error\":\"Input text exceeds 10KB limit\"}");
        return;
    }

    map_openai_voice(&treq);

    /* Prepare pronunciation override arrays */
    char pron_words[TTS_MAX_PRON_OVERRIDES][64];
    char pron_repls[TTS_MAX_PRON_OVERRIDES][256];
    if (treq.n_pron_overrides > 0)
        build_pron_arrays(&treq, pron_words, pron_repls);

    /* Serialize all TTS engine access — single engine, multiple workers */
    pthread_mutex_lock(&api->tts_mutex);

    /* Apply cloned voice if voice_id matches registry */
    if (treq.voice)
        apply_cloned_voice(api, treq.voice);

    /* Route through full prosody pipeline if available */
    if (eng->process_text) {
        int pret = eng->process_text(eng->process_ctx, eng->tts_engine, treq.text,
                          treq.speed, treq.volume,
                          treq.emotion ? treq.emotion : "",
                          treq.voice,
                          treq.n_pron_overrides > 0 ? (const char (*)[64])pron_words : NULL,
                          treq.n_pron_overrides > 0 ? (const char (*)[256])pron_repls : NULL,
                          treq.n_pron_overrides);
        if (pret != 0) {
            pthread_mutex_unlock(&api->tts_mutex);
            tts_request_cleanup(&treq);
            send_json(fd, 500, "{\"error\":\"TTS synthesis failed\"}");
            return;
        }
    } else {
        eng->tts_reset(eng->tts_engine);
        eng->tts_speak(eng->tts_engine, treq.text);
        int sret = eng->tts_set_text_done(eng->tts_engine);
        if (sret != 0) {
            pthread_mutex_unlock(&api->tts_mutex);
            tts_request_cleanup(&treq);
            send_json(fd, 500, "{\"error\":\"TTS synthesis failed\"}");
            return;
        }
    }

    /* ── Streaming path: chunked Transfer-Encoding ──────────────────── */
    if (treq.stream) {
        const char *ct;
        switch (treq.container) {
        case TTS_CONTAINER_RAW:  ct = "application/octet-stream"; break;
        case TTS_CONTAINER_OPUS: ct = "audio/opus"; break;
        case TTS_CONTAINER_MP3:  ct = "audio/mpeg"; break;
        default:                 ct = "audio/wav"; break;
        }
        send_chunked_header(fd, ct);

        const int native_rate = 24000;
        HWResampler *resampler = NULL;
        if (treq.sample_rate != native_rate)
            resampler = hw_resampler_create(native_rate, treq.sample_rate, 1, 3);

        int first_chunk = 1;
        float audio_buf[8192];
        int max_chunk = (int)(sizeof(audio_buf) / sizeof(float));

        while (!eng->tts_is_done(eng->tts_engine)) {
            eng->tts_step(eng->tts_engine);
            int got = eng->tts_get_audio(eng->tts_engine, audio_buf, max_chunk);
            if (got <= 0) continue;

            float *samples = audio_buf;
            int n_samples = got;

            /* Resample if needed */
            float resamp_buf[16384];
            if (resampler) {
                int max_out = (int)((double)got * treq.sample_rate / native_rate) + 256;
                if (max_out > 16384) max_out = 16384;
                int rn = hw_resample(resampler, audio_buf, got, resamp_buf, max_out);
                if (rn > 0) { samples = resamp_buf; n_samples = rn; }
            }

            /* Volume scaling */
            if (fabsf(treq.volume - 1.0f) > 0.01f) {
                for (int i = 0; i < n_samples; i++) {
                    samples[i] *= treq.volume;
                    if (samples[i] > 1.0f) samples[i] = 1.0f;
                    if (samples[i] < -1.0f) samples[i] = -1.0f;
                }
            }

            /* Send WAV header in the first chunk if WAV container */
            if (first_chunk && treq.container == TTS_CONTAINER_WAV) {
                uint8_t hdr[44];
                wav_header_ex(hdr, 0, treq.sample_rate, treq.encoding);
                send_chunk(fd, hdr, 44);
                first_chunk = 0;
            }

            /* Encode and send chunk */
            int enc_size = n_samples * 4;
            uint8_t *enc_buf = malloc((size_t)enc_size);
            if (!enc_buf) {
                fprintf(stderr, "[http] TTS stream chunk alloc failed\n");
                break;
            }
            int enc_len = encode_audio(samples, n_samples, treq.encoding, enc_buf);
            if (send_chunk(fd, enc_buf, enc_len) < 0) { free(enc_buf); break; }
            free(enc_buf);
        }

        send_chunk_end(fd);
        if (resampler) hw_resampler_destroy(resampler);
        pthread_mutex_unlock(&api->tts_mutex);
        tts_request_cleanup(&treq);
        return;
    }

    /* ── Non-streaming path: collect all audio, encode, respond ───── */

    float *audio = malloc(MAX_RESP_SIZE);
    if (!audio) { pthread_mutex_unlock(&api->tts_mutex); tts_request_cleanup(&treq); send_json(fd, 500, "{\"error\":\"OOM\"}"); return; }
    int total = 0;
    int max_samples = MAX_RESP_SIZE / (int)sizeof(float);

    while (!eng->tts_is_done(eng->tts_engine)) {
        eng->tts_step(eng->tts_engine);
        int got = eng->tts_get_audio(eng->tts_engine,
                                     audio + total, max_samples - total);
        if (got > 0) total += got;
        if (total >= max_samples) break;
    }

    pthread_mutex_unlock(&api->tts_mutex);

    if (total == 0) {
        free(audio);
        tts_request_cleanup(&treq);
        send_json(fd, 500, "{\"error\":\"TTS produced no audio\"}");
        return;
    }

    /* Resample if target rate differs from native 24kHz */
    float *resampled = audio;
    int out_samples = total;
    const int native_rate = 24000;
    HWResampler *resampler = NULL;

    if (treq.sample_rate != native_rate) {
        resampler = hw_resampler_create(native_rate, treq.sample_rate, 1, 3);
        if (resampler) {
            int max_out = (int)((double)total * treq.sample_rate / native_rate) + 1024;
            resampled = malloc((size_t)max_out * sizeof(float));
            if (resampled) {
                out_samples = hw_resample(resampler, audio, total, resampled, max_out);
                if (out_samples <= 0) {
                    free(resampled);
                    resampled = audio;
                    out_samples = total;
                    treq.sample_rate = native_rate;
                }
            } else {
                resampled = audio;
                treq.sample_rate = native_rate;
            }
            hw_resampler_destroy(resampler);
        } else {
            treq.sample_rate = native_rate;
        }
    }

    /* Apply volume scaling if non-default */
    if (fabsf(treq.volume - 1.0f) > 0.01f) {
        for (int i = 0; i < out_samples; i++) {
            resampled[i] *= treq.volume;
            if (resampled[i] > 1.0f) resampled[i] = 1.0f;
            if (resampled[i] < -1.0f) resampled[i] = -1.0f;
        }
    }

    /* Determine bytes per sample for the target encoding */
    int bytes_per_sample;
    switch (treq.encoding) {
    case TTS_ENC_PCM_F32LE: bytes_per_sample = 4; break;
    case TTS_ENC_PCM_MULAW: bytes_per_sample = 1; break;
    case TTS_ENC_PCM_ALAW:  bytes_per_sample = 1; break;
    default:                bytes_per_sample = 2; break;
    }

    /* MP3 path: encode directly from float PCM (ignores encoding param) */
    if (treq.container == TTS_CONTAINER_MP3) {
        int max_mp3 = (int)(1.25 * out_samples + 7200) + 1024;
        uint8_t *mp3_buf = malloc((size_t)max_mp3);
        if (!mp3_buf) {
            if (resampled != audio) free(resampled);
            free(audio);
            tts_request_cleanup(&treq);
            send_json(fd, 500, "{\"error\":\"OOM\"}");
            return;
        }

        int mp3_len = encode_mp3(resampled, out_samples, treq.sample_rate,
                                  mp3_buf, max_mp3);
        if (mp3_len <= 0) {
            free(mp3_buf);
            if (resampled != audio) free(resampled);
            free(audio);
            tts_request_cleanup(&treq);
            send_json(fd, 500, "{\"error\":\"MP3 encoding failed (is libmp3lame installed?)\"}");
            return;
        }

        send_response(fd, 200, "audio/mpeg", mp3_buf, mp3_len);
        free(mp3_buf);
        if (resampled != audio) free(resampled);
        free(audio);
        tts_request_cleanup(&treq);
        return;
    }

    /* Ogg Opus path (RFC 7845 compliant container) */
    if (treq.container == TTS_CONTAINER_OPUS) {
        int max_opus = out_samples * 2 + 8192;
        uint8_t *opus_buf = malloc((size_t)max_opus);
        if (!opus_buf) {
            if (resampled != audio) free(resampled);
            free(audio);
            tts_request_cleanup(&treq);
            send_json(fd, 500, "{\"error\":\"OOM\"}");
            return;
        }

        int opus_len = encode_ogg_opus(resampled, out_samples, treq.sample_rate,
                                        opus_buf, max_opus);
        if (opus_len <= 0) {
            free(opus_buf);
            if (resampled != audio) free(resampled);
            free(audio);
            tts_request_cleanup(&treq);
            send_json(fd, 500, "{\"error\":\"Opus encoding failed\"}");
            return;
        }

        send_response(fd, 200, "audio/ogg", opus_buf, opus_len);
        free(opus_buf);
        if (resampled != audio) free(resampled);
        free(audio);
        tts_request_cleanup(&treq);
        return;
    }

    /* Word timestamps JSON response (returns JSON with base64-encoded audio
     * or just the timestamps with a separate audio endpoint) */
    if (treq.word_timestamps && treq.text) {
        WordTimestamp wts[TTS_MAX_WORD_TIMESTAMPS];
        int n_wts = 0;
        if (eng->tts_get_words) {
            n_wts = eng->tts_get_words(eng->tts_engine, wts, TTS_MAX_WORD_TIMESTAMPS);
        }
        if (n_wts <= 0) {
            n_wts = estimate_word_timestamps(treq.text, out_samples,
                                              treq.sample_rate, wts,
                                              TTS_MAX_WORD_TIMESTAMPS);
        }

        /* Build JSON response with timestamps + audio duration */
        float duration = (float)out_samples / (float)treq.sample_rate;
        char json_buf[32768];
        int jpos = snprintf(json_buf, sizeof(json_buf),
            "{\"duration\":%.3f,\"sample_rate\":%d,\"samples\":%d,"
            "\"word_timestamps\":[",
            (double)duration, treq.sample_rate, out_samples);

        for (int i = 0; i < n_wts && jpos < (int)sizeof(json_buf) - 256; i++) {
            char escaped_word[256];
            json_escape(wts[i].word, (int)strlen(wts[i].word),
                       escaped_word, sizeof(escaped_word));
            jpos += snprintf(json_buf + jpos, sizeof(json_buf) - (size_t)jpos,
                "%s{\"word\":\"%s\",\"start\":%.3f,\"end\":%.3f}",
                i > 0 ? "," : "", escaped_word,
                (double)wts[i].start_s, (double)wts[i].end_s);
        }

        jpos += snprintf(json_buf + jpos, sizeof(json_buf) - (size_t)jpos, "]}");

        /* Send WAV audio with JSON timestamps header */
        int data_size = out_samples * bytes_per_sample;
        int wav_size = 44 + data_size;
        uint8_t *wav_buf = malloc((size_t)wav_size);
        if (wav_buf) {
            wav_header_ex(wav_buf, out_samples, treq.sample_rate, treq.encoding);
            encode_audio(resampled, out_samples, treq.encoding, wav_buf + 44);

            /* Multipart-ish: send JSON first, then audio */
            char header[1024];
            int hlen = snprintf(header, sizeof(header),
                "HTTP/1.1 200 OK\r\n"
                "Content-Type: application/json\r\n"
                "Content-Length: %d\r\n"
                "X-Audio-Length: %d\r\n"
                "X-Audio-Content-Type: audio/wav\r\n"
                "Access-Control-Allow-Origin: *\r\n"
                "Access-Control-Expose-Headers: X-Audio-Length, X-Audio-Content-Type\r\n"
                "Connection: close\r\n"
                "\r\n",
                jpos, wav_size);
            write(fd, header, (size_t)hlen);
            write(fd, json_buf, (size_t)jpos);
            free(wav_buf);
        } else {
            send_json(fd, 200, json_buf);
        }

        if (resampled != audio) free(resampled);
        free(audio);
        tts_request_cleanup(&treq);
        return;
    }

    /* PCM path: WAV or raw container */
    int data_size = out_samples * bytes_per_sample;
    int header_size = (treq.container == TTS_CONTAINER_WAV) ? 44 : 0;
    int resp_size = header_size + data_size;

    uint8_t *resp_buf = malloc((size_t)resp_size);
    if (!resp_buf) {
        if (resampled != audio) free(resampled);
        free(audio);
        tts_request_cleanup(&treq);
        send_json(fd, 500, "{\"error\":\"OOM\"}");
        return;
    }

    if (treq.container == TTS_CONTAINER_WAV)
        wav_header_ex(resp_buf, out_samples, treq.sample_rate, treq.encoding);

    encode_audio(resampled, out_samples, treq.encoding, resp_buf + header_size);

    const char *content_type;
    switch (treq.container) {
    case TTS_CONTAINER_RAW:
        switch (treq.encoding) {
        case TTS_ENC_PCM_F32LE: content_type = "audio/pcm;encoding=float"; break;
        case TTS_ENC_PCM_MULAW: content_type = "audio/basic"; break;
        case TTS_ENC_PCM_ALAW:  content_type = "audio/basic"; break;
        default:                content_type = "application/octet-stream"; break;
        }
        break;
    default:
        content_type = "audio/wav";
        break;
    }

    send_response(fd, 200, content_type, resp_buf, resp_size);

    free(resp_buf);
    if (resampled != audio) free(resampled);
    free(audio);
    tts_request_cleanup(&treq);
}

/* ─── Voice Clone Handler ──────────────────────────────────────────────── */

static void handle_voice_create(int fd, HttpRequest *req, HttpApi *api) {
    if (!api->eng.clone_voice) {
        send_json(fd, 501, "{\"error\":\"Voice cloning not available (no speaker encoder loaded)\"}");
        return;
    }

    if (!req->body || req->body_len < 44) {
        send_json(fd, 400, "{\"error\":\"Body must contain WAV audio\"}");
        return;
    }

    float *pcm = NULL;
    int n = pcm_from_wav((const uint8_t *)req->body, req->body_len, &pcm);
    if (n < 0 || !pcm) {
        send_json(fd, 400, "{\"error\":\"Invalid WAV format\"}");
        return;
    }

    float embedding[VOICE_EMBED_MAX_DIM];
    int dim = api->eng.clone_voice(api->eng.clone_ctx, pcm, n,
                                    embedding, VOICE_EMBED_MAX_DIM);
    free(pcm);

    if (dim <= 0) {
        send_json(fd, 500, "{\"error\":\"Speaker embedding extraction failed\"}");
        return;
    }

    /* Find a free slot and store */
    pthread_mutex_lock(&api->voice_mutex);
    int slot = -1;
    for (int i = 0; i < VOICE_REGISTRY_MAX; i++) {
        if (!api->voices[i].used) { slot = i; break; }
    }
    if (slot < 0) {
        /* Evict oldest (slot 0) and shift */
        slot = VOICE_REGISTRY_MAX - 1;
        for (int i = 0; i < VOICE_REGISTRY_MAX - 1; i++)
            api->voices[i] = api->voices[i + 1];
    }

    snprintf(api->voices[slot].id, sizeof(api->voices[slot].id),
             "voice_%d", slot);
    memcpy(api->voices[slot].embedding, embedding,
           (size_t)dim * sizeof(float));
    api->voices[slot].dim = dim;
    api->voices[slot].used = 1;

    char voice_id[64];
    snprintf(voice_id, sizeof(voice_id), "%s", api->voices[slot].id);
    pthread_mutex_unlock(&api->voice_mutex);

    char resp[256];
    snprintf(resp, sizeof(resp),
             "{\"voice_id\":\"%s\",\"embedding_dim\":%d}", voice_id, dim);
    send_json(fd, 200, resp);
}

static void handle_voice_list(int fd, HttpApi *api) {
    char resp[4096] = "{\"voices\":[";
    int pos = (int)strlen(resp);

    pthread_mutex_lock(&api->voice_mutex);
    int first = 1;
    for (int i = 0; i < VOICE_REGISTRY_MAX; i++) {
        if (!api->voices[i].used) continue;
        if (!first) resp[pos++] = ',';
        pos += snprintf(resp + pos, sizeof(resp) - (size_t)pos,
                        "{\"voice_id\":\"%s\",\"embedding_dim\":%d}",
                        api->voices[i].id, api->voices[i].dim);
        first = 0;
    }
    pthread_mutex_unlock(&api->voice_mutex);

    snprintf(resp + pos, sizeof(resp) - (size_t)pos, "]}");
    send_json(fd, 200, resp);
}

/* Look up a cloned voice by ID and set it on the TTS engine */
static int apply_cloned_voice(HttpApi *api, const char *voice_id) {
    if (!voice_id || !*voice_id || !api->eng.set_speaker_embedding) return 0;

    pthread_mutex_lock(&api->voice_mutex);
    for (int i = 0; i < VOICE_REGISTRY_MAX; i++) {
        if (api->voices[i].used && strcmp(api->voices[i].id, voice_id) == 0) {
            int dim = api->voices[i].dim;
            float emb[VOICE_EMBED_MAX_DIM];
            memcpy(emb, api->voices[i].embedding, (size_t)dim * sizeof(float));
            pthread_mutex_unlock(&api->voice_mutex);
            return api->eng.set_speaker_embedding(api->eng.tts_engine, emb, dim);
        }
    }
    pthread_mutex_unlock(&api->voice_mutex);
    return -1; /* not found — let the pipeline handle it as a name or ID */
}

/* ─── WebSocket Stream Handler ─────────────────────────────────────────── */

#define WS_BUF_SIZE 65536
#define WS_JSON_BUF 4096
#define OPUS_DECODE_BUF 4800

#define OPUS_APPLICATION_VOIP 2048

static int json_escape(const char *in, int in_len, char *out, int out_size) {
    int j = 0;
    for (int i = 0; i < in_len && j < out_size - 2; i++) {
        char c = in[i];
        if (c == '"' || c == '\\') {
            if (j + 2 >= out_size) break;
            out[j++] = '\\';
            out[j++] = c;
        } else if (c == '\n') {
            if (j + 2 >= out_size) break;
            out[j++] = '\\';
            out[j++] = 'n';
        } else {
            out[j++] = c;
        }
    }
    out[j] = '\0';
    return j;
}

static bool parse_config_opus(const char *txt) {
    return txt && strstr(txt, "\"config\"") && strstr(txt, "\"codec\"") && strstr(txt, "\"opus\"");
}

static void handle_websocket_stream(int fd, const char *ws_key, HttpApiEngines *eng) {
    WebSocket *ws = ws_upgrade(fd, ws_key);
    if (!ws) {
        send_json(fd, 400, "{\"error\":\"WebSocket upgrade failed\"}");
        close(fd);
        return;
    }

    if (!eng->stt_engine || !eng->tts_engine || !eng->llm_engine) {
        ws_send_text(ws, "{\"type\":\"error\",\"message\":\"Engines not available\"}");
        ws_close(ws);
        ws_destroy(ws);
        return;
    }

    ws_send_text(ws, "{\"type\":\"listening\"}");

    eng->stt_reset(eng->stt_engine);

    bool use_opus = false;
    PocketOpus *opus_enc = NULL;
    PocketOpus *opus_dec = NULL;
    float decode_buf[OPUS_DECODE_BUF];

    uint8_t recv_buf[WS_BUF_SIZE];
    unsigned char opus_packet[4000];
    WsOpcode type;
    bool running = true;

    while (running && ws_is_connected(ws)) {
        int n = ws_recv(ws, &type, recv_buf, sizeof(recv_buf));
        if (n < 0) break;
        if (n == 0 && type == WS_CLOSE) break;

        if (type == WS_BINARY) {
            if (use_opus && opus_dec) {
                int ns = pocket_opus_decode(opus_dec, recv_buf, n, decode_buf, OPUS_DECODE_BUF);
                if (ns > 0)
                    eng->stt_feed(eng->stt_engine, decode_buf, ns);
            } else {
                int n_samples = n / 4;
                if (n_samples > 0)
                    eng->stt_feed(eng->stt_engine, (const float *)recv_buf, n_samples);
            }
        } else if (type == WS_TEXT) {
            recv_buf[n] = '\0';
            const char *txt = (const char *)recv_buf;

            if (parse_config_opus(txt) && !use_opus) {
                opus_dec = pocket_opus_create(16000, 1, 24000, 20.0f, OPUS_APPLICATION_VOIP);
                opus_enc = pocket_opus_create(24000, 1, 24000, 20.0f, OPUS_APPLICATION_VOIP);
                if (opus_dec && opus_enc) {
                    use_opus = true;
                    ws_send_text(ws, "{\"type\":\"config_ack\",\"codec\":\"opus\",\"sample_rate\":16000}");
                }
            }

            if (strcmp(txt, "flush") == 0 || strcmp(txt, "end") == 0 || strcmp(txt, "") == 0) {
                eng->stt_flush(eng->stt_engine);

                char transcript[4096] = {0};
                eng->stt_get_text(eng->stt_engine, transcript, sizeof(transcript));

                if (transcript[0] != '\0') {
                    char escaped[4096];
                    int elen = json_escape(transcript, (int)strlen(transcript), escaped, sizeof(escaped));
                    char msg[WS_JSON_BUF];
                    if (elen >= 0)
                        snprintf(msg, sizeof(msg), "{\"type\":\"transcript\",\"text\":\"%s\"}", escaped);
                    else
                        snprintf(msg, sizeof(msg), "{\"type\":\"transcript\",\"text\":\"\"}");
                    ws_send_text(ws, msg);

                    ws_send_text(ws, "{\"type\":\"processing\"}");

                    if (eng->llm_send(eng->llm_engine, transcript) == 0) {
                        char response[16384] = {0};
                        int resp_len = 0;
                        int polls = 0;

                        while (!eng->llm_is_done(eng->llm_engine) && polls < 30000) {
                            eng->llm_poll(eng->llm_engine, 10);
                            int tok_len = 0;
                            const char *tok = eng->llm_peek(eng->llm_engine, &tok_len);
                            if (tok && tok_len > 0) {
                                char tok_esc[2048];
                                json_escape(tok, tok_len, tok_esc, sizeof(tok_esc));
                                char tok_msg[WS_JSON_BUF];
                                snprintf(tok_msg, sizeof(tok_msg), "{\"type\":\"llm_token\",\"text\":\"%s\"}", tok_esc);
                                if (ws_send_text(ws, tok_msg) != 0) break;

                                int space = (int)sizeof(response) - resp_len - 1;
                                int copy = tok_len < space ? tok_len : space;
                                if (copy > 0) {
                                    memcpy(response + resp_len, tok, (size_t)copy);
                                    resp_len += copy;
                                    response[resp_len] = '\0';
                                }
                                eng->llm_consume(eng->llm_engine, tok_len);
                            }
                            polls++;
                        }

                        if (resp_len > 0 && ws_is_connected(ws)) {
                            ws_send_text(ws, "{\"type\":\"speaking\"}");

                            eng->tts_reset(eng->tts_engine);
                            eng->tts_speak(eng->tts_engine, response);
                            int sret = eng->tts_set_text_done(eng->tts_engine);
                            if (sret != 0) {
                                fprintf(stderr, "[http] TTS synthesis failed in WebSocket stream\n");
                                ws_send_text(ws, "{\"type\":\"error\",\"message\":\"TTS synthesis failed\"}");
                            }
                            float audio_buf[8192];
                            int max_chunk = sizeof(audio_buf) / sizeof(float);

                            while (ws_is_connected(ws) && !eng->tts_is_done(eng->tts_engine)) {
                                eng->tts_step(eng->tts_engine);
                                int got = eng->tts_get_audio(eng->tts_engine, audio_buf, max_chunk);
                                if (got > 0) {
                                    if (use_opus && opus_enc) {
                                        int ob = pocket_opus_encode(opus_enc, audio_buf, got,
                                                                    opus_packet, sizeof(opus_packet));
                                        if (ob > 0 && ws_send_binary(ws, opus_packet, ob) != 0)
                                            break;
                                    } else {
                                        if (ws_send_binary(ws, (const uint8_t *)audio_buf, got * 4) != 0)
                                            break;
                                    }
                                }
                            }
                            if (use_opus && opus_enc) {
                                int ob = pocket_opus_flush(opus_enc, opus_packet, sizeof(opus_packet));
                                if (ob > 0)
                                    ws_send_binary(ws, opus_packet, ob);
                            }
                        }

                        ws_send_text(ws, "{\"type\":\"listening\"}");
                    }
                    eng->stt_reset(eng->stt_engine);
                }
            }
        }
    }

    if (opus_enc) pocket_opus_destroy(opus_enc);
    if (opus_dec) pocket_opus_destroy(opus_dec);
    ws_close(ws);
    ws_destroy(ws);
}

static void handle_chat(int fd, HttpRequest *req, HttpApiEngines *eng) {
    if (!req->body || req->body_len == 0) {
        send_json(fd, 400, "{\"error\":\"Body must contain message text\"}");
        return;
    }

    if (eng->llm_send(eng->llm_engine, req->body) != 0) {
        send_json(fd, 500, "{\"error\":\"Failed to send to LLM\"}");
        return;
    }

    char response[16384] = {0};
    int resp_len = 0;
    bool done = false;
    int polls = 0;

    while (!done && polls < 30000) {
        eng->llm_poll(eng->llm_engine, 10);
        int tok_len = 0;
        const char *tok = eng->llm_peek(eng->llm_engine, &tok_len);
        if (tok && tok_len > 0) {
            int space = (int)sizeof(response) - resp_len - 1;
            int copy = tok_len < space ? tok_len : space;
            if (copy > 0) {
                memcpy(response + resp_len, tok, (size_t)copy);
                resp_len += copy;
                response[resp_len] = '\0';
            }
            eng->llm_consume(eng->llm_engine, tok_len);
        }
        done = eng->llm_is_done(eng->llm_engine);
        polls++;
    }

    char json[32768];
    char escaped[16384];
    int ei = 0;
    for (int i = 0; i < resp_len && ei < (int)sizeof(escaped) - 2; i++) {
        if (response[i] == '"' || response[i] == '\\') escaped[ei++] = '\\';
        if (response[i] == '\n') { escaped[ei++] = '\\'; escaped[ei++] = 'n'; continue; }
        escaped[ei++] = response[i];
    }
    escaped[ei] = '\0';
    snprintf(json, sizeof(json), "{\"response\":\"%s\"}", escaped);
    send_json(fd, 200, json);
}

/* ─── Server Thread ───────────────────────────────────────────────────── */

static int check_auth(int fd, HttpApi *api, HttpRequest *req) {
    if (!api->api_key[0]) return 1; /* no key configured = open access */

    const char *auth = req->authorization;
    if (!auth[0]) {
        send_json(fd, 401,
            "{\"error\":\"Missing Authorization header. Use: Bearer <api_key>\"}");
        return 0;
    }

    /* Expect "Bearer <key>" */
    if (strncmp(auth, "Bearer ", 7) != 0) {
        send_json(fd, 401, "{\"error\":\"Invalid Authorization format. Use: Bearer <api_key>\"}");
        return 0;
    }
    const char *token = auth + 7;
    while (*token == ' ') token++;

    if (strcmp(token, api->api_key) != 0) {
        send_json(fd, 403, "{\"error\":\"Invalid API key\"}");
        return 0;
    }

    return 1;
}

static void handle_client(int fd, HttpApi *api) {
    HttpApiEngines *eng = &api->eng;
    HttpRequest req;
    if (parse_request(fd, &req) != 0) {
        send_json(fd, 400, "{\"error\":\"Bad request\"}");
        close(fd);
        return;
    }

    if (strcmp(req.method, "OPTIONS") == 0) {
        send_response(fd, 200, "text/plain", "", 0);
    } else if (strcmp(req.path, "/health") == 0 || strcmp(req.path, "/v1/health") == 0) {
        handle_health(fd);
    } else if (!check_auth(fd, api, &req)) {
        /* Auth failed — response already sent */
    } else if (!rl_allow(&api->rate_limiter)) {
        send_json(fd, 429, "{\"error\":\"Rate limit exceeded. Try again shortly.\"}");
    } else if (strcmp(req.path, "/v1/stream") == 0 && strcmp(req.method, "GET") == 0 &&
               req.ws_upgrade && req.ws_key[0] != '\0') {
        handle_websocket_stream(fd, req.ws_key, eng);
        free(req.body);
        return;
    } else if (strcmp(req.path, "/v1/stream") == 0 && strcmp(req.method, "GET") == 0) {
        send_json(fd, 400, "{\"error\":\"WebSocket upgrade required\"}");
    } else if (strcmp(req.path, "/v1/audio/transcriptions") == 0) {
        handle_stt(fd, &req, eng);
    } else if (strcmp(req.path, "/v1/audio/speech") == 0) {
        handle_tts(fd, &req, api);
    } else if (strcmp(req.path, "/v1/voices") == 0 && strcmp(req.method, "POST") == 0) {
        handle_voice_create(fd, &req, api);
    } else if (strcmp(req.path, "/v1/voices") == 0 && strcmp(req.method, "GET") == 0) {
        handle_voice_list(fd, api);
    } else if (strcmp(req.path, "/v1/chat") == 0) {
        handle_chat(fd, &req, eng);
    } else {
        send_json(fd, 404, "{\"error\":\"Not found\"}");
    }

    free(req.body);
    close(fd);
}

static void *worker_thread(void *arg) {
    HttpApi *api = (HttpApi *)arg;

    while (api->running) {
        int fd = cq_pop(&api->queue, &api->running);
        if (fd < 0) continue;

        struct timeval tv = { .tv_sec = 30, .tv_usec = 0 };
        setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
        setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));

        handle_client(fd, api);
    }

    return NULL;
}

static void *accept_thread(void *arg) {
    HttpApi *api = (HttpApi *)arg;

    while (api->running) {
        struct sockaddr_in client;
        socklen_t clen = sizeof(client);
        int fd = accept(api->server_fd, (struct sockaddr *)&client, &clen);
        if (fd < 0) {
            if (api->running) usleep(10000);
            continue;
        }
        cq_push(&api->queue, fd);
    }

    return NULL;
}

/* ─── Public API ──────────────────────────────────────────────────────── */

HttpApi *http_api_create(int port, HttpApiEngines engines) {
    HttpApi *api = calloc(1, sizeof(HttpApi));
    if (!api) return NULL;
    api->port = port;
    api->eng = engines;
    api->server_fd = -1;
    cq_init(&api->queue);
    pthread_mutex_init(&api->voice_mutex, NULL);
    pthread_mutex_init(&api->tts_mutex, NULL);
    rl_init(&api->rate_limiter, RATE_LIMIT_RPS, RATE_LIMIT_BURST);

    /* Auto-load API key from environment */
    const char *env_key = getenv("SONATA_API_KEY");
    if (env_key && *env_key)
        snprintf(api->api_key, sizeof(api->api_key), "%s", env_key);

    return api;
}

void http_api_set_api_key(HttpApi *api, const char *key) {
    if (!api) return;
    if (key && *key)
        snprintf(api->api_key, sizeof(api->api_key), "%s", key);
    else
        api->api_key[0] = '\0';
}

int http_api_start(HttpApi *api) {
    if (!api) return -1;

    api->server_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (api->server_fd < 0) {
        fprintf(stderr, "[http] Socket failed: %s\n", strerror(errno));
        return -1;
    }

    int opt = 1;
    setsockopt(api->server_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    struct sockaddr_in addr = {
        .sin_family = AF_INET,
        .sin_port = htons((uint16_t)api->port),
        .sin_addr.s_addr = INADDR_ANY,
    };

    if (bind(api->server_fd, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
        fprintf(stderr, "[http] Bind to port %d failed: %s\n",
                api->port, strerror(errno));
        close(api->server_fd);
        return -1;
    }

    if (listen(api->server_fd, BACKLOG) < 0) {
        fprintf(stderr, "[http] Listen failed: %s\n", strerror(errno));
        close(api->server_fd);
        return -1;
    }

    api->running = 1;
    api->worker_count = 0;
    api->accept_thread_started = 0;

    /* Start worker thread pool */
    for (int i = 0; i < THREAD_POOL_SIZE; i++) {
        if (pthread_create(&api->workers[i], NULL, worker_thread, api) != 0) {
            fprintf(stderr, "[http] Worker thread %d create failed\n", i);
        } else {
            api->worker_count++;
        }
    }

    /* Start accept thread */
    if (pthread_create(&api->accept_thread, NULL, accept_thread, api) != 0) {
        fprintf(stderr, "[http] Accept thread create failed\n");
        close(api->server_fd);
        return -1;
    }
    api->accept_thread_started = 1;

    fprintf(stderr, "[http] API server listening on http://0.0.0.0:%d (%d workers)\n",
            api->port, THREAD_POOL_SIZE);
    if (api->api_key[0])
        fprintf(stderr, "[http] API key authentication enabled (SONATA_API_KEY)\n");
    fprintf(stderr, "[http] Endpoints:\n");
    fprintf(stderr, "[http]   GET  /health                   - Health check\n");
    fprintf(stderr, "[http]   GET  /v1/stream                - WebSocket audio streaming\n");
    fprintf(stderr, "[http]   POST /v1/audio/transcriptions  - STT (WAV → JSON)\n");
    fprintf(stderr, "[http]   POST /v1/audio/speech          - TTS (text/JSON → audio)\n");
    fprintf(stderr, "[http]   POST /v1/voices                - Voice clone (WAV → voice_id)\n");
    fprintf(stderr, "[http]   GET  /v1/voices                - List cloned voices\n");
    fprintf(stderr, "[http]   POST /v1/chat                  - Chat (text → JSON)\n");
    return 0;
}

void http_api_stop(HttpApi *api) {
    if (!api) return;
    api->running = 0;
    if (api->server_fd >= 0) {
        shutdown(api->server_fd, SHUT_RDWR);
        close(api->server_fd);
        api->server_fd = -1;
    }
    if (api->accept_thread_started)
        pthread_join(api->accept_thread, NULL);
    api->accept_thread_started = 0;

    /* Wake all workers and join */
    pthread_cond_broadcast(&api->queue.not_empty);
    for (int i = 0; i < api->worker_count; i++)
        pthread_join(api->workers[i], NULL);
    api->worker_count = 0;
}

void http_api_destroy(HttpApi *api) {
    if (!api) return;
    http_api_stop(api);
    cq_destroy(&api->queue);
    pthread_mutex_destroy(&api->voice_mutex);
    pthread_mutex_destroy(&api->tts_mutex);
    rl_destroy(&api->rate_limiter);
    free(api);
}
