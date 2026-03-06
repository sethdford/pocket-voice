/**
 * streaming_llm.c — Unified WebSocket-based streaming LLM client.
 *
 * Supports Gemini Live and OpenAI Realtime APIs for bidirectional audio streaming.
 * Uses libcurl WebSocket API (curl_ws_send/recv, CURL 7.86+).
 */

#define _GNU_SOURCE
#include "streaming_llm.h"
#include "cJSON.h"
#include "neon_audio.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdarg.h>
#include <curl/curl.h>
#include <stdatomic.h>

/* ─── Base64 (self-contained, no external dependency) ───────────────────── */

static const char b64_table[] =
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

static int b64_encode(const unsigned char *in, int n_in, char *out, int max_out) {
    int i = 0, j = 0;
    for (; i + 3 <= n_in && j + 4 <= max_out; i += 3, j += 4) {
        unsigned int v = (in[i] << 16) | (in[i + 1] << 8) | in[i + 2];
        out[j + 0] = b64_table[(v >> 18) & 63];
        out[j + 1] = b64_table[(v >> 12) & 63];
        out[j + 2] = b64_table[(v >> 6) & 63];
        out[j + 3] = b64_table[v & 63];
    }
    if (i < n_in && j + 4 <= max_out) {
        unsigned int v = in[i] << 16;
        if (i + 1 < n_in) v |= in[i + 1] << 8;
        out[j++] = b64_table[(v >> 18) & 63];
        out[j++] = b64_table[(v >> 12) & 63];
        out[j++] = (i + 2 < n_in) ? b64_table[(v >> 6) & 63] : '=';
        out[j++] = '=';
    }
    if (j < max_out) out[j] = '\0';
    return j;
}

static const signed char b64_decode_table[256] = {
    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,62,-1,-1,-1,63,
    52,53,54,55,56,57,58,59,60,61,-1,-1,-1,-1,-1,-1,-1, 0, 1, 2, 3, 4, 5, 6,
     7, 8, 9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,-1,-1,-1,-1,-1,-1,
    26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,
    51,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
};

static int b64_decode(const char *in, int n_in, unsigned char *out, int max_out) {
    int i = 0, j = 0;
    while (i + 4 <= n_in && j + 3 <= max_out) {
        int a = (i + 0 < n_in) ? b64_decode_table[(unsigned char)in[i + 0]] : -1;
        int b = (i + 1 < n_in) ? b64_decode_table[(unsigned char)in[i + 1]] : -1;
        int c = (i + 2 < n_in && in[i + 2] != '=') ? b64_decode_table[(unsigned char)in[i + 2]] : -1;
        int d = (i + 3 < n_in && in[i + 3] != '=') ? b64_decode_table[(unsigned char)in[i + 3]] : -1;
        if (a < 0 || b < 0) break;
        unsigned int v = (a << 18) | (b << 12) | ((c < 0 ? 0 : c) << 6) | (d < 0 ? 0 : d);
        out[j++] = (unsigned char)(v >> 16);
        if (c >= 0) out[j++] = (unsigned char)(v >> 8);
        if (d >= 0) out[j++] = (unsigned char)v;
        i += 4;
    }
    return j;
}

/* ─── Core struct ───────────────────────────────────────────────────────── */

#define TEXT_BUF_SIZE 8192
#define ERROR_MSG_SIZE 512
#define TRANSCRIPT_SIZE 4096
#define AUDIO_RECV_CAP (24000 * 2)  /* 2 seconds at 24kHz */
#define AUDIO_SEND_INTERVAL_MS 100
#define B64_BUF_CAP (AUDIO_RECV_CAP * 4 * 2)  /* generous for base64 expansion */

struct StreamingLLM {
    StreamingLLMConfig cfg;
    StreamingLLMType type;

    CURL *ws_curl;
    CURLM *multi;
    int connected;

    char text_buf[TEXT_BUF_SIZE];
    int text_len;
    int text_read;

    float *audio_recv_buf;
    _Atomic int audio_recv_head;
    _Atomic int audio_recv_tail;
    int audio_recv_cap;

    float *audio_send_buf;
    int audio_send_len;
    int audio_send_cap;
    int audio_send_interval;

    _Atomic int response_done;
    _Atomic int has_error;
    char error_msg[ERROR_MSG_SIZE];
    char transcript[TRANSCRIPT_SIZE];
    int server_vad_enabled;

    char *b64_buf;
    int b64_cap;

    int setup_sent;
    char *tools_json;
};

static void set_error(StreamingLLM *llm, const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(llm->error_msg, sizeof(llm->error_msg), fmt, ap);
    va_end(ap);
    atomic_store_explicit(&llm->has_error, 1, memory_order_release);
}

/* ─── Audio ring buffer (lock-free) ─────────────────────────────────────── */

static int ring_available(const StreamingLLM *llm) {
    int head = atomic_load_explicit(&llm->audio_recv_head, memory_order_acquire);
    int tail = atomic_load_explicit(&llm->audio_recv_tail, memory_order_acquire);
    int avail = head - tail;
    if (avail < 0) avail += llm->audio_recv_cap;
    return avail;
}

static void ring_write(StreamingLLM *llm, const float *samples, int n) {
    int cap = llm->audio_recv_cap;
    int head = atomic_load_explicit(&llm->audio_recv_head, memory_order_relaxed);
    for (int i = 0; i < n; i++) {
        llm->audio_recv_buf[head] = samples[i];
        head = (head + 1) % cap;
    }
    atomic_store_explicit(&llm->audio_recv_head, head, memory_order_release);
}

static int ring_read(StreamingLLM *llm, float *out, int max_n) {
    int tail = atomic_load_explicit(&llm->audio_recv_tail, memory_order_relaxed);
    int head = atomic_load_explicit(&llm->audio_recv_head, memory_order_acquire);
    int cap = llm->audio_recv_cap;
    int avail = head - tail;
    if (avail < 0) avail += cap;
    int to_read = avail < max_n ? avail : max_n;
    for (int i = 0; i < to_read; i++) {
        out[i] = llm->audio_recv_buf[tail];
        tail = (tail + 1) % cap;
    }
    atomic_store_explicit(&llm->audio_recv_tail, tail, memory_order_release);
    return to_read;
}

/* ─── WebSocket send (JSON text) ─────────────────────────────────────────── */

static int ws_send_json(StreamingLLM *llm, const char *json) {
    if (!llm->ws_curl || !llm->connected) return -1;
    size_t len = strlen(json);
    size_t sent = 0;
    CURLcode r = curl_ws_send(llm->ws_curl, json, len, &sent, 0, CURLWS_TEXT);
    if (r != CURLE_OK && r != CURLE_AGAIN) {
        set_error(llm, "curl_ws_send: %s", curl_easy_strerror(r));
        return -1;
    }
    (void)sent;
    return 0;
}

/* ─── Gemini Live protocol ──────────────────────────────────────────────── */

static int gemini_send_setup(StreamingLLM *llm) {
    char url[512];
    snprintf(url, sizeof(url),
             "wss://generativelanguage.googleapis.com/ws/google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent?key=%s",
             llm->cfg.api_key ? llm->cfg.api_key : "");

    curl_easy_setopt(llm->ws_curl, CURLOPT_URL, url);
    curl_easy_setopt(llm->ws_curl, CURLOPT_CONNECT_ONLY, 2L);

    CURLMcode mcode = curl_multi_perform(llm->multi, NULL);
    if (mcode != CURLM_OK) {
        set_error(llm, "Gemini connect: %s", curl_multi_strerror(mcode));
        return -1;
    }

    /* Build setup message */
    cJSON *setup = cJSON_CreateObject();
    cJSON *setup_inner = cJSON_CreateObject();
    cJSON_AddItemToObject(setup, "setup", setup_inner);

    cJSON_AddStringToObject(setup_inner, "model",
                            llm->cfg.model ? llm->cfg.model : "models/gemini-2.0-flash-live");

    cJSON *gen_config = cJSON_CreateObject();
    cJSON *modalities = cJSON_CreateArray();
    cJSON_AddItemToArray(modalities, cJSON_CreateString("AUDIO"));
    cJSON_AddItemToArray(modalities, cJSON_CreateString("TEXT"));
    cJSON_AddItemToObject(gen_config, "responseModalities", modalities);
    cJSON_AddNumberToObject(gen_config, "temperature",
                            llm->cfg.temperature >= 0 ? llm->cfg.temperature : 0.8);
    cJSON_AddItemToObject(setup_inner, "generationConfig", gen_config);

    if (llm->cfg.system_prompt && llm->cfg.system_prompt[0]) {
        cJSON *si = cJSON_CreateObject();
        cJSON *parts = cJSON_CreateArray();
        cJSON *part = cJSON_CreateObject();
        cJSON_AddStringToObject(part, "text", llm->cfg.system_prompt);
        cJSON_AddItemToArray(parts, part);
        cJSON_AddItemToObject(si, "parts", parts);
        cJSON_AddItemToObject(setup_inner, "systemInstruction", si);
    }

    char *json = cJSON_PrintUnformatted(setup);
    cJSON_Delete(setup);
    if (!json) return -1;
    int ret = ws_send_json(llm, json);
    free(json);
    return ret;
}

static void gemini_handle_message(StreamingLLM *llm, const char *data, int len) {
    cJSON *root = cJSON_ParseWithLength(data, (size_t)len);
    if (!root) return;
    cJSON *sc = cJSON_GetObjectItem(root, "serverContent");
    if (!sc) { cJSON_Delete(root); return; }
    cJSON *model_turn = cJSON_GetObjectItem(sc, "modelTurn");
    if (model_turn) {
        cJSON *parts = cJSON_GetObjectItem(model_turn, "parts");
        if (cJSON_IsArray(parts)) {
            int n = cJSON_GetArraySize(parts);
            for (int i = 0; i < n; i++) {
                cJSON *p = cJSON_GetArrayItem(parts, i);
                cJSON *text = cJSON_GetObjectItem(p, "text");
                if (text && cJSON_IsString(text) && text->valuestring) {
                    int remain = TEXT_BUF_SIZE - llm->text_len - 1;
                    if (remain > 0) {
                        int add = (int)strlen(text->valuestring);
                        if (add > remain) add = remain;
                        memcpy(llm->text_buf + llm->text_len, text->valuestring, (size_t)add);
                        llm->text_len += add;
                        llm->text_buf[llm->text_len] = '\0';
                    }
                }
                cJSON *inline_data = cJSON_GetObjectItem(p, "inlineData");
                if (inline_data) {
                    cJSON *b64 = cJSON_GetObjectItem(inline_data, "data");
                    if (b64 && cJSON_IsString(b64) && b64->valuestring) {
                        int dec_len = (int)(strlen(b64->valuestring) * 3 / 4 + 4);
                        if (dec_len > llm->b64_cap) dec_len = llm->b64_cap;
                        int n_bytes = b64_decode(b64->valuestring, (int)strlen(b64->valuestring),
                                                 (unsigned char *)llm->b64_buf, dec_len);
                        int n_samples = n_bytes / (int)sizeof(float);
                        if (n_samples > 0 && n_samples <= AUDIO_RECV_CAP)
                            ring_write(llm, (const float *)llm->b64_buf, n_samples);
                    }
                }
            }
        }
    }
    cJSON *turn_complete = cJSON_GetObjectItem(sc, "turnComplete");
    if (cJSON_IsTrue(turn_complete))
        atomic_store_explicit(&llm->response_done, 1, memory_order_release);
    cJSON_Delete(root);
}

/* ─── OpenAI Realtime protocol ───────────────────────────────────────────── */

static int openai_send_setup(StreamingLLM *llm) {
    char url[256];
    snprintf(url, sizeof(url), "wss://api.openai.com/v1/realtime?model=%s",
             llm->cfg.model ? llm->cfg.model : "gpt-4o-realtime-preview");

    struct curl_slist *headers = NULL;
    if (llm->cfg.api_key) {
        char auth[512];
        snprintf(auth, sizeof(auth), "Authorization: Bearer %s", llm->cfg.api_key);
        headers = curl_slist_append(headers, auth);
    }
    headers = curl_slist_append(headers, "OpenAI-Beta: realtime=v1");
    headers = curl_slist_append(headers, "Content-Type: application/json");

    curl_easy_setopt(llm->ws_curl, CURLOPT_URL, url);
    curl_easy_setopt(llm->ws_curl, CURLOPT_HTTPHEADER, headers);
    curl_easy_setopt(llm->ws_curl, CURLOPT_CONNECT_ONLY, 2L);

    CURLMcode mcode = curl_multi_perform(llm->multi, NULL);
    curl_slist_free_all(headers);
    if (mcode != CURLM_OK) {
        set_error(llm, "OpenAI connect: %s", curl_multi_strerror(mcode));
        return -1;
    }
    llm->connected = 1;

    cJSON *session = cJSON_CreateObject();
    cJSON *mod = cJSON_CreateArray();
    cJSON_AddItemToArray(mod, cJSON_CreateString("text"));
    cJSON_AddItemToArray(mod, cJSON_CreateString("audio"));
    cJSON_AddItemToObject(session, "modalities", mod);
    if (llm->cfg.system_prompt && llm->cfg.system_prompt[0])
        cJSON_AddStringToObject(session, "instructions", llm->cfg.system_prompt);
    cJSON_AddStringToObject(session, "voice", llm->cfg.voice ? llm->cfg.voice : "alloy");
    cJSON_AddStringToObject(session, "input_audio_format", "pcm16");
    cJSON_AddStringToObject(session, "output_audio_format", "pcm16");
    cJSON_AddNumberToObject(session, "input_audio_transcription", llm->server_vad_enabled ? 1 : 0);

    cJSON *msg = cJSON_CreateObject();
    cJSON_AddStringToObject(msg, "type", "session.update");
    cJSON_AddItemToObject(msg, "session", session);

    char *json = cJSON_PrintUnformatted(msg);
    cJSON_Delete(msg);
    if (!json) return -1;
    int ret = ws_send_json(llm, json);
    free(json);
    return ret;
}

static void openai_handle_message(StreamingLLM *llm, const char *data, int len) {
    cJSON *root = cJSON_ParseWithLength(data, (size_t)len);
    if (!root) return;
    cJSON *type_j = cJSON_GetObjectItem(root, "type");
    const char *typ = type_j && cJSON_IsString(type_j) ? type_j->valuestring : "";
    if (strcmp(typ, "response.text.delta") == 0) {
        cJSON *delta = cJSON_GetObjectItem(root, "delta");
        if (delta && cJSON_IsString(delta) && delta->valuestring) {
            int remain = TEXT_BUF_SIZE - llm->text_len - 1;
            if (remain > 0) {
                int add = (int)strlen(delta->valuestring);
                if (add > remain) add = remain;
                memcpy(llm->text_buf + llm->text_len, delta->valuestring, (size_t)add);
                llm->text_len += add;
                llm->text_buf[llm->text_len] = '\0';
            }
        }
    } else if (strcmp(typ, "response.audio.delta") == 0) {
        cJSON *delta = cJSON_GetObjectItem(root, "delta");
        if (delta && cJSON_IsString(delta) && delta->valuestring) {
            int dec_len = (int)(strlen(delta->valuestring) * 3 / 4 + 4);
            if (dec_len > llm->b64_cap) dec_len = llm->b64_cap;
            int n_bytes = b64_decode(delta->valuestring, (int)strlen(delta->valuestring),
                                     (unsigned char *)llm->b64_buf, dec_len);
            int n_samples = n_bytes / 2;  /* PCM16 = 2 bytes per sample */
            if (n_samples > 0 && n_samples <= AUDIO_RECV_CAP) {
                float tmp[AUDIO_RECV_CAP];
                neon_s16_to_f32((const int16_t *)llm->b64_buf, tmp, n_samples);
                ring_write(llm, tmp, n_samples);
            }
        }
    } else if (strcmp(typ, "response.done") == 0) {
        atomic_store_explicit(&llm->response_done, 1, memory_order_release);
    } else if (strcmp(typ, "conversation.item.input_audio_transcription.completed") == 0) {
        cJSON *tr = cJSON_GetObjectItem(root, "transcript");
        if (tr && cJSON_IsString(tr) && tr->valuestring) {
            strncpy(llm->transcript, tr->valuestring, TRANSCRIPT_SIZE - 1);
            llm->transcript[TRANSCRIPT_SIZE - 1] = '\0';
        }
    }
    cJSON_Delete(root);
}

/* ─── Public API ─────────────────────────────────────────────────────────── */

StreamingLLM *streaming_llm_create(const StreamingLLMConfig *cfg) {
    if (!cfg) return NULL;
    StreamingLLM *llm = (StreamingLLM *)calloc(1, sizeof(StreamingLLM));
    if (!llm) return NULL;
    memcpy(&llm->cfg, cfg, sizeof(StreamingLLMConfig));
    llm->type = cfg->type;
    llm->audio_recv_cap = AUDIO_RECV_CAP;
    llm->audio_recv_buf = (float *)calloc((size_t)llm->audio_recv_cap, sizeof(float));
    if (!llm->audio_recv_buf) { free(llm); return NULL; }
    llm->audio_send_cap = llm->cfg.input_sample_rate * AUDIO_SEND_INTERVAL_MS / 1000;
    llm->audio_send_buf = (float *)calloc((size_t)llm->audio_send_cap, sizeof(float));
    if (!llm->audio_send_buf) {
        free(llm->audio_recv_buf);
        free(llm);
        return NULL;
    }
    llm->audio_send_interval = llm->cfg.input_sample_rate * AUDIO_SEND_INTERVAL_MS / 1000;
    if (llm->audio_send_interval <= 0) llm->audio_send_interval = 480;
    llm->b64_cap = B64_BUF_CAP;
    llm->b64_buf = (char *)malloc((size_t)llm->b64_cap);
    if (!llm->b64_buf) {
        free(llm->audio_send_buf);
        free(llm->audio_recv_buf);
        free(llm);
        return NULL;
    }
    return llm;
}

void streaming_llm_destroy(StreamingLLM *llm) {
    if (!llm) return;
    streaming_llm_disconnect(llm);
    free(llm->b64_buf);
    free(llm->audio_send_buf);
    free(llm->audio_recv_buf);
    free(llm->tools_json);
    free(llm);
}

int streaming_llm_connect(StreamingLLM *llm) {
    if (!llm || llm->connected) return 0;
    llm->ws_curl = curl_easy_init();
    if (!llm->ws_curl) {
        set_error(llm, "curl_easy_init failed");
        return -1;
    }
    llm->multi = curl_multi_init();
    if (!llm->multi) {
        curl_easy_cleanup(llm->ws_curl);
        llm->ws_curl = NULL;
        set_error(llm, "curl_multi_init failed");
        return -1;
    }
    curl_multi_add_handle(llm->multi, llm->ws_curl);
    int ret = -1;
    if (llm->type == STREAMING_LLM_GEMINI_LIVE)
        ret = gemini_send_setup(llm);
    else
        ret = openai_send_setup(llm);
    if (ret != 0) {
        streaming_llm_disconnect(llm);
        return -1;
    }
    llm->setup_sent = 1;
    return 0;
}

void streaming_llm_disconnect(StreamingLLM *llm) {
    if (!llm) return;
    if (llm->multi && llm->ws_curl) {
        curl_multi_remove_handle(llm->multi, llm->ws_curl);
    }
    if (llm->multi) { curl_multi_cleanup(llm->multi); llm->multi = NULL; }
    if (llm->ws_curl) { curl_easy_cleanup(llm->ws_curl); llm->ws_curl = NULL; }
    llm->connected = 0;
}

int streaming_llm_is_connected(const StreamingLLM *llm) {
    return llm && llm->connected;
}

int streaming_llm_input_sample_rate(const StreamingLLM *llm) {
    return llm ? llm->cfg.input_sample_rate : 16000;
}

int streaming_llm_output_sample_rate(const StreamingLLM *llm) {
    return llm ? llm->cfg.output_sample_rate : 24000;
}

int streaming_llm_send_text(StreamingLLM *llm, const char *user_text) {
    if (!llm || !user_text || !llm->connected) return -1;
    if (llm->type == STREAMING_LLM_GEMINI_LIVE) {
        cJSON *msg = cJSON_CreateObject();
        cJSON *client = cJSON_CreateObject();
        cJSON *turns = cJSON_CreateArray();
        cJSON *turn = cJSON_CreateObject();
        cJSON_AddStringToObject(turn, "role", "user");
        cJSON *parts = cJSON_CreateArray();
        cJSON *part = cJSON_CreateObject();
        cJSON_AddStringToObject(part, "text", user_text);
        cJSON_AddItemToArray(parts, part);
        cJSON_AddItemToObject(turn, "parts", parts);
        cJSON_AddItemToArray(turns, turn);
        cJSON_AddItemToObject(client, "turns", turns);
        cJSON_AddTrueToObject(client, "turnComplete");
        cJSON_AddItemToObject(msg, "clientContent", client);
        char *json = cJSON_PrintUnformatted(msg);
        cJSON_Delete(msg);
        if (!json) return -1;
        int r = ws_send_json(llm, json);
        free(json);
        return r;
    } else {
        cJSON *msg = cJSON_CreateObject();
        cJSON_AddStringToObject(msg, "type", "conversation.item.create");
        cJSON *item = cJSON_CreateObject();
        cJSON_AddStringToObject(item, "type", "message");
        cJSON_AddStringToObject(item, "role", "user");
        cJSON *content = cJSON_CreateArray();
        cJSON *ct = cJSON_CreateObject();
        cJSON_AddStringToObject(ct, "type", "input_text");
        cJSON_AddStringToObject(ct, "text", user_text);
        cJSON_AddItemToArray(content, ct);
        cJSON_AddItemToObject(item, "content", content);
        cJSON_AddItemToObject(msg, "item", item);
        char *json = cJSON_PrintUnformatted(msg);
        cJSON_Delete(msg);
        if (!json) return -1;
        int r = ws_send_json(llm, json);
        free(json);
        if (r == 0) {
            cJSON *rc = cJSON_CreateObject();
            cJSON_AddStringToObject(rc, "type", "response.create");
            char *rj = cJSON_PrintUnformatted(rc);
            cJSON_Delete(rc);
            if (rj) { ws_send_json(llm, rj); free(rj); }
        }
        return r;
    }
}

const char *streaming_llm_peek_text(StreamingLLM *llm, int *out_len) {
    if (!llm) { if (out_len) *out_len = 0; return ""; }
    int avail = llm->text_len - llm->text_read;
    if (avail < 0) avail = 0;
    if (out_len) *out_len = avail;
    return llm->text_buf + llm->text_read;
}

void streaming_llm_consume_text(StreamingLLM *llm, int count) {
    if (!llm || count <= 0) return;
    llm->text_read += count;
    if (llm->text_read >= llm->text_len) {
        llm->text_read = 0;
        llm->text_len = 0;
        llm->text_buf[0] = '\0';
    } else {
        memmove(llm->text_buf, llm->text_buf + llm->text_read,
                (size_t)(llm->text_len - llm->text_read) + 1);
        llm->text_len -= llm->text_read;
        llm->text_read = 0;
    }
}

int streaming_llm_send_audio(StreamingLLM *llm, const float *pcm, int n_samples) {
    if (!llm || !pcm || n_samples <= 0 || !llm->connected) return -1;
    int cap = llm->audio_send_cap;
    int idx = llm->audio_send_len;
    for (int i = 0; i < n_samples; i++) {
        if (idx >= cap) {
            /* Flush buffer */
            if (llm->type == STREAMING_LLM_GEMINI_LIVE) {
                int enc = b64_encode((const unsigned char *)llm->audio_send_buf,
                                    idx * (int)sizeof(float), llm->b64_buf, llm->b64_cap);
                if (enc <= 0) continue;
                cJSON *msg = cJSON_CreateObject();
                cJSON *ri = cJSON_CreateObject();
                cJSON *chunks = cJSON_CreateArray();
                cJSON *chunk = cJSON_CreateObject();
                char mime[64];
                snprintf(mime, sizeof(mime), "audio/pcm;rate=%d", llm->cfg.input_sample_rate);
                cJSON_AddStringToObject(chunk, "mimeType", mime);
                cJSON_AddStringToObject(chunk, "data", llm->b64_buf);
                cJSON_AddItemToArray(chunks, chunk);
                cJSON_AddItemToObject(ri, "mediaChunks", chunks);
                cJSON_AddItemToObject(msg, "realtimeInput", ri);
                char *json = cJSON_PrintUnformatted(msg);
                cJSON_Delete(msg);
                if (json) { ws_send_json(llm, json); free(json); }
            } else {
                int16_t s16[8192];
                int to_enc = idx < 8192 ? idx : 8192;
                neon_f32_to_s16(llm->audio_send_buf, s16, to_enc);
                int enc = b64_encode((const unsigned char *)s16, to_enc * 2, llm->b64_buf, llm->b64_cap);
                if (enc <= 0) { idx = 0; continue; }
                cJSON *msg = cJSON_CreateObject();
                cJSON_AddStringToObject(msg, "type", "input_audio_buffer.append");
                cJSON_AddStringToObject(msg, "audio", llm->b64_buf);
                char *json = cJSON_PrintUnformatted(msg);
                cJSON_Delete(msg);
                if (json) { ws_send_json(llm, json); free(json); }
            }
            idx = 0;
        }
        llm->audio_send_buf[idx++] = pcm[i];
    }
    llm->audio_send_len = idx;
    return 0;
}

int streaming_llm_recv_audio(StreamingLLM *llm, float *out_pcm, int max_samples) {
    if (!llm || !out_pcm || max_samples <= 0) return 0;
    return ring_read(llm, out_pcm, max_samples);
}

int streaming_llm_audio_available(const StreamingLLM *llm) {
    return llm ? ring_available(llm) : 0;
}

int streaming_llm_poll(StreamingLLM *llm, int timeout_ms) {
    if (!llm || !llm->ws_curl) return 0;
    char buf[65536];
    size_t rlen;
    const struct curl_ws_frame *meta;
    CURLcode r = curl_ws_recv(llm->ws_curl, buf, sizeof(buf), &rlen, &meta);
    if (r == CURLE_OK && rlen > 0 && meta && (meta->flags & CURLWS_TEXT)) {
        if (llm->type == STREAMING_LLM_GEMINI_LIVE)
            gemini_handle_message(llm, buf, (int)rlen);
        else
            openai_handle_message(llm, buf, (int)rlen);
        return 1;
    }
    if (r == CURLE_AGAIN) {
        if (timeout_ms > 0) {
            curl_multi_wait(llm->multi, NULL, 0, timeout_ms, NULL);
            curl_multi_perform(llm->multi, NULL);
        }
    } else if (r != CURLE_OK && r != CURLE_AGAIN) {
        set_error(llm, "curl_ws_recv: %s", curl_easy_strerror(r));
    }
    return 0;
}

bool streaming_llm_is_done(const StreamingLLM *llm) {
    return llm && atomic_load_explicit(&llm->response_done, memory_order_acquire);
}

void streaming_llm_cancel(StreamingLLM *llm) {
    if (!llm) return;
    if (llm->type == STREAMING_LLM_OPENAI_REALTIME && llm->connected) {
        cJSON *msg = cJSON_CreateObject();
        cJSON_AddStringToObject(msg, "type", "response.cancel");
        char *json = cJSON_PrintUnformatted(msg);
        cJSON_Delete(msg);
        if (json) { ws_send_json(llm, json); free(json); }
    }
    atomic_store_explicit(&llm->response_done, 1, memory_order_release);
}

void streaming_llm_end_turn(StreamingLLM *llm) {
    if (!llm || !llm->connected) return;
    if (llm->type == STREAMING_LLM_OPENAI_REALTIME) {
        cJSON *c1 = cJSON_CreateObject();
        cJSON_AddStringToObject(c1, "type", "input_audio_buffer.commit");
        char *j1 = cJSON_PrintUnformatted(c1);
        cJSON_Delete(c1);
        if (j1) { ws_send_json(llm, j1); free(j1); }
        cJSON *c2 = cJSON_CreateObject();
        cJSON_AddStringToObject(c2, "type", "response.create");
        char *j2 = cJSON_PrintUnformatted(c2);
        cJSON_Delete(c2);
        if (j2) { ws_send_json(llm, j2); free(j2); }
    }
}

void streaming_llm_commit_turn(StreamingLLM *llm, const char *user_text) {
    (void)user_text;
    streaming_llm_end_turn(llm);
}

bool streaming_llm_has_error(const StreamingLLM *llm) {
    return llm && atomic_load_explicit(&llm->has_error, memory_order_acquire);
}

const char *streaming_llm_error_message(const StreamingLLM *llm) {
    return llm ? llm->error_msg : "";
}

const char *streaming_llm_get_transcript(const StreamingLLM *llm) {
    return llm ? llm->transcript : "";
}

int streaming_llm_set_tools(StreamingLLM *llm, const char *tools_json) {
    if (!llm) return -1;
    free(llm->tools_json);
    llm->tools_json = tools_json ? strdup(tools_json) : NULL;
    return 0;
}

void streaming_llm_set_server_vad(StreamingLLM *llm, bool enabled) {
    if (llm) llm->server_vad_enabled = enabled ? 1 : 0;
}
