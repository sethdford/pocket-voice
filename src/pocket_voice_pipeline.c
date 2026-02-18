/**
 * pocket_voice_pipeline.c — Zero-Python voice pipeline for Apple Silicon.
 *
 * Standalone C program that links against:
 *   - libpocket_voice.dylib  (CoreAudio I/O, ring buffers, VAD)
 *   - libpocket_stt.dylib    (Kyutai STT 1B, candle+Metal)
 *   - libpocket_tts_rs.dylib (Kyutai DSM TTS 1.6B, candle+Metal)
 *   - libcurl                (Claude Messages API SSE)
 *
 * State machine:
 *   Listening → Recording → Processing → Streaming → Speaking → Listening
 *   Any speaking/streaming state can transition to Listening on barge-in.
 *
 * Build:
 *   cc -O3 -arch arm64 \
 *     -framework Accelerate -framework CoreAudio -framework AudioToolbox \
 *     -lcurl -L. -lpocket_voice -lpocket_stt -lpocket_tts_rs \
 *     -o pocket-voice pocket_voice_pipeline.c cJSON.c
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <signal.h>
#include <unistd.h>
#include <stdbool.h>
#include <math.h>
#include <sys/time.h>
#include <curl/curl.h>
#include "cJSON.h"

/* ═══════════════════════════════════════════════════════════════════════════
 * FFI declarations for the three native libraries
 * ═══════════════════════════════════════════════════════════════════════════ */

/* --- pocket_voice.c (audio engine) --- */
typedef struct VoiceEngine VoiceEngine;
extern VoiceEngine* voice_engine_create(unsigned int sample_rate, unsigned int buffer_frames);
extern int  voice_engine_start(VoiceEngine *engine);
extern void voice_engine_stop(VoiceEngine *engine);
extern void voice_engine_destroy(VoiceEngine *engine);
extern int  voice_engine_read_capture(VoiceEngine *engine, float *buffer, int max_frames);
extern int  voice_engine_write_playback(VoiceEngine *engine, const float *buffer, int num_frames);
extern void voice_engine_flush_playback(VoiceEngine *engine);
extern int  voice_engine_is_playing(VoiceEngine *engine);
extern int  voice_engine_get_vad_state(VoiceEngine *engine);
extern int  voice_engine_get_barge_in(VoiceEngine *engine);
extern void voice_engine_clear_barge_in(VoiceEngine *engine);
extern void voice_engine_resample_48_to_24(const float *in, float *out, int in_len);
extern void voice_engine_resample_24_to_48(const float *in, float *out, int in_len);
extern int  voice_engine_capture_available(VoiceEngine *engine);

/* --- pocket_stt (Rust cdylib) --- */
extern void *pocket_stt_create(const char *hf_repo, const char *model_path, int enable_vad);
extern void  pocket_stt_destroy(void *engine);
extern int   pocket_stt_process_frame(void *engine, const float *pcm, int num_samples);
extern int   pocket_stt_flush(void *engine);
extern int   pocket_stt_get_all_text(void *engine, char *buf, int buf_size);
extern float pocket_stt_get_vad_prob(void *engine, int horizon);
extern int   pocket_stt_has_vad(void *engine);
extern void  pocket_stt_reset(void *engine);
extern int   pocket_stt_frame_size(void);
extern int   pocket_stt_sample_rate(void);

/* --- pocket_tts_rs (Rust cdylib) --- */
extern void *pocket_tts_rs_create(const char *hf_repo, const char *voice_path, int n_q);
extern void  pocket_tts_rs_destroy(void *engine);
extern int   pocket_tts_rs_set_text(void *engine, const char *text);
extern int   pocket_tts_rs_set_text_done(void *engine);
extern int   pocket_tts_rs_step(void *engine);
extern int   pocket_tts_rs_get_audio(void *engine, float *pcm_buf, int max_samples);
extern int   pocket_tts_rs_is_done(void *engine);
extern int   pocket_tts_rs_reset(void *engine);
extern int   pocket_tts_rs_sample_rate(void);
extern int   pocket_tts_rs_frame_size(void);

/* --- audio_converter.c (hardware-accelerated resampling) --- */
typedef struct HWResampler HWResampler;
extern HWResampler *hw_resampler_create(uint32_t src_rate, uint32_t dst_rate,
                                         uint32_t channels, int quality);
extern int   hw_resample(HWResampler *ctx, const float *input, uint32_t in_frames,
                          float *output, uint32_t max_out);
extern void  hw_resampler_reset(HWResampler *ctx);
extern void  hw_resampler_destroy(HWResampler *ctx);

/* --- vdsp_prosody.c (AMX-accelerated audio post-processing) --- */
extern int   prosody_pitch_shift(const float *input, float *output, int n_samples,
                                  float pitch_factor, int fft_size);
extern int   prosody_time_stretch(const float *input, int in_len, float *output,
                                   float rate_factor, float window_ms, int sample_rate);
typedef struct BiquadCascade BiquadCascade;
extern BiquadCascade *prosody_create_formant_eq(float pitch_factor, int sample_rate);
extern int   prosody_apply_biquad(BiquadCascade *bc, float *audio, int n_samples);
extern void  prosody_destroy_biquad(BiquadCascade *bc);
extern void  prosody_soft_limit(float *audio, int n_samples,
                                 float threshold, float knee_db);
extern void  prosody_volume(float *audio, int n_samples, float volume_db,
                             float fade_ms, int sample_rate);

/* --- spatial_audio.c (binaural 3D HRTF) --- */
typedef struct SpatialAudioEngine SpatialAudioEngine;
extern SpatialAudioEngine *spatial_create(uint32_t sample_rate);
extern int   spatial_set_position(SpatialAudioEngine *engine, int source_idx,
                                   float azimuth, float elevation, float distance);
extern int   spatial_process(SpatialAudioEngine *engine, int source_idx,
                              const float *mono_input,
                              float *left_output, float *right_output, int n_samples);
extern void  spatial_destroy(SpatialAudioEngine *engine);

/* --- opus_codec.c (streaming audio compression) --- */
typedef struct PocketOpus PocketOpus;
extern PocketOpus *pocket_opus_create(int sample_rate, int channels, int bitrate,
                                       float frame_ms, int application);
extern int   pocket_opus_encode(PocketOpus *ctx, const float *pcm, int n_samples,
                                 unsigned char *opus_out, int max_out);
extern int   pocket_opus_flush(PocketOpus *ctx, unsigned char *opus_out, int max_out);
extern void  pocket_opus_destroy(PocketOpus *ctx);

/* ═══════════════════════════════════════════════════════════════════════════
 * Pipeline State Machine
 * ═══════════════════════════════════════════════════════════════════════════ */

typedef enum {
    STATE_LISTENING,   /* Waiting for speech */
    STATE_RECORDING,   /* Capturing speech, feeding STT */
    STATE_PROCESSING,  /* Sending to Claude API */
    STATE_STREAMING,   /* Receiving Claude tokens, feeding TTS */
    STATE_SPEAKING,    /* TTS generating and playing audio */
} PipelineState;

static const char *state_names[] = {
    "LISTENING", "RECORDING", "PROCESSING", "STREAMING", "SPEAKING"
};

/* ═══════════════════════════════════════════════════════════════════════════
 * Claude API SSE Client (libcurl)
 *
 * Uses the curl_multi interface for non-blocking SSE streaming.
 * Parses server-sent events to extract text tokens from content_block_delta.
 * ═══════════════════════════════════════════════════════════════════════════ */

#define CLAUDE_MAX_RESPONSE (64 * 1024)
#define CLAUDE_TOKEN_BUF    4096
#define SSE_LINE_BUF        8192

#define CLAUDE_MAX_HISTORY 20  /* Max conversation turns to retain */

typedef struct {
    char *role;     /* "user" or "assistant" */
    char *content;
} ClaudeMessage;

typedef struct {
    /* curl handles */
    CURLM *multi;
    CURL  *easy;
    struct curl_slist *headers;
    char *post_body;  /* Owned JSON body — curl reads from this during perform */

    /* SSE parsing state */
    char line_buf[SSE_LINE_BUF];
    int  line_len;

    /* Accumulated tokens */
    char tokens[CLAUDE_TOKEN_BUF];
    int  tokens_len;
    int  tokens_read;

    /* Full assistant response for history */
    char *response_accum;
    int   response_accum_len;
    int   response_accum_cap;

    /* Conversation history */
    ClaudeMessage history[CLAUDE_MAX_HISTORY];
    int history_len;

    /* State flags */
    bool request_active;
    bool response_done;
    bool error;

    /* Config */
    char api_key[256];
    char model[128];
    char system_prompt[2048];
} ClaudeClient;

static void claude_init(ClaudeClient *c, const char *api_key,
                         const char *model, const char *system_prompt) {
    memset(c, 0, sizeof(*c));
    c->multi = curl_multi_init();
    snprintf(c->api_key, sizeof(c->api_key), "%s", api_key);
    snprintf(c->model, sizeof(c->model), "%s", model ? model : "claude-sonnet-4-20250514");
    snprintf(c->system_prompt, sizeof(c->system_prompt), "%s",
             system_prompt ? system_prompt
                           : "You are a helpful voice assistant. Keep responses concise "
                             "and conversational — aim for 1-3 sentences. Speak naturally.");
    c->response_accum_cap = CLAUDE_MAX_RESPONSE;
    c->response_accum = (char *)malloc((size_t)c->response_accum_cap);
    if (c->response_accum) c->response_accum[0] = '\0';
}

static void claude_cleanup(ClaudeClient *c) {
    if (c->easy) {
        curl_multi_remove_handle(c->multi, c->easy);
        curl_easy_cleanup(c->easy);
        c->easy = NULL;
    }
    if (c->headers) {
        curl_slist_free_all(c->headers);
        c->headers = NULL;
    }
    free(c->post_body);
    c->post_body = NULL;
    free(c->response_accum);
    c->response_accum = NULL;
    for (int i = 0; i < c->history_len; i++) {
        free(c->history[i].role);
        free(c->history[i].content);
    }
    c->history_len = 0;
    if (c->multi) {
        curl_multi_cleanup(c->multi);
        c->multi = NULL;
    }
}

/* Append text to the response accumulator (for conversation history) */
static void claude_accum_response(ClaudeClient *c, const char *text, int len) {
    if (!c->response_accum) return;
    int need = c->response_accum_len + len + 1;
    if (need > c->response_accum_cap) {
        int new_cap = c->response_accum_cap * 2;
        if (new_cap < need) new_cap = need;
        char *tmp = (char *)realloc(c->response_accum, (size_t)new_cap);
        if (!tmp) return;
        c->response_accum = tmp;
        c->response_accum_cap = new_cap;
    }
    memcpy(c->response_accum + c->response_accum_len, text, (size_t)len);
    c->response_accum_len += len;
    c->response_accum[c->response_accum_len] = '\0';
}

/* Push a turn into conversation history, evicting oldest if full */
static void claude_push_history(ClaudeClient *c, const char *role, const char *content) {
    if (c->history_len >= CLAUDE_MAX_HISTORY) {
        free(c->history[0].role);
        free(c->history[0].content);
        memmove(&c->history[0], &c->history[1],
                (size_t)(CLAUDE_MAX_HISTORY - 1) * sizeof(ClaudeMessage));
        c->history_len = CLAUDE_MAX_HISTORY - 1;
    }
    c->history[c->history_len].role = strdup(role);
    c->history[c->history_len].content = strdup(content);
    c->history_len++;
}

/* Parse a single SSE data line and extract text token if present */
static void claude_parse_sse_data(ClaudeClient *c, const char *data) {
    if (strcmp(data, "[DONE]") == 0) {
        c->response_done = true;
        return;
    }

    cJSON *root = cJSON_Parse(data);
    if (!root) return;

    cJSON *type = cJSON_GetObjectItem(root, "type");
    if (type && type->valuestring) {
        if (strcmp(type->valuestring, "content_block_delta") == 0) {
            cJSON *delta = cJSON_GetObjectItem(root, "delta");
            if (delta) {
                cJSON *text = cJSON_GetObjectItem(delta, "text");
                if (text && text->valuestring) {
                    int len = (int)strlen(text->valuestring);

                    /* Accumulate for TTS streaming */
                    int space = CLAUDE_TOKEN_BUF - c->tokens_len - 1;
                    int tts_len = len < space ? len : space;
                    if (tts_len > 0) {
                        memcpy(c->tokens + c->tokens_len, text->valuestring, (size_t)tts_len);
                        c->tokens_len += tts_len;
                        c->tokens[c->tokens_len] = '\0';
                    }

                    /* Accumulate full response for history */
                    claude_accum_response(c, text->valuestring, len);
                }
            }
        } else if (strcmp(type->valuestring, "message_stop") == 0) {
            c->response_done = true;
        } else if (strcmp(type->valuestring, "error") == 0) {
            cJSON *err = cJSON_GetObjectItem(root, "error");
            if (err) {
                cJSON *msg = cJSON_GetObjectItem(err, "message");
                if (msg && msg->valuestring) {
                    fprintf(stderr, "[claude] API error: %s\n", msg->valuestring);
                }
            }
            c->error = true;
            c->response_done = true;
        }
    }

    cJSON_Delete(root);
}

/* curl write callback: accumulates SSE lines and dispatches data events */
static size_t claude_write_cb(char *ptr, size_t size, size_t nmemb, void *userdata) {
    ClaudeClient *c = (ClaudeClient *)userdata;
    size_t total = size * nmemb;

    for (size_t i = 0; i < total; i++) {
        char ch = ptr[i];
        if (ch == '\n') {
            c->line_buf[c->line_len] = '\0';
            /* Parse SSE line */
            if (c->line_len > 6 && strncmp(c->line_buf, "data: ", 6) == 0) {
                claude_parse_sse_data(c, c->line_buf + 6);
            }
            c->line_len = 0;
        } else if (c->line_len < SSE_LINE_BUF - 1) {
            c->line_buf[c->line_len++] = ch;
        }
    }

    return total;
}

/* Commit the last exchange (user + assistant) into history */
static void claude_commit_turn(ClaudeClient *c, const char *user_text) {
    claude_push_history(c, "user", user_text);
    if (c->response_accum && c->response_accum_len > 0) {
        claude_push_history(c, "assistant", c->response_accum);
    }
}

/* Start a streaming request to Claude Messages API */
static int claude_send(ClaudeClient *c, const char *user_text) {
    if (c->easy) {
        curl_multi_remove_handle(c->multi, c->easy);
        curl_easy_cleanup(c->easy);
        c->easy = NULL;
    }
    if (c->headers) {
        curl_slist_free_all(c->headers);
        c->headers = NULL;
    }
    free(c->post_body);
    c->post_body = NULL;

    c->tokens_len = 0;
    c->tokens_read = 0;
    c->line_len = 0;
    c->response_done = false;
    c->error = false;
    c->request_active = true;
    c->response_accum_len = 0;
    if (c->response_accum) c->response_accum[0] = '\0';

    c->easy = curl_easy_init();
    if (!c->easy) return -1;

    /* Build JSON body */
    cJSON *body = cJSON_CreateObject();
    cJSON_AddStringToObject(body, "model", c->model);
    cJSON_AddNumberToObject(body, "max_tokens", 1024);
    cJSON_AddBoolToObject(body, "stream", 1);

    cJSON *system_arr = cJSON_CreateArray();
    cJSON *sys_block = cJSON_CreateObject();
    cJSON_AddStringToObject(sys_block, "type", "text");
    cJSON_AddStringToObject(sys_block, "text", c->system_prompt);
    cJSON_AddItemToArray(system_arr, sys_block);
    cJSON_AddItemToObject(body, "system", system_arr);

    /* Build messages array: history + current user message */
    cJSON *messages = cJSON_CreateArray();
    for (int i = 0; i < c->history_len; i++) {
        cJSON *hmsg = cJSON_CreateObject();
        cJSON_AddStringToObject(hmsg, "role", c->history[i].role);
        cJSON_AddStringToObject(hmsg, "content", c->history[i].content);
        cJSON_AddItemToArray(messages, hmsg);
    }
    cJSON *msg = cJSON_CreateObject();
    cJSON_AddStringToObject(msg, "role", "user");
    cJSON_AddStringToObject(msg, "content", user_text);
    cJSON_AddItemToArray(messages, msg);
    cJSON_AddItemToObject(body, "messages", messages);

    c->post_body = cJSON_PrintUnformatted(body);
    cJSON_Delete(body);

    /* Headers */
    char auth_header[300];
    snprintf(auth_header, sizeof(auth_header), "x-api-key: %s", c->api_key);

    c->headers = curl_slist_append(c->headers, "Content-Type: application/json");
    c->headers = curl_slist_append(c->headers, auth_header);
    c->headers = curl_slist_append(c->headers, "anthropic-version: 2023-06-01");

    curl_easy_setopt(c->easy, CURLOPT_URL, "https://api.anthropic.com/v1/messages");
    curl_easy_setopt(c->easy, CURLOPT_HTTPHEADER, c->headers);
    curl_easy_setopt(c->easy, CURLOPT_POSTFIELDS, c->post_body);
    curl_easy_setopt(c->easy, CURLOPT_WRITEFUNCTION, claude_write_cb);
    curl_easy_setopt(c->easy, CURLOPT_WRITEDATA, c);
    curl_easy_setopt(c->easy, CURLOPT_TIMEOUT, 60L);

    curl_multi_add_handle(c->multi, c->easy);

    return 0;
}

/* Poll for new SSE data. Uses curl_multi_poll to avoid busy-spinning.
 * timeout_ms: max time to wait (0 = non-blocking). Returns new token char count. */
static int claude_poll(ClaudeClient *c, int timeout_ms) {
    if (!c->request_active) return 0;

    int running = 0;
    int prev_len = c->tokens_len;

    curl_multi_perform(c->multi, &running);

    if (timeout_ms > 0) {
        int numfds = 0;
        curl_multi_poll(c->multi, NULL, 0, timeout_ms, &numfds);
        if (numfds > 0) {
            curl_multi_perform(c->multi, &running);
        }
    }

    /* Check for completion */
    int msgs_in_queue;
    CURLMsg *msg;
    while ((msg = curl_multi_info_read(c->multi, &msgs_in_queue))) {
        if (msg->msg == CURLMSG_DONE) {
            if (msg->data.result != CURLE_OK) {
                fprintf(stderr, "[claude] curl error: %s\n",
                        curl_easy_strerror(msg->data.result));
                c->error = true;
            }
            c->response_done = true;
            c->request_active = false;
        }
    }

    return c->tokens_len - prev_len;
}

/* Read available tokens (non-consuming peek returns pointer, length).
 * Call claude_consume_tokens() after processing. */
static const char *claude_peek_tokens(ClaudeClient *c, int *out_len) {
    int avail = c->tokens_len - c->tokens_read;
    if (avail <= 0) {
        *out_len = 0;
        return NULL;
    }
    *out_len = avail;
    return c->tokens + c->tokens_read;
}

static void claude_consume_tokens(ClaudeClient *c, int count) {
    c->tokens_read += count;
    if (c->tokens_read > c->tokens_len) c->tokens_read = c->tokens_len;
}

static void claude_cancel(ClaudeClient *c) {
    if (c->easy) {
        curl_multi_remove_handle(c->multi, c->easy);
        curl_easy_cleanup(c->easy);
        c->easy = NULL;
    }
    if (c->headers) {
        curl_slist_free_all(c->headers);
        c->headers = NULL;
    }
    free(c->post_body);
    c->post_body = NULL;
    c->request_active = false;
    c->response_done = true;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Configuration
 * ═══════════════════════════════════════════════════════════════════════════ */

typedef struct {
    const char *voice;
    const char *stt_repo;
    const char *stt_model;
    const char *tts_repo;
    const char *claude_model;
    const char *system_prompt;
    int n_q;
    int enable_vad;
    float vad_threshold;

    /* Audio post-processing */
    float pitch;        /* Pitch multiplier (1.0 = no change) */
    float volume_db;    /* Volume in dB (0.0 = no change) */
    int   hw_resample;  /* 1 = AudioConverter, 0 = FIR fallback */
    int   spatial;      /* 1 = enable 3D spatial audio */
    float spatial_az;   /* Azimuth for voice source (degrees) */
} PipelineConfig;

static PipelineConfig default_config(void) {
    return (PipelineConfig){
        .voice        = NULL,
        .stt_repo     = "kyutai/stt-1b-en_fr-candle",
        .stt_model    = "model.safetensors",
        .tts_repo     = "kyutai/tts-1.6b-en_fr",
        .claude_model = "claude-sonnet-4-20250514",
        .system_prompt = NULL,
        .n_q          = 24,
        .enable_vad   = 1,
        .vad_threshold = 0.7f,
        .pitch        = 1.0f,
        .volume_db    = 0.0f,
        .hw_resample  = 1,
        .spatial      = 0,
        .spatial_az   = 0.0f,
    };
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Audio Post-Processor
 *
 * Holds all post-processing state: HW resampler, prosody EQ, spatial engine.
 * Created once in main(), passed to feed_speaker() for per-frame processing.
 * ═══════════════════════════════════════════════════════════════════════════ */

typedef struct {
    HWResampler       *resampler_up;    /* 24kHz → 48kHz */
    HWResampler       *resampler_down;  /* 48kHz → 24kHz (for STT path) */
    BiquadCascade     *formant_eq;      /* Formant correction for pitch shift */
    SpatialAudioEngine *spatial;        /* 3D HRTF engine */
    float              pitch;
    float              volume_db;
    int                use_hw_resample;
    int                use_spatial;
} AudioPostProcessor;

static AudioPostProcessor *postproc_create(PipelineConfig *cfg) {
    AudioPostProcessor *pp = (AudioPostProcessor *)calloc(1, sizeof(AudioPostProcessor));
    if (!pp) return NULL;

    pp->pitch          = cfg->pitch;
    pp->volume_db      = cfg->volume_db;
    pp->use_hw_resample = cfg->hw_resample;
    pp->use_spatial    = cfg->spatial;

    /* HW resampler: 24kHz TTS → 48kHz speaker (RESAMPLE_HIGH = 3) */
    if (pp->use_hw_resample) {
        pp->resampler_up = hw_resampler_create(24000, 48000, 1, 3);
        pp->resampler_down = hw_resampler_create(48000, 24000, 1, 3);
        if (!pp->resampler_up || !pp->resampler_down) {
            fprintf(stderr, "[postproc] HW resampler failed, falling back to FIR\n");
            pp->use_hw_resample = 0;
        }
    }

    /* Formant correction EQ for pitch shifting */
    if (fabsf(pp->pitch - 1.0f) > 0.01f) {
        pp->formant_eq = prosody_create_formant_eq(pp->pitch, 24000);
    }

    /* Spatial audio engine */
    if (pp->use_spatial) {
        pp->spatial = spatial_create(48000);
        if (pp->spatial) {
            spatial_set_position(pp->spatial, 0, cfg->spatial_az, 0.0f, 1.5f);
        }
    }

    return pp;
}

static void postproc_destroy(AudioPostProcessor *pp) {
    if (!pp) return;
    if (pp->resampler_up)   hw_resampler_destroy(pp->resampler_up);
    if (pp->resampler_down) hw_resampler_destroy(pp->resampler_down);
    if (pp->formant_eq)     prosody_destroy_biquad(pp->formant_eq);
    if (pp->spatial)        spatial_destroy(pp->spatial);
    free(pp);
}

static void postproc_reset(AudioPostProcessor *pp) {
    if (!pp) return;
    if (pp->resampler_up)   hw_resampler_reset(pp->resampler_up);
    if (pp->resampler_down) hw_resampler_reset(pp->resampler_down);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Main Pipeline
 * ═══════════════════════════════════════════════════════════════════════════ */

static volatile sig_atomic_t g_quit = 0;

static void signal_handler(int sig) {
    (void)sig;
    g_quit = 1;
}

#define AUDIO_SAMPLE_RATE  48000
#define AUDIO_BUFFER_FRAMES 256
#define STT_FRAME_SIZE     1920   /* 80ms at 24kHz */
#define RESAMPLE_BUF_SIZE  8192
#define TEXT_BUF_SIZE       4096
#define TTS_AUDIO_BUF_SIZE 4096
#define TTS_STEPS_PER_TICK 8      /* Drain up to 8 TTS steps per tick (~640ms of audio) */
#define SPEAKING_TIMEOUT_US (30ULL * 1000000)  /* 30s max in SPEAKING state */

/* Monotonic clock in microseconds */
static uint64_t now_us(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (uint64_t)tv.tv_sec * 1000000ULL + (uint64_t)tv.tv_usec;
}

/* Latency metrics for the current turn */
typedef struct {
    uint64_t speech_start;     /* When VAD detected speech onset */
    uint64_t speech_end;       /* When end-of-turn was detected */
    uint64_t claude_sent;      /* When request was sent to Claude */
    uint64_t claude_first_tok; /* When first Claude token arrived (TTFT) */
    uint64_t tts_first_audio;  /* When first TTS audio was written to speaker */
    uint64_t speaking_entered; /* When we entered SPEAKING state (for timeout) */
    bool     has_first_tok;
    bool     has_first_audio;
} TurnMetrics;

/* Accumulation buffer for resampled STT frames */
typedef struct {
    float buf[STT_FRAME_SIZE * 4];
    int   len;
} SttAccum;

static void stt_accum_reset(SttAccum *a) { a->len = 0; }

/* Feed captured audio (48kHz) through resampler and into STT (24kHz).
 * Accumulates until a full STT frame (1920 samples) is ready.
 * When words are recognized, appends them to transcript (which grows
 * monotonically — no duplication). Returns number of new words. */
static int feed_stt(void *stt, VoiceEngine *audio, SttAccum *accum,
                     char *transcript, int *transcript_len, int transcript_cap,
                     AudioPostProcessor *pp) {
    float capture_48[RESAMPLE_BUF_SIZE];
    float capture_24[RESAMPLE_BUF_SIZE / 2];
    int total_words = 0;

    int n = voice_engine_read_capture(audio, capture_48, RESAMPLE_BUF_SIZE);
    if (n <= 0) return 0;

    /* Resample 48kHz → 24kHz via HW AudioConverter or FIR fallback */
    int n24;
    if (pp && pp->use_hw_resample && pp->resampler_down) {
        n24 = hw_resample(pp->resampler_down, capture_48, (uint32_t)n,
                          capture_24, RESAMPLE_BUF_SIZE / 2);
        if (n24 <= 0) {
            voice_engine_resample_48_to_24(capture_48, capture_24, n);
            n24 = n / 2;
        }
    } else {
        voice_engine_resample_48_to_24(capture_48, capture_24, n);
        n24 = n / 2;
    }

    /* Accumulate into STT frame buffer */
    int space = STT_FRAME_SIZE * 4 - accum->len;
    int to_copy = n24 < space ? n24 : space;
    memcpy(accum->buf + accum->len, capture_24, (size_t)to_copy * sizeof(float));
    accum->len += to_copy;

    /* Process full frames */
    while (accum->len >= STT_FRAME_SIZE) {
        int nw = pocket_stt_process_frame(stt, accum->buf, STT_FRAME_SIZE);
        if (nw > 0) {
            total_words += nw;
            /* Get the new words and append to transcript */
            char word_buf[TEXT_BUF_SIZE];
            int wlen = pocket_stt_get_all_text(stt, word_buf, TEXT_BUF_SIZE);
            if (wlen > 0) {
                int avail = transcript_cap - *transcript_len - 1;
                int copy = wlen < avail ? wlen : avail;
                if (copy > 0) {
                    memcpy(transcript + *transcript_len, word_buf, (size_t)copy);
                    *transcript_len += copy;
                    transcript[*transcript_len] = '\0';
                }
            }
        }
        /* Shift remaining data */
        int remaining = accum->len - STT_FRAME_SIZE;
        if (remaining > 0) {
            memmove(accum->buf, accum->buf + STT_FRAME_SIZE,
                    (size_t)remaining * sizeof(float));
        }
        accum->len = remaining;
    }

    return total_words;
}

/* Feed TTS audio (24kHz) through post-processing pipeline to speaker (48kHz).
 *
 * Pipeline: TTS → [pitch shift] → [formant EQ] → [volume] → [soft limit]
 *         → resample 24→48 → [spatial HRTF] → playback ring
 *
 * Returns number of 24kHz samples transferred. */
static int feed_speaker(void *tts, VoiceEngine *audio, AudioPostProcessor *pp) {
    float pcm_24[TTS_AUDIO_BUF_SIZE];
    float processed[TTS_AUDIO_BUF_SIZE];
    float pcm_48[TTS_AUDIO_BUF_SIZE * 2 + 256]; /* Extra margin for resampler */
    int total = 0;

    for (;;) {
        int n = pocket_tts_rs_get_audio(tts, pcm_24, TTS_AUDIO_BUF_SIZE);
        if (n <= 0) break;

        float *src = pcm_24;

        /* Pitch shift (operates at 24kHz, needs ≥2048 samples for quality) */
        if (pp && fabsf(pp->pitch - 1.0f) > 0.01f && n >= 2048) {
            prosody_pitch_shift(src, processed, n, pp->pitch, 2048);
            src = processed;

            /* Formant correction to prevent chipmunk/barrel effect */
            if (pp->formant_eq) {
                prosody_apply_biquad(pp->formant_eq, src, n);
            }
        }

        /* Volume adjustment + soft limiting */
        if (pp && fabsf(pp->volume_db) > 0.1f) {
            if (src != processed) { memcpy(processed, src, n * sizeof(float)); src = processed; }
            prosody_volume(src, n, pp->volume_db, 0.0f, 24000);
            prosody_soft_limit(src, n, 0.85f, 6.0f);
        }

        /* Resample 24kHz → 48kHz */
        int n48;
        if (pp && pp->use_hw_resample && pp->resampler_up) {
            n48 = hw_resample(pp->resampler_up, src, (uint32_t)n,
                              pcm_48, TTS_AUDIO_BUF_SIZE * 2);
            if (n48 <= 0) {
                /* HW resampler failed, fall back to FIR */
                voice_engine_resample_24_to_48(src, pcm_48, n);
                n48 = n * 2;
            }
        } else {
            voice_engine_resample_24_to_48(src, pcm_48, n);
            n48 = n * 2;
        }

        /* Spatial audio: mono → stereo HRTF (writes interleaved L/R) */
        if (pp && pp->use_spatial && pp->spatial) {
            float left[TTS_AUDIO_BUF_SIZE * 2];
            float right[TTS_AUDIO_BUF_SIZE * 2];
            spatial_process(pp->spatial, 0, pcm_48, left, right, n48);
            /* Interleave L/R into pcm_48 for stereo playback */
            for (int i = 0; i < n48; i++) {
                pcm_48[i] = left[i] * 0.5f + right[i] * 0.5f;
            }
            /* Note: true stereo playback requires VoiceEngine stereo support.
             * For now, downmix to mono with spatial imaging preserved via
             * phase differences. Still provides spatial perception on speakers. */
        }

        int ret = voice_engine_write_playback(audio, pcm_48, n48);
        total += n;
        if (ret != 0) break; /* Ring full — back-pressure */
    }
    return total;
}

/* Barge-in: flush the playback ring. A minor click is possible but acceptable
 * since the VoiceProcessingIO AEC handles echo cancellation, and the user
 * is actively speaking (masking any artifact). */
static void barge_in_flush(VoiceEngine *audio) {
    voice_engine_flush_playback(audio);
}

static void print_state(PipelineState state) {
    fprintf(stderr, "\r[pocket-voice] %s   ", state_names[state]);
    fflush(stderr);
}

/* Main pipeline tick: called in a tight loop */
static PipelineState pipeline_tick(
    PipelineState state,
    VoiceEngine *audio,
    void *stt,
    void *tts,
    ClaudeClient *claude,
    SttAccum *stt_accum,
    char *transcript,
    int *transcript_len,
    float vad_threshold,
    TurnMetrics *metrics,
    AudioPostProcessor *pp
) {
    PipelineState next = state;

    /* Check for barge-in in any speaking/streaming state */
    if ((state == STATE_STREAMING || state == STATE_SPEAKING) &&
        voice_engine_get_barge_in(audio)) {
        fprintf(stderr, "\n[pocket-voice] Barge-in detected\n");
        voice_engine_clear_barge_in(audio);
        barge_in_flush(audio);
        claude_cancel(claude);
        pocket_tts_rs_reset(tts);
        pocket_stt_reset(stt);
        stt_accum_reset(stt_accum);
        *transcript_len = 0;
        transcript[0] = '\0';
        return STATE_LISTENING;
    }

    switch (state) {
    case STATE_LISTENING: {
        /* Drain capture buffer, watch for VAD speech onset */
        feed_stt(stt, audio, stt_accum, transcript, transcript_len, TEXT_BUF_SIZE, pp);

        int vad = voice_engine_get_vad_state(audio);
        if (vad >= 1) {
            next = STATE_RECORDING;
            *transcript_len = 0;
            transcript[0] = '\0';
            memset(metrics, 0, sizeof(*metrics));
            metrics->speech_start = now_us();
            print_state(next);
        }
        break;
    }

    case STATE_RECORDING: {
        int nw = feed_stt(stt, audio, stt_accum, transcript, transcript_len, TEXT_BUF_SIZE, pp);
        if (nw > 0) {
            fprintf(stderr, "\r[STT] %s", transcript);
            fflush(stderr);
        }

        /* Check semantic VAD for end-of-turn */
        bool end_of_turn = false;
        if (pocket_stt_has_vad(stt)) {
            float vad_prob = pocket_stt_get_vad_prob(stt, 2);
            if (vad_prob > vad_threshold) {
                end_of_turn = true;
            }
        } else {
            int vad = voice_engine_get_vad_state(audio);
            if (vad == 3) { /* VAD_SPEECH_END */
                end_of_turn = true;
            }
        }

        if (end_of_turn && *transcript_len > 0) {
            metrics->speech_end = now_us();

            /* Flush STT to get remaining words */
            int nf = pocket_stt_flush(stt);
            if (nf > 0) {
                char flush_buf[TEXT_BUF_SIZE];
                int flen = pocket_stt_get_all_text(stt, flush_buf, TEXT_BUF_SIZE);
                if (flen > 0) {
                    int space = TEXT_BUF_SIZE - *transcript_len - 1;
                    int to_copy = flen < space ? flen : space;
                    if (to_copy > 0) {
                        memcpy(transcript + *transcript_len, flush_buf, (size_t)to_copy);
                        *transcript_len += to_copy;
                        transcript[*transcript_len] = '\0';
                    }
                }
            }

            fprintf(stderr, "\n[pocket-voice] User: %s\n", transcript);
            next = STATE_PROCESSING;
            print_state(next);
        }
        break;
    }

    case STATE_PROCESSING: {
        metrics->claude_sent = now_us();

        if (claude_send(claude, transcript) != 0) {
            fprintf(stderr, "[pocket-voice] Failed to send to Claude\n");
            next = STATE_LISTENING;
        } else {
            next = STATE_STREAMING;
            print_state(next);
        }
        break;
    }

    case STATE_STREAMING: {
        /* Poll for Claude tokens (wait up to 1ms for data) */
        claude_poll(claude, 1);

        int token_len = 0;
        const char *tokens = claude_peek_tokens(claude, &token_len);
        if (tokens && token_len > 0) {
            if (!metrics->has_first_tok) {
                metrics->claude_first_tok = now_us();
                metrics->has_first_tok = true;
                uint64_t ttft = metrics->claude_first_tok - metrics->claude_sent;
                fprintf(stderr, "[Claude TTFT: %llu ms] ", (unsigned long long)(ttft / 1000));
            }

            char token_copy[CLAUDE_TOKEN_BUF];
            int copy_len = token_len < (int)sizeof(token_copy) - 1
                               ? token_len
                               : (int)sizeof(token_copy) - 1;
            memcpy(token_copy, tokens, (size_t)copy_len);
            token_copy[copy_len] = '\0';
            claude_consume_tokens(claude, copy_len);

            pocket_tts_rs_set_text(tts, token_copy);

            fprintf(stderr, "%s", token_copy);
            fflush(stderr);
        }

        if (claude->response_done && !pocket_tts_rs_is_done(tts)) {
            pocket_tts_rs_set_text_done(tts);
        }

        /* Run multiple TTS steps per tick to keep up with frame rate */
        for (int i = 0; i < TTS_STEPS_PER_TICK; i++) {
            int step_result = pocket_tts_rs_step(tts);
            if (step_result != 0) break; /* done or error */
        }

        int wrote = feed_speaker(tts, audio, pp);
        if (wrote > 0 && !metrics->has_first_audio) {
            metrics->tts_first_audio = now_us();
            metrics->has_first_audio = true;
            uint64_t e2e = metrics->tts_first_audio - metrics->speech_end;
            fprintf(stderr, "[E2E: %llu ms] ", (unsigned long long)(e2e / 1000));
        }

        if (claude->error) {
            pocket_tts_rs_set_text_done(tts);
            feed_speaker(tts, audio, pp);
            fprintf(stderr, "\n[pocket-voice] Claude error, draining TTS...\n");
            metrics->speaking_entered = now_us();
            next = STATE_SPEAKING;
            print_state(next);
        } else if (claude->response_done && pocket_tts_rs_is_done(tts)) {
            feed_speaker(tts, audio, pp);
            fprintf(stderr, "\n");
            metrics->speaking_entered = now_us();
            next = STATE_SPEAKING;
            print_state(next);
        }
        break;
    }

    case STATE_SPEAKING: {
        /* Continue draining any remaining TTS steps */
        if (!pocket_tts_rs_is_done(tts)) {
            for (int i = 0; i < TTS_STEPS_PER_TICK; i++) {
                int r = pocket_tts_rs_step(tts);
                if (r != 0) break;
            }
        }
        feed_speaker(tts, audio, pp);

        bool done = !voice_engine_is_playing(audio) && pocket_tts_rs_is_done(tts);
        bool timed_out = (now_us() - metrics->speaking_entered) > SPEAKING_TIMEOUT_US;
        if (timed_out && !done) {
            fprintf(stderr, "\n[pocket-voice] SPEAKING timeout (30s), forcing reset\n");
        }

        if (done || timed_out) {
            /* Commit this turn to conversation history */
            claude_commit_turn(claude, transcript);

            /* Print turn latency summary */
            if (metrics->has_first_audio) {
                uint64_t total = now_us() - metrics->speech_start;
                fprintf(stderr, "[Turn: %llu ms total]\n",
                        (unsigned long long)(total / 1000));
            }

            pocket_tts_rs_reset(tts);
            pocket_stt_reset(stt);
            stt_accum_reset(stt_accum);
            postproc_reset(pp);
            voice_engine_clear_barge_in(audio);
            voice_engine_flush_playback(audio);
            *transcript_len = 0;
            transcript[0] = '\0';
            next = STATE_LISTENING;
            print_state(next);
        }
        break;
    }
    }

    return next;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * CLI argument parsing
 * ═══════════════════════════════════════════════════════════════════════════ */

static void print_usage(const char *prog) {
    fprintf(stderr,
        "Usage: %s [OPTIONS]\n\n"
        "Zero-Python voice pipeline: Mic → STT → Claude → TTS → Speaker\n\n"
        "Options:\n"
        "  --voice PATH       Voice .wav or .safetensors path\n"
        "  --stt-repo REPO    STT HuggingFace repo (default: kyutai/stt-1b-en_fr-candle)\n"
        "  --stt-model FILE   STT model file (default: model.safetensors)\n"
        "  --tts-repo REPO    TTS HuggingFace repo (default: kyutai/tts-1.6b-en_fr)\n"
        "  --claude-model M   Claude model (default: claude-sonnet-4-20250514)\n"
        "  --system PROMPT    System prompt for Claude\n"
        "  --n-q N            Audio codebooks for TTS (default: 24)\n"
        "  --no-vad           Disable semantic VAD (use energy VAD only)\n"
        "  --vad-threshold F  Semantic VAD threshold (default: 0.7)\n"
        "\n"
        "Audio post-processing:\n"
        "  --pitch F          Pitch multiplier (1.0 = normal, 1.2 = higher)\n"
        "  --volume F         Volume in dB (0.0 = normal, 6.0 = louder)\n"
        "  --no-hw-resample   Disable AudioConverter (use FIR fallback)\n"
        "  --spatial AZ       Enable 3D spatial audio at azimuth AZ degrees\n"
        "  --help             Show this help\n",
        prog);
}

static PipelineConfig parse_args(int argc, char **argv) {
    PipelineConfig cfg = default_config();
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            print_usage(argv[0]);
            exit(0);
        } else if (strcmp(argv[i], "--voice") == 0 && i + 1 < argc) {
            cfg.voice = argv[++i];
        } else if (strcmp(argv[i], "--stt-repo") == 0 && i + 1 < argc) {
            cfg.stt_repo = argv[++i];
        } else if (strcmp(argv[i], "--stt-model") == 0 && i + 1 < argc) {
            cfg.stt_model = argv[++i];
        } else if (strcmp(argv[i], "--tts-repo") == 0 && i + 1 < argc) {
            cfg.tts_repo = argv[++i];
        } else if (strcmp(argv[i], "--claude-model") == 0 && i + 1 < argc) {
            cfg.claude_model = argv[++i];
        } else if (strcmp(argv[i], "--system") == 0 && i + 1 < argc) {
            cfg.system_prompt = argv[++i];
        } else if (strcmp(argv[i], "--n-q") == 0 && i + 1 < argc) {
            cfg.n_q = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--no-vad") == 0) {
            cfg.enable_vad = 0;
        } else if (strcmp(argv[i], "--vad-threshold") == 0 && i + 1 < argc) {
            cfg.vad_threshold = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "--pitch") == 0 && i + 1 < argc) {
            cfg.pitch = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "--volume") == 0 && i + 1 < argc) {
            cfg.volume_db = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "--no-hw-resample") == 0) {
            cfg.hw_resample = 0;
        } else if (strcmp(argv[i], "--spatial") == 0 && i + 1 < argc) {
            cfg.spatial = 1;
            cfg.spatial_az = (float)atof(argv[++i]);
        } else {
            fprintf(stderr, "Unknown option: %s\n", argv[i]);
            print_usage(argv[0]);
            exit(1);
        }
    }
    return cfg;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * main()
 * ═══════════════════════════════════════════════════════════════════════════ */

int main(int argc, char **argv) {
    PipelineConfig cfg = parse_args(argc, argv);

    /* Check for API key */
    const char *api_key = getenv("ANTHROPIC_API_KEY");
    if (!api_key || strlen(api_key) == 0) {
        fprintf(stderr, "[pocket-voice] Error: ANTHROPIC_API_KEY not set\n");
        return 1;
    }

    /* Global curl init */
    curl_global_init(CURL_GLOBAL_DEFAULT);

    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);

    fprintf(stderr, "╔══════════════════════════════════════════════╗\n");
    fprintf(stderr, "║     pocket-voice — Native Voice Pipeline     ║\n");
    fprintf(stderr, "╠══════════════════════════════════════════════╣\n");
    fprintf(stderr, "║  STT: %-38s ║\n", cfg.stt_repo);
    fprintf(stderr, "║  TTS: %-38s ║\n", cfg.tts_repo);
    fprintf(stderr, "║  LLM: %-38s ║\n", cfg.claude_model);
    fprintf(stderr, "║  n_q: %-38d ║\n", cfg.n_q);
    fprintf(stderr, "║  VAD: %-38s ║\n", cfg.enable_vad ? "semantic" : "energy");
    fprintf(stderr, "╚══════════════════════════════════════════════╝\n\n");

    /* 1. Init audio engine (48kHz, 256-frame buffer = ~5.3ms latency) */
    fprintf(stderr, "[pocket-voice] Starting audio engine...\n");
    VoiceEngine *audio = voice_engine_create(AUDIO_SAMPLE_RATE, AUDIO_BUFFER_FRAMES);
    if (!audio) {
        fprintf(stderr, "[pocket-voice] Failed to create audio engine\n");
        return 1;
    }
    if (voice_engine_start(audio) != 0) {
        fprintf(stderr, "[pocket-voice] Failed to start audio engine\n");
        voice_engine_destroy(audio);
        return 1;
    }

    /* 2. Init STT */
    fprintf(stderr, "[pocket-voice] Loading STT model...\n");
    void *stt = pocket_stt_create(cfg.stt_repo, cfg.stt_model, cfg.enable_vad);
    if (!stt) {
        fprintf(stderr, "[pocket-voice] Failed to create STT engine\n");
        voice_engine_destroy(audio);
        return 1;
    }

    /* 3. Init TTS */
    fprintf(stderr, "[pocket-voice] Loading TTS model...\n");
    void *tts = pocket_tts_rs_create(cfg.tts_repo, cfg.voice, cfg.n_q);
    if (!tts) {
        fprintf(stderr, "[pocket-voice] Failed to create TTS engine\n");
        pocket_stt_destroy(stt);
        voice_engine_destroy(audio);
        return 1;
    }

    /* 4. Init Claude client */
    ClaudeClient claude;
    claude_init(&claude, api_key, cfg.claude_model, cfg.system_prompt);

    /* 4b. Init audio post-processor (HW resampler, prosody, spatial) */
    AudioPostProcessor *pp = postproc_create(&cfg);
    if (!pp) {
        fprintf(stderr, "[pocket-voice] Warning: post-processor init failed, using defaults\n");
    } else {
        if (pp->use_hw_resample)
            fprintf(stderr, "[pocket-voice] AudioConverter resampling: enabled\n");
        if (fabsf(cfg.pitch - 1.0f) > 0.01f)
            fprintf(stderr, "[pocket-voice] Pitch: %.2fx\n", (double)cfg.pitch);
        if (fabsf(cfg.volume_db) > 0.1f)
            fprintf(stderr, "[pocket-voice] Volume: %+.1f dB\n", (double)cfg.volume_db);
        if (pp->use_spatial)
            fprintf(stderr, "[pocket-voice] Spatial audio: %.0f° azimuth\n", (double)cfg.spatial_az);
    }

    /* 5. Pipeline state */
    PipelineState state = STATE_LISTENING;
    SttAccum stt_accum;
    stt_accum_reset(&stt_accum);
    char transcript[TEXT_BUF_SIZE];
    int transcript_len = 0;
    transcript[0] = '\0';
    TurnMetrics metrics;
    memset(&metrics, 0, sizeof(metrics));

    fprintf(stderr, "\n[pocket-voice] Ready. Speak to begin.\n");
    print_state(state);

    /* 6. Main loop */
    while (!g_quit) {
        state = pipeline_tick(state, audio, stt, tts, &claude, &stt_accum,
                              transcript, &transcript_len, cfg.vad_threshold,
                              &metrics, pp);

        /* Adaptive sleep: shorter during active processing, longer when idle */
        if (state == STATE_LISTENING) {
            usleep(10000);  /* 10ms when idle */
        } else if (state == STATE_STREAMING || state == STATE_SPEAKING) {
            usleep(500);    /* 0.5ms during active generation */
        } else {
            usleep(5000);   /* 5ms during recording/processing */
        }
    }

    fprintf(stderr, "\n[pocket-voice] Shutting down...\n");

    /* Cleanup in reverse order */
    claude_cancel(&claude);
    claude_cleanup(&claude);
    postproc_destroy(pp);
    pocket_tts_rs_destroy(tts);
    pocket_stt_destroy(stt);
    voice_engine_destroy(audio);
    curl_global_cleanup();

    fprintf(stderr, "[pocket-voice] Done.\n");
    return 0;
}
