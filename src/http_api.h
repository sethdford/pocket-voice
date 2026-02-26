#ifndef HTTP_API_H
#define HTTP_API_H

#include <stdbool.h>
#include <stdint.h>

typedef struct HttpApi HttpApi;

/* Output encoding for TTS responses */
typedef enum {
    TTS_ENC_PCM_S16LE = 0,
    TTS_ENC_PCM_F32LE,
    TTS_ENC_PCM_MULAW,
    TTS_ENC_PCM_ALAW,
} TtsEncoding;

/* Output container for TTS responses */
typedef enum {
    TTS_CONTAINER_WAV = 0,
    TTS_CONTAINER_RAW,
    TTS_CONTAINER_MP3,
    TTS_CONTAINER_OPUS,
} TtsContainer;

/* Inline pronunciation override (from JSON request body) */
typedef struct {
    char word[64];
    char pronunciation[256];
} TtsPronOverride;

#define TTS_MAX_PRON_OVERRIDES 64

/* Word-level timestamp */
typedef struct {
    char  word[128];
    float start_s;
    float end_s;
} WordTimestamp;

#define TTS_MAX_WORD_TIMESTAMPS 256

/* Parsed TTS request parameters.
 * String fields are owned (strdup'd) — call tts_request_cleanup() when done. */
typedef struct {
    char        *text;
    char        *voice;
    char        *emotion;
    float        speed;          /* 0.25–4.0, default 1.0 */
    float        volume;         /* 0.5–2.0, default 1.0 */
    int          sample_rate;    /* 8000–48000, default 24000 */
    TtsEncoding  encoding;       /* default pcm_s16le */
    TtsContainer container;      /* default wav */
    int          stream;         /* 1 = chunked streaming response */
    int          word_timestamps; /* 1 = return JSON with word-level timestamps */
    TtsPronOverride pron_overrides[TTS_MAX_PRON_OVERRIDES];
    int          n_pron_overrides;
} TtsRequest;

/* In-memory voice registry for cloned voices */
#define VOICE_REGISTRY_MAX 32
#define VOICE_EMBED_MAX_DIM 512

typedef struct {
    char   id[64];
    float  embedding[VOICE_EMBED_MAX_DIM];
    int    dim;
    int    used;
} VoiceEntry;

typedef struct {
    void *stt_engine;
    void *tts_engine;
    void *llm_engine;

    int (*stt_feed)(void *engine, const float *pcm, int n_samples);
    int (*stt_flush)(void *engine);
    int (*stt_get_text)(void *engine, char *buf, int buf_size);
    /**
     * Optional: get word-level timestamps from last stt_flush.
     * Returns count or -1. NULL for engines that do not support timestamps.
     */
    int (*stt_get_words)(void *engine, WordTimestamp *out, int max_words);
    int (*stt_reset)(void *engine);

    int (*tts_speak)(void *engine, const char *text);
    int (*tts_step)(void *engine);
    int (*tts_is_done)(void *engine);
    int (*tts_set_text_done)(void *engine);
    int (*tts_get_audio)(void *engine, float *buf, int max_samples);
    int (*tts_reset)(void *engine);
    /**
     * Optional: get word-level timestamps from last TTS synthesis.
     * Returns count or -1. NULL for engines that do not support timestamps.
     */
    int (*tts_get_words)(void *engine, WordTimestamp *out, int max_words);

    int (*llm_send)(void *engine, const char *text);
    int (*llm_poll)(void *engine, int timeout_ms);
    const char *(*llm_peek)(void *engine, int *len);
    void (*llm_consume)(void *engine, int n);
    bool (*llm_is_done)(void *engine);

    /**
     * Full prosody pipeline: emphasis → SSML parse → process_segment.
     * If non-NULL, used by /v1/audio/speech instead of raw tts_speak().
     * Returns 0 on success.
     */
    int (*process_text)(void *ctx, void *tts_engine, const char *text,
                        float speed, float volume, const char *emotion,
                        const char *voice,
                        const char (*pron_words)[64],
                        const char (*pron_replacements)[256],
                        int n_pron);
    void *process_ctx;

    /**
     * Voice cloning: extract speaker embedding from PCM audio.
     * Returns embedding dimension, or -1 on error. Writes to emb_out.
     */
    int (*clone_voice)(void *ctx, const float *pcm, int n_samples,
                       float *emb_out, int max_dim);

    /**
     * Set speaker embedding on the TTS flow engine.
     * Returns 0 on success.
     */
    int (*set_speaker_embedding)(void *tts_engine, const float *emb, int dim);

    void *clone_ctx;
} HttpApiEngines;

HttpApi *http_api_create(int port, HttpApiEngines engines);
void     http_api_set_api_key(HttpApi *api, const char *key);
int      http_api_start(HttpApi *api);
void     http_api_stop(HttpApi *api);
void     http_api_destroy(HttpApi *api);

#endif
