#ifndef STREAMING_LLM_H
#define STREAMING_LLM_H

#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct StreamingLLM StreamingLLM;

typedef enum {
    STREAMING_LLM_GEMINI_LIVE = 0,
    STREAMING_LLM_OPENAI_REALTIME,
} StreamingLLMType;

typedef struct {
    StreamingLLMType type;
    const char *api_key;
    const char *model;           /* e.g. "gemini-2.0-flash-live" or "gpt-4o-realtime" */
    const char *system_prompt;
    int input_sample_rate;       /* Audio input rate (16000) */
    int output_sample_rate;      /* Audio output rate (24000) */
    const char *voice;           /* Voice name (NULL for default) */
    float temperature;           /* 0.0 - 1.0 */
} StreamingLLMConfig;

/* Create streaming LLM client.
 * Does NOT connect — call streaming_llm_connect() separately.
 * Returns NULL on failure. */
StreamingLLM *streaming_llm_create(const StreamingLLMConfig *cfg);
void streaming_llm_destroy(StreamingLLM *llm);

/* Connect to the API (WebSocket upgrade).
 * Returns 0 on success, -1 on error. */
int streaming_llm_connect(StreamingLLM *llm);

/* Disconnect gracefully. */
void streaming_llm_disconnect(StreamingLLM *llm);

/* Check if connected. */
int streaming_llm_is_connected(const StreamingLLM *llm);

/* Get configured input/output sample rates. */
int streaming_llm_input_sample_rate(const StreamingLLM *llm);
int streaming_llm_output_sample_rate(const StreamingLLM *llm);

/* === Text Mode (LLMClient-compatible) === */

/* Send user text for text-mode response (same as existing LLM backends).
 * Returns 0 on success. */
int streaming_llm_send_text(StreamingLLM *llm, const char *user_text);

/* Peek at text tokens received. Same semantics as LLMClient.peek_tokens. */
const char *streaming_llm_peek_text(StreamingLLM *llm, int *out_len);

/* Consume text tokens. */
void streaming_llm_consume_text(StreamingLLM *llm, int count);

/* === Audio Mode (streaming audio in/out) === */

/* Send audio chunk to the API.
 * pcm: PCM float32 at input_sample_rate.
 * Returns 0 on success. */
int streaming_llm_send_audio(StreamingLLM *llm, const float *pcm, int n_samples);

/* Receive audio from the API.
 * out_pcm: buffer for received audio at output_sample_rate.
 * Returns number of samples read, 0 if no audio available. */
int streaming_llm_recv_audio(StreamingLLM *llm, float *out_pcm, int max_samples);

/* Check if audio is available for reading. */
int streaming_llm_audio_available(const StreamingLLM *llm);

/* === Shared === */

/* Poll for events (call in main loop). timeout_ms=0 for non-blocking.
 * Returns number of events processed. */
int streaming_llm_poll(StreamingLLM *llm, int timeout_ms);

/* Check if response is done. */
bool streaming_llm_is_done(const StreamingLLM *llm);

/* Cancel current response. */
void streaming_llm_cancel(StreamingLLM *llm);

/* Signal end of user turn (for APIs that need explicit turn-taking). */
void streaming_llm_end_turn(StreamingLLM *llm);

/* Commit conversation turn. */
void streaming_llm_commit_turn(StreamingLLM *llm, const char *user_text);

/* Check for errors. */
bool streaming_llm_has_error(const StreamingLLM *llm);

/* Get error message. */
const char *streaming_llm_error_message(const StreamingLLM *llm);

/* Get transcript of audio received (if the API provides it). */
const char *streaming_llm_get_transcript(const StreamingLLM *llm);

/* Set function calling tools (JSON schema). */
int streaming_llm_set_tools(StreamingLLM *llm, const char *tools_json);

/* Enable/disable server-side VAD (for APIs that support it).
 * When enabled, the API decides turn boundaries from audio.
 * When disabled, we use our own VAP/EOU. */
void streaming_llm_set_server_vad(StreamingLLM *llm, bool enabled);

#ifdef __cplusplus
}
#endif

#endif /* STREAMING_LLM_H */
