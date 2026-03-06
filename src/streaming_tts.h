#ifndef STREAMING_TTS_H
#define STREAMING_TTS_H

#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct StreamingTTS StreamingTTS;

typedef struct {
    int   sample_rate;         /* Audio output rate (24000) */
    int   min_tokens_to_start; /* Min LLM tokens before first TTS chunk (default: 4) */
    int   lookahead_tokens;    /* Tokens to buffer before committing to TTS (default: 2) */
    float commit_latency_ms;   /* Max time to wait before committing audio (default: 100) */
    int   rollback_enabled;    /* Allow rolling back uncommitted audio (default: 1) */
    int   crossfade_samples;   /* Crossfade length for rollback splicing (default: 240) */
} StreamingTTSConfig;

/* Audio segment tracking */
typedef struct {
    int   token_start;    /* First LLM token index this segment covers */
    int   token_end;      /* Last LLM token index (exclusive) */
    int   audio_start;    /* Start sample in output buffer */
    int   audio_len;      /* Number of audio samples */
    int   committed;      /* 1 = already sent to speaker, can't rollback */
} AudioSegment;

/* Create streaming TTS controller. Returns NULL on failure. */
StreamingTTS *streaming_tts_create(const StreamingTTSConfig *cfg);
void streaming_tts_destroy(StreamingTTS *stts);

/* Feed an LLM token.
 * token_text: the token string
 * token_idx: sequential index (0, 1, 2, ...)
 * Returns: number of audio samples now available (0 if not enough tokens yet). */
int streaming_tts_feed_token(StreamingTTS *stts,
                             const char *token_text, int token_idx);

/* Notify that LLM response is complete (triggers final TTS flush). */
void streaming_tts_finish(StreamingTTS *stts);

/* Get available audio for playback.
 * out_pcm: output buffer.
 * Returns number of samples written to out_pcm. */
int streaming_tts_get_audio(StreamingTTS *stts,
                            float *out_pcm, int max_samples);

/* Peek at available audio without consuming.
 * Returns pointer to internal buffer and sets *out_len.
 * Returns NULL if no audio available. */
const float *streaming_tts_peek_audio(const StreamingTTS *stts, int *out_len);

/* Advance (consume) audio samples. */
void streaming_tts_advance_audio(StreamingTTS *stts, int n_samples);

/* Check if all audio has been generated and consumed. */
bool streaming_tts_is_done(const StreamingTTS *stts);

/* Rollback: invalidate audio from token_idx onward.
 * Call this when the LLM's speculative tokens were wrong or when barge-in occurs.
 * Any uncommitted audio after token_idx is discarded.
 * Committed audio is kept (already sent to speaker).
 * Returns number of samples rolled back. */
int streaming_tts_rollback(StreamingTTS *stts, int from_token_idx);

/* Mark audio up to sample_pos as committed (sent to speaker, can't roll back). */
void streaming_tts_commit_audio(StreamingTTS *stts, int sample_pos);

/* Get the segment map (for debugging/logging).
 * Returns array of AudioSegment, sets *out_count. */
const AudioSegment *streaming_tts_get_segments(const StreamingTTS *stts,
                                               int *out_count);

/* Reset for new utterance. Clears all tokens, audio, and segments. */
void streaming_tts_reset(StreamingTTS *stts);

/* Get statistics. */
typedef struct {
    int   tokens_received;      /* Total LLM tokens fed */
    int   tokens_synthesized;   /* Tokens converted to audio */
    int   samples_generated;    /* Total audio samples generated */
    int   samples_committed;    /* Samples sent to speaker */
    int   samples_rolled_back;  /* Samples discarded by rollback */
    int   rollback_count;       /* Number of rollbacks */
    float latency_first_audio_ms; /* Time from first token to first audio */
} StreamingTTSStats;

void streaming_tts_get_stats(const StreamingTTS *stts, StreamingTTSStats *stats);

/* Set TTS synthesize callback.
 * This is called when the controller decides to generate audio for accumulated tokens.
 *
 * int tts_synthesize(void *ctx, const char *text, int text_len,
 *                    float *out_pcm, int max_samples);
 * Returns: number of samples generated, or -1 on error.
 *
 * The text may be a partial sentence — the TTS should handle this gracefully
 * (using Sonata LM's append_text / finish_text capability). */
void streaming_tts_set_synthesizer(StreamingTTS *stts,
    int (*tts_synthesize)(void *ctx, const char *text, int text_len,
                          float *out_pcm, int max_samples),
    void *tts_ctx);

#ifdef __cplusplus
}
#endif

#endif /* STREAMING_TTS_H */
