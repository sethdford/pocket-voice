/**
 * streaming_tts.c — Unified streaming TTS controller with speculative generation
 * and token-level rollback.
 *
 * Enables continuous TTS generation as LLM tokens arrive, with ability to roll
 * back uncommitted audio when the LLM changes direction (speculative decode for TTS).
 */

#ifndef ACCELERATE_NEW_LAPACK
#define ACCELERATE_NEW_LAPACK
#endif

#include "streaming_tts.h"
#include <Accelerate/Accelerate.h>
#include <mach/mach_time.h>
#include <pthread.h>
#include <stdlib.h>
#include <string.h>

#define STTS_MAX_TOKENS 2048
#define STTS_MAX_SEGMENTS 256
#define STTS_AUDIO_CAP (24000 * 30)  /* 30 seconds max */
#define STTS_TOKEN_BUF_SIZE (32 * 1024)

struct StreamingTTS {
    StreamingTTSConfig cfg;

    /* Token buffer */
    char token_buf[STTS_TOKEN_BUF_SIZE];
    int  token_buf_len;
    int  token_offsets[STTS_MAX_TOKENS]; /* Start offset of each token in token_buf */
    int  n_tokens;

    /* Synthesis tracking */
    int  tokens_synthesized; /* How many tokens have been sent to TTS */
    int  pending_start;      /* Token index of first un-synthesized token */
    int  first_chunk_done;   /* 1 = first eager chunk already generated */

    /* Audio output buffer */
    float *audio_buf;
    int   audio_len;      /* Total samples in buffer */
    int   audio_read;     /* Samples consumed (read cursor) */
    int   audio_committed; /* Samples committed (can't rollback) */

    /* Segment tracking */
    AudioSegment segments[STTS_MAX_SEGMENTS];
    int n_segments;

    /* TTS callback */
    int (*tts_synthesize)(void *ctx, const char *text, int text_len,
                          float *out_pcm, int max_samples);
    void *tts_ctx;

    /* State */
    int finished;      /* LLM is done */
    int all_generated; /* All tokens synthesized */

    /* Stats */
    StreamingTTSStats stats;
    uint64_t first_token_time;
    uint64_t first_audio_time;

    /* Crossfade: save tail of committed audio for blend after rollback */
    float *crossfade_buf;
    int   crossfade_tail_len; /* Length of valid data in crossfade_buf */
    int   just_rolled_back;   /* 1 = last op was rollback, fade in next synthesis */

    pthread_mutex_t mutex;
};

/* mach_absolute_time to milliseconds */
static float mach_to_ms(uint64_t delta) {
    static mach_timebase_info_data_t timebase;
    static int initialized = 0;
    if (!initialized) {
        mach_timebase_info(&timebase);
        initialized = 1;
    }
    return (float)((double)delta * timebase.numer / timebase.denom / 1e6);
}

/* Blend saved rollback tail with start of new audio. crossfade_buf has the old tail;
 * new_audio points to the start of newly synthesized samples; we blend in place. */
static void blend_rollback_crossfade(float *crossfade_buf, int crossfade_len,
                                    float *new_audio, int new_len) {
    if (crossfade_len <= 0 || new_len <= 0) return;
    int n = crossfade_len;
    if (n > new_len) n = new_len;

    /* output[i] = crossfade_buf[i] * (1-t) + new_audio[i] * t, t = i/n */
    float *out = new_audio;
    for (int i = 0; i < n; i++) {
        float t = (float)i / (float)n;
        out[i] = crossfade_buf[i] * (1.0f - t) + out[i] * t;
    }
}

static void defaults(StreamingTTSConfig *cfg) {
    if (cfg->sample_rate <= 0) cfg->sample_rate = 24000;
    if (cfg->min_tokens_to_start <= 0) cfg->min_tokens_to_start = 4;
    if (cfg->lookahead_tokens < 0) cfg->lookahead_tokens = 2;
    if (cfg->commit_latency_ms <= 0) cfg->commit_latency_ms = 100.0f;
    if (cfg->rollback_enabled < 0) cfg->rollback_enabled = 1;
    if (cfg->crossfade_samples <= 0) cfg->crossfade_samples = 240;
}

StreamingTTS *streaming_tts_create(const StreamingTTSConfig *cfg) {
    StreamingTTS *stts = (StreamingTTS *)calloc(1, sizeof(StreamingTTS));
    if (!stts) return NULL;

    if (cfg) {
        stts->cfg = *cfg;
    }
    defaults(&stts->cfg);

    stts->audio_buf = (float *)calloc(STTS_AUDIO_CAP, sizeof(float));
    if (!stts->audio_buf) {
        free(stts);
        return NULL;
    }

    stts->crossfade_buf = (float *)calloc((size_t)stts->cfg.crossfade_samples, sizeof(float));
    if (!stts->crossfade_buf) {
        free(stts->audio_buf);
        free(stts);
        return NULL;
    }

    pthread_mutex_init(&stts->mutex, NULL);
    return stts;
}

void streaming_tts_destroy(StreamingTTS *stts) {
    if (!stts) return;
    pthread_mutex_destroy(&stts->mutex);
    free(stts->crossfade_buf);
    free(stts->audio_buf);
    free(stts);
}

/* Internal: perform synthesis for text [start_off, end_off) and append to audio. */
static int do_synthesize(StreamingTTS *stts, int start_off, int end_off,
                         int token_start_idx, int token_end_idx) {
    if (!stts->tts_synthesize) return 0;
    int text_len = end_off - start_off;
    if (text_len <= 0) return 0;

    int cap = STTS_AUDIO_CAP - stts->audio_len;
    if (cap <= 0) return 0;

    int written = stts->tts_synthesize(stts->tts_ctx,
                                       stts->token_buf + start_off, text_len,
                                       stts->audio_buf + stts->audio_len, cap);
    if (written <= 0) return 0;

    /* Apply rollback crossfade if we have a saved tail; else fade in to avoid click */
    if (stts->crossfade_tail_len > 0) {
        blend_rollback_crossfade(stts->crossfade_buf, stts->crossfade_tail_len,
                                 stts->audio_buf + stts->audio_len,
                                 written);
        stts->crossfade_tail_len = 0;
    } else {
        /* Fade in first N samples to avoid click at splice after rollback */
        int cf = stts->cfg.crossfade_samples;
        if (cf > 0 && written >= cf) {
            float start = 0.0f, step = 1.0f / (float)cf;
            float *dst = stts->audio_buf + stts->audio_len;
            vDSP_vramp(&start, &step, stts->crossfade_buf, 1, (vDSP_Length)cf);
            vDSP_vmul(dst, 1, stts->crossfade_buf, 1, dst, 1, (vDSP_Length)cf);
        }
    }

    /* Create segment */
    if (stts->n_segments < STTS_MAX_SEGMENTS) {
        AudioSegment *seg = &stts->segments[stts->n_segments++];
        seg->token_start = token_start_idx;
        seg->token_end = token_end_idx;
        seg->audio_start = stts->audio_len;
        seg->audio_len = written;
        seg->committed = 0;
    }

    stts->audio_len += written;
    stts->stats.samples_generated += written;
    stts->stats.tokens_synthesized += (token_end_idx - token_start_idx);
    stts->pending_start = token_end_idx;

    if (!stts->first_audio_time && written > 0) {
        stts->first_audio_time = mach_absolute_time();
    }

    return written;
}

int streaming_tts_feed_token(StreamingTTS *stts,
                             const char *token_text, int token_idx) {
    if (!stts || !token_text) return 0;

    pthread_mutex_lock(&stts->mutex);

    if (stts->n_tokens >= STTS_MAX_TOKENS) {
        pthread_mutex_unlock(&stts->mutex);
        return 0;
    }

    size_t tlen = strlen(token_text);
    size_t need = (size_t)stts->token_buf_len + tlen + 1;
    if (need > STTS_TOKEN_BUF_SIZE) {
        pthread_mutex_unlock(&stts->mutex);
        return 0;
    }

    if (stts->n_tokens == 0) {
        stts->first_token_time = mach_absolute_time();
    }

    /* Append token */
    stts->token_offsets[stts->n_tokens] = stts->token_buf_len;
    memcpy(stts->token_buf + stts->token_buf_len, token_text, tlen + 1);
    stts->token_buf_len += (int)(tlen);
    stts->n_tokens++;
    stts->token_offsets[stts->n_tokens] = stts->token_buf_len; /* end sentinel */
    stts->stats.tokens_received++;

    int min_tok = stts->cfg.min_tokens_to_start;
    int lookahead = stts->cfg.lookahead_tokens;

    /* First chunk: eager when we have enough tokens */
    if (!stts->first_chunk_done && stts->n_tokens >= min_tok) {
        do_synthesize(stts, 0, stts->token_buf_len, 0, stts->n_tokens);
        stts->first_chunk_done = 1;
    }
    /* Subsequent: synthesize when lookahead buffer is filled */
    else if (stts->first_chunk_done && stts->n_tokens - stts->pending_start > lookahead) {
        int end_token = stts->n_tokens - lookahead;
        int start_off = stts->token_offsets[stts->pending_start];
        int end_off = stts->token_offsets[end_token];
        do_synthesize(stts, start_off, end_off, stts->pending_start, end_token);
    }

    int samples_after = stts->audio_len - stts->audio_read;
    pthread_mutex_unlock(&stts->mutex);

    return samples_after;
}

void streaming_tts_finish(StreamingTTS *stts) {
    if (!stts) return;

    pthread_mutex_lock(&stts->mutex);
    stts->finished = 1;

    /* Synthesize all remaining tokens */
    if (stts->tts_synthesize && stts->pending_start < stts->n_tokens) {
        int start_off = stts->token_offsets[stts->pending_start];
        int end_off = stts->token_buf_len;
        do_synthesize(stts, start_off, end_off, stts->pending_start, stts->n_tokens);
    }
    stts->all_generated = 1;
    pthread_mutex_unlock(&stts->mutex);
}

int streaming_tts_get_audio(StreamingTTS *stts, float *out_pcm, int max_samples) {
    if (!stts || !out_pcm || max_samples <= 0) return 0;

    pthread_mutex_lock(&stts->mutex);
    int avail = stts->audio_len - stts->audio_read;
    if (avail <= 0) {
        pthread_mutex_unlock(&stts->mutex);
        return 0;
    }
    int n = (max_samples < avail) ? max_samples : avail;
    memcpy(out_pcm, stts->audio_buf + stts->audio_read, (size_t)n * sizeof(float));
    stts->audio_read += n;
    pthread_mutex_unlock(&stts->mutex);

    return n;
}

const float *streaming_tts_peek_audio(const StreamingTTS *stts, int *out_len) {
    if (!stts || !out_len) return NULL;

    pthread_mutex_lock((pthread_mutex_t *)&stts->mutex);
    int avail = stts->audio_len - stts->audio_read;
    if (avail <= 0) {
        pthread_mutex_unlock((pthread_mutex_t *)&stts->mutex);
        *out_len = 0;
        return NULL;
    }
    *out_len = avail;
    const float *ptr = stts->audio_buf + stts->audio_read;
    pthread_mutex_unlock((pthread_mutex_t *)&stts->mutex);

    return ptr;
}

void streaming_tts_advance_audio(StreamingTTS *stts, int n_samples) {
    if (!stts || n_samples <= 0) return;

    pthread_mutex_lock(&stts->mutex);
    int avail = stts->audio_len - stts->audio_read;
    int n = (n_samples < avail) ? n_samples : avail;
    stts->audio_read += n;
    pthread_mutex_unlock(&stts->mutex);
}

bool streaming_tts_is_done(const StreamingTTS *stts) {
    if (!stts) return true;

    pthread_mutex_lock((pthread_mutex_t *)&stts->mutex);
    /* Done when: no tokens and no audio (idle), or finished + all generated + all consumed */
    bool done = (stts->n_tokens == 0 && stts->audio_len == 0) ||
                (stts->finished && stts->all_generated &&
                 stts->audio_read >= stts->audio_len);
    pthread_mutex_unlock((pthread_mutex_t *)&stts->mutex);

    return done;
}

int streaming_tts_rollback(StreamingTTS *stts, int from_token_idx) {
    if (!stts || !stts->cfg.rollback_enabled) return 0;

    pthread_mutex_lock(&stts->mutex);

    int rolled = 0;

    /* Find first segment covering or after from_token_idx */
    int seg_start = -1;
    for (int i = 0; i < stts->n_segments; i++) {
        if (stts->segments[i].token_end > from_token_idx) {
            seg_start = i;
            break;
        }
    }

    if (seg_start >= 0) {
        /* Save tail of committed audio for crossfade before we truncate */
        int cf = stts->cfg.crossfade_samples;
        if (cf > 0 && stts->audio_committed >= cf) {
            int tail_start = stts->audio_committed - cf;
            memcpy(stts->crossfade_buf, stts->audio_buf + tail_start,
                   (size_t)cf * sizeof(float));
            stts->crossfade_tail_len = cf;
        }

        /* Remove segments and truncate audio */
        for (int i = seg_start; i < stts->n_segments; i++) {
            AudioSegment *seg = &stts->segments[i];
            if (seg->audio_start >= stts->audio_committed) {
                rolled += seg->audio_len;
            }
        }

        /* Truncate audio_len to the last sample before the first discarded segment */
        int new_audio_len = stts->audio_committed;
        for (int i = 0; i < stts->n_segments; i++) {
            AudioSegment *seg = &stts->segments[i];
            if (seg->token_end <= from_token_idx && seg->audio_start + seg->audio_len > new_audio_len) {
                new_audio_len = seg->audio_start + seg->audio_len;
            }
        }
        /* Only keep up to last committed or last kept segment */
        for (int i = seg_start; i < stts->n_segments; i++) {
            if (stts->segments[i].audio_start >= stts->audio_committed) {
                break;
            }
            new_audio_len = stts->segments[i].audio_start + stts->segments[i].audio_len;
        }
        stts->audio_len = new_audio_len;
        if (stts->audio_read > stts->audio_len) {
            stts->audio_read = stts->audio_len;
        }

        /* Compact segments: remove discarded ones */
        int wr = 0;
        for (int i = 0; i < stts->n_segments; i++) {
            if (stts->segments[i].audio_start < stts->audio_committed ||
                stts->segments[i].token_end <= from_token_idx) {
                if (wr != i) {
                    stts->segments[wr] = stts->segments[i];
                }
                wr++;
            }
        }
        stts->n_segments = wr;
    }

    /* Truncate token buffer and reset synthesis state */
    if (from_token_idx < stts->n_tokens) {
        stts->token_buf_len = stts->token_offsets[from_token_idx];
        stts->n_tokens = from_token_idx;
        stts->pending_start = from_token_idx;
        stts->first_chunk_done = (from_token_idx > 0);
        stts->just_rolled_back = 1;
    }

    stts->stats.samples_rolled_back += rolled;
    stts->stats.rollback_count++;
    pthread_mutex_unlock(&stts->mutex);

    return rolled;
}

void streaming_tts_commit_audio(StreamingTTS *stts, int sample_pos) {
    if (!stts || sample_pos <= stts->audio_committed) return;

    pthread_mutex_lock(&stts->mutex);
    if (sample_pos > stts->audio_len) sample_pos = stts->audio_len;
    stts->audio_committed = sample_pos;

    /* Mark segments as committed */
    for (int i = 0; i < stts->n_segments; i++) {
        if (stts->segments[i].audio_start + stts->segments[i].audio_len <= sample_pos) {
            stts->segments[i].committed = 1;
        }
    }
    stts->stats.samples_committed = stts->audio_committed;
    pthread_mutex_unlock(&stts->mutex);
}

const AudioSegment *streaming_tts_get_segments(const StreamingTTS *stts,
                                               int *out_count) {
    if (!stts || !out_count) return NULL;

    pthread_mutex_lock((pthread_mutex_t *)&stts->mutex);
    *out_count = stts->n_segments;
    const AudioSegment *seg = stts->segments;
    pthread_mutex_unlock((pthread_mutex_t *)&stts->mutex);

    return seg;
}

void streaming_tts_reset(StreamingTTS *stts) {
    if (!stts) return;

    pthread_mutex_lock(&stts->mutex);
    stts->token_buf_len = 0;
    stts->n_tokens = 0;
    stts->pending_start = 0;
    stts->tokens_synthesized = 0;
    stts->first_chunk_done = 0;
    stts->audio_len = 0;
    stts->audio_read = 0;
    stts->audio_committed = 0;
    stts->n_segments = 0;
    stts->finished = 0;
    stts->all_generated = 0;
    stts->crossfade_tail_len = 0;
    stts->just_rolled_back = 0;
    memset(&stts->stats, 0, sizeof(stts->stats));
    stts->first_token_time = 0;
    stts->first_audio_time = 0;
    pthread_mutex_unlock(&stts->mutex);
}

void streaming_tts_get_stats(const StreamingTTS *stts, StreamingTTSStats *stats) {
    if (!stts || !stats) return;

    pthread_mutex_lock((pthread_mutex_t *)&stts->mutex);
    *stats = stts->stats;
    if (stts->first_token_time && stts->first_audio_time) {
        stats->latency_first_audio_ms = mach_to_ms(stts->first_audio_time - stts->first_token_time);
    }
    pthread_mutex_unlock((pthread_mutex_t *)&stts->mutex);
}

void streaming_tts_set_synthesizer(StreamingTTS *stts,
    int (*tts_synthesize)(void *ctx, const char *text, int text_len,
                          float *out_pcm, int max_samples),
    void *tts_ctx) {
    if (!stts) return;

    pthread_mutex_lock(&stts->mutex);
    stts->tts_synthesize = tts_synthesize;
    stts->tts_ctx = tts_ctx;
    pthread_mutex_unlock(&stts->mutex);
}
