/**
 * latency_profiler.h — Real-time latency measurement for voice pipeline.
 *
 * Measures wall-clock latency at each pipeline stage:
 *   1. Audio capture → STT start (capture latency)
 *   2. STT processing time
 *   3. STT end → LLM first token (LLM TTFT)
 *   4. LLM first token → TTS first sample (TTS TTFS)
 *   5. TTS first sample → speaker (playback latency)
 *   6. End-to-end: user stops speaking → first audio response
 *
 * Uses mach_absolute_time for nanosecond precision on Apple Silicon.
 */

#ifndef LATENCY_PROFILER_H
#define LATENCY_PROFILER_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    /* Timestamps (mach_absolute_time units) */
    uint64_t vad_end;          /* User stopped speaking */
    uint64_t stt_start;        /* STT processing began */
    uint64_t stt_end;          /* STT returned text */
    uint64_t llm_start;        /* LLM request sent */
    uint64_t llm_first_token;  /* First LLM token received */
    uint64_t llm_end;          /* LLM response complete */
    uint64_t tts_start;        /* First TTS chunk sent */
    uint64_t tts_first_audio;  /* First audio sample from TTS */
    uint64_t speaker_start;    /* First sample written to speaker */

    /* Derived latencies (milliseconds) */
    float stt_ms;              /* STT processing time */
    float llm_ttft_ms;         /* LLM time to first token */
    float llm_total_ms;        /* Total LLM generation time */
    float tts_ttfs_ms;         /* TTS time to first sample */
    float e2e_ms;              /* End-to-end: vad_end → speaker_start */
    float pipeline_overhead_ms; /* Overhead beyond STT+LLM+TTS */

    /* Running statistics */
    int    n_turns;
    double sum_e2e;
    double sum_e2e_sq;
    float  min_e2e;
    float  max_e2e;
    float  p50_e2e;
    float  p95_e2e;
    float  p99_e2e;

    float  history[256];       /* Circular buffer of E2E latencies */
    int    history_len;
    int    history_pos;
} LatencyProfile;

/** Initialize profiler. */
void lp_init(LatencyProfile *lp);

/** Record a pipeline stage timestamp. */
void lp_mark_vad_end(LatencyProfile *lp);
void lp_mark_stt_start(LatencyProfile *lp);
void lp_mark_stt_end(LatencyProfile *lp);
void lp_mark_llm_start(LatencyProfile *lp);
void lp_mark_llm_first_token(LatencyProfile *lp);
void lp_mark_llm_end(LatencyProfile *lp);
void lp_mark_tts_start(LatencyProfile *lp);
void lp_mark_tts_first_audio(LatencyProfile *lp);
void lp_mark_speaker_start(LatencyProfile *lp);

/** Compute derived latencies for the current turn. */
void lp_compute(LatencyProfile *lp);

/** Print a summary of the current turn's latencies to stderr. */
void lp_print_turn(const LatencyProfile *lp);

/** Print aggregate statistics across all turns. */
void lp_print_summary(const LatencyProfile *lp);

/** Convert mach_absolute_time delta to milliseconds. */
float lp_mach_to_ms(uint64_t delta);

#ifdef __cplusplus
}
#endif

#endif /* LATENCY_PROFILER_H */
