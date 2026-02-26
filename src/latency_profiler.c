/**
 * latency_profiler.c — Real-time latency measurement for voice pipeline.
 * Uses mach_absolute_time for nanosecond precision on Apple Silicon.
 */

#include "latency_profiler.h"
#include <mach/mach_time.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

static mach_timebase_info_data_t timebase_info;
static int timebase_initialized = 0;

static void ensure_timebase(void) {
    if (!timebase_initialized) {
        mach_timebase_info(&timebase_info);
        timebase_initialized = 1;
    }
}

float lp_mach_to_ms(uint64_t delta) {
    ensure_timebase();
    return (float)((double)delta * timebase_info.numer / timebase_info.denom / 1e6);
}

static uint64_t now(void) { return mach_absolute_time(); }

void lp_init(LatencyProfile *lp) {
    ensure_timebase();
    memset(lp, 0, sizeof(*lp));
    lp->min_e2e = 1e9f;
}

void lp_mark_vad_end(LatencyProfile *lp)         { lp->vad_end = now(); }
void lp_mark_stt_start(LatencyProfile *lp)       { lp->stt_start = now(); }
void lp_mark_stt_end(LatencyProfile *lp)         { lp->stt_end = now(); }
void lp_mark_llm_start(LatencyProfile *lp)       { lp->llm_start = now(); }
void lp_mark_llm_first_token(LatencyProfile *lp) { lp->llm_first_token = now(); }
void lp_mark_llm_end(LatencyProfile *lp)         { lp->llm_end = now(); }
void lp_mark_tts_start(LatencyProfile *lp)       { lp->tts_start = now(); }
void lp_mark_tts_first_audio(LatencyProfile *lp) { lp->tts_first_audio = now(); }
void lp_mark_speaker_start(LatencyProfile *lp)   { lp->speaker_start = now(); }

void lp_compute(LatencyProfile *lp) {
    if (lp->stt_start && lp->stt_end)
        lp->stt_ms = lp_mach_to_ms(lp->stt_end - lp->stt_start);

    if (lp->llm_start && lp->llm_first_token)
        lp->llm_ttft_ms = lp_mach_to_ms(lp->llm_first_token - lp->llm_start);

    if (lp->llm_start && lp->llm_end)
        lp->llm_total_ms = lp_mach_to_ms(lp->llm_end - lp->llm_start);

    if (lp->tts_start && lp->tts_first_audio)
        lp->tts_ttfs_ms = lp_mach_to_ms(lp->tts_first_audio - lp->tts_start);

    if (lp->vad_end && lp->speaker_start) {
        lp->e2e_ms = lp_mach_to_ms(lp->speaker_start - lp->vad_end);

        float core = lp->stt_ms + lp->llm_ttft_ms + lp->tts_ttfs_ms;
        lp->pipeline_overhead_ms = lp->e2e_ms - core;
        if (lp->pipeline_overhead_ms < 0) lp->pipeline_overhead_ms = 0;
    }

    /* Update running stats */
    if (lp->e2e_ms > 0) {
        lp->n_turns++;
        lp->sum_e2e += (double)lp->e2e_ms;
        lp->sum_e2e_sq += (double)lp->e2e_ms * lp->e2e_ms;
        if (lp->e2e_ms < lp->min_e2e) lp->min_e2e = lp->e2e_ms;
        if (lp->e2e_ms > lp->max_e2e) lp->max_e2e = lp->e2e_ms;

        lp->history[lp->history_pos] = lp->e2e_ms;
        lp->history_pos = (lp->history_pos + 1) % 256;
        if (lp->history_len < 256) lp->history_len++;
    }
}

static int float_cmp(const void *a, const void *b) {
    float fa = *(const float *)a, fb = *(const float *)b;
    return (fa > fb) - (fa < fb);
}

void lp_print_turn(const LatencyProfile *lp) {
    fprintf(stderr, "┌─ Turn %d Latency ─────────────────────────┐\n", lp->n_turns);
    fprintf(stderr, "│  STT:          %7.1f ms                  │\n", (double)lp->stt_ms);
    fprintf(stderr, "│  LLM TTFT:     %7.1f ms                  │\n", (double)lp->llm_ttft_ms);
    fprintf(stderr, "│  LLM total:    %7.1f ms                  │\n", (double)lp->llm_total_ms);
    fprintf(stderr, "│  TTS TTFS:     %7.1f ms                  │\n", (double)lp->tts_ttfs_ms);
    fprintf(stderr, "│  Overhead:     %7.1f ms                  │\n", (double)lp->pipeline_overhead_ms);
    fprintf(stderr, "│  ──────────────────────────                │\n");
    fprintf(stderr, "│  E2E:          %7.1f ms                  │\n", (double)lp->e2e_ms);
    fprintf(stderr, "└────────────────────────────────────────────┘\n");
}

void lp_print_summary(const LatencyProfile *lp) {
    if (lp->n_turns == 0) {
        fprintf(stderr, "[latency] No turns recorded\n");
        return;
    }

    float mean = (float)(lp->sum_e2e / lp->n_turns);
    float var = (float)(lp->sum_e2e_sq / lp->n_turns - (double)mean * mean);
    float stddev = var > 0 ? sqrtf(var) : 0;

    /* Compute percentiles from history */
    float sorted[256];
    int n = lp->history_len;
    memcpy(sorted, lp->history, n * sizeof(float));
    qsort(sorted, n, sizeof(float), float_cmp);

    float p50 = sorted[n / 2];
    float p95 = sorted[(int)(n * 0.95f)];
    float p99 = sorted[(int)(n * 0.99f)];

    fprintf(stderr, "╔══════════════════════════════════════════════╗\n");
    fprintf(stderr, "║          Latency Summary (%3d turns)         ║\n", lp->n_turns);
    fprintf(stderr, "╠══════════════════════════════════════════════╣\n");
    fprintf(stderr, "║  Mean:   %7.1f ms                          ║\n", (double)mean);
    fprintf(stderr, "║  StdDev: %7.1f ms                          ║\n", (double)stddev);
    fprintf(stderr, "║  Min:    %7.1f ms                          ║\n", (double)lp->min_e2e);
    fprintf(stderr, "║  P50:    %7.1f ms                          ║\n", (double)p50);
    fprintf(stderr, "║  P95:    %7.1f ms                          ║\n", (double)p95);
    fprintf(stderr, "║  P99:    %7.1f ms                          ║\n", (double)p99);
    fprintf(stderr, "║  Max:    %7.1f ms                          ║\n", (double)lp->max_e2e);
    fprintf(stderr, "╚══════════════════════════════════════════════╝\n");
}
