/**
 * latency_harness.h — Precision latency measurement for voice pipelines.
 *
 * Measures every stage of the TTS/STT pipeline with nanosecond precision
 * using mach_absolute_time (Apple Silicon hardware timer, ~41.67ns resolution).
 *
 * Tracked checkpoints:
 *   1. TTFT    — Time to First Token (Claude API response start)
 *   2. TTFS    — Time to First Sentence (sentence buffer flush)
 *   3. TTFA    — Time to First Audio (first PCM chunk from TTS)
 *   4. TTFP    — Time to First Playback (first samples hit CoreAudio)
 *   5. E2E     — End-to-end: VAD speech_end → first audio playback
 *   6. RTF     — Real-time factor: audio_duration / generation_time
 *   7. Step    — Per-step TTS latency
 *
 * All metrics are accumulated into histograms for P50/P95/P99 reporting.
 *
 * Golden signals:
 * ┌──────────┬──────────┬──────────┬──────────────────────────────┐
 * │ Metric   │ P50 Goal │ P99 Goal │ Why                          │
 * ├──────────┼──────────┼──────────┼──────────────────────────────┤
 * │ TTFT     │ < 200ms  │ < 400ms  │ LLM streaming responsiveness │
 * │ TTFS     │ < 300ms  │ < 600ms  │ Sentence accumulation        │
 * │ TTFA     │ < 100ms  │ < 250ms  │ TTS model cold/warm start    │
 * │ TTFP     │ < 150ms  │ < 350ms  │ Full audio pipeline          │
 * │ E2E      │ < 500ms  │ < 900ms  │ Human-perceived turn latency │
 * │ RTF      │ < 0.3x   │ < 0.5x   │ Must be faster than realtime │
 * │ Step     │ < 5ms    │ < 10ms   │ Per-frame TTS generation     │
 * └──────────┴──────────┴──────────┴──────────────────────────────┘
 */

#ifndef LATENCY_HARNESS_H
#define LATENCY_HARNESS_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define LATENCY_MAX_SAMPLES 10000
#define LATENCY_N_METRICS   7

typedef enum {
    LAT_TTFT = 0,   /* Time to First Token */
    LAT_TTFS,       /* Time to First Sentence */
    LAT_TTFA,       /* Time to First Audio */
    LAT_TTFP,       /* Time to First Playback */
    LAT_E2E,        /* End-to-end turn latency */
    LAT_RTF,        /* Real-time factor (stored as ms equivalent) */
    LAT_STEP,       /* Per-step TTS latency */
} LatencyMetric;

typedef struct {
    double p50;
    double p95;
    double p99;
    double mean;
    double min;
    double max;
    double stddev;
    int    n;
} LatencyStats;

typedef struct LatencyHarness LatencyHarness;

/**
 * Create a latency harness. Allocates internal storage for up to
 * LATENCY_MAX_SAMPLES measurements per metric.
 */
LatencyHarness *latency_create(void);

void latency_destroy(LatencyHarness *h);

/**
 * Start a timer for the given metric. Returns a timestamp token
 * to pass to latency_stop().
 */
uint64_t latency_start(LatencyHarness *h, LatencyMetric metric);

/**
 * Stop a timer and record the elapsed time. Uses the token from
 * latency_start() to compute elapsed nanoseconds.
 */
void latency_stop(LatencyHarness *h, LatencyMetric metric, uint64_t start_token);

/**
 * Record an externally-measured latency value in milliseconds.
 * Use this for metrics measured outside the harness (e.g., RTF).
 */
void latency_record_ms(LatencyHarness *h, LatencyMetric metric, double ms);

/**
 * Compute statistics for a given metric.
 */
LatencyStats latency_stats(const LatencyHarness *h, LatencyMetric metric);

/**
 * Reset all measurements.
 */
void latency_reset(LatencyHarness *h);

/**
 * Print a formatted latency report for all metrics.
 */
void latency_print_report(const LatencyHarness *h, const char *label);

/**
 * Check if all metrics meet the golden signal thresholds.
 * Returns 0 if all pass, nonzero count of failures.
 */
int latency_check_golden(const LatencyHarness *h);

/**
 * Get the current timestamp (mach_absolute_time on macOS).
 * Useful for external timing.
 */
uint64_t latency_now(void);

/**
 * Convert a raw timestamp delta to milliseconds.
 */
double latency_elapsed_ms(uint64_t start, uint64_t end);

#ifdef __cplusplus
}
#endif

#endif /* LATENCY_HARNESS_H */
