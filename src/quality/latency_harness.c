/**
 * latency_harness.c — Precision latency measurement with mach_absolute_time.
 *
 * Uses the ARM performance counter (via mach_absolute_time) for nanosecond
 * precision. On Apple Silicon, this has ~41.67ns granularity (24 MHz timer).
 *
 * Percentiles are computed via quickselect (O(n) expected) to avoid
 * the cost of full sorting.
 */

#include "latency_harness.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <mach/mach_time.h>

/* ── Harness Structure ────────────────────────────────── */

typedef struct {
    double samples[LATENCY_MAX_SAMPLES];
    int count;
} MetricSamples;

struct LatencyHarness {
    MetricSamples metrics[LATENCY_N_METRICS];
    mach_timebase_info_data_t timebase;
};

/* ── Timing Utilities ─────────────────────────────────── */

static double ticks_to_ms(const LatencyHarness *h, uint64_t ticks)
{
    return (double)ticks * (double)h->timebase.numer /
           ((double)h->timebase.denom * 1e6);
}

uint64_t latency_now(void)
{
    return mach_absolute_time();
}

double latency_elapsed_ms(uint64_t start, uint64_t end)
{
    mach_timebase_info_data_t tb;
    mach_timebase_info(&tb);
    return (double)(end - start) * (double)tb.numer / ((double)tb.denom * 1e6);
}

/* ── Quickselect for percentiles ──────────────────────── */

static int partition(double *arr, int lo, int hi)
{
    double pivot = arr[hi];
    int i = lo;
    for (int j = lo; j < hi; j++) {
        if (arr[j] <= pivot) {
            double tmp = arr[i]; arr[i] = arr[j]; arr[j] = tmp;
            i++;
        }
    }
    double tmp = arr[i]; arr[i] = arr[hi]; arr[hi] = tmp;
    return i;
}

static double quickselect(double *arr, int n, int k)
{
    if (n <= 0) return 0.0;
    if (k >= n) k = n - 1;

    /* Work on a copy */
    double *buf = (double *)malloc((size_t)n * sizeof(double));
    memcpy(buf, arr, (size_t)n * sizeof(double));

    int lo = 0, hi = n - 1;
    while (lo < hi) {
        int p = partition(buf, lo, hi);
        if (p == k) break;
        else if (p < k) lo = p + 1;
        else hi = p - 1;
    }
    double result = buf[k];
    free(buf);
    return result;
}

/* ── Public API ───────────────────────────────────────── */

LatencyHarness *latency_create(void)
{
    LatencyHarness *h = (LatencyHarness *)calloc(1, sizeof(LatencyHarness));
    mach_timebase_info(&h->timebase);
    return h;
}

void latency_destroy(LatencyHarness *h)
{
    free(h);
}

uint64_t latency_start(LatencyHarness *h, LatencyMetric metric)
{
    (void)h; (void)metric;
    return mach_absolute_time();
}

void latency_stop(LatencyHarness *h, LatencyMetric metric, uint64_t start_token)
{
    uint64_t end = mach_absolute_time();
    double ms = ticks_to_ms(h, end - start_token);
    latency_record_ms(h, metric, ms);
}

void latency_record_ms(LatencyHarness *h, LatencyMetric metric, double ms)
{
    if (!h || metric < 0 || metric >= LATENCY_N_METRICS) return;

    MetricSamples *m = &h->metrics[metric];
    if (m->count < LATENCY_MAX_SAMPLES) {
        m->samples[m->count++] = ms;
    } else {
        /* Ring-buffer overwrite (keep most recent) */
        memmove(m->samples, m->samples + 1,
                (LATENCY_MAX_SAMPLES - 1) * sizeof(double));
        m->samples[LATENCY_MAX_SAMPLES - 1] = ms;
    }
}

LatencyStats latency_stats(const LatencyHarness *h, LatencyMetric metric)
{
    LatencyStats st = {0};
    if (!h || metric < 0 || metric >= LATENCY_N_METRICS) return st;

    const MetricSamples *m = &h->metrics[metric];
    st.n = m->count;
    if (st.n == 0) return st;

    /* Min, max, mean */
    double sum = 0;
    st.min = m->samples[0];
    st.max = m->samples[0];
    for (int i = 0; i < st.n; i++) {
        sum += m->samples[i];
        if (m->samples[i] < st.min) st.min = m->samples[i];
        if (m->samples[i] > st.max) st.max = m->samples[i];
    }
    st.mean = sum / (double)st.n;

    /* Stddev */
    double var = 0;
    for (int i = 0; i < st.n; i++) {
        double d = m->samples[i] - st.mean;
        var += d * d;
    }
    st.stddev = sqrt(var / (double)st.n);

    /* Percentiles via quickselect */
    st.p50 = quickselect((double *)m->samples, st.n, st.n / 2);
    st.p95 = quickselect((double *)m->samples, st.n, (int)((double)st.n * 0.95));
    st.p99 = quickselect((double *)m->samples, st.n, (int)((double)st.n * 0.99));

    return st;
}

void latency_reset(LatencyHarness *h)
{
    if (!h) return;
    for (int i = 0; i < LATENCY_N_METRICS; i++) {
        h->metrics[i].count = 0;
    }
}

/* ── Golden Signal Thresholds (ms) ────────────────────── */

static const struct {
    const char *name;
    double p50_goal;
    double p99_goal;
} golden_thresholds[LATENCY_N_METRICS] = {
    [LAT_TTFT] = { "TTFT (Time to First Token)",    200.0,  400.0 },
    [LAT_TTFS] = { "TTFS (Time to First Sentence)",  300.0,  600.0 },
    [LAT_TTFA] = { "TTFA (Time to First Audio)",     100.0,  250.0 },
    [LAT_TTFP] = { "TTFP (Time to First Playback)",  150.0,  350.0 },
    [LAT_E2E]  = { "E2E  (End-to-End Turn)",         500.0,  900.0 },
    [LAT_RTF]  = { "RTF  (Real-Time Factor)",           0.3,    0.5 },
    [LAT_STEP] = { "Step (Per-Frame TTS)",              5.0,   10.0 },
};

int latency_check_golden(const LatencyHarness *h)
{
    int failures = 0;
    for (int i = 0; i < LATENCY_N_METRICS; i++) {
        LatencyStats st = latency_stats(h, (LatencyMetric)i);
        if (st.n == 0) continue;

        int p50_fail = st.p50 > golden_thresholds[i].p50_goal;
        int p99_fail = st.p99 > golden_thresholds[i].p99_goal;

        if (p50_fail || p99_fail) failures++;
    }
    return failures;
}

/* ── Report ───────────────────────────────────────────── */

void latency_print_report(const LatencyHarness *h, const char *label)
{
    fprintf(stderr,
        "\n╔══════════════════════════════════════════════════════════════════════╗\n"
        "║  Latency Report: %-48s  ║\n"
        "╠══════════════════════════════════════════════════════════════════════╣\n"
        "║  Metric                       │  P50     P95     P99  │  Goal P50  ║\n"
        "╠───────────────────────────────┼────────────────────────┼────────────╣\n",
        label ? label : "");

    for (int i = 0; i < LATENCY_N_METRICS; i++) {
        LatencyStats st = latency_stats(h, (LatencyMetric)i);
        if (st.n == 0) {
            fprintf(stderr, "║  %-29s │     (no data)          │            ║\n",
                    golden_thresholds[i].name);
            continue;
        }

        const char *unit = (i == LAT_RTF) ? "x " : "ms";
        int p50_ok = st.p50 <= golden_thresholds[i].p50_goal;
        int p99_ok = st.p99 <= golden_thresholds[i].p99_goal;

        fprintf(stderr,
            "║  %-29s │ %6.1f  %6.1f  %6.1f%s │ %s%6.1f%s   ║\n",
            golden_thresholds[i].name,
            st.p50, st.p95, st.p99, unit,
            p50_ok ? " " : "!",
            golden_thresholds[i].p50_goal,
            p99_ok ? " " : "!");
    }

    int failures = latency_check_golden(h);
    fprintf(stderr,
        "╠══════════════════════════════════════════════════════════════════════╣\n"
        "║  Status: %s (%d metric%s out of spec)                         ║\n"
        "╚══════════════════════════════════════════════════════════════════════╝\n\n",
        failures == 0 ? "ALL GOLDEN " : "NEEDS WORK",
        failures, failures == 1 ? "" : "s");
}
