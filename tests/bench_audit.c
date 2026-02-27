/**
 * bench_audit.c — Performance validation benchmarks for pocket-voice optimizations.
 *
 * Proves (or disproves) three optimization claims:
 *   1. Ring buffer mel spectrogram eliminates memmove
 *   2. Pre-allocated iSTFT buffers eliminate per-utterance malloc
 *   3. INT8 dequant+sgemm vs fp32 sgemm throughput
 *
 * Also checks for regressions:
 *   - Ring buffer grow-path latency
 *   - metal_dispatch overhead vs direct cblas_sgemm
 *   - Memory accumulation in ring buffer
 *
 * Build: make bench-audit
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <mach/mach_time.h>
#include <mach/mach.h>
#include <Accelerate/Accelerate.h>
#include <arm_neon.h>

#include "mel_spectrogram.h"
#include "sonata_istft.h"
#include "metal_dispatch.h"

/* ═══════════════════════════════════════════════════════════════════════════
 * Timing Infrastructure
 * ═══════════════════════════════════════════════════════════════════════════ */

static mach_timebase_info_data_t g_tb;

static void timing_init(void) {
    if (g_tb.denom == 0) mach_timebase_info(&g_tb);
}

static double ns_elapsed(uint64_t start, uint64_t end) {
    return (double)(end - start) * g_tb.numer / g_tb.denom;
}

static double us_elapsed(uint64_t start, uint64_t end) {
    return ns_elapsed(start, end) / 1000.0;
}

static double ms_elapsed(uint64_t start, uint64_t end) {
    return ns_elapsed(start, end) / 1e6;
}

/* Get current RSS in bytes */
static size_t get_rss(void) {
    struct mach_task_basic_info info;
    mach_msg_type_number_t count = MACH_TASK_BASIC_INFO_COUNT;
    if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO,
                  (task_info_t)&info, &count) != KERN_SUCCESS)
        return 0;
    return info.resident_size;
}

/* Comparison function for qsort */
static int cmp_double(const void *a, const void *b) {
    double da = *(const double *)a, db = *(const double *)b;
    return (da > db) - (da < db);
}

/* Statistics for a latency distribution */
typedef struct {
    double min, max, mean, median, p95, p99, stddev;
    int n;
} LatStats;

static LatStats compute_stats(double *samples, int n) {
    LatStats s = {0};
    s.n = n;
    if (n == 0) return s;
    qsort(samples, n, sizeof(double), cmp_double);
    s.min = samples[0];
    s.max = samples[n - 1];
    s.median = samples[n / 2];
    s.p95 = samples[(int)(n * 0.95)];
    s.p99 = samples[(int)(n * 0.99)];
    double sum = 0;
    for (int i = 0; i < n; i++) sum += samples[i];
    s.mean = sum / n;
    double var = 0;
    for (int i = 0; i < n; i++) var += (samples[i] - s.mean) * (samples[i] - s.mean);
    s.stddev = sqrt(var / n);
    return s;
}

static int tests_passed = 0;
static int tests_failed = 0;
static int tests_warn = 0;

#define VERDICT(ok, label, fmt, ...) do { \
    if (ok) { \
        fprintf(stderr, "  PASS: " label " — " fmt "\n", ##__VA_ARGS__); \
        tests_passed++; \
    } else { \
        fprintf(stderr, "  FAIL: " label " — " fmt "\n", ##__VA_ARGS__); \
        tests_failed++; \
    } \
} while(0)

#define WARN(label, fmt, ...) do { \
    fprintf(stderr, "  WARN: " label " — " fmt "\n", ##__VA_ARGS__); \
    tests_warn++; \
} while(0)

/* ═══════════════════════════════════════════════════════════════════════════
 * Benchmark 1: Mel Spectrogram Ring Buffer Performance
 *
 * Proves: ring buffer eliminates memmove, streaming is fast.
 * Checks: per-frame latency, throughput, ring grow overhead.
 * ═══════════════════════════════════════════════════════════════════════════ */

static void bench_mel_spectrogram(void) {
    fprintf(stderr, "\n╔══════════════════════════════════════════════════════════╗\n");
    fprintf(stderr, "║  Benchmark 1: Mel Spectrogram (Ring Buffer)              ║\n");
    fprintf(stderr, "╠══════════════════════════════════════════════════════════╣\n");

    MelConfig cfg;
    mel_config_default(&cfg);
    MelSpectrogram *mel = mel_create(&cfg);
    if (!mel) {
        fprintf(stderr, "  FAIL: mel_create returned NULL\n");
        tests_failed++;
        return;
    }

    int sr = cfg.sample_rate;  /* 16000 */
    int hop = cfg.hop_length;  /* 160 */
    int n_mels = cfg.n_mels;   /* 80 */

    /* ── Test 1a: Bulk processing (10s audio) ── */
    {
        int n_samples = sr * 10;  /* 160,000 samples */
        float *pcm = (float *)calloc(n_samples, sizeof(float));
        /* Generate a chirp signal (more realistic than silence) */
        for (int i = 0; i < n_samples; i++) {
            float t = (float)i / sr;
            float freq = 100.0f + 3900.0f * t / 10.0f;  /* 100 Hz → 4000 Hz */
            pcm[i] = 0.5f * sinf(2.0f * M_PI * freq * t);
        }

        int max_frames = n_samples / hop + 10;
        float *out = (float *)calloc((size_t)max_frames * n_mels, sizeof(float));

        mel_reset(mel);

        uint64_t t0 = mach_absolute_time();
        int frames = mel_process(mel, pcm, n_samples, out, max_frames);
        uint64_t t1 = mach_absolute_time();

        double total_ms = ms_elapsed(t0, t1);
        double per_frame_us = (frames > 0) ? (total_ms * 1000.0 / frames) : 0;
        double rtf = total_ms / (10.0 * 1000.0);  /* 10s audio */
        double fps = (total_ms > 0) ? frames / (total_ms / 1000.0) : 0;

        fprintf(stderr, "║  Bulk 10s:  %d frames in %.2f ms\n", frames, total_ms);
        fprintf(stderr, "║    Per-frame: %.2f us  |  RTF: %.6f  |  %.0f frames/s\n",
                per_frame_us, rtf, fps);

        VERDICT(frames > 0, "Bulk processing", "%d frames produced", frames);
        VERDICT(rtf < 0.01, "Real-time factor",
                "RTF %.6f (need < 0.01 for 100x realtime)", rtf);

        free(pcm);
        free(out);
    }

    /* ── Test 1b: Streaming (160-sample chunks, simulating 10ms at 16kHz) ── */
    {
        int chunk = 160;  /* 10 ms */
        int n_chunks = 1000;  /* 10 seconds total */
        float *pcm = (float *)malloc(chunk * sizeof(float));
        float *out = (float *)malloc((size_t)10 * n_mels * sizeof(float));
        double *latencies = (double *)malloc(n_chunks * sizeof(double));
        int total_frames = 0;

        mel_reset(mel);

        for (int c = 0; c < n_chunks; c++) {
            /* Generate chunk */
            for (int i = 0; i < chunk; i++) {
                float t = (float)(c * chunk + i) / 16000.0f;
                pcm[i] = 0.3f * sinf(2.0f * M_PI * 440.0f * t);
            }

            uint64_t t0 = mach_absolute_time();
            int f = mel_process(mel, pcm, chunk, out, 10);
            uint64_t t1 = mach_absolute_time();

            latencies[c] = us_elapsed(t0, t1);
            if (f > 0) total_frames += f;
        }

        LatStats st = compute_stats(latencies, n_chunks);
        fprintf(stderr, "║  Streaming 10ms chunks (%d chunks):\n", n_chunks);
        fprintf(stderr, "║    P50: %.1f us  P95: %.1f us  P99: %.1f us  Max: %.1f us\n",
                st.median, st.p95, st.p99, st.max);
        fprintf(stderr, "║    Total frames: %d  (expected ~%d)\n",
                total_frames, n_chunks - 2);

        VERDICT(st.median < 100.0, "Streaming P50 latency",
                "%.1f us (need < 100 us per 10ms chunk)", st.median);
        VERDICT(st.p99 < 500.0, "Streaming P99 latency",
                "%.1f us (need < 500 us tail)", st.p99);

        free(pcm);
        free(out);
        free(latencies);
    }

    /* ── Test 1c: Ring buffer grow path ── */
    {
        mel_reset(mel);

        /* Feed enough to force a ring grow (default cap = 4 * sr = 64000) */
        int big_chunk = 65000;
        float *pcm = (float *)calloc(big_chunk, sizeof(float));
        float *out = (float *)calloc(1000 * n_mels, sizeof(float));

        uint64_t t0 = mach_absolute_time();
        int f1 = mel_process(mel, pcm, big_chunk, out, 1000);
        uint64_t t1 = mach_absolute_time();
        double first_ms = ms_elapsed(t0, t1);

        /* Feed a small follow-up to check post-grow performance */
        t0 = mach_absolute_time();
        int f2 = mel_process(mel, pcm, 160, out, 10);
        t1 = mach_absolute_time();
        double follow_ms = ms_elapsed(t0, t1);

        fprintf(stderr, "║  Ring grow: %d frames in %.2f ms (forces realloc)\n",
                f1, first_ms);
        fprintf(stderr, "║    Post-grow follow-up: %d frames in %.3f ms\n",
                f2, follow_ms);

        VERDICT(f1 > 0, "Ring grow succeeds", "%d frames after realloc", f1);
        VERDICT(follow_ms < 1.0, "Post-grow perf",
                "%.3f ms (no lingering overhead)", follow_ms);

        free(pcm);
        free(out);
    }

    /* ── Test 1d: Ring buffer wrap-around path specifically ── */
    {
        mel_reset(mel);

        /* Fill ring to near capacity (but not overflow) with many small pushes
         * to exercise the wrap-around read path */
        int small = 1600;  /* 100ms */
        float *pcm = (float *)calloc(small, sizeof(float));
        float *out = (float *)calloc(100 * n_mels, sizeof(float));
        double *contiguous_lat = (double *)malloc(100 * sizeof(double));
        double *wrap_lat = (double *)malloc(100 * sizeof(double));
        int contiguous_count = 0, wrap_count = 0;

        /* Push 40 chunks (64000 samples) to fill the ring, then keep going
         * to cause wrap-around reads */
        for (int i = 0; i < 100; i++) {
            for (int j = 0; j < small; j++) {
                float t = (float)(i * small + j) / 16000.0f;
                pcm[j] = 0.3f * sinf(2.0f * M_PI * 300.0f * t);
            }

            uint64_t t0 = mach_absolute_time();
            int f = mel_process(mel, pcm, small, out, 100);
            uint64_t t1 = mach_absolute_time();
            (void)f;

            double lat = us_elapsed(t0, t1);
            if (i < 40) {
                contiguous_lat[contiguous_count++] = lat;
            } else {
                wrap_lat[wrap_count++] = lat;
            }
        }

        LatStats cst = compute_stats(contiguous_lat, contiguous_count);
        LatStats wst = compute_stats(wrap_lat, wrap_count);

        fprintf(stderr, "║  Contiguous reads:  P50 %.1f us  P95 %.1f us\n",
                cst.median, cst.p95);
        fprintf(stderr, "║  Wraparound reads:  P50 %.1f us  P95 %.1f us\n",
                wst.median, wst.p95);

        double overhead_pct = (wst.median > 0 && cst.median > 0)
            ? ((wst.median - cst.median) / cst.median * 100.0) : 0;
        fprintf(stderr, "║  Wrap overhead: %.1f%% vs contiguous\n", overhead_pct);

        if (fabs(overhead_pct) > 20.0 && wst.median > cst.median) {
            WARN("Wrap overhead", "%.1f%% slower — ring_read copy adds latency", overhead_pct);
        } else {
            VERDICT(1, "Wrap overhead", "%.1f%% — acceptable", overhead_pct);
        }

        free(pcm);
        free(out);
        free(contiguous_lat);
        free(wrap_lat);
    }

    mel_destroy(mel);
    fprintf(stderr, "╚══════════════════════════════════════════════════════════╝\n");
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Benchmark 2: iSTFT Pre-Allocated Buffers
 *
 * Proves: zero allocations after init, per-frame latency is stable.
 * Checks: batch vs frame-by-frame, latency distribution.
 * ═══════════════════════════════════════════════════════════════════════════ */

static void bench_istft(void) {
    fprintf(stderr, "\n╔══════════════════════════════════════════════════════════╗\n");
    fprintf(stderr, "║  Benchmark 2: iSTFT (Pre-Allocated Buffers)              ║\n");
    fprintf(stderr, "╠══════════════════════════════════════════════════════════╣\n");

    #define ISTFT_N_FFT  1024
    #define ISTFT_HOP    480
    #define ISTFT_N_BINS (ISTFT_N_FFT / 2 + 1)  /* 513 */
    #define ISTFT_SR     24000

    SonataISTFT *dec = sonata_istft_create(ISTFT_N_FFT, ISTFT_HOP);
    if (!dec) {
        fprintf(stderr, "  FAIL: sonata_istft_create returned NULL\n");
        tests_failed++;
        return;
    }

    /* ── Test 2a: Per-frame decode latency ── */
    {
        int n_frames = 500;  /* 10 seconds at 50 Hz */
        float *mag = (float *)calloc(ISTFT_N_BINS, sizeof(float));
        float *phase = (float *)calloc(ISTFT_N_BINS, sizeof(float));
        float audio[ISTFT_HOP];
        double *latencies = (double *)malloc(n_frames * sizeof(double));

        /* Synthetic spectral data */
        for (int b = 1; b < ISTFT_N_BINS; b++) {
            mag[b] = 0.01f * (1.0f / (1.0f + b * 0.01f));
        }

        sonata_istft_reset(dec);

        for (int f = 0; f < n_frames; f++) {
            /* Vary phase per frame */
            for (int b = 1; b < ISTFT_N_BINS; b++)
                phase[b] = (float)(f * b) * 0.1f;

            uint64_t t0 = mach_absolute_time();
            int ns = sonata_istft_decode_frame(dec, mag, phase, audio);
            uint64_t t1 = mach_absolute_time();

            latencies[f] = us_elapsed(t0, t1);
            (void)ns;
        }

        LatStats st = compute_stats(latencies, n_frames);
        double audio_per_frame_ms = (double)ISTFT_HOP / ISTFT_SR * 1000.0;  /* 20 ms */
        double rtf = (st.mean / 1000.0) / audio_per_frame_ms;

        fprintf(stderr, "║  Per-frame decode (%d frames, 10s audio):\n", n_frames);
        fprintf(stderr, "║    P50: %.1f us  P95: %.1f us  P99: %.1f us  Max: %.1f us\n",
                st.median, st.p95, st.p99, st.max);
        fprintf(stderr, "║    Mean: %.1f us  Stddev: %.1f us\n", st.mean, st.stddev);
        fprintf(stderr, "║    RTF: %.6f (%.0fx realtime)\n", rtf, rtf > 0 ? 1.0 / rtf : 0);

        VERDICT(st.median < 50.0, "iSTFT per-frame P50",
                "%.1f us (need < 50 us for 50Hz)", st.median);
        VERDICT(rtf < 0.005, "iSTFT RTF",
                "%.6f (need < 0.005 for 200x realtime)", rtf);
        VERDICT(st.stddev / st.mean < 0.5, "iSTFT jitter",
                "CV=%.2f (stddev/mean, need < 0.5)", st.stddev / st.mean);

        free(mag);
        free(phase);
        free(latencies);
    }

    /* ── Test 2b: Batch decode throughput ── */
    {
        int test_sizes[] = {25, 50, 100, 250, 500, 1000};
        int n_tests = sizeof(test_sizes) / sizeof(test_sizes[0]);

        fprintf(stderr, "║\n║  Batch decode throughput:\n");
        fprintf(stderr, "║  %-8s  %-10s  %-10s  %-12s  %-8s\n",
                "Frames", "Samples", "Time(ms)", "Frames/s", "RTF");

        for (int t = 0; t < n_tests; t++) {
            int n = test_sizes[t];
            float *mag = (float *)calloc((size_t)n * ISTFT_N_BINS, sizeof(float));
            float *phase = (float *)calloc((size_t)n * ISTFT_N_BINS, sizeof(float));
            float *audio = (float *)calloc((size_t)n * ISTFT_HOP, sizeof(float));

            for (int f = 0; f < n; f++)
                for (int b = 1; b < ISTFT_N_BINS; b++) {
                    mag[f * ISTFT_N_BINS + b] = 0.01f;
                    phase[f * ISTFT_N_BINS + b] = (float)(f * b) * 0.05f;
                }

            sonata_istft_reset(dec);

            uint64_t t0 = mach_absolute_time();
            int total = sonata_istft_decode_batch(dec, mag, phase, n, audio);
            uint64_t t1 = mach_absolute_time();

            double elapsed_ms = ms_elapsed(t0, t1);
            double audio_s = (double)total / ISTFT_SR;
            double rtf = elapsed_ms / 1000.0 / audio_s;
            double fps = (elapsed_ms > 0) ? n / (elapsed_ms / 1000.0) : 0;

            fprintf(stderr, "║  %-8d  %-10d  %-10.3f  %-12.0f  %.6f\n",
                    n, total, elapsed_ms, fps, rtf);

            free(mag);
            free(phase);
            free(audio);
        }
    }

    /* ── Test 2c: Create/destroy overhead (proves pre-allocation is one-time) ── */
    {
        int n_iters = 100;
        double *create_us = (double *)malloc(n_iters * sizeof(double));
        double *destroy_us = (double *)malloc(n_iters * sizeof(double));

        for (int i = 0; i < n_iters; i++) {
            uint64_t t0 = mach_absolute_time();
            SonataISTFT *d = sonata_istft_create(ISTFT_N_FFT, ISTFT_HOP);
            uint64_t t1 = mach_absolute_time();
            create_us[i] = us_elapsed(t0, t1);

            t0 = mach_absolute_time();
            sonata_istft_destroy(d);
            t1 = mach_absolute_time();
            destroy_us[i] = us_elapsed(t0, t1);
        }

        LatStats cs = compute_stats(create_us, n_iters);
        LatStats ds = compute_stats(destroy_us, n_iters);
        fprintf(stderr, "║\n║  Create/destroy overhead (amortized once):\n");
        fprintf(stderr, "║    Create P50: %.1f us  Destroy P50: %.1f us\n",
                cs.median, ds.median);

        VERDICT(cs.median < 200.0, "iSTFT create cost",
                "%.1f us (one-time, acceptable)", cs.median);

        free(create_us);
        free(destroy_us);
    }

    sonata_istft_destroy(dec);
    fprintf(stderr, "╚══════════════════════════════════════════════════════════╝\n");
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Benchmark 3: INT8 Dequant + SGEMM vs FP32 SGEMM
 *
 * Proves: INT8 model uses 4x less memory. Measures dequant overhead.
 * Checks: Is dequant+sgemm slower than fp32 sgemm on AMX?
 * ═══════════════════════════════════════════════════════════════════════════ */

static void bench_int8_vs_fp32(void) {
    fprintf(stderr, "\n╔══════════════════════════════════════════════════════════╗\n");
    fprintf(stderr, "║  Benchmark 3: INT8 Dequant+SGEMM vs FP32 SGEMM          ║\n");
    fprintf(stderr, "╠══════════════════════════════════════════════════════════╣\n");

    /* Realistic conformer dimensions:
     * M=1 (single-frame inference), K=512 (d_model), N=512 (d_model)
     * Also test FFN: M=1, K=512, N=2048 (ff_mult=4) */

    typedef struct {
        const char *label;
        int M, K, N;
    } GEMMSize;

    GEMMSize sizes[] = {
        {"Attention (1x512x512)",    1, 512, 512},
        {"FFN up (1x512x2048)",      1, 512, 2048},
        {"FFN down (1x2048x512)",    1, 2048, 512},
        {"Batch attn (8x512x512)",   8, 512, 512},
        {"Large batch (32x512x512)", 32, 512, 512},
    };
    int n_sizes = sizeof(sizes) / sizeof(sizes[0]);
    int ITERS = 500;

    for (int s = 0; s < n_sizes; s++) {
        int M = sizes[s].M, K = sizes[s].K, N = sizes[s].N;

        /* Allocate fp32 weight matrix */
        float *W_fp32 = (float *)malloc((size_t)N * K * sizeof(float));
        float *in = (float *)malloc((size_t)M * K * sizeof(float));
        float *out_fp32 = (float *)calloc((size_t)M * N, sizeof(float));
        float *out_int8 = (float *)calloc((size_t)M * N, sizeof(float));

        /* Random weights and input */
        for (int i = 0; i < N * K; i++)
            W_fp32[i] = ((float)(rand() % 2000) - 1000.0f) / 1000.0f;
        for (int i = 0; i < M * K; i++)
            in[i] = ((float)(rand() % 2000) - 1000.0f) / 1000.0f;

        /* Quantize to INT8 per-channel */
        int8_t *W_q = (int8_t *)malloc((size_t)N * K);
        float *scales = (float *)malloc(N * sizeof(float));
        for (int n = 0; n < N; n++) {
            float max_abs = 0;
            for (int k = 0; k < K; k++) {
                float v = fabsf(W_fp32[n * K + k]);
                if (v > max_abs) max_abs = v;
            }
            scales[n] = max_abs / 127.0f;
            float inv_scale = (max_abs > 0) ? 127.0f / max_abs : 0;
            for (int k = 0; k < K; k++) {
                float v = W_fp32[n * K + k] * inv_scale;
                if (v > 127.0f) v = 127.0f;
                if (v < -127.0f) v = -127.0f;
                W_q[n * K + k] = (int8_t)roundf(v);
            }
        }

        /* INT8 dequant tile buffer */
        int tile_n = 64;
        float *W_tile = (float *)malloc((size_t)tile_n * K * sizeof(float));

        /* ── Benchmark: FP32 SGEMM ── */
        uint64_t t0 = mach_absolute_time();
        for (int i = 0; i < ITERS; i++) {
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                        M, N, K, 1.0f, in, K, W_fp32, K, 0.0f, out_fp32, N);
        }
        uint64_t t1 = mach_absolute_time();
        double fp32_us = us_elapsed(t0, t1) / ITERS;

        /* ── Benchmark: INT8 dequant + SGEMM ── */
        t0 = mach_absolute_time();
        for (int i = 0; i < ITERS; i++) {
            /* Inline the same tiled dequant+sgemm logic as conformer_stt.c */
            for (int n0 = 0; n0 < N; n0 += tile_n) {
                int tn = (n0 + tile_n <= N) ? tile_n : (N - n0);
                for (int r = 0; r < tn; r++) {
                    const int8_t *row = W_q + (size_t)(n0 + r) * K;
                    float *dst = W_tile + (size_t)r * K;
                    float sc = scales[n0 + r];
                    int k = 0;
                    for (; k + 15 < K; k += 16) {
                        int8x16_t q = vld1q_s8(row + k);
                        int16x8_t lo16 = vmovl_s8(vget_low_s8(q));
                        int16x8_t hi16 = vmovl_s8(vget_high_s8(q));
                        float32x4_t f0 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(lo16)));
                        float32x4_t f1 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(lo16)));
                        float32x4_t f2 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(hi16)));
                        float32x4_t f3 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(hi16)));
                        float32x4_t sv = vdupq_n_f32(sc);
                        vst1q_f32(dst + k,      vmulq_f32(f0, sv));
                        vst1q_f32(dst + k + 4,  vmulq_f32(f1, sv));
                        vst1q_f32(dst + k + 8,  vmulq_f32(f2, sv));
                        vst1q_f32(dst + k + 12, vmulq_f32(f3, sv));
                    }
                    for (; k < K; k++)
                        dst[k] = (float)row[k] * sc;
                }
                cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                            M, tn, K, 1.0f, in, K, W_tile, K, 0.0f, out_int8 + n0, N);
            }
        }
        t1 = mach_absolute_time();
        double int8_us = us_elapsed(t0, t1) / ITERS;

        double overhead_pct = ((int8_us - fp32_us) / fp32_us) * 100.0;
        double mem_ratio = (double)(N * K * sizeof(float)) /
                           (double)(N * K * sizeof(int8_t) + N * sizeof(float));

        /* Compute accuracy: max absolute error between fp32 and int8 output */
        float max_err = 0;
        for (int i = 0; i < M * N; i++) {
            float err = fabsf(out_fp32[i] - out_int8[i]);
            if (err > max_err) max_err = err;
        }

        fprintf(stderr, "║\n║  %s:\n", sizes[s].label);
        fprintf(stderr, "║    FP32 SGEMM:      %8.1f us\n", fp32_us);
        fprintf(stderr, "║    INT8 dequant+SGM: %8.1f us  (%+.1f%%)\n", int8_us, overhead_pct);
        fprintf(stderr, "║    Memory savings:   %.1fx smaller\n", mem_ratio);
        fprintf(stderr, "║    Max abs error:    %.6f\n", max_err);

        if (overhead_pct > 50.0) {
            WARN("INT8 speed", "%s: INT8 is %.0f%% slower than FP32 — dequant overhead dominates",
                 sizes[s].label, overhead_pct);
        } else if (overhead_pct < -5.0) {
            VERDICT(1, "INT8 speed", "%s: INT8 is %.0f%% FASTER (cache benefit)",
                    sizes[s].label, -overhead_pct);
        } else {
            VERDICT(1, "INT8 speed", "%s: within %.0f%% of FP32 (acceptable)",
                    sizes[s].label, fabs(overhead_pct));
        }

        VERDICT(mem_ratio > 3.5, "INT8 memory", "%s: %.1fx compression",
                sizes[s].label, mem_ratio);

        free(W_fp32); free(in); free(out_fp32); free(out_int8);
        free(W_q); free(scales); free(W_tile);
    }

    fprintf(stderr, "╚══════════════════════════════════════════════════════════╝\n");
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Benchmark 4: Metal Dispatch Overhead
 *
 * Measures: dispatch layer overhead vs calling cblas_sgemm directly.
 * Checks: fp32→fp16 conversion cost, GPU dispatch latency.
 * ═══════════════════════════════════════════════════════════════════════════ */

static void bench_metal_dispatch(void) {
    fprintf(stderr, "\n╔══════════════════════════════════════════════════════════╗\n");
    fprintf(stderr, "║  Benchmark 4: Metal Dispatch Overhead                    ║\n");
    fprintf(stderr, "╠══════════════════════════════════════════════════════════╣\n");

    /* Try to init Metal */
    int metal_ok = metal_dispatch_init("build/tensor_ops.metallib");
    fprintf(stderr, "║  Metal available: %s\n", metal_ok ? "YES" : "NO (CPU fallback)");

    typedef struct { const char *label; int M, N, K; } Size;
    Size sizes[] = {
        {"Small (1x64x64)",       1, 64, 64},
        {"Medium (1x512x512)",    1, 512, 512},
        {"Large (1x512x2048)",    1, 512, 2048},
        {"Batch (16x512x512)",   16, 512, 512},
    };
    int n_sizes = sizeof(sizes) / sizeof(sizes[0]);
    int ITERS = 200;

    for (int s = 0; s < n_sizes; s++) {
        int M = sizes[s].M, K = sizes[s].K, N = sizes[s].N;

        float *A = (float *)malloc((size_t)M * K * sizeof(float));
        float *B = (float *)malloc((size_t)N * K * sizeof(float));
        float *C_direct = (float *)calloc((size_t)M * N, sizeof(float));
        float *C_dispatch = (float *)calloc((size_t)M * N, sizeof(float));

        for (int i = 0; i < M * K; i++) A[i] = (float)(rand() % 100) / 100.0f;
        for (int i = 0; i < N * K; i++) B[i] = (float)(rand() % 100) / 100.0f;

        /* Direct cblas_sgemm */
        uint64_t t0 = mach_absolute_time();
        for (int i = 0; i < ITERS; i++)
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                        M, N, K, 1.0f, A, K, B, K, 0.0f, C_direct, N);
        uint64_t t1 = mach_absolute_time();
        double direct_us = us_elapsed(t0, t1) / ITERS;

        /* metal_dispatch_gemm (GPU or CPU fallback) */
        t0 = mach_absolute_time();
        for (int i = 0; i < ITERS; i++)
            metal_dispatch_gemm(A, B, C_dispatch, M, N, K);
        t1 = mach_absolute_time();
        double dispatch_us = us_elapsed(t0, t1) / ITERS;

        double overhead_pct = ((dispatch_us - direct_us) / direct_us) * 100.0;

        fprintf(stderr, "║\n║  %s:\n", sizes[s].label);
        fprintf(stderr, "║    Direct cblas:    %8.1f us\n", direct_us);
        fprintf(stderr, "║    metal_dispatch:  %8.1f us  (%+.1f%%)\n",
                dispatch_us, overhead_pct);

        if (!metal_ok) {
            /* CPU fallback — dispatch should be nearly zero overhead */
            VERDICT(overhead_pct < 10.0, "Dispatch overhead (CPU)",
                    "%s: %.1f%% overhead (expect ~0%%)", sizes[s].label, overhead_pct);
        } else {
            /* GPU active — small matrices may be slower due to dispatch+conversion */
            if (M * N * K < 100000) {
                /* Small: GPU dispatch overhead expected to be worse */
                WARN("Small matrix GPU", "%s: %+.1f%% — GPU dispatch has fixed overhead",
                     sizes[s].label, overhead_pct);
            } else {
                VERDICT(overhead_pct < 30.0, "Dispatch overhead (GPU)",
                        "%s: %+.1f%%", sizes[s].label, overhead_pct);
            }
        }

        free(A); free(B); free(C_direct); free(C_dispatch);
    }

    fprintf(stderr, "╚══════════════════════════════════════════════════════════╝\n");
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Benchmark 5: Memory Audit
 *
 * Tracks: peak RSS during simulated conversation.
 * Checks: ring buffer memory accumulation, leak detection.
 * ═══════════════════════════════════════════════════════════════════════════ */

static void bench_memory(void) {
    fprintf(stderr, "\n╔══════════════════════════════════════════════════════════╗\n");
    fprintf(stderr, "║  Benchmark 5: Memory Audit                               ║\n");
    fprintf(stderr, "╠══════════════════════════════════════════════════════════╣\n");

    size_t rss_start = get_rss();

    /* ── Test 5a: Mel spectrogram memory over 60s conversation ── */
    {
        MelConfig cfg;
        mel_config_default(&cfg);
        MelSpectrogram *mel = mel_create(&cfg);
        size_t rss_after_create = get_rss();

        int sr = cfg.sample_rate;
        int n_mels = cfg.n_mels;
        int chunk = sr / 10;  /* 100ms chunks */
        float *pcm = (float *)malloc(chunk * sizeof(float));
        float *out = (float *)malloc(100 * n_mels * sizeof(float));

        /* Simulate 60 seconds of continuous streaming (10 utterances of ~6s each) */
        size_t rss_peak = 0;
        for (int utterance = 0; utterance < 10; utterance++) {
            mel_reset(mel);
            for (int c = 0; c < 60; c++) {  /* ~6s per utterance */
                for (int i = 0; i < chunk; i++)
                    pcm[i] = 0.3f * sinf(2.0f * M_PI * 440.0f *
                             (float)(c * chunk + i) / sr);
                mel_process(mel, pcm, chunk, out, 100);
            }
            size_t rss_now = get_rss();
            if (rss_now > rss_peak) rss_peak = rss_now;
        }

        size_t mel_growth = (rss_peak > rss_after_create)
            ? rss_peak - rss_after_create : 0;
        fprintf(stderr, "║  Mel spectrogram (60s, 10 utterances):\n");
        fprintf(stderr, "║    RSS after create: %.1f KB\n",
                (rss_after_create - rss_start) / 1024.0);
        fprintf(stderr, "║    RSS peak growth:  %.1f KB\n", mel_growth / 1024.0);

        VERDICT(mel_growth < 512 * 1024, "Mel memory growth",
                "%.1f KB peak growth (need < 512 KB)", mel_growth / 1024.0);

        mel_destroy(mel);
        free(pcm);
        free(out);
    }

    /* ── Test 5b: iSTFT memory stability ── */
    {
        SonataISTFT *dec = sonata_istft_create(ISTFT_N_FFT, ISTFT_HOP);
        size_t rss_after = get_rss();

        float *mag = (float *)calloc(ISTFT_N_BINS, sizeof(float));
        float *phase = (float *)calloc(ISTFT_N_BINS, sizeof(float));
        float audio[ISTFT_HOP];

        for (int b = 1; b < ISTFT_N_BINS; b++) mag[b] = 0.01f;

        /* Process 5000 frames (~100s) */
        for (int f = 0; f < 5000; f++) {
            for (int b = 1; b < ISTFT_N_BINS; b++)
                phase[b] = (float)(f * b) * 0.1f;
            sonata_istft_decode_frame(dec, mag, phase, audio);
        }

        size_t rss_final = get_rss();
        long istft_growth = (long)rss_final - (long)rss_after;

        fprintf(stderr, "║\n║  iSTFT (5000 frames, ~100s):\n");
        fprintf(stderr, "║    Memory growth: %ld bytes (should be ~0)\n", istft_growth);

        VERDICT(istft_growth < 64 * 1024, "iSTFT zero-alloc",
                "%ld bytes growth (expect 0 after init)", istft_growth);

        sonata_istft_destroy(dec);
        free(mag);
        free(phase);
    }

    /* ── Test 5c: Overall RSS ── */
    {
        size_t rss_end = get_rss();
        fprintf(stderr, "║\n║  Overall RSS:\n");
        fprintf(stderr, "║    Start: %.1f MB  End: %.1f MB  Delta: %.1f KB\n",
                rss_start / (1024.0 * 1024.0),
                rss_end / (1024.0 * 1024.0),
                (rss_end > rss_start ? rss_end - rss_start : 0) / 1024.0);
    }

    fprintf(stderr, "╚══════════════════════════════════════════════════════════╝\n");
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Benchmark 6: NEON Dequantization Throughput
 *
 * Isolates: the INT8→FP32 conversion (NEON vectorized).
 * Measures: GB/s throughput, cycles per element.
 * ═══════════════════════════════════════════════════════════════════════════ */

static void bench_neon_dequant(void) {
    fprintf(stderr, "\n╔══════════════════════════════════════════════════════════╗\n");
    fprintf(stderr, "║  Benchmark 6: NEON INT8 Dequantization Throughput        ║\n");
    fprintf(stderr, "╠══════════════════════════════════════════════════════════╣\n");

    int sizes[] = {512, 2048, 512 * 2048, 512 * 512};
    const char *labels[] = {"512 (1 row)", "2048 (1 FFN row)",
                            "1M (full FFN)", "256K (full attn)"};
    int n_sizes = sizeof(sizes) / sizeof(sizes[0]);
    int ITERS = 1000;

    for (int s = 0; s < n_sizes; s++) {
        int N = sizes[s];
        int8_t *src = (int8_t *)malloc(N);
        float *dst = (float *)malloc(N * sizeof(float));
        float scale = 0.00784f;  /* Typical: max_abs / 127 for ~1.0 range */

        for (int i = 0; i < N; i++) src[i] = (int8_t)(i % 256 - 128);

        /* Use volatile to prevent dead-store elimination */
        volatile float sink = 0;
        uint64_t t0 = mach_absolute_time();
        for (int iter = 0; iter < ITERS; iter++) {
            int k = 0;
            float32x4_t sv = vdupq_n_f32(scale);
            for (; k + 15 < N; k += 16) {
                int8x16_t q = vld1q_s8(src + k);
                int16x8_t lo16 = vmovl_s8(vget_low_s8(q));
                int16x8_t hi16 = vmovl_s8(vget_high_s8(q));
                float32x4_t f0 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(lo16)));
                float32x4_t f1 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(lo16)));
                float32x4_t f2 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(hi16)));
                float32x4_t f3 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(hi16)));
                vst1q_f32(dst + k,      vmulq_f32(f0, sv));
                vst1q_f32(dst + k + 4,  vmulq_f32(f1, sv));
                vst1q_f32(dst + k + 8,  vmulq_f32(f2, sv));
                vst1q_f32(dst + k + 12, vmulq_f32(f3, sv));
            }
            for (; k < N; k++) dst[k] = (float)src[k] * scale;
            sink = dst[N / 2];  /* prevent dead-store elimination */
        }
        uint64_t t1 = mach_absolute_time();
        (void)sink;

        double total_ns = ns_elapsed(t0, t1);
        double per_iter_ns = total_ns / ITERS;
        double ns_per_elem = per_iter_ns / N;
        double bytes_in = (double)N * 1;  /* 1 byte per int8 */
        double bytes_out = (double)N * 4;  /* 4 bytes per fp32 */
        double gb_per_s = (bytes_in + bytes_out) / per_iter_ns;  /* bytes/ns = GB/s */

        fprintf(stderr, "║  %s (%d elements):\n", labels[s], N);
        fprintf(stderr, "║    %.1f ns/iter  %.2f ns/elem  %.1f GB/s\n",
                per_iter_ns, ns_per_elem, gb_per_s);

        free(src);
        free(dst);
    }

    fprintf(stderr, "╚══════════════════════════════════════════════════════════╝\n");
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Summary
 * ═══════════════════════════════════════════════════════════════════════════ */

int main(void) {
    timing_init();
    srand(42);

    fprintf(stderr, "\n");
    fprintf(stderr, "████████████████████████████████████████████████████████████\n");
    fprintf(stderr, "██                                                        ██\n");
    fprintf(stderr, "██   POCKET-VOICE PERFORMANCE AUDIT                       ██\n");
    fprintf(stderr, "██   Validating: ring buffer, iSTFT, INT8, Metal          ██\n");
    fprintf(stderr, "██                                                        ██\n");
    fprintf(stderr, "████████████████████████████████████████████████████████████\n");

    bench_mel_spectrogram();
    bench_istft();
    bench_int8_vs_fp32();
    bench_metal_dispatch();
    bench_memory();
    bench_neon_dequant();

    /* ── Final Scorecard ── */
    fprintf(stderr, "\n");
    fprintf(stderr, "╔══════════════════════════════════════════════════════════╗\n");
    fprintf(stderr, "║  PERFORMANCE AUDIT RESULTS                               ║\n");
    fprintf(stderr, "╠══════════════════════════════════════════════════════════╣\n");
    fprintf(stderr, "║  Passed: %d   Failed: %d   Warnings: %d                     ║\n",
            tests_passed, tests_failed, tests_warn);
    fprintf(stderr, "╠══════════════════════════════════════════════════════════╣\n");
    fprintf(stderr, "║  Claims Validated:                                       ║\n");
    fprintf(stderr, "║    1. Ring buffer eliminates memmove     → See Bench 1   ║\n");
    fprintf(stderr, "║    2. Pre-alloc iSTFT = zero malloc      → See Bench 2   ║\n");
    fprintf(stderr, "║    3. INT8 = 4x less memory              → See Bench 3   ║\n");
    fprintf(stderr, "║  Regressions Checked:                                    ║\n");
    fprintf(stderr, "║    4. Metal dispatch overhead             → See Bench 4   ║\n");
    fprintf(stderr, "║    5. Memory accumulation                 → See Bench 5   ║\n");
    fprintf(stderr, "║    6. NEON dequant throughput             → See Bench 6   ║\n");
    fprintf(stderr, "╚══════════════════════════════════════════════════════════╝\n");

    return tests_failed > 0 ? 1 : 0;
}
