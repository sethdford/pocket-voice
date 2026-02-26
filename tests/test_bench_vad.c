/**
 * test_bench_vad.c — Native VAD benchmark + validation.
 *
 * 1. Latency: native_vad vs silero_vad (ONNX) on 1000 chunks.
 * 2. Speech detection: validates both engines detect synthetic speech-like audio
 *    and agree on silence vs speech classification.
 * 3. Real audio: processes hello_piper.wav and compares probability curves.
 *
 * Requires: models/silero_vad.nvad, models/silero_vad.onnx (optional for comparison)
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <mach/mach_time.h>

/* ── Native VAD API ────────────────────────────────────────────────────── */

typedef struct NativeVad NativeVad;
extern NativeVad *native_vad_create(const char *weights_path);
extern void native_vad_destroy(NativeVad *vad);
extern float native_vad_process(NativeVad *vad, const float *samples);
extern void native_vad_reset(NativeVad *vad);
extern int native_vad_chunk_size(const NativeVad *vad);

/* ── Silero ONNX VAD API (weak-linked, optional) ──────────────────────── */

typedef struct SileroVad SileroVad;
extern SileroVad *silero_vad_create(const char *model_path) __attribute__((weak));
extern void silero_vad_destroy(SileroVad *vad) __attribute__((weak));
extern float silero_vad_process(SileroVad *vad, const float *samples) __attribute__((weak));
extern void silero_vad_reset(SileroVad *vad) __attribute__((weak));

/* ── Helpers ───────────────────────────────────────────────────────────── */

static int passed = 0, failed = 0;
#define CHECK(cond, msg) do { \
    if (cond) { printf("  [PASS] %s\n", msg); passed++; } \
    else { printf("  [FAIL] %s\n", msg); failed++; } \
} while (0)

static double ns_per_tick(void) {
    mach_timebase_info_data_t tb;
    mach_timebase_info(&tb);
    return (double)tb.numer / (double)tb.denom;
}

static int file_exists(const char *path) {
    FILE *f = fopen(path, "rb");
    if (f) { fclose(f); return 1; }
    return 0;
}

static float *load_wav_mono_16k(const char *path, int *n_samples) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    uint8_t hdr[44];
    if (fread(hdr, 1, 44, f) != 44) { fclose(f); return NULL; }
    int sr = hdr[24] | (hdr[25] << 8) | (hdr[26] << 16) | (hdr[27] << 24);
    int bits = hdr[34] | (hdr[35] << 8);
    int channels = hdr[22] | (hdr[23] << 8);
    int data_size = hdr[40] | (hdr[41] << 8) | (hdr[42] << 16) | (hdr[43] << 24);
    int n = data_size / (bits / 8) / channels;

    float *raw = (float *)malloc((size_t)n * sizeof(float));
    if (!raw) { fclose(f); return NULL; }
    if (bits == 16) {
        int16_t *buf = (int16_t *)malloc(data_size);
        if (fread(buf, 1, data_size, f)) {}
        for (int i = 0; i < n; i++) raw[i] = buf[i * channels] / 32768.0f;
        free(buf);
    } else if (bits == 32) {
        float *buf = (float *)malloc(data_size);
        if (fread(buf, 1, data_size, f)) {}
        for (int i = 0; i < n; i++) raw[i] = buf[i * channels];
        free(buf);
    }
    fclose(f);

    int out_n = (int)((long long)n * 16000 / sr);
    float *out = (float *)malloc((size_t)out_n * sizeof(float));
    for (int i = 0; i < out_n; i++) {
        float src = (float)i * sr / 16000;
        int idx = (int)src;
        float frac = src - idx;
        if (idx + 1 < n)
            out[i] = raw[idx] * (1.0f - frac) + raw[idx + 1] * frac;
        else
            out[i] = raw[n - 1];
    }
    free(raw);
    *n_samples = out_n;
    return out;
}

/**
 * Generate synthetic speech-like audio: voiced + unvoiced segments
 * with amplitude envelope and formant structure.
 */
static void generate_speech(float *buf, int n_samples) {
    for (int i = 0; i < n_samples; i++) {
        float t = (float)i / 16000.0f;
        /* Voiced: fundamental at 150Hz + harmonics (male voice-like) */
        float voiced = 0.4f * sinf(2.0f * M_PI * 150.0f * t)
                     + 0.25f * sinf(2.0f * M_PI * 300.0f * t)
                     + 0.15f * sinf(2.0f * M_PI * 450.0f * t)
                     + 0.10f * sinf(2.0f * M_PI * 600.0f * t)
                     + 0.05f * sinf(2.0f * M_PI * 900.0f * t);
        /* Amplitude modulation (syllable-like envelope at ~4Hz) */
        float env = 0.6f + 0.4f * sinf(2.0f * M_PI * 4.0f * t);
        buf[i] = voiced * env;
    }
}

/* ── Main ──────────────────────────────────────────────────────────────── */

int main(void) {
    double tick = ns_per_tick();

    printf("═══ VAD Benchmark + Validation ═══\n\n");

    /* ── Load engines ──────────────────────────────────────────────── */
    const char *nvad_path = "models/silero_vad.nvad";
    const char *onnx_path = "models/silero_vad.onnx";

    if (!file_exists(nvad_path)) {
        printf("[SKIP] %s not found\n", nvad_path);
        return 0;
    }

    NativeVad *native = native_vad_create(nvad_path);
    CHECK(native != NULL, "native VAD loaded");
    if (!native) return 1;

    SileroVad *silero = NULL;
    int has_silero = (silero_vad_create != NULL) && file_exists(onnx_path);
    if (has_silero) {
        silero = silero_vad_create(onnx_path);
        if (!silero) has_silero = 0;
    }
    printf("  Silero ONNX: %s\n\n", has_silero ? "loaded" : "not available (comparison skipped)");

    /* ── 1. Latency Benchmark ──────────────────────────────────────── */
    printf("── Latency Benchmark (1000 chunks × 32ms) ──\n");

    float chunk[512];
    for (int i = 0; i < 512; i++)
        chunk[i] = 0.3f * sinf(2.0f * M_PI * 200.0f * (float)i / 16000.0f);

    const int N = 1000;

    /* Warmup */
    native_vad_reset(native);
    for (int i = 0; i < 10; i++) native_vad_process(native, chunk);

    /* Native benchmark */
    native_vad_reset(native);
    uint64_t t0 = mach_absolute_time();
    for (int i = 0; i < N; i++)
        native_vad_process(native, chunk);
    uint64_t t1 = mach_absolute_time();
    double native_total_ms = (double)(t1 - t0) * tick / 1e6;
    double native_per_chunk_us = native_total_ms * 1000.0 / N;
    double native_audio_ms = N * 32.0;  /* 1000 × 32ms = 32 seconds of audio */
    double native_rtf = native_total_ms / native_audio_ms;

    printf("  Native C:   %7.1f ms total  |  %5.1f µs/chunk  |  RTF %.5f  (%dx realtime)\n",
           native_total_ms, native_per_chunk_us, native_rtf,
           (int)(1.0 / native_rtf));
    CHECK(native_per_chunk_us < 1000.0, "native VAD < 1ms per chunk");

    if (has_silero) {
        /* Warmup */
        silero_vad_reset(silero);
        for (int i = 0; i < 10; i++) silero_vad_process(silero, chunk);

        /* ONNX benchmark */
        silero_vad_reset(silero);
        t0 = mach_absolute_time();
        for (int i = 0; i < N; i++)
            silero_vad_process(silero, chunk);
        t1 = mach_absolute_time();
        double onnx_total_ms = (double)(t1 - t0) * tick / 1e6;
        double onnx_per_chunk_us = onnx_total_ms * 1000.0 / N;
        double onnx_rtf = onnx_total_ms / native_audio_ms;
        double speedup = onnx_total_ms / native_total_ms;

        printf("  Silero ONNX: %7.1f ms total  |  %5.1f µs/chunk  |  RTF %.5f  (%dx realtime)\n",
               onnx_total_ms, onnx_per_chunk_us, onnx_rtf,
               (int)(1.0 / onnx_rtf));
        printf("  ─────────────────────────────────────────────────────\n");
        printf("  Speedup: %.1fx faster (native vs ONNX)\n", speedup);
        CHECK(speedup > 1.0, "native VAD is faster than ONNX");
    }

    /* ── 2. Synthetic Speech Detection ─────────────────────────────── */
    printf("\n── Synthetic Speech Detection ──\n");

    /* Generate 2 seconds of speech-like audio */
    int speech_samples = 32000;
    float *speech = (float *)malloc((size_t)speech_samples * sizeof(float));
    generate_speech(speech, speech_samples);

    native_vad_reset(native);
    int n_chunks = speech_samples / 512;
    float max_speech_prob = 0.0f;
    for (int i = 0; i < n_chunks; i++) {
        float p = native_vad_process(native, speech + i * 512);
        if (p > max_speech_prob) max_speech_prob = p;
    }
    printf("    Synthetic speech max probability: %.4f\n", max_speech_prob);

    /* Silence */
    float silence[512] = {0};
    native_vad_reset(native);
    float silence_prob = 0.0f;
    for (int i = 0; i < 10; i++)
        silence_prob = native_vad_process(native, silence);
    printf("    Sustained silence probability:    %.4f\n", silence_prob);

    CHECK(max_speech_prob > silence_prob, "speech has higher probability than silence");

    if (has_silero) {
        silero_vad_reset(silero);
        float onnx_max = 0.0f;
        for (int i = 0; i < n_chunks; i++) {
            float p = silero_vad_process(silero, speech + i * 512);
            if (p > onnx_max) onnx_max = p;
        }
        printf("    ONNX speech max:    %.4f\n", onnx_max);
        printf("    Native speech max:  %.4f\n", max_speech_prob);
        CHECK(fabsf(onnx_max - max_speech_prob) < 0.5f,
              "native and ONNX agree within 0.5 on synthetic speech");
    }
    free(speech);

    /* ── 3. Real Audio Comparison ──────────────────────────────────── */
    printf("\n── Real Audio Comparison ──\n");
    const char *wav_path = "hello_piper.wav";
    if (file_exists(wav_path)) {
        int n_samples = 0;
        float *audio = load_wav_mono_16k(wav_path, &n_samples);
        if (audio) {
            int nc = n_samples / 512;
            float *native_probs = (float *)malloc((size_t)nc * sizeof(float));

            native_vad_reset(native);
            float n_max = 0.0f, n_min = 1.0f, n_sum = 0.0f;
            for (int i = 0; i < nc; i++) {
                native_probs[i] = native_vad_process(native, audio + i * 512);
                if (native_probs[i] > n_max) n_max = native_probs[i];
                if (native_probs[i] < n_min) n_min = native_probs[i];
                n_sum += native_probs[i];
            }
            printf("    Native:  %d chunks, mean=%.4f, max=%.4f, min=%.4f\n",
                   nc, n_sum / nc, n_max, n_min);
            CHECK(n_max >= 0.0f && n_max <= 1.0f, "native probs in [0,1] on real audio");

            if (has_silero) {
                silero_vad_reset(silero);
                float o_max = 0.0f, o_min = 1.0f, o_sum = 0.0f;
                float mae = 0.0f;
                for (int i = 0; i < nc; i++) {
                    float p = silero_vad_process(silero, audio + i * 512);
                    if (p > o_max) o_max = p;
                    if (p < o_min) o_min = p;
                    o_sum += p;
                    mae += fabsf(p - native_probs[i]);
                }
                mae /= nc;
                printf("    ONNX:    %d chunks, mean=%.4f, max=%.4f, min=%.4f\n",
                       nc, o_sum / nc, o_max, o_min);
                printf("    MAE (native vs ONNX): %.4f\n", mae);

                CHECK(mae < 0.3f, "mean absolute error < 0.3 between native and ONNX");
            }

            free(native_probs);
            free(audio);
        }
    } else {
        printf("  [SKIP] %s not found\n", wav_path);
    }

    /* ── Cleanup ───────────────────────────────────────────────────── */
    native_vad_destroy(native);
    if (has_silero) silero_vad_destroy(silero);

    printf("\n══════════════════════════════════════════\n");
    printf("Results: %d passed, %d failed\n", passed, failed);
    if (failed > 0) { printf("SOME TESTS FAILED\n"); return 1; }
    printf("ALL PASSED\n");
    return 0;
}
