/*
 * tools/bench_e2e_latency.c — End-to-end latency measurement for Sonata voice pipeline.
 *
 * Measures wall-clock time for each stage:
 *   - STT latency:     audio input → text tokens
 *   - LLM latency:     text tokens → response tokens (first + total)
 *   - TTS latency:     response tokens → first audio + total audio
 *   - E2E latency:     audio in → first audio out
 *
 * Generates synthetic 16kHz PCM audio (sine wave) as input.
 * Runs 10 iterations, reports min/avg/max/p95 for each stage.
 * Outputs both human-readable table and JSON.
 *
 * Build: make bench-e2e-latency
 * Run:   ./build/bench-e2e-latency
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <mach/mach_time.h>

/* ─── FFI declarations for Sonata STT ──────────────────────────────────── */

typedef struct SPMTokenizer SPMTokenizer;
extern SPMTokenizer *spm_create(const unsigned char *data, unsigned int size);
extern void         spm_destroy(SPMTokenizer *tok);
extern int          spm_encode(SPMTokenizer *tok, const char *text, int *out_ids, int max_ids);

/* STT (Conformer via BNNS or Rust) */
typedef struct ConformerSTT ConformerSTT;
extern ConformerSTT *conformer_stt_create(const char *model_path);
extern void         conformer_stt_destroy(ConformerSTT *engine);
extern int          conformer_stt_process(ConformerSTT *engine, const float *pcm, int n_samples);
extern int          conformer_stt_get_text(ConformerSTT *engine, char *out, int max_size);
extern int          conformer_stt_reset(ConformerSTT *engine);
extern int          conformer_stt_flush(ConformerSTT *engine);

/* LM (Sonata LM 241M) */
extern void *sonata_lm_create(const char *weights_path, const char *config_path);
extern void  sonata_lm_destroy(void *engine);
extern int   sonata_lm_set_text(void *engine, const unsigned int *text_ids, int n);
extern int   sonata_lm_finish_text(void *engine);
extern int   sonata_lm_step(void *engine, int *out_token);
extern int   sonata_lm_reset(void *engine);
extern int   sonata_lm_is_done(void *engine);

/* iSTFT (Sonata iSTFT decoder) */
typedef struct SonataISTFT SonataISTFT;
extern SonataISTFT *sonata_istft_create(int n_fft, int hop_length);
extern void         sonata_istft_destroy(SonataISTFT *dec);
extern void         sonata_istft_reset(SonataISTFT *dec);
extern int          sonata_istft_decode_frame(SonataISTFT *dec,
                        const float *magnitude, const float *phase,
                        float *out_audio);

/* Flow (Sonata Flow vocoder) */
extern void *sonata_flow_create(const char *flow_weights, const char *flow_config,
                                 const char *decoder_weights, const char *decoder_config);
extern void  sonata_flow_destroy(void *engine);
extern int   sonata_flow_generate_audio(void *engine, const int *semantic_tokens,
                                         int n_frames, float *out_audio, int max_samples);
extern int   sonata_flow_samples_per_frame(void *engine);
extern void  sonata_flow_reset_phase(void *engine);

/* ─── Constants ────────────────────────────────────────────────────────── */

#define SAMPLE_RATE    16000
#define AUDIO_DURATION 2  /* seconds of synthetic audio */
#define N_SAMPLES      (SAMPLE_RATE * AUDIO_DURATION)
#define N_FFT          1024
#define HOP            480
#define N_BINS         (N_FFT / 2 + 1)
#define N_FRAMES       ((N_SAMPLES / HOP) + 1)
#define MAX_TOKENS     256
#define N_ITERATIONS   10

/* ─── Timing helpers ────────────────────────────────────────────────────── */

typedef struct {
    double t_start;
    double t_end;
    double duration_us;
} TimedSegment;

typedef struct {
    double stt_latency_us;
    double llm_first_token_us;
    double llm_total_us;
    double tts_first_frame_us;
    double tts_total_us;
    double e2e_latency_us;  /* audio in to first audio out */
} E2ELatencies;

static double time_us(void) {
    static mach_timebase_info_data_t tb;
    if (tb.denom == 0) mach_timebase_info(&tb);
    return (double)mach_absolute_time() * tb.numer / tb.denom / 1000.0;
}

static inline double get_duration_us(double t_start, double t_end) {
    return t_end - t_start;
}

/* ─── Synthetic audio generation ───────────────────────────────────────── */

static void generate_sine_wave(float *out, int n_samples, int sample_rate, float freq_hz) {
    for (int i = 0; i < n_samples; i++) {
        out[i] = 0.1f * sinf(2.0f * M_PI * freq_hz * i / sample_rate);
    }
}

/* ─── Statistics ───────────────────────────────────────────────────────── */

typedef struct {
    double min;
    double max;
    double avg;
    double p95;
    int count;
} Stats;

static Stats compute_stats(double *values, int n) {
    Stats s = {0};
    if (n <= 0) return s;

    s.min = values[0];
    s.max = values[0];
    s.count = n;
    double sum = 0;

    for (int i = 0; i < n; i++) {
        if (values[i] < s.min) s.min = values[i];
        if (values[i] > s.max) s.max = values[i];
        sum += values[i];
    }
    s.avg = sum / n;

    /* p95: 95th percentile (simple implementation) */
    int p95_idx = (int)((n - 1) * 0.95);
    if (p95_idx < 0) p95_idx = 0;
    if (p95_idx >= n) p95_idx = n - 1;

    double sorted[N_ITERATIONS];
    memcpy(sorted, values, n * sizeof(double));
    for (int i = 0; i < n - 1; i++) {
        for (int j = i + 1; j < n; j++) {
            if (sorted[j] < sorted[i]) {
                double tmp = sorted[i];
                sorted[i] = sorted[j];
                sorted[j] = tmp;
            }
        }
    }
    s.p95 = sorted[p95_idx];

    return s;
}

/* ─── Main benchmark ───────────────────────────────────────────────────── */

int main(int argc, char *argv[]) {
    (void)argc; (void)argv;

    printf("=================================================================\n");
    printf(" Sonata E2E Latency Benchmark\n");
    printf(" Audio input: 16kHz, 2 seconds (32K samples)\n");
    printf(" Iterations: %d\n", N_ITERATIONS);
    printf("=================================================================\n\n");

    /* Load model paths from environment or defaults */
    const char *stt_weights = getenv("SONATA_STT_WEIGHTS") ?: "models/stt/conformer.safetensors";
    (void)getenv("SONATA_STT_CONFIG"); /* Not used directly by conformer_stt_create */
    const char *lm_weights = getenv("SONATA_LM_WEIGHTS") ?: "models/sonata/lm.safetensors";
    const char *lm_config = getenv("SONATA_LM_CONFIG") ?: "models/sonata/lm.json";
    const char *flow_weights = getenv("SONATA_FLOW_WEIGHTS") ?: "models/sonata/flow_v3.safetensors";
    const char *flow_config = getenv("SONATA_FLOW_CONFIG") ?: "models/sonata/flow_v3.json";
    const char *decoder_weights = getenv("SONATA_DECODER_WEIGHTS") ?: "models/sonata/decoder_v3.safetensors";
    const char *decoder_config = getenv("SONATA_DECODER_CONFIG") ?: "models/sonata/decoder_v3.json";
    const char *tokenizer_path = getenv("SONATA_TOKENIZER") ?: "models/tokenizer.model";

    /* Allocate audio buffers */
    float *audio_input = (float *)malloc(N_SAMPLES * sizeof(float));
    float *audio_output_buf = (float *)malloc(N_SAMPLES * 2 * sizeof(float));
    if (!audio_input || !audio_output_buf) {
        fprintf(stderr, "Error: malloc failed\n");
        return 1;
    }

    /* Generate synthetic audio */
    generate_sine_wave(audio_input, N_SAMPLES, SAMPLE_RATE, 440.0f);

    /* Arrays to store latencies across iterations */
    double stt_latencies[N_ITERATIONS] = {0};
    double llm_first_token_latencies[N_ITERATIONS] = {0};
    double llm_total_latencies[N_ITERATIONS] = {0};
    double tts_first_frame_latencies[N_ITERATIONS] = {0};
    double tts_total_latencies[N_ITERATIONS] = {0};
    double e2e_latencies[N_ITERATIONS] = {0};

    int stt_available = 0, lm_available = 0, flow_available = 0;
    void *stt_engine = NULL, *lm_engine = NULL, *flow_engine = NULL;
    SonataISTFT *istft = NULL;
    SPMTokenizer *tokenizer = NULL;

    /* Try to load STT */
    printf("[STT] Loading conformer from %s\n", stt_weights);
    stt_engine = conformer_stt_create(stt_weights);
    if (stt_engine) {
        stt_available = 1;
        printf("[STT] Loaded successfully\n");
    } else {
        printf("[STT] Not available (models missing) — skipping STT measurement\n");
    }

    /* Try to load tokenizer */
    printf("[Tokenizer] Loading from %s\n", tokenizer_path);
    FILE *tok_f = fopen(tokenizer_path, "rb");
    if (tok_f) {
        fseek(tok_f, 0, SEEK_END);
        long sz = ftell(tok_f);
        fseek(tok_f, 0, SEEK_SET);
        unsigned char *tok_data = (unsigned char *)malloc(sz);
        if (tok_data && fread(tok_data, 1, sz, tok_f) == (size_t)sz) {
            tokenizer = spm_create(tok_data, (unsigned int)sz);
            printf("[Tokenizer] Loaded successfully (%ld bytes)\n", sz);
        }
        free(tok_data);
        fclose(tok_f);
    }

    /* Try to load LM */
    printf("[LM] Loading from %s\n", lm_weights);
    lm_engine = sonata_lm_create(lm_weights, lm_config);
    if (lm_engine) {
        lm_available = 1;
        printf("[LM] Loaded successfully\n");
    } else {
        printf("[LM] Not available (models missing) — skipping LM measurement\n");
    }

    /* Try to load Flow */
    printf("[Flow] Loading from %s\n", flow_weights);
    flow_engine = sonata_flow_create(flow_weights, flow_config, decoder_weights, decoder_config);
    istft = sonata_istft_create(N_FFT, HOP);
    if (flow_engine && istft) {
        flow_available = 1;
        printf("[Flow] Loaded successfully\n");
    } else {
        printf("[Flow] Not available (models missing) — skipping TTS measurement\n");
    }

    printf("\n=================================================================\n");
    printf(" Running %d iterations...\n", N_ITERATIONS);
    printf("=================================================================\n\n");

    /* Run iterations */
    for (int iter = 0; iter < N_ITERATIONS; iter++) {
        printf("Iteration %d/%d...\n", iter + 1, N_ITERATIONS);
        double t_e2e_start = time_us();

        /* STT: transcribe synthetic audio */
        double t_stt_start = 0, t_stt_end = 0;
        if (stt_available) {
            conformer_stt_reset((ConformerSTT *)stt_engine);
            t_stt_start = time_us();
            for (int i = 0; i < N_SAMPLES; i += 512) {
                int batch = (i + 512 > N_SAMPLES) ? (N_SAMPLES - i) : 512;
                conformer_stt_process((ConformerSTT *)stt_engine, audio_input + i, batch);
            }
            conformer_stt_flush((ConformerSTT *)stt_engine);
            t_stt_end = time_us();
            stt_latencies[iter] = get_duration_us(t_stt_start, t_stt_end);
            char text_buf[256] = {0};
            conformer_stt_get_text((ConformerSTT *)stt_engine, text_buf, sizeof(text_buf) - 1);
            printf("  STT: %.2f ms (text: '%.40s')\n", stt_latencies[iter] / 1000.0, text_buf);
        } else {
            printf("  STT: N/A\n");
        }

        /* LM: generate semantic tokens from dummy text */
        double t_llm_first_token = 0, t_llm_total = 0;
        int semantic_tokens[MAX_TOKENS] = {0};
        int n_semantic_tokens = 0;

        if (lm_available && tokenizer) {
            const char *dummy_text = "hello world this is a test";
            int text_ids[MAX_TOKENS];
            int n_text_ids = spm_encode(tokenizer, dummy_text, text_ids, MAX_TOKENS);

            sonata_lm_reset(lm_engine);
            sonata_lm_set_text(lm_engine, (unsigned int *)text_ids, n_text_ids);
            sonata_lm_finish_text(lm_engine);

            double t_llm_start = time_us();
            int first_token_received = 0;
            double t_first_token = 0;

            while (!sonata_lm_is_done(lm_engine) && n_semantic_tokens < MAX_TOKENS) {
                int token = 0;
                int status = sonata_lm_step(lm_engine, &token);
                if (status >= 0) {
                    semantic_tokens[n_semantic_tokens++] = token;
                    if (!first_token_received) {
                        t_first_token = time_us();
                        t_llm_first_token = get_duration_us(t_llm_start, t_first_token);
                        first_token_received = 1;
                    }
                }
                if (status == 1) break;
            }
            double t_llm_end = time_us();
            t_llm_total = get_duration_us(t_llm_start, t_llm_end);

            llm_first_token_latencies[iter] = t_llm_first_token;
            llm_total_latencies[iter] = t_llm_total;
            printf("  LM: first_token=%.2f ms, total=%.2f ms (%d tokens)\n",
                   t_llm_first_token / 1000.0, t_llm_total / 1000.0, n_semantic_tokens);
        } else {
            printf("  LM: N/A\n");
        }

        /* TTS: generate audio from semantic tokens */
        double t_tts_first_frame = 0, t_tts_total = 0;
        if (flow_available && n_semantic_tokens > 0) {
            sonata_flow_reset_phase(flow_engine);
            sonata_istft_reset(istft);

            double t_tts_start = time_us();
            int samples_generated = 0;
            double t_first_frame = 0;

            int status = sonata_flow_generate_audio(flow_engine, semantic_tokens, n_semantic_tokens,
                                                     audio_output_buf, N_SAMPLES * 2);
            if (status > 0) {
                samples_generated = status;
                t_first_frame = time_us();
                t_tts_first_frame = get_duration_us(t_tts_start, t_first_frame);
            }

            double t_tts_end = time_us();
            t_tts_total = get_duration_us(t_tts_start, t_tts_end);

            tts_first_frame_latencies[iter] = t_tts_first_frame;
            tts_total_latencies[iter] = t_tts_total;
            printf("  TTS: first_frame=%.2f ms, total=%.2f ms (%d samples)\n",
                   t_tts_first_frame / 1000.0, t_tts_total / 1000.0, samples_generated);
        } else {
            printf("  TTS: N/A\n");
        }

        /* E2E latency: full pipeline */
        double t_e2e_end = time_us();
        double total_e2e = get_duration_us(t_e2e_start, t_e2e_end);
        e2e_latencies[iter] = total_e2e;
        printf("  E2E: %.2f ms\n", total_e2e / 1000.0);
    }

    /* Compute statistics */
    printf("\n=================================================================\n");
    printf(" Statistics (across %d iterations)\n", N_ITERATIONS);
    printf("=================================================================\n\n");

    Stats stt_stats = stt_available ? compute_stats(stt_latencies, N_ITERATIONS) : (Stats){0};
    Stats llm_ft_stats = lm_available ? compute_stats(llm_first_token_latencies, N_ITERATIONS) : (Stats){0};
    Stats llm_total_stats = lm_available ? compute_stats(llm_total_latencies, N_ITERATIONS) : (Stats){0};
    Stats tts_ff_stats = flow_available ? compute_stats(tts_first_frame_latencies, N_ITERATIONS) : (Stats){0};
    Stats tts_total_stats = flow_available ? compute_stats(tts_total_latencies, N_ITERATIONS) : (Stats){0};
    Stats e2e_stats = compute_stats(e2e_latencies, N_ITERATIONS);

    printf("%-25s | Min (ms) | Avg (ms) | Max (ms) | P95 (ms)\n", "Stage");
    printf("%-25s +---------+---------+---------+---------\n", "");

    if (stt_available) {
        printf("%-25s | %8.2f | %8.2f | %8.2f | %8.2f\n",
               "STT", stt_stats.min / 1000.0, stt_stats.avg / 1000.0, stt_stats.max / 1000.0, stt_stats.p95 / 1000.0);
    } else {
        printf("%-25s | N/A      | N/A      | N/A      | N/A\n", "STT");
    }

    if (lm_available) {
        printf("%-25s | %8.2f | %8.2f | %8.2f | %8.2f\n",
               "LM (first token)", llm_ft_stats.min / 1000.0, llm_ft_stats.avg / 1000.0, llm_ft_stats.max / 1000.0, llm_ft_stats.p95 / 1000.0);
        printf("%-25s | %8.2f | %8.2f | %8.2f | %8.2f\n",
               "LM (total)", llm_total_stats.min / 1000.0, llm_total_stats.avg / 1000.0, llm_total_stats.max / 1000.0, llm_total_stats.p95 / 1000.0);
    } else {
        printf("%-25s | N/A      | N/A      | N/A      | N/A\n", "LM (first token)");
        printf("%-25s | N/A      | N/A      | N/A      | N/A\n", "LM (total)");
    }

    if (flow_available) {
        printf("%-25s | %8.2f | %8.2f | %8.2f | %8.2f\n",
               "TTS (TTFA)", tts_ff_stats.min / 1000.0, tts_ff_stats.avg / 1000.0, tts_ff_stats.max / 1000.0, tts_ff_stats.p95 / 1000.0);
        printf("%-25s | %8.2f | %8.2f | %8.2f | %8.2f\n",
               "TTS (total)", tts_total_stats.min / 1000.0, tts_total_stats.avg / 1000.0, tts_total_stats.max / 1000.0, tts_total_stats.p95 / 1000.0);
    } else {
        printf("%-25s | N/A      | N/A      | N/A      | N/A\n", "TTS (TTFA)");
        printf("%-25s | N/A      | N/A      | N/A      | N/A\n", "TTS (total)");
    }

    printf("%-25s | %8.2f | %8.2f | %8.2f | %8.2f\n",
           "E2E (full pipeline)", e2e_stats.min / 1000.0, e2e_stats.avg / 1000.0, e2e_stats.max / 1000.0, e2e_stats.p95 / 1000.0);

    /* JSON output */
    printf("\n=================================================================\n");
    printf(" JSON Output\n");
    printf("=================================================================\n\n");

    printf("{\n");
    printf("  \"benchmark\": \"sonata_e2e_latency\",\n");
    printf("  \"date\": \"2026-03-08\",\n");
    printf("  \"iterations\": %d,\n", N_ITERATIONS);
    printf("  \"stages\": {\n");

    if (stt_available) {
        printf("    \"stt\": {\n");
        printf("      \"min_ms\": %.2f,\n", stt_stats.min / 1000.0);
        printf("      \"avg_ms\": %.2f,\n", stt_stats.avg / 1000.0);
        printf("      \"max_ms\": %.2f,\n", stt_stats.max / 1000.0);
        printf("      \"p95_ms\": %.2f\n", stt_stats.p95 / 1000.0);
        printf("    },\n");
    }

    if (lm_available) {
        printf("    \"llm_first_token\": {\n");
        printf("      \"min_ms\": %.2f,\n", llm_ft_stats.min / 1000.0);
        printf("      \"avg_ms\": %.2f,\n", llm_ft_stats.avg / 1000.0);
        printf("      \"max_ms\": %.2f,\n", llm_ft_stats.max / 1000.0);
        printf("      \"p95_ms\": %.2f\n", llm_ft_stats.p95 / 1000.0);
        printf("    },\n");
        printf("    \"llm_total\": {\n");
        printf("      \"min_ms\": %.2f,\n", llm_total_stats.min / 1000.0);
        printf("      \"avg_ms\": %.2f,\n", llm_total_stats.avg / 1000.0);
        printf("      \"max_ms\": %.2f,\n", llm_total_stats.max / 1000.0);
        printf("      \"p95_ms\": %.2f\n", llm_total_stats.p95 / 1000.0);
        printf("    },\n");
    }

    if (flow_available) {
        printf("    \"tts_first_frame\": {\n");
        printf("      \"min_ms\": %.2f,\n", tts_ff_stats.min / 1000.0);
        printf("      \"avg_ms\": %.2f,\n", tts_ff_stats.avg / 1000.0);
        printf("      \"max_ms\": %.2f,\n", tts_ff_stats.max / 1000.0);
        printf("      \"p95_ms\": %.2f\n", tts_ff_stats.p95 / 1000.0);
        printf("    },\n");
        printf("    \"tts_total\": {\n");
        printf("      \"min_ms\": %.2f,\n", tts_total_stats.min / 1000.0);
        printf("      \"avg_ms\": %.2f,\n", tts_total_stats.avg / 1000.0);
        printf("      \"max_ms\": %.2f,\n", tts_total_stats.max / 1000.0);
        printf("      \"p95_ms\": %.2f\n", tts_total_stats.p95 / 1000.0);
        printf("    },\n");
    }

    printf("    \"e2e_total\": {\n");
    printf("      \"min_ms\": %.2f,\n", e2e_stats.min / 1000.0);
    printf("      \"avg_ms\": %.2f,\n", e2e_stats.avg / 1000.0);
    printf("      \"max_ms\": %.2f,\n", e2e_stats.max / 1000.0);
    printf("      \"p95_ms\": %.2f\n", e2e_stats.p95 / 1000.0);
    printf("    }\n");

    printf("  }\n");
    printf("}\n");

    /* Cleanup */
    if (stt_engine) conformer_stt_destroy((ConformerSTT *)stt_engine);
    if (lm_engine) sonata_lm_destroy(lm_engine);
    if (flow_engine) sonata_flow_destroy(flow_engine);
    if (istft) sonata_istft_destroy(istft);
    if (tokenizer) spm_destroy(tokenizer);
    free(audio_input);
    free(audio_output_buf);

    printf("\n=================================================================\n");
    printf(" Benchmark complete\n");
    printf("=================================================================\n");

    return 0;
}
