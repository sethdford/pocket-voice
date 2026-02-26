/*
 * bench_sonata.c — Comprehensive Sonata TTS performance benchmark.
 *
 * Measures:
 *   1. Sonata LM:    tokens/second, RTF, time-to-first-token
 *   2. Sonata Flow:   generation RTF per chunk
 *   3. iSTFT decode:  RTF, per-frame latency
 *   4. End-to-end:    text → audio RTF, first-chunk latency
 *   5. Multi-sentence: streaming pipeline throughput
 *
 * Requires: models/sonata/ (LM, Flow, Decoder safetensors + configs)
 *           models/tokenizer.model (SPM tokenizer)
 *
 * Build: make bench-sonata
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <mach/mach_time.h>

/* ─── FFI declarations ──────────────────────────────────────────────────── */

/* Sonata LM */
extern void *sonata_lm_create(const char *weights_path, const char *config_path);
extern void  sonata_lm_destroy(void *engine);
extern int   sonata_lm_set_text(void *engine, const unsigned int *text_ids, int n);
extern int   sonata_lm_append_text(void *engine, const unsigned int *text_ids, int n);
extern int   sonata_lm_finish_text(void *engine);
extern int   sonata_lm_step(void *engine, int *out_token);
extern int   sonata_lm_reset(void *engine);
extern int   sonata_lm_is_done(void *engine);
extern int   sonata_lm_set_params(void *engine, float temperature, int top_k,
                                   float top_p, float rep_penalty);
extern int   sonata_lm_load_draft(void *engine, const char *weights, const char *config);
extern int   sonata_lm_speculate_step(void *engine, int *out_tokens, int max, int *count);
extern int   sonata_lm_set_speculate_k(void *engine, int k);

/* SPM Tokenizer */
typedef struct SPMTokenizer SPMTokenizer;
extern SPMTokenizer *spm_create(const unsigned char *data, unsigned int size);
extern void  spm_destroy(SPMTokenizer *tok);
extern int   spm_encode(SPMTokenizer *tok, const char *text, int *out_ids, int max_ids);

/* Sonata iSTFT */
typedef struct SonataISTFT SonataISTFT;
extern SonataISTFT *sonata_istft_create(int n_fft, int hop_length);
extern void         sonata_istft_destroy(SonataISTFT *dec);
extern void         sonata_istft_reset(SonataISTFT *dec);
extern int          sonata_istft_decode_frame(SonataISTFT *dec,
                        const float *magnitude, const float *phase,
                        float *out_audio);
extern int          sonata_istft_decode_batch(SonataISTFT *dec,
                        const float *magnitudes, const float *phases,
                        int n_frames, float *out_audio);

/* Sonata Flow */
extern void *sonata_flow_create(const char *flow_weights, const char *flow_config,
                                 const char *decoder_weights, const char *decoder_config);
extern void  sonata_flow_destroy(void *engine);
extern int   sonata_flow_generate(void *engine, const int *semantic_tokens,
                                   int n_frames, float *out_magnitude, float *out_phase);
extern int   sonata_flow_generate_audio(void *engine, const int *semantic_tokens,
                                         int n_frames, float *out_audio, int max_samples);
extern int   sonata_flow_decoder_type(void *engine);
extern int   sonata_flow_samples_per_frame(void *engine);
extern int   sonata_flow_set_cfg_scale(void *engine, float scale);
extern int   sonata_flow_set_n_steps(void *engine, int n_steps);
extern int   sonata_flow_set_solver(void *engine, int use_heun);
extern void  sonata_flow_reset_phase(void *engine);

/* ─── Timing helpers ────────────────────────────────────────────────────── */

#define N_FFT       1024
#define HOP         480
#define N_BINS      (N_FFT / 2 + 1)
#define SAMPLE_RATE 24000
#define FRAME_RATE  50

static double time_us(void) {
    static mach_timebase_info_data_t tb;
    if (tb.denom == 0) mach_timebase_info(&tb);
    return (double)mach_absolute_time() * tb.numer / tb.denom / 1000.0;
}

static SPMTokenizer *load_tokenizer(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    if (sz <= 0) { fclose(f); return NULL; }
    fseek(f, 0, SEEK_SET);
    unsigned char *data = malloc(sz);
    if (!data) { fclose(f); return NULL; }
    if (fread(data, 1, sz, f) != (size_t)sz) { free(data); fclose(f); return NULL; }
    fclose(f);
    SPMTokenizer *tok = spm_create(data, (unsigned int)sz);
    free(data);
    return tok;
}

static void write_wav(const char *path, const float *pcm, int n, int sr) {
    FILE *f = fopen(path, "wb");
    if (!f) return;
    int data_size = n * 2, file_size = 36 + data_size;
    short fmt = 1, ch = 1, bps = 16;
    int byte_rate = sr * 2, block_align = 2, fmt_size = 16;
    fwrite("RIFF", 1, 4, f); fwrite(&file_size, 4, 1, f);
    fwrite("WAVEfmt ", 1, 8, f); fwrite(&fmt_size, 4, 1, f);
    fwrite(&fmt, 2, 1, f); fwrite(&ch, 2, 1, f);
    fwrite(&sr, 4, 1, f); fwrite(&byte_rate, 4, 1, f);
    fwrite(&block_align, 2, 1, f); fwrite(&bps, 2, 1, f);
    fwrite("data", 1, 4, f); fwrite(&data_size, 4, 1, f);
    for (int i = 0; i < n; i++) {
        float s = pcm[i] > 1.0f ? 1.0f : (pcm[i] < -1.0f ? -1.0f : pcm[i]);
        short v = (short)(s * 32767.0f);
        fwrite(&v, 2, 1, f);
    }
    fclose(f);
}

/* ─── Benchmark sentences ───────────────────────────────────────────────── */

static const char *BENCH_SENTENCES[] = {
    "Hello, how are you doing today?",
    "The quick brown fox jumps over the lazy dog near the riverbank.",
    "Artificial intelligence is transforming the way we interact with technology.",
    "She sells seashells by the seashore on a warm summer afternoon.",
    "In the year two thousand twenty five, the world changed forever.",
};
#define N_SENTENCES (sizeof(BENCH_SENTENCES) / sizeof(BENCH_SENTENCES[0]))

/* ═══════════════════════════════════════════════════════════════════════════
 * Benchmark 1: Sonata LM — token generation throughput
 * ═══════════════════════════════════════════════════════════════════════════ */

static void bench_lm(void *lm, SPMTokenizer *tok) {
    printf("\n┌─── Benchmark 1: Sonata LM Throughput ─────────────────┐\n");
    fflush(stdout);

    double total_gen_ms = 0;
    int total_tokens = 0;
    double best_ttft = 1e9;

    /* 250 tokens at 50 Hz = 5s of audio — sufficient to measure steady-state */
    const int MAX_TOKENS = 250;

    for (int s = 0; s < (int)N_SENTENCES; s++) {
        int32_t ids[512];
        int n = spm_encode(tok, BENCH_SENTENCES[s], ids, 512);
        if (n <= 0) continue;

        unsigned int uids[512];
        for (int i = 0; i < n; i++) uids[i] = (unsigned int)ids[i];

        sonata_lm_reset(lm);
        sonata_lm_set_text(lm, uids, n);

        int tokens[250];
        int n_tok = 0;
        double ttft = 0;

        double t0 = time_us();
        while (n_tok < MAX_TOKENS && !sonata_lm_is_done(lm)) {
            int out;
            int status = sonata_lm_step(lm, &out);
            if (status == 0) {
                if (n_tok == 0) ttft = (time_us() - t0) / 1000.0;
                tokens[n_tok++] = out;
            } else break;
        }
        double elapsed_ms = (time_us() - t0) / 1000.0;

        double audio_s = n_tok / (double)FRAME_RATE;
        double tok_per_s = (elapsed_ms > 0) ? n_tok / (elapsed_ms / 1000.0) : 0;
        double rtf = (audio_s > 0) ? elapsed_ms / 1000.0 / audio_s : 0;

        printf("│  \"%.*s%s\" → %d tokens (%.1fs audio)\n",
               40, BENCH_SENTENCES[s],
               strlen(BENCH_SENTENCES[s]) > 40 ? "..." : "",
               n_tok, audio_s);
        printf("│    %.0f ms  │  TTFT: %.1f ms  │  %.0f tok/s  │  RTF: %.3f\n",
               elapsed_ms, ttft, tok_per_s, rtf);
        fflush(stdout);

        total_gen_ms += elapsed_ms;
        total_tokens += n_tok;
        if (ttft > 0 && ttft < best_ttft) best_ttft = ttft;
    }

    if (total_tokens > 0 && total_gen_ms > 0) {
        double avg_tok_s = total_tokens / (total_gen_ms / 1000.0);
        double avg_rtf = total_gen_ms / 1000.0 / (total_tokens / (double)FRAME_RATE);
        printf("│\n");
        printf("│  ═══ LM Summary ═══════════════════════════════════\n");
        printf("│  Total: %d tokens in %.0f ms\n", total_tokens, total_gen_ms);
        printf("│  Average: %.0f tok/s  │  RTF: %.3f  │  Best TTFT: %.1f ms\n",
               avg_tok_s, avg_rtf, best_ttft);
    }
    printf("└───────────────────────────────────────────────────────┘\n");
    fflush(stdout);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Benchmark 1b: Sonata LM with Self-Speculative Decoding
 * ═══════════════════════════════════════════════════════════════════════════ */

static void bench_lm_speculative(void *lm, SPMTokenizer *tok) {
    int rc = sonata_lm_load_draft(lm, "models/sonata/sonata_lm.safetensors", NULL);
    if (rc != 0) {
        printf("\n┌─── Benchmark 1b: Self-Speculative Decoding ───────────┐\n");
        printf("│  [SKIP] Failed to load 4-layer draft model\n");
        printf("└───────────────────────────────────────────────────────┘\n");
        return;
    }
    sonata_lm_set_speculate_k(lm, 5);

    printf("\n┌─── Benchmark 1b: Self-Speculative Decoding (k=5) ─────┐\n");
    fflush(stdout);

    double total_gen_ms = 0;
    int total_tokens = 0;
    double best_ttft = 1e9;
    const int MAX_TOKENS = 250;

    for (int s = 0; s < (int)N_SENTENCES; s++) {
        int32_t ids[512];
        int n = spm_encode(tok, BENCH_SENTENCES[s], ids, 512);
        if (n <= 0) continue;

        unsigned int uids[512];
        for (int i = 0; i < n; i++) uids[i] = (unsigned int)ids[i];

        sonata_lm_reset(lm);
        sonata_lm_set_text(lm, uids, n);

        int n_tok = 0;
        double ttft = 0;

        double t0 = time_us();
        while (n_tok < MAX_TOKENS && !sonata_lm_is_done(lm)) {
            int spec_toks[16];
            int spec_count = 0;
            int status = sonata_lm_speculate_step(lm, spec_toks, 16, &spec_count);
            if (spec_count > 0) {
                if (n_tok == 0) ttft = (time_us() - t0) / 1000.0;
                n_tok += spec_count;
            }
            if (status == 1 || status == -1) break;
        }
        double elapsed_ms = (time_us() - t0) / 1000.0;

        double audio_s = n_tok / (double)FRAME_RATE;
        double tok_per_s = (elapsed_ms > 0) ? n_tok / (elapsed_ms / 1000.0) : 0;
        double rtf = (audio_s > 0) ? elapsed_ms / 1000.0 / audio_s : 0;

        printf("│  \"%.*s%s\" → %d tokens (%.1fs audio)\n",
               40, BENCH_SENTENCES[s],
               strlen(BENCH_SENTENCES[s]) > 40 ? "..." : "",
               n_tok, audio_s);
        printf("│    %.0f ms  │  TTFT: %.1f ms  │  %.0f tok/s  │  RTF: %.3f\n",
               elapsed_ms, ttft, tok_per_s, rtf);
        fflush(stdout);

        total_gen_ms += elapsed_ms;
        total_tokens += n_tok;
        if (ttft > 0 && ttft < best_ttft) best_ttft = ttft;
    }

    if (total_tokens > 0 && total_gen_ms > 0) {
        double avg_tok_s = total_tokens / (total_gen_ms / 1000.0);
        double avg_rtf = total_gen_ms / 1000.0 / (total_tokens / (double)FRAME_RATE);
        printf("│\n");
        printf("│  ═══ Speculative LM Summary ═══════════════════════\n");
        printf("│  Total: %d tokens in %.0f ms\n", total_tokens, total_gen_ms);
        printf("│  Average: %.0f tok/s  │  RTF: %.3f  │  Best TTFT: %.1f ms\n",
               avg_tok_s, avg_rtf, best_ttft);
    }
    printf("└───────────────────────────────────────────────────────┘\n");
    fflush(stdout);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Benchmark 2: Sonata Flow — acoustic generation RTF
 * ═══════════════════════════════════════════════════════════════════════════ */

static void bench_flow(void *flow) {
    printf("\n┌─── Benchmark 2: Sonata Flow + Decoder ────────────────┐\n");
    fflush(stdout);

    int dec_type = sonata_flow_decoder_type(flow);
    int spf = sonata_flow_samples_per_frame(flow);
    printf("│  Decoder type: %s (samples/frame: %d)\n",
           dec_type == 1 ? "ConvDecoder (direct audio)" : "none/iSTFT", spf);

    int chunk_sizes[] = {25, 50, 50, 100};
    int n_chunks = sizeof(chunk_sizes) / sizeof(chunk_sizes[0]);

    for (int c = 0; c < n_chunks; c++) {
        int n = chunk_sizes[c];
        int *dummy_tokens = calloc(n, sizeof(int));
        for (int i = 0; i < n; i++) dummy_tokens[i] = 100 + (i % 500);

        int max_samples = n * HOP + 4096;
        float *audio = calloc(max_samples, sizeof(float));

        double t0 = time_us();
        int ns;
        if (dec_type == 1) {
            ns = sonata_flow_generate_audio(flow, dummy_tokens, n, audio, max_samples);
        } else {
            float *mag = calloc(n * N_BINS, sizeof(float));
            float *phase = calloc(n * N_BINS, sizeof(float));
            sonata_flow_reset_phase(flow);
            ns = sonata_flow_generate(flow, dummy_tokens, n, mag, phase);
            free(mag); free(phase);
        }
        double elapsed_ms = (time_us() - t0) / 1000.0;

        double audio_s = (dec_type == 1 && ns > 0)
            ? (double)ns / SAMPLE_RATE
            : (double)(n * HOP) / SAMPLE_RATE;
        double rtf = (audio_s > 0) ? elapsed_ms / 1000.0 / audio_s : 0;

        printf("│  %d frames → %d samples (%.1fs): %.1f ms → RTF %.4f (%.0fx RT)\n",
               n, ns, audio_s, elapsed_ms, rtf, rtf > 0 ? 1.0 / rtf : 0);
        fflush(stdout);

        free(dummy_tokens);
        free(audio);

        if (ns <= 0) {
            printf("│    (returned 0 — model may not be loaded)\n");
            break;
        }
    }
    printf("└───────────────────────────────────────────────────────┘\n");
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Benchmark 3: iSTFT decode — pure vDSP/AMX performance
 * ═══════════════════════════════════════════════════════════════════════════ */

static void bench_istft(void) {
    printf("\n┌─── Benchmark 3: iSTFT Decode Performance ─────────────┐\n");
    fflush(stdout);

    SonataISTFT *dec = sonata_istft_create(N_FFT, HOP);

    int test_frames[] = {25, 50, 100, 500, 1000};
    int n_tests = sizeof(test_frames) / sizeof(test_frames[0]);

    for (int t = 0; t < n_tests; t++) {
        int n = test_frames[t];
        float *mag = calloc(n * N_BINS, sizeof(float));
        float *phase = calloc(n * N_BINS, sizeof(float));
        float *audio = calloc(n * HOP, sizeof(float));

        for (int f = 0; f < n; f++)
            for (int b = 1; b < N_BINS; b++) {
                mag[f * N_BINS + b] = 0.01f;
                phase[f * N_BINS + b] = (float)(f * b) * 0.1f;
            }

        sonata_istft_reset(dec);
        double t0 = time_us();
        int total = sonata_istft_decode_batch(dec, mag, phase, n, audio);
        double elapsed_ms = (time_us() - t0) / 1000.0;

        double audio_s = (double)total / SAMPLE_RATE;
        double rtf = elapsed_ms / 1000.0 / audio_s;

        printf("│  %4d frames → %6d samples (%.2fs): %.3f ms → RTF %.6f (%.0fx RT)\n",
               n, total, audio_s, elapsed_ms, rtf, 1.0 / rtf);

        free(mag); free(phase); free(audio);
    }

    sonata_istft_destroy(dec);
    printf("└───────────────────────────────────────────────────────┘\n");
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Benchmark 4: End-to-end — text → audio with chunked streaming
 * ═══════════════════════════════════════════════════════════════════════════ */

static void bench_e2e(void *lm, void *flow, SPMTokenizer *tok) {
    printf("\n┌─── Benchmark 4: End-to-End Text → Audio ──────────────┐\n");
    fflush(stdout);

    int dec_type = flow ? sonata_flow_decoder_type(flow) : 0;
    SonataISTFT *istft = (dec_type != 1) ? sonata_istft_create(N_FFT, HOP) : NULL;
    int max_audio = SAMPLE_RATE * 30;
    float *audio_buf = calloc(max_audio, sizeof(float));

    for (int s = 0; s < (int)N_SENTENCES; s++) {
        int32_t ids[512];
        int n = spm_encode(tok, BENCH_SENTENCES[s], ids, 512);
        if (n <= 0) continue;
        unsigned int uids[512];
        for (int i = 0; i < n; i++) uids[i] = (unsigned int)ids[i];

        sonata_lm_reset(lm);
        if (istft) sonata_istft_reset(istft);
        sonata_lm_set_text(lm, uids, n);

        int sem_tokens[250], n_sem = 0;
        int audio_pos = 0;
        int first_chunk = 12, chunk_size = 50;
        double first_audio_ms = 0;
        int total_sem = 0;

        double t0 = time_us();

        while (total_sem < 250 && !sonata_lm_is_done(lm)) {
            int out;
            int status = sonata_lm_step(lm, &out);
            if (status == 0) {
                sem_tokens[n_sem++] = out;
            } else break;

            total_sem++;

            int target = (audio_pos == 0) ? first_chunk : chunk_size;
            if (n_sem >= target || sonata_lm_is_done(lm) || total_sem >= 250) {
                int chunk_n = n_sem;

                if (flow && dec_type == 1) {
                    int chunk_max = chunk_n * HOP + 4096;
                    float *chunk_audio = calloc(chunk_max, sizeof(float));
                    int ns = sonata_flow_generate_audio(flow, sem_tokens, chunk_n,
                                                         chunk_audio, chunk_max);
                    if (ns > 0 && audio_pos + ns < max_audio) {
                        memcpy(audio_buf + audio_pos, chunk_audio, ns * sizeof(float));
                        audio_pos += ns;
                    }
                    free(chunk_audio);
                } else if (flow) {
                    float *mag = calloc(chunk_n * N_BINS, sizeof(float));
                    float *phase_buf = calloc(chunk_n * N_BINS, sizeof(float));
                    sonata_flow_reset_phase(flow);
                    int bins = sonata_flow_generate(flow, sem_tokens, chunk_n, mag, phase_buf);
                    if (bins > 0 && istft) {
                        for (int t = 0; t < chunk_n; t++) {
                            float frame[HOP];
                            int ns = sonata_istft_decode_frame(istft,
                                &mag[t * bins], &phase_buf[t * bins], frame);
                            if (ns > 0 && audio_pos + ns < max_audio) {
                                memcpy(audio_buf + audio_pos, frame, ns * sizeof(float));
                                audio_pos += ns;
                            }
                        }
                    }
                    free(mag); free(phase_buf);
                } else {
                    audio_pos += chunk_n * HOP;
                }

                if (first_audio_ms == 0 && audio_pos > 0)
                    first_audio_ms = (time_us() - t0) / 1000.0;

                n_sem = 0;
            }
        }

        double total_ms = (time_us() - t0) / 1000.0;
        double audio_s = (double)audio_pos / SAMPLE_RATE;
        double rtf = (audio_s > 0) ? total_ms / 1000.0 / audio_s : 0;

        printf("│  \"%.*s%s\"\n",
               50, BENCH_SENTENCES[s],
               strlen(BENCH_SENTENCES[s]) > 50 ? "..." : "");
        printf("│    %d samples (%.2fs) in %.0f ms  │  RTF: %.3f  │  1st chunk: %.0f ms\n",
               audio_pos, audio_s, total_ms, rtf, first_audio_ms);
        fflush(stdout);

        if (s == 0 && audio_pos > 0) {
            system("mkdir -p bench_output");
            write_wav("bench_output/sonata_bench_e2e.wav", audio_buf, audio_pos, SAMPLE_RATE);
        }
    }

    free(audio_buf);
    if (istft) sonata_istft_destroy(istft);
    printf("└───────────────────────────────────────────────────────┘\n");
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Benchmark 5: Flow solver comparison (Euler vs Heun, step counts)
 * ═══════════════════════════════════════════════════════════════════════════ */

static void bench_flow_configs(void *flow) {
    if (!flow) return;

    printf("\n┌─── Benchmark 5: Flow Solver × Step Count Matrix ──────┐\n");
    printf("│  Solver   Steps   Time(ms)   RTF      Speed\n");
    printf("│  ──────   ─────   ────────   ─────    ─────\n");
    fflush(stdout);

    int steps[] = {4, 8, 16};
    const char *solver_names[] = {"Euler", "Heun "};
    int n_frames = 50;

    int dec_type = sonata_flow_decoder_type(flow);
    int *dummy = calloc(n_frames, sizeof(int));
    int max_samples = n_frames * HOP + 4096;
    float *audio = calloc(max_samples, sizeof(float));
    for (int i = 0; i < n_frames; i++) dummy[i] = 100 + (i % 500);

    for (int solver = 0; solver <= 1; solver++) {
        sonata_flow_set_solver(flow, solver);
        for (int si = 0; si < 3; si++) {
            sonata_flow_set_n_steps(flow, steps[si]);

            double t0 = time_us();
            int ns;
            if (dec_type == 1) {
                ns = sonata_flow_generate_audio(flow, dummy, n_frames, audio, max_samples);
            } else {
                float *mag = calloc(n_frames * N_BINS, sizeof(float));
                float *phase = calloc(n_frames * N_BINS, sizeof(float));
                sonata_flow_reset_phase(flow);
                ns = sonata_flow_generate(flow, dummy, n_frames, mag, phase);
                free(mag); free(phase);
            }
            double ms = (time_us() - t0) / 1000.0;

            double audio_s = (dec_type == 1 && ns > 0)
                ? (double)ns / SAMPLE_RATE
                : (double)(n_frames * HOP) / SAMPLE_RATE;
            double rtf = ms / 1000.0 / audio_s;

            if (ns > 0) {
                printf("│  %s   %2d      %6.1f     %.4f   %.0fx RT\n",
                       solver_names[solver], steps[si], ms, rtf, 1.0 / rtf);
            } else {
                printf("│  %s   %2d      (failed)\n",
                       solver_names[solver], steps[si]);
                goto done;
            }
        }
    }

done:
    sonata_flow_set_solver(flow, 0);
    sonata_flow_set_n_steps(flow, 8);

    free(dummy); free(audio);
    printf("└───────────────────────────────────────────────────────┘\n");
}

/* ═══════════════════════════════════════════════════════════════════════════ */

int main(void) {
    printf("\n╔═══════════════════════════════════════════════════════════╗\n");
    printf("║          Sonata TTS Performance Benchmark                ║\n");
    printf("╚═══════════════════════════════════════════════════════════╝\n");

    system("mkdir -p bench_output");

    /* Load tokenizer */
    SPMTokenizer *tok = load_tokenizer("models/tokenizer.model");
    if (!tok) {
        fprintf(stderr, "[bench] No tokenizer at models/tokenizer.model — SKIP\n");
        return 1;
    }

    /* Load Sonata LM */
    void *lm = sonata_lm_create(
        "models/sonata/sonata_lm.safetensors",
        "models/sonata/sonata_lm_config.json"
    );
    if (!lm) {
        fprintf(stderr, "[bench] No Sonata LM weights — SKIP\n");
        spm_destroy(tok);
        return 1;
    }

    sonata_lm_set_params(lm, 0.8f, 50, 0.92f, 1.15f);

    /* Load Sonata Flow (optional — benchmark degrades gracefully) */
    void *flow = sonata_flow_create(
        "models/sonata/sonata_flow.safetensors",
        "models/sonata/sonata_flow_config.json",
        "models/sonata/sonata_decoder.safetensors",
        "models/sonata/sonata_decoder_config.json"
    );
    if (!flow) {
        fprintf(stderr, "[bench] No Flow weights — E2E will use placeholder synthesis\n");
    }

    /* ── Run benchmarks ── */
    bench_lm(lm, tok);
    bench_lm_speculative(lm, tok);
    bench_flow(flow);
    bench_istft();
    bench_e2e(lm, flow, tok);
    bench_flow_configs(flow);

    /* ── Summary ── */
    printf("\n╔═══════════════════════════════════════════════════════════╗\n");
    printf("║  Benchmark complete. Audio saved to bench_output/        ║\n");
    printf("╚═══════════════════════════════════════════════════════════╝\n\n");

    /* Cleanup */
    if (flow) sonata_flow_destroy(flow);
    sonata_lm_destroy(lm);
    spm_destroy(tok);

    return 0;
}
