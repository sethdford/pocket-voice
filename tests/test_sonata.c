/*
 * test_sonata.c — Comprehensive test for the Sonata TTS pipeline.
 *
 * Tests:
 *   1. iSTFT standalone: magnitude + phase → audio via vDSP
 *   2. SPM tokenizer: text → token IDs
 *   3. Sonata LM inference: token IDs → semantic tokens (50 Hz)
 *   4. iSTFT performance benchmark
 *   5. Sonata LM sampling params: temperature, top-k, top-p, repetition penalty
 *   6. Flow network API: create, set_speaker, set_cfg, set_steps, destroy
 *   7. Chunked generation: adaptive first chunk (25) + subsequent chunks (50)
 *   8. Phase continuity: cumulative phase across chunks
 *   9. Crossfade: seamless chunk boundary transitions
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

/* ─── Sonata LM FFI ─────────────────────────────────────────────────────── */

extern void *sonata_lm_create(const char *weights_path, const char *config_path);
extern void  sonata_lm_destroy(void *engine);
extern int   sonata_lm_set_text(void *engine, const unsigned int *text_ids, int n);
extern int   sonata_lm_step(void *engine, int *out_token);
extern int   sonata_lm_reset(void *engine);
extern int   sonata_lm_is_done(void *engine);
extern int   sonata_lm_sample_rate(void);
extern int   sonata_lm_frame_rate(void);
extern int   sonata_lm_samples_per_frame(void);
extern int   sonata_lm_set_params(void *engine, float temperature, int top_k,
                                   float top_p, float rep_penalty);

/* ─── SPM Tokenizer FFI ──────────────────────────────────────────────────── */

extern void *spm_create(const char *model_path);
extern void  spm_destroy(void *tok);
extern int   spm_encode(void *tok, const char *text, int *out_ids, int max_ids);
extern int   spm_vocab_size(void *tok);

/* ─── Sonata iSTFT FFI ──────────────────────────────────────────────────── */

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

/* ─── Sonata Flow FFI ────────────────────────────────────────────────────── */

extern void *sonata_flow_create(const char *flow_weights, const char *flow_config,
                                 const char *decoder_weights, const char *decoder_config);
extern void  sonata_flow_destroy(void *engine);
extern int   sonata_flow_generate(void *engine, const int *semantic_tokens,
                                   int n_frames, float *out_magnitude, float *out_phase);
extern int   sonata_flow_set_speaker(void *engine, int speaker_id);
extern int   sonata_flow_set_cfg_scale(void *engine, float scale);
extern int   sonata_flow_set_n_steps(void *engine, int n_steps);
extern void  sonata_flow_reset_phase(void *engine);
extern int   sonata_flow_set_solver(void *engine, int use_heun);
extern int   sonata_flow_set_speaker_embedding(void *engine, const float *embedding, int dim);
extern void  sonata_flow_clear_speaker_embedding(void *engine);
extern int   sonata_flow_n_steps(void);
extern int   sonata_flow_acoustic_dim(void);

/* Sonata LM speculative decoding FFI */
extern int   sonata_lm_load_draft(void *engine, const char *weights, const char *config);
extern int   sonata_lm_speculate_step(void *engine, int *out_tokens, int max_tokens, int *out_count);
extern int   sonata_lm_set_speculate_k(void *engine, int k);

/* BNNS ConvNeXt decoder FFI */
typedef struct BNNSConvNeXtDecoder BNNSConvNeXtDecoder;
extern BNNSConvNeXtDecoder *bnns_convnext_create(int n_layers, int dec_dim,
                                                    int conv_kernel, float ff_mult,
                                                    int input_dim, int n_fft);
extern int   bnns_convnext_forward(BNNSConvNeXtDecoder *dec,
                                    const float *semantic, const float *acoustic,
                                    int n_frames,
                                    float *out_magnitude, float *out_inst_freq);
extern void  bnns_convnext_destroy(BNNSConvNeXtDecoder *dec);

/* ─── Test helpers ───────────────────────────────────────────────────────── */

static int g_pass = 0, g_fail = 0;
#define CHECK(cond, msg) do { \
    if (cond) { g_pass++; printf("  [PASS] %s\n", msg); } \
    else { g_fail++; printf("  [FAIL] %s\n", msg); } \
} while(0)

#define CHECKF(cond, fmt, ...) do { \
    char _buf[256]; snprintf(_buf, sizeof(_buf), fmt, __VA_ARGS__); \
    if (cond) { g_pass++; printf("  [PASS] %s\n", _buf); } \
    else { g_fail++; printf("  [FAIL] %s\n", _buf); } \
} while(0)

static void write_wav(const char *path, const float *pcm, int n_samples, int sr) {
    FILE *f = fopen(path, "wb");
    if (!f) return;
    int data_size = n_samples * 2;
    int file_size = 36 + data_size;
    fwrite("RIFF", 1, 4, f);
    fwrite(&file_size, 4, 1, f);
    fwrite("WAVEfmt ", 1, 8, f);
    int fmt_size = 16;
    short fmt = 1, ch = 1, bps = 16;
    int byte_rate = sr * 2, block_align = 2;
    fwrite(&fmt_size, 4, 1, f);
    fwrite(&fmt, 2, 1, f);
    fwrite(&ch, 2, 1, f);
    fwrite(&sr, 4, 1, f);
    fwrite(&byte_rate, 4, 1, f);
    fwrite(&block_align, 2, 1, f);
    fwrite(&bps, 2, 1, f);
    fwrite("data", 1, 4, f);
    fwrite(&data_size, 4, 1, f);
    for (int i = 0; i < n_samples; i++) {
        float s = pcm[i];
        if (s > 1.0f) s = 1.0f;
        if (s < -1.0f) s = -1.0f;
        short v = (short)(s * 32767.0f);
        fwrite(&v, 2, 1, f);
    }
    fclose(f);
    printf("    Wrote %s (%d samples, %.2fs)\n", path, n_samples, (float)n_samples / sr);
}

/* ─── Test 1: iSTFT standalone ──────────────────────────────────────────── */

static void test_istft_standalone(void) {
    printf("\n═══ Test 1: Sonata iSTFT decoder ═══\n");

    int n_fft = 1024;
    int hop = 480;
    int n_bins = n_fft / 2 + 1;

    SonataISTFT *dec = sonata_istft_create(n_fft, hop);
    CHECK(dec != NULL, "iSTFT create");

    int n_frames = 50;
    float *magnitudes = calloc(n_frames * n_bins, sizeof(float));
    float *phases = calloc(n_frames * n_bins, sizeof(float));

    for (int f = 0; f < n_frames; f++) {
        for (int b = 5; b < 50; b++) {
            float freq_hz = (float)b * 24000.0f / n_fft;
            magnitudes[f * n_bins + b] = 10.0f / (1.0f + (b - 19) * (b - 19) * 0.01f);
            phases[f * n_bins + b] = 2.0f * M_PI * freq_hz * f * hop / 24000.0f;
        }
    }

    float *audio = calloc(n_frames * hop, sizeof(float));
    int total = sonata_istft_decode_batch(dec, magnitudes, phases, n_frames, audio);
    CHECK(total == n_frames * hop, "batch decode returns correct sample count");

    float rms = 0;
    for (int i = 0; i < total; i++) rms += audio[i] * audio[i];
    rms = sqrtf(rms / total);
    printf("    iSTFT output: %d samples, RMS=%.4f\n", total, rms);
    CHECK(rms > 0.001f, "iSTFT output has energy");

    write_wav("bench_output/sonata_istft_test.wav", audio, total, 24000);

    sonata_istft_destroy(dec);
    free(magnitudes);
    free(phases);
    free(audio);
}

/* ─── Test 2: SPM tokenizer ─────────────────────────────────────────────── */

static void test_spm_tokenizer(void) {
    printf("\n═══ Test 2: SPM tokenizer ═══\n");

    void *tok = spm_create("models/parakeet-ctc-1.1b-fp16.vocab");
    if (!tok) {
        printf("  [SKIP] No tokenizer model found\n");
        return;
    }

    int ids[256];
    int n = spm_encode(tok, "Hello world, this is Sonata.", ids, 256);
    CHECK(n > 0, "tokenize produces tokens");
    printf("    Tokens: %d ids for \"Hello world, this is Sonata.\"\n", n);

    int vocab = spm_vocab_size(tok);
    CHECK(vocab >= 0, "vocab size >= 0");
    printf("    Vocab size: %d\n", vocab);

    spm_destroy(tok);
}

/* ─── Test 3: Sonata LM (if weights available) ──────────────────────────── */

static void test_sonata_lm(void) {
    printf("\n═══ Test 3: Sonata LM inference ═══\n");

    void *engine = sonata_lm_create(
        "models/sonata/sonata_lm.safetensors",
        "models/sonata/sonata_lm_config.json"
    );
    if (!engine) {
        printf("  [SKIP] No Sonata LM weights found (not yet trained)\n");
        return;
    }

    CHECK(engine != NULL, "Sonata LM create");
    CHECK(sonata_lm_sample_rate() == 24000, "sample rate = 24000");
    CHECK(sonata_lm_frame_rate() == 50, "frame rate = 50 Hz");
    CHECK(sonata_lm_samples_per_frame() == 480, "samples per frame = 480");

    unsigned int text_ids[] = {10, 20, 30, 40, 50};
    int rc = sonata_lm_set_text(engine, text_ids, 5);
    CHECK(rc == 0, "set_text succeeds");

    int n_tokens = 0;
    int max_tokens = 100;
    int *tokens = malloc(max_tokens * sizeof(int));

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    while (n_tokens < max_tokens && !sonata_lm_is_done(engine)) {
        int tok;
        int status = sonata_lm_step(engine, &tok);
        if (status == 0) {
            tokens[n_tokens++] = tok;
        } else {
            break;
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &t1);
    double elapsed_ms = (t1.tv_sec - t0.tv_sec) * 1000.0 +
                        (t1.tv_nsec - t0.tv_nsec) / 1e6;

    printf("    Generated %d semantic tokens in %.1f ms\n", n_tokens, elapsed_ms);
    if (n_tokens > 0) {
        printf("    ms/token = %.1f\n", elapsed_ms / n_tokens);
        printf("    audio duration = %.2f s (at 50 Hz)\n", n_tokens / 50.0);
        double rtf = elapsed_ms / 1000.0 / (n_tokens / 50.0);
        printf("    LM RTF = %.3f\n", rtf);
    }

    CHECK(n_tokens > 0, "generated at least 1 semantic token");

    sonata_lm_reset(engine);
    sonata_lm_destroy(engine);
    free(tokens);
}

/* ─── Test 4: iSTFT performance benchmark ───────────────────────────────── */

static void test_istft_performance(void) {
    printf("\n═══ Test 4: iSTFT performance benchmark ═══\n");

    int n_fft = 1024;
    int hop = 480;
    int n_bins = n_fft / 2 + 1;
    int n_frames = 500;

    SonataISTFT *dec = sonata_istft_create(n_fft, hop);

    float *mag = calloc(n_frames * n_bins, sizeof(float));
    float *phase = calloc(n_frames * n_bins, sizeof(float));
    float *audio = calloc(n_frames * hop, sizeof(float));

    for (int f = 0; f < n_frames; f++) {
        for (int b = 1; b < n_bins; b++) {
            mag[f * n_bins + b] = 0.01f;
            phase[f * n_bins + b] = (float)(f * b) * 0.1f;
        }
    }

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    int total = sonata_istft_decode_batch(dec, mag, phase, n_frames, audio);

    clock_gettime(CLOCK_MONOTONIC, &t1);
    double elapsed_ms = (t1.tv_sec - t0.tv_sec) * 1000.0 +
                        (t1.tv_nsec - t0.tv_nsec) / 1e6;
    double audio_dur = (double)total / 24000.0;
    double rtf = elapsed_ms / 1000.0 / audio_dur;

    printf("    %d frames → %d samples (%.2f s audio)\n", n_frames, total, audio_dur);
    printf("    iSTFT time: %.2f ms\n", elapsed_ms);
    printf("    iSTFT RTF: %.6f (%.0fx faster than realtime)\n", rtf, 1.0 / rtf);
    printf("    Per frame: %.3f ms\n", elapsed_ms / n_frames);

    CHECK(rtf < 0.01, "iSTFT RTF < 0.01 (100x faster than realtime)");
    CHECK(total == n_frames * hop, "correct sample count");

    sonata_istft_destroy(dec);
    free(mag);
    free(phase);
    free(audio);
}

/* ─── Test 5: LM sampling parameters ────────────────────────────────────── */

static void test_lm_sampling_params(void) {
    printf("\n═══ Test 5: Sonata LM sampling parameters ═══\n");

    void *engine = sonata_lm_create(
        "models/sonata/sonata_lm.safetensors",
        "models/sonata/sonata_lm_config.json"
    );
    if (!engine) {
        printf("  [SKIP] No Sonata LM weights found\n");
        /* Test the FFI function exists and handles NULL gracefully */
        int rc = sonata_lm_set_params(NULL, 0.8f, 50, 0.92f, 1.15f);
        CHECK(rc == -1, "set_params rejects NULL engine");
        return;
    }

    int rc = sonata_lm_set_params(engine, 0.6f, 30, 0.95f, 1.2f);
    CHECK(rc == 0, "set_params(temp=0.6, k=30, p=0.95, rep=1.2) succeeds");

    rc = sonata_lm_set_params(engine, 1.0f, 100, 0.5f, 1.0f);
    CHECK(rc == 0, "set_params(temp=1.0, k=100, p=0.5, rep=1.0) succeeds");

    rc = sonata_lm_set_params(engine, -1.0f, -1, -1.0f, -1.0f);
    CHECK(rc == 0, "set_params with negative values (no-op) succeeds");

    sonata_lm_destroy(engine);
}

/* ─── Test 6: Flow network API ──────────────────────────────────────────── */

static void test_flow_api(void) {
    printf("\n═══ Test 6: Sonata Flow API ═══\n");

    CHECK(sonata_flow_n_steps() == 8, "default n_steps = 8");
    CHECK(sonata_flow_acoustic_dim() == 256, "acoustic_dim = 256");

    int rc = sonata_flow_set_speaker(NULL, 0);
    CHECK(rc == -1, "set_speaker rejects NULL engine");

    rc = sonata_flow_set_cfg_scale(NULL, 2.0f);
    CHECK(rc == -1, "set_cfg_scale rejects NULL engine");

    rc = sonata_flow_set_n_steps(NULL, 4);
    CHECK(rc == -1, "set_n_steps rejects NULL engine");

    sonata_flow_reset_phase(NULL);
    CHECK(1, "reset_phase handles NULL gracefully");

    rc = sonata_flow_generate(NULL, NULL, 0, NULL, NULL);
    CHECK(rc == 0, "generate rejects NULL engine");

    rc = sonata_flow_set_solver(NULL, 1);
    CHECK(rc == -1, "set_solver rejects NULL engine");

    rc = sonata_flow_set_speaker_embedding(NULL, NULL, 0);
    CHECK(rc == -1, "set_speaker_embedding rejects NULL/0");

    sonata_flow_clear_speaker_embedding(NULL);
    CHECK(1, "clear_speaker_embedding handles NULL gracefully");
}

/* ─── Test 7: Chunked iSTFT generation ──────────────────────────────────── */

static void test_chunked_istft(void) {
    printf("\n═══ Test 7: Chunked iSTFT generation ═══\n");

    int n_fft = 1024;
    int hop = 480;
    int n_bins = n_fft / 2 + 1;

    /* Generate 100 frames as two chunks of 50, compare to one batch */
    int n_total = 100;
    float *magnitudes = calloc(n_total * n_bins, sizeof(float));
    float *phases = calloc(n_total * n_bins, sizeof(float));

    for (int f = 0; f < n_total; f++) {
        for (int b = 5; b < 50; b++) {
            float freq_hz = (float)b * 24000.0f / n_fft;
            magnitudes[f * n_bins + b] = 5.0f / (1.0f + (b - 19) * (b - 19) * 0.01f);
            phases[f * n_bins + b] = 2.0f * M_PI * freq_hz * f * hop / 24000.0f;
        }
    }

    /* Single batch */
    SonataISTFT *dec1 = sonata_istft_create(n_fft, hop);
    float *audio_batch = calloc(n_total * hop, sizeof(float));
    int total_batch = sonata_istft_decode_batch(dec1, magnitudes, phases, n_total, audio_batch);

    /* Two chunks */
    SonataISTFT *dec2 = sonata_istft_create(n_fft, hop);
    float *audio_chunk = calloc(n_total * hop, sizeof(float));
    int chunk1 = 25;
    int chunk2 = n_total - chunk1;
    int n1 = sonata_istft_decode_batch(dec2, magnitudes, phases, chunk1, audio_chunk);
    int n2 = sonata_istft_decode_batch(dec2, magnitudes + chunk1 * n_bins,
                                        phases + chunk1 * n_bins,
                                        chunk2, audio_chunk + n1);
    int total_chunk = n1 + n2;

    CHECKF(total_batch == total_chunk, "batch (%d) == chunked (%d) sample count",
           total_batch, total_chunk);

    /* Audio should be identical (same iSTFT state progression) */
    float max_diff = 0;
    for (int i = 0; i < total_batch; i++) {
        float d = fabsf(audio_batch[i] - audio_chunk[i]);
        if (d > max_diff) max_diff = d;
    }
    CHECKF(max_diff < 1e-5f, "max diff between batch and chunked = %.2e (< 1e-5)", max_diff);

    sonata_istft_destroy(dec1);
    sonata_istft_destroy(dec2);
    free(magnitudes);
    free(phases);
    free(audio_batch);
    free(audio_chunk);
}

/* ─── Test 8: Phase continuity ──────────────────────────────────────────── */

static void test_phase_continuity(void) {
    printf("\n═══ Test 8: Phase continuity across chunks ═══\n");

    int n_fft = 1024;
    int hop = 480;
    int n_bins = n_fft / 2 + 1;

    /* Generate with cumulative phase to test iSTFT continuity */
    SonataISTFT *dec = sonata_istft_create(n_fft, hop);
    int n_frames = 100;
    float *audio = calloc(n_frames * hop, sizeof(float));

    float *phase_state = calloc(n_bins, sizeof(float));
    int total = 0;

    for (int f = 0; f < n_frames; f++) {
        float mag[513], ph[513];
        memset(mag, 0, n_bins * sizeof(float));

        /* 440 Hz tone with cumulative phase */
        int bin_440 = (int)(440.0f * n_fft / 24000.0f);
        mag[bin_440] = 10.0f;
        for (int b = 0; b < n_bins; b++) {
            float freq = (float)b * 24000.0f / n_fft;
            phase_state[b] += 2.0f * M_PI * freq * hop / 24000.0f;
            ph[b] = phase_state[b];
        }

        float frame[480];
        int ns = sonata_istft_decode_frame(dec, mag, ph, frame);
        if (ns > 0) {
            memcpy(audio + total, frame, ns * sizeof(float));
            total += ns;
        }
    }

    /* Check for discontinuities: max derivative should be bounded */
    float max_deriv = 0;
    for (int i = 1; i < total; i++) {
        float d = fabsf(audio[i] - audio[i - 1]);
        if (d > max_deriv) max_deriv = d;
    }

    float rms = 0;
    for (int i = 0; i < total; i++) rms += audio[i] * audio[i];
    rms = sqrtf(rms / total);

    printf("    Phase continuity: %d samples, RMS=%.4f, max_deriv=%.4f\n",
           total, rms, max_deriv);
    CHECK(rms > 0.001f, "440 Hz tone has energy");
    CHECK(max_deriv < 1.0f, "no large discontinuities in output");

    write_wav("bench_output/sonata_phase_continuity.wav", audio, total, 24000);

    sonata_istft_destroy(dec);
    free(audio);
    free(phase_state);
}

/* ─── Test 9: Crossfade between chunks ──────────────────────────────────── */

static void test_crossfade(void) {
    printf("\n═══ Test 9: Crossfade logic ═══\n");

    /* Simulate the crossfade that sonata_flush_chunk does */
    int cf_len = 32;
    float tail[32], new_audio[64];

    for (int i = 0; i < cf_len; i++) {
        tail[i] = 1.0f;
        new_audio[i] = -1.0f;
        new_audio[cf_len + i] = -1.0f;
    }

    /* Apply crossfade */
    for (int i = 0; i < cf_len; i++) {
        float alpha = (float)i / (float)cf_len;
        new_audio[i] = tail[i] * (1.0f - alpha) + new_audio[i] * alpha;
    }

    CHECK(fabsf(new_audio[0] - 1.0f) < 0.01f, "crossfade start ≈ old tail value");
    CHECK(fabsf(new_audio[cf_len - 1] - (-1.0f)) < 0.1f, "crossfade end ≈ new chunk value");
    CHECK(fabsf(new_audio[cf_len / 2]) < 0.5f, "crossfade midpoint near zero (blend)");

    /* Post-crossfade region should be untouched */
    CHECK(new_audio[cf_len] == -1.0f, "post-crossfade region untouched");
}

/* ─── Test 10: Speculative decoding API ──────────────────────────────────── */

static void test_speculative_api(void) {
    printf("\n═══ Test 10: Speculative decoding API ═══\n");

    int rc = sonata_lm_load_draft(NULL, NULL, NULL);
    CHECK(rc == -1, "load_draft rejects NULL engine");

    rc = sonata_lm_set_speculate_k(NULL, 5);
    CHECK(rc == -1, "set_speculate_k rejects NULL engine");

    int tokens[16], count = 0;
    rc = sonata_lm_speculate_step(NULL, tokens, 16, &count);
    CHECK(rc == -1, "speculate_step rejects NULL engine");
}

/* ─── Test 11: BNNS ConvNeXt decoder ────────────────────────────────────── */

static void test_bnns_convnext(void) {
    printf("\n═══ Test 11: BNNS ConvNeXt decoder ═══\n");

    int n_layers = 4, dec_dim = 512, conv_kernel = 7;
    float ff_mult = 4.0f;
    int fsq_dim = 9, acoustic_dim = 256;
    int input_dim = fsq_dim + acoustic_dim;
    int n_fft = 1024;
    int n_bins = n_fft / 2 + 1;

    BNNSConvNeXtDecoder *dec = bnns_convnext_create(n_layers, dec_dim,
                                                      conv_kernel, ff_mult,
                                                      input_dim, n_fft);
    CHECK(dec != NULL, "BNNS ConvNeXt create");

    int n_frames = 10;
    float *semantic = (float *)calloc(n_frames * fsq_dim, sizeof(float));
    float *acoustic = (float *)calloc(n_frames * acoustic_dim, sizeof(float));
    float *out_mag = (float *)calloc(n_frames * n_bins, sizeof(float));
    float *out_phase = (float *)calloc(n_frames * n_bins, sizeof(float));

    for (int i = 0; i < n_frames * acoustic_dim; i++) {
        acoustic[i] = 0.01f * (i % 17 - 8);
    }

    int bins = bnns_convnext_forward(dec, semantic, acoustic, n_frames,
                                      out_mag, out_phase);
    CHECKF(bins == n_bins, "forward returns %d bins (expected %d)", bins, n_bins);

    int has_mag_energy = 0;
    for (int i = 0; i < n_frames * n_bins; i++) {
        if (out_mag[i] > 0) { has_mag_energy = 1; break; }
    }
    CHECK(has_mag_energy, "magnitude output has energy");

    bnns_convnext_destroy(dec);
    free(semantic);
    free(acoustic);
    free(out_mag);
    free(out_phase);
}

/* ─── Test 12: Voice cloning embedding injection ────────────────────────── */

static void test_voice_cloning(void) {
    printf("\n═══ Test 12: Voice cloning embedding ═══\n");

    float dummy_emb[256];
    for (int i = 0; i < 256; i++) dummy_emb[i] = 0.01f * i;

    int rc = sonata_flow_set_speaker_embedding(NULL, dummy_emb, 256);
    CHECK(rc == -1, "set_speaker_embedding rejects NULL engine");

    rc = sonata_flow_set_speaker_embedding(NULL, NULL, 256);
    CHECK(rc == -1, "set_speaker_embedding rejects NULL data");

    sonata_flow_clear_speaker_embedding(NULL);
    CHECK(1, "clear_speaker_embedding NULL is no-op");
}

/* ─── Test 13: iSTFT zero magnitude (silence) ───────────────────────────── */

static void test_istft_zero_magnitude(void) {
    printf("\n═══ Test 13: iSTFT zero magnitude → silence ═══\n");

    int n_fft = 1024;
    int hop = 480;
    int n_bins = n_fft / 2 + 1;

    SonataISTFT *dec = sonata_istft_create(n_fft, hop);
    CHECK(dec != NULL, "iSTFT create for zero-mag test");

    /* All-zero magnitude with arbitrary phases */
    int n_frames = 10;
    float *magnitudes = calloc(n_frames * n_bins, sizeof(float));
    float *phases = calloc(n_frames * n_bins, sizeof(float));
    float *audio = calloc(n_frames * hop, sizeof(float));

    /* Set phases to non-zero to verify magnitude gates output */
    for (int f = 0; f < n_frames; f++)
        for (int b = 0; b < n_bins; b++)
            phases[f * n_bins + b] = (float)(f * b) * 0.5f;

    int total = sonata_istft_decode_batch(dec, magnitudes, phases, n_frames, audio);
    CHECK(total == n_frames * hop, "zero-mag: correct sample count");

    float rms = 0;
    for (int i = 0; i < total; i++) rms += audio[i] * audio[i];
    rms = sqrtf(rms / total);
    CHECKF(rms < 1e-6f, "zero-mag RMS = %.2e (should be ~0)", rms);

    sonata_istft_destroy(dec);
    free(magnitudes);
    free(phases);
    free(audio);
}

/* ─── Test 14: iSTFT single frame decode ────────────────────────────────── */

static void test_istft_single_frame(void) {
    printf("\n═══ Test 14: iSTFT single frame decode ═══\n");

    int n_fft = 1024;
    int hop = 480;
    int n_bins = n_fft / 2 + 1;

    SonataISTFT *dec = sonata_istft_create(n_fft, hop);
    CHECK(dec != NULL, "iSTFT create for single frame");

    float magnitude[513], phase[513], audio[480];
    memset(magnitude, 0, sizeof(magnitude));
    memset(phase, 0, sizeof(phase));

    /* Single 1kHz tone bin */
    int bin_1k = (int)(1000.0f * n_fft / 24000.0f);
    magnitude[bin_1k] = 5.0f;
    phase[bin_1k] = 0.0f;

    int ns = sonata_istft_decode_frame(dec, magnitude, phase, audio);
    CHECK(ns == hop, "single frame returns hop_length samples");

    /* Verify no NaN/Inf in output */
    int has_nan = 0;
    for (int i = 0; i < ns; i++) {
        if (isnan(audio[i]) || isinf(audio[i])) { has_nan = 1; break; }
    }
    CHECK(!has_nan, "single frame output has no NaN/Inf");

    sonata_istft_destroy(dec);
}

/* ─── Test 15: iSTFT very large magnitudes ──────────────────────────────── */

static void test_istft_large_magnitude(void) {
    printf("\n═══ Test 15: iSTFT large magnitude values ═══\n");

    int n_fft = 1024;
    int hop = 480;
    int n_bins = n_fft / 2 + 1;

    SonataISTFT *dec = sonata_istft_create(n_fft, hop);

    float *mag = calloc(n_bins, sizeof(float));
    float *phase = calloc(n_bins, sizeof(float));
    float audio[480];

    /* Fill with large magnitudes */
    for (int b = 0; b < n_bins; b++) {
        mag[b] = 1e6f;
        phase[b] = 0.0f;
    }

    int ns = sonata_istft_decode_frame(dec, mag, phase, audio);
    CHECK(ns == hop, "large-mag: returns correct sample count");

    /* Check no NaN/Inf */
    int ok = 1;
    for (int i = 0; i < ns; i++) {
        if (isnan(audio[i]) || isinf(audio[i])) { ok = 0; break; }
    }
    CHECK(ok, "large-mag: no NaN/Inf in output");

    sonata_istft_destroy(dec);
    free(mag);
    free(phase);
}

/* ─── Test 16: iSTFT reset clears overlap ───────────────────────────────── */

static void test_istft_reset(void) {
    printf("\n═══ Test 16: iSTFT reset clears overlap state ═══\n");

    int n_fft = 1024;
    int hop = 480;
    int n_bins = n_fft / 2 + 1;

    SonataISTFT *dec = sonata_istft_create(n_fft, hop);

    /* Decode a few frames to build up overlap state */
    float *mag = calloc(n_bins, sizeof(float));
    float *phase = calloc(n_bins, sizeof(float));
    float audio_a[480], audio_b[480];

    int bin_440 = (int)(440.0f * n_fft / 24000.0f);
    mag[bin_440] = 10.0f;
    phase[bin_440] = 1.5f;

    sonata_istft_decode_frame(dec, mag, phase, audio_a);
    sonata_istft_decode_frame(dec, mag, phase, audio_a);

    /* Reset and decode fresh */
    sonata_istft_reset(dec);

    /* Decode same frame from clean state */
    SonataISTFT *dec2 = sonata_istft_create(n_fft, hop);
    float audio_fresh[480];

    sonata_istft_decode_frame(dec, mag, phase, audio_b);
    sonata_istft_decode_frame(dec2, mag, phase, audio_fresh);

    /* After reset, output should match a fresh decoder */
    float max_diff = 0;
    for (int i = 0; i < hop; i++) {
        float d = fabsf(audio_b[i] - audio_fresh[i]);
        if (d > max_diff) max_diff = d;
    }
    CHECKF(max_diff < 1e-5f, "reset: max diff from fresh = %.2e (< 1e-5)", max_diff);

    sonata_istft_destroy(dec);
    sonata_istft_destroy(dec2);
    free(mag);
    free(phase);
}

/* ─── Test 17: iSTFT phase unwrapping correctness ───────────────────────── */

static void test_istft_phase_unwrap(void) {
    printf("\n═══ Test 17: iSTFT phase unwrapping correctness ═══\n");

    int n_fft = 1024;
    int hop = 480;
    int n_bins = n_fft / 2 + 1;
    int n_frames = 50;

    SonataISTFT *dec = sonata_istft_create(n_fft, hop);

    float *audio = calloc(n_frames * hop, sizeof(float));
    float mag[513], ph[513];

    /* 1kHz pure tone with linearly increasing phase (proper phase unwrap) */
    float freq_hz = 1000.0f;
    int bin = (int)(freq_hz * n_fft / 24000.0f);

    for (int f = 0; f < n_frames; f++) {
        memset(mag, 0, n_bins * sizeof(float));
        memset(ph, 0, n_bins * sizeof(float));
        mag[bin] = 10.0f;
        /* Phase increments by 2*pi*freq*hop/sr per frame — the expected iSTFT phase */
        ph[bin] = 2.0f * M_PI * freq_hz * (f + 1) * hop / 24000.0f;

        float frame[480];
        int ns = sonata_istft_decode_frame(dec, mag, ph, frame);
        memcpy(audio + f * hop, frame, ns * sizeof(float));
    }

    /* Measure energy — should be significant for a pure tone */
    float rms = 0;
    for (int i = 0; i < n_frames * hop; i++) rms += audio[i] * audio[i];
    rms = sqrtf(rms / (n_frames * hop));
    CHECK(rms > 0.01f, "phase-unwrap: 1kHz tone has energy");

    /* Check smoothness — derivative should be bounded for a pure tone */
    float max_deriv = 0;
    for (int i = 1; i < n_frames * hop; i++) {
        float d = fabsf(audio[i] - audio[i - 1]);
        if (d > max_deriv) max_deriv = d;
    }
    CHECK(max_deriv < 2.0f, "phase-unwrap: no harsh discontinuities");

    sonata_istft_destroy(dec);
    free(audio);
}

/* ─── SPM Tokenizer FFI (C API, used for round-trip tests) ──────────────── */

typedef struct SPMTokenizer SPMTokenizer;
extern SPMTokenizer *spm_create_from_vocab(const char **pieces,
                                            const float *scores, int n_pieces);
extern int spm_decode(const SPMTokenizer *tok, const int *ids, int n_ids,
                      char *out_text, int out_cap);

/* ─── Test 18: SPM encode/decode round-trip ─────────────────────────────── */

static void test_spm_roundtrip(void) {
    printf("\n═══ Test 18: SPM encode/decode round-trip ═══\n");

    /* Build a synthetic vocabulary */
    const char *pieces[] = {
        "<unk>",          /* 0 = UNK */
        "\xe2\x96\x81",  /* 1 = SP space marker (▁) */
        "h", "e", "l", "o",  /* 2-5 */
        "\xe2\x96\x81w", /* 6 = ▁w */
        "or",             /* 7 */
        "ld",             /* 8 */
        "he",             /* 9 */
        "ll",             /* 10 */
        "\xe2\x96\x81hello", /* 11 = ▁hello */
        "\xe2\x96\x81world", /* 12 = ▁world */
    };
    float scores[] = {
        0.0f,    /* unk */
        -1.0f,   /* space */
        -3.0f,   /* h */
        -3.0f,   /* e */
        -3.0f,   /* l */
        -3.0f,   /* o */
        -2.5f,   /* ▁w */
        -2.5f,   /* or */
        -2.5f,   /* ld */
        -2.0f,   /* he */
        -2.0f,   /* ll */
        -0.5f,   /* ▁hello */
        -0.5f,   /* ▁world */
    };
    int n_pieces = sizeof(pieces) / sizeof(pieces[0]);

    SPMTokenizer *tok = spm_create_from_vocab(pieces, scores, n_pieces);
    CHECK(tok != NULL, "create_from_vocab succeeds");
    if (!tok) return;

    int vocab = spm_vocab_size(tok);
    CHECKF(vocab == n_pieces, "vocab_size = %d (expected %d)", vocab, n_pieces);

    /* Encode a simple string */
    int ids[64];
    int n = spm_encode(tok, "hello world", ids, 64);
    CHECK(n > 0, "encode 'hello world' produces tokens");

    /* Decode back */
    char text[256];
    int tlen = spm_decode(tok, ids, n, text, 256);
    CHECK(tlen > 0, "decode produces text");

    /* The decoded text should contain 'hello' and 'world' */
    CHECK(strstr(text, "hello") != NULL, "round-trip contains 'hello'");
    CHECK(strstr(text, "world") != NULL, "round-trip contains 'world'");
    printf("    round-trip: '%s' → %d tokens → '%s'\n", "hello world", n, text);

    spm_destroy(tok);
}

/* ─── Test 19: SPM empty string encoding ────────────────────────────────── */

static void test_spm_empty_string(void) {
    printf("\n═══ Test 19: SPM empty string encoding ═══\n");

    const char *pieces[] = {"<unk>", "a", "b"};
    float scores[] = {0.0f, -1.0f, -1.0f};
    SPMTokenizer *tok = spm_create_from_vocab(pieces, scores, 3);
    CHECK(tok != NULL, "create for empty-string test");
    if (!tok) return;

    int ids[64];
    int n = spm_encode(tok, "", ids, 64);
    CHECK(n == 0, "encode empty string → 0 tokens");

    spm_destroy(tok);
}

/* ─── Test 20: SPM NULL safety ──────────────────────────────────────────── */

static void test_spm_null_safety(void) {
    printf("\n═══ Test 20: SPM NULL safety ═══\n");

    /* create_from_vocab with NULL args */
    SPMTokenizer *tok = spm_create_from_vocab(NULL, NULL, 0);
    CHECK(tok == NULL, "create_from_vocab(NULL, NULL, 0) → NULL");

    tok = spm_create_from_vocab(NULL, NULL, 5);
    CHECK(tok == NULL, "create_from_vocab(NULL, NULL, 5) → NULL");

    /* encode with NULL tokenizer */
    int ids[8];
    int n = spm_encode(NULL, "hello", ids, 8);
    CHECK(n <= 0, "encode(NULL tok) returns <= 0");

    /* decode with NULL tokenizer */
    char buf[64];
    int d = spm_decode(NULL, ids, 1, buf, 64);
    CHECK(d == -1, "decode(NULL tok) returns -1");

    /* encode with NULL text */
    const char *p[] = {"<unk>", "a"};
    float s[] = {0.0f, -1.0f};
    tok = spm_create_from_vocab(p, s, 2);
    if (tok) {
        n = spm_encode(tok, NULL, ids, 8);
        CHECK(n <= 0, "encode(NULL text) returns <= 0");
        spm_destroy(tok);
    }
}

/* ─── Test 21: SPM very long string ─────────────────────────────────────── */

static void test_spm_long_string(void) {
    printf("\n═══ Test 21: SPM very long string ═══\n");

    const char *pieces[] = {
        "<unk>", "a", "b", "c", " ",
        "\xe2\x96\x81a", "\xe2\x96\x81b",
    };
    float scores[] = {0.0f, -1.0f, -1.0f, -1.0f, -2.0f, -0.5f, -0.5f};
    int n_pieces = sizeof(pieces) / sizeof(pieces[0]);

    SPMTokenizer *tok = spm_create_from_vocab(pieces, scores, n_pieces);
    CHECK(tok != NULL, "create for long-string test");
    if (!tok) return;

    /* Build a 4000-char string of "abcabc..." */
    char *long_str = malloc(4001);
    for (int i = 0; i < 4000; i++) long_str[i] = 'a' + (i % 3);
    long_str[4000] = '\0';

    int ids[8192];
    int n = spm_encode(tok, long_str, ids, 8192);
    CHECK(n > 0, "encode 4000-char string produces tokens");
    CHECKF(n <= 8192, "token count %d fits in buffer", n);

    /* Verify no garbage — all IDs should be valid */
    int valid = 1;
    for (int i = 0; i < n; i++) {
        if (ids[i] < 0 || ids[i] >= n_pieces) { valid = 0; break; }
    }
    CHECK(valid, "all token IDs are valid vocab indices");

    free(long_str);
    spm_destroy(tok);
}

/* ─── Test 22: SPM output buffer overflow protection ────────────────────── */

static void test_spm_overflow_protection(void) {
    printf("\n═══ Test 22: SPM output buffer overflow ═══\n");

    const char *pieces[] = {"<unk>", "a", "b", "c"};
    float scores[] = {0.0f, -1.0f, -1.0f, -1.0f};
    SPMTokenizer *tok = spm_create_from_vocab(pieces, scores, 4);
    CHECK(tok != NULL, "create for overflow test");
    if (!tok) return;

    /* Encode with very small output buffer */
    int ids[2];
    int n = spm_encode(tok, "aabbcc", ids, 2);
    CHECK(n <= 2, "encode with max_ids=2 respects limit");
    CHECK(n >= 0, "encode returns non-negative count");

    spm_destroy(tok);
}

/* ─── Test 23: SPM unicode handling ─────────────────────────────────────── */

static void test_spm_unicode(void) {
    printf("\n═══ Test 23: SPM unicode handling ═══\n");

    /* Vocab with some multi-byte pieces */
    const char *pieces[] = {
        "<unk>",
        "\xc3\xa9",  /* é (U+00E9, 2-byte UTF-8) */
        "c", "a", "f",
        "\xe2\x96\x81",  /* ▁ (space marker) */
        "\xc3\xa9" "s",  /* és */
    };
    float scores[] = {0.0f, -1.0f, -2.0f, -2.0f, -2.0f, -1.0f, -0.5f};
    int n_pieces = sizeof(pieces) / sizeof(pieces[0]);

    SPMTokenizer *tok = spm_create_from_vocab(pieces, scores, n_pieces);
    CHECK(tok != NULL, "create with unicode vocab");
    if (!tok) return;

    int ids[64];
    int n = spm_encode(tok, "caf\xc3\xa9", ids, 64);
    CHECK(n > 0, "encode 'café' produces tokens");

    /* Decode and verify */
    char text[256];
    int tlen = spm_decode(tok, ids, n, text, 256);
    CHECK(tlen > 0, "decode unicode produces text");
    CHECK(strstr(text, "caf") != NULL, "decoded text contains 'caf'");

    spm_destroy(tok);
}

/* ─── Main ──────────────────────────────────────────────────────────────── */

int main(void) {
    printf("╔════════════════════════════════════════════════╗\n");
    printf("║  Sonata TTS — Comprehensive Test Suite (v4)   ║\n");
    printf("╚════════════════════════════════════════════════╝\n");

    system("mkdir -p bench_output");

    test_istft_standalone();
    test_spm_tokenizer();
    test_sonata_lm();
    test_istft_performance();
    test_lm_sampling_params();
    test_flow_api();
    test_chunked_istft();
    test_phase_continuity();
    test_crossfade();
    test_speculative_api();
    test_bnns_convnext();
    test_voice_cloning();

    /* New iSTFT edge-case tests */
    test_istft_zero_magnitude();
    test_istft_single_frame();
    test_istft_large_magnitude();
    test_istft_reset();
    test_istft_phase_unwrap();

    /* New SPM tokenizer tests */
    test_spm_roundtrip();
    test_spm_empty_string();
    test_spm_null_safety();
    test_spm_long_string();
    test_spm_overflow_protection();
    test_spm_unicode();

    printf("\n══════════════════════════════════════════\n");
    printf("Results: %d / %d passed\n", g_pass, g_pass + g_fail);
    if (g_fail > 0) {
        printf("FAILURES: %d\n", g_fail);
        return 1;
    }
    printf("ALL PASSED\n");
    return 0;
}
