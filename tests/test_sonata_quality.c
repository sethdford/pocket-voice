/**
 * test_sonata_quality.c — Sonata TTS audio quality validation.
 *
 * Generates speech for test sentences via Sonata LM + Flow + ConvDecoder,
 * writes WAV files for listening, computes audio stats, and optionally
 * runs round-trip WER via Conformer STT.
 *
 * Run: make test-sonata-quality
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <mach/mach_time.h>

#ifndef ACCELERATE_NEW_LAPACK
#define ACCELERATE_NEW_LAPACK
#endif
#include <Accelerate/Accelerate.h>

/* ─── FFI: Sonata LM ──────────────────────────────────────────────────── */
extern void *sonata_lm_create(const char *weights, const char *config);
extern void  sonata_lm_destroy(void *e);
extern int   sonata_lm_set_text(void *e, const unsigned int *ids, int n);
extern int   sonata_lm_step(void *e, int *out);
extern int   sonata_lm_reset(void *e);
extern int   sonata_lm_is_done(void *e);
extern int   sonata_lm_set_params(void *e, float temp, int top_k, float top_p, float rep);

/* ─── FFI: Sonata Flow + ConvDecoder ──────────────────────────────────── */
extern void *sonata_flow_create(const char *fw, const char *fc, const char *dw, const char *dc);
extern void  sonata_flow_destroy(void *e);
extern int   sonata_flow_generate_audio(void *e, const int *tokens, int n, float *out, int max);
extern int   sonata_flow_decoder_type(void *e);
extern int   sonata_flow_samples_per_frame(void *e);

/* ─── FFI: SPM Tokenizer ──────────────────────────────────────────────── */
typedef struct SPMTokenizer SPMTokenizer;
extern SPMTokenizer *spm_create(const uint8_t *data, uint32_t size);
extern void  spm_destroy(SPMTokenizer *t);
extern int   spm_encode(SPMTokenizer *t, const char *text, int32_t *ids, int max);

/* ─── FFI: Conformer STT (optional — may not have model) ─────────────── */
typedef struct ConformerSTT ConformerSTT;
extern ConformerSTT *conformer_stt_create(const char *path);
extern void  conformer_stt_destroy(ConformerSTT *stt);
extern void  conformer_stt_reset(ConformerSTT *stt);
extern int   conformer_stt_process(ConformerSTT *stt, const float *pcm, int n);
extern int   conformer_stt_flush(ConformerSTT *stt);
extern int   conformer_stt_get_text(const ConformerSTT *stt, char *buf, int cap);

/* ─── FFI: WER ────────────────────────────────────────────────────────── */
typedef struct { int sub, del, ins, ref_words, hyp_words; float wer, cer, accuracy; } WERResult;
extern WERResult wer_compute(const char *ref, const char *hyp);

/* ─── FFI: Audio Quality ──────────────────────────────────────────────── */
extern float prosody_predict_mos(const float *audio, int len, int sr);

/* ─── Constants ───────────────────────────────────────────────────────── */
#define SAMPLE_RATE 24000
#define MAX_TOKENS  300
#define MAX_AUDIO   (MAX_TOKENS * 480 + 8192)

static mach_timebase_info_data_t g_tb;
static double now_ms(void) {
    if (g_tb.denom == 0) mach_timebase_info(&g_tb);
    return (double)mach_absolute_time() * g_tb.numer / g_tb.denom / 1e6;
}

static int g_pass = 0, g_fail = 0, g_skip = 0;
#define TEST(msg) fprintf(stderr, "  %-55s", msg)
#define PASS() do { fprintf(stderr, "[PASS]\n"); g_pass++; } while(0)
#define FAIL(r) do { fprintf(stderr, "[FAIL] %s\n", r); g_fail++; } while(0)
#define SKIP(r) do { fprintf(stderr, "[SKIP] %s\n", r); g_skip++; } while(0)

static void write_wav(const char *path, const float *audio, int n, int sr) {
    FILE *f = fopen(path, "wb");
    if (!f) return;
    int data_bytes = n * 2;
    int file_bytes = 36 + data_bytes;
    unsigned char hdr[44] = {
        'R','I','F','F', file_bytes&0xff, (file_bytes>>8)&0xff,
        (file_bytes>>16)&0xff, (file_bytes>>24)&0xff,
        'W','A','V','E', 'f','m','t',' ', 16,0,0,0, 1,0, 1,0,
        sr&0xff,(sr>>8)&0xff,(sr>>16)&0xff,(sr>>24)&0xff,
        (sr*2)&0xff,((sr*2)>>8)&0xff,((sr*2)>>16)&0xff,((sr*2)>>24)&0xff,
        2,0, 16,0, 'd','a','t','a',
        data_bytes&0xff,(data_bytes>>8)&0xff,(data_bytes>>16)&0xff,(data_bytes>>24)&0xff
    };
    fwrite(hdr, 1, 44, f);
    for (int i = 0; i < n; i++) {
        float s = audio[i];
        if (s > 1.0f) s = 1.0f;
        if (s < -1.0f) s = -1.0f;
        int16_t v = (int16_t)(s * 32767.0f);
        fwrite(&v, 2, 1, f);
    }
    fclose(f);
}

static float *resample_24k_to_16k(const float *src, int src_n, int *dst_n) {
    double ratio = 24000.0 / 16000.0;
    *dst_n = (int)((double)src_n / ratio);
    float *dst = (float *)malloc(*dst_n * sizeof(float));
    if (!dst) return NULL;
    for (int i = 0; i < *dst_n; i++) {
        double si = i * ratio;
        int idx = (int)si;
        double frac = si - idx;
        if (idx >= src_n - 1) idx = src_n - 2;
        if (idx < 0) idx = 0;
        dst[i] = src[idx] * (1.0f - (float)frac) + src[idx + 1] * (float)frac;
    }
    return dst;
}

typedef struct {
    float rms;
    float peak;
    float zero_crossing_rate;
    float duration_s;
    int   n_samples;
    int   is_silence;
} AudioStats;

static AudioStats compute_stats(const float *audio, int n, int sr) {
    AudioStats s = {0};
    s.n_samples = n;
    s.duration_s = (float)n / sr;

    float sum_sq = 0;
    int zero_crossings = 0;
    for (int i = 0; i < n; i++) {
        float v = fabsf(audio[i]);
        if (v > s.peak) s.peak = v;
        sum_sq += audio[i] * audio[i];
        if (i > 0 && ((audio[i] >= 0) != (audio[i-1] >= 0)))
            zero_crossings++;
    }
    s.rms = sqrtf(sum_sq / (n > 0 ? n : 1));
    s.zero_crossing_rate = (float)zero_crossings / (n > 0 ? n : 1);
    s.is_silence = (s.rms < 0.001f);
    return s;
}

static SPMTokenizer *load_tokenizer(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    uint8_t *data = (uint8_t *)malloc(sz);
    if (!data) { fclose(f); return NULL; }
    fread(data, 1, sz, f);
    fclose(f);
    SPMTokenizer *tok = spm_create(data, (uint32_t)sz);
    free(data);
    return tok;
}

typedef struct {
    const char *text;
    float max_wer;
} TestCase;

static TestCase test_cases[] = {
    {"Hello, how are you today?",                          0.60f},
    {"The quick brown fox jumps over the lazy dog.",       0.50f},
    {"One two three four five six seven eight nine ten.",  0.50f},
    {"The weather is sunny and warm.",                     0.60f},
    {"She sells seashells by the seashore.",               0.60f},
    {NULL, 0}
};

int main(void) {
    fprintf(stderr, "\n");
    fprintf(stderr, "╔═══════════════════════════════════════════════════════╗\n");
    fprintf(stderr, "║  Sonata TTS — Audio Quality Validation               ║\n");
    fprintf(stderr, "║  LM + Flow + ConvDecoder → WAV → STT → WER          ║\n");
    fprintf(stderr, "╚═══════════════════════════════════════════════════════╝\n\n");

    /* ─── Load Models ─────────────────────────────────────────────────── */

    fprintf(stderr, "═══ Loading Models ═══\n");

    SPMTokenizer *tok = load_tokenizer("models/tokenizer.model");
    if (!tok) {
        fprintf(stderr, "  [FATAL] Cannot load tokenizer\n");
        return 1;
    }
    fprintf(stderr, "  SPM tokenizer loaded\n");

    void *lm = sonata_lm_create(
        "models/sonata/sonata_lm.safetensors",
        "models/sonata/sonata_lm_config.json");
    if (!lm) {
        fprintf(stderr, "  [FATAL] Cannot load Sonata LM\n");
        spm_destroy(tok);
        return 1;
    }
    sonata_lm_set_params(lm, 0.7f, 40, 0.90f, 1.2f);
    fprintf(stderr, "  Sonata LM loaded\n");

    void *flow = sonata_flow_create(
        "models/sonata/sonata_flow.safetensors",
        "models/sonata/sonata_flow_config.json",
        "models/sonata/sonata_decoder.safetensors",
        "models/sonata/sonata_decoder_config.json");
    if (!flow) {
        fprintf(stderr, "  [FATAL] Cannot load Sonata Flow\n");
        sonata_lm_destroy(lm);
        spm_destroy(tok);
        return 1;
    }
    int dec_type = sonata_flow_decoder_type(flow);
    int spf = sonata_flow_samples_per_frame(flow);
    fprintf(stderr, "  Sonata Flow loaded (decoder=%s, spf=%d)\n",
            dec_type == 1 ? "ConvDecoder" : dec_type == 2 ? "VocosDecoder" : "iSTFT", spf);

    ConformerSTT *stt = conformer_stt_create("models/parakeet-ctc-1.1b-fp16.cstt");
    if (stt) {
        fprintf(stderr, "  Conformer STT loaded (round-trip enabled)\n");
    } else {
        fprintf(stderr, "  Conformer STT not available (round-trip disabled)\n");
    }

    fprintf(stderr, "\n");

    /* ─── Generate & Evaluate ─────────────────────────────────────────── */

    float *audio_buf = (float *)calloc(MAX_AUDIO, sizeof(float));
    int total_wer_tests = 0;
    float total_wer = 0;
    int total_audio_ok = 0;
    int total_intelligible = 0;

    for (int s = 0; test_cases[s].text != NULL; s++) {
        const char *text = test_cases[s].text;
        fprintf(stderr, "═══ Sentence %d: \"%s\" ═══\n", s + 1, text);

        /* Tokenize */
        int32_t text_ids[512];
        int n_ids = spm_encode(tok, text, text_ids, 512);
        if (n_ids <= 0) {
            fprintf(stderr, "  Tokenization failed\n");
            continue;
        }
        unsigned int uids[512];
        for (int i = 0; i < n_ids; i++) uids[i] = (unsigned int)text_ids[i];

        /* LM: generate semantic tokens */
        sonata_lm_reset(lm);
        sonata_lm_set_text(lm, uids, n_ids);

        int sem_tokens[MAX_TOKENS];
        int n_sem = 0;
        double t0 = now_ms();
        while (n_sem < MAX_TOKENS && !sonata_lm_is_done(lm)) {
            int tok_out = 0;
            int st = sonata_lm_step(lm, &tok_out);
            if (st == 1 || st == -1) break;
            sem_tokens[n_sem++] = tok_out;
        }
        double lm_ms = now_ms() - t0;
        float lm_toks = (lm_ms > 0) ? (n_sem * 1000.0f / lm_ms) : 0;
        fprintf(stderr, "  LM: %d tokens in %.0f ms (%.0f tok/s)\n", n_sem, lm_ms, lm_toks);

        if (n_sem < 5) {
            fprintf(stderr, "  [WARN] Very few tokens generated — LM may not have converged\n");
        }

        /* Flow + Decoder: generate audio */
        int audio_len = 0;
        double t1 = now_ms();
        if (dec_type == 1 || dec_type == 2) {
            int max_samples = n_sem * spf + 4096;
            if (max_samples > MAX_AUDIO) max_samples = MAX_AUDIO;
            audio_len = sonata_flow_generate_audio(flow, sem_tokens, n_sem, audio_buf, max_samples);
        }
        double flow_ms = now_ms() - t1;
        float audio_dur = (float)audio_len / SAMPLE_RATE;
        float flow_rtf = (audio_dur > 0) ? (flow_ms / 1000.0f / audio_dur) : 99;
        fprintf(stderr, "  Flow: %d samples (%.2fs) in %.0f ms (RTF %.3f)\n",
                audio_len, audio_dur, flow_ms, flow_rtf);

        /* Audio stats */
        AudioStats stats = compute_stats(audio_buf, audio_len, SAMPLE_RATE);
        fprintf(stderr, "  Audio: RMS=%.4f  peak=%.4f  ZCR=%.4f  silence=%s\n",
                stats.rms, stats.peak, stats.zero_crossing_rate,
                stats.is_silence ? "YES" : "no");

        /* Write WAV */
        char wav_path[256];
        snprintf(wav_path, sizeof(wav_path), "bench_output/sonata_quality_%d.wav", s);
        if (audio_len > 0) {
            write_wav(wav_path, audio_buf, audio_len, SAMPLE_RATE);
            fprintf(stderr, "  Wrote: %s (%.2fs, %d bytes)\n",
                    wav_path, audio_dur, 44 + audio_len * 2);
        }

        /* Test: audio has energy (not silence) */
        char test_msg[128];
        snprintf(test_msg, sizeof(test_msg), "sentence %d: audio has energy (RMS > 0.001)", s + 1);
        TEST(test_msg);
        if (audio_len > 0 && !stats.is_silence) {
            PASS();
            total_audio_ok++;
        } else if (audio_len == 0) {
            FAIL("no audio generated");
        } else {
            FAIL("output is silence");
        }

        /* Test: audio duration is reasonable (0.5-15s for these sentences) */
        snprintf(test_msg, sizeof(test_msg), "sentence %d: duration %.2fs in [0.5, 15]s", s + 1, audio_dur);
        TEST(test_msg);
        if (audio_dur >= 0.5f && audio_dur <= 15.0f) {
            PASS();
        } else if (audio_dur > 0) {
            FAIL("unusual duration");
        } else {
            FAIL("no audio");
        }

        /* Test: zero-crossing rate suggests speech (0.01-0.3 typical for speech) */
        snprintf(test_msg, sizeof(test_msg), "sentence %d: ZCR %.3f in speech range [0.01, 0.3]", s + 1, stats.zero_crossing_rate);
        TEST(test_msg);
        if (stats.zero_crossing_rate >= 0.01f && stats.zero_crossing_rate <= 0.3f) {
            PASS();
        } else if (audio_len == 0) {
            FAIL("no audio");
        } else {
            FAIL("ZCR outside speech range");
        }

        /* Round-trip: Sonata TTS → Conformer STT → WER */
        if (stt && audio_len > 0 && !stats.is_silence) {
            int resampled_n = 0;
            float *audio_16k = resample_24k_to_16k(audio_buf, audio_len, &resampled_n);
            if (audio_16k && resampled_n > 0) {
                conformer_stt_reset(stt);
                conformer_stt_process(stt, audio_16k, resampled_n);
                conformer_stt_flush(stt);
                char transcript[1024] = {0};
                conformer_stt_get_text(stt, transcript, sizeof(transcript));

                if (strlen(transcript) > 0) {
                    WERResult wer = wer_compute(text, transcript);
                    fprintf(stderr, "  STT transcript: \"%s\"\n", transcript);
                    fprintf(stderr, "  WER: %.1f%% (S=%d D=%d I=%d, ref=%d hyp=%d)\n",
                            wer.wer * 100, wer.sub, wer.del, wer.ins,
                            wer.ref_words, wer.hyp_words);

                    snprintf(test_msg, sizeof(test_msg),
                             "sentence %d: WER %.0f%% <= %.0f%% threshold",
                             s + 1, wer.wer * 100, test_cases[s].max_wer * 100);
                    TEST(test_msg);
                    if (wer.wer <= test_cases[s].max_wer) {
                        PASS();
                        total_intelligible++;
                    } else {
                        FAIL("WER too high");
                    }
                    total_wer += wer.wer;
                    total_wer_tests++;
                } else {
                    fprintf(stderr, "  STT: empty transcript (audio may not be speech)\n");
                    snprintf(test_msg, sizeof(test_msg), "sentence %d: STT produces transcript", s + 1);
                    TEST(test_msg);
                    FAIL("empty transcript");
                    total_wer_tests++;
                }
                free(audio_16k);
            }
        } else if (!stt) {
            fprintf(stderr, "  [skip] No STT model for round-trip\n");
        }

        fprintf(stderr, "\n");
    }

    /* ─── Summary ─────────────────────────────────────────────────────── */

    fprintf(stderr, "╔═══════════════════════════════════════════════════════╗\n");
    fprintf(stderr, "║  Quality Summary                                     ║\n");
    fprintf(stderr, "╠═══════════════════════════════════════════════════════╣\n");
    int n_cases = 0;
    for (int i = 0; test_cases[i].text; i++) n_cases++;
    fprintf(stderr, "║  Audio generated:  %d / %d sentences                   ║\n",
            total_audio_ok, n_cases);
    if (total_wer_tests > 0) {
        float mean_wer = total_wer / total_wer_tests;
        fprintf(stderr, "║  Round-trip WER:   %.1f%% mean (%d/%d intelligible)     ║\n",
                mean_wer * 100, total_intelligible, total_wer_tests);
        if (mean_wer < 0.05f)
            fprintf(stderr, "║  Verdict:          ★★★ HUMAN-LEVEL (WER < 5%%)       ║\n");
        else if (mean_wer < 0.15f)
            fprintf(stderr, "║  Verdict:          ★★ GOOD (WER < 15%%)              ║\n");
        else if (mean_wer < 0.50f)
            fprintf(stderr, "║  Verdict:          ★ INTELLIGIBLE (WER < 50%%)        ║\n");
        else
            fprintf(stderr, "║  Verdict:          ✗ NOT INTELLIGIBLE (WER >= 50%%)   ║\n");
    } else {
        fprintf(stderr, "║  Round-trip WER:   (no STT model available)          ║\n");
    }
    fprintf(stderr, "║  WAV files:        bench_output/sonata_quality_*.wav  ║\n");
    fprintf(stderr, "╚═══════════════════════════════════════════════════════╝\n\n");

    fprintf(stderr, "══════════════════════════════════════════\n");
    fprintf(stderr, "Results: %d passed, %d failed, %d skipped\n", g_pass, g_fail, g_skip);
    if (g_fail == 0)
        fprintf(stderr, "ALL PASSED\n\n");
    else
        fprintf(stderr, "SOME FAILURES\n\n");

    /* Cleanup */
    if (stt) conformer_stt_destroy(stt);
    sonata_flow_destroy(flow);
    sonata_lm_destroy(lm);
    spm_destroy(tok);
    free(audio_buf);

    return g_fail > 0 ? 1 : 0;
}
