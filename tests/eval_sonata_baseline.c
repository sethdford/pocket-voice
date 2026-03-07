/**
 * eval_sonata_baseline.c — Sonata TTS evaluation harness.
 *
 * V3 pipeline: Text → Flow V3 → mel → Vocoder → audio (24kHz)
 * Writes WAV files and outputs a JSON report with audio statistics,
 * timings, prosody MOS, and optional round-trip WER via Conformer STT.
 *
 * Usage: ./eval-sonata-baseline [--out-dir <path>] [--report <path>]
 * Run: make eval-generate
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

/* FFI: Audio Quality (from src/quality/) */
extern float prosody_predict_mos(const float *audio, int len, int sr);

/* FFI: Speaker Similarity */
typedef struct { float cosine_sim; float euclidean_dist; int n_coeffs; } SpeakerSimResult;
extern SpeakerSimResult speaker_similarity(const float *audio_a, int len_a,
                                            const float *audio_b, int len_b, int sr);

/* ─── FFI: Sonata Flow V3 (text → mel) ───────────────────────────────── */
extern void *sonata_flow_v3_create(const char *weights, const char *config);
extern void  sonata_flow_v3_destroy(void *e);
extern int   sonata_flow_v3_set_speaker(void *e, int speaker_id);
extern int   sonata_flow_v3_generate(void *e, const char *text, int text_len,
                                      const int *phoneme_ids, int phoneme_len,
                                      int target_frames, float *out_mel, int max_frames);

/* ─── FFI: Sonata Vocoder (mel → audio) ──────────────────────────────── */
extern void *sonata_vocoder_create(const char *weights, const char *config);
extern void  sonata_vocoder_destroy(void *e);
extern int   sonata_vocoder_generate(void *e, const float *mel, int n_frames,
                                      int mel_dim, float *out_audio, int max_samples);

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

/* ─── Constants ───────────────────────────────────────────────────────── */
#define SAMPLE_RATE   24000
#define MEL_DIM       80
#define MAX_MEL_FRAMES 800
#define HOP_LENGTH    480
#define MAX_AUDIO     (MAX_MEL_FRAMES * HOP_LENGTH + 8192)

static mach_timebase_info_data_t g_tb;
static double now_ms(void) {
    if (g_tb.denom == 0) mach_timebase_info(&g_tb);
    return (double)mach_absolute_time() * g_tb.numer / g_tb.denom / 1e6;
}

/* ─── Helper: JSON escape string ──────────────────────────────────────── */
static char *json_escape(const char *src) {
    if (!src) return strdup("\"\"");

    int len = strlen(src);
    char *dst = (char *)malloc(len * 6 + 3);
    if (!dst) return strdup("\"\"");

    char *p = dst;
    *p++ = '"';
    for (int i = 0; i < len; i++) {
        unsigned char c = (unsigned char)src[i];
        if (c == '"') { *p++ = '\\'; *p++ = '"'; }
        else if (c == '\\') { *p++ = '\\'; *p++ = '\\'; }
        else if (c == '\n') { *p++ = '\\'; *p++ = 'n'; }
        else if (c == '\r') { *p++ = '\\'; *p++ = 'r'; }
        else if (c == '\t') { *p++ = '\\'; *p++ = 't'; }
        else if (c < 32) { p += snprintf(p, 8, "\\u%04x", c); }
        else { *p++ = c; }
    }
    *p++ = '"';
    *p = '\0';
    return dst;
}

/* ─── Helper: Write WAV file ──────────────────────────────────────────── */
static void write_wav(const char *path, const float *audio, int n, int sr) {
    FILE *f = fopen(path, "wb");
    if (!f) {
        fprintf(stderr, "Warning: could not open %s for writing\n", path);
        return;
    }

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

/* ─── Helper: Resample 24k to 16k (linear interpolation) ──────────────── */
static float *resample_24k_to_16k(const float *src, int src_n, int *dst_n) {
    double ratio = 24000.0 / 16000.0;
    *dst_n = (int)((double)src_n / ratio);
    if (*dst_n <= 0) return NULL;
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

/* ─── Helper: Audio statistics ────────────────────────────────────────── */
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

    if (n == 0) return s;

    /* Use vDSP for RMS computation */
    float mean = 0.0f;
    vDSP_meanv(audio, 1, &mean, n);

    float *centered = (float *)malloc(n * sizeof(float));
    if (!centered) {
        float sum_sq = 0;
        for (int i = 0; i < n; i++) sum_sq += audio[i] * audio[i];
        s.rms = sqrtf(sum_sq / n);
    } else {
        cblas_scopy(n, audio, 1, centered, 1);
        cblas_saxpy(n, -1.0f, (float[]){mean}, 0, centered, 1);
        float sum_sq = 0;
        vDSP_dotpr(centered, 1, centered, 1, &sum_sq, n);
        s.rms = sqrtf(sum_sq / n);
        free(centered);
    }

    vDSP_maxv(audio, 1, &s.peak, n);
    float neg_peak = 0.0f;
    vDSP_minv(audio, 1, &neg_peak, n);
    neg_peak = fabsf(neg_peak);
    if (neg_peak > s.peak) s.peak = neg_peak;

    int zero_crossings = 0;
    for (int i = 1; i < n; i++) {
        if ((audio[i] >= 0) != (audio[i-1] >= 0)) zero_crossings++;
    }
    s.zero_crossing_rate = (float)zero_crossings / n;

    s.is_silence = (s.rms < 0.001f);

    return s;
}

/* ─── Test sentences (same as eval_comprehensive.py) ───────────────────── */
static const char *test_sentences[] = {
    "The quick brown fox jumps over the lazy dog.",
    "How are you doing today?",
    "I need to schedule a meeting for tomorrow at two PM.",
    "That sounds absolutely wonderful!",
    "The weather is beautiful today, isn't it?",
    "Can you repeat that again please?",
    "We should discuss the quarterly results.",
    "Thank you very much for your help.",
    "Let's meet at the coffee shop around noon.",
    "I'm sorry, I didn't catch that correctly."
};
#define N_TEST_SENTENCES (sizeof(test_sentences) / sizeof(test_sentences[0]))

/* ─── Per-sentence result structure ────────────────────────────────────── */
typedef struct {
    int idx;
    const char *text;
    char wav_path[256];
    int n_mel_frames;
    AudioStats stats;
    double flow_time_ms;
    double vocoder_time_ms;
    float prosody_mos;
    float wer;
    int wer_valid;
} SentenceResult;

/* ─── Main evaluation ─────────────────────────────────────────────────── */
int main(int argc, char *argv[]) {
    char out_dir[256] = "eval/generated";
    char report_path[256] = "eval/reports/c_eval_report.json";

    /* Parse CLI args */
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--out-dir") == 0 && i + 1 < argc) {
            snprintf(out_dir, sizeof(out_dir), "%s", argv[++i]);
        } else if (strcmp(argv[i], "--report") == 0 && i + 1 < argc) {
            snprintf(report_path, sizeof(report_path), "%s", argv[++i]);
        }
    }

    /* Create output directories */
    char mkdir_cmd[512];
    snprintf(mkdir_cmd, sizeof(mkdir_cmd), "mkdir -p %s && mkdir -p $(dirname %s)", out_dir, report_path);
    int ret = system(mkdir_cmd);
    if (ret != 0) {
        fprintf(stderr, "Warning: could not create output directories\n");
    }

    fprintf(stderr, "\n═══ Sonata TTS V3 Evaluation Harness ═══\n");
    fprintf(stderr, "Output dir: %s\n", out_dir);
    fprintf(stderr, "Report: %s\n", report_path);

    /* Load Flow V3 (text → mel) */
    fprintf(stderr, "\nLoading Sonata Flow V3...\n");
    void *flow = sonata_flow_v3_create("models/sonata/sonata_flow.safetensors",
                                        "models/sonata/sonata_flow_config.json");
    if (!flow) {
        fprintf(stderr, "Fatal: could not load Sonata Flow V3\n");
        return 1;
    }

    /* Set default speaker */
    sonata_flow_v3_set_speaker(flow, 0);

    /* Load Vocoder (mel → audio) */
    fprintf(stderr, "Loading Sonata Vocoder...\n");
    void *vocoder = sonata_vocoder_create("models/sonata/sonata_vocoder.safetensors",
                                           "models/sonata/sonata_vocoder_config.json");
    if (!vocoder) {
        fprintf(stderr, "Fatal: could not load Sonata Vocoder\n");
        sonata_flow_v3_destroy(flow);
        return 1;
    }

    /* Optionally load Conformer STT (for WER) */
    ConformerSTT *stt = NULL;
    fprintf(stderr, "Attempting to load Conformer STT (optional)...\n");
    stt = conformer_stt_create("models/parakeet-ctc-1.1b-fp16.cstt");
    if (!stt) {
        fprintf(stderr, "Note: Conformer STT not available, WER will be skipped\n");
    }

    /* Allocate result arrays and buffers */
    SentenceResult *results = (SentenceResult *)calloc(N_TEST_SENTENCES, sizeof(SentenceResult));
    if (!results) {
        fprintf(stderr, "Fatal: could not allocate results\n");
        return 1;
    }

    float *mel_buf = (float *)malloc(MAX_MEL_FRAMES * MEL_DIM * sizeof(float));
    float *audio_buf = (float *)malloc(MAX_AUDIO * sizeof(float));
    if (!mel_buf || !audio_buf) {
        fprintf(stderr, "Fatal: could not allocate buffers\n");
        return 1;
    }

    /* Process each test sentence */
    fprintf(stderr, "\nProcessing %d test sentences...\n", (int)N_TEST_SENTENCES);

    double total_flow_time = 0.0, total_vocoder_time = 0.0;
    double total_audio_duration = 0.0;
    float total_prosody_mos = 0.0;
    float total_wer = 0.0;
    int wer_count = 0;

    for (int idx = 0; idx < (int)N_TEST_SENTENCES; idx++) {
        SentenceResult *res = &results[idx];
        res->idx = idx;
        res->text = test_sentences[idx];
        res->wer_valid = 0;
        snprintf(res->wav_path, sizeof(res->wav_path), "%s/eval_%04d.wav", out_dir, idx);

        fprintf(stderr, "[%d/%d] \"%s\"\n", idx + 1, (int)N_TEST_SENTENCES, res->text);

        int text_len = (int)strlen(res->text);

        /* Flow V3: text → mel */
        double flow_start = now_ms();

        memset(mel_buf, 0, MAX_MEL_FRAMES * MEL_DIM * sizeof(float));
        int n_frames = sonata_flow_v3_generate(flow,
                                                res->text, text_len,
                                                NULL, 0,  /* no phoneme override */
                                                0,        /* auto target frames */
                                                mel_buf, MAX_MEL_FRAMES);

        double flow_end = now_ms();
        res->flow_time_ms = flow_end - flow_start;
        total_flow_time += res->flow_time_ms;
        res->n_mel_frames = n_frames;

        if (n_frames <= 0) {
            fprintf(stderr, "  Error: flow generation failed (returned %d)\n", n_frames);
            continue;
        }
        fprintf(stderr, "  Mel frames: %d, Flow time: %.2f ms\n", n_frames, res->flow_time_ms);

        /* Vocoder: mel → audio */
        double voc_start = now_ms();

        memset(audio_buf, 0, MAX_AUDIO * sizeof(float));
        int n_samples = sonata_vocoder_generate(vocoder,
                                                 mel_buf, n_frames, MEL_DIM,
                                                 audio_buf, MAX_AUDIO);

        double voc_end = now_ms();
        res->vocoder_time_ms = voc_end - voc_start;
        total_vocoder_time += res->vocoder_time_ms;

        if (n_samples <= 0) {
            fprintf(stderr, "  Error: vocoder generation failed\n");
            continue;
        }
        fprintf(stderr, "  Audio samples: %d, Vocoder time: %.2f ms\n", n_samples, res->vocoder_time_ms);

        /* Write WAV */
        write_wav(res->wav_path, audio_buf, n_samples, SAMPLE_RATE);
        fprintf(stderr, "  WAV: %s\n", res->wav_path);

        /* Compute audio stats */
        res->stats = compute_stats(audio_buf, n_samples, SAMPLE_RATE);
        total_audio_duration += res->stats.duration_s;
        fprintf(stderr, "  Stats: RMS=%.4f, peak=%.4f, duration=%.2fs\n",
                res->stats.rms, res->stats.peak, res->stats.duration_s);

        /* Prosody MOS */
        res->prosody_mos = prosody_predict_mos(audio_buf, n_samples, SAMPLE_RATE);
        total_prosody_mos += res->prosody_mos;
        fprintf(stderr, "  Prosody MOS: %.2f\n", res->prosody_mos);

        /* Optional: round-trip WER via STT */
        if (stt) {
            int resampled_n = 0;
            float *resampled = resample_24k_to_16k(audio_buf, n_samples, &resampled_n);
            if (resampled) {
                conformer_stt_reset(stt);
                conformer_stt_process(stt, resampled, resampled_n);
                conformer_stt_flush(stt);

                char transcribed[512] = {0};
                conformer_stt_get_text(stt, transcribed, sizeof(transcribed));

                WERResult wer_result = wer_compute(res->text, transcribed);
                res->wer = wer_result.wer;
                res->wer_valid = 1;
                total_wer += res->wer;
                wer_count++;
                fprintf(stderr, "  STT: \"%s\"\n", transcribed);
                fprintf(stderr, "  WER: %.2f%%\n", res->wer * 100.0f);

                free(resampled);
            }
        }
    }

    /* ---- Speaker consistency: pairwise similarity ---- */
    float mean_speaker_consistency = -1.0f;
    if (N_TEST_SENTENCES >= 2) {
        char ref_wav[512];
        snprintf(ref_wav, sizeof(ref_wav), "%s/eval_0000.wav", out_dir);
        FILE *rf = fopen(ref_wav, "rb");
        if (rf) {
            fseek(rf, 0, SEEK_END);
            long fsz = ftell(rf);
            int ref_n = (int)((fsz - 44) / 2);
            fseek(rf, 44, SEEK_SET);
            float *ref_audio = (float *)malloc(ref_n * sizeof(float));
            int16_t *tmp16 = (int16_t *)malloc(ref_n * sizeof(int16_t));
            if (ref_audio && tmp16) {
                fread(tmp16, 2, ref_n, rf);
                for (int j = 0; j < ref_n; j++) ref_audio[j] = tmp16[j] / 32768.0f;
                free(tmp16);

                double sim_sum = 0;
                int sim_count = 0;
                for (int k = 1; k < (int)N_TEST_SENTENCES; k++) {
                    char cmp_path[512];
                    snprintf(cmp_path, sizeof(cmp_path), "%s/eval_%04d.wav", out_dir, k);
                    FILE *cf = fopen(cmp_path, "rb");
                    if (!cf) continue;
                    fseek(cf, 0, SEEK_END);
                    long csz = ftell(cf);
                    int cmp_n = (int)((csz - 44) / 2);
                    fseek(cf, 44, SEEK_SET);
                    float *cmp_audio = (float *)malloc(cmp_n * sizeof(float));
                    int16_t *ctmp = (int16_t *)malloc(cmp_n * sizeof(int16_t));
                    if (cmp_audio && ctmp) {
                        fread(ctmp, 2, cmp_n, cf);
                        for (int j = 0; j < cmp_n; j++) cmp_audio[j] = ctmp[j] / 32768.0f;
                        SpeakerSimResult ssr = speaker_similarity(
                            ref_audio, ref_n, cmp_audio, cmp_n, SAMPLE_RATE);
                        sim_sum += ssr.cosine_sim;
                        sim_count++;
                    }
                    if (ctmp) free(ctmp);
                    if (cmp_audio) free(cmp_audio);
                    fclose(cf);
                }
                if (sim_count > 0) {
                    mean_speaker_consistency = (float)(sim_sum / sim_count);
                }
                free(ref_audio);
            } else {
                if (ref_audio) free(ref_audio);
                if (tmp16) free(tmp16);
            }
            fclose(rf);
        }
    }
    fprintf(stderr, "Speaker consistency: %.3f\n", mean_speaker_consistency);

    /* Generate JSON report */
    FILE *report_f = fopen(report_path, "w");
    if (!report_f) {
        fprintf(stderr, "Error: could not open report file %s\n", report_path);
        return 1;
    }

    fprintf(report_f, "{\n");
    fprintf(report_f, "  \"timestamp\": \"2026-03-07\",\n");
    fprintf(report_f, "  \"system\": \"Sonata TTS V3 (Flow V3 + Vocoder, Metal)\",\n");
    fprintf(report_f, "  \"model_path\": \"models/sonata/\",\n");
    fprintf(report_f, "  \"sentences\": [\n");

    for (int idx = 0; idx < (int)N_TEST_SENTENCES; idx++) {
        SentenceResult *res = &results[idx];

        char *text_json = json_escape(res->text);
        char *path_json = json_escape(res->wav_path);

        fprintf(report_f, "    {\n");
        fprintf(report_f, "      \"index\": %d,\n", res->idx);
        fprintf(report_f, "      \"text\": %s,\n", text_json);
        fprintf(report_f, "      \"wav_path\": %s,\n", path_json);
        fprintf(report_f, "      \"n_mel_frames\": %d,\n", res->n_mel_frames);
        fprintf(report_f, "      \"audio_stats\": {\n");
        fprintf(report_f, "        \"rms\": %.6f,\n", res->stats.rms);
        fprintf(report_f, "        \"peak\": %.6f,\n", res->stats.peak);
        fprintf(report_f, "        \"zero_crossing_rate\": %.6f,\n", res->stats.zero_crossing_rate);
        fprintf(report_f, "        \"duration_s\": %.6f,\n", res->stats.duration_s);
        fprintf(report_f, "        \"n_samples\": %d,\n", res->stats.n_samples);
        fprintf(report_f, "        \"is_silence\": %d\n", res->stats.is_silence);
        fprintf(report_f, "      },\n");
        fprintf(report_f, "      \"timings_ms\": {\n");
        fprintf(report_f, "        \"flow_inference\": %.2f,\n", res->flow_time_ms);
        fprintf(report_f, "        \"vocoder_inference\": %.2f\n", res->vocoder_time_ms);
        fprintf(report_f, "      },\n");
        fprintf(report_f, "      \"prosody_mos\": %.2f,\n", res->prosody_mos);

        if (res->wer_valid) {
            fprintf(report_f, "      \"wer\": %.4f\n", res->wer);
        } else {
            fprintf(report_f, "      \"wer\": null\n");
        }

        fprintf(report_f, "    }%s\n", (idx < (int)N_TEST_SENTENCES - 1) ? "," : "");

        free(text_json);
        free(path_json);
    }

    fprintf(report_f, "  ],\n");

    /* Aggregate metrics */
    fprintf(report_f, "  \"aggregate\": {\n");
    fprintf(report_f, "    \"mean_rtf\": %.4f,\n",
            total_audio_duration > 0.0 ?
            (total_flow_time + total_vocoder_time) / 1000.0 / total_audio_duration : 0.0);
    fprintf(report_f, "    \"mean_flow_time_ms\": %.2f,\n",
            total_flow_time / (double)N_TEST_SENTENCES);
    fprintf(report_f, "    \"mean_vocoder_time_ms\": %.2f,\n",
            total_vocoder_time / (double)N_TEST_SENTENCES);
    fprintf(report_f, "    \"mean_prosody_mos\": %.2f,\n",
            total_prosody_mos / (double)N_TEST_SENTENCES);
    fprintf(report_f, "    \"mean_speaker_consistency\": %.4f,\n", mean_speaker_consistency);

    if (wer_count > 0) {
        fprintf(report_f, "    \"mean_wer\": %.4f,\n", total_wer / wer_count);
        fprintf(report_f, "    \"n_wer_evaluated\": %d\n", wer_count);
    } else {
        fprintf(report_f, "    \"mean_wer\": null,\n");
        fprintf(report_f, "    \"n_wer_evaluated\": 0\n");
    }

    fprintf(report_f, "  },\n");

    /* SOTA targets */
    fprintf(report_f, "  \"sota_targets\": {\n");
    fprintf(report_f, "    \"prosody_mos\": 4.5,\n");
    fprintf(report_f, "    \"mean_wer\": 0.03,\n");
    fprintf(report_f, "    \"mean_rtf\": 0.3\n");
    fprintf(report_f, "  }\n");

    fprintf(report_f, "}\n");
    fclose(report_f);

    fprintf(stderr, "\nReport written to: %s\n", report_path);

    /* Summary */
    fprintf(stderr, "\n═══ Summary ═══\n");
    fprintf(stderr, "Mean Flow time: %.2f ms\n", total_flow_time / (double)N_TEST_SENTENCES);
    fprintf(stderr, "Mean Vocoder time: %.2f ms\n", total_vocoder_time / (double)N_TEST_SENTENCES);
    fprintf(stderr, "Mean RTF: %.4f\n",
            total_audio_duration > 0.0 ?
            (total_flow_time + total_vocoder_time) / 1000.0 / total_audio_duration : 0.0);
    fprintf(stderr, "Mean Prosody MOS: %.2f\n", total_prosody_mos / (double)N_TEST_SENTENCES);
    if (wer_count > 0) {
        fprintf(stderr, "Mean WER: %.4f (%d evaluated)\n", total_wer / wer_count, wer_count);
    }

    /* Cleanup */
    free(results);
    free(mel_buf);
    free(audio_buf);

    if (stt) conformer_stt_destroy(stt);
    sonata_vocoder_destroy(vocoder);
    sonata_flow_v3_destroy(flow);

    fprintf(stderr, "\nEvaluation complete.\n");
    return 0;
}
