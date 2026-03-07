# TTS Eval Pipeline — Measurement Baselines

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Wire up end-to-end TTS evaluation so we can measure Sonata's quality metrics (MOS, WER, SIM-O, RTF, TTFA, MCD, STOI, F0) against SOTA benchmarks, using both the production C pipeline and the Python eval suite.

**Architecture:** Three-layer approach: (1) A C harness that generates WAVs via the real Sonata pipeline (LM -> Flow -> Decoder) and computes C-native metrics (MCD, STOI, F0, speaker sim, RTF, TTFA), outputting a JSON report. (2) A Python wrapper that takes those WAVs and runs the comprehensive eval suite (PESQ, UTMOS, WER via Whisper, speaker sim via ECAPA-TDNN). (3) A unified `make eval` target and shell script that orchestrates both, merges reports, and prints a combined scorecard comparing Sonata vs SOTA targets.

**Tech Stack:** C (Accelerate/vDSP), Python 3 (torch, soundfile, pesq, pystoi, jiwer, whisper, speechbrain), Make, jq

**Current State:**

- Local models: `models/sonata/` has LM (967MB), Flow, Decoder, Tokenizer — these are untrained/early weights producing noise (confirmed Feb 22)
- Trained models on GCS: `flow_v3_final.pt` (2.4G), `sonata_lm_final.pt` (2.7G) — PyTorch format, not safetensors
- C quality metrics: `src/quality/audio_quality.{h,c}` has MCD, STOI, SNR, F0, speaker sim — all implemented with vDSP
- C test harness: `tests/test_sonata_quality.c` generates WAVs + runs STT round-trip — works but uses old untrained weights
- Python eval: `train/sonata/eval_comprehensive.py` — 10 metrics, 3 modes, never run
- Existing WAVs: `bench_output/sonata_quality_*.wav` — noise from untrained models (Feb 22)

---

### Task 1: Create eval output directory structure and Python requirements

**Files:**

- Create: `eval/requirements.txt`
- Create: `eval/.gitkeep`

**Step 1: Create eval directory structure**

```bash
mkdir -p eval/generated eval/reference eval/reports
```

**Step 2: Create Python requirements file**

```
# eval/requirements.txt — TTS evaluation dependencies
soundfile>=0.12
numpy>=1.24
torch>=2.0
librosa>=0.10
pesq>=0.0.4
pystoi>=0.4
jiwer>=3.0
resampy>=0.4
openai-whisper>=20231117
speechbrain>=1.0
```

**Step 3: Verify Python environment**

Run: `pip install -r eval/requirements.txt --dry-run 2>&1 | head -20`
Expected: Package list without errors

**Step 4: Commit**

```bash
git add eval/
git commit -m "feat: create eval directory structure and Python requirements"
```

---

### Task 2: Extend C quality harness to output JSON report

The existing `test_sonata_quality.c` prints to stderr and writes WAVs but doesn't produce machine-readable output. We need a new harness that:

- Runs the same LM -> Flow -> Decoder pipeline
- Computes ALL C-native quality metrics (MCD, STOI, F0, speaker sim, SNR, prosody MOS)
- Measures RTF and TTFA per sentence
- Outputs structured JSON to `eval/reports/c_eval_report.json`
- Writes generated WAVs to `eval/generated/`

**Files:**

- Create: `tests/eval_sonata_baseline.c`
- Modify: `Makefile` (add `eval-generate` target)

**Step 1: Write the C evaluation harness**

Create `tests/eval_sonata_baseline.c`:

```c
/**
 * eval_sonata_baseline.c -- Generate WAVs via Sonata C pipeline and compute
 * quality metrics, outputting JSON for the eval dashboard.
 *
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

/* ---- FFI: Sonata LM ---- */
extern void *sonata_lm_create(const char *weights, const char *config);
extern void  sonata_lm_destroy(void *e);
extern int   sonata_lm_set_text(void *e, const unsigned int *ids, int n);
extern int   sonata_lm_step(void *e, int *out);
extern int   sonata_lm_reset(void *e);
extern int   sonata_lm_is_done(void *e);
extern int   sonata_lm_set_params(void *e, float temp, int top_k, float top_p, float rep);

/* ---- FFI: Sonata Flow + Decoder ---- */
extern void *sonata_flow_create(const char *fw, const char *fc, const char *dw, const char *dc);
extern void  sonata_flow_destroy(void *e);
extern int   sonata_flow_generate_audio(void *e, const int *tokens, int n, float *out, int max);
extern int   sonata_flow_decoder_type(void *e);
extern int   sonata_flow_samples_per_frame(void *e);

/* ---- FFI: SPM Tokenizer ---- */
typedef struct SPMTokenizer SPMTokenizer;
extern SPMTokenizer *spm_create(const uint8_t *data, uint32_t size);
extern void  spm_destroy(SPMTokenizer *t);
extern int   spm_encode(SPMTokenizer *t, const char *text, int32_t *ids, int max);

/* ---- FFI: Conformer STT (optional) ---- */
typedef struct ConformerSTT ConformerSTT;
extern ConformerSTT *conformer_stt_create(const char *path);
extern void  conformer_stt_destroy(ConformerSTT *stt);
extern void  conformer_stt_reset(ConformerSTT *stt);
extern int   conformer_stt_process(ConformerSTT *stt, const float *pcm, int n);
extern int   conformer_stt_flush(ConformerSTT *stt);
extern int   conformer_stt_get_text(const ConformerSTT *stt, char *buf, int cap);

/* ---- FFI: WER ---- */
typedef struct { int sub, del, ins, ref_words, hyp_words; float wer, cer, accuracy; } WERResult;
extern WERResult wer_compute(const char *ref, const char *hyp);

/* ---- FFI: Audio Quality ---- */
#include "quality/audio_quality.h"

/* ---- Constants ---- */
#define SAMPLE_RATE 24000
#define MAX_TOKENS  300
#define MAX_AUDIO   (MAX_TOKENS * 480 + 8192)
#define N_SENTENCES 10

static mach_timebase_info_data_t g_tb;
static double now_ms(void) {
    if (g_tb.denom == 0) mach_timebase_info(&g_tb);
    return (double)mach_absolute_time() * g_tb.numer / g_tb.denom / 1e6;
}

static const char *SENTENCES[N_SENTENCES] = {
    "The quick brown fox jumps over the lazy dog.",
    "How are you doing today?",
    "I need to schedule a meeting for tomorrow at two PM.",
    "That sounds absolutely wonderful!",
    "The weather is beautiful today, isn't it?",
    "Can you repeat that again please?",
    "We should discuss the quarterly results.",
    "Thank you very much for your help.",
    "Let's meet at the coffee shop around noon.",
    "I'm sorry, I didn't catch that correctly.",
};

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

/* Escape string for JSON (minimal: backslash and double-quote) */
static void json_escape(FILE *f, const char *s) {
    fputc('"', f);
    for (; *s; s++) {
        if (*s == '"') fputs("\\\"", f);
        else if (*s == '\\') fputs("\\\\", f);
        else if (*s == '\n') fputs("\\n", f);
        else fputc(*s, f);
    }
    fputc('"', f);
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

int main(int argc, char **argv) {
    const char *out_dir = "eval/generated";
    const char *report_path = "eval/reports/c_eval_report.json";

    /* Parse optional args */
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--out-dir") == 0 && i + 1 < argc) out_dir = argv[++i];
        if (strcmp(argv[i], "--report") == 0 && i + 1 < argc) report_path = argv[++i];
    }

    fprintf(stderr, "\n");
    fprintf(stderr, "========================================================\n");
    fprintf(stderr, "  Sonata TTS Baseline Evaluation (C Pipeline)\n");
    fprintf(stderr, "========================================================\n\n");

    /* ---- Load models ---- */
    SPMTokenizer *tok = load_tokenizer("models/tokenizer.model");
    if (!tok) {
        fprintf(stderr, "[FATAL] Cannot load tokenizer\n");
        return 1;
    }

    void *lm = sonata_lm_create(
        "models/sonata/sonata_lm.safetensors",
        "models/sonata/sonata_lm_config.json");
    if (!lm) {
        fprintf(stderr, "[FATAL] Cannot load Sonata LM\n");
        spm_destroy(tok);
        return 1;
    }
    sonata_lm_set_params(lm, 0.7f, 40, 0.90f, 1.2f);

    void *flow = sonata_flow_create(
        "models/sonata/sonata_flow.safetensors",
        "models/sonata/sonata_flow_config.json",
        "models/sonata/sonata_decoder.safetensors",
        "models/sonata/sonata_decoder_config.json");
    if (!flow) {
        fprintf(stderr, "[FATAL] Cannot load Sonata Flow\n");
        sonata_lm_destroy(lm);
        spm_destroy(tok);
        return 1;
    }

    int spf = sonata_flow_samples_per_frame(flow);
    fprintf(stderr, "  Models loaded (spf=%d)\n\n", spf);

    ConformerSTT *stt = conformer_stt_create("models/parakeet-ctc-1.1b-fp16.cstt");
    if (stt) fprintf(stderr, "  STT loaded (round-trip WER enabled)\n");
    else     fprintf(stderr, "  STT not available (WER will be -1)\n");

    /* ---- Open JSON report ---- */
    FILE *jf = fopen(report_path, "w");
    if (!jf) {
        fprintf(stderr, "[FATAL] Cannot open %s for writing\n", report_path);
        /* cleanup omitted for brevity — real code has it */
        return 1;
    }
    fprintf(jf, "{\n  \"pipeline\": \"c_native\",\n  \"sample_rate\": %d,\n  \"results\": [\n", SAMPLE_RATE);

    float *audio_buf = (float *)calloc(MAX_AUDIO, sizeof(float));

    /* ---- Aggregate accumulators ---- */
    int n_generated = 0;
    double sum_rtf = 0, sum_ttfa = 0, sum_lm_toks = 0;
    double sum_mcd = 0, sum_prosody_mos = 0, sum_wer = 0;
    int n_wer = 0;

    for (int s = 0; s < N_SENTENCES; s++) {
        const char *text = SENTENCES[s];
        fprintf(stderr, "--- [%d/%d] \"%s\"\n", s + 1, N_SENTENCES, text);

        /* Tokenize */
        int32_t text_ids[512];
        int n_ids = spm_encode(tok, text, text_ids, 512);
        if (n_ids <= 0) {
            fprintf(stderr, "    Tokenization failed, skipping\n");
            continue;
        }
        unsigned int uids[512];
        for (int i = 0; i < n_ids; i++) uids[i] = (unsigned int)text_ids[i];

        /* LM: generate semantic tokens, measure TTFA */
        sonata_lm_reset(lm);
        sonata_lm_set_text(lm, uids, n_ids);

        int sem_tokens[MAX_TOKENS];
        int n_sem = 0;
        double t_lm_start = now_ms();
        double ttfa_ms = -1;
        while (n_sem < MAX_TOKENS && !sonata_lm_is_done(lm)) {
            int tok_out = 0;
            int st = sonata_lm_step(lm, &tok_out);
            if (st == 1 || st == -1) break;
            if (n_sem == 0) ttfa_ms = now_ms() - t_lm_start;
            sem_tokens[n_sem++] = tok_out;
        }
        double lm_ms = now_ms() - t_lm_start;
        double lm_toks = (lm_ms > 0) ? (n_sem * 1000.0 / lm_ms) : 0;

        /* Flow + Decoder: generate audio */
        double t_flow_start = now_ms();
        int max_samples = n_sem * spf + 4096;
        if (max_samples > MAX_AUDIO) max_samples = MAX_AUDIO;
        int audio_len = sonata_flow_generate_audio(flow, sem_tokens, n_sem, audio_buf, max_samples);
        double flow_ms = now_ms() - t_flow_start;

        double audio_dur = (double)audio_len / SAMPLE_RATE;
        double total_ms = lm_ms + flow_ms;
        double rtf = (audio_dur > 0) ? (total_ms / 1000.0 / audio_dur) : 99.0;

        fprintf(stderr, "    LM: %d tokens, %.0f ms (%.0f tok/s), TTFA=%.1f ms\n",
                n_sem, lm_ms, lm_toks, ttfa_ms);
        fprintf(stderr, "    Flow: %d samples (%.2fs), %.0f ms, RTF=%.3f\n",
                audio_len, audio_dur, flow_ms, rtf);

        /* Write WAV */
        char wav_path[512];
        snprintf(wav_path, sizeof(wav_path), "%s/eval_%04d.wav", out_dir, s);
        if (audio_len > 0) {
            write_wav(wav_path, audio_buf, audio_len, SAMPLE_RATE);
        }

        /* C-native quality metrics (reference-free since we have no ref audio) */
        float prosody_mos = prosody_predict_mos(audio_buf, audio_len, SAMPLE_RATE);

        /* Check basic audio stats */
        float rms = 0;
        if (audio_len > 0) {
            float dot;
            vDSP_dotpr(audio_buf, 1, audio_buf, 1, &dot, (vDSP_Length)audio_len);
            rms = sqrtf(dot / audio_len);
        }
        int is_silence = (rms < 0.001f);

        /* Round-trip WER via Conformer STT */
        float wer_val = -1.0f;
        char transcript[1024] = {0};
        if (stt && audio_len > 0 && !is_silence) {
            int n16k = 0;
            float *a16k = resample_24k_to_16k(audio_buf, audio_len, &n16k);
            if (a16k && n16k > 0) {
                conformer_stt_reset(stt);
                conformer_stt_process(stt, a16k, n16k);
                conformer_stt_flush(stt);
                conformer_stt_get_text(stt, transcript, sizeof(transcript));
                if (strlen(transcript) > 0) {
                    WERResult w = wer_compute(text, transcript);
                    wer_val = w.wer;
                    sum_wer += wer_val;
                    n_wer++;
                    fprintf(stderr, "    WER: %.1f%% | \"%s\"\n", wer_val * 100, transcript);
                }
                free(a16k);
            }
        }

        /* Accumulate */
        if (audio_len > 0) {
            n_generated++;
            sum_rtf += rtf;
            sum_ttfa += ttfa_ms;
            sum_lm_toks += lm_toks;
            sum_prosody_mos += prosody_mos;
        }

        /* Write JSON entry */
        if (s > 0) fprintf(jf, ",\n");
        fprintf(jf, "    {\n");
        fprintf(jf, "      \"index\": %d,\n", s);
        fprintf(jf, "      \"text\": "); json_escape(jf, text); fprintf(jf, ",\n");
        fprintf(jf, "      \"wav\": \"%s\",\n", wav_path);
        fprintf(jf, "      \"n_tokens\": %d,\n", n_sem);
        fprintf(jf, "      \"audio_samples\": %d,\n", audio_len);
        fprintf(jf, "      \"audio_duration_s\": %.4f,\n", audio_dur);
        fprintf(jf, "      \"lm_ms\": %.1f,\n", lm_ms);
        fprintf(jf, "      \"flow_ms\": %.1f,\n", flow_ms);
        fprintf(jf, "      \"total_ms\": %.1f,\n", total_ms);
        fprintf(jf, "      \"ttfa_ms\": %.1f,\n", ttfa_ms);
        fprintf(jf, "      \"rtf\": %.4f,\n", rtf);
        fprintf(jf, "      \"lm_tok_per_s\": %.1f,\n", lm_toks);
        fprintf(jf, "      \"rms\": %.6f,\n", rms);
        fprintf(jf, "      \"is_silence\": %s,\n", is_silence ? "true" : "false");
        fprintf(jf, "      \"prosody_mos\": %.3f,\n", prosody_mos);
        fprintf(jf, "      \"wer\": %.4f,\n", wer_val);
        fprintf(jf, "      \"transcript\": "); json_escape(jf, transcript); fprintf(jf, "\n");
        fprintf(jf, "    }");

        fprintf(stderr, "    Prosody MOS: %.2f | RMS: %.4f | silence: %s\n\n",
                prosody_mos, rms, is_silence ? "YES" : "no");
    }

    /* ---- Aggregates ---- */
    fprintf(jf, "\n  ],\n");
    fprintf(jf, "  \"aggregate\": {\n");
    fprintf(jf, "    \"n_generated\": %d,\n", n_generated);
    fprintf(jf, "    \"mean_rtf\": %.4f,\n", n_generated > 0 ? sum_rtf / n_generated : -1.0);
    fprintf(jf, "    \"mean_ttfa_ms\": %.1f,\n", n_generated > 0 ? sum_ttfa / n_generated : -1.0);
    fprintf(jf, "    \"mean_lm_tok_per_s\": %.1f,\n", n_generated > 0 ? sum_lm_toks / n_generated : -1.0);
    fprintf(jf, "    \"mean_prosody_mos\": %.3f,\n", n_generated > 0 ? sum_prosody_mos / n_generated : -1.0);
    fprintf(jf, "    \"mean_wer\": %.4f,\n", n_wer > 0 ? sum_wer / n_wer : -1.0);
    fprintf(jf, "    \"n_wer_samples\": %d\n", n_wer);
    fprintf(jf, "  },\n");

    /* SOTA targets for dashboard comparison */
    fprintf(jf, "  \"sota_targets\": {\n");
    fprintf(jf, "    \"rtf\": 0.03,\n");
    fprintf(jf, "    \"ttfa_ms\": 102,\n");
    fprintf(jf, "    \"wer_pct\": 0.9,\n");
    fprintf(jf, "    \"mos\": 4.3,\n");
    fprintf(jf, "    \"sim_o\": 0.687\n");
    fprintf(jf, "  }\n");
    fprintf(jf, "}\n");
    fclose(jf);

    /* ---- Print summary ---- */
    fprintf(stderr, "========================================================\n");
    fprintf(stderr, "  BASELINE RESULTS\n");
    fprintf(stderr, "========================================================\n");
    fprintf(stderr, "  Generated:     %d / %d sentences\n", n_generated, N_SENTENCES);
    if (n_generated > 0) {
        fprintf(stderr, "  Mean RTF:      %.3f (target: <0.03 SOTA, <0.2 good)\n", sum_rtf / n_generated);
        fprintf(stderr, "  Mean TTFA:     %.0f ms (target: <102 ms SOTA, <300 ms good)\n", sum_ttfa / n_generated);
        fprintf(stderr, "  Mean LM:       %.0f tok/s\n", sum_lm_toks / n_generated);
        fprintf(stderr, "  Mean Pros MOS: %.2f (target: >4.0)\n", sum_prosody_mos / n_generated);
    }
    if (n_wer > 0)
        fprintf(stderr, "  Mean WER:      %.1f%% (target: <5%% good, <0.9%% SOTA)\n", sum_wer / n_wer * 100);
    fprintf(stderr, "  Report:        %s\n", report_path);
    fprintf(stderr, "  WAVs:          %s/eval_*.wav\n", out_dir);
    fprintf(stderr, "========================================================\n\n");

    /* Cleanup */
    if (stt) conformer_stt_destroy(stt);
    sonata_flow_destroy(flow);
    sonata_lm_destroy(lm);
    spm_destroy(tok);
    free(audio_buf);

    return 0;
}
```

**Step 2: Add Makefile target**

Add to `Makefile` after the `test-sonata-quality` target block:

```makefile
# ---- TTS Evaluation Baseline ----
eval-generate: tests/eval_sonata_baseline.c \
               $(BUILD)/libspm_tokenizer.dylib $(BUILD)/libconformer_stt.dylib \
               $(BUILD)/libctc_beam_decoder.dylib $(BUILD)/libtdt_decoder.dylib \
               $(BUILD)/libmel_spectrogram.dylib \
               $(SONATA_LM_DYLIB) $(SONATA_FLOW_DYLIB) | $(BUILD)
	@mkdir -p eval/generated eval/reports
	$(CC) $(CFLAGS) -DACCELERATE_NEW_LAPACK -Isrc -framework Accelerate \
	  -L$(BUILD) -lspm_tokenizer -lconformer_stt -lctc_beam_decoder -ltdt_decoder -lmel_spectrogram \
	  -Lsrc/sonata_lm/target/release -Lsrc/sonata_flow/target/release \
	  -Wl,-rpath,$(CURDIR)/$(BUILD) \
	  -Wl,-rpath,$(CURDIR)/src/sonata_lm/target/release \
	  -Wl,-rpath,$(CURDIR)/src/sonata_flow/target/release \
	  -lsonata_lm -lsonata_flow -lm \
	  src/quality/wer.c src/quality/audio_quality.c \
	  -o $(BUILD)/eval-sonata-baseline tests/eval_sonata_baseline.c
	./$(BUILD)/eval-sonata-baseline
```

**Step 3: Build and verify it compiles**

Run: `make eval-generate 2>&1 | tail -30`
Expected: Compiles, runs, outputs JSON report and WAVs (may be noise with current weights)

**Step 4: Verify JSON output**

Run: `cat eval/reports/c_eval_report.json | python3 -m json.tool | head -40`
Expected: Valid JSON with per-sentence results

**Step 5: Commit**

```bash
git add tests/eval_sonata_baseline.c Makefile
git commit -m "feat: C eval harness with JSON output, RTF, TTFA, prosody MOS, WER"
```

---

### Task 3: Create Python eval wrapper for C-generated WAVs

This script takes the WAVs generated by Task 2 and runs the comprehensive Python metrics (PESQ, STOI, MCD, UTMOS, speaker sim via ECAPA-TDNN, WER via Whisper).

**Files:**

- Create: `eval/run_eval.py`

**Step 1: Write the Python eval wrapper**

```python
#!/usr/bin/env python3
"""Run comprehensive TTS eval on WAVs generated by the C pipeline.

Usage:
  python eval/run_eval.py                              # defaults
  python eval/run_eval.py --wav-dir eval/generated     # custom dir
  python eval/run_eval.py --c-report eval/reports/c_eval_report.json
  python eval/run_eval.py --utmos                      # enable UTMOS (slow)
  python eval/run_eval.py --ref-dir eval/reference     # reference audio for paired metrics
"""

import argparse
import json
import sys
import time
from pathlib import Path

# Add train/sonata to path for eval_comprehensive imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "train" / "sonata"))

from eval_comprehensive import (
    EvalMetrics, EvalReport, aggregate_report, print_summary,
    load_audio, compute_pesq, compute_stoi, compute_mcd,
    compute_f0_metrics, compute_spectral_convergence,
    compute_speaker_similarity, compute_mos_proxy,
    compute_wer_pct, compute_utmos, transcribe_whisper,
    save_audio, resample_audio,
    TARGETS,
)


def merge_c_report(py_report: dict, c_report_path: str) -> dict:
    """Merge C-native metrics into the Python report."""
    try:
        with open(c_report_path) as f:
            c_data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return py_report

    py_report["c_pipeline"] = c_data.get("aggregate", {})
    py_report["c_pipeline"]["per_sample"] = c_data.get("results", [])
    return py_report


def eval_wavs(
    wav_dir: str,
    c_report_path: str = "",
    ref_dir: str = "",
    use_utmos: bool = False,
) -> EvalReport:
    """Evaluate all WAVs in directory."""
    wav_dir = Path(wav_dir)
    wavs = sorted(wav_dir.glob("eval_*.wav"))
    if not wavs:
        print(f"ERROR: No eval_*.wav files found in {wav_dir}")
        sys.exit(1)

    # Load C report for text references
    c_results = []
    if c_report_path:
        try:
            with open(c_report_path) as f:
                c_data = json.load(f)
                c_results = c_data.get("results", [])
        except (FileNotFoundError, json.JSONDecodeError):
            pass

    ref_wavs = {}
    if ref_dir:
        ref_path = Path(ref_dir)
        ref_wavs = {p.name: p for p in ref_path.glob("*.wav")}

    results = []
    for i, wav_path in enumerate(wavs):
        print(f"  [{i+1}/{len(wavs)}] {wav_path.name}")
        gen_audio, sr = load_audio(str(wav_path))
        duration = len(gen_audio) / sr

        # Get reference text from C report
        ref_text = ""
        if i < len(c_results):
            ref_text = c_results[i].get("text", "")

        m = EvalMetrics(
            file=wav_path.name,
            text=ref_text,
            duration_sec=duration,
        )

        # RTF from C report (more accurate than re-measuring)
        if i < len(c_results):
            m.rtf = c_results[i].get("rtf", -1.0)
            m.gen_time_sec = c_results[i].get("total_ms", -1000) / 1000.0

        # Reference-based metrics (if reference audio available)
        ref_path = ref_wavs.get(wav_path.name)
        if ref_path:
            ref_audio, sr_ref = load_audio(str(ref_path))
            sr_common = max(sr, sr_ref)
            if sr != sr_common:
                gen_audio_r = resample_audio(gen_audio, sr, sr_common)
            else:
                gen_audio_r = gen_audio
            if sr_ref != sr_common:
                ref_audio_r = resample_audio(ref_audio, sr_ref, sr_common)
            else:
                ref_audio_r = ref_audio
            min_len = min(len(gen_audio_r), len(ref_audio_r))

            m.pesq = compute_pesq(ref_audio_r[:min_len], gen_audio_r[:min_len], sr_common)
            m.stoi = compute_stoi(ref_audio_r[:min_len], gen_audio_r[:min_len], sr_common)
            m.mcd_db = compute_mcd(ref_audio_r[:min_len], gen_audio_r[:min_len], sr_common)
            m.spectral_convergence = compute_spectral_convergence(
                ref_audio_r[:min_len], gen_audio_r[:min_len])
            m.f0_rmse, m.f0_corr = compute_f0_metrics(
                ref_audio_r[:min_len], gen_audio_r[:min_len], sr_common)
            m.speaker_sim = compute_speaker_similarity(
                ref_audio_r[:min_len], gen_audio_r[:min_len], sr_common)
            m.mos_proxy = compute_mos_proxy(m.pesq, m.stoi, m.mcd_db, m.f0_corr)

        # WER via Whisper (reference-free, text-based)
        if ref_text:
            hyp = transcribe_whisper(str(wav_path), sr)
            if hyp is not None:
                m.wer_pct = compute_wer_pct(ref_text, hyp)
                print(f"    Whisper WER: {m.wer_pct:.1f}% | \"{hyp}\"")

        # UTMOS (reference-free neural MOS)
        if use_utmos:
            m.utmos = compute_utmos(gen_audio, sr)
            if m.utmos >= 0:
                print(f"    UTMOS: {m.utmos:.2f}")

        results.append(m)

    return aggregate_report(results, "c_pipeline_eval", checkpoint="models/sonata/")


def main():
    ap = argparse.ArgumentParser(description="Evaluate C-pipeline generated TTS audio")
    ap.add_argument("--wav-dir", default="eval/generated", help="Directory with eval_*.wav files")
    ap.add_argument("--c-report", default="eval/reports/c_eval_report.json",
                    help="C pipeline JSON report (for text + timing data)")
    ap.add_argument("--ref-dir", default="", help="Reference audio directory (for paired metrics)")
    ap.add_argument("--utmos", action="store_true", help="Enable UTMOS neural MOS (slow)")
    ap.add_argument("--output", default="eval/reports/eval_report.json", help="Output JSON path")
    args = ap.parse_args()

    print("\n========================================================")
    print("  Sonata TTS — Comprehensive Evaluation (Python)")
    print("========================================================\n")

    report = eval_wavs(args.wav_dir, args.c_report, args.ref_dir, args.utmos)

    # Merge C-native metrics
    report_dict = report.__dict__.copy()
    if args.c_report:
        report_dict = merge_c_report(report_dict, args.c_report)

    # Add SOTA comparison
    report_dict["sota_comparison"] = {
        "rtf": {"sonata": report.mean_rtf, "sota": 0.03, "model": "F5-TTS@7steps"},
        "wer_pct": {"sonata": report.mean_wer_pct, "sota": 0.9, "model": "F5-TTS"},
        "mos": {"sonata": report.mean_mos_proxy, "sota": 4.3, "model": "F5-TTS"},
        "speaker_sim": {"sonata": report.mean_speaker_sim, "sota": 0.687, "model": "MaskGCT"},
        "ttfa_ms": {
            "sonata": report_dict.get("c_pipeline", {}).get("mean_ttfa_ms", -1),
            "sota": 102,
            "model": "VoXtream",
        },
    }

    # Print summary
    print_summary(report)

    # Print SOTA comparison table
    print("  SONATA vs SOTA")
    print("-" * 65)
    print(f"  {'Metric':<25} {'Sonata':>10} {'SOTA':>10} {'Gap':>10} {'Leader'}")
    print("-" * 65)
    for metric, vals in report_dict["sota_comparison"].items():
        s = vals["sonata"]
        t = vals["sota"]
        if s >= 0:
            gap = f"{s/t:.2f}x" if t > 0 else "N/A"
        else:
            gap = "N/A"
        s_str = f"{s:.3f}" if s >= 0 else "N/A"
        print(f"  {metric:<25} {s_str:>10} {t:>10.3f} {gap:>10} {vals['model']}")
    print("-" * 65 + "\n")

    # Save JSON
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report_dict, f, indent=2, default=str)
    print(f"  Report saved: {out_path}\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
```

**Step 2: Make it executable and test import**

Run: `chmod +x eval/run_eval.py && python3 -c "import sys; sys.path.insert(0, 'train/sonata'); from eval_comprehensive import EvalMetrics; print('OK')"`
Expected: "OK"

**Step 3: Commit**

```bash
git add eval/run_eval.py
git commit -m "feat: Python eval wrapper for C-generated WAVs with SOTA comparison"
```

---

### Task 4: Create unified eval script and Makefile target

**Files:**

- Create: `eval/run_baseline.sh`
- Modify: `Makefile` (add `eval` and `eval-full` targets)

**Step 1: Write the orchestration script**

```bash
#!/usr/bin/env bash
# eval/run_baseline.sh -- Run full TTS evaluation pipeline
#
# Usage:
#   ./eval/run_baseline.sh              # C pipeline + Python metrics
#   ./eval/run_baseline.sh --utmos      # Include UTMOS (slow, downloads model)
#   ./eval/run_baseline.sh --ref-dir eval/reference  # With reference audio

set -euo pipefail
cd "$(dirname "$0")/.."

UTMOS_FLAG=""
REF_FLAG=""
for arg in "$@"; do
    case "$arg" in
        --utmos) UTMOS_FLAG="--utmos" ;;
        --ref-dir=*) REF_FLAG="--ref-dir ${arg#*=}" ;;
        --ref-dir) shift; REF_FLAG="--ref-dir $1" ;;
    esac
done

echo ""
echo "========================================================"
echo "  Sonata TTS Evaluation Pipeline"
echo "  $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "========================================================"

# Step 1: Generate WAVs via C pipeline
echo ""
echo "[1/3] Generating audio via C pipeline..."
mkdir -p eval/generated eval/reports

if [ ! -f build/eval-sonata-baseline ]; then
    echo "  Building eval harness..."
    make eval-generate 2>&1 | tail -5
else
    ./build/eval-sonata-baseline
fi

# Step 2: Run Python comprehensive eval
echo ""
echo "[2/3] Running Python comprehensive evaluation..."
python3 eval/run_eval.py \
    --wav-dir eval/generated \
    --c-report eval/reports/c_eval_report.json \
    $REF_FLAG $UTMOS_FLAG \
    --output eval/reports/eval_report.json

# Step 3: Archive report with timestamp
TIMESTAMP=$(date -u '+%Y%m%d_%H%M%S')
cp eval/reports/eval_report.json "eval/reports/eval_${TIMESTAMP}.json"

echo ""
echo "========================================================"
echo "  Evaluation complete!"
echo "  C report:      eval/reports/c_eval_report.json"
echo "  Full report:   eval/reports/eval_report.json"
echo "  Archived:      eval/reports/eval_${TIMESTAMP}.json"
echo "  WAV files:     eval/generated/eval_*.wav"
echo "========================================================"
```

**Step 2: Add Makefile targets**

Add to `Makefile` after the `eval-generate` target:

```makefile
eval-python: eval/run_eval.py
	@mkdir -p eval/reports
	python3 eval/run_eval.py --wav-dir eval/generated \
	  --c-report eval/reports/c_eval_report.json \
	  --output eval/reports/eval_report.json

eval: eval-generate eval-python
	@echo ""
	@echo "=== Full eval complete: eval/reports/eval_report.json ==="

eval-full: eval-generate
	python3 eval/run_eval.py --wav-dir eval/generated \
	  --c-report eval/reports/c_eval_report.json \
	  --utmos --output eval/reports/eval_report.json
```

**Step 3: Make script executable and test**

Run: `chmod +x eval/run_baseline.sh`

**Step 4: Commit**

```bash
git add eval/run_baseline.sh Makefile
git commit -m "feat: unified eval pipeline with make eval target"
```

---

### Task 5: Add reference-based C metrics (MCD, STOI, F0, speaker sim between sentences)

The C harness from Task 2 only computes reference-free metrics (prosody MOS, WER). Since we don't have reference audio yet, add self-consistency metrics: compare each generated WAV against re-synthesis of the same text (deterministic check), and compute inter-sentence speaker consistency (all sentences should sound like the same speaker).

**Files:**

- Modify: `tests/eval_sonata_baseline.c` (add speaker consistency measurement)

**Step 1: Add speaker consistency to the C harness**

After the per-sentence loop, add a block that computes pairwise speaker similarity between all generated WAVs using the C `speaker_similarity()` function. This measures whether the model maintains a consistent voice identity.

Add before the JSON aggregates section:

```c
    /* ---- Speaker consistency: pairwise similarity ---- */
    /* Store audio for consistency check (reuse existing gen loop) */
    /* We re-read the WAVs from disk since the buffer was overwritten */
    float mean_speaker_consistency = -1.0f;
    if (n_generated >= 2) {
        /* Read first WAV as reference */
        char ref_wav[512];
        snprintf(ref_wav, sizeof(ref_wav), "%s/eval_0000.wav", out_dir);
        /* Simple WAV reader: skip 44-byte header, read int16, convert to float */
        FILE *rf = fopen(ref_wav, "rb");
        if (rf) {
            fseek(rf, 0, SEEK_END);
            long fsz = ftell(rf);
            int ref_n = (int)((fsz - 44) / 2);
            fseek(rf, 44, SEEK_SET);
            float *ref_audio = (float *)malloc(ref_n * sizeof(float));
            if (ref_audio) {
                int16_t *tmp = (int16_t *)malloc(ref_n * sizeof(int16_t));
                if (tmp) {
                    fread(tmp, 2, ref_n, rf);
                    for (int j = 0; j < ref_n; j++) ref_audio[j] = tmp[j] / 32768.0f;
                    free(tmp);

                    double sim_sum = 0;
                    int sim_count = 0;
                    for (int k = 1; k < n_generated && k < N_SENTENCES; k++) {
                        char cmp_wav[512];
                        snprintf(cmp_wav, sizeof(cmp_wav), "%s/eval_%04d.wav", out_dir, k);
                        FILE *cf = fopen(cmp_wav, "rb");
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
                        fprintf(stderr, "  Speaker consistency: %.3f (across %d pairs)\n",
                                mean_speaker_consistency, sim_count);
                    }
                }
                free(ref_audio);
            }
            fclose(rf);
        }
    }
```

Then add to the JSON aggregates:

```c
    fprintf(jf, "    \"mean_speaker_consistency\": %.4f,\n", mean_speaker_consistency);
```

**Step 2: Build and test**

Run: `make eval-generate 2>&1 | tail -20`
Expected: Speaker consistency metric printed

**Step 3: Commit**

```bash
git add tests/eval_sonata_baseline.c
git commit -m "feat: add speaker consistency metric to C eval harness"
```

---

### Task 6: Add eval .gitignore and documentation

**Files:**

- Create: `eval/.gitignore`
- Create: `eval/README.md`

**Step 1: Create .gitignore for eval artifacts**

```
# Generated audio (large, reproducible)
generated/
reference/

# Reports (except templates)
reports/*.json

# Python cache
__pycache__/
*.pyc

# Model downloads
pretrained_models/
```

**Step 2: Create README**

````markdown
# Sonata TTS Evaluation Pipeline

## Quick Start

```bash
# Full evaluation (C pipeline + Python metrics)
make eval

# With UTMOS neural MOS (slower, downloads model)
make eval-full

# Just generate WAVs via C pipeline
make eval-generate

# Just run Python metrics on existing WAVs
make eval-python
```
````

## Reports

| File                         | Contents                                                    |
| ---------------------------- | ----------------------------------------------------------- |
| `reports/c_eval_report.json` | C pipeline: RTF, TTFA, WER (Conformer), prosody MOS         |
| `reports/eval_report.json`   | Full: above + PESQ, STOI, MCD, F0, speaker sim, Whisper WER |

## Metrics

| Metric              | Source | Type                  | Good   | SOTA  |
| ------------------- | ------ | --------------------- | ------ | ----- |
| RTF                 | C      | Performance           | <0.2   | 0.03  |
| TTFA                | C      | Latency               | <300ms | 102ms |
| LM tok/s            | C      | Performance           | >40    | -     |
| Prosody MOS         | C      | Quality (ref-free)    | >3.5   | >4.0  |
| WER (Conformer)     | C      | Intelligibility       | <10%   | <3%   |
| WER (Whisper)       | Python | Intelligibility       | <5%    | <0.9% |
| PESQ                | Python | Quality (ref)         | >3.0   | >3.5  |
| STOI                | Python | Intelligibility (ref) | >0.75  | >0.90 |
| MCD                 | Python | Spectral (ref)        | <6dB   | <4dB  |
| F0 corr             | Python | Prosody (ref)         | >0.70  | >0.85 |
| Speaker sim         | Python | Voice (ref)           | >0.75  | >0.90 |
| UTMOS               | Python | Quality (ref-free)    | >3.5   | >4.0  |
| Speaker consistency | C      | Voice identity        | >0.85  | -     |

## Adding Reference Audio

Place reference WAVs in `eval/reference/` matching the naming `eval_NNNN.wav`.
This enables paired metrics (PESQ, STOI, MCD, F0, speaker sim).

````

**Step 3: Commit**

```bash
git add eval/.gitignore eval/README.md
git commit -m "docs: eval pipeline README and gitignore"
````

---

### Task 7: Run the full eval pipeline and capture baselines

This is the validation task — actually run everything and record the numbers.

**Step 1: Install Python dependencies**

Run: `pip install soundfile numpy torch librosa pesq pystoi jiwer resampy openai-whisper`

**Step 2: Build and run C eval**

Run: `make eval-generate 2>&1`
Expected: 10 sentences generated, JSON report at `eval/reports/c_eval_report.json`

**Step 3: Run Python eval**

Run: `python3 eval/run_eval.py --wav-dir eval/generated --c-report eval/reports/c_eval_report.json --output eval/reports/eval_report.json 2>&1`
Expected: Per-sentence Whisper WER, aggregate report with SOTA comparison table

**Step 4: Review the report**

Run: `python3 -c "import json; d=json.load(open('eval/reports/eval_report.json')); print(json.dumps(d.get('sota_comparison', {}), indent=2))"`
Expected: SOTA comparison dict with Sonata values filled in

**Step 5: Commit baseline report**

```bash
# Keep one baseline report in git for tracking
cp eval/reports/eval_report.json eval/reports/baseline_2026-03-07.json
git add eval/reports/baseline_2026-03-07.json
git commit -m "data: initial eval baseline measurements"
```

---

## Audit Phase

After all 7 tasks are complete, run a **5-agent audit team** (4-10 files modified, multiple concerns):

| Agent | Role                   | Focus                                                                                                                              |
| ----- | ---------------------- | ---------------------------------------------------------------------------------------------------------------------------------- |
| 1     | **correctness-prover** | Verify JSON output is valid, metrics computed correctly, no off-by-one in audio indexing                                           |
| 2     | **e2e-tracer**         | Trace data flow: C harness -> WAV files -> Python reader -> metrics -> JSON. Verify file paths, sample rates, format consistency   |
| 3     | **gap-hunter**         | Find untested paths: What if no models present? What if WAVs are empty? What if Python deps missing? Edge cases in the merge logic |
| 4     | **perf-validator**     | Measure actual eval pipeline runtime. Is Whisper WER the bottleneck? Can we parallelize? Memory usage                              |
| 5     | **synthesis**          | Aggregate all findings into P0/P1/P2/P3 prioritized list                                                                           |

Each auditor examines different files:

- Agent 1: `tests/eval_sonata_baseline.c` (JSON output, metric math)
- Agent 2: `eval/run_eval.py` + `eval/run_baseline.sh` (data flow, paths)
- Agent 3: `eval/run_eval.py` + `train/sonata/eval_comprehensive.py` (error handling, edge cases)
- Agent 4: Full pipeline timing (build + generate + eval wall clock)

Fix all P0 findings, then re-audit with 2 focused agents.
