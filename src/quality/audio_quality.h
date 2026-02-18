/**
 * audio_quality.h — Objective audio quality metrics for TTS evaluation.
 *
 * Implements the industry-standard metrics used to benchmark TTS systems:
 *
 *   MCD  — Mel-Cepstral Distortion (dB): spectral distance between
 *           synthesized and reference speech in mel-cepstral domain.
 *           Lower is better. <6 dB = high quality. <4 dB = near-human.
 *
 *   STOI — Short-Time Objective Intelligibility (0-1): predicts how
 *           intelligible the speech is to human listeners.
 *           Higher is better. >0.9 = excellent. >0.75 = good.
 *
 *   SNR  — Segmental Signal-to-Noise Ratio (dB): per-frame SNR averaged
 *           over active speech segments. Higher is better.
 *
 *   F0   — Fundamental frequency (pitch) extraction and comparison.
 *           F0 RMSE < 20 Hz = good prosody. F0 correlation > 0.8 = natural.
 *
 *   Speaker Similarity — MFCC-based cosine distance between source and
 *           synthesized voice. >0.85 = excellent cloning quality.
 *
 * All metrics use Apple vDSP/Accelerate for vectorized computation.
 *
 * Golden Signals (what "best-in-class" looks like):
 * ┌─────────────────────┬──────────────┬────────────────┐
 * │ Metric              │ Good         │ Best-in-class  │
 * ├─────────────────────┼──────────────┼────────────────┤
 * │ MCD                 │ < 6.0 dB     │ < 4.0 dB       │
 * │ STOI                │ > 0.75       │ > 0.90         │
 * │ Seg-SNR             │ > 15 dB      │ > 25 dB        │
 * │ F0 RMSE             │ < 30 Hz      │ < 15 Hz        │
 * │ F0 Correlation      │ > 0.70       │ > 0.85         │
 * │ Speaker Similarity  │ > 0.75       │ > 0.90         │
 * │ WER (round-trip)    │ < 10%        │ < 3%           │
 * │ RTF                 │ < 1.0        │ < 0.3          │
 * │ First-chunk (ms)    │ < 500        │ < 150          │
 * └─────────────────────┴──────────────┴────────────────┘
 */

#ifndef AUDIO_QUALITY_H
#define AUDIO_QUALITY_H

#ifdef __cplusplus
extern "C" {
#endif

/* ── Mel-Cepstral Distortion ──────────────────────────── */

typedef struct {
    float mcd_db;          /* Average MCD in dB (lower = better) */
    float mcd_std;         /* Standard deviation across frames */
    int   n_frames;        /* Number of frames compared */
} MCDResult;

/**
 * Compute Mel-Cepstral Distortion between reference and synthesized audio.
 * Uses 13 MFCCs (excluding C0) with 25ms frames, 10ms hop.
 *
 * @param ref        Reference audio (float32, mono)
 * @param ref_len    Number of samples
 * @param synth      Synthesized audio (float32, mono)
 * @param synth_len  Number of samples
 * @param sr         Sample rate
 * @return MCDResult
 */
MCDResult mcd_compute(const float *ref, int ref_len,
                      const float *synth, int synth_len, int sr);

/* ── Short-Time Objective Intelligibility ─────────────── */

typedef struct {
    float stoi;            /* STOI score (0-1, higher = better) */
    float estoi;           /* Extended STOI (better for noisy conditions) */
    int   n_frames;
} STOIResult;

/**
 * Compute STOI between clean reference and processed/degraded signal.
 * Reference and test must be time-aligned and same length.
 *
 * @param ref     Clean reference signal
 * @param test    Processed/degraded signal
 * @param n       Number of samples (must be same for both)
 * @param sr      Sample rate (must be 10kHz for standard STOI; will resample)
 * @return STOIResult
 */
STOIResult stoi_compute(const float *ref, const float *test, int n, int sr);

/* ── Segmental SNR ────────────────────────────────────── */

typedef struct {
    float seg_snr_db;      /* Average segmental SNR in dB */
    float overall_snr_db;  /* Global SNR in dB */
    int   n_active_frames; /* Frames above speech threshold */
} SNRResult;

/**
 * Compute segmental SNR between reference and synthesized audio.
 * Only active speech frames (above -40 dB) contribute.
 *
 * @param ref      Reference signal
 * @param test     Test signal
 * @param n        Number of samples
 * @param sr       Sample rate
 * @return SNRResult
 */
SNRResult snr_compute(const float *ref, const float *test, int n, int sr);

/* ── F0 (Pitch) Analysis ──────────────────────────────── */

typedef struct {
    float f0_rmse_hz;      /* RMS error of F0 contour in Hz */
    float f0_corr;         /* Pearson correlation of F0 contours (0-1) */
    float voicing_accuracy; /* % of frames with correct voiced/unvoiced decision */
    float ref_mean_f0;     /* Mean F0 of reference */
    float synth_mean_f0;   /* Mean F0 of synthesized */
    int   n_voiced_frames;
} F0Result;

/**
 * Extract F0 contours from both signals and compare.
 * Uses autocorrelation-based pitch detection (50-500 Hz range).
 *
 * @param ref       Reference audio
 * @param ref_len   Number of samples
 * @param synth     Synthesized audio
 * @param synth_len Number of samples
 * @param sr        Sample rate
 * @return F0Result
 */
F0Result f0_compare(const float *ref, int ref_len,
                    const float *synth, int synth_len, int sr);

/* ── Speaker Similarity ───────────────────────────────── */

typedef struct {
    float cosine_sim;      /* Cosine similarity of MFCC vectors (0-1) */
    float euclidean_dist;  /* Euclidean distance in MFCC space */
    int   n_coeffs;        /* Number of MFCC coefficients used */
} SpeakerSimResult;

/**
 * Compute speaker similarity between two audio signals using MFCC-based
 * voice characterization. Higher cosine similarity = more similar voices.
 *
 * @param audio_a   First audio signal
 * @param len_a     Number of samples
 * @param audio_b   Second audio signal
 * @param len_b     Number of samples
 * @param sr        Sample rate
 * @return SpeakerSimResult
 */
SpeakerSimResult speaker_similarity(const float *audio_a, int len_a,
                                     const float *audio_b, int len_b, int sr);

/* ── Latency Metrics ──────────────────────────────────── */

typedef struct {
    float rtf;             /* Real-time factor (audio_duration / gen_time) */
    float first_chunk_ms;  /* Time to first audio chunk */
    float ttft_ms;         /* Time to first Claude token */
    float e2e_ms;          /* End-to-end: speech_end → first_audio */
    float total_ms;        /* Total turn duration */
} LatencyMetrics;

/* ── Combined Scorecard ───────────────────────────────── */

typedef struct {
    MCDResult mcd;
    STOIResult stoi;
    SNRResult snr;
    F0Result f0;
    SpeakerSimResult speaker;
    LatencyMetrics latency;
    float wer;
    float cer;
    float overall_score;   /* Weighted composite (0-100) */
    char grade;            /* A/B/C/D/F */
} QualityScorecard;

/**
 * Compute an overall quality grade from individual metrics.
 * Weights: STOI 25%, MCD 20%, WER 20%, F0 15%, Speaker 10%, Latency 10%
 *
 * @param sc  Scorecard with individual metrics filled in
 * @return Updated scorecard with overall_score and grade
 */
QualityScorecard quality_grade(QualityScorecard sc);

/**
 * Print a formatted quality report to stderr.
 */
void quality_print_report(const QualityScorecard *sc, const char *test_name);

#ifdef __cplusplus
}
#endif

#endif /* AUDIO_QUALITY_H */
