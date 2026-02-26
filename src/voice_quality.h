/**
 * voice_quality.h — Objective voice quality metrics for TTS evaluation.
 *
 * Provides PESQ-lite, STOI-lite, Log-Spectral Distance, and MOS prediction.
 * All metrics are lightweight approximations suitable for automated testing
 * and A/B comparisons, not ITU-T certification.
 */

#ifndef VOICE_QUALITY_H
#define VOICE_QUALITY_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    float pesq;     /* PESQ-lite score [1.0 - 4.5] (perceptual quality) */
    float stoi;     /* STOI-lite [0.0 - 1.0] (intelligibility) */
    float lsd_db;   /* Log-Spectral Distance in dB (lower = better) */
    float mos;      /* Predicted MOS [1.0 - 5.0] */
    int   valid;    /* 1 if metrics were computed successfully */
} VoiceQualityReport;

/**
 * Compute PESQ-lite: Bark-domain frequency-weighted perceptual quality.
 * Higher = better, range [1.0, 4.5].
 */
float vq_pesq_lite(const float *ref, const float *deg,
                    int n_samples, int sample_rate);

/**
 * Compute STOI-lite: Short-Time Objective Intelligibility.
 * Higher = better, range [0.0, 1.0].
 */
float vq_stoi_lite(const float *ref, const float *deg,
                    int n_samples, int sample_rate);

/**
 * Compute Log-Spectral Distance in dB.
 * Lower = better. Typically < 3 dB for good quality.
 */
float vq_log_spectral_distance(const float *ref, const float *deg,
                                int n_samples, int sample_rate);

/**
 * Predict MOS from objective metrics.
 * Weighted combination of PESQ, STOI, and LSD.
 */
float vq_predict_mos(float pesq_score, float stoi_score, float lsd_db);

/**
 * Run all quality metrics in one call.
 * Returns a VoiceQualityReport with all fields populated.
 */
VoiceQualityReport vq_evaluate(const float *ref, const float *deg,
                                int n_samples, int sample_rate);

#ifdef __cplusplus
}
#endif

#endif /* VOICE_QUALITY_H */
