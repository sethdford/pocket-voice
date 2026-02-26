/*
 * audio_emotion.h — Real-time emotion detection from audio features.
 *
 * Lightweight classifier on mel-energy features that detects:
 *   - Valence (positive/negative)
 *   - Arousal (calm/excited)
 *   - Speaking rate relative to baseline
 *   - Pitch range and contour
 *
 * Designed to complement text-based emotion detection with acoustic cues
 * that text alone cannot capture (sarcasm, frustration, hesitancy).
 *
 * Runs on AMX via vDSP — no GPU required.
 */

#ifndef AUDIO_EMOTION_H
#define AUDIO_EMOTION_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct AudioEmotionDetector AudioEmotionDetector;

typedef enum {
    AUDIO_EMO_NEUTRAL = 0,
    AUDIO_EMO_HAPPY,
    AUDIO_EMO_SAD,
    AUDIO_EMO_ANGRY,
    AUDIO_EMO_FEARFUL,
    AUDIO_EMO_SURPRISED,
    AUDIO_EMO_CALM,
    AUDIO_EMO_EXCITED,
    AUDIO_EMO_FRUSTRATED,
    AUDIO_EMO_HESITANT,
    AUDIO_EMO_COUNT
} AudioEmotion;

typedef struct {
    AudioEmotion primary;
    float        confidence;

    float        valence;       /* -1.0 (negative) to +1.0 (positive) */
    float        arousal;       /* 0.0 (calm) to 1.0 (excited) */

    float        pitch_mean;    /* Hz, running mean */
    float        pitch_range;   /* Hz, max - min over window */
    float        energy_mean;   /* dB RMS */
    float        speaking_rate; /* Relative to baseline (1.0 = normal) */
    float        jitter;        /* Pitch instability (higher = more emotional) */
    float        spectral_tilt; /* dB/octave (more negative = breathy/tired) */
} AudioEmotionResult;

/* Create detector. sample_rate = 24000 typically. */
AudioEmotionDetector *audio_emotion_create(int sample_rate);
void audio_emotion_destroy(AudioEmotionDetector *det);

/* Feed audio frame (80ms = 1920 samples at 24kHz).
 * Accumulates features and updates running emotion estimate. */
void audio_emotion_feed(AudioEmotionDetector *det, const float *audio, int n_samples);

/* Get current emotion estimate. Valid after >= 3 frames (~240ms). */
AudioEmotionResult audio_emotion_get(const AudioEmotionDetector *det);

/* Reset state (call between turns). */
void audio_emotion_reset(AudioEmotionDetector *det);

/* Format emotion as string for LLM system prompt injection.
 * Writes to buf, returns chars written. */
int audio_emotion_describe(const AudioEmotionResult *result, char *buf, int buf_size);

#ifdef __cplusplus
}
#endif

#endif /* AUDIO_EMOTION_H */
