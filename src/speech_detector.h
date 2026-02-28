/**
 * speech_detector.h — Unified speech detection and end-of-utterance module.
 *
 * Consolidates native_vad, mimi_endpointer, and fused_eou into a single
 * interface. Encapsulates all buffer management, resampling, and signal
 * fusion that was previously scattered across pocket_voice_pipeline.c.
 *
 * Pipeline usage:
 *   sd = speech_detector_create(cfg);
 *   // In audio callback:
 *   speech_detector_feed(sd, pcm24, n_samples);
 *   // In LISTENING state:
 *   if (speech_detector_speech_active(sd, energy_vad)) → transition
 *   // In RECORDING state:
 *   EOUResult r = speech_detector_eou(sd, energy_vad, stt_eou_prob);
 *   if (r.triggered) → transition
 *
 * Internally manages:
 *   - 24kHz→16kHz resampling for native VAD
 *   - 16kHz 512-sample chunking for VAD
 *   - 24kHz 1920-sample framing for endpointer mel features
 *   - 3-signal fusion (energy + mimi + STT)
 */

#ifndef SPEECH_DETECTOR_H
#define SPEECH_DETECTOR_H

#include "fused_eou.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct SpeechDetector SpeechDetector;

typedef struct {
    const char *native_vad_path;  /* .nvad weights (NULL to skip) */
    int   mimi_latent_dim;        /* Mimi EP input dim (default: 80) */
    int   mimi_hidden_dim;        /* Mimi EP hidden dim (default: 64) */
    float eot_threshold;          /* Fused EOU threshold (default: 0.6) */
    int   eot_consec_frames;      /* Consecutive frames required (default: 2) */
} SpeechDetectorConfig;

/**
 * Create a speech detector with default configuration.
 * Returns NULL on allocation failure.
 */
SpeechDetector *speech_detector_create(const SpeechDetectorConfig *cfg);

/**
 * Destroy and free all resources.
 */
void speech_detector_destroy(SpeechDetector *sd);

/**
 * Reset all state for a new utterance. Clears LSTM states,
 * buffers, and fused EOU counters.
 */
void speech_detector_reset(SpeechDetector *sd);

/**
 * Feed 24kHz capture audio. Internally handles:
 *   - 24→16kHz resampling + 512-sample chunking for VAD
 *   - 80ms framing + mel-energy extraction for endpointer
 */
void speech_detector_feed(SpeechDetector *sd,
                          const float *pcm24, int n_samples);

/**
 * Feed pre-resampled 16kHz audio directly (skip internal resampling).
 * Use when the pipeline already has 16kHz audio available.
 */
void speech_detector_feed_16k(SpeechDetector *sd,
                              const float *pcm16, int n_samples);

/**
 * Get the latest neural speech probability [0,1].
 * Returns -1.0 if no VAD engine is loaded or no data fed yet.
 */
float speech_detector_speech_prob(const SpeechDetector *sd);

/**
 * Check if speech is active (for LISTENING→RECORDING transition).
 *
 * @param energy_vad  Energy VAD state from voice_engine_get_vad_state()
 * @return 1 if speech detected, 0 if silence
 */
int speech_detector_speech_active(const SpeechDetector *sd, int energy_vad);

/**
 * Get the latest Mimi endpointer end-of-turn probability [0,1].
 * Returns 0 if endpointer not loaded.
 */
float speech_detector_eot_prob(const SpeechDetector *sd);

/**
 * Run fused 3-signal EOU detection (for RECORDING state).
 *
 * @param energy_vad    Energy VAD state from voice_engine_get_vad_state()
 * @param stt_eou_prob  ASR-inline EOU probability (0 if unavailable)
 * @return Fusion result with triggered flag
 */
EOUResult speech_detector_eou(SpeechDetector *sd,
                              int energy_vad, float stt_eou_prob);

/**
 * Check if any VAD engine is loaded (native or ONNX).
 */
int speech_detector_has_vad(const SpeechDetector *sd);

/**
 * Check if endpointer is loaded.
 */
int speech_detector_has_endpointer(const SpeechDetector *sd);

/**
 * Feed semantic completion probability from SemanticEOU into the fused EOU.
 * Call whenever new STT transcript text has been processed.
 */
void speech_detector_feed_semantic(SpeechDetector *sd, float prob);

/**
 * Set the semantic signal weight in fused EOU (default 0, recommended 0.15).
 */
void speech_detector_set_semantic_weight(SpeechDetector *sd, float w);

#ifdef __cplusplus
}
#endif

#endif /* SPEECH_DETECTOR_H */
