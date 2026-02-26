/**
 * voice_onboard.h — Real-time voice onboarding for prosody transfer.
 *
 * Captures a short speech sample from the mic, extracts:
 *   1. Speaker embedding (via ONNX speaker encoder)
 *   2. Prosody profile (F0, rate, energy baseline)
 *
 * The extracted embedding can be fed to sonata_flow_set_speaker_embedding()
 * for zero-shot voice cloning, and the prosody profile can adjust TTS parameters
 * to match the user's speaking style.
 */

#ifndef VOICE_ONBOARD_H
#define VOICE_ONBOARD_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct VoiceOnboardSession VoiceOnboardSession;

typedef struct {
    float f0_mean;          /* Mean F0 in Hz */
    float f0_range;         /* F0 range (max - min) in Hz */
    float energy_mean_db;   /* Mean energy in dB */
    float speaking_rate;    /* Words per second estimate */
    int   sample_rate;      /* Audio sample rate */
    float duration_sec;     /* Total captured duration */
} VoiceProsodyProfile;

typedef struct {
    float *embedding;       /* Speaker embedding vector (caller must free) */
    int    embedding_dim;   /* Dimensionality of embedding */
    VoiceProsodyProfile prosody;
    int    success;         /* 1 = valid, 0 = not enough speech captured */
} VoiceOnboardResult;

/**
 * Create an onboarding session.
 * @param speaker_encoder_path  Path to ONNX speaker encoder model (NULL = skip embedding)
 * @param capture_duration_sec  How long to capture (default: 5.0)
 * @param sample_rate           Audio sample rate (typically 16000)
 */
VoiceOnboardSession *voice_onboard_create(
    const char *speaker_encoder_path,
    float capture_duration_sec,
    int sample_rate);

/** Feed audio frames (16kHz float32). Returns 1 when capture is complete. */
int voice_onboard_feed(VoiceOnboardSession *session,
                       const float *pcm, int n_samples);

/** Get progress as fraction [0.0, 1.0]. */
float voice_onboard_progress(const VoiceOnboardSession *session);

/**
 * Finalize: extract speaker embedding + prosody profile.
 * The result's embedding pointer must be freed by the caller.
 */
VoiceOnboardResult voice_onboard_finalize(VoiceOnboardSession *session);

/** Destroy the session. Does NOT free the result's embedding. */
void voice_onboard_destroy(VoiceOnboardSession *session);

/** Estimate F0 from a PCM buffer using autocorrelation. */
float voice_onboard_estimate_f0(const float *pcm, int n_samples, int sample_rate);

#ifdef __cplusplus
}
#endif

#endif /* VOICE_ONBOARD_H */
