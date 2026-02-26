/**
 * prosody_log.h — Real-time prosody logging for visualization/debugging.
 *
 * Logs per-segment prosody parameters (pitch, rate, energy, emotion, contour)
 * and per-turn timing to a JSONL file. Can be visualized with the companion
 * web dashboard or any JSONL-compatible tool.
 *
 * Usage:
 *   ProsodyLog *log = prosody_log_open("prosody.jsonl");
 *   prosody_log_segment(log, ...);
 *   prosody_log_turn(log, ...);
 *   prosody_log_close(log);
 */

#ifndef PROSODY_LOG_H
#define PROSODY_LOG_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ProsodyLog ProsodyLog;

/** Open a JSONL log file. Returns NULL on failure. */
ProsodyLog *prosody_log_open(const char *path);

/** Close and flush the log. NULL-safe. */
void prosody_log_close(ProsodyLog *log);

/**
 * Log a TTS segment with prosody parameters.
 */
void prosody_log_segment(ProsodyLog *log,
                         const char *text,
                         float pitch,
                         float rate,
                         float volume_db,
                         const char *emotion,
                         const char *contour,
                         int duration_ms);

/**
 * Log a complete turn with aggregate prosody stats.
 */
void prosody_log_turn(ProsodyLog *log,
                      int turn_id,
                      const char *user_text,
                      const char *response_text,
                      float user_rate_wps,
                      float response_pitch,
                      float response_rate,
                      float response_energy,
                      float vrl_ms,
                      float tts_rtf);

/**
 * Log raw F0/energy curves for detailed prosody analysis.
 * f0: float[n_frames] pitch in Hz, energy: float[n_frames] in dB.
 */
void prosody_log_contour(ProsodyLog *log,
                         const char *segment_id,
                         const float *f0, const float *energy,
                         int n_frames, int sr);

#ifdef __cplusplus
}
#endif

#endif /* PROSODY_LOG_H */
