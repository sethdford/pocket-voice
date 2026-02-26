/*
 * backchannel.h — Active listening backchannel generation.
 *
 * Detects natural backchannel timing from user speech features (pitch fall,
 * pause duration, phrase boundary) and mixes pre-synthesized audio snippets
 * ("mhm", "yeah", "right") into the speaker output at ~50ms latency.
 *
 * No LLM round-trip required — runs at audio level on mel features.
 */

#ifndef BACKCHANNEL_H
#define BACKCHANNEL_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct BackchannelGen BackchannelGen;

typedef enum {
    BC_NONE = 0,
    BC_MHM,
    BC_YEAH,
    BC_RIGHT,
    BC_OKAY,
    BC_UH_HUH,
    BC_COUNT
} BackchannelType;

typedef struct {
    BackchannelType type;
    float           confidence;
    int             ready;     /* 1 = should emit now */
} BackchannelEvent;

/* Create generator. sample_rate = capture audio rate (typically 24000). */
BackchannelGen *backchannel_create(int sample_rate);
void backchannel_destroy(BackchannelGen *bc);

/* Feed capture audio for timing analysis. Returns event if backchannel
 * should fire (event.ready == 1). Call at ~80ms intervals (frame-aligned). */
BackchannelEvent backchannel_feed(BackchannelGen *bc, const float *audio,
                                  int n_samples, float stt_eou_prob);

/* Get pre-synthesized audio for a backchannel type.
 * Returns pointer to internal buffer and sets *out_len.
 * Audio is at the specified sample_rate from create(). */
const float *backchannel_get_audio(BackchannelGen *bc, BackchannelType type,
                                   int *out_len);

/* Load custom backchannel audio from WAV file (overrides built-in). */
int backchannel_load_wav(BackchannelGen *bc, BackchannelType type,
                         const char *wav_path);

/* Reset state between turns. */
void backchannel_reset(BackchannelGen *bc);

/* Enable/disable. Disabled by default until explicitly enabled. */
void backchannel_set_enabled(BackchannelGen *bc, int enabled);
int  backchannel_is_enabled(BackchannelGen *bc);

#ifdef __cplusplus
}
#endif

#endif /* BACKCHANNEL_H */
