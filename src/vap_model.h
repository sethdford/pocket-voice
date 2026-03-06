/*
 * vap_model.h — Voice Activity Projection (VAP) model for turn-taking prediction.
 *
 * SOTA neural turn-taking predictor that replaces rule-based EOU detection
 * and backchannel timing with a single learned model.
 *
 * Architecture: [user_mel, system_mel] → Linear → transformer encoder (causal)
 *              → 4 sigmoid heads (user_speaking, system_turn, backchannel, eou)
 *
 * All matrix ops via cblas (AMX-accelerated). Runs at 50Hz (20ms per frame).
 */

#ifndef VAP_MODEL_H
#define VAP_MODEL_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct VAPModel VAPModel;

typedef struct {
    float p_user_speaking;      /* P(user continues speaking in next 0.5s) */
    float p_system_turn;       /* P(system should take turn in next 0.5s) */
    float p_backchannel;       /* P(backchannel appropriate in next 0.5s) */
    float p_eou;               /* P(end of user utterance in next 0.5s) */
} VAPPrediction;

/* Create VAP model from binary weights file (.vap format).
 * d_model, n_layers, n_heads loaded from file header.
 * Returns NULL on failure. */
VAPModel *vap_create(const char *weights_path);

/* Create with explicit config (for testing without weights).
 * Weights are zero-initialized; outputs are bias-dominated but valid. */
VAPModel *vap_create_config(int d_model, int n_layers, int n_heads, int ff_dim);

void vap_destroy(VAPModel *vap);

/* Feed one frame of features (20ms at 50Hz).
 * user_mel: [80] mel features from capture audio
 * system_mel: [80] mel features from TTS playback audio (NULL if silent)
 * Returns prediction for this timestep. */
VAPPrediction vap_feed(VAPModel *vap, const float *user_mel, const float *system_mel);

/* Reset state between conversations. Clears transformer KV cache. */
void vap_reset(VAPModel *vap);

/* Get smoothed predictions (EMA over recent frames). */
VAPPrediction vap_get_smoothed(const VAPModel *vap);

/* Set EMA smoothing factor (0.0 = no smoothing, 0.9 = heavy). Default 0.3. */
void vap_set_smoothing(VAPModel *vap, float alpha);

#ifdef __cplusplus
}
#endif

#endif /* VAP_MODEL_H */
