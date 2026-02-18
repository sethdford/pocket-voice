/**
 * mimi_endpointer.c — LSTM-based endpointer on Mimi codec features.
 *
 * Single-layer LSTM with 4-class output, optimized for Apple AMX via
 * cblas_sgemv for all matrix-vector products. The entire forward pass
 * is ~50 FLOPs/param, taking < 0.1ms per frame on M1+.
 *
 * LSTM equations (standard):
 *   [i, f, g, o] = sigmoid/tanh( W_i @ x + W_h @ h_prev + b )
 *   c = f * c_prev + i * g
 *   h = o * tanh(c)
 *
 * Where W_i is [4H, D], W_h is [4H, H], b is [4H].
 * Gates: i=input, f=forget, g=cell_gate, o=output.
 */

#include "mimi_endpointer.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>

#ifdef __APPLE__
#ifndef ACCELERATE_NEW_LAPACK
#define ACCELERATE_NEW_LAPACK
#endif
#include <Accelerate/Accelerate.h>
#endif

#define EP_N_CLASSES 4

/* ── LSTM State ───────────────────────────────────────── */

struct MimiEndpointer {
    int latent_dim;      /* D: input dimension */
    int hidden_dim;      /* H: LSTM hidden dimension */
    float eot_threshold;
    int consec_required;

    /* Pre-LayerNorm */
    float *norm_w;       /* [D] */
    float *norm_b;       /* [D] */

    /* LSTM weights */
    float *Wi;           /* [4H, D] — input-to-hidden */
    float *Wh;           /* [4H, H] — hidden-to-hidden */
    float *bias;         /* [4H] */

    /* Output projection */
    float *out_w;        /* [N_CLASSES, H] */
    float *out_b;        /* [N_CLASSES] */

    /* LSTM hidden state */
    float *h;            /* [H] — hidden state */
    float *c;            /* [H] — cell state */

    /* Working memory */
    float *gates;        /* [4H] */
    float *normed;       /* [D] */

    /* Tracking */
    EndpointClass prev_class;
    float smoothed_eot;  /* EMA of P(eot) */
    int consec_eot;
    int triggered;
    int speech_active;   /* Was speech detected before EOT? */
    int frames_since_speech_end; /* Frames since last speech → EOT transition */
    int total_frames;
};

/* ── Utility ──────────────────────────────────────────── */

static float sigmoid(float x) {
    if (x > 10.0f) return 1.0f;
    if (x < -10.0f) return 0.0f;
    return 1.0f / (1.0f + expf(-x));
}

static void layer_norm_1d(float *out, const float *in, const float *w,
                           const float *b, int D) {
    float mean = 0.0f, var = 0.0f;
#ifdef __APPLE__
    vDSP_meanv(in, 1, &mean, D);
#else
    for (int i = 0; i < D; i++) mean += in[i];
    mean /= (float)D;
#endif
    for (int i = 0; i < D; i++) {
        float d = in[i] - mean;
        var += d * d;
    }
    var /= (float)D;
    float inv_std = 1.0f / sqrtf(var + 1e-5f);
    for (int i = 0; i < D; i++)
        out[i] = (in[i] - mean) * inv_std * w[i] + b[i];
}

/* ── Public API ───────────────────────────────────────── */

MimiEndpointer *mimi_ep_create(int latent_dim, int hidden_dim,
                                 float eot_threshold, int consec_frames) {
    MimiEndpointer *ep = (MimiEndpointer *)calloc(1, sizeof(MimiEndpointer));
    if (!ep) return NULL;

    ep->latent_dim = latent_dim;
    ep->hidden_dim = hidden_dim;
    ep->eot_threshold = eot_threshold;
    ep->consec_required = consec_frames;

    int D = latent_dim, H = hidden_dim;

    ep->norm_w = (float *)calloc(D, sizeof(float));
    ep->norm_b = (float *)calloc(D, sizeof(float));
    ep->Wi     = (float *)calloc((size_t)4 * H * D, sizeof(float));
    ep->Wh     = (float *)calloc((size_t)4 * H * H, sizeof(float));
    ep->bias   = (float *)calloc((size_t)4 * H, sizeof(float));
    ep->out_w  = (float *)calloc((size_t)EP_N_CLASSES * H, sizeof(float));
    ep->out_b  = (float *)calloc(EP_N_CLASSES, sizeof(float));
    ep->h      = (float *)calloc(H, sizeof(float));
    ep->c      = (float *)calloc(H, sizeof(float));
    ep->gates  = (float *)calloc((size_t)4 * H, sizeof(float));
    ep->normed = (float *)calloc(D, sizeof(float));

    /* Initialize LayerNorm to identity */
    for (int i = 0; i < D; i++) ep->norm_w[i] = 1.0f;

    ep->prev_class = EP_SILENCE;
    ep->smoothed_eot = 0.0f;
    ep->frames_since_speech_end = -1;

    return ep;
}

int mimi_ep_load_weights(MimiEndpointer *ep, const char *path) {
    if (!ep || !path) return -1;
    FILE *f = fopen(path, "rb");
    if (!f) return -1;

    int D = ep->latent_dim, H = ep->hidden_dim;
    size_t total = 0;

    total += fread(ep->norm_w, sizeof(float), D, f);
    total += fread(ep->norm_b, sizeof(float), D, f);
    total += fread(ep->Wi, sizeof(float), (size_t)4 * H * D, f);
    total += fread(ep->Wh, sizeof(float), (size_t)4 * H * H, f);
    total += fread(ep->bias, sizeof(float), (size_t)4 * H, f);
    total += fread(ep->out_w, sizeof(float), (size_t)EP_N_CLASSES * H, f);
    total += fread(ep->out_b, sizeof(float), EP_N_CLASSES, f);

    fclose(f);

    size_t expected = (size_t)(D + D + 4*H*D + 4*H*H + 4*H + EP_N_CLASSES*H + EP_N_CLASSES);
    if (total != expected) {
        fprintf(stderr, "[mimi_ep] Weight size mismatch: got %zu, expected %zu\n",
                total, expected);
        return -1;
    }

    fprintf(stderr, "[mimi_ep] Loaded weights: D=%d H=%d (%.1fK params)\n",
            D, H, (float)expected / 1000.0f);
    return 0;
}

void mimi_ep_init_random(MimiEndpointer *ep, uint32_t seed) {
    if (!ep) return;
    int D = ep->latent_dim, H = ep->hidden_dim;

    /* Xavier initialization */
    float scale_i = sqrtf(2.0f / (float)(D + H));
    float scale_h = sqrtf(2.0f / (float)(H + H));

    srand(seed);
    for (int i = 0; i < 4 * H * D; i++)
        ep->Wi[i] = scale_i * (2.0f * (float)rand() / (float)RAND_MAX - 1.0f);
    for (int i = 0; i < 4 * H * H; i++)
        ep->Wh[i] = scale_h * (2.0f * (float)rand() / (float)RAND_MAX - 1.0f);

    /* Forget gate bias = 1.0 (standard LSTM initialization) */
    for (int i = 0; i < 4 * H; i++) ep->bias[i] = 0.0f;
    for (int i = H; i < 2 * H; i++) ep->bias[i] = 1.0f;

    float scale_o = sqrtf(2.0f / (float)(H + EP_N_CLASSES));
    for (int i = 0; i < EP_N_CLASSES * H; i++)
        ep->out_w[i] = scale_o * (2.0f * (float)rand() / (float)RAND_MAX - 1.0f);

    for (int i = 0; i < D; i++) ep->norm_w[i] = 1.0f;
}

void mimi_ep_destroy(MimiEndpointer *ep) {
    if (!ep) return;
    free(ep->norm_w); free(ep->norm_b);
    free(ep->Wi); free(ep->Wh); free(ep->bias);
    free(ep->out_w); free(ep->out_b);
    free(ep->h); free(ep->c);
    free(ep->gates); free(ep->normed);
    free(ep);
}

EndpointResult mimi_ep_process(MimiEndpointer *ep, const float *latents) {
    EndpointResult res = {0};
    if (!ep || !latents) return res;

    int D = ep->latent_dim;
    int H = ep->hidden_dim;

    /* LayerNorm input */
    layer_norm_1d(ep->normed, latents, ep->norm_w, ep->norm_b, D);

    /* LSTM: gates = Wi @ x + Wh @ h_prev + bias */
#ifdef __APPLE__
    cblas_sgemv(CblasRowMajor, CblasNoTrans,
                4 * H, D, 1.0f,
                ep->Wi, D, ep->normed, 1,
                0.0f, ep->gates, 1);
    cblas_sgemv(CblasRowMajor, CblasNoTrans,
                4 * H, H, 1.0f,
                ep->Wh, H, ep->h, 1,
                1.0f, ep->gates, 1);
    vDSP_vadd(ep->gates, 1, ep->bias, 1, ep->gates, 1, 4 * H);
#else
    /* Fallback: manual matmul */
    for (int i = 0; i < 4 * H; i++) {
        float sum = ep->bias[i];
        for (int j = 0; j < D; j++) sum += ep->Wi[i * D + j] * ep->normed[j];
        for (int j = 0; j < H; j++) sum += ep->Wh[i * H + j] * ep->h[j];
        ep->gates[i] = sum;
    }
#endif

    /* Split gates and apply activations */
    float *gi = ep->gates;           /* input gate */
    float *gf = ep->gates + H;      /* forget gate */
    float *gg = ep->gates + 2 * H;  /* cell gate */
    float *go = ep->gates + 3 * H;  /* output gate */

    for (int i = 0; i < H; i++) {
        float i_gate = sigmoid(gi[i]);
        float f_gate = sigmoid(gf[i]);
        float g_val  = tanhf(gg[i]);
        float o_gate = sigmoid(go[i]);

        ep->c[i] = f_gate * ep->c[i] + i_gate * g_val;
        ep->h[i] = o_gate * tanhf(ep->c[i]);
    }

    /* Output projection: logits = out_w @ h + out_b */
    float logits[EP_N_CLASSES];
#ifdef __APPLE__
    cblas_sgemv(CblasRowMajor, CblasNoTrans,
                EP_N_CLASSES, H, 1.0f,
                ep->out_w, H, ep->h, 1,
                0.0f, logits, 1);
    vDSP_vadd(logits, 1, ep->out_b, 1, logits, 1, EP_N_CLASSES);
#else
    for (int i = 0; i < EP_N_CLASSES; i++) {
        logits[i] = ep->out_b[i];
        for (int j = 0; j < H; j++) logits[i] += ep->out_w[i * H + j] * ep->h[j];
    }
#endif

    /* Softmax */
    float max_l = logits[0];
    for (int i = 1; i < EP_N_CLASSES; i++)
        if (logits[i] > max_l) max_l = logits[i];
    float sum_exp = 0.0f;
    float probs[EP_N_CLASSES];
    for (int i = 0; i < EP_N_CLASSES; i++) {
        probs[i] = expf(logits[i] - max_l);
        sum_exp += probs[i];
    }
    for (int i = 0; i < EP_N_CLASSES; i++)
        probs[i] /= sum_exp;

    res.prob_silence = probs[EP_SILENCE];
    res.prob_speech  = probs[EP_SPEECH];
    res.prob_ending  = probs[EP_ENDING];
    res.prob_eot     = probs[EP_END_TURN];

    /* Argmax for class */
    int best = 0;
    for (int i = 1; i < EP_N_CLASSES; i++)
        if (probs[i] > probs[best]) best = i;
    res.cls = (EndpointClass)best;

    /* EMA smoothing on P(eot) — alpha = 0.3 */
    ep->smoothed_eot = 0.3f * probs[EP_END_TURN] + 0.7f * ep->smoothed_eot;

    /* Track speech activity */
    if (res.cls == EP_SPEECH) ep->speech_active = 1;

    /* Consecutive EOT frame tracking */
    if (ep->smoothed_eot >= ep->eot_threshold && ep->speech_active) {
        ep->consec_eot++;
    } else {
        ep->consec_eot = 0;
    }

    /* Trigger detection */
    if (ep->consec_eot >= ep->consec_required && !ep->triggered) {
        ep->triggered = 1;
        ep->frames_since_speech_end = 0;
    }

    if (ep->triggered && ep->frames_since_speech_end >= 0)
        ep->frames_since_speech_end++;

    res.consec_eot = ep->consec_eot;
    res.triggered = ep->triggered;
    ep->prev_class = res.cls;
    ep->total_frames++;

    return res;
}

float mimi_ep_eot_prob(const MimiEndpointer *ep) {
    return ep ? ep->smoothed_eot : 0.0f;
}

int mimi_ep_triggered(const MimiEndpointer *ep) {
    return ep ? ep->triggered : 0;
}

void mimi_ep_reset(MimiEndpointer *ep) {
    if (!ep) return;
    int H = ep->hidden_dim;
    memset(ep->h, 0, H * sizeof(float));
    memset(ep->c, 0, H * sizeof(float));
    ep->prev_class = EP_SILENCE;
    ep->smoothed_eot = 0.0f;
    ep->consec_eot = 0;
    ep->triggered = 0;
    ep->speech_active = 0;
    ep->frames_since_speech_end = -1;
    ep->total_frames = 0;
}

void mimi_ep_set_threshold(MimiEndpointer *ep, float threshold) {
    if (ep) ep->eot_threshold = threshold;
}

int mimi_ep_latency_frames(const MimiEndpointer *ep) {
    return ep ? ep->frames_since_speech_end : -1;
}
