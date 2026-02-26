/**
 * tdt_decoder.c — Token Duration Transducer decoder for Apple Silicon.
 *
 * All matrix multiplies run on the AMX coprocessor via cblas_sgemm.
 * LSTM gates use vDSP for element-wise sigmoid/tanh via vecLib.
 *
 * TDT decode loop:
 *   t = 0, prev_token = blank
 *   while t < T:
 *     pred = prediction_net(prev_token)
 *     joint = joint_net(enc[t], pred)
 *     token_logits = joint[:vocab_size]
 *     dur_logits = joint[vocab_size:]
 *     token = argmax(token_logits)
 *     dur = argmax(dur_logits)
 *     if token != blank:
 *       emit(token); prev_token = token
 *     t += max(1, dur)
 */

#ifndef ACCELERATE_NEW_LAPACK
#define ACCELERATE_NEW_LAPACK
#endif
#include <Accelerate/Accelerate.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "tdt_decoder.h"

#define MAX_PRED_LAYERS 4
#define MAX_INNER_LOOP  10  /* Max non-blank emissions per time step */

struct TDTDecoder {
    TDTConfig config;

    /* Borrowed weight pointers */
    const float *embed_w;
    const float *lstm_wi[MAX_PRED_LAYERS];
    const float *lstm_bi[MAX_PRED_LAYERS];
    const float *lstm_wh[MAX_PRED_LAYERS];
    const float *lstm_bh[MAX_PRED_LAYERS];
    const float *joint_enc_w;
    const float *joint_enc_b;
    const float *joint_pred_w;
    const float *joint_pred_b;
    const float *joint_out_w;
    const float *joint_out_b;

    /* LSTM state: h and c for each layer */
    float *lstm_h[MAX_PRED_LAYERS];
    float *lstm_c[MAX_PRED_LAYERS];

    /* Working memory */
    float *lstm_gates;   /* [4 * pred_hidden] */
    float *pred_out;     /* [pred_hidden] — prediction network output */
    float *joint_enc;    /* [joint_dim] — projected encoder frame */
    float *joint_pred;   /* [joint_dim] — projected prediction */
    float *joint_sum;    /* [joint_dim] — enc + pred before activation */
    float *joint_logits; /* [vocab_size + n_durations] — final output */
};

/* ═══════════════════════════════════════════════════════════════════════════
 * LSTM helper: one step for one layer
 * ═══════════════════════════════════════════════════════════════════════════ */

static void sigmoid_vec(float *dst, const float *src, int n) {
    float neg_one = -1.0f;
    vDSP_vsmul(src, 1, &neg_one, dst, 1, n);
    /* exp(-x) */
    int ni = n;
    vvexpf(dst, dst, &ni);
    /* 1 / (1 + exp(-x)) */
    float one = 1.0f;
    vDSP_vsadd(dst, 1, &one, dst, 1, n);
    vDSP_svdiv(&one, dst, 1, dst, 1, n);
}

static void tanh_vec(float *dst, const float *src, int n) {
    int ni = n;
    vvtanhf(dst, src, &ni);
}

static void lstm_step(const float *input, int input_size,
                      float *h, float *c, int hidden_size,
                      const float *wi, const float *bi,
                      const float *wh, const float *bh,
                      float *gates) {
    int gate_size = 4 * hidden_size;

    /* gates = Wi @ input + bi + Wh @ h + bh */
    memcpy(gates, bi, gate_size * sizeof(float));
    vDSP_vadd(gates, 1, bh, 1, gates, 1, gate_size);

    /* Wi @ input: [4H, input_size] @ [input_size] → [4H] */
    cblas_sgemv(CblasRowMajor, CblasNoTrans,
                gate_size, input_size,
                1.0f, wi, input_size,
                input, 1,
                1.0f, gates, 1);

    /* Wh @ h_prev: [4H, H] @ [H] → [4H] */
    cblas_sgemv(CblasRowMajor, CblasNoTrans,
                gate_size, hidden_size,
                1.0f, wh, hidden_size,
                h, 1,
                1.0f, gates, 1);

    int H = hidden_size;
    float *i_gate = gates;
    float *f_gate = gates + H;
    float *g_gate = gates + 2 * H;
    float *o_gate = gates + 3 * H;

    sigmoid_vec(i_gate, i_gate, H);
    sigmoid_vec(f_gate, f_gate, H);
    tanh_vec(g_gate, g_gate, H);
    sigmoid_vec(o_gate, o_gate, H);

    /* c = f * c + i * g */
    vDSP_vmul(f_gate, 1, c, 1, c, 1, H);
    vDSP_vma(i_gate, 1, g_gate, 1, c, 1, c, 1, H);

    /* h = o * tanh(c) */
    tanh_vec(h, c, H);
    vDSP_vmul(o_gate, 1, h, 1, h, 1, H);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Prediction network: embedding + LSTM stack
 * ═══════════════════════════════════════════════════════════════════════════ */

static void prediction_step(TDTDecoder *dec, int token_id) {
    int H = dec->config.pred_hidden;
    int V = dec->config.vocab_size;

    /* Look up embedding for token_id */
    const float *emb;
    if (token_id >= 0 && token_id < V) {
        emb = dec->embed_w + (size_t)token_id * H;
    } else {
        memset(dec->pred_out, 0, H * sizeof(float));
        emb = dec->pred_out;
    }

    /* Run through LSTM layers */
    const float *layer_input = emb;
    for (int l = 0; l < dec->config.pred_layers; l++) {
        lstm_step(layer_input, H,
                  dec->lstm_h[l], dec->lstm_c[l], H,
                  dec->lstm_wi[l], dec->lstm_bi[l],
                  dec->lstm_wh[l], dec->lstm_bh[l],
                  dec->lstm_gates);
        layer_input = dec->lstm_h[l];
    }

    /* Output is the top LSTM layer's hidden state */
    memcpy(dec->pred_out, dec->lstm_h[dec->config.pred_layers - 1],
           H * sizeof(float));
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Joint network: combine encoder + prediction → logits
 * ═══════════════════════════════════════════════════════════════════════════ */

static void joint_step(TDTDecoder *dec, const float *enc_frame) {
    int E = dec->config.encoder_dim;
    int H = dec->config.pred_hidden;
    int J = dec->config.joint_dim;
    int V = dec->config.vocab_size;
    int D = dec->config.n_durations;

    /* Project encoder: joint_enc = joint_enc_w @ enc_frame + joint_enc_b */
    memcpy(dec->joint_enc, dec->joint_enc_b, J * sizeof(float));
    cblas_sgemv(CblasRowMajor, CblasNoTrans,
                J, E, 1.0f,
                dec->joint_enc_w, E,
                enc_frame, 1,
                1.0f, dec->joint_enc, 1);

    /* Project prediction: joint_pred = joint_pred_w @ pred_out + joint_pred_b */
    memcpy(dec->joint_pred, dec->joint_pred_b, J * sizeof(float));
    cblas_sgemv(CblasRowMajor, CblasNoTrans,
                J, H, 1.0f,
                dec->joint_pred_w, H,
                dec->pred_out, 1,
                1.0f, dec->joint_pred, 1);

    /* Sum + ReLU */
    vDSP_vadd(dec->joint_enc, 1, dec->joint_pred, 1, dec->joint_sum, 1, J);
    float zero = 0.0f;
    vDSP_vthres(dec->joint_sum, 1, &zero, dec->joint_sum, 1, J);

    /* Output projection: logits = joint_out_w @ joint_sum + joint_out_b */
    int total = V + D;
    memcpy(dec->joint_logits, dec->joint_out_b, total * sizeof(float));
    cblas_sgemv(CblasRowMajor, CblasNoTrans,
                total, J, 1.0f,
                dec->joint_out_w, J,
                dec->joint_sum, 1,
                1.0f, dec->joint_logits, 1);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Public API
 * ═══════════════════════════════════════════════════════════════════════════ */

TDTDecoder *tdt_decoder_create(const TDTConfig *config,
                                const float *embed_w,
                                const float *const *lstm_wi,
                                const float *const *lstm_bi,
                                const float *const *lstm_wh,
                                const float *const *lstm_bh,
                                const float *joint_enc_w,
                                const float *joint_enc_b,
                                const float *joint_pred_w,
                                const float *joint_pred_b,
                                const float *joint_out_w,
                                const float *joint_out_b) {
    if (!config || config->pred_layers > MAX_PRED_LAYERS) return NULL;

    TDTDecoder *dec = calloc(1, sizeof(TDTDecoder));
    if (!dec) return NULL;

    dec->config = *config;
    dec->embed_w = embed_w;
    dec->joint_enc_w = joint_enc_w;
    dec->joint_enc_b = joint_enc_b;
    dec->joint_pred_w = joint_pred_w;
    dec->joint_pred_b = joint_pred_b;
    dec->joint_out_w = joint_out_w;
    dec->joint_out_b = joint_out_b;

    int H = config->pred_hidden;
    for (int l = 0; l < config->pred_layers; l++) {
        dec->lstm_wi[l] = lstm_wi[l];
        dec->lstm_bi[l] = lstm_bi[l];
        dec->lstm_wh[l] = lstm_wh[l];
        dec->lstm_bh[l] = lstm_bh[l];
        dec->lstm_h[l] = calloc(H, sizeof(float));
        dec->lstm_c[l] = calloc(H, sizeof(float));
        if (!dec->lstm_h[l] || !dec->lstm_c[l]) {
            tdt_decoder_destroy(dec);
            return NULL;
        }
    }

    int J = config->joint_dim;
    int V = config->vocab_size;
    int D = config->n_durations;

    dec->lstm_gates   = malloc(4 * H * sizeof(float));
    dec->pred_out     = calloc(H, sizeof(float));
    dec->joint_enc    = malloc(J * sizeof(float));
    dec->joint_pred   = malloc(J * sizeof(float));
    dec->joint_sum    = malloc(J * sizeof(float));
    dec->joint_logits = malloc((V + D) * sizeof(float));

    if (!dec->lstm_gates || !dec->pred_out || !dec->joint_enc ||
        !dec->joint_pred || !dec->joint_sum || !dec->joint_logits) {
        tdt_decoder_destroy(dec);
        return NULL;
    }

    return dec;
}

int tdt_decoder_decode(TDTDecoder *dec,
                       const float *enc_out, int T,
                       int *tokens, int max_tokens) {
    if (!dec || !enc_out || T <= 0 || !tokens || max_tokens <= 0) return -1;

    int V = dec->config.vocab_size;
    int D = dec->config.n_durations;
    int blank = dec->config.blank_id;
    int E = dec->config.encoder_dim;
    int n_tokens = 0;

    tdt_decoder_reset(dec);

    prediction_step(dec, blank);

    int t = 0;
    while (t < T && n_tokens < max_tokens) {
        const float *enc_frame = enc_out + (size_t)t * E;
        int inner_count = 0;

        while (inner_count < MAX_INNER_LOOP && n_tokens < max_tokens) {
            joint_step(dec, enc_frame);

            float max_tok = dec->joint_logits[0];
            int best_token = 0;
            for (int i = 1; i < V; i++) {
                if (dec->joint_logits[i] > max_tok) {
                    max_tok = dec->joint_logits[i];
                    best_token = i;
                }
            }

            float max_dur = dec->joint_logits[V];
            int best_dur_idx = 0;
            for (int d = 1; d < D; d++) {
                if (dec->joint_logits[V + d] > max_dur) {
                    max_dur = dec->joint_logits[V + d];
                    best_dur_idx = d;
                }
            }

            int dur = dec->config.duration_values[best_dur_idx];

            if (best_token != blank) {
                tokens[n_tokens++] = best_token;
                prediction_step(dec, best_token);
                inner_count++;
            } else {
                t += (dur > 0) ? dur : 1;
                break;
            }
        }

        if (inner_count >= MAX_INNER_LOOP)
            t += 1;
    }

    return n_tokens;
}

void tdt_decoder_reset(TDTDecoder *dec) {
    if (!dec) return;
    int H = dec->config.pred_hidden;
    for (int l = 0; l < dec->config.pred_layers; l++) {
        memset(dec->lstm_h[l], 0, H * sizeof(float));
        memset(dec->lstm_c[l], 0, H * sizeof(float));
    }
    memset(dec->pred_out, 0, H * sizeof(float));
}

void tdt_decoder_destroy(TDTDecoder *dec) {
    if (!dec) return;
    for (int l = 0; l < MAX_PRED_LAYERS; l++) {
        free(dec->lstm_h[l]);
        free(dec->lstm_c[l]);
    }
    free(dec->lstm_gates);
    free(dec->pred_out);
    free(dec->joint_enc);
    free(dec->joint_pred);
    free(dec->joint_sum);
    free(dec->joint_logits);
    free(dec);
}
