/**
 * lstm_ops.h — Shared LSTM step function for AMX-accelerated inference.
 *
 * Header-only. Used by native_vad.c and mimi_endpointer.c to avoid
 * duplicating the ~30-line gate computation.
 *
 * Standard LSTM equations:
 *   gates = Wi @ x + Wh @ h_prev + bias
 *   [i, f, g, o] = sigmoid/tanh split of gates
 *   c = f * c_prev + i * g
 *   h = o * tanh(c)
 */

#ifndef LSTM_OPS_H
#define LSTM_OPS_H

#include <math.h>
#include <string.h>

#ifdef __APPLE__
#ifndef ACCELERATE_NEW_LAPACK
#define ACCELERATE_NEW_LAPACK
#endif
#include <Accelerate/Accelerate.h>
#endif

static inline float lstm_sigmoid(float x) {
    if (x > 10.0f) return 1.0f;
    if (x < -10.0f) return 0.0f;
    return 1.0f / (1.0f + expf(-x));
}

/**
 * Single LSTM time step: gates → activations → state update.
 *
 * @param Wi      Input-to-hidden weights [4H, D], row-major
 * @param Wh      Hidden-to-hidden weights [4H, H], row-major
 * @param bias    Gate biases [4H]
 * @param x       Input vector, accessed as x[i * x_stride] for i=0..D-1
 * @param x_stride Stride between input elements (1 for contiguous, >1 for column access)
 * @param h       Hidden state [H], read and updated in-place
 * @param c       Cell state [H], read and updated in-place
 * @param gates   Scratch buffer [4H]
 * @param D       Input dimension
 * @param H       Hidden dimension
 */
static inline void lstm_step(
    const float *Wi, const float *Wh, const float *bias,
    const float *x, int x_stride,
    float *h, float *c, float *gates,
    int D, int H)
{
    int G = 4 * H;

#ifdef __APPLE__
    /* gates = Wi @ x (with stride support for non-contiguous input) */
    cblas_sgemv(CblasRowMajor, CblasNoTrans,
                G, D, 1.0f, Wi, D, x, x_stride, 0.0f, gates, 1);
    /* gates += Wh @ h */
    cblas_sgemv(CblasRowMajor, CblasNoTrans,
                G, H, 1.0f, Wh, H, h, 1, 1.0f, gates, 1);
    /* gates += bias */
    vDSP_vadd(gates, 1, bias, 1, gates, 1, G);
#else
    for (int i = 0; i < G; i++) {
        float s = bias[i];
        for (int j = 0; j < D; j++)
            s += Wi[i * D + j] * x[j * x_stride];
        for (int j = 0; j < H; j++)
            s += Wh[i * H + j] * h[j];
        gates[i] = s;
    }
#endif

    /* Gate split (PyTorch order): i, f, g, o */
    float *gi = gates;
    float *gf = gates + H;
    float *gg = gates + 2 * H;
    float *go = gates + 3 * H;

    for (int i = 0; i < H; i++) {
        float i_gate = lstm_sigmoid(gi[i]);
        float f_gate = lstm_sigmoid(gf[i]);
        float g_val  = tanhf(gg[i]);
        float o_gate = lstm_sigmoid(go[i]);

        c[i] = f_gate * c[i] + i_gate * g_val;
        h[i] = o_gate * tanhf(c[i]);
    }
}

#endif /* LSTM_OPS_H */
