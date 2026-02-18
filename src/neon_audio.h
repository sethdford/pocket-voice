/**
 * neon_audio.h — ARM NEON SIMD intrinsics for audio hot loops.
 *
 * Inline header — include from pocket_voice.c (no separate compilation).
 * All functions are static inline for zero call overhead.
 *
 * Provides:
 *   - neon_copy_f32:      Vectorized float memcpy (8 floats/cycle)
 *   - neon_f32_to_s16:    Float32 → int16 PCM conversion with saturation
 *   - neon_s16_to_f32:    Int16 PCM → float32 conversion
 *   - neon_zero_stuff_2x: Zero-stuffing 2x upsample (pre-FIR)
 *   - neon_scale_f32:     Vectorized constant multiply
 *   - neon_mix_f32:       Vectorized a*x + b*(1-x) crossfade
 */

#ifndef NEON_AUDIO_H
#define NEON_AUDIO_H

#include <arm_neon.h>
#include <stdint.h>

/* ═══════════════════════════════════════════════════════════════════════════
 * Vectorized float32 copy — 8 floats per iteration (32 bytes)
 * Use for ring buffer non-wrapping segments instead of memcpy.
 * ═══════════════════════════════════════════════════════════════════════════ */

static inline void neon_copy_f32(float *dst, const float *src, int n) {
    int i = 0;
    for (; i + 8 <= n; i += 8) {
        vst1q_f32(dst + i,     vld1q_f32(src + i));
        vst1q_f32(dst + i + 4, vld1q_f32(src + i + 4));
    }
    for (; i + 4 <= n; i += 4) {
        vst1q_f32(dst + i, vld1q_f32(src + i));
    }
    for (; i < n; i++) dst[i] = src[i];
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Float32 → int16 PCM conversion with saturation
 *
 * Uses NEON saturating narrow (vqmovn) to clamp to [-32768, 32767].
 * Processes 8 samples per iteration.
 * ═══════════════════════════════════════════════════════════════════════════ */

static inline void neon_f32_to_s16(const float *in, int16_t *out, int n) {
    const float32x4_t scale = vdupq_n_f32(32767.0f);
    int i = 0;
    for (; i + 8 <= n; i += 8) {
        float32x4_t a = vmulq_f32(vld1q_f32(in + i), scale);
        float32x4_t b = vmulq_f32(vld1q_f32(in + i + 4), scale);
        int32x4_t ai = vcvtq_s32_f32(a);
        int32x4_t bi = vcvtq_s32_f32(b);
        int16x4_t lo = vqmovn_s32(ai);
        int16x4_t hi = vqmovn_s32(bi);
        vst1q_s16(out + i, vcombine_s16(lo, hi));
    }
    for (; i + 4 <= n; i += 4) {
        float32x4_t a = vmulq_f32(vld1q_f32(in + i), scale);
        int32x4_t ai = vcvtq_s32_f32(a);
        int16x4_t lo = vqmovn_s32(ai);
        vst1_s16(out + i, lo);
    }
    for (; i < n; i++) {
        float s = in[i] * 32767.0f;
        if (s > 32767.0f) s = 32767.0f;
        if (s < -32768.0f) s = -32768.0f;
        out[i] = (int16_t)s;
    }
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Int16 PCM → float32 conversion
 * ═══════════════════════════════════════════════════════════════════════════ */

static inline void neon_s16_to_f32(const int16_t *in, float *out, int n) {
    const float32x4_t inv_scale = vdupq_n_f32(1.0f / 32768.0f);
    int i = 0;
    for (; i + 8 <= n; i += 8) {
        int16x8_t s16 = vld1q_s16(in + i);
        int32x4_t lo32 = vmovl_s16(vget_low_s16(s16));
        int32x4_t hi32 = vmovl_s16(vget_high_s16(s16));
        float32x4_t flo = vmulq_f32(vcvtq_f32_s32(lo32), inv_scale);
        float32x4_t fhi = vmulq_f32(vcvtq_f32_s32(hi32), inv_scale);
        vst1q_f32(out + i, flo);
        vst1q_f32(out + i + 4, fhi);
    }
    for (; i < n; i++) {
        out[i] = (float)in[i] / 32768.0f;
    }
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Zero-stuffing 2x upsample (24kHz → 48kHz pre-FIR)
 *
 * Inserts a zero between each sample and applies 2x gain compensation.
 * Output length = 2 * n.
 *
 * Uses vst2q interleave to write [sample, 0, sample, 0, ...].
 * ═══════════════════════════════════════════════════════════════════════════ */

static inline void neon_zero_stuff_2x(const float *in, float *out, int n) {
    const float32x4_t gain = vdupq_n_f32(2.0f);
    const float32x4_t zero = vdupq_n_f32(0.0f);
    int i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4_t v = vmulq_f32(vld1q_f32(in + i), gain);
        float32x4x2_t interleaved;
        interleaved.val[0] = v;
        interleaved.val[1] = zero;
        vst2q_f32(out + i * 2, interleaved);
    }
    for (; i < n; i++) {
        out[i * 2]     = in[i] * 2.0f;
        out[i * 2 + 1] = 0.0f;
    }
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Vectorized constant scale: out[i] = in[i] * scale
 * ═══════════════════════════════════════════════════════════════════════════ */

static inline void neon_scale_f32(const float *in, float *out, int n, float scale) {
    float32x4_t vs = vdupq_n_f32(scale);
    int i = 0;
    for (; i + 8 <= n; i += 8) {
        vst1q_f32(out + i,     vmulq_f32(vld1q_f32(in + i), vs));
        vst1q_f32(out + i + 4, vmulq_f32(vld1q_f32(in + i + 4), vs));
    }
    for (; i + 4 <= n; i += 4) {
        vst1q_f32(out + i, vmulq_f32(vld1q_f32(in + i), vs));
    }
    for (; i < n; i++) out[i] = in[i] * scale;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Vectorized crossfade: out[i] = a[i] * (1 - t) + b[i] * t
 *   where t ramps from 0 to 1 over n samples.
 * ═══════════════════════════════════════════════════════════════════════════ */

static inline void neon_crossfade_f32(const float *a, const float *b,
                                       float *out, int n) {
    if (n <= 0) return;
    float step = 1.0f / (float)n;
    float32x4_t vstep4 = vdupq_n_f32(step * 4.0f);
    float32x4_t vt = { 0.0f, step, step * 2.0f, step * 3.0f };
    float32x4_t one = vdupq_n_f32(1.0f);

    int i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4_t va = vld1q_f32(a + i);
        float32x4_t vb = vld1q_f32(b + i);
        float32x4_t inv_t = vsubq_f32(one, vt);
        float32x4_t result = vmlaq_f32(vmulq_f32(va, inv_t), vb, vt);
        vst1q_f32(out + i, result);
        vt = vaddq_f32(vt, vstep4);
    }
    for (; i < n; i++) {
        float t = (float)i * step;
        out[i] = a[i] * (1.0f - t) + b[i] * t;
    }
}

#endif /* NEON_AUDIO_H */
