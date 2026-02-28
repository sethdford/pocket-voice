/**
 * audio_watermark.c — Spread-spectrum audio watermarking for TTS output.
 *
 * Technique:
 *   1. Zero-pad frame to fft_size, apply Hann analysis window
 *   2. Forward FFT via vDSP_fft_zrip (AMX-accelerated)
 *   3. Measure band RMS in the watermark frequency range (1–4 kHz)
 *   4. Generate per-frame Gold-code PN sequence (key + frame index)
 *   5. Create watermark-only spectrum: PN chips at embed level below band RMS
 *   6. IFFT the watermark-only spectrum back to time domain
 *   7. Add watermark signal directly to original PCM (no OLA needed)
 *
 * Detection (matched filtering):
 *   1. For each fft_size frame: Hann window, forward FFT
 *   2. Accumulate raw dot-product (val * chip) across all frames
 *   3. Compute z-score: |Σ val*chip| / sqrt(Σ val²)
 *   4. Map z-score to [0, 1] via z/6 (z > 1.8 → score > 0.3)
 *
 * All vector math uses Apple Accelerate (vDSP). Zero allocations after create().
 */

#ifndef ACCELERATE_NEW_LAPACK
#define ACCELERATE_NEW_LAPACK
#endif

#include "audio_watermark.h"
#include <Accelerate/Accelerate.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ── Constants ─────────────────────────────────────────────────────────── */

#define WM_FREQ_LO_HZ     1000    /* Lower watermark band (Hz) */
#define WM_FREQ_HI_HZ     4000    /* Upper watermark band (Hz) */
#define WM_EMBED_DB       -17.0f   /* Watermark level relative to band RMS */
#define WM_DETECT_THRESH    0.3f   /* Correlation threshold for detection */
#define WM_PAYLOAD_BITS      49    /* 1 + 32 + 16 = ai_flag + timestamp + model_id */
#define WM_CHIPS_PER_BIT     8    /* Gold-code chips per payload bit */
#define WM_MAX_FFT_SIZE  65536    /* Upper bound on computed FFT size */

/* ── Gold-code PN sequence generator ───────────────────────────────────── */

/**
 * Gold code generator: two maximal-length LFSRs XORed together.
 *   LFSR1: x^31 + x^3 + 1  (primitive, period 2^31 - 1)
 *   LFSR2: x^31 + x^13 + 1 (primitive, period 2^31 - 1)
 */
typedef struct {
    uint32_t lfsr1;
    uint32_t lfsr2;
} GoldGen;

static void gold_seed(GoldGen *g, const uint8_t *key, int key_len) {
    /* Derive two LFSR seeds from key via simple mixing */
    uint32_t s1 = 0x6A09E667u; /* sqrt(2) fractional */
    uint32_t s2 = 0xBB67AE85u; /* sqrt(3) fractional */
    for (int i = 0; i < key_len; i++) {
        s1 ^= (uint32_t)key[i] << ((i % 4) * 8);
        s1 = (s1 << 7) | (s1 >> 25);
        s2 ^= (uint32_t)key[i] << (((i + 1) % 4) * 8);
        s2 = (s2 << 13) | (s2 >> 19);
    }
    /* Ensure non-zero */
    g->lfsr1 = s1 ? s1 : 1u;
    g->lfsr2 = s2 ? s2 : 1u;
}

static int gold_next_bit(GoldGen *g) {
    /* LFSR1: x^31 + x^3 + 1 (period 2^31 - 1) */
    uint32_t bit1 = ((g->lfsr1 >> 30) ^ (g->lfsr1 >> 2)) & 1u;
    g->lfsr1 = (g->lfsr1 << 1) | bit1;

    /* LFSR2: x^31 + x^13 + 1 (primitive, period 2^31 - 1) */
    uint32_t bit2 = ((g->lfsr2 >> 30) ^ (g->lfsr2 >> 12)) & 1u;
    g->lfsr2 = (g->lfsr2 << 1) | bit2;

    return (int)(bit1 ^ bit2);
}

/** Generate PN chip: +1.0 or -1.0 */
static float gold_next_chip(GoldGen *g) {
    return gold_next_bit(g) ? 1.0f : -1.0f;
}

/**
 * Derive a per-frame seed by mixing the key-derived state with the frame
 * index using a splitmix64-style hash. This defeats spectral averaging
 * attacks by making the PN sequence different for every frame.
 */
static void gold_seed_for_frame(GoldGen *g, const uint8_t *key, int key_len,
                                 uint32_t frame_idx) {
    gold_seed(g, key, key_len);
    uint64_t z = ((uint64_t)g->lfsr1 << 32) | (uint64_t)g->lfsr2;
    z += (uint64_t)frame_idx * 0x9E3779B97F4A7C15ULL;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    z = z ^ (z >> 31);
    g->lfsr1 = (uint32_t)(z >> 32);
    if (!g->lfsr1) g->lfsr1 = 1u;
    g->lfsr2 = (uint32_t)(z & 0xFFFFFFFFu);
    if (!g->lfsr2) g->lfsr2 = 1u;
}

/* ── PN sequence generation (per-frame) ───────────────────────────────── */

/**
 * Generate a payload-modulated PN sequence for a specific frame.
 *
 * Chip-to-bit mapping is INTERLEAVED: chip i carries payload bit
 * (i % WM_PAYLOAD_BITS) if the chip-within-bit index
 * (i / WM_PAYLOAD_BITS) < WM_CHIPS_PER_BIT, else it is an
 * unmodulated pilot.  This spreads each bit's chips every
 * WM_PAYLOAD_BITS bins (~290 Hz) across the watermark band,
 * giving frequency diversity and robust extraction even when
 * formant peaks overlap some bins.
 */
static void gen_pn_for_frame(const uint8_t *key, int key_len,
                              const int *payload_bits, int n_wm_bins,
                              float *out, uint32_t frame_idx) {
    GoldGen gen;
    gold_seed_for_frame(&gen, key, key_len, frame_idx);

    int total_payload_chips = WM_PAYLOAD_BITS * WM_CHIPS_PER_BIT;
    for (int i = 0; i < n_wm_bins; i++) {
        float pn_chip = gold_next_chip(&gen);
        if (i < total_payload_chips) {
            int b = i % WM_PAYLOAD_BITS;        /* which payload bit */
            float data_sign = payload_bits[b] ? 1.0f : -1.0f;
            out[i] = pn_chip * data_sign;
        } else {
            out[i] = pn_chip;                   /* pilot */
        }
    }
}

/**
 * Generate unmodulated base PN sequence for a specific frame (for extraction).
 */
static void gen_base_pn_for_frame(const uint8_t *key, int key_len,
                                   int n_wm_bins, float *out,
                                   uint32_t frame_idx) {
    GoldGen gen;
    gold_seed_for_frame(&gen, key, key_len, frame_idx);
    for (int i = 0; i < n_wm_bins; i++) {
        out[i] = gold_next_chip(&gen);
    }
}

/* ── Payload serialization ─────────────────────────────────────────────── */

static void payload_to_bits(const AudioWatermarkPayload *p, int *bits) {
    /* Bit 0: AI-generated flag */
    bits[0] = p->ai_generated ? 1 : 0;

    /* Bits 1-32: timestamp (MSB first) */
    for (int i = 0; i < 32; i++) {
        bits[1 + i] = (p->timestamp >> (31 - i)) & 1;
    }

    /* Bits 33-48: model_id (MSB first) */
    for (int i = 0; i < 16; i++) {
        bits[33 + i] = (p->model_id >> (15 - i)) & 1;
    }
}

static void bits_to_payload(const int *bits, AudioWatermarkPayload *p) {
    p->ai_generated = (uint8_t)bits[0];

    p->timestamp = 0;
    for (int i = 0; i < 32; i++) {
        p->timestamp |= ((uint32_t)bits[1 + i]) << (31 - i);
    }

    p->model_id = 0;
    for (int i = 0; i < 16; i++) {
        p->model_id |= ((uint16_t)bits[33 + i]) << (15 - i);
    }
}

/* ── AudioWatermark struct ─────────────────────────────────────────────── */

struct AudioWatermark {
    int sample_rate;
    int frame_size;       /* User-requested frame size (hop for processing) */
    int fft_size;         /* Power-of-2 for watermark FFT (>= frame_size) */
    int half_fft;         /* fft_size / 2 */
    int log2n;

    int bin_lo;           /* First watermark bin index */
    int bin_hi;           /* Last watermark bin index (exclusive) */
    int n_wm_bins;        /* Number of watermark bins */

    int enabled;

    /* Secret key (copied) */
    uint8_t *key;
    int      key_len;

    /* Current payload */
    AudioWatermarkPayload payload;
    int payload_bits[WM_PAYLOAD_BITS];

    /* Pre-allocated PN sequence buffer (filled per-frame during embed).
     * Length = n_wm_bins. Each chip is +1.0 or -1.0. */
    float *pn_sequence;

    /* vDSP FFT */
    FFTSetup fft_setup;

    /* Pre-allocated buffers */
    float *fft_real;      /* [half_fft] split complex real */
    float *fft_imag;      /* [half_fft] split complex imag */
    float *frame_buf;     /* [fft_size] scratch for FFT/IFFT */
    float *window;        /* [fft_size] Hann window of frame_size, zero-padded */
};

/* ── Helpers ───────────────────────────────────────────────────────────── */

/** Round up to next power of 2. Uses unsigned to prevent overflow. */
static unsigned int next_pow2(unsigned int n) {
    if (n == 0) return 1u;
    if (n > (unsigned int)WM_MAX_FFT_SIZE) return (unsigned int)WM_MAX_FFT_SIZE;
    unsigned int p = 1u;
    while (p < n) p <<= 1;
    return p;
}

static int compute_log2(int n) {
    int l = 0;
    while (n > 1) { l++; n >>= 1; }
    return l;
}

/* ── Public API ────────────────────────────────────────────────────────── */

AudioWatermark *audio_watermark_create(int sample_rate, int frame_size,
                                       const uint8_t *key, int key_len) {
    if (sample_rate <= 0 || frame_size <= 0) {
        fprintf(stderr, "[audio_watermark] invalid sample_rate=%d frame_size=%d\n",
                sample_rate, frame_size);
        return NULL;
    }
    if (!key || key_len < 4) {
        fprintf(stderr, "[audio_watermark] key must be at least 4 bytes\n");
        return NULL;
    }

    AudioWatermark *wm = calloc(1, sizeof(AudioWatermark));
    if (!wm) return NULL;

    wm->sample_rate = sample_rate;
    wm->frame_size  = frame_size;
    wm->enabled     = 1;

    /* Compute minimum FFT size to get enough bins for the full payload.
     * Need at least WM_PAYLOAD_BITS * WM_CHIPS_PER_BIT bins in [1-4 kHz].
     * bins_in_band ~ (band_width / sample_rate) * fft_size
     * So fft_size >= chips_needed * sample_rate / band_width */
    int chips_needed = WM_PAYLOAD_BITS * WM_CHIPS_PER_BIT; /* 392 */
    float band_width = (float)(WM_FREQ_HI_HZ - WM_FREQ_LO_HZ);
    int min_fft_for_bins = (int)ceilf((float)(chips_needed + 4) *
                                       (float)sample_rate / band_width);
    int min_fft = min_fft_for_bins > frame_size ? min_fft_for_bins : frame_size;
    wm->fft_size = (int)next_pow2((unsigned int)min_fft);

    if (wm->fft_size > WM_MAX_FFT_SIZE) {
        fprintf(stderr, "[audio_watermark] computed fft_size %d exceeds max %d\n",
                wm->fft_size, WM_MAX_FFT_SIZE);
        free(wm);
        return NULL;
    }

    wm->half_fft = wm->fft_size / 2;
    wm->log2n    = compute_log2(wm->fft_size);

    /* Compute watermark bin range */
    float bin_hz = (float)sample_rate / (float)wm->fft_size;
    wm->bin_lo = (int)ceilf((float)WM_FREQ_LO_HZ / bin_hz);
    wm->bin_hi = (int)floorf((float)WM_FREQ_HI_HZ / bin_hz);
    if (wm->bin_hi > wm->half_fft) wm->bin_hi = wm->half_fft;
    if (wm->bin_lo < 1) wm->bin_lo = 1;
    wm->n_wm_bins = wm->bin_hi - wm->bin_lo;

    if (wm->n_wm_bins < chips_needed) {
        fprintf(stderr, "[audio_watermark] FFT too small: only %d bins in watermark band "
                "(need %d for %d payload bits x %d chips/bit). "
                "Increase frame_size or sample_rate.\n",
                wm->n_wm_bins, chips_needed, WM_PAYLOAD_BITS, WM_CHIPS_PER_BIT);
        free(wm);
        return NULL;
    }

    /* Copy key */
    wm->key = malloc((size_t)key_len);
    if (!wm->key) { free(wm); return NULL; }
    memcpy(wm->key, key, (size_t)key_len);
    wm->key_len = key_len;

    /* FFT setup */
    wm->fft_setup = vDSP_create_fftsetup(wm->log2n, FFT_RADIX2);
    if (!wm->fft_setup) {
        free(wm->key);
        free(wm);
        return NULL;
    }

    /* Allocate buffers */
    wm->fft_real    = calloc((size_t)wm->half_fft, sizeof(float));
    wm->fft_imag    = calloc((size_t)wm->half_fft, sizeof(float));
    wm->frame_buf   = calloc((size_t)wm->fft_size, sizeof(float));
    wm->window      = calloc((size_t)wm->fft_size, sizeof(float));
    wm->pn_sequence = calloc((size_t)wm->n_wm_bins, sizeof(float));

    if (!wm->fft_real || !wm->fft_imag || !wm->frame_buf ||
        !wm->window || !wm->pn_sequence) {
        audio_watermark_destroy(wm);
        return NULL;
    }

    /* Full fft_size Hann window for analysis.
     * Both embed and detect step by fft_size, so each frame spans
     * fft_size samples and needs a full-length window. */
    vDSP_hann_window(wm->window, (vDSP_Length)wm->fft_size, vDSP_HANN_DENORM);

    /* Default payload */
    wm->payload.ai_generated = 1;
    wm->payload.timestamp    = 0;
    wm->payload.model_id     = 0;
    payload_to_bits(&wm->payload, wm->payload_bits);

    return wm;
}

void audio_watermark_destroy(AudioWatermark *wm) {
    if (!wm) return;
    if (wm->fft_setup) vDSP_destroy_fftsetup(wm->fft_setup);
    free(wm->fft_real);
    free(wm->fft_imag);
    free(wm->frame_buf);
    free(wm->window);
    free(wm->pn_sequence);
    free(wm->key);
    free(wm);
}

void audio_watermark_set_payload(AudioWatermark *wm,
                                 const AudioWatermarkPayload *payload) {
    if (!wm || !payload) return;
    wm->payload = *payload;
    payload_to_bits(&wm->payload, wm->payload_bits);
    /* PN sequence is generated per-frame during embed/detect */
}

void audio_watermark_enable(AudioWatermark *wm, int enable) {
    if (wm) wm->enabled = enable;
}

int audio_watermark_is_enabled(const AudioWatermark *wm) {
    return wm ? wm->enabled : 0;
}

void audio_watermark_reset(AudioWatermark *wm) {
    if (!wm) return;
    /* No persistent frame state -- each frame is processed independently */
}

/**
 * Embed watermark into audio in-place.
 *
 * For each fft_size-aligned frame:
 *   1. Forward FFT (no windowing — keeps spectrum exact for detection)
 *   2. Measure band RMS to set watermark amplitude
 *   3. Generate per-frame PN sequence (key + frame_idx)
 *   4. ADD watermark chips directly to the signal spectrum
 *   5. IFFT to get modified signal, replace PCM
 *
 * Because we modify the spectrum in-place and round-trip through FFT/IFFT,
 * detection sees the exact same watermark chips without spectral smearing.
 * No OLA synthesis needed — non-overlapping frames.
 */
int audio_watermark_embed(AudioWatermark *wm, float *pcm, int n_samples) {
    if (!wm || !pcm || n_samples <= 0) return -1;
    if (!wm->enabled) return 0;

    const int N = wm->fft_size;
    const int half = wm->half_fft;
    int pos = 0;
    uint32_t frame_idx = 0;

    while (pos + N <= n_samples) {
        /* Generate per-frame PN sequence (payload-modulated) */
        gen_pn_for_frame(wm->key, wm->key_len, wm->payload_bits,
                          wm->n_wm_bins, wm->pn_sequence, frame_idx);

        /* Copy frame (no windowing — rectangular window for exact round-trip) */
        memcpy(wm->frame_buf, pcm + pos, (size_t)N * sizeof(float));

        /* Pack into split complex for vDSP real FFT */
        DSPSplitComplex sc = { wm->fft_real, wm->fft_imag };
        vDSP_ctoz((DSPComplex *)wm->frame_buf, 2, &sc, 1, (vDSP_Length)half);

        /* Forward FFT */
        vDSP_fft_zrip(wm->fft_setup, &sc, 1,
                       (vDSP_Length)wm->log2n, FFT_FORWARD);

        /* Compute signal RMS in watermark band for adaptive level */
        float band_energy = 0.0f;
        for (int i = wm->bin_lo; i < wm->bin_hi; i++) {
            band_energy += sc.realp[i] * sc.realp[i] +
                           sc.imagp[i] * sc.imagp[i];
        }
        float band_rms = sqrtf(band_energy / (float)wm->n_wm_bins);

        /* Also compute broadband spectral RMS so that narrowband
         * signals (e.g. pure tones below 1 kHz) still get a
         * meaningful watermark.  Use the larger of the two. */
        float bb_energy = 0.0f;
        for (int i = 1; i < half; i++) {
            bb_energy += sc.realp[i] * sc.realp[i] +
                         sc.imagp[i] * sc.imagp[i];
        }
        float bb_rms = sqrtf(bb_energy / (float)(half - 1));
        float ref_level = band_rms;
        if (bb_rms * 0.01f > ref_level) ref_level = bb_rms * 0.01f;

        /* Watermark amplitude: WM_EMBED_DB below reference level.
         * Use a small absolute floor for near-silent frames. */
        float wm_amp;
        if (ref_level > 1e-8f) {
            wm_amp = ref_level * powf(10.0f, WM_EMBED_DB / 20.0f);
        } else {
            wm_amp = 1e-6f;
        }

        /* Add watermark directly to signal spectrum in-place.
         * This ensures detection FFT sees exact chip values. */
        for (int i = 0; i < wm->n_wm_bins; i++) {
            int bin = wm->bin_lo + i;
            float chip = wm->pn_sequence[i];
            sc.realp[bin] += wm_amp * chip;
            sc.imagp[bin] += wm_amp * chip;
        }

        /* IFFT to reconstruct modified signal */
        vDSP_fft_zrip(wm->fft_setup, &sc, 1,
                       (vDSP_Length)wm->log2n, FFT_INVERSE);

        /* Unpack split complex to interleaved real */
        vDSP_ztoc(&sc, 1, (DSPComplex *)wm->frame_buf, 2, (vDSP_Length)half);

        /* Scale by 1/(2*N) -- vDSP convention for round-trip FFT */
        float scale = 1.0f / (2.0f * (float)N);
        vDSP_vsmul(wm->frame_buf, 1, &scale,
                   wm->frame_buf, 1, (vDSP_Length)N);

        /* Replace PCM with watermarked signal */
        memcpy(pcm + pos, wm->frame_buf, (size_t)N * sizeof(float));

        frame_idx++;
        pos += N;
    }

    return 0;
}

float audio_watermark_detect(const AudioWatermark *wm,
                             const float *pcm, int n_samples) {
    if (!wm || !pcm || n_samples <= 0) return -1.0f;

    const int N = wm->fft_size;
    const int half = wm->half_fft;

    /* Allocate temporary buffers (detect is logically const) */
    float *tmp_frame = (float *)malloc((size_t)N * sizeof(float));
    float *tmp_real  = (float *)malloc((size_t)half * sizeof(float));
    float *tmp_imag  = (float *)malloc((size_t)half * sizeof(float));
    float *tmp_pn    = (float *)malloc((size_t)wm->n_wm_bins * sizeof(float));
    if (!tmp_frame || !tmp_real || !tmp_imag || !tmp_pn) {
        free(tmp_frame);
        free(tmp_real);
        free(tmp_imag);
        free(tmp_pn);
        return -1.0f;
    }

    /* Matched filtering: accumulate raw correlation and noise energy
     * across ALL frames, then compute a single z-score.
     *
     * Under H0 (no watermark): val*chip are zero-mean, so matched_total
     * grows as sqrt(N_total) while noise_total grows as N_total.
     * z = |matched_total| / sqrt(noise_total) ~ N(0, 1).
     *
     * Under H1 (watermark present): each val*chip has mean ~ wm_amp,
     * so matched_total grows as N_total * wm_amp while noise_total
     * grows as N_total * (sig_var + wm_var).
     * z = N_total * wm_amp / sqrt(N_total * sig_var) = wm_amp/sig_rms * sqrt(N_total).
     * This grows with more frames — the key advantage over per-frame Pearson. */
    double matched_total = 0.0;
    double noise_total = 0.0;
    int n_frames = 0;

    int pos = 0;
    uint32_t frame_idx = 0;
    while (pos + N <= n_samples) {
        /* Generate per-frame PN (payload-modulated) */
        gen_pn_for_frame(wm->key, wm->key_len, wm->payload_bits,
                          wm->n_wm_bins, tmp_pn, frame_idx);

        /* Copy full fft_size frame (no windowing — matches embed) */
        memcpy(tmp_frame, pcm + pos, (size_t)N * sizeof(float));

        /* Forward FFT */
        DSPSplitComplex sc = { tmp_real, tmp_imag };
        vDSP_ctoz((DSPComplex *)tmp_frame, 2, &sc, 1, (vDSP_Length)half);
        vDSP_fft_zrip(wm->fft_setup, &sc, 1,
                       (vDSP_Length)wm->log2n, FFT_FORWARD);

        /* Accumulate matched filter across watermark bins */
        for (int i = 0; i < wm->n_wm_bins; i++) {
            int bin = wm->bin_lo + i;
            float val = sc.realp[bin] + sc.imagp[bin];
            float chip = tmp_pn[i];

            matched_total += (double)(val * chip);
            noise_total   += (double)(val * val);
        }

        n_frames++;
        frame_idx++;
        pos += N;
    }

    free(tmp_frame);
    free(tmp_real);
    free(tmp_imag);
    free(tmp_pn);

    if (n_frames == 0) return 0.0f;

    /* Compute z-score and map to [0, 1] range.
     * z = |matched_total| / sqrt(noise_total)
     * Under H0: z ~ |N(0,1)|, mean ~0.8, P(z>3.6) < 0.0002.
     * Under H1: z ~ wm_amp/sig_rms * sqrt(N_total), typically 5-20+.
     * Map: score = z / 12, clamped to [0, 1]. Threshold 0.3 → z=3.6. */
    if (noise_total < 1e-20) return 0.0f;
    double z = fabs(matched_total) / sqrt(noise_total);
    float score = (float)(z / 12.0);
    return score > 1.0f ? 1.0f : score;
}

int audio_watermark_extract(const AudioWatermark *wm,
                            const float *pcm, int n_samples,
                            AudioWatermarkPayload *out) {
    if (!wm || !pcm || !out || n_samples <= 0) return -1;

    const int N = wm->fft_size;
    const int half = wm->half_fft;

    float *tmp_frame = (float *)malloc((size_t)N * sizeof(float));
    float *tmp_real  = (float *)malloc((size_t)half * sizeof(float));
    float *tmp_imag  = (float *)malloc((size_t)half * sizeof(float));
    float *base_pn   = (float *)malloc((size_t)wm->n_wm_bins * sizeof(float));
    if (!tmp_frame || !tmp_real || !tmp_imag || !base_pn) {
        free(tmp_frame);
        free(tmp_real);
        free(tmp_imag);
        free(base_pn);
        return -1;
    }

    /* Accumulate chip values per payload bit across all frames */
    float bit_accum[WM_PAYLOAD_BITS];
    memset(bit_accum, 0, sizeof(bit_accum));
    int n_frames = 0;

    int pos = 0;
    uint32_t frame_idx = 0;
    while (pos + N <= n_samples) {
        /* Generate unmodulated base PN for this frame */
        gen_base_pn_for_frame(wm->key, wm->key_len,
                               wm->n_wm_bins, base_pn, frame_idx);

        /* Copy fft_size frame (no windowing — matches embed) */
        memcpy(tmp_frame, pcm + pos, (size_t)N * sizeof(float));

        /* Forward FFT */
        DSPSplitComplex sc = { tmp_real, tmp_imag };
        vDSP_ctoz((DSPComplex *)tmp_frame, 2, &sc, 1, (vDSP_Length)half);
        vDSP_fft_zrip(wm->fft_setup, &sc, 1,
                       (vDSP_Length)wm->log2n, FFT_FORWARD);

        /* Interleaved chip-to-bit mapping (matches gen_pn_for_frame):
         * Chip i → payload bit (i % WM_PAYLOAD_BITS) for the first
         * WM_PAYLOAD_BITS * WM_CHIPS_PER_BIT chips.
         * Use inverse-variance weighting so high-energy signal bins
         * don't dominate the per-bit correlation. */
        int total_payload_chips = WM_PAYLOAD_BITS * WM_CHIPS_PER_BIT;
        for (int i = 0; i < wm->n_wm_bins && i < total_payload_chips; i++) {
            int bin = wm->bin_lo + i;
            float val = sc.realp[bin] + sc.imagp[bin];
            float energy = sc.realp[bin] * sc.realp[bin] +
                           sc.imagp[bin] * sc.imagp[bin];
            float weight = 1.0f / (energy + 1e-10f);
            int b = i % WM_PAYLOAD_BITS;
            bit_accum[b] += val * base_pn[i] * weight;
        }
        n_frames++;
        frame_idx++;
        pos += N;
    }

    free(tmp_frame);
    free(tmp_real);
    free(tmp_imag);
    free(base_pn);

    if (n_frames == 0) return -1;

    /* Decode bits: positive correlation -> bit=1, negative -> bit=0 */
    int decoded_bits[WM_PAYLOAD_BITS];
    for (int b = 0; b < WM_PAYLOAD_BITS; b++) {
        decoded_bits[b] = (bit_accum[b] > 0.0f) ? 1 : 0;
    }

    bits_to_payload(decoded_bits, out);
    return 0;
}
