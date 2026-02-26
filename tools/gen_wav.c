/**
 * gen_wav.c — Generate expressive .wav files with SSML prosody processing.
 *
 * Usage:
 *   ./gen_wav --engine piper|supertonic [-o output.wav] "text or SSML"
 *   ./gen_wav --engine piper --ssml -o expressive.wav   (uses built-in SSML demo)
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>

/* Piper TTS */
extern void *piper_tts_create(const char *model_path);
extern void  piper_tts_destroy(void *tts);
extern int   piper_tts_set_text(void *tts, const char *text);
extern int   piper_tts_synthesize(void *tts);
extern int   piper_tts_get_audio(void *tts, float *buf, int max_samples);
extern int   piper_tts_sample_rate(const void *tts);
extern void  piper_tts_reset(void *tts);

/* Supertonic TTS */
extern void *supertonic_tts_create(const char *model_dir, const char *voice_path);
extern void  supertonic_tts_destroy(void *tts);
extern int   supertonic_tts_set_text(void *tts, const char *text);
extern int   supertonic_tts_synthesize(void *tts, int flow_steps);
extern int   supertonic_tts_get_audio(void *tts, float *buf, int max_samples);
extern int   supertonic_tts_sample_rate(const void *tts);
extern void  supertonic_tts_reset(void *tts);

/* SSML parser */
#include "ssml_parser.h"

/* Text normalizer */
extern void text_auto_normalize(const char *input, char *output, int output_size);

/* Prosody effects (AMX-accelerated) */
extern int  prosody_pitch_shift(const float *input, float *output, int n_samples,
                                float pitch_factor, int fft_size);
extern int  prosody_time_stretch(const float *input, int in_len, float *output,
                                 float rate_factor, float window_ms, int sample_rate);

typedef struct BiquadCascade BiquadCascade;
extern BiquadCascade *prosody_create_formant_eq(float pitch_factor, int sample_rate);
extern int   prosody_apply_biquad(BiquadCascade *bc, float *audio, int n_samples);
extern void  prosody_destroy_biquad(BiquadCascade *bc);
extern void  prosody_soft_limit(float *audio, int n_samples, float threshold, float knee_db);
extern void  prosody_volume(float *audio, int n_samples, float volume_db, float fade_ms, int sr);

/* ── Emotion → prosody mapping (same as pipeline) ── */
typedef struct { const char *name; float pitch, rate, vol_db; } EmoPro;
static const EmoPro EMOS[] = {
    {"happy",     1.08f, 1.05f,  1.5f},
    {"excited",   1.15f, 1.12f,  3.0f},
    {"sad",       0.92f, 0.88f, -2.0f},
    {"angry",     1.05f, 1.08f,  4.0f},
    {"surprised", 1.18f, 1.10f,  2.0f},
    {"warm",      0.97f, 0.95f,  0.5f},
    {"serious",   0.94f, 0.92f,  1.0f},
    {"calm",      0.96f, 0.90f, -1.0f},
    {"confident", 1.03f, 1.02f,  2.0f},
    {"whisper",   0.95f, 0.85f, -6.0f},
    {"emphatic",  1.10f, 0.95f,  3.5f},
    {NULL, 0, 0, 0}
};
static const EmoPro *find_emo(const char *n) {
    if (!n || !*n) return NULL;
    for (int i = 0; EMOS[i].name; i++)
        if (strcasecmp(n, EMOS[i].name) == 0) return &EMOS[i];
    return NULL;
}

/* ── Linear resampler (simple, for sample rate conversion) ── */
static int linear_resample(const float *in, int in_len, int in_sr,
                           float *out, int out_cap, int out_sr) {
    double ratio = (double)in_sr / (double)out_sr;
    int out_len = (int)((double)in_len / ratio);
    if (out_len > out_cap) out_len = out_cap;
    for (int i = 0; i < out_len; i++) {
        double pos = i * ratio;
        int idx = (int)pos;
        double frac = pos - idx;
        if (idx + 1 < in_len)
            out[i] = (float)((1.0 - frac) * in[idx] + frac * in[idx + 1]);
        else if (idx < in_len)
            out[i] = in[idx];
        else
            out[i] = 0.0f;
    }
    return out_len;
}

/* ── WAV writer ── */
static void write_wav(const char *path, const float *pcm, int n, int sr) {
    FILE *f = fopen(path, "wb");
    if (!f) { fprintf(stderr, "Cannot open %s\n", path); return; }
    int16_t *s16 = malloc(n * sizeof(int16_t));
    for (int i = 0; i < n; i++) {
        float s = pcm[i] * 32767.0f;
        if (s > 32767.0f) s = 32767.0f;
        if (s < -32768.0f) s = -32768.0f;
        s16[i] = (int16_t)s;
    }
    uint32_t ds = (uint32_t)(n * 2), fs = 36 + ds;
    uint32_t sr32 = (uint32_t)sr, br = sr32 * 2;
    uint16_t ch = 1, bits = 16, ba = 2, pcmf = 1;
    uint32_t fmtsz = 16;
    fwrite("RIFF", 1, 4, f); fwrite(&fs, 4, 1, f);
    fwrite("WAVE", 1, 4, f); fwrite("fmt ", 1, 4, f);
    fwrite(&fmtsz, 4, 1, f); fwrite(&pcmf, 2, 1, f);
    fwrite(&ch, 2, 1, f); fwrite(&sr32, 4, 1, f);
    fwrite(&br, 4, 1, f); fwrite(&ba, 2, 1, f);
    fwrite(&bits, 2, 1, f); fwrite("data", 1, 4, f);
    fwrite(&ds, 4, 1, f); fwrite(s16, 2, (size_t)n, f);
    fclose(f); free(s16);
    fprintf(stderr, "  Wrote %s: %.1fs @ %dHz\n", path, (float)n / sr, sr);
}

/* Max output buffer: 60s at 24kHz */
#define MAX_OUT (24000 * 60)
/* Per-segment max: 30s */
#define SEG_MAX (24000 * 30)

static float g_out[MAX_OUT];
static int g_out_len = 0;

/* Heap-allocated scratch buffers (avoid stack overflow) */
static float *g_raw = NULL;
static float *g_pcm24 = NULL;
static float *g_proc = NULL;
static float *g_stretched = NULL;

static void alloc_scratch(void) {
    g_raw = malloc(SEG_MAX * sizeof(float));
    g_pcm24 = malloc(SEG_MAX * sizeof(float));
    g_proc = malloc(SEG_MAX * sizeof(float));
    g_stretched = malloc(SEG_MAX * sizeof(float));
}

/* ── Synthesize one segment, apply prosody, append to g_out ── */
static void synth_segment(const SSMLSegment *seg, void *tts, int is_piper, int tts_sr) {
    if (seg->is_audio || seg->text[0] == '\0') return;

    /* Normalize text */
    char norm[4096];
    text_auto_normalize(seg->text, norm, sizeof(norm));
    if (norm[0] == '\0') return;

    /* Resolve prosody */
    float pitch = seg->pitch, rate = seg->rate, vol_db = 0.0f;
    if (fabsf(seg->volume - 1.0f) > 0.01f) vol_db = 20.0f * log10f(seg->volume);
    const EmoPro *emo = find_emo(seg->emotion);
    if (emo) { pitch *= emo->pitch; rate *= emo->rate; vol_db += emo->vol_db; }

    int nlen = (int)strlen(norm);
    if (nlen > 0 && norm[nlen-1] == '?' && fabsf(pitch - 1.0f) < 0.01f) {
        pitch = 1.08f;
        if (fabsf(rate - 1.0f) < 0.01f) rate = 0.95f;
    }
    if (nlen > 0 && norm[nlen-1] == '!' && fabsf(pitch - 1.0f) < 0.01f) {
        pitch = 1.06f; vol_db += 1.5f;
    }

    fprintf(stderr, "  Segment: \"%s\"", norm);
    if (emo) fprintf(stderr, " [%s]", emo->name);
    if (fabsf(pitch - 1.0f) > 0.01f) fprintf(stderr, " pitch=%.2f", pitch);
    if (fabsf(rate - 1.0f) > 0.01f) fprintf(stderr, " rate=%.2f", rate);
    if (fabsf(vol_db) > 0.1f) fprintf(stderr, " vol=%+.1fdB", vol_db);
    fprintf(stderr, "\n");

    /* Insert break before */
    if (seg->break_before_ms > 0) {
        int gap = 24000 * seg->break_before_ms / 1000;
        if (g_out_len + gap > MAX_OUT) gap = MAX_OUT - g_out_len;
        memset(g_out + g_out_len, 0, (size_t)gap * sizeof(float));
        g_out_len += gap;
    }

    /* Synthesize */
    int raw_len = 0;

    if (is_piper) {
        piper_tts_reset(tts);
        piper_tts_set_text(tts, norm);
        if (piper_tts_synthesize(tts) != 0) { fprintf(stderr, "  Synthesis failed\n"); return; }
        int n;
        while ((n = piper_tts_get_audio(tts, g_raw + raw_len, SEG_MAX - raw_len)) > 0)
            raw_len += n;
    } else {
        supertonic_tts_reset(tts);
        supertonic_tts_set_text(tts, norm);
        if (supertonic_tts_synthesize(tts, 32) != 0) { fprintf(stderr, "  Synthesis failed\n"); return; }
        int n;
        while ((n = supertonic_tts_get_audio(tts, g_raw + raw_len, SEG_MAX - raw_len)) > 0)
            raw_len += n;
    }

    if (raw_len <= 0) return;

    /* Resample to 24kHz */
    int n24;
    if (tts_sr != 24000) {
        n24 = linear_resample(g_raw, raw_len, tts_sr, g_pcm24, SEG_MAX, 24000);
    } else {
        memcpy(g_pcm24, g_raw, raw_len * sizeof(float));
        n24 = raw_len;
    }

    /* Peak normalize to ±0.9 */
    float peak = 0.0f;
    for (int i = 0; i < n24; i++) { float a = fabsf(g_pcm24[i]); if (a > peak) peak = a; }
    if (peak > 0.9f) {
        float s = 0.9f / peak;
        for (int i = 0; i < n24; i++) g_pcm24[i] *= s;
    }

    float *src = g_pcm24;

    /* Pitch shift */
    if (fabsf(pitch - 1.0f) > 0.01f && n24 >= 2048) {
        prosody_pitch_shift(src, g_proc, n24, pitch, 2048);
        src = g_proc;
        BiquadCascade *eq = prosody_create_formant_eq(pitch, 24000);
        if (eq) { prosody_apply_biquad(eq, src, n24); prosody_destroy_biquad(eq); }
    }

    /* Time-stretch (rate) */
    if (fabsf(rate - 1.0f) > 0.02f && n24 >= 1024) {
        int slen = prosody_time_stretch(src, n24, g_stretched, rate, 30.0f, 24000);
        if (slen > 0 && slen < SEG_MAX) {
            memcpy(g_proc, g_stretched, (size_t)slen * sizeof(float));
            src = g_proc;
            n24 = slen;
        }
    }

    /* Volume */
    if (fabsf(vol_db) > 0.1f) {
        if (src != g_proc) { memcpy(g_proc, src, (size_t)n24 * sizeof(float)); src = g_proc; }
        prosody_volume(src, n24, vol_db, 5.0f, 24000);
    }

    /* Soft limiter */
    if (src != g_proc) { memcpy(g_proc, src, (size_t)n24 * sizeof(float)); src = g_proc; }
    prosody_soft_limit(src, n24, 0.95f, 12.0f);

    /* Append to output */
    int room = MAX_OUT - g_out_len;
    if (n24 > room) n24 = room;
    memcpy(g_out + g_out_len, src, (size_t)n24 * sizeof(float));
    g_out_len += n24;

    /* Insert break after */
    if (seg->break_after_ms > 0) {
        int gap = 24000 * seg->break_after_ms / 1000;
        if (g_out_len + gap > MAX_OUT) gap = MAX_OUT - g_out_len;
        memset(g_out + g_out_len, 0, (size_t)gap * sizeof(float));
        g_out_len += gap;
    }
}

/* ── Built-in SSML demo ── */
static const char *SSML_DEMO =
    "<speak>"
    "<emotion type=\"warm\">Oh, hello there!</emotion> "
    "<break time=\"300ms\"/>"
    "I'm pocket voice, <emotion type=\"confident\">a tiny AI that lives "
    "entirely inside your computer.</emotion> "
    "<break time=\"200ms\"/>"
    "<emotion type=\"excited\">No cloud, no Python, just pure C and Rust!</emotion> "
    "<break time=\"400ms\"/>"
    "<emotion type=\"calm\">Whispering sweet nothings into your ears.</emotion> "
    "<break time=\"300ms\"/>"
    "<prosody rate=\"115%\" pitch=\"+10%\">I'm so fast I finished this sentence "
    "before you started reading it.</prosody> "
    "<break time=\"500ms\"/>"
    "<emotion type=\"happy\">Boop!</emotion>"
    "</speak>";

int main(int argc, char **argv) {
    const char *engine_name = "piper";
    const char *voice = NULL;
    const char *output = "expressive.wav";
    const char *text = NULL;
    int use_ssml_demo = 0;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--engine") == 0 && i + 1 < argc)
            engine_name = argv[++i];
        else if (strcmp(argv[i], "--voice") == 0 && i + 1 < argc)
            voice = argv[++i];
        else if (strcmp(argv[i], "-o") == 0 && i + 1 < argc)
            output = argv[++i];
        else if (strcmp(argv[i], "--ssml") == 0)
            use_ssml_demo = 1;
        else if (argv[i][0] != '-')
            text = argv[i];
    }

    if (!text && !use_ssml_demo) {
        use_ssml_demo = 1;
    }
    if (use_ssml_demo) text = SSML_DEMO;

    alloc_scratch();
    fprintf(stderr, "Engine: %s\n", engine_name);
    fprintf(stderr, "Text: %.100s%s\n", text, strlen(text) > 100 ? "..." : "");

    /* Create TTS engine */
    void *tts = NULL;
    int tts_sr = 22050;
    int is_piper = (strcmp(engine_name, "piper") == 0);

    if (is_piper) {
        const char *model = voice ? voice : "models/piper/en_US-amy-medium.onnx";
        fprintf(stderr, "Loading Piper TTS from %s...\n", model);
        tts = piper_tts_create(model);
        if (!tts) { fprintf(stderr, "Failed to create Piper TTS\n"); return 1; }
        tts_sr = piper_tts_sample_rate(tts);
    } else {
        const char *model_dir = "models/supertonic-2";
        const char *vp = voice ? voice : "models/supertonic-2/voice_styles/F1.json";
        fprintf(stderr, "Loading Supertonic from %s...\n", model_dir);
        tts = supertonic_tts_create(model_dir, vp);
        if (!tts) { fprintf(stderr, "Failed to create Supertonic TTS\n"); return 1; }
        tts_sr = supertonic_tts_sample_rate(tts);
    }
    fprintf(stderr, "Sample rate: %d\n\n", tts_sr);

    /* Parse SSML (passthrough if plain text) */
    SSMLSegment segments[SSML_MAX_SEGMENTS];
    int nseg = ssml_parse(text, segments, SSML_MAX_SEGMENTS);
    fprintf(stderr, "Parsed %d segment%s:\n", nseg, nseg > 1 ? "s" : "");

    g_out_len = 0;

    for (int i = 0; i < nseg; i++) {
        synth_segment(&segments[i], tts, is_piper, tts_sr);
    }

    if (g_out_len <= 0) {
        fprintf(stderr, "No audio generated!\n");
        is_piper ? piper_tts_destroy(tts) : supertonic_tts_destroy(tts);
        return 1;
    }

    fprintf(stderr, "\nTotal: %d samples (%.2fs)\n", g_out_len, (float)g_out_len / 24000.0f);
    write_wav(output, g_out, g_out_len, 24000);

    is_piper ? piper_tts_destroy(tts) : supertonic_tts_destroy(tts);

    fprintf(stderr, "\nDone! Play with: afplay %s\n", output);
    return 0;
}
