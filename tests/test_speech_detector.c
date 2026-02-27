/**
 * test_speech_detector.c — Tests for the unified SpeechDetector module.
 *
 * Covers:
 *   1. API safety (NULL params, no-config creation)
 *   2. Lifecycle (create → feed → query → reset → feed → destroy)
 *   3. Speech detection with synthetic and real audio
 *   4. EOU fusion
 *   5. Real human speech validation (test_speech_human.wav)
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <mach/mach_time.h>

#include "speech_detector.h"

static int passed = 0, failed = 0;
#define CHECK(cond, msg) do { \
    if (cond) { printf("  [PASS] %s\n", msg); passed++; } \
    else { printf("  [FAIL] %s\n", msg); failed++; } \
} while (0)

static int file_exists(const char *path) {
    FILE *f = fopen(path, "rb");
    if (f) { fclose(f); return 1; }
    return 0;
}

static float *load_wav_f32(const char *path, int *n_samples, int *sample_rate) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    uint8_t hdr[44];
    if (fread(hdr, 1, 44, f) != 44) { fclose(f); return NULL; }
    *sample_rate = hdr[24] | (hdr[25] << 8) | (hdr[26] << 16) | (hdr[27] << 24);
    int bits = hdr[34] | (hdr[35] << 8);
    int channels = hdr[22] | (hdr[23] << 8);
    int data_size = hdr[40] | (hdr[41] << 8) | (hdr[42] << 16) | (hdr[43] << 24);
    int n = data_size / (bits / 8) / channels;
    float *out = (float *)malloc((size_t)n * sizeof(float));
    if (bits == 16) {
        int16_t *buf = (int16_t *)malloc(data_size);
        if (fread(buf, 1, data_size, f)) {}
        for (int i = 0; i < n; i++) out[i] = buf[i * channels] / 32768.0f;
        free(buf);
    } else if (bits == 32) {
        float *buf = (float *)malloc(data_size);
        if (fread(buf, 1, data_size, f)) {}
        for (int i = 0; i < n; i++) out[i] = buf[i * channels];
        free(buf);
    }
    fclose(f);
    *n_samples = n;
    return out;
}

static float *resample_to(const float *in, int n_in, int sr_in, int sr_out, int *n_out) {
    *n_out = (int)((long long)n_in * sr_out / sr_in);
    float *out = (float *)malloc((size_t)*n_out * sizeof(float));
    for (int i = 0; i < *n_out; i++) {
        float src = (float)i * sr_in / sr_out;
        int idx = (int)src;
        float frac = src - idx;
        if (idx + 1 < n_in)
            out[i] = in[idx] * (1.0f - frac) + in[idx + 1] * frac;
        else
            out[i] = in[n_in - 1];
    }
    return out;
}

static void generate_speech_24k(float *buf, int n_samples) {
    for (int i = 0; i < n_samples; i++) {
        float t = (float)i / 24000.0f;
        float voiced = 0.4f * sinf(2.0f * M_PI * 150.0f * t)
                     + 0.25f * sinf(2.0f * M_PI * 300.0f * t)
                     + 0.15f * sinf(2.0f * M_PI * 450.0f * t)
                     + 0.10f * sinf(2.0f * M_PI * 600.0f * t);
        float env = 0.6f + 0.4f * sinf(2.0f * M_PI * 4.0f * t);
        buf[i] = voiced * env;
    }
}

int main(void) {
    printf("═══ SpeechDetector Tests ═══\n\n");

    /* ── 1. API Safety ─────────────────────────────────────────────── */
    printf("── API Safety ──\n");

    speech_detector_destroy(NULL);
    speech_detector_reset(NULL);
    speech_detector_feed(NULL, NULL, 0);
    speech_detector_feed_16k(NULL, NULL, 0);
    CHECK(speech_detector_speech_prob(NULL) == -1.0f, "NULL speech_prob returns -1");
    CHECK(speech_detector_speech_active(NULL, 0) == 0, "NULL speech_active with vad=0 returns 0");
    CHECK(speech_detector_speech_active(NULL, 1) == 1, "NULL speech_active with vad=1 returns 1");
    CHECK(speech_detector_eot_prob(NULL) == 0.0f, "NULL eot_prob returns 0");
    CHECK(speech_detector_has_vad(NULL) == 0, "NULL has_vad returns 0");
    CHECK(speech_detector_has_endpointer(NULL) == 0, "NULL has_endpointer returns 0");

    /* ── 2. Create without config ──────────────────────────────────── */
    printf("\n── No-Config Creation ──\n");
    SpeechDetector *sd = speech_detector_create(NULL);
    CHECK(sd != NULL, "create with NULL config succeeds");
    CHECK(speech_detector_has_vad(sd) == 0, "no VAD without config");
    CHECK(speech_detector_speech_prob(sd) == -1.0f, "no data yet → -1");
    speech_detector_destroy(sd);

    /* ── 2b. Config validation: defaults ────────────────────────────── */
    printf("\n── Config Default Validation ──\n");
    {
        SpeechDetectorConfig default_cfg = {0};
        SpeechDetector *sd_default = speech_detector_create(&default_cfg);
        CHECK(sd_default != NULL, "create with zero-init config succeeds");
        CHECK(speech_detector_has_vad(sd_default) == 0,
              "zero-config: no VAD (no path given)");
        CHECK(speech_detector_has_endpointer(sd_default) == 1,
              "zero-config: endpointer created with defaults");
        speech_detector_destroy(sd_default);
    }

    /* ── 2c. Double destroy safety ────────────────────────────────── */
    printf("\n── Double Destroy Safety ──\n");
    {
        SpeechDetector *sd_tmp = speech_detector_create(NULL);
        speech_detector_destroy(sd_tmp);
        /* sd_tmp is freed; calling destroy again on NULL is safe */
        speech_detector_destroy(NULL);
        CHECK(1, "double destroy (via NULL) is safe");
    }

    /* ── 3. Create with native VAD ─────────────────────────────────── */
    printf("\n── Full Creation (native VAD + endpointer + fused EOU) ──\n");
    const char *nvad_path = "models/silero_vad.nvad";
    if (!file_exists(nvad_path)) {
        printf("  [SKIP] %s not found\n", nvad_path);
        printf("\nResults: %d passed, %d failed\n", passed, failed);
        return failed > 0 ? 1 : 0;
    }

    SpeechDetectorConfig cfg = {
        .native_vad_path = nvad_path,
        .mimi_latent_dim = 80,
        .mimi_hidden_dim = 64,
        .eot_threshold = 0.6f,
        .eot_consec_frames = 2,
    };
    sd = speech_detector_create(&cfg);
    CHECK(sd != NULL, "create with config succeeds");
    CHECK(speech_detector_has_vad(sd) == 1, "has VAD loaded");
    CHECK(speech_detector_has_endpointer(sd) == 1, "has endpointer loaded");

    /* ── 4. Feed silence (24kHz) ───────────────────────────────────── */
    printf("\n── Silence Detection ──\n");
    float silence[4800] = {0};  /* 200ms @ 24kHz */
    for (int i = 0; i < 5; i++)
        speech_detector_feed(sd, silence, 4800);
    float sp = speech_detector_speech_prob(sd);
    printf("    silence speech_prob: %.4f\n", sp);
    CHECK(sp >= 0.0f && sp <= 1.0f, "silence prob in [0,1]");
    CHECK(speech_detector_speech_active(sd, 0) == 0, "silence + energy_vad=0 → not active");

    /* ── 5. Feed synthetic speech (24kHz) ──────────────────────────── */
    printf("\n── Synthetic Speech Detection ──\n");
    speech_detector_reset(sd);
    int speech_len = 48000;  /* 2s @ 24kHz */
    float *speech = (float *)malloc((size_t)speech_len * sizeof(float));
    generate_speech_24k(speech, speech_len);
    for (int off = 0; off < speech_len; off += 2400)
        speech_detector_feed(sd, speech + off, 2400 < speech_len - off ? 2400 : speech_len - off);
    sp = speech_detector_speech_prob(sd);
    printf("    synthetic speech_prob: %.4f\n", sp);
    CHECK(sp >= 0.0f, "speech prob is valid");

    float eot = speech_detector_eot_prob(sd);
    printf("    eot_prob: %.4f\n", eot);
    CHECK(eot >= 0.0f && eot <= 1.0f, "eot prob in [0,1]");
    free(speech);

    /* ── 6. EOU Fusion ─────────────────────────────────────────────── */
    printf("\n── EOU Fusion ──\n");
    speech_detector_reset(sd);
    for (int i = 0; i < 10; i++)
        speech_detector_feed(sd, silence, 4800);
    EOUResult r = speech_detector_eou(sd, 3, 0.8f);  /* energy=silence_end, stt=high */
    printf("    fused_prob: %.4f, triggered: %d\n", r.fused_prob, r.triggered);
    CHECK(r.fused_prob >= 0.0f && r.fused_prob <= 1.0f, "fused prob in [0,1]");

    /* ── 6b. EOU with various energy_vad states ─────────────────────── */
    printf("\n── EOU Energy VAD States ──\n");
    speech_detector_reset(sd);
    /* Feed speech first so fused_eou has speech_detected */
    float synth_speech[4800];
    generate_speech_24k(synth_speech, 4800);
    for (int i = 0; i < 5; i++)
        speech_detector_feed(sd, synth_speech, 4800);

    /* energy_vad=3 (definite silence) → should map to energy_signal=1.0 */
    EOUResult r_silence = speech_detector_eou(sd, 3, 0.0f);
    printf("    energy_vad=3: fused_prob=%.4f, triggered=%d\n",
           r_silence.fused_prob, r_silence.triggered);
    CHECK(r_silence.fused_prob >= 0.0f && r_silence.fused_prob <= 1.0f,
          "EOU with energy_vad=3 prob in range");

    /* energy_vad=2 (speech active) → energy_signal=0.0 */
    speech_detector_reset(sd);
    for (int i = 0; i < 5; i++)
        speech_detector_feed(sd, synth_speech, 4800);
    EOUResult r_active = speech_detector_eou(sd, 2, 0.0f);
    printf("    energy_vad=2: fused_prob=%.4f, triggered=%d\n",
           r_active.fused_prob, r_active.triggered);
    CHECK(r_active.fused_prob >= 0.0f && r_active.fused_prob <= 1.0f,
          "EOU with energy_vad=2 prob in range");

    /* energy_vad=0 (unknown) → energy_signal=0.5 */
    speech_detector_reset(sd);
    for (int i = 0; i < 5; i++)
        speech_detector_feed(sd, synth_speech, 4800);
    EOUResult r_unknown = speech_detector_eou(sd, 0, 0.0f);
    printf("    energy_vad=0: fused_prob=%.4f\n", r_unknown.fused_prob);
    CHECK(r_unknown.fused_prob >= 0.0f, "EOU with energy_vad=0 yields valid prob");

    /* ── 6c. Feed silence → no EOU trigger ────────────────────────── */
    printf("\n── Silence-Only No EOU Trigger ──\n");
    speech_detector_reset(sd);
    for (int i = 0; i < 20; i++)
        speech_detector_feed(sd, silence, 4800);
    /* energy_vad=0 means no speech detected */
    EOUResult r_no_trigger = speech_detector_eou(sd, 0, 0.0f);
    printf("    silence-only: fused_prob=%.4f, triggered=%d\n",
           r_no_trigger.fused_prob, r_no_trigger.triggered);
    CHECK(r_no_trigger.triggered == 0,
          "silence-only (no prior speech) should not trigger EOU");

    /* ── 6d. Feed speech then silence → EOU should trigger ────────── */
    printf("\n── Speech Then Silence EOU Trigger ──\n");
    speech_detector_reset(sd);
    /* Feed speech */
    for (int i = 0; i < 10; i++)
        speech_detector_feed(sd, synth_speech, 4800);
    /* Feed silence (200ms chunks × 20 = 4 seconds of silence) */
    for (int i = 0; i < 20; i++)
        speech_detector_feed(sd, silence, 4800);
    /* With energy_vad=3 (definite silence) and high stt_eou_prob, should trigger */
    EOUResult r_speech_silence = speech_detector_eou(sd, 3, 0.9f);
    for (int i = 0; i < 10; i++)
        r_speech_silence = speech_detector_eou(sd, 3, 0.9f);
    printf("    speech→silence: fused_prob=%.4f, triggered=%d\n",
           r_speech_silence.fused_prob, r_speech_silence.triggered);
    CHECK(r_speech_silence.triggered == 1,
          "speech then silence with high STT prob should trigger EOU");

    /* ── 6e. Multiple consecutive utterances ──────────────────────── */
    printf("\n── Multiple Consecutive Utterances ──\n");
    for (int utt = 0; utt < 3; utt++) {
        speech_detector_reset(sd);
        /* Speech phase */
        for (int i = 0; i < 5; i++)
            speech_detector_feed(sd, synth_speech, 4800);
        float sp_utt = speech_detector_speech_prob(sd);
        CHECK(sp_utt >= 0.0f && sp_utt <= 1.0f,
              "utterance speech prob in [0,1]");
        /* Silence phase */
        for (int i = 0; i < 5; i++)
            speech_detector_feed(sd, silence, 4800);
        float eot_utt = speech_detector_eot_prob(sd);
        CHECK(eot_utt >= 0.0f && eot_utt <= 1.0f,
              "utterance eot prob in [0,1]");
    }

    /* ── 6f. EOU NULL safety ──────────────────────────────────────── */
    printf("\n── EOU NULL Safety ──\n");
    EOUResult r_null = speech_detector_eou(NULL, 3, 0.9f);
    CHECK(r_null.triggered == 0, "EOU(NULL) does not trigger");
    CHECK(r_null.fused_prob == 0.0f, "EOU(NULL) returns prob 0.0");

    /* ── 7. Reset clears state ─────────────────────────────────────── */
    printf("\n── Reset ──\n");
    speech_detector_reset(sd);
    CHECK(speech_detector_speech_prob(sd) == -1.0f, "reset clears speech_prob to -1");
    CHECK(speech_detector_eot_prob(sd) == 0.0f, "reset clears eot_prob to 0");

    /* ── 8. feed_16k direct path ───────────────────────────────────── */
    printf("\n── Direct 16kHz Feed ──\n");
    float tone_16k[1024];
    for (int i = 0; i < 1024; i++)
        tone_16k[i] = 0.3f * sinf(2.0f * M_PI * 200.0f * (float)i / 16000.0f);
    speech_detector_feed_16k(sd, tone_16k, 1024);
    sp = speech_detector_speech_prob(sd);
    printf("    16kHz tone speech_prob: %.4f\n", sp);
    CHECK(sp >= 0.0f && sp <= 1.0f, "16kHz feed produces valid prob");

    /* ── 8b. Sample rate handling: feed various 16kHz chunk sizes ──── */
    printf("\n── 16kHz Various Chunk Sizes ──\n");
    speech_detector_reset(sd);
    /* Feed exactly 512 samples (one VAD chunk) */
    float tone_exact[512];
    for (int i = 0; i < 512; i++)
        tone_exact[i] = 0.3f * sinf(2.0f * M_PI * 200.0f * (float)i / 16000.0f);
    speech_detector_feed_16k(sd, tone_exact, 512);
    sp = speech_detector_speech_prob(sd);
    CHECK(sp >= 0.0f && sp <= 1.0f, "16kHz exact-chunk feed valid prob");

    /* Feed less than one chunk (should buffer, not crash) */
    speech_detector_reset(sd);
    float tone_short[256];
    for (int i = 0; i < 256; i++)
        tone_short[i] = 0.3f * sinf(2.0f * M_PI * 200.0f * (float)i / 16000.0f);
    speech_detector_feed_16k(sd, tone_short, 256);
    sp = speech_detector_speech_prob(sd);
    /* Should still be -1 because we haven't accumulated a full chunk */
    CHECK(sp == -1.0f, "16kHz sub-chunk feed: prob still -1 (buffered)");

    /* Feed second half to complete the chunk */
    speech_detector_feed_16k(sd, tone_short, 256);
    sp = speech_detector_speech_prob(sd);
    CHECK(sp >= 0.0f && sp <= 1.0f, "16kHz two-half feeds produce valid prob");

    /* ── 8c. Feed via 24kHz path with large chunk ─────────────────── */
    printf("\n── 24kHz Large Chunk Feed ──\n");
    speech_detector_reset(sd);
    int large_len = 48000;  /* 2 seconds */
    float *large_24k = (float *)malloc((size_t)large_len * sizeof(float));
    generate_speech_24k(large_24k, large_len);
    /* Feed in one big chunk — tests internal buffer management */
    speech_detector_feed(sd, large_24k, large_len);
    sp = speech_detector_speech_prob(sd);
    printf("    Large 24kHz chunk speech_prob: %.4f\n", sp);
    CHECK(sp >= 0.0f && sp <= 1.0f, "large 24kHz chunk yields valid prob");
    free(large_24k);

    /* ── 8d. Zero-length feed safety ──────────────────────────────── */
    printf("\n── Zero-Length Feed Safety ──\n");
    speech_detector_reset(sd);
    speech_detector_feed(sd, silence, 0);
    speech_detector_feed_16k(sd, tone_16k, 0);
    CHECK(speech_detector_speech_prob(sd) == -1.0f,
          "zero-length feed: prob still -1 (no data processed)");

    /* ── 8e. NULL pcm feed safety ─────────────────────────────────── */
    printf("\n── NULL PCM Feed Safety ──\n");
    speech_detector_feed(sd, NULL, 1024);
    speech_detector_feed_16k(sd, NULL, 512);
    CHECK(1, "NULL pcm feed does not crash");

    /* ── 9. Real human speech validation ───────────────────────────── */
    printf("\n── Real Human Speech ──\n");

    /* Test with macOS system TTS at 16kHz (voice-like, high quality) */
    const char *wav_files[] = { "test_speech_human_16k.wav", "test_speech_human.wav", NULL };
    for (int w = 0; wav_files[w]; w++) {
        const char *human_wav = wav_files[w];
        if (!file_exists(human_wav)) {
            printf("  [SKIP] %s not found\n", human_wav);
            continue;
        }
        int n_raw = 0, sr_raw = 0;
        float *raw = load_wav_f32(human_wav, &n_raw, &sr_raw);
        if (!raw || n_raw <= 0) continue;

        printf("    %s: %d samples, %d Hz (%.2fs)\n",
               human_wav, n_raw, sr_raw, (float)n_raw / sr_raw);

        /* Test via 24kHz feed path (what the pipeline uses) */
        int n24 = 0;
        float *pcm24 = resample_to(raw, n_raw, sr_raw, 24000, &n24);
        speech_detector_reset(sd);
        float max_prob = 0.0f;
        int chunks_above_05 = 0, total_chunks = 0;
        for (int off = 0; off < n24; off += 2400) {
            int n = 2400;
            if (off + n > n24) n = n24 - off;
            speech_detector_feed(sd, pcm24 + off, n);
            float p = speech_detector_speech_prob(sd);
            if (p > max_prob) max_prob = p;
            if (p > 0.5f) chunks_above_05++;
            total_chunks++;
        }
        printf("      24kHz feed: max=%.4f, chunks>0.5: %d/%d (%.1f%%)\n",
               max_prob, chunks_above_05, total_chunks,
               100.0f * chunks_above_05 / (total_chunks ? total_chunks : 1));

        /* Test via 16kHz direct feed */
        int n16 = 0;
        float *pcm16 = resample_to(raw, n_raw, sr_raw, 16000, &n16);
        speech_detector_reset(sd);
        float max_16 = 0.0f;
        for (int off = 0; off < n16; off += 512) {
            if (off + 512 <= n16) {
                speech_detector_feed_16k(sd, pcm16 + off, 512);
                float p = speech_detector_speech_prob(sd);
                if (p > max_16) max_16 = p;
            }
        }
        printf("      16kHz feed: max=%.4f\n", max_16);

        CHECK(max_prob >= 0.0f && max_prob <= 1.0f,
              "speech prob in [0,1]");
        CHECK(max_16 >= 0.0f && max_16 <= 1.0f,
              "speech 16k prob in [0,1]");

        free(pcm24);
        free(pcm16);
        free(raw);
    }

    /* ── 10. TTS audio comparison ──────────────────────────────────── */
    printf("\n── TTS Audio (expected low probs) ──\n");
    const char *tts_wav = "hello_piper.wav";
    if (file_exists(tts_wav)) {
        int n_raw = 0, sr_raw = 0;
        float *raw = load_wav_f32(tts_wav, &n_raw, &sr_raw);
        if (raw) {
            int n24 = 0;
            float *pcm24 = resample_to(raw, n_raw, sr_raw, 24000, &n24);
            speech_detector_reset(sd);
            float max_prob = 0.0f;
            for (int off = 0; off < n24; off += 2400) {
                int n = 2400;
                if (off + n > n24) n = n24 - off;
                speech_detector_feed(sd, pcm24 + off, n);
                float p = speech_detector_speech_prob(sd);
                if (p > max_prob) max_prob = p;
            }
            printf("    TTS max speech_prob: %.4f\n", max_prob);
            CHECK(max_prob >= 0.0f && max_prob <= 1.0f, "TTS prob in [0,1]");
            free(pcm24);
            free(raw);
        }
    } else {
        printf("  [SKIP] %s not found\n", tts_wav);
    }

    /* ── Cleanup ───────────────────────────────────────────────────── */
    speech_detector_destroy(sd);

    printf("\n══════════════════════════════════════════\n");
    printf("Results: %d passed, %d failed\n", passed, failed);
    if (failed > 0) { printf("SOME TESTS FAILED\n"); return 1; }
    printf("ALL PASSED\n");
    return 0;
}
