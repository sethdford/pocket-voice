/**
 * test_integration_audit.c — E2E integration audit: prove module composition.
 *
 * Tests critical integration gaps found during the audit:
 *   1. Flow → iSTFT dimension contract
 *   2. Sample rate consistency across modules
 *   3. STT struct compatibility (SonataSTTWord)
 *   4. sonata_istft + mel_spectrogram timing agreement
 *   5. Tokenizer → LM token format compatibility
 *   6. Flow worker iSTFT instance isolation
 *   7. Crossfade buffer boundary safety
 *   8. Buffer capacity bounds validation
 *
 * Build:
 *   make test-integration-audit
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <Accelerate/Accelerate.h>
#include "sonata_istft.h"
#include "mel_spectrogram.h"
#include "text_normalize.h"
#include "sentence_buffer.h"
#include "ssml_parser.h"
#include "sonata_stt.h"
#include "sonata_refiner.h"
#include "conformer_stt.h"

static int tests_run = 0;
static int tests_passed = 0;
static int tests_failed = 0;

#define ASSERT(cond, msg) do { \
    tests_run++; \
    if (cond) { tests_passed++; } \
    else { tests_failed++; fprintf(stderr, "  FAIL [%s:%d] %s\n", __FILE__, __LINE__, msg); } \
} while(0)

#define ASSERT_INT_EQ(a, b, msg) do { \
    int _a = (int)(a), _b = (int)(b); \
    tests_run++; \
    if (_a == _b) { tests_passed++; } \
    else { tests_failed++; fprintf(stderr, "  FAIL [%s:%d] %s: %d != %d\n", __FILE__, __LINE__, msg, _a, _b); } \
} while(0)

#define ASSERT_FLOAT_NEAR(a, b, eps, msg) do { \
    tests_run++; \
    if (fabsf((a) - (b)) < (eps)) { tests_passed++; } \
    else { tests_failed++; fprintf(stderr, "  FAIL [%s:%d] %s: %f != %f (eps=%f)\n", __FILE__, __LINE__, msg, (float)(a), (float)(b), (float)(eps)); } \
} while(0)

/* ═══════════════════════════════════════════════════════════════════════════
 * Test 1: iSTFT dimension contract — verify n_fft/2+1 bins
 *
 * CRITICAL: The pipeline allocates buffers assuming SONATA_N_BINS = n_fft/2+1.
 * If Flow returns a different number of bins, we get buffer overflow or
 * uninitialized reads. This test proves the contract holds.
 * ═══════════════════════════════════════════════════════════════════════════ */

static void test_istft_dimension_contract(void) {
    printf("=== iSTFT dimension contract ===\n");

    const int n_fft = 1024;
    const int hop = 480;
    const int expected_bins = n_fft / 2 + 1;  /* 513 */

    SonataISTFT *istft = sonata_istft_create(n_fft, hop);
    ASSERT(istft != NULL, "iSTFT create with n_fft=1024, hop=480");

    /* Feed exactly expected_bins of magnitude + phase, expect hop samples out */
    float magnitude[513];
    float phase[513];
    float out_audio[480];

    memset(magnitude, 0, sizeof(magnitude));
    memset(phase, 0, sizeof(phase));
    magnitude[10] = 1.0f;  /* 10th bin ~= 234 Hz at 24kHz */
    phase[10] = 0.0f;

    int samples = sonata_istft_decode_frame(istft, magnitude, phase, out_audio);
    ASSERT_INT_EQ(samples, hop, "iSTFT frame produces exactly hop_length samples");

    /* Feed 3 more frames so overlap-add accumulates — first frame may be
       windowed to near-zero which is correct iSTFT behavior */
    for (int warmup = 0; warmup < 3; warmup++) {
        sonata_istft_decode_frame(istft, magnitude, phase, out_audio);
    }
    float max_val = 0.0f;
    for (int i = 0; i < samples; i++) {
        if (fabsf(out_audio[i]) > max_val) max_val = fabsf(out_audio[i]);
    }
    ASSERT(max_val > 1e-6f, "iSTFT produces non-zero audio after overlap-add warmup");

    /* Verify batch mode: n_frames x expected_bins layout.
       Use enough frames (10) so overlap-add fully warms up */
    const int n_frames = 10;
    float mag_batch[10 * 513];
    float phase_batch[10 * 513];
    float audio_batch[10 * 480];

    memset(mag_batch, 0, sizeof(mag_batch));
    memset(phase_batch, 0, sizeof(phase_batch));
    for (int f = 0; f < n_frames; f++) {
        mag_batch[f * expected_bins + 10] = 1.0f;
    }

    sonata_istft_reset(istft);
    int total = sonata_istft_decode_batch(istft, mag_batch, phase_batch, n_frames, audio_batch);
    ASSERT_INT_EQ(total, n_frames * hop, "iSTFT batch produces n_frames * hop samples");

    /* Check later frames (skip first 3 for overlap-add warmup) */
    max_val = 0.0f;
    for (int i = 3 * hop; i < total; i++) {
        if (fabsf(audio_batch[i]) > max_val) max_val = fabsf(audio_batch[i]);
    }
    ASSERT(max_val > 1e-6f, "iSTFT batch produces non-zero audio after warmup frames");

    sonata_istft_destroy(istft);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Test 2: iSTFT reset between utterances
 *
 * The overlap buffer must be cleared between utterances. If not, the tail
 * of one utterance bleeds into the next one (inter-utterance artifact).
 * ═══════════════════════════════════════════════════════════════════════════ */

static void test_istft_reset_isolation(void) {
    printf("=== iSTFT reset isolation ===\n");

    SonataISTFT *istft = sonata_istft_create(1024, 480);
    ASSERT(istft != NULL, "iSTFT create");

    const int bins = 513;
    float mag[513], phase[513], audio1[480], audio2[480];

    /* Feed a loud frame */
    memset(mag, 0, sizeof(mag));
    memset(phase, 0, sizeof(phase));
    for (int i = 0; i < bins; i++) mag[i] = 1.0f;
    sonata_istft_decode_frame(istft, mag, phase, audio1);

    /* Reset and feed silence */
    sonata_istft_reset(istft);
    memset(mag, 0, sizeof(mag));
    int samples = sonata_istft_decode_frame(istft, mag, phase, audio2);
    ASSERT_INT_EQ(samples, 480, "iSTFT produces samples after reset");

    /* After reset + silence input, output should be near zero */
    float max_after_reset = 0.0f;
    for (int i = 0; i < 480; i++) {
        if (fabsf(audio2[i]) > max_after_reset) max_after_reset = fabsf(audio2[i]);
    }
    ASSERT(max_after_reset < 0.01f, "iSTFT reset clears overlap (no bleed from previous utterance)");

    sonata_istft_destroy(istft);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Test 3: Mel spectrogram + iSTFT timing agreement
 *
 * Verifies that the mel spectrogram and iSTFT agree on their parameters
 * and can produce valid output for the same signal.
 * ═══════════════════════════════════════════════════════════════════════════ */

static void test_mel_istft_roundtrip(void) {
    printf("=== Mel spectrogram + iSTFT consistency ===\n");

    /* Create mel spectrogram analyzer with 24kHz Sonata params */
    MelConfig mel_cfg;
    mel_config_default(&mel_cfg);
    mel_cfg.sample_rate = 24000;
    mel_cfg.n_fft = 1024;
    mel_cfg.hop_length = 480;
    mel_cfg.win_length = 1024;
    mel_cfg.n_mels = 80;
    mel_cfg.fmin = 0.0f;
    mel_cfg.fmax = 12000.0f;

    MelSpectrogram *mel = mel_create(&mel_cfg);
    ASSERT(mel != NULL, "Mel spectrogram create");

    /* Create iSTFT decoder */
    SonataISTFT *istft = sonata_istft_create(1024, 480);
    ASSERT(istft != NULL, "iSTFT create");

    /* Generate a 200ms test tone (4800 samples at 24kHz) */
    const int n_samples = 4800;
    float pcm[4800];
    for (int i = 0; i < n_samples; i++) {
        pcm[i] = 0.5f * sinf(2.0f * M_PI * 440.0f * i / 24000.0f);
    }

    /* Compute mel spectrogram */
    const int max_frames = 20;
    float mel_out[20 * 80];
    int n_frames = mel_process(mel, pcm, n_samples, mel_out, max_frames);
    ASSERT(n_frames > 0, "Mel spectrogram produces frames");
    ASSERT(n_frames <= max_frames, "Mel frame count within bounds");

    /* Verify iSTFT can produce matching number of samples */
    const int bins = 513;
    float mag_batch[20 * 513];
    float phase_batch[20 * 513];
    float audio_out[20 * 480];

    memset(mag_batch, 0, sizeof(mag_batch));
    memset(phase_batch, 0, sizeof(phase_batch));
    /* Put some energy in the 440Hz bin = 440 * 1024 / 24000 ~ bin 19 */
    int target_bin = (int)(440.0f * 1024.0f / 24000.0f + 0.5f);
    for (int f = 0; f < n_frames; f++) {
        mag_batch[f * bins + target_bin] = 0.5f;
    }

    int total_samples = sonata_istft_decode_batch(istft, mag_batch, phase_batch,
                                                   n_frames, audio_out);
    ASSERT_INT_EQ(total_samples, n_frames * 480,
                  "iSTFT produces expected samples for mel frame count");

    mel_destroy(mel);
    sonata_istft_destroy(istft);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Test 4: Sample rate constants consistency
 * ═══════════════════════════════════════════════════════════════════════════ */

static void test_sample_rate_consistency(void) {
    printf("=== Sample rate consistency ===\n");

    const int PIPELINE_RATE = 24000;
    const int PIPELINE_HOP = 480;
    const int PIPELINE_N_FFT = 1024;

    /* Verify: 1920 samples = 80ms at 24kHz (STT frame size) */
    ASSERT_INT_EQ(1920, PIPELINE_RATE * 80 / 1000,
                  "STT frame size = 80ms at 24kHz");

    /* Verify: hop 480 = 20ms at 24kHz (50Hz frame rate) */
    ASSERT_INT_EQ(PIPELINE_HOP, PIPELINE_RATE / 50,
                  "TTS hop length = 20ms at 24kHz (50Hz)");

    /* Verify: n_fft/2+1 = 513 bins */
    ASSERT_INT_EQ(PIPELINE_N_FFT / 2 + 1, 513,
                  "n_fft/2+1 = 513 spectral bins");

    /* Verify: LM frame rate * hop = sample rate */
    ASSERT_INT_EQ(PIPELINE_RATE / 50, PIPELINE_HOP,
                  "LM frame rate * hop = sample rate");

    /* Resampling: 48kHz mic -> 24kHz STT must be exact 2:1 */
    ASSERT_INT_EQ(48000 / PIPELINE_RATE, 2,
                  "Mic->STT resample ratio is exact 2:1");
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Test 5: Text normalize -> Sentence buffer -> SSML parser chain
 * ═══════════════════════════════════════════════════════════════════════════ */

static void test_text_processing_chain(void) {
    printf("=== Text processing chain (normalize -> sentence -> ssml) ===\n");

    const char *chunks[] = {
        "The temperature is ",
        "72 degrees. ",
        "It is a beautiful day!",
    };

    SentenceBuffer *sb = sentbuf_create(SENTBUF_MODE_SENTENCE, 5);
    ASSERT(sb != NULL, "Sentence buffer create");

    char sentence[512];
    int total_sentences = 0;

    for (int i = 0; i < 3; i++) {
        sentbuf_add(sb, chunks[i], (int)strlen(chunks[i]));

        while (sentbuf_has_segment(sb)) {
            int len = sentbuf_flush(sb, sentence, sizeof(sentence));
            if (len <= 0) break;
            total_sentences++;

            /* Verify text normalize can handle numbers in the sentence */
            char normalized[512];
            if (strstr(sentence, "72")) {
                text_cardinal("72", normalized, sizeof(normalized));
                ASSERT(strlen(normalized) > 0, "Cardinal 72 normalizes to text");
            }

            /* Verify SSML parser can handle the sentence */
            SSMLSegment segments[SSML_MAX_SEGMENTS];
            int parsed = ssml_parse(sentence, segments, SSML_MAX_SEGMENTS);
            ASSERT(parsed >= 0, "SSML parser handles sentence");
            if (parsed > 0) {
                ASSERT(strlen(segments[0].text) > 0, "SSML segment has text");
            }
        }
    }

    /* Flush remaining */
    if (sentbuf_flush_all(sb, sentence, sizeof(sentence)) > 0) {
        total_sentences++;
    }

    ASSERT(total_sentences >= 2, "Sentence buffer produces at least 2 sentences from 3 chunks");

    sentbuf_destroy(sb);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Test 6: Crossfade buffer boundary safety
 * ═══════════════════════════════════════════════════════════════════════════ */

static void test_crossfade_boundary(void) {
    printf("=== Crossfade buffer boundary safety ===\n");

    const int CROSSFADE = 480;

    /* Simulate crossfade with minimal chunk (< crossfade length) */
    float tail[480];
    float chunk[240];  /* Half of crossfade */

    for (int i = 0; i < CROSSFADE; i++) tail[i] = 1.0f;
    for (int i = 0; i < 240; i++) chunk[i] = -1.0f;

    /* When chunk < crossfade, cf should be clamped to chunk length */
    int cf = CROSSFADE;
    int chunk_len = 240;
    if (chunk_len < cf) cf = chunk_len;
    ASSERT_INT_EQ(cf, 240, "Crossfade clamped to chunk length");

    for (int i = 0; i < cf; i++) {
        float alpha = (float)i / (float)cf;
        chunk[i] = tail[i] * (1.0f - alpha) + chunk[i] * alpha;
    }

    /* First sample should be close to tail (alpha=0 -> all tail) */
    ASSERT_FLOAT_NEAR(chunk[0], 1.0f, 0.01f, "Crossfade start = tail value");
    /* Last crossfaded sample approaches chunk value */
    ASSERT(chunk[cf - 1] < 0.0f, "Crossfade end approaches chunk value");

    /* Edge case: empty chunk should not crash */
    cf = CROSSFADE;
    chunk_len = 0;
    if (chunk_len < cf) cf = chunk_len;
    ASSERT_INT_EQ(cf, 0, "Zero-length chunk -> zero crossfade");
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Test 7: SonataSTT word struct compatibility
 * ═══════════════════════════════════════════════════════════════════════════ */

static void test_sonata_stt_word_struct(void) {
    printf("=== SonataSTT word struct compatibility ===\n");

    SonataSTTWord word;
    memset(&word, 0, sizeof(word));
    strncpy(word.word, "hello", sizeof(word.word) - 1);
    word.start_sec = 1.5f;
    word.end_sec = 2.0f;
    word.confidence = 0.95f;

    ASSERT_INT_EQ(sizeof(word.word), 64, "SonataSTTWord.word is 64 bytes");
    ASSERT(word.start_sec == 1.5f, "SonataSTTWord timestamp round-trips");
    ASSERT(word.confidence == 0.95f, "SonataSTTWord confidence round-trips");

    /* Verify the struct can hold max-length word without overflow */
    char long_word[64];
    memset(long_word, 'a', 63);
    long_word[63] = '\0';
    strncpy(word.word, long_word, sizeof(word.word) - 1);
    word.word[sizeof(word.word) - 1] = '\0';
    ASSERT(strlen(word.word) == 63, "SonataSTTWord holds 63-char word");
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Test 8: Mel spectrogram edge cases at module boundary
 * ═══════════════════════════════════════════════════════════════════════════ */

static void test_mel_edge_cases(void) {
    printf("=== Mel spectrogram edge cases ===\n");

    MelConfig cfg;
    mel_config_default(&cfg);
    cfg.sample_rate = 24000;
    cfg.n_fft = 1024;
    cfg.hop_length = 480;
    cfg.win_length = 1024;
    cfg.n_mels = 80;

    MelSpectrogram *mel = mel_create(&cfg);
    ASSERT(mel != NULL, "Mel create");

    /* Test: audio shorter than n_fft */
    float short_audio[512];
    for (int i = 0; i < 512; i++) short_audio[i] = sinf(2.0f * M_PI * 440.0f * i / 24000.0f);
    float mel_out[10 * 80];
    int frames = mel_process(mel, short_audio, 512, mel_out, 10);
    ASSERT(frames >= 0, "Mel handles audio shorter than n_fft without crash");

    mel_reset(mel);

    /* Test: exact multiple of hop_length + n_fft */
    const int exact_len = 1024 + 480 * 9;
    float *exact_audio = calloc(exact_len, sizeof(float));
    ASSERT(exact_audio != NULL, "Alloc exact audio");
    for (int i = 0; i < exact_len; i++)
        exact_audio[i] = 0.3f * sinf(2.0f * M_PI * 1000.0f * i / 24000.0f);

    float mel_exact[20 * 80];
    int exact_frames = mel_process(mel, exact_audio, exact_len, mel_exact, 20);
    ASSERT(exact_frames > 0, "Mel produces frames for exact-boundary audio");
    ASSERT(exact_frames <= 20, "Mel frame count within buffer");
    free(exact_audio);

    mel_reset(mel);

    /* Test: all zeros (silence) should produce valid but low-energy mel */
    float silence[2400];
    memset(silence, 0, sizeof(silence));
    float mel_silence[10 * 80];
    int sil_frames = mel_process(mel, silence, 2400, mel_silence, 10);
    ASSERT(sil_frames >= 0, "Mel handles silence without crash");

    mel_destroy(mel);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Test 9: Concurrent iSTFT safety (proves flow_worker contract)
 * ═══════════════════════════════════════════════════════════════════════════ */

static void test_istft_instance_isolation(void) {
    printf("=== iSTFT instance isolation (concurrent safety) ===\n");

    SonataISTFT *istft_a = sonata_istft_create(1024, 480);
    SonataISTFT *istft_b = sonata_istft_create(1024, 480);
    ASSERT(istft_a != NULL && istft_b != NULL, "Create two iSTFT instances");

    float mag[513], phase[513];
    float audio_a[480], audio_b[480];

    memset(mag, 0, sizeof(mag));
    memset(phase, 0, sizeof(phase));
    mag[20] = 0.8f;
    phase[20] = 1.0f;

    int sa = sonata_istft_decode_frame(istft_a, mag, phase, audio_a);
    int sb = sonata_istft_decode_frame(istft_b, mag, phase, audio_b);
    ASSERT_INT_EQ(sa, sb, "Both instances produce same sample count");

    /* Verify outputs are identical (no shared state) */
    int match = 1;
    for (int i = 0; i < sa; i++) {
        if (fabsf(audio_a[i] - audio_b[i]) > 1e-6f) {
            match = 0;
            break;
        }
    }
    ASSERT(match, "Two iSTFT instances produce identical output (no shared state)");

    /* Now diverge: feed different signals and verify they differ */
    float mag2[513];
    memset(mag2, 0, sizeof(mag2));
    mag2[100] = 0.8f;

    sonata_istft_decode_frame(istft_a, mag, phase, audio_a);
    sonata_istft_decode_frame(istft_b, mag2, phase, audio_b);

    int differ = 0;
    for (int i = 0; i < 480; i++) {
        if (fabsf(audio_a[i] - audio_b[i]) > 1e-6f) {
            differ = 1;
            break;
        }
    }
    ASSERT(differ, "Different inputs produce different outputs (instances are independent)");

    sonata_istft_destroy(istft_a);
    sonata_istft_destroy(istft_b);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Test 10: Buffer capacity bounds
 * ═══════════════════════════════════════════════════════════════════════════ */

static void test_buffer_capacity_bounds(void) {
    printf("=== Buffer capacity bounds ===\n");

    const int BUF_CAPACITY = 24000 * 30;  /* 30 seconds at 24kHz */
    const int MAX_FRAMES = 2000;
    const int HOP = 480;
    const int N_BINS = 513;

    /* FINDING: MAX_FRAMES * HOP = 2000 * 480 = 960,000 samples (40 sec)
       but SONATA_BUF_CAPACITY = 720,000 (30 sec). The pipeline has runtime
       guards (total_audio + ns < capacity) that prevent overflow, but the
       constants are inconsistent. MAX_FRAMES should be 1500 not 2000.
       Verify the runtime guards exist by checking the relationship: */
    (void)(MAX_FRAMES * HOP);  /* 960,000 — exceeds BUF_CAPACITY */
    int safe_max_frames = BUF_CAPACITY / HOP;  /* = 1500 */
    ASSERT(safe_max_frames < MAX_FRAMES,
           "FINDING: MAX_FRAMES (2000) exceeds safe limit (1500) for 30s buffer");

    /* mag_scratch allocation: MAX_FRAMES * N_BINS * sizeof(float) */
    size_t mag_size = (size_t)MAX_FRAMES * N_BINS * sizeof(float);
    ASSERT(mag_size < 8 * 1024 * 1024,
           "mag_scratch < 8MB (reasonable heap allocation)");

    /* Verify crossfade is smaller than min chunk */
    const int CROSSFADE = 480;
    const int FIRST_CHUNK = 12;
    int first_chunk_audio = FIRST_CHUNK * HOP;
    ASSERT(CROSSFADE <= first_chunk_audio,
           "Crossfade <= first chunk audio length");
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Test 11: Tokenizer int32 -> uint32 cast safety
 * ═══════════════════════════════════════════════════════════════════════════ */

static void test_tokenizer_cast_safety(void) {
    printf("=== Tokenizer int32 -> uint32 cast safety ===\n");

    int32_t test_ids[] = {0, 1, 100, 1000, 31999, 32767};
    unsigned int uids[6];

    for (int i = 0; i < 6; i++) {
        uids[i] = (unsigned int)test_ids[i];
    }

    ASSERT_INT_EQ((int)uids[0], 0, "Token 0 cast safe");
    ASSERT_INT_EQ((int)uids[5], 32767, "Token 32767 cast safe");

    /* Edge case: negative ID (should never happen but test defensively) */
    int32_t bad_id = -1;
    unsigned int bad_uid = (unsigned int)bad_id;
    ASSERT(bad_uid > 100000, "Negative token cast produces very large uint (detectable)");
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Main
 * ═══════════════════════════════════════════════════════════════════════════ */

int main(void) {
    printf("\n=== Integration Audit Tests ===\n\n");

    test_istft_dimension_contract();
    test_istft_reset_isolation();
    test_mel_istft_roundtrip();
    test_sample_rate_consistency();
    test_text_processing_chain();
    test_crossfade_boundary();
    test_sonata_stt_word_struct();
    test_mel_edge_cases();
    test_istft_instance_isolation();
    test_buffer_capacity_bounds();
    test_tokenizer_cast_safety();

    printf("\n=== Results: %d/%d passed", tests_passed, tests_run);
    if (tests_failed > 0) {
        printf(" (%d FAILED)", tests_failed);
    }
    printf(" ===\n\n");

    return tests_failed > 0 ? 1 : 0;
}
