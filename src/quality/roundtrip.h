/**
 * roundtrip.h — Round-trip intelligibility testing.
 *
 * The ultimate proof of TTS+STT quality: synthesize text via TTS,
 * then transcribe the audio back via STT, and compare the output to
 * the original text. This tests the full pipeline end-to-end.
 *
 * Workflow:
 *   text → TTS → audio → STT → transcript → WER(text, transcript)
 *
 * Also supports:
 *   - A/B comparison: two TTS systems on the same text
 *   - Noise robustness: add noise to TTS output before STT
 *   - Domain testing: specialized vocabulary (medical, technical, etc.)
 *
 * Golden signals for round-trip:
 *   - Clean WER < 5%     → human-level intelligibility
 *   - Noisy WER < 15%    → robust in adverse conditions
 *   - Domain WER < 8%    → handles specialized vocabulary
 */

#ifndef ROUNDTRIP_H
#define ROUNDTRIP_H

#include "wer.h"
#include "audio_quality.h"
#include "latency_harness.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    const char *text;          /* Original text */
    const char *transcript;    /* STT output from synthesized audio */
    WERResult wer;             /* Word Error Rate */
    float cer;                 /* Character Error Rate */
    LatencyMetrics latency;    /* Generation latency */
    MCDResult mcd;             /* MCD if reference audio available */
    int passed;                /* 1 if WER < threshold */
} RoundTripResult;

typedef struct {
    RoundTripResult *results;
    int n_tests;
    int n_passed;
    float mean_wer;
    float mean_cer;
    float worst_wer;
    const char *worst_case_text;
} RoundTripSuite;

/**
 * Callback: synthesize text to audio via TTS.
 * Returns audio samples (caller must free) and sets *out_len.
 * Returns NULL on failure.
 */
typedef float *(*tts_synthesize_fn)(const char *text, int *out_len, int *out_sr,
                                     void *user_data);

/**
 * Callback: transcribe audio via STT.
 * Returns transcript string (caller must free).
 * Returns NULL on failure.
 */
typedef char *(*stt_transcribe_fn)(const float *audio, int n_samples, int sr,
                                    void *user_data);

/**
 * Run a single round-trip test: text → TTS → STT → WER.
 *
 * @param text        Input text
 * @param tts_fn      TTS callback
 * @param stt_fn      STT callback
 * @param user_data   Opaque pointer passed to callbacks
 * @param max_wer     Pass threshold
 * @return RoundTripResult
 */
RoundTripResult roundtrip_test(const char *text,
                                tts_synthesize_fn tts_fn,
                                stt_transcribe_fn stt_fn,
                                void *user_data,
                                float max_wer);

/**
 * Run the full round-trip test suite with built-in test sentences.
 * Covers: simple, numbers, punctuation, long text, edge cases.
 *
 * @param tts_fn     TTS callback
 * @param stt_fn     STT callback
 * @param user_data  Opaque pointer
 * @return RoundTripSuite (caller must call roundtrip_suite_free)
 */
RoundTripSuite roundtrip_run_suite(tts_synthesize_fn tts_fn,
                                    stt_transcribe_fn stt_fn,
                                    void *user_data);

void roundtrip_suite_free(RoundTripSuite *suite);

void roundtrip_print_report(const RoundTripSuite *suite);

/**
 * Generate a golden test set: text + TTS audio + expected STT output.
 * Saves to directory as {name}.wav and {name}.txt pairs.
 *
 * @param output_dir  Directory to write golden files
 * @param tts_fn      TTS callback
 * @param user_data   Opaque pointer
 * @return 0 on success
 */
int roundtrip_generate_golden(const char *output_dir,
                               tts_synthesize_fn tts_fn,
                               void *user_data);

#ifdef __cplusplus
}
#endif

#endif /* ROUNDTRIP_H */
