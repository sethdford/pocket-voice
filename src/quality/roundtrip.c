/**
 * roundtrip.c — Round-trip intelligibility testing.
 *
 * Synthesizes text via TTS, transcribes via STT, and compares.
 * The ultimate proof that the pipeline produces intelligible speech.
 */

#include "roundtrip.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/time.h>

/* ── Built-in Test Sentences ──────────────────────────── */

static const char *test_sentences[] = {
    /* Simple */
    "Hello, how are you doing today?",
    "The weather is nice outside.",
    "I would like to order a coffee, please.",

    /* Numbers and currency */
    "The total is forty two dollars and fifty cents.",
    "There are three hundred and sixty five days in a year.",
    "Call me at five five five, one two three four.",

    /* Punctuation and prosody */
    "Wait, really? You can't be serious!",
    "First, open the door. Second, turn on the light. Third, sit down.",
    "It's not what you said; it's how you said it.",

    /* Long sentences */
    "Artificial intelligence is rapidly transforming every aspect of modern "
    "life, from healthcare and education to transportation and entertainment.",

    /* Technical vocabulary */
    "The patient presented with bilateral pneumothorax requiring immediate "
    "chest tube insertion.",

    /* Homophones and tricky words */
    "Their car is over there, and they're coming to get it.",
    "I read the book you read last week.",

    /* Abbreviations */
    "Dr. Smith from NASA visited the UN headquarters on Jan. 15th.",

    /* Edge cases */
    "Yes.",
    "A.",
    "One two three four five six seven eight nine ten.",
};

static const int N_TEST_SENTENCES = sizeof(test_sentences) / sizeof(test_sentences[0]);

/* ── Timing ───────────────────────────────────────────── */

static double now_ms(void)
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)tv.tv_sec * 1000.0 + (double)tv.tv_usec / 1000.0;
}

/* ── Single Round-Trip Test ───────────────────────────── */

RoundTripResult roundtrip_test(const char *text,
                                tts_synthesize_fn tts_fn,
                                stt_transcribe_fn stt_fn,
                                void *user_data,
                                float max_wer)
{
    RoundTripResult r = {0};
    r.text = text;

    if (!tts_fn || !stt_fn || !text) {
        r.wer.wer = 1.0f;
        return r;
    }

    /* Step 1: TTS — synthesize text to audio */
    int audio_len = 0, audio_sr = 0;
    double t0 = now_ms();
    float *audio = tts_fn(text, &audio_len, &audio_sr, user_data);
    double t1 = now_ms();

    if (!audio || audio_len == 0) {
        fprintf(stderr, "  [FAIL] TTS returned no audio for: %.50s...\n", text);
        r.wer.wer = 1.0f;
        return r;
    }

    r.latency.first_chunk_ms = (float)(t1 - t0);
    float audio_duration_ms = (float)audio_len / (float)audio_sr * 1000.0f;
    r.latency.rtf = (float)(t1 - t0) / audio_duration_ms;

    /* Step 2: STT — transcribe audio back to text */
    double t2 = now_ms();
    char *transcript = stt_fn(audio, audio_len, audio_sr, user_data);
    double t3 = now_ms();
    r.latency.ttft_ms = (float)(t3 - t2); /* STT latency */

    free(audio);

    if (!transcript) {
        fprintf(stderr, "  [FAIL] STT returned no transcript for: %.50s...\n", text);
        r.wer.wer = 1.0f;
        return r;
    }

    r.transcript = transcript;
    r.latency.e2e_ms = (float)(t3 - t0);

    /* Step 3: Compare — compute WER */
    const char *expected = text;
    r.wer = wer_compute(expected, transcript);
    r.cer = r.wer.cer;
    r.passed = r.wer.wer <= max_wer;

    return r;
}

/* ── Full Test Suite ──────────────────────────────────── */

RoundTripSuite roundtrip_run_suite(tts_synthesize_fn tts_fn,
                                    stt_transcribe_fn stt_fn,
                                    void *user_data)
{
    RoundTripSuite suite = {0};
    suite.n_tests = N_TEST_SENTENCES;
    suite.results = (RoundTripResult *)calloc((size_t)N_TEST_SENTENCES, sizeof(RoundTripResult));

    float sum_wer = 0, sum_cer = 0;
    float worst_wer = 0;
    int passed = 0;

    for (int i = 0; i < N_TEST_SENTENCES; i++) {
        printf("  [%2d/%d] %.60s%s\n", i + 1, N_TEST_SENTENCES,
               test_sentences[i],
               strlen(test_sentences[i]) > 60 ? "..." : "");

        suite.results[i] = roundtrip_test(test_sentences[i], tts_fn, stt_fn,
                                           user_data, 0.10f);

        RoundTripResult *r = &suite.results[i];
        sum_wer += r->wer.wer;
        sum_cer += r->cer;
        if (r->passed) passed++;

        if (r->wer.wer > worst_wer) {
            worst_wer = r->wer.wer;
            suite.worst_case_text = test_sentences[i];
        }

        printf("         WER: %5.1f%%  CER: %5.1f%%  %s\n",
               r->wer.wer * 100.0f, r->cer * 100.0f,
               r->passed ? "PASS" : "FAIL");

        if (r->transcript) {
            /* Only print transcript if it differs */
            char ref_norm[1024], hyp_norm[1024];
            wer_normalize(test_sentences[i], ref_norm, 1024);
            wer_normalize(r->transcript, hyp_norm, 1024);
            if (strcmp(ref_norm, hyp_norm) != 0) {
                printf("         REF: %.70s\n", ref_norm);
                printf("         HYP: %.70s\n", hyp_norm);
            }
        }
    }

    suite.n_passed = passed;
    suite.mean_wer = sum_wer / (float)N_TEST_SENTENCES;
    suite.mean_cer = sum_cer / (float)N_TEST_SENTENCES;
    suite.worst_wer = worst_wer;

    return suite;
}

void roundtrip_suite_free(RoundTripSuite *suite)
{
    if (!suite) return;
    for (int i = 0; i < suite->n_tests; i++) {
        /* transcript was allocated by stt_fn; caller owns it */
        free((void *)suite->results[i].transcript);
    }
    free(suite->results);
    suite->results = NULL;
}

void roundtrip_print_report(const RoundTripSuite *suite)
{
    if (!suite) return;

    fprintf(stderr,
        "\n╔══════════════════════════════════════════════════════════╗\n"
        "║  Round-Trip Intelligibility Report                        ║\n"
        "╠══════════════════════════════════════════════════════════╣\n"
        "║                                                          ║\n"
        "║  Tests Run:        %3d                                   ║\n"
        "║  Tests Passed:     %3d  (%5.1f%%)                        ║\n"
        "║                                                          ║\n"
        "║  Mean WER:         %5.1f%%  %s                          ║\n"
        "║  Mean CER:         %5.1f%%                               ║\n"
        "║  Worst WER:        %5.1f%%                               ║\n",
        suite->n_tests,
        suite->n_passed,
        (float)suite->n_passed / (float)suite->n_tests * 100.0f,
        suite->mean_wer * 100.0f,
        suite->mean_wer < 0.05f ? "(human-level)" :
        suite->mean_wer < 0.10f ? "(good)       " : "(needs work) ",
        suite->mean_cer * 100.0f,
        suite->worst_wer * 100.0f);

    if (suite->worst_case_text) {
        fprintf(stderr,
            "║  Worst Case:       \"%.40s\"\n", suite->worst_case_text);
    }

    fprintf(stderr,
        "║                                                          ║\n"
        "║  Verdict: %s                                       ║\n"
        "╚══════════════════════════════════════════════════════════╝\n\n",
        suite->n_passed == suite->n_tests ? "ALL PASS   " :
        suite->n_passed > suite->n_tests * 8 / 10 ? "MOSTLY OK  " : "NEEDS WORK ");
}

/* ── Golden Test Generation ───────────────────────────── */

static void wav_write_f32(const char *path, const float *samples, int n, int sr)
{
    FILE *f = fopen(path, "wb");
    if (!f) return;

    unsigned int data_size = (unsigned int)(n * 2); /* 16-bit PCM */
    unsigned int file_size = 36 + data_size;
    unsigned short channels = 1;
    unsigned short bits = 16;
    unsigned int byte_rate = (unsigned int)(sr * channels * bits / 8);
    unsigned short block_align = (unsigned short)(channels * bits / 8);
    unsigned int fmt_size = 16;
    unsigned short audio_fmt = 1; /* PCM */

    fwrite("RIFF", 1, 4, f);
    fwrite(&file_size, 4, 1, f);
    fwrite("WAVE", 1, 4, f);
    fwrite("fmt ", 1, 4, f);
    fwrite(&fmt_size, 4, 1, f);
    fwrite(&audio_fmt, 2, 1, f);
    fwrite(&channels, 2, 1, f);
    unsigned int sr_u = (unsigned int)sr;
    fwrite(&sr_u, 4, 1, f);
    fwrite(&byte_rate, 4, 1, f);
    fwrite(&block_align, 2, 1, f);
    fwrite(&bits, 2, 1, f);
    fwrite("data", 1, 4, f);
    fwrite(&data_size, 4, 1, f);

    for (int i = 0; i < n; i++) {
        float s = samples[i];
        if (s > 1.0f) s = 1.0f;
        if (s < -1.0f) s = -1.0f;
        short pcm = (short)(s * 32767.0f);
        fwrite(&pcm, 2, 1, f);
    }

    fclose(f);
}

int roundtrip_generate_golden(const char *output_dir,
                               tts_synthesize_fn tts_fn,
                               void *user_data)
{
    if (!output_dir || !tts_fn) return -1;

    mkdir(output_dir, 0755);

    int generated = 0;
    for (int i = 0; i < N_TEST_SENTENCES; i++) {
        int audio_len = 0, sr = 0;
        float *audio = tts_fn(test_sentences[i], &audio_len, &sr, user_data);
        if (!audio) continue;

        char wav_path[512], txt_path[512];
        snprintf(wav_path, sizeof(wav_path), "%s/golden_%02d.wav", output_dir, i);
        snprintf(txt_path, sizeof(txt_path), "%s/golden_%02d.txt", output_dir, i);

        wav_write_f32(wav_path, audio, audio_len, sr);

        FILE *txt = fopen(txt_path, "w");
        if (txt) {
            fprintf(txt, "%s\n", test_sentences[i]);
            fclose(txt);
        }

        free(audio);
        generated++;
        printf("  Golden %02d: %s\n", i, wav_path);
    }

    printf("\n  Generated %d golden test files in %s\n", generated, output_dir);
    return 0;
}
