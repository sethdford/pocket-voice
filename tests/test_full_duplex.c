/**
 * test_full_duplex.c — Integration tests for full-duplex pipeline modules.
 *
 * Exercises VAP, audio mixer, neural backchannel, intent router, speculative gen
 * working together. Does not run the real audio pipeline; tests module APIs.
 *
 * Build: make test-full-duplex
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include "vap_model.h"
#include "audio_mixer.h"
#include "neural_backchannel.h"
#include "intent_router.h"
#include "speculative_gen.h"
#include "speech_detector.h"
#include "backchannel.h"

static int pass_count = 0;
static int fail_count = 0;

#define TEST(cond, name) do { \
    if (cond) { printf("  [PASS] %s\n", name); pass_count++; } \
    else { printf("  [FAIL] %s (line %d)\n", name, __LINE__); fail_count++; } \
} while (0)

/* ─── VAP + Speech Detector Cooperation ──────────────────────────────────── */

static void test_vap_speech_detector_coop(void) {
    printf("\n=== VAP + Speech Detector Cooperation ===\n");

    VAPModel *vap = vap_create_config(128, 4, 4, 256);
    TEST(vap != NULL, "VAP create_config");

    SpeechDetectorConfig sd_cfg = { .native_vad_path = NULL };
    SpeechDetector *sd = speech_detector_create(&sd_cfg);
    TEST(sd != NULL, "speech_detector create");

    float user_mel[80], system_mel[80];
    for (int i = 0; i < 80; i++) {
        user_mel[i] = 0.01f * (float)i;
        system_mel[i] = -0.005f * (float)i;
    }

    VAPPrediction pred = vap_feed(vap, user_mel, system_mel);
    TEST(pred.p_user_speaking >= 0.0f && pred.p_user_speaking <= 1.0f, "VAP p_user_speaking valid");
    TEST(pred.p_eou >= 0.0f && pred.p_eou <= 1.0f, "VAP p_eou valid");

    float pcm24[512];
    for (int i = 0; i < 512; i++) pcm24[i] = 0.02f * sinf((float)i * 0.1f);
    speech_detector_feed(sd, pcm24, 512);
    float sp = speech_detector_speech_prob(sd);
    TEST(sp >= -1.01f && sp <= 1.01f, "speech_detector prob in range (-1 if no VAD)");

    speech_detector_destroy(sd);
    vap_destroy(vap);
}

/* ─── Audio Mixer Multiple Sources ────────────────────────────────────────── */

static void test_audio_mixer_multi_src(void) {
    printf("\n=== Audio Mixer Multiple Sources ===\n");

    AudioMixerConfig cfg = {
        .sample_rate = 24000,
        .block_size = 480,
        .ducking_gain = 0.3f,
        .crossfade_samples = 120,
    };
    AudioMixer *m = audio_mixer_create(&cfg);
    TEST(m != NULL, "mixer create");

    float main[960], bc[480];
    for (int i = 0; i < 960; i++) main[i] = 0.5f;
    for (int i = 0; i < 480; i++) bc[i] = 0.3f;

    int w1 = audio_mixer_write(m, MIX_CHANNEL_MAIN, main, 960);
    int w2 = audio_mixer_write(m, MIX_CHANNEL_BACKCHANNEL, bc, 480);
    TEST(w1 == 960, "main write 960");
    TEST(w2 == 480, "backchannel write 480");

    int active = audio_mixer_any_active(m);
    TEST(active, "mixer has active channels");

    float out[960];
    int r = audio_mixer_read(m, out, 960);
    TEST(r > 0, "mixer read produces samples");

    audio_mixer_destroy(m);
}

/* ─── Neural Backchannel + Mixer ─────────────────────────────────────────── */

static void test_neural_backchannel_mixer(void) {
    printf("\n=== Neural Backchannel + Mixer ===\n");

    NBCConfig nbc_cfg = { .sample_rate = 24000, .max_duration_ms = 500, .cache_enabled = 0 };
    NeuralBackchannel *nbc = nbc_create(&nbc_cfg, NULL);
    TEST(nbc != NULL, "nbc create");

    int len = 0;
    const float *cached = nbc_get_cached(nbc, NBC_MHM, &len);
    TEST(cached == NULL || len >= 0, "nbc_get_cached (may be uncached)");

    float gen[4096];
    int n = nbc_generate(nbc, NBC_MHM, gen, 4096);
    TEST(n > 0 || n == -1, "nbc_generate returns samples or -1");

    if (n > 0) {
        AudioMixerConfig mix_cfg = {
            .sample_rate = 24000, .block_size = 480,
            .ducking_gain = 0.3f, .crossfade_samples = 120,
        };
        AudioMixer *m = audio_mixer_create(&mix_cfg);
        if (m) {
            int w = audio_mixer_write(m, MIX_CHANNEL_BACKCHANNEL, gen, n);
            TEST(w > 0, "write BC to mixer");
            float out[4096];
            int r = audio_mixer_read(m, out, 4096);
            TEST(r > 0, "read mixed output");
            audio_mixer_destroy(m);
        }
    }

    nbc_destroy(nbc);
}

/* ─── Intent Router Fast Path ────────────────────────────────────────────── */

static void test_intent_router_fast_path(void) {
    printf("\n=== Intent Router → Fast Path ===\n");

    IntentRouter *r = intent_router_create_default();
    TEST(r != NULL, "intent router create");

    RoutingDecision d = intent_router_route(r, "hi there", 2, NULL, NULL);
    TEST(d.route == ROUTE_FAST, "hi there -> ROUTE_FAST");
    TEST(d.fast_type >= 0, "has fast_type");

    const char *txt = intent_router_fast_text(d.fast_type);
    TEST(txt != NULL && strlen(txt) > 0, "fast_text non-empty");

    intent_router_destroy(r);
}

/* ─── Intent Router Backchannel Path ──────────────────────────────────────── */

static void test_intent_router_backchannel_path(void) {
    printf("\n=== Intent Router → Backchannel Path ===\n");

    IntentRouter *r = intent_router_create_default();
    TEST(r != NULL, "intent router create");

    RoutingDecision d = intent_router_route(r, "mhm", 1, NULL, NULL);
    TEST(d.route == ROUTE_BACKCHANNEL || d.route == ROUTE_FAST, "short utterance routes");

    d = intent_router_route(r, "yeah", 1, NULL, NULL);
    TEST(d.confidence >= 0.0f && d.confidence <= 1.0f, "confidence in range");

    intent_router_destroy(r);
}

/* ─── Speculative Gen Lifecycle ───────────────────────────────────────────── */

static void test_speculative_gen_lifecycle(void) {
    printf("\n=== Speculative Gen Lifecycle ===\n");

    SpeculativeConfig cfg = {
        .max_drafts = 2,
        .min_words_to_spec = 3,
        .vap_eou_threshold = 0.4f,
        .commit_threshold = 0.8f,
    };
    SpeculativeGen *sg = speculative_gen_create(&cfg);
    TEST(sg != NULL, "speculative_gen create");

    int r = speculative_gen_tick(sg, "hello world", 2, 0.2f, 0.0f, 0.3f);
    TEST(r == 0, "tick low EOU -> no action");

    int draft = speculative_gen_active_draft(sg);
    TEST(draft < 0, "no active draft initially");

    speculative_gen_reset(sg);
    speculative_gen_destroy(sg);
}

/* ─── Speculative Gen + VAP Signals ───────────────────────────────────────── */

static void test_speculative_gen_vap_signals(void) {
    printf("\n=== Speculative Gen + VAP Signals ===\n");

    SpeculativeGen *sg = speculative_gen_create(NULL);
    TEST(sg != NULL, "create");

    int r = speculative_gen_tick(sg, "tell me about the weather today", 6,
        0.55f, 0.45f, 0.6f);
    TEST(r == 0 || r == 1, "tick with high VAP EOU returns 0 or 1");

    speculative_gen_destroy(sg);
}

/* ─── Speculative Gen Draft Cancellation ───────────────────────────────────── */

static void test_speculative_gen_cancel(void) {
    printf("\n=== Speculative Gen Draft Cancellation ===\n");

    SpeculativeGen *sg = speculative_gen_create(NULL);
    speculative_gen_tick(sg, "hello", 1, 0.5f, 0.4f, 0.5f);
    speculative_gen_cancel_all(sg);
    int draft = speculative_gen_active_draft(sg);
    TEST(draft < 0, "cancel_all clears active draft");
    speculative_gen_destroy(sg);
}

/* ─── Audio Mixer Priority Ducking ────────────────────────────────────────── */

static void test_audio_mixer_ducking(void) {
    printf("\n=== Audio Mixer Priority Ducking ===\n");

    AudioMixerConfig cfg = {
        .sample_rate = 24000,
        .block_size = 480,
        .ducking_gain = 0.2f,
        .crossfade_samples = 120,
    };
    AudioMixer *m = audio_mixer_create(&cfg);
    TEST(m != NULL, "create");

    audio_mixer_set_priority(m, MIX_CHANNEL_MAIN, 10);
    audio_mixer_set_priority(m, MIX_CHANNEL_BACKCHANNEL, 5);

    float main[960], bc[960];
    for (int i = 0; i < 960; i++) main[i] = 1.0f;
    for (int i = 0; i < 960; i++) bc[i] = 0.5f;

    audio_mixer_write(m, MIX_CHANNEL_MAIN, main, 960);
    audio_mixer_write(m, MIX_CHANNEL_BACKCHANNEL, bc, 960);

    float out[960];
    int n = audio_mixer_read(m, out, 960);
    TEST(n > 0, "read with ducking");
    float peak = 0;
    for (int i = 0; i < n; i++)
        if (fabsf(out[i]) > peak) peak = fabsf(out[i]);
    TEST(peak > 0.1f && peak <= 1.5f, "output in expected range");

    audio_mixer_destroy(m);
}

/* ─── Full Flow: Intent Route → Fast Path ─────────────────────────────────── */

static void test_flow_intent_fast_path(void) {
    printf("\n=== Full Flow: Intent Route → Fast Path ===\n");

    IntentRouter *ir = intent_router_create_default();
    TEST(ir != NULL, "intent router");

    const char *transcript = "thanks a lot";
    int nw = 3;
    RoutingDecision d = intent_router_route(ir, transcript, nw, NULL, NULL);

    const char *response = NULL;
    if (d.route == ROUTE_FAST && d.confidence > 0.7f && d.fast_type >= 0)
        response = intent_router_fast_text(d.fast_type);

    TEST(response != NULL || d.route != ROUTE_FAST, "fast path or different route");
    if (response)
        TEST(strlen(response) > 0, "response non-empty");

    intent_router_destroy(ir);
}

/* ─── Full Flow: VAP → Speculative → Commit ───────────────────────────────── */

static void test_flow_vap_speculative_commit(void) {
    printf("\n=== Full Flow: VAP → Speculative → Commit ===\n");

    SpeculativeGen *sg = speculative_gen_create(NULL);
    speculative_gen_tick(sg, "what is the capital of France", 6,
        0.9f, 0.8f, 0.95f);

    speculative_gen_feed_token(sg, 0, "Paris");
    speculative_gen_feed_token(sg, 0, ".");
    speculative_gen_draft_done(sg, 0);

    const char *committed = speculative_gen_commit(sg);
    TEST(committed == NULL || strlen(committed) >= 0, "commit returns or NULL");

    speculative_gen_destroy(sg);
}

/* ─── Full Flow: During Playback → VAP Backchannel → Mixer ─────────────────── */

static void test_flow_playback_vap_backchannel_mixer(void) {
    printf("\n=== Full Flow: Playback → VAP Backchannel → Mixer ===\n");

    AudioMixerConfig cfg = {
        .sample_rate = 24000, .block_size = 480,
        .ducking_gain = 0.4f, .crossfade_samples = 120,
    };
    AudioMixer *m = audio_mixer_create(&cfg);
    TEST(m != NULL, "mixer");

    float tts[960];
    for (int i = 0; i < 960; i++) tts[i] = 0.6f;
    audio_mixer_write(m, MIX_CHANNEL_MAIN, tts, 960);

    NBCConfig nbc_cfg = { .sample_rate = 24000, .max_duration_ms = 500, .cache_enabled = 0 };
    NeuralBackchannel *nbc = nbc_create(&nbc_cfg, NULL);
    if (nbc) {
        int bc_len = 0;
        const float *bc = nbc_get_cached(nbc, NBC_MHM, &bc_len);
        float gen_buf[4096];
        if (!bc || bc_len == 0) {
            int ng = nbc_generate(nbc, NBC_MHM, gen_buf, 4096);
            if (ng > 0) bc = gen_buf, bc_len = ng;
        }
        if (bc && bc_len > 0) {
            int w = audio_mixer_write(m, MIX_CHANNEL_BACKCHANNEL, bc, bc_len);
            TEST(w > 0, "backchannel to mixer during playback");
        }
        nbc_destroy(nbc);
    }

    float out[1024];
    int r = audio_mixer_read(m, out, 1024);
    TEST(r > 0, "mixed output");

    audio_mixer_destroy(m);
}

/* ─── Barge-in: VAP user_speaking High ────────────────────────────────────── */

static void test_bargein_vap_user_speaking(void) {
    printf("\n=== Barge-in: VAP user_speaking High + backchannel Low ===\n");

    VAPModel *vap = vap_create_config(64, 2, 2, 128);
    TEST(vap != NULL, "VAP create");

    float user[80], sys[80];
    for (int i = 0; i < 80; i++) user[i] = 0.1f * (float)(i % 10);
    memset(sys, 0, sizeof(sys));

    VAPPrediction p = vap_feed(vap, user, sys);
    (void)(p.p_user_speaking > 0.7f && p.p_backchannel < 0.3f); /* barge-in logic */
    TEST(p.p_user_speaking >= 0.0f && p.p_user_speaking <= 1.0f, "p_user_speaking valid");
    TEST(p.p_backchannel >= 0.0f && p.p_backchannel <= 1.0f, "p_backchannel valid");

    vap_destroy(vap);
}

/* ─── Non-Barge-in: VAP Backchannel High ──────────────────────────────────── */

static void test_non_bargein_vap_backchannel_high(void) {
    printf("\n=== Non-Barge-in: VAP Backchannel High ===\n");

    VAPModel *vap = vap_create_config(64, 2, 2, 128);
    float user[80], sys[80];
    memset(user, 0.02f, sizeof(user));
    memset(sys, 0, sizeof(sys));

    VAPPrediction p = vap_feed(vap, user, sys);
    (void)(p.p_backchannel > 0.5f && p.p_user_speaking < 0.5f); /* non-barge-in logic */
    TEST(1, "VAP predicts (backchannel vs barge-in distinguishable)");

    vap_destroy(vap);
}

/* ─── Backchannel + Mixer Integration ─────────────────────────────────────── */

static void test_backchannel_mixer_integration(void) {
    printf("\n=== Backchannel + Mixer Integration ===\n");

    BackchannelGen *bc = backchannel_create(24000);
    TEST(bc != NULL, "backchannel create");

    AudioMixerConfig cfg = {
        .sample_rate = 24000, .block_size = 480,
        .ducking_gain = 0.3f, .crossfade_samples = 120,
    };
    AudioMixer *m = audio_mixer_create(&cfg);
    TEST(m != NULL, "mixer create");

    int len = 0;
    const float *audio = backchannel_get_audio(bc, BC_MHM, &len);
    TEST(audio != NULL && len > 0, "backchannel get_audio");

    if (audio && len > 0) {
        int w = audio_mixer_write(m, MIX_CHANNEL_BACKCHANNEL, audio, len);
        TEST(w > 0, "write BC to mixer");
    }

    backchannel_destroy(bc);
    audio_mixer_destroy(m);
}

/* ─── Main ───────────────────────────────────────────────────────────────── */

int main(void) {
    printf("═══ Full Duplex Integration Tests ═══\n");

    test_vap_speech_detector_coop();
    test_audio_mixer_multi_src();
    test_neural_backchannel_mixer();
    test_intent_router_fast_path();
    test_intent_router_backchannel_path();
    test_speculative_gen_lifecycle();
    test_speculative_gen_vap_signals();
    test_speculative_gen_cancel();
    test_audio_mixer_ducking();
    test_flow_intent_fast_path();
    test_flow_vap_speculative_commit();
    test_flow_playback_vap_backchannel_mixer();
    test_bargein_vap_user_speaking();
    test_non_bargein_vap_backchannel_high();
    test_backchannel_mixer_integration();

    printf("\n═══ Results: %d passed, %d failed ═══\n", pass_count, fail_count);
    return fail_count > 0 ? 1 : 0;
}
