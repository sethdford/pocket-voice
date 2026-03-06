/**
 * test_new_engines.c — Tests for phonemizer and speaker_encoder.
 *
 * Tests API contract, NULL handling, and error cases. No model files required.
 *
 * Requires: brew install espeak-ng onnxruntime
 *
 * Build: make test-new-engines
 * Run:   ./build/test-new-engines
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "phonemizer.h"
#include "speaker_encoder.h"

static int passed = 0, failed = 0;

#define CHECK(cond, msg) do { \
    if (cond) { printf("  [PASS] %s\n", msg); passed++; } \
    else { printf("  [FAIL] %s\n", msg); failed++; } \
} while (0)

static int is_ipa_char(char c) {
    /* IPA uses Unicode; ASCII subset: a-z, ɑ, ə, ɛ, ɪ, ɔ, ʊ, ʃ, ʒ, θ, ð, ŋ, etc. */
    if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z')) return 1;
    if (c == ' ' || c == '.' || c == '\'' || c == '-') return 1;
    if ((unsigned char)c >= 0xC0) return 1;  /* UTF-8 continuation / multibyte */
    return 0;
}

static int has_ipa_content(const char *s) {
    if (!s || !*s) return 0;
    for (; *s; s++) {
        if (is_ipa_char(*s)) return 1;
    }
    return 0;
}

int main(void) {
    printf("\n═══ Phonemizer Tests ═══\n");

    /* Test 1: create with valid language */
    Phonemizer *ph = phonemizer_create("en-us");
    CHECK(ph != NULL, "create with valid language");
    if (ph) {
        char ipa[256];
        int len = phonemizer_text_to_ipa(ph, "Hello world", ipa, (int)sizeof(ipa));
        CHECK(len > 0 && has_ipa_content(ipa), "text_to_ipa produces non-empty IPA");
        phonemizer_destroy(ph);
    }

    /* Test 2: create with NULL language returns NULL */
    CHECK(phonemizer_create(NULL) == NULL, "create with NULL language returns NULL");

    /* Test 3: text_to_ipa with NULL phonemizer returns -1 */
    {
        char buf[64];
        CHECK(phonemizer_text_to_ipa(NULL, "hello", buf, 64) == -1,
              "text_to_ipa with NULL phonemizer returns -1");
    }

    /* Test 4: text_to_ipa with NULL text returns -1 */
    {
        ph = phonemizer_create("en-us");
        if (ph) {
            char buf[64];
            CHECK(phonemizer_text_to_ipa(ph, NULL, buf, 64) == -1,
                  "text_to_ipa with NULL text returns -1");
            phonemizer_destroy(ph);
        }
    }

    /* Test 5: text_to_ipa with zero buffer returns -1 */
    {
        ph = phonemizer_create("en-us");
        if (ph) {
            char buf[64];
            CHECK(phonemizer_text_to_ipa(ph, "hello", buf, 0) == -1 &&
                  phonemizer_text_to_ipa(ph, "hello", NULL, 64) == -1,
                  "text_to_ipa with zero buffer returns -1");
            phonemizer_destroy(ph);
        }
    }

    /* Test 6: text_to_ids without phoneme map returns -1 */
    {
        ph = phonemizer_create("en-us");
        if (ph) {
            int ids[32];
            CHECK(phonemizer_text_to_ids(ph, "hello", ids, 32) == -1,
                  "text_to_ids without phoneme map returns -1");
            phonemizer_destroy(ph);
        }
    }

    /* Test 6b: text_to_ids with phoneme map adds BOS/EOS */
    {
        ph = phonemizer_create("en-us");
        if (ph) {
            const char *map_path = "models/sonata/phoneme_map.json";
            if (phonemizer_load_phoneme_map(ph, map_path) == 0) {
                int ids[64];
                int n = phonemizer_text_to_ids(ph, "Hi", ids, 64);
                CHECK(n >= 3, "text_to_ids produces at least BOS + 1 + EOS");
                CHECK(ids[0] == 1, "BOS token (id 1) at start");
                CHECK(ids[n - 1] == 2, "EOS token (id 2) at end");
            }
            phonemizer_destroy(ph);
        }
    }

    /* Test 7: vocab_size returns 0 when no map loaded */
    {
        ph = phonemizer_create("en-us");
        if (ph) {
            CHECK(phonemizer_vocab_size(ph) == 0, "vocab_size returns 0 when no map loaded");
            phonemizer_destroy(ph);
        }
    }

    /* Test 8: destroy NULL is safe */
    phonemizer_destroy(NULL);
    CHECK(1, "destroy NULL is safe");

    printf("\n═══ Speaker Encoder Tests ═══\n");

    /* Test 1: create with NULL path returns NULL */
    CHECK(speaker_encoder_create(NULL) == NULL, "create with NULL path returns NULL");

    /* Test 2: create with nonexistent path returns NULL */
    CHECK(speaker_encoder_create("/nonexistent/speaker_encoder.onnx") == NULL,
          "create with nonexistent path returns NULL");

    /* Test 3: extract with NULL encoder returns -1 */
    {
        float audio[16000] = {0};
        float emb[256] = {0};
        CHECK(speaker_encoder_encode_audio(NULL, audio, 16000, 16000, emb) == -1,
              "extract with NULL encoder returns -1");
    }

    /* Test 4: encode_mel with NULL encoder returns -1 */
    {
        float mel[80 * 100] = {0};
        float emb[256] = {0};
        CHECK(speaker_encoder_encode_mel(NULL, mel, 100, emb) == -1,
              "encode_mel with NULL encoder returns -1");
    }

    /* Test 5: encode_mel with NULL mel returns -1 */
    {
        float emb[256] = {0};
        CHECK(speaker_encoder_encode_mel(NULL, NULL, 100, emb) == -1,
              "encode_mel with NULL encoder and NULL mel returns -1");
    }

    /* Test 6: destroy NULL is safe */
    speaker_encoder_destroy(NULL);
    CHECK(1, "destroy NULL is safe");

    /* ═══ Additional Phonemizer Tests ═══ */
    printf("\n═══ Additional Phonemizer Tests ═══\n");

    /* Test 9: create/destroy lifecycle — repeated create+destroy */
    {
        int lifecycle_ok = 1;
        for (int i = 0; i < 5; i++) {
            Phonemizer *p = phonemizer_create("en-us");
            if (!p) { lifecycle_ok = 0; break; }
            char buf[128];
            int len = phonemizer_text_to_ipa(p, "test", buf, sizeof(buf));
            if (len <= 0) lifecycle_ok = 0;
            phonemizer_destroy(p);
        }
        CHECK(lifecycle_ok, "phonemizer create/destroy lifecycle (5 cycles)");
    }

    /* Test 10: empty string input */
    {
        ph = phonemizer_create("en-us");
        if (ph) {
            char buf[128];
            int len = phonemizer_text_to_ipa(ph, "", buf, sizeof(buf));
            CHECK(len == 0 || len == -1, "text_to_ipa with empty string returns 0 or -1");
            phonemizer_destroy(ph);
        }
    }

    /* Test 11: very long text */
    {
        ph = phonemizer_create("en-us");
        if (ph) {
            /* Build a 2000+ char string */
            char longtext[2048];
            memset(longtext, 0, sizeof(longtext));
            for (int i = 0; i < 200; i++) {
                strcat(longtext, "hello ");
            }
            char ipa[4096];
            int len = phonemizer_text_to_ipa(ph, longtext, ipa, sizeof(ipa));
            CHECK(len > 0, "text_to_ipa with very long text produces output");
            phonemizer_destroy(ph);
        }
    }

    /* Test 12: text_to_ipa with small buffer truncates safely */
    {
        ph = phonemizer_create("en-us");
        if (ph) {
            char tiny[4];
            int len = phonemizer_text_to_ipa(ph, "Hello world", tiny, sizeof(tiny));
            /* Either truncates or returns error; should not overflow */
            CHECK(len <= (int)sizeof(tiny) || len == -1,
                  "text_to_ipa with tiny buffer does not overflow");
            phonemizer_destroy(ph);
        }
    }

    /* Test 13: text_to_ids with NULL ids_out returns -1 */
    {
        ph = phonemizer_create("en-us");
        if (ph) {
            CHECK(phonemizer_text_to_ids(ph, "hello", NULL, 32) == -1,
                  "text_to_ids with NULL ids_out returns -1");
            phonemizer_destroy(ph);
        }
    }

    /* Test 14: text_to_ids with NULL text returns -1 */
    {
        ph = phonemizer_create("en-us");
        if (ph) {
            int ids[32];
            CHECK(phonemizer_text_to_ids(ph, NULL, ids, 32) == -1,
                  "text_to_ids with NULL text returns -1");
            phonemizer_destroy(ph);
        }
    }

    /* Test 15: ipa_to_ids with NULL phonemizer returns -1 */
    {
        int ids[32];
        CHECK(phonemizer_ipa_to_ids(NULL, "hello", ids, 32) == -1,
              "ipa_to_ids with NULL phonemizer returns -1");
    }

    /* Test 16: ipa_to_ids without phoneme map returns -1 */
    {
        ph = phonemizer_create("en-us");
        if (ph) {
            int ids[32];
            CHECK(phonemizer_ipa_to_ids(ph, "hɛloʊ", ids, 32) == -1,
                  "ipa_to_ids without phoneme map returns -1");
            phonemizer_destroy(ph);
        }
    }

    /* Test 17: load_phoneme_map with NULL path returns -1 */
    {
        ph = phonemizer_create("en-us");
        if (ph) {
            CHECK(phonemizer_load_phoneme_map(ph, NULL) == -1,
                  "load_phoneme_map with NULL path returns -1");
            phonemizer_destroy(ph);
        }
    }

    /* Test 18: load_phoneme_map with nonexistent path returns -1 */
    {
        ph = phonemizer_create("en-us");
        if (ph) {
            CHECK(phonemizer_load_phoneme_map(ph, "/nonexistent/map.json") == -1,
                  "load_phoneme_map with nonexistent path returns -1");
            phonemizer_destroy(ph);
        }
    }

    /* ═══ Additional Speaker Encoder Tests ═══ */
    printf("\n═══ Additional Speaker Encoder Tests ═══\n");

    /* Test 7: extract with NULL audio returns -1 */
    {
        float emb[256] = {0};
        CHECK(speaker_encoder_encode_audio(NULL, NULL, 16000, 16000, emb) == -1,
              "extract with NULL encoder and NULL audio returns -1");
    }

    /* Test 8: extract with NULL embedding_out returns -1 */
    {
        float audio[16000] = {0};
        CHECK(speaker_encoder_encode_audio(NULL, audio, 16000, 16000, NULL) == -1,
              "extract with NULL encoder and NULL output returns -1");
    }

    /* Test 9: extract with 0 samples returns -1 */
    {
        float audio[1] = {0};
        float emb[256] = {0};
        CHECK(speaker_encoder_encode_audio(NULL, audio, 0, 16000, emb) == -1,
              "extract with NULL encoder and 0 samples returns -1");
    }

    /* Test 10: embedding_dim with NULL returns -1 or 0 */
    {
        int dim = speaker_encoder_embedding_dim(NULL);
        CHECK(dim == -1 || dim == 0,
              "embedding_dim with NULL returns -1 or 0");
    }

    /* Test 11: removed — passing dummy pointer (SpeakerEncoder*)1 causes undefined behavior */

    /* Test 12: create with empty string path returns NULL */
    {
        SpeakerEncoder *enc = speaker_encoder_create("");
        CHECK(enc == NULL, "create with empty string path returns NULL");
        if (enc) speaker_encoder_destroy(enc);
    }

    printf("\n══════════════════════════════════════════\n");
    printf("Results: %d passed, %d failed\n", passed, failed);
    if (failed > 0) {
        printf("SOME TESTS FAILED\n");
        return 1;
    }
    printf("ALL PASSED\n");
    return 0;
}
