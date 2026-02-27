/**
 * test_phonemizer_v3.c — Cross-validation test: C phonemizer IPA + phoneme IDs.
 *
 * Phonemizes the same 20+ sentences as scripts/cross_validate_phonemizer.py
 * and prints IPA and phoneme IDs for manual comparison with Python output.
 *
 * Requires: brew install espeak-ng
 * Build: make test-phonemizer-v3
 * Run:   ./build/test-phonemizer-v3
 *
 * Compare output with: python scripts/cross_validate_phonemizer.py
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "phonemizer.h"

#define MAX_IPA 4096
#define MAX_IDS 512

/* Same test sentences as scripts/cross_validate_phonemizer.py */
static const char *const TEST_SENTENCES[] = {
    "Hello world",
    "Good morning",
    "How are you today?",
    "I have 42 cats",
    "The year is 2026",
    "It costs $99.99",
    "Really? Yes! Oh...",
    "Wait — what did you say?",
    "She said, \"Hello!\"",
    "I don't know, she's here",
    "We'll go there, won't we?",
    "That's 5 o'clock",
    "John went to Paris",
    "Dr. Smith from Washington D.C.",
    "NASA launched a rocket",
    "I will read the read book",
    "The bass swam near the bass guitar",
    "She wound the wound tightly",
    "The quick brown fox jumps over the lazy dog. Pack my box with five dozen liquor jugs. "
    "How vexingly quick daft zebras jump!",
    "Well — I suppose... maybe.",
    "A",
    "I",
};
static const int N_SENTENCES = sizeof(TEST_SENTENCES) / sizeof(TEST_SENTENCES[0]);

int main(int argc, char **argv) {
    const char *phoneme_map_path = "models/sonata/phoneme_map.json";
    if (argc >= 2) {
        phoneme_map_path = argv[1];
    }

    printf("\n═══ Phonemizer V3 Cross-Validation Test ═══\n");
    printf("Phoneme map: %s\n", phoneme_map_path);
    printf("Sentences: %d\n\n", N_SENTENCES);

    Phonemizer *ph = phonemizer_create("en-us");
    if (!ph) {
        fprintf(stderr, "[FAIL] phonemizer_create failed\n");
        return 1;
    }

    if (phonemizer_load_phoneme_map(ph, phoneme_map_path) != 0) {
        fprintf(stderr, "[FAIL] phonemizer_load_phoneme_map failed\n");
        phonemizer_destroy(ph);
        return 1;
    }

    printf("Phoneme map loaded. Vocab size: %d\n\n", phonemizer_vocab_size(ph));

    for (int i = 0; i < N_SENTENCES; i++) {
        const char *text = TEST_SENTENCES[i];
        char ipa[MAX_IPA];
        int ids[MAX_IDS];

        int ipa_len = phonemizer_text_to_ipa(ph, text, ipa, MAX_IPA);
        int n_ids = phonemizer_text_to_ids(ph, text, ids, MAX_IDS);

        if (ipa_len < 0) {
            printf("[%2d/%d] ERROR: text_to_ipa failed for: \"%.60s%s\"\n",
                   i + 1, N_SENTENCES, text, strlen(text) > 60 ? "..." : "");
            continue;
        }
        if (n_ids < 0) {
            printf("[%2d/%d] ERROR: text_to_ids failed for: \"%.60s%s\"\n",
                   i + 1, N_SENTENCES, text, strlen(text) > 60 ? "..." : "");
            continue;
        }

        printf("[%2d/%d] Text: \"%.60s%s\"\n", i + 1, N_SENTENCES, text,
               (int)strlen(text) > 60 ? "..." : "");
        printf("      IPA: %s\n", ipa);
        printf("      IDs: ");
        int show = n_ids > 25 ? 25 : n_ids;
        for (int j = 0; j < show; j++) {
            printf("%d ", ids[j]);
        }
        if (n_ids > 25) {
            printf("... ");
        }
        printf("(n=%d)\n\n", n_ids);
    }

    phonemizer_destroy(ph);
    printf("═══ Done. Compare with: python scripts/cross_validate_phonemizer.py ═══\n\n");

    /* ═══════════════════════════════════════════════════════════════════
     * Unit Tests — pass/fail assertions
     * ═══════════════════════════════════════════════════════════════════ */

    int t_pass = 0, t_fail = 0;

#define T_ASSERT(cond, msg) do { \
    if (cond) { t_pass++; printf("  [PASS] %s\n", msg); } \
    else { t_fail++; printf("  [FAIL] %s (line %d)\n", msg, __LINE__); } \
} while(0)

    printf("\n═══ Phonemizer V3 Unit Tests ═══\n");

    /* ── NULL Safety ────────────────────────────────────────────────── */
    printf("\n--- NULL Safety ---\n");

    T_ASSERT(phonemizer_create(NULL) == NULL, "phonemizer_create(NULL) returns NULL");

    {
        Phonemizer *p = phonemizer_create("en-us");
        T_ASSERT(p != NULL, "phonemizer_create(en-us) succeeds");

        if (p) {
            /* text_to_ipa with NULL text */
            char ipa_buf[256];
            int r = phonemizer_text_to_ipa(p, NULL, ipa_buf, sizeof(ipa_buf));
            T_ASSERT(r <= 0, "text_to_ipa(NULL text) returns <= 0");

            /* text_to_ipa with NULL buffer */
            r = phonemizer_text_to_ipa(p, "hello", NULL, 0);
            T_ASSERT(r <= 0, "text_to_ipa(NULL buffer) returns <= 0");

            /* text_to_ids without phoneme map */
            int ids_buf[64];
            r = phonemizer_text_to_ids(p, "hello", ids_buf, 64);
            T_ASSERT(r <= 0, "text_to_ids without map returns <= 0");

            /* vocab_size without map */
            T_ASSERT(phonemizer_vocab_size(p) == 0, "vocab_size without map is 0");

            phonemizer_destroy(p);
        }
    }

    /* phonemizer_destroy(NULL) should not crash */
    phonemizer_destroy(NULL);
    T_ASSERT(1, "phonemizer_destroy(NULL) doesn't crash");

    /* ── Empty Input ────────────────────────────────────────────────── */
    printf("\n--- Empty Input ---\n");

    {
        Phonemizer *p = phonemizer_create("en-us");
        T_ASSERT(p != NULL, "create for empty input tests");
        if (p) {
            char ipa_buf[256];
            int r = phonemizer_text_to_ipa(p, "", ipa_buf, sizeof(ipa_buf));
            T_ASSERT(r >= 0, "empty string text_to_ipa doesn't fail");

            r = phonemizer_text_to_ipa(p, " ", ipa_buf, sizeof(ipa_buf));
            T_ASSERT(r >= 0, "whitespace-only text_to_ipa doesn't fail");

            phonemizer_destroy(p);
        }
    }

    /* ── IPA Generation ─────────────────────────────────────────────── */
    printf("\n--- IPA Generation ---\n");

    {
        Phonemizer *p = phonemizer_create("en-us");
        T_ASSERT(p != NULL, "create for IPA tests");
        if (p) {
            char ipa_buf[MAX_IPA];

            /* Simple word */
            int r = phonemizer_text_to_ipa(p, "hello", ipa_buf, MAX_IPA);
            T_ASSERT(r > 0, "hello produces IPA output");
            T_ASSERT(strlen(ipa_buf) > 0, "hello IPA is non-empty string");
            printf("    hello IPA: %s\n", ipa_buf);

            /* Multiple words */
            r = phonemizer_text_to_ipa(p, "good morning", ipa_buf, MAX_IPA);
            T_ASSERT(r > 0, "good morning produces IPA output");
            printf("    good morning IPA: %s\n", ipa_buf);

            /* Question */
            r = phonemizer_text_to_ipa(p, "how are you?", ipa_buf, MAX_IPA);
            T_ASSERT(r > 0, "question produces IPA output");

            /* Numbers */
            r = phonemizer_text_to_ipa(p, "I have 42 cats", ipa_buf, MAX_IPA);
            T_ASSERT(r > 0, "numbers produce IPA output");

            /* Punctuation */
            r = phonemizer_text_to_ipa(p, "Really? Yes! Oh...", ipa_buf, MAX_IPA);
            T_ASSERT(r > 0, "punctuated text produces IPA");

            /* Contractions */
            r = phonemizer_text_to_ipa(p, "I don't know, she's here", ipa_buf, MAX_IPA);
            T_ASSERT(r > 0, "contractions produce IPA");

            /* Single character */
            r = phonemizer_text_to_ipa(p, "A", ipa_buf, MAX_IPA);
            T_ASSERT(r > 0, "single letter produces IPA");

            r = phonemizer_text_to_ipa(p, "I", ipa_buf, MAX_IPA);
            T_ASSERT(r > 0, "pronoun I produces IPA");

            /* Very small buffer */
            char tiny[4];
            r = phonemizer_text_to_ipa(p, "hello world", tiny, sizeof(tiny));
            T_ASSERT(r >= 0 || r == -1, "tiny buffer handled gracefully");

            phonemizer_destroy(p);
        }
    }

    /* ── Heteronyms ─────────────────────────────────────────────────── */
    printf("\n--- Heteronyms ---\n");

    {
        Phonemizer *p = phonemizer_create("en-us");
        T_ASSERT(p != NULL, "create for heteronym tests");
        if (p) {
            char ipa1[MAX_IPA], ipa2[MAX_IPA];

            /* "read" in present vs past tense context */
            phonemizer_text_to_ipa(p, "I will read the book", ipa1, MAX_IPA);
            phonemizer_text_to_ipa(p, "I read the book yesterday", ipa2, MAX_IPA);
            T_ASSERT(strlen(ipa1) > 0, "present-read produces IPA");
            T_ASSERT(strlen(ipa2) > 0, "past-read produces IPA");
            printf("    read (present): %s\n", ipa1);
            printf("    read (past): %s\n", ipa2);

            /* "bass" (fish) vs "bass" (music) */
            phonemizer_text_to_ipa(p, "The bass swam away", ipa1, MAX_IPA);
            phonemizer_text_to_ipa(p, "The bass guitar is loud", ipa2, MAX_IPA);
            T_ASSERT(strlen(ipa1) > 0, "bass-fish produces IPA");
            T_ASSERT(strlen(ipa2) > 0, "bass-guitar produces IPA");
            printf("    bass (fish): %s\n", ipa1);
            printf("    bass (guitar): %s\n", ipa2);

            /* "wound" (injury) vs "wound" (past tense of wind) */
            phonemizer_text_to_ipa(p, "She wound the clock", ipa1, MAX_IPA);
            phonemizer_text_to_ipa(p, "He has a wound on his arm", ipa2, MAX_IPA);
            T_ASSERT(strlen(ipa1) > 0, "wound-wind produces IPA");
            T_ASSERT(strlen(ipa2) > 0, "wound-injury produces IPA");

            phonemizer_destroy(p);
        }
    }

    /* ── Phoneme Map Loading ────────────────────────────────────────── */
    printf("\n--- Phoneme Map Loading ---\n");

    {
        Phonemizer *p = phonemizer_create("en-us");
        T_ASSERT(p != NULL, "create for map loading tests");
        if (p) {
            /* Load nonexistent map */
            int r = phonemizer_load_phoneme_map(p, "/nonexistent/path.json");
            T_ASSERT(r != 0, "nonexistent map returns error");

            /* Load NULL map */
            r = phonemizer_load_phoneme_map(p, NULL);
            T_ASSERT(r != 0, "NULL map path returns error");

            /* Load actual map */
            r = phonemizer_load_phoneme_map(p, phoneme_map_path);
            if (r == 0) {
                T_ASSERT(1, "phoneme map loaded successfully");
                int vs = phonemizer_vocab_size(p);
                T_ASSERT(vs > 0, "vocab_size > 0 after loading map");
                printf("    Vocab size: %d\n", vs);

                /* Now test text_to_ids */
                int ids[MAX_IDS];
                int n_ids = phonemizer_text_to_ids(p, "hello", ids, MAX_IDS);
                T_ASSERT(n_ids > 0, "hello produces phoneme IDs");
                printf("    hello IDs (%d): ", n_ids);
                for (int j = 0; j < n_ids && j < 15; j++) printf("%d ", ids[j]);
                printf("\n");

                /* All IDs should be non-negative */
                int all_valid = 1;
                for (int j = 0; j < n_ids; j++) {
                    if (ids[j] < 0) { all_valid = 0; break; }
                }
                T_ASSERT(all_valid, "all phoneme IDs are non-negative");

                /* Empty text → 0 or few IDs */
                n_ids = phonemizer_text_to_ids(p, "", ids, MAX_IDS);
                T_ASSERT(n_ids >= 0, "empty text doesn't fail with map");

                /* NULL text */
                n_ids = phonemizer_text_to_ids(p, NULL, ids, MAX_IDS);
                T_ASSERT(n_ids <= 0, "NULL text returns <= 0 IDs");

            } else {
                printf("    (skipping map-dependent tests: map not found at %s)\n",
                       phoneme_map_path);
            }

            phonemizer_destroy(p);
        }
    }

    /* ── ipa_to_ids Direct ──────────────────────────────────────────── */
    printf("\n--- IPA to IDs Direct ---\n");

    {
        Phonemizer *p = phonemizer_create("en-us");
        T_ASSERT(p != NULL, "create for ipa_to_ids tests");
        if (p) {
            int r = phonemizer_load_phoneme_map(p, phoneme_map_path);
            if (r == 0) {
                int ids[MAX_IDS];

                /* First get IPA, then convert to IDs directly */
                char ipa_buf[MAX_IPA];
                phonemizer_text_to_ipa(p, "hello world", ipa_buf, MAX_IPA);
                int n = phonemizer_ipa_to_ids(p, ipa_buf, ids, MAX_IDS);
                T_ASSERT(n > 0, "ipa_to_ids produces IDs from valid IPA");

                /* Compare with text_to_ids for same input */
                int ids2[MAX_IDS];
                int n2 = phonemizer_text_to_ids(p, "hello world", ids2, MAX_IDS);
                T_ASSERT(n == n2, "ipa_to_ids and text_to_ids produce same count");
                if (n == n2 && n > 0) {
                    int match = 1;
                    for (int j = 0; j < n; j++) {
                        if (ids[j] != ids2[j]) { match = 0; break; }
                    }
                    T_ASSERT(match, "ipa_to_ids and text_to_ids produce same IDs");
                }

                /* Empty IPA */
                n = phonemizer_ipa_to_ids(p, "", ids, MAX_IDS);
                T_ASSERT(n >= 0, "empty IPA → 0 or more IDs");

                /* NULL IPA */
                n = phonemizer_ipa_to_ids(p, NULL, ids, MAX_IDS);
                T_ASSERT(n <= 0, "NULL IPA → <= 0");

                /* Zero-size output buffer */
                n = phonemizer_ipa_to_ids(p, ipa_buf, ids, 0);
                T_ASSERT(n == 0, "zero-capacity output → 0 IDs");

            } else {
                printf("    (skipping: map not found)\n");
            }
            phonemizer_destroy(p);
        }
    }

    /* ── Unicode Input ──────────────────────────────────────────────── */
    printf("\n--- Unicode Input ---\n");

    {
        Phonemizer *p = phonemizer_create("en-us");
        T_ASSERT(p != NULL, "create for unicode tests");
        if (p) {
            char ipa_buf[MAX_IPA];

            /* Accented characters */
            int r = phonemizer_text_to_ipa(p, "café naïve résumé", ipa_buf, MAX_IPA);
            T_ASSERT(r > 0, "accented text produces IPA");
            printf("    café naïve résumé IPA: %s\n", ipa_buf);

            /* Em-dash and special punctuation */
            r = phonemizer_text_to_ipa(p, "Wait — what?", ipa_buf, MAX_IPA);
            T_ASSERT(r > 0, "em-dash text produces IPA");

            /* Curly quotes */
            r = phonemizer_text_to_ipa(p, "She said \xe2\x80\x9chello\xe2\x80\x9d", ipa_buf, MAX_IPA);
            T_ASSERT(r >= 0, "curly quotes don't crash");

            /* Ellipsis character */
            r = phonemizer_text_to_ipa(p, "Well\xe2\x80\xa6 maybe", ipa_buf, MAX_IPA);
            T_ASSERT(r >= 0, "ellipsis character handled");

            phonemizer_destroy(p);
        }
    }

    /* ── Language Variants ──────────────────────────────────────────── */
    printf("\n--- Language Variants ---\n");

    {
        /* British English */
        Phonemizer *p_gb = phonemizer_create("en-gb");
        if (p_gb) {
            char ipa_buf[MAX_IPA];
            int r = phonemizer_text_to_ipa(p_gb, "hello", ipa_buf, MAX_IPA);
            T_ASSERT(r > 0, "en-gb produces IPA for hello");
            printf("    en-gb hello: %s\n", ipa_buf);
            phonemizer_destroy(p_gb);
        } else {
            T_ASSERT(1, "en-gb not available (acceptable)");
        }

        /* Invalid language */
        Phonemizer *p_bad = phonemizer_create("xx-invalid-lang");
        T_ASSERT(p_bad == NULL || p_bad != NULL, "invalid language doesn't crash");
        if (p_bad) phonemizer_destroy(p_bad);
    }

    /* ── Pronunciation Dictionary ───────────────────────────────────── */
    printf("\n--- Pronunciation Dictionary ---\n");

    {
        /* NULL safety */
        T_ASSERT(pronunciation_dict_load(NULL) == NULL, "dict_load(NULL) returns NULL");
        T_ASSERT(pronunciation_dict_load("/nonexistent.json") == NULL,
                 "dict_load(bad path) returns NULL");

        pronunciation_dict_destroy(NULL);
        T_ASSERT(1, "dict_destroy(NULL) doesn't crash");

        T_ASSERT(pronunciation_dict_count(NULL) == 0, "dict_count(NULL) == 0");

        /* Apply with NULL dict */
        char out[256];
        int r = pronunciation_dict_apply(NULL, "hello", out, sizeof(out));
        T_ASSERT(r >= 0 || r == -1, "dict_apply(NULL dict) handled");
    }

    /* ── Long Text Stress ───────────────────────────────────────────── */
    printf("\n--- Long Text Stress ---\n");

    {
        Phonemizer *p = phonemizer_create("en-us");
        T_ASSERT(p != NULL, "create for stress test");
        if (p) {
            /* Build a very long sentence */
            char long_text[8192];
            memset(long_text, 0, sizeof(long_text));
            for (int i = 0; i < 50; i++) {
                strcat(long_text, "The quick brown fox jumps over the lazy dog. ");
            }

            char ipa_buf[MAX_IPA];
            int r = phonemizer_text_to_ipa(p, long_text, ipa_buf, MAX_IPA);
            T_ASSERT(r > 0 || r == -1, "very long text handled (success or truncation)");
            if (r > 0) {
                T_ASSERT(strlen(ipa_buf) > 0, "long text IPA is non-empty");
                printf("    Long text IPA length: %zu\n", strlen(ipa_buf));
            }

            phonemizer_destroy(p);
        }
    }

    /* ── Results ────────────────────────────────────────────────────── */
    printf("\n═══════════════════════════════════════════\n");
    printf("  Unit Test Results: %d passed, %d failed\n", t_pass, t_fail);
    printf("═══════════════════════════════════════════\n");
    return t_fail > 0 ? 1 : 0;
}
