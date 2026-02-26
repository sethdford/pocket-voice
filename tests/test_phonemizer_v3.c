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
    return 0;
}
