/**
 * phonemizer.h — espeak-ng phonemizer for TTS text preprocessing.
 *
 * Converts text → IPA phoneme string → integer phoneme IDs.
 * Used to improve TTS pronunciation accuracy for heteronyms, proper nouns,
 * and other ambiguous text that SentencePiece tokenization handles poorly.
 *
 * Requires: brew install espeak-ng
 *
 * Note: espeak-ng is NOT thread-safe. Do not call phonemizer functions
 * from multiple threads concurrently.
 */

#ifndef PHONEMIZER_H
#define PHONEMIZER_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct Phonemizer Phonemizer;

/**
 * Create a phonemizer for the given language.
 *
 * @param language  BCP-47 language tag (e.g. "en-us", "en-gb", "de", "fr")
 * @return          Opaque handle, or NULL on failure
 */
Phonemizer *phonemizer_create(const char *language);

/** Destroy phonemizer and free resources. */
void phonemizer_destroy(Phonemizer *ph);

/**
 * Convert text to IPA phoneme string.
 *
 * @param ph         Phonemizer handle
 * @param text       Input UTF-8 text
 * @param ipa_out    Output buffer for IPA string (null-terminated)
 * @param max_len    Size of ipa_out buffer
 * @return           Length of IPA string, or -1 on error
 */
int phonemizer_text_to_ipa(Phonemizer *ph, const char *text, char *ipa_out, int max_len);

/**
 * Load a phoneme-to-ID mapping from a JSON file.
 *
 * JSON format: { "phoneme_string": integer_id, ... }
 * e.g. { "a": 1, "b": 2, "ə": 3, " ": 4 }
 *
 * @param ph        Phonemizer handle
 * @param json_path Path to phoneme map JSON file
 * @return          0 on success, -1 on error
 */
int phonemizer_load_phoneme_map(Phonemizer *ph, const char *json_path);

/**
 * Convert IPA string to integer phoneme IDs.
 *
 * Requires a phoneme map to be loaded first via phonemizer_load_phoneme_map().
 * Multi-character IPA symbols are matched greedily (longest match first).
 *
 * @param ph        Phonemizer handle
 * @param ipa       Input IPA string (from phonemizer_text_to_ipa)
 * @param ids_out   Output buffer for phoneme IDs
 * @param max_ids   Size of ids_out buffer
 * @return          Number of IDs produced, or -1 on error
 */
int phonemizer_ipa_to_ids(Phonemizer *ph, const char *ipa, int *ids_out, int max_ids);

/**
 * Convenience: text → phoneme IDs in one call.
 *
 * Combines text_to_ipa + ipa_to_ids. Requires phoneme map loaded.
 *
 * @param ph        Phonemizer handle
 * @param text      Input UTF-8 text
 * @param ids_out   Output buffer for phoneme IDs
 * @param max_ids   Size of ids_out buffer
 * @return          Number of IDs produced, or -1 on error
 */
int phonemizer_text_to_ids(Phonemizer *ph, const char *text, int *ids_out, int max_ids);

/**
 * Get the number of unique phonemes in the loaded map.
 * Returns 0 if no map is loaded.
 */
int phonemizer_vocab_size(const Phonemizer *ph);

/* ── Pronunciation Dictionary ─────────────────────────────────── */

typedef struct PronunciationDict PronunciationDict;

/**
 * Load a pronunciation dictionary from a JSON file.
 *
 * JSON format: array of objects with "text" and "pronunciation" fields:
 *   [
 *     {"text": "Sonata", "pronunciation": "<<s|ə|ˈ|n|ɑː|t|ə>>"},
 *     {"text": "GIF",    "pronunciation": "jiff"}
 *   ]
 *
 * Pronunciation can be:
 *   - Inline IPA: "<<phonemes>>" (expanded before SSML parse)
 *   - Sounds-like: plain text replacement
 *
 * @param json_path  Path to JSON pronunciation dictionary file
 * @return           Opaque handle, or NULL on failure
 */
PronunciationDict *pronunciation_dict_load(const char *json_path);

/** Destroy pronunciation dictionary. NULL-safe. */
void pronunciation_dict_destroy(PronunciationDict *dict);

/**
 * Apply pronunciation dictionary to text.
 * Replaces matched words (case-insensitive, word boundary) with their
 * pronunciation strings. Run BEFORE inline IPA expansion and SSML parse.
 *
 * @param dict     Pronunciation dictionary
 * @param text     Input text
 * @param out      Output buffer
 * @param out_cap  Output buffer capacity
 * @return         Bytes written (excluding NUL), or -1 on error
 */
int pronunciation_dict_apply(const PronunciationDict *dict, const char *text,
                             char *out, int out_cap);

/** Number of entries in the dictionary. */
int pronunciation_dict_count(const PronunciationDict *dict);

#ifdef __cplusplus
}
#endif

#endif /* PHONEMIZER_H */
