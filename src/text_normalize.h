/**
 * text_normalize.h — Text normalization for SSML <say-as> and auto-detection.
 *
 * Converts numbers, dates, times, currencies, fractions, etc. to spoken word
 * forms. All functions write to caller-provided buffers — zero internal malloc.
 */

#ifndef TEXT_NORMALIZE_H
#define TEXT_NORMALIZE_H

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Main entry: normalize text based on interpret_as tag.
 * Writes null-terminated result to out. Returns bytes written (excluding NUL).
 * If out_cap is too small, output is truncated but always null-terminated.
 */
int text_normalize(const char *text, const char *interpret_as,
                   const char *fmt, char *out, int out_cap);

/* Individual converters (also usable standalone) */
int text_cardinal(const char *text, char *out, int out_cap);
int text_ordinal(const char *text, char *out, int out_cap);
int text_currency(const char *text, char *out, int out_cap);
int text_date(const char *text, const char *fmt, char *out, int out_cap);
int text_time(const char *text, const char *fmt, char *out, int out_cap);
int text_telephone(const char *text, char *out, int out_cap);
int text_fraction(const char *text, char *out, int out_cap);
int text_unit(const char *text, char *out, int out_cap);
int text_characters(const char *text, char *out, int out_cap);
int text_url(const char *text, char *out, int out_cap);
int text_email(const char *text, char *out, int out_cap);

/**
 * Auto-normalize: scans raw text for patterns like $42.50, 1/3, 12:30, 1st,
 * phone numbers, etc. and normalizes inline. Runs on every sentence before TTS,
 * even without SSML. Returns bytes written.
 */
int text_auto_normalize(const char *text, char *out, int out_cap);

/**
 * Expand nonverbalism markers to SSML tags.
 *
 * Supported markers:
 *   [laughter] / [laughs]  → break + happy emotion
 *   [sigh] / [sighs]       → break + sad emotion
 *   [gasp] / [gasps]       → break + surprised emotion
 *   [breath]               → break (uses breath_synthesis in pipeline)
 *   [pause]                → 500ms break
 *   [long pause]           → 1000ms break
 *   [whisper]...[/whisper] → whisper emotion tags
 *
 * Run BEFORE ssml_parse(). Returns bytes written.
 */
int text_expand_nonverbalisms(const char *text, char *out, int out_cap);

/**
 * Expand inline IPA syntax to SSML <phoneme> tags.
 *
 * Supports two syntaxes:
 *   - Cartesia-compatible: <<p|h|o|n|e|m|e>> (pipes stripped, joined as IPA)
 *   - Alternative:         [ipa:phoneme]
 *
 * Both expand to: <phoneme alphabet="ipa" ph="phoneme">phoneme</phoneme>
 * Run BEFORE ssml_parse(). Returns bytes written.
 */
int text_expand_inline_ipa(const char *text, char *out, int out_cap);

/**
 * Apply pronunciation dictionary: case-insensitive word substitution.
 *
 * Replaces matching words with their pronunciation overrides (IPA wrapped
 * in <<>> or raw text). Matches at word boundaries only.
 *
 * @param text          Input text
 * @param out           Output buffer
 * @param out_cap       Output buffer capacity
 * @param words         Array of words to match (case-insensitive)
 * @param replacements  Array of replacement strings (IPA or text)
 * @param n_entries     Number of entries in words/replacements
 * @return              Bytes written
 */
int text_apply_pronunciation_dict(const char *text, char *out, int out_cap,
                                  const char (*words)[64], const char (*replacements)[256],
                                  int n_entries);

#ifdef __cplusplus
}
#endif

#endif /* TEXT_NORMALIZE_H */
