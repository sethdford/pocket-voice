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

/**
 * Auto-normalize: scans raw text for patterns like $42.50, 1/3, 12:30, 1st,
 * phone numbers, etc. and normalizes inline. Runs on every sentence before TTS,
 * even without SSML. Returns bytes written.
 */
int text_auto_normalize(const char *text, char *out, int out_cap);

#ifdef __cplusplus
}
#endif

#endif /* TEXT_NORMALIZE_H */
