/**
 * ssml_parser.h — Lightweight SSML parser for the native voice pipeline.
 *
 * Parses SSML documents into a flat array of SSMLSegment structs.
 * Uses a hand-written XML tokenizer — no libxml2 dependency.
 * Integrates with text_normalize.c for <say-as> tag handling.
 */

#ifndef SSML_PARSER_H
#define SSML_PARSER_H

#ifdef __cplusplus
extern "C" {
#endif

#define SSML_MAX_TEXT    4096
#define SSML_MAX_VOICE   256
#define SSML_MAX_SEGMENTS 64

typedef struct {
    char  text[SSML_MAX_TEXT];
    char  voice[SSML_MAX_VOICE];
    float rate;              /* 1.0 = normal */
    float pitch;             /* 1.0 = normal */
    float volume;            /* 1.0 = normal */
    int   break_before_ms;
    int   break_after_ms;
    int   is_audio;          /* 1 if text is an audio URL */
} SSMLSegment;

/**
 * Parse SSML string into segments array. Returns segment count.
 * If input is not SSML (no <speak> tag), returns 1 segment with raw text.
 * @param input         The input text (may or may not be SSML)
 * @param segments      Output array of segments
 * @param max_segments  Size of segments array (use SSML_MAX_SEGMENTS)
 * @return Number of segments written, or -1 on parse error.
 */
int ssml_parse(const char *input, SSMLSegment *segments, int max_segments);

/** Returns 1 if text looks like SSML (starts with <speak or <?xml). */
int ssml_is_ssml(const char *text);

/** Parse prosody rate string to a float multiplier. */
float ssml_parse_rate(const char *value);

/** Parse prosody pitch string to a float multiplier. */
float ssml_parse_pitch(const char *value);

/** Parse prosody volume string to a float multiplier. */
float ssml_parse_volume(const char *value);

#ifdef __cplusplus
}
#endif

#endif /* SSML_PARSER_H */
