/**
 * emphasis_predict.h — Lightweight emphasis prediction for TTS.
 *
 * Analyzes plain text from the LLM and marks words that should receive
 * emphasis in speech, without requiring SSML from the LLM itself.
 * Wraps detected emphasis words in <emphasis> SSML tags.
 *
 * Rules (linguistics-informed):
 *   - Sentence-final content words (nouns, verbs, adjectives) get mild emphasis
 *   - Contrast markers: word after "but", "however", "actually", "instead"
 *   - Intensifiers: "very", "really", "extremely", "absolutely" boost the next word
 *   - Enumeration stress: last item in a list ("X, Y, and Z") gets emphasis
 *   - Negation stress: "not", "never", "no" + next word
 *   - Quoted speech detection: text between quotes gets voice tag
 */

#ifndef EMPHASIS_PREDICT_H
#define EMPHASIS_PREDICT_H

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Analyze text and insert <emphasis> SSML tags at predicted emphasis points.
 *
 * @param input   Plain text (may already contain SSML — existing tags preserved)
 * @param output  Output buffer for text with added emphasis tags
 * @param outsize Size of output buffer
 * @return        Number of emphasis points inserted
 */
int emphasis_predict(const char *input, char *output, int outsize);

/**
 * Detect quoted speech and wrap in <voice name="quoted"> tags.
 * Enables multi-voice rendering for reported speech.
 *
 * @param input   Input text
 * @param output  Output with <voice> tags around quoted speech
 * @param outsize Output buffer size
 * @return        Number of quoted segments found
 */
int emphasis_detect_quotes(const char *input, char *output, int outsize);

#ifdef __cplusplus
}
#endif

#endif /* EMPHASIS_PREDICT_H */
