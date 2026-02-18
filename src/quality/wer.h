/**
 * wer.h — Word Error Rate (WER) and Character Error Rate (CER).
 *
 * WER is the primary metric for STT quality. It measures the edit distance
 * between the reference transcript and the hypothesis, normalized by
 * reference length:
 *
 *   WER = (Substitutions + Deletions + Insertions) / Reference_Words
 *
 * Lower is better. 0.0 = perfect. Human-level WER is ~5% on clean speech.
 * State-of-the-art STT systems achieve 2-4% on LibriSpeech test-clean.
 *
 * Golden signals:
 *   - WER < 5%  on clean speech      → human-level
 *   - WER < 10% on noisy speech      → production-grade
 *   - WER < 3%  on domain-specific   → best-in-class
 */

#ifndef WER_H
#define WER_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int substitutions;
    int deletions;
    int insertions;
    int ref_words;
    int hyp_words;
    float wer;     /* Word Error Rate (0.0 - 1.0+) */
    float cer;     /* Character Error Rate */
    float accuracy; /* 1.0 - WER, clamped to [0, 1] */
} WERResult;

/**
 * Compute WER between reference and hypothesis text.
 * Both strings are split on whitespace. Case-insensitive comparison.
 * Punctuation is stripped before comparison.
 *
 * @param reference  Ground-truth transcript
 * @param hypothesis STT output
 * @return WERResult with all metrics
 */
WERResult wer_compute(const char *reference, const char *hypothesis);

/**
 * Compute CER (Character Error Rate) between reference and hypothesis.
 * Operates on individual characters rather than words.
 *
 * @param reference  Ground-truth text
 * @param hypothesis STT output
 * @return CER value (0.0 = perfect)
 */
float cer_compute(const char *reference, const char *hypothesis);

/**
 * Normalize text for WER comparison: lowercase, strip punctuation,
 * collapse whitespace. Writes result to `out`.
 * @return Length of normalized text
 */
int wer_normalize(const char *input, char *out, int out_cap);

#ifdef __cplusplus
}
#endif

#endif /* WER_H */
