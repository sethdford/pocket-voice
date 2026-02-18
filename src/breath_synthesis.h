/**
 * breath_synthesis.h — Human-like breath noise and micropause synthesis.
 *
 * Generates subtle breath noise between phrases and micro-pauses within
 * sentences to cross the uncanny valley between TTS and human speech.
 *
 * Uses Voss-McCartney pink noise filtered through a vocal tract bandpass
 * (200-2000Hz) at -30dB below speech level.
 */

#ifndef BREATH_SYNTHESIS_H
#define BREATH_SYNTHESIS_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct BreathSynth BreathSynth;

/**
 * Create a breath synthesizer.
 * @param sample_rate  Audio sample rate (e.g. 48000)
 * @return Opaque handle, or NULL on failure. NULL-safe destroy.
 */
BreathSynth *breath_create(int sample_rate);

/** Destroy and free all resources. NULL-safe. */
void breath_destroy(BreathSynth *bs);

/**
 * Generate breath noise and write into the audio buffer (additive).
 * Typically called at sentence boundaries or after SSML <break> tags.
 *
 * @param bs          Synth handle
 * @param audio       Buffer to add breath noise into (in-place addition)
 * @param n_samples   Number of samples to generate
 * @param amplitude   Breath volume relative to speech (0.03 = -30dB typical)
 */
void breath_generate(BreathSynth *bs, float *audio, int n_samples, float amplitude);

/**
 * Generate a micro-pause: brief fade-out + silence + fade-in.
 * Used at clause boundaries for natural phrasing rhythm.
 *
 * @param audio       Audio buffer (modified in-place)
 * @param n_samples   Total samples in the pause region
 * @param fade_ms     Duration of the fade-out and fade-in (each) in milliseconds
 * @param sample_rate Audio sample rate
 */
void breath_micropause(float *audio, int n_samples, float fade_ms, int sample_rate);

/**
 * Generate a complete sentence-gap breath: fade-out + breath noise + fade-in.
 * Writes into a caller-provided buffer. Total duration = n_samples.
 *
 * @param bs          Synth handle
 * @param out         Output buffer (overwritten)
 * @param n_samples   Total samples for the gap
 * @param speech_rms  RMS level of surrounding speech (for amplitude scaling)
 */
void breath_sentence_gap(BreathSynth *bs, float *out, int n_samples, float speech_rms);

#ifdef __cplusplus
}
#endif

#endif /* BREATH_SYNTHESIS_H */
