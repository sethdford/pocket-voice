#ifndef NEURAL_BACKCHANNEL_H
#define NEURAL_BACKCHANNEL_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct NeuralBackchannel NeuralBackchannel;

typedef enum {
    NBC_MHM = 0,
    NBC_YEAH,
    NBC_RIGHT,
    NBC_OKAY,
    NBC_UH_HUH,
    NBC_I_SEE,
    NBC_SURE,
    NBC_HMHM,
    NBC_LAUGH,      /* short laugh */
    NBC_COUNT
} NBCType;

typedef struct {
    int   sample_rate;     /* Output sample rate (24000) */
    int   max_duration_ms; /* Max backchannel duration (500ms) */
    int   cache_enabled;   /* Pre-generate cache at startup */
} NBCConfig;

/* Create neural backchannel generator.
 * tts_engine: opaque TTS engine pointer (SonataEngine*)
 *   Used for generating backchannel audio through the TTS pipeline.
 *   If NULL, falls back to built-in pink noise synthesis. */
NeuralBackchannel *nbc_create(const NBCConfig *cfg, void *tts_engine);
void nbc_destroy(NeuralBackchannel *nbc);

/* Pre-generate backchannel cache for all types.
 * Should be called at startup after TTS is initialized.
 * Generates each backchannel type through Sonata TTS and caches the audio.
 * If speaker embedding is set, backchannels match the voice. */
int nbc_warm_cache(NeuralBackchannel *nbc);

/* Get cached backchannel audio for immediate playback.
 * Returns pointer to internal buffer, sets *out_len.
 * Returns NULL if not cached — call nbc_generate() instead. */
const float *nbc_get_cached(NeuralBackchannel *nbc, NBCType type, int *out_len);

/* Generate backchannel audio on-demand (slower, ~50-100ms).
 * out_pcm: caller-provided buffer [max_samples].
 * Returns samples written, or -1 on error. */
int nbc_generate(NeuralBackchannel *nbc, NBCType type,
                 float *out_pcm, int max_samples);

/* Set speaker embedding for voice-matched backchannels.
 * embedding: float array from speaker encoder.
 * Regenerates cache if cache_enabled. */
int nbc_set_speaker(NeuralBackchannel *nbc, const float *embedding, int dim);

/* Set emotion for emotionally-appropriate backchannels.
 * The next generation/cache will use this emotion. */
void nbc_set_emotion(NeuralBackchannel *nbc, int emotion_id);

/* Load custom audio for a type (override neural generation). */
int nbc_load_wav(NeuralBackchannel *nbc, NBCType type, const char *wav_path);

#ifdef __cplusplus
}
#endif

#endif /* NEURAL_BACKCHANNEL_H */
