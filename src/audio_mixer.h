#ifndef AUDIO_MIXER_H
#define AUDIO_MIXER_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct AudioMixer AudioMixer;

typedef enum {
    MIX_CHANNEL_MAIN = 0,        /* TTS primary output */
    MIX_CHANNEL_BACKCHANNEL,     /* Short backchannel clips */
    MIX_CHANNEL_PRESYNTHESIZED,  /* Pre-cached response audio */
    MIX_CHANNEL_CLOUD_AUDIO,     /* Cloud LLM audio stream */
    MIX_CHANNEL_COUNT
} MixChannel;

typedef struct {
    int   sample_rate;         /* Output sample rate (e.g. 24000) */
    int   block_size;          /* Samples per mix block (e.g. 480 = 20ms) */
    float ducking_gain;        /* Gain applied to ducked channels (e.g. 0.3) */
    int   crossfade_samples;   /* Crossfade length for source transitions (e.g. 240) */
} AudioMixerConfig;

/* Create mixer. Returns NULL on failure. */
AudioMixer *audio_mixer_create(const AudioMixerConfig *cfg);
void audio_mixer_destroy(AudioMixer *mixer);

/* Write audio to a channel's ring buffer.
 * Non-blocking; drops samples if buffer is full.
 * Returns number of samples written. */
int audio_mixer_write(AudioMixer *mixer, MixChannel channel,
                      const float *pcm, int n_samples);

/* Read mixed output (all channels blended).
 * Applies priority, ducking, and crossfade.
 * out_pcm: output buffer [max_samples].
 * Returns samples written to out_pcm. */
int audio_mixer_read(AudioMixer *mixer, float *out_pcm, int max_samples);

/* Check if a channel has audio pending. */
int audio_mixer_channel_active(const AudioMixer *mixer, MixChannel channel);

/* Check if any channel has audio pending. */
int audio_mixer_any_active(const AudioMixer *mixer);

/* Set per-channel gain (0.0 = mute, 1.0 = full). Default 1.0 for all. */
void audio_mixer_set_gain(AudioMixer *mixer, MixChannel channel, float gain);

/* Set channel priority (higher = less likely to be ducked).
 * Default: MAIN=10, BACKCHANNEL=5, PRESYNTHESIZED=8, CLOUD=9. */
void audio_mixer_set_priority(AudioMixer *mixer, MixChannel channel, int priority);

/* Flush a channel's buffer (e.g. on barge-in, cancel main TTS). */
void audio_mixer_flush(AudioMixer *mixer, MixChannel channel);

/* Flush all channels. */
void audio_mixer_flush_all(AudioMixer *mixer);

/* Reset all state. */
void audio_mixer_reset(AudioMixer *mixer);

/* Get the number of samples pending in a channel. */
int audio_mixer_pending(const AudioMixer *mixer, MixChannel channel);

#ifdef __cplusplus
}
#endif

#endif /* AUDIO_MIXER_H */
