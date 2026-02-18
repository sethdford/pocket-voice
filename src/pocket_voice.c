/**
 * pocket_voice.c — Ultra-low-latency voice engine for Apple Silicon.
 *
 * Provides CoreAudio VoiceProcessingIO (mic+speaker+AEC), lock-free SPSC
 * ring buffers, energy-based VAD with hysteresis, and vDSP resampling.
 * The entire audio path runs on CoreAudio's real-time thread — no Python,
 * no GIL, no allocations in the hot path.
 *
 * Build: cc -O3 -shared -fPIC -arch arm64 \
 *        -framework Accelerate -framework CoreAudio -framework AudioToolbox \
 *        -o libpocket_voice.dylib pocket_voice.c
 */

#include <Accelerate/Accelerate.h>
#include <AudioToolbox/AudioToolbox.h>
#include <CoreAudio/CoreAudio.h>
#include <stdatomic.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "neon_audio.h"

/* -----------------------------------------------------------------------
 * Lock-Free SPSC Ring Buffer
 *
 * Single-Producer, Single-Consumer with cache-line aligned head/tail
 * to prevent false sharing. Power-of-2 size for bitwise masking.
 * ----------------------------------------------------------------------- */

#define CACHE_LINE 128  /* Apple M-series DMA cache line */

typedef struct {
    float *buffer;
    uint32_t size;
    uint32_t mask;       /* size - 1 */
    _Alignas(CACHE_LINE) _Atomic uint64_t head;
    _Alignas(CACHE_LINE) _Atomic uint64_t tail;
} RingBuffer;

static int ring_init(RingBuffer *rb, uint32_t size) {
    if (size == 0 || (size & (size - 1)) != 0) return -1;
    rb->buffer = (float *)calloc(size, sizeof(float));
    if (!rb->buffer) return -1;
    rb->size = size;
    rb->mask = size - 1;
    atomic_store_explicit(&rb->head, 0, memory_order_relaxed);
    atomic_store_explicit(&rb->tail, 0, memory_order_relaxed);
    return 0;
}

static void ring_destroy(RingBuffer *rb) {
    free(rb->buffer);
    rb->buffer = NULL;
}

static uint32_t ring_available_read(const RingBuffer *rb) {
    uint64_t h = atomic_load_explicit(&rb->head, memory_order_acquire);
    uint64_t t = atomic_load_explicit(&rb->tail, memory_order_relaxed);
    return (uint32_t)(h - t);
}

static uint32_t ring_available_write(const RingBuffer *rb) {
    uint64_t h = atomic_load_explicit(&rb->head, memory_order_relaxed);
    uint64_t t = atomic_load_explicit(&rb->tail, memory_order_acquire);
    return rb->size - (uint32_t)(h - t);
}

static int ring_write(RingBuffer *rb, const float *data, uint32_t count) {
    if (ring_available_write(rb) < count) return -1;
    uint64_t h = atomic_load_explicit(&rb->head, memory_order_relaxed);
    uint32_t offset = (uint32_t)(h & rb->mask);
    uint32_t first = rb->size - offset;
    if (first >= count) {
        neon_copy_f32(rb->buffer + offset, data, (int)count);
    } else {
        neon_copy_f32(rb->buffer + offset, data, (int)first);
        neon_copy_f32(rb->buffer, data + first, (int)(count - first));
    }
    atomic_store_explicit(&rb->head, h + count, memory_order_release);
    return 0;
}

static int ring_read(RingBuffer *rb, float *data, uint32_t count) {
    if (ring_available_read(rb) < count) return -1;
    uint64_t t = atomic_load_explicit(&rb->tail, memory_order_relaxed);
    uint32_t offset = (uint32_t)(t & rb->mask);
    uint32_t first = rb->size - offset;
    if (first >= count) {
        neon_copy_f32(data, rb->buffer + offset, (int)count);
    } else {
        neon_copy_f32(data, rb->buffer + offset, (int)first);
        neon_copy_f32(data + first, rb->buffer, (int)(count - first));
    }
    atomic_store_explicit(&rb->tail, t + count, memory_order_release);
    return 0;
}

static void ring_flush(RingBuffer *rb) {
    uint64_t h = atomic_load_explicit(&rb->head, memory_order_acquire);
    atomic_store_explicit(&rb->tail, h, memory_order_release);
}

/* -----------------------------------------------------------------------
 * Voice Activity Detection
 *
 * Energy-based VAD using vDSP_rmsqv with hysteresis state machine.
 * Runs inside the CoreAudio capture callback (<100us budget).
 * ----------------------------------------------------------------------- */

typedef enum {
    VAD_SILENCE     = 0,
    VAD_SPEECH_START = 1,
    VAD_SPEECH      = 2,
    VAD_SPEECH_END  = 3
} VADStateEnum;

typedef struct {
    float energy_threshold;    /* RMS threshold to trigger speech onset */
    float silence_threshold;   /* Lower threshold for speech offset (hysteresis) */
    int hangover_frames;       /* Frames to wait before declaring SPEECH_END */
    int hangover_counter;
    int speech_frame_count;
    int min_speech_frames;     /* Minimum speech frames to confirm (debounce) */
    VADStateEnum state;
} VADState;

static void vad_init(VADState *vad, float energy_thresh, float silence_thresh,
                     int hangover_frames, int min_speech_frames) {
    vad->energy_threshold  = energy_thresh;
    vad->silence_threshold = silence_thresh;
    vad->hangover_frames   = hangover_frames;
    vad->hangover_counter  = 0;
    vad->speech_frame_count = 0;
    vad->min_speech_frames = min_speech_frames;
    vad->state             = VAD_SILENCE;
}

static VADStateEnum vad_process_frame(VADState *vad, const float *frame,
                                      int frame_len) {
    float rms;
    vDSP_rmsqv(frame, 1, &rms, (vDSP_Length)frame_len);

    switch (vad->state) {
    case VAD_SILENCE:
        if (rms >= vad->energy_threshold) {
            vad->speech_frame_count = 1;
            vad->state = VAD_SPEECH_START;
        }
        break;
    case VAD_SPEECH_START:
        if (rms >= vad->silence_threshold) {
            vad->speech_frame_count++;
            if (vad->speech_frame_count >= vad->min_speech_frames) {
                vad->state = VAD_SPEECH;
                vad->hangover_counter = vad->hangover_frames;
            }
        } else {
            vad->state = VAD_SILENCE;
            vad->speech_frame_count = 0;
        }
        break;
    case VAD_SPEECH:
        if (rms >= vad->silence_threshold) {
            vad->hangover_counter = vad->hangover_frames;
        } else {
            vad->hangover_counter--;
            if (vad->hangover_counter <= 0) {
                vad->state = VAD_SPEECH_END;
            }
        }
        break;
    case VAD_SPEECH_END:
        vad->speech_frame_count = 0;
        vad->state = VAD_SILENCE;
        break;
    }

    return vad->state;
}

/* -----------------------------------------------------------------------
 * Audio Resampler (vDSP)
 *
 * 31-tap lowpass FIR for 48kHz -> 24kHz decimation and reverse.
 * ----------------------------------------------------------------------- */

#define RESAMPLE_TAPS 31
#define MAX_RESAMPLE_INPUT 16384

/* 31-tap Parks-McClellan lowpass at 0.45*Nyquist (Kaiser window, beta=5) */
static const float g_antialias_fir[RESAMPLE_TAPS] = {
    -0.0003f, -0.0012f, -0.0009f,  0.0031f,  0.0065f, -0.0008f,
    -0.0147f, -0.0147f,  0.0110f,  0.0402f,  0.0166f, -0.0498f,
    -0.0742f,  0.0127f,  0.1894f,  0.3312f,  0.3312f,  0.1894f,
     0.0127f, -0.0742f, -0.0498f,  0.0166f,  0.0402f,  0.0110f,
    -0.0147f, -0.0147f, -0.0008f,  0.0065f,  0.0031f, -0.0009f,
    -0.0012f
};

void voice_engine_resample_48_to_24(const float *input, float *output,
                                     int input_len) {
    int out_len = input_len / 2;
    if (out_len <= 0 || input_len <= 0) return;

    /* vDSP_desamp reads up to (out_len-1)*stride + RESAMPLE_TAPS samples from
     * the input. We must pad the input to avoid a buffer overrun. */
    int padded_len = (out_len - 1) * 2 + RESAMPLE_TAPS;
    if (padded_len <= input_len) {
        vDSP_desamp(input, 2, g_antialias_fir, output, (vDSP_Length)out_len,
                    (vDSP_Length)RESAMPLE_TAPS);
    } else {
        float padded[MAX_RESAMPLE_INPUT + RESAMPLE_TAPS];
        int copy_len = input_len < MAX_RESAMPLE_INPUT ? input_len : MAX_RESAMPLE_INPUT;
        memcpy(padded, input, (size_t)copy_len * sizeof(float));
        memset(padded + copy_len, 0,
               (size_t)(padded_len - copy_len) * sizeof(float));
        vDSP_desamp(padded, 2, g_antialias_fir, output, (vDSP_Length)out_len,
                    (vDSP_Length)RESAMPLE_TAPS);
    }
}

/* Static scratch buffer for upsample — avoids malloc in the hot audio path.
 * MAX_RESAMPLE_INPUT * 2 + RESAMPLE_TAPS - 1 covers the worst case. */
static float g_upsample_scratch[MAX_RESAMPLE_INPUT * 2 + RESAMPLE_TAPS];

void voice_engine_resample_24_to_48(const float *input, float *output,
                                     int input_len) {
    if (input_len <= 0) return;
    int clamped = input_len < MAX_RESAMPLE_INPUT ? input_len : MAX_RESAMPLE_INPUT;
    int stuffed_len = clamped * 2;
    int padded_len = stuffed_len + RESAMPLE_TAPS - 1;

    memset(g_upsample_scratch, 0, (size_t)padded_len * sizeof(float));
    neon_zero_stuff_2x(input, g_upsample_scratch, clamped);
    vDSP_conv(g_upsample_scratch, 1, g_antialias_fir + RESAMPLE_TAPS - 1, -1,
              output, 1, (vDSP_Length)stuffed_len, (vDSP_Length)RESAMPLE_TAPS);
}

/* -----------------------------------------------------------------------
 * Voice Engine — CoreAudio VoiceProcessingIO
 * ----------------------------------------------------------------------- */

#define RING_SIZE 262144  /* ~5.5s at 48kHz — enough for long TTS responses */

typedef struct {
    AudioComponentInstance voice_unit;
    RingBuffer capture_ring;
    RingBuffer playback_ring;
    VADState vad;
    uint32_t sample_rate;
    uint32_t buffer_frames;
    _Atomic int barge_in;
    _Atomic int playing;       /* Whether TTS audio is in the playback ring */
    _Atomic int running;
    float *resample_buf;       /* Scratch for 48->24 resampling in callback */
} VoiceEngine;

/* Input callback: CoreAudio delivers mic frames here (real-time thread) */
static OSStatus capture_callback(
    void *inRefCon,
    AudioUnitRenderActionFlags *ioActionFlags,
    const AudioTimeStamp *inTimeStamp,
    UInt32 inBusNumber,
    UInt32 inNumberFrames,
    AudioBufferList *ioData
) {
    VoiceEngine *engine = (VoiceEngine *)inRefCon;

    /* Clamp to prevent stack overrun if CoreAudio delivers unexpectedly large frames */
    if (inNumberFrames > 4096) inNumberFrames = 4096;

    /* Render mic input from the AudioUnit */
    AudioBufferList buf_list;
    buf_list.mNumberBuffers = 1;
    buf_list.mBuffers[0].mNumberChannels = 1;
    buf_list.mBuffers[0].mDataByteSize = inNumberFrames * sizeof(float);
    float frame_buf[4096];
    buf_list.mBuffers[0].mData = frame_buf;

    OSStatus status = AudioUnitRender(engine->voice_unit, ioActionFlags,
                                      inTimeStamp, inBusNumber,
                                      inNumberFrames, &buf_list);
    if (status != noErr) return status;

    float *samples = (float *)buf_list.mBuffers[0].mData;

    /* Run VAD on captured audio */
    VADStateEnum vad_state = vad_process_frame(&engine->vad, samples,
                                               (int)inNumberFrames);

    /* Barge-in: user started speaking while TTS is playing */
    if ((vad_state == VAD_SPEECH_START || vad_state == VAD_SPEECH) &&
        atomic_load_explicit(&engine->playing, memory_order_relaxed)) {
        atomic_store_explicit(&engine->barge_in, 1, memory_order_release);
        ring_flush(&engine->playback_ring);
    }

    /* Write captured audio to ring buffer */
    ring_write(&engine->capture_ring, samples, inNumberFrames);

    return noErr;
}

/* Output callback: CoreAudio needs frames for the speaker (real-time thread) */
static OSStatus render_callback(
    void *inRefCon,
    AudioUnitRenderActionFlags *ioActionFlags,
    const AudioTimeStamp *inTimeStamp,
    UInt32 inBusNumber,
    UInt32 inNumberFrames,
    AudioBufferList *ioData
) {
    VoiceEngine *engine = (VoiceEngine *)inRefCon;
    float *out = (float *)ioData->mBuffers[0].mData;

    uint32_t avail = ring_available_read(&engine->playback_ring);
    if (avail >= inNumberFrames) {
        ring_read(&engine->playback_ring, out, inNumberFrames);
    } else {
        /* Underrun: output silence */
        memset(out, 0, inNumberFrames * sizeof(float));
        if (avail > 0) {
            ring_read(&engine->playback_ring, out, avail);
        }
        /* Ring drained — clear the playing flag so VAD doesn't falsely
         * trigger barge-in when no TTS audio is playing. */
        atomic_store_explicit(&engine->playing, 0, memory_order_release);
    }

    return noErr;
}

/* -----------------------------------------------------------------------
 * Public API
 * ----------------------------------------------------------------------- */

VoiceEngine* voice_engine_create(uint32_t sample_rate, uint32_t buffer_frames) {
    VoiceEngine *engine = (VoiceEngine *)calloc(1, sizeof(VoiceEngine));
    if (!engine) return NULL;

    engine->sample_rate = sample_rate;
    engine->buffer_frames = buffer_frames;
    atomic_store(&engine->barge_in, 0);
    atomic_store(&engine->playing, 0);
    atomic_store(&engine->running, 0);

    if (ring_init(&engine->capture_ring, RING_SIZE) != 0) goto fail;
    if (ring_init(&engine->playback_ring, RING_SIZE) != 0) goto fail;

    /* VAD defaults: 0.01 onset, 0.005 offset, 300ms hangover at
     * 48kHz/buffer_frames => hangover_frames = 300ms / (buffer_frames/sample_rate)
     */
    int hangover = (int)(0.3f * (float)sample_rate / (float)buffer_frames);
    int min_speech = (int)(0.05f * (float)sample_rate / (float)buffer_frames);
    if (min_speech < 1) min_speech = 1;
    vad_init(&engine->vad, 0.01f, 0.005f, hangover, min_speech);

    engine->resample_buf = (float *)calloc(MAX_RESAMPLE_INPUT, sizeof(float));

    return engine;

fail:
    ring_destroy(&engine->capture_ring);
    ring_destroy(&engine->playback_ring);
    free(engine);
    return NULL;
}

int voice_engine_start(VoiceEngine *engine) {
    if (atomic_load(&engine->running)) return 0;

    /* Find VoiceProcessingIO AudioUnit */
    AudioComponentDescription desc = {
        .componentType         = kAudioUnitType_Output,
        .componentSubType      = kAudioUnitSubType_VoiceProcessingIO,
        .componentManufacturer = kAudioUnitManufacturer_Apple,
    };

    AudioComponent comp = AudioComponentFindNext(NULL, &desc);
    if (!comp) {
        fprintf(stderr, "pocket_voice: VoiceProcessingIO not found\n");
        return -1;
    }

    OSStatus status = AudioComponentInstanceNew(comp, &engine->voice_unit);
    if (status != noErr) {
        fprintf(stderr, "pocket_voice: AudioComponentInstanceNew failed: %d\n",
                (int)status);
        return -1;
    }

    /* Macro to clean up the AudioUnit on any subsequent error */
#define FAIL_START() do { \
    AudioComponentInstanceDispose(engine->voice_unit); \
    engine->voice_unit = NULL; \
    return -1; \
} while(0)

    /* Enable input on bus 1 */
    UInt32 enable = 1;
    status = AudioUnitSetProperty(engine->voice_unit,
                                  kAudioOutputUnitProperty_EnableIO,
                                  kAudioUnitScope_Input, 1,
                                  &enable, sizeof(enable));
    if (status != noErr) {
        fprintf(stderr, "pocket_voice: enable input failed: %d\n", (int)status);
        FAIL_START();
    }

    /* Set stream format: float32, mono, at requested sample rate */
    AudioStreamBasicDescription fmt = {
        .mSampleRate       = (Float64)engine->sample_rate,
        .mFormatID         = kAudioFormatLinearPCM,
        .mFormatFlags      = kAudioFormatFlagIsFloat |
                             kAudioFormatFlagIsPacked |
                             kAudioFormatFlagIsNonInterleaved,
        .mBytesPerPacket   = sizeof(float),
        .mFramesPerPacket  = 1,
        .mBytesPerFrame    = sizeof(float),
        .mChannelsPerFrame = 1,
        .mBitsPerChannel   = 32,
    };

    /* Input format on bus 1 (mic) */
    status = AudioUnitSetProperty(engine->voice_unit,
                                  kAudioUnitProperty_StreamFormat,
                                  kAudioUnitScope_Output, 1,
                                  &fmt, sizeof(fmt));
    if (status != noErr) {
        fprintf(stderr, "pocket_voice: set input format failed: %d\n",
                (int)status);
        FAIL_START();
    }

    /* Output format on bus 0 (speaker) */
    status = AudioUnitSetProperty(engine->voice_unit,
                                  kAudioUnitProperty_StreamFormat,
                                  kAudioUnitScope_Input, 0,
                                  &fmt, sizeof(fmt));
    if (status != noErr) {
        fprintf(stderr, "pocket_voice: set output format failed: %d\n",
                (int)status);
        FAIL_START();
    }

    /* Set buffer size hint */
    UInt32 frames = engine->buffer_frames;
    AudioUnitSetProperty(engine->voice_unit,
                         kAudioDevicePropertyBufferFrameSize,
                         kAudioUnitScope_Global, 0,
                         &frames, sizeof(frames));

    /* Register capture callback on bus 1 */
    AURenderCallbackStruct input_cb = {
        .inputProc       = capture_callback,
        .inputProcRefCon = engine,
    };
    status = AudioUnitSetProperty(engine->voice_unit,
                                  kAudioOutputUnitProperty_SetInputCallback,
                                  kAudioUnitScope_Global, 1,
                                  &input_cb, sizeof(input_cb));
    if (status != noErr) {
        fprintf(stderr, "pocket_voice: set capture callback failed: %d\n",
                (int)status);
        FAIL_START();
    }

    /* Register render callback on bus 0 */
    AURenderCallbackStruct output_cb = {
        .inputProc       = render_callback,
        .inputProcRefCon = engine,
    };
    status = AudioUnitSetProperty(engine->voice_unit,
                                  kAudioUnitProperty_SetRenderCallback,
                                  kAudioUnitScope_Input, 0,
                                  &output_cb, sizeof(output_cb));
    if (status != noErr) {
        fprintf(stderr, "pocket_voice: set render callback failed: %d\n",
                (int)status);
        FAIL_START();
    }

    /* Initialize and start */
    status = AudioUnitInitialize(engine->voice_unit);
    if (status != noErr) {
        fprintf(stderr, "pocket_voice: AudioUnitInitialize failed: %d\n",
                (int)status);
        FAIL_START();
    }

    status = AudioOutputUnitStart(engine->voice_unit);
    if (status != noErr) {
        fprintf(stderr, "pocket_voice: AudioOutputUnitStart failed: %d\n",
                (int)status);
        AudioUnitUninitialize(engine->voice_unit);
        FAIL_START();
    }
#undef FAIL_START

    atomic_store(&engine->running, 1);
    return 0;
}

void voice_engine_stop(VoiceEngine *engine) {
    if (!atomic_load(&engine->running)) return;
    AudioOutputUnitStop(engine->voice_unit);
    AudioUnitUninitialize(engine->voice_unit);
    AudioComponentInstanceDispose(engine->voice_unit);
    engine->voice_unit = NULL;
    atomic_store(&engine->running, 0);
}

void voice_engine_destroy(VoiceEngine *engine) {
    if (!engine) return;
    if (atomic_load(&engine->running)) voice_engine_stop(engine);
    ring_destroy(&engine->capture_ring);
    ring_destroy(&engine->playback_ring);
    free(engine->resample_buf);
    free(engine);
}

/* Read captured mic audio (called from C orchestrator or Python) */
int voice_engine_read_capture(VoiceEngine *engine, float *buffer,
                               int max_frames) {
    uint32_t avail = ring_available_read(&engine->capture_ring);
    uint32_t to_read = (uint32_t)max_frames < avail ? (uint32_t)max_frames : avail;
    if (to_read == 0) return 0;
    ring_read(&engine->capture_ring, buffer, to_read);
    return (int)to_read;
}

/* Write TTS audio for playback (called from C orchestrator or Python) */
int voice_engine_write_playback(VoiceEngine *engine, const float *buffer,
                                 int num_frames) {
    atomic_store_explicit(&engine->playing, 1, memory_order_release);
    return ring_write(&engine->playback_ring, buffer, (uint32_t)num_frames);
}

/* Flush playback buffer (instant silence on barge-in) */
void voice_engine_flush_playback(VoiceEngine *engine) {
    ring_flush(&engine->playback_ring);
    atomic_store_explicit(&engine->playing, 0, memory_order_release);
}

/* Check if playback ring has data remaining */
int voice_engine_is_playing(VoiceEngine *engine) {
    return ring_available_read(&engine->playback_ring) > 0 ? 1 : 0;
}

/* VAD state query */
int voice_engine_get_vad_state(VoiceEngine *engine) {
    return (int)engine->vad.state;
}

/* Barge-in flag management */
int voice_engine_get_barge_in(VoiceEngine *engine) {
    return atomic_load_explicit(&engine->barge_in, memory_order_acquire);
}

void voice_engine_clear_barge_in(VoiceEngine *engine) {
    atomic_store_explicit(&engine->barge_in, 0, memory_order_release);
}

/* Configure VAD thresholds at runtime */
void voice_engine_set_vad_thresholds(VoiceEngine *engine,
                                      float energy_thresh,
                                      float silence_thresh) {
    engine->vad.energy_threshold  = energy_thresh;
    engine->vad.silence_threshold = silence_thresh;
}

/* Get capture ring fill level (for monitoring) */
int voice_engine_capture_available(VoiceEngine *engine) {
    return (int)ring_available_read(&engine->capture_ring);
}

/* Get playback ring fill level (for monitoring) */
int voice_engine_playback_available(VoiceEngine *engine) {
    return (int)ring_available_read(&engine->playback_ring);
}
