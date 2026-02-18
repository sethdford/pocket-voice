/**
 * audio_converter.c — Hardware-accelerated audio resampling via AudioConverter.
 *
 * Replaces the manual 31-tap FIR resampling in pocket_voice.c with Apple's
 * AudioConverter API. AudioConverter uses Apple's internal SRC (Sample Rate
 * Converter) which leverages hardware acceleration on Apple Silicon and
 * provides artifact-free, polyphase resampling at any ratio.
 *
 * Advantages over manual FIR:
 *   - Handles arbitrary rate conversion (not just 2:1)
 *   - Higher quality (multi-stage polyphase, >120dB stopband)
 *   - Zero configuration (filter design is automatic)
 *   - Supports real-time streaming with minimal latency
 *
 * Build:
 *   cc -O3 -shared -fPIC -arch arm64 \
 *      -framework AudioToolbox -framework CoreFoundation \
 *      -o libaudio_converter.dylib audio_converter.c
 */

#include <AudioToolbox/AudioToolbox.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* -----------------------------------------------------------------------
 * Quality Levels
 * ----------------------------------------------------------------------- */

typedef enum {
    RESAMPLE_MIN       = 0,  /* Fastest, lowest quality */
    RESAMPLE_LOW       = 1,
    RESAMPLE_MEDIUM    = 2,
    RESAMPLE_HIGH      = 3,
    RESAMPLE_MAX       = 4,  /* Highest quality, most CPU */
} ResampleQuality;

/* -----------------------------------------------------------------------
 * Resampler Context
 * ----------------------------------------------------------------------- */

typedef struct {
    AudioConverterRef converter;
    uint32_t src_rate;
    uint32_t dst_rate;
    uint32_t channels;

    /* Input buffer for the AudioConverter callback */
    const float *input_data;
    uint32_t input_frames;
    uint32_t input_frames_consumed;

    int initialized;
} HWResampler;

/* AudioConverter input callback — provides data on demand */
static OSStatus input_callback(
    AudioConverterRef inAudioConverter,
    UInt32 *ioNumberDataPackets,
    AudioBufferList *ioData,
    AudioStreamPacketDescription **outDataPacketDescription,
    void *inUserData
) {
    (void)inAudioConverter;
    (void)outDataPacketDescription;

    HWResampler *ctx = (HWResampler *)inUserData;

    uint32_t available = ctx->input_frames - ctx->input_frames_consumed;
    uint32_t to_provide = *ioNumberDataPackets;
    if (to_provide > available) to_provide = available;

    ioData->mBuffers[0].mData = (void *)(ctx->input_data + ctx->input_frames_consumed * ctx->channels);
    ioData->mBuffers[0].mDataByteSize = to_provide * ctx->channels * sizeof(float);
    ioData->mBuffers[0].mNumberChannels = ctx->channels;

    *ioNumberDataPackets = to_provide;
    ctx->input_frames_consumed += to_provide;

    return noErr;
}

/* -----------------------------------------------------------------------
 * Public API
 * ----------------------------------------------------------------------- */

/**
 * Create a hardware resampler.
 *
 * @param src_rate    Source sample rate (e.g., 24000)
 * @param dst_rate    Destination sample rate (e.g., 48000)
 * @param channels    Number of channels (1 for mono)
 * @param quality     Resampling quality level
 * @return            Resampler context, or NULL on failure
 */
HWResampler *hw_resampler_create(uint32_t src_rate, uint32_t dst_rate,
                                  uint32_t channels, ResampleQuality quality) {
    HWResampler *ctx = (HWResampler *)calloc(1, sizeof(HWResampler));
    if (!ctx) return NULL;

    ctx->src_rate = src_rate;
    ctx->dst_rate = dst_rate;
    ctx->channels = channels;

    AudioStreamBasicDescription src_fmt = {
        .mSampleRate       = (Float64)src_rate,
        .mFormatID         = kAudioFormatLinearPCM,
        .mFormatFlags      = kAudioFormatFlagIsFloat | kAudioFormatFlagIsPacked,
        .mBytesPerPacket   = channels * sizeof(float),
        .mFramesPerPacket  = 1,
        .mBytesPerFrame    = channels * sizeof(float),
        .mChannelsPerFrame = channels,
        .mBitsPerChannel   = 32,
    };

    AudioStreamBasicDescription dst_fmt = src_fmt;
    dst_fmt.mSampleRate = (Float64)dst_rate;

    OSStatus status = AudioConverterNew(&src_fmt, &dst_fmt, &ctx->converter);
    if (status != noErr) {
        fprintf(stderr, "hw_resampler: AudioConverterNew failed: %d\n", (int)status);
        free(ctx);
        return NULL;
    }

    /* Set quality */
    UInt32 q;
    switch (quality) {
        case RESAMPLE_MIN:    q = kAudioConverterQuality_Min; break;
        case RESAMPLE_LOW:    q = kAudioConverterQuality_Low; break;
        case RESAMPLE_MEDIUM: q = kAudioConverterQuality_Medium; break;
        case RESAMPLE_HIGH:   q = kAudioConverterQuality_High; break;
        case RESAMPLE_MAX:    q = kAudioConverterQuality_Max; break;
        default:              q = kAudioConverterQuality_High; break;
    }
    AudioConverterSetProperty(ctx->converter,
                              kAudioConverterSampleRateConverterQuality,
                              sizeof(q), &q);

    /* Set complexity to maximum for best quality */
    UInt32 complexity = kAudioConverterSampleRateConverterComplexity_Mastering;
    AudioConverterSetProperty(ctx->converter,
                              kAudioConverterSampleRateConverterComplexity,
                              sizeof(complexity), &complexity);

    ctx->initialized = 1;
    return ctx;
}

/**
 * Resample audio.
 *
 * @param ctx         Resampler context
 * @param input       Input audio (float32)
 * @param in_frames   Number of input frames
 * @param output      Output buffer (caller-allocated)
 * @param max_out     Maximum output frames
 * @return            Number of output frames written, or <0 on error
 */
int hw_resample(HWResampler *ctx, const float *input, uint32_t in_frames,
                float *output, uint32_t max_out) {
    if (!ctx || !ctx->initialized) return -1;

    ctx->input_data = input;
    ctx->input_frames = in_frames;
    ctx->input_frames_consumed = 0;

    AudioBufferList out_buf;
    out_buf.mNumberBuffers = 1;
    out_buf.mBuffers[0].mNumberChannels = ctx->channels;
    out_buf.mBuffers[0].mDataByteSize = max_out * ctx->channels * sizeof(float);
    out_buf.mBuffers[0].mData = output;

    UInt32 out_frames = max_out;
    OSStatus status = AudioConverterFillComplexBuffer(
        ctx->converter, input_callback, ctx,
        &out_frames, &out_buf, NULL);

    if (status != noErr && status != -1) {
        fprintf(stderr, "hw_resampler: FillComplexBuffer failed: %d\n", (int)status);
        return -1;
    }

    return (int)out_frames;
}

/**
 * Reset the resampler state (for seek/flush).
 */
void hw_resampler_reset(HWResampler *ctx) {
    if (ctx && ctx->converter) {
        AudioConverterReset(ctx->converter);
    }
}

void hw_resampler_destroy(HWResampler *ctx) {
    if (!ctx) return;
    if (ctx->converter) AudioConverterDispose(ctx->converter);
    free(ctx);
}
