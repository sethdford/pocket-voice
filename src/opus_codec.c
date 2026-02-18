/**
 * opus_codec.c — Native Opus encoding/decoding for real-time TTS streaming.
 *
 * Wraps the libopus C library for ultra-low-latency audio compression.
 * Compresses 24kHz mono float32 PCM → Opus at 24-48kbps, reducing
 * WebSocket bandwidth by 10-20x with sub-millisecond encoding latency.
 *
 * Features:
 *   - Zero-copy ring buffer integration with pocket_voice.c
 *   - Configurable bitrate, frame size, and application mode
 *   - VoIP mode for lowest latency, Audio mode for highest quality
 *   - Packet loss concealment (PLC) for robust streaming
 *
 * Build:
 *   # Install libopus first: brew install opus
 *   cc -O3 -shared -fPIC -arch arm64 \
 *      -I/opt/homebrew/include -L/opt/homebrew/lib -lopus \
 *      -o libpocket_opus.dylib opus_codec.c
 *
 * Usage from Python:
 *   from pocket_tts.native.opus import OpusEncoder, OpusDecoder
 */

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Opus header — dynamically loaded to avoid hard dependency */
typedef struct OpusEncoder OpusEncoder_t;
typedef struct OpusDecoder OpusDecoder_t;

/* Function pointers loaded at runtime via dlopen */
typedef OpusEncoder_t *(*opus_encoder_create_fn)(int Fs, int channels, int application, int *error);
typedef int (*opus_encode_float_fn)(OpusEncoder_t *st, const float *pcm, int frame_size, unsigned char *data, int max_data_bytes);
typedef int (*opus_encoder_ctl_fn)(OpusEncoder_t *st, int request, ...);
typedef void (*opus_encoder_destroy_fn)(OpusEncoder_t *st);
typedef OpusDecoder_t *(*opus_decoder_create_fn)(int Fs, int channels, int *error);
typedef int (*opus_decode_float_fn)(OpusDecoder_t *st, const unsigned char *data, int len, float *pcm, int frame_size, int decode_fec);
typedef void (*opus_decoder_destroy_fn)(OpusDecoder_t *st);

/* Opus application modes */
#define OPUS_APPLICATION_VOIP          2048
#define OPUS_APPLICATION_AUDIO         2049
#define OPUS_APPLICATION_RESTRICTED_LD 2051

/* Opus CTL requests */
#define OPUS_SET_BITRATE_REQUEST       4002
#define OPUS_SET_COMPLEXITY_REQUEST     4010
#define OPUS_SET_SIGNAL_REQUEST         4024
#define OPUS_SIGNAL_VOICE              3001
#define OPUS_SIGNAL_MUSIC              3002

/* -----------------------------------------------------------------------
 * Encoder Context
 * ----------------------------------------------------------------------- */

#define MAX_OPUS_FRAME  5760  /* 120ms at 48kHz */
#define MAX_PACKET_SIZE 4000

typedef struct {
    void *lib_handle;
    OpusEncoder_t *enc;
    OpusDecoder_t *dec;

    int sample_rate;
    int channels;
    int frame_size;     /* samples per channel per frame */
    int bitrate;

    /* Encoding buffer for accumulating samples */
    float pcm_buffer[MAX_OPUS_FRAME];
    int pcm_pos;

    /* Packet output buffer */
    unsigned char packet[MAX_PACKET_SIZE];

    /* Function pointers */
    opus_encoder_create_fn  fn_enc_create;
    opus_encode_float_fn    fn_enc_float;
    opus_encoder_ctl_fn     fn_enc_ctl;
    opus_encoder_destroy_fn fn_enc_destroy;
    opus_decoder_create_fn  fn_dec_create;
    opus_decode_float_fn    fn_dec_float;
    opus_decoder_destroy_fn fn_dec_destroy;

    int initialized;
} PocketOpus;

/* -----------------------------------------------------------------------
 * Dynamic Loading
 * ----------------------------------------------------------------------- */

#include <dlfcn.h>

static int load_opus_lib(PocketOpus *ctx) {
    /* Try multiple paths */
    const char *paths[] = {
        "libopus.dylib",
        "/opt/homebrew/lib/libopus.dylib",
        "/usr/local/lib/libopus.dylib",
        NULL,
    };

    for (int i = 0; paths[i]; i++) {
        ctx->lib_handle = dlopen(paths[i], RTLD_LAZY);
        if (ctx->lib_handle) break;
    }
    if (!ctx->lib_handle) return -1;

    ctx->fn_enc_create  = (opus_encoder_create_fn)dlsym(ctx->lib_handle, "opus_encoder_create");
    ctx->fn_enc_float   = (opus_encode_float_fn)dlsym(ctx->lib_handle, "opus_encode_float");
    ctx->fn_enc_ctl     = (opus_encoder_ctl_fn)dlsym(ctx->lib_handle, "opus_encoder_ctl");
    ctx->fn_enc_destroy = (opus_encoder_destroy_fn)dlsym(ctx->lib_handle, "opus_encoder_destroy");
    ctx->fn_dec_create  = (opus_decoder_create_fn)dlsym(ctx->lib_handle, "opus_decoder_create");
    ctx->fn_dec_float   = (opus_decode_float_fn)dlsym(ctx->lib_handle, "opus_decode_float");
    ctx->fn_dec_destroy = (opus_decoder_destroy_fn)dlsym(ctx->lib_handle, "opus_decoder_destroy");

    if (!ctx->fn_enc_create || !ctx->fn_enc_float || !ctx->fn_enc_destroy) return -2;
    if (!ctx->fn_dec_create || !ctx->fn_dec_float || !ctx->fn_dec_destroy) return -3;

    return 0;
}

/* -----------------------------------------------------------------------
 * Public API
 * ----------------------------------------------------------------------- */

/**
 * Create an Opus encoder/decoder context.
 *
 * @param sample_rate  24000 or 48000
 * @param channels     1 (mono) or 2 (stereo)
 * @param bitrate      Target bitrate in bps (e.g., 24000 for 24kbps)
 * @param frame_ms     Frame duration in ms (2.5, 5, 10, 20, 40, 60)
 * @param application  0=VOIP, 1=Audio, 2=RestrictedLowDelay
 * @return             Context pointer, or NULL on failure
 */
PocketOpus *pocket_opus_create(int sample_rate, int channels, int bitrate,
                               float frame_ms, int application) {
    PocketOpus *ctx = (PocketOpus *)calloc(1, sizeof(PocketOpus));
    if (!ctx) return NULL;

    if (load_opus_lib(ctx) != 0) {
        fprintf(stderr, "pocket_opus: failed to load libopus\n");
        free(ctx);
        return NULL;
    }

    ctx->sample_rate = sample_rate;
    ctx->channels = channels;
    ctx->bitrate = bitrate;
    ctx->frame_size = (int)(sample_rate * frame_ms / 1000.0f);

    int app_mode;
    switch (application) {
        case 0:  app_mode = OPUS_APPLICATION_VOIP; break;
        case 1:  app_mode = OPUS_APPLICATION_AUDIO; break;
        case 2:  app_mode = OPUS_APPLICATION_RESTRICTED_LD; break;
        default: app_mode = OPUS_APPLICATION_AUDIO; break;
    }

    int err;
    ctx->enc = ctx->fn_enc_create(sample_rate, channels, app_mode, &err);
    if (err != 0 || !ctx->enc) {
        fprintf(stderr, "pocket_opus: encoder create failed (err=%d)\n", err);
        dlclose(ctx->lib_handle);
        free(ctx);
        return NULL;
    }

    ctx->fn_enc_ctl(ctx->enc, OPUS_SET_BITRATE_REQUEST, bitrate);
    ctx->fn_enc_ctl(ctx->enc, OPUS_SET_COMPLEXITY_REQUEST, 10);
    ctx->fn_enc_ctl(ctx->enc, OPUS_SET_SIGNAL_REQUEST, OPUS_SIGNAL_VOICE);

    ctx->dec = ctx->fn_dec_create(sample_rate, channels, &err);
    if (err != 0 || !ctx->dec) {
        fprintf(stderr, "pocket_opus: decoder create failed (err=%d)\n", err);
        ctx->fn_enc_destroy(ctx->enc);
        dlclose(ctx->lib_handle);
        free(ctx);
        return NULL;
    }

    ctx->initialized = 1;
    return ctx;
}

/**
 * Encode float32 PCM samples to Opus.
 *
 * Accumulates samples internally and encodes when a full frame is ready.
 *
 * @param ctx           Encoder context
 * @param pcm           Input float32 samples [-1, 1]
 * @param n_samples     Number of samples
 * @param opus_out      Output buffer for encoded Opus packet
 * @param max_out       Maximum output size
 * @return              Bytes written to opus_out (0 if frame not ready, <0 on error)
 */
int pocket_opus_encode(PocketOpus *ctx, const float *pcm, int n_samples,
                       unsigned char *opus_out, int max_out) {
    if (!ctx || !ctx->initialized) return -1;

    int total_encoded = 0;
    int remaining = n_samples;
    const float *src = pcm;

    while (remaining > 0) {
        int space = ctx->frame_size - ctx->pcm_pos;
        int to_copy = remaining < space ? remaining : space;

        memcpy(ctx->pcm_buffer + ctx->pcm_pos, src, to_copy * sizeof(float));
        ctx->pcm_pos += to_copy;
        src += to_copy;
        remaining -= to_copy;

        if (ctx->pcm_pos >= ctx->frame_size) {
            int nbytes = ctx->fn_enc_float(ctx->enc, ctx->pcm_buffer,
                                           ctx->frame_size, opus_out + total_encoded,
                                           max_out - total_encoded);
            if (nbytes < 0) return nbytes;
            total_encoded += nbytes;
            ctx->pcm_pos = 0;
        }
    }

    return total_encoded;
}

/**
 * Flush any remaining samples (pad with silence).
 */
int pocket_opus_flush(PocketOpus *ctx, unsigned char *opus_out, int max_out) {
    if (!ctx || !ctx->initialized || ctx->pcm_pos == 0) return 0;

    memset(ctx->pcm_buffer + ctx->pcm_pos, 0,
           (ctx->frame_size - ctx->pcm_pos) * sizeof(float));

    int nbytes = ctx->fn_enc_float(ctx->enc, ctx->pcm_buffer,
                                   ctx->frame_size, opus_out, max_out);
    ctx->pcm_pos = 0;
    return nbytes < 0 ? nbytes : nbytes;
}

/**
 * Decode an Opus packet to float32 PCM.
 *
 * @param ctx           Decoder context
 * @param opus_data     Encoded Opus packet (NULL for PLC)
 * @param opus_len      Packet length in bytes (0 for PLC)
 * @param pcm_out       Output float32 buffer
 * @param max_samples   Maximum samples to decode
 * @return              Number of samples decoded, or <0 on error
 */
int pocket_opus_decode(PocketOpus *ctx, const unsigned char *opus_data,
                       int opus_len, float *pcm_out, int max_samples) {
    if (!ctx || !ctx->initialized) return -1;

    return ctx->fn_dec_float(ctx->dec, opus_data, opus_len,
                             pcm_out, max_samples, 0);
}

/**
 * Get the frame size in samples.
 */
int pocket_opus_frame_size(PocketOpus *ctx) {
    return ctx ? ctx->frame_size : 0;
}

void pocket_opus_destroy(PocketOpus *ctx) {
    if (!ctx) return;
    if (ctx->enc) ctx->fn_enc_destroy(ctx->enc);
    if (ctx->dec) ctx->fn_dec_destroy(ctx->dec);
    if (ctx->lib_handle) dlclose(ctx->lib_handle);
    free(ctx);
}
