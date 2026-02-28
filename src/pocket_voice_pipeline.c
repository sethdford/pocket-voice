/**
 * pocket_voice_pipeline.c — Zero-Python voice pipeline for Apple Silicon.
 *
 * Standalone C program that links against:
 *   - libpocket_voice.dylib  (CoreAudio I/O, ring buffers, VAD)
 *   - libpocket_stt.dylib    (Kyutai STT 1B, candle+Metal)
 *   - libsonata_lm.dylib     (Sonata LM 241M, candle+Metal)
 *   - libsonata_flow.dylib   (Sonata Flow 35.7M, candle+Metal)
 *   - libcurl                (Claude Messages API SSE)
 *
 * State machine:
 *   Listening → Recording → Processing → Streaming → Speaking → Listening
 *   Any speaking/streaming state can transition to Listening on barge-in.
 *
 * Build:
 *   make pocket-voice
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <signal.h>
#include <stdatomic.h>
#include <unistd.h>
#include <time.h>
#include <stdbool.h>
#include <math.h>
#include <pthread.h>
#include <Accelerate/Accelerate.h>
#include <mach/mach_time.h>
#include <curl/curl.h>
#include "cJSON.h"
#include "sentence_buffer.h"
#include "ssml_parser.h"
#include "text_normalize.h"
#include "prosody_predict.h"
#include "prosody_log.h"
#include "emphasis_predict.h"
#include "breath_synthesis.h"
#include "audio_watermark.h"
#include "lufs.h"
#include "noise_gate.h"
#include "deep_filter.h"
#include "arena.h"
#include "spmc_ring.h"
#include "speech_detector.h"
#include "semantic_eou.h"
#include "conformer_stt.h"
#include "latency_profiler.h"
#include "bnns_conformer.h"
#include "metal_loader.h"
#include "apple_perf.h"
#include "web_remote.h"
#include "http_api.h"
#include "backchannel.h"
#include "audio_emotion.h"

/* ═══════════════════════════════════════════════════════════════════════════
 * FFI declarations for the three native libraries
 * ═══════════════════════════════════════════════════════════════════════════ */

/* --- pocket_voice.c (audio engine) --- */
typedef struct VoiceEngine VoiceEngine;
extern VoiceEngine* voice_engine_create(unsigned int sample_rate, unsigned int buffer_frames);
extern int  voice_engine_start(VoiceEngine *engine);
extern int  voice_engine_start_output_only(VoiceEngine *engine);
extern void voice_engine_stop(VoiceEngine *engine);
extern void voice_engine_destroy(VoiceEngine *engine);
extern int  voice_engine_read_capture(VoiceEngine *engine, float *buffer, int max_frames);
extern int  voice_engine_write_capture(VoiceEngine *engine, const float *buffer, int num_frames);
extern int  voice_engine_write_playback(VoiceEngine *engine, const float *buffer, int num_frames);
extern void voice_engine_flush_playback(VoiceEngine *engine);
extern int  voice_engine_is_playing(VoiceEngine *engine);
extern int  voice_engine_get_vad_state(VoiceEngine *engine);
extern int  voice_engine_get_barge_in(VoiceEngine *engine);
extern void voice_engine_clear_barge_in(VoiceEngine *engine);
extern void voice_engine_set_vad_thresholds(VoiceEngine *engine,
                                             float energy_thresh,
                                             float silence_thresh);
extern void voice_engine_get_drop_counts(VoiceEngine *engine,
                                          uint64_t *capture_out,
                                          uint64_t *playback_out);
extern void voice_engine_resample_48_to_24(const float *in, float *out, int in_len);
extern void voice_engine_resample_24_to_48(const float *in, float *out, int in_len);
extern int  voice_engine_capture_available(VoiceEngine *engine);

/* --- pocket_stt (Rust cdylib) --- */
extern void *pocket_stt_create(const char *hf_repo, const char *model_path, int enable_vad);
extern void  pocket_stt_destroy(void *engine);
extern int   pocket_stt_process_frame(void *engine, const float *pcm, int num_samples);
extern int   pocket_stt_flush(void *engine);
extern int   pocket_stt_get_all_text(void *engine, char *buf, int buf_size);
extern float pocket_stt_get_vad_prob(void *engine, int horizon);
extern int   pocket_stt_has_vad(void *engine);
extern int   pocket_stt_reset(void *engine);
extern int   pocket_stt_frame_size(void);
extern int   pocket_stt_sample_rate(void);

/* --- audio_converter.c (hardware-accelerated resampling) --- */
typedef struct HWResampler HWResampler;
extern HWResampler *hw_resampler_create(uint32_t src_rate, uint32_t dst_rate,
                                         uint32_t channels, int quality);
extern int   hw_resample(HWResampler *ctx, const float *input, uint32_t in_frames,
                          float *output, uint32_t max_out);
extern void  hw_resampler_reset(HWResampler *ctx);
extern void  hw_resampler_destroy(HWResampler *ctx);

/* --- vdsp_prosody.c (AMX-accelerated audio post-processing) --- */
extern int   prosody_pitch_shift(const float *input, float *output, int n_samples,
                                  float pitch_factor, int fft_size);
typedef struct PitchShiftContext PitchShiftContext;
extern PitchShiftContext *prosody_pitch_create(int fft_size);
extern void prosody_pitch_destroy(PitchShiftContext *psc);
extern int prosody_pitch_shift_ctx(PitchShiftContext *psc, const float *input,
                                    float *output, int n_samples, float pitch_factor);
typedef struct BiquadCascade BiquadCascade;
extern BiquadCascade *prosody_create_formant_eq(float pitch_factor, int sample_rate);
extern int   prosody_apply_biquad(BiquadCascade *bc, float *audio, int n_samples);
extern void  prosody_destroy_biquad(BiquadCascade *bc);
extern void  prosody_soft_limit(float *audio, int n_samples,
                                 float threshold, float knee_db);
extern void  prosody_volume(float *audio, int n_samples, float volume_db,
                             float fade_ms, int sample_rate);
extern int   prosody_time_stretch(const float *input, int in_len, float *output,
                                   float rate_factor, float window_ms, int sample_rate);

/* --- spatial_audio.c (binaural 3D HRTF) --- */
typedef struct SpatialAudioEngine SpatialAudioEngine;
extern SpatialAudioEngine *spatial_create(uint32_t sample_rate);
extern int   spatial_set_position(SpatialAudioEngine *engine, int source_idx,
                                   float azimuth, float elevation, float distance);
extern int   spatial_process(SpatialAudioEngine *engine, int source_idx,
                              const float *mono_input,
                              float *left_output, float *right_output, int n_samples);
extern void  spatial_destroy(SpatialAudioEngine *engine);

/* --- pocket_llm (on-device Llama, Rust cdylib) --- */
extern void *pocket_llm_create(const char *model_id, const char *tokenizer_path);
extern void  pocket_llm_destroy(void *engine);
extern int   pocket_llm_set_prompt(void *engine, const char *system, const char *user);
extern int   pocket_llm_step(void *engine);
extern int   pocket_llm_get_token(void *engine, char *buf, int buf_size);
extern int   pocket_llm_is_done(void *engine);
extern int   pocket_llm_set_temperature(void *engine, float temp);
extern int   pocket_llm_reset(void *engine);
extern int   pocket_llm_clear_context(void *engine);

/* --- spm_tokenizer.c (Pure C SentencePiece) --- */
typedef struct SPMTokenizer SPMTokenizer;
extern SPMTokenizer *spm_create(const uint8_t *model_data, uint32_t model_size);
extern void  spm_destroy(SPMTokenizer *tok);
extern int   spm_encode(const SPMTokenizer *tok, const char *text,
                         int32_t *out_ids, int max_ids);
extern int   spm_vocab_size(const SPMTokenizer *tok);

/* --- Phonemizer (espeak-ng IPA) --- */
typedef struct Phonemizer Phonemizer;
extern Phonemizer *phonemizer_create(const char *language);
extern void phonemizer_destroy(Phonemizer *ph);
extern int phonemizer_text_to_ipa(Phonemizer *ph, const char *text, char *ipa_out, int max_len);
extern int phonemizer_load_phoneme_map(Phonemizer *ph, const char *json_path);
extern int phonemizer_ipa_to_ids(Phonemizer *ph, const char *ipa, int *ids_out, int max_ids);
extern int phonemizer_text_to_ids(Phonemizer *ph, const char *text, int *ids_out, int max_ids);
extern int phonemizer_vocab_size(const Phonemizer *ph);

/* --- Pronunciation dictionary --- */
typedef struct PronunciationDict PronunciationDict;
extern PronunciationDict *pronunciation_dict_load(const char *json_path);
extern void pronunciation_dict_destroy(PronunciationDict *dict);
extern int pronunciation_dict_apply(const PronunciationDict *dict, const char *text,
                                     char *out, int out_cap);

/* --- Speaker encoder (ONNX) --- */
typedef struct SpeakerEncoder SpeakerEncoder;
extern SpeakerEncoder *speaker_encoder_create(const char *model_path);
extern void speaker_encoder_destroy(SpeakerEncoder *enc);
extern int speaker_encoder_extract(SpeakerEncoder *enc, const float *audio, int n_samples, float *embedding_out);
extern int speaker_encoder_embedding_dim(const SpeakerEncoder *enc);
extern int speaker_encoder_extract_from_wav(SpeakerEncoder *enc, const char *wav_path, float *embedding_out);

/* --- Speaker encoder (native Rust/ECAPA-TDNN) --- */
typedef void* SpeakerEncoderNative;
extern SpeakerEncoderNative speaker_encoder_native_create(const char *weights_path, const char *config_path);
extern void speaker_encoder_native_destroy(SpeakerEncoderNative encoder);
extern int speaker_encoder_native_embedding_dim(SpeakerEncoderNative encoder);
extern int speaker_encoder_native_sample_rate(SpeakerEncoderNative encoder);
extern int speaker_encoder_native_encode(SpeakerEncoderNative encoder, const float *mel_data,
                                        int n_frames, int n_mels, float *out);
extern int speaker_encoder_native_encode_audio(SpeakerEncoderNative encoder, const float *pcm,
                                              int n_samples, int sample_rate, float *out);

/* --- Speaker diarizer --- */
typedef struct SpeakerDiarizer SpeakerDiarizer;
extern SpeakerDiarizer *diarizer_create(const char *encoder_path, float threshold, int max_speakers);
extern void diarizer_destroy(SpeakerDiarizer *d);
extern int diarizer_identify(SpeakerDiarizer *d, const float *audio, int n_samples);
extern int diarizer_speaker_count(const SpeakerDiarizer *d);
extern const char *diarizer_get_label(const SpeakerDiarizer *d, int speaker_id);
extern int diarizer_set_label(SpeakerDiarizer *d, int speaker_id, const char *label);

/* --- Conversation memory --- */
typedef struct ConversationMemory ConversationMemory;
extern ConversationMemory *memory_create(const char *path, int max_turns, int max_tokens);
extern void memory_destroy(ConversationMemory *mem);
extern int memory_add_turn(ConversationMemory *mem, const char *role, const char *content);
extern char *memory_format_context(ConversationMemory *mem);
extern int memory_turn_count(const ConversationMemory *mem);

/* --- ctc_beam_decoder (forward declaration) --- */
typedef struct CTCBeamDecoder CTCBeamDecoder;
extern void ctc_beam_destroy(CTCBeamDecoder *dec);

/* --- sonata_stt.c (Sonata CTC STT on codec encoder) --- */
#include "sonata_stt.h"
/* --- sonata_refiner.h (Pass 2: semantic tokens → text) --- */
#include "sonata_refiner.h"

/* --- opus_codec.c (Opus encoding/decoding) --- */
typedef struct PocketOpus PocketOpus;
extern PocketOpus *pocket_opus_create(int sample_rate, int channels, int bitrate,
                                       float frame_ms, int application);
extern int   pocket_opus_encode(PocketOpus *ctx, const float *pcm, int n_samples,
                                 unsigned char *opus_out, int max_out);
extern int   pocket_opus_flush(PocketOpus *ctx, unsigned char *opus_out, int max_out);
extern int   pocket_opus_frame_size(PocketOpus *ctx);
extern void  pocket_opus_destroy(PocketOpus *ctx);

/* ═══════════════════════════════════════════════════════════════════════════
 * STT Engine Abstraction
 *
 * Allows the pipeline to use either the Rust (Kyutai) or C (Conformer) STT
 * engine via a uniform function-pointer interface.
 * ═══════════════════════════════════════════════════════════════════════════ */

typedef enum { STT_ENGINE_RUST, STT_ENGINE_CONFORMER, STT_ENGINE_BNNS, STT_ENGINE_SONATA } SttEngineType;

typedef struct {
    void           *engine;
    SttEngineType   type;
    int             sample_rate;    /* Expected input sample rate */
    int             frame_size;     /* Samples per process call */

    int   (*process_frame)(void *engine, const float *pcm, int n_samples);
    int   (*flush)(void *engine);
    int   (*get_text)(void *engine, char *buf, int buf_size);
    /**
     * Optional: get word-level timestamps from last flush.
     * out: array of WordTimestamp (from http_api.h). Returns count or -1.
     * NULL for engines that do not support timestamps (e.g. beam search).
     */
    int   (*get_words)(void *engine, WordTimestamp *out, int max_words);
    float (*get_vad_prob)(void *engine, int horizon);
    int   (*has_vad)(void *engine);
    int   (*reset)(void *engine);  /* Returns 0 on success, -1 on failure */
    void  (*destroy)(void *engine);
} SttInterface;

/* --- Conformer STT wrappers (adapt ConformerSTT* to void* interface) --- */

static int cstt_process_frame(void *engine, const float *pcm, int n_samples) {
    return conformer_stt_process((ConformerSTT *)engine, pcm, n_samples);
}
static int cstt_flush(void *engine) {
    return conformer_stt_flush((ConformerSTT *)engine);
}
static int cstt_get_text(void *engine, char *buf, int buf_size) {
    return conformer_stt_get_text((const ConformerSTT *)engine, buf, buf_size);
}
static float cstt_get_vad_prob(void *engine, int horizon) {
    return conformer_stt_eou_prob((const ConformerSTT *)engine, horizon);
}
static int cstt_has_vad(void *engine) {
    return conformer_stt_has_eou_support((const ConformerSTT *)engine);
}
static int cstt_reset(void *engine) {
    conformer_stt_reset((ConformerSTT *)engine);
    return 0;
}
static void cstt_destroy(void *engine) {
    conformer_stt_destroy((ConformerSTT *)engine);
}

/* --- Rust STT wrappers (already void*, just wrap for interface) --- */

static int rstt_process_frame(void *engine, const float *pcm, int n_samples) {
    return pocket_stt_process_frame(engine, pcm, n_samples);
}
static int rstt_flush(void *engine) {
    return pocket_stt_flush(engine);
}
static int rstt_get_text(void *engine, char *buf, int buf_size) {
    return pocket_stt_get_all_text(engine, buf, buf_size);
}
static float rstt_get_vad_prob(void *engine, int horizon) {
    return pocket_stt_get_vad_prob(engine, horizon);
}
static int rstt_has_vad(void *engine) {
    return pocket_stt_has_vad(engine);
}
static int rstt_reset(void *engine) {
    return pocket_stt_reset(engine);
}
static void rstt_destroy(void *engine) {
    pocket_stt_destroy(engine);
}

static SttInterface stt_create_rust(const char *repo, const char *model, int enable_vad) {
    SttInterface iface = {0};
    iface.type = STT_ENGINE_RUST;
    iface.engine = pocket_stt_create(repo, model, enable_vad);
    if (!iface.engine) {
        fprintf(stderr, "[stt] pocket_stt_create failed\n");
        return iface;
    }
    iface.sample_rate = pocket_stt_sample_rate();
    iface.frame_size = pocket_stt_frame_size();
    iface.process_frame = rstt_process_frame;
    iface.flush = rstt_flush;
    iface.get_text = rstt_get_text;
    iface.get_vad_prob = rstt_get_vad_prob;
    iface.has_vad = rstt_has_vad;
    iface.reset = rstt_reset;
    iface.destroy = rstt_destroy;
    return iface;
}

static SttInterface stt_create_conformer(const char *model_path) {
    SttInterface iface = {0};
    iface.type = STT_ENGINE_CONFORMER;
    ConformerSTT *cstt = conformer_stt_create(model_path);
    if (!cstt) {
        fprintf(stderr, "[stt] conformer_stt_create failed\n");
        return iface;
    }
    iface.engine = cstt;
    iface.sample_rate = conformer_stt_sample_rate(cstt);
    iface.frame_size = (iface.sample_rate / 1000) * 80; /* 80ms */
    iface.process_frame = cstt_process_frame;
    iface.flush = cstt_flush;
    iface.get_text = cstt_get_text;
    iface.get_vad_prob = cstt_get_vad_prob;
    iface.has_vad = cstt_has_vad;
    iface.reset = cstt_reset;
    iface.destroy = cstt_destroy;
    return iface;
}

/* --- BNNS Conformer STT wrappers (CoreML/ANE accelerated) --- */

typedef struct {
    BNNSConformer *bc;
    ConformerSTT  *cstt;  /* Fallback for mel extraction, VAD, text buffer */
} BNNSSttEngine;

static int bnns_stt_process_frame(void *engine, const float *pcm, int n_samples) {
    BNNSSttEngine *be = (BNNSSttEngine *)engine;
    return conformer_stt_process(be->cstt, pcm, n_samples);
}
static int bnns_stt_flush(void *engine) {
    BNNSSttEngine *be = (BNNSSttEngine *)engine;
    return conformer_stt_flush(be->cstt);
}
static int bnns_stt_get_text(void *engine, char *buf, int buf_size) {
    BNNSSttEngine *be = (BNNSSttEngine *)engine;
    return conformer_stt_get_text(be->cstt, buf, buf_size);
}
static float bnns_stt_get_vad_prob(void *engine, int horizon) {
    BNNSSttEngine *be = (BNNSSttEngine *)engine;
    return conformer_stt_eou_prob(be->cstt, horizon);
}
static int bnns_stt_has_vad(void *engine) {
    BNNSSttEngine *be = (BNNSSttEngine *)engine;
    return conformer_stt_has_eou_support(be->cstt);
}
static int bnns_stt_reset(void *engine) {
    BNNSSttEngine *be = (BNNSSttEngine *)engine;
    conformer_stt_reset(be->cstt);
    return 0;
}
static void bnns_stt_destroy(void *engine) {
    BNNSSttEngine *be = (BNNSSttEngine *)engine;
    if (be->bc) bnns_conformer_destroy(be->bc);
    if (be->cstt) conformer_stt_destroy(be->cstt);
    free(be);
}

static int bnns_external_forward(void *user_ctx,
    const float *mel_in, int T, int n_mels,
    float *logits_out, int max_T_sub) {
    BNNSConformer *bc = (BNNSConformer *)user_ctx;
    return bnns_conformer_forward(bc, mel_in, T, n_mels, logits_out, max_T_sub);
}

static SttInterface stt_create_bnns(const char *cstt_model, const char *mlmodelc_path) {
    SttInterface iface = {0};
    iface.type = STT_ENGINE_BNNS;

    BNNSSttEngine *be = (BNNSSttEngine *)calloc(1, sizeof(BNNSSttEngine));
    if (!be) return iface;

    be->cstt = conformer_stt_create(cstt_model);
    if (!be->cstt) {
        fprintf(stderr, "[bnns_stt] Failed to create conformer fallback\n");
        free(be);
        return iface;
    }

    if (bnns_conformer_available()) {
        int d_model = conformer_stt_d_model(be->cstt);
        int n_layers = conformer_stt_n_layers(be->cstt);
        int vocab = conformer_stt_vocab_size(be->cstt);
        be->bc = bnns_conformer_create(n_layers, d_model, 8, 4, 9, vocab);
        if (be->bc && mlmodelc_path) {
            if (bnns_conformer_load_mlmodelc(be->bc, mlmodelc_path) != 0) {
                fprintf(stderr, "[bnns_stt] Failed to load mlmodelc, using CPU fallback\n");
                bnns_conformer_destroy(be->bc);
                be->bc = NULL;
            } else {
                fprintf(stderr, "[bnns_stt] ANE-accelerated conformer loaded\n");
                conformer_stt_set_external_forward(be->cstt,
                    bnns_external_forward, be->bc);
            }
        }
    } else {
        fprintf(stderr, "[bnns_stt] BNNS Graph unavailable (macOS 15+ required), using CPU\n");
    }

    iface.engine = be;
    iface.sample_rate = conformer_stt_sample_rate(be->cstt);
    iface.frame_size = (iface.sample_rate / 1000) * 80;
    iface.process_frame = bnns_stt_process_frame;
    iface.flush = bnns_stt_flush;
    iface.get_text = bnns_stt_get_text;
    iface.get_vad_prob = bnns_stt_get_vad_prob;
    iface.has_vad = bnns_stt_has_vad;
    iface.reset = bnns_stt_reset;
    iface.destroy = bnns_stt_destroy;
    return iface;
}

/* --- Sonata STT wrappers (CTC on Sonata Codec encoder) --- */
/* Uses the streaming API for growing-window transcription + beam search + EOU.
 * Optional refiner: sonata_refiner_process() takes semantic token IDs (32768 vocab
 * from codec FSQ), not CTC text. The Sonata STT encoder → CTC projection produces
 * character logits, not semantic tokens. To wire the refiner:
 *   1. Add sonata_stt_get_semantic_tokens(stt, ids, max) that returns encoder
 *      output quantized via FSQ (requires dual-head export: CTC + FSQ).
 *   2. After flush, if refiner && n_sem > 0: call sonata_refiner_process();
 *      use refined text as final transcript.
 * Until semantic tokens are exposed, the refiner is loaded but unused. */

typedef struct {
    SonataSTT *stt;
    SonataRefiner *refiner;  /* Optional; used when semantic tokens available */
    CTCBeamDecoder *beam;
    char   text_buf[4096];
    int    streaming;
} SonataSTTEngine;

static int sonata_stt_process_frame_wrapper(void *engine, const float *pcm, int n_samples) {
    SonataSTTEngine *se = (SonataSTTEngine *)engine;
    if (!se || !pcm || n_samples <= 0) return -1;
    if (!se->streaming) {
        sonata_stt_stream_start(se->stt, 30.0f);
        se->streaming = 1;
    }
    return sonata_stt_stream_feed(se->stt, pcm, n_samples);
}
static int sonata_stt_flush_wrapper(void *engine) {
    SonataSTTEngine *se = (SonataSTTEngine *)engine;
    if (!se || !se->stt) return -1;
    se->text_buf[0] = '\0';
    if (se->streaming)
        sonata_stt_stream_flush(se->stt, se->text_buf, (int)sizeof(se->text_buf));

    /* Refiner pass: semantic tokens → refined text. Requires sonata_stt_get_semantic_tokens()
     * (not yet implemented — STT model has CTC head only). When available:
     *   int sem_ids[2048];
     *   int n_sem = sonata_stt_get_semantic_tokens(se->stt, sem_ids, 2048);
     *   if (se->refiner && n_sem > 0) {
     *       char refined[4096];
     *       int rlen = sonata_refiner_process(se->refiner, sem_ids, n_sem,
     *                                          refined, (int)sizeof(refined));
     *       if (rlen > 0) {
     *           strncpy(se->text_buf, refined, sizeof(se->text_buf) - 1);
     *           se->text_buf[sizeof(se->text_buf) - 1] = '\0';
     *       }
     *   }
     */

    return 0;
}
static int sonata_stt_get_text_wrapper(void *engine, char *buf, int buf_size) {
    SonataSTTEngine *se = (SonataSTTEngine *)engine;
    if (!se || !buf || buf_size <= 0) return -1;
    strncpy(buf, se->text_buf, buf_size - 1);
    buf[buf_size - 1] = '\0';
    return 0;
}
/* Word timestamps from CTC alignment (greedy only; beam search has align_len=0). */
static int sonata_stt_get_words_wrapper(void *engine, WordTimestamp *out, int max_words) {
    SonataSTTEngine *se = (SonataSTTEngine *)engine;
    if (!se || !se->stt || !out || max_words <= 0) return -1;
    SonataSTTWord stw[256];
    int n = sonata_stt_get_words(se->stt, stw, max_words < 256 ? max_words : 256);
    if (n <= 0) return n;
    for (int i = 0; i < n && i < max_words; i++) {
        strncpy(out[i].word, stw[i].word, sizeof(out[i].word) - 1);
        out[i].word[sizeof(out[i].word) - 1] = '\0';
        out[i].start_s = stw[i].start_sec;
        out[i].end_s = stw[i].end_sec;
    }
    return n;
}
static float sonata_stt_get_vad_prob_wrapper(void *engine, int horizon) {
    SonataSTTEngine *se = (SonataSTTEngine *)engine;
    if (!se || !se->stt) return 0.0f;
    float peak = sonata_stt_eou_peak(se->stt, horizon > 0 ? horizon : 10);
    return peak >= 0.0f ? peak : 0.0f;
}
static int sonata_stt_has_vad_wrapper(void *engine) {
    (void)engine;
    return 1;
}
static int sonata_stt_reset_wrapper(void *engine) {
    SonataSTTEngine *se = (SonataSTTEngine *)engine;
    if (se) {
        se->text_buf[0] = '\0';
        if (se->streaming) {
            sonata_stt_stream_end(se->stt);
            se->streaming = 0;
        }
        if (se->stt) sonata_stt_reset(se->stt);
    }
    return 0;
}
static void sonata_stt_destroy_wrapper(void *engine) {
    SonataSTTEngine *se = (SonataSTTEngine *)engine;
    if (!se) return;
    if (se->streaming) sonata_stt_stream_end(se->stt);
    if (se->beam) ctc_beam_destroy(se->beam);
    if (se->refiner) sonata_refiner_destroy(se->refiner);
    if (se->stt) sonata_stt_destroy(se->stt);
    free(se);
}

static SttInterface stt_create_sonata(const char *weights_path, const char *refiner_path) {
    SttInterface iface = {0};
    if (!weights_path) return iface;

    SonataSTT *stt = sonata_stt_create(weights_path);
    if (!stt) return iface;

    SonataSTTEngine *se = (SonataSTTEngine *)calloc(1, sizeof(SonataSTTEngine));
    if (!se) { sonata_stt_destroy(stt); return iface; }

    se->stt = stt;
    se->refiner = NULL;
    se->beam = NULL;
    se->streaming = 0;

    if (refiner_path) {
        se->refiner = sonata_refiner_create(refiner_path);
        if (se->refiner)
            fprintf(stderr, "[sonata_stt] Refiner loaded (awaiting sonata_stt_get_semantic_tokens for two-pass)\n");
        else
            fprintf(stderr, "[sonata_stt] Warning: refiner load failed, using CTC-only\n");
    }

    iface.type = STT_ENGINE_SONATA;
    iface.engine = se;
    iface.sample_rate = 24000;
    iface.frame_size = 24000 / 100;  /* 10ms chunks */
    iface.process_frame = sonata_stt_process_frame_wrapper;
    iface.flush = sonata_stt_flush_wrapper;
    iface.get_text = sonata_stt_get_text_wrapper;
    iface.get_words = sonata_stt_get_words_wrapper;
    iface.get_vad_prob = sonata_stt_get_vad_prob_wrapper;
    iface.has_vad = sonata_stt_has_vad_wrapper;
    iface.reset = sonata_stt_reset_wrapper;
    iface.destroy = sonata_stt_destroy_wrapper;
    return iface;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * TTS Engine Abstraction
 *
 * Allows the pipeline to use either the Rust (Kyutai) or C (Kyutai DSM) TTS
 * engine via a uniform function-pointer interface.
 * ═══════════════════════════════════════════════════════════════════════════ */

typedef enum { TTS_ENGINE_SONATA, TTS_ENGINE_SONATA_V2, TTS_ENGINE_SONATA_V3 } TtsEngineType;

typedef struct {
    void           *engine;
    TtsEngineType   type;
    int             sample_rate;
    int             frame_size;

    int   (*set_text)(void *engine, const char *text);
    int   (*set_text_ipa)(void *engine, const char *ipa, const char *text);
    int   (*set_text_done)(void *engine);
    int   (*step)(void *engine);
    int   (*get_audio)(void *engine, float *buf, int max);
    int   (*is_done)(void *engine);
    int   (*peek_audio)(void *engine, const float **ptr, int *count);
    int   (*advance_audio)(void *engine, int n);
    int   (*reset)(void *engine);  /* Returns 0 on success, -1 on failure */
    void  (*destroy)(void *engine);
} TtsInterface;

/* --- Sonata Flow v2 (text → mel → iSTFT) TTS wrappers --- */
/* Forward decls and constants needed before full Sonata section below */
typedef struct SonataISTFT SonataISTFT;
#if !defined(SONATA_N_FFT)
#define SONATA_N_FFT 1024
#define SONATA_HOP 480
#define SONATA_N_BINS (SONATA_N_FFT / 2 + 1)
#endif
extern SonataISTFT *sonata_istft_create(int n_fft, int hop_length);
extern void         sonata_istft_destroy(SonataISTFT *dec);
extern void         sonata_istft_reset(SonataISTFT *dec);
extern int          sonata_istft_decode_batch(SonataISTFT *dec,
    const float *magnitudes, const float *phases, int n_frames, float *out_audio);
extern void *sonata_flow_v2_create(const char *weights_path, const char *config_path);
extern void  sonata_flow_v2_destroy(void *engine);
extern int   sonata_flow_v2_generate(void *engine, const char *text,
    const float *ref_mel, int ref_mel_frames, int target_frames,
    float *out_mel, int max_frames);
extern void  sonata_flow_v2_set_cfg_scale(void *engine, float scale);
extern void  sonata_flow_v2_set_n_steps(void *engine, int steps);
extern void  sonata_flow_v2_set_speaker(void *engine, int speaker_id);
extern int   sonata_flow_v2_set_quality_mode(void *engine, int mode);

/* Sonata Flow v3 + Vocoder (Rust FFI) */
extern void *sonata_flow_v3_create(const char *weights, const char *config);
extern void  sonata_flow_v3_destroy(void *engine);
extern int   sonata_flow_v3_set_cfg_scale(void *engine, float scale);
extern int   sonata_flow_v3_set_n_steps(void *engine, int steps);
extern int   sonata_flow_v3_set_speaker(void *engine, int speaker_id);
extern int   sonata_flow_v3_set_solver(void *engine, int use_heun);
extern int   sonata_flow_v3_set_quality_mode(void *engine, int mode);
extern int   sonata_flow_v3_set_reference(void *engine, const float *ref_mel_data,
    int n_frames, int mel_dim);
extern void  sonata_flow_v3_clear_reference(void *engine);
extern int   sonata_flow_v3_set_emotion(void *engine, int emotion_id);
extern int   sonata_flow_v3_generate(void *engine, const char *text, int text_len,
    const int *phoneme_ids, int phoneme_len,
    int target_frames, float *out_mel, int max_frames);
extern int   sonata_flow_v3_get_durations(void *engine, float *out_durations, int max_n);
extern void *sonata_vocoder_create(const char *weights, const char *config);
extern void  sonata_vocoder_destroy(void *engine);
extern int   sonata_vocoder_generate(void *engine, const float *mel, int n_frames,
    int mel_dim, float *out_audio, int max_samples);

#define SONATA_V2_MEL_DIM 80
#define SONATA_V2_MAX_FRAMES 1000

typedef struct {
    void         *flow_v2;
    SonataISTFT  *istft;
    char          text_buf[4096];
    float        *audio_buf;
    int           audio_cap;
    int           audio_len;
    int           audio_pos;
    int           synthesized;  /* 1 = mel converted to audio and ready */
    float        *phase_accum_buf;  /* per-bin phase accumulation for iSTFT continuity */
    float        *mel_buf;          /* pre-allocated mel buffer to avoid malloc on hot path */
    float        *mag_buf;          /* pre-allocated iSTFT magnitude buffer */
    float        *phase_buf;        /* pre-allocated iSTFT phase buffer */
    int           istft_buf_cap;    /* capacity in frames for mag/phase buffers */
} SonataV2Engine;

/* Convert one frame of 80-bin log-mel to 513-bin magnitude + phase for iSTFT.
 * phase_accum is a per-bin accumulator that maintains phase continuity across
 * frames. Pass NULL for first-frame zero-phase behavior (not recommended). */
static void mel_frame_to_mag_phase(const float *mel, int mel_dim,
                                   float *mag, float *phase, int n_bins,
                                   float *phase_accum, int hop_size, int fft_size) {
    (void)mel_dim;  /* assumed 80 */
    for (int b = 0; b < n_bins; b++) {
        float t = (b < n_bins - 1) ? ((float)b / (n_bins - 1)) * 79.0f : 79.0f;
        int i0 = (int)t;
        int i1 = (i0 < 79) ? i0 + 1 : 79;
        float frac = t - i0;
        float mel_val = mel[i0] * (1.0f - frac) + mel[i1] * frac;
        mag[b] = expf(mel_val > 0.0f ? mel_val : 0.0f);
        if (mag[b] < 1e-8f) mag[b] = 1e-8f;
        if (phase_accum) {
            phase_accum[b] += 2.0f * (float)M_PI * (float)b * (float)hop_size / (float)fft_size;
            /* Wrap to [-pi, pi] to avoid numerical drift */
            while (phase_accum[b] > (float)M_PI)  phase_accum[b] -= 2.0f * (float)M_PI;
            while (phase_accum[b] < -(float)M_PI) phase_accum[b] += 2.0f * (float)M_PI;
            phase[b] = phase_accum[b];
        } else {
            phase[b] = 0.0f;
        }
    }
}

static int sonatav2_set_text(void *e, const char *t) {
    SonataV2Engine *ev = (SonataV2Engine *)e;
    if (!ev || !t) return -1;
    size_t n = strlen(t);
    if (n >= sizeof(ev->text_buf) - 1) n = sizeof(ev->text_buf) - 2;
    memcpy(ev->text_buf, t, n + 1);
    ev->text_buf[n] = '\0';
    ev->synthesized = 0;
    return 0;
}

static int sonatav2_set_text_ipa(void *e, const char *ipa, const char *text) {
    (void)ipa;
    return sonatav2_set_text(e, text);
}

static int sonatav2_set_text_done(void *e) {
    SonataV2Engine *ev = (SonataV2Engine *)e;
    if (!ev || !ev->flow_v2 || !ev->istft) return -1;
    if (ev->synthesized) return 0;

    /* Use pre-allocated mel buffer to avoid ~320KB malloc on latency-critical path */
    if (!ev->mel_buf) return -1;

    int n_frames = sonata_flow_v2_generate(ev->flow_v2, ev->text_buf,
                                          NULL, 0, 0,
                                          ev->mel_buf, SONATA_V2_MAX_FRAMES);
    if (n_frames <= 0) {
        return -1;
    }

    int n_bins = SONATA_N_BINS;

    /* Grow pre-allocated mag/phase buffers if needed */
    if (n_frames > ev->istft_buf_cap) {
        int new_cap = n_frames + 64;
        float *new_mag = (float *)malloc((size_t)new_cap * n_bins * sizeof(float));
        float *new_phase = (float *)malloc((size_t)new_cap * n_bins * sizeof(float));
        if (!new_mag || !new_phase) {
            free(new_mag);
            free(new_phase);
            return -1;
        }
        free(ev->mag_buf);
        free(ev->phase_buf);
        ev->mag_buf = new_mag;
        ev->phase_buf = new_phase;
        ev->istft_buf_cap = new_cap;
    }

    /* Reset phase accumulator for this utterance */
    if (ev->phase_accum_buf)
        memset(ev->phase_accum_buf, 0, (size_t)n_bins * sizeof(float));
    for (int f = 0; f < n_frames; f++) {
        mel_frame_to_mag_phase(ev->mel_buf + f * SONATA_V2_MEL_DIM, SONATA_V2_MEL_DIM,
                              ev->mag_buf + f * n_bins, ev->phase_buf + f * n_bins, n_bins,
                              ev->phase_accum_buf, SONATA_HOP, SONATA_N_FFT);
    }

    int total_samples = n_frames * SONATA_HOP;
    if (total_samples > ev->audio_cap) {
        float *narrow = realloc(ev->audio_buf, (size_t)total_samples * sizeof(float));
        if (!narrow) return -1;
        ev->audio_buf = narrow;
        ev->audio_cap = total_samples;
    }

    sonata_istft_reset(ev->istft);
    int written = sonata_istft_decode_batch(ev->istft, ev->mag_buf, ev->phase_buf, n_frames, ev->audio_buf);

    ev->audio_len = written;
    ev->audio_pos = 0;
    ev->synthesized = 1;
    return 0;
}

static int sonatav2_step(void *e) {
    (void)e;
    return 1;
}

static int sonatav2_get_audio(void *e, float *buf, int max) {
    SonataV2Engine *ev = (SonataV2Engine *)e;
    if (!ev || !buf) return -1;
    int avail = ev->audio_len - ev->audio_pos;
    if (avail <= 0) return 0;
    int n = (avail < max) ? avail : max;
    memcpy(buf, ev->audio_buf + ev->audio_pos, (size_t)n * sizeof(float));
    ev->audio_pos += n;
    return n;
}

static int sonatav2_is_done(void *e) {
    SonataV2Engine *ev = (SonataV2Engine *)e;
    if (!ev) return 1;
    return (ev->audio_pos >= ev->audio_len) ? 1 : 0;
}

static int sonatav2_peek_audio(void *e, const float **ptr, int *count) {
    SonataV2Engine *ev = (SonataV2Engine *)e;
    if (!ev || !ptr || !count) return -1;
    *ptr = ev->audio_buf + ev->audio_pos;
    *count = ev->audio_len - ev->audio_pos;
    return 0;
}

static int sonatav2_advance_audio(void *e, int n) {
    SonataV2Engine *ev = (SonataV2Engine *)e;
    if (!ev) return -1;
    ev->audio_pos += n;
    if (ev->audio_pos > ev->audio_len) ev->audio_pos = ev->audio_len;
    return 0;
}

static int sonatav2_reset(void *e) {
    SonataV2Engine *ev = (SonataV2Engine *)e;
    if (!ev) return -1;
    ev->audio_pos = 0;
    ev->audio_len = 0;
    ev->synthesized = 0;
    if (ev->phase_accum_buf)
        memset(ev->phase_accum_buf, 0, SONATA_N_BINS * sizeof(float));
    if (ev->istft) sonata_istft_reset(ev->istft);
    return 0;
}

static void sonatav2_destroy(void *e) {
    SonataV2Engine *ev = (SonataV2Engine *)e;
    if (!ev) return;
    if (ev->flow_v2) sonata_flow_v2_destroy(ev->flow_v2);
    if (ev->istft) sonata_istft_destroy(ev->istft);
    free(ev->audio_buf);
    free(ev->phase_accum_buf);
    free(ev->mel_buf);
    free(ev->mag_buf);
    free(ev->phase_buf);
    free(ev);
}

static TtsInterface tts_create_sonata_v2(const char *weights_path, const char *config_path) {
    TtsInterface iface = {0};
    SonataV2Engine *ev = calloc(1, sizeof(SonataV2Engine));
    if (!ev) return iface;

    ev->flow_v2 = sonata_flow_v2_create(weights_path, config_path);
    if (!ev->flow_v2) {
        free(ev);
        return iface;
    }

    ev->istft = sonata_istft_create(SONATA_N_FFT, SONATA_HOP);
    if (!ev->istft) {
        sonata_flow_v2_destroy(ev->flow_v2);
        free(ev);
        return iface;
    }

    ev->audio_cap = 24000 * 10;  /* 10 seconds */
    ev->audio_buf = malloc((size_t)ev->audio_cap * sizeof(float));
    ev->phase_accum_buf = calloc(SONATA_N_BINS, sizeof(float));
    ev->mel_buf = malloc(SONATA_V2_MAX_FRAMES * SONATA_V2_MEL_DIM * sizeof(float));
    if (!ev->audio_buf || !ev->phase_accum_buf || !ev->mel_buf) {
        free(ev->audio_buf);
        free(ev->phase_accum_buf);
        free(ev->mel_buf);
        sonata_istft_destroy(ev->istft);
        sonata_flow_v2_destroy(ev->flow_v2);
        free(ev);
        return iface;
    }

    iface.type = TTS_ENGINE_SONATA_V2;
    iface.engine = ev;
    iface.sample_rate = 24000;
    iface.frame_size = SONATA_HOP;
    iface.set_text = sonatav2_set_text;
    iface.set_text_ipa = sonatav2_set_text_ipa;
    iface.set_text_done = sonatav2_set_text_done;
    iface.step = sonatav2_step;
    iface.get_audio = sonatav2_get_audio;
    iface.is_done = sonatav2_is_done;
    iface.peek_audio = sonatav2_peek_audio;
    iface.advance_audio = sonatav2_advance_audio;
    iface.reset = sonatav2_reset;
    iface.destroy = sonatav2_destroy;
    return iface;
}

/* --- Sonata Flow v3 + Vocoder (text/phoneme → mel → audio) --- */
#define SONATA_V3_MEL_DIM 80
#define SONATA_V3_MAX_FRAMES 1000

typedef struct {
    void         *flow_v3;
    void         *vocoder;
    Phonemizer   *phonemizer;
    char          text_buf[4096];
    float        *audio_buf;
    int           audio_cap;
    int           audio_len;
    int           audio_pos;
    int           synthesized;
    float        *mel_buf;
} SonataV3Engine;

static int sonatav3_set_text(void *e, const char *t) {
    SonataV3Engine *ev = (SonataV3Engine *)e;
    if (!ev || !t) return -1;
    size_t n = strlen(t);
    if (n >= sizeof(ev->text_buf) - 1) n = sizeof(ev->text_buf) - 2;
    memcpy(ev->text_buf, t, n + 1);
    ev->text_buf[n] = '\0';
    ev->synthesized = 0;
    return 0;
}

static int sonatav3_set_text_ipa(void *e, const char *ipa, const char *text) {
    (void)ipa;
    return sonatav3_set_text(e, text);
}

static int sonatav3_set_text_done(void *e) {
    SonataV3Engine *ev = (SonataV3Engine *)e;
    if (!ev || !ev->flow_v3 || !ev->vocoder || !ev->mel_buf) return -1;
    if (ev->synthesized) return 0;

    int n_frames;
    const char *text = ev->text_buf;
    int text_len = (int)strlen(text);
    const int *phoneme_ids = NULL;
    int phoneme_len = 0;
    int pids[512];

    if (ev->phonemizer && phonemizer_vocab_size(ev->phonemizer) > 0 && text_len > 0) {
        int n = phonemizer_text_to_ids(ev->phonemizer, text, pids, 512);
        if (n > 0) {
            phoneme_ids = pids;
            phoneme_len = n;
            text = NULL;
            text_len = 0;
        }
    }

    n_frames = sonata_flow_v3_generate(ev->flow_v3, text, text_len,
                                       phoneme_ids, phoneme_len,
                                       0, ev->mel_buf, SONATA_V3_MAX_FRAMES);
    if (n_frames <= 0) return -1;

    int total_samples = sonata_vocoder_generate(ev->vocoder,
                                                ev->mel_buf, n_frames,
                                                SONATA_V3_MEL_DIM,
                                                ev->audio_buf, ev->audio_cap);
    if (total_samples <= 0) return -1;

    ev->audio_len = total_samples;
    ev->audio_pos = 0;
    ev->synthesized = 1;
    return 0;
}

static int sonatav3_step(void *e) {
    (void)e;
    return 1;
}

static int sonatav3_get_audio(void *e, float *b, int m) {
    SonataV3Engine *ev = (SonataV3Engine *)e;
    if (!ev || !b) return 0;
    int avail = ev->audio_len - ev->audio_pos;
    if (avail <= 0) return 0;
    if (avail > m) avail = m;
    memcpy(b, ev->audio_buf + ev->audio_pos, (size_t)avail * sizeof(float));
    ev->audio_pos += avail;
    return avail;
}

static int sonatav3_is_done(void *e) {
    SonataV3Engine *ev = (SonataV3Engine *)e;
    return !ev || ev->audio_pos >= ev->audio_len;
}

static int sonatav3_peek_audio(void *e, const float **p, int *c) {
    SonataV3Engine *ev = (SonataV3Engine *)e;
    if (!ev || !p || !c) return -1;
    *p = ev->audio_buf + ev->audio_pos;
    *c = ev->audio_len - ev->audio_pos;
    return 0;
}

static int sonatav3_advance_audio(void *e, int n) {
    SonataV3Engine *ev = (SonataV3Engine *)e;
    if (!ev) return -1;
    ev->audio_pos += n;
    if (ev->audio_pos > ev->audio_len) ev->audio_pos = ev->audio_len;
    return 0;
}

static int sonatav3_reset(void *e) {
    SonataV3Engine *ev = (SonataV3Engine *)e;
    if (!ev) return -1;
    ev->audio_pos = 0;
    ev->audio_len = 0;
    ev->synthesized = 0;
    return 0;
}

/* Map per-phoneme durations to word boundaries. Frame duration = 480/24000 = 20ms. */
#define SONATA_V3_FRAME_DUR_S 0.02f
#define SONATA_V3_MAX_DUR 512

static int sonatav3_get_words(void *e, WordTimestamp *out, int max_words) {
    SonataV3Engine *ev = (SonataV3Engine *)e;
    if (!ev || !out || max_words <= 0 || !ev->flow_v3 || !ev->synthesized) return 0;

    /* Phoneme durations don't map 1:1 to text characters; char-based logic is wrong. */
    if (ev->phonemizer) {
        /* TODO: implement phoneme→word mapping for accurate timestamps with phonemizer */
        return 0;
    }

    float durs[SONATA_V3_MAX_DUR];
    int n_dur = sonata_flow_v3_get_durations(ev->flow_v3, durs, SONATA_V3_MAX_DUR);
    if (n_dur <= 0) return 0;

    const char *text = ev->text_buf;
    int text_len = (int)strlen(text);
    if (text_len <= 0) return 0;

    /* Char-aligned: durations match char count. Split text by spaces. */
    int n_words = 0;
    int dur_pos = 0;  /* current position in duration array (in frames) */

    const char *p = text;
    while (*p && n_words < max_words && n_words < TTS_MAX_WORD_TIMESTAMPS) {
        while (*p == ' ' || *p == '\t' || *p == '\n') {
            if (dur_pos < n_dur) dur_pos++;  /* advance past space duration */
            p++;
        }
        if (!*p) break;
        const char *word_start = p;
        int wlen = 0;
        while (*p && *p != ' ' && *p != '\t' && *p != '\n') { p++; wlen++; }

        float word_start_s = (float)dur_pos * SONATA_V3_FRAME_DUR_S;
        float word_dur_s = 0.0f;
        for (int i = 0; i < wlen && dur_pos < n_dur; i++) {
            word_dur_s += durs[dur_pos] * SONATA_V3_FRAME_DUR_S;
            dur_pos++;
        }

        int word_copy = wlen;
        if (word_copy >= (int)sizeof(out[n_words].word)) word_copy = (int)sizeof(out[n_words].word) - 1;
        if (word_copy > 0) {
            memcpy(out[n_words].word, word_start, (size_t)word_copy);
            out[n_words].word[word_copy] = '\0';
        } else {
            out[n_words].word[0] = '\0';
        }
        out[n_words].start_s = word_start_s;
        out[n_words].end_s = word_start_s + word_dur_s;
        n_words++;
    }
    return n_words;
}

static void sonatav3_destroy(void *e) {
    SonataV3Engine *ev = (SonataV3Engine *)e;
    if (!ev) return;
    if (ev->flow_v3) sonata_flow_v3_destroy(ev->flow_v3);
    if (ev->vocoder) sonata_vocoder_destroy(ev->vocoder);
    if (ev->phonemizer) phonemizer_destroy(ev->phonemizer);
    free(ev->audio_buf);
    free(ev->mel_buf);
    free(ev);
}

static TtsInterface tts_create_sonata_v3(const char *flow_weights, const char *flow_config,
                                         const char *voc_weights, const char *voc_config,
                                         Phonemizer *phonemizer) {
    TtsInterface iface = {0};
    SonataV3Engine *ev = calloc(1, sizeof(SonataV3Engine));
    if (!ev) return iface;

    ev->flow_v3 = sonata_flow_v3_create(flow_weights, flow_config);
    if (!ev->flow_v3) {
        free(ev);
        return iface;
    }

    ev->vocoder = sonata_vocoder_create(voc_weights, voc_config);
    if (!ev->vocoder) {
        sonata_flow_v3_destroy(ev->flow_v3);
        free(ev);
        return iface;
    }

    ev->phonemizer = phonemizer;
    ev->audio_cap = 24000 * 10;
    ev->audio_buf = malloc((size_t)ev->audio_cap * sizeof(float));
    ev->mel_buf = malloc(SONATA_V3_MAX_FRAMES * SONATA_V3_MEL_DIM * sizeof(float));
    if (!ev->audio_buf || !ev->mel_buf) {
        free(ev->audio_buf);
        free(ev->mel_buf);
        sonata_vocoder_destroy(ev->vocoder);
        sonata_flow_v3_destroy(ev->flow_v3);
        free(ev);
        return iface;
    }

    iface.type = TTS_ENGINE_SONATA_V3;
    iface.engine = ev;
    iface.sample_rate = 24000;
    iface.frame_size = 480;  /* hop_length: 24000/480 = 50 Hz (matches Flow v3 + vocoder) */
    iface.set_text = sonatav3_set_text;
    iface.set_text_ipa = sonatav3_set_text_ipa;
    iface.set_text_done = sonatav3_set_text_done;
    iface.step = sonatav3_step;
    iface.get_audio = sonatav3_get_audio;
    iface.is_done = sonatav3_is_done;
    iface.peek_audio = sonatav3_peek_audio;
    iface.advance_audio = sonatav3_advance_audio;
    iface.reset = sonatav3_reset;
    iface.destroy = sonatav3_destroy;
    return iface;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * TTS Engine: Sonata (Semantic LM + Flow + iSTFT)
 *
 * Architecture: text → SPM → Sonata LM (Metal) → semantic tokens
 *               → (future: Flow) → iSTFT (vDSP/AMX) → audio
 *
 * Produces 480 samples per frame at 50 Hz = 24 kHz output.
 * iSTFT runs at 5000x+ realtime; LM is the bottleneck.
 * ═══════════════════════════════════════════════════════════════════════════ */

/* Sonata LM FFI */
extern void *sonata_lm_create(const char *weights_path, const char *config_path);
extern void  sonata_lm_destroy(void *engine);
extern int   sonata_lm_set_text(void *engine, const unsigned int *text_ids, int n);
extern int   sonata_lm_append_text(void *engine, const unsigned int *text_ids, int n);
extern int   sonata_lm_finish_text(void *engine);
extern int   sonata_lm_step(void *engine, int *out_token);
extern int   sonata_lm_reset(void *engine);
extern int   sonata_lm_is_done(void *engine);
extern int   sonata_lm_set_params(void *engine, float temperature, int top_k,
                                   float top_p, float rep_penalty);

/* SoundStorm parallel decoder FFI (drop-in LM replacement) */
extern void *sonata_storm_create(const char *weights_path, const char *config_path);
extern void  sonata_storm_destroy(void *engine);
extern int   sonata_storm_set_text(void *engine, const unsigned int *text_ids, int n);
extern int   sonata_storm_generate(void *engine, int *out_tokens, int max_tokens, int *out_count);
extern int   sonata_storm_set_params(void *engine, float temperature, int n_rounds);
extern int   sonata_storm_reset(void *engine);

/* Sonata iSTFT FFI */
typedef struct SonataISTFT SonataISTFT;
extern SonataISTFT *sonata_istft_create(int n_fft, int hop_length);
extern void         sonata_istft_destroy(SonataISTFT *dec);
extern void         sonata_istft_reset(SonataISTFT *dec);
extern int          sonata_istft_decode_frame(SonataISTFT *dec,
                        const float *magnitude, const float *phase,
                        float *out_audio);
extern int          sonata_istft_decode_batch(SonataISTFT *dec,
                        const float *magnitudes, const float *phases,
                        int n_frames, float *out_audio);

/* Sonata Flow + Decoder FFI (Rust/Metal) */
extern void *sonata_flow_create(const char *flow_weights, const char *flow_config,
                                 const char *decoder_weights, const char *decoder_config);
extern void  sonata_flow_destroy(void *engine);
extern int   sonata_flow_generate(void *engine, const int *semantic_tokens,
                                   int n_frames, float *out_magnitude, float *out_phase);
extern int   sonata_flow_generate_audio(void *engine, const int *semantic_tokens,
                                         int n_frames, float *out_audio, int max_samples);
extern int   sonata_flow_decoder_type(void *engine);
extern int   sonata_flow_samples_per_frame(void *engine);
extern int   sonata_flow_set_speaker(void *engine, int speaker_id);
extern int   sonata_flow_set_cfg_scale(void *engine, float scale);
extern int   sonata_flow_set_n_steps(void *engine, int n_steps);
extern void  sonata_flow_reset_phase(void *engine);
extern int   sonata_flow_set_causal(void *engine, int enable);
extern void  sonata_flow_reset_streaming(void *engine);
extern int   sonata_flow_generate_streaming_chunk(void *engine, const int *semantic_tokens,
    int n_frames, int chunk_offset, float *out_magnitude, float *out_phase);
extern int   sonata_flow_set_solver(void *engine, int use_heun);
extern int   sonata_flow_set_quality_mode(void *engine, int mode);
extern int   sonata_flow_set_speaker_embedding(void *engine, const float *embedding, int dim);
extern void  sonata_flow_clear_speaker_embedding(void *engine);

/* Sonata LM prosody FFI */
extern int   sonata_lm_set_prosody(void *engine, const float *features, int n);
extern int   sonata_lm_set_coarse_grained(void *engine, int enable);
extern int   sonata_lm_prosody_token_base(void *engine);
extern int   sonata_lm_num_prosody_tokens(void);
extern int   sonata_lm_inject_prosody_token(void *engine, int prosody_offset);
extern int   sonata_lm_inject_pause(void *engine, int n_frames);
extern int   sonata_lm_ms_to_frames(int ms);

/* Sonata LM speculative decoding FFI */
extern int   sonata_lm_load_draft(void *engine, const char *weights, const char *config);
extern int   sonata_lm_speculate_step(void *engine, int *out_tokens, int max_tokens, int *out_count);
extern int   sonata_lm_set_speculate_k(void *engine, int k);
extern int   sonata_lm_load_rnn_drafter(void *engine, const char *weights, const char *config);
extern int   sonata_lm_set_tree_config(void *engine, int width, int depth);

/* Sonata Flow prosody FFI */
extern int   sonata_flow_set_emotion(void *engine, int emotion_id);
extern int   sonata_flow_set_emotion_steering(void *engine, const float *direction,
                                               int dim, int layer_start, int layer_end, float scale);
extern void  sonata_flow_clear_emotion_steering(void *engine);
extern int   sonata_flow_set_prosody(void *engine, const float *features, int n);
extern int   sonata_flow_set_durations(void *engine, const float *durations, int n_frames);
extern int   sonata_flow_set_prosody_embedding(void *engine, const float *embedding, int dim);
extern void  sonata_flow_clear_prosody_embedding(void *engine);
extern int   sonata_flow_interpolate_speakers(void *engine, const float *emb_a,
                                               const float *emb_b, int dim, float alpha);

/* Sonata Flow v2 externs declared above near line ~812 */

#define SONATA_BUF_CAPACITY (24000 * 30)
#define SONATA_N_FFT 1024
#define SONATA_HOP 480
#define SONATA_N_BINS (SONATA_N_FFT / 2 + 1)
#define SONATA_MAX_FRAMES 2000
#define SONATA_FIRST_CHUNK 12
#define SONATA_CHUNK_SIZE 50
#define SONATA_CROSSFADE 480  /* 20ms at 24kHz — smooth spectral transition */

/* ─── Flow Worker: GPU-threaded flow generation for pipeline parallelism ─── */

typedef struct {
    pthread_t       thread;
    pthread_mutex_t mutex;
    pthread_cond_t  request_cond;
    pthread_cond_t  done_cond;

    void           *flow_engine;
    SonataISTFT    *istft;

    int             req_tokens[SONATA_MAX_FRAMES];
    int             req_n_tokens;
    int             req_pending;

    float          *result_audio;
    int             result_len;
    int             result_ready;

    float           crossfade_tail[SONATA_CROSSFADE];
    int             has_crossfade;

    _Atomic int     shutdown;
} FlowWorker;

static void *flow_worker_thread(void *arg) {
    FlowWorker *fw = (FlowWorker *)arg;
    float *mag_batch = (float *)calloc(SONATA_MAX_FRAMES * SONATA_N_BINS, sizeof(float));
    float *phase_batch = (float *)calloc(SONATA_MAX_FRAMES * SONATA_N_BINS, sizeof(float));
    if (!mag_batch || !phase_batch) {
        free(mag_batch);
        free(phase_batch);
        return NULL;
    }

    while (!atomic_load(&fw->shutdown)) {
        pthread_mutex_lock(&fw->mutex);
        while (!fw->req_pending && !atomic_load(&fw->shutdown)) {
            pthread_cond_wait(&fw->request_cond, &fw->mutex);
        }
        if (atomic_load(&fw->shutdown)) {
            pthread_mutex_unlock(&fw->mutex);
            break;
        }

        int n = fw->req_n_tokens;
        int tokens[SONATA_MAX_FRAMES];
        memcpy(tokens, fw->req_tokens, n * sizeof(int));
        fw->req_pending = 0;
        pthread_mutex_unlock(&fw->mutex);

        int total_audio = 0;
        int dec_type = fw->flow_engine ? sonata_flow_decoder_type(fw->flow_engine) : 0;
        if (fw->flow_engine && dec_type == 1) {
            int max_samples = n * SONATA_HOP + 4096;
            if (max_samples > SONATA_BUF_CAPACITY) max_samples = SONATA_BUF_CAPACITY;
            total_audio = sonata_flow_generate_audio(fw->flow_engine, tokens, n,
                                                      fw->result_audio, max_samples);
            if (total_audio > 0 && fw->has_crossfade) {
                int cf = SONATA_CROSSFADE;
                if (total_audio < cf) cf = total_audio;
                for (int i = 0; i < cf; i++) {
                    float alpha = (float)i / (float)cf;
                    fw->result_audio[i] =
                        fw->crossfade_tail[i] * (1.0f - alpha) +
                        fw->result_audio[i] * alpha;
                }
            }
            if (total_audio >= SONATA_CROSSFADE) {
                memcpy(fw->crossfade_tail,
                       &fw->result_audio[total_audio - SONATA_CROSSFADE],
                       SONATA_CROSSFADE * sizeof(float));
                fw->has_crossfade = 1;
            }
        } else if (fw->flow_engine && mag_batch && phase_batch) {
            int bins = sonata_flow_generate(fw->flow_engine, tokens, n,
                                             mag_batch, phase_batch);
            if (bins > 0) {
                for (int t = 0; t < n; t++) {
                    float frame_audio[SONATA_HOP];
                    int ns = sonata_istft_decode_frame(fw->istft,
                        &mag_batch[t * bins], &phase_batch[t * bins], frame_audio);
                    if (ns > 0 && total_audio + ns < SONATA_BUF_CAPACITY) {
                        memcpy(&fw->result_audio[total_audio], frame_audio, ns * sizeof(float));
                        total_audio += ns;
                    }
                }
                if (fw->has_crossfade && total_audio > 0) {
                    int cf = SONATA_CROSSFADE;
                    if (total_audio < cf) cf = total_audio;
                    for (int i = 0; i < cf; i++) {
                        float alpha = (float)i / (float)cf;
                        fw->result_audio[i] =
                            fw->crossfade_tail[i] * (1.0f - alpha) +
                            fw->result_audio[i] * alpha;
                    }
                }
                if (total_audio >= SONATA_CROSSFADE) {
                    memcpy(fw->crossfade_tail,
                           &fw->result_audio[total_audio - SONATA_CROSSFADE],
                           SONATA_CROSSFADE * sizeof(float));
                    fw->has_crossfade = 1;
                }
            }
        }

        pthread_mutex_lock(&fw->mutex);
        fw->result_len = total_audio;
        fw->result_ready = 1;
        pthread_cond_signal(&fw->done_cond);
        pthread_mutex_unlock(&fw->mutex);
    }

    free(mag_batch);
    free(phase_batch);
    return NULL;
}

static FlowWorker *flow_worker_create(void *flow_engine, SonataISTFT *istft) {
    FlowWorker *fw = (FlowWorker *)calloc(1, sizeof(FlowWorker));
    if (!fw) return NULL;

    fw->flow_engine = flow_engine;
    fw->istft = istft;
    fw->result_audio = (float *)calloc(SONATA_BUF_CAPACITY, sizeof(float));
    if (!fw->result_audio) {
        free(fw);
        return NULL;
    }
    atomic_store(&fw->shutdown, 0);
    pthread_mutex_init(&fw->mutex, NULL);
    pthread_cond_init(&fw->request_cond, NULL);
    pthread_cond_init(&fw->done_cond, NULL);

    if (pthread_create(&fw->thread, NULL, flow_worker_thread, fw) != 0) {
        free(fw->result_audio);
        free(fw);
        return NULL;
    }
    fprintf(stderr, "[sonata] Flow worker thread started for pipeline parallelism\n");
    return fw;
}

static void flow_worker_destroy(FlowWorker *fw) {
    if (!fw) return;
    atomic_store(&fw->shutdown, 1);
    pthread_cond_signal(&fw->request_cond);
    pthread_join(fw->thread, NULL);
    pthread_mutex_destroy(&fw->mutex);
    pthread_cond_destroy(&fw->request_cond);
    pthread_cond_destroy(&fw->done_cond);
    free(fw->result_audio);
    free(fw);
}

static void flow_worker_submit(FlowWorker *fw, const int *tokens, int n) {
    pthread_mutex_lock(&fw->mutex);
    memcpy(fw->req_tokens, tokens, n * sizeof(int));
    fw->req_n_tokens = n;
    fw->req_pending = 1;
    fw->result_ready = 0;
    pthread_cond_signal(&fw->request_cond);
    pthread_mutex_unlock(&fw->mutex);
}

static int flow_worker_collect(FlowWorker *fw, float *out_audio) {
    pthread_mutex_lock(&fw->mutex);
    int iterations = 0;
    while (!fw->result_ready) {
        if (atomic_load(&fw->shutdown)) {
            pthread_mutex_unlock(&fw->mutex);
            return 0;
        }
        if (++iterations > 100) { /* 100 * 50ms = 5s max wait */
            fprintf(stderr, "[flow_worker] collect timed out after 5s\n");
            pthread_mutex_unlock(&fw->mutex);
            return -1;
        }
        struct timespec ts;
        clock_gettime(CLOCK_REALTIME, &ts);
        ts.tv_nsec += 50000000; /* 50ms timeout */
        if (ts.tv_nsec >= 1000000000) {
            ts.tv_sec++;
            ts.tv_nsec -= 1000000000;
        }
        pthread_cond_timedwait(&fw->done_cond, &fw->mutex, &ts);
    }
    int len = fw->result_len;
    if (len > SONATA_BUF_CAPACITY) len = SONATA_BUF_CAPACITY;
    if (len > 0) memcpy(out_audio, fw->result_audio, len * sizeof(float));
    fw->result_ready = 0;
    pthread_mutex_unlock(&fw->mutex);
    return len;
}

static int flow_worker_try_collect(FlowWorker *fw, float *out_audio) {
    pthread_mutex_lock(&fw->mutex);
    if (!fw->result_ready) {
        pthread_mutex_unlock(&fw->mutex);
        return -1;
    }
    int len = fw->result_len;
    if (len > SONATA_BUF_CAPACITY) len = SONATA_BUF_CAPACITY;
    if (len > 0) memcpy(out_audio, fw->result_audio, len * sizeof(float));
    fw->result_ready = 0;
    pthread_mutex_unlock(&fw->mutex);
    return len;
}

typedef struct {
    void           *lm_engine;
    void           *storm_engine;    /* SoundStorm parallel decoder (alternative to LM) */
    void           *flow_engine;
    SonataISTFT    *istft;
    SPMTokenizer   *tokenizer;
    Phonemizer     *phonemizer;
    float          *audio_buf;
    int             buf_write;
    int             buf_read;
    int             done;
    int             semantic_tokens[SONATA_MAX_FRAMES];
    int             n_semantic_tokens;
    int             is_first_chunk;
    float           crossfade_tail[SONATA_CROSSFADE];
    int             has_crossfade;
    FlowWorker     *flow_worker;
    int             parallel_pending;
    int             active_gen;
    int             text_finalized;
    int             use_speculative;
    int             use_storm;       /* 1 = use SoundStorm instead of AR LM */
    int             tts_quality_mode;   /* 0=FAST, 1=BALANCED, 2=HIGH */
    int             tts_first_chunk_fast; /* 1 = FAST for first chunk, then revert */
    float          *collect_buf;
    float          *mag_scratch;
    float          *phase_scratch;
    float           phase_accum[SONATA_N_BINS]; /* per-instance phase state for placeholder synth */
    pthread_mutex_t crossfade_mutex;            /* protects crossfade_tail access */
} SonataEngine;

static SonataEngine *sonata_engine_create(
    const char *lm_weights, const char *lm_config, const char *tokenizer_path,
    const char *flow_weights, const char *flow_config,
    const char *dec_weights, const char *dec_config
) {
    SonataEngine *e = (SonataEngine *)calloc(1, sizeof(SonataEngine));
    if (!e) return NULL;

    /* Load tokenizer */
    FILE *f = fopen(tokenizer_path, "rb");
    if (!f) { fprintf(stderr, "[sonata] Cannot open tokenizer: %s\n", tokenizer_path); free(e); return NULL; }
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    uint8_t *tok_data = (uint8_t *)malloc(sz);
    if (!tok_data) { fclose(f); free(e); return NULL; }
    size_t nread = fread(tok_data, 1, (size_t)sz, f);
    fclose(f);
    if (nread != (size_t)sz) { free(tok_data); free(e); return NULL; }
    e->tokenizer = spm_create(tok_data, (uint32_t)sz);
    free(tok_data);
    if (!e->tokenizer) { fprintf(stderr, "[sonata] Tokenizer init failed\n"); free(e); return NULL; }

    /* Create iSTFT decoder */
    e->istft = sonata_istft_create(SONATA_N_FFT, SONATA_HOP);
    if (!e->istft) {
        fprintf(stderr, "[sonata] iSTFT init failed\n");
        spm_destroy(e->tokenizer);
        free(e);
        return NULL;
    }

    /* Load Rust LM */
    e->lm_engine = sonata_lm_create(lm_weights, lm_config);
    if (!e->lm_engine) {
        fprintf(stderr, "[sonata] LM init failed\n");
        sonata_istft_destroy(e->istft);
        spm_destroy(e->tokenizer);
        free(e);
        return NULL;
    }

    /* Load Flow + Decoder on Metal (optional — falls back to placeholder if missing) */
    e->flow_engine = NULL;
    if (flow_weights && flow_config) {
        e->flow_engine = sonata_flow_create(flow_weights, flow_config, dec_weights, dec_config);
        if (e->flow_engine) {
            fprintf(stderr, "[sonata] Flow network loaded on Metal GPU\n");
        } else {
            fprintf(stderr, "[sonata] Flow not loaded — using placeholder synthesis\n");
        }
    }

    e->audio_buf = (float *)calloc(SONATA_BUF_CAPACITY, sizeof(float));
    e->collect_buf = (float *)calloc(SONATA_BUF_CAPACITY, sizeof(float));
    e->mag_scratch = (float *)calloc((size_t)SONATA_MAX_FRAMES * SONATA_N_BINS, sizeof(float));
    e->phase_scratch = (float *)calloc((size_t)SONATA_MAX_FRAMES * SONATA_N_BINS, sizeof(float));
    if (!e->audio_buf || !e->collect_buf || !e->mag_scratch || !e->phase_scratch) {
        fprintf(stderr, "[sonata] Audio buffer alloc failed\n");
        free(e->audio_buf); free(e->collect_buf);
        free(e->mag_scratch); free(e->phase_scratch);
        if (e->flow_engine) sonata_flow_destroy(e->flow_engine);
        if (e->lm_engine) sonata_lm_destroy(e->lm_engine);
        if (e->istft) sonata_istft_destroy(e->istft);
        if (e->tokenizer) spm_destroy(e->tokenizer);
        free(e);
        return NULL;
    }
    e->buf_write = 0;
    e->buf_read = 0;
    e->done = 0;
    e->n_semantic_tokens = 0;
    e->is_first_chunk = 1;
    e->has_crossfade = 0;
    e->parallel_pending = 0;
    memset(e->phase_accum, 0, sizeof(e->phase_accum));
    pthread_mutex_init(&e->crossfade_mutex, NULL);

    if (e->flow_engine) {
        e->flow_worker = flow_worker_create(e->flow_engine, e->istft);
    } else {
        e->flow_worker = NULL;
    }

    fprintf(stderr, "[sonata] Engine ready (LM + %s + iSTFT + SPM%s)\n",
            e->flow_engine ? "Flow" : "placeholder",
            e->flow_worker ? " + parallel" : "");
    return e;
}

static int sonata_tokenize(SonataEngine *e, const char *text, unsigned int *uids, int max) {
    int n;
    if (e->phonemizer) {
        int pids[512];
        n = phonemizer_text_to_ids(e->phonemizer, text, pids, max);
        if (n > 0) {
            for (int i = 0; i < n; i++) uids[i] = (unsigned int)pids[i];
        } else {
            int32_t ids[512];
            n = spm_encode(e->tokenizer, text, ids, max);
            if (n <= 0) return -1;
            for (int i = 0; i < n; i++) uids[i] = (unsigned int)ids[i];
        }
    } else {
        int32_t ids[512];
        n = spm_encode(e->tokenizer, text, ids, max);
        if (n <= 0) return -1;
        for (int i = 0; i < n; i++) uids[i] = (unsigned int)ids[i];
    }
    return n;
}

static int is_sentence_ender(char c) {
    return c == '.' || c == '?' || c == '!' || c == ';';
}

static void sonata_flush_chunk(SonataEngine *e);
static void sonata_collect_parallel(SonataEngine *e);

static void sonata_force_finish(SonataEngine *e) {
    if (e->n_semantic_tokens > 0)
        sonata_flush_chunk(e);
    sonata_collect_parallel(e);
    e->active_gen = 0;
    e->text_finalized = 0;
}

static int sonata_set_text(void *engine, const char *text) {
    SonataEngine *e = (SonataEngine *)engine;
    if (!e || !text) return -1;

    /* If previous sentence was finalized and new text arrives, the LM may
       still be mid-generation. Flush remaining semantic tokens and force a
       clean reset — prevents cross-sentence text mixing. */
    if (e->active_gen && e->text_finalized) {
        sonata_force_finish(e);
    }

    /* Streaming append: if actively generating within the same sentence,
       tokenize the new fragment and extend the LM's text buffer.
       SPM greedy tokenization is stable at word boundaries (spaces),
       so fragment tokenization matches full-sentence tokenization. */
    if (e->active_gen && !e->done) {
        unsigned int uids[512];
        int n = sonata_tokenize(e, text, uids, 512);
        if (n <= 0) return -1;
        int ret = sonata_lm_append_text(e->lm_engine, uids, n);
        if (ret < 0) return ret;

        int tlen = (int)strlen(text);
        if (tlen > 0 && is_sentence_ender(text[tlen - 1])) {
            int fret = sonata_lm_finish_text(e->lm_engine);
            if (fret < 0)
                fprintf(stderr, "[sonata] Warning: sonata_lm_finish_text failed (%d)\n", fret);
            e->text_finalized = 1;
        }
        return ret;
    }

    /* Fresh start: full reset for new utterance */
    e->buf_write = 0;
    e->buf_read = 0;
    e->done = 0;
    e->n_semantic_tokens = 0;
    e->is_first_chunk = 1;
    e->has_crossfade = 0;
    e->active_gen = 1;
    e->text_finalized = 0;

    unsigned int uids[512];
    int n = sonata_tokenize(e, text, uids, 512);
    if (n <= 0) return -1;

    sonata_istft_reset(e->istft);
    if (e->flow_engine) sonata_flow_reset_phase(e->flow_engine);

    if (e->use_storm && e->storm_engine) {
        sonata_storm_reset(e->storm_engine);
        return sonata_storm_set_text(e->storm_engine, uids, n);
    }
    return sonata_lm_set_text(e->lm_engine, uids, n);
}

static int sonata_set_text_ipa(void *engine, const char *ipa, const char *text) {
    SonataEngine *e = (SonataEngine *)engine;
    if (!e || !ipa || !*ipa) return sonata_set_text(engine, text);
    if (!e->phonemizer) return sonata_set_text(engine, text);

    int pids[512];
    int n = phonemizer_ipa_to_ids(e->phonemizer, ipa, pids, 512);
    if (n <= 0) return sonata_set_text(engine, text);

    unsigned int uids[512];
    for (int i = 0; i < n; i++) uids[i] = (unsigned int)pids[i];

    if (e->active_gen && e->text_finalized)
        sonata_force_finish(e);

    if (e->active_gen && !e->done) {
        int ret = sonata_lm_append_text(e->lm_engine, uids, n);
        int tlen = text ? (int)strlen(text) : 0;
        if (tlen > 0 && is_sentence_ender(text[tlen - 1])) {
            int fret = sonata_lm_finish_text(e->lm_engine);
            if (fret < 0)
                fprintf(stderr, "[sonata] Warning: sonata_lm_finish_text failed (%d)\n", fret);
            e->text_finalized = 1;
        }
        return ret;
    }

    e->buf_write = 0;
    e->buf_read = 0;
    e->done = 0;
    e->n_semantic_tokens = 0;
    e->is_first_chunk = 1;
    e->has_crossfade = 0;
    e->active_gen = 1;
    e->text_finalized = 0;

    sonata_istft_reset(e->istft);
    if (e->flow_engine) sonata_flow_reset_phase(e->flow_engine);
    return sonata_lm_set_text(e->lm_engine, uids, n);
}

static int sonata_set_text_done(void *engine) {
    SonataEngine *e = (SonataEngine *)engine;
    if (e && e->active_gen) {
        int fret = sonata_lm_finish_text(e->lm_engine);
        if (fret < 0)
            fprintf(stderr, "[sonata] Warning: sonata_lm_finish_text failed (%d)\n", fret);
        e->text_finalized = 1;
    }
    return 0;
}

static void sonata_collect_parallel(SonataEngine *e) {
    if (!e->parallel_pending || !e->flow_worker) return;
    int len = flow_worker_collect(e->flow_worker, e->collect_buf);
    if (len > 0 && e->buf_write + len < SONATA_BUF_CAPACITY) {
        memcpy(&e->audio_buf[e->buf_write], e->collect_buf, len * sizeof(float));
        e->buf_write += len;
    }
    e->parallel_pending = 0;
}

static void sonata_try_collect_parallel(SonataEngine *e) {
    if (!e->parallel_pending || !e->flow_worker) return;
    int len = flow_worker_try_collect(e->flow_worker, e->collect_buf);
    if (len < 0) return;
    if (len > 0 && e->buf_write + len < SONATA_BUF_CAPACITY) {
        memcpy(&e->audio_buf[e->buf_write], e->collect_buf, len * sizeof(float));
        e->buf_write += len;
    }
    e->parallel_pending = 0;
}

static void sonata_flush_chunk(SonataEngine *e) {
    if (e->n_semantic_tokens <= 0) return;
    int n = e->n_semantic_tokens;

    /* First-chunk-fast: use FAST mode for first chunk, then revert to configured quality */
    if (e->tts_first_chunk_fast && e->is_first_chunk && e->flow_engine) {
        sonata_flow_set_quality_mode(e->flow_engine, 0 /* FAST */);
    }

    if (e->flow_worker) {
        sonata_collect_parallel(e);
        flow_worker_submit(e->flow_worker, e->semantic_tokens, n);
        e->parallel_pending = 1;
    } else if (e->flow_engine && sonata_flow_decoder_type(e->flow_engine) == 1) {
        int chunk_start = e->buf_write;
        int max_samples = n * SONATA_HOP + 4096;
        int remain = SONATA_BUF_CAPACITY - e->buf_write;
        if (max_samples > remain) max_samples = remain;
        int ns = sonata_flow_generate_audio(e->flow_engine, e->semantic_tokens, n,
                                             &e->audio_buf[e->buf_write], max_samples);
        if (ns > 0) {
            e->buf_write += ns;
            pthread_mutex_lock(&e->crossfade_mutex);
            if (e->has_crossfade && chunk_start > 0) {
                int cf = SONATA_CROSSFADE;
                if (ns < cf) cf = ns;
                for (int i = 0; i < cf; i++) {
                    float alpha = (float)i / (float)cf;
                    e->audio_buf[chunk_start + i] =
                        e->crossfade_tail[i] * (1.0f - alpha) +
                        e->audio_buf[chunk_start + i] * alpha;
                }
            }
            if (ns >= SONATA_CROSSFADE) {
                memcpy(e->crossfade_tail,
                       &e->audio_buf[e->buf_write - SONATA_CROSSFADE],
                       SONATA_CROSSFADE * sizeof(float));
                e->has_crossfade = 1;
            }
            pthread_mutex_unlock(&e->crossfade_mutex);
        }
    } else if (e->flow_engine) {
        int bins = sonata_flow_generate(e->flow_engine, e->semantic_tokens, n,
                                         e->mag_scratch, e->phase_scratch);
        if (bins > 0) {
            int chunk_start = e->buf_write;
            for (int t = 0; t < n; t++) {
                float frame_audio[SONATA_HOP];
                int ns = sonata_istft_decode_frame(e->istft,
                    &e->mag_scratch[t * bins], &e->phase_scratch[t * bins], frame_audio);
                if (ns > 0 && e->buf_write + ns < SONATA_BUF_CAPACITY) {
                    memcpy(&e->audio_buf[e->buf_write], frame_audio, ns * sizeof(float));
                    e->buf_write += ns;
                }
            }
            pthread_mutex_lock(&e->crossfade_mutex);
            if (e->has_crossfade && chunk_start > 0) {
                int cf = SONATA_CROSSFADE;
                if (e->buf_write - chunk_start < cf) cf = e->buf_write - chunk_start;
                for (int i = 0; i < cf; i++) {
                    float alpha = (float)i / (float)cf;
                    e->audio_buf[chunk_start + i] =
                        e->crossfade_tail[i] * (1.0f - alpha) +
                        e->audio_buf[chunk_start + i] * alpha;
                }
            }
            int total_written = e->buf_write - chunk_start;
            if (total_written >= SONATA_CROSSFADE) {
                memcpy(e->crossfade_tail,
                       &e->audio_buf[e->buf_write - SONATA_CROSSFADE],
                       SONATA_CROSSFADE * sizeof(float));
                e->has_crossfade = 1;
            }
            pthread_mutex_unlock(&e->crossfade_mutex);
        }
    } else {
        /* Placeholder per-frame synthesis (no flow model) */
        for (int t = 0; t < n; t++) {
            float magnitude[SONATA_N_BINS];
            float phase[SONATA_N_BINS];
            memset(magnitude, 0, sizeof(magnitude));
            int tok = e->semantic_tokens[t];
            int base_bin = 5 + (tok % 40);
            for (int b = base_bin; b < base_bin + 10 && b < SONATA_N_BINS; b++) {
                magnitude[b] = 0.5f / (1.0f + (b - base_bin) * (b - base_bin) * 0.1f);
            }
            for (int b = 0; b < SONATA_N_BINS; b++) {
                float freq = (float)b * 24000.0f / SONATA_N_FFT;
                e->phase_accum[b] += 2.0f * 3.14159f * freq * SONATA_HOP / 24000.0f;
                phase[b] = e->phase_accum[b];
            }
            float frame[SONATA_HOP];
            int ns = sonata_istft_decode_frame(e->istft, magnitude, phase, frame);
            if (ns > 0 && e->buf_write + ns <= SONATA_BUF_CAPACITY) {
                for (int i = 0; i < ns; i++) {
                    float s = frame[i];
                    if (s > 1.0f) s = 1.0f;
                    if (s < -1.0f) s = -1.0f;
                    e->audio_buf[e->buf_write++] = s;
                }
            }
        }
    }
    e->n_semantic_tokens = 0;
    if (e->is_first_chunk) {
        e->is_first_chunk = 0;
        /* Revert from FAST to configured quality after first chunk */
        if (e->tts_first_chunk_fast && e->flow_engine)
            sonata_flow_set_quality_mode(e->flow_engine, e->tts_quality_mode);
    }
}

static int sonata_step(void *engine) {
    SonataEngine *e = (SonataEngine *)engine;
    if (!e || e->done) return 1;

    sonata_try_collect_parallel(e);

    /* SoundStorm: generate all tokens in one call, flush, and mark done */
    if (e->use_storm && e->storm_engine) {
        int count = 0;
        int status = sonata_storm_generate(e->storm_engine,
            e->semantic_tokens, SONATA_MAX_FRAMES, &count);
        if (status == 0 && count > 0) {
            e->n_semantic_tokens = count;
            sonata_flush_chunk(e);
            sonata_collect_parallel(e);
        }
        e->done = 1;
        e->active_gen = 0;
        return 1;
    }

    if (e->use_speculative) {
        int spec_tokens[16];
        int spec_count = 0;
        int status = sonata_lm_speculate_step(e->lm_engine, spec_tokens, 16, &spec_count);

        if (status == 1 || status == -1) {
            for (int i = 0; i < spec_count && e->n_semantic_tokens < SONATA_MAX_FRAMES; i++)
                e->semantic_tokens[e->n_semantic_tokens++] = spec_tokens[i];
            sonata_flush_chunk(e);
            sonata_collect_parallel(e);
            e->done = 1;
            e->active_gen = 0;
            return 1;
        }

        for (int i = 0; i < spec_count; i++) {
            if (e->n_semantic_tokens < SONATA_MAX_FRAMES)
                e->semantic_tokens[e->n_semantic_tokens++] = spec_tokens[i];
        }
    } else {
        int semantic_token = 0;
        int status = sonata_lm_step(e->lm_engine, &semantic_token);

        if (status == 1 || status == -1) {
            sonata_flush_chunk(e);
            sonata_collect_parallel(e);
            e->done = 1;
            e->active_gen = 0;
            return 1;
        }

        if (e->n_semantic_tokens < SONATA_MAX_FRAMES) {
            e->semantic_tokens[e->n_semantic_tokens++] = semantic_token;
        }
    }

    /* Prosody-aware adaptive chunk sizing:
     * - First chunk: fixed small size for low TTFA
     * - Subsequent: prefer flushing at prosodic boundaries for seamless audio.
     *   Natural split points: sustained phonemes (token repetition ≥3),
     *   silence tokens (PAD=0), or after break tokens.
     * - Flexible range: allow chunks 15-80 tokens based on boundary quality.
     * - Hard max prevents unbounded accumulation. */
    int n = e->n_semantic_tokens;
    int last_tok = (n > 0) ? e->semantic_tokens[n - 1] : -1;
    int should_flush = 0;
    int hard_max = e->is_first_chunk ? SONATA_FIRST_CHUNK : 80;
    int soft_min = e->is_first_chunk ? SONATA_FIRST_CHUNK : 20;

    if (n >= hard_max) {
        should_flush = 1;
    } else if (!e->is_first_chunk && n >= soft_min) {
        /* Silence token (PAD=0): ideal split point */
        if (last_tok == 0 && n > 3) {
            should_flush = 1;
        }
        /* Sustained phoneme: 3+ consecutive identical tokens */
        else if (n >= 3 &&
                 e->semantic_tokens[n-1] == e->semantic_tokens[n-2] &&
                 e->semantic_tokens[n-2] == e->semantic_tokens[n-3]) {
            should_flush = 1;
        }
        /* Transition detection: large token value jump suggests phoneme boundary */
        else if (n >= 2) {
            int prev = e->semantic_tokens[n-2];
            int curr = e->semantic_tokens[n-1];
            int delta = abs(curr - prev);
            if (delta > 500 && n >= SONATA_FIRST_CHUNK + 5) {
                should_flush = 1;
            }
        }
    } else if (e->is_first_chunk && n >= SONATA_FIRST_CHUNK) {
        should_flush = 1;
    }

    if (should_flush) {
        sonata_flush_chunk(e);
    }

    return 0;
}

static int sonata_get_audio(void *engine, float *buf, int max) {
    SonataEngine *e = (SonataEngine *)engine;
    if (!e) return -1;
    int avail = e->buf_write - e->buf_read;
    int n = avail < max ? avail : max;
    if (n > 0) {
        memcpy(buf, e->audio_buf + e->buf_read, n * sizeof(float));
        e->buf_read += n;
    }
    return n;
}

static int sonata_is_done(void *engine) {
    SonataEngine *e = (SonataEngine *)engine;
    return e ? e->done : 1;
}

static int sonata_peek_audio(void *engine, const float **ptr, int *count) {
    SonataEngine *e = (SonataEngine *)engine;
    if (!e || !ptr || !count) return -1;
    *ptr = e->audio_buf + e->buf_read;
    *count = e->buf_write - e->buf_read;
    return 0;
}

static int sonata_advance_audio(void *engine, int n) {
    SonataEngine *e = (SonataEngine *)engine;
    if (!e) return -1;
    int avail = e->buf_write - e->buf_read;
    e->buf_read += (n < avail ? n : avail);
    return 0;
}

static int sonata_reset_engine(void *engine) {
    SonataEngine *e = (SonataEngine *)engine;
    if (!e) return -1;
    sonata_collect_parallel(e);
    sonata_lm_reset(e->lm_engine);
    sonata_istft_reset(e->istft);
    if (e->flow_engine) sonata_flow_reset_phase(e->flow_engine);
    if (e->flow_worker) {
        e->flow_worker->has_crossfade = 0;
    }
    e->buf_write = 0;
    e->buf_read = 0;
    e->done = 0;
    e->n_semantic_tokens = 0;
    e->is_first_chunk = 1;
    e->has_crossfade = 0;
    e->parallel_pending = 0;
    e->active_gen = 0;
    e->text_finalized = 0;
    memset(e->phase_accum, 0, sizeof(e->phase_accum));
    return 0;
}

static void sonata_destroy_engine(void *engine) {
    SonataEngine *e = (SonataEngine *)engine;
    if (!e) return;
    if (e->flow_worker) flow_worker_destroy(e->flow_worker);
    if (e->flow_engine) sonata_flow_destroy(e->flow_engine);
    if (e->storm_engine) sonata_storm_destroy(e->storm_engine);
    if (e->lm_engine) sonata_lm_destroy(e->lm_engine);
    if (e->istft) sonata_istft_destroy(e->istft);
    if (e->tokenizer) spm_destroy(e->tokenizer);
    pthread_mutex_destroy(&e->crossfade_mutex);
    free(e->audio_buf);
    free(e->collect_buf);
    free(e->mag_scratch);
    free(e->phase_scratch);
    free(e);
}

static TtsInterface tts_create_sonata(
    const char *lm_weights, const char *lm_config, const char *tokenizer_path,
    const char *flow_weights, const char *flow_config,
    const char *dec_weights, const char *dec_config
) {
    TtsInterface iface = {0};
    iface.type = TTS_ENGINE_SONATA;
    iface.engine = sonata_engine_create(lm_weights, lm_config, tokenizer_path,
                                         flow_weights, flow_config, dec_weights, dec_config);
    iface.sample_rate = 24000;
    iface.frame_size = SONATA_HOP;
    iface.set_text = sonata_set_text;
    iface.set_text_ipa = sonata_set_text_ipa;
    iface.set_text_done = sonata_set_text_done;
    iface.step = sonata_step;
    iface.get_audio = sonata_get_audio;
    iface.is_done = sonata_is_done;
    iface.peek_audio = sonata_peek_audio;
    iface.advance_audio = sonata_advance_audio;
    iface.reset = sonata_reset_engine;
    iface.destroy = sonata_destroy_engine;
    return iface;
}


/* ═══════════════════════════════════════════════════════════════════════════
 * LLM Client Abstraction
 *
 * Allows the pipeline to use different LLM backends (Claude, Gemini, etc.)
 * via a uniform function-pointer interface, matching SttInterface/TtsInterface.
 * ═══════════════════════════════════════════════════════════════════════════ */

typedef enum { LLM_ENGINE_CLAUDE, LLM_ENGINE_GEMINI, LLM_ENGINE_LOCAL } LLMEngineType;

typedef struct {
    void           *engine;
    LLMEngineType   type;

    int         (*send)(void *engine, const char *user_text);
    int         (*poll)(void *engine, int timeout_ms);
    const char *(*peek_tokens)(void *engine, int *out_len);
    void        (*consume_tokens)(void *engine, int count);
    void        (*cancel)(void *engine);
    void        (*commit_turn)(void *engine, const char *user_text);
    bool        (*is_response_done)(void *engine);
    bool        (*has_error)(void *engine);
    void        (*cleanup)(void *engine);
} LLMClient;

/* ═══════════════════════════════════════════════════════════════════════════
 * TTS Worker Thread
 *
 * Runs TTS generation on a dedicated thread, decoupled from the main loop.
 * Text sentences are posted via a lock-free queue. Generated audio is
 * written to an SPSC ring buffer for the main thread to consume.
 * ═══════════════════════════════════════════════════════════════════════════ */

#define TTS_QUEUE_SIZE 32
#define TTS_RING_SAMPLES (48000 * 10)  /* 10 seconds of audio at 48kHz */

typedef struct {
    TtsInterface    tts;
    pthread_t       thread;
    _Atomic int     running;
    _Atomic int     cancel;

    /* Lock-free text sentence queue (SPSC) */
    char           *text_queue[TTS_QUEUE_SIZE];
    _Atomic int     text_head;  /* Written by producer (main thread) */
    _Atomic int     text_tail;  /* Read by consumer (TTS thread) */
    _Atomic int     text_done_flag;  /* Signal no more text for this turn */

    /* Audio output ring buffer (SPSC: TTS thread writes, main reads) */
    float          *audio_ring;
    int             audio_ring_size;
    _Atomic int     audio_write_pos;
    _Atomic int     audio_read_pos;

    _Atomic int     gen_done;   /* 1 when TTS generation is complete */
    pthread_mutex_t wake_mutex;
    pthread_cond_t  wake_cond;
} TtsWorker;

static int tts_ring_available(TtsWorker *w) {
    int wp = atomic_load_explicit(&w->audio_write_pos, memory_order_acquire);
    int rp = atomic_load_explicit(&w->audio_read_pos, memory_order_acquire);
    int avail = wp - rp;
    if (avail < 0) avail += w->audio_ring_size;
    return avail;
}

static int tts_ring_free(TtsWorker *w) {
    return w->audio_ring_size - 1 - tts_ring_available(w);
}

static void tts_ring_write(TtsWorker *w, const float *data, int count) {
    int wp = atomic_load_explicit(&w->audio_write_pos, memory_order_relaxed);
    int sz = w->audio_ring_size;
    for (int i = 0; i < count; i++) {
        w->audio_ring[wp] = data[i];
        wp = (wp + 1) % sz;
    }
    atomic_store_explicit(&w->audio_write_pos, wp, memory_order_release);
}

__attribute__((unused))
static int tts_ring_read(TtsWorker *w, float *out, int max_samples) {
    int avail = tts_ring_available(w);
    int to_read = avail < max_samples ? avail : max_samples;
    int rp = atomic_load_explicit(&w->audio_read_pos, memory_order_relaxed);
    int sz = w->audio_ring_size;
    for (int i = 0; i < to_read; i++) {
        out[i] = w->audio_ring[rp];
        rp = (rp + 1) % sz;
    }
    atomic_store_explicit(&w->audio_read_pos, rp, memory_order_release);
    return to_read;
}

static void *tts_worker_thread(void *arg) {
    TtsWorker *w = (TtsWorker *)arg;
    float pcm_buf[4096];

    /* Elevate to real-time priority — TTS generation is latency-critical */
    ap_set_realtime_inference();

    while (atomic_load_explicit(&w->running, memory_order_acquire)) {
        /* Check for cancel */
        if (atomic_load_explicit(&w->cancel, memory_order_acquire)) {
            if (w->tts.reset(w->tts.engine) != 0) {
                fprintf(stderr, "[tts_worker] WARNING: TTS reset failed during cancel\n");
            }
            /* Drain any queued text (only the worker does this) */
            int ct = atomic_load_explicit(&w->text_tail, memory_order_acquire);
            int ch = atomic_load_explicit(&w->text_head, memory_order_acquire);
            while (ct != ch) {
                free(w->text_queue[ct % TTS_QUEUE_SIZE]);
                w->text_queue[ct % TTS_QUEUE_SIZE] = NULL;
                ct = (ct + 1) % TTS_QUEUE_SIZE;
            }
            atomic_store_explicit(&w->text_tail, ct, memory_order_release);
            atomic_store_explicit(&w->text_head, ct, memory_order_release);
            atomic_store_explicit(&w->text_done_flag, 0, memory_order_release);
            /* Reset audio ring positions from the worker thread — safe because
               both sides have stopped normal operation under cancel. */
            atomic_store_explicit(&w->audio_write_pos, 0, memory_order_release);
            atomic_store_explicit(&w->audio_read_pos, 0, memory_order_release);
            atomic_store_explicit(&w->cancel, 0, memory_order_release);
            atomic_store_explicit(&w->gen_done, 1, memory_order_release);
            continue;
        }

        /* Feed any queued text to TTS */
        int tail = atomic_load_explicit(&w->text_tail, memory_order_acquire);
        int head = atomic_load_explicit(&w->text_head, memory_order_acquire);
        while (tail != head) {
            char *text = w->text_queue[tail % TTS_QUEUE_SIZE];
            if (text) {
                w->tts.set_text(w->tts.engine, text);
                free(text);
                w->text_queue[tail % TTS_QUEUE_SIZE] = NULL;
            }
            tail = (tail + 1) % TTS_QUEUE_SIZE;
            atomic_store_explicit(&w->text_tail, tail, memory_order_release);
        }

        /* Signal text done if flagged */
        if (atomic_load_explicit(&w->text_done_flag, memory_order_acquire) == 1) {
            int ret = w->tts.set_text_done(w->tts.engine);
            if (ret != 0) {
                fprintf(stderr, "[pipeline] TTS synthesis failed (set_text_done=%d)\n", ret);
            }
            atomic_store_explicit(&w->text_done_flag, 2, memory_order_release);
        }

        /* Run TTS step if not done */
        if (!w->tts.is_done(w->tts.engine)) {
            int step_done = w->tts.step(w->tts.engine);

            /* Drain audio to ring buffer, applying backpressure if full */
            int n = w->tts.get_audio(w->tts.engine, pcm_buf, 4096);
            while (n > 0) {
                while (tts_ring_free(w) < n) {
                    if (atomic_load_explicit(&w->cancel, memory_order_acquire)) goto cancel_break;
                    usleep(500); /* 0.5ms backpressure — let consumer drain */
                }
                tts_ring_write(w, pcm_buf, n);
                n = w->tts.get_audio(w->tts.engine, pcm_buf, 4096);
            }
            cancel_break: (void)0;

            if (step_done) {
                atomic_store_explicit(&w->gen_done, 1, memory_order_release);
            }
        } else {
            /* No work — sleep briefly waiting for wake signal */
            pthread_mutex_lock(&w->wake_mutex);
            if (atomic_load_explicit(&w->text_head, memory_order_acquire) ==
                atomic_load_explicit(&w->text_tail, memory_order_acquire) &&
                w->tts.is_done(w->tts.engine) &&
                atomic_load_explicit(&w->running, memory_order_relaxed)) {
                struct timespec ts;
                clock_gettime(CLOCK_REALTIME, &ts);
                ts.tv_nsec += 10000000; /* 10ms timeout */
                if (ts.tv_nsec >= 1000000000) {
                    ts.tv_sec++;
                    ts.tv_nsec -= 1000000000;
                }
                pthread_cond_timedwait(&w->wake_cond, &w->wake_mutex, &ts);
            }
            pthread_mutex_unlock(&w->wake_mutex);
        }
    }

    return NULL;
}

__attribute__((unused))
static TtsWorker *tts_worker_create(TtsInterface tts) {
    TtsWorker *w = (TtsWorker *)calloc(1, sizeof(TtsWorker));
    if (!w) return NULL;

    w->tts = tts;
    w->audio_ring_size = TTS_RING_SAMPLES;
    w->audio_ring = (float *)calloc(TTS_RING_SAMPLES, sizeof(float));
    if (!w->audio_ring) { free(w); return NULL; }

    pthread_mutex_init(&w->wake_mutex, NULL);
    pthread_cond_init(&w->wake_cond, NULL);
    atomic_store_explicit(&w->running, 1, memory_order_relaxed);
    atomic_store_explicit(&w->gen_done, 1, memory_order_relaxed);

    if (pthread_create(&w->thread, NULL, tts_worker_thread, w) != 0) {
        pthread_mutex_destroy(&w->wake_mutex);
        pthread_cond_destroy(&w->wake_cond);
        free(w->audio_ring);
        free(w);
        return NULL;
    }

    return w;
}

__attribute__((unused))
static void tts_worker_post_text(TtsWorker *w, const char *text) {
    int head = atomic_load_explicit(&w->text_head, memory_order_acquire);
    int next = (head + 1) % TTS_QUEUE_SIZE;
    if (next == atomic_load_explicit(&w->text_tail, memory_order_acquire)) return; /* queue full */

    w->text_queue[head] = strdup(text);
    if (!w->text_queue[head]) return;
    atomic_store_explicit(&w->text_head, next, memory_order_release);
    atomic_store_explicit(&w->gen_done, 0, memory_order_release);

    pthread_mutex_lock(&w->wake_mutex);
    pthread_cond_signal(&w->wake_cond);
    pthread_mutex_unlock(&w->wake_mutex);
}

__attribute__((unused))
static void tts_worker_signal_text_done(TtsWorker *w) {
    atomic_store_explicit(&w->text_done_flag, 1, memory_order_release);
    pthread_mutex_lock(&w->wake_mutex);
    pthread_cond_signal(&w->wake_cond);
    pthread_mutex_unlock(&w->wake_mutex);
}

__attribute__((unused))
static void tts_worker_cancel(TtsWorker *w) {
    /* Signal cancel — the worker thread detects this flag, drains the text
       queue, resets ring positions, and clears state. We must NOT reset ring
       positions here because the worker may be mid-write, causing corruption. */
    atomic_store_explicit(&w->cancel, 1, memory_order_release);

    /* Wake the worker so it processes the cancel immediately */
    pthread_mutex_lock(&w->wake_mutex);
    pthread_cond_signal(&w->wake_cond);
    pthread_mutex_unlock(&w->wake_mutex);
}

__attribute__((unused))
static void tts_worker_destroy(TtsWorker *w) {
    if (!w) return;
    atomic_store_explicit(&w->running, 0, memory_order_release);
    pthread_mutex_lock(&w->wake_mutex);
    pthread_cond_signal(&w->wake_cond);
    pthread_mutex_unlock(&w->wake_mutex);
    pthread_join(w->thread, NULL);

    /* Drain remaining text queue entries */
    for (int i = 0; i < TTS_QUEUE_SIZE; i++)
        free(w->text_queue[i]);

    pthread_mutex_destroy(&w->wake_mutex);
    pthread_cond_destroy(&w->wake_cond);
    free(w->audio_ring);
    free(w);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Pipeline State Machine
 * ═══════════════════════════════════════════════════════════════════════════ */

typedef enum {
    STATE_LISTENING,   /* Waiting for speech */
    STATE_RECORDING,   /* Capturing speech, feeding STT */
    STATE_PROCESSING,  /* Sending to LLM API */
    STATE_STREAMING,   /* Receiving LLM tokens, feeding TTS */
    STATE_SPEAKING,    /* TTS generating and playing audio */
} PipelineState;

static const char *state_names[] = {
    "LISTENING", "RECORDING", "PROCESSING", "STREAMING", "SPEAKING"
};

/* ═══════════════════════════════════════════════════════════════════════════
 * Claude API SSE Client (libcurl)
 *
 * Uses the curl_multi interface for non-blocking SSE streaming.
 * Parses server-sent events to extract text tokens from content_block_delta.
 * ═══════════════════════════════════════════════════════════════════════════ */

#define CLAUDE_MAX_RESPONSE (64 * 1024)
#define CLAUDE_TOKEN_BUF    4096
#define SSE_LINE_BUF        8192

#define CLAUDE_MAX_HISTORY 20  /* Max conversation turns to retain */

typedef struct {
    char *role;     /* "user" or "assistant" */
    char *content;
} ClaudeMessage;

typedef struct {
    /* curl handles */
    CURLM *multi;
    CURL  *easy;
    struct curl_slist *headers;
    char *post_body;  /* Owned JSON body — curl reads from this during perform */

    /* SSE parsing state */
    char line_buf[SSE_LINE_BUF];
    int  line_len;

    /* Accumulated tokens */
    char tokens[CLAUDE_TOKEN_BUF];
    int  tokens_len;
    int  tokens_read;

    /* Full assistant response for history */
    char *response_accum;
    int   response_accum_len;
    int   response_accum_cap;

    /* Conversation history */
    ClaudeMessage history[CLAUDE_MAX_HISTORY];
    int history_len;

    /* State flags */
    bool request_active;
    bool response_done;
    bool error;

    /* Config */
    char api_key[256];
    char model[128];
    char system_prompt[2048];
} ClaudeClient;

static void claude_init(ClaudeClient *c, const char *api_key,
                         const char *model, const char *system_prompt) {
    memset(c, 0, sizeof(*c));
    c->multi = curl_multi_init();
    snprintf(c->api_key, sizeof(c->api_key), "%s", api_key);
    snprintf(c->model, sizeof(c->model), "%s", model ? model : "claude-sonnet-4-20250514");
    snprintf(c->system_prompt, sizeof(c->system_prompt), "%s",
             system_prompt ? system_prompt
                           : "You are a helpful voice assistant. Keep responses concise "
                             "and conversational — aim for 1-3 sentences. Speak naturally.");
    c->response_accum_cap = CLAUDE_MAX_RESPONSE;
    c->response_accum = (char *)malloc((size_t)c->response_accum_cap);
    if (c->response_accum) c->response_accum[0] = '\0';
}

static void claude_cleanup(ClaudeClient *c) {
    if (c->easy) {
        curl_multi_remove_handle(c->multi, c->easy);
        curl_easy_cleanup(c->easy);
        c->easy = NULL;
    }
    if (c->headers) {
        curl_slist_free_all(c->headers);
        c->headers = NULL;
    }
    free(c->post_body);
    c->post_body = NULL;
    free(c->response_accum);
    c->response_accum = NULL;
    for (int i = 0; i < c->history_len; i++) {
        free(c->history[i].role);
        free(c->history[i].content);
    }
    c->history_len = 0;
    if (c->multi) {
        curl_multi_cleanup(c->multi);
        c->multi = NULL;
    }
}

/* Append text to the response accumulator (for conversation history) */
static void claude_accum_response(ClaudeClient *c, const char *text, int len) {
    if (!c->response_accum) return;
    int need = c->response_accum_len + len + 1;
    if (need > c->response_accum_cap) {
        int new_cap = c->response_accum_cap * 2;
        if (new_cap < need) new_cap = need;
        char *tmp = (char *)realloc(c->response_accum, (size_t)new_cap);
        if (!tmp) {
            fprintf(stderr, "[claude] Response accumulator realloc failed (%d bytes)\n", new_cap);
            free(c->response_accum);
            c->response_accum = NULL;
            c->response_accum_len = 0;
            c->response_accum_cap = 0;
            return;
        }
        c->response_accum = tmp;
        c->response_accum_cap = new_cap;
    }
    memcpy(c->response_accum + c->response_accum_len, text, (size_t)len);
    c->response_accum_len += len;
    c->response_accum[c->response_accum_len] = '\0';
}

/* Push a turn into conversation history, evicting oldest if full */
static void claude_push_history(ClaudeClient *c, const char *role, const char *content) {
    if (c->history_len >= CLAUDE_MAX_HISTORY) {
        free(c->history[0].role);
        free(c->history[0].content);
        memmove(&c->history[0], &c->history[1],
                (size_t)(CLAUDE_MAX_HISTORY - 1) * sizeof(ClaudeMessage));
        c->history_len = CLAUDE_MAX_HISTORY - 1;
    }
    c->history[c->history_len].role = strdup(role);
    c->history[c->history_len].content = strdup(content);
    if (!c->history[c->history_len].role || !c->history[c->history_len].content) {
        free(c->history[c->history_len].role);
        free(c->history[c->history_len].content);
        c->history[c->history_len].role = NULL;
        c->history[c->history_len].content = NULL;
        return;
    }
    c->history_len++;
}

/* Parse a single SSE data line and extract text token if present */
static void claude_parse_sse_data(ClaudeClient *c, const char *data) {
    if (strcmp(data, "[DONE]") == 0) {
        c->response_done = true;
        return;
    }

    cJSON *root = cJSON_Parse(data);
    if (!root) return;

    cJSON *type = cJSON_GetObjectItem(root, "type");
    if (type && type->valuestring) {
        if (strcmp(type->valuestring, "content_block_delta") == 0) {
            cJSON *delta = cJSON_GetObjectItem(root, "delta");
            if (delta) {
                cJSON *text = cJSON_GetObjectItem(delta, "text");
                if (text && text->valuestring) {
                    int len = (int)strlen(text->valuestring);

                    /* Accumulate for TTS streaming */
                    int space = CLAUDE_TOKEN_BUF - c->tokens_len - 1;
                    int tts_len = len < space ? len : space;
                    if (tts_len > 0) {
                        memcpy(c->tokens + c->tokens_len, text->valuestring, (size_t)tts_len);
                        c->tokens_len += tts_len;
                        c->tokens[c->tokens_len] = '\0';
                    }

                    /* Accumulate full response for history */
                    claude_accum_response(c, text->valuestring, len);
                }
            }
        } else if (strcmp(type->valuestring, "message_stop") == 0) {
            c->response_done = true;
        } else if (strcmp(type->valuestring, "error") == 0) {
            cJSON *err = cJSON_GetObjectItem(root, "error");
            if (err) {
                cJSON *msg = cJSON_GetObjectItem(err, "message");
                if (msg && msg->valuestring) {
                    fprintf(stderr, "[claude] API error: %s\n", msg->valuestring);
                }
            }
            c->error = true;
            c->response_done = true;
        }
    }

    cJSON_Delete(root);
}

/* curl write callback: accumulates SSE lines and dispatches data events */
static size_t claude_write_cb(char *ptr, size_t size, size_t nmemb, void *userdata) {
    ClaudeClient *c = (ClaudeClient *)userdata;
    size_t total = size * nmemb;

    for (size_t i = 0; i < total; i++) {
        char ch = ptr[i];
        if (ch == '\n') {
            c->line_buf[c->line_len] = '\0';
            /* Parse SSE line */
            if (c->line_len > 6 && strncmp(c->line_buf, "data: ", 6) == 0) {
                claude_parse_sse_data(c, c->line_buf + 6);
            }
            c->line_len = 0;
        } else if (c->line_len < SSE_LINE_BUF - 1) {
            c->line_buf[c->line_len++] = ch;
        }
    }

    return total;
}

/* Commit the last exchange (user + assistant) into history */
static void claude_commit_turn(ClaudeClient *c, const char *user_text) {
    claude_push_history(c, "user", user_text);
    if (c->response_accum && c->response_accum_len > 0) {
        claude_push_history(c, "assistant", c->response_accum);
    }
}

/* Start a streaming request to Claude Messages API */
static int claude_send(ClaudeClient *c, const char *user_text) {
    if (c->easy) {
        curl_multi_remove_handle(c->multi, c->easy);
        curl_easy_cleanup(c->easy);
        c->easy = NULL;
    }
    if (c->headers) {
        curl_slist_free_all(c->headers);
        c->headers = NULL;
    }
    free(c->post_body);
    c->post_body = NULL;

    c->tokens_len = 0;
    c->tokens_read = 0;
    c->line_len = 0;
    c->response_done = false;
    c->error = false;
    c->request_active = true;
    c->response_accum_len = 0;
    if (c->response_accum) c->response_accum[0] = '\0';

    c->easy = curl_easy_init();
    if (!c->easy) return -1;

    /* Build JSON body */
    cJSON *body = cJSON_CreateObject();
    cJSON_AddStringToObject(body, "model", c->model);
    cJSON_AddNumberToObject(body, "max_tokens", 1024);
    cJSON_AddBoolToObject(body, "stream", 1);

    cJSON *system_arr = cJSON_CreateArray();
    cJSON *sys_block = cJSON_CreateObject();
    cJSON_AddStringToObject(sys_block, "type", "text");
    cJSON_AddStringToObject(sys_block, "text", c->system_prompt);
    cJSON_AddItemToArray(system_arr, sys_block);
    cJSON_AddItemToObject(body, "system", system_arr);

    /* Build messages array: history + current user message */
    cJSON *messages = cJSON_CreateArray();
    for (int i = 0; i < c->history_len; i++) {
        cJSON *hmsg = cJSON_CreateObject();
        cJSON_AddStringToObject(hmsg, "role", c->history[i].role);
        cJSON_AddStringToObject(hmsg, "content", c->history[i].content);
        cJSON_AddItemToArray(messages, hmsg);
    }
    cJSON *msg = cJSON_CreateObject();
    cJSON_AddStringToObject(msg, "role", "user");
    cJSON_AddStringToObject(msg, "content", user_text);
    cJSON_AddItemToArray(messages, msg);
    cJSON_AddItemToObject(body, "messages", messages);

    c->post_body = cJSON_PrintUnformatted(body);
    cJSON_Delete(body);

    /* Headers */
    char auth_header[300];
    snprintf(auth_header, sizeof(auth_header), "x-api-key: %s", c->api_key);

    c->headers = curl_slist_append(c->headers, "Content-Type: application/json");
    c->headers = curl_slist_append(c->headers, auth_header);
    c->headers = curl_slist_append(c->headers, "anthropic-version: 2023-06-01");

    curl_easy_setopt(c->easy, CURLOPT_URL, "https://api.anthropic.com/v1/messages");
    curl_easy_setopt(c->easy, CURLOPT_HTTPHEADER, c->headers);
    curl_easy_setopt(c->easy, CURLOPT_POSTFIELDS, c->post_body);
    curl_easy_setopt(c->easy, CURLOPT_WRITEFUNCTION, claude_write_cb);
    curl_easy_setopt(c->easy, CURLOPT_WRITEDATA, c);
    curl_easy_setopt(c->easy, CURLOPT_TIMEOUT, 60L);

    curl_multi_add_handle(c->multi, c->easy);

    return 0;
}

/* Poll for new SSE data. Uses curl_multi_poll to avoid busy-spinning.
 * timeout_ms: max time to wait (0 = non-blocking). Returns new token char count. */
static int claude_poll(ClaudeClient *c, int timeout_ms) {
    if (!c->request_active) return 0;

    int running = 0;
    int prev_len = c->tokens_len;

    curl_multi_perform(c->multi, &running);

    if (timeout_ms > 0) {
        int numfds = 0;
        curl_multi_poll(c->multi, NULL, 0, timeout_ms, &numfds);
        if (numfds > 0) {
            curl_multi_perform(c->multi, &running);
        }
    }

    /* Check for completion */
    int msgs_in_queue;
    CURLMsg *msg;
    while ((msg = curl_multi_info_read(c->multi, &msgs_in_queue))) {
        if (msg->msg == CURLMSG_DONE) {
            if (msg->data.result != CURLE_OK) {
                fprintf(stderr, "[claude] curl error: %s\n",
                        curl_easy_strerror(msg->data.result));
                c->error = true;
            }
            c->response_done = true;
            c->request_active = false;
        }
    }

    return c->tokens_len - prev_len;
}

/* Read available tokens (non-consuming peek returns pointer, length).
 * Call claude_consume_tokens() after processing. */
static const char *claude_peek_tokens(ClaudeClient *c, int *out_len) {
    int avail = c->tokens_len - c->tokens_read;
    if (avail <= 0) {
        *out_len = 0;
        return NULL;
    }
    *out_len = avail;
    return c->tokens + c->tokens_read;
}

static void claude_consume_tokens(ClaudeClient *c, int count) {
    c->tokens_read += count;
    if (c->tokens_read > c->tokens_len) c->tokens_read = c->tokens_len;
}

static void claude_cancel(ClaudeClient *c) {
    if (c->easy) {
        curl_multi_remove_handle(c->multi, c->easy);
        curl_easy_cleanup(c->easy);
        c->easy = NULL;
    }
    if (c->headers) {
        curl_slist_free_all(c->headers);
        c->headers = NULL;
    }
    free(c->post_body);
    c->post_body = NULL;
    c->request_active = false;
    c->response_done = true;
}

/* --- Claude LLM wrappers --- */

static int  llm_claude_send(void *e, const char *t)  { return claude_send((ClaudeClient*)e, t); }
static int  llm_claude_poll(void *e, int ms)          { return claude_poll((ClaudeClient*)e, ms); }
static const char *llm_claude_peek(void *e, int *n)   { return claude_peek_tokens((ClaudeClient*)e, n); }
static void llm_claude_consume(void *e, int n)        { claude_consume_tokens((ClaudeClient*)e, n); }
static void llm_claude_cancel(void *e)                { claude_cancel((ClaudeClient*)e); }
static void llm_claude_commit(void *e, const char *t) { claude_commit_turn((ClaudeClient*)e, t); }
static bool llm_claude_done(void *e)                  { return ((ClaudeClient*)e)->response_done; }
static bool llm_claude_error(void *e)                 { return ((ClaudeClient*)e)->error; }
static void llm_claude_cleanup(void *e)               { claude_cleanup((ClaudeClient*)e); }

static LLMClient llm_create_claude(ClaudeClient *c) {
    return (LLMClient){
        .engine           = c,
        .type             = LLM_ENGINE_CLAUDE,
        .send             = llm_claude_send,
        .poll             = llm_claude_poll,
        .peek_tokens      = llm_claude_peek,
        .consume_tokens   = llm_claude_consume,
        .cancel           = llm_claude_cancel,
        .commit_turn      = llm_claude_commit,
        .is_response_done = llm_claude_done,
        .has_error        = llm_claude_error,
        .cleanup          = llm_claude_cleanup,
    };
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Gemini API SSE Client (libcurl)
 *
 * Same streaming pattern as ClaudeClient but targets Google Gemini API
 * (streamGenerateContent with SSE). Supports conversation history.
 * ═══════════════════════════════════════════════════════════════════════════ */

#define GEMINI_MAX_RESPONSE (64 * 1024)
#define GEMINI_TOKEN_BUF    4096
#define GEMINI_MAX_HISTORY  20

typedef struct {
    char *role;     /* "user" or "model" */
    char *content;
} GeminiMessage;

typedef struct {
    /* curl handles */
    CURLM *multi;
    CURL  *easy;
    struct curl_slist *headers;
    char *post_body;

    /* SSE parsing state */
    char line_buf[SSE_LINE_BUF];
    int  line_len;

    /* Accumulated tokens */
    char tokens[GEMINI_TOKEN_BUF];
    int  tokens_len;
    int  tokens_read;

    /* Full assistant response for history */
    char *response_accum;
    int   response_accum_len;
    int   response_accum_cap;

    /* Conversation history */
    GeminiMessage history[GEMINI_MAX_HISTORY];
    int history_len;

    /* State flags */
    bool request_active;
    bool response_done;
    bool error;

    /* Config */
    char api_key[256];
    char model[128];
    char system_prompt[2048];
    char url_buf[512];
} GeminiClient;

static void gemini_init(GeminiClient *g, const char *api_key,
                         const char *model, const char *system_prompt) {
    memset(g, 0, sizeof(*g));
    g->multi = curl_multi_init();
    snprintf(g->api_key, sizeof(g->api_key), "%s", api_key);
    snprintf(g->model, sizeof(g->model), "%s",
             model ? model : "gemini-2.5-flash");
    snprintf(g->system_prompt, sizeof(g->system_prompt), "%s",
             system_prompt ? system_prompt
                           : "You are a helpful voice assistant. Keep responses concise "
                             "and conversational — aim for 1-3 sentences. Speak naturally.");
    g->response_accum_cap = GEMINI_MAX_RESPONSE;
    g->response_accum = (char *)malloc((size_t)g->response_accum_cap);
    if (g->response_accum) g->response_accum[0] = '\0';
}

static void gemini_cleanup(GeminiClient *g) {
    if (g->easy) {
        curl_multi_remove_handle(g->multi, g->easy);
        curl_easy_cleanup(g->easy);
        g->easy = NULL;
    }
    if (g->headers) {
        curl_slist_free_all(g->headers);
        g->headers = NULL;
    }
    free(g->post_body);
    g->post_body = NULL;
    free(g->response_accum);
    g->response_accum = NULL;
    for (int i = 0; i < g->history_len; i++) {
        free(g->history[i].role);
        free(g->history[i].content);
    }
    g->history_len = 0;
    if (g->multi) {
        curl_multi_cleanup(g->multi);
        g->multi = NULL;
    }
}

static void gemini_accum_response(GeminiClient *g, const char *text, int len) {
    if (!g->response_accum) return;
    int need = g->response_accum_len + len + 1;
    if (need > g->response_accum_cap) {
        int new_cap = g->response_accum_cap * 2;
        if (new_cap < need) new_cap = need;
        char *tmp = (char *)realloc(g->response_accum, (size_t)new_cap);
        if (!tmp) {
            fprintf(stderr, "[gemini] Response accumulator realloc failed (%d bytes)\n", new_cap);
            return;
        }
        g->response_accum = tmp;
        g->response_accum_cap = new_cap;
    }
    memcpy(g->response_accum + g->response_accum_len, text, (size_t)len);
    g->response_accum_len += len;
    g->response_accum[g->response_accum_len] = '\0';
}

static void gemini_push_history(GeminiClient *g, const char *role, const char *content) {
    if (g->history_len >= GEMINI_MAX_HISTORY) {
        free(g->history[0].role);
        free(g->history[0].content);
        memmove(&g->history[0], &g->history[1],
                (size_t)(GEMINI_MAX_HISTORY - 1) * sizeof(GeminiMessage));
        g->history_len = GEMINI_MAX_HISTORY - 1;
    }
    g->history[g->history_len].role = strdup(role);
    g->history[g->history_len].content = strdup(content);
    if (!g->history[g->history_len].role || !g->history[g->history_len].content) {
        free(g->history[g->history_len].role);
        free(g->history[g->history_len].content);
        g->history[g->history_len].role = NULL;
        g->history[g->history_len].content = NULL;
        return;
    }
    g->history_len++;
}

static void gemini_parse_sse_data(GeminiClient *g, const char *data) {
    cJSON *root = cJSON_Parse(data);
    if (!root) return;

    cJSON *candidates = cJSON_GetObjectItem(root, "candidates");
    if (candidates && cJSON_GetArraySize(candidates) > 0) {
        cJSON *cand = cJSON_GetArrayItem(candidates, 0);

        cJSON *content = cJSON_GetObjectItem(cand, "content");
        if (content) {
            cJSON *parts = cJSON_GetObjectItem(content, "parts");
            if (parts && cJSON_GetArraySize(parts) > 0) {
                cJSON *part = cJSON_GetArrayItem(parts, 0);
                cJSON *text = cJSON_GetObjectItem(part, "text");
                if (text && text->valuestring) {
                    int len = (int)strlen(text->valuestring);

                    int space = GEMINI_TOKEN_BUF - g->tokens_len - 1;
                    int tts_len = len < space ? len : space;
                    if (tts_len > 0) {
                        memcpy(g->tokens + g->tokens_len, text->valuestring, (size_t)tts_len);
                        g->tokens_len += tts_len;
                        g->tokens[g->tokens_len] = '\0';
                    }

                    gemini_accum_response(g, text->valuestring, len);
                }
            }
        }

        cJSON *finish = cJSON_GetObjectItem(cand, "finishReason");
        if (finish && finish->valuestring && finish->valuestring[0] != '\0') {
            /* Any non-empty finishReason is terminal (STOP, MAX_TOKENS,
               SAFETY, RECITATION, etc.) — not just "STOP" */
            g->response_done = true;
        }
    }

    cJSON *err_obj = cJSON_GetObjectItem(root, "error");
    if (err_obj) {
        cJSON *msg = cJSON_GetObjectItem(err_obj, "message");
        if (msg && msg->valuestring) {
            fprintf(stderr, "[gemini] API error: %s\n", msg->valuestring);
        }
        g->error = true;
        g->response_done = true;
    }

    cJSON_Delete(root);
}

static size_t gemini_write_cb(char *ptr, size_t size, size_t nmemb, void *userdata) {
    GeminiClient *g = (GeminiClient *)userdata;
    size_t total = size * nmemb;

    for (size_t i = 0; i < total; i++) {
        char ch = ptr[i];
        if (ch == '\n') {
            g->line_buf[g->line_len] = '\0';
            if (g->line_len > 6 && strncmp(g->line_buf, "data: ", 6) == 0) {
                gemini_parse_sse_data(g, g->line_buf + 6);
            }
            g->line_len = 0;
        } else if (g->line_len < SSE_LINE_BUF - 1) {
            g->line_buf[g->line_len++] = ch;
        }
    }

    return total;
}

static void gemini_commit_turn(GeminiClient *g, const char *user_text) {
    gemini_push_history(g, "user", user_text);
    if (g->response_accum && g->response_accum_len > 0) {
        gemini_push_history(g, "model", g->response_accum);
    }
}

static int gemini_send(GeminiClient *g, const char *user_text) {
    if (g->easy) {
        curl_multi_remove_handle(g->multi, g->easy);
        curl_easy_cleanup(g->easy);
        g->easy = NULL;
    }
    if (g->headers) {
        curl_slist_free_all(g->headers);
        g->headers = NULL;
    }
    free(g->post_body);
    g->post_body = NULL;

    g->tokens_len = 0;
    g->tokens_read = 0;
    g->line_len = 0;
    g->response_done = false;
    g->error = false;
    g->request_active = true;
    g->response_accum_len = 0;
    if (g->response_accum) g->response_accum[0] = '\0';

    g->easy = curl_easy_init();
    if (!g->easy) return -1;

    /* Build JSON body */
    cJSON *body = cJSON_CreateObject();

    /* System instruction */
    cJSON *sys_inst = cJSON_CreateObject();
    cJSON *sys_parts = cJSON_CreateArray();
    cJSON *sys_part = cJSON_CreateObject();
    cJSON_AddStringToObject(sys_part, "text", g->system_prompt);
    cJSON_AddItemToArray(sys_parts, sys_part);
    cJSON_AddItemToObject(sys_inst, "parts", sys_parts);
    cJSON_AddItemToObject(body, "systemInstruction", sys_inst);

    /* Contents: history + current message */
    cJSON *contents = cJSON_CreateArray();
    for (int i = 0; i < g->history_len; i++) {
        cJSON *turn = cJSON_CreateObject();
        cJSON_AddStringToObject(turn, "role", g->history[i].role);
        cJSON *parts = cJSON_CreateArray();
        cJSON *part = cJSON_CreateObject();
        cJSON_AddStringToObject(part, "text", g->history[i].content);
        cJSON_AddItemToArray(parts, part);
        cJSON_AddItemToObject(turn, "parts", parts);
        cJSON_AddItemToArray(contents, turn);
    }

    cJSON *user_turn = cJSON_CreateObject();
    cJSON_AddStringToObject(user_turn, "role", "user");
    cJSON *user_parts = cJSON_CreateArray();
    cJSON *user_part = cJSON_CreateObject();
    cJSON_AddStringToObject(user_part, "text", user_text);
    cJSON_AddItemToArray(user_parts, user_part);
    cJSON_AddItemToObject(user_turn, "parts", user_parts);
    cJSON_AddItemToArray(contents, user_turn);
    cJSON_AddItemToObject(body, "contents", contents);

    /* Generation config */
    cJSON *gen_config = cJSON_CreateObject();
    cJSON_AddNumberToObject(gen_config, "maxOutputTokens", 1024);
    cJSON_AddNumberToObject(gen_config, "temperature", 0.7);
    cJSON_AddItemToObject(body, "generationConfig", gen_config);

    g->post_body = cJSON_PrintUnformatted(body);
    cJSON_Delete(body);

    /* URL with model and API key */
    snprintf(g->url_buf, sizeof(g->url_buf),
             "https://generativelanguage.googleapis.com/v1beta/models/%s"
             ":streamGenerateContent?alt=sse&key=%s",
             g->model, g->api_key);

    g->headers = curl_slist_append(g->headers, "Content-Type: application/json");

    curl_easy_setopt(g->easy, CURLOPT_URL, g->url_buf);
    curl_easy_setopt(g->easy, CURLOPT_HTTPHEADER, g->headers);
    curl_easy_setopt(g->easy, CURLOPT_POSTFIELDS, g->post_body);
    curl_easy_setopt(g->easy, CURLOPT_WRITEFUNCTION, gemini_write_cb);
    curl_easy_setopt(g->easy, CURLOPT_WRITEDATA, g);
    curl_easy_setopt(g->easy, CURLOPT_TIMEOUT, 60L);

    curl_multi_add_handle(g->multi, g->easy);

    return 0;
}

static int gemini_poll(GeminiClient *g, int timeout_ms) {
    if (!g->request_active) return 0;

    int running = 0;
    int prev_len = g->tokens_len;

    curl_multi_perform(g->multi, &running);

    if (timeout_ms > 0) {
        int numfds = 0;
        curl_multi_poll(g->multi, NULL, 0, timeout_ms, &numfds);
        if (numfds > 0) {
            curl_multi_perform(g->multi, &running);
        }
    }

    int msgs_in_queue;
    CURLMsg *msg;
    while ((msg = curl_multi_info_read(g->multi, &msgs_in_queue))) {
        if (msg->msg == CURLMSG_DONE) {
            if (msg->data.result != CURLE_OK) {
                fprintf(stderr, "[gemini] curl error: %s\n",
                        curl_easy_strerror(msg->data.result));
                g->error = true;
            }
            g->response_done = true;
            g->request_active = false;
        }
    }

    return g->tokens_len - prev_len;
}

static const char *gemini_peek_tokens(GeminiClient *g, int *out_len) {
    int avail = g->tokens_len - g->tokens_read;
    if (avail <= 0) {
        *out_len = 0;
        return NULL;
    }
    *out_len = avail;
    return g->tokens + g->tokens_read;
}

static void gemini_consume_tokens(GeminiClient *g, int count) {
    g->tokens_read += count;
    if (g->tokens_read > g->tokens_len) g->tokens_read = g->tokens_len;
}

static void gemini_cancel(GeminiClient *g) {
    if (g->easy) {
        curl_multi_remove_handle(g->multi, g->easy);
        curl_easy_cleanup(g->easy);
        g->easy = NULL;
    }
    if (g->headers) {
        curl_slist_free_all(g->headers);
        g->headers = NULL;
    }
    free(g->post_body);
    g->post_body = NULL;
    g->request_active = false;
    g->response_done = true;
}

/* --- Gemini LLM wrappers --- */

static int  llm_gemini_send(void *e, const char *t)  { return gemini_send((GeminiClient*)e, t); }
static int  llm_gemini_poll(void *e, int ms)          { return gemini_poll((GeminiClient*)e, ms); }
static const char *llm_gemini_peek(void *e, int *n)   { return gemini_peek_tokens((GeminiClient*)e, n); }
static void llm_gemini_consume(void *e, int n)        { gemini_consume_tokens((GeminiClient*)e, n); }
static void llm_gemini_cancel(void *e)                { gemini_cancel((GeminiClient*)e); }
static void llm_gemini_commit(void *e, const char *t) { gemini_commit_turn((GeminiClient*)e, t); }
static bool llm_gemini_done(void *e)                  { return ((GeminiClient*)e)->response_done; }
static bool llm_gemini_error(void *e)                 { return ((GeminiClient*)e)->error; }
static void llm_gemini_cleanup(void *e)               { gemini_cleanup((GeminiClient*)e); }

static LLMClient llm_create_gemini(GeminiClient *g) {
    return (LLMClient){
        .engine           = g,
        .type             = LLM_ENGINE_GEMINI,
        .send             = llm_gemini_send,
        .poll             = llm_gemini_poll,
        .peek_tokens      = llm_gemini_peek,
        .consume_tokens   = llm_gemini_consume,
        .cancel           = llm_gemini_cancel,
        .commit_turn      = llm_gemini_commit,
        .is_response_done = llm_gemini_done,
        .has_error        = llm_gemini_error,
        .cleanup          = llm_gemini_cleanup,
    };
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Local LLM Client (on-device Llama via pocket_llm cdylib)
 *
 * Runs inference entirely on-device using Metal GPU. No network needed.
 * Supports Llama 3.2 1B/3B Instruct models from HuggingFace.
 * ═══════════════════════════════════════════════════════════════════════════ */

typedef struct {
    void       *llm;
    const char *system_prompt;
    char        token_buf[4096];
    int         token_len;
    int         response_done;
    int         error;
} LocalLLMClient;

static void local_llm_init(LocalLLMClient *c, const char *model_id, const char *system_prompt) {
    memset(c, 0, sizeof(*c));
    c->system_prompt = system_prompt ? system_prompt : "You are a helpful voice assistant. Keep responses concise and conversational.";
    c->llm = pocket_llm_create(model_id, NULL);
    if (!c->llm) {
        fprintf(stderr, "[local_llm] Failed to load model: %s\n", model_id);
        c->error = 1;
    }
}

static int llm_local_send(void *e, const char *user_text) {
    LocalLLMClient *c = (LocalLLMClient *)e;
    if (!c->llm) return -1;
    c->token_len = 0;
    c->token_buf[0] = '\0';
    c->response_done = 0;
    c->error = 0;
    return pocket_llm_set_prompt(c->llm, c->system_prompt, user_text);
}

static int llm_local_poll(void *e, int timeout_ms) {
    (void)timeout_ms;
    LocalLLMClient *c = (LocalLLMClient *)e;
    if (!c->llm || c->response_done) return 0;

    int tok = pocket_llm_step(c->llm);
    if (tok < 0) { c->error = 1; c->response_done = 1; return -1; }
    if (tok == 0) { c->response_done = 1; return 0; }

    char piece[256];
    int n = pocket_llm_get_token(c->llm, piece, sizeof(piece));
    if (n > 0 && c->token_len + n < (int)sizeof(c->token_buf) - 1) {
        memcpy(c->token_buf + c->token_len, piece, n);
        c->token_len += n;
        c->token_buf[c->token_len] = '\0';
    }
    return n;
}

static const char *llm_local_peek(void *e, int *out_len) {
    LocalLLMClient *c = (LocalLLMClient *)e;
    if (out_len) *out_len = c->token_len;
    return c->token_buf;
}

static void llm_local_consume(void *e, int count) {
    LocalLLMClient *c = (LocalLLMClient *)e;
    if (count >= c->token_len) {
        c->token_len = 0;
        c->token_buf[0] = '\0';
    } else {
        memmove(c->token_buf, c->token_buf + count, c->token_len - count);
        c->token_len -= count;
        c->token_buf[c->token_len] = '\0';
    }
}

static void llm_local_cancel(void *e) {
    LocalLLMClient *c = (LocalLLMClient *)e;
    if (c->llm) pocket_llm_reset(c->llm);
    c->response_done = 1;
    c->token_len = 0;
    c->token_buf[0] = '\0';
}

static void llm_local_commit(void *e, const char *user_text) { (void)e; (void)user_text; }
static bool llm_local_done(void *e) { return ((LocalLLMClient *)e)->response_done; }
static bool llm_local_error(void *e) { return ((LocalLLMClient *)e)->error; }
static void llm_local_cleanup(void *e) {
    LocalLLMClient *c = (LocalLLMClient *)e;
    if (c->llm) pocket_llm_destroy(c->llm);
    c->llm = NULL;
}

static LLMClient llm_create_local(LocalLLMClient *c) {
    return (LLMClient){
        .engine           = c,
        .type             = LLM_ENGINE_LOCAL,
        .send             = llm_local_send,
        .poll             = llm_local_poll,
        .peek_tokens      = llm_local_peek,
        .consume_tokens   = llm_local_consume,
        .cancel           = llm_local_cancel,
        .commit_turn      = llm_local_commit,
        .is_response_done = llm_local_done,
        .has_error        = llm_local_error,
        .cleanup          = llm_local_cleanup,
    };
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Prosody-Aware System Prompt
 *
 * When prosody mode is enabled, the LLM outputs SSML-annotated text that
 * flows through the existing ssml_parser → prosody pipeline. This is the
 * "strong prompting" technique that benchmarks show gives +10-15% quality.
 * ═══════════════════════════════════════════════════════════════════════════ */

static const char *PROSODY_SYSTEM_PROMPT =
    "You are a helpful voice assistant. Keep responses concise and "
    "conversational — aim for 1-3 sentences. Speak naturally.\n\n"
    "Your responses are spoken aloud via TTS with SSML support. "
    "Enhance delivery naturally when it matters:\n"
    "- <emphasis level=\"strong\">word</emphasis> for key words\n"
    "- <break time=\"200ms\"/> at natural pauses in complex sentences\n"
    "- <emotion type=\"happy\">text</emotion> for joyful or warm moments\n"
    "- <emotion type=\"serious\">text</emotion> for important or thoughtful content\n"
    "- <emotion type=\"excited\">text</emotion> for enthusiasm\n"
    "- <prosody rate=\"90%\" pitch=\"+10%\">text</prosody> for fine-grained control\n\n"
    "Available emotions: happy, excited, sad, angry, surprised, warm, "
    "serious, calm, confident. Use sparingly.\n\n"
    "Rules:\n"
    "- Most text should be plain — only add tags where emotion genuinely helps.\n"
    "- For questions, let punctuation guide natural rising intonation.\n"
    "- Spell out URLs, formulas, and non-obvious abbreviations in words.\n"
    "- Never wrap every sentence in tags — it sounds robotic.\n"
    "- Prefer <emotion> over <prosody> — it's simpler and more natural.";

/* ═══════════════════════════════════════════════════════════════════════════
 * Configuration
 * ═══════════════════════════════════════════════════════════════════════════ */

typedef struct {
    const char *voice;
    const char *stt_repo;
    const char *stt_model;
    const char *tts_repo;
    const char *llm_model;
    const char *system_prompt;
    int         prosody_prompt;  /* 1 = use SSML-aware system prompt */

    /* LLM engine selection */
    LLMEngineType llm_engine;
    int n_q;
    int enable_vad;
    float vad_threshold;

    /* STT engine selection */
    SttEngineType stt_engine;
    const char *cstt_model;     /* Path to .cstt model file (Conformer engine) */
    const char *bnns_model;     /* Path to .mlmodelc for BNNS/ANE (optional) */
    const char *metallib_path;  /* Path to .metallib for custom GPU kernels */

    /* CTC beam search + LM */
    int beam_size;              /* Beam size for CTC beam search (0 = greedy) */
    const char *lm_path;        /* Path to KenLM .bin or .arpa language model */
    float lm_weight;            /* LM score weight (default: 1.5) */
    float word_score;           /* Per-word insertion bonus (default: 0.0) */

    /* Latency profiler */
    int enable_profiler;        /* 1 = print per-turn latency breakdown */

    /* TTS engine selection */
    TtsEngineType tts_engine;
    const char *ctts_model;     /* Path to .ctts model file (C engine) */
    const char *ctts_voice;     /* Path to .voicekv voice conditioning file */

    /* Sonata engine paths */
    const char *sonata_lm_weights;   /* Path to sonata_lm.safetensors */
    const char *sonata_lm_config;    /* Path to sonata_lm.json */
    const char *sonata_tokenizer;    /* Path to SentencePiece tokenizer.model */
    const char *sonata_flow_weights; /* Path to sonata_flow.safetensors */
    const char *sonata_flow_config;  /* Path to sonata_flow_config.json */
    const char *sonata_dec_weights;  /* Path to sonata_decoder.safetensors */
    const char *sonata_dec_config;   /* Path to sonata_decoder_config.json */
    const char *sonata_flow_v2_weights; /* Path to flow_v2.safetensors */
    const char *sonata_flow_v2_config;  /* Path to flow_v2_config.json */
    /* Sonata v3 (flow + vocoder) */
    const char *flow_v3_weights;     /* --flow-v3-weights PATH */
    const char *flow_v3_config;      /* --flow-v3-config PATH */
    const char *vocoder_weights;     /* --vocoder-weights PATH */
    const char *vocoder_config;      /* --vocoder-config PATH */
    int         sonata_speaker;      /* Speaker ID for multi-voice (-1 = default) */
    float       sonata_cfg_scale;    /* Classifier-free guidance scale (1.0 = off, 2.0+ = stronger) */
    int         sonata_flow_steps;   /* ODE steps for flow (4=fast, 8=default, 16=quality) */
    int         sonata_heun;         /* 1 = Heun's 2nd-order solver, 0 = Euler */
    int         tts_quality_mode;    /* TTS quality mode: 0=FAST, 1=BALANCED (default), 2=HIGH */
    int         tts_first_chunk_fast; /* 1 = use FAST mode for first sentence, BALANCED for rest */
    const char *sonata_draft_weights; /* Draft model for speculative decoding (optional) */
    const char *sonata_draft_config;  /* Draft model config (optional) */
    int         sonata_speculate_k;   /* Tokens to speculate per step (default: 5) */
    int         sonata_self_draft;    /* 1 = reuse LM weights as 4-layer draft model */
    const char *sonata_ref_wav;       /* Reference WAV for voice cloning (optional) */

    /* SoundStorm parallel decoder (alternative to AR LM) */
    const char *sonata_storm_weights; /* Path to sonata_storm.safetensors */
    const char *sonata_storm_config;  /* Path to sonata_storm_config.json */

    /* Phonemizer (espeak-ng IPA) */
    int         use_phonemizer;       /* --phonemize: route text through espeak-ng IPA */
    const char *phoneme_map_path;     /* --phoneme-map: JSON phoneme-to-ID mapping */
    const char *pronunciation_dict_path; /* --pronunciation-dict: JSON pronunciation overrides */

    /* Speaker encoder / voice cloning */
    const char *speaker_encoder_path;  /* --speaker-encoder: ONNX speaker encoder model */
    const char *ref_wav_path;          /* --ref-wav: reference audio for voice cloning */
    const char *clone_voice_path;      /* --clone-voice: one-stop voice cloning from WAV */

    /* Sonata STT */
    const char *sonata_stt_model;      /* --sonata-stt-model: path to .cstt_sonata weight file */
    const char *sonata_refiner_path;   /* --sonata-refiner: optional .cref refiner for two-pass */

    /* Audio post-processing */
    const char *silero_vad_path;  /* deprecated — kept for config compat */
    const char *native_vad_path; /* --vad: path to .nvad native weights */
    const char *semantic_eou_path; /* --semantic-eou: path to .seou weights */
    const char *emosteer_path;   /* --emosteer: path to emotion directions JSON */
    const char *prosody_log_path; /* --prosody-log: JSONL prosody log file */

    /* Conversation memory (scaffolding) */
    const char *memory_path;
    int         memory_max_turns;
    int         memory_max_tokens;

    /* Speaker diarization (scaffolding) */
    const char *diarizer_encoder;
    float       diarizer_threshold;
    int         diarizer_max_speakers;

    /* Backchannel generation (scaffolding) */
    int         backchannel;
    float pitch;        /* Pitch multiplier (1.0 = no change) */
    float volume_db;    /* Volume in dB (0.0 = no change) */
    int   hw_resample;  /* 1 = AudioConverter, 0 = FIR fallback */
    int   spatial;      /* 1 = enable 3D spatial audio */
    float spatial_az;   /* Azimuth for voice source (degrees) */

    /* Opus output */
    int   opus_bitrate;     /* Opus bitrate in bps (0 = disabled) */
    const char *opus_output; /* Path for Opus output file (NULL = disabled) */

    /* Sentence buffering */
    int   sentbuf_mode;     /* SENTBUF_MODE_SENTENCE or SENTBUF_MODE_SPECULATIVE */
    int   sentbuf_min_words; /* Min words for speculative mode clause flush */

    /* Remote microphone (phone-as-mic via WebSocket) */
    int   remote_mic;       /* 1 = start web remote mic server */
    int   remote_port;      /* TCP port for web remote (default 8088) */

    /* Config file */
    const char *config_file;  /* Path to JSON config file */

    /* HTTP API server mode */
    int   server_mode;       /* 1 = run as HTTP API server instead of interactive */
    int   server_port;       /* TCP port for HTTP API (default: 8080) */
} PipelineConfig;

static PipelineConfig default_config(void) {
    return (PipelineConfig){
        .voice        = NULL,
        .stt_repo     = "kyutai/stt-1b-en_fr-candle",
        .stt_model    = "model.safetensors",
        .tts_repo     = "kyutai/tts-1.6b-en_fr",
        .llm_model    = NULL,
        .system_prompt = NULL,
        .prosody_prompt = 0,
        .llm_engine   = LLM_ENGINE_CLAUDE,
        .n_q          = 24,
        .enable_vad   = 1,
        .vad_threshold = 0.7f,
        .stt_engine   = STT_ENGINE_RUST,
        .cstt_model   = NULL,
        .bnns_model   = NULL,
        .metallib_path = NULL,
        .beam_size    = 0,
        .lm_path      = NULL,
        .lm_weight    = 1.5f,
        .word_score   = 0.0f,
        .enable_profiler = 0,
        .tts_engine   = TTS_ENGINE_SONATA,
        .ctts_model   = NULL,
        .ctts_voice   = NULL,
        .sonata_lm_weights = NULL,
        .sonata_lm_config  = NULL,
        .sonata_tokenizer  = NULL,
        .sonata_flow_weights = NULL,
        .sonata_flow_config  = NULL,
        .sonata_dec_weights  = NULL,
        .sonata_dec_config   = NULL,
        .sonata_flow_v2_weights = NULL,
        .sonata_flow_v2_config  = NULL,
        .flow_v3_weights    = NULL,
        .flow_v3_config     = NULL,
        .vocoder_weights    = NULL,
        .vocoder_config     = NULL,
        .sonata_speaker      = -1,
        .sonata_cfg_scale    = 1.5f,
        .sonata_flow_steps   = 8,
        .sonata_heun         = 0,
        .tts_quality_mode    = 1,   /* BALANCED */
        .tts_first_chunk_fast = 1,  /* Enabled by default */
        .sonata_draft_weights = NULL,
        .sonata_draft_config  = NULL,
        .sonata_speculate_k   = 5,
        .sonata_self_draft    = 0,
        .sonata_ref_wav       = NULL,
        .sonata_storm_weights = NULL,
        .sonata_storm_config  = NULL,
        .use_phonemizer       = 0,
        .phoneme_map_path     = NULL,
        .pronunciation_dict_path = NULL,
        .speaker_encoder_path = NULL,
        .ref_wav_path         = NULL,
        .clone_voice_path     = NULL,
        .sonata_stt_model     = NULL,
        .sonata_refiner_path  = NULL,
        .silero_vad_path      = NULL,
        .native_vad_path      = NULL,
        .semantic_eou_path    = NULL,
        .emosteer_path        = NULL,
        .prosody_log_path     = NULL,
        .pitch        = 1.0f,
        .volume_db    = 0.0f,
        .hw_resample  = 1,
        .spatial      = 0,
        .spatial_az   = 0.0f,
        .opus_bitrate  = 0,
        .opus_output   = NULL,
        .sentbuf_mode  = SENTBUF_MODE_SPECULATIVE,
        .sentbuf_min_words = 5,
        .remote_mic    = 0,
        .remote_port   = 8088,
        .config_file   = NULL,
        .server_mode   = 0,
        .server_port   = 8080,
        .memory_path        = NULL,
        .memory_max_turns   = 50,
        .memory_max_tokens  = 4000,
        .diarizer_encoder   = NULL,
        .diarizer_threshold = 0.5f,
        .diarizer_max_speakers = 4,
        .backchannel        = 0,
    };
}

/* ─── JSON Config File Loader ─────────────────────────────────────────────
 * Reads a JSON config file and applies values to PipelineConfig.
 * CLI arguments take priority (applied after config file). */

static const char *json_str(cJSON *obj, const char *key, const char *fallback) {
    cJSON *v = cJSON_GetObjectItemCaseSensitive(obj, key);
    return (v && cJSON_IsString(v)) ? v->valuestring : fallback;
}

static int json_int(cJSON *obj, const char *key, int fallback) {
    cJSON *v = cJSON_GetObjectItemCaseSensitive(obj, key);
    return (v && cJSON_IsNumber(v)) ? v->valueint : fallback;
}

static float json_float(cJSON *obj, const char *key, float fallback) {
    cJSON *v = cJSON_GetObjectItemCaseSensitive(obj, key);
    return (v && cJSON_IsNumber(v)) ? (float)v->valuedouble : fallback;
}

static int json_bool(cJSON *obj, const char *key, int fallback) {
    cJSON *v = cJSON_GetObjectItemCaseSensitive(obj, key);
    return v ? (cJSON_IsTrue(v) ? 1 : 0) : fallback;
}

static void load_config_file(PipelineConfig *cfg, const char *path) {
    FILE *f = fopen(path, "r");
    if (!f) {
        fprintf(stderr, "[config] Cannot open %s: %s\n", path, strerror(errno));
        return;
    }
    fseek(f, 0, SEEK_END);
    long len = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (len <= 0) {
        if (len < 0) fprintf(stderr, "[config] ftell failed for %s\n", path);
        fclose(f);
        return;
    }
    char *data = malloc((size_t)len + 1);
    if (!data) { fclose(f); return; }
    size_t nread = fread(data, 1, (size_t)len, f);
    fclose(f);
    if (nread != (size_t)len) { free(data); return; }
    data[len] = '\0';

    cJSON *root = cJSON_Parse(data);
    free(data);
    if (!root) {
        fprintf(stderr, "[config] JSON parse error near: %s\n",
                cJSON_GetErrorPtr() ? cJSON_GetErrorPtr() : "unknown");
        return;
    }

    /* STT */
    cJSON *stt = cJSON_GetObjectItemCaseSensitive(root, "stt");
    if (stt) {
        const char *eng = json_str(stt, "engine", NULL);
        if (eng) {
            if (strcmp(eng, "conformer") == 0) cfg->stt_engine = STT_ENGINE_CONFORMER;
            else if (strcmp(eng, "bnns") == 0) cfg->stt_engine = STT_ENGINE_BNNS;
            else if (strcmp(eng, "sonata") == 0) cfg->stt_engine = STT_ENGINE_SONATA;
            else cfg->stt_engine = STT_ENGINE_RUST;
        }
        cfg->cstt_model = json_str(stt, "cstt_model", cfg->cstt_model);
        cfg->sonata_stt_model = json_str(stt, "sonata_stt_model",
            json_str(stt, "sonata_model", cfg->sonata_stt_model));
        cfg->sonata_refiner_path = json_str(stt, "sonata_refiner", cfg->sonata_refiner_path);
        cfg->bnns_model = json_str(stt, "bnns_model", cfg->bnns_model);
        cfg->beam_size = json_int(stt, "beam_size", cfg->beam_size);
        cfg->lm_path = json_str(stt, "lm_path", cfg->lm_path);
        cfg->lm_weight = json_float(stt, "lm_weight", cfg->lm_weight);
        cfg->word_score = json_float(stt, "word_score", cfg->word_score);
    }

    /* TTS */
    cJSON *tts = cJSON_GetObjectItemCaseSensitive(root, "tts");
    if (tts) {
        const char *eng = json_str(tts, "engine", NULL);
        if (eng) {
            if (strcmp(eng, "sonata") == 0) cfg->tts_engine = TTS_ENGINE_SONATA;
            else if (strcmp(eng, "sonata-v2") == 0) cfg->tts_engine = TTS_ENGINE_SONATA_V2;
            else if (strcmp(eng, "sonata-v3") == 0) cfg->tts_engine = TTS_ENGINE_SONATA_V3;
        }
        cfg->use_phonemizer = json_bool(tts, "phonemize", cfg->use_phonemizer);
        cfg->phoneme_map_path = json_str(tts, "phoneme_map", cfg->phoneme_map_path);
        cfg->pronunciation_dict_path = json_str(tts, "pronunciation_dict", cfg->pronunciation_dict_path);
        cfg->speaker_encoder_path = json_str(tts, "speaker_encoder", cfg->speaker_encoder_path);
        cfg->ref_wav_path = json_str(tts, "ref_wav", cfg->ref_wav_path);
    }

    /* Sonata TTS */
    cJSON *sonata = cJSON_GetObjectItemCaseSensitive(root, "sonata");
    if (sonata) {
        cfg->sonata_lm_weights = json_str(sonata, "lm_weights", cfg->sonata_lm_weights);
        cfg->sonata_lm_config = json_str(sonata, "lm_config", cfg->sonata_lm_config);
        cfg->sonata_tokenizer = json_str(sonata, "tokenizer", cfg->sonata_tokenizer);
        cfg->sonata_flow_weights = json_str(sonata, "flow_weights", cfg->sonata_flow_weights);
        cfg->sonata_flow_config = json_str(sonata, "flow_config", cfg->sonata_flow_config);
        cfg->sonata_dec_weights = json_str(sonata, "decoder_weights", cfg->sonata_dec_weights);
        cfg->sonata_dec_config = json_str(sonata, "decoder_config", cfg->sonata_dec_config);
        cfg->sonata_flow_v2_weights = json_str(sonata, "flow_v2_weights", cfg->sonata_flow_v2_weights);
        cfg->sonata_flow_v2_config = json_str(sonata, "flow_v2_config", cfg->sonata_flow_v2_config);
        cfg->flow_v3_weights = json_str(sonata, "flow_v3_weights", cfg->flow_v3_weights);
        cfg->flow_v3_config = json_str(sonata, "flow_v3_config", cfg->flow_v3_config);
        cfg->vocoder_weights = json_str(sonata, "vocoder_weights", cfg->vocoder_weights);
        cfg->vocoder_config = json_str(sonata, "vocoder_config", cfg->vocoder_config);
        cfg->sonata_speaker = json_int(sonata, "speaker", cfg->sonata_speaker);
        cfg->sonata_cfg_scale = json_float(sonata, "cfg_scale", cfg->sonata_cfg_scale);
        cfg->sonata_flow_steps = json_int(sonata, "flow_steps", cfg->sonata_flow_steps);
        cfg->sonata_heun = json_bool(sonata, "heun", cfg->sonata_heun);
        cfg->sonata_speculate_k = json_int(sonata, "speculate_k", cfg->sonata_speculate_k);
        cfg->sonata_self_draft = json_bool(sonata, "self_draft", cfg->sonata_self_draft);
        cfg->sonata_draft_weights = json_str(sonata, "draft_weights", cfg->sonata_draft_weights);
        cfg->sonata_draft_config = json_str(sonata, "draft_config", cfg->sonata_draft_config);
        cfg->sonata_ref_wav = json_str(sonata, "ref_wav", cfg->sonata_ref_wav);
        cfg->sonata_storm_weights = json_str(sonata, "storm_weights", cfg->sonata_storm_weights);
        cfg->sonata_storm_config = json_str(sonata, "storm_config", cfg->sonata_storm_config);
    }

    /* Sonata v3 (flow + vocoder) — separate block for v3-specific options */
    cJSON *sonata_v3 = cJSON_GetObjectItemCaseSensitive(root, "sonata_v3");
    if (sonata_v3) {
        cfg->flow_v3_weights = json_str(sonata_v3, "flow_weights",
            json_str(sonata_v3, "flow_v3_weights", cfg->flow_v3_weights));
        cfg->flow_v3_config = json_str(sonata_v3, "flow_config",
            json_str(sonata_v3, "flow_v3_config", cfg->flow_v3_config));
        cfg->vocoder_weights = json_str(sonata_v3, "vocoder_weights", cfg->vocoder_weights);
        cfg->vocoder_config = json_str(sonata_v3, "vocoder_config", cfg->vocoder_config);
        cfg->sonata_flow_steps = json_int(sonata_v3, "n_steps",
            json_int(sonata_v3, "flow_steps", cfg->sonata_flow_steps));
        cfg->sonata_cfg_scale = json_float(sonata_v3, "cfg_scale", cfg->sonata_cfg_scale);
        cfg->sonata_heun = json_bool(sonata_v3, "heun", cfg->sonata_heun);
        cfg->use_phonemizer = json_bool(sonata_v3, "phonemes",
            json_bool(sonata_v3, "phonemize", cfg->use_phonemizer));
        cfg->phoneme_map_path = json_str(sonata_v3, "phoneme_map", cfg->phoneme_map_path);
    }

    /* LLM */
    cJSON *llm = cJSON_GetObjectItemCaseSensitive(root, "llm");
    if (llm) {
        const char *eng = json_str(llm, "engine", NULL);
        if (eng) {
            if (strcmp(eng, "claude") == 0) cfg->llm_engine = LLM_ENGINE_CLAUDE;
            else if (strcmp(eng, "gemini") == 0) cfg->llm_engine = LLM_ENGINE_GEMINI;
            else if (strcmp(eng, "local") == 0) cfg->llm_engine = LLM_ENGINE_LOCAL;
        }
        cfg->llm_model = json_str(llm, "model", cfg->llm_model);
        cfg->system_prompt = json_str(llm, "system_prompt", cfg->system_prompt);
    }

    /* Audio */
    cJSON *audio = cJSON_GetObjectItemCaseSensitive(root, "audio");
    if (audio) {
        cfg->silero_vad_path = json_str(audio, "silero_vad", cfg->silero_vad_path);
        cfg->native_vad_path = json_str(audio, "vad", cfg->native_vad_path);
        cfg->semantic_eou_path = json_str(audio, "semantic_eou", cfg->semantic_eou_path);
        cfg->emosteer_path = json_str(audio, "emosteer", cfg->emosteer_path);
        cfg->prosody_log_path = json_str(audio, "prosody_log", cfg->prosody_log_path);
        cfg->pitch = json_float(audio, "pitch", cfg->pitch);
        cfg->volume_db = json_float(audio, "volume_db", cfg->volume_db);
        cfg->spatial = json_bool(audio, "spatial", cfg->spatial);
        cfg->spatial_az = json_float(audio, "spatial_azimuth", cfg->spatial_az);
        cfg->hw_resample = json_bool(audio, "hw_resample", cfg->hw_resample);
    }

    /* Server */
    cJSON *server = cJSON_GetObjectItemCaseSensitive(root, "server");
    if (server) {
        cfg->server_mode = json_bool(server, "enabled", cfg->server_mode);
        cfg->server_port = json_int(server, "port", cfg->server_port);
    }

    cfg->enable_profiler = json_bool(root, "profiler", cfg->enable_profiler);

    fprintf(stderr, "[config] Loaded %s\n", path);
    cJSON_Delete(root);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Audio Post-Processor
 *
 * Holds all post-processing state: HW resampler, prosody EQ, spatial engine.
 * Created once in main(), passed to feed_speaker() for per-frame processing.
 * ═══════════════════════════════════════════════════════════════════════════ */

typedef struct {
    HWResampler       *resampler_up;    /* 24kHz → 48kHz */
    HWResampler       *resampler_down;  /* 48kHz → 24kHz (for STT path) */
    HWResampler       *resampler_24_16; /* 24kHz → 16kHz (band-limited, for Conformer STT) */
    BiquadCascade     *formant_eq;      /* Formant correction for pitch shift */
    SpatialAudioEngine *spatial;        /* 3D HRTF engine */
    PocketOpus        *opus;            /* Opus encoder (optional) */
    FILE              *opus_file;       /* Opus output file (optional) */
    BreathSynth       *breath;          /* Breath noise synthesizer */
    LUFSMeter         *lufs;            /* LUFS loudness meter */
    SPMCRing          *spmc;            /* SPMC ring: consumer 0=speaker, 1=opus */
    float              target_lufs;     /* Target LUFS level (-16 podcast, -23 broadcast) */
    float              pitch;
    float              volume_db;
    int                use_hw_resample;
    int                use_spatial;
    int                enable_breath;   /* Insert breath noise at sentence gaps */
    int                enable_lufs;     /* LUFS normalization */
    SpeechDetector    *speech_detector;  /* Unified VAD + EOU (native_vad + mimi_ep + fused_eou) */
    SemanticEOU       *semantic_eou;    /* Text-based sentence completion predictor (5th EOU signal) */
    int                speculative_sent; /* 1 if speculative Claude request in-flight */
    int                streaming_overlap; /* 1 if partial-transcript LLM warmup sent */
    int                overlap_word_count; /* Word count when overlap was sent */

    NoiseGate         *noise_gate;      /* Spectral noise gate for STT input */
    DeepFilter        *deep_filter;     /* Neural noise suppression (ERB-band GRU) */
    WebRemote         *web_remote;      /* Phone remote mic/speaker (NULL if unused) */
    float              seg_rate;        /* Current segment speech rate (1.0 = normal) */

    /* Conversational prosody adaptation */
    ConversationProsodyState conv_prosody; /* Tracks user speaking style across turns */

    /* EmoSteer direction vector bank (loaded from JSON) */
    EmoSteerBank      *emosteer_bank;   /* NULL if not loaded */

    /* Prosody logging for dashboard visualization */
    ProsodyLog        *prosody_log;     /* NULL if not enabled */

    /* Adaptive prompt state */
    char               adaptive_suffix[512]; /* Dynamic system prompt suffix */
    int                turns_since_adapt;    /* Counter for periodic re-evaluation */

    /* Prosody feedback loop */
    float              recent_f0_range;      /* F0 range of most recent TTS output */
    float              recent_energy_var;    /* Energy variance of recent output */
    float              prosody_boost;        /* Auto-boost factor (1.0 = no boost) */

    /* Emotion-aware barge-in: softer threshold for empathetic/calm content */
    DetectedEmotion    last_tts_emotion;     /* Emotion of most recently spoken segment */
    float              barge_in_energy_scale; /* 1.0 = default, >1 = harder to interrupt */

    /* ── Human-like quality enhancements ── */

    /* Backchannel generation: "mhm", "yeah" during user speech */
    BackchannelGen    *backchannel;

    /* Audio-based emotion detection from user's voice */
    AudioEmotionDetector *audio_emotion;
    char               audio_emotion_desc[256]; /* Cached description for LLM prompt */

    /* 16kHz recording buffer for diarizer (max 30s) */
    float             *rec_16k;
    int                rec_16k_len;
    int                rec_16k_cap;

    /* Contextual STT biasing: recent LLM output terms boost STT recognition */
    char               context_terms[1024];     /* Comma-separated bias terms */
    int                context_terms_len;

    /* Speculative TTS pre-generation */
    int                speculative_tts_started; /* 1 = pre-generating first words */
    char               speculative_tts_text[256]; /* Predicted first response words */

    /* Disfluency tracking for EOU */
    int                disfluency_count;        /* Consecutive "um/uh" detected */
    int                disfluency_hold;         /* 1 = suppress EOU (user is thinking) */

    /* Adaptive quality: fewer flow steps during fast conversation */
    float              conversation_tempo;      /* EMA of turn gap duration */
    int                adaptive_flow_steps;     /* Current flow steps (4-16) */
    float              adaptive_lufs_target;    /* Dynamic LUFS target */
    float              adaptive_limiter_thresh; /* Dynamic limiter threshold */

    /* Pronunciation dictionary for custom word overrides */
    PronunciationDict *pronunciation_dict;

    /* Cached pitch shift context (avoids FFT setup/teardown per call) */
    void *pitch_ctx;

    /* Audio watermarking for AI-generated speech detection */
    AudioWatermark    *watermark;
    int                enable_watermark;
} AudioPostProcessor;

static AudioPostProcessor *postproc_create(PipelineConfig *cfg) {
    AudioPostProcessor *pp = (AudioPostProcessor *)calloc(1, sizeof(AudioPostProcessor));
    if (!pp) return NULL;

    pp->pitch          = cfg->pitch;
    pp->volume_db      = cfg->volume_db;
    pp->seg_rate       = 1.0f;
    pp->use_hw_resample = cfg->hw_resample;
    pp->use_spatial    = cfg->spatial;
    pp->pitch_ctx      = prosody_pitch_create(2048);

    /* Initialize conversational prosody adaptation */
    prosody_conversation_init(&pp->conv_prosody);

    /* HW resampler: 24kHz TTS → 48kHz speaker (RESAMPLE_HIGH = 3) */
    if (pp->use_hw_resample) {
        pp->resampler_up = hw_resampler_create(24000, 48000, 1, 3);
        pp->resampler_down = hw_resampler_create(48000, 24000, 1, 3);
        if (!pp->resampler_up || !pp->resampler_down) {
            fprintf(stderr, "[postproc] HW resampler failed, falling back to FIR\n");
            pp->use_hw_resample = 0;
        }
    }

    /* Band-limited 24→16 kHz resampler for Conformer STT path */
    pp->resampler_24_16 = hw_resampler_create(24000, 16000, 1, 3);
    if (!pp->resampler_24_16) {
        fprintf(stderr, "[postproc] 24→16 kHz resampler failed (will use linear fallback)\n");
    }

    /* Formant correction EQ for pitch shifting */
    if (fabsf(pp->pitch - 1.0f) > 0.01f) {
        pp->formant_eq = prosody_create_formant_eq(pp->pitch, 24000);
    }

    /* Spatial audio engine */
    if (pp->use_spatial) {
        pp->spatial = spatial_create(48000);
        if (pp->spatial) {
            spatial_set_position(pp->spatial, 0, cfg->spatial_az, 0.0f, 1.5f);
        }
    }

    /* Opus encoder */
    if (cfg->opus_bitrate > 0 && cfg->opus_output) {
        pp->opus = pocket_opus_create(48000, 1, cfg->opus_bitrate, 20.0f, 2048);
        if (pp->opus) {
            pp->opus_file = fopen(cfg->opus_output, "wb");
            if (!pp->opus_file) {
                fprintf(stderr, "[postproc] Failed to open Opus output: %s\n", cfg->opus_output);
                pocket_opus_destroy(pp->opus);
                pp->opus = NULL;
            }
        }
    }

    /* Breath synthesis (always created, used at sentence boundaries) */
    pp->breath = breath_create(48000);
    pp->enable_breath = 1;

    /* LUFS loudness normalization */
    pp->lufs = lufs_create(24000, 400);  /* 400ms momentary window (24kHz pre-resample) */
    pp->target_lufs = -16.0f;  /* Podcast-friendly target */
    pp->enable_lufs = 1;

    /* Spectral noise gate for STT input (16kHz, 512-sample FFT, 50% overlap) */
    pp->noise_gate = noise_gate_create(16000, 512, 256);

    /* Neural noise suppression (ERB-band GRU, replaces spectral gate when weights available) */
    pp->deep_filter = deep_filter_create(16000, "models/denoiser.dnf");

    /* SPMC ring: 2 consumers (speaker=0, opus=1). 96000 floats = 2s @ 48kHz.
       When Opus is disabled, consumer 1 is deactivated so it doesn't block. */
    pp->spmc = (SPMCRing *)calloc(1, sizeof(SPMCRing));
    if (pp->spmc) {
        int n_consumers = pp->opus ? 2 : 1;
        if (spmc_create(pp->spmc, 96000, n_consumers) != 0) {
            free(pp->spmc);
            pp->spmc = NULL;
        }
    }

    /* Unified speech detector: native VAD + mimi endpointer + fused EOU */
    SpeechDetectorConfig sd_cfg = {
        .native_vad_path   = cfg->native_vad_path,
        .mimi_latent_dim   = 80,
        .mimi_hidden_dim   = 64,
        .eot_threshold     = 0.6f,
        .eot_consec_frames = 2,
    };
    pp->speech_detector = speech_detector_create(&sd_cfg);

    /* Semantic EOU: text-based sentence completion predictor (5th signal) */
    pp->semantic_eou = semantic_eou_create();
    if (pp->semantic_eou) {
        if (cfg->semantic_eou_path && cfg->semantic_eou_path[0]) {
            if (semantic_eou_load_weights(pp->semantic_eou, cfg->semantic_eou_path) == 0) {
                /* Enable semantic signal in fused EOU with weight 0.15 */
                speech_detector_set_semantic_weight(pp->speech_detector, 0.15f);
            }
        } else {
            semantic_eou_init_random(pp->semantic_eou, 42);
        }
    }

    pp->speculative_sent = 0;
    pp->streaming_overlap = 0;
    pp->overlap_word_count = 0;

    /* ── Human-like quality modules ── */

    /* Backchannel generator for active listening */
    pp->backchannel = backchannel_create(24000);
    if (pp->backchannel && cfg->backchannel)
        backchannel_set_enabled(pp->backchannel, 1);

    /* Audio-based emotion detection from user's voice */
    pp->audio_emotion = audio_emotion_create(24000);

    /* 16kHz recording accumulator for speaker diarization (max 30s) */
    pp->rec_16k_cap = 16000 * 30;
    pp->rec_16k = (float *)calloc(pp->rec_16k_cap, sizeof(float));
    pp->rec_16k_len = 0;

    /* Adaptive quality defaults */
    pp->adaptive_flow_steps = 8;
    pp->adaptive_lufs_target = -16.0f;
    pp->adaptive_limiter_thresh = 0.95f;
    pp->conversation_tempo = 2.0f;

    /* EmoSteer emotion direction vectors (optional) */
    if (cfg->emosteer_path) {
        pp->emosteer_bank = emosteer_load(cfg->emosteer_path);
    }

    /* Pronunciation dictionary (optional) */
    if (cfg->pronunciation_dict_path) {
        pp->pronunciation_dict = pronunciation_dict_load(cfg->pronunciation_dict_path);
    }

    /* Prosody logging for dashboard visualization (optional) */
    if (cfg->prosody_log_path) {
        pp->prosody_log = prosody_log_open(cfg->prosody_log_path);
    }

    /* Prosody feedback defaults */
    pp->prosody_boost = 1.0f;
    pp->recent_f0_range = 0.0f;
    pp->recent_energy_var = 0.0f;
    pp->adaptive_suffix[0] = '\0';
    pp->turns_since_adapt = 0;
    pp->last_tts_emotion = EMOTION_NEUTRAL;
    pp->barge_in_energy_scale = 1.0f;

    /* Audio watermarking for AI-generated speech detection (EU AI Act) */
    {
        static const uint8_t default_wm_key[] = "sonata-ai-watermark-v1";
        pp->watermark = audio_watermark_create(24000, 960, default_wm_key,
                                               (int)sizeof(default_wm_key));
        if (pp->watermark) {
            AudioWatermarkPayload wm_payload = {
                .ai_generated = 1,
                .timestamp    = (uint32_t)time(NULL),
                .model_id     = 1  /* Sonata TTS */
            };
            audio_watermark_set_payload(pp->watermark, &wm_payload);
            pp->enable_watermark = 1;
        }
    }

    return pp;
}

static void postproc_destroy(AudioPostProcessor *pp) {
    if (!pp) return;
    if (pp->resampler_up)   hw_resampler_destroy(pp->resampler_up);
    if (pp->resampler_down) hw_resampler_destroy(pp->resampler_down);
    if (pp->resampler_24_16) hw_resampler_destroy(pp->resampler_24_16);
    if (pp->formant_eq)     prosody_destroy_biquad(pp->formant_eq);
    if (pp->spatial)        spatial_destroy(pp->spatial);
    if (pp->opus)           pocket_opus_destroy(pp->opus);
    if (pp->opus_file)      fclose(pp->opus_file);
    if (pp->breath)         breath_destroy(pp->breath);
    if (pp->lufs)           lufs_destroy(pp->lufs);
    if (pp->spmc)           { spmc_destroy(pp->spmc); free(pp->spmc); }
    if (pp->speech_detector) speech_detector_destroy(pp->speech_detector);
    if (pp->semantic_eou)   semantic_eou_destroy(pp->semantic_eou);
    if (pp->noise_gate)     noise_gate_destroy(pp->noise_gate);
    if (pp->deep_filter)    deep_filter_destroy(pp->deep_filter);
    if (pp->emosteer_bank)  emosteer_destroy(pp->emosteer_bank);
    if (pp->pronunciation_dict) pronunciation_dict_destroy(pp->pronunciation_dict);
    if (pp->prosody_log)    prosody_log_close(pp->prosody_log);
    if (pp->backchannel)    backchannel_destroy(pp->backchannel);
    if (pp->audio_emotion)  audio_emotion_destroy(pp->audio_emotion);
    if (pp->pitch_ctx)      prosody_pitch_destroy(pp->pitch_ctx);
    if (pp->watermark)      audio_watermark_destroy(pp->watermark);
    free(pp->rec_16k);
    free(pp);
}

static void postproc_reset(AudioPostProcessor *pp) {
    if (!pp) return;
    if (pp->resampler_up)   hw_resampler_reset(pp->resampler_up);
    if (pp->resampler_down) hw_resampler_reset(pp->resampler_down);
    if (pp->resampler_24_16) hw_resampler_reset(pp->resampler_24_16);
    if (pp->lufs)           lufs_reset(pp->lufs);
    if (pp->speech_detector) speech_detector_reset(pp->speech_detector);
    if (pp->semantic_eou)   semantic_eou_reset(pp->semantic_eou);
    if (pp->noise_gate)     noise_gate_reset(pp->noise_gate);
    if (pp->deep_filter)    deep_filter_reset(pp->deep_filter);
    if (pp->backchannel)    backchannel_reset(pp->backchannel);
    if (pp->audio_emotion)  audio_emotion_reset(pp->audio_emotion);
    if (pp->watermark)      audio_watermark_reset(pp->watermark);
    pp->rec_16k_len = 0;
    pp->speculative_sent = 0;
    pp->streaming_overlap = 0;
    pp->overlap_word_count = 0;
    pp->speculative_tts_started = 0;
    pp->disfluency_count = 0;
    pp->disfluency_hold = 0;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Main Pipeline
 * ═══════════════════════════════════════════════════════════════════════════ */

static volatile sig_atomic_t g_quit = 0;

static void signal_handler(int sig) {
    (void)sig;
    g_quit = 1;
}

#define AUDIO_SAMPLE_RATE  48000
#define AUDIO_BUFFER_FRAMES 256
#define STT_FRAME_SIZE     1920   /* 80ms at 24kHz */
#define RESAMPLE_BUF_SIZE  8192
#define TEXT_BUF_SIZE       4096
#define TTS_AUDIO_BUF_SIZE 4096
#define TTS_STEPS_PER_TICK_MIN 4   /* Minimum TTS steps per tick */
#define TTS_STEPS_PER_TICK_MAX 16  /* Maximum TTS steps per tick */
#define TTS_STEPS_PER_TICK 8       /* Default TTS steps per tick */
#define SPEAKING_TIMEOUT_US (30ULL * 1000000)  /* 30s max in SPEAKING state */

/* Monotonic clock in microseconds (survives sleep, unlike mach_absolute_time) */
static uint64_t now_us(void) {
    static mach_timebase_info_data_t tb;
    if (tb.denom == 0) mach_timebase_info(&tb);
    uint64_t ticks = mach_continuous_time();
    return ticks * tb.numer / tb.denom / 1000ULL;
}

/* Latency metrics for the current turn — nanosecond precision via mach_absolute_time */
typedef struct {
    uint64_t speech_start;     /* When VAD detected speech onset */
    uint64_t speech_end;       /* When end-of-turn was detected */
    uint64_t stt_start;        /* When STT inference began */
    uint64_t stt_done;         /* When STT produced final transcript */
    uint64_t llm_sent;         /* When request was sent to LLM */
    uint64_t llm_first_tok;    /* When first LLM token arrived (TTFT) */
    uint64_t tts_start;        /* When TTS received first text */
    uint64_t tts_first_audio;  /* When first TTS audio was written to speaker */
    uint64_t speaking_entered; /* When we entered SPEAKING state (for timeout) */
    uint64_t turn_complete;    /* When speaking finished */
    bool     has_first_tok;
    bool     has_first_audio;
    int      stt_frames;       /* Number of audio frames processed by STT */
    int      llm_tokens;       /* Number of LLM tokens generated */
    int      tts_tokens;       /* Number of TTS semantic tokens generated */
    float    tts_audio_sec;    /* Total TTS audio duration in seconds */
} TurnMetrics;

static double mach_to_ms(uint64_t start, uint64_t end) {
    static mach_timebase_info_data_t tb = {0};
    if (tb.denom == 0) mach_timebase_info(&tb);
    return (double)(end - start) * tb.numer / tb.denom / 1e6;
}

static void print_turn_latency(const TurnMetrics *m) {
    if (!m->speech_start || !m->speech_end) return;
    fprintf(stderr, "\n┌─── Turn Latency Breakdown ───────────────────┐\n");

    double speech_dur = mach_to_ms(m->speech_start, m->speech_end);
    fprintf(stderr, "│ Speech duration:    %7.1f ms               │\n", speech_dur);

    if (m->stt_start && m->stt_done) {
        double stt = mach_to_ms(m->stt_start, m->stt_done);
        fprintf(stderr, "│ STT inference:      %7.1f ms (%d frames)    │\n", stt, m->stt_frames);
    }

    if (m->llm_sent && m->has_first_tok) {
        double ttft = mach_to_ms(m->llm_sent, m->llm_first_tok);
        fprintf(stderr, "│ LLM TTFT:           %7.1f ms               │\n", ttft);
    }

    if (m->tts_start && m->has_first_audio) {
        double tts_lat = mach_to_ms(m->tts_start, m->tts_first_audio);
        fprintf(stderr, "│ TTS first audio:    %7.1f ms               │\n", tts_lat);
    }

    if (m->speech_end && m->has_first_audio) {
        double vrl = mach_to_ms(m->speech_end, m->tts_first_audio);
        fprintf(stderr, "│ ═══════════════════════════════════════════ │\n");
        fprintf(stderr, "│ Voice Response Lat: %7.1f ms  ◄── KEY      │\n", vrl);
    }

    if (m->tts_audio_sec > 0 && m->tts_start && m->turn_complete) {
        double gen_time = mach_to_ms(m->tts_start, m->turn_complete) / 1000.0;
        double rtf = gen_time / m->tts_audio_sec;
        fprintf(stderr, "│ TTS RTF:            %7.3f (%.0fx realtime)  │\n",
                rtf, 1.0 / rtf);
    }

    if (m->llm_tokens > 0 && m->llm_sent && m->turn_complete) {
        double llm_time = mach_to_ms(m->llm_sent, m->turn_complete) / 1000.0;
        double tps = m->llm_tokens / llm_time;
        fprintf(stderr, "│ LLM throughput:     %7.1f tok/s             │\n", tps);
    }

    fprintf(stderr, "└───────────────────────────────────────────────┘\n");
}

/* Accumulation buffer for resampled STT frames */
typedef struct {
    float buf[STT_FRAME_SIZE * 4];
    int   len;
} SttAccum;

static void stt_accum_reset(SttAccum *a) { a->len = 0; }

/* feed_endpointer removed — now handled internally by speech_detector_feed() */

/** Downsample float audio from src_rate to dst_rate via linear interpolation.
 * Suitable for moderate ratio conversions (e.g. 24kHz to 16kHz). */
static int linear_resample(const float *in, int n_in, int src_rate,
                            float *out, int max_out, int dst_rate) {
    if (n_in <= 0 || src_rate <= 0 || dst_rate <= 0) return 0;
    double ratio = (double)src_rate / (double)dst_rate;
    int n_out = (int)((double)n_in / ratio);
    if (n_out > max_out) n_out = max_out;
    for (int i = 0; i < n_out; i++) {
        double src_idx = (double)i * ratio;
        int idx = (int)src_idx;
        double frac = src_idx - idx;
        if (idx + 1 < n_in)
            out[i] = (float)((1.0 - frac) * in[idx] + frac * in[idx + 1]);
        else
            out[i] = in[idx < n_in ? idx : n_in - 1];
    }
    return n_out;
}

/* Feed captured audio (48kHz) through resampler and into the STT engine.
 *
 * For Rust STT:     48kHz → 24kHz (frame = 1920 samples = 80ms)
 * For Conformer:    48kHz → 24kHz → 16kHz (frame = 1280 samples = 80ms)
 *
 * Accumulates until a full frame is ready. When text is recognized,
 * appends to transcript. Returns number of new recognized items. */
static int feed_stt(SttInterface *stt, VoiceEngine *audio, SttAccum *accum,
                     char *transcript, int *transcript_len, int transcript_cap,
                     AudioPostProcessor *pp) {
    float capture_48[RESAMPLE_BUF_SIZE];
    float capture_24[RESAMPLE_BUF_SIZE / 2];
    float capture_16[RESAMPLE_BUF_SIZE / 3 + 64];
    int total_words = 0;

    int n = voice_engine_read_capture(audio, capture_48, RESAMPLE_BUF_SIZE);
    if (n <= 0) return 0;

    /* Resample 48kHz → 24kHz via HW AudioConverter or FIR fallback */
    int n24;
    if (pp && pp->use_hw_resample && pp->resampler_down) {
        n24 = hw_resample(pp->resampler_down, capture_48, (uint32_t)n,
                          capture_24, RESAMPLE_BUF_SIZE / 2);
        if (n24 <= 0) {
            voice_engine_resample_48_to_24(capture_48, capture_24, n);
            n24 = n / 2;
        }
    } else {
        voice_engine_resample_48_to_24(capture_48, capture_24, n);
        n24 = n / 2;
    }

    /* Feed capture audio to unified speech detector (VAD + endpointer) */
    if (pp && pp->speech_detector)
        speech_detector_feed(pp->speech_detector, capture_24, n24);

    /* Feed audio emotion detector for user voice analysis */
    if (pp && pp->audio_emotion)
        audio_emotion_feed(pp->audio_emotion, capture_24, n24);

    /* Feed backchannel generator for timing analysis */
    if (pp && pp->backchannel && backchannel_is_enabled(pp->backchannel)) {
        float eou_prob = (pp->speech_detector)
            ? speech_detector_eou(pp->speech_detector, 0, 0.0f).fused_prob
            : 0.0f;
        BackchannelEvent evt = backchannel_feed(pp->backchannel, capture_24, n24, eou_prob);
        if (evt.ready) {
            int bc_len = 0;
            const float *bc_audio = backchannel_get_audio(pp->backchannel, evt.type, &bc_len);
            if (bc_audio && bc_len > 0)
                voice_engine_write_playback(audio, bc_audio, bc_len);
        }
    }

    /* Select audio at the STT engine's expected sample rate */
    const float *stt_audio;
    int n_stt;
    int frame_size = stt->frame_size;

    if (stt->sample_rate == 16000) {
        if (pp && pp->resampler_24_16) {
            n_stt = hw_resample(pp->resampler_24_16, capture_24, (uint32_t)n24,
                                capture_16, RESAMPLE_BUF_SIZE / 3 + 64);
            if (n_stt <= 0)
                n_stt = linear_resample(capture_24, n24, 24000,
                                        capture_16, RESAMPLE_BUF_SIZE / 3 + 64, 16000);
        } else {
            n_stt = linear_resample(capture_24, n24, 24000,
                                    capture_16, RESAMPLE_BUF_SIZE / 3 + 64, 16000);
        }
        stt_audio = capture_16;
    } else {
        stt_audio = capture_24;
        n_stt = n24;
    }

    /* Accumulate 16kHz audio for speaker diarization */
    if (pp && pp->rec_16k) {
        const float *src_16 = (stt->sample_rate == 16000) ? capture_16 : NULL;
        int n_16 = (stt->sample_rate == 16000) ? n_stt : 0;
        if (!src_16) {
            n_16 = linear_resample(capture_24, n24, 24000,
                                   capture_16, RESAMPLE_BUF_SIZE / 3 + 64, 16000);
            src_16 = capture_16;
        }
        if (n_16 > 0 && pp->rec_16k_len + n_16 <= pp->rec_16k_cap) {
            memcpy(pp->rec_16k + pp->rec_16k_len, src_16, n_16 * sizeof(float));
            pp->rec_16k_len += n_16;
        }
    }

    /* Noise suppression before STT: prefer neural deep_filter, fall back to spectral gate.
     * deep_filter and noise_gate operate at 16kHz. When STT is 24kHz, resample down
     * to 16kHz, apply suppression, then resample back to 24kHz. */
    if (pp && n_stt > 0) {
        if (pp->deep_filter) {
            if (stt->sample_rate == 16000) {
                deep_filter_process(pp->deep_filter, capture_16, n_stt);
            } else {
                /* STT at 24kHz — resample to 16kHz for deep_filter, process, resample back */
                int n_df16 = linear_resample(capture_24, n_stt, 24000,
                                              capture_16, RESAMPLE_BUF_SIZE / 3 + 64, 16000);
                if (n_df16 > 0) {
                    deep_filter_process(pp->deep_filter, capture_16, n_df16);
                    int n_back = linear_resample(capture_16, n_df16, 16000,
                                                  capture_24, RESAMPLE_BUF_SIZE / 2, 24000);
                    if (n_back > 0) n_stt = n_back;
                }
            }
        } else if (pp->noise_gate) {
            if (stt->sample_rate == 16000) {
                noise_gate_process(pp->noise_gate, capture_16, n_stt);
            } else {
                int n_ng16 = linear_resample(capture_24, n_stt, 24000,
                                              capture_16, RESAMPLE_BUF_SIZE / 3 + 64, 16000);
                if (n_ng16 > 0) {
                    noise_gate_process(pp->noise_gate, capture_16, n_ng16);
                    int n_back = linear_resample(capture_16, n_ng16, 16000,
                                                  capture_24, RESAMPLE_BUF_SIZE / 2, 24000);
                    if (n_back > 0) n_stt = n_back;
                }
            }
        }
    }

    /* Accumulate into STT frame buffer */
    int space = STT_FRAME_SIZE * 4 - accum->len;
    int to_copy = n_stt < space ? n_stt : space;
    memcpy(accum->buf + accum->len, stt_audio, (size_t)to_copy * sizeof(float));
    accum->len += to_copy;

    /* Process full frames */
    while (accum->len >= frame_size) {
        int nw = stt->process_frame(stt->engine, accum->buf, frame_size);
        if (nw > 0) {
            total_words += nw;
            char word_buf[TEXT_BUF_SIZE];
            int wlen = stt->get_text(stt->engine, word_buf, TEXT_BUF_SIZE);
            if (wlen > 0) {
                int avail = transcript_cap - *transcript_len - 1;
                int copy = wlen < avail ? wlen : avail;
                if (copy > 0) {
                    memcpy(transcript + *transcript_len, word_buf, (size_t)copy);
                    *transcript_len += copy;
                    transcript[*transcript_len] = '\0';
                }
            }
        }
        int remaining = accum->len - frame_size;
        if (remaining > 0) {
            memmove(accum->buf, accum->buf + frame_size,
                    (size_t)remaining * sizeof(float));
        }
        accum->len = remaining;
    }

    return total_words;
}

/* Feed TTS audio (24kHz) through post-processing pipeline to speaker (48kHz).
 *
 * Pipeline: TTS → [pitch shift] → [formant EQ] → [volume] → [LUFS]
 *         → [watermark] → [soft limit] → resample 24→48 → [spatial HRTF]
 *         → playback ring
 *
 * Returns number of 24kHz samples transferred. */
static int feed_speaker(TtsInterface *tts, VoiceEngine *audio, AudioPostProcessor *pp) {
    float pcm_24[TTS_AUDIO_BUF_SIZE];
    float processed[TTS_AUDIO_BUF_SIZE];
    float pcm_48[TTS_AUDIO_BUF_SIZE * 2 + 256]; /* Extra margin for resampler */
    int total = 0;

    for (;;) {
        /* Try zero-copy peek first, fall back to copy-based get_audio */
        const float *peek_ptr = NULL;
        int peek_count = 0;
        int n;

        if (tts->peek_audio(tts->engine, &peek_ptr, &peek_count) == 0
            && peek_ptr && peek_count > 0) {
            n = peek_count < TTS_AUDIO_BUF_SIZE ? peek_count : TTS_AUDIO_BUF_SIZE;
            memcpy(pcm_24, peek_ptr, (size_t)n * sizeof(float));
            tts->advance_audio(tts->engine, n);
        } else {
            n = tts->get_audio(tts->engine, pcm_24, TTS_AUDIO_BUF_SIZE);
        }
        if (n <= 0) break;

        float *src = pcm_24;

        /* Resample to 24kHz if TTS outputs different rate (e.g. Piper at 22.05kHz) */
        if (tts->sample_rate != 24000 && tts->sample_rate > 0) {
            int n24 = linear_resample(pcm_24, n, tts->sample_rate,
                                      processed, TTS_AUDIO_BUF_SIZE, 24000);
            if (n24 > 0) {
                memcpy(pcm_24, processed, (size_t)n24 * sizeof(float));
                n = n24;
            }
        }

        /* Peak normalization: TTS vocoder output routinely exceeds ±1.0
         * (avg peak ~2.0). Scale to ±0.9 before any processing to avoid
         * heavy tanh saturation in the soft limiter. */
        if (pp) {
            if (src != processed) { memcpy(processed, src, n * sizeof(float)); src = processed; }
            float peak = 0.0f;
            vDSP_maxmgv(src, 1, &peak, n);
            if (peak > 0.9f) {
                float scale = 0.9f / peak;
                vDSP_vsmul(src, 1, &scale, src, 1, n);
            }
        }

        /* Pitch shift (operates at 24kHz, needs ≥2048 samples for quality) */
        if (pp && fabsf(pp->pitch - 1.0f) > 0.01f && n >= 2048) {
            if (pp->pitch_ctx)
                prosody_pitch_shift_ctx(pp->pitch_ctx, src, processed, n, pp->pitch);
            else
                prosody_pitch_shift(src, processed, n, pp->pitch, 2048);
            src = processed;

            if (pp->formant_eq) {
                prosody_apply_biquad(pp->formant_eq, src, n);
            }
        }

        /* Rate / time-stretch (WSOLA at 24kHz) */
        if (pp && fabsf(pp->seg_rate - 1.0f) > 0.02f && n >= 1024) {
            float stretched[TTS_AUDIO_BUF_SIZE * 2];
            int out_len = prosody_time_stretch(src, n, stretched,
                                               pp->seg_rate, 30.0f, 24000);
            if (out_len > 0 && out_len <= TTS_AUDIO_BUF_SIZE) {
                memcpy(processed, stretched, (size_t)out_len * sizeof(float));
                src = processed;
                n = out_len;
            }
        }

        /* Volume adjustment */
        if (pp && fabsf(pp->volume_db) > 0.1f) {
            if (src != processed) { memcpy(processed, src, n * sizeof(float)); src = processed; }
            prosody_volume(src, n, pp->volume_db, 0.0f, 24000);
        }

        /* LUFS loudness normalization (before resampling, operates at 24kHz) */
        if (pp && pp->enable_lufs && pp->lufs && n >= 960) {
            if (src != processed) { memcpy(processed, src, n * sizeof(float)); src = processed; }
            lufs_normalize(pp->lufs, src, n, pp->target_lufs);
        }

        /* Audio watermark: embed imperceptible AI-generated marker (EU AI Act).
         * After volume/LUFS normalization, before limiter — ensures consistent
         * embedding level and the limiter won't clip the watermark. */
        if (pp && pp->enable_watermark && pp->watermark) {
            if (src != processed) { memcpy(processed, src, n * sizeof(float)); src = processed; }
            audio_watermark_embed(pp->watermark, src, n);
        }

        /* Final safety limiter: gentle tanh at 0.95 to catch any overshoot
         * from volume/LUFS without aggressive squashing. */
        if (pp) {
            if (src != processed) { memcpy(processed, src, n * sizeof(float)); src = processed; }
            prosody_soft_limit(src, n, 0.95f, 12.0f);
        }

        /* Resample 24kHz → 48kHz */
        int n48;
        if (pp && pp->use_hw_resample && pp->resampler_up) {
            n48 = hw_resample(pp->resampler_up, src, (uint32_t)n,
                              pcm_48, TTS_AUDIO_BUF_SIZE * 2);
            if (n48 <= 0) {
                /* HW resampler failed, fall back to FIR */
                voice_engine_resample_24_to_48(src, pcm_48, n);
                n48 = n * 2;
            }
        } else {
            voice_engine_resample_24_to_48(src, pcm_48, n);
            n48 = n * 2;
        }

        /* Spatial audio: mono → stereo HRTF (writes interleaved L/R) */
        if (pp && pp->use_spatial && pp->spatial) {
            float left[TTS_AUDIO_BUF_SIZE * 2];
            float right[TTS_AUDIO_BUF_SIZE * 2];
            int sp_ok = spatial_process(pp->spatial, 0, pcm_48, left, right, n48);
            if (sp_ok < 0) {
                /* Block too large for scratch buffers — process in chunks */
                const int chunk = 4096;
                for (int off = 0; off < n48; off += chunk) {
                    int rem = n48 - off < chunk ? n48 - off : chunk;
                    spatial_process(pp->spatial, 0, pcm_48 + off,
                                    left + off, right + off, rem);
                }
            }
            for (int i = 0; i < n48; i++) {
                pcm_48[i] = left[i] * 0.5f + right[i] * 0.5f;
            }
            /* Note: true stereo playback requires VoiceEngine stereo support.
             * For now, downmix to mono with spatial imaging preserved via
             * phase differences. Still provides spatial perception on speakers. */
        }

        /* If SPMC ring is active, write once and let all consumers read.
           Consumer 0 = speaker playback, Consumer 1 = Opus encoder. */
        if (pp && pp->spmc) {
            spmc_write(pp->spmc, pcm_48, (uint32_t)n48);

            /* Consumer 0: speaker playback */
            float spmc_out[TTS_AUDIO_BUF_SIZE * 2 + 256];
            uint32_t avail0 = spmc_available_read(pp->spmc, 0);
            if (avail0 > 0) {
                uint32_t to_read = avail0 < sizeof(spmc_out)/sizeof(float) ?
                                   avail0 : sizeof(spmc_out)/sizeof(float);
                spmc_read(pp->spmc, 0, spmc_out, to_read);
                voice_engine_write_playback(audio, spmc_out, (int)to_read);
                if (pp && pp->web_remote)
                    web_remote_send_audio(pp->web_remote, spmc_out, (int)to_read);
            }

            /* Consumer 1: Opus encoding (if active) */
            if (pp->opus && pp->opus_file) {
                uint32_t avail1 = spmc_available_read(pp->spmc, 1);
                if (avail1 > 0) {
                    float opus_pcm[TTS_AUDIO_BUF_SIZE * 2 + 256];
                    uint32_t to_read = avail1 < sizeof(opus_pcm)/sizeof(float) ?
                                       avail1 : sizeof(opus_pcm)/sizeof(float);
                    spmc_read(pp->spmc, 1, opus_pcm, to_read);
                    unsigned char opus_buf[4096];
                    int ob = pocket_opus_encode(pp->opus, opus_pcm, (int)to_read,
                                                opus_buf, sizeof(opus_buf));
                    if (ob > 0) {
                        fwrite(opus_buf, 1, (size_t)ob, pp->opus_file);
                    }
                }
            }
        } else {
            /* Fallback: direct write without SPMC */
            if (pp && pp->opus && pp->opus_file) {
                unsigned char opus_buf[4096];
                int ob = pocket_opus_encode(pp->opus, pcm_48, n48, opus_buf, sizeof(opus_buf));
                if (ob > 0) {
                    fwrite(opus_buf, 1, (size_t)ob, pp->opus_file);
                }
            }
            voice_engine_write_playback(audio, pcm_48, n48);
            if (pp && pp->web_remote)
                web_remote_send_audio(pp->web_remote, pcm_48, n48);
        }

        total += n;
    }
    return total;
}

/* Barge-in: flush the playback ring. A minor click is possible but acceptable
 * since the VoiceProcessingIO AEC handles echo cancellation, and the user
 * is actively speaking (masking any artifact). */
static void barge_in_flush(VoiceEngine *audio) {
    voice_engine_flush_playback(audio);
}

static void print_state(PipelineState state) {
    fprintf(stderr, "\r[pocket-voice] %s   ", state_names[state]);
    fflush(stderr);
}

/**
 * Process a single SSML segment: normalize text, set prosody, drive TTS,
 * insert breaks. Called from STATE_STREAMING when sentence buffer flushes.
 */
static void process_segment(const SSMLSegment *seg, TtsInterface *tts,
                             VoiceEngine *audio, AudioPostProcessor *pp,
                             TurnMetrics *metrics) {
    if (!seg || seg->is_audio) return;
    if (!metrics->tts_start) metrics->tts_start = now_us();

    /* Insert break before segment: for Sonata, inject pause tokens into the
     * semantic stream so the Flow model generates natural silence rather than
     * cutting audio. For other engines, use silence/breath audio. */
    if (seg->break_before_ms > 0 && tts->type == TTS_ENGINE_SONATA) {
        SonataEngine *se = (SonataEngine *)tts->engine;
        if (se->lm_engine) {
            int pause_frames = sonata_lm_ms_to_frames(seg->break_before_ms);
            if (pause_frames > 0) {
                sonata_lm_inject_pause(se->lm_engine, pause_frames);
            }
        }
    } else if (seg->break_before_ms > 0 && audio) {
        int gap_samples = 48000 * seg->break_before_ms / 1000;
        float gap_buf[4096];

        if (pp && pp->enable_breath && pp->breath && gap_samples >= 2400) {
            int wrote = 0;
            while (wrote < gap_samples) {
                int chunk = (gap_samples - wrote);
                if (chunk > 4096) chunk = 4096;
                memset(gap_buf, 0, (size_t)chunk * sizeof(float));
                breath_sentence_gap(pp->breath, gap_buf, chunk, 0.05f);
                voice_engine_write_playback(audio, gap_buf, chunk);
                wrote += chunk;
            }
        } else {
            memset(gap_buf, 0, sizeof(gap_buf));
            while (gap_samples > 0) {
                int chunk = gap_samples < 4096 ? gap_samples : 4096;
                voice_engine_write_playback(audio, gap_buf, chunk);
                gap_samples -= chunk;
            }
        }
    }

    /* Auto-normalize the text */
    char normalized[4096];
    text_auto_normalize(seg->text, normalized, sizeof(normalized));

    /* Hoist declarations above goto to avoid jumping over initializers (C UB) */
    float seg_pitch;
    float seg_rate;
    float seg_vol_db;
    int nlen;
    MultiScaleProsody msp;

    if (normalized[0] == '\0') goto segment_break_after;

    /* ── Resolve prosody from SSML tags ──
     * The SSML parser already applies emotion→prosody mapping in its walker,
     * so seg->pitch/rate/volume reflect the full SSML + emotion state.
     * We do NOT re-apply find_emotion() here to avoid double application. */
    seg_pitch  = seg->pitch;
    seg_rate   = seg->rate;
    seg_vol_db = 0.0f;

    if (fabsf(seg->volume - 1.0f) > 0.01f)
        seg_vol_db = 20.0f * log10f(seg->volume);

    nlen = (int)strlen(normalized);

    /* ── Auto-intonation rules (only when no explicit prosody is set) ── */
    if (pp && nlen > 0 && fabsf(seg_pitch - 1.0f) < 0.01f) {
        char last_char = normalized[nlen - 1];

        /* Question: rising pitch + slightly slower */
        if (last_char == '?') {
            seg_pitch = 1.08f;
            if (fabsf(seg_rate - 1.0f) < 0.01f)
                seg_rate = 0.95f;
        }
        /* Exclamation: energy boost */
        else if (last_char == '!') {
            seg_pitch = 1.06f;
            seg_vol_db += 1.5f;
        }
        /* Comma continuation rise: slight pitch lift for prosodic phrasing */
        else if (last_char == ',') {
            seg_pitch = 1.03f;
        }
        /* Semicolon / colon: moderate boundary */
        else if (last_char == ';' || last_char == ':') {
            seg_pitch = 0.97f;
            if (fabsf(seg_rate - 1.0f) < 0.01f)
                seg_rate = 0.95f;
        }
    }

    /* Em-dash or parenthetical aside: pitch drop + rate slow for aside effect */
    if (pp && fabsf(seg_pitch - 1.0f) < 0.01f) {
        if (strstr(normalized, "\xe2\x80\x94") /* UTF-8 em-dash */
            || strstr(normalized, " -- ")) {
            seg_pitch = 0.96f;
            if (fabsf(seg_rate - 1.0f) < 0.01f)
                seg_rate = 0.93f;
        }
    }

    /* Quoted speech: subtle pitch shift for reported/inner voice */
    if (pp && fabsf(seg_pitch - 1.0f) < 0.01f && nlen > 2) {
        if ((normalized[0] == '"' && normalized[nlen - 1] == '"') ||
            (normalized[0] == '\xe2' && (unsigned char)normalized[1] == 0x80 &&
             (unsigned char)normalized[2] == 0x9c)) {
            seg_pitch = 1.04f;
        }
    }

    /* Apply pitch to post-processor */
    if (pp && fabsf(seg_pitch - 1.0f) > 0.01f) {
        pp->pitch = seg_pitch;
        if (pp->formant_eq) prosody_destroy_biquad(pp->formant_eq);
        pp->formant_eq = prosody_create_formant_eq(seg_pitch, 24000);
    } else if (pp) {
        pp->pitch = 1.0f;
    }

    /* Apply volume */
    if (pp && fabsf(seg_vol_db) > 0.1f) {
        pp->volume_db = seg_vol_db;
    } else if (pp) {
        pp->volume_db = 0.0f;
    }

    /* Store segment rate for feed_speaker to apply time-stretching */
    if (pp) pp->seg_rate = seg_rate;

    /* ── Multi-scale prosody analysis ──
     * Analyze the normalized text for utterance contour, word emphasis,
     * and blend with conversational adaptation from the user's style. */
    msp = prosody_analyze_text(normalized);

    /* Apply conversational adaptation: blend user pace/energy into response */
    if (pp) {
        ProsodyHint adapt = prosody_conversation_adapt(&pp->conv_prosody);
        if (fabsf(seg_pitch - 1.0f) < 0.01f)
            seg_pitch *= adapt.pitch;
        if (fabsf(seg_rate - 1.0f) < 0.01f)
            seg_rate *= adapt.rate;
        seg_vol_db += adapt.energy;
    }

    /* Apply multi-scale utterance-level hints (if no explicit SSML) */
    if (fabsf(seg_pitch - 1.0f) < 0.01f)
        seg_pitch *= msp.utterance.pitch;
    if (fabsf(seg_rate - 1.0f) < 0.01f)
        seg_rate *= msp.utterance.rate;
    seg_vol_db += msp.utterance.energy;

    /* ── Prosody conditioning on Sonata LM + Flow ──
     * Pass prosody parameters into the model so it generates tokens
     * conditioned on the desired pitch/energy/rate. This supplements
     * the post-hoc vDSP prosody with model-aware generation. */
    if (tts->type == TTS_ENGINE_SONATA) {
        SonataEngine *se = (SonataEngine *)tts->engine;
        float prosody[3] = {
            logf(seg_pitch > 0.01f ? seg_pitch : 1.0f),
            seg_vol_db / 20.0f,
            seg_rate
        };
        if (se->lm_engine)
            sonata_lm_set_prosody(se->lm_engine, prosody, 1);
        if (se->flow_engine)
            sonata_flow_set_prosody(se->flow_engine, prosody, 3);

        /* Syllable-aware duration estimation → Flow duration conditioning */
        if (se->flow_engine) {
            float durations[512];
            int n_dur = prosody_estimate_durations(normalized, 256, durations, 512);
            if (n_dur > 0)
                sonata_flow_set_durations(se->flow_engine, durations, n_dur);
        }

        /* Map emotion name to Flow emotion ID if available */
        static const char *emo_names[] = {
            "happy", "excited", "sad", "angry", "fearful",
            "surprised", "warm", "serious", "calm", "confident",
            "whisper", "emphatic", NULL
        };
        if (seg->emotion[0] && se->flow_engine) {
            for (int i = 0; emo_names[i]; i++) {
                if (strcasecmp(seg->emotion, emo_names[i]) == 0) {
                    sonata_flow_set_emotion(se->flow_engine, i);
                    break;
                }
            }
        }

        /* EmoSteer: if emotion is set and we have direction vectors, apply steering */
        if (seg->emotion[0] && pp && pp->emosteer_bank && se->flow_engine) {
            const float *dir = emosteer_get_direction(pp->emosteer_bank, seg->emotion);
            if (dir) {
                sonata_flow_set_emotion_steering(
                    se->flow_engine, dir, pp->emosteer_bank->dim,
                    pp->emosteer_bank->layer_start,
                    pp->emosteer_bank->layer_end,
                    pp->emosteer_bank->default_scale);
            }
        } else if (se->flow_engine) {
            /* Auto-detect emotion from text if not specified by SSML.
             * DetectedEmotion enum order differs from emo_names[] (Flow embedding IDs),
             * so use an explicit lookup table. */
            static const int emotion_to_flow_id[EMOTION_COUNT] = {
                [EMOTION_NEUTRAL]   = -1,
                [EMOTION_HAPPY]     =  0,
                [EMOTION_EXCITED]   =  1,
                [EMOTION_SAD]       =  2,
                [EMOTION_ANGRY]     =  3,
                [EMOTION_SURPRISED] =  5,
                [EMOTION_WARM]      =  6,
                [EMOTION_SERIOUS]   =  7,
                [EMOTION_CALM]      =  8,
                [EMOTION_CONFIDENT] =  9,
                [EMOTION_FEARFUL]   =  4,
            };
            EmotionDetection det = prosody_detect_emotion(normalized);
            if (det.confidence >= 0.4f && det.emotion > EMOTION_NEUTRAL
                    && det.emotion < EMOTION_COUNT) {
                int flow_id = emotion_to_flow_id[det.emotion];
                if (flow_id >= 0) {
                    sonata_flow_set_emotion(se->flow_engine, flow_id);
                    if (pp && pp->emosteer_bank) {
                        const float *dir = emosteer_get_direction(
                            pp->emosteer_bank, emo_names[flow_id]);
                        if (dir) {
                            sonata_flow_set_emotion_steering(
                                se->flow_engine, dir, pp->emosteer_bank->dim,
                                pp->emosteer_bank->layer_start,
                                pp->emosteer_bank->layer_end,
                                pp->emosteer_bank->default_scale * det.confidence);
                        }
                    }
                }
            }
        }
    }

    /* Track the emotion for barge-in sensitivity adaptation */
    if (pp && seg->emotion[0]) {
        static const struct { const char *name; DetectedEmotion e; } emo_map[] = {
            {"sad", EMOTION_SAD}, {"calm", EMOTION_CALM}, {"warm", EMOTION_WARM},
            {"serious", EMOTION_SERIOUS}, {"happy", EMOTION_HAPPY},
            {"excited", EMOTION_EXCITED}, {"angry", EMOTION_ANGRY},
            {"fearful", EMOTION_FEARFUL}, {NULL, EMOTION_NEUTRAL}
        };
        for (int i = 0; emo_map[i].name; i++) {
            if (strcasecmp(seg->emotion, emo_map[i].name) == 0) {
                pp->last_tts_emotion = emo_map[i].e;
                break;
            }
        }
        /* Empathetic/calm content should be harder to interrupt */
        switch (pp->last_tts_emotion) {
            case EMOTION_SAD:
            case EMOTION_WARM:
            case EMOTION_CALM:
                pp->barge_in_energy_scale = 1.4f;
                break;
            case EMOTION_SERIOUS:
            case EMOTION_FEARFUL:
                pp->barge_in_energy_scale = 1.3f;
                break;
            case EMOTION_EXCITED:
            case EMOTION_ANGRY:
                pp->barge_in_energy_scale = 0.8f;
                break;
            default:
                pp->barge_in_energy_scale = 1.0f;
                break;
        }
    }

    /* Multi-voice: apply pitch shift for quoted speech segments.
     * This creates an audible voice differentiation without needing
     * a separate speaker embedding. */
    if (seg->voice[0] && strcasecmp(seg->voice, "quoted") == 0) {
        seg_pitch *= 1.08f;  /* +8% pitch for quoted voice */
        seg_rate *= 0.97f;   /* slightly slower for clarity */
    }

    /* Prosody feedback loop: apply boost from previous turn quality analysis */
    if (pp && pp->prosody_boost > 1.01f) {
        float boost = pp->prosody_boost;
        /* Widen pitch deviation from 1.0 */
        if (seg_pitch > 1.0f) seg_pitch = 1.0f + (seg_pitch - 1.0f) * boost;
        else if (seg_pitch < 1.0f) seg_pitch = 1.0f - (1.0f - seg_pitch) * boost;
        /* Widen volume deviation */
        if (fabsf(seg_vol_db) > 0.5f) seg_vol_db *= boost;
    }

    /* Log prosody parameters for dashboard visualization */
    if (pp && pp->prosody_log) {
        static const char *contour_names[] = {
            "declarative", "interrogative", "exclamatory",
            "imperative", "continuation", "list"
        };
        MultiScaleProsody msp_log = prosody_analyze_text(normalized);
        const char *contour = (msp_log.contour >= 0 && msp_log.contour <= 5)
            ? contour_names[msp_log.contour] : "unknown";
        int dur_ms = (int)(prosody_count_sentence_syllables(normalized) * 200);
        prosody_log_segment(pp->prosody_log, normalized,
                            seg_pitch, seg_rate, seg_vol_db,
                            seg->emotion, contour, dur_ms);
    }

    /* Feed text to TTS — use IPA override if available from <phoneme ph="..."> */
    if (seg->phoneme_ipa[0] && tts->set_text_ipa)
        tts->set_text_ipa(tts->engine, seg->phoneme_ipa, normalized);
    else
        tts->set_text(tts->engine, normalized);

segment_break_after:
    /* Insert break after segment */
    if (seg->break_after_ms > 0 && audio) {
        int silence_samples = 48000 * seg->break_after_ms / 1000;
        float silence[4096];
        memset(silence, 0, sizeof(silence));
        while (silence_samples > 0) {
            int chunk = silence_samples < 4096 ? silence_samples : 4096;
            voice_engine_write_playback(audio, silence, chunk);
            silence_samples -= chunk;
        }
    }
}

/**
 * Adaptive step batching: compute how many TTS steps to run per tick
 * based on audio ring buffer fill level and sentence count.
 *
 * When the playback ring is nearly empty, run more steps to avoid underrun.
 * When it's nearly full, run fewer steps to reduce latency.
 * For the first 2 sentences, run maximum steps for fastest first-chunk delivery.
 */
static int adaptive_steps_per_tick(VoiceEngine *audio, int sentence_count) {
    /* First sentence: run max steps for lowest first-chunk latency */
    if (sentence_count <= 1) return TTS_STEPS_PER_TICK_MAX;

    /* Heuristic: if playback ring is empty, the engine won't report "playing" */
    int playing = voice_engine_is_playing(audio);

    if (!playing) {
        /* Playback ring underrun — generate aggressively */
        return TTS_STEPS_PER_TICK_MAX;
    }

    /* Steady state: use default batch size */
    return TTS_STEPS_PER_TICK;
}

/* Main pipeline tick: called in a tight loop */
static PipelineState pipeline_tick(
    PipelineState state,
    VoiceEngine *audio,
    SttInterface *stt,
    TtsInterface *tts,
    LLMClient *llm,
    SttAccum *stt_accum,
    SentenceBuffer *sentbuf,
    char *transcript,
    int *transcript_len,
    float vad_threshold,
    TurnMetrics *metrics,
    AudioPostProcessor *pp,
    Arena *turn_arena,
    ConversationMemory *conv_memory,
    SpeakerDiarizer *diarizer,
    char *llm_response,
    int *llm_response_len
) {
    PipelineState next = state;

    /* Check for barge-in in any speaking/streaming state.
     * Emotion-aware: when TTS is speaking empathetic/calm content,
     * require a stronger voice signal before interrupting. */
    if ((state == STATE_STREAMING || state == STATE_SPEAKING) &&
        voice_engine_get_barge_in(audio)) {
        int allow_bargein = 1;

        if (pp && pp->barge_in_energy_scale > 1.01f) {
            float scale = pp->barge_in_energy_scale;
            voice_engine_set_vad_thresholds(audio,
                0.01f * scale, 0.005f * scale);
            int recheck_vad = voice_engine_get_vad_state(audio);
            if (recheck_vad < 2) allow_bargein = 0;
            voice_engine_set_vad_thresholds(audio, 0.01f, 0.005f);
        }

        if (!allow_bargein) {
            voice_engine_clear_barge_in(audio);
        } else {
        fprintf(stderr, "\n[pocket-voice] Barge-in detected%s\n",
                (pp && pp->last_tts_emotion != EMOTION_NEUTRAL)
                    ? " (emotion-gated)" : "");
        voice_engine_clear_barge_in(audio);
        barge_in_flush(audio);
        llm->cancel(llm->engine);
        if (tts->reset(tts->engine) != 0)
            fprintf(stderr, "[pocket-voice] WARNING: TTS reset failed on barge-in\n");
        if (stt->reset(stt->engine) != 0)
            fprintf(stderr, "[pocket-voice] WARNING: STT reset failed on barge-in\n");
        stt_accum_reset(stt_accum);
        sentbuf_reset(sentbuf);
        if (pp) {
            pp->speculative_sent = 0;
            pp->streaming_overlap = 0;
            pp->overlap_word_count = 0;
            pp->last_tts_emotion = EMOTION_NEUTRAL;
            pp->barge_in_energy_scale = 1.0f;
        }
        *transcript_len = 0;
        transcript[0] = '\0';
        return STATE_LISTENING;
        } /* else (allow_bargein) */
    }

    switch (state) {
    case STATE_LISTENING: {
        /* Drain capture buffer, watch for VAD speech onset */
        feed_stt(stt, audio, stt_accum, transcript, transcript_len, TEXT_BUF_SIZE, pp);

        int energy_vad = voice_engine_get_vad_state(audio);
        int vad_active = (pp && pp->speech_detector)
            ? speech_detector_speech_active(pp->speech_detector, energy_vad)
            : (energy_vad >= 1);
        if (vad_active) {
            next = STATE_RECORDING;
            *transcript_len = 0;
            transcript[0] = '\0';
            memset(metrics, 0, sizeof(*metrics));
            metrics->speech_start = now_us();
            print_state(next);
        }
        break;
    }

    case STATE_RECORDING: {
        if (!metrics->stt_start) metrics->stt_start = now_us();
        int nw = feed_stt(stt, audio, stt_accum, transcript, transcript_len, TEXT_BUF_SIZE, pp);
        if (nw > 0) {
            metrics->stt_frames++;
            fprintf(stderr, "\r[STT] %s", transcript);
            fflush(stderr);
        }

        /* ── Streaming STT→LLM Overlap ────────────────────────── *
         * After 8+ words, send the partial transcript to the LLM so it
         * starts generating while the user finishes. If the transcript
         * grows significantly (4+ new words), cancel and re-send with
         * the updated text. This cuts ~200-400ms off E2E latency for
         * longer utterances by overlapping LLM prefill with speech. */
        if (pp && !pp->speculative_sent && *transcript_len > 0) {
            int word_count = 0;
            for (int i = 0; i < *transcript_len; i++)
                if (transcript[i] == ' ') word_count++;
            word_count++; /* last word */

            if (!pp->streaming_overlap && word_count >= 5) {
                fprintf(stderr, "\n[pocket-voice] Streaming overlap: %d words, warming LLM\n",
                        word_count);
                llm->send(llm->engine, transcript);
                pp->streaming_overlap = 1;
                pp->overlap_word_count = word_count;
                metrics->llm_sent = now_us();
            } else if (pp->streaming_overlap &&
                       word_count >= pp->overlap_word_count + 4) {
                llm->cancel(llm->engine);
                llm->send(llm->engine, transcript);
                pp->overlap_word_count = word_count;
                metrics->llm_sent = now_us();
            }
        }

        /* ── Semantic EOU: feed transcript before fusion ── */
        if (pp && pp->semantic_eou && *transcript_len > 0 &&
            semantic_eou_word_count(transcript) >= 3) {
            float sem_prob = semantic_eou_process(pp->semantic_eou, transcript);
            speech_detector_feed_semantic(pp->speech_detector, sem_prob);
        }

        /* ── Fused EOU Detection (via SpeechDetector) ──
         * NOTE: fused_eou_process_partial() is available for Phase 2 parallel
         * EOU — allows early detection from energy+mimi before STT is ready.
         * Integration requires callback-driven parallelism refactoring. */
        int energy_vad = voice_engine_get_vad_state(audio);
        float stt_eou_prob = stt->has_vad(stt->engine)
            ? stt->get_vad_prob(stt->engine, 2) : 0.0f;

        bool end_of_turn = false;
        if (pp && pp->speech_detector) {
            EOUResult eou_res = speech_detector_eou(pp->speech_detector,
                                                     energy_vad, stt_eou_prob);
            if (eou_res.triggered)
                end_of_turn = true;

            /* Speculative prefill at 55%+ fused probability */
            if (!end_of_turn && !pp->speculative_sent &&
                eou_res.fused_prob >= 0.55f && *transcript_len > 0) {
                fprintf(stderr, "\n[pocket-voice] Speculative prefill (p=%.2f)\n",
                        eou_res.fused_prob);
                llm->send(llm->engine, transcript);
                pp->speculative_sent = 1;
                metrics->llm_sent = now_us();
            }

            /* Cancel speculative send if user resumes speaking */
            if (pp->speculative_sent && eou_res.fused_prob < 0.25f) {
                fprintf(stderr, "[pocket-voice] Speculative cancel (user resumed)\n");
                llm->cancel(llm->engine);
                pp->speculative_sent = 0;
                metrics->llm_sent = 0;
            }
        } else {
            if (stt_eou_prob > vad_threshold) {
                end_of_turn = true;
            } else if (energy_vad == 3) {
                end_of_turn = true;
            }
        }

        if (end_of_turn && *transcript_len > 0) {
            metrics->speech_end = now_us();
            metrics->stt_done = now_us();

            /* Flush STT to get remaining words */
            int nf = stt->flush(stt->engine);
            if (nf > 0) {
                char flush_buf[TEXT_BUF_SIZE];
                int flen = stt->get_text(stt->engine, flush_buf, TEXT_BUF_SIZE);
                if (flen > 0) {
                    int space = TEXT_BUF_SIZE - *transcript_len - 1;
                    int to_copy = flen < space ? flen : space;
                    if (to_copy > 0) {
                        memcpy(transcript + *transcript_len, flush_buf, (size_t)to_copy);
                        *transcript_len += to_copy;
                        transcript[*transcript_len] = '\0';
                    }
                }
            }

            fprintf(stderr, "\n[pocket-voice] User: %s\n", transcript);

            /* ── Conversational prosody: update user speech stats ──
             * Estimates speaking rate from speech duration + word count,
             * adjusts response prosody to match user's pace/energy. */
            if (pp && metrics->speech_start && metrics->speech_end) {
                float dur_sec = (float)(metrics->speech_end - metrics->speech_start) / 1e6f;
                int word_count = 0;
                { int in_w = 0;
                  for (int ci = 0; ci < *transcript_len; ci++) {
                      if (transcript[ci] == ' ') in_w = 0;
                      else if (!in_w) { word_count++; in_w = 1; }
                  }
                }
                prosody_conversation_update(&pp->conv_prosody, dur_sec,
                                            word_count, -20.0f, 150.0f);
            }

            /* Detect user emotion from transcript → pre-condition response */
            {
                EmotionDetection user_emo = prosody_detect_emotion(transcript);
                if (user_emo.confidence >= 0.3f) {
                    fprintf(stderr, "[prosody] Detected user emotion: %d (conf=%.2f)\n",
                            user_emo.emotion, user_emo.confidence);
                }
            }

            /* Adaptive system prompt: adjust LLM instructions based on
             * conversation dynamics (user pace, energy, emotion). Updated
             * every 3 turns to avoid prompt churn. */
            pp->turns_since_adapt++;
            if (pp->turns_since_adapt >= 3) {
                pp->turns_since_adapt = 0;
                (void)prosody_conversation_adapt(&pp->conv_prosody);
                float rate = pp->conv_prosody.ema_rate;
                float energy = pp->conv_prosody.ema_energy;

                pp->adaptive_suffix[0] = '\0';
                if (rate > 1.15f) {
                    strncat(pp->adaptive_suffix,
                            "\nThe user speaks quickly. Keep your responses brief and energetic.",
                            sizeof(pp->adaptive_suffix) - strlen(pp->adaptive_suffix) - 1);
                } else if (rate < 0.85f) {
                    strncat(pp->adaptive_suffix,
                            "\nThe user speaks slowly. Take your time, be thoughtful.",
                            sizeof(pp->adaptive_suffix) - strlen(pp->adaptive_suffix) - 1);
                }
                if (energy > -15.0f) {
                    strncat(pp->adaptive_suffix,
                            " They seem engaged — match their energy.",
                            sizeof(pp->adaptive_suffix) - strlen(pp->adaptive_suffix) - 1);
                } else if (energy < -25.0f) {
                    strncat(pp->adaptive_suffix,
                            " They're speaking quietly — be gentle.",
                            sizeof(pp->adaptive_suffix) - strlen(pp->adaptive_suffix) - 1);
                }

                /* Prosody feedback: if recent TTS output was monotone, ask
                 * the LLM to add more expressive tags */
                if (pp->prosody_boost > 1.1f) {
                    strncat(pp->adaptive_suffix,
                            " Use more <emphasis> and <emotion> tags for expressiveness.",
                            sizeof(pp->adaptive_suffix) - strlen(pp->adaptive_suffix) - 1);
                }

                if (pp->adaptive_suffix[0]) {
                    fprintf(stderr, "[adapt] Prompt suffix: %s\n", pp->adaptive_suffix);
                }
            }

            /* If speculative prefill or streaming overlap already sent,
             * the LLM is already working — skip to STREAMING. If the
             * transcript changed significantly, cancel and re-send. */
            if (pp && (pp->speculative_sent || pp->streaming_overlap)) {
                fprintf(stderr, "[pocket-voice] %s hit — skipping to STREAMING\n",
                        pp->speculative_sent ? "Speculative" : "Overlap");
                pp->speculative_sent = 0;
                pp->streaming_overlap = 0;
                pp->overlap_word_count = 0;
                next = STATE_STREAMING;
            } else {
                next = STATE_PROCESSING;
            }
            print_state(next);
        }
        break;
    }

    case STATE_PROCESSING: {
        metrics->llm_sent = now_us();

        /* Prepend conversation memory context and emotion cues if available.
           Speaker diarization runs asynchronously after LLM send to avoid
           adding 50-200ms to the critical Voice Response Latency path. */
        char *llm_input = (char *)transcript;
        char *mem_ctx = conv_memory ? memory_format_context(conv_memory) : NULL;
        char emotion_hint[256] = {0};
        if (pp && pp->audio_emotion) {
            AudioEmotionResult emo = audio_emotion_get(pp->audio_emotion);
            if (emo.confidence > 0.4f)
                audio_emotion_describe(&emo, emotion_hint, sizeof(emotion_hint));
        }
        char augmented[TEXT_BUF_SIZE * 2];
        if ((mem_ctx && mem_ctx[0]) || emotion_hint[0]) {
            int off = 0;
            if (mem_ctx && mem_ctx[0])
                off += snprintf(augmented + off, sizeof(augmented) - off, "%s\n", mem_ctx);
            if (emotion_hint[0])
                off += snprintf(augmented + off, sizeof(augmented) - off,
                                "[User tone: %s]\n", emotion_hint);
            snprintf(augmented + off, sizeof(augmented) - off, "%s", transcript);
            llm_input = augmented;
        }
        if (mem_ctx) free(mem_ctx);

        if (llm->send(llm->engine, llm_input) != 0) {
            fprintf(stderr, "[pocket-voice] Failed to send to LLM\n");
            next = STATE_LISTENING;
        } else {
            /* Save user turn to memory */
            if (conv_memory) memory_add_turn(conv_memory, "user", transcript);

            /* Run diarization AFTER LLM send (off critical path) */
            if (diarizer && pp && pp->rec_16k && pp->rec_16k_len >= 8000) {
                int spk = diarizer_identify(diarizer, pp->rec_16k, pp->rec_16k_len);
                if (spk >= 0) {
                    const char *label = diarizer_get_label(diarizer, spk);
                    fprintf(stderr, "[Speaker %d%s%s] ", spk,
                            label ? ": " : "", label ? label : "");
                }
            }

            next = STATE_STREAMING;
            print_state(next);
        }
        break;
    }

    case STATE_STREAMING: {
        /* Poll for LLM tokens (wait up to 1ms for data) */
        llm->poll(llm->engine, 1);

        int token_len = 0;
        const char *tokens = llm->peek_tokens(llm->engine, &token_len);
        if (tokens && token_len > 0) {
            metrics->llm_tokens++;
            if (!metrics->has_first_tok) {
                metrics->llm_first_tok = now_us();
                metrics->has_first_tok = true;
                uint64_t ttft = metrics->llm_first_tok - metrics->llm_sent;
                fprintf(stderr, "[LLM TTFT: %llu ms] ", (unsigned long long)(ttft / 1000));
            }

            char token_copy[4096];
            int copy_len = token_len < (int)sizeof(token_copy) - 1
                               ? token_len
                               : (int)sizeof(token_copy) - 1;
            memcpy(token_copy, tokens, (size_t)copy_len);
            token_copy[copy_len] = '\0';

            /* Accumulate for conversation memory */
            if (conv_memory && *llm_response_len + copy_len < TEXT_BUF_SIZE - 1) {
                memcpy(llm_response + *llm_response_len, token_copy, (size_t)copy_len);
                *llm_response_len += copy_len;
                llm_response[*llm_response_len] = '\0';
            }
            llm->consume_tokens(llm->engine, copy_len);

            /* Feed tokens to sentence buffer instead of directly to TTS */
            sentbuf_add(sentbuf, token_copy, copy_len);

            /* Speculative TTS warmup: while accumulating the first sentence,
               run empty TTS steps to keep the Metal command buffer warm and
               the GPU pipeline primed. We do NOT feed text here — that would
               cause duplication when the sentence buffer later flushes the
               complete segment through process_segment(). */
            if (sentbuf_sentence_count(sentbuf) == 0 && !sentbuf_has_segment(sentbuf)) {
                tts->step(tts->engine);
            }

            fprintf(stderr, "%s", token_copy);
            fflush(stderr);
        }

        /* When sentence buffer has a complete segment, process it */
        while (sentbuf_has_segment(sentbuf)) {
            char sentence[4096];
            int slen = sentbuf_flush(sentbuf, sentence, sizeof(sentence));
            if (slen <= 0) break;

            /* Get prosody hints detected during accumulation */
            SentBufProsodyHint sbhint = sentbuf_get_prosody_hint(sentbuf);

            /* Emphasis prediction: add <emphasis> to important words if the
             * LLM didn't produce SSML markup. Also detect quoted speech for
             * multi-voice rendering. */
            char with_quotes[8192];
            char enhanced[8192];
            emphasis_detect_quotes(sentence, with_quotes, sizeof(with_quotes));
            emphasis_predict(with_quotes, enhanced, sizeof(enhanced));

            /* Parse through SSML (passthrough if not SSML) then normalize */
            SSMLSegment segments[SSML_MAX_SEGMENTS];
            int nseg = ssml_parse(enhanced, segments, SSML_MAX_SEGMENTS);

            /* Apply sentence buffer prosody hints to non-SSML segments */
            for (int s = 0; s < nseg; s++) {
                if (fabsf(segments[s].pitch - 1.0f) < 0.01f)
                    segments[s].pitch *= sbhint.suggested_pitch;
                if (fabsf(segments[s].rate - 1.0f) < 0.01f)
                    segments[s].rate *= sbhint.suggested_rate;
                if (fabsf(segments[s].volume - 1.0f) < 0.01f && fabsf(sbhint.suggested_energy) > 0.1f)
                    segments[s].volume *= powf(10.0f, sbhint.suggested_energy / 20.0f);
                process_segment(&segments[s], tts, audio, pp, metrics);
            }
        }

        /* On response done, flush remaining buffer through the pipeline */
        if (llm->is_response_done(llm->engine)) {
            char remaining[4096];
            int rlen = sentbuf_flush_all(sentbuf, remaining, sizeof(remaining));
            if (rlen > 0) {
                SSMLSegment segments[SSML_MAX_SEGMENTS];
                int nseg = ssml_parse(remaining, segments, SSML_MAX_SEGMENTS);
                for (int s = 0; s < nseg; s++) {
                    process_segment(&segments[s], tts, audio, pp, metrics);
                }
            }
            if (!tts->is_done(tts->engine)) {
                tts->set_text_done(tts->engine);
            }
        }

        /* Run adaptive number of TTS steps per tick based on buffer fill */
        int steps = adaptive_steps_per_tick(audio, sentbuf_sentence_count(sentbuf));
        for (int i = 0; i < steps; i++) {
            int step_result = tts->step(tts->engine);
            if (step_result != 0) break; /* done or error */
        }

        int wrote = feed_speaker(tts, audio, pp);
        if (wrote > 0 && !metrics->has_first_audio) {
            metrics->tts_first_audio = now_us();
            metrics->has_first_audio = true;
            uint64_t e2e = metrics->tts_first_audio - metrics->speech_end;
            fprintf(stderr, "[E2E: %llu ms] ", (unsigned long long)(e2e / 1000));
        }

        if (llm->has_error(llm->engine)) {
            tts->set_text_done(tts->engine);
            feed_speaker(tts, audio, pp);
            fprintf(stderr, "\n[pocket-voice] LLM error, draining TTS...\n");
            metrics->speaking_entered = now_us();
            next = STATE_SPEAKING;
            print_state(next);
        } else if (llm->is_response_done(llm->engine) && tts->is_done(tts->engine)) {
            feed_speaker(tts, audio, pp);
            fprintf(stderr, "\n");
            metrics->speaking_entered = now_us();
            next = STATE_SPEAKING;
            print_state(next);
        }
        break;
    }

    case STATE_SPEAKING: {
        /* Continue draining any remaining TTS steps (max burst for fast drain) */
        if (!tts->is_done(tts->engine)) {
            for (int i = 0; i < TTS_STEPS_PER_TICK_MAX; i++) {
                int r = tts->step(tts->engine);
                if (r != 0) break;
            }
        }
        feed_speaker(tts, audio, pp);

        bool done = !voice_engine_is_playing(audio) && tts->is_done(tts->engine);
        bool timed_out = (now_us() - metrics->speaking_entered) > SPEAKING_TIMEOUT_US;
        if (timed_out && !done) {
            fprintf(stderr, "\n[pocket-voice] SPEAKING timeout (30s), forcing reset\n");
        }

        if (done || timed_out) {
            /* Commit this turn to conversation history */
            llm->commit_turn(llm->engine, transcript);

            /* Save assistant response to persistent memory */
            if (conv_memory && *llm_response_len > 0) {
                memory_add_turn(conv_memory, "assistant", llm_response);
            }
            *llm_response_len = 0;
            llm_response[0] = '\0';

            /* Print turn latency summary */
            if (metrics->has_first_audio) {
                uint64_t total = now_us() - metrics->speech_start;
                fprintf(stderr, "[Turn: %llu ms total]\n",
                        (unsigned long long)(total / 1000));
            }

            /* Report audio drops (if any) since last turn */
            {
                uint64_t cap_drops = 0, play_drops = 0;
                voice_engine_get_drop_counts(audio, &cap_drops, &play_drops);
                if (cap_drops > 0)
                    fprintf(stderr, "[pocket-voice] WARNING: %llu capture frames dropped (ring full)\n",
                            (unsigned long long)cap_drops);
                if (play_drops > 0)
                    fprintf(stderr, "[pocket-voice] WARNING: %llu playback frames dropped (ring full)\n",
                            (unsigned long long)play_drops);
            }

            metrics->turn_complete = now_us();
            print_turn_latency(metrics);

            /* Prosody feedback loop: analyze quality of recent TTS output
             * and adjust boost factor for next turn.
             * Low F0 range or low energy variance → increase expressiveness. */
            if (pp) {
                float target_f0_range = 50.0f;  /* Hz — minimum for natural speech */
                float target_energy_var = 4.0f;  /* dB^2 — minimum dynamic range */
                float f0_ok = (pp->recent_f0_range >= target_f0_range) ? 1.0f : 0.5f;
                float e_ok = (pp->recent_energy_var >= target_energy_var) ? 1.0f : 0.5f;
                float quality = 0.6f * f0_ok + 0.4f * e_ok;

                /* Adjust boost: if quality is poor, increase; if good, decay toward 1.0 */
                if (quality < 0.7f) {
                    pp->prosody_boost = fminf(pp->prosody_boost + 0.05f, 1.3f);
                    fprintf(stderr, "[prosody-feedback] Low expressiveness → boost=%.2f\n",
                            pp->prosody_boost);
                } else if (pp->prosody_boost > 1.0f) {
                    pp->prosody_boost = fmaxf(pp->prosody_boost - 0.02f, 1.0f);
                }
            }

            /* Log complete turn to prosody dashboard */
            if (pp && pp->prosody_log) {
                static int turn_id = 0;
                turn_id++;
                float vrl = (metrics->speech_end && metrics->has_first_audio)
                    ? (float)mach_to_ms(metrics->speech_end, metrics->tts_first_audio)
                    : 0.0f;
                float rtf = 0.0f;
                if (metrics->tts_audio_sec > 0 && metrics->tts_start && metrics->turn_complete) {
                    double gen_time = mach_to_ms(metrics->tts_start, metrics->turn_complete) / 1000.0;
                    rtf = (float)(gen_time / metrics->tts_audio_sec);
                }
                prosody_log_turn(pp->prosody_log, turn_id, transcript, NULL,
                                 pp->conv_prosody.ema_rate,
                                 1.0f, pp->conv_prosody.ema_rate,
                                 pp->conv_prosody.ema_energy,
                                 vrl, rtf);
            }

            if (tts->reset(tts->engine) != 0)
                fprintf(stderr, "[pocket-voice] WARNING: TTS reset failed at turn end\n");
            if (stt->reset(stt->engine) != 0)
                fprintf(stderr, "[pocket-voice] WARNING: STT reset failed at turn end\n");
            stt_accum_reset(stt_accum);
            sentbuf_reset(sentbuf);
            postproc_reset(pp);
            voice_engine_clear_barge_in(audio);
            voice_engine_flush_playback(audio);
            *transcript_len = 0;
            transcript[0] = '\0';
            next = STATE_LISTENING;
            print_state(next);
        }
        break;
    }
    }

    return next;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * CLI argument parsing
 * ═══════════════════════════════════════════════════════════════════════════ */

static void print_usage(const char *prog) {
    fprintf(stderr,
        "Usage: %s [OPTIONS]\n\n"
        "Zero-Python voice pipeline: Mic → STT → LLM → TTS → Speaker\n\n"
        "Options:\n"
        "  --voice PATH       Voice .wav or .safetensors path for cloning\n"
        "  --stt-engine E     STT engine: rust (default), conformer, bnns, or sonata\n"
        "  --cstt-model PATH  Conformer STT .cstt model file path\n"
        "  --bnns-model PATH  BNNS .mlmodelc path (ANE accelerated)\n"
        "  --metallib PATH    Custom .metallib path for GPU kernels\n"
        "  --beam-size N      CTC beam search width (0=greedy, default: 0)\n"
        "  --lm-path PATH     KenLM language model (.bin or .arpa) for beam search\n"
        "  --lm-weight F      LM score weight (default: 1.5)\n"
        "  --word-score F     Per-word insertion bonus (default: 0.0)\n"
        "  --profiler         Enable per-turn latency profiling\n"
        "  --stt-repo REPO    STT HuggingFace repo (default: kyutai/stt-1b-en_fr-candle)\n"
        "  --stt-model FILE   STT model file (default: model.safetensors)\n"
        "  --tts-repo REPO    TTS HuggingFace repo (default: kyutai/tts-1.6b-en_fr)\n"
        "  --tts-engine E     TTS engine: sonata (default), sonata-v2, or sonata-v3\n"
        "  --llm ENGINE       LLM backend: claude (default) or gemini\n"
        "  --llm-model M      LLM model name (auto-detected per engine)\n"
        "  --system PROMPT    System prompt for LLM\n"
        "  --prosody          Enable SSML prosody-aware system prompt\n"
        "  --phonemize        Enable espeak-ng IPA phonemization for TTS\n"
        "  --phoneme-map PATH JSON phoneme-to-ID mapping file\n"
        "  --speaker-encoder PATH  ONNX speaker encoder model for voice cloning\n"
        "  --ref-wav PATH     Reference WAV for voice cloning\n"
        "  --sonata-stt-model PATH Sonata CTC STT .cstt_sonata (default: models/sonata/sonata_stt.cstt_sonata)\n"
        "  --sonata-refiner PATH  Optional .cref refiner for two-pass STT\n"
        "  --flow-v2-weights PATH  Sonata Flow v2 .safetensors (for --tts-engine sonata-v2)\n"
        "  --flow-v2-config PATH  Sonata Flow v2 config JSON\n"
        "  --flow-v3-weights PATH  Sonata Flow v3 .safetensors (for --tts-engine sonata-v3)\n"
        "  --flow-v3-config PATH  Sonata Flow v3 config JSON\n"
        "  --vocoder-weights PATH  Sonata Vocoder weights (for --tts-engine sonata-v3)\n"
        "  --vocoder-config PATH   Sonata Vocoder config JSON\n"
        "  --quality MODE     TTS quality: fast (0), balanced (1, default), high (2)\n"
        "  --n-q N            Audio codebooks for TTS (default: 24)\n"
        "  --no-vad           Disable semantic VAD (use energy VAD only)\n"
        "  --vad-threshold F  Semantic VAD threshold (default: 0.7)\n"
        "\n"
        "Audio post-processing:\n"
        "  --pitch F          Pitch multiplier (1.0 = normal, 1.2 = higher)\n"
        "  --volume F         Volume in dB (0.0 = normal, 6.0 = louder)\n"
        "  --no-hw-resample   Disable AudioConverter (use FIR fallback)\n"
        "  --spatial AZ       Enable 3D spatial audio at azimuth AZ degrees\n"
        "  --vad PATH         Native C VAD weights (e.g. models/silero_vad.nvad)\n"
        "\n"
        "Opus output:\n"
        "  --opus-bitrate N   Opus bitrate in bps (e.g. 64000). 0 = disabled\n"
        "  --opus-output PATH Path for Opus output file\n"
        "\n"
        "Sentence buffering:\n"
        "  --sentence-mode    Use sentence-only mode (default: speculative)\n"
        "  --min-words N      Min words before clause flush (default: 5)\n"
        "\n"
        "Remote microphone (use your phone as a mic):\n"
        "  --remote           Start web server for phone mic input\n"
        "  --remote-port N    Port for remote mic server (default: 8088)\n"
        "\n"
        "Configuration:\n"
        "  --config FILE      Load JSON config file (CLI args override)\n"
        "  --server           Run as HTTP API server instead of interactive\n"
        "  --server-port N    Port for HTTP API server (default: 8080)\n"
        "  --help             Show this help\n"
        "\n"
        "Environment variables:\n"
        "  ANTHROPIC_API_KEY  Required for --llm claude\n"
        "  GEMINI_API_KEY     Required for --llm gemini\n",
        prog);
}

/* Remote mic callback: upsample 16kHz→48kHz and inject into capture ring.
 * Uses vDSP_vlint (vector linear interpolation) for proper anti-aliased
 * upsampling instead of zero-order hold which creates aliasing artifacts. */
static void remote_mic_audio_cb(void *user_ctx, const float *pcm, int n_samples)
{
    VoiceEngine *eng = (VoiceEngine *)user_ctx;
    if (n_samples <= 0) return;
    int n_up = n_samples * 3;
    if (n_up > 16384) { n_samples = 16384 / 3; n_up = n_samples * 3; }
    float up[16384];

    /* Build fractional index array for 3x upsampling: 0.0, 0.333, 0.667, 1.0, ... */
    float indices[16384];
    float base = 0.0f;
    float step = 1.0f / 3.0f;
    vDSP_vramp(&base, &step, indices, 1, (vDSP_Length)n_up);

    /* vDSP_vlint: vector linear interpolation — proper anti-aliased upsampling */
    vDSP_vlint(pcm, indices, 1, up, 1, (vDSP_Length)n_up, (vDSP_Length)n_samples);

    voice_engine_write_capture(eng, up, n_up);
}

static PipelineConfig parse_args_with_base(int argc, char **argv, PipelineConfig cfg) {
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            print_usage(argv[0]);
            exit(0);
        } else if (strcmp(argv[i], "--voice") == 0 && i + 1 < argc) {
            cfg.voice = argv[++i];
        } else if (strcmp(argv[i], "--stt-engine") == 0 && i + 1 < argc) {
            const char *e = argv[++i];
            if (strcmp(e, "conformer") == 0 || strcmp(e, "c") == 0)
                cfg.stt_engine = STT_ENGINE_CONFORMER;
            else if (strcmp(e, "bnns") == 0 || strcmp(e, "ane") == 0)
                cfg.stt_engine = STT_ENGINE_BNNS;
            else if (strcmp(e, "sonata") == 0)
                cfg.stt_engine = STT_ENGINE_SONATA;
            else
                cfg.stt_engine = STT_ENGINE_RUST;
        } else if (strcmp(argv[i], "--sonata-stt-model") == 0 && i + 1 < argc) {
            cfg.sonata_stt_model = argv[++i];
        } else if (strcmp(argv[i], "--sonata-refiner") == 0 && i + 1 < argc) {
            cfg.sonata_refiner_path = argv[++i];
        } else if (strcmp(argv[i], "--cstt-model") == 0 && i + 1 < argc) {
            cfg.cstt_model = argv[++i];
        } else if (strcmp(argv[i], "--bnns-model") == 0 && i + 1 < argc) {
            cfg.bnns_model = argv[++i];
        } else if (strcmp(argv[i], "--metallib") == 0 && i + 1 < argc) {
            cfg.metallib_path = argv[++i];
        } else if (strcmp(argv[i], "--beam-size") == 0 && i + 1 < argc) {
            cfg.beam_size = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--lm-path") == 0 && i + 1 < argc) {
            cfg.lm_path = argv[++i];
        } else if (strcmp(argv[i], "--lm-weight") == 0 && i + 1 < argc) {
            cfg.lm_weight = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "--word-score") == 0 && i + 1 < argc) {
            cfg.word_score = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "--profiler") == 0) {
            cfg.enable_profiler = 1;
        } else if (strcmp(argv[i], "--stt-repo") == 0 && i + 1 < argc) {
            cfg.stt_repo = argv[++i];
        } else if (strcmp(argv[i], "--stt-model") == 0 && i + 1 < argc) {
            cfg.stt_model = argv[++i];
        } else if (strcmp(argv[i], "--tts-engine") == 0 && i + 1 < argc) {
            const char *e = argv[++i];
            if (strcmp(e, "sonata-v2") == 0)
                cfg.tts_engine = TTS_ENGINE_SONATA_V2;
            else if (strcmp(e, "sonata-v3") == 0)
                cfg.tts_engine = TTS_ENGINE_SONATA_V3;
            else
                cfg.tts_engine = TTS_ENGINE_SONATA;
        } else if (strcmp(argv[i], "--ctts-model") == 0 && i + 1 < argc) {
            fprintf(stderr, "[pocket-voice] --ctts-model is deprecated (Kyutai C TTS removed)\n");
            ++i;
        } else if (strcmp(argv[i], "--ctts-voice") == 0 && i + 1 < argc) {
            fprintf(stderr, "[pocket-voice] --ctts-voice is deprecated (Kyutai C TTS removed)\n");
            ++i;
        } else if (strcmp(argv[i], "--sonata-lm") == 0 && i + 1 < argc) {
            cfg.sonata_lm_weights = argv[++i];
        } else if (strcmp(argv[i], "--sonata-config") == 0 && i + 1 < argc) {
            cfg.sonata_lm_config = argv[++i];
        } else if (strcmp(argv[i], "--sonata-tokenizer") == 0 && i + 1 < argc) {
            cfg.sonata_tokenizer = argv[++i];
        } else if (strcmp(argv[i], "--sonata-flow") == 0 && i + 1 < argc) {
            cfg.sonata_flow_weights = argv[++i];
        } else if (strcmp(argv[i], "--sonata-flow-config") == 0 && i + 1 < argc) {
            cfg.sonata_flow_config = argv[++i];
        } else if (strcmp(argv[i], "--sonata-decoder") == 0 && i + 1 < argc) {
            cfg.sonata_dec_weights = argv[++i];
        } else if (strcmp(argv[i], "--sonata-decoder-config") == 0 && i + 1 < argc) {
            cfg.sonata_dec_config = argv[++i];
        } else if (strcmp(argv[i], "--sonata-speaker") == 0 && i + 1 < argc) {
            cfg.sonata_speaker = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--sonata-cfg") == 0 && i + 1 < argc) {
            cfg.sonata_cfg_scale = atof(argv[++i]);
        } else if (strcmp(argv[i], "--sonata-steps") == 0 && i + 1 < argc) {
            cfg.sonata_flow_steps = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--sonata-heun") == 0) {
            cfg.sonata_heun = 1;
        } else if (strcmp(argv[i], "--quality") == 0 && i + 1 < argc) {
            const char *q = argv[++i];
            if (strcmp(q, "fast") == 0 || strcmp(q, "0") == 0)
                cfg.tts_quality_mode = 0;
            else if (strcmp(q, "balanced") == 0 || strcmp(q, "1") == 0)
                cfg.tts_quality_mode = 1;
            else if (strcmp(q, "high") == 0 || strcmp(q, "2") == 0)
                cfg.tts_quality_mode = 2;
        } else if (strcmp(argv[i], "--sonata-self-draft") == 0) {
            cfg.sonata_self_draft = 1;
        } else if (strcmp(argv[i], "--sonata-draft") == 0 && i + 1 < argc) {
            cfg.sonata_draft_weights = argv[++i];
        } else if (strcmp(argv[i], "--sonata-draft-config") == 0 && i + 1 < argc) {
            cfg.sonata_draft_config = argv[++i];
        } else if (strcmp(argv[i], "--sonata-speculate-k") == 0 && i + 1 < argc) {
            cfg.sonata_speculate_k = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--sonata-ref-wav") == 0 && i + 1 < argc) {
            cfg.sonata_ref_wav = argv[++i];
        } else if (strcmp(argv[i], "--sonata-storm") == 0 && i + 1 < argc) {
            cfg.sonata_storm_weights = argv[++i];
        } else if (strcmp(argv[i], "--sonata-storm-config") == 0 && i + 1 < argc) {
            cfg.sonata_storm_config = argv[++i];
        } else if (strcmp(argv[i], "--flow-v2-weights") == 0 && i + 1 < argc) {
            cfg.sonata_flow_v2_weights = argv[++i];
        } else if (strcmp(argv[i], "--flow-v2-config") == 0 && i + 1 < argc) {
            cfg.sonata_flow_v2_config = argv[++i];
        } else if (strcmp(argv[i], "--flow-v3-weights") == 0 && i + 1 < argc) {
            cfg.flow_v3_weights = argv[++i];
        } else if (strcmp(argv[i], "--flow-v3-config") == 0 && i + 1 < argc) {
            cfg.flow_v3_config = argv[++i];
        } else if (strcmp(argv[i], "--vocoder-weights") == 0 && i + 1 < argc) {
            cfg.vocoder_weights = argv[++i];
        } else if (strcmp(argv[i], "--vocoder-config") == 0 && i + 1 < argc) {
            cfg.vocoder_config = argv[++i];
        } else if (strcmp(argv[i], "--tts-repo") == 0 && i + 1 < argc) {
            cfg.tts_repo = argv[++i];
        } else if (strcmp(argv[i], "--llm") == 0 && i + 1 < argc) {
            const char *e = argv[++i];
            if (strcmp(e, "gemini") == 0)
                cfg.llm_engine = LLM_ENGINE_GEMINI;
            else if (strcmp(e, "local") == 0)
                cfg.llm_engine = LLM_ENGINE_LOCAL;
            else
                cfg.llm_engine = LLM_ENGINE_CLAUDE;
        } else if (strcmp(argv[i], "--llm-model") == 0 && i + 1 < argc) {
            cfg.llm_model = argv[++i];
        } else if (strcmp(argv[i], "--claude-model") == 0 && i + 1 < argc) {
            cfg.llm_model = argv[++i];
        } else if (strcmp(argv[i], "--prosody") == 0) {
            cfg.prosody_prompt = 1;
        } else if (strcmp(argv[i], "--phonemize") == 0) {
            cfg.use_phonemizer = 1;
        } else if (strcmp(argv[i], "--phoneme-map") == 0 && i + 1 < argc) {
            cfg.phoneme_map_path = argv[++i];
        } else if (strcmp(argv[i], "--pronunciation-dict") == 0 && i + 1 < argc) {
            cfg.pronunciation_dict_path = argv[++i];
        } else if (strcmp(argv[i], "--speaker-encoder") == 0 && i + 1 < argc) {
            cfg.speaker_encoder_path = argv[++i];
        } else if (strcmp(argv[i], "--ref-wav") == 0 && i + 1 < argc) {
            cfg.ref_wav_path = argv[++i];
        } else if (strcmp(argv[i], "--clone-voice") == 0 && i + 1 < argc) {
            cfg.clone_voice_path = argv[++i];
        } else if (strcmp(argv[i], "--system") == 0 && i + 1 < argc) {
            cfg.system_prompt = argv[++i];
        } else if (strcmp(argv[i], "--n-q") == 0 && i + 1 < argc) {
            cfg.n_q = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--no-vad") == 0) {
            cfg.enable_vad = 0;
        } else if (strcmp(argv[i], "--vad-threshold") == 0 && i + 1 < argc) {
            cfg.vad_threshold = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "--pitch") == 0 && i + 1 < argc) {
            cfg.pitch = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "--volume") == 0 && i + 1 < argc) {
            cfg.volume_db = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "--no-hw-resample") == 0) {
            cfg.hw_resample = 0;
        } else if (strcmp(argv[i], "--spatial") == 0 && i + 1 < argc) {
            cfg.spatial = 1;
            cfg.spatial_az = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "--vad") == 0 && i + 1 < argc) {
            cfg.native_vad_path = argv[++i];
        } else if (strcmp(argv[i], "--semantic-eou") == 0 && i + 1 < argc) {
            cfg.semantic_eou_path = argv[++i];
        } else if (strcmp(argv[i], "--silero-vad") == 0 && i + 1 < argc) {
            fprintf(stderr, "[pocket-voice] --silero-vad is deprecated; use --vad with .nvad weights\n");
            ++i;
        } else if (strcmp(argv[i], "--emosteer") == 0 && i + 1 < argc) {
            cfg.emosteer_path = argv[++i];
        } else if (strcmp(argv[i], "--prosody-log") == 0 && i + 1 < argc) {
            cfg.prosody_log_path = argv[++i];
        } else if (strcmp(argv[i], "--opus-bitrate") == 0 && i + 1 < argc) {
            cfg.opus_bitrate = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--opus-output") == 0 && i + 1 < argc) {
            cfg.opus_output = argv[++i];
        } else if (strcmp(argv[i], "--sentence-mode") == 0) {
            cfg.sentbuf_mode = SENTBUF_MODE_SENTENCE;
        } else if (strcmp(argv[i], "--min-words") == 0 && i + 1 < argc) {
            cfg.sentbuf_min_words = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--remote") == 0) {
            cfg.remote_mic = 1;
        } else if (strcmp(argv[i], "--remote-port") == 0 && i + 1 < argc) {
            cfg.remote_port = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--config") == 0 && i + 1 < argc) {
            cfg.config_file = argv[++i];
        } else if (strcmp(argv[i], "--server") == 0) {
            cfg.server_mode = 1;
        } else if (strcmp(argv[i], "--server-port") == 0 && i + 1 < argc) {
            cfg.server_port = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--memory") == 0 && i + 1 < argc) {
            cfg.memory_path = argv[++i];
        } else if (strcmp(argv[i], "--memory-turns") == 0 && i + 1 < argc) {
            cfg.memory_max_turns = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--diarizer") == 0 && i + 1 < argc) {
            cfg.diarizer_encoder = argv[++i];
        } else if (strcmp(argv[i], "--diarizer-threshold") == 0 && i + 1 < argc) {
            cfg.diarizer_threshold = (float)atof(argv[++i]);
        } else {
            fprintf(stderr, "Unknown option: %s\n", argv[i]);
            print_usage(argv[0]);
            exit(1);
        }
    }
    return cfg;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * HTTP API: Full prosody pipeline for /v1/audio/speech
 *
 * Runs emphasis detection → SSML parse → process_segment for each segment,
 * applying API-level speed/volume/emotion overrides. This gives HTTP callers
 * the same rich prosody pipeline as the interactive voice loop.
 * ═══════════════════════════════════════════════════════════════════════════ */

typedef struct {
    TtsInterface        *tts;
    AudioPostProcessor  *pp;
} HttpProcessCtx;

static int http_process_text_impl(void *ctx, void *tts_engine_unused,
                                   const char *text, float speed, float volume,
                                   const char *emotion, const char *voice,
                                   const char (*pron_words)[64],
                                   const char (*pron_replacements)[256],
                                   int n_pron) {
    HttpProcessCtx *hctx = (HttpProcessCtx *)ctx;
    if (!hctx || !hctx->tts || !text) return -1;

    TtsInterface *tts = hctx->tts;
    AudioPostProcessor *pp = hctx->pp;

    tts->reset(tts->engine);

    /* Step 0a: Apply inline pronunciation overrides from API request */
    char pron_applied[8192];
    if (pron_words && pron_replacements && n_pron > 0) {
        text_apply_pronunciation_dict(text, pron_applied, sizeof(pron_applied),
                                      pron_words, pron_replacements, n_pron);
    } else {
        snprintf(pron_applied, sizeof(pron_applied), "%s", text);
    }

    /* Step 0b: Apply file-based pronunciation dictionary */
    char dict_applied[8192];
    if (pp && pp->pronunciation_dict) {
        pronunciation_dict_apply(pp->pronunciation_dict, pron_applied,
                                 dict_applied, sizeof(dict_applied));
    } else {
        snprintf(dict_applied, sizeof(dict_applied), "%s", pron_applied);
    }

    /* Step 1: Nonverbalisms → Inline IPA expansion → Quote detection → emphasis prediction */
    char nv_expanded[8192];
    text_expand_nonverbalisms(dict_applied, nv_expanded, sizeof(nv_expanded));
    char ipa_expanded[8192];
    text_expand_inline_ipa(nv_expanded, ipa_expanded, sizeof(ipa_expanded));
    char quoted_buf[8192];
    emphasis_detect_quotes(ipa_expanded, quoted_buf, sizeof(quoted_buf));
    char enhanced_buf[8192];
    emphasis_predict(quoted_buf, enhanced_buf, sizeof(enhanced_buf));

    /* Step 2: Wrap with API-level emotion if specified */
    char ssml_input[8192];
    if (emotion && *emotion && *emotion != '\0') {
        snprintf(ssml_input, sizeof(ssml_input),
                 "<speak><emotion type=\"%s\">%s</emotion></speak>",
                 emotion, enhanced_buf);
    } else if (ssml_is_ssml(enhanced_buf)) {
        snprintf(ssml_input, sizeof(ssml_input), "%s", enhanced_buf);
    } else {
        snprintf(ssml_input, sizeof(ssml_input), "<speak>%s</speak>", enhanced_buf);
    }

    /* Step 3: Parse SSML into segments */
    SSMLSegment segments[SSML_MAX_SEGMENTS];
    int n_segs = ssml_parse(ssml_input, segments, SSML_MAX_SEGMENTS);
    if (n_segs <= 0) {
        tts->set_text(tts->engine, text);
        int ret = tts->set_text_done(tts->engine);
        if (ret != 0) {
            fprintf(stderr, "[pipeline] TTS synthesis failed (set_text_done=%d)\n", ret);
            return -1;
        }
        return 0;
    }

    /* Step 4: Apply API-level speed/volume overrides and voice */
    for (int i = 0; i < n_segs; i++) {
        segments[i].rate *= speed;
        segments[i].volume *= volume;
        if (voice && *voice && segments[i].voice[0] == '\0')
            snprintf(segments[i].voice, SSML_MAX_VOICE, "%s", voice);
    }

    /* Step 5: Apply voice → speaker ID mapping if a numeric voice ID is given.
     * Named voices are passed through to the segment voice field for
     * process_segment to handle via per-segment voice switching. Integer IDs
     * (e.g. "0", "3") set the Sonata Flow speaker embedding table directly. */
    if (voice && *voice) {
        char *endp = NULL;
        long vid = strtol(voice, &endp, 10);
        if (endp && *endp == '\0' && vid >= 0) {
            SonataEngine *se = (SonataEngine *)tts->engine;
            if (se && se->flow_engine)
                sonata_flow_set_speaker(se->flow_engine, (int)vid);
        }
    }

    /* Step 6: Process each segment through the full prosody pipeline.
     * NULL audio engine skips speaker playback (break silences handled by
     * Sonata pause tokens for Sonata engine, skipped for others). */
    TurnMetrics http_metrics = {0};
    for (int i = 0; i < n_segs; i++) {
        process_segment(&segments[i], tts, NULL, pp, &http_metrics);
    }

    int ret = tts->set_text_done(tts->engine);
    if (ret != 0) {
        fprintf(stderr, "[pipeline] TTS synthesis failed (set_text_done=%d)\n", ret);
        return -1;
    }
    (void)tts_engine_unused;
    return 0;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * main()
 * ═══════════════════════════════════════════════════════════════════════════ */

int main(int argc, char **argv) {
    /* Two-pass config: load JSON file as base, then CLI args override.
     * Pass 1: Quick scan for --config path.
     * Pass 2: Load file into defaults, then parse CLI args on top. */
    const char *config_path = NULL;
    for (int i = 1; i < argc - 1; i++) {
        if (strcmp(argv[i], "--config") == 0) { config_path = argv[i + 1]; break; }
    }
    PipelineConfig file_cfg = default_config();
    if (config_path) load_config_file(&file_cfg, config_path);
    PipelineConfig cfg = parse_args_with_base(argc, argv, file_cfg);

    /* Check for API key based on LLM engine */
    const char *api_key = NULL;
    if (cfg.llm_engine == LLM_ENGINE_GEMINI) {
        api_key = getenv("GEMINI_API_KEY");
        if (!api_key || strlen(api_key) == 0) {
            fprintf(stderr, "[pocket-voice] Error: GEMINI_API_KEY not set\n");
            return 1;
        }
    } else {
        api_key = getenv("ANTHROPIC_API_KEY");
        if (!api_key || strlen(api_key) == 0) {
            fprintf(stderr, "[pocket-voice] Error: ANTHROPIC_API_KEY not set\n");
            return 1;
        }
    }

    /* Resolve default model names per engine */
    const char *llm_model = cfg.llm_model;
    if (!llm_model) {
        llm_model = (cfg.llm_engine == LLM_ENGINE_GEMINI)
            ? "gemini-2.5-flash"
            : "claude-sonnet-4-20250514";
    }

    /* Select system prompt: explicit > prosody > default */
    const char *system_prompt = cfg.system_prompt;
    if (!system_prompt && cfg.prosody_prompt) {
        system_prompt = PROSODY_SYSTEM_PROMPT;
    }

    /* Global curl init */
    curl_global_init(CURL_GLOBAL_DEFAULT);

    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);

    const char *stt_label = (cfg.stt_engine == STT_ENGINE_BNNS)
        ? "BNNS/ANE Conformer"
        : (cfg.stt_engine == STT_ENGINE_CONFORMER)
        ? (cfg.cstt_model ? cfg.cstt_model : "conformer (no model)")
        : (cfg.stt_engine == STT_ENGINE_SONATA)
        ? "Sonata CTC (codec encoder)"
        : cfg.stt_repo;

    const char *llm_engine_label = (cfg.llm_engine == LLM_ENGINE_LOCAL)
        ? "Local" : (cfg.llm_engine == LLM_ENGINE_GEMINI)
        ? "Gemini" : "Claude";

    fprintf(stderr, "╔══════════════════════════════════════════════╗\n");
    fprintf(stderr, "║     pocket-voice — Native Voice Pipeline     ║\n");
    fprintf(stderr, "╠══════════════════════════════════════════════╣\n");
    fprintf(stderr, "║  STT: %-38s ║\n", stt_label);
    const char *tts_label = (cfg.tts_engine == TTS_ENGINE_SONATA)
        ? "Sonata (LM+Flow+iSTFT)"
        : (cfg.tts_engine == TTS_ENGINE_SONATA_V2)
        ? "Sonata Flow v2 (text→mel)"
        : (cfg.tts_engine == TTS_ENGINE_SONATA_V3)
        ? "Sonata Flow v3 (text/phoneme→mel→vocoder)"
        : "Sonata (default)";
    fprintf(stderr, "║  TTS: %-38s ║\n", tts_label);
    char llm_label[64];
    snprintf(llm_label, sizeof(llm_label), "%s/%s", llm_engine_label, llm_model);
    fprintf(stderr, "║  LLM: %-38s ║\n", llm_label);
    fprintf(stderr, "║  n_q: %-38d ║\n", cfg.n_q);
    fprintf(stderr, "║  VAD: %-38s ║\n", cfg.enable_vad ? "semantic" : "energy");
    if (cfg.prosody_prompt)
        fprintf(stderr, "║  SSML prosody prompt: enabled              ║\n");
    if (cfg.remote_mic)
        fprintf(stderr, "║  Remote mic: port %-26d ║\n", cfg.remote_port);
    fprintf(stderr, "╚══════════════════════════════════════════════╝\n\n");

    /* 1. Init audio engine (48kHz, 256-frame buffer = ~5.3ms latency) */
    fprintf(stderr, "[pocket-voice] Starting audio engine%s...\n",
            cfg.remote_mic ? " (output-only, mic via phone)" : "");
    VoiceEngine *audio = voice_engine_create(AUDIO_SAMPLE_RATE, AUDIO_BUFFER_FRAMES);
    if (!audio) {
        fprintf(stderr, "[pocket-voice] Failed to create audio engine\n");
        return 1;
    }
    int audio_rc = cfg.remote_mic
        ? voice_engine_start_output_only(audio)
        : voice_engine_start(audio);
    if (audio_rc != 0) {
        fprintf(stderr, "[pocket-voice] Failed to start audio engine\n");
        voice_engine_destroy(audio);
        return 1;
    }

    /* 2. Init STT */
    const char *stt_type_str = cfg.stt_engine == STT_ENGINE_BNNS ? "BNNS/ANE" :
                               cfg.stt_engine == STT_ENGINE_CONFORMER ? "Conformer/C" :
                               cfg.stt_engine == STT_ENGINE_SONATA ? "Sonata/CTC" :
                               "Kyutai/Rust";
    fprintf(stderr, "[pocket-voice] Loading STT model (%s)...\n", stt_type_str);
    SttInterface stt;
    if (cfg.stt_engine == STT_ENGINE_BNNS) {
        if (!cfg.cstt_model) {
            fprintf(stderr, "[pocket-voice] Error: --cstt-model required with --stt-engine bnns\n");
            voice_engine_destroy(audio);
            return 1;
        }
        stt = stt_create_bnns(cfg.cstt_model, cfg.bnns_model);
    } else if (cfg.stt_engine == STT_ENGINE_CONFORMER) {
        if (!cfg.cstt_model) {
            fprintf(stderr, "[pocket-voice] Error: --cstt-model required with --stt-engine conformer\n");
            voice_engine_destroy(audio);
            return 1;
        }
        stt = stt_create_conformer(cfg.cstt_model);
    } else if (cfg.stt_engine == STT_ENGINE_SONATA) {
        const char *model = cfg.sonata_stt_model
            ? cfg.sonata_stt_model : "models/sonata/sonata_stt.cstt_sonata";
        if (!cfg.sonata_stt_model) {
            fprintf(stderr, "[pocket-voice] Using default Sonata STT model: %s\n", model);
        }
        stt = stt_create_sonata(model, cfg.sonata_refiner_path);
    } else {
        stt = stt_create_rust(cfg.stt_repo, cfg.stt_model, cfg.enable_vad);
    }
    if (!stt.engine) {
        fprintf(stderr, "[pocket-voice] Failed to create STT engine\n");
        voice_engine_destroy(audio);
        return 1;
    }

    /* Enable beam search + LM for Conformer/BNNS engines */
    if (cfg.beam_size > 0 &&
        (cfg.stt_engine == STT_ENGINE_CONFORMER || cfg.stt_engine == STT_ENGINE_BNNS)) {
        ConformerSTT *cstt_ptr = (cfg.stt_engine == STT_ENGINE_BNNS)
            ? ((BNNSSttEngine *)stt.engine)->cstt
            : (ConformerSTT *)stt.engine;
        if (cstt_ptr) {
            int rc = conformer_stt_enable_beam_search(cstt_ptr, cfg.lm_path,
                                                       cfg.beam_size, cfg.lm_weight,
                                                       cfg.word_score);
            if (rc == 0) {
                fprintf(stderr, "[pocket-voice] Beam search enabled: beam=%d, lm_weight=%.2f%s\n",
                        cfg.beam_size, cfg.lm_weight,
                        cfg.lm_path ? ", with LM" : ", no LM");
            }
        }
    }

    /* 3. Init TTS */
    const char *tts_type_str = cfg.tts_engine == TTS_ENGINE_SONATA_V2 ? "Sonata Flow v2" :
                               cfg.tts_engine == TTS_ENGINE_SONATA_V3 ? "Sonata Flow v3" :
                               "Sonata (LM+Flow+iSTFT)";
    fprintf(stderr, "[pocket-voice] Loading TTS model (%s)...\n", tts_type_str);
    TtsInterface tts;
    if (cfg.tts_engine == TTS_ENGINE_SONATA) {
        const char *lm_w = cfg.sonata_lm_weights ? cfg.sonata_lm_weights : "models/sonata/sonata_lm.safetensors";
        const char *lm_c = cfg.sonata_lm_config ? cfg.sonata_lm_config : "models/sonata/sonata_lm_config.json";
        const char *tok  = cfg.sonata_tokenizer ? cfg.sonata_tokenizer : "models/tokenizer.model";
        const char *fl_w = cfg.sonata_flow_weights ? cfg.sonata_flow_weights : "models/sonata/sonata_flow.safetensors";
        const char *fl_c = cfg.sonata_flow_config ? cfg.sonata_flow_config : "models/sonata/sonata_flow_config.json";
        const char *dc_w = cfg.sonata_dec_weights ? cfg.sonata_dec_weights : "models/sonata/sonata_decoder.safetensors";
        const char *dc_c = cfg.sonata_dec_config ? cfg.sonata_dec_config : "models/sonata/sonata_decoder_config.json";
        tts = tts_create_sonata(lm_w, lm_c, tok, fl_w, fl_c, dc_w, dc_c);
        if (!tts.engine) {
            fprintf(stderr, "[sonata] Error: Failed to load Sonata. Check model paths.\n");
            stt.destroy(stt.engine);
            voice_engine_destroy(audio);
            return 1;
        }
        /* Apply Sonata tuning parameters */
        SonataEngine *se = (SonataEngine *)tts.engine;
        if (se->flow_engine) {
            if (cfg.sonata_speaker >= 0)
                sonata_flow_set_speaker(se->flow_engine, cfg.sonata_speaker);
            if (cfg.sonata_cfg_scale > 0.0f)
                sonata_flow_set_cfg_scale(se->flow_engine, cfg.sonata_cfg_scale);
            if (cfg.sonata_flow_steps > 0)
                sonata_flow_set_n_steps(se->flow_engine, cfg.sonata_flow_steps);
            if (cfg.sonata_heun)
                sonata_flow_set_solver(se->flow_engine, 1);
            sonata_flow_set_quality_mode(se->flow_engine, cfg.tts_quality_mode);
        }
        se->tts_quality_mode = cfg.tts_quality_mode;
        se->tts_first_chunk_fast = cfg.tts_first_chunk_fast;
        if (se->lm_engine) {
            sonata_lm_set_params(se->lm_engine, 0.8f, 50, 0.92f, 1.15f);
            const char *draft_w = cfg.sonata_draft_weights;
            const char *draft_c = cfg.sonata_draft_config;
            if (!draft_w && cfg.sonata_self_draft) {
                draft_w = cfg.sonata_lm_weights
                    ? cfg.sonata_lm_weights : "models/sonata/sonata_lm.safetensors";
                draft_c = NULL;
            }
            if (draft_w) {
                /* Prefer RNN drafter (ReDrafter tree) if weights exist */
                const char *rnn_drafter_w = "models/sonata/rnn_drafter.safetensors";
                const char *rnn_drafter_c = "models/sonata/rnn_drafter_config.json";
                if (access(rnn_drafter_w, R_OK) == 0 &&
                    sonata_lm_load_rnn_drafter(se->lm_engine, rnn_drafter_w,
                        access(rnn_drafter_c, R_OK) == 0 ? rnn_drafter_c : NULL) == 0) {
                    se->use_speculative = 1;
                    fprintf(stderr, "[sonata] ReDrafter tree speculative decoding enabled\n");
                } else if (sonata_lm_load_draft(se->lm_engine, draft_w, draft_c) == 0) {
                    se->use_speculative = 1;
                    fprintf(stderr, "[sonata] Speculative decoding enabled (k=%d)%s\n",
                            cfg.sonata_speculate_k,
                            cfg.sonata_self_draft ? " [self-draft: 4-layer]" : "");
                }
                if (cfg.sonata_speculate_k > 0)
                    sonata_lm_set_speculate_k(se->lm_engine, cfg.sonata_speculate_k);
            }
        }
        /* Phonemizer: wire espeak-ng G2P to Sonata for better pronunciation */
        if (cfg.use_phonemizer) {
            se->phonemizer = phonemizer_create("en-us");
            if (se->phonemizer) {
                if (cfg.phoneme_map_path) {
                    phonemizer_load_phoneme_map(se->phonemizer, cfg.phoneme_map_path);
                    fprintf(stderr, "[sonata] Phonemizer enabled (G2P: %s)\n", cfg.phoneme_map_path);
                } else {
                    fprintf(stderr, "[sonata] Phonemizer enabled (IPA mode, no phoneme map)\n");
                }
            }
        }
        /* Voice cloning: extract speaker embedding from reference WAV.
         * --clone-voice: one-stop flag that auto-detects encoder model.
         * Falls back to --speaker-encoder + --ref-wav for explicit control. */
        {
            const char *ref_wav = cfg.clone_voice_path
                ? cfg.clone_voice_path
                : (cfg.ref_wav_path ? cfg.ref_wav_path : cfg.sonata_ref_wav);
            const char *enc_path = cfg.speaker_encoder_path;
            /* Auto-detect encoder model for --clone-voice */
            if (ref_wav && !enc_path) {
                static const char *auto_paths[] = {
                    "models/ecapa_tdnn.onnx",
                    "models/speaker_encoder.onnx",
                    "models/sonata/speaker_encoder.onnx",
                    NULL
                };
                for (int ap = 0; auto_paths[ap]; ap++) {
                    FILE *test = fopen(auto_paths[ap], "rb");
                    if (test) { fclose(test); enc_path = auto_paths[ap]; break; }
                }
                if (!enc_path && cfg.clone_voice_path) {
                    fprintf(stderr, "[voice-clone] No speaker encoder found. "
                            "Download one with: huggingface-cli download "
                            "speechbrain/spkrec-ecapa-voxceleb --include *.onnx\n");
                }
            }
            if (enc_path && ref_wav && se->flow_engine) {
                SpeakerEncoder *spk_enc = speaker_encoder_create(cfg.speaker_encoder_path);
                if (spk_enc) {
                    int dim = speaker_encoder_embedding_dim(spk_enc);
                    float *emb = (float *)malloc((size_t)dim * sizeof(float));
                    if (emb) {
                        int result = speaker_encoder_extract_from_wav(spk_enc, ref_wav, emb);
                        if (result > 0) {
                            sonata_flow_set_speaker_embedding(se->flow_engine, emb, dim);
                            fprintf(stderr, "[voice-clone] Extracted %d-dim embedding from %s\n",
                                    dim, ref_wav);
                        }
                        free(emb);
                    }
                    speaker_encoder_destroy(spk_enc);
                }
            }
        }
        /* SoundStorm: load parallel decoder as LM replacement */
        if (cfg.sonata_storm_weights) {
            const char *storm_c = cfg.sonata_storm_config
                ? cfg.sonata_storm_config : "models/sonata/sonata_storm_config.json";
            se->storm_engine = sonata_storm_create(cfg.sonata_storm_weights, storm_c);
            if (se->storm_engine) {
                se->use_storm = 1;
                fprintf(stderr, "[sonata] SoundStorm parallel decoder loaded (replaces AR LM)\n");
            } else {
                fprintf(stderr, "[sonata] SoundStorm load failed, falling back to AR LM\n");
            }
        }
    } else if (cfg.tts_engine == TTS_ENGINE_SONATA_V2) {
        const char *fl_w = cfg.sonata_flow_v2_weights ? cfg.sonata_flow_v2_weights : "models/sonata/flow_v2.safetensors";
        const char *fl_c = cfg.sonata_flow_v2_config ? cfg.sonata_flow_v2_config : "models/sonata/flow_v2_config.json";
        tts = tts_create_sonata_v2(fl_w, fl_c);
        if (!tts.engine) {
            fprintf(stderr, "[sonata-v2] Error: Failed to load Flow v2. Check model paths.\n");
            stt.destroy(stt.engine);
            voice_engine_destroy(audio);
            return 1;
        }
        SonataV2Engine *sv2 = (SonataV2Engine *)tts.engine;
        if (cfg.sonata_speaker >= 0 && sv2->flow_v2)
            sonata_flow_v2_set_speaker(sv2->flow_v2, cfg.sonata_speaker);
        if (cfg.sonata_cfg_scale > 0.0f && sv2->flow_v2)
            sonata_flow_v2_set_cfg_scale(sv2->flow_v2, cfg.sonata_cfg_scale);
        if (cfg.sonata_flow_steps > 0 && sv2->flow_v2)
            sonata_flow_v2_set_n_steps(sv2->flow_v2, cfg.sonata_flow_steps);
        if (sv2->flow_v2)
            sonata_flow_v2_set_quality_mode(sv2->flow_v2, cfg.tts_quality_mode);
    } else if (cfg.tts_engine == TTS_ENGINE_SONATA_V3) {
        const char *fl_w = cfg.flow_v3_weights ? cfg.flow_v3_weights : "models/sonata/flow_v3.safetensors";
        const char *fl_c = cfg.flow_v3_config ? cfg.flow_v3_config : "models/sonata/flow_v3_config.json";
        const char *voc_w = cfg.vocoder_weights ? cfg.vocoder_weights : "models/sonata/vocoder.safetensors";
        const char *voc_c = cfg.vocoder_config ? cfg.vocoder_config : "models/sonata/vocoder_config.json";

        /* Auto-enable phonemization when model expects phonemes (char_vocab_size <= 67) */
        if (!cfg.use_phonemizer && fl_c) {
            FILE *fc = fopen(fl_c, "r");
            if (fc) {
                fseek(fc, 0, SEEK_END);
                long fclen = ftell(fc);
                fseek(fc, 0, SEEK_SET);
                if (fclen > 0 && fclen < 65536) {
                    char *fcdata = malloc((size_t)fclen + 1);
                    if (fcdata && fread(fcdata, 1, (size_t)fclen, fc) == (size_t)fclen) {
                        fcdata[fclen] = '\0';
                        cJSON *froot = cJSON_Parse(fcdata);
                        if (froot) {
                            cJSON *cv = cJSON_GetObjectItemCaseSensitive(froot, "char_vocab_size");
                            if (cv && cJSON_IsNumber(cv) && cv->valueint > 0 && cv->valueint <= 67) {
                                cfg.use_phonemizer = 1;
                                fprintf(stderr, "[config] Sonata v3 model expects phonemes (char_vocab_size=%d), auto-enabled phonemization\n", cv->valueint);
                            }
                            cJSON_Delete(froot);
                        }
                        free(fcdata);
                    }
                }
                fclose(fc);
            }
        }

        Phonemizer *ph = NULL;
        if (cfg.use_phonemizer) {
            ph = phonemizer_create("en-us");
            if (ph && cfg.phoneme_map_path)
                phonemizer_load_phoneme_map(ph, cfg.phoneme_map_path);
        }
        tts = tts_create_sonata_v3(fl_w, fl_c, voc_w, voc_c, ph);
        if (!tts.engine) {
            if (ph) phonemizer_destroy(ph);
            fprintf(stderr, "[sonata-v3] Error: Failed to load Flow v3 or Vocoder. Check model paths.\n");
            stt.destroy(stt.engine);
            voice_engine_destroy(audio);
            return 1;
        }
        SonataV3Engine *sv3 = (SonataV3Engine *)tts.engine;
        if (cfg.sonata_speaker >= 0 && sv3->flow_v3)
            sonata_flow_v3_set_speaker(sv3->flow_v3, cfg.sonata_speaker);
        if (cfg.sonata_cfg_scale > 0.0f && sv3->flow_v3)
            sonata_flow_v3_set_cfg_scale(sv3->flow_v3, cfg.sonata_cfg_scale);
        if (cfg.sonata_flow_steps > 0 && sv3->flow_v3)
            sonata_flow_v3_set_n_steps(sv3->flow_v3, cfg.sonata_flow_steps);
        if (cfg.sonata_heun && sv3->flow_v3)
            sonata_flow_v3_set_solver(sv3->flow_v3, 1);
        if (sv3->flow_v3)
            sonata_flow_v3_set_quality_mode(sv3->flow_v3, cfg.tts_quality_mode);
    } else {
        fprintf(stderr, "[pocket-voice] Unknown TTS engine, defaulting to Sonata\n");
        cfg.tts_engine = TTS_ENGINE_SONATA;
        const char *lm_w = cfg.sonata_lm_weights ? cfg.sonata_lm_weights : "models/sonata/sonata_lm.safetensors";
        const char *lm_c = cfg.sonata_lm_config ? cfg.sonata_lm_config : "models/sonata/sonata_lm_config.json";
        const char *tok  = cfg.sonata_tokenizer ? cfg.sonata_tokenizer : "models/tokenizer.model";
        const char *fl_w = cfg.sonata_flow_weights ? cfg.sonata_flow_weights : "models/sonata/sonata_flow.safetensors";
        const char *fl_c = cfg.sonata_flow_config ? cfg.sonata_flow_config : "models/sonata/sonata_flow_config.json";
        const char *dc_w = cfg.sonata_dec_weights ? cfg.sonata_dec_weights : "models/sonata/sonata_decoder.safetensors";
        const char *dc_c = cfg.sonata_dec_config ? cfg.sonata_dec_config : "models/sonata/sonata_decoder_config.json";
        tts = tts_create_sonata(lm_w, lm_c, tok, fl_w, fl_c, dc_w, dc_c);
    }
    if (!tts.engine) {
        fprintf(stderr, "[pocket-voice] Failed to create TTS engine\n");
        stt.destroy(stt.engine);
        voice_engine_destroy(audio);
        return 1;
    }

    /* 4. Init LLM client */
    ClaudeClient claude_storage;
    GeminiClient gemini_storage;
    LocalLLMClient local_llm_storage;
    LLMClient llm;

    if (cfg.llm_engine == LLM_ENGINE_LOCAL) {
        const char *local_model = llm_model ? llm_model : "meta-llama/Llama-3.2-3B-Instruct";
        fprintf(stderr, "[pocket-voice] LLM backend: Local (%s)\n", local_model);
        local_llm_init(&local_llm_storage, local_model, system_prompt);
        llm = llm_create_local(&local_llm_storage);
    } else if (cfg.llm_engine == LLM_ENGINE_GEMINI) {
        fprintf(stderr, "[pocket-voice] LLM backend: Gemini (%s)\n", llm_model);
        gemini_init(&gemini_storage, api_key, llm_model, system_prompt);
        llm = llm_create_gemini(&gemini_storage);
    } else {
        fprintf(stderr, "[pocket-voice] LLM backend: Claude (%s)\n", llm_model);
        claude_init(&claude_storage, api_key, llm_model, system_prompt);
        llm = llm_create_claude(&claude_storage);
    }

    /* 4b. Init conversation memory */
    ConversationMemory *conv_memory = NULL;
    if (cfg.memory_path) {
        conv_memory = memory_create(cfg.memory_path, cfg.memory_max_turns, cfg.memory_max_tokens);
        if (conv_memory) {
            fprintf(stderr, "[pocket-voice] Conversation memory: %s (%d turns loaded)\n",
                    cfg.memory_path, memory_turn_count(conv_memory));
        }
    }

    /* 4c. Init speaker diarizer */
    SpeakerDiarizer *diarizer = NULL;
    if (cfg.diarizer_encoder) {
        diarizer = diarizer_create(cfg.diarizer_encoder, cfg.diarizer_threshold,
                                    cfg.diarizer_max_speakers);
        if (diarizer) {
            fprintf(stderr, "[pocket-voice] Speaker diarizer: threshold=%.2f, max=%d speakers\n",
                    (double)cfg.diarizer_threshold, cfg.diarizer_max_speakers);
        }
    }

    /* 4d. Init audio post-processor (HW resampler, prosody, spatial) */
    AudioPostProcessor *pp = postproc_create(&cfg);
    if (!pp) {
        fprintf(stderr, "[pocket-voice] Warning: post-processor init failed, using defaults\n");
    } else {
        if (pp->use_hw_resample)
            fprintf(stderr, "[pocket-voice] AudioConverter resampling: enabled\n");
        if (fabsf(cfg.pitch - 1.0f) > 0.01f)
            fprintf(stderr, "[pocket-voice] Pitch: %.2fx\n", (double)cfg.pitch);
        if (fabsf(cfg.volume_db) > 0.1f)
            fprintf(stderr, "[pocket-voice] Volume: %+.1f dB\n", (double)cfg.volume_db);
        if (pp->use_spatial)
            fprintf(stderr, "[pocket-voice] Spatial audio: %.0f° azimuth\n", (double)cfg.spatial_az);
    }

    /* 4c. Load custom Metal kernels (if available) */
    MetalKernels *metal_kernels = NULL;
    if (cfg.metallib_path) {
        metal_kernels = metal_kernels_load(cfg.metallib_path);
        if (metal_kernels_available(metal_kernels)) {
            const char *kernel_names[16];
            int nk = metal_kernels_list(metal_kernels, kernel_names, 16);
            fprintf(stderr, "[pocket-voice] Metal kernels loaded: %d function(s)\n", nk);
            for (int k = 0; k < nk; k++)
                fprintf(stderr, "  - %s\n", kernel_names[k]);
        } else {
            fprintf(stderr, "[pocket-voice] Warning: failed to load metallib from %s\n",
                    cfg.metallib_path);
        }
    }

    /* 5. Init sentence buffer with adaptive warmup */
    SentenceBuffer *sentbuf = sentbuf_create(cfg.sentbuf_mode, cfg.sentbuf_min_words);
    if (!sentbuf) {
        fprintf(stderr, "[pocket-voice] Failed to create sentence buffer\n");
    } else {
        /* Adaptive warmup: first 2 sentences flush aggressively (3 words min)
           for fastest first-chunk latency, then revert to normal threshold */
        sentbuf_set_adaptive(sentbuf, 2, 3);

        /* Eager sub-sentence flush for Sonata: start TTS generation after just
           4 words instead of waiting for a sentence boundary. The Sonata LM
           supports streaming text append, so more words get fed as they arrive.
           Inspired by Liquid AI's interleaved generation approach. */
        if (cfg.tts_engine == TTS_ENGINE_SONATA) {
            sentbuf_set_eager(sentbuf, 4);
            fprintf(stderr, "[pocket-voice] Sentence buffer: eager flush at 4 words (Sonata streaming)\n");
        } else {
            sentbuf_set_eager(sentbuf, 6);
            fprintf(stderr, "[pocket-voice] Sentence buffer: eager flush at 6 words\n");
        }

        fprintf(stderr, "[pocket-voice] Sentence buffer: %s mode, min_words=%d (adaptive warmup)\n",
                cfg.sentbuf_mode == SENTBUF_MODE_SPECULATIVE ? "speculative" : "sentence",
                cfg.sentbuf_min_words);
    }

    /* 5b. Init latency profiler */
    LatencyProfile lp;
    lp_init(&lp);
    if (cfg.enable_profiler)
        fprintf(stderr, "[pocket-voice] Latency profiler: enabled\n");

    /* 6. Pipeline state */
    PipelineState state = STATE_LISTENING;
    SttAccum stt_accum;
    stt_accum_reset(&stt_accum);
    char transcript[TEXT_BUF_SIZE];
    int transcript_len = 0;
    transcript[0] = '\0';
    char llm_response[TEXT_BUF_SIZE];
    int llm_response_len = 0;
    llm_response[0] = '\0';
    TurnMetrics metrics;
    memset(&metrics, 0, sizeof(metrics));

    /* Per-turn arena allocator: all temporary allocations within a turn
       come from this arena, freed in one shot at turn end. */
    Arena turn_arena = arena_create(256 * 1024); /* 256 KiB initial */

    /* 6b. Audio warmup: write a tiny silent buffer to prime CoreAudio's
     * real-time thread and avoid the ~50ms allocation stall on first play.
     * The VoiceProcessingIO unit needs its first render callback triggered
     * before it enters steady-state. 480 samples = 10ms at 48kHz. */
    {
        float warmup_silence[480];
        memset(warmup_silence, 0, sizeof(warmup_silence));
        voice_engine_write_playback(audio, warmup_silence, 480);
        fprintf(stderr, "[pocket-voice] Audio pipeline warmed up\n");
    }

    /* 6c. Elevate pipeline thread to User Interactive QoS + RT hints */
    ap_set_qos_user_interactive();
    if (ap_set_realtime_priority(10000000, 5000000, 10000000) == 0) {
        fprintf(stderr, "[pocket-voice] Pipeline thread: real-time priority set\n");
    } else {
        fprintf(stderr, "[pocket-voice] Pipeline thread: using QoS User Interactive\n");
    }

    /* 6d. Start remote mic server if requested */
    WebRemote *web_remote = NULL;
    if (cfg.remote_mic) {
        web_remote = web_remote_create(cfg.remote_port, 16000,
            remote_mic_audio_cb, NULL, audio);
        if (pp) pp->web_remote = web_remote;
    }

    fprintf(stderr, "\n[pocket-voice] Ready. Speak to begin.\n");
    if (web_remote)
        fprintf(stderr, "  Phone remote mic active on port %d\n", web_remote_port(web_remote));
    print_state(state);

    /* 6c. HTTP API server mode */
    HttpApi *http_server = NULL;

    HttpProcessCtx http_ctx = { .tts = &tts, .pp = pp };

    if (cfg.server_mode) {
        HttpApiEngines http_eng = {
            .stt_engine = stt.engine,
            .tts_engine = tts.engine,
            .llm_engine = llm.engine,
            .stt_feed = stt.process_frame,
            .stt_flush = stt.flush,
            .stt_get_text = stt.get_text,
            .stt_get_words = stt.get_words,
            .stt_reset = stt.reset,
            .tts_speak = tts.set_text,
            .tts_step = tts.step,
            .tts_is_done = tts.is_done,
            .tts_set_text_done = tts.set_text_done,
            .tts_get_audio = tts.get_audio,
            .tts_reset = tts.reset,
            .tts_get_words = (cfg.tts_engine == TTS_ENGINE_SONATA_V3) ? sonatav3_get_words : NULL,
            .llm_send = llm.send,
            .llm_poll = llm.poll,
            .llm_peek = llm.peek_tokens,
            .llm_consume = llm.consume_tokens,
            .llm_is_done = llm.is_response_done,
            .process_text = http_process_text_impl,
            .process_ctx = &http_ctx,
        };
        http_server = http_api_create(cfg.server_port, http_eng);
        if (http_server && http_api_start(http_server) == 0) {
            fprintf(stderr, "[pocket-voice] Running in server mode on port %d\n",
                    cfg.server_port);
            fprintf(stderr, "[pocket-voice] Press Ctrl+C to stop\n");
            while (!g_quit) usleep(100000);
            http_api_destroy(http_server);
            goto cleanup;
        }
        fprintf(stderr, "[pocket-voice] Server start failed, falling back to interactive\n");
        http_api_destroy(http_server);
        http_server = NULL;
    }

    /* 7. Main loop */
    PipelineState prev_state;
    while (!g_quit) {
        prev_state = state;
        state = pipeline_tick(state, audio, &stt, &tts, &llm, &stt_accum,
                              sentbuf, transcript, &transcript_len,
                              cfg.vad_threshold, &metrics, pp,
                              &turn_arena, conv_memory, diarizer,
                              llm_response, &llm_response_len);

        /* Latency profiler: mark timestamps on state transitions */
        if (cfg.enable_profiler && state != prev_state) {
            switch (state) {
            case STATE_RECORDING:
                lp_mark_stt_start(&lp);
                break;
            case STATE_PROCESSING:
                lp_mark_vad_end(&lp);
                lp_mark_stt_end(&lp);
                lp_mark_llm_start(&lp);
                break;
            case STATE_STREAMING:
                if (prev_state == STATE_RECORDING) {
                    lp_mark_vad_end(&lp);
                    lp_mark_stt_end(&lp);
                    lp_mark_llm_start(&lp);
                }
                if (prev_state == STATE_PROCESSING) {
                    lp_mark_tts_start(&lp);
                }
                break;
            case STATE_SPEAKING:
                lp_mark_llm_end(&lp);
                break;
            case STATE_LISTENING:
                if (prev_state == STATE_SPEAKING) {
                    lp_mark_speaker_start(&lp);
                    lp_compute(&lp);
                    lp_print_turn(&lp);
                }
                break;
            }

            if (prev_state == STATE_STREAMING && metrics.has_first_tok &&
                lp.llm_first_token == 0) {
                lp_mark_llm_first_token(&lp);
            }
            if (prev_state == STATE_STREAMING && metrics.has_first_audio &&
                lp.tts_first_audio == 0) {
                lp_mark_tts_first_audio(&lp);
            }
        }

        /* Reset per-turn arena on state transition back to LISTENING */
        if (state == STATE_LISTENING && prev_state != STATE_LISTENING) {
            arena_reset(&turn_arena);
        }

        /* Adaptive sleep: shorter during active processing, longer when idle */
        if (state == STATE_LISTENING) {
            usleep(2000);   /* 2ms when idle — minimizes VAD onset latency */
        } else if (state == STATE_STREAMING || state == STATE_SPEAKING) {
            usleep(250);    /* 0.25ms during active generation */
        } else {
            usleep(2000);   /* 2ms during recording/processing */
        }
    }

cleanup:
    fprintf(stderr, "\n[pocket-voice] Shutting down...\n");

    if (cfg.enable_profiler && lp.n_turns > 0) {
        fprintf(stderr, "\n");
        lp_print_summary(&lp);
    }

    /* Cleanup in reverse order */
    arena_destroy(&turn_arena);
    llm.cancel(llm.engine);
    llm.cleanup(llm.engine);
    sentbuf_destroy(sentbuf);
    postproc_destroy(pp);
    tts.destroy(tts.engine);
    stt.destroy(stt.engine);
    diarizer_destroy(diarizer);
    memory_destroy(conv_memory);
    web_remote_destroy(web_remote);
    metal_kernels_destroy(metal_kernels);
    voice_engine_destroy(audio);
    curl_global_cleanup();

    fprintf(stderr, "[pocket-voice] Done.\n");
    return 0;
}
