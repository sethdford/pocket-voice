/**
 * speaker_encoder.c — ONNX-based speaker embedding extraction for voice cloning.
 *
 * Uses ONNX Runtime to run ECAPA-TDNN or similar speaker verification models,
 * extracting fixed-size embeddings from reference audio for sonata_flow voice cloning.
 *
 * Build: cc -O3 -shared -fPIC -arch arm64
 *        -I/opt/homebrew/include/onnxruntime -Isrc
 *        -L/opt/homebrew/lib -lonnxruntime
 *        -framework Accelerate
 *        -install_name @rpath/libspeaker_encoder.dylib -o libspeaker_encoder.dylib speaker_encoder.c
 */

#include "speaker_encoder.h"

#ifdef __APPLE__
#ifndef ACCELERATE_NEW_LAPACK
#define ACCELERATE_NEW_LAPACK
#endif
#include <Accelerate/Accelerate.h>
#endif

#include <onnxruntime/onnxruntime_c_api.h>

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define SE_DUMMY_FBANK_FRAMES 10   /* 10 frames of zeros for embedding_dim probe */
#define SE_MAX_EMBEDDING 1024
#define SE_MAX_WAV_SAMPLES (48000 * 60)  /* 60 seconds max */

/* Fbank params (WeSpeaker ECAPA-TDNN) */
#define SE_FFT_SIZE 512
#define SE_WIN_LEN  400   /* 25ms at 16kHz */
#define SE_HOP      160   /* 10ms */
#define SE_N_MELS   80
#define SE_FMIN     20.0f
#define SE_FMAX     7600.0f
#define SE_FBANK_SAMPLE_RATE 16000
#define SE_LOG_FLOOR 1e-10f

struct SpeakerEncoder {
    const OrtApi       *api;
    OrtEnv             *env;
    OrtSessionOptions  *session_opts;
    OrtSession         *session;
    OrtMemoryInfo      *mem_info;
    OrtAllocator       *allocator;
    char               *input_name;
    char               *output_name;
    int                 embedding_dim;
    int                 model_sample_rate;

    /* Fbank extraction resources */
    vDSP_DFT_Setup      fft_setup;
    float               *hann_window;
    float               *mel_fb;      /* [80 × (n_fft/2+1)] row-major */
    int                 n_bins;       /* n_fft/2 + 1 */
    float               *windowed;    /* [n_fft] */
    float               *power_spec;  /* [n_bins] */
    DSPSplitComplex     fft_split;    /* for DFT */
    float               *fft_real;
    float               *fft_imag;
};

/* ── Mel filterbank construction ──────────────────────────────────────────── */

static void build_mel_filterbank(float *fb, int n_fft, int n_mels, int sample_rate,
                                 float fmin, float fmax) {
    int n_bins = n_fft / 2 + 1;
    float mel_min = 2595.0f * log10f(1.0f + fmin / 700.0f);
    float mel_max = 2595.0f * log10f(1.0f + fmax / 700.0f);

    float mel_points[SE_N_MELS + 2];
    for (int i = 0; i <= n_mels + 1; i++)
        mel_points[i] = mel_min + (mel_max - mel_min) * i / (n_mels + 1);

    float hz_points[SE_N_MELS + 2];
    for (int i = 0; i <= n_mels + 1; i++)
        hz_points[i] = 700.0f * (powf(10.0f, mel_points[i] / 2595.0f) - 1.0f);

    float freq_step = (float)sample_rate / n_fft;
    memset(fb, 0, (size_t)n_mels * n_bins * sizeof(float));

    for (int m = 0; m < n_mels; m++) {
        for (int k = 0; k < n_bins; k++) {
            float freq = k * freq_step;
            if (freq >= hz_points[m] && freq <= hz_points[m + 1]) {
                fb[m * n_bins + k] = (freq - hz_points[m]) / (hz_points[m + 1] - hz_points[m]);
            } else if (freq > hz_points[m + 1] && freq <= hz_points[m + 2]) {
                fb[m * n_bins + k] = (hz_points[m + 2] - freq) / (hz_points[m + 2] - hz_points[m + 1]);
            }
        }
    }
}

/* ── Fbank extraction ─────────────────────────────────────────────────────── */

static float *compute_fbank(SpeakerEncoder *enc, const float *audio, int n_samples,
                            int *out_n_frames) {
    if (!enc || !audio || n_samples < SE_WIN_LEN || !out_n_frames) {
        *out_n_frames = 0;
        return NULL;
    }

    int n_frames = 1 + (n_samples - SE_WIN_LEN) / SE_HOP;
    if (n_frames <= 0) {
        *out_n_frames = 0;
        return NULL;
    }

    float *fbank = (float *)malloc((size_t)n_frames * SE_N_MELS * sizeof(float));
    if (!fbank) {
        *out_n_frames = 0;
        return NULL;
    }

    int n_bins = enc->n_bins;
    const float *win = enc->hann_window;
    float *windowed = enc->windowed;
    float *power_spec = enc->power_spec;
    float *mel_fb = enc->mel_fb;
    DSPSplitComplex *sp = &enc->fft_split;

    for (int f = 0; f < n_frames; f++) {
        int start = f * SE_HOP;

        /* 1. Apply Hann window */
        memset(windowed, 0, SE_FFT_SIZE * sizeof(float));
        vDSP_vmul(audio + start, 1, win, 1, windowed, 1, SE_WIN_LEN);

        /* 2. Pack for real DFT: realp[i]=windowed[i], imagp[i]=windowed[n_fft/2+i] */
        for (int i = 0; i < SE_FFT_SIZE / 2; i++) {
            sp->realp[i] = windowed[i];
            sp->imagp[i] = windowed[SE_FFT_SIZE / 2 + i];
        }

        /* 3. Forward real DFT */
        vDSP_DFT_Execute(enc->fft_setup, sp->realp, sp->imagp, sp->realp, sp->imagp);

        /* 4. Power spectrum |X[k]|^2; DC and Nyquist packed in realp[0], imagp[0] */
        float dc_power  = sp->realp[0] * sp->realp[0];
        float nyq_power = sp->imagp[0] * sp->imagp[0];

        DSPSplitComplex bins_offset = { sp->realp + 1, sp->imagp + 1 };
        vDSP_zvmags(&bins_offset, 1, power_spec + 1, 1, n_bins - 2);

        power_spec[0]          = dc_power;
        power_spec[n_bins - 1] = nyq_power;

        float fft_scale = 0.25f;
        vDSP_vsmul(power_spec, 1, &fft_scale, power_spec, 1, n_bins);

        /* 5. Mel filterbank multiply: mel_frame = mel_fb @ power_spec */
        float *mel_frame = fbank + f * SE_N_MELS;
        cblas_sgemv(CblasRowMajor, CblasNoTrans,
                    SE_N_MELS, n_bins,
                    1.0f, mel_fb, n_bins,
                    power_spec, 1,
                    0.0f, mel_frame, 1);

        /* 6. Log with floor */
        float eps = SE_LOG_FLOOR;
        vDSP_vsadd(mel_frame, 1, &eps, mel_frame, 1, SE_N_MELS);
        int n = SE_N_MELS;
        vvlogf(mel_frame, mel_frame, &n);
    }

    *out_n_frames = n_frames;
    return fbank;
}

/* ── WAV parsing ─────────────────────────────────────────────────────────── */

typedef struct {
    int sample_rate;
    int channels;
    int bits_per_sample;
    const uint8_t *data;
    int data_bytes;
} WavInfo;

static int parse_wav_file(const char *path, WavInfo *info) {
    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "[speaker_encoder] Cannot open WAV: %s\n", path);
        return -1;
    }

    uint8_t header[12];
    if (fread(header, 1, 12, f) != 12) {
        fclose(f);
        fprintf(stderr, "[speaker_encoder] WAV read error\n");
        return -1;
    }
    if (memcmp(header, "RIFF", 4) != 0 || memcmp(header + 8, "WAVE", 4) != 0) {
        fclose(f);
        fprintf(stderr, "[speaker_encoder] Not a valid WAV file\n");
        return -1;
    }

    info->sample_rate = 16000;
    info->channels = 1;
    info->bits_per_sample = 16;
    info->data = NULL;
    info->data_bytes = 0;

    uint8_t chunk_id[4];
    uint32_t chunk_size;
    int found_fmt = 0, found_data = 0;
    uint8_t *data_buf = NULL;

    while (fread(chunk_id, 1, 4, f) == 4 && fread(&chunk_size, 1, 4, f) == 4) {
        if (memcmp(chunk_id, "fmt ", 4) == 0) {
            uint8_t fmt[16];
            if (chunk_size < 16 || fread(fmt, 1, 16, f) != 16) {
                fclose(f);
                return -1;
            }
            info->channels = fmt[2] | (fmt[3] << 8);
            info->sample_rate = (int)(fmt[4] | (fmt[5] << 8) | (fmt[6] << 16) | (fmt[7] << 24));
            info->bits_per_sample = fmt[14] | (fmt[15] << 8);
            if (chunk_size > 16)
                fseek(f, (long)(chunk_size - 16), SEEK_CUR);
            found_fmt = 1;
        } else if (memcmp(chunk_id, "data", 4) == 0) {
            data_buf = (uint8_t *)malloc(chunk_size);
            if (!data_buf) {
                fclose(f);
                return -1;
            }
            if (fread(data_buf, 1, chunk_size, f) != chunk_size) {
                free(data_buf);
                fclose(f);
                return -1;
            }
            info->data = data_buf;
            info->data_bytes = (int)chunk_size;
            found_data = 1;
            break;
        } else {
            fseek(f, (long)(chunk_size + (chunk_size & 1)), SEEK_CUR);  /* word-align */
        }
    }

    fclose(f);

    if (!found_fmt || !found_data) {
        fprintf(stderr, "[speaker_encoder] WAV missing fmt or data chunk\n");
        if (data_buf) free(data_buf);
        return -1;
    }

    if (info->bits_per_sample != 16 && info->bits_per_sample != 32) {
        fprintf(stderr, "[speaker_encoder] WAV unsupported bits: %d\n", info->bits_per_sample);
        free(data_buf);
        return -1;
    }

    return 0;
}

/* Load WAV samples into float buffer, mono. Caller frees *out. Returns n_samples or -1. */
static int wav_to_float_mono(const char *path, float **out, int *out_sr) {
    WavInfo info;
    memset(&info, 0, sizeof(info));
    if (parse_wav_file(path, &info) != 0)
        return -1;

    int bytes_per_sample = info.bits_per_sample / 8;
    int n_frames = info.data_bytes / (bytes_per_sample * info.channels);
    if (n_frames <= 0 || n_frames > SE_MAX_WAV_SAMPLES) {
        if (info.data) free((void *)info.data);
        return -1;
    }

    float *pcm = (float *)malloc((size_t)n_frames * sizeof(float));
    if (!pcm) {
        free((void *)info.data);
        return -1;
    }

    if (info.bits_per_sample == 16) {
        const int16_t *src = (const int16_t *)info.data;
        for (int i = 0; i < n_frames; i++) {
            int idx = i * info.channels;
            pcm[i] = (float)src[idx] / 32768.0f;
        }
    } else {
        const float *src = (const float *)info.data;
        for (int i = 0; i < n_frames; i++) {
            int idx = i * info.channels;
            pcm[i] = src[idx];
        }
    }

    free((void *)info.data);
    *out = pcm;
    *out_sr = info.sample_rate;
    return n_frames;
}

/* Simple linear interpolation resample: src_sr -> dst_sr */
static void resample_linear(const float *src, int n_src, int src_sr,
                            float *dst, int *n_dst, int dst_sr) {
    if (src_sr == dst_sr) {
        *n_dst = n_src;
        memcpy(dst, src, (size_t)n_src * sizeof(float));
        return;
    }
    double ratio = (double)dst_sr / (double)src_sr;
    int out_len = (int)((double)n_src * ratio + 0.5);
    if (out_len <= 0) { *n_dst = 0; return; }
    for (int i = 0; i < out_len; i++) {
        double pos = (double)i / ratio;
        int idx = (int)pos;
        float frac = (float)(pos - (double)idx);
        if (idx >= n_src - 1) {
            dst[i] = src[n_src - 1];
        } else {
            dst[i] = src[idx] * (1.0f - frac) + src[idx + 1] * frac;
        }
    }
    *n_dst = out_len;
}

/* ── Public API ───────────────────────────────────────────────────────────── */

SpeakerEncoder *speaker_encoder_create(const char *model_path) {
    if (!model_path) {
        fprintf(stderr, "[speaker_encoder] model_path is NULL\n");
        return NULL;
    }

    SpeakerEncoder *enc = (SpeakerEncoder *)calloc(1, sizeof(SpeakerEncoder));
    if (!enc) return NULL;

    enc->api = OrtGetApiBase()->GetApi(ORT_API_VERSION);
    if (!enc->api) {
        fprintf(stderr, "[speaker_encoder] Failed to get ONNX Runtime API\n");
        free(enc);
        return NULL;
    }

    OrtStatus *status = NULL;

    status = enc->api->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "speaker_encoder", &enc->env);
    if (status) {
        fprintf(stderr, "[speaker_encoder] CreateEnv: %s\n", enc->api->GetErrorMessage(status));
        enc->api->ReleaseStatus(status);
        free(enc);
        return NULL;
    }

    status = enc->api->CreateSessionOptions(&enc->session_opts);
    if (status) goto fail;
    (void)enc->api->SetIntraOpNumThreads(enc->session_opts, 4);
    (void)enc->api->SetSessionGraphOptimizationLevel(enc->session_opts, ORT_ENABLE_ALL);

    /* Enable CoreML EP for ANE acceleration (falls back to CPU for unsupported ops) */
    {
        const char *keys[] = {"ModelFormat", "MLComputeUnits"};
        const char *vals[] = {"NeuralNetwork", "ALL"};
        OrtStatus *cml_st = enc->api->SessionOptionsAppendExecutionProvider(
            enc->session_opts, "CoreML", keys, vals, 2);
        if (cml_st) {
            enc->api->ReleaseStatus(cml_st);
            fprintf(stderr, "[speaker_encoder] CoreML EP not available, using CPU\n");
        } else {
            fprintf(stderr, "[speaker_encoder] CoreML EP enabled\n");
        }
    }

    status = enc->api->CreateSession(enc->env, model_path, enc->session_opts, &enc->session);
    if (status) {
        fprintf(stderr, "[speaker_encoder] CreateSession: %s\n", enc->api->GetErrorMessage(status));
        enc->api->ReleaseStatus(status);
        status = NULL;  /* avoid double-release in fail */
        goto fail;
    }

    status = enc->api->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &enc->mem_info);
    if (status) goto fail;

    status = enc->api->GetAllocatorWithDefaultOptions(&enc->allocator);
    if (status) goto fail;

    /* Get input/output names */
    char *input_name = NULL;
    char *output_name = NULL;
    status = enc->api->SessionGetInputName(enc->session, 0, enc->allocator, &input_name);
    if (status) {
        fprintf(stderr, "[speaker_encoder] SessionGetInputName: %s\n", enc->api->GetErrorMessage(status));
        enc->api->ReleaseStatus(status);
        goto fail;
    }
    status = enc->api->SessionGetOutputName(enc->session, 0, enc->allocator, &output_name);
    if (status) {
        fprintf(stderr, "[speaker_encoder] SessionGetOutputName: %s\n", enc->api->GetErrorMessage(status));
        enc->api->ReleaseStatus(status);
        (void)enc->api->AllocatorFree(enc->allocator, input_name);
        goto fail;
    }
    enc->input_name = strdup(input_name);
    enc->output_name = strdup(output_name);
    (void)enc->api->AllocatorFree(enc->allocator, input_name);
    (void)enc->api->AllocatorFree(enc->allocator, output_name);
    if (!enc->input_name || !enc->output_name) goto fail;

    /* Initialize fbank extraction resources */
    enc->n_bins = SE_FFT_SIZE / 2 + 1;

    enc->hann_window = (float *)malloc(SE_WIN_LEN * sizeof(float));
    if (!enc->hann_window) goto fail;
    for (int i = 0; i < SE_WIN_LEN; i++)
        enc->hann_window[i] = 0.5f * (1.0f - cosf(2.0f * (float)M_PI * i / (SE_WIN_LEN - 1)));

    enc->mel_fb = (float *)calloc((size_t)SE_N_MELS * enc->n_bins, sizeof(float));
    if (!enc->mel_fb) goto fail;
    build_mel_filterbank(enc->mel_fb, SE_FFT_SIZE, SE_N_MELS, SE_FBANK_SAMPLE_RATE, SE_FMIN, SE_FMAX);

    enc->fft_setup = vDSP_DFT_zrop_CreateSetup(NULL, SE_FFT_SIZE, vDSP_DFT_FORWARD);
    if (!enc->fft_setup) goto fail;

    enc->fft_real = (float *)calloc(SE_FFT_SIZE / 2, sizeof(float));
    enc->fft_imag = (float *)calloc(SE_FFT_SIZE / 2, sizeof(float));
    if (!enc->fft_real || !enc->fft_imag) goto fail;
    enc->fft_split.realp = enc->fft_real;
    enc->fft_split.imagp = enc->fft_imag;

    enc->windowed = (float *)calloc(SE_FFT_SIZE, sizeof(float));
    enc->power_spec = (float *)calloc(enc->n_bins, sizeof(float));
    if (!enc->windowed || !enc->power_spec) goto fail;

    /* Run dummy inference with fbank input [1, 10, 80] to detect embedding_dim */
    float *dummy_fbank = (float *)calloc(SE_DUMMY_FBANK_FRAMES * SE_N_MELS, sizeof(float));
    if (!dummy_fbank) goto fail;

    int64_t shape[] = {1, SE_DUMMY_FBANK_FRAMES, SE_N_MELS};
    OrtValue *input_tensor = NULL;
    status = enc->api->CreateTensorWithDataAsOrtValue(
        enc->mem_info, dummy_fbank, (size_t)(SE_DUMMY_FBANK_FRAMES * SE_N_MELS) * sizeof(float),
        shape, 3, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &input_tensor);
    free(dummy_fbank);
    if (status) goto fail;

    OrtValue *output_tensor = NULL;
    OrtValue *inputs[] = { input_tensor };
    status = enc->api->Run(enc->session, NULL,
                          (const char *const *)&enc->input_name, (const OrtValue *const *)inputs, 1,
                          (const char *const *)&enc->output_name, 1, &output_tensor);
    enc->api->ReleaseValue(input_tensor);
    if (status) {
        fprintf(stderr, "[speaker_encoder] Dummy Run (fbank): %s\n", enc->api->GetErrorMessage(status));
        enc->api->ReleaseStatus(status);
        goto fail;
    }

    OrtTensorTypeAndShapeInfo *shape_info = NULL;
    status = enc->api->GetTensorTypeAndShape(output_tensor, &shape_info);
    if (status) {
        enc->api->ReleaseValue(output_tensor);
        goto fail;
    }

    size_t elem_count = 0;
    OrtStatus *count_status = enc->api->GetTensorShapeElementCount(shape_info, &elem_count);
    enc->embedding_dim = count_status ? 0 : (int)elem_count;
    if (count_status) {
        fprintf(stderr, "[speaker_encoder] Failed to get embedding element count\n");
        enc->api->ReleaseStatus(count_status);
    }

    enc->api->ReleaseTensorTypeAndShapeInfo(shape_info);
    enc->api->ReleaseValue(output_tensor);

    if (enc->embedding_dim <= 0 || enc->embedding_dim > SE_MAX_EMBEDDING) {
        fprintf(stderr, "[speaker_encoder] Invalid embedding dim: %d\n", enc->embedding_dim);
        goto fail;
    }

    enc->model_sample_rate = SE_FBANK_SAMPLE_RATE;

    fprintf(stderr, "[speaker_encoder] Loaded %s (embedding_dim=%d)\n", model_path, enc->embedding_dim);
    return enc;

fail:
    if (status) {
        fprintf(stderr, "[speaker_encoder] ORT error: %s\n", enc->api->GetErrorMessage(status));
        enc->api->ReleaseStatus(status);
    }
    speaker_encoder_destroy(enc);
    return NULL;
}

void speaker_encoder_destroy(SpeakerEncoder *enc) {
    if (!enc) return;
    free(enc->input_name);
    free(enc->output_name);
    if (enc->fft_setup) vDSP_DFT_DestroySetup(enc->fft_setup);
    free(enc->hann_window);
    free(enc->mel_fb);
    free(enc->fft_real);
    free(enc->fft_imag);
    free(enc->windowed);
    free(enc->power_spec);
    if (enc->api) {
        if (enc->mem_info)      enc->api->ReleaseMemoryInfo(enc->mem_info);
        if (enc->session)       enc->api->ReleaseSession(enc->session);
        if (enc->session_opts)  enc->api->ReleaseSessionOptions(enc->session_opts);
        if (enc->env)           enc->api->ReleaseEnv(enc->env);
    }
    free(enc);
}

int speaker_encoder_embedding_dim(const SpeakerEncoder *enc) {
    return enc ? enc->embedding_dim : 0;
}

static void l2_normalize(float *x, int n) {
#ifdef __APPLE__
    float norm_sq;
    vDSP_dotpr(x, 1, x, 1, &norm_sq, n);
    float norm = sqrtf(norm_sq);
    if (norm > 1e-8f) {
        float inv_norm = 1.0f / norm;
        vDSP_vsmul(x, 1, &inv_norm, x, 1, n);
    }
#else
    float norm = 0.0f;
    for (int i = 0; i < n; i++)
        norm += x[i] * x[i];
    norm = sqrtf(norm);
    if (norm > 1e-8f)
        for (int i = 0; i < n; i++)
            x[i] /= norm;
#endif
}

int speaker_encoder_extract(SpeakerEncoder *enc, const float *audio, int n_samples,
                             float *embedding_out) {
    if (!enc || !audio || !embedding_out || n_samples <= 0) {
        if (enc) fprintf(stderr, "[speaker_encoder] extract: invalid args\n");
        return -1;
    }

    /* Convert raw audio to 80-bin log-mel fbank features */
    int n_frames = 0;
    float *fbank = compute_fbank(enc, audio, n_samples, &n_frames);
    if (!fbank || n_frames <= 0) {
        fprintf(stderr, "[speaker_encoder] extract: fbank failed (need >= %d samples)\n", SE_WIN_LEN);
        if (fbank) free(fbank);
        return -1;
    }

    int64_t shape[] = {1, n_frames, SE_N_MELS};
    OrtValue *input_tensor = NULL;
    OrtStatus *status = enc->api->CreateTensorWithDataAsOrtValue(
        enc->mem_info, fbank, (size_t)(n_frames * SE_N_MELS) * sizeof(float),
        shape, 3, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &input_tensor);
    if (status) {
        fprintf(stderr, "[speaker_encoder] CreateTensor: %s\n", enc->api->GetErrorMessage(status));
        enc->api->ReleaseStatus(status);
        free(fbank);
        return -1;
    }

    OrtValue *output_tensor = NULL;
    OrtValue *inputs[] = { input_tensor };
    status = enc->api->Run(enc->session, NULL,
                          (const char *const *)&enc->input_name, (const OrtValue *const *)inputs, 1,
                          (const char *const *)&enc->output_name, 1, &output_tensor);
    enc->api->ReleaseValue(input_tensor);
    free(fbank);
    if (status) {
        fprintf(stderr, "[speaker_encoder] Run: %s\n", enc->api->GetErrorMessage(status));
        enc->api->ReleaseStatus(status);
        return -1;
    }

    float *output_data = NULL;
    status = enc->api->GetTensorMutableData(output_tensor, (void **)&output_data);
    if (status) {
        enc->api->ReleaseValue(output_tensor);
        fprintf(stderr, "[speaker_encoder] GetTensorMutableData: %s\n", enc->api->GetErrorMessage(status));
        enc->api->ReleaseStatus(status);
        return -1;
    }

    memcpy(embedding_out, output_data, (size_t)enc->embedding_dim * sizeof(float));
    enc->api->ReleaseValue(output_tensor);

    l2_normalize(embedding_out, enc->embedding_dim);
    return enc->embedding_dim;
}

int speaker_encoder_extract_from_wav(SpeakerEncoder *enc, const char *wav_path,
                                      float *embedding_out) {
    if (!enc || !wav_path || !embedding_out) {
        fprintf(stderr, "[speaker_encoder] extract_from_wav: invalid args\n");
        return -1;
    }

    float *pcm = NULL;
    int wav_sr = 0;
    int n = wav_to_float_mono(wav_path, &pcm, &wav_sr);
    if (n < 0) return -1;

    int result = -1;
    if (wav_sr == enc->model_sample_rate) {
        result = speaker_encoder_extract(enc, pcm, n, embedding_out);
    } else {
        int resampled_len = (int)((double)n * (double)enc->model_sample_rate / (double)wav_sr + 0.5);
        if (resampled_len <= 0 || resampled_len > SE_MAX_WAV_SAMPLES) {
            fprintf(stderr, "[speaker_encoder] Resampled length out of range\n");
            free(pcm);
            return -1;
        }
        float *resampled = (float *)malloc((size_t)resampled_len * sizeof(float));
        if (!resampled) {
            free(pcm);
            return -1;
        }
        int actual_len;
        resample_linear(pcm, n, wav_sr, resampled, &actual_len, enc->model_sample_rate);
        result = speaker_encoder_extract(enc, resampled, actual_len, embedding_out);
        free(resampled);
    }
    free(pcm);
    return result;
}
