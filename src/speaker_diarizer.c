/**
 * speaker_diarizer.c — Speaker diarization using ECAPA-TDNN embeddings.
 *
 * Maintains speaker profiles with running centroid embeddings. Identifies
 * speakers via cosine similarity; registers new speakers when no match found.
 * Uses speaker_encoder for embedding extraction from audio.
 */

#include "speaker_diarizer.h"
#include "speaker_encoder.h"
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

#define MAX_EMB_DIM 1024
#define DEFAULT_EMB_DIM 192
#define DEFAULT_THRESHOLD 0.75f
#define DEFAULT_MAX_SPEAKERS 8

typedef struct {
    float centroid[MAX_EMB_DIM];
    float last_emb[MAX_EMB_DIM];
    int n_utterances;
    char label[64];
} SpeakerProfile;

struct SpeakerDiarizer {
    SpeakerEncoder *encoder;
    int emb_dim;
    float threshold;
    int max_speakers;
    int n_speakers;
    SpeakerProfile *profiles;
};

static float cosine_sim(const float *a, const float *b, int dim) {
    float dot = 0.0f;
    for (int i = 0; i < dim; i++)
        dot += a[i] * b[i];
    return dot;
}

static void l2_normalize(float *x, int n) {
    float norm_sq = 0.0f;
    for (int i = 0; i < n; i++)
        norm_sq += x[i] * x[i];
    float norm = sqrtf(norm_sq);
    if (norm > 1e-8f) {
        float inv = 1.0f / norm;
        for (int i = 0; i < n; i++)
            x[i] *= inv;
    }
}

static void update_centroid(SpeakerProfile *profile, const float *embedding, int dim) {
    profile->n_utterances++;
    float alpha = 1.0f / (float)(profile->n_utterances);
    for (int i = 0; i < dim; i++)
        profile->centroid[i] = (1.0f - alpha) * profile->centroid[i] + alpha * embedding[i];
    memcpy(profile->last_emb, embedding, (size_t)dim * sizeof(float));
    l2_normalize(profile->centroid, dim);
}

static int register_new_speaker(SpeakerDiarizer *d, const float *embedding, int dim) {
    if (d->n_speakers >= d->max_speakers)
        return -1;
    SpeakerProfile *p = &d->profiles[d->n_speakers];
    memcpy(p->centroid, embedding, (size_t)dim * sizeof(float));
    memcpy(p->last_emb, embedding, (size_t)dim * sizeof(float));
    p->n_utterances = 1;
    p->label[0] = '\0';
    return d->n_speakers++;
}

SpeakerDiarizer *diarizer_create(const char *encoder_path, float threshold, int max_speakers) {
    SpeakerDiarizer *d = (SpeakerDiarizer *)calloc(1, sizeof(SpeakerDiarizer));
    if (!d)
        return NULL;

    if (encoder_path && encoder_path[0] != '\0') {
        d->encoder = speaker_encoder_create(encoder_path);
        if (!d->encoder) {
            free(d);
            return NULL;
        }
        d->emb_dim = speaker_encoder_embedding_dim(d->encoder);
        if (d->emb_dim <= 0 || d->emb_dim > MAX_EMB_DIM) {
            fprintf(stderr, "[speaker_diarizer] embedding dim %d exceeds MAX_EMB_DIM %d\n",
                    d->emb_dim, MAX_EMB_DIM);
            speaker_encoder_destroy(d->encoder);
            free(d);
            return NULL;
        }
    } else {
        d->encoder = NULL;
        d->emb_dim = DEFAULT_EMB_DIM;
    }

    d->threshold = (threshold > 0.0f) ? threshold : DEFAULT_THRESHOLD;
    d->max_speakers = (max_speakers > 0 && max_speakers <= 64) ? max_speakers : DEFAULT_MAX_SPEAKERS;
    d->n_speakers = 0;
    d->profiles = (SpeakerProfile *)calloc((size_t)d->max_speakers, sizeof(SpeakerProfile));
    if (!d->profiles) {
        if (d->encoder)
            speaker_encoder_destroy(d->encoder);
        free(d);
        return NULL;
    }
    return d;
}

void diarizer_destroy(SpeakerDiarizer *d) {
    if (!d)
        return;
    if (d->encoder)
        speaker_encoder_destroy(d->encoder);
    free(d->profiles);
    free(d);
}

int diarizer_identify(SpeakerDiarizer *d, const float *audio, int n_samples) {
    if (!d || !audio || n_samples <= 0)
        return -1;
    if (!d->encoder) {
        fprintf(stderr, "[speaker_diarizer] identify requires encoder (create with model path)\n");
        return -1;
    }
    float *emb = (float *)malloc((size_t)d->emb_dim * sizeof(float));
    if (!emb)
        return -1;
    int dim = speaker_encoder_extract(d->encoder, audio, n_samples, emb);
    if (dim <= 0) {
        free(emb);
        return -1;
    }
    int result = diarizer_identify_embedding(d, emb, dim);
    free(emb);
    return result;
}

int diarizer_identify_embedding(SpeakerDiarizer *d, const float *embedding, int dim) {
    if (!d || !embedding)
        return -1;
    if (dim != d->emb_dim) {
        fprintf(stderr, "[speaker_diarizer] embedding dim must be %d, got %d\n", d->emb_dim, dim);
        return -1;
    }

    int best_id = -1;
    float best_sim = -2.0f;

    for (int i = 0; i < d->n_speakers; i++) {
        float sim = cosine_sim(d->profiles[i].centroid, embedding, dim);
        if (sim > best_sim) {
            best_sim = sim;
            best_id = i;
        }
    }

    if (best_sim >= d->threshold) {
        update_centroid(&d->profiles[best_id], embedding, dim);
        return best_id;
    }

    if (d->n_speakers < d->max_speakers) {
        return register_new_speaker(d, embedding, dim);
    }

    if (best_id >= 0) {
        update_centroid(&d->profiles[best_id], embedding, dim);
        return best_id;
    }
    return -1;
}

int diarizer_speaker_count(const SpeakerDiarizer *d) {
    return d ? d->n_speakers : 0;
}

int diarizer_get_embedding(const SpeakerDiarizer *d, int speaker_id, float *out) {
    if (!d || !out)
        return -1;
    if (speaker_id < 0 || speaker_id >= d->n_speakers)
        return -1;
    memcpy(out, d->profiles[speaker_id].centroid, (size_t)d->emb_dim * sizeof(float));
    return d->emb_dim;
}

int diarizer_set_label(SpeakerDiarizer *d, int speaker_id, const char *label) {
    if (!d)
        return -1;
    if (speaker_id < 0 || speaker_id >= d->n_speakers)
        return -1;
    if (!label)
        label = "";
    strncpy(d->profiles[speaker_id].label, label, sizeof(d->profiles[0].label) - 1);
    d->profiles[speaker_id].label[sizeof(d->profiles[0].label) - 1] = '\0';
    return 0;
}

const char *diarizer_get_label(const SpeakerDiarizer *d, int speaker_id) {
    if (!d)
        return NULL;
    if (speaker_id < 0 || speaker_id >= d->n_speakers)
        return NULL;
    return d->profiles[speaker_id].label[0] ? d->profiles[speaker_id].label : NULL;
}

void diarizer_reset(SpeakerDiarizer *d) {
    if (!d)
        return;
    memset(d->profiles, 0, (size_t)d->max_speakers * sizeof(SpeakerProfile));
    d->n_speakers = 0;
}
