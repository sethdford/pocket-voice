/**
 * semantic_eou.c — Lightweight text-based sentence completion predictor.
 *
 * Architecture: byte embedding → LayerNorm → 1-layer LSTM → Linear → sigmoid.
 * ~33K parameters, runs in <1ms on Apple Silicon (AMX-accelerated via cblas).
 *
 * Follows the project pattern established by native_vad.c:
 *   - All vector math via Apple Accelerate (cblas/vDSP)
 *   - Zero allocations after create()
 *   - Every allocation checked
 *   - Binary weight format with magic number validation
 */

#include "semantic_eou.h"
#include "lstm_ops.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

#ifdef __APPLE__
#ifndef ACCELERATE_NEW_LAPACK
#define ACCELERATE_NEW_LAPACK
#endif
#include <Accelerate/Accelerate.h>
#endif

/* ── Binary format ───────────────────────────────────────────────────────── */

#define SEOU_MAGIC   0x554F4553  /* "SEOU" little-endian */
#define SEOU_VERSION 1

typedef struct {
    uint32_t magic;
    uint32_t version;
    uint32_t vocab_size;
    uint32_t embed_dim;
    uint32_t hidden_dim;
    uint32_t max_seq_len;
} SeouHeader;

/* ── Engine struct ───────────────────────────────────────────────────────── */

struct SemanticEOU {
    int vocab_size;
    int embed_dim;
    int hidden_dim;
    int max_seq_len;

    /* Embedding table [vocab_size × embed_dim] */
    float *embedding;

    /* LayerNorm on embeddings */
    float *norm_w;     /* [embed_dim] */
    float *norm_b;     /* [embed_dim] */

    /* LSTM weights (shared lstm_ops.h format) */
    float *lstm_wi;    /* [4H × E] input-to-hidden */
    float *lstm_wh;    /* [4H × H] hidden-to-hidden */
    float *lstm_bias;  /* [4H] */

    /* Output projection */
    float *out_w;      /* [H] */
    float  out_b;

    /* LSTM persistent state */
    float *h;          /* [H] hidden state */
    float *c;          /* [H] cell state */

    /* Pre-allocated working memory (no allocs in hot path) */
    float *gates;      /* [4H] */
    float *normed;     /* [embed_dim] scratch for LayerNorm output */
};

/* ── Utility ─────────────────────────────────────────────────────────────── */

static void layer_norm_1d(float *out, const float *in,
                           const float *w, const float *b, int D) {
    float mean = 0.0f, var = 0.0f;
#ifdef __APPLE__
    vDSP_meanv(in, 1, &mean, D);
#else
    for (int i = 0; i < D; i++) mean += in[i];
    mean /= (float)D;
#endif
    for (int i = 0; i < D; i++) {
        float d = in[i] - mean;
        var += d * d;
    }
    var /= (float)D;
    float inv_std = 1.0f / sqrtf(var + 1e-5f);
    for (int i = 0; i < D; i++)
        out[i] = (in[i] - mean) * inv_std * w[i] + b[i];
}

/* ── Public API ──────────────────────────────────────────────────────────── */

SemanticEOU *semantic_eou_create(void) {
    SemanticEOU *se = (SemanticEOU *)calloc(1, sizeof(SemanticEOU));
    if (!se) return NULL;

    se->vocab_size  = SEOU_VOCAB_SIZE;
    se->embed_dim   = SEOU_EMBED_DIM;
    se->hidden_dim  = SEOU_HIDDEN_DIM;
    se->max_seq_len = SEOU_MAX_SEQ_LEN;

    int V = se->vocab_size;
    int E = se->embed_dim;
    int H = se->hidden_dim;
    int G = 4 * H;

    se->embedding = (float *)calloc((size_t)V * E, sizeof(float));
    se->norm_w    = (float *)calloc(E, sizeof(float));
    se->norm_b    = (float *)calloc(E, sizeof(float));
    se->lstm_wi   = (float *)calloc((size_t)G * E, sizeof(float));
    se->lstm_wh   = (float *)calloc((size_t)G * H, sizeof(float));
    se->lstm_bias = (float *)calloc(G, sizeof(float));
    se->out_w     = (float *)calloc(H, sizeof(float));
    se->h         = (float *)calloc(H, sizeof(float));
    se->c         = (float *)calloc(H, sizeof(float));
    se->gates     = (float *)calloc(G, sizeof(float));
    se->normed    = (float *)calloc(E, sizeof(float));

    if (!se->embedding || !se->norm_w || !se->norm_b ||
        !se->lstm_wi || !se->lstm_wh || !se->lstm_bias ||
        !se->out_w || !se->h || !se->c || !se->gates || !se->normed) {
        semantic_eou_destroy(se);
        return NULL;
    }

    /* Initialize LayerNorm to identity */
    for (int i = 0; i < E; i++) se->norm_w[i] = 1.0f;

    return se;
}

void semantic_eou_destroy(SemanticEOU *se) {
    if (!se) return;
    free(se->embedding);
    free(se->norm_w);
    free(se->norm_b);
    free(se->lstm_wi);
    free(se->lstm_wh);
    free(se->lstm_bias);
    free(se->out_w);
    free(se->h);
    free(se->c);
    free(se->gates);
    free(se->normed);
    free(se);
}

int semantic_eou_load_weights(SemanticEOU *se, const char *path) {
    if (!se || !path) return -1;

    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "[semantic_eou] Cannot open weights: %s\n", path);
        return -1;
    }

    SeouHeader hdr;
    if (fread(&hdr, sizeof(hdr), 1, f) != 1) {
        fprintf(stderr, "[semantic_eou] Failed to read header\n");
        fclose(f);
        return -1;
    }

    if (hdr.magic != SEOU_MAGIC) {
        fprintf(stderr, "[semantic_eou] Bad magic: 0x%08X (expected 0x%08X)\n",
                hdr.magic, SEOU_MAGIC);
        fclose(f);
        return -1;
    }
    if (hdr.version != SEOU_VERSION) {
        fprintf(stderr, "[semantic_eou] Unsupported version: %u\n", hdr.version);
        fclose(f);
        return -1;
    }
    if (hdr.vocab_size != (uint32_t)se->vocab_size ||
        hdr.embed_dim != (uint32_t)se->embed_dim ||
        hdr.hidden_dim != (uint32_t)se->hidden_dim) {
        fprintf(stderr, "[semantic_eou] Architecture mismatch: V=%u E=%u H=%u "
                "(expected V=%d E=%d H=%d)\n",
                hdr.vocab_size, hdr.embed_dim, hdr.hidden_dim,
                se->vocab_size, se->embed_dim, se->hidden_dim);
        fclose(f);
        return -1;
    }

    int V = se->vocab_size;
    int E = se->embed_dim;
    int H = se->hidden_dim;
    int G = 4 * H;

    size_t total = 0;
    size_t got;
    int read_ok = 1;

#define READ_W(dst, count) do { \
    got = fread((dst), sizeof(float), (count), f); \
    total += got; \
    if (got != (size_t)(count)) read_ok = 0; \
} while (0)

    READ_W(se->embedding, (size_t)V * E);
    READ_W(se->norm_w, E);
    READ_W(se->norm_b, E);
    READ_W(se->lstm_wi, (size_t)G * E);
    READ_W(se->lstm_wh, (size_t)G * H);
    READ_W(se->lstm_bias, G);
    READ_W(se->out_w, H);
    READ_W(&se->out_b, 1);

#undef READ_W

    fclose(f);

    if (!read_ok) {
        fprintf(stderr, "[semantic_eou] Truncated weight file\n");
        return -1;
    }

    size_t expected = (size_t)V * E + E + E +
                      (size_t)G * E + (size_t)G * H + G +
                      H + 1;
    if (total != expected) {
        fprintf(stderr, "[semantic_eou] Weight count mismatch: got %zu, expected %zu\n",
                total, expected);
        return -1;
    }

    fprintf(stderr, "[semantic_eou] Loaded %s: %zu params (%.1f KB)\n",
            path, expected, (float)(expected * sizeof(float)) / 1024.0f);
    return 0;
}

void semantic_eou_init_random(SemanticEOU *se, uint32_t seed) {
    if (!se) return;

    int V = se->vocab_size;
    int E = se->embed_dim;
    int H = se->hidden_dim;
    int G = 4 * H;

    srand(seed);

    /* Embedding: small random values */
    float emb_scale = 1.0f / sqrtf((float)E);
    for (int i = 0; i < V * E; i++)
        se->embedding[i] = emb_scale * (2.0f * (float)rand() / (float)RAND_MAX - 1.0f);

    /* LSTM: Xavier initialization */
    float scale_i = sqrtf(2.0f / (float)(E + H));
    float scale_h = sqrtf(2.0f / (float)(H + H));
    for (int i = 0; i < G * E; i++)
        se->lstm_wi[i] = scale_i * (2.0f * (float)rand() / (float)RAND_MAX - 1.0f);
    for (int i = 0; i < G * H; i++)
        se->lstm_wh[i] = scale_h * (2.0f * (float)rand() / (float)RAND_MAX - 1.0f);

    /* Forget gate bias = 1.0 */
    for (int i = 0; i < G; i++) se->lstm_bias[i] = 0.0f;
    for (int i = H; i < 2 * H; i++) se->lstm_bias[i] = 1.0f;

    /* Output: small random */
    float scale_o = sqrtf(2.0f / (float)(H + 1));
    for (int i = 0; i < H; i++)
        se->out_w[i] = scale_o * (2.0f * (float)rand() / (float)RAND_MAX - 1.0f);
    se->out_b = 0.0f;

    /* LayerNorm to identity */
    for (int i = 0; i < E; i++) {
        se->norm_w[i] = 1.0f;
        se->norm_b[i] = 0.0f;
    }
}

float semantic_eou_process(SemanticEOU *se, const char *text) {
    if (!se) return 0.5f;
    if (!text || !text[0]) return 0.5f;

    int E = se->embed_dim;
    int H = se->hidden_dim;

    /* Determine input: last max_seq_len bytes of text */
    size_t len = strlen(text);
    const unsigned char *start;
    int seq_len;
    if ((int)len <= se->max_seq_len) {
        start = (const unsigned char *)text;
        seq_len = (int)len;
    } else {
        start = (const unsigned char *)(text + len - se->max_seq_len);
        seq_len = se->max_seq_len;
    }

    /* Reset LSTM state for each inference (stateless per call) */
    memset(se->h, 0, (size_t)H * sizeof(float));
    memset(se->c, 0, (size_t)H * sizeof(float));

    /* Process each byte through embedding → LayerNorm → LSTM */
    for (int t = 0; t < seq_len; t++) {
        int token = (int)start[t];  /* 0-255 */

        /* Lookup embedding */
        const float *emb = se->embedding + token * E;

        /* LayerNorm */
        layer_norm_1d(se->normed, emb, se->norm_w, se->norm_b, E);

        /* LSTM step */
        lstm_step(se->lstm_wi, se->lstm_wh, se->lstm_bias,
                  se->normed, 1,  /* contiguous input */
                  se->h, se->c, se->gates,
                  E, H);
    }

    /* Output: ReLU(h) → dot(out_w, h) + out_b → sigmoid */
    float logit;
#ifdef __APPLE__
    /* ReLU in-place on a temp copy (don't modify h — might be reused) */
    float relu_h[SEOU_HIDDEN_DIM];
    for (int i = 0; i < H; i++)
        relu_h[i] = se->h[i] > 0.0f ? se->h[i] : 0.0f;
    logit = cblas_sdot(H, se->out_w, 1, relu_h, 1);
#else
    logit = 0.0f;
    for (int i = 0; i < H; i++) {
        float rv = se->h[i] > 0.0f ? se->h[i] : 0.0f;
        logit += se->out_w[i] * rv;
    }
#endif
    logit += se->out_b;

    return lstm_sigmoid(logit);
}

void semantic_eou_reset(SemanticEOU *se) {
    if (!se) return;
    memset(se->h, 0, (size_t)se->hidden_dim * sizeof(float));
    memset(se->c, 0, (size_t)se->hidden_dim * sizeof(float));
}

int semantic_eou_word_count(const char *text) {
    if (!text) return 0;
    int count = 0;
    int in_word = 0;
    for (const char *p = text; *p; p++) {
        if (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r') {
            in_word = 0;
        } else {
            if (!in_word) count++;
            in_word = 1;
        }
    }
    return count;
}
