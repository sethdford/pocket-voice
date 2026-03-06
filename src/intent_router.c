/**
 * intent_router.c — Learned classifier for routing user utterances.
 *
 * Routes to: ROUTE_FAST (template), ROUTE_MEDIUM (local LLM), ROUTE_FULL (cloud),
 * or ROUTE_BACKCHANNEL. Uses small MLP when weights loaded, else heuristics.
 *
 * MLP: input(20) → hidden(128, ReLU) → hidden(64, ReLU) → output(4, softmax)
 * All inference via cblas_sgemv. Zero allocations in hot path.
 */

#include "intent_router.h"
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <ctype.h>
#include <math.h>

#ifndef ACCELERATE_NEW_LAPACK
#define ACCELERATE_NEW_LAPACK
#endif
#include <Accelerate/Accelerate.h>

#define IR_N_FEATURES  20
#define IR_H1          128
#define IR_H2          64
#define IR_N_ROUTES    4

/* VAP prediction layout (matches vap_model.h VAPPrediction) */
typedef struct {
    float p_user_speaking;
    float p_system_turn;
    float p_backchannel;
    float p_eou;
} VAPPredLayout;

struct IntentRouter {
    int use_neural;           /* 1 = MLP loaded, 0 = heuristics */

    /* MLP weights (row-major) */
    float *W1;                /* [IR_H1][IR_N_FEATURES] */
    float *b1;                /* [IR_H1] */
    float *W2;                /* [IR_H2][IR_H1] */
    float *b2;                /* [IR_H2] */
    float *W3;                /* [IR_N_ROUTES][IR_H2] */
    float *b3;                /* [IR_N_ROUTES] */

    /* Pre-allocated activation buffers (no malloc in route()) */
    float h1[IR_H1];
    float h2[IR_H2];
    float logits[IR_N_ROUTES];

    /* Context for heuristic improvement */
    char context_buf[4096];
};

/* Fast response texts */
static const char *FAST_RESPONSES[FAST_COUNT] = {
    [FAST_GREETING]    = "Hey there!",
    [FAST_ACKNOWLEDGE] = "Got it.",
    [FAST_THINKING]    = "Let me think about that.",
    [FAST_YES]         = "Absolutely.",
    [FAST_NO]          = "I don't think so.",
    [FAST_THANKS]      = "You're welcome!",
    [FAST_GOODBYE]     = "See you later!",
};

/* Helper: count words in string */
static int count_words(const char *s) {
    int n = 0;
    int in_word = 0;
    while (*s) {
        if (isspace((unsigned char)*s)) {
            in_word = 0;
        } else if (!in_word) {
            in_word = 1;
            n++;
        }
        s++;
    }
    return n;
}

/* Helper: average word length (chars per word) */
static float avg_word_len(const char *s, int n_words) {
    if (!s || n_words <= 0) return 0.0f;
    int chars = 0;
    while (*s) { if (!isspace((unsigned char)*s)) chars++; s++; }
    return (float)chars / (float)n_words;
}

/* Helper: first word lowercase for comparison */
static void first_word_lower(char *buf, size_t cap, const char *s) {
    if (!buf || cap == 0 || !s) { buf[0] = '\0'; return; }
    size_t i = 0;
    while (*s && isspace((unsigned char)*s)) s++;
    while (*s && !isspace((unsigned char)*s) && i < cap - 1)
        buf[i++] = (char)tolower((unsigned char)*s++);
    buf[i] = '\0';
}

/* Helper: last word */
static void last_word_lower(char *buf, size_t cap, const char *s) {
    if (!buf || cap == 0 || !s) { buf[0] = '\0'; return; }
    const char *p = s + strlen(s);
    while (p > s && isspace((unsigned char)p[-1])) p--;
    const char *end = p;
    while (p > s && !isspace((unsigned char)p[-1])) p--;
    size_t i = 0;
    while (p < end && i < cap - 1)
        buf[i++] = (char)tolower((unsigned char)*p++);
    buf[i] = '\0';
}

/* Extract features into out[IR_N_FEATURES] */
static void extract_features(const char *transcript, int n_words,
                             const float *audio_features, const void *vap_pred,
                             float *out) {
    memset(out, 0, IR_N_FEATURES * sizeof(float));

    if (!transcript) return;

    /* Text features */
    out[0] = (float)n_words / 30.0f;  /* normalized word count */
    if (out[0] > 1.0f) out[0] = 1.0f;

    out[1] = avg_word_len(transcript, n_words) / 10.0f;
    if (out[1] > 1.0f) out[1] = 1.0f;

    out[2] = strchr(transcript, '?') ? 1.0f : 0.0f;

    char fw[32], lw[32];
    first_word_lower(fw, sizeof(fw), transcript);
    last_word_lower(lw, sizeof(lw), transcript);

    out[3] = (strcmp(fw, "hi") == 0 || strcmp(fw, "hey") == 0 || strcmp(fw, "hello") == 0) ? 1.0f : 0.0f;
    out[4] = (strstr(fw, "thank") != NULL) ? 1.0f : 0.0f;
    out[5] = (strcmp(fw, "bye") == 0 || strcmp(fw, "goodbye") == 0 || strstr(fw, "see") != NULL) ? 1.0f : 0.0f;
    out[6] = (strcmp(lw, "what") == 0 || strcmp(lw, "where") == 0 || strcmp(lw, "when") == 0 ||
              strcmp(lw, "how") == 0 || strcmp(lw, "why") == 0 || strcmp(lw, "who") == 0) ? 1.0f : 0.0f;

    /* Audio features (optional) */
    if (audio_features) {
        out[7] = audio_features[0];   /* energy */
        out[8] = audio_features[1];  /* pitch */
        if (out[7] > 1.0f) out[7] = 1.0f;
        if (out[8] > 1.0f) out[8] = 1.0f;
    }

    /* VAP features (optional) */
    if (vap_pred) {
        const VAPPredLayout *v = (const VAPPredLayout *)vap_pred;
        out[9]  = v->p_backchannel;
        out[10] = v->p_system_turn;
        out[11] = v->p_eou;
        out[12] = v->p_user_speaking;
    }

    /* Padding for remaining features (up to 20) */
    for (int i = 13; i < IR_N_FEATURES; i++)
        out[i] = 0.0f;
}

/* Heuristic routing (no neural weights) */
static RoutingDecision route_heuristic(IntentRouter *r, const char *transcript, int n_words) {
    RoutingDecision dec = { ROUTE_FULL, 0.7f, -1 };

    if (!transcript || n_words < 0) return dec;

    char fw[32];
    first_word_lower(fw, sizeof(fw), transcript);

    /* 1 word: backchannel unless greeting, thanks, or bye */
    if (n_words == 1) {
        if (strcmp(fw, "hi") == 0 || strcmp(fw, "hey") == 0 || strcmp(fw, "hello") == 0 ||
            strcmp(fw, "yo") == 0) {
            dec.route = ROUTE_FAST;
            dec.fast_type = FAST_GREETING;
            dec.confidence = 0.95f;
        } else if (strstr(fw, "thank") != NULL) {
            dec.route = ROUTE_FAST;
            dec.fast_type = FAST_THANKS;
            dec.confidence = 0.95f;
        } else if (strcmp(fw, "bye") == 0 || strcmp(fw, "goodbye") == 0) {
            dec.route = ROUTE_FAST;
            dec.fast_type = FAST_GOODBYE;
            dec.confidence = 0.95f;
        } else {
            dec.route = ROUTE_BACKCHANNEL;
            dec.confidence = 0.85f;
        }
        return dec;
    }

    /* Greeting */
    if (strcmp(fw, "hi") == 0 || strcmp(fw, "hey") == 0 || strcmp(fw, "hello") == 0 ||
        strstr(transcript, "hello") != NULL) {
        dec.route = ROUTE_FAST;
        dec.fast_type = FAST_GREETING;
        dec.confidence = 0.9f;
        return dec;
    }

    /* Thanks */
    if (strstr(fw, "thank") != NULL || strstr(transcript, "thanks") != NULL) {
        dec.route = ROUTE_FAST;
        dec.fast_type = FAST_THANKS;
        dec.confidence = 0.9f;
        return dec;
    }

    /* Bye */
    if (strcmp(fw, "bye") == 0 || strcmp(fw, "goodbye") == 0 ||
        strstr(transcript, "see ya") != NULL || strstr(transcript, "see you") != NULL) {
        dec.route = ROUTE_FAST;
        dec.fast_type = FAST_GOODBYE;
        dec.confidence = 0.9f;
        return dec;
    }

    /* Short + question: medium (local LLM) */
    if (n_words <= 4 && strchr(transcript, '?')) {
        dec.route = ROUTE_MEDIUM;
        dec.confidence = 0.75f;
        return dec;
    }

    /* Default: full cloud */
    dec.route = ROUTE_FULL;
    dec.confidence = 0.7f;
    return dec;
}

/* ReLU */
static inline float relu(float x) { return x > 0.0f ? x : 0.0f; }

/* Softmax in-place */
static void softmax4(float *x) {
    float maxv = x[0];
    for (int i = 1; i < 4; i++) if (x[i] > maxv) maxv = x[i];
    float sum = 0.0f;
    for (int i = 0; i < 4; i++) {
        x[i] = expf(x[i] - maxv);
        sum += x[i];
    }
    for (int i = 0; i < 4; i++) x[i] /= sum;
}

/* Neural forward pass */
static RoutingDecision route_neural(IntentRouter *r, const float *feat) {
    /* h1 = ReLU(W1 @ feat + b1): y = b1, then y := 1*W1*feat + 1*y */
    memcpy(r->h1, r->b1, IR_H1 * sizeof(float));
    cblas_sgemv(CblasRowMajor, CblasNoTrans, IR_H1, IR_N_FEATURES, 1.0f,
                r->W1, IR_N_FEATURES, feat, 1, 1.0f, r->h1, 1);
    for (int i = 0; i < IR_H1; i++) r->h1[i] = relu(r->h1[i]);

    /* h2 = ReLU(W2 @ h1 + b2) */
    memcpy(r->h2, r->b2, IR_H2 * sizeof(float));
    cblas_sgemv(CblasRowMajor, CblasNoTrans, IR_H2, IR_H1, 1.0f,
                r->W2, IR_H1, r->h1, 1, 1.0f, r->h2, 1);
    for (int i = 0; i < IR_H2; i++) r->h2[i] = relu(r->h2[i]);

    /* logits = W3 @ h2 + b3 */
    memcpy(r->logits, r->b3, IR_N_ROUTES * sizeof(float));
    cblas_sgemv(CblasRowMajor, CblasNoTrans, IR_N_ROUTES, IR_H2, 1.0f,
                r->W3, IR_H2, r->h2, 1, 1.0f, r->logits, 1);
    softmax4(r->logits);

    /* Argmax */
    int best = 0;
    float bestp = r->logits[0];
    for (int i = 1; i < IR_N_ROUTES; i++) {
        if (r->logits[i] > bestp) { bestp = r->logits[i]; best = i; }
    }

    RoutingDecision dec;
    dec.route = (ResponseRoute)best;
    dec.confidence = bestp;
    dec.fast_type = (best == ROUTE_FAST) ? FAST_GREETING : -1;  /* Simplified; real model would output fast_type */
    if (dec.route == ROUTE_FAST && dec.fast_type < 0) dec.fast_type = FAST_ACKNOWLEDGE;
    return dec;
}

/* .router file format: magic(4) + n_feat(4) + h1(4) + h2(4) + floats... */
#define ROUTER_MAGIC 0x52544E52  /* 'RTNR' */

IntentRouter *intent_router_create(const char *weights_path) {
    IntentRouter *r = (IntentRouter *)calloc(1, sizeof(IntentRouter));
    if (!r) return NULL;

    FILE *f = fopen(weights_path, "rb");
    if (!f) {
        free(r);
        return NULL;
    }

    uint32_t magic, nf, h1, h2;
    if (fread(&magic, 4, 1, f) != 1 || magic != ROUTER_MAGIC ||
        fread(&nf, 4, 1, f) != 1 || nf != IR_N_FEATURES ||
        fread(&h1, 4, 1, f) != 1 || h1 != IR_H1 ||
        fread(&h2, 4, 1, f) != 1 || h2 != IR_H2) {
        fclose(f);
        free(r);
        return NULL;
    }

    size_t nW1 = (size_t)IR_H1 * IR_N_FEATURES;
    size_t nb1 = IR_H1;
    size_t nW2 = (size_t)IR_H2 * IR_H1;
    size_t nb2 = IR_H2;
    size_t nW3 = (size_t)IR_N_ROUTES * IR_H2;
    size_t nb3 = IR_N_ROUTES;

    r->W1 = (float *)malloc((nW1 + nb1 + nW2 + nb2 + nW3 + nb3) * sizeof(float));
    if (!r->W1) { fclose(f); free(r); return NULL; }
    r->b1 = r->W1 + nW1;
    r->W2 = r->b1 + nb1;
    r->b2 = r->W2 + nW2;
    r->W3 = r->b2 + nb2;
    r->b3 = r->W3 + nW3;

    size_t total = (nW1 + nb1 + nW2 + nb2 + nW3 + nb3) * sizeof(float);
    if (fread(r->W1, 1, total, f) != total) {
        fclose(f);
        intent_router_destroy(r);
        return NULL;
    }
    fclose(f);

    r->use_neural = 1;
    return r;
}

IntentRouter *intent_router_create_default(void) {
    IntentRouter *r = (IntentRouter *)calloc(1, sizeof(IntentRouter));
    if (r) r->use_neural = 0;
    return r;
}

void intent_router_destroy(IntentRouter *router) {
    if (!router) return;
    if (router->use_neural && router->W1) free(router->W1);
    free(router);
}

RoutingDecision intent_router_route(IntentRouter *router, const char *transcript,
                                     int n_words, const float *audio_features,
                                     const void *vap_pred) {
    RoutingDecision dec = { ROUTE_FULL, 0.5f, -1 };
    if (!router) return dec;

    if (n_words < 0 && transcript)
        n_words = count_words(transcript);

    if (router->use_neural && router->W1) {
        float feat[IR_N_FEATURES];
        extract_features(transcript, n_words, audio_features, vap_pred, feat);
        return route_neural(router, feat);
    }
    return route_heuristic(router, transcript, n_words);
}

void intent_router_set_context(IntentRouter *router, const char *history) {
    if (!router || !history) return;
    size_t n = strlen(history);
    if (n >= sizeof(router->context_buf) - 1) n = sizeof(router->context_buf) - 2;
    memcpy(router->context_buf, history, n + 1);
}

const char *intent_router_fast_text(FastResponseType type) {
    if (type < 0 || type >= FAST_COUNT) return "";
    return FAST_RESPONSES[type];
}
