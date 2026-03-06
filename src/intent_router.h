#ifndef INTENT_ROUTER_H
#define INTENT_ROUTER_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct IntentRouter IntentRouter;

typedef enum {
    ROUTE_FAST = 0,      /* Pre-synth or template response (<100ms) */
    ROUTE_MEDIUM,        /* Local LLM (<300ms) */
    ROUTE_FULL,          /* Cloud LLM full reasoning (<500ms) */
    ROUTE_BACKCHANNEL,   /* Just emit a backchannel, don't respond */
} ResponseRoute;

typedef struct {
    ResponseRoute route;
    float         confidence;   /* [0,1] confidence in routing decision */
    int           fast_type;    /* If ROUTE_FAST: which pre-synth response (-1 if none) */
} RoutingDecision;

/* Pre-synthesized fast response types */
typedef enum {
    FAST_GREETING = 0,    /* "Hey!" / "Hi there!" */
    FAST_ACKNOWLEDGE,     /* "Got it" / "Sure thing" */
    FAST_THINKING,        /* "Let me think about that" */
    FAST_YES,             /* "Yes" / "Absolutely" */
    FAST_NO,              /* "No" / "I don't think so" */
    FAST_THANKS,          /* "You're welcome" */
    FAST_GOODBYE,         /* "Bye!" / "Take care" */
    FAST_COUNT
} FastResponseType;

/* Create intent router from weights file (.router format).
 * Architecture: small MLP on text features.
 * Returns NULL on failure. Falls back to heuristics if no weights. */
IntentRouter *intent_router_create(const char *weights_path);

/* Create with heuristic fallback (no neural weights). */
IntentRouter *intent_router_create_default(void);

void intent_router_destroy(IntentRouter *router);

/* Route a user utterance.
 * transcript: STT output text
 * n_words: word count (from STT)
 * audio_features: optional audio features [n_features] (NULL to skip)
 * vap_pred: optional VAP prediction (NULL to skip)
 * Returns routing decision. */
RoutingDecision intent_router_route(
    IntentRouter *router,
    const char *transcript,
    int n_words,
    const float *audio_features,
    const void *vap_pred
);

/* Set conversation history context (improves routing accuracy).
 * history: last N turns as concatenated text. */
void intent_router_set_context(IntentRouter *router, const char *history);

/* Get the text for a fast response type. */
const char *intent_router_fast_text(FastResponseType type);

#ifdef __cplusplus
}
#endif

#endif /* INTENT_ROUTER_H */
