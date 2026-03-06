#ifndef SPECULATIVE_GEN_H
#define SPECULATIVE_GEN_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct SpeculativeGen SpeculativeGen;

typedef enum {
    SPEC_IDLE = 0,       /* No speculation active */
    SPEC_PENDING,        /* Draft being generated */
    SPEC_READY,          /* Draft ready for commitment */
    SPEC_COMMITTED,      /* Draft committed, feeding to TTS */
    SPEC_CANCELLED       /* Draft cancelled (user continued) */
} SpecState;

typedef struct {
    int   max_drafts;         /* Max concurrent drafts (default: 2) */
    int   min_words_to_spec;  /* Min STT words before speculating (default: 3) */
    float vap_eou_threshold;  /* VAP p_eou to start speculating (default: 0.4) */
    float vap_turn_threshold; /* VAP p_system_turn to boost priority (default: 0.3) */
    float commit_threshold;   /* Fused EOU to commit best draft (default: 0.8) */
    float cancel_threshold;   /* EOU drop to cancel drafts (default: 0.2) */
    int   max_spec_tokens;    /* Max LLM tokens to speculatively generate (default: 30) */
} SpeculativeConfig;

typedef struct {
    char  transcript[4096];   /* Transcript at time of speculation */
    int   n_words;
    char  response[4096];     /* LLM response tokens so far */
    int   n_tokens;
    float eou_at_start;       /* EOU probability when draft started */
    SpecState state;
    int   draft_id;
} SpecDraft;

/* Create speculative generation manager. */
SpeculativeGen *speculative_gen_create(const SpeculativeConfig *cfg);
void speculative_gen_destroy(SpeculativeGen *sg);

/* Update with current state. Called every pipeline tick.
 * transcript: current STT output
 * n_words: word count
 * vap_eou: VAP p_eou (or fused_eou if no VAP, 0 if unavailable)
 * vap_turn: VAP p_system_turn (0 if unavailable)
 * fused_eou: fused EOU probability
 *
 * Returns: action to take (0 = nothing, 1 = start new draft, 2 = commit best draft) */
int speculative_gen_tick(SpeculativeGen *sg,
                         const char *transcript, int n_words,
                         float vap_eou, float vap_turn,
                         float fused_eou);

/* Feed LLM response tokens for the active draft.
 * Called when the LLM produces a token for a speculative request.
 * draft_id: which draft this token belongs to.
 * token: the text token. */
void speculative_gen_feed_token(SpeculativeGen *sg, int draft_id, const char *token);

/* Mark a draft as complete (LLM finished generating). */
void speculative_gen_draft_done(SpeculativeGen *sg, int draft_id);

/* Get the best ready draft for commitment.
 * Returns pointer to internal SpecDraft, or NULL if none ready.
 * The transcript in the returned draft should be compared to the current
 * transcript to verify it's still valid. */
const SpecDraft *speculative_gen_get_best(const SpeculativeGen *sg);

/* Commit the best draft — marks it as COMMITTED, cancels others.
 * Returns the committed draft's response text, or NULL if nothing to commit.
 * The caller should feed this text directly to TTS. */
const char *speculative_gen_commit(SpeculativeGen *sg);

/* Cancel all active drafts (user kept speaking). */
void speculative_gen_cancel_all(SpeculativeGen *sg);

/* Reset for new conversation. */
void speculative_gen_reset(SpeculativeGen *sg);

/* Get the ID of the active draft that should receive LLM tokens.
 * Returns -1 if no active draft. */
int speculative_gen_active_draft(const SpeculativeGen *sg);

/* Check if a draft's transcript is still a prefix of the current transcript.
 * Used to decide whether to keep or discard a draft when new STT arrives. */
int speculative_gen_draft_valid(const SpeculativeGen *sg, int draft_id,
                                const char *current_transcript);

/* Get number of active (non-idle) drafts. */
int speculative_gen_active_count(const SpeculativeGen *sg);

/* Get draft info by ID. */
const SpecDraft *speculative_gen_get_draft(const SpeculativeGen *sg, int draft_id);

#ifdef __cplusplus
}
#endif

#endif /* SPECULATIVE_GEN_H */
