/**
 * speculative_gen.c — Continuous speculative LLM generation.
 *
 * Maintains rolling drafts that start before user finishes speaking.
 * VAP-informed triggering, draft validity (prefix match), best-draft selection.
 * Thread-safe: tick from pipeline thread, feed_token from LLM thread.
 */

#include "speculative_gen.h"
#include <stdlib.h>
#include <string.h>
#include <pthread.h>

#ifndef SPEC_MAX_DRAFTS
#define SPEC_MAX_DRAFTS 8
#endif

static const char *SPEC_PREFIXES[] __attribute__((unused)) = {
    "Sure, ",
    "Great question! ",
    "Well, ",
    "I think ",
    "That's ",
    "Absolutely, ",
    "Hmm, ",
    "Interesting, ",
};
#define NUM_SPEC_PREFIXES (sizeof(SPEC_PREFIXES) / sizeof(SPEC_PREFIXES[0]))

static void default_config(SpeculativeConfig *cfg) {
    cfg->max_drafts = 2;
    cfg->min_words_to_spec = 3;
    cfg->vap_eou_threshold = 0.4f;
    cfg->vap_turn_threshold = 0.3f;
    cfg->commit_threshold = 0.8f;
    cfg->cancel_threshold = 0.2f;
    cfg->max_spec_tokens = 30;
}

struct SpeculativeGen {
    SpeculativeConfig cfg;
    SpecDraft drafts[SPEC_MAX_DRAFTS];
    int next_draft_id;
    char current_transcript[4096];
    int current_n_words;
    pthread_mutex_t mutex;
};

static int count_words(const char *s) {
    int n = 0;
    int in_word = 0;
    for (const char *p = s; *p; p++) {
        int space = (*p == ' ' || *p == '\t' || *p == '\n');
        if (!space && !in_word) { in_word = 1; n++; }
        else if (space) in_word = 0;
    }
    return n;
}

static int is_prefix(const char *prefix, const char *full) {
    while (*prefix) {
        if (*prefix != *full) return 0;
        prefix++;
        full++;
    }
    return 1; /* prefix exhausted; full may have more */
}

static void clear_draft(SpecDraft *d) {
    d->transcript[0] = '\0';
    d->n_words = 0;
    d->response[0] = '\0';
    d->n_tokens = 0;
    d->eou_at_start = 0.f;
    d->state = SPEC_IDLE;
}

static int has_pending_draft(const SpeculativeGen *sg) {
    for (int i = 0; i < sg->cfg.max_drafts; i++) {
        if (sg->drafts[i].state == SPEC_PENDING) return 1;
    }
    return 0;
}

static int all_pending(const SpeculativeGen *sg) {
    int has_pending = 0;
    for (int i = 0; i < sg->cfg.max_drafts; i++) {
        SpecState s = sg->drafts[i].state;
        if (s == SPEC_PENDING) has_pending = 1;
        if (s == SPEC_READY) return 0; /* not "all pending" */
    }
    return has_pending;
}

static SpecDraft *find_free_slot(SpeculativeGen *sg) {
    for (int i = 0; i < sg->cfg.max_drafts; i++) {
        if (sg->drafts[i].state == SPEC_IDLE || sg->drafts[i].state == SPEC_CANCELLED)
            return &sg->drafts[i];
    }
    return NULL;
}

static SpecDraft *find_best_valid(const SpeculativeGen *sg, const char *current) {
    SpecDraft *best = NULL;
    int best_tokens = -1;
    float best_eou = -1.f;
    for (int i = 0; i < sg->cfg.max_drafts; i++) {
        const SpecDraft *d = &sg->drafts[i];
        if (d->state != SPEC_PENDING && d->state != SPEC_READY) continue;
        if (!is_prefix(d->transcript, current)) continue;
        if (d->n_tokens > best_tokens ||
            (d->n_tokens == best_tokens && d->eou_at_start > best_eou)) {
            best = (SpecDraft *)d;
            best_tokens = d->n_tokens;
            best_eou = d->eou_at_start;
        }
    }
    return best;
}

static void cancel_all_internal(SpeculativeGen *sg) {
    for (int i = 0; i < sg->cfg.max_drafts; i++) {
        if (sg->drafts[i].state == SPEC_PENDING || sg->drafts[i].state == SPEC_READY)
            sg->drafts[i].state = SPEC_CANCELLED;
    }
}

SpeculativeGen *speculative_gen_create(const SpeculativeConfig *cfg) {
    SpeculativeGen *sg = (SpeculativeGen *)malloc(sizeof(SpeculativeGen));
    if (!sg) return NULL;
    if (cfg) {
        memcpy(&sg->cfg, cfg, sizeof(SpeculativeConfig));
    } else {
        default_config(&sg->cfg);
    }
    if (sg->cfg.max_drafts > SPEC_MAX_DRAFTS) sg->cfg.max_drafts = SPEC_MAX_DRAFTS;
    if (sg->cfg.max_drafts < 1) sg->cfg.max_drafts = 1;

    sg->next_draft_id = 0;
    sg->current_transcript[0] = '\0';
    sg->current_n_words = 0;
    pthread_mutex_init(&sg->mutex, NULL);
    for (int i = 0; i < SPEC_MAX_DRAFTS; i++) clear_draft(&sg->drafts[i]);
    return sg;
}

void speculative_gen_destroy(SpeculativeGen *sg) {
    if (!sg) return;
    pthread_mutex_destroy(&sg->mutex);
    free(sg);
}

int speculative_gen_tick(SpeculativeGen *sg,
                         const char *transcript, int n_words,
                         float vap_eou, float vap_turn,
                         float fused_eou) {
    pthread_mutex_lock(&sg->mutex);

    size_t tlen = strlen(transcript);
    if (tlen >= sizeof(sg->current_transcript)) tlen = sizeof(sg->current_transcript) - 1;
    memcpy(sg->current_transcript, transcript, tlen + 1);
    sg->current_n_words = n_words >= 0 ? n_words : count_words(transcript);

    int action = 0;

    /* 1. Commit if fused_eou high enough */
    if (fused_eou >= sg->cfg.commit_threshold) {
        SpecDraft *best = find_best_valid(sg, sg->current_transcript);
        if (best) {
            pthread_mutex_unlock(&sg->mutex);
            return 2; /* caller will call get_best + commit */
        }
    }

    /* 2. Cancel if EOU dropped and all drafts are PENDING */
    if (fused_eou < sg->cfg.cancel_threshold && all_pending(sg)) {
        cancel_all_internal(sg);
        action = 0;
        pthread_mutex_unlock(&sg->mutex);
        return 0;
    }

    /* 3. Start new draft if VAP/words/not-already-pending */
    if (vap_eou >= sg->cfg.vap_eou_threshold &&
        sg->current_n_words >= sg->cfg.min_words_to_spec &&
        !has_pending_draft(sg)) {
        SpecDraft *slot = find_free_slot(sg);
        if (slot) {
            size_t cplen = tlen;
            if (cplen >= sizeof(slot->transcript) - 1) cplen = sizeof(slot->transcript) - 1;
            memcpy(slot->transcript, sg->current_transcript, cplen + 1);
            slot->n_words = sg->current_n_words;
            slot->response[0] = '\0';
            slot->n_tokens = 0;
            slot->eou_at_start = fused_eou;
            slot->state = SPEC_PENDING;
            slot->draft_id = sg->next_draft_id++;
            action = 1;
        }
    } else {
        /* 4. Check existing drafts: if transcript diverged, cancel and optionally restart */
        for (int i = 0; i < sg->cfg.max_drafts; i++) {
            SpecDraft *d = &sg->drafts[i];
            if (d->state != SPEC_PENDING && d->state != SPEC_READY) continue;
            if (!is_prefix(d->transcript, sg->current_transcript)) {
                d->state = SPEC_CANCELLED;
            }
        }
    }

    pthread_mutex_unlock(&sg->mutex);
    return action;
}

void speculative_gen_feed_token(SpeculativeGen *sg, int draft_id, const char *token) {
    pthread_mutex_lock(&sg->mutex);
    for (int i = 0; i < sg->cfg.max_drafts; i++) {
        SpecDraft *d = &sg->drafts[i];
        if (d->draft_id == draft_id && (d->state == SPEC_PENDING || d->state == SPEC_READY)) {
            size_t resp_len = strlen(d->response);
            size_t tok_len = strlen(token);
            if (resp_len + tok_len < sizeof(d->response) - 1 &&
                d->n_tokens < sg->cfg.max_spec_tokens) {
                memcpy(d->response + resp_len, token, tok_len + 1);
                d->n_tokens++;
            }
            break;
        }
    }
    pthread_mutex_unlock(&sg->mutex);
}

void speculative_gen_draft_done(SpeculativeGen *sg, int draft_id) {
    pthread_mutex_lock(&sg->mutex);
    for (int i = 0; i < sg->cfg.max_drafts; i++) {
        SpecDraft *d = &sg->drafts[i];
        if (d->draft_id == draft_id) {
            if (d->state == SPEC_PENDING) d->state = SPEC_READY;
            break;
        }
    }
    pthread_mutex_unlock(&sg->mutex);
}

const SpecDraft *speculative_gen_get_best(const SpeculativeGen *sg) {
    pthread_mutex_lock((pthread_mutex_t *)&((SpeculativeGen *)sg)->mutex);
    SpecDraft *best = find_best_valid((SpeculativeGen *)sg, ((SpeculativeGen *)sg)->current_transcript);
    pthread_mutex_unlock((pthread_mutex_t *)&((SpeculativeGen *)sg)->mutex);
    return best;
}

const char *speculative_gen_commit(SpeculativeGen *sg) {
    pthread_mutex_lock(&sg->mutex);
    SpecDraft *best = find_best_valid(sg, sg->current_transcript);
    if (!best) {
        pthread_mutex_unlock(&sg->mutex);
        return NULL;
    }
    best->state = SPEC_COMMITTED;
    cancel_all_internal(sg);
    for (int i = 0; i < sg->cfg.max_drafts; i++) {
        if (&sg->drafts[i] != best && sg->drafts[i].state == SPEC_CANCELLED)
            ; /* already cancelled */
    }
    const char *resp = best->response[0] ? best->response : NULL;
    pthread_mutex_unlock(&sg->mutex);
    return resp;
}

void speculative_gen_cancel_all(SpeculativeGen *sg) {
    pthread_mutex_lock(&sg->mutex);
    cancel_all_internal(sg);
    pthread_mutex_unlock(&sg->mutex);
}

void speculative_gen_reset(SpeculativeGen *sg) {
    pthread_mutex_lock(&sg->mutex);
    for (int i = 0; i < SPEC_MAX_DRAFTS; i++) clear_draft(&sg->drafts[i]);
    sg->current_transcript[0] = '\0';
    sg->current_n_words = 0;
    pthread_mutex_unlock(&sg->mutex);
}

int speculative_gen_active_draft(const SpeculativeGen *sg) {
    pthread_mutex_lock((pthread_mutex_t *)&((SpeculativeGen *)sg)->mutex);
    int id = -1;
    for (int i = 0; i < sg->cfg.max_drafts; i++) {
        if (sg->drafts[i].state == SPEC_PENDING) {
            id = sg->drafts[i].draft_id;
            break;
        }
    }
    pthread_mutex_unlock((pthread_mutex_t *)&((SpeculativeGen *)sg)->mutex);
    return id;
}

int speculative_gen_draft_valid(const SpeculativeGen *sg, int draft_id,
                                const char *current_transcript) {
    pthread_mutex_lock((pthread_mutex_t *)&((SpeculativeGen *)sg)->mutex);
    int valid = 0;
    for (int i = 0; i < sg->cfg.max_drafts; i++) {
        if (sg->drafts[i].draft_id == draft_id) {
            valid = is_prefix(sg->drafts[i].transcript, current_transcript);
            break;
        }
    }
    pthread_mutex_unlock((pthread_mutex_t *)&((SpeculativeGen *)sg)->mutex);
    return valid;
}

int speculative_gen_active_count(const SpeculativeGen *sg) {
    pthread_mutex_lock((pthread_mutex_t *)&((SpeculativeGen *)sg)->mutex);
    int n = 0;
    for (int i = 0; i < sg->cfg.max_drafts; i++) {
        SpecState s = sg->drafts[i].state;
        if (s != SPEC_IDLE && s != SPEC_CANCELLED) n++;
    }
    pthread_mutex_unlock((pthread_mutex_t *)&((SpeculativeGen *)sg)->mutex);
    return n;
}

const SpecDraft *speculative_gen_get_draft(const SpeculativeGen *sg, int draft_id) {
    pthread_mutex_lock((pthread_mutex_t *)&((SpeculativeGen *)sg)->mutex);
    const SpecDraft *d = NULL;
    for (int i = 0; i < sg->cfg.max_drafts; i++) {
        if (sg->drafts[i].draft_id == draft_id) {
            d = &sg->drafts[i];
            break;
        }
    }
    pthread_mutex_unlock((pthread_mutex_t *)&((SpeculativeGen *)sg)->mutex);
    return d;
}
