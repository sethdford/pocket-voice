/**
 * prosody_predict.c — Text-based prosody prediction for TTS.
 *
 * Estimates prosody from text without neural models. Covers:
 *   - Syllable counting (English vowel-cluster heuristic)
 *   - Duration estimation (syllable-proportional frame allocation)
 *   - Multi-scale prosody analysis (utterance contour + word emphasis)
 *   - Text emotion detection (punctuation, capitalization, keywords)
 *   - Conversational adaptation (match user pace/energy via EMA)
 *   - EmoSteer direction vector loading from JSON
 *
 * Build: cc -O3 -shared -fPIC -o libprosody_predict.dylib prosody_predict.c cJSON.c
 */

#include "prosody_predict.h"
#include "cJSON.h"
#include <ctype.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ═══════════════════════════════════════════════════════════════════════════
 * Syllable Counting
 * ═══════════════════════════════════════════════════════════════════════════ */

static int is_vowel(char c) {
    c = (char)tolower((unsigned char)c);
    return c == 'a' || c == 'e' || c == 'i' || c == 'o' || c == 'u' || c == 'y';
}

int prosody_count_syllables(const char *word) {
    if (!word || !*word) return 0;

    int len = (int)strlen(word);
    int count = 0;
    int prev_vowel = 0;

    for (int i = 0; i < len; i++) {
        if (!isalpha((unsigned char)word[i])) {
            prev_vowel = 0;
            continue;
        }
        if (is_vowel(word[i])) {
            if (!prev_vowel) count++;
            prev_vowel = 1;
        } else {
            prev_vowel = 0;
        }
    }

    /* Silent-e: subtract 1 if word ends in 'e' and has >1 syllable */
    if (count > 1 && len > 1 && tolower((unsigned char)word[len - 1]) == 'e' &&
        !is_vowel(word[len - 2]))
        count--;

    /* Words like "the", "me" — at least 1 syllable */
    if (count == 0 && len > 0) count = 1;

    return count;
}

int prosody_count_sentence_syllables(const char *text) {
    if (!text) return 0;
    int total = 0;
    char word[256];
    int wi = 0;

    for (int i = 0; ; i++) {
        char c = text[i];
        if (isalpha((unsigned char)c) || c == '\'') {
            if (wi < 255) word[wi++] = c;
        } else {
            if (wi > 0) {
                word[wi] = '\0';
                total += prosody_count_syllables(word);
                wi = 0;
            }
        }
        if (c == '\0') break;
    }
    return total;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Duration Estimation
 * ═══════════════════════════════════════════════════════════════════════════ */

int prosody_estimate_durations(const char *text, int n_tokens,
                               float *durations, int max_frames) {
    if (!text || !durations || n_tokens <= 0 || max_frames <= 0) return 0;

    int syllables = prosody_count_sentence_syllables(text);
    if (syllables <= 0) syllables = 1;

    /* Average speaking rate: ~4.5 syllables/sec = ~9 frames/syllable at 50Hz */
    float frames_per_syllable = 9.0f;
    float total_frames = (float)syllables * frames_per_syllable;

    int n = n_tokens < max_frames ? n_tokens : max_frames;

    /* Parse words and compute per-word syllable proportions */
    int word_syllables[256];
    int word_start_token[256]; /* approximate token index for each word */
    int n_words = 0;
    int total_syl = 0;
    char word[256];
    int wi = 0;
    int char_pos = 0;

    for (int i = 0; ; i++) {
        char c = text[i];
        if (isalpha((unsigned char)c) || c == '\'' || c == '-') {
            if (wi < 255) word[wi++] = c;
        } else {
            if (wi > 0 && n_words < 256) {
                word[wi] = '\0';
                int syl = prosody_count_syllables(word);
                word_syllables[n_words] = syl > 0 ? syl : 1;
                /* Map character position to approximate token index */
                word_start_token[n_words] = (int)((float)char_pos / (float)(strlen(text) + 1) * n);
                total_syl += word_syllables[n_words];
                n_words++;
                wi = 0;
            }
            char_pos = i + 1;
        }
        if (c == '\0') break;
    }

    if (total_syl <= 0) total_syl = 1;

    /* Distribute frames proportional to syllable count per word */
    float base = logf(total_frames / (float)n);
    for (int i = 0; i < n; i++) {
        durations[i] = base;
    }

    /* Overlay word-level duration variation */
    for (int w = 0; w < n_words; w++) {
        float word_frac = (float)word_syllables[w] / (float)total_syl;
        float word_log_dur = logf(word_frac * total_frames / (word_syllables[w] > 0 ? word_syllables[w] : 1));

        int start = word_start_token[w];
        int end = (w + 1 < n_words) ? word_start_token[w + 1] : n;
        if (end > n) end = n;
        for (int i = start; i < end; i++) {
            durations[i] = word_log_dur;
        }
    }

    return n;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Multi-Scale Prosody Analysis
 * ═══════════════════════════════════════════════════════════════════════════ */

static int word_is_all_caps(const char *s, int len) {
    if (len < 2) return 0;
    int alpha_count = 0;
    for (int i = 0; i < len; i++) {
        if (isalpha((unsigned char)s[i])) {
            if (islower((unsigned char)s[i])) return 0;
            alpha_count++;
        }
    }
    return alpha_count >= 2;
}

static int is_function_word(const char *word) {
    static const char *fw[] = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been",
        "am", "do", "does", "did", "have", "has", "had", "will",
        "would", "could", "should", "may", "might", "can", "shall",
        "to", "of", "in", "for", "on", "at", "by", "with", "from",
        "and", "or", "but", "not", "so", "if", "as", "it", "its",
        "that", "this", "he", "she", "we", "they", "my", "your",
        NULL
    };
    char lower[32];
    int len = (int)strlen(word);
    if (len > 31) return 0;
    for (int i = 0; i < len; i++) lower[i] = (char)tolower((unsigned char)word[i]);
    lower[len] = '\0';
    for (int i = 0; fw[i]; i++) {
        if (strcmp(lower, fw[i]) == 0) return 1;
    }
    return 0;
}

MultiScaleProsody prosody_analyze_text(const char *text) {
    MultiScaleProsody msp;
    memset(&msp, 0, sizeof(msp));
    msp.utterance.pitch = 1.0f;
    msp.utterance.rate = 1.0f;
    msp.utterance.energy = 0.0f;

    if (!text || !*text) return msp;

    int len = (int)strlen(text);

    /* Detect utterance contour from terminal punctuation */
    char last = '\0';
    for (int i = len - 1; i >= 0; i--) {
        if (!isspace((unsigned char)text[i])) { last = text[i]; break; }
    }

    if (last == '?') {
        msp.contour = PROSODY_CONTOUR_INTERROGATIVE;
        msp.utterance.pitch = 1.08f;
        msp.utterance.rate = 0.95f;
    } else if (last == '!') {
        msp.contour = PROSODY_CONTOUR_EXCLAMATORY;
        msp.utterance.pitch = 1.06f;
        msp.utterance.energy = 1.5f;
    } else if (last == ',') {
        msp.contour = PROSODY_CONTOUR_CONTINUATION;
        msp.utterance.pitch = 1.03f;
    } else {
        msp.contour = PROSODY_CONTOUR_DECLARATIVE;
    }

    /* Detect list pattern (comma-separated items) */
    int comma_count = 0;
    for (int i = 0; i < len; i++) {
        if (text[i] == ',') comma_count++;
    }
    if (comma_count >= 2 && last != '?') {
        msp.contour = PROSODY_CONTOUR_LIST;
    }

    /* Parse words for word-level prosody */
    int word_start = 0;
    int in_word = 0;
    for (int i = 0; i <= len && msp.n_words < 64; i++) {
        char c = (i < len) ? text[i] : '\0';
        int is_ws = (c == ' ' || c == '\t' || c == '\n' || c == '\0');

        if (!is_ws && !in_word) {
            word_start = i;
            in_word = 1;
        } else if (is_ws && in_word) {
            int wlen = i - word_start;
            char word[128];
            int copy = wlen < 127 ? wlen : 127;
            memcpy(word, text + word_start, copy);
            word[copy] = '\0';

            int w = msp.n_words;
            msp.word_hints[w].pitch = 1.0f;
            msp.word_hints[w].rate = 1.0f;
            msp.word_hints[w].energy = 0.0f;

            /* ALL CAPS words get emphasis */
            if (word_is_all_caps(word, wlen)) {
                msp.emphasis_mask[w] = 1;
                msp.word_hints[w].pitch = 1.10f;
                msp.word_hints[w].rate = 0.90f;
                msp.word_hints[w].energy = 2.0f;
            }

            /* Function words: de-emphasize */
            if (is_function_word(word)) {
                msp.word_hints[w].rate = 1.05f;
                msp.word_hints[w].energy = -0.5f;
            }

            /* List items: rise on non-final items */
            if (msp.contour == PROSODY_CONTOUR_LIST && wlen > 0 &&
                word[wlen - 1] == ',') {
                msp.word_hints[w].pitch = 1.05f;
            }

            msp.n_words++;
            in_word = 0;
        }
    }

    return msp;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Text Emotion Detection
 * ═══════════════════════════════════════════════════════════════════════════ */

static const struct {
    const char *keyword;
    DetectedEmotion emotion;
    float weight;
} EMOTION_KEYWORDS[] = {
    /* Happy / Joy */
    {"wonderful",    EMOTION_HAPPY,     0.6f},
    {"great",        EMOTION_HAPPY,     0.4f},
    {"love",         EMOTION_HAPPY,     0.5f},
    {"happy",        EMOTION_HAPPY,     0.6f},
    {"glad",         EMOTION_HAPPY,     0.5f},
    {"delighted",    EMOTION_HAPPY,     0.6f},
    {"pleased",      EMOTION_HAPPY,     0.5f},
    {"cheerful",     EMOTION_HAPPY,     0.5f},
    {"joyful",       EMOTION_HAPPY,     0.6f},
    {"thrilled",     EMOTION_HAPPY,     0.7f},
    {"grateful",     EMOTION_WARM,      0.5f},
    {"thankful",     EMOTION_WARM,      0.5f},
    /* Excited / Enthusiastic */
    {"amazing",      EMOTION_EXCITED,   0.7f},
    {"fantastic",    EMOTION_EXCITED,   0.7f},
    {"incredible",   EMOTION_EXCITED,   0.6f},
    {"awesome",      EMOTION_EXCITED,   0.6f},
    {"extraordinary",EMOTION_EXCITED,   0.6f},
    {"brilliant",    EMOTION_EXCITED,   0.6f},
    {"exciting",     EMOTION_EXCITED,   0.6f},
    {"passionate",   EMOTION_EXCITED,   0.6f},
    {"energetic",    EMOTION_EXCITED,   0.5f},
    /* Sad / Melancholy */
    {"sorry",        EMOTION_SAD,       0.5f},
    {"unfortunately",EMOTION_SAD,       0.6f},
    {"sadly",        EMOTION_SAD,       0.6f},
    {"heartbreaking",EMOTION_SAD,       0.7f},
    {"devastating",  EMOTION_SAD,       0.7f},
    {"tragic",       EMOTION_SAD,       0.7f},
    {"grief",        EMOTION_SAD,       0.7f},
    {"mourning",     EMOTION_SAD,       0.6f},
    {"depressing",   EMOTION_SAD,       0.6f},
    {"lonely",       EMOTION_SAD,       0.5f},
    {"disappointed", EMOTION_SAD,       0.5f},
    {"nostalgic",    EMOTION_SAD,       0.4f},
    {"miss",         EMOTION_SAD,       0.3f},
    /* Angry / Frustrated */
    {"terrible",     EMOTION_ANGRY,     0.5f},
    {"awful",        EMOTION_ANGRY,     0.5f},
    {"angry",        EMOTION_ANGRY,     0.7f},
    {"furious",      EMOTION_ANGRY,     0.8f},
    {"outrageous",   EMOTION_ANGRY,     0.7f},
    {"infuriating",  EMOTION_ANGRY,     0.7f},
    {"disgusting",   EMOTION_ANGRY,     0.6f},
    {"frustrating",  EMOTION_ANGRY,     0.6f},
    {"annoying",     EMOTION_ANGRY,     0.5f},
    {"irritating",   EMOTION_ANGRY,     0.5f},
    {"unacceptable", EMOTION_ANGRY,     0.6f},
    {"ridiculous",   EMOTION_ANGRY,     0.5f},
    /* Surprised / Amazed */
    {"wow",          EMOTION_SURPRISED, 0.6f},
    {"unbelievable", EMOTION_SURPRISED, 0.6f},
    {"shocking",     EMOTION_SURPRISED, 0.6f},
    {"astonishing",  EMOTION_SURPRISED, 0.6f},
    {"remarkable",   EMOTION_SURPRISED, 0.5f},
    {"unexpected",   EMOTION_SURPRISED, 0.5f},
    {"stunning",     EMOTION_SURPRISED, 0.5f},
    {"mind-blowing", EMOTION_SURPRISED, 0.7f},
    /* Fearful / Anxious */
    {"scary",        EMOTION_FEARFUL,   0.6f},
    {"afraid",       EMOTION_FEARFUL,   0.6f},
    {"worried",      EMOTION_FEARFUL,   0.5f},
    {"terrifying",   EMOTION_FEARFUL,   0.7f},
    {"frightening",  EMOTION_FEARFUL,   0.6f},
    {"alarming",     EMOTION_FEARFUL,   0.5f},
    {"nervous",      EMOTION_FEARFUL,   0.5f},
    {"panicked",     EMOTION_FEARFUL,   0.7f},
    {"dreadful",     EMOTION_FEARFUL,   0.6f},
    {"anxious",      EMOTION_FEARFUL,   0.5f},
    /* Warm / Friendly */
    {"caring",       EMOTION_WARM,      0.5f},
    {"kind",         EMOTION_WARM,      0.4f},
    {"gentle",       EMOTION_WARM,      0.5f},
    {"tender",       EMOTION_WARM,      0.5f},
    {"compassionate",EMOTION_WARM,      0.6f},
    {"empathetic",   EMOTION_WARM,      0.5f},
    {"sympathetic",  EMOTION_WARM,      0.5f},
    {"heartfelt",    EMOTION_WARM,      0.6f},
    {"hopeful",      EMOTION_WARM,      0.5f},
    /* Serious / Thoughtful */
    {"important",    EMOTION_SERIOUS,   0.4f},
    {"critical",     EMOTION_SERIOUS,   0.5f},
    {"significant",  EMOTION_SERIOUS,   0.4f},
    {"concerning",   EMOTION_SERIOUS,   0.5f},
    {"grave",        EMOTION_SERIOUS,   0.6f},
    {"solemn",       EMOTION_SERIOUS,   0.6f},
    {"thoughtful",   EMOTION_SERIOUS,   0.5f},
    /* Calm / Peaceful */
    {"calm",         EMOTION_CALM,      0.5f},
    {"peaceful",     EMOTION_CALM,      0.6f},
    {"relaxed",      EMOTION_CALM,      0.5f},
    {"serene",       EMOTION_CALM,      0.6f},
    {"tranquil",     EMOTION_CALM,      0.6f},
    {"soothing",     EMOTION_CALM,      0.5f},
    {"quiet",        EMOTION_CALM,      0.3f},
    /* Confident / Authoritative */
    {"sure",         EMOTION_CONFIDENT, 0.4f},
    {"certainly",    EMOTION_CONFIDENT, 0.5f},
    {"absolutely",   EMOTION_CONFIDENT, 0.6f},
    {"definitely",   EMOTION_CONFIDENT, 0.5f},
    {"confident",    EMOTION_CONFIDENT, 0.6f},
    {"determined",   EMOTION_CONFIDENT, 0.5f},
    {"proud",        EMOTION_CONFIDENT, 0.5f},
    {"bold",         EMOTION_CONFIDENT, 0.5f},
    {NULL, EMOTION_NEUTRAL, 0.0f}
};

static const ProsodyHint EMOTION_HINTS[EMOTION_COUNT] = {
    [EMOTION_NEUTRAL]   = { 1.00f, 1.00f,  0.0f },
    [EMOTION_HAPPY]     = { 1.08f, 1.05f,  1.5f },
    [EMOTION_EXCITED]   = { 1.15f, 1.12f,  3.0f },
    [EMOTION_SAD]       = { 0.92f, 0.88f, -2.0f },
    [EMOTION_ANGRY]     = { 1.05f, 1.08f,  4.0f },
    [EMOTION_SURPRISED] = { 1.18f, 1.10f,  2.0f },
    [EMOTION_WARM]      = { 0.97f, 0.95f,  0.5f },
    [EMOTION_SERIOUS]   = { 0.94f, 0.92f,  1.0f },
    [EMOTION_CALM]      = { 0.96f, 0.90f, -1.0f },
    [EMOTION_CONFIDENT] = { 1.03f, 1.02f,  2.0f },
    [EMOTION_FEARFUL]   = { 1.10f, 1.05f, -1.0f },
};

static int str_contains_ci(const char *haystack, const char *needle) {
    if (!haystack || !needle) return 0;
    int hlen = (int)strlen(haystack);
    int nlen = (int)strlen(needle);
    if (nlen > hlen) return 0;
    for (int i = 0; i <= hlen - nlen; i++) {
        int match = 1;
        for (int j = 0; j < nlen; j++) {
            if (tolower((unsigned char)haystack[i + j]) !=
                tolower((unsigned char)needle[j])) {
                match = 0;
                break;
            }
        }
        if (match) return 1;
    }
    return 0;
}

EmotionDetection prosody_detect_emotion(const char *text) {
    EmotionDetection det;
    memset(&det, 0, sizeof(det));
    det.emotion = EMOTION_NEUTRAL;
    det.hint = EMOTION_HINTS[EMOTION_NEUTRAL];

    if (!text || !*text) return det;

    int len = (int)strlen(text);
    float scores[EMOTION_COUNT] = {0};

    /* Punctuation signals */
    int excl_count = 0, quest_count = 0, caps_words = 0, total_words = 0;
    int ellipsis = 0;
    for (int i = 0; i < len; i++) {
        if (text[i] == '!') excl_count++;
        if (text[i] == '?') quest_count++;
        if (i + 2 < len && text[i] == '.' && text[i+1] == '.' && text[i+2] == '.') ellipsis++;
    }

    /* Count capitalized words */
    int in_word = 0;
    char word[128];
    int wi = 0;
    for (int i = 0; i <= len; i++) {
        char c = (i < len) ? text[i] : '\0';
        if (isalpha((unsigned char)c)) {
            if (wi < 127) word[wi++] = c;
            if (!in_word) { in_word = 1; total_words++; }
        } else {
            if (in_word && wi > 0) {
                word[wi] = '\0';
                if (word_is_all_caps(word, wi)) caps_words++;
                wi = 0;
            }
            in_word = 0;
        }
    }

    /* Multiple exclamation marks → excited */
    if (excl_count >= 2) scores[EMOTION_EXCITED] += 0.5f;
    else if (excl_count == 1) scores[EMOTION_HAPPY] += 0.2f;

    /* Multiple question marks → surprised or confused */
    if (quest_count >= 2) scores[EMOTION_SURPRISED] += 0.4f;

    /* Ellipsis → sad or hesitant */
    if (ellipsis > 0) scores[EMOTION_SAD] += 0.3f;

    /* Lots of caps → angry or excited */
    if (total_words > 3 && caps_words >= total_words / 2) {
        scores[EMOTION_ANGRY] += 0.4f;
        scores[EMOTION_EXCITED] += 0.3f;
    }

    /* Keyword matching */
    for (int k = 0; EMOTION_KEYWORDS[k].keyword; k++) {
        if (str_contains_ci(text, EMOTION_KEYWORDS[k].keyword)) {
            scores[EMOTION_KEYWORDS[k].emotion] += EMOTION_KEYWORDS[k].weight;
        }
    }

    /* Find best */
    float best_score = 0.0f;
    int best_idx = EMOTION_NEUTRAL;
    for (int i = 1; i < EMOTION_COUNT; i++) {
        if (scores[i] > best_score) {
            best_score = scores[i];
            best_idx = i;
        }
    }

    /* Need minimum confidence to override neutral */
    if (best_score >= 0.3f) {
        det.emotion = (DetectedEmotion)best_idx;
        det.confidence = best_score > 1.0f ? 1.0f : best_score;
    }
    det.hint = EMOTION_HINTS[det.emotion];

    return det;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Conversational Prosody Adaptation
 * ═══════════════════════════════════════════════════════════════════════════ */

void prosody_conversation_init(ConversationProsodyState *state) {
    if (!state) return;
    memset(state, 0, sizeof(*state));
    state->ema_rate = 3.0f;     /* default ~3 words/sec */
    state->ema_energy = -20.0f; /* default energy baseline (dB) */
    state->ema_pitch = 150.0f;  /* default pitch baseline (Hz) */
}

void prosody_conversation_update(ConversationProsodyState *state,
                                 float speech_duration_sec,
                                 int word_count,
                                 float mean_energy_db,
                                 float mean_pitch_hz) {
    if (!state || speech_duration_sec <= 0.0f) return;

    float rate = (float)word_count / speech_duration_sec;
    state->user_rate = rate;
    state->user_energy = mean_energy_db;
    state->user_pitch_mean = mean_pitch_hz;
    state->n_samples++;

    float alpha = 0.3f;
    state->ema_rate = alpha * rate + (1.0f - alpha) * state->ema_rate;
    state->ema_energy = alpha * mean_energy_db + (1.0f - alpha) * state->ema_energy;
    state->ema_pitch = alpha * mean_pitch_hz + (1.0f - alpha) * state->ema_pitch;
}

ProsodyHint prosody_conversation_adapt(const ConversationProsodyState *state) {
    ProsodyHint hint = { 1.0f, 1.0f, 0.0f };
    if (!state || state->n_samples == 0) return hint;

    /* Adapt rate: if user speaks fast (>4 wps), speed up slightly; if slow (<2), slow down */
    float rate_ratio = state->ema_rate / 3.0f; /* normalize to ~3 wps baseline */
    if (rate_ratio >= 1.15f) {
        hint.rate = 1.0f + (rate_ratio - 1.15f) * 0.4f; /* partial match, not full mirror */
        if (hint.rate > 1.15f) hint.rate = 1.15f;
    } else if (rate_ratio < 0.8f) {
        hint.rate = 1.0f - (0.8f - rate_ratio) * 0.3f;
        if (hint.rate < 0.88f) hint.rate = 0.88f;
    }

    /* Adapt energy: if user is loud, slightly louder response; if quiet, quieter */
    float energy_offset = state->ema_energy - (-20.0f); /* offset from baseline */
    hint.energy = energy_offset * 0.2f; /* 20% of user's energy deviation */
    if (hint.energy > 3.0f) hint.energy = 3.0f;
    if (hint.energy < -3.0f) hint.energy = -3.0f;

    /* Adapt pitch: subtle mirror of user's average pitch deviation */
    float pitch_ratio = state->ema_pitch / 150.0f;
    if (pitch_ratio > 1.1f) hint.pitch = 1.0f + (pitch_ratio - 1.1f) * 0.2f;
    else if (pitch_ratio < 0.9f) hint.pitch = 1.0f - (0.9f - pitch_ratio) * 0.2f;
    if (hint.pitch > 1.10f) hint.pitch = 1.10f;
    if (hint.pitch < 0.92f) hint.pitch = 0.92f;

    return hint;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * EmoSteer Direction Vector I/O
 * ═══════════════════════════════════════════════════════════════════════════ */

static const char *EMOSTEER_EMOTION_NAMES[] = {
    "happy", "excited", "sad", "angry", "fearful",
    "surprised", "warm", "serious", "calm", "confident",
    "whisper", "emphatic", NULL
};

EmoSteerBank *emosteer_load(const char *json_path) {
    if (!json_path) return NULL;

    FILE *f = fopen(json_path, "rb");
    if (!f) {
        fprintf(stderr, "[emosteer] Cannot open: %s\n", json_path);
        return NULL;
    }
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *buf = (char *)malloc(sz + 1);
    if (!buf) { fclose(f); return NULL; }
    fread(buf, 1, sz, f);
    buf[sz] = '\0';
    fclose(f);

    cJSON *root = cJSON_Parse(buf);
    free(buf);
    if (!root) {
        fprintf(stderr, "[emosteer] JSON parse failed: %s\n", json_path);
        return NULL;
    }

    cJSON *dim_j = cJSON_GetObjectItem(root, "dim");
    cJSON *ls_j  = cJSON_GetObjectItem(root, "layer_start");
    cJSON *le_j  = cJSON_GetObjectItem(root, "layer_end");
    cJSON *sc_j  = cJSON_GetObjectItem(root, "scale");
    cJSON *emo_j = cJSON_GetObjectItem(root, "emotions");

    if (!dim_j || !emo_j) {
        fprintf(stderr, "[emosteer] Missing 'dim' or 'emotions' in JSON\n");
        cJSON_Delete(root);
        return NULL;
    }

    int dim = dim_j->valueint;
    int n_emotions = cJSON_GetArraySize(emo_j);
    /* Count actual named emotions */
    int count = 0;
    for (int i = 0; EMOSTEER_EMOTION_NAMES[i]; i++) {
        if (cJSON_GetObjectItem(emo_j, EMOSTEER_EMOTION_NAMES[i])) count++;
    }
    if (count == 0) {
        fprintf(stderr, "[emosteer] No recognized emotions in JSON\n");
        cJSON_Delete(root);
        return NULL;
    }

    EmoSteerBank *bank = (EmoSteerBank *)calloc(1, sizeof(EmoSteerBank));
    if (!bank) { cJSON_Delete(root); return NULL; }
    bank->dim = dim;
    bank->n_emotions = count;
    bank->layer_start = ls_j ? ls_j->valueint : 0;
    bank->layer_end = le_j ? le_j->valueint : 7;
    bank->default_scale = sc_j ? (float)sc_j->valuedouble : 0.5f;
    bank->directions = (float *)calloc(count * dim, sizeof(float));
    bank->name_indices = (int *)calloc(count, sizeof(int));
    if (!bank->directions || !bank->name_indices) {
        free(bank->directions);
        free(bank->name_indices);
        free(bank);
        cJSON_Delete(root);
        return NULL;
    }

    int idx = 0;
    for (int e = 0; EMOSTEER_EMOTION_NAMES[e]; e++) {
        cJSON *arr = cJSON_GetObjectItem(emo_j, EMOSTEER_EMOTION_NAMES[e]);
        if (!arr || !cJSON_IsArray(arr)) continue;
        bank->name_indices[idx] = e;
        int arr_size = cJSON_GetArraySize(arr);
        int copy = arr_size < dim ? arr_size : dim;
        for (int i = 0; i < copy; i++) {
            cJSON *v = cJSON_GetArrayItem(arr, i);
            bank->directions[idx * dim + i] = v ? (float)v->valuedouble : 0.0f;
        }
        idx++;
    }

    (void)n_emotions;
    cJSON_Delete(root);
    fprintf(stderr, "[emosteer] Loaded %d emotions (%d-dim) from %s\n", count, dim, json_path);
    return bank;
}

void emosteer_destroy(EmoSteerBank *bank) {
    if (!bank) return;
    free(bank->directions);
    free(bank->name_indices);
    free(bank);
}

const float *emosteer_get_direction(const EmoSteerBank *bank, const char *emotion_name) {
    if (!bank || !emotion_name || !bank->directions) return NULL;

    for (int i = 0; i < bank->n_emotions; i++) {
        if (strcasecmp(EMOSTEER_EMOTION_NAMES[bank->name_indices[i]], emotion_name) == 0) {
            return &bank->directions[i * bank->dim];
        }
    }
    return NULL;
}

int emosteer_count(const EmoSteerBank *bank) {
    return bank ? bank->n_emotions : 0;
}
