/**
 * prosody_predict.h — Text-based prosody prediction for TTS.
 *
 * Estimates prosody features from text without neural models:
 *   - Syllable counting for duration estimation
 *   - Multi-scale prosody (utterance / word / phoneme level)
 *   - Emotion detection from text patterns
 *   - Conversational prosody adaptation from user speech features
 *   - EmoSteer direction vector loading
 */

#ifndef PROSODY_PREDICT_H
#define PROSODY_PREDICT_H

#ifdef __cplusplus
extern "C" {
#endif

/* ── Syllable-Based Duration Estimation ─────────────────── */

/** Count syllables in a word (English heuristic: vowel cluster counting). */
int prosody_count_syllables(const char *word);

/** Count total syllables in a sentence. */
int prosody_count_sentence_syllables(const char *text);

/**
 * Estimate per-token durations (in 50Hz frames) from text.
 * Distributes expected speech duration across tokens proportional
 * to estimated syllable count per word.
 *
 * @param text        Input text
 * @param n_tokens    Number of semantic tokens the LM will produce (estimate)
 * @param durations   Output: float[n_tokens] log-duration per frame
 * @param max_frames  Capacity of durations array
 * @return Number of frames written
 */
int prosody_estimate_durations(const char *text, int n_tokens,
                               float *durations, int max_frames);

/* ── Multi-Scale Prosody ────────────────────────────────── */

typedef enum {
    PROSODY_CONTOUR_DECLARATIVE,   /* falling final pitch */
    PROSODY_CONTOUR_INTERROGATIVE, /* rising final pitch */
    PROSODY_CONTOUR_EXCLAMATORY,   /* high-falling with energy */
    PROSODY_CONTOUR_IMPERATIVE,    /* level-to-falling, louder */
    PROSODY_CONTOUR_CONTINUATION,  /* slight rise (mid-sentence) */
    PROSODY_CONTOUR_LIST,          /* enumeration: rise on all but last */
} ProsodyContour;

typedef struct {
    float pitch;      /* multiplier (1.0 = neutral) */
    float rate;       /* multiplier (1.0 = neutral) */
    float energy;     /* dB offset (0.0 = neutral) */
} ProsodyHint;

typedef struct {
    ProsodyContour contour;
    ProsodyHint    utterance;     /* utterance-level prosody */
    int            n_words;
    ProsodyHint    word_hints[64]; /* per-word prosody hints */
    int            emphasis_mask[64]; /* 1 = emphasized word */
} MultiScaleProsody;

/**
 * Analyze text and produce multi-scale prosody annotations.
 * Detects utterance contour, word-level emphasis, and prosodic phrasing.
 */
MultiScaleProsody prosody_analyze_text(const char *text);

/* ── Text Emotion Detection ─────────────────────────────── */

typedef enum {
    EMOTION_NEUTRAL = 0,
    EMOTION_HAPPY,
    EMOTION_EXCITED,
    EMOTION_SAD,
    EMOTION_ANGRY,
    EMOTION_SURPRISED,
    EMOTION_WARM,
    EMOTION_SERIOUS,
    EMOTION_CALM,
    EMOTION_CONFIDENT,
    EMOTION_FEARFUL,
    EMOTION_COUNT
} DetectedEmotion;

typedef struct {
    DetectedEmotion emotion;
    float           confidence;   /* 0.0-1.0 */
    ProsodyHint     hint;         /* suggested prosody for this emotion */
} EmotionDetection;

/**
 * Detect emotion from text patterns (punctuation, keywords, capitalization).
 * Returns the most likely emotion with confidence score.
 */
EmotionDetection prosody_detect_emotion(const char *text);

/* ── Conversational Prosody Adaptation ──────────────────── */

typedef struct {
    float user_rate;          /* estimated speaking rate (words/sec) */
    float user_energy;        /* mean energy (dB, normalized) */
    float user_pitch_mean;    /* mean pitch offset from baseline */
    int   n_samples;          /* number of turns observed */
    /* EMA state */
    float ema_rate;
    float ema_energy;
    float ema_pitch;
} ConversationProsodyState;

/** Initialize conversational adaptation state. */
void prosody_conversation_init(ConversationProsodyState *state);

/**
 * Update state from user speech features.
 * Called after each STT turn with measured features.
 */
void prosody_conversation_update(ConversationProsodyState *state,
                                 float speech_duration_sec,
                                 int word_count,
                                 float mean_energy_db,
                                 float mean_pitch_hz);

/**
 * Get adapted prosody for the next response based on user's style.
 * Returns a ProsodyHint that matches/complements the user's pace and energy.
 */
ProsodyHint prosody_conversation_adapt(const ConversationProsodyState *state);

/* ── EmoSteer Direction Vector I/O ──────────────────────── */

typedef struct {
    float *directions;     /* [n_emotions * dim] packed direction vectors */
    int   *name_indices;   /* EMOSTEER_EMOTION_NAMES index for each loaded slot */
    int    n_emotions;
    int    dim;            /* d_model of the flow model */
    int    layer_start;
    int    layer_end;
    float  default_scale;
} EmoSteerBank;

/**
 * Load EmoSteer direction vectors from a JSON file.
 * Format: { "dim": 512, "layer_start": 2, "layer_end": 7, "scale": 0.5,
 *           "emotions": { "happy": [...], "sad": [...], ... } }
 * Returns NULL on failure.
 */
EmoSteerBank *emosteer_load(const char *json_path);

/** Free an EmoSteer bank. NULL-safe. */
void emosteer_destroy(EmoSteerBank *bank);

/**
 * Get the direction vector for a named emotion.
 * Returns pointer to float[dim] within the bank, or NULL if not found.
 */
const float *emosteer_get_direction(const EmoSteerBank *bank, const char *emotion_name);

/** Number of loaded emotions. */
int emosteer_count(const EmoSteerBank *bank);

#ifdef __cplusplus
}
#endif

#endif /* PROSODY_PREDICT_H */
