/**
 * emphasis_predict.c — Lightweight emphasis and quoted-speech prediction.
 *
 * Scans LLM output text and inserts SSML <emphasis> / <voice> tags where
 * speech naturally needs stress, without requiring the LLM to produce SSML.
 *
 * Build: cc -O3 -shared -fPIC -o libemphasis_predict.dylib emphasis_predict.c
 */

#include "emphasis_predict.h"

#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>

/* ── Function word list (unstressed in English) ───────────────────────────── */

static const char *FUNCTION_WORDS[] = {
    "a", "an", "the", "is", "am", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will", "would",
    "shall", "should", "may", "might", "can", "could", "must",
    "to", "of", "in", "on", "at", "for", "by", "with", "from",
    "and", "or", "but", "if", "so", "that", "this", "it", "its",
    "i", "me", "my", "we", "us", "our", "you", "your", "he", "him",
    "his", "she", "her", "they", "them", "their",
    NULL
};

static int is_function_word(const char *word, int len) {
    char lower[32];
    if (len <= 0 || len > 30) return 0;
    for (int i = 0; i < len; i++) lower[i] = (char)tolower(word[i]);
    lower[len] = '\0';
    for (int i = 0; FUNCTION_WORDS[i]; i++) {
        if (strcmp(lower, FUNCTION_WORDS[i]) == 0) return 1;
    }
    return 0;
}

/* ── Contrast / intensifier markers ───────────────────────────────────────── */

static const char *CONTRAST_MARKERS[] = {
    "but", "however", "actually", "instead", "rather",
    "although", "yet", "still", "nevertheless", "nonetheless",
    NULL
};

static const char *INTENSIFIERS[] = {
    "very", "really", "extremely", "absolutely", "incredibly",
    "truly", "quite", "remarkably", "exceptionally", "particularly",
    NULL
};

static const char *NEGATIONS[] = {
    "not", "never", "no", "neither", "nor", "none", "nothing",
    "nowhere", "nobody", "cannot", "can't", "don't", "doesn't",
    "didn't", "won't", "wouldn't", "shouldn't", "couldn't",
    NULL
};

static int matches_list(const char *word, int len, const char **list) {
    char lower[32];
    if (len <= 0 || len > 30) return 0;
    for (int i = 0; i < len; i++) lower[i] = (char)tolower(word[i]);
    lower[len] = '\0';
    for (int i = 0; list[i]; i++) {
        if (strcmp(lower, list[i]) == 0) return 1;
    }
    return 0;
}

/* ── Word tokenizer ──────────────────────────────────────────────────────── */

typedef struct {
    int start;    /* offset in source string */
    int len;      /* word length */
    int emphasis;  /* 0=none, 1=mild, 2=strong */
} WordInfo;

#define MAX_WORDS 256

static int tokenize_words(const char *text, WordInfo *words, int max_words) {
    int n = 0;
    int i = 0;
    int text_len = (int)strlen(text);

    while (i < text_len && n < max_words) {
        /* Skip non-alpha */
        while (i < text_len && !isalpha(text[i]) && text[i] != '<') i++;
        if (i >= text_len) break;

        /* Skip existing SSML tags entirely */
        if (text[i] == '<') {
            while (i < text_len && text[i] != '>') i++;
            if (i < text_len) i++;
            continue;
        }

        int start = i;
        while (i < text_len && (isalpha(text[i]) || text[i] == '\'' || text[i] == '-'))
            i++;
        int len = i - start;
        if (len > 0 && len < 64) {
            words[n].start = start;
            words[n].len = len;
            words[n].emphasis = 0;
            n++;
        }
    }
    return n;
}

/* ── Emphasis prediction ─────────────────────────────────────────────────── */

int emphasis_predict(const char *input, char *output, int outsize) {
    if (!input || !output || outsize < 2) return 0;

    /* Already has SSML emphasis tags → pass through */
    if (strstr(input, "<emphasis") || strstr(input, "<EMPHASIS")) {
        int len = (int)strlen(input);
        int copy = len < outsize - 1 ? len : outsize - 1;
        memcpy(output, input, copy);
        output[copy] = '\0';
        return 0;
    }

    WordInfo words[MAX_WORDS];
    int nw = tokenize_words(input, words, MAX_WORDS);
    if (nw == 0) {
        int len = (int)strlen(input);
        int copy = len < outsize - 1 ? len : outsize - 1;
        memcpy(output, input, copy);
        output[copy] = '\0';
        return 0;
    }

    int emphasis_count = 0;

    /* Rule 1: Contrast markers → emphasize next content word */
    for (int i = 0; i < nw - 1; i++) {
        if (matches_list(input + words[i].start, words[i].len, CONTRAST_MARKERS)) {
            for (int j = i + 1; j < nw && j <= i + 2; j++) {
                if (!is_function_word(input + words[j].start, words[j].len)) {
                    words[j].emphasis = 2;
                    break;
                }
            }
        }
    }

    /* Rule 2: Intensifiers → emphasize next word */
    for (int i = 0; i < nw - 1; i++) {
        if (matches_list(input + words[i].start, words[i].len, INTENSIFIERS)) {
            if (i + 1 < nw) {
                words[i + 1].emphasis = (words[i + 1].emphasis > 1) ? 2 : 1;
            }
        }
    }

    /* Rule 3: Negation stress → emphasize negation + next word */
    for (int i = 0; i < nw; i++) {
        if (matches_list(input + words[i].start, words[i].len, NEGATIONS)) {
            words[i].emphasis = (words[i].emphasis > 1) ? 2 : 1;
            if (i + 1 < nw && !is_function_word(input + words[i+1].start, words[i+1].len)) {
                words[i + 1].emphasis = (words[i + 1].emphasis > 1) ? 2 : 1;
            }
        }
    }

    /* Rule 4: Sentence-final content word gets mild emphasis
     * (look at last 3 words, pick last content word) */
    for (int i = nw - 1; i >= 0 && i >= nw - 3; i--) {
        if (!is_function_word(input + words[i].start, words[i].len) &&
            words[i].emphasis == 0) {
            words[i].emphasis = 1;
            break;
        }
    }

    /* Rule 5: List-final item ("X, Y, and Z" → Z gets emphasis) */
    for (int i = 1; i < nw; i++) {
        char lower[32];
        int wl = words[i].len;
        if (wl > 0 && wl <= 3) {
            for (int k = 0; k < wl; k++) lower[k] = (char)tolower(input[words[i].start + k]);
            lower[wl] = '\0';
            if (strcmp(lower, "and") == 0 && i + 1 < nw) {
                /* Check there's a comma before this "and" */
                int before = words[i].start - 1;
                while (before >= 0 && isspace(input[before])) before--;
                if (before >= 0 && input[before] == ',') {
                    words[i + 1].emphasis = (words[i + 1].emphasis > 1) ? 2 : 1;
                }
            }
        }
    }

    /* Cap emphasis: max 3 emphasized words per sentence to avoid over-marking */
    int total_emph = 0;
    for (int i = 0; i < nw; i++) {
        if (words[i].emphasis > 0) total_emph++;
    }
    if (total_emph > 3) {
        /* Keep only the strongest */
        int keep = 0;
        for (int pass = 2; pass >= 1 && keep < 3; pass--) {
            for (int i = 0; i < nw && keep < 3; i++) {
                if (words[i].emphasis == pass) keep++;
                else if (words[i].emphasis > 0 && words[i].emphasis < pass)
                    words[i].emphasis = 0;
            }
        }
        /* Clear excess */
        int found = 0;
        for (int i = 0; i < nw; i++) {
            if (words[i].emphasis > 0) {
                found++;
                if (found > 3) words[i].emphasis = 0;
            }
        }
    }

    /* Build output with <emphasis> tags */
    int out_pos = 0;
    int in_pos = 0;
    int in_len = (int)strlen(input);

    for (int w = 0; w < nw; w++) {
        /* Copy text between previous position and this word */
        while (in_pos < words[w].start && out_pos < outsize - 1) {
            output[out_pos++] = input[in_pos++];
        }

        if (words[w].emphasis > 0) {
            const char *level = (words[w].emphasis >= 2) ? "strong" : "moderate";
            int tag_len = snprintf(output + out_pos, outsize - out_pos,
                                   "<emphasis level=\"%s\">", level);
            if (tag_len > 0) out_pos += tag_len;

            /* Copy word */
            int copy = words[w].len;
            if (out_pos + copy >= outsize - 30) copy = outsize - out_pos - 30;
            if (copy > 0) {
                memcpy(output + out_pos, input + words[w].start, copy);
                out_pos += copy;
            }
            in_pos = words[w].start + words[w].len;

            int end_len = snprintf(output + out_pos, outsize - out_pos, "</emphasis>");
            if (end_len > 0) out_pos += end_len;
            emphasis_count++;
        } else {
            /* Copy word as-is */
            int copy = words[w].len;
            if (out_pos + copy >= outsize - 1) copy = outsize - out_pos - 1;
            if (copy > 0) {
                memcpy(output + out_pos, input + words[w].start, copy);
                out_pos += copy;
            }
            in_pos = words[w].start + words[w].len;
        }
    }

    /* Copy remaining text */
    while (in_pos < in_len && out_pos < outsize - 1) {
        output[out_pos++] = input[in_pos++];
    }
    output[out_pos] = '\0';

    return emphasis_count;
}

/* ── Quoted speech detection ─────────────────────────────────────────────── */

int emphasis_detect_quotes(const char *input, char *output, int outsize) {
    if (!input || !output || outsize < 2) return 0;

    int in_len = (int)strlen(input);
    int out_pos = 0;
    int in_pos = 0;
    int quote_count = 0;

    while (in_pos < in_len && out_pos < outsize - 50) {
        /* Look for opening quote */
        if (input[in_pos] == '"' || input[in_pos] == '\xe2') {
            /* Handle smart quotes (UTF-8: E2 80 9C / E2 80 9D) */
            int is_smart_open = 0;
            if (input[in_pos] == '\xe2' && in_pos + 2 < in_len &&
                (unsigned char)input[in_pos+1] == 0x80 &&
                (unsigned char)input[in_pos+2] == 0x9C) {
                is_smart_open = 1;
            }

            if (input[in_pos] == '"' || is_smart_open) {
                int skip_open = is_smart_open ? 3 : 1;

                /* Find closing quote */
                int close_pos = in_pos + skip_open;
                int found_close = 0;
                while (close_pos < in_len) {
                    if (input[close_pos] == '"') {
                        found_close = 1;
                        break;
                    }
                    if (input[close_pos] == '\xe2' && close_pos + 2 < in_len &&
                        (unsigned char)input[close_pos+1] == 0x80 &&
                        (unsigned char)input[close_pos+2] == 0x9D) {
                        found_close = 1;
                        break;
                    }
                    close_pos++;
                }

                if (found_close) {
                    int skip_close = (input[close_pos] == '\xe2') ? 3 : 1;
                    int content_start = in_pos + skip_open;
                    int content_len = close_pos - content_start;

                    /* Only wrap substantial quoted text (>3 chars) */
                    if (content_len > 3 && content_len < 500) {
                        /* Copy the opening quote */
                        for (int k = 0; k < skip_open && out_pos < outsize - 50; k++)
                            output[out_pos++] = input[in_pos + k];

                        /* Insert <voice> tag */
                        int tag_len = snprintf(output + out_pos, outsize - out_pos,
                                               "<voice name=\"quoted\">");
                        if (tag_len > 0) out_pos += tag_len;

                        /* Copy quoted content */
                        int copy = content_len;
                        if (out_pos + copy >= outsize - 30) copy = outsize - out_pos - 30;
                        if (copy > 0) {
                            memcpy(output + out_pos, input + content_start, copy);
                            out_pos += copy;
                        }

                        int end_len = snprintf(output + out_pos, outsize - out_pos,
                                               "</voice>");
                        if (end_len > 0) out_pos += end_len;

                        /* Copy closing quote */
                        for (int k = 0; k < skip_close && out_pos < outsize - 1; k++)
                            output[out_pos++] = input[close_pos + k];

                        in_pos = close_pos + skip_close;
                        quote_count++;
                        continue;
                    }
                }
            }
        }

        output[out_pos++] = input[in_pos++];
    }

    /* Copy any remaining */
    while (in_pos < in_len && out_pos < outsize - 1) {
        output[out_pos++] = input[in_pos++];
    }
    output[out_pos] = '\0';

    return quote_count;
}
