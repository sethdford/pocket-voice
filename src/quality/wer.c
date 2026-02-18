/**
 * wer.c — Word Error Rate and Character Error Rate via Levenshtein distance.
 *
 * Uses dynamic programming with O(n*m) time and O(m) space (rolling row).
 * Text is normalized before comparison: lowercased, punctuation stripped,
 * whitespace collapsed.
 */

#include "wer.h"
#include <ctype.h>
#include <stdlib.h>
#include <string.h>

#define MAX_WORDS 4096
#define MAX_CHARS 16384

/* ── Text Normalization ───────────────────────────────── */

int wer_normalize(const char *input, char *out, int out_cap)
{
    if (!input || !out || out_cap <= 0) return 0;

    int oi = 0;
    int prev_space = 1; /* Suppress leading spaces */

    for (int i = 0; input[i] && oi < out_cap - 1; i++) {
        unsigned char ch = (unsigned char)input[i];

        /* Strip punctuation except apostrophes (contractions) */
        if (ispunct(ch) && ch != '\'') continue;

        /* Collapse whitespace */
        if (isspace(ch)) {
            if (!prev_space && oi < out_cap - 1) {
                out[oi++] = ' ';
                prev_space = 1;
            }
            continue;
        }

        out[oi++] = (char)tolower(ch);
        prev_space = 0;
    }

    /* Trim trailing space */
    if (oi > 0 && out[oi - 1] == ' ') oi--;
    out[oi] = '\0';
    return oi;
}

/* ── Word Tokenization ────────────────────────────────── */

static int tokenize(const char *text, const char **words, int max_words)
{
    int count = 0;
    const char *p = text;

    while (*p && count < max_words) {
        while (*p == ' ') p++;
        if (!*p) break;

        words[count++] = p;
        while (*p && *p != ' ') p++;
    }

    return count;
}

static int word_len(const char *word)
{
    int len = 0;
    while (word[len] && word[len] != ' ') len++;
    return len;
}

static int words_equal(const char *a, const char *b)
{
    int la = word_len(a);
    int lb = word_len(b);
    if (la != lb) return 0;
    return strncmp(a, b, (size_t)la) == 0;
}

/* ── Levenshtein Distance (word-level) ────────────────── */

static void levenshtein_words(const char **ref, int ref_n,
                               const char **hyp, int hyp_n,
                               int *out_sub, int *out_del, int *out_ins)
{
    /* O(m) space DP with traceback for S/D/I counts */
    int m = hyp_n;
    int *prev = (int *)calloc((size_t)(m + 1), sizeof(int));
    int *curr = (int *)calloc((size_t)(m + 1), sizeof(int));

    /* Track operation counts via parallel arrays */
    int *prev_s = (int *)calloc((size_t)(m + 1), sizeof(int));
    int *prev_d = (int *)calloc((size_t)(m + 1), sizeof(int));
    int *prev_i = (int *)calloc((size_t)(m + 1), sizeof(int));
    int *curr_s = (int *)calloc((size_t)(m + 1), sizeof(int));
    int *curr_d = (int *)calloc((size_t)(m + 1), sizeof(int));
    int *curr_i = (int *)calloc((size_t)(m + 1), sizeof(int));

    /* Base case: inserting all hypothesis words */
    for (int j = 0; j <= m; j++) {
        prev[j] = j;
        prev_i[j] = j;
    }

    for (int i = 1; i <= ref_n; i++) {
        curr[0] = i;
        curr_s[0] = 0;
        curr_d[0] = i;
        curr_i[0] = 0;

        for (int j = 1; j <= m; j++) {
            int cost = words_equal(ref[i - 1], hyp[j - 1]) ? 0 : 1;

            int sub_cost = prev[j - 1] + cost;
            int del_cost = prev[j] + 1;
            int ins_cost = curr[j - 1] + 1;

            if (sub_cost <= del_cost && sub_cost <= ins_cost) {
                curr[j] = sub_cost;
                curr_s[j] = prev_s[j - 1] + cost;
                curr_d[j] = prev_d[j - 1];
                curr_i[j] = prev_i[j - 1];
            } else if (del_cost <= ins_cost) {
                curr[j] = del_cost;
                curr_s[j] = prev_s[j];
                curr_d[j] = prev_d[j] + 1;
                curr_i[j] = prev_i[j];
            } else {
                curr[j] = ins_cost;
                curr_s[j] = curr_s[j - 1];
                curr_d[j] = curr_d[j - 1];
                curr_i[j] = curr_i[j - 1] + 1;
            }
        }

        int *tmp;
        tmp = prev; prev = curr; curr = tmp;
        tmp = prev_s; prev_s = curr_s; curr_s = tmp;
        tmp = prev_d; prev_d = curr_d; curr_d = tmp;
        tmp = prev_i; prev_i = curr_i; curr_i = tmp;
    }

    *out_sub = prev_s[m];
    *out_del = prev_d[m];
    *out_ins = prev_i[m];

    free(prev); free(curr);
    free(prev_s); free(curr_s);
    free(prev_d); free(curr_d);
    free(prev_i); free(curr_i);
}

/* ── Levenshtein Distance (character-level) ───────────── */

static int levenshtein_chars(const char *a, int la, const char *b, int lb)
{
    int *prev = (int *)calloc((size_t)(lb + 1), sizeof(int));
    int *curr = (int *)calloc((size_t)(lb + 1), sizeof(int));

    for (int j = 0; j <= lb; j++) prev[j] = j;

    for (int i = 1; i <= la; i++) {
        curr[0] = i;
        for (int j = 1; j <= lb; j++) {
            int cost = (a[i - 1] == b[j - 1]) ? 0 : 1;
            int s = prev[j - 1] + cost;
            int d = prev[j] + 1;
            int ins = curr[j - 1] + 1;
            curr[j] = s < d ? (s < ins ? s : ins) : (d < ins ? d : ins);
        }
        int *tmp = prev; prev = curr; curr = tmp;
    }

    int result = prev[lb];
    free(prev);
    free(curr);
    return result;
}

/* ── Public API ───────────────────────────────────────── */

WERResult wer_compute(const char *reference, const char *hypothesis)
{
    WERResult r = {0};
    if (!reference || !hypothesis) {
        r.wer = 1.0f;
        return r;
    }

    /* Normalize both texts */
    char ref_norm[MAX_CHARS], hyp_norm[MAX_CHARS];
    wer_normalize(reference, ref_norm, MAX_CHARS);
    wer_normalize(hypothesis, hyp_norm, MAX_CHARS);

    /* Tokenize into words */
    const char *ref_words[MAX_WORDS], *hyp_words[MAX_WORDS];
    int ref_n = tokenize(ref_norm, ref_words, MAX_WORDS);
    int hyp_n = tokenize(hyp_norm, hyp_words, MAX_WORDS);

    r.ref_words = ref_n;
    r.hyp_words = hyp_n;

    if (ref_n == 0) {
        r.insertions = hyp_n;
        r.wer = hyp_n > 0 ? (float)hyp_n : 0.0f;
        r.accuracy = hyp_n > 0 ? 0.0f : 1.0f;
        return r;
    }

    /* Compute word-level Levenshtein with S/D/I breakdown */
    levenshtein_words(ref_words, ref_n, hyp_words, hyp_n,
                      &r.substitutions, &r.deletions, &r.insertions);

    int total_errors = r.substitutions + r.deletions + r.insertions;
    r.wer = (float)total_errors / (float)ref_n;
    r.accuracy = 1.0f - r.wer;
    if (r.accuracy < 0.0f) r.accuracy = 0.0f;

    /* Also compute CER */
    r.cer = cer_compute(ref_norm, hyp_norm);

    return r;
}

float cer_compute(const char *reference, const char *hypothesis)
{
    if (!reference || !hypothesis) return 1.0f;

    char ref_norm[MAX_CHARS], hyp_norm[MAX_CHARS];
    wer_normalize(reference, ref_norm, MAX_CHARS);
    wer_normalize(hypothesis, hyp_norm, MAX_CHARS);

    int ref_len = (int)strlen(ref_norm);
    int hyp_len = (int)strlen(hyp_norm);

    if (ref_len == 0) return hyp_len > 0 ? (float)hyp_len : 0.0f;

    int dist = levenshtein_chars(ref_norm, ref_len, hyp_norm, hyp_len);
    return (float)dist / (float)ref_len;
}
