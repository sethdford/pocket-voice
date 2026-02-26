/**
 * spm_tokenizer.c — Minimal SentencePiece unigram tokenizer in C.
 *
 * Implements the core Viterbi-based unigram model tokenization.
 * Parses the SentencePiece protobuf to extract vocabulary pieces and scores,
 * then tokenizes via longest-match Viterbi dynamic programming.
 *
 * Build: cc -O3 -shared -fPIC -o libspm_tokenizer.dylib spm_tokenizer.c
 */

#include "spm_tokenizer.h"
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_PIECE_LEN 128
#define UNK_ID 0

/* Unicode replacement character for SentencePiece space marker */
#define SP_SPACE_MARKER "\xe2\x96\x81"  /* U+2581 LOWER ONE EIGHTH BLOCK */
#define SP_SPACE_MARKER_LEN 3

/* ── Trie for fast prefix lookup ─────────────────────────────────── */

#define TRIE_CHILDREN 256

typedef struct TrieNode {
    struct TrieNode *children[TRIE_CHILDREN];
    int token_id;   /* -1 if not a terminal */
    float score;
} TrieNode;

static TrieNode *trie_new(void) {
    TrieNode *n = (TrieNode *)calloc(1, sizeof(TrieNode));
    if (!n) return NULL;
    n->token_id = -1;
    n->score = -INFINITY;
    return n;
}

static void trie_insert(TrieNode *root, const char *piece, int piece_len,
                         int token_id, float score) {
    TrieNode *cur = root;
    for (int i = 0; i < piece_len; i++) {
        unsigned char c = (unsigned char)piece[i];
        if (!cur->children[c]) {
            cur->children[c] = trie_new();
            if (!cur->children[c]) return;
        }
        cur = cur->children[c];
    }
    cur->token_id = token_id;
    cur->score = score;
}

static void trie_free(TrieNode *node) {
    if (!node) return;
    for (int i = 0; i < TRIE_CHILDREN; i++) {
        trie_free(node->children[i]);
    }
    free(node);
}

/* ── Tokenizer struct ────────────────────────────────────────────── */

struct SPMTokenizer {
    char **pieces;     /* piece[i] = the string for token i */
    int  *piece_lens;  /* byte length of each piece */
    float *scores;     /* log probability scores */
    int n_pieces;
    TrieNode *trie;
};

/* ── Protobuf parser for SentencePiece .model ────────────────────── */

/* SentencePiece protobuf schema (simplified):
 *   ModelProto {
 *     repeated SentencePiece pieces = 1;
 *     ...
 *   }
 *   SentencePiece {
 *     optional string piece = 1;
 *     optional float score = 2;
 *     optional Type type = 3;
 *   }
 */

static uint64_t read_varint(const uint8_t **p, const uint8_t *end) {
    uint64_t result = 0;
    int shift = 0;
    while (*p < end) {
        uint8_t byte = **p;
        (*p)++;
        result |= (uint64_t)(byte & 0x7F) << shift;
        if (!(byte & 0x80)) break;
        shift += 7;
    }
    return result;
}

static int parse_sentencepiece_model(const uint8_t *data, uint32_t size,
                                      char ***out_pieces, float **out_scores,
                                      int **out_lens, int *out_count) {
    const uint8_t *p = data;
    const uint8_t *end = data + size;

    int capacity = 8192;
    char **pieces = (char **)calloc(capacity, sizeof(char *));
    float *scores = (float *)calloc(capacity, sizeof(float));
    int *lens = (int *)calloc(capacity, sizeof(int));
    int count = 0;

    if (!pieces || !scores || !lens) {
        free(pieces); free(scores); free(lens);
        *out_count = 0;
        return -1;
    }

    while (p < end) {
        uint64_t tag = read_varint(&p, end);
        int field = (int)(tag >> 3);
        int wire = (int)(tag & 7);

        if (field == 1 && wire == 2) {
            /* ModelProto.pieces — length-delimited submessage */
            uint64_t sub_len = read_varint(&p, end);
            const uint8_t *sub_end = p + sub_len;
            if (sub_end > end) break;

            char piece[MAX_PIECE_LEN] = {0};
            int piece_len = 0;
            float score = 0.0f;

            while (p < sub_end) {
                uint64_t stag = read_varint(&p, sub_end);
                int sfield = (int)(stag >> 3);
                int swire = (int)(stag & 7);

                if (sfield == 1 && swire == 2) {
                    /* piece string */
                    uint64_t str_len = read_varint(&p, sub_end);
                    if (p + str_len > sub_end) break;
                    if (str_len < MAX_PIECE_LEN) {
                        memcpy(piece, p, str_len);
                        piece[str_len] = '\0';
                        piece_len = (int)str_len;
                    }
                    p += str_len;
                } else if (sfield == 2 && swire == 5) {
                    /* score (fixed32 float) */
                    if (p + 4 <= sub_end) {
                        memcpy(&score, p, 4);
                        p += 4;
                    }
                } else if (swire == 0) {
                    read_varint(&p, sub_end);
                } else if (swire == 2) {
                    uint64_t skip = read_varint(&p, sub_end);
                    if (p + skip > sub_end) break;
                    p += skip;
                } else if (swire == 5) {
                    p += 4;
                } else if (swire == 1) {
                    p += 8;
                } else {
                    break;
                }
            }
            p = sub_end;

            if (count >= capacity) {
                int new_cap = capacity * 2;
                char **new_pieces = (char **)realloc(pieces, new_cap * sizeof(char *));
                float *new_scores = (float *)realloc(scores, new_cap * sizeof(float));
                int *new_lens = (int *)realloc(lens, new_cap * sizeof(int));
                if (!new_pieces || !new_scores || !new_lens) {
                    if (new_pieces) pieces = new_pieces;
                    if (new_scores) scores = new_scores;
                    if (new_lens) lens = new_lens;
                    break;
                }
                pieces = new_pieces;
                scores = new_scores;
                lens = new_lens;
                capacity = new_cap;
            }
            pieces[count] = (char *)malloc(piece_len + 1);
            if (!pieces[count]) break;
            memcpy(pieces[count], piece, piece_len + 1);
            lens[count] = piece_len;
            scores[count] = score;
            count++;
        } else {
            /* Skip other fields */
            if (wire == 0) {
                read_varint(&p, end);
            } else if (wire == 2) {
                uint64_t skip = read_varint(&p, end);
                p += skip;
            } else if (wire == 5) {
                p += 4;
            } else if (wire == 1) {
                p += 8;
            } else {
                break;
            }
        }
    }

    *out_pieces = pieces;
    *out_scores = scores;
    *out_lens = lens;
    *out_count = count;
    return 0;
}

/* ── Constructor / Destructor ────────────────────────────────────── */

SPMTokenizer *spm_create(const uint8_t *model_data, uint32_t model_size) {
    char **pieces = NULL;
    float *scores = NULL;
    int *lens = NULL;
    int count = 0;

    if (parse_sentencepiece_model(model_data, model_size,
                                   &pieces, &scores, &lens, &count) != 0) {
        return NULL;
    }

    SPMTokenizer *tok = (SPMTokenizer *)calloc(1, sizeof(SPMTokenizer));
    if (!tok) {
        for (int i = 0; i < count; i++) free(pieces[i]);
        free(pieces); free(scores); free(lens);
        return NULL;
    }
    tok->pieces = pieces;
    tok->scores = scores;
    tok->piece_lens = lens;
    tok->n_pieces = count;

    /* Build trie */
    tok->trie = trie_new();
    if (!tok->trie) { spm_destroy(tok); return NULL; }
    for (int i = 0; i < count; i++) {
        if (lens[i] > 0) {
            trie_insert(tok->trie, pieces[i], lens[i], i, scores[i]);
        }
    }

    fprintf(stderr, "[spm] Loaded %d pieces\n", count);
    return tok;
}

SPMTokenizer *spm_create_from_vocab(const char **pieces, const float *scores,
                                     int n_pieces) {
    if (!pieces || !scores || n_pieces <= 0) return NULL;

    SPMTokenizer *tok = (SPMTokenizer *)calloc(1, sizeof(SPMTokenizer));
    if (!tok) return NULL;
    tok->n_pieces = n_pieces;
    tok->pieces = (char **)calloc(n_pieces, sizeof(char *));
    tok->piece_lens = (int *)calloc(n_pieces, sizeof(int));
    tok->scores = (float *)calloc(n_pieces, sizeof(float));
    if (!tok->pieces || !tok->piece_lens || !tok->scores) {
        spm_destroy(tok);
        return NULL;
    }

    tok->trie = trie_new();
    if (!tok->trie) { spm_destroy(tok); return NULL; }
    for (int i = 0; i < n_pieces; i++) {
        int len = (int)strlen(pieces[i]);
        tok->pieces[i] = (char *)malloc(len + 1);
        if (!tok->pieces[i]) { spm_destroy(tok); return NULL; }
        memcpy(tok->pieces[i], pieces[i], len + 1);
        tok->piece_lens[i] = len;
        tok->scores[i] = scores[i];
        if (len > 0) {
            trie_insert(tok->trie, pieces[i], len, i, scores[i]);
        }
    }
    return tok;
}

void spm_destroy(SPMTokenizer *tok) {
    if (!tok) return;
    trie_free(tok->trie);
    for (int i = 0; i < tok->n_pieces; i++) {
        free(tok->pieces[i]);
    }
    free(tok->pieces);
    free(tok->piece_lens);
    free(tok->scores);
    free(tok);
}

int spm_vocab_size(const SPMTokenizer *tok) {
    return tok ? tok->n_pieces : 0;
}

/* ── Viterbi tokenization ────────────────────────────────────────── */

int spm_encode(const SPMTokenizer *tok, const char *text,
               int32_t *out_ids, int max_ids) {
    if (!tok || !text || !out_ids) return -1;

    /* Prepend space marker (SentencePiece convention: leading space -> U+2581) */
    int text_len = (int)strlen(text);
    int buf_len = text_len + SP_SPACE_MARKER_LEN + 1;
    char *buf = (char *)malloc(buf_len);
    if (!buf) return -1;
    int pos = 0;

    /* Replace spaces with the SP marker and prepend one */
    memcpy(buf, SP_SPACE_MARKER, SP_SPACE_MARKER_LEN);
    pos = SP_SPACE_MARKER_LEN;
    for (int i = 0; i < text_len; i++) {
        if (text[i] == ' ') {
            /* Ensure enough space */
            if (pos + SP_SPACE_MARKER_LEN >= buf_len) {
                buf_len = pos + SP_SPACE_MARKER_LEN + text_len - i + 1;
                char *new_buf = (char *)realloc(buf, buf_len);
                if (!new_buf) { free(buf); return -1; }
                buf = new_buf;
            }
            memcpy(buf + pos, SP_SPACE_MARKER, SP_SPACE_MARKER_LEN);
            pos += SP_SPACE_MARKER_LEN;
        } else {
            if (pos >= buf_len - 1) {
                buf_len = pos + text_len - i + 1;
                char *new_buf = (char *)realloc(buf, buf_len);
                if (!new_buf) { free(buf); return -1; }
                buf = new_buf;
            }
            buf[pos++] = text[i];
        }
    }
    buf[pos] = '\0';
    int n = pos;

    /* Viterbi forward pass */
    float *best_score = (float *)malloc((n + 1) * sizeof(float));
    int *best_len = (int *)malloc((n + 1) * sizeof(int));
    int *best_id = (int *)malloc((n + 1) * sizeof(int));
    if (!best_score || !best_len || !best_id) {
        free(buf); free(best_score); free(best_len); free(best_id);
        return -1;
    }

    best_score[0] = 0.0f;
    for (int i = 1; i <= n; i++) {
        best_score[i] = -INFINITY;
        best_len[i] = 1;
        best_id[i] = UNK_ID;
    }

    for (int i = 0; i < n; i++) {
        if (best_score[i] == -INFINITY && i > 0) continue;

        TrieNode *cur = tok->trie;
        for (int j = i; j < n && cur; j++) {
            unsigned char c = (unsigned char)buf[j];
            cur = cur->children[c];
            if (!cur) break;
            if (cur->token_id >= 0) {
                int piece_len = j - i + 1;
                float candidate = best_score[i] + cur->score;
                if (candidate > best_score[j + 1]) {
                    best_score[j + 1] = candidate;
                    best_len[j + 1] = piece_len;
                    best_id[j + 1] = cur->token_id;
                }
            }
        }

        /* Fall back to single-byte UNK if no trie match at all */
        if (best_score[i + 1] == -INFINITY) {
            best_score[i + 1] = best_score[i] - 100.0f;
            best_len[i + 1] = 1;
            best_id[i + 1] = UNK_ID;
        }
    }

    /* Backtrace */
    int n_tokens = 0;
    int *token_starts = (int *)malloc(n * sizeof(int));
    int *token_ids = (int *)malloc(n * sizeof(int));
    if (!token_starts || !token_ids) {
        free(token_starts); free(token_ids);
        free(best_score); free(best_len); free(best_id); free(buf);
        return -1;
    }

    int p2 = n;
    while (p2 > 0) {
        token_starts[n_tokens] = p2 - best_len[p2];
        token_ids[n_tokens] = best_id[p2];
        p2 -= best_len[p2];
        n_tokens++;
    }

    /* Reverse and write output */
    int out_count = n_tokens < max_ids ? n_tokens : max_ids;
    for (int i = 0; i < out_count; i++) {
        out_ids[i] = token_ids[n_tokens - 1 - i];
    }

    free(token_starts);
    free(token_ids);
    free(best_score);
    free(best_len);
    free(best_id);
    free(buf);

    return out_count;
}

int spm_decode(const SPMTokenizer *tok, const int32_t *ids, int n_ids,
               char *out_text, int out_cap) {
    if (!tok || !ids || !out_text) return -1;

    int pos = 0;
    for (int i = 0; i < n_ids; i++) {
        int id = ids[i];
        if (id < 0 || id >= tok->n_pieces) continue;

        const char *piece = tok->pieces[id];
        int plen = tok->piece_lens[id];

        for (int j = 0; j < plen; j++) {
            /* Replace SP space marker with actual space */
            if (j + SP_SPACE_MARKER_LEN - 1 < plen &&
                memcmp(piece + j, SP_SPACE_MARKER, SP_SPACE_MARKER_LEN) == 0) {
                if (pos < out_cap - 1) out_text[pos++] = ' ';
                j += SP_SPACE_MARKER_LEN - 1;
            } else {
                if (pos < out_cap - 1) out_text[pos++] = piece[j];
            }
        }
    }

    /* Trim leading space */
    if (pos > 0 && out_text[0] == ' ') {
        memmove(out_text, out_text + 1, pos - 1);
        pos--;
    }
    out_text[pos] = '\0';
    return pos;
}
