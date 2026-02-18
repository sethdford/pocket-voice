/**
 * sentence_buffer.c — Sentence buffer for streaming LLM token accumulation.
 *
 * Port of pocket_tts/voice/sentence_buffer.py. Accumulates text tokens from
 * streaming LLM responses and flushes at sentence boundaries or clause
 * boundaries (speculative mode). Filters out code blocks and markdown.
 *
 * Internal layout:
 *   - Fixed 8KB accumulation buffer (no realloc in hot path)
 *   - 32-slot segment ring for pending ready segments
 *   - Code-block tracking (triple backticks)
 *   - Markdown stripping via hand-written scanners
 *
 * Build: cc -O3 -shared -fPIC -o libsentence_buffer.dylib sentence_buffer.c
 */

#include "sentence_buffer.h"
#include <ctype.h>
#include <stdlib.h>
#include <string.h>

#define BUF_SIZE       8192
#define MAX_SEGMENTS    32
#define SEG_SIZE       2048

struct SentenceBuffer {
    /* Accumulation buffer */
    char buf[BUF_SIZE];
    int  buf_len;

    /* Pending segments ring */
    char segments[MAX_SEGMENTS][SEG_SIZE];
    int  seg_lens[MAX_SEGMENTS];
    int  seg_head;        /* Next slot to write */
    int  seg_tail;        /* Next slot to read */
    int  seg_count;

    /* State */
    int  in_code_block;   /* 1 = inside ``` ... ``` */
    int  mode;            /* SENTBUF_MODE_SENTENCE or SENTBUF_MODE_SPECULATIVE */
    int  min_words;

    /* Predictive sentence boundary tracking */
    float ema_length;      /* EMA of sentence length in chars (α=0.3) */
    int   total_flushed;   /* Total sentences flushed this turn */

    /* Adaptive warmup: aggressive flushing for first N sentences */
    int   warmup_n;        /* Number of warmup sentences (0 = disabled) */
    int   warmup_min_words; /* Min words during warmup phase */
    int   base_min_words;  /* Original min_words (restored after warmup) */
};

/* ── helpers ─────────────────────────────────────────────────────────────── */

static int count_words(const char *s, int len) {
    int words = 0, in_word = 0;
    for (int i = 0; i < len; i++) {
        if (s[i] == ' ' || s[i] == '\t' || s[i] == '\n' || s[i] == '\r') {
            in_word = 0;
        } else if (!in_word) {
            words++;
            in_word = 1;
        }
    }
    return words;
}

static int count_fences(const char *s, int len) {
    int count = 0;
    for (int i = 0; i + 2 < len; i++) {
        if (s[i] == '`' && s[i+1] == '`' && s[i+2] == '`') {
            count++;
            i += 2;
        }
    }
    /* Check exact end match (i + 2 == len) */
    if (len >= 3 && s[len-3] == '`' && s[len-2] == '`' && s[len-1] == '`') {
        /* Already counted if was within loop bounds — only count if loop stopped short */
    }
    return count;
}

/**
 * Clean markdown from a text segment for TTS speech.
 * Strips: triple backticks, inline `code`, **bold**, *italic*,
 *         [link](url), # headings, and collapses whitespace.
 */
static int clean_for_speech(const char *in, int in_len, char *out, int out_cap) {
    int oi = 0;
    int i = 0;

    #define EMIT(c) do { if (oi < out_cap - 1) out[oi++] = (c); } while(0)

    while (i < in_len) {
        /* Triple backticks — skip */
        if (i + 2 < in_len && in[i] == '`' && in[i+1] == '`' && in[i+2] == '`') {
            i += 3;
            continue;
        }

        /* Inline code `...` — skip content */
        if (in[i] == '`') {
            i++;
            while (i < in_len && in[i] != '`') i++;
            if (i < in_len) i++;  /* skip closing backtick */
            continue;
        }

        /* Bold **...** — keep content */
        if (i + 1 < in_len && in[i] == '*' && in[i+1] == '*') {
            i += 2;
            while (i + 1 < in_len && !(in[i] == '*' && in[i+1] == '*')) {
                EMIT(in[i]);
                i++;
            }
            if (i + 1 < in_len) i += 2;  /* skip closing ** */
            continue;
        }

        /* Italic *...* — keep content */
        if (in[i] == '*' && (i + 1 < in_len && in[i+1] != '*')) {
            i++;
            while (i < in_len && in[i] != '*') {
                EMIT(in[i]);
                i++;
            }
            if (i < in_len) i++;  /* skip closing * */
            continue;
        }

        /* Markdown link [text](url) — keep text */
        if (in[i] == '[') {
            int j = i + 1;
            while (j < in_len && in[j] != ']') j++;
            if (j < in_len && j + 1 < in_len && in[j+1] == '(') {
                /* Emit link text */
                for (int k = i + 1; k < j; k++) EMIT(in[k]);
                /* Skip ](url) */
                j += 2;
                while (j < in_len && in[j] != ')') j++;
                if (j < in_len) j++;
                i = j;
                continue;
            }
        }

        /* Heading: # at start of line — skip # chars and space */
        if (in[i] == '#') {
            int is_heading = (i == 0 || in[i-1] == '\n');
            if (is_heading) {
                while (i < in_len && in[i] == '#') i++;
                while (i < in_len && in[i] == ' ') i++;
                continue;
            }
        }

        /* Collapse whitespace */
        if (in[i] == ' ' || in[i] == '\t' || in[i] == '\n' || in[i] == '\r') {
            if (oi > 0 && out[oi-1] != ' ') EMIT(' ');
            i++;
            continue;
        }

        EMIT(in[i]);
        i++;
    }

    #undef EMIT

    /* Trim trailing space */
    while (oi > 0 && out[oi-1] == ' ') oi--;
    /* Trim leading space */
    int start = 0;
    while (start < oi && out[start] == ' ') start++;
    if (start > 0) {
        memmove(out, out + start, (size_t)(oi - start));
        oi -= start;
    }

    out[oi] = '\0';
    return oi;
}

static void push_segment(SentenceBuffer *sb, const char *text, int len) {
    if (len <= 0 || sb->seg_count >= MAX_SEGMENTS) return;

    char cleaned[SEG_SIZE];
    int clen = clean_for_speech(text, len, cleaned, SEG_SIZE);
    if (clen <= 0) return;

    int slot = sb->seg_head;
    int copy = clen < SEG_SIZE - 1 ? clen : SEG_SIZE - 1;
    memcpy(sb->segments[slot], cleaned, (size_t)copy);
    sb->segments[slot][copy] = '\0';
    sb->seg_lens[slot] = copy;
    sb->seg_head = (sb->seg_head + 1) % MAX_SEGMENTS;
    sb->seg_count++;

    /* Update predictive sentence length (EMA with α=0.3) */
    if (sb->ema_length <= 0.0f) {
        sb->ema_length = (float)clen;
    } else {
        sb->ema_length = 0.3f * (float)clen + 0.7f * sb->ema_length;
    }
    sb->total_flushed++;

    /* Adaptive warmup: restore normal min_words after warmup phase */
    if (sb->warmup_n > 0 && sb->total_flushed >= sb->warmup_n) {
        sb->min_words = sb->base_min_words;
    }
}

/* Check for sentence-ending pattern at position i in buffer */
static int is_sentence_end(const char *buf, int len, int i) {
    if (i < 0 || i >= len) return 0;
    char ch = buf[i];

    /* Newline is always a boundary */
    if (ch == '\n') return 1;

    /* Sentence-ending punctuation followed by space or end */
    if (ch == '.' || ch == '!' || ch == '?') {
        if (i + 1 >= len) return 1;   /* at end of buffer */
        if (buf[i+1] == ' ' || buf[i+1] == '\n' || buf[i+1] == '\t') return 1;
    }

    return 0;
}

/* Check for clause-ending pattern */
static int is_clause_end(const char *buf, int len, int i) {
    if (i < 0 || i >= len) return 0;
    unsigned char ch = (unsigned char)buf[i];

    /* ASCII clause separators: , ; : */
    if ((ch == ',' || ch == ';' || ch == ':') && i + 1 < len &&
        (buf[i+1] == ' ' || buf[i+1] == '\n')) return 1;

    /* Em-dash (UTF-8: E2 80 94) or en-dash (E2 80 93) */
    if (ch == 0xE2 && i + 2 < len) {
        unsigned char b1 = (unsigned char)buf[i+1];
        unsigned char b2 = (unsigned char)buf[i+2];
        if (b1 == 0x80 && (b2 == 0x94 || b2 == 0x93)) {
            if (i + 3 < len && (buf[i+3] == ' ' || buf[i+3] == '\n')) return 1;
        }
    }

    return 0;
}

static int boundary_len(const char *buf, int i) {
    /* Return how many characters the boundary delimiter occupies */
    unsigned char ch = (unsigned char)buf[i];
    if (ch == '\n') return 1;
    if (ch == '.' || ch == '!' || ch == '?') return 2;  /* punct + space */
    if (ch == ',' || ch == ';' || ch == ':') return 2;
    if (ch == 0xE2) return 4;  /* em/en-dash (3 bytes) + space */
    return 1;
}

static void try_flush(SentenceBuffer *sb) {
    char *buf = sb->buf;
    int len = sb->buf_len;

    /* Find sentence boundary */
    for (int i = 0; i < len; i++) {
        if (is_sentence_end(buf, len, i)) {
            int end = i + 1;  /* include the punctuation */
            if (buf[i] != '\n' && end < len &&
                (buf[end] == ' ' || buf[end] == '\n')) end++;

            push_segment(sb, buf, i + 1);

            /* Shift remainder */
            int remain = len - end;
            if (remain > 0) memmove(buf, buf + end, (size_t)remain);
            sb->buf_len = remain;
            buf[remain] = '\0';
            return;
        }
    }

    /* In speculative mode, also check clause boundaries */
    if (sb->mode == SENTBUF_MODE_SPECULATIVE) {
        int words = count_words(buf, len);
        if (words >= sb->min_words) {
            for (int i = 0; i < len; i++) {
                if (is_clause_end(buf, len, i)) {
                    int blen = boundary_len(buf, i);
                    int end = i + blen;
                    if (end > len) end = len;

                    push_segment(sb, buf, i + 1);

                    int remain = len - end;
                    if (remain > 0) memmove(buf, buf + end, (size_t)remain);
                    sb->buf_len = remain;
                    buf[remain] = '\0';
                    return;
                }
            }
        }
    }
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Public API
 * ═══════════════════════════════════════════════════════════════════════════ */

SentenceBuffer *sentbuf_create(int mode, int min_words) {
    SentenceBuffer *sb = (SentenceBuffer *)calloc(1, sizeof(SentenceBuffer));
    if (!sb) return NULL;
    sb->mode = mode;
    sb->min_words = min_words > 0 ? min_words : 5;
    return sb;
}

void sentbuf_destroy(SentenceBuffer *sb) {
    free(sb);
}

void sentbuf_add(SentenceBuffer *sb, const char *text, int len) {
    if (!sb || !text || len <= 0) return;

    /* Append to buffer */
    int space = BUF_SIZE - sb->buf_len - 1;
    int to_copy = len < space ? len : space;
    if (to_copy > 0) {
        memcpy(sb->buf + sb->buf_len, text, (size_t)to_copy);
        sb->buf_len += to_copy;
        sb->buf[sb->buf_len] = '\0';
    }

    /* Update code block tracking */
    int fences = count_fences(sb->buf, sb->buf_len);
    sb->in_code_block = (fences % 2 == 1) ? 1 : 0;

    if (sb->in_code_block) return;

    try_flush(sb);
}

int sentbuf_has_segment(const SentenceBuffer *sb) {
    return sb && sb->seg_count > 0;
}

int sentbuf_flush(SentenceBuffer *sb, char *out, int out_cap) {
    if (!sb || sb->seg_count <= 0 || !out || out_cap <= 0) {
        if (out && out_cap > 0) out[0] = '\0';
        return 0;
    }

    int slot = sb->seg_tail;
    int len = sb->seg_lens[slot];
    int copy = len < out_cap - 1 ? len : out_cap - 1;
    memcpy(out, sb->segments[slot], (size_t)copy);
    out[copy] = '\0';

    sb->seg_tail = (sb->seg_tail + 1) % MAX_SEGMENTS;
    sb->seg_count--;

    return copy;
}

int sentbuf_flush_all(SentenceBuffer *sb, char *out, int out_cap) {
    if (!sb || !out || out_cap <= 0) {
        if (out && out_cap > 0) out[0] = '\0';
        return 0;
    }

    /* Push any remaining buffer content */
    if (sb->buf_len > 0 && !sb->in_code_block) {
        push_segment(sb, sb->buf, sb->buf_len);
        sb->buf_len = 0;
        sb->buf[0] = '\0';
    }

    /* Join all pending segments */
    int pos = 0;
    while (sb->seg_count > 0) {
        int slot = sb->seg_tail;
        int len = sb->seg_lens[slot];
        if (pos > 0 && pos < out_cap - 1) {
            out[pos++] = ' ';
        }
        int avail = out_cap - pos - 1;
        int copy = len < avail ? len : avail;
        if (copy > 0) {
            memcpy(out + pos, sb->segments[slot], (size_t)copy);
            pos += copy;
        }
        sb->seg_tail = (sb->seg_tail + 1) % MAX_SEGMENTS;
        sb->seg_count--;
    }

    out[pos] = '\0';
    return pos;
}

void sentbuf_reset(SentenceBuffer *sb) {
    if (!sb) return;
    sb->buf_len = 0;
    sb->buf[0] = '\0';
    sb->in_code_block = 0;
    sb->seg_head = 0;
    sb->seg_tail = 0;
    sb->seg_count = 0;
    sb->ema_length = 0.0f;
    sb->total_flushed = 0;

    /* Re-arm adaptive warmup */
    if (sb->warmup_n > 0) {
        sb->min_words = sb->warmup_min_words;
    }
}

int sentbuf_predicted_length(const SentenceBuffer *sb) {
    if (!sb || sb->ema_length <= 0.0f) return 0;
    return (int)(sb->ema_length + 0.5f);
}

int sentbuf_sentence_count(const SentenceBuffer *sb) {
    return sb ? sb->total_flushed : 0;
}

void sentbuf_set_adaptive(SentenceBuffer *sb, int warmup_n, int warmup_min) {
    if (!sb) return;
    sb->warmup_n = warmup_n > 0 ? warmup_n : 2;
    sb->warmup_min_words = warmup_min > 0 ? warmup_min : 3;
    sb->base_min_words = sb->min_words;
    sb->min_words = sb->warmup_min_words;
}
