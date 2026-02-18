/**
 * ssml_parser.c — Lightweight SSML parser for the native voice pipeline.
 *
 * Port of pocket_tts/ssml/parser.py. Hand-written XML tokenizer — no libxml2
 * dependency. Handles: open tags with attributes, close tags, self-closing tags,
 * text content, entity references (&amp; etc). Under 500 lines.
 *
 * Integrates with text_normalize.c for <say-as> tag handling.
 *
 * Build: cc -O3 -shared -fPIC -o libssml_parser.dylib ssml_parser.c text_normalize.c
 */

#include "ssml_parser.h"
#include "text_normalize.h"
#include <ctype.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ═══════════════════════════════════════════════════════════════════════════
 * Constants
 * ═══════════════════════════════════════════════════════════════════════════ */

#define PARAGRAPH_BREAK_MS  500
#define SENTENCE_BREAK_MS   300
#define MAX_NEST_DEPTH       16
#define MAX_ATTRS            16
#define ATTR_KEY_MAX         64
#define ATTR_VAL_MAX        512

/* ═══════════════════════════════════════════════════════════════════════════
 * Prosody parsers (also exported for pipeline use)
 * ═══════════════════════════════════════════════════════════════════════════ */

static int streqi(const char *a, const char *b) {
    while (*a && *b) {
        if (tolower((unsigned char)*a) != tolower((unsigned char)*b)) return 0;
        a++; b++;
    }
    return *a == *b;
}

static int str_ends_with_ci(const char *s, const char *suffix) {
    int slen = (int)strlen(s), suflen = (int)strlen(suffix);
    if (suflen > slen) return 0;
    return streqi(s + slen - suflen, suffix);
}

float ssml_parse_rate(const char *value) {
    if (!value || !*value) return 1.0f;
    char lower[64];
    int i;
    for (i = 0; value[i] && i < 63; i++) lower[i] = (char)tolower((unsigned char)value[i]);
    lower[i] = '\0';

    if (streqi(lower, "x-slow"))  return 0.5f;
    if (streqi(lower, "slow"))    return 0.75f;
    if (streqi(lower, "medium"))  return 1.0f;
    if (streqi(lower, "fast"))    return 1.25f;
    if (streqi(lower, "x-fast"))  return 1.75f;
    if (streqi(lower, "default")) return 1.0f;

    if (str_ends_with_ci(lower, "%")) {
        return (float)atof(lower) / 100.0f;
    }
    float v = (float)atof(lower);
    return v > 0.0f ? v : 1.0f;
}

float ssml_parse_pitch(const char *value) {
    if (!value || !*value) return 1.0f;
    char lower[64];
    int i;
    for (i = 0; value[i] && i < 63; i++) lower[i] = (char)tolower((unsigned char)value[i]);
    lower[i] = '\0';

    if (streqi(lower, "x-low"))   return 0.5f;
    if (streqi(lower, "low"))     return 0.75f;
    if (streqi(lower, "medium"))  return 1.0f;
    if (streqi(lower, "high"))    return 1.25f;
    if (streqi(lower, "x-high"))  return 1.5f;
    if (streqi(lower, "default")) return 1.0f;

    if (str_ends_with_ci(lower, "%")) {
        const char *s = lower;
        if (*s == '+') return 1.0f + (float)atof(s + 1) / 100.0f;
        if (*s == '-') return 1.0f - (float)atof(s + 1) / 100.0f;
        return (float)atof(s) / 100.0f;
    }
    if (str_ends_with_ci(lower, "st")) {
        float st = (float)atof(lower);
        return powf(2.0f, st / 12.0f);
    }
    if (str_ends_with_ci(lower, "hz")) {
        float hz = (float)atof(lower);
        return hz / 200.0f;
    }
    float v = (float)atof(lower);
    return v > 0.0f ? v : 1.0f;
}

float ssml_parse_volume(const char *value) {
    if (!value || !*value) return 1.0f;
    char lower[64];
    int i;
    for (i = 0; value[i] && i < 63; i++) lower[i] = (char)tolower((unsigned char)value[i]);
    lower[i] = '\0';

    if (streqi(lower, "silent"))  return 0.0f;
    if (streqi(lower, "x-soft")) return 0.25f;
    if (streqi(lower, "soft"))   return 0.5f;
    if (streqi(lower, "medium")) return 1.0f;
    if (streqi(lower, "loud"))   return 1.5f;
    if (streqi(lower, "x-loud")) return 2.0f;
    if (streqi(lower, "default"))return 1.0f;

    if (str_ends_with_ci(lower, "db")) {
        float db = (float)atof(lower);
        return powf(10.0f, db / 20.0f);
    }
    if (str_ends_with_ci(lower, "%")) {
        return (float)atof(lower) / 100.0f;
    }
    float v = (float)atof(lower);
    return v > 0.0f ? v : 1.0f;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Minimal XML Tokenizer
 * ═══════════════════════════════════════════════════════════════════════════ */

typedef struct {
    char key[ATTR_KEY_MAX];
    char val[ATTR_VAL_MAX];
} Attr;

typedef enum {
    XML_OPEN_TAG,
    XML_CLOSE_TAG,
    XML_SELF_CLOSE,
    XML_TEXT,
    XML_EOF,
} XmlTokenType;

typedef struct {
    XmlTokenType type;
    char tag[128];          /* tag name (for open/close/self-close) */
    Attr attrs[MAX_ATTRS];
    int  attr_count;
    char text[SSML_MAX_TEXT]; /* text content */
} XmlToken;

typedef struct {
    const char *src;
    int pos;
    int len;
} XmlScanner;

static void xml_init(XmlScanner *xs, const char *src) {
    xs->src = src;
    xs->pos = 0;
    xs->len = (int)strlen(src);
}

static const char *xml_get_attr(const XmlToken *tok, const char *key) {
    for (int i = 0; i < tok->attr_count; i++) {
        if (streqi(tok->attrs[i].key, key)) return tok->attrs[i].val;
    }
    return NULL;
}

static void xml_decode_entities(char *s) {
    char *r = s, *w = s;
    while (*r) {
        if (*r == '&') {
            if (strncmp(r, "&amp;", 5) == 0)  { *w++ = '&'; r += 5; continue; }
            if (strncmp(r, "&lt;", 4) == 0)   { *w++ = '<'; r += 4; continue; }
            if (strncmp(r, "&gt;", 4) == 0)   { *w++ = '>'; r += 4; continue; }
            if (strncmp(r, "&apos;", 6) == 0) { *w++ = '\''; r += 6; continue; }
            if (strncmp(r, "&quot;", 6) == 0) { *w++ = '"'; r += 6; continue; }
            /* Numeric entity &#NNN; or &#xHH; — pass through */
        }
        *w++ = *r++;
    }
    *w = '\0';
}

static void skip_whitespace(XmlScanner *xs) {
    while (xs->pos < xs->len && isspace((unsigned char)xs->src[xs->pos])) xs->pos++;
}

static int xml_next(XmlScanner *xs, XmlToken *tok) {
    memset(tok, 0, sizeof(*tok));

    if (xs->pos >= xs->len) {
        tok->type = XML_EOF;
        return 0;
    }

    /* Text content (up to next '<') */
    if (xs->src[xs->pos] != '<') {
        int start = xs->pos;
        while (xs->pos < xs->len && xs->src[xs->pos] != '<') xs->pos++;
        int tlen = xs->pos - start;
        if (tlen > SSML_MAX_TEXT - 1) tlen = SSML_MAX_TEXT - 1;
        memcpy(tok->text, xs->src + start, (size_t)tlen);
        tok->text[tlen] = '\0';
        xml_decode_entities(tok->text);
        tok->type = XML_TEXT;
        return 1;
    }

    /* Skip '<' */
    xs->pos++;

    /* XML declaration <?xml ... ?> — skip entirely */
    if (xs->pos < xs->len && xs->src[xs->pos] == '?') {
        while (xs->pos < xs->len) {
            if (xs->src[xs->pos] == '?' && xs->pos + 1 < xs->len && xs->src[xs->pos+1] == '>') {
                xs->pos += 2;
                return xml_next(xs, tok);
            }
            xs->pos++;
        }
        tok->type = XML_EOF;
        return 0;
    }

    /* Comment <!-- ... --> — skip */
    if (xs->pos + 2 < xs->len && xs->src[xs->pos] == '!' &&
        xs->src[xs->pos+1] == '-' && xs->src[xs->pos+2] == '-') {
        xs->pos += 3;
        while (xs->pos + 2 < xs->len) {
            if (xs->src[xs->pos] == '-' && xs->src[xs->pos+1] == '-' && xs->src[xs->pos+2] == '>') {
                xs->pos += 3;
                return xml_next(xs, tok);
            }
            xs->pos++;
        }
        tok->type = XML_EOF;
        return 0;
    }

    /* CDATA <![CDATA[...]]> — extract text */
    if (xs->pos + 8 < xs->len && strncmp(xs->src + xs->pos, "![CDATA[", 8) == 0) {
        xs->pos += 8;
        int start = xs->pos;
        while (xs->pos + 2 < xs->len) {
            if (xs->src[xs->pos] == ']' && xs->src[xs->pos+1] == ']' && xs->src[xs->pos+2] == '>') {
                int tlen = xs->pos - start;
                if (tlen > SSML_MAX_TEXT - 1) tlen = SSML_MAX_TEXT - 1;
                memcpy(tok->text, xs->src + start, (size_t)tlen);
                tok->text[tlen] = '\0';
                tok->type = XML_TEXT;
                xs->pos += 3;
                return 1;
            }
            xs->pos++;
        }
        tok->type = XML_EOF;
        return 0;
    }

    /* Close tag </...> */
    if (xs->pos < xs->len && xs->src[xs->pos] == '/') {
        xs->pos++;
        int ti = 0;
        while (xs->pos < xs->len && xs->src[xs->pos] != '>' && ti < 126) {
            if (!isspace((unsigned char)xs->src[xs->pos])) {
                tok->tag[ti++] = xs->src[xs->pos];
            }
            xs->pos++;
        }
        tok->tag[ti] = '\0';
        if (xs->pos < xs->len) xs->pos++; /* skip '>' */
        tok->type = XML_CLOSE_TAG;

        /* Strip namespace */
        char *colon = strchr(tok->tag, ':');
        if (colon) memmove(tok->tag, colon + 1, strlen(colon));
        return 1;
    }

    /* Open tag <name attr="val" ...> or self-close <name .../> */
    int ti = 0;
    while (xs->pos < xs->len && !isspace((unsigned char)xs->src[xs->pos]) &&
           xs->src[xs->pos] != '>' && xs->src[xs->pos] != '/' && ti < 126) {
        tok->tag[ti++] = xs->src[xs->pos++];
    }
    tok->tag[ti] = '\0';

    /* Strip namespace */
    char *colon = strchr(tok->tag, ':');
    if (colon) memmove(tok->tag, colon + 1, strlen(colon));

    /* Parse attributes */
    tok->attr_count = 0;
    while (xs->pos < xs->len) {
        skip_whitespace(xs);
        if (xs->pos >= xs->len) break;

        /* Self-closing? */
        if (xs->src[xs->pos] == '/' && xs->pos + 1 < xs->len && xs->src[xs->pos+1] == '>') {
            xs->pos += 2;
            tok->type = XML_SELF_CLOSE;
            return 1;
        }
        if (xs->src[xs->pos] == '>') {
            xs->pos++;
            tok->type = XML_OPEN_TAG;
            return 1;
        }

        /* Parse key */
        if (tok->attr_count >= MAX_ATTRS) {
            while (xs->pos < xs->len && xs->src[xs->pos] != '>') xs->pos++;
            if (xs->pos < xs->len) xs->pos++;
            tok->type = XML_OPEN_TAG;
            return 1;
        }

        Attr *a = &tok->attrs[tok->attr_count];
        int ki = 0;
        while (xs->pos < xs->len && xs->src[xs->pos] != '=' &&
               !isspace((unsigned char)xs->src[xs->pos]) &&
               xs->src[xs->pos] != '>' && xs->src[xs->pos] != '/' && ki < ATTR_KEY_MAX - 1) {
            a->key[ki++] = xs->src[xs->pos++];
        }
        a->key[ki] = '\0';

        skip_whitespace(xs);
        if (xs->pos < xs->len && xs->src[xs->pos] == '=') {
            xs->pos++;
            skip_whitespace(xs);

            char quote = 0;
            if (xs->pos < xs->len && (xs->src[xs->pos] == '"' || xs->src[xs->pos] == '\'')) {
                quote = xs->src[xs->pos++];
            }

            int vi = 0;
            if (quote) {
                while (xs->pos < xs->len && xs->src[xs->pos] != quote && vi < ATTR_VAL_MAX - 1) {
                    a->val[vi++] = xs->src[xs->pos++];
                }
                if (xs->pos < xs->len) xs->pos++; /* skip closing quote */
            } else {
                while (xs->pos < xs->len && !isspace((unsigned char)xs->src[xs->pos]) &&
                       xs->src[xs->pos] != '>' && xs->src[xs->pos] != '/' && vi < ATTR_VAL_MAX - 1) {
                    a->val[vi++] = xs->src[xs->pos++];
                }
            }
            a->val[vi] = '\0';
            xml_decode_entities(a->val);
        }
        tok->attr_count++;
    }

    tok->type = XML_OPEN_TAG;
    return 1;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * SSML Walker — recursive descent over the XML token stream
 * ═══════════════════════════════════════════════════════════════════════════ */

typedef struct {
    float rate;
    float pitch;
    float volume;
} ProsodyStack;

typedef struct {
    SSMLSegment *segs;
    int          seg_count;
    int          max_segs;

    char current_voice[SSML_MAX_VOICE];

    ProsodyStack pstack[MAX_NEST_DEPTH];
    int          pdepth;

    char pending_text[SSML_MAX_TEXT];
    int  pending_len;
    int  pending_break_before;

    XmlScanner scanner;
} Walker;

static ProsodyStack *w_prosody(Walker *w) {
    return &w->pstack[w->pdepth];
}

static void w_flush(Walker *w, int break_after_ms) {
    /* Trim whitespace */
    while (w->pending_len > 0 && w->pending_text[w->pending_len - 1] == ' ')
        w->pending_len--;
    int start = 0;
    while (start < w->pending_len && w->pending_text[start] == ' ')
        start++;

    int tlen = w->pending_len - start;
    if (tlen <= 0 && w->pending_break_before <= 0) {
        w->pending_len = 0;
        return;
    }

    if (tlen <= 0 && w->pending_break_before > 0) {
        /* Break-only segment: attach break to next segment if possible */
        return;
    }

    if (w->seg_count >= w->max_segs) {
        w->pending_len = 0;
        return;
    }

    SSMLSegment *seg = &w->segs[w->seg_count];
    memset(seg, 0, sizeof(*seg));

    int copy = tlen < SSML_MAX_TEXT - 1 ? tlen : SSML_MAX_TEXT - 1;
    memcpy(seg->text, w->pending_text + start, (size_t)copy);
    seg->text[copy] = '\0';

    snprintf(seg->voice, SSML_MAX_VOICE, "%s", w->current_voice);
    seg->rate   = w_prosody(w)->rate;
    seg->pitch  = w_prosody(w)->pitch;
    seg->volume = w_prosody(w)->volume;
    seg->break_before_ms = w->pending_break_before;
    seg->break_after_ms  = break_after_ms;
    seg->is_audio = 0;

    w->seg_count++;
    w->pending_len = 0;
    w->pending_text[0] = '\0';
    w->pending_break_before = 0;
}

static void w_add_text(Walker *w, const char *text) {
    if (!text) return;
    int len = (int)strlen(text);
    int space = SSML_MAX_TEXT - w->pending_len - 1;
    int to_copy = len < space ? len : space;
    if (to_copy > 0) {
        memcpy(w->pending_text + w->pending_len, text, (size_t)to_copy);
        w->pending_len += to_copy;
        w->pending_text[w->pending_len] = '\0';
    }
}

/* Break strength → milliseconds */
static int break_strength_ms(const char *strength) {
    if (!strength) return 400;
    if (streqi(strength, "none"))     return 0;
    if (streqi(strength, "x-weak"))   return 100;
    if (streqi(strength, "weak"))     return 200;
    if (streqi(strength, "medium"))   return 400;
    if (streqi(strength, "strong"))   return 600;
    if (streqi(strength, "x-strong")) return 1000;
    return 400;
}

/* Parse time string like "500ms" or "1.5s" into milliseconds */
static int parse_time_ms(const char *s) {
    if (!s || !*s) return 0;
    float val = (float)atof(s);
    if (strstr(s, "ms")) return (int)val;
    if (strchr(s, 's'))  return (int)(val * 1000.0f);
    return (int)(val * 1000.0f);
}

/**
 * Walk children of the current tag. Reads tokens until a matching close tag
 * for `parent_tag` is found (or EOF).
 */
static void walk_children(Walker *w, const char *parent_tag);

static void walk_element(Walker *w, XmlToken *open_tok) {
    const char *tag = open_tok->tag;

    if (streqi(tag, "speak")) {
        walk_children(w, "speak");
        w_flush(w, 0);

    } else if (streqi(tag, "p")) {
        w_flush(w, PARAGRAPH_BREAK_MS);
        walk_children(w, "p");
        w_flush(w, PARAGRAPH_BREAK_MS);

    } else if (streqi(tag, "s")) {
        w_flush(w, SENTENCE_BREAK_MS);
        walk_children(w, "s");
        w_flush(w, SENTENCE_BREAK_MS);

    } else if (streqi(tag, "break")) {
        const char *time_attr = xml_get_attr(open_tok, "time");
        const char *strength  = xml_get_attr(open_tok, "strength");
        int ms;
        if (time_attr) {
            ms = parse_time_ms(time_attr);
        } else {
            ms = break_strength_ms(strength);
        }
        w_flush(w, 0);
        w->pending_break_before = ms;

        /* Self-closing is already handled; if it's an open tag, read close */
        if (open_tok->type == XML_OPEN_TAG) {
            walk_children(w, "break");
        }

    } else if (streqi(tag, "say-as")) {
        const char *interpret_as = xml_get_attr(open_tok, "interpret-as");
        const char *fmt = xml_get_attr(open_tok, "format");
        if (!interpret_as) interpret_as = "";
        if (!fmt) fmt = "";

        /* Collect text content */
        char raw[SSML_MAX_TEXT] = {0};
        int rlen = 0;
        XmlToken child;
        while (xml_next(&w->scanner, &child)) {
            if (child.type == XML_CLOSE_TAG && streqi(child.tag, "say-as")) break;
            if (child.type == XML_TEXT) {
                int tl = (int)strlen(child.text);
                int sp = SSML_MAX_TEXT - rlen - 1;
                int cp = tl < sp ? tl : sp;
                if (cp > 0) { memcpy(raw + rlen, child.text, (size_t)cp); rlen += cp; }
            }
            if (child.type == XML_EOF) break;
        }
        raw[rlen] = '\0';

        char normalized[SSML_MAX_TEXT];
        text_normalize(raw, interpret_as, fmt, normalized, SSML_MAX_TEXT);
        w_add_text(w, normalized);

    } else if (streqi(tag, "sub")) {
        const char *alias = xml_get_attr(open_tok, "alias");
        if (alias && *alias) {
            w_add_text(w, alias);
        }
        if (open_tok->type == XML_OPEN_TAG) {
            /* Skip inner content, use alias */
            XmlToken child;
            while (xml_next(&w->scanner, &child)) {
                if (child.type == XML_CLOSE_TAG && streqi(child.tag, "sub")) break;
                if (!alias || !*alias) {
                    if (child.type == XML_TEXT) w_add_text(w, child.text);
                }
                if (child.type == XML_EOF) break;
            }
        }

    } else if (streqi(tag, "prosody")) {
        /* Push prosody stack */
        if (w->pdepth < MAX_NEST_DEPTH - 1) {
            w->pdepth++;
            w->pstack[w->pdepth] = w->pstack[w->pdepth - 1];  /* inherit */
        }
        const char *r = xml_get_attr(open_tok, "rate");
        const char *pi = xml_get_attr(open_tok, "pitch");
        const char *vo = xml_get_attr(open_tok, "volume");
        if (r)  w_prosody(w)->rate   *= ssml_parse_rate(r);
        if (pi) w_prosody(w)->pitch  *= ssml_parse_pitch(pi);
        if (vo) w_prosody(w)->volume *= ssml_parse_volume(vo);

        w_flush(w, 0);
        walk_children(w, "prosody");
        w_flush(w, 0);

        if (w->pdepth > 0) w->pdepth--;

    } else if (streqi(tag, "emphasis")) {
        if (w->pdepth < MAX_NEST_DEPTH - 1) {
            w->pdepth++;
            w->pstack[w->pdepth] = w->pstack[w->pdepth - 1];
        }
        const char *level = xml_get_attr(open_tok, "level");
        if (!level) level = "moderate";

        if (streqi(level, "strong"))   { w_prosody(w)->rate *= 0.9f; w_prosody(w)->volume *= 1.3f; }
        else if (streqi(level, "moderate")) { w_prosody(w)->rate *= 0.95f; w_prosody(w)->volume *= 1.15f; }
        else if (streqi(level, "none"))     { w_prosody(w)->volume *= 0.7f; }
        else if (streqi(level, "reduced"))  { w_prosody(w)->rate *= 0.8f; w_prosody(w)->volume *= 0.8f; }

        w_flush(w, 0);
        walk_children(w, "emphasis");
        w_flush(w, 0);

        if (w->pdepth > 0) w->pdepth--;

    } else if (streqi(tag, "voice")) {
        const char *name = xml_get_attr(open_tok, "name");
        char saved_voice[SSML_MAX_VOICE];
        snprintf(saved_voice, sizeof(saved_voice), "%s", w->current_voice);
        if (name && *name) {
            snprintf(w->current_voice, sizeof(w->current_voice), "%s", name);
        }
        w_flush(w, 0);
        walk_children(w, "voice");
        w_flush(w, 0);
        snprintf(w->current_voice, sizeof(w->current_voice), "%s", saved_voice);

    } else if (streqi(tag, "audio")) {
        const char *src = xml_get_attr(open_tok, "src");
        if (src && *src && w->seg_count < w->max_segs) {
            w_flush(w, 0);
            SSMLSegment *seg = &w->segs[w->seg_count];
            memset(seg, 0, sizeof(*seg));
            snprintf(seg->text, SSML_MAX_TEXT, "%s", src);
            snprintf(seg->voice, SSML_MAX_VOICE, "%s", w->current_voice);
            seg->rate = w_prosody(w)->rate;
            seg->pitch = w_prosody(w)->pitch;
            seg->volume = w_prosody(w)->volume;
            seg->is_audio = 1;
            w->seg_count++;
        }
        if (open_tok->type == XML_OPEN_TAG) {
            walk_children(w, "audio");
        }

    } else if (streqi(tag, "mark") || streqi(tag, "lexicon") || streqi(tag, "desc")) {
        /* Skip — marks are timing-only, lexicon loading not supported in C, desc is ignored */
        if (open_tok->type == XML_OPEN_TAG) {
            walk_children(w, tag);
        }

    } else if (streqi(tag, "phoneme")) {
        /* Use text content as best-effort */
        if (open_tok->type == XML_OPEN_TAG) {
            walk_children(w, "phoneme");
        }

    } else {
        /* Unknown tag — passthrough (walk children) */
        if (open_tok->type == XML_OPEN_TAG) {
            walk_children(w, tag);
        }
    }
}

static void walk_children(Walker *w, const char *parent_tag) {
    XmlToken tok;
    while (xml_next(&w->scanner, &tok)) {
        switch (tok.type) {
        case XML_TEXT:
            w_add_text(w, tok.text);
            break;
        case XML_OPEN_TAG:
            walk_element(w, &tok);
            break;
        case XML_SELF_CLOSE:
            walk_element(w, &tok);
            break;
        case XML_CLOSE_TAG:
            if (streqi(tok.tag, parent_tag)) return;
            break;
        case XML_EOF:
            return;
        }
    }
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Public API
 * ═══════════════════════════════════════════════════════════════════════════ */

int ssml_is_ssml(const char *text) {
    if (!text) return 0;
    while (*text == ' ' || *text == '\t' || *text == '\n' || *text == '\r') text++;
    if (strncmp(text, "<speak", 6) == 0) return 1;
    if (strncmp(text, "<?xml", 5) == 0)  return 1;
    return 0;
}

int ssml_parse(const char *input, SSMLSegment *segments, int max_segments) {
    if (!input || !segments || max_segments <= 0) return -1;

    /* Trim leading whitespace */
    while (*input == ' ' || *input == '\t' || *input == '\n' || *input == '\r') input++;

    /* Not SSML — return single segment with raw text */
    if (!ssml_is_ssml(input)) {
        memset(&segments[0], 0, sizeof(SSMLSegment));
        int len = (int)strlen(input);
        int copy = len < SSML_MAX_TEXT - 1 ? len : SSML_MAX_TEXT - 1;
        memcpy(segments[0].text, input, (size_t)copy);
        segments[0].text[copy] = '\0';
        segments[0].rate = 1.0f;
        segments[0].pitch = 1.0f;
        segments[0].volume = 1.0f;
        return 1;
    }

    /* Initialize walker */
    Walker w;
    memset(&w, 0, sizeof(w));
    w.segs = segments;
    w.max_segs = max_segments;
    w.seg_count = 0;
    w.pdepth = 0;
    w.pstack[0].rate = 1.0f;
    w.pstack[0].pitch = 1.0f;
    w.pstack[0].volume = 1.0f;
    w.current_voice[0] = '\0';

    xml_init(&w.scanner, input);

    /* Find <speak> and walk it */
    XmlToken tok;
    while (xml_next(&w.scanner, &tok)) {
        if ((tok.type == XML_OPEN_TAG || tok.type == XML_SELF_CLOSE) &&
            streqi(tok.tag, "speak")) {
            walk_element(&w, &tok);
            break;
        }
    }

    /* Filter empty segments */
    int out = 0;
    for (int i = 0; i < w.seg_count; i++) {
        if (segments[i].text[0] != '\0' || segments[i].is_audio ||
            segments[i].break_before_ms > 0) {
            if (out != i) segments[out] = segments[i];
            out++;
        }
    }

    return out;
}
