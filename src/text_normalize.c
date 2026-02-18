/**
 * text_normalize.c — Text normalization for SSML <say-as> and auto-detection.
 *
 * Port of pocket_tts/ssml/text_normalizer.py. Pure string manipulation with
 * zero allocations for result strings — all output written to caller buffers.
 *
 * Build: cc -O3 -shared -fPIC -o libtext_normalize.dylib text_normalize.c
 */

#include "text_normalize.h"
#include <ctype.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ═══════════════════════════════════════════════════════════════════════════
 * Lookup Tables (compile-time constants)
 * ═══════════════════════════════════════════════════════════════════════════ */

static const char *ONES[20] = {
    "", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
    "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen",
    "seventeen", "eighteen", "nineteen",
};

static const char *TENS[10] = {
    "", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety",
};

static const char *ORDINAL_ONES[20] = {
    "", "first", "second", "third", "fourth", "fifth", "sixth", "seventh",
    "eighth", "ninth", "tenth", "eleventh", "twelfth", "thirteenth",
    "fourteenth", "fifteenth", "sixteenth", "seventeenth", "eighteenth", "nineteenth",
};

static const char *ORDINAL_TENS[10] = {
    "", "", "twentieth", "thirtieth", "fortieth", "fiftieth",
    "sixtieth", "seventieth", "eightieth", "ninetieth",
};

static const char *MONTHS[13] = {
    "", "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
};

typedef struct { char sym[8]; const char *s; const char *p; const char *cs; const char *cp; } CurrencyEntry;

static const CurrencyEntry CURRENCIES[] = {
    { "$",   "dollar",      "dollars",      "cent",    "cents" },
    { "\xe2\x82\xac", "euro", "euros",      "cent",    "cents" },       /* € */
    { "\xc2\xa3",      "pound", "pounds",   "penny",   "pence" },       /* £ */
    { "\xc2\xa5",      "yen",   "yen",      "sen",     "sen" },         /* ¥ */
    { "\xe2\x82\xb9",  "rupee", "rupees",   "paisa",   "paise" },       /* ₹ */
    { "\xe2\x82\xa9",  "won",   "won",      "",        "" },            /* ₩ */
    { "\xe2\x82\xbf",  "bitcoin","bitcoins","satoshi","satoshis" },      /* ₿ */
    { "CHF", "Swiss franc", "Swiss francs", "centime", "centimes" },
    { "kr",  "krone",       "kroner",       "\xc3\xb8re", "\xc3\xb8re" },
    { "R$",  "real",        "reais",        "centavo", "centavos" },
    { "",    NULL,          NULL,           NULL,      NULL },
};

typedef struct { const char *abbrev; const char *full; } UnitEntry;

static const UnitEntry UNITS[] = {
    { "km/h", "kilometers per hour" },
    { "kph",  "kilometers per hour" },
    { "mph",  "miles per hour" },
    { "\xc2\xb0""C", "degrees Celsius" },    /* °C */
    { "\xc2\xb0""F", "degrees Fahrenheit" },  /* °F */
    { "km",   "kilometers" },
    { "cm",   "centimeters" },
    { "mm",   "millimeters" },
    { "mi",   "miles" },
    { "ft",   "feet" },
    { "in",   "inches" },
    { "yd",   "yards" },
    { "kg",   "kilograms" },
    { "mg",   "milligrams" },
    { "lbs",  "pounds" },
    { "lb",   "pounds" },
    { "oz",   "ounces" },
    { "ml",   "milliliters" },
    { "gal",  "gallons" },
    { "m",    "meters" },
    { "g",    "grams" },
    { "l",    "liters" },
    { NULL, NULL },
};

typedef struct { char ch; const char *name; } CharName;

static const CharName CHAR_NAMES[] = {
    { ' ', "space" }, { '.', "dot" }, { ',', "comma" },
    { '!', "exclamation mark" }, { '?', "question mark" }, { '@', "at" },
    { '#', "hash" }, { '$', "dollar sign" }, { '%', "percent" },
    { '&', "ampersand" }, { '*', "asterisk" }, { '+', "plus" },
    { '-', "dash" }, { '/', "slash" }, { '\\', "backslash" },
    { '=', "equals" }, { '(', "open parenthesis" }, { ')', "close parenthesis" },
    { '[', "open bracket" }, { ']', "close bracket" }, { '{', "open brace" },
    { '}', "close brace" }, { '<', "less than" }, { '>', "greater than" },
    { ':', "colon" }, { ';', "semicolon" }, { '\'', "apostrophe" },
    { '"', "quote" }, { '_', "underscore" }, { '~', "tilde" },
    { '|', "pipe" }, { '^', "caret" }, { '`', "backtick" },
    { 0, NULL },
};

/* Common fractions table */
typedef struct { int num; int den; const char *name; } FractionEntry;

static const FractionEntry FRACTIONS[] = {
    { 1, 2, "one half" }, { 1, 3, "one third" }, { 2, 3, "two thirds" },
    { 1, 4, "one quarter" }, { 3, 4, "three quarters" },
    { 1, 5, "one fifth" }, { 2, 5, "two fifths" },
    { 3, 5, "three fifths" }, { 4, 5, "four fifths" },
    { 1, 6, "one sixth" }, { 1, 8, "one eighth" },
    { 3, 8, "three eighths" }, { 5, 8, "five eighths" },
    { 7, 8, "seven eighths" }, { 1, 10, "one tenth" },
    { 0, 0, NULL },
};

/* ═══════════════════════════════════════════════════════════════════════════
 * Helper: safe snprintf-style append to output buffer
 * ═══════════════════════════════════════════════════════════════════════════ */

typedef struct {
    char *buf;
    int   cap;
    int   pos;
} OutBuf;

static void ob_init(OutBuf *ob, char *buf, int cap) {
    ob->buf = buf;
    ob->cap = cap;
    ob->pos = 0;
    if (cap > 0) buf[0] = '\0';
}

static void ob_append(OutBuf *ob, const char *s) {
    if (!s || ob->pos >= ob->cap - 1) return;
    int n = snprintf(ob->buf + ob->pos, (size_t)(ob->cap - ob->pos), "%s", s);
    if (n > 0) {
        ob->pos += n;
        if (ob->pos >= ob->cap) ob->pos = ob->cap - 1;
    }
}

static void ob_append_ch(OutBuf *ob, char c) {
    if (ob->pos < ob->cap - 1) {
        ob->buf[ob->pos++] = c;
        ob->buf[ob->pos] = '\0';
    }
}

static int ob_len(const OutBuf *ob) { return ob->pos; }

/* ═══════════════════════════════════════════════════════════════════════════
 * Core: number_to_words, ordinal_to_words
 * ═══════════════════════════════════════════════════════════════════════════ */

static void number_to_words(long long n, OutBuf *ob) {
    if (n < 0) {
        ob_append(ob, "negative ");
        number_to_words(-n, ob);
        return;
    }
    if (n == 0) {
        ob_append(ob, "zero");
        return;
    }

    int first = 1;
    #define EMIT_GROUP(divisor, label) \
        if (n >= (divisor)) { \
            if (!first) ob_append(ob, " "); \
            number_to_words(n / (divisor), ob); \
            ob_append(ob, " " label); \
            n %= (divisor); \
            first = 0; \
        }

    EMIT_GROUP(1000000000000LL, "trillion")
    EMIT_GROUP(1000000000LL,    "billion")
    EMIT_GROUP(1000000LL,       "million")
    EMIT_GROUP(1000LL,          "thousand")
    #undef EMIT_GROUP

    if (n >= 100) {
        if (!first) ob_append(ob, " ");
        ob_append(ob, ONES[n / 100]);
        ob_append(ob, " hundred");
        n %= 100;
        first = 0;
    }

    if (n >= 20) {
        if (!first) ob_append(ob, " ");
        int t = (int)(n / 10), o = (int)(n % 10);
        ob_append(ob, TENS[t]);
        if (o) { ob_append_ch(ob, '-'); ob_append(ob, ONES[o]); }
        first = 0;
    } else if (n > 0) {
        if (!first) ob_append(ob, " ");
        ob_append(ob, ONES[n]);
    }
}

static void ordinal_to_words(long long n, OutBuf *ob) {
    if (n < 0) {
        ob_append(ob, "negative ");
        ordinal_to_words(-n, ob);
        return;
    }
    if (n == 0) {
        ob_append(ob, "zeroth");
        return;
    }
    if (n >= 100) {
        number_to_words((n / 100) * 100, ob);
        long long rem = n % 100;
        if (rem == 0) {
            ob_append(ob, "th");
        } else {
            ob_append(ob, " ");
            ordinal_to_words(rem, ob);
        }
        return;
    }
    if (n >= 20) {
        int t = (int)(n / 10), o = (int)(n % 10);
        if (o) {
            ob_append(ob, TENS[t]);
            ob_append_ch(ob, '-');
            ob_append(ob, ORDINAL_ONES[o]);
        } else {
            ob_append(ob, ORDINAL_TENS[t]);
        }
        return;
    }
    ob_append(ob, ORDINAL_ONES[n]);
}

static void year_to_words(int year, OutBuf *ob) {
    if (year >= 2000 && year < 2010) {
        ob_append(ob, "two thousand");
        if (year > 2000) { ob_append(ob, " "); ob_append(ob, ONES[year - 2000]); }
    } else if (year >= 2010 && year < 2100) {
        ob_append(ob, "twenty ");
        number_to_words(year - 2000, ob);
    } else if (year >= 1000) {
        int hi = year / 100, lo = year % 100;
        number_to_words(hi, ob);
        if (lo == 0) {
            ob_append(ob, " hundred");
        } else {
            ob_append(ob, " ");
            number_to_words(lo, ob);
        }
    } else {
        number_to_words(year, ob);
    }
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Helpers
 * ═══════════════════════════════════════════════════════════════════════════ */

static const char *skip_ws(const char *s) {
    while (*s == ' ' || *s == '\t' || *s == '\n' || *s == '\r') s++;
    return s;
}

static long long parse_int(const char *s, const char **endp) {
    long long val = 0;
    int neg = 0;
    s = skip_ws(s);
    if (*s == '-') { neg = 1; s++; }
    else if (*s == '+') { s++; }
    while (*s >= '0' && *s <= '9') {
        val = val * 10 + (*s - '0');
        s++;
    }
    if (endp) *endp = s;
    return neg ? -val : val;
}

/* Parse integer, skipping commas (e.g. "1,234,567") */
static long long parse_int_with_commas(const char *s, const char **endp) {
    long long val = 0;
    int neg = 0;
    s = skip_ws(s);
    if (*s == '-') { neg = 1; s++; }
    else if (*s == '+') { s++; }
    while ((*s >= '0' && *s <= '9') || *s == ',') {
        if (*s != ',') val = val * 10 + (*s - '0');
        s++;
    }
    if (endp) *endp = s;
    return neg ? -val : val;
}

static int str_starts_with(const char *s, const char *prefix) {
    return strncmp(s, prefix, strlen(prefix)) == 0;
}

static int str_ends_with(const char *s, const char *suffix) {
    int slen = (int)strlen(s), suflen = (int)strlen(suffix);
    if (suflen > slen) return 0;
    return strcmp(s + slen - suflen, suffix) == 0;
}

static int streqi(const char *a, const char *b) {
    while (*a && *b) {
        if (tolower((unsigned char)*a) != tolower((unsigned char)*b)) return 0;
        a++; b++;
    }
    return *a == *b;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Public converters
 * ═══════════════════════════════════════════════════════════════════════════ */

int text_cardinal(const char *text, char *out, int out_cap) {
    OutBuf ob; ob_init(&ob, out, out_cap);
    text = skip_ws(text);

    /* Check for decimal point */
    const char *dot = strchr(text, '.');
    if (dot && dot != text) {
        const char *ep;
        long long int_part = parse_int_with_commas(text, &ep);
        if (ep == dot) {
            const char *dec = dot + 1;
            number_to_words(int_part, &ob);
            ob_append(&ob, " point");
            while (*dec >= '0' && *dec <= '9') {
                int d = *dec - '0';
                ob_append(&ob, " ");
                ob_append(&ob, d == 0 ? "zero" : ONES[d]);
                dec++;
            }
            return ob_len(&ob);
        }
    }

    long long val = parse_int_with_commas(text, NULL);
    number_to_words(val, &ob);
    return ob_len(&ob);
}

int text_ordinal(const char *text, char *out, int out_cap) {
    OutBuf ob; ob_init(&ob, out, out_cap);
    text = skip_ws(text);

    /* Strip ordinal suffix (st, nd, rd, th) */
    char cleaned[256];
    int len = (int)strlen(text);
    if (len > 2) {
        const char *end2 = text + len - 2;
        if (streqi(end2, "st") || streqi(end2, "nd") || streqi(end2, "rd") || streqi(end2, "th")) {
            int cl = len - 2;
            if (cl > 255) cl = 255;
            memcpy(cleaned, text, (size_t)cl);
            cleaned[cl] = '\0';
            text = cleaned;
        }
    }

    long long val = parse_int_with_commas(text, NULL);
    ordinal_to_words(val, &ob);
    return ob_len(&ob);
}

int text_characters(const char *text, char *out, int out_cap) {
    OutBuf ob; ob_init(&ob, out, out_cap);
    int first = 1;
    while (*text) {
        unsigned char ch = (unsigned char)*text;
        if (!first) ob_append(&ob, " ");
        first = 0;

        if (isalpha(ch)) {
            char upper[2] = { (char)toupper(ch), 0 };
            ob_append(&ob, upper);
        } else if (isdigit(ch)) {
            int d = ch - '0';
            ob_append(&ob, d == 0 ? "zero" : ONES[d]);
        } else {
            int found = 0;
            for (const CharName *cn = CHAR_NAMES; cn->name; cn++) {
                if (cn->ch == (char)ch) {
                    ob_append(&ob, cn->name);
                    found = 1;
                    break;
                }
            }
            if (!found) {
                ob_append_ch(&ob, (char)ch);
            }
        }
        text++;
    }
    return ob_len(&ob);
}

int text_fraction(const char *text, char *out, int out_cap) {
    OutBuf ob; ob_init(&ob, out, out_cap);
    text = skip_ws(text);

    /* Try "N/D" */
    const char *slash = strchr(text, '/');
    if (slash) {
        /* Check for mixed number "W N/D" */
        const char *sp = strchr(text, ' ');
        if (sp && sp < slash) {
            long long whole = parse_int(text, NULL);
            long long num = parse_int(sp + 1, NULL);
            long long den = parse_int(slash + 1, NULL);
            if (den != 0) {
                number_to_words(whole, &ob);
                ob_append(&ob, " and ");
                /* Recurse for fraction part */
                char frac_buf[128];
                char frac_text[64];
                snprintf(frac_text, sizeof(frac_text), "%lld/%lld", num, den);
                text_fraction(frac_text, frac_buf, sizeof(frac_buf));
                ob_append(&ob, frac_buf);
                return ob_len(&ob);
            }
        }

        long long num = parse_int(text, NULL);
        long long den = parse_int(slash + 1, NULL);
        if (den == 0) { ob_append(&ob, text); return ob_len(&ob); }

        /* Check named fractions */
        for (const FractionEntry *f = FRACTIONS; f->name; f++) {
            if (f->num == (int)num && f->den == (int)den) {
                ob_append(&ob, f->name);
                return ob_len(&ob);
            }
        }

        /* Build from ordinal denominator */
        number_to_words(num, &ob);
        ob_append(&ob, " ");
        if (den == 2) {
            ob_append(&ob, num == 1 ? "half" : "halves");
        } else {
            char den_buf[128];
            OutBuf db; ob_init(&db, den_buf, sizeof(den_buf));
            ordinal_to_words(den, &db);
            ob_append(&ob, den_buf);
            if (num != 1) ob_append(&ob, "s");
        }
        return ob_len(&ob);
    }

    ob_append(&ob, text);
    return ob_len(&ob);
}

int text_date(const char *text, const char *fmt, char *out, int out_cap) {
    OutBuf ob; ob_init(&ob, out, out_cap);
    text = skip_ws(text);
    if (!fmt) fmt = "";

    int a = 0, b = 0, c = 0;
    char sep = 0;
    int parts_found = 0;

    /* Try MM/DD/YYYY or M/D/YY with various separators */
    {
        const char *p = text;
        a = (int)parse_int(p, &p);
        if (*p == '/' || *p == '-' || *p == '.') { sep = *p; p++; }
        else goto passthrough;
        b = (int)parse_int(p, &p);
        if (*p == sep) {
            p++;
            c = (int)parse_int(p, &p);
            parts_found = 3;
        } else {
            parts_found = 2;
        }
    }

    if (parts_found < 2) goto passthrough;

    int month, day, year;
    if (parts_found == 3 && a >= 1000) {
        /* YYYY-MM-DD (ISO) */
        year = a; month = b; day = c;
    } else if (streqi(fmt, "dmy") || streqi(fmt, "d/m/y") || streqi(fmt, "dm") || streqi(fmt, "d/m")) {
        day = a; month = b; year = c;
    } else if (streqi(fmt, "ymd") || streqi(fmt, "y/m/d")) {
        year = a; month = b; day = c;
    } else {
        /* Default: MDY */
        month = a; day = b; year = c;
    }

    if (parts_found == 3 && year < 100) {
        year += (year < 50) ? 2000 : 1900;
    }

    const char *month_name = (month >= 1 && month <= 12) ? MONTHS[month] : "?";

    /* Format-only modes */
    if (streqi(fmt, "d")) {
        ordinal_to_words(day, &ob);
        return ob_len(&ob);
    }
    if (streqi(fmt, "m")) {
        ob_append(&ob, month_name);
        return ob_len(&ob);
    }
    if (streqi(fmt, "y")) {
        year_to_words(year, &ob);
        return ob_len(&ob);
    }
    if (streqi(fmt, "md") || streqi(fmt, "m/d")) {
        ob_append(&ob, month_name); ob_append(&ob, " ");
        ordinal_to_words(day, &ob);
        return ob_len(&ob);
    }
    if (streqi(fmt, "dm") || streqi(fmt, "d/m")) {
        ordinal_to_words(day, &ob); ob_append(&ob, " of "); ob_append(&ob, month_name);
        return ob_len(&ob);
    }
    if (streqi(fmt, "my") || streqi(fmt, "m/y") || streqi(fmt, "ym") || streqi(fmt, "y/m")) {
        ob_append(&ob, month_name); ob_append(&ob, " ");
        year_to_words(year, &ob);
        return ob_len(&ob);
    }

    /* Full date */
    ob_append(&ob, month_name);
    ob_append(&ob, " ");
    ordinal_to_words(day, &ob);
    if (parts_found == 3) {
        ob_append(&ob, ", ");
        year_to_words(year, &ob);
    }
    return ob_len(&ob);

passthrough:
    ob_append(&ob, text);
    return ob_len(&ob);
}

int text_time(const char *text, const char *fmt, char *out, int out_cap) {
    OutBuf ob; ob_init(&ob, out, out_cap);
    text = skip_ws(text);
    (void)fmt;

    /* HH:MM[:SS] [AM/PM] */
    const char *p = text;
    int hour = (int)parse_int(p, &p);
    if (*p != ':') { ob_append(&ob, text); return ob_len(&ob); }
    p++;
    int minute = (int)parse_int(p, &p);
    int second = -1;
    if (*p == ':') { p++; second = (int)parse_int(p, &p); }

    p = skip_ws(p);
    char ampm[4] = {0};
    if ((*p == 'A' || *p == 'a' || *p == 'P' || *p == 'p') &&
        (*(p+1) == 'M' || *(p+1) == 'm')) {
        ampm[0] = (char)toupper((unsigned char)*p);
        ampm[1] = 'M';
        ampm[2] = '\0';
    }

    /* Special: midnight and noon */
    if (minute == 0 && (second <= 0)) {
        if (ampm[0]) {
            if (hour == 12 && ampm[0] == 'A') { ob_append(&ob, "midnight"); return ob_len(&ob); }
            if (hour == 12 && ampm[0] == 'P') { ob_append(&ob, "noon"); return ob_len(&ob); }
        } else {
            if (hour == 0)  { ob_append(&ob, "midnight"); return ob_len(&ob); }
            if (hour == 12) { ob_append(&ob, "noon"); return ob_len(&ob); }
        }
    }

    if (hour > 23) { ob_append(&ob, text); return ob_len(&ob); }

    number_to_words(hour, &ob);
    if (minute == 0) {
        if (!ampm[0]) ob_append(&ob, " o'clock");
    } else if (minute < 10) {
        ob_append(&ob, " oh ");
        number_to_words(minute, &ob);
    } else {
        ob_append(&ob, " ");
        number_to_words(minute, &ob);
    }

    if (second > 0) {
        ob_append(&ob, " and ");
        number_to_words(second, &ob);
        ob_append(&ob, " seconds");
    }

    if (ampm[0]) {
        ob_append(&ob, " ");
        ob_append(&ob, ampm);
    }

    return ob_len(&ob);
}

int text_telephone(const char *text, char *out, int out_cap) {
    OutBuf ob; ob_init(&ob, out, out_cap);
    int first = 1;
    while (*text) {
        unsigned char ch = (unsigned char)*text;
        if (isdigit(ch)) {
            if (!first) ob_append(&ob, " ");
            int d = ch - '0';
            ob_append(&ob, d == 0 ? "zero" : ONES[d]);
            first = 0;
        } else if (ch == ' ' || ch == '-' || ch == '(' || ch == ')' || ch == '.') {
            if (!first) ob_append(&ob, ", ");
        }
        text++;
    }
    return ob_len(&ob);
}

int text_currency(const char *text, char *out, int out_cap) {
    OutBuf ob; ob_init(&ob, out, out_cap);
    text = skip_ws(text);

    for (const CurrencyEntry *ce = CURRENCIES; ce->s; ce++) {
        if (ce->sym[0] == '\0') continue;
        const char *found = strstr(text, ce->sym);
        if (!found) continue;

        /* Extract amount by collecting digits, commas, dots from the rest */
        char amount_str[128] = {0};
        int ai = 0;
        for (const char *p = text; *p && ai < 126; p++) {
            if (p >= found && p < found + strlen(ce->sym)) continue;
            if (isdigit((unsigned char)*p) || *p == '.' || *p == ',') {
                if (*p != ',') amount_str[ai++] = *p;
            }
        }
        amount_str[ai] = '\0';
        if (ai == 0) continue;

        double amount = atof(amount_str);
        long long integer_part = (long long)amount;
        int decimal_part = (int)((amount - (double)integer_part) * 100.0 + 0.5);

        const char *dollar_word = (integer_part == 1) ? ce->s : ce->p;
        number_to_words(integer_part, &ob);
        ob_append(&ob, " ");
        ob_append(&ob, dollar_word);

        if (decimal_part > 0 && ce->cs[0]) {
            const char *cent_word = (decimal_part == 1) ? ce->cs : ce->cp;
            ob_append(&ob, " and ");
            number_to_words(decimal_part, &ob);
            ob_append(&ob, " ");
            ob_append(&ob, cent_word);
        }

        return ob_len(&ob);
    }

    ob_append(&ob, text);
    return ob_len(&ob);
}

int text_unit(const char *text, char *out, int out_cap) {
    OutBuf ob; ob_init(&ob, out, out_cap);
    text = skip_ws(text);

    for (const UnitEntry *u = UNITS; u->abbrev; u++) {
        if (str_ends_with(text, u->abbrev)) {
            int num_len = (int)strlen(text) - (int)strlen(u->abbrev);
            if (num_len <= 0) continue;

            char num_str[128] = {0};
            int ni = 0;
            for (int i = 0; i < num_len && ni < 126; i++) {
                if (text[i] != ',' && text[i] != ' ') num_str[ni++] = text[i];
            }
            num_str[ni] = '\0';

            const char *dot = strchr(num_str, '.');
            if (dot) {
                double val = atof(num_str);
                long long ip = (long long)val;
                number_to_words(ip, &ob);
                ob_append(&ob, " point");
                const char *dp = dot + 1;
                while (*dp >= '0' && *dp <= '9') {
                    int d = *dp - '0';
                    ob_append(&ob, " ");
                    ob_append(&ob, d == 0 ? "zero" : ONES[d]);
                    dp++;
                }
            } else {
                long long val = parse_int_with_commas(num_str, NULL);
                number_to_words(val, &ob);
            }
            ob_append(&ob, " ");
            ob_append(&ob, u->full);
            return ob_len(&ob);
        }
    }

    ob_append(&ob, text);
    return ob_len(&ob);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Main entry: text_normalize
 * ═══════════════════════════════════════════════════════════════════════════ */

int text_normalize(const char *text, const char *interpret_as,
                   const char *fmt, char *out, int out_cap) {
    if (!interpret_as || !*interpret_as) {
        snprintf(out, (size_t)out_cap, "%s", text);
        return (int)strlen(out);
    }

    if (streqi(interpret_as, "cardinal") || streqi(interpret_as, "number"))
        return text_cardinal(text, out, out_cap);
    if (streqi(interpret_as, "ordinal"))
        return text_ordinal(text, out, out_cap);
    if (streqi(interpret_as, "characters") || streqi(interpret_as, "spell-out") || streqi(interpret_as, "verbatim"))
        return text_characters(text, out, out_cap);
    if (streqi(interpret_as, "date"))
        return text_date(text, fmt, out, out_cap);
    if (streqi(interpret_as, "time"))
        return text_time(text, fmt, out, out_cap);
    if (streqi(interpret_as, "telephone") || streqi(interpret_as, "phone"))
        return text_telephone(text, out, out_cap);
    if (streqi(interpret_as, "currency"))
        return text_currency(text, out, out_cap);
    if (streqi(interpret_as, "fraction"))
        return text_fraction(text, out, out_cap);
    if (streqi(interpret_as, "unit"))
        return text_unit(text, out, out_cap);

    /* Unknown interpret_as — passthrough */
    snprintf(out, (size_t)out_cap, "%s", text);
    return (int)strlen(out);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Auto-normalize: scan raw text for patterns and normalize inline
 * ═══════════════════════════════════════════════════════════════════════════ */

int text_auto_normalize(const char *text, char *out, int out_cap) {
    OutBuf ob; ob_init(&ob, out, out_cap);
    const char *p = text;

    while (*p) {
        /* Currency: $123.45 or €50 */
        if (*p == '$' || (unsigned char)*p == 0xc2 || (unsigned char)*p == 0xe2) {
            int sym_len = 0;
            for (const CurrencyEntry *ce = CURRENCIES; ce->s; ce++) {
                if (ce->sym[0] && str_starts_with(p, ce->sym)) {
                    sym_len = (int)strlen(ce->sym);
                    break;
                }
            }
            if (sym_len > 0 && isdigit((unsigned char)p[sym_len])) {
                /* Find end of number — don't consume trailing dot if not followed by digit */
                const char *end = p + sym_len;
                while (isdigit((unsigned char)*end) || *end == ',' ||
                       (*end == '.' && isdigit((unsigned char)*(end + 1)))) end++;
                char token[128] = {0};
                int tlen = (int)(end - p);
                if (tlen > 127) tlen = 127;
                memcpy(token, p, (size_t)tlen);
                token[tlen] = '\0';
                char norm[256];
                text_currency(token, norm, sizeof(norm));
                ob_append(&ob, norm);
                p = end;
                continue;
            }
        }

        /* Ordinal suffix: 1st, 2nd, 3rd, 4th, 21st, etc. */
        if (isdigit((unsigned char)*p)) {
            const char *ds = p;
            while (isdigit((unsigned char)*ds) || *ds == ',') ds++;

            int has_suffix = 0;
            if (ds - p >= 1 && ds - p <= 10) {
                if (str_starts_with(ds, "st") || str_starts_with(ds, "nd") ||
                    str_starts_with(ds, "rd") || str_starts_with(ds, "th")) {
                    /* Check that suffix is followed by space or end */
                    char after = *(ds + 2);
                    if (after == '\0' || after == ' ' || after == ',' || after == '.') {
                        has_suffix = 1;
                    }
                }
            }

            if (has_suffix) {
                char token[64] = {0};
                int tlen = (int)(ds + 2 - p);
                if (tlen > 63) tlen = 63;
                memcpy(token, p, (size_t)tlen);
                token[tlen] = '\0';
                char norm[128];
                text_ordinal(token, norm, sizeof(norm));
                ob_append(&ob, norm);
                p = ds + 2;
                continue;
            }

            /* Fraction: N/M (not part of a URL or path) */
            if (*ds == '/' && isdigit((unsigned char)*(ds + 1))) {
                const char *de = ds + 1;
                while (isdigit((unsigned char)*de)) de++;
                char after = *de;
                if (after == '\0' || after == ' ' || after == ',' || after == '.') {
                    char token[64] = {0};
                    int tlen = (int)(de - p);
                    if (tlen > 63) tlen = 63;
                    memcpy(token, p, (size_t)tlen);
                    token[tlen] = '\0';
                    char norm[128];
                    text_fraction(token, norm, sizeof(norm));
                    ob_append(&ob, norm);
                    p = de;
                    continue;
                }
            }

            /* Time: digits followed by colon and digits (e.g. 12:30, 3:45 PM) */
            if (*ds == ':' && isdigit((unsigned char)*(ds + 1))) {
                const char *te = ds + 1;
                while (isdigit((unsigned char)*te)) te++;
                if (*te == ':' && isdigit((unsigned char)*(te + 1))) {
                    te++; while (isdigit((unsigned char)*te)) te++;
                }
                const char *ts = skip_ws(te);
                if ((*ts == 'A' || *ts == 'a' || *ts == 'P' || *ts == 'p') &&
                    (*(ts+1) == 'M' || *(ts+1) == 'm')) {
                    te = ts + 2;
                }
                char token[64] = {0};
                int tlen = (int)(te - p);
                if (tlen > 63) tlen = 63;
                memcpy(token, p, (size_t)tlen);
                token[tlen] = '\0';
                char norm[128];
                text_time(token, "", norm, sizeof(norm));
                ob_append(&ob, norm);
                p = te;
                continue;
            }

            /* Plain number — just copy digits through */
        }

        /* Default: copy character through */
        /* Handle multi-byte UTF-8 */
        unsigned char ch = (unsigned char)*p;
        if (ch < 0x80) {
            ob_append_ch(&ob, *p);
            p++;
        } else {
            int bytes = 1;
            if ((ch & 0xE0) == 0xC0) bytes = 2;
            else if ((ch & 0xF0) == 0xE0) bytes = 3;
            else if ((ch & 0xF8) == 0xF0) bytes = 4;
            for (int i = 0; i < bytes && *p; i++) {
                ob_append_ch(&ob, *p);
                p++;
            }
        }
    }

    return ob_len(&ob);
}
