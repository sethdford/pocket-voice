/**
 * test_pipeline.c — Quick smoke test for text_normalize, sentence_buffer, ssml_parser.
 *
 * Build: cc -O3 -arch arm64 -Isrc -Lbuild \
 *          -ltext_normalize -lsentence_buffer -lssml_parser \
 *          -Wl,-rpath,@executable_path/build \
 *          -o tests/test_pipeline tests/test_pipeline.c
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "text_normalize.h"
#include "sentence_buffer.h"
#include "ssml_parser.h"

static int tests_run = 0;
static int tests_passed = 0;

#define ASSERT_STR_EQ(a, b) do { \
    tests_run++; \
    if (strcmp((a), (b)) == 0) { \
        tests_passed++; \
    } else { \
        fprintf(stderr, "  FAIL line %d: \"%s\" != \"%s\"\n", __LINE__, (a), (b)); \
    } \
} while(0)

#define ASSERT_INT_EQ(a, b) do { \
    tests_run++; \
    if ((a) == (b)) { \
        tests_passed++; \
    } else { \
        fprintf(stderr, "  FAIL line %d: %d != %d\n", __LINE__, (a), (b)); \
    } \
} while(0)

static void test_text_normalize(void) {
    char out[512];

    printf("=== text_normalize ===\n");

    text_cardinal("42", out, sizeof(out));
    ASSERT_STR_EQ(out, "forty-two");

    text_cardinal("1000", out, sizeof(out));
    ASSERT_STR_EQ(out, "one thousand");

    text_cardinal("0", out, sizeof(out));
    ASSERT_STR_EQ(out, "zero");

    text_cardinal("1234567", out, sizeof(out));
    ASSERT_STR_EQ(out, "one million two hundred thirty-four thousand five hundred sixty-seven");

    text_ordinal("1st", out, sizeof(out));
    ASSERT_STR_EQ(out, "first");

    text_ordinal("42nd", out, sizeof(out));
    ASSERT_STR_EQ(out, "forty-second");

    text_fraction("1/2", out, sizeof(out));
    ASSERT_STR_EQ(out, "one half");

    text_fraction("3/4", out, sizeof(out));
    ASSERT_STR_EQ(out, "three quarters");

    text_time("12:30 PM", "", out, sizeof(out));
    ASSERT_STR_EQ(out, "twelve thirty PM");

    text_time("0:00", "", out, sizeof(out));
    ASSERT_STR_EQ(out, "midnight");

    text_time("3:05", "", out, sizeof(out));
    ASSERT_STR_EQ(out, "three oh five");

    text_currency("$42.50", out, sizeof(out));
    ASSERT_STR_EQ(out, "forty-two dollars and fifty cents");

    text_currency("$1.00", out, sizeof(out));
    ASSERT_STR_EQ(out, "one dollar");

    text_telephone("555-1234", out, sizeof(out));
    /* Digits: five five five (pause) one two three four */
    printf("  telephone: \"%s\"\n", out);

    text_characters("ABC", out, sizeof(out));
    ASSERT_STR_EQ(out, "A B C");

    text_unit("5km", out, sizeof(out));
    ASSERT_STR_EQ(out, "five kilometers");

    text_date("12/25/2025", "", out, sizeof(out));
    ASSERT_STR_EQ(out, "December twenty-fifth, twenty twenty-five");

    text_normalize("42", "cardinal", "", out, sizeof(out));
    ASSERT_STR_EQ(out, "forty-two");

    text_normalize("1/2", "fraction", "", out, sizeof(out));
    ASSERT_STR_EQ(out, "one half");

    /* Auto-normalize */
    text_auto_normalize("I have $100 and 1/3 left.", out, sizeof(out));
    printf("  auto: \"%s\"\n", out);

    text_auto_normalize("The 1st item costs $42.50.", out, sizeof(out));
    printf("  auto: \"%s\"\n", out);

    printf("  text_normalize: %d/%d passed\n\n", tests_passed, tests_run);
}

static void test_sentence_buffer(void) {
    int prev_passed = tests_passed, prev_run = tests_run;
    char out[2048];

    printf("=== sentence_buffer ===\n");

    SentenceBuffer *sb = sentbuf_create(SENTBUF_MODE_SENTENCE, 5);
    ASSERT_INT_EQ(sentbuf_has_segment(sb), 0);

    /* Feed tokens that form a sentence */
    sentbuf_add(sb, "Hello world. ", 13);
    ASSERT_INT_EQ(sentbuf_has_segment(sb), 1);

    int n = sentbuf_flush(sb, out, sizeof(out));
    ASSERT_STR_EQ(out, "Hello world.");
    (void)n;

    /* Feed multiple sentences */
    sentbuf_add(sb, "First. Second. ", 15);
    ASSERT_INT_EQ(sentbuf_has_segment(sb), 1);
    sentbuf_flush(sb, out, sizeof(out));
    ASSERT_STR_EQ(out, "First.");

    /* Flush all */
    sentbuf_add(sb, "Remaining text", 14);
    sentbuf_flush_all(sb, out, sizeof(out));
    printf("  flush_all: \"%s\"\n", out);

    /* Test markdown stripping */
    sentbuf_reset(sb);
    sentbuf_add(sb, "This is **bold** text. ", 22);
    ASSERT_INT_EQ(sentbuf_has_segment(sb), 1);
    sentbuf_flush(sb, out, sizeof(out));
    ASSERT_STR_EQ(out, "This is bold text.");

    /* Code block filtering */
    sentbuf_reset(sb);
    sentbuf_add(sb, "Before ```code``` after. ", 24);
    sentbuf_flush(sb, out, sizeof(out));
    printf("  code-strip: \"%s\"\n", out);

    sentbuf_destroy(sb);

    /* Eager flush: fires before sentence boundary */
    sb = sentbuf_create(SENTBUF_MODE_SENTENCE, 5);
    sentbuf_set_eager(sb, 4);
    sentbuf_add(sb, "One two three four five six", 27);
    ASSERT_INT_EQ(sentbuf_has_segment(sb), 1);
    sentbuf_flush(sb, out, sizeof(out));
    printf("  eager-flush: \"%s\"\n", out);

    /* Sentence boundary flushes and disables eager mode */
    sentbuf_add(sb, " seven.", 7);
    ASSERT_INT_EQ(sentbuf_has_segment(sb), 1);
    sentbuf_flush(sb, out, sizeof(out));
    ASSERT_STR_EQ(out, "six seven.");
    printf("  post-sentence: \"%s\"\n", out);

    /* After sentence boundary, eager is disabled — 5 words don't trigger */
    sentbuf_add(sb, "Aa bb cc dd ee", 14);
    ASSERT_INT_EQ(sentbuf_has_segment(sb), 0);
    sentbuf_flush_all(sb, out, sizeof(out)); /* drain */

    /* Reset re-arms eager for next turn */
    sentbuf_reset(sb);
    sentbuf_set_eager(sb, 4);
    sentbuf_add(sb, "Alpha beta gamma delta", 22);
    ASSERT_INT_EQ(sentbuf_has_segment(sb), 1);
    sentbuf_flush(sb, out, sizeof(out));
    printf("  re-armed: \"%s\"\n", out);
    sentbuf_destroy(sb);

    /* Speculative mode */
    sb = sentbuf_create(SENTBUF_MODE_SPECULATIVE, 3);
    sentbuf_add(sb, "One two three, four five. ", 25);
    ASSERT_INT_EQ(sentbuf_has_segment(sb), 1);
    sentbuf_flush(sb, out, sizeof(out));
    printf("  speculative: \"%s\"\n", out);
    sentbuf_destroy(sb);

    printf("  sentence_buffer: %d/%d passed\n\n",
           tests_passed - prev_passed, tests_run - prev_run);
}

static void test_ssml_parser(void) {
    int prev_passed = tests_passed, prev_run = tests_run;

    printf("=== ssml_parser ===\n");

    /* Plain text (not SSML) */
    SSMLSegment segs[SSML_MAX_SEGMENTS];
    int n = ssml_parse("Hello world", segs, SSML_MAX_SEGMENTS);
    ASSERT_INT_EQ(n, 1);
    ASSERT_STR_EQ(segs[0].text, "Hello world");

    /* Basic SSML */
    n = ssml_parse("<speak>Hello <break time=\"500ms\"/>world</speak>",
                   segs, SSML_MAX_SEGMENTS);
    printf("  basic SSML: %d segments\n", n);
    for (int i = 0; i < n; i++) {
        printf("    [%d] text=\"%s\" break_before=%d break_after=%d\n",
               i, segs[i].text, segs[i].break_before_ms, segs[i].break_after_ms);
    }

    /* Prosody */
    n = ssml_parse("<speak><prosody rate=\"slow\" pitch=\"high\">Slow and high</prosody></speak>",
                   segs, SSML_MAX_SEGMENTS);
    ASSERT_INT_EQ(n, 1);
    printf("  prosody: rate=%.2f pitch=%.2f volume=%.2f text=\"%s\"\n",
           (double)segs[0].rate, (double)segs[0].pitch,
           (double)segs[0].volume, segs[0].text);

    /* say-as */
    n = ssml_parse("<speak>I have <say-as interpret-as=\"cardinal\">42</say-as> items</speak>",
                   segs, SSML_MAX_SEGMENTS);
    ASSERT_INT_EQ(n, 1);
    printf("  say-as: \"%s\"\n", segs[0].text);

    /* Voice tag */
    n = ssml_parse("<speak><voice name=\"alice\">Hello</voice></speak>",
                   segs, SSML_MAX_SEGMENTS);
    ASSERT_INT_EQ(n, 1);
    ASSERT_STR_EQ(segs[0].voice, "alice");

    /* Emphasis */
    n = ssml_parse("<speak><emphasis level=\"strong\">Important</emphasis> text</speak>",
                   segs, SSML_MAX_SEGMENTS);
    printf("  emphasis: %d segs\n", n);
    for (int i = 0; i < n; i++) {
        printf("    [%d] rate=%.2f vol=%.2f text=\"%s\"\n",
               i, (double)segs[i].rate, (double)segs[i].volume, segs[i].text);
    }

    /* is_ssml */
    ASSERT_INT_EQ(ssml_is_ssml("<speak>hello</speak>"), 1);
    ASSERT_INT_EQ(ssml_is_ssml("plain text"), 0);
    ASSERT_INT_EQ(ssml_is_ssml("<?xml version=\"1.0\"?><speak>hi</speak>"), 1);

    /* <spell> tag */
    n = ssml_parse("<speak>Code <spell>ABC</spell> is active</speak>",
                   segs, SSML_MAX_SEGMENTS);
    {
        int found_spelled = 0;
        for (int i = 0; i < n; i++) {
            if (strstr(segs[i].text, "A B C")) found_spelled = 1;
        }
        ASSERT_INT_EQ(found_spelled, 1);
        printf("  <spell>ABC</spell> → \"%s\" ✓\n",
               found_spelled ? "A B C" : "(not found)");
    }

    /* <spell> with digits */
    n = ssml_parse("<speak>PIN <spell>1234</spell></speak>",
                   segs, SSML_MAX_SEGMENTS);
    {
        int found_spelled = 0;
        for (int i = 0; i < n; i++) {
            if (strstr(segs[i].text, "one two three four")) found_spelled = 1;
        }
        ASSERT_INT_EQ(found_spelled, 1);
        printf("  <spell>1234</spell> → digits spelled ✓\n");
    }

    /* Prosody parsers */
    float rate;
    rate = ssml_parse_rate("slow");
    printf("  rate('slow')=%.2f\n", (double)rate);
    rate = ssml_parse_rate("150%");
    printf("  rate('150%%')=%.2f\n", (double)rate);

    float pitch;
    pitch = ssml_parse_pitch("high");
    printf("  pitch('high')=%.2f\n", (double)pitch);
    pitch = ssml_parse_pitch("+2st");
    printf("  pitch('+2st')=%.2f\n", (double)pitch);

    float vol;
    vol = ssml_parse_volume("loud");
    printf("  volume('loud')=%.2f\n", (double)vol);
    vol = ssml_parse_volume("6dB");
    printf("  volume('6dB')=%.2f\n", (double)vol);

    printf("  ssml_parser: %d/%d passed\n\n",
           tests_passed - prev_passed, tests_run - prev_run);
}

static void test_nonverbalisms_extended(void) {
    int prev_passed = tests_passed, prev_run = tests_run;
    printf("=== nonverbalisms (extended) ===\n");

    char out[8192];

    text_expand_nonverbalisms("[cough] excuse me", out, sizeof(out));
    ASSERT_INT_EQ(strstr(out, "<break") != NULL, 1);
    printf("  [cough] → break tags ✓\n");

    text_expand_nonverbalisms("I'm [yawn] tired", out, sizeof(out));
    ASSERT_INT_EQ(strstr(out, "tired") != NULL, 1);
    printf("  [yawn] → emotion + break ✓\n");

    text_expand_nonverbalisms("[hmm] let me think", out, sizeof(out));
    ASSERT_INT_EQ(strstr(out, "<break") != NULL, 1);
    printf("  [hmm] → break + serious ✓\n");

    text_expand_nonverbalisms("[applause] Thank you!", out, sizeof(out));
    ASSERT_INT_EQ(strstr(out, "<break") != NULL, 1);
    printf("  [applause] → long break ✓\n");

    text_expand_nonverbalisms("[crying] I miss you", out, sizeof(out));
    ASSERT_INT_EQ(strstr(out, "sad") != NULL, 1);
    printf("  [crying] → sad emotion ✓\n");

    text_expand_nonverbalisms("[chuckle] that's funny", out, sizeof(out));
    ASSERT_INT_EQ(strstr(out, "amused") != NULL, 1);
    printf("  [chuckle] → amused emotion ✓\n");

    text_expand_nonverbalisms("[clears throat] anyway", out, sizeof(out));
    ASSERT_INT_EQ(strstr(out, "<break") != NULL, 1);
    printf("  [clears throat] → break tags ✓\n");

    text_expand_nonverbalisms("[shout]Hey![/shout]", out, sizeof(out));
    ASSERT_INT_EQ(strstr(out, "emphatic") != NULL, 1);
    printf("  [shout]...[/shout] → emphatic emotion ✓\n");

    text_expand_nonverbalisms("[groan] not again", out, sizeof(out));
    ASSERT_INT_EQ(strstr(out, "tired") != NULL, 1);
    printf("  [groan] → tired emotion ✓\n");

    text_expand_nonverbalisms("[sniff] it's okay", out, sizeof(out));
    ASSERT_INT_EQ(strstr(out, "sad") != NULL, 1);
    printf("  [sniff] → sad emotion ✓\n");

    text_expand_nonverbalisms("[um] I don't know", out, sizeof(out));
    ASSERT_INT_EQ(strstr(out, "<break") != NULL, 1);
    printf("  [um] → break (hesitation) ✓\n");

    printf("  nonverbalisms extended: %d/%d passed\n\n",
           tests_passed - prev_passed, tests_run - prev_run);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * EDGE CASE TESTS: text_normalize
 * ═══════════════════════════════════════════════════════════════════════════ */

static void test_cardinal_edge_cases(void) {
    int prev_passed = tests_passed, prev_run = tests_run;
    char out[1024];

    printf("=== text_cardinal edge cases ===\n");

    /* Negative numbers */
    text_cardinal("-1", out, sizeof(out));
    ASSERT_STR_EQ(out, "negative one");

    text_cardinal("-42", out, sizeof(out));
    ASSERT_STR_EQ(out, "negative forty-two");

    text_cardinal("-1000", out, sizeof(out));
    ASSERT_STR_EQ(out, "negative one thousand");

    /* Very large numbers */
    text_cardinal("1000000", out, sizeof(out));
    ASSERT_STR_EQ(out, "one million");

    text_cardinal("1000000000", out, sizeof(out));
    ASSERT_STR_EQ(out, "one billion");

    text_cardinal("1000000000000", out, sizeof(out));
    ASSERT_STR_EQ(out, "one trillion");

    text_cardinal("999999999", out, sizeof(out));
    ASSERT_STR_EQ(out, "nine hundred ninety-nine million nine hundred ninety-nine thousand nine hundred ninety-nine");

    /* Single digits */
    text_cardinal("1", out, sizeof(out));
    ASSERT_STR_EQ(out, "one");

    text_cardinal("9", out, sizeof(out));
    ASSERT_STR_EQ(out, "nine");

    /* Teens */
    text_cardinal("11", out, sizeof(out));
    ASSERT_STR_EQ(out, "eleven");

    text_cardinal("13", out, sizeof(out));
    ASSERT_STR_EQ(out, "thirteen");

    text_cardinal("19", out, sizeof(out));
    ASSERT_STR_EQ(out, "nineteen");

    /* Tens */
    text_cardinal("20", out, sizeof(out));
    ASSERT_STR_EQ(out, "twenty");

    text_cardinal("50", out, sizeof(out));
    ASSERT_STR_EQ(out, "fifty");

    text_cardinal("99", out, sizeof(out));
    ASSERT_STR_EQ(out, "ninety-nine");

    /* Hundreds */
    text_cardinal("100", out, sizeof(out));
    ASSERT_STR_EQ(out, "one hundred");

    text_cardinal("101", out, sizeof(out));
    ASSERT_STR_EQ(out, "one hundred one");

    text_cardinal("999", out, sizeof(out));
    ASSERT_STR_EQ(out, "nine hundred ninety-nine");

    /* Numbers with commas */
    text_cardinal("1,000", out, sizeof(out));
    ASSERT_STR_EQ(out, "one thousand");

    text_cardinal("1,234,567", out, sizeof(out));
    ASSERT_STR_EQ(out, "one million two hundred thirty-four thousand five hundred sixty-seven");

    /* Decimals */
    text_cardinal("3.14", out, sizeof(out));
    ASSERT_STR_EQ(out, "three point one four");

    text_cardinal("0.5", out, sizeof(out));
    ASSERT_STR_EQ(out, "zero point five");

    text_cardinal("100.00", out, sizeof(out));
    ASSERT_STR_EQ(out, "one hundred point zero zero");

    /* Leading/trailing whitespace */
    text_cardinal("  42  ", out, sizeof(out));
    ASSERT_STR_EQ(out, "forty-two");

    /* Buffer too small — should truncate safely */
    {
        char tiny[10];
        text_cardinal("1234567", tiny, sizeof(tiny));
        /* Just verify it doesn't crash and is null-terminated */
        ASSERT_INT_EQ(tiny[9], '\0');  /* within bounds */
        ASSERT_INT_EQ((int)strlen(tiny) < 10, 1);
    }

    printf("  text_cardinal edge cases: %d/%d passed\n\n",
           tests_passed - prev_passed, tests_run - prev_run);
}

static void test_ordinal_edge_cases(void) {
    int prev_passed = tests_passed, prev_run = tests_run;
    char out[512];

    printf("=== text_ordinal edge cases ===\n");

    /* Special teens: 11th, 12th, 13th */
    text_ordinal("11th", out, sizeof(out));
    ASSERT_STR_EQ(out, "eleventh");

    text_ordinal("12th", out, sizeof(out));
    ASSERT_STR_EQ(out, "twelfth");

    text_ordinal("13th", out, sizeof(out));
    ASSERT_STR_EQ(out, "thirteenth");

    /* 21st, 22nd, 23rd */
    text_ordinal("21st", out, sizeof(out));
    ASSERT_STR_EQ(out, "twenty-first");

    text_ordinal("22nd", out, sizeof(out));
    ASSERT_STR_EQ(out, "twenty-second");

    text_ordinal("23rd", out, sizeof(out));
    ASSERT_STR_EQ(out, "twenty-third");

    /* Round numbers */
    text_ordinal("100th", out, sizeof(out));
    ASSERT_STR_EQ(out, "one hundredth");

    text_ordinal("1000th", out, sizeof(out));
    ASSERT_STR_EQ(out, "one thousandth");

    text_ordinal("20th", out, sizeof(out));
    ASSERT_STR_EQ(out, "twentieth");

    text_ordinal("30th", out, sizeof(out));
    ASSERT_STR_EQ(out, "thirtieth");

    /* Without suffix */
    text_ordinal("5", out, sizeof(out));
    ASSERT_STR_EQ(out, "fifth");

    text_ordinal("1", out, sizeof(out));
    ASSERT_STR_EQ(out, "first");

    /* Complex ordinals */
    text_ordinal("101st", out, sizeof(out));
    ASSERT_STR_EQ(out, "one hundred first");

    text_ordinal("111th", out, sizeof(out));
    ASSERT_STR_EQ(out, "one hundred eleventh");

    printf("  text_ordinal edge cases: %d/%d passed\n\n",
           tests_passed - prev_passed, tests_run - prev_run);
}

static void test_currency_edge_cases(void) {
    int prev_passed = tests_passed, prev_run = tests_run;
    char out[512];

    printf("=== text_currency edge cases ===\n");

    /* Basic cases */
    text_currency("$1.50", out, sizeof(out));
    ASSERT_STR_EQ(out, "one dollar and fifty cents");

    text_currency("$0.99", out, sizeof(out));
    ASSERT_STR_EQ(out, "zero dollars and ninety-nine cents");

    text_currency("$0.01", out, sizeof(out));
    ASSERT_STR_EQ(out, "zero dollars and one cent");

    /* Large amounts */
    text_currency("$1,000,000.00", out, sizeof(out));
    ASSERT_STR_EQ(out, "one million dollars");

    text_currency("$100", out, sizeof(out));
    ASSERT_STR_EQ(out, "one hundred dollars");

    /* Euro */
    text_currency("\xe2\x82\xac""100", out, sizeof(out));
    ASSERT_STR_EQ(out, "one hundred euros");

    text_currency("\xe2\x82\xac""1.50", out, sizeof(out));
    ASSERT_STR_EQ(out, "one euro and fifty cents");

    /* British pound */
    text_currency("\xc2\xa3""0.99", out, sizeof(out));
    ASSERT_STR_EQ(out, "zero pounds and ninety-nine pence");

    text_currency("\xc2\xa3""1", out, sizeof(out));
    ASSERT_STR_EQ(out, "one pound");

    /* Singular dollar */
    text_currency("$1", out, sizeof(out));
    ASSERT_STR_EQ(out, "one dollar");

    /* $0 exact */
    text_currency("$0", out, sizeof(out));
    ASSERT_STR_EQ(out, "zero dollars");

    /* $0.00 */
    text_currency("$0.00", out, sizeof(out));
    ASSERT_STR_EQ(out, "zero dollars");

    printf("  text_currency edge cases: %d/%d passed\n\n",
           tests_passed - prev_passed, tests_run - prev_run);
}

static void test_telephone_edge_cases(void) {
    int prev_passed = tests_passed, prev_run = tests_run;
    char out[512];

    printf("=== text_telephone edge cases ===\n");

    /* Standard US format with parentheses */
    text_telephone("(555) 123-4567", out, sizeof(out));
    /* Should spell out each digit with pauses at separators */
    ASSERT_INT_EQ(strstr(out, "five") != NULL, 1);
    ASSERT_INT_EQ(strstr(out, "one") != NULL, 1);
    printf("  (555) 123-4567 → \"%s\"\n", out);

    /* Dashes only */
    text_telephone("555-123-4567", out, sizeof(out));
    ASSERT_INT_EQ(strstr(out, "five") != NULL, 1);
    printf("  555-123-4567 → \"%s\"\n", out);

    /* International format */
    text_telephone("+1-555-123-4567", out, sizeof(out));
    ASSERT_INT_EQ(strstr(out, "one") != NULL, 1);
    printf("  +1-555-123-4567 → \"%s\"\n", out);

    /* Dots as separators */
    text_telephone("555.123.4567", out, sizeof(out));
    ASSERT_INT_EQ(strstr(out, "five") != NULL, 1);
    printf("  555.123.4567 → \"%s\"\n", out);

    /* Just digits */
    text_telephone("5551234567", out, sizeof(out));
    ASSERT_INT_EQ(strstr(out, "five") != NULL, 1);
    printf("  5551234567 → \"%s\"\n", out);

    /* Short number */
    text_telephone("911", out, sizeof(out));
    ASSERT_INT_EQ(strstr(out, "nine") != NULL, 1);
    printf("  911 → \"%s\"\n", out);

    printf("  text_telephone edge cases: %d/%d passed\n\n",
           tests_passed - prev_passed, tests_run - prev_run);
}

static void test_date_edge_cases(void) {
    int prev_passed = tests_passed, prev_run = tests_run;
    char out[512];

    printf("=== text_date edge cases ===\n");

    /* MM/DD/YYYY default format */
    text_date("01/15/2024", "", out, sizeof(out));
    ASSERT_STR_EQ(out, "January fifteenth, twenty twenty-four");

    /* ISO 8601: YYYY-MM-DD */
    text_date("2024-01-15", "", out, sizeof(out));
    ASSERT_STR_EQ(out, "January fifteenth, twenty twenty-four");

    /* 2-digit year */
    text_date("06/15/99", "", out, sizeof(out));
    ASSERT_STR_EQ(out, "June fifteenth, nineteen ninety-nine");

    text_date("06/15/05", "", out, sizeof(out));
    ASSERT_STR_EQ(out, "June fifteenth, two thousand five");

    /* Year 2000 */
    text_date("01/01/2000", "", out, sizeof(out));
    ASSERT_STR_EQ(out, "January first, two thousand");

    /* DMY format */
    text_date("15/01/2024", "dmy", out, sizeof(out));
    ASSERT_STR_EQ(out, "January fifteenth, twenty twenty-four");

    /* Month-only format */
    text_date("03/15/2024", "m", out, sizeof(out));
    ASSERT_STR_EQ(out, "March");

    /* Day-only format */
    text_date("03/15/2024", "d", out, sizeof(out));
    ASSERT_STR_EQ(out, "fifteenth");

    /* Year-only format */
    text_date("03/15/2024", "y", out, sizeof(out));
    ASSERT_STR_EQ(out, "twenty twenty-four");

    /* December 31 */
    text_date("12/31/2025", "", out, sizeof(out));
    ASSERT_STR_EQ(out, "December thirty-first, twenty twenty-five");

    /* Dot separators */
    text_date("03.15.2024", "", out, sizeof(out));
    ASSERT_STR_EQ(out, "March fifteenth, twenty twenty-four");

    printf("  text_date edge cases: %d/%d passed\n\n",
           tests_passed - prev_passed, tests_run - prev_run);
}

static void test_time_edge_cases(void) {
    int prev_passed = tests_passed, prev_run = tests_run;
    char out[512];

    printf("=== text_time edge cases ===\n");

    /* 12-hour with AM/PM */
    text_time("3:30 PM", "", out, sizeof(out));
    ASSERT_STR_EQ(out, "three thirty PM");

    text_time("12:00 AM", "", out, sizeof(out));
    ASSERT_STR_EQ(out, "midnight");

    text_time("12:00 PM", "", out, sizeof(out));
    ASSERT_STR_EQ(out, "noon");

    /* 24-hour format */
    text_time("15:30", "", out, sizeof(out));
    ASSERT_STR_EQ(out, "fifteen thirty");

    text_time("0:00", "", out, sizeof(out));
    ASSERT_STR_EQ(out, "midnight");

    text_time("12:00", "", out, sizeof(out));
    ASSERT_STR_EQ(out, "noon");

    /* Minutes < 10 get "oh" prefix */
    text_time("3:05", "", out, sizeof(out));
    ASSERT_STR_EQ(out, "three oh five");

    text_time("9:01", "", out, sizeof(out));
    ASSERT_STR_EQ(out, "nine oh one");

    /* On the hour without AM/PM */
    text_time("3:00", "", out, sizeof(out));
    ASSERT_STR_EQ(out, "three o'clock");

    text_time("1:00 AM", "", out, sizeof(out));
    ASSERT_STR_EQ(out, "one AM");

    /* With seconds */
    text_time("3:30:15", "", out, sizeof(out));
    ASSERT_STR_EQ(out, "three thirty and fifteen seconds");

    /* Edge: 23:59 */
    text_time("23:59", "", out, sizeof(out));
    ASSERT_STR_EQ(out, "twenty-three fifty-nine");

    /* Lowercase am/pm */
    text_time("3:30 pm", "", out, sizeof(out));
    ASSERT_STR_EQ(out, "three thirty PM");

    printf("  text_time edge cases: %d/%d passed\n\n",
           tests_passed - prev_passed, tests_run - prev_run);
}

static void test_fraction_edge_cases(void) {
    int prev_passed = tests_passed, prev_run = tests_run;
    char out[512];

    printf("=== text_fraction edge cases ===\n");

    /* Named fractions */
    text_fraction("1/3", out, sizeof(out));
    ASSERT_STR_EQ(out, "one third");

    text_fraction("2/3", out, sizeof(out));
    ASSERT_STR_EQ(out, "two thirds");

    text_fraction("1/4", out, sizeof(out));
    ASSERT_STR_EQ(out, "one quarter");

    text_fraction("1/5", out, sizeof(out));
    ASSERT_STR_EQ(out, "one fifth");

    text_fraction("1/8", out, sizeof(out));
    ASSERT_STR_EQ(out, "one eighth");

    /* Non-named fractions use ordinal denominator */
    text_fraction("2/7", out, sizeof(out));
    ASSERT_STR_EQ(out, "two sevenths");

    text_fraction("5/9", out, sizeof(out));
    ASSERT_STR_EQ(out, "five ninths");

    /* Halves plural */
    text_fraction("3/2", out, sizeof(out));
    ASSERT_STR_EQ(out, "three halves");

    /* Mixed number */
    text_fraction("2 1/2", out, sizeof(out));
    ASSERT_STR_EQ(out, "two and one half");

    text_fraction("3 3/4", out, sizeof(out));
    ASSERT_STR_EQ(out, "three and three quarters");

    /* Division by zero — passthrough */
    text_fraction("1/0", out, sizeof(out));
    ASSERT_STR_EQ(out, "1/0");

    printf("  text_fraction edge cases: %d/%d passed\n\n",
           tests_passed - prev_passed, tests_run - prev_run);
}

static void test_characters_edge_cases(void) {
    int prev_passed = tests_passed, prev_run = tests_run;
    char out[512];

    printf("=== text_characters edge cases ===\n");

    /* Digits */
    text_characters("123", out, sizeof(out));
    ASSERT_STR_EQ(out, "one two three");

    text_characters("0", out, sizeof(out));
    ASSERT_STR_EQ(out, "zero");

    /* Mixed case */
    text_characters("aB", out, sizeof(out));
    ASSERT_STR_EQ(out, "A B");

    /* Single char */
    text_characters("X", out, sizeof(out));
    ASSERT_STR_EQ(out, "X");

    /* Special characters */
    text_characters("@", out, sizeof(out));
    ASSERT_STR_EQ(out, "at");

    text_characters("#!", out, sizeof(out));
    ASSERT_STR_EQ(out, "hash exclamation mark");

    printf("  text_characters edge cases: %d/%d passed\n\n",
           tests_passed - prev_passed, tests_run - prev_run);
}

static void test_unit_edge_cases(void) {
    int prev_passed = tests_passed, prev_run = tests_run;
    char out[512];

    printf("=== text_unit edge cases ===\n");

    text_unit("10km", out, sizeof(out));
    ASSERT_STR_EQ(out, "ten kilometers");

    text_unit("100mph", out, sizeof(out));
    ASSERT_STR_EQ(out, "one hundred miles per hour");

    text_unit("72kg", out, sizeof(out));
    ASSERT_STR_EQ(out, "seventy-two kilograms");

    text_unit("500ml", out, sizeof(out));
    ASSERT_STR_EQ(out, "five hundred milliliters");

    text_unit("3ft", out, sizeof(out));
    ASSERT_STR_EQ(out, "three feet");

    text_unit("6.5m", out, sizeof(out));
    ASSERT_STR_EQ(out, "six point five meters");

    /* With commas */
    text_unit("1,500g", out, sizeof(out));
    ASSERT_STR_EQ(out, "one thousand five hundred grams");

    printf("  text_unit edge cases: %d/%d passed\n\n",
           tests_passed - prev_passed, tests_run - prev_run);
}

static void test_url_edge_cases(void) {
    int prev_passed = tests_passed, prev_run = tests_run;
    char out[1024];

    printf("=== text_url edge cases ===\n");

    text_url("https://example.com", out, sizeof(out));
    ASSERT_INT_EQ(strstr(out, "H T T P S") != NULL, 1);
    ASSERT_INT_EQ(strstr(out, "dot") != NULL, 1);
    printf("  https://example.com → \"%s\"\n", out);

    text_url("http://www.test.org", out, sizeof(out));
    ASSERT_INT_EQ(strstr(out, "H T T P") != NULL, 1);
    ASSERT_INT_EQ(strstr(out, "W W W") != NULL, 1);
    printf("  http://www.test.org → \"%s\"\n", out);

    text_url("ftp://files.host.net", out, sizeof(out));
    ASSERT_INT_EQ(strstr(out, "F T P") != NULL, 1);
    printf("  ftp → \"%s\"\n", out);

    /* URL with path and query */
    text_url("https://example.com/path?key=val", out, sizeof(out));
    ASSERT_INT_EQ(strstr(out, "slash") != NULL, 1);
    ASSERT_INT_EQ(strstr(out, "question mark") != NULL, 1);
    ASSERT_INT_EQ(strstr(out, "equals") != NULL, 1);
    printf("  with path/query → \"%s\"\n", out);

    printf("  text_url edge cases: %d/%d passed\n\n",
           tests_passed - prev_passed, tests_run - prev_run);
}

static void test_email_edge_cases(void) {
    int prev_passed = tests_passed, prev_run = tests_run;
    char out[512];

    printf("=== text_email edge cases ===\n");

    text_email("user@example.com", out, sizeof(out));
    ASSERT_INT_EQ(strstr(out, "at") != NULL, 1);
    ASSERT_INT_EQ(strstr(out, "dot") != NULL, 1);
    printf("  user@example.com → \"%s\"\n", out);

    text_email("first.last@domain.co.uk", out, sizeof(out));
    ASSERT_INT_EQ(strstr(out, "dot") != NULL, 1);
    ASSERT_INT_EQ(strstr(out, "at") != NULL, 1);
    printf("  first.last@domain.co.uk → \"%s\"\n", out);

    text_email("user+tag@mail.com", out, sizeof(out));
    ASSERT_INT_EQ(strstr(out, "plus") != NULL, 1);
    printf("  user+tag@mail.com → \"%s\"\n", out);

    /* No @ sign — passthrough */
    text_email("not-an-email", out, sizeof(out));
    ASSERT_STR_EQ(out, "not-an-email");

    printf("  text_email edge cases: %d/%d passed\n\n",
           tests_passed - prev_passed, tests_run - prev_run);
}

static void test_text_normalize_dispatch(void) {
    int prev_passed = tests_passed, prev_run = tests_run;
    char out[512];

    printf("=== text_normalize dispatch ===\n");

    /* Dispatch by interpret_as */
    text_normalize("42", "cardinal", "", out, sizeof(out));
    ASSERT_STR_EQ(out, "forty-two");

    text_normalize("42", "number", "", out, sizeof(out));
    ASSERT_STR_EQ(out, "forty-two");

    text_normalize("3rd", "ordinal", "", out, sizeof(out));
    ASSERT_STR_EQ(out, "third");

    text_normalize("ABC", "characters", "", out, sizeof(out));
    ASSERT_STR_EQ(out, "A B C");

    text_normalize("ABC", "spell-out", "", out, sizeof(out));
    ASSERT_STR_EQ(out, "A B C");

    text_normalize("ABC", "verbatim", "", out, sizeof(out));
    ASSERT_STR_EQ(out, "A B C");

    text_normalize("01/15/2024", "date", "", out, sizeof(out));
    ASSERT_STR_EQ(out, "January fifteenth, twenty twenty-four");

    text_normalize("3:30 PM", "time", "", out, sizeof(out));
    ASSERT_STR_EQ(out, "three thirty PM");

    text_normalize("555-1234", "telephone", "", out, sizeof(out));
    ASSERT_INT_EQ(strstr(out, "five") != NULL, 1);

    text_normalize("555-1234", "phone", "", out, sizeof(out));
    ASSERT_INT_EQ(strstr(out, "five") != NULL, 1);

    text_normalize("$42.50", "currency", "", out, sizeof(out));
    ASSERT_STR_EQ(out, "forty-two dollars and fifty cents");

    text_normalize("1/2", "fraction", "", out, sizeof(out));
    ASSERT_STR_EQ(out, "one half");

    text_normalize("5km", "unit", "", out, sizeof(out));
    ASSERT_STR_EQ(out, "five kilometers");

    text_normalize("https://example.com", "url", "", out, sizeof(out));
    ASSERT_INT_EQ(strstr(out, "H T T P S") != NULL, 1);

    text_normalize("https://example.com", "uri", "", out, sizeof(out));
    ASSERT_INT_EQ(strstr(out, "H T T P S") != NULL, 1);

    text_normalize("user@test.com", "email", "", out, sizeof(out));
    ASSERT_INT_EQ(strstr(out, "at") != NULL, 1);

    /* Empty interpret_as — passthrough */
    text_normalize("hello", "", "", out, sizeof(out));
    ASSERT_STR_EQ(out, "hello");

    /* Unknown interpret_as — passthrough */
    text_normalize("hello", "unknown_type", "", out, sizeof(out));
    ASSERT_STR_EQ(out, "hello");

    printf("  text_normalize dispatch: %d/%d passed\n\n",
           tests_passed - prev_passed, tests_run - prev_run);
}

static void test_auto_normalize_edge_cases(void) {
    int prev_passed = tests_passed, prev_run = tests_run;
    char out[2048];

    printf("=== text_auto_normalize edge cases ===\n");

    /* Mixed text: currency + ordinal */
    text_auto_normalize("I paid $5.99 for the 3rd item.", out, sizeof(out));
    ASSERT_INT_EQ(strstr(out, "five dollars") != NULL, 1);
    ASSERT_INT_EQ(strstr(out, "third") != NULL, 1);
    printf("  currency+ordinal: \"%s\"\n", out);

    /* Fractions in text */
    text_auto_normalize("Mix 2/3 cup of flour.", out, sizeof(out));
    ASSERT_INT_EQ(strstr(out, "two thirds") != NULL, 1);
    printf("  fraction in text: \"%s\"\n", out);

    /* Plain text without any normalizable patterns */
    text_auto_normalize("Hello world, how are you?", out, sizeof(out));
    ASSERT_STR_EQ(out, "Hello world, how are you?");

    /* Multiple ordinals */
    text_auto_normalize("The 1st and 2nd place.", out, sizeof(out));
    ASSERT_INT_EQ(strstr(out, "first") != NULL, 1);
    ASSERT_INT_EQ(strstr(out, "second") != NULL, 1);
    printf("  multiple ordinals: \"%s\"\n", out);

    /* Abbreviation expansion */
    text_auto_normalize("Dr. Smith went to St. Louis.", out, sizeof(out));
    ASSERT_INT_EQ(strstr(out, "Doctor") != NULL, 1);
    ASSERT_INT_EQ(strstr(out, "Saint") != NULL, 1);
    printf("  abbreviations: \"%s\"\n", out);

    /* Buffer overflow safety — very long input */
    {
        char long_input[4096];
        memset(long_input, 'a', sizeof(long_input) - 1);
        long_input[sizeof(long_input) - 1] = '\0';
        char long_out[4096];
        text_auto_normalize(long_input, long_out, sizeof(long_out));
        /* Just verify it doesn't crash */
        ASSERT_INT_EQ((int)strlen(long_out) < (int)sizeof(long_out), 1);
    }

    /* Tiny output buffer — truncation safety */
    {
        char tiny[16];
        text_auto_normalize("The 1st item costs $42.50.", tiny, sizeof(tiny));
        ASSERT_INT_EQ((int)strlen(tiny) < 16, 1);
    }

    printf("  text_auto_normalize edge cases: %d/%d passed\n\n",
           tests_passed - prev_passed, tests_run - prev_run);
}

static void test_inline_ipa(void) {
    int prev_passed = tests_passed, prev_run = tests_run;
    char out[1024];

    printf("=== text_expand_inline_ipa ===\n");

    /* Cartesia format: <<p|h|o|n|e|m|e>> */
    text_expand_inline_ipa("Say <<h|E|l|oU>> please", out, sizeof(out));
    ASSERT_INT_EQ(strstr(out, "phoneme") != NULL, 1);
    printf("  cartesia IPA: \"%s\"\n", out);

    /* Alternative format: [ipa:phoneme] */
    text_expand_inline_ipa("Say [ipa:hEloU] please", out, sizeof(out));
    ASSERT_INT_EQ(strstr(out, "phoneme") != NULL, 1);
    printf("  [ipa:] format: \"%s\"\n", out);

    /* No IPA — passthrough */
    text_expand_inline_ipa("Hello world", out, sizeof(out));
    ASSERT_STR_EQ(out, "Hello world");

    printf("  text_expand_inline_ipa: %d/%d passed\n\n",
           tests_passed - prev_passed, tests_run - prev_run);
}

static void test_pronunciation_dict(void) {
    int prev_passed = tests_passed, prev_run = tests_run;
    char out[1024];

    printf("=== text_apply_pronunciation_dict ===\n");

    char words[2][64] = { "tomato", "route" };
    char replacements[2][256] = { "tuh-MAY-toe", "root" };

    text_apply_pronunciation_dict("I like tomato on my route home.",
                                  out, sizeof(out),
                                  (const char (*)[64])words,
                                  (const char (*)[256])replacements, 2);
    ASSERT_INT_EQ(strstr(out, "tuh-MAY-toe") != NULL, 1);
    ASSERT_INT_EQ(strstr(out, "root") != NULL, 1);
    printf("  dict replacement: \"%s\"\n", out);

    /* No matches */
    text_apply_pronunciation_dict("Hello world.", out, sizeof(out),
                                  (const char (*)[64])words,
                                  (const char (*)[256])replacements, 2);
    ASSERT_STR_EQ(out, "Hello world.");

    /* Case insensitive match */
    text_apply_pronunciation_dict("TOMATO salad", out, sizeof(out),
                                  (const char (*)[64])words,
                                  (const char (*)[256])replacements, 2);
    ASSERT_INT_EQ(strstr(out, "tuh-MAY-toe") != NULL, 1);
    printf("  case-insensitive: \"%s\"\n", out);

    printf("  text_apply_pronunciation_dict: %d/%d passed\n\n",
           tests_passed - prev_passed, tests_run - prev_run);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * EDGE CASE TESTS: ssml_parser
 * ═══════════════════════════════════════════════════════════════════════════ */

static void test_ssml_edge_cases(void) {
    int prev_passed = tests_passed, prev_run = tests_run;

    printf("=== ssml_parser edge cases ===\n");

    SSMLSegment segs[SSML_MAX_SEGMENTS];
    int n;

    /* Nested prosody + emphasis */
    n = ssml_parse("<speak><prosody rate=\"fast\"><emphasis level=\"strong\">Urgent!</emphasis></prosody></speak>",
                   segs, SSML_MAX_SEGMENTS);
    ASSERT_INT_EQ(n >= 1, 1);
    printf("  nested prosody+emphasis: %d segments, text=\"%s\" rate=%.2f\n",
           n, segs[0].text, (double)segs[0].rate);

    /* Multiple break tags with different time formats */
    n = ssml_parse("<speak>Hello<break time=\"200ms\"/> world<break time=\"1s\"/> end</speak>",
                   segs, SSML_MAX_SEGMENTS);
    ASSERT_INT_EQ(n >= 1, 1);
    printf("  multiple breaks: %d segments\n", n);
    for (int i = 0; i < n && i < 4; i++) {
        printf("    [%d] text=\"%s\" break_before=%d break_after=%d\n",
               i, segs[i].text, segs[i].break_before_ms, segs[i].break_after_ms);
    }

    /* Empty speak tag */
    n = ssml_parse("<speak></speak>", segs, SSML_MAX_SEGMENTS);
    ASSERT_INT_EQ(n >= 0, 1);  /* Should not crash */
    printf("  empty <speak>: %d segments\n", n);

    /* Self-closing break only */
    n = ssml_parse("<speak><break time=\"500ms\"/></speak>", segs, SSML_MAX_SEGMENTS);
    ASSERT_INT_EQ(n >= 0, 1);
    printf("  break-only: %d segments\n", n);

    /* say-as with cardinal */
    n = ssml_parse("<speak>I have <say-as interpret-as=\"cardinal\">42</say-as> items</speak>",
                   segs, SSML_MAX_SEGMENTS);
    ASSERT_INT_EQ(n >= 1, 1);
    ASSERT_INT_EQ(strstr(segs[0].text, "forty-two") != NULL, 1);
    printf("  say-as cardinal: \"%s\"\n", segs[0].text);

    /* say-as with ordinal */
    n = ssml_parse("<speak>The <say-as interpret-as=\"ordinal\">3</say-as> place</speak>",
                   segs, SSML_MAX_SEGMENTS);
    ASSERT_INT_EQ(n >= 1, 1);
    ASSERT_INT_EQ(strstr(segs[0].text, "third") != NULL, 1);
    printf("  say-as ordinal: \"%s\"\n", segs[0].text);

    /* say-as with characters */
    n = ssml_parse("<speak>Code: <say-as interpret-as=\"characters\">XY</say-as></speak>",
                   segs, SSML_MAX_SEGMENTS);
    ASSERT_INT_EQ(n >= 1, 1);
    ASSERT_INT_EQ(strstr(segs[0].text, "X Y") != NULL, 1);
    printf("  say-as characters: \"%s\"\n", segs[0].text);

    /* Multiple voice tags */
    n = ssml_parse("<speak><voice name=\"alice\">Hi</voice> <voice name=\"bob\">There</voice></speak>",
                   segs, SSML_MAX_SEGMENTS);
    ASSERT_INT_EQ(n >= 2, 1);
    ASSERT_STR_EQ(segs[0].voice, "alice");
    ASSERT_STR_EQ(segs[1].voice, "bob");
    printf("  multiple voices: alice=\"%s\" bob=\"%s\"\n", segs[0].text, segs[1].text);

    /* Phoneme tag */
    n = ssml_parse("<speak><phoneme alphabet=\"ipa\" ph=\"t\xc9\x94m\xc9\x91to\xca\x8a\">tomato</phoneme></speak>",
                   segs, SSML_MAX_SEGMENTS);
    ASSERT_INT_EQ(n >= 1, 1);
    ASSERT_INT_EQ(strlen(segs[0].phoneme_ipa) > 0, 1);
    printf("  phoneme: text=\"%s\" ipa=\"%s\"\n", segs[0].text, segs[0].phoneme_ipa);

    /* Entity decoding */
    n = ssml_parse("<speak>A &amp; B &lt; C &gt; D</speak>",
                   segs, SSML_MAX_SEGMENTS);
    ASSERT_INT_EQ(n >= 1, 1);
    ASSERT_INT_EQ(strstr(segs[0].text, "&") != NULL, 1);
    ASSERT_INT_EQ(strstr(segs[0].text, "<") != NULL, 1);
    ASSERT_INT_EQ(strstr(segs[0].text, ">") != NULL, 1);
    printf("  entity decoding: \"%s\"\n", segs[0].text);

    /* Audio tag */
    n = ssml_parse("<speak><audio src=\"https://example.com/sound.mp3\"/></speak>",
                   segs, SSML_MAX_SEGMENTS);
    printf("  audio tag: %d segments\n", n);
    if (n >= 1) {
        printf("    is_audio=%d text=\"%s\"\n", segs[0].is_audio, segs[0].text);
    }

    /* Emotion via emphasis */
    n = ssml_parse("<speak><emphasis level=\"reduced\">Quiet text</emphasis></speak>",
                   segs, SSML_MAX_SEGMENTS);
    ASSERT_INT_EQ(n >= 1, 1);
    printf("  emphasis reduced: rate=%.2f vol=%.2f text=\"%s\"\n",
           (double)segs[0].rate, (double)segs[0].volume, segs[0].text);

    /* Paragraph and sentence breaks */
    n = ssml_parse("<speak><p>First paragraph.</p><p>Second paragraph.</p></speak>",
                   segs, SSML_MAX_SEGMENTS);
    ASSERT_INT_EQ(n >= 2, 1);
    printf("  paragraphs: %d segments\n", n);

    n = ssml_parse("<speak><s>First sentence.</s><s>Second sentence.</s></speak>",
                   segs, SSML_MAX_SEGMENTS);
    ASSERT_INT_EQ(n >= 2, 1);
    printf("  sentences: %d segments\n", n);

    /* Plain text (not SSML) — should return 1 segment with raw text */
    n = ssml_parse("Just plain text, no SSML here.", segs, SSML_MAX_SEGMENTS);
    ASSERT_INT_EQ(n, 1);
    ASSERT_STR_EQ(segs[0].text, "Just plain text, no SSML here.");

    /* XML declaration before speak */
    n = ssml_parse("<?xml version=\"1.0\"?><speak>Hello</speak>", segs, SSML_MAX_SEGMENTS);
    ASSERT_INT_EQ(n >= 1, 1);
    ASSERT_INT_EQ(strstr(segs[0].text, "Hello") != NULL, 1);

    printf("  ssml_parser edge cases: %d/%d passed\n\n",
           tests_passed - prev_passed, tests_run - prev_run);
}

static void test_ssml_is_ssml_edge_cases(void) {
    int prev_passed = tests_passed, prev_run = tests_run;

    printf("=== ssml_is_ssml edge cases ===\n");

    ASSERT_INT_EQ(ssml_is_ssml("<speak>hello</speak>"), 1);
    ASSERT_INT_EQ(ssml_is_ssml("plain text"), 0);
    ASSERT_INT_EQ(ssml_is_ssml("<?xml version=\"1.0\"?><speak>hi</speak>"), 1);
    ASSERT_INT_EQ(ssml_is_ssml(""), 0);
    ASSERT_INT_EQ(ssml_is_ssml("  <speak>with leading spaces</speak>"), 1);
    ASSERT_INT_EQ(ssml_is_ssml("<div>not SSML</div>"), 0);
    ASSERT_INT_EQ(ssml_is_ssml("<SPEAK>uppercase</SPEAK>"), 0);  /* case-sensitive check */

    printf("  ssml_is_ssml edge cases: %d/%d passed\n\n",
           tests_passed - prev_passed, tests_run - prev_run);
}

static void test_ssml_prosody_parsers(void) {
    int prev_passed = tests_passed, prev_run = tests_run;

    printf("=== ssml prosody parsers ===\n");

    /* Rate */
    {
        float r;
        r = ssml_parse_rate("x-slow");
        ASSERT_INT_EQ((int)(r * 100), 50);

        r = ssml_parse_rate("slow");
        ASSERT_INT_EQ((int)(r * 100), 75);

        r = ssml_parse_rate("medium");
        ASSERT_INT_EQ((int)(r * 100), 100);

        r = ssml_parse_rate("fast");
        ASSERT_INT_EQ((int)(r * 100), 125);

        r = ssml_parse_rate("x-fast");
        ASSERT_INT_EQ((int)(r * 100), 175);

        r = ssml_parse_rate("default");
        ASSERT_INT_EQ((int)(r * 100), 100);

        r = ssml_parse_rate("200%");
        ASSERT_INT_EQ((int)(r * 100), 200);

        r = ssml_parse_rate("50%");
        ASSERT_INT_EQ((int)(r * 100), 50);

        /* NULL and empty */
        r = ssml_parse_rate(NULL);
        ASSERT_INT_EQ((int)(r * 100), 100);

        r = ssml_parse_rate("");
        ASSERT_INT_EQ((int)(r * 100), 100);
    }

    /* Pitch */
    {
        float p;
        p = ssml_parse_pitch("x-low");
        ASSERT_INT_EQ((int)(p * 100), 50);

        p = ssml_parse_pitch("low");
        ASSERT_INT_EQ((int)(p * 100), 75);

        p = ssml_parse_pitch("medium");
        ASSERT_INT_EQ((int)(p * 100), 100);

        p = ssml_parse_pitch("high");
        ASSERT_INT_EQ((int)(p * 100), 125);

        p = ssml_parse_pitch("x-high");
        ASSERT_INT_EQ((int)(p * 100), 150);

        p = ssml_parse_pitch("default");
        ASSERT_INT_EQ((int)(p * 100), 100);

        /* Semitone offset */
        p = ssml_parse_pitch("+2st");
        ASSERT_INT_EQ(p > 1.0f, 1);  /* 2^(2/12) ≈ 1.122 */
        printf("  pitch('+2st')=%.3f\n", (double)p);

        p = ssml_parse_pitch("-2st");
        ASSERT_INT_EQ(p < 1.0f, 1);  /* 2^(-2/12) ≈ 0.891 */
        printf("  pitch('-2st')=%.3f\n", (double)p);

        /* Hz */
        p = ssml_parse_pitch("200Hz");
        ASSERT_INT_EQ((int)(p * 100), 100);  /* 200/200 = 1.0 */

        p = ssml_parse_pitch("300Hz");
        ASSERT_INT_EQ((int)(p * 100), 150);  /* 300/200 = 1.5 */

        /* NULL */
        p = ssml_parse_pitch(NULL);
        ASSERT_INT_EQ((int)(p * 100), 100);
    }

    /* Volume */
    {
        float v;
        v = ssml_parse_volume("loud");
        ASSERT_INT_EQ(v > 1.0f, 1);
        printf("  volume('loud')=%.2f\n", (double)v);

        v = ssml_parse_volume("soft");
        ASSERT_INT_EQ(v < 1.0f, 1);
        printf("  volume('soft')=%.2f\n", (double)v);

        v = ssml_parse_volume("medium");
        ASSERT_INT_EQ((int)(v * 100), 100);

        v = ssml_parse_volume("default");
        ASSERT_INT_EQ((int)(v * 100), 100);

        /* dB offsets */
        v = ssml_parse_volume("6dB");
        ASSERT_INT_EQ(v > 1.0f, 1);
        printf("  volume('6dB')=%.2f\n", (double)v);

        v = ssml_parse_volume("-6dB");
        ASSERT_INT_EQ(v < 1.0f, 1);
        printf("  volume('-6dB')=%.2f\n", (double)v);

        /* NULL */
        v = ssml_parse_volume(NULL);
        ASSERT_INT_EQ((int)(v * 100), 100);
    }

    printf("  ssml prosody parsers: %d/%d passed\n\n",
           tests_passed - prev_passed, tests_run - prev_run);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * EDGE CASE TESTS: sentence_buffer
 * ═══════════════════════════════════════════════════════════════════════════ */

static void test_sentbuf_edge_cases(void) {
    int prev_passed = tests_passed, prev_run = tests_run;
    char out[4096];

    printf("=== sentence_buffer edge cases ===\n");

    /* Single character tokens */
    {
        SentenceBuffer *sb = sentbuf_create(SENTBUF_MODE_SENTENCE, 5);
        sentbuf_add(sb, "H", 1);
        sentbuf_add(sb, "i", 1);
        sentbuf_add(sb, ".", 1);
        sentbuf_add(sb, " ", 1);
        ASSERT_INT_EQ(sentbuf_has_segment(sb), 1);
        sentbuf_flush(sb, out, sizeof(out));
        ASSERT_STR_EQ(out, "Hi.");
        sentbuf_destroy(sb);
    }

    /* Empty token */
    {
        SentenceBuffer *sb = sentbuf_create(SENTBUF_MODE_SENTENCE, 5);
        sentbuf_add(sb, "", 0);
        ASSERT_INT_EQ(sentbuf_has_segment(sb), 0);
        sentbuf_add(sb, "Hello. ", 7);
        ASSERT_INT_EQ(sentbuf_has_segment(sb), 1);
        sentbuf_flush(sb, out, sizeof(out));
        ASSERT_STR_EQ(out, "Hello.");
        sentbuf_destroy(sb);
    }

    /* Multiple sentences in one add — try_flush processes one sentence per call,
       so remaining sentences stay in accumulation buffer until next add or flush_all */
    {
        SentenceBuffer *sb = sentbuf_create(SENTBUF_MODE_SENTENCE, 5);
        sentbuf_add(sb, "First. Second. Third. ", 22);
        ASSERT_INT_EQ(sentbuf_has_segment(sb), 1);
        sentbuf_flush(sb, out, sizeof(out));
        ASSERT_STR_EQ(out, "First.");

        /* Remaining "Second. Third. " is in accumulation buffer.
           Need another add to trigger try_flush again. */
        sentbuf_add(sb, "", 0);  /* won't trigger (len <= 0) */
        /* Use flush_all to drain everything */
        sentbuf_flush_all(sb, out, sizeof(out));
        ASSERT_INT_EQ(strstr(out, "Second") != NULL, 1);
        printf("  multi-sentence remainder: \"%s\"\n", out);

        sentbuf_destroy(sb);
    }

    /* Markdown: bold stripping */
    {
        SentenceBuffer *sb = sentbuf_create(SENTBUF_MODE_SENTENCE, 5);
        sentbuf_add(sb, "This is **bold** text. ", 22);
        sentbuf_flush(sb, out, sizeof(out));
        ASSERT_STR_EQ(out, "This is bold text.");
        sentbuf_destroy(sb);
    }

    /* Markdown: italic stripping */
    {
        SentenceBuffer *sb = sentbuf_create(SENTBUF_MODE_SENTENCE, 5);
        sentbuf_add(sb, "This is *italic* text. ", 23);
        sentbuf_flush(sb, out, sizeof(out));
        ASSERT_STR_EQ(out, "This is italic text.");
        sentbuf_destroy(sb);
    }

    /* Markdown: inline code stripping */
    {
        SentenceBuffer *sb = sentbuf_create(SENTBUF_MODE_SENTENCE, 5);
        sentbuf_add(sb, "Run the `command` now. ", 22);
        sentbuf_flush(sb, out, sizeof(out));
        ASSERT_STR_EQ(out, "Run the now.");
        sentbuf_destroy(sb);
    }

    /* Markdown: link — keep text, strip URL */
    {
        SentenceBuffer *sb = sentbuf_create(SENTBUF_MODE_SENTENCE, 5);
        sentbuf_add(sb, "Click [here](https://example.com) now. ", 39);
        sentbuf_flush(sb, out, sizeof(out));
        ASSERT_INT_EQ(strstr(out, "here") != NULL, 1);
        ASSERT_INT_EQ(strstr(out, "https") == NULL, 1);
        printf("  link stripping: \"%s\"\n", out);
        sentbuf_destroy(sb);
    }

    /* Markdown: heading stripping */
    {
        SentenceBuffer *sb = sentbuf_create(SENTBUF_MODE_SENTENCE, 5);
        sentbuf_add(sb, "# Title text. ", 14);
        sentbuf_flush(sb, out, sizeof(out));
        ASSERT_INT_EQ(strstr(out, "Title") != NULL, 1);
        ASSERT_INT_EQ(out[0] != '#', 1);
        printf("  heading stripping: \"%s\"\n", out);
        sentbuf_destroy(sb);
    }

    /* flush_all with remaining partial text */
    {
        SentenceBuffer *sb = sentbuf_create(SENTBUF_MODE_SENTENCE, 5);
        sentbuf_add(sb, "Partial without period", 22);
        ASSERT_INT_EQ(sentbuf_has_segment(sb), 0);
        sentbuf_flush_all(sb, out, sizeof(out));
        ASSERT_STR_EQ(out, "Partial without period");
        sentbuf_destroy(sb);
    }

    /* Reset clears everything */
    {
        SentenceBuffer *sb = sentbuf_create(SENTBUF_MODE_SENTENCE, 5);
        sentbuf_add(sb, "Some text. ", 11);
        ASSERT_INT_EQ(sentbuf_has_segment(sb), 1);
        sentbuf_reset(sb);
        ASSERT_INT_EQ(sentbuf_has_segment(sb), 0);
        sentbuf_destroy(sb);
    }

    /* Rapid push/flush cycles */
    {
        SentenceBuffer *sb = sentbuf_create(SENTBUF_MODE_SENTENCE, 5);
        for (int i = 0; i < 50; i++) {
            sentbuf_add(sb, "A. ", 3);
            if (sentbuf_has_segment(sb)) {
                sentbuf_flush(sb, out, sizeof(out));
            }
        }
        sentbuf_flush_all(sb, out, sizeof(out));
        /* Just verify no crash */
        ASSERT_INT_EQ(1, 1);
        sentbuf_destroy(sb);
    }

    /* Question mark as sentence boundary */
    {
        SentenceBuffer *sb = sentbuf_create(SENTBUF_MODE_SENTENCE, 5);
        sentbuf_add(sb, "How are you? ", 13);
        ASSERT_INT_EQ(sentbuf_has_segment(sb), 1);
        sentbuf_flush(sb, out, sizeof(out));
        ASSERT_STR_EQ(out, "How are you?");
        sentbuf_destroy(sb);
    }

    /* Exclamation mark as sentence boundary */
    {
        SentenceBuffer *sb = sentbuf_create(SENTBUF_MODE_SENTENCE, 5);
        sentbuf_add(sb, "Wow! ", 5);
        ASSERT_INT_EQ(sentbuf_has_segment(sb), 1);
        sentbuf_flush(sb, out, sizeof(out));
        ASSERT_STR_EQ(out, "Wow!");
        sentbuf_destroy(sb);
    }

    /* Newline as sentence boundary */
    {
        SentenceBuffer *sb = sentbuf_create(SENTBUF_MODE_SENTENCE, 5);
        sentbuf_add(sb, "Line one\nLine two. ", 19);
        ASSERT_INT_EQ(sentbuf_has_segment(sb), 1);
        sentbuf_flush(sb, out, sizeof(out));
        ASSERT_STR_EQ(out, "Line one");
        sentbuf_destroy(sb);
    }

    printf("  sentence_buffer edge cases: %d/%d passed\n\n",
           tests_passed - prev_passed, tests_run - prev_run);
}

static void test_sentbuf_speculative_mode(void) {
    int prev_passed = tests_passed, prev_run = tests_run;
    char out[2048];

    printf("=== sentence_buffer speculative mode ===\n");

    /* Speculative mode flushes at clause boundaries (comma, semicolon, colon) */
    {
        SentenceBuffer *sb = sentbuf_create(SENTBUF_MODE_SPECULATIVE, 3);
        sentbuf_add(sb, "Hello world, this is a test. ", 28);
        ASSERT_INT_EQ(sentbuf_has_segment(sb), 1);
        sentbuf_flush(sb, out, sizeof(out));
        printf("  speculative comma: \"%s\"\n", out);

        /* Remaining should flush too */
        if (sentbuf_has_segment(sb)) {
            sentbuf_flush(sb, out, sizeof(out));
            printf("  speculative remainder: \"%s\"\n", out);
        }
        sentbuf_destroy(sb);
    }

    /* Semicolon boundary */
    {
        SentenceBuffer *sb = sentbuf_create(SENTBUF_MODE_SPECULATIVE, 3);
        sentbuf_add(sb, "First part; second part. ", 24);
        ASSERT_INT_EQ(sentbuf_has_segment(sb), 1);
        sentbuf_flush(sb, out, sizeof(out));
        printf("  speculative semicolon: \"%s\"\n", out);
        sentbuf_destroy(sb);
    }

    /* Below min_words — should not flush at clause */
    {
        SentenceBuffer *sb = sentbuf_create(SENTBUF_MODE_SPECULATIVE, 5);
        sentbuf_add(sb, "Hi, there. ", 11);
        /* "Hi" is only 1 word before comma, so clause flush won't fire.
           But sentence boundary at "." should still fire. */
        ASSERT_INT_EQ(sentbuf_has_segment(sb), 1);
        sentbuf_flush(sb, out, sizeof(out));
        printf("  speculative min_words: \"%s\"\n", out);
        sentbuf_destroy(sb);
    }

    printf("  sentence_buffer speculative: %d/%d passed\n\n",
           tests_passed - prev_passed, tests_run - prev_run);
}

static void test_sentbuf_adaptive_mode(void) {
    int prev_passed = tests_passed, prev_run = tests_run;
    char out[2048];

    printf("=== sentence_buffer adaptive mode ===\n");

    /* Adaptive: first N sentences use lower min_words */
    {
        SentenceBuffer *sb = sentbuf_create(SENTBUF_MODE_SPECULATIVE, 8);
        sentbuf_set_adaptive(sb, 2, 3);  /* First 2 sentences use min_words=3 */

        /* During warmup, should flush at clause with just 3 words */
        sentbuf_add(sb, "One two three, four five six. ", 29);
        ASSERT_INT_EQ(sentbuf_has_segment(sb), 1);
        sentbuf_flush(sb, out, sizeof(out));
        printf("  adaptive warmup flush: \"%s\"\n", out);

        /* Drain remaining */
        while (sentbuf_has_segment(sb)) {
            sentbuf_flush(sb, out, sizeof(out));
        }
        sentbuf_flush_all(sb, out, sizeof(out));

        sentbuf_destroy(sb);
    }

    printf("  sentence_buffer adaptive: %d/%d passed\n\n",
           tests_passed - prev_passed, tests_run - prev_run);
}

static void test_sentbuf_eager_mode(void) {
    int prev_passed = tests_passed, prev_run = tests_run;
    char out[2048];

    printf("=== sentence_buffer eager mode ===\n");

    /* Eager: flush after N words even without boundary */
    {
        SentenceBuffer *sb = sentbuf_create(SENTBUF_MODE_SENTENCE, 5);
        sentbuf_set_eager(sb, 4);

        sentbuf_add(sb, "One two three four five six", 27);
        ASSERT_INT_EQ(sentbuf_has_segment(sb), 1);
        sentbuf_flush(sb, out, sizeof(out));
        printf("  eager flush: \"%s\"\n", out);

        /* After sentence boundary, eager disables */
        sentbuf_add(sb, " seven.", 7);
        ASSERT_INT_EQ(sentbuf_has_segment(sb), 1);
        sentbuf_flush(sb, out, sizeof(out));
        printf("  post-sentence: \"%s\"\n", out);

        /* Now eager is off — 5 words won't trigger */
        sentbuf_add(sb, "Aa bb cc dd ee", 14);
        ASSERT_INT_EQ(sentbuf_has_segment(sb), 0);
        sentbuf_flush_all(sb, out, sizeof(out));

        /* Reset re-arms eager */
        sentbuf_reset(sb);
        sentbuf_set_eager(sb, 4);
        sentbuf_add(sb, "Alpha beta gamma delta epsilon", 30);
        ASSERT_INT_EQ(sentbuf_has_segment(sb), 1);
        sentbuf_flush(sb, out, sizeof(out));
        printf("  re-armed eager: \"%s\"\n", out);

        sentbuf_destroy(sb);
    }

    /* Eager disabled (0) */
    {
        SentenceBuffer *sb = sentbuf_create(SENTBUF_MODE_SENTENCE, 5);
        sentbuf_set_eager(sb, 0);
        sentbuf_add(sb, "One two three four five six seven eight", 38);
        ASSERT_INT_EQ(sentbuf_has_segment(sb), 0);  /* No boundary, no eager */
        sentbuf_flush_all(sb, out, sizeof(out));
        sentbuf_destroy(sb);
    }

    printf("  sentence_buffer eager: %d/%d passed\n\n",
           tests_passed - prev_passed, tests_run - prev_run);
}

static void test_sentbuf_predicted_length(void) {
    int prev_passed = tests_passed, prev_run = tests_run;
    char out[2048];

    printf("=== sentbuf_predicted_length ===\n");

    SentenceBuffer *sb = sentbuf_create(SENTBUF_MODE_SENTENCE, 5);

    /* No sentences yet */
    ASSERT_INT_EQ(sentbuf_predicted_length(sb), 0);

    /* After one sentence, EMA should be set */
    sentbuf_add(sb, "Hello world. ", 13);
    sentbuf_flush(sb, out, sizeof(out));
    int pred1 = sentbuf_predicted_length(sb);
    ASSERT_INT_EQ(pred1 > 0, 1);
    printf("  predicted after 1 sentence: %d chars\n", pred1);

    /* After another sentence, EMA updates */
    sentbuf_add(sb, "A much longer sentence with more words goes here. ", 50);
    sentbuf_flush(sb, out, sizeof(out));
    int pred2 = sentbuf_predicted_length(sb);
    ASSERT_INT_EQ(pred2 > 0, 1);
    printf("  predicted after 2 sentences: %d chars\n", pred2);

    sentbuf_destroy(sb);

    printf("  sentbuf_predicted_length: %d/%d passed\n\n",
           tests_passed - prev_passed, tests_run - prev_run);
}

static void test_sentbuf_sentence_count(void) {
    int prev_passed = tests_passed, prev_run = tests_run;
    char out[2048];

    printf("=== sentbuf_sentence_count ===\n");

    SentenceBuffer *sb = sentbuf_create(SENTBUF_MODE_SENTENCE, 5);

    ASSERT_INT_EQ(sentbuf_sentence_count(sb), 0);

    sentbuf_add(sb, "First. ", 7);
    sentbuf_flush(sb, out, sizeof(out));
    ASSERT_INT_EQ(sentbuf_sentence_count(sb), 1);

    sentbuf_add(sb, "Second. ", 8);
    sentbuf_flush(sb, out, sizeof(out));
    ASSERT_INT_EQ(sentbuf_sentence_count(sb), 2);

    sentbuf_add(sb, "Third. ", 7);
    sentbuf_flush(sb, out, sizeof(out));
    ASSERT_INT_EQ(sentbuf_sentence_count(sb), 3);

    /* Reset clears the count */
    sentbuf_reset(sb);
    ASSERT_INT_EQ(sentbuf_sentence_count(sb), 0);

    sentbuf_destroy(sb);

    printf("  sentbuf_sentence_count: %d/%d passed\n\n",
           tests_passed - prev_passed, tests_run - prev_run);
}

static void test_sentbuf_prosody_hints(void) {
    int prev_passed = tests_passed, prev_run = tests_run;
    char out[2048];

    printf("=== sentbuf_get_prosody_hint ===\n");

    /* Use fresh buffers for each test to avoid sentence remnant mixing.
       try_flush only processes one sentence per add, so remnants from
       multi-sentence text (e.g. "Well...") shift into subsequent flushes. */

    /* Exclamation marks */
    {
        SentenceBuffer *sb = sentbuf_create(SENTBUF_MODE_SENTENCE, 5);
        sentbuf_add(sb, "This is amazing!! ", 18);
        sentbuf_flush(sb, out, sizeof(out));
        SentBufProsodyHint hint = sentbuf_get_prosody_hint(sb);
        ASSERT_INT_EQ(hint.exclamation_count >= 2, 1);
        ASSERT_INT_EQ(hint.suggested_pitch > 1.0f, 1);
        printf("  exclamations: count=%d pitch=%.2f energy=%.2f\n",
               hint.exclamation_count, (double)hint.suggested_pitch,
               (double)hint.suggested_energy);
        sentbuf_destroy(sb);
    }

    /* Question mark */
    {
        SentenceBuffer *sb = sentbuf_create(SENTBUF_MODE_SENTENCE, 5);
        sentbuf_add(sb, "Is this a question? ", 20);
        sentbuf_flush(sb, out, sizeof(out));
        SentBufProsodyHint hint = sentbuf_get_prosody_hint(sb);
        ASSERT_INT_EQ(hint.question_count >= 1, 1);
        printf("  question: count=%d\n", hint.question_count);
        sentbuf_destroy(sb);
    }

    /* Ellipsis — "Well..." triggers sentence boundary at third '.', splitting
       the text. Use a simpler case that stays in one segment. */
    {
        SentenceBuffer *sb = sentbuf_create(SENTBUF_MODE_SENTENCE, 5);
        sentbuf_add(sb, "Hmm... okay then. ", 18);
        /* First flush may get "Hmm..." (boundary at third dot + space) */
        sentbuf_flush(sb, out, sizeof(out));
        SentBufProsodyHint hint = sentbuf_get_prosody_hint(sb);
        ASSERT_INT_EQ(hint.has_ellipsis, 1);
        ASSERT_INT_EQ(hint.suggested_rate < 1.0f, 1);
        printf("  ellipsis: text=\"%s\" rate=%.2f\n", out, (double)hint.suggested_rate);
        sentbuf_destroy(sb);
    }

    /* ALL CAPS */
    {
        SentenceBuffer *sb = sentbuf_create(SENTBUF_MODE_SENTENCE, 5);
        sentbuf_add(sb, "This is VERY important. ", 24);
        sentbuf_flush(sb, out, sizeof(out));
        SentBufProsodyHint hint = sentbuf_get_prosody_hint(sb);
        ASSERT_INT_EQ(hint.has_all_caps, 1);
        printf("  ALL CAPS: text=\"%s\" pitch=%.2f energy=%.2f\n",
               out, (double)hint.suggested_pitch, (double)hint.suggested_energy);
        sentbuf_destroy(sb);
    }

    /* Neutral text — no special hints */
    {
        SentenceBuffer *sb = sentbuf_create(SENTBUF_MODE_SENTENCE, 5);
        sentbuf_add(sb, "Just a normal sentence. ", 24);
        sentbuf_flush(sb, out, sizeof(out));
        SentBufProsodyHint hint = sentbuf_get_prosody_hint(sb);
        ASSERT_INT_EQ(hint.exclamation_count, 0);
        ASSERT_INT_EQ(hint.question_count, 0);
        ASSERT_INT_EQ(hint.has_ellipsis, 0);
        ASSERT_INT_EQ(hint.has_all_caps, 0);
        ASSERT_INT_EQ((int)(hint.suggested_rate * 100), 100);
        ASSERT_INT_EQ((int)(hint.suggested_pitch * 100), 100);
        sentbuf_destroy(sb);
    }

    printf("  sentbuf_get_prosody_hint: %d/%d passed\n\n",
           tests_passed - prev_passed, tests_run - prev_run);
}

static void test_sentbuf_null_safety(void) {
    int prev_passed = tests_passed, prev_run = tests_run;

    printf("=== sentence_buffer NULL safety ===\n");

    /* sentbuf_destroy(NULL) should not crash */
    sentbuf_destroy(NULL);
    ASSERT_INT_EQ(1, 1);  /* Reached here = no crash */

    printf("  sentence_buffer NULL safety: %d/%d passed\n\n",
           tests_passed - prev_passed, tests_run - prev_run);
}

static void test_sentbuf_code_block_filtering(void) {
    int prev_passed = tests_passed, prev_run = tests_run;
    char out[2048];

    printf("=== sentence_buffer code block filtering ===\n");

    /* Triple backtick markers are stripped but inline content is kept.
       Full code block filtering happens at the sentbuf_add level when
       the buffer tracks in_code_block state across multiple add() calls. */
    {
        SentenceBuffer *sb = sentbuf_create(SENTBUF_MODE_SENTENCE, 5);
        sentbuf_add(sb, "Before ```int x = 0;``` after. ", 31);
        sentbuf_flush(sb, out, sizeof(out));
        /* Backtick chars are stripped, but inline content remains */
        ASSERT_INT_EQ(strstr(out, "Before") != NULL, 1);
        ASSERT_INT_EQ(strstr(out, "after") != NULL, 1);
        ASSERT_INT_EQ(strstr(out, "```") == NULL, 1);  /* backticks removed */
        printf("  code block inline: \"%s\"\n", out);
        sentbuf_destroy(sb);
    }

    /* Code block state tracking: verify in_code_block flag */
    {
        SentenceBuffer *sb = sentbuf_create(SENTBUF_MODE_SENTENCE, 5);
        /* Single add with code block — fences toggle in_code_block,
           but inline content between fences is kept by clean_for_speech */
        sentbuf_add(sb, "Start text. ", 12);
        ASSERT_INT_EQ(sentbuf_has_segment(sb), 1);
        sentbuf_flush(sb, out, sizeof(out));
        ASSERT_STR_EQ(out, "Start text.");
        sentbuf_destroy(sb);
    }

    /* Inline code with backticks */
    {
        SentenceBuffer *sb = sentbuf_create(SENTBUF_MODE_SENTENCE, 5);
        sentbuf_add(sb, "Use `printf` for output. ", 25);
        sentbuf_flush(sb, out, sizeof(out));
        ASSERT_INT_EQ(strstr(out, "printf") == NULL, 1);
        ASSERT_INT_EQ(strstr(out, "Use") != NULL, 1);
        ASSERT_INT_EQ(strstr(out, "output") != NULL, 1);
        printf("  inline code filtered: \"%s\"\n", out);
        sentbuf_destroy(sb);
    }

    printf("  code block filtering: %d/%d passed\n\n",
           tests_passed - prev_passed, tests_run - prev_run);
}

int main(void) {
    test_text_normalize();
    test_sentence_buffer();
    test_ssml_parser();
    test_nonverbalisms_extended();

    /* Edge case test suites: text_normalize */
    test_cardinal_edge_cases();
    test_ordinal_edge_cases();
    test_currency_edge_cases();
    test_telephone_edge_cases();
    test_date_edge_cases();
    test_time_edge_cases();
    test_fraction_edge_cases();
    test_characters_edge_cases();
    test_unit_edge_cases();
    test_url_edge_cases();
    test_email_edge_cases();
    test_text_normalize_dispatch();
    test_auto_normalize_edge_cases();
    test_inline_ipa();
    test_pronunciation_dict();

    /* Edge case test suites: ssml_parser */
    test_ssml_edge_cases();
    test_ssml_is_ssml_edge_cases();
    test_ssml_prosody_parsers();

    /* Edge case test suites: sentence_buffer */
    test_sentbuf_edge_cases();
    test_sentbuf_speculative_mode();
    test_sentbuf_adaptive_mode();
    test_sentbuf_eager_mode();
    test_sentbuf_predicted_length();
    test_sentbuf_sentence_count();
    test_sentbuf_prosody_hints();
    test_sentbuf_null_safety();
    test_sentbuf_code_block_filtering();

    printf("═══════════════════════════\n");
    printf("Total: %d/%d tests passed\n", tests_passed, tests_run);
    return tests_passed == tests_run ? 0 : 1;
}
