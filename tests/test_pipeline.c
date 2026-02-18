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

int main(void) {
    test_text_normalize();
    test_sentence_buffer();
    test_ssml_parser();

    printf("═══════════════════════════\n");
    printf("Total: %d/%d tests passed\n", tests_passed, tests_run);
    return tests_passed == tests_run ? 0 : 1;
}
