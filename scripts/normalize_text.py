#!/usr/bin/env python3
"""
Text normalization for Sonata TTS pipeline.

Python port of text_normalize.c — implements the same normalization rules
for numbers, dates, currencies, abbreviations, phone numbers, URLs, emails,
time, ordinals, and Roman numerals.

Usage:
    python scripts/normalize_text.py                    # Print examples
    python scripts/normalize_text.py --test             # Run built-in tests
    python scripts/normalize_text.py -i manifest.jsonl -o out.jsonl
"""

import argparse
import json
import re
import sys
from typing import Optional

# Lookup tables matching text_normalize.c
ONES = [
    "", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
    "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen",
    "seventeen", "eighteen", "nineteen",
]
TENS = [
    "", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy",
    "eighty", "ninety",
]
ORDINAL_ONES = [
    "", "first", "second", "third", "fourth", "fifth", "sixth", "seventh",
    "eighth", "ninth", "tenth", "eleventh", "twelfth", "thirteenth",
    "fourteenth", "fifteenth", "sixteenth", "seventeenth", "eighteenth",
    "nineteenth",
]
ORDINAL_TENS = [
    "", "", "twentieth", "thirtieth", "fortieth", "fiftieth",
    "sixtieth", "seventieth", "eightieth", "ninetieth",
]
MONTHS = [
    "", "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]

CURRENCIES = [
    ("$", "dollar", "dollars", "cent", "cents"),
    ("€", "euro", "euros", "cent", "cents"),
    ("£", "pound", "pounds", "penny", "pence"),
    ("¥", "yen", "yen", "sen", "sen"),
    ("₹", "rupee", "rupees", "paisa", "paise"),
    ("₩", "won", "won", "", ""),
    ("₿", "bitcoin", "bitcoins", "satoshi", "satoshis"),
    ("CHF", "Swiss franc", "Swiss francs", "centime", "centimes"),
    ("kr", "krone", "kroner", "øre", "øre"),
    ("R$", "real", "reais", "centavo", "centavos"),
]

FRACTIONS = [
    (1, 2, "one half"), (1, 3, "one third"), (2, 3, "two thirds"),
    (1, 4, "one quarter"), (3, 4, "three quarters"),
    (1, 5, "one fifth"), (2, 5, "two fifths"), (3, 5, "three fifths"),
    (4, 5, "four fifths"), (1, 6, "one sixth"), (1, 8, "one eighth"),
    (3, 8, "three eighths"), (5, 8, "five eighths"), (7, 8, "seven eighths"),
    (1, 10, "one tenth"),
]

ABBREVIATIONS = [
    ("Dr.", "doctor"), ("Mr.", "mister"), ("Mrs.", "missus"), ("Ms.", "miz"),
    ("Prof.", "professor"), ("Jr.", "junior"), ("Sr.", "senior"),
    ("St.", "saint"),  # or "street" — default saint
    ("vs.", "versus"), ("e.g.", "for example"), ("i.e.", "that is"),
    ("etc.", "etcetera"), ("approx.", "approximately"), ("dept.", "department"),
    ("govt.", "government"), ("Inc.", "incorporated"), ("Corp.", "corporation"),
    ("Ltd.", "limited"), ("Ave.", "avenue"), ("Blvd.", "boulevard"),
    ("Rd.", "road"), ("Apt.", "apartment"),
    ("Feb.", "February"), ("Jan.", "January"), ("Mar.", "March"),
    ("Apr.", "April"), ("Jun.", "June"), ("Jul.", "July"), ("Aug.", "August"),
    ("Sep.", "September"), ("Oct.", "October"), ("Nov.", "November"),
    ("Dec.", "December"),
]

ROMAN_NUMERALS = {
    "I": 1, "II": 2, "III": 3, "IV": 4, "V": 5, "VI": 6, "VII": 7, "VIII": 8,
    "IX": 9, "X": 10, "XI": 11, "XII": 12, "XIII": 13, "XIV": 14, "XV": 15,
    "XVI": 16, "XVII": 17, "XVIII": 18, "XIX": 19, "XX": 20,
}


def number_to_words(n: int) -> str:
    """Convert integer to words (cardinal)."""
    if n < 0:
        return "negative " + number_to_words(-n)
    if n == 0:
        return "zero"
    parts = []
    for divisor, label in [(10**12, "trillion"), (10**9, "billion"),
                          (10**6, "million"), (10**3, "thousand")]:
        if n >= divisor:
            parts.append(number_to_words(n // divisor) + " " + label)
            n %= divisor
    if n >= 100:
        parts.append(ONES[n // 100] + " hundred")
        n %= 100
    if n >= 20:
        t, o = divmod(n, 10)
        parts.append(TENS[t] + ("-" + ONES[o] if o else ""))
    elif n > 0:
        parts.append(ONES[n])
    return " ".join(parts)


def ordinal_to_words(n: int) -> str:
    """Convert integer to ordinal words."""
    if n < 0:
        return "negative " + ordinal_to_words(-n)
    if n == 0:
        return "zeroth"
    if n >= 100:
        hundreds = (n // 100) * 100
        rem = n % 100
        if rem == 0:
            return number_to_words(hundreds) + "th"
        return number_to_words(hundreds) + " " + ordinal_to_words(rem)
    if n >= 20:
        t, o = divmod(n, 10)
        if o:
            return TENS[t] + "-" + ORDINAL_ONES[o]
        return ORDINAL_TENS[t]
    return ORDINAL_ONES[n]


def year_to_words(year: int) -> str:
    """Convert year to words."""
    if 2000 <= year < 2010:
        return "two thousand" + (" " + ONES[year - 2000] if year > 2000 else "")
    if 2010 <= year < 2100:
        return "twenty " + number_to_words(year - 2000).replace(" ", "-")
    if year >= 1000:
        hi, lo = divmod(year, 100)
        if lo == 0:
            return number_to_words(hi) + " hundred"
        return number_to_words(hi) + " " + number_to_words(lo)
    return number_to_words(year)


def normalize_cardinal(text: str) -> str:
    """Normalize number/cardinal."""
    text = text.strip().replace(",", "")
    if "." in text:
        parts = text.split(".", 1)
        try:
            int_val = int(parts[0])
            result = number_to_words(int_val) + " point"
            for d in parts[1]:
                if d.isdigit():
                    result += " " + ("zero" if d == "0" else ONES[int(d)])
            return result
        except ValueError:
            pass
    try:
        val = int(text)
        return number_to_words(val)
    except ValueError:
        return text


def normalize_ordinal(text: str) -> str:
    """Normalize ordinal (1st, 2nd, 22nd, etc.)."""
    text = text.strip()
    m = re.match(r"^(-?\d[\d,]*)\s*(st|nd|rd|th)$", text, re.I)
    if m:
        try:
            val = int(m.group(1).replace(",", ""))
            return ordinal_to_words(val)
        except ValueError:
            pass
    return text


def normalize_currency(text: str) -> str:
    """Normalize currency amount."""
    text = text.strip()
    for sym, sing, plur, c_sing, c_plur in CURRENCIES:
        if sym and sym in text:
            amount_str = "".join(c for c in text if c.isdigit() or c == ".")
            if not amount_str:
                continue
            try:
                amount = float(amount_str)
                int_part = int(amount)
                dec_part = int(round((amount - int_part) * 100))
                dollar = sing if int_part == 1 else plur
                result = number_to_words(int_part) + " " + dollar
                if dec_part > 0 and c_sing:
                    cent = c_sing if dec_part == 1 else c_plur
                    result += " and " + number_to_words(dec_part) + " " + cent
                return result
            except ValueError:
                pass
    return text


def normalize_date(text: str) -> str:
    """Normalize date (12/25/2024, January 5th, etc.)."""
    text = text.strip()
    # Try "Month Day" or "Month Dayth" (e.g., January 5, January 5th)
    for i, name in enumerate(MONTHS[1:], 1):
        pat = rf"^{re.escape(name)}\s+(\d{{1,2}})(?:st|nd|rd|th)?\s*,?\s*(\d{{4}})?\s*$"
        m = re.match(pat, text, re.I)
        if m:
            day = int(m.group(1))
            year = m.group(2)
            result = name + " " + ordinal_to_words(day)
            if year:
                result += ", " + year_to_words(int(year))
            return result
    # Try numeric date MM/DD/YYYY or MM-DD-YYYY
    m = re.match(r"^(\d{1,4})[/\-.](\d{1,2})[/\-.](\d{1,4})$", text)
    if m:
        a, b, c = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if a >= 1000:  # YYYY-MM-DD
            year, month, day = a, b, c
        else:
            month, day, year = a, b, c
        if year < 100:
            year += 2000 if year < 50 else 1900
        month_name = MONTHS[month] if 1 <= month <= 12 else "?"
        return f"{month_name} {ordinal_to_words(day)}, {year_to_words(year)}"
    return text


def normalize_time(text: str) -> str:
    """Normalize time (3:30 PM, 12:00, etc.)."""
    text = text.strip()
    m = re.match(r"^(\d{1,2}):(\d{2})(?::(\d{2}))?\s*(AM|PM|am|pm)?\s*$", text)
    if not m:
        return text
    hour, minute, second, ampm = int(m.group(1)), int(m.group(2)), m.group(3), m.group(4)
    if ampm:
        ampm = ampm.upper()
    if minute == 0 and (second is None or int(second) == 0):
        if ampm == "AM" and hour == 12:
            return "midnight"
        if ampm == "PM" and hour == 12:
            return "noon"
        if not ampm:
            if hour == 0:
                return "midnight"
            if hour == 12:
                return "noon"
    hour_str = number_to_words(hour)
    if minute == 0:
        result = hour_str + " o'clock" if not ampm else hour_str
    elif minute < 10:
        result = hour_str + " oh " + number_to_words(minute)
    else:
        result = hour_str + " " + number_to_words(minute)
    if second and int(second) > 0:
        result += " and " + number_to_words(int(second)) + " seconds"
    if ampm:
        result += " " + ampm
    return result


def normalize_telephone(text: str) -> str:
    """Normalize phone number to digit-by-digit."""
    digits = [d for d in text if d.isdigit()]
    if not digits:
        return text
    return " ".join("zero" if d == "0" else ONES[int(d)] for d in digits)


def normalize_fraction(text: str) -> str:
    """Normalize fraction (1/2, 3/4, etc.)."""
    text = text.strip()
    m = re.match(r"^(\d+)\s*/\s*(\d+)$", text)
    if not m:
        return text
    num, den = int(m.group(1)), int(m.group(2))
    if den == 0:
        return text
    for n, d, name in FRACTIONS:
        if num == n and den == d:
            return name
    if den == 2:
        return "one half" if num == 1 else "two halves"
    ord_den = ordinal_to_words(den)
    result = number_to_words(num) + " " + ord_den
    if num != 1:
        result += "s"
    return result


def normalize_url(text: str) -> str:
    """Normalize URL: strip protocol, expand dots and special chars."""
    text = text.strip()
    for prefix in ("https://", "http://", "ftp://", "www."):
        if text.lower().startswith(prefix.lower()):
            text = text[len(prefix):]
            break
    if text.lower().startswith("www."):
        text = text[4:]
    result = []
    for c in text:
        if c == ".":
            result.append(" dot ")
        elif c == "/":
            result.append(" slash ")
        elif c == "-":
            result.append(" dash ")
        elif c == "_":
            result.append(" underscore ")
        elif c == "?":
            result.append(" question mark ")
        elif c == "=":
            result.append(" equals ")
        elif c == "&":
            result.append(" ampersand ")
        elif c == "#":
            result.append(" hash ")
        elif c == "%":
            result.append(" percent ")
        elif c == ":":
            result.append(" colon ")
        elif c == "@":
            result.append(" at ")
        else:
            result.append(c)
    return "".join(result).strip()


def normalize_email(text: str) -> str:
    """Normalize email: expand @ to 'at', . to 'dot'."""
    if "@" not in text:
        return text
    local, _, domain = text.partition("@")
    parts = []
    for s, sep in [(local, " at "), (domain, "")]:
        for c in s:
            if c == ".":
                parts.append(" dot ")
            elif c == "-":
                parts.append(" dash ")
            elif c == "_":
                parts.append(" underscore ")
            elif c == "+":
                parts.append(" plus ")
            else:
                parts.append(c)
        parts.append(sep)
    return "".join(parts).rstrip()


def normalize_roman(text: str) -> Optional[str]:
    """Normalize Roman numeral to ordinal (e.g., III -> the third)."""
    t = text.strip().upper()
    if t in ROMAN_NUMERALS:
        return "the " + ordinal_to_words(ROMAN_NUMERALS[t])
    return None


def _word_boundary(c: str) -> bool:
    return not c or c in " \t\n\r,.)]}\""


def normalize_text(text: str) -> str:
    """
    Auto-normalize raw text: scan for URLs, emails, currency, ordinals,
    abbreviations, time, fractions, phone numbers, Roman numerals, etc.
    Returns normalized string matching C text_auto_normalize behavior.
    """
    result = []
    i = 0
    text_len = len(text)

    while i < text_len:
        # URL
        for prefix in ("https://", "http://", "ftp://", "www."):
            if text[i:i+len(prefix)].lower() == prefix.lower():
                j = i + len(prefix)
                while j < text_len and text[j] not in " \t\n,)]":
                    j += 1
                while j > i and text[j-1] in ".,;":
                    j -= 1
                token = text[i:j]
                result.append(normalize_url(token))
                i = j
                break
        else:
            # Email: word@word
            if text[i] == "@" and i > 0:
                start = i - 1
                while start > 0 and text[start-1] in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._+-":
                    start -= 1
                j = i + 1
                while j < text_len and text[j] in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.-":
                    j += 1
                if j > i + 1 and (j >= text_len or text[j] not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"):
                    # Remove already-copied local part from result (C rewinds output)
                    if len(result) >= i - start:
                        del result[-(i - start):]
                    result.append(normalize_email(text[start:j]))
                    i = j
                    continue
            # Abbreviation
            abbrev_matched = False
            for abbrev, expansion in ABBREVIATIONS:
                if text[i:i+len(abbrev)].lower() == abbrev.lower():
                    end = i + len(abbrev)
                    if end >= text_len or _word_boundary(text[end]):
                        result.append(expansion)
                        i = end
                        abbrev_matched = True
                        break
            if abbrev_matched:
                continue
            # Currency
            curr_matched = False
            for sym, *_ in CURRENCIES:
                if sym and text[i:i+len(sym)] == sym:
                    j = i + len(sym)
                    while j < text_len and (text[j].isdigit() or text[j] in ".,"):
                        j += 1
                    if j > i + len(sym) and text[j-1] != ".":
                        token = text[i:j]
                        result.append(normalize_currency(token))
                        i = j
                        curr_matched = True
                    break
            if curr_matched:
                continue
            # Ordinal (1st, 2nd, 22nd, etc.)
            m = re.match(r"(\d[\d,]*)(st|nd|rd|th)(?=[\s,.)\]]|$)", text[i:], re.I)
            if m:
                token = text[i:i+m.end()]
                result.append(normalize_ordinal(token))
                i += m.end()
                continue
            # Date (MM/DD/YYYY or MM-DD-YYYY) — before fraction to avoid 25/2024 parsed as fraction
            m = re.match(r"(\d{1,4})[/\-.](\d{1,2})[/\-.](\d{1,4})(?=[\s,.)\]]|$)", text[i:])
            if m:
                a, b, c = int(m.group(1)), int(m.group(2)), int(m.group(3))
                # Valid date: 3 groups with / or - or .
                if (1 <= b <= 31 and 1 <= c <= 9999) or (1 <= b <= 12 and c >= 1000):
                    token = text[i:i+m.end()]
                    result.append(normalize_date(token))
                    i += m.end()
                    continue
            # Fraction (N/M)
            m = re.match(r"(\d+)/(\d+)(?=[\s,.)\]]|$)", text[i:])
            if m:
                token = text[i:i+m.end()]
                result.append(normalize_fraction(token))
                i += m.end()
                continue
            # Time
            m = re.match(r"(\d{1,2}):(\d{2})(?::(\d{2}))?\s*(?:AM|PM|am|pm)?", text[i:])
            if m:
                token = text[i:i+m.end()]
                result.append(normalize_time(token))
                i += m.end()
                continue
            # Roman numeral (standalone word)
            roman_m = re.match(r"\b([IVXLCDM]+)\b", text[i:])
            if roman_m:
                rn = roman_m.group(1)
                if rn.upper() in ROMAN_NUMERALS:
                    result.append(normalize_roman(rn))
                    i += len(rn)
                    continue
            # Phone number: (555) 123-4567 style
            if text[i] in r"(\d" and i + 1 < text_len:
                chunk = text[i:]
                digits = sum(1 for c in chunk[:20] if c.isdigit())
                if 7 <= digits <= 15:
                    j = i
                    while j < min(i+30, text_len) and (text[j].isdigit() or text[j] in " ()-.x"):
                        j += 1
                    token = text[i:j]
                    if sum(1 for c in token if c.isdigit()) >= 7:
                        result.append(normalize_telephone(token))
                        i = j
                        continue
            # Standalone number
            m = re.match(r"(-?\d[\d,]*\.?\d*)(?=[\s,.)\]]|$)", text[i:])
            if m:
                token = m.group(1)
                if "/" not in token and ":" not in text[i:i+len(token)+3]:
                    result.append(normalize_cardinal(token))
                    i += len(token)
                    continue
            # Date: Month Day or numeric
            for name in MONTHS[1:]:
                if text[i:i+len(name)].lower() == name.lower():
                    rest = text[i+len(name):i+len(name)+20]
                    dm = re.match(r"\s+(\d{1,2})(?:st|nd|rd|th)?\s*,?\s*(\d{4})?\b", rest)
                    if dm:
                        day = int(dm.group(1))
                        year = dm.group(2)
                        result.append(name + " " + ordinal_to_words(day) +
                                     (", " + year_to_words(int(year)) if year else ""))
                        i += len(name) + dm.end()
                        break
                    break
            else:
                # Default: copy char
                result.append(text[i])
                i += 1

    return "".join(result)


def run_tests() -> bool:
    """Run built-in test cases. Returns True if all pass."""
    tests = [
        ("42", "forty-two"),
        ("1000", "one thousand"),
        ("0", "zero"),
        ("1st", "first"),
        ("2nd", "second"),
        ("3rd", "third"),
        ("22nd", "twenty-second"),
        ("$5.99", "five dollars and ninety-nine cents"),
        ("£10", "ten pounds"),
        ("(555) 123-4567", "five five five one two three four five six seven"),
        ("3:30 PM", "three thirty PM"),
        ("Dr.", "doctor"),
        ("Mr.", "mister"),
        ("Mrs.", "missus"),
        ("12/25/2024", "December twenty-fifth, twenty twenty-four"),  # year: "twenty twenty-four"
        ("1/2", "one half"),
        ("III", "the third"),
        ("IV", "the fourth"),
        ("user@example.com", "user at example dot com"),
    ]
    passed = 0
    for inp, expected in tests:
        got = normalize_text(inp)
        if got.lower() == expected.lower():
            passed += 1
        else:
            print(f"  FAIL: {inp!r} -> {got!r} (expected {expected!r})")
    print(f"  {passed}/{len(tests)} tests passed")
    return passed == len(tests)


def main():
    parser = argparse.ArgumentParser(description="Normalize text for TTS")
    parser.add_argument("--input", "-i", help="Input manifest JSONL (normalize 'text' fields)")
    parser.add_argument("--output", "-o", help="Output manifest JSONL")
    parser.add_argument("--test", action="store_true", help="Run built-in tests")
    args = parser.parse_args()

    if args.test:
        print("=== normalize_text tests ===")
        ok = run_tests()
        sys.exit(0 if ok else 1)

    if args.input and args.output:
        count = 0
        with open(args.input) as fin, open(args.output, "w") as fout:
            for line in fin:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if "text" in obj:
                    obj["text"] = normalize_text(obj["text"])
                    count += 1
                fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
        print(f"Normalized {count} entries: {args.input} -> {args.output}")
        return

    # Standalone: print examples
    examples = [
        "42",
        "1st",
        "January 5th",
        "12/25/2024",
        "$5.99",
        "£10",
        "Dr.",
        "Mr.",
        "(555) 123-4567",
        "https://example.com",
        "user@example.com",
        "3:30 PM",
        "22nd",
        "III",
        "IV",
    ]
    print("Text normalization examples (matching text_normalize.c):")
    print("-" * 50)
    for ex in examples:
        print(f"  {ex!r} -> {normalize_text(ex)!r}")
    print()
    print("Usage: --input manifest.jsonl --output out.jsonl")
    print("       --test  (run built-in tests)")


if __name__ == "__main__":
    main()
