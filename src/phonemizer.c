/**
 * phonemizer.c — espeak-ng phonemizer for TTS text preprocessing.
 *
 * Converts text → IPA phoneme string → integer phoneme IDs.
 * Uses espeak_TextToPhonemes() and a JSON phoneme map for ID assignment.
 *
 * Note: espeak-ng is NOT thread-safe. Do not call phonemizer functions
 * from multiple threads concurrently.
 */

#include "phonemizer.h"
#include "cJSON.h"

#include <espeak-ng/speak_lib.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>

#define PHONEMIZER_MAX_SYMBOL_LEN  15
#define PHONEMIZER_MAX_PHONEMES    512
#define PHONEMIZER_IPA_BUF_SIZE    4096

typedef struct {
    char symbol[16];
    int  id;
    int  symbol_len;
} PhonemeEntry;

struct Phonemizer {
    int initialized;
    PhonemeEntry *phoneme_map;
    int n_phonemes;
    int vocab_size;
    int bos_id;   /* BOS token ID (default 1) */
    int eos_id;   /* EOS token ID (default 2) */
};

/* espeak-ng is process-global; shared across all Phonemizer instances. */
static int g_espeak_initialized = 0;
static pthread_mutex_t g_espeak_mutex = PTHREAD_MUTEX_INITIALIZER;

/* Return the byte length of the next UTF-8 character at p, or 0 at end of string. */
static int utf8_char_len(const unsigned char *p) {
    if (!p || !*p) return 0;
    if (*p < 0x80) return 1;
    if ((*p & 0xE0) == 0xC0) return 2;
    if ((*p & 0xF0) == 0xE0) return 3;
    if ((*p & 0xF8) == 0xF0) return 4;
    return 1; /* invalid lead byte, skip one */
}

/* Compare two PhonemeEntry by symbol length descending (for qsort). */
static int compare_by_len_desc(const void *a, const void *b) {
    const PhonemeEntry *ea = (const PhonemeEntry *)a;
    const PhonemeEntry *eb = (const PhonemeEntry *)b;
    if (ea->symbol_len > eb->symbol_len) return -1;
    if (ea->symbol_len < eb->symbol_len) return 1;
    return 0;
}

Phonemizer *phonemizer_create(const char *language) {
    if (!language) return NULL;

    /* Initialize espeak-ng once per process (thread-safe). */
    pthread_mutex_lock(&g_espeak_mutex);
    if (!g_espeak_initialized) {
        int sr = espeak_Initialize(AUDIO_OUTPUT_RETRIEVAL, 0, NULL, 0);
        if (sr < 0) {
            fprintf(stderr, "[phonemizer] espeak_Initialize failed\n");
            pthread_mutex_unlock(&g_espeak_mutex);
            return NULL;
        }
        g_espeak_initialized = 1;
    }
    pthread_mutex_unlock(&g_espeak_mutex);

    espeak_SetVoiceByName(language);

    Phonemizer *ph = (Phonemizer *)calloc(1, sizeof(Phonemizer));
    if (!ph) return NULL;
    ph->initialized = 1;
    ph->bos_id = 1;
    ph->eos_id = 2;
    return ph;
}

void phonemizer_destroy(Phonemizer *ph) {
    if (!ph) return;
    free(ph->phoneme_map);
    ph->phoneme_map = NULL;
    ph->n_phonemes = 0;
    ph->vocab_size = 0;
    free(ph);
    /* Do NOT call espeak_Terminate — espeak is process-global, shared with Piper etc. */
}

int phonemizer_text_to_ipa(Phonemizer *ph, const char *text, char *ipa_out, int max_len) {
    if (!ph || !ph->initialized || !text || !ipa_out || max_len <= 0)
        return -1;

    const void *text_ptr = text;
    int flags = espeakCHARS_UTF8 | espeakPHONEMES_IPA;
    int ipa_pos = 0;
    int first = 1;

    while (text_ptr) {
        const char *phonemes = espeak_TextToPhonemes((void *)&text_ptr, espeakCHARS_UTF8, flags);
        if (!phonemes) break;

        int plen = (int)strlen(phonemes);
        int need = plen + (first ? 0 : 1); /* space between clauses */
        if (ipa_pos + need + 1 >= max_len) break;

        if (!first) {
            ipa_out[ipa_pos++] = ' ';
        }
        memcpy(ipa_out + ipa_pos, phonemes, plen);
        ipa_pos += plen;
        first = 0;
    }

    ipa_out[ipa_pos] = '\0';
    return ipa_pos;
}

int phonemizer_load_phoneme_map(Phonemizer *ph, const char *json_path) {
    if (!ph || !ph->initialized || !json_path) return -1;

    FILE *f = fopen(json_path, "rb");
    if (!f) {
        fprintf(stderr, "[phonemizer] Cannot open phoneme map: %s\n", json_path);
        return -1;
    }
    fseek(f, 0, SEEK_END);
    long fsize = ftell(f);
    if (fsize <= 0) { fclose(f); return -1; }
    fseek(f, 0, SEEK_SET);

    char *json_str = (char *)malloc((size_t)fsize + 1);
    if (!json_str) {
        fclose(f);
        return -1;
    }
    size_t nread = fread(json_str, 1, (size_t)fsize, f);
    fclose(f);
    json_str[nread] = '\0';

    cJSON *root = cJSON_Parse(json_str);
    free(json_str);
    if (!root) {
        fprintf(stderr, "[phonemizer] JSON parse error: %s\n", json_path);
        return -1;
    }

    if (!cJSON_IsObject(root)) {
        fprintf(stderr, "[phonemizer] Phoneme map must be a JSON object\n");
        cJSON_Delete(root);
        return -1;
    }

    /* Support nested format {"phone_to_id": {"a": 1, ...}} or flat {"a": 1, ...} */
    cJSON *obj = root;
    cJSON *phone_to_id = cJSON_GetObjectItemCaseSensitive(root, "phone_to_id");
    if (cJSON_IsObject(phone_to_id)) {
        obj = phone_to_id;
    }

    /* Count entries. */
    int count = 0;
    cJSON *entry;
    cJSON_ArrayForEach(entry, obj) {
        if (entry->string && cJSON_IsNumber(entry))
            count++;
    }

    PhonemeEntry *map = (PhonemeEntry *)calloc((size_t)count, sizeof(PhonemeEntry));
    if (!map) {
        cJSON_Delete(root);
        return -1;
    }

    /* Free old map if any. */
    free(ph->phoneme_map);
    ph->phoneme_map = map;
    ph->n_phonemes = 0;
    ph->vocab_size = 0;

    int max_id = -1;
    count = 0;
    cJSON_ArrayForEach(entry, obj) {
        if (!entry->string || !cJSON_IsNumber(entry)) continue;

        const char *key = entry->string;
        int id = entry->valueint;
        if (id < 0) continue;

        size_t key_len = strlen(key);
        if (key_len >= sizeof(map[0].symbol)) continue;

        memcpy(map[count].symbol, key, key_len + 1);
        map[count].id = id;
        map[count].symbol_len = (int)key_len;
        count++;

        if (id > max_id) max_id = id;
    }

    /* Parse special_tokens for BOS/EOS from root (before we delete it). */
    cJSON *special = cJSON_GetObjectItemCaseSensitive(root, "special_tokens");
    if (cJSON_IsObject(special)) {
        cJSON *bos = cJSON_GetObjectItemCaseSensitive(special, "bos");
        cJSON *eos = cJSON_GetObjectItemCaseSensitive(special, "eos");
        if (cJSON_IsNumber(bos) && bos->valueint >= 0)
            ph->bos_id = bos->valueint;
        if (cJSON_IsNumber(eos) && eos->valueint >= 0)
            ph->eos_id = eos->valueint;
    }

    cJSON_Delete(root);

    ph->n_phonemes = count;
    ph->vocab_size = (max_id >= 0) ? (max_id + 1) : 0;

    /* Sort by symbol length descending for greedy longest-match. */
    qsort(ph->phoneme_map, (size_t)ph->n_phonemes, sizeof(PhonemeEntry), compare_by_len_desc);

    return 0;
}

int phonemizer_ipa_to_ids(Phonemizer *ph, const char *ipa, int *ids_out, int max_ids) {
    if (!ph || !ipa || !ids_out || max_ids <= 0) return -1;
    if (!ph->phoneme_map || ph->n_phonemes == 0) return -1;

    const unsigned char *p = (const unsigned char *)ipa;
    int n = 0;

    while (*p && n < max_ids) {
        int matched = 0;
        for (int i = 0; i < ph->n_phonemes; i++) {
            PhonemeEntry *e = &ph->phoneme_map[i];
            if (e->symbol_len > 0 &&
                (int)(strlen((const char *)p)) >= e->symbol_len &&
                memcmp(p, e->symbol, (size_t)e->symbol_len) == 0) {
                ids_out[n++] = e->id;
                p += e->symbol_len;
                matched = 1;
                break;
            }
        }
        if (!matched) {
            int len = utf8_char_len(p);
            if (len <= 0) break;
            p += len;
            /* Skip unknown symbol (tolerance). */
        }
    }

    return n;
}

int phonemizer_text_to_ids(Phonemizer *ph, const char *text, int *ids_out, int max_ids) {
    if (!ph || !text || !ids_out || max_ids <= 0) return -1;
    if (max_ids < 3) return -1; /* need at least BOS + 1 content + EOS */

    char ipa_buf[PHONEMIZER_IPA_BUF_SIZE];
    int ipa_len = phonemizer_text_to_ipa(ph, text, ipa_buf, (int)sizeof(ipa_buf));
    if (ipa_len < 0) return -1;

    /* Write content to ids_out[1..]; reserve [0] for BOS, [n+1] for EOS */
    int n = phonemizer_ipa_to_ids(ph, ipa_buf, ids_out + 1, max_ids - 2);
    if (n < 0) return -1;

    ids_out[0] = ph->bos_id;
    ids_out[n + 1] = ph->eos_id;
    return n + 2;
}

int phonemizer_vocab_size(const Phonemizer *ph) {
    return ph ? ph->vocab_size : 0;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Pronunciation Dictionary
 * ═══════════════════════════════════════════════════════════════════════════ */

#define PRONDICT_MAX_ENTRIES 1024

struct PronunciationDict {
    char (*words)[64];
    char (*pronunciations)[256];
    int   n_entries;
};

PronunciationDict *pronunciation_dict_load(const char *json_path) {
    if (!json_path) return NULL;

    FILE *f = fopen(json_path, "rb");
    if (!f) {
        fprintf(stderr, "[prondict] Cannot open: %s\n", json_path);
        return NULL;
    }
    fseek(f, 0, SEEK_END);
    long fsize = ftell(f);
    if (fsize <= 0) { fclose(f); return NULL; }
    fseek(f, 0, SEEK_SET);

    char *json_str = (char *)malloc((size_t)fsize + 1);
    if (!json_str) { fclose(f); return NULL; }
    size_t nread = fread(json_str, 1, (size_t)fsize, f);
    fclose(f);
    json_str[nread] = '\0';

    cJSON *root = cJSON_Parse(json_str);
    free(json_str);
    if (!root || !cJSON_IsArray(root)) {
        fprintf(stderr, "[prondict] JSON must be an array: %s\n", json_path);
        if (root) cJSON_Delete(root);
        return NULL;
    }

    int count = cJSON_GetArraySize(root);
    if (count > PRONDICT_MAX_ENTRIES) count = PRONDICT_MAX_ENTRIES;

    PronunciationDict *dict = (PronunciationDict *)calloc(1, sizeof(PronunciationDict));
    if (!dict) { cJSON_Delete(root); return NULL; }

    dict->words = (char (*)[64])calloc((size_t)count, 64);
    dict->pronunciations = (char (*)[256])calloc((size_t)count, 256);
    if (!dict->words || !dict->pronunciations) {
        free(dict->words);
        free(dict->pronunciations);
        free(dict);
        cJSON_Delete(root);
        return NULL;
    }

    int n = 0;
    cJSON *entry;
    cJSON_ArrayForEach(entry, root) {
        if (n >= count) break;
        cJSON *text_j = cJSON_GetObjectItemCaseSensitive(entry, "text");
        cJSON *pron_j = cJSON_GetObjectItemCaseSensitive(entry, "pronunciation");
        if (!cJSON_IsString(text_j) || !cJSON_IsString(pron_j)) continue;
        if (!text_j->valuestring[0] || !pron_j->valuestring[0]) continue;

        snprintf(dict->words[n], 64, "%s", text_j->valuestring);
        snprintf(dict->pronunciations[n], 256, "%s", pron_j->valuestring);
        n++;
    }
    dict->n_entries = n;

    cJSON_Delete(root);
    fprintf(stderr, "[prondict] Loaded %d entries from %s\n", n, json_path);
    return dict;
}

void pronunciation_dict_destroy(PronunciationDict *dict) {
    if (!dict) return;
    free(dict->words);
    free(dict->pronunciations);
    free(dict);
}

static int is_word_boundary(char c) {
    return c == '\0' || c == ' ' || c == ',' || c == '.' || c == '!' || c == '?'
        || c == ';' || c == ':' || c == '\'' || c == '"' || c == '\n' || c == '\t';
}

int pronunciation_dict_apply(const PronunciationDict *dict, const char *text,
                             char *out, int out_cap) {
    if (!dict || !text || !out || out_cap <= 0)
        return -1;

    int pos = 0;
    const char *p = text;

    while (*p && pos < out_cap - 1) {
        int matched = 0;
        for (int i = 0; i < dict->n_entries; i++) {
            int wlen = (int)strlen(dict->words[i]);
            if (wlen == 0) continue;
            if (strncasecmp(p, dict->words[i], (size_t)wlen) == 0
                    && is_word_boundary(p[wlen])) {
                int plen = (int)strlen(dict->pronunciations[i]);
                int space = out_cap - pos - 1;
                int copy = plen < space ? plen : space;
                memcpy(out + pos, dict->pronunciations[i], (size_t)copy);
                pos += copy;
                p += wlen;
                matched = 1;
                break;
            }
        }
        if (!matched) {
            out[pos++] = *p++;
        }
    }

    out[pos] = '\0';
    return pos;
}

int pronunciation_dict_count(const PronunciationDict *dict) {
    return dict ? dict->n_entries : 0;
}
