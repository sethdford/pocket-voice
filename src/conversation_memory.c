/**
 * conversation_memory.c — Persistent multi-turn conversation memory.
 *
 * Stores user/assistant turns to JSONL file, loads on startup, provides
 * context for LLM prompts. Token approximation: strlen/4.
 */

#include "conversation_memory.h"
#include "cJSON.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <errno.h>
#include <sys/stat.h>
#include <unistd.h>

typedef struct {
    char *role;
    char *content;
    double timestamp;
    int approx_tokens;
} Turn;

struct ConversationMemory {
    char *path;
    Turn *turns;
    int n_turns;
    int capacity;
    int max_turns;
    int max_tokens;
};

static int approx_tokens(const char *s) {
    if (!s) return 0;
    size_t len = strlen(s);
    return (int)((len + 3) / 4);  /* rough: ~4 chars per token */
}

static char *expand_path(const char *path) {
    if (!path || path[0] != '~') return strdup(path);
    const char *home = getenv("HOME");
    if (!home) home = "/";
    if (path[1] == '/' || path[1] == '\0') {
        size_t home_len = strlen(home);
        size_t rest = (path[1] == '/') ? strlen(path + 1) : 0;
        char *out = malloc(home_len + rest + 2);
        if (!out) return NULL;
        memcpy(out, home, home_len + 1);
        if (path[1] == '/') strcat(out, path + 1);
        return out;
    }
    return strdup(path);
}

static int mkdir_p(const char *path) {
    char *buf = strdup(path);
    if (!buf) return -1;
    for (char *p = buf + 1; *p; p++) {
        if (*p == '/') {
            *p = '\0';
            mkdir(buf, 0755);
            *p = '/';
        }
    }
    mkdir(buf, 0755);
    free(buf);
    return 0;
}

static int ensure_parent_dirs(const char *path) {
    char *copy = strdup(path);
    if (!copy) return -1;
    char *last = strrchr(copy, '/');
    if (last && last != copy) {
        *last = '\0';
        mkdir_p(copy);
    }
    free(copy);
    return 0;
}

static void turn_free(Turn *t) {
    if (!t) return;
    free(t->role);
    free(t->content);
}

static int load_from_file(ConversationMemory *mem) {
    FILE *f = fopen(mem->path, "r");
    if (!f) return 0;  /* no file yet, ok */

    char line[65536];
    while (fgets(line, sizeof(line), f)) {
        /* trim newline */
        size_t len = strlen(line);
        if (len > 0 && line[len - 1] == '\n') line[--len] = '\0';
        if (len == 0) continue;

        cJSON *obj = cJSON_Parse(line);
        if (!obj) continue;

        const char *role = cJSON_GetStringValue(cJSON_GetObjectItem(obj, "role"));
        const char *content = cJSON_GetStringValue(cJSON_GetObjectItem(obj, "content"));
        double ts = 0.0;
        cJSON *ts_j = cJSON_GetObjectItem(obj, "ts");
        if (ts_j && cJSON_IsNumber(ts_j)) ts = ts_j->valuedouble;

        if (!role || !content) {
            cJSON_Delete(obj);
            continue;
        }

        if (mem->n_turns >= mem->capacity) {
            int new_cap = mem->capacity ? mem->capacity * 2 : 16;
            Turn *next = realloc(mem->turns, (size_t)new_cap * sizeof(Turn));
            if (!next) {
                cJSON_Delete(obj);
                break;
            }
            mem->turns = next;
            mem->capacity = new_cap;
        }

        Turn *t = &mem->turns[mem->n_turns];
        t->role = strdup(role);
        t->content = strdup(content);
        t->timestamp = ts;
        t->approx_tokens = approx_tokens(content);
        mem->n_turns++;

        cJSON_Delete(obj);
    }
    fclose(f);
    return 0;
}

static int append_line_to_file(const char *path, const char *line) {
    FILE *f = fopen(path, "a");
    if (!f) return -1;
    fprintf(f, "%s\n", line);
    fclose(f);
    return 0;
}

static int trim_and_rewrite(ConversationMemory *mem, int keep) {
    if (keep >= mem->n_turns) return 0;

    FILE *f = fopen(mem->path, "w");
    if (!f) return -1;

    int start = mem->n_turns - keep;
    for (int i = start; i < mem->n_turns; i++) {
        cJSON *obj = cJSON_CreateObject();
        cJSON_AddStringToObject(obj, "role", mem->turns[i].role);
        cJSON_AddStringToObject(obj, "content", mem->turns[i].content);
        cJSON_AddNumberToObject(obj, "ts", mem->turns[i].timestamp);
        char *line = cJSON_PrintUnformatted(obj);
        cJSON_Delete(obj);
        if (line) {
            fprintf(f, "%s\n", line);
            cJSON_free(line);
        }
    }
    fclose(f);

    /* free discarded turns (indices 0..start-1) */
    for (int i = 0; i < start; i++) {
        turn_free(&mem->turns[i]);
    }
    /* shift kept turns (indices start..n_turns-1) to 0..keep-1 */
    for (int i = 0; i < keep; i++) {
        mem->turns[i] = mem->turns[start + i];
    }
    mem->n_turns = keep;
    return 0;
}

ConversationMemory *memory_create(const char *path, int max_turns, int max_tokens) {
    if (!path) return NULL;
    if (max_turns <= 0) max_turns = 50;
    if (max_tokens <= 0) max_tokens = 4000;

    char *expanded = expand_path(path);
    if (!expanded) return NULL;

    ensure_parent_dirs(expanded);

    ConversationMemory *mem = calloc(1, sizeof(ConversationMemory));
    if (!mem) {
        free(expanded);
        return NULL;
    }

    mem->path = expanded;
    mem->max_turns = max_turns;
    mem->max_tokens = max_tokens;
    mem->capacity = 16;
    mem->turns = calloc((size_t)mem->capacity, sizeof(Turn));
    if (!mem->turns) {
        free(mem->path);
        free(mem);
        return NULL;
    }

    load_from_file(mem);
    return mem;
}

void memory_destroy(ConversationMemory *mem) {
    if (!mem) return;
    for (int i = 0; i < mem->n_turns; i++) turn_free(&mem->turns[i]);
    free(mem->turns);
    free(mem->path);
    free(mem);
}

int memory_add_turn(ConversationMemory *mem, const char *role, const char *content) {
    if (!mem || !role || !content) return -1;

    if (mem->n_turns >= mem->capacity) {
        int new_cap = mem->capacity * 2;
        Turn *next = realloc(mem->turns, (size_t)new_cap * sizeof(Turn));
        if (!next) return -1;
        mem->turns = next;
        mem->capacity = new_cap;
    }

    double ts = (double)time(NULL);
    Turn *t = &mem->turns[mem->n_turns];
    t->role = strdup(role);
    t->content = strdup(content);
    t->timestamp = ts;
    t->approx_tokens = approx_tokens(content);

    if (!t->role || !t->content) {
        turn_free(t);
        return -1;
    }

    cJSON *obj = cJSON_CreateObject();
    cJSON_AddStringToObject(obj, "role", role);
    cJSON_AddStringToObject(obj, "content", content);
    cJSON_AddNumberToObject(obj, "ts", ts);
    char *line = cJSON_PrintUnformatted(obj);
    cJSON_Delete(obj);
    if (!line) {
        turn_free(t);
        return -1;
    }

    if (append_line_to_file(mem->path, line) != 0) {
        cJSON_free(line);
        turn_free(t);
        return -1;
    }
    cJSON_free(line);

    mem->n_turns++;

    if (mem->n_turns > mem->max_turns) {
        trim_and_rewrite(mem, mem->max_turns);
    }
    return 0;
}

int memory_get_context(ConversationMemory *mem, MemoryTurn *turns_out, int max_out) {
    if (!mem || !turns_out || max_out <= 0) return 0;

    int tokens = 0;
    int count = 0;
    for (int i = mem->n_turns - 1; i >= 0 && count < max_out; i--) {
        int add = mem->turns[i].approx_tokens;
        if (tokens + add > mem->max_tokens) break;
        tokens += add;
        turns_out[count].role = mem->turns[i].role;
        turns_out[count].content = mem->turns[i].content;
        turns_out[count].timestamp = mem->turns[i].timestamp;
        count++;
    }
    return count;
}

char *memory_format_context(ConversationMemory *mem) {
    if (!mem) return NULL;

    MemoryTurn tmp[64];
    int n = memory_get_context(mem, tmp, 64);
    if (n == 0) {
        char *s = malloc(22);
        if (s) strcpy(s, "Previous conversation:\n");
        return s;
    }

    size_t total = 22;  /* "Previous conversation:\n" */
    for (int i = n - 1; i >= 0; i--) {  /* chronological: oldest first */
        const char *role = tmp[i].role;
        const char *content = tmp[i].content;
        const char *label = (role && strcmp(role, "user") == 0) ? "User" : "Assistant";
        total += strlen(label) + 2 + strlen(content ? content : "") + 1;
    }

    char *out = malloc(total + 1);
    if (!out) return NULL;
    char *p = out;
    p += sprintf(p, "Previous conversation:\n");
    for (int i = n - 1; i >= 0; i--) {
        const char *role = tmp[i].role;
        const char *content = tmp[i].content ? tmp[i].content : "";
        const char *label = (role && strcmp(role, "user") == 0) ? "User" : "Assistant";
        p += sprintf(p, "%s: %s\n", label, content);
    }
    *p = '\0';
    return out;
}

int memory_turn_count(const ConversationMemory *mem) {
    return mem ? mem->n_turns : 0;
}

void memory_clear(ConversationMemory *mem) {
    if (!mem) return;
    for (int i = 0; i < mem->n_turns; i++) turn_free(&mem->turns[i]);
    mem->n_turns = 0;
    FILE *f = fopen(mem->path, "w");
    if (f) fclose(f);
}

int memory_search(ConversationMemory *mem, const char *keyword,
                  MemoryTurn *turns_out, int max_out) {
    if (!mem || !keyword || !turns_out || max_out <= 0) return 0;

    int count = 0;
    for (int i = 0; i < mem->n_turns && count < max_out; i++) {
        if (strcasestr(mem->turns[i].content, keyword)) {
            turns_out[count].role = mem->turns[i].role;
            turns_out[count].content = mem->turns[i].content;
            turns_out[count].timestamp = mem->turns[i].timestamp;
            count++;
        }
    }
    return count;
}
