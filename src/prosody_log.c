/**
 * prosody_log.c — Real-time prosody logging to JSONL.
 *
 * Each line is a self-contained JSON object with a "type" field:
 *   {"type":"segment", "text":"...", "pitch":1.08, ...}
 *   {"type":"turn", "turn_id":1, "vrl_ms":450.0, ...}
 *   {"type":"contour", "segment_id":"...", "f0":[...], "energy":[...]}
 *
 * Build: cc -O3 -shared -fPIC -o libprosody_log.dylib prosody_log.c cJSON.c
 */

#include "prosody_log.h"
#include "cJSON.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

struct ProsodyLog {
    FILE *fp;
    int   n_entries;
};

ProsodyLog *prosody_log_open(const char *path) {
    if (!path) return NULL;
    FILE *fp = fopen(path, "a");
    if (!fp) {
        fprintf(stderr, "[prosody_log] Cannot open: %s\n", path);
        return NULL;
    }
    ProsodyLog *log = (ProsodyLog *)calloc(1, sizeof(ProsodyLog));
    log->fp = fp;

    /* Write session marker */
    cJSON *obj = cJSON_CreateObject();
    cJSON_AddStringToObject(obj, "type", "session_start");
    time_t now = time(NULL);
    char tbuf[64];
    strftime(tbuf, sizeof(tbuf), "%Y-%m-%dT%H:%M:%S", localtime(&now));
    cJSON_AddStringToObject(obj, "timestamp", tbuf);
    char *s = cJSON_PrintUnformatted(obj);
    fprintf(fp, "%s\n", s);
    fflush(fp);
    free(s);
    cJSON_Delete(obj);

    fprintf(stderr, "[prosody_log] Logging to %s\n", path);
    return log;
}

void prosody_log_close(ProsodyLog *log) {
    if (!log) return;
    if (log->fp) {
        cJSON *obj = cJSON_CreateObject();
        cJSON_AddStringToObject(obj, "type", "session_end");
        cJSON_AddNumberToObject(obj, "n_entries", log->n_entries);
        char *s = cJSON_PrintUnformatted(obj);
        fprintf(log->fp, "%s\n", s);
        free(s);
        cJSON_Delete(obj);
        fclose(log->fp);
    }
    free(log);
}

void prosody_log_segment(ProsodyLog *log,
                         const char *text,
                         float pitch,
                         float rate,
                         float volume_db,
                         const char *emotion,
                         const char *contour,
                         int duration_ms) {
    if (!log || !log->fp) return;

    cJSON *obj = cJSON_CreateObject();
    cJSON_AddStringToObject(obj, "type", "segment");
    if (text) cJSON_AddStringToObject(obj, "text", text);
    cJSON_AddNumberToObject(obj, "pitch", pitch);
    cJSON_AddNumberToObject(obj, "rate", rate);
    cJSON_AddNumberToObject(obj, "volume_db", volume_db);
    if (emotion && emotion[0]) cJSON_AddStringToObject(obj, "emotion", emotion);
    if (contour && contour[0]) cJSON_AddStringToObject(obj, "contour", contour);
    cJSON_AddNumberToObject(obj, "duration_ms", duration_ms);

    char *s = cJSON_PrintUnformatted(obj);
    fprintf(log->fp, "%s\n", s);
    fflush(log->fp);
    free(s);
    cJSON_Delete(obj);
    log->n_entries++;
}

void prosody_log_turn(ProsodyLog *log,
                      int turn_id,
                      const char *user_text,
                      const char *response_text,
                      float user_rate_wps,
                      float response_pitch,
                      float response_rate,
                      float response_energy,
                      float vrl_ms,
                      float tts_rtf) {
    if (!log || !log->fp) return;

    cJSON *obj = cJSON_CreateObject();
    cJSON_AddStringToObject(obj, "type", "turn");
    cJSON_AddNumberToObject(obj, "turn_id", turn_id);
    if (user_text) cJSON_AddStringToObject(obj, "user_text", user_text);
    if (response_text) {
        /* Truncate long responses for log readability */
        char trunc[256];
        int len = (int)strlen(response_text);
        int copy = len < 255 ? len : 255;
        memcpy(trunc, response_text, copy);
        trunc[copy] = '\0';
        cJSON_AddStringToObject(obj, "response_text", trunc);
    }
    cJSON_AddNumberToObject(obj, "user_rate_wps", user_rate_wps);
    cJSON_AddNumberToObject(obj, "response_pitch", response_pitch);
    cJSON_AddNumberToObject(obj, "response_rate", response_rate);
    cJSON_AddNumberToObject(obj, "response_energy", response_energy);
    cJSON_AddNumberToObject(obj, "vrl_ms", vrl_ms);
    cJSON_AddNumberToObject(obj, "tts_rtf", tts_rtf);

    char *s = cJSON_PrintUnformatted(obj);
    fprintf(log->fp, "%s\n", s);
    fflush(log->fp);
    free(s);
    cJSON_Delete(obj);
    log->n_entries++;
}

void prosody_log_contour(ProsodyLog *log,
                         const char *segment_id,
                         const float *f0, const float *energy,
                         int n_frames, int sr) {
    if (!log || !log->fp || n_frames <= 0) return;

    cJSON *obj = cJSON_CreateObject();
    cJSON_AddStringToObject(obj, "type", "contour");
    if (segment_id) cJSON_AddStringToObject(obj, "segment_id", segment_id);
    cJSON_AddNumberToObject(obj, "n_frames", n_frames);
    cJSON_AddNumberToObject(obj, "sr", sr);

    /* Downsample to max 200 points for JSON size */
    int step = n_frames > 200 ? (n_frames + 199) / 200 : 1;
    int out_n = (n_frames + step - 1) / step;

    cJSON *f0_arr = cJSON_CreateArray();
    cJSON *energy_arr = cJSON_CreateArray();
    for (int i = 0; i < n_frames; i += step) {
        if (f0) cJSON_AddItemToArray(f0_arr, cJSON_CreateNumber(f0[i]));
        if (energy) cJSON_AddItemToArray(energy_arr, cJSON_CreateNumber(energy[i]));
    }
    cJSON_AddItemToObject(obj, "f0", f0_arr);
    cJSON_AddItemToObject(obj, "energy", energy_arr);

    char *s = cJSON_PrintUnformatted(obj);
    fprintf(log->fp, "%s\n", s);
    fflush(log->fp);
    free(s);
    cJSON_Delete(obj);
    log->n_entries++;
    (void)out_n;
}
