/**
 * test_diarizer.c — Tests for the SpeakerDiarizer module.
 *
 * Covers: create/destroy, NULL handling, speaker identification via synthetic
 * embeddings, labels, reset. No model files required (embedding-only mode).
 *
 * Build: make test-diarizer
 * Run:   ./build/test-diarizer
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "speaker_diarizer.h"

#define EMB_DIM 192

static int passed = 0, failed = 0;

#define CHECK(cond, msg) do { \
    if (cond) { printf("  [PASS] %s\n", msg); passed++; } \
    else { printf("  [FAIL] %s\n", msg); failed++; } \
} while (0)

/* Create a unit-norm embedding with a deterministic pattern. */
static void make_embedding(float *out, int dim, unsigned seed) {
    float norm_sq = 0.0f;
    for (int i = 0; i < dim; i++) {
        out[i] = (float)((seed * 31 + i * 7) % 1000) / 500.0f - 1.0f;
        norm_sq += out[i] * out[i];
    }
    float inv = 1.0f / sqrtf(norm_sq);
    for (int i = 0; i < dim; i++)
        out[i] *= inv;
}

/* Create embedding orthogonal to a given vector (dot product ≈ 0). */
static void make_orthogonal_embedding(float *out, const float *ref, int dim, unsigned seed) {
    float dot = 0.0f;
    for (int i = 0; i < dim; i++) {
        out[i] = (float)((seed * 17 + i * 11) % 1000) / 500.0f - 1.0f;
        dot += out[i] * ref[i];
    }
    for (int i = 0; i < dim; i++)
        out[i] -= dot * ref[i];
    float norm_sq = 0.0f;
    for (int i = 0; i < dim; i++)
        norm_sq += out[i] * out[i];
    if (norm_sq > 1e-10f) {
        float inv = 1.0f / sqrtf(norm_sq);
        for (int i = 0; i < dim; i++)
            out[i] *= inv;
    }
}

int main(void) {
    printf("═══ SpeakerDiarizer Tests ═══\n\n");

    /* ── 1. Create/destroy with valid/invalid params ───────────────────── */
    printf("── Create/Destroy ──\n");

    SpeakerDiarizer *d = diarizer_create(NULL, 0.75f, 8);
    CHECK(d != NULL, "create with NULL path (embedding-only) succeeds");
    if (d) {
        diarizer_destroy(d);
    }

    CHECK(diarizer_create("/nonexistent/encoder.onnx", 0.75f, 8) == NULL,
          "create with nonexistent path returns NULL");

    d = diarizer_create(NULL, 0.0f, 8);
    CHECK(d != NULL, "create with default threshold (0) uses default");
    if (d)
        diarizer_destroy(d);

    /* ── 2. NULL handling ─────────────────────────────────────────────── */
    printf("\n── NULL Handling ──\n");

    diarizer_destroy(NULL);
    CHECK(1, "destroy NULL is safe");

    diarizer_reset(NULL);
    CHECK(1, "reset NULL is safe");

    CHECK(diarizer_identify(NULL, NULL, 0) == -1, "identify NULL diarizer returns -1");
    CHECK(diarizer_identify_embedding(NULL, NULL, EMB_DIM) == -1,
          "identify_embedding NULL diarizer returns -1");

    d = diarizer_create(NULL, 0.75f, 8);
    if (d) {
        CHECK(diarizer_identify(d, NULL, 16000) == -1, "identify NULL audio returns -1");
        float emb[EMB_DIM];
        make_embedding(emb, EMB_DIM, 42);
        CHECK(diarizer_identify_embedding(d, NULL, EMB_DIM) == -1,
              "identify_embedding NULL embedding returns -1");
        CHECK(diarizer_speaker_count(NULL) == 0, "speaker_count NULL returns 0");
        CHECK(diarizer_get_embedding(NULL, 0, emb) == -1, "get_embedding NULL diarizer returns -1");
        CHECK(diarizer_get_embedding(d, 0, NULL) == -1, "get_embedding NULL out returns -1");
        CHECK(diarizer_set_label(NULL, 0, "x") == -1, "set_label NULL diarizer returns -1");
        CHECK(diarizer_get_label(NULL, 0) == NULL, "get_label NULL diarizer returns NULL");
        diarizer_destroy(d);
    }

    /* ── 3. speaker_count starts at 0 ─────────────────────────────────── */
    printf("\n── Speaker Count ──\n");

    d = diarizer_create(NULL, 0.75f, 8);
    CHECK(d != NULL && diarizer_speaker_count(d) == 0,
          "speaker_count starts at 0");
    if (d)
        diarizer_destroy(d);

    /* ── 4. identify_embedding registers speaker 0 ────────────────────── */
    printf("\n── Synthetic Embedding: First Speaker ──\n");

    d = diarizer_create(NULL, 0.75f, 8);
    if (d) {
        float emb[EMB_DIM];
        make_embedding(emb, EMB_DIM, 100);
        int id = diarizer_identify_embedding(d, emb, EMB_DIM);
        CHECK(id == 0, "first embedding registers speaker 0");
        CHECK(diarizer_speaker_count(d) == 1, "speaker_count is 1 after first");
        diarizer_destroy(d);
    }

    /* ── 5. Same embedding returns speaker 0 ──────────────────────────── */
    printf("\n── Same Embedding: Same Speaker ──\n");

    d = diarizer_create(NULL, 0.75f, 8);
    if (d) {
        float emb[EMB_DIM];
        make_embedding(emb, EMB_DIM, 200);
        int id1 = diarizer_identify_embedding(d, emb, EMB_DIM);
        int id2 = diarizer_identify_embedding(d, emb, EMB_DIM);
        CHECK(id1 == 0 && id2 == 0, "same embedding returns speaker 0 twice");
        CHECK(diarizer_speaker_count(d) == 1, "still one speaker");
        diarizer_destroy(d);
    }

    /* ── 6. Orthogonal embedding registers speaker 1 ─────────────────────── */
    printf("\n── Orthogonal Embedding: New Speaker ──\n");

    d = diarizer_create(NULL, 0.75f, 8);
    if (d) {
        float emb0[EMB_DIM], emb1[EMB_DIM];
        make_embedding(emb0, EMB_DIM, 300);
        make_orthogonal_embedding(emb1, emb0, EMB_DIM, 400);
        int id0 = diarizer_identify_embedding(d, emb0, EMB_DIM);
        int id1 = diarizer_identify_embedding(d, emb1, EMB_DIM);
        CHECK(id0 == 0, "first embedding → speaker 0");
        CHECK(id1 == 1, "orthogonal embedding → speaker 1");
        CHECK(diarizer_speaker_count(d) == 2, "two speakers");
        diarizer_destroy(d);
    }

    /* ── 7. Label set/get ────────────────────────────────────────────── */
    printf("\n── Labels ──\n");

    d = diarizer_create(NULL, 0.75f, 8);
    if (d) {
        float emb[EMB_DIM];
        make_embedding(emb, EMB_DIM, 500);
        (void)diarizer_identify_embedding(d, emb, EMB_DIM);
        CHECK(diarizer_set_label(d, 0, "Alice") == 0, "set_label succeeds");
        CHECK(diarizer_get_label(d, 0) != NULL && strcmp(diarizer_get_label(d, 0), "Alice") == 0,
              "get_label returns Alice");
        CHECK(diarizer_get_label(d, 99) == NULL, "get_label invalid id returns NULL");
        CHECK(diarizer_set_label(d, 99, "X") == -1, "set_label invalid id returns -1");
        diarizer_destroy(d);
    }

    /* ── 8. Reset clears all speakers ───────────────────────────────────── */
    printf("\n── Reset ──\n");

    d = diarizer_create(NULL, 0.75f, 8);
    if (d) {
        float emb[EMB_DIM];
        make_embedding(emb, EMB_DIM, 600);
        (void)diarizer_identify_embedding(d, emb, EMB_DIM);
        CHECK(diarizer_speaker_count(d) == 1, "one speaker before reset");
        diarizer_reset(d);
        CHECK(diarizer_speaker_count(d) == 0, "speaker_count 0 after reset");
        int id = diarizer_identify_embedding(d, emb, EMB_DIM);
        CHECK(id == 0, "embedding after reset registers as new speaker 0");
        diarizer_destroy(d);
    }

    /* ── 9. get_embedding returns centroid ─────────────────────────────── */
    printf("\n── Get Embedding ──\n");

    d = diarizer_create(NULL, 0.75f, 8);
    if (d) {
        float emb[EMB_DIM], out[EMB_DIM];
        make_embedding(emb, EMB_DIM, 700);
        (void)diarizer_identify_embedding(d, emb, EMB_DIM);
        int ret = diarizer_get_embedding(d, 0, out);
        CHECK(ret == EMB_DIM, "get_embedding returns dim");
        float sim = 0.0f;
        for (int i = 0; i < EMB_DIM; i++)
            sim += emb[i] * out[i];
        CHECK(sim > 0.99f && sim < 1.01f, "centroid matches input (L2-norm, cos~1)");
        CHECK(diarizer_get_embedding(d, 99, out) == -1, "get_embedding invalid id returns -1");
        diarizer_destroy(d);
    }

    /* ── 10. Wrong dim in identify_embedding ───────────────────────────── */
    printf("\n── Invalid Dim ──\n");

    d = diarizer_create(NULL, 0.75f, 8);
    if (d) {
        float emb[256];
        for (int i = 0; i < 256; i++)
            emb[i] = (i < 192) ? 0.0f : 0.1f;
        CHECK(diarizer_identify_embedding(d, emb, 256) == -1,
              "identify_embedding wrong dim returns -1");
        diarizer_destroy(d);
    }

    /* ── 11. Create with max_speakers=1 (single-speaker mode) ────────── */
    printf("\n── Single Speaker Mode ──\n");

    d = diarizer_create(NULL, 0.75f, 1);
    CHECK(d != NULL, "create with max_speakers=1 succeeds");
    if (d) {
        float emb0[EMB_DIM], emb1[EMB_DIM];
        make_embedding(emb0, EMB_DIM, 800);
        make_orthogonal_embedding(emb1, emb0, EMB_DIM, 801);
        int id0 = diarizer_identify_embedding(d, emb0, EMB_DIM);
        CHECK(id0 == 0, "first speaker in single-speaker mode is 0");
        int id1 = diarizer_identify_embedding(d, emb1, EMB_DIM);
        CHECK(id1 == -1, "second distinct speaker in single-speaker mode returns -1");
        CHECK(diarizer_speaker_count(d) == 1, "still only 1 speaker after reject");
        diarizer_destroy(d);
    }

    /* ── 12. Create with max_speakers=0 edge case ─────────────────── */
    printf("\n── Max Speakers Zero ──\n");

    d = diarizer_create(NULL, 0.75f, 0);
    if (d) {
        float emb[EMB_DIM];
        make_embedding(emb, EMB_DIM, 810);
        int id = diarizer_identify_embedding(d, emb, EMB_DIM);
        CHECK(id == -1, "max_speakers=0 rejects all embeddings");
        CHECK(diarizer_speaker_count(d) == 0, "no speakers registered with max=0");
        diarizer_destroy(d);
    } else {
        CHECK(1, "create with max_speakers=0 returns NULL (acceptable)");
    }

    /* ── 13. Create with negative threshold ───────────────────────── */
    printf("\n── Negative Threshold ──\n");

    d = diarizer_create(NULL, -1.0f, 8);
    CHECK(d != NULL, "create with negative threshold succeeds (uses default)");
    if (d)
        diarizer_destroy(d);

    /* ── 14. Label overwrite ──────────────────────────────────────── */
    printf("\n── Label Overwrite ──\n");

    d = diarizer_create(NULL, 0.75f, 8);
    if (d) {
        float emb[EMB_DIM];
        make_embedding(emb, EMB_DIM, 820);
        (void)diarizer_identify_embedding(d, emb, EMB_DIM);
        CHECK(diarizer_set_label(d, 0, "Alice") == 0, "set initial label");
        CHECK(diarizer_set_label(d, 0, "Bob") == 0, "overwrite label succeeds");
        const char *lbl = diarizer_get_label(d, 0);
        CHECK(lbl != NULL && strcmp(lbl, "Bob") == 0, "get_label returns overwritten value");
        diarizer_destroy(d);
    }

    /* ── 15. Label with NULL value ────────────────────────────────── */
    printf("\n── NULL Label ──\n");

    d = diarizer_create(NULL, 0.75f, 8);
    if (d) {
        float emb[EMB_DIM];
        make_embedding(emb, EMB_DIM, 830);
        (void)diarizer_identify_embedding(d, emb, EMB_DIM);
        int ret = diarizer_set_label(d, 0, NULL);
        CHECK(ret == 0 || ret == -1, "set_label NULL value handled gracefully");
        diarizer_destroy(d);
    }

    /* ── 16. Label with empty string ──────────────────────────────── */
    printf("\n── Empty Label ──\n");

    d = diarizer_create(NULL, 0.75f, 8);
    if (d) {
        float emb[EMB_DIM];
        make_embedding(emb, EMB_DIM, 835);
        (void)diarizer_identify_embedding(d, emb, EMB_DIM);
        CHECK(diarizer_set_label(d, 0, "") == 0, "set_label empty string succeeds");
        const char *lbl = diarizer_get_label(d, 0);
        CHECK(lbl != NULL && strcmp(lbl, "") == 0, "get_label returns empty string");
        diarizer_destroy(d);
    }

    /* ── 17. Multiple speakers up to max ──────────────────────────── */
    printf("\n── Fill Max Speakers ──\n");

    d = diarizer_create(NULL, 0.01f, 4);  /* very low threshold → almost everything is new */
    if (d) {
        float emb[EMB_DIM];
        int registered = 0;
        for (int s = 0; s < 6; s++) {
            make_embedding(emb, EMB_DIM, 900 + s * 100);
            int id = diarizer_identify_embedding(d, emb, EMB_DIM);
            if (id >= 0) registered++;
        }
        CHECK(diarizer_speaker_count(d) <= 4, "speaker_count does not exceed max_speakers=4");
        diarizer_destroy(d);
    }

    /* ── 18. Centroid averaging (multiple embeds for same speaker) ── */
    printf("\n── Centroid Averaging ──\n");

    d = diarizer_create(NULL, 0.75f, 8);
    if (d) {
        float emb[EMB_DIM], perturbed[EMB_DIM], centroid[EMB_DIM];
        make_embedding(emb, EMB_DIM, 950);
        (void)diarizer_identify_embedding(d, emb, EMB_DIM);

        /* Feed slightly perturbed version of same embedding */
        for (int i = 0; i < EMB_DIM; i++)
            perturbed[i] = emb[i] + 0.01f * ((i % 3) - 1);
        /* Re-normalize */
        float norm_sq = 0.0f;
        for (int i = 0; i < EMB_DIM; i++)
            norm_sq += perturbed[i] * perturbed[i];
        float inv = 1.0f / sqrtf(norm_sq);
        for (int i = 0; i < EMB_DIM; i++)
            perturbed[i] *= inv;

        int id2 = diarizer_identify_embedding(d, perturbed, EMB_DIM);
        CHECK(id2 == 0, "slightly perturbed embedding matches speaker 0");

        /* Centroid should now differ slightly from original */
        diarizer_get_embedding(d, 0, centroid);
        float dot_orig = 0.0f, dot_pert = 0.0f;
        for (int i = 0; i < EMB_DIM; i++) {
            dot_orig += centroid[i] * emb[i];
            dot_pert += centroid[i] * perturbed[i];
        }
        CHECK(dot_orig > 0.95f, "centroid still close to original embedding");
        CHECK(dot_pert > 0.95f, "centroid still close to perturbed embedding");
        diarizer_destroy(d);
    }

    /* ── 19. Get embedding before any speakers registered ─────────── */
    printf("\n── Get Embedding Empty ──\n");

    d = diarizer_create(NULL, 0.75f, 8);
    if (d) {
        float out[EMB_DIM];
        CHECK(diarizer_get_embedding(d, 0, out) == -1,
              "get_embedding on empty diarizer returns -1");
        diarizer_destroy(d);
    }

    /* ── 20. Multiple instances simultaneously ────────────────────── */
    printf("\n── Multiple Instances ──\n");

    {
        SpeakerDiarizer *d1 = diarizer_create(NULL, 0.75f, 8);
        SpeakerDiarizer *d2 = diarizer_create(NULL, 0.80f, 4);
        CHECK(d1 != NULL && d2 != NULL, "two diarizer instances created");
        if (d1 && d2) {
            float emb[EMB_DIM];
            make_embedding(emb, EMB_DIM, 1000);
            int id1 = diarizer_identify_embedding(d1, emb, EMB_DIM);
            int id2 = diarizer_identify_embedding(d2, emb, EMB_DIM);
            CHECK(id1 == 0 && id2 == 0, "same embedding in both instances gets speaker 0");
            CHECK(diarizer_speaker_count(d1) == 1, "instance 1 has 1 speaker");
            CHECK(diarizer_speaker_count(d2) == 1, "instance 2 has 1 speaker");
            diarizer_reset(d1);
            CHECK(diarizer_speaker_count(d1) == 0, "reset d1 doesn't affect d2");
            CHECK(diarizer_speaker_count(d2) == 1, "d2 still has 1 speaker after d1 reset");
        }
        if (d1) diarizer_destroy(d1);
        if (d2) diarizer_destroy(d2);
    }

    /* ── 21. Repeated reset is idempotent ─────────────────────────── */
    printf("\n── Repeated Reset ──\n");

    d = diarizer_create(NULL, 0.75f, 8);
    if (d) {
        float emb[EMB_DIM];
        make_embedding(emb, EMB_DIM, 1100);
        (void)diarizer_identify_embedding(d, emb, EMB_DIM);
        diarizer_reset(d);
        diarizer_reset(d);
        diarizer_reset(d);
        CHECK(diarizer_speaker_count(d) == 0, "triple reset leaves 0 speakers");
        int id = diarizer_identify_embedding(d, emb, EMB_DIM);
        CHECK(id == 0, "can register speaker after repeated resets");
        diarizer_destroy(d);
    }

    /* ── 22. Reset clears labels too ──────────────────────────────── */
    printf("\n── Reset Clears Labels ──\n");

    d = diarizer_create(NULL, 0.75f, 8);
    if (d) {
        float emb[EMB_DIM];
        make_embedding(emb, EMB_DIM, 1200);
        (void)diarizer_identify_embedding(d, emb, EMB_DIM);
        diarizer_set_label(d, 0, "Alice");
        diarizer_reset(d);
        CHECK(diarizer_get_label(d, 0) == NULL,
              "label cleared after reset");
        diarizer_destroy(d);
    }

    /* ── 23. Speaker persistence across many identifications ──────── */
    printf("\n── Speaker Persistence ──\n");

    d = diarizer_create(NULL, 0.75f, 8);
    if (d) {
        float emb[EMB_DIM];
        make_embedding(emb, EMB_DIM, 1300);
        for (int i = 0; i < 20; i++) {
            int id = diarizer_identify_embedding(d, emb, EMB_DIM);
            CHECK(id == 0, "speaker 0 persists across repeated identifications");
        }
        CHECK(diarizer_speaker_count(d) == 1, "still 1 speaker after 20 identifications");
        diarizer_destroy(d);
    }

    /* ── 24. Negative speaker_id in get/set ───────────────────────── */
    printf("\n── Negative Speaker ID ──\n");

    d = diarizer_create(NULL, 0.75f, 8);
    if (d) {
        float out[EMB_DIM];
        CHECK(diarizer_get_embedding(d, -1, out) == -1, "get_embedding(-1) returns -1");
        CHECK(diarizer_set_label(d, -1, "X") == -1, "set_label(-1) returns -1");
        CHECK(diarizer_get_label(d, -1) == NULL, "get_label(-1) returns NULL");
        diarizer_destroy(d);
    }

    /* ── 25. Identify zero-length embedding ───────────────────────── */
    printf("\n── Zero Dim Embedding ──\n");

    d = diarizer_create(NULL, 0.75f, 8);
    if (d) {
        float emb[1] = {1.0f};
        CHECK(diarizer_identify_embedding(d, emb, 0) == -1,
              "identify_embedding dim=0 returns -1");
        CHECK(diarizer_identify_embedding(d, emb, -1) == -1,
              "identify_embedding dim=-1 returns -1");
        diarizer_destroy(d);
    }

    printf("\n═══ Results: %d passed, %d failed ═══\n", passed, failed);
    return failed > 0 ? 1 : 0;
}
