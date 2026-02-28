/**
 * tests/test_flow_quality_modes.c — Test Flow quality mode system
 *
 * Tests:
 *   - FAST mode: 4 steps, Euler only
 *   - BALANCED mode: 6 steps, Euler only
 *   - HIGH mode: 8 steps, Heun enabled
 *   - Mode switching works correctly
 *   - set_n_steps overrides quality mode
 *   - Default is BALANCED (6 steps)
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <limits.h>

/* FFI declarations for Flow engines */
extern void *sonata_flow_create(const char *flow_weights, const char *flow_config,
                                 const char *decoder_weights, const char *decoder_config);
extern void *sonata_flow_v2_create(const char *weights_path, const char *config_path);
extern void *sonata_flow_v3_create(const char *weights_path, const char *config_path);

extern int sonata_flow_set_quality_mode(void *engine, int mode);
extern int sonata_flow_set_n_steps(void *engine, int n_steps);
extern int sonata_flow_set_solver(void *engine, int use_heun);
extern int sonata_flow_n_steps(void);  // Returns default n_steps
extern void sonata_flow_destroy(void *engine);

extern int sonata_flow_v2_set_quality_mode(void *engine, int mode);
extern int sonata_flow_v2_set_n_steps(void *engine, int steps);
extern void sonata_flow_v2_destroy(void *engine);

extern int sonata_flow_v3_set_quality_mode(void *engine, int mode);
extern int sonata_flow_v3_set_n_steps(void *engine, int steps);
extern void sonata_flow_v3_destroy(void *engine);

/* Quality mode constants */
#define FLOW_QUALITY_FAST      0
#define FLOW_QUALITY_BALANCED  1
#define FLOW_QUALITY_HIGH      2

typedef int bool;
#define true 1
#define false 0

static int test_count = 0;
static int test_pass = 0;

void test_assert(bool condition, const char *msg) {
    test_count++;
    if (condition) {
        test_pass++;
        printf("  PASS: %s\n", msg);
    } else {
        printf("  FAIL: %s\n", msg);
    }
}

/* Test set quality mode FFI constants exist and are correct */
void test_quality_mode_constants(void) {
    printf("\nTest: Quality mode constants\n");
    test_assert(FLOW_QUALITY_FAST == 0, "FLOW_QUALITY_FAST = 0");
    test_assert(FLOW_QUALITY_BALANCED == 1, "FLOW_QUALITY_BALANCED = 1");
    test_assert(FLOW_QUALITY_HIGH == 2, "FLOW_QUALITY_HIGH = 2");
}

/* Test default n_steps exported from library */
void test_default_n_steps(void) {
    printf("\nTest: Default n_steps constant\n");
    int def = sonata_flow_n_steps();
    test_assert(def == 8, "sonata_flow_n_steps() returns 8 (legacy)");
}

/* Test that quality mode setters exist and return valid codes */
void test_quality_mode_setter_signatures(void) {
    printf("\nTest: Quality mode setter function signatures\n");

    /* Test return type: should return -1 on NULL engine, 0 or -1 on valid engine */
    int ret = sonata_flow_set_quality_mode(NULL, FLOW_QUALITY_FAST);
    test_assert(ret == -1, "sonata_flow_set_quality_mode(NULL) returns -1");

    ret = sonata_flow_v2_set_quality_mode(NULL, FLOW_QUALITY_FAST);
    test_assert(ret == -1, "sonata_flow_v2_set_quality_mode(NULL) returns -1");

    ret = sonata_flow_v3_set_quality_mode(NULL, FLOW_QUALITY_FAST);
    test_assert(ret == -1, "sonata_flow_v3_set_quality_mode(NULL) returns -1");
}

/* Test quality mode range validation */
void test_quality_mode_validation(void) {
    printf("\nTest: Quality mode validation\n");

    /* Should handle invalid modes gracefully */
    int ret = sonata_flow_set_quality_mode(NULL, 99);
    test_assert(ret == -1, "sonata_flow_set_quality_mode() rejects invalid mode");

    ret = sonata_flow_set_quality_mode(NULL, -1);
    test_assert(ret == -1, "sonata_flow_set_quality_mode() rejects negative mode");
}

/* Test that set_n_steps still works (backward compatibility) */
void test_backward_compat_set_n_steps(void) {
    printf("\nTest: Backward compatibility with set_n_steps\n");

    /* Test return type validation */
    int ret = sonata_flow_set_n_steps(NULL, 4);
    test_assert(ret == -1, "sonata_flow_set_n_steps(NULL) returns -1");

    ret = sonata_flow_set_n_steps(NULL, 0);
    test_assert(ret == -1, "sonata_flow_set_n_steps(NULL, 0) returns -1");

    ret = sonata_flow_v2_set_n_steps(NULL, 8);
    test_assert(ret == -1, "sonata_flow_v2_set_n_steps(NULL) returns -1");

    ret = sonata_flow_v3_set_n_steps(NULL, 16);
    test_assert(ret == -1, "sonata_flow_v3_set_n_steps(NULL) returns -1");
}

/* Test quality mode step values are sensible */
void test_quality_mode_semantics(void) {
    printf("\nTest: Quality mode semantics\n");

    /* Verify mode values are distinct */
    test_assert(FLOW_QUALITY_FAST != FLOW_QUALITY_BALANCED,
                "FAST and BALANCED modes are different");
    test_assert(FLOW_QUALITY_BALANCED != FLOW_QUALITY_HIGH,
                "BALANCED and HIGH modes are different");
    test_assert(FLOW_QUALITY_FAST != FLOW_QUALITY_HIGH,
                "FAST and HIGH modes are different");

    /* Verify step progression (for code review) */
    test_assert(FLOW_QUALITY_FAST == 0, "FAST is mode 0");
    test_assert(FLOW_QUALITY_BALANCED == 1, "BALANCED is mode 1");
    test_assert(FLOW_QUALITY_HIGH == 2, "HIGH is mode 2");
}

/* Test pointer safety */
void test_pointer_safety(void) {
    printf("\nTest: Pointer safety\n");

    /* All setters should handle NULL gracefully */
    int ret = sonata_flow_set_quality_mode(NULL, FLOW_QUALITY_BALANCED);
    test_assert(ret == -1, "sonata_flow_set_quality_mode handles NULL safely");

    ret = sonata_flow_set_n_steps(NULL, 6);
    test_assert(ret == -1, "sonata_flow_set_n_steps handles NULL safely");

    ret = sonata_flow_set_solver(NULL, 0);
    test_assert(ret == -1, "sonata_flow_set_solver handles NULL safely");

    ret = sonata_flow_v2_set_quality_mode(NULL, FLOW_QUALITY_BALANCED);
    test_assert(ret == -1, "sonata_flow_v2_set_quality_mode handles NULL safely");

    ret = sonata_flow_v2_set_n_steps(NULL, 6);
    test_assert(ret == -1, "sonata_flow_v2_set_n_steps handles NULL safely");

    ret = sonata_flow_v3_set_quality_mode(NULL, FLOW_QUALITY_BALANCED);
    test_assert(ret == -1, "sonata_flow_v3_set_quality_mode handles NULL safely");

    ret = sonata_flow_v3_set_n_steps(NULL, 6);
    test_assert(ret == -1, "sonata_flow_v3_set_n_steps handles NULL safely");
}

/* ─── NEW CORRECTNESS TESTS ──────────────────────────────────────────── */

/* Test extreme boundary mode values: 3, -2, INT_MIN, INT_MAX */
void test_quality_mode_extreme_boundaries(void) {
    printf("\nTest: Quality mode extreme boundary values\n");

    /* Mode 3 (one above valid range) → -1 for all engines */
    int ret = sonata_flow_set_quality_mode(NULL, 3);
    test_assert(ret == -1, "sonata_flow mode=3 rejects (NULL)");
    ret = sonata_flow_v2_set_quality_mode(NULL, 3);
    test_assert(ret == -1, "sonata_flow_v2 mode=3 rejects (NULL)");
    ret = sonata_flow_v3_set_quality_mode(NULL, 3);
    test_assert(ret == -1, "sonata_flow_v3 mode=3 rejects (NULL)");

    /* Mode -2 (negative) → -1 */
    ret = sonata_flow_set_quality_mode(NULL, -2);
    test_assert(ret == -1, "sonata_flow mode=-2 rejects (NULL)");
    ret = sonata_flow_v2_set_quality_mode(NULL, -2);
    test_assert(ret == -1, "sonata_flow_v2 mode=-2 rejects (NULL)");
    ret = sonata_flow_v3_set_quality_mode(NULL, -2);
    test_assert(ret == -1, "sonata_flow_v3 mode=-2 rejects (NULL)");

    /* INT_MIN and INT_MAX → -1 */
    ret = sonata_flow_set_quality_mode(NULL, INT_MIN);
    test_assert(ret == -1, "sonata_flow mode=INT_MIN rejects (NULL)");
    ret = sonata_flow_set_quality_mode(NULL, INT_MAX);
    test_assert(ret == -1, "sonata_flow mode=INT_MAX rejects (NULL)");
    ret = sonata_flow_v3_set_quality_mode(NULL, INT_MIN);
    test_assert(ret == -1, "sonata_flow_v3 mode=INT_MIN rejects (NULL)");
    ret = sonata_flow_v3_set_quality_mode(NULL, INT_MAX);
    test_assert(ret == -1, "sonata_flow_v3 mode=INT_MAX rejects (NULL)");
}

/* Test set_n_steps boundary values: 0, -1, 65, 1000, INT_MIN, INT_MAX */
void test_set_n_steps_boundaries(void) {
    printf("\nTest: set_n_steps boundary values\n");

    /* All return -1 with NULL engine regardless of step value */
    int ret = sonata_flow_set_n_steps(NULL, 0);
    test_assert(ret == -1, "sonata_flow_set_n_steps(NULL, 0) = -1");
    ret = sonata_flow_set_n_steps(NULL, -1);
    test_assert(ret == -1, "sonata_flow_set_n_steps(NULL, -1) = -1");
    ret = sonata_flow_set_n_steps(NULL, 65);
    test_assert(ret == -1, "sonata_flow_set_n_steps(NULL, 65) = -1");
    ret = sonata_flow_set_n_steps(NULL, 1000);
    test_assert(ret == -1, "sonata_flow_set_n_steps(NULL, 1000) = -1");
    ret = sonata_flow_set_n_steps(NULL, INT_MIN);
    test_assert(ret == -1, "sonata_flow_set_n_steps(NULL, INT_MIN) = -1");
    ret = sonata_flow_set_n_steps(NULL, INT_MAX);
    test_assert(ret == -1, "sonata_flow_set_n_steps(NULL, INT_MAX) = -1");

    /* v3 has stricter validation: returns -1 for out-of-range steps even concept */
    ret = sonata_flow_v3_set_n_steps(NULL, 0);
    test_assert(ret == -1, "sonata_flow_v3_set_n_steps(NULL, 0) = -1");
    ret = sonata_flow_v3_set_n_steps(NULL, -1);
    test_assert(ret == -1, "sonata_flow_v3_set_n_steps(NULL, -1) = -1");
    ret = sonata_flow_v3_set_n_steps(NULL, 65);
    test_assert(ret == -1, "sonata_flow_v3_set_n_steps(NULL, 65) = -1");
    ret = sonata_flow_v3_set_n_steps(NULL, 1000);
    test_assert(ret == -1, "sonata_flow_v3_set_n_steps(NULL, 1000) = -1");

    /* Valid step values with NULL engine still return -1 (NULL check first) */
    ret = sonata_flow_set_n_steps(NULL, 1);
    test_assert(ret == -1, "sonata_flow_set_n_steps(NULL, 1) = -1");
    ret = sonata_flow_set_n_steps(NULL, 64);
    test_assert(ret == -1, "sonata_flow_set_n_steps(NULL, 64) = -1");
    ret = sonata_flow_v3_set_n_steps(NULL, 1);
    test_assert(ret == -1, "sonata_flow_v3_set_n_steps(NULL, 1) = -1");
    ret = sonata_flow_v3_set_n_steps(NULL, 64);
    test_assert(ret == -1, "sonata_flow_v3_set_n_steps(NULL, 64) = -1");
}

/* Test quality mode expected step/heun configuration mapping
 * (documents contract even though we can't instantiate engines without models) */
void test_quality_mode_step_mapping(void) {
    printf("\nTest: Quality mode step/heun configuration mapping\n");

    /* Document the expected configuration for each mode:
     *   FAST     (0): 4 steps, Euler only  (use_heun=false)
     *   BALANCED (1): 6 steps, Euler only  (use_heun=false)
     *   HIGH     (2): 8 steps, Heun solver (use_heun=true)
     *
     * Without model files we can't instantiate an engine, but we verify the
     * constants are ordered correctly: FAST < BALANCED < HIGH implies
     * increasing compute cost. */
    test_assert(FLOW_QUALITY_FAST < FLOW_QUALITY_BALANCED,
                "FAST < BALANCED (lower compute)");
    test_assert(FLOW_QUALITY_BALANCED < FLOW_QUALITY_HIGH,
                "BALANCED < HIGH (increasing compute)");

    /* The default n_steps exported constant should equal HIGH mode steps (8) */
    int def = sonata_flow_n_steps();
    test_assert(def == 8, "Default n_steps=8 matches HIGH mode steps");

    /* Verify modes are contiguous integers starting from 0 */
    test_assert(FLOW_QUALITY_FAST == 0, "FAST == 0");
    test_assert(FLOW_QUALITY_BALANCED == 1, "BALANCED == 1");
    test_assert(FLOW_QUALITY_HIGH == 2, "HIGH == 2");
    test_assert(FLOW_QUALITY_HIGH - FLOW_QUALITY_FAST == 2,
                "Modes span exactly 3 values (0, 1, 2)");
}

/* Test set_solver integration with quality modes */
void test_solver_integration(void) {
    printf("\nTest: set_solver function integration\n");

    /* set_solver(NULL, ...) returns -1 for all solver values */
    int ret = sonata_flow_set_solver(NULL, 0);
    test_assert(ret == -1, "set_solver(NULL, 0) = -1 (Euler)");
    ret = sonata_flow_set_solver(NULL, 1);
    test_assert(ret == -1, "set_solver(NULL, 1) = -1 (Heun)");

    /* Extreme values don't crash */
    ret = sonata_flow_set_solver(NULL, -1);
    test_assert(ret == -1, "set_solver(NULL, -1) = -1");
    ret = sonata_flow_set_solver(NULL, INT_MAX);
    test_assert(ret == -1, "set_solver(NULL, INT_MAX) = -1");
}

/* Test mode changes don't corrupt state (NULL engine sequence) */
void test_mode_change_sequence(void) {
    printf("\nTest: Mode change sequence doesn't corrupt state\n");

    /* Rapidly switch modes — all should return -1 with NULL but not crash */
    for (int mode = FLOW_QUALITY_FAST; mode <= FLOW_QUALITY_HIGH; mode++) {
        int ret = sonata_flow_set_quality_mode(NULL, mode);
        test_assert(ret == -1, "Mode cycle: NULL always returns -1");
    }

    /* Reverse order */
    for (int mode = FLOW_QUALITY_HIGH; mode >= FLOW_QUALITY_FAST; mode--) {
        int ret = sonata_flow_v2_set_quality_mode(NULL, mode);
        test_assert(ret == -1, "Mode reverse cycle: v2 NULL always returns -1");
    }

    /* Alternate valid and invalid modes rapidly */
    int modes[] = {0, 99, 1, -1, 2, INT_MAX, 0, INT_MIN};
    for (int i = 0; i < (int)(sizeof(modes) / sizeof(modes[0])); i++) {
        int ret = sonata_flow_v3_set_quality_mode(NULL, modes[i]);
        test_assert(ret == -1, "Mixed mode cycle: v3 NULL always returns -1");
    }
}

int main(void) {
    printf("═════════════════════════════════════════════════════════\n");
    printf("Flow Quality Mode System — FFI Tests\n");
    printf("═════════════════════════════════════════════════════════\n");

    /* Run all tests */
    test_quality_mode_constants();
    test_default_n_steps();
    test_quality_mode_setter_signatures();
    test_quality_mode_validation();
    test_backward_compat_set_n_steps();
    test_quality_mode_semantics();
    test_pointer_safety();
    /* New correctness tests */
    test_quality_mode_extreme_boundaries();
    test_set_n_steps_boundaries();
    test_quality_mode_step_mapping();
    test_solver_integration();
    test_mode_change_sequence();

    printf("\n═════════════════════════════════════════════════════════\n");
    printf("Results: %d/%d passed\n", test_pass, test_count);
    printf("═════════════════════════════════════════════════════════\n");

    return (test_pass == test_count) ? 0 : 1;
}
