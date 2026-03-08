#!/bin/bash
set -e

# Ship Readiness Test Suite — Comprehensive test runner for Sonata
# Runs all Rust, C, and Python tests, validates build artifacts and models,
# and produces a final GO/NO-GO report.

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Counters
RUST_PASS=0
RUST_FAIL=0
C_PASS=0
C_FAIL=0
PY_PASS=0
PY_FAIL=0
BUILD_PASS=0
BUILD_FAIL=0
MODEL_PASS=0
MODEL_FAIL=0
SECURITY_PASS=0
SECURITY_FAIL=0

# Test results tracking
FAILED_TESTS=()

# Utility functions
print_header() {
    echo ""
    echo -e "${BLUE}╔════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${BLUE}║ $1${NC}"
    echo -e "${BLUE}╚════════════════════════════════════════════════════════════╝${NC}"
    echo ""
}

print_section() {
    echo ""
    echo -e "${BLUE}──── $1 ────${NC}"
}

print_pass() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_fail() {
    echo -e "${RED}✗ $1${NC}"
}

print_warn() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ $1${NC}"
}

# Get project root
PROJ_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${PROJ_ROOT}/build"
TARGET_DIR="${PROJ_ROOT}/target/release"
MODELS_DIR="${PROJ_ROOT}/models"

print_header "SONATA SHIP READINESS TEST SUITE"
echo "Project: $PROJ_ROOT"
echo "Started: $(date)"

# ─────────────────────────────────────────────────────────────────────────────
# 1. RUST TESTS
# ─────────────────────────────────────────────────────────────────────────────

print_section "1. Rust Tests (cargo test --workspace)"

cd "$PROJ_ROOT"
if cargo test --workspace 2>&1; then
    RUST_PASS=1
    print_pass "Rust workspace tests"
else
    RUST_FAIL=1
    print_fail "Rust workspace tests"
    FAILED_TESTS+=("Rust workspace tests")
fi

# ─────────────────────────────────────────────────────────────────────────────
# 2. C TESTS
# ─────────────────────────────────────────────────────────────────────────────

print_section "2. C Tests (make test)"

cd "$PROJ_ROOT"
if make test 2>&1 | tee /tmp/make_test.log; then
    C_PASS=1
    print_pass "C test suite"
else
    C_FAIL=1
    print_fail "C test suite"
    FAILED_TESTS+=("C test suite")
    # Try to extract summary
    if grep -q "FAILED" /tmp/make_test.log; then
        print_warn "Some C tests failed. Check output above for details."
    fi
fi

# ─────────────────────────────────────────────────────────────────────────────
# 3. PYTHON TESTS
# ─────────────────────────────────────────────────────────────────────────────

print_section "3. Python Tests"

python_tests=(
    "train/sonata/test_rvq_module.py"
    "scripts/test_codec_roundtrip.py"
    "experiments/personaplex/test_duplex.py"
)

for test in "${python_tests[@]}"; do
    if [ -f "$PROJ_ROOT/$test" ]; then
        if python "$PROJ_ROOT/$test" 2>&1 > /tmp/py_test.log; then
            PY_PASS=$((PY_PASS + 1))
            print_pass "$test"
        else
            PY_FAIL=$((PY_FAIL + 1))
            print_fail "$test"
            FAILED_TESTS+=("$test")
        fi
    fi
done

if [ $PY_PASS -eq 0 ] && [ $PY_FAIL -eq 0 ]; then
    print_warn "No Python tests found to run"
fi

# ─────────────────────────────────────────────────────────────────────────────
# 4. BUILD ARTIFACTS
# ─────────────────────────────────────────────────────────────────────────────

print_section "4. Build Artifacts"

dylibs=(
    "libsonata_lm.dylib"
    "libsonata_flow.dylib"
    "libsonata_speaker.dylib"
)

for dylib in "${dylibs[@]}"; do
    if [ -f "$TARGET_DIR/$dylib" ]; then
        SIZE=$(ls -lh "$TARGET_DIR/$dylib" | awk '{print $5}')
        print_pass "$dylib (${SIZE})"
        BUILD_PASS=$((BUILD_PASS + 1))
    else
        print_fail "$dylib not found"
        BUILD_FAIL=$((BUILD_FAIL + 1))
        FAILED_TESTS+=("Build artifact: $dylib")
    fi
done

# Check for undefined symbols
print_info "Checking for undefined symbols..."
for dylib in "${dylibs[@]}"; do
    if [ -f "$TARGET_DIR/$dylib" ]; then
        UNDEFINED=$(nm -u "$TARGET_DIR/$dylib" 2>/dev/null | grep -v "^$" || true)
        if [ -n "$UNDEFINED" ]; then
            print_warn "Undefined symbols in $dylib:"
            echo "$UNDEFINED" | head -5
        fi
    fi
done

# ─────────────────────────────────────────────────────────────────────────────
# 5. MODEL FILES
# ─────────────────────────────────────────────────────────────────────────────

print_section "5. Model Files"

models=(
    "sonata_flow_distilled/sonata_flow_distilled.safetensors"
    "sonata_flow_distilled/sonata_flow_distilled_config.json"
)

for model in "${models[@]}"; do
    if [ -f "$MODELS_DIR/$model" ]; then
        SIZE=$(ls -lh "$MODELS_DIR/$model" | awk '{print $5}')
        print_pass "$model (${SIZE})"
        MODEL_PASS=$((MODEL_PASS + 1))
    else
        print_fail "$model not found"
        MODEL_FAIL=$((MODEL_FAIL + 1))
        FAILED_TESTS+=("Model: $model")
    fi
done

# Validate model JSON
print_info "Validating model configs..."
for config in "$MODELS_DIR"/sonata_flow_distilled/*_config.json; do
    if [ -f "$config" ]; then
        if python3 -c "import json; json.load(open('$config'))" 2>/dev/null; then
            print_pass "Config valid: $(basename "$config")"
        else
            print_fail "Invalid JSON: $(basename "$config")"
            FAILED_TESTS+=("Model config: $config")
        fi
    fi
done

# ─────────────────────────────────────────────────────────────────────────────
# 6. SECURITY & QUALITY AUDITS
# ─────────────────────────────────────────────────────────────────────────────

print_section "6. Security & Quality Status"

# Check for P0 security issues
audit_reports=".claude/CORRECTNESS_AUDIT_REPORT.md .claude/RED_TEAM_REPORT.md .claude/SECURITY_AUDIT.md"
p0_found=0

for report in $audit_reports; do
    if [ -f "$PROJ_ROOT/$report" ]; then
        if grep -q "P0" "$PROJ_ROOT/$report"; then
            P0_COUNT=$(grep -c "P0" "$PROJ_ROOT/$report" || echo "0")
            if [ "$P0_COUNT" -gt 0 ]; then
                print_fail "Found $P0_COUNT P0 items in $report"
                p0_found=1
                FAILED_TESTS+=("Security audit P0: $report")
                SECURITY_FAIL=$((SECURITY_FAIL + 1))
            fi
        fi
    fi
done

if [ $p0_found -eq 0 ]; then
    print_pass "No P0 security issues found"
    SECURITY_PASS=$((SECURITY_PASS + 1))
fi

# Check for critical TODOs in code
print_info "Scanning for critical TODOs..."
CRITICAL_TODOS=$(grep -r "FIXME\|TODO.*CRITICAL" "$PROJ_ROOT/src" "$PROJ_ROOT/crates" 2>/dev/null | wc -l)
if [ "$CRITICAL_TODOS" -gt 0 ]; then
    print_warn "Found $CRITICAL_TODOS critical TODOs"
else
    print_pass "No critical TODOs found"
fi

# ─────────────────────────────────────────────────────────────────────────────
# 7. FINAL REPORT
# ─────────────────────────────────────────────────────────────────────────────

print_header "SHIP READINESS REPORT"

echo "Test Results:"
echo "  Rust Tests:        $([ $RUST_PASS -eq 1 ] && echo -e "${GREEN}PASS${NC}" || echo -e "${RED}FAIL${NC}")"
echo "  C Tests:           $([ $C_PASS -eq 1 ] && echo -e "${GREEN}PASS${NC}" || echo -e "${RED}FAIL${NC}")"
echo "  Python Tests:      $([ $PY_FAIL -eq 0 ] && echo -e "${GREEN}PASS${NC}" || echo -e "${RED}$PY_FAIL FAILED${NC}")"

echo ""
echo "Build Status:"
echo "  Dylibs Present:    $([ $BUILD_FAIL -eq 0 ] && echo -e "${GREEN}${BUILD_PASS}/3${NC}" || echo -e "${RED}${BUILD_PASS}/3${NC}")"
echo "  Model Files:       $([ $MODEL_FAIL -eq 0 ] && echo -e "${GREEN}${MODEL_PASS}/2${NC}" || echo -e "${RED}${MODEL_PASS}/2${NC}")"

echo ""
echo "Security & Quality:"
echo "  P0 Issues:         $([ $SECURITY_PASS -eq 1 ] && echo -e "${GREEN}NONE${NC}" || echo -e "${RED}FOUND${NC}")"
echo "  Critical TODOs:    $([ $CRITICAL_TODOS -eq 0 ] && echo -e "${GREEN}NONE${NC}" || echo -e "${YELLOW}$CRITICAL_TODOS${NC}")"

echo ""
echo "Summary:"
TOTAL_FAIL=$((RUST_FAIL + C_FAIL + PY_FAIL + BUILD_FAIL + MODEL_FAIL + SECURITY_FAIL))
if [ $TOTAL_FAIL -eq 0 ]; then
    echo -e "${GREEN}═══════════════════════════════════════════════════════════${NC}"
    echo -e "${GREEN}                      🚀 GO FOR SHIP 🚀${NC}"
    echo -e "${GREEN}═══════════════════════════════════════════════════════════${NC}"
    exit 0
else
    echo -e "${RED}═══════════════════════════════════════════════════════════${NC}"
    echo -e "${RED}                     ⚠️  NO-GO: FIX REQUIRED${NC}"
    echo -e "${RED}═══════════════════════════════════════════════════════════${NC}"
    echo ""
    echo "Failed items:"
    for item in "${FAILED_TESTS[@]}"; do
        echo -e "  ${RED}✗${NC} $item"
    done
    exit 1
fi
