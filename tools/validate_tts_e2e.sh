#!/bin/bash
###############################################################################
# TTS E2E Validation Wrapper — downloads checkpoints and runs validation
#
# Usage:
#   # Quick test (3 sentences)
#   ./tools/validate_tts_e2e.sh --mode quick
#
#   # Full validation (all 20 sentences)
#   ./tools/validate_tts_e2e.sh --mode full
#
#   # Compare models with custom setup
#   ./tools/validate_tts_e2e.sh --mode compare \
#     --flow-full checkpoints/my_flow.pt \
#     --vocoder checkpoints/my_vocoder.pt
#
# Features:
#   - Auto-detects missing checkpoints
#   - Downloads from GCS if needed
#   - Runs Python validation
#   - Reports summary
###############################################################################

set -euo pipefail

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
PYTHON_SCRIPT="$SCRIPT_DIR/validate_tts_e2e.py"

# Defaults
MODE="compare"
FLOW_FULL="$PROJECT_ROOT/checkpoints/flow_v3_final.pt"
FLOW_DISTILLED="$PROJECT_ROOT/models/sonata_flow_distilled"
VOCODER="$PROJECT_ROOT/checkpoints/vocoder_v3_latest.pt"
OUTPUT_DIR="$PROJECT_ROOT/eval_output"
OUTPUT_JSON="$PROJECT_ROOT/eval_tts_e2e.json"
DEVICE="mps"
STEPS_FULL=8
STEPS_DISTILLED=1
SENTENCES=""

# GCS paths
GCS_BUCKET="gs://sonata-training-johnb-2025"
GCS_VOCODER="$GCS_BUCKET/checkpoints/vocoder/vocoder_v3_snake_fix_latest.pt"
GCS_FLOW_FULL="$GCS_BUCKET/checkpoints/flow/flow_v3_final.pt"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --mode)
            MODE="$2"
            shift 2
            ;;
        --flow-full)
            FLOW_FULL="$2"
            shift 2
            ;;
        --flow-distilled)
            FLOW_DISTILLED="$2"
            shift 2
            ;;
        --vocoder)
            VOCODER="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --output)
            OUTPUT_JSON="$2"
            shift 2
            ;;
        --sentences)
            SENTENCES="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --steps-full)
            STEPS_FULL="$2"
            shift 2
            ;;
        --steps-distilled)
            STEPS_DISTILLED="$2"
            shift 2
            ;;
        --help|-h)
            sed -n '2,16p' "$0" | sed 's/^# *//'
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# ─── Utility Functions ─────────────────────────────────────────────────────

log_info() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] INFO: $*"
}

log_warn() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] WARN: $*" >&2
}

log_error() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: $*" >&2
}

check_file_or_gcs() {
    local path=$1
    local gcs_path=$2
    local name=$3

    if [[ -f "$path" ]]; then
        log_info "Found $name: $path"
        return 0
    fi

    log_warn "$name not found: $path"

    # Try GCS download
    if command -v gsutil &> /dev/null; then
        log_info "Attempting to download from GCS..."
        if gsutil cp "$gcs_path" "$path" 2>/dev/null; then
            log_info "Downloaded $name from GCS"
            return 0
        fi
        log_warn "Failed to download from GCS: $gcs_path"
    else
        log_warn "gsutil not installed (cannot download from GCS)"
    fi

    return 1
}

check_python_deps() {
    local python_exe="${PYTHON:-python3}"

    # Check torch
    if ! $python_exe -c "import torch" 2>/dev/null; then
        log_error "torch not installed. Please run: pip install torch"
        return 1
    fi

    # Check soundfile
    if ! $python_exe -c "import soundfile" 2>/dev/null; then
        log_warn "soundfile not installed. Run: pip install soundfile"
    fi

    # Optional (for metrics)
    $python_exe -c "import pesq" 2>/dev/null || log_warn "pesq not installed (metrics skipped)"
    $python_exe -c "import pystoi" 2>/dev/null || log_warn "pystoi not installed (metrics skipped)"
    $python_exe -c "import librosa" 2>/dev/null || log_warn "librosa not installed (metrics skipped)"

    return 0
}

# ─── Main Logic ────────────────────────────────────────────────────────────

main() {
    log_info "TTS E2E Validation"
    log_info "Mode: $MODE"
    log_info ""

    # Check Python deps
    if ! check_python_deps; then
        log_error "Missing required Python dependencies"
        return 1
    fi

    # Check checkpoints
    log_info "Checking checkpoints..."

    # Vocoder is always required
    if ! check_file_or_gcs "$VOCODER" "$GCS_VOCODER" "Vocoder"; then
        log_error "Vocoder checkpoint required and not available"
        log_error "Please download: gsutil cp $GCS_VOCODER $VOCODER"
        return 1
    fi

    # Flow full required for compare/full modes
    if [[ "$MODE" != "quick" ]]; then
        if ! check_file_or_gcs "$FLOW_FULL" "$GCS_FLOW_FULL" "Flow (full)"; then
            log_warn "Flow (full) not available; will skip full model comparison"
            FLOW_FULL=""
        fi
    fi

    # Distilled model (should always exist)
    if [[ ! -d "$FLOW_DISTILLED" ]]; then
        log_error "Distilled Flow directory not found: $FLOW_DISTILLED"
        return 1
    fi

    log_info ""
    log_info "Configuration:"
    log_info "  Mode: $MODE"
    log_info "  Output dir: $OUTPUT_DIR"
    log_info "  Output JSON: $OUTPUT_JSON"
    log_info "  Device: $DEVICE"
    if [[ -n "$FLOW_FULL" ]]; then
        log_info "  Flow (full): $FLOW_FULL"
        log_info "  Flow (full) steps: $STEPS_FULL"
    fi
    log_info "  Flow (distilled): $FLOW_DISTILLED"
    log_info "  Flow (distilled) steps: $STEPS_DISTILLED"
    log_info "  Vocoder: $VOCODER"
    [[ -n "$SENTENCES" ]] && log_info "  Custom sentences: $SENTENCES"
    log_info ""

    # Create output directory
    mkdir -p "$OUTPUT_DIR"

    # Build Python arguments
    local py_args=(
        "--mode" "$MODE"
        "--flow-distilled" "$FLOW_DISTILLED"
        "--vocoder" "$VOCODER"
        "--output-dir" "$OUTPUT_DIR"
        "--output" "$OUTPUT_JSON"
        "--device" "$DEVICE"
        "--steps-full" "$STEPS_FULL"
        "--steps-distilled" "$STEPS_DISTILLED"
    )

    if [[ -n "$FLOW_FULL" ]]; then
        py_args+=("--flow-full" "$FLOW_FULL")
    fi

    if [[ -n "$SENTENCES" ]]; then
        py_args+=("--sentences" "$SENTENCES")
    fi

    # Run validation
    log_info "Starting validation..."
    log_info ""

    if python3 "$PYTHON_SCRIPT" "${py_args[@]}"; then
        log_info ""
        log_info "Validation completed successfully"
        log_info "Results: $OUTPUT_JSON"
        return 0
    else
        log_error "Validation failed"
        return 1
    fi
}

# Run
main
