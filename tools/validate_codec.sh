#!/bin/bash
##############################################################################
# Codec Validation Wrapper
#
# Downloads codec checkpoint from GCS and runs validation.
#
# Usage:
#   ./tools/validate_codec.sh [OPTIONS]
#
# Options:
#   --checkpoint CKPT      Local checkpoint path (default: download from GCS)
#   --step N               GCS step number (default: 135000)
#   --quick                Quick mode: 5 samples
#   --full                 Full mode: all samples
#   --output-dir DIR       Output directory (default: ./codec_eval_results)
#   --device DEVICE        Device: cpu, cuda, mps (default: cpu)
#
# Examples:
#   # Quick eval of step 135000 (downloads from GCS)
#   ./tools/validate_codec.sh --quick
#
#   # Full eval of local checkpoint
#   ./tools/validate_codec.sh --checkpoint train/checkpoints/codec/sonata_codec_12hz_final.pt --full
#
##############################################################################

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Defaults
CHECKPOINT=""
GCS_STEP=135000
QUICK=false
FULL=false
OUTPUT_DIR="./codec_eval_results"
DEVICE="cpu"
GCS_BUCKET="gs://sonata-training-johnb-2025"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --checkpoint)
            CHECKPOINT="$2"
            shift 2
            ;;
        --step)
            GCS_STEP="$2"
            shift 2
            ;;
        --quick)
            QUICK=true
            shift
            ;;
        --full)
            FULL=true
            shift
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --help|-h)
            grep "^#" "$0" | head -30
            exit 0
            ;;
        *)
            echo "[ERROR] Unknown option: $1" >&2
            exit 1
            ;;
    esac
done

# If no checkpoint specified, download from GCS
if [[ -z "$CHECKPOINT" ]]; then
    echo "[setup] No local checkpoint specified, checking GCS..."

    # Check if gcloud is available
    if ! command -v gsutil &> /dev/null; then
        echo "[ERROR] gsutil not found. Install Google Cloud SDK or provide --checkpoint" >&2
        exit 1
    fi

    # Download from GCS
    GCS_PATH="${GCS_BUCKET}/checkpoints/codec_12hz/sonata_codec_step_${GCS_STEP}.pt"
    LOCAL_CKPT="train/checkpoints/codec/sonata_codec_step_${GCS_STEP}.pt"

    echo "[download] Attempting to download from GCS: $GCS_PATH"
    mkdir -p "$(dirname "$LOCAL_CKPT")"

    if gsutil -m cp "$GCS_PATH" "$LOCAL_CKPT" 2>/dev/null; then
        echo "[download] Success: $LOCAL_CKPT"
        CHECKPOINT="$LOCAL_CKPT"
    else
        echo "[ERROR] Failed to download from GCS. Make sure:" >&2
        echo "  - gcloud is authenticated: gcloud auth application-default login" >&2
        echo "  - Checkpoint exists at: $GCS_PATH" >&2
        echo "  - Or provide local path with --checkpoint" >&2
        exit 1
    fi
fi

# Verify checkpoint exists
if [[ ! -f "$CHECKPOINT" ]]; then
    echo "[ERROR] Checkpoint not found: $CHECKPOINT" >&2
    exit 1
fi

echo "[setup] Using checkpoint: $(cd "$(dirname "$CHECKPOINT")" && pwd)/$(basename "$CHECKPOINT")"
echo "[setup] Output directory: $OUTPUT_DIR"

# Build Python command
PYTHON_ARGS=(
    "--checkpoint" "$CHECKPOINT"
    "--device" "$DEVICE"
    "--output-dir" "$OUTPUT_DIR"
)

if [[ "$QUICK" == "true" ]]; then
    PYTHON_ARGS+=("--quick")
    echo "[mode] Quick validation (5 samples)"
elif [[ "$FULL" == "true" ]]; then
    PYTHON_ARGS+=("--full")
    echo "[mode] Full validation (all samples)"
else
    echo "[mode] Standard validation (20 samples)"
fi

# Run validation
cd "$PROJECT_ROOT"
echo ""
python3 tools/validate_codec.py "${PYTHON_ARGS[@]}"
RESULT=$?

# Summary
echo ""
if [[ $RESULT -eq 0 ]]; then
    echo "✓ Validation completed successfully"
    echo "  Results: $OUTPUT_DIR/results.json"
else
    echo "✗ Validation completed with errors"
fi

exit $RESULT
