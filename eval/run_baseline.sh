#!/usr/bin/env bash
# eval/run_baseline.sh -- Run full TTS evaluation pipeline
#
# Usage:
#   ./eval/run_baseline.sh              # C pipeline + Python metrics
#   ./eval/run_baseline.sh --utmos      # Include UTMOS (slow, downloads model)
#   ./eval/run_baseline.sh --ref-dir eval/reference  # With reference audio

set -euo pipefail
cd "$(dirname "$0")/.."

UTMOS_FLAG=""
REF_FLAG=""
for arg in "$@"; do
    case "$arg" in
        --utmos) UTMOS_FLAG="--utmos" ;;
        --ref-dir=*) REF_FLAG="--ref-dir ${arg#*=}" ;;
    esac
done

echo ""
echo "========================================================"
echo "  Sonata TTS Evaluation Pipeline"
echo "  $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "========================================================"

# Step 1: Generate WAVs via C pipeline
echo ""
echo "[1/3] Generating audio via C pipeline..."
mkdir -p eval/generated eval/reports

if [ ! -f build/eval-sonata-baseline ]; then
    echo "  Building eval harness..."
    make eval-generate 2>&1 | tail -5
else
    ./build/eval-sonata-baseline
fi

# Step 2: Run Python comprehensive eval
echo ""
echo "[2/3] Running Python comprehensive evaluation..."
python3 eval/run_eval.py \
    --wav-dir eval/generated \
    --c-report eval/reports/c_eval_report.json \
    $REF_FLAG $UTMOS_FLAG \
    --output eval/reports/eval_report.json

# Step 3: Archive report with timestamp
TIMESTAMP=$(date -u '+%Y%m%d_%H%M%S')
cp eval/reports/eval_report.json "eval/reports/eval_${TIMESTAMP}.json"

echo ""
echo "========================================================"
echo "  Evaluation complete!"
echo "  C report:      eval/reports/c_eval_report.json"
echo "  Full report:   eval/reports/eval_report.json"
echo "  Archived:      eval/reports/eval_${TIMESTAMP}.json"
echo "  WAV files:     eval/generated/eval_*.wav"
echo "========================================================"
