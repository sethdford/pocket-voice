#!/bin/bash
# Unified TTS Evaluation Pipeline
#
# Runs all evaluation tools and produces a single markdown report.
#
# Usage:
#   ./scripts/eval_all.sh --checkpoint PATH --vocoder PATH [--output report.md]
#   ./scripts/eval_all.sh --generated-dir gen/ --ref-dir ref/ [--output report.md]
#
# Steps:
#   1. Generate test audio (if checkpoint + vocoder provided)
#   2. Run eval_tts.py --mode batch (or synthesize) on generated audio
#   3. Run eval_prosody_ab.py --synthetic
#   4. Collect all results into a markdown report

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
TRAIN_DIR="$REPO_ROOT/train/sonata"
WORK_DIR="${WORK_DIR:-$REPO_ROOT/eval_output}"
REPORT_PATH=""
FLOW_CHECKPOINT=""
VOCODER_CHECKPOINT=""
GENERATED_DIR=""
REF_DIR=""
REF_TEXTS=""
UTMOS_FLAG=""
DEVICE="${DEVICE:-mps}"

usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --checkpoint PATH     Flow v3 checkpoint (triggers synthesize mode)"
    echo "  --vocoder PATH        Vocoder checkpoint (required with --checkpoint)"
    echo "  --generated-dir PATH  Directory of generated WAVs (batch mode)"
    echo "  --ref-dir PATH        Reference WAV directory"
    echo "  --ref-texts PATH      Reference texts file, one per line"
    echo "  --output PATH         Output markdown report (default: eval_output/eval_report.md)"
    echo "  --utmos               Enable UTMOS metric (requires speechmos or transformers)"
    echo "  --device DEVICE       Device: mps, cuda, cpu (default: mps)"
    echo ""
    echo "Examples:"
    echo "  $0 --checkpoint ckpt.pt --vocoder voc.pt --output report.md"
    echo "  $0 --generated-dir gen/ --ref-dir ref/ --ref-texts texts.txt"
    exit 1
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --checkpoint)    FLOW_CHECKPOINT="$2"; shift 2 ;;
        --vocoder)       VOCODER_CHECKPOINT="$2"; shift 2 ;;
        --generated-dir) GENERATED_DIR="$2"; shift 2 ;;
        --ref-dir)       REF_DIR="$2"; shift 2 ;;
        --ref-texts)     REF_TEXTS="$2"; shift 2 ;;
        --output)        REPORT_PATH="$2"; shift 2 ;;
        --utmos)         UTMOS_FLAG="--utmos"; shift ;;
        --device)        DEVICE="$2"; shift 2 ;;
        -h|--help)       usage ;;
        *)               echo "Unknown option: $1"; usage ;;
    esac
done

if [[ -z "$REPORT_PATH" ]]; then
    REPORT_PATH="$WORK_DIR/eval_report.md"
fi

mkdir -p "$(dirname "$REPORT_PATH")"
mkdir -p "$WORK_DIR"

EVAL_JSON="$WORK_DIR/eval_tts_report.json"
PROSODY_JSON="$WORK_DIR/prosody_ab_report.json"

# -----------------------------------------------------------------------------
# Step 1 & 2: Generate and/or evaluate with eval_tts.py
# -----------------------------------------------------------------------------
if [[ -n "$FLOW_CHECKPOINT" && -n "$VOCODER_CHECKPOINT" ]]; then
    echo "[1/3] Generating test audio and evaluating (synthesize mode)..."
    cd "$TRAIN_DIR"
    EXTRA_ARGS=()
    [[ -n "$REF_DIR" ]] && EXTRA_ARGS+=(--ref-dir "$REF_DIR")
    python eval_tts.py --mode synthesize \
        --flow-checkpoint "$FLOW_CHECKPOINT" \
        --vocoder-checkpoint "$VOCODER_CHECKPOINT" \
        --output-dir "$WORK_DIR/generated" \
        --output "$EVAL_JSON" \
        --device "$DEVICE" \
        $UTMOS_FLAG \
        "${EXTRA_ARGS[@]}"
    GENERATED_DIR="$WORK_DIR/generated"
elif [[ -n "$GENERATED_DIR" && -n "$REF_DIR" ]]; then
    echo "[1/3] Evaluating generated audio (batch mode)..."
    cd "$TRAIN_DIR"
    EXTRA_ARGS=()
    [[ -n "$REF_TEXTS" ]] && EXTRA_ARGS+=(--ref-texts "$REF_TEXTS")
    python eval_tts.py --mode batch \
        --generated-dir "$GENERATED_DIR" \
        --ref-dir "$REF_DIR" \
        --output "$EVAL_JSON" \
        $UTMOS_FLAG \
        "${EXTRA_ARGS[@]}"
else
    echo "Either provide --checkpoint + --vocoder, or --generated-dir + --ref-dir"
    exit 1
fi

# -----------------------------------------------------------------------------
# Step 3: Prosody A/B evaluation (synthetic)
# -----------------------------------------------------------------------------
echo "[2/3] Running prosody A/B evaluation (synthetic)..."
cd "$TRAIN_DIR"
python eval_prosody_ab.py --synthetic --output "$PROSODY_JSON" 2>/dev/null || true

# -----------------------------------------------------------------------------
# Step 4: Build markdown report
# -----------------------------------------------------------------------------
echo "[3/3] Building markdown report..."

format_json_num() {
    python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('$1', 'N/A'))" 2>/dev/null || echo "N/A"
}

{
    echo "# TTS Evaluation Report"
    echo ""
    echo "**Generated:** $(date -u '+%Y-%m-%d %H:%M UTC')"
    echo ""

    # Model info
    echo "## Model Info"
    echo ""
    if [[ -n "$FLOW_CHECKPOINT" ]]; then
        echo "| Parameter | Value |"
        echo "|-----------|-------|"
        echo "| Flow checkpoint | \`$FLOW_CHECKPOINT\` |"
        echo "| Vocoder checkpoint | \`$VOCODER_CHECKPOINT\` |"
        echo "| Generated audio | \`$GENERATED_DIR\` |"
    else
        echo "| Parameter | Value |"
        echo "|-----------|-------|"
        echo "| Generated audio | \`$GENERATED_DIR\` |"
        echo "| Reference audio | \`$REF_DIR\` |"
    fi
    echo ""

    # Audio quality (from eval_tts)
    if [[ -f "$EVAL_JSON" ]]; then
        echo "## Audio Quality Metrics"
        echo ""
        echo "| Metric | Value | Target |"
        echo "|--------|-------|--------|"

        for key in mean_pesq mean_stoi mean_mcd_db mean_f0_corr mean_wer_pct mean_rtf mean_speaker_sim mean_mos_proxy mean_utmos; do
            val=$(python3 -c "
import json
try:
    with open('$EVAL_JSON') as f:
        d = json.load(f)
    v = d.get('$key', -1)
    print(f'{v:.3f}' if isinstance(v, (int, float)) and v >= 0 else 'N/A')
except Exception:
    print('N/A')
" 2>/dev/null)
            case "$key" in
                mean_pesq)          name="PESQ"; target="3.5" ;;
                mean_stoi)          name="STOI"; target="0.90" ;;
                mean_mcd_db)        name="MCD (dB)"; target="<4.0" ;;
                mean_f0_corr)       name="F0 correlation"; target="0.85" ;;
                mean_wer_pct)       name="WER (%)"; target="<5.0" ;;
                mean_rtf)           name="RTF"; target="<0.2" ;;
                mean_speaker_sim)   name="Speaker similarity"; target="0.85" ;;
                mean_mos_proxy)     name="MOS proxy"; target="4.0" ;;
                mean_utmos)         name="UTMOS"; target="4.0" ;;
                *)                  name="$key"; target="-"
            esac
            [[ "$val" != "N/A" ]] && echo "| $name | $val | $target |"
        done

        grade=$(python3 -c "
import json
try:
    with open('$EVAL_JSON') as f:
        d = json.load(f)
    print(d.get('grade', 'N/A'))
except Exception:
    print('N/A')
" 2>/dev/null)
        grade_score=$(python3 -c "
import json
try:
    with open('$EVAL_JSON') as f:
        d = json.load(f)
    print(f\"{d.get('grade_score', 0):.2f}\")
except Exception:
    print('N/A')
" 2>/dev/null)
        echo ""
        echo "**Grade: $grade** (score: $grade_score)"
        echo ""
    fi

    # Prosody metrics
    if [[ -f "$PROSODY_JSON" ]]; then
        echo "## Prosody Metrics (Synthetic A/B)"
        echo ""
        avg_base=$(python3 -c "
import json
try:
    with open('$PROSODY_JSON') as f:
        d = json.load(f)
    print(f\"{d.get('avg_baseline_mos', 0):.2f}\")
except Exception:
    print('N/A')
" 2>/dev/null)
        avg_enh=$(python3 -c "
import json
try:
    with open('$PROSODY_JSON') as f:
        d = json.load(f)
    print(f\"{d.get('avg_enhanced_mos', 0):.2f}\")
except Exception:
    print('N/A')
" 2>/dev/null)
        echo "| Metric | Value |"
        echo "|--------|-------|"
        echo "| Baseline prosody MOS | $avg_base |"
        echo "| Enhanced prosody MOS | $avg_enh |"
        echo ""
    fi

    # Summary table
    echo "## Summary"
    echo ""
    if [[ -f "$EVAL_JSON" ]]; then
        n_samples=$(python3 -c "
import json
try:
    with open('$EVAL_JSON') as f:
        d = json.load(f)
    print(d.get('n_samples', 0))
except Exception:
    print(0)
" 2>/dev/null)
        echo "- **Samples evaluated:** $n_samples"
        echo "- **Overall grade:** $grade"
        echo "- **Report JSON:** \`$EVAL_JSON\`"
    fi
    if [[ -f "$PROSODY_JSON" ]]; then
        echo "- **Prosody report:** \`$PROSODY_JSON\`"
    fi
    echo ""

} > "$REPORT_PATH"

echo ""
echo "Report saved to: $REPORT_PATH"
echo ""
