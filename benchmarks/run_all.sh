#!/usr/bin/env bash
# run_all.sh — Master benchmark runner for pocket-voice SOTA comparison.
#
# Runs all available benchmarks and produces a unified comparison report.
#
# Usage:
#   ./benchmarks/run_all.sh                    # Run all benchmarks
#   ./benchmarks/run_all.sh --stt-only         # STT benchmarks only
#   ./benchmarks/run_all.sh --tts-only         # TTS benchmarks only
#   ./benchmarks/run_all.sh --quick            # Quick mode (fewer samples)
#   ./benchmarks/run_all.sh --max-samples 20   # Limit LibriSpeech samples

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
OUTPUT_DIR="$PROJECT_DIR/bench_output"
RESULTS_DIR="$OUTPUT_DIR/results"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Defaults
RUN_STT=true
RUN_TTS=true
MAX_SAMPLES=""
QUICK_MODE=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --stt-only)   RUN_TTS=false; shift ;;
        --tts-only)   RUN_STT=false; shift ;;
        --quick)      QUICK_MODE=true; MAX_SAMPLES=10; shift ;;
        --max-samples) MAX_SAMPLES="$2"; shift 2 ;;
        -h|--help)
            echo "Usage: $0 [--stt-only] [--tts-only] [--quick] [--max-samples N]"
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

mkdir -p "$RESULTS_DIR"

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

log() { echo "$(date +%H:%M:%S) │ $*"; }
separator() { echo "════════════════════════════════════════════════════════════════"; }

check_command() {
    if command -v "$1" &>/dev/null; then
        return 0
    fi
    return 1
}

measure_peak_rss_kb() {
    # Run a command and capture peak RSS via /usr/bin/time on macOS
    local output_file="$1"; shift
    /usr/bin/time -l "$@" 2>"$output_file.time" || true
    grep "maximum resident set size" "$output_file.time" | awk '{print $1}' || echo "0"
}

# ──────────────────────────────────────────────────────────────────────────────
# System Info
# ──────────────────────────────────────────────────────────────────────────────

log "Collecting system info..."
CHIP=$(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo "Unknown")
CORES_P=$(sysctl -n hw.perflevel0.logicalcpu 2>/dev/null || echo "?")
CORES_E=$(sysctl -n hw.perflevel1.logicalcpu 2>/dev/null || echo "?")
RAM_GB=$(( $(sysctl -n hw.memsize 2>/dev/null || echo 0) / 1073741824 ))
OS_VER=$(sw_vers -productVersion 2>/dev/null || echo "Unknown")

SYSINFO="{\"chip\": \"$CHIP\", \"cores_p\": \"$CORES_P\", \"cores_e\": \"$CORES_E\", \"ram_gb\": $RAM_GB, \"os\": \"$OS_VER\", \"timestamp\": \"$TIMESTAMP\"}"
echo "$SYSINFO" > "$RESULTS_DIR/system_info.json"

separator
echo "  pocket-voice SOTA Benchmark Suite"
separator
echo "  Chip:    $CHIP"
echo "  Cores:   ${CORES_P}P + ${CORES_E}E"
echo "  RAM:     ${RAM_GB} GB"
echo "  macOS:   $OS_VER"
echo "  Time:    $TIMESTAMP"
if [ "$QUICK_MODE" = true ]; then
    echo "  Mode:    QUICK (${MAX_SAMPLES} samples)"
fi
separator
echo ""

# ──────────────────────────────────────────────────────────────────────────────
# STT Benchmarks
# ──────────────────────────────────────────────────────────────────────────────

if [ "$RUN_STT" = true ]; then
    log "Starting STT benchmarks..."
    echo ""

    SAMPLE_ARGS=""
    if [ -n "$MAX_SAMPLES" ]; then
        SAMPLE_ARGS="--max-samples $MAX_SAMPLES"
    fi

    # 1. pocket-voice C Conformer STT
    if [ -f "$PROJECT_DIR/build/libconformer_stt.dylib" ] && [ -f "$PROJECT_DIR/models/parakeet-ctc-0.6b.cstt" ]; then
        log "[STT] Running pocket-voice C Conformer..."
        cd "$PROJECT_DIR"
        python3 scripts/bench_stt.py \
            --model models/parakeet-ctc-0.6b.cstt \
            $SAMPLE_ARGS \
            --output "$RESULTS_DIR/stt_pocket_voice.json" || \
            log "[STT] pocket-voice benchmark failed (build may be broken)"
    else
        log "[STT] SKIP pocket-voice: build/libconformer_stt.dylib or model not found"
        log "      Run 'make libs' and download the model first."
    fi

    # 2. whisper.cpp
    if [ -x "$PROJECT_DIR/benchmarks/bench_whisper_cpp.sh" ]; then
        log "[STT] Running whisper.cpp benchmarks..."
        bash "$PROJECT_DIR/benchmarks/bench_whisper_cpp.sh" \
            --output "$RESULTS_DIR/stt_whisper_cpp.json" \
            $SAMPLE_ARGS || \
            log "[STT] whisper.cpp benchmark failed or not installed"
    else
        log "[STT] SKIP whisper.cpp: bench script not found"
    fi

    # 3. MLX Whisper
    if python3 -c "import mlx_whisper" 2>/dev/null; then
        log "[STT] Running MLX Whisper benchmark..."
        python3 "$PROJECT_DIR/benchmarks/bench_mlx_whisper.py" \
            --output "$RESULTS_DIR/stt_mlx_whisper.json" \
            $SAMPLE_ARGS || \
            log "[STT] MLX Whisper benchmark failed"
    else
        log "[STT] SKIP MLX Whisper: not installed (pip install mlx-whisper)"
    fi

    # 4. Sherpa-onnx
    if python3 -c "import sherpa_onnx" 2>/dev/null; then
        log "[STT] Running sherpa-onnx benchmark..."
        python3 "$PROJECT_DIR/benchmarks/bench_sherpa_onnx.py" \
            --output "$RESULTS_DIR/stt_sherpa_onnx.json" \
            $SAMPLE_ARGS || \
            log "[STT] sherpa-onnx benchmark failed"
    else
        log "[STT] SKIP sherpa-onnx: not installed (pip install sherpa-onnx)"
    fi

    echo ""
fi

# ──────────────────────────────────────────────────────────────────────────────
# TTS Benchmarks
# ──────────────────────────────────────────────────────────────────────────────

if [ "$RUN_TTS" = true ]; then
    log "Starting TTS benchmarks..."
    echo ""

    # 1. pocket-voice C Kyutai DSM TTS
    if [ -f "$PROJECT_DIR/build/libkyutai_dsm_tts.dylib" ] && [ -f "$PROJECT_DIR/models/kyutai_dsm.ctts" ]; then
        log "[TTS] Running pocket-voice C Kyutai DSM..."
        cd "$PROJECT_DIR"
        python3 scripts/bench_tts.py \
            --model models/kyutai_dsm.ctts \
            --output "$RESULTS_DIR/tts_pocket_voice.json" || \
            log "[TTS] pocket-voice TTS benchmark failed (build may be broken)"
    else
        log "[TTS] SKIP pocket-voice: build/libkyutai_dsm_tts.dylib or model not found"
        log "      Run 'make libs' and convert the model first."
    fi

    # 2. Piper TTS
    if command -v piper &>/dev/null; then
        log "[TTS] Running Piper TTS benchmark..."
        bash "$PROJECT_DIR/benchmarks/bench_piper.sh" \
            --output "$RESULTS_DIR/tts_piper.json" || \
            log "[TTS] Piper benchmark failed"
    else
        log "[TTS] SKIP Piper: not installed"
    fi

    # 3. MLX TTS (if available)
    if python3 -c "import mlx_audio" 2>/dev/null || python3 -c "import f5_tts_mlx" 2>/dev/null; then
        log "[TTS] Running MLX TTS benchmark..."
        python3 "$PROJECT_DIR/benchmarks/bench_mlx_tts.py" \
            --output "$RESULTS_DIR/tts_mlx.json" || \
            log "[TTS] MLX TTS benchmark failed"
    else
        log "[TTS] SKIP MLX TTS: not installed"
    fi

    echo ""
fi

# ──────────────────────────────────────────────────────────────────────────────
# Generate Comparison Report
# ──────────────────────────────────────────────────────────────────────────────

separator
log "Generating comparison report..."
separator

python3 "$PROJECT_DIR/benchmarks/compare.py" \
    --results-dir "$RESULTS_DIR" \
    --output "$OUTPUT_DIR/comparison_${TIMESTAMP}.md" \
    --json-output "$OUTPUT_DIR/comparison_${TIMESTAMP}.json"

echo ""
separator
echo "  Benchmark complete!"
echo "  Results: $RESULTS_DIR/"
echo "  Report:  $OUTPUT_DIR/comparison_${TIMESTAMP}.md"
separator
