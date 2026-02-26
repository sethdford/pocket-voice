#!/usr/bin/env bash
# bench_pocket_voice.sh — Benchmark pocket-voice STT + TTS engines.
#
# Runs both C Conformer STT and C Kyutai DSM TTS benchmarks,
# plus memory profiling and cold/warm start timing.
#
# Usage:
#   ./benchmarks/bench_pocket_voice.sh
#   ./benchmarks/bench_pocket_voice.sh --max-samples 50

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
RESULTS_DIR="$PROJECT_DIR/bench_output/results"

MAX_SAMPLES=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --max-samples) MAX_SAMPLES="$2"; shift 2 ;;
        -h|--help)     echo "Usage: $0 [--max-samples N]"; exit 0 ;;
        *)             echo "Unknown: $1"; exit 1 ;;
    esac
done

mkdir -p "$RESULTS_DIR"

echo "════════════════════════════════════════════════════════════"
echo "  pocket-voice Benchmark"
echo "════════════════════════════════════════════════════════════"

# ── STT ──────────────────────────────────────────────────────────────────────

STT_MODEL="$PROJECT_DIR/models/parakeet-ctc-0.6b.cstt"
STT_LIB="$PROJECT_DIR/build/libconformer_stt.dylib"

if [ -f "$STT_LIB" ] && [ -f "$STT_MODEL" ]; then
    echo ""
    echo "▶ STT: C Conformer (parakeet-ctc-0.6b)"
    echo ""

    SAMPLE_ARGS=""
    if [ -n "$MAX_SAMPLES" ]; then
        SAMPLE_ARGS="--max-samples $MAX_SAMPLES"
    fi

    # Measure cold start (first load)
    echo "  Measuring cold start..."
    COLD_START=$(python3 -c "
import ctypes, time
t0 = time.monotonic()
lib = ctypes.CDLL('$STT_LIB')
lib.conformer_stt_create.restype = ctypes.c_void_p
lib.conformer_stt_create.argtypes = [ctypes.c_char_p]
lib.conformer_stt_destroy.argtypes = [ctypes.c_void_p]
engine = lib.conformer_stt_create(b'$STT_MODEL')
elapsed = time.monotonic() - t0
if engine: lib.conformer_stt_destroy(engine)
print(f'{elapsed:.3f}')
" 2>/dev/null || echo "0.000")
    echo "  Cold start: ${COLD_START}s"

    # Run full benchmark
    cd "$PROJECT_DIR"
    python3 scripts/bench_stt.py \
        --model "$STT_MODEL" \
        $SAMPLE_ARGS \
        --output "$RESULTS_DIR/stt_pocket_voice.json"

    # Model size
    STT_MODEL_MB=$(du -m "$STT_MODEL" | cut -f1)
    echo "  Model size: ${STT_MODEL_MB} MB"

    # Append cold_start and model_size to results
    python3 -c "
import json
with open('$RESULTS_DIR/stt_pocket_voice.json') as f:
    data = json.load(f)
data['cold_start_seconds'] = $COLD_START
data['model_size_mb'] = $STT_MODEL_MB
with open('$RESULTS_DIR/stt_pocket_voice.json', 'w') as f:
    json.dump(data, f, indent=2)
"
else
    echo ""
    echo "  SKIP STT: library or model not found"
    echo "    Need: $STT_LIB"
    echo "    Need: $STT_MODEL"
fi

# ── TTS ──────────────────────────────────────────────────────────────────────

TTS_MODEL="$PROJECT_DIR/models/kyutai_dsm.ctts"
TTS_LIB="$PROJECT_DIR/build/libkyutai_dsm_tts.dylib"

if [ -f "$TTS_LIB" ] && [ -f "$TTS_MODEL" ]; then
    echo ""
    echo "▶ TTS: C Kyutai DSM (1.6B)"
    echo ""

    # Run full benchmark
    cd "$PROJECT_DIR"
    python3 scripts/bench_tts.py \
        --model "$TTS_MODEL" \
        --output "$RESULTS_DIR/tts_pocket_voice.json"

    # Model size
    TTS_MODEL_MB=$(du -m "$TTS_MODEL" | cut -f1)
    echo "  Model size: ${TTS_MODEL_MB} MB"

    python3 -c "
import json
with open('$RESULTS_DIR/tts_pocket_voice.json') as f:
    data = json.load(f)
data['model_size_mb'] = $TTS_MODEL_MB
with open('$RESULTS_DIR/tts_pocket_voice.json', 'w') as f:
    json.dump(data, f, indent=2)
"
else
    echo ""
    echo "  SKIP TTS: library or model not found"
    echo "    Need: $TTS_LIB"
    echo "    Need: $TTS_MODEL"
fi

echo ""
echo "════════════════════════════════════════════════════════════"
echo "  pocket-voice benchmark complete"
echo "  Results: $RESULTS_DIR/"
echo "════════════════════════════════════════════════════════════"
