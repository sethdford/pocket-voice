#!/usr/bin/env bash
# bench_whisper_cpp.sh — Benchmark whisper.cpp against LibriSpeech test-clean.
#
# Requires whisper.cpp built with Metal support.
# Downloads models if not present.
#
# Usage:
#   ./benchmarks/bench_whisper_cpp.sh --output results/stt_whisper_cpp.json
#   ./benchmarks/bench_whisper_cpp.sh --model-size large-v3 --max-samples 50

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# Defaults
OUTPUT=""
MODEL_SIZE="base"
MAX_SAMPLES=""
WHISPER_CPP_DIR=""

# Common install locations
WHISPER_SEARCH_PATHS=(
    "$HOME/whisper.cpp"
    "$HOME/src/whisper.cpp"
    "/opt/whisper.cpp"
    "/usr/local/whisper.cpp"
)

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --output)       OUTPUT="$2"; shift 2 ;;
        --model-size)   MODEL_SIZE="$2"; shift 2 ;;
        --max-samples)  MAX_SAMPLES="$2"; shift 2 ;;
        --whisper-dir)  WHISPER_CPP_DIR="$2"; shift 2 ;;
        -h|--help)
            echo "Usage: $0 [--output FILE] [--model-size SIZE] [--max-samples N] [--whisper-dir DIR]"
            echo "  Sizes: tiny, base, small, medium, large-v2, large-v3"
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# Find whisper.cpp
if [ -z "$WHISPER_CPP_DIR" ]; then
    for path in "${WHISPER_SEARCH_PATHS[@]}"; do
        if [ -x "$path/build/bin/whisper-cli" ] || [ -x "$path/main" ]; then
            WHISPER_CPP_DIR="$path"
            break
        fi
    done
fi

if [ -z "$WHISPER_CPP_DIR" ]; then
    echo "ERROR: whisper.cpp not found. Install it or pass --whisper-dir."
    echo ""
    echo "Quick install:"
    echo "  git clone https://github.com/ggerganov/whisper.cpp ~/whisper.cpp"
    echo "  cd ~/whisper.cpp && cmake -B build -DWHISPER_METAL=ON && cmake --build build -j"
    echo "  bash models/download-ggml-model.sh base"
    exit 1
fi

# Find the whisper binary
WHISPER_BIN=""
if [ -x "$WHISPER_CPP_DIR/build/bin/whisper-cli" ]; then
    WHISPER_BIN="$WHISPER_CPP_DIR/build/bin/whisper-cli"
elif [ -x "$WHISPER_CPP_DIR/build/bin/main" ]; then
    WHISPER_BIN="$WHISPER_CPP_DIR/build/bin/main"
elif [ -x "$WHISPER_CPP_DIR/main" ]; then
    WHISPER_BIN="$WHISPER_CPP_DIR/main"
fi

if [ -z "$WHISPER_BIN" ]; then
    echo "ERROR: whisper.cpp binary not found in $WHISPER_CPP_DIR"
    exit 1
fi

# Find or download model
MODEL_FILE="$WHISPER_CPP_DIR/models/ggml-${MODEL_SIZE}.bin"
if [ ! -f "$MODEL_FILE" ]; then
    echo "Downloading whisper model: $MODEL_SIZE..."
    if [ -x "$WHISPER_CPP_DIR/models/download-ggml-model.sh" ]; then
        bash "$WHISPER_CPP_DIR/models/download-ggml-model.sh" "$MODEL_SIZE"
    else
        echo "ERROR: Model $MODEL_FILE not found and download script missing."
        exit 1
    fi
fi

echo "whisper.cpp benchmark"
echo "  Binary:  $WHISPER_BIN"
echo "  Model:   $MODEL_FILE"
echo "  Size:    $MODEL_SIZE"
echo ""

# Run the Python harness that drives whisper.cpp and collects metrics
python3 "$SCRIPT_DIR/bench_whisper_cpp_harness.py" \
    --whisper-bin "$WHISPER_BIN" \
    --model "$MODEL_FILE" \
    --model-size "$MODEL_SIZE" \
    ${MAX_SAMPLES:+--max-samples "$MAX_SAMPLES"} \
    ${OUTPUT:+--output "$OUTPUT"}
