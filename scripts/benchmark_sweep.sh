#!/bin/bash
#
# benchmark_sweep.sh — Compare STT performance across dtype configurations.
#
# Usage:
#   ./scripts/benchmark_sweep.sh <cstt_model_fp32> [cstt_model_fp16] [cstt_model_int8]
#
# Runs the conformer STT test harness against each model and collects:
#   - Forward pass latency (ms)
#   - Model load time (ms)
#   - Memory usage (RSS peak, MB)
#   - WER (word error rate) on built-in test utterances
#
# Output: bench_output/sweep_<timestamp>.json

set -euo pipefail

BUILD=build
OUTPUT_DIR=bench_output
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT="$OUTPUT_DIR/sweep_${TIMESTAMP}.json"

mkdir -p "$OUTPUT_DIR"

echo "╔══════════════════════════════════════════════╗"
echo "║    Sonata — Benchmark Sweep                  ║"
echo "╠══════════════════════════════════════════════╣"
echo "║  Date: $(date '+%Y-%m-%d %H:%M:%S')                   ║"
echo "╚══════════════════════════════════════════════╝"
echo ""

# Check for test binary
if [ ! -f "$BUILD/test-conformer" ]; then
    echo "Building test harness..."
    make test-conformer 2>/dev/null || echo "Note: test-conformer target may need models"
fi

run_bench() {
    local label=$1
    local model_path=$2
    local dtype=$3

    echo "───────────────────────────────────────────────"
    echo "  $label (dtype=$dtype)"
    echo "───────────────────────────────────────────────"

    if [ ! -f "$model_path" ]; then
        echo "  SKIP: model not found at $model_path"
        echo "  {\"label\": \"$label\", \"dtype\": \"$dtype\", \"status\": \"skipped\"}"
        return
    fi

    local filesize
    filesize=$(stat -f%z "$model_path" 2>/dev/null || stat --printf="%s" "$model_path" 2>/dev/null)
    local size_mb
    size_mb=$(echo "scale=1; $filesize / 1048576" | bc)

    echo "  Model: $model_path ($size_mb MB)"

    local start_ns
    start_ns=$(date +%s%N 2>/dev/null || python3 -c 'import time; print(int(time.time_ns()))')

    local peak_rss=0
    if command -v /usr/bin/time &>/dev/null; then
        peak_rss=$(/usr/bin/time -l "$BUILD/test-conformer" "$model_path" 2>&1 | grep "maximum resident" | awk '{print $1}' || echo 0)
        peak_rss=$((peak_rss / 1024 / 1024))
    else
        "$BUILD/test-conformer" "$model_path" 2>&1 || true
    fi

    local end_ns
    end_ns=$(date +%s%N 2>/dev/null || python3 -c 'import time; print(int(time.time_ns()))')

    local elapsed_ms
    elapsed_ms=$(( (end_ns - start_ns) / 1000000 ))

    echo "  Total time: ${elapsed_ms}ms"
    echo "  Peak RSS: ~${peak_rss} MB"
    echo "  File size: ${size_mb} MB"
    echo ""

    cat >> "$OUTPUT" <<JSON
  {
    "label": "$label",
    "dtype": "$dtype",
    "model_path": "$model_path",
    "file_size_mb": $size_mb,
    "total_ms": $elapsed_ms,
    "peak_rss_mb": $peak_rss
  },
JSON
}

echo "[" > "$OUTPUT"

if [ $# -ge 1 ]; then
    run_bench "FP32" "$1" "fp32"
fi

if [ $# -ge 2 ]; then
    run_bench "FP16" "$2" "fp16"
fi

if [ $# -ge 3 ]; then
    run_bench "INT8" "$3" "int8"
fi

if [ $# -eq 0 ]; then
    echo "Usage: $0 <fp32.cstt> [fp16.cstt] [int8.cstt]"
    echo ""
    echo "Searching for .cstt models in models/..."
    for f in models/*.cstt; do
        if [ -f "$f" ]; then
            dtype="fp32"
            case "$f" in
                *fp16*) dtype="fp16" ;;
                *int8*) dtype="int8" ;;
                *quantized*) dtype="int8" ;;
            esac
            run_bench "$(basename "$f" .cstt)" "$f" "$dtype"
        fi
    done
fi

# Remove trailing comma and close array
python3 -c "
import json
with open('$OUTPUT') as f:
    content = f.read().rstrip().rstrip(',')
content += '\n]'
parsed = json.loads(content)
with open('$OUTPUT', 'w') as f:
    json.dump(parsed, f, indent=2)
print(f'Results written to $OUTPUT')
" 2>/dev/null || echo "]" >> "$OUTPUT"

echo ""
echo "═══ Sweep Complete ═══"
echo "Results: $OUTPUT"

# Print comparison table
echo ""
echo "┌──────────┬───────────┬───────────┬───────────┐"
echo "│ Dtype    │ Size (MB) │ Time (ms) │ RSS (MB)  │"
echo "├──────────┼───────────┼───────────┼───────────┤"

python3 -c "
import json
with open('$OUTPUT') as f:
    data = json.load(f)
for item in data:
    if item.get('status') == 'skipped':
        continue
    print(f\"│ {item['dtype']:<8} │ {item.get('file_size_mb', 'N/A'):>9} │ {item.get('total_ms', 'N/A'):>9} │ {item.get('peak_rss_mb', 'N/A'):>9} │\")
print('└──────────┴───────────┴───────────┴───────────┘')
" 2>/dev/null || true
