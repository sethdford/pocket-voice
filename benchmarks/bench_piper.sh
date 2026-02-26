#!/usr/bin/env bash
# bench_piper.sh — Benchmark Piper TTS on standard test sentences.
#
# Requires: piper binary installed (https://github.com/rhasspy/piper)
#
# Usage:
#   ./benchmarks/bench_piper.sh --output results/tts_piper.json

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT=""
SAMPLE_RATE=22050

while [[ $# -gt 0 ]]; do
    case "$1" in
        --output)  OUTPUT="$2"; shift 2 ;;
        -h|--help) echo "Usage: $0 [--output FILE]"; exit 0 ;;
        *)         echo "Unknown: $1"; exit 1 ;;
    esac
done

if ! command -v piper &>/dev/null; then
    echo "ERROR: piper not found. Install from https://github.com/rhasspy/piper"
    exit 1
fi

# Find a piper voice model
VOICE_DIR="${PIPER_VOICE_DIR:-$HOME/.local/share/piper-voices}"
VOICE=""
if [ -d "$VOICE_DIR" ]; then
    VOICE=$(find "$VOICE_DIR" -name "*.onnx" -type f | head -1)
fi

if [ -z "$VOICE" ]; then
    echo "ERROR: No Piper voice model found."
    echo "Download one from: https://github.com/rhasspy/piper/blob/master/VOICES.md"
    echo "Set PIPER_VOICE_DIR to your voice directory."
    exit 1
fi

echo "Piper TTS Benchmark"
echo "  Voice: $VOICE"
echo ""

SENTENCES=(
    "Hello, how are you today?"
    "The quick brown fox jumps over the lazy dog."
    "Artificial intelligence is transforming the world."
    "Can you please tell me where the nearest hospital is?"
    "It was the best of times, it was the worst of times."
    "Technology continues to advance at an unprecedented pace."
    "The weather forecast predicts rain for the entire weekend."
    "She sold seashells by the seashore."
    "To be or not to be, that is the question."
    "A stitch in time saves nine."
)

TMPDIR=$(mktemp -d)
trap "rm -rf $TMPDIR" EXIT

TOTAL_AUDIO_SEC=0
TOTAL_PROC_SEC=0
TOTAL_VALID=0
JSON_RESULTS="["

for i in "${!SENTENCES[@]}"; do
    TEXT="${SENTENCES[$i]}"
    WAV="$TMPDIR/sample_${i}.wav"

    START=$(python3 -c "import time; print(time.monotonic())")
    echo "$TEXT" | piper --model "$VOICE" --output_file "$WAV" 2>/dev/null
    END=$(python3 -c "import time; print(time.monotonic())")

    PROC_SEC=$(python3 -c "print($END - $START)")

    if [ -f "$WAV" ]; then
        AUDIO_SEC=$(python3 -c "
import wave
with wave.open('$WAV', 'rb') as wf:
    print(wf.getnframes() / wf.getframerate())
")
        RTF=$(python3 -c "print($PROC_SEC / $AUDIO_SEC if $AUDIO_SEC > 0 else 999)")
        TTFS_MS=$(python3 -c "print($PROC_SEC * 1000)")  # Non-streaming, approximate

        echo "  [$((i+1))/${#SENTENCES[@]}] ${AUDIO_SEC}s audio, RTF=${RTF}, time=${PROC_SEC}s"

        TOTAL_AUDIO_SEC=$(python3 -c "print($TOTAL_AUDIO_SEC + $AUDIO_SEC)")
        TOTAL_PROC_SEC=$(python3 -c "print($TOTAL_PROC_SEC + $PROC_SEC)")
        TOTAL_VALID=$((TOTAL_VALID + 1))

        [ "$i" -gt 0 ] && JSON_RESULTS+=","
        JSON_RESULTS+="{\"text\":\"$TEXT\",\"audio_duration_s\":$AUDIO_SEC,\"total_time_s\":$PROC_SEC,\"rtf\":$RTF,\"ttfs_ms\":$TTFS_MS}"
    else
        echo "  [$((i+1))/${#SENTENCES[@]}] FAILED"
        [ "$i" -gt 0 ] && JSON_RESULTS+=","
        JSON_RESULTS+="{\"text\":\"$TEXT\",\"audio_duration_s\":0,\"total_time_s\":$PROC_SEC,\"rtf\":999,\"ttfs_ms\":null}"
    fi
done

JSON_RESULTS+="]"

AVG_RTF=$(python3 -c "print($TOTAL_PROC_SEC / $TOTAL_AUDIO_SEC if $TOTAL_AUDIO_SEC > 0 else 999)")
AVG_TTFS=$(python3 -c "print($TOTAL_PROC_SEC / $TOTAL_VALID * 1000 if $TOTAL_VALID > 0 else 0)")

echo ""
echo "  Average RTF: $AVG_RTF"
echo "  Average TTFS: ${AVG_TTFS}ms (approx, non-streaming)"

if [ -n "$OUTPUT" ]; then
    mkdir -p "$(dirname "$OUTPUT")"
    python3 -c "
import json
data = {
    'engine': 'piper',
    'voice': '$VOICE',
    'summary': {
        'n_sentences': ${#SENTENCES[@]},
        'n_valid': $TOTAL_VALID,
        'avg_rtf': $AVG_RTF,
        'avg_ttfs_ms': $AVG_TTFS,
        'total_audio_s': $TOTAL_AUDIO_SEC,
        'total_time_s': $TOTAL_PROC_SEC,
    },
    'per_sentence': json.loads('''$JSON_RESULTS'''),
}
with open('$OUTPUT', 'w') as f:
    json.dump(data, f, indent=2)
print(f'Results saved to $OUTPUT')
"
fi
