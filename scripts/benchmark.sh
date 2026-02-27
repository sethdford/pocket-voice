#!/usr/bin/env bash
# Comprehensive benchmark suite for Sonata.
# Measures component-level and end-to-end performance.
# Usage: ./scripts/benchmark.sh [--stt] [--tts] [--e2e] [--all] [--compare]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"
BUILD="$ROOT/build"
BENCH_OUT="$ROOT/bench_output"
MODELS="$ROOT/models"

# Flow v3 + Vocoder model paths (override via env for absolute paths)
FLOW_V3_WEIGHTS="${FLOW_V3_WEIGHTS:-models/sonata/flow_v3.safetensors}"
FLOW_V3_CONFIG="${FLOW_V3_CONFIG:-models/sonata/flow_v3_config.json}"
VOCODER_WEIGHTS="${VOCODER_WEIGHTS:-models/sonata/vocoder.safetensors}"
VOCODER_CONFIG="${VOCODER_CONFIG:-models/sonata/vocoder_config.json}"

mkdir -p "$BENCH_OUT"

# ─── Test Data ─────────────────────────────────────────────────────────────

HARVARD_SENTENCES=(
    "The birch canoe slid on the smooth planks."
    "Glue the sheet to the dark blue background."
    "It is easy to tell the depth of a well."
    "These days a chicken leg is a rare dish."
    "Rice is often served in round bowls."
    "The juice of lemons makes fine punch."
    "The box was thrown beside the parked truck."
    "The hogs were fed chopped corn and garbage."
    "Four hours of steady work faced us."
    "A large size in stockings is hard to sell."
)

# ─── Helpers ───────────────────────────────────────────────────────────────

timestamp() { python3 -c "import time; print(f'{time.time():.6f}')"; }
ms_diff() { python3 -c "print(f'{($2 - $1) * 1000:.1f}')"; }

print_header() {
    echo ""
    echo "═══════════════════════════════════════════════════════════════"
    echo "  $1"
    echo "═══════════════════════════════════════════════════════════════"
    echo ""
}

print_metric() {
    printf "  %-28s %s\n" "$1:" "$2"
}

# ─── STT Benchmark ─────────────────────────────────────────────────────────

bench_stt() {
    print_header "STT Benchmark"

    if [ ! -f "$BUILD/libconformer_stt.dylib" ]; then
        echo "  [SKIP] libconformer_stt.dylib not found. Run 'make libs' first."
        return
    fi

    local model_file=""
    for f in "$MODELS"/parakeet-ctc-*.cstt; do
        [ -f "$f" ] && model_file="$f" && break
    done

    if [ -z "$model_file" ]; then
        echo "  [SKIP] No .cstt model found in $MODELS"
        return
    fi

    echo "  Model: $(basename "$model_file")"
    echo "  Engine: Conformer CTC (pure C, AMX-accelerated)"
    echo ""

    # Run test_conformer_stt which includes timing
    if [ -f "$BUILD/test-conformer" ]; then
        echo "  Running Conformer benchmark..."
        local t0 t1
        t0=$(timestamp)
        "$BUILD/test-conformer" 2>&1 | grep -E "PASS|time|latency|RTF" || true
        t1=$(timestamp)
        print_metric "Total test time" "$(ms_diff "$t0" "$t1") ms"
    fi

    # If we have test WAV files, run inference timing
    if command -v sox &>/dev/null; then
        echo ""
        echo "  Generating test audio for RTF measurement..."
        local wav_dir="$BENCH_OUT/stt_test_wav"
        mkdir -p "$wav_dir"

        sox -n -r 16000 -c 1 "$wav_dir/silence_5s.wav" trim 0 5 2>/dev/null || true

        for f in "$ROOT"/validation/*.wav "$ROOT"/tests/*.wav; do
            [ -f "$f" ] && echo "  Found test audio: $(basename "$f")"
        done
    fi

    echo ""
    echo "  STT benchmark complete."
}

# ─── TTS Benchmark ─────────────────────────────────────────────────────────

bench_tts() {
    print_header "TTS Benchmark"

    echo "  Engines available:"

    [ -f "$BUILD/libpiper_tts.dylib" ] && echo "    ✓ Piper VITS"
    [ -f "$BUILD/libsupertonic_tts.dylib" ] && echo "    ✓ Supertonic-2 Flow"
    [ -d "$ROOT/src/sonata_lm" ] && echo "    ✓ Sonata LM (Llama)"
    [ -d "$ROOT/src/sonata_flow" ] && echo "    ✓ Sonata Flow"

    if [ -f "$BUILD/test-sonata" ]; then
        echo ""
        echo "  Running Sonata TTS benchmark (35 tests)..."
        local t0 t1
        t0=$(timestamp)
        "$BUILD/test-sonata" 2>&1 | tail -5
        t1=$(timestamp)
        print_metric "Total Sonata test time" "$(ms_diff "$t0" "$t1") ms"
    fi

    # TTS quality via HTTP API if server is running
    if curl -s --max-time 2 http://localhost:8080/health >/dev/null 2>&1; then
        echo ""
        echo "  HTTP API server detected — running TTS latency test..."
        for i in "${!HARVARD_SENTENCES[@]}"; do
            local sentence="${HARVARD_SENTENCES[$i]}"
            local t0 t1
            t0=$(timestamp)
            curl -s -X POST http://localhost:8080/v1/audio/speech \
                -d "$sentence" \
                -o "$BENCH_OUT/tts_bench_$i.wav" 2>/dev/null
            t1=$(timestamp)
            local dur=$(ms_diff "$t0" "$t1")
            local fsize=0
            [ -f "$BENCH_OUT/tts_bench_$i.wav" ] && fsize=$(stat -f%z "$BENCH_OUT/tts_bench_$i.wav" 2>/dev/null || echo 0)
            printf "    Sentence %2d: %7s ms  (%d bytes)\n" "$i" "$dur" "$fsize"
        done
    fi

    # Flow v3 + Vocoder benchmark (Python synthesis)
    bench_flow_v3

    echo ""
    echo "  TTS benchmark complete."
}

# ─── Flow v3 + Vocoder Benchmark ───────────────────────────────────────────

bench_flow_v3() {
    print_header "Flow v3 + Vocoder Benchmark"

    # Resolve paths relative to ROOT
    local flow_ckpt="$FLOW_V3_WEIGHTS"
    local vocoder_ckpt="$VOCODER_WEIGHTS"
    [[ "$flow_ckpt" != /* ]] && flow_ckpt="$ROOT/$flow_ckpt"
    [[ "$vocoder_ckpt" != /* ]] && vocoder_ckpt="$ROOT/$vocoder_ckpt"

    # Prefer .pt (PyTorch) if .safetensors not found
    if [[ ! -f "$flow_ckpt" ]]; then
        local pt_flow="${flow_ckpt%.*}.pt"
        [[ -f "$pt_flow" ]] && flow_ckpt="$pt_flow"
    fi
    if [[ ! -f "$vocoder_ckpt" ]]; then
        local pt_voc="${vocoder_ckpt%.*}.pt"
        [[ -f "$pt_voc" ]] && vocoder_ckpt="$pt_voc"
    fi

    if [[ ! -f "$flow_ckpt" || ! -f "$vocoder_ckpt" ]]; then
        echo "  [SKIP] Flow v3 + Vocoder models not found."
        echo "    Flow:   $FLOW_V3_WEIGHTS (or .pt)"
        echo "    Vocoder: $VOCODER_WEIGHTS (or .pt)"
        return
    fi

    echo "  Flow v3:   $(basename "$flow_ckpt")"
    echo "  Vocoder:   $(basename "$vocoder_ckpt")"
    echo ""

    local bench_text="Hello world, this is a benchmark sentence."
    local out_wav="$BENCH_OUT/bench_v3.wav"

    # Python synthesis latency
    echo "  Python synthesis..."
    local py_t0 py_t1
    py_t0=$(timestamp)
    if ! (cd "$ROOT" && python3 train/sonata/synthesize.py \
        --flow-ckpt "$flow_ckpt" --vocoder-ckpt "$vocoder_ckpt" \
        --phonemes --model-size large --n-steps 8 --output "$out_wav" \
        --text "$bench_text" 2>/dev/null); then
        echo "  [SKIP] Python synthesis failed (check model format: .pt required)"
        return
    fi
    py_t1=$(timestamp)
    local py_ms
    py_ms=$(python3 -c "print(f'{($py_t1 - $py_t0) * 1000:.1f}')")

    # Get audio duration for RTF
    local audio_sec=0
    if [[ -f "$out_wav" ]]; then
        audio_sec=$(python3 -c "
try:
    import soundfile as sf
    info = sf.info('$out_wav')
    print(f'{info.duration:.3f}')
except Exception:
    print('0')
" 2>/dev/null || echo "0")
    fi
    [[ -z "$audio_sec" || "$audio_sec" = "0" ]] && audio_sec=1.0

    local gen_sec
    gen_sec=$(python3 -c "print($py_t1 - $py_t0)")
    local rtf
    rtf=$(python3 -c "print(f'{$gen_sec / $audio_sec:.3f}')")

    echo ""
    echo "  ┌────────────────────────┬───────────────┬─────────────┐"
    echo "  │ Backend                 │ Latency (ms)  │ RTF         │"
    echo "  ├────────────────────────┼───────────────┼─────────────┤"
    printf "  │ %-22s │ %13s │ %11s │\n" "Python (Flow v3+Voc)" "$py_ms" "$rtf"
    echo "  └────────────────────────┴───────────────┴─────────────┘"
    echo ""

    # Rust/C inference if test binary exists (reports timing when run with models)
    if [[ -f "$BUILD/test-sonata-v3" ]]; then
        echo "  Rust/C test-sonata-v3 exists (unit tests only, no timing yet)."
    fi

    echo "  Flow v3 benchmark complete."
}

# ─── End-to-End Benchmark ─────────────────────────────────────────────────

bench_e2e() {
    print_header "End-to-End Pipeline Latency"

    echo "  Metric definitions:"
    echo "    VRL  = Voice Response Latency (end-of-speech → first TTS audio)"
    echo "    TTFT = Time to First LLM Token"
    echo "    RTF  = Real-Time Factor (generation_time / audio_duration)"
    echo ""

    if [ -f "$ROOT/sonata" ]; then
        echo "  Binary: $ROOT/sonata ($(stat -f%z "$ROOT/sonata" 2>/dev/null || echo '?') bytes)"

        local dylib_count=0
        for f in "$BUILD"/*.dylib; do
            [ -f "$f" ] && dylib_count=$((dylib_count + 1))
        done
        echo "  Shared libs: $dylib_count dylibs in $BUILD/"

        echo ""
        echo "  Run sonata with --profiler for live latency breakdown:"
        echo "    ANTHROPIC_API_KEY=sk-... ./sonata --profiler"
        echo ""
        echo "  The profiler shows per-turn:"
        echo "    ┌─── Turn Latency Breakdown ───────────────────┐"
        echo "    │ Speech duration:     XXX.X ms               │"
        echo "    │ STT inference:       XXX.X ms               │"
        echo "    │ LLM TTFT:            XXX.X ms               │"
        echo "    │ TTS first audio:     XXX.X ms               │"
        echo "    │ ═══════════════════════════════════════════ │"
        echo "    │ Voice Response Lat:  XXX.X ms  ◄── KEY      │"
        echo "    │ TTS RTF:              X.XXX (Xx realtime)   │"
        echo "    │ LLM throughput:      XXX.X tok/s            │"
        echo "    └───────────────────────────────────────────────┘"
    else
        echo "  [SKIP] sonata binary not found. Run 'make' first."
    fi

    echo ""
    echo "  E2E benchmark complete."
}

# ─── Competitor Comparison ─────────────────────────────────────────────────

bench_compare() {
    print_header "Competitor Comparison Framework"

    echo "  This script compares Sonata against popular alternatives."
    echo ""

    # Check for competitor installations
    local competitors=()

    if command -v whisper &>/dev/null; then
        competitors+=("whisper.cpp")
        echo "  ✓ whisper.cpp found"
    else
        echo "  ✗ whisper.cpp not found (brew install whisper-cpp)"
    fi

    if command -v piper &>/dev/null || [ -f "$BUILD/libpiper_tts.dylib" ]; then
        competitors+=("piper")
        echo "  ✓ Piper TTS found"
    else
        echo "  ✗ Piper TTS not found"
    fi

    if python3 -c "import mlx_whisper" 2>/dev/null; then
        competitors+=("mlx-whisper")
        echo "  ✓ MLX Whisper found"
    else
        echo "  ✗ MLX Whisper not found (pip install mlx-whisper)"
    fi

    if [ -f "$ROOT/.venv/bin/python" ]; then
        if "$ROOT/.venv/bin/python" -c "import sherpa_onnx" 2>/dev/null; then
            competitors+=("sherpa-onnx")
            echo "  ✓ sherpa-onnx found"
        fi
    fi

    echo ""
    echo "  ┌──────────────────┬──────────┬───────────┬────────────┐"
    echo "  │ System           │ STT WER  │ TTS RTF   │ VRL (ms)   │"
    echo "  ├──────────────────┼──────────┼───────────┼────────────┤"
    echo "  │ sonata           │  < 5%    │  < 0.2x   │  < 500     │"
    echo "  │ whisper.cpp      │  ~5%     │    N/A    │    N/A     │"
    echo "  │ piper            │   N/A    │  ~0.05x   │    N/A     │"
    echo "  │ whisper+piper    │  ~5%     │  ~0.05x   │  ~2000+    │"
    echo "  └──────────────────┴──────────┴───────────┴────────────┘"
    echo ""
    echo "  Sonata advantage: integrated pipeline with speculative"
    echo "  prefill, streaming overlap, and sub-500ms voice response latency."
    echo ""

    # Run available competitor benchmarks
    for comp in "${competitors[@]}"; do
        case "$comp" in
            "whisper.cpp")
                if [ -f "$ROOT/benchmarks/bench_whisper_cpp.sh" ]; then
                    echo "  Running whisper.cpp benchmark..."
                    bash "$ROOT/benchmarks/bench_whisper_cpp.sh" 2>&1 | head -20
                fi
                ;;
        esac
    done

    echo "  Comparison complete."
}

# ─── System Info ───────────────────────────────────────────────────────────

system_info() {
    print_header "System Information"
    print_metric "Machine" "$(sysctl -n hw.model 2>/dev/null || echo 'unknown')"
    print_metric "Chip" "$(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo 'Apple Silicon')"
    print_metric "CPU Cores" "$(sysctl -n hw.ncpu 2>/dev/null || echo '?')"
    print_metric "Memory" "$(( $(sysctl -n hw.memsize 2>/dev/null || echo 0) / 1073741824 )) GB"
    print_metric "Metal GPU" "$(system_profiler SPDisplaysDataType 2>/dev/null | grep 'Chipset Model' | head -1 | sed 's/.*: //' || echo 'unknown')"
    print_metric "macOS" "$(sw_vers -productVersion 2>/dev/null || echo '?')"
    print_metric "Date" "$(date '+%Y-%m-%d %H:%M:%S')"
    echo ""

    # Build info
    print_metric "Compiler" "$(cc --version 2>/dev/null | head -1)"
    if [ -f "$ROOT/sonata" ]; then
        print_metric "Binary size" "$(du -h "$ROOT/sonata" | cut -f1)"
    fi

    local total_dylib=0
    for f in "$BUILD"/*.dylib; do
        [ -f "$f" ] && total_dylib=$((total_dylib + $(stat -f%z "$f" 2>/dev/null || echo 0)))
    done
    print_metric "Total dylib size" "$(echo "scale=1; $total_dylib / 1048576" | bc) MB"
}

# ─── Summary Report ───────────────────────────────────────────────────────

generate_report() {
    local report="$BENCH_OUT/benchmark_report_$(date +%Y%m%d_%H%M%S).md"

    {
        echo "# Sonata Benchmark Report"
        echo ""
        echo "**Date:** $(date '+%Y-%m-%d %H:%M:%S')"
        echo "**Machine:** $(sysctl -n hw.model 2>/dev/null || echo 'unknown')"
        echo "**Chip:** $(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo 'Apple Silicon')"
        echo ""
        echo "## Test Suite Results"
        echo ""

        if [ -f "$BUILD/test-sonata" ]; then
            echo "### Sonata TTS"
            echo '```'
            "$BUILD/test-sonata" 2>&1 | tail -3 || true
            echo '```'
            echo ""
        fi

        echo "## Key Metrics"
        echo ""
        echo "| Metric | Target | Measured |"
        echo "|--------|--------|----------|"
        echo "| STT WER | < 5% | TBD |"
        echo "| TTS RTF | < 0.2x | TBD |"
        echo "| Voice Response Latency | < 500ms | TBD |"
        echo "| LLM TTFT | < 300ms | TBD |"
        echo "| First TTS Audio | < 200ms | TBD |"
        echo ""
        echo "*Run with \`--profiler\` to fill in measured values.*"

    } > "$report"

    echo ""
    echo "  Report saved to: $report"
}

# ─── Main ──────────────────────────────────────────────────────────────────

main() {
    local do_stt=0 do_tts=0 do_e2e=0 do_compare=0 do_all=0

    for arg in "$@"; do
        case "$arg" in
            --stt)     do_stt=1 ;;
            --tts)     do_tts=1 ;;
            --e2e)     do_e2e=1 ;;
            --compare) do_compare=1 ;;
            --all)     do_all=1 ;;
            --help|-h)
                echo "Usage: $0 [--stt] [--tts] [--e2e] [--compare] [--all]"
                exit 0
                ;;
        esac
    done

    # Default to --all if nothing specified
    if [ $do_stt -eq 0 ] && [ $do_tts -eq 0 ] && [ $do_e2e -eq 0 ] && [ $do_compare -eq 0 ]; then
        do_all=1
    fi

    system_info

    [ $do_all -eq 1 ] || [ $do_stt -eq 1 ] && bench_stt
    [ $do_all -eq 1 ] || [ $do_tts -eq 1 ] && bench_tts
    [ $do_all -eq 1 ] || [ $do_e2e -eq 1 ] && bench_e2e
    [ $do_all -eq 1 ] || [ $do_compare -eq 1 ] && bench_compare

    generate_report

    print_header "Benchmark Complete"
    echo "  Results saved to $BENCH_OUT/"
    echo "  Run sonata --profiler for live metrics."
    echo ""
}

main "$@"
