#!/bin/bash
##############################################################################
# Sonata v2 Training Orchestrator
#
# Trains all 5 models in dependency order on GCE with NVIDIA L4 GPU:
#   Wave 1 (parallel): Speaker Encoder (CAM++) + Audio Codec
#   Wave 2 (after Wave 1): STT + TTS (parallel)
#   Wave 3 (after Wave 2): CFM Decoder
#
# After each model: export to safetensors + Core ML, upload to GCS.
#
# Usage:
#   ./orchestrate.sh                    # Full pipeline
#   ./orchestrate.sh --model codec      # Train single model
#   ./orchestrate.sh --wave 2           # Start from wave 2
#   ./orchestrate.sh --export-only      # Re-export all checkpoints
#   ./orchestrate.sh --synthetic        # Test run with synthetic data
##############################################################################

set -euo pipefail

# ── Configuration ──────────────────────────────────────────────────────────

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Data paths (GCE gcsfuse mount)
DATA_DIR="${SONATA_DATA_DIR:-/mnt/sonata/data}"
CHECKPOINT_DIR="${SONATA_CHECKPOINT_DIR:-/mnt/sonata/checkpoints}"
EXPORT_DIR="${SONATA_EXPORT_DIR:-/mnt/sonata/models}"
GCS_BUCKET="${SONATA_GCS_BUCKET:-gs://sonata-training-johnb-2025}"

# Training data subdirectories
LIBRITTS_DIR="${DATA_DIR}/libritts-r/LibriTTS_R"
MUSAN_DIR="${DATA_DIR}/musan"
RIR_DIR="${DATA_DIR}/RIRS_NOISES/simulated_rirs"
AUDIO_DIR="${DATA_DIR}/audio"          # General audio for codec training
STT_TEXT_DIR="${DATA_DIR}/transcripts"  # Text transcripts for STT

# GPU
DEVICE="cuda"
NUM_WORKERS=8

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
NC='\033[0m'

# ── State tracking ─────────────────────────────────────────────────────────

LOG_FILE="${CHECKPOINT_DIR}/orchestrator.log"
STATE_FILE="${CHECKPOINT_DIR}/orchestrator_state.json"

log() { echo -e "${BLUE}[$(date +%H:%M:%S)]${NC} $1" | tee -a "$LOG_FILE"; }
success() { echo -e "${GREEN}[✓ $(date +%H:%M:%S)]${NC} $1" | tee -a "$LOG_FILE"; }
warn() { echo -e "${YELLOW}[⚠ $(date +%H:%M:%S)]${NC} $1" | tee -a "$LOG_FILE"; }
fail() { echo -e "${RED}[✗ $(date +%H:%M:%S)]${NC} $1" | tee -a "$LOG_FILE"; }
header() {
    echo -e "\n${MAGENTA}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}" | tee -a "$LOG_FILE"
    echo -e "${MAGENTA}  $1${NC}" | tee -a "$LOG_FILE"
    echo -e "${MAGENTA}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}\n" | tee -a "$LOG_FILE"
}

save_state() {
    local model="$1" status="$2"
    python3 -c "
import json, os
state_file = '$STATE_FILE'
state = {}
if os.path.exists(state_file):
    with open(state_file) as f:
        state = json.load(f)
state['$model'] = {'status': '$status', 'timestamp': '$(date -Iseconds)'}
with open(state_file, 'w') as f:
    json.dump(state, f, indent=2)
"
}

check_state() {
    local model="$1"
    if [[ -f "$STATE_FILE" ]]; then
        python3 -c "
import json
with open('$STATE_FILE') as f:
    state = json.load(f)
s = state.get('$model', {}).get('status', 'pending')
print(s)
" 2>/dev/null || echo "pending"
    else
        echo "pending"
    fi
}

# ── Argument parsing ───────────────────────────────────────────────────────

SINGLE_MODEL=""
START_WAVE=1
EXPORT_ONLY=false
SYNTHETIC=false
DRY_RUN=false
SKIP_EXPORT=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --model) SINGLE_MODEL="$2"; shift 2 ;;
        --wave) START_WAVE="$2"; shift 2 ;;
        --export-only) EXPORT_ONLY=true; shift ;;
        --synthetic) SYNTHETIC=true; shift ;;
        --dry-run) DRY_RUN=true; shift ;;
        --skip-export) SKIP_EXPORT=true; shift ;;
        --help|-h) cat << 'EOF'
Sonata v2 Training Orchestrator

Usage: ./orchestrate.sh [OPTIONS]

Options:
  --model <name>     Train single model: cam, codec, stt, tts, cfm
  --wave <N>         Start from wave N (1, 2, or 3)
  --export-only      Skip training, re-export all checkpoints
  --synthetic        Use synthetic data for testing
  --dry-run          Print commands without executing
  --skip-export      Skip export step after training
  -h, --help         Show this help

Training Order:
  Wave 1 (parallel):  Speaker Encoder (CAM++) + Audio Codec
  Wave 2 (parallel):  STT + TTS  (requires Wave 1)
  Wave 3:             CFM Decoder (requires Wave 1)

Examples:
  ./orchestrate.sh                        # Full pipeline
  ./orchestrate.sh --synthetic            # Test with synthetic data
  ./orchestrate.sh --model codec          # Train codec only
  ./orchestrate.sh --wave 2              # Resume from wave 2
  ./orchestrate.sh --export-only         # Re-export all models
EOF
            exit 0 ;;
        *) fail "Unknown option: $1"; exit 1 ;;
    esac
done

# ── Prerequisites ──────────────────────────────────────────────────────────

check_prerequisites() {
    header "Checking Prerequisites"

    # Python
    if ! command -v python3 &> /dev/null; then
        fail "python3 not found"; exit 1
    fi
    log "Python: $(python3 --version)"

    # PyTorch + CUDA
    python3 -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'; print(f'PyTorch {torch.__version__} + CUDA {torch.version.cuda}')" 2>/dev/null || {
        fail "PyTorch with CUDA not available"; exit 1
    }
    log "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'unknown')"

    # Required packages
    python3 -c "import safetensors; import torchaudio; import numpy" 2>/dev/null || {
        warn "Missing packages, installing..."
        pip install -r "$SCRIPT_DIR/requirements.txt"
    }

    # Data directories
    if [[ "$SYNTHETIC" == false ]]; then
        [[ -d "$LIBRITTS_DIR" ]] || warn "LibriTTS-R not found at $LIBRITTS_DIR"
        [[ -d "$AUDIO_DIR" ]] || warn "Audio data not found at $AUDIO_DIR (needed for codec)"
    fi

    # Checkpoint directory
    mkdir -p "$CHECKPOINT_DIR"/{cam,codec,stt,tts,cfm}
    mkdir -p "$EXPORT_DIR"
    mkdir -p "$(dirname "$LOG_FILE")"

    success "Prerequisites OK"
}

# ── Training functions ─────────────────────────────────────────────────────

train_speaker_encoder() {
    header "Training Speaker Encoder (CAM++) — ~13 hours on L4"

    local state=$(check_state "cam")
    if [[ "$state" == "complete" ]]; then
        success "Speaker encoder already trained, skipping"
        return 0
    fi

    save_state "cam" "training"

    local cmd="python3 $SCRIPT_DIR/train_speaker_encoder.py \
        --device $DEVICE \
        --batch-size 64 \
        --n-epochs 40 \
        --lr 0.001 \
        --scale 30.0 \
        --margin 0.2 \
        --sub-centers 2 \
        --warmup-epochs 2 \
        --val-every 2 \
        --patience 10 \
        --asnorm \
        --num-workers $NUM_WORKERS \
        --pin-memory \
        --mixed-precision \
        --output-dir $CHECKPOINT_DIR/cam"

    if [[ "$SYNTHETIC" == false ]]; then
        cmd="$cmd --data-dir $LIBRITTS_DIR"
        [[ -d "$MUSAN_DIR" ]] && cmd="$cmd --musan-dir $MUSAN_DIR"
        [[ -d "$RIR_DIR" ]] && cmd="$cmd --rir-dir $RIR_DIR"
    else
        cmd="$cmd --data-dir /tmp/synthetic --synthetic"
    fi

    if [[ "$DRY_RUN" == true ]]; then
        log "DRY RUN: $cmd"
        return 0
    fi

    log "Starting speaker encoder training..."
    eval $cmd 2>&1 | tee -a "$LOG_FILE"

    # Stage 2: Large margin fine-tuning
    if [[ -f "$CHECKPOINT_DIR/cam/speaker_encoder_best.pt" ]]; then
        log "Stage 2: Large margin fine-tuning (5 epochs)..."
        local ft_cmd="python3 $SCRIPT_DIR/train_speaker_encoder.py \
            --device $DEVICE \
            --batch-size 32 \
            --sub-centers 2 \
            --asnorm \
            --num-workers $NUM_WORKERS \
            --pin-memory \
            --mixed-precision \
            --fine-tune \
            --fine-tune-margin 0.5 \
            --fine-tune-crop 6.0 \
            --resume $CHECKPOINT_DIR/cam/speaker_encoder_best.pt \
            --output-dir $CHECKPOINT_DIR/cam"

        if [[ "$SYNTHETIC" == false ]]; then
            ft_cmd="$ft_cmd --data-dir $LIBRITTS_DIR"
            [[ -d "$MUSAN_DIR" ]] && ft_cmd="$ft_cmd --musan-dir $MUSAN_DIR"
            [[ -d "$RIR_DIR" ]] && ft_cmd="$ft_cmd --rir-dir $RIR_DIR"
        else
            ft_cmd="$ft_cmd --data-dir /tmp/synthetic --synthetic"
        fi

        eval $ft_cmd 2>&1 | tee -a "$LOG_FILE"
    fi

    save_state "cam" "complete"
    success "Speaker encoder training complete"
}

train_codec() {
    header "Training Audio Codec — ~3-5 days on L4"

    local state=$(check_state "codec")
    if [[ "$state" == "complete" ]]; then
        success "Codec already trained, skipping"
        return 0
    fi

    save_state "codec" "training"

    local cmd="python3 $SCRIPT_DIR/train_codec.py \
        --batch_size 16 \
        --lr 3e-4 \
        --epochs 100 \
        --save_every 10 \
        --val_split 0.1"

    if [[ "$SYNTHETIC" == true ]]; then
        cmd="$cmd --data_dir /tmp/synthetic --synthetic"
    else
        cmd="$cmd --data_dir $AUDIO_DIR"
    fi

    # Resume if checkpoint exists
    if [[ -f "$CHECKPOINT_DIR/codec/codec_best.pt" ]]; then
        cmd="$cmd --resume $CHECKPOINT_DIR/codec/codec_best.pt"
        log "Resuming from existing codec checkpoint"
    fi

    if [[ "$DRY_RUN" == true ]]; then
        log "DRY RUN: $cmd"
        return 0
    fi

    log "Starting codec training..."
    eval $cmd 2>&1 | tee -a "$LOG_FILE"

    save_state "codec" "complete"
    success "Codec training complete"
}

train_stt() {
    header "Training STT (Conformer CTC) — ~3-4 days on L4"

    local state=$(check_state "stt")
    if [[ "$state" == "complete" ]]; then
        success "STT already trained, skipping"
        return 0
    fi

    # Check dependency: codec
    local codec_state=$(check_state "codec")
    if [[ "$codec_state" != "complete" && "$SYNTHETIC" == false ]]; then
        fail "STT requires trained codec. Train codec first."
        return 1
    fi

    save_state "stt" "training"

    local cmd="python3 $SCRIPT_DIR/train_stt.py \
        --batch_size 16 \
        --lr 5e-4 \
        --epochs 100"

    if [[ "$SYNTHETIC" == true ]]; then
        cmd="$cmd --data_dir /tmp/synthetic --text_dir /tmp/synthetic --synthetic --mel_mode"
    else
        cmd="$cmd --data_dir $AUDIO_DIR --text_dir $STT_TEXT_DIR"
        # Use codec embeddings if codec checkpoint exists
        if [[ -f "$CHECKPOINT_DIR/codec/codec_best.pt" ]]; then
            cmd="$cmd --codec_checkpoint $CHECKPOINT_DIR/codec/codec_best.pt"
        else
            cmd="$cmd --mel_mode"
            warn "No codec checkpoint — using mel mode for STT"
        fi
    fi

    if [[ "$DRY_RUN" == true ]]; then
        log "DRY RUN: $cmd"
        return 0
    fi

    log "Starting STT training..."
    eval $cmd 2>&1 | tee -a "$LOG_FILE"

    save_state "stt" "complete"
    success "STT training complete"
}

train_tts() {
    header "Training TTS (AdaIN + Emotion) — ~3-5 days on L4"

    local state=$(check_state "tts")
    if [[ "$state" == "complete" ]]; then
        success "TTS already trained, skipping"
        return 0
    fi

    # Check dependencies: codec + speaker encoder
    if [[ "$SYNTHETIC" == false ]]; then
        local codec_state=$(check_state "codec")
        local cam_state=$(check_state "cam")
        if [[ "$codec_state" != "complete" ]]; then
            fail "TTS requires trained codec. Train codec first."; return 1
        fi
        if [[ "$cam_state" != "complete" ]]; then
            fail "TTS requires trained speaker encoder. Train CAM++ first."; return 1
        fi
    fi

    save_state "tts" "training"

    local cmd="python3 $SCRIPT_DIR/train_tts.py \
        --batch_size 16 \
        --lr 1e-4 \
        --epochs 100"

    if [[ "$SYNTHETIC" == true ]]; then
        cmd="$cmd --synthetic"
    else
        cmd="$cmd --data_dir $AUDIO_DIR --text_dir $STT_TEXT_DIR"
        [[ -f "$CHECKPOINT_DIR/codec/codec_best.pt" ]] && \
            cmd="$cmd --codec_checkpoint $CHECKPOINT_DIR/codec/codec_best.pt"
        [[ -f "$CHECKPOINT_DIR/cam/speaker_encoder_best.pt" ]] && \
            cmd="$cmd --speaker_checkpoint $CHECKPOINT_DIR/cam/speaker_encoder_best.pt"
    fi

    if [[ "$DRY_RUN" == true ]]; then
        log "DRY RUN: $cmd"
        return 0
    fi

    log "Starting TTS training..."
    eval $cmd 2>&1 | tee -a "$LOG_FILE"

    save_state "tts" "complete"
    success "TTS training complete"
}

train_cfm() {
    header "Training CFM Decoder — ~3-5 days on L4"

    local state=$(check_state "cfm")
    if [[ "$state" == "complete" ]]; then
        success "CFM already trained, skipping"
        return 0
    fi

    # Check dependency: speaker encoder
    if [[ "$SYNTHETIC" == false ]]; then
        local cam_state=$(check_state "cam")
        if [[ "$cam_state" != "complete" ]]; then
            fail "CFM requires trained speaker encoder. Train CAM++ first."; return 1
        fi
    fi

    save_state "cfm" "training"

    local cmd="python3 $SCRIPT_DIR/train_cfm.py \
        --batch_size 16 \
        --lr 1e-4 \
        --epochs 100"

    if [[ "$SYNTHETIC" == true ]]; then
        cmd="$cmd --synthetic"
    else
        cmd="$cmd --data_dir $AUDIO_DIR"
        [[ -f "$CHECKPOINT_DIR/cam/speaker_encoder_best.pt" ]] && \
            cmd="$cmd --speaker_checkpoint $CHECKPOINT_DIR/cam/speaker_encoder_best.pt"
    fi

    if [[ "$DRY_RUN" == true ]]; then
        log "DRY RUN: $cmd"
        return 0
    fi

    log "Starting CFM training..."
    eval $cmd 2>&1 | tee -a "$LOG_FILE"

    save_state "cfm" "complete"
    success "CFM training complete"
}

# ── Export functions ───────────────────────────────────────────────────────

export_model() {
    local model="$1"
    local checkpoint="$2"
    local output="$EXPORT_DIR/sonata-${model}.safetensors"

    if [[ ! -f "$checkpoint" ]]; then
        warn "No checkpoint for $model at $checkpoint, skipping export"
        return 1
    fi

    log "Exporting $model to safetensors..."
    python3 "$SCRIPT_DIR/export_candle.py" \
        --model "$model" \
        --checkpoint "$checkpoint" \
        --output "$output" \
        --verify 2>&1 | tee -a "$LOG_FILE"

    success "Exported $model → $output ($(du -h "$output" | cut -f1))"

    # Also export Core ML for ANE inference (if coremltools available)
    if python3 -c "import coremltools" 2>/dev/null; then
        local coreml_output="$EXPORT_DIR/sonata-${model}.mlpackage"
        log "Exporting $model to Core ML for ANE..."
        python3 "$SCRIPT_DIR/export_coreml.py" \
            --model "$model" \
            --checkpoint "$checkpoint" \
            --output "$coreml_output" 2>&1 | tee -a "$LOG_FILE" || \
            warn "Core ML export failed for $model (non-fatal)"
    fi

    # Upload to GCS
    if command -v gsutil &> /dev/null; then
        log "Uploading $model to GCS..."
        gsutil -q cp "$output" "$GCS_BUCKET/models/" 2>/dev/null || \
            warn "GCS upload failed (non-fatal)"
    fi
}

export_all() {
    header "Exporting All Models"

    export_model "cam" "$CHECKPOINT_DIR/cam/speaker_encoder_best.pt"
    export_model "codec" "$CHECKPOINT_DIR/codec/codec_best.pt"
    export_model "stt" "$CHECKPOINT_DIR/stt/stt_best.pt"
    export_model "tts" "$CHECKPOINT_DIR/tts/tts_best.pt"
    export_model "cfm" "$CHECKPOINT_DIR/cfm/cfm_best.pt"

    # Quantize models (4-bit for mobile, 8-bit for desktop)
    if [[ -f "$EXPORT_DIR/sonata-codec.safetensors" ]]; then
        log "Quantizing models..."
        for model in codec stt tts cfm; do
            local input="$EXPORT_DIR/sonata-${model}.safetensors"
            [[ -f "$input" ]] || continue
            python3 "$SCRIPT_DIR/quantize.py" \
                --input "$input" \
                --output "$EXPORT_DIR/sonata-${model}-int8.safetensors" \
                --bits 8 2>&1 | tee -a "$LOG_FILE" || true
            python3 "$SCRIPT_DIR/quantize.py" \
                --input "$input" \
                --output "$EXPORT_DIR/sonata-${model}-int4.safetensors" \
                --bits 4 2>&1 | tee -a "$LOG_FILE" || true
        done
    fi

    success "All exports complete"
    log "Models at: $EXPORT_DIR/"
    ls -lh "$EXPORT_DIR/"*.safetensors 2>/dev/null || true
}

# ── Single model training ─────────────────────────────────────────────────

train_single() {
    local model="$1"
    case "$model" in
        cam|speaker) train_speaker_encoder ;;
        codec) train_codec ;;
        stt) train_stt ;;
        tts) train_tts ;;
        cfm) train_cfm ;;
        *) fail "Unknown model: $model. Options: cam, codec, stt, tts, cfm"; exit 1 ;;
    esac

    if [[ "$SKIP_EXPORT" == false ]]; then
        local ckpt_map=(
            "cam:$CHECKPOINT_DIR/cam/speaker_encoder_best.pt"
            "codec:$CHECKPOINT_DIR/codec/codec_best.pt"
            "stt:$CHECKPOINT_DIR/stt/stt_best.pt"
            "tts:$CHECKPOINT_DIR/tts/tts_best.pt"
            "cfm:$CHECKPOINT_DIR/cfm/cfm_best.pt"
        )
        for entry in "${ckpt_map[@]}"; do
            local m="${entry%%:*}"
            local c="${entry#*:}"
            if [[ "$m" == "$model" ]]; then
                export_model "$m" "$c"
                break
            fi
        done
    fi
}

# ── Full pipeline ──────────────────────────────────────────────────────────

run_full_pipeline() {
    header "Sonata v2 Full Training Pipeline"

    log "Training order:"
    log "  Wave 1 (parallel): Speaker Encoder (CAM++) + Audio Codec"
    log "  Wave 2 (parallel): STT + TTS"
    log "  Wave 3:            CFM Decoder"
    log ""
    log "Estimated total time: ~2 weeks on single L4 GPU"
    log "  (or ~1 week with 2 GPUs running Wave 1/2 in parallel)"
    log ""

    local start_time=$(date +%s)

    # Wave 1: Speaker Encoder + Codec (independent, can run in parallel on 2 GPUs)
    if [[ $START_WAVE -le 1 ]]; then
        header "Wave 1: Foundation Models"

        # On single GPU, train sequentially
        train_speaker_encoder
        if [[ "$SKIP_EXPORT" == false ]]; then
            export_model "cam" "$CHECKPOINT_DIR/cam/speaker_encoder_best.pt"
        fi

        train_codec
        if [[ "$SKIP_EXPORT" == false ]]; then
            export_model "codec" "$CHECKPOINT_DIR/codec/codec_best.pt"
        fi
    fi

    # Wave 2: STT + TTS (depend on Wave 1)
    if [[ $START_WAVE -le 2 ]]; then
        header "Wave 2: Language Models"

        train_stt
        if [[ "$SKIP_EXPORT" == false ]]; then
            export_model "stt" "$CHECKPOINT_DIR/stt/stt_best.pt"
        fi

        train_tts
        if [[ "$SKIP_EXPORT" == false ]]; then
            export_model "tts" "$CHECKPOINT_DIR/tts/tts_best.pt"
        fi
    fi

    # Wave 3: CFM (depends on speaker encoder)
    if [[ $START_WAVE -le 3 ]]; then
        header "Wave 3: Generative Decoder"

        train_cfm
        if [[ "$SKIP_EXPORT" == false ]]; then
            export_model "cfm" "$CHECKPOINT_DIR/cfm/cfm_best.pt"
        fi
    fi

    local end_time=$(date +%s)
    local duration=$(( (end_time - start_time) / 3600 ))

    header "Training Pipeline Complete!"
    log "Total time: ${duration} hours"
    log "Checkpoints: $CHECKPOINT_DIR/"
    log "Exports: $EXPORT_DIR/"

    # Final summary
    echo ""
    echo "Model Status:"
    for model in cam codec stt tts cfm; do
        local state=$(check_state "$model")
        local emoji="❌"
        [[ "$state" == "complete" ]] && emoji="✅"
        [[ "$state" == "training" ]] && emoji="🔄"
        echo "  $emoji $model: $state"
    done
}

# ── Main ───────────────────────────────────────────────────────────────────

main() {
    check_prerequisites

    if [[ "$EXPORT_ONLY" == true ]]; then
        export_all
        exit 0
    fi

    if [[ -n "$SINGLE_MODEL" ]]; then
        train_single "$SINGLE_MODEL"
    else
        run_full_pipeline
    fi

    success "Done!"
}

main
