#!/bin/bash
# train_all.sh — Unified training pipeline for Sonata TTS with prosody conditioning.
#
# Runs the complete training sequence:
#   1. Codec → encode dataset with prosody features
#   2. Semantic LM (optionally with prosody conditioning)
#   3. Flow network (with emotion + prosody + duration conditioning)
#   4. EmoSteer direction vectors (training-free emotion control)
#   5. Flow distillation (8→1 step for real-time inference)
#
# Usage:
#   # Full pipeline from scratch
#   bash train_all.sh --data-dir data/librispeech/ --emotion-dir data/emov-db/
#
#   # Resume from encoded data
#   bash train_all.sh --skip-encode --encoded-data data/encoded.pt
#
#   # LM + Flow only (codec already trained)
#   bash train_all.sh --skip-codec --codec-ckpt checkpoints/codec_final.pt

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

log() { echo "[$(date +%H:%M:%S)] $*"; }
die() { echo "ERROR: $*" >&2; exit 1; }

check_deps() {
    python3 -c "import torch" 2>/dev/null || die "PyTorch not found. pip install torch"
    python3 -c "import safetensors" 2>/dev/null || die "safetensors not found. pip install safetensors"
    python3 -c "import numpy" 2>/dev/null || die "numpy not found. pip install numpy"
}

# ── Data Acquisition ──────────────────────────────────────────────────────────
# All dataset operations now go through acquire_data.py for consistency.

acquire() {
    python acquire_data.py "$@"
}

download_dataset() {
    local dataset="$1"
    local splits="${2:-all}"
    local dest="${3:-data}"
    acquire download --dataset "$dataset" --splits "$splits" --out-dir "$dest"
}

build_manifest() {
    local data_dir="$1"
    local output="$2"
    acquire manifest --data-dir "$data_dir" --output "$output"
}

preprocess_manifest() {
    local manifest="$1"
    local output="$2"
    local resample_dir="${3:-}"
    local extra_flags=""
    if [[ -n "$resample_dir" ]]; then
        extra_flags="--resample-dir $resample_dir"
    fi
    acquire preprocess --manifest "$manifest" --output "$output" \
        --target-sr 24000 --min-duration 0.5 --max-duration 30.0 \
        --min-snr 10 $extra_flags
}

# ── Defaults ──────────────────────────────────────────────────────────────────
DATA_DIR=""
EMOTION_DIR=""
ENCODED_DATA=""
CODEC_CKPT="checkpoints/codec/sonata_codec_final.pt"
LM_CKPT=""
FLOW_CKPT=""
OUTPUT_DIR="checkpoints"
DEVICE="mps"
EPOCHS_CODEC=100
EPOCHS_LM=50
EPOCHS_FLOW=50
BATCH_SIZE=8
LR="3e-4"
SKIP_CODEC=0
SKIP_ENCODE=0
SKIP_LM=0
SKIP_FLOW=0
USE_EMOTION=1
USE_PROSODY=1
USE_DURATION=1
EMOSTEER=1
DISTILL=0
SOUNDSTORM=0
JOINT=0
CONSISTENCY=0

# ── Parse arguments ──────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case $1 in
        --data-dir) DATA_DIR="$2"; shift 2;;
        --emotion-dir) EMOTION_DIR="$2"; shift 2;;
        --encoded-data) ENCODED_DATA="$2"; shift 2;;
        --codec-ckpt) CODEC_CKPT="$2"; shift 2;;
        --lm-ckpt) LM_CKPT="$2"; shift 2;;
        --flow-ckpt) FLOW_CKPT="$2"; shift 2;;
        --output-dir) OUTPUT_DIR="$2"; shift 2;;
        --device) DEVICE="$2"; shift 2;;
        --epochs-codec) EPOCHS_CODEC="$2"; shift 2;;
        --epochs-lm) EPOCHS_LM="$2"; shift 2;;
        --epochs-flow) EPOCHS_FLOW="$2"; shift 2;;
        --batch-size) BATCH_SIZE="$2"; shift 2;;
        --lr) LR="$2"; shift 2;;
        --skip-codec) SKIP_CODEC=1; shift;;
        --skip-encode) SKIP_ENCODE=1; shift;;
        --skip-lm) SKIP_LM=1; shift;;
        --skip-flow) SKIP_FLOW=1; shift;;
        --no-emotion) USE_EMOTION=0; shift;;
        --no-prosody) USE_PROSODY=0; shift;;
        --no-duration) USE_DURATION=0; shift;;
        --no-emosteer) EMOSTEER=0; shift;;
        --distill) DISTILL=1; shift;;
        --consistency) CONSISTENCY=1; DISTILL=1; shift;;
        --soundstorm) SOUNDSTORM=1; shift;;
        --joint) JOINT=1; shift;;
        --download) DOWNLOAD_DATASET="${2:-libritts-r}"; shift 2;;
        --download-all) DOWNLOAD_ALL=1; shift;;
        --manifest-only) MANIFEST_ONLY=1; shift;;
        --export) EXPORT=1; shift;;
        --resume) RESUME=1; shift;;
        --dry-run) DRY_RUN=1; shift;;
        --skip-v3) SKIP_V3=1; shift;;
        --joint-v3) JOINT_V3=1; shift;;
        --distill-v3) DISTILL_V3=1; shift;;
        --v3-data-dir) V3_DATA_DIR="$2"; shift 2;;
        *) echo "Unknown: $1"; exit 1;;
    esac
done

EXPORT="${EXPORT:-0}"
RESUME="${RESUME:-0}"
DRY_RUN="${DRY_RUN:-0}"
DOWNLOAD_DATASET="${DOWNLOAD_DATASET:-}"
DOWNLOAD_ALL="${DOWNLOAD_ALL:-0}"
MANIFEST_ONLY="${MANIFEST_ONLY:-0}"

check_deps
mkdir -p "$OUTPUT_DIR"/{codec,lm,flow,emosteer}

echo "╔══════════════════════════════════════════════════════════╗"
echo "║  SONATA TTS — UNIFIED TRAINING PIPELINE                 ║"
echo "╠══════════════════════════════════════════════════════════╣"
echo "║  Device:    $DEVICE"
echo "║  Emotion:   $([ $USE_EMOTION -eq 1 ] && echo ON || echo OFF)"
echo "║  Prosody:   $([ $USE_PROSODY -eq 1 ] && echo ON || echo OFF)"
echo "║  Duration:  $([ $USE_DURATION -eq 1 ] && echo ON || echo OFF)"
echo "║  EmoSteer:  $([ $EMOSTEER -eq 1 ] && echo ON || echo OFF)"
echo "║  Distill:   $([ $DISTILL -eq 1 ] && echo ON || echo OFF)"
echo "╚══════════════════════════════════════════════════════════╝"

# ── Step 0: Data Acquisition (optional) ──────────────────────────────────────
if [[ $DOWNLOAD_ALL -eq 1 ]]; then
    echo ""
    echo "━━━ Step 0: Download All Datasets ━━━"
    download_dataset libritts-r all "data"
    download_dataset vctk all "data"
    download_dataset ljspeech all "data"
    download_dataset hifi-tts all "data"
    echo ""
    echo "  Building unified manifest..."
    build_manifest "data" "data/manifest_raw.jsonl"
    echo "  Preprocessing (resample to 24kHz, SNR filter)..."
    preprocess_manifest "data/manifest_raw.jsonl" "data/manifest_clean.jsonl" "data/resampled"
    acquire stats --manifest "data/manifest_clean.jsonl"
    DATA_DIR="data"
elif [[ -n "$DOWNLOAD_DATASET" ]]; then
    echo ""
    echo "━━━ Step 0: Download $DOWNLOAD_DATASET ━━━"
    download_dataset "$DOWNLOAD_DATASET" all "data"
    build_manifest "data" "data/manifest_raw.jsonl"
    preprocess_manifest "data/manifest_raw.jsonl" "data/manifest_clean.jsonl" "data/resampled"
    acquire stats --manifest "data/manifest_clean.jsonl"
    DATA_DIR="data"
fi

if [[ $MANIFEST_ONLY -eq 1 ]]; then
    log "Manifest-only mode: exiting."
    exit 0
fi

# ── Step 1: Train Codec ──────────────────────────────────────────────────────
if [[ $SKIP_CODEC -eq 0 && -n "$DATA_DIR" ]]; then
    echo ""
    echo "━━━ Step 1/5: Train Codec ━━━"
    MANIFEST="$DATA_DIR/manifest_clean.jsonl"
    CODEC_FLAGS=""
    if [[ -f "$MANIFEST" ]]; then
        CODEC_FLAGS="--manifest $MANIFEST"
    else
        CODEC_FLAGS="--audio-dir $DATA_DIR"
    fi
    python train_codec.py \
        $CODEC_FLAGS \
        --checkpoint-dir "$OUTPUT_DIR/codec" \
        --steps "$EPOCHS_CODEC" \
        --batch-size "$BATCH_SIZE" \
        --grad-accum 4 \
        --lr "$LR" \
        --device "$DEVICE"
    CODEC_CKPT="$OUTPUT_DIR/codec/sonata_codec_best.pt"
    if [[ ! -f "$CODEC_CKPT" ]]; then
        CODEC_CKPT="$OUTPUT_DIR/codec/sonata_codec_final.pt"
    fi
    echo "  Codec saved to $CODEC_CKPT"
else
    echo ""
    echo "━━━ Step 1/5: Skip Codec (using $CODEC_CKPT) ━━━"
fi

# ── Step 2: Encode Dataset ───────────────────────────────────────────────────
if [[ $SKIP_ENCODE -eq 0 && -n "$DATA_DIR" ]]; then
    echo ""
    echo "━━━ Step 2/5: Encode Dataset with Prosody ━━━"

    # Use preprocessed manifest if available, otherwise build one
    MANIFEST="$DATA_DIR/manifest_clean.jsonl"
    if [[ ! -f "$MANIFEST" ]]; then
        MANIFEST="$OUTPUT_DIR/manifest.jsonl"
        build_manifest "$DATA_DIR" "$MANIFEST"
    fi

    ENCODED_DATA="$OUTPUT_DIR/encoded_data.pt"
    python data_pipeline.py \
        --source manifest \
        --manifest "$MANIFEST" \
        --codec-ckpt "$CODEC_CKPT" \
        --output "$ENCODED_DATA" \
        --shard-size 10000 \
        --device "$DEVICE"
    echo "  ✓ Encoded data saved to $ENCODED_DATA"

    # Also prepare emotion-labeled data if available
    if [[ -n "$EMOTION_DIR" ]]; then
        echo "  Preparing emotion-labeled prosody data..."
        python prepare_prosody_data.py \
            --dataset emov-db \
            --data-dir "$EMOTION_DIR" \
            --output "$OUTPUT_DIR/prosody_data/"
        echo "  ✓ Prosody data in $OUTPUT_DIR/prosody_data/"
    fi
else
    echo ""
    echo "━━━ Step 2/5: Skip Encode (using $ENCODED_DATA) ━━━"
fi

# ── Step 3: Train Semantic LM ────────────────────────────────────────────────
if [[ $SKIP_LM -eq 0 && -n "$ENCODED_DATA" ]]; then
    echo ""
    echo "━━━ Step 3/5: Train Semantic LM ━━━"
    LM_FLAGS=""
    if [[ $USE_PROSODY -eq 1 ]]; then
        LM_FLAGS="$LM_FLAGS --use-prosody"
    fi
    # Detect sharded data
    SHARDS_INDEX="${ENCODED_DATA%.pt}.shards.txt"
    if [[ -f "$SHARDS_INDEX" ]]; then
        LM_DATA="$SHARDS_INDEX"
    else
        LM_DATA="$ENCODED_DATA"
    fi
    python train_lm.py \
        --data "$LM_DATA" \
        --checkpoint-dir "$OUTPUT_DIR/lm" \
        --steps "$EPOCHS_LM" \
        --batch-size "$BATCH_SIZE" \
        --grad-accum 4 \
        --label-smoothing 0.1 \
        --device "$DEVICE" \
        $LM_FLAGS
    LM_CKPT="$OUTPUT_DIR/lm/sonata_lm_best.pt"
    if [[ ! -f "$LM_CKPT" ]]; then
        LM_CKPT="$OUTPUT_DIR/lm/sonata_lm_final.pt"
    fi
    echo "  LM saved to $LM_CKPT"
else
    echo ""
    echo "━━━ Step 3/5: Skip LM ━━━"
fi

# ── Step 4: Train Flow Network ───────────────────────────────────────────────
if [[ $SKIP_FLOW -eq 0 && -n "$ENCODED_DATA" ]]; then
    echo ""
    echo "━━━ Step 4/5: Train Flow Network ━━━"
    FLOW_FLAGS=""
    if [[ $USE_EMOTION -eq 1 ]]; then FLOW_FLAGS="$FLOW_FLAGS --use-emotion"; fi
    if [[ $USE_PROSODY -eq 1 ]]; then FLOW_FLAGS="$FLOW_FLAGS --use-prosody"; fi
    if [[ $USE_DURATION -eq 1 ]]; then FLOW_FLAGS="$FLOW_FLAGS --use-duration"; fi

    # Support sharded data via --manifest
    SHARDS_INDEX="${ENCODED_DATA%.pt}.shards.txt"
    if [[ -f "$SHARDS_INDEX" ]]; then
        FLOW_FLAGS="$FLOW_FLAGS --manifest $SHARDS_INDEX"
    else
        FLOW_FLAGS="$FLOW_FLAGS --manifest $ENCODED_DATA"
    fi

    python train_flow.py \
        --data-dir "$ENCODED_DATA" \
        --output-dir "$OUTPUT_DIR/flow" \
        --steps "$EPOCHS_FLOW" \
        --batch-size "$BATCH_SIZE" \
        --grad-accum 4 \
        --device "$DEVICE" \
        $FLOW_FLAGS
    FLOW_CKPT="$OUTPUT_DIR/flow/flow_best.pt"
    if [[ ! -f "$FLOW_CKPT" ]]; then
        FLOW_CKPT="$OUTPUT_DIR/flow/flow_final.pt"
    fi
    echo "  Flow saved to $FLOW_CKPT"

    # Optional: Flow distillation (8→1 step)
    if [[ $DISTILL -eq 1 && -n "$FLOW_CKPT" ]]; then
        DISTILL_DATA="${ENCODED_DATA%.pt}.shards.txt"
        [[ -f "$DISTILL_DATA" ]] || DISTILL_DATA="$ENCODED_DATA"
        DISTILL_FLAGS=""
        if [[ $CONSISTENCY -eq 1 ]]; then
            echo "  Running consistency distillation (direct 1-step)..."
            DISTILL_FLAGS="--consistency --consistency-steps 50000"
        else
            echo "  Running progressive distillation (8→4→2→1)..."
        fi
        python train_flow_distill.py \
            --teacher "$FLOW_CKPT" \
            --teacher-config "$OUTPUT_DIR/flow/flow_config.json" \
            --data-dir "$DISTILL_DATA" \
            --output-dir "$OUTPUT_DIR/flow_distill" \
            --device "$DEVICE" \
            $DISTILL_FLAGS
        echo "  ✓ Distilled flow saved to $OUTPUT_DIR/flow_distill/"
    fi
else
    echo ""
    echo "━━━ Step 4/5: Skip Flow ━━━"
fi

# ── Step 5: Compute EmoSteer Directions ──────────────────────────────────────
if [[ $EMOSTEER -eq 1 ]]; then
    echo ""
    echo "━━━ Step 5/5: Compute EmoSteer Directions ━━━"

    EMOSTEER_DATA="$OUTPUT_DIR/prosody_data/"
    EMOSTEER_FLAGS=""
    if [[ -d "$EMOSTEER_DATA" ]]; then
        MANIFEST_FILE="$EMOSTEER_DATA/manifest.jsonl"
        if [[ -f "$MANIFEST_FILE" ]]; then
            EMOSTEER_FLAGS="--from-manifest $MANIFEST_FILE"
        else
            EMOSTEER_FLAGS="--data-dir $EMOSTEER_DATA"
        fi
    else
        echo "  No emotion data available. Using synthetic directions."
        EMOSTEER_FLAGS="--synthetic"
    fi

    FLOW_W="${FLOW_CKPT:-$OUTPUT_DIR/flow/flow_final.pt}"
    FLOW_C="$OUTPUT_DIR/flow/flow_config.json"

    python compute_emosteer.py \
        $EMOSTEER_FLAGS \
        --flow-weights "$FLOW_W" \
        --flow-config "$FLOW_C" \
        --output "$OUTPUT_DIR/emosteer/emosteer_directions.json" \
        --device "$DEVICE"
    echo "  ✓ EmoSteer saved to $OUTPUT_DIR/emosteer/emosteer_directions.json"
else
    echo ""
    echo "━━━ Step 5/5: Skip EmoSteer ━━━"
fi

# ── Step 6 (optional): Export to safetensors for Rust/C inference ─────────────
if [[ $EXPORT -eq 1 ]]; then
    echo ""
    echo "━━━ Step 6: Export Weights to Safetensors ━━━"
    EXPORT_FLAGS=""
    if [[ -n "${CODEC_CKPT:-}" && -f "$CODEC_CKPT" ]]; then
        EXPORT_FLAGS="$EXPORT_FLAGS --codec-ckpt $CODEC_CKPT"
    fi
    if [[ -n "${LM_CKPT:-}" && -f "$LM_CKPT" ]]; then
        EXPORT_FLAGS="$EXPORT_FLAGS --lm-ckpt $LM_CKPT"
    fi
    if [[ -n "${FLOW_CKPT:-}" && -f "$FLOW_CKPT" ]]; then
        EXPORT_FLAGS="$EXPORT_FLAGS --flow-ckpt $FLOW_CKPT"
    fi

    if [[ -n "$EXPORT_FLAGS" ]]; then
        python "$SCRIPT_DIR/export_weights.py" \
            $EXPORT_FLAGS \
            --output-dir "$OUTPUT_DIR/exported"
        log "Exported weights to $OUTPUT_DIR/exported/"
    else
        log "No checkpoints to export."
    fi
fi

# ── Step 7 (optional): SoundStorm parallel decoder ───────────────────────────
if [[ $SOUNDSTORM -eq 1 ]]; then
    echo ""
    echo "━━━ Step 7: SoundStorm Parallel Decoder ━━━"
    STORM_DATA="${ENCODED_DATA:-$OUTPUT_DIR/encoded_data.pt}"
    STORM_DATA_ARG="${STORM_DATA%.pt}.shards.txt"
    [[ -f "$STORM_DATA_ARG" ]] || STORM_DATA_ARG="$STORM_DATA"

    python train_soundstorm.py \
        --data "$STORM_DATA_ARG" \
        --checkpoint-dir "$OUTPUT_DIR/soundstorm" \
        --device "$DEVICE" \
        --steps "$EPOCHS_LM" \
        --batch-size "$BATCH_SIZE" \
        --lr "$LR"
    echo "  ✓ SoundStorm saved to $OUTPUT_DIR/soundstorm/"
else
    echo ""
    echo "━━━ Step 7: Skip SoundStorm ━━━"
fi

# ── Step 8 (optional): Joint end-to-end fine-tuning ──────────────────────────
if [[ $JOINT -eq 1 ]]; then
    echo ""
    echo "━━━ Step 8: Joint Fine-Tuning (LM + Flow + Decoder) ━━━"
    JOINT_FLAGS=""
    [[ -n "${CODEC_CKPT:-}" && -f "$CODEC_CKPT" ]] && JOINT_FLAGS="$JOINT_FLAGS --codec-ckpt $CODEC_CKPT"
    [[ -n "${LM_CKPT:-}" && -f "$LM_CKPT" ]] && JOINT_FLAGS="$JOINT_FLAGS --lm-ckpt $LM_CKPT"
    [[ -n "${FLOW_CKPT:-}" && -f "$FLOW_CKPT" ]] && JOINT_FLAGS="$JOINT_FLAGS --flow-ckpt $FLOW_CKPT"

    JOINT_DATA="${ENCODED_DATA:-$OUTPUT_DIR/encoded_data.pt}"
    JOINT_MANIFEST="${JOINT_DATA%.pt}.shards.txt"
    [[ -f "$JOINT_MANIFEST" ]] || JOINT_MANIFEST="$OUTPUT_DIR/manifest_clean.jsonl"
    python train_joint.py \
        $JOINT_FLAGS \
        --manifest "$JOINT_MANIFEST" \
        --output-dir "$OUTPUT_DIR/joint" \
        --device "$DEVICE" \
        --steps 20000 \
        --batch-size 4 \
        --lr 1e-5
    echo "  ✓ Joint model saved to $OUTPUT_DIR/joint/"
else
    echo ""
    echo "━━━ Step 8: Skip Joint Fine-Tuning ━━━"
fi

# ═══════════════════════════════════════════════════════════════
# FLOW V3 PIPELINE (text → mel → audio)
# ═══════════════════════════════════════════════════════════════
# Phase V3-1: Vocoder training (mel → waveform)
# Phase V3-2: Flow v3 training (text → mel)
# Phase V3-3: Joint fine-tuning (optional)
# Phase V3-4: Distillation (optional, 8→1 step)
# Phase V3-5: Export to safetensors

SKIP_V3="${SKIP_V3:-0}"
V3_DATA_DIR="${V3_DATA_DIR:-$DATA_DIR}"
V3_MANIFEST="${MANIFEST:-$DATA_DIR/manifest_clean.jsonl}"
FLOW_V3_CKPT=""
VOCODER_CKPT=""

# Phase V3-1: Vocoder training (mel → waveform)
if [[ $SKIP_V3 -eq 0 && -n "$V3_DATA_DIR" ]]; then
    echo ""
    echo "━━━ V3-1: Train Vocoder (mel → waveform) ━━━"
    VOCODER_OUT="$OUTPUT_DIR/vocoder"
    mkdir -p "$VOCODER_OUT"
    VOCODER_FLAGS="--data-dir $V3_DATA_DIR --output-dir $VOCODER_OUT --device $DEVICE"
    if [[ $RESUME -eq 1 ]]; then
        LATEST_VOC=$(ls -t "$VOCODER_OUT"/vocoder_epoch*.pt 2>/dev/null | head -1)
        [[ -n "$LATEST_VOC" ]] && VOCODER_FLAGS="$VOCODER_FLAGS --resume $LATEST_VOC"
    fi
    python train_vocoder.py $VOCODER_FLAGS \
        --epochs 200 --batch-size 16 --lr 2e-4
    VOCODER_CKPT="$VOCODER_OUT/vocoder_generator.pt"
    [[ -f "$VOCODER_CKPT" ]] || VOCODER_CKPT="$VOCODER_OUT/vocoder_epoch200.pt"
    log "Vocoder saved to $VOCODER_CKPT"
else
    echo ""
    echo "━━━ V3-1: Skip Vocoder (SKIP_V3=1 or no V3_DATA_DIR) ━━━"
fi

# Phase V3-2: Flow v3 training (text → mel)
if [[ $SKIP_V3 -eq 0 ]]; then
    echo ""
    echo "━━━ V3-2: Train Flow v3 (text → mel) ━━━"
    FLOW_V3_OUT="$OUTPUT_DIR/flow_v3"
    mkdir -p "$FLOW_V3_OUT"
    FLOW_V3_FLAGS="--output-dir $FLOW_V3_OUT --device $DEVICE --phonemes --model-size large"
    if [[ $RESUME -eq 1 ]]; then
        [[ -f "$FLOW_V3_OUT/flow_v3_best.pt" ]] && FLOW_V3_FLAGS="$FLOW_V3_FLAGS --resume $FLOW_V3_OUT/flow_v3_best.pt"
    fi
    if [[ -n "$V3_MANIFEST" && -f "$V3_MANIFEST" ]]; then
        python train_flow_v3.py --manifest "$V3_MANIFEST" $FLOW_V3_FLAGS \
            --steps 50000 --batch-size 8
    elif [[ -n "$V3_DATA_DIR" && -d "$V3_DATA_DIR" ]]; then
        FLOW_V2_DATA="$OUTPUT_DIR/flow_v2_data"
        if [[ ! -d "$FLOW_V2_DATA" || -z "$(ls -A $FLOW_V2_DATA/*.pt 2>/dev/null)" ]]; then
            log "Preparing flow v2 data for Flow v3..."
            python prepare_flow_v2_data.py --audio-dir "$V3_DATA_DIR" \
                --output-dir "$FLOW_V2_DATA" --sample-rate 24000 --n-mels 80
        fi
        python train_flow_v3.py --data-dir "$FLOW_V2_DATA" $FLOW_V3_FLAGS \
            --steps 50000 --batch-size 8
    else
        echo "  [SKIP] No manifest or data-dir for Flow v3"
    fi
    FLOW_V3_CKPT="$FLOW_V3_OUT/flow_v3_best.pt"
    [[ -f "$FLOW_V3_CKPT" ]] || FLOW_V3_CKPT="$FLOW_V3_OUT/flow_v3_final.pt"
    log "Flow v3 saved to $FLOW_V3_CKPT"
else
    echo ""
    echo "━━━ V3-2: Skip Flow v3 ━━━"
fi

# Phase V3-3: Joint fine-tuning (optional)
if [[ $SKIP_V3 -eq 0 && -n "${FLOW_V3_CKPT:-}" && -f "${FLOW_V3_CKPT:-}" && -n "${VOCODER_CKPT:-}" && -f "${VOCODER_CKPT:-}" && "${JOINT_V3:-0}" -eq 1 ]]; then
    echo ""
    echo "━━━ V3-3: Joint Fine-Tuning (Flow v3 + Vocoder) ━━━"
    JOINT_V3_OUT="$OUTPUT_DIR/joint_v3"
    mkdir -p "$JOINT_V3_OUT"
    JOINT_V3_MANIFEST="${V3_MANIFEST:-$DATA_DIR/manifest_clean.jsonl}"
    [[ -f "$JOINT_V3_MANIFEST" ]] || JOINT_V3_MANIFEST="$OUTPUT_DIR/manifest_clean.jsonl"
    python train_joint_v3.py \
        --flow-ckpt "$FLOW_V3_CKPT" \
        --vocoder-ckpt "$VOCODER_CKPT" \
        --manifest "$JOINT_V3_MANIFEST" \
        --output-dir "$JOINT_V3_OUT" \
        --device "$DEVICE"
    log "Joint v3 saved to $JOINT_V3_OUT/"
else
    echo ""
    echo "━━━ V3-3: Skip Joint v3 (use --joint-v3 to enable) ━━━"
fi

# Phase V3-4: Distillation (optional, 8→1 step)
if [[ $SKIP_V3 -eq 0 && -n "${FLOW_V3_CKPT:-}" && -f "${FLOW_V3_CKPT:-}" && "${DISTILL_V3:-0}" -eq 1 ]]; then
    echo ""
    echo "━━━ V3-4: Distillation (8→1 step) ━━━"
    DISTILL_V3_OUT="$OUTPUT_DIR/flow_v3_distilled"
    mkdir -p "$DISTILL_V3_OUT"
    DISTILL_MANIFEST="${V3_MANIFEST:-$DATA_DIR/manifest_clean.jsonl}"
    [[ -f "$DISTILL_MANIFEST" ]] || DISTILL_MANIFEST="$OUTPUT_DIR/manifest_clean.jsonl"
    python train_distill_v3.py \
        --teacher-checkpoint "$FLOW_V3_CKPT" \
        --manifest "$DISTILL_MANIFEST" \
        --output-dir "$DISTILL_V3_OUT" \
        --device "$DEVICE" \
        --phonemes --model-size large
    log "Distilled Flow v3 saved to $DISTILL_V3_OUT/"
else
    echo ""
    echo "━━━ V3-4: Skip Distillation (use --distill-v3 to enable) ━━━"
fi

# Phase V3-5: Export to safetensors
if [[ $EXPORT -eq 1 && -n "${FLOW_V3_CKPT:-}" && -f "${FLOW_V3_CKPT:-}" ]]; then
    echo ""
    echo "━━━ V3-5: Export Flow v3 to Safetensors ━━━"
    FLOW_V3_CONFIG="$OUTPUT_DIR/flow_v3/flow_config.json"
    [[ -f "$FLOW_V3_CONFIG" ]] || FLOW_V3_CONFIG=""
    python "$SCRIPT_DIR/export_weights.py" \
        --causal-flow-ckpt "$FLOW_V3_CKPT" \
        ${FLOW_V3_CONFIG:+--causal-flow-config "$FLOW_V3_CONFIG"} \
        --output-dir "$OUTPUT_DIR/exported"
    log "Flow v3 exported to $OUTPUT_DIR/exported/"
fi

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║  TRAINING COMPLETE                                       ║"
echo "╠══════════════════════════════════════════════════════════╣"
echo "║  Run with:                                               ║"
echo "║    ./pocket-voice --tts-engine sonata \\                  ║"
echo "║      --sonata-lm-weights $OUTPUT_DIR/lm/*.safetensors \\ ║"
echo "║      --sonata-flow-weights $OUTPUT_DIR/flow/*.safetensors║"
if [[ $EMOSTEER -eq 1 ]]; then
echo "║      --emosteer $OUTPUT_DIR/emosteer/emosteer_directions.json ║"
fi
if [[ $EXPORT -eq 1 ]]; then
echo "║                                                          ║"
echo "║  Exported safetensors:                                   ║"
echo "║    $OUTPUT_DIR/exported/                                  ║"
fi
echo "╚══════════════════════════════════════════════════════════╝"
