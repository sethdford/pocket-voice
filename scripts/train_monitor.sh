#!/bin/bash
# Automated training monitor for Sonata models.
# Checks training status every 60 seconds and auto-starts next jobs.
#
# Usage: bash scripts/train_monitor.sh

set -euo pipefail
cd "$(dirname "$0")/.."

VENV=".venv/bin/python3"
CKPT_DIR="train/checkpoints"

echo "╔══════════════════════════════════════════════════╗"
echo "║       SONATA TRAINING MONITOR                    ║"
echo "╚══════════════════════════════════════════════════╝"
echo ""

check_process() {
    pgrep -f "$1" > /dev/null 2>&1
}

wait_for_finish() {
    local name="$1"
    local pattern="$2"
    echo "[monitor] Waiting for $name to finish..."
    while check_process "$pattern"; do
        sleep 60
        echo -n "."
    done
    echo ""
    echo "[monitor] $name finished!"
}

# Step 1: Wait for LM to finish
if check_process "train_lm.py"; then
    echo "[monitor] LM training is running..."
    wait_for_finish "LM training" "train_lm.py"
fi

# Step 2: When LM done, start CTC v2 on CPU
echo "[monitor] Starting CTC v2 training (with noise augmentation)..."
PYTHONUNBUFFERED=1 $VENV -u train/sonata/train_stt.py \
    --mode ctc \
    --audio-dir train/data/LibriSpeech/train-clean-360 \
    --codec-ckpt $CKPT_DIR/codec_v3/sonata_codec_step_20000.pt \
    --output-dir $CKPT_DIR/stt_v2 \
    --resume $CKPT_DIR/stt/ctc/sonata_ctc_final.pt \
    --device cpu \
    --encoder-size base \
    --unfreeze \
    --ctc-steps 100000 \
    --batch-size 4 \
    --lr 3e-4 \
    --warmup 1000 \
    --log-every 100 \
    --save-every 10000 \
    --num-workers 4 &
CTC_PID=$!
echo "[monitor] CTC v2 PID: $CTC_PID"

# Step 3: Wait for Flow to finish, then start codec retrain on GPU
if check_process "train_flow.py"; then
    echo "[monitor] Waiting for Flow training to finish..."
    wait_for_finish "Flow training" "train_flow.py"
fi

echo "[monitor] Starting Codec retrain (with NaN safeguards)..."
PYTHONUNBUFFERED=1 $VENV -u train/sonata/train_codec.py \
    --resume $CKPT_DIR/codec_v3/sonata_codec_step_20000.pt \
    --checkpoint-dir $CKPT_DIR/codec_v4 \
    --audio-dir train/data/LibriSpeech/train-clean-360 \
    --steps 100000 \
    --batch-size 8 \
    --lr 2e-4 \
    --adv-weight 0.5 \
    --fm-weight 1.0 \
    --clip-grad 0.3 \
    --device mps \
    --save-every 5000 \
    --log-every 50 &
CODEC_PID=$!
echo "[monitor] Codec retrain PID: $CODEC_PID"

echo ""
echo "[monitor] All jobs launched."
echo "  CTC v2: PID $CTC_PID (CPU)"
echo "  Codec:  PID $CODEC_PID (MPS)"
echo ""
echo "[monitor] Monitoring..."

wait $CTC_PID $CODEC_PID
echo "[monitor] All training complete!"
