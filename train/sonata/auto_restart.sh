#!/bin/bash
# Auto-restart codec training with bug fixes at step 20000 (before GAN kicks in)
# Monitors checkpoint directory since terminal logs may be buffered.
set -e

TRAIN_PID=86223
CKPT_DIR="/Users/sethford/Documents/pocket-voice/train/checkpoints/codec"
TARGET_CKPT="$CKPT_DIR/sonata_codec_step_20000.pt"

echo "=== Codec Training Auto-Restart ==="
echo "Watching PID $TRAIN_PID for checkpoint: $TARGET_CKPT"
echo "Checkpoints save every 5000 steps. Current: step_5000 exists."
echo "Expected checkpoints: step_10000, step_15000, step_20000"
echo "At ~7.4 steps/s: ~27 min from step 8000 to step 20000"
echo "Checking every 60 seconds..."
echo ""

while true; do
    # Check if process is still running
    if ! kill -0 "$TRAIN_PID" 2>/dev/null; then
        echo "[$(date)] Training process $TRAIN_PID died!"
        ls -la "$CKPT_DIR/"
        echo "You'll need to restart manually."
        exit 1
    fi

    # Check which checkpoints exist
    LATEST_STEP=0
    for f in "$CKPT_DIR"/sonata_codec_step_*.pt; do
        [ -f "$f" ] || continue
        STEP=$(echo "$f" | grep -oE '[0-9]+\.pt' | grep -oE '^[0-9]+')
        if [ "$STEP" -gt "$LATEST_STEP" ]; then
            LATEST_STEP=$STEP
        fi
    done

    echo "[$(date)] Latest checkpoint: step_${LATEST_STEP} | Process CPU: $(ps -p $TRAIN_PID -o %cpu= 2>/dev/null || echo 'N/A')%"

    if [ -f "$TARGET_CKPT" ]; then
        echo ""
        echo "=== step_20000 checkpoint found! ==="
        echo "Waiting 15s for checkpoint write to fully flush..."
        sleep 15

        echo "Killing training process $TRAIN_PID..."
        kill "$TRAIN_PID" 2>/dev/null || true
        sleep 5

        if kill -0 "$TRAIN_PID" 2>/dev/null; then
            echo "Process still alive, sending SIGKILL..."
            kill -9 "$TRAIN_PID" 2>/dev/null || true
            sleep 3
        fi
        echo "Old process terminated."
        echo ""

        echo "=== Restarting with bug-fixed code ==="
        echo "Key fixes in the new code:"
        echo "  1. exp() clamped to max=15.0 to prevent overflow"
        echo "  2. sc_loss denominator clamp raised 1e-4 → 1e-1"
        echo "  3. Progressive training works with num_workers>0"
        echo "  4. best_val_loss persisted in checkpoints"
        echo "  5. LR warmup non-zero at step 0"
        echo ""
        echo "Resuming from: $TARGET_CKPT"
        echo "GAN training will be ACTIVE from step 20000 onward."
        echo ""

        cd /Users/sethford/Documents/pocket-voice/train/sonata
        exec python3 train_codec.py \
            --manifest ../data/manifest_v2.jsonl \
            --checkpoint-dir ../checkpoints/codec \
            --resume "$TARGET_CKPT" \
            --device mps \
            --steps 200000 \
            --batch-size 8 \
            --grad-accum 4 \
            --lr 1e-4 \
            --warmup 1000 \
            --segment-sec 2.0 \
            --decoder-type istft \
            --gan-start-step 20000 \
            --adv-weight 0.5 \
            --fm-weight 1.0 \
            --ema-decay 0.999 \
            --progressive \
            --augment \
            --save-every 5000 \
            --val-every 2000 \
            --log-every 100 \
            --clip-grad 0.5 \
            --r1-weight 10.0
    fi

    sleep 60
done
