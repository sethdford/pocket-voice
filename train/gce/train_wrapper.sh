#!/bin/bash
# train_wrapper.sh — Runs on the VM: auto-resume training, sync checkpoints to GCS
# Usage: ./train_wrapper.sh <flow_v3|vocoder> <gs://bucket>
set -euo pipefail

JOB="${1:?Usage: train_wrapper.sh <flow_v3|vocoder> <bucket>}"
BUCKET="${2:?Usage: train_wrapper.sh <job> <gs://bucket>}"

cd /opt/sonata/train/sonata

DATA_DIR="../data"
CKPT_DIR="../checkpoints"
DEVICE="cuda"

# ─── Detect preemption ────────────────────────────────────────────────────────
PREEMPTED=false
trap 'echo "[$(date)] SIGTERM received (preemption)"; PREEMPTED=true' SIGTERM

sync_checkpoints() {
    local job_ckpt_dir="$1"
    local gcs_ckpt_dir="$2"
    echo "[$(date)] Syncing checkpoints to GCS..."
    gsutil -m rsync -r "$job_ckpt_dir" "$gcs_ckpt_dir"
    echo "[$(date)] Checkpoint sync complete"
}

# ─── Find latest checkpoint ──────────────────────────────────────────────────
find_latest_checkpoint() {
    local ckpt_dir="$1"
    local pattern="$2"
    ls -t "$ckpt_dir"/$pattern 2>/dev/null | head -1
}

# ─── Training loop with auto-resume ──────────────────────────────────────────
MAX_RETRIES=50  # Enough for many preemptions
RETRY=0

while [ $RETRY -lt $MAX_RETRIES ]; do
    RETRY=$((RETRY + 1))
    echo ""
    echo "=== Training attempt $RETRY/$MAX_RETRIES ==="
    echo "  Job: $JOB"
    echo "  Date: $(date)"

    case "$JOB" in
        flow_v3)
            JOB_CKPT_DIR="$CKPT_DIR/flow_v3_large_fixed"
            GCS_CKPT_DIR="$BUCKET/checkpoints/flow_v3_large_fixed"
            mkdir -p "$JOB_CKPT_DIR"

            # Find latest checkpoint to resume from
            RESUME=$(find_latest_checkpoint "$JOB_CKPT_DIR" "flow_v3_step_*.pt")
            if [ -z "$RESUME" ]; then
                RESUME=$(find_latest_checkpoint "$JOB_CKPT_DIR" "flow_v3_best.pt")
            fi

            RESUME_FLAG=""
            if [ -n "$RESUME" ]; then
                echo "  Resuming from: $RESUME"
                RESUME_FLAG="--resume $RESUME"
            else
                echo "  Starting from scratch (no checkpoint found)"
            fi

            python3 -u train_flow_v3.py \
                --manifest "$DATA_DIR/libritts_r_full_manifest.jsonl" \
                --output-dir "$JOB_CKPT_DIR" \
                --device "$DEVICE" \
                --steps 200000 \
                --batch-size 16 \
                --lr 3e-5 \
                --warmup 2000 \
                --model-size large \
                --phonemes \
                --interleaved \
                --save-every 2500 \
                --num-workers 4 \
                $RESUME_FLAG &
            TRAIN_PID=$!
            ;;

        vocoder)
            JOB_CKPT_DIR="$CKPT_DIR/vocoder_large_fixed"
            GCS_CKPT_DIR="$BUCKET/checkpoints/vocoder_large_fixed"
            mkdir -p "$JOB_CKPT_DIR"

            # Find latest checkpoint
            RESUME=$(find_latest_checkpoint "$JOB_CKPT_DIR" "vocoder_epoch*.pt")
            RESUME_FLAG=""
            if [ -n "$RESUME" ]; then
                echo "  Resuming from: $RESUME"
                RESUME_FLAG="--resume $RESUME"
            else
                echo "  Starting from scratch"
            fi

            python3 -u train_vocoder.py \
                --data-dir "$DATA_DIR/libritts-r/LibriTTS_R" \
                --output-dir "$JOB_CKPT_DIR" \
                --device "$DEVICE" \
                --epochs 200 \
                --batch-size 32 \
                --save-interval 5 \
                --log-interval 50 \
                --num-workers 4 \
                $RESUME_FLAG &
            TRAIN_PID=$!
            ;;

        speaker_encoder)
            JOB_CKPT_DIR="$CKPT_DIR/speaker_encoder"
            GCS_CKPT_DIR="$BUCKET/checkpoints/speaker_encoder"
            mkdir -p "$JOB_CKPT_DIR"

            echo "  Starting speaker encoder training"

            python3 -u train_speaker_encoder.py \
                --data-dir "$DATA_DIR/libritts-r/LibriTTS_R" \
                --output-dir "$JOB_CKPT_DIR" \
                --device "$DEVICE" \
                --batch-size 32 \
                --n-epochs 100 \
                --lr 0.001 &
            TRAIN_PID=$!
            ;;

        codec_12hz)
            JOB_CKPT_DIR="$CKPT_DIR/codec_12hz"
            GCS_CKPT_DIR="$BUCKET/checkpoints/codec_12hz"
            mkdir -p "$JOB_CKPT_DIR"

            RESUME=$(find_latest_checkpoint "$JOB_CKPT_DIR" "sonata_codec_step_*.pt")
            if [ -z "$RESUME" ]; then
                RESUME=$(find_latest_checkpoint "$JOB_CKPT_DIR" "sonata_codec_best.pt")
            fi

            RESUME_FLAG=""
            if [ -n "$RESUME" ]; then
                echo "  Resuming from: $RESUME"
                RESUME_FLAG="--resume $RESUME"
            else
                echo "  Starting from scratch"
            fi

            python3 -u train_codec.py \
                --codec-version 12hz \
                --manifest "$DATA_DIR/libritts_r_full_manifest.jsonl" \
                --checkpoint-dir "$JOB_CKPT_DIR" \
                --device "$DEVICE" \
                --steps 200000 \
                --batch-size 8 \
                --grad-accum 4 \
                --lr 2e-4 \
                --warmup 2000 \
                --segment-sec 2.0 \
                --gan-start-step 50000 \
                --adv-weight 1.0 \
                --fm-weight 2.0 \
                --entropy-weight 0.1 \
                --wavlm-weight 0.5 \
                --ema-decay 0.999 \
                --progressive \
                --augment \
                --save-every 5000 \
                --num-workers 4 \
                $RESUME_FLAG &
            TRAIN_PID=$!
            ;;

        distill_v3)
            JOB_CKPT_DIR="$CKPT_DIR/flow_v3_distilled"
            GCS_CKPT_DIR="$BUCKET/checkpoints/flow_v3_distilled"
            TEACHER_DIR="$CKPT_DIR/flow_v3_large_fixed"
            mkdir -p "$JOB_CKPT_DIR"

            # Find teacher checkpoint (must exist)
            TEACHER=$(find_latest_checkpoint "$TEACHER_DIR" "flow_v3_best.pt")
            if [ -z "$TEACHER" ]; then
                TEACHER=$(find_latest_checkpoint "$TEACHER_DIR" "flow_v3_step_*.pt")
            fi
            if [ -z "$TEACHER" ]; then
                echo "ERROR: No flow v3 teacher checkpoint found in $TEACHER_DIR"
                exit 1
            fi
            echo "  Teacher: $TEACHER"

            python3 -u train_distill_v3.py \
                --teacher-checkpoint "$TEACHER" \
                --manifest "$DATA_DIR/libritts_r_full_manifest.jsonl" \
                --output-dir "$JOB_CKPT_DIR" \
                --device "$DEVICE" \
                --steps 50000 \
                --batch-size 4 \
                --teacher-steps 8 \
                --model-size large \
                --phonemes \
                --lr 1e-4 \
                --save-every 5000 \
                --val-every 5000 \
                --num-workers 4 &
            TRAIN_PID=$!
            ;;

        *)
            echo "ERROR: Unknown job '$JOB'. Use: flow_v3, vocoder, speaker_encoder, codec_12hz, distill_v3"
            exit 1
            ;;
    esac

    echo "  Training PID: $TRAIN_PID"

    # ─── Checkpoint sync loop ─────────────────────────────────────────────────
    # Sync checkpoints to GCS every 10 minutes while training runs
    while kill -0 $TRAIN_PID 2>/dev/null; do
        sleep 600  # 10 minutes
        if kill -0 $TRAIN_PID 2>/dev/null; then
            sync_checkpoints "$JOB_CKPT_DIR" "$GCS_CKPT_DIR"
        fi
    done

    # Training process exited — capture exit code without triggering set -e
    EXIT_CODE=0
    wait $TRAIN_PID || EXIT_CODE=$?

    # Final sync
    sync_checkpoints "$JOB_CKPT_DIR" "$GCS_CKPT_DIR"

    if $PREEMPTED; then
        echo "[$(date)] VM was preempted. Will resume on next startup."
        exit 0
    fi

    if [ $EXIT_CODE -eq 0 ]; then
        echo "[$(date)] Training completed successfully!"
        echo "  Checkpoints at: $GCS_CKPT_DIR"

        # Check for chained next job
        NEXT_JOB=$(curl -sf -H "Metadata-Flavor: Google" \
            http://metadata.google.internal/computeMetadata/v1/instance/attributes/next-job 2>/dev/null || true)
        if [ -n "$NEXT_JOB" ]; then
            echo "[$(date)] Chaining to next job: $NEXT_JOB"
            exec "$0" "$NEXT_JOB" "$BUCKET"
        fi

        exit 0
    fi

    echo "[$(date)] Training crashed (exit code $EXIT_CODE). Restarting in 30s..."
    sleep 30
done

echo "[$(date)] Exceeded max retries ($MAX_RETRIES). Giving up."
exit 1
