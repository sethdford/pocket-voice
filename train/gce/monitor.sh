#!/bin/bash
# monitor.sh — Tail training logs, check VM status, show progress
# Usage: ./monitor.sh <flow_v3|vocoder> [--status|--logs|--loss]
set -euo pipefail

JOB="${1:?Usage: ./monitor.sh <flow_v3|vocoder|speaker_encoder|codec_12hz|distill_v3> [--status|--logs|--loss]}"
MODE="${2:---status}"

# ─── Configuration ────────────────────────────────────────────────────────────
PROJECT="${GCE_PROJECT:?Set GCE_PROJECT to your GCP project ID}"
BUCKET="${GCE_BUCKET:-gs://sonata-training-${PROJECT}}"
ZONE="${GCE_ZONE:-us-central1-a}"
VM_NAME="sonata-train-${JOB//_/-}"

case "$MODE" in
    --status)
        echo "=== VM Status ==="
        gcloud compute instances describe "$VM_NAME" \
            --project="$PROJECT" \
            --zone="$ZONE" \
            --format="table(name, status, scheduling.preemptible, zone, machineType.basename(), networkInterfaces[0].accessConfigs[0].natIP)" \
            2>/dev/null || echo "  VM $VM_NAME not found"

        echo ""
        echo "=== GPU Utilization ==="
        gcloud compute ssh "$VM_NAME" \
            --project="$PROJECT" \
            --zone="$ZONE" \
            --command="nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu --format=csv,noheader" \
            2>/dev/null || echo "  Cannot connect to VM"

        echo ""
        echo "=== Latest Checkpoints (GCS) ==="
        case "$JOB" in
            flow_v3)
                gsutil ls -l "$BUCKET/checkpoints/flow_v3_large_fixed/" 2>/dev/null \
                    | grep "flow_v3_step_\|flow_v3_best" | sort -k2 | tail -5
                ;;
            vocoder)
                gsutil ls -l "$BUCKET/checkpoints/vocoder_large_fixed/" 2>/dev/null \
                    | grep "vocoder_" | sort -k2 | tail -5
                ;;
            speaker_encoder)
                gsutil ls -l "$BUCKET/checkpoints/speaker_encoder/" 2>/dev/null \
                    | grep "speaker_encoder_" | sort -k2 | tail -5
                ;;
            codec_12hz)
                gsutil ls -l "$BUCKET/checkpoints/codec_12hz/" 2>/dev/null \
                    | grep "sonata_codec_" | sort -k2 | tail -5
                ;;
            distill_v3)
                gsutil ls -l "$BUCKET/checkpoints/flow_v3_distilled/" 2>/dev/null \
                    | grep "distilled_\|best" | sort -k2 | tail -5
                ;;
        esac

        echo ""
        echo "=== Disk Usage ==="
        gcloud compute ssh "$VM_NAME" \
            --project="$PROJECT" \
            --zone="$ZONE" \
            --command="df -h / | tail -1" \
            2>/dev/null || echo "  Cannot connect to VM"
        ;;

    --logs)
        echo "=== Tailing training logs (Ctrl+C to stop) ==="
        gcloud compute ssh "$VM_NAME" \
            --project="$PROJECT" \
            --zone="$ZONE" \
            --command="tail -f /var/log/sonata-training.log"
        ;;

    --loss)
        echo "=== Recent Loss Values ==="
        case "$JOB" in
            flow_v3)
                LOSS_FILE="$BUCKET/checkpoints/flow_v3_large_fixed/losses.jsonl"
                ;;
            vocoder)
                LOSS_FILE="$BUCKET/checkpoints/vocoder_large_fixed/losses.jsonl"
                ;;
            speaker_encoder)
                LOSS_FILE="$BUCKET/checkpoints/speaker_encoder/losses.jsonl"
                ;;
            codec_12hz)
                LOSS_FILE="$BUCKET/checkpoints/codec_12hz/losses.jsonl"
                ;;
            distill_v3)
                LOSS_FILE="$BUCKET/checkpoints/flow_v3_distilled/losses.jsonl"
                ;;
            *)
                echo "ERROR: Unknown job '$JOB'"
                exit 1
                ;;
        esac

        TMPFILE=$(mktemp)
        gsutil cp "$LOSS_FILE" "$TMPFILE" 2>/dev/null

        if [ -s "$TMPFILE" ]; then
            echo "  Last 20 entries:"
            tail -20 "$TMPFILE"
            echo ""
            TOTAL=$(wc -l < "$TMPFILE")
            echo "  Total logged steps: $TOTAL"
        else
            echo "  No loss data found at $LOSS_FILE"
        fi
        rm -f "$TMPFILE"
        ;;

    --startup)
        echo "=== Startup Log ==="
        gcloud compute ssh "$VM_NAME" \
            --project="$PROJECT" \
            --zone="$ZONE" \
            --command="cat /var/log/sonata-startup.log"
        ;;

    *)
        echo "Usage: ./monitor.sh <flow_v3|vocoder|speaker_encoder|codec_12hz|distill_v3> [--status|--logs|--loss|--startup]"
        echo ""
        echo "  --status   VM status, GPU utilization, latest checkpoints (default)"
        echo "  --logs     Tail training log in real-time"
        echo "  --loss     Show recent loss values from GCS"
        echo "  --startup  Show VM startup log"
        exit 1
        ;;
esac
