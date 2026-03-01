#!/bin/bash
# teardown.sh — Stop VM, download checkpoints, optionally delete resources
# Usage: ./teardown.sh <flow_v3|vocoder> [--download-only|--delete-all]
set -euo pipefail

JOB="${1:?Usage: ./teardown.sh <flow_v3|vocoder|speaker_encoder|codec_12hz|distill_v3> [--download-only|--delete-all]}"
MODE="${2:---stop}"

# ─── Configuration ────────────────────────────────────────────────────────────
PROJECT="${GCE_PROJECT:?Set GCE_PROJECT to your GCP project ID}"
BUCKET="${GCE_BUCKET:-gs://sonata-training-${PROJECT}}"
ZONE="${GCE_ZONE:-us-central1-a}"
VM_NAME="sonata-train-${JOB//_/-}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOCAL_CKPT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)/checkpoints"

case "$JOB" in
    flow_v3)
        GCS_CKPT_DIR="$BUCKET/checkpoints/flow_v3_large_fixed"
        LOCAL_JOB_DIR="$LOCAL_CKPT_DIR/flow_v3_large_fixed"
        ;;
    vocoder)
        GCS_CKPT_DIR="$BUCKET/checkpoints/vocoder_large_fixed"
        LOCAL_JOB_DIR="$LOCAL_CKPT_DIR/vocoder_large_fixed"
        ;;
    speaker_encoder)
        GCS_CKPT_DIR="$BUCKET/checkpoints/speaker_encoder"
        LOCAL_JOB_DIR="$LOCAL_CKPT_DIR/speaker_encoder"
        ;;
    codec_12hz)
        GCS_CKPT_DIR="$BUCKET/checkpoints/codec_12hz"
        LOCAL_JOB_DIR="$LOCAL_CKPT_DIR/codec_12hz"
        ;;
    distill_v3)
        GCS_CKPT_DIR="$BUCKET/checkpoints/flow_v3_distilled"
        LOCAL_JOB_DIR="$LOCAL_CKPT_DIR/flow_v3_distilled"
        ;;
    *)
        echo "ERROR: Unknown job '$JOB'. Use: flow_v3, vocoder, speaker_encoder, codec_12hz, distill_v3"
        exit 1
        ;;
esac

download_checkpoints() {
    echo "=== Downloading checkpoints ==="
    echo "  From: $GCS_CKPT_DIR"
    echo "  To:   $LOCAL_JOB_DIR"
    mkdir -p "$LOCAL_JOB_DIR"
    gsutil -m rsync -r "$GCS_CKPT_DIR" "$LOCAL_JOB_DIR/"
    echo "  Download complete."
    echo ""
    echo "  Local files:"
    ls -lh "$LOCAL_JOB_DIR/" | tail -10
}

stop_vm() {
    echo "=== Stopping VM ==="
    VM_STATUS=$(gcloud compute instances describe "$VM_NAME" \
        --project="$PROJECT" \
        --zone="$ZONE" \
        --format="value(status)" 2>/dev/null || echo "NOT_FOUND")

    if [ "$VM_STATUS" = "NOT_FOUND" ]; then
        echo "  VM $VM_NAME not found (already deleted?)"
        return
    fi

    if [ "$VM_STATUS" = "TERMINATED" ] || [ "$VM_STATUS" = "STOPPED" ]; then
        echo "  VM $VM_NAME already stopped (status: $VM_STATUS)"
        return
    fi

    echo "  Stopping $VM_NAME..."
    gcloud compute instances stop "$VM_NAME" \
        --project="$PROJECT" \
        --zone="$ZONE"
    echo "  VM stopped."
}

delete_vm() {
    echo "=== Deleting VM ==="
    gcloud compute instances delete "$VM_NAME" \
        --project="$PROJECT" \
        --zone="$ZONE" \
        --quiet
    echo "  VM $VM_NAME deleted."
}

case "$MODE" in
    --stop)
        echo "=== Teardown: Stop VM + Download Checkpoints ==="
        echo "  Job: $JOB"
        echo "  VM:  $VM_NAME"
        echo ""
        download_checkpoints
        echo ""
        stop_vm
        echo ""
        echo "=== Done ==="
        echo "  Checkpoints saved to: $LOCAL_JOB_DIR"
        echo "  VM stopped (still billed for disk). Delete with:"
        echo "    ./teardown.sh $JOB --delete-all"
        ;;

    --download-only)
        echo "=== Download Checkpoints Only ==="
        echo "  Job: $JOB"
        echo ""
        download_checkpoints
        echo ""
        echo "=== Done ==="
        echo "  VM is still running. Stop with:"
        echo "    ./teardown.sh $JOB"
        ;;

    --delete-all)
        echo "=== Full Teardown: Download + Delete VM ==="
        echo "  Job:    $JOB"
        echo "  VM:     $VM_NAME"
        echo "  Bucket: $BUCKET (NOT deleted — manual cleanup)"
        echo ""
        read -p "  Delete VM $VM_NAME? This cannot be undone. [y/N] " CONFIRM
        if [ "$CONFIRM" != "y" ] && [ "$CONFIRM" != "Y" ]; then
            echo "  Aborted."
            exit 0
        fi
        echo ""
        download_checkpoints
        echo ""
        delete_vm
        echo ""
        echo "=== Done ==="
        echo "  Checkpoints saved to: $LOCAL_JOB_DIR"
        echo "  VM deleted."
        echo "  GCS bucket $BUCKET still exists. Delete manually if no longer needed:"
        echo "    gsutil rm -r $BUCKET"
        ;;

    *)
        echo "Usage: ./teardown.sh <flow_v3|vocoder> [--stop|--download-only|--delete-all]"
        echo ""
        echo "  --stop           Stop VM + download checkpoints (default)"
        echo "  --download-only  Download checkpoints without stopping VM"
        echo "  --delete-all     Download checkpoints, then delete VM entirely"
        exit 1
        ;;
esac
