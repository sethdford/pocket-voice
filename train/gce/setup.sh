#!/bin/bash
# setup.sh — One-time: create GCS bucket and upload training data + checkpoints
set -euo pipefail

# ─── Configuration ────────────────────────────────────────────────────────────
PROJECT="${GCE_PROJECT:?Set GCE_PROJECT to your GCP project ID}"
BUCKET="${GCE_BUCKET:-gs://sonata-training-${PROJECT}}"
REGION="${GCE_REGION:-us-central1}"

LOCAL_DATA_DIR="${1:-$(cd "$(dirname "$0")/.." && pwd)/data}"
LOCAL_CKPT_DIR="${2:-$(cd "$(dirname "$0")/.." && pwd)/checkpoints}"

# ─── Preflight ────────────────────────────────────────────────────────────────
echo "=== Sonata GCE Training Setup ==="
echo "  Project:    $PROJECT"
echo "  Bucket:     $BUCKET"
echo "  Region:     $REGION"
echo "  Data dir:   $LOCAL_DATA_DIR"
echo "  Ckpt dir:   $LOCAL_CKPT_DIR"
echo ""

if ! command -v gcloud &>/dev/null; then
    echo "ERROR: gcloud CLI not found. Install: https://cloud.google.com/sdk/docs/install"
    exit 1
fi

# Check auth
if ! gcloud auth print-access-token &>/dev/null 2>&1; then
    echo "Not authenticated. Running gcloud auth login..."
    gcloud auth login
fi

gcloud config set project "$PROJECT"

# ─── Create GCS Bucket ───────────────────────────────────────────────────────
echo "=== Creating GCS bucket ==="
if gsutil ls "$BUCKET" &>/dev/null 2>&1; then
    echo "  Bucket $BUCKET already exists"
else
    gsutil mb -l "$REGION" "$BUCKET"
    echo "  Created $BUCKET"
fi

# ─── Upload Training Data ────────────────────────────────────────────────────
echo ""
echo "=== Uploading training data ==="
echo "  Source: $LOCAL_DATA_DIR"
echo "  Destination: $BUCKET/data/"

DATA_SIZE=$(du -sh "$LOCAL_DATA_DIR" | cut -f1)
echo "  Size: $DATA_SIZE"
echo ""
echo "  This will take 1-2 hours on a fast connection."
echo "  Using gsutil -m for parallel uploads."
echo ""

# Upload manifests first (small, needed to start training)
echo "  [1/3] Uploading manifests..."
gsutil -m cp "$LOCAL_DATA_DIR"/*.jsonl "$BUCKET/data/" 2>/dev/null || true

# Upload encoded shards
echo "  [2/3] Uploading encoded data shards..."
if ls "$LOCAL_CKPT_DIR"/encoded_data.shard*.pt &>/dev/null 2>&1; then
    gsutil -m cp "$LOCAL_CKPT_DIR"/encoded_data.shard*.pt "$BUCKET/checkpoints/"
fi

# Upload audio data (largest)
echo "  [3/3] Uploading audio data (this is the big one)..."
gsutil -m rsync -r "$LOCAL_DATA_DIR" "$BUCKET/data/"

echo "  Data upload complete."

# ─── Upload Checkpoints ──────────────────────────────────────────────────────
echo ""
echo "=== Uploading checkpoints ==="

# Flow v3
if [ -d "$LOCAL_CKPT_DIR/flow_v3_large_fixed" ]; then
    echo "  Uploading flow_v3_large_fixed..."
    gsutil -m cp "$LOCAL_CKPT_DIR/flow_v3_large_fixed/flow_v3_step_"*.pt "$BUCKET/checkpoints/flow_v3_large_fixed/"
    gsutil -m cp "$LOCAL_CKPT_DIR/flow_v3_large_fixed/flow_v3_best.pt" "$BUCKET/checkpoints/flow_v3_large_fixed/" 2>/dev/null || true
    gsutil -m cp "$LOCAL_CKPT_DIR/flow_v3_large_fixed/losses.jsonl" "$BUCKET/checkpoints/flow_v3_large_fixed/"
fi

# Vocoder
if [ -d "$LOCAL_CKPT_DIR/vocoder_large_fixed" ]; then
    echo "  Uploading vocoder_large_fixed..."
    gsutil -m rsync -r "$LOCAL_CKPT_DIR/vocoder_large_fixed" "$BUCKET/checkpoints/vocoder_large_fixed/"
fi

# Codec (needed for data pipeline)
if [ -d "$LOCAL_CKPT_DIR/codec" ]; then
    echo "  Uploading codec (best + final only)..."
    gsutil -m cp "$LOCAL_CKPT_DIR/codec/sonata_codec_best.pt" "$BUCKET/checkpoints/codec/" 2>/dev/null || true
    gsutil -m cp "$LOCAL_CKPT_DIR/codec/sonata_codec_final.pt" "$BUCKET/checkpoints/codec/" 2>/dev/null || true
fi

echo ""
echo "=== Setup complete ==="
echo ""
echo "  Bucket: $BUCKET"
echo "  Next: ./launch.sh flow_v3   (or: ./launch.sh vocoder)"
