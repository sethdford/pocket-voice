#!/bin/bash
# launch.sh — Create a preemptible L4 VM and start training
set -euo pipefail

JOB="${1:?Usage: ./launch.sh <flow_v3|vocoder|speaker_encoder|codec_12hz|distill_v3> [--on-demand]}"
ON_DEMAND=false
NEXT_JOB=""
for arg in "${@:2}"; do
    case "$arg" in
        --on-demand) ON_DEMAND=true ;;
        --next-job=*) NEXT_JOB="${arg#--next-job=}" ;;
    esac
done

case "$JOB" in
    flow_v3|vocoder|speaker_encoder|codec_12hz|distill_v3) ;;
    *) echo "ERROR: Unknown job '$JOB'. Use: flow_v3, vocoder, speaker_encoder, codec_12hz, distill_v3"; exit 1 ;;
esac

# ─── Configuration ────────────────────────────────────────────────────────────
PROJECT="${GCE_PROJECT:?Set GCE_PROJECT to your GCP project ID}"
BUCKET="${GCE_BUCKET:-gs://sonata-training-${PROJECT}}"
ZONE="${GCE_ZONE:-us-central1-a}"
VM_NAME="sonata-train-${JOB//_/-}"
MACHINE_TYPE="g2-standard-8"  # 1x L4 24GB, 8 vCPU, 32GB RAM
BOOT_DISK_SIZE="100GB"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

echo "=== Launching GCE Training VM ==="
echo "  Job:      $JOB"
echo "  VM:       $VM_NAME"
echo "  Machine:  $MACHINE_TYPE"
echo "  Zone:     $ZONE"
echo "  Bucket:   $BUCKET"
if $ON_DEMAND; then
    echo "  Pricing:  ON-DEMAND (~\$0.70/hr)"
else
    echo "  Pricing:  SPOT (~\$0.22/hr)"
fi
[ -n "$NEXT_JOB" ] && echo "  Next job: $NEXT_JOB (auto-chain)"
echo ""

# ─── Build startup script ────────────────────────────────────────────────────
STARTUP_SCRIPT=$(cat <<'STARTUP_EOF'
#!/bin/bash
set -euo pipefail
exec > /var/log/sonata-startup.log 2>&1

echo "=== Sonata Training VM Startup ==="
echo "Date: $(date)"

# Metadata
BUCKET=$(curl -sH "Metadata-Flavor: Google" http://metadata.google.internal/computeMetadata/v1/instance/attributes/bucket)
JOB=$(curl -sH "Metadata-Flavor: Google" http://metadata.google.internal/computeMetadata/v1/instance/attributes/job)

echo "Bucket: $BUCKET"
echo "Job: $JOB"

# ─── Install dependencies ────────────────────────────────────────────────────
# PyTorch 2.7 + CUDA 12.8 are pre-installed on the deep learning VM image

# Add gcsfuse repo
export GCSFUSE_REPO=gcsfuse-$(lsb_release -c -s)
echo "deb [signed-by=/usr/share/keyrings/cloud.google.asc] https://packages.cloud.google.com/apt $GCSFUSE_REPO main" \
    | tee /etc/apt/sources.list.d/gcsfuse.list
curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg \
    | tee /usr/share/keyrings/cloud.google.asc > /dev/null

apt-get update -qq
apt-get install -y -qq gcsfuse espeak-ng sox libsndfile1

# Install extra Python packages into the pre-installed environment
pip install --quiet soundfile safetensors

# ─── Mount GCS ────────────────────────────────────────────────────────────────
mkdir -p /mnt/sonata
# Enable allow_other so non-root users can access mount
echo "user_allow_other" >> /etc/fuse.conf
gcsfuse --implicit-dirs --file-mode=666 --dir-mode=777 -o allow_other \
    "$(echo $BUCKET | sed 's|gs://||')" /mnt/sonata

echo "GCS mounted at /mnt/sonata"
ls /mnt/sonata/

# ─── Clone training code ─────────────────────────────────────────────────────
# Copy training scripts from GCS (uploaded by launch.sh)
mkdir -p /opt/sonata/train/sonata
gsutil -m cp "$BUCKET/code/train/sonata/*.py" /opt/sonata/train/sonata/

# ─── Symlink data ────────────────────────────────────────────────────────────
ln -sf /mnt/sonata/data /opt/sonata/train/data
ln -sf /mnt/sonata/checkpoints /opt/sonata/train/checkpoints

# Manifest has absolute Mac paths — create symlink so they resolve on VM
# e.g. /Users/sethford/Documents/pocket-voice/train/data/... → /mnt/sonata/data/...
mkdir -p /Users/sethford/Documents/pocket-voice/train
ln -sf /mnt/sonata/data /Users/sethford/Documents/pocket-voice/train/data

# ─── Start training ──────────────────────────────────────────────────────────
cd /opt/sonata/train/sonata

# Copy train_wrapper.sh
gsutil cp "$BUCKET/code/train/gce/train_wrapper.sh" /opt/sonata/train_wrapper.sh
chmod +x /opt/sonata/train_wrapper.sh

echo "Starting training wrapper for job: $JOB"
nohup /opt/sonata/train_wrapper.sh "$JOB" "$BUCKET" > /var/log/sonata-training.log 2>&1 &
echo "Training PID: $!"
echo "Startup complete at $(date)"
STARTUP_EOF
)

# ─── Upload training code to GCS ─────────────────────────────────────────────
echo "Uploading training code to GCS..."
gsutil -m rsync -r -x '__pycache__/.*|.*\.pyc' "$REPO_DIR/train/sonata/" "$BUCKET/code/train/sonata/"
gsutil -m cp "$SCRIPT_DIR/train_wrapper.sh" "$BUCKET/code/train/gce/"

# ─── Create VM ────────────────────────────────────────────────────────────────
echo "Creating VM $VM_NAME..."

PREEMPTIBLE_FLAG=""
if ! $ON_DEMAND; then
    PREEMPTIBLE_FLAG="--provisioning-model=SPOT --instance-termination-action=STOP"
fi

# Write startup script to temp file (process substitution doesn't work with gcloud)
STARTUP_TMP=$(mktemp)
echo "$STARTUP_SCRIPT" > "$STARTUP_TMP"

gcloud compute instances create "$VM_NAME" \
    --project="$PROJECT" \
    --zone="$ZONE" \
    --machine-type="$MACHINE_TYPE" \
    --accelerator="type=nvidia-l4,count=1" \
    --boot-disk-size="$BOOT_DISK_SIZE" \
    --image-family="pytorch-2-7-cu128-ubuntu-2204-nvidia-570" \
    --image-project="deeplearning-platform-release" \
    --maintenance-policy=TERMINATE \
    $PREEMPTIBLE_FLAG \
    --scopes="storage-full" \
    --metadata="bucket=$BUCKET,job=$JOB${NEXT_JOB:+,next-job=$NEXT_JOB}" \
    --metadata-from-file="startup-script=$STARTUP_TMP"

rm -f "$STARTUP_TMP"

echo ""
echo "=== VM Created ==="
echo "  Name: $VM_NAME"
echo "  Zone: $ZONE"
echo ""
echo "  Monitor training:"
echo "    ./monitor.sh $JOB"
echo ""
echo "  SSH into VM:"
echo "    gcloud compute ssh $VM_NAME --zone=$ZONE"
echo ""
echo "  View startup log:"
echo "    gcloud compute ssh $VM_NAME --zone=$ZONE -- tail -f /var/log/sonata-startup.log"
