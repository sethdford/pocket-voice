#!/bin/bash
##############################################################################
# Deploy Sonata v2 Training Pipeline to GCE
#
# Updates the training instance with latest code and starts the full
# training pipeline in a screen session.
#
# Usage:
#   ./deploy_gce.sh                    # Deploy and start full pipeline
#   ./deploy_gce.sh --model cam        # Train single model
#   ./deploy_gce.sh --synthetic        # Test with synthetic data first
#   ./deploy_gce.sh --status           # Check training status
#   ./deploy_gce.sh --stop             # Stop current training
#   ./deploy_gce.sh --logs             # Tail training logs
##############################################################################

set -euo pipefail

# ── Configuration ──────────────────────────────────────────────────────────

INSTANCE="sonata-train-speaker-encoder"
ZONE="us-central1-a"
PROJECT="johnb-2025"
REMOTE_DIR="/opt/sonata/train"
GCS_BUCKET="gs://sonata-training-johnb-2025"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log() { echo -e "${BLUE}[deploy]${NC} $1"; }
success() { echo -e "${GREEN}[✓]${NC} $1"; }
warn() { echo -e "${YELLOW}[⚠]${NC} $1"; }
fail() { echo -e "${RED}[✗]${NC} $1"; }

# ── SSH helper ─────────────────────────────────────────────────────────────

gce_ssh() {
    gcloud compute ssh "$INSTANCE" \
        --zone="$ZONE" \
        --project="$PROJECT" \
        --command="$1" \
        2>/dev/null
}

gce_scp() {
    gcloud compute scp "$1" "$INSTANCE:$2" \
        --zone="$ZONE" \
        --project="$PROJECT" \
        2>/dev/null
}

# ── Commands ───────────────────────────────────────────────────────────────

check_status() {
    log "Checking training status on $INSTANCE..."

    # Check if instance is running
    local status=$(gcloud compute instances describe "$INSTANCE" \
        --zone="$ZONE" --project="$PROJECT" \
        --format="value(status)" 2>/dev/null)

    if [[ "$status" != "RUNNING" ]]; then
        fail "Instance $INSTANCE is $status (not running)"
        return 1
    fi
    success "Instance is running"

    # Check GPU
    gce_ssh "nvidia-smi --query-gpu=name,temperature.gpu,utilization.gpu,memory.used,memory.total --format=csv,noheader" || true

    # Check training processes
    log "Active training processes:"
    gce_ssh "ps aux | grep -E 'train_|orchestrate' | grep -v grep" || echo "  (none)"

    # Check screen sessions
    log "Screen sessions:"
    gce_ssh "screen -ls 2>/dev/null" || echo "  (none)"

    # Check orchestrator state
    log "Training state:"
    gce_ssh "cat $REMOTE_DIR/checkpoints/orchestrator_state.json 2>/dev/null" || echo "  (no state file)"

    # Check checkpoints
    log "Checkpoints:"
    gce_ssh "ls -lh $REMOTE_DIR/checkpoints/*/best*.pt 2>/dev/null" || echo "  (none yet)"

    # Check disk space
    log "Disk space:"
    gce_ssh "df -h / | tail -1"
}

stop_training() {
    log "Stopping training on $INSTANCE..."

    # Kill screen sessions
    gce_ssh "screen -ls | grep -o '[0-9]*\.sonata' | xargs -I{} screen -S {} -X quit 2>/dev/null" || true

    # Kill training processes
    gce_ssh "pkill -f 'train_|orchestrate' 2>/dev/null" || true

    success "Training stopped"
}

tail_logs() {
    log "Tailing training logs..."
    gcloud compute ssh "$INSTANCE" \
        --zone="$ZONE" \
        --project="$PROJECT" \
        --command="tail -f $REMOTE_DIR/checkpoints/orchestrator.log 2>/dev/null || echo 'No log file yet'"
}

deploy_and_start() {
    local extra_args="${1:-}"

    log "Deploying training pipeline to $INSTANCE..."

    # 1. Check instance is running
    local status=$(gcloud compute instances describe "$INSTANCE" \
        --zone="$ZONE" --project="$PROJECT" \
        --format="value(status)" 2>/dev/null)

    if [[ "$status" != "RUNNING" ]]; then
        log "Starting instance $INSTANCE..."
        gcloud compute instances start "$INSTANCE" \
            --zone="$ZONE" --project="$PROJECT"
        sleep 30  # Wait for boot
    fi
    success "Instance is running"

    # 2. Stop any existing training
    log "Stopping existing training..."
    gce_ssh "pkill -f 'train_|orchestrate' 2>/dev/null; screen -ls | grep -o '[0-9]*\.sonata' | xargs -I{} screen -S {} -X quit 2>/dev/null" || true
    sleep 2

    # 3. Upload latest training scripts
    log "Uploading training scripts..."
    local train_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

    for f in train_speaker_encoder.py train_codec.py train_stt.py train_tts.py \
             train_cfm.py export_candle.py export_coreml.py quantize.py \
             orchestrate.sh requirements.txt; do
        if [[ -f "$train_dir/$f" ]]; then
            gce_scp "$train_dir/$f" "$REMOTE_DIR/$f"
        fi
    done

    # Upload data module
    gce_ssh "mkdir -p $REMOTE_DIR/data"
    for f in __init__.py codec_dataset.py; do
        if [[ -f "$train_dir/data/$f" ]]; then
            gce_scp "$train_dir/data/$f" "$REMOTE_DIR/data/$f"
        fi
    done

    gce_ssh "chmod +x $REMOTE_DIR/orchestrate.sh"
    success "Scripts uploaded"

    # 4. Install/update dependencies
    log "Installing dependencies..."
    gce_ssh "pip install -q safetensors torchaudio numpy tqdm 2>/dev/null" || true

    # 5. Ensure data directories exist
    gce_ssh "mkdir -p $REMOTE_DIR/checkpoints/{cam,codec,stt,tts,cfm} $REMOTE_DIR/models"

    # 6. Check for training data
    log "Checking training data..."
    gce_ssh "ls -d /mnt/sonata/data/libritts-r/LibriTTS_R 2>/dev/null && echo 'LibriTTS-R: found' || echo 'LibriTTS-R: NOT FOUND'" || true
    gce_ssh "ls -d /mnt/sonata/data/audio 2>/dev/null && echo 'Audio data: found' || echo 'Audio data: NOT FOUND'" || true

    # 7. Start training in detached screen session
    log "Starting training pipeline in screen session 'sonata'..."
    local screen_cmd="cd $REMOTE_DIR && ./orchestrate.sh $extra_args 2>&1 | tee -a checkpoints/orchestrator.log"

    gce_ssh "screen -dmS sonata bash -c '$screen_cmd; echo DONE'"
    sleep 2

    # Verify it started
    gce_ssh "screen -ls | grep sonata" && success "Training started in screen session 'sonata'" || fail "Failed to start screen session"

    echo ""
    log "Monitor with:"
    echo "  ./deploy_gce.sh --status    # Check progress"
    echo "  ./deploy_gce.sh --logs      # Tail logs"
    echo "  gcloud compute ssh $INSTANCE --zone=$ZONE --project=$PROJECT"
    echo "  screen -r sonata            # Attach to session"
}

download_models() {
    log "Downloading trained models from GCS..."
    local local_models="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/models"
    mkdir -p "$local_models"

    gsutil -m cp "$GCS_BUCKET/models/sonata-*.safetensors" "$local_models/" 2>/dev/null || \
        warn "No models in GCS yet"

    log "Downloading from instance..."
    gcloud compute scp "$INSTANCE:$REMOTE_DIR/models/sonata-*.safetensors" "$local_models/" \
        --zone="$ZONE" --project="$PROJECT" 2>/dev/null || \
        warn "No models on instance yet"

    ls -lh "$local_models/"*.safetensors 2>/dev/null || echo "  (no models downloaded)"
}

# ── Main ───────────────────────────────────────────────────────────────────

case "${1:-deploy}" in
    --status|-s)
        check_status
        ;;
    --stop)
        stop_training
        ;;
    --logs|-l)
        tail_logs
        ;;
    --download|-d)
        download_models
        ;;
    --synthetic)
        deploy_and_start "--synthetic"
        ;;
    --model)
        deploy_and_start "--model ${2:-cam}"
        ;;
    deploy|--deploy)
        deploy_and_start "${2:-}"
        ;;
    --help|-h)
        cat << 'EOF'
Sonata v2 GCE Training Deployment

Usage: ./deploy_gce.sh [COMMAND]

Commands:
  deploy (default)    Deploy scripts and start full training pipeline
  --model <name>      Deploy and train single model (cam, codec, stt, tts, cfm)
  --synthetic         Deploy and run with synthetic data (test mode)
  --status, -s        Check training status and GPU utilization
  --logs, -l          Tail training logs
  --stop              Stop all training processes
  --download, -d      Download trained models from GCE/GCS
  --help, -h          Show this help

Examples:
  ./deploy_gce.sh                     # Deploy and start full pipeline
  ./deploy_gce.sh --synthetic         # Test run first
  ./deploy_gce.sh --model codec       # Train codec only
  ./deploy_gce.sh --status            # Check progress
  ./deploy_gce.sh --logs              # Watch training
  ./deploy_gce.sh --download          # Get trained models

GCE Instance: sonata-train-speaker-encoder (g2-standard-8, L4 GPU)
EOF
        ;;
    *)
        fail "Unknown command: $1"
        echo "Run ./deploy_gce.sh --help for usage"
        exit 1
        ;;
esac
