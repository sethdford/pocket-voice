#!/bin/bash
# autopilot.sh — Automated training management for all GCE jobs
#
# Features:
#   - Monitors all sonata-train-* VMs every 5 minutes
#   - Auto-restarts preempted SPOT VMs (tries multiple zones)
#   - Auto-deletes VMs when training completes (exit code 0)
#   - Logs everything to autopilot.log
#   - Sends desktop notifications on macOS
#
# Usage:
#   ./autopilot.sh              # Run in foreground
#   nohup ./autopilot.sh &      # Run in background
#   ./autopilot.sh --once       # Single check, no loop
set -euo pipefail

PROJECT="${GCE_PROJECT:?Set GCE_PROJECT}"
BUCKET="${GCE_BUCKET:-gs://sonata-training-${PROJECT}}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG="$SCRIPT_DIR/autopilot.log"
ONCE=false
INTERVAL=300  # 5 minutes

for arg in "$@"; do
    case "$arg" in
        --once) ONCE=true ;;
        --interval=*) INTERVAL="${arg#--interval=}" ;;
    esac
done

ZONES=(us-central1-a us-central1-b us-central1-c us-east1-b us-east1-c us-west1-a us-west1-b)

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"
}

notify() {
    # macOS desktop notification
    osascript -e "display notification \"$1\" with title \"Sonata Training\"" 2>/dev/null || true
}

check_training_complete() {
    local name="$1" zone="$2"
    # Check if training log shows completion
    local last_lines
    last_lines=$(gcloud compute ssh "$name" --zone="$zone" --project="$PROJECT" \
        --command="tail -5 /var/log/sonata-training.log 2>/dev/null" 2>/dev/null) || return 1

    if echo "$last_lines" | grep -q "completed successfully"; then
        return 0
    fi
    return 1
}

try_restart_vm() {
    local name="$1" original_zone="$2"

    # First try the original zone
    log "  Restarting $name in $original_zone..."
    if gcloud compute instances start "$name" --zone="$original_zone" --project="$PROJECT" 2>/dev/null; then
        log "  Restarted in $original_zone"
        notify "$name restarted in $original_zone"
        return 0
    fi

    log "  Zone $original_zone unavailable, trying other zones..."

    # Get job name from VM name and validate it
    local job="${name#sonata-train-}"
    job="${job//-/_}"
    if [[ ! "$job" =~ ^[a-zA-Z0-9_]+$ ]]; then
        log "  WARNING: Invalid job name '$job' from VM '$name', skipping restart"
        return 1
    fi

    # Try other zones by recreating
    for zone in "${ZONES[@]}"; do
        [ "$zone" = "$original_zone" ] && continue
        log "  Trying zone $zone..."
        if GCE_ZONE="$zone" GCE_PROJECT="$PROJECT" GCE_BUCKET="$BUCKET" \
            "$SCRIPT_DIR/launch.sh" "$job" 2>/dev/null; then
            log "  Launched $name in $zone"
            notify "$name relaunched in $zone"
            # Delete the old terminated VM
            gcloud compute instances delete "$name" --zone="$original_zone" \
                --project="$PROJECT" --quiet 2>/dev/null || true
            return 0
        fi
    done

    log "  WARNING: Could not restart $name in any zone"
    notify "FAILED to restart $name - all zones exhausted"
    return 1
}

teardown_vm() {
    local name="$1" zone="$2"
    log "  Training complete on $name — tearing down VM"

    # Final checkpoint sync (the wrapper should have done this, but be safe)
    local job="${name#sonata-train-}"
    job="${job//-/_}"

    # Validate job name
    if [[ ! "$job" =~ ^[a-zA-Z0-9_]+$ ]]; then
        log "  WARNING: Invalid job name '$job' from VM '$name', skipping final sync"
    else
        log "  Final checkpoint sync..."
        gcloud compute ssh "$name" --zone="$zone" --project="$PROJECT" \
            --command="sudo bash -c 'gsutil -m rsync /opt/sonata/train/checkpoints/*/ \"$BUCKET\"/checkpoints/ 2>/dev/null'" \
            2>/dev/null || true
    fi

    # Delete the VM
    gcloud compute instances delete "$name" --zone="$zone" \
        --project="$PROJECT" --quiet 2>/dev/null || true
    log "  Deleted $name"
    notify "$name completed and deleted - saving cost!"
}

check_all() {
    log "--- Checking all training VMs ---"

    local vms
    vms=$(gcloud compute instances list --project="$PROJECT" --filter="name~sonata-train" \
        --format="csv[no-heading](name,zone,status)" 2>/dev/null) || true

    if [ -z "$vms" ]; then
        log "  No training VMs found. All done!"
        return 1  # Signal to exit loop
    fi

    local any_running=false

    while IFS=',' read -r name zone status; do
        [ -z "$name" ] && continue

        case "$status" in
            RUNNING)
                any_running=true
                # Check if training completed
                if check_training_complete "$name" "$zone"; then
                    teardown_vm "$name" "$zone"
                else
                    # Get latest step for logging
                    local step
                    step=$(gcloud compute ssh "$name" --zone="$zone" --project="$PROJECT" \
                        --command="tail -1 /var/log/sonata-training.log 2>/dev/null" 2>/dev/null \
                        | grep -oP 'step\s+\K\d+' | tail -1) || true
                    [ -n "$step" ] && log "  $name: RUNNING (step $step)" || log "  $name: RUNNING"
                fi
                ;;
            TERMINATED)
                log "  $name: TERMINATED (preempted?) — attempting restart"
                try_restart_vm "$name" "$zone" && any_running=true
                ;;
            STAGING|PROVISIONING)
                any_running=true
                log "  $name: $status (starting up)"
                ;;
            *)
                log "  $name: $status"
                ;;
        esac
    done <<< "$vms"

    $any_running && return 0 || return 1
}

# ─── Main ─────────────────────────────────────────────────────────────────────
log "=== Sonata Autopilot Started ==="
log "  Project: $PROJECT"
log "  Bucket: $BUCKET"
log "  Interval: ${INTERVAL}s"

if $ONCE; then
    check_all
    exit 0
fi

while true; do
    if ! check_all; then
        log "=== All training complete! Autopilot shutting down. ==="
        notify "All training jobs complete!"
        break
    fi
    sleep "$INTERVAL"
done
