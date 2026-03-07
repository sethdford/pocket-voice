#!/bin/bash
# auto_recover.sh — Auto-recover preempted SPOT training VMs
#
# Monitors active training jobs. When a VM is preempted, deletes it and
# relaunches in the next available zone via launch.sh.
#
# Usage:
#   ./auto_recover.sh                    # Run once (cron-friendly)
#   ./auto_recover.sh --loop             # Run continuously (every 5 min)
#   ./auto_recover.sh --loop --interval 120  # Custom interval (seconds)
#
# Cron example (every 5 min):
#   */5 * * * * cd /path/to/pocket-voice && train/gce/auto_recover.sh >> /tmp/sonata-recover.log 2>&1
#
# Environment:
#   GCE_PROJECT  — GCP project ID (default: johnb-2025)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LAUNCH_SCRIPT="$SCRIPT_DIR/launch.sh"
PROJECT="${GCE_PROJECT:-johnb-2025}"
STATE_FILE="${SCRIPT_DIR}/.recover_state.json"
LOG_PREFIX="[recover]"

# Zone rotation — try each zone in order until one works
ZONES=(
    us-central1-a
    us-central1-b
    us-east1-c
    us-east1-b
    us-east4-c
    us-east4-a
    us-west1-a
    us-west1-b
    us-west4-a
    us-west4-c
)

# Jobs to monitor — add/remove as needed
JOBS=(
    drafter
    distill_v3
    vocoder
    codec_12hz
)

# Max recovery attempts per job before giving up (resets on success)
MAX_ATTEMPTS=5

log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') $LOG_PREFIX $*"
}

# Get VM name for a job
vm_name() {
    local job="$1"
    echo "sonata-train-${job//_/-}"
}

# Find which zone a VM is in (if it exists)
find_vm_zone() {
    local vm="$1"
    gcloud compute instances list \
        --project="$PROJECT" \
        --filter="name=$vm" \
        --format="value(zone.basename())" 2>/dev/null | head -1
}

# Get VM status
vm_status() {
    local vm="$1"
    local zone="$2"
    gcloud compute instances describe "$vm" \
        --project="$PROJECT" --zone="$zone" \
        --format="value(status)" 2>/dev/null || echo "NOT_FOUND"
}

# Check if termination was due to preemption
was_preempted() {
    local vm="$1"
    local zone="$2"

    # Check scheduling.preemptible flag and last stop reason
    local preemptible
    preemptible=$(gcloud compute instances describe "$vm" \
        --project="$PROJECT" --zone="$zone" \
        --format="value(scheduling.preemptible)" 2>/dev/null) || true

    if [ "$preemptible" != "True" ]; then
        return 1  # Not a preemptible VM
    fi

    # Check for recent preemption operation
    local preempt_time
    preempt_time=$(gcloud compute operations list \
        --project="$PROJECT" \
        --filter="operationType=compute.instances.preempted AND targetLink~'$vm'" \
        --sort-by=~insertTime --limit=1 \
        --format="value(insertTime)" 2>/dev/null) || true

    [ -n "$preempt_time" ]
}

# Read attempt count from state file
get_attempts() {
    local job="$1"
    if [ -f "$STATE_FILE" ]; then
        python3 -c "
import json, sys
try:
    d = json.load(open('$STATE_FILE'))
    print(d.get('attempts', {}).get('$job', 0))
except: print(0)
" 2>/dev/null || echo 0
    else
        echo 0
    fi
}

# Update attempt count in state file
set_attempts() {
    local job="$1"
    local count="$2"
    python3 -c "
import json, os
path = '$STATE_FILE'
try:
    d = json.load(open(path)) if os.path.exists(path) else {}
except: d = {}
d.setdefault('attempts', {})['$job'] = $count
d.setdefault('zones', {})
json.dump(d, open(path, 'w'), indent=2)
" 2>/dev/null || true
}

# Record which zone was last used for a job
set_last_zone() {
    local job="$1"
    local zone="$2"
    python3 -c "
import json, os
path = '$STATE_FILE'
try:
    d = json.load(open(path)) if os.path.exists(path) else {}
except: d = {}
d.setdefault('zones', {})['$job'] = '$zone'
json.dump(d, open(path, 'w'), indent=2)
" 2>/dev/null || true
}

# Get last zone used for a job (to start rotation from next one)
get_last_zone() {
    local job="$1"
    if [ -f "$STATE_FILE" ]; then
        python3 -c "
import json
try:
    d = json.load(open('$STATE_FILE'))
    print(d.get('zones', {}).get('$job', ''))
except: print('')
" 2>/dev/null || echo ""
    else
        echo ""
    fi
}

# Delete a VM
delete_vm() {
    local vm="$1"
    local zone="$2"
    log "Deleting $vm in $zone..."
    gcloud compute instances delete "$vm" \
        --project="$PROJECT" --zone="$zone" --quiet 2>&1 || true
}

# Launch a job in a specific zone via launch.sh
launch_in_zone() {
    local job="$1"
    local zone="$2"
    log "Launching $job in $zone..."
    GCE_PROJECT="$PROJECT" GCE_ZONE="$zone" bash "$LAUNCH_SCRIPT" "$job" 2>&1
}

# Try to relaunch a job, rotating through zones
recover_job() {
    local job="$1"
    local vm
    vm=$(vm_name "$job")

    local attempts
    attempts=$(get_attempts "$job")

    if [ "$attempts" -ge "$MAX_ATTEMPTS" ]; then
        log "SKIP $job — exceeded $MAX_ATTEMPTS recovery attempts. Reset with: echo '{}' > $STATE_FILE"
        return 1
    fi

    # Find current zone and delete
    local current_zone
    current_zone=$(find_vm_zone "$vm")
    if [ -n "$current_zone" ]; then
        delete_vm "$vm" "$current_zone"
    fi

    # Get last zone to start rotation from next one
    local last_zone
    last_zone=$(get_last_zone "$job")
    local started=false
    if [ -z "$last_zone" ]; then
        started=true
    fi

    # Try each zone in rotation
    for zone in "${ZONES[@]}"; do
        # Skip zones until we pass the last used one
        if ! $started; then
            if [ "$zone" = "$last_zone" ]; then
                started=true
            fi
            continue
        fi

        # Skip zones that already have a running sonata-train instance (quota: 1 L4 per region)
        local region="${zone%-*}"
        local conflict
        conflict=$(gcloud compute instances list \
            --project="$PROJECT" \
            --filter="name~'sonata-train' AND zone~'$region' AND status=RUNNING" \
            --format="value(name)" 2>/dev/null | head -1) || true
        if [ -n "$conflict" ]; then
            log "  Skip $zone — $conflict already running in $region"
            continue
        fi

        # Try to launch
        if launch_in_zone "$job" "$zone"; then
            log "SUCCESS: $job launched in $zone"
            set_attempts "$job" 0
            set_last_zone "$job" "$zone"
            return 0
        else
            log "  Failed in $zone (stockout or quota). Trying next..."
            # Clean up failed instance if it was partially created
            gcloud compute instances delete "$vm" \
                --project="$PROJECT" --zone="$zone" --quiet 2>/dev/null || true
        fi
    done

    # Wrapped around — try zones before the last used one
    if [ -n "$last_zone" ]; then
        for zone in "${ZONES[@]}"; do
            if [ "$zone" = "$last_zone" ]; then
                break
            fi

            local region="${zone%-*}"
            local conflict
            conflict=$(gcloud compute instances list \
                --project="$PROJECT" \
                --filter="name~'sonata-train' AND zone~'$region' AND status=RUNNING" \
                --format="value(name)" 2>/dev/null | head -1) || true
            if [ -n "$conflict" ]; then
                continue
            fi

            if launch_in_zone "$job" "$zone"; then
                log "SUCCESS: $job launched in $zone"
                set_attempts "$job" 0
                set_last_zone "$job" "$zone"
                return 0
            else
                gcloud compute instances delete "$vm" \
                    --project="$PROJECT" --zone="$zone" --quiet 2>/dev/null || true
            fi
        done
    fi

    # All zones exhausted
    set_attempts "$job" $((attempts + 1))
    log "FAILED: $job could not be launched in any zone (attempt $((attempts + 1))/$MAX_ATTEMPTS)"
    return 1
}

# Main check — run once for all jobs
check_all() {
    for job in "${JOBS[@]}"; do
        local vm
        vm=$(vm_name "$job")

        local zone
        zone=$(find_vm_zone "$vm")

        if [ -z "$zone" ]; then
            log "$vm not found — skipping (not launched yet?)"
            continue
        fi

        local status
        status=$(vm_status "$vm" "$zone")

        case "$status" in
            RUNNING)
                log "$vm is RUNNING in $zone"
                ;;
            STAGING|PROVISIONING)
                log "$vm is $status in $zone — starting up"
                ;;
            TERMINATED)
                if was_preempted "$vm" "$zone"; then
                    log "$vm was PREEMPTED in $zone — recovering..."
                    recover_job "$job" || true
                else
                    log "$vm is TERMINATED in $zone (not preempted — manual stop or completed)"
                fi
                ;;
            SUSPENDED|STOPPING)
                log "$vm is $status in $zone — unusual state"
                ;;
            *)
                log "$vm has unknown status: $status"
                ;;
        esac
    done
}

# Parse args
LOOP=false
INTERVAL=300

while [ $# -gt 0 ]; do
    case "$1" in
        --loop) LOOP=true; shift ;;
        --interval) INTERVAL="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

if $LOOP; then
    log "Starting recovery loop (interval: ${INTERVAL}s)"
    while true; do
        check_all
        sleep "$INTERVAL"
    done
else
    check_all
fi
