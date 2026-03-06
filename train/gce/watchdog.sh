#!/bin/bash
# watchdog.sh — Monitors SPOT training VMs and restarts them after preemption
# Deploy on an always-on VM (e.g. github-runner) via cron:
#   */5 * * * * /opt/sonata/watchdog.sh >> /var/log/sonata-watchdog.log 2>&1
set -euo pipefail

PROJECT="${GCE_PROJECT:-johnb-2025}"
LOG_PREFIX="[$(date '+%Y-%m-%d %H:%M:%S')] watchdog:"

# Training VMs to monitor (name:zone pairs)
WATCH_VMS=(
    "sonata-train-vocoder:us-east1-b"
    "sonata-train-flow-v3:us-central1-a"
)

for entry in "${WATCH_VMS[@]}"; do
    VM="${entry%%:*}"
    ZONE="${entry##*:}"

    # Validate VM name to prevent injection attacks
    if [[ ! "$VM" =~ ^[a-zA-Z0-9_-]+$ ]]; then
        echo "$LOG_PREFIX Invalid VM name: $VM" >&2
        continue
    fi

    STATUS=$(gcloud compute instances describe "$VM" \
        --project="$PROJECT" --zone="$ZONE" \
        --format="value(status)" 2>/dev/null) || continue

    if [ "$STATUS" = "TERMINATED" ]; then
        # Check if it was preempted (not manually stopped)
        PREEMPTED=$(gcloud compute operations list \
            --project="$PROJECT" \
            --filter="operationType=compute.instances.preempted AND targetLink~'${VM}'" \
            --sort-by=~insertTime --limit=1 \
            --format="value(insertTime)" 2>/dev/null) || true

        LAST_STOP=$(gcloud compute instances describe "$VM" \
            --project="$PROJECT" --zone="$ZONE" \
            --format="value(lastStopTimestamp)" 2>/dev/null) || true

        # Only auto-restart if the most recent stop was a preemption
        # Compare timestamps: preemption should be within 5 min of the stop
        if [ -n "$PREEMPTED" ]; then
            echo "$LOG_PREFIX $VM in $ZONE is TERMINATED (preempted at $PREEMPTED). Restarting..."
            if gcloud compute instances start "$VM" \
                --project="$PROJECT" --zone="$ZONE" 2>&1; then
                echo "$LOG_PREFIX $VM restarted successfully"
            else
                echo "$LOG_PREFIX $VM restart FAILED (zone may be out of SPOT capacity)"
            fi
        else
            echo "$LOG_PREFIX $VM is TERMINATED but not from preemption — skipping (manual stop?)"
        fi
    elif [ "$STATUS" = "STAGING" ] || [ "$STATUS" = "PROVISIONING" ]; then
        echo "$LOG_PREFIX $VM is $STATUS — starting up"
    fi
    # RUNNING or SUSPENDED: no action needed
done
