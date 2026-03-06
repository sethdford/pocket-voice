#!/bin/bash
# reset_training.sh — Clean reset for Sonata training
#
# Removes training artifacts (checkpoints, logs, encoded data) while preserving
# manifests and raw data by default. Use --all to also remove manifests and
# raw data for a full redownload.
#
# Usage:
#   bash reset_training.sh              # Default: clean artifacts, ask confirm
#   bash reset_training.sh --dry-run    # Show what would be deleted
#   bash reset_training.sh --yes        # Skip confirmation prompt
#   bash reset_training.sh --all        # Also remove manifests + raw data
#   bash reset_training.sh --all --yes   # Full reset, no prompts
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

DRY_RUN=0
ALL=0
YES=0

while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run) DRY_RUN=1; shift ;;
        --all)     ALL=1; shift ;;
        --yes|-y)  YES=1; shift ;;
        -h|--help)
            echo "reset_training.sh — Clean reset for Sonata training"
            echo ""
            echo "Usage: $0 [--dry-run] [--all] [--yes|-y]"
            echo ""
            echo "Options:"
            echo "  --dry-run    Show what would be deleted, do not delete"
            echo "  --all        Also remove manifests and raw data (requires redownload)"
            echo "  --yes, -y    Skip confirmation prompt"
            exit 0
            ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

# ── Paths to clean (always) ────────────────────────────────────────────────────
CLEAN_DIRS=(
    "train/checkpoints"
    "train/checkpoints_test"
    "train/sonata/checkpoints"
    "train/sonata/checkpoints_test"
    "train/sonata/__pycache__"
    "wandb"
    "tensorboard"
)

# Encoded data patterns
CLEAN_ENCODED=(
    "train/data/encoded_*.pt"
    "train/data/encoded_*.shards.txt"
    "train/sonata/data/encoded_*.pt"
    "train/sonata/data/encoded_*.shards.txt"
    "train/sonata/checkpoints/encoded_*.pt"
    "train/sonata/checkpoints/encoded_*.shards.txt"
)

# ── Paths to clean only with --all ────────────────────────────────────────────
ALL_DIRS=(
    "train/data/manifest_*.jsonl"
    "train/data/_old_manifests"
    "train/data/librispeech"
    "train/data/libritts-r"
    "train/data/vctk"
    "train/data/ljspeech"
    "train/data/hifi-tts"
    "train/data/common-voice"
    "train/data/resampled"
    "train/sonata/data/manifest_*.jsonl"
    "train/sonata/data/_old_manifests"
    "train/sonata/data/librispeech"
    "train/sonata/data/libritts-r"
    "train/sonata/data/vctk"
    "train/sonata/data/ljspeech"
    "train/sonata/data/hifi-tts"
    "train/sonata/data/common-voice"
    "train/sonata/data/resampled"
)

delete_item() {
    local item="$1"
    if [[ $DRY_RUN -eq 1 ]]; then
        echo "  [dry-run] would remove: $item"
    else
        if [[ -d "$item" ]]; then
            rm -rf "$item"
            echo "  removed dir:  $item"
        else
            rm -f "$item"
            echo "  removed file: $item"
        fi
    fi
}

# Collect everything to delete
TO_DELETE=()
for d in "${CLEAN_DIRS[@]}"; do
    [[ -e "$d" ]] && TO_DELETE+=("$d")
done
for pat in "${CLEAN_ENCODED[@]}"; do
    for f in $pat; do
        [[ -e "$f" ]] && TO_DELETE+=("$f")
    done
done
if [[ $ALL -eq 1 ]]; then
    for pat in "${ALL_DIRS[@]}"; do
        for f in $pat; do
            [[ -e "$f" ]] && TO_DELETE+=("$f")
        done
    done
fi

# Deduplicate
TO_DELETE=($(printf '%s\n' "${TO_DELETE[@]}" | sort -u))

if [[ ${#TO_DELETE[@]} -eq 0 ]]; then
    echo "Nothing to delete. Training tree is already clean."
    exit 0
fi

echo "═══════════════════════════════════════════════════════════════"
echo "  Sonata Training Reset"
echo "═══════════════════════════════════════════════════════════════"
if [[ $ALL -eq 1 ]]; then
    echo "  Mode: FULL (manifests + raw data will be removed)"
else
    echo "  Mode: artifacts only (manifests + raw data preserved)"
fi
echo ""
echo "  Items to remove (${#TO_DELETE[@]}):"
for item in "${TO_DELETE[@]}"; do
    echo "    $item"
done
echo "═══════════════════════════════════════════════════════════════"

if [[ $DRY_RUN -eq 1 ]]; then
    echo ""
    echo "Dry run complete. Run without --dry-run to actually delete."
    exit 0
fi

if [[ $YES -ne 1 ]]; then
    echo ""
    read -p "Proceed? [y/N] " -r
    [[ $REPLY =~ ^[Yy]$ ]] || { echo "Aborted."; exit 0; }
fi

echo ""
for item in "${TO_DELETE[@]}"; do
    delete_item "$item"
done

echo ""
echo "Reset complete."
