#!/bin/bash
# download_all.sh — Download all speech datasets for Sonata training.
#
# Runs parallel curl downloads for maximum throughput, then:
#   1. Extracts archives
#   2. Builds unified manifest
#   3. Preprocesses (resample to 24kHz, filter)
#   4. Encodes with Sonata Codec
#
# Usage:
#   bash train/sonata/download_all.sh
#   bash train/sonata/download_all.sh --skip-encode   # Download + manifest only
#   bash train/sonata/download_all.sh --small          # Dev splits only (~2GB)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DATA_DIR="${SCRIPT_DIR}/../../train/data"
ARCHIVE_DIR="${DATA_DIR}/_archives"

log() { echo "[$(date +%H:%M:%S)] $*"; }

SKIP_ENCODE=0
SMALL=0
CODEC_CKPT="${SCRIPT_DIR}/../../train/checkpoints/codec/sonata_codec_final.pt"
DEVICE="mps"

while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-encode) SKIP_ENCODE=1; shift;;
        --small) SMALL=1; shift;;
        --codec-ckpt) CODEC_CKPT="$2"; shift 2;;
        --device) DEVICE="$2"; shift 2;;
        --data-dir) DATA_DIR="$2"; ARCHIVE_DIR="$DATA_DIR/_archives"; shift 2;;
        *) echo "Unknown: $1"; exit 1;;
    esac
done

mkdir -p "$ARCHIVE_DIR"

echo "╔══════════════════════════════════════════════════════════╗"
echo "║  SONATA DATA ACQUISITION PIPELINE                       ║"
echo "╠══════════════════════════════════════════════════════════╣"
echo "║  Data dir:  $DATA_DIR"
echo "║  Mode:      $([ $SMALL -eq 1 ] && echo 'SMALL (dev splits)' || echo 'FULL (~950h)')"
echo "║  Encode:    $([ $SKIP_ENCODE -eq 0 ] && echo ON || echo OFF)"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""

# ── Download URLs ─────────────────────────────────────────────────────────────
LIBRITTS_R_BASE="https://openslr.trmal.net/resources/141"
LJSPEECH_URL="https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2"
VCTK_URL="https://datashare.ed.ac.uk/bitstream/handle/10283/3443/VCTK-Corpus-0.92.zip"

download() {
    local url="$1" dest="$2" name="$3"
    if [[ -f "$dest" && $(stat -f%z "$dest" 2>/dev/null || stat -c%s "$dest" 2>/dev/null) -gt 1000 ]]; then
        log "[skip] $name already downloaded ($(du -sh "$dest" | cut -f1))"
        return 0
    fi
    log "Downloading $name ..."
    curl -L -C - --progress-bar -o "$dest" "$url"
    log "[done] $name ($(du -sh "$dest" | cut -f1))"
}

# ── Step 1: Parallel Downloads ────────────────────────────────────────────────
log "Step 1: Downloading datasets..."
echo ""

if [[ $SMALL -eq 1 ]]; then
    # Small mode: just dev-clean splits
    download "$LIBRITTS_R_BASE/dev_clean.tar.gz" "$ARCHIVE_DIR/libritts_r_dev_clean.tar.gz" "LibriTTS-R dev-clean (1.3GB)"
    download "$LJSPEECH_URL" "$ARCHIVE_DIR/LJSpeech-1.1.tar.bz2" "LJSpeech (2.6GB)"
else
    # Full mode: download in parallel (4 concurrent)
    PIDS=()

    download "$LIBRITTS_R_BASE/train_clean_100.tar.gz" "$ARCHIVE_DIR/libritts_r_train_clean_100.tar.gz" "LibriTTS-R train-clean-100 (8.1GB)" &
    PIDS+=($!)

    download "$LIBRITTS_R_BASE/train_clean_360.tar.gz" "$ARCHIVE_DIR/libritts_r_train_clean_360.tar.gz" "LibriTTS-R train-clean-360 (28GB)" &
    PIDS+=($!)

    download "$LJSPEECH_URL" "$ARCHIVE_DIR/LJSpeech-1.1.tar.bz2" "LJSpeech (2.6GB)" &
    PIDS+=($!)

    download "$VCTK_URL" "$ARCHIVE_DIR/VCTK-Corpus-0.92.zip" "VCTK (11GB)" &
    PIDS+=($!)

    # Wait for all downloads
    FAIL=0
    for pid in "${PIDS[@]}"; do
        wait "$pid" || FAIL=$((FAIL+1))
    done

    if [[ $FAIL -gt 0 ]]; then
        log "WARNING: $FAIL download(s) failed. Continuing with available data."
    fi
fi

echo ""
log "Downloads complete. Disk usage:"
du -sh "$ARCHIVE_DIR" 2>/dev/null || true
echo ""

# ── Step 2: Extract ───────────────────────────────────────────────────────────
log "Step 2: Extracting archives..."

extract() {
    local archive="$1" dest="$2" marker="$3"
    if [[ -f "$marker" ]]; then
        log "[skip] Already extracted: $(basename "$archive")"
        return
    fi
    if [[ ! -f "$archive" ]]; then
        log "[skip] Archive not found: $(basename "$archive")"
        return
    fi
    mkdir -p "$dest"
    log "Extracting $(basename "$archive") ..."
    case "$archive" in
        *.tar.gz|*.tgz) tar -xzf "$archive" -C "$dest" ;;
        *.tar.bz2)      tar -xjf "$archive" -C "$dest" ;;
        *.zip)           unzip -qo "$archive" -d "$dest" ;;
    esac
    touch "$marker"
    log "[done] Extracted to $dest"
}

LIBRITTS_DIR="$DATA_DIR/libritts-r"
LJSPEECH_DIR="$DATA_DIR/ljspeech"
VCTK_DIR="$DATA_DIR/vctk"
mkdir -p "$LIBRITTS_DIR" "$LJSPEECH_DIR" "$VCTK_DIR"

if [[ $SMALL -eq 1 ]]; then
    extract "$ARCHIVE_DIR/libritts_r_dev_clean.tar.gz" "$LIBRITTS_DIR" "$LIBRITTS_DIR/.dev_clean_extracted"
else
    extract "$ARCHIVE_DIR/libritts_r_train_clean_100.tar.gz" "$LIBRITTS_DIR" "$LIBRITTS_DIR/.train_clean_100_extracted"
    extract "$ARCHIVE_DIR/libritts_r_train_clean_360.tar.gz" "$LIBRITTS_DIR" "$LIBRITTS_DIR/.train_clean_360_extracted"
fi
extract "$ARCHIVE_DIR/LJSpeech-1.1.tar.bz2" "$LJSPEECH_DIR" "$LJSPEECH_DIR/.extracted"
extract "$ARCHIVE_DIR/VCTK-Corpus-0.92.zip" "$VCTK_DIR" "$VCTK_DIR/.extracted"

echo ""
log "Extraction complete. Disk usage:"
du -sh "$LIBRITTS_DIR" "$LJSPEECH_DIR" "$VCTK_DIR" 2>/dev/null || true
echo ""

# ── Step 3: Build Manifest ────────────────────────────────────────────────────
log "Step 3: Building unified manifest..."
cd "$SCRIPT_DIR"
python3 acquire_data.py manifest --data-dir "$DATA_DIR" --output "$DATA_DIR/manifest_raw.jsonl"
echo ""

# ── Step 4: Preprocess ────────────────────────────────────────────────────────
log "Step 4: Preprocessing (resample to 24kHz, SNR filter)..."
python3 acquire_data.py preprocess \
    --manifest "$DATA_DIR/manifest_raw.jsonl" \
    --output "$DATA_DIR/manifest_clean.jsonl" \
    --resample-dir "$DATA_DIR/resampled" \
    --target-sr 24000 \
    --min-duration 0.5 \
    --max-duration 30.0 \
    --min-snr 10
echo ""

# ── Step 5: Stats ─────────────────────────────────────────────────────────────
log "Step 5: Dataset statistics..."
python3 acquire_data.py stats --manifest "$DATA_DIR/manifest_clean.jsonl"
echo ""

# ── Step 6: Encode (optional) ─────────────────────────────────────────────────
if [[ $SKIP_ENCODE -eq 0 ]]; then
    if [[ -f "$CODEC_CKPT" ]]; then
        log "Step 6: Encoding with Sonata Codec..."
        python3 data_pipeline.py \
            --source manifest \
            --manifest "$DATA_DIR/manifest_clean.jsonl" \
            --codec-ckpt "$CODEC_CKPT" \
            --output "$DATA_DIR/encoded.pt" \
            --shard-size 10000 \
            --device "$DEVICE"
    else
        log "Step 6: [skip] Codec checkpoint not found at $CODEC_CKPT"
        log "  Encode later with:"
        log "    python3 data_pipeline.py --source manifest \\"
        log "      --manifest $DATA_DIR/manifest_clean.jsonl \\"
        log "      --codec-ckpt <your_codec.pt> --output $DATA_DIR/encoded.pt"
    fi
else
    log "Step 6: [skip] Encoding skipped (--skip-encode)"
fi

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║  DATA ACQUISITION COMPLETE                               ║"
echo "╠══════════════════════════════════════════════════════════╣"
echo "║  Manifest:  $DATA_DIR/manifest_clean.jsonl"
if [[ $SKIP_ENCODE -eq 0 && -f "$CODEC_CKPT" ]]; then
echo "║  Encoded:   $DATA_DIR/encoded.pt (+ shards)"
fi
echo "║  Disk:      $(du -sh "$DATA_DIR" | cut -f1)"
echo "╚══════════════════════════════════════════════════════════╝"
