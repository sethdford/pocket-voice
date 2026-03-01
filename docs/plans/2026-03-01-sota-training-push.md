# SOTA Training Push — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Generalize GCE training infrastructure to support 3 new model training jobs (speaker encoder, 12.5Hz codec, flow distillation), then launch VM2 to begin training.

**Architecture:** Extend `launch.sh` and `train_wrapper.sh` to dispatch multiple job types with correct CLI args. Add `--codec-version` flag to `train_codec.py`. Launch on-demand L4 VM2 chaining speaker_encoder → codec_12hz, then later swap VM1 to flow distillation after flow v3 completes.

**Tech Stack:** Bash (GCE scripts), Python (training scripts), gcloud CLI, gsutil

---

### Task 1: Add `--codec-version` flag to train_codec.py

**Files:**

- Modify: `train/sonata/train_codec.py:529-581` (argparse section)
- Modify: `train/sonata/train_codec.py:231-246` (config loading section)

**Step 1: Add the CLI argument**

In the argparse block (~line 581, after last `add_argument`), add:

```python
    parser.add_argument("--codec-version", default="50hz", choices=["50hz", "12hz"],
                        help="Codec version: 50hz (default) or 12hz (12.5 Hz frame rate)")
```

**Step 2: Import Codec12HzConfig and SonataCodec12Hz**

At top of file (~line 29), change:

```python
from config import CodecConfig
```

to:

```python
from config import CodecConfig, Codec12HzConfig
from codec_12hz import SonataCodec12Hz, Codec12HzLoss
```

**Step 3: Update config selection logic**

Replace lines 239-246 (the `else` branch when not resuming):

```python
    else:
        if args.codec_version == "12hz":
            cfg = Codec12HzConfig()
        else:
            cfg = CodecConfig(
                enc_dim=args.enc_dim,
                enc_n_layers=args.enc_n_layers,
                dec_dim=args.dec_dim,
                dec_n_layers=args.dec_n_layers,
                decoder_type=args.decoder_type,
            )
```

**Step 4: Update model instantiation**

Replace line 248:

```python
    model = SonataCodec(cfg).to(device)
```

with:

```python
    if args.codec_version == "12hz":
        model = SonataCodec12Hz(cfg).to(device)
    else:
        model = SonataCodec(cfg).to(device)
```

Also update the resume config loading (lines 231-237) to handle 12hz configs:

```python
    if args.resume:
        _ckpt_peek = torch.load(args.resume, map_location="cpu", weights_only=False)
        cfg_dict = _ckpt_peek.get("config", {})
        if args.codec_version == "12hz":
            cfg = Codec12HzConfig(**{k: v for k, v in cfg_dict.items()
                                     if k in Codec12HzConfig.__dataclass_fields__})
        else:
            cfg = CodecConfig(**{k: v for k, v in cfg_dict.items()
                                 if k in CodecConfig.__dataclass_fields__})
        print(f"  Config loaded from checkpoint: {args.resume}")
        del _ckpt_peek
```

**Step 5: Verify import works**

Run: `cd train/sonata && python3 -c "from train_codec import *; print('OK')"`

**Step 6: Commit**

```bash
git add train/sonata/train_codec.py
git commit -m "feat: add --codec-version 12hz flag to train_codec.py"
```

---

### Task 2: Add speaker_encoder job to train_wrapper.sh

**Files:**

- Modify: `train/gce/train_wrapper.sh`

**Step 1: Add speaker_encoder case**

After the `vocoder)` case block (after line ~109), add a new case before `*)`):

```bash
        speaker_encoder)
            JOB_CKPT_DIR="$CKPT_DIR/speaker_encoder"
            GCS_CKPT_DIR="$BUCKET/checkpoints/speaker_encoder"
            mkdir -p "$JOB_CKPT_DIR"

            # Find latest checkpoint
            RESUME=$(find_latest_checkpoint "$JOB_CKPT_DIR" "speaker_encoder_epoch*.pt")
            RESUME_FLAG=""
            if [ -n "$RESUME" ]; then
                echo "  Resuming from: $RESUME"
                RESUME_FLAG="--resume $RESUME"
            else
                echo "  Starting from scratch"
            fi

            python3 -u train_speaker_encoder.py \
                --data-dir "$DATA_DIR/libritts-r/LibriTTS_R" \
                --output-dir "$JOB_CKPT_DIR" \
                --device "$DEVICE" \
                --batch-size 32 \
                --n-epochs 100 \
                --lr 0.001 &
            TRAIN_PID=$!
            ;;
```

Note: `train_speaker_encoder.py` doesn't have a `--resume` flag, so we skip it. The script auto-saves best checkpoint. If we need resume, that's a future enhancement.

**Step 2: Verify syntax**

Run: `bash -n train/gce/train_wrapper.sh`
Expected: No output (clean parse)

**Step 3: Commit**

```bash
git add train/gce/train_wrapper.sh
git commit -m "feat: add speaker_encoder job to train_wrapper.sh"
```

---

### Task 3: Add codec_12hz job to train_wrapper.sh

**Files:**

- Modify: `train/gce/train_wrapper.sh`

**Step 1: Add codec_12hz case**

After the `speaker_encoder)` case:

```bash
        codec_12hz)
            JOB_CKPT_DIR="$CKPT_DIR/codec_12hz"
            GCS_CKPT_DIR="$BUCKET/checkpoints/codec_12hz"
            mkdir -p "$JOB_CKPT_DIR"

            RESUME=$(find_latest_checkpoint "$JOB_CKPT_DIR" "sonata_codec_step_*.pt")
            if [ -z "$RESUME" ]; then
                RESUME=$(find_latest_checkpoint "$JOB_CKPT_DIR" "sonata_codec_best.pt")
            fi

            RESUME_FLAG=""
            if [ -n "$RESUME" ]; then
                echo "  Resuming from: $RESUME"
                RESUME_FLAG="--resume $RESUME"
            else
                echo "  Starting from scratch"
            fi

            python3 -u train_codec.py \
                --codec-version 12hz \
                --manifest "$DATA_DIR/libritts_r_full_manifest.jsonl" \
                --checkpoint-dir "$JOB_CKPT_DIR" \
                --device "$DEVICE" \
                --steps 200000 \
                --batch-size 8 \
                --grad-accum 4 \
                --lr 2e-4 \
                --warmup 2000 \
                --segment-sec 2.0 \
                --gan-start-step 50000 \
                --adv-weight 1.0 \
                --fm-weight 2.0 \
                --entropy-weight 0.1 \
                --wavlm-weight 0.5 \
                --ema-decay 0.999 \
                --progressive \
                --augment \
                --save-every 5000 \
                --num-workers 4 \
                $RESUME_FLAG &
            TRAIN_PID=$!
            ;;
```

**Step 2: Verify syntax**

Run: `bash -n train/gce/train_wrapper.sh`

**Step 3: Commit**

```bash
git add train/gce/train_wrapper.sh
git commit -m "feat: add codec_12hz job to train_wrapper.sh"
```

---

### Task 4: Add distill_v3 job to train_wrapper.sh

**Files:**

- Modify: `train/gce/train_wrapper.sh`

**Step 1: Add distill_v3 case**

After `codec_12hz)` case:

```bash
        distill_v3)
            JOB_CKPT_DIR="$CKPT_DIR/flow_v3_distilled"
            GCS_CKPT_DIR="$BUCKET/checkpoints/flow_v3_distilled"
            TEACHER_DIR="$CKPT_DIR/flow_v3_large_fixed"
            mkdir -p "$JOB_CKPT_DIR"

            # Find teacher checkpoint (must exist)
            TEACHER=$(find_latest_checkpoint "$TEACHER_DIR" "flow_v3_best.pt")
            if [ -z "$TEACHER" ]; then
                TEACHER=$(find_latest_checkpoint "$TEACHER_DIR" "flow_v3_step_*.pt")
            fi
            if [ -z "$TEACHER" ]; then
                echo "ERROR: No flow v3 teacher checkpoint found in $TEACHER_DIR"
                exit 1
            fi
            echo "  Teacher: $TEACHER"

            RESUME=$(find_latest_checkpoint "$JOB_CKPT_DIR" "distilled_step_*.pt")
            RESUME_FLAG=""
            if [ -n "$RESUME" ]; then
                echo "  Resuming from: $RESUME"
                # distill_v3 doesn't have --resume, will start fresh
            fi

            python3 -u train_distill_v3.py \
                --teacher-checkpoint "$TEACHER" \
                --manifest "$DATA_DIR/libritts_r_full_manifest.jsonl" \
                --output-dir "$JOB_CKPT_DIR" \
                --device "$DEVICE" \
                --steps 50000 \
                --batch-size 4 \
                --teacher-steps 8 \
                --model-size large \
                --phonemes \
                --lr 1e-4 \
                --save-every 5000 \
                --val-every 5000 \
                --num-workers 4 &
            TRAIN_PID=$!
            ;;
```

**Step 2: Verify syntax**

Run: `bash -n train/gce/train_wrapper.sh`

**Step 3: Commit**

```bash
git add train/gce/train_wrapper.sh
git commit -m "feat: add distill_v3 job to train_wrapper.sh"
```

---

### Task 5: Generalize launch.sh for new job types

**Files:**

- Modify: `train/gce/launch.sh`

**Step 1: Update usage string and validation**

Change line 5 from:

```bash
JOB="${1:?Usage: ./launch.sh <flow_v3|vocoder> [--on-demand]}"
```

to:

```bash
JOB="${1:?Usage: ./launch.sh <flow_v3|vocoder|speaker_encoder|codec_12hz|distill_v3> [--on-demand]}"
```

**Step 2: Validate job name**

After line 7 (`ON_DEMAND` assignment), add:

```bash
case "$JOB" in
    flow_v3|vocoder|speaker_encoder|codec_12hz|distill_v3) ;;
    *) echo "ERROR: Unknown job '$JOB'. Use: flow_v3, vocoder, speaker_encoder, codec_12hz, distill_v3"; exit 1 ;;
esac
```

**Step 3: Verify syntax**

Run: `bash -n train/gce/launch.sh`

**Step 4: Commit**

```bash
git add train/gce/launch.sh
git commit -m "feat: generalize launch.sh for 5 job types"
```

---

### Task 6: Generalize teardown.sh and monitor.sh for new jobs

**Files:**

- Modify: `train/gce/teardown.sh`
- Modify: `train/gce/monitor.sh`

**Step 1: Update teardown.sh case block**

Replace lines 18-31 with expanded job routing:

```bash
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
```

**Step 2: Update monitor.sh checkpoint listing**

In the `--status` mode's checkpoint listing section (~lines 34-43), expand the case:

```bash
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
```

Also update the `--loss` mode case block similarly for loss file paths.

**Step 3: Verify syntax**

Run: `bash -n train/gce/teardown.sh && bash -n train/gce/monitor.sh`

**Step 4: Commit**

```bash
git add train/gce/teardown.sh train/gce/monitor.sh
git commit -m "feat: generalize teardown.sh and monitor.sh for all job types"
```

---

### Task 7: Upload updated code and launch VM2

**Step 1: Upload all updated scripts to GCS**

```bash
GCE_PROJECT=johnb-2025
gsutil -m rsync -r -x '__pycache__/.*|.*\.pyc' train/sonata/ gs://sonata-training-johnb-2025/code/train/sonata/
gsutil -m cp train/gce/train_wrapper.sh gs://sonata-training-johnb-2025/code/train/gce/
```

**Step 2: Launch VM2 (on-demand) for speaker encoder**

```bash
GCE_PROJECT=johnb-2025 bash train/gce/launch.sh speaker_encoder --on-demand
```

**Step 3: Verify VM2 startup**

Wait ~90 seconds, then:

```bash
GCE_PROJECT=johnb-2025 gcloud compute ssh sonata-train-speaker-encoder --zone=us-central1-a \
    --command="tail -20 /var/log/sonata-startup.log"
```

**Step 4: Verify training is running**

Wait ~60 more seconds:

```bash
GCE_PROJECT=johnb-2025 gcloud compute ssh sonata-train-speaker-encoder --zone=us-central1-a \
    --command="tail -10 /var/log/sonata-training.log && echo '---' && nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader"
```

**Step 5: Commit all changes**

```bash
git add -A train/gce/ train/sonata/train_codec.py
git commit -m "feat: SOTA training push — generalize GCE infra for 5 job types, launch VM2"
```

---

### Task 8: Plan VM2 job chaining (speaker_encoder → codec_12hz)

After speaker encoder completes (~12h), VM2 needs to switch to codec_12hz training.

**Option A (manual):** SSH into VM2, kill wrapper, restart with new job:

```bash
GCE_PROJECT=johnb-2025 gcloud compute ssh sonata-train-speaker-encoder --zone=us-central1-a
# On VM:
sudo kill $(pgrep -f train_wrapper)
sudo nohup /opt/sonata/train_wrapper.sh codec_12hz gs://sonata-training-johnb-2025 > /var/log/sonata-training.log 2>&1 &
```

**Option B (automated):** Add job chaining to train_wrapper.sh. After the training loop completes successfully for speaker_encoder, check for a `NEXT_JOB` metadata attribute and re-exec:

Add to the end of train_wrapper.sh (before the "Exceeded max retries" line):

```bash
# ─── Job chaining ────────────────────────────────────────────────────────────
NEXT_JOB=$(curl -sf -H "Metadata-Flavor: Google" \
    http://metadata.google.internal/computeMetadata/v1/instance/attributes/next-job 2>/dev/null || true)

if [ -n "$NEXT_JOB" ]; then
    echo "[$(date)] Job $JOB complete. Chaining to next job: $NEXT_JOB"
    exec "$0" "$NEXT_JOB" "$BUCKET"
fi
```

Then launch VM2 with metadata:

```bash
# In launch.sh, add next-job to metadata when applicable:
--metadata="bucket=$BUCKET,job=$JOB,next-job=codec_12hz"
```

**Recommended: Option B** — automated chaining avoids manual intervention at 3am.

**Step 1: Add chaining to train_wrapper.sh**

Insert after the successful completion `exit 0` (line 143):

Replace:

```bash
    if [ $EXIT_CODE -eq 0 ]; then
        echo "[$(date)] Training completed successfully!"
        echo "  Checkpoints at: $GCS_CKPT_DIR"
        exit 0
    fi
```

With:

```bash
    if [ $EXIT_CODE -eq 0 ]; then
        echo "[$(date)] Training completed successfully!"
        echo "  Checkpoints at: $GCS_CKPT_DIR"

        # Check for chained next job
        NEXT_JOB=$(curl -sf -H "Metadata-Flavor: Google" \
            http://metadata.google.internal/computeMetadata/v1/instance/attributes/next-job 2>/dev/null || true)
        if [ -n "$NEXT_JOB" ]; then
            echo "[$(date)] Chaining to next job: $NEXT_JOB"
            exec "$0" "$NEXT_JOB" "$BUCKET"
        fi

        exit 0
    fi
```

**Step 2: Add --next-job flag to launch.sh**

After the `ON_DEMAND` parsing, add:

```bash
NEXT_JOB="${3:-}"
[[ "$NEXT_JOB" == "--next-job="* ]] && NEXT_JOB="${NEXT_JOB#--next-job=}"
# Handle --next-job as 3rd positional or flag
for arg in "$@"; do
    [[ "$arg" == "--next-job="* ]] && NEXT_JOB="${arg#--next-job=}"
done
```

Update metadata line:

```bash
METADATA="bucket=$BUCKET,job=$JOB"
[ -n "$NEXT_JOB" ] && METADATA="$METADATA,next-job=$NEXT_JOB"
```

Then use `--metadata="$METADATA"` in the gcloud create command.

**Step 3: Verify and commit**

```bash
bash -n train/gce/train_wrapper.sh && bash -n train/gce/launch.sh
git add train/gce/train_wrapper.sh train/gce/launch.sh
git commit -m "feat: add job chaining (--next-job) for sequential training on same VM"
```

---

### Task 9: Launch VM2 with chained jobs

**Step 1: Upload final scripts**

```bash
GCE_PROJECT=johnb-2025
gsutil -m rsync -r -x '__pycache__/.*|.*\.pyc' train/sonata/ gs://sonata-training-johnb-2025/code/train/sonata/
gsutil -m cp train/gce/train_wrapper.sh gs://sonata-training-johnb-2025/code/train/gce/
```

**Step 2: Launch VM2 with chaining**

```bash
GCE_PROJECT=johnb-2025 bash train/gce/launch.sh speaker_encoder --on-demand --next-job=codec_12hz
```

This will:

1. Train speaker encoder (~12h)
2. Automatically chain to 12.5Hz codec training (~3-4 days)
3. No manual intervention needed

**Step 3: Verify startup and training**

```bash
# After ~2 minutes:
GCE_PROJECT=johnb-2025 gcloud compute ssh sonata-train-speaker-encoder --zone=us-central1-a \
    --command="grep -E 'step|epoch|loss|Training|Starting' /var/log/sonata-training.log | tail -10"
```

---

### Task 10: Monitor and swap VM1 to distillation (after flow v3 completes)

This is a manual step to execute when flow v3 finishes (~4.6 days from now).

**Step 1: Check flow v3 completion**

```bash
GCE_PROJECT=johnb-2025 bash train/gce/monitor.sh flow_v3 --status
```

Look for: training process exited, `flow_v3_best.pt` exists in GCS.

**Step 2: Download flow v3 checkpoints**

```bash
GCE_PROJECT=johnb-2025 bash train/gce/teardown.sh flow_v3 --download-only
```

**Step 3: Delete flow v3 VM and launch distillation**

```bash
GCE_PROJECT=johnb-2025 bash train/gce/teardown.sh flow_v3 --delete-all
GCE_PROJECT=johnb-2025 bash train/gce/launch.sh distill_v3
```

**Step 4: Verify distillation starts**

```bash
# After ~2 minutes:
GCE_PROJECT=johnb-2025 gcloud compute ssh sonata-train-distill-v3 --zone=us-central1-a \
    --command="tail -20 /var/log/sonata-training.log"
```

Expected: Teacher checkpoint loaded, consistency distillation training steps progressing.
