# GCE Training Infrastructure Design

## Overview

Move Sonata model training (flow v3, vocoder) to GCE preemptible A100 VMs for ~10x speedup at ~$10-15 per training run.

## Architecture

- GCS bucket for persistent data + checkpoints
- Preemptible a2-highgpu-1g VM (1x A100 40GB)
- Auto-restart on preemption via instance group or startup script
- gcsfuse to mount GCS as filesystem

## Scripts

| File                         | Purpose                                       |
| ---------------------------- | --------------------------------------------- |
| `train/gce/setup.sh`         | Create GCS bucket, upload data + checkpoints  |
| `train/gce/launch.sh`        | Create preemptible A100 VM, start training    |
| `train/gce/train_wrapper.sh` | On-VM: auto-resume training, sync checkpoints |
| `train/gce/monitor.sh`       | Local: tail logs, check status                |
| `train/gce/teardown.sh`      | Stop VM, download checkpoints                 |

## Cost Estimate

- Preemptible A100: ~$1.20/hr
- Flow v3 (120K steps): ~8 hours = ~$10
- Vocoder (full run): ~12 hours = ~$15
- GCS storage (250GB): ~$5/month
- Data transfer: ~$5 egress

## Data Transfer

- Upload: 241GB training data + 7.2GB checkpoint to GCS (~1-2 hrs)
- Mount via gcsfuse on VM (transparent filesystem access)
- Checkpoint sync: every save, copy to GCS for durability
