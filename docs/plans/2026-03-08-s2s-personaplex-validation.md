# PersonaPlex M4 Validation — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Download PersonaPlex 7B, quantize to INT4, run on Apple Silicon M4 to validate full-duplex S2S feasibility and measure real latency numbers.

**Architecture:** PersonaPlex 7B is an NVIDIA Moshi-derived full-duplex S2S model. We validate it on-device to get real latency baselines before building our custom 500M Talker.

**Tech Stack:** Python, MLX (Apple's ML framework), PersonaPlex weights (CC-BY-4.0), HuggingFace

---

### Task 1: Download and inspect PersonaPlex weights

**Step 1: Create validation directory**

```bash
mkdir -p experiments/personaplex
```

**Step 2: Write download script**

**Files:**

- Create: `experiments/personaplex/download.py`

```python
#!/usr/bin/env python3
"""Download PersonaPlex 7B weights from HuggingFace."""
import os
from huggingface_hub import snapshot_download

MODEL_ID = "nvidia/PersonaPlex-7B"
LOCAL_DIR = os.path.join(os.path.dirname(__file__), "weights")

def main():
    print(f"Downloading {MODEL_ID}...")
    path = snapshot_download(
        repo_id=MODEL_ID,
        local_dir=LOCAL_DIR,
        ignore_patterns=["*.md", "*.txt"],
    )
    print(f"Downloaded to: {path}")
    for f in sorted(os.listdir(path)):
        size_mb = os.path.getsize(os.path.join(path, f)) / (1024 * 1024)
        print(f"  {f}: {size_mb:.1f} MB")

if __name__ == "__main__":
    main()
```

**Step 3: Run download**

Run: `cd experiments/personaplex && python download.py`

**Step 4: Commit**

```bash
git add experiments/personaplex/download.py
git commit -m "feat: PersonaPlex 7B download script for S2S validation"
```

---

### Task 2: INT4 quantization via MLX

**Files:**

- Create: `experiments/personaplex/quantize_mlx.py`

**Step 1: Write quantization script**

```python
#!/usr/bin/env python3
"""Quantize PersonaPlex 7B to INT4 via MLX for Apple Silicon inference."""
import os

def main():
    try:
        import mlx.core as mx
        from mlx_lm import convert
    except ImportError:
        print("Install MLX: pip install mlx mlx-lm")
        return

    weights_dir = os.path.join(os.path.dirname(__file__), "weights")
    output_dir = os.path.join(os.path.dirname(__file__), "weights_int4")

    if not os.path.exists(weights_dir):
        print("Run download.py first")
        return

    print("Quantizing to INT4...")
    convert(weights_dir, quantize=True, q_bits=4, q_group_size=64)
    print(f"Quantized model saved to: {output_dir}")

if __name__ == "__main__":
    main()
```

**Step 2: Run quantization**

Run: `cd experiments/personaplex && python quantize_mlx.py`

**Step 3: Commit**

```bash
git add experiments/personaplex/quantize_mlx.py
git commit -m "feat: PersonaPlex INT4 quantization via MLX"
```

---

### Task 3: Latency benchmark on M4

**Files:**

- Create: `experiments/personaplex/benchmark.py`

**Step 1: Write benchmark**

Measure per-step inference latency on Apple Silicon. PersonaPlex claims 68ms/step on M-series. We need to verify and compare against our 100ms target for the 500M Talker.

Use MLX matmul benchmarks matching PersonaPlex's hidden dimension to estimate transformer throughput. Once the actual model API is understood, replace with real forward pass timing.

Frame budget: 80ms (12.5Hz frame rate).

**Step 2: Commit**

```bash
git add experiments/personaplex/benchmark.py
git commit -m "bench: PersonaPlex M4 latency benchmark for S2S validation"
```

---

### Task 4: Full-duplex audio test

**Files:**

- Create: `experiments/personaplex/test_duplex.py`

End-to-end test: feed synthetic audio (sine wave at 440Hz, 24kHz), chunk into 80ms frames (1920 samples), process through PersonaPlex, measure round-trip latency. This validates the full-duplex claim on real hardware.

**Step 1: Write test (depends on PersonaPlex's actual API)**

Generate test audio, chunk at 12.5Hz, measure processing latency per frame. Save results as JSON for comparison with our Talker benchmarks.

**Step 2: Commit**

```bash
git add experiments/personaplex/test_duplex.py
git commit -m "test: PersonaPlex full-duplex audio validation on M4"
```

---

## Summary

| Task | Component        | Files           | Notes                  |
| ---- | ---------------- | --------------- | ---------------------- |
| 1    | Download weights | download.py     | PersonaPlex 7B from HF |
| 2    | INT4 quantize    | quantize_mlx.py | MLX quantization       |
| 3    | Latency bench    | benchmark.py    | Per-step timing        |
| 4    | Full-duplex test | test_duplex.py  | Audio round-trip       |

**Expected Outcomes:**

- Validate full-duplex S2S works on M4
- Get real latency numbers (target <200ms, PersonaPlex claims 68ms/step)
- Identify quality gaps to address in custom 500M Talker
- Total cost: ~$0 (open weights, local inference)
