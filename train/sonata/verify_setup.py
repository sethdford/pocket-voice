#!/usr/bin/env python3
"""verify_setup.py — Sanity check for Sonata training environment.

Verifies:
  - Python dependencies (torch, safetensors, numpy)
  - All required .py modules are importable (no syntax errors)
  - Config dataclasses can be instantiated
  - Device is available (MPS/CUDA/CPU)
  - Disk space available

Usage:
  python train/sonata/verify_setup.py
  cd train/sonata && python verify_setup.py
"""

import importlib
import os
import shutil
import sys
from pathlib import Path

# Ensure train/sonata is on path
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
os.chdir(SCRIPT_DIR)

CHECKS_OK = True


def fail(msg: str) -> None:
    global CHECKS_OK
    CHECKS_OK = False
    print(f"✗ {msg}")


def ok(msg: str) -> None:
    print(f"✓ {msg}")


def main() -> None:
    print("Sonata Training — Environment Verification")
    print("=" * 50)

    # 1. Python dependencies
    deps = [
        ("torch", "PyTorch"),
        ("safetensors", "safetensors"),
        ("numpy", "numpy"),
    ]
    for mod, name in deps:
        try:
            m = importlib.import_module(mod)
            ver = getattr(m, "__version__", "?")
            ok(f"{name} {ver}")
        except ImportError as e:
            fail(f"{name} not installed: {e}")

    # 2. Device
    try:
        import torch

        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            ok("Device: MPS (Apple Silicon)")
        elif torch.cuda.is_available():
            ok(f"Device: CUDA ({torch.cuda.get_device_name(0)})")
        else:
            ok("Device: CPU (MPS/CUDA not available)")
    except Exception as e:
        fail(f"Device check failed: {e}")

    # 3. All .py modules importable
    py_files = sorted(SCRIPT_DIR.glob("*.py"))
    py_modules = [f.stem for f in py_files if f.name != "verify_setup.py"]
    importable = []
    failed = []

    for mod in py_modules:
        try:
            importlib.import_module(mod)
            importable.append(mod)
        except Exception as e:
            failed.append((mod, str(e)))

    if failed:
        for mod, err in failed:
            fail(f"Module '{mod}': {err}")
    else:
        ok(f"All {len(importable)} modules importable")

    # 4. Config dataclasses
    try:
        from config import (
            CodecConfig,
            SemanticLMConfig,
            FlowConfig,
            FlowLargeConfig,
            FlowV2Config,
            STTConfig,
            STTLargeConfig,
            RefinerConfig,
            SoundStormConfig,
        )

        configs = [
            CodecConfig(),
            SemanticLMConfig(),
            FlowConfig(),
            FlowLargeConfig(),
            FlowV2Config(),
            STTConfig(),
            STTLargeConfig(),
            RefinerConfig(),
            SoundStormConfig(),
        ]
        names = [c.__class__.__name__ for c in configs]
        ok(f"Configs: {', '.join(names[:5])}... ({len(configs)} total)")
    except Exception as e:
        fail(f"Config instantiation: {e}")

    # 5. Disk space
    try:
        stat = shutil.disk_usage(REPO_ROOT)
        free_gb = stat.free / (1024**3)
        ok(f"{free_gb:.1f} GB free disk space")
    except Exception as e:
        fail(f"Disk space check: {e}")

    print("=" * 50)
    if CHECKS_OK:
        print("Ready to train!")
        sys.exit(0)
    else:
        print("Some checks failed. Fix the issues above before training.")
        sys.exit(1)


if __name__ == "__main__":
    main()
