#!/usr/bin/env python3
"""Comprehensive Sonata TTS diagnostic — check every stage for bugs.

Inspects:
  1. Weight statistics (random init vs trained)
  2. LM architecture match (config vs actual weights)
  3. Flow architecture match
  4. Decoder architecture match
  5. FSQ decomposition correctness
  6. End-to-end Python inference
"""

import json
import os
import sys
import numpy as np
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "train", "sonata"))

import torch
from safetensors import safe_open


def fmt(x):
    return f"{x:.6f}" if abs(x) < 100 else f"{x:.2f}"


def check_weights(path, label):
    """Check weight statistics to determine if model is trained or random init."""
    print(f"\n{'='*60}")
    print(f"  {label}: {os.path.basename(path)}")
    print(f"  Size: {os.path.getsize(path) / 1e6:.1f} MB")
    print(f"{'='*60}")

    tensors = {}
    with safe_open(path, framework="pt", device="cpu") as f:
        for name in f.keys():
            tensors[name] = f.get_tensor(name)

    print(f"  Tensors: {len(tensors)}")

    # Categorize tensors
    weights = {}
    biases = {}
    embeddings = {}
    norms = {}
    other = {}

    for name, t in tensors.items():
        if "bias" in name.lower():
            biases[name] = t
        elif "embed" in name.lower() or "emb" in name.lower():
            embeddings[name] = t
        elif "norm" in name.lower() or "gamma" in name.lower():
            norms[name] = t
        elif "weight" in name.lower():
            weights[name] = t
        else:
            other[name] = t

    print(f"  Weights: {len(weights)}, Biases: {len(biases)}, "
          f"Embeddings: {len(embeddings)}, Norms: {len(norms)}, Other: {len(other)}")

    # Check bias statistics (key indicator of training)
    n_zero_bias = 0
    n_nonzero_bias = 0
    bias_stats = []
    for name, t in biases.items():
        t_np = t.float().numpy()
        mean = float(np.mean(t_np))
        std = float(np.std(t_np))
        amax = float(np.max(np.abs(t_np)))
        is_zero = amax < 1e-6
        if is_zero:
            n_zero_bias += 1
        else:
            n_nonzero_bias += 1
        bias_stats.append((name, t.shape, mean, std, amax, is_zero))

    print(f"\n  Bias check: {n_zero_bias} zero, {n_nonzero_bias} non-zero")
    if n_nonzero_bias > 0:
        print(f"  → TRAINED: some biases have non-zero values")
        for name, shape, mean, std, amax, is_zero in sorted(bias_stats, key=lambda x: -x[4])[:5]:
            print(f"    {name} {list(shape)}: mean={fmt(mean)} std={fmt(std)} max={fmt(amax)}")
    else:
        print(f"  → RANDOM INIT: all biases are zero")

    # Check weight statistics
    print(f"\n  Weight statistics (top 5 by std):")
    weight_stats = []
    for name, t in weights.items():
        t_np = t.float().numpy()
        mean = float(np.mean(t_np))
        std = float(np.std(t_np))
        weight_stats.append((name, list(t.shape), mean, std))

    for name, shape, mean, std in sorted(weight_stats, key=lambda x: -x[3])[:5]:
        print(f"    {name} {shape}: mean={fmt(mean)} std={fmt(std)}")

    # Check embedding statistics
    if embeddings:
        print(f"\n  Embedding statistics:")
        for name, t in embeddings.items():
            t_np = t.float().numpy()
            mean = float(np.mean(t_np))
            std = float(np.std(t_np))
            print(f"    {name} {list(t.shape)}: mean={fmt(mean)} std={fmt(std)}")

    # Report all tensor names and shapes
    print(f"\n  All tensors ({len(tensors)}):")
    for name in sorted(tensors.keys()):
        t = tensors[name]
        print(f"    {name:50s} {str(list(t.shape)):20s} {t.dtype}")

    return tensors


def check_architecture_match(tensors, expected_prefix, expected_layers):
    """Verify tensor names match expected architecture."""
    missing = []
    unexpected = []
    tensor_names = set(tensors.keys())

    for exp in expected_layers:
        if exp not in tensor_names:
            missing.append(exp)

    known_prefixes = set()
    for exp in expected_layers:
        parts = exp.rsplit(".", 1)
        if len(parts) > 1:
            known_prefixes.add(parts[0])

    for name in tensor_names:
        if name not in expected_layers:
            unexpected.append(name)

    if missing:
        print(f"\n  MISSING tensors ({len(missing)}):")
        for m in missing[:10]:
            print(f"    - {m}")
    if unexpected:
        print(f"\n  UNEXPECTED tensors ({len(unexpected)}):")
        for u in unexpected[:10]:
            print(f"    + {u}")
    if not missing and not unexpected:
        print(f"\n  ✓ All {len(expected_layers)} tensors match expected architecture")

    return missing, unexpected


def check_fsq_decomposition():
    """Verify FSQ decomposition matches between Python and Rust."""
    print(f"\n{'='*60}")
    print(f"  FSQ Decomposition Check")
    print(f"{'='*60}")

    levels = [8, 8, 8, 8]
    codebook_size = 1
    for l in levels:
        codebook_size *= l

    n_errors = 0
    for idx in range(codebook_size):
        # Python decomposition (from codec.py indices_to_codes)
        py_codes = []
        remainder = idx
        for d in reversed(range(len(levels))):
            py_codes.append((remainder % levels[d]) - (levels[d] - 1) / 2.0)
            remainder = remainder // levels[d]
        py_codes = list(reversed(py_codes))

        # Rust decomposition (from lib.rs)
        rust_codes = [0.0] * len(levels)
        remainder = idx
        for d in reversed(range(len(levels))):
            level = levels[d]
            code_val = float(remainder % level)
            half = (level - 1.0) / 2.0
            rust_codes[d] = code_val - half
            remainder = remainder // level

        for d in range(len(levels)):
            if abs(py_codes[d] - rust_codes[d]) > 1e-6:
                n_errors += 1
                if n_errors <= 3:
                    print(f"  MISMATCH at idx={idx}: py={py_codes} rust={rust_codes}")
                break

    if n_errors == 0:
        print(f"  ✓ All {codebook_size} indices decompose identically (Python == Rust)")
    else:
        print(f"  ✗ {n_errors}/{codebook_size} mismatches found!")

    # Also verify roundtrip: codes → index → codes
    print(f"\n  Roundtrip check (index → codes → index):")
    n_rt_errors = 0
    for idx in range(codebook_size):
        codes = []
        remainder = idx
        for d in reversed(range(len(levels))):
            codes.insert(0, remainder % levels[d])
            remainder //= levels[d]
        # Reconstruct index
        reconstructed = 0
        for d in range(len(levels)):
            reconstructed = reconstructed * levels[d] + codes[d]
        if reconstructed != idx:
            n_rt_errors += 1

    if n_rt_errors == 0:
        print(f"  ✓ All {codebook_size} indices roundtrip correctly")
    else:
        print(f"  ✗ {n_rt_errors} roundtrip failures!")


def build_expected_flow_tensors(cfg):
    """Build expected tensor names for the flow model."""
    names = []
    names.append("semantic_emb.weight")
    names.extend([f"time_emb.mlp.{i}.weight" for i in [0, 2]])
    names.extend([f"time_emb.mlp.{i}.bias" for i in [0, 2]])
    names.extend(["cond_proj.weight", "cond_proj.bias"])
    names.extend(["input_proj.weight", "input_proj.bias"])
    names.extend(["output_proj.weight", "output_proj.bias"])

    for i in range(cfg.get("n_layers", 8)):
        p = f"blocks.{i}"
        names.extend([f"{p}.norm1.proj.weight", f"{p}.norm1.proj.bias"])
        names.extend([f"{p}.attn.qkv.weight", f"{p}.attn.qkv.bias"])
        names.extend([f"{p}.attn.out.weight", f"{p}.attn.out.bias"])
        names.extend([f"{p}.norm2.proj.weight", f"{p}.norm2.proj.bias"])
        names.extend([f"{p}.ff.0.weight", f"{p}.ff.0.bias"])
        names.extend([f"{p}.ff.2.weight", f"{p}.ff.2.bias"])

    if cfg.get("n_speakers", 0) > 0:
        names.extend(["speaker_emb.weight", "speaker_proj.weight", "speaker_proj.bias"])
    return names


def build_expected_decoder_tensors(cfg):
    """Build expected tensor names for the ConvDecoder."""
    names = []
    names.extend(["decoder.input_proj.0.weight", "decoder.input_proj.0.bias"])

    n_layers = cfg.get("dec_n_layers", 8)
    for i in range(n_layers):
        p = f"decoder.backbone.{i}"
        names.extend([f"{p}.dwconv.weight", f"{p}.dwconv.bias"])
        names.extend([f"{p}.norm.weight", f"{p}.norm.bias"])
        names.extend([f"{p}.pwconv1.weight", f"{p}.pwconv1.bias"])
        names.extend([f"{p}.pwconv2.weight", f"{p}.pwconv2.bias"])
        names.append(f"{p}.gamma")

    strides = [8, 5, 4, 3]
    d = cfg.get("dec_dim", 512)
    channels = [d, d // 2, d // 4, d // 8, d // 16]
    for i in range(len(strides)):
        p = f"decoder.upsample.{i}"
        names.extend([f"{p}.upsample.weight", f"{p}.upsample.bias"])
        for j in range(3):
            names.extend([f"{p}.residuals.{j}.net.1.weight", f"{p}.residuals.{j}.net.1.bias"])
            names.extend([f"{p}.residuals.{j}.net.3.weight", f"{p}.residuals.{j}.net.3.bias"])

    names.extend(["decoder.output.1.weight", "decoder.output.1.bias"])
    return names


def build_expected_lm_tensors(cfg):
    """Build expected tensor names for the Sonata LM."""
    names = []
    names.extend(["text_embed.weight", "semantic_embed.weight"])
    names.extend(["final_norm.weight"])

    n_layers = cfg.get("n_layers", 16)
    for i in range(n_layers):
        p = f"layers.{i}"
        names.extend([f"{p}.attn_norm.weight"])
        names.extend([f"{p}.wq.weight", f"{p}.wk.weight", f"{p}.wv.weight", f"{p}.wo.weight"])
        names.extend([f"{p}.ffn_norm.weight"])
        names.extend([f"{p}.w1.weight", f"{p}.w2.weight", f"{p}.w3.weight"])

    names.append("output_head.weight")
    return names


def run_python_e2e():
    """Run full end-to-end Python inference and inspect each stage."""
    print(f"\n{'='*60}")
    print(f"  End-to-End Python Inference")
    print(f"{'='*60}")

    from config import FlowConfig, CodecConfig
    from flow import SonataFlow
    from codec import ConvDecoder

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"  Device: {device}")

    # Load Flow
    flow_cfg_path = os.path.join(ROOT, "models/sonata/sonata_flow_config.json")
    flow_wt_path = os.path.join(ROOT, "models/sonata/sonata_flow.safetensors")
    with open(flow_cfg_path) as f:
        fcfg = json.load(f)
    flow_config = FlowConfig(**{k: v for k, v in fcfg.items() if hasattr(FlowConfig, k)})
    flow = SonataFlow(flow_config).to(device).eval()

    from safetensors.torch import load_file
    flow_sd = load_file(flow_wt_path)
    flow.load_state_dict(flow_sd, strict=False)
    print(f"  Flow loaded: {sum(p.numel() for p in flow.parameters())/1e6:.1f}M params")

    # Load Decoder
    dec_cfg_path = os.path.join(ROOT, "models/sonata/sonata_decoder_config.json")
    dec_wt_path = os.path.join(ROOT, "models/sonata/sonata_decoder.safetensors")
    with open(dec_cfg_path) as f:
        dcfg = json.load(f)
    codec_cfg = CodecConfig(
        fsq_levels=dcfg.get("fsq_levels", [8, 8, 8, 8]),
        dec_dim=dcfg.get("dec_dim", 512),
        dec_n_layers=dcfg.get("dec_n_layers", 8),
        dec_conv_kernel=dcfg.get("dec_conv_kernel", 7),
        dec_ff_mult=dcfg.get("dec_ff_mult", 4.0),
        acoustic_dim=dcfg.get("acoustic_dim", 256),
        decoder_type="conv",
    )
    decoder = ConvDecoder(codec_cfg).to(device).eval()
    dec_sd = load_file(dec_wt_path)
    stripped = {k.replace("decoder.", "", 1) if k.startswith("decoder.") else k: v
                for k, v in dec_sd.items()}
    decoder.load_state_dict(stripped, strict=False)
    print(f"  Decoder loaded: {sum(p.numel() for p in decoder.parameters())/1e6:.1f}M params")

    # Test with deterministic tokens
    levels = [8, 8, 8, 8]
    codebook_size = 1
    for l in levels:
        codebook_size *= l

    torch.manual_seed(42)
    np.random.seed(42)
    n_frames = 50

    sem_ids = np.random.randint(0, codebook_size, size=n_frames)
    sem_idx = torch.from_numpy(sem_ids).unsqueeze(0).long().to(device)

    print(f"\n  Semantic tokens: {sem_ids[:10]}... (range: {sem_ids.min()}-{sem_ids.max()})")

    # Stage 1: Semantic embedding
    with torch.no_grad():
        sem_emb = flow.semantic_emb(sem_idx)
    print(f"  Semantic embedding: {sem_emb.shape}")
    sem_emb_cpu = sem_emb.float().cpu().numpy()
    print(f"    mean={np.mean(sem_emb_cpu):.6f} std={np.std(sem_emb_cpu):.6f} "
          f"min={np.min(sem_emb_cpu):.6f} max={np.max(sem_emb_cpu):.6f}")

    # Stage 2: Flow ODE → acoustic latents
    torch.manual_seed(42)
    with torch.no_grad():
        acoustic = flow.sample(sem_idx, n_steps=8)
    print(f"  Acoustic latents: {acoustic.shape}")
    ac_cpu = acoustic.float().cpu().numpy()
    print(f"    mean={np.mean(ac_cpu):.6f} std={np.std(ac_cpu):.6f} "
          f"min={np.min(ac_cpu):.4f} max={np.max(ac_cpu):.4f}")

    # Stage 3: FSQ decompose
    from codec import FSQ
    fsq = FSQ(levels)
    fsq_codes = fsq.indices_to_codes(torch.from_numpy(sem_ids).long()).unsqueeze(0).to(device)
    print(f"  FSQ codes: {fsq_codes.shape}")
    fc_cpu = fsq_codes.float().cpu().numpy()
    print(f"    mean={np.mean(fc_cpu):.6f} std={np.std(fc_cpu):.6f} "
          f"range=[{np.min(fc_cpu):.1f}, {np.max(fc_cpu):.1f}]")

    # Stage 4: Decoder → audio
    with torch.no_grad():
        audio = decoder(fsq_codes, acoustic)
    audio_np = audio.squeeze().float().cpu().numpy()
    print(f"  Audio: {audio_np.shape} ({len(audio_np)/24000:.2f}s at 24kHz)")

    rms = float(np.sqrt(np.mean(audio_np ** 2)))
    peak = float(np.max(np.abs(audio_np)))
    zc = int(np.sum(np.diff(np.sign(audio_np)) != 0))
    zcr = zc / max(len(audio_np), 1)

    print(f"    RMS={rms:.4f}  peak={peak:.4f}  ZCR={zcr:.4f}")

    if rms < 0.001:
        print(f"    → SILENCE: audio is essentially silent")
    elif zcr > 0.25:
        print(f"    → NOISE-LIKE: ZCR too high for speech (expected 0.01-0.20)")
    elif zcr < 0.01:
        print(f"    → DC/LOW-FREQ: ZCR too low")
    else:
        print(f"    → POTENTIALLY SPEECH: ZCR in normal range")

    # Check if acoustic latents are just noise
    print(f"\n  Diagnostic: is the Flow model just passing through noise?")
    torch.manual_seed(42)
    noise = torch.randn(1, n_frames, flow_config.acoustic_dim, device=device)
    noise_cpu = noise.float().cpu().numpy()
    corr = np.corrcoef(ac_cpu.flatten(), noise_cpu.flatten())[0, 1]
    print(f"    Correlation(acoustic, initial_noise): {corr:.4f}")
    if abs(corr) > 0.8:
        print(f"    → FLOW NOT WORKING: output is highly correlated with input noise")
        print(f"    → This means the flow network isn't transforming the noise into structured latents")
    elif abs(corr) > 0.3:
        print(f"    → FLOW PARTIALLY WORKING: moderate correlation with noise")
    else:
        print(f"    → FLOW IS TRANSFORMING: low correlation with initial noise ✓")


def main():
    print(f"\n{'#'*60}")
    print(f"  SONATA TTS DIAGNOSTIC")
    print(f"  Checking every stage for bugs")
    print(f"{'#'*60}")

    # 1. Check weight statistics
    lm_path = os.path.join(ROOT, "models/sonata/sonata_lm.safetensors")
    flow_path = os.path.join(ROOT, "models/sonata/sonata_flow.safetensors")
    dec_path = os.path.join(ROOT, "models/sonata/sonata_decoder.safetensors")

    lm_cfg_path = os.path.join(ROOT, "models/sonata/sonata_lm_config.json")
    flow_cfg_path = os.path.join(ROOT, "models/sonata/sonata_flow_config.json")
    dec_cfg_path = os.path.join(ROOT, "models/sonata/sonata_decoder_config.json")

    with open(lm_cfg_path) as f:
        lm_cfg = json.load(f)
    with open(flow_cfg_path) as f:
        flow_cfg = json.load(f)
    with open(dec_cfg_path) as f:
        dec_cfg = json.load(f)

    lm_tensors = check_weights(lm_path, "Sonata LM")
    flow_tensors = check_weights(flow_path, "Sonata Flow")
    dec_tensors = check_weights(dec_path, "Sonata Decoder")

    # 2. Check architecture matches
    print(f"\n{'='*60}")
    print(f"  Architecture Match Check")
    print(f"{'='*60}")

    print(f"\n  --- LM ---")
    expected_lm = build_expected_lm_tensors(lm_cfg)
    check_architecture_match(lm_tensors, "lm", expected_lm)

    print(f"\n  --- Flow ---")
    expected_flow = build_expected_flow_tensors(flow_cfg)
    check_architecture_match(flow_tensors, "flow", expected_flow)

    print(f"\n  --- Decoder ---")
    expected_dec = build_expected_decoder_tensors(dec_cfg)
    check_architecture_match(dec_tensors, "decoder", expected_dec)

    # 3. FSQ decomposition check
    check_fsq_decomposition()

    # 4. End-to-end Python inference
    try:
        run_python_e2e()
    except Exception as e:
        print(f"\n  E2E ERROR: {e}")
        import traceback
        traceback.print_exc()

    # 5. Summary
    print(f"\n{'#'*60}")
    print(f"  DIAGNOSTIC SUMMARY")
    print(f"{'#'*60}\n")

    print(f"  Model configs:")
    print(f"    LM:  d={lm_cfg['d_model']}, L={lm_cfg['n_layers']}, "
          f"semantic_vocab={lm_cfg['semantic_vocab_size']}")
    print(f"    Flow: d={flow_cfg['d_model']}, L={flow_cfg['n_layers']}, "
          f"acoustic_dim={flow_cfg['acoustic_dim']}, "
          f"semantic_vocab={flow_cfg['semantic_vocab_size']}")
    print(f"    Dec:  d={dec_cfg['dec_dim']}, L={dec_cfg['dec_n_layers']}, "
          f"fsq_levels={dec_cfg['fsq_levels']}, "
          f"acoustic_dim={dec_cfg['acoustic_dim']}")

    vocab_match = lm_cfg['semantic_vocab_size'] == flow_cfg['semantic_vocab_size']
    fsq_codebook = 1
    for l in dec_cfg['fsq_levels']:
        fsq_codebook *= l
    vocab_fsq_match = lm_cfg['semantic_vocab_size'] == fsq_codebook
    acoustic_match = flow_cfg['acoustic_dim'] == dec_cfg['acoustic_dim']

    print(f"\n  Config consistency:")
    print(f"    LM semantic_vocab == Flow semantic_vocab: {vocab_match} "
          f"({lm_cfg['semantic_vocab_size']} vs {flow_cfg['semantic_vocab_size']})")
    print(f"    LM semantic_vocab == FSQ codebook: {vocab_fsq_match} "
          f"({lm_cfg['semantic_vocab_size']} vs {fsq_codebook})")
    print(f"    Flow acoustic_dim == Dec acoustic_dim: {acoustic_match} "
          f"({flow_cfg['acoustic_dim']} vs {dec_cfg['acoustic_dim']})")


if __name__ == "__main__":
    main()
