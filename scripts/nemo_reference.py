#!/usr/bin/env python3
"""
nemo_reference.py — Run NeMo inference on the same audio as the C engine.

Usage:
    python scripts/nemo_reference.py validation/real_speech.pcm
"""
import sys
import struct
import numpy as np
import torch


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/nemo_reference.py <audio.pcm>", file=sys.stderr)
        sys.exit(1)

    pcm = np.fromfile(sys.argv[1], dtype=np.float32)
    sr = 16000
    print(f"Audio: {len(pcm)} samples ({len(pcm)/sr:.1f}s)")
    print(f"PCM stats: min={pcm.min():.4f} max={pcm.max():.4f} mean={pcm.mean():.6f}")

    import nemo.collections.asr as nemo_asr

    print("\nLoading NeMo model...")
    model = nemo_asr.models.ASRModel.from_pretrained("nvidia/parakeet-tdt_ctc-110m")
    model.eval()
    model.freeze()

    audio_t = torch.from_numpy(pcm).unsqueeze(0).float()
    audio_len = torch.tensor([len(pcm)])

    with torch.no_grad():
        # Preprocessor (mel spectrogram + normalization)
        processed, processed_len = model.preprocessor(
            input_signal=audio_t, length=audio_len)
        mel = processed.squeeze(0).cpu().numpy()
        print(f"\nMel features: shape={mel.shape} min={mel.min():.4f} max={mel.max():.4f} "
              f"mean={mel.mean():.6f} std={mel.std():.4f}")
        print(f"  Frame 0 (first 10): {mel[0,:10]}")

        # Encoder
        encoded, encoded_len = model.encoder(
            audio_signal=processed, length=processed_len)
        enc = encoded.squeeze(0).cpu().numpy()
        print(f"\nEncoder output: shape={enc.shape} min={enc.min():.4f} max={enc.max():.4f} "
              f"mean={enc.mean():.6f}")

        # CTC logits
        if hasattr(model, 'ctc_decoder'):
            logits = model.ctc_decoder(encoder_output=encoded)
        elif hasattr(model, 'decoder'):
            logits = model.decoder(encoder_output=encoded)
        else:
            print("No CTC decoder found!")
            sys.exit(1)

        log_probs = logits.squeeze(0).cpu().numpy()
        print(f"\nCTC logits: shape={log_probs.shape}")

        # Print argmax per frame
        print("\nPer-frame argmax (first 20 frames):")
        blank_id = log_probs.shape[1] - 1
        for t in range(min(20, log_probs.shape[0])):
            row = log_probs[t]
            argmax = np.argmax(row)
            print(f"  t={t:3d}: argmax={argmax:4d} (val={row[argmax]:.3f}) "
                  f"blank_val={row[blank_id]:.3f}")

        # Greedy decode
        prev = -1
        tokens = []
        for t in range(log_probs.shape[0]):
            idx = np.argmax(log_probs[t])
            if idx != blank_id and idx != prev:
                tokens.append(idx)
            prev = idx

    # Decode from vocab file
    vocab_path = "model.vocab"
    try:
        with open(vocab_path) as vf:
            vocab = [line.rstrip('\n') for line in vf]
        pieces = [vocab[i] if i < len(vocab) else f"<{i}>" for i in tokens]
        text = "".join(pieces).replace("▁", " ").strip()
    except Exception:
        text = str(tokens)

    print(f"\nNeMo greedy CTC transcript: \"{text}\"")
    print(f"Token IDs: {tokens}")

    # Also try the full transcribe API
    print("\n--- Full model.transcribe() ---")
    import tempfile, soundfile as sf
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
        sf.write(f.name, pcm, sr)
        result = model.transcribe([f.name])
        if isinstance(result, list):
            r = result[0]
            if hasattr(r, 'text'):
                print(f"Transcribe result: \"{r.text}\"")
            else:
                print(f"Transcribe result: \"{r}\"")
        else:
            print(f"Transcribe result: \"{result}\"")

    # Save reference logits for C comparison
    print("\n--- Saving reference data ---")
    np.save("validation/ref_mel.npy", mel)
    np.save("validation/ref_logits.npy", log_probs)
    print(f"Saved ref_mel.npy {mel.shape} and ref_logits.npy {log_probs.shape}")

    # Print more subsampling debug info
    print("\n--- Subsampling weights stats ---")
    sub_keys = [k for k in model.state_dict().keys() if 'pre_encode' in k]
    for k in sorted(sub_keys):
        w = model.state_dict()[k]
        print(f"  {k}: shape={list(w.shape)} min={w.min():.4f} max={w.max():.4f}")

    # Run encoder forward with hooks to capture post-subsampling
    hook_data = {}
    def capture_hook(name):
        def hook(module, input, output):
            if isinstance(output, tuple):
                hook_data[name] = output[0].detach().cpu()
            else:
                hook_data[name] = output.detach().cpu()
        return hook

    model.encoder.pre_encode.register_forward_hook(capture_hook("pre_encode"))
    with torch.no_grad():
        encoded2, _ = model.encoder(audio_signal=processed, length=processed_len)

    if "pre_encode" in hook_data:
        sub_out = hook_data["pre_encode"].squeeze(0).numpy()
        print(f"\nPost-subsampling output: shape={sub_out.shape} "
              f"min={sub_out.min():.4f} max={sub_out.max():.4f} "
              f"mean={sub_out.mean():.6f}")


if __name__ == "__main__":
    main()
