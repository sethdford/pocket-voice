#!/usr/bin/env python3
"""
Convert a NeMo Conformer-CTC encoder to CoreML (.mlmodelc) for BNNSGraph execution.

Usage:
  pip install nemo_toolkit[asr] coremltools
  python scripts/convert_nemo_coreml.py nvidia/parakeet-ctc-0.6b -o models/conformer_ctc_0.6b.mlmodelc
"""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile

def main():
    parser = argparse.ArgumentParser(description="Convert NeMo Conformer to CoreML .mlmodelc")
    parser.add_argument("model", help="NeMo model name (e.g. nvidia/parakeet-ctc-0.6b) or .nemo file")
    parser.add_argument("-o", "--output", required=True, help="Output .mlmodelc path")
    parser.add_argument("--seq-len", type=int, default=512, help="Trace sequence length")
    parser.add_argument("--n-mels", type=int, default=80, help="Number of mel bins")
    args = parser.parse_args()

    import torch
    import nemo.collections.asr as nemo_asr
    import coremltools as ct

    print(f"Loading NeMo model: {args.model}")
    if args.model.endswith(".nemo"):
        model = nemo_asr.models.EncDecCTCModelBPE.restore_from(args.model)
    else:
        model = nemo_asr.models.EncDecCTCModelBPE.from_pretrained(args.model)

    model.eval()
    model.cpu()

    class ConformerEncoderCTC(torch.nn.Module):
        """Wraps just the encoder + CTC head (skips preprocessor — our C code does mel)."""
        def __init__(self, nemo_model):
            super().__init__()
            self.encoder = nemo_model.encoder
            self.decoder = nemo_model.decoder

        def forward(self, mel_features):
            length = torch.tensor([mel_features.shape[2]], dtype=torch.long)
            encoded, _ = self.encoder(audio_signal=mel_features, length=length)
            log_probs = self.decoder(encoder_output=encoded)
            return log_probs

    wrapper = ConformerEncoderCTC(model)
    wrapper.eval()

    T = args.seq_len
    example_input = torch.randn(1, args.n_mels, T)

    print(f"Tracing model with input shape [1, {args.n_mels}, {T}]...")
    with torch.no_grad():
        traced = torch.jit.trace(wrapper, example_input)

    print("Converting traced model → CoreML .mlpackage...")
    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(name="mel_features", shape=(1, args.n_mels, ct.RangeDim(1, 4096, T)))],
        outputs=[ct.TensorType(name="log_probs")],
        convert_to="mlprogram",
        compute_units=ct.ComputeUnit.ALL,
        minimum_deployment_target=ct.target.macOS15,
    )

    mlpackage_out = args.output.replace(".mlmodelc", ".mlpackage")

    if os.path.exists(mlpackage_out):
        shutil.rmtree(mlpackage_out)
    mlmodel.save(mlpackage_out)
    print(f"  Saved .mlpackage: {mlpackage_out}")

    output_dir = args.output
    has_compiler = subprocess.run(["xcrun", "-f", "coremlcompiler"],
                                  capture_output=True).returncode == 0

    if has_compiler:
        print("Compiling → .mlmodelc via xcrun coremlcompiler...")
        with tempfile.TemporaryDirectory() as tmpdir:
            result = subprocess.run(
                ["xcrun", "coremlcompiler", "compile", mlpackage_out, tmpdir],
                capture_output=True, text=True
            )
            if result.returncode != 0:
                print(f"WARNING: coremlcompiler failed:\n{result.stderr}")
                print(f"  .mlpackage saved at: {mlpackage_out}")
                print(f"  Compile manually: xcrun coremlcompiler compile {mlpackage_out} models/")
            else:
                if os.path.exists(output_dir):
                    shutil.rmtree(output_dir)
                for item in os.listdir(tmpdir):
                    if item.endswith(".mlmodelc"):
                        shutil.copytree(os.path.join(tmpdir, item), output_dir)
                        break
    else:
        print("NOTE: coremlcompiler not found (requires Xcode)")
        print(f"  .mlpackage saved at: {mlpackage_out}")
        print(f"  Compile manually: xcrun coremlcompiler compile {mlpackage_out} models/")

    size_mb = sum(
        os.path.getsize(os.path.join(dp, f))
        for dp, _, fns in os.walk(output_dir)
        for f in fns
    ) / (1024 * 1024)

    print(f"\nConversion complete!")
    print(f"  Output: {output_dir} ({size_mb:.1f} MB)")
    print(f"\nUsage in C:")
    print(f'  bnns_conformer_load_mlmodelc(bc, "{output_dir}");')


if __name__ == "__main__":
    main()
