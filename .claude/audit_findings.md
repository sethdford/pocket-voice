# Sonata Training Configuration & Wiring Audit

## Executive Summary

Found 8 configuration inconsistencies and wiring mismatches across the training pipeline:

- **2 P0 (Critical)**: Runtime failures if these configs are used as-is
- **2 P1 (High)**: E2E pipeline breakage when specific models/codecs used
- **2 P2 (Medium)**: Silent data quality degradation, future codec breaks
- **2 P3 (Low)**: Design debt, inconsistent naming

---

## P0 — Critical (Must Fix Before Any Training Run)

### P0.1: FlowV3LargeConfig speaker_dim mismatch → shape mismatch crash

**Location**: `/Users/sethford/Documents/pocket-voice/train/sonata/config.py:420`

**Issue**:

- FlowV3LargeConfig sets `speaker_dim: int = 384` (line 420)
- But speaker encoder (ECAPA-TDNN) outputs fixed `embedding_dim: int = 256` (train_speaker_encoder.py:226)
- When Flow model creates speaker embeddings table: `nn.Embedding(n_speakers, 384)`
- But encoder produces 256-dim vectors → shape mismatch in embedding lookup

**Evidence**:

```python
# config.py line 420 (FlowV3LargeConfig)
speaker_dim: int = 384  # ← WRONG

# config.py line 113 (FlowV3Config base)
speaker_dim: int = 256  # ← CORRECT

# train_speaker_encoder.py line 226 (ECAPA-TDNN)
class EcapaTdnn(nn.Module):
    def __init__(self, n_mels: int = 80, embedding_dim: int = 256, ...):
        # ...
        self.final_linear = nn.Linear(last_ch * 2, embedding_dim)  # = 256
```

**Impact**: Training will crash with "expected shape [256] but got [384]" when processing speaker embeddings.

**Fix**:

```python
# config.py line 420, change from:
speaker_dim: int = 384
# to:
speaker_dim: int = 256
```

---

### P0.2: VoicePromptWrapper dimension assertion failure → init crash

**Location**: `/Users/sethford/Documents/pocket-voice/train/sonata/voice_prompt.py:91-96`

**Issue**:

- ReferenceEncoder.**init** has default `output_dim: int = 512` (line 42)
- FlowConfig.ref_audio_dim has default `int = 80` (config.py:128)
- VoicePromptWrapper asserts they match (line 91)
- Assertion will always fail: 512 ≠ 80

**Evidence**:

```python
# voice_prompt.py line 42
def __init__(self, mel_dim: int = 80, hidden_dim: int = 512, output_dim: int = 512, ...):
    # ...
    self.output_proj = nn.Linear(hidden_dim, output_dim)  # output_dim defaults to 512

# voice_prompt.py line 91-96
class VoicePromptWrapper(nn.Module):
    def __init__(self, flow_model, ref_encoder: ReferenceEncoder, mel_extractor=None):
        super().__init__()
        ref_out = ref_encoder.output_proj.out_features  # = 512
        flow_expects = flow_model.cfg.ref_audio_dim     # = 80 (default)
        assert ref_out == flow_expects  # AssertionError: 512 != 80
```

**Impact**: Any attempt to use voice prompting (zero-shot voice cloning) will crash at initialization.

**Fix** (recommended):
Change ReferenceEncoder output_dim default in voice_prompt.py:42:

```python
def __init__(self, mel_dim: int = 80, hidden_dim: int = 512, output_dim: int = 80,  # changed from 512
             n_layers: int = 6, kernel_size: int = 5):
```

Rationale:

- ref_audio_dim=80 makes semantic sense: directly conditions mel generation (mel is 80-dim)
- ReferenceEncoder doesn't need 512-dim bottleneck; 80-dim is more efficient
- Reduces cross-attention compute: (B, T_mel, 80) instead of (B, T_mel, 512)

---

## P1 — High Priority (Fix Before Codec12Hz or Large Model Use)

### P1.1: Codec12HzConfig mel dimension mismatch → E2E pipeline break

**Location**: `/Users/sethford/Documents/pocket-voice/train/sonata/config.py:436`

**Issue**:

- Codec12HzConfig has `n_mels: int = 160` (line 436)
- All downstream models expect `n_mels: int = 80`:
  - FlowV3Config.mel_dim = 80
  - VocoderConfig.n_mels = 80
  - mel_utils.mel_spectrogram() defaults to 80
- Codec12Hz will encode/decode using 160-dim mels
- Flow and Vocoder won't accept 160-dim input

**Evidence**:

```python
# config.py line 436 (Codec12HzConfig)
n_mels: int = 160  # 2x normal for 80ms frames

# config.py line 345 (FlowV3Config)
mel_dim: int = 80

# config.py line 545 (VocoderConfig)
n_mels: int = 80

# mel_utils.py line 53
def mel_spectrogram(..., n_mels: int = 80, ...):
```

**Impact**: If Codec12Hz training ever completes, inference will fail:

- Codec outputs 160-dim mel
- Flow expects 80-dim input → shape error
- Silent breakage until inference attempt

**Fix**:
Option A (Recommended): Use same mel binning as 50Hz codec

```python
# config.py line 436, change from:
n_mels: int = 160
# to:
n_mels: int = 80
```

Justification: Codec with longer frames (80ms vs 20ms) doesn't need more mel bins—can use same filterbank. Longer frames = more temporal context, not more frequency resolution.

Option B (If SOTA requires it): Make mel_dim dynamic across Flow/Vocoder

- Update FlowV3Config to read codec n_mels at runtime
- Update VocoderConfig similarly
- Update data pipelines to detect and adapt
- Document why 160 bins are necessary

**Recommendation**: Use Option A unless there's published evidence that 160 bins significantly improve codec quality. Current design is incomplete.

---

### P1.2: FlowV3LargeConfig instantiation path (dependent on P0.1)

**Location**: `/Users/sethford/Documents/pocket-voice/train/sonata/train_flow_v3.py:535-536`

**Issue**:
When user runs: `python train_flow_v3.py --model-size large`
Code creates: `FlowV3LargeConfig(n_speakers=args.n_speakers)`
Which has broken speaker_dim (per P0.1)

**Evidence**:

```python
# train_flow_v3.py line 535-536
if args.model_size == "large":
    cfg = FlowV3LargeConfig(n_speakers=args.n_speakers)  # creates large config
    # cfg.speaker_dim = 384 (WRONG, per P0.1)
```

**Impact**: Large model training will crash if speaker conditioning enabled.

**Fix**: Fix P0.1 first (speaker_dim=256), then no additional changes needed here.

---

## P2 — Medium Priority (Fix Before Quality Release)

### P2.1: Speaker encoder sample rate mismatch → silent audio degradation

**Location**:

- `/Users/sethford/Documents/pocket-voice/train/sonata/train_speaker_encoder.py:53` (16 kHz)
- `/Users/sethford/Documents/pocket-voice/train/sonata/voice_prompt.py:131` (24 kHz)

**Issue**:

- train_speaker_encoder.py hardcodes `sample_rate: int = 16000` (line 53)
- VoicePromptDataset hardcodes `sample_rate: int = 24000` (line 131)
- If speaker encoder trained at 16kHz but voice_prompt loads audio at 24kHz:
  - Audio gets resampled from 24kHz → 16kHz silently
  - Mel spectrogram features degrade
  - Model learns features from degraded audio

**Evidence**:

```python
# train_speaker_encoder.py line 53
class MelSpectrogramExtractor(nn.Module):
    def __init__(self, sample_rate: int = 16000):
        # ...

# voice_prompt.py line 131
class VoicePromptDataset(torch.utils.data.Dataset):
    def __init__(self, manifest_path: str, encoded_data_path: str,
                 ref_duration: float = 5.0, sample_rate: int = 24000, ...):
        # ...
```

**Impact**: Low-moderate. Speaker encoder may underperform if voice_prompt audio not resampled to match training.

**Fix**:
Option A: Update voice_prompt to match speaker encoder sample rate

```python
# voice_prompt.py line 131, change from:
sample_rate: int = 24000,
# to:
sample_rate: int = 16000,
```

Option B: Retrain speaker encoder at 24kHz (matches pipeline)

```python
# train_speaker_encoder.py line 53, change from:
sample_rate: int = 16000,
# to:
sample_rate: int = 24000,
```

**Recommendation**: Use Option B (keep 24kHz everywhere in pipeline). Requires retraining speaker encoder.

---

### P2.2: extract_prosody hardcodes 50Hz frame rate → breaks with Codec12Hz

**Location**: `/Users/sethford/Documents/pocket-voice/train/sonata/encode_dataset.py:77`

**Issue**:

- extract_prosody() hardcodes `hop = sr // 50 = 480` (line 77) → 50 Hz frame rate
- Works fine for CodecConfig (hop_length=480, 50Hz) ✓
- But if someone uses Codec12HzConfig (hop_length=1920, 12.5Hz):
  - extract_prosody still forces 50Hz frame rate
  - Returned prosody has wrong frame count
  - Prosody frames won't align with codec semantic_tokens

**Evidence**:

```python
# encode_dataset.py line 77
def extract_prosody(audio: torch.Tensor, sr: int, n_frames: int) -> torch.Tensor:
    hop = sr // 50  # Hardcoded 50 Hz frame rate
    # ...
    for i in range(n_frames):
        # extracts prosody at 50Hz, but n_frames may be from 12.5Hz codec
```

**Impact**: Future-proof issue. Won't break current (50Hz codec) training, but will silently fail when Codec12Hz training uses encoded data with prosody.

**Fix**:
Update extract_prosody to accept frame rate parameter:

```python
# encode_dataset.py, change signature from:
def extract_prosody(audio: torch.Tensor, sr: int, n_frames: int) -> torch.Tensor:
# to:
def extract_prosody(audio: torch.Tensor, sr: int, n_frames: int, hop_length: int = 480) -> torch.Tensor:
    hop = hop_length  # Use passed value instead of sr // 50

# encode_dataset.py line ~95, compute speaking_rate with hop
```

Then update data_pipeline.py call:

```python
# data_pipeline.py line 245, change from:
prosody = extract_prosody(audio, cfg.sample_rate, n_frames)
# to:
prosody = extract_prosody(audio, cfg.sample_rate, n_frames, cfg.hop_length)
```

---

## P3 — Low Priority (Design Debt, Polish)

### P3.1: No stride product validation in Codec12HzConfig

**Location**: `/Users/sethford/Documents/pocket-voice/train/sonata/config.py:463-465`

**Issue**:

- Codec12HzConfig encoder_strides = [4, 8, 5, 4, 3], product = 1920 ✓
- Codec12HzConfig decoder_strides = [3, 4, 5, 8, 4], product = 1920 ✓
- But no runtime assertion that product matches hop_length
- If someone edits strides without updating product, codec silently produces wrong output
- Example: change last stride from 3→2, product becomes 1280 ≠ 1920, silent corruption

**Evidence**:

```python
# config.py line 463-465
encoder_strides: List[int] = field(default_factory=lambda: [4, 8, 5, 4, 3])
# Decoder strides: mirror of encoder, product = 1920
decoder_strides: List[int] = field(default_factory=lambda: [3, 4, 5, 8, 4])
# No assertion that product == hop_length
```

**Impact**: Design debt. Low risk if no one edits strides, but dangerous if they do.

**Fix**:
Add property and **post_init** validation:

```python
@property
def encoder_stride_product(self) -> int:
    result = 1
    for s in self.encoder_strides:
        result *= s
    return result

@property
def decoder_stride_product(self) -> int:
    result = 1
    for s in self.decoder_strides:
        result *= s
    return result

def __post_init__(self):
    # Only if using dataclass; otherwise add to __init__
    if self.encoder_stride_product != self.hop_length:
        raise ValueError(
            f"encoder_strides product {self.encoder_stride_product} != "
            f"hop_length {self.hop_length}"
        )
    if self.decoder_stride_product != self.hop_length:
        raise ValueError(
            f"decoder_strides product {self.decoder_stride_product} != "
            f"hop_length {self.hop_length}"
        )
```

---

### P3.2: Inconsistent data key naming (acoustic_latents vs acoustic_latent)

**Location**: `/Users/sethford/Documents/pocket-voice/train/sonata/data_pipeline.py:242,252`

**Issue**:
Variable name: `acoustic_latent` (singular)
Dictionary key: `"acoustic_latents"` (plural)
Inconsistent naming, potential source of bugs if anyone loads the data manually

**Evidence**:

```python
# data_pipeline.py line 242
semantic_tokens, acoustic_latent, _ = model.encode(audio_device)

# data_pipeline.py line 252
shard_buf.append({
    "text": entry["text"],
    ...
    "acoustic_latents": acoustic_latent[0].cpu(),  # ← plural key, singular var
    ...
})
```

**Impact**: Low. Code works, but confusing for documentation and manual data loading.

**Fix**:
Rename key to singular for consistency:

```python
# data_pipeline.py line 252, change from:
"acoustic_latents": acoustic_latent[0].cpu(),
# to:
"acoustic_latent": acoustic_latent[0].cpu(),
```

Or rename everywhere to plural for semantic clarity (since it represents multiple frames).

---

## Testing Checklist Before Merge

- [ ] **P0.1 Fix**: FlowV3LargeConfig speaker_dim → 256
  - Test: `python train_flow_v3.py --model-size large --n-speakers 100`

- [ ] **P0.2 Fix**: ReferenceEncoder output_dim → 80
  - Test: `ref_enc = ReferenceEncoder(); flow_cfg = FlowV3Config(use_ref_audio=True); wrapper = VoicePromptWrapper(flow, ref_enc)` (should not crash)

- [ ] **P1.1 Fix**: Codec12HzConfig decision (80 or 160 mel bins)
  - Test: Codec12Hz training pipeline (once complete)

- [ ] **P2.1 Fix**: Speaker encoder sample rate alignment
  - Test: Verify speaker encoder + voice_prompt use same sample rate

- [ ] **P2.2 Fix**: extract_prosody hop_length parameter
  - Test: encode_dataset.py with both CodecConfig (480) and Codec12HzConfig (1920)

- [ ] **P3.1 Fix**: Add stride product validation
  - Test: Codec12HzConfig() initialization

- [ ] **P3.2 Fix**: Rename acoustic_latents key
  - Test: Data loading script reads renamed key correctly

---

## E2E Regression Tests (After All Fixes)

1. **Codec training (50Hz)**:

   ```bash
   python train_codec.py --manifest data/manifest.jsonl --steps 1000 --device mps
   ```

2. **Flow v3 with speaker conditioning**:

   ```bash
   python train_flow_v3.py --manifest data/manifest.jsonl --n-speakers 50 --steps 500 --device mps
   ```

3. **Flow v3 large model**:

   ```bash
   python train_flow_v3.py --manifest data/manifest.jsonl --model-size large --steps 500 --device mps
   ```

4. **Voice prompting (zero-shot cloning)**:

   ```python
   from voice_prompt import ReferenceEncoder, VoicePromptWrapper
   ref_enc = ReferenceEncoder()  # should not crash (P0.2 fix)
   wrapper = VoicePromptWrapper(flow_model, ref_enc)  # should work
   ```

5. **End-to-end encode→train→generate→vocoder**:
   - Encode: `python data_pipeline.py --codec-ckpt ... --manifest ... --output ...`
   - Train: Flow training using encoded data
   - Generate & vocoder (once vocoder training complete)

---

## Files Modified Summary

| Priority | File                                        | Line(s)           | Change                                               |
| -------- | ------------------------------------------- | ----------------- | ---------------------------------------------------- |
| P0.1     | config.py                                   | 420               | speaker_dim: 384 → 256                               |
| P0.2     | voice_prompt.py                             | 42                | output_dim: 512 → 80                                 |
| P1.1     | config.py                                   | 436               | n_mels: 160 → 80 (or keep 160 + update Flow/Vocoder) |
| P2.1     | voice_prompt.py or train_speaker_encoder.py | 53 or 131         | Align sample rates (16kHz or 24kHz)                  |
| P2.2     | encode_dataset.py, data_pipeline.py         | 77, 245           | Add hop_length parameter to extract_prosody          |
| P3.1     | config.py                                   | Add **post_init** | Validate stride products                             |
| P3.2     | data_pipeline.py                            | 252               | Rename "acoustic_latents" → "acoustic_latent"        |
