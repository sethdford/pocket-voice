# Wave 3 Implementation Plan — Post-Training SOTA Speedups

**Status**: Planning (blocked on Flow v3 training completion)
**Target**: Sub-200ms E2E TTS latency, 9.2x compound speedup
**Timeline**: ~4 calendar days after Flow v3 finishes (~Mar 1-6)

## Dependency Graph

```
Flow v3 Training (85K/200K, ~25hrs remaining)
    │
    ▼
[W3P1] Flow Distillation (5 hrs)          ←── CRITICAL PATH
    │   VCT (10K steps, 3h) + APT (5K steps, 2h)
    │   Result: 8-step → 1-2 step (8-16x speedup)
    │
    ├──────────────────────────────────────┐
    ▼                                      ▼
[W3P2] 12.5Hz Codec Training    [W3P3] ReDrafter Draft Model
    Stage 1: Recon (6-8h)           KD from frozen LM (4-6h)
    Stage 2: Perceptual (8-10h)     Result: 2.3x LM speedup
    Stage 3: Fine-tune (4-6h)
    │
    ├─→ Retrain Semantic LM on 12.5Hz tokens (~20h)
    └─→ Retrain Flow v3 on 12.5Hz tokens (~20h)
    │
    ├──────────────────────────────────────┐
    ▼                                      ▼
[W3P4A] Streaming Codec          [W3P4B] Denoiser Training
    Causal Conformer (2-3h)          DEMAND+MUSAN (2-3h)
    80ms frame latency budget        54K params, export .dnf
    │                                      │
    └──────────────┬───────────────────────┘
                   ▼
         E2E Integration + 8-Agent Compound Audit
         P0 fixes → Re-audit → Ship
```

## Package 1: Flow Distillation (HIGHEST PRIORITY)

**Goal**: 8-step Flow v3 → 1-2 step student
**Impact**: 8-16x TTS decode speedup

Two-phase training:

1. **Velocity Consistency Training (VCT)**: 10K steps, ~3h
   - Teacher: frozen Flow v3 (8-step ODE)
   - Student: same architecture, learns direct 1-step prediction
   - Loss: L2 between teacher ODE output and student direct output
   - Sway sampling for uniform timestep coverage
   - Expected: ~85% of 8-step quality

2. **Adversarial Post-Training (APT)**: 5K steps, ~2h
   - Add multi-scale discriminator (HiFi-GAN MPD/MSD)
   - Fixes "blurry 1-step" problem
   - Loss: adversarial + feature matching + mel reconstruction

**Inference changes** (sonata_flow/src/lib.rs):

- Fast path: n_steps=1, Euler only
- Quality path: n_steps=2, Heun half-steps
- Runtime switch based on latency budget

**Infrastructure**: train_distill_v3.py already has VCT + sway sampling. Needs APT phase added.

## Package 2: 12.5Hz Codec Training

**Goal**: Train codec_12hz.py architecture, then retrain LM + Flow on compressed tokens
**Impact**: 4x token reduction (compounds to 9.2x with spec decode)

Three-stage training:

1. **Reconstruction** (6-8h): Multi-scale STFT + mel loss, no GAN
2. **Perceptual + Adversarial** (8-10h): Add discriminator + WavLM loss
3. **Fine-tuning** (4-6h): Diverse data (LibriTTS-R full), lower LR

Then cascade retrain:

- Encode all LibriTTS-R with trained codec → 12.5Hz token sequences
- Retrain Semantic LM on 12.5Hz tokens (~20h, 100K steps)
- Retrain Flow v3 on 12.5Hz acoustic latents (~20h)

**Hardware**: ~4GB VRAM for 135M params, feasible on MPS

## Package 3: ReDrafter Draft Model Training

**Goal**: Train 3.5M GRU draft model via knowledge distillation
**Impact**: 2.3x LM inference speedup

- Teacher: frozen Sonata LM (241M params)
- Student: GRU drafter (3.5M params)
- Loss: 0.7×KL + 0.3×CE, cosine LR decay
- Training: ~4-6 hours, batch 64
- Export to safetensors for Rust candle inference

## Package 4A: Streaming Codec

**Goal**: Causal masking for real-time 12.5Hz encode/decode

- Add causal attention mask to Conformer encoder
- KV cache for sliding-window context (512 frames = 40sec)
- Fine-tune from trained codec (5K steps, ~2h)
- Per-frame budget: <10ms (well under 80ms frame)

## Package 4B: Denoiser Training

**Goal**: Train 54K-param ERB denoiser, validate deep_filter.c

- Data: LibriSpeech clean + DEMAND/MUSAN noise
- Training: 10K steps, ~2-3 hours
- Export to .dnf binary for C inference
- Validates Wave 1 deep_filter.c end-to-end

## Latency Targets

| Stage               | Current    | After Wave 3      |
| ------------------- | ---------- | ----------------- |
| STT                 | 200ms      | 200ms (unchanged) |
| LM (with ReDrafter) | 80ms       | ~35ms             |
| Flow (distilled)    | 350ms      | 50-100ms          |
| Vocoder             | 30ms       | 30ms              |
| **Total**           | **~660ms** | **~315-365ms**    |

With 12.5Hz codec (fewer tokens through LM+Flow):
| **Total (12.5Hz)** | — | **<200ms** |

## Agent Team Sizing

| Package              | Build Agents | Audit Agents           |
| -------------------- | ------------ | ---------------------- |
| Flow distill         | 1            | 2 (correctness + perf) |
| Codec training       | 1            | 2 (e2e + gap)          |
| ReDrafter            | 1            | 1 (perf)               |
| Streaming + Denoiser | 1            | 1 (assumption)         |
| Synthesis            | —            | 1                      |
| **Total**            | **4 build**  | **7 audit**            |

## Success Criteria

- Flow 1-step MCD ≤ 0.5dB vs 8-step teacher
- Codec PESQ ≥ 3.8
- ReDrafter acceptance rate ≥ 90%
- E2E compound speedup ≥ 1.7x (measured, not theoretical)
- Zero P0 findings after compound audit
