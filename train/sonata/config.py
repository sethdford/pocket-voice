"""Sonata — shared configuration for all models.

Architecture:
  Text → Semantic LM → Semantic Tokens (50 Hz)
       → Flow Matching → Acoustic Latents (50 Hz)
       → iSTFT Decoder → Waveform (24 kHz)
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class CodecConfig:
    """Sonata Codec: Conformer encoder + FSQ + iSTFT decoder."""
    sample_rate: int = 24000
    n_fft: int = 1024
    hop_length: int = 480           # 24000/480 = 50 Hz frame rate
    n_mels: int = 80                # Mel spectrogram input bins
    win_length: int = 1024

    # Conformer encoder
    enc_dim: int = 256
    enc_n_layers: int = 4
    enc_n_heads: int = 4
    enc_conv_kernel: int = 31       # Conformer depthwise conv kernel
    enc_ff_mult: float = 4.0

    # Subsampling: Conv2d with stride 2 on mel frames → 2x downsample
    subsample_factor: int = 1       # Total temporal downsample (1 = no subsample beyond hop)

    # FSQ semantic quantizer (5-dim for 32768 entries — captures finer detail)
    fsq_levels: List[int] = field(default_factory=lambda: [8, 8, 8, 8, 8])
    # 8^5 = 32768 codebook entries

    # Acoustic latent (continuous, for flow matching)
    acoustic_dim: int = 256

    # Decoder (ConvTranspose + ConvNeXt backbone)
    dec_dim: int = 512
    dec_n_layers: int = 8
    dec_conv_kernel: int = 7
    dec_ff_mult: float = 4.0
    decoder_type: str = "conv"  # "conv" (Encodec-style) or "istft" (Vocos-style)

    @property
    def frame_rate(self) -> float:
        return self.sample_rate / self.hop_length

    @property
    def n_fft_bins(self) -> int:
        return self.n_fft // 2 + 1

    @property
    def fsq_codebook_size(self) -> int:
        result = 1
        for l in self.fsq_levels:
            result *= l
        return result

    @property
    def fsq_dim(self) -> int:
        return len(self.fsq_levels)


@dataclass
class SemanticLMConfig:
    """Sonata LM: Llama-style autoregressive semantic token predictor."""
    d_model: int = 1024
    n_layers: int = 16
    n_heads: int = 16
    n_kv_heads: int = 4             # GQA
    ffn_mult: float = 2.667
    max_seq_len: int = 4096
    rope_theta: float = 10000.0
    norm_eps: float = 1e-5

    text_vocab_size: int = 32000    # SentencePiece
    semantic_vocab_size: int = 32768 # From FSQ codec (8^5)
    n_special_tokens: int = 4      # PAD=0, BOS=1, EOS=2, MASK=3

    dropout: float = 0.0

    @property
    def d_ff(self) -> int:
        raw = int(self.d_model * self.ffn_mult)
        return raw - (raw % 256)

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads


@dataclass
class FlowConfig:
    """Sonata Flow: Conditional Flow Matching for acoustic latent prediction."""
    d_model: int = 512
    n_layers: int = 8
    n_heads: int = 8
    ff_mult: float = 4.0
    norm_eps: float = 1e-5

    # Input/output
    semantic_vocab_size: int = 32768
    acoustic_dim: int = 256         # Must match codec acoustic_dim
    cond_dim: int = 256             # Conditioning dimension

    # Flow matching
    sigma_min: float = 1e-4         # Minimum noise level
    n_steps_inference: int = 8      # ODE steps at inference (more = better quality)

    # Speaker conditioning (optional)
    speaker_dim: int = 256
    n_speakers: int = 0             # 0 = no speaker conditioning

    # Emotion conditioning (optional)
    n_emotions: int = 0             # 0 = no emotion conditioning
    emotion_dim: int = 64           # Emotion embedding dimension

    # Prosody conditioning (optional)
    prosody_dim: int = 3            # (log_pitch, energy, rate)

    # Positional encoding
    use_rope: bool = False          # RoPE in flow attention (better for long sequences)

    # Reference audio cross-attention for zero-shot voice cloning (optional)
    use_ref_audio: bool = False     # Cross-attention to reference audio features
    ref_audio_dim: int = 80         # Dimension of reference audio features (mel bins)

    # Energy predictor (optional) — predicts per-frame log-energy from conditioning
    use_energy_predictor: bool = False
    energy_dim: int = 128           # Energy predictor hidden dim

    @classmethod
    def _normalize_loaded_dict(cls, d: dict) -> dict:
        """Normalize dict for FlowConfig loading (backward compat for renamed fields)."""
        d = dict(d)
        if "use_duration_predictor" in d and "use_energy_predictor" not in d:
            d["use_energy_predictor"] = d.pop("use_duration_predictor")
        if "duration_dim" in d and "energy_dim" not in d:
            d["energy_dim"] = d.pop("duration_dim")
        return {k: v for k, v in d.items() if k in cls.__dataclass_fields__}


@dataclass
class FlowLargeConfig(FlowConfig):
    """Flow model scaled to ~150M params for SOTA quality."""
    d_model: int = 768
    n_layers: int = 16
    n_heads: int = 12
    ff_mult: float = 4.0
    cond_dim: int = 384
    use_rope: bool = True


@dataclass
class FlowV2Config:
    """Sonata Flow v2: Single-stage text → mel via conditional flow matching.

    F5-TTS-inspired: text characters are padded with filler tokens to mel length,
    refined by ConvNeXt, then used as conditioning for the DiT flow backbone.
    No semantic tokens, no LM, no codec in the TTS critical path.
    """
    d_model: int = 512
    n_layers: int = 12
    n_heads: int = 8
    ff_mult: float = 4.0
    norm_eps: float = 1e-5

    mel_dim: int = 80
    cond_dim: int = 512

    # Text encoder (ConvNeXt V2)
    text_encoder_layers: int = 4
    text_encoder_dim: int = 512
    text_encoder_kernel: int = 7
    char_vocab_size: int = 256
    filler_token_id: int = 0

    # Audio
    sample_rate: int = 24000
    n_fft: int = 1024
    hop_length: int = 480
    n_mels_extract: int = 80

    # Flow matching
    sigma_min: float = 1e-4
    n_steps_inference: int = 8

    # Speaker conditioning (optional)
    speaker_dim: int = 256
    n_speakers: int = 0

    # Emotion conditioning (optional)
    n_emotions: int = 0
    emotion_dim: int = 64

    # Prosody conditioning (optional)
    prosody_dim: int = 3

    # Sway sampling
    sway_coefficient: float = -1.0

    @property
    def frame_rate(self) -> float:
        return self.sample_rate / self.hop_length

    @property
    def n_fft_bins(self) -> int:
        return self.n_fft // 2 + 1


@dataclass
class STTConfig:
    """Sonata STT Pass 1: CTC head on codec conformer encoder."""
    # Text vocabulary: blank(0) + space(1) + a-z(2-27) + apostrophe(28) + <eou>(29)
    text_vocab_size: int = 30
    blank_id: int = 0
    eou_id: int = 29

    # Encoder params — "base" reuses codec (4L d=256), "large" is STT-optimized (12L d=512)
    enc_dim: int = 256
    enc_n_layers: int = 4
    enc_n_heads: int = 4
    enc_conv_kernel: int = 31
    enc_ff_mult: float = 4.0
    n_mels: int = 80
    use_rope: bool = True           # RoPE positional encoding in self-attention

    # CTC projection
    ctc_dropout: float = 0.1

    # Audio
    sample_rate: int = 24000
    n_fft: int = 1024
    hop_length: int = 480
    win_length: int = 1024

    # SpecAugment
    spec_augment: bool = True
    freq_mask_width: int = 15       # F in SpecAugment (max freq bins masked)
    time_mask_width: int = 50       # T in SpecAugment (max time frames masked)
    n_freq_masks: int = 2
    n_time_masks: int = 2

    # Speed perturbation
    speed_perturb: bool = True
    speed_factors: List[float] = field(default_factory=lambda: [0.9, 1.0, 1.1])

    # Noise/reverb augmentation (for robustness)
    noise_augment: bool = True
    noise_snr_range: tuple = (5, 25)       # dB
    reverb_rt60_range: tuple = (0.1, 0.6)  # seconds
    augment_prob: float = 0.3              # per-augmentation probability

    @property
    def frame_rate(self) -> float:
        return self.sample_rate / self.hop_length


@dataclass
class STTLargeConfig(STTConfig):
    """STT-optimized larger encoder (12L d=512, ~80M params)."""
    enc_dim: int = 512
    enc_n_layers: int = 12
    enc_n_heads: int = 8
    enc_conv_kernel: int = 31
    enc_ff_mult: float = 4.0
    text_vocab_size: int = 30
    ctc_dropout: float = 0.1
    use_rope: bool = True


@dataclass
class RefinerConfig:
    """Sonata STT Pass 2: Semantic token → text encoder-decoder transformer."""
    # Semantic input
    semantic_vocab_size: int = 32768

    # Text output (SentencePiece subword)
    text_vocab_size: int = 4096
    text_pad_id: int = 0
    text_bos_id: int = 1
    text_eos_id: int = 2

    # Encoder (processes semantic token sequence)
    enc_d_model: int = 512
    enc_n_layers: int = 4
    enc_n_heads: int = 8
    enc_ff_mult: float = 4.0

    # Decoder (generates text autoregressively)
    dec_d_model: int = 512
    dec_n_layers: int = 4
    dec_n_heads: int = 8
    dec_n_kv_heads: int = 4
    dec_ff_mult: float = 4.0

    max_text_len: int = 512
    max_audio_len: int = 2048
    norm_eps: float = 1e-5
    dropout: float = 0.1
    rope_theta: float = 10000.0

    @property
    def enc_d_ff(self) -> int:
        return int(self.enc_d_model * self.enc_ff_mult)

    @property
    def dec_d_ff(self) -> int:
        return int(self.dec_d_model * self.dec_ff_mult)

    @property
    def dec_head_dim(self) -> int:
        return self.dec_d_model // self.dec_n_heads


@dataclass
class SoundStormConfig(SemanticLMConfig):
    """SoundStorm: MaskGIT-style parallel semantic token predictor.

    Reuses SemanticLMConfig plus adds text encoder and masking params.
    """
    n_text_layers: int = 4           # Text encoder transformer layers
    mask_schedule: str = "cosine"    # "cosine" or "linear" masking schedule


@dataclass
class FlowV3Config:
    """Sonata Flow v3: interleaved streaming text → mel with causal attention.

    SpeakStream-inspired: causal sliding-window DiT with interleaved text-speech
    training for streaming TTS and implicit learned alignment.

    Supports two streaming modes:
      1. Causal streaming (sample_streaming): strict causal mask, lower quality, lower latency
      2. Dragon-FM (sample_dragon): bidirectional within chunk, AR across chunks, higher quality
    """
    d_model: int = 512
    n_layers: int = 12
    n_heads: int = 8
    ff_mult: float = 4.0
    norm_eps: float = 1e-5

    mel_dim: int = 80
    cond_dim: int = 512
    char_vocab_size: int = 256

    # Streaming
    window_size: int = 256          # Sliding window for attention
    chunk_size: int = 25            # Mel frames per streaming chunk
    overlap_frames: int = 5         # Crossfade overlap between chunks

    # Audio
    sample_rate: int = 24000
    n_fft: int = 1024
    hop_length: int = 480
    n_mels_extract: int = 80

    # Flow matching
    sigma_min: float = 1e-4
    n_steps_inference: int = 8
    sway_coefficient: float = -1.0

    # Speaker conditioning
    speaker_dim: int = 256
    n_speakers: int = 0

    # Emotion conditioning (optional, for TokenLevelEmoSteer)
    n_emotions: int = 0

    # Reference audio prompting
    max_ref_frames: int = 200

    # Data augmentation
    speed_perturb: bool = True
    spec_augment: bool = True

    # Interleaved training (SpeakStream-style)
    interleaved_training: bool = False

    @property
    def frame_rate(self) -> float:
        return self.sample_rate / self.hop_length

    @property
    def n_fft_bins(self) -> int:
        return self.n_fft // 2 + 1


@dataclass
class InterleavedTrainingConfig:
    """Configuration for SpeakStream-style interleaved text-speech training data.

    Controls how (text, audio) pairs are segmented into interleaved chunks
    for training Flow v3 with learned alignment.
    """
    chunk_words: int = 6                # Words per text chunk (4-8 recommended)
    overlap_words: int = 1              # Word overlap between adjacent chunks
    min_chunk_duration_ms: float = 200.0  # Minimum speech chunk duration
    bos_token_id: int = 1               # BOS token prepended to each speech chunk
    eos_token_id: int = 2               # EOS token appended to each speech chunk
    max_chunks_per_utterance: int = 32  # Cap to prevent memory issues
    speech_only_loss: bool = True       # Loss on speech tokens only (text = conditioning)


@dataclass
class FlowV3LargeConfig(FlowV3Config):
    """Large Sonata Flow v3 for maximum quality (~150M params).

    Scaled from base (55M) to compete with F5-TTS (335M).
    Sweet spot for Apple Silicon M-series: fits in unified memory,
    fast enough for near-realtime with Heun 4-step inference.
    """
    d_model: int = 768
    n_layers: int = 16
    n_heads: int = 12
    ff_mult: float = 4.0
    cond_dim: int = 768
    speaker_dim: int = 384  # matches trained flow_v3_final.pt teacher checkpoint
    window_size: int = 512
    chunk_size: int = 50


@dataclass
class Codec12HzConfig:
    """12.5Hz codec for 4x token reduction (Mimi/DualCodec-inspired).

    At 12.5Hz, each frame represents 80ms of audio (vs 20ms at 50Hz).
    Compensates with larger encoder, wider acoustic latent, and deeper decoder.
    Compounds with speculative decoding: 4x fewer tokens × 2.3x ReDrafter = ~9x speedup.
    """
    sample_rate: int = 24000
    n_fft: int = 4096               # Larger for better freq resolution at 80ms frames
    hop_length: int = 1920          # 24000/1920 = 12.5 Hz frame rate
    n_mels: int = 80                # Match downstream model expectations (Flow, Vocoder)
    win_length: int = 4096

    # Conformer encoder (larger for 4x temporal compression)
    enc_dim: int = 512
    enc_n_layers: int = 6
    enc_n_heads: int = 8
    enc_conv_kernel: int = 31
    enc_ff_mult: float = 4.0

    subsample_factor: int = 1

    # FSQ semantic quantizer (4096 entries, proven stable)
    fsq_levels: List[int] = field(default_factory=lambda: [8, 8, 8, 8])
    # 8^4 = 4096 codebook entries

    # Acoustic latent (larger to compensate for temporal compression)
    acoustic_dim: int = 512

    # Decoder (deeper for harder 1920x upsample task)
    dec_dim: int = 768
    dec_n_layers: int = 10
    dec_conv_kernel: int = 7
    dec_ff_mult: float = 4.0
    decoder_type: str = "conv"      # "conv" (Encodec-style) or "istft" (Vocos-style)

    # Waveform encoder strides: product = 1920 for 12.5Hz
    encoder_strides: List[int] = field(default_factory=lambda: [4, 8, 5, 4, 3])
    # Decoder strides: mirror of encoder, product = 1920
    decoder_strides: List[int] = field(default_factory=lambda: [3, 4, 5, 8, 4])

    def __post_init__(self):
        """Validate encoder and decoder stride products match hop_length."""
        enc_product = 1
        for s in self.encoder_strides:
            enc_product *= s
        dec_product = 1
        for s in self.decoder_strides:
            dec_product *= s
        if enc_product != self.hop_length:
            raise ValueError(f"encoder_strides product {enc_product} != hop_length {self.hop_length}")
        if dec_product != self.hop_length:
            raise ValueError(f"decoder_strides product {dec_product} != hop_length {self.hop_length}")

    @property
    def frame_rate(self) -> float:
        return self.sample_rate / self.hop_length

    @property
    def n_fft_bins(self) -> int:
        return self.n_fft // 2 + 1

    @property
    def fsq_codebook_size(self) -> int:
        result = 1
        for l in self.fsq_levels:
            result *= l
        return result

    @property
    def fsq_dim(self) -> int:
        return len(self.fsq_levels)


@dataclass
class CodecV2Config:
    """WavTokenizer-inspired: fewer tokens, richer codebook, attention decoder."""
    sample_rate: int = 24000
    n_fft: int = 1024
    hop_length: int = 960              # 24000/960 = 25 Hz
    n_mels: int = 80
    win_length: int = 1024

    enc_dim: int = 512
    enc_n_layers: int = 6
    enc_n_heads: int = 8
    enc_conv_kernel: int = 31
    enc_ff_mult: float = 4.0

    fsq_levels: List[int] = field(default_factory=lambda: [16, 16, 16, 16])
    acoustic_dim: int = 512

    dec_dim: int = 512
    dec_n_layers: int = 8
    dec_n_heads: int = 8
    dec_ff_mult: float = 4.0
    decoder_type: str = "istft"

    @property
    def frame_rate(self) -> float:
        return self.sample_rate / self.hop_length

    @property
    def n_fft_bins(self) -> int:
        return self.n_fft // 2 + 1

    @property
    def fsq_codebook_size(self) -> int:
        result = 1
        for l in self.fsq_levels:
            result *= l
        return result

    @property
    def fsq_dim(self) -> int:
        return len(self.fsq_levels)


@dataclass
class MedusaConfig:
    """Medusa speculative decoding heads for Sonata LM."""
    n_medusa_heads: int = 3          # Number of prediction heads (each +1 token lookahead)
    n_residual_layers: int = 1       # Residual blocks per head
    head_decay: float = 0.8          # Loss weight decay per head (0.8^k)


@dataclass
class VocoderConfig:
    """BigVGAN-lite vocoder: mel → waveform for Flow v3 output."""
    sample_rate: int = 24000
    n_fft: int = 1024
    hop_length: int = 480            # Must match Flow v3 (24000/480 = 50 Hz)
    n_mels: int = 80
    win_length: int = 1024

    upsample_initial_channel: int = 512
    upsample_rates: List[int] = field(default_factory=lambda: [10, 6, 2, 2, 2])
    upsample_kernel_sizes: List[int] = field(default_factory=lambda: [20, 12, 4, 4, 4])
    resblock_kernel_sizes: List[int] = field(default_factory=lambda: [3, 7, 11])
    resblock_dilation_sizes: List[List[int]] = field(
        default_factory=lambda: [[1, 3, 5], [1, 3, 5], [1, 3, 5]]
    )
    mpd_periods: List[int] = field(default_factory=lambda: [2, 3, 5, 7, 11])
    mel_loss_weight: float = 45.0
    feature_loss_weight: float = 2.0


@dataclass
class VocoderLargeConfig(VocoderConfig):
    """Large vocoder for high-fidelity synthesis.

    Increased capacity for richer temporal patterns and finer detail.
    """
    upsample_initial_channel: int = 512
    resblock_kernel_sizes: List[int] = field(default_factory=lambda: [3, 7, 11, 15])
    resblock_dilation_sizes: List[List[int]] = field(
        default_factory=lambda: [[1, 3, 5], [1, 3, 5], [1, 3, 5], [1, 3, 5]]
    )


