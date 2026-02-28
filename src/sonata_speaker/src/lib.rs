// sonata_speaker — Native ECAPA-TDNN speaker encoder on Apple Silicon Metal GPU.
//
// Architecture: ECAPA-TDNN (~6M params)
//   - 1D convolutions with SE-Res2Net blocks
//   - Attentive statistics pooling
//   - Input: 80-bin log-mel spectrogram
//   - Output: 256-dim L2-normalized d-vector
//
// C FFI:
//   speaker_encoder_native_create(weights, config) -> *encoder
//   speaker_encoder_native_encode(encoder, mel, n_frames, n_mels, out) -> dim
//   speaker_encoder_native_encode_audio(encoder, pcm, n_samples, sr, out) -> dim
//   speaker_encoder_native_embedding_dim(encoder) -> dim
//   speaker_encoder_native_destroy(encoder)

use candle_core::{DType, Device, Result, Tensor, D};
use candle_nn::{linear, Conv1d, Conv1dConfig, Linear, Module, VarBuilder};
use serde::Deserialize;
use std::ffi::{c_char, c_float, c_int, c_void, CStr};
use std::path::Path;

fn panic_message(e: Box<dyn std::any::Any + Send>) -> String {
    if let Some(s) = e.downcast_ref::<&str>() {
        s.to_string()
    } else if let Some(s) = e.downcast_ref::<String>() {
        s.clone()
    } else {
        "unknown panic".to_string()
    }
}

// ─── Config ──────────────────────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
struct SpeakerEncoderConfig {
    #[serde(default = "default_n_mels")]
    n_mels: usize,
    #[serde(default = "default_channels")]
    channels: Vec<usize>,         // [1024, 1024, 1024, 1024, 1536]
    #[serde(default = "default_kernel_sizes")]
    kernel_sizes: Vec<usize>,     // [5, 3, 3, 3, 1]
    #[serde(default = "default_dilations")]
    dilations: Vec<usize>,        // [1, 2, 3, 4, 1]
    #[serde(default = "default_embedding_dim")]
    embedding_dim: usize,         // 256
    #[serde(default = "default_res2net_scale")]
    res2net_scale: usize,         // 8
    #[serde(default = "default_se_channels")]
    se_channels: usize,           // 128
    #[serde(default = "default_attention_channels")]
    attention_channels: usize,    // 128
    #[serde(default = "default_sample_rate")]
    sample_rate: usize,
}

fn default_n_mels() -> usize { 80 }
fn default_channels() -> Vec<usize> { vec![1024, 1024, 1024, 1024, 1536] }
fn default_kernel_sizes() -> Vec<usize> { vec![5, 3, 3, 3, 1] }
fn default_dilations() -> Vec<usize> { vec![1, 2, 3, 4, 1] }
fn default_embedding_dim() -> usize { 256 }
fn default_res2net_scale() -> usize { 8 }
fn default_se_channels() -> usize { 128 }
fn default_attention_channels() -> usize { 128 }
fn default_sample_rate() -> usize { 16000 }

// ─── 1D Batch Norm ──────────────────────────────────────────────────────────

struct BatchNorm1d {
    weight: Tensor,
    bias: Tensor,
    running_mean: Tensor,
    running_var: Tensor,
    eps: f64,
}

impl BatchNorm1d {
    fn load(channels: usize, vb: VarBuilder) -> Result<Self> {
        let weight = vb.get(channels, "weight")?;
        let bias = vb.get(channels, "bias")?;
        let running_mean = vb.get(channels, "running_mean")?;
        let running_var = vb.get(channels, "running_var")?;
        Ok(Self { weight, bias, running_mean, running_var, eps: 1e-5 })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // x: [B, C, T] — inference mode (use running stats)
        let mean = self.running_mean.reshape((1, (), 1))?;
        let var = self.running_var.reshape((1, (), 1))?;
        let w = self.weight.reshape((1, (), 1))?;
        let b = self.bias.reshape((1, (), 1))?;
        let normed = x.broadcast_sub(&mean)?
            .broadcast_div(&(var + self.eps)?.sqrt()?)?;
        normed.broadcast_mul(&w)?.broadcast_add(&b)
    }
}

// ─── SE (Squeeze-and-Excitation) Block ──────────────────────────────────────

struct SEBlock {
    conv1: Conv1d,
    conv2: Conv1d,
}

impl SEBlock {
    fn load(channels: usize, se_channels: usize, vb: VarBuilder) -> Result<Self> {
        let cfg = Conv1dConfig { padding: 0, stride: 1, dilation: 1, groups: 1 };
        let conv1 = candle_nn::conv1d(channels, se_channels, 1, cfg, vb.pp("conv1"))?;
        let conv2 = candle_nn::conv1d(se_channels, channels, 1, cfg, vb.pp("conv2"))?;
        Ok(Self { conv1, conv2 })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // x: [B, C, T]
        let s = x.mean(D::Minus1)?.unsqueeze(D::Minus1)?; // [B, C, 1]
        let s = self.conv1.forward(&s)?.relu()?;
        let s = self.conv2.forward(&s)?;
        let s = candle_nn::ops::sigmoid(&s)?;
        x.broadcast_mul(&s)
    }
}

// ─── Res2Net Block ──────────────────────────────────────────────────────────

struct Res2NetBlock {
    convs: Vec<Conv1d>,
    bns: Vec<BatchNorm1d>,
    scale: usize,
    width: usize,
}

impl Res2NetBlock {
    fn load(channels: usize, kernel_size: usize, dilation: usize,
            scale: usize, vb: VarBuilder) -> Result<Self> {
        let width = channels / scale;
        let mut convs = Vec::new();
        let mut bns = Vec::new();
        let padding = (kernel_size - 1) * dilation / 2;
        let cfg = Conv1dConfig { padding, stride: 1, dilation, groups: 1 };
        // scale-2 convolutions (first chunk is identity, last chunk is identity)
        for i in 0..(scale - 1) {
            let c = candle_nn::conv1d(width, width, kernel_size, cfg, vb.pp(format!("convs.{}", i)))?;
            let b = BatchNorm1d::load(width, vb.pp(format!("bns.{}", i)))?;
            convs.push(c);
            bns.push(b);
        }
        Ok(Self { convs, bns, scale, width })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // x: [B, C, T] → split into scale chunks along C
        let (_b, _c, _t) = x.dims3()?;
        let chunks: Vec<Tensor> = (0..self.scale)
            .map(|i| {
                let start = i * self.width;
                x.narrow(1, start, self.width)
            })
            .collect::<Result<Vec<_>>>()?;

        let mut outputs = Vec::with_capacity(self.scale);
        outputs.push(chunks[0].clone());

        for i in 1..self.scale {
            let input = if i == 1 {
                chunks[i].clone()
            } else {
                (&chunks[i] + outputs.last().unwrap())?
            };
            let y = self.convs[i - 1].forward(&input)?;
            let y = self.bns[i - 1].forward(&y)?;
            let y = y.relu()?;
            outputs.push(y);
        }

        Tensor::cat(&outputs, 1)
    }
}

// ─── SE-Res2Net Block (one ECAPA-TDNN block) ───────────────────────────────

struct SERes2NetBlock {
    conv1: Conv1d,
    bn1: BatchNorm1d,
    res2net: Res2NetBlock,
    conv2: Conv1d,
    bn2: BatchNorm1d,
    se: SEBlock,
    shortcut: Option<(Conv1d, BatchNorm1d)>,
}

impl SERes2NetBlock {
    fn load(in_channels: usize, out_channels: usize, kernel_size: usize,
            dilation: usize, scale: usize, se_channels: usize,
            vb: VarBuilder) -> Result<Self> {
        let cfg1 = Conv1dConfig { padding: 0, stride: 1, dilation: 1, groups: 1 };
        let conv1 = candle_nn::conv1d(in_channels, out_channels, 1, cfg1, vb.pp("conv1"))?;
        let bn1 = BatchNorm1d::load(out_channels, vb.pp("bn1"))?;

        let res2net = Res2NetBlock::load(out_channels, kernel_size, dilation, scale,
                                          vb.pp("res2net"))?;

        let conv2 = candle_nn::conv1d(out_channels, out_channels, 1, cfg1, vb.pp("conv2"))?;
        let bn2 = BatchNorm1d::load(out_channels, vb.pp("bn2"))?;

        let se = SEBlock::load(out_channels, se_channels, vb.pp("se"))?;

        let shortcut = if in_channels != out_channels {
            let sc = candle_nn::conv1d(in_channels, out_channels, 1, cfg1, vb.pp("shortcut.0"))?;
            let sbn = BatchNorm1d::load(out_channels, vb.pp("shortcut.1"))?;
            Some((sc, sbn))
        } else {
            None
        };

        Ok(Self { conv1, bn1, res2net, conv2, bn2, se, shortcut })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let residual = if let Some((ref conv, ref bn)) = self.shortcut {
            bn.forward(&conv.forward(x)?)?
        } else {
            x.clone()
        };

        let h = self.conv1.forward(x)?;
        let h = self.bn1.forward(&h)?.relu()?;
        let h = self.res2net.forward(&h)?;
        let h = self.conv2.forward(&h)?;
        let h = self.bn2.forward(&h)?;
        let h = self.se.forward(&h)?;

        (h + residual)?.relu()
    }
}

// ─── Attentive Statistics Pooling ───────────────────────────────────────────

struct AttentiveStatisticsPooling {
    attention: Conv1d,
    bn: BatchNorm1d,
    proj: Conv1d,
}

impl AttentiveStatisticsPooling {
    fn load(channels: usize, attention_channels: usize, vb: VarBuilder) -> Result<Self> {
        let cfg = Conv1dConfig { padding: 0, stride: 1, dilation: 1, groups: 1 };
        let attention = candle_nn::conv1d(channels, attention_channels, 1, cfg,
                                           vb.pp("attention"))?;
        let bn = BatchNorm1d::load(attention_channels, vb.pp("bn"))?;
        let proj = candle_nn::conv1d(attention_channels, channels, 1, cfg,
                                      vb.pp("proj"))?;
        Ok(Self { attention, bn, proj })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // x: [B, C, T]
        let alpha = self.attention.forward(x)?;
        let alpha = self.bn.forward(&alpha)?.tanh()?;
        let alpha = self.proj.forward(&alpha)?;
        let alpha = candle_nn::ops::softmax(&alpha, D::Minus1)?; // [B, C, T]

        // Weighted mean
        let mean = x.broadcast_mul(&alpha)?.sum(D::Minus1)?; // [B, C]

        // Weighted std
        let mean_expanded = mean.unsqueeze(D::Minus1)?;
        let diff = x.broadcast_sub(&mean_expanded)?;
        let var = diff.sqr()?.broadcast_mul(&alpha)?.sum(D::Minus1)?; // [B, C]
        let std = (var + 1e-8)?.sqrt()?;

        Tensor::cat(&[&mean, &std], D::Minus1) // [B, 2*C]
    }
}

// ─── ECAPA-TDNN Model ───────────────────────────────────────────────────────

struct EcapaTdnn {
    input_conv: Conv1d,
    input_bn: BatchNorm1d,
    blocks: Vec<SERes2NetBlock>,
    mfa_conv: Conv1d,
    mfa_bn: BatchNorm1d,
    asp: AttentiveStatisticsPooling,
    final_linear: Linear,
    final_bn: BatchNorm1d,
    config: SpeakerEncoderConfig,
}

impl EcapaTdnn {
    fn load(config: SpeakerEncoderConfig, vb: VarBuilder) -> Result<Self> {
        let first_channels = config.channels[0];
        let padding = (config.kernel_sizes[0] - 1) / 2;
        let cfg = Conv1dConfig {
            padding, stride: 1,
            dilation: config.dilations[0], groups: 1,
        };
        let input_conv = candle_nn::conv1d(config.n_mels, first_channels,
                                            config.kernel_sizes[0], cfg,
                                            vb.pp("input_conv"))?;
        let input_bn = BatchNorm1d::load(first_channels, vb.pp("input_bn"))?;

        let n_blocks = config.channels.len() - 1;
        let mut blocks = Vec::with_capacity(n_blocks);
        for i in 0..n_blocks {
            let in_ch = config.channels[i];
            let out_ch = config.channels[i + 1];
            let ks = config.kernel_sizes.get(i + 1).copied().unwrap_or(3);
            let dil = config.dilations.get(i + 1).copied().unwrap_or(1);
            let block = SERes2NetBlock::load(
                in_ch, out_ch, ks, dil,
                config.res2net_scale, config.se_channels,
                vb.pp(format!("blocks.{}", i)),
            )?;
            blocks.push(block);
        }

        // Multi-layer feature aggregation: cat all block outputs → project
        let last_ch = *config.channels.last().unwrap_or(&1536);
        let total_cat = config.channels.iter().sum::<usize>(); // input + all block outputs
        let mfa_cfg = Conv1dConfig { padding: 0, stride: 1, dilation: 1, groups: 1 };
        let mfa_conv = candle_nn::conv1d(total_cat, last_ch, 1, mfa_cfg,
                                          vb.pp("mfa_conv"))?;
        let mfa_bn = BatchNorm1d::load(last_ch, vb.pp("mfa_bn"))?;

        let asp = AttentiveStatisticsPooling::load(last_ch, config.attention_channels,
                                                    vb.pp("asp"))?;

        let final_linear = linear(last_ch * 2, config.embedding_dim, vb.pp("final_linear"))?;
        let final_bn = BatchNorm1d::load(config.embedding_dim, vb.pp("final_bn"))?;

        Ok(Self {
            input_conv, input_bn, blocks, mfa_conv, mfa_bn, asp,
            final_linear, final_bn, config,
        })
    }

    fn forward(&self, mel: &Tensor) -> Result<Tensor> {
        // mel: [B, n_mels, T]
        let x = self.input_conv.forward(mel)?;
        let x = self.input_bn.forward(&x)?.relu()?;

        let mut features = vec![x.clone()];
        let mut h = x;

        for block in &self.blocks {
            h = block.forward(&h)?;
            features.push(h.clone());
        }

        // Multi-layer feature aggregation
        let cat = Tensor::cat(&features, 1)?; // [B, sum(channels), T]
        let h = self.mfa_conv.forward(&cat)?;
        let h = self.mfa_bn.forward(&h)?.relu()?;

        // Attentive statistics pooling
        let stats = self.asp.forward(&h)?; // [B, 2*C]

        // Final projection + BN + L2 normalize
        let emb = self.final_linear.forward(&stats)?;
        let emb = self.final_bn.forward(&emb.unsqueeze(D::Minus1)?)?
            .squeeze(D::Minus1)?;

        // L2 normalize
        let norm = emb.sqr()?.sum(D::Minus1)?.sqrt()?.unsqueeze(D::Minus1)?;
        let norm = norm.clamp(1e-8, f64::MAX)?;
        emb.broadcast_div(&norm)
    }
}

// ─── Mel Spectrogram Extraction ─────────────────────────────────────────────

const FBANK_N_FFT: usize = 512;
const FBANK_WIN_LEN: usize = 400; // 25ms at 16kHz
const FBANK_HOP: usize = 160;     // 10ms
const FBANK_N_MELS: usize = 80;
const FBANK_FMIN: f32 = 20.0;
const FBANK_FMAX: f32 = 7600.0;

fn hz_to_mel(hz: f32) -> f32 { 2595.0 * (1.0 + hz / 700.0).log10() }
fn mel_to_hz(mel: f32) -> f32 { 700.0 * (10.0_f32.powf(mel / 2595.0) - 1.0) }

fn build_mel_filterbank(n_fft: usize, n_mels: usize, sr: usize,
                         fmin: f32, fmax: f32) -> Vec<f32> {
    let n_bins = n_fft / 2 + 1;
    let mel_min = hz_to_mel(fmin);
    let mel_max = hz_to_mel(fmax);
    let mut mel_points = vec![0.0f32; n_mels + 2];
    for i in 0..=(n_mels + 1) {
        mel_points[i] = mel_min + (mel_max - mel_min) * i as f32 / (n_mels + 1) as f32;
    }
    let hz_points: Vec<f32> = mel_points.iter().map(|&m| mel_to_hz(m)).collect();
    let freq_step = sr as f32 / n_fft as f32;

    let mut fb = vec![0.0f32; n_mels * n_bins];
    for m in 0..n_mels {
        for k in 0..n_bins {
            let freq = k as f32 * freq_step;
            if freq >= hz_points[m] && freq <= hz_points[m + 1] {
                fb[m * n_bins + k] = (freq - hz_points[m]) / (hz_points[m + 1] - hz_points[m]);
            } else if freq > hz_points[m + 1] && freq <= hz_points[m + 2] {
                fb[m * n_bins + k] = (hz_points[m + 2] - freq) / (hz_points[m + 2] - hz_points[m + 1]);
            }
        }
    }
    fb
}

fn compute_log_mel(audio: &[f32], sr: usize, device: &Device) -> Result<Tensor> {
    let n_bins = FBANK_N_FFT / 2 + 1;
    let mel_fb = build_mel_filterbank(FBANK_N_FFT, FBANK_N_MELS, sr, FBANK_FMIN, FBANK_FMAX);

    // Hann window
    let hann: Vec<f32> = (0..FBANK_WIN_LEN)
        .map(|i| 0.5 * (1.0 - (2.0 * std::f32::consts::PI * i as f32 / (FBANK_WIN_LEN - 1) as f32).cos()))
        .collect();

    let n_frames = if audio.len() >= FBANK_WIN_LEN {
        1 + (audio.len() - FBANK_WIN_LEN) / FBANK_HOP
    } else {
        return Err(candle_core::Error::Msg("Audio too short for fbank".to_string()));
    };

    let mut fbank = vec![0.0f32; n_frames * FBANK_N_MELS];
    let mut windowed = vec![0.0f32; FBANK_N_FFT];
    let mut power_spec = vec![0.0f32; n_bins];

    for f in 0..n_frames {
        let start = f * FBANK_HOP;

        // Apply window
        windowed.iter_mut().for_each(|v| *v = 0.0);
        for i in 0..FBANK_WIN_LEN {
            windowed[i] = audio[start + i] * hann[i];
        }

        // Simple DFT for power spectrum (real-valued input)
        for k in 0..n_bins {
            let mut re = 0.0f32;
            let mut im = 0.0f32;
            let freq = -2.0 * std::f32::consts::PI * k as f32 / FBANK_N_FFT as f32;
            for n in 0..FBANK_N_FFT {
                let angle = freq * n as f32;
                re += windowed[n] * angle.cos();
                im += windowed[n] * angle.sin();
            }
            power_spec[k] = re * re + im * im;
        }

        // Mel filterbank multiply + log
        for m in 0..FBANK_N_MELS {
            let mut val = 0.0f32;
            for k in 0..n_bins {
                val += mel_fb[m * n_bins + k] * power_spec[k];
            }
            fbank[f * FBANK_N_MELS + m] = (val + 1e-10).ln();
        }
    }

    // [1, n_mels, T] — model expects channels-first
    let t = Tensor::from_vec(fbank, (n_frames, FBANK_N_MELS), device)?;
    let t = t.t()?.unsqueeze(0)?; // [1, n_mels, T]
    Ok(t)
}

// ─── Engine ─────────────────────────────────────────────────────────────────

struct SpeakerEncoderEngine {
    model: EcapaTdnn,
    device: Device,
    dtype: DType,
}

// ─── C FFI ──────────────────────────────────────────────────────────────────

#[no_mangle]
pub extern "C" fn speaker_encoder_native_create(
    weights_path: *const c_char,
    config_path: *const c_char,
) -> *mut c_void {
    if weights_path.is_null() || config_path.is_null() {
        eprintln!("[sonata_speaker] Create error: NULL paths");
        return std::ptr::null_mut();
    }
    let weights_str = unsafe { CStr::from_ptr(weights_path).to_str().unwrap_or("") };
    let config_str = unsafe { CStr::from_ptr(config_path).to_str().unwrap_or("") };

    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        (|| -> Result<SpeakerEncoderEngine> {
            #[cfg(feature = "metal")]
            let device = Device::new_metal(0)?;
            #[cfg(not(feature = "metal"))]
            let device = Device::Cpu;

            let dtype = DType::F32; // Speaker encoder needs F32 for batchnorm accuracy

            let config_content = std::fs::read_to_string(config_str)
                .map_err(|e| candle_core::Error::Msg(format!("Config read: {}", e)))?;
            let config: SpeakerEncoderConfig = serde_json::from_str(&config_content)
                .map_err(|e| candle_core::Error::Msg(format!("Config parse: {}", e)))?;

            let emb_dim = config.embedding_dim;

            let vb = unsafe {
                VarBuilder::from_mmaped_safetensors(&[weights_str], dtype, &device)?
            };
            let model = EcapaTdnn::load(config, vb)?;

            eprintln!("[sonata_speaker] Loaded on {:?} (dtype={:?}, emb_dim={})",
                      device, dtype, emb_dim);

            Ok(SpeakerEncoderEngine { model, device, dtype })
        })()
    }));

    match result {
        Ok(Ok(engine)) => Box::into_raw(Box::new(engine)) as *mut c_void,
        Ok(Err(e)) => {
            eprintln!("[sonata_speaker] Create error: {}", e);
            std::ptr::null_mut()
        }
        Err(e) => {
            eprintln!("[sonata_speaker] Panic in create: {}", panic_message(e));
            std::ptr::null_mut()
        }
    }
}

#[no_mangle]
pub extern "C" fn speaker_encoder_native_destroy(engine: *mut c_void) {
    if !engine.is_null() {
        unsafe { drop(Box::from_raw(engine as *mut SpeakerEncoderEngine)); }
        eprintln!("[sonata_speaker] Destroyed");
    }
}

#[no_mangle]
pub extern "C" fn speaker_encoder_native_embedding_dim(engine: *const c_void) -> c_int {
    if engine.is_null() { return 0; }
    let eng = unsafe { &*(engine as *const SpeakerEncoderEngine) };
    eng.model.config.embedding_dim as c_int
}

/// Encode from pre-computed mel spectrogram.
/// mel_data: [n_frames * n_mels] row-major (frame-major: mel_data[f*n_mels + m])
/// out: must have space for embedding_dim floats.
/// Returns embedding_dim on success, -1 on error.
#[no_mangle]
pub extern "C" fn speaker_encoder_native_encode(
    engine: *mut c_void,
    mel_data: *const c_float,
    n_frames: c_int,
    n_mels: c_int,
    out: *mut c_float,
) -> c_int {
    if engine.is_null() || mel_data.is_null() || out.is_null()
        || n_frames <= 0 || n_mels <= 0 {
        return -1;
    }
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let eng = unsafe { &*(engine as *const SpeakerEncoderEngine) };
        let nf = n_frames as usize;
        let nm = n_mels as usize;
        let data: Vec<f32> = (0..(nf * nm))
            .map(|i| unsafe { *mel_data.add(i) })
            .collect();

        (|| -> Result<c_int> {
            // Input: [n_frames, n_mels] → transpose to [1, n_mels, n_frames]
            let mel = Tensor::from_vec(data, (nf, nm), &eng.device)?;
            let mel = mel.t()?.unsqueeze(0)?.to_dtype(eng.dtype)?;

            let emb = eng.model.forward(&mel)?;
            let emb = emb.to_dtype(DType::F32)?.squeeze(0)?;
            let emb_data = emb.to_vec1::<f32>()?;
            let dim = emb_data.len();
            for i in 0..dim {
                unsafe { *out.add(i) = emb_data[i]; }
            }
            Ok(dim as c_int)
        })()
    }));
    match result {
        Ok(Ok(dim)) => dim,
        Ok(Err(e)) => { eprintln!("[sonata_speaker] encode error: {}", e); -1 }
        Err(e) => { eprintln!("[sonata_speaker] panic in encode: {}", panic_message(e)); -1 }
    }
}

/// Encode from raw PCM audio (mono float32).
/// Computes mel spectrogram internally, then encodes.
/// Returns embedding_dim on success, -1 on error.
#[no_mangle]
pub extern "C" fn speaker_encoder_native_encode_audio(
    engine: *mut c_void,
    pcm: *const c_float,
    n_samples: c_int,
    sample_rate: c_int,
    out: *mut c_float,
) -> c_int {
    if engine.is_null() || pcm.is_null() || out.is_null()
        || n_samples <= 0 || sample_rate <= 0 {
        return -1;
    }
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let eng = unsafe { &*(engine as *const SpeakerEncoderEngine) };
        let ns = n_samples as usize;
        let sr = sample_rate as usize;
        let audio: Vec<f32> = (0..ns).map(|i| unsafe { *pcm.add(i) }).collect();

        (|| -> Result<c_int> {
            // Resample to 16kHz if needed
            let target_sr = eng.model.config.sample_rate;
            let resampled = if sr != target_sr {
                let ratio = target_sr as f64 / sr as f64;
                let out_len = (ns as f64 * ratio + 0.5) as usize;
                let mut out = vec![0.0f32; out_len];
                for i in 0..out_len {
                    let pos = i as f64 / ratio;
                    let idx = pos as usize;
                    let frac = (pos - idx as f64) as f32;
                    if idx >= ns - 1 {
                        out[i] = audio[ns - 1];
                    } else {
                        out[i] = audio[idx] * (1.0 - frac) + audio[idx + 1] * frac;
                    }
                }
                out
            } else {
                audio
            };

            let mel = compute_log_mel(&resampled, target_sr, &eng.device)?
                .to_dtype(eng.dtype)?;
            let emb = eng.model.forward(&mel)?;
            let emb = emb.to_dtype(DType::F32)?.squeeze(0)?;
            let emb_data = emb.to_vec1::<f32>()?;
            let dim = emb_data.len();
            for i in 0..dim {
                unsafe { *out.add(i) = emb_data[i]; }
            }
            Ok(dim as c_int)
        })()
    }));
    match result {
        Ok(Ok(dim)) => dim,
        Ok(Err(e)) => { eprintln!("[sonata_speaker] encode_audio error: {}", e); -1 }
        Err(e) => { eprintln!("[sonata_speaker] panic in encode_audio: {}", panic_message(e)); -1 }
    }
}

/// Get the model's expected sample rate (typically 16000).
#[no_mangle]
pub extern "C" fn speaker_encoder_native_sample_rate(engine: *const c_void) -> c_int {
    if engine.is_null() { return 16000; }
    let eng = unsafe { &*(engine as *const SpeakerEncoderEngine) };
    eng.model.config.sample_rate as c_int
}
