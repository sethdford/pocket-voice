//! Mel spectrogram computation for audio feature extraction.
//!
//! Computes log-mel spectrograms from raw audio using STFT with a Hann window,
//! mel filterbank projection, and log compression. Used as the audio frontend
//! for encoder models.

use candle_core::{DType, Device, Result, Tensor};

use crate::{FFT_SIZE, MEL_BINS, SAMPLE_RATE};

// Mel-specific constants (distinct from codec HOP_LENGTH)
const WINDOW_SIZE: usize = 600; // 25ms at 24kHz
const HOP_SIZE: usize = 240; // 10ms at 24kHz
const FMIN: f32 = 0.0;
const FMAX: f32 = 12000.0;
const LOG_FLOOR: f32 = 1e-10;

/// Log-mel spectrogram extractor.
///
/// Converts raw audio waveforms to log-mel spectrograms via STFT, mel filterbank
/// projection, and log compression.
pub struct MelSpectrogram {
    /// Hop size in samples between STFT frames.
    hop_size: usize,
    /// Precomputed Hann window [window_size].
    window: Tensor,
    /// DFT cosine basis [fft_size/2+1, window_size].
    dft_cos: Tensor,
    /// DFT sine basis [fft_size/2+1, window_size].
    dft_sin: Tensor,
    /// Mel filterbank [mel_bins, fft_size/2+1].
    mel_basis: Tensor,
}

impl MelSpectrogram {
    /// Create a new mel spectrogram extractor on the given device.
    pub fn new(dev: &Device) -> Result<Self> {
        let window = Self::hann_window(WINDOW_SIZE, dev)?;
        let (dft_cos, dft_sin) = Self::dft_matrices(FFT_SIZE, WINDOW_SIZE, dev)?;
        let mel_basis = Self::mel_filterbank(MEL_BINS, FFT_SIZE, SAMPLE_RATE, FMIN, FMAX, dev)?;

        Ok(Self {
            hop_size: HOP_SIZE,
            window,
            dft_cos,
            dft_sin,
            mel_basis,
        })
    }

    /// Compute log-mel spectrogram from raw audio.
    ///
    /// # Arguments
    /// * `audio` - Raw audio tensor of shape `[batch, samples]`
    ///
    /// # Returns
    /// Log-mel spectrogram of shape `[batch, mel_bins, num_frames]`
    pub fn forward(&self, audio: &Tensor) -> Result<Tensor> {
        let dims = audio.dims();
        let (batch, samples) = match dims.len() {
            2 => (dims[0], dims[1]),
            3 => {
                // [batch, 1, samples] -> squeeze channel dim
                let audio = audio.squeeze(1)?;
                return self.forward(&audio);
            }
            _ => {
                return Err(candle_core::Error::Msg(format!(
                    "Expected 2D or 3D audio tensor, got {}D",
                    dims.len()
                )));
            }
        };

        let n_frames = Self::num_frames(samples, self.hop_size);
        if n_frames == 0 {
            return Tensor::zeros(&[batch, MEL_BINS, 0], DType::F32, audio.device());
        }

        let n_freqs = FFT_SIZE / 2 + 1; // 513
        let mut batch_mels = Vec::with_capacity(batch);

        for b in 0..batch {
            let waveform = audio.narrow(0, b, 1)?.squeeze(0)?; // [samples]

            // Compute magnitude spectrogram for all frames
            let mut frame_magnitudes = Vec::with_capacity(n_frames);
            for f in 0..n_frames {
                let start = f * self.hop_size;
                let end = (start + WINDOW_SIZE).min(samples);
                let len = end - start;

                // Extract frame and zero-pad if needed
                let frame = if len < WINDOW_SIZE {
                    let partial = waveform.narrow(0, start, len)?;
                    let padding =
                        Tensor::zeros(&[WINDOW_SIZE - len], DType::F32, audio.device())?;
                    Tensor::cat(&[&partial, &padding], 0)?
                } else {
                    waveform.narrow(0, start, WINDOW_SIZE)?
                };

                // Apply window
                let windowed = (&frame * &self.window)?;

                // DFT via matrix multiply: real = dft_cos @ windowed, imag = dft_sin @ windowed
                let windowed_col = windowed.unsqueeze(1)?; // [window_size, 1]
                let real = self.dft_cos.matmul(&windowed_col)?.squeeze(1)?; // [n_freqs]
                let imag = self.dft_sin.matmul(&windowed_col)?.squeeze(1)?; // [n_freqs]

                // Magnitude: sqrt(real^2 + imag^2)
                let mag = ((&real * &real)? + (&imag * &imag)?)?.sqrt()?; // [n_freqs]
                frame_magnitudes.push(mag);
            }

            // Stack frames: [n_frames, n_freqs]
            let refs: Vec<&Tensor> = frame_magnitudes.iter().collect();
            let spec = Tensor::stack(&refs, 0)?; // [n_frames, n_freqs]

            // Apply mel filterbank: [n_frames, n_freqs] @ [n_freqs, mel_bins] -> [n_frames, mel_bins]
            let mel_basis_t = self.mel_basis.t()?; // [n_freqs, mel_bins]
            debug_assert_eq!(spec.dim(1)?, n_freqs);
            debug_assert_eq!(mel_basis_t.dim(0)?, n_freqs);
            let mel = spec.matmul(&mel_basis_t)?; // [n_frames, mel_bins]

            // Log compression: log(max(mel, 1e-10))
            let floor = Tensor::new(&[LOG_FLOOR], audio.device())?
                .broadcast_as(mel.shape())?;
            let mel_clamped = mel.maximum(&floor)?;
            let log_mel = mel_clamped.log()?; // [n_frames, mel_bins]

            // Transpose to [mel_bins, n_frames] and add batch dim
            let log_mel_t = log_mel.t()?.unsqueeze(0)?; // [1, mel_bins, n_frames]
            batch_mels.push(log_mel_t);
        }

        // Concatenate batch: [batch, mel_bins, n_frames]
        let refs: Vec<&Tensor> = batch_mels.iter().collect();
        Tensor::cat(&refs, 0)
    }

    /// Compute the number of STFT frames for a given sample count.
    pub fn num_frames(num_samples: usize, hop_size: usize) -> usize {
        if num_samples < WINDOW_SIZE {
            if num_samples > 0 { 1 } else { 0 }
        } else {
            1 + (num_samples - WINDOW_SIZE) / hop_size
        }
    }

    // --- Private helpers ---

    /// Hann window: w[n] = 0.5 * (1 - cos(2*pi*n / (N-1)))
    fn hann_window(size: usize, dev: &Device) -> Result<Tensor> {
        let mut win = Vec::with_capacity(size);
        for n in 0..size {
            let val = 0.5 * (1.0 - (2.0 * std::f32::consts::PI * n as f32 / (size - 1) as f32).cos());
            win.push(val);
        }
        Tensor::new(win.as_slice(), dev)
    }

    /// Precompute DFT cosine and sine basis matrices.
    ///
    /// Returns (cos_basis, sin_basis) each of shape [fft_size/2+1, window_size].
    fn dft_matrices(fft_size: usize, window_size: usize, dev: &Device) -> Result<(Tensor, Tensor)> {
        let n_freqs = fft_size / 2 + 1;
        let mut cos_data = Vec::with_capacity(n_freqs * window_size);
        let mut sin_data = Vec::with_capacity(n_freqs * window_size);

        for k in 0..n_freqs {
            for n in 0..window_size {
                let angle = 2.0 * std::f32::consts::PI * k as f32 * n as f32 / fft_size as f32;
                cos_data.push(angle.cos());
                sin_data.push(-angle.sin()); // negative for DFT convention
            }
        }

        let cos_t = Tensor::new(cos_data.as_slice(), dev)?
            .reshape(&[n_freqs, window_size])?;
        let sin_t = Tensor::new(sin_data.as_slice(), dev)?
            .reshape(&[n_freqs, window_size])?;

        Ok((cos_t, sin_t))
    }

    /// Build triangular mel filterbank.
    ///
    /// Returns tensor of shape [mel_bins, fft_size/2+1].
    fn mel_filterbank(
        n_mels: usize,
        fft_size: usize,
        sample_rate: u32,
        fmin: f32,
        fmax: f32,
        dev: &Device,
    ) -> Result<Tensor> {
        let n_freqs = fft_size / 2 + 1;

        // Convert Hz boundaries to mel scale
        let mel_min = Self::hz_to_mel(fmin);
        let mel_max = Self::hz_to_mel(fmax);

        // n_mels + 2 equally spaced points in mel domain (includes edges)
        let n_points = n_mels + 2;
        let mel_points: Vec<f32> = (0..n_points)
            .map(|i| mel_min + (mel_max - mel_min) * i as f32 / (n_points - 1) as f32)
            .collect();

        // Convert mel points back to Hz, then to FFT bin indices
        let bin_indices: Vec<f32> = mel_points
            .iter()
            .map(|&m| {
                let hz = Self::mel_to_hz(m);
                hz * fft_size as f32 / sample_rate as f32
            })
            .collect();

        // Build triangular filters
        let mut bank = vec![0.0f32; n_mels * n_freqs];
        for m in 0..n_mels {
            let left = bin_indices[m];
            let center = bin_indices[m + 1];
            let right = bin_indices[m + 2];

            for k in 0..n_freqs {
                let kf = k as f32;
                let val = if kf >= left && kf <= center && center > left {
                    (kf - left) / (center - left)
                } else if kf > center && kf <= right && right > center {
                    (right - kf) / (right - center)
                } else {
                    0.0
                };
                bank[m * n_freqs + k] = val;
            }
        }

        Tensor::new(bank.as_slice(), dev)?.reshape(&[n_mels, n_freqs])
    }

    fn hz_to_mel(hz: f32) -> f32 {
        2595.0 * (1.0 + hz / 700.0).log10()
    }

    fn mel_to_hz(mel: f32) -> f32 {
        700.0 * (10.0f32.powf(mel / 2595.0) - 1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mel_creation() {
        let dev = Device::Cpu;
        let mel = MelSpectrogram::new(&dev).unwrap();
        assert_eq!(mel.window.dims(), &[WINDOW_SIZE]);
        assert_eq!(mel.mel_basis.dims(), &[MEL_BINS, FFT_SIZE / 2 + 1]);
        assert_eq!(mel.dft_cos.dims(), &[FFT_SIZE / 2 + 1, WINDOW_SIZE]);
        assert_eq!(mel.dft_sin.dims(), &[FFT_SIZE / 2 + 1, WINDOW_SIZE]);
    }

    #[test]
    fn test_mel_output_shape() {
        let dev = Device::Cpu;
        let mel = MelSpectrogram::new(&dev).unwrap();

        // 1 second of audio at 24kHz
        let audio = Tensor::zeros(&[1, 24000], DType::F32, &dev).unwrap();
        let out = mel.forward(&audio).unwrap();

        let expected_frames = MelSpectrogram::num_frames(24000, HOP_SIZE);
        assert_eq!(out.dims(), &[1, MEL_BINS, expected_frames]);
    }

    #[test]
    fn test_mel_batch() {
        let dev = Device::Cpu;
        let mel = MelSpectrogram::new(&dev).unwrap();

        let audio = Tensor::zeros(&[4, 24000], DType::F32, &dev).unwrap();
        let out = mel.forward(&audio).unwrap();

        let expected_frames = MelSpectrogram::num_frames(24000, HOP_SIZE);
        assert_eq!(out.dims(), &[4, MEL_BINS, expected_frames]);
    }

    #[test]
    fn test_mel_short_audio() {
        let dev = Device::Cpu;
        let mel = MelSpectrogram::new(&dev).unwrap();

        // Less than one window (100 samples < 600 window)
        let audio = Tensor::zeros(&[1, 100], DType::F32, &dev).unwrap();
        let out = mel.forward(&audio).unwrap();

        // Should still produce 1 frame (zero-padded)
        assert_eq!(out.dims(), &[1, MEL_BINS, 1]);
    }

    #[test]
    fn test_mel_num_frames() {
        // Exact: (24000 - 600) / 240 + 1 = 98.5 -> 98 + 1 = 98... let's compute
        // 1 + (24000 - 600) / 240 = 1 + 97 = 98
        assert_eq!(MelSpectrogram::num_frames(24000, 240), 98);
        assert_eq!(MelSpectrogram::num_frames(600, 240), 1); // exactly one window
        assert_eq!(MelSpectrogram::num_frames(840, 240), 2); // 600 + 240 = one hop
        assert_eq!(MelSpectrogram::num_frames(100, 240), 1); // short audio, still 1 frame
        assert_eq!(MelSpectrogram::num_frames(0, 240), 0); // empty
    }

    #[test]
    fn test_mel_filterbank_shape() {
        let dev = Device::Cpu;
        let mel = MelSpectrogram::new(&dev).unwrap();
        assert_eq!(mel.mel_basis.dims(), &[80, 513]);
    }

    #[test]
    fn test_mel_log_compression() {
        let dev = Device::Cpu;
        let mel = MelSpectrogram::new(&dev).unwrap();

        // Silent audio (zeros) should produce negative log values (log of small numbers)
        let audio = Tensor::zeros(&[1, 24000], DType::F32, &dev).unwrap();
        let out = mel.forward(&audio).unwrap();

        // All values should be log(1e-10) ≈ -23.03 since input is silence
        let max_val: f32 = out
            .max(candle_core::D::Minus1).unwrap()
            .max(candle_core::D::Minus1).unwrap()
            .squeeze(0).unwrap()
            .to_scalar().unwrap();
        assert!(max_val < 0.0, "Log-mel of silence should be negative, got {}", max_val);
    }

    #[test]
    fn test_mel_sine_wave() {
        let dev = Device::Cpu;
        let mel = MelSpectrogram::new(&dev).unwrap();

        // Generate 440Hz sine wave at 24kHz, 1 second
        let samples = 24000;
        let freq = 440.0f32;
        let sine: Vec<f32> = (0..samples)
            .map(|i| (2.0 * std::f32::consts::PI * freq * i as f32 / SAMPLE_RATE as f32).sin())
            .collect();
        let audio = Tensor::new(sine.as_slice(), &dev).unwrap().unsqueeze(0).unwrap(); // [1, 24000]

        let out = mel.forward(&audio).unwrap();

        // Output should have the right shape
        let expected_frames = MelSpectrogram::num_frames(samples, HOP_SIZE);
        assert_eq!(out.dims(), &[1, MEL_BINS, expected_frames]);

        // The 440Hz bin should have more energy than very high mel bins
        // Average energy across frames for each mel bin
        let mean_energy = out.squeeze(0).unwrap().mean(1).unwrap(); // [mel_bins]
        let energies: Vec<f32> = mean_energy.to_vec1().unwrap();

        // 440Hz should be in the lower mel bins; the highest bins (near 12kHz) should have less energy
        let low_bins_energy: f32 = energies[..20].iter().sum::<f32>() / 20.0;
        let high_bins_energy: f32 = energies[60..].iter().sum::<f32>() / 20.0;
        assert!(
            low_bins_energy > high_bins_energy,
            "440Hz sine should have more energy in low mel bins ({}) than high ({})",
            low_bins_energy, high_bins_energy
        );
    }

    #[test]
    fn test_mel_3d_input() {
        let dev = Device::Cpu;
        let mel = MelSpectrogram::new(&dev).unwrap();

        // [batch, 1, samples] should work (squeeze channel dim)
        let audio = Tensor::zeros(&[2, 1, 4800], DType::F32, &dev).unwrap();
        let out = mel.forward(&audio).unwrap();

        let expected_frames = MelSpectrogram::num_frames(4800, HOP_SIZE);
        assert_eq!(out.dims(), &[2, MEL_BINS, expected_frames]);
    }

    #[test]
    fn test_hz_mel_roundtrip() {
        // Verify Hz -> mel -> Hz roundtrip
        for hz in [0.0, 440.0, 1000.0, 4000.0, 8000.0, 12000.0] {
            let mel = MelSpectrogram::hz_to_mel(hz);
            let hz_back = MelSpectrogram::mel_to_hz(mel);
            assert!(
                (hz - hz_back).abs() < 0.01,
                "Roundtrip failed for {} Hz: got {} Hz",
                hz, hz_back
            );
        }
    }
}
