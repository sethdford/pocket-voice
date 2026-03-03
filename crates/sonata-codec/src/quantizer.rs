//! Residual Vector Quantizer with split semantic/acoustic codebooks.

use candle_core::{Device, Result, Tensor, D};
use sonata_common::{ACOUSTIC_CODEBOOKS, SEMANTIC_CODEBOOKS};

pub struct VectorQuantizer {
    codebook: Tensor,
    project_in: Option<Tensor>,
    project_out: Option<Tensor>,
}

impl VectorQuantizer {
    pub fn new(
        input_dim: usize,
        codebook_size: usize,
        codebook_dim: usize,
        dev: &Device,
    ) -> Result<Self> {
        let codebook = Tensor::randn(0.0f32, 0.02, (codebook_size, codebook_dim), dev)?;
        let (project_in, project_out) = if input_dim != codebook_dim {
            let pi = Tensor::randn(0.0f32, 0.02, (input_dim, codebook_dim), dev)?;
            let po = Tensor::randn(0.0f32, 0.02, (codebook_dim, input_dim), dev)?;
            (Some(pi), Some(po))
        } else {
            (None, None)
        };
        Ok(Self {
            codebook,
            project_in,
            project_out,
        })
    }

    /// Get the codebook embedding tensor [CODEBOOK_SIZE, CODEBOOK_DIM].
    pub fn codebook(&self) -> &Tensor {
        &self.codebook
    }

    pub fn encode(&self, z: &Tensor) -> Result<(Tensor, Tensor)> {
        // z shape: [batch, channels, length]
        let z_t = z.transpose(1, 2)?; // [batch, length, channels]
        let z_proj = if let Some(ref proj) = self.project_in {
            // proj is [input_dim, codebook_dim], z_t is [batch, length, input_dim]
            // We need to reshape z_t to [batch*length, input_dim], matmul, then reshape back
            let (batch, length, _) = (z_t.dim(0)?, z_t.dim(1)?, z_t.dim(2)?);
            let z_flat = z_t.reshape((batch * length, z_t.dim(2)?))?; // [batch*length, input_dim]
            let z_proj_flat = z_flat.matmul(proj)?; // [batch*length, codebook_dim]
            z_proj_flat.reshape((batch, length, proj.dim(1)?))?
        } else {
            z_t.clone()
        };

        // Compute distances: ||z - e||^2 = ||z||^2 + ||e||^2 - 2 * z @ e^T
        // z_proj: [batch, length, codebook_dim]
        // codebook: [codebook_size, codebook_dim]
        let z_sq = z_proj.sqr()?.sum(D::Minus1)?.unsqueeze(D::Minus1)?; // [batch, length, 1]
        let codebook_t = self.codebook.t()?; // [codebook_dim, codebook_size]
        // e_sq = sum of codebook_dim along codebook entries, shape [codebook_size]
        let e_sq = self.codebook
            .sqr()?
            .sum(D::Minus1)? // sum along last dim (codebook_dim) -> [codebook_size]
            .unsqueeze(0)?
            .unsqueeze(0)?; // [1, 1, codebook_size]
        // z_proj is [batch, length, codebook_dim], codebook_t is [codebook_dim, codebook_size]
        // Flatten z_proj to [batch*length, codebook_dim], matmul, reshape back
        let (batch, length, _) = (z_proj.dim(0)?, z_proj.dim(1)?, z_proj.dim(2)?);
        let z_proj_flat = z_proj.reshape((batch * length, z_proj.dim(2)?))?;
        let ze_flat = z_proj_flat.matmul(&codebook_t)?; // [batch*length, codebook_size]
        let ze = ze_flat.reshape((batch, length, self.codebook.dim(0)?))?; // [batch, length, codebook_size]

        let two = Tensor::new(&[2.0f32], z.device())?;
        let dist = (z_sq.broadcast_add(&e_sq)? - ze.broadcast_mul(&two)?)?;

        // Get nearest codebook entry
        let codes = dist.argmin(D::Minus1)?;
        let quantized = self.lookup(&codes)?;
        Ok((codes, quantized))
    }

    pub fn lookup(&self, codes: &Tensor) -> Result<Tensor> {
        // codes shape: [batch, length]
        let flat = codes.flatten_all()?;
        let emb = self.codebook.index_select(&flat, 0)?;

        // Reshape back to [batch, length, codebook_dim]
        let shape = codes.dims().to_vec();
        let codebook_dim = self.codebook.dim(1)?;
        let emb = emb.reshape((shape[0], shape[1], codebook_dim))?;

        // Project back to input space if needed
        let emb = if let Some(ref proj) = self.project_out {
            // emb is [batch, length, codebook_dim], proj is [codebook_dim, input_dim]
            // Reshape to [batch*length, codebook_dim], matmul, reshape back
            let (batch, length, _) = (emb.dim(0)?, emb.dim(1)?, emb.dim(2)?);
            let emb_flat = emb.reshape((batch * length, emb.dim(2)?))?;
            let emb_proj_flat = emb_flat.matmul(proj)?;
            emb_proj_flat.reshape((batch, length, proj.dim(1)?))?
        } else {
            emb
        };

        // Transpose back to [batch, channels, length]
        emb.transpose(1, 2)
    }
}

pub struct ResidualVQ {
    quantizers: Vec<VectorQuantizer>,
}

impl ResidualVQ {
    pub fn new(
        input_dim: usize,
        num_codebooks: usize,
        codebook_size: usize,
        codebook_dim: usize,
        dev: &Device,
    ) -> Result<Self> {
        let mut quantizers = Vec::new();
        for _ in 0..num_codebooks {
            quantizers.push(VectorQuantizer::new(
                input_dim,
                codebook_size,
                codebook_dim,
                dev,
            )?);
        }
        Ok(Self { quantizers })
    }

    /// Get the codebook embedding tensor for a specific codebook index.
    ///
    /// Returns the raw codebook tensor [CODEBOOK_SIZE, CODEBOOK_DIM].
    pub fn get_codebook_embeddings(&self, book_idx: usize) -> Result<&Tensor> {
        self.quantizers
            .get(book_idx)
            .map(|vq| vq.codebook())
            .ok_or_else(|| candle_core::Error::Msg(
                format!("Codebook index {} out of range (have {})", book_idx, self.quantizers.len())
            ))
    }

    /// Number of codebooks in this RVQ.
    pub fn num_codebooks(&self) -> usize {
        self.quantizers.len()
    }

    pub fn encode(&self, z: &Tensor) -> Result<(Tensor, Tensor)> {
        let mut residual = z.clone();
        let mut all_codes = Vec::new();
        let mut quantized_sum = Tensor::zeros_like(z)?;

        for vq in &self.quantizers {
            let (codes, quantized) = vq.encode(&residual)?;
            all_codes.push(codes.unsqueeze(1)?);
            quantized_sum = (quantized_sum + &quantized)?;
            residual = (residual - quantized)?;
        }

        // Stack codes: [batch, num_codebooks, length]
        let codes = Tensor::cat(&all_codes, 1)?;
        Ok((codes, quantized_sum))
    }

    pub fn decode(&self, codes: &Tensor) -> Result<Tensor> {
        // codes shape: [batch, num_codebooks, length]
        let mut output = None;
        for (i, vq) in self.quantizers.iter().enumerate() {
            let book_codes = codes.narrow(1, i, 1)?.squeeze(1)?;
            let quantized = vq.lookup(&book_codes)?;
            output = Some(match output {
                Some(o) => (o + quantized)?,
                None => quantized,
            });
        }
        output.ok_or_else(|| candle_core::Error::Msg("No codebooks".to_string()))
    }

    pub fn split_codes(&self, codes: &Tensor) -> Result<(Tensor, Tensor)> {
        let semantic = codes.narrow(1, 0, SEMANTIC_CODEBOOKS)?;
        let acoustic = codes.narrow(1, SEMANTIC_CODEBOOKS, ACOUSTIC_CODEBOOKS)?;
        Ok((semantic, acoustic))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sonata_common::{CODEBOOK_DIM, CODEBOOK_SIZE, NUM_CODEBOOKS};

    #[test]
    fn test_rvq_output_shape() {
        let dev = &Device::Cpu;
        let rvq = ResidualVQ::new(512, NUM_CODEBOOKS, CODEBOOK_SIZE, CODEBOOK_DIM, dev).unwrap();
        let z = Tensor::randn(0.0f32, 1.0, (1, 512, 50), dev).unwrap();
        let (codes, _) = rvq.encode(&z).unwrap();
        assert_eq!(codes.dims(), &[1, NUM_CODEBOOKS, 50]);
    }

    #[test]
    fn test_rvq_roundtrip() {
        let dev = &Device::Cpu;
        let rvq = ResidualVQ::new(512, NUM_CODEBOOKS, CODEBOOK_SIZE, CODEBOOK_DIM, dev).unwrap();
        let z = Tensor::randn(0.0f32, 0.1, (1, 512, 10), dev).unwrap();
        let (codes, _) = rvq.encode(&z).unwrap();
        let z_hat = rvq.decode(&codes).unwrap();
        assert_eq!(z_hat.dims(), z.dims());
    }

    #[test]
    fn test_semantic_acoustic_split() {
        let dev = &Device::Cpu;
        let rvq = ResidualVQ::new(512, NUM_CODEBOOKS, CODEBOOK_SIZE, CODEBOOK_DIM, dev).unwrap();
        let z = Tensor::randn(0.0f32, 1.0, (1, 512, 5), dev).unwrap();
        let (codes, _) = rvq.encode(&z).unwrap();
        let (semantic, acoustic) = rvq.split_codes(&codes).unwrap();
        assert_eq!(semantic.dim(1).unwrap(), SEMANTIC_CODEBOOKS);
        assert_eq!(acoustic.dim(1).unwrap(), ACOUSTIC_CODEBOOKS);
    }

    #[test]
    fn test_quantizer_codebook_size() {
        let dev = &Device::Cpu;
        let vq = VectorQuantizer::new(512, CODEBOOK_SIZE, CODEBOOK_DIM, dev).unwrap();
        assert_eq!(vq.codebook.dim(0).unwrap(), CODEBOOK_SIZE);
        assert_eq!(vq.codebook.dim(1).unwrap(), CODEBOOK_DIM);
    }

    #[test]
    fn test_rvq_batch_processing() {
        let dev = &Device::Cpu;
        let rvq = ResidualVQ::new(512, NUM_CODEBOOKS, CODEBOOK_SIZE, CODEBOOK_DIM, dev).unwrap();
        let z = Tensor::randn(0.0f32, 1.0, (4, 512, 20), dev).unwrap();
        let (codes, quantized) = rvq.encode(&z).unwrap();
        assert_eq!(codes.dim(0).unwrap(), 4);
        assert_eq!(quantized.dim(0).unwrap(), 4);
    }
}
