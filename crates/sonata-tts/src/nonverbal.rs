//! Nonverbal token vocabulary for expressive TTS.

use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::{Embedding, VarBuilder};
use sonata_common::{NonverbalTag, NUM_NONVERBAL_TOKENS};

pub struct NonverbalEncoder {
    embed: Embedding,
}

impl NonverbalEncoder {
    pub fn new(embed_dim: usize, dev: &Device) -> Result<Self> {
        let vb = VarBuilder::zeros(DType::F32, dev);
        let embed = candle_nn::embedding(NUM_NONVERBAL_TOKENS, embed_dim, vb.pp("nonverbal"))?;
        Ok(Self { embed })
    }

    pub fn encode(&self, tag: NonverbalTag, dev: &Device) -> Result<Tensor> {
        let id = Tensor::new(&[tag as u32], dev)?;
        self.embed.forward(&id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_nonverbal_encoder() {
        let dev = Device::Cpu;
        let enc = NonverbalEncoder::new(512, &dev).unwrap();
        let laugh = enc.encode(NonverbalTag::Laugh, &dev).unwrap();
        assert_eq!(laugh.dims(), &[1, 512]);
    }

    #[test]
    fn test_all_nonverbal_tags() {
        let dev = Device::Cpu;
        let enc = NonverbalEncoder::new(256, &dev).unwrap();
        // All 24 tags should encode successfully
        for tag_id in 0..24u32 {
            let id = Tensor::new(&[tag_id], &dev).unwrap();
            let emb = enc.embed.forward(&id).unwrap();
            assert_eq!(emb.dims(), &[1, 256]);
        }
    }
}
