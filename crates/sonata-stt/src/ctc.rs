//! CTC (Connectionist Temporal Classification) decoder.
//!
//! Implements greedy decoding: argmax → collapse repeats → remove blanks.

use candle_core::{Result, Tensor, D};

pub const BLANK_TOKEN: u32 = 0;
pub const TEXT_VOCAB_SIZE: usize = 32000;

/// Greedy CTC decoding: argmax → collapse repeats → remove blanks.
///
/// # Arguments
/// * `logits` - Shape [B, T, vocab_size] - raw CTC logits from the network
///
/// # Returns
/// Vector of decoded token sequences (one per batch element)
pub fn greedy_decode(logits: &Tensor) -> Result<Vec<Vec<u32>>> {
    // logits: [B, T, vocab_size]
    let batch_size = logits.dim(0)?;

    // Use neg().argmin() as fallback for argmax in older candle versions
    let neg_logits = logits.neg()?;
    let predictions = neg_logits.argmin(D::Minus1)?; // [B, T]

    let mut results = Vec::new();
    for b in 0..batch_size {
        let pred_b = predictions.get(b)?; // [T]

        // Try to get as u32 first, fall back to i64
        let pred_u32: Vec<u32> = match pred_b.to_vec1::<u32>() {
            Ok(v) => v,
            Err(_) => {
                // Fallback: argmin may return i64
                let vec_i64: Vec<i64> = pred_b.to_vec1::<i64>()?;
                vec_i64.iter().map(|&x| x as u32).collect()
            }
        };

        let decoded = collapse_and_remove_blanks(&pred_u32);
        results.push(decoded);
    }
    Ok(results)
}

/// Collapse consecutive duplicate tokens and remove blanks.
///
/// CTC encodes with repeats and blanks:
/// - Repeats: [1, 1, 2] → [1, 2]
/// - Blanks: [1, 0, 2, 0] → [1, 2]
fn collapse_and_remove_blanks(tokens: &[u32]) -> Vec<u32> {
    let mut result = Vec::new();
    let mut prev = None;

    for &tok in tokens {
        // Skip blanks and consecutive duplicates
        if tok != BLANK_TOKEN && Some(tok) != prev {
            result.push(tok);
        }
        prev = Some(tok);
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_collapse_blanks() {
        let tokens = vec![0, 1, 1, 0, 2, 2, 2, 0, 3];
        let decoded = collapse_and_remove_blanks(&tokens);
        assert_eq!(decoded, vec![1, 2, 3]);
    }

    #[test]
    fn test_collapse_all_blanks() {
        let tokens = vec![0, 0, 0, 0];
        let decoded = collapse_and_remove_blanks(&tokens);
        assert_eq!(decoded, Vec::<u32>::new());
    }

    #[test]
    fn test_collapse_no_blanks() {
        let tokens = vec![1, 2, 3, 4, 5];
        let decoded = collapse_and_remove_blanks(&tokens);
        assert_eq!(decoded, vec![1, 2, 3, 4, 5]);
    }

    #[test]
    fn test_collapse_single_token() {
        let tokens = vec![5];
        let decoded = collapse_and_remove_blanks(&tokens);
        assert_eq!(decoded, vec![5]);
    }

    #[test]
    fn test_collapse_single_blank() {
        let tokens = vec![0];
        let decoded = collapse_and_remove_blanks(&tokens);
        assert_eq!(decoded, Vec::<u32>::new());
    }

    #[test]
    fn test_collapse_repeated_tokens() {
        let tokens = vec![1, 1, 1, 2, 2, 3];
        let decoded = collapse_and_remove_blanks(&tokens);
        assert_eq!(decoded, vec![1, 2, 3]);
    }

    #[test]
    fn test_collapse_blank_token_constant() {
        assert_eq!(BLANK_TOKEN, 0);
        assert_eq!(TEXT_VOCAB_SIZE, 32000);
    }

    #[test]
    fn test_greedy_decode_simple() {
        use candle_core::DType;
        let dev = candle_core::Device::Cpu;
        let logits = Tensor::zeros(&[1, 5, 100], DType::F32, &dev).unwrap();
        let decoded = greedy_decode(&logits).unwrap();
        assert_eq!(decoded.len(), 1);
    }

    #[test]
    fn test_greedy_decode_batch() {
        use candle_core::DType;
        let dev = candle_core::Device::Cpu;
        let logits = Tensor::zeros(&[4, 10, TEXT_VOCAB_SIZE], DType::F32, &dev).unwrap();
        let decoded = greedy_decode(&logits).unwrap();
        assert_eq!(decoded.len(), 4);
    }
}
