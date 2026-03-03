//! Streaming STT: process audio in chunks with state caching.

use candle_core::Tensor;
use std::collections::VecDeque;

/// Streaming state for chunk-based STT processing.
///
/// Manages buffering and chunking for real-time transcription without
/// waiting for the entire audio to be available.
pub struct StreamingState {
    pub buffer: VecDeque<Tensor>,
    pub chunk_size: usize,
    pub overlap: usize,
}

impl StreamingState {
    /// Create a new streaming state with specified chunk size and overlap.
    ///
    /// # Arguments
    /// * `chunk_size` - Number of frames per chunk (e.g., 100 frames = 2s at 50Hz)
    /// * `overlap` - Number of frames to overlap between chunks for continuity
    pub fn new(chunk_size: usize, overlap: usize) -> Self {
        Self {
            buffer: VecDeque::new(),
            chunk_size,
            overlap,
        }
    }

    /// Add a new audio chunk to the buffer.
    pub fn add_chunk(&mut self, chunk: Tensor) {
        self.buffer.push_back(chunk);
    }

    /// Check if we have enough data to process.
    pub fn has_enough(&self) -> bool {
        !self.buffer.is_empty()
    }

    /// Get the number of buffered chunks.
    pub fn buffer_size(&self) -> usize {
        self.buffer.len()
    }

    /// Take the next chunk from the buffer for processing. O(1) via VecDeque.
    pub fn take_chunk(&mut self) -> Option<Tensor> {
        self.buffer.pop_front()
    }

    /// Peek at the next chunk without removing it.
    pub fn peek_chunk(&self) -> Option<&Tensor> {
        self.buffer.front()
    }

    /// Clear all buffered chunks.
    pub fn reset(&mut self) {
        self.buffer.clear();
    }

    /// Get configuration info for this streaming session.
    pub fn info(&self) -> StreamingInfo {
        StreamingInfo {
            chunk_size: self.chunk_size,
            overlap: self.overlap,
            buffered_chunks: self.buffer.len(),
        }
    }
}

/// Information about a streaming session.
#[derive(Debug, Clone)]
pub struct StreamingInfo {
    pub chunk_size: usize,
    pub overlap: usize,
    pub buffered_chunks: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, DType, Tensor};

    #[test]
    fn test_streaming_state_new() {
        let state = StreamingState::new(100, 20);
        assert_eq!(state.chunk_size, 100);
        assert_eq!(state.overlap, 20);
        assert_eq!(state.buffer_size(), 0);
    }

    #[test]
    fn test_streaming_state_add_chunk() {
        let dev = Device::Cpu;
        let mut state = StreamingState::new(100, 20);

        let chunk = Tensor::zeros(&[50, 512], DType::F32, &dev).unwrap();
        state.add_chunk(chunk);

        assert_eq!(state.buffer_size(), 1);
        assert!(state.has_enough());
    }

    #[test]
    fn test_streaming_state_take_chunk() {
        let dev = Device::Cpu;
        let mut state = StreamingState::new(100, 20);

        let chunk = Tensor::zeros(&[50, 512], DType::F32, &dev).unwrap();
        state.add_chunk(chunk);

        assert!(state.take_chunk().is_some());
        assert_eq!(state.buffer_size(), 0);
        assert!(!state.has_enough());
    }

    #[test]
    fn test_streaming_state_multiple_chunks() {
        let dev = Device::Cpu;
        let mut state = StreamingState::new(100, 20);

        for _ in 0..5 {
            let chunk = Tensor::zeros(&[50, 512], DType::F32, &dev).unwrap();
            state.add_chunk(chunk);
        }

        assert_eq!(state.buffer_size(), 5);

        for i in (0..5).rev() {
            assert_eq!(state.buffer_size(), i + 1);
            let _chunk = state.take_chunk();
        }

        assert_eq!(state.buffer_size(), 0);
    }

    #[test]
    fn test_streaming_state_peek_chunk() {
        let dev = Device::Cpu;
        let mut state = StreamingState::new(100, 20);

        let chunk = Tensor::zeros(&[50, 512], DType::F32, &dev).unwrap();
        state.add_chunk(chunk);

        // Peek shouldn't remove
        assert!(state.peek_chunk().is_some());
        assert_eq!(state.buffer_size(), 1);

        // Take should remove
        assert!(state.take_chunk().is_some());
        assert_eq!(state.buffer_size(), 0);
    }

    #[test]
    fn test_streaming_state_reset() {
        let dev = Device::Cpu;
        let mut state = StreamingState::new(100, 20);

        for _ in 0..10 {
            let chunk = Tensor::zeros(&[50, 512], DType::F32, &dev).unwrap();
            state.add_chunk(chunk);
        }

        assert_eq!(state.buffer_size(), 10);
        state.reset();
        assert_eq!(state.buffer_size(), 0);
        assert!(!state.has_enough());
    }

    #[test]
    fn test_streaming_state_info() {
        let dev = Device::Cpu;
        let mut state = StreamingState::new(100, 20);

        let chunk = Tensor::zeros(&[50, 512], DType::F32, &dev).unwrap();
        state.add_chunk(chunk);

        let info = state.info();
        assert_eq!(info.chunk_size, 100);
        assert_eq!(info.overlap, 20);
        assert_eq!(info.buffered_chunks, 1);
    }

    #[test]
    fn test_streaming_state_take_empty() {
        let mut state = StreamingState::new(100, 20);
        assert!(state.take_chunk().is_none());
    }

    #[test]
    fn test_streaming_state_peek_empty() {
        let state = StreamingState::new(100, 20);
        assert!(state.peek_chunk().is_none());
    }

    #[test]
    fn test_streaming_fifo_order() {
        let dev = Device::Cpu;
        let mut state = StreamingState::new(100, 20);

        // Add chunks with distinct shapes to verify FIFO order
        for i in 1..=3 {
            let chunk = Tensor::zeros(&[i * 10, 512], DType::F32, &dev).unwrap();
            state.add_chunk(chunk);
        }

        // Should come out in FIFO order
        let c1 = state.take_chunk().unwrap();
        assert_eq!(c1.dims()[0], 10);
        let c2 = state.take_chunk().unwrap();
        assert_eq!(c2.dims()[0], 20);
        let c3 = state.take_chunk().unwrap();
        assert_eq!(c3.dims()[0], 30);
    }
}
