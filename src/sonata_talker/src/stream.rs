/// Ring buffer for audio code frames (pre-allocated, zero-alloc in hot path).
pub struct AudioRingBuffer {
    buf: Vec<[u32; 8]>,
    head: usize,
    tail: usize,
    capacity: usize,
    count: usize,
}

impl AudioRingBuffer {
    pub fn new(capacity: usize) -> Self {
        Self {
            buf: vec![[0u32; 8]; capacity],
            head: 0, tail: 0, capacity, count: 0,
        }
    }

    pub fn push(&mut self, codes: &[u32; 8]) {
        self.buf[self.tail] = *codes;
        self.tail = (self.tail + 1) % self.capacity;
        if self.count < self.capacity {
            self.count += 1;
        } else {
            self.head = (self.head + 1) % self.capacity;
        }
    }

    pub fn pop(&mut self) -> Option<[u32; 8]> {
        if self.count == 0 { return None; }
        let frame = self.buf[self.head];
        self.head = (self.head + 1) % self.capacity;
        self.count -= 1;
        Some(frame)
    }

    pub fn len(&self) -> usize { self.count }
    pub fn is_full(&self) -> bool { self.count == self.capacity }
}

/// Dual-stream token engine for full-duplex S2S.
/// Handles acoustic delay (tau) and user/assistant stream interleaving.
pub struct DualStream {
    tau: usize,                        // acoustic delay in frames
    user_buffer: AudioRingBuffer,      // delayed user codes
    asst_prev: [u32; 8],              // previous assistant output
    frame_count: usize,
}

impl DualStream {
    pub fn new(tau: usize) -> Self {
        Self {
            tau,
            user_buffer: AudioRingBuffer::new(tau + 1),
            asst_prev: [0u32; 8],
            frame_count: 0,
        }
    }

    /// Interleave user + assistant codes + optional text token.
    pub fn interleave(
        user_codes: &[u32; 8], asst_codes: &[u32; 8], text_token: Option<u32>,
    ) -> Vec<u32> {
        let mut out = Vec::with_capacity(17);
        out.extend_from_slice(user_codes);
        out.extend_from_slice(asst_codes);
        out.push(text_token.unwrap_or(0));
        out
    }

    /// Process one 12.5Hz frame:
    /// - Buffers user codes for acoustic delay
    /// - Returns interleaved tokens when delay is filled
    pub fn process_frame(
        &mut self, user_codes: &[u32; 8], text_token: Option<u32>,
    ) -> Option<Vec<u32>> {
        let mut codes = [0u32; 8];
        codes.copy_from_slice(user_codes);
        self.user_buffer.push(&codes);
        self.frame_count += 1;

        if self.frame_count <= self.tau {
            return None; // delay not yet filled
        }

        let delayed_user = self.user_buffer.pop().unwrap();
        Some(Self::interleave(&delayed_user, &self.asst_prev, text_token))
    }

    /// Update assistant output codes (after Talker generates them).
    pub fn set_assistant_output(&mut self, codes: &[u32; 8]) {
        self.asst_prev.copy_from_slice(codes);
    }

    pub fn reset(&mut self) {
        self.user_buffer = AudioRingBuffer::new(self.tau + 1);
        self.asst_prev = [0u32; 8];
        self.frame_count = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_interleave_tokens() {
        let user_codes = [1u32, 2, 3, 4, 5, 6, 7, 8];
        let asst_codes = [10u32, 20, 30, 40, 50, 60, 70, 80];
        let text_token = Some(100u32);
        let interleaved = DualStream::interleave(&user_codes, &asst_codes, text_token);
        assert_eq!(interleaved.len(), 17); // 8 + 8 + 1
        assert_eq!(interleaved[0], 1);   // user semantic
        assert_eq!(interleaved[8], 10);  // asst semantic
        assert_eq!(interleaved[16], 100); // text token
    }

    #[test]
    fn test_ring_buffer_push_pop() {
        let mut ring = AudioRingBuffer::new(4);
        ring.push(&[1, 2, 3, 4, 5, 6, 7, 8]);
        ring.push(&[10, 20, 30, 40, 50, 60, 70, 80]);
        assert_eq!(ring.len(), 2);
        let oldest = ring.pop().unwrap();
        assert_eq!(oldest, [1, 2, 3, 4, 5, 6, 7, 8]);
    }

    #[test]
    fn test_acoustic_delay_tau1() {
        let mut stream = DualStream::new(1); // tau=1 (default, 80ms)
        let out0 = stream.process_frame(&[0; 8], None);
        assert!(out0.is_none()); // delay not filled
        let out1 = stream.process_frame(&[1; 8], None);
        assert!(out1.is_some()); // delay filled, output frame 0
    }

    #[test]
    fn test_acoustic_delay_tau2() {
        let mut stream = DualStream::new(2); // tau=2 (160ms)
        let out0 = stream.process_frame(&[0; 8], None);
        assert!(out0.is_none());
        let out1 = stream.process_frame(&[1; 8], None);
        assert!(out1.is_none());
        let out2 = stream.process_frame(&[2; 8], None);
        assert!(out2.is_some()); // delay filled, output frame 0
    }
}
