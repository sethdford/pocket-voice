//! Streaming LLM → TTS bridge for real-time speech synthesis.
//!
//! Buffers text from LLM streaming output and triggers TTS synthesis
//! at sentence boundaries for low-latency speech generation.

/// Sentence boundary detection for streaming TTS.
///
/// This bridge accumulates text deltas from an LLM's streaming output and detects
/// sentence boundaries (periods, exclamation marks, question marks, semicolons).
/// When a complete sentence is detected, it's emitted as ready for TTS synthesis.
///
/// # Example
/// ```
/// use sonata_pipeline::StreamingBridge;
///
/// let mut bridge = StreamingBridge::new();
///
/// // Simulate streaming LLM output
/// let sentences = bridge.push("Hello world");
/// assert!(sentences.is_empty()); // No sentence boundary yet
///
/// let sentences = bridge.push(". How are you?");
/// assert_eq!(sentences.len(), 2);
/// assert_eq!(sentences[0], "Hello world.");
/// assert_eq!(sentences[1], " How are you?");
/// ```
#[derive(Debug, Clone)]
pub struct StreamingBridge {
    buffer: String,
    sentence_delimiters: Vec<char>,
    min_chunk_chars: usize,
}

impl StreamingBridge {
    /// Create a new streaming bridge with default settings.
    ///
    /// Default sentence delimiters: '.', '!', '?', ';'
    /// Default min_chunk_chars: 10
    pub fn new() -> Self {
        Self {
            buffer: String::new(),
            sentence_delimiters: vec!['.', '!', '?', ';'],
            min_chunk_chars: 10,
        }
    }

    /// Create a streaming bridge with custom sentence delimiters.
    pub fn with_delimiters(delimiters: Vec<char>) -> Self {
        Self {
            buffer: String::new(),
            sentence_delimiters: delimiters,
            min_chunk_chars: 10,
        }
    }

    /// Create a streaming bridge with custom minimum chunk size.
    pub fn with_min_chunk(min_chunk_chars: usize) -> Self {
        Self {
            buffer: String::new(),
            sentence_delimiters: vec!['.', '!', '?', ';'],
            min_chunk_chars,
        }
    }

    /// Push a text delta from the LLM.
    ///
    /// Appends the delta to the internal buffer and checks for complete sentences.
    /// Returns any complete sentence(s) ready for TTS (each with trailing punctuation).
    ///
    /// A sentence is emitted when:
    /// 1. A sentence delimiter is found in the buffer
    /// 2. The sentence is at least `min_chunk_chars` long
    /// 3. All text up to and including the delimiter is returned
    pub fn push(&mut self, text_delta: &str) -> Vec<String> {
        self.buffer.push_str(text_delta);
        self.extract_sentences()
    }

    /// Flush remaining buffered text (for end of stream).
    ///
    /// Returns the entire remaining buffer if non-empty, or None if empty.
    /// Clears the buffer after returning.
    pub fn flush(&mut self) -> Option<String> {
        if self.buffer.is_empty() {
            None
        } else {
            Some(self.buffer.drain(..).collect())
        }
    }

    /// Reset the bridge state.
    ///
    /// Clears the internal buffer. Any pending text is lost.
    pub fn reset(&mut self) {
        self.buffer.clear();
    }

    /// Check if there's pending text in the buffer.
    pub fn has_pending(&self) -> bool {
        !self.buffer.is_empty()
    }

    /// Get the current buffer contents (for debugging).
    pub fn peek_buffer(&self) -> &str {
        &self.buffer
    }

    /// Internal helper to extract complete sentences from the buffer.
    fn extract_sentences(&mut self) -> Vec<String> {
        let mut sentences = Vec::new();
        let mut offset = 0;

        loop {
            // Find the next sentence delimiter starting from offset
            let delimiter_pos = self.buffer[offset..].find(|c| self.sentence_delimiters.contains(&c));

            match delimiter_pos {
                Some(pos) => {
                    let absolute_pos = offset + pos;
                    // Sentence length includes text from start of buffer up to and including the delimiter
                    let sentence_len = absolute_pos + 1;

                    // Check if sentence is long enough
                    if sentence_len >= self.min_chunk_chars {
                        // Extract sentence including the delimiter
                        let sentence: String = self.buffer.drain(..=absolute_pos).collect();
                        sentences.push(sentence);
                        // Reset offset since we've drained the buffer
                        offset = 0;
                    } else {
                        // Sentence too short, skip past this delimiter and look for the next one
                        offset = absolute_pos + 1;
                    }
                }
                None => {
                    // No more delimiters, stop looking
                    break;
                }
            }
        }

        sentences
    }
}

impl Default for StreamingBridge {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bridge_creation() {
        let bridge = StreamingBridge::new();
        assert_eq!(bridge.peek_buffer(), "");
        assert!(!bridge.has_pending());
        assert_eq!(bridge.sentence_delimiters.len(), 4);
        assert_eq!(bridge.min_chunk_chars, 10);
    }

    #[test]
    fn test_bridge_simple_sentence() {
        let mut bridge = StreamingBridge::new();
        let sentences = bridge.push("Hello world.");
        assert_eq!(sentences.len(), 1);
        assert_eq!(sentences[0], "Hello world.");
        assert_eq!(bridge.peek_buffer(), "");
    }

    #[test]
    fn test_bridge_multiple_sentences() {
        let mut bridge = StreamingBridge::new();
        let sentences = bridge.push("This is great. How are you doing?");
        assert_eq!(sentences.len(), 2);
        assert_eq!(sentences[0], "This is great.");
        assert_eq!(sentences[1], " How are you doing?");
        assert_eq!(bridge.peek_buffer(), "");
    }

    #[test]
    fn test_bridge_partial_sentence() {
        let mut bridge = StreamingBridge::new();
        let sentences = bridge.push("Hello");
        assert!(sentences.is_empty());
        assert_eq!(bridge.peek_buffer(), "Hello");

        let sentences = bridge.push(" world.");
        assert_eq!(sentences.len(), 1);
        assert_eq!(sentences[0], "Hello world.");
        assert_eq!(bridge.peek_buffer(), "");
    }

    #[test]
    fn test_bridge_min_chunk() {
        let mut bridge = StreamingBridge::new();
        // "Hi." is only 3 chars, below default min_chunk_chars (10)
        let sentences = bridge.push("Hi. This is a longer sentence.");
        assert_eq!(sentences.len(), 1);
        // First short sentence stays buffered, second (long) is extracted
        assert_eq!(sentences[0], "Hi. This is a longer sentence.");
        assert_eq!(bridge.peek_buffer(), "");

        // Now test that short sentences alone stay buffered
        let mut bridge2 = StreamingBridge::new();
        let sentences2 = bridge2.push("Hi.");
        assert_eq!(sentences2.len(), 0); // Too short, buffered
        assert_eq!(bridge2.peek_buffer(), "Hi.");
    }

    #[test]
    fn test_bridge_flush() {
        let mut bridge = StreamingBridge::new();
        bridge.push("Hello world");
        let remaining = bridge.flush();
        assert_eq!(remaining, Some("Hello world".to_string()));
        assert_eq!(bridge.peek_buffer(), "");
        assert!(!bridge.has_pending());
    }

    #[test]
    fn test_bridge_flush_empty() {
        let mut bridge = StreamingBridge::new();
        let remaining = bridge.flush();
        assert_eq!(remaining, None);
    }

    #[test]
    fn test_bridge_reset() {
        let mut bridge = StreamingBridge::new();
        bridge.push("Hello world");
        assert!(bridge.has_pending());
        bridge.reset();
        assert!(!bridge.has_pending());
        assert_eq!(bridge.peek_buffer(), "");
    }

    #[test]
    fn test_bridge_empty_delta() {
        let mut bridge = StreamingBridge::new();
        let sentences = bridge.push("");
        assert!(sentences.is_empty());
        assert_eq!(bridge.peek_buffer(), "");
    }

    #[test]
    fn test_bridge_has_pending() {
        let mut bridge = StreamingBridge::new();
        assert!(!bridge.has_pending());

        bridge.push("Hello world");
        assert!(bridge.has_pending());

        bridge.push(".");
        // After adding ".", we have "Hello world." (12 chars) which is long enough to emit
        assert!(!bridge.has_pending());
    }

    #[test]
    fn test_bridge_streaming_simulation() {
        let mut bridge = StreamingBridge::new();

        // Simulate word-by-word streaming
        let w1 = bridge.push("The ");
        assert!(w1.is_empty());

        let w2 = bridge.push("quick ");
        assert!(w2.is_empty());

        let w3 = bridge.push("brown ");
        assert!(w3.is_empty());

        let w4 = bridge.push("fox ");
        assert!(w4.is_empty());

        let w5 = bridge.push("jumps");
        assert!(w5.is_empty());

        // Buffer now: "The quick brown fox jumps" (26 chars, > min_chunk_chars)
        let w6 = bridge.push(".");
        assert_eq!(w6.len(), 1);
        assert_eq!(w6[0], "The quick brown fox jumps.");

        assert_eq!(bridge.peek_buffer(), "");
    }

    #[test]
    fn test_bridge_exclamation_mark() {
        let mut bridge = StreamingBridge::new();
        let sentences = bridge.push("Watch out!");
        assert_eq!(sentences.len(), 1);
        assert_eq!(sentences[0], "Watch out!");
    }

    #[test]
    fn test_bridge_question_mark() {
        let mut bridge = StreamingBridge::new();
        let sentences = bridge.push("Are you okay?");
        assert_eq!(sentences.len(), 1);
        assert_eq!(sentences[0], "Are you okay?");
    }

    #[test]
    fn test_bridge_semicolon() {
        let mut bridge = StreamingBridge::new();
        let sentences = bridge.push("First part; second part.");
        assert_eq!(sentences.len(), 2);
        assert_eq!(sentences[0], "First part;");
        assert_eq!(sentences[1], " second part.");
    }

    #[test]
    fn test_bridge_consecutive_sentences() {
        let mut bridge = StreamingBridge::new();
        let sentences = bridge.push("First sentence here. Second one too. Third as well.");
        assert_eq!(sentences.len(), 3);
        assert_eq!(sentences[0], "First sentence here.");
        assert_eq!(sentences[1], " Second one too.");
        assert_eq!(sentences[2], " Third as well.");
    }

    #[test]
    fn test_bridge_mixed_delimiters() {
        let mut bridge = StreamingBridge::new();
        let sentences = bridge.push("What is that? I said no! Really now.");
        assert_eq!(sentences.len(), 3);
        assert_eq!(sentences[0], "What is that?");
        assert_eq!(sentences[1], " I said no!");
        assert_eq!(sentences[2], " Really now.");
    }
}
