//! Dual-stream token interleaver for full-duplex conversation.
//!
//! Interleaves user (STT) and agent (TTS) token streams on a shared timeline,
//! enabling overlapping speech detection and Moshi-style multi-stream conversation context.
//!
//! # Architecture
//!
//! Maintains two separate token streams (user + agent) with timestamps, merging them
//! into a unified sequence sorted by time. Supports overlapping speech where both
//! streams produce tokens simultaneously.
//!
//! # Example
//! ```ignore
//! let mut interleaver = DualStreamInterleaver::new();
//! interleaver.push_user_token(5, 1000);  // token 5 at 1000ms
//! interleaver.push_agent_token(42, 950); // token 42 at 950ms
//! let merged = interleaver.get_interleaved();
//! assert_eq!(merged[0].source, StreamSource::Agent);   // 950ms comes first
//! assert_eq!(merged[1].source, StreamSource::User);    // 1000ms comes second
//! ```

/// Source of a token in the dual stream.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StreamSource {
    User,
    Agent,
}

/// A single token with source and timestamp.
#[derive(Debug, Clone)]
pub struct StreamToken {
    pub source: StreamSource,
    pub timestamp_ms: u64,
    pub token_id: u32,
}

/// Dual-stream token interleaver for full-duplex conversation.
///
/// Maintains user and agent token streams separately, with methods to:
/// - Push tokens from either stream
/// - Get interleaved (merged) token sequence sorted by timestamp
/// - Extract context windows for LLM feeding
/// - Garbage-collect old tokens to manage memory
#[derive(Debug, Clone)]
pub struct DualStreamInterleaver {
    user_tokens: Vec<StreamToken>,
    agent_tokens: Vec<StreamToken>,
}

impl DualStreamInterleaver {
    /// Create a new empty dual-stream interleaver.
    pub fn new() -> Self {
        Self {
            user_tokens: Vec::new(),
            agent_tokens: Vec::new(),
        }
    }

    /// Push a user (STT) token.
    pub fn push_user_token(&mut self, token_id: u32, timestamp_ms: u64) {
        self.user_tokens.push(StreamToken {
            source: StreamSource::User,
            timestamp_ms,
            token_id,
        });
    }

    /// Push an agent (TTS) token.
    pub fn push_agent_token(&mut self, token_id: u32, timestamp_ms: u64) {
        self.agent_tokens.push(StreamToken {
            source: StreamSource::Agent,
            timestamp_ms,
            token_id,
        });
    }

    /// Get the full interleaved token sequence sorted by timestamp.
    ///
    /// Returns a vector of references to tokens from both streams, ordered by
    /// increasing timestamp. Tokens with the same timestamp maintain their
    /// relative order (user first, then agent).
    pub fn get_interleaved(&self) -> Vec<&StreamToken> {
        let mut merged = Vec::new();

        // Collect all tokens
        merged.extend(self.user_tokens.iter());
        merged.extend(self.agent_tokens.iter());

        // Sort by timestamp (stable sort maintains relative order for equal timestamps)
        merged.sort_by(|a, b| {
            a.timestamp_ms.cmp(&b.timestamp_ms)
        });

        merged
    }

    /// Get a context window of the last N tokens.
    ///
    /// Useful for feeding to the LLM with bounded context size.
    /// Returns the most recent N tokens from the interleaved sequence.
    pub fn get_context_window(&self, max_tokens: usize) -> Vec<&StreamToken> {
        let interleaved = self.get_interleaved();
        if interleaved.len() <= max_tokens {
            interleaved
        } else {
            interleaved[interleaved.len() - max_tokens..].to_vec()
        }
    }

    /// Remove all tokens with timestamp < cutoff_ms.
    ///
    /// Used for garbage collection to prevent unbounded memory growth
    /// in long conversation sessions.
    pub fn clear_before(&mut self, timestamp_ms: u64) {
        self.user_tokens.retain(|t| t.timestamp_ms >= timestamp_ms);
        self.agent_tokens.retain(|t| t.timestamp_ms >= timestamp_ms);
    }

    /// Get the number of user tokens currently stored.
    pub fn user_token_count(&self) -> usize {
        self.user_tokens.len()
    }

    /// Get the number of agent tokens currently stored.
    pub fn agent_token_count(&self) -> usize {
        self.agent_tokens.len()
    }

    /// Get the total number of tokens (user + agent).
    pub fn total_token_count(&self) -> usize {
        self.user_tokens.len() + self.agent_tokens.len()
    }

    /// Get the earliest timestamp from either stream, or None if empty.
    pub fn earliest_timestamp(&self) -> Option<u64> {
        let user_min = self.user_tokens.iter().map(|t| t.timestamp_ms).min();
        let agent_min = self.agent_tokens.iter().map(|t| t.timestamp_ms).min();
        match (user_min, agent_min) {
            (Some(u), Some(a)) => Some(u.min(a)),
            (Some(u), None) => Some(u),
            (None, Some(a)) => Some(a),
            (None, None) => None,
        }
    }

    /// Get the latest timestamp from either stream, or None if empty.
    pub fn latest_timestamp(&self) -> Option<u64> {
        let user_max = self.user_tokens.iter().map(|t| t.timestamp_ms).max();
        let agent_max = self.agent_tokens.iter().map(|t| t.timestamp_ms).max();
        match (user_max, agent_max) {
            (Some(u), Some(a)) => Some(u.max(a)),
            (Some(u), None) => Some(u),
            (None, Some(a)) => Some(a),
            (None, None) => None,
        }
    }

    /// Clear all tokens from both streams.
    pub fn reset(&mut self) {
        self.user_tokens.clear();
        self.agent_tokens.clear();
    }
}

impl Default for DualStreamInterleaver {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_interleaver_empty() {
        let interleaver = DualStreamInterleaver::new();
        assert_eq!(interleaver.user_token_count(), 0);
        assert_eq!(interleaver.agent_token_count(), 0);
        assert_eq!(interleaver.total_token_count(), 0);
        assert_eq!(interleaver.get_interleaved().len(), 0);
    }

    #[test]
    fn test_interleaver_single_stream() {
        let mut interleaver = DualStreamInterleaver::new();
        interleaver.push_user_token(5, 1000);
        interleaver.push_user_token(10, 2000);
        interleaver.push_user_token(15, 3000);

        assert_eq!(interleaver.user_token_count(), 3);
        assert_eq!(interleaver.agent_token_count(), 0);
        assert_eq!(interleaver.total_token_count(), 3);

        let merged = interleaver.get_interleaved();
        assert_eq!(merged.len(), 3);
        assert_eq!(merged[0].token_id, 5);
        assert_eq!(merged[1].token_id, 10);
        assert_eq!(merged[2].token_id, 15);
    }

    #[test]
    fn test_interleaver_dual_streams() {
        let mut interleaver = DualStreamInterleaver::new();
        interleaver.push_user_token(5, 1000);
        interleaver.push_agent_token(42, 500);
        interleaver.push_user_token(10, 2000);
        interleaver.push_agent_token(100, 1500);

        let merged = interleaver.get_interleaved();
        assert_eq!(merged.len(), 4);
        assert_eq!(merged[0].timestamp_ms, 500);
        assert_eq!(merged[0].source, StreamSource::Agent);
        assert_eq!(merged[1].timestamp_ms, 1000);
        assert_eq!(merged[1].source, StreamSource::User);
        assert_eq!(merged[2].timestamp_ms, 1500);
        assert_eq!(merged[2].source, StreamSource::Agent);
        assert_eq!(merged[3].timestamp_ms, 2000);
        assert_eq!(merged[3].source, StreamSource::User);
    }

    #[test]
    fn test_interleaver_overlapping() {
        let mut interleaver = DualStreamInterleaver::new();
        interleaver.push_user_token(5, 1000);
        interleaver.push_agent_token(42, 1000);
        interleaver.push_user_token(10, 1000);

        let merged = interleaver.get_interleaved();
        assert_eq!(merged.len(), 3);
        // All have same timestamp, but should maintain insertion order
        assert!(merged.iter().all(|t| t.timestamp_ms == 1000));
    }

    #[test]
    fn test_interleaver_context_window() {
        let mut interleaver = DualStreamInterleaver::new();
        for i in 0..10 {
            interleaver.push_user_token(i as u32, i as u64 * 100);
        }

        let window = interleaver.get_context_window(3);
        assert_eq!(window.len(), 3);
        assert_eq!(window[0].token_id, 7);
        assert_eq!(window[1].token_id, 8);
        assert_eq!(window[2].token_id, 9);

        let window = interleaver.get_context_window(100);
        assert_eq!(window.len(), 10); // Cap at available tokens
    }

    #[test]
    fn test_interleaver_gc() {
        let mut interleaver = DualStreamInterleaver::new();
        interleaver.push_user_token(5, 1000);
        interleaver.push_agent_token(42, 2000);
        interleaver.push_user_token(10, 3000);

        interleaver.clear_before(2000);
        assert_eq!(interleaver.user_token_count(), 1); // only token at 3000
        assert_eq!(interleaver.agent_token_count(), 1); // only token at 2000

        let merged = interleaver.get_interleaved();
        assert_eq!(merged.len(), 2);
        assert!(merged.iter().all(|t| t.timestamp_ms >= 2000));
    }

    #[test]
    fn test_interleaver_timestamps() {
        let mut interleaver = DualStreamInterleaver::new();
        assert_eq!(interleaver.earliest_timestamp(), None);
        assert_eq!(interleaver.latest_timestamp(), None);

        interleaver.push_user_token(5, 1000);
        assert_eq!(interleaver.earliest_timestamp(), Some(1000));
        assert_eq!(interleaver.latest_timestamp(), Some(1000));

        interleaver.push_agent_token(42, 500);
        assert_eq!(interleaver.earliest_timestamp(), Some(500));
        assert_eq!(interleaver.latest_timestamp(), Some(1000));

        interleaver.push_user_token(10, 3000);
        assert_eq!(interleaver.earliest_timestamp(), Some(500));
        assert_eq!(interleaver.latest_timestamp(), Some(3000));
    }

    #[test]
    fn test_interleaver_reset() {
        let mut interleaver = DualStreamInterleaver::new();
        interleaver.push_user_token(5, 1000);
        interleaver.push_agent_token(42, 500);
        assert_eq!(interleaver.total_token_count(), 2);

        interleaver.reset();
        assert_eq!(interleaver.total_token_count(), 0);
        assert_eq!(interleaver.user_token_count(), 0);
        assert_eq!(interleaver.agent_token_count(), 0);
    }

    #[test]
    fn test_interleaver_default() {
        let interleaver = DualStreamInterleaver::default();
        assert_eq!(interleaver.total_token_count(), 0);
    }
}
