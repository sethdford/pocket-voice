//! Backchannel generator for full-duplex conversation.
//!
//! Lightweight backchannel generator that produces engagement signals (`[hmm]`, `[right]`, `[oh]`)
//! while waiting for LLM response. These backchannels acknowledge the user's speech and signal
//! active listening without blocking the conversation flow.
//!
//! # Architecture
//!
//! Maintains timing state to ensure backchannels are:
//! - Spaced out (minimum interval between consecutive backchannels)
//! - Only during user speech or immediate silence (not while agent speaking)
//! - Varied and natural (cycling through different tokens)
//!
//! # Example
//! ```ignore
//! let mut gen = BackchannelGenerator::new();
//! if gen.should_backchannel(1000, false) {
//!     if let Some(tag) = gen.generate(1000, 500) {
//!         // Feed tag to TTS for audio generation
//!     }
//! }
//! ```

use sonata_common::NonverbalTag;

/// Backchannel generator for full-duplex conversation engagement.
///
/// Produces engagement signals that acknowledge user speech and show active listening
/// while waiting for LLM response.
#[derive(Debug, Clone)]
pub struct BackchannelGenerator {
    last_backchannel_ms: u64,
    min_interval_ms: u64,
    next_tag_index: usize,
    has_generated: bool,
}

impl BackchannelGenerator {
    /// Backchannel tags used for engagement, in order of frequency.
    const BACKCHANNEL_TAGS: &'static [NonverbalTag] = &[
        NonverbalTag::Hmm,      // Most common acknowledgment
        NonverbalTag::Right,    // Agreement
        NonverbalTag::Oh,       // Realization/attention
        NonverbalTag::UhHuh,    // Affirmation
        NonverbalTag::Wow,      // Surprise
        NonverbalTag::Ooh,      // Interest
        NonverbalTag::Breath,   // Natural breathing
    ];

    /// Create a new backchannel generator with default settings.
    ///
    /// Default minimum interval is 2000ms between backchannels.
    pub fn new() -> Self {
        Self {
            last_backchannel_ms: 0,
            min_interval_ms: 2000,
            next_tag_index: 0,
            has_generated: false,
        }
    }

    /// Create a new backchannel generator with custom minimum interval.
    pub fn with_interval(min_interval_ms: u64) -> Self {
        Self {
            last_backchannel_ms: 0,
            min_interval_ms,
            next_tag_index: 0,
            has_generated: false,
        }
    }

    /// Check if a backchannel should be generated at the given time.
    ///
    /// Returns `true` if:
    /// - Enough time has passed since the last backchannel (min_interval)
    /// - The user is not currently speaking
    ///
    /// # Arguments
    /// * `current_ms` - Current timestamp in milliseconds
    /// * `user_speaking` - Whether the user is actively speaking
    pub fn should_backchannel(&self, current_ms: u64, user_speaking: bool) -> bool {
        if user_speaking {
            return false; // Don't backchannel while user speaks
        }
        if !self.has_generated {
            return true; // First backchannel is always allowed
        }
        current_ms.saturating_sub(self.last_backchannel_ms) >= self.min_interval_ms
    }

    /// Generate a backchannel token at the given time.
    ///
    /// Returns `Some(NonverbalTag)` if:
    /// - Enough time has passed since last backchannel
    /// - Silence duration is reasonable (not during active speech)
    ///
    /// Otherwise returns `None`.
    ///
    /// # Arguments
    /// * `current_ms` - Current timestamp in milliseconds
    /// * `silence_ms` - Duration of current silence in milliseconds
    pub fn generate(&mut self, current_ms: u64, silence_ms: u64) -> Option<NonverbalTag> {
        // Require at least 500ms of silence before backchanneling
        if silence_ms < 500 {
            return None;
        }

        // Check timing constraint (enough time since last backchannel)
        if self.has_generated {
            let elapsed = current_ms.saturating_sub(self.last_backchannel_ms);
            if elapsed < self.min_interval_ms {
                return None;
            }
        }

        // Cycle through backchannel tags
        let tag = Self::BACKCHANNEL_TAGS[self.next_tag_index];
        self.next_tag_index =
            (self.next_tag_index + 1) % Self::BACKCHANNEL_TAGS.len();

        self.last_backchannel_ms = current_ms;
        self.has_generated = true;
        Some(tag)
    }

    /// Reset the backchannel generator state.
    ///
    /// Clears timing state, allowing immediate backchannel generation.
    pub fn reset(&mut self) {
        self.last_backchannel_ms = 0;
        self.next_tag_index = 0;
        self.has_generated = false;
    }

    /// Get the time until the next backchannel can be generated (in ms).
    ///
    /// Returns 0 if a backchannel can be generated immediately.
    pub fn time_until_next(&self, current_ms: u64) -> u64 {
        if !self.has_generated {
            return 0;
        }
        let elapsed = current_ms.saturating_sub(self.last_backchannel_ms);
        self.min_interval_ms.saturating_sub(elapsed)
    }

    /// Set the minimum interval between backchannels.
    pub fn set_min_interval(&mut self, min_interval_ms: u64) {
        self.min_interval_ms = min_interval_ms;
    }

    /// Get the current minimum interval.
    pub fn min_interval(&self) -> u64 {
        self.min_interval_ms
    }

    /// Get the timestamp of the last backchannel, or 0 if none yet.
    pub fn last_backchannel(&self) -> u64 {
        self.last_backchannel_ms
    }
}

impl Default for BackchannelGenerator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_backchannel_creation() {
        let gen = BackchannelGenerator::new();
        assert_eq!(gen.last_backchannel(), 0);
        assert_eq!(gen.min_interval(), 2000);
        assert_eq!(gen.next_tag_index, 0);
        assert!(!gen.has_generated);
    }

    #[test]
    fn test_backchannel_with_interval() {
        let gen = BackchannelGenerator::with_interval(5000);
        assert_eq!(gen.min_interval(), 5000);
    }

    #[test]
    fn test_backchannel_timing() {
        let mut gen = BackchannelGenerator::new();

        // First backchannel succeeds (enough silence, first call)
        let tag1 = gen.generate(1000, 1000);
        assert!(tag1.is_some());
        assert_eq!(tag1.unwrap(), NonverbalTag::Hmm);
        assert_eq!(gen.last_backchannel(), 1000);

        // Too soon for next backchannel
        let tag2 = gen.generate(2000, 1000);
        assert!(tag2.is_none());

        // After min_interval, next backchannel succeeds
        let tag3 = gen.generate(3500, 1000);
        assert!(tag3.is_some());
        assert_eq!(tag3.unwrap(), NonverbalTag::Right); // Cycles to next
        assert_eq!(gen.last_backchannel(), 3500);
    }

    #[test]
    fn test_backchannel_during_speech() {
        let mut gen = BackchannelGenerator::new();

        // No backchannel while user speaks
        let tag = gen.generate(1000, 0); // 0 silence = user still speaking
        assert!(tag.is_none());

        // Also should_backchannel returns false
        assert!(!gen.should_backchannel(1000, true));
    }

    #[test]
    fn test_backchannel_after_silence() {
        let mut gen = BackchannelGenerator::new();

        // Not enough silence (< 500ms)
        let tag1 = gen.generate(1000, 300);
        assert!(tag1.is_none());

        // Enough silence (>= 500ms)
        let tag2 = gen.generate(1000, 500);
        assert!(tag2.is_some());

        // Long silence also works
        let tag3 = gen.generate(5000, 3000);
        assert!(tag3.is_some());
    }

    #[test]
    fn test_backchannel_reset() {
        let mut gen = BackchannelGenerator::new();
        let tag1 = gen.generate(1000, 1000);
        assert!(tag1.is_some());
        assert_eq!(gen.last_backchannel(), 1000);

        gen.reset();
        assert_eq!(gen.last_backchannel(), 0);
        assert!(!gen.has_generated);

        // Can generate immediately after reset
        let tag = gen.generate(1000, 1000);
        assert!(tag.is_some());
    }

    #[test]
    fn test_backchannel_tag_validity() {
        let mut gen = BackchannelGenerator::new();

        // Generate multiple backchannels and check they're all valid NonverbalTag values
        let mut generated_tags = Vec::new();
        for i in 0..20 {
            if let Some(tag) = gen.generate(1000 + (i * 3000) as u64, 1000) {
                generated_tags.push(tag);
            }
        }

        // Should have generated some tags
        assert!(!generated_tags.is_empty());

        // All tags should be from the expected set
        for tag in generated_tags {
            assert!(BackchannelGenerator::BACKCHANNEL_TAGS.contains(&tag));
        }
    }

    #[test]
    fn test_backchannel_cycling() {
        let mut gen = BackchannelGenerator::new();

        // Generate backchannels and verify they cycle through the list
        let tags: Vec<_> = (0..BackchannelGenerator::BACKCHANNEL_TAGS.len() * 2)
            .filter_map(|i| {
                gen.generate(1000 + (i as u64 * 3000), 1000)
            })
            .collect();

        // Should cycle through at least one full sequence
        assert!(tags.len() >= BackchannelGenerator::BACKCHANNEL_TAGS.len());

        // Check that tags repeat in order
        for (i, tag) in tags.iter().enumerate() {
            let expected_idx = i % BackchannelGenerator::BACKCHANNEL_TAGS.len();
            assert_eq!(*tag, BackchannelGenerator::BACKCHANNEL_TAGS[expected_idx]);
        }
    }

    #[test]
    fn test_backchannel_time_until_next() {
        let mut gen = BackchannelGenerator::new();

        // Initially, can generate immediately (never generated before)
        assert_eq!(gen.time_until_next(0), 0);
        assert_eq!(gen.time_until_next(1000), 0);

        gen.generate(1000, 1000).unwrap();
        // After generation at 1000, need to wait until 3000
        assert_eq!(gen.time_until_next(1000), 2000);
        assert_eq!(gen.time_until_next(2000), 1000);
        assert_eq!(gen.time_until_next(3000), 0);
    }

    #[test]
    fn test_backchannel_default() {
        let gen = BackchannelGenerator::default();
        assert_eq!(gen.min_interval(), 2000);
        assert_eq!(gen.last_backchannel(), 0);
    }

    #[test]
    fn test_backchannel_first_call_at_time_zero() {
        let mut gen = BackchannelGenerator::new();

        // First call at t=0 should succeed (never generated before)
        let tag1 = gen.generate(0, 1000);
        assert!(tag1.is_some());
        assert_eq!(gen.last_backchannel(), 0);
        assert!(gen.has_generated);

        // Second call at t=0 should fail (interval not met: 0 - 0 = 0 < 2000)
        let tag2 = gen.generate(0, 1000);
        assert!(tag2.is_none());

        // Call within interval should fail
        let tag3 = gen.generate(500, 1000);
        assert!(tag3.is_none());

        // After interval should succeed
        let tag4 = gen.generate(2000, 1000);
        assert!(tag4.is_some());
    }

    #[test]
    fn test_backchannel_should_backchannel_before_first_generation() {
        let gen = BackchannelGenerator::new();

        // Before any generation, should_backchannel returns true (not speaking)
        assert!(gen.should_backchannel(0, false));
        assert!(gen.should_backchannel(100, false));

        // Still false while user speaking
        assert!(!gen.should_backchannel(0, true));
    }
}
