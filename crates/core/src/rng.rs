//! Deterministic, cross-platform explicit-seed PRNG built on `ChaCha8Rng`.
//!
//! No OS entropy or `getrandom` dependency is used, ensuring bit-exact reproducibility
//! across Native Rust, Python, and WebAssembly (`wasm32-unknown-unknown`).

use rand_chacha::ChaCha8Rng;
use rand_core::{RngCore, SeedableRng};

/// Expands a 64-bit integer seed using SplitMix64 to populate a 256-bit ChaCha seed.
#[inline]
fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

/// Deterministic explicit-seed pseudo-random number generator.
///
/// Seeds are expanded from `u64` via SplitMix64 into a 256-bit key for [`ChaCha8Rng`],
/// avoiding low-Hamming-weight seed artifacts while remaining identical across 32-bit
/// and 64-bit targets.
#[derive(Debug, Clone)]
pub struct DeterministicRng {
    inner: ChaCha8Rng,
}

impl DeterministicRng {
    /// Creates a deterministic RNG initialized from an explicit `u64` seed.
    pub fn from_seed_u64(seed: u64) -> Self {
        let mut state = seed;
        let mut key = [0u8; 32];
        for chunk in key.as_chunks_mut::<8>().0 {
            *chunk = splitmix64(&mut state).to_le_bytes();
        }
        Self {
            inner: ChaCha8Rng::from_seed(key),
        }
    }

    /// Creates a deterministic RNG directly from a 256-bit seed.
    pub fn from_seed_bytes(seed: [u8; 32]) -> Self {
        Self {
            inner: ChaCha8Rng::from_seed(seed),
        }
    }

    /// Generates the next uniformly distributed `u64`.
    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        self.inner.next_u64()
    }

    /// Generates the next uniformly distributed `u32`.
    #[inline]
    pub fn next_u32(&mut self) -> u32 {
        self.inner.next_u32()
    }

    /// Generates a uniformly distributed `f64` in `[0.0, 1.0)` with 53 random significand bits.
    #[inline]
    pub fn next_f64(&mut self) -> f64 {
        const SCALE: f64 = 1.0 / ((1u64 << 53) as f64);
        ((self.next_u64() >> 11) as f64) * SCALE
    }

    /// Generates a uniformly distributed `f32` in `[0.0, 1.0)` with 24 random significand bits.
    #[inline]
    pub fn next_f32(&mut self) -> f32 {
        const SCALE: f32 = 1.0 / ((1u32 << 24) as f32);
        ((self.next_u32() >> 8) as f32) * SCALE
    }

    /// Samples a uniformly distributed `u32` in `[0, upper_exclusive)` using Lemire's
    /// unbiased nearly-divisionless multiply-high algorithm.
    ///
    /// # Panics
    ///
    /// Panics if `upper_exclusive == 0`.
    #[inline]
    pub fn gen_range_u32(&mut self, upper_exclusive: u32) -> u32 {
        assert!(upper_exclusive > 0, "upper_exclusive must be > 0");
        if upper_exclusive == 1 {
            return 0;
        }
        let s = upper_exclusive as u64;
        let mut m = (self.next_u32() as u64) * s;
        let mut l = m as u32;
        if l < upper_exclusive {
            let t = upper_exclusive.wrapping_neg() % upper_exclusive;
            while l < t {
                m = (self.next_u32() as u64) * s;
                l = m as u32;
            }
        }
        (m >> 32) as u32
    }

    /// Samples a uniformly distributed `usize` in `[0, upper_exclusive)` without modulo bias.
    ///
    /// Uses 32-bit Lemire range sampling whenever `upper_exclusive <= u32::MAX` for bit-exact
    /// parity across `wasm32` and 64-bit native targets, falling back to 64-bit Lemire for
    /// larger bounds.
    ///
    /// # Panics
    ///
    /// Panics if `upper_exclusive == 0`.
    #[inline]
    pub fn gen_range_usize(&mut self, upper_exclusive: usize) -> usize {
        assert!(upper_exclusive > 0, "upper_exclusive must be > 0");
        if let Ok(upper_u32) = u32::try_from(upper_exclusive) {
            return self.gen_range_u32(upper_u32) as usize;
        }
        let upper_u64 = upper_exclusive as u64;
        let s = upper_u64 as u128;
        let mut m = (self.next_u64() as u128) * s;
        let mut l = m as u64;
        if l < upper_u64 {
            let t = upper_u64.wrapping_neg() % upper_u64;
            while l < t {
                m = (self.next_u64() as u128) * s;
                l = m as u64;
            }
        }
        (m >> 64) as usize
    }

    /// Performs an in-place Fisher-Yates shuffle using unbiased range sampling.
    pub fn shuffle<T>(&mut self, slice: &mut [T]) {
        for i in (1..slice.len()).rev() {
            let j = self.gen_range_usize(i + 1);
            slice.swap(i, j);
        }
    }

    /// Returns a mutable reference to the underlying `ChaCha8Rng` implementing `RngCore`.
    #[inline]
    pub fn as_rng_core_mut(&mut self) -> &mut ChaCha8Rng {
        &mut self.inner
    }
}

impl RngCore for DeterministicRng {
    #[inline]
    fn next_u32(&mut self) -> u32 {
        self.inner.next_u32()
    }

    #[inline]
    fn next_u64(&mut self) -> u64 {
        self.inner.next_u64()
    }

    #[inline]
    fn fill_bytes(&mut self, dest: &mut [u8]) {
        self.inner.fill_bytes(dest);
    }

    #[inline]
    fn try_fill_bytes(&mut self, dest: &mut [u8]) -> Result<(), rand_core::Error> {
        self.inner.try_fill_bytes(dest)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_deterministic_reproducibility_and_seed_separation() {
        let mut r1 = DeterministicRng::from_seed_u64(42);
        let mut r2 = DeterministicRng::from_seed_u64(42);
        let mut r3 = DeterministicRng::from_seed_u64(43);

        let seq1: Vec<u64> = (0..16).map(|_| r1.next_u64()).collect();
        let seq2: Vec<u64> = (0..16).map(|_| r2.next_u64()).collect();
        let seq3: Vec<u64> = (0..16).map(|_| r3.next_u64()).collect();

        assert_eq!(seq1, seq2);
        assert_ne!(seq1, seq3);
    }

    #[test]
    fn test_f64_f32_and_gen_range_bounds() {
        let mut rng = DeterministicRng::from_seed_u64(0);
        for _ in 0..1000 {
            let x = rng.next_f64();
            assert!((0.0..1.0).contains(&x));
            let y = rng.next_f32();
            assert!((0.0..1.0).contains(&y));
            let idx = rng.gen_range_usize(7);
            assert!(idx < 7);
            assert_eq!(rng.gen_range_usize(1), 0);
        }
    }

    #[test]
    fn test_as_rng_core_mut_and_shuffle() {
        let mut r1 = DeterministicRng::from_seed_u64(99);
        let mut r2 = DeterministicRng::from_seed_u64(99);
        assert_eq!(r1.as_rng_core_mut().next_u64(), r2.next_u64());

        let mut a = [0, 1, 2, 3, 4, 5, 6, 7];
        let mut b = a;
        r1.shuffle(&mut a);
        r2.shuffle(&mut b);
        assert_eq!(a, b);
    }

    #[test]
    #[should_panic(expected = "upper_exclusive must be > 0")]
    fn test_gen_range_zero_panics() {
        let mut rng = DeterministicRng::from_seed_u64(1);
        let _ = rng.gen_range_usize(0);
    }

    #[test]
    #[should_panic(expected = "upper_exclusive must be > 0")]
    fn test_gen_range_u32_zero_panics() {
        let mut rng = DeterministicRng::from_seed_u64(1);
        let _ = rng.gen_range_u32(0);
    }

    #[test]
    fn test_gen_range_u32_and_usize_parity() {
        for &bound in &[1u32, 2, 7, 13, 255, 16_384, 1_000_003] {
            let mut r_u32 = DeterministicRng::from_seed_u64(12345);
            let mut r_usize = DeterministicRng::from_seed_u64(12345);
            for _ in 0..256 {
                let a = r_u32.gen_range_u32(bound);
                let b = r_usize.gen_range_usize(bound as usize);
                assert!(a < bound);
                assert_eq!(a as usize, b);
            }
        }
    }
}
