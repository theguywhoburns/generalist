//! Deterministic seeded RNG (SplitMix64). No external deps, stable across
//! backends and platforms: same seed -> same stream, everywhere.

#[derive(Clone, Debug)]
pub struct HarnessRng(u64);

impl HarnessRng {
    pub fn new(seed: u64) -> Self {
        Self(seed)
    }

    pub fn next_u64(&mut self) -> u64 {
        // SplitMix64.
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform usize in `[0, n)`. Panics on `n == 0`.
    pub fn below(&mut self, n: usize) -> usize {
        assert!(n > 0);
        (self.next_u64() % n as u64) as usize
    }

    /// Uniform usize in `[lo, hi)`.
    pub fn range(&mut self, lo: usize, hi: usize) -> usize {
        assert!(lo < hi);
        lo + self.below(hi - lo)
    }

    pub fn pick<'a, T>(&mut self, xs: &'a [T]) -> &'a T {
        &xs[self.below(xs.len())]
    }

    pub fn shuffle<T>(&mut self, xs: &mut [T]) {
        for i in (1..xs.len()).rev() {
            let j = self.below(i + 1);
            xs.swap(i, j);
        }
    }

    pub fn prob(&mut self, p: f64) -> bool {
        assert!((0.0..=1.0).contains(&p));
        (self.next_u64() as f64 / u64::MAX as f64) < p
    }

    /// Random ASCII string over `alphabet`, length in `[lo, hi)`.
    pub fn token_string(&mut self, alphabet: &[u8], lo: usize, hi: usize) -> String {
        let n = self.range(lo, hi);
        (0..n).map(|_| *self.pick(alphabet) as char).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deterministic_stream() {
        let mut a = HarnessRng::new(42);
        let mut b = HarnessRng::new(42);
        for _ in 0..100 {
            assert_eq!(a.next_u64(), b.next_u64());
        }
        let mut c = HarnessRng::new(43);
        assert_ne!(a.next_u64(), c.next_u64());
    }

    #[test]
    fn ranges_and_shuffle() {
        let mut rng = HarnessRng::new(7);
        for _ in 0..1000 {
            let v = rng.range(4, 12);
            assert!((4..12).contains(&v));
        }
        let mut xs = vec![1, 2, 3, 4, 5];
        rng.shuffle(&mut xs);
        xs.sort();
        assert_eq!(xs, vec![1, 2, 3, 4, 5]);
    }
}
