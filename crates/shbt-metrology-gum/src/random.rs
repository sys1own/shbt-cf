//! Deterministic counter-based RNG (SplitMix64 seed expansion + Xoshiro256**)
//! so Monte Carlo batches are reproducible across platforms.

/// SplitMix64 used to expand a seed into a Xoshiro state.
fn splitmix64(x: &mut u64) -> u64 {
    *x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *x;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Xoshiro256** uniform generator.
#[derive(Clone, Debug)]
pub struct Rng {
    s: [u64; 4],
}

impl Rng {
    /// Seed a generator.
    pub fn new(seed: u64) -> Self {
        let mut x = seed;
        Self {
            s: [
                splitmix64(&mut x),
                splitmix64(&mut x),
                splitmix64(&mut x),
                splitmix64(&mut x),
            ],
        }
    }

    /// Uniform `u64`.
    pub fn next_u64(&mut self) -> u64 {
        let s = &mut self.s;
        let result = (s[1].wrapping_mul(5)).rotate_left(7).wrapping_mul(9);
        let t = s[1] << 17;
        s[2] ^= s[0];
        s[3] ^= s[1];
        s[1] ^= s[2];
        s[0] ^= s[3];
        s[2] ^= t;
        s[3] = s[3].rotate_left(45);
        result
    }

    /// Uniform `f64` in `[0, 1)`.
    pub fn uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }

    /// Uniform `f64` in `(0, 1)`.
    pub fn uniform_open(&mut self) -> f64 {
        loop {
            let u = self.uniform();
            if u > 0.0 {
                return u;
            }
        }
    }

    /// Standard normal via Box–Muller.
    pub fn normal(&mut self) -> f64 {
        let u1 = self.uniform_open();
        let u2 = self.uniform();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deterministic_and_well_behaved() {
        let mut a = Rng::new(42);
        let mut b = Rng::new(42);
        for _ in 0..1000 {
            assert_eq!(a.next_u64(), b.next_u64());
        }
        let mut r = Rng::new(7);
        let n = 200_000;
        let (mut m, mut v) = (0.0f64, 0.0f64);
        for _ in 0..n {
            m += r.normal();
        }
        m /= n as f64;
        let mut r = Rng::new(7);
        for _ in 0..n {
            let z = r.normal() - m;
            v += z * z;
        }
        v /= n as f64;
        assert!(m.abs() < 0.01, "{m}");
        assert!((v - 1.0).abs() < 0.01, "{v}");
    }
}
