//! Reduced-order surrogate evaluation and real-time normalised-residual
//! monitoring `χ²(t) = r(t)ᵀ Σ⁻¹ r(t)` (simulator_spec.pdf §4). The monitor is
//! plain state owned by the single consumer task — no locks anywhere.

use crate::telemetry::Frame;

/// A reduced-order surrogate mapping a raw frame to predicted lane values.
///
/// `predict` writes the expected measurement into `out` (same lane ordering
/// as the frame payload). Implementations should be allocation-free — they
/// run on every 10 kHz sample.
pub trait ReducedOrderModel {
    /// Predicted lane values for `frame` written into `out`; returns the lane
    /// count used (`0` → frame skipped by residual monitoring).
    fn predict(&self, frame: &Frame, out: &mut [f64]) -> usize;
}

impl<F: Fn(&Frame, &mut [f64]) -> usize> ReducedOrderModel for F {
    fn predict(&self, frame: &Frame, out: &mut [f64]) -> usize {
        self(frame, out)
    }
}

/// χ² statistic of a residual against a diagonal inverse covariance
/// `Σ⁻¹ = diag(1/σᵢ²)`: `χ² = Σᵢ rᵢ²/σᵢ²`.
pub fn chi2(residual: &[f64], inv_var: &[f64]) -> f64 {
    residual.iter().zip(inv_var).map(|(r, iv)| r * r * iv).sum()
}

/// Normalised residual: `χ²` reduced by degrees of freedom; ≈1 when the
/// stream agrees with the surrogate within stated uncertainty.
pub fn reduced_chi2(residual: &[f64], inv_var: &[f64]) -> f64 {
    if residual.is_empty() {
        0.0
    } else {
        chi2(residual, inv_var) / residual.len() as f64
    }
}

/// Running residual monitor over one channel.
#[derive(Clone, Debug, Default)]
pub struct ChannelResiduals {
    /// Frames scored.
    pub count: u64,
    /// Mean of the per-frame `χ²/ν` statistic.
    pub mean_reduced_chi2: f64,
    /// Maximum observed `χ²/ν`.
    pub max_reduced_chi2: f64,
    /// Count of frames above the alarm threshold.
    pub alarms: u64,
}

/// Per-channel residual monitor with diagonal measurement covariance.
#[derive(Debug)]
pub struct ResidualMonitor {
    /// `σᵢ⁻²` per channel (lane-uniform): index = channel tag.
    inv_var: Vec<f64>,
    /// Alarm when `χ²/ν > threshold`.
    threshold: f64,
    /// Per-channel stats (index = channel tag).
    stats: Vec<ChannelResiduals>,
    /// Scratch prediction buffer (no per-frame allocation).
    predict: Vec<f64>,
}

impl ResidualMonitor {
    /// `inv_var[ch]` is `1/σ²` for that channel's lanes.
    pub fn new(inv_var: Vec<f64>, threshold: f64) -> Self {
        let n = inv_var.len();
        Self {
            inv_var,
            threshold,
            stats: vec![ChannelResiduals::default(); n],
            predict: vec![0.0; crate::telemetry::FRAME_VALUES],
        }
    }

    /// Scores a frame against `rom`; returns the frame's `χ²/ν` if scored.
    pub fn score<M: ReducedOrderModel>(&mut self, frame: &Frame, rom: &M) -> Option<f64> {
        let lanes = rom
            .predict(frame, &mut self.predict)
            .min(frame.lanes as usize);
        if lanes == 0 {
            return None;
        }
        let ch = (frame.channel as usize).min(self.inv_var.len() - 1);
        let iv = self.inv_var[ch];
        let mut c2 = 0.0;
        for i in 0..lanes {
            let r = frame.values[i] - self.predict[i];
            c2 += r * r * iv;
        }
        let nu = c2 / lanes as f64;
        let s = &mut self.stats[ch];
        s.count += 1;
        s.mean_reduced_chi2 += (nu - s.mean_reduced_chi2) / s.count as f64;
        s.max_reduced_chi2 = s.max_reduced_chi2.max(nu);
        if nu > self.threshold {
            s.alarms += 1;
        }
        Some(nu)
    }

    /// Per-channel running statistics.
    pub fn stats(&self) -> &[ChannelResiduals] {
        &self.stats
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::telemetry::Channel;

    #[test]
    fn chi2_zero_for_perfect_surrogate() {
        let mut mon = ResidualMonitor::new(vec![0.0; 256], 9.0);
        mon.inv_var[Channel::ThermocoupleN as usize] = 4.0;
        let f = Frame::scalar(Channel::ThermocoupleN, 0, 0, 623.0);
        let rom = |_: &Frame, out: &mut [f64]| {
            out[0] = 623.0;
            1
        };
        assert_eq!(mon.score(&f, &rom), Some(0.0));
        let bad = Frame::scalar(Channel::ThermocoupleN, 1, 1, 624.0);
        // r = 1.0, σ⁻² = 4 ⇒ χ²/ν = 4 > threshold? 9 → no alarm; use 3.0.
        assert_eq!(mon.score(&bad, &rom), Some(4.0));
        let mut mon = ResidualMonitor::new(
            {
                let mut v = vec![0.0; 256];
                v[Channel::ThermocoupleN as usize] = 4.0;
                v
            },
            3.0,
        );
        mon.score(&bad, &rom);
        assert_eq!(mon.stats()[Channel::ThermocoupleN as usize].alarms, 1);
    }
}
