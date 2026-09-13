//! Structure-preserving integrators.
//!
//! * [`Yoshida6`] — 6th-order Yoshida composition of Störmer–Verlet flows for
//!   separable Hamiltonians `H(q, p) = T(p) + V(q)` on ℝⁿ. Each sub-flow is an
//!   exact shear of phase space, so the composed map is an exact canonical
//!   transformation: the symplectic 2-form `dq ∧ dp` is preserved to rounding
//!   and the energy error stays bounded (no secular drift) for all time.
//! * [`RigidBodyYoshida6`] — the same composition applied to the free rigid
//!   body on `SO(3) × so(3)*` using exact axis-rotation sub-flows of the
//!   Lie–Poisson splitting `H = Σ πᵢ²/(2Iᵢ)`. The angular momentum magnitude
//!   is conserved exactly and orientation stays on the group manifold.

use crate::lie::{Rotation, Vec3};

/// Yoshida (1990) solution A weights `w₁, w₂, w₃`; `w₀ = 1 - 2(w₁+w₂+w₃)`.
pub const YOSHIDA6_W1: f64 = -1.177_679_984_178_871;
/// See [`YOSHIDA6_W1`].
pub const YOSHIDA6_W2: f64 = 0.235_573_213_359_358_13;
/// See [`YOSHIDA6_W1`].
pub const YOSHIDA6_W3: f64 = 0.784_513_610_477_557_3;
/// See [`YOSHIDA6_W1`].
pub const YOSHIDA6_W0: f64 = 1.0 - 2.0 * (YOSHIDA6_W1 + YOSHIDA6_W2 + YOSHIDA6_W3);

/// Drift (position-update) weights of the 7-stage composition
/// `S₂(w₃h) S₂(w₂h) S₂(w₁h) S₂(w₀h) S₂(w₁h) S₂(w₂h) S₂(w₃h)`.
pub const YOSHIDA6_DRIFT: [f64; 7] = [
    YOSHIDA6_W3,
    YOSHIDA6_W2,
    YOSHIDA6_W1,
    YOSHIDA6_W0,
    YOSHIDA6_W1,
    YOSHIDA6_W2,
    YOSHIDA6_W3,
];

/// Kick (momentum-update) weights with adjacent half-kicks fused.
pub const YOSHIDA6_KICK: [f64; 8] = [
    YOSHIDA6_W3 * 0.5,
    (YOSHIDA6_W3 + YOSHIDA6_W2) * 0.5,
    (YOSHIDA6_W2 + YOSHIDA6_W1) * 0.5,
    (YOSHIDA6_W1 + YOSHIDA6_W0) * 0.5,
    (YOSHIDA6_W0 + YOSHIDA6_W1) * 0.5,
    (YOSHIDA6_W1 + YOSHIDA6_W2) * 0.5,
    (YOSHIDA6_W2 + YOSHIDA6_W3) * 0.5,
    YOSHIDA6_W3 * 0.5,
];

/// Separable Hamiltonian `H(q, p) = T(p) + V(q)` with `N` degrees of freedom.
pub trait SeparableHamiltonian<const N: usize> {
    /// `∂T/∂p` (generalised velocity).
    fn velocity(&self, p: &[f64; N]) -> [f64; N];
    /// `∂V/∂q` (negative generalised force).
    fn potential_gradient(&self, q: &[f64; N]) -> [f64; N];
    /// Total energy.
    fn energy(&self, q: &[f64; N], p: &[f64; N]) -> f64;
}

/// Phase-space state with compensated accumulators.
///
/// `q` and `p` hold the leading `f64` parts; `q_lo` / `p_lo` carry the
/// rounding residual of every increment (Neumaier two-sum), so the effective
/// accumulation width is ~106 bits. Without this the `O(ε‖q‖)` rounding of
/// each of the 8 sub-steps random-walks to `~10⁻¹¹` relative energy error over
/// `10⁷` steps, swamping the `O(Δt⁶)` truncation error of the scheme.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PhaseState<const N: usize> {
    /// Generalised coordinates.
    pub q: [f64; N],
    /// Conjugate momenta.
    pub p: [f64; N],
    /// Rounding residuals of `q`.
    pub q_lo: [f64; N],
    /// Rounding residuals of `p`.
    pub p_lo: [f64; N],
}

impl<const N: usize> PhaseState<N> {
    /// State with zero residuals.
    pub const fn new(q: [f64; N], p: [f64; N]) -> Self {
        Self {
            q,
            p,
            q_lo: [0.0; N],
            p_lo: [0.0; N],
        }
    }

    /// Folds the residuals into the leading parts (`q + q_lo`, `p + p_lo`).
    pub fn rounded(&self) -> ([f64; N], [f64; N]) {
        let mut q = self.q;
        let mut p = self.p;
        for i in 0..N {
            q[i] += self.q_lo[i];
            p[i] += self.p_lo[i];
        }
        (q, p)
    }
}

/// `x += inc` with the rounding error of the addition captured in `lo`.
#[inline]
fn compensated_add(x: &mut f64, lo: &mut f64, inc: f64) {
    let y = inc + *lo;
    let t = *x + y;
    *lo = if x.abs() >= y.abs() {
        (*x - t) + y
    } else {
        (y - t) + *x
    };
    *x = t;
}

#[inline]
fn kick<const N: usize, H: SeparableHamiltonian<N>>(sys: &H, s: &mut PhaseState<N>, h: f64) {
    let g = sys.potential_gradient(&s.q);
    for ((p, lo), g) in s.p.iter_mut().zip(s.p_lo.iter_mut()).zip(g) {
        compensated_add(p, lo, -h * g);
    }
}

#[inline]
fn drift<const N: usize, H: SeparableHamiltonian<N>>(sys: &H, s: &mut PhaseState<N>, h: f64) {
    let v = sys.velocity(&s.p);
    for ((q, lo), v) in s.q.iter_mut().zip(s.q_lo.iter_mut()).zip(v) {
        compensated_add(q, lo, h * v);
    }
}

/// Second-order Störmer–Verlet (leapfrog) step, kick–drift–kick form.
#[derive(Clone, Copy, Debug, Default)]
pub struct StormerVerlet;

impl StormerVerlet {
    /// Advances `state` by `dt`.
    pub fn step<const N: usize, H: SeparableHamiltonian<N>>(
        sys: &H,
        state: &mut PhaseState<N>,
        dt: f64,
    ) {
        kick(sys, state, 0.5 * dt);
        drift(sys, state, dt);
        kick(sys, state, 0.5 * dt);
    }
}

/// Sixth-order Yoshida symplectic integrator `Ψ⁽⁶⁾`.
#[derive(Clone, Copy, Debug, Default)]
pub struct Yoshida6;

impl Yoshida6 {
    /// Formal order of accuracy.
    pub const ORDER: u32 = 6;

    /// Advances `state` by `dt` with one 7-stage composition step.
    pub fn step<const N: usize, H: SeparableHamiltonian<N>>(
        sys: &H,
        state: &mut PhaseState<N>,
        dt: f64,
    ) {
        kick(sys, state, YOSHIDA6_KICK[0] * dt);
        for stage in 0..7 {
            drift(sys, state, YOSHIDA6_DRIFT[stage] * dt);
            kick(sys, state, YOSHIDA6_KICK[stage + 1] * dt);
        }
    }

    /// Advances `state` by `steps` steps of size `dt`.
    pub fn integrate<const N: usize, H: SeparableHamiltonian<N>>(
        sys: &H,
        state: &mut PhaseState<N>,
        dt: f64,
        steps: u64,
    ) {
        for _ in 0..steps {
            Self::step(sys, state, dt);
        }
    }
}

/// Free rigid body with diagonal inertia tensor in the body frame.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RigidBody {
    /// Principal moments of inertia.
    pub inertia: Vec3,
}

/// Rigid-body state: orientation and body-frame angular momentum.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RigidBodyState {
    /// Orientation `R ∈ SO(3)`.
    pub rotation: Rotation,
    /// Body-frame angular momentum `π ∈ so(3)*`.
    pub momentum: Vec3,
}

impl RigidBody {
    /// Kinetic energy `Σ πᵢ² / (2Iᵢ)`.
    pub fn energy(&self, s: &RigidBodyState) -> f64 {
        let pi = s.momentum;
        0.5 * (pi[0] * pi[0] / self.inertia[0]
            + pi[1] * pi[1] / self.inertia[1]
            + pi[2] * pi[2] / self.inertia[2])
    }

    /// Exact flow of the partial Hamiltonian `Hᵢ = πᵢ²/(2Iᵢ)` for time `h`:
    /// `π` rotates about `eᵢ` by `-ωᵢh` and `R` about the body axis by `ωᵢh`,
    /// with `ωᵢ = πᵢ/Iᵢ`. Both updates are exact group operations.
    fn axis_flow(&self, s: &mut RigidBodyState, axis: usize, h: f64) {
        let omega = s.momentum[axis] / self.inertia[axis];
        let angle = omega * h;
        let (sn, cs) = angle.sin_cos();
        let (a, b) = ((axis + 1) % 3, (axis + 2) % 3);
        let (pa, pb) = (s.momentum[a], s.momentum[b]);
        // π' = exp(-angle · hat(e_axis)) π
        s.momentum[a] = cs * pa + sn * pb;
        s.momentum[b] = -sn * pa + cs * pb;
        let mut w = [0.0; 3];
        w[axis] = angle;
        s.rotation = s.rotation.compose(&Rotation::exp(w));
    }

    /// Symmetric second-order splitting `e^{hA/2} e^{hB/2} e^{hC} e^{hB/2} e^{hA/2}`.
    fn strang_step(&self, s: &mut RigidBodyState, h: f64) {
        self.axis_flow(s, 0, 0.5 * h);
        self.axis_flow(s, 1, 0.5 * h);
        self.axis_flow(s, 2, h);
        self.axis_flow(s, 1, 0.5 * h);
        self.axis_flow(s, 0, 0.5 * h);
    }
}

/// Sixth-order Yoshida composition of the Lie–Poisson rigid-body splitting.
#[derive(Clone, Copy, Debug, Default)]
pub struct RigidBodyYoshida6;

impl RigidBodyYoshida6 {
    /// Advances the rigid-body state by `dt`.
    pub fn step(body: &RigidBody, state: &mut RigidBodyState, dt: f64) {
        for w in YOSHIDA6_DRIFT {
            body.strang_step(state, w * dt);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lie::norm;

    struct Harmonic;

    impl SeparableHamiltonian<1> for Harmonic {
        fn velocity(&self, p: &[f64; 1]) -> [f64; 1] {
            *p
        }
        fn potential_gradient(&self, q: &[f64; 1]) -> [f64; 1] {
            *q
        }
        fn energy(&self, q: &[f64; 1], p: &[f64; 1]) -> f64 {
            0.5 * (q[0] * q[0] + p[0] * p[0])
        }
    }

    #[test]
    fn weights_sum_to_one() {
        let s: f64 = YOSHIDA6_DRIFT.iter().sum();
        assert!((s - 1.0).abs() < 1e-15);
        let k: f64 = YOSHIDA6_KICK.iter().sum();
        assert!((k - 1.0).abs() < 1e-15);
    }

    #[test]
    fn sixth_order_convergence() {
        // Global error at t = 2π scales as dt^6: halving dt shrinks it ~64x.
        let err = |steps: u64| {
            let dt = core::f64::consts::TAU / steps as f64;
            let mut s = PhaseState::new([1.0], [0.0]);
            Yoshida6::integrate(&Harmonic, &mut s, dt, steps);
            ((s.q[0] - 1.0).powi(2) + s.p[0].powi(2)).sqrt()
        };
        let e1 = err(32);
        let e2 = err(64);
        let ratio = e1 / e2;
        assert!(ratio > 40.0 && ratio < 100.0, "ratio {ratio}");
    }

    #[test]
    fn harmonic_energy_bounded() {
        let mut s = PhaseState::new([1.0], [0.0]);
        let e0 = Harmonic.energy(&s.q, &s.p);
        let mut worst = 0.0_f64;
        for _ in 0..100_000 {
            Yoshida6::step(&Harmonic, &mut s, 0.05);
            worst = worst.max(((Harmonic.energy(&s.q, &s.p) - e0) / e0).abs());
        }
        assert!(worst < 1e-10, "{worst}");
    }

    /// Undamped, periodically forced double-well Duffing oscillator
    /// `ẍ = x − x³ + γ cos(ωt)` lifted to the autonomous extended phase space
    /// `(x, τ; p, p_τ)` with `H = p²/2 + p_τ − x²/2 + x⁴/4 − γ x cos(ωτ)`.
    /// The extended Hamiltonian is separable and exactly conserved along the
    /// (chaotic, for these parameters) flow, so it is a strict drift gauge.
    struct Duffing {
        gamma: f64,
        omega: f64,
    }

    impl Duffing {
        /// Forcing period; chosen exactly representable so the `τ ↦ τ - P`
        /// symmetry reduction below is an exact operation.
        const PERIOD: f64 = 5.0;

        fn new(gamma: f64) -> Self {
            Self {
                gamma,
                omega: core::f64::consts::TAU / Self::PERIOD,
            }
        }
    }

    impl SeparableHamiltonian<2> for Duffing {
        fn velocity(&self, p: &[f64; 2]) -> [f64; 2] {
            [p[0], 1.0]
        }
        fn potential_gradient(&self, q: &[f64; 2]) -> [f64; 2] {
            let (x, t) = (q[0], q[1]);
            let (s, c) = (self.omega * t).sin_cos();
            [
                -x + x * x * x - self.gamma * c,
                self.gamma * self.omega * x * s,
            ]
        }
        fn energy(&self, q: &[f64; 2], p: &[f64; 2]) -> f64 {
            let (x, t) = (q[0], q[1]);
            0.5 * p[0] * p[0] + p[1] - 0.5 * x * x + 0.25 * x * x * x * x
                - self.gamma * x * (self.omega * t).cos()
        }
    }

    #[test]
    fn duffing_chaotic_energy_drift_below_1e12_over_1e7_steps() {
        let sys = Duffing::new(0.3);
        let mut s = PhaseState::new([0.0, 0.0], [1.0, 0.0]);
        // Chaos witness: a shadow trajectory perturbed by one ulp must separate to O(1).
        let mut shadow = s;
        shadow.q[0] = f64::from_bits(s.q[0].to_bits() + 1) + 1e-15;
        let e0 = sys.energy(&s.q, &s.p);
        let dt = 2.5e-3;
        let steps = 10_000_000_u64;
        let mut worst = 0.0_f64;
        for i in 0..steps {
            Yoshida6::step(&sys, &mut s, dt);
            Yoshida6::step(&sys, &mut shadow, dt);
            // H is P-periodic in τ; keeping τ ∈ [0, P) stops ulp(τ) growing with
            // elapsed time, which would otherwise inject non-symplectic
            // phase noise into cos(ωτ).
            for st in [&mut s, &mut shadow] {
                if st.q[1] >= Duffing::PERIOD {
                    st.q[1] -= Duffing::PERIOD;
                }
            }
            if i % 1_000 == 0 {
                worst = worst.max(((sys.energy(&s.q, &s.p) - e0) / e0).abs());
            }
        }
        let final_drift = ((sys.energy(&s.q, &s.p) - e0) / e0).abs();
        worst = worst.max(final_drift);
        assert!(worst < 1e-12, "worst relative energy drift {worst:e}");
        let separation = (s.q[0] - shadow.q[0]).abs();
        assert!(
            separation > 0.1,
            "trajectory not chaotic: separation {separation:e}"
        );
        assert!(s.q[0].abs() < 3.0 && s.q[0].is_finite());
    }

    #[test]
    fn rigid_body_conserves_momentum_norm_and_energy() {
        let body = RigidBody {
            inertia: [1.0, 2.0, 3.0],
        };
        let mut s = RigidBodyState {
            rotation: Rotation::IDENTITY,
            momentum: [0.1, 1.0, 0.1], // near the unstable intermediate axis
        };
        let e0 = body.energy(&s);
        let n0 = norm(s.momentum);
        let mut worst_e = 0.0_f64;
        for _ in 0..1_000_000 {
            RigidBodyYoshida6::step(&body, &mut s, 0.01);
            worst_e = worst_e.max(((body.energy(&s) - e0) / e0).abs());
        }
        assert!((norm(s.momentum) - n0).abs() < 1e-12);
        assert!(worst_e < 1e-11, "{worst_e}");
        assert!(s.rotation.orthonormality_defect() < 1e-9);
    }
}
