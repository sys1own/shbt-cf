//! Lie-group kinematics on SO(3) and SE(3).
//!
//! Rotations are stored as orthonormal 3x3 matrices and poses as
//! (rotation, translation) pairs. State is advanced with the group exponential
//! map, `g_{k+1} = g_k · exp(ξ Δt)`, so every step lands exactly on the group
//! manifold: there is no quaternion renormalisation drift and no Euler-angle
//! singularity. The only error source is floating-point rounding in the
//! matrix product, which stays at the level of a few ulps per step.
//!
//! Conventions: column vectors, right-multiplicative body-frame twists,
//! `hat` maps ℝ³ → so(3) with `hat(ω) v = ω × v`.

/// 3-vector.
pub type Vec3 = [f64; 3];
/// Row-major 3x3 matrix.
pub type Mat3 = [[f64; 3]; 3];

/// Below this angle the closed-form Rodrigues coefficients are replaced by
/// their Taylor expansions to avoid catastrophic cancellation.
const SMALL_ANGLE: f64 = 1e-6;

/// Identity matrix.
pub const IDENTITY: Mat3 = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

/// Cross product.
#[inline]
pub fn cross(a: Vec3, b: Vec3) -> Vec3 {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// Dot product.
#[inline]
pub fn dot(a: Vec3, b: Vec3) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// Euclidean norm.
#[inline]
pub fn norm(a: Vec3) -> f64 {
    dot(a, a).sqrt()
}

/// Skew-symmetric matrix `[ω]×` such that `hat(ω) v = ω × v`.
#[inline]
pub fn hat(w: Vec3) -> Mat3 {
    [[0.0, -w[2], w[1]], [w[2], 0.0, -w[0]], [-w[1], w[0], 0.0]]
}

/// Inverse of [`hat`] for a skew-symmetric matrix.
#[inline]
pub fn vee(m: &Mat3) -> Vec3 {
    [m[2][1], m[0][2], m[1][0]]
}

/// `vee(M - Mᵀ)`: twice the axial vector of the skew-symmetric part of `M`.
#[inline]
fn skew_part_vee(m: &Mat3) -> Vec3 {
    [m[2][1] - m[1][2], m[0][2] - m[2][0], m[1][0] - m[0][1]]
}

/// Matrix product.
pub fn mat_mul(a: &Mat3, b: &Mat3) -> Mat3 {
    let mut out = [[0.0; 3]; 3];
    for (i, row) in out.iter_mut().enumerate() {
        for (j, cell) in row.iter_mut().enumerate() {
            *cell = a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j];
        }
    }
    out
}

/// Matrix–vector product.
#[inline]
pub fn mat_vec(m: &Mat3, v: Vec3) -> Vec3 {
    [
        m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
        m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
        m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
    ]
}

/// Transpose.
#[inline]
pub fn transpose(m: &Mat3) -> Mat3 {
    [
        [m[0][0], m[1][0], m[2][0]],
        [m[0][1], m[1][1], m[2][1]],
        [m[0][2], m[1][2], m[2][2]],
    ]
}

/// Determinant.
pub fn det(m: &Mat3) -> f64 {
    m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
}

/// Rodrigues coefficients `(A, B, C)` for angle `θ`, with series fallback:
/// `A = sin θ / θ`, `B = (1 - cos θ) / θ²`, `C = (θ - sin θ) / θ³`.
fn rodrigues_coefficients(theta: f64) -> (f64, f64, f64) {
    if theta < SMALL_ANGLE {
        let t2 = theta * theta;
        (
            1.0 - t2 / 6.0 + t2 * t2 / 120.0,
            0.5 - t2 / 24.0 + t2 * t2 / 720.0,
            1.0 / 6.0 - t2 / 120.0 + t2 * t2 / 5040.0,
        )
    } else {
        let (s, c) = theta.sin_cos();
        let t2 = theta * theta;
        (s / theta, (1.0 - c) / t2, (theta - s) / (t2 * theta))
    }
}

/// Rotation in SO(3) as an orthonormal matrix with determinant +1.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Rotation(pub Mat3);

impl Default for Rotation {
    fn default() -> Self {
        Self::IDENTITY
    }
}

impl Rotation {
    /// Identity rotation.
    pub const IDENTITY: Self = Self(IDENTITY);

    /// Exponential map so(3) → SO(3) (Rodrigues' formula):
    /// `exp([ω]×) = I + A [ω]× + B [ω]×²`.
    pub fn exp(omega: Vec3) -> Self {
        let theta = norm(omega);
        let (a, b, _) = rodrigues_coefficients(theta);
        let k = hat(omega);
        let k2 = mat_mul(&k, &k);
        let mut r = IDENTITY;
        for i in 0..3 {
            for j in 0..3 {
                r[i][j] += a * k[i][j] + b * k2[i][j];
            }
        }
        Self(r)
    }

    /// Logarithm map SO(3) → so(3), returning the rotation vector `ω` with
    /// `‖ω‖ ∈ [0, π]`.
    pub fn log(&self) -> Vec3 {
        let m = &self.0;
        let trace = m[0][0] + m[1][1] + m[2][2];
        let cos_theta = ((trace - 1.0) * 0.5).clamp(-1.0, 1.0);
        let theta = cos_theta.acos();
        if theta < SMALL_ANGLE {
            // log ≈ (R - Rᵀ)/2 to first order.
            let v = skew_part_vee(m);
            let scale = 0.5 * (1.0 + theta * theta / 6.0);
            return [v[0] * scale, v[1] * scale, v[2] * scale];
        }
        if core::f64::consts::PI - theta < SMALL_ANGLE {
            // Near π the antisymmetric part vanishes; recover the axis from
            // the symmetric part R + I = 2 n nᵀ (for θ = π).
            let mut axis = [0.0; 3];
            let mut best = 0;
            for i in 1..3 {
                if m[i][i] > m[best][best] {
                    best = i;
                }
            }
            let d = ((m[best][best] + 1.0) * 0.5).max(0.0).sqrt();
            axis[best] = d;
            if d > 0.0 {
                for i in 0..3 {
                    if i != best {
                        axis[i] = (m[best][i] + m[i][best]) * 0.25 / d;
                    }
                }
            }
            let n = norm(axis).max(f64::MIN_POSITIVE);
            // Fix sign with the antisymmetric part where possible.
            let v = skew_part_vee(m);
            let sign = if dot(v, axis) < 0.0 { -1.0 } else { 1.0 };
            return [
                sign * theta * axis[0] / n,
                sign * theta * axis[1] / n,
                sign * theta * axis[2] / n,
            ];
        }
        let v = skew_part_vee(m);
        let scale = theta / (2.0 * theta.sin());
        [v[0] * scale, v[1] * scale, v[2] * scale]
    }

    /// Group composition `self · other`.
    #[inline]
    pub fn compose(&self, other: &Self) -> Self {
        Self(mat_mul(&self.0, &other.0))
    }

    /// Inverse (transpose).
    #[inline]
    pub fn inverse(&self) -> Self {
        Self(transpose(&self.0))
    }

    /// Rotates a vector.
    #[inline]
    pub fn apply(&self, v: Vec3) -> Vec3 {
        mat_vec(&self.0, v)
    }

    /// Body-frame exponential-map update: `R_{k+1} = R_k · exp([ω Δt]×)`.
    #[inline]
    pub fn advance(&self, body_angular_velocity: Vec3, dt: f64) -> Self {
        let w = body_angular_velocity;
        self.compose(&Self::exp([w[0] * dt, w[1] * dt, w[2] * dt]))
    }

    /// Frobenius norm of `RᵀR - I`; a diagnostic for orthonormality drift.
    pub fn orthonormality_defect(&self) -> f64 {
        let g = mat_mul(&transpose(&self.0), &self.0);
        let mut acc = 0.0;
        for i in 0..3 {
            for j in 0..3 {
                let d = g[i][j] - IDENTITY[i][j];
                acc += d * d;
            }
        }
        acc.sqrt()
    }

    /// Re-projects onto SO(3) with one step of the symmetric Newton iteration
    /// `R ← R (3I - RᵀR)/2`. Only needed to scrub accumulated rounding after
    /// very long runs; the exponential update itself never leaves the group.
    pub fn reorthonormalize(&self) -> Self {
        let g = mat_mul(&transpose(&self.0), &self.0);
        let mut corr = [[0.0; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                corr[i][j] = (3.0 * IDENTITY[i][j] - g[i][j]) * 0.5;
            }
        }
        Self(mat_mul(&self.0, &corr))
    }
}

/// Rigid transform in SE(3): `x ↦ R x + t`.
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub struct Pose {
    /// Orientation.
    pub rotation: Rotation,
    /// Position.
    pub translation: Vec3,
}

/// Body-frame twist `ξ = (ω, v)` ∈ se(3).
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub struct Twist {
    /// Angular velocity.
    pub angular: Vec3,
    /// Linear velocity.
    pub linear: Vec3,
}

impl Pose {
    /// Identity pose.
    pub const IDENTITY: Self = Self {
        rotation: Rotation::IDENTITY,
        translation: [0.0; 3],
    };

    /// Exponential map se(3) → SE(3): `exp(ξ) = (exp([ω]×), V(ω) v)` with
    /// `V = I + B [ω]× + C [ω]×²`.
    pub fn exp(xi: Twist) -> Self {
        let theta = norm(xi.angular);
        let (_, b, c) = rodrigues_coefficients(theta);
        let k = hat(xi.angular);
        let k2 = mat_mul(&k, &k);
        let mut v_mat = IDENTITY;
        for i in 0..3 {
            for j in 0..3 {
                v_mat[i][j] += b * k[i][j] + c * k2[i][j];
            }
        }
        Self {
            rotation: Rotation::exp(xi.angular),
            translation: mat_vec(&v_mat, xi.linear),
        }
    }

    /// Logarithm map SE(3) → se(3).
    pub fn log(&self) -> Twist {
        let omega = self.rotation.log();
        let theta = norm(omega);
        let k = hat(omega);
        let k2 = mat_mul(&k, &k);
        // V⁻¹ = I - ½[ω]× + (1/θ²)(1 - A/(2B)) [ω]×²
        let coeff = if theta < SMALL_ANGLE {
            1.0 / 12.0 + theta * theta / 720.0
        } else {
            let (a, b, _) = rodrigues_coefficients(theta);
            (1.0 - a / (2.0 * b)) / (theta * theta)
        };
        let mut v_inv = IDENTITY;
        for i in 0..3 {
            for j in 0..3 {
                v_inv[i][j] += -0.5 * k[i][j] + coeff * k2[i][j];
            }
        }
        Twist {
            angular: omega,
            linear: mat_vec(&v_inv, self.translation),
        }
    }

    /// Group composition `self · other`.
    pub fn compose(&self, other: &Self) -> Self {
        let rt = self.rotation.apply(other.translation);
        Self {
            rotation: self.rotation.compose(&other.rotation),
            translation: [
                rt[0] + self.translation[0],
                rt[1] + self.translation[1],
                rt[2] + self.translation[2],
            ],
        }
    }

    /// Inverse transform.
    pub fn inverse(&self) -> Self {
        let r_inv = self.rotation.inverse();
        let t = r_inv.apply(self.translation);
        Self {
            rotation: r_inv,
            translation: [-t[0], -t[1], -t[2]],
        }
    }

    /// Applies the transform to a point.
    #[inline]
    pub fn apply(&self, p: Vec3) -> Vec3 {
        let r = self.rotation.apply(p);
        [
            r[0] + self.translation[0],
            r[1] + self.translation[1],
            r[2] + self.translation[2],
        ]
    }

    /// Body-frame exponential-map update: `g_{k+1} = g_k · exp(ξ Δt)`.
    pub fn advance(&self, body_twist: Twist, dt: f64) -> Self {
        let scaled = Twist {
            angular: [
                body_twist.angular[0] * dt,
                body_twist.angular[1] * dt,
                body_twist.angular[2] * dt,
            ],
            linear: [
                body_twist.linear[0] * dt,
                body_twist.linear[1] * dt,
                body_twist.linear[2] * dt,
            ],
        };
        self.compose(&Self::exp(scaled))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::f64::consts::PI;

    fn assert_vec_close(a: Vec3, b: Vec3, tol: f64) {
        for i in 0..3 {
            assert!((a[i] - b[i]).abs() < tol, "{a:?} vs {b:?}");
        }
    }

    #[test]
    fn exp_log_round_trip_so3() {
        for &w in &[
            [0.1, -0.2, 0.3],
            [1e-9, 2e-9, -1e-9],
            [2.5, 0.0, 0.0],
            [0.0, 3.0, 0.5],
        ] {
            let r = Rotation::exp(w);
            assert!(r.orthonormality_defect() < 1e-14);
            assert!((det(&r.0) - 1.0).abs() < 1e-14);
            assert_vec_close(r.log(), w, 1e-12);
        }
    }

    #[test]
    fn log_near_pi() {
        let axis = [0.0, 0.0, 1.0];
        let r = Rotation::exp([0.0, 0.0, PI - 1e-8]);
        let w = r.log();
        assert!((norm(w) - (PI - 1e-8)).abs() < 1e-6);
        assert!(dot(w, axis) > 0.0);
    }

    #[test]
    fn exp_log_round_trip_se3() {
        let xi = Twist {
            angular: [0.3, -0.4, 0.5],
            linear: [1.0, -2.0, 0.25],
        };
        let g = Pose::exp(xi);
        let back = g.log();
        assert_vec_close(back.angular, xi.angular, 1e-12);
        assert_vec_close(back.linear, xi.linear, 1e-12);
        let id = g.compose(&g.inverse());
        assert_vec_close(id.translation, [0.0; 3], 1e-14);
        assert!(id.rotation.orthonormality_defect() < 1e-14);
    }

    #[test]
    fn constant_twist_integrates_to_exp() {
        // Composition of exponentials of a constant body twist equals a
        // single exponential of the total displacement.
        let xi = Twist {
            angular: [0.0, 0.0, 1.0],
            linear: [1.0, 0.0, 0.0],
        };
        let mut g = Pose::IDENTITY;
        let n = 1000;
        let dt = 0.001;
        for _ in 0..n {
            g = g.advance(xi, dt);
        }
        let expected = Pose::exp(Twist {
            angular: [0.0, 0.0, 1.0],
            linear: [1.0, 0.0, 0.0],
        });
        assert_vec_close(g.translation, expected.translation, 1e-12);
        assert_vec_close(g.rotation.log(), expected.rotation.log(), 1e-12);
    }

    #[test]
    fn long_run_stays_on_manifold() {
        // 10^6 exponential-map steps around a skew axis; orthonormality
        // defect must stay at rounding level with no renormalisation.
        let mut r = Rotation::IDENTITY;
        let w = [0.7, -0.3, 0.2];
        for _ in 0..1_000_000 {
            r = r.advance(w, 1e-3);
        }
        assert!(
            r.orthonormality_defect() < 1e-10,
            "{}",
            r.orthonormality_defect()
        );
        assert!((det(&r.0) - 1.0).abs() < 1e-10);
        let scrubbed = r.reorthonormalize();
        assert!(scrubbed.orthonormality_defect() < 1e-14);
    }
}
