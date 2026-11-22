//! Non-Euclidean 2D manifold geometries: Hyperbolic Poincaré disk, Unit Sphere, and 2D Torus.

use crate::error::{CoreError, CoreResult};
use crate::layout::Layout2d;
use crate::scalar::Scalar;
use std::f64::consts::{PI, TAU};

/// Maximum radius enforced inside the open Poincaré unit disk to keep `1 - ||p||^2 > 0`.
const POINCARE_EPS_F64: f64 = 1e-5;
const POINCARE_EPS_F32: f64 = 1e-4;

/// Supported 2D non-Euclidean manifold geometries.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum ManifoldKind {
    /// 2D Poincaré unit disk model of hyperbolic space (`x^2 + y^2 < 1`).
    HyperbolicPoincare,
    /// 2D unit sphere parameterized by `(longitude, colatitude)` in `[0, 2π) × [0, π]`.
    Spherical,
    /// 2D flat unit torus `[0, 1)^2` with periodic wrap-around boundaries.
    Torus2d,
}

/// 2D node layout constrained to a specific non-Euclidean [`ManifoldKind`].
#[derive(Debug, Clone, PartialEq)]
pub struct ManifoldLayout2d<S: Scalar = f64> {
    kind: ManifoldKind,
    layout: Layout2d<S>,
}

impl<S: Scalar> ManifoldLayout2d<S> {
    /// Creates a zero-initialized manifold layout of `n` vertices.
    pub fn zeros(kind: ManifoldKind, n: usize) -> Self {
        Self {
            kind,
            layout: Layout2d::zeros(n),
        }
    }

    /// Wraps an existing [`Layout2d<S>`], validates finiteness, and projects all vertices
    /// onto `kind`'s canonical domain.
    pub fn from_layout(kind: ManifoldKind, layout: Layout2d<S>) -> CoreResult<Self> {
        let mut manifold = Self { kind, layout };
        manifold.project_all()?;
        Ok(manifold)
    }

    /// Constructs a manifold layout from `x` and `y` vectors and projects them onto `kind`.
    pub fn from_xy(kind: ManifoldKind, x: Vec<S>, y: Vec<S>) -> CoreResult<Self> {
        Self::from_layout(kind, Layout2d::from_xy(x, y)?)
    }

    /// Returns the manifold geometry type.
    #[inline]
    pub fn kind(&self) -> ManifoldKind {
        self.kind
    }

    /// Returns the number of vertices.
    #[inline]
    pub fn len(&self) -> usize {
        self.layout.len()
    }

    /// Returns `true` if there are zero vertices.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.layout.is_empty()
    }

    /// Returns an immutable reference to the underlying [`Layout2d<S>`].
    #[inline]
    pub fn as_layout(&self) -> &Layout2d<S> {
        &self.layout
    }

    /// Consumes `self` and returns the underlying [`Layout2d<S>`].
    #[inline]
    pub fn into_layout(self) -> Layout2d<S> {
        self.layout
    }

    /// Returns an immutable slice of the first coordinate axis (`x` or `lon`).
    #[inline]
    pub fn x(&self) -> &[S] {
        self.layout.x()
    }

    /// Returns an immutable slice of the second coordinate axis (`y` or `colat`).
    #[inline]
    pub fn y(&self) -> &[S] {
        self.layout.y()
    }

    /// Projects all coordinates onto the manifold's valid domain after checking finiteness.
    pub fn project_all(&mut self) -> CoreResult<()> {
        self.layout.validate_finite()?;
        let n = self.layout.len();
        let (xs, ys) = self.layout.xy_mut();
        for i in 0..n {
            let (px, py) = project_point(self.kind, xs[i], ys[i]);
            xs[i] = px;
            ys[i] = py;
        }
        Ok(())
    }

    /// Sets vertex `index`'s coordinates, projecting onto the manifold domain.
    pub fn set_point(&mut self, index: usize, x: S, y: S) -> CoreResult<()> {
        if index >= self.len() {
            return Err(CoreError::index_out_of_bounds(
                index,
                self.len(),
                "ManifoldLayout2d::set_point",
            ));
        }
        if !x.is_finite() || !y.is_finite() {
            return Err(CoreError::non_finite("ManifoldLayout2d::set_point"));
        }
        let (px, py) = project_point(self.kind, x, y);
        self.layout.x_mut()[index] = px;
        self.layout.y_mut()[index] = py;
        Ok(())
    }

    /// Computes the exact geodesic distance between vertices `i` and `j` on the manifold.
    pub fn geodesic_distance(&self, i: usize, j: usize) -> CoreResult<S> {
        let n = self.len();
        if i >= n {
            return Err(CoreError::index_out_of_bounds(
                i,
                n,
                "ManifoldLayout2d::geodesic_distance",
            ));
        }
        if j >= n {
            return Err(CoreError::index_out_of_bounds(
                j,
                n,
                "ManifoldLayout2d::geodesic_distance",
            ));
        }
        let x1 = self.layout.x()[i];
        let y1 = self.layout.y()[i];
        let x2 = self.layout.x()[j];
        let y2 = self.layout.y()[j];
        Ok(match self.kind {
            ManifoldKind::HyperbolicPoincare => poincare_distance(x1, y1, x2, y2),
            ManifoldKind::Spherical => spherical_distance(x1, y1, x2, y2),
            ManifoldKind::Torus2d => torus_distance(x1, y1, x2, y2),
        })
    }

    /// Computes the Riemannian logarithmic map (tangent vector at vertex `i` pointing toward `j`
    /// with Euclidean norm equal to the geodesic distance).
    pub fn tangent_delta(&self, i: usize, j: usize) -> CoreResult<(S, S)> {
        let n = self.len();
        if i >= n {
            return Err(CoreError::index_out_of_bounds(
                i,
                n,
                "ManifoldLayout2d::tangent_delta",
            ));
        }
        if j >= n {
            return Err(CoreError::index_out_of_bounds(
                j,
                n,
                "ManifoldLayout2d::tangent_delta",
            ));
        }
        let x1 = self.layout.x()[i];
        let y1 = self.layout.y()[i];
        let x2 = self.layout.x()[j];
        let y2 = self.layout.y()[j];
        Ok(match self.kind {
            ManifoldKind::HyperbolicPoincare => poincare_log_map(x1, y1, x2, y2),
            ManifoldKind::Spherical => spherical_log_map(x1, y1, x2, y2),
            ManifoldKind::Torus2d => torus_delta(x2, y2, x1, y1),
        })
    }

    /// Advances vertex `i` along the geodesic defined by tangent vector `(vx, vy)`.
    pub fn apply_tangent_step(&mut self, i: usize, vx: S, vy: S) -> CoreResult<()> {
        let n = self.len();
        if i >= n {
            return Err(CoreError::index_out_of_bounds(
                i,
                n,
                "ManifoldLayout2d::apply_tangent_step",
            ));
        }
        if !vx.is_finite() || !vy.is_finite() {
            return Err(CoreError::non_finite(
                "ManifoldLayout2d::apply_tangent_step",
            ));
        }
        let x = self.layout.x()[i];
        let y = self.layout.y()[i];
        let (nx, ny) = match self.kind {
            ManifoldKind::HyperbolicPoincare => poincare_exp_map(x, y, vx, vy),
            ManifoldKind::Spherical => spherical_exp_map(x, y, vx, vy),
            ManifoldKind::Torus2d => wrap_torus_point(x + vx, y + vy),
        };
        self.layout.x_mut()[i] = nx;
        self.layout.y_mut()[i] = ny;
        Ok(())
    }
}

#[inline]
fn project_point<S: Scalar>(kind: ManifoldKind, x: S, y: S) -> (S, S) {
    match kind {
        ManifoldKind::HyperbolicPoincare => clamp_poincare_point(x, y),
        ManifoldKind::Spherical => project_spherical_coords(x, y),
        ManifoldKind::Torus2d => wrap_torus_point(x, y),
    }
}

#[inline]
fn poincare_max_radius<S: Scalar>() -> f64 {
    if std::mem::size_of::<S>() == 4 {
        1.0 - POINCARE_EPS_F32
    } else {
        1.0 - POINCARE_EPS_F64
    }
}

/// Clamps a point `(x, y)` strictly inside the open Poincaré unit disk (`||p|| <= 1 - eps`).
#[inline]
pub fn clamp_poincare_point<S: Scalar>(x: S, y: S) -> (S, S) {
    let xf = x.to_f64();
    let yf = y.to_f64();
    if !xf.is_finite() || !yf.is_finite() {
        return (S::ZERO, S::ZERO);
    }
    let max_r = poincare_max_radius::<S>();
    let r = xf.hypot(yf);
    if r > max_r {
        let scale = max_r / r;
        (S::from_f64(xf * scale), S::from_f64(yf * scale))
    } else {
        (x, y)
    }
}

/// Computes the Poincaré disk hyperbolic geodesic distance between `(x1, y1)` and `(x2, y2)`.
pub fn poincare_distance<S: Scalar>(x1: S, y1: S, x2: S, y2: S) -> S {
    let (u0, u1) = clamp_poincare_point(x1, y1);
    let (v0, v1) = clamp_poincare_point(x2, y2);
    let ux = u0.to_f64();
    let uy = u1.to_f64();
    let vx = v0.to_f64();
    let vy = v1.to_f64();

    let dx = ux - vx;
    let dy = uy - vy;
    let diff_sq = dx * dx + dy * dy;
    let denom = (1.0 - (ux * ux + uy * uy)) * (1.0 - (vx * vx + vy * vy));
    let arg = (1.0 + 2.0 * diff_sq / denom.max(1e-18)).max(1.0);
    S::from_f64(arg.acosh())
}

/// Computes the Poincaré disk logarithmic map $\log_x(y)$ in the tangent space at `x = (x1, y1)`.
///
/// The returned vector points toward `y = (x2, y2)` via the Möbius translation $-x \oplus y$
/// and has Euclidean norm equal to [`poincare_distance`].
pub fn poincare_log_map<S: Scalar>(x1: S, y1: S, x2: S, y2: S) -> (S, S) {
    let (cx1, cy1) = clamp_poincare_point(x1, y1);
    let (cx2, cy2) = clamp_poincare_point(x2, y2);
    let xx = cx1.to_f64();
    let xy = cy1.to_f64();
    let yx = cx2.to_f64();
    let yy = cy2.to_f64();

    let dx = yx - xx;
    let dy = yy - xy;
    let dr = 1.0 - xx * yx - xy * yy;
    let di = xy * yx - xx * yy;
    let denom = dr * dr + di * di;
    if denom <= 1e-24 {
        return (S::ZERO, S::ZERO);
    }
    let zx = (dr * dx + di * dy) / denom;
    let zy = (dr * dy - di * dx) / denom;
    let z_norm = zx.hypot(zy);
    if z_norm <= 1e-15 {
        return (S::ZERO, S::ZERO);
    }
    let z_clamped = z_norm.min(1.0 - 1e-15);
    let dist = 2.0 * z_clamped.atanh();
    let scale = dist / z_norm;
    (S::from_f64(zx * scale), S::from_f64(zy * scale))
}

/// Computes the Poincaré disk exponential map $\exp_x(v)$ advancing `x = (x, y)` by tangent
/// vector `v = (vx, vy)` and clamping the result inside the open unit disk.
pub fn poincare_exp_map<S: Scalar>(x: S, y: S, vx: S, vy: S) -> (S, S) {
    let (cx, cy) = clamp_poincare_point(x, y);
    let xx = cx.to_f64();
    let xy = cy.to_f64();
    let vxf = vx.to_f64();
    let vyf = vy.to_f64();

    let v_norm = vxf.hypot(vyf);
    if v_norm <= 1e-15 {
        return (cx, cy);
    }
    let scale = (0.5 * v_norm).tanh() / v_norm;
    let ux = vxf * scale;
    let uy = vyf * scale;

    // Complex Mobius addition (x + u) / (1 + conj(x) * u), exact inverse of `poincare_log_map`.
    let num_x = xx + ux;
    let num_y = xy + uy;
    let den_r = 1.0 + xx * ux + xy * uy;
    let den_i = xx * uy - xy * ux;
    let den_sq = den_r * den_r + den_i * den_i;
    if den_sq <= 1e-24 {
        return clamp_poincare_point(S::from_f64(ux), S::from_f64(uy));
    }
    let rx = (num_x * den_r + num_y * den_i) / den_sq;
    let ry = (num_y * den_r - num_x * den_i) / den_sq;
    clamp_poincare_point(S::from_f64(rx), S::from_f64(ry))
}

/// Normalizes `(longitude, colatitude)` coordinates onto `[0, 2π) × [0, π]`.
///
/// Colatitudes outside `[0, π]` reflect across the north/south poles while shifting
/// longitude by `π`, preserving the 3D point on the unit sphere.
pub fn project_spherical_coords<S: Scalar>(lon: S, colat: S) -> (S, S) {
    let mut lon_f = lon.to_f64();
    let mut t = colat.to_f64().rem_euclid(TAU);
    if t > PI {
        t = TAU - t;
        lon_f += PI;
    }
    lon_f = lon_f.rem_euclid(TAU);
    if lon_f >= TAU {
        lon_f = 0.0;
    }
    (S::from_f64(lon_f), S::from_f64(t.clamp(0.0, PI)))
}

/// Converts `(longitude, colatitude)` to a 3D unit vector `(x, y, z)` on $S^2$.
///
/// Uses the `egraph-rs` convention where colatitude `0` is the north pole `(0, 1, 0)`.
#[inline]
pub fn spherical_to_cartesian<S: Scalar>(lon: S, colat: S) -> [S; 3] {
    let (slon, clon) = lon.to_f64().sin_cos();
    let (scolat, ccolat) = colat.to_f64().sin_cos();
    [
        S::from_f64(clon * scolat),
        S::from_f64(ccolat),
        S::from_f64(slon * scolat),
    ]
}

/// Projects a 3D vector `(x, y, z)` onto the unit sphere `(longitude, colatitude)` in `[0, 2π) × [0, π]`.
///
/// Zero or non-finite vectors map to the north pole `(0, 0)`.
pub fn cartesian_to_spherical<S: Scalar>(x: S, y: S, z: S) -> (S, S) {
    let xf = x.to_f64();
    let yf = y.to_f64();
    let zf = z.to_f64();
    let norm = (xf * xf + yf * yf + zf * zf).sqrt();
    if !norm.is_finite() || norm <= 1e-18 {
        return (S::ZERO, S::ZERO);
    }
    let nx = xf / norm;
    let ny = (yf / norm).clamp(-1.0, 1.0);
    let nz = zf / norm;
    let colat = ny.acos();
    let lon = if nx * nx + nz * nz <= 1e-24 {
        0.0
    } else {
        let mut l = nz.atan2(nx).rem_euclid(TAU);
        if l >= TAU {
            l = 0.0;
        }
        l
    };
    (S::from_f64(lon), S::from_f64(colat))
}

#[inline]
fn unit_sphere_f64(lon: f64, colat: f64) -> [f64; 3] {
    let (slon, clon) = lon.sin_cos();
    let (scolat, ccolat) = colat.sin_cos();
    [clon * scolat, ccolat, slon * scolat]
}

/// Computes the great-circle geodesic distance on the unit sphere in `[0, π]`.
///
/// Uses `atan2(||p1 × p2||, p1 · p2)` in `f64` for full precision near `0` and `π`.
pub fn spherical_distance<S: Scalar>(lon1: S, colat1: S, lon2: S, colat2: S) -> S {
    let p1 = unit_sphere_f64(lon1.to_f64(), colat1.to_f64());
    let p2 = unit_sphere_f64(lon2.to_f64(), colat2.to_f64());
    let dot = (p1[0] * p2[0] + p1[1] * p2[1] + p1[2] * p2[2]).clamp(-1.0, 1.0);
    let cx = p1[1] * p2[2] - p1[2] * p2[1];
    let cy = p1[2] * p2[0] - p1[0] * p2[2];
    let cz = p1[0] * p2[1] - p1[1] * p2[0];
    let cross_norm = (cx * cx + cy * cy + cz * cz).sqrt();
    S::from_f64(cross_norm.atan2(dot))
}

/// Computes the tangent vector at `(lon1, colat1)` pointing along the great circle toward `(lon2, colat2)`.
pub fn spherical_log_map<S: Scalar>(lon1: S, colat1: S, lon2: S, colat2: S) -> (S, S) {
    let lon1_f = lon1.to_f64();
    let colat1_f = colat1.to_f64();
    let p1 = unit_sphere_f64(lon1_f, colat1_f);
    let p2 = unit_sphere_f64(lon2.to_f64(), colat2.to_f64());

    let dot = (p1[0] * p2[0] + p1[1] * p2[1] + p1[2] * p2[2]).clamp(-1.0, 1.0);
    let cx = p1[1] * p2[2] - p1[2] * p2[1];
    let cy = p1[2] * p2[0] - p1[0] * p2[2];
    let cz = p1[0] * p2[1] - p1[1] * p2[0];
    let cross_norm = (cx * cx + cy * cy + cz * cz).sqrt();
    let angle = cross_norm.atan2(dot);
    if angle <= 1e-15 {
        return (S::ZERO, S::ZERO);
    }

    // Orthonormal tangent basis (e_lon, e_colat) at (lon1, colat1).
    let (slon, clon) = lon1_f.sin_cos();
    let (scolat, ccolat) = colat1_f.sin_cos();
    let e_lon = [-slon, 0.0, clon];
    let e_colat = [clon * ccolat, -scolat, slon * ccolat];

    // Tangent direction in R^3: p2 - (p1 · p2) p1.
    let tx = p2[0] - dot * p1[0];
    let ty = p2[1] - dot * p1[1];
    let tz = p2[2] - dot * p1[2];
    let t_norm = (tx * tx + ty * ty + tz * tz).sqrt();
    if t_norm <= 1e-15 {
        // Antipodal fallback: advance southward along the local meridian.
        return (S::ZERO, S::from_f64(angle));
    }
    let scale = angle / t_norm;
    let v_lon = scale * (tx * e_lon[0] + ty * e_lon[1] + tz * e_lon[2]);
    let v_colat = scale * (tx * e_colat[0] + ty * e_colat[1] + tz * e_colat[2]);
    (S::from_f64(v_lon), S::from_f64(v_colat))
}

/// Advances `(lon, colat)` along the great circle determined by tangent vector `(vx, vy)`.
pub fn spherical_exp_map<S: Scalar>(lon: S, colat: S, vx: S, vy: S) -> (S, S) {
    let lon_f = lon.to_f64();
    let colat_f = colat.to_f64();
    let vxf = vx.to_f64();
    let vyf = vy.to_f64();
    let theta = vxf.hypot(vyf);
    if theta <= 1e-15 {
        return project_spherical_coords(lon, colat);
    }

    let p = unit_sphere_f64(lon_f, colat_f);
    let (slon, clon) = lon_f.sin_cos();
    let (scolat, ccolat) = colat_f.sin_cos();
    let e_lon = [-slon, 0.0, clon];
    let e_colat = [clon * ccolat, -scolat, slon * ccolat];

    let dir = [
        (vxf * e_lon[0] + vyf * e_colat[0]) / theta,
        (vxf * e_lon[1] + vyf * e_colat[1]) / theta,
        (vxf * e_lon[2] + vyf * e_colat[2]) / theta,
    ];
    let (st, ct) = theta.sin_cos();
    let qx = ct * p[0] + st * dir[0];
    let qy = ct * p[1] + st * dir[1];
    let qz = ct * p[2] + st * dir[2];
    cartesian_to_spherical(S::from_f64(qx), S::from_f64(qy), S::from_f64(qz))
}

/// Wraps a scalar coordinate into the half-open unit torus interval `[0, 1)`.
#[inline]
pub fn wrap_torus_coord<S: Scalar>(value: S) -> S {
    let f = value.to_f64();
    if !f.is_finite() {
        return S::ZERO;
    }
    let w = f.rem_euclid(1.0);
    let wrapped = if w >= 1.0 { 0.0 } else { w };
    let res = S::from_f64(wrapped);
    // Guard f32 rounding where values like `-1e-9` can round up to `1.0_f32`.
    if res >= S::ONE { S::ZERO } else { res }
}

/// Wraps a 2D coordinate pair onto the flat unit torus `[0, 1)^2`.
#[inline]
pub fn wrap_torus_point<S: Scalar>(x: S, y: S) -> (S, S) {
    (wrap_torus_coord(x), wrap_torus_coord(y))
}

/// Computes the shortest signed displacement vector `(dx, dy)` in `[-0.5, 0.5]^2`
/// from `(x2, y2)` to `(x1, y1)` on the unit torus `[0, 1)^2`.
#[inline]
pub fn torus_delta<S: Scalar>(x1: S, y1: S, x2: S, y2: S) -> (S, S) {
    let dx = wrap_1d_signed_delta(wrap_torus_coord(x1).to_f64() - wrap_torus_coord(x2).to_f64());
    let dy = wrap_1d_signed_delta(wrap_torus_coord(y1).to_f64() - wrap_torus_coord(y2).to_f64());
    (S::from_f64(dx), S::from_f64(dy))
}

#[inline]
fn wrap_1d_signed_delta(mut d: f64) -> f64 {
    if d > 0.5 {
        d -= 1.0;
    } else if d < -0.5 {
        d += 1.0;
    }
    d
}

/// Computes the geodesic distance on the flat unit torus `[0, 1)^2`.
#[inline]
pub fn torus_distance<S: Scalar>(x1: S, y1: S, x2: S, y2: S) -> S {
    let (dx, dy) = torus_delta(x1, y1, x2, y2);
    dx.hypot(dy)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_poincare_clamp_distance_and_exp_log_roundtrip() {
        let (cx, cy) = clamp_poincare_point(3.0_f64, 4.0);
        assert!(cx.hypot(cy) < 1.0);
        assert!((cx / cy - 0.75).abs() < 1e-12);

        let origin_dist = poincare_distance(0.0_f64, 0.0, 0.5, 0.0);
        let expected = 2.0 * 0.5_f64.atanh();
        assert!((origin_dist - expected).abs() < 1e-12);

        let x = (0.2_f64, -0.3);
        let y = (-0.4_f64, 0.25);
        let (vx, vy) = poincare_log_map(x.0, x.1, y.0, y.1);
        let d = poincare_distance(x.0, x.1, y.0, y.1);
        assert!((vx.hypot(vy) - d).abs() < 1e-10);
        let (rx, ry) = poincare_exp_map(x.0, x.1, vx, vy);
        assert!((rx - y.0).abs() < 1e-10);
        assert!((ry - y.1).abs() < 1e-10);
    }

    #[test]
    fn test_spherical_projection_distance_and_exp_log() {
        let (lon, colat) = project_spherical_coords(-0.5_f64, -0.25);
        assert!((0.0..TAU).contains(&lon));
        assert!((0.0..=PI).contains(&colat));
        assert!((colat - 0.25).abs() < 1e-12);

        let p1 = (0.2_f64, PI / 2.0);
        let p2 = (0.2 + PI / 2.0, PI / 2.0);
        let d = spherical_distance(p1.0, p1.1, p2.0, p2.1);
        assert!((d - PI / 2.0).abs() < 1e-12);

        let (vx, vy) = spherical_log_map(0.4_f64, 1.1, 1.2, 0.7);
        let d_pair = spherical_distance(0.4_f64, 1.1, 1.2, 0.7);
        assert!((vx.hypot(vy) - d_pair).abs() < 1e-12);
        let (rlon, rcolat) = spherical_exp_map(0.4_f64, 1.1, vx, vy);
        assert!(spherical_distance(rlon, rcolat, 1.2, 0.7) < 1e-11);
    }

    #[test]
    fn test_torus_wrap_delta_and_distance() {
        assert_eq!(wrap_torus_coord(-1.0_f64), 0.0);
        assert_eq!(wrap_torus_coord(-1e-17_f64), 0.0);
        assert!((wrap_torus_coord(-0.125_f64) - 0.875).abs() < 1e-12);
        assert_eq!(wrap_torus_coord(-1e-9_f32), 0.0_f32);

        let (dx, dy) = torus_delta(0.0_f64, 0.75, 0.5, 0.5);
        assert!((dx - (-0.5)).abs() < 1e-12);
        assert!((dy - 0.25).abs() < 1e-12);

        let d_wrap = torus_distance(0.95_f64, 0.05, 0.05, 0.95);
        assert!((d_wrap - (0.1_f64).hypot(0.1)).abs() < 1e-12);
    }

    #[test]
    fn test_manifold_layout_2d_api() {
        let mut m =
            ManifoldLayout2d::from_xy(ManifoldKind::Torus2d, vec![1.2, 0.9], vec![-0.1, 0.2])
                .unwrap();
        assert_eq!(m.kind(), ManifoldKind::Torus2d);
        assert_eq!(m.len(), 2);
        assert!((m.x()[0] - 0.2).abs() < 1e-12);
        assert!((m.y()[0] - 0.9).abs() < 1e-12);

        let d_before = m.geodesic_distance(0, 1).unwrap();
        let (vx, vy) = m.tangent_delta(0, 1).unwrap();
        assert!((vx.hypot(vy) - d_before).abs() < 1e-12);
        m.apply_tangent_step(0, vx * 0.5, vy * 0.5).unwrap();
        let d_after = m.geodesic_distance(0, 1).unwrap();
        assert!((d_after - d_before * 0.5).abs() < 1e-12);

        assert!(matches!(
            m.set_point(2, 0.0, 0.0),
            Err(CoreError::IndexOutOfBounds { .. })
        ));
        assert!(matches!(
            ManifoldLayout2d::from_xy(ManifoldKind::Spherical, vec![f64::NAN], vec![0.0]),
            Err(CoreError::NonFiniteValue { .. })
        ));
    }
}
