use std::f64::consts::TAU;

use super::lorentz::{lorentz_distance, lorentz_exp_map, lorentz_log_map};
use super::poincare::{
    clamp_poincare_point, poincare_distance, poincare_exp_map, poincare_log_map,
    poincare_max_radius,
};
use super::spherical::{
    project_spherical_coords, spherical_distance, spherical_exp_map, spherical_log_map,
};
use super::torus::{torus_delta, wrap_torus_point};
use super::{ManifoldKind, ManifoldMetric};
use crate::error::{CoreError, CoreResult};
use crate::layout::Layout2d;
use crate::rng::DeterministicRng;
use crate::scalar::Scalar;

/// 2D node layout constrained to a specific non-Euclidean [`ManifoldKind`].
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ManifoldLayout2d<S: Scalar = f64> {
    kind: ManifoldKind,
    layout: Layout2d<S>,
    metric: ManifoldMetric,
}

impl<S: Scalar> ManifoldLayout2d<S> {
    #[inline]
    pub(super) fn from_parts_unchecked(
        kind: ManifoldKind,
        layout: Layout2d<S>,
        metric: ManifoldMetric,
    ) -> Self {
        Self {
            kind,
            layout,
            metric,
        }
    }

    #[inline]
    pub(super) fn metric_ref(&self) -> &ManifoldMetric {
        &self.metric
    }

    #[inline]
    pub(super) fn metric_mut(&mut self) -> &mut ManifoldMetric {
        &mut self.metric
    }

    #[inline]
    pub(super) fn xy_mut(&mut self) -> (&mut [S], &mut [S]) {
        self.layout.xy_mut()
    }

    /// Creates a zero-initialized manifold layout of `n` vertices.
    pub fn zeros(kind: ManifoldKind, n: usize) -> Self {
        Self {
            kind,
            layout: Layout2d::zeros(n),
            metric: ManifoldMetric::default(),
        }
    }

    /// Wraps an existing [`Layout2d<S>`], validates finiteness, and projects all vertices
    /// onto `kind`'s canonical domain.
    pub fn from_layout(kind: ManifoldKind, layout: Layout2d<S>) -> CoreResult<Self> {
        let mut manifold = Self {
            kind,
            layout,
            metric: ManifoldMetric::default(),
        };
        manifold.project_all()?;
        Ok(manifold)
    }

    /// Constructs a manifold layout from `x` and `y` vectors and projects them onto `kind`.
    pub fn from_xy(kind: ManifoldKind, x: Vec<S>, y: Vec<S>) -> CoreResult<Self> {
        Self::from_layout(kind, Layout2d::from_xy(x, y)?)
    }

    /// Creates a deterministically seeded safe random initialization on `kind`.
    ///
    /// Arguments follow the `(kind, num_nodes, seed, scale)` convention shared with
    /// [`Layout2d::init`] and [`crate::LayoutNd::init`], delegating to [`Self::random_init`].
    #[inline]
    pub fn init(kind: ManifoldKind, num_nodes: usize, seed: u64, scale: f64) -> CoreResult<Self> {
        Self::random_init(kind, num_nodes, scale, seed)
    }

    /// Validates that every coordinate in the manifold layout is finite.
    #[inline]
    pub fn validate_finite(&self) -> CoreResult<()> {
        self.layout.validate_finite()
    }

    /// Creates a deterministically seeded safe random initialization on `kind`.
    ///
    /// For [`ManifoldKind::HyperbolicPoincare`], points are sampled uniformly by area inside the
    /// open disk of radius `scale.min(1 - eps)` (`0 < scale < 1`). For
    /// [`ManifoldKind::HyperbolicLorentz`], spatial hyperboloid coordinates are sampled inside a
    /// disk of radius `scale` (`scale > 0`). For [`ManifoldKind::Spherical`], points are sampled
    /// uniformly by surface area on $S^2$. For [`ManifoldKind::Torus2d`], points are sampled
    /// uniformly in $[0, 1)^2$.
    pub fn random_init(kind: ManifoldKind, n: usize, scale: f64, seed: u64) -> CoreResult<Self> {
        if !scale.is_finite() {
            return Err(CoreError::non_finite(
                "ManifoldLayout2d::random_init::scale",
            ));
        }
        if scale <= 0.0 {
            return Err(CoreError::invalid_input("scale must be strictly positive"));
        }
        if kind == ManifoldKind::HyperbolicPoincare && scale >= 1.0 {
            return Err(CoreError::invalid_input(
                "scale must be strictly in (0, 1) for HyperbolicPoincare",
            ));
        }
        if n == 0 {
            return Ok(Self::zeros(kind, 0));
        }

        let mut rng = DeterministicRng::from_seed_u64(seed);
        let mut xs = Vec::with_capacity(n);
        let mut ys = Vec::with_capacity(n);
        match kind {
            ManifoldKind::HyperbolicPoincare => {
                let max_r = scale.min(poincare_max_radius::<S>());
                for _ in 0..n {
                    let r = max_r * rng.next_f64().sqrt();
                    let theta = TAU * rng.next_f64();
                    let (s, c) = theta.sin_cos();
                    let (px, py) = clamp_poincare_point(S::from_f64(r * c), S::from_f64(r * s));
                    xs.push(px);
                    ys.push(py);
                }
            }
            ManifoldKind::HyperbolicLorentz => {
                for _ in 0..n {
                    let r = scale * rng.next_f64().sqrt();
                    let theta = TAU * rng.next_f64();
                    let (s, c) = theta.sin_cos();
                    xs.push(S::from_f64(r * c));
                    ys.push(S::from_f64(r * s));
                }
            }
            ManifoldKind::Spherical => {
                for _ in 0..n {
                    let lon = TAU * rng.next_f64();
                    let u = (2.0 * rng.next_f64() - 1.0).clamp(-1.0, 1.0);
                    let colat = u.acos();
                    let (px, py) = project_spherical_coords(S::from_f64(lon), S::from_f64(colat));
                    xs.push(px);
                    ys.push(py);
                }
            }
            ManifoldKind::Torus2d => {
                for _ in 0..n {
                    let (px, py) =
                        wrap_torus_point(S::from_f64(rng.next_f64()), S::from_f64(rng.next_f64()));
                    xs.push(px);
                    ys.push(py);
                }
            }
        }
        Ok(Self {
            kind,
            layout: Layout2d::from_xy(xs, ys)?,
            metric: ManifoldMetric::default(),
        })
    }

    /// Constructs a [`ManifoldLayout2d`] from a Euclidean [`Layout2d`] by centering and radially
    /// scaling coordinates inside `max_radius` (`0 < max_radius < 1`) on
    /// [`ManifoldKind::HyperbolicPoincare`], avoiding boundary rim collapse.
    ///
    /// For other manifold kinds, projects `layout` onto the manifold domain identically to
    /// [`ManifoldLayout2d::from_layout`].
    pub fn from_layout_scaled(
        kind: ManifoldKind,
        layout: &Layout2d<S>,
        max_radius: f64,
    ) -> CoreResult<Self> {
        if !max_radius.is_finite() {
            return Err(CoreError::non_finite(
                "ManifoldLayout2d::from_layout_scaled::max_radius",
            ));
        }
        if max_radius <= 0.0 || max_radius >= 1.0 {
            return Err(CoreError::invalid_input(
                "max_radius must be strictly in (0, 1)",
            ));
        }
        layout.validate_finite()?;
        if kind != ManifoldKind::HyperbolicPoincare {
            return Self::from_layout(kind, layout.clone());
        }

        let n = layout.len();
        if n == 0 {
            return Ok(Self::zeros(kind, 0));
        }
        let inv_n = 1.0 / (n as f64);
        let mx = layout.x().iter().map(|v| v.to_f64() * inv_n).sum::<f64>();
        let my = layout.y().iter().map(|v| v.to_f64() * inv_n).sum::<f64>();
        let mut r_max = 0.0_f64;
        for (&x, &y) in layout.x().iter().zip(layout.y().iter()) {
            let r = (x.to_f64() - mx).hypot(y.to_f64() - my);
            if r > r_max {
                r_max = r;
            }
        }
        let eff_max_r = max_radius.min(poincare_max_radius::<S>());
        let scale = if r_max.is_finite() && r_max > 0.0 {
            eff_max_r / r_max
        } else {
            0.0
        };
        let mut xs = Vec::with_capacity(n);
        let mut ys = Vec::with_capacity(n);
        for (&x, &y) in layout.x().iter().zip(layout.y().iter()) {
            let dx = x.to_f64() - mx;
            let dy = y.to_f64() - my;
            let (px, py) = clamp_poincare_point(S::from_f64(dx * scale), S::from_f64(dy * scale));
            xs.push(px);
            ys.push(py);
        }
        Ok(Self {
            kind,
            layout: Layout2d::from_xy(xs, ys)?,
            metric: ManifoldMetric::default(),
        })
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

    /// Computes the exact metric-scaled geodesic distance between vertices `i` and `j`.
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
        let s = self.metric.scale;
        Ok(match self.kind {
            ManifoldKind::HyperbolicPoincare => {
                S::from_f64(s * poincare_distance(x1, y1, x2, y2).to_f64())
            }
            ManifoldKind::HyperbolicLorentz => {
                S::from_f64(s * lorentz_distance(x1, y1, x2, y2).to_f64())
            }
            ManifoldKind::Spherical => S::from_f64(s * spherical_distance(x1, y1, x2, y2).to_f64()),
            ManifoldKind::Torus2d => {
                let (dx_norm, dy_norm) = torus_delta(x2, y2, x1, y1);
                let lx = s * self.metric.aspect;
                let ly = s;
                S::from_f64((lx * dx_norm.to_f64()).hypot(ly * dy_norm.to_f64()))
            }
        })
    }

    /// Computes the metric-scaled Riemannian logarithmic map (tangent vector at vertex `i`
    /// pointing toward `j` with Euclidean norm equal to the geodesic distance).
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
        let s = self.metric.scale;
        Ok(match self.kind {
            ManifoldKind::HyperbolicPoincare => {
                let (vx, vy) = poincare_log_map(x1, y1, x2, y2);
                (S::from_f64(s * vx.to_f64()), S::from_f64(s * vy.to_f64()))
            }
            ManifoldKind::HyperbolicLorentz => {
                let (vx, vy) = lorentz_log_map(x1, y1, x2, y2);
                (S::from_f64(s * vx.to_f64()), S::from_f64(s * vy.to_f64()))
            }
            ManifoldKind::Spherical => {
                let (vx, vy) = spherical_log_map(x1, y1, x2, y2);
                (S::from_f64(s * vx.to_f64()), S::from_f64(s * vy.to_f64()))
            }
            ManifoldKind::Torus2d => {
                let (dx_norm, dy_norm) = torus_delta(x2, y2, x1, y1);
                let lx = s * self.metric.aspect;
                let ly = s;
                (
                    S::from_f64(lx * dx_norm.to_f64()),
                    S::from_f64(ly * dy_norm.to_f64()),
                )
            }
        })
    }

    /// Advances vertex `i` along the geodesic defined by metric-scaled tangent vector `(vx, vy)`.
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
        let s = self.metric.scale;
        let (nx, ny) = match self.kind {
            ManifoldKind::HyperbolicPoincare => poincare_exp_map(
                x,
                y,
                S::from_f64(vx.to_f64() / s),
                S::from_f64(vy.to_f64() / s),
            ),
            ManifoldKind::HyperbolicLorentz => lorentz_exp_map(
                x,
                y,
                S::from_f64(vx.to_f64() / s),
                S::from_f64(vy.to_f64() / s),
            ),
            ManifoldKind::Spherical => spherical_exp_map(
                x,
                y,
                S::from_f64(vx.to_f64() / s),
                S::from_f64(vy.to_f64() / s),
            ),
            ManifoldKind::Torus2d => {
                let lx = s * self.metric.aspect;
                let ly = s;
                wrap_torus_point(
                    S::from_f64(x.to_f64() + vx.to_f64() / lx),
                    S::from_f64(y.to_f64() + vy.to_f64() / ly),
                )
            }
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
        ManifoldKind::HyperbolicLorentz => (x, y),
        ManifoldKind::Spherical => project_spherical_coords(x, y),
        ManifoldKind::Torus2d => wrap_torus_point(x, y),
    }
}
