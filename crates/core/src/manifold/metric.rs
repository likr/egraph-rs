use crate::csr::CsrGraph;
use crate::error::{CoreError, CoreResult};
use crate::layout::Layout2d;
use crate::scalar::Scalar;

use super::lorentz::{
    lorentz_einstein_midpoint, lorentz_exp_map, lorentz_log_map, lorentz_to_poincare_point,
};
use super::poincare::{
    clamp_poincare_point, poincare_einstein_midpoint, poincare_exp_map, poincare_log_map,
    poincare_mobius_add,
};
use super::torus::{extract_unique_undirected_edges, optimal_torus_cut_1d, wrap_torus_point};
use super::{ManifoldKind, ManifoldLayout2d};

/// Physical Riemannian metric scaling parameters carried by [`ManifoldLayout2d`].
///
/// For conformal manifolds ([`ManifoldKind::HyperbolicPoincare`],
/// [`ManifoldKind::HyperbolicLorentz`], [`ManifoldKind::Spherical`]), `scale` scales geodesic
/// distances and tangent vectors (`d_phys = scale * d_unit`).
/// For [`ManifoldKind::Torus2d`], coordinates remain stored in canonical `[0, 1)^2` while physical
/// box extents are `(Lx, Ly) = (scale * aspect, scale)`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ManifoldMetric {
    pub scale: f64,
    pub aspect: f64,
}

impl Default for ManifoldMetric {
    fn default() -> Self {
        Self {
            scale: 1.0,
            aspect: 1.0,
        }
    }
}

impl<S: Scalar> ManifoldLayout2d<S> {
    /// Returns the manifold's physical metric parameters `(scale, aspect)`.
    #[inline]
    pub fn metric(&self) -> ManifoldMetric {
        self.metric_ref()
    }

    /// Returns the manifold's physical geodesic scale factor (`> 0`).
    #[inline]
    pub fn scale(&self) -> f64 {
        self.metric_ref().scale
    }

    /// Returns the manifold's physical aspect ratio (`Lx / Ly` on [`ManifoldKind::Torus2d`], `> 0`).
    #[inline]
    pub fn aspect(&self) -> f64 {
        self.metric_ref().aspect
    }

    /// Sets the manifold's physical scale factor without modifying stored coordinates.
    pub fn set_scale(&mut self, scale: f64) -> CoreResult<()> {
        if !scale.is_finite() {
            return Err(CoreError::non_finite("ManifoldLayout2d::set_scale"));
        }
        if scale <= 0.0 {
            return Err(CoreError::invalid_input("scale must be strictly positive"));
        }
        self.metric_mut().scale = scale;
        Ok(())
    }

    /// Sets the manifold's aspect ratio without modifying stored coordinates.
    pub fn set_aspect(&mut self, aspect: f64) -> CoreResult<()> {
        if !aspect.is_finite() {
            return Err(CoreError::non_finite("ManifoldLayout2d::set_aspect"));
        }
        if aspect <= 0.0 {
            return Err(CoreError::invalid_input("aspect must be strictly positive"));
        }
        self.metric_mut().aspect = aspect;
        Ok(())
    }

    /// Sets physical rectangular torus extents `(lx, ly)` (`scale = ly`, `aspect = lx / ly`)
    /// without mutating canonical `[0, 1)^2` coordinates.
    ///
    /// Returns [`CoreError::UnsupportedOperation`] on non-torus manifolds.
    pub fn set_torus_extents(&mut self, lx: f64, ly: f64) -> CoreResult<()> {
        if self.kind() != ManifoldKind::Torus2d {
            return Err(CoreError::unsupported(
                "set_torus_extents is only supported on Torus2d manifolds",
            ));
        }
        if !lx.is_finite() || !ly.is_finite() {
            return Err(CoreError::non_finite("ManifoldLayout2d::set_torus_extents"));
        }
        if lx <= 0.0 || ly <= 0.0 {
            return Err(CoreError::invalid_input(
                "torus extents lx and ly must be strictly positive",
            ));
        }
        let metric = self.metric_mut();
        metric.scale = ly;
        metric.aspect = lx / ly;
        Ok(())
    }

    /// Updates the geodesic scale to `new_scale`, applying hyperbolic Einstein-midpoint radial
    /// rescaling $x_i' = \exp_o((s_{\mathrm{old}} / s_{\mathrm{new}}) \log_o(x_i))$ on
    /// [`ManifoldKind::HyperbolicPoincare`] and [`ManifoldKind::HyperbolicLorentz`].
    ///
    /// On [`ManifoldKind::Spherical`] and [`ManifoldKind::Torus2d`], updates `scale` while
    /// preserving canonical coordinates.
    pub fn rescale_geodesic(&mut self, new_scale: f64) -> CoreResult<()> {
        if !new_scale.is_finite() {
            return Err(CoreError::non_finite("ManifoldLayout2d::rescale_geodesic"));
        }
        if new_scale <= 0.0 {
            return Err(CoreError::invalid_input(
                "new_scale must be strictly positive",
            ));
        }
        self.validate_finite()?;

        let s_old = self.scale();
        let kind = self.kind();
        if !self.is_empty() && (s_old - new_scale).abs() > 0.0 {
            let ratio = s_old / new_scale;
            match kind {
                ManifoldKind::HyperbolicPoincare => {
                    let (ox, oy) = poincare_einstein_midpoint(self.x(), self.y())?;
                    let (xs, ys) = self.xy_mut();
                    for i in 0..xs.len() {
                        let (vx, vy) = poincare_log_map(ox, oy, xs[i], ys[i]);
                        let (nx, ny) = poincare_exp_map(
                            ox,
                            oy,
                            S::from_f64(vx.to_f64() * ratio),
                            S::from_f64(vy.to_f64() * ratio),
                        );
                        xs[i] = nx;
                        ys[i] = ny;
                    }
                }
                ManifoldKind::HyperbolicLorentz => {
                    let (ox, oy) = lorentz_einstein_midpoint(self.x(), self.y())?;
                    let (xs, ys) = self.xy_mut();
                    for i in 0..xs.len() {
                        let (vx, vy) = lorentz_log_map(ox, oy, xs[i], ys[i]);
                        let (nx, ny) = lorentz_exp_map(
                            ox,
                            oy,
                            S::from_f64(vx.to_f64() * ratio),
                            S::from_f64(vy.to_f64() * ratio),
                        );
                        xs[i] = nx;
                        ys[i] = ny;
                    }
                }
                ManifoldKind::Spherical | ManifoldKind::Torus2d => {}
            }
        }

        self.metric_mut().scale = new_scale;
        Ok(())
    }

    /// Converts a [`ManifoldKind::HyperbolicLorentz`] (or [`ManifoldKind::HyperbolicPoincare`])
    /// layout into a [`ManifoldKind::HyperbolicPoincare`] layout via $(x, y) \mapsto (x/(1+t), y/(1+t))$,
    /// clamping strictly inside the open unit disk and preserving [`ManifoldMetric`].
    pub fn to_poincare_layout(&self) -> CoreResult<Self> {
        self.validate_finite()?;
        match self.kind() {
            ManifoldKind::HyperbolicPoincare => {
                let mut cloned = self.clone();
                cloned.project_all()?;
                Ok(cloned)
            }
            ManifoldKind::HyperbolicLorentz => {
                let n = self.len();
                let mut xs = Vec::with_capacity(n);
                let mut ys = Vec::with_capacity(n);
                for (&x, &y) in self.x().iter().zip(self.y().iter()) {
                    let (px, py) = lorentz_to_poincare_point(x, y);
                    xs.push(px);
                    ys.push(py);
                }
                Ok(Self::from_parts_unchecked(
                    ManifoldKind::HyperbolicPoincare,
                    Layout2d::from_xy(xs, ys)?,
                    self.metric(),
                ))
            }
            ManifoldKind::Spherical | ManifoldKind::Torus2d => Err(CoreError::unsupported(
                "to_poincare_layout is only supported on hyperbolic manifolds",
            )),
        }
    }

    /// Returns a rigidly panned [`ManifoldKind::Torus2d`] layout minimizing boundary-wrapped edges
    /// and maximizing empty boundary gap width independently along both axes (Chen et al. 2021).
    ///
    /// Preserves [`ManifoldMetric`] (`scale`, `aspect`) and all pairwise geodesic distances.
    pub fn torus_auto_pan<W: Scalar>(&self, graph: &CsrGraph<W>) -> CoreResult<Self> {
        if self.kind() != ManifoldKind::Torus2d {
            return Err(CoreError::unsupported(
                "torus_auto_pan is only supported on Torus2d manifolds",
            ));
        }
        if self.len() != graph.num_nodes() {
            return Err(CoreError::dimension_mismatch(
                graph.num_nodes(),
                self.len(),
                "ManifoldLayout2d::torus_auto_pan",
            ));
        }
        self.validate_finite()?;
        if self.is_empty() {
            return Ok(self.clone());
        }

        let edges = extract_unique_undirected_edges(graph);
        let cut_x = optimal_torus_cut_1d(self.x(), &edges);
        let cut_y = optimal_torus_cut_1d(self.y(), &edges);

        let n = self.len();
        let mut xs = Vec::with_capacity(n);
        let mut ys = Vec::with_capacity(n);
        for (&x, &y) in self.x().iter().zip(self.y().iter()) {
            let (px, py) = wrap_torus_point(
                S::from_f64(x.to_f64() - cut_x),
                S::from_f64(y.to_f64() - cut_y),
            );
            xs.push(px);
            ys.push(py);
        }

        Ok(Self::from_parts_unchecked(
            ManifoldKind::Torus2d,
            Layout2d::from_xy(xs, ys)?,
            self.metric(),
        ))
    }

    /// Computes the closed-form Möbius-equivariant hyperbolic centroid (Einstein midpoint) of all
    /// vertices in the Poincaré disk.
    ///
    /// Returns [`CoreError::UnsupportedOperation`] on non-Poincaré manifolds.
    pub fn poincare_centroid(&self) -> CoreResult<(S, S)> {
        if self.kind() != ManifoldKind::HyperbolicPoincare {
            return Err(CoreError::unsupported(
                "poincare_centroid is only supported on HyperbolicPoincare manifolds",
            ));
        }
        poincare_einstein_midpoint(self.x(), self.y())
    }

    /// Isometrically recenters all Poincaré disk points at `(cx, cy)` via the holomorphic disk
    /// automorphism $z \mapsto (-c) \oplus_M z = \frac{z - c}{1 - \bar{c} z}$.
    ///
    /// Returns [`CoreError::UnsupportedOperation`] on non-Poincaré manifolds.
    pub fn recenter_poincare(&mut self, cx: S, cy: S) -> CoreResult<()> {
        if self.kind() != ManifoldKind::HyperbolicPoincare {
            return Err(CoreError::unsupported(
                "recenter_poincare is only supported on HyperbolicPoincare manifolds",
            ));
        }
        if !cx.is_finite() || !cy.is_finite() {
            return Err(CoreError::non_finite("ManifoldLayout2d::recenter_poincare"));
        }
        self.validate_finite()?;
        let (ccx, ccy) = clamp_poincare_point(cx, cy);
        let neg_cx = -ccx;
        let neg_cy = -ccy;
        let (xs, ys) = self.xy_mut();
        for i in 0..xs.len() {
            let (nx, ny) = poincare_mobius_add(neg_cx, neg_cy, xs[i], ys[i]);
            xs[i] = nx;
            ys[i] = ny;
        }
        Ok(())
    }

    /// Isometrically recenters the Poincaré disk layout so vertex `index` moves to `(0, 0)`.
    ///
    /// Returns [`CoreError::UnsupportedOperation`] on non-Poincaré manifolds.
    pub fn recenter_poincare_at(&mut self, index: usize) -> CoreResult<()> {
        if self.kind() != ManifoldKind::HyperbolicPoincare {
            return Err(CoreError::unsupported(
                "recenter_poincare_at is only supported on HyperbolicPoincare manifolds",
            ));
        }
        let n = self.len();
        if index >= n {
            return Err(CoreError::index_out_of_bounds(
                index,
                n,
                "ManifoldLayout2d::recenter_poincare_at",
            ));
        }
        let cx = self.x()[index];
        let cy = self.y()[index];
        self.recenter_poincare(cx, cy)
    }

    /// Isometrically recenters the Poincaré disk layout so its Einstein midpoint lies at `(0, 0)`.
    ///
    /// Returns [`CoreError::UnsupportedOperation`] on non-Poincaré manifolds.
    pub fn center_poincare_centroid(&mut self) -> CoreResult<()> {
        let (cx, cy) = self.poincare_centroid()?;
        self.recenter_poincare(cx, cy)
    }
}
