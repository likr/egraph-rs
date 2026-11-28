//! Coarse Louvain edge-length-first area reservation and non-SPD backbone initialization.
//!
//! Implements multi-level Louvain coarsening (`|V_L| <= max_coarse_nodes`), sublinear
//! cluster area reservation $r_C = c_0 |C|^{\alpha / 2}$, edge-length-first
//! meta-edge augmentation $\ell_{\mathrm{edge}} = \ell_{\mathrm{cond}} + \kappa (r_{C_1} + r_{C_2})$
//! before coarse geodesic solving (`RHGH` / `ChebyshevDoG` / `ShortestPath`), mass-weighted
//! stress + one-sided disk-barrier SGD (`RectangleOverlapSweep` + `Delaunay2d`), and
//! centroid rigid translation prolongation.

mod config;
mod geodesics;
mod overlap;
mod stress_barrier;
#[cfg(test)]
mod tests;

pub use config::{
    CoarseBackboneConfig, CoarseBackboneResult, CoarseBackboneStats, CoarseDistanceBackend,
    CoarseEdgeLengthMode,
};

use std::marker::PhantomData;

use egraph_clustering::{CoarsenedHierarchy, louvain_coarsen, project_clustered_layout_with_radii};
use egraph_core::{CoreError, CoreResult, CsrGraph, Layout2d, LayoutGenerator, Scalar};
use egraph_linalg_mds::robust_normalize_layout_2d;

use geodesics::{build_coarse_meta_edges, solve_coarse_mds_layout};
use stress_barrier::optimize_coarse_stress_and_barrier;

pub(super) fn extract_robust_transform<S: Scalar>(
    before: &Layout2d<S>,
    after: &Layout2d<S>,
) -> Option<(f64, f64, f64)> {
    let n = before.len();
    if n < 2 {
        return None;
    }
    let bx = before.x();
    let by = before.y();
    let ax = after.x();
    let ay = after.y();
    let bx0 = bx[0].to_f64();
    let by0 = by[0].to_f64();
    let ax0 = ax[0].to_f64();
    let ay0 = ay[0].to_f64();

    let mut best_dx = 0.0_f64;
    let mut best_pair_x = None;
    let mut best_dy = 0.0_f64;
    let mut best_pair_y = None;
    for i in 1..n {
        let dx = (bx[i].to_f64() - bx0).abs();
        if dx > best_dx {
            best_dx = dx;
            best_pair_x = Some(i);
        }
        let dy = (by[i].to_f64() - by0).abs();
        if dy > best_dy {
            best_dy = dy;
            best_pair_y = Some(i);
        }
    }
    if best_dx <= 1e-12 && best_dy <= 1e-12 {
        return None;
    }
    let sx_opt = if best_dx > 1e-12 {
        best_pair_x.and_then(|ix| {
            let sx = (ax[ix].to_f64() - ax0) / (bx[ix].to_f64() - bx0);
            (sx.is_finite() && sx > 0.0).then_some(sx)
        })
    } else {
        None
    };
    let sy_opt = if best_dy > 1e-12 {
        best_pair_y.and_then(|iy| {
            let sy = (ay[iy].to_f64() - ay0) / (by[iy].to_f64() - by0);
            (sy.is_finite() && sy > 0.0).then_some(sy)
        })
    } else {
        None
    };
    let scale = match (sx_opt, sy_opt) {
        (Some(sx), Some(sy)) => (best_dx * sx + best_dy * sy) / (best_dx + best_dy),
        (Some(sx), None) => sx,
        (None, Some(sy)) => sy,
        (None, None) => return None,
    };
    let cx = bx0 - ax0 / scale;
    let cy = by0 - ay0 / scale;
    if cx.is_finite() && cy.is_finite() {
        Some((cx, cy, scale))
    } else {
        None
    }
}

fn coarse_backbone_init_core<S: Scalar>(
    hierarchy: CoarsenedHierarchy,
    config: &CoarseBackboneConfig,
) -> CoreResult<CoarseBackboneResult<S>> {
    let coarsest = &hierarchy.coarsest;
    let n_fine = coarsest.node_to_coarse.len();
    let n_l = coarsest.num_clusters;
    let louvain_levels = hierarchy.levels.len();

    if n_fine == 0 || n_l == 0 {
        return Ok(CoarseBackboneResult {
            fine_layout: Layout2d::zeros(n_fine),
            coarse_layout: Layout2d::zeros(n_l),
            hierarchy,
            radii: Vec::new(),
            stats: CoarseBackboneStats {
                num_fine_nodes: n_fine,
                num_coarse_nodes: n_l,
                num_coarse_edges: 0,
                louvain_levels,
                mean_cluster_size: 0.0,
                max_cluster_size: 0,
                median_l_edge: 0.0,
                max_radius: 0.0,
                coarse_overlap_pairs_initial: 0,
                coarse_overlap_pairs_final: 0,
                scale_factor_applied: 1.0,
            },
        });
    }

    let mut radii = Vec::with_capacity(n_l);
    let mut max_cluster_size = 0usize;
    let mut max_radius = 0.0_f64;
    for &sz in &coarsest.coarse_sizes {
        let s = sz as usize;
        if s > max_cluster_size {
            max_cluster_size = s;
        }
        let r = config.c0 * (sz as f64).powf(0.5 * config.alpha);
        if r > max_radius {
            max_radius = r;
        }
        radii.push(r);
    }
    let mean_cluster_size = (n_fine as f64) / (n_l as f64);

    if n_l == 1 {
        let coarse_layout = Layout2d::<S>::zeros(1);
        let mut fine_layout = project_clustered_layout_with_radii(
            &coarse_layout,
            &coarsest.node_to_coarse,
            &radii,
            config.jitter_scale,
            config.seed,
        )?;
        let mut scale_factor_applied = 1.0_f64;
        if let Some(t_std) = config.target_std
            && n_fine > 1
        {
            let before = fine_layout.clone();
            robust_normalize_layout_2d(
                &mut fine_layout,
                t_std,
                config.robust_quantile,
                config.seed,
            )?;
            if let Some((_, _, sc)) = extract_robust_transform(&before, &fine_layout) {
                scale_factor_applied = sc;
            }
        }
        return Ok(CoarseBackboneResult {
            fine_layout,
            coarse_layout,
            hierarchy,
            radii,
            stats: CoarseBackboneStats {
                num_fine_nodes: n_fine,
                num_coarse_nodes: 1,
                num_coarse_edges: 0,
                louvain_levels,
                mean_cluster_size,
                max_cluster_size,
                median_l_edge: 0.0,
                max_radius,
                coarse_overlap_pairs_initial: 0,
                coarse_overlap_pairs_final: 0,
                scale_factor_applied,
            },
        });
    }

    let meta = build_coarse_meta_edges::<S>(&coarsest.graph, n_l, &radii, config)?;
    let (landmarks, mut coarse_layout) =
        solve_coarse_mds_layout::<S>(n_l, &meta, &radii, max_radius, config)?;
    let (coarse_overlap_pairs_initial, coarse_overlap_pairs_final) =
        optimize_coarse_stress_and_barrier(
            &coarsest.graph,
            &coarsest.coarse_sizes,
            n_l,
            &meta,
            &landmarks,
            &radii,
            config,
            &mut coarse_layout,
        )?;

    // Prolong coarse coordinates via radius-scaled jitter and consistent robust normalization.
    let mut fine_layout = project_clustered_layout_with_radii(
        &coarse_layout,
        &coarsest.node_to_coarse,
        &radii,
        config.jitter_scale,
        config.seed,
    )?;

    let mut scale_factor_applied = 1.0_f64;
    if let Some(t_std) = config.target_std
        && n_fine > 1
    {
        let before = fine_layout.clone();
        robust_normalize_layout_2d(&mut fine_layout, t_std, config.robust_quantile, config.seed)?;
        if let Some((cx, cy, sc)) = extract_robust_transform(&before, &fine_layout) {
            scale_factor_applied = sc;
            let (cxs, cys) = coarse_layout.xy_mut();
            for i in 0..n_l {
                cxs[i] = S::from_f64((cxs[i].to_f64() - cx) * sc);
                cys[i] = S::from_f64((cys[i].to_f64() - cy) * sc);
            }
        }
    }

    let m_l = meta.edge_u.len();
    let median_l_edge = meta.median_l_edge;

    Ok(CoarseBackboneResult {
        fine_layout,
        coarse_layout,
        hierarchy,
        radii,
        stats: CoarseBackboneStats {
            num_fine_nodes: n_fine,
            num_coarse_nodes: n_l,
            num_coarse_edges: m_l,
            louvain_levels,
            mean_cluster_size,
            max_cluster_size,
            median_l_edge,
            max_radius,
            coarse_overlap_pairs_initial,
            coarse_overlap_pairs_final,
            scale_factor_applied,
        },
    })
}

/// Coarse Louvain edge-length-first area reservation and non-SPD backbone generator implementing
/// [`LayoutGenerator`] with `Output = CoarseBackboneResult<S>`.
///
/// Contracts `graph` via [`louvain_coarsen`] to at most `config.louvain.max_coarse_nodes`
/// super-nodes, assigns sublinear radii $r_C = c_0 |C|^{\alpha/2}$, augments coarse meta-edge
/// lengths **first** via $\ell_{\mathrm{edge}} = \ell_{\mathrm{cond}} + \kappa (r_{C_1} + r_{C_2})$
/// before solving coarse `RHGH` / `ChebyshevDoG` / `ShortestPath` geodesics, optimizes
/// mass-weighted stress + one-sided disk-barrier hinges, and prolongs the coarse backbone to
/// `fine_layout` with consistent optional robust normalization.
#[non_exhaustive]
#[derive(Debug)]
pub struct CoarseBackbone<'a, S: Scalar = f64, W = f64> {
    graph: &'a CsrGraph<W>,
    config: CoarseBackboneConfig,
    _scalar: PhantomData<fn() -> S>,
}

impl<S: Scalar, W> Clone for CoarseBackbone<'_, S, W> {
    fn clone(&self) -> Self {
        Self {
            graph: self.graph,
            config: self.config.clone(),
            _scalar: PhantomData,
        }
    }
}

impl<'a, S: Scalar, W: Scalar> CoarseBackbone<'a, S, W> {
    /// Creates a [`CoarseBackbone`] generator with default [`CoarseBackboneConfig`] on `graph`.
    pub fn new(graph: &'a CsrGraph<W>) -> Self {
        Self {
            graph,
            config: CoarseBackboneConfig::default(),
            _scalar: PhantomData,
        }
    }

    /// Creates a [`CoarseBackbone`] generator on `graph` with an explicit [`CoarseBackboneConfig`].
    pub fn from_config(graph: &'a CsrGraph<W>, config: CoarseBackboneConfig) -> Self {
        Self {
            graph,
            config,
            _scalar: PhantomData,
        }
    }
}

impl<S: Scalar, W> CoarseBackbone<'_, S, W> {
    /// Replaces the configuration on this generator.
    pub fn with_config(mut self, config: CoarseBackboneConfig) -> Self {
        self.config = config;
        self
    }

    /// Sets the deterministic PRNG seed `seed`.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.config.seed = seed;
        self
    }

    /// Sets the optional target robust standard deviation `target_std`.
    pub fn with_target_std(mut self, target_std: Option<f64>) -> Self {
        self.config.target_std = target_std;
        self
    }

    /// Returns a reference to the current [`CoarseBackboneConfig`].
    pub fn config(&self) -> &CoarseBackboneConfig {
        &self.config
    }
}

impl<S: Scalar, W: Scalar> LayoutGenerator for CoarseBackbone<'_, S, W> {
    type Output = CoarseBackboneResult<S>;

    fn generate(&mut self) -> CoreResult<Self::Output> {
        self.config.validate()?;
        if self.graph.is_directed() {
            return Err(CoreError::invalid_input(
                "CoarseBackbone requires an undirected graph",
            ));
        }

        let hierarchy = louvain_coarsen(self.graph, &self.config.louvain)?;
        coarse_backbone_init_core::<S>(hierarchy, &self.config)
    }
}
