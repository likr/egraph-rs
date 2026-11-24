//! Coarse Louvain edge-length-first area reservation and non-SPD backbone initialization.
//!
//! Implements Stage 2B multi-level Louvain coarsening (`|V_L| <= max_coarse_nodes`), sublinear
//! cluster area reservation $r_C = c_0 |C|^{\alpha / 2}$, Pitfall #5 edge-length-first
//! meta-edge augmentation $\ell_{\mathrm{edge}} = \ell_{\mathrm{cond}} + \kappa (r_{C_1} + r_{C_2})$
//! before coarse geodesic solving (`RHGH` / `ChebyshevDoG` / `ShortestPath`), mass-weighted
//! stress + one-sided disk-barrier SGD (`RectangleOverlapSweep` + `Delaunay2d` / `StaticKdTree`
//! fallback), and Pitfall #7 centroid rigid translation prolongation.

use egraph_clustering::{
    CoarseLouvainConfig, CoarsenedHierarchy, louvain_coarsen, project_clustered_layout_with_radii,
};
use egraph_core::{
    CoreError, CoreResult, CsrGraph, DeterministicRng, IterativeSolver, Layout2d, Scalar,
    VisitStamp,
};
use egraph_linalg::{
    DijkstraWorkspace, EffectiveResistanceConfig, ResistanceCurvatureConfig,
    pivot_mds_from_distances, resistance_curvature_flow, robust_normalize_layout_2d,
    select_maxmin_sp_pivots,
};
use egraph_neighborhood::{
    DogVaradhanConfig, DogVaradhanLandmarks, RhghLandmarkConfig, compute_pivoted_rhgh_landmarks,
    compute_regularized_dog_varadhan_landmarks,
};
#[cfg(feature = "delaunay")]
use egraph_spatial::Delaunay2d;
use egraph_spatial::RectangleOverlapSweep;
#[cfg(not(feature = "delaunay"))]
use egraph_spatial::{KdQueryScratch, StaticKdTree};

use crate::sgd::{SgdScheduleConfig, SgdScheduleKind, SgdSolver};

#[cfg(not(feature = "delaunay"))]
const K_PROX_FALLBACK: usize = 8;

/// Base meta-edge length model on the coarsened super-graph before area augmentation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum CoarseEdgeLengthMode {
    /// Mean-normalized inverse conductance $\ell_{\mathrm{cond}} \propto 1 / W_{C_1 C_2}$.
    #[default]
    ConductanceInverse,
    /// Devriendt-Lambiotte effective-resistance curvature flow edge lengths
    /// (`weights_are_lengths = false` on `CoarsenedGraph::graph`).
    ResistanceCurvatureFlow,
}

/// Ideal distance solver used on the edge-length-augmented coarse super-graph $G_L$.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum CoarseDistanceBackend {
    /// Pure-Rust $O((h+m)|E_L|)$ pivoted Resolvent Heat-Geodesic Hybrid (`RHGH`) via `AmgPcgSolver`.
    #[default]
    Rhgh,
    /// Regularized Chebyshev Difference-of-Gaussians (`DoG`) Varadhan landmarks.
    ChebyshevDog,
    /// Exact weighted Dijkstra shortest-path landmarks on `coarse_length_graph`.
    ShortestPath,
}

/// Configuration for [`coarse_backbone_init`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CoarseBackboneConfig {
    /// Multi-level Louvain coarsening configuration (`max_coarse_nodes = 10_000`).
    pub louvain: CoarseLouvainConfig,
    /// Base meta-edge length model (`default: CoarseEdgeLengthMode::ConductanceInverse`).
    pub edge_length_mode: CoarseEdgeLengthMode,
    /// Coarse geodesic backend (`default: CoarseDistanceBackend::Rhgh`).
    pub distance_backend: CoarseDistanceBackend,
    /// Sublinear area-radius exponent in $r_C = c_0 |C|^{\alpha / 2}$ (`default: 0.75`).
    pub alpha: f64,
    /// Area-radius scale factor $c_0 \ge 0$ (`default: 0.5`; `0.0` disables area reservation).
    pub c0: f64,
    /// Edge-length-first augmentation multiplier $\kappa \ge 0$ in
    /// $\ell_{\mathrm{edge}} = \ell_{\mathrm{cond}} + \kappa (r_{C_1} + r_{C_2})$ (`default: 1.0`).
    pub kappa: f64,
    /// Meta-edge multiplicity weight exponent $\gamma_w \ge 0$ in $W_{C_1 C_2}^{\gamma_w}$ (`default: 0.5`).
    pub gamma_w: f64,
    /// Cluster mass exponent $\gamma_m \ge 0$ in $(|C_1| |C_2|)^{\gamma_m / 2}$ (`default: 0.5`).
    pub gamma_m: f64,
    /// One-sided disk-barrier hinge weight $\mu_{\mathrm{ov}} \ge 0$ (`default: 1.0`).
    pub mu_ov: f64,
    /// Minimum clearance margin $m_{\mathrm{gap}} \ge 0$ added to $r_{C_1} + r_{C_2}$ (`default: 0.1`).
    pub m_gap: f64,
    /// Maximum number of coarse geodesic pivots/landmarks (`default: 64`).
    pub coarse_pivots: usize,
    /// Sampled non-edge stress pairs per coarse node (`default: 16`).
    pub coarse_neg_per_node: usize,
    /// Total coarse stress + disk-barrier SGD iterations (`default: 80`).
    pub coarse_t_max: usize,
    /// Optional explicit $\gamma > 0$ override for coarse `RHGH` (`None` uses auto-$\gamma$).
    pub rhgh_gamma: Option<f64>,
    /// Optional explicit $p > 0$ override for coarse `RHGH` (`None` uses auto-$p$).
    pub rhgh_p: Option<f64>,
    /// Number of Devriendt-Lambiotte curvature-flow steps when `ResistanceCurvatureFlow` is active (`default: 2`).
    pub rcf_steps: usize,
    /// Step size $\eta \ge 0$ for `ResistanceCurvatureFlow` (`default: 0.2`).
    pub rcf_eta: f64,
    /// Per-cluster radius jitter multiplier used during prolongation (`default: 0.5`).
    pub jitter_scale: f64,
    /// Recommended centroid tether strength $\gamma_{\mathrm{tether}} \in [0, 1]$ (`default: 0.5`).
    pub gamma_tether: f64,
    /// Optional target robust standard deviation applied to the prolonged layout (`default: Some(1.0)`).
    pub target_std: Option<f64>,
    /// Radial quantile in `(0, 1)` used by `robust_normalize_layout_2d` (`default: 0.95`).
    pub robust_quantile: f64,
    /// Explicit deterministic PRNG seed (`default: 42`).
    pub seed: u64,
}

impl Default for CoarseBackboneConfig {
    fn default() -> Self {
        Self {
            louvain: CoarseLouvainConfig::default(),
            edge_length_mode: CoarseEdgeLengthMode::ConductanceInverse,
            distance_backend: CoarseDistanceBackend::Rhgh,
            alpha: 0.75,
            c0: 0.5,
            kappa: 1.0,
            gamma_w: 0.5,
            gamma_m: 0.5,
            mu_ov: 1.0,
            m_gap: 0.1,
            coarse_pivots: 64,
            coarse_neg_per_node: 16,
            coarse_t_max: 80,
            rhgh_gamma: None,
            rhgh_p: None,
            rcf_steps: 2,
            rcf_eta: 0.2,
            jitter_scale: 0.5,
            gamma_tether: 0.5,
            target_std: Some(1.0),
            robust_quantile: 0.95,
            seed: 42,
        }
    }
}

impl CoarseBackboneConfig {
    /// Validates all coarse area-reservation and backbone hyperparameters.
    pub fn validate(&self) -> CoreResult<()> {
        self.louvain.validate()?;
        if !self.alpha.is_finite() {
            return Err(CoreError::non_finite("CoarseBackboneConfig::alpha"));
        }
        if !(0.0..=2.0).contains(&self.alpha) {
            return Err(CoreError::invalid_input(format!(
                "CoarseBackboneConfig::alpha must be in [0.0, 2.0], got {}",
                self.alpha
            )));
        }
        for (ctx, val) in [
            ("CoarseBackboneConfig::c0", self.c0),
            ("CoarseBackboneConfig::kappa", self.kappa),
            ("CoarseBackboneConfig::gamma_w", self.gamma_w),
            ("CoarseBackboneConfig::gamma_m", self.gamma_m),
            ("CoarseBackboneConfig::mu_ov", self.mu_ov),
            ("CoarseBackboneConfig::m_gap", self.m_gap),
            ("CoarseBackboneConfig::rcf_eta", self.rcf_eta),
            ("CoarseBackboneConfig::jitter_scale", self.jitter_scale),
        ] {
            if !val.is_finite() {
                return Err(CoreError::non_finite(ctx));
            }
            if val < 0.0 {
                return Err(CoreError::invalid_input(format!(
                    "{ctx} must be >= 0.0, got {val}"
                )));
            }
        }
        if !self.gamma_tether.is_finite() {
            return Err(CoreError::non_finite("CoarseBackboneConfig::gamma_tether"));
        }
        if !(0.0..=1.0).contains(&self.gamma_tether) {
            return Err(CoreError::invalid_input(format!(
                "CoarseBackboneConfig::gamma_tether must be in [0.0, 1.0], got {}",
                self.gamma_tether
            )));
        }
        if self.coarse_pivots == 0 {
            return Err(CoreError::invalid_input(
                "CoarseBackboneConfig::coarse_pivots must be >= 1",
            ));
        }
        if let Some(g) = self.rhgh_gamma {
            if !g.is_finite() {
                return Err(CoreError::non_finite("CoarseBackboneConfig::rhgh_gamma"));
            }
            if g <= 0.0 {
                return Err(CoreError::invalid_input(format!(
                    "CoarseBackboneConfig::rhgh_gamma must be > 0.0, got {g}"
                )));
            }
        }
        if let Some(p) = self.rhgh_p {
            if !p.is_finite() {
                return Err(CoreError::non_finite("CoarseBackboneConfig::rhgh_p"));
            }
            if p <= 0.0 {
                return Err(CoreError::invalid_input(format!(
                    "CoarseBackboneConfig::rhgh_p must be > 0.0, got {p}"
                )));
            }
        }
        if let Some(ts) = self.target_std {
            if !ts.is_finite() {
                return Err(CoreError::non_finite("CoarseBackboneConfig::target_std"));
            }
            if ts <= 0.0 {
                return Err(CoreError::invalid_input(format!(
                    "CoarseBackboneConfig::target_std must be > 0.0, got {ts}"
                )));
            }
        }
        if !self.robust_quantile.is_finite() {
            return Err(CoreError::non_finite(
                "CoarseBackboneConfig::robust_quantile",
            ));
        }
        if self.robust_quantile <= 0.0 || self.robust_quantile >= 1.0 {
            return Err(CoreError::invalid_input(format!(
                "CoarseBackboneConfig::robust_quantile must be in (0.0, 1.0), got {}",
                self.robust_quantile
            )));
        }
        Ok(())
    }
}

/// Summary diagnostics emitted by [`coarse_backbone_init`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CoarseBackboneStats {
    /// Number of fine vertices $N$.
    pub num_fine_nodes: usize,
    /// Number of coarsest super-nodes $|V_L|$.
    pub num_coarse_nodes: usize,
    /// Number of canonical undirected meta-edges $|E_L|$.
    pub num_coarse_edges: usize,
    /// Number of outer Louvain contraction levels.
    pub louvain_levels: usize,
    /// Mean cluster size $|V| / |V_L|$.
    pub mean_cluster_size: f64,
    /// Maximum cluster size $\max_C |C|$.
    pub max_cluster_size: usize,
    /// Median augmented coarse edge length $\ell_{\mathrm{edge}}$ (`0.0` if edgeless).
    pub median_l_edge: f64,
    /// Maximum reserved cluster radius $\max_C r_C$.
    pub max_radius: f64,
    /// Number of coarse disk pairs violating $r_{C_1} + r_{C_2} + m_{\mathrm{gap}}$ before SGD.
    pub coarse_overlap_pairs_initial: usize,
    /// Number of coarse disk pairs violating $r_{C_1} + r_{C_2} + m_{\mathrm{gap}}$ after SGD.
    pub coarse_overlap_pairs_final: usize,
    /// Scale factor applied by `robust_normalize_layout_2d` (`1.0` if `target_std` is `None`).
    pub scale_factor_applied: f64,
}

/// Output of [`coarse_backbone_init`].
#[derive(Debug, Clone, PartialEq)]
pub struct CoarseBackboneResult<S: Scalar> {
    /// Prolonged (and optionally normalized) fine-level 2D initialization layout.
    pub fine_layout: Layout2d<S>,
    /// Coarse super-node 2D layout scaled consistently with `fine_layout`.
    pub coarse_layout: Layout2d<S>,
    /// Multi-level Louvain coarsening hierarchy (`levels` + `coarsest`).
    pub hierarchy: CoarsenedHierarchy,
    /// Unnormalized reserved cluster radii $r_C = c_0 |C|^{\alpha / 2}$ in coarse metric units.
    pub radii: Vec<f64>,
    /// Diagnostic metrics recorded during coarse backbone construction.
    pub stats: CoarseBackboneStats,
}

fn median_f64_slice(vals: &mut [f64]) -> f64 {
    if vals.is_empty() {
        return 0.0;
    }
    vals.sort_by(f64::total_cmp);
    let mid = vals.len() / 2;
    if vals.len() % 2 == 1 {
        vals[mid]
    } else {
        0.5 * (vals[mid - 1] + vals[mid])
    }
}

fn count_disk_overlaps<S: Scalar>(
    layout: &Layout2d<S>,
    radii: &[f64],
    m_gap: f64,
    half_boxes: &[S],
) -> CoreResult<usize> {
    let n = layout.len();
    if n < 2 || half_boxes.iter().all(|&h| h <= S::ZERO) {
        return Ok(0);
    }
    let boxes = RectangleOverlapSweep::find_layout_overlaps(layout, half_boxes, half_boxes)?;
    let xs = layout.x();
    let ys = layout.y();
    let mut cnt = 0usize;
    for (u, v) in boxes {
        let thr = radii[u] + radii[v] + m_gap;
        if thr > 0.0 {
            let d = (xs[u].to_f64() - xs[v].to_f64()).hypot(ys[u].to_f64() - ys[v].to_f64());
            if d < thr {
                cnt += 1;
            }
        }
    }
    Ok(cnt)
}

fn collect_disk_hinge_pairs<S: Scalar>(
    layout: &Layout2d<S>,
    radii: &[f64],
    m_gap: f64,
    half_boxes: &[S],
    #[cfg(not(feature = "delaunay"))] kd_scratch: &mut KdQueryScratch,
    #[cfg(not(feature = "delaunay"))] knn_out: &mut Vec<(usize, f64)>,
) -> CoreResult<Vec<(u32, u32)>> {
    let n = layout.len();
    if n < 2 {
        return Ok(Vec::new());
    }
    let mut pairs = Vec::new();
    if half_boxes.iter().any(|&h| h > S::ZERO) {
        let overlaps = RectangleOverlapSweep::find_layout_overlaps(layout, half_boxes, half_boxes)?;
        for (u, v) in overlaps {
            if radii[u] + radii[v] + m_gap > 0.0 {
                let (a, b) = if u < v { (u, v) } else { (v, u) };
                pairs.push((a as u32, b as u32));
            }
        }
    }

    #[cfg(feature = "delaunay")]
    {
        let mut dt_edges = Vec::new();
        Delaunay2d::triangulate_edges_into(layout, &mut dt_edges)?;
        for (u, v) in dt_edges {
            if radii[u as usize] + radii[v as usize] + m_gap > 0.0 {
                pairs.push((u, v));
            }
        }
    }

    #[cfg(not(feature = "delaunay"))]
    {
        let tree = StaticKdTree::<2, S>::from_layout(layout)?;
        let k = K_PROX_FALLBACK.min(n - 1);
        for u in 0..n {
            tree.k_nearest_excluding_self_with_scratch(u, k, kd_scratch, knn_out)?;
            for &(v, _) in knn_out.iter() {
                if radii[u] + radii[v] + m_gap > 0.0 {
                    let (a, b) = if u < v { (u, v) } else { (v, u) };
                    pairs.push((a as u32, b as u32));
                }
            }
        }
    }

    pairs.sort_unstable();
    pairs.dedup();
    Ok(pairs)
}

fn build_sp_landmarks<S: Scalar>(
    length_graph: &CsrGraph<S>,
    num_pivots: usize,
    seed: u64,
) -> CoreResult<DogVaradhanLandmarks> {
    let n = length_graph.num_nodes();
    if n == 0 {
        return Ok(DogVaradhanLandmarks {
            num_nodes: 0,
            landmarks: Vec::new(),
            d_dog: Vec::new(),
            d_coarse: None,
        });
    }
    let k_lm = num_pivots.min(n).max(1);
    let landmarks = select_maxmin_sp_pivots(length_graph, k_lm, seed, 1)?;
    let k_actual = landmarks.len();
    let mut d_table = vec![f64::INFINITY; n * k_actual];
    let mut row_col = vec![f64::INFINITY; n];
    let mut ws = DijkstraWorkspace::new(n);
    let mut max_finite = 0.0_f64;

    for (col, &lm) in landmarks.iter().enumerate() {
        ws.dijkstra_from(length_graph, lm as usize, &mut row_col)?;
        for i in 0..n {
            let d = row_col[i];
            d_table[i * k_actual + col] = d;
            if d.is_finite() && d > max_finite {
                max_finite = d;
            }
        }
    }

    let fallback = if max_finite > 0.0 {
        2.0 * max_finite
    } else {
        1.0
    };
    for val in &mut d_table {
        if !val.is_finite() {
            *val = fallback;
        }
    }

    Ok(DogVaradhanLandmarks {
        num_nodes: n,
        landmarks,
        d_dog: d_table,
        d_coarse: None,
    })
}

fn scale_landmarks_to_l_edge(
    landmarks: &mut DogVaradhanLandmarks,
    edge_u: &[u32],
    edge_v: &[u32],
    med_l_edge: f64,
) -> CoreResult<()> {
    if edge_u.is_empty() || med_l_edge <= 0.0 {
        return Ok(());
    }
    let mut edge_alts = landmarks.alt_pair_distances(edge_u, edge_v)?;
    let med_alt = median_f64_slice(&mut edge_alts);
    if med_alt.is_finite() && med_alt > 1e-12 {
        let scale = med_l_edge / med_alt;
        for val in &mut landmarks.d_dog {
            *val *= scale;
        }
        if let Some(dc) = &mut landmarks.d_coarse {
            for val in dc {
                *val *= scale;
            }
        }
    }
    Ok(())
}

fn extract_robust_transform<S: Scalar>(
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

    // Stage C: Extract canonical coarse meta-edges and compute base l_cond.
    let cg = &coarsest.graph;
    let mut edge_u = Vec::new();
    let mut edge_v = Vec::new();
    let mut edge_w = Vec::new();
    let w_opt = cg.weights();
    for u in 0..n_l {
        let start = cg.indptr()[u] as usize;
        let end = cg.indptr()[u + 1] as usize;
        for pos in start..end {
            let v = cg.indices()[pos] as usize;
            if v > u {
                let w = match w_opt {
                    Some(ws) => ws[pos],
                    None => 1.0,
                };
                edge_u.push(u as u32);
                edge_v.push(v as u32);
                edge_w.push(w);
            }
        }
    }
    let m_l = edge_u.len();

    let l_cond: Vec<f64> = if m_l == 0 {
        Vec::new()
    } else {
        match config.edge_length_mode {
            CoarseEdgeLengthMode::ConductanceInverse => {
                let mut raw: Vec<f64> = edge_w.iter().map(|&w| 1.0 / w.max(1e-12)).collect();
                let mean_raw = raw.iter().sum::<f64>() / (m_l as f64);
                let inv_mean = if mean_raw.is_finite() && mean_raw > 1e-12 {
                    1.0 / mean_raw
                } else {
                    1.0
                };
                for val in &mut raw {
                    *val *= inv_mean;
                }
                raw
            }
            CoarseEdgeLengthMode::ResistanceCurvatureFlow => {
                // CoarsenedGraph::graph stores aggregated conductances W_{C1, C2}, NOT lengths.
                // Filter non-positive edge weights safely so explicit 0.0 edges do not fail RCF.
                let mut pos_pairs = Vec::with_capacity(m_l);
                let mut pos_weights: Vec<S> = Vec::with_capacity(m_l);
                let mut pos_edge_indices = Vec::with_capacity(m_l);
                for e in 0..m_l {
                    let w = edge_w[e];
                    if w.is_finite() && w > 0.0 {
                        let ws = S::from_f64(w.min(1e30));
                        if ws.is_finite() && ws > S::ZERO {
                            pos_pairs.push((edge_u[e], edge_v[e]));
                            pos_weights.push(ws);
                            pos_edge_indices.push(e);
                        }
                    }
                }
                if pos_pairs.is_empty() {
                    vec![1.0; m_l]
                } else {
                    let cg_rcf = CsrGraph::<S>::from_edges(
                        n_l,
                        &pos_pairs,
                        Some(&pos_weights),
                        false,
                        false,
                    )?;
                    let rcf_cfg = ResistanceCurvatureConfig {
                        steps: config.rcf_steps,
                        eta: config.rcf_eta,
                        resistance: EffectiveResistanceConfig {
                            weights_are_lengths: false,
                            seed: config.seed,
                            ..EffectiveResistanceConfig::default()
                        },
                        ..ResistanceCurvatureConfig::default()
                    };
                    let rcf = resistance_curvature_flow(&cg_rcf, &rcf_cfg)?;
                    if pos_edge_indices.len() == m_l {
                        rcf.edge_lengths
                    } else {
                        let mut max_pos_len = 0.0_f64;
                        for &len in &rcf.edge_lengths {
                            if len.is_finite() && len > max_pos_len {
                                max_pos_len = len;
                            }
                        }
                        let zero_fallback = max_pos_len.max(1.0) * 1e6;
                        let mut out = vec![zero_fallback; m_l];
                        for (&orig_idx, len) in pos_edge_indices.iter().zip(rcf.edge_lengths) {
                            out[orig_idx] = len;
                        }
                        out
                    }
                }
            }
        }
    };

    // Stage D (Pitfall #5): Edge-Length-First meta-edge augmentation BEFORE coarse geodesics.
    let mut l_edge = Vec::with_capacity(m_l);
    let mut pairs = Vec::with_capacity(m_l);
    for e in 0..m_l {
        let u = edge_u[e];
        let v = edge_v[e];
        let len = (l_cond[e] + config.kappa * (radii[u as usize] + radii[v as usize])).max(1e-12);
        l_edge.push(len);
        pairs.push((u, v));
    }

    let mut l_edge_sorted = l_edge.clone();
    let median_l_edge = median_f64_slice(&mut l_edge_sorted);

    // Stage E: Coarse ideal distance landmark table and PivotMDS initialization.
    let k_lm = config.coarse_pivots.min(n_l).max(1);
    let mut landmarks = if m_l == 0 {
        DogVaradhanLandmarks {
            num_nodes: n_l,
            landmarks: (0..k_lm as u32).collect(),
            d_dog: vec![1.0 + 2.0 * max_radius; n_l * k_lm],
            d_coarse: None,
        }
    } else {
        match config.distance_backend {
            CoarseDistanceBackend::Rhgh => {
                let cond_edge_s: Vec<S> = l_edge
                    .iter()
                    .map(|&len| S::from_f64((1.0 / len).clamp(1e-30, 1e30)))
                    .collect();
                let coarse_conductance_graph =
                    CsrGraph::<S>::from_edges(n_l, &pairs, Some(&cond_edge_s), false, false)?;
                let rhgh_cfg = RhghLandmarkConfig {
                    gamma: config.rhgh_gamma,
                    p_exp: config.rhgh_p,
                    n_landmarks: k_lm,
                    min_degree: 1,
                    normalize_1hop: false,
                    seed: config.seed,
                    ..RhghLandmarkConfig::default()
                };
                let (mut lm, _) =
                    compute_pivoted_rhgh_landmarks(&coarse_conductance_graph, &rhgh_cfg)?;
                scale_landmarks_to_l_edge(&mut lm, &edge_u, &edge_v, median_l_edge)?;
                lm
            }
            CoarseDistanceBackend::ChebyshevDog => {
                let cond_edge_s: Vec<S> = l_edge
                    .iter()
                    .map(|&len| S::from_f64((1.0 / len).clamp(1e-30, 1e30)))
                    .collect();
                let coarse_conductance_graph =
                    CsrGraph::<S>::from_edges(n_l, &pairs, Some(&cond_edge_s), false, false)?;
                let dog_cfg = DogVaradhanConfig {
                    n_landmarks: k_lm,
                    min_degree: 1,
                    seed: config.seed,
                    ..DogVaradhanConfig::default()
                };
                let (mut lm, _) = compute_regularized_dog_varadhan_landmarks(
                    &coarse_conductance_graph,
                    &dog_cfg,
                )?;
                scale_landmarks_to_l_edge(&mut lm, &edge_u, &edge_v, median_l_edge)?;
                lm
            }
            CoarseDistanceBackend::ShortestPath => {
                let l_edge_s: Vec<S> = l_edge
                    .iter()
                    .map(|&len| S::from_f64(len.clamp(1e-12, 1e30)))
                    .collect();
                let coarse_length_graph =
                    CsrGraph::<S>::from_edges(n_l, &pairs, Some(&l_edge_s), false, false)?;
                build_sp_landmarks(&coarse_length_graph, k_lm, config.seed)?
            }
        }
    };

    if m_l == 0 {
        for (col, &lm_id) in landmarks.landmarks.iter().enumerate() {
            landmarks.d_dog[(lm_id as usize) * k_lm + col] = 0.0;
        }
    }

    let mut coarse_layout = pivot_mds_from_distances::<S>(
        landmarks.d_dog(),
        n_l,
        landmarks.num_landmarks(),
        None,
        config.seed,
    )?;

    // Ensure non-degenerate initial coordinates aligned with median_l_edge scale.
    let mut init_spread = 0.0_f64;
    if m_l > 0 {
        let mut edge_2d = Vec::with_capacity(m_l);
        let xs = coarse_layout.x();
        let ys = coarse_layout.y();
        for e in 0..m_l {
            let u = edge_u[e] as usize;
            let v = edge_v[e] as usize;
            edge_2d.push((xs[u].to_f64() - xs[v].to_f64()).hypot(ys[u].to_f64() - ys[v].to_f64()));
        }
        let med_2d = median_f64_slice(&mut edge_2d);
        if med_2d.is_finite() && med_2d > 1e-12 && median_l_edge > 0.0 {
            let sc = median_l_edge / med_2d;
            let (xs_m, ys_m) = coarse_layout.xy_mut();
            for i in 0..n_l {
                xs_m[i] = S::from_f64(xs_m[i].to_f64() * sc);
                ys_m[i] = S::from_f64(ys_m[i].to_f64() * sc);
            }
            init_spread = median_l_edge;
        }
    }
    if init_spread <= 0.0 {
        let target = (1.0 + 2.0 * max_radius).max(1.0);
        robust_normalize_layout_2d(
            &mut coarse_layout,
            target,
            config.robust_quantile,
            config.seed,
        )?;
    }

    // Stage F: Mass-weighted stress + one-sided disk-barrier SGD.
    let half_boxes: Vec<S> = radii
        .iter()
        .map(|&r| S::from_f64(r + 0.5 * config.m_gap))
        .collect();
    let coarse_overlap_pairs_initial =
        count_disk_overlaps(&coarse_layout, &radii, config.m_gap, &half_boxes)?;

    if config.coarse_t_max > 0 && (m_l > 0 || config.mu_ov > 0.0 || config.coarse_neg_per_node > 0)
    {
        let mut rng = DeterministicRng::from_seed_u64(config.seed);
        let mut spring_src = Vec::with_capacity(m_l + n_l * config.coarse_neg_per_node);
        let mut spring_tgt = Vec::with_capacity(m_l + n_l * config.coarse_neg_per_node);
        let mut spring_dist: Vec<S> = Vec::with_capacity(m_l + n_l * config.coarse_neg_per_node);
        let mut spring_wt: Vec<S> = Vec::with_capacity(m_l + n_l * config.coarse_neg_per_node);

        let edge_alts = if m_l > 0 {
            landmarks.alt_pair_distances(&edge_u, &edge_v)?
        } else {
            Vec::new()
        };
        for e in 0..m_l {
            let u = edge_u[e] as usize;
            let v = edge_v[e] as usize;
            let d = edge_alts[e].max(l_edge[e]).max(1e-6);
            let sz_u = coarsest.coarse_sizes[u] as f64;
            let sz_v = coarsest.coarse_sizes[v] as f64;
            let mass = edge_w[e].max(1e-12).powf(config.gamma_w)
                * (sz_u * sz_v).powf(0.5 * config.gamma_m);
            let w = mass / (d * d);
            let d_s = S::from_f64(d);
            let w_s = S::from_f64(w);
            if d_s.is_finite() && d_s > S::ZERO && w_s.is_finite() && w_s > S::ZERO {
                spring_src.push(edge_u[e]);
                spring_tgt.push(edge_v[e]);
                spring_dist.push(d_s);
                spring_wt.push(w_s);
            }
        }

        // Sample canonical non-edge coarse pairs.
        if config.coarse_neg_per_node > 0 && n_l > 2 {
            let mut neg_pairs: Vec<(u32, u32)> =
                Vec::with_capacity(n_l * config.coarse_neg_per_node);
            let mut stamp = VisitStamp::new(n_l);
            let cg_indptr = cg.indptr();
            let cg_indices = cg.indices();
            for u in 0..n_l {
                stamp.clear();
                stamp.mark(u);
                for &nbr in &cg_indices[cg_indptr[u] as usize..cg_indptr[u + 1] as usize] {
                    stamp.mark(nbr as usize);
                }
                let mut collected = 0usize;
                let max_trials = config.coarse_neg_per_node * 6;
                for _ in 0..max_trials {
                    if collected >= config.coarse_neg_per_node {
                        break;
                    }
                    let v = rng.gen_range_usize(n_l);
                    if !stamp.visit(v) {
                        continue;
                    }
                    let (a, b) = if u < v { (u, v) } else { (v, u) };
                    neg_pairs.push((a as u32, b as u32));
                    collected += 1;
                }
            }
            neg_pairs.sort_unstable();
            neg_pairs.dedup();
            if !neg_pairs.is_empty() {
                let neg_u: Vec<u32> = neg_pairs.iter().map(|&(a, _)| a).collect();
                let neg_v: Vec<u32> = neg_pairs.iter().map(|&(_, b)| b).collect();
                let neg_alts = landmarks.alt_pair_distances(&neg_u, &neg_v)?;
                let floor_d = (1e-3 * median_l_edge.max(1.0)).max(1e-6);
                for idx in 0..neg_pairs.len() {
                    let u = neg_u[idx] as usize;
                    let v = neg_v[idx] as usize;
                    let d = neg_alts[idx].max(floor_d);
                    let sz_u = coarsest.coarse_sizes[u] as f64;
                    let sz_v = coarsest.coarse_sizes[v] as f64;
                    let mass = (sz_u * sz_v).powf(0.5 * config.gamma_m);
                    let w = mass / (d * d);
                    let d_s = S::from_f64(d);
                    let w_s = S::from_f64(w);
                    if d_s.is_finite() && d_s > S::ZERO && w_s.is_finite() && w_s > S::ZERO {
                        spring_src.push(neg_u[idx]);
                        spring_tgt.push(neg_v[idx]);
                        spring_dist.push(d_s);
                        spring_wt.push(w_s);
                    }
                }
            }
        }

        let n_rounds = if config.mu_ov > 0.0 && config.coarse_t_max >= 4 {
            4usize
        } else {
            1usize
        };
        let base_iters = config.coarse_t_max / n_rounds;
        let rem_iters = config.coarse_t_max % n_rounds;

        #[cfg(not(feature = "delaunay"))]
        let mut kd_scratch = KdQueryScratch::with_capacity(K_PROX_FALLBACK);
        #[cfg(not(feature = "delaunay"))]
        let mut knn_out = Vec::new();

        for round in 0..n_rounds {
            let round_iters = base_iters + usize::from(round < rem_iters);
            if round_iters == 0 {
                continue;
            }
            let mut hinge_src = Vec::new();
            let mut hinge_tgt = Vec::new();
            let mut hinge_dist: Vec<S> = Vec::new();
            let mut hinge_wt: Vec<S> = Vec::new();

            if config.mu_ov > 0.0 {
                let h_pairs = collect_disk_hinge_pairs(
                    &coarse_layout,
                    &radii,
                    config.m_gap,
                    &half_boxes,
                    #[cfg(not(feature = "delaunay"))]
                    &mut kd_scratch,
                    #[cfg(not(feature = "delaunay"))]
                    &mut knn_out,
                )?;
                let mu_ov_s = S::from_f64(config.mu_ov);
                for (u, v) in h_pairs {
                    let d_bar = radii[u as usize] + radii[v as usize] + config.m_gap;
                    let d_bar_s = S::from_f64(d_bar);
                    if d_bar_s.is_finite() && d_bar_s > S::ZERO && mu_ov_s.is_finite() {
                        hinge_src.push(u);
                        hinge_tgt.push(v);
                        hinge_dist.push(d_bar_s);
                        hinge_wt.push(mu_ov_s);
                    }
                }
            }

            let step_scale = 1.0 - 0.7 * (round as f64) / (n_rounds.max(1) as f64);
            let sched = SgdScheduleConfig {
                t_max: round_iters,
                eps: 0.01,
                kind: SgdScheduleKind::Exponential,
                step_scale,
            };
            let mut solver = SgdSolver::<S>::from_pairs_with_hinges(
                n_l,
                spring_src.clone(),
                spring_tgt.clone(),
                spring_dist.clone(),
                spring_dist.clone(),
                spring_wt.clone(),
                spring_wt.clone(),
                hinge_src,
                hinge_tgt,
                hinge_dist.clone(),
                hinge_dist,
                hinge_wt.clone(),
                hinge_wt,
                sched,
                config.seed.wrapping_add(round as u64),
            )?;
            solver.run(&mut coarse_layout)?;
        }
    }

    let coarse_overlap_pairs_final =
        count_disk_overlaps(&coarse_layout, &radii, config.m_gap, &half_boxes)?;

    // Stage G: Prolongation via radius-scaled jitter + consistent robust normalization.
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

/// Runs coarse Louvain edge-length-first area reservation and non-SPD backbone initialization.
///
/// Contracts `graph` via [`louvain_coarsen`] to at most `config.louvain.max_coarse_nodes`
/// super-nodes, assigns sublinear radii $r_C = c_0 |C|^{\alpha/2}$, augments coarse meta-edge
/// lengths **first** via $\ell_{\mathrm{edge}} = \ell_{\mathrm{cond}} + \kappa (r_{C_1} + r_{C_2})$
/// before solving coarse `RHGH` / `ChebyshevDoG` / `ShortestPath` geodesics, optimizes
/// mass-weighted stress + one-sided disk-barrier hinges, and prolongs the coarse backbone to
/// `fine_layout` with consistent optional robust normalization.
pub fn coarse_backbone_init<W: Scalar, S: Scalar>(
    graph: &CsrGraph<W>,
    config: &CoarseBackboneConfig,
) -> CoreResult<CoarseBackboneResult<S>> {
    config.validate()?;
    if graph.is_directed() {
        return Err(CoreError::invalid_input(
            "coarse_backbone_init requires an undirected graph",
        ));
    }

    let hierarchy = louvain_coarsen(graph, &config.louvain)?;
    coarse_backbone_init_core::<S>(hierarchy, config)
}

#[cfg(test)]
mod tests {
    use super::*;
    use egraph_clustering::align_clustered_centroids;

    fn make_three_cluster_chain(big_size: usize) -> CsrGraph<f64> {
        // Cluster A: 6-clique {0..6}, Cluster B: big_size-clique {6..6+big_size},
        // Cluster C: 6-clique {6+big_size..12+big_size}, linked only by A-B and B-C bridges.
        let small = 6usize;
        let n = 2 * small + big_size;
        let mut edges = Vec::new();
        for i in 0..small {
            for j in (i + 1)..small {
                edges.push((i as u32, j as u32));
            }
        }
        for i in 0..big_size {
            for j in (i + 1)..big_size {
                edges.push(((small + i) as u32, (small + j) as u32));
            }
        }
        let c0 = small + big_size;
        for i in 0..small {
            for j in (i + 1)..small {
                edges.push(((c0 + i) as u32, (c0 + j) as u32));
            }
        }
        edges.push(((small - 1) as u32, small as u32));
        edges.push(((small + big_size - 1) as u32, c0 as u32));
        CsrGraph::from_edges(n, &edges, None, false, false).unwrap()
    }

    #[test]
    fn test_pitfall5_three_cluster_chain_no_tunneling_and_rcf_mode() {
        let g = make_three_cluster_chain(24);
        let mut cfg = CoarseBackboneConfig {
            c0: 0.8,
            kappa: 1.0,
            mu_ov: 2.0,
            m_gap: 0.1,
            target_std: None,
            distance_backend: CoarseDistanceBackend::ShortestPath,
            ..CoarseBackboneConfig::default()
        };
        cfg.louvain.max_coarse_nodes = 3;

        let res_sp = coarse_backbone_init::<f64, f64>(&g, &cfg).unwrap();
        assert_eq!(res_sp.stats.num_coarse_nodes, 3);
        assert_eq!(res_sp.stats.coarse_overlap_pairs_final, 0);

        // Identify clusters A, B, C (B is the large central cluster).
        let sizes = &res_sp.hierarchy.coarsest.coarse_sizes;
        let b_id = sizes.iter().enumerate().max_by_key(|&(_, s)| *s).unwrap().0;
        assert!(sizes[b_id] >= 20, "expected large cluster B, got {sizes:?}");
        let ac_ids: Vec<usize> = (0..3).filter(|&i| i != b_id).collect();
        let a_id = ac_ids[0];
        let c_id = ac_ids[1];
        let dx = res_sp.coarse_layout.x()[a_id] - res_sp.coarse_layout.x()[c_id];
        let dy = res_sp.coarse_layout.y()[a_id] - res_sp.coarse_layout.y()[c_id];
        let dist_ac = dx.hypot(dy);
        assert!(
            dist_ac >= 2.0 * res_sp.radii[b_id],
            "Pitfall #5 tunneling detected: dist(A,C)={dist_ac} < 2*r_B={}",
            2.0 * res_sp.radii[b_id]
        );

        // RHGH + ResistanceCurvatureFlow (weights_are_lengths=false) determinism check.
        let cfg_rcf = CoarseBackboneConfig {
            edge_length_mode: CoarseEdgeLengthMode::ResistanceCurvatureFlow,
            distance_backend: CoarseDistanceBackend::Rhgh,
            target_std: Some(1.0),
            ..cfg
        };
        let r1 = coarse_backbone_init::<f64, f64>(&g, &cfg_rcf).unwrap();
        let r2 = coarse_backbone_init::<f64, f64>(&g, &cfg_rcf).unwrap();
        assert_eq!(r1.fine_layout, r2.fine_layout);
        assert_eq!(r1.coarse_layout, r2.coarse_layout);
        assert_eq!(r1.stats.coarse_overlap_pairs_final, 0);

        // Centroid tether alignment keeps fine centroids aligned with coarse_layout after normalization.
        let mut tethered = r1.fine_layout.clone();
        align_clustered_centroids(
            &mut tethered,
            &r1.coarse_layout,
            &r1.hierarchy.coarsest.node_to_coarse,
            1.0,
        )
        .unwrap();
        assert!(tethered.is_all_finite());
    }

    #[test]
    fn test_ablation_zero_values_and_chebyshev_dog_and_edge_cases() {
        let g = make_three_cluster_chain(12);
        let ablation_cfg = CoarseBackboneConfig {
            distance_backend: CoarseDistanceBackend::ChebyshevDog,
            c0: 0.0,
            kappa: 0.0,
            gamma_w: 0.0,
            gamma_m: 0.0,
            mu_ov: 0.0,
            m_gap: 0.0,
            jitter_scale: 0.0,
            gamma_tether: 0.0,
            target_std: Some(1.0),
            ..CoarseBackboneConfig::default()
        };
        let res = coarse_backbone_init::<f64, f32>(&g, &ablation_cfg).unwrap();
        assert!(res.fine_layout.is_all_finite());
        assert_eq!(res.stats.max_radius, 0.0);

        // Empty, singleton, edgeless, and invalid config handling.
        let empty = CsrGraph::<f64>::from_edges(0, &[], None, false, false).unwrap();
        assert_eq!(
            coarse_backbone_init::<f64, f64>(&empty, &CoarseBackboneConfig::default())
                .unwrap()
                .fine_layout
                .len(),
            0
        );
        let single = CsrGraph::<f64>::from_edges(1, &[], None, false, false).unwrap();
        assert_eq!(
            coarse_backbone_init::<f64, f64>(&single, &CoarseBackboneConfig::default())
                .unwrap()
                .fine_layout
                .len(),
            1
        );
        let edgeless = CsrGraph::<f64>::from_edges(5, &[], None, false, false).unwrap();
        let re =
            coarse_backbone_init::<f64, f64>(&edgeless, &CoarseBackboneConfig::default()).unwrap();
        assert!(re.fine_layout.is_all_finite());

        let bad_cfg = CoarseBackboneConfig {
            alpha: -0.1,
            ..CoarseBackboneConfig::default()
        };
        assert!(coarse_backbone_init::<f64, f64>(&g, &bad_cfg).is_err());
    }

    #[test]
    fn test_collinear_two_cluster_consistent_normalization_f64_and_f32() {
        // Challenger reproducer: 20-node two-10-clique graph connected by a single bridge (9, 10).
        let mut edges = Vec::new();
        for i in 0..10u32 {
            for j in (i + 1)..10u32 {
                edges.push((i, j));
                edges.push((10 + i, 10 + j));
            }
        }
        edges.push((9, 10));
        let g64 = CsrGraph::<f64>::from_edges(20, &edges, None, false, false).unwrap();
        let g32 = CsrGraph::<f32>::from_edges(20, &edges, None, false, false).unwrap();

        let mut cfg = CoarseBackboneConfig {
            jitter_scale: 0.0,
            target_std: Some(1.0),
            seed: 42,
            ..CoarseBackboneConfig::default()
        };
        cfg.louvain.max_coarse_nodes = 2;

        // f64 check: scale_factor_applied != 1.0 and cluster mean == coarse_layout[0] within 1e-12.
        let r64 = coarse_backbone_init::<f64, f64>(&g64, &cfg).unwrap();
        assert_eq!(r64.stats.num_coarse_nodes, 2);
        assert!(
            (r64.stats.scale_factor_applied - 1.0).abs() > 1e-3,
            "expected non-unit scale_factor_applied, got {}",
            r64.stats.scale_factor_applied
        );
        for cid in 0..2u32 {
            let mut sx = 0.0_f64;
            let mut sy = 0.0_f64;
            let mut cnt = 0usize;
            for (i, &c) in r64.hierarchy.coarsest.node_to_coarse.iter().enumerate() {
                if c == cid {
                    sx += r64.fine_layout.x()[i];
                    sy += r64.fine_layout.y()[i];
                    cnt += 1;
                }
            }
            let mx = sx / (cnt as f64);
            let my = sy / (cnt as f64);
            let diff = (mx - r64.coarse_layout.x()[cid as usize])
                .hypot(my - r64.coarse_layout.y()[cid as usize]);
            assert!(
                diff < 1e-12,
                "f64 cluster {cid} centroid mismatch: diff={diff}"
            );
        }

        // f32 check: scale_factor_applied != 1.0 and cluster mean == coarse_layout[0] within 1e-5.
        let r32 = coarse_backbone_init::<f32, f32>(&g32, &cfg).unwrap();
        assert_eq!(r32.stats.num_coarse_nodes, 2);
        assert!(
            (r32.stats.scale_factor_applied - 1.0).abs() > 1e-3,
            "expected non-unit scale_factor_applied in f32, got {}",
            r32.stats.scale_factor_applied
        );
        for cid in 0..2u32 {
            let mut sx = 0.0_f64;
            let mut sy = 0.0_f64;
            let mut cnt = 0usize;
            for (i, &c) in r32.hierarchy.coarsest.node_to_coarse.iter().enumerate() {
                if c == cid {
                    sx += r32.fine_layout.x()[i] as f64;
                    sy += r32.fine_layout.y()[i] as f64;
                    cnt += 1;
                }
            }
            let mx = sx / (cnt as f64);
            let my = sy / (cnt as f64);
            let diff = (mx - r32.coarse_layout.x()[cid as usize] as f64)
                .hypot(my - r32.coarse_layout.y()[cid as usize] as f64);
            assert!(
                diff < 1e-5,
                "f32 cluster {cid} centroid mismatch: diff={diff}"
            );
        }

        // Direct vertical 1D collinear extract_robust_transform oracle.
        let mut vert_before =
            Layout2d::from_xy(vec![0.0_f64, 0.0, 0.0], vec![-3.0, 0.0, 3.0]).unwrap();
        let before_snap = vert_before.clone();
        robust_normalize_layout_2d(&mut vert_before, 1.0, 0.95, 42).unwrap();
        let (cx, cy, sc) = extract_robust_transform(&before_snap, &vert_before)
            .expect("vertical 1D layout should yield valid transform");
        assert!(cx.abs() < 1e-12 && cy.abs() < 1e-12);
        assert!((sc - 1.0).abs() > 1e-3);
    }

    #[test]
    fn test_zero_weight_coarse_edges_guarded_in_rcf_mode() {
        // Three 4-cliques: C0-C1 has positive edge weight 1.0; C1-C2 has explicit 0.0 edge weight.
        let mut edges = Vec::new();
        let mut weights = Vec::new();
        for c in 0..3u32 {
            let off = c * 4;
            for i in 0..4u32 {
                for j in (i + 1)..4u32 {
                    edges.push((off + i, off + j));
                    weights.push(1.0_f64);
                }
            }
        }
        edges.push((3, 4));
        weights.push(1.0_f64);
        edges.push((7, 8));
        weights.push(0.0_f64);

        let g_mixed =
            CsrGraph::<f64>::from_edges(12, &edges, Some(&weights), false, false).unwrap();
        let mut cfg = CoarseBackboneConfig {
            edge_length_mode: CoarseEdgeLengthMode::ResistanceCurvatureFlow,
            distance_backend: CoarseDistanceBackend::Rhgh,
            ..CoarseBackboneConfig::default()
        };
        cfg.louvain.max_coarse_nodes = 3;
        let res_mixed = coarse_backbone_init::<f64, f64>(&g_mixed, &cfg).unwrap();
        assert!(res_mixed.fine_layout.is_all_finite());
        assert!(res_mixed.coarse_layout.is_all_finite());

        // Also test f32 path with all-zero inter-cluster weights.
        let all_zero_weights = vec![0.0_f32; edges.len()];
        let g_zero32 =
            CsrGraph::<f32>::from_edges(12, &edges, Some(&all_zero_weights), false, false).unwrap();
        let res_zero = coarse_backbone_init::<f32, f32>(&g_zero32, &cfg).unwrap();
        assert!(res_zero.fine_layout.is_all_finite());
        assert!(res_zero.coarse_layout.is_all_finite());
    }
}
