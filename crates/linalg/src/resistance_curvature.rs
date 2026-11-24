//! Devriendt-Lambiotte effective resistance curvature and resistance curvature flow.
//!
//! Implements single-threaded, deterministic effective resistance estimation via
//! Spielman-Srivastava Johnson-Lindenstrauss projection solved with [`AmgPcgSolver`]
//! (or exact Moore-Penrose pseudoinverse via [`symmetric_eigen_jacobi`] on small graphs),
//! Foster's theorem diagnostics, vertex and link resistance curvatures, geometry regime
//! classification, and multi-step resistance curvature flow.

use std::collections::VecDeque;

use egraph_core::{
    CoreError, CoreResult, CsrGraph, DeterministicRng, ManifoldKind, Scalar, VisitStamp,
};

use crate::amg_pcg::{AmgPcgConfig, AmgPcgSolver};
use crate::csr_matrix::CsrMatrix;
use crate::eigen::symmetric_eigen_jacobi;

/// Hard upper bound on dense exact Jacobi eigendecomposition to avoid OOM.
const MAX_EXACT_NODES: usize = 512;
const EXACT_JACOBI_MAX_SWEEPS: usize = 120;
const EXACT_JACOBI_TOL: f64 = 1e-12;
const LINK_RESISTANCE_EPS: f64 = 1e-15;
const TORUS_DEG_CV_THRESHOLD: f64 = 0.15;

/// Target 2D geometry regime suggested by link resistance curvature and degree regularity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SuggestedGeometry {
    /// Negative-curvature regime matched to the 2D Poincare hyperbolic disk.
    HyperbolicPoincare,
    /// Positive-curvature regime matched to the 2D unit sphere.
    Spherical,
    /// Flat periodic regular regime matched to the 2D flat torus.
    Torus2d,
    /// Default flat Euclidean plane.
    Euclidean,
}

impl SuggestedGeometry {
    /// Returns the canonical lowercase identifier (`"hyperbolic"`, `"spherical"`, `"torus"`, `"euclidean"`).
    #[inline]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::HyperbolicPoincare => "hyperbolic",
            Self::Spherical => "spherical",
            Self::Torus2d => "torus",
            Self::Euclidean => "euclidean",
        }
    }

    /// Converts a non-Euclidean suggestion to [`ManifoldKind`], returning `None` for [`SuggestedGeometry::Euclidean`].
    #[inline]
    pub const fn to_manifold_kind(self) -> Option<ManifoldKind> {
        match self {
            Self::HyperbolicPoincare => Some(ManifoldKind::HyperbolicPoincare),
            Self::Spherical => Some(ManifoldKind::Spherical),
            Self::Torus2d => Some(ManifoldKind::Torus2d),
            Self::Euclidean => None,
        }
    }
}

/// Configuration for [`compute_effective_resistances`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EffectiveResistanceConfig {
    /// Number of Spielman-Srivastava Johnson-Lindenstrauss projection columns (`default: 64`).
    pub jl_dim: usize,
    /// Deterministic PRNG seed for JL Rademacher edge probes (`default: 42`).
    pub seed: u64,
    /// If `true`, input graph weights are interpreted as edge lengths $l_e$ ($w_e = 1/l_e$);
    /// if `false`, weights are interpreted directly as conductances $w_e$ ($l_e = 1/w_e$).
    pub weights_are_lengths: bool,
    /// Optional exact solver override: `Some(true)` forces dense $L^\dagger$ (allowed for $N \le 512$),
    /// `Some(false)` forces JL + `AmgPcgSolver`, and `None` uses exact when $N \le \mathrm{exact\_max\_nodes}$.
    pub exact: Option<bool>,
    /// Automatic exact-solver node threshold when `exact` is `None` (`default: 64`).
    pub exact_max_nodes: usize,
    /// Relative eigenvalue threshold for Moore-Penrose pseudoinverse $L^\dagger$ (`default: 1e-12`).
    pub eig_eps: f64,
    /// AMG-PCG configuration used by the JL path (`default: AmgPcgConfig::default()`).
    pub amg_config: AmgPcgConfig,
}

impl Default for EffectiveResistanceConfig {
    fn default() -> Self {
        Self {
            jl_dim: 64,
            seed: 42,
            weights_are_lengths: true,
            exact: None,
            exact_max_nodes: 64,
            eig_eps: 1e-12,
            amg_config: AmgPcgConfig::default(),
        }
    }
}

impl EffectiveResistanceConfig {
    /// Validates all effective-resistance parameters.
    pub fn validate(&self) -> CoreResult<()> {
        if self.jl_dim == 0 {
            return Err(CoreError::invalid_input("jl_dim must be >= 1"));
        }
        if !self.eig_eps.is_finite() || self.eig_eps < 0.0 {
            return Err(CoreError::invalid_input(format!(
                "eig_eps must be finite and >= 0.0, got {}",
                self.eig_eps
            )));
        }
        self.amg_config.validate()
    }
}

/// Effective resistance and Foster's theorem diagnostics for canonical simple undirected edges.
#[derive(Debug, Clone, PartialEq)]
pub struct EffectiveResistanceResult {
    /// Number of vertices in the graph.
    pub num_nodes: usize,
    /// Number of canonical undirected simple edges (`u < v`).
    pub num_edges: usize,
    /// Number of connected components (including isolated vertices).
    pub num_components: usize,
    /// Source endpoint `u < v` for each canonical edge.
    pub edge_u: Vec<u32>,
    /// Target endpoint `v > u` for each canonical edge.
    pub edge_v: Vec<u32>,
    /// Edge lengths $l_e = 1 / w_e$ for each canonical edge.
    pub edge_lengths: Vec<f64>,
    /// Edge conductances $w_e = 1 / l_e$ for each canonical edge.
    pub conductances: Vec<f64>,
    /// Effective resistance $R_e$ for each canonical edge.
    pub effective_resistances: Vec<f64>,
    /// Observed Foster sum $\sum_e w_e R_e$.
    pub foster_sum: f64,
    /// Exact theoretical Foster sum $N - c_{\mathrm{cc}}$.
    pub expected_foster: f64,
    /// Relative Foster error $|\mathrm{foster\_sum} - \mathrm{expected\_foster}| / \max(1, \mathrm{expected\_foster})$.
    pub foster_rel_err: f64,
    /// Whether the exact dense pseudoinverse path was used.
    pub used_exact: bool,
    /// Total PCG iterations across all JL RHS columns (`0` if exact or edgeless).
    pub amg_total_iterations: usize,
    /// Maximum relative PCG residual across all JL RHS columns (`0.0` if exact or edgeless).
    pub amg_max_relative_residual: f64,
    /// `true` if every JL PCG column converged (or exact / edgeless path was used).
    pub amg_all_converged: bool,
}

/// Configuration for [`resistance_curvature_flow`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ResistanceCurvatureConfig {
    /// Number of curvature-flow reweighting iterations (`default: 2`).
    pub steps: usize,
    /// Exponential step size $\eta \ge 0$ (`default: 0.2`).
    pub eta: f64,
    /// Symmetric clamp $\kappa_{\max} > 0$ on link curvatures during length updates (`default: 2.0`).
    pub kappa_max: f64,
    /// Positive floor enforced on edge lengths before mean normalization (`default: 1e-6`).
    pub min_edge_length: f64,
    /// Effective resistance solver configuration.
    pub resistance: EffectiveResistanceConfig,
    /// Threshold below which mean link curvature classifies as hyperbolic (`default: -0.05`).
    pub hyperbolic_threshold: f64,
    /// Threshold above which mean link curvature classifies as spherical (`default: 0.2`).
    pub spherical_threshold: f64,
    /// Maximum link-curvature standard deviation for flat regular graphs to classify as torus (`default: 0.05`).
    pub torus_std_threshold: f64,
}

impl Default for ResistanceCurvatureConfig {
    fn default() -> Self {
        Self {
            steps: 2,
            eta: 0.2,
            kappa_max: 2.0,
            min_edge_length: 1e-6,
            resistance: EffectiveResistanceConfig::default(),
            hyperbolic_threshold: -0.05,
            spherical_threshold: 0.2,
            torus_std_threshold: 0.05,
        }
    }
}

impl ResistanceCurvatureConfig {
    /// Validates all resistance curvature flow parameters.
    pub fn validate(&self) -> CoreResult<()> {
        if !self.eta.is_finite() || self.eta < 0.0 {
            return Err(CoreError::invalid_input(format!(
                "eta must be finite and >= 0.0, got {}",
                self.eta
            )));
        }
        if !self.kappa_max.is_finite() || self.kappa_max <= 0.0 {
            return Err(CoreError::invalid_input(format!(
                "kappa_max must be finite and > 0.0, got {}",
                self.kappa_max
            )));
        }
        if !self.min_edge_length.is_finite() || self.min_edge_length <= 0.0 {
            return Err(CoreError::invalid_input(format!(
                "min_edge_length must be finite and > 0.0, got {}",
                self.min_edge_length
            )));
        }
        if !self.hyperbolic_threshold.is_finite() {
            return Err(CoreError::non_finite(
                "ResistanceCurvatureConfig::hyperbolic_threshold",
            ));
        }
        if !self.spherical_threshold.is_finite() {
            return Err(CoreError::non_finite(
                "ResistanceCurvatureConfig::spherical_threshold",
            ));
        }
        if self.hyperbolic_threshold > self.spherical_threshold {
            return Err(CoreError::invalid_input(format!(
                "hyperbolic_threshold ({}) must be <= spherical_threshold ({})",
                self.hyperbolic_threshold, self.spherical_threshold
            )));
        }
        if !self.torus_std_threshold.is_finite() || self.torus_std_threshold < 0.0 {
            return Err(CoreError::invalid_input(format!(
                "torus_std_threshold must be finite and >= 0.0, got {}",
                self.torus_std_threshold
            )));
        }
        self.resistance.validate()
    }
}

/// Per-iteration summary recorded during [`resistance_curvature_flow`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ResistanceCurvatureStep {
    /// Flow step index (`0..steps`, plus final evaluation step `steps`).
    pub step: usize,
    /// Observed Foster sum $\sum_e w_e R_e$.
    pub foster_sum: f64,
    /// Theoretical Foster sum $N - c_{\mathrm{cc}}$.
    pub expected_foster: f64,
    /// Relative Foster error.
    pub foster_rel_err: f64,
    /// Mean link resistance curvature $\bar{\kappa}_{\mathrm{res}}$.
    pub mean_link_curvature: f64,
    /// Population standard deviation of link resistance curvatures.
    pub std_link_curvature: f64,
    /// Minimum link resistance curvature.
    pub min_link_curvature: f64,
    /// Maximum link resistance curvature.
    pub max_link_curvature: f64,
    /// Mean vertex resistance curvature $\bar{p}$.
    pub mean_vertex_curvature: f64,
    /// Minimum vertex resistance curvature.
    pub min_vertex_curvature: f64,
    /// Maximum vertex resistance curvature.
    pub max_vertex_curvature: f64,
    /// Whether this step used the exact dense pseudoinverse path.
    pub used_exact: bool,
    /// Total AMG-PCG iterations across all JL columns in this step.
    pub amg_total_iterations: usize,
    /// Maximum relative residual across all JL columns in this step.
    pub amg_max_relative_residual: f64,
}

/// Final output of Devriendt-Lambiotte [`resistance_curvature_flow`].
#[derive(Debug, Clone, PartialEq)]
pub struct ResistanceCurvatureFlowResult {
    /// Number of vertices in the graph.
    pub num_nodes: usize,
    /// Number of canonical undirected simple edges (`u < v`).
    pub num_edges: usize,
    /// Number of connected components (including isolated vertices).
    pub num_components: usize,
    /// Source endpoint `u < v` for each canonical edge.
    pub edge_u: Vec<u32>,
    /// Target endpoint `v > u` for each canonical edge.
    pub edge_v: Vec<u32>,
    /// Final normalized edge lengths $l_e^*$ (mean `1.0` when `num_edges > 0`).
    pub edge_lengths: Vec<f64>,
    /// Final edge conductances $w_e^* = 1 / l_e^*$.
    pub conductances: Vec<f64>,
    /// Effective resistances $R_e^*$ evaluated at the final edge conductances $w_e^*$.
    pub effective_resistances: Vec<f64>,
    /// Vertex resistance curvatures $p_i = 1 - \frac{1}{2} \sum_{j \sim i} w_{ij} R_{ij}$ at the final state.
    pub vertex_curvatures: Vec<f64>,
    /// Link resistance curvatures $\kappa_{\mathrm{res}}(u,v) = 2(p_u + p_v)/R_{uv}$ at the final state.
    pub link_curvatures: Vec<f64>,
    /// Final Foster sum $\sum_e w_e^* R_e^*$.
    pub foster_sum: f64,
    /// Exact theoretical Foster sum $N - c_{\mathrm{cc}}$.
    pub expected_foster: f64,
    /// Final relative Foster error.
    pub foster_rel_err: f64,
    /// Mean final link curvature.
    pub mean_link_curvature: f64,
    /// Population standard deviation of final link curvatures.
    pub std_link_curvature: f64,
    /// Minimum final link curvature.
    pub min_link_curvature: f64,
    /// Maximum final link curvature.
    pub max_link_curvature: f64,
    /// Mean final vertex curvature.
    pub mean_vertex_curvature: f64,
    /// Minimum final vertex curvature.
    pub min_vertex_curvature: f64,
    /// Maximum final vertex curvature.
    pub max_vertex_curvature: f64,
    /// Whether the final evaluation used the exact dense pseudoinverse path.
    pub used_exact: bool,
    /// Total AMG-PCG iterations in the final evaluation.
    pub amg_total_iterations: usize,
    /// Maximum relative PCG residual in the final evaluation.
    pub amg_max_relative_residual: f64,
    /// `true` if every JL PCG column converged in the final evaluation.
    pub amg_all_converged: bool,
    /// Geometry regime suggested from the final curvature and degree profile.
    pub suggested_geometry: SuggestedGeometry,
    /// Diagnostics recorded at each flow step plus the final evaluation.
    pub step_history: Vec<ResistanceCurvatureStep>,
}

impl ResistanceCurvatureFlowResult {
    /// Converts [`Self::suggested_geometry`] to [`Option<ManifoldKind>`].
    #[inline]
    pub const fn suggested_manifold_kind(&self) -> Option<ManifoldKind> {
        self.suggested_geometry.to_manifold_kind()
    }

    /// Builds an undirected weighted `CsrGraph<f64>` storing the final edge lengths $l_e^*$.
    pub fn to_length_graph(&self) -> CoreResult<CsrGraph<f64>> {
        build_canonical_graph_f64(
            self.num_nodes,
            &self.edge_u,
            &self.edge_v,
            &self.edge_lengths,
        )
    }

    /// Builds an undirected weighted `CsrGraph<f64>` storing the final conductances $w_e^* = 1/l_e^*$.
    pub fn to_conductance_graph(&self) -> CoreResult<CsrGraph<f64>> {
        build_canonical_graph_f64(
            self.num_nodes,
            &self.edge_u,
            &self.edge_v,
            &self.conductances,
        )
    }

    /// Builds an undirected weighted `CsrGraph<f32>` storing the final edge lengths $l_e^*$
    /// (clamped to strictly positive finite `f32` values).
    pub fn to_length_graph_f32(&self) -> CoreResult<CsrGraph<f32>> {
        build_canonical_graph_f32(
            self.num_nodes,
            &self.edge_u,
            &self.edge_v,
            &self.edge_lengths,
        )
    }

    /// Builds an undirected weighted `CsrGraph<f32>` storing the final conductances $w_e^* = 1/l_e^*$
    /// (clamped to strictly positive finite `f32` values).
    pub fn to_conductance_graph_f32(&self) -> CoreResult<CsrGraph<f32>> {
        build_canonical_graph_f32(
            self.num_nodes,
            &self.edge_u,
            &self.edge_v,
            &self.conductances,
        )
    }
}

/// Classifies a target 2D geometry from link resistance curvature and vertex degree regularity.
#[allow(clippy::too_many_arguments)]
pub fn classify_geometry_from_curvature(
    num_edges: usize,
    mean_link_curvature: f64,
    std_link_curvature: f64,
    leaf_fraction: f64,
    deg_cv: f64,
    hyperbolic_threshold: f64,
    spherical_threshold: f64,
    torus_std_threshold: f64,
) -> SuggestedGeometry {
    if num_edges == 0 {
        return SuggestedGeometry::Euclidean;
    }
    if mean_link_curvature < hyperbolic_threshold {
        return SuggestedGeometry::HyperbolicPoincare;
    }
    if mean_link_curvature > spherical_threshold {
        return SuggestedGeometry::Spherical;
    }
    if leaf_fraction == 0.0
        && deg_cv < TORUS_DEG_CV_THRESHOLD
        && std_link_curvature <= torus_std_threshold
    {
        SuggestedGeometry::Torus2d
    } else {
        SuggestedGeometry::Euclidean
    }
}

/// Computes effective edge resistances $R_e$ and Foster's theorem diagnostics for `graph`.
pub fn compute_effective_resistances<W: Scalar>(
    graph: &CsrGraph<W>,
    config: &EffectiveResistanceConfig,
) -> CoreResult<EffectiveResistanceResult> {
    config.validate()?;
    let n = graph.num_nodes();
    let use_exact = resolve_exact_flag(n, config)?;
    let canonical = extract_canonical_edges(graph, config.weights_are_lengths)?;
    let num_components = count_components(n, &canonical.edge_u, &canonical.edge_v);

    let eval = evaluate_effective_resistances(
        n,
        num_components,
        &canonical.edge_u,
        &canonical.edge_v,
        &canonical.conductances,
        config,
        use_exact,
        config.seed,
    )?;

    Ok(EffectiveResistanceResult {
        num_nodes: n,
        num_edges: canonical.edge_u.len(),
        num_components,
        edge_u: canonical.edge_u,
        edge_v: canonical.edge_v,
        edge_lengths: canonical.edge_lengths,
        conductances: canonical.conductances,
        effective_resistances: eval.effective_resistances,
        foster_sum: eval.foster_sum,
        expected_foster: eval.expected_foster,
        foster_rel_err: eval.foster_rel_err,
        used_exact: eval.used_exact,
        amg_total_iterations: eval.amg_total_iterations,
        amg_max_relative_residual: eval.amg_max_relative_residual,
        amg_all_converged: eval.amg_all_converged,
    })
}

/// Runs Devriendt-Lambiotte resistance curvature flow on `graph` and classifies geometry.
pub fn resistance_curvature_flow<W: Scalar>(
    graph: &CsrGraph<W>,
    config: &ResistanceCurvatureConfig,
) -> CoreResult<ResistanceCurvatureFlowResult> {
    config.validate()?;
    let n = graph.num_nodes();
    let use_exact = resolve_exact_flag(n, &config.resistance)?;
    let canonical = extract_canonical_edges(graph, config.resistance.weights_are_lengths)?;
    let m = canonical.edge_u.len();
    let num_components = count_components(n, &canonical.edge_u, &canonical.edge_v);
    let (leaf_fraction, deg_cv) = degree_regularity_stats(n, &canonical.edge_u, &canonical.edge_v);

    let mut edge_lengths = canonical.edge_lengths;
    let mut conductances = canonical.conductances;
    let mut step_history = Vec::with_capacity(config.steps + 1);

    for step_idx in 0..config.steps {
        let step_seed = config.resistance.seed.wrapping_add(step_idx as u64);
        let eval = evaluate_effective_resistances(
            n,
            num_components,
            &canonical.edge_u,
            &canonical.edge_v,
            &conductances,
            &config.resistance,
            use_exact,
            step_seed,
        )?;
        let curv = compute_curvatures(
            n,
            &canonical.edge_u,
            &canonical.edge_v,
            &conductances,
            &eval.effective_resistances,
        );
        step_history.push(ResistanceCurvatureStep {
            step: step_idx,
            foster_sum: eval.foster_sum,
            expected_foster: eval.expected_foster,
            foster_rel_err: eval.foster_rel_err,
            mean_link_curvature: curv.mean_link,
            std_link_curvature: curv.std_link,
            min_link_curvature: curv.min_link,
            max_link_curvature: curv.max_link,
            mean_vertex_curvature: curv.mean_vertex,
            min_vertex_curvature: curv.min_vertex,
            max_vertex_curvature: curv.max_vertex,
            used_exact: eval.used_exact,
            amg_total_iterations: eval.amg_total_iterations,
            amg_max_relative_residual: eval.amg_max_relative_residual,
        });

        if m > 0 {
            let mut sum_len = 0.0_f64;
            for e in 0..m {
                let kappa_clamped =
                    curv.link_curvatures[e].clamp(-config.kappa_max, config.kappa_max);
                let raw_len = (edge_lengths[e] * (-config.eta * kappa_clamped).exp())
                    .max(config.min_edge_length);
                edge_lengths[e] = raw_len;
                sum_len += raw_len;
            }
            let mean_len = sum_len / (m as f64);
            if !mean_len.is_finite() || mean_len <= 0.0 {
                return Err(CoreError::invalid_input(
                    "resistance_curvature_flow produced non-positive mean edge length",
                ));
            }
            for e in 0..m {
                let norm_len = edge_lengths[e] / mean_len;
                let cond = 1.0 / norm_len;
                if !norm_len.is_finite() || norm_len <= 0.0 || !cond.is_finite() || cond <= 0.0 {
                    return Err(CoreError::invalid_input(
                        "resistance_curvature_flow edge length normalization overflow",
                    ));
                }
                edge_lengths[e] = norm_len;
                conductances[e] = cond;
            }
        }
    }

    let final_seed = config.resistance.seed.wrapping_add(config.steps as u64);
    let final_eval = evaluate_effective_resistances(
        n,
        num_components,
        &canonical.edge_u,
        &canonical.edge_v,
        &conductances,
        &config.resistance,
        use_exact,
        final_seed,
    )?;
    let final_curv = compute_curvatures(
        n,
        &canonical.edge_u,
        &canonical.edge_v,
        &conductances,
        &final_eval.effective_resistances,
    );
    step_history.push(ResistanceCurvatureStep {
        step: config.steps,
        foster_sum: final_eval.foster_sum,
        expected_foster: final_eval.expected_foster,
        foster_rel_err: final_eval.foster_rel_err,
        mean_link_curvature: final_curv.mean_link,
        std_link_curvature: final_curv.std_link,
        min_link_curvature: final_curv.min_link,
        max_link_curvature: final_curv.max_link,
        mean_vertex_curvature: final_curv.mean_vertex,
        min_vertex_curvature: final_curv.min_vertex,
        max_vertex_curvature: final_curv.max_vertex,
        used_exact: final_eval.used_exact,
        amg_total_iterations: final_eval.amg_total_iterations,
        amg_max_relative_residual: final_eval.amg_max_relative_residual,
    });

    let suggested_geometry = classify_geometry_from_curvature(
        m,
        final_curv.mean_link,
        final_curv.std_link,
        leaf_fraction,
        deg_cv,
        config.hyperbolic_threshold,
        config.spherical_threshold,
        config.torus_std_threshold,
    );

    Ok(ResistanceCurvatureFlowResult {
        num_nodes: n,
        num_edges: m,
        num_components,
        edge_u: canonical.edge_u,
        edge_v: canonical.edge_v,
        edge_lengths,
        conductances,
        effective_resistances: final_eval.effective_resistances,
        vertex_curvatures: final_curv.vertex_curvatures,
        link_curvatures: final_curv.link_curvatures,
        foster_sum: final_eval.foster_sum,
        expected_foster: final_eval.expected_foster,
        foster_rel_err: final_eval.foster_rel_err,
        mean_link_curvature: final_curv.mean_link,
        std_link_curvature: final_curv.std_link,
        min_link_curvature: final_curv.min_link,
        max_link_curvature: final_curv.max_link,
        mean_vertex_curvature: final_curv.mean_vertex,
        min_vertex_curvature: final_curv.min_vertex,
        max_vertex_curvature: final_curv.max_vertex,
        used_exact: final_eval.used_exact,
        amg_total_iterations: final_eval.amg_total_iterations,
        amg_max_relative_residual: final_eval.amg_max_relative_residual,
        amg_all_converged: final_eval.amg_all_converged,
        suggested_geometry,
        step_history,
    })
}

struct CanonicalEdges {
    edge_u: Vec<u32>,
    edge_v: Vec<u32>,
    edge_lengths: Vec<f64>,
    conductances: Vec<f64>,
}

struct ResistanceEval {
    effective_resistances: Vec<f64>,
    foster_sum: f64,
    expected_foster: f64,
    foster_rel_err: f64,
    used_exact: bool,
    amg_total_iterations: usize,
    amg_max_relative_residual: f64,
    amg_all_converged: bool,
}

struct CurvatureEval {
    vertex_curvatures: Vec<f64>,
    link_curvatures: Vec<f64>,
    mean_link: f64,
    std_link: f64,
    min_link: f64,
    max_link: f64,
    mean_vertex: f64,
    min_vertex: f64,
    max_vertex: f64,
}

fn resolve_exact_flag(num_nodes: usize, config: &EffectiveResistanceConfig) -> CoreResult<bool> {
    match config.exact {
        Some(true) => {
            if num_nodes > MAX_EXACT_NODES {
                return Err(CoreError::invalid_input(format!(
                    "exact=true is capped at {MAX_EXACT_NODES} nodes, got {num_nodes}"
                )));
            }
            Ok(true)
        }
        Some(false) => Ok(false),
        None => Ok(num_nodes <= config.exact_max_nodes),
    }
}

fn extract_canonical_edges<W: Scalar>(
    graph: &CsrGraph<W>,
    weights_are_lengths: bool,
) -> CoreResult<CanonicalEdges> {
    let n = graph.num_nodes();
    if let Some(ws) = graph.weights() {
        for &w in ws {
            let wf = w.to_f64();
            if !wf.is_finite() {
                return Err(CoreError::non_finite("CsrGraph::weights"));
            }
            if wf <= 0.0 {
                return Err(CoreError::invalid_input(format!(
                    "graph edge weights must be strictly positive, got {wf}"
                )));
            }
        }
    }

    if n == 0 || graph.nnz() == 0 {
        return Ok(CanonicalEdges {
            edge_u: Vec::new(),
            edge_v: Vec::new(),
            edge_lengths: Vec::new(),
            conductances: Vec::new(),
        });
    }

    let weights_opt = graph.weights();
    let mut raw_edges: Vec<(u32, u32, f64)> = Vec::with_capacity(graph.nnz());
    for u in 0..n {
        let start = graph.indptr()[u] as usize;
        let end = graph.indptr()[u + 1] as usize;
        let u32_u = u as u32;
        for pos in start..end {
            let v32 = graph.indices()[pos];
            let v = v32 as usize;
            if v == u || v >= n {
                continue;
            }
            let val = match weights_opt {
                Some(ws) => ws[pos].to_f64(),
                None => 1.0,
            };
            let (a, b) = if u32_u < v32 {
                (u32_u, v32)
            } else {
                (v32, u32_u)
            };
            raw_edges.push((a, b, val));
        }
    }

    raw_edges.sort_unstable_by(|x, y| x.0.cmp(&y.0).then_with(|| x.1.cmp(&y.1)));

    let mut edge_u = Vec::with_capacity(raw_edges.len());
    let mut edge_v = Vec::with_capacity(raw_edges.len());
    let mut edge_lengths = Vec::with_capacity(raw_edges.len());
    let mut conductances = Vec::with_capacity(raw_edges.len());

    let mut i = 0usize;
    while i < raw_edges.len() {
        let (u, v, mut val) = raw_edges[i];
        i += 1;
        while i < raw_edges.len() && raw_edges[i].0 == u && raw_edges[i].1 == v {
            val = val.max(raw_edges[i].2);
            i += 1;
        }
        let inv = 1.0 / val;
        if !inv.is_finite() || inv <= 0.0 {
            return Err(CoreError::invalid_input(format!(
                "edge weight {val} produces invalid reciprocal {inv}"
            )));
        }
        edge_u.push(u);
        edge_v.push(v);
        if weights_are_lengths {
            edge_lengths.push(val);
            conductances.push(inv);
        } else {
            conductances.push(val);
            edge_lengths.push(inv);
        }
    }

    Ok(CanonicalEdges {
        edge_u,
        edge_v,
        edge_lengths,
        conductances,
    })
}

fn count_components(num_nodes: usize, edge_u: &[u32], edge_v: &[u32]) -> usize {
    if num_nodes == 0 {
        return 0;
    }
    if edge_u.is_empty() {
        return num_nodes;
    }

    let mut deg = vec![0usize; num_nodes];
    for (&u, &v) in edge_u.iter().zip(edge_v.iter()) {
        deg[u as usize] += 1;
        deg[v as usize] += 1;
    }
    let mut indptr = Vec::with_capacity(num_nodes + 1);
    indptr.push(0usize);
    let mut acc = 0usize;
    for &d in &deg {
        acc += d;
        indptr.push(acc);
    }
    let mut cursor = indptr[..num_nodes].to_vec();
    let mut adj = vec![0u32; acc];
    for (&u, &v) in edge_u.iter().zip(edge_v.iter()) {
        let uu = u as usize;
        let vv = v as usize;
        adj[cursor[uu]] = v;
        cursor[uu] += 1;
        adj[cursor[vv]] = u;
        cursor[vv] += 1;
    }

    let mut visited = VisitStamp::new(num_nodes);
    let mut queue = VecDeque::new();
    let mut components = 0usize;
    for start in 0..num_nodes {
        if !visited.visit(start) {
            continue;
        }
        components += 1;
        queue.push_back(start);
        while let Some(u) = queue.pop_front() {
            for &v32 in &adj[indptr[u]..indptr[u + 1]] {
                let v = v32 as usize;
                if visited.visit(v) {
                    queue.push_back(v);
                }
            }
        }
    }
    components
}

fn degree_regularity_stats(num_nodes: usize, edge_u: &[u32], edge_v: &[u32]) -> (f64, f64) {
    if num_nodes == 0 {
        return (0.0, 0.0);
    }
    let mut deg = vec![0usize; num_nodes];
    for (&u, &v) in edge_u.iter().zip(edge_v.iter()) {
        deg[u as usize] += 1;
        deg[v as usize] += 1;
    }
    let mut leaf_count = 0usize;
    let mut sum_deg = 0.0_f64;
    for &d in &deg {
        if d == 1 {
            leaf_count += 1;
        }
        sum_deg += d as f64;
    }
    let leaf_fraction = (leaf_count as f64) / (num_nodes as f64);
    let mean_deg = sum_deg / (num_nodes as f64);
    if mean_deg <= 0.0 {
        return (leaf_fraction, 0.0);
    }
    let mut var = 0.0_f64;
    for &d in &deg {
        let diff = (d as f64) - mean_deg;
        var += diff * diff;
    }
    let std_deg = (var / (num_nodes as f64)).sqrt();
    (leaf_fraction, std_deg / mean_deg)
}

fn build_combinatorial_laplacian_from_edges(
    num_nodes: usize,
    edge_u: &[u32],
    edge_v: &[u32],
    conductances: &[f64],
) -> CoreResult<CsrMatrix> {
    if num_nodes == 0 {
        return CsrMatrix::from_csr(0, vec![0], Vec::new(), Vec::new());
    }

    let mut rows: Vec<Vec<(u32, f64)>> = vec![Vec::new(); num_nodes];
    let mut diag = vec![0.0_f64; num_nodes];
    for e in 0..edge_u.len() {
        let u = edge_u[e];
        let v = edge_v[e];
        let w = conductances[e];
        rows[u as usize].push((v, -w));
        rows[v as usize].push((u, -w));
        diag[u as usize] += w;
        diag[v as usize] += w;
    }

    let mut indptr = Vec::with_capacity(num_nodes + 1);
    indptr.push(0u32);
    let mut indices = Vec::with_capacity(edge_u.len() * 2 + num_nodes);
    let mut values = Vec::with_capacity(edge_u.len() * 2 + num_nodes);

    for (u, row) in rows.iter_mut().enumerate() {
        let u32_u = u as u32;
        let d = diag[u];
        if d > 0.0 {
            row.push((u32_u, d));
        }
        row.sort_unstable_by_key(|&(col, _)| col);
        for &(col, val) in row.iter() {
            indices.push(col);
            values.push(val);
        }
        indptr.push(indices.len() as u32);
    }

    CsrMatrix::from_csr(num_nodes, indptr, indices, values)
}

#[allow(clippy::too_many_arguments)]
fn evaluate_effective_resistances(
    num_nodes: usize,
    num_components: usize,
    edge_u: &[u32],
    edge_v: &[u32],
    conductances: &[f64],
    config: &EffectiveResistanceConfig,
    use_exact: bool,
    seed: u64,
) -> CoreResult<ResistanceEval> {
    let m = edge_u.len();
    let expected_foster = if num_nodes == 0 {
        0.0
    } else {
        (num_nodes.saturating_sub(num_components)) as f64
    };

    if num_nodes == 0 || m == 0 {
        return Ok(ResistanceEval {
            effective_resistances: Vec::new(),
            foster_sum: 0.0,
            expected_foster,
            foster_rel_err: 0.0,
            used_exact: use_exact,
            amg_total_iterations: 0,
            amg_max_relative_residual: 0.0,
            amg_all_converged: true,
        });
    }

    let l_mat = build_combinatorial_laplacian_from_edges(num_nodes, edge_u, edge_v, conductances)?;
    let mut effective_resistances = vec![0.0_f64; m];
    let mut amg_total_iterations = 0usize;
    let mut amg_max_relative_residual = 0.0_f64;
    let mut amg_all_converged = true;

    if use_exact {
        let dense = l_mat.to_dense()?;
        let (lambdas, vectors) =
            symmetric_eigen_jacobi(&dense, num_nodes, EXACT_JACOBI_MAX_SWEEPS, EXACT_JACOBI_TOL)?;
        let lambda_max = lambdas.iter().copied().fold(0.0_f64, f64::max);
        let cutoff = if lambda_max > 0.0 {
            config.eig_eps * lambda_max
        } else {
            0.0
        };
        let mut active_modes: Vec<(usize, f64)> = Vec::new();
        for (k, &lam) in lambdas.iter().enumerate() {
            if lambda_max > 0.0 && lam > cutoff {
                let inv = 1.0 / lam;
                if inv.is_finite() {
                    active_modes.push((k, inv));
                }
            }
        }
        for e in 0..m {
            let u_off = (edge_u[e] as usize) * num_nodes;
            let v_off = (edge_v[e] as usize) * num_nodes;
            let mut r_uv = 0.0_f64;
            for &(k, inv_l) in &active_modes {
                let diff = vectors[u_off + k] - vectors[v_off + k];
                r_uv += inv_l * diff * diff;
            }
            effective_resistances[e] = r_uv.max(0.0);
        }
    } else {
        let k_jl = config.jl_dim;
        let inv_sqrt_k = 1.0 / (k_jl as f64).sqrt();
        let total = num_nodes
            .checked_mul(k_jl)
            .ok_or_else(|| CoreError::invalid_input("JL projection matrix size overflow"))?;
        let mut b_rhs = vec![0.0_f64; total];
        let mut rng = DeterministicRng::from_seed_u64(seed);
        let mut bits = 0u64;
        let mut bit_rem = 0u32;

        for e in 0..m {
            let u_off = (edge_u[e] as usize) * k_jl;
            let v_off = (edge_v[e] as usize) * k_jl;
            let s = conductances[e].sqrt() * inv_sqrt_k;
            for c in 0..k_jl {
                if bit_rem == 0 {
                    bits = rng.next_u64();
                    bit_rem = 64;
                }
                let signed = if (bits & 1) == 0 { s } else { -s };
                bits >>= 1;
                bit_rem -= 1;
                b_rhs[u_off + c] += signed;
                b_rhs[v_off + c] -= signed;
            }
        }

        let solver = AmgPcgSolver::with_near_nullspace(&l_mat, None, config.amg_config)?;
        let mut ws = solver.new_workspace();
        let mut z_sol = vec![0.0_f64; total];
        let batch = solver.solve_batch(&b_rhs, &mut z_sol, k_jl, &mut ws)?;
        amg_total_iterations = batch.total_iterations;
        amg_max_relative_residual = batch.max_relative_residual;
        amg_all_converged = batch.all_converged;

        for e in 0..m {
            let u_row = &z_sol[(edge_u[e] as usize) * k_jl..((edge_u[e] as usize) + 1) * k_jl];
            let v_row = &z_sol[(edge_v[e] as usize) * k_jl..((edge_v[e] as usize) + 1) * k_jl];
            let mut r_uv = 0.0_f64;
            for c in 0..k_jl {
                let diff = u_row[c] - v_row[c];
                r_uv += diff * diff;
            }
            effective_resistances[e] = r_uv.max(0.0);
        }
    }

    let mut foster_sum = 0.0_f64;
    for e in 0..m {
        foster_sum += conductances[e] * effective_resistances[e];
    }
    let foster_rel_err = if expected_foster > 0.0 {
        (foster_sum - expected_foster).abs() / expected_foster
    } else {
        foster_sum.abs()
    };

    Ok(ResistanceEval {
        effective_resistances,
        foster_sum,
        expected_foster,
        foster_rel_err,
        used_exact: use_exact,
        amg_total_iterations,
        amg_max_relative_residual,
        amg_all_converged,
    })
}

fn compute_curvatures(
    num_nodes: usize,
    edge_u: &[u32],
    edge_v: &[u32],
    conductances: &[f64],
    effective_resistances: &[f64],
) -> CurvatureEval {
    let mut vertex_curvatures = vec![1.0_f64; num_nodes];
    let m = edge_u.len();
    for e in 0..m {
        let u = edge_u[e] as usize;
        let v = edge_v[e] as usize;
        let contrib = 0.5 * conductances[e] * effective_resistances[e];
        vertex_curvatures[u] -= contrib;
        vertex_curvatures[v] -= contrib;
    }

    let mut link_curvatures = Vec::with_capacity(m);
    for e in 0..m {
        let r_uv = effective_resistances[e];
        if r_uv > LINK_RESISTANCE_EPS {
            let u = edge_u[e] as usize;
            let v = edge_v[e] as usize;
            link_curvatures.push(2.0 * (vertex_curvatures[u] + vertex_curvatures[v]) / r_uv);
        } else {
            link_curvatures.push(0.0);
        }
    }

    let (mean_link, std_link, min_link, max_link) = summarize_slice(&link_curvatures);
    let (mean_vertex, _, min_vertex, max_vertex) = summarize_slice(&vertex_curvatures);

    CurvatureEval {
        vertex_curvatures,
        link_curvatures,
        mean_link,
        std_link,
        min_link,
        max_link,
        mean_vertex,
        min_vertex,
        max_vertex,
    }
}

fn summarize_slice(vals: &[f64]) -> (f64, f64, f64, f64) {
    if vals.is_empty() {
        return (0.0, 0.0, 0.0, 0.0);
    }
    let mut sum = 0.0_f64;
    let mut min_v = f64::INFINITY;
    let mut max_v = f64::NEG_INFINITY;
    for &v in vals {
        sum += v;
        if v < min_v {
            min_v = v;
        }
        if v > max_v {
            max_v = v;
        }
    }
    let mean = sum / (vals.len() as f64);
    let mut var = 0.0_f64;
    for &v in vals {
        let d = v - mean;
        var += d * d;
    }
    let std = (var / (vals.len() as f64)).sqrt();
    (mean, std, min_v, max_v)
}

fn build_canonical_graph_f64(
    num_nodes: usize,
    edge_u: &[u32],
    edge_v: &[u32],
    weights: &[f64],
) -> CoreResult<CsrGraph<f64>> {
    let edges: Vec<(u32, u32)> = edge_u.iter().copied().zip(edge_v.iter().copied()).collect();
    CsrGraph::<f64>::from_edges(num_nodes, &edges, Some(weights), false, false)
}

fn build_canonical_graph_f32(
    num_nodes: usize,
    edge_u: &[u32],
    edge_v: &[u32],
    weights: &[f64],
) -> CoreResult<CsrGraph<f32>> {
    let edges: Vec<(u32, u32)> = edge_u.iter().copied().zip(edge_v.iter().copied()).collect();
    let weights_f32: Vec<f32> = weights
        .iter()
        .map(|&v| {
            let vf = v as f32;
            if vf.is_infinite() {
                f32::MAX
            } else if vf <= 0.0 {
                f32::MIN_POSITIVE
            } else {
                vf
            }
        })
        .collect();
    CsrGraph::<f32>::from_edges(num_nodes, &edges, Some(&weights_f32), false, false)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_complete_graph(n: usize) -> CsrGraph<f64> {
        let mut edges = Vec::new();
        for u in 0..n {
            for v in (u + 1)..n {
                edges.push((u as u32, v as u32));
            }
        }
        CsrGraph::from_edges(n, &edges, None, false, false).unwrap()
    }

    fn make_cycle_graph(n: usize) -> CsrGraph<f64> {
        let edges: Vec<(u32, u32)> = (0..n).map(|i| (i as u32, ((i + 1) % n) as u32)).collect();
        CsrGraph::from_edges(n, &edges, None, false, false).unwrap()
    }

    fn make_torus_graph(rows: usize, cols: usize) -> CsrGraph<f64> {
        let mut edges = Vec::new();
        for r in 0..rows {
            for c in 0..cols {
                let u = (r * cols + c) as u32;
                let r_next = (r * cols + ((c + 1) % cols)) as u32;
                let d_next = (((r + 1) % rows) * cols + c) as u32;
                edges.push((u, r_next));
                edges.push((u, d_next));
            }
        }
        CsrGraph::from_edges(rows * cols, &edges, None, false, true).unwrap()
    }

    fn make_star_graph(n: usize) -> CsrGraph<f64> {
        let edges: Vec<(u32, u32)> = (1..n).map(|i| (0, i as u32)).collect();
        CsrGraph::from_edges(n, &edges, None, false, false).unwrap()
    }

    fn make_barbell_graph(clique_size: usize) -> CsrGraph<f64> {
        let mut edges = Vec::new();
        for u in 0..clique_size {
            for v in (u + 1)..clique_size {
                edges.push((u as u32, v as u32));
            }
        }
        let off = clique_size;
        for u in 0..clique_size {
            for v in (u + 1)..clique_size {
                edges.push(((off + u) as u32, (off + v) as u32));
            }
        }
        edges.push(((clique_size - 1) as u32, off as u32));
        CsrGraph::from_edges(2 * clique_size, &edges, None, false, false).unwrap()
    }

    #[test]
    fn test_exact_analytical_resistances_and_curvatures() {
        // Complete graph K_8: R_e = 2/n, p_i = 1/n, kappa = 2
        let k8 = make_complete_graph(8);
        let res_k8 = resistance_curvature_flow(
            &k8,
            &ResistanceCurvatureConfig {
                steps: 0,
                ..Default::default()
            },
        )
        .unwrap();
        assert!(res_k8.used_exact);
        assert_eq!(res_k8.num_components, 1);
        assert!(res_k8.foster_rel_err < 1e-12);
        for &r in &res_k8.effective_resistances {
            assert!((r - 0.25).abs() < 1e-11);
        }
        for &p in &res_k8.vertex_curvatures {
            assert!((p - 0.125).abs() < 1e-11);
        }
        for &k in &res_k8.link_curvatures {
            assert!((k - 2.0).abs() < 1e-10);
        }
        assert_eq!(res_k8.suggested_geometry, SuggestedGeometry::Spherical);
        assert_eq!(
            res_k8.suggested_manifold_kind(),
            Some(ManifoldKind::Spherical)
        );

        // Cycle C_10: R_e = (n-1)/n, p_i = 1/n, kappa = 4/(n-1)
        let c10 = make_cycle_graph(10);
        let res_c10 = resistance_curvature_flow(
            &c10,
            &ResistanceCurvatureConfig {
                steps: 0,
                ..Default::default()
            },
        )
        .unwrap();
        assert!(res_c10.used_exact);
        assert!(res_c10.foster_rel_err < 1e-12);
        for &r in &res_c10.effective_resistances {
            assert!((r - 0.9).abs() < 1e-11);
        }
        for &k in &res_c10.link_curvatures {
            assert!((k - 4.0 / 9.0).abs() < 1e-10);
        }

        // Torus T_{8x8}: N=64, uniform p_i = 1/64, kappa = 8/63, std = 0 => Torus2d
        let t8 = make_torus_graph(8, 8);
        let res_t8 = resistance_curvature_flow(
            &t8,
            &ResistanceCurvatureConfig {
                steps: 0,
                ..Default::default()
            },
        )
        .unwrap();
        assert!(res_t8.used_exact);
        assert!(res_t8.foster_rel_err < 1e-11);
        assert!((res_t8.mean_link_curvature - 8.0 / 63.0).abs() < 1e-9);
        assert!(res_t8.std_link_curvature < 1e-9);
        assert_eq!(res_t8.suggested_geometry, SuggestedGeometry::Torus2d);
        assert_eq!(
            res_t8.suggested_manifold_kind(),
            Some(ManifoldKind::Torus2d)
        );

        // Star K_{1,12}: R_e = 1.0, p_center = 1 - 12/2 = -5, p_leaf = 0.5, kappa = 2*(-4.5)/1 = -9
        let star = make_star_graph(13);
        let res_star = resistance_curvature_flow(
            &star,
            &ResistanceCurvatureConfig {
                steps: 0,
                ..Default::default()
            },
        )
        .unwrap();
        assert!(res_star.used_exact);
        assert!(res_star.foster_rel_err < 1e-12);
        for &r in &res_star.effective_resistances {
            assert!((r - 1.0).abs() < 1e-11);
        }
        assert!((res_star.mean_link_curvature - (-9.0)).abs() < 1e-10);
        assert_eq!(
            res_star.suggested_geometry,
            SuggestedGeometry::HyperbolicPoincare
        );
        assert_eq!(
            res_star.suggested_manifold_kind(),
            Some(ManifoldKind::HyperbolicPoincare)
        );

        // Weighted tree: R_e = l_e, w_e * R_e = 1
        let tree_edges = vec![(0, 1), (1, 2), (1, 3), (3, 4)];
        let tree_lengths = vec![2.0_f64, 0.5, 4.0, 1.5];
        let tree = CsrGraph::from_edges(5, &tree_edges, Some(&tree_lengths), false, false).unwrap();
        let er_tree = compute_effective_resistances(
            &tree,
            &EffectiveResistanceConfig {
                weights_are_lengths: true,
                exact: Some(true),
                ..Default::default()
            },
        )
        .unwrap();
        assert!(er_tree.foster_rel_err < 1e-12);
        for (e, &r) in er_tree.effective_resistances.iter().enumerate() {
            assert!((r - er_tree.edge_lengths[e]).abs() < 1e-11);
        }
    }

    #[test]
    fn test_jl_amg_pcg_foster_and_multicc_isolated() {
        // Connected 10x10 grid (N=100) forced through JL + AmgPcgSolver
        let mut grid_edges = Vec::new();
        for r in 0..10u32 {
            for c in 0..10u32 {
                let u = r * 10 + c;
                if c + 1 < 10 {
                    grid_edges.push((u, u + 1));
                }
                if r + 1 < 10 {
                    grid_edges.push((u, u + 10));
                }
            }
        }
        let g100 = CsrGraph::<f64>::from_edges(100, &grid_edges, None, false, false).unwrap();
        let cfg = EffectiveResistanceConfig {
            jl_dim: 64,
            seed: 42,
            exact: Some(false),
            ..Default::default()
        };
        let r1 = compute_effective_resistances(&g100, &cfg).unwrap();
        let r2 = compute_effective_resistances(&g100, &cfg).unwrap();
        assert!(!r1.used_exact);
        assert!(r1.amg_all_converged);
        assert!(r1.amg_total_iterations > 0);
        assert_eq!(r1.num_components, 1);
        assert!((r1.expected_foster - 99.0).abs() < 1e-12);
        assert!(
            r1.foster_rel_err < 0.05,
            "Foster relative error too large: {}",
            r1.foster_rel_err
        );
        assert_eq!(r1.effective_resistances, r2.effective_resistances);

        // Multi-component + isolated nodes + weighted conductances
        let mut multi_edges = Vec::new();
        let mut multi_weights = Vec::new();
        // Component 1: 6x6 grid (0..36)
        for r in 0..6u32 {
            for c in 0..6u32 {
                let u = r * 6 + c;
                if c + 1 < 6 {
                    multi_edges.push((u, u + 1));
                    multi_weights.push(2.5);
                }
                if r + 1 < 6 {
                    multi_edges.push((u, u + 6));
                    multi_weights.push(1.25);
                }
            }
        }
        // Component 2: K_40 (36..76)
        for u in 36..76u32 {
            for v in (u + 1)..76u32 {
                multi_edges.push((u, v));
                multi_weights.push(0.5);
            }
        }
        // Isolated vertices: 76..80 (4 isolated nodes) => c_cc = 1 + 1 + 4 = 6
        let g_multi =
            CsrGraph::from_edges(80, &multi_edges, Some(&multi_weights), false, false).unwrap();
        let flow_multi = resistance_curvature_flow(
            &g_multi,
            &ResistanceCurvatureConfig {
                steps: 1,
                resistance: EffectiveResistanceConfig {
                    jl_dim: 96,
                    seed: 7,
                    weights_are_lengths: false,
                    exact: Some(false),
                    ..Default::default()
                },
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(flow_multi.num_components, 6);
        assert!((flow_multi.expected_foster - 74.0).abs() < 1e-12);
        assert!(flow_multi.amg_all_converged);
        assert!(flow_multi.foster_rel_err < 0.05);
        for &p in &flow_multi.vertex_curvatures[76..80] {
            assert!((p - 1.0).abs() < 1e-12);
        }
        let sum_p: f64 = flow_multi.vertex_curvatures.iter().sum();
        assert!((sum_p - (80.0 - flow_multi.foster_sum)).abs() < 1e-10);
    }

    #[test]
    fn test_barbell_curvature_flow_separates_bridge_and_cliques() {
        let barbell = make_barbell_graph(6);
        let flow = resistance_curvature_flow(
            &barbell,
            &ResistanceCurvatureConfig {
                steps: 3,
                eta: 0.2,
                kappa_max: 2.0,
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(flow.step_history.len(), 4);
        assert_eq!(flow.step_history[0].step, 0);
        assert_eq!(flow.step_history[3].step, 3);
        let mean_len = flow.edge_lengths.iter().sum::<f64>() / (flow.num_edges as f64);
        assert!((mean_len - 1.0).abs() < 1e-12);

        let mut bridge_len = 0.0_f64;
        let mut intra_lens = Vec::new();
        for e in 0..flow.num_edges {
            if flow.edge_u[e] == 5 && flow.edge_v[e] == 6 {
                bridge_len = flow.edge_lengths[e];
            } else {
                intra_lens.push(flow.edge_lengths[e]);
            }
        }
        assert!(
            bridge_len > 1.0,
            "Bridge edge should lengthen (>1.0), got {bridge_len}"
        );
        let mean_intra = intra_lens.iter().sum::<f64>() / (intra_lens.len() as f64);
        assert!(
            mean_intra < 1.0,
            "Intra-clique edges should shorten (<1.0), got {mean_intra}"
        );

        let lg = flow.to_length_graph().unwrap();
        let cg = flow.to_conductance_graph().unwrap();
        let lg32 = flow.to_length_graph_f32().unwrap();
        let cg32 = flow.to_conductance_graph_f32().unwrap();
        assert_eq!(lg.num_nodes(), barbell.num_nodes());
        assert_eq!(cg.num_edges(), barbell.num_edges());
        assert_eq!(lg32.num_edges(), barbell.num_edges());
        assert_eq!(cg32.num_edges(), barbell.num_edges());
        assert!(lg.is_weighted());
        assert!(cg.is_weighted());
    }

    #[test]
    fn test_adversarial_edge_cases_and_validation() {
        let g0 = CsrGraph::<f64>::from_edges(0, &[], None, false, false).unwrap();
        let r0 = resistance_curvature_flow(&g0, &ResistanceCurvatureConfig::default()).unwrap();
        assert_eq!(r0.num_nodes, 0);
        assert_eq!(r0.num_edges, 0);
        assert_eq!(r0.num_components, 0);
        assert_eq!(r0.foster_sum, 0.0);
        assert_eq!(r0.suggested_geometry, SuggestedGeometry::Euclidean);

        let g1 = CsrGraph::<f64>::from_edges(1, &[], None, false, false).unwrap();
        let r1 = resistance_curvature_flow(&g1, &ResistanceCurvatureConfig::default()).unwrap();
        assert_eq!(r1.num_nodes, 1);
        assert_eq!(r1.num_components, 1);
        assert!((r1.vertex_curvatures[0] - 1.0).abs() < 1e-12);
        assert_eq!(r1.suggested_geometry, SuggestedGeometry::Euclidean);

        let g_edgeless = CsrGraph::<f64>::from_edges(5, &[], None, false, false).unwrap();
        let r_edgeless =
            resistance_curvature_flow(&g_edgeless, &ResistanceCurvatureConfig::default()).unwrap();
        assert_eq!(r_edgeless.num_components, 5);
        assert_eq!(r_edgeless.expected_foster, 0.0);
        assert_eq!(r_edgeless.foster_sum, 0.0);

        // Directed input graph with self-loop and parallel edge should sanitize cleanly
        let g_dir = CsrGraph::from_edges(
            3,
            &[(0, 0), (0, 1), (1, 0), (1, 2), (1, 2)],
            Some(&[1.0, 2.0, 1.0, 3.0, 4.0]),
            true,
            true,
        )
        .unwrap();
        let r_dir =
            compute_effective_resistances(&g_dir, &EffectiveResistanceConfig::default()).unwrap();
        assert_eq!(r_dir.num_edges, 2);
        assert!(r_dir.foster_rel_err < 1e-12);

        // Invalid weights: zero, negative (and NaN rejected by CsrGraph)
        for bad_w in [0.0, -1.0] {
            let g_bad = CsrGraph::from_edges(2, &[(0, 1)], Some(&[bad_w]), false, false).unwrap();
            assert!(
                compute_effective_resistances(&g_bad, &EffectiveResistanceConfig::default())
                    .is_err()
            );
        }
        assert!(CsrGraph::from_edges(2, &[(0, 1)], Some(&[f64::NAN]), false, false).is_err());

        // Invalid config parameters
        assert!(
            compute_effective_resistances(
                &g1,
                &EffectiveResistanceConfig {
                    jl_dim: 0,
                    ..Default::default()
                }
            )
            .is_err()
        );
        assert!(
            resistance_curvature_flow(
                &g1,
                &ResistanceCurvatureConfig {
                    eta: -0.1,
                    ..Default::default()
                }
            )
            .is_err()
        );
        assert!(
            resistance_curvature_flow(
                &g1,
                &ResistanceCurvatureConfig {
                    kappa_max: 0.0,
                    ..Default::default()
                }
            )
            .is_err()
        );
        assert!(
            resistance_curvature_flow(
                &g1,
                &ResistanceCurvatureConfig {
                    min_edge_length: 0.0,
                    ..Default::default()
                }
            )
            .is_err()
        );

        // exact = Some(true) cap at 512 nodes
        let g_513 = CsrGraph::<f64>::from_edges(513, &[(0, 1)], None, false, false).unwrap();
        assert!(
            compute_effective_resistances(
                &g_513,
                &EffectiveResistanceConfig {
                    exact: Some(true),
                    ..Default::default()
                }
            )
            .is_err()
        );
    }
}
