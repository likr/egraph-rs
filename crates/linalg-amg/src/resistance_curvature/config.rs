use egraph_core::{CoreError, CoreResult, CsrGraph, ManifoldKind};

use crate::amg_pcg::AmgPcgConfig;

use super::TORUS_DEG_CV_THRESHOLD;
use super::graph_ops::{build_canonical_graph_f32, build_canonical_graph_f64};

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

/// Configuration for [`compute_effective_resistances`](super::compute_effective_resistances).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EffectiveResistanceConfig {
    /// Number of Spielman-Srivastava Johnson-Lindenstrauss projection columns (`default: 128`).
    pub jl_dim: usize,
    /// Deterministic PRNG seed for JL Rademacher edge probes (`default: 42`).
    pub seed: u64,
    /// If `true`, input graph weights are interpreted as edge lengths $l_e$ ($w_e = 1/l_e$);
    /// if `false`, weights are interpreted directly as conductances $w_e$ ($l_e = 1/w_e$).
    pub weights_are_lengths: bool,
    /// Optional exact solver override: `Some(true)` forces dense $L^\dagger$ (allowed for $N \le 512$),
    /// `Some(false)` forces JL + `AmgPcgSolver`, and `None` uses exact when $N \le \mathrm{exact\_max\_nodes}$.
    pub exact: Option<bool>,
    /// Automatic exact-solver node threshold when `exact` is `None` (`default: 256`).
    pub exact_max_nodes: usize,
    /// Relative eigenvalue threshold for Moore-Penrose pseudoinverse $L^\dagger$ (`default: 1e-12`).
    pub eig_eps: f64,
    /// AMG-PCG configuration used by the JL path (`default: AmgPcgConfig::default()`).
    pub amg_config: AmgPcgConfig,
}

impl Default for EffectiveResistanceConfig {
    fn default() -> Self {
        Self {
            jl_dim: 128,
            seed: 42,
            weights_are_lengths: true,
            exact: None,
            exact_max_nodes: 256,
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

/// Configuration for [`resistance_curvature_flow`](super::resistance_curvature_flow).
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

/// Per-iteration summary recorded during [`resistance_curvature_flow`](super::resistance_curvature_flow).
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

/// Final output of Devriendt-Lambiotte [`resistance_curvature_flow`](super::resistance_curvature_flow).
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
    classify_geometry_from_curvature_with_sampling(
        num_edges,
        mean_link_curvature,
        std_link_curvature,
        leaf_fraction,
        deg_cv,
        hyperbolic_threshold,
        spherical_threshold,
        torus_std_threshold,
        true,
        1,
        false,
    )
}

/// Classifies a target 2D geometry from link resistance curvature, degree regularity, JL
/// sampling regime (`used_exact`, `jl_dim`), and closed-surface topological predicate.
///
/// When `used_exact` is `false`, `torus_std_threshold` is scaled smoothly by $O(1/\sqrt{k_{\mathrm{JL}}})$
/// so Johnson-Lindenstrauss resistance noise on regular flat tori ($N > 256$) does not trigger
/// false Euclidean fallback, while `deg_cv < 0.02` guards against irregular meshes.
#[allow(clippy::too_many_arguments)]
pub fn classify_geometry_from_curvature_with_sampling(
    num_edges: usize,
    mean_link_curvature: f64,
    std_link_curvature: f64,
    leaf_fraction: f64,
    deg_cv: f64,
    hyperbolic_threshold: f64,
    spherical_threshold: f64,
    torus_std_threshold: f64,
    used_exact: bool,
    jl_dim: usize,
    is_closed_spherical_surface: bool,
) -> SuggestedGeometry {
    if num_edges == 0 {
        return SuggestedGeometry::Euclidean;
    }
    if is_closed_spherical_surface {
        return SuggestedGeometry::Spherical;
    }
    if mean_link_curvature < hyperbolic_threshold {
        return SuggestedGeometry::HyperbolicPoincare;
    }
    if mean_link_curvature > spherical_threshold {
        return SuggestedGeometry::Spherical;
    }
    let effective_torus_std_threshold = if used_exact {
        torus_std_threshold
    } else {
        // JL variance scales as 1 / sqrt(jl_dim); covers 4-regular and 6-regular tori.
        let jl_scale = (64.0 / (jl_dim.max(1) as f64)).sqrt();
        torus_std_threshold * (1.0 + 16.0 * jl_scale)
    };
    if leaf_fraction == 0.0
        && deg_cv < TORUS_DEG_CV_THRESHOLD
        && std_link_curvature <= effective_torus_std_threshold
    {
        SuggestedGeometry::Torus2d
    } else {
        SuggestedGeometry::Euclidean
    }
}
