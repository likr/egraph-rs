use egraph_clustering::{CoarseLouvainConfig, CoarsenedHierarchy};
use egraph_core::{CoreError, CoreResult, Layout2d, Scalar};

/// Base meta-edge length model on the coarsened super-graph before area augmentation.
#[non_exhaustive]
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
#[non_exhaustive]
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

/// Configuration for [`CoarseBackbone`](super::CoarseBackbone).
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
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
    /// Creates a new [`CoarseBackboneConfig`] initialized to [`Self::default`].
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the multi-level Louvain coarsening configuration `louvain`.
    #[inline]
    pub fn with_louvain(mut self, louvain: CoarseLouvainConfig) -> Self {
        self.louvain = louvain;
        self
    }

    /// Sets the base meta-edge length model `edge_length_mode`.
    #[inline]
    pub fn with_edge_length_mode(mut self, edge_length_mode: CoarseEdgeLengthMode) -> Self {
        self.edge_length_mode = edge_length_mode;
        self
    }

    /// Sets the coarse geodesic solver `distance_backend`.
    #[inline]
    pub fn with_distance_backend(mut self, distance_backend: CoarseDistanceBackend) -> Self {
        self.distance_backend = distance_backend;
        self
    }

    /// Sets the sublinear area-radius exponent `alpha`.
    #[inline]
    pub fn with_alpha(mut self, alpha: f64) -> Self {
        self.alpha = alpha;
        self
    }

    /// Sets the area-radius scale factor `c0`.
    #[inline]
    pub fn with_c0(mut self, c0: f64) -> Self {
        self.c0 = c0;
        self
    }

    /// Sets the edge-length-first augmentation multiplier `kappa`.
    #[inline]
    pub fn with_kappa(mut self, kappa: f64) -> Self {
        self.kappa = kappa;
        self
    }

    /// Sets the meta-edge multiplicity weight exponent `gamma_w`.
    #[inline]
    pub fn with_gamma_w(mut self, gamma_w: f64) -> Self {
        self.gamma_w = gamma_w;
        self
    }

    /// Sets the cluster mass exponent `gamma_m`.
    #[inline]
    pub fn with_gamma_m(mut self, gamma_m: f64) -> Self {
        self.gamma_m = gamma_m;
        self
    }

    /// Sets the one-sided disk-barrier hinge weight `mu_ov`.
    #[inline]
    pub fn with_mu_ov(mut self, mu_ov: f64) -> Self {
        self.mu_ov = mu_ov;
        self
    }

    /// Sets the minimum disk clearance margin `m_gap`.
    #[inline]
    pub fn with_m_gap(mut self, m_gap: f64) -> Self {
        self.m_gap = m_gap;
        self
    }

    /// Sets the maximum number of coarse geodesic pivots/landmarks `coarse_pivots`.
    #[inline]
    pub fn with_coarse_pivots(mut self, coarse_pivots: usize) -> Self {
        self.coarse_pivots = coarse_pivots;
        self
    }

    /// Sets the sampled non-edge stress pairs per coarse node `coarse_neg_per_node`.
    #[inline]
    pub fn with_coarse_neg_per_node(mut self, coarse_neg_per_node: usize) -> Self {
        self.coarse_neg_per_node = coarse_neg_per_node;
        self
    }

    /// Sets the total coarse stress + disk-barrier SGD iterations `coarse_t_max`.
    #[inline]
    pub fn with_coarse_t_max(mut self, coarse_t_max: usize) -> Self {
        self.coarse_t_max = coarse_t_max;
        self
    }

    /// Sets the optional explicit `RHGH` regularization `rhgh_gamma`.
    #[inline]
    pub fn with_rhgh_gamma(mut self, rhgh_gamma: Option<f64>) -> Self {
        self.rhgh_gamma = rhgh_gamma;
        self
    }

    /// Sets the optional explicit `RHGH` power exponent `rhgh_p`.
    #[inline]
    pub fn with_rhgh_p(mut self, rhgh_p: Option<f64>) -> Self {
        self.rhgh_p = rhgh_p;
        self
    }

    /// Sets the number of resistance-curvature flow steps `rcf_steps`.
    #[inline]
    pub fn with_rcf_steps(mut self, rcf_steps: usize) -> Self {
        self.rcf_steps = rcf_steps;
        self
    }

    /// Sets the resistance-curvature flow step size `rcf_eta`.
    #[inline]
    pub fn with_rcf_eta(mut self, rcf_eta: f64) -> Self {
        self.rcf_eta = rcf_eta;
        self
    }

    /// Sets the per-cluster radius jitter multiplier `jitter_scale`.
    #[inline]
    pub fn with_jitter_scale(mut self, jitter_scale: f64) -> Self {
        self.jitter_scale = jitter_scale;
        self
    }

    /// Sets the recommended centroid tether strength `gamma_tether`.
    #[inline]
    pub fn with_gamma_tether(mut self, gamma_tether: f64) -> Self {
        self.gamma_tether = gamma_tether;
        self
    }

    /// Sets the optional target robust standard deviation `target_std`.
    #[inline]
    pub fn with_target_std(mut self, target_std: Option<f64>) -> Self {
        self.target_std = target_std;
        self
    }

    /// Sets the radial quantile `robust_quantile` used during robust normalization.
    #[inline]
    pub fn with_robust_quantile(mut self, robust_quantile: f64) -> Self {
        self.robust_quantile = robust_quantile;
        self
    }

    /// Sets the deterministic PRNG seed `seed`.
    #[inline]
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

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

/// Summary diagnostics emitted by [`CoarseBackbone`](super::CoarseBackbone).
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
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

/// Output of [`CoarseBackbone`](super::CoarseBackbone).
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
pub struct CoarseBackboneResult<S: Scalar = f64> {
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
