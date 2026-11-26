use egraph_core::{CoreError, CoreResult, Layout2d, Scalar};
use egraph_metrics_cluster::cluster_area::ClusterAreaConfig;

/// Configuration parameters for `eval-baselines` cluster separation, CVI, and Sprawlter targets.
#[cfg(feature = "eval-baselines")]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ClusterBaselineConfig {
    /// Neighborhood size `k` for `SepmeKnng` (must be `>= 1`; default `2`).
    pub sepme_knng_k: usize,
    /// Neighborhood size `k` for `NeighborhoodHit` (must be `>= 1`; default `7`).
    pub neighborhood_hit_k: usize,
    /// Grid resolution for `DistributionConsistency` (must be `>= 1`; default `20`).
    pub dc_grid_size: usize,
    /// Grid resolution for `ClassDensityMeasure` and `HistogramDensityMeasure` (must be `>= 1`; default `64`).
    pub cdm_grid_size: usize,
    /// Optional Gaussian KDE bandwidth for `ClassDensityMeasure` (`None` uses `1.5 / cdm_grid_size as f64`).
    pub cdm_bandwidth: Option<f64>,
    /// Histogram bin count for `PairwiseDistanceOverlapError` (must be `>= 1`; default `100`).
    pub num_bins: usize,
    /// Observable angle parameter `gamma` in `[0.0, 1.0]` for `SepmeGong` (default `0.35`).
    pub gong_gamma: f64,
    /// Number of $k$-means++ initializations for `CqKmeans*` (must be `>= 1`; default `5`).
    pub kmeans_n_init: usize,
    /// Maximum Lloyd iterations for `CqKmeans*` (must be `>= 1`; default `100`).
    pub kmeans_max_iters: usize,
    /// Deterministic RNG seed for `CqKmeans*` (default `42`).
    pub seed: u64,
    /// Uniform square half-extent radius for `SprawlterNn` via the dispatcher (default `1.0`).
    pub sprawlter_node_radius: f64,
}

#[cfg(feature = "eval-baselines")]
impl Default for ClusterBaselineConfig {
    fn default() -> Self {
        Self {
            sepme_knng_k: 2,
            neighborhood_hit_k: 7,
            dc_grid_size: 20,
            cdm_grid_size: 64,
            cdm_bandwidth: None,
            num_bins: 100,
            gong_gamma: 0.35,
            kmeans_n_init: 5,
            kmeans_max_iters: 100,
            seed: 42,
            sprawlter_node_radius: 1.0,
        }
    }
}

#[cfg(feature = "eval-baselines")]
impl ClusterBaselineConfig {
    /// Validates all cluster-baseline configuration parameters.
    pub fn validate(&self) -> CoreResult<()> {
        if self.sepme_knng_k == 0 {
            return Err(CoreError::invalid_input(
                "ClusterBaselineConfig::sepme_knng_k must be >= 1",
            ));
        }
        if self.neighborhood_hit_k == 0 {
            return Err(CoreError::invalid_input(
                "ClusterBaselineConfig::neighborhood_hit_k must be >= 1",
            ));
        }
        if self.dc_grid_size == 0 {
            return Err(CoreError::invalid_input(
                "ClusterBaselineConfig::dc_grid_size must be >= 1",
            ));
        }
        if self.cdm_grid_size == 0 {
            return Err(CoreError::invalid_input(
                "ClusterBaselineConfig::cdm_grid_size must be >= 1",
            ));
        }
        if let Some(bw) = self.cdm_bandwidth {
            if !bw.is_finite() {
                return Err(CoreError::non_finite(
                    "ClusterBaselineConfig::cdm_bandwidth",
                ));
            }
            if bw <= 0.0 {
                return Err(CoreError::invalid_input(
                    "ClusterBaselineConfig::cdm_bandwidth must be strictly positive",
                ));
            }
        }
        if self.num_bins == 0 {
            return Err(CoreError::invalid_input(
                "ClusterBaselineConfig::num_bins must be >= 1",
            ));
        }
        if !self.gong_gamma.is_finite() {
            return Err(CoreError::non_finite("ClusterBaselineConfig::gong_gamma"));
        }
        if !(0.0..=1.0).contains(&self.gong_gamma) {
            return Err(CoreError::invalid_input(
                "ClusterBaselineConfig::gong_gamma must be in [0.0, 1.0]",
            ));
        }
        if self.kmeans_n_init == 0 {
            return Err(CoreError::invalid_input(
                "ClusterBaselineConfig::kmeans_n_init must be >= 1",
            ));
        }
        if self.kmeans_max_iters == 0 {
            return Err(CoreError::invalid_input(
                "ClusterBaselineConfig::kmeans_max_iters must be >= 1",
            ));
        }
        if !self.sprawlter_node_radius.is_finite() {
            return Err(CoreError::non_finite(
                "ClusterBaselineConfig::sprawlter_node_radius",
            ));
        }
        if self.sprawlter_node_radius <= 0.0 {
            return Err(CoreError::invalid_input(
                "ClusterBaselineConfig::sprawlter_node_radius must be strictly positive",
            ));
        }
        Ok(())
    }
}

/// Configuration parameters for parameterized targets in [`super::evaluate_metrics`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MetricConfig {
    /// Number of deterministic degree-stratified BFS pivot sources for `SampledScaleNormalizedStress`.
    pub sns_num_pivots: usize,
    /// Explicit RNG seed for `SampledScaleNormalizedStress`.
    pub sns_seed: u64,
    /// Number of deterministic samples for `SampledCrossingNumber`, `SampledCrosslessness`, `SampledNp1`, and `SampledNp2`.
    pub crossing_num_samples: usize,
    /// Explicit RNG seed for `SampledCrossingNumber`, `SampledCrosslessness`, `SampledNp1`, and `SampledNp2`.
    pub crossing_seed: u64,
    /// Unit target edge-length scale factor in `IdealEdgeLengths` (uniform length on unweighted graphs, weight multiplier on weighted graphs).
    pub ideal_edge_length: f64,
    /// Number of Core-First MaxMin shortest-path landmarks for `LandmarkStratifiedRta`.
    pub rta_num_landmarks: usize,
    /// Total triplet sample budget across octave strata for `LandmarkStratifiedRta`.
    pub rta_num_triplets: usize,
    /// Minimum vertex degree preference for MaxMin landmark selection in `LandmarkStratifiedRta`.
    pub rta_min_degree: usize,
    /// Explicit RNG seed for `LandmarkStratifiedRta`.
    pub rta_seed: u64,
    /// Half-width patch radius (side `2 * r`) for `AdjustedPatchNig`.
    pub patch_nig_radius: usize,
    /// Equal-aspect canvas size in pixels for `AdjustedPatchNig`.
    pub patch_nig_canvas_size: usize,
    /// Equal-aspect viewport padding fraction in `[0.0, 0.5)` for `AdjustedPatchNig`.
    pub patch_nig_pad_frac: f64,
    /// Explicit RNG seed for the permutation null in `AdjustedPatchNig`.
    pub patch_nig_seed: u64,
    /// Configuration parameters for `ClusterAreaFaithfulness` (`G-CAF`).
    pub cluster_area: ClusterAreaConfig,
    /// Configuration parameters for `eval-baselines` cluster/separation/Sprawlter targets.
    #[cfg(feature = "eval-baselines")]
    pub cluster_baselines: ClusterBaselineConfig,
}

impl Default for MetricConfig {
    fn default() -> Self {
        Self {
            sns_num_pivots: 64,
            sns_seed: 42,
            crossing_num_samples: 4096,
            crossing_seed: 42,
            ideal_edge_length: 1.0,
            rta_num_landmarks: 64,
            rta_num_triplets: 1_000_000,
            rta_min_degree: 2,
            rta_seed: 42,
            patch_nig_radius: 8,
            patch_nig_canvas_size: 2048,
            patch_nig_pad_frac: 0.04,
            patch_nig_seed: 42,
            cluster_area: ClusterAreaConfig::default(),
            #[cfg(feature = "eval-baselines")]
            cluster_baselines: ClusterBaselineConfig::default(),
        }
    }
}

impl MetricConfig {
    /// Validates all configuration parameters.
    pub fn validate(&self) -> CoreResult<()> {
        if self.sns_num_pivots == 0 {
            return Err(CoreError::invalid_input("sns_num_pivots must be >= 1"));
        }
        if self.crossing_num_samples == 0 {
            return Err(CoreError::invalid_input(
                "crossing_num_samples must be >= 1",
            ));
        }
        if !self.ideal_edge_length.is_finite() {
            return Err(CoreError::non_finite("MetricConfig::ideal_edge_length"));
        }
        if self.ideal_edge_length <= 0.0 {
            return Err(CoreError::invalid_input(
                "ideal_edge_length must be strictly positive",
            ));
        }
        if self.rta_num_landmarks == 0 {
            return Err(CoreError::invalid_input("rta_num_landmarks must be >= 1"));
        }
        if self.rta_num_triplets == 0 {
            return Err(CoreError::invalid_input("rta_num_triplets must be >= 1"));
        }
        if self.patch_nig_radius == 0 {
            return Err(CoreError::invalid_input("patch_nig_radius must be >= 1"));
        }
        if self.patch_nig_canvas_size == 0 {
            return Err(CoreError::invalid_input(
                "patch_nig_canvas_size must be >= 1",
            ));
        }
        if !self.patch_nig_pad_frac.is_finite() {
            return Err(CoreError::non_finite("MetricConfig::patch_nig_pad_frac"));
        }
        if !(0.0..0.5).contains(&self.patch_nig_pad_frac) {
            return Err(CoreError::invalid_input(
                "patch_nig_pad_frac must be in [0.0, 0.5)",
            ));
        }
        self.cluster_area.validate()?;
        #[cfg(feature = "eval-baselines")]
        self.cluster_baselines.validate()?;
        Ok(())
    }
}

/// Optional auxiliary inputs passed to [`super::evaluate_metrics_with_inputs`].
#[derive(Debug, Clone, Copy)]
pub struct MetricInputs<'a, S: Scalar = f64> {
    /// Optional cluster/community label slice of length `N`.
    pub labels: Option<&'a [u32]>,
    /// Optional positive finite vertex weights of length `N` for `ClusterAreaFaithfulness`
    /// (mutually exclusive with `cluster_weights`).
    pub node_weights: Option<&'a [f64]>,
    /// Optional positive finite cluster weights of length `K` for `ClusterAreaFaithfulness`
    /// (mutually exclusive with `node_weights`).
    pub cluster_weights: Option<&'a [f64]>,
    /// Optional reference layout of length `N` required by `PrismDistortion` and `NorMetrics`.
    pub original_layout: Option<&'a Layout2d<S>>,
    /// Optional per-node rectangle half-widths of length `N` for `SprawlterNn` and `NorMetrics`.
    pub sprawlter_half_widths: Option<&'a [S]>,
    /// Optional per-node rectangle half-heights of length `N` for `SprawlterNn` and `NorMetrics`.
    pub sprawlter_half_heights: Option<&'a [S]>,
}

impl<'a, S: Scalar> Default for MetricInputs<'a, S> {
    fn default() -> Self {
        Self {
            labels: None,
            node_weights: None,
            cluster_weights: None,
            original_layout: None,
            sprawlter_half_widths: None,
            sprawlter_half_heights: None,
        }
    }
}

impl<'a, S: Scalar> MetricInputs<'a, S> {
    /// Creates empty [`MetricInputs`] with all optional fields set to `None`.
    pub fn new() -> Self {
        Self::default()
    }
}
