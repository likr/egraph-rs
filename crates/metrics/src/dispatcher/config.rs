use egraph_core::{CoreError, CoreResult};
use egraph_metrics_cluster::ClusterAreaConfig;

/// Configuration parameters for `eval-baselines` cluster separation, CVI, and Sprawlter targets.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
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

impl ClusterBaselineConfig {
    /// Creates a new [`ClusterBaselineConfig`] initialized to [`Self::default`].
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the neighborhood size `k` for `SepmeKnng`.
    #[inline]
    pub fn with_sepme_knng_k(mut self, sepme_knng_k: usize) -> Self {
        self.sepme_knng_k = sepme_knng_k;
        self
    }

    /// Sets the neighborhood size `k` for `NeighborhoodHit`.
    #[inline]
    pub fn with_neighborhood_hit_k(mut self, neighborhood_hit_k: usize) -> Self {
        self.neighborhood_hit_k = neighborhood_hit_k;
        self
    }

    /// Sets the grid resolution for `DistributionConsistency`.
    #[inline]
    pub fn with_dc_grid_size(mut self, dc_grid_size: usize) -> Self {
        self.dc_grid_size = dc_grid_size;
        self
    }

    /// Sets the grid resolution for `ClassDensityMeasure` and `HistogramDensityMeasure`.
    #[inline]
    pub fn with_cdm_grid_size(mut self, cdm_grid_size: usize) -> Self {
        self.cdm_grid_size = cdm_grid_size;
        self
    }

    /// Sets the optional Gaussian KDE bandwidth for `ClassDensityMeasure`.
    #[inline]
    pub fn with_cdm_bandwidth(mut self, cdm_bandwidth: Option<f64>) -> Self {
        self.cdm_bandwidth = cdm_bandwidth;
        self
    }

    /// Sets the histogram bin count for `PairwiseDistanceOverlapError`.
    #[inline]
    pub fn with_num_bins(mut self, num_bins: usize) -> Self {
        self.num_bins = num_bins;
        self
    }

    /// Sets the observable angle parameter `gamma` in `[0.0, 1.0]` for `SepmeGong`.
    #[inline]
    pub fn with_gong_gamma(mut self, gong_gamma: f64) -> Self {
        self.gong_gamma = gong_gamma;
        self
    }

    /// Sets the number of $k$-means++ initializations for `CqKmeans*`.
    #[inline]
    pub fn with_kmeans_n_init(mut self, kmeans_n_init: usize) -> Self {
        self.kmeans_n_init = kmeans_n_init;
        self
    }

    /// Sets the maximum Lloyd iterations for `CqKmeans*`.
    #[inline]
    pub fn with_kmeans_max_iters(mut self, kmeans_max_iters: usize) -> Self {
        self.kmeans_max_iters = kmeans_max_iters;
        self
    }

    /// Sets the deterministic RNG seed for `CqKmeans*`.
    #[inline]
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Sets the uniform square half-extent radius for `SprawlterNn` and `NorMetrics`.
    #[inline]
    pub fn with_sprawlter_node_radius(mut self, sprawlter_node_radius: f64) -> Self {
        self.sprawlter_node_radius = sprawlter_node_radius;
        self
    }

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
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
pub struct MetricConfig {
    /// Number of deterministic degree-stratified BFS pivot sources for `SampledScaleNormalizedStress`.
    pub sns_num_pivots: usize,
    /// Explicit RNG seed for `SampledScaleNormalizedStress`.
    pub sns_seed: u64,
    /// Number of deterministic samples for `SampledCrossingNumber` and `SampledCrosslessness`.
    pub crossing_num_samples: usize,
    /// Explicit RNG seed for `SampledCrossingNumber` and `SampledCrosslessness`.
    pub crossing_seed: u64,
    /// Number of deterministic vertex samples for `SampledNp1` and `SampledNp2`.
    pub np_num_samples: usize,
    /// Explicit RNG seed for `SampledNp1` and `SampledNp2`.
    pub np_seed: u64,
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
    cluster_baselines: ClusterBaselineConfig,
}

impl Default for MetricConfig {
    fn default() -> Self {
        Self {
            sns_num_pivots: 64,
            sns_seed: 42,
            crossing_num_samples: 4096,
            crossing_seed: 42,
            np_num_samples: 4096,
            np_seed: 42,
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
            cluster_baselines: ClusterBaselineConfig::default(),
        }
    }
}

impl MetricConfig {
    /// Creates a new [`MetricConfig`] initialized to [`Self::default`].
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the number of deterministic pivots for `SampledScaleNormalizedStress`.
    #[inline]
    pub fn with_sns_num_pivots(mut self, sns_num_pivots: usize) -> Self {
        self.sns_num_pivots = sns_num_pivots;
        self
    }

    /// Sets the explicit RNG seed for `SampledScaleNormalizedStress`.
    #[inline]
    pub fn with_sns_seed(mut self, sns_seed: u64) -> Self {
        self.sns_seed = sns_seed;
        self
    }

    /// Sets the number of deterministic edge-pair samples for `SampledCrossingNumber` and `SampledCrosslessness`.
    #[inline]
    pub fn with_crossing_num_samples(mut self, crossing_num_samples: usize) -> Self {
        self.crossing_num_samples = crossing_num_samples;
        self
    }

    /// Sets the explicit RNG seed for `SampledCrossingNumber` and `SampledCrosslessness`.
    #[inline]
    pub fn with_crossing_seed(mut self, crossing_seed: u64) -> Self {
        self.crossing_seed = crossing_seed;
        self
    }

    /// Sets the number of deterministic vertex samples for `SampledNp1` and `SampledNp2`.
    #[inline]
    pub fn with_np_num_samples(mut self, np_num_samples: usize) -> Self {
        self.np_num_samples = np_num_samples;
        self
    }

    /// Sets the explicit RNG seed for `SampledNp1` and `SampledNp2`.
    #[inline]
    pub fn with_np_seed(mut self, np_seed: u64) -> Self {
        self.np_seed = np_seed;
        self
    }

    /// Sets the unit target edge-length scale factor for `IdealEdgeLengths`.
    #[inline]
    pub fn with_ideal_edge_length(mut self, ideal_edge_length: f64) -> Self {
        self.ideal_edge_length = ideal_edge_length;
        self
    }

    /// Sets the number of MaxMin shortest-path landmarks for `LandmarkStratifiedRta`.
    #[inline]
    pub fn with_rta_num_landmarks(mut self, rta_num_landmarks: usize) -> Self {
        self.rta_num_landmarks = rta_num_landmarks;
        self
    }

    /// Sets the triplet sample budget for `LandmarkStratifiedRta`.
    #[inline]
    pub fn with_rta_num_triplets(mut self, rta_num_triplets: usize) -> Self {
        self.rta_num_triplets = rta_num_triplets;
        self
    }

    /// Sets the minimum vertex degree preference for `LandmarkStratifiedRta`.
    #[inline]
    pub fn with_rta_min_degree(mut self, rta_min_degree: usize) -> Self {
        self.rta_min_degree = rta_min_degree;
        self
    }

    /// Sets the explicit RNG seed for `LandmarkStratifiedRta`.
    #[inline]
    pub fn with_rta_seed(mut self, rta_seed: u64) -> Self {
        self.rta_seed = rta_seed;
        self
    }

    /// Sets the half-width patch radius for `AdjustedPatchNig`.
    #[inline]
    pub fn with_patch_nig_radius(mut self, patch_nig_radius: usize) -> Self {
        self.patch_nig_radius = patch_nig_radius;
        self
    }

    /// Sets the equal-aspect canvas size in pixels for `AdjustedPatchNig`.
    #[inline]
    pub fn with_patch_nig_canvas_size(mut self, patch_nig_canvas_size: usize) -> Self {
        self.patch_nig_canvas_size = patch_nig_canvas_size;
        self
    }

    /// Sets the viewport padding fraction for `AdjustedPatchNig`.
    #[inline]
    pub fn with_patch_nig_pad_frac(mut self, patch_nig_pad_frac: f64) -> Self {
        self.patch_nig_pad_frac = patch_nig_pad_frac;
        self
    }

    /// Sets the explicit RNG seed for `AdjustedPatchNig`.
    #[inline]
    pub fn with_patch_nig_seed(mut self, patch_nig_seed: u64) -> Self {
        self.patch_nig_seed = patch_nig_seed;
        self
    }

    /// Sets the configuration parameters for `ClusterAreaFaithfulness` (`G-CAF`).
    #[inline]
    pub fn with_cluster_area(mut self, cluster_area: ClusterAreaConfig) -> Self {
        self.cluster_area = cluster_area;
        self
    }

    /// Sets the configuration parameters for `eval-baselines` cluster/separation/Sprawlter targets.
    #[inline]
    pub fn with_baseline_config(mut self, cfg: ClusterBaselineConfig) -> Self {
        self.cluster_baselines = cfg;
        self
    }

    /// Returns the configuration parameters for `eval-baselines` cluster/separation/Sprawlter targets.
    #[inline]
    pub fn baseline_config(&self) -> &ClusterBaselineConfig {
        &self.cluster_baselines
    }

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
        if self.np_num_samples == 0 {
            return Err(CoreError::invalid_input("np_num_samples must be >= 1"));
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
        self.cluster_baselines.validate()?;
        Ok(())
    }
}
