//! Target-selective metric evaluation dispatcher avoiding unnecessary computation.

use egraph_core::{CoreError, CoreResult, CsrGraph, Layout2d, Scalar};
use egraph_spatial::StaticKdTree;

use crate::{
    build_kdtree_f64,
    cluster_area::{ClusterAreaConfig, cluster_area_faithfulness_with_weights},
    community::adjusted_patch_nig,
    crossings::{
        crossing_number, crossing_summary, crosslessness_from_count, sampled_crossing_number,
    },
    geometry::{
        angular_resolution, aspect_ratio, ideal_edge_lengths, layout_degeneracy,
        minimum_angle_summary,
    },
    neighborhood::{
        neighborhood_preservation_1hop_with_tree, neighborhood_preservation_2hop_with_tree,
        sampled_neighborhood_preservation_1hop_with_tree,
        sampled_neighborhood_preservation_2hop_with_tree,
    },
    proximity::{
        gabriel_graph_preservation_with_tree, gabriel_graph_property_with_tree,
        node_resolution_with_tree, relative_neighborhood_preservation_with_tree,
    },
    rta::landmark_stratified_rta,
    stress::{sampled_scale_normalized_stress, scale_normalized_stress},
    validate_graph_layout,
};

#[cfg(feature = "eval-baselines")]
use crate::cluster_baselines::{
    calinski_harabasz_score, class_density_measure, cq_hac, cq_kmeans, davies_bouldin_score,
    distance_consistency, distribution_consistency, neighborhood_hit_with_tree,
    pairwise_distance_overlap_error, sepme_gab_with_tree, sepme_gong_with_tree,
    sepme_knng_with_tree, sepme_rng_with_tree, silhouette_score, sprawlter_nn_radii,
};

/// Individual quality metric selectors passed to [`evaluate_metrics`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MetricTarget {
    /// Exact closed-form Scale-Normalized Stress (`SNS`).
    ScaleNormalizedStress,
    /// $O(kN)$ degree-stratified pivot-sampled Scale-Normalized Stress (`SNS`).
    SampledScaleNormalizedStress,
    /// Exact sweep-line edge crossing count.
    CrossingNumber,
    /// Exact sweep-line edge crossing-angle penalty ($\sum \cos^2(\theta)$).
    CrossingAngle,
    /// $O(\text{num\_samples})$ deterministic sampled crossing count.
    SampledCrossingNumber,
    /// DRGraph Eq. (17) exact crosslessness $1 - \sqrt{c / c_{\max}} \in [0.0, 1.0]$.
    Crosslessness,
    /// DRGraph Eq. (17) sampled crosslessness $1 - \sqrt{\hat{c} / c_{\max}} \in [0.0, 1.0]$.
    SampledCrosslessness,
    /// 1-hop neighborhood preservation (`NP1`).
    NeighborhoodPreservation1Hop,
    /// 2-hop neighborhood preservation (`NP2`).
    NeighborhoodPreservation2Hop,
    /// $O(\text{num\_samples} \cdot k \log N)$ deterministic sampled 1-hop neighborhood preservation (`Sampled NP1`).
    SampledNp1,
    /// $O(\text{num\_samples} \cdot k \log N)$ deterministic sampled 2-hop neighborhood preservation (`Sampled NP2`).
    SampledNp2,
    /// Gabriel graph property violation sum.
    GabrielGraphProperty,
    /// One-sided hinge node-resolution violation sum.
    NodeResolution,
    /// Eades et al. (2017) unweighted shape-based 1-hop vs RNG Mean Jaccard Similarity (`S_RNG`;
    /// distinct from Hong et al. 2022 `dRNG` and Fang et al. 2025 `dcRNG`).
    RelativeNeighborhoodPreservation,
    /// Eades et al. (2017) unweighted shape-based 1-hop vs Gabriel Graph Mean Jaccard Similarity (`S_GG`;
    /// distinct from Hong et al. 2022 `dGG` and Fang et al. 2025 `dcGG`).
    GabrielGraphPreservation,
    /// PCA singular-value aspect ratio ($\sigma_2 / \sigma_1$).
    AspectRatio,
    /// Exponential angular-resolution penalty at shared vertices.
    AngularResolution,
    /// DRGraph Eq. (19) strict minimum-angle resolution (`A2`, zero-length edges penalize vertex to `0.0`).
    MinimumAngle,
    /// DRGraph Eq. (19) minimum-angle resolution (`A1`, `atan2(0, 0) = 0.0`).
    MinimumAngleA1,
    /// Scalar layout-degeneracy summary in `[0.0, 1.0]` combining duplicate ratio, zero-length edge ratio, and bbox collapse.
    LayoutDegeneracy,
    /// Weighted or unit ideal edge-length relative squared error.
    IdealEdgeLengths,
    /// Adjusted Patch Normalized Information Gain (`Adjusted Patch-NIG`); requires `community_labels`.
    AdjustedPatchNig,
    /// Grid-based Cluster Area Faithfulness (`G-CAF`); requires `community_labels`.
    ClusterAreaFaithfulness,
    /// Landmark-Stratified Random Triplet Accuracy (`L-Stratified-RTA`).
    LandmarkStratifiedRta,
    /// Distance Consistency (`DSC`); requires `community_labels`.
    #[cfg(feature = "eval-baselines")]
    Dsc,
    /// Distribution Consistency (`DC`); requires `community_labels`.
    #[cfg(feature = "eval-baselines")]
    Dc,
    /// Class Density Measure (`CDM`); requires `community_labels`.
    #[cfg(feature = "eval-baselines")]
    Cdm,
    /// 2D Histogram Density Measure (`2D-HDM`); requires `community_labels`.
    #[cfg(feature = "eval-baselines")]
    Hdm,
    /// SepMe $k$-NN graph DIR CPT; requires `community_labels`.
    #[cfg(feature = "eval-baselines")]
    SepmeKnng,
    /// SepMe Relative Neighborhood Graph DIR CPT; requires `community_labels`.
    #[cfg(feature = "eval-baselines")]
    SepmeRng,
    /// SepMe Gabriel Graph DIR CPT; requires `community_labels`.
    #[cfg(feature = "eval-baselines")]
    SepmeGab,
    /// SepMe $\gamma$-Observable Neighbor Graph DIR CPT; requires `community_labels`.
    #[cfg(feature = "eval-baselines")]
    SepmeGong,
    /// Neighborhood Hit ($M_{\mathrm{NH}}$); requires `community_labels`.
    #[cfg(feature = "eval-baselines")]
    NeighborhoodHit,
    /// Silhouette coefficient; requires `community_labels`.
    #[cfg(feature = "eval-baselines")]
    Silhouette,
    /// Calinski-Harabasz Index; requires `community_labels`.
    #[cfg(feature = "eval-baselines")]
    CalinskiHarabasz,
    /// Davies-Bouldin Index; requires `community_labels`.
    #[cfg(feature = "eval-baselines")]
    DaviesBouldin,
    /// $k$-means Cluster Quality (`CQ-kmeans` ARI); requires `community_labels`.
    #[cfg(feature = "eval-baselines")]
    CqKmeansAri,
    /// $k$-means Cluster Quality (`CQ-kmeans` AMI); requires `community_labels`.
    #[cfg(feature = "eval-baselines")]
    CqKmeansAmi,
    /// Average-linkage HAC Cluster Quality (`CQ-HAC` Homogeneity); requires `community_labels`.
    #[cfg(feature = "eval-baselines")]
    CqHacHom,
    /// Average-linkage HAC Cluster Quality (`CQ-HAC` AMI); requires `community_labels`.
    #[cfg(feature = "eval-baselines")]
    CqHacAmi,
    /// Pairwise Distance-Overlap Error (`PDOE`); requires `community_labels`.
    #[cfg(feature = "eval-baselines")]
    PairwiseDistanceOverlapError,
    /// Sprawlter-NN composite overlap clutter measure using uniform `sprawlter_node_radius`.
    #[cfg(feature = "eval-baselines")]
    SprawlterNn,
}

/// Configuration parameters for `eval-baselines` cluster separation, CVI, and Sprawlter targets.
#[cfg(feature = "eval-baselines")]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ClusterBaselineConfig {
    /// Neighborhood size `k` for `SepmeKnng` (must be `>= 1`; default `2`).
    pub sepme_knng_k: usize,
    /// Neighborhood size `k` for `NeighborhoodHit` (must be `>= 1`; default `7`).
    pub neighborhood_hit_k: usize,
    /// Grid resolution for `Dc` (must be `>= 1`; default `20`).
    pub dc_grid_size: usize,
    /// Grid resolution for `Cdm` and `Hdm` (must be `>= 1`; default `64`).
    pub cdm_grid_size: usize,
    /// Optional Gaussian KDE bandwidth for `Cdm` (`None` uses `1.5 / cdm_grid_size as f64`).
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

/// Configuration parameters for parameterized targets in [`evaluate_metrics`].
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
    /// Target edge length fallback for unweighted graphs in `IdealEdgeLengths`.
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

/// Selective evaluation report containing `Some(...)` strictly for requested [`MetricTarget`]s.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct MetricReport {
    pub scale_normalized_stress: Option<f64>,
    pub sampled_scale_normalized_stress: Option<f64>,
    pub crossing_number: Option<usize>,
    pub crossing_angle: Option<f64>,
    pub sampled_crossing_number: Option<f64>,
    pub crosslessness: Option<f64>,
    pub sampled_crosslessness: Option<f64>,
    pub neighborhood_preservation_1hop: Option<f64>,
    pub neighborhood_preservation_2hop: Option<f64>,
    pub sampled_np1: Option<f64>,
    pub sampled_np2: Option<f64>,
    pub gabriel_graph_property: Option<f64>,
    pub node_resolution: Option<f64>,
    pub relative_neighborhood_preservation: Option<f64>,
    pub gabriel_graph_preservation: Option<f64>,
    pub aspect_ratio: Option<f64>,
    pub angular_resolution: Option<f64>,
    pub minimum_angle: Option<f64>,
    pub minimum_angle_a1: Option<f64>,
    pub layout_degeneracy: Option<f64>,
    pub ideal_edge_lengths: Option<f64>,
    pub adjusted_patch_nig: Option<f64>,
    pub cluster_area_faithfulness: Option<f64>,
    pub landmark_stratified_rta: Option<f64>,
    #[cfg(feature = "eval-baselines")]
    pub dsc: Option<f64>,
    #[cfg(feature = "eval-baselines")]
    pub dc: Option<f64>,
    #[cfg(feature = "eval-baselines")]
    pub cdm: Option<f64>,
    #[cfg(feature = "eval-baselines")]
    pub hdm: Option<f64>,
    #[cfg(feature = "eval-baselines")]
    pub sepme_knng: Option<f64>,
    #[cfg(feature = "eval-baselines")]
    pub sepme_rng: Option<f64>,
    #[cfg(feature = "eval-baselines")]
    pub sepme_gab: Option<f64>,
    #[cfg(feature = "eval-baselines")]
    pub sepme_gong: Option<f64>,
    #[cfg(feature = "eval-baselines")]
    pub neighborhood_hit: Option<f64>,
    #[cfg(feature = "eval-baselines")]
    pub silhouette: Option<f64>,
    #[cfg(feature = "eval-baselines")]
    pub calinski_harabasz: Option<f64>,
    #[cfg(feature = "eval-baselines")]
    pub davies_bouldin: Option<f64>,
    #[cfg(feature = "eval-baselines")]
    pub cq_kmeans_ari: Option<f64>,
    #[cfg(feature = "eval-baselines")]
    pub cq_kmeans_ami: Option<f64>,
    #[cfg(feature = "eval-baselines")]
    pub cq_hac_hom: Option<f64>,
    #[cfg(feature = "eval-baselines")]
    pub cq_hac_ami: Option<f64>,
    #[cfg(feature = "eval-baselines")]
    pub pairwise_distance_overlap_error: Option<f64>,
    #[cfg(feature = "eval-baselines")]
    pub sprawlter_nn: Option<f64>,
}

#[derive(Default)]
struct RequestedFlags {
    sns: bool,
    sampled_sns: bool,
    crossing_number: bool,
    crossing_angle: bool,
    sampled_crossing_number: bool,
    crosslessness: bool,
    sampled_crosslessness: bool,
    np1: bool,
    np2: bool,
    sampled_np1: bool,
    sampled_np2: bool,
    gabriel_property: bool,
    node_resolution: bool,
    s_rng: bool,
    s_gg: bool,
    aspect_ratio: bool,
    angular_resolution: bool,
    minimum_angle: bool,
    minimum_angle_a1: bool,
    layout_degeneracy: bool,
    ideal_edge_lengths: bool,
    adjusted_patch_nig: bool,
    cluster_area_faithfulness: bool,
    landmark_stratified_rta: bool,
    #[cfg(feature = "eval-baselines")]
    dsc: bool,
    #[cfg(feature = "eval-baselines")]
    dc: bool,
    #[cfg(feature = "eval-baselines")]
    cdm: bool,
    #[cfg(feature = "eval-baselines")]
    hdm: bool,
    #[cfg(feature = "eval-baselines")]
    sepme_knng: bool,
    #[cfg(feature = "eval-baselines")]
    sepme_rng: bool,
    #[cfg(feature = "eval-baselines")]
    sepme_gab: bool,
    #[cfg(feature = "eval-baselines")]
    sepme_gong: bool,
    #[cfg(feature = "eval-baselines")]
    neighborhood_hit: bool,
    #[cfg(feature = "eval-baselines")]
    silhouette: bool,
    #[cfg(feature = "eval-baselines")]
    calinski_harabasz: bool,
    #[cfg(feature = "eval-baselines")]
    davies_bouldin: bool,
    #[cfg(feature = "eval-baselines")]
    cq_kmeans_ari: bool,
    #[cfg(feature = "eval-baselines")]
    cq_kmeans_ami: bool,
    #[cfg(feature = "eval-baselines")]
    cq_hac_hom: bool,
    #[cfg(feature = "eval-baselines")]
    cq_hac_ami: bool,
    #[cfg(feature = "eval-baselines")]
    pairwise_distance_overlap_error: bool,
    #[cfg(feature = "eval-baselines")]
    sprawlter_nn: bool,
}

impl RequestedFlags {
    fn from_targets(targets: &[MetricTarget]) -> Self {
        let mut flags = Self::default();
        for &target in targets {
            match target {
                MetricTarget::ScaleNormalizedStress => flags.sns = true,
                MetricTarget::SampledScaleNormalizedStress => flags.sampled_sns = true,
                MetricTarget::CrossingNumber => flags.crossing_number = true,
                MetricTarget::CrossingAngle => flags.crossing_angle = true,
                MetricTarget::SampledCrossingNumber => flags.sampled_crossing_number = true,
                MetricTarget::Crosslessness => flags.crosslessness = true,
                MetricTarget::SampledCrosslessness => flags.sampled_crosslessness = true,
                MetricTarget::NeighborhoodPreservation1Hop => flags.np1 = true,
                MetricTarget::NeighborhoodPreservation2Hop => flags.np2 = true,
                MetricTarget::SampledNp1 => flags.sampled_np1 = true,
                MetricTarget::SampledNp2 => flags.sampled_np2 = true,
                MetricTarget::GabrielGraphProperty => flags.gabriel_property = true,
                MetricTarget::NodeResolution => flags.node_resolution = true,
                MetricTarget::RelativeNeighborhoodPreservation => flags.s_rng = true,
                MetricTarget::GabrielGraphPreservation => flags.s_gg = true,
                MetricTarget::AspectRatio => flags.aspect_ratio = true,
                MetricTarget::AngularResolution => flags.angular_resolution = true,
                MetricTarget::MinimumAngle => flags.minimum_angle = true,
                MetricTarget::MinimumAngleA1 => flags.minimum_angle_a1 = true,
                MetricTarget::LayoutDegeneracy => flags.layout_degeneracy = true,
                MetricTarget::IdealEdgeLengths => flags.ideal_edge_lengths = true,
                MetricTarget::AdjustedPatchNig => flags.adjusted_patch_nig = true,
                MetricTarget::ClusterAreaFaithfulness => flags.cluster_area_faithfulness = true,
                MetricTarget::LandmarkStratifiedRta => flags.landmark_stratified_rta = true,
                #[cfg(feature = "eval-baselines")]
                MetricTarget::Dsc => flags.dsc = true,
                #[cfg(feature = "eval-baselines")]
                MetricTarget::Dc => flags.dc = true,
                #[cfg(feature = "eval-baselines")]
                MetricTarget::Cdm => flags.cdm = true,
                #[cfg(feature = "eval-baselines")]
                MetricTarget::Hdm => flags.hdm = true,
                #[cfg(feature = "eval-baselines")]
                MetricTarget::SepmeKnng => flags.sepme_knng = true,
                #[cfg(feature = "eval-baselines")]
                MetricTarget::SepmeRng => flags.sepme_rng = true,
                #[cfg(feature = "eval-baselines")]
                MetricTarget::SepmeGab => flags.sepme_gab = true,
                #[cfg(feature = "eval-baselines")]
                MetricTarget::SepmeGong => flags.sepme_gong = true,
                #[cfg(feature = "eval-baselines")]
                MetricTarget::NeighborhoodHit => flags.neighborhood_hit = true,
                #[cfg(feature = "eval-baselines")]
                MetricTarget::Silhouette => flags.silhouette = true,
                #[cfg(feature = "eval-baselines")]
                MetricTarget::CalinskiHarabasz => flags.calinski_harabasz = true,
                #[cfg(feature = "eval-baselines")]
                MetricTarget::DaviesBouldin => flags.davies_bouldin = true,
                #[cfg(feature = "eval-baselines")]
                MetricTarget::CqKmeansAri => flags.cq_kmeans_ari = true,
                #[cfg(feature = "eval-baselines")]
                MetricTarget::CqKmeansAmi => flags.cq_kmeans_ami = true,
                #[cfg(feature = "eval-baselines")]
                MetricTarget::CqHacHom => flags.cq_hac_hom = true,
                #[cfg(feature = "eval-baselines")]
                MetricTarget::CqHacAmi => flags.cq_hac_ami = true,
                #[cfg(feature = "eval-baselines")]
                MetricTarget::PairwiseDistanceOverlapError => {
                    flags.pairwise_distance_overlap_error = true;
                }
                #[cfg(feature = "eval-baselines")]
                MetricTarget::SprawlterNn => flags.sprawlter_nn = true,
            }
        }
        flags
    }

    #[inline]
    fn needs_kdtree(&self) -> bool {
        let base = self.np1
            || self.np2
            || self.sampled_np1
            || self.sampled_np2
            || self.gabriel_property
            || self.node_resolution
            || self.s_rng
            || self.s_gg;
        #[cfg(feature = "eval-baselines")]
        {
            base || self.sepme_knng
                || self.sepme_rng
                || self.sepme_gab
                || self.sepme_gong
                || self.neighborhood_hit
        }
        #[cfg(not(feature = "eval-baselines"))]
        {
            base
        }
    }
}

/// Evaluates strictly the metrics listed in `targets`, skipping all unrequested work.
///
/// Shares a single [`StaticKdTree<2, f64>`] across requested spatial/neighborhood targets,
/// a single sweep-line pass across `CrossingNumber`/`CrossingAngle`/`Crosslessness`,
/// a single sampling pass across `SampledCrossingNumber`/`SampledCrosslessness`, and
/// a single angular scan across `MinimumAngle`/`MinimumAngleA1`.
/// For [`MetricTarget::AdjustedPatchNig`] and [`MetricTarget::ClusterAreaFaithfulness`],
/// use [`evaluate_metrics_with_communities`].
pub fn evaluate_metrics<W: Scalar, S: Scalar>(
    graph: &CsrGraph<W>,
    layout: &Layout2d<S>,
    targets: &[MetricTarget],
    config: &MetricConfig,
) -> CoreResult<MetricReport> {
    evaluate_metrics_with_communities(graph, layout, targets, config, None)
}

/// Evaluates strictly the metrics listed in `targets` with optional `community_labels`
/// (required when [`MetricTarget::AdjustedPatchNig`] or [`MetricTarget::ClusterAreaFaithfulness`]
/// is requested).
pub fn evaluate_metrics_with_communities<W: Scalar, S: Scalar>(
    graph: &CsrGraph<W>,
    layout: &Layout2d<S>,
    targets: &[MetricTarget],
    config: &MetricConfig,
    community_labels: Option<&[u32]>,
) -> CoreResult<MetricReport> {
    evaluate_metrics_with_community_weights(
        graph,
        layout,
        targets,
        config,
        community_labels,
        None,
        None,
    )
}

/// Evaluates strictly the metrics listed in `targets` with optional `community_labels` and
/// optional positive finite G-CAF target weights (`node_weights` or `cluster_weights`,
/// mutually exclusive, evaluated lazily when [`MetricTarget::ClusterAreaFaithfulness`] is requested).
pub fn evaluate_metrics_with_community_weights<W: Scalar, S: Scalar>(
    graph: &CsrGraph<W>,
    layout: &Layout2d<S>,
    targets: &[MetricTarget],
    config: &MetricConfig,
    community_labels: Option<&[u32]>,
    node_weights: Option<&[f64]>,
    cluster_weights: Option<&[f64]>,
) -> CoreResult<MetricReport> {
    validate_graph_layout(graph, layout, "evaluate_metrics")?;
    let flags = RequestedFlags::from_targets(targets);
    let mut report = MetricReport::default();

    if flags.sns {
        report.scale_normalized_stress = Some(scale_normalized_stress(graph, layout)?);
    }
    if flags.sampled_sns {
        report.sampled_scale_normalized_stress = Some(sampled_scale_normalized_stress(
            graph,
            layout,
            config.sns_num_pivots,
            config.sns_seed,
        )?);
    }

    if flags.crossing_angle {
        let summary = crossing_summary(graph, layout)?;
        report.crossing_angle = Some(summary.angle_cos_sq_sum);
        if flags.crossing_number {
            report.crossing_number = Some(summary.count);
        }
        if flags.crosslessness {
            report.crosslessness = Some(crosslessness_from_count(graph, summary.count as f64)?);
        }
    } else if flags.crossing_number || flags.crosslessness {
        let count = crossing_number(graph, layout)?;
        if flags.crossing_number {
            report.crossing_number = Some(count);
        }
        if flags.crosslessness {
            report.crosslessness = Some(crosslessness_from_count(graph, count as f64)?);
        }
    }

    if flags.sampled_crossing_number || flags.sampled_crosslessness {
        let est = sampled_crossing_number(
            graph,
            layout,
            config.crossing_num_samples,
            config.crossing_seed,
        )?;
        if flags.sampled_crossing_number {
            report.sampled_crossing_number = Some(est);
        }
        if flags.sampled_crosslessness {
            report.sampled_crosslessness = Some(crosslessness_from_count(graph, est)?);
        }
    }
    if (flags.sampled_np1 || flags.sampled_np2) && config.crossing_num_samples == 0 {
        return Err(CoreError::invalid_input(
            "crossing_num_samples must be >= 1",
        ));
    }

    let kdtree: Option<StaticKdTree<2, f64>> = if flags.needs_kdtree() {
        Some(build_kdtree_f64(layout)?)
    } else {
        None
    };

    if let Some(tree) = kdtree.as_ref() {
        if flags.np1 {
            report.neighborhood_preservation_1hop = Some(neighborhood_preservation_1hop_with_tree(
                graph, layout, tree,
            )?);
        }
        if flags.np2 {
            report.neighborhood_preservation_2hop = Some(neighborhood_preservation_2hop_with_tree(
                graph, layout, tree,
            )?);
        }
        if flags.sampled_np1 {
            report.sampled_np1 = Some(sampled_neighborhood_preservation_1hop_with_tree(
                graph,
                layout,
                tree,
                config.crossing_num_samples,
                config.crossing_seed,
            )?);
        }
        if flags.sampled_np2 {
            report.sampled_np2 = Some(sampled_neighborhood_preservation_2hop_with_tree(
                graph,
                layout,
                tree,
                config.crossing_num_samples,
                config.crossing_seed,
            )?);
        }
        if flags.gabriel_property {
            report.gabriel_graph_property =
                Some(gabriel_graph_property_with_tree(graph, layout, tree)?);
        }
        if flags.node_resolution {
            report.node_resolution = Some(node_resolution_with_tree(layout, tree)?);
        }
        if flags.s_rng {
            report.relative_neighborhood_preservation = Some(
                relative_neighborhood_preservation_with_tree(graph, layout, tree)?,
            );
        }
        if flags.s_gg {
            report.gabriel_graph_preservation =
                Some(gabriel_graph_preservation_with_tree(graph, layout, tree)?);
        }
        #[cfg(feature = "eval-baselines")]
        {
            if flags.sepme_knng {
                let labels = community_labels.ok_or_else(|| {
                    CoreError::invalid_input("SepmeKnng target requires community_labels")
                })?;
                report.sepme_knng = Some(sepme_knng_with_tree(
                    layout,
                    labels,
                    config.cluster_baselines.sepme_knng_k,
                    tree,
                )?);
            }
            if flags.sepme_rng {
                let labels = community_labels.ok_or_else(|| {
                    CoreError::invalid_input("SepmeRng target requires community_labels")
                })?;
                report.sepme_rng = Some(sepme_rng_with_tree(layout, labels, tree)?);
            }
            if flags.sepme_gab {
                let labels = community_labels.ok_or_else(|| {
                    CoreError::invalid_input("SepmeGab target requires community_labels")
                })?;
                report.sepme_gab = Some(sepme_gab_with_tree(layout, labels, tree)?);
            }
            if flags.sepme_gong {
                let labels = community_labels.ok_or_else(|| {
                    CoreError::invalid_input("SepmeGong target requires community_labels")
                })?;
                report.sepme_gong = Some(sepme_gong_with_tree(
                    layout,
                    labels,
                    config.cluster_baselines.gong_gamma,
                    tree,
                )?);
            }
            if flags.neighborhood_hit {
                let labels = community_labels.ok_or_else(|| {
                    CoreError::invalid_input("NeighborhoodHit target requires community_labels")
                })?;
                report.neighborhood_hit = Some(neighborhood_hit_with_tree(
                    layout,
                    labels,
                    config.cluster_baselines.neighborhood_hit_k,
                    tree,
                )?);
            }
        }
    }

    if flags.aspect_ratio {
        report.aspect_ratio = Some(aspect_ratio(layout)?);
    }
    if flags.angular_resolution {
        report.angular_resolution = Some(angular_resolution(graph, layout)?);
    }
    if flags.minimum_angle || flags.minimum_angle_a1 {
        let summary = minimum_angle_summary(graph, layout)?;
        if flags.minimum_angle {
            report.minimum_angle = Some(summary.a2);
        }
        if flags.minimum_angle_a1 {
            report.minimum_angle_a1 = Some(summary.a1);
        }
    }
    if flags.layout_degeneracy {
        let deg = layout_degeneracy(graph, layout)?;
        let bbox_penalty = if deg.is_bbox_degenerate && layout.len() > 1 {
            1.0
        } else {
            0.0
        };
        report.layout_degeneracy = Some(
            deg.duplicate_ratio_f32
                .max(deg.zero_length_edge_ratio)
                .max(bbox_penalty),
        );
    }
    if flags.ideal_edge_lengths {
        if !config.ideal_edge_length.is_finite() {
            return Err(CoreError::non_finite("MetricConfig::ideal_edge_length"));
        }
        if config.ideal_edge_length <= 0.0 {
            return Err(CoreError::invalid_input(
                "ideal_edge_length must be strictly positive",
            ));
        }
        let val = if graph.is_weighted() {
            ideal_edge_lengths(graph, layout)?
        } else {
            crate::geometry::ideal_edge_lengths_with_unit(graph, layout, config.ideal_edge_length)?
        };
        report.ideal_edge_lengths = Some(val);
    }
    if flags.adjusted_patch_nig {
        let labels = community_labels.ok_or_else(|| {
            CoreError::invalid_input("AdjustedPatchNig target requires community_labels")
        })?;
        report.adjusted_patch_nig = Some(adjusted_patch_nig(
            layout,
            labels,
            config.patch_nig_radius,
            config.patch_nig_canvas_size,
            config.patch_nig_pad_frac,
            config.patch_nig_seed,
        )?);
    }
    if flags.cluster_area_faithfulness {
        let labels = community_labels.ok_or_else(|| {
            CoreError::invalid_input("ClusterAreaFaithfulness target requires community_labels")
        })?;
        report.cluster_area_faithfulness = Some(
            cluster_area_faithfulness_with_weights(
                layout,
                labels,
                &config.cluster_area,
                node_weights,
                cluster_weights,
            )?
            .caf,
        );
    }
    if flags.landmark_stratified_rta {
        report.landmark_stratified_rta = Some(landmark_stratified_rta(
            graph,
            layout,
            config.rta_num_landmarks,
            config.rta_num_triplets,
            config.rta_min_degree,
            config.rta_seed,
        )?);
    }
    #[cfg(feature = "eval-baselines")]
    {
        if flags.dsc {
            let labels = community_labels
                .ok_or_else(|| CoreError::invalid_input("Dsc target requires community_labels"))?;
            report.dsc = Some(distance_consistency(layout, labels)?);
        }
        if flags.dc {
            let labels = community_labels
                .ok_or_else(|| CoreError::invalid_input("Dc target requires community_labels"))?;
            report.dc = Some(distribution_consistency(
                layout,
                labels,
                config.cluster_baselines.dc_grid_size,
            )?);
        }
        if flags.cdm || flags.hdm {
            let labels = community_labels.ok_or_else(|| {
                CoreError::invalid_input("Cdm/Hdm target requires community_labels")
            })?;
            let (cdm, hdm) = class_density_measure(
                layout,
                labels,
                config.cluster_baselines.cdm_grid_size,
                config.cluster_baselines.cdm_bandwidth,
            )?;
            if flags.cdm {
                report.cdm = Some(cdm);
            }
            if flags.hdm {
                report.hdm = Some(hdm);
            }
        }
        if flags.silhouette {
            let labels = community_labels.ok_or_else(|| {
                CoreError::invalid_input("Silhouette target requires community_labels")
            })?;
            report.silhouette = Some(silhouette_score(layout, labels)?);
        }
        if flags.calinski_harabasz {
            let labels = community_labels.ok_or_else(|| {
                CoreError::invalid_input("CalinskiHarabasz target requires community_labels")
            })?;
            report.calinski_harabasz = Some(calinski_harabasz_score(layout, labels)?);
        }
        if flags.davies_bouldin {
            let labels = community_labels.ok_or_else(|| {
                CoreError::invalid_input("DaviesBouldin target requires community_labels")
            })?;
            report.davies_bouldin = Some(davies_bouldin_score(layout, labels)?);
        }
        if flags.cq_kmeans_ari || flags.cq_kmeans_ami {
            let labels = community_labels.ok_or_else(|| {
                CoreError::invalid_input("CqKmeans target requires community_labels")
            })?;
            let pa = cq_kmeans(
                layout,
                labels,
                config.cluster_baselines.kmeans_n_init,
                config.cluster_baselines.kmeans_max_iters,
                config.cluster_baselines.seed,
            )?;
            if flags.cq_kmeans_ari {
                report.cq_kmeans_ari = Some(pa.ari);
            }
            if flags.cq_kmeans_ami {
                report.cq_kmeans_ami = Some(pa.ami);
            }
        }
        if flags.cq_hac_hom || flags.cq_hac_ami {
            let labels = community_labels.ok_or_else(|| {
                CoreError::invalid_input("CqHac target requires community_labels")
            })?;
            let pa = cq_hac(layout, labels)?;
            if flags.cq_hac_hom {
                report.cq_hac_hom = Some(pa.homogeneity);
            }
            if flags.cq_hac_ami {
                report.cq_hac_ami = Some(pa.ami);
            }
        }
        if flags.pairwise_distance_overlap_error {
            let labels = community_labels.ok_or_else(|| {
                CoreError::invalid_input(
                    "PairwiseDistanceOverlapError target requires community_labels",
                )
            })?;
            report.pairwise_distance_overlap_error = Some(pairwise_distance_overlap_error(
                layout,
                labels,
                config.cluster_baselines.num_bins,
            )?);
        }
        if flags.sprawlter_nn {
            let r_f64 = config.cluster_baselines.sprawlter_node_radius;
            if !r_f64.is_finite() {
                return Err(CoreError::non_finite(
                    "ClusterBaselineConfig::sprawlter_node_radius",
                ));
            }
            if r_f64 <= 0.0 {
                return Err(CoreError::invalid_input(
                    "ClusterBaselineConfig::sprawlter_node_radius must be strictly positive",
                ));
            }
            let r = S::from_f64(r_f64);
            let radii = vec![r; layout.len()];
            report.sprawlter_nn = Some(sprawlter_nn_radii(layout, &radii)?);
        }
    }

    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        cluster_area_faithfulness, crossing_angle, gabriel_graph_preservation,
        gabriel_graph_property, neighborhood_preservation_1hop, neighborhood_preservation_2hop,
        node_resolution, relative_neighborhood_preservation,
        sampled_neighborhood_preservation_1hop, sampled_neighborhood_preservation_2hop,
    };

    #[test]
    fn test_selective_dispatcher_computes_only_requested_targets() {
        let g = CsrGraph::<f64>::from_edges(
            4,
            &[(0, 1), (1, 2), (2, 3), (3, 0), (0, 2), (1, 3)],
            None,
            false,
            false,
        )
        .unwrap();
        let layout = Layout2d::from_xy(vec![0.0, 1.0, 1.0, 0.0], vec![0.0, 0.0, 1.0, 1.0]).unwrap();
        let cfg = MetricConfig {
            sns_num_pivots: 2,
            sns_seed: 9,
            crossing_num_samples: 64,
            crossing_seed: 9,
            ideal_edge_length: 1.0,
            rta_num_landmarks: 4,
            rta_num_triplets: 10_000,
            rta_min_degree: 1,
            rta_seed: 9,
            patch_nig_radius: 16,
            patch_nig_canvas_size: 1024,
            patch_nig_pad_frac: 0.04,
            patch_nig_seed: 9,
            cluster_area: ClusterAreaConfig::default(),
            #[cfg(feature = "eval-baselines")]
            cluster_baselines: ClusterBaselineConfig::default(),
        };
        assert_eq!(MetricConfig::default().sns_num_pivots, 64);

        // Request strictly NP2 + SNS: all other fields must remain None.
        let subset = evaluate_metrics(
            &g,
            &layout,
            &[
                MetricTarget::NeighborhoodPreservation2Hop,
                MetricTarget::ScaleNormalizedStress,
                MetricTarget::NeighborhoodPreservation2Hop,
            ],
            &cfg,
        )
        .unwrap();
        assert_eq!(
            subset.scale_normalized_stress,
            Some(scale_normalized_stress(&g, &layout).unwrap())
        );
        assert_eq!(
            subset.neighborhood_preservation_2hop,
            Some(neighborhood_preservation_2hop(&g, &layout).unwrap())
        );
        assert!(subset.crossing_number.is_none());
        assert!(subset.crossing_angle.is_none());
        assert!(subset.sampled_crossing_number.is_none());
        assert!(subset.crosslessness.is_none());
        assert!(subset.sampled_crosslessness.is_none());
        assert!(subset.sampled_scale_normalized_stress.is_none());
        assert!(subset.neighborhood_preservation_1hop.is_none());
        assert!(subset.sampled_np1.is_none());
        assert!(subset.sampled_np2.is_none());
        assert!(subset.gabriel_graph_property.is_none());
        assert!(subset.node_resolution.is_none());
        assert!(subset.relative_neighborhood_preservation.is_none());
        assert!(subset.gabriel_graph_preservation.is_none());
        assert!(subset.aspect_ratio.is_none());
        assert!(subset.angular_resolution.is_none());
        assert!(subset.minimum_angle.is_none());
        assert!(subset.minimum_angle_a1.is_none());
        assert!(subset.layout_degeneracy.is_none());
        assert!(subset.ideal_edge_lengths.is_none());
        assert!(subset.adjusted_patch_nig.is_none());
        assert!(subset.cluster_area_faithfulness.is_none());
        assert!(subset.landmark_stratified_rta.is_none());

        // Empty targets list returns all None cleanly.
        assert_eq!(
            evaluate_metrics(&g, &layout, &[], &cfg).unwrap(),
            MetricReport::default()
        );

        // Request every target and verify exact parity with standalone calls.
        let all_targets = [
            MetricTarget::ScaleNormalizedStress,
            MetricTarget::SampledScaleNormalizedStress,
            MetricTarget::CrossingNumber,
            MetricTarget::CrossingAngle,
            MetricTarget::SampledCrossingNumber,
            MetricTarget::Crosslessness,
            MetricTarget::SampledCrosslessness,
            MetricTarget::NeighborhoodPreservation1Hop,
            MetricTarget::NeighborhoodPreservation2Hop,
            MetricTarget::SampledNp1,
            MetricTarget::SampledNp2,
            MetricTarget::GabrielGraphProperty,
            MetricTarget::NodeResolution,
            MetricTarget::RelativeNeighborhoodPreservation,
            MetricTarget::GabrielGraphPreservation,
            MetricTarget::AspectRatio,
            MetricTarget::AngularResolution,
            MetricTarget::MinimumAngle,
            MetricTarget::MinimumAngleA1,
            MetricTarget::LayoutDegeneracy,
            MetricTarget::IdealEdgeLengths,
            MetricTarget::AdjustedPatchNig,
            MetricTarget::ClusterAreaFaithfulness,
            MetricTarget::LandmarkStratifiedRta,
        ];
        let communities = [0_u32, 0, 1, 1];
        let full =
            evaluate_metrics_with_communities(&g, &layout, &all_targets, &cfg, Some(&communities))
                .unwrap();
        assert_eq!(
            full.scale_normalized_stress,
            Some(scale_normalized_stress(&g, &layout).unwrap())
        );
        assert_eq!(
            full.sampled_scale_normalized_stress,
            Some(sampled_scale_normalized_stress(&g, &layout, 2, 9).unwrap())
        );
        assert_eq!(
            full.crossing_number,
            Some(crossing_number(&g, &layout).unwrap())
        );
        assert_eq!(
            full.crossing_angle,
            Some(crossing_angle(&g, &layout).unwrap())
        );
        assert_eq!(
            full.sampled_crossing_number,
            Some(sampled_crossing_number(&g, &layout, 64, 9).unwrap())
        );
        assert_eq!(
            full.crosslessness,
            Some(crate::crosslessness(&g, &layout).unwrap())
        );
        assert_eq!(
            full.sampled_crosslessness,
            Some(crate::sampled_crosslessness(&g, &layout, 64, 9).unwrap())
        );
        assert_eq!(
            full.neighborhood_preservation_1hop,
            Some(neighborhood_preservation_1hop(&g, &layout).unwrap())
        );
        assert_eq!(
            full.neighborhood_preservation_2hop,
            Some(neighborhood_preservation_2hop(&g, &layout).unwrap())
        );
        assert_eq!(
            full.sampled_np1,
            Some(sampled_neighborhood_preservation_1hop(&g, &layout, 64, 9).unwrap())
        );
        assert_eq!(
            full.sampled_np2,
            Some(sampled_neighborhood_preservation_2hop(&g, &layout, 64, 9).unwrap())
        );
        assert_eq!(
            full.gabriel_graph_property,
            Some(gabriel_graph_property(&g, &layout).unwrap())
        );
        assert_eq!(
            full.node_resolution,
            Some(node_resolution(&layout).unwrap())
        );
        assert_eq!(
            full.relative_neighborhood_preservation,
            Some(relative_neighborhood_preservation(&g, &layout).unwrap())
        );
        assert_eq!(
            full.gabriel_graph_preservation,
            Some(gabriel_graph_preservation(&g, &layout).unwrap())
        );
        assert_eq!(full.aspect_ratio, Some(aspect_ratio(&layout).unwrap()));
        assert_eq!(
            full.angular_resolution,
            Some(angular_resolution(&g, &layout).unwrap())
        );
        let min_ang = minimum_angle_summary(&g, &layout).unwrap();
        assert_eq!(full.minimum_angle, Some(min_ang.a2));
        assert_eq!(full.minimum_angle_a1, Some(min_ang.a1));
        assert_eq!(full.layout_degeneracy, Some(0.0));
        assert_eq!(
            full.ideal_edge_lengths,
            Some(ideal_edge_lengths(&g, &layout).unwrap())
        );
        assert_eq!(
            full.adjusted_patch_nig,
            Some(adjusted_patch_nig(&layout, &communities, 16, 1024, 0.04, 9).unwrap())
        );
        assert_eq!(
            full.cluster_area_faithfulness,
            Some(
                cluster_area_faithfulness(&layout, &communities, &cfg.cluster_area)
                    .unwrap()
                    .caf
            )
        );
        assert_eq!(
            full.landmark_stratified_rta,
            Some(landmark_stratified_rta(&g, &layout, 4, 10_000, 1, 9).unwrap())
        );
        assert!(matches!(
            evaluate_metrics(&g, &layout, &[MetricTarget::AdjustedPatchNig], &cfg),
            Err(CoreError::InvalidInput(_))
        ));
        assert!(matches!(
            evaluate_metrics(&g, &layout, &[MetricTarget::ClusterAreaFaithfulness], &cfg),
            Err(CoreError::InvalidInput(_))
        ));

        // Invalid config only errors when the corresponding metric is requested.
        let bad_caf_cfg = MetricConfig {
            cluster_area: ClusterAreaConfig {
                g_max: 4,
                ..cfg.cluster_area
            },
            ..cfg
        };
        assert!(evaluate_metrics(&g, &layout, &[MetricTarget::AspectRatio], &bad_caf_cfg).is_ok());
        assert!(matches!(
            evaluate_metrics_with_communities(
                &g,
                &layout,
                &[MetricTarget::ClusterAreaFaithfulness],
                &bad_caf_cfg,
                Some(&communities),
            ),
            Err(CoreError::InvalidInput(_))
        ));

        // Invalid config only errors when the corresponding metric is requested.
        let bad_cfg = MetricConfig {
            sns_num_pivots: 0,
            ..cfg
        };
        assert!(evaluate_metrics(&g, &layout, &[MetricTarget::AspectRatio], &bad_cfg).is_ok());
        assert!(matches!(
            evaluate_metrics(
                &g,
                &layout,
                &[MetricTarget::SampledScaleNormalizedStress],
                &bad_cfg
            ),
            Err(CoreError::InvalidInput(_))
        ));

        let bad_sample_cfg = MetricConfig {
            crossing_num_samples: 0,
            ..cfg
        };
        assert!(
            evaluate_metrics(&g, &layout, &[MetricTarget::AspectRatio], &bad_sample_cfg).is_ok()
        );
        assert!(matches!(
            evaluate_metrics(&g, &layout, &[MetricTarget::SampledNp1], &bad_sample_cfg),
            Err(CoreError::InvalidInput(_))
        ));
        assert!(matches!(
            evaluate_metrics(&g, &layout, &[MetricTarget::SampledNp2], &bad_sample_cfg),
            Err(CoreError::InvalidInput(_))
        ));

        // Verify sampled NP1/NP2 with sub-sample (< N) via evaluate_metrics shares KD-tree
        // and matches standalone calls bit-for-bit.
        let sub_cfg = MetricConfig {
            crossing_num_samples: 2,
            crossing_seed: 123,
            ..cfg
        };
        let sub_rep = evaluate_metrics(
            &g,
            &layout,
            &[MetricTarget::SampledNp1, MetricTarget::SampledNp2],
            &sub_cfg,
        )
        .unwrap();
        assert_eq!(
            sub_rep.sampled_np1,
            Some(sampled_neighborhood_preservation_1hop(&g, &layout, 2, 123).unwrap())
        );
        assert_eq!(
            sub_rep.sampled_np2,
            Some(sampled_neighborhood_preservation_2hop(&g, &layout, 2, 123).unwrap())
        );

        // Verify f32 layouts and f32-weighted graphs work uniformly across all targets.
        let g32 = CsrGraph::<f32>::from_edges(
            4,
            &[(0, 1), (1, 2), (2, 3), (3, 0)],
            Some(&[1.0_f32, 1.0, 1.0, 1.0]),
            false,
            false,
        )
        .unwrap();
        let l32 =
            Layout2d::from_xy(vec![0.0_f32, 1.0, 1.0, 0.0], vec![0.0_f32, 0.0, 1.0, 1.0]).unwrap();
        let rep32 =
            evaluate_metrics_with_communities(&g32, &l32, &all_targets, &cfg, Some(&communities))
                .unwrap();
        assert_eq!(rep32.crossing_number, Some(0));
        assert!((rep32.aspect_ratio.unwrap() - 1.0).abs() < 1e-6);
        assert!((rep32.ideal_edge_lengths.unwrap()).abs() < 1e-6);
        assert!((rep32.relative_neighborhood_preservation.unwrap() - 1.0).abs() < 1e-6);
        assert!((rep32.gabriel_graph_preservation.unwrap() - 1.0).abs() < 1e-6);
        assert!(rep32.cluster_area_faithfulness.unwrap().is_finite());
        assert!(cfg.validate().is_ok());
        assert!(bad_caf_cfg.validate().is_err());

        // Weighted G-CAF via evaluate_metrics_with_community_weights + lazy weight validation.
        let nw = [1.0, 2.0, 3.0, 4.0];
        let cw = [3.0, 7.0];
        let rep_nw = evaluate_metrics_with_community_weights(
            &g,
            &layout,
            &[MetricTarget::ClusterAreaFaithfulness],
            &cfg,
            Some(&communities),
            Some(&nw),
            None,
        )
        .unwrap();
        let rep_cw = evaluate_metrics_with_community_weights(
            &g,
            &layout,
            &[MetricTarget::ClusterAreaFaithfulness],
            &cfg,
            Some(&communities),
            None,
            Some(&cw),
        )
        .unwrap();
        assert_eq!(
            rep_nw.cluster_area_faithfulness,
            Some(
                cluster_area_faithfulness_with_weights(
                    &layout,
                    &communities,
                    &cfg.cluster_area,
                    Some(&nw),
                    None
                )
                .unwrap()
                .caf
            )
        );
        assert!(
            (rep_nw.cluster_area_faithfulness.unwrap() - rep_cw.cluster_area_faithfulness.unwrap())
                .abs()
                < 1e-12
        );
        // Lazy when ClusterAreaFaithfulness is not requested.
        assert!(
            evaluate_metrics_with_community_weights(
                &g,
                &layout,
                &[MetricTarget::AspectRatio],
                &cfg,
                Some(&communities),
                Some(&[-1.0]),
                Some(&[0.0]),
            )
            .is_ok()
        );
        assert!(matches!(
            evaluate_metrics_with_community_weights(
                &g,
                &layout,
                &[MetricTarget::ClusterAreaFaithfulness],
                &cfg,
                Some(&communities),
                Some(&nw),
                Some(&cw),
            ),
            Err(CoreError::InvalidInput(_))
        ));
    }

    #[cfg(feature = "eval-baselines")]
    #[test]
    fn test_eval_baselines_dispatcher_targets_and_lazy_validation() {
        let g = CsrGraph::<f64>::from_edges(
            6,
            &[(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3), (2, 3)],
            None,
            false,
            false,
        )
        .unwrap();
        let layout = Layout2d::from_xy(
            vec![0.0, 0.2, 0.1, 5.0, 5.2, 5.1],
            vec![0.0, 0.1, -0.1, 5.0, 5.1, 4.9],
        )
        .unwrap();
        let communities = [0_u32, 0, 0, 1, 1, 1];
        let cfg = MetricConfig::default();

        // SprawlterNn works without community_labels; all other eval-baselines fields remain None.
        let sprawl_only =
            evaluate_metrics(&g, &layout, &[MetricTarget::SprawlterNn], &cfg).unwrap();
        assert!(sprawl_only.sprawlter_nn.is_some());
        assert!(sprawl_only.dsc.is_none());
        assert!(sprawl_only.cq_kmeans_ari.is_none());

        // Label-dependent targets error when community_labels is None.
        for &target in &[
            MetricTarget::Dsc,
            MetricTarget::Dc,
            MetricTarget::Cdm,
            MetricTarget::Hdm,
            MetricTarget::SepmeKnng,
            MetricTarget::SepmeRng,
            MetricTarget::SepmeGab,
            MetricTarget::SepmeGong,
            MetricTarget::NeighborhoodHit,
            MetricTarget::Silhouette,
            MetricTarget::CalinskiHarabasz,
            MetricTarget::DaviesBouldin,
            MetricTarget::CqKmeansAri,
            MetricTarget::CqKmeansAmi,
            MetricTarget::CqHacHom,
            MetricTarget::CqHacAmi,
            MetricTarget::PairwiseDistanceOverlapError,
        ] {
            assert!(
                matches!(
                    evaluate_metrics(&g, &layout, &[target], &cfg),
                    Err(CoreError::InvalidInput(_))
                ),
                "expected InvalidInput when community_labels is None for {target:?}"
            );
        }

        // Request all 18 eval-baselines targets and verify exact parity with standalone calls.
        let baseline_targets = [
            MetricTarget::Dsc,
            MetricTarget::Dc,
            MetricTarget::Cdm,
            MetricTarget::Hdm,
            MetricTarget::SepmeKnng,
            MetricTarget::SepmeRng,
            MetricTarget::SepmeGab,
            MetricTarget::SepmeGong,
            MetricTarget::NeighborhoodHit,
            MetricTarget::Silhouette,
            MetricTarget::CalinskiHarabasz,
            MetricTarget::DaviesBouldin,
            MetricTarget::CqKmeansAri,
            MetricTarget::CqKmeansAmi,
            MetricTarget::CqHacHom,
            MetricTarget::CqHacAmi,
            MetricTarget::PairwiseDistanceOverlapError,
            MetricTarget::SprawlterNn,
        ];
        let rep = evaluate_metrics_with_communities(
            &g,
            &layout,
            &baseline_targets,
            &cfg,
            Some(&communities),
        )
        .unwrap();

        let cb = &cfg.cluster_baselines;
        assert_eq!(
            rep.dsc,
            Some(distance_consistency(&layout, &communities).unwrap())
        );
        assert_eq!(
            rep.dc,
            Some(distribution_consistency(&layout, &communities, cb.dc_grid_size).unwrap())
        );
        let (cdm, hdm) =
            class_density_measure(&layout, &communities, cb.cdm_grid_size, cb.cdm_bandwidth)
                .unwrap();
        assert_eq!(rep.cdm, Some(cdm));
        assert_eq!(rep.hdm, Some(hdm));
        assert_eq!(
            rep.sepme_knng,
            Some(
                crate::cluster_baselines::sepme_knng(&layout, &communities, cb.sepme_knng_k)
                    .unwrap()
            )
        );
        assert_eq!(
            rep.sepme_rng,
            Some(crate::cluster_baselines::sepme_rng(&layout, &communities).unwrap())
        );
        assert_eq!(
            rep.sepme_gab,
            Some(crate::cluster_baselines::sepme_gab(&layout, &communities).unwrap())
        );
        assert_eq!(
            rep.sepme_gong,
            Some(
                crate::cluster_baselines::sepme_gong(&layout, &communities, cb.gong_gamma).unwrap()
            )
        );
        assert_eq!(
            rep.neighborhood_hit,
            Some(
                crate::cluster_baselines::neighborhood_hit(
                    &layout,
                    &communities,
                    cb.neighborhood_hit_k
                )
                .unwrap()
            )
        );
        assert_eq!(
            rep.silhouette,
            Some(silhouette_score(&layout, &communities).unwrap())
        );
        assert_eq!(
            rep.calinski_harabasz,
            Some(calinski_harabasz_score(&layout, &communities).unwrap())
        );
        assert_eq!(
            rep.davies_bouldin,
            Some(davies_bouldin_score(&layout, &communities).unwrap())
        );
        let km = cq_kmeans(
            &layout,
            &communities,
            cb.kmeans_n_init,
            cb.kmeans_max_iters,
            cb.seed,
        )
        .unwrap();
        assert_eq!(rep.cq_kmeans_ari, Some(km.ari));
        assert_eq!(rep.cq_kmeans_ami, Some(km.ami));
        let hac = cq_hac(&layout, &communities).unwrap();
        assert_eq!(rep.cq_hac_hom, Some(hac.homogeneity));
        assert_eq!(rep.cq_hac_ami, Some(hac.ami));
        assert_eq!(
            rep.pairwise_distance_overlap_error,
            Some(pairwise_distance_overlap_error(&layout, &communities, cb.num_bins).unwrap())
        );
        let radii = vec![cb.sprawlter_node_radius; layout.len()];
        assert_eq!(
            rep.sprawlter_nn,
            Some(sprawlter_nn_radii(&layout, &radii).unwrap())
        );

        // Lazy config validation: invalid baseline config ignored when baseline targets not requested.
        let bad_cb_cfg = MetricConfig {
            cluster_baselines: ClusterBaselineConfig {
                gong_gamma: 2.0,
                ..cfg.cluster_baselines
            },
            ..cfg
        };
        assert!(bad_cb_cfg.validate().is_err());
        assert!(evaluate_metrics(&g, &layout, &[MetricTarget::AspectRatio], &bad_cb_cfg).is_ok());
        assert!(matches!(
            evaluate_metrics_with_communities(
                &g,
                &layout,
                &[MetricTarget::SepmeGong],
                &bad_cb_cfg,
                Some(&communities),
            ),
            Err(CoreError::InvalidInput(_))
        ));
    }
}
