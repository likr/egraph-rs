/// Individual quality metric selectors passed to [`super::evaluate_metrics`].
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
pub(super) struct RequestedFlags {
    pub(super) sns: bool,
    pub(super) sampled_sns: bool,
    pub(super) crossing_number: bool,
    pub(super) crossing_angle: bool,
    pub(super) sampled_crossing_number: bool,
    pub(super) crosslessness: bool,
    pub(super) sampled_crosslessness: bool,
    pub(super) np1: bool,
    pub(super) np2: bool,
    pub(super) sampled_np1: bool,
    pub(super) sampled_np2: bool,
    pub(super) gabriel_property: bool,
    pub(super) node_resolution: bool,
    pub(super) s_rng: bool,
    pub(super) s_gg: bool,
    pub(super) aspect_ratio: bool,
    pub(super) angular_resolution: bool,
    pub(super) minimum_angle: bool,
    pub(super) minimum_angle_a1: bool,
    pub(super) layout_degeneracy: bool,
    pub(super) ideal_edge_lengths: bool,
    pub(super) adjusted_patch_nig: bool,
    pub(super) cluster_area_faithfulness: bool,
    pub(super) landmark_stratified_rta: bool,
    #[cfg(feature = "eval-baselines")]
    pub(super) dsc: bool,
    #[cfg(feature = "eval-baselines")]
    pub(super) dc: bool,
    #[cfg(feature = "eval-baselines")]
    pub(super) cdm: bool,
    #[cfg(feature = "eval-baselines")]
    pub(super) hdm: bool,
    #[cfg(feature = "eval-baselines")]
    pub(super) sepme_knng: bool,
    #[cfg(feature = "eval-baselines")]
    pub(super) sepme_rng: bool,
    #[cfg(feature = "eval-baselines")]
    pub(super) sepme_gab: bool,
    #[cfg(feature = "eval-baselines")]
    pub(super) sepme_gong: bool,
    #[cfg(feature = "eval-baselines")]
    pub(super) neighborhood_hit: bool,
    #[cfg(feature = "eval-baselines")]
    pub(super) silhouette: bool,
    #[cfg(feature = "eval-baselines")]
    pub(super) calinski_harabasz: bool,
    #[cfg(feature = "eval-baselines")]
    pub(super) davies_bouldin: bool,
    #[cfg(feature = "eval-baselines")]
    pub(super) cq_kmeans_ari: bool,
    #[cfg(feature = "eval-baselines")]
    pub(super) cq_kmeans_ami: bool,
    #[cfg(feature = "eval-baselines")]
    pub(super) cq_hac_hom: bool,
    #[cfg(feature = "eval-baselines")]
    pub(super) cq_hac_ami: bool,
    #[cfg(feature = "eval-baselines")]
    pub(super) pairwise_distance_overlap_error: bool,
    #[cfg(feature = "eval-baselines")]
    pub(super) sprawlter_nn: bool,
}

impl RequestedFlags {
    pub(super) fn from_targets(targets: &[MetricTarget]) -> Self {
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
    pub(super) fn needs_kdtree(&self) -> bool {
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
