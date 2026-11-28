use egraph_metrics_geometry::{CrossingPair, TorusCrossing};

/// Individual quality metric selectors passed to [`super::evaluate_metrics`].
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MetricTarget {
    /// Exact closed-form Scale-Normalized Stress (`SNS`).
    ScaleNormalizedStress,
    /// $O(kN)$ degree-stratified pivot-sampled Scale-Normalized Stress (`SNS`).
    SampledScaleNormalizedStress,
    /// Miller et al. (2024) Eq. (2) Geodesic Distortion without post-hoc rescaling
    /// (supported on [`egraph_core::ManifoldLayout2d`]).
    GeodesicDistortion,
    /// Exact sweep-line edge crossing count.
    CrossingNumber,
    /// Exact sweep-line list of non-adjacent crossing edge pairs.
    CrossingEdges,
    /// Exact sweep-line edge crossing-angle penalty ($\sum \cos^2(\theta)$).
    CrossingAngle,
    /// $O(\text{num\_samples})$ deterministic sampled crossing count.
    SampledCrossingNumber,
    /// DRGraph Eq. (17) exact crosslessness $1 - \sqrt{c / c_{\max}} \in [0.0, 1.0]$.
    Crosslessness,
    /// DRGraph Eq. (17) sampled crosslessness $1 - \sqrt{\hat{c} / c_{\max}} \in [0.0, 1.0]$.
    SampledCrosslessness,
    /// Exact edge crossing count on a 2D torus (`ManifoldKind::Torus2d`).
    TorusCrossingNumber,
    /// Exact list of crossing edge sub-segment pairs on a 2D torus (`ManifoldKind::Torus2d`).
    CrossingEdgesTorus,
    /// 1-hop neighborhood preservation (`NP1`).
    NeighborhoodPreservation1Hop,
    /// 2-hop neighborhood preservation (`NP2`).
    NeighborhoodPreservation2Hop,
    /// $O(\text{np\_num\_samples} \cdot k \log N)$ deterministic sampled 1-hop neighborhood preservation (`Sampled NP1`).
    SampledNp1,
    /// $O(\text{np\_num\_samples} \cdot k \log N)$ deterministic sampled 2-hop neighborhood preservation (`Sampled NP2`).
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
    /// PRISM layout-preservation distortion ($\sigma_{\mathrm{dist}}, \sigma_{\mathrm{disp}}$);
    /// requires `original_layout` in [`super::MetricInputs`].
    PrismDistortion,
    /// Adjusted Patch Normalized Information Gain (`Adjusted Patch-NIG`); requires `labels`.
    AdjustedPatchNig,
    /// Grid-based Cluster Area Faithfulness (`G-CAF`); requires `labels`.
    ClusterAreaFaithfulness,
    /// Landmark-Stratified Random Triplet Accuracy (`L-Stratified-RTA`).
    LandmarkStratifiedRta,
    /// Distance Consistency (`DSC`); requires `labels`.
    #[cfg(feature = "eval-baselines")]
    #[cfg_attr(docsrs, doc(cfg(feature = "eval-baselines")))]
    DistanceConsistency,
    /// Distribution Consistency (`DC`); requires `labels`.
    #[cfg(feature = "eval-baselines")]
    #[cfg_attr(docsrs, doc(cfg(feature = "eval-baselines")))]
    DistributionConsistency,
    /// Class Density Measure (`CDM`); requires `labels`.
    #[cfg(feature = "eval-baselines")]
    #[cfg_attr(docsrs, doc(cfg(feature = "eval-baselines")))]
    ClassDensityMeasure,
    /// 2D Histogram Density Measure (`2D-HDM`); requires `labels`.
    #[cfg(feature = "eval-baselines")]
    #[cfg_attr(docsrs, doc(cfg(feature = "eval-baselines")))]
    HistogramDensityMeasure,
    /// SepMe $k$-NN graph DIR CPT; requires `labels`.
    #[cfg(feature = "eval-baselines")]
    #[cfg_attr(docsrs, doc(cfg(feature = "eval-baselines")))]
    SepmeKnng,
    /// SepMe Relative Neighborhood Graph DIR CPT; requires `labels`.
    #[cfg(feature = "eval-baselines")]
    #[cfg_attr(docsrs, doc(cfg(feature = "eval-baselines")))]
    SepmeRng,
    /// SepMe Gabriel Graph DIR CPT; requires `labels`.
    #[cfg(feature = "eval-baselines")]
    #[cfg_attr(docsrs, doc(cfg(feature = "eval-baselines")))]
    SepmeGab,
    /// SepMe $\gamma$-Observable Neighbor Graph DIR CPT; requires `labels`.
    #[cfg(feature = "eval-baselines")]
    #[cfg_attr(docsrs, doc(cfg(feature = "eval-baselines")))]
    SepmeGong,
    /// Neighborhood Hit ($M_{\mathrm{NH}}$); requires `labels`.
    #[cfg(feature = "eval-baselines")]
    #[cfg_attr(docsrs, doc(cfg(feature = "eval-baselines")))]
    NeighborhoodHit,
    /// Silhouette coefficient; requires `labels`.
    #[cfg(feature = "eval-baselines")]
    #[cfg_attr(docsrs, doc(cfg(feature = "eval-baselines")))]
    SilhouetteScore,
    /// Calinski-Harabasz Index; requires `labels`.
    #[cfg(feature = "eval-baselines")]
    #[cfg_attr(docsrs, doc(cfg(feature = "eval-baselines")))]
    CalinskiHarabaszScore,
    /// Davies-Bouldin Index; requires `labels`.
    #[cfg(feature = "eval-baselines")]
    #[cfg_attr(docsrs, doc(cfg(feature = "eval-baselines")))]
    DaviesBouldinScore,
    /// $k$-means Cluster Quality (`CQ-kmeans` ARI); requires `labels`.
    #[cfg(feature = "eval-baselines")]
    #[cfg_attr(docsrs, doc(cfg(feature = "eval-baselines")))]
    CqKmeansAri,
    /// $k$-means Cluster Quality (`CQ-kmeans` AMI); requires `labels`.
    #[cfg(feature = "eval-baselines")]
    #[cfg_attr(docsrs, doc(cfg(feature = "eval-baselines")))]
    CqKmeansAmi,
    /// Average-linkage HAC Cluster Quality (`CQ-HAC` Homogeneity); requires `labels`.
    #[cfg(feature = "eval-baselines")]
    #[cfg_attr(docsrs, doc(cfg(feature = "eval-baselines")))]
    CqHacHom,
    /// Average-linkage HAC Cluster Quality (`CQ-HAC` AMI); requires `labels`.
    #[cfg(feature = "eval-baselines")]
    #[cfg_attr(docsrs, doc(cfg(feature = "eval-baselines")))]
    CqHacAmi,
    /// Pairwise Distance-Overlap Error (`PDOE`); requires `labels`.
    #[cfg(feature = "eval-baselines")]
    #[cfg_attr(docsrs, doc(cfg(feature = "eval-baselines")))]
    PairwiseDistanceOverlapError,
    /// Sprawlter-NN composite overlap clutter measure using per-node `sprawlter_extents` or uniform `sprawlter_node_radius`.
    #[cfg(feature = "eval-baselines")]
    #[cfg_attr(docsrs, doc(cfg(feature = "eval-baselines")))]
    SprawlterNn,
    /// Chen et al. (2020) 5-metric Node Overlap Removal (`NOR`) summary;
    /// requires `original_layout` in [`super::MetricInputs`].
    #[cfg(feature = "eval-baselines")]
    #[cfg_attr(docsrs, doc(cfg(feature = "eval-baselines")))]
    NorMetrics,
}

/// Optional cluster/separation/NOR baseline sub-report populated when `eval-baselines` targets
/// are requested in [`super::evaluate_metrics`].
#[non_exhaustive]
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ClusterBaselineReport {
    /// Distance Consistency (`DSC`).
    pub distance_consistency: Option<f64>,
    /// Distribution Consistency (`DC`).
    pub distribution_consistency: Option<f64>,
    /// Class Density Measure (`CDM`).
    pub class_density_measure: Option<f64>,
    /// 2D Histogram Density Measure (`2D-HDM`).
    pub histogram_density_measure: Option<f64>,
    /// SepMe $k$-NN graph DIR CPT.
    pub sepme_knng: Option<f64>,
    /// SepMe Relative Neighborhood Graph DIR CPT.
    pub sepme_rng: Option<f64>,
    /// SepMe Gabriel Graph DIR CPT.
    pub sepme_gab: Option<f64>,
    /// SepMe $\gamma$-Observable Neighbor Graph DIR CPT.
    pub sepme_gong: Option<f64>,
    /// Neighborhood Hit ($M_{\mathrm{NH}}$).
    pub neighborhood_hit: Option<f64>,
    /// Silhouette coefficient.
    pub silhouette_score: Option<f64>,
    /// Calinski-Harabasz Index.
    pub calinski_harabasz_score: Option<f64>,
    /// Davies-Bouldin Index.
    pub davies_bouldin_score: Option<f64>,
    /// $k$-means Cluster Quality ARI.
    pub cq_kmeans_ari: Option<f64>,
    /// $k$-means Cluster Quality AMI.
    pub cq_kmeans_ami: Option<f64>,
    /// Average-linkage HAC Cluster Quality Homogeneity.
    pub cq_hac_hom: Option<f64>,
    /// Average-linkage HAC Cluster Quality AMI.
    pub cq_hac_ami: Option<f64>,
    /// Pairwise Distance-Overlap Error (`PDOE`).
    pub pairwise_distance_overlap_error: Option<f64>,
    /// Sprawlter-NN composite overlap clutter measure.
    pub sprawlter_nn: Option<f64>,
    /// NOR Object Overlap (`OO-NNI`).
    pub nor_oo_nni: Option<f64>,
    /// NOR Spread Minimization (`SP-CH-A`).
    pub nor_sp_ch_a: Option<f64>,
    /// NOR Global Shape (`GS-BB-IAR`).
    pub nor_gs_bb_iar: Option<f64>,
    /// NOR Node Movement (`NM-DM-IMSE`).
    pub nor_nm_dm_imse: Option<f64>,
    /// NOR Edge Length Preservation (`EL-RSD`).
    pub nor_el_rsd: Option<f64>,
}

impl ClusterBaselineReport {
    /// Creates an empty [`ClusterBaselineReport`] with all fields set to `None`.
    pub fn new() -> Self {
        Self::default()
    }
}

/// Selective evaluation report containing `Some(...)` strictly for requested [`MetricTarget`]s.
#[non_exhaustive]
#[derive(Debug, Clone, Default, PartialEq)]
pub struct MetricReport {
    /// Exact closed-form Scale-Normalized Stress (`SNS`).
    pub scale_normalized_stress: Option<f64>,
    /// Pivot-sampled Scale-Normalized Stress (`SNS`).
    pub sampled_scale_normalized_stress: Option<f64>,
    /// Geodesic distortion on `ManifoldLayout2d`.
    pub geodesic_distortion: Option<f64>,
    /// Exact sweep-line edge crossing count.
    pub crossing_number: Option<usize>,
    /// Exact sweep-line list of non-adjacent crossing edge pairs.
    pub crossing_edges: Option<Vec<CrossingPair>>,
    /// Exact sweep-line edge crossing-angle penalty.
    pub crossing_angle: Option<f64>,
    /// Sampled edge crossing count estimate.
    pub sampled_crossing_number: Option<f64>,
    /// Exact crosslessness score.
    pub crosslessness: Option<f64>,
    /// Sampled crosslessness score.
    pub sampled_crosslessness: Option<f64>,
    /// Exact torus edge crossing count.
    pub torus_crossing_number: Option<usize>,
    /// Exact torus crossing edge sub-segment pairs.
    pub crossing_edges_torus: Option<Vec<TorusCrossing>>,
    /// 1-hop neighborhood preservation (`NP1`).
    pub neighborhood_preservation_1hop: Option<f64>,
    /// 2-hop neighborhood preservation (`NP2`).
    pub neighborhood_preservation_2hop: Option<f64>,
    /// Sampled 1-hop neighborhood preservation (`Sampled NP1`).
    pub sampled_np1: Option<f64>,
    /// Sampled 2-hop neighborhood preservation (`Sampled NP2`).
    pub sampled_np2: Option<f64>,
    /// Gabriel graph property violation sum.
    pub gabriel_graph_property: Option<f64>,
    /// One-sided hinge node-resolution violation sum.
    pub node_resolution: Option<f64>,
    /// Unweighted shape-based `S_RNG` Mean Jaccard Similarity.
    pub relative_neighborhood_preservation: Option<f64>,
    /// Unweighted shape-based `S_GG` Mean Jaccard Similarity.
    pub gabriel_graph_preservation: Option<f64>,
    /// PCA singular-value aspect ratio.
    pub aspect_ratio: Option<f64>,
    /// Exponential angular-resolution penalty.
    pub angular_resolution: Option<f64>,
    /// DRGraph Eq. (19) strict minimum-angle resolution (`A2`).
    pub minimum_angle: Option<f64>,
    /// DRGraph Eq. (19) minimum-angle resolution (`A1`).
    pub minimum_angle_a1: Option<f64>,
    /// Scalar layout-degeneracy summary in `[0.0, 1.0]`.
    pub layout_degeneracy: Option<f64>,
    /// Weighted or unit ideal edge-length relative squared error.
    pub ideal_edge_lengths: Option<f64>,
    /// PRISM distance distortion ($\sigma_{\mathrm{dist}}$).
    pub prism_sigma_dist: Option<f64>,
    /// PRISM Procrustes dissimilarity ($\sigma_{\mathrm{disp}}$).
    pub prism_sigma_disp: Option<f64>,
    /// Adjusted Patch Normalized Information Gain.
    pub adjusted_patch_nig: Option<f64>,
    /// Grid-based Cluster Area Faithfulness (`G-CAF`).
    pub cluster_area_faithfulness: Option<f64>,
    /// Landmark-Stratified Random Triplet Accuracy (`L-Stratified-RTA`).
    pub landmark_stratified_rta: Option<f64>,
    baselines: Option<ClusterBaselineReport>,
}

impl MetricReport {
    /// Creates an empty [`MetricReport`] with all fields set to `None`.
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns the cluster/separation/NOR baseline sub-report when `eval-baselines` targets
    /// were requested and evaluated.
    #[inline]
    pub fn baseline_metrics(&self) -> Option<&ClusterBaselineReport> {
        self.baselines.as_ref()
    }

    #[cfg(feature = "eval-baselines")]
    #[inline]
    pub(super) fn set_baselines(&mut self, baselines: ClusterBaselineReport) {
        self.baselines = Some(baselines);
    }
}

#[derive(Default)]
pub(super) struct RequestedFlags {
    pub(super) sns: bool,
    pub(super) sampled_sns: bool,
    pub(super) geodesic_distortion: bool,
    pub(super) crossing_number: bool,
    pub(super) crossing_edges: bool,
    pub(super) crossing_angle: bool,
    pub(super) sampled_crossing_number: bool,
    pub(super) crosslessness: bool,
    pub(super) sampled_crosslessness: bool,
    pub(super) torus_crossing_number: bool,
    pub(super) crossing_edges_torus: bool,
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
    pub(super) prism_distortion: bool,
    pub(super) adjusted_patch_nig: bool,
    pub(super) cluster_area_faithfulness: bool,
    pub(super) landmark_stratified_rta: bool,
    #[cfg(feature = "eval-baselines")]
    pub(super) distance_consistency: bool,
    #[cfg(feature = "eval-baselines")]
    pub(super) distribution_consistency: bool,
    #[cfg(feature = "eval-baselines")]
    pub(super) class_density_measure: bool,
    #[cfg(feature = "eval-baselines")]
    pub(super) histogram_density_measure: bool,
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
    pub(super) silhouette_score: bool,
    #[cfg(feature = "eval-baselines")]
    pub(super) calinski_harabasz_score: bool,
    #[cfg(feature = "eval-baselines")]
    pub(super) davies_bouldin_score: bool,
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
    #[cfg(feature = "eval-baselines")]
    pub(super) nor_metrics: bool,
}

impl RequestedFlags {
    pub(super) fn from_targets(targets: &[MetricTarget]) -> Self {
        let mut flags = Self::default();
        for &target in targets {
            match target {
                MetricTarget::ScaleNormalizedStress => flags.sns = true,
                MetricTarget::SampledScaleNormalizedStress => flags.sampled_sns = true,
                MetricTarget::GeodesicDistortion => flags.geodesic_distortion = true,
                MetricTarget::CrossingNumber => flags.crossing_number = true,
                MetricTarget::CrossingEdges => flags.crossing_edges = true,
                MetricTarget::CrossingAngle => flags.crossing_angle = true,
                MetricTarget::SampledCrossingNumber => flags.sampled_crossing_number = true,
                MetricTarget::Crosslessness => flags.crosslessness = true,
                MetricTarget::SampledCrosslessness => flags.sampled_crosslessness = true,
                MetricTarget::TorusCrossingNumber => flags.torus_crossing_number = true,
                MetricTarget::CrossingEdgesTorus => flags.crossing_edges_torus = true,
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
                MetricTarget::PrismDistortion => flags.prism_distortion = true,
                MetricTarget::AdjustedPatchNig => flags.adjusted_patch_nig = true,
                MetricTarget::ClusterAreaFaithfulness => flags.cluster_area_faithfulness = true,
                MetricTarget::LandmarkStratifiedRta => flags.landmark_stratified_rta = true,
                #[cfg(feature = "eval-baselines")]
                MetricTarget::DistanceConsistency => flags.distance_consistency = true,
                #[cfg(feature = "eval-baselines")]
                MetricTarget::DistributionConsistency => flags.distribution_consistency = true,
                #[cfg(feature = "eval-baselines")]
                MetricTarget::ClassDensityMeasure => flags.class_density_measure = true,
                #[cfg(feature = "eval-baselines")]
                MetricTarget::HistogramDensityMeasure => flags.histogram_density_measure = true,
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
                MetricTarget::SilhouetteScore => flags.silhouette_score = true,
                #[cfg(feature = "eval-baselines")]
                MetricTarget::CalinskiHarabaszScore => flags.calinski_harabasz_score = true,
                #[cfg(feature = "eval-baselines")]
                MetricTarget::DaviesBouldinScore => flags.davies_bouldin_score = true,
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
                #[cfg(feature = "eval-baselines")]
                MetricTarget::NorMetrics => flags.nor_metrics = true,
            }
        }
        flags
    }

    #[cfg(feature = "eval-baselines")]
    #[inline]
    pub(super) fn has_any_baseline(&self) -> bool {
        self.distance_consistency
            || self.distribution_consistency
            || self.class_density_measure
            || self.histogram_density_measure
            || self.sepme_knng
            || self.sepme_rng
            || self.sepme_gab
            || self.sepme_gong
            || self.neighborhood_hit
            || self.silhouette_score
            || self.calinski_harabasz_score
            || self.davies_bouldin_score
            || self.cq_kmeans_ari
            || self.cq_kmeans_ami
            || self.cq_hac_hom
            || self.cq_hac_ami
            || self.pairwise_distance_overlap_error
            || self.sprawlter_nn
            || self.nor_metrics
    }

    #[inline]
    pub(super) fn needs_spatial_index(&self) -> bool {
        let base = self.np1
            || self.np2
            || self.sampled_np1
            || self.sampled_np2
            || self.gabriel_property
            || self.node_resolution
            || self.s_rng
            || self.s_gg
            || self.layout_degeneracy;
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
