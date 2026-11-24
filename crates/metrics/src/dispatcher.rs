//! Target-selective metric evaluation dispatcher avoiding unnecessary computation.

use egraph_core::{CoreError, CoreResult, CsrGraph, Layout2d, Scalar};
use egraph_spatial::StaticKdTree;

use crate::{
    build_kdtree_f64,
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
    /// Shape-based 1-hop vs RNG Mean Jaccard Similarity (`dRNG`).
    RelativeNeighborhoodPreservation,
    /// Shape-based 1-hop vs Gabriel Graph Mean Jaccard Similarity (`dGG`).
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
    /// Landmark-Stratified Random Triplet Accuracy (`L-Stratified-RTA`).
    LandmarkStratifiedRta,
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
    pub landmark_stratified_rta: Option<f64>,
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
    drng: bool,
    dgg: bool,
    aspect_ratio: bool,
    angular_resolution: bool,
    minimum_angle: bool,
    minimum_angle_a1: bool,
    layout_degeneracy: bool,
    ideal_edge_lengths: bool,
    adjusted_patch_nig: bool,
    landmark_stratified_rta: bool,
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
                MetricTarget::RelativeNeighborhoodPreservation => flags.drng = true,
                MetricTarget::GabrielGraphPreservation => flags.dgg = true,
                MetricTarget::AspectRatio => flags.aspect_ratio = true,
                MetricTarget::AngularResolution => flags.angular_resolution = true,
                MetricTarget::MinimumAngle => flags.minimum_angle = true,
                MetricTarget::MinimumAngleA1 => flags.minimum_angle_a1 = true,
                MetricTarget::LayoutDegeneracy => flags.layout_degeneracy = true,
                MetricTarget::IdealEdgeLengths => flags.ideal_edge_lengths = true,
                MetricTarget::AdjustedPatchNig => flags.adjusted_patch_nig = true,
                MetricTarget::LandmarkStratifiedRta => flags.landmark_stratified_rta = true,
            }
        }
        flags
    }

    #[inline]
    fn needs_kdtree(&self) -> bool {
        self.np1
            || self.np2
            || self.sampled_np1
            || self.sampled_np2
            || self.gabriel_property
            || self.node_resolution
            || self.drng
            || self.dgg
    }
}

/// Evaluates strictly the metrics listed in `targets`, skipping all unrequested work.
///
/// Shares a single [`StaticKdTree<2, f64>`] across requested spatial/neighborhood targets,
/// a single sweep-line pass across `CrossingNumber`/`CrossingAngle`/`Crosslessness`,
/// a single sampling pass across `SampledCrossingNumber`/`SampledCrosslessness`, and
/// a single angular scan across `MinimumAngle`/`MinimumAngleA1`.
/// For [`MetricTarget::AdjustedPatchNig`], use [`evaluate_metrics_with_communities`].
pub fn evaluate_metrics<W: Scalar, S: Scalar>(
    graph: &CsrGraph<W>,
    layout: &Layout2d<S>,
    targets: &[MetricTarget],
    config: &MetricConfig,
) -> CoreResult<MetricReport> {
    evaluate_metrics_with_communities(graph, layout, targets, config, None)
}

/// Evaluates strictly the metrics listed in `targets` with optional `community_labels`
/// (required when [`MetricTarget::AdjustedPatchNig`] is requested).
pub fn evaluate_metrics_with_communities<W: Scalar, S: Scalar>(
    graph: &CsrGraph<W>,
    layout: &Layout2d<S>,
    targets: &[MetricTarget],
    config: &MetricConfig,
    community_labels: Option<&[u32]>,
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
        if flags.drng {
            report.relative_neighborhood_preservation = Some(
                relative_neighborhood_preservation_with_tree(graph, layout, tree)?,
            );
        }
        if flags.dgg {
            report.gabriel_graph_preservation =
                Some(gabriel_graph_preservation_with_tree(graph, layout, tree)?);
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

    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        crossing_angle, gabriel_graph_preservation, gabriel_graph_property,
        neighborhood_preservation_1hop, neighborhood_preservation_2hop, node_resolution,
        relative_neighborhood_preservation, sampled_neighborhood_preservation_1hop,
        sampled_neighborhood_preservation_2hop,
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
            full.landmark_stratified_rta,
            Some(landmark_stratified_rta(&g, &layout, 4, 10_000, 1, 9).unwrap())
        );
        assert!(matches!(
            evaluate_metrics(&g, &layout, &[MetricTarget::AdjustedPatchNig], &cfg),
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
    }
}
