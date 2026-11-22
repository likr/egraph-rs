//! Target-selective metric evaluation dispatcher avoiding unnecessary computation.

use egraph_core::{CoreError, CoreResult, CsrGraph, Layout2d, Scalar};
use egraph_spatial::StaticKdTree;

use crate::{
    build_kdtree_f64,
    crossings::{crossing_number, crossing_summary, sampled_crossing_number},
    geometry::{angular_resolution, aspect_ratio, ideal_edge_lengths},
    neighborhood::{
        neighborhood_preservation_1hop_with_tree, neighborhood_preservation_2hop_with_tree,
    },
    proximity::{
        gabriel_graph_preservation_with_tree, gabriel_graph_property_with_tree,
        node_resolution_with_tree, relative_neighborhood_preservation_with_tree,
    },
    stress::{sampled_scale_normalized_stress, scale_normalized_stress},
    validate_graph_layout,
};

/// Individual quality metric selectors passed to [`evaluate_metrics`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MetricTarget {
    /// Exact closed-form Scale-Normalized Stress (`SNS`).
    ScaleNormalizedStress,
    /// $O(kN)$ pivot-sampled Scale-Normalized Stress (`SNS`).
    SampledScaleNormalizedStress,
    /// Exact sweep-line edge crossing count.
    CrossingNumber,
    /// Exact sweep-line edge crossing-angle penalty ($\sum \cos^2(\theta)$).
    CrossingAngle,
    /// $O(\text{num\_samples})$ deterministic sampled crossing count.
    SampledCrossingNumber,
    /// 1-hop neighborhood preservation (`NP1`).
    NeighborhoodPreservation1Hop,
    /// 2-hop neighborhood preservation (`NP2`).
    NeighborhoodPreservation2Hop,
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
    /// Weighted or unit ideal edge-length relative squared error.
    IdealEdgeLengths,
}

/// Configuration parameters for parameterized targets in [`evaluate_metrics`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MetricConfig {
    /// Number of deterministic BFS pivot sources for `SampledScaleNormalizedStress`.
    pub sns_num_pivots: usize,
    /// Explicit RNG seed for `SampledScaleNormalizedStress`.
    pub sns_seed: u64,
    /// Number of edge-pair samples for `SampledCrossingNumber`.
    pub crossing_num_samples: usize,
    /// Explicit RNG seed for `SampledCrossingNumber`.
    pub crossing_seed: u64,
    /// Target edge length fallback for unweighted graphs in `IdealEdgeLengths`.
    pub ideal_edge_length: f64,
}

impl Default for MetricConfig {
    fn default() -> Self {
        Self {
            sns_num_pivots: 32,
            sns_seed: 42,
            crossing_num_samples: 4096,
            crossing_seed: 42,
            ideal_edge_length: 1.0,
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
    pub neighborhood_preservation_1hop: Option<f64>,
    pub neighborhood_preservation_2hop: Option<f64>,
    pub gabriel_graph_property: Option<f64>,
    pub node_resolution: Option<f64>,
    pub relative_neighborhood_preservation: Option<f64>,
    pub gabriel_graph_preservation: Option<f64>,
    pub aspect_ratio: Option<f64>,
    pub angular_resolution: Option<f64>,
    pub ideal_edge_lengths: Option<f64>,
}

#[derive(Default)]
struct RequestedFlags {
    sns: bool,
    sampled_sns: bool,
    crossing_number: bool,
    crossing_angle: bool,
    sampled_crossing_number: bool,
    np1: bool,
    np2: bool,
    gabriel_property: bool,
    node_resolution: bool,
    drng: bool,
    dgg: bool,
    aspect_ratio: bool,
    angular_resolution: bool,
    ideal_edge_lengths: bool,
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
                MetricTarget::NeighborhoodPreservation1Hop => flags.np1 = true,
                MetricTarget::NeighborhoodPreservation2Hop => flags.np2 = true,
                MetricTarget::GabrielGraphProperty => flags.gabriel_property = true,
                MetricTarget::NodeResolution => flags.node_resolution = true,
                MetricTarget::RelativeNeighborhoodPreservation => flags.drng = true,
                MetricTarget::GabrielGraphPreservation => flags.dgg = true,
                MetricTarget::AspectRatio => flags.aspect_ratio = true,
                MetricTarget::AngularResolution => flags.angular_resolution = true,
                MetricTarget::IdealEdgeLengths => flags.ideal_edge_lengths = true,
            }
        }
        flags
    }

    #[inline]
    fn needs_kdtree(&self) -> bool {
        self.np1
            || self.np2
            || self.gabriel_property
            || self.node_resolution
            || self.drng
            || self.dgg
    }
}

/// Evaluates strictly the metrics listed in `targets`, skipping all unrequested work.
///
/// Shares a single [`StaticKdTree<2, f64>`] across requested spatial/neighborhood targets
/// and a single sweep-line pass when both `CrossingNumber` and `CrossingAngle` are requested.
/// Callers requesting e.g. `NP2 + SNS` never pay sweep-line crossing costs.
pub fn evaluate_metrics<W: Scalar, S: Scalar>(
    graph: &CsrGraph<W>,
    layout: &Layout2d<S>,
    targets: &[MetricTarget],
    config: &MetricConfig,
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
    } else if flags.crossing_number {
        report.crossing_number = Some(crossing_number(graph, layout)?);
    }

    if flags.sampled_crossing_number {
        report.sampled_crossing_number = Some(sampled_crossing_number(
            graph,
            layout,
            config.crossing_num_samples,
            config.crossing_seed,
        )?);
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

    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        crossing_angle, gabriel_graph_preservation, gabriel_graph_property,
        neighborhood_preservation_1hop, neighborhood_preservation_2hop, node_resolution,
        relative_neighborhood_preservation,
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
        };

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
        assert!(subset.sampled_scale_normalized_stress.is_none());
        assert!(subset.neighborhood_preservation_1hop.is_none());
        assert!(subset.gabriel_graph_property.is_none());
        assert!(subset.node_resolution.is_none());
        assert!(subset.relative_neighborhood_preservation.is_none());
        assert!(subset.gabriel_graph_preservation.is_none());
        assert!(subset.aspect_ratio.is_none());
        assert!(subset.angular_resolution.is_none());
        assert!(subset.ideal_edge_lengths.is_none());

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
            MetricTarget::NeighborhoodPreservation1Hop,
            MetricTarget::NeighborhoodPreservation2Hop,
            MetricTarget::GabrielGraphProperty,
            MetricTarget::NodeResolution,
            MetricTarget::RelativeNeighborhoodPreservation,
            MetricTarget::GabrielGraphPreservation,
            MetricTarget::AspectRatio,
            MetricTarget::AngularResolution,
            MetricTarget::IdealEdgeLengths,
        ];
        let full = evaluate_metrics(&g, &layout, &all_targets, &cfg).unwrap();
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
            full.neighborhood_preservation_1hop,
            Some(neighborhood_preservation_1hop(&g, &layout).unwrap())
        );
        assert_eq!(
            full.neighborhood_preservation_2hop,
            Some(neighborhood_preservation_2hop(&g, &layout).unwrap())
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
        assert_eq!(
            full.ideal_edge_lengths,
            Some(ideal_edge_lengths(&g, &layout).unwrap())
        );

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
        let rep32 = evaluate_metrics(&g32, &l32, &all_targets, &cfg).unwrap();
        assert_eq!(rep32.crossing_number, Some(0));
        assert!((rep32.aspect_ratio.unwrap() - 1.0).abs() < 1e-6);
        assert!((rep32.ideal_edge_lengths.unwrap()).abs() < 1e-6);
        assert!((rep32.relative_neighborhood_preservation.unwrap() - 1.0).abs() < 1e-6);
        assert!((rep32.gabriel_graph_preservation.unwrap() - 1.0).abs() < 1e-6);
    }
}
