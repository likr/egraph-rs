use std::any::Any;

use egraph_core::{
    CoreError, CoreResult, CsrGraph, EuclideanLayout, EuclideanStorageRef, Layout2d, Layout3d,
    LayoutNd, ManifoldLayout2d, MetricSpace, Scalar,
};
use egraph_metrics_geometry::{
    BfsScratch, bfs_single_source, gabriel_graph_preservation,
    gabriel_graph_preservation_with_index, gabriel_graph_property,
    gabriel_graph_property_with_index, layout_degeneracy, layout_degeneracy_with_index,
    neighborhood_preservation_1hop, neighborhood_preservation_1hop_with_index,
    neighborhood_preservation_2hop, neighborhood_preservation_2hop_with_index, node_resolution,
    node_resolution_with_index, relative_neighborhood_preservation,
    relative_neighborhood_preservation_with_index, sampled_neighborhood_preservation_1hop,
    sampled_neighborhood_preservation_1hop_with_index, sampled_neighborhood_preservation_2hop,
    sampled_neighborhood_preservation_2hop_with_index,
};
use egraph_spatial_trees::{DynamicKdTree, NearestNeighborIndex, StaticKdTree};

use super::config::MetricConfig;
use super::report::{MetricReport, RequestedFlags};

/// Classified concrete layout space for unified target dispatch.
pub(super) enum SpaceClass<'a, S: Scalar> {
    Euclidean2d(&'a Layout2d<S>),
    Euclidean3d(&'a Layout3d<S>),
    EuclideanNd(&'a LayoutNd<S>),
    Manifold(&'a ManifoldLayout2d<S>),
    ExternalMetric,
}

impl<'a, S: Scalar> SpaceClass<'a, S> {
    pub(super) fn classify<L: MetricSpace<Scalar = S> + 'static>(layout: &'a L) -> Self {
        let any: &dyn Any = layout;
        if let Some(l2) = any.downcast_ref::<Layout2d<S>>() {
            return Self::Euclidean2d(l2);
        }
        if let Some(l3) = any.downcast_ref::<Layout3d<S>>() {
            return Self::Euclidean3d(l3);
        }
        if let Some(lnd) = any.downcast_ref::<LayoutNd<S>>() {
            return Self::EuclideanNd(lnd);
        }
        if let Some(ml) = any.downcast_ref::<ManifoldLayout2d<S>>() {
            return Self::Manifold(ml);
        }
        Self::ExternalMetric
    }
}

fn build_spatial_index<L: EuclideanLayout + ?Sized>(
    layout: &L,
) -> CoreResult<Box<dyn NearestNeighborIndex>> {
    layout.validate_finite()?;
    match layout.as_euclidean_storage() {
        EuclideanStorageRef::Soa2d { .. } => Ok(Box::new(
            StaticKdTree::<2, L::Scalar>::from_euclidean(layout)?,
        )),
        EuclideanStorageRef::Soa3d { .. } => Ok(Box::new(
            StaticKdTree::<3, L::Scalar>::from_euclidean(layout)?,
        )),
        EuclideanStorageRef::FlatNd { dims, .. } => match dims {
            1 => Ok(Box::new(StaticKdTree::<1, L::Scalar>::from_euclidean(
                layout,
            )?)),
            2 => Ok(Box::new(StaticKdTree::<2, L::Scalar>::from_euclidean(
                layout,
            )?)),
            3 => Ok(Box::new(StaticKdTree::<3, L::Scalar>::from_euclidean(
                layout,
            )?)),
            4 => Ok(Box::new(StaticKdTree::<4, L::Scalar>::from_euclidean(
                layout,
            )?)),
            5 => Ok(Box::new(StaticKdTree::<5, L::Scalar>::from_euclidean(
                layout,
            )?)),
            6 => Ok(Box::new(StaticKdTree::<6, L::Scalar>::from_euclidean(
                layout,
            )?)),
            7 => Ok(Box::new(StaticKdTree::<7, L::Scalar>::from_euclidean(
                layout,
            )?)),
            8 => Ok(Box::new(StaticKdTree::<8, L::Scalar>::from_euclidean(
                layout,
            )?)),
            _ => Ok(Box::new(DynamicKdTree::<L::Scalar>::from_euclidean(
                layout,
            )?)),
        },
    }
}

pub(super) fn try_build_spatial_index<S: Scalar>(
    space: &SpaceClass<'_, S>,
) -> CoreResult<Option<Box<dyn NearestNeighborIndex>>> {
    match space {
        SpaceClass::Euclidean2d(l) => build_spatial_index(*l).map(Some),
        SpaceClass::Euclidean3d(l) => build_spatial_index(*l).map(Some),
        SpaceClass::EuclideanNd(l) => build_spatial_index(*l).map(Some),
        SpaceClass::Manifold(_) | SpaceClass::ExternalMetric => Ok(None),
    }
}

#[inline]
fn degeneracy_scalar<W, L: MetricSpace + 'static>(
    graph: &CsrGraph<W>,
    layout: &L,
    deg: egraph_metrics_geometry::LayoutDegeneracy,
) -> f64 {
    let _ = graph;
    let bbox_penalty = if deg.is_bbox_degenerate && layout.len() > 1 {
        1.0
    } else {
        0.0
    };
    deg.duplicate_ratio_f32
        .max(deg.zero_length_edge_ratio)
        .max(bbox_penalty)
}

fn evaluate_euclidean_spatial_targets<W, E: EuclideanLayout + 'static>(
    graph: &CsrGraph<W>,
    layout: &E,
    index: &dyn NearestNeighborIndex,
    flags: &RequestedFlags,
    config: &MetricConfig,
    report: &mut MetricReport,
) -> CoreResult<()> {
    if flags.np1 {
        report.neighborhood_preservation_1hop = Some(neighborhood_preservation_1hop_with_index(
            graph, layout, index,
        )?);
    }
    if flags.np2 {
        report.neighborhood_preservation_2hop = Some(neighborhood_preservation_2hop_with_index(
            graph, layout, index,
        )?);
    }
    if flags.sampled_np1 {
        report.sampled_np1 = Some(sampled_neighborhood_preservation_1hop_with_index(
            graph,
            layout,
            index,
            config.np_num_samples,
            config.np_seed,
        )?);
    }
    if flags.sampled_np2 {
        report.sampled_np2 = Some(sampled_neighborhood_preservation_2hop_with_index(
            graph,
            layout,
            index,
            config.np_num_samples,
            config.np_seed,
        )?);
    }
    if flags.gabriel_property {
        report.gabriel_graph_property =
            Some(gabriel_graph_property_with_index(graph, layout, index)?);
    }
    if flags.node_resolution {
        report.node_resolution = Some(node_resolution_with_index(layout, index)?);
    }
    if flags.s_rng {
        report.relative_neighborhood_preservation = Some(
            relative_neighborhood_preservation_with_index(graph, layout, index)?,
        );
    }
    if flags.s_gg {
        report.gabriel_graph_preservation =
            Some(gabriel_graph_preservation_with_index(graph, layout, index)?);
    }
    if flags.layout_degeneracy {
        let deg = layout_degeneracy_with_index(graph, layout, index)?;
        report.layout_degeneracy = Some(degeneracy_scalar(graph, layout, deg));
    }
    Ok(())
}

fn evaluate_metric_spatial_targets<W, L: MetricSpace + 'static>(
    graph: &CsrGraph<W>,
    layout: &L,
    flags: &RequestedFlags,
    config: &MetricConfig,
    report: &mut MetricReport,
) -> CoreResult<()> {
    if flags.np1 {
        report.neighborhood_preservation_1hop =
            Some(neighborhood_preservation_1hop(graph, layout)?);
    }
    if flags.np2 {
        report.neighborhood_preservation_2hop =
            Some(neighborhood_preservation_2hop(graph, layout)?);
    }
    if flags.sampled_np1 {
        report.sampled_np1 = Some(sampled_neighborhood_preservation_1hop(
            graph,
            layout,
            config.np_num_samples,
            config.np_seed,
        )?);
    }
    if flags.sampled_np2 {
        report.sampled_np2 = Some(sampled_neighborhood_preservation_2hop(
            graph,
            layout,
            config.np_num_samples,
            config.np_seed,
        )?);
    }
    if flags.gabriel_property {
        report.gabriel_graph_property = Some(gabriel_graph_property(graph, layout)?);
    }
    if flags.node_resolution {
        report.node_resolution = Some(node_resolution(layout)?);
    }
    if flags.s_rng {
        report.relative_neighborhood_preservation =
            Some(relative_neighborhood_preservation(graph, layout)?);
    }
    if flags.s_gg {
        report.gabriel_graph_preservation = Some(gabriel_graph_preservation(graph, layout)?);
    }
    if flags.layout_degeneracy {
        let deg = layout_degeneracy(graph, layout)?;
        report.layout_degeneracy = Some(degeneracy_scalar(graph, layout, deg));
    }
    Ok(())
}

pub(super) fn evaluate_spatial_and_degeneracy_targets<W, L: MetricSpace + 'static>(
    graph: &CsrGraph<W>,
    layout: &L,
    space: &SpaceClass<'_, L::Scalar>,
    spatial_index: Option<&dyn NearestNeighborIndex>,
    flags: &RequestedFlags,
    config: &MetricConfig,
    report: &mut MetricReport,
) -> CoreResult<()> {
    if let Some(index) = spatial_index {
        match space {
            SpaceClass::Euclidean2d(l2) => {
                return evaluate_euclidean_spatial_targets(
                    graph, *l2, index, flags, config, report,
                );
            }
            SpaceClass::Euclidean3d(l3) => {
                return evaluate_euclidean_spatial_targets(
                    graph, *l3, index, flags, config, report,
                );
            }
            SpaceClass::EuclideanNd(lnd) => {
                return evaluate_euclidean_spatial_targets(
                    graph, *lnd, index, flags, config, report,
                );
            }
            SpaceClass::Manifold(_) | SpaceClass::ExternalMetric => {}
        }
    }
    evaluate_metric_spatial_targets(graph, layout, flags, config, report)
}

/// Evaluates unweighted BFS geodesic distortion for `CsrGraph<W>` without `W: Scalar`.
pub(super) fn geodesic_distortion_unweighted<W, L: MetricSpace + ?Sized>(
    graph: &CsrGraph<W>,
    layout: &L,
) -> CoreResult<f64> {
    if graph.is_weighted() {
        return Err(CoreError::unsupported(
            "GeodesicDistortion on weighted graphs requires f32 or f64 edge weights",
        ));
    }
    let n = layout.len();
    if n <= 1 {
        return Ok(0.0);
    }
    let directed = graph.is_directed();
    let mut scratch = BfsScratch::new(n);
    let mut sum_rel_err = 0.0_f64;
    let mut pair_count = 0usize;

    for i in 0..n {
        let dists = bfs_single_source(graph, i, 1.0, &mut scratch)?;
        let j_start = if directed { 0 } else { i + 1 };
        for (j, &d) in dists.iter().enumerate().skip(j_start) {
            if j == i || !d.is_finite() || d <= 0.0 {
                continue;
            }
            let r = layout.distance(i, j);
            sum_rel_err += (r - d).abs() / d;
            pair_count += 1;
        }
    }

    if pair_count == 0 {
        Ok(0.0)
    } else {
        Ok(sum_rel_err / pair_count as f64)
    }
}
