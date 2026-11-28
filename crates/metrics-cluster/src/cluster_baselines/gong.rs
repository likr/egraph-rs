use std::any::Any;

use egraph_core::{
    CoreError, CoreResult, EuclideanLayout, Layout2d, Layout3d, LayoutNd, ManifoldLayout2d,
    MetricSpace, Scalar,
};
use egraph_metrics_geometry::{gabriel_graph_edges, gabriel_graph_edges_with_index};
use egraph_spatial_trees::{KdQueryScratch, NearestNeighborIndex};

use super::compact_labels;
use super::manifold_ops::{manifold_geodesic_interpolate_point, manifold_point_dist};
use super::separation::dir_cpt_from_degree_counts;
use super::space_dispatch::{
    copy_euclidean_coords_f64, dist_sq_f64, try_build_spatial_index, validate_index_layout,
    validate_metric_labels,
};

fn validate_gong_gamma(gamma: f64) -> CoreResult<()> {
    if !gamma.is_finite() {
        return Err(CoreError::non_finite("sepme_gong::gamma"));
    }
    if !(0.0..=1.0).contains(&gamma) {
        return Err(CoreError::invalid_input(
            "sepme_gong requires gamma in [0.0, 1.0]",
        ));
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn is_gong_directed_edge_euclidean<I: NearestNeighborIndex + ?Sized>(
    i: usize,
    j: usize,
    gamma: f64,
    coords: &[f64],
    dims: usize,
    index: &I,
    mid_buf: &mut [f64],
    scratch: &mut KdQueryScratch,
    witness_buf: &mut Vec<(usize, f64)>,
) -> CoreResult<bool> {
    let pi = &coords[i * dims..(i + 1) * dims];
    let pj = &coords[j * dims..(j + 1) * dims];
    let d_ij_sq = dist_sq_f64(pi, pj);
    if d_ij_sq <= 0.0 {
        return Ok(false);
    }
    let one_minus_g = 1.0 - gamma;
    for axis in 0..dims {
        mid_buf[axis] = gamma * pj[axis] + one_minus_g * pi[axis];
    }
    let r_sq = one_minus_g * one_minus_g * d_ij_sq;
    let query_r_sq = (r_sq * (1.0 + 1e-11)).min(f64::MAX);
    index.query_radius_sq_with_scratch(mid_buf, query_r_sq, scratch, witness_buf)?;

    let block_thresh = r_sq * (1.0 + 1e-12);
    for &(q, d_mq_sq) in witness_buf.iter() {
        if q == i || q == j {
            continue;
        }
        if d_mq_sq <= block_thresh {
            return Ok(false);
        }
    }
    Ok(true)
}

fn sepme_gong_euclidean_inner<L: EuclideanLayout + ?Sized, I: NearestNeighborIndex + ?Sized>(
    layout: &L,
    labels: &[u32],
    gamma: f64,
    index: &I,
) -> CoreResult<f64> {
    let n = layout.len();
    let (coords, dims) = copy_euclidean_coords_f64(layout);
    let (comp, class_sizes, k_classes) = compact_labels(labels);
    let mut deg_out = vec![0usize; n];
    let mut same_out = vec![0usize; n];
    let mut mid_buf = vec![0.0_f64; dims];
    let mut scratch = KdQueryScratch::with_capacity(0);
    let mut witness_buf = Vec::new();

    if gamma <= 0.5 {
        let gg_edges = gabriel_graph_edges_with_index(layout, index)?;
        for (u, v) in gg_edges {
            let ui = u as usize;
            let vi = v as usize;
            let same_class = comp[ui] == comp[vi];
            if is_gong_directed_edge_euclidean(
                ui,
                vi,
                gamma,
                &coords,
                dims,
                index,
                &mut mid_buf,
                &mut scratch,
                &mut witness_buf,
            )? {
                deg_out[ui] += 1;
                if same_class {
                    same_out[ui] += 1;
                }
            }
            if is_gong_directed_edge_euclidean(
                vi,
                ui,
                gamma,
                &coords,
                dims,
                index,
                &mut mid_buf,
                &mut scratch,
                &mut witness_buf,
            )? {
                deg_out[vi] += 1;
                if same_class {
                    same_out[vi] += 1;
                }
            }
        }
    } else {
        for i in 0..n {
            for j in 0..n {
                if i == j {
                    continue;
                }
                if is_gong_directed_edge_euclidean(
                    i,
                    j,
                    gamma,
                    &coords,
                    dims,
                    index,
                    &mut mid_buf,
                    &mut scratch,
                    &mut witness_buf,
                )? {
                    deg_out[i] += 1;
                    if comp[i] == comp[j] {
                        same_out[i] += 1;
                    }
                }
            }
        }
    }

    Ok(dir_cpt_from_degree_counts(
        &comp,
        &class_sizes,
        k_classes,
        &deg_out,
        &same_out,
    ))
}

fn is_gong_directed_edge_manifold<S: Scalar>(
    layout: &ManifoldLayout2d<S>,
    i: usize,
    j: usize,
    gamma: f64,
) -> CoreResult<bool> {
    let d_ij = layout.distance(i, j);
    if d_ij <= 0.0 {
        return Ok(false);
    }
    let mid = manifold_geodesic_interpolate_point(layout, i, j, gamma)?;
    let r = (1.0 - gamma) * d_ij;
    let block_thresh = r * (1.0 + 1e-12);
    let kind = layout.kind();
    let metric = layout.metric();
    let xs = layout.x();
    let ys = layout.y();
    let n = layout.len();
    for q in 0..n {
        if q == i || q == j {
            continue;
        }
        let d_mq = manifold_point_dist(kind, &metric, mid, (xs[q].to_f64(), ys[q].to_f64()))?;
        if d_mq <= block_thresh {
            return Ok(false);
        }
    }
    Ok(true)
}

fn sepme_gong_manifold_inner<S: Scalar>(
    layout: &ManifoldLayout2d<S>,
    labels: &[u32],
    gamma: f64,
) -> CoreResult<f64> {
    let n = layout.len();
    let (comp, class_sizes, k_classes) = compact_labels(labels);
    let mut deg_out = vec![0usize; n];
    let mut same_out = vec![0usize; n];

    if gamma <= 0.5 {
        let gg_edges = gabriel_graph_edges(layout)?;
        for (u, v) in gg_edges {
            let ui = u as usize;
            let vi = v as usize;
            let same_class = comp[ui] == comp[vi];
            if is_gong_directed_edge_manifold(layout, ui, vi, gamma)? {
                deg_out[ui] += 1;
                if same_class {
                    same_out[ui] += 1;
                }
            }
            if is_gong_directed_edge_manifold(layout, vi, ui, gamma)? {
                deg_out[vi] += 1;
                if same_class {
                    same_out[vi] += 1;
                }
            }
        }
    } else {
        for i in 0..n {
            for j in 0..n {
                if i == j {
                    continue;
                }
                if is_gong_directed_edge_manifold(layout, i, j, gamma)? {
                    deg_out[i] += 1;
                    if comp[i] == comp[j] {
                        same_out[i] += 1;
                    }
                }
            }
        }
    }

    Ok(dir_cpt_from_degree_counts(
        &comp,
        &class_sizes,
        k_classes,
        &deg_out,
        &same_out,
    ))
}

fn is_gong_directed_edge_metric(i: usize, j: usize, gamma: f64, n: usize, dist_sq: &[f64]) -> bool {
    let d_ij_sq = dist_sq[i * n + j];
    if d_ij_sq <= 0.0 {
        return false;
    }
    let one_minus_g = 1.0 - gamma;
    let block_thresh = one_minus_g * one_minus_g * d_ij_sq * (1.0 + 1e-12);
    let row_i = &dist_sq[i * n..(i + 1) * n];
    let row_j = &dist_sq[j * n..(j + 1) * n];
    for q in 0..n {
        if q == i || q == j {
            continue;
        }
        // Stewart's theorem gives exact squared distance to `(1 - gamma) * p_i + gamma * p_j`.
        let d_mq_sq =
            (one_minus_g * row_i[q] + gamma * row_j[q] - gamma * one_minus_g * d_ij_sq).max(0.0);
        if d_mq_sq <= block_thresh {
            return false;
        }
    }
    true
}

fn sepme_gong_metric_inner<L: MetricSpace + 'static>(
    layout: &L,
    labels: &[u32],
    gamma: f64,
) -> CoreResult<f64> {
    let n = layout.len();
    let mut dist_sq = vec![0.0_f64; n * n];
    for i in 0..n {
        for j in (i + 1)..n {
            let d = layout.distance(i, j);
            let d2 = d * d;
            dist_sq[i * n + j] = d2;
            dist_sq[j * n + i] = d2;
        }
    }

    let (comp, class_sizes, k_classes) = compact_labels(labels);
    let mut deg_out = vec![0usize; n];
    let mut same_out = vec![0usize; n];

    if gamma <= 0.5 {
        let gg_edges = gabriel_graph_edges(layout)?;
        for (u, v) in gg_edges {
            let ui = u as usize;
            let vi = v as usize;
            let same_class = comp[ui] == comp[vi];
            if is_gong_directed_edge_metric(ui, vi, gamma, n, &dist_sq) {
                deg_out[ui] += 1;
                if same_class {
                    same_out[ui] += 1;
                }
            }
            if is_gong_directed_edge_metric(vi, ui, gamma, n, &dist_sq) {
                deg_out[vi] += 1;
                if same_class {
                    same_out[vi] += 1;
                }
            }
        }
    } else {
        for i in 0..n {
            for j in 0..n {
                if i == j {
                    continue;
                }
                if is_gong_directed_edge_metric(i, j, gamma, n, &dist_sq) {
                    deg_out[i] += 1;
                    if comp[i] == comp[j] {
                        same_out[i] += 1;
                    }
                }
            }
        }
    }

    Ok(dir_cpt_from_degree_counts(
        &comp,
        &class_sizes,
        k_classes,
        &deg_out,
        &same_out,
    ))
}

/// Computes Aupetit & Sedlmair (2016) `SepMe` $\gamma$-Observable Neighbor Graph (`GONG DIR CPT`,
/// default `gamma = 0.35`) in `[0.0, 1.0]` across any [`MetricSpace`].
///
/// For `0.0 <= gamma <= 0.5`, prunes candidates via `gabriel_graph_edges` since
/// $\mathrm{GONG}_\gamma \subseteq \mathrm{GG}$ holds strictly on `[0, 0.5]`. Automatically
/// accelerates [`EuclideanLayout`] inputs via [`NearestNeighborIndex`] and uses geodesic
/// interpolation on [`ManifoldLayout2d`].
pub fn sepme_gong<L: MetricSpace + 'static>(
    layout: &L,
    labels: &[u32],
    gamma: f64,
) -> CoreResult<f64> {
    validate_gong_gamma(gamma)?;
    validate_metric_labels(layout, labels, "sepme_gong")?;
    let n = layout.len();
    if n == 0 {
        return Ok(0.0);
    }
    if n == 1 {
        return Ok(1.0);
    }

    if let Some(index) = try_build_spatial_index(layout)? {
        let any: &dyn Any = layout;
        if let Some(l) = any.downcast_ref::<Layout2d<f64>>() {
            return sepme_gong_euclidean_inner(l, labels, gamma, index.as_ref());
        }
        if let Some(l) = any.downcast_ref::<Layout2d<f32>>() {
            return sepme_gong_euclidean_inner(l, labels, gamma, index.as_ref());
        }
        if let Some(l) = any.downcast_ref::<Layout3d<f64>>() {
            return sepme_gong_euclidean_inner(l, labels, gamma, index.as_ref());
        }
        if let Some(l) = any.downcast_ref::<Layout3d<f32>>() {
            return sepme_gong_euclidean_inner(l, labels, gamma, index.as_ref());
        }
        if let Some(l) = any.downcast_ref::<LayoutNd<f64>>() {
            return sepme_gong_euclidean_inner(l, labels, gamma, index.as_ref());
        }
        if let Some(l) = any.downcast_ref::<LayoutNd<f32>>() {
            return sepme_gong_euclidean_inner(l, labels, gamma, index.as_ref());
        }
    }

    let any: &dyn Any = layout;
    if let Some(ml) = any.downcast_ref::<ManifoldLayout2d<f64>>() {
        return sepme_gong_manifold_inner(ml, labels, gamma);
    }
    if let Some(ml) = any.downcast_ref::<ManifoldLayout2d<f32>>() {
        return sepme_gong_manifold_inner(ml, labels, gamma);
    }

    sepme_gong_metric_inner(layout, labels, gamma)
}

/// Computes [`sepme_gong`] reusing a prebuilt [`NearestNeighborIndex`].
pub fn sepme_gong_with_index<L: EuclideanLayout + ?Sized, I: NearestNeighborIndex + ?Sized>(
    layout: &L,
    labels: &[u32],
    gamma: f64,
    index: &I,
) -> CoreResult<f64> {
    validate_gong_gamma(gamma)?;
    validate_metric_labels(layout, labels, "sepme_gong_with_index")?;
    validate_index_layout(index, layout, "sepme_gong_with_index")?;

    let n = layout.len();
    if n == 0 {
        return Ok(0.0);
    }
    if n == 1 {
        return Ok(1.0);
    }
    sepme_gong_euclidean_inner(layout, labels, gamma, index)
}
