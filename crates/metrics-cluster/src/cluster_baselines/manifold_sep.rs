use egraph_core::{CoreError, CoreResult, ManifoldLayout2d, Scalar};
use egraph_metrics_geometry::{
    neighborhood::k_nearest_manifold_excluding_self_into,
    proximity::{
        gabriel_graph_edges_manifold, manifold_geodesic_interpolate_point,
        manifold_point_distance_to_node, relative_neighborhood_edges_manifold,
    },
};

use super::compact_labels;
use super::manifold_centroid::validate_manifold_labels;
use super::separation::{dir_cpt_from_degree_counts, dir_cpt_from_undirected_edges};

/// Computes geodesic Neighborhood Hit (`M_NH`) on a [`ManifoldLayout2d`] in `[0.0, 1.0]`.
pub fn neighborhood_hit_manifold<S: Scalar>(
    layout: &ManifoldLayout2d<S>,
    labels: &[u32],
    k: usize,
) -> CoreResult<f64> {
    if k == 0 {
        return Err(CoreError::invalid_input(
            "neighborhood_hit_manifold requires k >= 1",
        ));
    }
    validate_manifold_labels(layout, labels, "neighborhood_hit_manifold")?;
    let n = layout.len();
    if n < 2 {
        return Ok(0.0);
    }

    let k_eff = k.min(n - 1);
    let mut nbrs = Vec::with_capacity(k_eff);
    let mut total_same = 0usize;

    for i in 0..n {
        k_nearest_manifold_excluding_self_into(layout, i, k_eff, &mut nbrs)?;
        let li = labels[i];
        total_same += nbrs.iter().filter(|&&(j, _)| labels[j] == li).count();
    }
    Ok(total_same as f64 / (k_eff * n) as f64)
}

/// Computes geodesic `SepMe` $k$-Nearest Neighbor Graph (`KNNG DIR CPT`) on a
/// [`ManifoldLayout2d`] in `[0.0, 1.0]`.
pub fn sepme_knng_manifold<S: Scalar>(
    layout: &ManifoldLayout2d<S>,
    labels: &[u32],
    k: usize,
) -> CoreResult<f64> {
    if k == 0 {
        return Err(CoreError::invalid_input(
            "sepme_knng_manifold requires k >= 1",
        ));
    }
    validate_manifold_labels(layout, labels, "sepme_knng_manifold")?;
    let n = layout.len();
    if n == 0 {
        return Ok(0.0);
    }
    if n == 1 {
        return Ok(1.0);
    }

    let (comp, class_sizes, k_classes) = compact_labels(labels);
    let k_eff = k.min(n - 1);
    let mut nbrs = Vec::with_capacity(k_eff);
    let mut deg = vec![0usize; n];
    let mut same = vec![0usize; n];

    for i in 0..n {
        k_nearest_manifold_excluding_self_into(layout, i, k_eff, &mut nbrs)?;
        deg[i] = nbrs.len();
        let ci = comp[i];
        same[i] = nbrs.iter().filter(|&&(j, _)| comp[j] == ci).count();
    }

    Ok(dir_cpt_from_degree_counts(
        &comp,
        &class_sizes,
        k_classes,
        &deg,
        &same,
    ))
}

/// Computes geodesic `SepMe` Relative Neighborhood Graph (`RNG DIR CPT`) on a
/// [`ManifoldLayout2d`] in `[0.0, 1.0]`.
pub fn sepme_rng_manifold<S: Scalar>(
    layout: &ManifoldLayout2d<S>,
    labels: &[u32],
) -> CoreResult<f64> {
    validate_manifold_labels(layout, labels, "sepme_rng_manifold")?;
    let n = layout.len();
    if n == 0 {
        return Ok(0.0);
    }
    if n == 1 {
        return Ok(1.0);
    }
    let (comp, class_sizes, k_classes) = compact_labels(labels);
    let edges = relative_neighborhood_edges_manifold(layout)?;
    Ok(dir_cpt_from_undirected_edges(
        n,
        &comp,
        &class_sizes,
        k_classes,
        &edges,
    ))
}

/// Computes geodesic `SepMe` Gabriel Graph (`GAB DIR CPT`) on a [`ManifoldLayout2d`] in
/// `[0.0, 1.0]`.
pub fn sepme_gab_manifold<S: Scalar>(
    layout: &ManifoldLayout2d<S>,
    labels: &[u32],
) -> CoreResult<f64> {
    validate_manifold_labels(layout, labels, "sepme_gab_manifold")?;
    let n = layout.len();
    if n == 0 {
        return Ok(0.0);
    }
    if n == 1 {
        return Ok(1.0);
    }
    let (comp, class_sizes, k_classes) = compact_labels(labels);
    let edges = gabriel_graph_edges_manifold(layout)?;
    Ok(dir_cpt_from_undirected_edges(
        n,
        &comp,
        &class_sizes,
        k_classes,
        &edges,
    ))
}

#[inline]
fn is_gong_directed_edge_manifold<S: Scalar>(
    layout: &ManifoldLayout2d<S>,
    i: usize,
    j: usize,
    gamma: f64,
) -> CoreResult<bool> {
    let d_ij = layout.geodesic_distance(i, j)?.to_f64();
    if d_ij <= 0.0 {
        return Ok(false);
    }
    let mid = manifold_geodesic_interpolate_point(layout, i, j, gamma)?;
    let r = (1.0 - gamma) * d_ij;
    let block_thresh = r * (1.0 + 1e-12);
    let n = layout.len();
    for q in 0..n {
        if q == i || q == j {
            continue;
        }
        if manifold_point_distance_to_node(layout, mid, q)? <= block_thresh {
            return Ok(false);
        }
    }
    Ok(true)
}

/// Computes geodesic `SepMe` $\gamma$-Observable Neighbor Graph (`GONG DIR CPT`, default
/// `gamma = 0.35`) on a [`ManifoldLayout2d`] in `[0.0, 1.0]`.
pub fn sepme_gong_manifold<S: Scalar>(
    layout: &ManifoldLayout2d<S>,
    labels: &[u32],
    gamma: f64,
) -> CoreResult<f64> {
    if !gamma.is_finite() {
        return Err(CoreError::non_finite("sepme_gong_manifold::gamma"));
    }
    if !(0.0..=1.0).contains(&gamma) {
        return Err(CoreError::invalid_input(
            "sepme_gong_manifold requires gamma in [0.0, 1.0]",
        ));
    }
    validate_manifold_labels(layout, labels, "sepme_gong_manifold")?;
    let n = layout.len();
    if n == 0 {
        return Ok(0.0);
    }
    if n == 1 {
        return Ok(1.0);
    }

    let (comp, class_sizes, k_classes) = compact_labels(labels);
    let mut deg_out = vec![0usize; n];
    let mut same_out = vec![0usize; n];

    if gamma <= 0.5 {
        let gg_edges = gabriel_graph_edges_manifold(layout)?;
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
