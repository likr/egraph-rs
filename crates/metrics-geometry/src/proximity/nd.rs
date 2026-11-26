use egraph_core::{CoreResult, CsrGraph, Layout2d, LayoutNd, Scalar};
use egraph_spatial_trees::KdQueryScratch;

use crate::extract_unique_edges;
use crate::nd_index::{
    NdSpatialIndex, build_nd_spatial_index, dist_sq_f64, euclidean_diameter_nd, slice_dist_nd,
    validate_graph_layout_nd, validate_index_layout_nd, validate_layout_nd,
};

use super::sector_pruning::{ProximityMode, proximity_edges_with_tree};
use super::{
    gabriel_graph_property_with_tree, mean_jaccard_graph_vs_edges, node_resolution_with_tree,
};

/// Evaluates violations of the Gabriel graph condition over N-D `layout` and `graph`.
pub fn gabriel_graph_property_nd<W, S: Scalar>(
    graph: &CsrGraph<W>,
    layout: &LayoutNd<S>,
) -> CoreResult<f64> {
    validate_graph_layout_nd(graph, layout)?;
    if layout.num_nodes() <= 2 || graph.num_edges() == 0 {
        return Ok(0.0);
    }
    let index = build_nd_spatial_index(layout)?;
    gabriel_graph_property_nd_with_index(graph, layout, &index)
}

/// Computes [`gabriel_graph_property_nd`] reusing a prebuilt [`NdSpatialIndex`].
#[doc(hidden)]
pub fn gabriel_graph_property_nd_with_index<W, S: Scalar>(
    graph: &CsrGraph<W>,
    layout: &LayoutNd<S>,
    index: &NdSpatialIndex,
) -> CoreResult<f64> {
    validate_graph_layout_nd(graph, layout)?;
    validate_index_layout_nd(index, layout, "gabriel_graph_property_nd_with_index")?;
    let n = layout.num_nodes();
    if n <= 2 || graph.num_edges() == 0 {
        return Ok(0.0);
    }
    let dims = layout.dims();
    if dims == 2
        && let NdSpatialIndex::D2(tree) = index
    {
        let layout2d = Layout2d::from_interleaved(layout.coords())?;
        return gabriel_graph_property_with_tree(graph, &layout2d, tree);
    }

    let coords = layout.coords();
    let edges = extract_unique_edges(graph);
    let mut mid = vec![0.0_f64; dims];
    let mut disk_buf: Vec<(usize, f64)> = Vec::new();
    let mut violation_sum = 0.0_f64;

    for (u, v) in edges {
        let ui = u as usize;
        let vi = v as usize;
        let pu = &coords[ui * dims..(ui + 1) * dims];
        let pv = &coords[vi * dims..(vi + 1) * dims];
        let r = 0.5 * slice_dist_nd(pu, pv);
        if r <= 0.0 {
            continue;
        }
        for d in 0..dims {
            mid[d] = 0.5 * (pu[d].to_f64() + pv[d].to_f64());
        }
        index.within_radius_into(&mid, r, &mut disk_buf)?;
        for &(idx, d_sq) in &disk_buf {
            if idx == ui || idx == vi {
                continue;
            }
            let dist = d_sq.sqrt();
            if dist < r {
                let viol = r - dist;
                violation_sum += viol * viol;
            }
        }
    }

    Ok(violation_sum)
}

/// Computes the one-sided hinge node-resolution violation penalty for an N-D [`LayoutNd`].
pub fn node_resolution_nd<S: Scalar>(layout: &LayoutNd<S>) -> CoreResult<f64> {
    validate_layout_nd(layout)?;
    let n = layout.num_nodes();
    if n <= 1 {
        return Ok(0.0);
    }
    let index = build_nd_spatial_index(layout)?;
    node_resolution_nd_with_index(layout, &index)
}

/// Computes [`node_resolution_nd`] reusing a prebuilt [`NdSpatialIndex`].
#[doc(hidden)]
pub fn node_resolution_nd_with_index<S: Scalar>(
    layout: &LayoutNd<S>,
    index: &NdSpatialIndex,
) -> CoreResult<f64> {
    validate_layout_nd(layout)?;
    validate_index_layout_nd(index, layout, "node_resolution_nd_with_index")?;
    let n = layout.num_nodes();
    if n <= 1 {
        return Ok(0.0);
    }
    let dims = layout.dims();
    if dims == 2
        && let NdSpatialIndex::D2(tree) = index
    {
        let layout2d = Layout2d::from_interleaved(layout.coords())?;
        return node_resolution_with_tree(&layout2d, tree);
    }

    let d_max = euclidean_diameter_nd(layout)?;
    if d_max <= 0.0 {
        let num_pairs = (n as f64) * ((n - 1) as f64) * 0.5;
        return Ok(num_pairs);
    }

    let threshold = d_max / (n as f64).sqrt();
    let pts: Vec<f64> = layout.coords().iter().map(|v| v.to_f64()).collect();
    let mut buf: Vec<(usize, f64)> = Vec::new();
    let mut sum = 0.0_f64;

    for i in 0..n {
        let q = &pts[i * dims..(i + 1) * dims];
        index.within_radius_into(q, threshold, &mut buf)?;
        for &(j, d_sq) in &buf {
            if j <= i {
                continue;
            }
            let dist = d_sq.sqrt();
            if dist < threshold {
                let hinge = 1.0 - dist / threshold;
                sum += hinge * hinge;
            }
        }
    }

    Ok(sum)
}

/// Extracts exact undirected Relative Neighborhood Graph (`RNG`) edges `(u, v)` (`u < v`)
/// over the N-D coordinates in `layout`.
pub fn relative_neighborhood_edges_nd<S: Scalar>(
    layout: &LayoutNd<S>,
) -> CoreResult<Vec<(u32, u32)>> {
    validate_layout_nd(layout)?;
    let index = build_nd_spatial_index(layout)?;
    relative_neighborhood_edges_nd_with_index(layout, &index)
}

/// Extracts [`relative_neighborhood_edges_nd`] reusing a prebuilt [`NdSpatialIndex`].
#[doc(hidden)]
pub fn relative_neighborhood_edges_nd_with_index<S: Scalar>(
    layout: &LayoutNd<S>,
    index: &NdSpatialIndex,
) -> CoreResult<Vec<(u32, u32)>> {
    validate_layout_nd(layout)?;
    validate_index_layout_nd(index, layout, "relative_neighborhood_edges_nd_with_index")?;
    proximity_edges_nd_with_index(layout, index, ProximityMode::RelativeNeighborhood)
}

/// Extracts exact undirected Gabriel Graph (`GG`) edges `(u, v)` (`u < v`)
/// over the N-D coordinates in `layout`.
pub fn gabriel_graph_edges_nd<S: Scalar>(layout: &LayoutNd<S>) -> CoreResult<Vec<(u32, u32)>> {
    validate_layout_nd(layout)?;
    let index = build_nd_spatial_index(layout)?;
    gabriel_graph_edges_nd_with_index(layout, &index)
}

/// Extracts [`gabriel_graph_edges_nd`] reusing a prebuilt [`NdSpatialIndex`].
#[doc(hidden)]
pub fn gabriel_graph_edges_nd_with_index<S: Scalar>(
    layout: &LayoutNd<S>,
    index: &NdSpatialIndex,
) -> CoreResult<Vec<(u32, u32)>> {
    validate_layout_nd(layout)?;
    validate_index_layout_nd(index, layout, "gabriel_graph_edges_nd_with_index")?;
    proximity_edges_nd_with_index(layout, index, ProximityMode::Gabriel)
}

/// Computes unweighted shape-based Relative Neighborhood Graph preservation (`S_RNG`)
/// in `[0.0, 1.0]` for an N-D [`LayoutNd`].
pub fn relative_neighborhood_preservation_nd<W, S: Scalar>(
    graph: &CsrGraph<W>,
    layout: &LayoutNd<S>,
) -> CoreResult<f64> {
    validate_graph_layout_nd(graph, layout)?;
    if layout.is_empty() {
        return Ok(0.0);
    }
    let index = build_nd_spatial_index(layout)?;
    relative_neighborhood_preservation_nd_with_index(graph, layout, &index)
}

/// Computes [`relative_neighborhood_preservation_nd`] reusing a prebuilt [`NdSpatialIndex`].
#[doc(hidden)]
pub fn relative_neighborhood_preservation_nd_with_index<W, S: Scalar>(
    graph: &CsrGraph<W>,
    layout: &LayoutNd<S>,
    index: &NdSpatialIndex,
) -> CoreResult<f64> {
    validate_graph_layout_nd(graph, layout)?;
    validate_index_layout_nd(
        index,
        layout,
        "relative_neighborhood_preservation_nd_with_index",
    )?;
    if layout.is_empty() {
        return Ok(0.0);
    }
    let rng_edges =
        proximity_edges_nd_with_index(layout, index, ProximityMode::RelativeNeighborhood)?;
    Ok(mean_jaccard_graph_vs_edges(graph, &rng_edges))
}

/// Computes unweighted shape-based Gabriel Graph preservation (`S_GG`)
/// in `[0.0, 1.0]` for an N-D [`LayoutNd`].
pub fn gabriel_graph_preservation_nd<W, S: Scalar>(
    graph: &CsrGraph<W>,
    layout: &LayoutNd<S>,
) -> CoreResult<f64> {
    validate_graph_layout_nd(graph, layout)?;
    if layout.is_empty() {
        return Ok(0.0);
    }
    let index = build_nd_spatial_index(layout)?;
    gabriel_graph_preservation_nd_with_index(graph, layout, &index)
}

/// Computes [`gabriel_graph_preservation_nd`] reusing a prebuilt [`NdSpatialIndex`].
#[doc(hidden)]
pub fn gabriel_graph_preservation_nd_with_index<W, S: Scalar>(
    graph: &CsrGraph<W>,
    layout: &LayoutNd<S>,
    index: &NdSpatialIndex,
) -> CoreResult<f64> {
    validate_graph_layout_nd(graph, layout)?;
    validate_index_layout_nd(index, layout, "gabriel_graph_preservation_nd_with_index")?;
    if layout.is_empty() {
        return Ok(0.0);
    }
    let gg_edges = proximity_edges_nd_with_index(layout, index, ProximityMode::Gabriel)?;
    Ok(mean_jaccard_graph_vs_edges(graph, &gg_edges))
}

fn proximity_edges_nd_with_index<S: Scalar>(
    layout: &LayoutNd<S>,
    index: &NdSpatialIndex,
    mode: ProximityMode,
) -> CoreResult<Vec<(u32, u32)>> {
    let n = layout.num_nodes();
    if n < 2 {
        return Ok(Vec::new());
    }
    let dims = layout.dims();
    if dims == 2
        && let NdSpatialIndex::D2(tree) = index
    {
        let layout2d = Layout2d::from_interleaved(layout.coords())?;
        return proximity_edges_with_tree(&layout2d, tree, mode);
    }

    let pts: Vec<f64> = layout.coords().iter().map(|v| v.to_f64()).collect();
    let k_witness = 32usize.min(n - 1);
    let mut knn_flat = Vec::with_capacity(n * k_witness);
    let mut gg_blocked = vec![false; n];
    let mut kd_scratch = KdQueryScratch::with_capacity(k_witness);
    let mut knn_buf: Vec<(usize, f64)> = Vec::new();

    for (i, blocked) in gg_blocked.iter_mut().enumerate() {
        index.k_nearest_excluding_self_with_scratch(i, k_witness, &mut kd_scratch, &mut knn_buf)?;
        if matches!(mode, ProximityMode::Gabriel)
            && knn_buf.first().is_some_and(|&(_, d2)| d2 <= 0.0)
        {
            *blocked = true;
        }
        knn_flat.extend_from_slice(&knn_buf);
    }

    let mut edges = Vec::new();
    let mut mid = vec![0.0_f64; dims];
    let mut witness_buf: Vec<(usize, f64)> = Vec::new();

    for i in 0..n {
        if gg_blocked[i] {
            continue;
        }
        let pi = &pts[i * dims..(i + 1) * dims];
        let wi = &knn_flat[i * k_witness..(i + 1) * k_witness];

        for j in (i + 1)..n {
            if gg_blocked[j] {
                continue;
            }
            let pj = &pts[j * dims..(j + 1) * dims];
            let d_ij_sq = dist_sq_f64(pi, pj);
            if d_ij_sq <= 0.0 {
                continue;
            }
            if is_blocked_by_witnesses(j, pi, pj, d_ij_sq, wi, &pts, dims, mode)
                || is_blocked_by_witnesses(
                    i,
                    pj,
                    pi,
                    d_ij_sq,
                    &knn_flat[j * k_witness..(j + 1) * k_witness],
                    &pts,
                    dims,
                    mode,
                )
            {
                continue;
            }
            if is_proximity_edge_nd(
                i,
                j,
                d_ij_sq,
                &pts,
                dims,
                index,
                mode,
                &mut mid,
                &mut witness_buf,
            )? {
                edges.push((i as u32, j as u32));
            }
        }
    }

    Ok(edges)
}

#[inline]
#[allow(clippy::too_many_arguments)]
fn is_blocked_by_witnesses(
    other_endpoint: usize,
    _p_center: &[f64],
    p_other: &[f64],
    d_ij_sq: f64,
    witnesses: &[(usize, f64)],
    pts: &[f64],
    dims: usize,
    mode: ProximityMode,
) -> bool {
    match mode {
        ProximityMode::Gabriel => {
            let thresh = d_ij_sq * (1.0 + 1e-12);
            for &(r, d_cr_sq) in witnesses {
                if d_cr_sq > thresh {
                    break;
                }
                if r == other_endpoint {
                    continue;
                }
                let pr = &pts[r * dims..(r + 1) * dims];
                let d_or_sq = dist_sq_f64(p_other, pr);
                if d_cr_sq + d_or_sq <= thresh {
                    return true;
                }
            }
        }
        ProximityMode::RelativeNeighborhood => {
            let thresh = d_ij_sq * (1.0 - 1e-12);
            for &(r, d_cr_sq) in witnesses {
                if d_cr_sq >= thresh {
                    break;
                }
                if r == other_endpoint {
                    continue;
                }
                let pr = &pts[r * dims..(r + 1) * dims];
                let d_or_sq = dist_sq_f64(p_other, pr);
                if d_or_sq < thresh {
                    return true;
                }
            }
        }
    }
    false
}

#[allow(clippy::too_many_arguments)]
fn is_proximity_edge_nd(
    i: usize,
    j: usize,
    d_ij_sq: f64,
    pts: &[f64],
    dims: usize,
    index: &NdSpatialIndex,
    mode: ProximityMode,
    mid: &mut [f64],
    witness_buf: &mut Vec<(usize, f64)>,
) -> CoreResult<bool> {
    let pi = &pts[i * dims..(i + 1) * dims];
    let pj = &pts[j * dims..(j + 1) * dims];
    let mut max_coord_abs = 0.0_f64;
    for d in 0..dims {
        mid[d] = pi[d] + 0.5 * (pj[d] - pi[d]);
        max_coord_abs = max_coord_abs.max(pi[d].abs()).max(pj[d].abs());
    }

    let base_r_sq = match mode {
        ProximityMode::Gabriel => 0.25 * d_ij_sq * (1.0 + 1e-11),
        ProximityMode::RelativeNeighborhood => 0.75 * d_ij_sq * (1.0 + 1e-11),
    };
    let eps_mid = 4.0 * (dims as f64).sqrt() * f64::EPSILON * max_coord_abs;
    let padded_r = base_r_sq.sqrt() + eps_mid;
    let query_r_sq = (padded_r * padded_r).min(f64::MAX);
    index.within_radius_sq_into(mid, query_r_sq, witness_buf)?;

    for &(r, _) in witness_buf.iter() {
        if r == i || r == j {
            continue;
        }
        let pr = &pts[r * dims..(r + 1) * dims];
        let d_ir_sq = dist_sq_f64(pi, pr);
        let d_jr_sq = dist_sq_f64(pj, pr);
        let blocked = match mode {
            ProximityMode::Gabriel => d_ir_sq + d_jr_sq <= d_ij_sq * (1.0 + 1e-12),
            ProximityMode::RelativeNeighborhood => {
                let thresh = d_ij_sq * (1.0 - 1e-12);
                d_ir_sq < thresh && d_jr_sq < thresh
            }
        };
        if blocked {
            return Ok(false);
        }
    }
    Ok(true)
}
