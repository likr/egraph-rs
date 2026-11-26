use std::cmp::Ordering;

use egraph_core::{CoreError, CoreResult, CsrGraph, LayoutNd, Scalar};
use egraph_spatial_trees::KdQueryScratch;

use crate::extract_unique_edges;
use crate::nd_index::{
    NdSpatialIndex, build_nd_spatial_index, slice_dist_nd, validate_graph_layout_nd,
    validate_index_layout_nd,
};

use super::angles_degeneracy::{LayoutDegeneracy, canonical_f32, canonical_f64};

/// Evaluates ideal edge-length preservation $\sum_{(u,v)\in E} ((\|x_u - x_v\| - \ell_{uv}) / \ell_{uv})^2$
/// over an N-D [`LayoutNd`].
pub fn ideal_edge_lengths_nd<W: Scalar, S: Scalar>(
    graph: &CsrGraph<W>,
    layout: &LayoutNd<S>,
) -> CoreResult<f64> {
    validate_graph_layout_nd(graph, layout)?;
    let Some(weights) = graph.weights() else {
        return ideal_edge_lengths_with_unit_nd(graph, layout, 1.0);
    };

    let n = layout.num_nodes();
    let dims = layout.dims();
    let coords = layout.coords();
    let directed = graph.is_directed();
    let indptr = graph.indptr();
    let indices = graph.indices();
    let mut sum = 0.0_f64;

    for u in 0..n {
        let u32_u = u as u32;
        let pu = &coords[u * dims..(u + 1) * dims];
        let start = indptr[u] as usize;
        let end = indptr[u + 1] as usize;
        for pos in start..end {
            let v = indices[pos];
            if v == u32_u || (!directed && u32_u >= v) {
                continue;
            }
            let target_len = weights[pos].to_f64();
            if !target_len.is_finite() || target_len <= 0.0 {
                continue;
            }
            let vi = v as usize;
            let pv = &coords[vi * dims..(vi + 1) * dims];
            let actual = slice_dist_nd(pu, pv);
            let err = (actual - target_len) / target_len;
            sum += err * err;
        }
    }

    Ok(sum)
}

/// Evaluates [`ideal_edge_lengths_nd`] against a uniform `unit_edge_length > 0.0`.
pub fn ideal_edge_lengths_with_unit_nd<W, S: Scalar>(
    graph: &CsrGraph<W>,
    layout: &LayoutNd<S>,
    unit_edge_length: f64,
) -> CoreResult<f64> {
    validate_graph_layout_nd(graph, layout)?;
    if !unit_edge_length.is_finite() {
        return Err(CoreError::non_finite(
            "ideal_edge_lengths_with_unit_nd::unit_edge_length",
        ));
    }
    if unit_edge_length <= 0.0 {
        return Err(CoreError::invalid_input(
            "unit_edge_length must be strictly positive",
        ));
    }

    let dims = layout.dims();
    let coords = layout.coords();
    let mut sum = 0.0_f64;
    for (u, v) in extract_unique_edges(graph) {
        let ui = u as usize;
        let vi = v as usize;
        let pu = &coords[ui * dims..(ui + 1) * dims];
        let pv = &coords[vi * dims..(vi + 1) * dims];
        let actual = slice_dist_nd(pu, pv);
        let err = (actual - unit_edge_length) / unit_edge_length;
        sum += err * err;
    }
    Ok(sum)
}

/// Evaluates [`ideal_edge_lengths_nd`] scaled by `unit_edge_length > 0.0`.
pub fn ideal_edge_lengths_scaled_nd<W: Scalar, S: Scalar>(
    graph: &CsrGraph<W>,
    layout: &LayoutNd<S>,
    unit_edge_length: f64,
) -> CoreResult<f64> {
    validate_graph_layout_nd(graph, layout)?;
    if !unit_edge_length.is_finite() {
        return Err(CoreError::non_finite(
            "ideal_edge_lengths_scaled_nd::unit_edge_length",
        ));
    }
    if unit_edge_length <= 0.0 {
        return Err(CoreError::invalid_input(
            "unit_edge_length must be strictly positive",
        ));
    }
    let Some(weights) = graph.weights() else {
        return ideal_edge_lengths_with_unit_nd(graph, layout, unit_edge_length);
    };

    let n = layout.num_nodes();
    let dims = layout.dims();
    let coords = layout.coords();
    let directed = graph.is_directed();
    let indptr = graph.indptr();
    let indices = graph.indices();
    let mut sum = 0.0_f64;

    for u in 0..n {
        let u32_u = u as u32;
        let pu = &coords[u * dims..(u + 1) * dims];
        let start = indptr[u] as usize;
        let end = indptr[u + 1] as usize;
        for pos in start..end {
            let v = indices[pos];
            if v == u32_u || (!directed && u32_u >= v) {
                continue;
            }
            let target_len = weights[pos].to_f64() * unit_edge_length;
            if !target_len.is_finite() || target_len <= 0.0 {
                continue;
            }
            let vi = v as usize;
            let pv = &coords[vi * dims..(vi + 1) * dims];
            let actual = slice_dist_nd(pu, pv);
            let err = (actual - target_len) / target_len;
            sum += err * err;
        }
    }

    Ok(sum)
}

/// Evaluates [`ideal_edge_lengths_nd`] using a flat row-major `N * N` target distance matrix.
pub fn ideal_edge_lengths_from_distances_nd<W, S: Scalar>(
    graph: &CsrGraph<W>,
    layout: &LayoutNd<S>,
    distances: &[f64],
) -> CoreResult<f64> {
    validate_graph_layout_nd(graph, layout)?;
    let n = layout.num_nodes();
    let expected = n
        .checked_mul(n)
        .ok_or_else(|| CoreError::invalid_input("distance matrix size overflow"))?;
    if distances.len() != expected {
        return Err(CoreError::dimension_mismatch(
            expected,
            distances.len(),
            "ideal_edge_lengths_from_distances_nd::distances",
        ));
    }

    let dims = layout.dims();
    let coords = layout.coords();
    let mut sum = 0.0_f64;
    for (u, v) in extract_unique_edges(graph) {
        let ui = u as usize;
        let vi = v as usize;
        let target_len = distances[ui * n + vi];
        if !target_len.is_finite() || target_len <= 0.0 {
            continue;
        }
        let pu = &coords[ui * dims..(ui + 1) * dims];
        let pv = &coords[vi * dims..(vi + 1) * dims];
        let actual = slice_dist_nd(pu, pv);
        let err = (actual - target_len) / target_len;
        sum += err * err;
    }
    Ok(sum)
}

/// Evaluates duplicate-node (`f32`/`f64`), near-collision, zero-length edge, and bounding-box
/// degeneracies over an N-D [`LayoutNd`].
pub fn layout_degeneracy_nd<W, S: Scalar>(
    graph: &CsrGraph<W>,
    layout: &LayoutNd<S>,
) -> CoreResult<LayoutDegeneracy> {
    layout_degeneracy_nd_core(graph, layout, None)
}

/// Computes [`layout_degeneracy_nd`] reusing a prebuilt [`NdSpatialIndex`].
#[doc(hidden)]
pub fn layout_degeneracy_nd_with_index<W, S: Scalar>(
    graph: &CsrGraph<W>,
    layout: &LayoutNd<S>,
    index: &NdSpatialIndex,
) -> CoreResult<LayoutDegeneracy> {
    validate_graph_layout_nd(graph, layout)?;
    validate_index_layout_nd(index, layout, "layout_degeneracy_nd_with_index")?;
    layout_degeneracy_nd_core(graph, layout, Some(index))
}

#[inline]
fn cmp_f32_rows(a: &[f32], b: &[f32]) -> Ordering {
    for d in 0..a.len() {
        let ord = a[d].to_bits().cmp(&b[d].to_bits());
        if ord != Ordering::Equal {
            return ord;
        }
    }
    Ordering::Equal
}

#[inline]
fn cmp_f64_rows(a: &[f64], b: &[f64]) -> Ordering {
    for d in 0..a.len() {
        let ord = a[d].to_bits().cmp(&b[d].to_bits());
        if ord != Ordering::Equal {
            return ord;
        }
    }
    Ordering::Equal
}

fn layout_degeneracy_nd_core<W, S: Scalar>(
    graph: &CsrGraph<W>,
    layout: &LayoutNd<S>,
    prebuilt: Option<&NdSpatialIndex>,
) -> CoreResult<LayoutDegeneracy> {
    validate_graph_layout_nd(graph, layout)?;
    let n = layout.num_nodes();
    let dims = layout.dims();
    let edges = extract_unique_edges(graph);
    let m = edges.len();
    if n == 0 {
        return Ok(LayoutDegeneracy {
            num_nodes: 0,
            num_edges: m,
            ..LayoutDegeneracy::default()
        });
    }

    let coords = layout.coords();
    let mut coords_64 = Vec::with_capacity(n * dims);
    let mut coords_32 = Vec::with_capacity(n * dims);
    let mut mins = vec![f64::INFINITY; dims];
    let mut maxs = vec![f64::NEG_INFINITY; dims];

    for i in 0..n {
        let base = i * dims;
        for d in 0..dims {
            let v64 = canonical_f64(coords[base + d].to_f64());
            mins[d] = mins[d].min(v64);
            maxs[d] = maxs[d].max(v64);
            coords_64.push(v64);
            coords_32.push(canonical_f32(v64 as f32));
        }
    }

    let mut order_32: Vec<usize> = (0..n).collect();
    order_32.sort_unstable_by(|&a, &b| {
        cmp_f32_rows(
            &coords_32[a * dims..(a + 1) * dims],
            &coords_32[b * dims..(b + 1) * dims],
        )
    });
    let mut dup_f32 = 0usize;
    let mut run = 1usize;
    for i in 1..n {
        let prev = &coords_32[order_32[i - 1] * dims..(order_32[i - 1] + 1) * dims];
        let curr = &coords_32[order_32[i] * dims..(order_32[i] + 1) * dims];
        if cmp_f32_rows(prev, curr) == Ordering::Equal {
            run += 1;
        } else {
            if run > 1 {
                dup_f32 += run;
            }
            run = 1;
        }
    }
    if run > 1 {
        dup_f32 += run;
    }

    let mut order_64: Vec<usize> = (0..n).collect();
    order_64.sort_unstable_by(|&a, &b| {
        cmp_f64_rows(
            &coords_64[a * dims..(a + 1) * dims],
            &coords_64[b * dims..(b + 1) * dims],
        )
    });
    let mut dup_f64 = 0usize;
    let mut excess_dup64 = 0usize;
    let mut run64 = 1usize;
    for i in 1..n {
        let prev = &coords_64[order_64[i - 1] * dims..(order_64[i - 1] + 1) * dims];
        let curr = &coords_64[order_64[i] * dims..(order_64[i] + 1) * dims];
        if cmp_f64_rows(prev, curr) == Ordering::Equal {
            run64 += 1;
        } else {
            if run64 > 1 {
                dup_f64 += run64;
                excess_dup64 += run64 - 1;
            }
            run64 = 1;
        }
    }
    if run64 > 1 {
        dup_f64 += run64;
        excess_dup64 += run64 - 1;
    }

    let mut zero_edges = 0usize;
    for &(u, v) in &edges {
        let ui = u as usize;
        let vi = v as usize;
        let pu = &coords_64[ui * dims..(ui + 1) * dims];
        let pv = &coords_64[vi * dims..(vi + 1) * dims];
        if slice_dist_nd(pu, pv) == 0.0 {
            zero_edges += 1;
        }
    }

    let bbox_width = (maxs[0] - mins[0]).max(0.0);
    let bbox_height = if dims >= 2 {
        (maxs[1] - mins[1]).max(0.0)
    } else {
        0.0
    };
    let mut is_bbox_degenerate = false;
    let mut max_span = 0.0_f64;
    for d in 0..dims {
        let span = (maxs[d] - mins[d]).max(0.0);
        if span <= 1e-12 {
            is_bbox_degenerate = true;
        }
        max_span = max_span.max(span);
    }

    let diag = if dims == 2 {
        bbox_width.hypot(bbox_height)
    } else if max_span == 0.0 {
        0.0
    } else {
        let scale = if max_span > 1e150 || max_span < 1e-150 {
            1.0 / max_span
        } else {
            1.0
        };
        let mut sum_sq = 0.0_f64;
        for d in 0..dims {
            let s = (maxs[d] - mins[d]).max(0.0) * scale;
            sum_sq += s * s;
        }
        sum_sq.sqrt() / scale
    };

    let near_collision_ratio = if n <= 1 {
        0.0
    } else if diag == 0.0 {
        1.0
    } else {
        let tau = 1e-7 * diag;
        let owned_index;
        let index = match prebuilt {
            Some(idx) => idx,
            None => {
                owned_index = build_nd_spatial_index(layout)?;
                &owned_index
            }
        };
        let mut scratch = KdQueryScratch::with_capacity(1);
        let mut near_collisions = 0usize;
        for i in 0..n {
            if let Some((_, d2)) = index.nearest_excluding_self_with_scratch(i, &mut scratch)?
                && d2.sqrt() <= tau
            {
                near_collisions += 1;
            }
        }
        near_collisions as f64 / n as f64
    };

    Ok(LayoutDegeneracy {
        num_nodes: n,
        num_edges: m,
        duplicate_nodes_f32: dup_f32,
        duplicate_ratio_f32: dup_f32 as f64 / n as f64,
        duplicate_nodes_f64: dup_f64,
        duplicate_ratio_f64: dup_f64 as f64 / n as f64,
        excess_duplicate_ratio: excess_dup64 as f64 / n as f64,
        near_collision_ratio,
        zero_length_edges: zero_edges,
        zero_length_edge_ratio: if m == 0 {
            0.0
        } else {
            zero_edges as f64 / m as f64
        },
        bbox_width,
        bbox_height,
        is_bbox_degenerate,
    })
}
