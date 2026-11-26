use egraph_core::{CoreError, CoreResult, CsrGraph, ManifoldLayout2d, Scalar};

use crate::proximity::manifold_geodesic_diameter;
use crate::{extract_unique_edges, validate_graph_manifold_layout};

use super::angles_degeneracy::{LayoutDegeneracy, canonical_f32, canonical_f64};

/// Evaluates ideal edge-length preservation
/// $\sum_{(u,v)\in E} ((d_{\mathcal{M}_s}(x_u, x_v) - \ell_{uv}) / \ell_{uv})^2$
/// over a [`ManifoldLayout2d`].
pub fn ideal_edge_lengths_manifold<W: Scalar, S: Scalar>(
    graph: &CsrGraph<W>,
    layout: &ManifoldLayout2d<S>,
) -> CoreResult<f64> {
    validate_graph_manifold_layout(graph, layout, "ideal_edge_lengths_manifold")?;
    let Some(weights) = graph.weights() else {
        return ideal_edge_lengths_with_unit_manifold(graph, layout, 1.0);
    };

    let n = layout.len();
    let directed = graph.is_directed();
    let indptr = graph.indptr();
    let indices = graph.indices();
    let mut sum = 0.0_f64;

    for u in 0..n {
        let u32_u = u as u32;
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
            let actual = layout.geodesic_distance(u, vi)?.to_f64();
            let err = (actual - target_len) / target_len;
            sum += err * err;
        }
    }

    Ok(sum)
}

/// Evaluates [`ideal_edge_lengths_manifold`] against a uniform `unit_edge_length > 0.0`.
pub fn ideal_edge_lengths_with_unit_manifold<W, S: Scalar>(
    graph: &CsrGraph<W>,
    layout: &ManifoldLayout2d<S>,
    unit_edge_length: f64,
) -> CoreResult<f64> {
    validate_graph_manifold_layout(graph, layout, "ideal_edge_lengths_with_unit_manifold")?;
    if !unit_edge_length.is_finite() {
        return Err(CoreError::non_finite(
            "ideal_edge_lengths_with_unit_manifold::unit_edge_length",
        ));
    }
    if unit_edge_length <= 0.0 {
        return Err(CoreError::invalid_input(
            "unit_edge_length must be strictly positive",
        ));
    }

    let mut sum = 0.0_f64;
    for (u, v) in extract_unique_edges(graph) {
        let actual = layout.geodesic_distance(u as usize, v as usize)?.to_f64();
        let err = (actual - unit_edge_length) / unit_edge_length;
        sum += err * err;
    }
    Ok(sum)
}

/// Evaluates [`ideal_edge_lengths_manifold`] scaled by `unit_edge_length > 0.0`.
pub fn ideal_edge_lengths_scaled_manifold<W: Scalar, S: Scalar>(
    graph: &CsrGraph<W>,
    layout: &ManifoldLayout2d<S>,
    unit_edge_length: f64,
) -> CoreResult<f64> {
    validate_graph_manifold_layout(graph, layout, "ideal_edge_lengths_scaled_manifold")?;
    if !unit_edge_length.is_finite() {
        return Err(CoreError::non_finite(
            "ideal_edge_lengths_scaled_manifold::unit_edge_length",
        ));
    }
    if unit_edge_length <= 0.0 {
        return Err(CoreError::invalid_input(
            "unit_edge_length must be strictly positive",
        ));
    }
    let Some(weights) = graph.weights() else {
        return ideal_edge_lengths_with_unit_manifold(graph, layout, unit_edge_length);
    };

    let n = layout.len();
    let directed = graph.is_directed();
    let indptr = graph.indptr();
    let indices = graph.indices();
    let mut sum = 0.0_f64;

    for u in 0..n {
        let u32_u = u as u32;
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
            let actual = layout.geodesic_distance(u, vi)?.to_f64();
            let err = (actual - target_len) / target_len;
            sum += err * err;
        }
    }

    Ok(sum)
}

/// Evaluates [`ideal_edge_lengths_manifold`] using a flat row-major `N * N` target distance matrix.
pub fn ideal_edge_lengths_from_distances_manifold<W, S: Scalar>(
    graph: &CsrGraph<W>,
    layout: &ManifoldLayout2d<S>,
    distances: &[f64],
) -> CoreResult<f64> {
    validate_graph_manifold_layout(graph, layout, "ideal_edge_lengths_from_distances_manifold")?;
    let n = layout.len();
    let expected = n
        .checked_mul(n)
        .ok_or_else(|| CoreError::invalid_input("distance matrix size overflow"))?;
    if distances.len() != expected {
        return Err(CoreError::dimension_mismatch(
            expected,
            distances.len(),
            "ideal_edge_lengths_from_distances_manifold::distances",
        ));
    }

    let mut sum = 0.0_f64;
    for (u, v) in extract_unique_edges(graph) {
        let ui = u as usize;
        let vi = v as usize;
        let target_len = distances[ui * n + vi];
        if !target_len.is_finite() || target_len <= 0.0 {
            continue;
        }
        let actual = layout.geodesic_distance(ui, vi)?.to_f64();
        let err = (actual - target_len) / target_len;
        sum += err * err;
    }
    Ok(sum)
}

/// Evaluates duplicate-node (`f32`/`f64`), zero-geodesic-length edge, near-collision (against
/// geodesic diameter $d_{\max}$), and bounding-box degeneracies over a [`ManifoldLayout2d`].
pub fn layout_degeneracy_manifold<W, S: Scalar>(
    graph: &CsrGraph<W>,
    layout: &ManifoldLayout2d<S>,
) -> CoreResult<LayoutDegeneracy> {
    validate_graph_manifold_layout(graph, layout, "layout_degeneracy_manifold")?;
    let n = layout.len();
    let edges = extract_unique_edges(graph);
    let m = edges.len();
    if n == 0 {
        return Ok(LayoutDegeneracy {
            num_nodes: 0,
            num_edges: m,
            ..LayoutDegeneracy::default()
        });
    }

    let xs = layout.x();
    let ys = layout.y();
    let mut keys_f32 = Vec::with_capacity(n);
    let mut keys_f64 = Vec::with_capacity(n);
    let mut min_x = f64::INFINITY;
    let mut max_x = f64::NEG_INFINITY;
    let mut min_y = f64::INFINITY;
    let mut max_y = f64::NEG_INFINITY;

    for i in 0..n {
        let x64 = canonical_f64(xs[i].to_f64());
        let y64 = canonical_f64(ys[i].to_f64());
        min_x = min_x.min(x64);
        max_x = max_x.max(x64);
        min_y = min_y.min(y64);
        max_y = max_y.max(y64);
        keys_f64.push((x64.to_bits(), y64.to_bits()));

        let x32 = canonical_f32(x64 as f32);
        let y32 = canonical_f32(y64 as f32);
        keys_f32.push(((x32.to_bits() as u64) << 32) | (y32.to_bits() as u64));
    }

    keys_f32.sort_unstable();
    let mut dup_f32 = 0usize;
    let mut run = 1usize;
    for i in 1..n {
        if keys_f32[i] == keys_f32[i - 1] {
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

    keys_f64.sort_unstable();
    let mut dup_f64 = 0usize;
    let mut excess_dup64 = 0usize;
    let mut run64 = 1usize;
    for i in 1..n {
        if keys_f64[i] == keys_f64[i - 1] {
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
        if layout.geodesic_distance(u as usize, v as usize)?.to_f64() == 0.0 {
            zero_edges += 1;
        }
    }

    let bbox_width = (max_x - min_x).max(0.0);
    let bbox_height = (max_y - min_y).max(0.0);
    let is_bbox_degenerate = bbox_width <= 1e-12 || bbox_height <= 1e-12;

    let d_max = manifold_geodesic_diameter(layout)?;
    let near_collision_ratio = if n <= 1 {
        0.0
    } else if d_max == 0.0 {
        1.0
    } else {
        let tau = 1e-7 * d_max;
        let mut near_collisions = 0usize;
        for i in 0..n {
            for j in 0..n {
                if i != j && layout.geodesic_distance(i, j)?.to_f64() <= tau {
                    near_collisions += 1;
                    break;
                }
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
