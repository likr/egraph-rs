use std::any::Any;
use std::cmp::Ordering;

use egraph_core::{
    CoreResult, CsrGraph, EuclideanLayout, Layout2d, Layout3d, LayoutNd, ManifoldLayout2d,
    MetricSpace, Scalar,
};
use egraph_spatial_trees::{KdQueryScratch, NearestNeighborIndex};

use crate::extract_unique_edges;
use crate::spatial_index::{
    build_spatial_index, copy_euclidean_coords_f64, metric_diameter, validate_graph_layout,
    validate_index_layout,
};

/// Coordinate-collapse and zero-length edge degeneracy diagnostics for a layout.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Default)]
pub struct LayoutDegeneracy {
    /// Total number of vertices $N$.
    pub num_nodes: usize,
    /// Total number of unique non-self-loop edges $M$.
    pub num_edges: usize,
    /// Number of vertices sharing an exact `f32` coordinate with at least one other vertex.
    pub duplicate_nodes_f32: usize,
    /// Fraction `duplicate_nodes_f32 / num_nodes` (`0.0` when `num_nodes == 0`).
    pub duplicate_ratio_f32: f64,
    /// Number of vertices sharing an exact `f64` coordinate with at least one other vertex.
    pub duplicate_nodes_f64: usize,
    /// Fraction `duplicate_nodes_f64 / num_nodes` (`0.0` when `num_nodes == 0`).
    pub duplicate_ratio_f64: f64,
    /// Excess duplicate fraction $1 - n_{\mathrm{uniq}} / N$ using exact `f64` coordinates (`0.0` when `num_nodes == 0`).
    pub excess_duplicate_ratio: f64,
    /// Fraction of vertices with at least one other distinct vertex within distance
    /// $\tau = 10^{-7} \cdot \mathrm{diag}$ (`0.0` when `num_nodes <= 1`, `1.0` when `num_nodes >= 2`
    /// and `diag == 0.0`).
    pub near_collision_ratio: f64,
    /// Number of unique non-self-loop edges whose endpoints have distance `0.0`.
    pub zero_length_edges: usize,
    /// Fraction `zero_length_edges / num_edges` (`0.0` when `num_edges == 0`).
    pub zero_length_edge_ratio: f64,
    /// Axis-aligned bounding-box width `max_x - min_x` (or `0.0` for non-coordinate spaces).
    pub bbox_width: f64,
    /// Axis-aligned bounding-box height `max_y - min_y` (or `0.0` when `dims < 2` or non-coordinate).
    pub bbox_height: f64,
    /// `true` when `num_nodes > 0` and any coordinate axis span `<= 1e-12` (or metric diameter `<= 1e-12`).
    pub is_bbox_degenerate: bool,
}

#[inline]
fn canonical_f32(v: f32) -> f32 {
    if v == 0.0 { 0.0 } else { v }
}

#[inline]
fn canonical_f64(v: f64) -> f64 {
    if v == 0.0 { 0.0 } else { v }
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

fn count_row_duplicates(
    coords_64: &[f64],
    coords_32: &[f32],
    n: usize,
    dims: usize,
) -> (usize, usize, usize) {
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

    (dup_f32, dup_f64, excess_dup64)
}

/// Evaluates duplicate-node (`f32`/`f64`), near-collision, zero-length edge, and bounding-box
/// degeneracies across any [`MetricSpace`].
///
/// Canonicalizes `-0.0` to `+0.0` before bitwise key comparison on coordinate spaces.
pub fn layout_degeneracy<W, L: MetricSpace + 'static>(
    graph: &CsrGraph<W>,
    layout: &L,
) -> CoreResult<LayoutDegeneracy> {
    validate_graph_layout(graph, layout, "layout_degeneracy")?;
    let any: &dyn Any = layout;
    if let Some(l) = any.downcast_ref::<Layout2d<f64>>() {
        return layout_degeneracy_euclidean_core(graph, l, None);
    }
    if let Some(l) = any.downcast_ref::<Layout2d<f32>>() {
        return layout_degeneracy_euclidean_core(graph, l, None);
    }
    if let Some(l) = any.downcast_ref::<Layout3d<f64>>() {
        return layout_degeneracy_euclidean_core(graph, l, None);
    }
    if let Some(l) = any.downcast_ref::<Layout3d<f32>>() {
        return layout_degeneracy_euclidean_core(graph, l, None);
    }
    if let Some(l) = any.downcast_ref::<LayoutNd<f64>>() {
        return layout_degeneracy_euclidean_core(graph, l, None);
    }
    if let Some(l) = any.downcast_ref::<LayoutNd<f32>>() {
        return layout_degeneracy_euclidean_core(graph, l, None);
    }
    if let Some(ml) = any.downcast_ref::<ManifoldLayout2d<f64>>() {
        return Ok(layout_degeneracy_manifold_core(graph, ml));
    }
    if let Some(ml) = any.downcast_ref::<ManifoldLayout2d<f32>>() {
        return Ok(layout_degeneracy_manifold_core(graph, ml));
    }
    Ok(layout_degeneracy_metric_core(graph, layout))
}

/// Computes [`layout_degeneracy`] reusing a prebuilt [`NearestNeighborIndex`].
pub fn layout_degeneracy_with_index<
    W,
    L: EuclideanLayout + ?Sized,
    I: NearestNeighborIndex + ?Sized,
>(
    graph: &CsrGraph<W>,
    layout: &L,
    index: &I,
) -> CoreResult<LayoutDegeneracy> {
    validate_graph_layout(graph, layout, "layout_degeneracy_with_index")?;
    validate_index_layout(index, layout, "layout_degeneracy_with_index")?;
    layout_degeneracy_euclidean_core(graph, layout, Some(&index))
}

fn layout_degeneracy_euclidean_core<W, L: EuclideanLayout + ?Sized>(
    graph: &CsrGraph<W>,
    layout: &L,
    prebuilt: Option<&dyn NearestNeighborIndex>,
) -> CoreResult<LayoutDegeneracy> {
    let n = layout.len();
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

    let raw_64 = copy_euclidean_coords_f64(layout);
    let mut coords_64 = Vec::with_capacity(n * dims);
    let mut coords_32 = Vec::with_capacity(n * dims);
    let mut mins = vec![f64::INFINITY; dims];
    let mut maxs = vec![f64::NEG_INFINITY; dims];

    for i in 0..n {
        let base = i * dims;
        for d in 0..dims {
            let v64 = canonical_f64(raw_64[base + d]);
            mins[d] = mins[d].min(v64);
            maxs[d] = maxs[d].max(v64);
            coords_64.push(v64);
            coords_32.push(canonical_f32(v64 as f32));
        }
    }

    let (dup_f32, dup_f64, excess_dup64) = count_row_duplicates(&coords_64, &coords_32, n, dims);

    let mut zero_edges = 0usize;
    for &(u, v) in &edges {
        if layout.distance(u as usize, v as usize) == 0.0 {
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
        let index: &dyn NearestNeighborIndex = match prebuilt {
            Some(idx) => idx,
            None => {
                owned_index = build_spatial_index(layout)?;
                owned_index.as_ref()
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

fn layout_degeneracy_manifold_core<W, S: Scalar>(
    graph: &CsrGraph<W>,
    layout: &ManifoldLayout2d<S>,
) -> LayoutDegeneracy {
    let n = layout.len();
    let edges = extract_unique_edges(graph);
    let m = edges.len();
    if n == 0 {
        return LayoutDegeneracy {
            num_nodes: 0,
            num_edges: m,
            ..LayoutDegeneracy::default()
        };
    }

    let xs = layout.x();
    let ys = layout.y();
    let mut coords_64 = Vec::with_capacity(n * 2);
    let mut coords_32 = Vec::with_capacity(n * 2);
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
        coords_64.push(x64);
        coords_64.push(y64);
        coords_32.push(canonical_f32(x64 as f32));
        coords_32.push(canonical_f32(y64 as f32));
    }

    let (dup_f32, dup_f64, excess_dup64) = count_row_duplicates(&coords_64, &coords_32, n, 2);

    let mut zero_edges = 0usize;
    for &(u, v) in &edges {
        if layout.distance(u as usize, v as usize) == 0.0 {
            zero_edges += 1;
        }
    }

    let bbox_width = (max_x - min_x).max(0.0);
    let bbox_height = (max_y - min_y).max(0.0);
    let is_bbox_degenerate = bbox_width <= 1e-12 || bbox_height <= 1e-12;

    let d_max = metric_diameter(layout);
    let near_collision_ratio = if n <= 1 {
        0.0
    } else if d_max == 0.0 {
        1.0
    } else {
        let tau = 1e-7 * d_max;
        let mut near_collisions = 0usize;
        for i in 0..n {
            for j in 0..n {
                if i != j && layout.distance(i, j) <= tau {
                    near_collisions += 1;
                    break;
                }
            }
        }
        near_collisions as f64 / n as f64
    };

    LayoutDegeneracy {
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
    }
}

fn layout_degeneracy_metric_core<W, L: MetricSpace + ?Sized>(
    graph: &CsrGraph<W>,
    layout: &L,
) -> LayoutDegeneracy {
    let n = layout.len();
    let edges = extract_unique_edges(graph);
    let m = edges.len();
    if n == 0 {
        return LayoutDegeneracy {
            num_nodes: 0,
            num_edges: m,
            ..LayoutDegeneracy::default()
        };
    }

    let mut dup_f32 = 0usize;
    let mut dup_f64 = 0usize;
    let mut seen_uniq = vec![false; n];
    let mut n_uniq = 0usize;
    for i in 0..n {
        let mut has_dup64 = false;
        let mut has_dup32 = false;
        if !seen_uniq[i] {
            n_uniq += 1;
            seen_uniq[i] = true;
        }
        for (j, seen_j) in seen_uniq.iter_mut().enumerate().take(n) {
            if i == j {
                continue;
            }
            let d = layout.distance(i, j);
            if d == 0.0 {
                has_dup64 = true;
                *seen_j = true;
            }
            if (d as f32) == 0.0 {
                has_dup32 = true;
            }
        }
        if has_dup64 {
            dup_f64 += 1;
        }
        if has_dup32 {
            dup_f32 += 1;
        }
    }
    let excess_dup64 = n - n_uniq;

    let mut zero_edges = 0usize;
    for &(u, v) in &edges {
        if layout.distance(u as usize, v as usize) == 0.0 {
            zero_edges += 1;
        }
    }

    let d_max = metric_diameter(layout);
    let near_collision_ratio = if n <= 1 {
        0.0
    } else if d_max == 0.0 {
        1.0
    } else {
        let tau = 1e-7 * d_max;
        let mut near_collisions = 0usize;
        for i in 0..n {
            for j in 0..n {
                if i != j && layout.distance(i, j) <= tau {
                    near_collisions += 1;
                    break;
                }
            }
        }
        near_collisions as f64 / n as f64
    };

    LayoutDegeneracy {
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
        bbox_width: 0.0,
        bbox_height: 0.0,
        is_bbox_degenerate: d_max <= 1e-12,
    }
}
