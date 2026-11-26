use std::f64::consts::TAU;

use egraph_core::{CoreResult, CsrGraph, Layout2d, Scalar};
use egraph_spatial_trees::StaticKdTree;

use crate::{build_undirected_adjacency, extract_unique_edges, validate_graph_layout};

/// DRGraph Eq. (19) minimum-angle resolution summary (`a1` and `a2`) in `[0.0, 1.0]`.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct MinimumAngleSummary {
    /// A1 score using `atan2(0.0, 0.0) = 0.0` for zero-length incident edges.
    pub a1: f64,
    /// A2 score penalizing any `deg >= 2` vertex with at least one zero-length incident edge to `0.0`.
    pub a2: f64,
}

/// Computes DRGraph Eq. (19) minimum-angle resolution (`A1` and `A2`) in `[0.0, 1.0]`.
///
/// Vertices with effective undirected degree `< 2` receive score `1.0`, while empty graphs
/// (`N == 0`) return `MinimumAngleSummary { a1: 0.0, a2: 0.0 }`.
pub fn minimum_angle_summary<W, S: Scalar>(
    graph: &CsrGraph<W>,
    layout: &Layout2d<S>,
) -> CoreResult<MinimumAngleSummary> {
    validate_graph_layout(graph, layout, "minimum_angle_summary")?;
    let n = layout.len();
    if n == 0 {
        return Ok(MinimumAngleSummary { a1: 0.0, a2: 0.0 });
    }

    let directed = graph.is_directed();
    let directed_adj = if directed {
        Some(build_undirected_adjacency(graph))
    } else {
        None
    };

    let xs = layout.x();
    let ys = layout.y();
    let mut angles = Vec::<f64>::new();
    let mut sum_a1 = 0.0_f64;
    let mut sum_a2 = 0.0_f64;

    for u in 0..n {
        let u32_u = u as u32;
        let nbrs = match directed_adj.as_ref() {
            Some(adj) => adj[u].as_slice(),
            None => graph.neighbors(u),
        };
        angles.clear();
        let ux = xs[u].to_f64();
        let uy = ys[u].to_f64();
        let mut has_zero = false;
        for &v in nbrs {
            if v == u32_u {
                continue;
            }
            let vi = v as usize;
            let dx = xs[vi].to_f64() - ux;
            let dy = ys[vi].to_f64() - uy;
            if dx == 0.0 && dy == 0.0 {
                has_zero = true;
                angles.push(0.0);
            } else {
                angles.push(dy.atan2(dx));
            }
        }
        let d = angles.len();
        if d < 2 {
            sum_a1 += 1.0;
            sum_a2 += 1.0;
            continue;
        }
        angles.sort_unstable_by(f64::total_cmp);
        let mut min_diff = (TAU - (angles[d - 1] - angles[0])).max(0.0);
        for k in 0..(d - 1) {
            let gap = (angles[k + 1] - angles[k]).max(0.0);
            if gap < min_diff {
                min_diff = gap;
            }
        }
        let ideal = TAU / (d as f64);
        let score = (1.0 - (ideal - min_diff).abs() / ideal).clamp(0.0, 1.0);
        sum_a1 += score;
        if !has_zero {
            sum_a2 += score;
        }
    }

    let inv_n = 1.0 / (n as f64);
    Ok(MinimumAngleSummary {
        a1: (sum_a1 * inv_n).clamp(0.0, 1.0),
        a2: (sum_a2 * inv_n).clamp(0.0, 1.0),
    })
}

/// Computes DRGraph Eq. (19) strict minimum-angle resolution (`A2`) in `[0.0, 1.0]`.
///
/// Equivalent to `minimum_angle_summary(graph, layout)?.a2`.
pub fn minimum_angle<W, S: Scalar>(graph: &CsrGraph<W>, layout: &Layout2d<S>) -> CoreResult<f64> {
    Ok(minimum_angle_summary(graph, layout)?.a2)
}

/// Coordinate-collapse and zero-length edge degeneracy diagnostics for a 2D layout.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct LayoutDegeneracy {
    /// Total number of vertices $N$.
    pub num_nodes: usize,
    /// Total number of unique non-self-loop edges $M$.
    pub num_edges: usize,
    /// Number of vertices sharing an exact `f32` `(x, y)` coordinate with at least one other vertex.
    pub duplicate_nodes_f32: usize,
    /// Fraction `duplicate_nodes_f32 / num_nodes` (`0.0` when `num_nodes == 0`).
    pub duplicate_ratio_f32: f64,
    /// Number of vertices sharing an exact `f64` `(x, y)` coordinate with at least one other vertex.
    pub duplicate_nodes_f64: usize,
    /// Fraction `duplicate_nodes_f64 / num_nodes` (`0.0` when `num_nodes == 0`).
    pub duplicate_ratio_f64: f64,
    /// Excess duplicate fraction $1 - n_{\mathrm{uniq}} / N$ using exact `f64` coordinates (`0.0` when `num_nodes == 0`).
    pub excess_duplicate_ratio: f64,
    /// Fraction of vertices with at least one other distinct vertex within Euclidean distance
    /// $\tau = 10^{-7} \cdot \mathrm{diag}$ ($\mathrm{diag} = \sqrt{\mathrm{bbox\_width}^2 + \mathrm{bbox\_height}^2}$;
    /// `0.0` when `num_nodes <= 1`, `1.0` when `num_nodes >= 2` and `diag == 0.0`).
    pub near_collision_ratio: f64,
    /// Number of unique non-self-loop edges whose endpoints coincide (`hypot(dx, dy) == 0.0`).
    pub zero_length_edges: usize,
    /// Fraction `zero_length_edges / num_edges` (`0.0` when `num_edges == 0`).
    pub zero_length_edge_ratio: f64,
    /// Axis-aligned bounding-box width `max_x - min_x`.
    pub bbox_width: f64,
    /// Axis-aligned bounding-box height `max_y - min_y`.
    pub bbox_height: f64,
    /// `true` when `num_nodes > 0` and either `bbox_width <= 1e-12` or `bbox_height <= 1e-12`.
    pub is_bbox_degenerate: bool,
}

#[inline]
pub(super) fn canonical_f32(v: f32) -> f32 {
    if v == 0.0 { 0.0 } else { v }
}

#[inline]
pub(super) fn canonical_f64(v: f64) -> f64 {
    if v == 0.0 { 0.0 } else { v }
}

/// Evaluates duplicate-node (`f32`/`f64`), zero-length edge, and bounding-box degeneracies.
///
/// Canonicalizes `-0.0` to `+0.0` before bitwise key comparison.
pub fn layout_degeneracy<W, S: Scalar>(
    graph: &CsrGraph<W>,
    layout: &Layout2d<S>,
) -> CoreResult<LayoutDegeneracy> {
    validate_graph_layout(graph, layout, "layout_degeneracy")?;
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
    let mut points = Vec::with_capacity(n);
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
        points.push([x64, y64]);

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
        let ui = u as usize;
        let vi = v as usize;
        if (xs[ui].to_f64() - xs[vi].to_f64()).hypot(ys[ui].to_f64() - ys[vi].to_f64()) == 0.0 {
            zero_edges += 1;
        }
    }

    let bbox_width = (max_x - min_x).max(0.0);
    let bbox_height = (max_y - min_y).max(0.0);
    let is_bbox_degenerate = bbox_width <= 1e-12 || bbox_height <= 1e-12;
    let diag = bbox_width.hypot(bbox_height);
    let near_collision_ratio = if n <= 1 {
        0.0
    } else if diag == 0.0 {
        1.0
    } else {
        let tau = 1e-7 * diag;
        let tree = StaticKdTree::<2, f64>::build(&points)?;
        let mut near_collisions = 0usize;
        for i in 0..n {
            if let Some((_, d2)) = tree.nearest_excluding_self(i)?
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
