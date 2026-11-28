use std::f64::consts::TAU;

use egraph_core::{CoreResult, CsrGraph, Layout2d, Scalar};

use crate::build_undirected_adjacency;
use crate::spatial_index::validate_graph_layout;

/// DRGraph Eq. (19) minimum-angle resolution summary (`a1` and `a2`) in `[0.0, 1.0]`.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Default)]
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
