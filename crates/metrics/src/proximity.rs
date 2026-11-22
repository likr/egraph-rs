//! Spatial proximity metrics (`gabriel_graph_property`, `node_resolution`, `dRNG`, `dGG`)
//! accelerated via `StaticKdTree<2, f64>`.

use std::cmp::Ordering;
use std::f64::consts::{FRAC_PI_2, FRAC_PI_3, PI, TAU};

use egraph_core::{CoreResult, CsrGraph, Layout2d, Scalar, VisitStamp};
use egraph_spatial::StaticKdTree;

use crate::{
    build_kdtree_f64, build_undirected_adjacency, extract_unique_edges, validate_graph_layout,
    validate_tree_layout,
};

const NUM_SECTORS: usize = 16;
const SECTOR_WIDTH: f64 = TAU / NUM_SECTORS as f64;

#[derive(Debug, Clone, Copy)]
enum ProximityMode {
    Gabriel,
    RelativeNeighborhood,
}

/// Evaluates violations of the Gabriel graph condition across all edges in `graph`.
///
/// For each edge $(u, v)$, computes the disk with diameter $\overline{uv}$ (center $c_{uv}$,
/// radius $r_{uv}$) and accumulates $\max(0, r_{uv} - \|p_i - c_{uv}\|)^2$ over vertices
/// inside the disk via [`StaticKdTree::within_radius_into`].
pub fn gabriel_graph_property<W, S: Scalar>(
    graph: &CsrGraph<W>,
    layout: &Layout2d<S>,
) -> CoreResult<f64> {
    validate_graph_layout(graph, layout, "gabriel_graph_property")?;
    if layout.len() <= 2 || graph.num_edges() == 0 {
        return Ok(0.0);
    }
    let tree = build_kdtree_f64(layout)?;
    gabriel_graph_property_with_tree(graph, layout, &tree)
}

/// Computes [`gabriel_graph_property`] reusing a prebuilt [`StaticKdTree<2, f64>`].
pub fn gabriel_graph_property_with_tree<W, S: Scalar>(
    graph: &CsrGraph<W>,
    layout: &Layout2d<S>,
    tree: &StaticKdTree<2, f64>,
) -> CoreResult<f64> {
    validate_graph_layout(graph, layout, "gabriel_graph_property_with_tree")?;
    validate_tree_layout(tree, layout, "gabriel_graph_property_with_tree")?;
    let n = layout.len();
    if n <= 2 || graph.num_edges() == 0 {
        return Ok(0.0);
    }

    let xs = layout.x();
    let ys = layout.y();
    let edges = extract_unique_edges(graph);
    let mut disk_buf: Vec<(usize, f64)> = Vec::new();
    let mut violation_sum = 0.0_f64;

    for (u, v) in edges {
        let ui = u as usize;
        let vi = v as usize;
        let x1 = xs[ui].to_f64();
        let y1 = ys[ui].to_f64();
        let x2 = xs[vi].to_f64();
        let y2 = ys[vi].to_f64();
        let r = 0.5 * (x1 - x2).hypot(y1 - y2);
        if r <= 0.0 {
            continue;
        }
        let cx = 0.5 * (x1 + x2);
        let cy = 0.5 * (y1 + y2);
        tree.within_radius_into(&[cx, cy], r, &mut disk_buf)?;
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

/// Computes the one-sided hinge node-resolution violation penalty:
/// $\sum_{0 \le i < j < N} \max\left(0, 1 - \frac{\|x_i - x_j\|}{r \cdot d_{\max}}\right)^2$,
/// where $r = 1 / \sqrt{N}$ and $d_{\max}$ is the exact Euclidean diameter of `layout`.
pub fn node_resolution<S: Scalar>(layout: &Layout2d<S>) -> CoreResult<f64> {
    layout.validate_finite()?;
    let n = layout.len();
    if n <= 1 {
        return Ok(0.0);
    }
    let tree = build_kdtree_f64(layout)?;
    node_resolution_with_tree(layout, &tree)
}

/// Computes [`node_resolution`] reusing a prebuilt [`StaticKdTree<2, f64>`].
pub fn node_resolution_with_tree<S: Scalar>(
    layout: &Layout2d<S>,
    tree: &StaticKdTree<2, f64>,
) -> CoreResult<f64> {
    layout.validate_finite()?;
    validate_tree_layout(tree, layout, "node_resolution_with_tree")?;
    let n = layout.len();
    if n <= 1 {
        return Ok(0.0);
    }

    let d_max = euclidean_diameter_2d(layout);
    if d_max <= 0.0 {
        // Completely collapsed layout: every pair sits at distance 0 < ideal spacing.
        let num_pairs = (n as f64) * ((n - 1) as f64) * 0.5;
        return Ok(num_pairs);
    }

    let threshold = d_max / (n as f64).sqrt();
    let xs = layout.x();
    let ys = layout.y();
    let mut buf: Vec<(usize, f64)> = Vec::new();
    let mut sum = 0.0_f64;

    for i in 0..n {
        let q = [xs[i].to_f64(), ys[i].to_f64()];
        tree.within_radius_into(&q, threshold, &mut buf)?;
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
/// over the 2D coordinates in `layout`.
pub fn relative_neighborhood_edges<S: Scalar>(layout: &Layout2d<S>) -> CoreResult<Vec<(u32, u32)>> {
    let tree = build_kdtree_f64(layout)?;
    relative_neighborhood_edges_with_tree(layout, &tree)
}

/// Extracts [`relative_neighborhood_edges`] reusing a prebuilt [`StaticKdTree<2, f64>`].
pub fn relative_neighborhood_edges_with_tree<S: Scalar>(
    layout: &Layout2d<S>,
    tree: &StaticKdTree<2, f64>,
) -> CoreResult<Vec<(u32, u32)>> {
    layout.validate_finite()?;
    validate_tree_layout(tree, layout, "relative_neighborhood_edges_with_tree")?;
    proximity_edges_with_tree(layout, tree, ProximityMode::RelativeNeighborhood)
}

/// Extracts exact undirected Gabriel Graph (`GG`) edges `(u, v)` (`u < v`)
/// over the 2D coordinates in `layout`.
pub fn gabriel_graph_edges<S: Scalar>(layout: &Layout2d<S>) -> CoreResult<Vec<(u32, u32)>> {
    let tree = build_kdtree_f64(layout)?;
    gabriel_graph_edges_with_tree(layout, &tree)
}

/// Extracts [`gabriel_graph_edges`] reusing a prebuilt [`StaticKdTree<2, f64>`].
pub fn gabriel_graph_edges_with_tree<S: Scalar>(
    layout: &Layout2d<S>,
    tree: &StaticKdTree<2, f64>,
) -> CoreResult<Vec<(u32, u32)>> {
    layout.validate_finite()?;
    validate_tree_layout(tree, layout, "gabriel_graph_edges_with_tree")?;
    proximity_edges_with_tree(layout, tree, ProximityMode::Gabriel)
}

/// Computes shape-based Relative Neighborhood Graph preservation (`dRNG`) in `[0.0, 1.0]`
/// as Mean Jaccard Similarity between graph 1-hop neighborhoods and spatial `RNG(layout)`.
pub fn relative_neighborhood_preservation<W, S: Scalar>(
    graph: &CsrGraph<W>,
    layout: &Layout2d<S>,
) -> CoreResult<f64> {
    validate_graph_layout(graph, layout, "relative_neighborhood_preservation")?;
    if layout.is_empty() {
        return Ok(0.0);
    }
    let tree = build_kdtree_f64(layout)?;
    relative_neighborhood_preservation_with_tree(graph, layout, &tree)
}

/// Computes [`relative_neighborhood_preservation`] reusing a prebuilt [`StaticKdTree<2, f64>`].
pub fn relative_neighborhood_preservation_with_tree<W, S: Scalar>(
    graph: &CsrGraph<W>,
    layout: &Layout2d<S>,
    tree: &StaticKdTree<2, f64>,
) -> CoreResult<f64> {
    validate_graph_layout(
        graph,
        layout,
        "relative_neighborhood_preservation_with_tree",
    )?;
    validate_tree_layout(tree, layout, "relative_neighborhood_preservation_with_tree")?;
    if layout.is_empty() {
        return Ok(0.0);
    }
    let rng_edges = proximity_edges_with_tree(layout, tree, ProximityMode::RelativeNeighborhood)?;
    Ok(mean_jaccard_graph_vs_edges(graph, &rng_edges))
}

/// Computes shape-based Gabriel Graph preservation (`dGG`) in `[0.0, 1.0]`
/// as Mean Jaccard Similarity between graph 1-hop neighborhoods and spatial `GG(layout)`.
pub fn gabriel_graph_preservation<W, S: Scalar>(
    graph: &CsrGraph<W>,
    layout: &Layout2d<S>,
) -> CoreResult<f64> {
    validate_graph_layout(graph, layout, "gabriel_graph_preservation")?;
    if layout.is_empty() {
        return Ok(0.0);
    }
    let tree = build_kdtree_f64(layout)?;
    gabriel_graph_preservation_with_tree(graph, layout, &tree)
}

/// Computes [`gabriel_graph_preservation`] reusing a prebuilt [`StaticKdTree<2, f64>`].
pub fn gabriel_graph_preservation_with_tree<W, S: Scalar>(
    graph: &CsrGraph<W>,
    layout: &Layout2d<S>,
    tree: &StaticKdTree<2, f64>,
) -> CoreResult<f64> {
    validate_graph_layout(graph, layout, "gabriel_graph_preservation_with_tree")?;
    validate_tree_layout(tree, layout, "gabriel_graph_preservation_with_tree")?;
    if layout.is_empty() {
        return Ok(0.0);
    }
    let gg_edges = proximity_edges_with_tree(layout, tree, ProximityMode::Gabriel)?;
    Ok(mean_jaccard_graph_vs_edges(graph, &gg_edges))
}

fn mean_jaccard_graph_vs_edges<W>(graph: &CsrGraph<W>, prox_edges: &[(u32, u32)]) -> f64 {
    let n = graph.num_nodes();
    if n == 0 {
        return 0.0;
    }

    let mut prox_adj = vec![Vec::new(); n];
    for &(u, v) in prox_edges {
        prox_adj[u as usize].push(v);
        prox_adj[v as usize].push(u);
    }

    let directed_adj = if graph.is_directed() {
        Some(build_undirected_adjacency(graph))
    } else {
        None
    };

    let mut in_ng = VisitStamp::new(n);
    let mut sum_jaccard = 0.0_f64;

    for i in 0..n {
        in_ng.clear();
        in_ng.mark(i);
        let row: &[u32] = match &directed_adj {
            Some(adj) => &adj[i],
            None => graph.neighbors(i),
        };
        let mut deg_g = 0usize;
        for &v in row {
            if in_ng.visit(v as usize) {
                deg_g += 1;
            }
        }
        let prox_row = &prox_adj[i];
        let deg_p = prox_row.len();
        if deg_g == 0 && deg_p == 0 {
            sum_jaccard += 1.0;
            continue;
        }
        let mut intersection = 0usize;
        for &v in prox_row {
            if in_ng.is_visited(v as usize) {
                intersection += 1;
            }
        }
        let union = deg_g + deg_p - intersection;
        sum_jaccard += intersection as f64 / union as f64;
    }

    sum_jaccard / n as f64
}

fn proximity_edges_with_tree<S: Scalar>(
    layout: &Layout2d<S>,
    tree: &StaticKdTree<2, f64>,
    mode: ProximityMode,
) -> CoreResult<Vec<(u32, u32)>> {
    let n = layout.len();
    if n < 2 {
        return Ok(Vec::new());
    }

    let xs = layout.x();
    let ys = layout.y();
    let mut pts = Vec::with_capacity(n);
    for i in 0..n {
        pts.push([xs[i].to_f64(), ys[i].to_f64()]);
    }

    let mut sector_bounds = vec![[f64::INFINITY; NUM_SECTORS]; n];
    let mut r_max = vec![f64::INFINITY; n];
    let mut covered = vec![false; n];
    let mut uncovered = Vec::new();
    let mut knn_buf: Vec<(usize, f64)> = Vec::new();

    // Pass 1: derive angular sector upper bounds from local k-NN witnesses so interior
    // vertices only inspect O(1) candidates within `r_max[i]`.
    for i in 0..n {
        let mut k = 32usize.min(n - 1);
        tree.k_nearest_excluding_self_into(i, k, &mut knn_buf)?;
        update_sector_bounds(i, &pts, &knn_buf, mode, &mut sector_bounds[i]);
        let mut bound = sector_bounds[i].iter().copied().fold(0.0_f64, f64::max);

        if !bound.is_finite() && n - 1 > 32 {
            k = 128usize.min(n - 1);
            tree.k_nearest_excluding_self_into(i, k, &mut knn_buf)?;
            update_sector_bounds(i, &pts, &knn_buf, mode, &mut sector_bounds[i]);
            bound = sector_bounds[i].iter().copied().fold(0.0_f64, f64::max);
        }

        r_max[i] = bound;
        if bound.is_finite() {
            covered[i] = true;
        } else {
            uncovered.push(i);
        }
    }

    let mut edges = Vec::new();
    let mut cand_buf: Vec<(usize, f64)> = Vec::new();
    let mut witness_buf: Vec<(usize, f64)> = Vec::new();

    // Pass 2a: enumerate all valid edges with at least one covered endpoint.
    for i in 0..n {
        if !covered[i] {
            continue;
        }
        tree.within_radius_into(&pts[i], r_max[i], &mut cand_buf)?;
        for &(j, d_ij_sq) in &cand_buf {
            if j == i || d_ij_sq <= 0.0 {
                continue;
            }
            if covered[j] && j < i {
                continue;
            }
            let s_j = sector_index(pts[j][0] - pts[i][0], pts[j][1] - pts[i][1]);
            let d_ij = d_ij_sq.sqrt();
            if d_ij > sector_bounds[i][s_j] {
                continue;
            }
            if is_proximity_edge(i, j, d_ij_sq, &pts, tree, mode, &mut witness_buf)? {
                let (a, b) = if i < j { (i, j) } else { (j, i) };
                edges.push((a as u32, b as u32));
            }
        }
    }

    // Pass 2b: remaining pairs whose endpoints both lie on open/boundary wedges.
    for (idx_a, &u) in uncovered.iter().enumerate() {
        for &v in &uncovered[(idx_a + 1)..] {
            let dx = pts[v][0] - pts[u][0];
            let dy = pts[v][1] - pts[u][1];
            let d_uv_sq = dx * dx + dy * dy;
            if d_uv_sq <= 0.0 {
                continue;
            }
            let d_uv = d_uv_sq.sqrt();
            let su = sector_index(dx, dy);
            if d_uv > sector_bounds[u][su] {
                continue;
            }
            let sv = sector_index(-dx, -dy);
            if d_uv > sector_bounds[v][sv] {
                continue;
            }
            if is_proximity_edge(u, v, d_uv_sq, &pts, tree, mode, &mut witness_buf)? {
                edges.push((u as u32, v as u32));
            }
        }
    }

    edges.sort_unstable();
    edges.dedup();
    Ok(edges)
}

#[inline]
fn wrap_abs_angle(mut d: f64) -> f64 {
    while d > PI {
        d -= TAU;
    }
    while d < -PI {
        d += TAU;
    }
    d.abs()
}

#[inline]
fn sector_index(dx: f64, dy: f64) -> usize {
    let theta = dy.atan2(dx);
    let idx = ((theta + PI) / SECTOR_WIDTH).floor() as isize;
    idx.clamp(0, (NUM_SECTORS - 1) as isize) as usize
}

fn update_sector_bounds(
    center: usize,
    pts: &[[f64; 2]],
    witnesses: &[(usize, f64)],
    mode: ProximityMode,
    bounds: &mut [f64; NUM_SECTORS],
) {
    let px = pts[center][0];
    let py = pts[center][1];
    let max_allow = match mode {
        ProximityMode::Gabriel => FRAC_PI_2 - 1e-3,
        ProximityMode::RelativeNeighborhood => FRAC_PI_3 - 1e-3,
    };

    for &(r_idx, d_r_sq) in witnesses {
        if d_r_sq <= 0.0 {
            continue;
        }
        let dx = pts[r_idx][0] - px;
        let dy = pts[r_idx][1] - py;
        let d_r = d_r_sq.sqrt();
        let theta_r = dy.atan2(dx);

        for (s, bound_slot) in bounds.iter_mut().enumerate() {
            let phi0 = -PI + (s as f64) * SECTOR_WIDTH;
            let phi1 = phi0 + SECTOR_WIDTH;
            let max_delta = wrap_abs_angle(theta_r - phi0).max(wrap_abs_angle(theta_r - phi1));
            if max_delta < max_allow {
                let candidate = match mode {
                    ProximityMode::Gabriel => (d_r / max_delta.cos()) * (1.0 + 1e-7),
                    ProximityMode::RelativeNeighborhood => d_r * (1.0 + 1e-7),
                };
                if candidate < *bound_slot {
                    *bound_slot = candidate;
                }
            }
        }
    }
}

fn is_proximity_edge(
    i: usize,
    j: usize,
    d_ij_sq: f64,
    pts: &[[f64; 2]],
    tree: &StaticKdTree<2, f64>,
    mode: ProximityMode,
    witness_buf: &mut Vec<(usize, f64)>,
) -> CoreResult<bool> {
    let mid = [0.5 * (pts[i][0] + pts[j][0]), 0.5 * (pts[i][1] + pts[j][1])];
    let query_r_sq = match mode {
        ProximityMode::Gabriel => 0.25 * d_ij_sq * (1.0 + 1e-11),
        ProximityMode::RelativeNeighborhood => 0.75 * d_ij_sq * (1.0 + 1e-11),
    };
    tree.within_radius_sq_into(&mid, query_r_sq, witness_buf)?;

    for &(r, _) in witness_buf.iter() {
        if r == i || r == j {
            continue;
        }
        let d_ir_sq = dist_sq_2d(pts[i], pts[r]);
        let d_jr_sq = dist_sq_2d(pts[j], pts[r]);
        let blocked = match mode {
            // Use 1e-12 relative ULP tolerance so rotated cocircular grids and sqrt(3)/trig
            // equilateral lattices behave identically to exact Euclidean geometry.
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

#[inline]
fn dist_sq_2d(a: [f64; 2], b: [f64; 2]) -> f64 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    dx * dx + dy * dy
}

#[inline]
fn cross_2d(o: [f64; 2], a: [f64; 2], b: [f64; 2]) -> f64 {
    (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
}

/// Computes the exact Euclidean diameter $\max_{i, j} \|p_i - p_j\|_2$ in $O(N \log N)$ time
/// via Andrew's monotone chain convex hull and rotating calipers.
fn euclidean_diameter_2d<S: Scalar>(layout: &Layout2d<S>) -> f64 {
    let n = layout.len();
    if n <= 1 {
        return 0.0;
    }
    let xs = layout.x();
    let ys = layout.y();
    if n <= 64 {
        let mut max_sq = 0.0_f64;
        for i in 0..n {
            let xi = xs[i].to_f64();
            let yi = ys[i].to_f64();
            for j in (i + 1)..n {
                let dx = xi - xs[j].to_f64();
                let dy = yi - ys[j].to_f64();
                max_sq = max_sq.max(dx * dx + dy * dy);
            }
        }
        return max_sq.sqrt();
    }

    let mut pts: Vec<[f64; 2]> = (0..n).map(|i| [xs[i].to_f64(), ys[i].to_f64()]).collect();
    pts.sort_unstable_by(|a, b| a[0].total_cmp(&b[0]).then_with(|| a[1].total_cmp(&b[1])));
    pts.dedup_by(|b, a| a[0] == b[0] && a[1] == b[1]);

    if pts.len() <= 1 {
        return 0.0;
    }
    if pts.len() == 2 {
        return dist_sq_2d(pts[0], pts[1]).sqrt();
    }

    let mut hull: Vec<[f64; 2]> = Vec::with_capacity(pts.len());
    for &p in &pts {
        while hull.len() >= 2 && cross_2d(hull[hull.len() - 2], hull[hull.len() - 1], p) <= 0.0 {
            hull.pop();
        }
        hull.push(p);
    }
    let lower_len = hull.len();
    for &p in pts.iter().rev().skip(1) {
        while hull.len() > lower_len
            && cross_2d(hull[hull.len() - 2], hull[hull.len() - 1], p) <= 0.0
        {
            hull.pop();
        }
        hull.push(p);
    }
    hull.pop();

    let m = hull.len();
    if m <= 1 {
        return 0.0;
    }
    if m == 2 {
        return dist_sq_2d(hull[0], hull[1]).sqrt();
    }

    // Standard rotating calipers: walk antipodal vertex `j` around the CCW convex hull.
    let mut max_sq = 0.0_f64;
    let mut j = 1usize;
    for i in 0..m {
        let next_i = (i + 1) % m;
        loop {
            let next_j = (j + 1) % m;
            let cur_area = cross_2d(hull[i], hull[next_i], hull[j]).abs();
            let next_area = cross_2d(hull[i], hull[next_i], hull[next_j]).abs();
            match next_area.total_cmp(&cur_area) {
                Ordering::Greater => j = next_j,
                Ordering::Equal => {
                    max_sq = max_sq.max(dist_sq_2d(hull[i], hull[next_j]));
                    max_sq = max_sq.max(dist_sq_2d(hull[next_i], hull[next_j]));
                    break;
                }
                Ordering::Less => break,
            }
        }
        max_sq = max_sq.max(dist_sq_2d(hull[i], hull[j]));
        max_sq = max_sq.max(dist_sq_2d(hull[next_i], hull[j]));
    }
    max_sq.sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use egraph_core::{CoreError, DeterministicRng};

    fn brute_gg_edges(layout: &Layout2d<f64>) -> Vec<(u32, u32)> {
        let n = layout.len();
        let xs = layout.x();
        let ys = layout.y();
        let mut out = Vec::new();
        for i in 0..n {
            for j in (i + 1)..n {
                let d_ij_sq = (xs[i] - xs[j]).powi(2) + (ys[i] - ys[j]).powi(2);
                if d_ij_sq <= 0.0 {
                    continue;
                }
                let mut ok = true;
                for r in 0..n {
                    if r == i || r == j {
                        continue;
                    }
                    let d_ir_sq = (xs[i] - xs[r]).powi(2) + (ys[i] - ys[r]).powi(2);
                    let d_jr_sq = (xs[j] - xs[r]).powi(2) + (ys[j] - ys[r]).powi(2);
                    if d_ir_sq + d_jr_sq <= d_ij_sq * (1.0 + 1e-12) {
                        ok = false;
                        break;
                    }
                }
                if ok {
                    out.push((i as u32, j as u32));
                }
            }
        }
        out
    }

    fn brute_rng_edges(layout: &Layout2d<f64>) -> Vec<(u32, u32)> {
        let n = layout.len();
        let xs = layout.x();
        let ys = layout.y();
        let mut out = Vec::new();
        for i in 0..n {
            for j in (i + 1)..n {
                let d_ij_sq = (xs[i] - xs[j]).powi(2) + (ys[i] - ys[j]).powi(2);
                if d_ij_sq <= 0.0 {
                    continue;
                }
                let mut ok = true;
                for r in 0..n {
                    if r == i || r == j {
                        continue;
                    }
                    let d_ir_sq = (xs[i] - xs[r]).powi(2) + (ys[i] - ys[r]).powi(2);
                    let d_jr_sq = (xs[j] - xs[r]).powi(2) + (ys[j] - ys[r]).powi(2);
                    let thresh = d_ij_sq * (1.0 - 1e-12);
                    if d_ir_sq < thresh && d_jr_sq < thresh {
                        ok = false;
                        break;
                    }
                }
                if ok {
                    out.push((i as u32, j as u32));
                }
            }
        }
        out
    }

    #[test]
    fn test_planar_crystalline_drng_and_dgg_equal_one() {
        // 1. 4x4 square lattice (planar crystalline grid).
        let mut grid_edges = Vec::new();
        let mut gx = Vec::new();
        let mut gy = Vec::new();
        for r in 0..4u32 {
            for c in 0..4u32 {
                let id = r * 4 + c;
                gx.push(c as f64);
                gy.push(r as f64);
                if c + 1 < 4 {
                    grid_edges.push((id, id + 1));
                }
                if r + 1 < 4 {
                    grid_edges.push((id, id + 4));
                }
            }
        }
        let grid = CsrGraph::<f64>::from_edges(16, &grid_edges, None, false, false).unwrap();
        let grid_layout = Layout2d::from_xy(gx, gy).unwrap();
        assert!(
            (relative_neighborhood_preservation(&grid, &grid_layout).unwrap() - 1.0).abs() < 1e-12
        );
        assert!((gabriel_graph_preservation(&grid, &grid_layout).unwrap() - 1.0).abs() < 1e-12);
        assert_eq!(gabriel_graph_property(&grid, &grid_layout).unwrap(), 0.0);
        // On a 4x4 unit grid, r * d_max = (3 * sqrt(2)) / 4 > 1.0, so the 24 unit edges violate slightly.
        let thresh_4x4 = 3.0 * 2.0_f64.sqrt() / 4.0;
        let expected_vr_4x4 = 24.0 * (1.0 - 1.0 / thresh_4x4).powi(2);
        assert!((node_resolution(&grid_layout).unwrap() - expected_vr_4x4).abs() < 1e-12);

        // On a 3x3 unit grid, r * d_max = (2 * sqrt(2)) / 3 < 1.0, so node_resolution == 0.0.
        let mut g3x = Vec::new();
        let mut g3y = Vec::new();
        for r in 0..3 {
            for c in 0..3 {
                g3x.push(c as f64);
                g3y.push(r as f64);
            }
        }
        let grid3_layout = Layout2d::from_xy(g3x, g3y).unwrap();
        assert_eq!(node_resolution(&grid3_layout).unwrap(), 0.0);

        // Rotated 4x4 grid preserves exact dRNG == 1.0 and dGG == 1.0 despite floating-point rotation.
        let (st, ct) = (0.37_f64).sin_cos();
        let rx: Vec<f64> = grid_layout
            .x()
            .iter()
            .zip(grid_layout.y().iter())
            .map(|(&x, &y)| ct * x - st * y)
            .collect();
        let ry: Vec<f64> = grid_layout
            .x()
            .iter()
            .zip(grid_layout.y().iter())
            .map(|(&x, &y)| st * x + ct * y)
            .collect();
        let rot_grid_layout = Layout2d::from_xy(rx, ry).unwrap();
        assert!(
            (relative_neighborhood_preservation(&grid, &rot_grid_layout).unwrap() - 1.0).abs()
                < 1e-12
        );
        assert!((gabriel_graph_preservation(&grid, &rot_grid_layout).unwrap() - 1.0).abs() < 1e-12);

        // 2. Triangular crystalline lattice (equilateral triangles).
        let rows = 3usize;
        let cols = 4usize;
        let mut tx = Vec::new();
        let mut ty = Vec::new();
        let mut tri_edges = Vec::new();
        let dy = 0.5 * 3.0_f64.sqrt();
        for r in 0..rows {
            for c in 0..cols {
                let id = (r * cols + c) as u32;
                let shift = if r % 2 == 1 { 0.5 } else { 0.0 };
                tx.push(c as f64 + shift);
                ty.push(r as f64 * dy);
                if c + 1 < cols {
                    tri_edges.push((id, id + 1));
                }
                if r + 1 < rows {
                    tri_edges.push((id, id + cols as u32));
                    if r % 2 == 0 && c > 0 {
                        tri_edges.push((id, id + cols as u32 - 1));
                    } else if r % 2 == 1 && c + 1 < cols {
                        tri_edges.push((id, id + cols as u32 + 1));
                    }
                }
            }
        }
        let tri = CsrGraph::<f64>::from_edges(rows * cols, &tri_edges, None, false, false).unwrap();
        let tri_layout = Layout2d::from_xy(tx, ty).unwrap();
        assert!(
            (relative_neighborhood_preservation(&tri, &tri_layout).unwrap() - 1.0).abs() < 1e-12
        );
        assert!((gabriel_graph_preservation(&tri, &tri_layout).unwrap() - 1.0).abs() < 1e-12);

        // 3. Regular 10-cycle.
        let c_edges: Vec<(u32, u32)> = (0..10u32).map(|i| (i, (i + 1) % 10)).collect();
        let cycle = CsrGraph::<f64>::from_edges(10, &c_edges, None, false, false).unwrap();
        let cx: Vec<f64> = (0..10).map(|i| (i as f64 * TAU / 10.0).cos()).collect();
        let cy: Vec<f64> = (0..10).map(|i| (i as f64 * TAU / 10.0).sin()).collect();
        let cycle_layout = Layout2d::from_xy(cx, cy).unwrap();
        assert!(
            (relative_neighborhood_preservation(&cycle, &cycle_layout).unwrap() - 1.0).abs()
                < 1e-12
        );
        assert!((gabriel_graph_preservation(&cycle, &cycle_layout).unwrap() - 1.0).abs() < 1e-12);

        // Scrambled random positions drop preservation strictly below 1.0.
        let mut rng = DeterministicRng::from_seed_u64(123);
        let rand_layout = Layout2d::from_xy(
            (0..16).map(|_| rng.next_f64() * 10.0).collect(),
            (0..16).map(|_| rng.next_f64() * 10.0).collect(),
        )
        .unwrap();
        assert!(relative_neighborhood_preservation(&grid, &rand_layout).unwrap() < 0.7);
        assert!(gabriel_graph_preservation(&grid, &rand_layout).unwrap() < 0.7);
    }

    #[test]
    fn test_kdtree_gg_rng_and_diameter_exact_brute_force_parity() {
        // Test random, clustered+outlier, collinear, cocircular, and duplicate point sets.
        for seed in 1..=10u64 {
            let mut rng = DeterministicRng::from_seed_u64(seed);
            let n = 80usize;
            let mut xs = Vec::with_capacity(n);
            let mut ys = Vec::with_capacity(n);
            for i in 0..n {
                if i < 65 {
                    xs.push((rng.next_f64() - 0.5) * 4.0);
                    ys.push((rng.next_f64() - 0.5) * 4.0);
                } else {
                    // Distant outlier ring testing sector bounds across long edges.
                    xs.push((rng.next_f64() - 0.5) * 200.0);
                    ys.push((rng.next_f64() - 0.5) * 200.0);
                }
            }
            let layout = Layout2d::from_xy(xs, ys).unwrap();
            assert_eq!(
                gabriel_graph_edges(&layout).unwrap(),
                brute_gg_edges(&layout)
            );
            assert_eq!(
                relative_neighborhood_edges(&layout).unwrap(),
                brute_rng_edges(&layout)
            );

            // Verify O(N log N) rotating calipers diameter equals O(N^2) brute force.
            let got_diam = euclidean_diameter_2d(&layout);
            let mut expected_sq = 0.0_f64;
            for i in 0..n {
                for j in (i + 1)..n {
                    let dx = layout.x()[i] - layout.x()[j];
                    let dy = layout.y()[i] - layout.y()[j];
                    expected_sq = expected_sq.max(dx * dx + dy * dy);
                }
            }
            assert!((got_diam - expected_sq.sqrt()).abs() < 1e-10);
        }

        // Duplicate points & collapsed layout checks.
        let dup = Layout2d::from_xy(vec![0.0, 0.0, 2.0, 2.0], vec![0.0, 0.0, 0.0, 1.0]).unwrap();
        assert_eq!(gabriel_graph_edges(&dup).unwrap(), brute_gg_edges(&dup));
        assert_eq!(
            relative_neighborhood_edges(&dup).unwrap(),
            brute_rng_edges(&dup)
        );

        let collapsed = Layout2d::<f64>::zeros(6);
        let k6 = CsrGraph::<f64>::from_edges(
            6,
            &[(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)],
            None,
            false,
            false,
        )
        .unwrap();
        assert_eq!(
            relative_neighborhood_preservation(&k6, &collapsed).unwrap(),
            0.0
        );
        assert_eq!(gabriel_graph_preservation(&k6, &collapsed).unwrap(), 0.0);
        assert_eq!(node_resolution(&collapsed).unwrap(), 15.0);

        // Large (> 64) collinear, coincident, and skewed parallelogram diameter tests.
        let col100 =
            Layout2d::from_xy((0..100).map(|i| i as f64 * 0.5).collect(), vec![2.0; 100]).unwrap();
        assert!((euclidean_diameter_2d(&col100) - 49.5).abs() < 1e-12);

        let dup100 = Layout2d::<f64>::zeros(100);
        assert_eq!(euclidean_diameter_2d(&dup100), 0.0);

        let mut px = vec![0.0, 10.0, 14.0, 4.0];
        let mut py = vec![0.0, 0.0, 3.0, 3.0];
        let mut rng_p = DeterministicRng::from_seed_u64(99);
        for _ in 0..96 {
            let a = rng_p.next_f64();
            let b = rng_p.next_f64();
            px.push(a * 10.0 + b * 4.0);
            py.push(b * 3.0);
        }
        let para = Layout2d::from_xy(px, py).unwrap();
        assert!((euclidean_diameter_2d(&para) - 14.0_f64.hypot(3.0)).abs() < 1e-12);
    }

    #[test]
    fn test_gabriel_property_and_node_resolution_brute_force_parity() {
        let g =
            CsrGraph::<f64>::from_edges(4, &[(0, 1), (1, 2), (2, 3), (0, 2)], None, false, false)
                .unwrap();
        let layout = Layout2d::from_xy(vec![0.0, 4.0, 2.0, 1.9], vec![0.0, 0.0, 1.0, 0.2]).unwrap();
        let gg_prop = gabriel_graph_property(&g, &layout).unwrap();
        // Verify exact match against naive O(|E| * N) sum.
        let mut expected_gg = 0.0_f64;
        for (u, v) in extract_unique_edges(&g) {
            let ui = u as usize;
            let vi = v as usize;
            let cx = 0.5 * (layout.x()[ui] + layout.x()[vi]);
            let cy = 0.5 * (layout.y()[ui] + layout.y()[vi]);
            let r = 0.5 * (layout.x()[ui] - layout.x()[vi]).hypot(layout.y()[ui] - layout.y()[vi]);
            for i in 0..4 {
                let d: f64 = (layout.x()[i] - cx).hypot(layout.y()[i] - cy);
                expected_gg += (r - d).max(0.0).powi(2);
            }
        }
        assert!((gg_prop - expected_gg).abs() < 1e-12);

        let vr = node_resolution(&layout).unwrap();
        let d_max = euclidean_diameter_2d(&layout);
        let thresh = d_max / 2.0;
        let mut expected_vr = 0.0_f64;
        for i in 0..4 {
            for j in (i + 1)..4 {
                let d = (layout.x()[i] - layout.x()[j]).hypot(layout.y()[i] - layout.y()[j]);
                expected_vr += (1.0 - d / thresh).max(0.0).powi(2);
            }
        }
        assert!((vr - expected_vr).abs() < 1e-12);

        let bad = Layout2d::from_xy(vec![f64::NAN, 1.0], vec![0.0, 0.0]).unwrap();
        assert!(matches!(
            node_resolution(&bad),
            Err(CoreError::NonFiniteValue { .. })
        ));
    }
}
