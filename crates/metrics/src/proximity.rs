//! Spatial proximity metrics (`gabriel_graph_property`, `node_resolution`, and Eades et al.
//! 2017 unweighted shape-based `S_RNG` / `S_GG`) accelerated via `StaticKdTree<2, f64>`.

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

/// Evaluates violations of the Gabriel graph condition across all unique edges yielded by
/// `extract_unique_edges(graph)`.
///
/// For each edge `(u, v)`, computes the disk with diameter $\overline{uv}$ (center $c_{uv}$,
/// radius $r_{uv}$) and accumulates $\max(0, r_{uv} - \|p_i - c_{uv}\|)^2$ over vertices
/// inside the disk via [`StaticKdTree::within_radius_into`]. On directed graphs, every stored
/// directed edge `(u, v)` is evaluated independently (so reciprocal pairs `(u, v)` and `(v, u)`
/// are both summed), and zero-length edges (`r_{uv} <= 0.0`) are skipped.
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

/// Computes Eades et al. (2017) unweighted shape-based Relative Neighborhood Graph preservation
/// (`S_RNG`) in `[0.0, 1.0]` as Mean Jaccard Similarity (`MJS`) between undirected graph
/// 1-hop neighborhoods and spatial `RNG(layout)`.
///
/// Note: This implements the classical unweighted `S_RNG` metric of Eades et al. (JGAA 2017),
/// not Hong et al. (PacificVis 2022) degree-sensitive `dRNG` or Fang et al. (PacificVis 2025)
/// degree-constrained `dcRNG` (both deferred pending primary specification PDFs). Input graphs
/// are deduplicated as undirected simple neighborhoods (directed graphs are symmetrized and
/// self-loops are excluded). Returns `0.0` for `N == 0`, and scores `N == 1` or mutually
/// isolated vertices (`deg_G == 0 && deg_RNG == 0`) as `1.0` (`0/0` convention).
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

/// Computes Eades et al. (2017) unweighted shape-based Gabriel Graph preservation (`S_GG`)
/// in `[0.0, 1.0]` as Mean Jaccard Similarity (`MJS`) between undirected graph 1-hop
/// neighborhoods and spatial `GG(layout)`.
///
/// Note: This implements the classical unweighted `S_GG` metric of Eades et al. (JGAA 2017),
/// not Hong et al. (PacificVis 2022) degree-sensitive `dGG` or Fang et al. (PacificVis 2025)
/// degree-constrained `dcGG` (both deferred pending primary specification PDFs). Input graphs
/// are deduplicated as undirected simple neighborhoods (directed graphs are symmetrized and
/// self-loops are excluded). Returns `0.0` for `N == 0`, and scores `N == 1` or mutually
/// isolated vertices (`deg_G == 0 && deg_GG == 0`) as `1.0` (`0/0` convention).
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

/// Eades et al. (2017) Eq. (2) unweighted Mean Jaccard Similarity (`MJS`) between graph 1-hop
/// and proximity edges.
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

    let mut min_x = f64::INFINITY;
    let mut max_x = f64::NEG_INFINITY;
    let mut min_y = f64::INFINITY;
    let mut max_y = f64::NEG_INFINITY;
    for &[x, y] in &pts {
        min_x = min_x.min(x);
        max_x = max_x.max(x);
        min_y = min_y.min(y);
        max_y = max_y.max(y);
    }
    let d_max_bound = (max_x - min_x).hypot(max_y - min_y) * (1.0 + 1e-9);

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
        update_sector_bounds(i, &pts, &knn_buf, mode, d_max_bound, &mut sector_bounds[i]);
        let mut bound = sector_bounds[i].iter().copied().fold(0.0_f64, f64::max);

        if !bound.is_finite() && n - 1 > 32 {
            k = 128usize.min(n - 1);
            tree.k_nearest_excluding_self_into(i, k, &mut knn_buf)?;
            update_sector_bounds(i, &pts, &knn_buf, mode, d_max_bound, &mut sector_bounds[i]);
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
        let query_r = (r_max[i] * (1.0 + 1e-12)).min(f64::MAX);
        tree.within_radius_into(&pts[i], query_r, &mut cand_buf)?;
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
    d_max_bound: f64,
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
                let cos_min = max_delta.cos();
                if let ProximityMode::RelativeNeighborhood = mode {
                    // For RNG, `is_proximity_edge` requires `d_jr^2 < d_ij^2 * (1 - 1e-12)`.
                    // A tiny witness `d_r << 1e-12 * d_ij` fails to shrink `d_jr^2` below the
                    // tolerance threshold, so a witness only safely prunes up to `d_max_bound`
                    // when `2 * d_r * cos_min - d_r^2/d_ij >= 1e-11 * d_ij`.
                    if d_r < d_max_bound * (1e-11 / cos_min) {
                        continue;
                    }
                }
                let candidate = match mode {
                    ProximityMode::Gabriel => (d_r / cos_min) * (1.0 + 1e-7),
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
    // Difference-based midpoint avoids catastrophic cancellation at large coordinate offsets,
    // and ULP padding on `query_r_sq` covers the remaining midpoint representability error.
    let mid = [
        pts[i][0] + 0.5 * (pts[j][0] - pts[i][0]),
        pts[i][1] + 0.5 * (pts[j][1] - pts[i][1]),
    ];
    let base_r_sq = match mode {
        ProximityMode::Gabriel => 0.25 * d_ij_sq * (1.0 + 1e-11),
        ProximityMode::RelativeNeighborhood => 0.75 * d_ij_sq * (1.0 + 1e-11),
    };
    let max_coord_abs = pts[i][0]
        .abs()
        .max(pts[i][1].abs())
        .max(pts[j][0].abs())
        .max(pts[j][1].abs());
    let eps_mid = 4.0 * f64::EPSILON * max_coord_abs;
    let padded_r = base_r_sq.sqrt() + eps_mid;
    let query_r_sq = (padded_r * padded_r).min(f64::MAX);
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

/// Error-free Knuth `TwoSum` for `a - b`, exact for all finite `f64` when `|a|, |b| <= 0.25 * f64::MAX`.
#[inline]
fn two_diff(a: f64, b: f64) -> (f64, f64) {
    let s = a - b;
    let b_virt = s - a;
    let a_virt = s - b_virt;
    let b_err = (-b) - b_virt;
    let a_err = a - a_virt;
    (s, a_err + b_err)
}

#[inline]
fn f64_to_mant_exp(v: f64) -> Option<(i64, i32)> {
    if v == 0.0 {
        return None;
    }
    let bits = v.to_bits();
    let neg = (bits >> 63) != 0;
    let exp_bits = ((bits >> 52) & 0x7ff) as i32;
    let frac = bits & ((1_u64 << 52) - 1);
    let (mag, exp) = if exp_bits == 0 {
        (frac as i64, -1074)
    } else {
        ((frac | (1_u64 << 52)) as i64, exp_bits - 1075)
    };
    Some((if neg { -mag } else { mag }, exp))
}

#[inline]
fn push_prod_term(u: f64, v: f64, sign: i8, terms: &mut [(i128, i32); 16], len: &mut usize) {
    if let (Some((mu, eu)), Some((mv, ev))) = (f64_to_mant_exp(u), f64_to_mant_exp(v)) {
        let prod = (mu as i128) * (mv as i128) * (sign as i128);
        terms[*len] = (prod, eu + ev);
        *len += 1;
    }
}

fn push_cross_terms(
    o: [f64; 2],
    a: [f64; 2],
    b: [f64; 2],
    sign: i8,
    terms: &mut [(i128, i32); 16],
    len: &mut usize,
) {
    let (dx1_hi, dx1_lo) = two_diff(a[0], o[0]);
    let (dy1_hi, dy1_lo) = two_diff(a[1], o[1]);
    let (dx2_hi, dx2_lo) = two_diff(b[0], o[0]);
    let (dy2_hi, dy2_lo) = two_diff(b[1], o[1]);

    push_prod_term(dx1_hi, dy2_hi, sign, terms, len);
    push_prod_term(dx1_hi, dy2_lo, sign, terms, len);
    push_prod_term(dx1_lo, dy2_hi, sign, terms, len);
    push_prod_term(dx1_lo, dy2_lo, sign, terms, len);

    push_prod_term(dy1_hi, dx2_hi, -sign, terms, len);
    push_prod_term(dy1_hi, dx2_lo, -sign, terms, len);
    push_prod_term(dy1_lo, dx2_hi, -sign, terms, len);
    push_prod_term(dy1_lo, dx2_lo, -sign, terms, len);
}

#[inline]
fn add_limb_triple(limbs: &mut [u64; 68], word: usize, w0: u64, w1: u64, w2: u64) {
    let (sum0, c0) = limbs[word].overflowing_add(w0);
    limbs[word] = sum0;
    let (sum1a, c1a) = limbs[word + 1].overflowing_add(w1);
    let (sum1b, c1b) = sum1a.overflowing_add(u64::from(c0));
    limbs[word + 1] = sum1b;
    let (sum2a, c2a) = limbs[word + 2].overflowing_add(w2);
    let (sum2b, c2b) = sum2a.overflowing_add(u64::from(c1a) + u64::from(c1b));
    limbs[word + 2] = sum2b;
    let mut carry = u64::from(c2a) + u64::from(c2b);
    let mut idx = word + 3;
    while carry != 0 && idx < limbs.len() {
        let (sum, c) = limbs[idx].overflowing_add(carry);
        limbs[idx] = sum;
        carry = u64::from(c);
        idx += 1;
    }
}

/// Evaluates the exact sign of `\sum_k m_k * 2^{e_k}` with zero floating-point error,
/// covering the full IEEE-754 finite exponent range (`|coords| <= 0.25 * f64::MAX`).
fn exact_sum_sign(terms: &[(i128, i32)]) -> Ordering {
    if terms.is_empty() {
        return Ordering::Equal;
    }
    let mut min_e = i32::MAX;
    let mut max_e = i32::MIN;
    for &(_, e) in terms {
        min_e = min_e.min(e);
        max_e = max_e.max(e);
    }
    let num_limbs = ((max_e - min_e) as usize / 64 + 4).min(68);
    let mut pos = [0_u64; 68];
    let mut neg = [0_u64; 68];

    for &(m, e) in terms {
        if m == 0 {
            continue;
        }
        let shift = (e - min_e) as usize;
        let word = shift / 64;
        let bit = (shift % 64) as u32;
        let abs_m = m.unsigned_abs();
        let w0 = (abs_m as u64) << bit;
        let rest = abs_m >> (64 - bit);
        let w1 = rest as u64;
        let w2 = (rest >> 64) as u64;
        if m > 0 {
            add_limb_triple(&mut pos, word, w0, w1, w2);
        } else {
            add_limb_triple(&mut neg, word, w0, w1, w2);
        }
    }

    for idx in (0..num_limbs).rev() {
        match pos[idx].cmp(&neg[idx]) {
            Ordering::Equal => {}
            ord => return ord,
        }
    }
    Ordering::Equal
}

fn orient2d_exact(o: [f64; 2], a: [f64; 2], b: [f64; 2]) -> Ordering {
    let mut terms = [(0_i128, 0_i32); 16];
    let mut len = 0usize;
    push_cross_terms(o, a, b, 1, &mut terms, &mut len);
    exact_sum_sign(&terms[..len])
}

/// Exact orientation sign of `(a - o) x (b - o)` with a fast `f64` error-bound filter.
#[inline]
fn orient2d(o: [f64; 2], a: [f64; 2], b: [f64; 2]) -> Ordering {
    let det_left = (a[0] - o[0]) * (b[1] - o[1]);
    let det_right = (a[1] - o[1]) * (b[0] - o[0]);
    let det = det_left - det_right;
    let det_sum = det_left.abs() + det_right.abs();
    if det_sum.is_finite() && (det_sum == 0.0 || det_sum > 1e-280) {
        let err_bound = 8.0 * f64::EPSILON * det_sum;
        if det.abs() > err_bound {
            return if det > 0.0 {
                Ordering::Greater
            } else {
                Ordering::Less
            };
        }
    }
    orient2d_exact(o, a, b)
}

/// Exact comparison of `|(a - o) x (b1 - o)|` vs `|(a - o) x (b2 - o)|` with a fast `f64` filter.
#[inline]
fn cross_abs_cmp(o: [f64; 2], a: [f64; 2], b1: [f64; 2], b2: [f64; 2]) -> Ordering {
    let dx = a[0] - o[0];
    let dy = a[1] - o[1];

    let l1 = dx * (b1[1] - o[1]);
    let r1 = dy * (b1[0] - o[0]);
    let det1 = l1 - r1;
    let sum1 = l1.abs() + r1.abs();

    let l2 = dx * (b2[1] - o[1]);
    let r2 = dy * (b2[0] - o[0]);
    let det2 = l2 - r2;
    let sum2 = l2.abs() + r2.abs();

    if sum1.is_finite()
        && sum2.is_finite()
        && (sum1 == 0.0 || sum1 > 1e-280)
        && (sum2 == 0.0 || sum2 > 1e-280)
    {
        let err1 = 8.0 * f64::EPSILON * sum1;
        let err2 = 8.0 * f64::EPSILON * sum2;
        let abs1 = det1.abs();
        let abs2 = det2.abs();
        if abs1 - err1 > abs2 + err2 {
            return Ordering::Greater;
        }
        if abs2 - err2 > abs1 + err1 {
            return Ordering::Less;
        }
    }

    let s1 = orient2d_exact(o, a, b1);
    let s2 = orient2d_exact(o, a, b2);
    match (s1, s2) {
        (Ordering::Equal, Ordering::Equal) => Ordering::Equal,
        (Ordering::Equal, _) => Ordering::Less,
        (_, Ordering::Equal) => Ordering::Greater,
        _ => {
            let mut terms = [(0_i128, 0_i32); 16];
            let mut len = 0usize;
            let sign1 = if s1 == Ordering::Greater { 1 } else { -1 };
            let sign2 = if s2 == Ordering::Greater { -1 } else { 1 };
            push_cross_terms(o, a, b1, sign1, &mut terms, &mut len);
            push_cross_terms(o, a, b2, sign2, &mut terms, &mut len);
            exact_sum_sign(&terms[..len])
        }
    }
}

#[inline]
fn dist_2d_safe(a: [f64; 2], b: [f64; 2]) -> f64 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    let sq = dx * dx + dy * dy;
    if sq.is_finite() && (sq > 0.0 || (dx == 0.0 && dy == 0.0)) {
        sq.sqrt()
    } else {
        dx.hypot(dy)
    }
}

/// Computes the exact Euclidean diameter $\max_{i, j} \|p_i - p_j\|_2$ in $O(N \log N)$ time
/// via exact-predicate Andrew's monotone chain convex hull and rotating calipers.
fn euclidean_diameter_2d<S: Scalar>(layout: &Layout2d<S>) -> f64 {
    let n = layout.len();
    if n <= 1 {
        return 0.0;
    }
    let xs = layout.x();
    let ys = layout.y();
    if n <= 64 {
        let mut best_d = 0.0_f64;
        for i in 0..n {
            let p_i = [xs[i].to_f64(), ys[i].to_f64()];
            for j in (i + 1)..n {
                let p_j = [xs[j].to_f64(), ys[j].to_f64()];
                best_d = best_d.max(dist_2d_safe(p_i, p_j));
            }
        }
        return best_d;
    }

    // Canonicalize `-0.0` to `+0.0` so `total_cmp` and `dedup_by` (`==`) agree
    // and never leave non-adjacent `(0.0, 0.0)` duplicates that collapse the hull.
    let mut max_abs = 0.0_f64;
    let mut pts: Vec<[f64; 2]> = (0..n)
        .map(|i| {
            let x = xs[i].to_f64() + 0.0;
            let y = ys[i].to_f64() + 0.0;
            max_abs = max_abs.max(x.abs()).max(y.abs());
            [x, y]
        })
        .collect();
    if max_abs == 0.0 {
        return 0.0;
    }
    // Scale by an exact power of 2 when coordinates exceed `0.25 * f64::MAX` so `two_diff`
    // and intermediate differences inside exact orientation predicates can never overflow.
    let pre_scale = if max_abs > 1e300 { 0.125 } else { 1.0 };
    if pre_scale != 1.0 {
        for p in &mut pts {
            p[0] *= pre_scale;
            p[1] *= pre_scale;
        }
    }
    let inv_scale = 1.0 / pre_scale;

    pts.sort_unstable_by(|a, b| a[0].total_cmp(&b[0]).then_with(|| a[1].total_cmp(&b[1])));
    pts.dedup_by(|b, a| a[0] == b[0] && a[1] == b[1]);

    if pts.len() <= 1 {
        return 0.0;
    }
    if pts.len() == 2 {
        return dist_2d_safe(pts[0], pts[1]) * inv_scale;
    }

    let mut hull: Vec<[f64; 2]> = Vec::with_capacity(pts.len());
    for &p in &pts {
        while hull.len() >= 2
            && orient2d(hull[hull.len() - 2], hull[hull.len() - 1], p) != Ordering::Greater
        {
            hull.pop();
        }
        hull.push(p);
    }
    let lower_len = hull.len();
    for &p in pts.iter().rev().skip(1) {
        while hull.len() > lower_len
            && orient2d(hull[hull.len() - 2], hull[hull.len() - 1], p) != Ordering::Greater
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
        return dist_2d_safe(hull[0], hull[1]) * inv_scale;
    }
    if m <= 64 {
        let mut best_d = 0.0_f64;
        for i in 0..m {
            for j in (i + 1)..m {
                best_d = best_d.max(dist_2d_safe(hull[i], hull[j]));
            }
        }
        return best_d * inv_scale;
    }

    // Standard rotating calipers with exact area comparisons on the strictly convex CCW hull.
    let mut best_d = 0.0_f64;
    let mut j = 1usize;
    for i in 0..m {
        let next_i = (i + 1) % m;
        best_d = best_d.max(dist_2d_safe(hull[i], hull[next_i]));
        let mut steps = 0usize;
        while steps < m {
            let next_j = (j + 1) % m;
            match cross_abs_cmp(hull[i], hull[next_i], hull[next_j], hull[j]) {
                Ordering::Greater => {
                    j = next_j;
                    steps += 1;
                }
                Ordering::Equal => {
                    best_d = best_d.max(dist_2d_safe(hull[i], hull[next_j]));
                    best_d = best_d.max(dist_2d_safe(hull[next_i], hull[next_j]));
                    break;
                }
                Ordering::Less => break,
            }
        }
        best_d = best_d.max(dist_2d_safe(hull[i], hull[j]));
        best_d = best_d.max(dist_2d_safe(hull[next_i], hull[j]));
    }
    best_d * inv_scale
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
    fn test_planar_crystalline_s_rng_and_s_gg_equal_one() {
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

        // Rotated 4x4 grid preserves exact S_RNG == 1.0 and S_GG == 1.0 despite floating-point rotation.
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

    fn brute_gabriel_graph_property<W>(graph: &CsrGraph<W>, layout: &Layout2d<f64>) -> f64 {
        let n = layout.len();
        let xs = layout.x();
        let ys = layout.y();
        let mut sum = 0.0_f64;
        for (u, v) in extract_unique_edges(graph) {
            let ui = u as usize;
            let vi = v as usize;
            let r = 0.5 * (xs[ui] - xs[vi]).hypot(ys[ui] - ys[vi]);
            if r <= 0.0 {
                continue;
            }
            let cx = 0.5 * (xs[ui] + xs[vi]);
            let cy = 0.5 * (ys[ui] + ys[vi]);
            for i in 0..n {
                if i == ui || i == vi {
                    continue;
                }
                let d = (xs[i] - cx).hypot(ys[i] - cy);
                if d < r {
                    let viol = r - d;
                    sum += viol * viol;
                }
            }
        }
        sum
    }

    fn brute_node_resolution(layout: &Layout2d<f64>) -> f64 {
        let n = layout.len();
        if n <= 1 {
            return 0.0;
        }
        let xs = layout.x();
        let ys = layout.y();
        let mut d_max_sq = 0.0_f64;
        for i in 0..n {
            for j in (i + 1)..n {
                let d_sq = (xs[i] - xs[j]).powi(2) + (ys[i] - ys[j]).powi(2);
                d_max_sq = d_max_sq.max(d_sq);
            }
        }
        if d_max_sq <= 0.0 {
            return (n as f64) * ((n - 1) as f64) * 0.5;
        }
        let thresh = d_max_sq.sqrt() / (n as f64).sqrt();
        let mut sum = 0.0_f64;
        for i in 0..n {
            for j in (i + 1)..n {
                let d = (xs[i] - xs[j]).hypot(ys[i] - ys[j]);
                if d < thresh {
                    let hinge = 1.0 - d / thresh;
                    sum += hinge * hinge;
                }
            }
        }
        sum
    }

    fn assert_proximity_determinism_tree_and_dispatcher<W: Scalar, S: Scalar>(
        graph: &CsrGraph<W>,
        layout: &Layout2d<S>,
    ) {
        use crate::dispatcher::{MetricConfig, MetricTarget, evaluate_metrics};

        let tree = build_kdtree_f64(layout).unwrap();

        let gg_prop_1 = gabriel_graph_property(graph, layout).unwrap();
        let gg_prop_2 = gabriel_graph_property(graph, layout).unwrap();
        let gg_prop_tree = gabriel_graph_property_with_tree(graph, layout, &tree).unwrap();
        assert_eq!(gg_prop_1.to_bits(), gg_prop_2.to_bits());
        assert_eq!(gg_prop_1.to_bits(), gg_prop_tree.to_bits());

        let nr_1 = node_resolution(layout).unwrap();
        let nr_2 = node_resolution(layout).unwrap();
        let nr_tree = node_resolution_with_tree(layout, &tree).unwrap();
        assert_eq!(nr_1.to_bits(), nr_2.to_bits());
        assert_eq!(nr_1.to_bits(), nr_tree.to_bits());

        let rng_edges_1 = relative_neighborhood_edges(layout).unwrap();
        let rng_edges_tree = relative_neighborhood_edges_with_tree(layout, &tree).unwrap();
        assert_eq!(rng_edges_1, rng_edges_tree);

        let gg_edges_1 = gabriel_graph_edges(layout).unwrap();
        let gg_edges_tree = gabriel_graph_edges_with_tree(layout, &tree).unwrap();
        assert_eq!(gg_edges_1, gg_edges_tree);

        let s_rng_1 = relative_neighborhood_preservation(graph, layout).unwrap();
        let s_rng_2 = relative_neighborhood_preservation(graph, layout).unwrap();
        let s_rng_tree =
            relative_neighborhood_preservation_with_tree(graph, layout, &tree).unwrap();
        assert_eq!(s_rng_1.to_bits(), s_rng_2.to_bits());
        assert_eq!(s_rng_1.to_bits(), s_rng_tree.to_bits());

        let s_gg_1 = gabriel_graph_preservation(graph, layout).unwrap();
        let s_gg_2 = gabriel_graph_preservation(graph, layout).unwrap();
        let s_gg_tree = gabriel_graph_preservation_with_tree(graph, layout, &tree).unwrap();
        assert_eq!(s_gg_1.to_bits(), s_gg_2.to_bits());
        assert_eq!(s_gg_1.to_bits(), s_gg_tree.to_bits());

        let targets = [
            MetricTarget::GabrielGraphProperty,
            MetricTarget::NodeResolution,
            MetricTarget::RelativeNeighborhoodPreservation,
            MetricTarget::GabrielGraphPreservation,
        ];
        let report = evaluate_metrics(graph, layout, &targets, &MetricConfig::default()).unwrap();
        assert_eq!(
            report.gabriel_graph_property.unwrap().to_bits(),
            gg_prop_1.to_bits()
        );
        assert_eq!(report.node_resolution.unwrap().to_bits(), nr_1.to_bits());
        assert_eq!(
            report.relative_neighborhood_preservation.unwrap().to_bits(),
            s_rng_1.to_bits()
        );
        assert_eq!(
            report.gabriel_graph_preservation.unwrap().to_bits(),
            s_gg_1.to_bits()
        );
    }

    #[test]
    fn test_empty_and_singleton_boundaries_all_12_functions_and_dispatcher() {
        // N = 0 (empty graph & empty layout)
        let g0 = CsrGraph::<f64>::from_edges(0, &[], None, false, false).unwrap();
        let l0 = Layout2d::<f64>::zeros(0);
        let t0 = build_kdtree_f64(&l0).unwrap();
        assert_eq!(gabriel_graph_property(&g0, &l0).unwrap(), 0.0);
        assert_eq!(
            gabriel_graph_property_with_tree(&g0, &l0, &t0).unwrap(),
            0.0
        );
        assert_eq!(node_resolution(&l0).unwrap(), 0.0);
        assert_eq!(node_resolution_with_tree(&l0, &t0).unwrap(), 0.0);
        assert!(relative_neighborhood_edges(&l0).unwrap().is_empty());
        assert!(
            relative_neighborhood_edges_with_tree(&l0, &t0)
                .unwrap()
                .is_empty()
        );
        assert!(gabriel_graph_edges(&l0).unwrap().is_empty());
        assert!(gabriel_graph_edges_with_tree(&l0, &t0).unwrap().is_empty());
        assert_eq!(relative_neighborhood_preservation(&g0, &l0).unwrap(), 0.0);
        assert_eq!(
            relative_neighborhood_preservation_with_tree(&g0, &l0, &t0).unwrap(),
            0.0
        );
        assert_eq!(gabriel_graph_preservation(&g0, &l0).unwrap(), 0.0);
        assert_eq!(
            gabriel_graph_preservation_with_tree(&g0, &l0, &t0).unwrap(),
            0.0
        );
        assert_proximity_determinism_tree_and_dispatcher(&g0, &l0);

        // N = 1 (singleton graph & layout): S_RNG = 1.0, S_GG = 1.0 (0/0 convention).
        let g1 = CsrGraph::<f64>::from_edges(1, &[], None, false, false).unwrap();
        let l1 = Layout2d::from_xy(vec![3.5], vec![-2.25]).unwrap();
        let t1 = build_kdtree_f64(&l1).unwrap();
        assert_eq!(gabriel_graph_property(&g1, &l1).unwrap(), 0.0);
        assert_eq!(
            gabriel_graph_property_with_tree(&g1, &l1, &t1).unwrap(),
            0.0
        );
        assert_eq!(node_resolution(&l1).unwrap(), 0.0);
        assert_eq!(node_resolution_with_tree(&l1, &t1).unwrap(), 0.0);
        assert!(relative_neighborhood_edges(&l1).unwrap().is_empty());
        assert!(
            relative_neighborhood_edges_with_tree(&l1, &t1)
                .unwrap()
                .is_empty()
        );
        assert!(gabriel_graph_edges(&l1).unwrap().is_empty());
        assert!(gabriel_graph_edges_with_tree(&l1, &t1).unwrap().is_empty());
        assert_eq!(relative_neighborhood_preservation(&g1, &l1).unwrap(), 1.0);
        assert_eq!(
            relative_neighborhood_preservation_with_tree(&g1, &l1, &t1).unwrap(),
            1.0
        );
        assert_eq!(gabriel_graph_preservation(&g1, &l1).unwrap(), 1.0);
        assert_eq!(
            gabriel_graph_preservation_with_tree(&g1, &l1, &t1).unwrap(),
            1.0
        );
        assert_proximity_determinism_tree_and_dispatcher(&g1, &l1);

        // f32 singleton check.
        let g1_32 = CsrGraph::<f32>::from_edges(1, &[], None, false, false).unwrap();
        let l1_32 = Layout2d::from_xy(vec![1.25_f32], vec![-0.5_f32]).unwrap();
        assert_proximity_determinism_tree_and_dispatcher(&g1_32, &l1_32);
    }

    #[test]
    fn test_edgeless_disconnected_and_isolated_vertex_cases() {
        // 1. Edgeless graph (|E| = 0, N = 4) with non-collocated layout.
        let g_edgeless = CsrGraph::<f64>::from_edges(4, &[], None, false, false).unwrap();
        let l_square =
            Layout2d::from_xy(vec![0.0, 2.0, 2.0, 0.0], vec![0.0, 0.0, 2.0, 2.0]).unwrap();
        assert_eq!(gabriel_graph_property(&g_edgeless, &l_square).unwrap(), 0.0);
        assert_eq!(
            relative_neighborhood_preservation(&g_edgeless, &l_square).unwrap(),
            0.0
        );
        assert_eq!(
            gabriel_graph_preservation(&g_edgeless, &l_square).unwrap(),
            0.0
        );
        assert!(
            (node_resolution(&l_square).unwrap() - brute_node_resolution(&l_square)).abs() < 1e-12
        );
        assert_eq!(
            gabriel_graph_edges(&l_square).unwrap(),
            brute_gg_edges(&l_square)
        );
        assert_eq!(
            relative_neighborhood_edges(&l_square).unwrap(),
            brute_rng_edges(&l_square)
        );
        assert_proximity_determinism_tree_and_dispatcher(&g_edgeless, &l_square);

        // 2. Two disjoint triangles: inter-component GG/RNG bridges lower S_RNG/S_GG < 1.0,
        // and an inter-component node placed inside a Gabriel disk contributes to gabriel_graph_property.
        let g_disjoint = CsrGraph::<f64>::from_edges(
            6,
            &[(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3)],
            None,
            false,
            false,
        )
        .unwrap();
        let l_disjoint = Layout2d::from_xy(
            vec![0.0, 4.0, 2.0, 2.0, 7.0, 7.0],
            vec![0.0, 0.0, 3.0, 0.5, 0.0, 2.0],
        )
        .unwrap();
        let gg_edges = gabriel_graph_edges(&l_disjoint).unwrap();
        let rng_edges = relative_neighborhood_edges(&l_disjoint).unwrap();
        assert_eq!(gg_edges, brute_gg_edges(&l_disjoint));
        assert_eq!(rng_edges, brute_rng_edges(&l_disjoint));
        let expected_s_gg = mean_jaccard_graph_vs_edges(&g_disjoint, &gg_edges);
        let expected_s_rng = mean_jaccard_graph_vs_edges(&g_disjoint, &rng_edges);
        assert!(expected_s_gg < 1.0 && expected_s_rng < 1.0);
        assert!(
            (gabriel_graph_preservation(&g_disjoint, &l_disjoint).unwrap() - expected_s_gg).abs()
                < 1e-12
        );
        assert!(
            (relative_neighborhood_preservation(&g_disjoint, &l_disjoint).unwrap()
                - expected_s_rng)
                .abs()
                < 1e-12
        );
        let expected_gg_prop = brute_gabriel_graph_property(&g_disjoint, &l_disjoint);
        assert!(expected_gg_prop > 0.0);
        assert!(
            (gabriel_graph_property(&g_disjoint, &l_disjoint).unwrap() - expected_gg_prop).abs()
                < 1e-12
        );
        assert_proximity_determinism_tree_and_dispatcher(&g_disjoint, &l_disjoint);

        // 3. Connected triangle (0,1,2) + isolated degree-0 vertex (3):
        // (a) Separated isolated vertex has spatial neighbors (deg_p >= 1) so it scores 0.0
        // and any cross-bridge to {0,1,2} lowers S_GG / S_RNG strictly below 1.0; verify exact MJS.
        let g_iso =
            CsrGraph::<f64>::from_edges(4, &[(0, 1), (1, 2), (2, 0)], None, false, false).unwrap();
        let l_iso_sep = Layout2d::from_xy(
            vec![0.0, 2.0, 1.0, 10.0],
            vec![0.0, 0.0, 3.0_f64.sqrt(), 0.0],
        )
        .unwrap();
        let sep_gg = brute_gg_edges(&l_iso_sep);
        let sep_rng = brute_rng_edges(&l_iso_sep);
        assert_eq!(gabriel_graph_edges(&l_iso_sep).unwrap(), sep_gg);
        assert_eq!(relative_neighborhood_edges(&l_iso_sep).unwrap(), sep_rng);
        assert_eq!(
            gabriel_graph_preservation(&g_iso, &l_iso_sep)
                .unwrap()
                .to_bits(),
            mean_jaccard_graph_vs_edges(&g_iso, &sep_gg).to_bits()
        );
        assert_eq!(
            relative_neighborhood_preservation(&g_iso, &l_iso_sep)
                .unwrap()
                .to_bits(),
            mean_jaccard_graph_vs_edges(&g_iso, &sep_rng).to_bits()
        );
        // Triangle (0,1,2) is preserved with 1-hop overlap, while vertex 3 has deg_g == 0 and deg_p >= 1 (0.0 Jaccard).
        assert!(gabriel_graph_preservation(&g_iso, &l_iso_sep).unwrap() < 1.0);
        assert!(relative_neighborhood_preservation(&g_iso, &l_iso_sep).unwrap() < 1.0);

        // (b) Isolated vertex co-located onto vertex 0 at (0.0, 0.0):
        // In GG, co-located points block all GG edges incident to 0 and 3 (deg_p == 0 for node 3 -> 0/0 = 1.0).
        // In RNG, co-located points do NOT block RNG edges to {1, 2} (deg_p == 2 for node 3 -> 0.0 Jaccard).
        let l_iso_coloc = Layout2d::from_xy(
            vec![0.0, 2.0, 1.0, 0.0],
            vec![0.0, 0.0, 3.0_f64.sqrt(), 0.0],
        )
        .unwrap();
        let coloc_gg = brute_gg_edges(&l_iso_coloc);
        let coloc_rng = brute_rng_edges(&l_iso_coloc);
        assert_eq!(gabriel_graph_edges(&l_iso_coloc).unwrap(), coloc_gg);
        assert_eq!(
            relative_neighborhood_edges(&l_iso_coloc).unwrap(),
            coloc_rng
        );
        assert!(
            !coloc_gg
                .iter()
                .any(|&(u, v)| u == 0 || v == 0 || u == 3 || v == 3)
        );
        assert!(coloc_rng.iter().any(|&(u, v)| u == 3 || v == 3));
        // For node 0: GG edges = {}, G edges = {1,2} -> 0.0. Nodes 1,2: GG = {1,2}, G = {0,1,2} -> 0.5 each. Node 3: 0/0 -> 1.0. Mean = 0.5.
        assert!((gabriel_graph_preservation(&g_iso, &l_iso_coloc).unwrap() - 0.5).abs() < 1e-12);
        assert_eq!(
            relative_neighborhood_preservation(&g_iso, &l_iso_coloc)
                .unwrap()
                .to_bits(),
            mean_jaccard_graph_vs_edges(&g_iso, &coloc_rng).to_bits()
        );
        assert_proximity_determinism_tree_and_dispatcher(&g_iso, &l_iso_coloc);
    }

    #[test]
    fn test_signed_zero_diameter_colocated_pass2b_and_gabriel_property_colocation() {
        // 1. Regression for signed-zero (-0.0 / +0.0) in euclidean_diameter_2d (n > 64 branch).
        let mut xs_y = vec![-0.0_f64, -0.0_f64];
        let mut ys_y = vec![0.0_f64, 1.0_f64];
        xs_y.extend(std::iter::repeat_n(0.0_f64, 63));
        ys_y.extend(std::iter::repeat_n(0.0_f64, 63));
        let layout_sz_y = Layout2d::from_xy(xs_y, ys_y).unwrap();
        assert_eq!(layout_sz_y.len(), 65);
        assert!((euclidean_diameter_2d(&layout_sz_y) - 1.0).abs() < 1e-12);
        let nr_sz_y = node_resolution(&layout_sz_y).unwrap();
        assert!((nr_sz_y - brute_node_resolution(&layout_sz_y)).abs() < 1e-10);
        // Must NOT trigger collapsed-layout branch (65 * 64 / 2 = 2080.0).
        assert!(nr_sz_y < 2080.0);

        let mut xs_x = vec![0.0_f64, 1.0_f64];
        let mut ys_x = vec![-0.0_f64, -0.0_f64];
        xs_x.extend(std::iter::repeat_n(0.0_f64, 63));
        ys_x.extend(std::iter::repeat_n(0.0_f64, 63));
        let layout_sz_x = Layout2d::from_xy(xs_x, ys_x).unwrap();
        assert!((euclidean_diameter_2d(&layout_sz_x) - 1.0).abs() < 1e-12);
        assert!(
            (node_resolution(&layout_sz_x).unwrap() - brute_node_resolution(&layout_sz_x)).abs()
                < 1e-10
        );

        // 2. > 128 co-located points (n = 160) exercising Pass 2b `uncovered` fallback.
        let mut rng = DeterministicRng::from_seed_u64(2026);
        let mut xs_160 = vec![0.0_f64; 135];
        let mut ys_160 = vec![0.0_f64; 135];
        for _ in 0..25 {
            xs_160.push((rng.next_f64() - 0.5) * 20.0 + 5.0);
            ys_160.push((rng.next_f64() - 0.5) * 20.0 + 5.0);
        }
        let layout_160 = Layout2d::from_xy(xs_160, ys_160).unwrap();
        assert_eq!(
            gabriel_graph_edges(&layout_160).unwrap(),
            brute_gg_edges(&layout_160)
        );
        assert_eq!(
            relative_neighborhood_edges(&layout_160).unwrap(),
            brute_rng_edges(&layout_160)
        );

        // 3. gabriel_graph_property with zero-length edge and co-located 3rd vertex.
        // Zero-length edge (0, 1) with p0 == p1 is skipped, even when vertex 2 sits nearby or on top.
        let g_zero_edge = CsrGraph::<f64>::from_edges(3, &[(0, 1)], None, false, false).unwrap();
        let l_zero_edge = Layout2d::from_xy(vec![1.0, 1.0, 1.0], vec![2.0, 2.0, 2.0]).unwrap();
        assert_eq!(
            gabriel_graph_property(&g_zero_edge, &l_zero_edge).unwrap(),
            0.0
        );

        // Non-zero edge (0, 1) from (0,0) to (4,0): radius r = 2.0, center (2,0).
        // Vertex 2 co-located at endpoint (0,0) -> dist == r, not strictly inside -> 0.0 violation.
        let l_endpoint_coloc = Layout2d::from_xy(vec![0.0, 4.0, 0.0], vec![0.0, 0.0, 0.0]).unwrap();
        assert_eq!(
            gabriel_graph_property(&g_zero_edge, &l_endpoint_coloc).unwrap(),
            0.0
        );

        // Vertex 2 at midpoint (2,0) -> dist == 0.0 < r -> violation = r^2 = 4.0.
        let l_midpoint = Layout2d::from_xy(vec![0.0, 4.0, 2.0], vec![0.0, 0.0, 0.0]).unwrap();
        assert!((gabriel_graph_property(&g_zero_edge, &l_midpoint).unwrap() - 4.0).abs() < 1e-12);
    }

    #[test]
    fn test_directed_reciprocal_and_sanitized_self_loop_parallel_edges() {
        let base_edges = [(0_u32, 1_u32), (1, 2), (2, 3), (3, 0), (0, 2)];
        let g_undir = CsrGraph::<f64>::from_edges(4, &base_edges, None, false, false).unwrap();

        let mut recip_edges = Vec::new();
        for &(u, v) in &base_edges {
            recip_edges.push((u, v));
            recip_edges.push((v, u));
        }
        let g_dir_recip = CsrGraph::<f64>::from_edges(4, &recip_edges, None, true, false).unwrap();
        let g_dir_oneway = CsrGraph::<f64>::from_edges(4, &base_edges, None, true, false).unwrap();

        let layout = Layout2d::from_xy(vec![0.0, 4.0, 2.0, 1.9], vec![0.0, 0.0, 1.0, 0.2]).unwrap();

        // S_RNG and S_GG deduplicate directed edges via undirected symmetrization.
        assert_eq!(
            relative_neighborhood_preservation(&g_undir, &layout)
                .unwrap()
                .to_bits(),
            relative_neighborhood_preservation(&g_dir_recip, &layout)
                .unwrap()
                .to_bits()
        );
        assert_eq!(
            relative_neighborhood_preservation(&g_undir, &layout)
                .unwrap()
                .to_bits(),
            relative_neighborhood_preservation(&g_dir_oneway, &layout)
                .unwrap()
                .to_bits()
        );
        assert_eq!(
            gabriel_graph_preservation(&g_undir, &layout)
                .unwrap()
                .to_bits(),
            gabriel_graph_preservation(&g_dir_recip, &layout)
                .unwrap()
                .to_bits()
        );
        assert_eq!(
            gabriel_graph_preservation(&g_undir, &layout)
                .unwrap()
                .to_bits(),
            gabriel_graph_preservation(&g_dir_oneway, &layout)
                .unwrap()
                .to_bits()
        );

        // gabriel_graph_property evaluates each directed edge yielded by extract_unique_edges.
        let prop_undir = gabriel_graph_property(&g_undir, &layout).unwrap();
        let prop_dir_recip = gabriel_graph_property(&g_dir_recip, &layout).unwrap();
        let prop_dir_oneway = gabriel_graph_property(&g_dir_oneway, &layout).unwrap();
        assert!(prop_undir > 0.0);
        assert!((prop_dir_recip - 2.0 * prop_undir).abs() < 1e-12);
        assert!((prop_dir_oneway - prop_undir).abs() < 1e-12);

        // Sanitized construction with self-loops and parallel/reverse duplicates.
        let dirty_edges = [
            (0_u32, 0_u32),
            (1, 1),
            (0, 1),
            (0, 1),
            (1, 0),
            (1, 2),
            (2, 3),
            (3, 0),
            (0, 2),
            (2, 0),
        ];
        let g_sanitized_undir =
            CsrGraph::<f64>::from_edges(4, &dirty_edges, None, false, true).unwrap();
        let g_sanitized_dir =
            CsrGraph::<f64>::from_edges(4, &dirty_edges, None, true, true).unwrap();

        assert_eq!(
            relative_neighborhood_preservation(&g_sanitized_undir, &layout)
                .unwrap()
                .to_bits(),
            relative_neighborhood_preservation(&g_undir, &layout)
                .unwrap()
                .to_bits()
        );
        assert_eq!(
            gabriel_graph_preservation(&g_sanitized_undir, &layout)
                .unwrap()
                .to_bits(),
            gabriel_graph_preservation(&g_undir, &layout)
                .unwrap()
                .to_bits()
        );
        assert_eq!(
            gabriel_graph_property(&g_sanitized_undir, &layout)
                .unwrap()
                .to_bits(),
            prop_undir.to_bits()
        );
        assert_eq!(
            relative_neighborhood_preservation(&g_sanitized_dir, &layout)
                .unwrap()
                .to_bits(),
            relative_neighborhood_preservation(&g_undir, &layout)
                .unwrap()
                .to_bits()
        );
        assert_eq!(
            gabriel_graph_preservation(&g_sanitized_dir, &layout)
                .unwrap()
                .to_bits(),
            gabriel_graph_preservation(&g_undir, &layout)
                .unwrap()
                .to_bits()
        );
        assert_proximity_determinism_tree_and_dispatcher(&g_dir_recip, &layout);
    }

    fn assert_all_12_reject_non_finite<W: Scalar, S: Scalar>(
        graph: &CsrGraph<W>,
        bad_layout: &Layout2d<S>,
        valid_tree_same_len: &StaticKdTree<2, f64>,
    ) {
        assert!(matches!(
            gabriel_graph_property(graph, bad_layout),
            Err(CoreError::NonFiniteValue { .. })
        ));
        assert!(matches!(
            gabriel_graph_property_with_tree(graph, bad_layout, valid_tree_same_len),
            Err(CoreError::NonFiniteValue { .. })
        ));
        assert!(matches!(
            node_resolution(bad_layout),
            Err(CoreError::NonFiniteValue { .. })
        ));
        assert!(matches!(
            node_resolution_with_tree(bad_layout, valid_tree_same_len),
            Err(CoreError::NonFiniteValue { .. })
        ));
        assert!(matches!(
            relative_neighborhood_edges(bad_layout),
            Err(CoreError::NonFiniteValue { .. })
        ));
        assert!(matches!(
            relative_neighborhood_edges_with_tree(bad_layout, valid_tree_same_len),
            Err(CoreError::NonFiniteValue { .. })
        ));
        assert!(matches!(
            gabriel_graph_edges(bad_layout),
            Err(CoreError::NonFiniteValue { .. })
        ));
        assert!(matches!(
            gabriel_graph_edges_with_tree(bad_layout, valid_tree_same_len),
            Err(CoreError::NonFiniteValue { .. })
        ));
        assert!(matches!(
            relative_neighborhood_preservation(graph, bad_layout),
            Err(CoreError::NonFiniteValue { .. })
        ));
        assert!(matches!(
            relative_neighborhood_preservation_with_tree(graph, bad_layout, valid_tree_same_len),
            Err(CoreError::NonFiniteValue { .. })
        ));
        assert!(matches!(
            gabriel_graph_preservation(graph, bad_layout),
            Err(CoreError::NonFiniteValue { .. })
        ));
        assert!(matches!(
            gabriel_graph_preservation_with_tree(graph, bad_layout, valid_tree_same_len),
            Err(CoreError::NonFiniteValue { .. })
        ));
    }

    #[test]
    fn test_non_finite_extreme_overflow_and_dimension_mismatch_all_12() {
        // 1. Non-finite (NaN, +Inf, -Inf) at N = 1 and N = 2 for f64 and f32 across all 12 functions.
        let g1_64 = CsrGraph::<f64>::from_edges(1, &[], None, false, false).unwrap();
        let t1 = build_kdtree_f64(&Layout2d::<f64>::zeros(1)).unwrap();
        for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let lx = Layout2d::from_xy(vec![bad], vec![0.0]).unwrap();
            let ly = Layout2d::from_xy(vec![0.0], vec![bad]).unwrap();
            assert_all_12_reject_non_finite(&g1_64, &lx, &t1);
            assert_all_12_reject_non_finite(&g1_64, &ly, &t1);
        }

        let g2_64 = CsrGraph::<f64>::from_edges(2, &[(0, 1)], None, false, false).unwrap();
        let t2 = build_kdtree_f64(&Layout2d::<f64>::zeros(2)).unwrap();
        for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let l = Layout2d::from_xy(vec![0.0, bad], vec![0.0, 1.0]).unwrap();
            assert_all_12_reject_non_finite(&g2_64, &l, &t2);
        }

        let g1_32 = CsrGraph::<f32>::from_edges(1, &[], None, false, false).unwrap();
        let g2_32 = CsrGraph::<f32>::from_edges(2, &[(0, 1)], None, false, false).unwrap();
        for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let l1 = Layout2d::from_xy(vec![bad], vec![0.0_f32]).unwrap();
            let l2 = Layout2d::from_xy(vec![0.0_f32, 1.0], vec![0.0_f32, bad]).unwrap();
            assert_all_12_reject_non_finite(&g1_32, &l1, &t1);
            assert_all_12_reject_non_finite(&g2_32, &l2, &t2);
        }

        // 2. Extreme finite coordinates (1e200): must never panic or hang across all 12 functions.
        let g3_64 =
            CsrGraph::<f64>::from_edges(3, &[(0, 1), (1, 2), (2, 0)], None, false, false).unwrap();
        let l_huge = Layout2d::from_xy(vec![0.0, 1e200, -1e200], vec![0.0, 1e200, 0.0]).unwrap();
        let t_huge = build_kdtree_f64(&l_huge).unwrap();
        for res in [
            gabriel_graph_property(&g3_64, &l_huge),
            gabriel_graph_property_with_tree(&g3_64, &l_huge, &t_huge),
            node_resolution(&l_huge),
            node_resolution_with_tree(&l_huge, &t_huge),
            relative_neighborhood_preservation(&g3_64, &l_huge),
            relative_neighborhood_preservation_with_tree(&g3_64, &l_huge, &t_huge),
            gabriel_graph_preservation(&g3_64, &l_huge),
            gabriel_graph_preservation_with_tree(&g3_64, &l_huge, &t_huge),
        ] {
            assert!(matches!(res, Ok(_) | Err(CoreError::NonFiniteValue { .. })));
        }
        assert!(matches!(
            relative_neighborhood_edges(&l_huge),
            Ok(_) | Err(CoreError::NonFiniteValue { .. })
        ));
        assert!(matches!(
            relative_neighborhood_edges_with_tree(&l_huge, &t_huge),
            Ok(_) | Err(CoreError::NonFiniteValue { .. })
        ));
        assert!(matches!(
            gabriel_graph_edges(&l_huge),
            Ok(_) | Err(CoreError::NonFiniteValue { .. })
        ));
        assert!(matches!(
            gabriel_graph_edges_with_tree(&l_huge, &t_huge),
            Ok(_) | Err(CoreError::NonFiniteValue { .. })
        ));

        // 3. DimensionMismatch: graph/layout mismatch and tree/layout mismatch across all applicable functions.
        let l4 = Layout2d::from_xy(vec![0.0, 1.0, 1.0, 0.0], vec![0.0, 0.0, 1.0, 1.0]).unwrap();
        let t4 = build_kdtree_f64(&l4).unwrap();
        let t3 = build_kdtree_f64(&Layout2d::<f64>::zeros(3)).unwrap();

        // Graph (3) vs layout (4).
        assert!(matches!(
            gabriel_graph_property(&g3_64, &l4),
            Err(CoreError::DimensionMismatch { .. })
        ));
        assert!(matches!(
            gabriel_graph_property_with_tree(&g3_64, &l4, &t4),
            Err(CoreError::DimensionMismatch { .. })
        ));
        assert!(matches!(
            relative_neighborhood_preservation(&g3_64, &l4),
            Err(CoreError::DimensionMismatch { .. })
        ));
        assert!(matches!(
            relative_neighborhood_preservation_with_tree(&g3_64, &l4, &t4),
            Err(CoreError::DimensionMismatch { .. })
        ));
        assert!(matches!(
            gabriel_graph_preservation(&g3_64, &l4),
            Err(CoreError::DimensionMismatch { .. })
        ));
        assert!(matches!(
            gabriel_graph_preservation_with_tree(&g3_64, &l4, &t4),
            Err(CoreError::DimensionMismatch { .. })
        ));

        // Graph (4) & layout (4) vs tree (3).
        let g4_64 = CsrGraph::<f64>::from_edges(4, &[(0, 1), (1, 2)], None, false, false).unwrap();
        assert!(matches!(
            gabriel_graph_property_with_tree(&g4_64, &l4, &t3),
            Err(CoreError::DimensionMismatch { .. })
        ));
        assert!(matches!(
            node_resolution_with_tree(&l4, &t3),
            Err(CoreError::DimensionMismatch { .. })
        ));
        assert!(matches!(
            relative_neighborhood_edges_with_tree(&l4, &t3),
            Err(CoreError::DimensionMismatch { .. })
        ));
        assert!(matches!(
            gabriel_graph_edges_with_tree(&l4, &t3),
            Err(CoreError::DimensionMismatch { .. })
        ));
        assert!(matches!(
            relative_neighborhood_preservation_with_tree(&g4_64, &l4, &t3),
            Err(CoreError::DimensionMismatch { .. })
        ));
        assert!(matches!(
            gabriel_graph_preservation_with_tree(&g4_64, &l4, &t3),
            Err(CoreError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn test_regression_rotated_collinear_and_near_collinear_diameter() {
        // Reproduces Bug 1: rotated collinear & near-collinear layouts (n > 64) across angles
        // including `(0.6, 0.8)` at n=90, `angle = 5.0959_f64` at n=80, n=200, and 50 random angles.
        let mut xs_0608 = Vec::with_capacity(90);
        let mut ys_0608 = Vec::with_capacity(90);
        let mut rng0 = DeterministicRng::from_seed_u64(43);
        for _ in 0..90 {
            let t = (rng0.next_f64() - 0.5) * 50.0;
            xs_0608.push(t * 0.6);
            ys_0608.push(t * 0.8);
        }
        let l_0608 = Layout2d::from_xy(xs_0608, ys_0608).unwrap();
        assert!((node_resolution(&l_0608).unwrap() - brute_node_resolution(&l_0608)).abs() < 1e-10);

        let mut angles = vec![5.0959_f64, 0.37_f64, 1.2345_f64, 2.468_f64];
        let mut rng_a = DeterministicRng::from_seed_u64(777);
        for _ in 0..46 {
            angles.push(rng_a.next_f64() * TAU);
        }

        for &eps in &[0.0_f64, 1e-15_f64] {
            for (k, &angle) in angles.iter().enumerate() {
                let (sa, ca) = angle.sin_cos();
                for &n in &[80usize, 200usize] {
                    if n == 200 && k >= 6 {
                        continue;
                    }
                    let mut rng = DeterministicRng::from_seed_u64(5000 + k as u64);
                    let mut xs = Vec::with_capacity(n);
                    let mut ys = Vec::with_capacity(n);
                    for _ in 0..n {
                        let t = (rng.next_f64() - 0.5) * 100.0;
                        let perp = (rng.next_f64() - 0.5) * 2.0 * eps;
                        xs.push(t * ca - perp * sa);
                        ys.push(t * sa + perp * ca);
                    }
                    let layout = Layout2d::from_xy(xs, ys).unwrap();
                    let got_diam = euclidean_diameter_2d(&layout);
                    let mut exp_sq = 0.0_f64;
                    for i in 0..n {
                        for j in (i + 1)..n {
                            let dx = layout.x()[i] - layout.x()[j];
                            let dy = layout.y()[i] - layout.y()[j];
                            exp_sq = exp_sq.max(dx * dx + dy * dy);
                        }
                    }
                    let exp_diam = exp_sq.sqrt();
                    assert!(
                        (got_diam - exp_diam).abs() <= 1e-11 * exp_diam.max(1.0),
                        "diameter mismatch at angle={angle}, eps={eps}, n={n}: got={got_diam} exp={exp_diam}"
                    );
                    let got_nr = node_resolution(&layout).unwrap();
                    let exp_nr = brute_node_resolution(&layout);
                    assert!(
                        (got_nr - exp_nr).abs() <= 1e-9 * exp_nr.max(1.0),
                        "node_resolution mismatch at angle={angle}, eps={eps}, n={n}: got={got_nr} exp={exp_nr}"
                    );
                }
            }
        }

        // Also test > 64 hull vertices on a rotated thin convex ellipse (aspect ratio 1e13)
        // to directly exercise the `m > 64` rotating calipers path with near-collinear jitter.
        let (sa, ca) = (5.0959_f64).sin_cos();
        let mut arc_x = Vec::with_capacity(200);
        let mut arc_y = Vec::with_capacity(200);
        for idx in 0..200 {
            let theta = (idx as f64) * TAU / 200.0;
            let lx = 50.0 * theta.cos();
            let ly = 1e-12 * theta.sin();
            arc_x.push(lx * ca - ly * sa);
            arc_y.push(lx * sa + ly * ca);
        }
        let arc_layout = Layout2d::from_xy(arc_x, arc_y).unwrap();
        let got_arc_diam = euclidean_diameter_2d(&arc_layout);
        let mut exp_arc_sq = 0.0_f64;
        for i in 0..200 {
            for j in (i + 1)..200 {
                let dx = arc_layout.x()[i] - arc_layout.x()[j];
                let dy = arc_layout.y()[i] - arc_layout.y()[j];
                exp_arc_sq = exp_arc_sq.max(dx * dx + dy * dy);
            }
        }
        let exp_arc_diam = exp_arc_sq.sqrt();
        assert!((got_arc_diam - exp_arc_diam).abs() <= 1e-11 * exp_arc_diam);
        let got_arc_nr = node_resolution(&arc_layout).unwrap();
        let exp_arc_nr = brute_node_resolution(&arc_layout);
        assert!((got_arc_nr - exp_arc_nr).abs() <= 1e-9 * exp_arc_nr.max(1.0));

        // Direct checks on exact geometric predicates (collinear, parallel, huge/tiny dynamic range).
        assert_eq!(
            orient2d([0.0, 0.0], [7.0, 3.0], [14.0, 6.0]),
            Ordering::Equal
        );
        assert_eq!(
            orient2d([0.0, 0.0], [1e-305, 1e-305], [2e-305, 2e-305]),
            Ordering::Equal
        );
        assert_eq!(
            cross_abs_cmp([0.0, 0.0], [10.0, 0.0], [5.0, 2.0], [8.0, -2.0]),
            Ordering::Equal
        );
        assert_eq!(
            cross_abs_cmp(
                [0.0, 0.0],
                [1e150, 1e150],
                [2e150, 2e150 + 1e135],
                [3e150, 3e150]
            ),
            Ordering::Greater
        );

        // n = 80 at 1e200 must never panic or misbehave.
        let mut rng_huge = DeterministicRng::from_seed_u64(9001);
        let huge_x: Vec<f64> = (0..80)
            .map(|_| (rng_huge.next_f64() - 0.5) * 2e200)
            .collect();
        let huge_y: Vec<f64> = (0..80)
            .map(|_| (rng_huge.next_f64() - 0.5) * 2e200)
            .collect();
        let huge_layout = Layout2d::from_xy(huge_x, huge_y).unwrap();
        let huge_diam = euclidean_diameter_2d(&huge_layout);
        assert!(huge_diam.is_finite() && huge_diam > 0.0);
        assert!(matches!(
            node_resolution(&huge_layout),
            Ok(_) | Err(CoreError::NonFiniteValue { .. })
        ));
    }

    #[test]
    fn test_regression_rng_tiny_witness_sector_pruning_soundness() {
        // Reproduces Bug 2a: near_cocircular (N = 7) in [0, 2] x [-1e-14, 1 + 1e-10].
        let l_cocirc = Layout2d::from_xy(
            vec![0.0, 2.0, 1.0, 1.0 + 1e-14, 0.0 + 1e-14, 2.0 - 1e-14, 1.0],
            vec![0.0, 0.0, 1.0 + 1e-10, 1.0, -1e-14, 1e-14, 0.0],
        )
        .unwrap();
        let got_cocirc = relative_neighborhood_edges(&l_cocirc).unwrap();
        let exp_cocirc = brute_rng_edges(&l_cocirc);
        assert_eq!(got_cocirc, exp_cocirc);

        let g_cocirc =
            CsrGraph::<f64>::from_edges(7, &[(0, 1), (1, 2), (2, 0), (3, 6)], None, false, false)
                .unwrap();
        assert_eq!(
            relative_neighborhood_preservation(&g_cocirc, &l_cocirc)
                .unwrap()
                .to_bits(),
            mean_jaccard_graph_vs_edges(&g_cocirc, &exp_cocirc).to_bits()
        );

        // Reproduces Bug 2b: 1e-17 tight cluster (17 points) around (0, 0) + distant point (1.0, 0.0).
        let mut rng = DeterministicRng::from_seed_u64(31415);
        let mut cx = Vec::with_capacity(18);
        let mut cy = Vec::with_capacity(18);
        for _ in 0..17 {
            cx.push((rng.next_f64() - 0.5) * 1e-17);
            cy.push((rng.next_f64() - 0.5) * 1e-17);
        }
        cx.push(1.0);
        cy.push(0.0);
        let l_cluster18 = Layout2d::from_xy(cx, cy).unwrap();
        let got_cluster18 = relative_neighborhood_edges(&l_cluster18).unwrap();
        let exp_cluster18 = brute_rng_edges(&l_cluster18);
        assert_eq!(got_cluster18, exp_cluster18);

        // Also test > 128 points in a 1e-17 tight cluster + distant point so k = 128 witnesses
        // stay entirely inside the tight cluster.
        let mut cx2 = Vec::with_capacity(141);
        let mut cy2 = Vec::with_capacity(141);
        for _ in 0..140 {
            cx2.push((rng.next_f64() - 0.5) * 1e-17);
            cy2.push((rng.next_f64() - 0.5) * 1e-17);
        }
        cx2.push(1.0);
        cy2.push(0.0);
        let l_cluster141 = Layout2d::from_xy(cx2, cy2).unwrap();
        assert_eq!(
            relative_neighborhood_edges(&l_cluster141).unwrap(),
            brute_rng_edges(&l_cluster141)
        );
    }

    #[test]
    fn test_regression_gg_large_offset_midpoint_cancellation() {
        // Reproduces Bug 3: 1-ULP-spaced cluster in [1e10, 1e10 + 1e-5]^2 (n = 40 and n = 80).
        for &n in &[40usize, 80usize] {
            let mut rng = DeterministicRng::from_seed_u64(20260 + n as u64);
            let xs: Vec<f64> = (0..n).map(|_| 1e10 + rng.next_f64() * 1e-5).collect();
            let ys: Vec<f64> = (0..n).map(|_| 1e10 + rng.next_f64() * 1e-5).collect();
            let layout = Layout2d::from_xy(xs, ys).unwrap();

            let got_gg = gabriel_graph_edges(&layout).unwrap();
            let exp_gg = brute_gg_edges(&layout);
            assert_eq!(got_gg, exp_gg, "GG edges mismatch at 1e10 offset for n={n}");

            let got_rng = relative_neighborhood_edges(&layout).unwrap();
            let exp_rng = brute_rng_edges(&layout);
            assert_eq!(
                got_rng, exp_rng,
                "RNG edges mismatch at 1e10 offset for n={n}"
            );

            let edges: Vec<(u32, u32)> = (0..(n as u32 - 1)).map(|i| (i, i + 1)).collect();
            let g = CsrGraph::<f64>::from_edges(n, &edges, None, false, false).unwrap();
            assert_eq!(
                gabriel_graph_preservation(&g, &layout).unwrap().to_bits(),
                mean_jaccard_graph_vs_edges(&g, &exp_gg).to_bits()
            );
            assert_eq!(
                relative_neighborhood_preservation(&g, &layout)
                    .unwrap()
                    .to_bits(),
                mean_jaccard_graph_vs_edges(&g, &exp_rng).to_bits()
            );
        }
    }
}
