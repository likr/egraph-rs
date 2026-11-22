//! BFS shortest-path utilities and closed-form / pivot-sampled Scale-Normalized Stress (`SNS`).
//!
//! Unscaled normalized stress `NS(X, D)` varies quadratically when coordinates are uniformly
//! rescaled, allowing tiny random layouts to outrank structured drawings. `SNS` eliminates
//! scale artifacts analytically via the closed-form optimal multiplier
//! $\alpha^* = \frac{\sum w_{ij} d_{ij} \|x_i - x_j\|}{\sum w_{ij} \|x_i - x_j\|^2}$
//! accumulated in `f64`.

use egraph_core::{
    CoreError, CoreResult, CsrGraph, DeterministicRng, Layout2d, Scalar, VisitStamp,
};

use crate::validate_graph_layout;

/// Reusable scratch buffers for allocation-free repeated single-source BFS over [`CsrGraph`].
#[derive(Debug, Clone)]
pub struct BfsScratch {
    visited: VisitStamp,
    queue: Vec<u32>,
    distances: Vec<f64>,
}

impl BfsScratch {
    /// Allocates reusable scratch state sized for `num_nodes` vertices.
    pub fn new(num_nodes: usize) -> Self {
        Self {
            visited: VisitStamp::new(num_nodes),
            queue: Vec::with_capacity(num_nodes),
            distances: vec![f64::INFINITY; num_nodes],
        }
    }

    /// Resizes scratch buffers if `num_nodes` changed and resets distances to `f64::INFINITY`.
    pub fn ensure_capacity(&mut self, num_nodes: usize) {
        if self.distances.len() != num_nodes {
            self.visited.resize_and_clear(num_nodes);
            self.distances.resize(num_nodes, f64::INFINITY);
            self.distances.fill(f64::INFINITY);
            self.queue.clear();
            if self.queue.capacity() < num_nodes {
                self.queue.reserve(num_nodes - self.queue.capacity());
            }
        } else {
            self.visited.clear();
            for &u in &self.queue {
                self.distances[u as usize] = f64::INFINITY;
            }
            self.queue.clear();
        }
    }

    /// Returns the distance slice from the most recent [`bfs_single_source`] call.
    ///
    /// Unreachable vertices retain `f64::INFINITY`.
    #[inline]
    pub fn distances(&self) -> &[f64] {
        &self.distances
    }
}

/// Detailed result from evaluating closed-form Scale-Normalized Stress (`SNS`).
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct StressEvaluation {
    /// Scale-normalized stress divided by $\sum w_{ij} d_{ij}^2$, lying in `[0.0, 1.0]`
    /// (equals mean relative squared error over evaluated pairs when $w_{ij} = d_{ij}^{-2}$).
    pub normalized_stress: f64,
    /// Unnormalized optimal-scale stress sum $\sum w_{ij} (\alpha^* \|x_i - x_j\| - d_{ij})^2$.
    pub raw_stress: f64,
    /// Closed-form optimal uniform coordinate scale factor $\alpha^*$.
    pub alpha_star: f64,
    /// Number of valid finite positive-distance pairs included in the evaluation.
    pub pair_count: usize,
    /// Sum of $\sum w_{ij} d_{ij}^2$ over evaluated pairs (equals `pair_count` for $w_{ij} = d_{ij}^{-2}$).
    pub weight_distance_sq_sum: f64,
}

/// Runs single-source unweighted BFS from `source` on `graph`, scaling each hop by
/// `unit_edge_length` and reusing `scratch` without heap thrashing.
///
/// Unreachable vertices hold `f64::INFINITY`. Returns a slice referencing `scratch.distances()`.
pub fn bfs_single_source<'a, W>(
    graph: &CsrGraph<W>,
    source: usize,
    unit_edge_length: f64,
    scratch: &'a mut BfsScratch,
) -> CoreResult<&'a [f64]> {
    let n = graph.num_nodes();
    if source >= n {
        return Err(CoreError::index_out_of_bounds(
            source,
            n,
            "bfs_single_source",
        ));
    }
    if !unit_edge_length.is_finite() {
        return Err(CoreError::non_finite("bfs_single_source::unit_edge_length"));
    }
    if unit_edge_length <= 0.0 {
        return Err(CoreError::invalid_input(
            "unit_edge_length must be strictly positive",
        ));
    }

    scratch.ensure_capacity(n);
    let src_u32 = source as u32;
    scratch.visited.mark(source);
    scratch.distances[source] = 0.0;
    scratch.queue.push(src_u32);

    let mut head = 0usize;
    while head < scratch.queue.len() {
        let u = scratch.queue[head] as usize;
        head += 1;
        let next_dist = scratch.distances[u] + unit_edge_length;
        for &v in graph.neighbors(u) {
            let vi = v as usize;
            if scratch.visited.visit(vi) {
                scratch.distances[vi] = next_dist;
                scratch.queue.push(v);
            }
        }
    }

    Ok(scratch.distances())
}

/// Computes an all-pairs shortest-path (APSP) distance matrix via repeated single-source BFS
/// into a reusable flat row-major `out` buffer of length `N * N`.
pub fn all_pairs_bfs_into<W>(
    graph: &CsrGraph<W>,
    unit_edge_length: f64,
    out: &mut Vec<f64>,
    scratch: &mut BfsScratch,
) -> CoreResult<()> {
    if !unit_edge_length.is_finite() {
        return Err(CoreError::non_finite(
            "all_pairs_bfs_into::unit_edge_length",
        ));
    }
    if unit_edge_length <= 0.0 {
        return Err(CoreError::invalid_input(
            "unit_edge_length must be strictly positive",
        ));
    }
    let n = graph.num_nodes();
    let total = n
        .checked_mul(n)
        .ok_or_else(|| CoreError::invalid_input("APSP matrix size overflow"))?;
    out.resize(total, f64::INFINITY);

    for src in 0..n {
        let dists = bfs_single_source(graph, src, unit_edge_length, scratch)?;
        let row_start = src * n;
        out[row_start..row_start + n].copy_from_slice(dists);
    }
    Ok(())
}

/// Computes an all-pairs shortest-path (APSP) flat row-major `N * N` distance matrix using BFS.
pub fn all_pairs_bfs<W>(graph: &CsrGraph<W>, unit_edge_length: f64) -> CoreResult<Vec<f64>> {
    let n = graph.num_nodes();
    let mut out = Vec::new();
    let mut scratch = BfsScratch::new(n);
    all_pairs_bfs_into(graph, unit_edge_length, &mut out, &mut scratch)?;
    Ok(out)
}

#[derive(Default)]
struct SnsMoments {
    sum_w_r2: f64,
    sum_w_d_r: f64,
    sum_w_d2: f64,
    pair_count: usize,
}

impl SnsMoments {
    #[inline]
    fn push(&mut self, r: f64, d: f64, w: f64) {
        self.sum_w_r2 += w * r * r;
        self.sum_w_d_r += w * d * r;
        self.sum_w_d2 += w * d * d;
        self.pair_count += 1;
    }
}

#[inline]
fn finalize_moments(m: &SnsMoments) -> Option<StressEvaluation> {
    if m.pair_count == 0 || m.sum_w_d2 <= 0.0 {
        return Some(StressEvaluation {
            normalized_stress: 0.0,
            raw_stress: 0.0,
            alpha_star: 0.0,
            pair_count: m.pair_count,
            weight_distance_sq_sum: m.sum_w_d2,
        });
    }
    if m.sum_w_r2 <= 0.0 {
        return Some(StressEvaluation {
            normalized_stress: 1.0,
            raw_stress: m.sum_w_d2,
            alpha_star: 0.0,
            pair_count: m.pair_count,
            weight_distance_sq_sum: m.sum_w_d2,
        });
    }
    let alpha_star = m.sum_w_d_r / m.sum_w_r2;
    let one_pass_norm = (1.0 - (m.sum_w_d_r / m.sum_w_r2) * (m.sum_w_d_r / m.sum_w_d2)).max(0.0);
    // Use two-pass direct residual summation when stress is small so cancellation near zero
    // never drops precision on near-isometric embeddings.
    if one_pass_norm < 1e-4 {
        return None;
    }
    Some(StressEvaluation {
        normalized_stress: one_pass_norm.clamp(0.0, 1.0),
        raw_stress: (one_pass_norm * m.sum_w_d2).max(0.0),
        alpha_star,
        pair_count: m.pair_count,
        weight_distance_sq_sum: m.sum_w_d2,
    })
}

/// Computes closed-form Scale-Normalized Stress (`SNS`) in `[0.0, 1.0]` over `graph` and `layout`.
///
/// Uses unweighted BFS hop distances (`1.0` per edge) with stress weights $w_{ij} = d_{ij}^{-2}$,
/// $O(N)$ scratch memory via [`BfsScratch`], and `f64` accumulation.
pub fn scale_normalized_stress<W, S: Scalar>(
    graph: &CsrGraph<W>,
    layout: &Layout2d<S>,
) -> CoreResult<f64> {
    Ok(scale_normalized_stress_detail(graph, layout)?.normalized_stress)
}

/// Computes full [`StressEvaluation`] details for closed-form `SNS` over `graph` and `layout`.
pub fn scale_normalized_stress_detail<W, S: Scalar>(
    graph: &CsrGraph<W>,
    layout: &Layout2d<S>,
) -> CoreResult<StressEvaluation> {
    validate_graph_layout(graph, layout, "scale_normalized_stress")?;
    let n = layout.len();
    if n <= 1 {
        return Ok(StressEvaluation::default());
    }

    let xs = layout.x();
    let ys = layout.y();
    let mut scratch = BfsScratch::new(n);
    let mut moments = SnsMoments::default();
    let directed = graph.is_directed();

    for i in 0..n {
        let xi = xs[i].to_f64();
        let yi = ys[i].to_f64();
        let dists = bfs_single_source(graph, i, 1.0, &mut scratch)?;
        let j_start = if directed { 0 } else { i + 1 };
        for (j, &d) in dists.iter().enumerate().skip(j_start) {
            if j == i || !d.is_finite() || d <= 0.0 {
                continue;
            }
            let r = (xi - xs[j].to_f64()).hypot(yi - ys[j].to_f64());
            let w = 1.0 / (d * d);
            moments.push(r, d, w);
        }
    }

    if let Some(eval) = finalize_moments(&moments) {
        return Ok(eval);
    }

    let alpha_star = moments.sum_w_d_r / moments.sum_w_r2;
    let mut raw_stress = 0.0_f64;
    for i in 0..n {
        let xi = xs[i].to_f64();
        let yi = ys[i].to_f64();
        let dists = bfs_single_source(graph, i, 1.0, &mut scratch)?;
        let j_start = if directed { 0 } else { i + 1 };
        for (j, &d) in dists.iter().enumerate().skip(j_start) {
            if j == i || !d.is_finite() || d <= 0.0 {
                continue;
            }
            let r = (xi - xs[j].to_f64()).hypot(yi - ys[j].to_f64());
            let err = (alpha_star * r - d) / d;
            raw_stress += err * err;
        }
    }

    Ok(StressEvaluation {
        normalized_stress: (raw_stress / moments.sum_w_d2).clamp(0.0, 1.0),
        raw_stress,
        alpha_star,
        pair_count: moments.pair_count,
        weight_distance_sq_sum: moments.sum_w_d2,
    })
}

/// Computes closed-form Scale-Normalized Stress (`SNS`) from a precomputed flat row-major `N * N`
/// `distances` matrix and optional `weights` matrix (`w_ij = 1 / d_ij^2` when `None`).
pub fn scale_normalized_stress_from_distances<S: Scalar>(
    layout: &Layout2d<S>,
    distances: &[f64],
    weights: Option<&[f64]>,
) -> CoreResult<f64> {
    Ok(
        scale_normalized_stress_from_distances_detail(layout, distances, weights)?
            .normalized_stress,
    )
}

/// Computes full [`StressEvaluation`] details from precomputed `N * N` `distances` and optional `weights`.
pub fn scale_normalized_stress_from_distances_detail<S: Scalar>(
    layout: &Layout2d<S>,
    distances: &[f64],
    weights: Option<&[f64]>,
) -> CoreResult<StressEvaluation> {
    layout.validate_finite()?;
    let n = layout.len();
    let expected = n
        .checked_mul(n)
        .ok_or_else(|| CoreError::invalid_input("distance matrix size overflow"))?;
    if distances.len() != expected {
        return Err(CoreError::dimension_mismatch(
            expected,
            distances.len(),
            "scale_normalized_stress_from_distances::distances",
        ));
    }
    if let Some(ws) = weights
        && ws.len() != expected
    {
        return Err(CoreError::dimension_mismatch(
            expected,
            ws.len(),
            "scale_normalized_stress_from_distances::weights",
        ));
    }
    if n <= 1 {
        return Ok(StressEvaluation::default());
    }

    let xs = layout.x();
    let ys = layout.y();
    let mut moments = SnsMoments::default();

    for i in 0..n {
        let xi = xs[i].to_f64();
        let yi = ys[i].to_f64();
        let row = i * n;
        for j in (i + 1)..n {
            let d = distances[row + j];
            if !d.is_finite() || d <= 0.0 {
                continue;
            }
            let w = match weights {
                Some(ws) => {
                    let val = ws[row + j];
                    if !val.is_finite() {
                        return Err(CoreError::non_finite(
                            "scale_normalized_stress_from_distances::weights",
                        ));
                    }
                    if val < 0.0 {
                        return Err(CoreError::invalid_input(
                            "stress weights must be non-negative",
                        ));
                    }
                    if val == 0.0 {
                        continue;
                    }
                    val
                }
                None => 1.0 / (d * d),
            };
            let r = (xi - xs[j].to_f64()).hypot(yi - ys[j].to_f64());
            moments.push(r, d, w);
        }
    }

    if moments.pair_count == 0 || moments.sum_w_d2 <= 0.0 {
        return Ok(StressEvaluation {
            normalized_stress: 0.0,
            raw_stress: 0.0,
            alpha_star: 0.0,
            pair_count: moments.pair_count,
            weight_distance_sq_sum: moments.sum_w_d2,
        });
    }
    if moments.sum_w_r2 <= 0.0 {
        return Ok(StressEvaluation {
            normalized_stress: 1.0,
            raw_stress: moments.sum_w_d2,
            alpha_star: 0.0,
            pair_count: moments.pair_count,
            weight_distance_sq_sum: moments.sum_w_d2,
        });
    }

    let alpha_star = moments.sum_w_d_r / moments.sum_w_r2;
    let mut raw_stress = 0.0_f64;
    for i in 0..n {
        let xi = xs[i].to_f64();
        let yi = ys[i].to_f64();
        let row = i * n;
        for j in (i + 1)..n {
            let d = distances[row + j];
            if !d.is_finite() || d <= 0.0 {
                continue;
            }
            let w = match weights {
                Some(ws) => {
                    let val = ws[row + j];
                    if val == 0.0 {
                        continue;
                    }
                    val
                }
                None => 1.0 / (d * d),
            };
            let r = (xi - xs[j].to_f64()).hypot(yi - ys[j].to_f64());
            let diff = alpha_star * r - d;
            raw_stress += w * diff * diff;
        }
    }

    Ok(StressEvaluation {
        normalized_stress: (raw_stress / moments.sum_w_d2).clamp(0.0, 1.0),
        raw_stress,
        alpha_star,
        pair_count: moments.pair_count,
        weight_distance_sq_sum: moments.sum_w_d2,
    })
}

/// Computes closed-form Scale-Normalized Stress (`SNS`) from sparse `(i, j, d_ij, w_ij)` pairs.
pub fn scale_normalized_stress_from_pairs<S: Scalar>(
    layout: &Layout2d<S>,
    pairs: &[(u32, u32, f64, f64)],
) -> CoreResult<StressEvaluation> {
    layout.validate_finite()?;
    let n = layout.len();
    let xs = layout.x();
    let ys = layout.y();
    let mut moments = SnsMoments::default();

    for &(u, v, d, w) in pairs {
        let ui = u as usize;
        let vi = v as usize;
        if ui >= n || vi >= n {
            return Err(CoreError::index_out_of_bounds(
                ui.max(vi),
                n,
                "scale_normalized_stress_from_pairs",
            ));
        }
        if !w.is_finite() {
            return Err(CoreError::non_finite(
                "scale_normalized_stress_from_pairs::weight",
            ));
        }
        if w < 0.0 {
            return Err(CoreError::invalid_input(
                "stress weights must be non-negative",
            ));
        }
        if ui == vi || !d.is_finite() || d <= 0.0 || w == 0.0 {
            continue;
        }
        let r = (xs[ui].to_f64() - xs[vi].to_f64()).hypot(ys[ui].to_f64() - ys[vi].to_f64());
        moments.push(r, d, w);
    }

    if moments.pair_count == 0 || moments.sum_w_d2 <= 0.0 {
        return Ok(StressEvaluation {
            normalized_stress: 0.0,
            raw_stress: 0.0,
            alpha_star: 0.0,
            pair_count: moments.pair_count,
            weight_distance_sq_sum: moments.sum_w_d2,
        });
    }
    if moments.sum_w_r2 <= 0.0 {
        return Ok(StressEvaluation {
            normalized_stress: 1.0,
            raw_stress: moments.sum_w_d2,
            alpha_star: 0.0,
            pair_count: moments.pair_count,
            weight_distance_sq_sum: moments.sum_w_d2,
        });
    }

    let alpha_star = moments.sum_w_d_r / moments.sum_w_r2;
    let mut raw_stress = 0.0_f64;
    for &(u, v, d, w) in pairs {
        let ui = u as usize;
        let vi = v as usize;
        if ui == vi || !d.is_finite() || d <= 0.0 || w == 0.0 {
            continue;
        }
        let r = (xs[ui].to_f64() - xs[vi].to_f64()).hypot(ys[ui].to_f64() - ys[vi].to_f64());
        let diff = alpha_star * r - d;
        raw_stress += w * diff * diff;
    }

    Ok(StressEvaluation {
        normalized_stress: (raw_stress / moments.sum_w_d2).clamp(0.0, 1.0),
        raw_stress,
        alpha_star,
        pair_count: moments.pair_count,
        weight_distance_sq_sum: moments.sum_w_d2,
    })
}

/// Computes $O(kN)$ pivot-sampled Scale-Normalized Stress (`SNS`) from `num_pivots`
/// deterministic [`DeterministicRng`] source vertices.
pub fn sampled_scale_normalized_stress<W, S: Scalar>(
    graph: &CsrGraph<W>,
    layout: &Layout2d<S>,
    num_pivots: usize,
    seed: u64,
) -> CoreResult<f64> {
    Ok(sampled_scale_normalized_stress_detail(graph, layout, num_pivots, seed)?.normalized_stress)
}

/// Computes full [`StressEvaluation`] details for $O(kN)$ pivot-sampled `SNS`.
pub fn sampled_scale_normalized_stress_detail<W, S: Scalar>(
    graph: &CsrGraph<W>,
    layout: &Layout2d<S>,
    num_pivots: usize,
    seed: u64,
) -> CoreResult<StressEvaluation> {
    if num_pivots == 0 {
        return Err(CoreError::invalid_input("num_pivots must be >= 1"));
    }
    validate_graph_layout(graph, layout, "sampled_scale_normalized_stress")?;
    let n = layout.len();
    if n <= 1 {
        return Ok(StressEvaluation::default());
    }

    let k = num_pivots.min(n);
    let mut pivots: Vec<usize> = (0..n).collect();
    let mut is_pivot = vec![false; n];
    if k == n {
        is_pivot.fill(true);
    } else {
        let mut rng = DeterministicRng::from_seed_u64(seed);
        for i in 0..k {
            let pick = i + rng.gen_range_usize(n - i);
            pivots.swap(i, pick);
            is_pivot[pivots[i]] = true;
        }
        pivots.truncate(k);
    }

    let xs = layout.x();
    let ys = layout.y();
    let directed = graph.is_directed();
    let mut scratch = BfsScratch::new(n);
    let mut moments = SnsMoments::default();

    for &p in &pivots {
        let xp = xs[p].to_f64();
        let yp = ys[p].to_f64();
        let dists = bfs_single_source(graph, p, 1.0, &mut scratch)?;
        for (j, &d) in dists.iter().enumerate() {
            if j == p || !d.is_finite() || d <= 0.0 {
                continue;
            }
            if !directed && is_pivot[j] && j < p {
                continue;
            }
            let r = (xp - xs[j].to_f64()).hypot(yp - ys[j].to_f64());
            let w = 1.0 / (d * d);
            moments.push(r, d, w);
        }
    }

    if let Some(eval) = finalize_moments(&moments) {
        return Ok(eval);
    }

    let alpha_star = moments.sum_w_d_r / moments.sum_w_r2;
    let mut raw_stress = 0.0_f64;
    for &p in &pivots {
        let xp = xs[p].to_f64();
        let yp = ys[p].to_f64();
        let dists = bfs_single_source(graph, p, 1.0, &mut scratch)?;
        for (j, &d) in dists.iter().enumerate() {
            if j == p || !d.is_finite() || d <= 0.0 {
                continue;
            }
            if !directed && is_pivot[j] && j < p {
                continue;
            }
            let r = (xp - xs[j].to_f64()).hypot(yp - ys[j].to_f64());
            let err = (alpha_star * r - d) / d;
            raw_stress += err * err;
        }
    }

    Ok(StressEvaluation {
        normalized_stress: (raw_stress / moments.sum_w_d2).clamp(0.0, 1.0),
        raw_stress,
        alpha_star,
        pair_count: moments.pair_count,
        weight_distance_sq_sum: moments.sum_w_d2,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scale_layout(layout: &Layout2d<f64>, factor: f64) -> Layout2d<f64> {
        let x: Vec<f64> = layout.x().iter().map(|&v| v * factor).collect();
        let y: Vec<f64> = layout.y().iter().map(|&v| v * factor).collect();
        Layout2d::from_xy(x, y).unwrap()
    }

    fn random_layout(n: usize, seed: u64, scale: f64) -> Layout2d<f64> {
        let mut rng = DeterministicRng::from_seed_u64(seed);
        let x: Vec<f64> = (0..n).map(|_| (rng.next_f64() - 0.5) * scale).collect();
        let y: Vec<f64> = (0..n).map(|_| (rng.next_f64() - 0.5) * scale).collect();
        Layout2d::from_xy(x, y).unwrap()
    }

    fn make_path(n: usize) -> (CsrGraph<f64>, Layout2d<f64>) {
        let edges: Vec<(u32, u32)> = (0..n - 1).map(|i| (i as u32, (i + 1) as u32)).collect();
        let g = CsrGraph::from_edges(n, &edges, None, false, false).unwrap();
        let x: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let y = vec![0.0; n];
        (g, Layout2d::from_xy(x, y).unwrap())
    }

    fn make_cycle(n: usize) -> (CsrGraph<f64>, Layout2d<f64>) {
        let edges: Vec<(u32, u32)> = (0..n).map(|i| (i as u32, ((i + 1) % n) as u32)).collect();
        let g = CsrGraph::from_edges(n, &edges, None, false, false).unwrap();
        let x: Vec<f64> = (0..n)
            .map(|i| (i as f64 * std::f64::consts::TAU / n as f64).cos())
            .collect();
        let y: Vec<f64> = (0..n)
            .map(|i| (i as f64 * std::f64::consts::TAU / n as f64).sin())
            .collect();
        (g, Layout2d::from_xy(x, y).unwrap())
    }

    fn make_grid(rows: usize, cols: usize) -> (CsrGraph<f64>, Layout2d<f64>) {
        let n = rows * cols;
        let mut edges = Vec::new();
        let mut x = Vec::with_capacity(n);
        let mut y = Vec::with_capacity(n);
        for r in 0..rows {
            for c in 0..cols {
                let id = (r * cols + c) as u32;
                x.push(c as f64);
                y.push(r as f64);
                if c + 1 < cols {
                    edges.push((id, id + 1));
                }
                if r + 1 < rows {
                    edges.push((id, id + cols as u32));
                }
            }
        }
        let g = CsrGraph::from_edges(n, &edges, None, false, false).unwrap();
        (g, Layout2d::from_xy(x, y).unwrap())
    }

    fn make_barbell(clique_size: usize, bridge_len: usize) -> (CsrGraph<f64>, Layout2d<f64>) {
        let n = 2 * clique_size + bridge_len;
        let mut edges = Vec::new();
        for i in 0..clique_size {
            for j in (i + 1)..clique_size {
                edges.push((i as u32, j as u32));
            }
        }
        let offset = clique_size + bridge_len;
        for i in 0..clique_size {
            for j in (i + 1)..clique_size {
                edges.push(((offset + i) as u32, (offset + j) as u32));
            }
        }
        let mut prev = (clique_size - 1) as u32;
        for b in 0..bridge_len {
            let cur = (clique_size + b) as u32;
            edges.push((prev, cur));
            prev = cur;
        }
        edges.push((prev, offset as u32));

        let g = CsrGraph::from_edges(n, &edges, None, false, false).unwrap();
        let mut x = Vec::with_capacity(n);
        let mut y = Vec::with_capacity(n);
        let radius = 0.6;
        for i in 0..clique_size {
            let ang = i as f64 * std::f64::consts::TAU / clique_size as f64;
            x.push(-3.0 + radius * ang.cos());
            y.push(radius * ang.sin());
        }
        for b in 0..bridge_len {
            let t = (b + 1) as f64 / (bridge_len + 1) as f64;
            x.push(-2.0 + 4.0 * t);
            y.push(0.0);
        }
        for i in 0..clique_size {
            let ang = i as f64 * std::f64::consts::TAU / clique_size as f64;
            x.push(3.0 + radius * ang.cos());
            y.push(radius * ang.sin());
        }
        (g, Layout2d::from_xy(x, y).unwrap())
    }

    fn make_karate() -> (CsrGraph<f64>, Layout2d<f64>) {
        const KARATE_EDGES: &[(u32, u32)] = &[
            (1, 0),
            (1, 2),
            (1, 3),
            (1, 7),
            (1, 13),
            (1, 17),
            (1, 19),
            (1, 21),
            (1, 30),
            (0, 2),
            (0, 3),
            (0, 4),
            (0, 5),
            (0, 6),
            (0, 7),
            (0, 8),
            (0, 10),
            (0, 11),
            (0, 12),
            (0, 13),
            (0, 17),
            (0, 19),
            (0, 21),
            (0, 31),
            (2, 3),
            (2, 7),
            (2, 8),
            (2, 9),
            (2, 13),
            (2, 27),
            (2, 28),
            (2, 32),
            (3, 7),
            (3, 12),
            (3, 13),
            (4, 6),
            (4, 10),
            (5, 6),
            (5, 10),
            (5, 16),
            (6, 16),
            (8, 30),
            (8, 32),
            (8, 33),
            (9, 33),
            (13, 33),
            (19, 33),
            (25, 23),
            (25, 24),
            (25, 31),
            (23, 27),
            (23, 29),
            (23, 32),
            (23, 33),
            (24, 27),
            (24, 31),
            (27, 33),
            (28, 31),
            (28, 33),
            (29, 26),
            (29, 32),
            (29, 33),
            (26, 33),
            (30, 32),
            (30, 33),
            (31, 32),
            (31, 33),
            (32, 14),
            (32, 15),
            (32, 18),
            (32, 20),
            (32, 22),
            (32, 33),
            (14, 33),
            (15, 33),
            (18, 33),
            (20, 33),
            (22, 33),
        ];
        let n = 34usize;
        let g = CsrGraph::from_edges(n, KARATE_EDGES, None, false, true).unwrap();
        let apsp = all_pairs_bfs(&g, 1.0).unwrap();
        // Deterministic stress-SGD embedding from circular initialization.
        let mut xs: Vec<f64> = (0..n)
            .map(|i| (i as f64 * std::f64::consts::TAU / n as f64).cos())
            .collect();
        let mut ys: Vec<f64> = (0..n)
            .map(|i| (i as f64 * std::f64::consts::TAU / n as f64).sin())
            .collect();
        let mut pairs = Vec::new();
        for i in 0..n {
            for j in (i + 1)..n {
                pairs.push((i, j, apsp[i * n + j]));
            }
        }
        let mut rng = DeterministicRng::from_seed_u64(7);
        for iter in 0..60 {
            rng.shuffle(&mut pairs);
            let eta = 0.6 * (0.05_f64).powf(iter as f64 / 59.0);
            for &(i, j, d) in &pairs {
                let w = 1.0 / (d * d);
                let mu = (eta * w).min(1.0);
                let dx = xs[i] - xs[j];
                let dy = ys[i] - ys[j];
                let r = dx.hypot(dy).max(1e-9);
                let step = 0.5 * mu * (r - d) / r;
                xs[i] -= step * dx;
                ys[i] -= step * dy;
                xs[j] += step * dx;
                ys[j] += step * dy;
            }
        }
        (g, Layout2d::from_xy(xs, ys).unwrap())
    }

    #[test]
    fn test_bfs_scratch_reuse_and_apsp() {
        let g =
            CsrGraph::<f64>::from_edges(6, &[(0, 1), (1, 2), (2, 3), (4, 5)], None, false, false)
                .unwrap();
        let mut scratch = BfsScratch::new(6);
        let d0 = bfs_single_source(&g, 0, 2.0, &mut scratch)
            .unwrap()
            .to_vec();
        assert_eq!(d0[0], 0.0);
        assert_eq!(d0[1], 2.0);
        assert_eq!(d0[2], 4.0);
        assert_eq!(d0[3], 6.0);
        assert!(d0[4].is_infinite());
        assert!(d0[5].is_infinite());

        let d4 = bfs_single_source(&g, 4, 1.0, &mut scratch).unwrap();
        assert!(d4[0].is_infinite());
        assert_eq!(d4[4], 0.0);
        assert_eq!(d4[5], 1.0);

        let apsp = all_pairs_bfs(&g, 1.0).unwrap();
        assert_eq!(apsp.len(), 36);
        assert_eq!(apsp[3], 3.0);
        assert!(apsp[4].is_infinite());

        assert!(matches!(
            bfs_single_source(&g, 6, 1.0, &mut scratch),
            Err(CoreError::IndexOutOfBounds { .. })
        ));
        assert!(matches!(
            bfs_single_source(&g, 0, 0.0, &mut scratch),
            Err(CoreError::InvalidInput(_))
        ));
        assert!(matches!(
            bfs_single_source(&g, 0, f64::NAN, &mut scratch),
            Err(CoreError::NonFiniteValue { .. })
        ));
    }

    #[test]
    fn test_scale_invariance_and_zero_inversion_against_random() {
        let fixtures: Vec<(&str, CsrGraph<f64>, Layout2d<f64>)> = vec![
            {
                let (g, l) = make_path(16);
                ("path", g, l)
            },
            {
                let (g, l) = make_cycle(20);
                ("cycle", g, l)
            },
            {
                let (g, l) = make_grid(5, 5);
                ("grid", g, l)
            },
            {
                let (g, l) = make_barbell(5, 3);
                ("barbell", g, l)
            },
            {
                let (g, l) = make_karate();
                ("karate", g, l)
            },
        ];

        let mut total_comparisons = 0usize;
        let mut inversions = 0usize;

        for (name, g, layout) in &fixtures {
            let s1 = scale_normalized_stress(g, layout).unwrap();
            let s001 = scale_normalized_stress(g, &scale_layout(layout, 0.01)).unwrap();
            let s100 = scale_normalized_stress(g, &scale_layout(layout, 100.0)).unwrap();
            assert!(
                (s001 - s1).abs() < 1e-12 && (s100 - s1).abs() < 1e-12,
                "{name}: scale invariance failed: s001={s001}, s1={s1}, s100={s100}"
            );

            // Also verify precomputed APSP variant matches graph-computing SNS identically.
            let apsp = all_pairs_bfs(g, 1.0).unwrap();
            let s_pre = scale_normalized_stress_from_distances(layout, &apsp, None).unwrap();
            assert!((s_pre - s1).abs() < 1e-12, "{name}: precomputed mismatch");

            // Check sampled SNS exact parity when all nodes are pivots, and scale invariance.
            let s_sampled_all =
                sampled_scale_normalized_stress(g, layout, g.num_nodes(), 42).unwrap();
            assert!((s_sampled_all - s1).abs() < 1e-12);
            let s_sampled = sampled_scale_normalized_stress(g, layout, 8, 42).unwrap();
            let s_sampled_100 =
                sampled_scale_normalized_stress(g, &scale_layout(layout, 100.0), 8, 42).unwrap();
            assert!((s_sampled - s_sampled_100).abs() < 1e-12);

            // Test across 20 random layouts at adversarial scales (0.01, 1.0, 100.0).
            for seed in 1..=20u64 {
                for &rand_scale in &[0.01, 1.0, 100.0] {
                    let r_layout = random_layout(g.num_nodes(), seed, rand_scale);
                    let r_sns = scale_normalized_stress(g, &r_layout).unwrap();
                    total_comparisons += 1;
                    if s1 >= r_sns {
                        inversions += 1;
                    }
                    assert!(
                        s1 < r_sns,
                        "{name} seed={seed} scale={rand_scale}: SNS inverted ({s1} >= {r_sns})"
                    );
                }
            }
        }

        assert_eq!(inversions, 0);
        assert_eq!(total_comparisons, 5 * 20 * 3);
    }

    #[test]
    fn test_sns_degeneracies_and_error_cases() {
        // Collapsed layout -> alpha_star = 0, normalized_stress = 1.0, no NaN.
        let (g, _) = make_path(4);
        let collapsed = Layout2d::<f64>::zeros(4);
        let eval = scale_normalized_stress_detail(&g, &collapsed).unwrap();
        assert_eq!(eval.alpha_star, 0.0);
        assert_eq!(eval.normalized_stress, 1.0);
        assert!(eval.raw_stress.is_finite());

        // Edgeless graph -> no positive-distance pairs -> 0.0.
        let edgeless = CsrGraph::<f64>::from_edges(4, &[], None, false, false).unwrap();
        assert_eq!(
            scale_normalized_stress(&edgeless, &Layout2d::<f64>::zeros(4)).unwrap(),
            0.0
        );

        // Precomputed zero / NaN / Inf distances skipped cleanly.
        let dm_zero = vec![0.0; 16];
        assert_eq!(
            scale_normalized_stress_from_distances(&collapsed, &dm_zero, None).unwrap(),
            0.0
        );
        let dm_nan = vec![f64::NAN; 16];
        assert_eq!(
            scale_normalized_stress_from_distances(&collapsed, &dm_nan, None).unwrap(),
            0.0
        );

        // Validation errors.
        let bad = Layout2d::from_xy(vec![0.0, f64::NAN, 1.0, 2.0], vec![0.0; 4]).unwrap();
        assert!(matches!(
            scale_normalized_stress(&g, &bad),
            Err(CoreError::NonFiniteValue { .. })
        ));
        assert!(matches!(
            scale_normalized_stress(&g, &Layout2d::<f64>::zeros(3)),
            Err(CoreError::DimensionMismatch { .. })
        ));
        assert!(matches!(
            sampled_scale_normalized_stress(&g, &collapsed, 0, 1),
            Err(CoreError::InvalidInput(_))
        ));
    }
}
