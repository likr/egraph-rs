//! 1-hop and 2-hop neighborhood preservation metrics backed by `StaticKdTree<2, f64>`.

use std::collections::HashSet;

use egraph_core::{
    CoreError, CoreResult, CsrGraph, DeterministicRng, Layout2d, Scalar, VisitStamp,
};
use egraph_spatial::{KdQueryScratch, StaticKdTree};

use crate::{
    build_kdtree_f64, build_undirected_adjacency, validate_graph_layout, validate_tree_layout,
};

/// Computes 1-hop neighborhood preservation (`NP1`) in `[0.0, 1.0]`.
///
/// Compares each vertex's $d_u$ graph neighbors against its $d_u$ nearest spatial neighbors
/// in `layout` via global directed-pair Jaccard similarity:
/// $\frac{\sum_u |N_G(u) \cap N_D(u)|}{\sum_u |N_G(u) \cup N_D(u)|}$.
/// Returns `0.0` for empty or edgeless graphs.
pub fn neighborhood_preservation_1hop<W, S: Scalar>(
    graph: &CsrGraph<W>,
    layout: &Layout2d<S>,
) -> CoreResult<f64> {
    validate_graph_layout(graph, layout, "neighborhood_preservation_1hop")?;
    let n = layout.len();
    if n <= 1 || graph.num_edges() == 0 {
        return Ok(0.0);
    }
    let tree = build_kdtree_f64(layout)?;
    neighborhood_preservation_1hop_with_tree(graph, layout, &tree)
}

/// Computes [`neighborhood_preservation_1hop`] reusing a prebuilt [`StaticKdTree<2, f64>`].
pub fn neighborhood_preservation_1hop_with_tree<W, S: Scalar>(
    graph: &CsrGraph<W>,
    layout: &Layout2d<S>,
    tree: &StaticKdTree<2, f64>,
) -> CoreResult<f64> {
    validate_graph_layout(graph, layout, "neighborhood_preservation_1hop_with_tree")?;
    validate_tree_layout(tree, layout, "neighborhood_preservation_1hop_with_tree")?;

    let n = layout.len();
    if n <= 1 || graph.num_edges() == 0 {
        return Ok(0.0);
    }

    let directed_adj = if graph.is_directed() {
        Some(build_undirected_adjacency(graph))
    } else {
        None
    };

    let mut in_ng = VisitStamp::new(n);
    let mut knn_buf: Vec<(usize, f64)> = Vec::new();
    let mut cap = 0usize;
    let mut cup = 0usize;

    for i in 0..n {
        let row: &[u32] = match &directed_adj {
            Some(adj) => &adj[i],
            None => graph.neighbors(i),
        };
        in_ng.clear();
        in_ng.mark(i);
        let mut deg = 0usize;
        for &v in row {
            if in_ng.visit(v as usize) {
                deg += 1;
            }
        }
        if deg == 0 {
            continue;
        }
        cup += deg;
        tree.k_nearest_excluding_self_into(i, deg, &mut knn_buf)?;
        for &(j, _) in &knn_buf {
            if in_ng.is_visited(j) {
                cap += 1;
            } else {
                cup += 1;
            }
        }
    }

    if cup == 0 {
        Ok(0.0)
    } else {
        Ok(cap as f64 / cup as f64)
    }
}

/// Computes 2-hop neighborhood preservation (`NP2`) in `[0.0, 1.0]`.
///
/// For each vertex $u$, computes its 2-hop graph neighborhood $N_G(u, 2)$ (vertices at
/// shortest-path distance $1$ or $2$), finds its $|N_G(u, 2)|$ nearest spatial neighbors
/// $N_D(u, 2)$ in `layout`, and averages node-wise Jaccard coefficients
/// $|N_G(u, 2) \cap N_D(u, 2)| / |N_G(u, 2) \cup N_D(u, 2)|$ (defined as `1.0` when
/// $N_G(u, 2) = \emptyset$, and `0.0` when `n == 0`).
pub fn neighborhood_preservation_2hop<W, S: Scalar>(
    graph: &CsrGraph<W>,
    layout: &Layout2d<S>,
) -> CoreResult<f64> {
    validate_graph_layout(graph, layout, "neighborhood_preservation_2hop")?;
    let n = layout.len();
    if n == 0 {
        return Ok(0.0);
    }
    let tree = build_kdtree_f64(layout)?;
    neighborhood_preservation_2hop_with_tree(graph, layout, &tree)
}

/// Computes [`neighborhood_preservation_2hop`] reusing a prebuilt [`StaticKdTree<2, f64>`].
pub fn neighborhood_preservation_2hop_with_tree<W, S: Scalar>(
    graph: &CsrGraph<W>,
    layout: &Layout2d<S>,
    tree: &StaticKdTree<2, f64>,
) -> CoreResult<f64> {
    validate_graph_layout(graph, layout, "neighborhood_preservation_2hop_with_tree")?;
    validate_tree_layout(tree, layout, "neighborhood_preservation_2hop_with_tree")?;

    let n = layout.len();
    if n == 0 {
        return Ok(0.0);
    }

    let directed_adj = if graph.is_directed() {
        Some(build_undirected_adjacency(graph))
    } else {
        None
    };

    let mut in_ng2 = VisitStamp::new(n);
    let mut knn_buf: Vec<(usize, f64)> = Vec::new();
    let mut sum_jaccard = 0.0_f64;

    for i in 0..n {
        in_ng2.clear();
        in_ng2.mark(i);
        let mut k = 0usize;

        let nbrs_1: &[u32] = match &directed_adj {
            Some(adj) => &adj[i],
            None => graph.neighbors(i),
        };
        for &v in nbrs_1 {
            let vi = v as usize;
            if in_ng2.visit(vi) {
                k += 1;
            }
            let nbrs_2: &[u32] = match &directed_adj {
                Some(adj) => &adj[vi],
                None => graph.neighbors(vi),
            };
            for &w in nbrs_2 {
                if in_ng2.visit(w as usize) {
                    k += 1;
                }
            }
        }

        if k == 0 {
            sum_jaccard += 1.0;
            continue;
        }

        tree.k_nearest_excluding_self_into(i, k, &mut knn_buf)?;
        let mut intersection = 0usize;
        for &(j, _) in &knn_buf {
            if in_ng2.is_visited(j) {
                intersection += 1;
            }
        }
        let union = k + knn_buf.len() - intersection;
        sum_jaccard += intersection as f64 / union as f64;
    }

    Ok(sum_jaccard / n as f64)
}

/// Samples `k` distinct vertex indices from `0..n` in ascending order without replacement.
///
/// Uses Robert Floyd's $O(k)$ sampling algorithm when $k \le n / 4$ to avoid $O(N)$ auxiliary
/// allocations on multi-million-node graphs, and partial Fisher-Yates when $k > n / 4$.
fn sample_distinct_vertices(n: usize, k: usize, seed: u64) -> Vec<usize> {
    debug_assert!(k > 0 && k < n);
    let mut rng = DeterministicRng::from_seed_u64(seed);
    let mut samples = if k <= n / 4 {
        let mut chosen = HashSet::with_capacity(k);
        let mut out = Vec::with_capacity(k);
        for j in (n - k)..n {
            let t = rng.gen_range_usize(j + 1);
            let picked = if chosen.insert(t) {
                t
            } else {
                chosen.insert(j);
                j
            };
            out.push(picked);
        }
        out
    } else {
        let mut perm: Vec<usize> = (0..n).collect();
        for i in 0..k {
            let j = i + rng.gen_range_usize(n - i);
            perm.swap(i, j);
        }
        perm.truncate(k);
        perm
    };
    samples.sort_unstable();
    samples
}

/// Computes deterministic sampled 1-hop neighborhood preservation (`Sampled NP1`) in `[0.0, 1.0]`.
///
/// Samples `min(num_samples, n)` distinct vertices without replacement using `seed` and computes
/// the micro-averaged directed-pair Jaccard similarity
/// $\frac{\sum_{u \in S} |N_G(u) \cap N_D(u)|}{\sum_{u \in S} |N_G(u) \cup N_D(u)|}$.
/// When `num_samples >= n`, all `0..n` vertices are evaluated in order for bit-exact parity
/// with [`neighborhood_preservation_1hop`]. Returns `0.0` for empty, single-vertex, or edgeless
/// graphs, or when all sampled vertices are isolated.
pub fn sampled_neighborhood_preservation_1hop<W, S: Scalar>(
    graph: &CsrGraph<W>,
    layout: &Layout2d<S>,
    num_samples: usize,
    seed: u64,
) -> CoreResult<f64> {
    if num_samples == 0 {
        return Err(CoreError::invalid_input("num_samples must be >= 1"));
    }
    validate_graph_layout(graph, layout, "sampled_neighborhood_preservation_1hop")?;
    let n = layout.len();
    if n <= 1 || graph.num_edges() == 0 {
        return Ok(0.0);
    }
    let tree = build_kdtree_f64(layout)?;
    sampled_neighborhood_preservation_1hop_with_tree(graph, layout, &tree, num_samples, seed)
}

/// Computes [`sampled_neighborhood_preservation_1hop`] reusing a prebuilt [`StaticKdTree<2, f64>`].
pub fn sampled_neighborhood_preservation_1hop_with_tree<W, S: Scalar>(
    graph: &CsrGraph<W>,
    layout: &Layout2d<S>,
    tree: &StaticKdTree<2, f64>,
    num_samples: usize,
    seed: u64,
) -> CoreResult<f64> {
    if num_samples == 0 {
        return Err(CoreError::invalid_input("num_samples must be >= 1"));
    }
    validate_graph_layout(
        graph,
        layout,
        "sampled_neighborhood_preservation_1hop_with_tree",
    )?;
    validate_tree_layout(
        tree,
        layout,
        "sampled_neighborhood_preservation_1hop_with_tree",
    )?;

    let n = layout.len();
    if n <= 1 || graph.num_edges() == 0 {
        return Ok(0.0);
    }

    let directed_adj = if graph.is_directed() {
        Some(build_undirected_adjacency(graph))
    } else {
        None
    };

    let sampled = if num_samples < n {
        Some(sample_distinct_vertices(n, num_samples, seed))
    } else {
        None
    };

    let mut in_ng = VisitStamp::new(n);
    let mut kd_scratch = KdQueryScratch::with_capacity(64);
    let mut knn_buf: Vec<(usize, f64)> = Vec::new();
    let mut cap = 0usize;
    let mut cup = 0usize;

    let mut eval_vertex = |i: usize| -> CoreResult<()> {
        let row: &[u32] = match &directed_adj {
            Some(adj) => &adj[i],
            None => graph.neighbors(i),
        };
        in_ng.clear();
        in_ng.mark(i);
        let mut deg = 0usize;
        for &v in row {
            if in_ng.visit(v as usize) {
                deg += 1;
            }
        }
        if deg == 0 {
            return Ok(());
        }
        cup += deg;
        tree.k_nearest_excluding_self_with_scratch(i, deg, &mut kd_scratch, &mut knn_buf)?;
        for &(j, _) in &knn_buf {
            if in_ng.is_visited(j) {
                cap += 1;
            } else {
                cup += 1;
            }
        }
        Ok(())
    };

    match sampled.as_deref() {
        Some(indices) => {
            for &i in indices {
                eval_vertex(i)?;
            }
        }
        None => {
            for i in 0..n {
                eval_vertex(i)?;
            }
        }
    }

    if cup == 0 {
        Ok(0.0)
    } else {
        Ok(cap as f64 / cup as f64)
    }
}

/// Computes deterministic sampled 2-hop neighborhood preservation (`Sampled NP2`) in `[0.0, 1.0]`.
///
/// Samples `min(num_samples, n)` distinct vertices without replacement using `seed` and averages
/// node-wise 2-hop Jaccard coefficients $|N_G(u, 2) \cap N_D(u, 2)| / |N_G(u, 2) \cup N_D(u, 2)|$
/// (defined as `1.0` when $N_G(u, 2) = \emptyset$, and `0.0` when `n == 0`).
/// When `num_samples >= n`, all `0..n` vertices are evaluated in order for bit-exact parity
/// with [`neighborhood_preservation_2hop`].
pub fn sampled_neighborhood_preservation_2hop<W, S: Scalar>(
    graph: &CsrGraph<W>,
    layout: &Layout2d<S>,
    num_samples: usize,
    seed: u64,
) -> CoreResult<f64> {
    if num_samples == 0 {
        return Err(CoreError::invalid_input("num_samples must be >= 1"));
    }
    validate_graph_layout(graph, layout, "sampled_neighborhood_preservation_2hop")?;
    let n = layout.len();
    if n == 0 {
        return Ok(0.0);
    }
    let tree = build_kdtree_f64(layout)?;
    sampled_neighborhood_preservation_2hop_with_tree(graph, layout, &tree, num_samples, seed)
}

/// Computes [`sampled_neighborhood_preservation_2hop`] reusing a prebuilt [`StaticKdTree<2, f64>`].
pub fn sampled_neighborhood_preservation_2hop_with_tree<W, S: Scalar>(
    graph: &CsrGraph<W>,
    layout: &Layout2d<S>,
    tree: &StaticKdTree<2, f64>,
    num_samples: usize,
    seed: u64,
) -> CoreResult<f64> {
    if num_samples == 0 {
        return Err(CoreError::invalid_input("num_samples must be >= 1"));
    }
    validate_graph_layout(
        graph,
        layout,
        "sampled_neighborhood_preservation_2hop_with_tree",
    )?;
    validate_tree_layout(
        tree,
        layout,
        "sampled_neighborhood_preservation_2hop_with_tree",
    )?;

    let n = layout.len();
    if n == 0 {
        return Ok(0.0);
    }

    let directed_adj = if graph.is_directed() {
        Some(build_undirected_adjacency(graph))
    } else {
        None
    };

    let sample_count = num_samples.min(n);
    let sampled = if num_samples < n {
        Some(sample_distinct_vertices(n, num_samples, seed))
    } else {
        None
    };

    let mut in_ng2 = VisitStamp::new(n);
    let mut kd_scratch = KdQueryScratch::with_capacity(64);
    let mut knn_buf: Vec<(usize, f64)> = Vec::new();
    let mut sum_jaccard = 0.0_f64;

    let mut eval_vertex = |i: usize| -> CoreResult<()> {
        in_ng2.clear();
        in_ng2.mark(i);
        let mut k = 0usize;

        let nbrs_1: &[u32] = match &directed_adj {
            Some(adj) => &adj[i],
            None => graph.neighbors(i),
        };
        for &v in nbrs_1 {
            let vi = v as usize;
            if in_ng2.visit(vi) {
                k += 1;
            }
            let nbrs_2: &[u32] = match &directed_adj {
                Some(adj) => &adj[vi],
                None => graph.neighbors(vi),
            };
            for &w in nbrs_2 {
                if in_ng2.visit(w as usize) {
                    k += 1;
                }
            }
        }

        if k == 0 {
            sum_jaccard += 1.0;
            return Ok(());
        }

        tree.k_nearest_excluding_self_with_scratch(i, k, &mut kd_scratch, &mut knn_buf)?;
        let mut intersection = 0usize;
        for &(j, _) in &knn_buf {
            if in_ng2.is_visited(j) {
                intersection += 1;
            }
        }
        let union = k + knn_buf.len() - intersection;
        sum_jaccard += intersection as f64 / union as f64;
        Ok(())
    };

    match sampled.as_deref() {
        Some(indices) => {
            for &i in indices {
                eval_vertex(i)?;
            }
        }
        None => {
            for i in 0..n {
                eval_vertex(i)?;
            }
        }
    }

    Ok(sum_jaccard / sample_count as f64)
}

#[cfg(test)]
mod tests {
    use super::*;
    use egraph_core::CoreError;

    #[test]
    fn test_np1_and_np2_exact_and_edge_cases() {
        // 3-node path on a line: NP1 == 1.0, NP2 == 1.0.
        let path3 = CsrGraph::<f64>::from_edges(3, &[(0, 1), (1, 2)], None, false, false).unwrap();
        let line3 = Layout2d::from_xy(vec![0.0_f32, 1.0, 2.0], vec![0.0, 0.0, 0.0]).unwrap();
        assert!((neighborhood_preservation_1hop(&path3, &line3).unwrap() - 1.0).abs() < 1e-12);
        assert!((neighborhood_preservation_2hop(&path3, &line3).unwrap() - 1.0).abs() < 1e-12);

        // 4x4 grid layout: every vertex's closest neighbors match graph adjacency.
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
        assert!((neighborhood_preservation_1hop(&grid, &grid_layout).unwrap() - 1.0).abs() < 1e-12);
        // On a unit grid, 2-hop graph radius is 2 (Manhattan <= 2), which matches Euclidean radius <= sqrt(2) < 2.
        assert!((neighborhood_preservation_2hop(&grid, &grid_layout).unwrap() - 1.0).abs() < 1e-12);

        // Empty and edgeless graphs: safe against 0 division.
        let empty = CsrGraph::<f64>::from_edges(0, &[], None, false, false).unwrap();
        let empty_l = Layout2d::<f64>::zeros(0);
        assert_eq!(
            neighborhood_preservation_1hop(&empty, &empty_l).unwrap(),
            0.0
        );
        assert_eq!(
            neighborhood_preservation_2hop(&empty, &empty_l).unwrap(),
            0.0
        );

        let edgeless = CsrGraph::<f64>::from_edges(5, &[], None, false, false).unwrap();
        let edgeless_l = Layout2d::<f64>::zeros(5);
        assert_eq!(
            neighborhood_preservation_1hop(&edgeless, &edgeless_l).unwrap(),
            0.0
        );
        assert_eq!(
            neighborhood_preservation_2hop(&edgeless, &edgeless_l).unwrap(),
            1.0
        );

        // Isolated node mixed with an edge: no panic, finite in [0, 1].
        let mixed = CsrGraph::<f64>::from_edges(4, &[(0, 1)], None, false, false).unwrap();
        let mixed_l =
            Layout2d::from_xy(vec![0.0, 1.0, 5.0, 8.0], vec![0.0, 0.0, 0.0, 0.0]).unwrap();
        let np1 = neighborhood_preservation_1hop(&mixed, &mixed_l).unwrap();
        let np2 = neighborhood_preservation_2hop(&mixed, &mixed_l).unwrap();
        assert!((0.0..=1.0).contains(&np1));
        assert!((0.0..=1.0).contains(&np2));
        assert!((np1 - 1.0).abs() < 1e-12);
        assert!((np2 - 1.0).abs() < 1e-12);

        // Error cases.
        assert!(matches!(
            neighborhood_preservation_1hop(&path3, &Layout2d::<f64>::zeros(2)),
            Err(CoreError::DimensionMismatch { .. })
        ));
        let bad = Layout2d::from_xy(vec![0.0, f64::INFINITY, 1.0], vec![0.0; 3]).unwrap();
        assert!(matches!(
            neighborhood_preservation_2hop(&path3, &bad),
            Err(CoreError::NonFiniteValue { .. })
        ));
    }

    #[test]
    fn test_np1_and_np2_brute_force_parity_and_directed() {
        use egraph_core::DeterministicRng;
        use std::collections::HashSet;

        let mut rng = DeterministicRng::from_seed_u64(515);
        let n = 30usize;
        let mut edges = Vec::new();
        for _ in 0..45 {
            let u = rng.gen_range_usize(n) as u32;
            let v = rng.gen_range_usize(n) as u32;
            if u != v {
                edges.push((u, v));
            }
        }
        let g = CsrGraph::<f64>::from_edges(n, &edges, None, true, true).unwrap();
        let xs: Vec<f64> = (0..n)
            .map(|_| (rng.gen_range_usize(7) as f64) * 0.5)
            .collect();
        let ys: Vec<f64> = (0..n)
            .map(|_| (rng.gen_range_usize(7) as f64) * 0.5)
            .collect();
        let layout = Layout2d::from_xy(xs.clone(), ys.clone()).unwrap();

        let mut adj = vec![HashSet::new(); n];
        for u in 0..n {
            for &v in g.neighbors(u) {
                adj[u].insert(v as usize);
                adj[v as usize].insert(u);
            }
        }

        let mut cap1 = 0usize;
        let mut cup1 = 0usize;
        let mut sum_j2 = 0.0_f64;
        for i in 0..n {
            let mut dists: Vec<(f64, usize)> = (0..n)
                .filter(|&j| j != i)
                .map(|j| {
                    let d2 = (xs[i] - xs[j]).powi(2) + (ys[i] - ys[j]).powi(2);
                    (d2, j)
                })
                .collect();
            dists.sort_unstable_by(|a, b| a.0.total_cmp(&b.0).then_with(|| a.1.cmp(&b.1)));

            let d1 = adj[i].len();
            cup1 += d1;
            for &(_, j) in dists.iter().take(d1) {
                if adj[i].contains(&j) {
                    cap1 += 1;
                } else {
                    cup1 += 1;
                }
            }

            let mut ng2 = adj[i].clone();
            for &v in &adj[i] {
                for &w in &adj[v] {
                    if w != i {
                        ng2.insert(w);
                    }
                }
            }
            if ng2.is_empty() {
                sum_j2 += 1.0;
            } else {
                let k2 = ng2.len();
                let inter = dists
                    .iter()
                    .take(k2)
                    .filter(|&&(_, j)| ng2.contains(&j))
                    .count();
                sum_j2 += inter as f64 / (2 * k2 - inter) as f64;
            }
        }

        let expected_np1 = if cup1 == 0 {
            0.0
        } else {
            cap1 as f64 / cup1 as f64
        };
        let expected_np2 = sum_j2 / n as f64;
        assert!(
            (neighborhood_preservation_1hop(&g, &layout).unwrap() - expected_np1).abs() < 1e-12
        );
        assert!(
            (neighborhood_preservation_2hop(&g, &layout).unwrap() - expected_np2).abs() < 1e-12
        );
    }

    #[test]
    fn test_sampled_np1_and_np2_determinism_exact_parity_and_edge_cases() {
        // 1. Bit-exact parity with exact NP1/NP2 when num_samples >= n (== n and > n)
        // across undirected and directed graphs, plus f32/f64 layouts.
        let mut rng = DeterministicRng::from_seed_u64(808);
        let n = 40usize;
        let mut edges = Vec::new();
        for _ in 0..80 {
            let u = rng.gen_range_usize(n) as u32;
            let v = rng.gen_range_usize(n) as u32;
            if u != v {
                edges.push((u, v));
            }
        }
        let g_undir = CsrGraph::<f64>::from_edges(n, &edges, None, false, true).unwrap();
        let g_dir = CsrGraph::<f32>::from_edges(n, &edges, None, true, true).unwrap();
        let xs64: Vec<f64> = (0..n).map(|_| rng.next_f64() * 10.0).collect();
        let ys64: Vec<f64> = (0..n).map(|_| rng.next_f64() * 10.0).collect();
        let l64 = Layout2d::from_xy(xs64.clone(), ys64.clone()).unwrap();
        let l32 = Layout2d::from_xy(
            xs64.iter().map(|&v| v as f32).collect(),
            ys64.iter().map(|&v| v as f32).collect(),
        )
        .unwrap();

        let exact_np1_u = neighborhood_preservation_1hop(&g_undir, &l64).unwrap();
        let exact_np2_u = neighborhood_preservation_2hop(&g_undir, &l64).unwrap();
        assert_eq!(
            sampled_neighborhood_preservation_1hop(&g_undir, &l64, n, 42).unwrap(),
            exact_np1_u
        );
        assert_eq!(
            sampled_neighborhood_preservation_1hop(&g_undir, &l64, n + 100, 999).unwrap(),
            exact_np1_u
        );
        assert_eq!(
            sampled_neighborhood_preservation_2hop(&g_undir, &l64, n, 42).unwrap(),
            exact_np2_u
        );
        assert_eq!(
            sampled_neighborhood_preservation_2hop(&g_undir, &l64, n + 100, 999).unwrap(),
            exact_np2_u
        );

        let exact_np1_d = neighborhood_preservation_1hop(&g_dir, &l32).unwrap();
        let exact_np2_d = neighborhood_preservation_2hop(&g_dir, &l32).unwrap();
        assert_eq!(
            sampled_neighborhood_preservation_1hop(&g_dir, &l32, n, 1).unwrap(),
            exact_np1_d
        );
        assert_eq!(
            sampled_neighborhood_preservation_2hop(&g_dir, &l32, n + 10, 1).unwrap(),
            exact_np2_d
        );

        // 2. Deterministic reproducibility and seed sensitivity for both Floyd (k <= n/4)
        // and partial Fisher-Yates (k > n/4), plus _with_tree parity.
        let tree64 = build_kdtree_f64(&l64).unwrap();
        for &k in &[5usize, 25usize] {
            let s1_a = sampled_neighborhood_preservation_1hop(&g_undir, &l64, k, 42).unwrap();
            let s1_b =
                sampled_neighborhood_preservation_1hop_with_tree(&g_undir, &l64, &tree64, k, 42)
                    .unwrap();
            let s1_diff = sampled_neighborhood_preservation_1hop(&g_undir, &l64, k, 43).unwrap();
            assert_eq!(s1_a, s1_b);
            assert!((0.0..=1.0).contains(&s1_a));
            assert_ne!(s1_a, s1_diff);

            let s2_a = sampled_neighborhood_preservation_2hop(&g_undir, &l64, k, 42).unwrap();
            let s2_b =
                sampled_neighborhood_preservation_2hop_with_tree(&g_undir, &l64, &tree64, k, 42)
                    .unwrap();
            let s2_diff = sampled_neighborhood_preservation_2hop(&g_undir, &l64, k, 43).unwrap();
            assert_eq!(s2_a, s2_b);
            assert!((0.0..=1.0).contains(&s2_a));
            assert_ne!(s2_a, s2_diff);
        }

        // Verify sample_distinct_vertices returns strictly sorted distinct indices in 0..n.
        for &(nn, kk) in &[(100, 1), (100, 10), (100, 25), (100, 26), (100, 99)] {
            let idxs = sample_distinct_vertices(nn, kk, 77);
            assert_eq!(idxs.len(), kk);
            for w in idxs.windows(2) {
                assert!(w[0] < w[1]);
            }
            assert!(*idxs.last().unwrap() < nn);
        }

        // 3. Edge cases: n == 0, n == 1, edgeless, and disconnected graphs with isolated vertices.
        let empty = CsrGraph::<f64>::from_edges(0, &[], None, false, false).unwrap();
        let empty_l = Layout2d::<f64>::zeros(0);
        assert_eq!(
            sampled_neighborhood_preservation_1hop(&empty, &empty_l, 10, 42).unwrap(),
            0.0
        );
        assert_eq!(
            sampled_neighborhood_preservation_2hop(&empty, &empty_l, 10, 42).unwrap(),
            0.0
        );

        let single = CsrGraph::<f64>::from_edges(1, &[], None, false, false).unwrap();
        let single_l = Layout2d::<f64>::zeros(1);
        assert_eq!(
            sampled_neighborhood_preservation_1hop(&single, &single_l, 10, 42).unwrap(),
            0.0
        );
        assert_eq!(
            sampled_neighborhood_preservation_2hop(&single, &single_l, 10, 42).unwrap(),
            1.0
        );

        let edgeless = CsrGraph::<f64>::from_edges(8, &[], None, false, false).unwrap();
        let edgeless_l = Layout2d::<f64>::zeros(8);
        assert_eq!(
            sampled_neighborhood_preservation_1hop(&edgeless, &edgeless_l, 4, 42).unwrap(),
            0.0
        );
        assert_eq!(
            sampled_neighborhood_preservation_2hop(&edgeless, &edgeless_l, 4, 42).unwrap(),
            1.0
        );

        // Disconnected graph where vertices 2..20 are isolated: find a seed sampling only
        // isolated vertices (num_samples=2) to verify no 0/0 NaN in sampled NP1 and 1.0 in NP2.
        let mostly_isolated =
            CsrGraph::<f64>::from_edges(20, &[(0, 1)], None, false, false).unwrap();
        let mostly_l =
            Layout2d::from_xy((0..20).map(|i| i as f64).collect(), vec![0.0; 20]).unwrap();
        let mut isolated_seed = None;
        for s in 0..200u64 {
            let pick = sample_distinct_vertices(20, 2, s);
            if pick[0] >= 2 && pick[1] >= 2 {
                isolated_seed = Some(s);
                break;
            }
        }
        let iso_s = isolated_seed.expect("expected a seed picking only isolated vertices");
        assert_eq!(
            sampled_neighborhood_preservation_1hop(&mostly_isolated, &mostly_l, 2, iso_s).unwrap(),
            0.0
        );
        assert_eq!(
            sampled_neighborhood_preservation_2hop(&mostly_isolated, &mostly_l, 2, iso_s).unwrap(),
            1.0
        );

        // 4. Error handling: num_samples == 0, DimensionMismatch, NonFiniteValue, tree mismatch.
        assert!(matches!(
            sampled_neighborhood_preservation_1hop(&empty, &empty_l, 0, 42),
            Err(CoreError::InvalidInput(_))
        ));
        assert!(matches!(
            sampled_neighborhood_preservation_2hop(&empty, &empty_l, 0, 42),
            Err(CoreError::InvalidInput(_))
        ));
        assert!(matches!(
            sampled_neighborhood_preservation_1hop(&g_undir, &Layout2d::<f64>::zeros(3), 5, 42),
            Err(CoreError::DimensionMismatch { .. })
        ));
        let bad_l = Layout2d::from_xy(vec![0.0, f64::NAN, 1.0], vec![0.0; 3]).unwrap();
        let path3 = CsrGraph::<f64>::from_edges(3, &[(0, 1), (1, 2)], None, false, false).unwrap();
        assert!(matches!(
            sampled_neighborhood_preservation_2hop(&path3, &bad_l, 2, 42),
            Err(CoreError::NonFiniteValue { .. })
        ));
        let wrong_tree = build_kdtree_f64(&Layout2d::<f64>::zeros(5)).unwrap();
        assert!(matches!(
            sampled_neighborhood_preservation_1hop_with_tree(&g_undir, &l64, &wrong_tree, 5, 42),
            Err(CoreError::DimensionMismatch { .. })
        ));
        assert!(matches!(
            sampled_neighborhood_preservation_2hop_with_tree(&g_undir, &l64, &wrong_tree, 5, 42),
            Err(CoreError::DimensionMismatch { .. })
        ));
    }
}
