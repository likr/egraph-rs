use std::collections::HashSet;

use egraph_core::{
    CoreError, CoreResult, CsrGraph, DeterministicRng, Layout2d, Scalar, VisitStamp,
};
use egraph_spatial_trees::{KdQueryScratch, StaticKdTree};

use crate::{
    build_kdtree_f64, build_undirected_adjacency, validate_graph_layout, validate_tree_layout,
};

/// Samples `k` distinct vertex indices from `0..n` in ascending order without replacement.
///
/// Uses Robert Floyd's $O(k)$ sampling algorithm when $k \le n / 4$ to avoid $O(N)$ auxiliary
/// allocations on multi-million-node graphs, and partial Fisher-Yates when $k > n / 4$.
pub(super) fn sample_distinct_vertices(n: usize, k: usize, seed: u64) -> Vec<usize> {
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
/// with [`super::neighborhood_preservation_1hop`]. Returns `0.0` for empty, single-vertex, or
/// edgeless graphs, or when all sampled vertices are isolated.
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
/// with [`super::neighborhood_preservation_2hop`].
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
