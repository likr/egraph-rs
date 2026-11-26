//! 1-hop and 2-hop neighborhood preservation metrics backed by `StaticKdTree<2, f64>`.

mod sampled;
#[cfg(test)]
mod tests;

use egraph_core::{CoreResult, CsrGraph, Layout2d, Scalar, VisitStamp};
use egraph_spatial_trees::StaticKdTree;

use crate::{
    build_kdtree_f64, build_undirected_adjacency, validate_graph_layout, validate_tree_layout,
};

pub use sampled::{
    sampled_neighborhood_preservation_1hop, sampled_neighborhood_preservation_1hop_with_tree,
    sampled_neighborhood_preservation_2hop, sampled_neighborhood_preservation_2hop_with_tree,
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
