//! Exact 2-hop neighborhood size counting and node-wise / median adaptive perplexity sizing.

use egraph_core::{CoreError, CoreResult, CsrGraph, VisitStamp};

use crate::knn::canonical_unweighted_1hop;

/// Computes exact distinct 1-hop + 2-hop neighborhood sizes $|B_2(u) \setminus \{u\}|$ for every
/// vertex in `graph` using a single reusable `VisitStamp` (`O(1)` clear per vertex).
///
/// Directed inputs are treated as undirected simple adjacency.
pub fn compute_2hop_neighborhood_sizes<W>(graph: &CsrGraph<W>) -> Vec<u32> {
    let n = graph.num_nodes();
    if n == 0 {
        return Vec::new();
    }

    if !graph.is_directed() {
        let mut stamp = VisitStamp::new(n);
        let mut sizes = vec![0u32; n];
        for (u, size_out) in sizes.iter_mut().enumerate() {
            stamp.clear();
            stamp.mark(u);
            let mut count = 0u32;
            let n1 = graph.neighbors(u);
            for &w in n1 {
                let wi = w as usize;
                if wi < n && stamp.visit(wi) {
                    count += 1;
                }
            }
            for &w in n1 {
                let wi = w as usize;
                if wi >= n || wi == u {
                    continue;
                }
                for &v in graph.neighbors(wi) {
                    let vi = v as usize;
                    if vi < n && stamp.visit(vi) {
                        count += 1;
                    }
                }
            }
            *size_out = count;
        }
        return sizes;
    }

    let Ok(adj1) = canonical_unweighted_1hop(graph) else {
        return vec![0; n];
    };
    compute_2hop_neighborhood_sizes(&adj1)
}

fn validate_adaptive_perplexity_params(
    c: f64,
    min_perplexity: f64,
    max_perplexity: f64,
    context: &'static str,
) -> CoreResult<()> {
    if !c.is_finite() {
        return Err(CoreError::non_finite(context));
    }
    if c <= 0.0 {
        return Err(CoreError::invalid_input(format!("c must be > 0, got {c}")));
    }
    if !min_perplexity.is_finite() {
        return Err(CoreError::non_finite(context));
    }
    if min_perplexity <= 0.0 {
        return Err(CoreError::invalid_input(format!(
            "min_perplexity must be > 0, got {min_perplexity}"
        )));
    }
    if !max_perplexity.is_finite() {
        return Err(CoreError::non_finite(context));
    }
    if max_perplexity < min_perplexity {
        return Err(CoreError::invalid_input(format!(
            "max_perplexity ({max_perplexity}) must be >= min_perplexity ({min_perplexity})"
        )));
    }
    Ok(())
}

/// Computes per-node 2-hop adaptive perplexities
/// $u_i = \mathrm{clip}(c \cdot k_i^{(2)},\; \min(\mathrm{min\_perplexity}, U),\; U)$
/// where $U = \min(\mathrm{max\_perplexity}, (N - 1) / 3.0)$.
///
/// Recommended defaults: `c = 0.9`, `min_perplexity = 5.0`, `max_perplexity = 1200.0`.
pub fn compute_nodewise_2hop_perplexities<W>(
    graph: &CsrGraph<W>,
    c: f64,
    min_perplexity: f64,
    max_perplexity: f64,
) -> CoreResult<Vec<f64>> {
    validate_adaptive_perplexity_params(
        c,
        min_perplexity,
        max_perplexity,
        "compute_nodewise_2hop_perplexities",
    )?;
    let n = graph.num_nodes();
    if n == 0 {
        return Ok(Vec::new());
    }
    if n == 1 {
        return Ok(vec![1.0]);
    }
    let upper = max_perplexity.min((n - 1) as f64 / 3.0);
    if upper < 1.0 {
        return Ok(vec![1.0; n]);
    }
    let lower = min_perplexity.min(upper);
    let sizes = compute_2hop_neighborhood_sizes(graph);
    Ok(sizes
        .into_iter()
        .map(|k| (c * (k as f64)).clamp(lower, upper))
        .collect())
}

/// Computes scalar median 2-hop adaptive perplexity
/// $u = \mathrm{clip}(c \cdot \mathrm{median}_i(k_i^{(2)}),\; \min(\mathrm{min\_perplexity}, U),\; U)$
/// where $U = \min(\mathrm{max\_perplexity}, (N - 1) / 3.0)$.
///
/// Recommended defaults: `c = 1.5`, `min_perplexity = 15.0`, `max_perplexity = 1200.0`.
pub fn compute_median_2hop_perplexity<W>(
    graph: &CsrGraph<W>,
    c: f64,
    min_perplexity: f64,
    max_perplexity: f64,
) -> CoreResult<f64> {
    validate_adaptive_perplexity_params(
        c,
        min_perplexity,
        max_perplexity,
        "compute_median_2hop_perplexity",
    )?;
    let n = graph.num_nodes();
    if n <= 1 {
        return Ok(1.0);
    }
    let upper = max_perplexity.min((n - 1) as f64 / 3.0);
    if upper < 1.0 {
        return Ok(1.0);
    }
    let lower = min_perplexity.min(upper);
    let mut sizes = compute_2hop_neighborhood_sizes(graph);
    sizes.sort_unstable();
    let mid = n / 2;
    let med = if n % 2 == 1 {
        sizes[mid] as f64
    } else {
        0.5 * ((sizes[mid - 1] as f64) + (sizes[mid] as f64))
    };
    Ok((c * med).clamp(lower, upper))
}
