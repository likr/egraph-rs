//! Directed graph cycle detection, cycle removal, and hierarchical layer assignment on `CsrGraph`.
//!
//! Provides iterative DFS cycle breaking (preventing recursion stack overflow on deep graphs),
//! $O(V + E)$ Kahn topological `LongestPath` layering, and Gansner et al. (1993)
//! `NetworkSimplex` layering minimizing total edge span $\sum (\mathrm{layer}\[v\] - \mathrm{layer}\[u\])$.

mod network_simplex;
mod transitive_reduction;

use std::collections::VecDeque;

use egraph_core::{CoreError, CoreResult, CsrGraph};

use network_simplex::network_simplex_on_dag;
pub use transitive_reduction::{transitive_reduction, transitive_reduction_weighted};

/// Layer assignment algorithm for [`assign_layers`].
#[non_exhaustive]
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, Default)]
pub enum LayeringMethod {
    /// $O(V + E)$ topological longest-path layering from sources.
    #[default]
    LongestPath,
    /// Gansner et al. (1993) network simplex layering minimizing total edge span.
    NetworkSimplex,
}

fn find_back_edge_positions<W>(graph: &CsrGraph<W>) -> CoreResult<Vec<bool>> {
    if !graph.is_directed() {
        return Err(CoreError::unsupported(
            "layering requires a directed CsrGraph",
        ));
    }
    let n = graph.num_nodes();
    let indptr = graph.indptr();
    let indices = graph.indices();
    let mut is_back = vec![false; indices.len()];

    if n == 0 || indices.is_empty() {
        return Ok(is_back);
    }

    // 3-color iterative DFS: 0 = unvisited, 1 = on stack (gray), 2 = finished (black).
    let mut state = vec![0u8; n];
    let mut stack: Vec<(u32, usize)> = Vec::with_capacity(n.min(1024));

    for start in 0..n {
        if state[start] != 0 {
            continue;
        }
        state[start] = 1;
        stack.push((start as u32, indptr[start] as usize));

        while let Some(&mut (u, ref mut pos)) = stack.last_mut() {
            let end = indptr[u as usize + 1] as usize;
            if *pos < end {
                let cur_pos = *pos;
                *pos += 1;
                let v = indices[cur_pos] as usize;
                if state[v] == 1 {
                    is_back[cur_pos] = true;
                } else if state[v] == 0 {
                    state[v] = 1;
                    stack.push((v as u32, indptr[v] as usize));
                }
            } else {
                state[u as usize] = 2;
                stack.pop();
            }
        }
    }

    Ok(is_back)
}

/// Identifies back-edges `(u, v)` whose reversal eliminates all directed cycles in `graph`.
///
/// Uses an iterative 3-color DFS in $O(V + E)$ time without recursion.
pub fn cycle_edges<W>(graph: &CsrGraph<W>) -> CoreResult<Vec<(u32, u32)>> {
    let is_back = find_back_edge_positions(graph)?;
    let n = graph.num_nodes();
    let indptr = graph.indptr();
    let indices = graph.indices();
    let mut result = Vec::new();

    for u in 0..n {
        let start = indptr[u] as usize;
        let end = indptr[u + 1] as usize;
        let u32_u = u as u32;
        for pos in start..end {
            if is_back[pos] {
                result.push((u32_u, indices[pos]));
            }
        }
    }
    Ok(result)
}

/// Constructs an unweighted directed acyclic [`CsrGraph<()>`] by reversing cycle-closing
/// back-edges.
///
/// Cleanly deduplicates any parallel edges created when reversing a 2-cycle.
pub fn remove_cycles<W>(graph: &CsrGraph<W>) -> CoreResult<CsrGraph<()>> {
    let is_back = find_back_edge_positions(graph)?;
    let n = graph.num_nodes();
    let indptr = graph.indptr();
    let indices = graph.indices();

    let mut edges = Vec::with_capacity(indices.len());

    for u in 0..n {
        let start = indptr[u] as usize;
        let end = indptr[u + 1] as usize;
        let u32_u = u as u32;
        for pos in start..end {
            let v = indices[pos];
            if is_back[pos] {
                edges.push((v, u32_u));
            } else {
                edges.push((u32_u, v));
            }
        }
    }

    CsrGraph::from_unweighted_edges(n, &edges, true, true)
}

/// Builds a deduplicated directed acyclic CSR adjacency `(dag_indptr, dag_indices)`
/// from `graph` by internally reversing back-edges.
fn build_internal_dag<W>(graph: &CsrGraph<W>) -> CoreResult<(Vec<u32>, Vec<u32>)> {
    let is_back = find_back_edge_positions(graph)?;
    let n = graph.num_nodes();
    let indptr = graph.indptr();
    let indices = graph.indices();

    let mut buckets: Vec<Vec<u32>> = vec![Vec::new(); n];
    for u in 0..n {
        let start = indptr[u] as usize;
        let end = indptr[u + 1] as usize;
        let u32_u = u as u32;
        for pos in start..end {
            let v = indices[pos];
            if is_back[pos] {
                if v != u32_u {
                    buckets[v as usize].push(u32_u);
                }
            } else if u32_u != v {
                buckets[u].push(v);
            }
        }
    }

    let mut dag_indptr = Vec::with_capacity(n + 1);
    dag_indptr.push(0u32);
    let mut dag_indices = Vec::with_capacity(indices.len());
    for row in &mut buckets {
        row.sort_unstable();
        row.dedup();
        dag_indices.extend_from_slice(row);
        let ptr = u32::try_from(dag_indices.len())
            .map_err(|_| CoreError::invalid_csr("DAG edge count exceeds u32::MAX"))?;
        dag_indptr.push(ptr);
    }

    Ok((dag_indptr, dag_indices))
}

fn longest_path_on_dag(n: usize, dag_indptr: &[u32], dag_indices: &[u32]) -> Vec<u32> {
    let mut in_deg = vec![0u32; n];
    for &v in dag_indices {
        in_deg[v as usize] += 1;
    }

    let mut layers = vec![0u32; n];
    let mut queue = VecDeque::new();
    for (u, &deg) in in_deg.iter().enumerate() {
        if deg == 0 {
            queue.push_back(u as u32);
        }
    }

    while let Some(u) = queue.pop_front() {
        let lu = layers[u as usize];
        let start = dag_indptr[u as usize] as usize;
        let end = dag_indptr[u as usize + 1] as usize;
        for &v in &dag_indices[start..end] {
            let vi = v as usize;
            if layers[vi] <= lu {
                layers[vi] = lu + 1;
            }
            in_deg[vi] -= 1;
            if in_deg[vi] == 0 {
                queue.push_back(v);
            }
        }
    }

    layers
}

/// Assigns hierarchical layers (`0, 1, ...`) to all vertices of a directed [`CsrGraph`].
///
/// Cycles are handled gracefully by internally reversing back-edges, and disconnected or
/// sourceless components are all assigned valid layers with `layer[v] >= layer[u] + 1`
/// for every DAG edge `(u, v)` and `min(layer) == 0` for non-empty graphs.
pub fn assign_layers<W>(graph: &CsrGraph<W>, method: LayeringMethod) -> CoreResult<Vec<u32>> {
    let n = graph.num_nodes();
    if !graph.is_directed() {
        return Err(CoreError::unsupported(
            "layering requires a directed CsrGraph",
        ));
    }
    if n == 0 {
        return Ok(Vec::new());
    }

    let (dag_indptr, dag_indices) = build_internal_dag(graph)?;
    let lp_layers = longest_path_on_dag(n, &dag_indptr, &dag_indices);

    match method {
        LayeringMethod::LongestPath => Ok(lp_layers),
        LayeringMethod::NetworkSimplex => {
            network_simplex_on_dag(n, &dag_indptr, &dag_indices, &lp_layers)
        }
    }
}
