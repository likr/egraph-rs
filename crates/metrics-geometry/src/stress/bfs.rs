use egraph_core::{CoreError, CoreResult, CsrGraph, VisitStamp};

/// Reusable scratch buffers for allocation-free repeated single-source BFS over [`CsrGraph`].
#[non_exhaustive]
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
