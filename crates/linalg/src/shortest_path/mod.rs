//! Reusable shortest-path workspaces (`BfsWorkspace`, `DijkstraWorkspace`) and
//! Core-First MaxMin shortest-path pivot selection.

use std::cmp::{Ordering, Reverse};
use std::collections::BinaryHeap;

use egraph_core::{CoreError, CoreResult, CsrGraph, DeterministicRng, EdgeWeight, VisitStamp};

/// Reusable zero-per-call-allocation Breadth-First Search workspace over [`CsrGraph`].
#[derive(Debug, Clone)]
pub struct BfsWorkspace {
    visited: VisitStamp,
    queue: Vec<u32>,
}

impl BfsWorkspace {
    /// Creates a BFS workspace pre-sized for `num_nodes` vertices.
    pub fn new(num_nodes: usize) -> Self {
        Self {
            visited: VisitStamp::new(num_nodes),
            queue: Vec::with_capacity(num_nodes),
        }
    }

    /// Ensures internal scratch capacity covers at least `num_nodes` vertices and resets marks.
    pub fn ensure_capacity(&mut self, num_nodes: usize) {
        if self.visited.len() < num_nodes {
            self.visited.resize_and_clear(num_nodes);
        } else {
            self.visited.clear();
        }
        self.queue.clear();
        if self.queue.capacity() < num_nodes {
            self.queue.reserve(num_nodes - self.queue.capacity());
        }
    }

    /// Runs unweighted single-source BFS from `source`, writing hop distances into `dist_out`
    /// (`f64::INFINITY` for unreachable nodes) with zero heap allocations once sized.
    pub fn bfs_from<W>(
        &mut self,
        graph: &CsrGraph<W>,
        source: usize,
        dist_out: &mut [f64],
    ) -> CoreResult<()> {
        self.bfs_from_scaled(graph, source, 1.0, dist_out)
    }

    /// Runs single-source BFS with uniform edge step `unit_edge_length` (`> 0`, finite).
    pub fn bfs_from_scaled<W>(
        &mut self,
        graph: &CsrGraph<W>,
        source: usize,
        unit_edge_length: f64,
        dist_out: &mut [f64],
    ) -> CoreResult<()> {
        let n = graph.num_nodes();
        if dist_out.len() != n {
            return Err(CoreError::dimension_mismatch(
                n,
                dist_out.len(),
                "BfsWorkspace::bfs_from::dist_out",
            ));
        }
        if source >= n {
            return Err(CoreError::index_out_of_bounds(
                source,
                n,
                "BfsWorkspace::bfs_from::source",
            ));
        }
        if !unit_edge_length.is_finite() {
            return Err(CoreError::non_finite(
                "BfsWorkspace::bfs_from_scaled::unit_edge_length",
            ));
        }
        if unit_edge_length <= 0.0 {
            return Err(CoreError::invalid_input(
                "unit_edge_length must be strictly positive",
            ));
        }

        self.ensure_capacity(n);
        dist_out.fill(f64::INFINITY);

        self.visited.mark(source);
        dist_out[source] = 0.0;
        self.queue.push(source as u32);

        let mut head = 0usize;
        while head < self.queue.len() {
            let u = self.queue[head] as usize;
            head += 1;
            let next_dist = dist_out[u] + unit_edge_length;
            for &v in graph.neighbors(u) {
                let vi = v as usize;
                if self.visited.visit(vi) {
                    dist_out[vi] = next_dist;
                    self.queue.push(v);
                }
            }
        }
        Ok(())
    }

    /// Runs unweighted single-source BFS from `source`, writing integer hop distances into
    /// `dist_out` (`u32::MAX` for unreachable nodes) with zero heap allocations once sized.
    pub fn bfs_from_u32<W>(
        &mut self,
        graph: &CsrGraph<W>,
        source: usize,
        dist_out: &mut [u32],
    ) -> CoreResult<()> {
        let n = graph.num_nodes();
        if u32::try_from(n).is_err() {
            return Err(CoreError::invalid_csr("num_nodes exceeds u32::MAX"));
        }
        if dist_out.len() != n {
            return Err(CoreError::dimension_mismatch(
                n,
                dist_out.len(),
                "BfsWorkspace::bfs_from_u32::dist_out",
            ));
        }
        if source >= n {
            return Err(CoreError::index_out_of_bounds(
                source,
                n,
                "BfsWorkspace::bfs_from_u32::source",
            ));
        }

        self.ensure_capacity(n);
        dist_out.fill(u32::MAX);

        self.visited.mark(source);
        dist_out[source] = 0;
        self.queue.push(source as u32);

        let mut head = 0usize;
        while head < self.queue.len() {
            let u = self.queue[head] as usize;
            head += 1;
            let next_dist = dist_out[u] + 1;
            for &v in graph.neighbors(u) {
                let vi = v as usize;
                if self.visited.visit(vi) {
                    dist_out[vi] = next_dist;
                    self.queue.push(v);
                }
            }
        }
        Ok(())
    }
}

impl Default for BfsWorkspace {
    fn default() -> Self {
        Self::new(0)
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct HeapEntry {
    dist: f64,
    node: u32,
}

impl Eq for HeapEntry {}

impl Ord for HeapEntry {
    #[inline]
    fn cmp(&self, other: &Self) -> Ordering {
        self.dist
            .total_cmp(&other.dist)
            .then_with(|| self.node.cmp(&other.node))
    }
}

impl PartialOrd for HeapEntry {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Reusable single-source Dijkstra workspace using epoch-stamped distance/settlement tracking
/// and a reusable `BinaryHeap<Reverse<_>>`.
#[derive(Debug, Clone)]
pub struct DijkstraWorkspace {
    settled: VisitStamp,
    dist_epoch: VisitStamp,
    dist_scratch: Vec<f64>,
    heap: BinaryHeap<Reverse<HeapEntry>>,
}

impl DijkstraWorkspace {
    /// Creates a Dijkstra workspace pre-sized for `num_nodes` vertices.
    pub fn new(num_nodes: usize) -> Self {
        Self {
            settled: VisitStamp::new(num_nodes),
            dist_epoch: VisitStamp::new(num_nodes),
            dist_scratch: vec![f64::INFINITY; num_nodes],
            heap: BinaryHeap::with_capacity(num_nodes),
        }
    }

    /// Ensures internal scratch buffers cover at least `num_nodes` vertices and resets epoch marks.
    pub fn ensure_capacity(&mut self, num_nodes: usize) {
        if self.settled.len() < num_nodes {
            self.settled.resize_and_clear(num_nodes);
            self.dist_epoch.resize_and_clear(num_nodes);
            self.dist_scratch.resize(num_nodes, f64::INFINITY);
        } else {
            self.settled.clear();
            self.dist_epoch.clear();
        }
        self.heap.clear();
        if self.heap.capacity() < num_nodes {
            self.heap.reserve(num_nodes - self.heap.capacity());
        }
    }

    /// Runs single-source Dijkstra from `source` on `graph`, writing shortest-path distances
    /// into `dist_out` (`f64::INFINITY` for unreachable nodes).
    ///
    /// Rejects graphs containing negative or non-finite edge weights. Unweighted graphs treat
    /// every edge as length `1.0`.
    pub fn dijkstra_from<W: EdgeWeight>(
        &mut self,
        graph: &CsrGraph<W>,
        source: usize,
        dist_out: &mut [f64],
    ) -> CoreResult<()> {
        let n = graph.num_nodes();
        if dist_out.len() != n {
            return Err(CoreError::dimension_mismatch(
                n,
                dist_out.len(),
                "DijkstraWorkspace::dijkstra_from::dist_out",
            ));
        }
        if source >= n {
            return Err(CoreError::index_out_of_bounds(
                source,
                n,
                "DijkstraWorkspace::dijkstra_from::source",
            ));
        }

        let weights_opt = graph.weights();
        if let Some(ws) = weights_opt {
            for &w in ws {
                let wf = w.weight_to_f64();
                if !wf.is_finite() {
                    return Err(CoreError::non_finite(
                        "DijkstraWorkspace::dijkstra_from::weights",
                    ));
                }
                if wf < 0.0 {
                    return Err(CoreError::invalid_input(
                        "Dijkstra edge weights must be non-negative",
                    ));
                }
            }
        }

        self.ensure_capacity(n);
        dist_out.fill(f64::INFINITY);

        self.dist_epoch.mark(source);
        self.dist_scratch[source] = 0.0;
        dist_out[source] = 0.0;
        self.heap.push(Reverse(HeapEntry {
            dist: 0.0,
            node: source as u32,
        }));

        while let Some(Reverse(HeapEntry { dist: d_u, node })) = self.heap.pop() {
            let u = node as usize;
            if !self.settled.visit(u) {
                continue;
            }
            let start = graph.indptr()[u] as usize;
            let end = graph.indptr()[u + 1] as usize;
            for pos in start..end {
                let v = graph.indices()[pos] as usize;
                if self.settled.is_visited(v) {
                    continue;
                }
                let w = match weights_opt {
                    Some(ws) => ws[pos].weight_to_f64(),
                    None => 1.0,
                };
                let cand = d_u + w;
                if !cand.is_finite() {
                    return Err(CoreError::numerical_breakdown(
                        "Dijkstra distance overflowed f64",
                    ));
                }
                let cur_v = if self.dist_epoch.is_visited(v) {
                    self.dist_scratch[v]
                } else {
                    f64::INFINITY
                };
                if cand < cur_v {
                    self.dist_epoch.mark(v);
                    self.dist_scratch[v] = cand;
                    dist_out[v] = cand;
                    self.heap.push(Reverse(HeapEntry {
                        dist: cand,
                        node: v as u32,
                    }));
                }
            }
        }

        Ok(())
    }
}

impl Default for DijkstraWorkspace {
    fn default() -> Self {
        Self::new(0)
    }
}

/// Computes single-source weighted shortest-path distances from `source` on `graph`,
/// returning a length-`N` vector (`f64::INFINITY` for unreachable vertices).
///
/// Unweighted graphs treat every edge as length `1.0`. Rejects negative or non-finite
/// edge weights and out-of-bounds `source`.
pub fn dijkstra_single_source<W: EdgeWeight>(
    graph: &CsrGraph<W>,
    source: usize,
) -> CoreResult<Vec<f64>> {
    let n = graph.num_nodes();
    let mut dist = vec![f64::INFINITY; n];
    let mut ws = DijkstraWorkspace::new(n);
    ws.dijkstra_from(graph, source, &mut dist)?;
    Ok(dist)
}

/// Computes an all-pairs shortest-path (APSP) distance matrix via repeated single-source
/// Dijkstra into a reusable flat row-major `out` buffer of length `N * N`.
pub fn all_pairs_dijkstra_into<W: EdgeWeight>(
    graph: &CsrGraph<W>,
    out: &mut Vec<f64>,
    ws: &mut DijkstraWorkspace,
) -> CoreResult<()> {
    let n = graph.num_nodes();
    let total = n
        .checked_mul(n)
        .ok_or_else(|| CoreError::invalid_input("APSP matrix size overflow"))?;
    out.resize(total, f64::INFINITY);
    for src in 0..n {
        let row_start = src * n;
        ws.dijkstra_from(graph, src, &mut out[row_start..row_start + n])?;
    }
    Ok(())
}

/// Computes an all-pairs shortest-path (APSP) flat row-major `N * N` distance matrix using
/// repeated single-source Dijkstra (`f64::INFINITY` for unreachable pairs).
pub fn all_pairs_dijkstra<W: EdgeWeight>(graph: &CsrGraph<W>) -> CoreResult<Vec<f64>> {
    let n = graph.num_nodes();
    let mut out = Vec::new();
    let mut ws = DijkstraWorkspace::new(n);
    all_pairs_dijkstra_into(graph, &mut out, &mut ws)?;
    Ok(out)
}

/// Selects well-spaced shortest-path pivots via Core-First MaxMin BFS using `seed`.
///
/// Prioritizes vertices with `degree >= min_degree` so scale-free hubs/cores are covered
/// first instead of degree-1 leaves, falling back cleanly to non-isolated vertices and
/// finally isolated vertices when fewer core nodes exist.
pub fn select_maxmin_sp_pivots<W>(
    graph: &CsrGraph<W>,
    num_pivots: usize,
    seed: u64,
    min_degree: usize,
) -> CoreResult<Vec<u32>> {
    let n = graph.num_nodes();
    if n == 0 || num_pivots == 0 {
        return Ok(Vec::new());
    }
    let target = num_pivots.min(n);
    if target == n {
        return Ok((0..n as u32).collect());
    }

    let mut core_count = 0usize;
    let mut non_iso_count = 0usize;
    for u in 0..n {
        let deg = graph.degree(u);
        if deg >= min_degree {
            core_count += 1;
        }
        if deg > 0 {
            non_iso_count += 1;
        }
    }

    let mut rng = DeterministicRng::from_seed_u64(seed);
    let first_pivot = if core_count > 0 {
        let pick = rng.gen_range_usize(core_count);
        (0..n)
            .filter(|&u| graph.degree(u) >= min_degree)
            .nth(pick)
            .unwrap() as u32
    } else if non_iso_count > 0 {
        let pick = rng.gen_range_usize(non_iso_count);
        (0..n).filter(|&u| graph.degree(u) > 0).nth(pick).unwrap() as u32
    } else {
        rng.gen_range_usize(n) as u32
    };

    let mut pivots = Vec::with_capacity(target);
    let mut chosen = vec![false; n];
    let mut min_dist = vec![u32::MAX; n];
    let mut bfs_dist = vec![u32::MAX; n];
    let mut ws = BfsWorkspace::new(n);

    pivots.push(first_pivot);
    chosen[first_pivot as usize] = true;
    let mut last = first_pivot as usize;

    for _ in 1..target {
        ws.bfs_from_u32(graph, last, &mut bfs_dist)?;

        let mut best_core: Option<(usize, u32)> = None;
        let mut best_non_iso: Option<(usize, u32)> = None;
        let mut best_any: Option<(usize, u32)> = None;

        for u in 0..n {
            let d = min_dist[u].min(bfs_dist[u]);
            min_dist[u] = d;
            if chosen[u] {
                continue;
            }
            let deg = graph.degree(u);
            if deg >= min_degree && best_core.is_none_or(|(_, best_d)| d > best_d) {
                best_core = Some((u, d));
            }
            if deg > 0 && best_non_iso.is_none_or(|(_, best_d)| d > best_d) {
                best_non_iso = Some((u, d));
            }
            if best_any.is_none_or(|(_, best_d)| d > best_d) {
                best_any = Some((u, d));
            }
        }

        let next_pivot = best_core
            .or(best_non_iso)
            .or(best_any)
            .ok_or_else(|| {
                CoreError::numerical_breakdown("no unchosen vertex available for MaxMin pivot")
            })?
            .0 as u32;
        chosen[next_pivot as usize] = true;
        pivots.push(next_pivot);
        last = next_pivot as usize;
    }

    Ok(pivots)
}

#[cfg(test)]
mod tests;
