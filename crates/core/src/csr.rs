//! Zero-copy `u32`-indexed Compressed Sparse Row (`CsrGraph`) representation.

use std::collections::VecDeque;

use crate::error::{CoreError, CoreResult};
use crate::scalar::Scalar;
use crate::visit_stamp::VisitStamp;

/// Compact `u32`-indexed Compressed Sparse Row (CSR) graph with optional numeric edge weights.
///
/// Vertex identifiers are contiguous `0..num_nodes` stored as `u32`. Undirected graphs
/// store symmetric adjacency entries (`u -> v` and `v -> u` with equal weights) and
/// disallow self-loops and parallel edges.
#[derive(Debug, Clone, PartialEq)]
pub struct CsrGraph<W = f64> {
    num_nodes: usize,
    directed: bool,
    indptr: Vec<u32>,
    indices: Vec<u32>,
    weights: Option<Vec<W>>,
}

impl<W> CsrGraph<W> {
    /// Returns the number of vertices in the graph.
    #[inline]
    pub fn num_nodes(&self) -> usize {
        self.num_nodes
    }

    /// Returns the number of logical edges in the graph.
    ///
    /// For directed graphs this equals `indices().len()`; for undirected graphs it equals
    /// `indices().len() / 2`.
    #[inline]
    pub fn num_edges(&self) -> usize {
        if self.directed {
            self.indices.len()
        } else {
            self.indices.len() / 2
        }
    }

    /// Returns the number of stored CSR half-edge entries (`indices().len()`).
    #[inline]
    pub fn nnz(&self) -> usize {
        self.indices.len()
    }

    /// Returns `true` if the graph is directed.
    #[inline]
    pub fn is_directed(&self) -> bool {
        self.directed
    }

    /// Returns `true` if explicit edge weights are stored.
    #[inline]
    pub fn is_weighted(&self) -> bool {
        self.weights.is_some()
    }

    /// Returns the CSR row pointer slice of length `num_nodes + 1`.
    #[inline]
    pub fn indptr(&self) -> &[u32] {
        &self.indptr
    }

    /// Returns the flat neighbor index slice of length `nnz`.
    #[inline]
    pub fn indices(&self) -> &[u32] {
        &self.indices
    }

    /// Returns the optional flat edge weight slice aligned with `indices()`.
    #[inline]
    pub fn weights(&self) -> Option<&[W]> {
        self.weights.as_deref()
    }

    /// Returns the neighbor indices adjacent to `node`.
    ///
    /// # Panics
    ///
    /// Panics if `node >= self.num_nodes()`.
    #[inline]
    pub fn neighbors(&self, node: usize) -> &[u32] {
        let start = self.indptr[node] as usize;
        let end = self.indptr[node + 1] as usize;
        &self.indices[start..end]
    }

    /// Fallible variant of [`CsrGraph::neighbors`] returning [`CoreError::IndexOutOfBounds`].
    #[inline]
    pub fn try_neighbors(&self, node: usize) -> CoreResult<&[u32]> {
        if node >= self.num_nodes {
            return Err(CoreError::index_out_of_bounds(
                node,
                self.num_nodes,
                "CsrGraph::try_neighbors",
            ));
        }
        Ok(self.neighbors(node))
    }

    /// Returns the weights of edges incident to `node`, or `None` if unweighted.
    ///
    /// # Panics
    ///
    /// Panics if `node >= self.num_nodes()`.
    #[inline]
    pub fn neighbor_weights(&self, node: usize) -> Option<&[W]> {
        let start = self.indptr[node] as usize;
        let end = self.indptr[node + 1] as usize;
        self.weights.as_ref().map(|w| &w[start..end])
    }

    /// Fallible variant of [`CsrGraph::neighbor_weights`].
    #[inline]
    pub fn try_neighbor_weights(&self, node: usize) -> CoreResult<Option<&[W]>> {
        if node >= self.num_nodes {
            return Err(CoreError::index_out_of_bounds(
                node,
                self.num_nodes,
                "CsrGraph::try_neighbor_weights",
            ));
        }
        Ok(self.neighbor_weights(node))
    }

    /// Returns the degree (outgoing degree for directed graphs) of `node`.
    ///
    /// # Panics
    ///
    /// Panics if `node >= self.num_nodes()`.
    #[inline]
    pub fn degree(&self, node: usize) -> usize {
        (self.indptr[node + 1] - self.indptr[node]) as usize
    }

    /// Fallible variant of [`CsrGraph::degree`].
    #[inline]
    pub fn try_degree(&self, node: usize) -> CoreResult<usize> {
        if node >= self.num_nodes {
            return Err(CoreError::index_out_of_bounds(
                node,
                self.num_nodes,
                "CsrGraph::try_degree",
            ));
        }
        Ok(self.degree(node))
    }

    /// Returns the sorted original vertex IDs belonging to the largest (weakly) connected component.
    ///
    /// Ties are broken deterministically in favor of the component containing the smallest node ID.
    pub fn largest_connected_component_nodes(&self) -> Vec<u32> {
        if self.num_nodes == 0 {
            return Vec::new();
        }

        // For directed graphs, build lightweight incoming adjacency so BFS visits weak components.
        let reverse_adj = if self.directed {
            Some(self.build_reverse_adjacency())
        } else {
            None
        };

        let mut visited = VisitStamp::new(self.num_nodes);
        let mut queue = VecDeque::new();
        let mut best: Vec<u32> = Vec::new();
        let mut current: Vec<u32> = Vec::new();

        for start in 0..self.num_nodes {
            if !visited.visit(start) {
                continue;
            }
            current.clear();
            queue.push_back(start as u32);
            while let Some(u) = queue.pop_front() {
                current.push(u);
                for &v in self.neighbors(u as usize) {
                    if visited.visit(v as usize) {
                        queue.push_back(v);
                    }
                }
                if let Some((rev_indptr, rev_indices)) = &reverse_adj {
                    let rs = rev_indptr[u as usize] as usize;
                    let re = rev_indptr[u as usize + 1] as usize;
                    for &v in &rev_indices[rs..re] {
                        if visited.visit(v as usize) {
                            queue.push_back(v);
                        }
                    }
                }
            }
            if current.len() > best.len() {
                std::mem::swap(&mut best, &mut current);
            }
        }

        best.sort_unstable();
        best
    }

    /// Extracts the induced subgraph on `nodes` (with compact `0..nodes.len()` vertex IDs).
    ///
    /// Returns an error if any entry in `nodes` is out of range or duplicated.
    pub fn induced_subgraph(&self, nodes: &[u32]) -> CoreResult<Self>
    where
        W: Clone,
    {
        let mut remap = vec![u32::MAX; self.num_nodes];
        for (new_idx, &old_u) in nodes.iter().enumerate() {
            let old = old_u as usize;
            if old >= self.num_nodes {
                return Err(CoreError::index_out_of_bounds(
                    old,
                    self.num_nodes,
                    "CsrGraph::induced_subgraph",
                ));
            }
            if remap[old] != u32::MAX {
                return Err(CoreError::invalid_input(format!(
                    "duplicate vertex {old_u} in induced_subgraph node subset"
                )));
            }
            remap[old] = new_idx as u32;
        }

        let is_sorted = nodes.windows(2).all(|w| w[0] < w[1]);
        let sub_n = nodes.len();
        let mut out_indptr = Vec::with_capacity(sub_n + 1);
        out_indptr.push(0u32);
        let mut out_indices = Vec::new();
        let mut out_weights = self.weights.as_ref().map(|_| Vec::new());

        for &old_u in nodes {
            let start = self.indptr[old_u as usize] as usize;
            let end = self.indptr[old_u as usize + 1] as usize;
            let row_start_pos = out_indices.len();

            if let Some(w_src) = &self.weights {
                let w_dst = out_weights.as_mut().unwrap();
                for pos in start..end {
                    let mapped_v = remap[self.indices[pos] as usize];
                    if mapped_v != u32::MAX {
                        out_indices.push(mapped_v);
                        w_dst.push(w_src[pos].clone());
                    }
                }
                if !is_sorted && out_indices.len() - row_start_pos > 1 {
                    let mut pairs: Vec<(u32, W)> = out_indices
                        .drain(row_start_pos..)
                        .zip(w_dst.drain(row_start_pos..))
                        .collect();
                    pairs.sort_unstable_by_key(|(v, _)| *v);
                    for (v, w) in pairs {
                        out_indices.push(v);
                        w_dst.push(w);
                    }
                }
            } else {
                for pos in start..end {
                    let mapped_v = remap[self.indices[pos] as usize];
                    if mapped_v != u32::MAX {
                        out_indices.push(mapped_v);
                    }
                }
                if !is_sorted && out_indices.len() - row_start_pos > 1 {
                    out_indices[row_start_pos..].sort_unstable();
                }
            }

            let ptr = u32::try_from(out_indices.len())
                .map_err(|_| CoreError::invalid_csr("subgraph edge count exceeds u32::MAX"))?;
            out_indptr.push(ptr);
        }

        Ok(Self {
            num_nodes: sub_n,
            directed: self.directed,
            indptr: out_indptr,
            indices: out_indices,
            weights: out_weights,
        })
    }

    /// Extracts the largest (weakly) connected component as `(subgraph, original_node_ids)`.
    ///
    /// `original_node_ids[new_id]` maps each compact vertex index in the returned subgraph back
    /// to its original vertex index in `self`, sorted in ascending order.
    pub fn largest_connected_component(&self) -> (Self, Vec<u32>)
    where
        W: Clone,
    {
        let lcc_nodes = self.largest_connected_component_nodes();
        if lcc_nodes.len() == self.num_nodes {
            return (self.clone(), lcc_nodes);
        }
        let subgraph = self
            .induced_subgraph(&lcc_nodes)
            .expect("LCC nodes are strictly valid and unique");
        (subgraph, lcc_nodes)
    }

    fn build_reverse_adjacency(&self) -> (Vec<u32>, Vec<u32>) {
        let mut counts = vec![0u32; self.num_nodes];
        for &v in &self.indices {
            counts[v as usize] += 1;
        }
        let mut rev_indptr = Vec::with_capacity(self.num_nodes + 1);
        rev_indptr.push(0u32);
        let mut acc = 0u32;
        for c in &counts {
            acc += *c;
            rev_indptr.push(acc);
        }
        let mut cursor = rev_indptr[..self.num_nodes].to_vec();
        let mut rev_indices = vec![0u32; self.indices.len()];
        for u in 0..self.num_nodes {
            let start = self.indptr[u] as usize;
            let end = self.indptr[u + 1] as usize;
            for &v in &self.indices[start..end] {
                let dst = cursor[v as usize] as usize;
                rev_indices[dst] = u as u32;
                cursor[v as usize] += 1;
            }
        }
        (rev_indptr, rev_indices)
    }
}

impl<W: Scalar> CsrGraph<W> {
    /// Constructs a `CsrGraph` from raw CSR arrays (`indptr`, `indices`, optional `weights`).
    ///
    /// When `sanitize == false`, strictly validates CSR offsets, vertex bounds, finite weights,
    /// absence of self-loops and duplicate edges, and (when `directed == false`) exact symmetry.
    ///
    /// When `sanitize == true`, removes self-loops, sorts each neighbor list, deduplicates parallel
    /// edges by retaining the maximum weight, and symmetrizes undirected graphs without doubling
    /// already-symmetric edge weights.
    pub fn from_csr(
        indptr: &[u32],
        indices: &[u32],
        weights: Option<&[W]>,
        directed: bool,
        sanitize: bool,
    ) -> CoreResult<Self> {
        let num_nodes = validate_csr_structure(indptr, indices, weights)?;

        if !sanitize {
            validate_clean_csr(num_nodes, indptr, indices, weights, directed)?;
            return Ok(Self {
                num_nodes,
                directed,
                indptr: indptr.to_vec(),
                indices: indices.to_vec(),
                weights: weights.map(|w| w.to_vec()),
            });
        }

        if directed {
            sanitize_directed_csr(num_nodes, indptr, indices, weights)
        } else {
            build_undirected_from_pairs(
                num_nodes,
                (0..num_nodes).flat_map(|u| {
                    let start = indptr[u] as usize;
                    let end = indptr[u + 1] as usize;
                    (start..end).map(move |pos| (u as u32, indices[pos], weights.map(|w| w[pos])))
                }),
                weights.is_some(),
                true,
            )
        }
    }

    /// Constructs a `CsrGraph` from an edge list `[(u, v)]` with optional weights.
    ///
    /// When `sanitize == false`, rejects self-loops and duplicate edges (including reverse
    /// duplicates `(u, v)` / `(v, u)` in undirected mode). When `sanitize == true`, drops
    /// self-loops and deduplicates parallel edges by retaining the maximum weight.
    pub fn from_edges(
        num_nodes: usize,
        edges: &[(u32, u32)],
        weights: Option<&[W]>,
        directed: bool,
        sanitize: bool,
    ) -> CoreResult<Self> {
        validate_edge_inputs(num_nodes, edges.len(), weights)?;
        for &(u, v) in edges {
            if (u as usize) >= num_nodes {
                return Err(CoreError::index_out_of_bounds(
                    u as usize,
                    num_nodes,
                    "CsrGraph::from_edges",
                ));
            }
            if (v as usize) >= num_nodes {
                return Err(CoreError::index_out_of_bounds(
                    v as usize,
                    num_nodes,
                    "CsrGraph::from_edges",
                ));
            }
        }

        if directed {
            build_directed_from_pairs(
                num_nodes,
                edges
                    .iter()
                    .enumerate()
                    .map(|(i, &(u, v))| (u, v, weights.map(|w| w[i]))),
                weights.is_some(),
                sanitize,
            )
        } else {
            build_undirected_from_pairs(
                num_nodes,
                edges
                    .iter()
                    .enumerate()
                    .map(|(i, &(u, v))| (u, v, weights.map(|w| w[i]))),
                weights.is_some(),
                sanitize,
            )
        }
    }

    /// Constructs a `CsrGraph` from flat interleaved edge endpoints `[u0, v0, u1, v1, ...]`.
    pub fn from_flat_edges(
        num_nodes: usize,
        edges: &[u32],
        weights: Option<&[W]>,
        directed: bool,
        sanitize: bool,
    ) -> CoreResult<Self> {
        if !edges.len().is_multiple_of(2) {
            return Err(CoreError::dimension_mismatch(
                edges.len() + 1,
                edges.len(),
                "CsrGraph::from_flat_edges",
            ));
        }
        let num_edges = edges.len() / 2;
        validate_edge_inputs(num_nodes, num_edges, weights)?;
        for &endpoint in edges {
            if (endpoint as usize) >= num_nodes {
                return Err(CoreError::index_out_of_bounds(
                    endpoint as usize,
                    num_nodes,
                    "CsrGraph::from_flat_edges",
                ));
            }
        }

        let pairs = (0..num_edges).map(|i| {
            let u = edges[2 * i];
            let v = edges[2 * i + 1];
            (u, v, weights.map(|w| w[i]))
        });

        if directed {
            build_directed_from_pairs(num_nodes, pairs, weights.is_some(), sanitize)
        } else {
            build_undirected_from_pairs(num_nodes, pairs, weights.is_some(), sanitize)
        }
    }
}

fn validate_csr_structure<W: Scalar>(
    indptr: &[u32],
    indices: &[u32],
    weights: Option<&[W]>,
) -> CoreResult<usize> {
    if indptr.is_empty() {
        return Err(CoreError::invalid_csr(
            "indptr must contain at least 1 element (num_nodes + 1)",
        ));
    }
    let num_nodes = indptr.len() - 1;
    if u32::try_from(num_nodes).is_err() {
        return Err(CoreError::invalid_csr("num_nodes exceeds u32::MAX"));
    }
    if indptr[0] != 0 {
        return Err(CoreError::invalid_csr(format!(
            "indptr[0] must be 0, got {}",
            indptr[0]
        )));
    }
    for (row, win) in indptr.windows(2).enumerate() {
        if win[0] > win[1] {
            return Err(CoreError::invalid_csr(format!(
                "indptr is not monotonic at row {row}: {} > {}",
                win[0], win[1]
            )));
        }
    }
    let last_ptr = *indptr.last().unwrap() as usize;
    if last_ptr != indices.len() {
        return Err(CoreError::invalid_csr(format!(
            "indptr[num_nodes] ({last_ptr}) != indices.len() ({})",
            indices.len()
        )));
    }
    for &v in indices {
        if (v as usize) >= num_nodes {
            return Err(CoreError::index_out_of_bounds(
                v as usize,
                num_nodes,
                "CsrGraph::indices",
            ));
        }
    }
    if let Some(w) = weights {
        if w.len() != indices.len() {
            return Err(CoreError::dimension_mismatch(
                indices.len(),
                w.len(),
                "CsrGraph::weights",
            ));
        }
        if w.iter().any(|val| !val.is_finite()) {
            return Err(CoreError::non_finite("CsrGraph::weights"));
        }
    }
    Ok(num_nodes)
}

fn validate_edge_inputs<W: Scalar>(
    num_nodes: usize,
    num_edges: usize,
    weights: Option<&[W]>,
) -> CoreResult<()> {
    if u32::try_from(num_nodes).is_err() {
        return Err(CoreError::invalid_csr("num_nodes exceeds u32::MAX"));
    }
    if let Some(w) = weights {
        if w.len() != num_edges {
            return Err(CoreError::dimension_mismatch(
                num_edges,
                w.len(),
                "CsrGraph::weights",
            ));
        }
        if w.iter().any(|val| !val.is_finite()) {
            return Err(CoreError::non_finite("CsrGraph::weights"));
        }
    }
    Ok(())
}

fn validate_clean_csr<W: Scalar>(
    num_nodes: usize,
    indptr: &[u32],
    indices: &[u32],
    weights: Option<&[W]>,
    directed: bool,
) -> CoreResult<()> {
    let mut seen = VisitStamp::new(num_nodes);
    let mut all_sorted = true;
    for u in 0..num_nodes {
        seen.clear();
        let start = indptr[u] as usize;
        let end = indptr[u + 1] as usize;
        let row = &indices[start..end];
        if !row.windows(2).all(|w| w[0] < w[1]) {
            all_sorted = false;
        }
        for &v in row {
            if v as usize == u {
                return Err(CoreError::invalid_csr(format!(
                    "self-loop ({u}, {u}) not permitted when sanitize=false"
                )));
            }
            if !seen.visit(v as usize) {
                return Err(CoreError::invalid_csr(format!(
                    "duplicate edge ({u}, {v}) not permitted when sanitize=false"
                )));
            }
        }
    }

    if !directed {
        if !indices.len().is_multiple_of(2) {
            return Err(CoreError::invalid_csr(
                "undirected CSR must have an even number of directed entries",
            ));
        }
        if all_sorted {
            for u in 0..num_nodes {
                let start = indptr[u] as usize;
                let end = indptr[u + 1] as usize;
                for pos in start..end {
                    let v = indices[pos] as usize;
                    let v_start = indptr[v] as usize;
                    let v_end = indptr[v + 1] as usize;
                    let Ok(offset) = indices[v_start..v_end].binary_search(&(u as u32)) else {
                        return Err(CoreError::invalid_csr(format!(
                            "undirected CSR missing reverse edge ({v}, {u})"
                        )));
                    };
                    if let Some(w) = weights
                        && w[pos] != w[v_start + offset]
                    {
                        return Err(CoreError::invalid_csr(format!(
                            "undirected CSR asymmetric weight for edge ({u}, {v})"
                        )));
                    }
                }
            }
        } else {
            // Fallback for unsorted but duplicate-free rows: sort canonical endpoint pairs and check 2-by-2 pairing.
            let mut pairs: Vec<(u32, u32, W)> = Vec::with_capacity(indices.len());
            for u in 0..num_nodes {
                let u32_u = u as u32;
                let start = indptr[u] as usize;
                let end = indptr[u + 1] as usize;
                for pos in start..end {
                    let v = indices[pos];
                    let (a, b) = if u32_u < v { (u32_u, v) } else { (v, u32_u) };
                    let w = weights.map_or(W::ONE, |ws| ws[pos]);
                    pairs.push((a, b, w));
                }
            }
            pairs.sort_unstable_by_key(|x| (x.0, x.1));
            for [p0, p1] in pairs.as_chunks::<2>().0 {
                if p0.0 != p1.0 || p0.1 != p1.1 {
                    return Err(CoreError::invalid_csr(format!(
                        "undirected CSR missing reverse edge for ({}, {})",
                        p0.0, p0.1
                    )));
                }
                if weights.is_some() && p0.2 != p1.2 {
                    return Err(CoreError::invalid_csr(format!(
                        "undirected CSR asymmetric weight for edge ({}, {})",
                        p0.0, p0.1
                    )));
                }
            }
        }
    }

    Ok(())
}

fn sanitize_directed_csr<W: Scalar>(
    num_nodes: usize,
    indptr: &[u32],
    indices: &[u32],
    weights: Option<&[W]>,
) -> CoreResult<CsrGraph<W>> {
    let mut out_indptr = Vec::with_capacity(num_nodes + 1);
    out_indptr.push(0u32);
    let mut out_indices = Vec::new();
    let mut out_weights = weights.map(|_| Vec::new());

    if let Some(w_src) = weights {
        let mut row: Vec<(u32, W)> = Vec::new();
        let w_dst = out_weights.as_mut().unwrap();
        for u in 0..num_nodes {
            row.clear();
            let start = indptr[u] as usize;
            let end = indptr[u + 1] as usize;
            for pos in start..end {
                let v = indices[pos];
                if v as usize != u {
                    row.push((v, w_src[pos]));
                }
            }
            row.sort_unstable_by_key(|&(v, _)| v);
            for (v, w) in row.drain(..) {
                if out_indices.len() > *out_indptr.last().unwrap() as usize
                    && *out_indices.last().unwrap() == v
                {
                    let last_w: &mut W = w_dst.last_mut().unwrap();
                    *last_w = (*last_w).max(w);
                } else {
                    out_indices.push(v);
                    w_dst.push(w);
                }
            }
            let ptr = u32::try_from(out_indices.len())
                .map_err(|_| CoreError::invalid_csr("edge count exceeds u32::MAX"))?;
            out_indptr.push(ptr);
        }
    } else {
        let mut row: Vec<u32> = Vec::new();
        for u in 0..num_nodes {
            row.clear();
            let start = indptr[u] as usize;
            let end = indptr[u + 1] as usize;
            for &v in &indices[start..end] {
                if v as usize != u {
                    row.push(v);
                }
            }
            row.sort_unstable();
            row.dedup();
            out_indices.extend_from_slice(&row);
            let ptr = u32::try_from(out_indices.len())
                .map_err(|_| CoreError::invalid_csr("edge count exceeds u32::MAX"))?;
            out_indptr.push(ptr);
        }
    }

    Ok(CsrGraph {
        num_nodes,
        directed: true,
        indptr: out_indptr,
        indices: out_indices,
        weights: out_weights,
    })
}

fn build_directed_from_pairs<W: Scalar>(
    num_nodes: usize,
    pairs: impl Iterator<Item = (u32, u32, Option<W>)>,
    has_weights: bool,
    sanitize: bool,
) -> CoreResult<CsrGraph<W>> {
    if has_weights {
        let mut buckets: Vec<Vec<(u32, W)>> = vec![Vec::new(); num_nodes];
        for (u, v, w) in pairs {
            if u == v {
                if !sanitize {
                    return Err(CoreError::invalid_csr(format!(
                        "self-loop ({u}, {u}) not permitted when sanitize=false"
                    )));
                }
                continue;
            }
            buckets[u as usize].push((v, w.unwrap_or(W::ONE)));
        }

        let mut out_indptr = Vec::with_capacity(num_nodes + 1);
        out_indptr.push(0u32);
        let mut out_indices = Vec::new();
        let mut out_weights = Vec::new();

        for (u, row) in buckets.iter_mut().enumerate() {
            row.sort_unstable_by_key(|&(v, _)| v);
            let row_start = out_indices.len();
            for &(v, w) in row.iter() {
                if out_indices.len() > row_start && *out_indices.last().unwrap() == v {
                    if !sanitize {
                        return Err(CoreError::invalid_csr(format!(
                            "duplicate edge ({u}, {v}) not permitted when sanitize=false"
                        )));
                    }
                    let last_w: &mut W = out_weights.last_mut().unwrap();
                    *last_w = (*last_w).max(w);
                } else {
                    out_indices.push(v);
                    out_weights.push(w);
                }
            }
            let ptr = u32::try_from(out_indices.len())
                .map_err(|_| CoreError::invalid_csr("edge count exceeds u32::MAX"))?;
            out_indptr.push(ptr);
        }

        Ok(CsrGraph {
            num_nodes,
            directed: true,
            indptr: out_indptr,
            indices: out_indices,
            weights: Some(out_weights),
        })
    } else {
        let mut buckets: Vec<Vec<u32>> = vec![Vec::new(); num_nodes];
        for (u, v, _) in pairs {
            if u == v {
                if !sanitize {
                    return Err(CoreError::invalid_csr(format!(
                        "self-loop ({u}, {u}) not permitted when sanitize=false"
                    )));
                }
                continue;
            }
            buckets[u as usize].push(v);
        }

        let mut out_indptr = Vec::with_capacity(num_nodes + 1);
        out_indptr.push(0u32);
        let mut out_indices = Vec::new();

        for (u, row) in buckets.iter_mut().enumerate() {
            row.sort_unstable();
            let prev_len = row.len();
            row.dedup();
            if !sanitize && row.len() != prev_len {
                return Err(CoreError::invalid_csr(format!(
                    "duplicate directed edge from node {u} not permitted when sanitize=false"
                )));
            }
            out_indices.extend_from_slice(row);
            let ptr = u32::try_from(out_indices.len())
                .map_err(|_| CoreError::invalid_csr("edge count exceeds u32::MAX"))?;
            out_indptr.push(ptr);
        }

        Ok(CsrGraph {
            num_nodes,
            directed: true,
            indptr: out_indptr,
            indices: out_indices,
            weights: None,
        })
    }
}

fn build_undirected_from_pairs<W: Scalar>(
    num_nodes: usize,
    pairs: impl Iterator<Item = (u32, u32, Option<W>)>,
    has_weights: bool,
    sanitize: bool,
) -> CoreResult<CsrGraph<W>> {
    if has_weights {
        let mut half_buckets: Vec<Vec<(u32, W)>> = vec![Vec::new(); num_nodes];
        for (u, v, w) in pairs {
            if u == v {
                if !sanitize {
                    return Err(CoreError::invalid_csr(format!(
                        "self-loop ({u}, {u}) not permitted when sanitize=false"
                    )));
                }
                continue;
            }
            let (a, b) = if u < v { (u, v) } else { (v, u) };
            half_buckets[a as usize].push((b, w.unwrap_or(W::ONE)));
        }

        let mut deg = vec![0u32; num_nodes];
        for (a, row) in half_buckets.iter_mut().enumerate() {
            row.sort_unstable_by_key(|&(b, _)| b);
            let mut write = 0usize;
            for read in 0..row.len() {
                let (b, w) = row[read];
                if write > 0 && row[write - 1].0 == b {
                    if !sanitize {
                        return Err(CoreError::invalid_csr(format!(
                            "duplicate undirected edge ({a}, {b}) not permitted when sanitize=false"
                        )));
                    }
                    row[write - 1].1 = row[write - 1].1.max(w);
                } else {
                    row[write] = (b, w);
                    write += 1;
                    deg[a] += 1;
                    deg[b as usize] += 1;
                }
            }
            row.truncate(write);
        }

        let (out_indptr, total_nnz) = prefix_sum_degrees(&deg)?;
        let mut cursor = out_indptr[..num_nodes].to_vec();
        let mut out_indices = vec![0u32; total_nnz];
        let mut out_weights = vec![W::ZERO; total_nnz];

        // Iterating ascending `a` with ascending `b > a` populates every CSR row in strictly sorted order.
        for (a, row) in half_buckets.iter().enumerate() {
            let a_u32 = a as u32;
            for &(b, w) in row {
                let pa = cursor[a] as usize;
                out_indices[pa] = b;
                out_weights[pa] = w;
                cursor[a] += 1;

                let pb = cursor[b as usize] as usize;
                out_indices[pb] = a_u32;
                out_weights[pb] = w;
                cursor[b as usize] += 1;
            }
        }

        Ok(CsrGraph {
            num_nodes,
            directed: false,
            indptr: out_indptr,
            indices: out_indices,
            weights: Some(out_weights),
        })
    } else {
        let mut half_buckets: Vec<Vec<u32>> = vec![Vec::new(); num_nodes];
        for (u, v, _) in pairs {
            if u == v {
                if !sanitize {
                    return Err(CoreError::invalid_csr(format!(
                        "self-loop ({u}, {u}) not permitted when sanitize=false"
                    )));
                }
                continue;
            }
            let (a, b) = if u < v { (u, v) } else { (v, u) };
            half_buckets[a as usize].push(b);
        }

        let mut deg = vec![0u32; num_nodes];
        for (a, row) in half_buckets.iter_mut().enumerate() {
            row.sort_unstable();
            let prev_len = row.len();
            row.dedup();
            if !sanitize && row.len() != prev_len {
                return Err(CoreError::invalid_csr(format!(
                    "duplicate undirected edge incident to node {a} not permitted when sanitize=false"
                )));
            }
            deg[a] += row.len() as u32;
            for &b in row.iter() {
                deg[b as usize] += 1;
            }
        }

        let (out_indptr, total_nnz) = prefix_sum_degrees(&deg)?;
        let mut cursor = out_indptr[..num_nodes].to_vec();
        let mut out_indices = vec![0u32; total_nnz];

        for (a, row) in half_buckets.iter().enumerate() {
            let a_u32 = a as u32;
            for &b in row {
                let pa = cursor[a] as usize;
                out_indices[pa] = b;
                cursor[a] += 1;

                let pb = cursor[b as usize] as usize;
                out_indices[pb] = a_u32;
                cursor[b as usize] += 1;
            }
        }

        Ok(CsrGraph {
            num_nodes,
            directed: false,
            indptr: out_indptr,
            indices: out_indices,
            weights: None,
        })
    }
}

fn prefix_sum_degrees(deg: &[u32]) -> CoreResult<(Vec<u32>, usize)> {
    let mut out_indptr = Vec::with_capacity(deg.len() + 1);
    out_indptr.push(0u32);
    let mut acc = 0usize;
    for &d in deg {
        acc += d as usize;
        let ptr =
            u32::try_from(acc).map_err(|_| CoreError::invalid_csr("CSR nnz exceeds u32::MAX"))?;
        out_indptr.push(ptr);
    }
    Ok((out_indptr, acc))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_csr_from_edges_undirected_and_directed() {
        let g =
            CsrGraph::<f64>::from_edges(4, &[(0, 1), (1, 2), (2, 3)], None, false, false).unwrap();
        assert_eq!(g.num_nodes(), 4);
        assert_eq!(g.num_edges(), 3);
        assert_eq!(g.nnz(), 6);
        assert!(!g.is_directed());
        assert!(!g.is_weighted());
        assert_eq!(g.indptr(), &[0, 1, 3, 5, 6]);
        assert_eq!(g.indices(), &[1, 0, 2, 1, 3, 2]);
        assert_eq!(g.neighbors(1), &[0, 2]);
        assert_eq!(g.degree(1), 2);
        assert!(g.weights().is_none());
        assert!(g.neighbor_weights(1).is_none());

        let dg = CsrGraph::<f32>::from_edges(
            3,
            &[(0, 1), (0, 2)],
            Some(&[1.5_f32, 2.5_f32]),
            true,
            false,
        )
        .unwrap();
        assert!(dg.is_directed());
        assert!(dg.is_weighted());
        assert_eq!(dg.num_edges(), 2);
        assert_eq!(dg.neighbors(0), &[1, 2]);
        assert_eq!(dg.neighbor_weights(0), Some(&[1.5_f32, 2.5_f32][..]));
        assert_eq!(dg.degree(1), 0);
    }

    #[test]
    fn test_csr_sanitize_vs_strict_validation() {
        // Unsorted symmetric + self-loop + duplicate + missing reverse edge:
        // 0 -> [0, 2, 1, 2], 1 -> [0], 2 -> []
        let indptr = [0, 4, 5, 5];
        let indices = [0, 2, 1, 2, 0];
        let weights = [9.0, 2.0, 1.5, 4.0, 1.5];

        let err = CsrGraph::from_csr(&indptr, &indices, Some(&weights), false, false).unwrap_err();
        assert!(matches!(err, CoreError::InvalidCsr(_)));

        let g = CsrGraph::from_csr(&indptr, &indices, Some(&weights), false, true).unwrap();
        assert_eq!(g.num_nodes(), 3);
        assert_eq!(g.num_edges(), 2);
        assert_eq!(g.neighbors(0), &[1, 2]);
        assert_eq!(g.neighbors(1), &[0]);
        assert_eq!(g.neighbors(2), &[0]);
        // Max weight kept on duplicate (0, 2); symmetric (0, 1) does NOT double weight.
        assert_eq!(g.neighbor_weights(0), Some(&[1.5, 4.0][..]));
        assert_eq!(g.neighbor_weights(1), Some(&[1.5][..]));
        assert_eq!(g.neighbor_weights(2), Some(&[4.0][..]));
    }

    #[test]
    fn test_csr_strict_validation_errors() {
        assert!(matches!(
            CsrGraph::<f64>::from_csr(&[], &[], None, false, false),
            Err(CoreError::InvalidCsr(_))
        ));
        assert!(matches!(
            CsrGraph::<f64>::from_csr(&[1, 1], &[], None, false, false),
            Err(CoreError::InvalidCsr(_))
        ));
        assert!(matches!(
            CsrGraph::<f64>::from_csr(&[0, 2, 1], &[0], None, false, false),
            Err(CoreError::InvalidCsr(_))
        ));
        assert!(matches!(
            CsrGraph::<f64>::from_csr(&[0, 1, 1], &[5], None, false, false),
            Err(CoreError::IndexOutOfBounds {
                index: 5,
                len: 2,
                ..
            })
        ));
        assert!(matches!(
            CsrGraph::<f64>::from_csr(&[0, 1, 2], &[1, 0], Some(&[f64::NAN, 1.0]), false, true),
            Err(CoreError::NonFiniteValue { .. })
        ));
        // Asymmetric undirected CSR rejected when sanitize=false.
        assert!(matches!(
            CsrGraph::<f64>::from_csr(&[0, 1, 2, 3], &[1, 2, 0], None, false, false),
            Err(CoreError::InvalidCsr(_))
        ));
        // Duplicate undirected edge via (0,1) and (1,0) rejected when sanitize=false.
        assert!(matches!(
            CsrGraph::<f64>::from_edges(2, &[(0, 1), (1, 0)], None, false, false),
            Err(CoreError::InvalidCsr(_))
        ));
        // Odd flat edge length rejected.
        assert!(matches!(
            CsrGraph::<f64>::from_flat_edges(3, &[0, 1, 2], None, false, true),
            Err(CoreError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn test_csr_largest_connected_component_and_subgraph() {
        // Component A: {0, 2} (1 edge); Component B: {1, 3, 4, 5} (3 edges, size 4).
        let g = CsrGraph::from_edges(
            6,
            &[(0, 2), (1, 3), (3, 4), (4, 5)],
            Some(&[1.0, 2.0, 3.0, 4.0]),
            false,
            true,
        )
        .unwrap();
        let (lcc, orig_ids) = g.largest_connected_component();
        assert_eq!(orig_ids, vec![1, 3, 4, 5]);
        assert_eq!(lcc.num_nodes(), 4);
        assert_eq!(lcc.num_edges(), 3);
        assert_eq!(lcc.neighbors(0), &[1]); // old 1 -> old 3 (new 1)
        assert_eq!(lcc.neighbor_weights(0), Some(&[2.0][..]));
        assert_eq!(lcc.neighbors(1), &[0, 2]); // old 3 -> old 1, 4 (new 0, 2)
        assert_eq!(lcc.neighbor_weights(1), Some(&[2.0, 3.0][..]));

        // Directed weak component test: 0 <- 1 -> 2, isolated 3.
        let dg = CsrGraph::<f64>::from_edges(4, &[(1, 0), (1, 2)], None, true, false).unwrap();
        let (dlcc, d_orig) = dg.largest_connected_component();
        assert_eq!(d_orig, vec![0, 1, 2]);
        assert_eq!(dlcc.num_nodes(), 3);
        assert_eq!(dlcc.neighbors(1), &[0, 2]);

        // Empty graph LCC.
        let empty = CsrGraph::<f64>::from_edges(0, &[], None, false, false).unwrap();
        let (elcc, e_orig) = empty.largest_connected_component();
        assert_eq!(elcc.num_nodes(), 0);
        assert!(e_orig.is_empty());
    }
}
