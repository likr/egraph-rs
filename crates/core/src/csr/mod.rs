//! Zero-copy `u32`-indexed Compressed Sparse Row (`CsrGraph`) representation.

mod builders;
mod components;
mod constructors;
mod graph_builder;
mod validation;

#[cfg(test)]
mod tests;

pub use graph_builder::CsrGraphBuilder;

use crate::error::{CoreError, CoreResult};
use builders::build_undirected_unweighted_from_pairs;

/// Compact `u32`-indexed Compressed Sparse Row (CSR) graph with optional numeric edge weights.
///
/// Vertex identifiers are contiguous `0..num_nodes` stored as `u32`. Undirected graphs
/// store symmetric adjacency entries (`u -> v` and `v -> u` with equal weights) and
/// disallow self-loops and parallel edges.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
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

    /// Returns all logical edge pairs `(u, v)` in the graph.
    ///
    /// For directed graphs, emits every stored `(u, v)` entry in row order; for undirected
    /// graphs, emits each canonical pair `u < v` once (`num_edges()` total).
    pub fn edges(&self) -> Vec<(u32, u32)> {
        let mut out = Vec::with_capacity(self.num_edges());
        for u in 0..self.num_nodes {
            let u32_u = u as u32;
            for &v in self.neighbors(u) {
                if self.directed || u32_u < v {
                    out.push((u32_u, v));
                }
            }
        }
        out
    }

    /// Returns canonical sorted, deduplicated, loopless undirected edge pairs `(u, v)` with `u < v`.
    pub fn canonical_undirected_edges(&self) -> Vec<(u32, u32)> {
        let n = self.num_nodes;
        let mut edges = Vec::with_capacity(self.num_edges());
        for u in 0..n {
            let u32_u = u as u32;
            for &v in self.neighbors(u) {
                if v as usize >= n || v == u32_u {
                    continue;
                }
                if !self.directed {
                    if u32_u < v {
                        edges.push((u32_u, v));
                    }
                } else {
                    edges.push((u32_u.min(v), u32_u.max(v)));
                }
            }
        }
        if self.directed || !edges.windows(2).all(|w| w[0] < w[1]) {
            edges.sort_unstable();
            edges.dedup();
        }
        edges
    }

    /// Converts any graph (directed or undirected, weighted or unweighted) into a clean,
    /// sorted, loopless, duplicate-free undirected unweighted [`CsrGraph<()>`].
    pub fn to_undirected_simple(&self) -> CoreResult<CsrGraph<()>> {
        let pairs = self.canonical_undirected_edges();
        build_undirected_unweighted_from_pairs(self.num_nodes, pairs.into_iter(), true)
    }

    #[inline]
    fn find_neighbor_pos(&self, u: usize, v: usize) -> Option<usize> {
        if u >= self.num_nodes || v >= self.num_nodes {
            return None;
        }
        let start = self.indptr[u] as usize;
        let end = self.indptr[u + 1] as usize;
        let target = v as u32;
        // Linear scan handles unsorted duplicate-free rows permitted by `sanitize = false`.
        self.indices[start..end]
            .iter()
            .position(|&nbr| nbr == target)
            .map(|offset| start + offset)
    }

    /// Returns `true` if edge `(u, v)` exists in the graph (`false` if either vertex is out of bounds).
    #[inline]
    pub fn has_edge(&self, u: usize, v: usize) -> bool {
        self.find_neighbor_pos(u, v).is_some()
    }

    /// Fallible variant of [`CsrGraph::has_edge`] returning [`CoreError::IndexOutOfBounds`].
    #[inline]
    pub fn try_has_edge(&self, u: usize, v: usize) -> CoreResult<bool> {
        if u >= self.num_nodes {
            return Err(CoreError::index_out_of_bounds(
                u,
                self.num_nodes,
                "CsrGraph::try_has_edge",
            ));
        }
        if v >= self.num_nodes {
            return Err(CoreError::index_out_of_bounds(
                v,
                self.num_nodes,
                "CsrGraph::try_has_edge",
            ));
        }
        Ok(self.has_edge(u, v))
    }

    /// Returns the weight of edge `(u, v)`, or `None` if the edge is absent, out of bounds,
    /// or the graph is unweighted (`self.weights().is_none()`).
    #[inline]
    pub fn edge_weight(&self, u: usize, v: usize) -> Option<W>
    where
        W: Copy,
    {
        let weights = self.weights.as_ref()?;
        let pos = self.find_neighbor_pos(u, v)?;
        Some(weights[pos])
    }

    /// Fallible variant of [`CsrGraph::edge_weight`] returning [`CoreError::IndexOutOfBounds`].
    #[inline]
    pub fn try_edge_weight(&self, u: usize, v: usize) -> CoreResult<Option<W>>
    where
        W: Copy,
    {
        if u >= self.num_nodes {
            return Err(CoreError::index_out_of_bounds(
                u,
                self.num_nodes,
                "CsrGraph::try_edge_weight",
            ));
        }
        if v >= self.num_nodes {
            return Err(CoreError::index_out_of_bounds(
                v,
                self.num_nodes,
                "CsrGraph::try_edge_weight",
            ));
        }
        Ok(self.edge_weight(u, v))
    }

    /// Returns a vertex-permuted graph where `old_to_new[old_id] = new_id`, keeping CSR rows
    /// sorted in ascending neighbor order.
    pub fn permute_nodes(&self, old_to_new: &[u32]) -> CoreResult<Self>
    where
        W: Clone,
    {
        let n = self.num_nodes;
        if old_to_new.len() != n {
            return Err(CoreError::dimension_mismatch(
                n,
                old_to_new.len(),
                "CsrGraph::permute_nodes",
            ));
        }
        let mut new_to_old = vec![u32::MAX; n];
        for (old_u, &new_u) in old_to_new.iter().enumerate() {
            let idx = new_u as usize;
            if idx >= n || new_to_old[idx] != u32::MAX {
                return Err(CoreError::invalid_input(format!(
                    "CsrGraph::permute_nodes: permutation must be a bijection on 0..{n}"
                )));
            }
            new_to_old[idx] = old_u as u32;
        }

        let mut out_indptr = Vec::with_capacity(n + 1);
        out_indptr.push(0u32);
        let mut out_indices = Vec::with_capacity(self.indices.len());
        let mut out_weights = self.weights.as_ref().map(|w| Vec::with_capacity(w.len()));

        if let (Some(w_src), Some(w_dst)) = (&self.weights, &mut out_weights) {
            let mut row_buf: Vec<(u32, W)> = Vec::new();
            for &old_u_u32 in &new_to_old {
                let old_u = old_u_u32 as usize;
                let start = self.indptr[old_u] as usize;
                let end = self.indptr[old_u + 1] as usize;
                row_buf.clear();
                for pos in start..end {
                    let new_v = old_to_new[self.indices[pos] as usize];
                    row_buf.push((new_v, w_src[pos].clone()));
                }
                row_buf.sort_unstable_by_key(|&(v, _)| v);
                for (v, w) in row_buf.drain(..) {
                    out_indices.push(v);
                    w_dst.push(w);
                }
                let ptr = u32::try_from(out_indices.len())
                    .map_err(|_| CoreError::invalid_csr("CSR nnz exceeds u32::MAX"))?;
                out_indptr.push(ptr);
            }
        } else {
            for &old_u_u32 in &new_to_old {
                let old_u = old_u_u32 as usize;
                let start = self.indptr[old_u] as usize;
                let end = self.indptr[old_u + 1] as usize;
                let row_start = out_indices.len();
                for pos in start..end {
                    out_indices.push(old_to_new[self.indices[pos] as usize]);
                }
                out_indices[row_start..].sort_unstable();
                let ptr = u32::try_from(out_indices.len())
                    .map_err(|_| CoreError::invalid_csr("CSR nnz exceeds u32::MAX"))?;
                out_indptr.push(ptr);
            }
        }

        Ok(Self {
            num_nodes: n,
            directed: self.directed,
            indptr: out_indptr,
            indices: out_indices,
            weights: out_weights,
        })
    }
}
