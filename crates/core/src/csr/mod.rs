//! Zero-copy `u32`-indexed Compressed Sparse Row (`CsrGraph`) representation.

mod builders;
mod components;
mod validation;

#[cfg(test)]
mod tests;

use crate::error::{CoreError, CoreResult};
use crate::scalar::Scalar;
use builders::{build_directed_from_pairs, build_undirected_from_pairs, sanitize_directed_csr};
use validation::{validate_clean_csr, validate_csr_structure, validate_edge_inputs};

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
