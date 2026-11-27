use super::CsrGraph;
use super::builders::{
    build_directed_from_pairs, build_directed_unweighted_from_pairs, build_undirected_from_pairs,
    build_undirected_unweighted_from_pairs, sanitize_directed_csr,
    sanitize_directed_unweighted_csr,
};
use super::validation::{
    validate_clean_csr, validate_clean_csr_topology, validate_csr_structure, validate_csr_topology,
    validate_edge_inputs,
};
use crate::error::{CoreError, CoreResult};
use crate::scalar::Scalar;

impl<W> CsrGraph<W> {
    /// Constructs an unweighted `CsrGraph<W>` (including `CsrGraph<()>`) from raw CSR arrays.
    pub fn from_unweighted_csr(
        indptr: &[u32],
        indices: &[u32],
        directed: bool,
        sanitize: bool,
    ) -> CoreResult<Self> {
        let num_nodes = validate_csr_topology(indptr, indices)?;
        if !sanitize {
            validate_clean_csr_topology(num_nodes, indptr, indices, directed)?;
            return Ok(Self {
                num_nodes,
                directed,
                indptr: indptr.to_vec(),
                indices: indices.to_vec(),
                weights: None,
            });
        }

        if directed {
            sanitize_directed_unweighted_csr(num_nodes, indptr, indices)
        } else {
            build_undirected_unweighted_from_pairs(
                num_nodes,
                (0..num_nodes).flat_map(|u| {
                    let start = indptr[u] as usize;
                    let end = indptr[u + 1] as usize;
                    (start..end).map(move |pos| (u as u32, indices[pos]))
                }),
                true,
            )
        }
    }

    /// Constructs an unweighted `CsrGraph<W>` (including `CsrGraph<()>`) from edge pairs `[(u, v)]`.
    pub fn from_unweighted_edges(
        num_nodes: usize,
        edges: &[(u32, u32)],
        directed: bool,
        sanitize: bool,
    ) -> CoreResult<Self> {
        if u32::try_from(num_nodes).is_err() {
            return Err(CoreError::invalid_csr("num_nodes exceeds u32::MAX"));
        }
        for &(u, v) in edges {
            if (u as usize) >= num_nodes {
                return Err(CoreError::index_out_of_bounds(
                    u as usize,
                    num_nodes,
                    "CsrGraph::from_unweighted_edges",
                ));
            }
            if (v as usize) >= num_nodes {
                return Err(CoreError::index_out_of_bounds(
                    v as usize,
                    num_nodes,
                    "CsrGraph::from_unweighted_edges",
                ));
            }
        }
        if directed {
            build_directed_unweighted_from_pairs(num_nodes, edges.iter().copied(), sanitize)
        } else {
            build_undirected_unweighted_from_pairs(num_nodes, edges.iter().copied(), sanitize)
        }
    }

    /// Constructs an unweighted `CsrGraph<W>` (including `CsrGraph<()>`) from flat `[u0, v0, ...]` endpoints.
    pub fn from_unweighted_flat_edges(
        num_nodes: usize,
        edges: &[u32],
        directed: bool,
        sanitize: bool,
    ) -> CoreResult<Self> {
        if !edges.len().is_multiple_of(2) {
            return Err(CoreError::dimension_mismatch(
                edges.len() + 1,
                edges.len(),
                "CsrGraph::from_unweighted_flat_edges",
            ));
        }
        if u32::try_from(num_nodes).is_err() {
            return Err(CoreError::invalid_csr("num_nodes exceeds u32::MAX"));
        }
        for &endpoint in edges {
            if (endpoint as usize) >= num_nodes {
                return Err(CoreError::index_out_of_bounds(
                    endpoint as usize,
                    num_nodes,
                    "CsrGraph::from_unweighted_flat_edges",
                ));
            }
        }
        let pairs = edges.as_chunks::<2>().0.iter().map(|&[u, v]| (u, v));
        if directed {
            build_directed_unweighted_from_pairs(num_nodes, pairs, sanitize)
        } else {
            build_undirected_unweighted_from_pairs(num_nodes, pairs, sanitize)
        }
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
        let Some(weights_slice) = weights else {
            return Self::from_unweighted_csr(indptr, indices, directed, sanitize);
        };
        let num_nodes = validate_csr_structure(indptr, indices, Some(weights_slice))?;

        if !sanitize {
            validate_clean_csr(num_nodes, indptr, indices, Some(weights_slice), directed)?;
            return Ok(Self {
                num_nodes,
                directed,
                indptr: indptr.to_vec(),
                indices: indices.to_vec(),
                weights: Some(weights_slice.to_vec()),
            });
        }

        if directed {
            sanitize_directed_csr(num_nodes, indptr, indices, Some(weights_slice))
        } else {
            build_undirected_from_pairs(
                num_nodes,
                (0..num_nodes).flat_map(|u| {
                    let start = indptr[u] as usize;
                    let end = indptr[u + 1] as usize;
                    (start..end).map(move |pos| (u as u32, indices[pos], Some(weights_slice[pos])))
                }),
                true,
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
        let Some(weights_slice) = weights else {
            return Self::from_unweighted_edges(num_nodes, edges, directed, sanitize);
        };
        validate_edge_inputs(num_nodes, edges.len(), Some(weights_slice))?;
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
                    .map(|(i, &(u, v))| (u, v, Some(weights_slice[i]))),
                true,
                sanitize,
            )
        } else {
            build_undirected_from_pairs(
                num_nodes,
                edges
                    .iter()
                    .enumerate()
                    .map(|(i, &(u, v))| (u, v, Some(weights_slice[i]))),
                true,
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
        let Some(weights_slice) = weights else {
            return Self::from_unweighted_flat_edges(num_nodes, edges, directed, sanitize);
        };
        if !edges.len().is_multiple_of(2) {
            return Err(CoreError::dimension_mismatch(
                edges.len() + 1,
                edges.len(),
                "CsrGraph::from_flat_edges",
            ));
        }
        let num_edges = edges.len() / 2;
        validate_edge_inputs(num_nodes, num_edges, Some(weights_slice))?;
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
            (u, v, Some(weights_slice[i]))
        });

        if directed {
            build_directed_from_pairs(num_nodes, pairs, true, sanitize)
        } else {
            build_undirected_from_pairs(num_nodes, pairs, true, sanitize)
        }
    }
}
