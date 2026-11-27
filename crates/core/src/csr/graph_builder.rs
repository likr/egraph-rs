use super::CsrGraph;
use crate::error::{CoreError, CoreResult};
use crate::scalar::Scalar;

/// Incremental graph builder producing a compact [`CsrGraph<W>`] without external graph crates.
#[derive(Debug, Clone)]
pub struct CsrGraphBuilder<W = f64> {
    num_nodes: usize,
    directed: bool,
    edges: Vec<(u32, u32)>,
    weights: Option<Vec<W>>,
    has_unweighted: bool,
}

impl<W> CsrGraphBuilder<W> {
    /// Creates an empty incremental builder.
    pub fn new(directed: bool) -> Self {
        Self {
            num_nodes: 0,
            directed,
            edges: Vec::new(),
            weights: None,
            has_unweighted: false,
        }
    }

    /// Creates an incremental builder pre-populated with `num_nodes` isolated vertices.
    pub fn with_num_nodes(num_nodes: usize, directed: bool) -> CoreResult<Self> {
        if u32::try_from(num_nodes).is_err() {
            return Err(CoreError::invalid_csr("num_nodes exceeds u32::MAX"));
        }
        Ok(Self {
            num_nodes,
            directed,
            edges: Vec::new(),
            weights: None,
            has_unweighted: false,
        })
    }

    /// Returns the current number of vertices registered in the builder.
    #[inline]
    pub fn num_nodes(&self) -> usize {
        self.num_nodes
    }

    /// Returns the number of edges added so far.
    #[inline]
    pub fn num_edges(&self) -> usize {
        self.edges.len()
    }

    /// Returns `true` if the graph to be built is directed.
    #[inline]
    pub fn is_directed(&self) -> bool {
        self.directed
    }

    /// Appends a single new vertex and returns its `u32` vertex ID.
    pub fn add_node(&mut self) -> CoreResult<u32> {
        let id = u32::try_from(self.num_nodes)
            .map_err(|_| CoreError::invalid_csr("num_nodes exceeds u32::MAX"))?;
        self.num_nodes += 1;
        Ok(id)
    }

    /// Appends `count` new vertices.
    pub fn add_nodes(&mut self, count: usize) -> CoreResult<()> {
        let next = self
            .num_nodes
            .checked_add(count)
            .ok_or_else(|| CoreError::invalid_csr("num_nodes overflow"))?;
        if u32::try_from(next).is_err() {
            return Err(CoreError::invalid_csr("num_nodes exceeds u32::MAX"));
        }
        self.num_nodes = next;
        Ok(())
    }

    /// Appends an unweighted edge `(u, v)`.
    ///
    /// Rejects out-of-bounds endpoints and rejects mixing unweighted and weighted edges in the
    /// same builder.
    pub fn add_edge(&mut self, u: u32, v: u32) -> CoreResult<()> {
        if (u as usize) >= self.num_nodes {
            return Err(CoreError::index_out_of_bounds(
                u as usize,
                self.num_nodes,
                "CsrGraphBuilder::add_edge",
            ));
        }
        if (v as usize) >= self.num_nodes {
            return Err(CoreError::index_out_of_bounds(
                v as usize,
                self.num_nodes,
                "CsrGraphBuilder::add_edge",
            ));
        }
        if self.weights.is_some() {
            return Err(CoreError::invalid_input(
                "cannot mix unweighted add_edge and weighted add_weighted_edge in CsrGraphBuilder",
            ));
        }
        self.has_unweighted = true;
        self.edges.push((u, v));
        Ok(())
    }

    /// Builds an unweighted [`CsrGraph<W>`] (e.g., `CsrGraph<()>` or unweighted `CsrGraph<f64>`).
    pub fn build_unweighted(&self, sanitize: bool) -> CoreResult<CsrGraph<W>> {
        if self.weights.is_some() {
            return Err(CoreError::invalid_input(
                "cannot call build_unweighted after add_weighted_edge",
            ));
        }
        CsrGraph::from_unweighted_edges(self.num_nodes, &self.edges, self.directed, sanitize)
    }
}

impl<W: Scalar> CsrGraphBuilder<W> {
    /// Appends a weighted edge `(u, v)` with finite scalar `weight`.
    ///
    /// Rejects out-of-bounds endpoints, non-finite weights, and mixing weighted and unweighted
    /// edges in the same builder.
    pub fn add_weighted_edge(&mut self, u: u32, v: u32, weight: W) -> CoreResult<()> {
        if (u as usize) >= self.num_nodes {
            return Err(CoreError::index_out_of_bounds(
                u as usize,
                self.num_nodes,
                "CsrGraphBuilder::add_weighted_edge",
            ));
        }
        if (v as usize) >= self.num_nodes {
            return Err(CoreError::index_out_of_bounds(
                v as usize,
                self.num_nodes,
                "CsrGraphBuilder::add_weighted_edge",
            ));
        }
        if !weight.is_finite() {
            return Err(CoreError::non_finite("CsrGraphBuilder::add_weighted_edge"));
        }
        if self.has_unweighted {
            return Err(CoreError::invalid_input(
                "cannot mix unweighted add_edge and weighted add_weighted_edge in CsrGraphBuilder",
            ));
        }
        self.weights.get_or_insert_with(Vec::new).push(weight);
        self.edges.push((u, v));
        Ok(())
    }

    /// Builds the [`CsrGraph<W>`] from all accumulated edges and optional weights.
    pub fn build(&self, sanitize: bool) -> CoreResult<CsrGraph<W>> {
        CsrGraph::from_edges(
            self.num_nodes,
            &self.edges,
            self.weights.as_deref(),
            self.directed,
            sanitize,
        )
    }
}
