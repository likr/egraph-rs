//! Graph data structures for the Python bindings
//!
//! This module provides the core graph data structures for the egraph-rs Python
//! bindings. It offers both undirected and directed graph implementations that
//! efficiently wrap Rust's petgraph library, exposing its functionality to Python.
//!
//! The graph implementations are designed to be flexible, allowing arbitrary Python
//! objects to be stored as node and edge data. This enables seamless integration
//! with existing Python code and data structures.
//!
//! # Main components
//!
//! - `Graph`: Undirected graph implementation where edges have no direction
//! - `DiGraph`: Directed graph implementation where edges have a source and target
//! - `GraphAdapter`: Base class providing common functionality for both graph types
//!
//! These graph classes support common graph operations such as adding/removing nodes
//! and edges, querying neighbors, finding paths, and traversal.

mod adapter;
mod edge_length;
mod graph_base;

use petgraph::prelude::*;
use pyo3::prelude::*;

pub use adapter::PyGraphAdapter;
#[allow(unused_imports)]
pub(crate) use edge_length::{
    EdgeLength, ResolvedEdgeLengths, build_graph_from_edges_numpy, extract_edges_2col,
};

/// Type alias for node data in a graph, which can be any Python object
pub type Node = Py<PyAny>;

/// Type alias for edge data in a graph, which can be any Python object
pub type Edge = Py<PyAny>;

/// Type alias for the index type used in graphs
pub type IndexType = u32;

/// Type alias for node indices in a graph
pub type NodeId = NodeIndex<IndexType>;

/// Enum representing either an undirected graph or a directed graph
///
/// This enum allows the code to work with both directed and undirected graphs
/// through a common interface, while preserving the specific behavior of each
/// graph type.
///
/// # Variants
///
/// * `Graph` - An undirected graph, where edges have no direction
/// * `DiGraph` - A directed graph, where edges have a source and target node
pub enum GraphType {
    /// An undirected graph, where edges have no direction
    Graph(Graph<Node, Edge, Undirected, IndexType>),
    /// A directed graph, where edges have a source and target
    DiGraph(Graph<Node, Edge, Directed, IndexType>),
}

/// Registers graph-related classes with the Python module
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    adapter::register(m)?;
    graph_base::register(m)?;
    Ok(())
}
