//! # RdMds (Resistance-distance MDS)
//!
//! This crate provides spectral embedding computation for graphs using the
//! Resistance-distance Multidimensional Scaling (RdMds) algorithm.
//!
//! RdMds computes d-dimensional coordinates by finding the smallest non-zero
//! eigenvalues and eigenvectors of the graph Laplacian matrix. These spectral
//! coordinates can be used as initial embeddings for graph layout algorithms.
//!
//! ## Example
//!
//! ```rust
//! use petgraph::Graph;
//! use petgraph_linalg_rdmds::RdMds;
//! use rand::thread_rng;
//!
//! // Create a graph
//! let mut graph = Graph::new_undirected();
//! let a = graph.add_node(());
//! let b = graph.add_node(());
//! let c = graph.add_node(());
//! graph.add_edge(a, b, ());
//! graph.add_edge(b, c, ());
//! graph.add_edge(c, a, ());
//!
//! // Compute spectral embedding
//! let mut rng = thread_rng();
//! let embedding = RdMds::new()
//!     .d(2)
//!     .shift(1e-3f32)
//!     .embedding(&graph, |_| 1.0f32, &mut rng);
//!
//! // embedding is an Array2 where embedding.row(i) contains the 2D coordinate for node i
//! println!("Embedding shape: {:?}", embedding.dim());
//! ```
//!
//! ## Computational Complexity
//!
//! - Eigenvalue computation: O(d(|V| + |E|))
//! - Total: O(d(|V| + |E|))
//!
//! where d is the number of spectral dimensions, |V| is the number of vertices,
//! and |E| is the number of edges.

mod eigenvalue;
mod rdmds;

pub use eigenvalue::{
    IncompleteCholeskyPreconditioner, LaplacianStructure, compute_smallest_eigenvalues,
    eigendecomposition,
};
pub use rdmds::RdMds;
