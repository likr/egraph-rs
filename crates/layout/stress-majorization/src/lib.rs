//! Stress Majorization implementation for graph layout.
//!
//! This crate provides an implementation of the Stress Majorization algorithm,
//! a force-directed graph layout technique that minimizes a stress function.
//! The stress function measures the difference between the Euclidean distances
//! in the layout and the desired or theoretical distances (typically shortest path
//! distances in the graph).
//!
//! # Algorithm
//!
//! Stress Majorization works by iteratively solving a sequence of quadratic problems
//! that approximate the stress function. Each iteration improves the layout by
//! moving nodes to positions that reduce the overall stress.
//!
//! The algorithm implemented here is based on:
//!
//! Gansner, E. R., Koren, Y., & North, S. (2004). Graph drawing by stress
//! majorization. In International Symposium on Graph Drawing (pp. 239-250).
//!
//! # Usage
//!
//! ```
//! use petgraph::prelude::*;
//! use petgraph_drawing::DrawingEuclidean2d;
//! use petgraph_layout_stress_majorization::StressMajorization;
//!
//! // Create a graph
//! let mut graph = Graph::new_undirected();
//! let n1 = graph.add_node(());
//! let n2 = graph.add_node(());
//! let n3 = graph.add_node(());
//! graph.add_edge(n1, n2, ());
//! graph.add_edge(n2, n3, ());
//!
//! // Create initial placement
//! let mut drawing = DrawingEuclidean2d::initial_placement(&graph);
//!
//! // Run stress majorization
//! let mut sm = StressMajorization::new(&graph, &drawing, |_| 1.0);
//! sm.run(&mut drawing);
//! ```

mod cg;
mod solver;

pub use cg::conjugate_gradient;
pub use solver::StressMajorization;

#[cfg(test)]
mod tests;
