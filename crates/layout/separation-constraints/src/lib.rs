//! # Separation Constraints for Graph Layouts
//!
//! This crate implements one-dimensional separation constraints for graph layouts,
//! based on the Quadratic Programming Separation Constraints (QPSC) algorithm
//! presented in the IPSEP-COLA (Incremental Placement with Separation Constraints) paper.
//!
//! Separation constraints are useful for enforcing minimum distances between nodes,
//! ensuring hierarchical relationships, or maintaining specific layout properties
//! while allowing other forces (like stress minimization) to operate.
//!
//! ## Algorithm
//!
//! The implementation uses a gradient projection approach to enforce separation constraints
//! of the form `v_left + gap <= v_right`, where:
//! - `v_left` and `v_right` are variables (typically node coordinates in one dimension)
//! - `gap` is the minimum required separation distance
//!
//! The algorithm works by:
//! 1. Maintaining a block structure where variables are grouped into rigid blocks
//! 2. Iteratively resolving the most violated constraint by either merging blocks
//!    or rearranging variables within a block
//! 3. Projecting the desired positions onto the feasible region defined by the constraints
//!
//! ## Usage
//!
//! ```
//! use petgraph::prelude::*;
//! use petgraph_drawing::{Drawing, DrawingEuclidean2d};
//! use petgraph_layout_separation_constraints::{Constraint, ConstraintGraph};
//!
//! // Create a simple graph
//! let mut graph = Graph::new_undirected();
//! let n1 = graph.add_node(());
//! let n2 = graph.add_node(());
//! graph.add_edge(n1, n2, ());
//! let mut drawing = DrawingEuclidean2d::initial_placement(&graph);
//!
//! // Create a constraint: n1 must be at least 5.0 units left of n2
//! let constraints = vec![
//!     Constraint::new(0, 1, 5.0)
//! ];
//!
//! // Create constraint graph for x-dimension (dimension 0)
//! let mut constraint_graph = ConstraintGraph::new(&drawing, 0, &constraints);
//!
//! // Extract coordinates to apply constraints
//! let mut coords_x = vec![
//!     drawing.x(n1).unwrap(),
//!     drawing.x(n2).unwrap()
//! ];
//!
//! // Project coordinates to satisfy constraints
//! constraint_graph.project(&mut coords_x);
//!
//! // Update the drawing with projected coordinates
//! drawing.set_x(n1, coords_x[0]);
//! drawing.set_x(n2, coords_x[1]);
//!
//! // Verify that the constraint is now satisfied
//! assert!(coords_x[1] - coords_x[0] >= 5.0);
//! ```
//!
//! ## References
//!
//! [1] Dwyer, T., Marriott, K., & Stuckey, P. J. (2006). "Fast node overlap removal—correction."
//! In International Symposium on Graph Drawing (pp. 446-447).
//!
//! [2] Dwyer, T., Koren, Y., & Marriott, K. (2006). "IPSEP-COLA: An incremental procedure for
//! separation constraint layout of graphs." IEEE Transactions on Visualization and Computer Graphics,
//! 12(5), 821-828.

mod constraint_graph;
mod constraints;

pub use constraint_graph::{project_1d, Constraint, ConstraintGraph};
pub use constraints::cluster_overlap::project_clustered_rectangle_no_overlap_constraints;
pub use constraints::layered::generate_layered_constraints;
pub use constraints::rectangle_overlap::generate_rectangle_no_overlap_constraints;
pub use constraints::rectangle_overlap_2d::project_rectangle_no_overlap_constraints_2d;
