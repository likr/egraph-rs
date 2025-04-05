//! Multidimensional Scaling (MDS) implementations for graph layout.
//!
//! This crate provides implementations of MDS algorithms for visualizing graph structures.
//! MDS is a technique used to visualize the level of similarity of individual nodes in a graph
//! by placing them in a lower-dimensional space (typically 2D or 3D) such that the distances
//! between points in the resulting space approximate the graph-theoretic distances between nodes.
//!
//! Two implementations are provided:
//! * [`ClassicalMds`] - Standard implementation that computes a full distance matrix
//! * [`PivotMds`] - More efficient implementation that uses a subset of nodes as pivots
//!
//! # References
//!
//! * Cox, T. F., & Cox, M. A. (2000). Multidimensional scaling. Chapman & Hall/CRC.
//! * Brandes, U., & Pich, C. (2007). Eigensolver methods for progressive multidimensional scaling of large data.
//!   In Graph Drawing (pp. 42-53). Springer.

mod classical_mds;
mod double_centering;
mod eigendecomposition;
mod pivot_mds;

pub use classical_mds::ClassicalMds;
pub use pivot_mds::PivotMds;
