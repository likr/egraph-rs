// #[macro_use]
// extern crate serde_derive;

//! WebAssembly binding for the egraph library.
//!
//! This crate provides JavaScript/WebAssembly bindings for the egraph library, enabling
//! graph visualization and analysis capabilities in web environments. It exposes Rust
//! implementations of graph data structures, layout algorithms, and drawing utilities
//! through wasm-bindgen.
//!
//! The primary components of this crate include:
//!
//! * Graph data structures ([`graph`]): Undirected and directed graph implementations
//! * Drawing utilities ([`drawing`]): Various geometric representations for graph drawings
//! * Layout algorithms ([`layout`]): Force-directed and other layout algorithms
//! * Edge bundling ([`edge_bundling`]): Methods to reduce visual clutter by bundling edges
//! * Clustering ([`clustering`]): Graph clustering algorithms
//! * Quality metrics ([`quality_metrics`]): Functions to evaluate drawing quality
//! * Random number generation ([`rng`]): Seeded random number generation

// pub mod algorithm;
pub mod clustering;
pub mod drawing;
pub mod edge_bundling;
pub mod graph;
// pub mod grouping;
pub mod layout;
pub mod quality_metrics;
pub mod rng;
