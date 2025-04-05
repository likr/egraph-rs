//! Graph data structures for WebAssembly.
//!
//! This module provides WebAssembly bindings for graph data structures
//! based on petgraph, exposed via wasm-bindgen.

mod base;
mod directed;
mod types;
mod undirected;

pub use directed::JsDiGraph;
pub use types::{Edge, IndexType, Node};
pub use undirected::JsGraph;
