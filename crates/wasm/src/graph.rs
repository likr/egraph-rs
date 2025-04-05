//! Graph data structures for WebAssembly.
//!
//! This module provides WebAssembly bindings for graph data structures
//! based on petgraph, exposed via wasm-bindgen.

mod graph_impl;

pub use graph_impl::{JsDiGraph, JsGraph};
use wasm_bindgen::prelude::*;

/// Type alias for node data, can be any JavaScript value.
pub type Node = JsValue;

/// Type alias for edge data, can be any JavaScript value.
pub type Edge = JsValue;

/// Type alias for the index type used in the graph, maps to u32.
pub type IndexType = u32;
