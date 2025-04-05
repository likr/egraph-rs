//! Common type definitions for graph data structures.
//!
//! This module defines shared types used across graph implementations.

use wasm_bindgen::prelude::*;

/// Type alias for node data, can be any JavaScript value.
pub type Node = JsValue;

/// Type alias for edge data, can be any JavaScript value.
pub type Edge = JsValue;

/// Type alias for the index type used in the graph, maps to u32.
pub type IndexType = u32;
