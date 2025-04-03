mod graph_impl;

pub use graph_impl::{JsDiGraph, JsGraph};
use wasm_bindgen::prelude::*;

pub type Node = JsValue;
pub type Edge = JsValue;
pub type IndexType = u32;
