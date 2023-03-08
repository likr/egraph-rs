mod graph;

pub use graph::{JsDiGraph, JsGraph};
use wasm_bindgen::prelude::*;

pub type Node = JsValue;
pub type Edge = JsValue;
pub type IndexType = u32;
