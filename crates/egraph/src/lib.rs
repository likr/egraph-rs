#[macro_use]
extern crate serde_derive;

pub mod algorithm;
pub mod edge_bundling;
pub mod graph;
pub mod grouping;
pub mod layout;

pub use graph::{Graph, NodeIndex};
