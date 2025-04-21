//! Module containing various layering algorithm implementations.
//!
//! This module provides different algorithms for assigning layers to nodes
//! in a directed graph, which is a key step in hierarchical graph layout.

pub mod longest_path;

pub use longest_path::LongestPath;
