//! Bidirectional adapters between `petgraph` graph types and `egraph-core`'s
//! compressed sparse row (`CsrGraph`) representation.
//!
//! This crate is an integration boundary for callers that construct graphs using
//! `petgraph` while keeping solver internals strictly decoupled from `petgraph`.
