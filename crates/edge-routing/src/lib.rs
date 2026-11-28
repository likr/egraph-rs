//! Single-threaded, pure-Rust edge routing and edge bundling solvers for `egraph-rs`.

#![warn(missing_docs)]

mod fdeb;

pub use fdeb::{FdebConfig, FdebSolver};
