//! Shared zero-copy flat-buffer ABI layer for Python (PyO3) and WebAssembly (`wasm-bindgen`).
//!
//! `egraph-ffi` bridges host contiguous buffers (`NumPy` arrays and `WebAssembly` linear-memory
//! `TypedArray`s) to [`egraph_core`] without duplicating validation logic or storing host
//! objects inside core Rust structs.

pub mod error;
pub mod graph;
pub mod layout;

pub use error::{ErrorKind, FfiError, FfiResult};
pub use graph::{build_csr_f32, build_csr_f64, build_edges_f32, build_edges_f64};
pub use layout::InterleavedLayout2d;
