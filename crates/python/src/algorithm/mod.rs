/// Graph algorithm modules for the Python bindings
///
/// This module exports various graph algorithms to Python, providing efficient
/// implementations of common operations on graph data structures.
///
/// # Submodules
///
/// - `shortest_path`: Shortest path algorithms (BFS, Dijkstra, Warshall-Floyd)
mod shortest_path;
use pyo3::prelude::*;

/// Registers algorithm-related functions with the Python module
///
/// This function adds all the graph algorithm functions to the Python module,
/// making them available to be called from Python code.
pub fn register(py: Python<'_>, m: &Bound<PyModule>) -> PyResult<()> {
    shortest_path::register(py, m)?;
    Ok(())
}
