/// Drawing classes for different geometric spaces
///
/// This module provides classes that represent graph drawings in various geometric spaces.
/// A drawing maps nodes of a graph to coordinates in some geometric space, which can be
/// 2D or higher dimensional, and can use different geometries (Euclidean, spherical, etc.).
///
/// # Submodules
///
/// - `drawing_base`: Base class for all drawing types
/// - `drawing_euclidean_2d`: 2D Euclidean space drawings with (x,y) coordinates
/// - `drawing_euclidean`: N-dimensional Euclidean space drawings
/// - `drawing_hyperbolic_2d`: 2D Hyperbolic space drawings
/// - `drawing_spherical_2d`: 2D Spherical space drawings
/// - `drawing_torus_2d`: 2D Torus space drawings with periodic boundary conditions
mod drawing_base;
mod drawing_euclidean;
mod drawing_euclidean_2d;
mod drawing_hyperbolic_2d;
mod drawing_spherical_2d;
mod drawing_torus_2d;

pub use drawing_base::*;
pub use drawing_euclidean::*;
pub use drawing_euclidean_2d::*;
pub use drawing_hyperbolic_2d::*;
pub use drawing_spherical_2d::*;
pub use drawing_torus_2d::*;

use pyo3::prelude::*;

/// Registers drawing-related classes with the Python module
///
/// This function adds all the drawing classes to the Python module,
/// making them available to be instantiated and used from Python code.
/// Drawing classes provide the foundation for graph visualization by
/// mapping nodes to positions in various geometric spaces.
pub fn register(_py: Python<'_>, m: &Bound<PyModule>) -> PyResult<()> {
    // Register the base drawing class
    m.add_class::<PyDrawing>()?;

    // Register specific drawing implementation classes
    m.add_class::<PyDrawingEuclidean2d>()?;
    m.add_class::<PyDrawingEuclidean>()?;
    m.add_class::<PyDrawingHyperbolic2d>()?;
    m.add_class::<PyDrawingSpherical2d>()?;
    m.add_class::<PyDrawingTorus2d>()?;
    Ok(())
}
