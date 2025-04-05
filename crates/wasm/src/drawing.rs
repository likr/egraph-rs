//! Graph drawing utilities for WebAssembly.
//!
//! This module provides WebAssembly bindings for various types of graph drawings,
//! each representing different geometric spaces in which graphs can be drawn.
//! These drawings typically store node positions and provide methods for
//! manipulating layouts in different spaces.
//!
//! The module includes:
//! * Euclidean drawing: Regular Cartesian coordinate system
//! * Euclidean 2D drawing: 2D Cartesian coordinate system
//! * Hyperbolic 2D drawing: Drawing on a hyperbolic plane
//! * Spherical 2D drawing: Drawing on the surface of a sphere
//! * Torus 2D drawing: Drawing on the surface of a torus

mod drawing_euclidean;
mod drawing_euclidean_2d;
mod drawing_hyperbolic_2d;
mod drawing_spherical_2d;
mod drawing_torus_2d;

pub use drawing_euclidean::JsDrawingEuclidean;
pub use drawing_euclidean_2d::JsDrawingEuclidean2d;
pub use drawing_hyperbolic_2d::JsDrawingHyperbolic2d;
pub use drawing_spherical_2d::JsDrawingSpherical2d;
pub use drawing_torus_2d::JsDrawingTorus2d;
