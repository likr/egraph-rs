/// Overwrap removal algorithm for graph layouts
///
/// This module provides functionality to remove overlaps between nodes in a graph drawing.
/// It implements a force-based algorithm that iteratively adjusts node positions based
/// on their radii, ensuring proper spacing between nodes while attempting to preserve
/// the overall structure of the layout.
use petgraph_layout_overwrap_removal::OverwrapRemoval;
use pyo3::prelude::*;

use crate::{
    drawing::{
        PyDrawingEuclidean, PyDrawingEuclidean2d, PyDrawingHyperbolic2d, PyDrawingSpherical2d,
        PyDrawingTorus2d,
    },
    graph::{GraphType, PyGraphAdapter},
};

/// Python class for the overwrap removal algorithm
///
/// This class provides a post-processing algorithm that resolves node overlaps in
/// graph layouts. It iteratively adjusts node positions based on their defined radii
/// to ensure proper spacing, while attempting to preserve the overall structure of
/// the layout as much as possible.
///
/// The algorithm uses a force-directed approach, where overlapping nodes exert
/// repulsive forces on each other, with the magnitude of the force proportional
/// to the degree of overlap and the specified strength parameter.
#[pyclass]
#[pyo3(name = "OverwrapRemoval")]
struct PyOverwrapRemoval {
    overwrap_removal: OverwrapRemoval<f32>,
}

#[pymethods]
impl PyOverwrapRemoval {
    /// Creates a new overwrap removal algorithm instance
    ///
    /// This constructor initializes an overwrap removal algorithm using a graph
    /// and a function that returns the radius for each node.
    ///
    /// # Parameters
    /// * `graph` - The graph whose layout will be processed
    /// * `f` - A Python function that takes a node index and returns its radius
    ///
    /// # Returns
    /// A new OverwrapRemoval instance
    #[new]
    fn new(graph: &PyGraphAdapter, f: &Bound<PyAny>) -> PyOverwrapRemoval {
        match graph.graph() {
            GraphType::Graph(native_graph) => PyOverwrapRemoval {
                overwrap_removal: OverwrapRemoval::new(native_graph, |u| {
                    f.call1((u.index(),)).unwrap().extract().unwrap()
                }),
            },
            _ => panic!("unsupported graph type"),
        }
    }

    /// Applies the overwrap removal algorithm to a 2D Euclidean drawing
    ///
    /// This method adjusts node positions in the drawing to resolve any overlaps
    /// between nodes based on their radii.
    ///
    /// # Parameters
    /// * `drawing` - The 2D Euclidean drawing to process
    fn apply_with_drawing_euclidean_2d(&self, drawing: &mut PyDrawingEuclidean2d) {
        self.overwrap_removal.apply(drawing.drawing_mut());
    }

    /// Applies the overwrap removal algorithm to an N-dimensional Euclidean drawing
    ///
    /// This method adjusts node positions in the drawing to resolve any overlaps
    /// between nodes based on their radii.
    ///
    /// # Parameters
    /// * `drawing` - The N-dimensional Euclidean drawing to process
    fn apply_with_drawing_euclidean(&self, drawing: &mut PyDrawingEuclidean) {
        self.overwrap_removal.apply(drawing.drawing_mut());
    }

    /// Applies the overwrap removal algorithm to a 2D Hyperbolic drawing
    ///
    /// This method adjusts node positions in the drawing to resolve any overlaps
    /// between nodes based on their radii.
    ///
    /// # Parameters
    /// * `drawing` - The 2D Hyperbolic drawing to process
    fn apply_with_drawing_hyperbolic_2d(&self, drawing: &mut PyDrawingHyperbolic2d) {
        self.overwrap_removal.apply(drawing.drawing_mut());
    }

    /// Applies the overwrap removal algorithm to a 2D Spherical drawing
    ///
    /// This method adjusts node positions in the drawing to resolve any overlaps
    /// between nodes based on their radii.
    ///
    /// # Parameters
    /// * `drawing` - The 2D Spherical drawing to process
    fn apply_with_drawing_spherical_2d(&self, drawing: &mut PyDrawingSpherical2d) {
        self.overwrap_removal.apply(drawing.drawing_mut());
    }

    /// Applies the overwrap removal algorithm to a 2D Torus drawing
    ///
    /// This method adjusts node positions in the drawing to resolve any overlaps
    /// between nodes based on their radii.
    ///
    /// # Parameters
    /// * `drawing` - The 2D Torus drawing to process
    fn apply_with_drawing_torus_2d(&self, drawing: &mut PyDrawingTorus2d) {
        self.overwrap_removal.apply(drawing.drawing_mut());
    }

    /// Gets the strength parameter of the overwrap removal algorithm
    ///
    /// The strength parameter controls how aggressively nodes are pushed apart
    /// when they overlap. Higher values result in more forceful separation.
    ///
    /// # Returns
    /// The current strength value
    #[getter]
    fn get_strength(&self) -> f32 {
        self.overwrap_removal.strength
    }

    /// Sets the strength parameter of the overwrap removal algorithm
    ///
    /// # Parameters
    /// * `value` - The new strength value (typically in range 0.0-1.0)
    #[setter]
    fn set_strength(&mut self, value: f32) {
        self.overwrap_removal.strength = value;
    }

    /// Gets the number of iterations for the overwrap removal algorithm
    ///
    /// This parameter controls how many passes of the algorithm are applied.
    /// More iterations usually result in fewer remaining overlaps but take longer.
    ///
    /// # Returns
    /// The current number of iterations
    #[getter]
    fn get_iterations(&self) -> usize {
        self.overwrap_removal.iterations
    }

    /// Sets the number of iterations for the overwrap removal algorithm
    ///
    /// # Parameters
    /// * `value` - The new number of iterations
    #[setter]
    fn set_iterations(&mut self, value: usize) {
        self.overwrap_removal.iterations = value;
    }

    /// Gets the minimum distance parameter of the overwrap removal algorithm
    ///
    /// This parameter defines the minimum spacing to maintain between nodes,
    /// in addition to their radii.
    ///
    /// # Returns
    /// The current minimum distance value
    #[getter]
    fn get_min_distance(&self) -> f32 {
        self.overwrap_removal.min_distance
    }

    /// Sets the minimum distance parameter of the overwrap removal algorithm
    ///
    /// # Parameters
    /// * `value` - The new minimum distance value
    #[setter]
    fn set_min_distance(&mut self, value: f32) {
        self.overwrap_removal.min_distance = value;
    }
}

pub fn register(_py: Python<'_>, m: &Bound<PyModule>) -> PyResult<()> {
    m.add_class::<PyOverwrapRemoval>()?;
    Ok(())
}
