/// Stress Majorization layout algorithm for Python
///
/// This module provides a Python binding for the Stress Majorization algorithm,
/// a force-directed graph layout method that iteratively minimizes the layout stress
/// by solving a series of quadratic problems.
///
/// Stress Majorization is a powerful technique for creating aesthetically pleasing
/// graph layouts that accurately represent the graph-theoretical distances between nodes.
/// It often produces more stable and visually appealing results compared to other
/// force-directed methods.
use crate::{
    distance_matrix::{DistanceMatrixType, PyDistanceMatrix},
    drawing::PyDrawingEuclidean2d,
    graph::{GraphType, PyGraphAdapter},
};
use petgraph::visit::EdgeRef;
use petgraph_layout_stress_majorization::StressMajorization;
use pyo3::{prelude::*, types::PyType};

/// Python class for the Stress Majorization layout algorithm
///
/// This class implements the Stress Majorization algorithm, which creates graph
/// layouts by iteratively minimizing a stress function that measures how well
/// the layout preserves the desired distances between nodes.
///
/// The algorithm works by:
/// 1. Starting with an initial layout (either provided or randomly generated)
/// 2. Iteratively solving a series of quadratic problems using a conjugate gradient method
/// 3. At each step, minimizing the weighted sum of squared differences between
///    actual distances in the layout and desired distances (typically from shortest paths)
///
/// The implementation supports initialization from either a graph with edge weights
/// or a pre-computed distance matrix, and allows for customizable edge weights
/// and convergence criteria.
///
/// Reference: Gansner, E. R., Koren, Y., & North, S. (2004). Graph drawing by stress
/// majorization. In Graph Drawing (pp. 239-250). Springer.
#[pyclass]
#[pyo3(name = "StressMajorization")]
struct PyStressMajorization {
    stress_majorization: StressMajorization,
}

#[pymethods]
impl PyStressMajorization {
    /// Creates a new StressMajorization instance from a graph
    ///
    /// This constructor initializes a StressMajorization layout algorithm using a graph
    /// and an initial drawing. Edge weights are determined by the provided function.
    ///
    /// # Parameters
    /// * `graph` - The graph to layout
    /// * `drawing` - The initial drawing (node positions)
    /// * `f` - A Python function that takes an edge index and returns its weight
    ///
    /// # Returns
    /// A new StressMajorization instance
    #[new]
    fn new(
        graph: &PyGraphAdapter,
        drawing: &PyDrawingEuclidean2d,
        f: &Bound<PyAny>,
    ) -> PyStressMajorization {
        PyStressMajorization {
            stress_majorization: match graph.graph() {
                GraphType::Graph(native_graph) => {
                    StressMajorization::new(native_graph, drawing.drawing(), |e| {
                        f.call1((e.id().index(),)).unwrap().extract().unwrap()
                    })
                }
                _ => panic!("unsupported graph type"),
            },
        }
    }

    /// Creates a new StressMajorization instance from a distance matrix
    ///
    /// This class method initializes a StressMajorization layout algorithm using a
    /// pre-computed distance matrix and an initial drawing. This can be more efficient
    /// when the same distance matrix is reused for multiple layout operations.
    ///
    /// # Parameters
    /// * `drawing` - The initial drawing (node positions)
    /// * `distance_matrix` - A pre-computed matrix of distances between nodes
    ///
    /// # Returns
    /// A new StressMajorization instance
    #[classmethod]
    fn with_distance_matrix(
        _cls: &Bound<PyType>,
        drawing: &PyDrawingEuclidean2d,
        distance_matrix: &PyDistanceMatrix,
    ) -> PyStressMajorization {
        match distance_matrix.distance_matrix() {
            DistanceMatrixType::Full(distance_matrix) => PyStressMajorization {
                stress_majorization: StressMajorization::new_with_distance_matrix(
                    drawing.drawing(),
                    distance_matrix,
                ),
            },
            _ => unimplemented!(),
        }
    }

    /// Performs a single iteration of the stress majorization algorithm
    ///
    /// This method applies one iteration of the stress majorization algorithm to the drawing,
    /// updating node positions to reduce stress.
    ///
    /// # Parameters
    /// * `drawing` - The drawing to update
    ///
    /// # Returns
    /// The new stress value after the iteration (lower is better)
    fn apply(&mut self, drawing: &mut PyDrawingEuclidean2d) -> f32 {
        self.stress_majorization.apply(drawing.drawing_mut())
    }

    /// Runs the stress majorization algorithm until convergence
    ///
    /// This method repeatedly applies the stress majorization algorithm until the
    /// layout converges (the stress no longer decreases significantly) or the maximum
    /// number of iterations is reached.
    ///
    /// # Parameters
    /// * `drawing` - The drawing to optimize
    pub fn run(&mut self, drawing: &mut PyDrawingEuclidean2d) {
        self.stress_majorization.run(drawing.drawing_mut())
    }

    /// Updates the weight matrix using a Python function
    ///
    /// This method allows customizing the weights used in the stress majorization
    /// calculation, which can be used to emphasize or de-emphasize certain node pairs.
    ///
    /// # Parameters
    /// * `f` - A Python function that takes (i, j, distance, weight) and returns a new weight value
    pub fn update_weight(&mut self, f: &Bound<PyAny>) {
        self.stress_majorization
            .update_weight(|i, j, dij, wij| f.call1((i, j, dij, wij)).unwrap().extract().unwrap())
    }
}

pub fn register(_py: Python<'_>, m: &Bound<PyModule>) -> PyResult<()> {
    m.add_class::<PyStressMajorization>()?;
    Ok(())
}
