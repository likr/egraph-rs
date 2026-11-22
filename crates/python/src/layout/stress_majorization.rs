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
///
/// Examples:
///
/// ```python
/// import networkx as nx
/// import egraph as eg
///
/// # Create a graph from NetworkX
/// nx_graph = nx.les_miserables_graph()
/// graph = eg.Graph()
/// indices = {}
/// for u in nx_graph.nodes:
///     indices[u] = graph.add_node(u)
/// for u, v in nx_graph.edges:
///     graph.add_edge(indices[u], indices[v], (u, v))
///
/// # Create an initial drawing
/// drawing = eg.DrawingEuclidean2d.initial_placement(graph)
///
/// # Create and run stress majorization using constructor
/// sm = eg.StressMajorization(graph, drawing, lambda _: 100)
/// sm.run(drawing)
/// ```
use crate::{
    FloatType,
    distance_matrix::{DistanceMatrixType, PyDistanceMatrix},
    drawing::PyDrawingEuclidean2d,
    graph::{GraphType, PyGraphAdapter},
};
use petgraph::visit::EdgeRef;
use petgraph_drawing::Drawing;
use petgraph_layout_stress_majorization::StressMajorization;
use pyo3::{prelude::*, types::PyType};

/// Python class for the Stress Majorization layout algorithm
///
/// This class implements the Stress Majorization algorithm, which creates graph
/// layouts by iteratively minimizing a stress function that measures how well
/// the layout preserves the desired distances between nodes.
///
/// The algorithm works by:
///
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
    stress_majorization: StressMajorization<FloatType>,
}

#[pymethods]
impl PyStressMajorization {
    /// Creates a new StressMajorization instance from a graph
    ///
    /// This constructor initializes a StressMajorization layout algorithm using a graph
    /// and an initial drawing. Edge weights are determined by the provided function.
    ///
    /// :param graph: The graph to layout
    /// :type graph: Graph or DiGraph
    /// :param drawing: The initial drawing (node positions)
    /// :type drawing: DrawingEuclidean2d
    /// :param f: A Python function that takes an edge index and returns its weight
    /// :type f: callable
    /// :return: A new StressMajorization instance
    /// :rtype: StressMajorization
    #[new]
    #[pyo3(signature = (graph, drawing, f=None))]
    fn new(
        graph: &PyGraphAdapter,
        drawing: &PyDrawingEuclidean2d,
        f: Option<&Bound<PyAny>>,
    ) -> PyResult<PyStressMajorization> {
        let edge_len = crate::graph::EdgeLength::from_py(f)?;
        let stress_majorization = match graph.graph() {
            GraphType::Graph(native_graph) => {
                if drawing.drawing().len() != native_graph.node_count() {
                    return Err(pyo3::exceptions::PyValueError::new_err(format!(
                        "drawing length ({}) does not match graph node count ({})",
                        drawing.drawing().len(),
                        native_graph.node_count()
                    )));
                }
                let resolved_len = edge_len.resolve_for_graph(native_graph, true)?;
                StressMajorization::try_new(native_graph, drawing.drawing(), |e| {
                    resolved_len.get(e.id().index())
                })
                .map_err(pyo3::exceptions::PyValueError::new_err)?
            }
            _ => {
                return Err(pyo3::exceptions::PyTypeError::new_err(
                    "unsupported graph type for StressMajorization",
                ));
            }
        };
        Ok(PyStressMajorization {
            stress_majorization,
        })
    }

    /// Creates a new StressMajorization instance from a distance matrix
    ///
    /// This class method initializes a StressMajorization layout algorithm using a
    /// pre-computed distance matrix and an initial drawing. This can be more efficient
    /// when the same distance matrix is reused for multiple layout operations.
    ///
    /// :param drawing: The initial drawing (node positions)
    /// :type drawing: DrawingEuclidean2d
    /// :param distance_matrix: A pre-computed matrix of distances between nodes
    /// :type distance_matrix: DistanceMatrix
    /// :return: A new StressMajorization instance
    /// :rtype: StressMajorization
    #[classmethod]
    fn with_distance_matrix(
        _cls: &Bound<PyType>,
        drawing: &PyDrawingEuclidean2d,
        distance_matrix: &PyDistanceMatrix,
    ) -> PyResult<PyStressMajorization> {
        match distance_matrix.distance_matrix() {
            DistanceMatrixType::Full(distance_matrix) => {
                let stress_majorization = StressMajorization::try_new_with_distance_matrix(
                    drawing.drawing(),
                    distance_matrix,
                )
                .map_err(pyo3::exceptions::PyValueError::new_err)?;
                Ok(PyStressMajorization {
                    stress_majorization,
                })
            }
            DistanceMatrixType::Pivoted(_) => Err(pyo3::exceptions::PyTypeError::new_err(
                "Expected a full (N, N) distance matrix, not a pivoted DistanceMatrix",
            )),
        }
    }

    /// Performs a single iteration of the stress majorization algorithm
    ///
    /// This method applies one iteration of the stress majorization algorithm to the drawing,
    /// updating node positions to reduce stress.
    ///
    /// :param drawing: The drawing to update
    /// :type drawing: DrawingEuclidean2d
    /// :return: The new stress value after the iteration (lower is better)
    /// :rtype: float
    fn apply(&mut self, drawing: &mut PyDrawingEuclidean2d) -> PyResult<FloatType> {
        self.stress_majorization
            .try_apply(drawing.drawing_mut())
            .map_err(pyo3::exceptions::PyValueError::new_err)
    }

    /// Runs the stress majorization algorithm until convergence
    ///
    /// This method repeatedly applies the stress majorization algorithm until the
    /// layout converges (the stress no longer decreases significantly) or the maximum
    /// number of iterations is reached.
    ///
    /// :param drawing: The drawing to optimize
    /// :type drawing: DrawingEuclidean2d
    /// :return: None
    /// :rtype: None
    pub fn run(&mut self, drawing: &mut PyDrawingEuclidean2d) -> PyResult<()> {
        self.stress_majorization
            .try_run(drawing.drawing_mut())
            .map_err(pyo3::exceptions::PyValueError::new_err)
    }

    /// Updates the weight matrix using a Python function
    ///
    /// This method allows customizing the weights used in the stress majorization
    /// calculation, which can be used to emphasize or de-emphasize certain node pairs.
    ///
    /// :param f: A Python function that takes (i, j, distance, weight) and returns a new weight value
    /// :type f: callable
    /// :return: None
    /// :rtype: None
    pub fn update_weight(&mut self, f: &Bound<PyAny>) -> PyResult<()> {
        if !f.is_callable() {
            return Err(pyo3::exceptions::PyTypeError::new_err("f must be callable"));
        }
        self.stress_majorization
            .try_update_weight(|i, j, dij, wij| {
                let res = f.call1((i, j, dij, wij))?;
                if res.is_instance_of::<pyo3::types::PyBool>() {
                    return Err(pyo3::exceptions::PyTypeError::new_err(
                        "weight callback must return a float, not bool",
                    ));
                }
                let val: FloatType = res.extract()?;
                if !val.is_finite() || val < 0.0 {
                    return Err(pyo3::exceptions::PyValueError::new_err(format!(
                        "weight for pair ({i}, {j}) must be finite and non-negative, got {val}"
                    )));
                }
                Ok(val)
            })
    }

    /// Update stress majorization weights from an `(N, N)` float64 NumPy matrix.
    pub fn update_weight_from_numpy(
        &mut self,
        w: numpy::PyReadonlyArray2<'_, crate::FloatType>,
    ) -> PyResult<()> {
        let w_view = w.as_array();
        self.stress_majorization
            .try_update_weight_from_array(w_view)
            .map_err(pyo3::exceptions::PyValueError::new_err)
    }

    /// Gets the convergence threshold (epsilon)
    ///
    /// The algorithm stops when the relative change in stress falls below this threshold.
    ///
    /// :return: The current epsilon value
    /// :rtype: float
    #[getter]
    pub fn epsilon(&self) -> FloatType {
        self.stress_majorization.epsilon
    }

    /// Sets the convergence threshold (epsilon)
    ///
    /// A smaller value leads to more precise layouts but may require more iterations.
    ///
    /// :param value: The new epsilon value
    /// :type value: float
    /// :return: None
    /// :rtype: None
    #[setter]
    pub fn set_epsilon(&mut self, value: FloatType) -> PyResult<()> {
        if !value.is_finite() || value <= 0.0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "epsilon must be finite and positive",
            ));
        }
        self.stress_majorization.epsilon = value;
        Ok(())
    }

    /// Gets the maximum number of iterations
    ///
    /// The algorithm will stop after this many iterations even if convergence is not reached.
    ///
    /// :return: The current maximum iterations value
    /// :rtype: int
    #[getter]
    pub fn max_iterations(&self) -> usize {
        self.stress_majorization.max_iterations
    }

    /// Sets the maximum number of iterations
    ///
    /// A larger value allows more iterations for potentially better convergence.
    ///
    /// :param value: The new maximum iterations value
    /// :type value: int
    /// :return: None
    /// :rtype: None
    #[setter]
    pub fn set_max_iterations(&mut self, value: usize) {
        self.stress_majorization.max_iterations = value;
    }
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyStressMajorization>()?;
    Ok(())
}
