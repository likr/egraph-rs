use crate::FloatType;
use crate::csr_graph::PyCsrGraph;
use crate::drawing::PyDrawingEuclidean2d;
use crate::ffi_convert::StrictF64;
use crate::graph::{GraphType, PyGraphAdapter};
use crate::v6_m4::{v6_generate_layered_constraints_impl, v6_project_1d_impl};
use petgraph_drawing::Drawing;
use petgraph_layout_separation_constraints::{
    Constraint, generate_layered_constraints, project_1d,
    project_clustered_rectangle_no_overlap_constraints,
    project_rectangle_no_overlap_constraints_2d,
};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;

/// Represents a separation constraint for graph layouts.
///
/// A separation constraint specifies that a node (left) must be at least a certain
/// distance (gap) away from another node (right) in a particular dimension.
///
/// Parameters
/// ----------
/// left : int
///     The index of the left node in the constraint.
/// right : int
///     The index of the right node in the constraint.
/// gap : float
///     The minimum required separation distance between the nodes.
///
/// Examples
/// --------
/// >>> import egraph as eg
/// >>> # Create a constraint: node 0 must be at least 5.0 units left of node 1
/// >>> constraint = eg.Constraint(0, 1, 5.0)
/// >>> constraint.left
/// 0
/// >>> constraint.right
/// 1
/// >>> constraint.gap
/// 5.0
#[pyclass(from_py_object)]
#[derive(Clone)]
#[pyo3(name = "Constraint")]
pub struct PyConstraint {
    constraint: Constraint<FloatType>,
}

#[pymethods]
impl PyConstraint {
    #[new]
    fn new(left: usize, right: usize, gap: FloatType) -> Self {
        PyConstraint {
            constraint: Constraint::new(left, right, gap),
        }
    }

    #[getter]
    fn left(&self) -> usize {
        self.constraint.left
    }

    #[getter]
    fn right(&self) -> usize {
        self.constraint.right
    }

    #[getter]
    fn gap(&self) -> FloatType {
        self.constraint.gap
    }
}

/// Projects node positions to satisfy separation constraints in one dimension.
///
/// This function takes a drawing and a list of constraints, then modifies the node
/// positions in the specified dimension to satisfy all constraints while minimizing
/// the total squared displacement from the original positions.
///
/// Parameters
/// ----------
/// drawing : DrawingEuclidean2d
///     The drawing to modify.
/// dimension : int
///     The dimension (0 for x, 1 for y) along which to apply the constraints.
/// constraints : list of Constraint
///     The separation constraints to satisfy.
///
/// Examples
/// --------
/// >>> import egraph as eg
/// >>> # Create a graph with 2 nodes
/// >>> graph = eg.Graph()
/// >>> n1 = graph.add_node(None)
/// >>> n2 = graph.add_node(None)
/// >>> # Create a drawing with the nodes positioned closely
/// >>> drawing = eg.DrawingEuclidean2d.initial_placement(graph)
/// >>> drawing.set_x(n1, 0.0)
/// >>> drawing.set_y(n1, 0.0)
/// >>> drawing.set_x(n2, 1.0)
/// >>> drawing.set_y(n2, 0.0)
/// >>> # Create a constraint to separate the nodes
/// >>> constraint = eg.Constraint(0, 1, 5.0)  # n1 should be at least 5.0 units left of n2
/// >>> eg.project_1d(drawing, 0, [constraint])  # Apply in x dimension
/// >>> # The nodes should now be properly separated
/// >>> drawing.x(n1) < drawing.x(n2) - 5.0  # This should be True
#[pyfunction]
#[pyo3(name = "project_1d", signature = (target, arg2, arg3 = None))]
fn py_project_1d<'py>(
    py: Python<'py>,
    target: &Bound<'py, PyAny>,
    arg2: &Bound<'py, PyAny>,
    arg3: Option<&Bound<'py, PyAny>>,
) -> PyResult<Option<Bound<'py, PyAny>>> {
    if let Ok(drawing) = target.cast::<PyDrawingEuclidean2d>() {
        let Some(constraints_obj) = arg3 else {
            return Err(PyTypeError::new_err(
                "project_1d(drawing, dimension, constraints) requires 3 arguments for DrawingEuclidean2d",
            ));
        };
        let dimension = arg2.extract::<usize>()?;
        let constraints = constraints_obj.extract::<Vec<PyConstraint>>()?;
        let rust_constraints: Vec<Constraint<FloatType>> =
            constraints.iter().map(|c| c.constraint.clone()).collect();

        let mut borrowed = drawing.try_borrow_mut()?;
        if dimension >= borrowed.drawing().dimension() {
            return Err(PyValueError::new_err(format!(
                "dimension {} out of bounds. Drawing has {} dimensions",
                dimension,
                borrowed.drawing().dimension()
            )));
        }

        project_1d(borrowed.drawing_mut(), dimension, &rust_constraints);
        return Ok(None);
    }

    if arg3.is_some() {
        return Err(PyTypeError::new_err(
            "project_1d(coords, constraints) takes 2 arguments",
        ));
    }
    let arr = v6_project_1d_impl(py, target, arg2)?;
    Ok(Some(arr))
}

/// Projects rectangle positions to satisfy non-overlap constraints in both X and Y dimensions.
///
/// This function generates and applies constraints for both X and Y dimensions to ensure
/// rectangles don't overlap. It's a convenience wrapper that combines constraint generation
/// and application.
///
/// Parameters
/// ----------
/// drawing : DrawingEuclidean2d
///     A drawing containing the positions of the nodes.
/// size_fn : callable
///     A function that takes a node and a dimension index, and returns the size of the node in that dimension.
///
/// Examples
/// --------
/// >>> import egraph as eg
/// >>> # Create a graph with 2 nodes
/// >>> graph = eg.Graph()
/// >>> n1 = graph.add_node(None)
/// >>> n2 = graph.add_node(None)
/// >>> # Create a drawing with overlapping nodes
/// >>> drawing = eg.DrawingEuclidean2d.initial_placement(graph)
/// >>> drawing.set_x(n1, 0.0)
/// >>> drawing.set_y(n1, 0.0)
/// >>> drawing.set_x(n2, 0.5)
/// >>> drawing.set_y(n2, 0.5)  # Overlapping with n1 if size is 1.0
/// >>> # Apply constraints to remove overlaps
/// >>> eg.project_rectangle_no_overlap_constraints_2d(
/// ...     drawing,
/// ...     lambda node, dim: 1.0,  # Each node has size 1.0 in both dimensions
/// ... )
/// >>> # The nodes should no longer overlap
#[pyfunction]
#[pyo3(name = "project_rectangle_no_overlap_constraints_2d")]
fn py_project_rectangle_no_overlap_constraints_2d(
    drawing: &mut PyDrawingEuclidean2d,
    size_fn: &Bound<PyAny>,
) -> PyResult<()> {
    project_rectangle_no_overlap_constraints_2d(drawing.drawing_mut(), |node_id, dim| {
        size_fn
            .call1((node_id.index(), dim))
            .unwrap()
            .extract()
            .unwrap()
    });

    Ok(())
}

/// Generates layered constraints for a directed graph based on the Sugiyama Framework.
///
/// This function performs cycle removal and layer assignment using the LongestPath algorithm,
/// then generates separation constraints for edges that span multiple layers.
///
/// Parameters
/// ----------
/// graph : Graph
///     A directed graph.
/// gap : float
///     The minimum distance between adjacent layers.
///
/// Returns
/// -------
/// list of Constraint
///     A list of separation constraints for the layered layout.
///
/// Examples
/// --------
/// >>> import egraph as eg
/// >>> # Create a directed graph with 3 nodes
/// >>> graph = eg.DiGraph()
/// >>> n1 = graph.add_node(None)
/// >>> n2 = graph.add_node(None)
/// >>> n3 = graph.add_node(None)
/// >>> # Add edges to form a path: n1 -> n2 -> n3
/// >>> graph.add_edge(n1, n2, None)
/// >>> graph.add_edge(n2, n3, None)
/// >>> # Generate constraints for layered layout
/// >>> constraints = eg.generate_layered_constraints(graph, 2.0)
/// >>> # Create a drawing to test constraint application
/// >>> drawing = eg.DrawingEuclidean2d.initial_placement(graph)
/// >>> for i in range(3):
/// ...     # Initial positions in a horizontal line
/// ...     drawing.set_x(i, i * 1.0)
/// ...     drawing.set_y(i, 0.0)
/// >>> # Apply constraints to the y-dimension (vertical)
/// >>> eg.project_1d(drawing, 1, constraints)
#[pyfunction]
#[pyo3(
    name = "generate_layered_constraints",
    signature = (graph, layers_or_gap = None, min_layer_distance = StrictF64(1.0), *, layers = None, gap = None)
)]
fn py_generate_layered_constraints<'py>(
    py: Python<'py>,
    graph: &Bound<'py, PyAny>,
    layers_or_gap: Option<&Bound<'py, PyAny>>,
    min_layer_distance: StrictF64,
    layers: Option<&Bound<'py, PyAny>>,
    gap: Option<StrictF64>,
) -> PyResult<Bound<'py, PyAny>> {
    if let Ok(csr) = graph.cast::<PyCsrGraph>() {
        if gap.is_some() {
            return Err(PyTypeError::new_err(
                "generate_layered_constraints(csr_graph, layers, min_layer_distance=1.0) does not accept 'gap'",
            ));
        }
        let layers_obj = match (layers, layers_or_gap) {
            (Some(_), Some(_)) => {
                return Err(PyTypeError::new_err(
                    "generate_layered_constraints got multiple values for 'layers'",
                ));
            }
            (Some(l), None) | (None, Some(l)) => l,
            (None, None) => {
                return Err(PyTypeError::new_err(
                    "generate_layered_constraints(csr_graph, layers, ...) requires 'layers'",
                ));
            }
        };
        let cs = v6_generate_layered_constraints_impl(
            py,
            &csr.borrow(),
            layers_obj,
            min_layer_distance,
        )?;
        return Ok(cs.into_pyobject(py)?.into_any());
    }

    if let Ok(adapter) = graph.cast::<PyGraphAdapter>() {
        if layers.is_some() {
            return Err(PyTypeError::new_err(
                "generate_layered_constraints(digraph, gap) does not accept 'layers'",
            ));
        }
        let gap_val = match (gap, layers_or_gap) {
            (Some(_), Some(_)) => {
                return Err(PyTypeError::new_err(
                    "generate_layered_constraints got multiple values for 'gap'",
                ));
            }
            (Some(g), None) => g.0,
            (None, Some(obj)) => obj.extract::<StrictF64>()?.0,
            (None, None) => {
                return Err(PyTypeError::new_err(
                    "generate_layered_constraints(digraph, gap) requires 'gap'",
                ));
            }
        };
        let borrowed = adapter.borrow();
        return match borrowed.graph() {
            GraphType::DiGraph(g) => {
                let rust_constraints = generate_layered_constraints(g, gap_val);
                let py_constraints: Vec<PyConstraint> = rust_constraints
                    .into_iter()
                    .map(|c| PyConstraint { constraint: c })
                    .collect();
                Ok(py_constraints.into_pyobject(py)?.into_any())
            }
            _ => Err(PyValueError::new_err(
                "generate_layered_constraints requires a directed graph",
            )),
        };
    }

    Err(PyTypeError::new_err(
        "generate_layered_constraints: graph must be CsrGraph or DiGraph",
    ))
}

/// Removes overlaps between rectangular regions that represent clusters of nodes.
///
/// This function takes a graph, a drawing, a function to get cluster IDs for nodes,
/// and a function to get node sizes. It then creates a cluster graph, determines
/// the size of each cluster, and applies separation constraints to prevent cluster overlaps.
///
/// Parameters
/// ----------
/// graph : Graph
///     The input graph.
/// drawing : DrawingEuclidean2d
///     The drawing to modify.
/// cluster_id_fn : callable
///     A function that takes a node and returns its cluster ID.
/// size_fn : callable
///     A function that takes a node and a dimension, and returns the node's size in that dimension.
///
/// Examples
/// --------
/// >>> import egraph as eg
/// >>> # Create a graph with 4 nodes in 2 clusters
/// >>> graph = eg.Graph()
/// >>> n1 = graph.add_node(None)
/// >>> n2 = graph.add_node(None)
/// >>> n3 = graph.add_node(None)
/// >>> n4 = graph.add_node(None)
/// >>> # Create a drawing
/// >>> drawing = eg.DrawingEuclidean2d.initial_placement(graph)
/// >>> # Position nodes in two clusters
/// >>> drawing.set_x(n1, 0.0)
/// >>> drawing.set_y(n1, 0.0)
/// >>> drawing.set_x(n2, 1.0)
/// >>> drawing.set_y(n2, 1.0)
/// >>> drawing.set_x(n3, 3.0)  # Cluster 2 starts close to cluster 1
/// >>> drawing.set_y(n3, 1.0)
/// >>> drawing.set_x(n4, 4.0)
/// >>> drawing.set_y(n4, 1.0)
/// >>> # Define a cluster ID function
/// >>> def get_cluster(node):
/// ...     # Nodes 0,1 in cluster 0; nodes 2,3 in cluster 1
/// ...     return 0 if node < 2 else 1
/// >>> # Apply cluster-based separation
/// >>> eg.project_clustered_rectangle_no_overlap_constraints(
/// ...     graph,
/// ...     drawing,
/// ...     get_cluster,
/// ...     lambda node, dim: 1.0  # Size function
/// ... )
#[pyfunction]
#[pyo3(name = "project_clustered_rectangle_no_overlap_constraints")]
fn py_project_clustered_rectangle_no_overlap_constraints(
    graph: &PyGraphAdapter,
    drawing: &mut PyDrawingEuclidean2d,
    cluster_id_fn: &Bound<PyAny>,
    size_fn: &Bound<PyAny>,
) -> PyResult<()> {
    match graph.graph() {
        GraphType::Graph(graph) => project_clustered_rectangle_no_overlap_constraints(
            graph,
            drawing.drawing_mut(),
            |node_id| {
                cluster_id_fn
                    .call1((node_id.index(),))
                    .unwrap()
                    .extract()
                    .unwrap()
            },
            |node_id, dim| {
                size_fn
                    .call1((node_id.index(), dim))
                    .unwrap()
                    .extract()
                    .unwrap()
            },
        ),
        _ => unimplemented!(),
    }

    Ok(())
}

/// Register the separation constraints module with Python
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyConstraint>()?;
    m.add_function(wrap_pyfunction!(py_project_1d, m)?)?;
    m.add_function(wrap_pyfunction!(
        py_project_rectangle_no_overlap_constraints_2d,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(py_generate_layered_constraints, m)?)?;
    m.add_function(wrap_pyfunction!(
        py_project_clustered_rectangle_no_overlap_constraints,
        m
    )?)?;
    Ok(())
}
