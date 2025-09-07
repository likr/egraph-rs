use crate::drawing::PyDrawingEuclidean2d;
use crate::graph::PyGraphAdapter;
use petgraph_algorithm_triangulation;
use pyo3::prelude::*;

/// Performs Delaunay triangulation based on node positions in a 2D Euclidean drawing.
///
/// This function takes a drawing as input, extracts the node positions from the drawing,
/// computes the Delaunay triangulation of these points, and returns a new
/// graph with nodes corresponding to the drawing's nodes and edges representing the triangulation.
///
/// Parameters
/// ----------
/// drawing : DrawingEuclidean2d
///     A 2D Euclidean drawing that contains the positions of the nodes.
///
/// Returns
/// -------
/// Graph
///     A new undirected graph with nodes corresponding to the drawing's nodes,
///     and with edges representing the Delaunay triangulation.
///
/// Examples
/// --------
/// >>> import egraph as eg
/// >>> # Create a graph
/// >>> graph = eg.Graph()
/// >>> n1 = graph.add_node()
/// >>> n2 = graph.add_node()
/// >>> n3 = graph.add_node()
/// >>> n4 = graph.add_node()
/// >>> # Create a drawing
/// >>> drawing = eg.DrawingEuclidean2d(graph)
/// >>> drawing.set_position(n1, 0.0, 0.0)
/// >>> drawing.set_position(n2, 1.0, 0.0)
/// >>> drawing.set_position(n3, 0.0, 1.0)
/// >>> drawing.set_position(n4, 1.0, 1.0)
/// >>> # Compute the Delaunay triangulation
/// >>> triangulated_graph = eg.triangulation(drawing)
/// >>> # The triangulated graph should have 4 nodes and 5 edges
/// >>> triangulated_graph.number_of_nodes()
/// 4
/// >>> triangulated_graph.number_of_edges()
/// 5
#[pyfunction]
#[pyo3(name = "triangulation")]
pub fn py_triangulation(
    _py: Python<'_>,
    drawing: &PyDrawingEuclidean2d,
) -> PyResult<PyGraphAdapter> {
    let drawing_ref = drawing.drawing();
    let triangulated = petgraph_algorithm_triangulation::triangulation(drawing_ref).map(
        |_, _| Python::attach(|py| py.None()),
        |_, _| Python::attach(|py| py.None()),
    );

    // Create a new PyGraphAdapter from the triangulated graph
    Ok(PyGraphAdapter::new(triangulated))
}

/// Register the triangulation module with Python
pub fn register(_py: Python<'_>, m: &Bound<PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(py_triangulation, m)?)?;
    Ok(())
}
