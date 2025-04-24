/// Python binding for graph coarsening
use crate::graph::{GraphType, PyGraphAdapter};
use petgraph_clustering::coarsen;
use pyo3::prelude::*;
use std::collections::HashMap;

/// Create a coarsened graph by grouping nodes based on communities.
///
/// This function creates a new graph where each node represents a community of nodes
/// from the original graph. Edges in the coarsened graph represent the connections
/// between communities in the original graph.
///
/// Args:
///     graph (Graph or DiGraph): The input graph to coarsen.
///     node_group_func (callable): A function that takes a node index and returns its group ID.
///     node_merge_func (callable, optional): A function that takes a list of node indices and
///         returns the node data for the new coarsened node. Defaults to counting the number
///         of nodes in each group.
///     edge_merge_func (callable, optional): A function that takes a list of edge indices and
///         returns the edge data for the new coarsened edge. Defaults to counting the number
///         of edges between groups.
///
/// Returns:
///     tuple: A tuple containing:
///         - The coarsened graph (same type as input)
///         - A dictionary mapping group IDs to node indices in the coarsened graph
///
/// Example:
///     >>> import egraph as eg
///     >>> graph = eg.Graph()
///     >>> # Add nodes and edges...
///     >>> louvain = eg.clustering.Louvain()
///     >>> communities = louvain.detect_communities(graph)
///     >>> coarsened_graph, node_map = eg.clustering.coarsen(
///     ...     graph,
///     ...     lambda node: communities[node],
///     ...     lambda nodes: len(nodes),
///     ...     lambda edges: len(edges)
///     ... )
#[pyfunction]
#[pyo3(signature = (graph, node_group_func, node_merge_func, edge_merge_func))]
pub fn py_coarsen(
    graph: &PyGraphAdapter,
    node_group_func: &Bound<PyAny>,
    node_merge_func: &Bound<PyAny>,
    edge_merge_func: &Bound<PyAny>,
) -> (PyGraphAdapter, HashMap<usize, usize>) {
    let (coarsened_graph, node_map) = match graph.graph() {
        GraphType::Graph(graph) => coarsen(
            &graph,
            &mut |_, u| {
                node_group_func
                    .call1((u.index(),))
                    .unwrap()
                    .extract()
                    .unwrap()
            },
            &mut |_, nodes| {
                node_merge_func
                    .call1((nodes.iter().map(|u| u.index()).collect::<Vec<_>>(),))
                    .unwrap()
                    .extract()
                    .unwrap()
            },
            &mut |_, edges| {
                edge_merge_func
                    .call1((edges.iter().map(|e| e.index()).collect::<Vec<_>>(),))
                    .unwrap()
                    .extract()
                    .unwrap()
            },
        ),
        _ => unimplemented!(),
    };

    (
        PyGraphAdapter::new(coarsened_graph),
        node_map
            .into_iter()
            .map(|(c, u)| (c, u.index()))
            .collect::<HashMap<_, _>>(),
    )
}
