use std::os::raw::c_uint;
use egraph::algorithms;
use egraph::algorithms::biclustering::{QuasiBiclique};
use graph::{Graph, Node, Edge};
use biclustering::Biclusters;
use super::copy_to_vec;

#[no_mangle]
pub unsafe fn edge_concentration(p_graph: *mut Graph, p_biclusters: *mut Biclusters) -> *mut Graph {
    let transformed = Box::new(algorithms::edge_concentration(&*p_graph, &*p_biclusters,
                                                              |_u| Node::empty(),
                                                              |_u| Edge::new(),
                                                              |_bicluster| Node::empty()));
    Box::into_raw(transformed)
}

#[no_mangle]
pub unsafe fn inter_group_edge_concentration_with_quasi_biclique(
    p_graph: *mut Graph,
    p_node_groups: *mut c_uint,
    p_biclustering: *mut QuasiBiclique,
) -> *mut Graph {
    let node_groups = copy_to_vec(p_node_groups, (*p_graph).node_count());
    let transformed = Box::new(algorithms::inter_group_edge_concentration(&*p_graph, &node_groups, &*p_biclustering,
                                                                          |_u| Node::empty(),
                                                                          |_u, _v| Edge::new(),
                                                                          |_u| Edge::new(),
                                                                          |_bicluster| Node::empty(),
                                                                          |_bicluster| Edge::new()));
    Box::into_raw(transformed)
}
