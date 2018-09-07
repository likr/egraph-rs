use egraph;
use graph::{Graph, Node};
use biclustering::Biclusters;

#[no_mangle]
pub unsafe fn edge_concentration(p_graph: *mut Graph, p_biclusters: *mut Biclusters) -> *mut Graph {
    let transformed = Box::new(egraph::algorithms::edge_concentration(&*p_graph, &*p_biclusters,
                                                                              |_u| Node::empty(),
                                                                              |_u| (),
                                                                              |_bicluster| Node::empty()));
    Box::into_raw(transformed)
}
