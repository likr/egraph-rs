extern crate petgraph;

use std::os::raw::{c_double, c_uint};
use petgraph::graph::NodeIndex;

#[derive(Default)]
pub struct Node {
    pub x: c_double,
    pub y: c_double,
}

pub type Graph = petgraph::Graph<Node, (), petgraph::Undirected>;

#[no_mangle]
pub unsafe fn graph_new() -> *mut Graph {
    let graph = Box::new(Graph::new_undirected());
    Box::into_raw(graph)
}

#[no_mangle]
pub unsafe fn graph_add_node(p_graph: *mut Graph) -> c_uint {
    (*p_graph).add_node(Node::default()).index() as c_uint
}

#[no_mangle]
pub unsafe fn graph_add_edge(p_graph: *mut Graph, u: c_uint, v: c_uint) -> c_uint {
    (*p_graph).add_edge(NodeIndex::new(u as usize), NodeIndex::new(v as usize), ()).index() as c_uint
}

#[no_mangle]
pub unsafe fn graph_node_count(p_graph: *mut Graph) -> c_uint {
    (*p_graph).node_count() as c_uint
}

#[no_mangle]
pub unsafe fn graph_edge_count(p_graph: *mut Graph) -> c_uint {
    (*p_graph).edge_count() as c_uint
}

#[no_mangle]
pub unsafe fn graph_get_x(p_graph: *mut Graph, u: c_uint) -> c_double {
    (*p_graph).raw_nodes()[u as usize].weight.x
}

#[no_mangle]
pub unsafe fn graph_get_y(p_graph: *mut Graph, u: c_uint) -> c_double {
    (*p_graph).raw_nodes()[u as usize].weight.y
}

#[no_mangle]
pub unsafe fn graph_set_x(p_graph: *mut Graph, u: c_uint, value: c_double) {
    let weight = (*p_graph).node_weight_mut(NodeIndex::new(u as usize)).unwrap();
    weight.x = value;
}

#[no_mangle]
pub unsafe fn graph_set_y(p_graph: *mut Graph, u: c_uint, value: c_double) {
    let weight = (*p_graph).node_weight_mut(NodeIndex::new(u as usize)).unwrap();
    weight.y = value;
}
