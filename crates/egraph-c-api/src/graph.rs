extern crate petgraph;

use petgraph::prelude::*;
use std::os::raw::{c_double, c_int, c_uint};
use std::ptr;

#[derive(Default, Clone)]
pub struct Node {
    pub x: c_double,
    pub y: c_double,
}

impl Node {
    pub fn new(x: c_double, y: c_double) -> Node {
        Node { x, y }
    }

    pub fn empty() -> Node {
        Node { x: 0., y: 0. }
    }
}

#[derive(Default, Clone)]
pub struct Edge {}

impl Edge {
    pub fn new() -> Edge {
        Edge {}
    }
}

pub type Ix = c_uint;
pub type Graph = petgraph::Graph<Node, Edge, Undirected, Ix>;

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
    (*p_graph)
        .add_edge(
            NodeIndex::new(u as usize),
            NodeIndex::new(v as usize),
            Edge::new(),
        )
        .index() as c_uint
}

#[no_mangle]
pub unsafe fn graph_remove_node(p_graph: *mut Graph, index: c_uint) {
    (*p_graph).remove_node(NodeIndex::new(index as usize));
}

#[no_mangle]
pub unsafe fn graph_remove_edge(p_graph: *mut Graph, index: c_uint) {
    (*p_graph).remove_edge(EdgeIndex::new(index as usize));
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
pub unsafe fn graph_node_at(p_graph: *mut Graph, i: c_uint) -> *const petgraph::graph::Node<Node> {
    &(*p_graph).raw_nodes()[i as usize]
}

#[no_mangle]
pub unsafe fn graph_edge_at(p_graph: *mut Graph, i: c_uint) -> *const petgraph::graph::Edge<Edge> {
    &(*p_graph).raw_edges()[i as usize]
}

#[no_mangle]
pub unsafe fn graph_node_indices(p_graph: *const Graph) -> *const petgraph::graph::NodeIndices<Ix> {
    let indices = Box::new((*p_graph).node_indices());
    Box::into_raw(indices)
}

#[no_mangle]
pub unsafe fn graph_edge_indices(p_graph: *const Graph) -> *const petgraph::graph::EdgeIndices<Ix> {
    let indices = Box::new((*p_graph).edge_indices());
    Box::into_raw(indices)
}

#[no_mangle]
pub unsafe fn graph_neighbors(
    p_graph: *mut Graph,
    u: c_uint,
) -> *mut petgraph::graph::WalkNeighbors<Ix> {
    let neighbors = Box::new((*p_graph).neighbors(NodeIndex::new(u as usize)).detach());
    Box::into_raw(neighbors)
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
    let weight = (*p_graph)
        .node_weight_mut(NodeIndex::new(u as usize))
        .unwrap();
    weight.x = value;
}

#[no_mangle]
pub unsafe fn graph_set_y(p_graph: *mut Graph, u: c_uint, value: c_double) {
    let weight = (*p_graph)
        .node_weight_mut(NodeIndex::new(u as usize))
        .unwrap();
    weight.y = value;
}

#[no_mangle]
pub unsafe fn graph_degree(p_graph: *const Graph, u: c_uint) -> c_uint {
    (*p_graph).neighbors(NodeIndex::new(u as usize)).count() as c_uint
}

#[no_mangle]
pub unsafe fn graph_source(p_graph: *mut Graph, i: c_uint) -> c_uint {
    (*p_graph).raw_edges()[i as usize].source().index() as c_uint
}

#[no_mangle]
pub unsafe fn graph_target(p_graph: *mut Graph, i: c_uint) -> c_uint {
    (*p_graph).raw_edges()[i as usize].target().index() as c_uint
}

#[no_mangle]
pub unsafe fn graph_filter(
    p_graph: *const Graph,
    node_map: extern "C" fn(c_uint) -> c_int,
    edge_map: extern "C" fn(c_uint) -> c_int,
) -> *mut Graph {
    let graph = Box::new((*p_graph).filter_map(
        |index, node| {
            if node_map(index.index() as c_uint) == 0 {
                None
            } else {
                Some(node.clone())
            }
        },
        |index, edge| {
            if edge_map(index.index() as c_uint) == 0 {
                None
            } else {
                Some(edge.clone())
            }
        },
    ));
    Box::into_raw(graph)
}

#[no_mangle]
pub unsafe fn node_get_x(p_node: *mut petgraph::graph::Node<Node>) -> c_double {
    (*p_node).weight.x
}

#[no_mangle]
pub unsafe fn node_get_y(p_node: *mut petgraph::graph::Node<Node>) -> c_double {
    (*p_node).weight.y
}

#[no_mangle]
pub unsafe fn node_set_x(p_node: *mut petgraph::graph::Node<Node>, value: c_double) {
    (*p_node).weight.x = value;
}

#[no_mangle]
pub unsafe fn node_set_y(p_node: *mut petgraph::graph::Node<Node>, value: c_double) {
    (*p_node).weight.y = value;
}

#[no_mangle]
pub unsafe fn edge_source(p_edge: *mut petgraph::graph::Edge<Edge>) -> c_uint {
    (*p_edge).source().index() as c_uint
}

#[no_mangle]
pub unsafe fn edge_target(p_edge: *mut petgraph::graph::Edge<Edge>) -> c_uint {
    (*p_edge).target().index() as c_uint
}

#[no_mangle]
pub unsafe fn node_indices_next(
    p_node_indices: *mut petgraph::graph::NodeIndices<Ix>,
) -> *mut NodeIndex<Ix> {
    match (*p_node_indices).next() {
        Some(index) => Box::into_raw(Box::new(index)),
        None => ptr::null_mut(),
    }
}

#[no_mangle]
pub unsafe fn edge_indices_next(
    p_edge_indices: *mut petgraph::graph::EdgeIndices<Ix>,
) -> *mut EdgeIndex<Ix> {
    match (*p_edge_indices).next() {
        Some(index) => Box::into_raw(Box::new(index)),
        None => ptr::null_mut(),
    }
}

#[no_mangle]
pub unsafe fn neighbors_next_node(
    p_neighbors: *mut petgraph::graph::WalkNeighbors<Ix>,
    p_graph: *const Graph,
) -> *mut NodeIndex<Ix> {
    match (*p_neighbors).next_node(&*p_graph) {
        Some(index) => Box::into_raw(Box::new(index)),
        None => ptr::null_mut(),
    }
}

#[no_mangle]
pub unsafe fn neighbors_next_edge(
    p_neighbors: *mut petgraph::graph::WalkNeighbors<Ix>,
    p_graph: *const Graph,
) -> *mut EdgeIndex<Ix> {
    match (*p_neighbors).next_edge(&*p_graph) {
        Some(index) => Box::into_raw(Box::new(index)),
        None => ptr::null_mut(),
    }
}

#[no_mangle]
pub unsafe fn node_index_index(p_node_index: *const NodeIndex<Ix>) -> c_uint {
    (*p_node_index).index() as c_uint
}

#[no_mangle]
pub unsafe fn edge_index_index(p_edge_index: *const EdgeIndex<Ix>) -> c_uint {
    (*p_edge_index).index() as c_uint
}
