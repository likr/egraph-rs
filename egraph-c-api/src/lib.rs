extern crate clp;
extern crate egraph;
extern crate egraph_force_directed;
extern crate petgraph;

use std::os::raw::{c_double, c_uint};
use petgraph::graph::NodeIndex;
use egraph_force_directed::center_force::CenterForce;
use egraph_force_directed::force::{Force, Link, Point};
use egraph_force_directed::link_force::LinkForce;
use egraph_force_directed::many_body_force::ManyBodyForce;
use egraph_force_directed::simulation::start_simulation;

#[derive(Default)]
pub struct Node {
    x: c_double,
    y: c_double,
}

type Graph = petgraph::Graph<Node, (), petgraph::Undirected>;

#[no_mangle]
pub fn hoge() {
    let mut model = clp::Model::new();
    model.resize(3, 3);
    println!("{} {}", model.number_rows(), model.number_columns());
}

#[no_mangle]
pub unsafe fn force_directed(
    p_graph: *mut Graph
) {
    let mut points = (*p_graph).node_indices()
        .map(|node| {
            let i = node.index();
            let r = 10. * (i as usize as f32).sqrt();
            let theta = std::f32::consts::PI * (3. - (5. as f32).sqrt()) * (i as usize as f32);
            let x = r * theta.cos();
            let y = r * theta.sin();
            Point::new(x, y)
        })
        .collect::<Vec<_>>();
    let links = (*p_graph).edge_indices()
        .map(|edge| {
            let (source, target) = (*p_graph).edge_endpoints(edge).unwrap();
            Link::new(source.index(), target.index())
        })
        .collect::<Vec<_>>();
    let forces = {
        let mut forces: Vec<Box<Force>> = Vec::new();
        forces.push(Box::new(ManyBodyForce::new()));
        forces.push(Box::new(LinkForce::new(&links)));
        forces.push(Box::new(CenterForce::new()));
        forces
    };
    start_simulation(&mut points, &forces);
    for (node, point) in (*p_graph).node_indices().zip(points) {
        let mut node = (*p_graph).node_weight_mut(node).unwrap();
        node.x = point.x as f64;
        node.y = point.y as f64;
    }
}

#[no_mangle]
pub fn graph_new() -> *mut Graph {
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
pub unsafe fn connected_components(p_graph: *mut Graph) -> *mut c_uint {
    egraph::algorithms::connected_components(&*p_graph)
        .iter()
        .map(|&c| c as c_uint)
        .collect::<Vec<_>>()
        .as_mut_ptr()
}

#[no_mangle]
pub unsafe fn rust_free(pointer: *mut c_uint) {
    let _b = Box::from_raw(pointer);
}
