extern crate egraph_force_directed;

use std::f32::consts::PI;
use std::os::raw::c_double;
use egraph_force_directed::center_force::CenterForce;
use egraph_force_directed::force::{Force, Link, Point};
use egraph_force_directed::group_force::{GroupForce};
use egraph_force_directed::link_force::LinkForce;
use egraph_force_directed::many_body_force::ManyBodyForce;
use egraph_force_directed::simulation::start_simulation;
use graph::Graph;

pub struct Simulation {
    forces: Vec<Box<Force>>,
}

pub struct Group {
    pub x: c_double,
    pub y: c_double,
}

impl Simulation {
    fn new() -> Simulation {
        Simulation {
            forces: Vec::new(),
        }
    }
}

#[no_mangle]
pub unsafe fn simulation_new() -> *mut Simulation {
    let simulation = Box::new(Simulation::new());
    Box::into_raw(simulation)
}

#[no_mangle]
pub unsafe fn simulation_add_center_force(p_simulation: *mut Simulation) {
    (*p_simulation).forces.push(Box::new(CenterForce::new()));
}

#[no_mangle]
pub unsafe fn simulation_add_group_force(p_simulation: *mut Simulation, num_groups: usize, p_groups: *mut Group, num_nodes: usize, p_node_groups: *mut usize) {
    let groups = Vec::from_raw_parts(p_groups, num_groups, num_groups);
    let groups = groups.iter()
        .map(|group| egraph_force_directed::group_force::Group::new(group.x as f32, group.y as f32))
        .collect::<Vec<_>>();
    let node_groups = Vec::from_raw_parts(p_node_groups, num_nodes, num_nodes);
    (*p_simulation).forces.push(Box::new(GroupForce::new(groups, node_groups)));
}

#[no_mangle]
pub unsafe fn simulation_add_link_force(p_simulation: *mut Simulation, p_graph: *mut Graph) {
    let links = (*p_graph).edge_indices()
        .map(|edge| {
            let (source, target) = (*p_graph).edge_endpoints(edge).unwrap();
            Link::new(source.index(), target.index())
        })
        .collect::<Vec<_>>();
    (*p_simulation).forces.push(Box::new(LinkForce::new(&links)));
}

#[no_mangle]
pub unsafe fn simulation_add_many_body_force(p_simulation: *mut Simulation) {
    (*p_simulation).forces.push(Box::new(ManyBodyForce::new()));
}

#[no_mangle]
pub unsafe fn simulation_start(p_simulation: *mut Simulation, p_graph: *mut Graph) {
    let mut points = (*p_graph).node_indices()
        .map(|node| {
            let i = node.index();
            let r = 10. * (i as usize as f32).sqrt();
            let theta = PI * (3. - (5. as f32).sqrt()) * (i as usize as f32);
            let x = r * theta.cos();
            let y = r * theta.sin();
            Point::new(x, y)
        })
        .collect::<Vec<_>>();
    start_simulation(&mut points, &(*p_simulation).forces);
    for (node, point) in (*p_graph).node_indices().zip(points) {
        let mut node = (*p_graph).node_weight_mut(node).unwrap();
        node.x = point.x as f64;
        node.y = point.y as f64;
    }
}
