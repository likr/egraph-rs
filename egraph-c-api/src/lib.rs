extern crate clp;
extern crate egraph_force_directed;

use egraph_force_directed::center_force::CenterForce;
use egraph_force_directed::force::{Force, Link, Point};
use egraph_force_directed::link_force::LinkForce;
use egraph_force_directed::many_body_force::ManyBodyForce;
use egraph_force_directed::simulation::start_simulation;
use std::os::raw::{c_double, c_uint};

#[no_mangle]
pub fn hoge() {
    let mut model = clp::Model::new();
    model.resize(3, 3);
    println!("{} {}", model.number_rows(), model.number_columns());
}

#[no_mangle]
pub fn force_directed(
    num_vertices: c_uint,
    num_edges: c_uint,
    edges: *mut c_uint,
    result: *mut c_double,
) {
    let num_vertices = num_vertices as usize;
    let num_edges = num_edges as usize;
    let edges = unsafe { Vec::from_raw_parts(edges, num_edges * 2, num_edges * 2) };
    let mut points = (0..num_vertices)
        .map(|i| {
            let r = 10. * (i as usize as f32).sqrt();
            let theta = std::f32::consts::PI * (3. - (5. as f32).sqrt()) * (i as usize as f32);
            let x = r * theta.cos();
            let y = r * theta.sin();
            Point::new(x, y)
        })
        .collect::<Vec<_>>();
    let links = (0..num_edges)
        .map(|i| {
            let source = edges[2 * i] as usize;
            let target = edges[2 * i + 1] as usize;
            Link::new(source, target)
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
    let offset = 4; // XXX monkey patch. this value should be larger than 3(?)
    let mut result = unsafe { Vec::from_raw_parts(result, offset + num_vertices * 2, offset + num_vertices * 2) };
    for (i, point) in points.iter().enumerate() {
        result[offset + 2 * i] = point.x as f64;
        result[offset + 2 * i + 1] = point.y as f64;
    }
}
