use petgraph::{Graph, EdgeType};
use petgraph::graph::IndexType;
use ::layout::force_directed::{initial_placement, initial_links};
use ::layout::force_directed::simulation::start_simulation;
use ::layout::force_directed::force::{
    Link,
    Force,
    CenterForce,
    LinkForce,
    ManyBodyForce,
};
use super::{Group, Grouping};

pub struct ForceDirectedGrouping {
    links: Vec<Link>,
}

impl ForceDirectedGrouping {
    pub fn new<N, E, Ty: EdgeType, Ix: IndexType>(graph: &Graph<N, E, Ty, Ix>) -> ForceDirectedGrouping {
        let links = initial_links(&graph);
        ForceDirectedGrouping {
            links,
        }
    }
}

impl Grouping for ForceDirectedGrouping {
    fn call(&self, width: f64, height: f64, values: &Vec<f64>) -> Vec<Group> {
        force_directed_grouping(width, height, &values, &self.links)
    }
}

pub fn force_directed_grouping(width: f64, height: f64, values: &Vec<f64>, links: &Vec<Link>) -> Vec<Group> {
    let mut forces : Vec<Box<Force>> = Vec::new();
    forces.push(Box::new(ManyBodyForce::new()));
    forces.push(Box::new(CenterForce::new()));
    forces.push(Box::new(LinkForce::new_with_links(links.to_vec())));
    let mut points = initial_placement(values.len());
    start_simulation(&mut points, &forces);

    let left = points.iter().fold(0. / 0., |m : f64, p| m.min(p.x as f64));
    let right = points.iter().fold(0. / 0., |m : f64, p| m.max(p.x as f64));
    let top = points.iter().fold(0. / 0., |m : f64, p| m.min(p.y as f64));
    let bottom = points.iter().fold(0. / 0., |m : f64, p| m.max(p.y as f64));
    let layout_width = right - left;
    let layout_height = bottom - top;
    let horizontal_scale = width / layout_width;
    let vertical_scale = height / layout_height;
    let scale = horizontal_scale.min(vertical_scale);

    values.iter()
        .zip(points)
        .map(|(&value, point)| {
            let x = scale * (point.x as f64 - left);
            let y = scale * (point.y as f64 - top);
            let size = scale * value.sqrt() * 5.0;
            Group::new(x, y, size, size)
        })
        .collect()
}
