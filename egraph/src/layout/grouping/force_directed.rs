use std::f64::consts::PI;
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
    pub link_length: f64,
    pub many_body_force_strength: f64,
    pub link_force_strength: f64,
    pub center_force_strength: f64,
}

impl ForceDirectedGrouping {
    pub fn new<N, E, Ty: EdgeType, Ix: IndexType>(graph: &Graph<N, E, Ty, Ix>) -> ForceDirectedGrouping {
        let links = initial_links(&graph);
        ForceDirectedGrouping {
            links,
            link_length: 30.,
            many_body_force_strength: 1.,
            link_force_strength: 1.,
            center_force_strength: 1.,
        }
    }
}

impl Grouping for ForceDirectedGrouping {
    fn call(&self, width: f64, height: f64, values: &Vec<f64>) -> Vec<Group> {
        let links = self.links.iter()
            .map(|link| {
                Link {
                    source: link.source,
                    target: link.target,
                    length: self.link_length as f32,
                    strength: link.strength,
                    bias: link.bias
                }
            })
            .collect();
        let mut forces : Vec<Box<Force>> = Vec::new();
        forces.push(Box::new(ManyBodyForce::new()));
        forces.push(Box::new(LinkForce::new_with_links(links)));
        forces.push(Box::new(CenterForce::new()));
        forces[0].set_strength(self.many_body_force_strength as f32);
        forces[1].set_strength(self.link_force_strength as f32);
        forces[2].set_strength(self.center_force_strength as f32);
        let mut points = initial_placement(values.len());
        start_simulation(&mut points, &forces);

        let total_value = values.iter().fold(0.0, |s, v| s + v);

        values.iter()
            .zip(points)
            .map(|(&value, point)| {
                let x = point.x as f64;
                let y = point.y as f64;
                let size = (width * height * value / total_value / PI).sqrt() * 2.;
                Group::new(x, y, size, size)
            })
        .collect()
    }
}
