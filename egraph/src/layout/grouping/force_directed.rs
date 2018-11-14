use super::{Group, Grouping};
use layout::force_directed::force::{CenterForce, Link, LinkForce, ManyBodyForce};
use layout::force_directed::simulation::Simulation;
use layout::force_directed::{initial_links, initial_placement};
use petgraph::graph::IndexType;
use petgraph::{EdgeType, Graph};
use std::f64::consts::PI;

pub struct ForceDirectedGrouping {
    links: Vec<Link>,
    pub link_length: f64,
    pub many_body_force_strength: f64,
    pub link_force_strength: f64,
    pub center_force_strength: f64,
}

impl ForceDirectedGrouping {
    pub fn new<N, E, Ty: EdgeType, Ix: IndexType>(
        graph: &Graph<N, E, Ty, Ix>,
    ) -> ForceDirectedGrouping {
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
        let links = self
            .links
            .iter()
            .map(|link| Link {
                source: link.source,
                target: link.target,
                length: self.link_length as f32,
                strength: link.strength,
                bias: link.bias,
            })
            .collect();
        let mut simulation = Simulation::new();
        simulation.forces.push(Box::new(ManyBodyForce::new()));
        simulation
            .forces
            .push(Box::new(LinkForce::new_with_links(links)));
        simulation.forces.push(Box::new(CenterForce::new()));
        simulation.forces[0].set_strength(self.many_body_force_strength as f32);
        simulation.forces[1].set_strength(self.link_force_strength as f32);
        simulation.forces[2].set_strength(self.center_force_strength as f32);
        let mut points = initial_placement(values.len());
        simulation.start(&mut points);

        let total_value = values.iter().fold(0.0, |s, v| s + v);

        values
            .iter()
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
