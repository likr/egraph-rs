extern crate petgraph;

use std::collections::HashMap;
use petgraph::{Graph, EdgeType};
use petgraph::graph::IndexType;
use super::force::{Force, Link, Point};

pub struct LinkForce {
    links: Vec<Link>,
    strength: f32,
}

impl LinkForce {
    pub fn new<N, E, Ty: EdgeType, Ix: IndexType>(graph: &Graph<N, E, Ty, Ix>) -> LinkForce {
        let mut links = graph.edge_indices()
            .map(|edge| {
                let (source, target) = graph.edge_endpoints(edge).unwrap();
                Link::new(source.index(), target.index())
            })
            .collect::<Vec<_>>();
        let mut count: HashMap<usize, usize> = HashMap::new();
        for link in &links {
            if !count.contains_key(&link.source) {
                count.insert(link.source, 0);
            }
            if !count.contains_key(&link.target) {
                count.insert(link.target, 0);
            }
            {
                let v = count.get_mut(&link.source).unwrap();
                *v += 1;
            }
            {
                let v = count.get_mut(&link.target).unwrap();
                *v += 1;
            }
        }
        for mut link in &mut links {
            let source_count = *count.get(&link.source).unwrap();
            let target_count = *count.get(&link.target).unwrap();
            link.strength = 1. / source_count.min(target_count) as f32;
            link.bias = source_count as f32 / (source_count + target_count) as f32
        }
        LinkForce {
            links: links,
            strength: 1.,
        }
    }

    pub fn new_with_links(links: Vec<Link>) -> LinkForce {
        LinkForce {
            links,
            strength: 1.,
        }
    }
}

impl Force for LinkForce {
    fn apply(&self, points: &mut Vec<Point>, alpha: f32) {
        let links = &self.links;
        for link in links {
            let source = points[link.source];
            let target = points[link.target];
            let dx = (target.x + self.strength * target.vx) - (source.x + self.strength * source.vx);
            let dy = (target.y + self.strength * target.vy) - (source.y + self.strength * source.vy);
            let l = (dx * dx + dy * dy).sqrt().max(1e-6);
            let w = (l - link.length) / l * alpha * link.strength;
            {
                let ref mut target = points[link.target];
                target.vx -= dx * w * link.bias;
                target.vy -= dy * w * link.bias;
            }
            {
                let ref mut source = points[link.source];
                source.vx += dx * w * (1. - link.bias);
                source.vy += dy * w * (1. - link.bias);
            }
        }
    }

    fn get_strength(&self) -> f32 {
        self.strength
    }

    fn set_strength(&mut self, strength: f32) {
        self.strength = strength;
    }
}

#[test]
fn test_link() {
    let mut links = Vec::new();
    links.push(Link::new(0, 1));
    links.push(Link::new(1, 3));
    links.push(Link::new(3, 2));
    links.push(Link::new(2, 0));
    let mut force = LinkForce::new(&links);
    force.strength = 0.0;
    let mut points = Vec::new();
    points.push(Point::new(10., 10.));
    points.push(Point::new(10., -10.));
    points.push(Point::new(-10., 10.));
    points.push(Point::new(-10., -10.));
    force.apply(&mut points, 1.);
    assert_eq!(points[0].vx, 2.5);
    assert_eq!(points[0].vy, 2.5);
    assert_eq!(points[1].vx, 2.5);
    assert_eq!(points[1].vy, -2.5);
    assert_eq!(points[2].vx, -2.5);
    assert_eq!(points[2].vy, 2.5);
    assert_eq!(points[3].vx, -2.5);
    assert_eq!(points[3].vy, -2.5);
}
