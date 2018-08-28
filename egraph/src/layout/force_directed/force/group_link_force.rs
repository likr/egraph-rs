extern crate petgraph;

use std::collections::HashMap;
use petgraph::{Graph, EdgeType};
use petgraph::graph::IndexType;
use super::force::{Force, Link, Point};

pub struct GroupLinkForce {
    pub links: Vec<Link>,
    pub node_groups: Vec<usize>,
    pub strength: f32,
}

impl GroupLinkForce {
    pub fn new<N, E, Ty: EdgeType, Ix: IndexType>(graph: &Graph<N, E, Ty, Ix>, node_groups: &Vec<usize>) -> GroupLinkForce {
        GroupLinkForce::new_with_strength(&graph, &node_groups, 0.5, 0.01)
    }

    pub fn new_with_strength<N, E, Ty: EdgeType, Ix: IndexType>(graph: &Graph<N, E, Ty, Ix>, node_groups: &Vec<usize>, intra_group: f32, inter_group: f32) -> GroupLinkForce {
        let links = graph.edge_indices()
            .map(|edge| {
                let (source, target) = graph.edge_endpoints(edge).unwrap();
                let mut link = Link::new(source.index(), target.index());
                if node_groups[link.source] == node_groups[link.target] {
                    link.strength = intra_group;
                } else {
                    link.strength = inter_group;
                }
                link
            })
            .collect::<Vec<_>>();
        GroupLinkForce {
            links: links,
            node_groups: node_groups.clone().to_vec(),
            strength: 0.1,
        }
    }
}

impl Force for GroupLinkForce {
    fn apply(&self, points: &mut Vec<Point>, alpha: f32) {
        let links = &self.links;
        let mut count: HashMap<usize, usize> = HashMap::new();
        for link in links.iter() {
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
        let bias = links
            .iter()
            .map(|link| {
                let source_count = *count.get(&link.source).unwrap();
                let target_count = *count.get(&link.target).unwrap();
                source_count as f32 / (source_count + target_count) as f32
            })
            .collect::<Vec<f32>>();
        for (link, b) in links.iter().zip(bias.iter()) {
            let source = points[link.source];
            let target = points[link.target];
            let source_count = count.get(&link.source).unwrap();
            let target_count = count.get(&link.target).unwrap();
            let dx = (target.x + self.strength * target.vx) - (source.x + self.strength * source.vx);
            let dy = (target.y + self.strength * target.vy) - (source.y + self.strength * source.vy);
            let l = (dx * dx + dy * dy).sqrt().max(1e-6);
            let strength = link.strength / *source_count.min(target_count) as f32;
            let w = (l - link.length) / l * alpha * strength;
            {
                let ref mut target = points[link.target];
                target.vx -= dx * w * b;
                target.vy -= dy * w * b;
            }
            {
                let ref mut source = points[link.source];
                source.vx += dx * w * (1. - b);
                source.vy += dy * w * (1. - b);
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
