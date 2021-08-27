#[macro_use]
extern crate force_derive;
use petgraph::visit::{IntoNeighbors, IntoNodeIdentifiers};
use petgraph_layout_force_simulation::{Force, ForceToNode, Point};
use std::collections::HashMap;
use std::hash::Hash;

#[derive(Force)]
pub struct FruchtermanReingoldForce {
    links: Vec<Vec<usize>>,
    k: f32,
    min_distance: f32,
}

impl FruchtermanReingoldForce {
    pub fn new<G>(graph: G, k: f32, min_distance: f32) -> FruchtermanReingoldForce
    where
        G: IntoNodeIdentifiers + IntoNeighbors,
        G::NodeId: Eq + Hash,
    {
        let node_indices = graph
            .node_identifiers()
            .enumerate()
            .map(|(i, u)| (u, i))
            .collect::<HashMap<_, _>>();
        let mut links = vec![vec![]; node_indices.len()];
        for u in graph.node_identifiers() {
            for v in graph.neighbors(u) {
                links[node_indices[&u]].push(node_indices[&v]);
            }
        }
        FruchtermanReingoldForce {
            links,
            k,
            min_distance,
        }
    }
}

impl ForceToNode for FruchtermanReingoldForce {
    fn apply_to_node(&self, u: usize, points: &mut [Point], alpha: f32) {
        let n = points.len();
        let k = self.k;
        for v in 0..n {
            if u == v {
                continue;
            }
            let dx = points[v].x - points[u].x;
            let dy = points[v].y - points[u].y;
            let d = (dx * dx + dy * dy).sqrt().max(self.min_distance);
            let kd = k / d;
            points[u].vx -= alpha * kd * kd * dx;
            points[u].vy -= alpha * kd * kd * dy;
        }
        for &v in self.links[u].iter() {
            let dx = points[v].x - points[u].x;
            let dy = points[v].y - points[u].y;
            let d = (dx * dx + dy * dy).sqrt();
            points[u].vx += alpha * d / k * dx;
            points[u].vy += alpha * d / k * dy;
        }
    }
}
