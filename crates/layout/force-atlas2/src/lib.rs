#[macro_use]
extern crate force_derive;
use petgraph::visit::{IntoNeighbors, IntoNodeIdentifiers};
use petgraph_layout_force_simulation::{Force, ForceToNode, Point};
use std::collections::HashMap;
use std::hash::Hash;

#[derive(Force)]
pub struct ForceAtlas2Force {
    degree: Vec<usize>,
    links: Vec<Vec<usize>>,
    k: f32,
    min_distance: f32,
}

impl ForceAtlas2Force {
    pub fn new<G>(graph: G) -> ForceAtlas2Force
    where
        G: IntoNodeIdentifiers + IntoNeighbors,
        G::NodeId: Eq + Hash,
    {
        let node_indices = graph
            .node_identifiers()
            .enumerate()
            .map(|(i, u)| (u, i))
            .collect::<HashMap<_, _>>();
        let mut degree = vec![0; node_indices.len()];
        let mut links = vec![vec![]; node_indices.len()];
        for u in graph.node_identifiers() {
            for v in graph.neighbors(u) {
                degree[node_indices[&u]] += 1;
                degree[node_indices[&v]] += 1;
                links[node_indices[&u]].push(node_indices[&v]);
            }
        }
        ForceAtlas2Force {
            degree,
            links,
            k: 1.,
            min_distance: 1.,
        }
    }
}

impl ForceToNode for ForceAtlas2Force {
    fn apply_to_node(&self, u: usize, points: &mut [Point], alpha: f32) {
        let n = points.len();
        for v in 0..n {
            if u == v {
                continue;
            }
            let dx = points[v].x - points[u].x;
            let dy = points[v].y - points[u].y;
            let d = (dx * dx + dy * dy).sqrt().max(self.min_distance);
            let c = self.k * (self.degree[u] as f32 + 1.) * (self.degree[v] as f32 + 1.) / d;
            points[u].vx -= alpha * c * dx;
            points[u].vy -= alpha * c * dy;
        }
        for &v in self.links[u].iter() {
            let dx = points[v].x - points[u].x;
            let dy = points[v].y - points[u].y;
            let d = (dx * dx + dy * dy).sqrt();
            points[u].vx += alpha * d * dx;
            points[u].vy += alpha * d * dy;
        }
    }
}
