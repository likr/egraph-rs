use petgraph::visit::{IntoNeighbors, IntoNodeIdentifiers};
use petgraph_drawing::{Drawing, DrawingEuclidean2d, DrawingIndex, DrawingValue};
use std::collections::HashMap;

pub struct ForceAtlas2<S>
where
    S: DrawingValue,
{
    degree: Vec<usize>,
    links: Vec<Vec<usize>>,
    k: S,
    min_distance: S,
}

impl<S> ForceAtlas2<S>
where
    S: DrawingValue + Default,
{
    pub fn new<G>(graph: G) -> ForceAtlas2<S>
    where
        G: IntoNodeIdentifiers + IntoNeighbors,
        G::NodeId: DrawingIndex,
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
        ForceAtlas2 {
            degree,
            links,
            k: S::one(),
            min_distance: S::one(),
        }
    }

    pub fn apply_to_node<N>(&self, u: usize, drawing: &mut DrawingEuclidean2d<N, S>, alpha: S)
    where
        N: DrawingIndex,
    {
        let n = drawing.len();
        for v in 0..n {
            if u == v {
                continue;
            }
            let dx = drawing.raw_entry(v).0 - drawing.raw_entry(u).0;
            let dy = drawing.raw_entry(v).1 - drawing.raw_entry(u).1;
            let d = (dx * dx + dy * dy).sqrt().max(self.min_distance);
            let du = S::from(self.degree[u] + 1).unwrap();
            let c = self.k * du * du / d;
            drawing.raw_entry_mut(u).0 -= alpha * c * dx;
            drawing.raw_entry_mut(u).1 -= alpha * c * dy;
        }
        for &v in self.links[u].iter() {
            let dx = drawing.raw_entry(v).0 - drawing.raw_entry(u).0;
            let dy = drawing.raw_entry(v).1 - drawing.raw_entry(u).1;
            let d = (dx * dx + dy * dy).sqrt();
            drawing.raw_entry_mut(u).0 += alpha * d * dx;
            drawing.raw_entry_mut(u).1 += alpha * d * dy;
        }
    }

    pub fn apply<N>(&self, drawing: &mut DrawingEuclidean2d<N, S>, alpha: S)
    where
        N: DrawingIndex,
    {
        let n = drawing.len();
        for u in 0..n {
            self.apply_to_node(u, drawing, alpha);
        }
    }
}
