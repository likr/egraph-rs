use ndarray::prelude::*;
use petgraph::{
    graph::{EdgeReference, IndexType},
    prelude::*,
    visit::IntoNodeIdentifiers,
    EdgeType,
};
use petgraph_algorithm_shortest_path::warshall_floyd;
use petgraph_layout_force_simulation::Coordinates;
use rand::prelude::SliceRandom;
use std::{collections::HashMap, f32::INFINITY};

pub struct Sgd {
    d: Array2<f32>,
    node_pairs: Vec<(usize, usize, f32)>,
    t: usize,
    a: f32,
    b: f32,
}

impl Sgd {
    pub fn new<N, E, Ty: EdgeType, Ix: IndexType, F: FnMut(EdgeReference<'_, E, Ix>) -> f32>(
        graph: &Graph<N, E, Ty, Ix>,
        length: &mut F,
    ) -> Sgd {
        let indices = graph
            .node_identifiers()
            .enumerate()
            .map(|(i, u)| (u, i))
            .collect::<HashMap<_, _>>();
        let n = indices.len();
        let d = warshall_floyd(graph, length);

        let mut node_pairs = vec![];
        let mut w_min = INFINITY;
        let mut w_max = 0.;
        for j in 1..n {
            for i in 0..j {
                let wij = 1. / (d[[i, j]] * d[[i, j]]);
                node_pairs.push((i, j, wij));
                if wij < w_min {
                    w_min = wij;
                }
                if wij > w_max {
                    w_max = wij;
                }
            }
        }

        let t_max = 15;
        let epsilon = 0.1;
        let eta_max = 1. / w_min;
        let eta_min = epsilon / w_max;
        Sgd {
            d,
            node_pairs,
            t: 0,
            a: eta_max,
            b: (eta_min / eta_max).ln() / (t_max - 1) as f32,
        }
    }

    pub fn apply<Ix: IndexType>(&mut self, coordinates: &mut Coordinates<Ix>) {
        let eta = self.a * (self.b * self.t as f32).exp();
        let mut rng = rand::thread_rng();
        self.node_pairs.shuffle(&mut rng);
        for &(i, j, wij) in self.node_pairs.iter() {
            let mu = (eta * wij).min(1.);
            let dij = self.d[[i, j]];
            let dx = coordinates.points[i].x - coordinates.points[j].x;
            let dy = coordinates.points[i].y - coordinates.points[j].y;
            let norm = (dx * dx + dy * dy).sqrt();
            let r = 0.5 * mu * (norm - dij) / norm;
            coordinates.points[i].x -= r * dx;
            coordinates.points[i].y -= r * dy;
            coordinates.points[j].x += r * dx;
            coordinates.points[j].y += r * dy;
        }
        self.t += 1;
    }
}
