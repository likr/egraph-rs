use ndarray::prelude::*;
use petgraph::graph::IndexType;
use petgraph_layout_force_simulation::Coordinates;

pub fn stress<Ix: IndexType>(coordinates: &Coordinates<Ix>, d: &Array2<f32>) -> f32 {
    let n = coordinates.len();
    let mut s = 0.;
    for j in 1..n {
        for i in 0..j {
            let dx = coordinates.points[i].x - coordinates.points[j].x;
            let dy = coordinates.points[i].y - coordinates.points[j].y;
            let norm = (dx * dx + dy * dy).sqrt();
            let dij = d[[i, j]];
            let e = (norm - dij) / dij;
            s += e * e;
        }
    }
    s
}
