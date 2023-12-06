use ndarray::prelude::*;
use petgraph_drawing::{Drawing2D, DrawingIndex};

pub fn stress<N>(drawing: &Drawing2D<N, f32>, d: &Array2<f32>) -> f32
where
    N: DrawingIndex,
{
    let n = drawing.len();
    let mut s = 0.;
    for j in 1..n {
        for i in 0..j {
            let dx = drawing.coordinates[i].0 - drawing.coordinates[j].0;
            let dy = drawing.coordinates[i].1 - drawing.coordinates[j].1;
            let norm = (dx * dx + dy * dy).sqrt();
            let dij = d[[i, j]];
            let e = (norm - dij) / dij;
            s += e * e;
        }
    }
    s
}
