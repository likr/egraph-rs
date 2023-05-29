use petgraph_drawing::{Drawing, DrawingIndex};

pub fn node_resolution<N>(drawing: &Drawing<N, f32>) -> f32
where
    N: DrawingIndex,
{
    let n = drawing.len();
    let r = 1. / (n as f32).sqrt();

    let mut d_max = 0f32;
    for i in 1..n {
        for j in 0..i {
            let dx = drawing.coordinates[[i, 0]] - drawing.coordinates[[j, 0]];
            let dy = drawing.coordinates[[i, 1]] - drawing.coordinates[[j, 1]];
            d_max = d_max.max((dx).hypot(dy));
        }
    }

    let mut s = 0.;
    for i in 1..n {
        for j in 0..i {
            let dx = drawing.coordinates[[i, 0]] - drawing.coordinates[[j, 0]];
            let dy = drawing.coordinates[[i, 1]] - drawing.coordinates[[j, 1]];
            s += (1. - (dx).hypot(dy) / (r * d_max)).powi(2);
        }
    }
    s
}
