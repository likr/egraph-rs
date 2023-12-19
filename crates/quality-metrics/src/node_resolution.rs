use petgraph_drawing::{Difference, Drawing, DrawingIndex, DrawingValue, Metric};

pub fn node_resolution<N, M, D, S>(drawing: &Drawing<N, M>) -> S
where
    N: DrawingIndex,
    M: Copy + Metric<D = D>,
    D: Difference<S = S>,
    S: DrawingValue,
{
    let n = drawing.len();
    let r = S::one() / S::from_usize(n).unwrap().sqrt();

    let mut d_max = S::zero();
    for i in 1..n {
        for j in 0..i {
            let delta = drawing.coordinates[i] - drawing.coordinates[j];
            d_max = d_max.max(delta.norm());
        }
    }

    let mut s = S::zero();
    for i in 1..n {
        for j in 0..i {
            let delta = drawing.coordinates[i] - drawing.coordinates[j];
            s += (S::one() - delta.norm() / (r * d_max))
                .powi(2)
                .max(S::zero());
        }
    }
    s
}
