use petgraph_drawing::{Delta, Drawing, DrawingValue, Metric};

pub fn node_resolution<Diff, D, M, S>(drawing: &D) -> S
where
    D: Drawing<Item = M>,
    Diff: Delta<S = S>,
    M: Copy + Metric<D = Diff>,
    S: DrawingValue,
{
    let n = drawing.len();
    let r = S::one() / S::from_usize(n).unwrap().sqrt();

    let mut d_max = S::zero();
    for i in 1..n {
        for j in 0..i {
            let delta = drawing.delta(i, j);
            d_max = d_max.max(delta.norm());
        }
    }

    let mut s = S::zero();
    for i in 1..n {
        for j in 0..i {
            let delta = drawing.delta(i, j);
            s += (S::one() - delta.norm() / (r * d_max))
                .powi(2)
                .max(S::zero());
        }
    }
    s
}
