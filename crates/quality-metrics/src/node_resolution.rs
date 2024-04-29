use petgraph_drawing::{Difference, Drawing, DrawingValue, Metric};

pub fn node_resolution<Diff, D, M, S>(drawing: &D) -> S
where
    D: Drawing<Item = M>,
    Diff: Difference<S = S>,
    M: Copy + Metric<D = Diff>,
    S: DrawingValue,
{
    let n = drawing.len();
    let r = S::one() / S::from_usize(n).unwrap().sqrt();

    let mut d_max = S::zero();
    for i in 1..n {
        for j in 0..i {
            let delta = *drawing.raw_entry(i) - *drawing.raw_entry(j);
            d_max = d_max.max(delta.norm());
        }
    }

    let mut s = S::zero();
    for i in 1..n {
        for j in 0..i {
            let delta = *drawing.raw_entry(i) - *drawing.raw_entry(j);
            s += (S::one() - delta.norm() / (r * d_max))
                .powi(2)
                .max(S::zero());
        }
    }
    s
}
