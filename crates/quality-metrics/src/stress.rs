use petgraph_algorithm_shortest_path::{DistanceMatrix, FullDistanceMatrix};
use petgraph_drawing::{Delta, Drawing, DrawingIndex, DrawingValue, Metric};

pub fn stress<Diff, D, N, M, S>(drawing: &D, d: &FullDistanceMatrix<N, S>) -> S
where
    D: Drawing<Item = M, Index = N>,
    Diff: Delta<S = S>,
    N: DrawingIndex,
    M: Copy + Metric<D = Diff>,
    S: DrawingValue,
{
    let n = drawing.len();
    let mut s = S::zero();
    for j in 1..n {
        for i in 0..j {
            let delta = drawing.delta(i, j);
            let norm = delta.norm();
            let dij = d.get_by_index(i, j);
            let e = (norm - dij) / dij;
            s += e * e;
        }
    }
    s
}
