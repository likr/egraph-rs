use petgraph_algorithm_shortest_path::{DistanceMatrix, FullDistanceMatrix};
use petgraph_drawing::{Difference, Drawing, DrawingIndex, DrawingValue, Metric};

pub fn stress<N, M, D, S>(drawing: &Drawing<N, M>, d: &FullDistanceMatrix<N, S>) -> S
where
    N: DrawingIndex,
    M: Copy + Metric<D = D>,
    D: Difference<S = S>,
    S: DrawingValue,
{
    let n = drawing.len();
    let mut s = S::zero();
    for j in 1..n {
        for i in 0..j {
            let delta = drawing.coordinates[i] - drawing.coordinates[j];
            let norm = delta.norm();
            let dij = d.get_by_index(i, j);
            let e = (norm - dij) / dij;
            s += e * e;
        }
    }
    s
}
