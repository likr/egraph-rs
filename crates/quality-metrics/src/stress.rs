use petgraph_algorithm_shortest_path::{DistanceMatrix, FullDistanceMatrix};
use petgraph_drawing::{Delta, Drawing, DrawingIndex, DrawingValue, Metric};

/// Calculates the stress metric for a graph layout.
///
/// Stress is a fundamental metric in graph drawing that measures how well the
/// Euclidean distances in the layout match the graph-theoretical distances
/// (typically shortest path distances). It is calculated as the sum of squared
/// relative differences between these distances.
///
/// This implementation computes stress as:
/// Σ [(||pos(i) - pos(j)|| - d(i,j))² / d(i,j)²]
/// where:
/// - pos(i) is the position of node i in the layout
/// - d(i,j) is the graph-theoretical distance between nodes i and j
/// - ||.|| denotes the Euclidean norm
///
/// A lower stress value indicates a better layout in terms of faithfully
/// representing the graph distances in the embedding space.
///
/// # Parameters
///
/// * `drawing`: The layout of the graph
/// * `d`: The full distance matrix containing shortest path distances between all node pairs
///
/// # Returns
///
/// A value of type `S` representing the stress metric. Lower values indicate
/// better preservation of graph distances.
///
/// # Type Parameters
///
/// * `Diff`: A type for representing differences between metric values
/// * `D`: A drawing type
/// * `N`: Node ID type
/// * `M`: Metric type used in the drawing
/// * `S`: Numeric type for distance calculations
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
