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
            let dij = d.get_by_index(i, j);
            if !dij.is_finite() || dij <= S::zero() {
                continue;
            }
            let delta = drawing.delta(i, j);
            let norm = delta.norm();
            if !norm.is_finite() {
                continue;
            }
            let e = (norm - dij) / dij;
            s += e * e;
        }
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;
    use petgraph::graph::{NodeIndex, UnGraph};
    use petgraph_drawing::DrawingEuclidean2d;

    #[test]
    fn test_stress_skips_zero_and_nonfinite_target_distances() {
        let mut g = UnGraph::<(), ()>::new_undirected();
        for _ in 0..3 {
            g.add_node(());
        }
        let d = DrawingEuclidean2d::<NodeIndex, f64>::initial_placement(&g);
        let dm_zero = FullDistanceMatrix::from_array(&g, Array2::zeros((3, 3)));
        assert_eq!(stress(&d, &dm_zero), 0.0);

        let dm_nan = FullDistanceMatrix::from_array(&g, Array2::from_elem((3, 3), f64::NAN));
        assert_eq!(stress(&d, &dm_nan), 0.0);

        let dm_inf = FullDistanceMatrix::from_array(&g, Array2::from_elem((3, 3), f64::INFINITY));
        assert_eq!(stress(&d, &dm_inf), 0.0);
    }
}
