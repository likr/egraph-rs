use crate::Sgd;
use ndarray::ArrayView2;
use petgraph::visit::{IntoEdges, IntoNodeIdentifiers};
use petgraph_algorithm_shortest_path::all_sources_dijkstra;
use petgraph_drawing::{DrawingIndex, DrawingValue};
use petgraph_linalg_kernel::Distance;

/// Builder for creating Full SGD instances.
///
/// This builder computes the shortest-path distances between all pairs of nodes
/// in the graph and creates an SGD instance that optimizes the layout to match
/// these distances geometrically. It considers all possible node pairs during
/// the optimization process, which makes it accurate but potentially slow for large graphs.
pub struct FullSgd;

impl Default for FullSgd {
    fn default() -> Self {
        Self::new()
    }
}

impl FullSgd {
    pub fn new() -> Self {
        Self
    }

    /// Creates a new SGD instance from a graph using all-pairs shortest paths.
    ///
    /// This method computes the all-pairs shortest path distances for the input graph
    /// and uses them as target distances for the layout optimization.
    ///
    /// # Parameters
    /// * `graph` - The input graph to be laid out
    /// * `length` - A function that maps edges to their lengths/weights
    ///
    /// # Returns
    /// A new SGD instance configured with all node pairs
    pub fn build<G, F, S>(&self, graph: G, length: F) -> Sgd<S>
    where
        G: IntoEdges + IntoNodeIdentifiers,
        G::NodeId: DrawingIndex + Ord,
        F: FnMut(G::EdgeRef) -> S,
        S: DrawingValue,
    {
        let d = all_sources_dijkstra(graph, length);
        self.build_with_distance_matrix(&d)
    }

    /// Creates a new SGD instance from a pre-computed distance matrix.
    ///
    /// This method is useful when you already have a distance matrix computed
    /// or want to use custom distances rather than shortest-path distances.
    ///
    /// # Parameters
    /// * `d` - A distance matrix containing distances between all node pairs
    ///
    /// # Returns
    /// A new SGD instance configured with all node pairs
    pub fn build_with_distance_matrix<N, S>(&self, d: &dyn Distance<N, S>) -> Sgd<S>
    where
        S: DrawingValue,
    {
        let n = d.shape().0;
        let mut node_pairs = vec![];
        for j in 1..n {
            for i in 0..j {
                let dij = d.get_by_index(i, j);
                let wij = S::one() / (dij * dij);
                node_pairs.push((i, j, dij, dij, wij, wij));
            }
        }
        Sgd::new(node_pairs)
    }

    /// Creates a new SGD instance from pre-computed `(n, n)` distance and optional weight arrays.
    ///
    /// Iterates over pairs `0 <= i < j < n`, skipping pairs where distances are non-finite or `<= 0`
    /// or where provided weights are non-finite, negative, or both zero. When `w` is `None`,
    /// weights are computed as `1 / d^2`.
    pub fn build_with_arrays<S>(
        &self,
        n: usize,
        d: ArrayView2<S>,
        w: Option<ArrayView2<S>>,
    ) -> Sgd<S>
    where
        S: DrawingValue,
    {
        assert_eq!(
            d.dim(),
            (n, n),
            "FullSgd::build_with_arrays: expected distance shape ({}, {}), got {:?}",
            n,
            n,
            d.dim()
        );
        if let Some(w_arr) = w {
            assert_eq!(
                w_arr.dim(),
                (n, n),
                "FullSgd::build_with_arrays: expected weight shape ({}, {}), got {:?}",
                n,
                n,
                w_arr.dim()
            );
        }

        let mut node_pairs = Vec::with_capacity(n.saturating_mul(n.saturating_sub(1)) / 2);
        for j in 1..n {
            for i in 0..j {
                let dij = d[[i, j]];
                let dji = d[[j, i]];
                if !dij.is_finite() || dij <= S::zero() || !dji.is_finite() || dji <= S::zero() {
                    continue;
                }
                let (wij, wji) = match w {
                    Some(w_arr) => {
                        let wij = w_arr[[i, j]];
                        let wji = w_arr[[j, i]];
                        if !wij.is_finite()
                            || !wji.is_finite()
                            || wij < S::zero()
                            || wji < S::zero()
                            || (wij == S::zero() && wji == S::zero())
                        {
                            continue;
                        }
                        (wij, wji)
                    }
                    None => {
                        let wij = S::one() / (dij * dij);
                        let wji = S::one() / (dji * dji);
                        if !wij.is_finite() || !wji.is_finite() {
                            continue;
                        }
                        (wij, wji)
                    }
                };
                node_pairs.push((i, j, dij, dji, wij, wji));
            }
        }
        Sgd::new(node_pairs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr2;

    #[test]
    fn test_full_sgd_build_with_arrays_default_weights_and_filtering() {
        let d = arr2(&[
            [0.0, 2.0, f64::INFINITY, 1.0],
            [2.0, 0.0, 0.0, f64::NAN],
            [f64::INFINITY, 0.0, 0.0, -1.0],
            [1.0, f64::NAN, -1.0, 0.0],
        ]);
        let sgd = FullSgd::new().build_with_arrays(4, d.view(), None);
        assert_eq!(
            sgd.node_pairs(),
            &vec![(0, 1, 2.0, 2.0, 0.25, 0.25), (0, 3, 1.0, 1.0, 1.0, 1.0),]
        );
    }

    #[test]
    fn test_full_sgd_build_with_arrays_custom_weights() {
        let d = arr2(&[[0.0, 1.0, 2.0], [1.0, 0.0, 3.0], [2.0, 3.0, 0.0]]);
        let w = arr2(&[[0.0, 5.0, 0.0], [5.0, 0.0, f64::NAN], [0.0, f64::NAN, 0.0]]);
        let sgd = FullSgd::new().build_with_arrays(3, d.view(), Some(w.view()));
        assert_eq!(sgd.node_pairs(), &vec![(0, 1, 1.0, 1.0, 5.0, 5.0)]);
    }
}
