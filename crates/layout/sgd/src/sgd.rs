use ndarray::ArrayView2;
use petgraph_drawing::{Delta, Drawing, DrawingValue, Metric};
use rand::prelude::*;

/// Stochastic Gradient Descent (SGD) implementation for graph layout algorithms.
///
/// This struct holds node pairs for SGD-based graph layout.
/// It replaces the previous trait-based approach with a concrete implementation that
/// all SGD algorithm variants can use.
///
/// The type parameter `S` represents the scalar type used for calculations
/// (typically `f32` or `f64`).
pub struct Sgd<S> {
    /// List of node pairs to be considered during layout optimization.
    /// Each tuple contains (i, j, distance_ij, distance_ji, weight_ij, weight_ji)
    node_pairs: Vec<(usize, usize, S, S, S, S)>,
}

impl<S> Sgd<S>
where
    S: DrawingValue,
{
    /// Creates a new SGD instance with the given node pairs.
    ///
    /// # Parameters
    /// * `node_pairs` - List of node pairs with distances and weights
    ///
    /// # Returns
    /// A new SGD instance ready for layout optimization
    pub fn new(node_pairs: Vec<(usize, usize, S, S, S, S)>) -> Self {
        Self { node_pairs }
    }

    /// Creates a new SGD instance from explicit `(i, j, dij, dji, wij, wji)` node pairs.
    pub fn from_node_pairs(node_pairs: Vec<(usize, usize, S, S, S, S)>) -> Self {
        Self::new(node_pairs)
    }

    /// Returns a reference to the node pairs used in the SGD algorithm.
    ///
    /// Each tuple contains:
    /// - `(i, j)`: Indices of the node pair
    /// - `(dij, dji)`: Target distances from i to j and j to i
    /// - `(wij, wji)`: Weights for the forces from i to j and j to i
    pub fn node_pairs(&self) -> &Vec<(usize, usize, S, S, S, S)> {
        &self.node_pairs
    }

    /// Creates a scheduler with parameters suitable for this SGD instance.
    ///
    /// This method creates a scheduler that uses eta_min and eta_max calculated
    /// from the current weight distribution in the node pairs.
    ///
    /// # Parameters
    /// * `t_max` - The maximum number of iterations for the scheduler
    /// * `epsilon` - A small value used to calculate eta_min
    ///
    /// # Returns
    /// A scheduler instance configured with appropriate learning rate bounds
    pub fn scheduler<T: crate::scheduler::Scheduler<S>>(&self, t_max: usize, epsilon: S) -> T {
        let (eta_min, eta_max) = self.calculate_eta_bounds(epsilon);
        T::init(t_max, eta_min, eta_max)
    }

    /// Calculates eta_min and eta_max from the current weight distribution.
    fn calculate_eta_bounds(&self, epsilon: S) -> (S, S) {
        let mut w_min = S::infinity();
        let mut w_max = S::zero();
        for &(_, _, _, _, wij, wji) in &self.node_pairs {
            for w in [wij, wji] {
                if w == S::zero() {
                    continue;
                }
                if w < w_min {
                    w_min = w;
                }
                if w > w_max {
                    w_max = w;
                }
            }
        }
        let eta_max = S::one() / w_min;
        let eta_min = epsilon / w_max;
        (eta_min, eta_max)
    }

    /// Randomly shuffles the node pairs to improve convergence.
    ///
    /// SGD algorithms typically process node pairs in a random order to avoid
    /// getting stuck in local minima. This method randomizes the order using
    /// the provided random number generator.
    pub fn shuffle<R: Rng>(&mut self, rng: &mut R) {
        self.node_pairs.shuffle(rng);
    }

    /// Applies the SGD force calculations to the drawing, moving nodes toward their optimal positions.
    ///
    /// This is the core method that performs a single iteration of the SGD algorithm.
    /// The eta parameter is expected to be the actual learning rate (not normalized),
    /// typically coming from a scheduler that was created using this SGD instance.
    ///
    /// For each node pair (i, j):
    /// 1. Calculates the learning rate factors (mu_i, mu_j) based on weights and the learning rate
    /// 2. Computes the displacement vectors based on the difference between current and target distances
    /// 3. Moves the nodes according to the calculated forces
    ///
    /// # Parameters
    /// * `drawing` - The current node position drawing to update
    /// * `eta` - The current learning rate (from scheduler)
    pub fn apply<Diff, D, M>(&self, drawing: &mut D, eta: S)
    where
        D: Drawing<Item = M>,
        Diff: Delta<S = S>,
        M: Metric<D = Diff>,
    {
        for &(i, j, dij, dji, wij, wji) in &self.node_pairs {
            if dij.is_infinite() || dji.is_infinite() || (wij <= S::zero() && wji <= S::zero()) {
                continue;
            }
            let mu_i = (eta * wij).min(S::one());
            let mu_j = (eta * wji).min(S::one());
            let delta = drawing.delta(i, j);
            let norm = delta.norm();
            if norm > S::zero() {
                let r_i = S::from_f32(0.5).unwrap() * (norm - dij) / norm;
                let r_j = S::from_f32(0.5).unwrap() * (norm - dji) / norm;
                *drawing.raw_entry_mut(i) += delta.clone() * -r_i * mu_i;
                *drawing.raw_entry_mut(j) += delta.clone() * r_j * mu_j;
            }
        }
    }

    /// Updates the target distances for all node pairs using the provided function.
    ///
    /// This method allows dynamically adjusting the target distances during the layout process,
    /// which can be useful for algorithms that refine their distance model over time.
    ///
    /// # Parameters
    /// * `distance` - A function that takes (node_i, node_j, current_distance, current_weight)
    ///   and returns a new target distance
    pub fn update_distance<F>(&mut self, mut distance: F)
    where
        F: FnMut(usize, usize, S, S) -> S,
        S: Copy,
    {
        for p in &mut self.node_pairs {
            let (i, j, dij, dji, wij, wji) = *p;
            p.2 = distance(i, j, dij, wij);
            p.3 = distance(j, i, dji, wji);
        }
    }

    /// Updates the weights for all node pairs using the provided function.
    ///
    /// Weights control how strongly each node pair affects the layout. Higher weights
    /// cause node pairs to more strongly enforce their target distances.
    ///
    /// # Parameters
    /// * `weight` - A function that takes (node_i, node_j, current_distance, current_weight)
    ///   and returns a new weight value
    pub fn update_weight<F>(&mut self, mut weight: F)
    where
        F: FnMut(usize, usize, S, S) -> S,
        S: Copy,
    {
        for p in &mut self.node_pairs {
            let (i, j, dij, dji, wij, wji) = *p;
            p.4 = weight(i, j, dij, wij);
            p.5 = weight(j, i, dji, wji);
        }
    }

    fn validate_pair_bounds(&self, n: usize) {
        for &(i, j, _, _, _, _) in &self.node_pairs {
            assert!(
                i < n && j < n,
                "Sgd: pair ({}, {}) out of bounds for matrix size {}",
                i,
                j,
                n
            );
        }
    }

    /// Updates weights for all node pairs from an `(n, n)` weight matrix.
    ///
    /// Sets `pair.4 = w[[pair.0, pair.1]]` and `pair.5 = w[[pair.1, pair.0]]`.
    pub fn update_weight_from_array(&mut self, w: ArrayView2<S>) {
        let (rows, cols) = w.dim();
        assert_eq!(
            rows, cols,
            "Sgd::update_weight_from_array: expected square matrix, got ({}, {})",
            rows, cols
        );
        self.validate_pair_bounds(rows);
        for p in &mut self.node_pairs {
            p.4 = w[[p.0, p.1]];
            p.5 = w[[p.1, p.0]];
        }
    }

    /// Updates target distances (and optionally weights) from `(n, n)` matrices.
    ///
    /// Sets `pair.2 = d[[pair.0, pair.1]]`, `pair.3 = d[[pair.1, pair.0]]`,
    /// and when `w` is `Some`, updates `pair.4 = w[[pair.0, pair.1]]` and `pair.5 = w[[pair.1, pair.0]]`.
    pub fn update_distance_from_array(&mut self, d: ArrayView2<S>, w: Option<ArrayView2<S>>) {
        let (rows, cols) = d.dim();
        assert_eq!(
            rows, cols,
            "Sgd::update_distance_from_array: expected square distance matrix, got ({}, {})",
            rows, cols
        );
        if let Some(w_arr) = w {
            assert_eq!(
                w_arr.dim(),
                (rows, cols),
                "Sgd::update_distance_from_array: weight matrix shape {:?} does not match distance matrix shape ({}, {})",
                w_arr.dim(),
                rows,
                cols
            );
        }
        self.validate_pair_bounds(rows);
        for p in &mut self.node_pairs {
            p.2 = d[[p.0, p.1]];
            p.3 = d[[p.1, p.0]];
            if let Some(w_arr) = w {
                p.4 = w_arr[[p.0, p.1]];
                p.5 = w_arr[[p.1, p.0]];
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr2;

    #[test]
    fn test_sgd_from_node_pairs_and_bulk_updates() {
        let pairs = vec![(0, 1, 1.0, 1.0, 1.0, 1.0), (1, 2, 2.0, 2.0, 0.25, 0.25)];
        let mut sgd = Sgd::from_node_pairs(pairs);
        assert_eq!(sgd.node_pairs().len(), 2);

        let w = arr2(&[[0.0, 2.0, 0.0], [3.0, 0.0, 4.0], [0.0, 5.0, 0.0]]);
        sgd.update_weight_from_array(w.view());
        assert_eq!(sgd.node_pairs()[0], (0, 1, 1.0, 1.0, 2.0, 3.0));
        assert_eq!(sgd.node_pairs()[1], (1, 2, 2.0, 2.0, 4.0, 5.0));

        let d = arr2(&[[0.0, 10.0, 0.0], [11.0, 0.0, 12.0], [0.0, 13.0, 0.0]]);
        sgd.update_distance_from_array(d.view(), None);
        assert_eq!(sgd.node_pairs()[0], (0, 1, 10.0, 11.0, 2.0, 3.0));
        assert_eq!(sgd.node_pairs()[1], (1, 2, 12.0, 13.0, 4.0, 5.0));

        let w2 = arr2(&[[0.0, 0.5, 0.0], [0.6, 0.0, 0.7], [0.0, 0.8, 0.0]]);
        sgd.update_distance_from_array(d.view(), Some(w2.view()));
        assert_eq!(sgd.node_pairs()[0], (0, 1, 10.0, 11.0, 0.5, 0.6));
        assert_eq!(sgd.node_pairs()[1], (1, 2, 12.0, 13.0, 0.7, 0.8));
    }
}
