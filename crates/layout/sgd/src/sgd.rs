use crate::Scheduler;
use petgraph_drawing::{Delta, Drawing, DrawingValue, Metric};
use rand::prelude::*;

/// Base trait for Stochastic Gradient Descent (SGD) layout algorithms.
///
/// This trait defines the core functionality required for all SGD algorithm variants.
/// It handles operations like shuffling node pairs, applying forces, creating schedulers,
/// and updating distances and weights.
///
/// The type parameter `S` represents the scalar type used for calculations
/// (typically `f32` or `f64`).
pub trait Sgd<S> {
    /// Returns a reference to the node pairs used in the SGD algorithm.
    ///
    /// Each tuple contains:
    /// - `(i, j)`: Indices of the node pair
    /// - `(dij, dji)`: Target distances from i to j and j to i
    /// - `(wij, wji)`: Weights for the forces from i to j and j to i
    fn node_pairs(&self) -> &Vec<(usize, usize, S, S, S, S)>;

    /// Returns a mutable reference to the node pairs.
    ///
    /// This allows modifying the node pairs data structure for algorithms
    /// that need to update pair information during execution.
    fn node_pairs_mut(&mut self) -> &mut Vec<(usize, usize, S, S, S, S)>;

    /// Randomly shuffles the node pairs to improve convergence.
    ///
    /// SGD algorithms typically process node pairs in a random order to avoid
    /// getting stuck in local minima. This method randomizes the order using
    /// the provided random number generator.
    fn shuffle<R: Rng>(&mut self, rng: &mut R) {
        self.node_pairs_mut().shuffle(rng);
    }

    /// Applies the SGD force calculations to the drawing, moving nodes toward their optimal positions.
    ///
    /// This is the core method that performs a single iteration of the SGD algorithm.
    /// For each node pair (i, j):
    /// 1. Calculates the learning rate factors (mu_i, mu_j) based on weights and the global learning rate eta
    /// 2. Computes the displacement vectors based on the difference between current and target distances
    /// 3. Moves the nodes according to the calculated forces
    ///
    /// # Parameters
    /// * `drawing` - The current node position drawing to update
    /// * `eta` - The current learning rate (typically decreases over time via a scheduler)
    fn apply<Diff, D, M>(&self, drawing: &mut D, eta: S)
    where
        D: Drawing<Item = M>,
        Diff: Delta<S = S>,
        M: Metric<D = Diff>,
        S: DrawingValue,
    {
        for &(i, j, dij, dji, wij, wji) in self.node_pairs().iter() {
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

    /// Creates a scheduler for controlling the learning rate over time.
    ///
    /// This method analyzes the weight distribution in the node pairs to determine
    /// appropriate minimum and maximum learning rates, then initializes a scheduler
    /// with these values.
    ///
    /// # Parameters
    /// * `t_max` - The maximum number of iterations to run
    /// * `epsilon` - A small value used to calculate the minimum learning rate
    ///
    /// # Returns
    /// A scheduler initialized with appropriate learning rate bounds
    fn scheduler<SC>(&self, t_max: usize, epsilon: S) -> SC
    where
        SC: Scheduler<S>,
        S: DrawingValue,
    {
        let mut w_min = S::infinity();
        let mut w_max = S::zero();
        for &(_, _, _, _, wij, wji) in self.node_pairs().iter() {
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
        SC::init(t_max, eta_min, eta_max)
    }

    /// Updates the target distances for all node pairs using the provided function.
    ///
    /// This method allows dynamically adjusting the target distances during the layout process,
    /// which can be useful for algorithms that refine their distance model over time.
    ///
    /// # Parameters
    /// * `distance` - A function that takes (node_i, node_j, current_distance, current_weight)
    ///   and returns a new target distance
    fn update_distance<F>(&mut self, mut distance: F)
    where
        F: FnMut(usize, usize, S, S) -> S,
        S: Copy,
    {
        for p in self.node_pairs_mut() {
            let (i, j, dij, dji, wij, wji) = p;
            p.2 = distance(*i, *j, *dij, *wij);
            p.3 = distance(*j, *i, *dji, *wji);
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
    fn update_weight<F>(&mut self, mut weight: F)
    where
        F: FnMut(usize, usize, S, S) -> S,
        S: Copy,
    {
        for p in self.node_pairs_mut() {
            let (i, j, dij, dji, wij, wji) = p;
            p.4 = weight(*i, *j, *dij, *wij);
            p.5 = weight(*j, *i, *dji, *wji);
        }
    }
}
