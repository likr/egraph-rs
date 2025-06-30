use petgraph_drawing::{Delta, Drawing, DrawingValue, Metric};
use rand::prelude::*;

/// Stochastic Gradient Descent (SGD) implementation for graph layout algorithms.
///
/// This struct holds node pairs and learning rate parameters for SGD-based graph layout.
/// It replaces the previous trait-based approach with a concrete implementation that
/// all SGD algorithm variants can use.
///
/// The type parameter `S` represents the scalar type used for calculations
/// (typically `f32` or `f64`).
pub struct Sgd<S> {
    /// List of node pairs to be considered during layout optimization.
    /// Each tuple contains (i, j, distance_ij, distance_ji, weight_ij, weight_ji)
    node_pairs: Vec<(usize, usize, S, S, S, S)>,
    /// Small value used for numerical stability
    epsilon: S,
    /// Minimum learning rate (calculated from weights and epsilon)
    eta_min: S,
    /// Maximum learning rate (calculated from weights)
    eta_max: S,
}

impl<S> Sgd<S>
where
    S: DrawingValue,
{
    /// Creates a new SGD instance with the given node pairs and epsilon.
    ///
    /// This constructor automatically calculates eta_min and eta_max from the
    /// weight distribution in the node pairs to avoid repeated calculations
    /// during the layout process.
    ///
    /// # Parameters
    /// * `node_pairs` - List of node pairs with distances and weights
    /// * `epsilon` - Small value for numerical stability
    ///
    /// # Returns
    /// A new SGD instance ready for layout optimization
    pub fn new(node_pairs: Vec<(usize, usize, S, S, S, S)>, epsilon: S) -> Self {
        let mut w_min = S::infinity();
        let mut w_max = S::zero();
        for &(_, _, _, _, wij, wji) in &node_pairs {
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

        Self {
            node_pairs,
            epsilon,
            eta_min,
            eta_max,
        }
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
    /// The eta parameter is expected to be in the range [0, 1] and is internally normalized
    /// to the appropriate learning rate range [eta_min, eta_max] based on the weight distribution.
    ///
    /// For each node pair (i, j):
    /// 1. Normalizes the learning rate from [0,1] to [eta_min, eta_max]
    /// 2. Calculates the learning rate factors (mu_i, mu_j) based on weights and the normalized learning rate
    /// 3. Computes the displacement vectors based on the difference between current and target distances
    /// 4. Moves the nodes according to the calculated forces
    ///
    /// # Parameters
    /// * `drawing` - The current node position drawing to update
    /// * `eta` - The current learning rate in [0, 1] range (from scheduler)
    pub fn apply<Diff, D, M>(&self, drawing: &mut D, eta: S)
    where
        D: Drawing<Item = M>,
        Diff: Delta<S = S>,
        M: Metric<D = Diff>,
    {
        // Normalize eta from [0,1] to [eta_min, eta_max]
        let normalized_eta = self.eta_min + eta * (self.eta_max - self.eta_min);

        for &(i, j, dij, dji, wij, wji) in &self.node_pairs {
            let mu_i = (normalized_eta * wij).min(S::one());
            let mu_j = (normalized_eta * wji).min(S::one());
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
    /// After updating weights, this method recalculates eta_min and eta_max to ensure
    /// proper learning rate normalization.
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

        // Recalculate eta_min and eta_max after weight updates
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
        self.eta_max = S::one() / w_min;
        self.eta_min = self.epsilon / w_max;
    }
}
