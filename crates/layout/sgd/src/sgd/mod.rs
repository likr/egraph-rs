mod updates;

use petgraph_drawing::{Delta, Drawing, DrawingValue, Metric};
use rand::prelude::*;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PairSlot {
    Spring(usize),
    Hinge(usize),
}

/// Stochastic Gradient Descent (SGD) implementation for graph layout algorithms.
///
/// This struct holds node pairs for SGD-based graph layout.
/// It replaces the previous trait-based approach with a concrete implementation that
/// all SGD algorithm variants can use.
///
/// The type parameter `S` represents the scalar type used for calculations
/// (typically `f32` or `f64`).
pub struct Sgd<S> {
    /// List of two-sided spring node pairs to be considered during layout optimization.
    /// Each tuple contains (i, j, distance_ij, distance_ji, weight_ij, weight_ji)
    node_pairs: Vec<(usize, usize, S, S, S, S)>,
    /// List of one-sided hinge repulsion pairs that exert force strictly when `norm < d_target`.
    hinge_pairs: Vec<(usize, usize, S, S, S, S)>,
    /// Interleaved shuffle schedule populated only when `hinge_pairs` is non-empty so that
    /// spring-only callers incur zero extra allocation and preserve legacy RNG streams.
    schedule: Vec<PairSlot>,
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
        Self {
            node_pairs,
            hinge_pairs: Vec::new(),
            schedule: Vec::new(),
        }
    }

    /// Creates a new SGD instance from explicit `(i, j, dij, dji, wij, wji)` node pairs.
    pub fn from_node_pairs(node_pairs: Vec<(usize, usize, S, S, S, S)>) -> Self {
        Self::new(node_pairs)
    }

    /// Creates a new SGD instance combining two-sided spring pairs and one-sided hinge pairs.
    pub fn new_with_hinge_pairs(
        node_pairs: Vec<(usize, usize, S, S, S, S)>,
        hinge_pairs: Vec<(usize, usize, S, S, S, S)>,
    ) -> Self {
        let mut sgd = Self {
            node_pairs,
            hinge_pairs,
            schedule: Vec::new(),
        };
        if !sgd.hinge_pairs.is_empty() {
            sgd.rebuild_schedule();
        }
        sgd
    }

    /// Appends one-sided hinge repulsion pairs `(i, j, dij, dji, wij, wji)` to this instance.
    pub fn add_hinge_pairs<I>(&mut self, pairs: I)
    where
        I: IntoIterator<Item = (usize, usize, S, S, S, S)>,
    {
        self.hinge_pairs.extend(pairs);
        if !self.hinge_pairs.is_empty() {
            self.rebuild_schedule();
        }
    }

    fn rebuild_schedule(&mut self) {
        self.schedule.clear();
        self.schedule
            .reserve(self.node_pairs.len() + self.hinge_pairs.len());
        self.schedule
            .extend((0..self.node_pairs.len()).map(PairSlot::Spring));
        self.schedule
            .extend((0..self.hinge_pairs.len()).map(PairSlot::Hinge));
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

    /// Returns a slice of the one-sided hinge repulsion pairs.
    pub fn hinge_pairs(&self) -> &[(usize, usize, S, S, S, S)] {
        &self.hinge_pairs
    }

    /// Returns the total number of active pairs (spring pairs plus hinge pairs).
    pub fn total_pair_count(&self) -> usize {
        self.node_pairs.len() + self.hinge_pairs.len()
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

    /// Calculates eta_min and eta_max from the current weight distribution across spring and hinge pairs.
    fn calculate_eta_bounds(&self, epsilon: S) -> (S, S) {
        let mut w_min = S::infinity();
        let mut w_max = S::zero();
        for &(_, _, _, _, wij, wji) in self.node_pairs.iter().chain(self.hinge_pairs.iter()) {
            for w in [wij, wji] {
                if !w.is_finite() || w <= S::zero() {
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
        if !epsilon.is_finite() || epsilon <= S::zero() || w_max <= S::zero() || !w_min.is_finite()
        {
            return (S::zero(), S::zero());
        }
        let eta_max = S::one() / w_min;
        let eta_min = epsilon / w_max;
        if !eta_min.is_finite() || !eta_max.is_finite() {
            return (S::zero(), S::zero());
        }
        (eta_min, eta_max)
    }

    /// Randomly shuffles the node pairs to improve convergence.
    ///
    /// SGD algorithms typically process node pairs in a random order to avoid
    /// getting stuck in local minima. This method randomizes the order using
    /// the provided random number generator.
    pub fn shuffle<R: Rng>(&mut self, rng: &mut R) {
        if self.hinge_pairs.is_empty() {
            self.node_pairs.shuffle(rng);
        } else {
            self.schedule.shuffle(rng);
        }
    }

    #[inline]
    fn apply_spring_pair<Diff, D, M>(drawing: &mut D, eta: S, pair: (usize, usize, S, S, S, S))
    where
        D: Drawing<Item = M>,
        Diff: Delta<S = S>,
        M: Metric<D = Diff>,
    {
        let (i, j, dij, dji, wij, wji) = pair;
        if !dij.is_finite() || !dji.is_finite() {
            return;
        }
        let mu_i = if wij.is_finite() && wij > S::zero() {
            (eta * wij).min(S::one())
        } else {
            S::zero()
        };
        let mu_j = if wji.is_finite() && wji > S::zero() {
            (eta * wji).min(S::one())
        } else {
            S::zero()
        };
        if mu_i <= S::zero() && mu_j <= S::zero() {
            return;
        }
        let delta = drawing.delta(i, j);
        let norm = delta.norm();
        if norm.is_finite() && norm > S::zero() {
            let r_i = S::from_f32(0.5).unwrap() * (norm - dij) / norm;
            let r_j = S::from_f32(0.5).unwrap() * (norm - dji) / norm;
            *drawing.raw_entry_mut(i) += delta.clone() * -r_i * mu_i;
            *drawing.raw_entry_mut(j) += delta * r_j * mu_j;
        }
    }

    #[inline]
    fn apply_hinge_pair<Diff, D, M>(drawing: &mut D, eta: S, pair: (usize, usize, S, S, S, S))
    where
        D: Drawing<Item = M>,
        Diff: Delta<S = S>,
        M: Metric<D = Diff>,
    {
        let (i, j, dij, dji, wij, wji) = pair;
        if !dij.is_finite()
            || !dji.is_finite()
            || dij <= S::zero()
            || dji <= S::zero()
            || (wij <= S::zero() && wji <= S::zero())
        {
            return;
        }
        let delta = drawing.delta(i, j);
        let norm = delta.norm();
        // Hinge barriers exert strictly repulsive force when 0 < ||x_i - x_j|| < d_target
        // and exact zero force once the target separation is satisfied.
        if !norm.is_finite() || norm <= S::zero() {
            return;
        }
        let update_i = wij > S::zero() && norm < dij;
        let update_j = wji > S::zero() && norm < dji;
        if !update_i && !update_j {
            return;
        }
        let half = S::from_f32(0.5).unwrap();
        if update_i && update_j {
            let mu_i = (eta * wij).min(S::one());
            let r_i = half * (norm - dij) / norm;
            *drawing.raw_entry_mut(i) += delta.clone() * -r_i * mu_i;
            let mu_j = (eta * wji).min(S::one());
            let r_j = half * (norm - dji) / norm;
            *drawing.raw_entry_mut(j) += delta * r_j * mu_j;
        } else if update_i {
            let mu_i = (eta * wij).min(S::one());
            let r_i = half * (norm - dij) / norm;
            *drawing.raw_entry_mut(i) += delta * -r_i * mu_i;
        } else {
            let mu_j = (eta * wji).min(S::one());
            let r_j = half * (norm - dji) / norm;
            *drawing.raw_entry_mut(j) += delta * r_j * mu_j;
        }
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
        if !eta.is_finite() || eta <= S::zero() {
            return;
        }
        if self.hinge_pairs.is_empty() {
            for &pair in &self.node_pairs {
                Self::apply_spring_pair(drawing, eta, pair);
            }
        } else {
            for &slot in &self.schedule {
                match slot {
                    PairSlot::Spring(idx) => {
                        Self::apply_spring_pair(drawing, eta, self.node_pairs[idx])
                    }
                    PairSlot::Hinge(idx) => {
                        Self::apply_hinge_pair(drawing, eta, self.hinge_pairs[idx])
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests;
