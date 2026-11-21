use ndarray::ArrayView2;
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
        for p in self
            .node_pairs
            .iter_mut()
            .chain(self.hinge_pairs.iter_mut())
        {
            let (i, j, dij, dji, wij, wji) = *p;
            p.2 = distance(i, j, dij, wij);
            p.3 = distance(j, i, dji, wji);
        }
    }

    /// Fallible variant of [`Sgd::update_distance`] propagating errors without partial updates.
    pub fn try_update_distance<E, F>(&mut self, mut distance: F) -> Result<(), E>
    where
        F: FnMut(usize, usize, S, S) -> Result<S, E>,
        S: Copy,
    {
        let mut updates = Vec::with_capacity(self.node_pairs.len() + self.hinge_pairs.len());
        for p in self.node_pairs.iter().chain(self.hinge_pairs.iter()) {
            let (i, j, dij, dji, wij, wji) = *p;
            let new_dij = distance(i, j, dij, wij)?;
            let new_dji = distance(j, i, dji, wji)?;
            updates.push((new_dij, new_dji));
        }
        for (p, (new_dij, new_dji)) in self
            .node_pairs
            .iter_mut()
            .chain(self.hinge_pairs.iter_mut())
            .zip(updates)
        {
            p.2 = new_dij;
            p.3 = new_dji;
        }
        Ok(())
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
        for p in self
            .node_pairs
            .iter_mut()
            .chain(self.hinge_pairs.iter_mut())
        {
            let (i, j, dij, dji, wij, wji) = *p;
            p.4 = weight(i, j, dij, wij);
            p.5 = weight(j, i, dji, wji);
        }
    }

    /// Fallible variant of [`Sgd::update_weight`] propagating errors without partial updates.
    pub fn try_update_weight<E, F>(&mut self, mut weight: F) -> Result<(), E>
    where
        F: FnMut(usize, usize, S, S) -> Result<S, E>,
        S: Copy,
    {
        let mut updates = Vec::with_capacity(self.node_pairs.len() + self.hinge_pairs.len());
        for p in self.node_pairs.iter().chain(self.hinge_pairs.iter()) {
            let (i, j, dij, dji, wij, wji) = *p;
            let new_wij = weight(i, j, dij, wij)?;
            let new_wji = weight(j, i, dji, wji)?;
            updates.push((new_wij, new_wji));
        }
        for (p, (new_wij, new_wji)) in self
            .node_pairs
            .iter_mut()
            .chain(self.hinge_pairs.iter_mut())
            .zip(updates)
        {
            p.4 = new_wij;
            p.5 = new_wji;
        }
        Ok(())
    }

    /// Validates that all spring and hinge pair indices are strictly less than `n`.
    pub fn try_validate_bounds(&self, n: usize) -> Result<(), String> {
        for &(i, j, _, _, _, _) in self.node_pairs.iter().chain(self.hinge_pairs.iter()) {
            if i >= n || j >= n {
                return Err(format!("Sgd: pair ({i}, {j}) out of bounds for size {n}"));
            }
        }
        Ok(())
    }

    /// Updates weights for all node pairs from an `(n, n)` weight matrix, returning an error on shape or bounds mismatch.
    pub fn try_update_weight_from_array(&mut self, w: ArrayView2<S>) -> Result<(), String> {
        let (rows, cols) = w.dim();
        if rows != cols {
            return Err(format!(
                "Sgd::update_weight_from_array: expected square matrix, got ({rows}, {cols})"
            ));
        }
        self.try_validate_bounds(rows)?;
        for p in self
            .node_pairs
            .iter_mut()
            .chain(self.hinge_pairs.iter_mut())
        {
            p.4 = w[[p.0, p.1]];
            p.5 = w[[p.1, p.0]];
        }
        Ok(())
    }

    /// Updates weights for all node pairs from an `(n, n)` weight matrix.
    ///
    /// Sets `pair.4 = w[[pair.0, pair.1]]` and `pair.5 = w[[pair.1, pair.0]]`.
    pub fn update_weight_from_array(&mut self, w: ArrayView2<S>) {
        self.try_update_weight_from_array(w)
            .unwrap_or_else(|e| panic!("{e}"));
    }

    /// Updates target distances (and optionally weights) from `(n, n)` matrices, returning an error on shape or bounds mismatch.
    pub fn try_update_distance_from_array(
        &mut self,
        d: ArrayView2<S>,
        w: Option<ArrayView2<S>>,
    ) -> Result<(), String> {
        let (rows, cols) = d.dim();
        if rows != cols {
            return Err(format!(
                "Sgd::update_distance_from_array: expected square distance matrix, got ({rows}, {cols})"
            ));
        }
        if let Some(w_arr) = w {
            if w_arr.dim() != (rows, cols) {
                return Err(format!(
                    "Sgd::update_distance_from_array: weight matrix shape {:?} does not match distance matrix shape ({rows}, {cols})",
                    w_arr.dim()
                ));
            }
        }
        self.try_validate_bounds(rows)?;
        for p in self
            .node_pairs
            .iter_mut()
            .chain(self.hinge_pairs.iter_mut())
        {
            p.2 = d[[p.0, p.1]];
            p.3 = d[[p.1, p.0]];
            if let Some(w_arr) = w {
                p.4 = w_arr[[p.0, p.1]];
                p.5 = w_arr[[p.1, p.0]];
            }
        }
        Ok(())
    }

    /// Updates target distances (and optionally weights) from `(n, n)` matrices.
    ///
    /// Sets `pair.2 = d[[pair.0, pair.1]]`, `pair.3 = d[[pair.1, pair.0]]`,
    /// and when `w` is `Some`, updates `pair.4 = w[[pair.0, pair.1]]` and `pair.5 = w[[pair.1, pair.0]]`.
    pub fn update_distance_from_array(&mut self, d: ArrayView2<S>, w: Option<ArrayView2<S>>) {
        self.try_update_distance_from_array(d, w)
            .unwrap_or_else(|e| panic!("{e}"));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr2;
    use petgraph::graph::UnGraph;
    use petgraph_drawing::DrawingEuclidean2d;

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

    #[test]
    fn test_sgd_hinge_zero_force_when_separated_and_repels_when_too_close() {
        let mut g = UnGraph::<(), ()>::new_undirected();
        let u = g.add_node(());
        let v = g.add_node(());

        // Case 1: ||x_0 - x_1|| = 3.0 >= d_target = 2.0 -> exact zero displacement
        let mut drawing_far = DrawingEuclidean2d::<_, f64>::initial_placement(&g);
        drawing_far.raw_entry_mut(0).0 = 0.0;
        drawing_far.raw_entry_mut(0).1 = 0.0;
        drawing_far.raw_entry_mut(1).0 = 3.0;
        drawing_far.raw_entry_mut(1).1 = 0.0;
        let sgd = Sgd::new_with_hinge_pairs(vec![], vec![(0, 1, 2.0, 2.0, 1.0, 1.0)]);
        assert_eq!(sgd.total_pair_count(), 1);
        assert!(sgd.node_pairs().is_empty());
        assert_eq!(sgd.hinge_pairs().len(), 1);
        sgd.apply(&mut drawing_far, 0.5);
        assert_eq!(drawing_far.x(u).unwrap().to_bits(), 0.0_f64.to_bits());
        assert_eq!(drawing_far.y(u).unwrap().to_bits(), 0.0_f64.to_bits());
        assert_eq!(drawing_far.x(v).unwrap().to_bits(), 3.0_f64.to_bits());
        assert_eq!(drawing_far.y(v).unwrap().to_bits(), 0.0_f64.to_bits());

        // Case 2: Exact boundary ||x_0 - x_1|| == d_target -> exact zero displacement
        drawing_far.raw_entry_mut(1).0 = 2.0;
        sgd.apply(&mut drawing_far, 0.5);
        assert_eq!(drawing_far.x(u).unwrap().to_bits(), 0.0_f64.to_bits());
        assert_eq!(drawing_far.x(v).unwrap().to_bits(), 2.0_f64.to_bits());

        // Case 3: ||x_0 - x_1|| = 0.5 < d_target = 2.0 -> repels symmetrically outward
        let mut drawing_close = DrawingEuclidean2d::<_, f64>::initial_placement(&g);
        drawing_close.raw_entry_mut(0).0 = 0.0;
        drawing_close.raw_entry_mut(0).1 = 0.0;
        drawing_close.raw_entry_mut(1).0 = 0.5;
        drawing_close.raw_entry_mut(1).1 = 0.0;
        sgd.apply(&mut drawing_close, 0.5);
        assert!(drawing_close.x(u).unwrap() < 0.0);
        assert!(drawing_close.x(v).unwrap() > 0.5);
        assert_eq!(drawing_close.y(u).unwrap().to_bits(), 0.0_f64.to_bits());
        assert_eq!(drawing_close.y(v).unwrap().to_bits(), 0.0_f64.to_bits());

        // Case 4: Asymmetric hinge (wij = 0, wji = 1) moves only j
        let mut drawing_asym = DrawingEuclidean2d::<_, f64>::initial_placement(&g);
        drawing_asym.raw_entry_mut(0).0 = 0.0;
        drawing_asym.raw_entry_mut(0).1 = 0.0;
        drawing_asym.raw_entry_mut(1).0 = 0.5;
        drawing_asym.raw_entry_mut(1).1 = 0.0;
        let sgd_asym = Sgd::new_with_hinge_pairs(vec![], vec![(0, 1, 2.0, 2.0, 0.0, 1.0)]);
        sgd_asym.apply(&mut drawing_asym, 0.5);
        assert_eq!(drawing_asym.x(u).unwrap().to_bits(), 0.0_f64.to_bits());
        assert!(drawing_asym.x(v).unwrap() > 0.5);
    }

    #[test]
    fn test_sgd_mixed_spring_and_hinge_behavior_and_eta_bounds() {
        let mut g = UnGraph::<(), ()>::new_undirected();
        let n0 = g.add_node(());
        let n1 = g.add_node(());
        let n2 = g.add_node(());

        let mut drawing = DrawingEuclidean2d::<_, f64>::initial_placement(&g);
        // (0, 1) is stretched spring (dist 4.0 > target 1.0 -> pulls inward)
        // (0, 2) is separated hinge (dist 5.0 > target 2.0 -> exact zero force on node 2)
        drawing.raw_entry_mut(0).0 = 0.0;
        drawing.raw_entry_mut(0).1 = 0.0;
        drawing.raw_entry_mut(1).0 = 4.0;
        drawing.raw_entry_mut(1).1 = 0.0;
        drawing.raw_entry_mut(2).0 = 0.0;
        drawing.raw_entry_mut(2).1 = 5.0;

        let mut sgd = Sgd::new(vec![(0, 1, 1.0, 1.0, 1.0, 1.0)]);
        sgd.add_hinge_pairs(vec![(0, 2, 2.0, 2.0, 0.25, 0.25)]);
        assert_eq!(sgd.total_pair_count(), 2);

        let (eta_min, eta_max) = sgd.calculate_eta_bounds(0.01_f64);
        assert!((eta_min - 0.01_f64).abs() < 1e-12);
        assert!((eta_max - 4.0_f64).abs() < 1e-12);

        let mut rng = StdRng::seed_from_u64(42);
        sgd.shuffle(&mut rng);
        sgd.apply(&mut drawing, 0.5);

        assert!(drawing.x(n0).unwrap() > 0.0);
        assert!(drawing.x(n1).unwrap() < 4.0);
        assert_eq!(drawing.x(n2).unwrap().to_bits(), 0.0_f64.to_bits());
        assert_eq!(drawing.y(n2).unwrap().to_bits(), 5.0_f64.to_bits());

        // Guard when no positive finite weights exist
        let empty_sgd = Sgd::<f64>::new(vec![]);
        assert_eq!(empty_sgd.calculate_eta_bounds(0.01), (0.0, 0.0));
        let zero_w = Sgd::new(vec![(0, 1, 1.0_f64, 1.0, 0.0, 0.0)]);
        assert_eq!(zero_w.calculate_eta_bounds(0.01), (0.0, 0.0));
    }

    #[test]
    fn test_sgd_schedulers_empty_pairs_and_small_tmax_finite() {
        use crate::{
            Scheduler, SchedulerConstant, SchedulerExponential, SchedulerLinear,
            SchedulerQuadratic, SchedulerReciprocal,
        };

        let empty_sgd = Sgd::<f64>::new(vec![]);
        for t_max in [0usize, 1, 5] {
            let mut c: SchedulerConstant<f64> = empty_sgd.scheduler(t_max, 0.01);
            let mut l: SchedulerLinear<f64> = empty_sgd.scheduler(t_max, 0.01);
            let mut q: SchedulerQuadratic<f64> = empty_sgd.scheduler(t_max, 0.01);
            let mut e: SchedulerExponential<f64> = empty_sgd.scheduler(t_max, 0.01);
            let mut r: SchedulerReciprocal<f64> = empty_sgd.scheduler(t_max, 0.01);
            c.run(&mut |eta| assert!(eta.is_finite()));
            l.run(&mut |eta| assert!(eta.is_finite()));
            q.run(&mut |eta| assert!(eta.is_finite()));
            e.run(&mut |eta| assert!(eta.is_finite()));
            r.run(&mut |eta| assert!(eta.is_finite()));
        }

        // t_max=1 on non-empty Sgd with SchedulerLinear must not emit NaN
        let sgd = Sgd::new(vec![(0, 1, 1.0_f64, 1.0, 1.0, 1.0)]);
        let mut lin1: SchedulerLinear<f64> = sgd.scheduler(1, 0.01);
        let mut seen = Vec::new();
        lin1.run(&mut |eta| seen.push(eta));
        assert_eq!(seen.len(), 1);
        assert!(seen[0].is_finite());

        // apply(drawing, NaN) must be a no-op
        let mut g = UnGraph::<(), ()>::new_undirected();
        let u = g.add_node(());
        let v = g.add_node(());
        let mut d = DrawingEuclidean2d::<_, f64>::initial_placement(&g);
        d.raw_entry_mut(0).0 = 0.0;
        d.raw_entry_mut(1).0 = 3.0;
        sgd.apply(&mut d, f64::NAN);
        assert_eq!(d.x(u).unwrap(), 0.0);
        assert_eq!(d.x(v).unwrap(), 3.0);
    }
}
