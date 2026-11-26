use super::Sgd;
use ndarray::ArrayView2;
use petgraph_drawing::DrawingValue;

impl<S> Sgd<S>
where
    S: DrawingValue,
{
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
