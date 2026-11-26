use super::{stress, StressMajorization};
use petgraph_drawing::DrawingValue;

impl<S> StressMajorization<S>
where
    S: DrawingValue,
{
    fn rebuild_laplacian_and_stress(&mut self) {
        let n = self.n();
        if n <= 1 {
            self.stress = S::zero();
            return;
        }
        for i in 0..n - 1 {
            self.l_w[[i, i]] = S::zero();
        }
        for j in 1..n - 1 {
            for i in 0..j {
                let raw = self.w[[i, j]];
                let wij = if raw.is_finite() { raw } else { S::zero() };
                self.l_w[[i, j]] = -wij;
                self.l_w[[j, i]] = -wij;
                self.l_w[[i, i]] += wij;
                self.l_w[[j, j]] += wij;
            }
        }
        for i in 0..n - 1 {
            let j = n - 1;
            let raw = self.w[[i, j]];
            let wij = if raw.is_finite() { raw } else { S::zero() };
            self.l_w[[i, i]] += wij;
        }
        self.stress = stress(&self.x_x, &self.x_y, &self.w, &self.d);
    }

    /// Updates the weights used in the stress calculation.
    ///
    /// This allows customizing how different pairs of nodes contribute to the overall
    /// stress. By default, weights are set to 1/(d_ij^2) where d_ij is the desired distance.
    ///
    /// # Arguments
    ///
    /// * `weight` - A function that takes (i, j, d_ij, current_w_ij) and returns the new weight
    pub fn update_weight<F>(&mut self, mut weight: F)
    where
        F: FnMut(usize, usize, S, S) -> S,
    {
        let _ = self.try_update_weight::<std::convert::Infallible, _>(|i, j, dij, wij| {
            Ok(weight(i, j, dij, wij))
        });
    }

    /// Fallible variant of [`StressMajorization::update_weight`] propagating callback errors
    /// without partial updates.
    pub fn try_update_weight<E, F>(&mut self, mut weight: F) -> Result<(), E>
    where
        F: FnMut(usize, usize, S, S) -> Result<S, E>,
    {
        let n = self.n();
        if n <= 1 {
            self.rebuild_laplacian_and_stress();
            return Ok(());
        }

        let mut updates = Vec::with_capacity(n * (n - 1) / 2);
        for j in 1..n {
            for i in 0..j {
                let wij = weight(i, j, self.d[[i, j]], self.w[[i, j]])?;
                updates.push((i, j, wij));
            }
        }
        for (i, j, wij) in updates {
            self.w[[i, j]] = wij;
            self.w[[j, i]] = wij;
        }

        self.rebuild_laplacian_and_stress();
        Ok(())
    }

    /// Fallible variant of [`StressMajorization::update_weight_from_array`] validating shape
    /// and finite non-negative weights without partial updates.
    pub fn try_update_weight_from_array(
        &mut self,
        w: ndarray::ArrayView2<S>,
    ) -> Result<(), String> {
        let n = self.n();
        if w.dim() != (n, n) {
            return Err(format!(
                "StressMajorization::update_weight_from_array: expected shape ({}, {}), got {:?}",
                n,
                n,
                w.dim()
            ));
        }
        for j in 1..n {
            for i in 0..j {
                let wij = w[[i, j]];
                if !wij.is_finite() || wij < S::zero() {
                    return Err(format!(
                        "StressMajorization::update_weight_from_array: weight at ({}, {}) must be finite and non-negative",
                        i, j
                    ));
                }
            }
        }
        for j in 1..n {
            for i in 0..j {
                let wij = w[[i, j]];
                self.w[[i, j]] = wij;
                self.w[[j, i]] = wij;
            }
        }
        self.rebuild_laplacian_and_stress();
        Ok(())
    }

    /// Updates the weights from an `(n, n)` array and rebuilds the reduced Laplacian `l_w`.
    ///
    /// # Panics
    ///
    /// Panics if `w.dim() != (n, n)` or any off-diagonal weight is non-finite or negative.
    pub fn update_weight_from_array(&mut self, w: ndarray::ArrayView2<S>) {
        self.try_update_weight_from_array(w)
            .unwrap_or_else(|e| panic!("{}", e));
    }
}
