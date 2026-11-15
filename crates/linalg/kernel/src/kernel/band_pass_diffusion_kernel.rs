use crate::*;
use ndarray::{Array2, ScalarOperand};
use num_traits::Float;

/// Band-pass diffusion kernel sum_m c_m exp(-t_m L) computed via Chebyshev polynomial expansion.
#[derive(Debug, Clone)]
pub struct BandPassDiffusionKernel<S> {
    pub n: usize,
    pub matrix: Array2<S>,
}

impl<S: Float + ScalarOperand> Kernel<usize, S> for BandPassDiffusionKernel<S> {
    fn get(&self, u: usize, v: usize) -> Option<S> {
        if u < self.n && v < self.n {
            Some(self.matrix[[u, v]])
        } else {
            None
        }
    }

    fn get_by_index(&self, i: usize, j: usize) -> S {
        assert!(i < self.n && j < self.n, "Index out of bounds");
        self.matrix[[i, j]]
    }

    fn shape(&self) -> (usize, usize) {
        (self.n, self.n)
    }

    fn row_index(&self, u: usize) -> Option<usize> {
        Some(u).filter(|&i| i < self.n)
    }

    fn col_index(&self, u: usize) -> Option<usize> {
        Some(u).filter(|&i| i < self.n)
    }
}
