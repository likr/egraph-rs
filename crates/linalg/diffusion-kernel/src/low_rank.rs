//! Low-rank spectral approximation of diffusion and multiscale kernels.

use ndarray::{Array1, Array2, ScalarOperand};
use num_traits::Float;
use petgraph_distance::{Kernel, SparseSymmetricMatrix};
use petgraph_drawing::DrawingValue;
use petgraph_linalg_rdmds::compute_smallest_eigenvalues;
use rand::Rng;

/// Builder for LowRankDiffusionKernel
#[derive(Clone)]
pub struct LowRankDiffusionKernelBuilder<S> {
    t: S,
    rank: usize,
    shift: S,
    eigenvalue_max_iterations: usize,
    cg_max_iterations: usize,
    eigenvalue_tolerance: S,
    cg_tolerance: S,
}

impl<S> LowRankDiffusionKernelBuilder<S>
where
    S: Float
        + std::iter::Sum
        + std::ops::AddAssign
        + Default
        + ScalarOperand
        + num_traits::FromPrimitive
        + DrawingValue,
{
    #[allow(clippy::new_without_default)]
    pub fn new() -> Self {
        Self {
            t: S::one(),
            rank: 10,
            shift: S::from_f64(1e-3).unwrap(),
            eigenvalue_max_iterations: 1000,
            cg_max_iterations: 100,
            eigenvalue_tolerance: S::from_f64(1e-2).unwrap(),
            cg_tolerance: S::from_f64(1e-4).unwrap(),
        }
    }

    pub fn t(mut self, t: S) -> Self {
        self.t = t;
        self
    }

    pub fn rank(mut self, rank: usize) -> Self {
        self.rank = rank;
        self
    }

    pub fn shift(mut self, shift: S) -> Self {
        self.shift = shift;
        self
    }

    pub fn eigenvalue_max_iterations(mut self, max_iterations: usize) -> Self {
        self.eigenvalue_max_iterations = max_iterations;
        self
    }

    pub fn cg_max_iterations(mut self, cg_max_iterations: usize) -> Self {
        self.cg_max_iterations = cg_max_iterations;
        self
    }

    pub fn eigenvalue_tolerance(mut self, tolerance: S) -> Self {
        self.eigenvalue_tolerance = tolerance;
        self
    }

    pub fn cg_tolerance(mut self, cg_tolerance: S) -> Self {
        self.cg_tolerance = cg_tolerance;
        self
    }

    pub fn build_unnormalized_laplacian<R: Rng>(
        self,
        laplacian: &SparseSymmetricMatrix<S>,
        rng: &mut R,
    ) -> Result<LowRankDiffusionKernel<S>, String> {
        let n = laplacian.dim();
        let rank = self.rank.min(n.saturating_sub(1));
        let shifted_laplacian = laplacian.scale_and_shift(S::one(), -self.shift);

        let solver = petgraph_linalg_rdmds::solvers::Ic0CgSolver { max_iterations: self.cg_max_iterations, tolerance: self.cg_tolerance };
        let result = compute_smallest_eigenvalues(
            &shifted_laplacian,
            rank,
            self.eigenvalue_max_iterations,
            self.eigenvalue_tolerance,
            &solver,
            rng,
        );
        let all_eigenvalues = result.eigenvalues;
        let all_eigenvectors = result.eigenvectors;

        let mut eigenvalues = Array1::zeros(rank + 1);
        let mut eigenvectors = Array2::zeros((n, rank + 1));

        for k in 0..=rank {
            let lambda_est = (all_eigenvalues[k] - self.shift).max(S::zero());
            eigenvalues[k] = lambda_est;
            eigenvectors
                .column_mut(k)
                .assign(&all_eigenvectors.column(k));
        }

        Ok(LowRankDiffusionKernel::new(
            self.t,
            eigenvalues,
            eigenvectors,
        ))
    }

    pub fn build_symmetric_normalized_laplacian<R: Rng>(
        self,
        laplacian: &SparseSymmetricMatrix<S>,
        rng: &mut R,
    ) -> Result<LowRankDiffusionKernel<S>, String> {
        let n = laplacian.dim();
        let rank = self.rank.min(n.saturating_sub(1));
        let shifted_laplacian = laplacian.scale_and_shift(S::one(), -self.shift);

        let solver = petgraph_linalg_rdmds::solvers::Ic0CgSolver { max_iterations: self.cg_max_iterations, tolerance: self.cg_tolerance };
        let result = compute_smallest_eigenvalues(
            &shifted_laplacian,
            rank,
            self.eigenvalue_max_iterations,
            self.eigenvalue_tolerance,
            &solver,
            rng,
        );
        let all_eigenvalues = result.eigenvalues;
        let all_eigenvectors = result.eigenvectors;

        let mut eigenvalues = Array1::zeros(rank + 1);
        let mut eigenvectors = Array2::zeros((n, rank + 1));

        for k in 0..=rank {
            let lambda_est = (all_eigenvalues[k] - self.shift).max(S::zero());
            eigenvalues[k] = lambda_est;
            eigenvectors
                .column_mut(k)
                .assign(&all_eigenvectors.column(k));
        }

        let stationary = laplacian.stationary_vector();
        if stationary.len() == n {
            for i in 0..n {
                let stat = stationary[i];
                if stat > S::zero() {
                    let mut row = eigenvectors.row_mut(i);
                    for j in 0..=rank {
                        row[j] = row[j] / stat;
                    }
                }
            }
        }

        Ok(LowRankDiffusionKernel::new(
            self.t,
            eigenvalues,
            eigenvectors,
        ))
    }
}

/// Low-rank spectral approximation of the heat kernel exp(-tL).
#[derive(Debug, Clone)]
pub struct LowRankDiffusionKernel<S> {
    eigenvectors: Array2<S>,
    coefficients: Array1<S>,
}

impl<S: Float> LowRankDiffusionKernel<S> {
    pub fn new(t: S, eigenvalues: Array1<S>, eigenvectors: Array2<S>) -> Self {
        let rank_plus_1 = eigenvalues.len();
        let mut coefficients = Array1::zeros(rank_plus_1);
        for k in 0..rank_plus_1 {
            coefficients[k] = (-t * eigenvalues[k]).exp();
        }
        Self {
            eigenvectors,
            coefficients,
        }
    }
}

impl<S: Float + ScalarOperand> Kernel<S> for LowRankDiffusionKernel<S> {
    fn get(&self, i: usize, j: usize) -> S {
        let n = self.n();
        assert!(i < n && j < n, "Index out of bounds");
        let mut sum = S::zero();
        for k in 0..self.coefficients.len() {
            sum =
                sum + self.coefficients[k] * self.eigenvectors[[i, k]] * self.eigenvectors[[j, k]];
        }
        sum
    }

    fn n(&self) -> usize {
        self.eigenvectors.nrows()
    }
}

/// Builder for LowRankMultiscaleDiffusionKernel
#[derive(Clone)]
pub struct LowRankMultiscaleDiffusionKernelBuilder<S> {
    alpha: S,
    rank: usize,
    shift: S,
    eigenvalue_max_iterations: usize,
    cg_max_iterations: usize,
    eigenvalue_tolerance: S,
    cg_tolerance: S,
}

impl<S> LowRankMultiscaleDiffusionKernelBuilder<S>
where
    S: Float
        + std::iter::Sum
        + std::ops::AddAssign
        + Default
        + ScalarOperand
        + num_traits::FromPrimitive
        + DrawingValue,
{
    #[allow(clippy::new_without_default)]
    pub fn new() -> Self {
        Self {
            alpha: S::from_f64(0.85).unwrap(),
            rank: 10,
            shift: S::from_f64(1e-3).unwrap(),
            eigenvalue_max_iterations: 1000,
            cg_max_iterations: 100,
            eigenvalue_tolerance: S::from_f64(1e-2).unwrap(),
            cg_tolerance: S::from_f64(1e-4).unwrap(),
        }
    }

    pub fn alpha(mut self, alpha: S) -> Self {
        self.alpha = alpha;
        self
    }

    pub fn rank(mut self, rank: usize) -> Self {
        self.rank = rank;
        self
    }

    pub fn shift(mut self, shift: S) -> Self {
        self.shift = shift;
        self
    }

    pub fn eigenvalue_max_iterations(mut self, max_iterations: usize) -> Self {
        self.eigenvalue_max_iterations = max_iterations;
        self
    }

    pub fn cg_max_iterations(mut self, cg_max_iterations: usize) -> Self {
        self.cg_max_iterations = cg_max_iterations;
        self
    }

    pub fn eigenvalue_tolerance(mut self, tolerance: S) -> Self {
        self.eigenvalue_tolerance = tolerance;
        self
    }

    pub fn cg_tolerance(mut self, cg_tolerance: S) -> Self {
        self.cg_tolerance = cg_tolerance;
        self
    }

    pub fn build_unnormalized_laplacian<R: Rng>(
        self,
        laplacian: &SparseSymmetricMatrix<S>,
        rng: &mut R,
    ) -> Result<LowRankMultiscaleDiffusionKernel<S>, String> {
        let n = laplacian.dim();
        let rank = self.rank.min(n.saturating_sub(1));
        let shifted_laplacian = laplacian.scale_and_shift(S::one(), -self.shift);

        let solver = petgraph_linalg_rdmds::solvers::Ic0CgSolver { max_iterations: self.cg_max_iterations, tolerance: self.cg_tolerance };
        let result = compute_smallest_eigenvalues(
            &shifted_laplacian,
            rank,
            self.eigenvalue_max_iterations,
            self.eigenvalue_tolerance,
            &solver,
            rng,
        );
        let all_eigenvalues = result.eigenvalues;
        let all_eigenvectors = result.eigenvectors;

        let mut eigenvalues = Array1::zeros(rank + 1);
        let mut eigenvectors = Array2::zeros((n, rank + 1));

        for k in 0..=rank {
            let lambda_est = (all_eigenvalues[k] - self.shift).max(S::zero());
            eigenvalues[k] = lambda_est;
            eigenvectors
                .column_mut(k)
                .assign(&all_eigenvectors.column(k));
        }

        Ok(LowRankMultiscaleDiffusionKernel::new(
            self.alpha,
            eigenvalues,
            eigenvectors,
        ))
    }

    pub fn build_symmetric_normalized_laplacian<R: Rng>(
        self,
        laplacian: &SparseSymmetricMatrix<S>,
        rng: &mut R,
    ) -> Result<LowRankMultiscaleDiffusionKernel<S>, String> {
        let n = laplacian.dim();
        let rank = self.rank.min(n.saturating_sub(1));
        let shifted_laplacian = laplacian.scale_and_shift(S::one(), -self.shift);

        let solver = petgraph_linalg_rdmds::solvers::Ic0CgSolver { max_iterations: self.cg_max_iterations, tolerance: self.cg_tolerance };
        let result = compute_smallest_eigenvalues(
            &shifted_laplacian,
            rank,
            self.eigenvalue_max_iterations,
            self.eigenvalue_tolerance,
            &solver,
            rng,
        );
        let all_eigenvalues = result.eigenvalues;
        let all_eigenvectors = result.eigenvectors;

        let mut eigenvalues = Array1::zeros(rank + 1);
        let mut eigenvectors = Array2::zeros((n, rank + 1));

        for k in 0..=rank {
            let lambda_est = (all_eigenvalues[k] - self.shift).max(S::zero());
            eigenvalues[k] = lambda_est;
            eigenvectors
                .column_mut(k)
                .assign(&all_eigenvectors.column(k));
        }

        let stationary = laplacian.stationary_vector();
        if stationary.len() == n {
            for i in 0..n {
                let stat = stationary[i];
                if stat > S::zero() {
                    let mut row = eigenvectors.row_mut(i);
                    for j in 0..=rank {
                        row[j] = row[j] / stat;
                    }
                }
            }
        }

        Ok(LowRankMultiscaleDiffusionKernel::new(
            self.alpha,
            eigenvalues,
            eigenvectors,
        ))
    }
}

/// Low-rank spectral approximation of the multiscale diffusion kernel (I - alpha P)^{-1}.
#[derive(Debug, Clone)]
pub struct LowRankMultiscaleDiffusionKernel<S> {
    eigenvectors: Array2<S>,
    coefficients: Array1<S>,
}

impl<S: Float> LowRankMultiscaleDiffusionKernel<S> {
    pub fn new(alpha: S, eigenvalues: Array1<S>, eigenvectors: Array2<S>) -> Self {
        let rank_plus_1 = eigenvalues.len();
        let mut coefficients = Array1::zeros(rank_plus_1);
        let one = S::one();
        for k in 0..rank_plus_1 {
            coefficients[k] = one / ((one - alpha) + alpha * eigenvalues[k]);
        }
        Self {
            eigenvectors,
            coefficients,
        }
    }
}

impl<S: Float + ScalarOperand> Kernel<S> for LowRankMultiscaleDiffusionKernel<S> {
    fn get(&self, i: usize, j: usize) -> S {
        let n = self.n();
        assert!(i < n && j < n, "Index out of bounds");
        let mut sum = S::zero();
        for k in 0..self.coefficients.len() {
            sum =
                sum + self.coefficients[k] * self.eigenvectors[[i, k]] * self.eigenvectors[[j, k]];
        }
        sum
    }

    fn n(&self) -> usize {
        self.eigenvectors.nrows()
    }
}
