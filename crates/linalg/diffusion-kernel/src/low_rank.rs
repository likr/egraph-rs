//! Low-rank spectral approximation of diffusion and multiscale kernels.

use ndarray::{Array1, Array2, ScalarOperand};
use num_traits::Float;
use petgraph::visit::{EdgeRef, IntoEdges, IntoNodeIdentifiers, NodeCount, NodeIndexable};
use petgraph_distance::Kernel;
use petgraph_drawing::{DrawingIndex, DrawingValue};
use petgraph_linalg_rdmds::solvers::Ic0CgSolver;
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

    pub fn build_laplacian<G, F, R>(
        self,
        graph: G,
        length: F,
        rng: &mut R,
    ) -> Result<LowRankDiffusionKernel<S>, String>
    where
        G: IntoEdges + IntoNodeIdentifiers + NodeIndexable + NodeCount + Copy,
        G::NodeId: DrawingIndex,
        F: FnMut(G::EdgeRef) -> S,
        R: Rng,
    {
        let n = graph.node_count();
        let rank = self.rank.min(n.saturating_sub(1));

        let builder = Ic0CgSolver {
            max_iterations: self.cg_max_iterations,
            tolerance: self.cg_tolerance,
        };
        let mut rdmds = petgraph_linalg_rdmds::RdMds::new();
        rdmds
            .d(rank)
            .shift(self.shift)
            .eigenvalue_max_iterations(self.eigenvalue_max_iterations)
            .eigenvalue_tolerance(self.eigenvalue_tolerance);

        let result = rdmds.eigendecomposition(graph, length, &builder, rng);

        let mut eigenvalues = result.eigenvalues;
        for k in 0..rank {
            eigenvalues[k] = eigenvalues[k].max(S::zero());
        }
        let eigenvectors = result.eigenvectors;

        Ok(LowRankDiffusionKernel::new(
            self.t,
            eigenvalues,
            eigenvectors,
        ))
    }

    pub fn build_normalized_laplacian<G, F, R>(
        self,
        graph: G,
        mut length: F,
        rng: &mut R,
    ) -> Result<LowRankDiffusionKernel<S>, String>
    where
        G: IntoEdges + IntoNodeIdentifiers + NodeIndexable + NodeCount + Copy,
        G::NodeId: DrawingIndex,
        F: FnMut(G::EdgeRef) -> S,
        R: Rng,
    {
        let n = graph.node_count();
        let rank = self.rank.min(n.saturating_sub(1));

        let builder = Ic0CgSolver {
            max_iterations: self.cg_max_iterations,
            tolerance: self.cg_tolerance,
        };
        let mut rdmds = petgraph_linalg_rdmds::RdMds::new();
        rdmds
            .d(rank)
            .shift(self.shift)
            .eigenvalue_max_iterations(self.eigenvalue_max_iterations)
            .eigenvalue_tolerance(self.eigenvalue_tolerance);

        let mut degrees = Array1::<S>::zeros(n);
        let mut two_m = S::zero();
        for node in graph.node_identifiers() {
            for edge in graph.edges(node) {
                let i = graph.to_index(edge.source());
                let j = graph.to_index(edge.target());
                let w = (&mut length)(edge);
                if i != j {
                    degrees[i] = degrees[i] + w;
                    two_m = two_m + w;
                }
            }
        }

        let result = rdmds.eigendecomposition_symmetric_normalized(graph, length, &builder, rng);

        let mut eigenvalues = result.eigenvalues;
        for k in 0..rank {
            eigenvalues[k] = eigenvalues[k].max(S::zero());
        }
        let mut eigenvectors = result.eigenvectors;

        if two_m > S::zero() {
            for i in 0..n {
                let stat = (degrees[i] / two_m).sqrt();
                if stat > S::zero() {
                    let mut row = eigenvectors.row_mut(i);
                    for j in 0..rank {
                        row[j] = row[j] / stat;
                    }
                }
            }
        } else if n > 0 {
            let inv_sqrt_n = S::one() / S::from_usize(n).unwrap().sqrt();
            for i in 0..n {
                let mut row = eigenvectors.row_mut(i);
                for j in 0..rank {
                    row[j] = row[j] / inv_sqrt_n;
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

impl<S: Float + ScalarOperand + num_traits::FromPrimitive> Kernel<usize, S> for LowRankDiffusionKernel<S> {
    fn get(&self, u: usize, v: usize) -> Option<S> {
        let n = self.eigenvectors.nrows();
        if u < n && v < n {
            Some(self.get_by_index(u, v))
        } else {
            None
        }
    }

    fn get_by_index(&self, i: usize, j: usize) -> S {
        let n = self.eigenvectors.nrows();
        assert!(i < n && j < n, "Index out of bounds");
        let mut sum = S::one() / S::from_usize(n).unwrap();
        for k in 0..self.coefficients.len() {
            sum =
                sum + self.coefficients[k] * self.eigenvectors[[i, k]] * self.eigenvectors[[j, k]];
        }
        sum
    }

    fn shape(&self) -> (usize, usize) {
        let n = self.eigenvectors.nrows();
        (n, n)
    }

    fn row_index(&self, u: usize) -> Option<usize> {
        Some(u).filter(|&i| i < self.eigenvectors.nrows())
    }

    fn col_index(&self, v: usize) -> Option<usize> {
        Some(v).filter(|&j| j < self.eigenvectors.nrows())
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

    pub fn build_laplacian<G, F, R>(
        self,
        graph: G,
        length: F,
        rng: &mut R,
    ) -> Result<LowRankMultiscaleDiffusionKernel<S>, String>
    where
        G: IntoEdges + IntoNodeIdentifiers + NodeIndexable + NodeCount + Copy,
        G::NodeId: DrawingIndex,
        F: FnMut(G::EdgeRef) -> S,
        R: Rng,
    {
        let n = graph.node_count();
        let rank = self.rank.min(n.saturating_sub(1));

        let builder = Ic0CgSolver {
            max_iterations: self.cg_max_iterations,
            tolerance: self.cg_tolerance,
        };
        let mut rdmds = petgraph_linalg_rdmds::RdMds::new();
        rdmds
            .d(rank)
            .shift(self.shift)
            .eigenvalue_max_iterations(self.eigenvalue_max_iterations)
            .eigenvalue_tolerance(self.eigenvalue_tolerance);

        let result = rdmds.eigendecomposition(graph, length, &builder, rng);

        let mut eigenvalues = result.eigenvalues;
        for k in 0..rank {
            eigenvalues[k] = eigenvalues[k].max(S::zero());
        }
        let eigenvectors = result.eigenvectors;

        Ok(LowRankMultiscaleDiffusionKernel::new(
            self.alpha,
            eigenvalues,
            eigenvectors,
        ))
    }

    pub fn build_normalized_laplacian<G, F, R>(
        self,
        graph: G,
        mut length: F,
        rng: &mut R,
    ) -> Result<LowRankMultiscaleDiffusionKernel<S>, String>
    where
        G: IntoEdges + IntoNodeIdentifiers + NodeIndexable + NodeCount + Copy,
        G::NodeId: DrawingIndex,
        F: FnMut(G::EdgeRef) -> S,
        R: Rng,
    {
        let n = graph.node_count();
        let rank = self.rank.min(n.saturating_sub(1));

        let builder = Ic0CgSolver {
            max_iterations: self.cg_max_iterations,
            tolerance: self.cg_tolerance,
        };
        let mut rdmds = petgraph_linalg_rdmds::RdMds::new();
        rdmds
            .d(rank)
            .shift(self.shift)
            .eigenvalue_max_iterations(self.eigenvalue_max_iterations)
            .eigenvalue_tolerance(self.eigenvalue_tolerance);

        let mut degrees = Array1::<S>::zeros(n);
        let mut two_m = S::zero();
        for node in graph.node_identifiers() {
            for edge in graph.edges(node) {
                let i = graph.to_index(edge.source());
                let j = graph.to_index(edge.target());
                let w = (&mut length)(edge);
                if i != j {
                    degrees[i] = degrees[i] + w;
                    two_m = two_m + w;
                }
            }
        }

        let result = rdmds.eigendecomposition_symmetric_normalized(graph, length, &builder, rng);

        let mut eigenvalues = result.eigenvalues;
        for k in 0..rank {
            eigenvalues[k] = eigenvalues[k].max(S::zero());
        }
        let mut eigenvectors = result.eigenvectors;

        if two_m > S::zero() {
            for i in 0..n {
                let stat = (degrees[i] / two_m).sqrt();
                if stat > S::zero() {
                    let mut row = eigenvectors.row_mut(i);
                    for j in 0..rank {
                        row[j] = row[j] / stat;
                    }
                }
            }
        } else if n > 0 {
            let inv_sqrt_n = S::one() / S::from_usize(n).unwrap().sqrt();
            for i in 0..n {
                let mut row = eigenvectors.row_mut(i);
                for j in 0..rank {
                    row[j] = row[j] / inv_sqrt_n;
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
    alpha: S,
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
            alpha,
            eigenvectors,
            coefficients,
        }
    }
}

impl<S: Float + ScalarOperand + num_traits::FromPrimitive> Kernel<usize, S>
    for LowRankMultiscaleDiffusionKernel<S>
{
    fn get(&self, u: usize, v: usize) -> Option<S> {
        let n = self.eigenvectors.nrows();
        if u < n && v < n {
            Some(self.get_by_index(u, v))
        } else {
            None
        }
    }

    fn get_by_index(&self, i: usize, j: usize) -> S {
        let n = self.eigenvectors.nrows();
        assert!(i < n && j < n, "Index out of bounds");
        let mut sum = S::one() / (S::from_usize(n).unwrap() * (S::one() - self.alpha));
        for k in 0..self.coefficients.len() {
            sum =
                sum + self.coefficients[k] * self.eigenvectors[[i, k]] * self.eigenvectors[[j, k]];
        }
        sum
    }

    fn shape(&self) -> (usize, usize) {
        let n = self.eigenvectors.nrows();
        (n, n)
    }

    fn row_index(&self, u: usize) -> Option<usize> {
        Some(u).filter(|&i| i < self.eigenvectors.nrows())
    }

    fn col_index(&self, v: usize) -> Option<usize> {
        Some(v).filter(|&j| j < self.eigenvectors.nrows())
    }
}
