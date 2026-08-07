//! Low-rank spectral approximation of the heat diffusion kernel matrix and distance matrix.
//!
//! Computes heat kernel matrix elements using the spectral expansion:
//! $$K_{ij}^{(r)} = \sum_{k=0}^{r} \exp(-t \lambda_k) u_{k,i} u_{k,j}$$
//! where $\lambda_k$ and $u_k$ are the smallest eigenvalues and orthonormal eigenvectors
//! of the graph Laplacian matrix (Standard or Symmetric Normalized).

use crate::heat_kernel::HeatKernel;
use ndarray::{Array1, Array2, ScalarOperand};
use num_traits::Float;
use petgraph::visit::IntoNodeIdentifiers;
use petgraph_distance::{Distance, SparseSymmetricMatrix};
use petgraph_drawing::DrawingValue;
use petgraph_linalg_rdmds::compute_smallest_eigenvalues;
use rand::Rng;
use std::collections::HashMap;
use std::hash::Hash;

/// Low-rank spectral approximation of the heat kernel exp(-tL).
#[derive(Debug, Clone)]
pub struct LowRankDiffusionKernel<S> {
    n: usize,
    t: S,
    rank: usize,
    eigenvalues: Array1<S>,
    eigenvectors: Array2<S>,
    coefficients: Array1<S>,
    eta: S,
}

impl<S> LowRankDiffusionKernel<S>
where
    S: Float
        + std::iter::Sum
        + std::ops::AddAssign
        + Default
        + ScalarOperand
        + num_traits::FromPrimitive
        + DrawingValue,
{
    /// Creates a new LowRankDiffusionKernel by computing the smallest r eigenvalues and eigenvectors
    /// of the given Laplacian matrix (Standard or Symmetric Normalized) with default eta = 0.0.
    pub fn new<R: Rng>(
        laplacian: &SparseSymmetricMatrix<S>,
        t: S,
        rank: usize,
        rng: &mut R,
    ) -> Self {
        Self::new_with_eta(laplacian, t, rank, S::zero(), rng)
    }

    /// Creates a new LowRankDiffusionKernel with a specified eta parameter.
    pub fn new_with_eta<R: Rng>(
        laplacian: &SparseSymmetricMatrix<S>,
        t: S,
        rank: usize,
        eta: S,
        rng: &mut R,
    ) -> Self {
        let shift = S::from_f64(1e-3).unwrap();
        Self::new_with_params(
            laplacian,
            t,
            rank,
            shift,
            1000,
            100,
            S::from_f64(1e-2).unwrap(),
            S::from_f64(1e-4).unwrap(),
            eta,
            rng,
        )
    }

    /// Creates a LowRankDiffusionKernel with custom eigensolver iteration, tolerance, and eta parameters.
    #[allow(clippy::too_many_arguments)]
    pub fn new_with_params<R: Rng>(
        laplacian: &SparseSymmetricMatrix<S>,
        t: S,
        rank: usize,
        shift: S,
        eigenvalue_max_iterations: usize,
        cg_max_iterations: usize,
        eigenvalue_tolerance: S,
        cg_tolerance: S,
        eta: S,
        rng: &mut R,
    ) -> Self {
        let n = laplacian.dim();
        let rank = rank.min(n.saturating_sub(1));

        // Shift Laplacian L + shift * I to ensure positive definiteness for CG solver
        let shifted_laplacian = laplacian.scale_and_shift(S::one(), -shift);

        let (all_eigenvalues, all_eigenvectors) = compute_smallest_eigenvalues(
            &shifted_laplacian,
            rank,
            eigenvalue_max_iterations,
            cg_max_iterations,
            eigenvalue_tolerance,
            cg_tolerance,
            rng,
        );

        let mut eigenvalues = Array1::zeros(rank + 1);
        let mut eigenvectors = Array2::zeros((n, rank + 1));
        let mut coefficients = Array1::zeros(rank + 1);

        for k in 0..=rank {
            let lambda_est = (all_eigenvalues[k] - shift).max(S::zero());
            eigenvalues[k] = lambda_est;
            eigenvectors
                .column_mut(k)
                .assign(&all_eigenvectors.column(k));
            coefficients[k] = (-t * lambda_est).exp();
        }

        Self {
            n,
            t,
            rank,
            eigenvalues,
            eigenvectors,
            coefficients,
            eta,
        }
    }

    /// Queries the (i, j) element of the low-rank approximated heat kernel matrix.
    pub fn get(&self, i: usize, j: usize) -> S {
        assert!(i < self.n && j < self.n, "Index out of bounds");
        let mut sum = S::zero();
        for k in 0..=self.rank {
            sum += self.coefficients[k] * self.eigenvectors[[i, k]] * self.eigenvectors[[j, k]];
        }
        sum
    }

    /// Computes the heat diffusion distance between node i and node j.
    pub fn distance(&self, i: usize, j: usize) -> S {
        if i == j {
            return S::zero();
        }
        let k_ii = self.get(i, i);
        let k_jj = self.get(j, j);
        let k_ij = self.get(i, j);

        let eps = S::from_f64(1e-15).unwrap();
        let k_ii_eta = (k_ii + self.eta).max(eps);
        let k_jj_eta = (k_jj + self.eta).max(eps);
        let sqrt_prod = (k_ii_eta * k_jj_eta).sqrt();

        let ratio = ((k_ij + self.eta) / sqrt_prod).clamp(eps, S::one());
        let four_t = S::from_f64(4.0).unwrap() * self.t;
        (-four_t * ratio.ln()).max(S::zero()).sqrt()
    }

    /// Computes the single-source heat diffusion distance vector from a pivot node.
    pub fn pivot_distance_vector(&self, pivot: usize) -> Vec<S> {
        let n = self.n;
        assert!(pivot < n, "Pivot index out of bounds");

        let eps = S::from_f64(1e-15).unwrap();
        let mut distances = vec![S::zero(); n];
        let four_t = S::from_f64(4.0).unwrap() * self.t;

        // Precompute diagonal elements (K_jj + eta) for all j
        let mut diag = vec![S::zero(); n];
        for (j, diag_j) in diag.iter_mut().enumerate() {
            let mut sum = S::zero();
            for k in 0..=self.rank {
                sum += self.coefficients[k] * self.eigenvectors[[j, k]] * self.eigenvectors[[j, k]];
            }
            *diag_j = (sum + self.eta).max(eps);
        }

        let sqrt_k_pp = diag[pivot].sqrt();

        // Compute row K_{pivot, j} for all j
        for j in 0..n {
            if j == pivot {
                distances[j] = S::zero();
            } else {
                let mut k_pj = S::zero();
                for k in 0..=self.rank {
                    k_pj += self.coefficients[k]
                        * self.eigenvectors[[pivot, k]]
                        * self.eigenvectors[[j, k]];
                }
                let sqrt_k_jj = diag[j].sqrt();
                let ratio = ((k_pj + self.eta) / (sqrt_k_pp * sqrt_k_jj)).clamp(eps, S::one());
                distances[j] = (-four_t * ratio.ln()).max(S::zero()).sqrt();
            }
        }

        distances
    }

    /// Returns the number of nodes in the graph.
    pub fn n(&self) -> usize {
        self.n
    }

    /// Returns the diffusion time parameter t.
    pub fn t(&self) -> S {
        self.t
    }

    /// Returns the approximation rank r.
    pub fn rank(&self) -> usize {
        self.rank
    }

    /// Returns the regularization parameter eta.
    pub fn eta(&self) -> S {
        self.eta
    }

    /// Returns a reference to the computed eigenvalues.
    pub fn eigenvalues(&self) -> &Array1<S> {
        &self.eigenvalues
    }
}

impl<S> HeatKernel<S> for LowRankDiffusionKernel<S>
where
    S: Float
        + std::iter::Sum
        + std::ops::AddAssign
        + Default
        + ScalarOperand
        + num_traits::FromPrimitive
        + DrawingValue,
{
    fn get(&self, i: usize, j: usize) -> S {
        self.get(i, j)
    }

    fn n(&self) -> usize {
        self.n()
    }

    fn t(&self) -> S {
        self.t()
    }
}

/// Distance matrix implementation backed by any HeatKernel implementation.
#[derive(Debug, Clone)]
pub struct HeatGeodesicDistanceMatrix<N, S, K> {
    kernel: K,
    node_indices: HashMap<N, usize>,
    min_dist: S,
    pivots: Option<Vec<usize>>,
    pivot_distances: Option<Vec<Vec<S>>>,
}

/// Backward-compatible type alias for LowRankDiffusionDistanceMatrix.
pub type LowRankDiffusionDistanceMatrix<N, S> =
    HeatGeodesicDistanceMatrix<N, S, LowRankDiffusionKernel<S>>;

impl<N, S, K> HeatGeodesicDistanceMatrix<N, S, K>
where
    N: Eq + Hash + Copy,
    S: Float
        + std::iter::Sum
        + std::ops::AddAssign
        + Default
        + ScalarOperand
        + num_traits::FromPrimitive
        + DrawingValue,
    K: HeatKernel<S>,
{
    /// Creates a new HeatGeodesicDistanceMatrix for all node pairs.
    pub fn new<G>(graph: G, kernel: K, min_dist: S) -> Self
    where
        G: IntoNodeIdentifiers,
        G::NodeId: Into<N>,
    {
        let node_indices: HashMap<N, usize> = graph
            .node_identifiers()
            .enumerate()
            .map(|(i, node_id)| (node_id.into(), i))
            .collect();

        Self {
            kernel,
            node_indices,
            min_dist,
            pivots: None,
            pivot_distances: None,
        }
    }

    /// Creates a new HeatGeodesicDistanceMatrix with precomputed pivot distance vectors.
    pub fn new_with_pivots<G>(graph: G, kernel: K, pivots: &[usize], min_dist: S) -> Self
    where
        G: IntoNodeIdentifiers,
        G::NodeId: Into<N>,
    {
        let node_indices: HashMap<N, usize> = graph
            .node_identifiers()
            .enumerate()
            .map(|(i, node_id)| (node_id.into(), i))
            .collect();

        let mut pivot_distances = Vec::with_capacity(pivots.len());
        for &p in pivots {
            pivot_distances.push(kernel.pivot_distance_vector(p));
        }

        Self {
            kernel,
            node_indices,
            min_dist,
            pivots: Some(pivots.to_vec()),
            pivot_distances: Some(pivot_distances),
        }
    }
}

impl<N, S, K> Distance<N, S> for HeatGeodesicDistanceMatrix<N, S, K>
where
    N: Eq + Hash + Copy,
    S: Float + std::iter::Sum + std::ops::AddAssign + Default + ScalarOperand + DrawingValue,
    K: HeatKernel<S>,
{
    fn get(&self, u: N, v: N) -> Option<S> {
        let i = self.row_index(u)?;
        let j = self.col_index(v)?;
        Some(self.get_by_index(i, j))
    }

    fn get_by_index(&self, i: usize, j: usize) -> S {
        if i == j {
            S::zero()
        } else if let (Some(pivots), Some(distances)) = (&self.pivots, &self.pivot_distances) {
            if let Some(pivot_idx) = pivots.iter().position(|&p| p == i) {
                distances[pivot_idx][j].max(self.min_dist)
            } else if let Some(pivot_idx) = pivots.iter().position(|&p| p == j) {
                distances[pivot_idx][i].max(self.min_dist)
            } else {
                S::infinity()
            }
        } else {
            self.kernel.distance(i, j).max(self.min_dist)
        }
    }

    fn shape(&self) -> (usize, usize) {
        let n = self.kernel.n();
        (n, n)
    }

    fn row_index(&self, u: N) -> Option<usize> {
        self.node_indices.get(&u).copied()
    }

    fn col_index(&self, u: N) -> Option<usize> {
        self.node_indices.get(&u).copied()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use petgraph::graph::UnGraph;
    use petgraph_distance::{Laplacian, StandardLaplacian};
    use rand::SeedableRng;

    #[test]
    fn test_low_rank_kernel_eta() {
        let mut g = UnGraph::<(), ()>::new_undirected();
        let n0 = g.add_node(());
        let n1 = g.add_node(());
        let n2 = g.add_node(());
        g.add_edge(n0, n1, ());
        g.add_edge(n1, n2, ());

        let laplacian = StandardLaplacian.build(&g, &mut |_| 1.0);
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);

        // Test default eta = 0.0
        let kernel_default = LowRankDiffusionKernel::new(&laplacian, 1.0, 2, &mut rng);
        assert_eq!(kernel_default.eta(), 0.0);

        // Test custom eta = 0.01
        let eta = 0.01;
        let kernel_eta = LowRankDiffusionKernel::new_with_eta(&laplacian, 1.0, 2, eta, &mut rng);
        assert_eq!(kernel_eta.eta(), eta);

        // Verify distance formula with eta:
        // ratio = (k_ij + eta) / sqrt((k_ii + eta) * (k_jj + eta))
        let k_00 = kernel_eta.get(0, 0);
        let k_11 = kernel_eta.get(1, 1);
        let k_01 = kernel_eta.get(0, 1);

        let expected_ratio = (k_01 + eta) / ((k_00 + eta) * (k_11 + eta)).sqrt();
        let expected_ratio_clamped = expected_ratio.clamp(1e-15, 1.0);
        let expected_dist = (-4.0 * expected_ratio_clamped.ln()).max(0.0).sqrt();

        assert!((kernel_eta.distance(0, 1) - expected_dist).abs() < 1e-12);

        // Verify pivot distance vector matches pairwise distance
        let pivot_dists = kernel_eta.pivot_distance_vector(0);
        assert!((pivot_dists[1] - expected_dist).abs() < 1e-12);
    }
}
