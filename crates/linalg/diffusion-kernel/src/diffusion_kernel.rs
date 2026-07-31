use crate::chebyshev::chebyshev_approximation;
use crate::hutchinson::{generate_rademacher_vectors, HutchinsonEstimator};
use crate::power_method::estimate_lambda_max;
use ndarray::ScalarOperand;
use num_traits::Float;
use petgraph::visit::IntoNodeIdentifiers;
use petgraph_distance::{Distance, SparseSymmetricMatrix};
use petgraph_drawing::DrawingValue;
use rand::Rng;
use std::collections::HashMap;
use std::hash::Hash;

/// DiffusionKernel provides random access to exp(-tL) matrix elements.
#[derive(Debug, Clone)]
pub struct DiffusionKernel<S> {
    estimator: HutchinsonEstimator<S>,
}

impl<S> DiffusionKernel<S>
where
    S: Float
        + std::iter::Sum
        + std::ops::AddAssign
        + Default
        + ScalarOperand
        + num_traits::FromPrimitive,
{
    /// Creates a new DiffusionKernel with automatic lambda_max estimation.
    pub fn new<R: Rng>(
        laplacian: &SparseSymmetricMatrix<S>,
        t: S,
        degree: usize,
        num_vectors: usize,
        rng: &mut R,
    ) -> Self {
        let lambda_max = estimate_lambda_max(laplacian, rng, 100, S::from_f64(1e-6).unwrap());
        Self::new_with_lambda_max(laplacian, t, degree, lambda_max, num_vectors, rng)
    }

    /// Creates a new DiffusionKernel with externally provided lambda_max.
    pub fn new_with_lambda_max<R: Rng>(
        laplacian: &SparseSymmetricMatrix<S>,
        t: S,
        degree: usize,
        lambda_max: S,
        num_vectors: usize,
        rng: &mut R,
    ) -> Self {
        let n = laplacian.dim();
        let v = generate_rademacher_vectors(n, num_vectors, rng);
        let kv = chebyshev_approximation(laplacian, t, degree, lambda_max, &v);
        let estimator = HutchinsonEstimator::new(v, kv);
        Self { estimator }
    }

    /// Queries the (i, j) element of the diffusion kernel matrix.
    pub fn get(&self, i: usize, j: usize) -> S {
        self.estimator.query(i, j)
    }

    /// Computes multiscale / diffusion distance between node i and node j.
    pub fn distance(&self, i: usize, j: usize) -> S {
        self.estimator.distance(i, j)
    }

    /// Returns the number of nodes in the graph.
    pub fn n(&self) -> usize {
        self.estimator.n()
    }
}

/// A read-only distance matrix that computes distance from a DiffusionKernel:
/// d(i, j) = max(sqrt(K[i,i] + K[j,j] - 2*K[i,j]), min_dist)
#[derive(Debug, Clone)]
pub struct DiffusionDistanceMatrix<N, S> {
    kernel: DiffusionKernel<S>,
    node_indices: HashMap<N, usize>,
    min_dist: S,
}

impl<N, S> DiffusionDistanceMatrix<N, S>
where
    N: Eq + Hash + Copy,
    S: Float
        + std::iter::Sum
        + std::ops::AddAssign
        + Default
        + ScalarOperand
        + num_traits::FromPrimitive,
{
    /// Creates a new DiffusionDistanceMatrix from a DiffusionKernel and a node mapping.
    pub fn new<G>(graph: G, kernel: DiffusionKernel<S>, min_dist: S) -> Self
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
        }
    }
}

impl<N, S> Distance<N, S> for DiffusionDistanceMatrix<N, S>
where
    N: Eq + Hash + Copy,
    S: Float + std::iter::Sum + std::ops::AddAssign + Default + ScalarOperand + DrawingValue,
{
    fn get(&self, u: N, v: N) -> Option<S> {
        let i = self.row_index(u)?;
        let j = self.col_index(v)?;
        Some(self.get_by_index(i, j))
    }

    fn get_by_index(&self, i: usize, j: usize) -> S {
        if i == j {
            S::zero()
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
