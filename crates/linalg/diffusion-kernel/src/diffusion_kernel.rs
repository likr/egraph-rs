use crate::chebyshev::{chebyshev_approximation_vec, chebyshev_approximation_with_baseline};
use crate::hutchinson::{generate_rademacher_vectors, HutchinsonEstimator};
use crate::power_method::estimate_lambda_max;
use ndarray::{Array1, ScalarOperand};
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
    lambda_max: S,
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
        let u1 = laplacian.stationary_vector();
        let v = generate_rademacher_vectors(n, num_vectors, rng);
        let (kv, w_mat, residual_diag_sum) =
            chebyshev_approximation_with_baseline(laplacian, t, degree, lambda_max, &v, &u1);
        let u1_slice = u1
            .as_slice()
            .expect("Stationary vector u1 must be contiguous");
        let estimator =
            HutchinsonEstimator::new_with_baseline(v, kv, w_mat, &residual_diag_sum, u1_slice);
        Self {
            estimator,
            lambda_max,
        }
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

    /// Returns the estimated maximum eigenvalue of the Laplacian.
    pub fn lambda_max(&self) -> S {
        self.lambda_max
    }

    /// Computes the exact single-source heat diffusion vector K e_pivot for a given pivot node.
    pub fn single_source_heat_vector(
        laplacian: &SparseSymmetricMatrix<S>,
        t: S,
        degree: usize,
        pivot: usize,
    ) -> Array1<S> {
        let mut rng = rand::thread_rng();
        let lambda_max = estimate_lambda_max(laplacian, &mut rng, 100, S::from_f64(1e-6).unwrap());
        Self::single_source_heat_vector_with_lambda_max(laplacian, t, degree, lambda_max, pivot)
    }

    /// Computes the exact single-source heat diffusion vector K e_pivot with given lambda_max.
    pub fn single_source_heat_vector_with_lambda_max(
        laplacian: &SparseSymmetricMatrix<S>,
        t: S,
        degree: usize,
        lambda_max: S,
        pivot: usize,
    ) -> Array1<S> {
        let n = laplacian.dim();
        let mut e = Array1::zeros(n);
        e[pivot] = S::one();
        chebyshev_approximation_vec(laplacian, t, degree, lambda_max, &e)
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

/// A read-only distance matrix that computes single-source heat diffusion distances from pre-selected pivots.
#[derive(Debug, Clone)]
pub struct PivotDiffusionDistanceMatrix<N, S> {
    pivots: Vec<usize>,
    distances: Vec<Vec<S>>,
    node_indices: HashMap<N, usize>,
    min_dist: S,
}

impl<S> PivotDiffusionDistanceMatrix<(), S>
where
    S: Float
        + std::iter::Sum
        + std::ops::AddAssign
        + Default
        + ScalarOperand
        + num_traits::FromPrimitive,
{
    /// Computes single-source heat diffusion distance vector from a pivot node.
    pub fn pivot_distance_vector(
        laplacian: &SparseSymmetricMatrix<S>,
        kernel: &DiffusionKernel<S>,
        t: S,
        degree: usize,
        pivot: usize,
    ) -> Vec<S> {
        let heat_vec = DiffusionKernel::single_source_heat_vector_with_lambda_max(
            laplacian,
            t,
            degree,
            kernel.lambda_max(),
            pivot,
        );
        let n = kernel.n();
        let mut distances = vec![S::zero(); n];
        let four_t = S::from_f64(4.0).unwrap() * t;
        let eps = S::from_f64(1e-15).unwrap();

        let k_pp = kernel.estimator.query_diagonal(pivot).max(eps);
        let sqrt_k_pp = k_pp.sqrt();

        for (j, &k_pj) in heat_vec.iter().enumerate() {
            if j == pivot {
                distances[j] = S::zero();
            } else {
                let k_pj_clamped = k_pj.max(eps);
                let k_jj = kernel.estimator.query_diagonal(j).max(eps);
                let sqrt_k_jj = k_jj.sqrt();
                let ratio = k_pj_clamped / (sqrt_k_pp * sqrt_k_jj);
                let ratio_clamped = ratio.min(S::one()).max(eps);
                distances[j] = (-four_t * ratio_clamped.ln()).max(S::zero()).sqrt();
            }
        }
        distances
    }

    /// Creates a new PivotDiffusionDistanceMatrix by computing single-source heat diffusion from the given pivots.
    pub fn new<G, N>(
        graph: G,
        laplacian: &SparseSymmetricMatrix<S>,
        kernel: &DiffusionKernel<S>,
        t: S,
        degree: usize,
        pivots: &[usize],
        min_dist: S,
    ) -> PivotDiffusionDistanceMatrix<N, S>
    where
        G: IntoNodeIdentifiers,
        G::NodeId: Into<N>,
        N: Eq + Hash + Copy,
    {
        let node_indices: HashMap<N, usize> = graph
            .node_identifiers()
            .enumerate()
            .map(|(i, node_id)| (node_id.into(), i))
            .collect();

        let mut distances = Vec::with_capacity(pivots.len());
        for &p in pivots {
            let dist_vec = Self::pivot_distance_vector(laplacian, kernel, t, degree, p);
            distances.push(dist_vec);
        }

        PivotDiffusionDistanceMatrix {
            pivots: pivots.to_vec(),
            distances,
            node_indices,
            min_dist,
        }
    }

    /// Creates a PivotDiffusionDistanceMatrix from pre-computed pivot distance vectors.
    ///
    /// Used by incremental pivot selection to avoid recomputing distances.
    pub fn from_precomputed<G, N>(
        graph: G,
        pivots: &[usize],
        distances: Vec<Vec<S>>,
        min_dist: S,
    ) -> PivotDiffusionDistanceMatrix<N, S>
    where
        G: IntoNodeIdentifiers,
        G::NodeId: Into<N>,
        N: Eq + Hash + Copy,
    {
        let node_indices: HashMap<N, usize> = graph
            .node_identifiers()
            .enumerate()
            .map(|(i, node_id)| (node_id.into(), i))
            .collect();

        PivotDiffusionDistanceMatrix {
            pivots: pivots.to_vec(),
            distances,
            node_indices,
            min_dist,
        }
    }
}

impl<N, S> Distance<N, S> for PivotDiffusionDistanceMatrix<N, S>
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
        } else if let Some(pivot_idx) = self.pivots.iter().position(|&p| p == i) {
            self.distances[pivot_idx][j].max(self.min_dist)
        } else if let Some(pivot_idx) = self.pivots.iter().position(|&p| p == j) {
            self.distances[pivot_idx][i].max(self.min_dist)
        } else {
            S::infinity()
        }
    }

    fn shape(&self) -> (usize, usize) {
        let n = self.node_indices.len();
        (n, n)
    }

    fn row_index(&self, u: N) -> Option<usize> {
        self.node_indices.get(&u).copied()
    }

    fn col_index(&self, u: N) -> Option<usize> {
        self.node_indices.get(&u).copied()
    }
}
