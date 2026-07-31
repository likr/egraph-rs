//! Eigenvalue-free Multiscale Diffusion Distance Kernel using Batched BiCGSTAB and Hutchinson sampling.

use crate::bicgstab::{apply_p, solve_batched_bicgstab, BicgstabSolverBuffers};
use crate::hutchinson::{generate_rademacher_vectors, HutchinsonEstimator};
use ndarray::Array2;
use num_traits::Float;
use petgraph::visit::IntoNodeIdentifiers;
use petgraph_distance::{Distance, SparseSymmetricMatrix};
use petgraph_drawing::DrawingValue;
use rand::Rng;
use std::collections::HashMap;
use std::hash::Hash;

/// Core kernel for computing multiscale diffusion distance without full eigenvalue decomposition.
#[derive(Debug, Clone)]
pub struct MultiscaleDiffusionKernel {
    matrix: SparseSymmetricMatrix<f64>,
    alpha: f64,
    num_samples: usize,
    tol: f64,
    max_iter: usize,
    degrees: Vec<f64>,
    index: Option<HutchinsonEstimator<f64>>,
    buffers: BicgstabSolverBuffers,
}

impl MultiscaleDiffusionKernel {
    /// Creates a new multiscale kernel from a `SparseSymmetricMatrix` and solver parameters.
    pub fn new(
        matrix: SparseSymmetricMatrix<f64>,
        alpha: f64,
        num_samples: usize,
        tol: f64,
        max_iter: usize,
    ) -> Self {
        let n = matrix.dim();
        let degrees = compute_degrees(&matrix);
        let buffers = BicgstabSolverBuffers::new(n, num_samples);

        Self {
            matrix,
            alpha,
            num_samples,
            tol,
            max_iter,
            degrees,
            index: None,
            buffers,
        }
    }

    /// Returns the number of nodes in the graph.
    pub fn n(&self) -> usize {
        self.matrix.dim()
    }

    /// Builds the Hutchinson index using a single Batched BiCGSTAB solve.
    pub fn build_index(&mut self) {
        let mut rng = rand::thread_rng();
        self.build_index_with_rng(&mut rng);
    }

    /// Builds the Hutchinson index using a custom random number generator.
    pub fn build_index_with_rng<R: Rng>(&mut self, rng: &mut R) {
        let n = self.matrix.dim();
        let m = self.num_samples;

        // Generate Rademacher random matrix Z in {-1.0, 1.0} of shape (n, m)
        let z_matrix = generate_rademacher_vectors(n, m, rng);

        // Convert Z to f64 RHS matrix B = alpha * P * Z
        let mut b = vec![0.0; n * m];
        let z_slice = z_matrix.as_slice().expect("Array2 Z must be contiguous");
        apply_p(&self.matrix, &self.degrees, z_slice, &mut b, m);
        for val in b.iter_mut() {
            *val *= self.alpha;
        }

        // Solve (I - alpha * P) Y = B using Batched BiCGSTAB
        let mut y_samples = vec![0.0; n * m];
        solve_batched_bicgstab(
            &self.matrix,
            &self.degrees,
            self.alpha,
            &b,
            &mut y_samples,
            m,
            self.tol,
            self.max_iter,
            &mut self.buffers,
        );

        let y_matrix = Array2::from_shape_vec((n, m), y_samples).expect("Shape should match n x m");

        self.index = Some(HutchinsonEstimator::new(z_matrix, y_matrix));
    }

    /// Computes multiscale diffusion distance between nodes x and y using precomputed index.
    pub fn sample_distance(&self, x: usize, y: usize) -> Result<f64, String> {
        match &self.index {
            Some(idx) => Ok(idx.distance_l2(x, y)),
            None => Err("Index not built. Call build_index() first.".to_string()),
        }
    }

    /// Returns a reference to the built Hutchinson index, if available.
    pub fn index(&self) -> Option<&HutchinsonEstimator<f64>> {
        self.index.as_ref()
    }
}

/// A read-only distance matrix backed by `MultiscaleDiffusionKernel`.
#[derive(Debug, Clone)]
pub struct MultiscaleDiffusionDistanceMatrix<N, S> {
    kernel: MultiscaleDiffusionKernel,
    node_indices: HashMap<N, usize>,
    min_dist: S,
}

impl<N, S> MultiscaleDiffusionDistanceMatrix<N, S>
where
    N: Eq + Hash + Copy,
    S: DrawingValue,
{
    /// Constructs a new distance matrix wrapping a `MultiscaleDiffusionKernel`.
    pub fn new<G>(graph: G, mut kernel: MultiscaleDiffusionKernel, min_dist: S) -> Self
    where
        G: IntoNodeIdentifiers,
        G::NodeId: Into<N>,
    {
        if kernel.index().is_none() {
            kernel.build_index();
        }

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

impl<N, S> Distance<N, S> for MultiscaleDiffusionDistanceMatrix<N, S>
where
    N: Eq + Hash + Copy,
    S: DrawingValue + Float,
{
    fn get(&self, u: N, v: N) -> Option<S> {
        let i = self.row_index(u)?;
        let j = self.col_index(v)?;
        Some(self.get_by_index(i, j))
    }

    fn get_by_index(&self, i: usize, j: usize) -> S {
        let raw_dist = self.kernel.sample_distance(i, j).unwrap_or(0.0);
        let s_dist = S::from_f64(raw_dist).unwrap_or_else(S::zero);
        if i == j {
            S::zero()
        } else {
            s_dist.max(self.min_dist)
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

/// Helper function to compute vertex degree array for symmetric graph matrix.
fn compute_degrees(matrix: &SparseSymmetricMatrix<f64>) -> Vec<f64> {
    let mut degrees = vec![0.0; matrix.dim()];
    for &(i, j, w) in matrix.edges() {
        let abs_w = w.abs();
        degrees[i] += abs_w;
        degrees[j] += abs_w;
    }
    for (deg, &diag) in degrees.iter_mut().zip(matrix.diagonal()) {
        if *deg == 0.0 && diag > 0.0 {
            *deg = diag;
        }
    }
    degrees
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_multiscale_kernel_basic() {
        // Create a simple 4-node path graph: 0 - 1 - 2 - 3
        let mut matrix: SparseSymmetricMatrix<f64> = SparseSymmetricMatrix::new(4);
        matrix.add_edge(0, 1, 1.0);
        matrix.add_edge(1, 2, 1.0);
        matrix.add_edge(2, 3, 1.0);

        let mut kernel = MultiscaleDiffusionKernel::new(matrix, 0.85, 32, 1e-7, 100);
        kernel.build_index();

        let d01 = kernel.sample_distance(0, 1).unwrap();
        let d03 = kernel.sample_distance(0, 3).unwrap();
        let d00 = kernel.sample_distance(0, 0).unwrap();

        assert_eq!(d00, 0.0);
        assert!(d01 > 0.0, "d01 should be > 0, got {}", d01);
        assert!(d03 > d01, "Distance to further node should be larger");
    }

    #[test]
    fn test_multiscale_kernel_symmetry() {
        let mut matrix: SparseSymmetricMatrix<f64> = SparseSymmetricMatrix::new(3);
        matrix.add_edge(0, 1, 1.0);
        matrix.add_edge(1, 2, 2.0);

        let mut kernel = MultiscaleDiffusionKernel::new(matrix, 0.85, 32, 1e-7, 100);
        kernel.build_index();

        let d01 = kernel.sample_distance(0, 1).unwrap();
        let d10 = kernel.sample_distance(1, 0).unwrap();

        assert!((d01 - d10).abs() < 1e-12);
    }

    #[test]
    fn test_unbuilt_index_error() {
        let matrix: SparseSymmetricMatrix<f64> = SparseSymmetricMatrix::new(2);
        let kernel = MultiscaleDiffusionKernel::new(matrix, 0.85, 32, 1e-7, 100);
        assert!(kernel.sample_distance(0, 1).is_err());
    }

    #[test]
    fn test_petgraph_integration_and_distance_trait() {
        use petgraph::graph::UnGraph;

        let mut graph = UnGraph::<&str, f64>::new_undirected();
        let n0 = graph.add_node("A");
        let n1 = graph.add_node("B");
        let n2 = graph.add_node("C");
        let n3 = graph.add_node("D");

        graph.add_edge(n0, n1, 1.0);
        graph.add_edge(n1, n2, 1.0);
        graph.add_edge(n2, n3, 1.0);

        let laplacian = SparseSymmetricMatrix::standard_laplacian(&graph, |e| *e.weight());
        let kernel = MultiscaleDiffusionKernel::new(laplacian, 0.85, 32, 1e-7, 100);

        let dist_matrix = MultiscaleDiffusionDistanceMatrix::new(&graph, kernel, 1e-4);

        assert_eq!(dist_matrix.get(n0, n0), Some(0.0));
        let d01 = dist_matrix.get(n0, n1).unwrap();
        let d03 = dist_matrix.get(n0, n3).unwrap();

        assert!(d01 > 0.0);
        assert!(
            d03 > d01,
            "d03 ({}) should be larger than d01 ({})",
            d03,
            d01
        );
    }

    #[test]
    fn test_sample_count_stability() {
        use rand::SeedableRng;

        let mut matrix: SparseSymmetricMatrix<f64> = SparseSymmetricMatrix::new(5);
        matrix.add_edge(0, 1, 1.0);
        matrix.add_edge(1, 2, 1.0);
        matrix.add_edge(2, 3, 1.0);
        matrix.add_edge(3, 4, 1.0);

        let sample_sizes = [32, 64, 128];
        let mut distances = Vec::new();

        for &m in &sample_sizes {
            let mut kernel = MultiscaleDiffusionKernel::new(matrix.clone(), 0.85, m, 1e-7, 100);
            let mut rng = rand::rngs::StdRng::seed_from_u64(42);
            kernel.build_index_with_rng(&mut rng);
            let d = kernel.sample_distance(0, 4).unwrap();
            assert!(d > 0.0, "Distance should be positive");
            distances.push(d);
        }

        // Verify that sample estimates for (0, 4) stay consistent across sample sizes
        let mean: f64 = distances.iter().sum::<f64>() / distances.len() as f64;
        for &d in &distances {
            assert!(
                (d - mean).abs() < 0.6,
                "Distance {} deviates significantly from mean {}",
                d,
                mean
            );
        }
    }
}
