//! Eigenvalue-free Multiscale Diffusion Distance Engine using Batched BiCGSTAB and Hutchinson sampling.

use crate::hutchinson::{generate_rademacher_vectors, HutchinsonEstimator};
use ndarray::Array2;
use num_traits::Float;
use petgraph::visit::{EdgeRef, IntoEdgeReferences, IntoNodeIdentifiers, NodeCount};
use petgraph_distance::Distance;
use petgraph_drawing::{DrawingIndex, DrawingValue};
use petgraph_linalg_spmv::SparseSymmetricMatrix;
use rand::Rng;
use std::collections::HashMap;
use std::hash::Hash;

/// Configuration parameters for Multiscale Diffusion Distance computation.
#[derive(Debug, Clone)]
pub struct MultiscaleDiffusionConfig {
    /// Decay factor alpha in (0, 1). Controls diffusion time scale aggregation.
    pub alpha: f64,
    /// Number of Rademacher random sample vectors M (recommended to be a multiple of 8 for SIMD).
    pub num_samples: usize,
    /// Convergence tolerance for Batched BiCGSTAB solver (relative residual norm).
    pub tol: f64,
    /// Maximum iteration count for Batched BiCGSTAB solver.
    pub max_iter: usize,
}

impl Default for MultiscaleDiffusionConfig {
    fn default() -> Self {
        Self {
            alpha: 0.85,
            num_samples: 32,
            tol: 1e-7,
            max_iter: 100,
        }
    }
}

/// Workspace memory buffer reused across iterations during Batched BiCGSTAB solving.
#[derive(Debug, Clone)]
pub struct SolverBuffers {
    pub n: usize,
    pub m: usize,
    pub r: Vec<f64>,
    pub r0_hat: Vec<f64>,
    pub p: Vec<f64>,
    pub v: Vec<f64>,
    pub s: Vec<f64>,
    pub t: Vec<f64>,
    pub temp: Vec<f64>,
}

impl SolverBuffers {
    /// Creates pre-allocated zero buffers sized for n x m Row-Major arrays.
    pub fn new(n: usize, m: usize) -> Self {
        let size = n * m;
        Self {
            n,
            m,
            r: vec![0.0; size],
            r0_hat: vec![0.0; size],
            p: vec![0.0; size],
            v: vec![0.0; size],
            s: vec![0.0; size],
            t: vec![0.0; size],
            temp: vec![0.0; size],
        }
    }
}

/// Core engine for computing multiscale diffusion distance without full eigenvalue decomposition.
#[derive(Debug, Clone)]
pub struct MultiscaleDiffusionEngine {
    matrix: SparseSymmetricMatrix<f64>,
    config: MultiscaleDiffusionConfig,
    degrees: Vec<f64>,
    index: Option<HutchinsonEstimator<f64>>,
    buffers: SolverBuffers,
}

impl MultiscaleDiffusionEngine {
    /// Creates a new engine from a `SparseSymmetricMatrix` and solver configuration.
    pub fn new(matrix: SparseSymmetricMatrix<f64>, config: MultiscaleDiffusionConfig) -> Self {
        let n = matrix.dim();
        let m = config.num_samples;
        let degrees = compute_degrees(&matrix);
        let buffers = SolverBuffers::new(n, m);

        Self {
            matrix,
            config,
            degrees,
            index: None,
            buffers,
        }
    }

    /// Creates a new engine directly from any petgraph graph and edge weight closure.
    pub fn from_petgraph<G, F>(
        graph: G,
        mut edge_weight: F,
        config: MultiscaleDiffusionConfig,
    ) -> Self
    where
        G: IntoEdgeReferences + IntoNodeIdentifiers + NodeCount + Copy,
        G::NodeId: DrawingIndex,
        F: FnMut(G::EdgeRef) -> f64,
    {
        let n = graph.node_count();
        let node_indices: HashMap<G::NodeId, usize> = graph
            .node_identifiers()
            .enumerate()
            .map(|(i, node_id)| (node_id, i))
            .collect();

        let mut matrix = SparseSymmetricMatrix::new(n);
        for edge in graph.edge_references() {
            let u = node_indices[&edge.source()];
            let v = node_indices[&edge.target()];
            let w = edge_weight(edge);
            if u == v {
                matrix.add_to_diagonal(u, w);
            } else {
                let (i, j) = if u < v { (u, v) } else { (v, u) };
                matrix.add_edge(i, j, w);
            }
        }

        Self::new(matrix, config)
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
        let m = self.config.num_samples;

        // Generate Rademacher random matrix Z in {-1.0, 1.0} of shape (n, m)
        let z_matrix = generate_rademacher_vectors(n, m, rng);

        // Convert Z to f64 RHS matrix B = alpha * P * Z
        let mut b = vec![0.0; n * m];
        let z_slice = z_matrix.as_slice().expect("Array2 Z must be contiguous");
        apply_p(&self.matrix, &self.degrees, z_slice, &mut b, m);
        for val in b.iter_mut() {
            *val *= self.config.alpha;
        }

        // Solve (I - alpha * P) Y = B using Batched BiCGSTAB
        let mut y_samples = vec![0.0; n * m];
        solve_batched_bicgstab(
            &self.matrix,
            &self.degrees,
            self.config.alpha,
            &b,
            &mut y_samples,
            m,
            self.config.tol,
            self.config.max_iter,
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

/// A read-only distance matrix backed by `MultiscaleDiffusionEngine`.
#[derive(Debug, Clone)]
pub struct MultiscaleDiffusionDistanceMatrix<N, S> {
    engine: MultiscaleDiffusionEngine,
    node_indices: HashMap<N, usize>,
    min_dist: S,
}

impl<N, S> MultiscaleDiffusionDistanceMatrix<N, S>
where
    N: Eq + Hash + Copy,
    S: DrawingValue,
{
    /// Constructs a new distance matrix wrapping a `MultiscaleDiffusionEngine`.
    pub fn new<G>(graph: G, mut engine: MultiscaleDiffusionEngine, min_dist: S) -> Self
    where
        G: IntoNodeIdentifiers,
        G::NodeId: Into<N>,
    {
        if engine.index().is_none() {
            engine.build_index();
        }

        let node_indices: HashMap<N, usize> = graph
            .node_identifiers()
            .enumerate()
            .map(|(i, node_id)| (node_id.into(), i))
            .collect();

        Self {
            engine,
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
        let raw_dist = self.engine.sample_distance(i, j).unwrap_or(0.0);
        let s_dist = S::from_f64(raw_dist).unwrap_or_else(S::zero);
        if i == j {
            S::zero()
        } else {
            s_dist.max(self.min_dist)
        }
    }

    fn shape(&self) -> (usize, usize) {
        let n = self.engine.n();
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
    let mut degrees = matrix.diagonal().to_vec();
    for &(i, j, w) in matrix.edges() {
        degrees[i] += w;
        degrees[j] += w;
    }
    degrees
}

/// Computes output matrix `y = P * v` where P = D^{-1} W is Markov transition matrix.
#[allow(clippy::needless_range_loop)]
fn apply_p(
    matrix: &SparseSymmetricMatrix<f64>,
    degrees: &[f64],
    v: &[f64],
    y: &mut [f64],
    m: usize,
) {
    let n = matrix.dim();
    y.fill(0.0);

    // Diagonal elements W_ii * V_{i,m}
    let diagonal = matrix.diagonal();
    for i in 0..n {
        let d_val = diagonal[i];
        if d_val != 0.0 {
            let v_row = &v[i * m..(i + 1) * m];
            let y_row = &mut y[i * m..(i + 1) * m];
            for k in 0..m {
                y_row[k] += d_val * v_row[k];
            }
        }
    }

    // Off-diagonal edge interactions W_ij * V_{j,m} and W_ij * V_{i,m}
    for &(i, j, w) in matrix.edges() {
        let v_i = &v[i * m..(i + 1) * m];
        let v_j = &v[j * m..(j + 1) * m];

        // Safely extract non-overlapping mutable slices for rows i and j
        if i < j {
            let (left, right) = y.split_at_mut(j * m);
            let y_i = &mut left[i * m..(i + 1) * m];
            let y_j = &mut right[0..m];
            for k in 0..m {
                y_i[k] += w * v_j[k];
                y_j[k] += w * v_i[k];
            }
        }
    }

    // Row degree normalization: Y = D^{-1} (W * V)
    for i in 0..n {
        let d = degrees[i];
        if d > 0.0 {
            let inv_d = 1.0 / d;
            let y_row = &mut y[i * m..(i + 1) * m];
            for k in 0..m {
                y_row[k] *= inv_d;
            }
        }
    }
}

/// Computes `y = (I - alpha * P) * v` (SpMM for coefficient matrix A).
fn apply_a(
    matrix: &SparseSymmetricMatrix<f64>,
    degrees: &[f64],
    alpha: f64,
    v: &[f64],
    y: &mut [f64],
    temp: &mut [f64],
    m: usize,
) {
    apply_p(matrix, degrees, v, temp, m);
    let size = matrix.dim() * m;
    for i in 0..size {
        y[i] = v[i] - alpha * temp[i];
    }
}

/// Solves (I - alpha * P) Y = B for M right-hand side columns using Batched BiCGSTAB.
#[allow(clippy::too_many_arguments, clippy::needless_range_loop)]
fn solve_batched_bicgstab(
    matrix: &SparseSymmetricMatrix<f64>,
    degrees: &[f64],
    alpha: f64,
    b: &[f64],
    y: &mut [f64],
    m: usize,
    tol: f64,
    max_iter: usize,
    buffers: &mut SolverBuffers,
) {
    let n = matrix.dim();
    let size = n * m;

    y.fill(0.0);
    buffers.r.copy_from_slice(b);
    buffers.r0_hat.copy_from_slice(b);
    buffers.p.fill(0.0);
    buffers.v.fill(0.0);

    let mut rho_prev = vec![1.0; m];
    let mut alpha_vec = vec![1.0; m];
    let mut omega_vec = vec![1.0; m];

    let b_norm: f64 = b.iter().map(|&val| val * val).sum::<f64>().sqrt();
    if b_norm == 0.0 {
        return;
    }

    for _iter in 0..max_iter {
        // Calculate per-column rho_k = dot(r0_hat_m, r_m)
        let mut rho_curr = vec![0.0; m];
        for i in 0..n {
            let r0_row = &buffers.r0_hat[i * m..(i + 1) * m];
            let r_row = &buffers.r[i * m..(i + 1) * m];
            for k in 0..m {
                rho_curr[k] += r0_row[k] * r_row[k];
            }
        }

        // Calculate beta_m = (rho_curr / rho_prev) * (alpha_prev / omega_prev)
        let mut beta = vec![0.0; m];
        for k in 0..m {
            if rho_prev[k].abs() > 1e-30 && omega_vec[k].abs() > 1e-30 {
                beta[k] = (rho_curr[k] / rho_prev[k]) * (alpha_vec[k] / omega_vec[k]);
            }
        }

        // Update P: p_m = r_m + beta_m * (p_m - omega_m * v_m)
        for i in 0..n {
            let r_row = &buffers.r[i * m..(i + 1) * m];
            let v_row = &buffers.v[i * m..(i + 1) * m];
            let p_row = &mut buffers.p[i * m..(i + 1) * m];
            for k in 0..m {
                p_row[k] = r_row[k] + beta[k] * (p_row[k] - omega_vec[k] * v_row[k]);
            }
        }

        // Compute V = A * P
        apply_a(
            matrix,
            degrees,
            alpha,
            &buffers.p,
            &mut buffers.v,
            &mut buffers.temp,
            m,
        );

        // Calculate alpha_m = rho_curr / dot(r0_hat_m, v_m)
        for k in 0..m {
            let mut v_dot = 0.0;
            for i in 0..n {
                v_dot += buffers.r0_hat[i * m + k] * buffers.v[i * m + k];
            }
            alpha_vec[k] = if v_dot.abs() > 1e-30 {
                rho_curr[k] / v_dot
            } else {
                0.0
            };
        }

        // Compute S = R - alpha * V
        for i in 0..n {
            let r_row = &buffers.r[i * m..(i + 1) * m];
            let v_row = &buffers.v[i * m..(i + 1) * m];
            let s_row = &mut buffers.s[i * m..(i + 1) * m];
            for k in 0..m {
                s_row[k] = r_row[k] - alpha_vec[k] * v_row[k];
            }
        }

        // Early check for residual convergence on S
        let s_norm: f64 = buffers.s.iter().map(|&val| val * val).sum::<f64>().sqrt();
        if s_norm / b_norm <= tol {
            for i in 0..size {
                y[i] += alpha_vec[i % m] * buffers.p[i];
            }
            break;
        }

        // Compute T = A * S
        apply_a(
            matrix,
            degrees,
            alpha,
            &buffers.s,
            &mut buffers.t,
            &mut buffers.temp,
            m,
        );

        // Calculate omega_m = dot(t_m, s_m) / dot(t_m, t_m)
        for k in 0..m {
            let mut t_t = 0.0;
            let mut t_s = 0.0;
            for i in 0..n {
                let t_val = buffers.t[i * m + k];
                let s_val = buffers.s[i * m + k];
                t_t += t_val * t_val;
                t_s += t_val * s_val;
            }
            omega_vec[k] = if t_t > 1e-30 { t_s / t_t } else { 0.0 };
        }

        // Update solution Y += alpha * P + omega * S
        // and residual R = S - omega * T
        for i in 0..n {
            let p_row = &buffers.p[i * m..(i + 1) * m];
            let s_row = &buffers.s[i * m..(i + 1) * m];
            let t_row = &buffers.t[i * m..(i + 1) * m];
            let y_row = &mut y[i * m..(i + 1) * m];
            let r_row = &mut buffers.r[i * m..(i + 1) * m];
            for k in 0..m {
                y_row[k] += alpha_vec[k] * p_row[k] + omega_vec[k] * s_row[k];
                r_row[k] = s_row[k] - omega_vec[k] * t_row[k];
            }
        }

        let r_norm: f64 = buffers.r.iter().map(|&val| val * val).sum::<f64>().sqrt();
        if r_norm / b_norm <= tol {
            break;
        }

        rho_prev = rho_curr;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_multiscale_engine_basic() {
        // Create a simple 4-node path graph: 0 - 1 - 2 - 3
        let mut matrix: SparseSymmetricMatrix<f64> = SparseSymmetricMatrix::new(4);
        matrix.add_edge(0, 1, 1.0);
        matrix.add_edge(1, 2, 1.0);
        matrix.add_edge(2, 3, 1.0);

        let config = MultiscaleDiffusionConfig {
            alpha: 0.85,
            num_samples: 32,
            tol: 1e-7,
            max_iter: 100,
        };

        let mut engine = MultiscaleDiffusionEngine::new(matrix, config);
        engine.build_index();

        let d01 = engine.sample_distance(0, 1).unwrap();
        let d03 = engine.sample_distance(0, 3).unwrap();
        let d00 = engine.sample_distance(0, 0).unwrap();

        assert_eq!(d00, 0.0);
        assert!(d01 > 0.0, "d01 should be > 0, got {}", d01);
        assert!(d03 > d01, "Distance to further node should be larger");
    }

    #[test]
    fn test_multiscale_engine_symmetry() {
        let mut matrix: SparseSymmetricMatrix<f64> = SparseSymmetricMatrix::new(3);
        matrix.add_edge(0, 1, 1.0);
        matrix.add_edge(1, 2, 2.0);

        let config = MultiscaleDiffusionConfig::default();
        let mut engine = MultiscaleDiffusionEngine::new(matrix, config);
        engine.build_index();

        let d01 = engine.sample_distance(0, 1).unwrap();
        let d10 = engine.sample_distance(1, 0).unwrap();

        assert!((d01 - d10).abs() < 1e-12);
    }

    #[test]
    fn test_unbuilt_index_error() {
        let matrix: SparseSymmetricMatrix<f64> = SparseSymmetricMatrix::new(2);
        let engine = MultiscaleDiffusionEngine::new(matrix, MultiscaleDiffusionConfig::default());
        assert!(engine.sample_distance(0, 1).is_err());
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

        let engine = MultiscaleDiffusionEngine::from_petgraph(
            &graph,
            |e| *e.weight(),
            MultiscaleDiffusionConfig::default(),
        );

        let dist_matrix = MultiscaleDiffusionDistanceMatrix::new(&graph, engine, 1e-4);

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
            let config = MultiscaleDiffusionConfig {
                num_samples: m,
                ..Default::default()
            };
            let mut engine = MultiscaleDiffusionEngine::new(matrix.clone(), config);
            let mut rng = rand::rngs::StdRng::seed_from_u64(42);
            engine.build_index_with_rng(&mut rng);
            let d = engine.sample_distance(0, 4).unwrap();
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
