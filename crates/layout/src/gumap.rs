//! Pure-Rust GuMap (Graph UMAP) layout solver with `dist_sq.powf(b)` gradient evaluation
//! and cooperative [`IterativeSolver`] stepping.

use egraph_core::{
    CoreError, CoreResult, CsrGraph, DeterministicRng, IterativeSolver, Layout2d, Scalar,
    SolverStatus,
};
use egraph_neighborhood::{
    DogVaradhanConfig, DogVaradhanLandmarks, KnnGraph, SparseProbabilities,
    build_gumap_12hop_priority_csr, compute_fuzzy_simplicial_set,
    compute_regularized_dog_varadhan_landmarks, extract_sparse_1hop_2hop_pairs, find_ab_params,
};

const MIN_DIST_SQ_CLAMP: f64 = 1e-8;
const REPULSIVE_DIST_SQ_EPS: f64 = 1e-4;
const GRAD_CLAMP: f64 = 4.0;
const DEFAULT_TIER2_MARGIN: f64 = 1.1;
const DEFAULT_EPS_DIST: f64 = 1e-6;

/// Hyperparameters for [`GuMapSolver`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GuMapConfig {
    /// Total number of optimization epochs (`default: 200`).
    pub n_epochs: usize,
    /// Number of negative samples drawn per attractive edge (`default: 5`).
    pub negative_sample_rate: usize,
    /// Initial SGD learning rate (`default: 1.0`).
    pub learning_rate: f64,
    /// Target minimum separation in the 2D embedding (`default: 0.1`).
    pub min_dist: f64,
    /// Effective scale/spread parameter in the 2D embedding (`default: 1.0`).
    pub spread: f64,
    /// Optional explicit UMAP curve parameter `a` (`None` fits from `min_dist` and `spread`).
    pub a: Option<f64>,
    /// Optional explicit UMAP curve parameter `b` (`None` fits from `min_dist` and `spread`).
    pub b: Option<f64>,
    /// Maximum binary-search iterations for $\sigma_i$ in fuzzy simplicial set construction (`default: 64`).
    pub sigma_iters: usize,
    /// Convergence tolerance for $\sigma_i$ binary search (`default: 1e-5`).
    pub sigma_tolerance: f64,
    /// Explicit deterministic seed for negative sampling and edge subsampling (`default: 42`).
    pub seed: u64,
}

impl Default for GuMapConfig {
    fn default() -> Self {
        Self {
            n_epochs: 200,
            negative_sample_rate: 5,
            learning_rate: 1.0,
            min_dist: 0.1,
            spread: 1.0,
            a: None,
            b: None,
            sigma_iters: 64,
            sigma_tolerance: 1e-5,
            seed: 42,
        }
    }
}

impl GuMapConfig {
    /// Validates all hyperparameters and resolves `(a, b)` curve parameters if omitted.
    pub fn validate(&self) -> CoreResult<()> {
        let _ = self.resolve_ab()?;
        if !self.learning_rate.is_finite() {
            return Err(CoreError::non_finite("GuMapConfig::learning_rate"));
        }
        if self.learning_rate <= 0.0 {
            return Err(CoreError::invalid_input(format!(
                "learning_rate must be > 0.0, got {}",
                self.learning_rate
            )));
        }
        if self.sigma_iters == 0 {
            return Err(CoreError::invalid_input("sigma_iters must be >= 1"));
        }
        if !self.sigma_tolerance.is_finite() {
            return Err(CoreError::non_finite("GuMapConfig::sigma_tolerance"));
        }
        if self.sigma_tolerance <= 0.0 {
            return Err(CoreError::invalid_input(format!(
                "sigma_tolerance must be > 0.0, got {}",
                self.sigma_tolerance
            )));
        }
        Ok(())
    }

    /// Validates `(min_dist, spread, a, b)` and returns effective `(a, b)`.
    pub fn resolve_ab(&self) -> CoreResult<(f64, f64)> {
        if !self.min_dist.is_finite() {
            return Err(CoreError::non_finite("GuMapConfig::min_dist"));
        }
        if self.min_dist < 0.0 {
            return Err(CoreError::invalid_input(format!(
                "min_dist must be >= 0.0, got {}",
                self.min_dist
            )));
        }
        if !self.spread.is_finite() {
            return Err(CoreError::non_finite("GuMapConfig::spread"));
        }
        if self.spread <= 0.0 {
            return Err(CoreError::invalid_input(format!(
                "spread must be > 0.0, got {}",
                self.spread
            )));
        }
        if let Some(a_val) = self.a {
            if !a_val.is_finite() {
                return Err(CoreError::non_finite("GuMapConfig::a"));
            }
            if a_val <= 0.0 {
                return Err(CoreError::invalid_input(format!(
                    "a must be > 0.0, got {a_val}"
                )));
            }
        }
        if let Some(b_val) = self.b {
            if !b_val.is_finite() {
                return Err(CoreError::non_finite("GuMapConfig::b"));
            }
            if b_val <= 0.0 {
                return Err(CoreError::invalid_input(format!(
                    "b must be > 0.0, got {b_val}"
                )));
            }
        }

        let (eff_a, eff_b) = match (self.a, self.b) {
            (Some(a_val), Some(b_val)) => (a_val, b_val),
            _ => {
                let (fitted_a, fitted_b) = find_ab_params(self.min_dist, self.spread)?;
                (self.a.unwrap_or(fitted_a), self.b.unwrap_or(fitted_b))
            }
        };
        if !eff_a.is_finite() || eff_a <= 0.0 || !eff_b.is_finite() || eff_b <= 0.0 {
            return Err(CoreError::numerical_breakdown(
                "resolved UMAP (a, b) parameters must be finite and positive",
            ));
        }
        Ok((eff_a, eff_b))
    }
}

/// Cooperative, zero-per-step-allocation Graph UMAP (`GuMap`) layout solver.
///
/// Uses the `dist_sq.powf(b)` identity ($d^{2b} = (d^2)^b$, $d^{2b-2} = (d^2)^b / d^2$)
/// to avoid per-pair square roots in both attractive and repulsive gradient evaluations.
#[derive(Debug, Clone)]
pub struct GuMapSolver<S: Scalar = f64> {
    config: GuMapConfig,
    num_nodes: usize,
    a: f64,
    b: f64,
    edge_src: Vec<u32>,
    edge_tgt: Vec<u32>,
    edge_weight: Vec<f64>,
    epoch: usize,
    status: SolverStatus,
    rng: DeterministicRng,
    _marker: std::marker::PhantomData<S>,
}

impl<S: Scalar> GuMapSolver<S> {
    /// Constructs a [`GuMapSolver`] directly from a precomputed fuzzy simplicial set [`SparseProbabilities`].
    pub fn from_fuzzy_set(probs: SparseProbabilities, config: GuMapConfig) -> CoreResult<Self> {
        config.validate()?;
        validate_sparse_probabilities(&probs)?;
        let (a, b) = config.resolve_ab()?;

        let n = probs.n_nodes;
        let mut raw_edges: Vec<(u32, u32, f64)> = Vec::with_capacity(probs.nnz());
        for i in 0..n {
            let i_u32 = i as u32;
            let (cols, vals) = probs.row(i);
            for (&j, &w) in cols.iter().zip(vals.iter()) {
                if j != i_u32 && w > 0.0 {
                    raw_edges.push((i_u32, j, w));
                }
            }
        }
        // Emit canonical undirected pairs `(u, v)` and `(v, u)` (`u < v`) consecutively
        // instead of row-major order, avoiding directional sweep bias on ordered vertex IDs.
        raw_edges.sort_unstable_by_key(|&(u, v, _)| (u.min(v), u.max(v), u > v));

        let mut edge_src = Vec::with_capacity(raw_edges.len());
        let mut edge_tgt = Vec::with_capacity(raw_edges.len());
        let mut edge_weight = Vec::with_capacity(raw_edges.len());
        for (u, v, w) in raw_edges {
            edge_src.push(u);
            edge_tgt.push(v);
            edge_weight.push(w);
        }

        let status = if config.n_epochs == 0 {
            SolverStatus::Completed
        } else {
            SolverStatus::Initialized
        };

        Ok(Self {
            config,
            num_nodes: n,
            a,
            b,
            edge_src,
            edge_tgt,
            edge_weight,
            epoch: 0,
            status,
            rng: DeterministicRng::from_seed_u64(config.seed),
            _marker: std::marker::PhantomData,
        })
    }

    /// Constructs a [`GuMapSolver`] from a validated [`KnnGraph`] via [`compute_fuzzy_simplicial_set`].
    pub fn from_knn(knn: &KnnGraph, config: GuMapConfig) -> CoreResult<Self> {
        config.validate()?;
        let probs = compute_fuzzy_simplicial_set(knn, config.sigma_iters, config.sigma_tolerance)?;
        Self::from_fuzzy_set(probs, config)
    }

    /// Constructs a [`GuMapSolver`] from a [`CsrGraph`] and precomputed
    /// [`DogVaradhanLandmarks`] using $O(|E|)$ hub-safe 1+2-hop pairs, landmark ALT
    /// distances, `eps_dist` flooring, 1+2-hop priority CSR, and fuzzy simplicial sets.
    pub fn from_landmarks<W: Scalar>(
        graph: &CsrGraph<W>,
        landmarks: &DogVaradhanLandmarks,
        max_2hop_per_node: usize,
        config: GuMapConfig,
    ) -> CoreResult<Self> {
        config.validate()?;
        let n = graph.num_nodes();
        if u32::try_from(n).is_err() {
            return Err(CoreError::invalid_csr("n_nodes exceeds u32::MAX"));
        }
        if landmarks.num_nodes() != n {
            return Err(CoreError::dimension_mismatch(
                n,
                landmarks.num_nodes(),
                "GuMapSolver::from_landmarks::landmarks",
            ));
        }
        let k_lm = landmarks.num_landmarks();
        if n > 0 && k_lm == 0 {
            return Err(CoreError::invalid_input(
                "landmarks must have at least 1 landmark when num_nodes > 0",
            ));
        }
        let expected = n
            .checked_mul(k_lm)
            .ok_or_else(|| CoreError::invalid_input("landmark table size overflow"))?;
        if landmarks.d_dog().len() != expected {
            return Err(CoreError::dimension_mismatch(
                expected,
                landmarks.d_dog().len(),
                "DogVaradhanLandmarks::d_dog",
            ));
        }
        if landmarks.d_dog().iter().any(|v| !v.is_finite()) {
            return Err(CoreError::non_finite("DogVaradhanLandmarks::d_dog"));
        }
        if let Some(dc) = landmarks.d_coarse() {
            if dc.len() != expected {
                return Err(CoreError::dimension_mismatch(
                    expected,
                    dc.len(),
                    "DogVaradhanLandmarks::d_coarse",
                ));
            }
            if dc.iter().any(|v| !v.is_finite()) {
                return Err(CoreError::non_finite("DogVaradhanLandmarks::d_coarse"));
            }
        }
        for &lm in landmarks.landmarks() {
            if (lm as usize) >= n {
                return Err(CoreError::index_out_of_bounds(
                    lm as usize,
                    n,
                    "DogVaradhanLandmarks::landmarks",
                ));
            }
        }

        if n <= 1 {
            return Self::from_fuzzy_set(SparseProbabilities::empty(n), config);
        }

        let pairs = extract_sparse_1hop_2hop_pairs(graph, max_2hop_per_node, config.seed)?;
        let mut d1 = landmarks.alt_pair_distances(&pairs.u1, &pairs.v1)?;
        let mut d2 = landmarks.alt_pair_distances(&pairs.u2, &pairs.v2)?;
        for d in &mut d1 {
            *d = d.max(DEFAULT_EPS_DIST);
        }
        for d in &mut d2 {
            *d = d.max(DEFAULT_EPS_DIST);
        }
        let knn = build_gumap_12hop_priority_csr(
            n,
            &pairs.u1,
            &pairs.v1,
            &d1,
            &pairs.u2,
            &pairs.v2,
            &d2,
            DEFAULT_TIER2_MARGIN,
        )?;
        Self::from_knn(&knn, config)
    }

    /// Constructs a [`GuMapSolver`] from a [`CsrGraph`] using $O(|E|)$ hub-safe 1+2-hop pairs,
    /// regularized Chebyshev DoG Varadhan landmark ALT distances, `eps_dist` distance flooring,
    /// 1+2-hop priority CSR construction, and fuzzy simplicial set computation.
    pub fn from_graph_dog<W: Scalar>(
        graph: &CsrGraph<W>,
        max_2hop_per_node: usize,
        dog_config: &DogVaradhanConfig,
        config: GuMapConfig,
    ) -> CoreResult<Self> {
        config.validate()?;
        let n = graph.num_nodes();
        if n <= 1 {
            return Self::from_fuzzy_set(SparseProbabilities::empty(n), config);
        }

        let (landmarks, _dog_stats) =
            compute_regularized_dog_varadhan_landmarks(graph, dog_config)?;
        Self::from_landmarks(graph, &landmarks, max_2hop_per_node, config)
    }

    /// Returns the solver configuration.
    #[inline]
    pub fn config(&self) -> &GuMapConfig {
        &self.config
    }

    /// Returns the expected number of vertices in [`Layout2d`].
    #[inline]
    pub fn num_nodes(&self) -> usize {
        self.num_nodes
    }

    /// Returns the number of stored directed positive probability edges.
    #[inline]
    pub fn num_edges(&self) -> usize {
        self.edge_src.len()
    }

    /// Returns the resolved UMAP low-dimensional curve parameters `(a, b)`.
    #[inline]
    pub fn ab(&self) -> (f64, f64) {
        (self.a, self.b)
    }
}

impl<S: Scalar> IterativeSolver<S> for GuMapSolver<S> {
    fn step(&mut self, layout: &mut Layout2d<S>) -> CoreResult<SolverStatus> {
        if layout.len() != self.num_nodes {
            return Err(CoreError::dimension_mismatch(
                self.num_nodes,
                layout.len(),
                "GuMapSolver::step",
            ));
        }
        layout.validate_finite()?;

        if self.status.is_finished() {
            return Ok(self.status);
        }

        let n = self.num_nodes;
        if n < 2 || self.edge_src.is_empty() {
            self.status = SolverStatus::Converged;
            return Ok(self.status);
        }

        let (xs, ys) = layout.xy_mut();

        // Match legacy GuMap behavior when an uninitialized all-zero layout is supplied.
        if self.epoch == 0 && xs.iter().all(|&v| v == S::ZERO) && ys.iter().all(|&v| v == S::ZERO) {
            for idx in 0..n {
                xs[idx] = S::from_f64(self.rng.next_f64() * 20.0 - 10.0);
                ys[idx] = S::from_f64(self.rng.next_f64() * 20.0 - 10.0);
            }
        }

        let alpha = S::from_f64(
            self.config.learning_rate * (1.0 - (self.epoch as f64) / (self.config.n_epochs as f64)),
        );
        let a_s = S::from_f64(self.a);
        let b_s = S::from_f64(self.b);
        let two_b_s = S::from_f64(2.0 * self.b);
        let two_ab_s = S::from_f64(2.0 * self.a * self.b);
        let four_s = S::from_f64(GRAD_CLAMP);
        let neg_four_s = -four_s;
        let eps_sq_s = S::from_f64(MIN_DIST_SQ_CLAMP);
        let rep_eps_s = S::from_f64(REPULSIVE_DIST_SQ_EPS);

        let neg_rate = self.config.negative_sample_rate;
        let m = self.edge_src.len();

        for idx in 0..m {
            let p_edge = self.edge_weight[idx];
            if p_edge <= 0.0 || (p_edge < 1.0 && self.rng.next_f64() > p_edge) {
                continue;
            }

            let i = self.edge_src[idx] as usize;
            let j = self.edge_tgt[idx] as usize;

            // 1. Attractive gradient via d2.powf(b) without sqrt.
            let dx = xs[i] - xs[j];
            let dy = ys[i] - ys[j];
            let d2 = (dx * dx + dy * dy).max(eps_sq_s);
            let d_2b = d2.powf(b_s);
            let d_pow = d_2b / d2;
            let raw_grad = -two_ab_s * d_pow / (S::ONE + a_s * d_2b);
            if raw_grad.is_finite() {
                let grad_coeff = raw_grad.clamp(neg_four_s, four_s);
                let delta_x = alpha * grad_coeff * dx;
                let delta_y = alpha * grad_coeff * dy;
                let new_ix = xs[i] + delta_x;
                let new_iy = ys[i] + delta_y;
                let new_jx = xs[j] - delta_x;
                let new_jy = ys[j] - delta_y;
                if new_ix.is_finite() {
                    xs[i] = new_ix;
                }
                if new_iy.is_finite() {
                    ys[i] = new_iy;
                }
                if new_jx.is_finite() {
                    xs[j] = new_jx;
                }
                if new_jy.is_finite() {
                    ys[j] = new_jy;
                }
            }

            // 2. Deterministic negative sampling via d2.powf(b) without sqrt.
            for _ in 0..neg_rate {
                let k = self.rng.gen_range_usize(n);
                if k == i {
                    continue;
                }
                let r_dx = xs[i] - xs[k];
                let r_dy = ys[i] - ys[k];
                let r_d2_raw = r_dx * r_dx + r_dy * r_dy;
                let r_d2 = r_d2_raw.max(eps_sq_s);
                let r_d_2b = r_d2.powf(b_s);
                let denom = (r_d2_raw + rep_eps_s) * (S::ONE + a_s * r_d_2b);
                let rep_coeff = if denom > S::ZERO && denom.is_finite() {
                    let raw_rep = two_b_s / denom;
                    if raw_rep.is_finite() {
                        raw_rep.clamp(S::ZERO, four_s)
                    } else {
                        four_s
                    }
                } else {
                    S::ZERO
                };
                let r_delta_x = alpha * rep_coeff * r_dx;
                let r_delta_y = alpha * rep_coeff * r_dy;
                let new_ix = xs[i] + r_delta_x;
                let new_iy = ys[i] + r_delta_y;
                if new_ix.is_finite() {
                    xs[i] = new_ix;
                }
                if new_iy.is_finite() {
                    ys[i] = new_iy;
                }
            }
        }

        self.epoch += 1;
        self.status = if self.epoch >= self.config.n_epochs {
            SolverStatus::Completed
        } else {
            SolverStatus::Running
        };
        Ok(self.status)
    }

    #[inline]
    fn status(&self) -> SolverStatus {
        self.status
    }

    #[inline]
    fn iteration(&self) -> usize {
        self.epoch
    }

    #[inline]
    fn total_iterations(&self) -> usize {
        self.config.n_epochs
    }
}

fn validate_sparse_probabilities(probs: &SparseProbabilities) -> CoreResult<()> {
    let n = probs.n_nodes;
    if u32::try_from(n).is_err() {
        return Err(CoreError::invalid_csr("n_nodes exceeds u32::MAX"));
    }
    if probs.indptr.len() != n + 1 {
        return Err(CoreError::dimension_mismatch(
            n + 1,
            probs.indptr.len(),
            "SparseProbabilities::indptr",
        ));
    }
    if probs.indptr[0] != 0 {
        return Err(CoreError::invalid_csr(format!(
            "SparseProbabilities indptr[0] must be 0, got {}",
            probs.indptr[0]
        )));
    }
    for row in 0..n {
        if probs.indptr[row + 1] < probs.indptr[row] {
            return Err(CoreError::invalid_csr(format!(
                "SparseProbabilities indptr must be non-decreasing at row {row}: {} > {}",
                probs.indptr[row],
                probs.indptr[row + 1]
            )));
        }
    }
    let nnz = probs.indptr[n];
    if probs.indices.len() != nnz {
        return Err(CoreError::dimension_mismatch(
            nnz,
            probs.indices.len(),
            "SparseProbabilities::indices",
        ));
    }
    if probs.values.len() != nnz {
        return Err(CoreError::dimension_mismatch(
            nnz,
            probs.values.len(),
            "SparseProbabilities::values",
        ));
    }
    for (&idx, &val) in probs.indices.iter().zip(probs.values.iter()) {
        if (idx as usize) >= n {
            return Err(CoreError::index_out_of_bounds(
                idx as usize,
                n,
                "SparseProbabilities::indices",
            ));
        }
        if !val.is_finite() {
            return Err(CoreError::non_finite("SparseProbabilities::values"));
        }
        if val < 0.0 {
            return Err(CoreError::invalid_input(format!(
                "SparseProbabilities values must be >= 0.0, got {val}"
            )));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn barbell_graph(clique_size: usize) -> CsrGraph<f64> {
        let n = 2 * clique_size;
        let mut edges = Vec::new();
        for c in 0..2 {
            let off = (c * clique_size) as u32;
            for i in 0..clique_size as u32 {
                for j in (i + 1)..clique_size as u32 {
                    edges.push((off + i, off + j));
                }
            }
        }
        edges.push((0, clique_size as u32));
        CsrGraph::from_edges(n, &edges, None, false, false).unwrap()
    }

    fn grid_graph(rows: usize, cols: usize) -> CsrGraph<f64> {
        let n = rows * cols;
        let mut edges = Vec::new();
        for r in 0..rows {
            for c in 0..cols {
                let u = (r * cols + c) as u32;
                if c + 1 < cols {
                    edges.push((u, u + 1));
                }
                if r + 1 < rows {
                    edges.push((u, u + cols as u32));
                }
            }
        }
        CsrGraph::from_edges(n, &edges, None, false, false).unwrap()
    }

    fn seeded_layout<S: Scalar>(n: usize, scale: f64, seed: u64) -> Layout2d<S> {
        let mut rng = DeterministicRng::from_seed_u64(seed);
        let mut x = Vec::with_capacity(n);
        let mut y = Vec::with_capacity(n);
        for _ in 0..n {
            x.push(S::from_f64((rng.next_f64() * 2.0 - 1.0) * scale));
            y.push(S::from_f64((rng.next_f64() * 2.0 - 1.0) * scale));
        }
        Layout2d::from_xy(x, y).unwrap()
    }

    #[test]
    fn test_gumap_config_defaults_ab_and_validation() {
        let cfg = GuMapConfig::default();
        assert_eq!(cfg.n_epochs, 200);
        assert_eq!(cfg.negative_sample_rate, 5);
        assert_eq!(cfg.learning_rate, 1.0);
        assert_eq!(cfg.min_dist, 0.1);
        assert_eq!(cfg.spread, 1.0);
        assert_eq!(cfg.a, None);
        assert_eq!(cfg.b, None);
        assert_eq!(cfg.sigma_iters, 64);
        assert_eq!(cfg.sigma_tolerance, 1e-5);
        assert_eq!(cfg.seed, 42);
        cfg.validate().unwrap();
        let (a, b) = cfg.resolve_ab().unwrap();
        assert!((a - 1.577).abs() < 0.02);
        assert!((b - 0.895).abs() < 0.02);

        let bad_cases = [
            GuMapConfig {
                learning_rate: 0.0,
                ..cfg
            },
            GuMapConfig {
                learning_rate: f64::NAN,
                ..cfg
            },
            GuMapConfig {
                min_dist: -0.1,
                ..cfg
            },
            GuMapConfig { spread: 0.0, ..cfg },
            GuMapConfig {
                a: Some(-1.0),
                ..cfg
            },
            GuMapConfig {
                b: Some(0.0),
                ..cfg
            },
            GuMapConfig {
                sigma_iters: 0,
                ..cfg
            },
            GuMapConfig {
                sigma_tolerance: 0.0,
                ..cfg
            },
        ];
        for bad in bad_cases {
            assert!(bad.validate().is_err());
        }
    }

    #[test]
    fn test_gumap_step_vs_run_bit_exact_f64_and_f32() {
        let grid = grid_graph(4, 4);
        let cfg = GuMapConfig {
            n_epochs: 25,
            seed: 99,
            ..Default::default()
        };
        let dog_cfg = DogVaradhanConfig::default();

        // f64 bit-exact parity
        let mut s_run = GuMapSolver::<f64>::from_graph_dog(&grid, 32, &dog_cfg, cfg).unwrap();
        let mut s_step = GuMapSolver::<f64>::from_graph_dog(&grid, 32, &dog_cfg, cfg).unwrap();
        let mut l_run = seeded_layout::<f64>(grid.num_nodes(), 1.0, 17);
        let mut l_step = l_run.clone();

        assert_eq!(s_run.status(), SolverStatus::Initialized);
        assert_eq!(s_run.total_iterations(), 25);
        assert_eq!(s_run.run(&mut l_run).unwrap(), SolverStatus::Completed);

        assert_eq!(
            s_step.run_steps(&mut l_step, 7).unwrap(),
            SolverStatus::Running
        );
        assert_eq!(s_step.iteration(), 7);
        while !s_step.status().is_finished() {
            s_step.step(&mut l_step).unwrap();
        }
        assert_eq!(s_step.status(), SolverStatus::Completed);
        assert_eq!(s_step.iteration(), 25);

        for i in 0..grid.num_nodes() {
            assert_eq!(l_run.x()[i].to_bits(), l_step.x()[i].to_bits());
            assert_eq!(l_run.y()[i].to_bits(), l_step.y()[i].to_bits());
        }

        // f32 bit-exact parity
        let mut s32_run = GuMapSolver::<f32>::from_graph_dog(&grid, 32, &dog_cfg, cfg).unwrap();
        let mut s32_step = GuMapSolver::<f32>::from_graph_dog(&grid, 32, &dog_cfg, cfg).unwrap();
        let mut l32_run = seeded_layout::<f32>(grid.num_nodes(), 1.0, 17);
        let mut l32_step = l32_run.clone();
        s32_run.run(&mut l32_run).unwrap();
        for _ in 0..25 {
            s32_step.step(&mut l32_step).unwrap();
        }
        assert_eq!(s32_step.status(), SolverStatus::Completed);
        for i in 0..grid.num_nodes() {
            assert_eq!(l32_run.x()[i].to_bits(), l32_step.x()[i].to_bits());
            assert_eq!(l32_run.y()[i].to_bits(), l32_step.y()[i].to_bits());
        }
    }

    #[test]
    fn test_gumap_dist_sq_powf_b_and_dog_separates_barbell_clusters() {
        let barbell = barbell_graph(6);
        let cfg = GuMapConfig {
            n_epochs: 150,
            negative_sample_rate: 5,
            learning_rate: 1.0,
            seed: 42,
            ..Default::default()
        };
        let dog_cfg = DogVaradhanConfig::default();
        let mut solver = GuMapSolver::<f64>::from_graph_dog(&barbell, 32, &dog_cfg, cfg).unwrap();
        assert!(solver.num_edges() > 0);
        let (a, b) = solver.ab();
        assert!(a > 0.0 && b > 0.0);

        // Also test starting from all-zeros layout (triggers deterministic spread init).
        let mut layout = Layout2d::<f64>::zeros(barbell.num_nodes());
        assert_eq!(solver.run(&mut layout).unwrap(), SolverStatus::Completed);
        assert!(layout.is_all_finite());

        let c0x = layout.x()[..6].iter().sum::<f64>() / 6.0;
        let c0y = layout.y()[..6].iter().sum::<f64>() / 6.0;
        let c1x = layout.x()[6..].iter().sum::<f64>() / 6.0;
        let c1y = layout.y()[6..].iter().sum::<f64>() / 6.0;
        let centroid_dist = (c0x - c1x).hypot(c0y - c1y);
        let mut intra = 0.0_f64;
        for i in 0..6 {
            intra += (layout.x()[i] - c0x).hypot(layout.y()[i] - c0y);
            intra += (layout.x()[6 + i] - c1x).hypot(layout.y()[6 + i] - c1y);
        }
        intra /= 12.0;
        assert!(
            centroid_dist > intra * 1.5,
            "expected barbell cluster separation, got centroid_dist={centroid_dist}, intra={intra}"
        );
    }

    #[test]
    fn test_gumap_edge_cases_and_validation_errors() {
        let cfg = GuMapConfig::default();

        // Empty and singleton graphs via from_graph_dog
        let empty = CsrGraph::<f64>::from_edges(0, &[], None, false, false).unwrap();
        let mut s_empty =
            GuMapSolver::<f64>::from_graph_dog(&empty, 16, &DogVaradhanConfig::default(), cfg)
                .unwrap();
        let mut l_empty = Layout2d::<f64>::zeros(0);
        assert_eq!(s_empty.run(&mut l_empty).unwrap(), SolverStatus::Converged);

        let single = CsrGraph::<f64>::from_edges(1, &[], None, false, false).unwrap();
        let mut s_single =
            GuMapSolver::<f64>::from_graph_dog(&single, 16, &DogVaradhanConfig::default(), cfg)
                .unwrap();
        let mut l_single = Layout2d::from_xy(vec![2.0], vec![-3.0]).unwrap();
        assert_eq!(
            s_single.run(&mut l_single).unwrap(),
            SolverStatus::Converged
        );
        assert_eq!(l_single.x(), &[2.0]);
        assert_eq!(l_single.y(), &[-3.0]);

        // Zero epochs completes immediately
        let grid = grid_graph(3, 3);
        let cfg_zero = GuMapConfig { n_epochs: 0, ..cfg };
        let s_zero =
            GuMapSolver::<f64>::from_graph_dog(&grid, 16, &DogVaradhanConfig::default(), cfg_zero)
                .unwrap();
        assert_eq!(s_zero.status(), SolverStatus::Completed);
        assert_eq!(s_zero.total_iterations(), 0);

        // Dimension mismatch & non-finite layout rejected
        let mut s_grid = GuMapSolver::<f64>::from_knn(
            &KnnGraph::new_validated(2, vec![0, 1, 2], vec![1, 0], vec![1.0, 1.0]).unwrap(),
            cfg,
        )
        .unwrap();
        let mut l_wrong = Layout2d::<f64>::zeros(3);
        assert!(matches!(
            s_grid.step(&mut l_wrong),
            Err(CoreError::DimensionMismatch { .. })
        ));
        let mut l_nan = Layout2d::from_xy(vec![0.0, f64::NAN], vec![0.0, 1.0]).unwrap();
        assert!(matches!(
            s_grid.step(&mut l_nan),
            Err(CoreError::NonFiniteValue { .. })
        ));

        // Invalid SparseProbabilities rejected
        let bad_probs = SparseProbabilities {
            n_nodes: 2,
            indptr: vec![0, 1, 1],
            indices: vec![2],
            values: vec![0.5],
        };
        assert!(matches!(
            GuMapSolver::<f64>::from_fuzzy_set(bad_probs, cfg),
            Err(CoreError::IndexOutOfBounds { .. })
        ));
    }

    #[test]
    fn test_gumap_from_fuzzy_set_interleaves_symmetric_pairs() {
        // Symmetric 4-node graph plus one asymmetric edge to confirm both cases work cleanly.
        let probs = SparseProbabilities {
            n_nodes: 4,
            indptr: vec![0, 2, 4, 6, 7],
            indices: vec![1, 3, 0, 2, 1, 3, 0],
            values: vec![0.8, 0.5, 0.8, 0.6, 0.6, 0.4, 0.5],
        };
        let solver = GuMapSolver::<f64>::from_fuzzy_set(probs, GuMapConfig::default()).unwrap();
        let got: Vec<(u32, u32)> = solver
            .edge_src
            .iter()
            .copied()
            .zip(solver.edge_tgt.iter().copied())
            .collect();
        assert_eq!(
            got,
            vec![(0, 1), (1, 0), (0, 3), (3, 0), (1, 2), (2, 1), (2, 3)]
        );
    }

    #[test]
    fn test_gumap_from_landmarks_bit_exact_and_validation() {
        let grid = grid_graph(4, 4);
        let n = grid.num_nodes();
        let dog_cfg = DogVaradhanConfig::default();
        let cfg = GuMapConfig {
            n_epochs: 20,
            seed: 55,
            ..Default::default()
        };
        let (landmarks, _) = compute_regularized_dog_varadhan_landmarks(&grid, &dog_cfg).unwrap();

        let mut s_dog = GuMapSolver::<f64>::from_graph_dog(&grid, 32, &dog_cfg, cfg).unwrap();
        let mut s_lm = GuMapSolver::<f64>::from_landmarks(&grid, &landmarks, 32, cfg).unwrap();
        let mut l_dog = seeded_layout::<f64>(n, 1.0, 21);
        let mut l_lm = l_dog.clone();
        s_dog.run(&mut l_dog).unwrap();
        s_lm.run(&mut l_lm).unwrap();
        for i in 0..n {
            assert_eq!(l_dog.x()[i].to_bits(), l_lm.x()[i].to_bits());
            assert_eq!(l_dog.y()[i].to_bits(), l_lm.y()[i].to_bits());
        }

        // Validation errors
        let mut bad_nodes = landmarks.clone();
        bad_nodes.num_nodes = n + 1;
        assert!(matches!(
            GuMapSolver::<f64>::from_landmarks(&grid, &bad_nodes, 32, cfg),
            Err(CoreError::DimensionMismatch { .. })
        ));

        let mut bad_empty_lm = landmarks.clone();
        bad_empty_lm.landmarks.clear();
        bad_empty_lm.d_dog.clear();
        bad_empty_lm.d_coarse = None;
        assert!(matches!(
            GuMapSolver::<f64>::from_landmarks(&grid, &bad_empty_lm, 32, cfg),
            Err(CoreError::InvalidInput(_))
        ));

        let mut bad_nan = landmarks.clone();
        bad_nan.d_dog[0] = f64::NAN;
        assert!(matches!(
            GuMapSolver::<f64>::from_landmarks(&grid, &bad_nan, 32, cfg),
            Err(CoreError::NonFiniteValue { .. })
        ));

        let mut bad_idx = landmarks;
        bad_idx.landmarks[0] = n as u32;
        assert!(matches!(
            GuMapSolver::<f64>::from_landmarks(&grid, &bad_idx, 32, cfg),
            Err(CoreError::IndexOutOfBounds { .. })
        ));
    }
}
