use crate::{
    fit_interpolation::{compute_fit_entropy_gradient, compute_fit_kl_repulsion},
    l_ts_net_builder::LTsNetBuilder,
    partial_bfs::compute_partial_bfs_probabilities,
};
use num_traits::Float;
use petgraph::visit::{IntoNeighbors, IntoNodeIdentifiers, NodeCount, NodeIndexable};
use petgraph_drawing::{Drawing, DrawingEuclidean2d, DrawingIndex, DrawingValue};
use rand::prelude::*;

/// L-tsNET (Linear-tsNET) graph layout algorithm.
///
/// Reduces the computational complexity of tsNET strictly to linear time O(N) by combining:
/// - (C0) O(N) Partial Breadth-First Search for high-dimensional Gaussian probabilities.
/// - (C1) O(N) 2D FFT-accelerated interpolation for low-dimensional Student-t KL repulsion forces.
/// - (C2) O(N) 2D FFT-accelerated interpolation for entropy / repulsion gradient.
#[derive(Debug, Clone)]
pub struct LTsNet<S> {
    pub intervals: usize,
    pub interpolation_points: usize,
    pub perplexity: S,
    pub k: usize,
    pub iterations_stage1: usize,
    pub exaggeration: S,
    pub iterations_stage2: usize,
    pub lambda_c_stage2: S,
    pub iterations_stage3: usize,
    pub lambda_c_stage3: S,
    pub lambda_r_stage3: S,
    pub learning_rate: S,
    pub momentum: S,
    pub power: S,
    pub epsilon_r: S,
    pub sigma_iters: usize,
    pub sigma_tolerance: S,
}

impl<S> LTsNet<S>
where
    S: DrawingValue + Float + std::iter::Sum + std::ops::AddAssign + Default,
{
    /// Creates a builder to configure and construct an `LTsNet` layout instance.
    pub fn builder() -> LTsNetBuilder<S> {
        LTsNetBuilder::new()
    }

    /// Creates an `LTsNet` instance with default hyperparameters.
    pub fn new() -> Self {
        LTsNetBuilder::new().build().unwrap()
    }

    /// Executes the L-tsNET layout algorithm on the graph and drawing.
    #[allow(clippy::needless_range_loop)]
    pub fn run<G, N, R>(&self, drawing: &mut DrawingEuclidean2d<N, S>, graph: G, rng: &mut R)
    where
        G: IntoNeighbors + IntoNodeIdentifiers + NodeCount + NodeIndexable,
        G::NodeId: Copy + Eq + std::hash::Hash + DrawingIndex,
        N: Copy + Eq + std::hash::Hash + DrawingIndex,
        R: Rng + ?Sized,
    {
        let n = drawing.len();
        if n < 2 {
            return;
        }

        // Extract coordinates from drawing
        let mut points: Vec<[S; 2]> = Vec::with_capacity(n);
        for i in 0..n {
            let entry = drawing.raw_entry(i);
            points.push([entry.0, entry.1]);
        }

        // Step 1 (C0): Compute sparse symmetrized joint probabilities P via Partial BFS in O(N)
        let sparse_p = compute_partial_bfs_probabilities(
            graph,
            self.perplexity,
            self.k,
            self.power,
            self.sigma_iters,
            self.sigma_tolerance,
            rng,
        );

        let mut velocities = vec![[S::zero(), S::zero()]; n];
        let n_s = S::from_usize(n).unwrap();
        let n_sq_s = n_s * n_s;
        let four_s = S::from_f32(4.0).unwrap();

        // 3-Stage Dynamic Optimization Pipeline
        let stages = [
            (
                self.iterations_stage1,
                self.exaggeration,
                S::one(),
                S::from_f32(0.1).unwrap(),
                S::zero(),
            ),
            (
                self.iterations_stage2,
                S::one(),
                S::one(),
                self.lambda_c_stage2,
                S::zero(),
            ),
            (
                self.iterations_stage3,
                S::one(),
                S::one(),
                self.lambda_c_stage3,
                self.lambda_r_stage3,
            ),
        ];

        for &(iters, exaggeration, lambda_kl, lambda_c, lambda_r) in &stages {
            if iters == 0 {
                continue;
            }

            let lambda_sum = (lambda_kl + lambda_c + lambda_r).max(S::from_f32(1e-12).unwrap());

            for _ in 0..iters {
                // Phase A1: Compute exact sparse attraction forces in O(N)
                let mut attr_forces = vec![[S::zero(), S::zero()]; n];
                for i in 0..n {
                    let xi = points[i][0];
                    let yi = points[i][1];
                    for &(j, p_ij) in &sparse_p[i] {
                        if i == j {
                            continue;
                        }
                        let dx = xi - points[j][0];
                        let dy = yi - points[j][1];
                        let dist_sq = dx * dx + dy * dy;
                        let q_unnorm = S::one() / (S::one() + dist_sq);
                        let p_eff = p_ij * exaggeration;
                        let mult = p_eff * q_unnorm;
                        attr_forces[i][0] += mult * dx;
                        attr_forces[i][1] += mult * dy;
                    }
                }

                // Phase A2: Compute low-dimensional Student-t KL repulsion in O(N) via 2D FFT interpolation (C1)
                let (rep_forces, _total_z) =
                    compute_fit_kl_repulsion(&points, self.intervals, self.interpolation_points);

                // Phase A3: Compute entropy / repulsion gradient in O(N) via 2D FFT interpolation (C2)
                let ent_grad_raw = if lambda_r > S::zero() {
                    compute_fit_entropy_gradient(
                        &points,
                        self.intervals,
                        self.interpolation_points,
                        self.epsilon_r,
                    )
                } else {
                    vec![[S::zero(), S::zero()]; n]
                };

                // Phase B: Compute composite gradient and update coordinates with momentum
                let mut mean_x = S::zero();
                let mut mean_y = S::zero();

                for i in 0..n {
                    let xi = points[i][0];
                    let yi = points[i][1];

                    // KL Divergence gradient = 4 * (F_attr - F_rep)
                    let grad_kl_x = four_s * (attr_forces[i][0] - rep_forces[i][0]);
                    let grad_kl_y = four_s * (attr_forces[i][1] - rep_forces[i][1]);

                    // Compression gradient = (1 / N) * y_i
                    let grad_c_x = xi / n_s;
                    let grad_c_y = yi / n_s;

                    // Entropy gradient = - (1 / N^2) * (h_2 - x_i * h_1)
                    let grad_r_x = if lambda_r > S::zero() {
                        -ent_grad_raw[i][0] / n_sq_s
                    } else {
                        S::zero()
                    };
                    let grad_r_y = if lambda_r > S::zero() {
                        -ent_grad_raw[i][1] / n_sq_s
                    } else {
                        S::zero()
                    };

                    // Combined weighted gradient
                    let grad_x =
                        (lambda_kl * grad_kl_x + lambda_c * grad_c_x + lambda_r * grad_r_x)
                            / lambda_sum;
                    let grad_y =
                        (lambda_kl * grad_kl_y + lambda_c * grad_c_y + lambda_r * grad_r_y)
                            / lambda_sum;

                    // Momentum step: v = momentum * v - learning_rate * grad
                    velocities[i][0] =
                        self.momentum * velocities[i][0] - self.learning_rate * grad_x;
                    velocities[i][1] =
                        self.momentum * velocities[i][1] - self.learning_rate * grad_y;

                    let new_x = points[i][0] + velocities[i][0];
                    let new_y = points[i][1] + velocities[i][1];

                    if new_x.is_finite() {
                        points[i][0] = new_x;
                    }
                    if new_y.is_finite() {
                        points[i][1] = new_y;
                    }

                    mean_x += points[i][0];
                    mean_y += points[i][1];
                }

                // Recenter coordinates to origin
                mean_x /= n_s;
                mean_y /= n_s;
                for p in &mut points {
                    p[0] -= mean_x;
                    p[1] -= mean_y;
                }
            }
        }

        // Store optimized coordinates back into drawing
        for i in 0..n {
            drawing.raw_entry_mut(i).0 = points[i][0];
            drawing.raw_entry_mut(i).1 = points[i][1];
        }
    }
}

impl<S> Default for LTsNet<S>
where
    S: DrawingValue + Float + std::iter::Sum + std::ops::AddAssign + Default,
{
    fn default() -> Self {
        Self::new()
    }
}
