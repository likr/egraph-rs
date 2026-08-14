use crate::TsNetBuilder;
use ndarray::Array2;
use num_traits::Float;
use petgraph_drawing::{Drawing, DrawingEuclidean2d, DrawingIndex, DrawingValue};
use petgraph_linalg_kernel::Distance;

/// tsNET graph layout algorithm.
///
/// This structure implements the tsNET layout algorithm by optimizing a t-SNE-like
/// cost function with early compression and repulsion terms.
///
/// # References
///
/// Kruiger, J. F., Rauber, P. E., Martins, R. M., Kerren, A., Kobourov, S., & Telea, A. C. (2017).
/// Graph Layouts by t-SNE. *Computer Graphics Forum*, 36(3), 283-294.
#[derive(Debug, Clone)]
pub struct TsNet<S> {
    /// Target perplexity for t-SNE probability distribution
    pub perplexity: S,
    /// Number of iterations for Stage 1 (early exaggeration)
    pub iterations_stage1: usize,
    /// Exaggeration factor for Stage 1
    pub exaggeration: S,
    /// Number of iterations for Stage 2 (early compression / untangling)
    pub iterations_stage2: usize,
    /// Compression weight lambda_c for Stage 2
    pub lambda_c_stage2: S,
    /// Number of iterations for Stage 3 (final adjustment / refinement)
    pub iterations_stage3: usize,
    /// Compression weight lambda_c for Stage 3
    pub lambda_c_stage3: S,
    /// Repulsion weight lambda_r for Stage 3
    pub lambda_r_stage3: S,
    /// Learning rate for momentum-based gradient descent
    pub learning_rate: S,
    /// Momentum parameter in `[0, 1)`
    pub momentum: S,
    /// Power exponent for input space distance matrix (default: 2.0)
    pub power: S,
    /// Repulsion offset parameter to prevent division by zero
    pub epsilon_r: S,
    /// Maximum number of binary search iterations for sigma_i
    pub sigma_iters: usize,
    /// Tolerance threshold for perplexity binary search convergence
    pub sigma_tolerance: S,
}

impl<S> TsNet<S>
where
    S: DrawingValue
        + Float
        + std::iter::Sum
        + std::ops::AddAssign
        + Default
        + ndarray::ScalarOperand,
{
    /// Creates a new `TsNet` instance with default hyperparameters.
    pub fn new() -> Self {
        TsNetBuilder::new().build().unwrap()
    }

    /// Creates a new `TsNetBuilder` for configuring hyperparameters.
    pub fn builder() -> TsNetBuilder<S> {
        TsNetBuilder::new()
    }

    /// Runs the tsNET layout algorithm on the provided drawing using the distance matrix.
    ///
    /// The algorithm operates in 3 distinct dynamic optimization stages:
    /// - **Stage 1 (Early Exaggeration)**: Amplifies joint probabilities to separate distinct clusters.
    /// - **Stage 2 (Early Compression)**: Applies strong compression force ($\lambda_c = 1.2$) to untangle large graph structures.
    /// - **Stage 3 (Final Refinement)**: Applies balanced compression ($\lambda_c = 0.01$) and repulsion ($\lambda_r = 0.6$) to push apart overlapping nodes.
    #[allow(clippy::needless_range_loop)]
    pub fn run<N, D>(&self, drawing: &mut DrawingEuclidean2d<N, S>, distance_matrix: &D)
    where
        N: Copy + Eq + std::hash::Hash + DrawingIndex,
        D: Distance<N, S> + ?Sized,
    {
        let n = drawing.len();
        if n < 2 {
            return;
        }

        // Clamp perplexity to at most N - 1 - 0.01 and at least 1.0 to ensure valid entropy target
        let perplexity = self
            .perplexity
            .min(S::from_usize(n - 1).unwrap() - S::from_f32(0.01).unwrap())
            .max(S::one());
        let entropy_target = perplexity.ln();
        let power = self.power;

        // Step 2: Compute high-dimensional probability distribution P via binary search
        let mut p = Array2::zeros((n, n));

        for i in 0..n {
            let mut beta = S::one();
            let mut beta_min = S::zero();
            let mut beta_max = S::from_f32(1e12).unwrap();

            for _ in 0..self.sigma_iters {
                let mut max_neg_dp = S::neg_infinity();

                // Find maximum exponent for numerical stability (log-sum-exp trick)
                for j in 0..n {
                    if i != j {
                        let d = distance_matrix.get_by_index(i, j);
                        let dp = d.powf(power);
                        let neg_dp = -beta * dp;
                        if neg_dp > max_neg_dp {
                            max_neg_dp = neg_dp;
                        }
                    }
                }

                // Compute unnormalized Gaussian kernel weights
                let mut sum_w = S::zero();
                let mut weights = vec![S::zero(); n];
                for j in 0..n {
                    if i != j {
                        let d = distance_matrix.get_by_index(i, j);
                        let dp = d.powf(power);
                        let w = (-beta * dp - max_neg_dp).exp();
                        weights[j] = w;
                        sum_w += w;
                    }
                }

                sum_w = sum_w.max(S::from_f32(1e-12).unwrap());

                // Compute conditional expectation of powered distance
                let mut sum_dp_p = S::zero();
                for j in 0..n {
                    if i != j {
                        let prob = weights[j] / sum_w;
                        let d = distance_matrix.get_by_index(i, j);
                        let dp = d.powf(power);
                        sum_dp_p += prob * dp;
                    }
                }

                // Calculate Shannon entropy in nats
                let h = beta * sum_dp_p + max_neg_dp + sum_w.ln();
                let h_diff = h - entropy_target;

                if h_diff.abs() < self.sigma_tolerance {
                    break;
                }

                if h_diff > S::zero() {
                    // Entropy is too high; distribution is too uniform -> increase beta (decrease sigma)
                    beta_min = beta;
                    if beta_max == S::from_f32(1e12).unwrap() {
                        beta *= S::from_f32(2.0).unwrap();
                    } else {
                        beta = (beta_min + beta_max) / S::from_f32(2.0).unwrap();
                    }
                } else {
                    // Entropy is too low; distribution is too peaky -> decrease beta (increase sigma)
                    beta_max = beta;
                    beta = (beta_min + beta_max) / S::from_f32(2.0).unwrap();
                }
            }

            // Compute final normalized conditional probabilities for node i
            let mut max_neg_dp = S::neg_infinity();
            for j in 0..n {
                if i != j {
                    let d = distance_matrix.get_by_index(i, j);
                    let dp = d.powf(power);
                    let neg_dp = -beta * dp;
                    if neg_dp > max_neg_dp {
                        max_neg_dp = neg_dp;
                    }
                }
            }

            let mut sum_w = S::zero();
            let mut weights = vec![S::zero(); n];
            for j in 0..n {
                if i != j {
                    let d = distance_matrix.get_by_index(i, j);
                    let dp = d.powf(power);
                    let w = (-beta * dp - max_neg_dp).exp();
                    weights[j] = w;
                    sum_w += w;
                }
            }

            sum_w = sum_w.max(S::from_f32(1e-12).unwrap());
            for j in 0..n {
                if i != j {
                    p[[i, j]] = weights[j] / sum_w;
                }
            }
        }

        // Convert conditional probabilities to symmetrized joint probabilities P
        let mut p_joint = Array2::zeros((n, n));
        for i in 0..n {
            for j in 0..n {
                if i != j {
                    p_joint[[i, j]] = (p[[i, j]] + p[[j, i]]) / S::from_usize(2 * n).unwrap();
                }
            }
        }

        let mut velocity = Array2::zeros((n, 2));

        // Step 4 & 5: Three-stage dynamic optimization using momentum gradient descent
        // Stage 1: Early Exaggeration
        if self.iterations_stage1 > 0 {
            let p_exaggerated = &p_joint * self.exaggeration;
            self.optimize_stage(
                drawing,
                &p_exaggerated,
                &mut velocity,
                self.iterations_stage1,
                S::one(),
                S::from_f32(0.1).unwrap(),
                S::zero(),
            );
        }

        // Stage 2: Early Compression (Untangling)
        if self.iterations_stage2 > 0 {
            self.optimize_stage(
                drawing,
                &p_joint,
                &mut velocity,
                self.iterations_stage2,
                S::one(),
                self.lambda_c_stage2,
                S::zero(),
            );
        }

        // Stage 3: Final Refinement
        if self.iterations_stage3 > 0 {
            self.optimize_stage(
                drawing,
                &p_joint,
                &mut velocity,
                self.iterations_stage3,
                S::one(),
                self.lambda_c_stage3,
                self.lambda_r_stage3,
            );
        }
    }

    /// Performs gradient descent optimization for a specific stage with given loss weights.
    #[allow(clippy::too_many_arguments)]
    fn optimize_stage<N>(
        &self,
        drawing: &mut DrawingEuclidean2d<N, S>,
        p_matrix: &Array2<S>,
        velocity: &mut Array2<S>,
        iterations: usize,
        lambda_kl: S,
        lambda_c: S,
        lambda_r: S,
    ) where
        N: Copy + Eq + std::hash::Hash + DrawingIndex,
    {
        let n = drawing.len();
        let learning_rate = self.learning_rate;
        let momentum = self.momentum;
        let epsilon_r = self.epsilon_r;
        let l_sum = (lambda_kl + lambda_c + lambda_r).max(S::from_f32(1e-12).unwrap());

        for _ in 0..iterations {
            // Step 3: Compute Cauchy-distributed low-dimensional similarities Q
            let mut w_out = Array2::zeros((n, n));
            let mut z_q = S::zero();

            for i in 0..n {
                let yi_0 = drawing.raw_entry(i).0;
                let yi_1 = drawing.raw_entry(i).1;
                for j in 0..n {
                    if i != j {
                        let yj_0 = drawing.raw_entry(j).0;
                        let yj_1 = drawing.raw_entry(j).1;
                        let dx = yi_0 - yj_0;
                        let dy = yi_1 - yj_1;
                        let dist2 = dx * dx + dy * dy;
                        let w = S::one() / (S::one() + dist2);
                        w_out[[i, j]] = w;
                        z_q += w;
                    }
                }
            }

            z_q = z_q.max(S::from_f32(1e-12).unwrap());

            let mut q = Array2::zeros((n, n));
            for i in 0..n {
                for j in 0..n {
                    if i != j {
                        q[[i, j]] = w_out[[i, j]] / z_q;
                    }
                }
            }

            // Step 4: Compute gradients for KL divergence, compression, and repulsion
            let mut gradients = Array2::zeros((n, 2));

            for i in 0..n {
                let yi_0 = drawing.raw_entry(i).0;
                let yi_1 = drawing.raw_entry(i).1;

                // 1. KL Divergence term gradient: 4 * sum_j (p_ij - q_ij) * w_out_ij * (y_i - y_j)
                let mut grad_kl_0 = S::zero();
                let mut grad_kl_1 = S::zero();
                for j in 0..n {
                    if i != j {
                        let yj_0 = drawing.raw_entry(j).0;
                        let yj_1 = drawing.raw_entry(j).1;
                        let dx = yi_0 - yj_0;
                        let dy = yi_1 - yj_1;
                        let factor = S::from_f32(4.0).unwrap()
                            * (p_matrix[[i, j]] - q[[i, j]])
                            * w_out[[i, j]];
                        grad_kl_0 += factor * dx;
                        grad_kl_1 += factor * dy;
                    }
                }

                // 2. Compression term gradient: (1 / N) * y_i
                let grad_c_0 = yi_0 / S::from_usize(n).unwrap();
                let grad_c_1 = yi_1 / S::from_usize(n).unwrap();

                // 3. Repulsion term gradient: - (1 / N^2) * sum_j (y_i - y_j) / (||y_i - y_j|| * (||y_i - y_j|| + eps_r))
                let mut grad_r_0 = S::zero();
                let mut grad_r_1 = S::zero();
                if lambda_r > S::zero() {
                    for j in 0..n {
                        if i != j {
                            let yj_0 = drawing.raw_entry(j).0;
                            let yj_1 = drawing.raw_entry(j).1;
                            let dx = yi_0 - yj_0;
                            let dy = yi_1 - yj_1;
                            let dist = (dx * dx + dy * dy).sqrt();
                            if dist > S::from_f32(1e-6).unwrap() {
                                let factor = -S::one()
                                    / (S::from_usize(n * n).unwrap() * dist * (dist + epsilon_r));
                                grad_r_0 += factor * dx;
                                grad_r_1 += factor * dy;
                            }
                        }
                    }
                }

                // Total normalized gradient
                let g0 =
                    (lambda_kl * grad_kl_0 + lambda_c * grad_c_0 + lambda_r * grad_r_0) / l_sum;
                let g1 =
                    (lambda_kl * grad_kl_1 + lambda_c * grad_c_1 + lambda_r * grad_r_1) / l_sum;

                gradients[[i, 0]] = if g0.is_finite() { g0 } else { S::zero() };
                gradients[[i, 1]] = if g1.is_finite() { g1 } else { S::zero() };
            }

            // Update positions and velocities with momentum
            for i in 0..n {
                velocity[[i, 0]] = momentum * velocity[[i, 0]] - learning_rate * gradients[[i, 0]];
                velocity[[i, 1]] = momentum * velocity[[i, 1]] - learning_rate * gradients[[i, 1]];

                let new_x = drawing.raw_entry(i).0 + velocity[[i, 0]];
                let new_y = drawing.raw_entry(i).1 + velocity[[i, 1]];

                if new_x.is_finite() {
                    drawing.raw_entry_mut(i).0 = new_x;
                }
                if new_y.is_finite() {
                    drawing.raw_entry_mut(i).1 = new_y;
                }
            }
        }
    }
}

impl<S> Default for TsNet<S>
where
    S: DrawingValue
        + Float
        + std::iter::Sum
        + std::ops::AddAssign
        + Default
        + ndarray::ScalarOperand,
{
    fn default() -> Self {
        Self::new()
    }
}
