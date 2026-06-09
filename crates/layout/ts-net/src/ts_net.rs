use ndarray::Array2;
use num_traits::Float;
use petgraph_distance::Distance;
use petgraph_drawing::{Drawing, DrawingEuclidean2d, DrawingIndex, DrawingValue};

/// tsNET graph layout algorithm.
///
/// This structure implements the tsNET layout algorithm by optimizing a t-SNE-like
/// cost function containing KL divergence, early compression, and entropy repulsion terms.
#[derive(Debug, Clone)]
pub struct TsNet<S> {
    /// Target perplexity for t-SNE probability distribution
    pub perplexity: S,
    /// Number of iterations for Stage 2 (compression)
    pub iterations_stage2: usize,
    /// Number of iterations for Stage 3 (refinement)
    pub iterations_stage3: usize,
    /// Learning rate for momentum-based gradient descent
    pub learning_rate: S,
    /// Momentum parameter
    pub momentum: S,
    /// Minimum distance for input space
    pub epsilon_d: S,
    /// Repulsion parameter to prevent zero division
    pub epsilon_r: S,
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
    /// Creates a new TsNet with default values.
    pub fn new() -> Self {
        Self {
            perplexity: S::from_f32(30.0).unwrap(),
            iterations_stage2: 250,
            iterations_stage3: 250,
            learning_rate: S::from_f32(200.0).unwrap(),
            momentum: S::from_f32(0.8).unwrap(),
            epsilon_d: S::from_f32(0.01).unwrap(),
            epsilon_r: S::from_f32(0.05).unwrap(),
        }
    }

    /// Sets the target perplexity.
    pub fn perplexity(&mut self, perplexity: S) -> &mut Self {
        self.perplexity = perplexity;
        self
    }

    /// Sets the number of iterations for Stage 2.
    pub fn iterations_stage2(&mut self, iterations: usize) -> &mut Self {
        self.iterations_stage2 = iterations;
        self
    }

    /// Sets the number of iterations for Stage 3.
    pub fn iterations_stage3(&mut self, iterations: usize) -> &mut Self {
        self.iterations_stage3 = iterations;
        self
    }

    /// Sets the learning rate.
    pub fn learning_rate(&mut self, learning_rate: S) -> &mut Self {
        self.learning_rate = learning_rate;
        self
    }

    /// Sets the momentum parameter.
    pub fn momentum(&mut self, momentum: S) -> &mut Self {
        self.momentum = momentum;
        self
    }

    /// Sets the epsilon_d parameter.
    pub fn epsilon_d(&mut self, epsilon_d: S) -> &mut Self {
        self.epsilon_d = epsilon_d;
        self
    }

    /// Sets the epsilon_r parameter.
    pub fn epsilon_r(&mut self, epsilon_r: S) -> &mut Self {
        self.epsilon_r = epsilon_r;
        self
    }

    /// Runs the tsNET layout algorithm.
    #[allow(clippy::needless_range_loop)]
    pub fn run<N, D>(&self, drawing: &mut DrawingEuclidean2d<N, S>, distance_matrix: &D)
    where
        N: Copy + Eq + std::hash::Hash + DrawingIndex,
        D: Distance<N, S>,
    {
        let n = drawing.len();
        if n < 2 {
            return;
        }

        // Clamp perplexity to at most N - 1.01
        let perplexity = self
            .perplexity
            .min(S::from_usize(n - 1).unwrap() - S::from_f32(0.01).unwrap())
            .max(S::one());
        let entropy_target = perplexity.ln();

        // Step 3: Compute joint probabilities P
        let mut p = Array2::zeros((n, n));

        for i in 0..n {
            let mut beta = S::one();
            let mut beta_min = S::zero();
            let mut beta_max = S::from_f32(1e12).unwrap();

            for _ in 0..50 {
                let mut sum_w = S::zero();
                let mut max_neg_d2 = S::neg_infinity();

                // Find max negative squared distance for numerical stability
                for j in 0..n {
                    if i != j {
                        let d = distance_matrix.get_by_index(i, j);
                        let neg_d2 = -beta * d * d;
                        if neg_d2 > max_neg_d2 {
                            max_neg_d2 = neg_d2;
                        }
                    }
                }

                // Compute sum of weights
                let mut weights = vec![S::zero(); n];
                for j in 0..n {
                    if i != j {
                        let d = distance_matrix.get_by_index(i, j);
                        let w = (-beta * d * d - max_neg_d2).exp();
                        weights[j] = w;
                        sum_w += w;
                    }
                }

                sum_w = sum_w.max(S::from_f32(1e-12).unwrap());

                // Compute conditional expectation of squared distance
                let mut sum_d2_p = S::zero();
                for j in 0..n {
                    if i != j {
                        let prob = weights[j] / sum_w;
                        let d = distance_matrix.get_by_index(i, j);
                        sum_d2_p += prob * d * d;
                    }
                }

                // Calculate Shannon entropy in nats
                let h = beta * sum_d2_p + max_neg_d2 + sum_w.ln();

                let h_diff = h - entropy_target;
                if h_diff.abs() < S::from_f32(1e-5).unwrap() {
                    break;
                }

                if h_diff > S::zero() {
                    beta_min = beta;
                    if beta_max == S::from_f32(1e12).unwrap() {
                        beta *= S::from_f32(2.0).unwrap();
                    } else {
                        beta = (beta_min + beta_max) / S::from_f32(2.0).unwrap();
                    }
                } else {
                    beta_max = beta;
                    beta = (beta_min + beta_max) / S::from_f32(2.0).unwrap();
                }
            }

            // Compute final conditional probabilities for row i
            let mut sum_w = S::zero();
            let mut max_neg_d2 = S::neg_infinity();
            for j in 0..n {
                if i != j {
                    let d = distance_matrix.get_by_index(i, j);
                    let neg_d2 = -beta * d * d;
                    if neg_d2 > max_neg_d2 {
                        max_neg_d2 = neg_d2;
                    }
                }
            }
            let mut weights = vec![S::zero(); n];
            for j in 0..n {
                if i != j {
                    let d = distance_matrix.get_by_index(i, j);
                    let w = (-beta * d * d - max_neg_d2).exp();
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

        // Compute joint probabilities
        let mut p_joint = Array2::zeros((n, n));
        for i in 0..n {
            for j in 0..n {
                if i != j {
                    p_joint[[i, j]] = (p[[i, j]] + p[[j, i]]) / S::from_usize(2 * n).unwrap();
                }
            }
        }

        let mut velocity = Array2::zeros((n, 2));

        // Step 5: Momentum-based gradient descent optimization
        // Stage 2: Compression optimization
        self.optimize_stage(
            drawing,
            &p_joint,
            &mut velocity,
            self.iterations_stage2,
            S::one(),
            S::from_f32(0.1).unwrap(),
            S::zero(),
        );

        // Stage 3: Refinement optimization
        self.optimize_stage(
            drawing,
            &p_joint,
            &mut velocity,
            self.iterations_stage3,
            S::one(),
            S::from_f32(0.01).unwrap(),
            S::from_f32(0.6).unwrap(),
        );
    }

    #[allow(clippy::too_many_arguments)]
    fn optimize_stage<N>(
        &self,
        drawing: &mut DrawingEuclidean2d<N, S>,
        p_joint: &Array2<S>,
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

        for _ in 0..iterations {
            // Compute output similarities Q
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

            // Compute Gradients
            let mut gradients = Array2::zeros((n, 2));

            for i in 0..n {
                let yi_0 = drawing.raw_entry(i).0;
                let yi_1 = drawing.raw_entry(i).1;

                // KL term gradient
                let mut grad_kl_0 = S::zero();
                let mut grad_kl_1 = S::zero();
                for j in 0..n {
                    if i != j {
                        let yj_0 = drawing.raw_entry(j).0;
                        let yj_1 = drawing.raw_entry(j).1;
                        let dx = yi_0 - yj_0;
                        let dy = yi_1 - yj_1;
                        let factor = S::from_f32(4.0).unwrap()
                            * (p_joint[[i, j]] - q[[i, j]])
                            * w_out[[i, j]];
                        grad_kl_0 += factor * dx;
                        grad_kl_1 += factor * dy;
                    }
                }

                // Compression term gradient
                let grad_c_0 = yi_0 / S::from_usize(n).unwrap();
                let grad_c_1 = yi_1 / S::from_usize(n).unwrap();

                // Repulsion term gradient
                let mut grad_r_0 = S::zero();
                let mut grad_r_1 = S::zero();
                for j in 0..n {
                    if i != j {
                        let yj_0 = drawing.raw_entry(j).0;
                        let yj_1 = drawing.raw_entry(j).1;
                        let dx = yi_0 - yj_0;
                        let dy = yi_1 - yj_1;
                        let dist = (dx * dx + dy * dy).sqrt().max(S::from_f32(1e-9).unwrap());
                        let factor =
                            -S::one() / (S::from_usize(n * n).unwrap() * dist * (dist + epsilon_r));
                        grad_r_0 += factor * dx;
                        grad_r_1 += factor * dy;
                    }
                }

                gradients[[i, 0]] =
                    lambda_kl * grad_kl_0 + lambda_c * grad_c_0 + lambda_r * grad_r_0;
                gradients[[i, 1]] =
                    lambda_kl * grad_kl_1 + lambda_c * grad_c_1 + lambda_r * grad_r_1;
            }

            // Update positions and velocities
            for i in 0..n {
                velocity[[i, 0]] = momentum * velocity[[i, 0]] - learning_rate * gradients[[i, 0]];
                velocity[[i, 1]] = momentum * velocity[[i, 1]] - learning_rate * gradients[[i, 1]];

                drawing.raw_entry_mut(i).0 += velocity[[i, 0]];
                drawing.raw_entry_mut(i).1 += velocity[[i, 1]];
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
