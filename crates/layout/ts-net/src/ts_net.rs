use crate::TsNetBuilder;
use ndarray::Array2;
use num_traits::Float;
use petgraph_drawing::{Drawing, DrawingEuclidean2d, DrawingIndex, DrawingValue};
use petgraph_linalg_kernel::Distance;
use std::collections::VecDeque;

/// tsNET graph layout algorithm.
///
/// This structure implements the tsNET layout algorithm by faithfully porting the reference
/// implementation (HanKruiger/tsNET) which optimizes a t-SNE-like objective function with early
/// compression and repulsion terms.
///
/// # References
///
/// Kruiger, J. F., Rauber, P. E., Martins, R. M., Kerren, A., Kobourov, S., & Telea, A. C. (2017).
/// Graph Layouts by t-SNE. *Computer Graphics Forum*, 36(3), 283-294.
#[derive(Debug, Clone)]
pub struct TsNet<S> {
    /// Target perplexity for t-SNE probability distribution
    pub perplexity: S,
    /// Total number of epochs (iterations)
    pub n_epochs: usize,
    /// Initial learning rate
    pub initial_lr: S,
    /// Final learning rate
    pub final_lr: S,
    /// Epoch at which learning rate switches
    pub lr_switch: Option<usize>,
    /// Initial momentum
    pub initial_momentum: S,
    /// Final momentum
    pub final_momentum: S,
    /// Epoch at which momentum switches
    pub momentum_switch: Option<usize>,
    /// Initial KL divergence weight
    pub initial_l_kl: Option<S>,
    /// Final KL divergence weight
    pub final_l_kl: Option<S>,
    /// Epoch at which lambda_kl switches
    pub l_kl_switch: Option<usize>,
    /// Initial compression weight
    pub initial_l_c: Option<S>,
    /// Final compression weight
    pub final_l_c: Option<S>,
    /// Epoch at which lambda_c switches
    pub l_c_switch: Option<usize>,
    /// Initial repulsion weight
    pub initial_l_r: Option<S>,
    /// Final repulsion weight
    pub final_l_r: Option<S>,
    /// Epoch at which lambda_r switches
    pub l_r_switch: Option<usize>,
    /// Repulsion offset parameter
    pub r_eps: S,
    /// Maximum number of binary search iterations for sigma_i
    pub sigma_iters: usize,
    /// Autostop convergence tolerance threshold
    pub autostop: S,
    /// Sliding window size for autostop convergence checking
    pub window_size: usize,
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
    /// Creates a new `TsNet` instance with default hyperparameters matching `tsnet.py` (tsNET).
    pub fn new() -> Self {
        TsNetBuilder::new().build().unwrap()
    }

    /// Creates a new `TsNet` instance with tsNET* hyperparameters (`initial_l_c = 0.1`).
    pub fn new_star() -> Self {
        TsNetBuilder::new_star().build().unwrap()
    }

    /// Creates a new `TsNetBuilder` for configuring hyperparameters.
    pub fn builder() -> TsNetBuilder<S> {
        TsNetBuilder::new()
    }

    /// Creates a new `TsNetBuilder` initialized with tsNET* hyperparameters.
    pub fn builder_star() -> TsNetBuilder<S> {
        TsNetBuilder::new_star()
    }

    /// Runs the tsNET layout algorithm on the provided drawing using the distance matrix.
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

        // Step 1: Preprocess distance matrix:
        // Symmetrize, zero the diagonal, and normalize by max finite distance matching distance_matrix.py
        let mut max_d = S::zero();
        for i in 0..n {
            for j in 0..n {
                if i != j {
                    let d = distance_matrix.get_by_index(i, j);
                    if d.is_finite() && d > max_d {
                        max_d = d;
                    }
                }
            }
        }
        if max_d <= S::zero() {
            max_d = S::one();
        }

        let mut x_mat = Array2::zeros((n, n));
        for i in 0..n {
            for j in (i + 1)..n {
                let dij = distance_matrix.get_by_index(i, j);
                let dji = distance_matrix.get_by_index(j, i);
                let d_sym = (dij + dji) / S::from_f32(2.0).unwrap();
                let d_norm = (d_sym / max_d).max(S::zero());
                x_mat[[i, j]] = d_norm;
                x_mat[[j, i]] = d_norm;
            }
        }

        // Step 2: Binary search on node standard deviations sigma_i matching thesne.py: find_sigma
        let perplexity = self
            .perplexity
            .min(S::from_usize(n - 1).unwrap() - S::from_f32(0.01).unwrap())
            .max(S::one());
        let entropy_target = perplexity.ln();
        let eps = S::from_f32(1e-16).unwrap();
        let sqrt_eps = S::from_f32(1e-8).unwrap();

        let mut p_conditional = Array2::zeros((n, n));
        for i in 0..n {
            let mut sigma = S::one();
            let mut sig_min = sqrt_eps;
            let mut sig_max = S::infinity();

            for _ in 0..self.sigma_iters {
                let two_sigma_sq = S::from_f32(2.0).unwrap() * sigma * sigma;
                let mut max_exp = S::neg_infinity();
                let mut exps = vec![S::zero(); n];

                for j in 0..n {
                    if i != j {
                        let d = x_mat[[i, j]];
                        let dp = d * d;
                        let val = -dp / two_sigma_sq;
                        exps[j] = val;
                        if val > max_exp {
                            max_exp = val;
                        }
                    }
                }

                let mut sum_w = S::zero();
                let mut weights = vec![S::zero(); n];
                for j in 0..n {
                    if i != j {
                        let w = (exps[j] - max_exp).exp();
                        weights[j] = w;
                        sum_w += w;
                    }
                }

                sum_w = sum_w.max(eps);

                // Compute conditional probabilities and Shannon entropy
                let mut entropy = S::zero();
                for j in 0..n {
                    if i != j {
                        let prob = (weights[j] / sum_w).max(eps);
                        entropy -= prob * prob.ln();
                    }
                }

                if entropy < entropy_target {
                    sig_min = sigma;
                } else {
                    sig_max = sigma;
                }

                if sig_max.is_infinite() {
                    sigma *= S::from_f32(2.0).unwrap();
                } else {
                    sigma = (sig_min + sig_max) / S::from_f32(2.0).unwrap();
                }
            }

            // Compute final conditional probabilities for node i with the converged sigma
            let two_sigma_sq = S::from_f32(2.0).unwrap() * sigma * sigma;
            let mut max_exp = S::neg_infinity();
            let mut exps = vec![S::zero(); n];
            for j in 0..n {
                if i != j {
                    let d = x_mat[[i, j]];
                    let dp = d * d;
                    let val = -dp / two_sigma_sq;
                    exps[j] = val;
                    if val > max_exp {
                        max_exp = val;
                    }
                }
            }

            let mut sum_w = S::zero();
            let mut weights = vec![S::zero(); n];
            for j in 0..n {
                if i != j {
                    let w = (exps[j] - max_exp).exp();
                    weights[j] = w;
                    sum_w += w;
                }
            }
            sum_w = sum_w.max(eps);
            for j in 0..n {
                if i != j {
                    p_conditional[[i, j]] = weights[j] / sum_w;
                }
            }
        }

        // Symmetrized joint probabilities P_ij = (p_{j|i} + p_{i|j}) / (2N)
        let two_n_s = S::from_usize(2 * n).unwrap();
        let mut p_joint = Array2::zeros((n, n));
        for i in 0..n {
            for j in 0..n {
                if i != j {
                    p_joint[[i, j]] = (p_conditional[[i, j]] + p_conditional[[j, i]]) / two_n_s;
                }
            }
        }

        let mut velocity = Array2::zeros((n, 2));

        // Step 3: Gradient descent optimization loop matching thesne.py
        let default_switch = self.n_epochs / 2;
        let lr_switch = self.lr_switch.unwrap_or(default_switch);
        let momentum_switch = self.momentum_switch.unwrap_or(default_switch);
        let l_kl_switch = self.l_kl_switch.unwrap_or(default_switch);
        let l_c_switch = self.l_c_switch.unwrap_or(default_switch);
        let l_r_switch = self.l_r_switch.unwrap_or(default_switch);

        let n_s = S::from_usize(n).unwrap();
        let n_sq_s = S::from_usize(n * n).unwrap();
        let four_s = S::from_f32(4.0).unwrap();
        let r_eps = self.r_eps;

        let mut stepsize_history = VecDeque::with_capacity(self.window_size);

        for epoch in 0..self.n_epochs {
            let lr = if epoch < lr_switch {
                self.initial_lr
            } else {
                self.final_lr
            };
            let momentum = if epoch < momentum_switch {
                self.initial_momentum
            } else {
                self.final_momentum
            };
            let l_kl = if epoch < l_kl_switch {
                self.initial_l_kl
            } else {
                self.final_l_kl
            };
            let l_c = if epoch < l_c_switch {
                self.initial_l_c
            } else {
                self.final_l_c
            };
            let l_r = if epoch < l_r_switch {
                self.initial_l_r
            } else {
                self.final_l_r
            };

            let w_kl = l_kl.unwrap_or(S::zero());
            let w_c = l_c.unwrap_or(S::zero());
            let w_r = l_r.unwrap_or(S::zero());
            let l_sum = (w_kl + w_c + w_r).max(S::from_f32(1e-12).unwrap());

            // Compute Cauchy-distributed low-dimensional similarities Q
            let mut w_out = Array2::zeros((n, n));
            let mut sum_w = S::zero();

            for i in 0..n {
                let yi_0 = drawing.raw_entry(i).0;
                let yi_1 = drawing.raw_entry(i).1;
                for j in (i + 1)..n {
                    let yj_0 = drawing.raw_entry(j).0;
                    let yj_1 = drawing.raw_entry(j).1;
                    let dx = yi_0 - yj_0;
                    let dy = yi_1 - yj_1;
                    let dist2 = dx * dx + dy * dy;
                    let w = S::one() / (S::one() + dist2);
                    w_out[[i, j]] = w;
                    w_out[[j, i]] = w;
                    sum_w += w + w;
                }
            }

            sum_w = sum_w.max(eps);

            let mut q = Array2::zeros((n, n));
            for i in 0..n {
                for j in 0..n {
                    if i != j {
                        q[[i, j]] = w_out[[i, j]] / sum_w;
                    }
                }
            }

            // Compute analytical gradients:
            // 1. KL Divergence term: 4 * sum_j (P_ij - Q_ij) * w_out_ij * (y_i - y_j)
            // 2. Compression term: (1 / N) * y_i
            // 3. Repulsion term: - (1 / N^2) * sum_j (y_i - y_j) / (||y_i - y_j|| * (||y_i - y_j|| + r_eps))
            let mut gradients = Array2::zeros((n, 2));

            for i in 0..n {
                let yi_0 = drawing.raw_entry(i).0;
                let yi_1 = drawing.raw_entry(i).1;

                let mut grad_kl_0 = S::zero();
                let mut grad_kl_1 = S::zero();
                if w_kl > S::zero() {
                    for j in 0..n {
                        if i != j {
                            let yj_0 = drawing.raw_entry(j).0;
                            let yj_1 = drawing.raw_entry(j).1;
                            let dx = yi_0 - yj_0;
                            let dy = yi_1 - yj_1;
                            let factor = four_s * (p_joint[[i, j]] - q[[i, j]]) * w_out[[i, j]];
                            grad_kl_0 += factor * dx;
                            grad_kl_1 += factor * dy;
                        }
                    }
                }

                let (grad_c_0, grad_c_1) = if w_c > S::zero() {
                    (yi_0 / n_s, yi_1 / n_s)
                } else {
                    (S::zero(), S::zero())
                };

                let mut grad_r_0 = S::zero();
                let mut grad_r_1 = S::zero();
                if w_r > S::zero() {
                    for j in 0..n {
                        if i != j {
                            let yj_0 = drawing.raw_entry(j).0;
                            let yj_1 = drawing.raw_entry(j).1;
                            let dx = yi_0 - yj_0;
                            let dy = yi_1 - yj_1;
                            let dist = (dx * dx + dy * dy).sqrt();
                            if dist > S::from_f32(1e-12).unwrap() {
                                let factor = -S::one() / (n_sq_s * dist * (dist + r_eps));
                                grad_r_0 += factor * dx;
                                grad_r_1 += factor * dy;
                            }
                        }
                    }
                }

                // Total normalized gradient
                let g0 = (w_kl * grad_kl_0 + w_c * grad_c_0 + w_r * grad_r_0) / l_sum;
                let g1 = (w_kl * grad_kl_1 + w_c * grad_c_1 + w_r * grad_r_1) / l_sum;

                gradients[[i, 0]] = if g0.is_finite() { g0 } else { S::zero() };
                gradients[[i, 1]] = if g1.is_finite() { g1 } else { S::zero() };
            }

            // Update positions and velocities with momentum
            let mut sum_v_norm = S::zero();
            for i in 0..n {
                velocity[[i, 0]] = momentum * velocity[[i, 0]] - lr * gradients[[i, 0]];
                velocity[[i, 1]] = momentum * velocity[[i, 1]] - lr * gradients[[i, 1]];

                let vx = velocity[[i, 0]];
                let vy = velocity[[i, 1]];
                sum_v_norm += (vx * vx + vy * vy).sqrt();

                let new_x = drawing.raw_entry(i).0 + vx;
                let new_y = drawing.raw_entry(i).1 + vy;

                if new_x.is_finite() {
                    drawing.raw_entry_mut(i).0 = new_x;
                }
                if new_y.is_finite() {
                    drawing.raw_entry_mut(i).1 = new_y;
                }
            }

            // Normalized stepsize computation and sliding window convergence check matching thesne.py
            if self.autostop > S::zero() {
                let mut min_x = S::infinity();
                let mut max_x = S::neg_infinity();
                let mut min_y = S::infinity();
                let mut max_y = S::neg_infinity();
                for i in 0..n {
                    let x = drawing.raw_entry(i).0;
                    let y = drawing.raw_entry(i).1;
                    if x < min_x {
                        min_x = x;
                    }
                    if x > max_x {
                        max_x = x;
                    }
                    if y < min_y {
                        min_y = y;
                    }
                    if y > max_y {
                        max_y = y;
                    }
                }
                let range_x = (max_x - min_x).max(eps);
                let range_y = (max_y - min_y).max(eps);
                let range_max = range_x.max(range_y);

                let stepsize = sum_v_norm / (n_s * lr * range_max);

                if stepsize_history.len() >= self.window_size {
                    stepsize_history.pop_front();
                }
                stepsize_history.push_back(stepsize);

                if epoch >= self.window_size && stepsize_history.len() == self.window_size {
                    let mut max_recent_step = S::zero();
                    for &s in &stepsize_history {
                        if s > max_recent_step {
                            max_recent_step = s;
                        }
                    }
                    if max_recent_step < self.autostop {
                        // Layout has converged early
                        break;
                    }
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
