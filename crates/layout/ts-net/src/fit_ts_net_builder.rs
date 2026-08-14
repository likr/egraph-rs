use crate::FitTsNet;
use num_traits::Float;
use petgraph_drawing::DrawingValue;

/// Builder for constructing a `FitTsNet` layout instance with customized hyperparameters.
#[derive(Debug, Clone)]
pub struct FitTsNetBuilder<S> {
    pub intervals: usize,
    pub interpolation_points: usize,
    pub perplexity: S,
    pub theta: S,
    pub k: Option<usize>,
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

impl<S> FitTsNetBuilder<S>
where
    S: DrawingValue + Float + std::iter::Sum + std::ops::AddAssign + Default,
{
    /// Creates a new `FitTsNetBuilder` initialized with default hyperparameters.
    pub fn new() -> Self {
        Self {
            intervals: 25,
            interpolation_points: 3,
            perplexity: S::from_f32(40.0).unwrap(),
            theta: S::from_f32(0.5).unwrap(),
            k: None,
            iterations_stage1: 250,
            exaggeration: S::from_f32(4.0).unwrap(),
            iterations_stage2: 250,
            lambda_c_stage2: S::from_f32(1.2).unwrap(),
            iterations_stage3: 250,
            lambda_c_stage3: S::from_f32(0.01).unwrap(),
            lambda_r_stage3: S::from_f32(0.6).unwrap(),
            learning_rate: S::from_f32(200.0).unwrap(),
            momentum: S::from_f32(0.8).unwrap(),
            power: S::from_f32(2.0).unwrap(),
            epsilon_r: S::from_f32(0.05).unwrap(),
            sigma_iters: 50,
            sigma_tolerance: S::from_f32(1e-5).unwrap(),
        }
    }

    /// Sets the number of intervals `I` for FFT interpolation (default: 25).
    pub fn intervals(mut self, intervals: usize) -> Self {
        self.intervals = intervals;
        self
    }

    /// Sets the number of interpolation points `P` per interval (default: 3).
    pub fn interpolation_points(mut self, interpolation_points: usize) -> Self {
        self.interpolation_points = interpolation_points;
        self
    }

    /// Sets the target perplexity (default: 40.0).
    pub fn perplexity(mut self, perplexity: S) -> Self {
        self.perplexity = perplexity;
        self
    }

    /// Sets the Barnes-Hut opening angle threshold theta for C2 entropy quadtree (default: 0.5).
    pub fn theta(mut self, theta: S) -> Self {
        self.theta = theta;
        self
    }

    /// Sets the number of nearest neighbors `k` for Partial BFS. Defaults to `3 * perplexity`.
    pub fn k(mut self, k: usize) -> Self {
        self.k = Some(k);
        self
    }

    /// Sets the number of iterations for Stage 1 (early exaggeration).
    pub fn iterations_stage1(mut self, iterations: usize) -> Self {
        self.iterations_stage1 = iterations;
        self
    }

    /// Sets the early exaggeration multiplier for Stage 1.
    pub fn exaggeration(mut self, exaggeration: S) -> Self {
        self.exaggeration = exaggeration;
        self
    }

    /// Sets the number of iterations for Stage 2 (early compression / untangling).
    pub fn iterations_stage2(mut self, iterations: usize) -> Self {
        self.iterations_stage2 = iterations;
        self
    }

    /// Sets the compression penalty parameter lambda_c for Stage 2.
    pub fn lambda_c_stage2(mut self, lambda_c: S) -> Self {
        self.lambda_c_stage2 = lambda_c;
        self
    }

    /// Sets the number of iterations for Stage 3 (final refinement).
    pub fn iterations_stage3(mut self, iterations: usize) -> Self {
        self.iterations_stage3 = iterations;
        self
    }

    /// Sets the compression penalty parameter lambda_c for Stage 3.
    pub fn lambda_c_stage3(mut self, lambda_c: S) -> Self {
        self.lambda_c_stage3 = lambda_c;
        self
    }

    /// Sets the repulsion penalty parameter lambda_r for Stage 3.
    pub fn lambda_r_stage3(mut self, lambda_r: S) -> Self {
        self.lambda_r_stage3 = lambda_r;
        self
    }

    /// Sets the learning rate for gradient descent.
    pub fn learning_rate(mut self, learning_rate: S) -> Self {
        self.learning_rate = learning_rate;
        self
    }

    /// Sets the momentum parameter in `[0, 1)`.
    pub fn momentum(mut self, momentum: S) -> Self {
        self.momentum = momentum;
        self
    }

    /// Sets the distance power exponent.
    pub fn power(mut self, power: S) -> Self {
        self.power = power;
        self
    }

    /// Sets the repulsion offset epsilon_r to avoid division by zero.
    pub fn epsilon_r(mut self, epsilon_r: S) -> Self {
        self.epsilon_r = epsilon_r;
        self
    }

    /// Sets the maximum number of binary search iterations for sigma_i.
    pub fn sigma_iters(mut self, sigma_iters: usize) -> Self {
        self.sigma_iters = sigma_iters;
        self
    }

    /// Sets the tolerance threshold for perplexity binary search convergence.
    pub fn sigma_tolerance(mut self, sigma_tolerance: S) -> Self {
        self.sigma_tolerance = sigma_tolerance;
        self
    }

    /// Validates the hyperparameters and builds a new `FitTsNet` instance.
    pub fn build(self) -> Result<FitTsNet<S>, String> {
        if self.intervals == 0 {
            return Err("intervals must be greater than 0".to_string());
        }
        if self.interpolation_points == 0 {
            return Err("interpolation_points must be greater than 0".to_string());
        }
        if self.perplexity <= S::zero() {
            return Err("perplexity must be positive".to_string());
        }
        if self.theta <= S::zero() {
            return Err("theta must be positive".to_string());
        }
        if let Some(k_val) = self.k {
            if k_val == 0 {
                return Err("k must be greater than 0".to_string());
            }
        }
        if self.exaggeration < S::one() {
            return Err("exaggeration must be at least 1.0".to_string());
        }
        if self.learning_rate <= S::zero() {
            return Err("learning_rate must be positive".to_string());
        }
        if self.momentum < S::zero() || self.momentum >= S::one() {
            return Err("momentum must be in the range [0.0, 1.0)".to_string());
        }
        if self.power <= S::zero() {
            return Err("power must be positive".to_string());
        }
        if self.epsilon_r <= S::zero() {
            return Err("epsilon_r must be positive".to_string());
        }
        if self.lambda_c_stage2 < S::zero() {
            return Err("lambda_c_stage2 must be non-negative".to_string());
        }
        if self.lambda_c_stage3 < S::zero() {
            return Err("lambda_c_stage3 must be non-negative".to_string());
        }
        if self.lambda_r_stage3 < S::zero() {
            return Err("lambda_r_stage3 must be non-negative".to_string());
        }
        if self.sigma_iters == 0 {
            return Err("sigma_iters must be greater than 0".to_string());
        }
        if self.sigma_tolerance <= S::zero() {
            return Err("sigma_tolerance must be positive".to_string());
        }

        let effective_k = match self.k {
            Some(k_val) => k_val,
            None => (self.perplexity * S::from_f32(3.0).unwrap())
                .to_usize()
                .unwrap_or(120)
                .max(1),
        };

        Ok(FitTsNet {
            intervals: self.intervals,
            interpolation_points: self.interpolation_points,
            perplexity: self.perplexity,
            theta: self.theta,
            k: effective_k,
            iterations_stage1: self.iterations_stage1,
            exaggeration: self.exaggeration,
            iterations_stage2: self.iterations_stage2,
            lambda_c_stage2: self.lambda_c_stage2,
            iterations_stage3: self.iterations_stage3,
            lambda_c_stage3: self.lambda_c_stage3,
            lambda_r_stage3: self.lambda_r_stage3,
            learning_rate: self.learning_rate,
            momentum: self.momentum,
            power: self.power,
            epsilon_r: self.epsilon_r,
            sigma_iters: self.sigma_iters,
            sigma_tolerance: self.sigma_tolerance,
        })
    }
}

impl<S> Default for FitTsNetBuilder<S>
where
    S: DrawingValue + Float + std::iter::Sum + std::ops::AddAssign + Default,
{
    fn default() -> Self {
        Self::new()
    }
}
