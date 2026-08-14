use crate::TsNet;
use num_traits::Float;
use petgraph_drawing::DrawingValue;

/// Builder for constructing a `TsNet` graph layout instance with customized hyperparameters.
///
/// `TsNetBuilder` implements the Builder pattern with explicit validation to ensure
/// all numerical parameters lie within their valid theoretical ranges.
#[derive(Debug, Clone)]
pub struct TsNetBuilder<S> {
    pub perplexity: S,
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

impl<S> TsNetBuilder<S>
where
    S: DrawingValue
        + Float
        + std::iter::Sum
        + std::ops::AddAssign
        + Default
        + ndarray::ScalarOperand,
{
    /// Creates a new `TsNetBuilder` initialized with default hyperparameters.
    pub fn new() -> Self {
        Self {
            perplexity: S::from_f32(30.0).unwrap(),
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

    /// Sets the target perplexity (effective number of neighbors).
    pub fn perplexity(mut self, perplexity: S) -> Self {
        self.perplexity = perplexity;
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

    /// Sets the compression penalty coefficient lambda_c for Stage 2.
    pub fn lambda_c_stage2(mut self, lambda_c: S) -> Self {
        self.lambda_c_stage2 = lambda_c;
        self
    }

    /// Sets the number of iterations for Stage 3 (final adjustment / refinement).
    pub fn iterations_stage3(mut self, iterations: usize) -> Self {
        self.iterations_stage3 = iterations;
        self
    }

    /// Sets the compression penalty coefficient lambda_c for Stage 3.
    pub fn lambda_c_stage3(mut self, lambda_c: S) -> Self {
        self.lambda_c_stage3 = lambda_c;
        self
    }

    /// Sets the repulsion penalty coefficient lambda_r for Stage 3.
    pub fn lambda_r_stage3(mut self, lambda_r: S) -> Self {
        self.lambda_r_stage3 = lambda_r;
        self
    }

    /// Sets the learning rate for momentum-based gradient descent.
    pub fn learning_rate(mut self, learning_rate: S) -> Self {
        self.learning_rate = learning_rate;
        self
    }

    /// Sets the momentum parameter (must be in `[0, 1)`).
    pub fn momentum(mut self, momentum: S) -> Self {
        self.momentum = momentum;
        self
    }

    /// Sets the power exponent applied to graph distances (default is 2.0 for standard Gaussian).
    pub fn power(mut self, power: S) -> Self {
        self.power = power;
        self
    }

    /// Sets the repulsion offset epsilon_r to prevent division by zero in repulsion force.
    pub fn epsilon_r(mut self, epsilon_r: S) -> Self {
        self.epsilon_r = epsilon_r;
        self
    }

    /// Sets the maximum number of binary search iterations for finding node standard deviations sigma_i.
    pub fn sigma_iters(mut self, sigma_iters: usize) -> Self {
        self.sigma_iters = sigma_iters;
        self
    }

    /// Sets the tolerance threshold for perplexity binary search convergence.
    pub fn sigma_tolerance(mut self, sigma_tolerance: S) -> Self {
        self.sigma_tolerance = sigma_tolerance;
        self
    }

    /// Validates the hyperparameters and builds a new `TsNet` instance.
    pub fn build(self) -> Result<TsNet<S>, String> {
        if self.perplexity <= S::zero() {
            return Err("perplexity must be positive".to_string());
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

        Ok(TsNet {
            perplexity: self.perplexity,
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

impl<S> Default for TsNetBuilder<S>
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
