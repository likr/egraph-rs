use crate::{curve_fit::find_ab_params, GuMap};
use num_traits::Float;
use petgraph_drawing::DrawingValue;

/// Builder for constructing a `GuMap` (Graph UMAP) layout instance with customized hyperparameters.
#[derive(Debug, Clone)]
pub struct GuMapBuilder<S> {
    pub k: usize,
    pub min_dist: S,
    pub spread: S,
    pub n_epochs: usize,
    pub negative_sample_rate: usize,
    pub learning_rate: S,
    pub a: Option<S>,
    pub b: Option<S>,
    pub sigma_iters: usize,
    pub sigma_tolerance: S,
}

impl<S> GuMapBuilder<S>
where
    S: DrawingValue + Float + std::iter::Sum + std::ops::AddAssign + Default,
{
    /// Creates a new `GuMapBuilder` initialized with standard UMAP / GUMAP defaults.
    pub fn new() -> Self {
        Self {
            k: 15,
            min_dist: S::from_f32(0.1).unwrap(),
            spread: S::from_f32(1.0).unwrap(),
            n_epochs: 300,
            negative_sample_rate: 5,
            learning_rate: S::from_f32(1.0).unwrap(),
            a: None,
            b: None,
            sigma_iters: 64,
            sigma_tolerance: S::from_f32(1e-5).unwrap(),
        }
    }

    /// Sets the number of nearest neighbors `k` (default: 15).
    pub fn k(mut self, k: usize) -> Self {
        self.k = k;
        self
    }

    /// Sets the minimum distance in low-dimensional space (default: 0.1).
    pub fn min_dist(mut self, min_dist: S) -> Self {
        self.min_dist = min_dist;
        self
    }

    /// Sets the scale of spread in low-dimensional space (default: 1.0).
    pub fn spread(mut self, spread: S) -> Self {
        self.spread = spread;
        self
    }

    /// Sets the number of SGD optimization epochs (default: 300).
    pub fn n_epochs(mut self, n_epochs: usize) -> Self {
        self.n_epochs = n_epochs;
        self
    }

    /// Sets the number of negative samples per attractive edge (default: 5).
    pub fn negative_sample_rate(mut self, negative_sample_rate: usize) -> Self {
        self.negative_sample_rate = negative_sample_rate;
        self
    }

    /// Sets the initial learning rate (default: 1.0).
    pub fn learning_rate(mut self, learning_rate: S) -> Self {
        self.learning_rate = learning_rate;
        self
    }

    /// Explicitly sets the `a` parameter for low-dimensional probability $1 / (1 + a d^{2b})$.
    pub fn a(mut self, a: S) -> Self {
        self.a = Some(a);
        self
    }

    /// Explicitly sets the `b` parameter for low-dimensional probability $1 / (1 + a d^{2b})$.
    pub fn b(mut self, b: S) -> Self {
        self.b = Some(b);
        self
    }

    /// Sets the maximum number of binary search iterations for sigma_i (default: 64).
    pub fn sigma_iters(mut self, sigma_iters: usize) -> Self {
        self.sigma_iters = sigma_iters;
        self
    }

    /// Sets the tolerance threshold for perplexity binary search convergence (default: 1e-5).
    pub fn sigma_tolerance(mut self, sigma_tolerance: S) -> Self {
        self.sigma_tolerance = sigma_tolerance;
        self
    }

    /// Validates the hyperparameters and builds a new `GuMap` instance.
    pub fn build(self) -> Result<GuMap<S>, String> {
        if self.k == 0 {
            return Err("k must be greater than 0".to_string());
        }
        if self.min_dist <= S::zero() {
            return Err("min_dist must be positive".to_string());
        }
        if self.spread <= S::zero() {
            return Err("spread must be positive".to_string());
        }
        if self.n_epochs == 0 {
            return Err("n_epochs must be greater than 0".to_string());
        }
        if self.negative_sample_rate == 0 {
            return Err("negative_sample_rate must be greater than 0".to_string());
        }
        if self.learning_rate <= S::zero() {
            return Err("learning_rate must be positive".to_string());
        }
        if self.sigma_iters == 0 {
            return Err("sigma_iters must be greater than 0".to_string());
        }
        if self.sigma_tolerance <= S::zero() {
            return Err("sigma_tolerance must be positive".to_string());
        }

        let (fitted_a, fitted_b) = if self.a.is_none() || self.b.is_none() {
            find_ab_params(self.min_dist, self.spread)
        } else {
            (S::zero(), S::zero())
        };

        let effective_a = self.a.unwrap_or(fitted_a);
        let effective_b = self.b.unwrap_or(fitted_b);

        if effective_a <= S::zero() {
            return Err("a must be positive".to_string());
        }
        if effective_b <= S::zero() {
            return Err("b must be positive".to_string());
        }

        Ok(GuMap {
            k: self.k,
            min_dist: self.min_dist,
            spread: self.spread,
            n_epochs: self.n_epochs,
            negative_sample_rate: self.negative_sample_rate,
            learning_rate: self.learning_rate,
            a: effective_a,
            b: effective_b,
            sigma_iters: self.sigma_iters,
            sigma_tolerance: self.sigma_tolerance,
        })
    }
}

impl<S> Default for GuMapBuilder<S>
where
    S: DrawingValue + Float + std::iter::Sum + std::ops::AddAssign + Default,
{
    fn default() -> Self {
        Self::new()
    }
}
