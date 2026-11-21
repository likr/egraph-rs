use crate::TsNet;
use num_traits::Float;
use petgraph_drawing::DrawingValue;

/// Builder for constructing a `TsNet` graph layout instance with customized hyperparameters.
///
/// `TsNetBuilder` implements the Builder pattern with explicit validation to ensure
/// all numerical parameters lie within their valid theoretical ranges according to `thesne.py`.
///
/// # References
///
/// Kruiger, J. F., Rauber, P. E., Martins, R. M., Kerren, A., Kobourov, S., & Telea, A. C. (2017).
/// Graph Layouts by t-SNE. *Computer Graphics Forum*, 36(3), 283-294.
#[derive(Debug, Clone)]
pub struct TsNetBuilder<S> {
    pub perplexity: S,
    pub n_epochs: usize,
    pub initial_lr: S,
    pub final_lr: S,
    pub lr_switch: Option<usize>,
    pub initial_momentum: S,
    pub final_momentum: S,
    pub momentum_switch: Option<usize>,
    pub initial_l_kl: Option<S>,
    pub final_l_kl: Option<S>,
    pub l_kl_switch: Option<usize>,
    pub initial_l_c: Option<S>,
    pub final_l_c: Option<S>,
    pub l_c_switch: Option<usize>,
    pub initial_l_r: Option<S>,
    pub final_l_r: Option<S>,
    pub l_r_switch: Option<usize>,
    pub r_eps: S,
    pub sigma_iters: usize,
    pub autostop: S,
    pub window_size: usize,
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
    /// Creates a new `TsNetBuilder` initialized with default hyperparameters matching `tsnet.py` (tsNET).
    pub fn new() -> Self {
        Self {
            perplexity: S::from_f32(40.0).unwrap(),
            n_epochs: 5000,
            initial_lr: S::from_f32(50.0).unwrap(),
            final_lr: S::from_f32(50.0).unwrap(),
            lr_switch: None,
            initial_momentum: S::from_f32(0.5).unwrap(),
            final_momentum: S::from_f32(0.5).unwrap(),
            momentum_switch: None,
            initial_l_kl: Some(S::from_f32(1.0).unwrap()),
            final_l_kl: Some(S::from_f32(1.0).unwrap()),
            l_kl_switch: None,
            initial_l_c: Some(S::from_f32(1.2).unwrap()),
            final_l_c: Some(S::from_f32(0.01).unwrap()),
            l_c_switch: None,
            initial_l_r: Some(S::from_f32(0.0).unwrap()),
            final_l_r: Some(S::from_f32(0.6).unwrap()),
            l_r_switch: None,
            r_eps: S::from_f32(0.05).unwrap(),
            sigma_iters: 50,
            autostop: S::from_f32(1e-7).unwrap(),
            window_size: 40,
        }
    }

    /// Creates a new `TsNetBuilder` initialized with tsNET* hyperparameters (`initial_l_c = 0.1`).
    pub fn new_star() -> Self {
        Self {
            initial_l_c: Some(S::from_f32(0.1).unwrap()),
            ..Self::new()
        }
    }

    /// Sets the target perplexity (effective number of neighbors).
    pub fn perplexity(mut self, perplexity: S) -> Self {
        self.perplexity = perplexity;
        self
    }

    /// Sets the total number of epochs (iterations).
    pub fn n_epochs(mut self, n_epochs: usize) -> Self {
        self.n_epochs = n_epochs;
        self
    }

    /// Sets the initial learning rate.
    pub fn initial_lr(mut self, initial_lr: S) -> Self {
        self.initial_lr = initial_lr;
        self
    }

    /// Sets the final learning rate.
    pub fn final_lr(mut self, final_lr: S) -> Self {
        self.final_lr = final_lr;
        self
    }

    /// Sets the epoch at which the learning rate switches from `initial_lr` to `final_lr`.
    pub fn lr_switch(mut self, lr_switch: Option<usize>) -> Self {
        self.lr_switch = lr_switch;
        self
    }

    /// Sets both the initial and final learning rate to the same value.
    pub fn learning_rate(mut self, learning_rate: S) -> Self {
        self.initial_lr = learning_rate;
        self.final_lr = learning_rate;
        self
    }

    /// Sets the initial momentum parameter.
    pub fn initial_momentum(mut self, initial_momentum: S) -> Self {
        self.initial_momentum = initial_momentum;
        self
    }

    /// Sets the final momentum parameter.
    pub fn final_momentum(mut self, final_momentum: S) -> Self {
        self.final_momentum = final_momentum;
        self
    }

    /// Sets the epoch at which momentum switches from `initial_momentum` to `final_momentum`.
    pub fn momentum_switch(mut self, momentum_switch: Option<usize>) -> Self {
        self.momentum_switch = momentum_switch;
        self
    }

    /// Sets both the initial and final momentum parameter to the same value.
    pub fn momentum(mut self, momentum: S) -> Self {
        self.initial_momentum = momentum;
        self.final_momentum = momentum;
        self
    }

    /// Sets the initial KL divergence weight.
    pub fn initial_l_kl(mut self, initial_l_kl: Option<S>) -> Self {
        self.initial_l_kl = initial_l_kl;
        self
    }

    /// Sets the final KL divergence weight.
    pub fn final_l_kl(mut self, final_l_kl: Option<S>) -> Self {
        self.final_l_kl = final_l_kl;
        self
    }

    /// Sets the epoch at which lambda_kl switches.
    pub fn l_kl_switch(mut self, l_kl_switch: Option<usize>) -> Self {
        self.l_kl_switch = l_kl_switch;
        self
    }

    /// Sets the initial compression weight.
    pub fn initial_l_c(mut self, initial_l_c: Option<S>) -> Self {
        self.initial_l_c = initial_l_c;
        self
    }

    /// Sets the final compression weight.
    pub fn final_l_c(mut self, final_l_c: Option<S>) -> Self {
        self.final_l_c = final_l_c;
        self
    }

    /// Sets the epoch at which lambda_c switches.
    pub fn l_c_switch(mut self, l_c_switch: Option<usize>) -> Self {
        self.l_c_switch = l_c_switch;
        self
    }

    /// Sets the initial repulsion weight.
    pub fn initial_l_r(mut self, initial_l_r: Option<S>) -> Self {
        self.initial_l_r = initial_l_r;
        self
    }

    /// Sets the final repulsion weight.
    pub fn final_l_r(mut self, final_l_r: Option<S>) -> Self {
        self.final_l_r = final_l_r;
        self
    }

    /// Sets the epoch at which lambda_r switches.
    pub fn l_r_switch(mut self, l_r_switch: Option<usize>) -> Self {
        self.l_r_switch = l_r_switch;
        self
    }

    /// Sets the repulsion offset parameter `r_eps` to prevent division by zero in repulsion force.
    pub fn r_eps(mut self, r_eps: S) -> Self {
        self.r_eps = r_eps;
        self
    }

    /// Sets the maximum number of binary search iterations for finding node standard deviations sigma_i.
    pub fn sigma_iters(mut self, sigma_iters: usize) -> Self {
        self.sigma_iters = sigma_iters;
        self
    }

    /// Sets the autostop convergence tolerance threshold.
    pub fn autostop(mut self, autostop: S) -> Self {
        self.autostop = autostop;
        self
    }

    /// Sets the sliding window size for checking autostop convergence.
    pub fn window_size(mut self, window_size: usize) -> Self {
        self.window_size = window_size;
        self
    }

    /// Validates the hyperparameters and builds a new `TsNet` instance.
    pub fn build(self) -> Result<TsNet<S>, String> {
        if !self.perplexity.is_finite() || self.perplexity <= S::zero() {
            return Err("perplexity must be finite and positive".to_string());
        }
        if self.n_epochs == 0 {
            return Err("n_epochs must be greater than 0".to_string());
        }
        if !self.initial_lr.is_finite() || self.initial_lr <= S::zero() {
            return Err("initial_lr must be finite and positive".to_string());
        }
        if !self.final_lr.is_finite() || self.final_lr <= S::zero() {
            return Err("final_lr must be finite and positive".to_string());
        }
        if !self.initial_momentum.is_finite()
            || self.initial_momentum < S::zero()
            || self.initial_momentum >= S::one()
        {
            return Err("initial_momentum must be finite and in the range [0.0, 1.0)".to_string());
        }
        if !self.final_momentum.is_finite()
            || self.final_momentum < S::zero()
            || self.final_momentum >= S::one()
        {
            return Err("final_momentum must be finite and in the range [0.0, 1.0)".to_string());
        }
        if let Some(l_kl) = self.initial_l_kl {
            if !l_kl.is_finite() || l_kl < S::zero() {
                return Err("initial_l_kl must be finite and non-negative".to_string());
            }
        }
        if let Some(l_kl) = self.final_l_kl {
            if !l_kl.is_finite() || l_kl < S::zero() {
                return Err("final_l_kl must be finite and non-negative".to_string());
            }
        }
        if let Some(l_c) = self.initial_l_c {
            if !l_c.is_finite() || l_c < S::zero() {
                return Err("initial_l_c must be finite and non-negative".to_string());
            }
        }
        if let Some(l_c) = self.final_l_c {
            if !l_c.is_finite() || l_c < S::zero() {
                return Err("final_l_c must be finite and non-negative".to_string());
            }
        }
        if let Some(l_r) = self.initial_l_r {
            if !l_r.is_finite() || l_r < S::zero() {
                return Err("initial_l_r must be finite and non-negative".to_string());
            }
        }
        if let Some(l_r) = self.final_l_r {
            if !l_r.is_finite() || l_r < S::zero() {
                return Err("final_l_r must be finite and non-negative".to_string());
            }
        }
        if !self.r_eps.is_finite() || self.r_eps <= S::zero() {
            return Err("r_eps must be finite and positive".to_string());
        }
        if self.sigma_iters == 0 {
            return Err("sigma_iters must be greater than 0".to_string());
        }
        if !self.autostop.is_finite() || self.autostop < S::zero() {
            return Err("autostop must be finite and non-negative".to_string());
        }
        if self.window_size == 0 {
            return Err("window_size must be greater than 0".to_string());
        }

        Ok(TsNet {
            perplexity: self.perplexity,
            n_epochs: self.n_epochs,
            initial_lr: self.initial_lr,
            final_lr: self.final_lr,
            lr_switch: self.lr_switch,
            initial_momentum: self.initial_momentum,
            final_momentum: self.final_momentum,
            momentum_switch: self.momentum_switch,
            initial_l_kl: self.initial_l_kl,
            final_l_kl: self.final_l_kl,
            l_kl_switch: self.l_kl_switch,
            initial_l_c: self.initial_l_c,
            final_l_c: self.final_l_c,
            l_c_switch: self.l_c_switch,
            initial_l_r: self.initial_l_r,
            final_l_r: self.final_l_r,
            l_r_switch: self.l_r_switch,
            r_eps: self.r_eps,
            sigma_iters: self.sigma_iters,
            autostop: self.autostop,
            window_size: self.window_size,
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
