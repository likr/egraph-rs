use egraph_core::{CoreError, CoreResult};

/// Hyperparameters for [`ForceAtlas2Solver`](super::ForceAtlas2Solver).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ForceAtlas2Config {
    /// Total number of ForceAtlas2 iterations (`default: 200`).
    pub iterations: usize,
    /// Barnes-Hut quadtree opening angle threshold $\theta \ge 0$ (`default: 1.2`).
    pub theta: f64,
    /// Whether to use $O(N \log N)$ Barnes-Hut quadtree repulsion (`default: true`).
    pub use_barnes_hut: bool,
    /// Whether to use Jacomy et al. (2014) ForceAtlas2 logarithmic attraction
    /// $F_{\mathrm{att}}(d) = w \frac{\ln(1 + d)}{d} (p_v - p_u)$ (`default: false`).
    ///
    /// Note: This ForceAtlas2 logarithmic mode decays linearly to zero as $d \to 0$ and is
    /// distinct from Noack (2007) true LinLog (`LinLogSolver`), whose constant unit attraction
    /// $w_{uv} \frac{p_v - p_u}{\|p_v - p_u\|}$ does not vanish as $d \to 0$.
    pub linlog_mode: bool,
    /// Whether to use strong gravity proportional to distance from origin (`default: false`).
    pub strong_gravity: bool,
    /// Gravitational pull coefficient toward the origin (`default: 1.0`).
    pub gravity: f64,
    /// Repulsion scaling ratio $k_r$ (`default: 2.0`).
    pub scaling_ratio: f64,
    /// Exponent applied to edge weights (`default: 1.0`).
    pub edge_weight_influence: f64,
    /// Dissuade hubs by distributing attraction by node mass (`default: false`).
    pub outbound_attraction_distribution: bool,
    /// Adaptive speed tolerance controlling swing dampening (`default: 1.0`).
    pub jitter_tolerance: f64,
    /// Optional per-step vertex displacement cap (`None` adapts to `50.0 * max(1.0, sqrt(N / 1000.0))`, `default: None`).
    pub max_displacement: Option<f64>,
    /// Enable Jacomy (2014) border-to-border node overlap prevention (`default: false`).
    pub prevent_overlap: bool,
    /// Default node radius used when constructed without explicit `node_radii` (`default: 1.0`).
    pub default_node_radius: f64,
    /// Deterministic explicit RNG seed for collapsed-layout symmetry breaking (`default: 42`).
    pub seed: u64,
}

impl Default for ForceAtlas2Config {
    fn default() -> Self {
        Self {
            iterations: 200,
            theta: 1.2,
            use_barnes_hut: true,
            linlog_mode: false,
            strong_gravity: false,
            gravity: 1.0,
            scaling_ratio: 2.0,
            edge_weight_influence: 1.0,
            outbound_attraction_distribution: false,
            jitter_tolerance: 1.0,
            max_displacement: None,
            prevent_overlap: false,
            default_node_radius: 1.0,
            seed: 42,
        }
    }
}

impl ForceAtlas2Config {
    /// Validates all ForceAtlas2 hyperparameters.
    pub fn validate(&self) -> CoreResult<()> {
        if !self.theta.is_finite() || self.theta < 0.0 {
            return Err(CoreError::invalid_input(format!(
                "theta must be finite and >= 0.0, got {}",
                self.theta
            )));
        }
        if !self.gravity.is_finite() || self.gravity < 0.0 {
            return Err(CoreError::invalid_input(format!(
                "gravity must be finite and >= 0.0, got {}",
                self.gravity
            )));
        }
        if !self.scaling_ratio.is_finite() || self.scaling_ratio < 0.0 {
            return Err(CoreError::invalid_input(format!(
                "scaling_ratio must be finite and >= 0.0, got {}",
                self.scaling_ratio
            )));
        }
        if !self.edge_weight_influence.is_finite() || self.edge_weight_influence < 0.0 {
            return Err(CoreError::invalid_input(format!(
                "edge_weight_influence must be finite and >= 0.0, got {}",
                self.edge_weight_influence
            )));
        }
        if !self.jitter_tolerance.is_finite() || self.jitter_tolerance <= 0.0 {
            return Err(CoreError::invalid_input(format!(
                "jitter_tolerance must be finite and > 0.0, got {}",
                self.jitter_tolerance
            )));
        }
        if let Some(max_disp) = self.max_displacement
            && (!max_disp.is_finite() || max_disp <= 0.0)
        {
            return Err(CoreError::invalid_input(format!(
                "max_displacement must be finite and > 0.0, got {max_disp}"
            )));
        }
        if !self.default_node_radius.is_finite() || self.default_node_radius < 0.0 {
            return Err(CoreError::invalid_input(format!(
                "default_node_radius must be finite and >= 0.0, got {}",
                self.default_node_radius
            )));
        }
        Ok(())
    }
}

/// Hyperparameters for [`YifanHuSolver`](super::YifanHuSolver).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct YifanHuConfig {
    /// Maximum iterations executed at each multilevel scale (`default: 200`).
    pub max_iterations_per_level: usize,
    /// Initial step size multiplier relative to level optimal distance (`default: 1.0`).
    pub step_size: f64,
    /// Adaptive step cooling/expansion ratio in `(0.0, 1.0)` (`default: 0.9`).
    pub step_ratio: f64,
    /// Repulsion strength coefficient $C$ in $C K^2 / d$ (`default: 0.2`).
    pub relative_strength: f64,
    /// Optional explicit finest-level optimal edge length $K_0$ (`None` defaults to `1.0`).
    pub optimal_distance: Option<f64>,
    /// Barnes-Hut quadtree opening angle threshold $\theta \ge 0$ (`default: 1.2`).
    pub theta: f64,
    /// Stop coarsening once vertex count drops to or below this size (`default: 30`).
    pub min_coarsen_size: usize,
    /// Stop coarsening if `coarse_nodes / fine_nodes >= min_coarsen_ratio` (`default: 0.75`).
    pub min_coarsen_ratio: f64,
    /// Convergence displacement tolerance relative to $K_{\text{level}}$ (`default: 1e-4`).
    pub convergence_tolerance: f64,
    /// Deterministic explicit RNG seed for prolongation jitter (`default: 42`).
    pub seed: u64,
}

impl Default for YifanHuConfig {
    fn default() -> Self {
        Self {
            max_iterations_per_level: 200,
            step_size: 1.0,
            step_ratio: 0.9,
            relative_strength: 0.2,
            optimal_distance: None,
            theta: 1.2,
            min_coarsen_size: 30,
            min_coarsen_ratio: 0.75,
            convergence_tolerance: 1e-4,
            seed: 42,
        }
    }
}

impl YifanHuConfig {
    /// Validates all Yifan Hu hyperparameters.
    pub fn validate(&self) -> CoreResult<()> {
        if !self.step_size.is_finite() || self.step_size <= 0.0 {
            return Err(CoreError::invalid_input(format!(
                "step_size must be finite and > 0.0, got {}",
                self.step_size
            )));
        }
        if !self.step_ratio.is_finite() || self.step_ratio <= 0.0 || self.step_ratio >= 1.0 {
            return Err(CoreError::invalid_input(format!(
                "step_ratio must be finite and in (0.0, 1.0), got {}",
                self.step_ratio
            )));
        }
        if !self.relative_strength.is_finite() || self.relative_strength < 0.0 {
            return Err(CoreError::invalid_input(format!(
                "relative_strength must be finite and >= 0.0, got {}",
                self.relative_strength
            )));
        }
        if let Some(k) = self.optimal_distance
            && (!k.is_finite() || k <= 0.0)
        {
            return Err(CoreError::invalid_input(format!(
                "optimal_distance must be finite and > 0.0, got {k}"
            )));
        }
        if !self.theta.is_finite() || self.theta < 0.0 {
            return Err(CoreError::invalid_input(format!(
                "theta must be finite and >= 0.0, got {}",
                self.theta
            )));
        }
        if self.min_coarsen_size == 0 {
            return Err(CoreError::invalid_input("min_coarsen_size must be >= 1"));
        }
        if !self.min_coarsen_ratio.is_finite()
            || self.min_coarsen_ratio <= 0.0
            || self.min_coarsen_ratio > 1.0
        {
            return Err(CoreError::invalid_input(format!(
                "min_coarsen_ratio must be finite and in (0.0, 1.0], got {}",
                self.min_coarsen_ratio
            )));
        }
        if !self.convergence_tolerance.is_finite() || self.convergence_tolerance < 0.0 {
            return Err(CoreError::invalid_input(format!(
                "convergence_tolerance must be finite and >= 0.0, got {}",
                self.convergence_tolerance
            )));
        }
        Ok(())
    }
}
