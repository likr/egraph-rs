use egraph_core::{CoreError, CoreResult};

/// Hyperparameters for [`ForceAtlas2Solver`](super::ForceAtlas2Solver).
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
pub struct ForceAtlas2Config {
    /// Total number of ForceAtlas2 iterations (`default: 200`).
    pub iterations: usize,
    /// Barnes-Hut spatial tree opening angle threshold $\theta \ge 0$ (`default: 1.2`).
    pub theta: f64,
    /// Whether to use $O(N \log N)$ Barnes-Hut spatial tree repulsion (`default: true`).
    ///
    /// Supported on [`Layout2d`](egraph_core::Layout2d) and [`Layout3d`](egraph_core::Layout3d);
    /// must be `false` on [`LayoutNd`](egraph_core::LayoutNd).
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
    /// Creates a new [`ForceAtlas2Config`] initialized with [`Self::default`].
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the total number of iterations.
    pub fn with_iterations(mut self, iterations: usize) -> Self {
        self.iterations = iterations;
        self
    }

    /// Sets the Barnes-Hut opening angle threshold `theta`.
    pub fn with_theta(mut self, theta: f64) -> Self {
        self.theta = theta;
        self
    }

    /// Enables or disables Barnes-Hut spatial tree repulsion.
    pub fn with_use_barnes_hut(mut self, use_barnes_hut: bool) -> Self {
        self.use_barnes_hut = use_barnes_hut;
        self
    }

    /// Enables or disables ForceAtlas2 logarithmic attraction mode.
    pub fn with_linlog_mode(mut self, linlog_mode: bool) -> Self {
        self.linlog_mode = linlog_mode;
        self
    }

    /// Enables or disables strong gravity mode.
    pub fn with_strong_gravity(mut self, strong_gravity: bool) -> Self {
        self.strong_gravity = strong_gravity;
        self
    }

    /// Sets the gravitational pull coefficient toward the origin.
    pub fn with_gravity(mut self, gravity: f64) -> Self {
        self.gravity = gravity;
        self
    }

    /// Sets the repulsion scaling ratio `scaling_ratio`.
    pub fn with_scaling_ratio(mut self, scaling_ratio: f64) -> Self {
        self.scaling_ratio = scaling_ratio;
        self
    }

    /// Sets the exponent applied to edge weights.
    pub fn with_edge_weight_influence(mut self, edge_weight_influence: f64) -> Self {
        self.edge_weight_influence = edge_weight_influence;
        self
    }

    /// Enables or disables outbound attraction distribution (hub dissuasion).
    pub fn with_outbound_attraction_distribution(
        mut self,
        outbound_attraction_distribution: bool,
    ) -> Self {
        self.outbound_attraction_distribution = outbound_attraction_distribution;
        self
    }

    /// Sets the adaptive swing jitter tolerance.
    pub fn with_jitter_tolerance(mut self, jitter_tolerance: f64) -> Self {
        self.jitter_tolerance = jitter_tolerance;
        self
    }

    /// Sets an optional per-step vertex displacement cap.
    pub fn with_max_displacement(mut self, max_displacement: Option<f64>) -> Self {
        self.max_displacement = max_displacement;
        self
    }

    /// Enables or disables border-to-border node overlap prevention.
    pub fn with_prevent_overlap(mut self, prevent_overlap: bool) -> Self {
        self.prevent_overlap = prevent_overlap;
        self
    }

    /// Sets the default node radius used when constructed without explicit `node_radii`.
    pub fn with_default_node_radius(mut self, default_node_radius: f64) -> Self {
        self.default_node_radius = default_node_radius;
        self
    }

    /// Sets the deterministic explicit RNG seed.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

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
