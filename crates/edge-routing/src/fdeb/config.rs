use egraph_core::{CoreError, CoreResult};

/// Hyperparameters for [`FdebSolver`](super::FdebSolver).
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
pub struct FdebConfig {
    /// Number of subdivision cycles (`default: 6`).
    pub cycles: usize,
    /// Initial force step size (`default: 0.1`).
    pub s0: f64,
    /// Initial number of iterations in cycle 0 (`default: 90`).
    pub steps_per_cycle: usize,
    /// Multiplicative step-size decay factor per cycle (`default: 0.5`).
    pub s_step: f64,
    /// Multiplicative iteration-count decay factor per cycle (`default: 2.0 / 3.0`).
    pub i_step: f64,
    /// Edge spring constant (`default: 0.1`).
    pub spring_k: f64,
    /// Minimum edge compatibility threshold in `[0.0, 1.0]` (`default: 0.6`).
    pub minimum_edge_compatibility: f64,
}

impl Default for FdebConfig {
    fn default() -> Self {
        Self {
            cycles: 6,
            s0: 0.1,
            steps_per_cycle: 90,
            s_step: 0.5,
            i_step: 2.0 / 3.0,
            spring_k: 0.1,
            minimum_edge_compatibility: 0.6,
        }
    }
}

impl FdebConfig {
    /// Creates a new [`FdebConfig`] initialized to [`Self::default`].
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the number of subdivision cycles.
    pub fn with_cycles(mut self, cycles: usize) -> Self {
        self.cycles = cycles;
        self
    }

    /// Sets the initial force step size `s0`.
    pub fn with_s0(mut self, s0: f64) -> Self {
        self.s0 = s0;
        self
    }

    /// Sets the initial number of iterations in cycle 0.
    pub fn with_steps_per_cycle(mut self, steps_per_cycle: usize) -> Self {
        self.steps_per_cycle = steps_per_cycle;
        self
    }

    /// Sets the multiplicative step-size decay factor per cycle (`s_step`).
    pub fn with_s_step(mut self, s_step: f64) -> Self {
        self.s_step = s_step;
        self
    }

    /// Sets the multiplicative iteration-count decay factor per cycle (`i_step`).
    pub fn with_i_step(mut self, i_step: f64) -> Self {
        self.i_step = i_step;
        self
    }

    /// Sets the edge spring constant (`spring_k`).
    pub fn with_spring_k(mut self, spring_k: f64) -> Self {
        self.spring_k = spring_k;
        self
    }

    /// Sets the minimum edge compatibility threshold in `[0.0, 1.0]`.
    pub fn with_minimum_edge_compatibility(mut self, minimum_edge_compatibility: f64) -> Self {
        self.minimum_edge_compatibility = minimum_edge_compatibility;
        self
    }

    /// Validates all FDEB configuration parameters.
    pub fn validate(&self) -> CoreResult<()> {
        for (name, val) in [
            ("s0", self.s0),
            ("s_step", self.s_step),
            ("i_step", self.i_step),
            ("spring_k", self.spring_k),
            (
                "minimum_edge_compatibility",
                self.minimum_edge_compatibility,
            ),
        ] {
            if !val.is_finite() {
                return Err(CoreError::non_finite("FdebConfig"));
            }
            if val < 0.0 {
                return Err(CoreError::invalid_input(format!(
                    "{name} must be >= 0.0, got {val}"
                )));
            }
        }
        if self.minimum_edge_compatibility > 1.0 {
            return Err(CoreError::invalid_input(format!(
                "minimum_edge_compatibility must be <= 1.0, got {}",
                self.minimum_edge_compatibility
            )));
        }
        if self.cycles >= 30 {
            return Err(CoreError::invalid_input(format!(
                "cycles must be < 30 to avoid subdivision overflow, got {}",
                self.cycles
            )));
        }
        Ok(())
    }
}
