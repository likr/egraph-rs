use egraph_core::{CoreError, CoreResult};

/// Hyperparameters for [`super::TwinDeclutterer`] and [`super::declutter_twins_2d`].
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
pub struct TwinDeclutterConfig {
    /// Relative spatial-hash merge tolerance as a fraction of layout bounding-box diagonal (`default: 1e-6`).
    pub eps_rel: f64,
    /// Fraction of nearest external cluster clearance `d_clear` used as spiral radius budget (`default: 0.35`).
    pub r_frac: f64,
    /// Hard upper cap on cluster spiral radius relative to bounding-box diagonal (`default: 3e-3`).
    pub r_cap_rel: f64,
    /// Minimum pairwise separation target relative to diagonal (`default: 5e-7`, strictly preserving `f32` precision).
    pub min_pair_rel: f64,
    /// Maximum decluttering passes (`default: 4`).
    pub iterations: usize,
}

impl Default for TwinDeclutterConfig {
    fn default() -> Self {
        Self {
            eps_rel: 1e-6,
            r_frac: 0.35,
            r_cap_rel: 3e-3,
            min_pair_rel: 5e-7,
            iterations: 4,
        }
    }
}

impl TwinDeclutterConfig {
    /// Creates a new [`TwinDeclutterConfig`] initialized with default values.
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the relative spatial-hash merge tolerance `eps_rel`.
    pub fn with_eps_rel(mut self, eps_rel: f64) -> Self {
        self.eps_rel = eps_rel;
        self
    }

    /// Sets the fraction `r_frac` of nearest external cluster clearance used as spiral radius budget.
    pub fn with_r_frac(mut self, r_frac: f64) -> Self {
        self.r_frac = r_frac;
        self
    }

    /// Sets the hard upper cap `r_cap_rel` on cluster spiral radius relative to diagonal.
    pub fn with_r_cap_rel(mut self, r_cap_rel: f64) -> Self {
        self.r_cap_rel = r_cap_rel;
        self
    }

    /// Sets the minimum pairwise separation target `min_pair_rel` relative to diagonal.
    pub fn with_min_pair_rel(mut self, min_pair_rel: f64) -> Self {
        self.min_pair_rel = min_pair_rel;
        self
    }

    /// Sets the maximum number of decluttering passes.
    pub fn with_iterations(mut self, iterations: usize) -> Self {
        self.iterations = iterations;
        self
    }

    /// Validates all twin decluttering configuration parameters.
    pub fn validate(&self) -> CoreResult<()> {
        let checks = [
            (self.eps_rel, "TwinDeclutterConfig::eps_rel"),
            (self.r_frac, "TwinDeclutterConfig::r_frac"),
            (self.r_cap_rel, "TwinDeclutterConfig::r_cap_rel"),
            (self.min_pair_rel, "TwinDeclutterConfig::min_pair_rel"),
        ];
        for (val, name) in checks {
            if !val.is_finite() {
                return Err(CoreError::non_finite(name));
            }
            if val < 0.0 {
                return Err(CoreError::invalid_input(format!(
                    "{name} must be >= 0.0, got {val}"
                )));
            }
        }
        Ok(())
    }
}
