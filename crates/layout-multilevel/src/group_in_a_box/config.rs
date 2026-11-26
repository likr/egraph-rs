use egraph_core::{CoreError, CoreResult};

/// Box placement strategy for [`GroupInABoxSolver`](super::GroupInABoxSolver).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum GibBoxLayoutMode {
    /// Rodrigues et al. (2011) `TR-GIB`: Bruls et al. (2000) squarified treemap partitioning.
    #[default]
    SquarifiedTreemap,
    /// `FD-GIB`: force-directed inter-cluster box placement followed by strict overlap removal.
    ForceDirected,
}

/// Intra-cluster subgraph layout solver used inside each [`GibBox`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum GibIntraLayoutMode {
    /// Fruchterman–Reingold force-directed layout (`FruchtermanReingoldSolver`).
    #[default]
    Fr,
    /// Full-APSP Stochastic Gradient Descent (`SgdSolver::full`).
    SgdFull,
    /// Multilevel Yifan Hu force-directed layout (`YifanHuSolver`).
    YifanHu,
}

/// Configuration for [`GroupInABoxSolver`](super::GroupInABoxSolver).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GroupInABoxConfig {
    /// Cluster box placement strategy (`default: GibBoxLayoutMode::SquarifiedTreemap`).
    pub box_mode: GibBoxLayoutMode,
    /// Intra-cluster subgraph layout solver (`default: GibIntraLayoutMode::Fr`).
    pub intra_mode: GibIntraLayoutMode,
    /// Canvas width (`> 0.0`, `default: 100.0`).
    pub total_width: f64,
    /// Canvas height (`> 0.0`, `default: 100.0`).
    pub total_height: f64,
    /// Relative inner padding fraction per side in `[0.0, 0.5)` (`default: 0.05`).
    pub box_padding: f64,
    /// Maximum cooperative intra-box layout iterations (`default: 100`).
    pub intra_iterations: usize,
    /// Number of inter-cluster force-directed iterations in `ForceDirected` mode (`default: 100`).
    pub fd_iterations: usize,
    /// Deterministic explicit RNG seed (`default: 42`).
    pub seed: u64,
}

impl Default for GroupInABoxConfig {
    fn default() -> Self {
        Self {
            box_mode: GibBoxLayoutMode::SquarifiedTreemap,
            intra_mode: GibIntraLayoutMode::Fr,
            total_width: 100.0,
            total_height: 100.0,
            box_padding: 0.05,
            intra_iterations: 100,
            fd_iterations: 100,
            seed: 42,
        }
    }
}

impl GroupInABoxConfig {
    /// Validates all Group-in-a-Box hyperparameters.
    pub fn validate(&self) -> CoreResult<()> {
        if !self.total_width.is_finite() {
            return Err(CoreError::non_finite("GroupInABoxConfig::total_width"));
        }
        if self.total_width <= 0.0 {
            return Err(CoreError::invalid_input(format!(
                "total_width must be > 0.0, got {}",
                self.total_width
            )));
        }
        if !self.total_height.is_finite() {
            return Err(CoreError::non_finite("GroupInABoxConfig::total_height"));
        }
        if self.total_height <= 0.0 {
            return Err(CoreError::invalid_input(format!(
                "total_height must be > 0.0, got {}",
                self.total_height
            )));
        }
        if !self.box_padding.is_finite() {
            return Err(CoreError::non_finite("GroupInABoxConfig::box_padding"));
        }
        if !(0.0..0.5).contains(&self.box_padding) {
            return Err(CoreError::invalid_input(format!(
                "box_padding must be in [0.0, 0.5), got {}",
                self.box_padding
            )));
        }
        Ok(())
    }
}

/// Axis-aligned cluster bounding box produced by [`GroupInABoxSolver`](super::GroupInABoxSolver).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GibBox {
    /// Original cluster label assigned to this box.
    pub cluster_id: u32,
    /// Minimum x-coordinate of the box.
    pub x_min: f64,
    /// Maximum x-coordinate of the box.
    pub x_max: f64,
    /// Minimum y-coordinate of the box.
    pub y_min: f64,
    /// Maximum y-coordinate of the box.
    pub y_max: f64,
    /// Number of vertices in this cluster.
    pub node_count: usize,
    /// Positive weight used to size this box's area.
    pub weight: f64,
}
