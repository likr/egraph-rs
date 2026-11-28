use egraph_core::{CoreError, CoreResult};
use egraph_layout_force::{ForceBackend, FrBackend};

/// Box placement strategy for [`GroupInABoxSolver`](super::GroupInABoxSolver).
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum GibBoxLayoutMode {
    /// Rodrigues et al. (2011) `TR-GIB`: Bruls et al. (2000) squarified treemap partitioning.
    #[default]
    SquarifiedTreemap,
    /// `FD-GIB`: force-directed inter-cluster box placement followed by strict overlap removal.
    ForceDirected,
}

/// Intra-cluster subgraph layout solver used inside each [`GibBox`].
#[non_exhaustive]
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
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
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
    /// When `true`, uniformly rescale `ForceDirected` boxes into `[0, total_width] x [0, total_height]` (`default: true`).
    pub fit_bounds: bool,
    /// Deterministic explicit RNG seed (`default: 42`).
    pub seed: u64,
    /// Repulsion backend used by `FruchtermanReingoldSolver` (`default: FrBackend::Exact`).
    pub fr_backend: FrBackend,
    /// Repulsion backend used by `YifanHuSolver` (`default: ForceBackend::BarnesHut`).
    pub yifan_hu_backend: ForceBackend,
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
            fit_bounds: true,
            seed: 42,
            fr_backend: FrBackend::default(),
            yifan_hu_backend: ForceBackend::default(),
        }
    }
}

impl GroupInABoxConfig {
    /// Creates a new [`GroupInABoxConfig`] initialized with [`Self::default`].
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the cluster box placement strategy `box_mode`.
    #[inline]
    pub fn with_box_mode(mut self, box_mode: GibBoxLayoutMode) -> Self {
        self.box_mode = box_mode;
        self
    }

    /// Sets the intra-cluster subgraph layout mode `intra_mode`.
    #[inline]
    pub fn with_intra_mode(mut self, intra_mode: GibIntraLayoutMode) -> Self {
        self.intra_mode = intra_mode;
        self
    }

    /// Sets the canvas width `total_width`.
    #[inline]
    pub fn with_total_width(mut self, total_width: f64) -> Self {
        self.total_width = total_width;
        self
    }

    /// Sets the canvas height `total_height`.
    #[inline]
    pub fn with_total_height(mut self, total_height: f64) -> Self {
        self.total_height = total_height;
        self
    }

    /// Sets the relative inner padding fraction `box_padding` per side in `[0.0, 0.5)`.
    #[inline]
    pub fn with_box_padding(mut self, box_padding: f64) -> Self {
        self.box_padding = box_padding;
        self
    }

    /// Sets the maximum cooperative intra-box layout iterations `intra_iterations`.
    #[inline]
    pub fn with_intra_iterations(mut self, intra_iterations: usize) -> Self {
        self.intra_iterations = intra_iterations;
        self
    }

    /// Sets the number of inter-cluster force-directed iterations `fd_iterations`.
    #[inline]
    pub fn with_fd_iterations(mut self, fd_iterations: usize) -> Self {
        self.fd_iterations = fd_iterations;
        self
    }

    /// Sets whether `ForceDirected` boxes are rescaled into `[0, total_width] x [0, total_height]`.
    #[inline]
    pub fn with_fit_bounds(mut self, fit_bounds: bool) -> Self {
        self.fit_bounds = fit_bounds;
        self
    }

    /// Sets the deterministic explicit RNG seed.
    #[inline]
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Sets the repulsion backend used by `FruchtermanReingoldSolver`.
    #[inline]
    pub fn with_fr_backend(mut self, fr_backend: FrBackend) -> Self {
        self.fr_backend = fr_backend;
        self
    }

    /// Sets the repulsion backend used by `YifanHuSolver`.
    #[inline]
    pub fn with_yifan_hu_backend(mut self, yifan_hu_backend: ForceBackend) -> Self {
        self.yifan_hu_backend = yifan_hu_backend;
        self
    }

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
        self.fr_backend.validate()?;
        self.yifan_hu_backend.validate()?;
        Ok(())
    }
}

/// Axis-aligned cluster bounding box produced by [`GroupInABoxSolver`](super::GroupInABoxSolver).
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
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

impl GibBox {
    /// Replaces `cluster_id` on this [`GibBox`].
    #[inline]
    pub fn with_cluster_id(mut self, cluster_id: u32) -> Self {
        self.cluster_id = cluster_id;
        self
    }

    /// Replaces `x_min` on this [`GibBox`].
    #[inline]
    pub fn with_x_min(mut self, x_min: f64) -> Self {
        self.x_min = x_min;
        self
    }

    /// Replaces `x_max` on this [`GibBox`].
    #[inline]
    pub fn with_x_max(mut self, x_max: f64) -> Self {
        self.x_max = x_max;
        self
    }

    /// Replaces `y_min` on this [`GibBox`].
    #[inline]
    pub fn with_y_min(mut self, y_min: f64) -> Self {
        self.y_min = y_min;
        self
    }

    /// Replaces `y_max` on this [`GibBox`].
    #[inline]
    pub fn with_y_max(mut self, y_max: f64) -> Self {
        self.y_max = y_max;
        self
    }

    /// Replaces `node_count` on this [`GibBox`].
    #[inline]
    pub fn with_node_count(mut self, node_count: usize) -> Self {
        self.node_count = node_count;
        self
    }

    /// Replaces `weight` on this [`GibBox`].
    #[inline]
    pub fn with_weight(mut self, weight: f64) -> Self {
        self.weight = weight;
        self
    }
}
