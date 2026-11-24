//! Field-based graph layout solvers (`t-FDP` and `ibUMAP`) built on interpolation FFT,
//! Barnes-Hut quadtrees, and exact $O(N^2)$ repulsion references.

use egraph_core::{
    CoreError, CoreResult, CsrGraph, DeterministicRng, IterativeSolver, Layout2d, Scalar,
    SolverStatus,
};
use egraph_neighborhood::{
    DogVaradhanConfig, DogVaradhanLandmarks, KnnGraph, Sparse12HopPairs, SparseProbabilities,
    build_gumap_12hop_priority_csr, compute_fuzzy_simplicial_set,
    compute_regularized_dog_varadhan_landmarks, extract_sparse_1hop_2hop_pairs, find_ab_params,
};
use egraph_spatial::{ArenaQuadtree2d, Fft2dWorkspace, IbFft2d, IbFftConfig};

const MIN_DIST_SQ_CLAMP: f64 = 1e-8;
const REPULSIVE_DIST_SQ_EPS: f64 = 0.05;
const DEFAULT_SIGMA_ITERS: usize = 64;
const DEFAULT_SIGMA_TOL: f64 = 1e-5;
const DEFAULT_TIER2_MARGIN: f64 = 1.1;
const DEFAULT_EPS_DIST: f64 = 1e-6;
const QUADTREE_MAX_DEPTH: usize = 32;
const QUADTREE_COINCIDENT_DIST_SQ: f64 = 1e-12;

/// Repulsion approximation backend used by [`FieldLayoutSolver`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum FieldRepulsionBackend {
    /// $O(N)$ interpolation-based 2D FFT convolution via [`IbFft2d`].
    #[default]
    IbFft,
    /// $O(N \log N)$ Barnes-Hut quadtree spatial approximation.
    BarnesHut,
    /// Exact $O(N^2)$ pairwise repulsion reference.
    Exact,
}

/// Power-law schedule $\rho(t)$ weighting repulsion across normalized progress $t \in [0, 1]$.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RhoSchedule {
    /// Repulsion multiplier at progress $t = 0.0$ (`default: 1.0`).
    pub start_weight: f64,
    /// Repulsion multiplier at progress $t = 1.0$ (`default: 0.1`).
    pub end_weight: f64,
    /// Exponent $p > 0$ controlling decay curvature $t^p$ (`default: 1.0`).
    pub power: f64,
}

impl Default for RhoSchedule {
    fn default() -> Self {
        Self {
            start_weight: 1.0,
            end_weight: 0.1,
            power: 1.0,
        }
    }
}

impl RhoSchedule {
    /// Validates schedule bounds and exponent.
    pub fn validate(&self) -> CoreResult<()> {
        if !self.start_weight.is_finite() {
            return Err(CoreError::non_finite("RhoSchedule::start_weight"));
        }
        if self.start_weight < 0.0 {
            return Err(CoreError::invalid_input(format!(
                "start_weight must be >= 0.0, got {}",
                self.start_weight
            )));
        }
        if !self.end_weight.is_finite() {
            return Err(CoreError::non_finite("RhoSchedule::end_weight"));
        }
        if self.end_weight < 0.0 {
            return Err(CoreError::invalid_input(format!(
                "end_weight must be >= 0.0, got {}",
                self.end_weight
            )));
        }
        if !self.power.is_finite() {
            return Err(CoreError::non_finite("RhoSchedule::power"));
        }
        if self.power <= 0.0 {
            return Err(CoreError::invalid_input(format!(
                "power must be > 0.0, got {}",
                self.power
            )));
        }
        Ok(())
    }

    /// Evaluates $\rho(\mathrm{progress}) = \rho_{\mathrm{start}} + (\rho_{\mathrm{end}} - \rho_{\mathrm{start}}) t^p$
    /// with `progress` clamped to `[0.0, 1.0]`.
    #[inline]
    pub fn evaluate(&self, progress: f64) -> f64 {
        let t = if progress.is_finite() {
            progress.clamp(0.0, 1.0)
        } else {
            0.0
        };
        let p = if self.power.is_finite() && self.power > 0.0 {
            self.power
        } else {
            1.0
        };
        self.start_weight + (self.end_weight - self.start_weight) * t.powf(p)
    }
}

/// Configuration for `t-FDP` bounded Student-$t$ force-directed placement.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TFdpConfig {
    /// Repulsion evaluation backend (`IbFft` by default).
    pub backend: FieldRepulsionBackend,
    /// Total optimization steps (`default: 200`).
    pub total_iterations: usize,
    /// Base gradient descent learning rate (`default: 1.0`).
    pub learning_rate: f64,
    /// Velocity momentum coefficient in `[0.0, 1.0)` (`default: 0.8`).
    pub momentum: f64,
    /// Barnes-Hut opening angle threshold $\theta \ge 0$ (`default: 0.5`).
    pub theta: f64,
    /// Attractive Student-$t$ scale $\alpha > 0$ (`default: 1.0`).
    pub alpha: f64,
    /// Repulsive kernel exponent $\gamma > 0$ in $(1 + d^2)^{-\gamma}$ (`default: 1.0`).
    pub gamma: f64,
    /// Dynamic repulsion schedule $\rho(t)$ across `0..total_iterations`.
    pub rho: RhoSchedule,
    /// Interpolation-based 2D FFT grid configuration.
    pub ibfft: IbFftConfig,
    /// Explicit deterministic seed for fallback zero-layout jitter (`default: 42`).
    pub seed: u64,
}

impl Default for TFdpConfig {
    fn default() -> Self {
        Self {
            backend: FieldRepulsionBackend::IbFft,
            total_iterations: 200,
            learning_rate: 1.0,
            momentum: 0.8,
            theta: 0.5,
            alpha: 1.0,
            gamma: 1.0,
            rho: RhoSchedule::default(),
            ibfft: IbFftConfig::default(),
            seed: 42,
        }
    }
}

impl TFdpConfig {
    /// Validates all `t-FDP` hyperparameters and nested schedules/configs.
    pub fn validate(&self) -> CoreResult<()> {
        if !self.learning_rate.is_finite() {
            return Err(CoreError::non_finite("TFdpConfig::learning_rate"));
        }
        if self.learning_rate <= 0.0 {
            return Err(CoreError::invalid_input(format!(
                "learning_rate must be > 0.0, got {}",
                self.learning_rate
            )));
        }
        if !self.momentum.is_finite() {
            return Err(CoreError::non_finite("TFdpConfig::momentum"));
        }
        if !(0.0..1.0).contains(&self.momentum) {
            return Err(CoreError::invalid_input(format!(
                "momentum must be in [0.0, 1.0), got {}",
                self.momentum
            )));
        }
        if !self.theta.is_finite() {
            return Err(CoreError::non_finite("TFdpConfig::theta"));
        }
        if self.theta < 0.0 {
            return Err(CoreError::invalid_input(format!(
                "theta must be >= 0.0, got {}",
                self.theta
            )));
        }
        if !self.alpha.is_finite() {
            return Err(CoreError::non_finite("TFdpConfig::alpha"));
        }
        if self.alpha <= 0.0 {
            return Err(CoreError::invalid_input(format!(
                "alpha must be > 0.0, got {}",
                self.alpha
            )));
        }
        if !self.gamma.is_finite() {
            return Err(CoreError::non_finite("TFdpConfig::gamma"));
        }
        if self.gamma <= 0.0 {
            return Err(CoreError::invalid_input(format!(
                "gamma must be > 0.0, got {}",
                self.gamma
            )));
        }
        self.rho.validate()?;
        self.ibfft.validate()
    }
}

/// Configuration for `ibUMAP` (interpolation-based batch Graph UMAP).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct IbUmapConfig {
    /// Repulsion evaluation backend (`IbFft` by default).
    pub backend: FieldRepulsionBackend,
    /// Total optimization epochs (`default: 200`).
    pub n_epochs: usize,
    /// Initial learning rate $\alpha_0$ (`default: 1.0`).
    pub learning_rate: f64,
    /// Target minimum separation in the 2D embedding (`default: 0.1`).
    pub min_dist: f64,
    /// Effective scale/spread parameter in the 2D embedding (`default: 1.0`).
    pub spread: f64,
    /// Optional explicit UMAP curve parameter `a` (`None` fits from `min_dist` and `spread`).
    pub a: Option<f64>,
    /// Optional explicit UMAP curve parameter `b` (`None` fits from `min_dist` and `spread`).
    pub b: Option<f64>,
    /// Repulsion calibration constant $c_K$ (`default: 4.0`).
    pub c_k: f64,
    /// Per-node step clipping multiplier $\tau_t = \tau_{\mathrm{scale}} \alpha_t$ (`default: 4.0`).
    pub tau_scale: f64,
    /// Whether to apply inverse-degree hub damping $\eta_i$ (`default: true`).
    pub hub_damping: bool,
    /// Barnes-Hut opening angle threshold $\theta \ge 0$ (`default: 0.5`).
    pub theta: f64,
    /// Interpolation-based 2D FFT grid configuration.
    pub ibfft: IbFftConfig,
    /// Explicit deterministic seed for fallback zero-layout jitter (`default: 42`).
    pub seed: u64,
}

impl Default for IbUmapConfig {
    fn default() -> Self {
        Self {
            backend: FieldRepulsionBackend::IbFft,
            n_epochs: 200,
            learning_rate: 1.0,
            min_dist: 0.1,
            spread: 1.0,
            a: None,
            b: None,
            c_k: 4.0,
            tau_scale: 4.0,
            hub_damping: true,
            theta: 0.5,
            ibfft: IbFftConfig::default(),
            seed: 42,
        }
    }
}

impl IbUmapConfig {
    /// Validates `(min_dist, spread, a, b)` and returns effective `(a, b)` parameters.
    pub fn resolve_ab(&self) -> CoreResult<(f64, f64)> {
        if !self.min_dist.is_finite() {
            return Err(CoreError::non_finite("IbUmapConfig::min_dist"));
        }
        if self.min_dist < 0.0 {
            return Err(CoreError::invalid_input(format!(
                "min_dist must be >= 0.0, got {}",
                self.min_dist
            )));
        }
        if !self.spread.is_finite() {
            return Err(CoreError::non_finite("IbUmapConfig::spread"));
        }
        if self.spread <= 0.0 {
            return Err(CoreError::invalid_input(format!(
                "spread must be > 0.0, got {}",
                self.spread
            )));
        }
        if let Some(a_val) = self.a {
            if !a_val.is_finite() {
                return Err(CoreError::non_finite("IbUmapConfig::a"));
            }
            if a_val <= 0.0 {
                return Err(CoreError::invalid_input(format!(
                    "a must be > 0.0, got {a_val}"
                )));
            }
        }
        if let Some(b_val) = self.b {
            if !b_val.is_finite() {
                return Err(CoreError::non_finite("IbUmapConfig::b"));
            }
            if b_val <= 0.0 {
                return Err(CoreError::invalid_input(format!(
                    "b must be > 0.0, got {b_val}"
                )));
            }
        }

        let (eff_a, eff_b) = match (self.a, self.b) {
            (Some(a_val), Some(b_val)) => (a_val, b_val),
            _ => {
                let (fitted_a, fitted_b) = find_ab_params(self.min_dist, self.spread)?;
                (self.a.unwrap_or(fitted_a), self.b.unwrap_or(fitted_b))
            }
        };
        if !eff_a.is_finite() || eff_a <= 0.0 || !eff_b.is_finite() || eff_b <= 0.0 {
            return Err(CoreError::numerical_breakdown(
                "resolved UMAP (a, b) parameters must be finite and positive",
            ));
        }
        Ok((eff_a, eff_b))
    }

    /// Validates all `ibUMAP` hyperparameters and nested FFT configuration.
    pub fn validate(&self) -> CoreResult<()> {
        let _ = self.resolve_ab()?;
        if !self.learning_rate.is_finite() {
            return Err(CoreError::non_finite("IbUmapConfig::learning_rate"));
        }
        if self.learning_rate <= 0.0 {
            return Err(CoreError::invalid_input(format!(
                "learning_rate must be > 0.0, got {}",
                self.learning_rate
            )));
        }
        if !self.c_k.is_finite() {
            return Err(CoreError::non_finite("IbUmapConfig::c_k"));
        }
        if self.c_k < 0.0 {
            return Err(CoreError::invalid_input(format!(
                "c_k must be >= 0.0, got {}",
                self.c_k
            )));
        }
        if !self.tau_scale.is_finite() {
            return Err(CoreError::non_finite("IbUmapConfig::tau_scale"));
        }
        if self.tau_scale <= 0.0 {
            return Err(CoreError::invalid_input(format!(
                "tau_scale must be > 0.0, got {}",
                self.tau_scale
            )));
        }
        if !self.theta.is_finite() {
            return Err(CoreError::non_finite("IbUmapConfig::theta"));
        }
        if self.theta < 0.0 {
            return Err(CoreError::invalid_input(format!(
                "theta must be >= 0.0, got {}",
                self.theta
            )));
        }
        self.ibfft.validate()
    }
}

#[derive(Debug, Clone, Copy)]
struct FieldQuadNode {
    center_x: f64,
    center_y: f64,
    width: f64,
    height: f64,
    count: usize,
    com_x: f64,
    com_y: f64,
    leaf_head: u32,
    first_child: u32,
}

impl FieldQuadNode {
    #[inline]
    fn new(center_x: f64, center_y: f64, width: f64, height: f64) -> Self {
        Self {
            center_x,
            center_y,
            width,
            height,
            count: 0,
            com_x: 0.0,
            com_y: 0.0,
            leaf_head: u32::MAX,
            first_child: u32::MAX,
        }
    }

    #[inline]
    fn quadrant(&self, x: f64, y: f64) -> usize {
        let right = usize::from(x >= self.center_x);
        let bottom = usize::from(y >= self.center_y) * 2;
        right + bottom
    }

    #[inline]
    fn is_unsplitable_at_depth(&self, depth: usize) -> bool {
        if depth >= QUADTREE_MAX_DEPTH {
            return true;
        }
        let quarter_w = self.width * 0.25;
        let quarter_h = self.height * 0.25;
        quarter_w <= 0.0
            || quarter_h <= 0.0
            || (self.center_x - quarter_w) == self.center_x
            || (self.center_x + quarter_w) == self.center_x
            || (self.center_y - quarter_h) == self.center_y
            || (self.center_y + quarter_h) == self.center_y
    }
}

/// Reusable arena quadtree for arbitrary isotropic radial kernels $K(d^2)$ in Barnes-Hut mode.
#[derive(Debug, Clone, Default)]
struct FieldQuadtree {
    nodes: Vec<FieldQuadNode>,
    coords_x: Vec<f64>,
    coords_y: Vec<f64>,
    leaf_of_point: Vec<u32>,
    next_in_leaf: Vec<u32>,
}

impl FieldQuadtree {
    fn new() -> Self {
        Self::default()
    }

    fn clear(&mut self) {
        self.nodes.clear();
        self.coords_x.clear();
        self.coords_y.clear();
        self.leaf_of_point.clear();
        self.next_in_leaf.clear();
    }

    fn build_xy<S: Scalar>(&mut self, xs: &[S], ys: &[S]) -> CoreResult<()> {
        if xs.len() != ys.len() {
            return Err(CoreError::dimension_mismatch(
                xs.len(),
                ys.len(),
                "FieldQuadtree::build_xy",
            ));
        }
        let n = xs.len();
        for i in 0..n {
            if !xs[i].is_finite() || !ys[i].is_finite() {
                return Err(CoreError::non_finite("FieldQuadtree::build_xy"));
            }
        }
        self.clear();
        if n == 0 {
            return Ok(());
        }

        self.coords_x.reserve(n);
        self.coords_y.reserve(n);
        let mut min_x = xs[0].to_f64();
        let mut max_x = min_x;
        let mut min_y = ys[0].to_f64();
        let mut max_y = min_y;

        for i in 0..n {
            let x = xs[i].to_f64();
            let y = ys[i].to_f64();
            min_x = min_x.min(x);
            max_x = max_x.max(x);
            min_y = min_y.min(y);
            max_y = max_y.max(y);
            self.coords_x.push(x);
            self.coords_y.push(y);
        }

        self.leaf_of_point.resize(n, u32::MAX);
        self.next_in_leaf.resize(n, u32::MAX);

        let center_x = 0.5 * (min_x + max_x);
        let center_y = 0.5 * (min_y + max_y);
        let width = ((max_x - min_x) * 1.01).max(1e-4);
        let height = ((max_y - min_y) * 1.01).max(1e-4);
        self.nodes
            .push(FieldQuadNode::new(center_x, center_y, width, height));

        for i in 0..n {
            self.insert_iterative(i, self.coords_x[i], self.coords_y[i]);
        }
        Ok(())
    }

    fn insert_iterative(&mut self, point_idx: usize, x: f64, y: f64) {
        let mut cur = 0usize;
        let mut depth = 0usize;

        loop {
            let node = self.nodes[cur];
            if node.count == 0 {
                let slot = &mut self.nodes[cur];
                slot.count = 1;
                slot.com_x = x;
                slot.com_y = y;
                slot.leaf_head = point_idx as u32;
                self.leaf_of_point[point_idx] = cur as u32;
                self.next_in_leaf[point_idx] = u32::MAX;
                return;
            }

            let old_c = node.count as f64;
            let new_c = (node.count + 1) as f64;
            {
                let slot = &mut self.nodes[cur];
                slot.com_x = (slot.com_x * old_c + x) / new_c;
                slot.com_y = (slot.com_y * old_c + y) / new_c;
                slot.count += 1;
            }

            if node.first_child == u32::MAX {
                let ref_idx = node.leaf_head as usize;
                let rx = self.coords_x[ref_idx];
                let ry = self.coords_y[ref_idx];
                let dist_sq = (rx - x) * (rx - x) + (ry - y) * (ry - y);
                if !dist_sq.is_finite()
                    || dist_sq < QUADTREE_COINCIDENT_DIST_SQ
                    || node.is_unsplitable_at_depth(depth)
                {
                    self.next_in_leaf[point_idx] = self.nodes[cur].leaf_head;
                    self.nodes[cur].leaf_head = point_idx as u32;
                    self.leaf_of_point[point_idx] = cur as u32;
                    return;
                }

                let head = self.nodes[cur].leaf_head;
                self.nodes[cur].leaf_head = u32::MAX;
                let base_child = self.subdivide_node(cur);

                let mut existing = head;
                while existing != u32::MAX {
                    let e_idx = existing as usize;
                    let next = self.next_in_leaf[e_idx];
                    self.next_in_leaf[e_idx] = u32::MAX;
                    let ex = self.coords_x[e_idx];
                    let ey = self.coords_y[e_idx];
                    let q = self.nodes[cur].quadrant(ex, ey);
                    self.insert_into_subtree(base_child + q, depth + 1, e_idx, ex, ey);
                    existing = next;
                }

                let q_new = self.nodes[cur].quadrant(x, y);
                cur = base_child + q_new;
                depth += 1;
            } else {
                let q = node.quadrant(x, y);
                cur = (node.first_child as usize) + q;
                depth += 1;
            }
        }
    }

    fn insert_into_subtree(
        &mut self,
        mut cur: usize,
        mut depth: usize,
        point_idx: usize,
        x: f64,
        y: f64,
    ) {
        loop {
            let node = self.nodes[cur];
            if node.count == 0 {
                let slot = &mut self.nodes[cur];
                slot.count = 1;
                slot.com_x = x;
                slot.com_y = y;
                slot.leaf_head = point_idx as u32;
                self.leaf_of_point[point_idx] = cur as u32;
                self.next_in_leaf[point_idx] = u32::MAX;
                return;
            }

            let old_c = node.count as f64;
            let new_c = (node.count + 1) as f64;
            {
                let slot = &mut self.nodes[cur];
                slot.com_x = (slot.com_x * old_c + x) / new_c;
                slot.com_y = (slot.com_y * old_c + y) / new_c;
                slot.count += 1;
            }

            if node.first_child == u32::MAX {
                let ref_idx = node.leaf_head as usize;
                let rx = self.coords_x[ref_idx];
                let ry = self.coords_y[ref_idx];
                let dist_sq = (rx - x) * (rx - x) + (ry - y) * (ry - y);
                if !dist_sq.is_finite()
                    || dist_sq < QUADTREE_COINCIDENT_DIST_SQ
                    || node.is_unsplitable_at_depth(depth)
                {
                    self.next_in_leaf[point_idx] = self.nodes[cur].leaf_head;
                    self.nodes[cur].leaf_head = point_idx as u32;
                    self.leaf_of_point[point_idx] = cur as u32;
                    return;
                }

                let head = self.nodes[cur].leaf_head;
                self.nodes[cur].leaf_head = u32::MAX;
                let base_child = self.subdivide_node(cur);
                let mut existing = head;
                while existing != u32::MAX {
                    let e_idx = existing as usize;
                    let next = self.next_in_leaf[e_idx];
                    self.next_in_leaf[e_idx] = u32::MAX;
                    let ex = self.coords_x[e_idx];
                    let ey = self.coords_y[e_idx];
                    let q = self.nodes[cur].quadrant(ex, ey);
                    let child = base_child + q;
                    let c_node = &mut self.nodes[child];
                    if c_node.count == 0 {
                        c_node.count = 1;
                        c_node.com_x = ex;
                        c_node.com_y = ey;
                        c_node.leaf_head = e_idx as u32;
                        self.leaf_of_point[e_idx] = child as u32;
                    } else {
                        let oc = c_node.count as f64;
                        let nc = (c_node.count + 1) as f64;
                        c_node.com_x = (c_node.com_x * oc + ex) / nc;
                        c_node.com_y = (c_node.com_y * oc + ey) / nc;
                        c_node.count += 1;
                        self.next_in_leaf[e_idx] = c_node.leaf_head;
                        c_node.leaf_head = e_idx as u32;
                        self.leaf_of_point[e_idx] = child as u32;
                    }
                    existing = next;
                }
                let q_new = self.nodes[cur].quadrant(x, y);
                cur = base_child + q_new;
                depth += 1;
            } else {
                let q = node.quadrant(x, y);
                cur = (node.first_child as usize) + q;
                depth += 1;
            }
        }
    }

    fn subdivide_node(&mut self, parent_idx: usize) -> usize {
        let p = self.nodes[parent_idx];
        let half_w = p.width * 0.5;
        let half_h = p.height * 0.5;
        let qw = p.width * 0.25;
        let qh = p.height * 0.25;

        let base = self.nodes.len();
        self.nodes[parent_idx].first_child = base as u32;
        self.nodes.push(FieldQuadNode::new(
            p.center_x - qw,
            p.center_y - qh,
            half_w,
            half_h,
        ));
        self.nodes.push(FieldQuadNode::new(
            p.center_x + qw,
            p.center_y - qh,
            half_w,
            half_h,
        ));
        self.nodes.push(FieldQuadNode::new(
            p.center_x - qw,
            p.center_y + qh,
            half_w,
            half_h,
        ));
        self.nodes.push(FieldQuadNode::new(
            p.center_x + qw,
            p.center_y + qh,
            half_w,
            half_h,
        ));
        base
    }

    fn compute_isotropic_repulsion<S: Scalar, F: Fn(f64) -> f64>(
        &self,
        theta: f64,
        kernel_fn: F,
        out_x: &mut [S],
        out_y: &mut [S],
    ) {
        let n = self.coords_x.len();
        if n < 2 || self.nodes.is_empty() {
            out_x.fill(S::ZERO);
            out_y.fill(S::ZERO);
            return;
        }
        let theta_sq = theta * theta;
        let mut stack = [0u32; 128];

        for i in 0..n {
            let x = self.coords_x[i];
            let y = self.coords_y[i];
            let own_leaf = self.leaf_of_point[i];
            let mut sp = 1usize;
            stack[0] = 0;

            let mut fx = 0.0_f64;
            let mut fy = 0.0_f64;

            while sp > 0 {
                sp -= 1;
                let node_idx = stack[sp];
                let node = &self.nodes[node_idx as usize];
                if node.count == 0 {
                    continue;
                }

                if node.first_child == u32::MAX {
                    if node_idx == own_leaf {
                        let mut p = node.leaf_head;
                        while p != u32::MAX {
                            let other = p as usize;
                            if other != i {
                                let dx = x - self.coords_x[other];
                                let dy = y - self.coords_y[other];
                                let k = kernel_fn(dx * dx + dy * dy);
                                fx += k * dx;
                                fy += k * dy;
                            }
                            p = self.next_in_leaf[other];
                        }
                    } else {
                        let dx = x - node.com_x;
                        let dy = y - node.com_y;
                        let k = (node.count as f64) * kernel_fn(dx * dx + dy * dy);
                        fx += k * dx;
                        fy += k * dy;
                    }
                    continue;
                }

                let dx = x - node.com_x;
                let dy = y - node.com_y;
                let dist_sq = dx * dx + dy * dy;
                let max_dim = node.width.max(node.height);
                if (max_dim * max_dim) < theta_sq * dist_sq {
                    let k = (node.count as f64) * kernel_fn(dist_sq);
                    fx += k * dx;
                    fy += k * dy;
                } else {
                    let base = node.first_child;
                    for c in 0..4 {
                        let child = base + c;
                        if self.nodes[child as usize].count > 0 && sp < stack.len() {
                            stack[sp] = child;
                            sp += 1;
                        }
                    }
                }
            }

            out_x[i] = S::from_f64(fx);
            out_y[i] = S::from_f64(fy);
        }
    }
}

#[derive(Debug, Clone, Copy)]
enum FieldSolverMode {
    TFdp {
        config: TFdpConfig,
        rep_scale: f64,
    },
    IbUmap {
        config: IbUmapConfig,
        a: f64,
        b: f64,
    },
}

/// Cooperative, zero-per-step-allocation field-based layout solver supporting `t-FDP` and `ibUMAP`.
///
/// Implements [`IterativeSolver`] with bit-exact equivalence between `step()`, `run_steps(k)`,
/// and `run()` across both `f64` and `f32`.
#[derive(Debug, Clone)]
pub struct FieldLayoutSolver<S: Scalar = f64> {
    mode: FieldSolverMode,
    backend: FieldRepulsionBackend,
    num_nodes: usize,
    indptr: Vec<usize>,
    indices: Vec<u32>,
    edge_weights: Vec<f32>,
    hub_eta: Vec<f64>,
    attr_x: Vec<f64>,
    attr_y: Vec<f64>,
    vel_x: Vec<f64>,
    vel_y: Vec<f64>,
    next_x: Vec<f64>,
    next_y: Vec<f64>,
    phi_1: Vec<S>,
    rep_x: Vec<S>,
    rep_y: Vec<S>,
    fft_ws: Fft2dWorkspace,
    ibfft: Option<IbFft2d<S>>,
    quadtree: ArenaQuadtree2d<S>,
    field_quadtree: FieldQuadtree,
    rng: DeterministicRng,
    iteration: usize,
    total_iterations: usize,
    status: SolverStatus,
}

/// Convenient alias for [`FieldLayoutSolver`].
pub type FieldLayout<S = f64> = FieldLayoutSolver<S>;

impl<S: Scalar> FieldLayoutSolver<S> {
    /// Constructs a `t-FDP` [`FieldLayoutSolver`] directly from a [`CsrGraph`].
    pub fn t_fdp<W: Scalar>(graph: &CsrGraph<W>, config: TFdpConfig) -> CoreResult<Self> {
        config.validate()?;
        let n = graph.num_nodes();
        if u32::try_from(n).is_err() {
            return Err(CoreError::invalid_csr("n_nodes exceeds u32::MAX"));
        }
        if let Some(ws) = graph.weights() {
            for &w in ws {
                let wf = w.to_f64();
                if !wf.is_finite() {
                    return Err(CoreError::non_finite("FieldLayoutSolver::t_fdp::weights"));
                }
                if wf < 0.0 {
                    return Err(CoreError::invalid_input(format!(
                        "graph edge weights must be >= 0.0, got {wf}"
                    )));
                }
            }
        }

        let (indptr, indices, edge_weights) = extract_symmetric_csr_edges(graph);
        Self::build_t_fdp_inner(n, indptr, indices, edge_weights, config)
    }

    /// Constructs a `t-FDP` [`FieldLayoutSolver`] from a validated [`KnnGraph`], converting
    /// neighbor distances $d_{ij} \ge 0$ to symmetric weights $1 / (1 + d_{ij})$.
    pub fn t_fdp_from_knn(knn: &KnnGraph, config: TFdpConfig) -> CoreResult<Self> {
        config.validate()?;
        let n = knn.num_nodes();
        if u32::try_from(n).is_err() {
            return Err(CoreError::invalid_csr("n_nodes exceeds u32::MAX"));
        }

        let mut pairs: Vec<(u32, u32, f32)> = Vec::with_capacity(knn.nnz() * 2);
        for u in 0..n {
            let u_u32 = u as u32;
            let (nbrs, dists) = knn.row(u);
            for (&v, &d) in nbrs.iter().zip(dists.iter()) {
                let v_usize = v as usize;
                if v_usize == u || v_usize >= n {
                    continue;
                }
                if !d.is_finite() {
                    return Err(CoreError::non_finite(
                        "FieldLayoutSolver::t_fdp_from_knn::distances",
                    ));
                }
                if d < 0.0 {
                    return Err(CoreError::invalid_input(format!(
                        "KnnGraph distances must be >= 0.0, got {d}"
                    )));
                }
                let w = (1.0 / (1.0 + d)) as f32;
                pairs.push((u_u32, v, w));
                pairs.push((v, u_u32, w));
            }
        }

        let (indptr, indices, edge_weights) = symmetrize_pair_list(n, pairs);
        Self::build_t_fdp_inner(n, indptr, indices, edge_weights, config)
    }

    fn build_t_fdp_inner(
        n: usize,
        indptr: Vec<usize>,
        indices: Vec<u32>,
        edge_weights: Vec<f32>,
        config: TFdpConfig,
    ) -> CoreResult<Self> {
        let ibfft = if config.backend == FieldRepulsionBackend::IbFft {
            Some(IbFft2d::new(config.ibfft)?)
        } else {
            None
        };
        let total_iterations = config.total_iterations;
        let status = if total_iterations == 0 {
            SolverStatus::Completed
        } else {
            SolverStatus::Initialized
        };
        let buf_len = if config.backend == FieldRepulsionBackend::IbFft {
            n
        } else {
            0
        };
        let total_w: f64 = edge_weights.iter().map(|&w| w as f64).sum();
        let avg_deg = if n > 0 {
            (total_w / (n as f64)).max(1.0)
        } else {
            1.0
        };
        let rep_scale = if n > 1 {
            avg_deg / ((n - 1) as f64)
        } else {
            1.0
        };

        Ok(Self {
            mode: FieldSolverMode::TFdp { config, rep_scale },
            backend: config.backend,
            num_nodes: n,
            indptr,
            indices,
            edge_weights,
            hub_eta: Vec::new(),
            attr_x: vec![0.0; n],
            attr_y: vec![0.0; n],
            vel_x: vec![0.0; n],
            vel_y: vec![0.0; n],
            next_x: vec![0.0; n],
            next_y: vec![0.0; n],
            phi_1: vec![S::ZERO; buf_len],
            rep_x: vec![S::ZERO; n],
            rep_y: vec![S::ZERO; n],
            fft_ws: Fft2dWorkspace::new(),
            ibfft,
            quadtree: ArenaQuadtree2d::new(),
            field_quadtree: FieldQuadtree::new(),
            rng: DeterministicRng::from_seed_u64(config.seed),
            iteration: 0,
            total_iterations,
            status,
        })
    }

    fn compute_12hop_alt_distances<W: Scalar>(
        graph: &CsrGraph<W>,
        landmarks: &DogVaradhanLandmarks,
        max_2hop_per_node: usize,
        seed: u64,
    ) -> CoreResult<(Sparse12HopPairs, Vec<f64>, Vec<f64>)> {
        let n = graph.num_nodes();
        if u32::try_from(n).is_err() {
            return Err(CoreError::invalid_csr("n_nodes exceeds u32::MAX"));
        }
        if landmarks.num_nodes() != n {
            return Err(CoreError::dimension_mismatch(
                n,
                landmarks.num_nodes(),
                "FieldLayoutSolver::ib_umap_from_landmarks::landmarks",
            ));
        }
        let k_lm = landmarks.num_landmarks();
        if n > 0 && k_lm == 0 {
            return Err(CoreError::invalid_input(
                "landmarks must have at least 1 landmark when num_nodes > 0",
            ));
        }
        let expected = n
            .checked_mul(k_lm)
            .ok_or_else(|| CoreError::invalid_input("landmark table size overflow"))?;
        if landmarks.d_dog().len() != expected {
            return Err(CoreError::dimension_mismatch(
                expected,
                landmarks.d_dog().len(),
                "DogVaradhanLandmarks::d_dog",
            ));
        }
        if landmarks.d_dog().iter().any(|v| !v.is_finite()) {
            return Err(CoreError::non_finite("DogVaradhanLandmarks::d_dog"));
        }
        if let Some(dc) = landmarks.d_coarse() {
            if dc.len() != expected {
                return Err(CoreError::dimension_mismatch(
                    expected,
                    dc.len(),
                    "DogVaradhanLandmarks::d_coarse",
                ));
            }
            if dc.iter().any(|v| !v.is_finite()) {
                return Err(CoreError::non_finite("DogVaradhanLandmarks::d_coarse"));
            }
        }
        for &lm in landmarks.landmarks() {
            if (lm as usize) >= n {
                return Err(CoreError::index_out_of_bounds(
                    lm as usize,
                    n,
                    "DogVaradhanLandmarks::landmarks",
                ));
            }
        }

        let pairs = extract_sparse_1hop_2hop_pairs(graph, max_2hop_per_node, seed)?;
        let mut d1 = landmarks.alt_pair_distances(&pairs.u1, &pairs.v1)?;
        let mut d2 = landmarks.alt_pair_distances(&pairs.u2, &pairs.v2)?;
        for d in &mut d1 {
            *d = d.max(DEFAULT_EPS_DIST);
        }
        for d in &mut d2 {
            *d = d.max(DEFAULT_EPS_DIST);
        }
        Ok((pairs, d1, d2))
    }

    fn finish_ib_umap_from_pairs<W: Scalar>(
        graph: &CsrGraph<W>,
        pairs: Sparse12HopPairs,
        d1: Vec<f64>,
        d2: Vec<f64>,
        config: IbUmapConfig,
    ) -> CoreResult<Self> {
        let n = graph.num_nodes();
        let knn = build_gumap_12hop_priority_csr(
            n,
            &pairs.u1,
            &pairs.v1,
            &d1,
            &pairs.u2,
            &pairs.v2,
            &d2,
            DEFAULT_TIER2_MARGIN,
        )?;
        drop(pairs);
        drop(d1);
        drop(d2);
        let probs = compute_fuzzy_simplicial_set(&knn, DEFAULT_SIGMA_ITERS, DEFAULT_SIGMA_TOL)?;
        drop(knn);
        let mut degrees = vec![0u32; n];
        if !graph.is_directed() {
            for (u, deg_out) in degrees.iter_mut().enumerate() {
                *deg_out = graph.degree(u) as u32;
            }
        } else {
            for u in 0..n {
                for &v in graph.neighbors(u) {
                    let v_usize = v as usize;
                    if v_usize != u && v_usize < n {
                        degrees[u] = degrees[u].saturating_add(1);
                        degrees[v_usize] = degrees[v_usize].saturating_add(1);
                    }
                }
            }
        }
        Self::ib_umap_from_fuzzy_set(probs, Some(&degrees), config)
    }

    /// Constructs an `ibUMAP` [`FieldLayoutSolver`] from a [`CsrGraph`] and precomputed
    /// [`DogVaradhanLandmarks`] using $O(|E|)$ hub-safe 1+2-hop pairs, landmark ALT
    /// distances, priority CSR construction, and fuzzy simplicial sets.
    pub fn ib_umap_from_landmarks<W: Scalar>(
        graph: &CsrGraph<W>,
        landmarks: &DogVaradhanLandmarks,
        max_2hop_per_node: usize,
        config: IbUmapConfig,
    ) -> CoreResult<Self> {
        config.validate()?;
        let n = graph.num_nodes();
        let (pairs, d1, d2) =
            Self::compute_12hop_alt_distances(graph, landmarks, max_2hop_per_node, config.seed)?;
        if n <= 1 {
            drop(pairs);
            drop(d1);
            drop(d2);
            let degs = vec![0u32; n];
            return Self::ib_umap_from_fuzzy_set(
                SparseProbabilities::empty(n),
                Some(&degs),
                config,
            );
        }
        Self::finish_ib_umap_from_pairs(graph, pairs, d1, d2, config)
    }

    /// Constructs an `ibUMAP` [`FieldLayoutSolver`] from a [`CsrGraph`] using $O(|E|)$
    /// 1+2-hop pair sampling, regularized DoG Varadhan landmark ALT distances, priority CSR
    /// construction, and fuzzy simplicial set computation.
    pub fn ib_umap<W: Scalar>(
        graph: &CsrGraph<W>,
        max_2hop_per_node: usize,
        dog_cfg: &DogVaradhanConfig,
        config: IbUmapConfig,
    ) -> CoreResult<Self> {
        config.validate()?;
        let n = graph.num_nodes();
        if u32::try_from(n).is_err() {
            return Err(CoreError::invalid_csr("n_nodes exceeds u32::MAX"));
        }
        let (landmarks, _dog_stats) = compute_regularized_dog_varadhan_landmarks(graph, dog_cfg)?;
        let (pairs, d1, d2) =
            Self::compute_12hop_alt_distances(graph, &landmarks, max_2hop_per_node, config.seed)?;
        drop(landmarks);
        if n <= 1 {
            drop(pairs);
            drop(d1);
            drop(d2);
            let degs = vec![0u32; n];
            return Self::ib_umap_from_fuzzy_set(
                SparseProbabilities::empty(n),
                Some(&degs),
                config,
            );
        }
        Self::finish_ib_umap_from_pairs(graph, pairs, d1, d2, config)
    }

    /// Constructs an `ibUMAP` [`FieldLayoutSolver`] from a validated [`KnnGraph`] via
    /// [`compute_fuzzy_simplicial_set`].
    pub fn ib_umap_from_knn(
        knn: &KnnGraph,
        sigma_iters: usize,
        sigma_tol: f64,
        config: IbUmapConfig,
    ) -> CoreResult<Self> {
        config.validate()?;
        let probs = compute_fuzzy_simplicial_set(knn, sigma_iters, sigma_tol)?;
        Self::ib_umap_from_fuzzy_set(probs, None, config)
    }

    fn build_ib_umap_inner(
        n: usize,
        indptr: Vec<usize>,
        indices: Vec<u32>,
        edge_weights: Vec<f32>,
        degrees: Option<&[u32]>,
        config: IbUmapConfig,
    ) -> CoreResult<Self> {
        let (a, b) = config.resolve_ab()?;
        let ibfft = if config.backend == FieldRepulsionBackend::IbFft {
            Some(IbFft2d::new(config.ibfft)?)
        } else {
            None
        };

        let mut hub_eta = vec![1.0_f64; n];
        if config.hub_damping {
            for (i, eta_out) in hub_eta.iter_mut().enumerate() {
                let deg_i = match degrees {
                    Some(deg) => deg[i],
                    None => (indptr[i + 1] - indptr[i]) as u32,
                };
                *eta_out = 1.0 / (deg_i.max(1) as f64);
            }
        }

        let total_iterations = config.n_epochs;
        let status = if total_iterations == 0 {
            SolverStatus::Completed
        } else {
            SolverStatus::Initialized
        };
        let buf_len = if config.backend == FieldRepulsionBackend::IbFft {
            n
        } else {
            0
        };

        Ok(Self {
            mode: FieldSolverMode::IbUmap { config, a, b },
            backend: config.backend,
            num_nodes: n,
            indptr,
            indices,
            edge_weights,
            hub_eta,
            attr_x: vec![0.0; n],
            attr_y: vec![0.0; n],
            vel_x: Vec::new(),
            vel_y: Vec::new(),
            next_x: vec![0.0; n],
            next_y: vec![0.0; n],
            phi_1: vec![S::ZERO; buf_len],
            rep_x: vec![S::ZERO; n],
            rep_y: vec![S::ZERO; n],
            fft_ws: Fft2dWorkspace::new(),
            ibfft,
            quadtree: ArenaQuadtree2d::new(),
            field_quadtree: FieldQuadtree::new(),
            rng: DeterministicRng::from_seed_u64(config.seed),
            iteration: 0,
            total_iterations,
            status,
        })
    }

    /// Constructs an `ibUMAP` [`FieldLayoutSolver`] directly from a precomputed fuzzy simplicial
    /// set [`SparseProbabilities`] and optional per-node degrees for inverse-degree hub damping.
    pub fn ib_umap_from_fuzzy_set(
        probs: SparseProbabilities,
        degrees: Option<&[u32]>,
        config: IbUmapConfig,
    ) -> CoreResult<Self> {
        config.validate()?;
        validate_sparse_probabilities(&probs)?;
        let n = probs.n_nodes;
        if let Some(deg) = degrees
            && deg.len() != n
        {
            return Err(CoreError::dimension_mismatch(
                n,
                deg.len(),
                "FieldLayoutSolver::ib_umap_from_fuzzy_set::degrees",
            ));
        }
        let SparseProbabilities {
            n_nodes: _,
            indptr,
            indices,
            values,
        } = probs;
        let edge_weights: Vec<f32> = values.into_iter().map(|w| w as f32).collect();
        Self::build_ib_umap_inner(n, indptr, indices, edge_weights, degrees, config)
    }

    /// Constructs an `ibUMAP` [`FieldLayoutSolver`] from a borrowed [`SparseProbabilities`]
    /// without cloning its `f64` values vector.
    pub fn ib_umap_from_fuzzy_set_ref(
        probs: &SparseProbabilities,
        degrees: Option<&[u32]>,
        config: IbUmapConfig,
    ) -> CoreResult<Self> {
        config.validate()?;
        validate_sparse_probabilities(probs)?;
        let n = probs.n_nodes;
        if let Some(deg) = degrees
            && deg.len() != n
        {
            return Err(CoreError::dimension_mismatch(
                n,
                deg.len(),
                "FieldLayoutSolver::ib_umap_from_fuzzy_set::degrees",
            ));
        }
        let indptr = probs.indptr.clone();
        let indices = probs.indices.clone();
        let edge_weights: Vec<f32> = probs.values.iter().map(|&w| w as f32).collect();
        Self::build_ib_umap_inner(n, indptr, indices, edge_weights, degrees, config)
    }

    /// Returns the number of vertices expected in [`Layout2d`].
    #[inline]
    pub fn num_nodes(&self) -> usize {
        self.num_nodes
    }

    /// Returns the active repulsion evaluation backend.
    #[inline]
    pub fn backend(&self) -> FieldRepulsionBackend {
        self.backend
    }

    /// Returns the number of stored directed positive attraction entries.
    #[inline]
    pub fn num_edges(&self) -> usize {
        self.indices.len()
    }

    /// Returns the cumulative number of `fft_2d` calls executed across all steps.
    #[inline]
    pub fn total_fft_2d_calls(&self) -> u64 {
        self.fft_ws.fft_2d_call_count()
    }

    /// Returns the resolved UMAP `(a, b)` parameters when in `ibUMAP` mode, or `None` for `t-FDP`.
    #[inline]
    pub fn ab(&self) -> Option<(f64, f64)> {
        match self.mode {
            FieldSolverMode::TFdp { .. } => None,
            FieldSolverMode::IbUmap { a, b, .. } => Some((a, b)),
        }
    }

    /// Returns the current $\rho(t)$ repulsion weight in `t-FDP` mode (or `1.0` in `ibUMAP` mode).
    #[inline]
    pub fn current_rho(&self) -> f64 {
        match self.mode {
            FieldSolverMode::TFdp { config, .. } => {
                let progress = if self.total_iterations <= 1 {
                    if self.iteration >= self.total_iterations {
                        1.0
                    } else {
                        0.0
                    }
                } else {
                    (self.iteration as f64) / ((self.total_iterations - 1) as f64)
                };
                config.rho.evaluate(progress)
            }
            FieldSolverMode::IbUmap { .. } => 1.0,
        }
    }

    fn step_t_fdp(
        &mut self,
        layout: &mut Layout2d<S>,
        config: TFdpConfig,
        rep_scale: f64,
    ) -> CoreResult<()> {
        let n = self.num_nodes;
        if self.iteration == 0 {
            let (xs, ys) = layout.xy_mut();
            if xs.iter().all(|&v| v == S::ZERO) && ys.iter().all(|&v| v == S::ZERO) {
                for idx in 0..n {
                    xs[idx] = S::from_f64(self.rng.next_f64() * 2.0 - 1.0);
                    ys[idx] = S::from_f64(self.rng.next_f64() * 2.0 - 1.0);
                }
            }
        }

        // 1. Bounded Student-t short-range attractive forces along CSR edges.
        let inv_alpha_sq = 1.0 / (config.alpha * config.alpha);
        {
            let xs = layout.x();
            let ys = layout.y();
            for i in 0..n {
                let xi = xs[i].to_f64();
                let yi = ys[i].to_f64();
                let start = self.indptr[i];
                let end = self.indptr[i + 1];
                let mut fx = 0.0_f64;
                let mut fy = 0.0_f64;
                for pos in start..end {
                    let j = self.indices[pos] as usize;
                    if j == i {
                        continue;
                    }
                    let dx = xi - xs[j].to_f64();
                    let dy = yi - ys[j].to_f64();
                    let d2 = dx * dx + dy * dy;
                    let mult = (self.edge_weights[pos] as f64) / (1.0 + d2 * inv_alpha_sq);
                    fx -= mult * dx;
                    fy -= mult * dy;
                }
                self.attr_x[i] = fx;
                self.attr_y[i] = fy;
            }
        }

        // 2. Global repulsive field evaluation.
        let gamma = config.gamma;
        match config.backend {
            FieldRepulsionBackend::IbFft => {
                let ibfft = self.ibfft.as_mut().ok_or_else(|| {
                    CoreError::numerical_breakdown("IbFft2d workspace missing for FieldLayout")
                })?;
                ibfft.prepare_xy(&mut self.fft_ws, layout.x(), layout.y())?;
                let soft = ibfft.grid_spec().map_or(0.0, |spec| spec.softening_sq);
                ibfft.evaluate_isotropic_kernel(
                    &mut self.fft_ws,
                    |d2| 1.0 / (1.0 + soft + d2).powf(gamma),
                    &mut self.phi_1,
                    &mut self.rep_x,
                    &mut self.rep_y,
                )?;
            }
            FieldRepulsionBackend::BarnesHut => {
                if (gamma - 1.0).abs() <= 1e-12 {
                    self.quadtree.build(layout)?;
                    self.quadtree.compute_coulomb_repulsion(
                        S::from_f64(config.theta),
                        S::ONE,
                        &mut self.rep_x,
                        &mut self.rep_y,
                    )?;
                } else if (gamma - 2.0).abs() <= 1e-12 {
                    self.quadtree.build(layout)?;
                    let total_z = self.quadtree.compute_kl_repulsion(
                        S::from_f64(config.theta),
                        &mut self.rep_x,
                        &mut self.rep_y,
                    )?;
                    let z_f64 = total_z.to_f64();
                    for i in 0..n {
                        self.rep_x[i] = S::from_f64(self.rep_x[i].to_f64() * z_f64);
                        self.rep_y[i] = S::from_f64(self.rep_y[i].to_f64() * z_f64);
                    }
                } else {
                    self.field_quadtree.build_xy(layout.x(), layout.y())?;
                    self.field_quadtree.compute_isotropic_repulsion(
                        config.theta,
                        |d2| 1.0 / (1.0 + d2).powf(gamma),
                        &mut self.rep_x,
                        &mut self.rep_y,
                    );
                }
            }
            FieldRepulsionBackend::Exact => {
                let xs = layout.x();
                let ys = layout.y();
                for i in 0..n {
                    let xi = xs[i].to_f64();
                    let yi = ys[i].to_f64();
                    let mut fx = 0.0_f64;
                    let mut fy = 0.0_f64;
                    for j in 0..n {
                        if j == i {
                            continue;
                        }
                        let dx = xi - xs[j].to_f64();
                        let dy = yi - ys[j].to_f64();
                        let d2 = dx * dx + dy * dy;
                        let k = 1.0 / (1.0 + d2).powf(gamma);
                        fx += k * dx;
                        fy += k * dy;
                    }
                    self.rep_x[i] = S::from_f64(fx);
                    self.rep_y[i] = S::from_f64(fy);
                }
            }
        }

        // 3. Dynamic rho(t) weighting, velocity momentum, bounded step, and recentering.
        let rho = self.current_rho() * rep_scale;
        let step_lr =
            config.learning_rate * (1.0 - (self.iteration as f64) / (self.total_iterations as f64));
        let momentum = config.momentum;
        let max_step = 5.0 * config.alpha.max(1.0) * config.learning_rate.max(1.0);

        let mut mean_x = 0.0_f64;
        let mut mean_y = 0.0_f64;
        {
            let xs = layout.x();
            let ys = layout.y();
            for i in 0..n {
                let xi = xs[i].to_f64();
                let yi = ys[i].to_f64();
                let raw_fx = self.attr_x[i] + rho * self.rep_x[i].to_f64();
                let raw_fy = self.attr_y[i] + rho * self.rep_y[i].to_f64();
                let fx = if raw_fx.is_finite() { raw_fx } else { 0.0 };
                let fy = if raw_fy.is_finite() { raw_fy } else { 0.0 };

                let mut vx = momentum * self.vel_x[i] + step_lr * fx;
                let mut vy = momentum * self.vel_y[i] + step_lr * fy;
                if !vx.is_finite() {
                    vx = 0.0;
                }
                if !vy.is_finite() {
                    vy = 0.0;
                }
                let v_norm = vx.hypot(vy);
                if v_norm > max_step && v_norm > 0.0 {
                    let scale = max_step / v_norm;
                    vx *= scale;
                    vy *= scale;
                }
                self.vel_x[i] = vx;
                self.vel_y[i] = vy;

                let cand_x = xi + vx;
                let cand_y = yi + vy;
                let nx_val = if cand_x.is_finite() { cand_x } else { xi };
                let ny_val = if cand_y.is_finite() { cand_y } else { yi };
                self.next_x[i] = nx_val;
                self.next_y[i] = ny_val;
                mean_x += nx_val;
                mean_y += ny_val;
            }
        }

        let inv_n = 1.0 / (n as f64);
        mean_x *= inv_n;
        mean_y *= inv_n;
        let shift_x = if mean_x.is_finite() { mean_x } else { 0.0 };
        let shift_y = if mean_y.is_finite() { mean_y } else { 0.0 };

        let (xs_mut, ys_mut) = layout.xy_mut();
        for i in 0..n {
            let cx = S::from_f64(self.next_x[i] - shift_x);
            let cy = S::from_f64(self.next_y[i] - shift_y);
            if cx.is_finite() {
                xs_mut[i] = cx;
            }
            if cy.is_finite() {
                ys_mut[i] = cy;
            }
        }
        Ok(())
    }

    fn step_ib_umap(
        &mut self,
        layout: &mut Layout2d<S>,
        config: IbUmapConfig,
        a: f64,
        b: f64,
    ) -> CoreResult<()> {
        let n = self.num_nodes;
        if self.iteration == 0 {
            let (xs, ys) = layout.xy_mut();
            if xs.iter().all(|&v| v == S::ZERO) && ys.iter().all(|&v| v == S::ZERO) {
                for idx in 0..n {
                    xs[idx] = S::from_f64(self.rng.next_f64() * 20.0 - 10.0);
                    ys[idx] = S::from_f64(self.rng.next_f64() * 20.0 - 10.0);
                }
            }
        }

        // 1. Attractive UMAP forces along fuzzy simplicial set edges.
        let two_ab = 2.0 * a * b;
        {
            let xs = layout.x();
            let ys = layout.y();
            for i in 0..n {
                let xi = xs[i].to_f64();
                let yi = ys[i].to_f64();
                let start = self.indptr[i];
                let end = self.indptr[i + 1];
                let mut fx = 0.0_f64;
                let mut fy = 0.0_f64;
                for pos in start..end {
                    let j = self.indices[pos] as usize;
                    if j == i {
                        continue;
                    }
                    let p_ij = self.edge_weights[pos] as f64;
                    if p_ij <= 0.0 {
                        continue;
                    }
                    let dx = xi - xs[j].to_f64();
                    let dy = yi - ys[j].to_f64();
                    let d2 = (dx * dx + dy * dy).max(MIN_DIST_SQ_CLAMP);
                    let d_2b = d2.powf(b);
                    let d_pow = d_2b / d2;
                    let coeff = -two_ab * p_ij * d_pow / (1.0 + a * d_2b);
                    if coeff.is_finite() {
                        fx += coeff * dx;
                        fy += coeff * dy;
                    }
                }
                self.attr_x[i] = fx;
                self.attr_y[i] = fy;
            }
        }

        // 2. UMAP repulsive field evaluation: 2 b c_K / ((eps + d^2) (1 + a (d^2)^b)).
        let two_b_ck = 2.0 * b * config.c_k;
        match config.backend {
            FieldRepulsionBackend::IbFft => {
                let ibfft = self.ibfft.as_mut().ok_or_else(|| {
                    CoreError::numerical_breakdown("IbFft2d workspace missing for FieldLayout")
                })?;
                ibfft.prepare_xy(&mut self.fft_ws, layout.x(), layout.y())?;
                let soft = ibfft.grid_spec().map_or(0.0, |spec| spec.softening_sq);
                let eps_eff = REPULSIVE_DIST_SQ_EPS + soft;
                ibfft.evaluate_isotropic_kernel(
                    &mut self.fft_ws,
                    |d2| {
                        let d2_pos = d2.max(0.0);
                        two_b_ck / ((eps_eff + d2_pos) * (1.0 + a * d2_pos.powf(b)))
                    },
                    &mut self.phi_1,
                    &mut self.rep_x,
                    &mut self.rep_y,
                )?;
            }
            FieldRepulsionBackend::BarnesHut => {
                self.field_quadtree.build_xy(layout.x(), layout.y())?;
                self.field_quadtree.compute_isotropic_repulsion(
                    config.theta,
                    |d2| {
                        let d2_pos = d2.max(0.0);
                        two_b_ck / ((REPULSIVE_DIST_SQ_EPS + d2_pos) * (1.0 + a * d2_pos.powf(b)))
                    },
                    &mut self.rep_x,
                    &mut self.rep_y,
                );
            }
            FieldRepulsionBackend::Exact => {
                let xs = layout.x();
                let ys = layout.y();
                for i in 0..n {
                    let xi = xs[i].to_f64();
                    let yi = ys[i].to_f64();
                    let mut fx = 0.0_f64;
                    let mut fy = 0.0_f64;
                    for j in 0..n {
                        if j == i {
                            continue;
                        }
                        let dx = xi - xs[j].to_f64();
                        let dy = yi - ys[j].to_f64();
                        let d2 = dx * dx + dy * dy;
                        let k = two_b_ck / ((REPULSIVE_DIST_SQ_EPS + d2) * (1.0 + a * d2.powf(b)));
                        fx += k * dx;
                        fy += k * dy;
                    }
                    self.rep_x[i] = S::from_f64(fx);
                    self.rep_y[i] = S::from_f64(fy);
                }
            }
        }

        // 3. Hub damping eta_i, per-node step clipping tau_t = tau_scale * alpha_t, and recenter.
        let alpha_t =
            config.learning_rate * (1.0 - (self.iteration as f64) / (self.total_iterations as f64));
        let tau_t = config.tau_scale * alpha_t;
        let inv_n = 1.0 / (n as f64);

        let mut mean_x = 0.0_f64;
        let mut mean_y = 0.0_f64;
        {
            let xs = layout.x();
            let ys = layout.y();
            for i in 0..n {
                let xi = xs[i].to_f64();
                let yi = ys[i].to_f64();
                let raw_fx = self.attr_x[i] + self.rep_x[i].to_f64() * inv_n;
                let raw_fy = self.attr_y[i] + self.rep_y[i].to_f64() * inv_n;
                let fx = if raw_fx.is_finite() { raw_fx } else { 0.0 };
                let fy = if raw_fy.is_finite() { raw_fy } else { 0.0 };

                let step_scale = alpha_t * self.hub_eta[i];
                let mut dx = step_scale * fx;
                let mut dy = step_scale * fy;
                if !dx.is_finite() {
                    dx = 0.0;
                }
                if !dy.is_finite() {
                    dy = 0.0;
                }
                let norm = dx.hypot(dy);
                if norm > tau_t && norm > 0.0 {
                    let clip = tau_t / norm;
                    dx *= clip;
                    dy *= clip;
                }

                let cand_x = xi + dx;
                let cand_y = yi + dy;
                let nx_val = if cand_x.is_finite() { cand_x } else { xi };
                let ny_val = if cand_y.is_finite() { cand_y } else { yi };
                self.next_x[i] = nx_val;
                self.next_y[i] = ny_val;
                mean_x += nx_val;
                mean_y += ny_val;
            }
        }

        mean_x *= inv_n;
        mean_y *= inv_n;
        let shift_x = if mean_x.is_finite() { mean_x } else { 0.0 };
        let shift_y = if mean_y.is_finite() { mean_y } else { 0.0 };

        let (xs_mut, ys_mut) = layout.xy_mut();
        for i in 0..n {
            let cx = S::from_f64(self.next_x[i] - shift_x);
            let cy = S::from_f64(self.next_y[i] - shift_y);
            if cx.is_finite() {
                xs_mut[i] = cx;
            }
            if cy.is_finite() {
                ys_mut[i] = cy;
            }
        }
        Ok(())
    }
}

impl<S: Scalar> IterativeSolver<S> for FieldLayoutSolver<S> {
    fn step(&mut self, layout: &mut Layout2d<S>) -> CoreResult<SolverStatus> {
        if layout.len() != self.num_nodes {
            return Err(CoreError::dimension_mismatch(
                self.num_nodes,
                layout.len(),
                "FieldLayoutSolver::step",
            ));
        }
        layout.validate_finite()?;

        if self.status.is_finished() {
            return Ok(self.status);
        }

        if self.num_nodes < 2 {
            self.status = SolverStatus::Converged;
            return Ok(self.status);
        }

        match self.mode {
            FieldSolverMode::TFdp { config, rep_scale } => {
                self.step_t_fdp(layout, config, rep_scale)?;
            }
            FieldSolverMode::IbUmap { config, a, b } => {
                self.step_ib_umap(layout, config, a, b)?;
            }
        }

        self.iteration += 1;
        self.status = if self.iteration >= self.total_iterations {
            SolverStatus::Completed
        } else {
            SolverStatus::Running
        };
        Ok(self.status)
    }

    #[inline]
    fn status(&self) -> SolverStatus {
        self.status
    }

    #[inline]
    fn iteration(&self) -> usize {
        self.iteration
    }

    #[inline]
    fn total_iterations(&self) -> usize {
        self.total_iterations
    }
}

fn extract_symmetric_csr_edges<W: Scalar>(graph: &CsrGraph<W>) -> (Vec<usize>, Vec<u32>, Vec<f32>) {
    let n = graph.num_nodes();
    if n == 0 {
        return (vec![0], Vec::new(), Vec::new());
    }
    let mut pairs: Vec<(u32, u32, f32)> = Vec::with_capacity(if graph.is_directed() {
        graph.nnz() * 2
    } else {
        graph.nnz()
    });
    for u in 0..n {
        let u_u32 = u as u32;
        let nbrs = graph.neighbors(u);
        let w_opt = graph.neighbor_weights(u);
        for (idx, &v) in nbrs.iter().enumerate() {
            let v_usize = v as usize;
            if v_usize == u || v_usize >= n {
                continue;
            }
            let w = w_opt.map_or(1.0, |ws| ws[idx].to_f64());
            if w <= 0.0 {
                continue;
            }
            let w_f32 = w as f32;
            pairs.push((u_u32, v, w_f32));
            if graph.is_directed() {
                pairs.push((v, u_u32, w_f32));
            }
        }
    }
    symmetrize_pair_list(n, pairs)
}

fn symmetrize_pair_list(
    n: usize,
    mut pairs: Vec<(u32, u32, f32)>,
) -> (Vec<usize>, Vec<u32>, Vec<f32>) {
    if pairs.is_empty() {
        return (vec![0; n + 1], Vec::new(), Vec::new());
    }
    pairs.sort_unstable_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.cmp(&b.1)));

    let mut deg = vec![0usize; n];
    let mut write_idx = 0usize;
    let mut pos = 0usize;
    let len = pairs.len();
    while pos < len {
        let (u, v, first_w) = pairs[pos];
        let mut sum_w = first_w as f64;
        let mut count = 1usize;
        pos += 1;
        while pos < len && pairs[pos].0 == u && pairs[pos].1 == v {
            sum_w += pairs[pos].2 as f64;
            count += 1;
            pos += 1;
        }
        let avg_w = (sum_w / (count as f64)) as f32;
        if avg_w > 0.0 {
            pairs[write_idx] = (u, v, avg_w);
            write_idx += 1;
            deg[u as usize] += 1;
        }
    }
    pairs.truncate(write_idx);

    let mut indptr = Vec::with_capacity(n + 1);
    indptr.push(0);
    let mut total = 0usize;
    for &d in &deg {
        total += d;
        indptr.push(total);
    }
    drop(deg);
    let mut cursor = indptr[..n].to_vec();
    let mut indices = vec![0u32; total];
    let mut weights = vec![0.0_f32; total];
    for (u, v, w) in pairs {
        let p = cursor[u as usize];
        indices[p] = v;
        weights[p] = w;
        cursor[u as usize] += 1;
    }
    (indptr, indices, weights)
}

fn validate_sparse_probabilities(probs: &SparseProbabilities) -> CoreResult<()> {
    let n = probs.n_nodes;
    if u32::try_from(n).is_err() {
        return Err(CoreError::invalid_csr("n_nodes exceeds u32::MAX"));
    }
    if probs.indptr.len() != n + 1 {
        return Err(CoreError::dimension_mismatch(
            n + 1,
            probs.indptr.len(),
            "SparseProbabilities::indptr",
        ));
    }
    if probs.indptr[0] != 0 {
        return Err(CoreError::invalid_csr(format!(
            "SparseProbabilities indptr[0] must be 0, got {}",
            probs.indptr[0]
        )));
    }
    for row in 0..n {
        if probs.indptr[row + 1] < probs.indptr[row] {
            return Err(CoreError::invalid_csr(format!(
                "SparseProbabilities indptr must be non-decreasing at row {row}: {} > {}",
                probs.indptr[row],
                probs.indptr[row + 1]
            )));
        }
    }
    let nnz = probs.indptr[n];
    if probs.indices.len() != nnz {
        return Err(CoreError::dimension_mismatch(
            nnz,
            probs.indices.len(),
            "SparseProbabilities::indices",
        ));
    }
    if probs.values.len() != nnz {
        return Err(CoreError::dimension_mismatch(
            nnz,
            probs.values.len(),
            "SparseProbabilities::values",
        ));
    }
    for (&idx, &val) in probs.indices.iter().zip(probs.values.iter()) {
        if (idx as usize) >= n {
            return Err(CoreError::index_out_of_bounds(
                idx as usize,
                n,
                "SparseProbabilities::indices",
            ));
        }
        if !val.is_finite() {
            return Err(CoreError::non_finite("SparseProbabilities::values"));
        }
        if val < 0.0 {
            return Err(CoreError::invalid_input(format!(
                "SparseProbabilities values must be >= 0.0, got {val}"
            )));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn grid_graph(rows: usize, cols: usize) -> CsrGraph<f64> {
        let n = rows * cols;
        let mut edges = Vec::new();
        for r in 0..rows {
            for c in 0..cols {
                let u = (r * cols + c) as u32;
                if c + 1 < cols {
                    edges.push((u, u + 1));
                }
                if r + 1 < rows {
                    edges.push((u, u + cols as u32));
                }
            }
        }
        CsrGraph::from_edges(n, &edges, None, false, false).unwrap()
    }

    fn barbell_graph(clique_size: usize) -> CsrGraph<f64> {
        let n = 2 * clique_size;
        let mut edges = Vec::new();
        for c in 0..2 {
            let off = (c * clique_size) as u32;
            for i in 0..clique_size as u32 {
                for j in (i + 1)..clique_size as u32 {
                    edges.push((off + i, off + j));
                }
            }
        }
        edges.push((0, clique_size as u32));
        CsrGraph::from_edges(n, &edges, None, false, false).unwrap()
    }

    fn star_graph(num_leaves: usize) -> CsrGraph<f64> {
        let n = 1 + num_leaves;
        let edges: Vec<(u32, u32)> = (1..=num_leaves as u32).map(|v| (0, v)).collect();
        CsrGraph::from_edges(n, &edges, None, false, false).unwrap()
    }

    fn seeded_layout<S: Scalar>(n: usize, scale: f64, seed: u64) -> Layout2d<S> {
        let mut rng = DeterministicRng::from_seed_u64(seed);
        let mut x = Vec::with_capacity(n);
        let mut y = Vec::with_capacity(n);
        for _ in 0..n {
            x.push(S::from_f64((rng.next_f64() * 2.0 - 1.0) * scale));
            y.push(S::from_f64((rng.next_f64() * 2.0 - 1.0) * scale));
        }
        Layout2d::from_xy(x, y).unwrap()
    }

    #[test]
    fn test_rho_schedule_evaluation_and_validation() {
        let sched = RhoSchedule::default();
        assert_eq!(sched.start_weight, 1.0);
        assert_eq!(sched.end_weight, 0.1);
        assert_eq!(sched.power, 1.0);
        sched.validate().unwrap();

        assert!((sched.evaluate(-0.5) - 1.0).abs() < 1e-12);
        assert!((sched.evaluate(0.0) - 1.0).abs() < 1e-12);
        assert!((sched.evaluate(0.5) - 0.55).abs() < 1e-12);
        assert!((sched.evaluate(1.0) - 0.1).abs() < 1e-12);
        assert!((sched.evaluate(2.0) - 0.1).abs() < 1e-12);
        assert!((sched.evaluate(f64::NAN) - 1.0).abs() < 1e-12);

        let quad = RhoSchedule {
            start_weight: 2.0,
            end_weight: 0.2,
            power: 2.0,
        };
        quad.validate().unwrap();
        assert!((quad.evaluate(0.5) - (2.0 + (0.2 - 2.0) * 0.25)).abs() < 1e-12);

        let bad_cases = [
            RhoSchedule {
                start_weight: -0.1,
                ..sched
            },
            RhoSchedule {
                start_weight: f64::NAN,
                ..sched
            },
            RhoSchedule {
                end_weight: -0.1,
                ..sched
            },
            RhoSchedule {
                end_weight: f64::INFINITY,
                ..sched
            },
            RhoSchedule {
                power: 0.0,
                ..sched
            },
            RhoSchedule {
                power: -1.0,
                ..sched
            },
            RhoSchedule {
                power: f64::NAN,
                ..sched
            },
        ];
        for bad in bad_cases {
            assert!(bad.validate().is_err());
        }
    }

    #[test]
    fn test_t_fdp_step_vs_run_bit_exact_f64_and_f32_all_backends() {
        let grid = grid_graph(4, 4);
        for backend in [
            FieldRepulsionBackend::IbFft,
            FieldRepulsionBackend::BarnesHut,
            FieldRepulsionBackend::Exact,
        ] {
            let cfg = TFdpConfig {
                backend,
                total_iterations: 20,
                seed: 77,
                ..Default::default()
            };

            // f64 bit-exact parity
            let mut s_run = FieldLayout::<f64>::t_fdp(&grid, cfg).unwrap();
            let mut s_step = FieldLayout::<f64>::t_fdp(&grid, cfg).unwrap();
            let mut l_run = seeded_layout::<f64>(grid.num_nodes(), 1.5, 19);
            let mut l_step = l_run.clone();

            assert_eq!(s_run.status(), SolverStatus::Initialized);
            assert_eq!(s_run.num_nodes(), 16);
            assert_eq!(s_run.total_iterations(), 20);
            assert!((s_run.current_rho() - 1.0).abs() < 1e-12);

            assert_eq!(s_run.run(&mut l_run).unwrap(), SolverStatus::Completed);
            assert!((s_run.current_rho() - 0.1).abs() < 1e-12);

            assert_eq!(
                s_step.run_steps(&mut l_step, 6).unwrap(),
                SolverStatus::Running
            );
            assert_eq!(s_step.iteration(), 6);
            while !s_step.status().is_finished() {
                s_step.step(&mut l_step).unwrap();
            }
            assert_eq!(s_step.status(), SolverStatus::Completed);
            assert_eq!(s_step.iteration(), 20);

            for i in 0..grid.num_nodes() {
                assert_eq!(l_run.x()[i].to_bits(), l_step.x()[i].to_bits());
                assert_eq!(l_run.y()[i].to_bits(), l_step.y()[i].to_bits());
            }

            // f32 bit-exact parity
            let mut s32_run = FieldLayoutSolver::<f32>::t_fdp(&grid, cfg).unwrap();
            let mut s32_step = FieldLayoutSolver::<f32>::t_fdp(&grid, cfg).unwrap();
            let mut l32_run = seeded_layout::<f32>(grid.num_nodes(), 1.5, 19);
            let mut l32_step = l32_run.clone();
            s32_run.run(&mut l32_run).unwrap();
            for _ in 0..20 {
                s32_step.step(&mut l32_step).unwrap();
            }
            for i in 0..grid.num_nodes() {
                assert_eq!(l32_run.x()[i].to_bits(), l32_step.x()[i].to_bits());
                assert_eq!(l32_run.y()[i].to_bits(), l32_step.y()[i].to_bits());
            }
        }
    }

    #[test]
    fn test_ib_umap_step_vs_run_bit_exact_f64_and_f32_all_backends() {
        let grid = grid_graph(4, 4);
        let dog_cfg = DogVaradhanConfig::default();
        for backend in [
            FieldRepulsionBackend::IbFft,
            FieldRepulsionBackend::BarnesHut,
            FieldRepulsionBackend::Exact,
        ] {
            let cfg = IbUmapConfig {
                backend,
                n_epochs: 20,
                seed: 55,
                ..Default::default()
            };

            let mut s_run = FieldLayoutSolver::<f64>::ib_umap(&grid, 32, &dog_cfg, cfg).unwrap();
            let mut s_step = FieldLayoutSolver::<f64>::ib_umap(&grid, 32, &dog_cfg, cfg).unwrap();
            let mut l_run = seeded_layout::<f64>(grid.num_nodes(), 2.0, 23);
            let mut l_step = l_run.clone();

            assert_eq!(s_run.run(&mut l_run).unwrap(), SolverStatus::Completed);
            assert_eq!(
                s_step.run_steps(&mut l_step, 5).unwrap(),
                SolverStatus::Running
            );
            while !s_step.status().is_finished() {
                s_step.step(&mut l_step).unwrap();
            }
            for i in 0..grid.num_nodes() {
                assert_eq!(l_run.x()[i].to_bits(), l_step.x()[i].to_bits());
                assert_eq!(l_run.y()[i].to_bits(), l_step.y()[i].to_bits());
            }

            let mut s32_run = FieldLayoutSolver::<f32>::ib_umap(&grid, 32, &dog_cfg, cfg).unwrap();
            let mut s32_step = FieldLayoutSolver::<f32>::ib_umap(&grid, 32, &dog_cfg, cfg).unwrap();
            let mut l32_run = seeded_layout::<f32>(grid.num_nodes(), 2.0, 23);
            let mut l32_step = l32_run.clone();
            s32_run.run(&mut l32_run).unwrap();
            for _ in 0..20 {
                s32_step.step(&mut l32_step).unwrap();
            }
            for i in 0..grid.num_nodes() {
                assert_eq!(l32_run.x()[i].to_bits(), l32_step.x()[i].to_bits());
                assert_eq!(l32_run.y()[i].to_bits(), l32_step.y()[i].to_bits());
            }
        }
    }

    #[test]
    fn test_t_fdp_ibfft_vs_exact_and_barnes_hut_consistency() {
        let barbell = barbell_graph(6);
        let init = seeded_layout::<f64>(barbell.num_nodes(), 2.0, 101);

        for gamma in [1.0, 1.5, 2.0] {
            let mut s_exact = FieldLayoutSolver::<f64>::t_fdp(
                &barbell,
                TFdpConfig {
                    backend: FieldRepulsionBackend::Exact,
                    total_iterations: 15,
                    gamma,
                    ..Default::default()
                },
            )
            .unwrap();
            let mut s_ibfft = FieldLayoutSolver::<f64>::t_fdp(
                &barbell,
                TFdpConfig {
                    backend: FieldRepulsionBackend::IbFft,
                    total_iterations: 15,
                    gamma,
                    ..Default::default()
                },
            )
            .unwrap();
            let mut s_bh_exact = FieldLayoutSolver::<f64>::t_fdp(
                &barbell,
                TFdpConfig {
                    backend: FieldRepulsionBackend::BarnesHut,
                    total_iterations: 15,
                    theta: 0.0,
                    gamma,
                    ..Default::default()
                },
            )
            .unwrap();

            let mut l_exact = init.clone();
            let mut l_ibfft = init.clone();
            let mut l_bh = init.clone();
            s_exact.run(&mut l_exact).unwrap();
            s_ibfft.run(&mut l_ibfft).unwrap();
            s_bh_exact.run(&mut l_bh).unwrap();
            assert!(s_ibfft.total_fft_2d_calls() > 0);

            for i in 0..barbell.num_nodes() {
                let bh_err = (l_bh.x()[i] - l_exact.x()[i]).hypot(l_bh.y()[i] - l_exact.y()[i]);
                assert!(
                    bh_err < 1e-10,
                    "gamma={gamma}: BarnesHut(theta=0) vs Exact err={bh_err}"
                );
                let fft_err =
                    (l_ibfft.x()[i] - l_exact.x()[i]).hypot(l_ibfft.y()[i] - l_exact.y()[i]);
                assert!(
                    fft_err < 0.05,
                    "gamma={gamma}: IbFft vs Exact err={fft_err}"
                );
            }
        }
    }

    #[test]
    fn test_ib_umap_ibfft_vs_exact_and_barbell_separation() {
        let barbell = barbell_graph(6);
        let dog_cfg = DogVaradhanConfig::default();
        let init = seeded_layout::<f64>(barbell.num_nodes(), 2.5, 202);

        let mut s_exact = FieldLayoutSolver::<f64>::ib_umap(
            &barbell,
            32,
            &dog_cfg,
            IbUmapConfig {
                backend: FieldRepulsionBackend::Exact,
                n_epochs: 25,
                ..Default::default()
            },
        )
        .unwrap();
        let mut s_ibfft = FieldLayoutSolver::<f64>::ib_umap(
            &barbell,
            32,
            &dog_cfg,
            IbUmapConfig {
                backend: FieldRepulsionBackend::IbFft,
                n_epochs: 25,
                ..Default::default()
            },
        )
        .unwrap();
        let mut s_bh_exact = FieldLayoutSolver::<f64>::ib_umap(
            &barbell,
            32,
            &dog_cfg,
            IbUmapConfig {
                backend: FieldRepulsionBackend::BarnesHut,
                n_epochs: 25,
                theta: 0.0,
                ..Default::default()
            },
        )
        .unwrap();

        let mut l_exact = init.clone();
        let mut l_ibfft = init.clone();
        let mut l_bh = init.clone();
        s_exact.run(&mut l_exact).unwrap();
        s_ibfft.run(&mut l_ibfft).unwrap();
        s_bh_exact.run(&mut l_bh).unwrap();

        for i in 0..barbell.num_nodes() {
            let bh_err = (l_bh.x()[i] - l_exact.x()[i]).hypot(l_bh.y()[i] - l_exact.y()[i]);
            assert!(bh_err < 1e-10, "ibUMAP BH(theta=0) err={bh_err}");
            let fft_err = (l_ibfft.x()[i] - l_exact.x()[i]).hypot(l_ibfft.y()[i] - l_exact.y()[i]);
            assert!(fft_err < 0.08, "ibUMAP IbFft vs Exact err={fft_err}");
        }

        // Full 150-epoch run from all-zeros separates the two barbell cliques cleanly.
        let mut full_solver = FieldLayoutSolver::<f64>::ib_umap(
            &barbell,
            32,
            &dog_cfg,
            IbUmapConfig {
                n_epochs: 150,
                ..Default::default()
            },
        )
        .unwrap();
        let mut layout = Layout2d::<f64>::zeros(barbell.num_nodes());
        assert_eq!(
            full_solver.run(&mut layout).unwrap(),
            SolverStatus::Completed
        );
        assert!(layout.is_all_finite());

        let c0x = layout.x()[..6].iter().sum::<f64>() / 6.0;
        let c0y = layout.y()[..6].iter().sum::<f64>() / 6.0;
        let c1x = layout.x()[6..].iter().sum::<f64>() / 6.0;
        let c1y = layout.y()[6..].iter().sum::<f64>() / 6.0;
        let centroid_dist = (c0x - c1x).hypot(c0y - c1y);
        let mut intra = 0.0_f64;
        for i in 0..6 {
            intra += (layout.x()[i] - c0x).hypot(layout.y()[i] - c0y);
            intra += (layout.x()[6 + i] - c1x).hypot(layout.y()[6 + i] - c1y);
        }
        intra /= 12.0;
        assert!(
            centroid_dist > intra * 1.5,
            "expected barbell separation: centroid_dist={centroid_dist}, intra={intra}"
        );
    }

    #[test]
    fn test_ib_umap_hub_damping_and_tau_clipping() {
        let star = star_graph(24);
        let probs = SparseProbabilities {
            n_nodes: star.num_nodes(),
            indptr: {
                let mut p = Vec::with_capacity(star.num_nodes() + 1);
                p.push(0);
                p.push(24);
                for i in 1..star.num_nodes() {
                    p.push(24 + i);
                }
                p
            },
            indices: {
                let mut idx: Vec<u32> = (1..star.num_nodes() as u32).collect();
                idx.resize(48, 0);
                idx
            },
            values: vec![1.0; 48],
        };

        // Place hub far away at (100, 0) and leaves around (0, 0) to trigger large hub attraction
        // and verify both hub_damping eta_i and tau_t = tau_scale * alpha_t step clipping.
        let mut xs = vec![0.0_f64; star.num_nodes()];
        let mut ys = vec![0.0_f64; star.num_nodes()];
        xs[0] = 100.0;
        for i in 1..star.num_nodes() {
            let angle = std::f64::consts::TAU * ((i - 1) as f64) / 24.0;
            xs[i] = angle.cos();
            ys[i] = angle.sin();
        }
        let init = Layout2d::from_xy(xs, ys).unwrap();

        let cfg_damped = IbUmapConfig {
            backend: FieldRepulsionBackend::Exact,
            n_epochs: 10,
            learning_rate: 1.0,
            tau_scale: 4.0,
            hub_damping: true,
            ..Default::default()
        };
        let cfg_undamped = IbUmapConfig {
            hub_damping: false,
            ..cfg_damped
        };
        let cfg_tight_tau = IbUmapConfig {
            tau_scale: 0.25,
            hub_damping: false,
            ..cfg_damped
        };

        let s_damped = FieldLayoutSolver::<f64>::ib_umap_from_fuzzy_set(
            probs.clone(),
            Some(&[24; 25]), // explicit degrees: even stronger damping test below uses natural degrees
            cfg_damped,
        )
        .unwrap();
        assert_eq!(s_damped.num_nodes(), 25);
        assert_eq!(s_damped.num_edges(), 48);
        assert!(s_damped.ab().is_some());
        assert_eq!(s_damped.backend(), FieldRepulsionBackend::Exact);
        assert!((s_damped.current_rho() - 1.0).abs() < 1e-12);

        // Natural degrees from fuzzy set: hub deg = 24, leaf deg = 1.
        let mut s_nat_damped =
            FieldLayoutSolver::<f64>::ib_umap_from_fuzzy_set(probs.clone(), None, cfg_damped)
                .unwrap();
        let mut s_undamped =
            FieldLayoutSolver::<f64>::ib_umap_from_fuzzy_set(probs.clone(), None, cfg_undamped)
                .unwrap();
        let mut s_tight =
            FieldLayoutSolver::<f64>::ib_umap_from_fuzzy_set(probs, None, cfg_tight_tau).unwrap();

        let mut l_damped = init.clone();
        let mut l_undamped = init.clone();
        let mut l_tight = init.clone();

        s_nat_damped.step(&mut l_damped).unwrap();
        s_undamped.step(&mut l_undamped).unwrap();
        s_tight.step(&mut l_tight).unwrap();

        // Relative displacement of hub vs centroid shift must be strictly smaller with hub damping.
        let mean_x_before = init.x().iter().sum::<f64>() / (star.num_nodes() as f64);
        let hub_rel_before = init.x()[0] - mean_x_before;
        let hub_rel_damped = l_damped.x()[0]; // post-step layouts are zero-mean centered
        let hub_rel_undamped = l_undamped.x()[0];
        let hub_rel_tight = l_tight.x()[0];

        let damped_hub_delta = (hub_rel_before - hub_rel_damped).abs();
        let undamped_hub_delta = (hub_rel_before - hub_rel_undamped).abs();
        let tight_hub_delta = (hub_rel_before - hub_rel_tight).abs();

        assert!(
            damped_hub_delta < undamped_hub_delta,
            "expected hub damping to reduce step: damped={damped_hub_delta}, undamped={undamped_hub_delta}"
        );
        assert!(
            tight_hub_delta < undamped_hub_delta,
            "expected tighter tau_scale=0.25 to clip step: tight={tight_hub_delta}, undamped={undamped_hub_delta}"
        );
    }

    #[test]
    fn test_t_fdp_and_ib_umap_from_knn_and_t_fdp_separates_barbell() {
        let barbell = barbell_graph(6);
        let mut s_fdp = FieldLayoutSolver::<f64>::t_fdp(
            &barbell,
            TFdpConfig {
                total_iterations: 150,
                ..Default::default()
            },
        )
        .unwrap();
        let mut l_fdp = Layout2d::<f64>::zeros(barbell.num_nodes());
        assert_eq!(s_fdp.run(&mut l_fdp).unwrap(), SolverStatus::Completed);
        assert!(l_fdp.is_all_finite());

        let c0x = l_fdp.x()[..6].iter().sum::<f64>() / 6.0;
        let c0y = l_fdp.y()[..6].iter().sum::<f64>() / 6.0;
        let c1x = l_fdp.x()[6..].iter().sum::<f64>() / 6.0;
        let c1y = l_fdp.y()[6..].iter().sum::<f64>() / 6.0;
        let centroid_dist = (c0x - c1x).hypot(c0y - c1y);
        let mut intra = 0.0_f64;
        for i in 0..6 {
            intra += (l_fdp.x()[i] - c0x).hypot(l_fdp.y()[i] - c0y);
            intra += (l_fdp.x()[6 + i] - c1x).hypot(l_fdp.y()[6 + i] - c1y);
        }
        intra /= 12.0;
        assert!(
            centroid_dist > intra * 1.5,
            "t-FDP should separate barbell clusters: centroid_dist={centroid_dist}, intra={intra}"
        );

        let knn = KnnGraph::new_validated(
            4,
            vec![0, 2, 4, 6, 8],
            vec![1, 3, 0, 2, 1, 3, 0, 2],
            vec![0.5, 0.8, 0.5, 0.5, 0.5, 0.8, 0.8, 0.8],
        )
        .unwrap();
        let mut s_knn_fdp = FieldLayoutSolver::<f64>::t_fdp_from_knn(
            &knn,
            TFdpConfig {
                total_iterations: 30,
                ..Default::default()
            },
        )
        .unwrap();
        let mut l_kf = seeded_layout::<f64>(4, 1.0, 9);
        assert_eq!(s_knn_fdp.run(&mut l_kf).unwrap(), SolverStatus::Completed);
        assert!(l_kf.is_all_finite());

        let mut s_knn_umap = FieldLayoutSolver::<f64>::ib_umap_from_knn(
            &knn,
            64,
            1e-5,
            IbUmapConfig {
                n_epochs: 30,
                ..Default::default()
            },
        )
        .unwrap();
        let mut l_ku = seeded_layout::<f64>(4, 1.0, 9);
        assert_eq!(s_knn_umap.run(&mut l_ku).unwrap(), SolverStatus::Completed);
        assert!(l_ku.is_all_finite());
    }

    #[test]
    fn test_field_layout_edge_cases_and_error_guards() {
        let tfdp_cfg = TFdpConfig::default();
        let ib_cfg = IbUmapConfig::default();
        let dog_cfg = DogVaradhanConfig::default();

        // Empty and singleton graphs for both t-FDP and ibUMAP
        let empty = CsrGraph::<f64>::from_edges(0, &[], None, false, false).unwrap();
        let mut se_fdp = FieldLayoutSolver::<f64>::t_fdp(&empty, tfdp_cfg).unwrap();
        let mut se_umap = FieldLayoutSolver::<f64>::ib_umap(&empty, 16, &dog_cfg, ib_cfg).unwrap();
        let mut le = Layout2d::<f64>::zeros(0);
        assert_eq!(se_fdp.run(&mut le).unwrap(), SolverStatus::Converged);
        assert_eq!(se_umap.run(&mut le).unwrap(), SolverStatus::Converged);

        let single = CsrGraph::<f64>::from_edges(1, &[], None, false, false).unwrap();
        let mut ss_fdp = FieldLayoutSolver::<f64>::t_fdp(&single, tfdp_cfg).unwrap();
        let mut ss_umap = FieldLayoutSolver::<f64>::ib_umap(&single, 16, &dog_cfg, ib_cfg).unwrap();
        let mut ls = Layout2d::from_xy(vec![3.5], vec![-2.5]).unwrap();
        assert_eq!(ss_fdp.run(&mut ls).unwrap(), SolverStatus::Converged);
        assert_eq!(ss_umap.run(&mut ls).unwrap(), SolverStatus::Converged);
        assert_eq!(ls.x(), &[3.5]);
        assert_eq!(ls.y(), &[-2.5]);

        // Disconnected graph with isolated vertices runs cleanly
        let disc = CsrGraph::<f64>::from_edges(6, &[(0, 1), (3, 4)], None, false, false).unwrap();
        let mut sd_fdp = FieldLayoutSolver::<f64>::t_fdp(
            &disc,
            TFdpConfig {
                total_iterations: 15,
                ..tfdp_cfg
            },
        )
        .unwrap();
        let mut sd_umap = FieldLayoutSolver::<f64>::ib_umap(
            &disc,
            16,
            &dog_cfg,
            IbUmapConfig {
                n_epochs: 15,
                ..ib_cfg
            },
        )
        .unwrap();
        let mut ld1 = seeded_layout::<f64>(6, 1.0, 31);
        let mut ld2 = ld1.clone();
        assert_eq!(sd_fdp.run(&mut ld1).unwrap(), SolverStatus::Completed);
        assert_eq!(sd_umap.run(&mut ld2).unwrap(), SolverStatus::Completed);
        assert!(ld1.is_all_finite() && ld2.is_all_finite());

        // Zero-iteration solvers complete immediately
        let zero_fdp = FieldLayoutSolver::<f64>::t_fdp(
            &disc,
            TFdpConfig {
                total_iterations: 0,
                ..tfdp_cfg
            },
        )
        .unwrap();
        assert_eq!(zero_fdp.status(), SolverStatus::Completed);
        assert_eq!(zero_fdp.total_iterations(), 0);
        let zero_umap = FieldLayoutSolver::<f64>::ib_umap(
            &disc,
            16,
            &dog_cfg,
            IbUmapConfig {
                n_epochs: 0,
                ..ib_cfg
            },
        )
        .unwrap();
        assert_eq!(zero_umap.status(), SolverStatus::Completed);
        assert_eq!(zero_umap.total_iterations(), 0);

        // Dimension mismatch & NaN/Inf layout guards
        let mut l_wrong = Layout2d::<f64>::zeros(4);
        assert!(matches!(
            sd_fdp.step(&mut l_wrong),
            Err(CoreError::DimensionMismatch { .. })
        ));
        let mut l_nan = Layout2d::from_xy(vec![0.0; 6], vec![f64::NAN; 6]).unwrap();
        assert!(matches!(
            sd_umap.step(&mut l_nan),
            Err(CoreError::NonFiniteValue { .. })
        ));

        // Negative or NaN config & weight guards
        let bad_tfdp_configs = [
            TFdpConfig {
                learning_rate: 0.0,
                ..tfdp_cfg
            },
            TFdpConfig {
                learning_rate: f64::NAN,
                ..tfdp_cfg
            },
            TFdpConfig {
                momentum: 1.0,
                ..tfdp_cfg
            },
            TFdpConfig {
                alpha: 0.0,
                ..tfdp_cfg
            },
            TFdpConfig {
                gamma: -0.5,
                ..tfdp_cfg
            },
            TFdpConfig {
                theta: -0.1,
                ..tfdp_cfg
            },
            TFdpConfig {
                rho: RhoSchedule {
                    power: 0.0,
                    ..RhoSchedule::default()
                },
                ..tfdp_cfg
            },
        ];
        for bad in bad_tfdp_configs {
            assert!(FieldLayoutSolver::<f64>::t_fdp(&disc, bad).is_err());
        }

        let bad_ib_configs = [
            IbUmapConfig {
                learning_rate: f64::NAN,
                ..ib_cfg
            },
            IbUmapConfig {
                min_dist: -0.1,
                ..ib_cfg
            },
            IbUmapConfig {
                spread: 0.0,
                ..ib_cfg
            },
            IbUmapConfig {
                c_k: -1.0,
                ..ib_cfg
            },
            IbUmapConfig {
                tau_scale: 0.0,
                ..ib_cfg
            },
            IbUmapConfig {
                theta: f64::INFINITY,
                ..ib_cfg
            },
            IbUmapConfig {
                a: Some(0.0),
                ..ib_cfg
            },
            IbUmapConfig {
                b: Some(f64::NAN),
                ..ib_cfg
            },
        ];
        for bad in bad_ib_configs {
            assert!(FieldLayoutSolver::<f64>::ib_umap(&disc, 16, &dog_cfg, bad).is_err());
        }

        // Degree slice length mismatch in ib_umap_from_fuzzy_set
        let probs = SparseProbabilities::empty(3);
        assert!(matches!(
            FieldLayoutSolver::<f64>::ib_umap_from_fuzzy_set(probs, Some(&[1, 2]), ib_cfg),
            Err(CoreError::DimensionMismatch { .. })
        ));

        // Negative edge weight in t_fdp
        let neg_w_graph =
            CsrGraph::<f64>::from_edges(2, &[(0, 1)], Some(&[-1.0]), false, false).unwrap();
        assert!(matches!(
            FieldLayoutSolver::<f64>::t_fdp(&neg_w_graph, tfdp_cfg),
            Err(CoreError::InvalidInput(_))
        ));
    }

    #[test]
    fn test_zero_per_step_allocations_directed_graph_and_custom_ab_alpha() {
        assert!(!std::mem::needs_drop::<FieldSolverMode>());
        // Directed weighted graph is cleanly symmetrized in t_fdp and supported in ib_umap.
        let mut dir_edges = Vec::new();
        for i in 0..24u32 {
            dir_edges.push((i, (i + 1) % 24));
            if i % 3 == 0 {
                dir_edges.push((i, (i + 6) % 24));
            }
        }
        let dir_weights = vec![1.5_f64; dir_edges.len()];
        let dir_graph =
            CsrGraph::<f64>::from_edges(24, &dir_edges, Some(&dir_weights), true, false).unwrap();

        let mut s_fdp = FieldLayoutSolver::<f64>::t_fdp(
            &dir_graph,
            TFdpConfig {
                backend: FieldRepulsionBackend::IbFft,
                total_iterations: 20,
                alpha: 2.0,
                gamma: 1.0,
                ..Default::default()
            },
        )
        .unwrap();
        assert!(s_fdp.ab().is_none());
        let mut l_fdp = seeded_layout::<f64>(24, 2.0, 11);
        s_fdp.step(&mut l_fdp).unwrap();

        // Warmed-up solver buffers must retain capacities with zero reallocations on subsequent steps.
        let attr_cap = s_fdp.attr_x.capacity();
        let vel_cap = s_fdp.vel_x.capacity();
        let next_cap = s_fdp.next_x.capacity();
        let rep_cap = s_fdp.rep_x.capacity();
        let phi_cap = s_fdp.phi_1.capacity();
        s_fdp.run(&mut l_fdp).unwrap();
        assert_eq!(s_fdp.attr_x.capacity(), attr_cap);
        assert_eq!(s_fdp.vel_x.capacity(), vel_cap);
        assert_eq!(s_fdp.next_x.capacity(), next_cap);
        assert_eq!(s_fdp.rep_x.capacity(), rep_cap);
        assert_eq!(s_fdp.phi_1.capacity(), phi_cap);
        assert!(l_fdp.is_all_finite());

        let mut s_umap = FieldLayoutSolver::<f64>::ib_umap(
            &dir_graph,
            16,
            &DogVaradhanConfig::default(),
            IbUmapConfig {
                backend: FieldRepulsionBackend::BarnesHut,
                n_epochs: 15,
                a: Some(1.6),
                b: Some(0.9),
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(s_umap.ab(), Some((1.6, 0.9)));
        let mut l_umap = seeded_layout::<f64>(24, 2.0, 11);
        s_umap.step(&mut l_umap).unwrap();
        let qt_node_cap = s_umap.field_quadtree.nodes.capacity();
        s_umap.run(&mut l_umap).unwrap();
        assert!(s_umap.field_quadtree.nodes.capacity() >= qt_node_cap);
        assert!(l_umap.is_all_finite());
    }

    #[test]
    fn test_ib_umap_from_landmarks_bit_exact_and_validation() {
        let g = grid_graph(4, 4);
        let n = g.num_nodes();
        let dog_cfg = DogVaradhanConfig::default();
        let cfg = IbUmapConfig {
            n_epochs: 15,
            seed: 67,
            ..Default::default()
        };
        let (landmarks, _) = compute_regularized_dog_varadhan_landmarks(&g, &dog_cfg).unwrap();

        let mut s_dog = FieldLayoutSolver::<f64>::ib_umap(&g, 32, &dog_cfg, cfg).unwrap();
        let mut s_lm =
            FieldLayoutSolver::<f64>::ib_umap_from_landmarks(&g, &landmarks, 32, cfg).unwrap();
        let mut l_dog = seeded_layout::<f64>(n, 1.0, 23);
        let mut l_lm = l_dog.clone();
        s_dog.run(&mut l_dog).unwrap();
        s_lm.run(&mut l_lm).unwrap();
        for i in 0..n {
            assert_eq!(l_dog.x()[i].to_bits(), l_lm.x()[i].to_bits());
            assert_eq!(l_dog.y()[i].to_bits(), l_lm.y()[i].to_bits());
        }

        // Validation errors
        let mut bad_nodes = landmarks.clone();
        bad_nodes.num_nodes = n + 1;
        assert!(matches!(
            FieldLayoutSolver::<f64>::ib_umap_from_landmarks(&g, &bad_nodes, 32, cfg),
            Err(CoreError::DimensionMismatch { .. })
        ));

        let mut bad_empty_lm = landmarks.clone();
        bad_empty_lm.landmarks.clear();
        bad_empty_lm.d_dog.clear();
        bad_empty_lm.d_coarse = None;
        assert!(matches!(
            FieldLayoutSolver::<f64>::ib_umap_from_landmarks(&g, &bad_empty_lm, 32, cfg),
            Err(CoreError::InvalidInput(_))
        ));

        let mut bad_nan = landmarks.clone();
        bad_nan.d_dog[0] = f64::NAN;
        assert!(matches!(
            FieldLayoutSolver::<f64>::ib_umap_from_landmarks(&g, &bad_nan, 32, cfg),
            Err(CoreError::NonFiniteValue { .. })
        ));

        let mut bad_idx = landmarks;
        bad_idx.landmarks[0] = n as u32;
        assert!(matches!(
            FieldLayoutSolver::<f64>::ib_umap_from_landmarks(&g, &bad_idx, 32, cfg),
            Err(CoreError::IndexOutOfBounds { .. })
        ));
    }
}
