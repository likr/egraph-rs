//! Continuous force-directed placement solvers (`ForceAtlas2` and multilevel `Yifan Hu`).

use std::cmp::Reverse;

use egraph_core::{
    CoreError, CoreResult, CsrGraph, DeterministicRng, IterativeSolver, Layout2d, Scalar,
    SolverStatus, VisitStamp,
};
use egraph_spatial::ArenaQuadtree2d;

const COULOMB_EPS_SQ: f64 = 1e-6;
const OVERLAP_DPRIME_MIN: f64 = 1e-6;
const OVERLAP_REPULSION_K: f64 = 100.0;
const MIN_FA2_SPEED: f64 = 1e-4;
const MAX_FA2_DISPLACEMENT: f64 = 50.0;
const MAX_COARSEN_LEVELS: usize = 64;
const PROLONGATION_JITTER_FACTOR: f64 = 0.05;
const MAX_OVERLAP_QUAD_DEPTH: usize = 32;
const OVERLAP_QUAD_COINCIDENT_SQ: f64 = 1e-12;

/// Hyperparameters for [`ForceAtlas2Solver`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ForceAtlas2Config {
    /// Total number of ForceAtlas2 iterations (`default: 200`).
    pub iterations: usize,
    /// Barnes-Hut quadtree opening angle threshold $\theta \ge 0$ (`default: 1.2`).
    pub theta: f64,
    /// Whether to use $O(N \log N)$ Barnes-Hut quadtree repulsion (`default: true`).
    pub use_barnes_hut: bool,
    /// Whether to use LinLog logarithmic attraction $\log(1 + d)$ (`default: false`).
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

/// Hyperparameters for [`YifanHuSolver`].
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

#[derive(Debug, Clone, Copy)]
struct OverlapQuadNode {
    center_x: f64,
    center_y: f64,
    width: f64,
    height: f64,
    count: usize,
    total_mass: f64,
    com_x: f64,
    com_y: f64,
    max_radius: f64,
    avg_radius: f64,
    leaf_head: u32,
    first_child: u32,
}

impl OverlapQuadNode {
    #[inline]
    fn new(center_x: f64, center_y: f64, width: f64, height: f64) -> Self {
        Self {
            center_x,
            center_y,
            width,
            height,
            count: 0,
            total_mass: 0.0,
            com_x: 0.0,
            com_y: 0.0,
            max_radius: 0.0,
            avg_radius: 0.0,
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
        if depth >= MAX_OVERLAP_QUAD_DEPTH {
            return true;
        }
        let qw = self.width * 0.25;
        let qh = self.height * 0.25;
        qw <= 0.0
            || qh <= 0.0
            || (self.center_x - qw) == self.center_x
            || (self.center_x + qw) == self.center_x
            || (self.center_y - qh) == self.center_y
            || (self.center_y + qh) == self.center_y
    }
}

/// Radius-aware flat arena Barnes-Hut quadtree for Jacomy (2014) `prevent_overlap` repulsion.
///
/// Tracks `max_radius` per quadtree cell so multipole approximation is only used when
/// `max_dim < theta * d` AND `d > 2 * (r_i + cell.max_radius)` (strictly well-separated,
/// guaranteeing `d' > 0`), while descending to exact pairwise `d' = d - r_i - r_j` checks
/// for all nearby or overlapping pairs.
#[derive(Debug, Clone, Default)]
struct OverlapQuadtree {
    nodes: Vec<OverlapQuadNode>,
    leaf_of_point: Vec<u32>,
    next_in_leaf: Vec<u32>,
}

impl OverlapQuadtree {
    fn build(&mut self, xs: &[f64], ys: &[f64], masses: &[f64], radii: &[f64]) {
        self.nodes.clear();
        self.leaf_of_point.clear();
        self.next_in_leaf.clear();

        let n = xs.len();
        if n == 0 {
            return;
        }

        let mut min_x = xs[0];
        let mut max_x = xs[0];
        let mut min_y = ys[0];
        let mut max_y = ys[0];
        for i in 1..n {
            min_x = min_x.min(xs[i]);
            max_x = max_x.max(xs[i]);
            min_y = min_y.min(ys[i]);
            max_y = max_y.max(ys[i]);
        }

        self.leaf_of_point.resize(n, u32::MAX);
        self.next_in_leaf.resize(n, u32::MAX);

        let center_x = 0.5 * (min_x + max_x);
        let center_y = 0.5 * (min_y + max_y);
        let width = ((max_x - min_x) * 1.01).max(1e-4);
        let height = ((max_y - min_y) * 1.01).max(1e-4);
        self.nodes
            .push(OverlapQuadNode::new(center_x, center_y, width, height));

        for i in 0..n {
            self.insert(i, xs, ys, masses, radii);
        }
    }

    fn insert(&mut self, point_idx: usize, xs: &[f64], ys: &[f64], masses: &[f64], radii: &[f64]) {
        let x = xs[point_idx];
        let y = ys[point_idx];
        let mass = masses[point_idx];
        let rad = radii[point_idx];
        let mut cur = 0usize;
        let mut depth = 0usize;

        loop {
            let node = self.nodes[cur];
            if node.count == 0 {
                let slot = &mut self.nodes[cur];
                slot.count = 1;
                slot.total_mass = mass;
                slot.com_x = x;
                slot.com_y = y;
                slot.max_radius = rad;
                slot.avg_radius = rad;
                slot.leaf_head = point_idx as u32;
                self.leaf_of_point[point_idx] = cur as u32;
                self.next_in_leaf[point_idx] = u32::MAX;
                return;
            }

            let old_c = node.count as f64;
            let new_c = (node.count + 1) as f64;
            let old_m = node.total_mass;
            let new_m = old_m + mass;
            {
                let slot = &mut self.nodes[cur];
                if new_m > 0.0 {
                    slot.com_x = (slot.com_x * old_m + x * mass) / new_m;
                    slot.com_y = (slot.com_y * old_m + y * mass) / new_m;
                    slot.avg_radius = (slot.avg_radius * old_m + rad * mass) / new_m;
                } else {
                    slot.com_x = (slot.com_x * old_c + x) / new_c;
                    slot.com_y = (slot.com_y * old_c + y) / new_c;
                    slot.avg_radius = (slot.avg_radius * old_c + rad) / new_c;
                }
                slot.max_radius = slot.max_radius.max(rad);
                slot.total_mass = new_m;
                slot.count += 1;
            }

            if node.first_child == u32::MAX {
                let ref_idx = node.leaf_head as usize;
                let rx = xs[ref_idx];
                let ry = ys[ref_idx];
                let dist_sq = (rx - x) * (rx - x) + (ry - y) * (ry - y);
                if !dist_sq.is_finite()
                    || dist_sq < OVERLAP_QUAD_COINCIDENT_SQ
                    || node.is_unsplitable_at_depth(depth)
                {
                    self.next_in_leaf[point_idx] = self.nodes[cur].leaf_head;
                    self.nodes[cur].leaf_head = point_idx as u32;
                    self.leaf_of_point[point_idx] = cur as u32;
                    return;
                }

                let head = self.nodes[cur].leaf_head;
                self.nodes[cur].leaf_head = u32::MAX;
                let base_child = self.subdivide(cur);

                let mut existing = head;
                while existing != u32::MAX {
                    let e_idx = existing as usize;
                    let next = self.next_in_leaf[e_idx];
                    self.next_in_leaf[e_idx] = u32::MAX;
                    let ex = xs[e_idx];
                    let ey = ys[e_idx];
                    let em = masses[e_idx];
                    let er = radii[e_idx];
                    let q = self.nodes[cur].quadrant(ex, ey);
                    let child = base_child + q;
                    let c_node = &mut self.nodes[child];
                    if c_node.count == 0 {
                        c_node.count = 1;
                        c_node.total_mass = em;
                        c_node.com_x = ex;
                        c_node.com_y = ey;
                        c_node.max_radius = er;
                        c_node.avg_radius = er;
                        c_node.leaf_head = e_idx as u32;
                        self.leaf_of_point[e_idx] = child as u32;
                    } else {
                        let oc = c_node.count as f64;
                        let nc = (c_node.count + 1) as f64;
                        let om = c_node.total_mass;
                        let nm = om + em;
                        if nm > 0.0 {
                            c_node.com_x = (c_node.com_x * om + ex * em) / nm;
                            c_node.com_y = (c_node.com_y * om + ey * em) / nm;
                            c_node.avg_radius = (c_node.avg_radius * om + er * em) / nm;
                        } else {
                            c_node.com_x = (c_node.com_x * oc + ex) / nc;
                            c_node.com_y = (c_node.com_y * oc + ey) / nc;
                            c_node.avg_radius = (c_node.avg_radius * oc + er) / nc;
                        }
                        c_node.max_radius = c_node.max_radius.max(er);
                        c_node.total_mass = nm;
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

    fn subdivide(&mut self, parent_idx: usize) -> usize {
        let p = self.nodes[parent_idx];
        let hw = p.width * 0.5;
        let hh = p.height * 0.5;
        let qw = p.width * 0.25;
        let qh = p.height * 0.25;
        let base = self.nodes.len();
        self.nodes[parent_idx].first_child = base as u32;
        self.nodes.push(OverlapQuadNode::new(
            p.center_x - qw,
            p.center_y - qh,
            hw,
            hh,
        ));
        self.nodes.push(OverlapQuadNode::new(
            p.center_x + qw,
            p.center_y - qh,
            hw,
            hh,
        ));
        self.nodes.push(OverlapQuadNode::new(
            p.center_x - qw,
            p.center_y + qh,
            hw,
            hh,
        ));
        self.nodes.push(OverlapQuadNode::new(
            p.center_x + qw,
            p.center_y + qh,
            hw,
            hh,
        ));
        base
    }

    #[allow(clippy::too_many_arguments)]
    fn compute_repulsion(
        &self,
        xs: &[f64],
        ys: &[f64],
        masses: &[f64],
        radii: &[f64],
        theta: f64,
        scaling_ratio: f64,
        seed: u64,
        iter: usize,
        out_fx: &mut [f64],
        out_fy: &mut [f64],
    ) {
        let n = xs.len();
        if n < 2 || self.nodes.is_empty() {
            out_fx.fill(0.0);
            out_fy.fill(0.0);
            return;
        }
        let theta_sq = theta * theta;
        let mut stack = [0u32; 128];

        for i in 0..n {
            let x = xs[i];
            let y = ys[i];
            let mi = masses[i];
            let ri = radii[i];
            let mut sp = 1usize;
            stack[0] = 0;
            let mut fx = 0.0_f64;
            let mut fy = 0.0_f64;

            while sp > 0 {
                sp -= 1;
                let node_idx = stack[sp];
                let node = &self.nodes[node_idx as usize];
                if node.count == 0 || node.total_mass == 0.0 {
                    continue;
                }

                if node.first_child == u32::MAX {
                    let mut p = node.leaf_head;
                    while p != u32::MAX {
                        let other = p as usize;
                        if other != i {
                            let (px, py) = overlap_repulsion_pair(
                                x - xs[other],
                                y - ys[other],
                                mi,
                                masses[other],
                                ri + radii[other],
                                scaling_ratio,
                                seed,
                                iter,
                                i,
                                other,
                            );
                            fx += px;
                            fy += py;
                        }
                        p = self.next_in_leaf[other];
                    }
                    continue;
                }

                let dx = x - node.com_x;
                let dy = y - node.com_y;
                let dist_sq = dx * dx + dy * dy;
                let max_dim = node.width.max(node.height);
                if theta > 0.0 && (max_dim * max_dim) < theta_sq * dist_sq {
                    let d = dist_sq.sqrt();
                    let min_sep = 2.0 * (ri + node.max_radius);
                    if d > min_sep && d > 1e-12 {
                        let d_prime = (d - ri - node.avg_radius).max(OVERLAP_DPRIME_MIN);
                        let mult = (scaling_ratio * mi * node.total_mass) / (d * d_prime);
                        fx += mult * dx;
                        fy += mult * dy;
                        continue;
                    }
                }
                let base = node.first_child;
                for c in 0..4 {
                    let child = base + c;
                    if self.nodes[child as usize].count > 0 {
                        stack[sp] = child;
                        sp += 1;
                    }
                }
            }

            out_fx[i] = fx;
            out_fy[i] = fy;
        }
    }
}

#[inline]
#[allow(clippy::too_many_arguments)]
fn overlap_repulsion_pair(
    dx: f64,
    dy: f64,
    mi: f64,
    mj: f64,
    r_sum: f64,
    scaling_ratio: f64,
    seed: u64,
    iter: usize,
    i: usize,
    j: usize,
) -> (f64, f64) {
    let d = dx.hypot(dy);
    let d_prime = d - r_sum;
    if d_prime > 0.0 {
        let d_prime_safe = d_prime.max(OVERLAP_DPRIME_MIN);
        let mult = (scaling_ratio * mi * mj) / (d * d_prime_safe);
        (mult * dx, mult * dy)
    } else if d_prime < 0.0 {
        let (ux, uy) = if d > 1e-12 {
            (dx / d, dy / d)
        } else {
            deterministic_pair_unit_vector(seed, iter, i, j)
        };
        let mag = OVERLAP_REPULSION_K * mi * mj;
        (mag * ux, mag * uy)
    } else {
        (0.0, 0.0)
    }
}

#[inline]
fn deterministic_pair_unit_vector(seed: u64, iter: usize, i: usize, j: usize) -> (f64, f64) {
    let (a, b) = if i <= j {
        (i as u64, j as u64)
    } else {
        (j as u64, i as u64)
    };
    let mut z = seed
        .wrapping_add((iter as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15))
        .wrapping_add(a.wrapping_mul(0xBF58_476D_1CE4_E5B9))
        .wrapping_add(b.wrapping_mul(0x94D0_49BB_1331_11EB));
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^= z >> 31;
    let u = ((z >> 11) as f64) * (1.0 / ((1u64 << 53) as f64));
    let angle = u * std::f64::consts::TAU;
    let sign = if i <= j { 1.0 } else { -1.0 };
    (sign * angle.cos(), sign * angle.sin())
}

/// Cooperative ForceAtlas2 layout solver implementing [`IterativeSolver`].
///
/// Uses degree-weighted masses (`m_i = deg(i) + 1`) with either $O(N \log N)$ Barnes-Hut
/// [`ArenaQuadtree2d`] Coulomb repulsion or exact $O(N^2)$ repulsion, linear/LinLog attraction,
/// standard/strong gravity, optional Jacomy (2014) border-to-border `prevent_overlap`, and
/// adaptive swing-traction step sizing without per-step allocations.
#[derive(Debug, Clone)]
pub struct ForceAtlas2Solver<S: Scalar = f64> {
    config: ForceAtlas2Config,
    num_nodes: usize,
    edges_u: Vec<u32>,
    edges_v: Vec<u32>,
    edges_w: Vec<f64>,
    masses: Vec<f64>,
    node_radii: Vec<f64>,
    x_buf: Vec<f64>,
    y_buf: Vec<f64>,
    fx: Vec<f64>,
    fy: Vec<f64>,
    prev_fx: Vec<f64>,
    prev_fy: Vec<f64>,
    swing_buf: Vec<f64>,
    rep_x: Vec<f64>,
    rep_y: Vec<f64>,
    quadtree: ArenaQuadtree2d<f64>,
    overlap_tree: OverlapQuadtree,
    global_speed: f64,
    max_displacement: f64,
    iter: usize,
    status: SolverStatus,
    _marker: std::marker::PhantomData<S>,
}

impl<S: Scalar> ForceAtlas2Solver<S> {
    /// Constructs a [`ForceAtlas2Solver`] from a [`CsrGraph`] using `config.default_node_radius`.
    pub fn new<W: Scalar>(graph: &CsrGraph<W>, config: ForceAtlas2Config) -> CoreResult<Self> {
        config.validate()?;
        let n = graph.num_nodes();
        let radii = vec![config.default_node_radius; n];
        Self::with_node_radii(graph, config, radii)
    }

    /// Constructs a [`ForceAtlas2Solver`] from a [`CsrGraph`] with explicit per-node radii.
    pub fn with_node_radii<W: Scalar>(
        graph: &CsrGraph<W>,
        config: ForceAtlas2Config,
        node_radii: Vec<f64>,
    ) -> CoreResult<Self> {
        config.validate()?;
        let n_graph = graph.num_nodes();
        if node_radii.len() != n_graph {
            return Err(CoreError::dimension_mismatch(
                n_graph,
                node_radii.len(),
                "ForceAtlas2Solver::with_node_radii::node_radii",
            ));
        }
        for &r in &node_radii {
            if !r.is_finite() {
                return Err(CoreError::non_finite(
                    "ForceAtlas2Solver::with_node_radii::node_radii",
                ));
            }
            if r < 0.0 {
                return Err(CoreError::invalid_input(
                    "node_radii must be finite and >= 0.0",
                ));
            }
        }

        let (n, edges_u, edges_v, raw_weights, deg) = extract_canonical_undirected_edges(graph)?;
        let max_displacement = config
            .max_displacement
            .unwrap_or_else(|| MAX_FA2_DISPLACEMENT * ((n as f64) / 1000.0).sqrt().max(1.0));

        let mut masses = Vec::with_capacity(n);
        let mut total_mass = 0.0_f64;
        for &d in &deg {
            let m = (d as f64) + 1.0;
            masses.push(m);
            total_mass += m;
        }
        let mean_mass = if n > 0 { total_mass / (n as f64) } else { 1.0 };

        let influence = config.edge_weight_influence;
        let mut edges_w = Vec::with_capacity(edges_u.len());
        for idx in 0..edges_u.len() {
            let raw_w = raw_weights[idx];
            let w_pow = if influence == 0.0 {
                1.0
            } else if (influence - 1.0).abs() <= 1e-15 {
                raw_w
            } else if raw_w == 0.0 {
                0.0
            } else {
                raw_w.powf(influence)
            };
            let eff_w = if config.outbound_attraction_distribution {
                let mu = masses[edges_u[idx] as usize];
                let mv = masses[edges_v[idx] as usize];
                w_pow * mean_mass * 0.5 * (1.0 / mu + 1.0 / mv)
            } else {
                w_pow
            };
            edges_w.push(eff_w);
        }

        let rep_len = if config.use_barnes_hut && !config.prevent_overlap {
            n
        } else {
            0
        };
        let status = if config.iterations == 0 {
            SolverStatus::Completed
        } else {
            SolverStatus::Initialized
        };

        Ok(Self {
            config,
            num_nodes: n,
            edges_u,
            edges_v,
            edges_w,
            masses,
            node_radii,
            x_buf: vec![0.0; n],
            y_buf: vec![0.0; n],
            fx: vec![0.0; n],
            fy: vec![0.0; n],
            prev_fx: vec![0.0; n],
            prev_fy: vec![0.0; n],
            swing_buf: vec![0.0; n],
            rep_x: vec![0.0; rep_len],
            rep_y: vec![0.0; rep_len],
            quadtree: ArenaQuadtree2d::new(),
            overlap_tree: OverlapQuadtree::default(),
            global_speed: 1.0,
            max_displacement,
            iter: 0,
            status,
            _marker: std::marker::PhantomData,
        })
    }

    /// Returns the solver configuration.
    #[inline]
    pub fn config(&self) -> &ForceAtlas2Config {
        &self.config
    }

    /// Returns the expected number of vertices in [`Layout2d`].
    #[inline]
    pub fn num_nodes(&self) -> usize {
        self.num_nodes
    }

    /// Returns the per-node collision radii.
    #[inline]
    pub fn node_radii(&self) -> &[f64] {
        &self.node_radii
    }

    /// Returns the effective per-step vertex displacement cap.
    #[inline]
    pub fn max_displacement(&self) -> f64 {
        self.max_displacement
    }
}

impl<S: Scalar> IterativeSolver<S> for ForceAtlas2Solver<S> {
    fn step(&mut self, layout: &mut Layout2d<S>) -> CoreResult<SolverStatus> {
        if layout.len() != self.num_nodes {
            return Err(CoreError::dimension_mismatch(
                self.num_nodes,
                layout.len(),
                "ForceAtlas2Solver::step",
            ));
        }
        layout.validate_finite()?;

        if self.status.is_finished() {
            return Ok(self.status);
        }

        let n = self.num_nodes;
        if n <= 1 {
            self.status = SolverStatus::Converged;
            return Ok(self.status);
        }

        {
            let xs = layout.x();
            let ys = layout.y();
            let mut min_x = f64::INFINITY;
            let mut max_x = f64::NEG_INFINITY;
            let mut min_y = f64::INFINITY;
            let mut max_y = f64::NEG_INFINITY;
            for i in 0..n {
                let x = xs[i].to_f64();
                let y = ys[i].to_f64();
                self.x_buf[i] = x;
                self.y_buf[i] = y;
                min_x = min_x.min(x);
                max_x = max_x.max(x);
                min_y = min_y.min(y);
                max_y = max_y.max(y);
            }

            // Break completely collapsed initial layouts deterministically so forces are non-zero.
            if self.iter == 0 && (max_x - min_x) < 1e-12 && (max_y - min_y) < 1e-12 {
                let mut rng = DeterministicRng::from_seed_u64(self.config.seed);
                for i in 0..n {
                    self.x_buf[i] += 1e-4 * (rng.next_f64() - 0.5);
                    self.y_buf[i] += 1e-4 * (rng.next_f64() - 0.5);
                }
            }
        }

        // 1. Repulsion: Jacomy (2014) border-to-border `d' = d - r1 - r2` when `prevent_overlap`,
        //    otherwise standard O(N log N) weighted Barnes-Hut or O(N^2) exact.
        let prevent_overlap = self.config.prevent_overlap;
        if prevent_overlap {
            if self.config.use_barnes_hut {
                self.overlap_tree
                    .build(&self.x_buf, &self.y_buf, &self.masses, &self.node_radii);
                self.overlap_tree.compute_repulsion(
                    &self.x_buf,
                    &self.y_buf,
                    &self.masses,
                    &self.node_radii,
                    self.config.theta,
                    self.config.scaling_ratio,
                    self.config.seed,
                    self.iter,
                    &mut self.fx,
                    &mut self.fy,
                );
            } else {
                self.fx.fill(0.0);
                self.fy.fill(0.0);
                let sr = self.config.scaling_ratio;
                let seed = self.config.seed;
                let iter = self.iter;
                for i in 0..n {
                    let xi = self.x_buf[i];
                    let yi = self.y_buf[i];
                    let mi = self.masses[i];
                    let ri = self.node_radii[i];
                    let mut acc_x = 0.0_f64;
                    let mut acc_y = 0.0_f64;
                    for j in (i + 1)..n {
                        let (fx_ij, fy_ij) = overlap_repulsion_pair(
                            xi - self.x_buf[j],
                            yi - self.y_buf[j],
                            mi,
                            self.masses[j],
                            ri + self.node_radii[j],
                            sr,
                            seed,
                            iter,
                            i,
                            j,
                        );
                        acc_x += fx_ij;
                        acc_y += fy_ij;
                        self.fx[j] -= fx_ij;
                        self.fy[j] -= fy_ij;
                    }
                    self.fx[i] += acc_x;
                    self.fy[i] += acc_y;
                }
            }
        } else if self.config.use_barnes_hut {
            self.quadtree
                .build_weighted_xy(&self.x_buf, &self.y_buf, &self.masses)?;
            self.quadtree.compute_coulomb_repulsion(
                self.config.theta,
                COULOMB_EPS_SQ,
                &mut self.rep_x,
                &mut self.rep_y,
            )?;
            let sr = self.config.scaling_ratio;
            for i in 0..n {
                let scale = sr * self.masses[i];
                self.fx[i] = scale * self.rep_x[i];
                self.fy[i] = scale * self.rep_y[i];
            }
        } else {
            self.fx.fill(0.0);
            self.fy.fill(0.0);
            let sr = self.config.scaling_ratio;
            for i in 0..n {
                let xi = self.x_buf[i];
                let yi = self.y_buf[i];
                let mi_sr = sr * self.masses[i];
                let mut acc_x = 0.0_f64;
                let mut acc_y = 0.0_f64;
                for j in (i + 1)..n {
                    let dx = xi - self.x_buf[j];
                    let dy = yi - self.y_buf[j];
                    let denom = dx * dx + dy * dy + COULOMB_EPS_SQ;
                    let mult = (mi_sr * self.masses[j]) / denom;
                    let fx_ij = mult * dx;
                    let fy_ij = mult * dy;
                    acc_x += fx_ij;
                    acc_y += fy_ij;
                    self.fx[j] -= fx_ij;
                    self.fy[j] -= fy_ij;
                }
                self.fx[i] += acc_x;
                self.fy[i] += acc_y;
            }
        }

        // 2. Attraction: linear or LinLog along canonical undirected edges.
        //    When `prevent_overlap` is active, use border distance `d' = d - r_u - r_v` when
        //    `d' > 0` and disable attraction when `d' <= 0` (Jacomy 2014 Eqs. 7-9).
        let linlog = self.config.linlog_mode;
        for idx in 0..self.edges_u.len() {
            let u = self.edges_u[idx] as usize;
            let v = self.edges_v[idx] as usize;
            let w = self.edges_w[idx];
            if w == 0.0 {
                continue;
            }
            let dx = self.x_buf[v] - self.x_buf[u];
            let dy = self.y_buf[v] - self.y_buf[u];
            let factor = if prevent_overlap {
                let d = dx.hypot(dy);
                let d_prime = d - (self.node_radii[u] + self.node_radii[v]);
                if d_prime > 0.0 && d > 1e-12 {
                    if linlog {
                        w * d_prime.ln_1p() / d
                    } else {
                        w * d_prime / d
                    }
                } else {
                    0.0
                }
            } else if linlog {
                let d_sq = dx * dx + dy * dy;
                if d_sq > 1e-24 {
                    let d = d_sq.sqrt();
                    w * d.ln_1p() / d
                } else {
                    0.0
                }
            } else {
                w
            };
            if factor == 0.0 {
                continue;
            }
            let ax = factor * dx;
            let ay = factor * dy;
            self.fx[u] += ax;
            self.fy[u] += ay;
            self.fx[v] -= ax;
            self.fy[v] -= ay;
        }

        // 3. Gravity: standard (constant pull m_i * g) or strong (m_i * g * d).
        if self.config.gravity > 0.0 {
            let g = self.config.gravity;
            if self.config.strong_gravity {
                for i in 0..n {
                    let gf = g * self.masses[i];
                    self.fx[i] -= gf * self.x_buf[i];
                    self.fy[i] -= gf * self.y_buf[i];
                }
            } else {
                for i in 0..n {
                    let d = self.x_buf[i].hypot(self.y_buf[i]);
                    if d > 1e-12 {
                        let gf = (g * self.masses[i]) / d;
                        self.fx[i] -= gf * self.x_buf[i];
                        self.fy[i] -= gf * self.y_buf[i];
                    }
                }
            }
        }

        // 4. Adaptive speed from per-node swing and traction.
        let mut total_swing = 0.0_f64;
        let mut total_traction = 0.0_f64;
        for i in 0..n {
            let fxi = if self.fx[i].is_finite() {
                self.fx[i]
            } else {
                0.0
            };
            let fyi = if self.fy[i].is_finite() {
                self.fy[i]
            } else {
                0.0
            };
            self.fx[i] = fxi;
            self.fy[i] = fyi;

            let swg = (fxi - self.prev_fx[i]).hypot(fyi - self.prev_fy[i]);
            let tra = 0.5 * (fxi + self.prev_fx[i]).hypot(fyi + self.prev_fy[i]);
            self.swing_buf[i] = swg;
            let m = self.masses[i];
            total_swing += m * swg;
            total_traction += m * tra;
        }

        let jt = self.config.jitter_tolerance;
        let max_speed = 10.0 * jt.max(1.0);
        let target_speed = if total_swing > 1e-12 {
            (jt * jt * total_traction / total_swing).clamp(MIN_FA2_SPEED, max_speed)
        } else {
            (self.global_speed * 1.5).min(max_speed)
        };
        if self.iter > 0 {
            let max_rise = 0.5 * self.global_speed;
            self.global_speed += (target_speed - self.global_speed).min(max_rise);
        } else {
            self.global_speed = target_speed;
        }
        self.global_speed = self.global_speed.clamp(MIN_FA2_SPEED, max_speed);

        // 5. Apply adaptive displacements back into layout (dividing speed by 10 when
        //    `prevent_overlap` is active per Jacomy 2014 p. 5).
        let speed = self.global_speed;
        let speed_scale = if prevent_overlap { 0.1 } else { 1.0 };
        let max_disp = self.max_displacement;
        let (xs_mut, ys_mut) = layout.xy_mut();
        for i in 0..n {
            let fxi = self.fx[i];
            let fyi = self.fy[i];
            let local_factor = speed_scale * (speed / (1.0 + (speed * self.swing_buf[i]).sqrt()));
            let mut dx = fxi * local_factor;
            let mut dy = fyi * local_factor;
            let disp = dx.hypot(dy);
            if disp > max_disp {
                let cap = max_disp / disp;
                dx *= cap;
                dy *= cap;
            }
            let nx = S::from_f64(self.x_buf[i] + dx);
            let ny = S::from_f64(self.y_buf[i] + dy);
            if nx.is_finite() {
                xs_mut[i] = nx;
            }
            if ny.is_finite() {
                ys_mut[i] = ny;
            }
            self.prev_fx[i] = fxi;
            self.prev_fy[i] = fyi;
        }

        self.iter += 1;
        self.status = if self.iter >= self.config.iterations {
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
        self.iter
    }

    #[inline]
    fn total_iterations(&self) -> usize {
        self.config.iterations
    }
}

#[derive(Debug, Clone)]
struct MultilevelGraphLevel {
    graph: CsrGraph<f64>,
    optimal_distance: f64,
    /// Maps each vertex at this level to its coarse parent in level `l + 1`.
    coarse_of_finer: Option<Vec<u32>>,
    /// Number of vertices at this level collapsed into each coarse vertex in level `l + 1`.
    cluster_counts_of_finer: Option<Vec<u32>>,
}

/// Cooperative multilevel Yifan Hu (2005) layout solver implementing [`IterativeSolver`].
///
/// Constructs an alternating Edge Collapsing (EC) + Maximal Independent Vertex Set (MIVS)
/// coarsening hierarchy directly on [`CsrGraph`], evaluates forces using [`ArenaQuadtree2d`]
/// Coulomb repulsion (`C K^2 / d`) and quadratic spring attraction (`d^2 / K`) with adaptive
/// step length control, and prolongates coarse layouts down to the finest level.
#[derive(Debug, Clone)]
pub struct YifanHuSolver<S: Scalar = f64> {
    config: YifanHuConfig,
    num_nodes: usize,
    has_explicit_weights: bool,
    levels: Vec<MultilevelGraphLevel>,
    fine_to_level: Vec<Vec<u32>>,
    current_level: usize,
    level_iter: usize,
    total_iter_executed: usize,
    max_total_iterations: usize,
    initialized_coords: bool,
    level_step_len: f64,
    level_progress: usize,
    prev_energy: f64,
    active_x: Vec<f64>,
    active_y: Vec<f64>,
    fx: Vec<f64>,
    fy: Vec<f64>,
    rep_x: Vec<f64>,
    rep_y: Vec<f64>,
    unit_masses: Vec<f64>,
    quadtree: ArenaQuadtree2d<f64>,
    rng: DeterministicRng,
    status: SolverStatus,
    _marker: std::marker::PhantomData<S>,
}

impl<S: Scalar> YifanHuSolver<S> {
    /// Constructs a [`YifanHuSolver`] and builds its `EC + MIVS` coarsening hierarchy.
    pub fn new<W: Scalar>(graph: &CsrGraph<W>, config: YifanHuConfig) -> CoreResult<Self> {
        config.validate()?;
        let has_explicit_weights = graph.is_weighted();
        let (n, edges_u, edges_v, edge_weights, _) = extract_canonical_undirected_edges(graph)?;
        let edge_pairs: Vec<(u32, u32)> = edges_u.into_iter().zip(edges_v).collect();
        let level0_graph = CsrGraph::from_edges(n, &edge_pairs, Some(&edge_weights), false, false)?;

        let k0 = config.optimal_distance.unwrap_or(1.0);
        let mut levels = vec![MultilevelGraphLevel {
            graph: level0_graph,
            optimal_distance: k0,
            coarse_of_finer: None,
            cluster_counts_of_finer: None,
        }];

        let mut coarsen_step = 0usize;
        let mut stamp = VisitStamp::new(n);
        while levels.len() < MAX_COARSEN_LEVELS {
            let cur_len = levels.len();
            let cur_graph = &levels[cur_len - 1].graph;
            let cur_n = cur_graph.num_nodes();
            if cur_n <= config.min_coarsen_size || cur_graph.num_edges() == 0 {
                break;
            }

            let primary_is_ec = coarsen_step.is_multiple_of(2);
            let mut candidate = if primary_is_ec {
                ec_coarsen(cur_graph)
            } else {
                mivs_coarsen(cur_graph)
            };

            let mut ratio_ok = (candidate.coarse_n as f64)
                < config.min_coarsen_ratio * (cur_n as f64)
                && candidate.coarse_n < cur_n;
            if !ratio_ok {
                // Fall back once to the complementary coarsening strategy before terminating.
                let alt = if primary_is_ec {
                    mivs_coarsen(cur_graph)
                } else {
                    ec_coarsen(cur_graph)
                };
                if alt.coarse_n < cur_n
                    && (alt.coarse_n as f64) < config.min_coarsen_ratio * (cur_n as f64)
                {
                    candidate = alt;
                    ratio_ok = true;
                }
            }
            if !ratio_ok || candidate.coarse_n == 0 {
                break;
            }

            let coarse_graph = build_coarse_csr(
                cur_graph,
                candidate.coarse_n,
                &candidate.coarse_of,
                &candidate.cluster_counts,
                &mut stamp,
            )?;
            let k_next = k0 * ((n as f64) / (candidate.coarse_n as f64)).sqrt();

            levels[cur_len - 1].coarse_of_finer = Some(candidate.coarse_of);
            levels[cur_len - 1].cluster_counts_of_finer = Some(candidate.cluster_counts);
            levels.push(MultilevelGraphLevel {
                graph: coarse_graph,
                optimal_distance: k_next,
                coarse_of_finer: None,
                cluster_counts_of_finer: None,
            });
            coarsen_step += 1;
        }

        let num_levels = levels.len();
        let mut fine_to_level = Vec::with_capacity(num_levels);
        fine_to_level.push(Vec::new());
        for l in 1..num_levels {
            let mut map = Vec::with_capacity(n);
            let step_map = levels[l - 1]
                .coarse_of_finer
                .as_ref()
                .ok_or_else(|| CoreError::invalid_csr("missing coarse_of_finer mapping"))?;
            if l == 1 {
                map.extend_from_slice(step_map);
            } else {
                let prev = &fine_to_level[l - 1];
                for &idx in prev {
                    map.push(step_map[idx as usize]);
                }
            }
            fine_to_level.push(map);
        }

        let coarsest_level = num_levels - 1;
        let max_total_iterations = num_levels
            .checked_mul(config.max_iterations_per_level)
            .ok_or_else(|| CoreError::invalid_input("YifanHu total_iterations overflow"))?;
        let status = if max_total_iterations == 0 {
            SolverStatus::Completed
        } else {
            SolverStatus::Initialized
        };
        let init_step_len = config.step_size * levels[coarsest_level].optimal_distance;

        Ok(Self {
            config,
            num_nodes: n,
            has_explicit_weights,
            levels,
            fine_to_level,
            current_level: coarsest_level,
            level_iter: 0,
            total_iter_executed: 0,
            max_total_iterations,
            initialized_coords: false,
            level_step_len: init_step_len,
            level_progress: 0,
            prev_energy: f64::INFINITY,
            active_x: vec![0.0; n],
            active_y: vec![0.0; n],
            fx: vec![0.0; n],
            fy: vec![0.0; n],
            rep_x: vec![0.0; n],
            rep_y: vec![0.0; n],
            unit_masses: vec![1.0; n],
            quadtree: ArenaQuadtree2d::new(),
            rng: DeterministicRng::from_seed_u64(config.seed),
            status,
            _marker: std::marker::PhantomData,
        })
    }

    /// Returns the solver configuration.
    #[inline]
    pub fn config(&self) -> &YifanHuConfig {
        &self.config
    }

    /// Returns the expected number of vertices in [`Layout2d`].
    #[inline]
    pub fn num_nodes(&self) -> usize {
        self.num_nodes
    }

    /// Returns the number of levels in the `EC + MIVS` hierarchy (`>= 1`).
    #[inline]
    pub fn num_levels(&self) -> usize {
        self.levels.len()
    }

    /// Returns the currently active level index (`num_levels - 1` coarsest down to `0` finest).
    #[inline]
    pub fn current_level(&self) -> usize {
        self.current_level
    }

    fn initialize_coarsest_coords(&mut self, layout: &Layout2d<S>) {
        let n = self.num_nodes;
        let xs = layout.x();
        let ys = layout.y();
        let mut min_x = f64::INFINITY;
        let mut max_x = f64::NEG_INFINITY;
        let mut min_y = f64::INFINITY;
        let mut max_y = f64::NEG_INFINITY;

        for i in 0..n {
            let x = xs[i].to_f64();
            let y = ys[i].to_f64();
            self.active_x[i] = x;
            self.active_y[i] = y;
            min_x = min_x.min(x);
            max_x = max_x.max(x);
            min_y = min_y.min(y);
            max_y = max_y.max(y);
        }

        let coarsest = self.levels.len() - 1;
        let coarse_n = self.levels[coarsest].graph.num_nodes();
        if (max_x - min_x) < 1e-12 && (max_y - min_y) < 1e-12 {
            let k_coarse = self.levels[coarsest].optimal_distance;
            let spread = k_coarse * (coarse_n as f64).sqrt().max(1.0);
            for i in 0..coarse_n {
                self.active_x[i] = spread * (self.rng.next_f64() - 0.5);
                self.active_y[i] = spread * (self.rng.next_f64() - 0.5);
            }
        } else if coarsest > 0 {
            let map = &self.fine_to_level[coarsest];
            self.fx[..coarse_n].fill(0.0);
            self.fy[..coarse_n].fill(0.0);
            self.rep_x[..coarse_n].fill(0.0);
            for u in 0..n {
                let c = map[u] as usize;
                self.fx[c] += self.active_x[u];
                self.fy[c] += self.active_y[u];
                self.rep_x[c] += 1.0;
            }
            for c in 0..coarse_n {
                let cnt = self.rep_x[c].max(1.0);
                self.active_x[c] = self.fx[c] / cnt;
                self.active_y[c] = self.fy[c] / cnt;
            }
        }
        self.initialized_coords = true;
    }

    fn prolongate_to_finer_level(&mut self) {
        let finer = self.current_level - 1;
        let fine_n = self.levels[finer].graph.num_nodes();
        let k_fine = self.levels[finer].optimal_distance;
        let Some(coarse_of) = self.levels[finer].coarse_of_finer.as_ref() else {
            return;
        };
        let Some(counts) = self.levels[finer].cluster_counts_of_finer.as_ref() else {
            return;
        };

        // Temporarily assemble fine positions in `fx`/`fy` before writing into `active_x`/`active_y`.
        for u in 0..fine_n {
            let c = coarse_of[u] as usize;
            let cx = self.active_x[c];
            let cy = self.active_y[c];
            if counts[c] <= 1 {
                self.fx[u] = cx;
                self.fy[u] = cy;
            } else {
                let angle = std::f64::consts::TAU * self.rng.next_f64();
                let radius =
                    PROLONGATION_JITTER_FACTOR * k_fine * (0.25 + 0.75 * self.rng.next_f64());
                let (s, cs) = angle.sin_cos();
                self.fx[u] = cx + radius * cs;
                self.fy[u] = cy + radius * s;
            }
        }
        self.active_x[..fine_n].copy_from_slice(&self.fx[..fine_n]);
        self.active_y[..fine_n].copy_from_slice(&self.fy[..fine_n]);

        self.current_level = finer;
        self.level_iter = 0;
        self.level_step_len = self.config.step_size * k_fine;
        self.level_progress = 0;
        self.prev_energy = f64::INFINITY;
    }

    fn write_active_coords_to_layout(&self, layout: &mut Layout2d<S>) {
        let (xs_mut, ys_mut) = layout.xy_mut();
        if self.current_level == 0 {
            for i in 0..self.num_nodes {
                let nx = S::from_f64(self.active_x[i]);
                let ny = S::from_f64(self.active_y[i]);
                if nx.is_finite() {
                    xs_mut[i] = nx;
                }
                if ny.is_finite() {
                    ys_mut[i] = ny;
                }
            }
        } else {
            let map = &self.fine_to_level[self.current_level];
            for (u, &c) in map.iter().enumerate() {
                let nx = S::from_f64(self.active_x[c as usize]);
                let ny = S::from_f64(self.active_y[c as usize]);
                if nx.is_finite() {
                    xs_mut[u] = nx;
                }
                if ny.is_finite() {
                    ys_mut[u] = ny;
                }
            }
        }
    }
}

impl<S: Scalar> IterativeSolver<S> for YifanHuSolver<S> {
    fn step(&mut self, layout: &mut Layout2d<S>) -> CoreResult<SolverStatus> {
        if layout.len() != self.num_nodes {
            return Err(CoreError::dimension_mismatch(
                self.num_nodes,
                layout.len(),
                "YifanHuSolver::step",
            ));
        }
        layout.validate_finite()?;

        if self.status.is_finished() {
            return Ok(self.status);
        }

        let n = self.num_nodes;
        if n <= 1 {
            self.status = SolverStatus::Converged;
            return Ok(self.status);
        }

        if !self.initialized_coords {
            self.initialize_coarsest_coords(layout);
        } else if self.current_level == 0 {
            // Sync level 0 coordinates from `layout` so external updates are respected.
            let xs = layout.x();
            let ys = layout.y();
            for i in 0..n {
                self.active_x[i] = xs[i].to_f64();
                self.active_y[i] = ys[i].to_f64();
            }
        }

        let level = &self.levels[self.current_level];
        let ln = level.graph.num_nodes();
        let k_level = level.optimal_distance;

        if ln <= 1 {
            self.level_iter += 1;
            self.total_iter_executed += 1;
            if self.current_level > 0 {
                self.prolongate_to_finer_level();
                self.write_active_coords_to_layout(layout);
                self.status = SolverStatus::Running;
            } else {
                self.write_active_coords_to_layout(layout);
                self.status = SolverStatus::Converged;
            }
            return Ok(self.status);
        }

        // 1. Repulsion at active level via unit-mass ArenaQuadtree2d Coulomb evaluation.
        self.quadtree.build_weighted_xy(
            &self.active_x[..ln],
            &self.active_y[..ln],
            &self.unit_masses[..ln],
        )?;
        let eps_sq = (1e-8 * k_level * k_level).max(1e-24);
        self.quadtree.compute_coulomb_repulsion(
            self.config.theta,
            eps_sq,
            &mut self.rep_x[..ln],
            &mut self.rep_y[..ln],
        )?;

        let rep_scale = self.config.relative_strength * k_level * k_level;
        for i in 0..ln {
            self.fx[i] = rep_scale * self.rep_x[i];
            self.fy[i] = rep_scale * self.rep_y[i];
        }

        // 2. Quadratic spring attraction (magnitude d^2 / K along unit vector -> (d / K) * delta).
        let level_graph = &self.levels[self.current_level].graph;
        let use_weights = self.current_level == 0 && self.has_explicit_weights;
        for u in 0..ln {
            let ux = self.active_x[u];
            let uy = self.active_y[u];
            let nbrs = level_graph.neighbors(u);
            let weights = level_graph.neighbor_weights(u);
            let mut ax = 0.0_f64;
            let mut ay = 0.0_f64;
            for (pos, &v_u32) in nbrs.iter().enumerate() {
                let v = v_u32 as usize;
                let w = if use_weights {
                    weights.map_or(1.0, |ws| ws[pos])
                } else {
                    1.0
                };
                if w == 0.0 {
                    continue;
                }
                let dx = self.active_x[v] - ux;
                let dy = self.active_y[v] - uy;
                let d = dx.hypot(dy);
                let mult = w * (d / k_level);
                ax += mult * dx;
                ay += mult * dy;
            }
            self.fx[u] += ax;
            self.fy[u] += ay;
        }

        // 3. Compute total squared force energy and adapt step length.
        let mut energy = 0.0_f64;
        for i in 0..ln {
            let fxi = if self.fx[i].is_finite() {
                self.fx[i]
            } else {
                0.0
            };
            let fyi = if self.fy[i].is_finite() {
                self.fy[i]
            } else {
                0.0
            };
            self.fx[i] = fxi;
            self.fy[i] = fyi;
            energy += fxi * fxi + fyi * fyi;
        }
        if !energy.is_finite() {
            energy = f64::MAX;
        }

        if self.level_iter > 0 {
            if energy < self.prev_energy {
                self.level_progress += 1;
                if self.level_progress >= 5 {
                    self.level_progress = 0;
                    self.level_step_len =
                        (self.level_step_len / self.config.step_ratio).min(4.0 * k_level);
                }
            } else {
                self.level_progress = 0;
                self.level_step_len *= self.config.step_ratio;
            }
        }
        self.prev_energy = energy;

        // 4. Displace vertices along normalized force direction by adaptive step length.
        let mut max_disp = 0.0_f64;
        let mut max_force = 0.0_f64;
        let step_len = self.level_step_len;
        for i in 0..ln {
            let fxi = self.fx[i];
            let fyi = self.fy[i];
            let f_norm = fxi.hypot(fyi);
            if f_norm > max_force {
                max_force = f_norm;
            }
            if f_norm > 1e-12 {
                let scale = step_len / f_norm;
                let nx = self.active_x[i] + fxi * scale;
                let ny = self.active_y[i] + fyi * scale;
                if nx.is_finite() && ny.is_finite() {
                    self.active_x[i] = nx;
                    self.active_y[i] = ny;
                    if step_len > max_disp {
                        max_disp = step_len;
                    }
                }
            }
        }

        self.level_iter += 1;
        self.total_iter_executed += 1;

        let level_converged =
            max_disp <= self.config.convergence_tolerance * k_level || max_force <= 1e-12;
        let level_exhausted = self.level_iter >= self.config.max_iterations_per_level;

        if level_converged || level_exhausted {
            if self.current_level > 0 {
                self.prolongate_to_finer_level();
                self.write_active_coords_to_layout(layout);
                self.status = SolverStatus::Running;
            } else {
                self.write_active_coords_to_layout(layout);
                self.status = if level_converged {
                    SolverStatus::Converged
                } else {
                    SolverStatus::Completed
                };
            }
        } else {
            self.write_active_coords_to_layout(layout);
            self.status = SolverStatus::Running;
        }

        Ok(self.status)
    }

    #[inline]
    fn status(&self) -> SolverStatus {
        self.status
    }

    #[inline]
    fn iteration(&self) -> usize {
        self.total_iter_executed
    }

    #[inline]
    fn total_iterations(&self) -> usize {
        self.max_total_iterations
    }
}

/// Unified continuous force-directed placement solver wrapping [`ForceAtlas2Solver`] and [`YifanHuSolver`].
#[allow(clippy::large_enum_variant)]
#[derive(Debug, Clone)]
pub enum ContinuousFdp<S: Scalar = f64> {
    /// Continuous ForceAtlas2 solver.
    ForceAtlas2(ForceAtlas2Solver<S>),
    /// Multilevel Yifan Hu solver.
    YifanHu(YifanHuSolver<S>),
}

impl<S: Scalar> ContinuousFdp<S> {
    /// Creates a [`ContinuousFdp::ForceAtlas2`] solver from `graph` and `config`.
    pub fn force_atlas2<W: Scalar>(
        graph: &CsrGraph<W>,
        config: ForceAtlas2Config,
    ) -> CoreResult<Self> {
        Ok(Self::ForceAtlas2(ForceAtlas2Solver::new(graph, config)?))
    }

    /// Creates a [`ContinuousFdp::YifanHu`] solver from `graph` and `config`.
    pub fn yifan_hu<W: Scalar>(graph: &CsrGraph<W>, config: YifanHuConfig) -> CoreResult<Self> {
        Ok(Self::YifanHu(YifanHuSolver::new(graph, config)?))
    }

    /// Returns the expected number of vertices in [`Layout2d`].
    #[inline]
    pub fn num_nodes(&self) -> usize {
        match self {
            Self::ForceAtlas2(solver) => solver.num_nodes(),
            Self::YifanHu(solver) => solver.num_nodes(),
        }
    }
}

impl<S: Scalar> IterativeSolver<S> for ContinuousFdp<S> {
    #[inline]
    fn step(&mut self, layout: &mut Layout2d<S>) -> CoreResult<SolverStatus> {
        match self {
            Self::ForceAtlas2(solver) => solver.step(layout),
            Self::YifanHu(solver) => solver.step(layout),
        }
    }

    #[inline]
    fn status(&self) -> SolverStatus {
        match self {
            Self::ForceAtlas2(solver) => solver.status(),
            Self::YifanHu(solver) => solver.status(),
        }
    }

    #[inline]
    fn iteration(&self) -> usize {
        match self {
            Self::ForceAtlas2(solver) => solver.iteration(),
            Self::YifanHu(solver) => solver.iteration(),
        }
    }

    #[inline]
    fn total_iterations(&self) -> usize {
        match self {
            Self::ForceAtlas2(solver) => solver.total_iterations(),
            Self::YifanHu(solver) => solver.total_iterations(),
        }
    }
}

struct CoarsenMapping {
    coarse_n: usize,
    coarse_of: Vec<u32>,
    cluster_counts: Vec<u32>,
}

fn ec_coarsen(graph: &CsrGraph<f64>) -> CoarsenMapping {
    let n = graph.num_nodes();
    let mut order: Vec<u32> = (0..n as u32).collect();
    order.sort_unstable_by_key(|&u| (graph.degree(u as usize), u));

    let mut coarse_of = vec![u32::MAX; n];
    let mut cluster_counts = Vec::with_capacity(n);
    let mut next_cid = 0u32;

    for &u_u32 in &order {
        let u = u_u32 as usize;
        if coarse_of[u] != u32::MAX {
            continue;
        }

        let nbrs = graph.neighbors(u);
        let weights = graph.neighbor_weights(u);
        let mut best_v = None;
        let mut best_score = (f64::NEG_INFINITY, Reverse(usize::MAX), Reverse(u32::MAX));

        for (pos, &v_u32) in nbrs.iter().enumerate() {
            let v = v_u32 as usize;
            if v == u || coarse_of[v] != u32::MAX {
                continue;
            }
            let w = weights.map_or(1.0, |ws| ws[pos]);
            let score = (w, Reverse(graph.degree(v)), Reverse(v_u32));
            if best_v.is_none() || score > best_score {
                best_score = score;
                best_v = Some(v);
            }
        }

        coarse_of[u] = next_cid;
        if let Some(v) = best_v {
            coarse_of[v] = next_cid;
            cluster_counts.push(2);
        } else {
            cluster_counts.push(1);
        }
        next_cid += 1;
    }

    CoarsenMapping {
        coarse_n: next_cid as usize,
        coarse_of,
        cluster_counts,
    }
}

fn mivs_coarsen(graph: &CsrGraph<f64>) -> CoarsenMapping {
    let n = graph.num_nodes();
    let mut order: Vec<u32> = (0..n as u32).collect();
    order.sort_unstable_by_key(|&u| (graph.degree(u as usize), u));

    let mut covered = VisitStamp::new(n);
    let mut is_indep = vec![false; n];
    let mut coarse_of = vec![u32::MAX; n];
    let mut next_cid = 0u32;

    for &u_u32 in &order {
        let u = u_u32 as usize;
        if !covered.is_visited(u) {
            is_indep[u] = true;
            coarse_of[u] = next_cid;
            next_cid += 1;
            covered.mark(u);
            for &v in graph.neighbors(u) {
                covered.mark(v as usize);
            }
        }
    }

    let coarse_n = next_cid as usize;
    let mut cluster_counts = vec![1u32; coarse_n];

    for u in 0..n {
        if is_indep[u] {
            continue;
        }
        let nbrs = graph.neighbors(u);
        let weights = graph.neighbor_weights(u);
        let mut best_cid = None;
        let mut best_score = (f64::NEG_INFINITY, Reverse(u32::MAX));

        for (pos, &v_u32) in nbrs.iter().enumerate() {
            let v = v_u32 as usize;
            if !is_indep[v] {
                continue;
            }
            let w = weights.map_or(1.0, |ws| ws[pos]);
            let score = (w, Reverse(v_u32));
            if best_cid.is_none() || score > best_score {
                best_score = score;
                best_cid = Some(coarse_of[v]);
            }
        }

        if let Some(cid) = best_cid {
            coarse_of[u] = cid;
            cluster_counts[cid as usize] += 1;
        } else {
            // Defensive singleton assignment (unreachable for valid symmetric graphs).
            let cid = cluster_counts.len() as u32;
            coarse_of[u] = cid;
            cluster_counts.push(1);
        }
    }

    CoarsenMapping {
        coarse_n: cluster_counts.len(),
        coarse_of,
        cluster_counts,
    }
}

fn build_coarse_csr(
    fine: &CsrGraph<f64>,
    coarse_n: usize,
    coarse_of: &[u32],
    cluster_counts: &[u32],
    stamp: &mut VisitStamp,
) -> CoreResult<CsrGraph<f64>> {
    let fine_n = fine.num_nodes();
    let mut group_indptr = Vec::with_capacity(coarse_n + 1);
    group_indptr.push(0usize);
    let mut acc = 0usize;
    for &cnt in cluster_counts {
        acc += cnt as usize;
        group_indptr.push(acc);
    }

    let mut cursor = group_indptr[..coarse_n].to_vec();
    let mut group_nodes = vec![0u32; fine_n];
    for (u, &c) in coarse_of.iter().enumerate() {
        let pos = cursor[c as usize];
        group_nodes[pos] = u as u32;
        cursor[c as usize] = pos + 1;
    }

    stamp.resize_and_clear(coarse_n);
    let mut edge_acc = vec![0.0_f64; coarse_n];
    let mut row_nbrs: Vec<u32> = Vec::new();
    let mut deg = vec![0u32; coarse_n];
    let mut upper_edges: Vec<(u32, u32, f64)> = Vec::new();

    // Sum each undirected coarse edge once in canonical (c < cv) order so both (c, cv) and (cv, c)
    // receive bit-identical f64 weights without non-associative 1-ULP drift.
    for c in 0..coarse_n {
        stamp.clear();
        row_nbrs.clear();

        let start = group_indptr[c];
        let end = group_indptr[c + 1];
        for &u_u32 in &group_nodes[start..end] {
            let u = u_u32 as usize;
            let nbrs = fine.neighbors(u);
            let weights = fine.neighbor_weights(u);
            for (pos, &v_u32) in nbrs.iter().enumerate() {
                let cv = coarse_of[v_u32 as usize] as usize;
                if cv <= c {
                    continue;
                }
                let w = weights.map_or(1.0, |ws| ws[pos]);
                if stamp.visit(cv) {
                    row_nbrs.push(cv as u32);
                    edge_acc[cv] = w;
                } else {
                    edge_acc[cv] += w;
                }
            }
        }

        row_nbrs.sort_unstable();
        let c_u32 = c as u32;
        for &cv_u32 in &row_nbrs {
            let w = edge_acc[cv_u32 as usize];
            upper_edges.push((c_u32, cv_u32, w));
            deg[c] += 1;
            deg[cv_u32 as usize] += 1;
        }
    }

    let mut out_indptr = Vec::with_capacity(coarse_n + 1);
    out_indptr.push(0u32);
    let mut total_nnz = 0u32;
    for &d in &deg {
        total_nnz = total_nnz
            .checked_add(d)
            .ok_or_else(|| CoreError::invalid_csr("coarse edge count exceeds u32::MAX"))?;
        out_indptr.push(total_nnz);
    }

    let nnz = total_nnz as usize;
    let mut out_indices = vec![0u32; nnz];
    let mut out_weights = vec![0.0_f64; nnz];
    let mut write_pos = out_indptr[..coarse_n].to_vec();

    // Since `upper_edges` is ordered by ascending `c` and then ascending `cv` (`c < cv`),
    // appending `cv` to row `c` and `c` to row `cv` yields strictly sorted column indices per row.
    for &(c_u32, cv_u32, w) in &upper_edges {
        let c = c_u32 as usize;
        let cv = cv_u32 as usize;
        let p_c = write_pos[c] as usize;
        out_indices[p_c] = cv_u32;
        out_weights[p_c] = w;
        write_pos[c] = (p_c + 1) as u32;

        let p_cv = write_pos[cv] as usize;
        out_indices[p_cv] = c_u32;
        out_weights[p_cv] = w;
        write_pos[cv] = (p_cv + 1) as u32;
    }

    CsrGraph::from_csr(&out_indptr, &out_indices, Some(&out_weights), false, false)
}

#[allow(clippy::type_complexity)]
fn extract_canonical_undirected_edges<W: Scalar>(
    graph: &CsrGraph<W>,
) -> CoreResult<(usize, Vec<u32>, Vec<u32>, Vec<f64>, Vec<usize>)> {
    let n = graph.num_nodes();
    if u32::try_from(n).is_err() {
        return Err(CoreError::invalid_csr("num_nodes exceeds u32::MAX"));
    }
    if let Some(ws) = graph.weights() {
        for &w in ws {
            let wf = w.to_f64();
            if !wf.is_finite() {
                return Err(CoreError::non_finite("continuous_fdp::graph_weights"));
            }
            if wf < 0.0 {
                return Err(CoreError::invalid_input(
                    "graph edge weights must be non-negative",
                ));
            }
        }
    }

    let mut edges_u = Vec::new();
    let mut edges_v = Vec::new();
    let mut edges_w = Vec::new();
    let mut deg = vec![0usize; n];

    if !graph.is_directed() {
        for (u, deg_u) in deg.iter_mut().enumerate() {
            let u32_u = u as u32;
            let nbrs = graph.neighbors(u);
            let ws = graph.neighbor_weights(u);
            *deg_u = nbrs.len();
            for (pos, &v_u32) in nbrs.iter().enumerate() {
                if u32_u < v_u32 {
                    edges_u.push(u32_u);
                    edges_v.push(v_u32);
                    edges_w.push(ws.map_or(1.0, |w| w[pos].to_f64()));
                }
            }
        }
    } else {
        let mut raw: Vec<(u32, u32, f64)> = Vec::with_capacity(graph.nnz());
        for u in 0..n {
            let u32_u = u as u32;
            let nbrs = graph.neighbors(u);
            let ws = graph.neighbor_weights(u);
            for (pos, &v_u32) in nbrs.iter().enumerate() {
                if u32_u == v_u32 {
                    continue;
                }
                let (a, b) = if u32_u < v_u32 {
                    (u32_u, v_u32)
                } else {
                    (v_u32, u32_u)
                };
                raw.push((a, b, ws.map_or(1.0, |w| w[pos].to_f64())));
            }
        }
        raw.sort_unstable_by_key(|x| (x.0, x.1));

        let weighted = graph.is_weighted();
        let mut idx = 0usize;
        while idx < raw.len() {
            let (a, b, first_w) = raw[idx];
            let mut sum_w = first_w;
            let mut cnt = 1usize;
            idx += 1;
            while idx < raw.len() && raw[idx].0 == a && raw[idx].1 == b {
                sum_w += raw[idx].2;
                cnt += 1;
                idx += 1;
            }
            let merged_w = if weighted { sum_w / (cnt as f64) } else { 1.0 };
            edges_u.push(a);
            edges_v.push(b);
            edges_w.push(merged_w);
            deg[a as usize] += 1;
            deg[b as usize] += 1;
        }
    }

    Ok((n, edges_u, edges_v, edges_w, deg))
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

    fn barbell_graph(k: usize) -> CsrGraph<f64> {
        let n = 2 * k;
        let mut edges = Vec::new();
        for i in 0..k {
            for j in (i + 1)..k {
                edges.push((i as u32, j as u32));
                edges.push(((k + i) as u32, (k + j) as u32));
            }
        }
        edges.push(((k - 1) as u32, k as u32));
        CsrGraph::from_edges(n, &edges, None, false, false).unwrap()
    }

    fn seeded_layout<S: Scalar>(n: usize, seed: u64) -> Layout2d<S> {
        let mut rng = DeterministicRng::from_seed_u64(seed);
        let mut x = Vec::with_capacity(n);
        let mut y = Vec::with_capacity(n);
        for _ in 0..n {
            x.push(S::from_f64(2.0 * (rng.next_f64() - 0.5)));
            y.push(S::from_f64(2.0 * (rng.next_f64() - 0.5)));
        }
        Layout2d::from_xy(x, y).unwrap()
    }

    fn cluster_separation_ratio<S: Scalar>(layout: &Layout2d<S>, k: usize) -> f64 {
        let mut c0x = 0.0;
        let mut c0y = 0.0;
        let mut c1x = 0.0;
        let mut c1y = 0.0;
        for i in 0..k {
            c0x += layout.x()[i].to_f64();
            c0y += layout.y()[i].to_f64();
            c1x += layout.x()[k + i].to_f64();
            c1y += layout.y()[k + i].to_f64();
        }
        c0x /= k as f64;
        c0y /= k as f64;
        c1x /= k as f64;
        c1y /= k as f64;
        let inter = (c0x - c1x).hypot(c0y - c1y);

        let mut intra = 0.0;
        for i in 0..k {
            intra += (layout.x()[i].to_f64() - c0x).hypot(layout.y()[i].to_f64() - c0y);
            intra += (layout.x()[k + i].to_f64() - c1x).hypot(layout.y()[k + i].to_f64() - c1y);
        }
        intra /= (2 * k) as f64;
        inter / intra.max(1e-12)
    }

    #[test]
    fn test_configs_defaults_and_validation() {
        let fa2 = ForceAtlas2Config::default();
        assert_eq!(fa2.iterations, 200);
        assert_eq!(fa2.theta, 1.2);
        assert!(fa2.use_barnes_hut);
        assert!(!fa2.linlog_mode);
        assert!(!fa2.strong_gravity);
        assert_eq!(fa2.gravity, 1.0);
        assert_eq!(fa2.scaling_ratio, 2.0);
        assert_eq!(fa2.edge_weight_influence, 1.0);
        assert!(!fa2.outbound_attraction_distribution);
        assert_eq!(fa2.jitter_tolerance, 1.0);
        assert_eq!(fa2.max_displacement, None);
        assert_eq!(fa2.seed, 42);
        fa2.validate().unwrap();

        for bad in [
            ForceAtlas2Config { theta: -0.1, ..fa2 },
            ForceAtlas2Config {
                gravity: f64::NAN,
                ..fa2
            },
            ForceAtlas2Config {
                scaling_ratio: -1.0,
                ..fa2
            },
            ForceAtlas2Config {
                edge_weight_influence: -0.5,
                ..fa2
            },
            ForceAtlas2Config {
                jitter_tolerance: 0.0,
                ..fa2
            },
            ForceAtlas2Config {
                max_displacement: Some(0.0),
                ..fa2
            },
            ForceAtlas2Config {
                max_displacement: Some(-5.0),
                ..fa2
            },
            ForceAtlas2Config {
                max_displacement: Some(f64::NAN),
                ..fa2
            },
            ForceAtlas2Config {
                max_displacement: Some(f64::INFINITY),
                ..fa2
            },
            ForceAtlas2Config {
                max_displacement: Some(f64::NEG_INFINITY),
                ..fa2
            },
        ] {
            assert!(bad.validate().is_err());
        }

        let yh = YifanHuConfig::default();
        assert_eq!(yh.max_iterations_per_level, 200);
        assert_eq!(yh.step_size, 1.0);
        assert_eq!(yh.step_ratio, 0.9);
        assert_eq!(yh.relative_strength, 0.2);
        assert_eq!(yh.optimal_distance, None);
        assert_eq!(yh.theta, 1.2);
        assert_eq!(yh.min_coarsen_size, 30);
        assert_eq!(yh.min_coarsen_ratio, 0.75);
        assert_eq!(yh.convergence_tolerance, 1e-4);
        assert_eq!(yh.seed, 42);
        yh.validate().unwrap();

        for bad in [
            YifanHuConfig {
                step_size: 0.0,
                ..yh
            },
            YifanHuConfig {
                step_ratio: 1.0,
                ..yh
            },
            YifanHuConfig {
                step_ratio: 0.0,
                ..yh
            },
            YifanHuConfig {
                relative_strength: -0.1,
                ..yh
            },
            YifanHuConfig {
                optimal_distance: Some(0.0),
                ..yh
            },
            YifanHuConfig { theta: -0.1, ..yh },
            YifanHuConfig {
                min_coarsen_size: 0,
                ..yh
            },
            YifanHuConfig {
                min_coarsen_ratio: 0.0,
                ..yh
            },
            YifanHuConfig {
                min_coarsen_ratio: 1.1,
                ..yh
            },
            YifanHuConfig {
                convergence_tolerance: -1e-4,
                ..yh
            },
        ] {
            assert!(bad.validate().is_err());
        }
    }

    #[test]
    fn test_force_atlas2_barnes_hut_vs_exact_and_modes() {
        let graph = barbell_graph(6);
        let init = seeded_layout::<f64>(graph.num_nodes(), 7);

        // 1. Barnes-Hut with theta=0.0 matches Exact to machine-level precision.
        let mut l_exact = init.clone();
        let mut l_bh_zero = init.clone();
        let cfg_exact = ForceAtlas2Config {
            iterations: 20,
            use_barnes_hut: false,
            ..Default::default()
        };
        let cfg_bh_zero = ForceAtlas2Config {
            iterations: 20,
            theta: 0.0,
            use_barnes_hut: true,
            ..Default::default()
        };
        ForceAtlas2Solver::new(&graph, cfg_exact)
            .unwrap()
            .run(&mut l_exact)
            .unwrap();
        ForceAtlas2Solver::new(&graph, cfg_bh_zero)
            .unwrap()
            .run(&mut l_bh_zero)
            .unwrap();
        for i in 0..graph.num_nodes() {
            assert!((l_exact.x()[i] - l_bh_zero.x()[i]).abs() < 1e-8);
            assert!((l_exact.y()[i] - l_bh_zero.y()[i]).abs() < 1e-8);
        }

        // 2. Standard Barnes-Hut (theta=0.5), LinLog, strong gravity, outbound attraction distribution.
        for cfg in [
            ForceAtlas2Config {
                iterations: 150,
                use_barnes_hut: true,
                ..Default::default()
            },
            ForceAtlas2Config {
                iterations: 150,
                linlog_mode: true,
                strong_gravity: true,
                gravity: 0.5,
                outbound_attraction_distribution: true,
                ..Default::default()
            },
        ] {
            let mut layout = init.clone();
            let mut solver = ContinuousFdp::force_atlas2(&graph, cfg).unwrap();
            assert_eq!(solver.run(&mut layout).unwrap(), SolverStatus::Completed);
            assert!(layout.is_all_finite());
            let sep = cluster_separation_ratio(&layout, 6);
            assert!(
                sep > 1.5,
                "expected barbell cluster separation > 1.5, got {sep}"
            );
        }
    }

    #[test]
    fn test_force_atlas2_step_vs_run_bit_exact_f64_and_f32() {
        let graph = grid_graph(4, 4);
        let cfg = ForceAtlas2Config {
            iterations: 25,
            linlog_mode: true,
            outbound_attraction_distribution: true,
            ..Default::default()
        };

        // f64 bit-exact check
        let mut l_run = seeded_layout::<f64>(16, 99);
        let mut l_step = l_run.clone();
        let mut s_run = ForceAtlas2Solver::<f64>::new(&graph, cfg).unwrap();
        let mut s_step = ForceAtlas2Solver::<f64>::new(&graph, cfg).unwrap();
        assert_eq!(s_run.run(&mut l_run).unwrap(), SolverStatus::Completed);
        assert_eq!(
            s_step.run_steps(&mut l_step, 10).unwrap(),
            SolverStatus::Running
        );
        assert_eq!(s_step.iteration(), 10);
        while !s_step.status().is_finished() {
            s_step.step(&mut l_step).unwrap();
        }
        assert_eq!(s_step.iteration(), 25);
        for i in 0..16 {
            assert_eq!(l_run.x()[i].to_bits(), l_step.x()[i].to_bits());
            assert_eq!(l_run.y()[i].to_bits(), l_step.y()[i].to_bits());
        }

        // f32 bit-exact check via ContinuousFdp facade
        let mut l32_run = seeded_layout::<f32>(16, 99);
        let mut l32_step = l32_run.clone();
        let mut s32_run = ContinuousFdp::<f32>::force_atlas2(&graph, cfg).unwrap();
        let mut s32_step = ContinuousFdp::<f32>::force_atlas2(&graph, cfg).unwrap();
        s32_run.run(&mut l32_run).unwrap();
        while !s32_step.status().is_finished() {
            s32_step.step(&mut l32_step).unwrap();
        }
        for i in 0..16 {
            assert_eq!(l32_run.x()[i].to_bits(), l32_step.x()[i].to_bits());
            assert_eq!(l32_run.y()[i].to_bits(), l32_step.y()[i].to_bits());
        }
    }

    #[test]
    fn test_force_atlas2_scale_adaptive_max_displacement_and_step_parity() {
        let grid_16 = grid_graph(4, 4);
        let solver_small =
            ForceAtlas2Solver::<f64>::new(&grid_16, ForceAtlas2Config::default()).unwrap();
        assert_eq!(solver_small.max_displacement(), MAX_FA2_DISPLACEMENT);

        // N = 90_000 sparse path graph scales max_displacement to 50 * sqrt(90) ~= 474.3416.
        let n_large = 90_000usize;
        let large_edges: Vec<(u32, u32)> = (0..n_large - 1)
            .map(|i| (i as u32, (i + 1) as u32))
            .collect();
        let g_large =
            CsrGraph::<f64>::from_edges(n_large, &large_edges, None, false, false).unwrap();
        let solver_large =
            ForceAtlas2Solver::<f64>::new(&g_large, ForceAtlas2Config::default()).unwrap();
        let expected_large = MAX_FA2_DISPLACEMENT * ((n_large as f64) / 1000.0).sqrt();
        assert!((solver_large.max_displacement() - expected_large).abs() < 1e-12);
        assert!((solver_large.max_displacement() - 474.3416490252569).abs() < 1e-9);

        // Explicit override clamps single-step displacement to <= max_displacement.
        let pair_g = CsrGraph::<f64>::from_edges(2, &[(0, 1)], None, false, false).unwrap();
        let cap_val = 12.5_f64;
        let mut s_cap = ForceAtlas2Solver::<f64>::new(
            &pair_g,
            ForceAtlas2Config {
                iterations: 1,
                max_displacement: Some(cap_val),
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(s_cap.max_displacement(), cap_val);
        let init_pair = Layout2d::from_xy(vec![-500.0, 500.0], vec![0.0, 0.0]).unwrap();
        let mut l_cap = init_pair.clone();
        s_cap.step(&mut l_cap).unwrap();
        for i in 0..2 {
            let disp = (l_cap.x()[i] - init_pair.x()[i]).hypot(l_cap.y()[i] - init_pair.y()[i]);
            assert!(
                disp <= cap_val + 1e-12,
                "vertex {i} moved {disp} > {cap_val}"
            );
        }

        // step == run bit-exactness (f64 and f32) for both None and Some(2.5).
        for max_disp in [None, Some(2.5)] {
            let cfg = ForceAtlas2Config {
                iterations: 12,
                max_displacement: max_disp,
                ..Default::default()
            };
            let mut l64_r = seeded_layout::<f64>(16, 21);
            let mut l64_s = l64_r.clone();
            let mut s64_r = ForceAtlas2Solver::<f64>::new(&grid_16, cfg).unwrap();
            let mut s64_s = ForceAtlas2Solver::<f64>::new(&grid_16, cfg).unwrap();
            s64_r.run(&mut l64_r).unwrap();
            s64_s.run_steps(&mut l64_s, 4).unwrap();
            while !s64_s.status().is_finished() {
                s64_s.step(&mut l64_s).unwrap();
            }
            for i in 0..16 {
                assert_eq!(l64_r.x()[i].to_bits(), l64_s.x()[i].to_bits());
                assert_eq!(l64_r.y()[i].to_bits(), l64_s.y()[i].to_bits());
            }

            let mut l32_r = seeded_layout::<f32>(16, 21);
            let mut l32_s = l32_r.clone();
            let mut s32_r = ForceAtlas2Solver::<f32>::new(&grid_16, cfg).unwrap();
            let mut s32_s = ForceAtlas2Solver::<f32>::new(&grid_16, cfg).unwrap();
            s32_r.run(&mut l32_r).unwrap();
            s32_s.run_steps(&mut l32_s, 4).unwrap();
            while !s32_s.status().is_finished() {
                s32_s.step(&mut l32_s).unwrap();
            }
            for i in 0..16 {
                assert_eq!(l32_r.x()[i].to_bits(), l32_s.x()[i].to_bits());
                assert_eq!(l32_r.y()[i].to_bits(), l32_s.y()[i].to_bits());
            }
        }
    }

    #[test]
    fn test_yifan_hu_ec_mivs_multilevel_and_step_bit_exact() {
        // 15x15 grid (225 nodes) must produce multilevel EC+MIVS hierarchy (depth > 1).
        let grid = grid_graph(15, 15);
        let cfg = YifanHuConfig {
            max_iterations_per_level: 40,
            min_coarsen_size: 20,
            ..Default::default()
        };
        let mut yh_run = YifanHuSolver::<f64>::new(&grid, cfg).unwrap();
        let mut yh_step = YifanHuSolver::<f64>::new(&grid, cfg).unwrap();
        assert!(
            yh_run.num_levels() >= 3,
            "expected EC+MIVS hierarchy >= 3 levels on 225-node grid, got {}",
            yh_run.num_levels()
        );
        assert_eq!(yh_run.current_level(), yh_run.num_levels() - 1);
        assert_eq!(yh_run.total_iterations(), yh_run.num_levels() * 40);

        let mut l_run = Layout2d::<f64>::zeros(225);
        let mut l_step = l_run.clone();
        let st_run = yh_run.run(&mut l_run).unwrap();
        assert!(st_run.is_finished());
        assert_eq!(yh_run.current_level(), 0);
        assert!(l_run.is_all_finite());

        // Prolongated & refined grid must have positive spatial spread.
        let spread_x = l_run.x().iter().copied().fold(f64::NEG_INFINITY, f64::max)
            - l_run.x().iter().copied().fold(f64::INFINITY, f64::min);
        assert!(spread_x > 1.0, "expected spread > 1.0, got {spread_x}");

        while !yh_step.status().is_finished() {
            yh_step.step(&mut l_step).unwrap();
        }
        assert_eq!(yh_run.iteration(), yh_step.iteration());
        assert_eq!(yh_run.status(), yh_step.status());
        for i in 0..225 {
            assert_eq!(l_run.x()[i].to_bits(), l_step.x()[i].to_bits());
            assert_eq!(l_run.y()[i].to_bits(), l_step.y()[i].to_bits());
        }

        // f32 bit-exact check via ContinuousFdp::yifan_hu
        let barbell = barbell_graph(18);
        let cfg_b = YifanHuConfig {
            max_iterations_per_level: 30,
            min_coarsen_size: 10,
            ..Default::default()
        };
        let mut l32_run = seeded_layout::<f32>(barbell.num_nodes(), 123);
        let mut l32_step = l32_run.clone();
        let mut c32_run = ContinuousFdp::<f32>::yifan_hu(&barbell, cfg_b).unwrap();
        let mut c32_step = ContinuousFdp::<f32>::yifan_hu(&barbell, cfg_b).unwrap();
        c32_run.run(&mut l32_run).unwrap();
        while !c32_step.status().is_finished() {
            c32_step.step(&mut l32_step).unwrap();
        }
        assert!(l32_run.is_all_finite());
        for i in 0..barbell.num_nodes() {
            assert_eq!(l32_run.x()[i].to_bits(), l32_step.x()[i].to_bits());
            assert_eq!(l32_run.y()[i].to_bits(), l32_step.y()[i].to_bits());
        }
    }

    #[test]
    fn test_continuous_fdp_edge_cases_directed_weighted_and_errors() {
        // Empty, singleton, edgeless, and zero-iteration graphs.
        for n in [0usize, 1, 5] {
            let g = CsrGraph::<f64>::from_edges(n, &[], None, false, false).unwrap();
            let mut l_fa2 = Layout2d::<f64>::zeros(n);
            let mut fa2 = ForceAtlas2Solver::new(&g, ForceAtlas2Config::default()).unwrap();
            assert!(fa2.run(&mut l_fa2).unwrap().is_finished());
            assert!(l_fa2.is_all_finite());

            let mut l_yh = Layout2d::<f64>::zeros(n);
            let mut yh = YifanHuSolver::new(&g, YifanHuConfig::default()).unwrap();
            assert!(yh.run(&mut l_yh).unwrap().is_finished());
            assert!(l_yh.is_all_finite());
        }

        // Zero-iteration configs complete immediately.
        let g4 = grid_graph(2, 2);
        let fa2_zero = ForceAtlas2Solver::<f64>::new(
            &g4,
            ForceAtlas2Config {
                iterations: 0,
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(fa2_zero.status(), SolverStatus::Completed);
        assert_eq!(fa2_zero.total_iterations(), 0);

        let yh_zero = YifanHuSolver::<f64>::new(
            &g4,
            YifanHuConfig {
                max_iterations_per_level: 0,
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(yh_zero.status(), SolverStatus::Completed);
        assert_eq!(yh_zero.total_iterations(), 0);

        // Directed weighted graph canonicalization + negative weight rejection.
        let dir_g = CsrGraph::from_edges(
            4,
            &[(0, 1), (1, 0), (1, 2), (2, 3)],
            Some(&[2.0, 4.0, 1.5, 3.0]),
            true,
            false,
        )
        .unwrap();
        let mut l_dir = seeded_layout::<f64>(4, 55);
        let mut s_dir = ContinuousFdp::force_atlas2(&dir_g, ForceAtlas2Config::default()).unwrap();
        assert!(s_dir.run(&mut l_dir).unwrap().is_finished());
        assert!(l_dir.is_all_finite());

        let neg_g = CsrGraph::from_edges(2, &[(0, 1)], Some(&[-1.0]), false, false).unwrap();
        assert!(matches!(
            ForceAtlas2Solver::<f64>::new(&neg_g, ForceAtlas2Config::default()),
            Err(CoreError::InvalidInput(_))
        ));
        assert!(matches!(
            YifanHuSolver::<f64>::new(&neg_g, YifanHuConfig::default()),
            Err(CoreError::InvalidInput(_))
        ));

        // Dimension mismatch & non-finite layout errors.
        let mut fa2_err = ForceAtlas2Solver::<f64>::new(&g4, ForceAtlas2Config::default()).unwrap();
        let mut yh_err = YifanHuSolver::<f64>::new(&g4, YifanHuConfig::default()).unwrap();
        let mut wrong = Layout2d::<f64>::zeros(2);
        assert!(matches!(
            fa2_err.step(&mut wrong),
            Err(CoreError::DimensionMismatch { .. })
        ));
        assert!(matches!(
            yh_err.step(&mut wrong),
            Err(CoreError::DimensionMismatch { .. })
        ));

        let mut bad = Layout2d::from_xy(vec![0.0, f64::NAN, 1.0, 2.0], vec![0.0; 4]).unwrap();
        assert!(matches!(
            fa2_err.step(&mut bad),
            Err(CoreError::NonFiniteValue { .. })
        ));
        assert!(matches!(
            yh_err.step(&mut bad),
            Err(CoreError::NonFiniteValue { .. })
        ));
    }

    fn assert_copy<T: Copy>(_: T) {}

    fn count_disk_overlaps<S: Scalar>(layout: &Layout2d<S>, radii: &[f64]) -> usize {
        let n = layout.len();
        let mut overlaps = 0usize;
        for i in 0..n {
            for j in (i + 1)..n {
                let d = (layout.x()[i].to_f64() - layout.x()[j].to_f64())
                    .hypot(layout.y()[i].to_f64() - layout.y()[j].to_f64());
                if d < radii[i] + radii[j] - 1e-6 {
                    overlaps += 1;
                }
            }
        }
        overlaps
    }

    #[test]
    fn test_force_atlas2_prevent_overlap_resolves_overlaps_and_validation() {
        let cfg_default = ForceAtlas2Config::default();
        assert!(!cfg_default.prevent_overlap);
        assert_eq!(cfg_default.default_node_radius, 1.0);
        assert_copy(cfg_default);

        // Invalid default_node_radius rejected.
        for bad_r in [-0.1, f64::NAN, f64::INFINITY] {
            assert!(
                ForceAtlas2Config {
                    default_node_radius: bad_r,
                    ..cfg_default
                }
                .validate()
                .is_err()
            );
        }

        // with_node_radii validation.
        let g4 = grid_graph(2, 2);
        assert!(matches!(
            ForceAtlas2Solver::<f64>::with_node_radii(&g4, cfg_default, vec![1.0; 3]),
            Err(CoreError::DimensionMismatch { .. })
        ));
        assert!(matches!(
            ForceAtlas2Solver::<f64>::with_node_radii(&g4, cfg_default, vec![1.0, -0.5, 1.0, 1.0]),
            Err(CoreError::InvalidInput(_))
        ));
        assert!(matches!(
            ForceAtlas2Solver::<f64>::with_node_radii(
                &g4,
                cfg_default,
                vec![1.0, f64::NAN, 1.0, 1.0]
            ),
            Err(CoreError::NonFiniteValue { .. })
        ));

        // 1. Overlapping initial layout on a 4x4 grid with heterogeneous radii -> 0 overlaps
        //    under both Exact and Barnes-Hut.
        let grid = grid_graph(4, 4);
        let radii: Vec<f64> = (0..16).map(|i| 0.8 + 0.1 * ((i % 4) as f64)).collect();
        let mut init_overlap = seeded_layout::<f64>(16, 42);
        // Compress initial positions so many pairs overlap initially.
        for i in 0..16 {
            init_overlap.x_mut()[i] *= 0.25;
            init_overlap.y_mut()[i] *= 0.25;
        }
        assert!(count_disk_overlaps(&init_overlap, &radii) > 10);

        for use_bh in [false, true] {
            let mut layout = init_overlap.clone();
            let cfg = ForceAtlas2Config {
                iterations: 300,
                use_barnes_hut: use_bh,
                prevent_overlap: true,
                linlog_mode: use_bh,
                ..Default::default()
            };
            let mut solver =
                ForceAtlas2Solver::<f64>::with_node_radii(&grid, cfg, radii.clone()).unwrap();
            assert_eq!(solver.node_radii(), radii.as_slice());
            assert_eq!(solver.run(&mut layout).unwrap(), SolverStatus::Completed);
            assert!(layout.is_all_finite());
            assert_eq!(
                count_disk_overlaps(&layout, &radii),
                0,
                "expected 0 disk overlaps with use_barnes_hut={use_bh}"
            );
        }

        // 2. Barnes-Hut with theta=0.0 matches Exact even with prevent_overlap=true and coincident starts.
        let mut l_exact = Layout2d::<f64>::zeros(6);
        let mut l_bh0 = Layout2d::<f64>::zeros(6);
        let g6 = barbell_graph(3);
        let r6 = vec![1.5; 6];
        ForceAtlas2Solver::with_node_radii(
            &g6,
            ForceAtlas2Config {
                iterations: 25,
                use_barnes_hut: false,
                prevent_overlap: true,
                ..Default::default()
            },
            r6.clone(),
        )
        .unwrap()
        .run(&mut l_exact)
        .unwrap();
        ForceAtlas2Solver::with_node_radii(
            &g6,
            ForceAtlas2Config {
                iterations: 25,
                use_barnes_hut: true,
                theta: 0.0,
                prevent_overlap: true,
                ..Default::default()
            },
            r6.clone(),
        )
        .unwrap()
        .run(&mut l_bh0)
        .unwrap();
        for i in 0..6 {
            assert!((l_exact.x()[i] - l_bh0.x()[i]).abs() < 1e-8);
            assert!((l_exact.y()[i] - l_bh0.y()[i]).abs() < 1e-8);
        }

        // 3. Bit-exact step() == run() for f64 and f32 with prevent_overlap=true.
        for use_bh in [false, true] {
            let cfg = ForceAtlas2Config {
                iterations: 25,
                use_barnes_hut: use_bh,
                prevent_overlap: true,
                default_node_radius: 1.25,
                ..Default::default()
            };
            let mut l64_r = seeded_layout::<f64>(16, 88);
            let mut l64_s = l64_r.clone();
            let mut s64_r = ForceAtlas2Solver::<f64>::new(&grid, cfg).unwrap();
            let mut s64_s = ForceAtlas2Solver::<f64>::new(&grid, cfg).unwrap();
            s64_r.run(&mut l64_r).unwrap();
            s64_s.run_steps(&mut l64_s, 9).unwrap();
            while !s64_s.status().is_finished() {
                s64_s.step(&mut l64_s).unwrap();
            }
            for i in 0..16 {
                assert_eq!(l64_r.x()[i].to_bits(), l64_s.x()[i].to_bits());
                assert_eq!(l64_r.y()[i].to_bits(), l64_s.y()[i].to_bits());
            }

            let mut l32_r = seeded_layout::<f32>(16, 88);
            let mut l32_s = l32_r.clone();
            let mut s32_r = ForceAtlas2Solver::<f32>::new(&grid, cfg).unwrap();
            let mut s32_s = ForceAtlas2Solver::<f32>::new(&grid, cfg).unwrap();
            s32_r.run(&mut l32_r).unwrap();
            while !s32_s.status().is_finished() {
                s32_s.step(&mut l32_s).unwrap();
            }
            for i in 0..16 {
                assert_eq!(l32_r.x()[i].to_bits(), l32_s.x()[i].to_bits());
                assert_eq!(l32_r.y()[i].to_bits(), l32_s.y()[i].to_bits());
            }
        }
    }
}
