//! True Noack (2007/2009) LinLog and generalized $(a, r)$-PolyLog energy layout solver.
//!
//! Implements Noack's $(a, r)$ energy model (`noack2007_linlog`, `noack2009_modularity_layout`):
//! $$U_{(a, r)}(p) = \sum_{\{u,v\} \in E} w_{\{u,v\}} \frac{\|p_u - p_v\|^{a+1}}{a+1}
//!     - \sum_{\{u,v\} \subset V} m_u m_v \frac{\|p_u - p_v\|^{r+1}}{r+1}
//!     + U_{\text{grav}}(p)$$
//! with $a > r$ (where exponent $-1$ yields $\ln \|p_u - p_v\|$), supporting both
//! [`LinLogRepulsion::EdgeRepulsion`] ($m_u = \deg_w(u)$) and
//! [`LinLogRepulsion::NodeRepulsion`] ($m_u = 1$), mass-barycenter gravity, and
//! $O(N \log N)$ Barnes-Hut acceleration via [`ArenaQuadtree2d`].

use egraph_core::{
    CoreError, CoreResult, CsrGraph, DeterministicRng, IterativeSolver, Layout2d, Scalar,
    SolverStatus,
};
use egraph_spatial::ArenaQuadtree2d;

const COULOMB_EPS_SQ: f64 = 1e-6;
const MIN_LINLOG_SPEED: f64 = 1e-4;
const MAX_LINLOG_SPEED: f64 = 10.0;
const MAX_LINLOG_DISPLACEMENT: f64 = 50.0;
const ISOLATED_NODE_MASS: f64 = 1e-3;
const MAX_QUAD_DEPTH: usize = 32;
const COINCIDENT_DIST_SQ: f64 = 1e-12;

/// Repulsion vertex-mass weighting model for [`LinLogSolver`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum LinLogRepulsion {
    /// Edge-repulsion LinLog ($m_u = \deg_w(u)$, Noack 2007/2009 modularity connection).
    #[default]
    EdgeRepulsion,
    /// Node-repulsion LinLog ($m_u = 1.0$ for all vertices).
    NodeRepulsion,
}

/// Hyperparameters for [`LinLogSolver`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LinLogConfig {
    /// Total number of optimization iterations (`default: 200`).
    pub iterations: usize,
    /// Repulsion mass weighting model (`default: LinLogRepulsion::EdgeRepulsion`).
    pub repulsion_model: LinLogRepulsion,
    /// Attraction force exponent $a$ in $\|d\|^a$ (Noack 2009 `a`; `a = 0.0` is unit pull, `r_polylog = a + 1`, `default: 0.0`).
    pub attraction_exponent: f64,
    /// Repulsion force exponent $r$ in $\|d\|^r$ (Noack 2009 `r`; `r = -1.0` is $1/d$ Coulomb, `default: -1.0`).
    pub repulsion_exponent: f64,
    /// Barnes-Hut quadtree opening angle threshold $\theta \ge 0$ (`default: 1.2`).
    pub theta: f64,
    /// Whether to use $O(N \log N)$ Barnes-Hut quadtree repulsion (`default: true`).
    pub use_barnes_hut: bool,
    /// Gravitational pull coefficient toward mass barycenter $b(p)$ (`default: 0.01`).
    pub gravity: f64,
    /// Exponent applied to edge weights (`default: 1.0`).
    pub edge_weight_influence: f64,
    /// Optional per-step vertex displacement cap (`None` adapts to `50.0 * max(1.0, sqrt(N / 1000.0))`, `default: None`).
    pub max_displacement: Option<f64>,
    /// Deterministic explicit RNG seed for collapsed-layout symmetry breaking (`default: 42`).
    pub seed: u64,
}

impl Default for LinLogConfig {
    fn default() -> Self {
        Self {
            iterations: 200,
            repulsion_model: LinLogRepulsion::EdgeRepulsion,
            attraction_exponent: 0.0,
            repulsion_exponent: -1.0,
            theta: 1.2,
            use_barnes_hut: true,
            gravity: 0.01,
            edge_weight_influence: 1.0,
            max_displacement: None,
            seed: 42,
        }
    }
}

impl LinLogConfig {
    /// Validates all LinLog hyperparameters.
    pub fn validate(&self) -> CoreResult<()> {
        if !self.attraction_exponent.is_finite() {
            return Err(CoreError::invalid_input(format!(
                "attraction_exponent must be finite, got {}",
                self.attraction_exponent
            )));
        }
        if !self.repulsion_exponent.is_finite() {
            return Err(CoreError::invalid_input(format!(
                "repulsion_exponent must be finite, got {}",
                self.repulsion_exponent
            )));
        }
        if self.attraction_exponent <= self.repulsion_exponent {
            return Err(CoreError::invalid_input(format!(
                "attraction_exponent ({}) must be strictly greater than repulsion_exponent ({})",
                self.attraction_exponent, self.repulsion_exponent
            )));
        }
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
        if !self.edge_weight_influence.is_finite() || self.edge_weight_influence < 0.0 {
            return Err(CoreError::invalid_input(format!(
                "edge_weight_influence must be finite and >= 0.0, got {}",
                self.edge_weight_influence
            )));
        }
        if let Some(max_disp) = self.max_displacement
            && (!max_disp.is_finite() || max_disp <= 0.0)
        {
            return Err(CoreError::invalid_input(format!(
                "max_displacement must be finite and > 0.0, got {max_disp}"
            )));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy)]
struct GenQuadNode {
    center_x: f64,
    center_y: f64,
    width: f64,
    height: f64,
    count: usize,
    total_mass: f64,
    com_x: f64,
    com_y: f64,
    leaf_head: u32,
    first_child: u32,
}

impl GenQuadNode {
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
        if depth >= MAX_QUAD_DEPTH {
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

/// Flat arena Barnes-Hut quadtree for generalized $d^r$ repulsion ($r \neq -1$).
#[derive(Debug, Clone, Default)]
struct GenRepulsionQuadtree {
    nodes: Vec<GenQuadNode>,
    leaf_of_point: Vec<u32>,
    next_in_leaf: Vec<u32>,
}

impl GenRepulsionQuadtree {
    fn build(&mut self, xs: &[f64], ys: &[f64], masses: &[f64]) {
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
            .push(GenQuadNode::new(center_x, center_y, width, height));

        for i in 0..n {
            self.insert(i, xs, ys, masses);
        }
    }

    fn insert(&mut self, point_idx: usize, xs: &[f64], ys: &[f64], masses: &[f64]) {
        let x = xs[point_idx];
        let y = ys[point_idx];
        let mass = masses[point_idx];
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
                } else {
                    slot.com_x = (slot.com_x * old_c + x) / new_c;
                    slot.com_y = (slot.com_y * old_c + y) / new_c;
                }
                slot.total_mass = new_m;
                slot.count += 1;
            }

            if node.first_child == u32::MAX {
                let ref_idx = node.leaf_head as usize;
                let rx = xs[ref_idx];
                let ry = ys[ref_idx];
                let dist_sq = (rx - x) * (rx - x) + (ry - y) * (ry - y);
                if !dist_sq.is_finite()
                    || dist_sq < COINCIDENT_DIST_SQ
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
                    let q = self.nodes[cur].quadrant(ex, ey);
                    let child = base_child + q;
                    let c_node = &mut self.nodes[child];
                    if c_node.count == 0 {
                        c_node.count = 1;
                        c_node.total_mass = em;
                        c_node.com_x = ex;
                        c_node.com_y = ey;
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
                        } else {
                            c_node.com_x = (c_node.com_x * oc + ex) / nc;
                            c_node.com_y = (c_node.com_y * oc + ey) / nc;
                        }
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
        self.nodes
            .push(GenQuadNode::new(p.center_x - qw, p.center_y - qh, hw, hh));
        self.nodes
            .push(GenQuadNode::new(p.center_x + qw, p.center_y - qh, hw, hh));
        self.nodes
            .push(GenQuadNode::new(p.center_x - qw, p.center_y + qh, hw, hh));
        self.nodes
            .push(GenQuadNode::new(p.center_x + qw, p.center_y + qh, hw, hh));
        base
    }

    #[allow(clippy::too_many_arguments)]
    fn compute_general_repulsion(
        &self,
        xs: &[f64],
        ys: &[f64],
        masses: &[f64],
        theta: f64,
        repulsion_exponent: f64,
        eps_sq: f64,
        out_x: &mut [f64],
        out_y: &mut [f64],
    ) {
        let n = xs.len();
        if n < 2 || self.nodes.is_empty() {
            out_x.fill(0.0);
            out_y.fill(0.0);
            return;
        }
        let theta_sq = theta * theta;
        let pow_exp = 0.5 * (repulsion_exponent - 1.0);
        let mut stack = [0u32; 128];

        for i in 0..n {
            let x = xs[i];
            let y = ys[i];
            let own_leaf = self.leaf_of_point[i];
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
                    if node_idx == own_leaf {
                        let mut p = node.leaf_head;
                        while p != u32::MAX {
                            let other = p as usize;
                            if other != i {
                                let dx = x - xs[other];
                                let dy = y - ys[other];
                                let d_sq = dx * dx + dy * dy + eps_sq;
                                let mult = masses[other] * d_sq.powf(pow_exp);
                                fx += mult * dx;
                                fy += mult * dy;
                            }
                            p = self.next_in_leaf[other];
                        }
                    } else {
                        let dx = x - node.com_x;
                        let dy = y - node.com_y;
                        let d_sq = dx * dx + dy * dy + eps_sq;
                        let mult = node.total_mass * d_sq.powf(pow_exp);
                        fx += mult * dx;
                        fy += mult * dy;
                    }
                    continue;
                }

                let dx = x - node.com_x;
                let dy = y - node.com_y;
                let dist_sq = dx * dx + dy * dy;
                let max_dim = node.width.max(node.height);
                if (max_dim * max_dim) < theta_sq * dist_sq {
                    let mult = node.total_mass * (dist_sq + eps_sq).powf(pow_exp);
                    fx += mult * dx;
                    fy += mult * dy;
                } else {
                    let base = node.first_child;
                    for c in 0..4 {
                        let child = base + c;
                        if self.nodes[child as usize].count > 0 {
                            stack[sp] = child;
                            sp += 1;
                        }
                    }
                }
            }

            out_x[i] = fx;
            out_y[i] = fy;
        }
    }
}

/// Cooperative true Noack (2007/2009) LinLog / $(a, r)$-PolyLog solver implementing [`IterativeSolver`].
#[derive(Debug, Clone)]
pub struct LinLogSolver<S: Scalar = f64> {
    config: LinLogConfig,
    num_nodes: usize,
    edges_u: Vec<u32>,
    edges_v: Vec<u32>,
    edges_w: Vec<f64>,
    masses: Vec<f64>,
    total_mass: f64,
    x_buf: Vec<f64>,
    y_buf: Vec<f64>,
    fx: Vec<f64>,
    fy: Vec<f64>,
    prev_fx: Vec<f64>,
    prev_fy: Vec<f64>,
    swing_buf: Vec<f64>,
    att_hess: Vec<f64>,
    rep_x: Vec<f64>,
    rep_y: Vec<f64>,
    quadtree: ArenaQuadtree2d<f64>,
    gen_quadtree: GenRepulsionQuadtree,
    global_speed: f64,
    max_displacement: f64,
    iter: usize,
    status: SolverStatus,
    _marker: std::marker::PhantomData<S>,
}

impl<S: Scalar> LinLogSolver<S> {
    /// Constructs a [`LinLogSolver`] from a [`CsrGraph`] and [`LinLogConfig`].
    pub fn new<W: Scalar>(graph: &CsrGraph<W>, config: LinLogConfig) -> CoreResult<Self> {
        config.validate()?;
        let (n, edges_u, edges_v, raw_weights) = extract_canonical_undirected_edges(graph)?;
        let max_displacement = config
            .max_displacement
            .unwrap_or_else(|| MAX_LINLOG_DISPLACEMENT * ((n as f64) / 1000.0).sqrt().max(1.0));
        let influence = config.edge_weight_influence;
        let mut edges_w = Vec::with_capacity(edges_u.len());
        let mut weighted_deg = vec![0.0_f64; n];
        for idx in 0..edges_u.len() {
            let raw_w = raw_weights[idx];
            let w_eff = if influence == 0.0 {
                1.0
            } else if (influence - 1.0).abs() <= 1e-15 {
                raw_w
            } else if raw_w == 0.0 {
                0.0
            } else {
                raw_w.powf(influence)
            };
            edges_w.push(w_eff);
            weighted_deg[edges_u[idx] as usize] += w_eff;
            weighted_deg[edges_v[idx] as usize] += w_eff;
        }

        // Per Noack (2007) §3.6, assign a small positive fallback weight to degree-0 nodes
        // so barycenter and repulsion remain well-defined without NaNs or inert overlaps.
        let mut masses = Vec::with_capacity(n);
        let mut total_mass = 0.0_f64;
        for &d in &weighted_deg {
            let m = match config.repulsion_model {
                LinLogRepulsion::NodeRepulsion => 1.0,
                LinLogRepulsion::EdgeRepulsion => {
                    if d > 0.0 {
                        d
                    } else {
                        ISOLATED_NODE_MASS
                    }
                }
            };
            masses.push(m);
            total_mass += m;
        }

        let rep_len = if config.use_barnes_hut { n } else { 0 };
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
            total_mass,
            x_buf: vec![0.0; n],
            y_buf: vec![0.0; n],
            fx: vec![0.0; n],
            fy: vec![0.0; n],
            prev_fx: vec![0.0; n],
            prev_fy: vec![0.0; n],
            swing_buf: vec![0.0; n],
            att_hess: vec![0.0; n],
            rep_x: vec![0.0; rep_len],
            rep_y: vec![0.0; rep_len],
            quadtree: ArenaQuadtree2d::new(),
            gen_quadtree: GenRepulsionQuadtree::default(),
            global_speed: 1.0,
            max_displacement,
            iter: 0,
            status,
            _marker: std::marker::PhantomData,
        })
    }

    /// Returns the solver configuration.
    #[inline]
    pub fn config(&self) -> &LinLogConfig {
        &self.config
    }

    /// Returns the expected number of vertices in [`Layout2d`].
    #[inline]
    pub fn num_nodes(&self) -> usize {
        self.num_nodes
    }

    /// Returns the per-node repulsion masses ($1$ for node-repulsion, $\deg_w(u)$ for edge-repulsion).
    #[inline]
    pub fn masses(&self) -> &[f64] {
        &self.masses
    }

    /// Returns the effective per-step vertex displacement cap.
    #[inline]
    pub fn max_displacement(&self) -> f64 {
        self.max_displacement
    }

    /// Evaluates the exact Noack (2009) $(a, r)$-LinLog energy $U_{(a,r)}(p)$ of `layout`.
    pub fn energy<T: Scalar>(&self, layout: &Layout2d<T>) -> CoreResult<f64> {
        if layout.len() != self.num_nodes {
            return Err(CoreError::dimension_mismatch(
                self.num_nodes,
                layout.len(),
                "LinLogSolver::energy",
            ));
        }
        layout.validate_finite()?;
        let n = self.num_nodes;
        if n <= 1 {
            return Ok(0.0);
        }

        let xs = layout.x();
        let ys = layout.y();
        let a = self.config.attraction_exponent;
        let r = self.config.repulsion_exponent;
        let a_plus_1 = a + 1.0;
        let r_plus_1 = r + 1.0;

        let mut u_att = 0.0_f64;
        for idx in 0..self.edges_u.len() {
            let u = self.edges_u[idx] as usize;
            let v = self.edges_v[idx] as usize;
            let w = self.edges_w[idx];
            if w == 0.0 {
                continue;
            }
            let d = (xs[u].to_f64() - xs[v].to_f64()).hypot(ys[u].to_f64() - ys[v].to_f64());
            let d_safe = d.max(1e-12);
            let term = if a_plus_1.abs() <= 1e-12 {
                d_safe.ln()
            } else {
                d_safe.powf(a_plus_1) / a_plus_1
            };
            u_att += w * term;
        }

        let mut u_rep = 0.0_f64;
        for i in 0..n {
            let xi = xs[i].to_f64();
            let yi = ys[i].to_f64();
            let mi = self.masses[i];
            for j in (i + 1)..n {
                let d_safe = (xi - xs[j].to_f64()).hypot(yi - ys[j].to_f64()).max(1e-12);
                let term = if r_plus_1.abs() <= 1e-12 {
                    d_safe.ln()
                } else {
                    d_safe.powf(r_plus_1) / r_plus_1
                };
                u_rep -= mi * self.masses[j] * term;
            }
        }

        let mut u_grav = 0.0_f64;
        if self.config.gravity > 0.0 && self.total_mass > 0.0 {
            let mut bx = 0.0_f64;
            let mut by = 0.0_f64;
            for i in 0..n {
                bx += self.masses[i] * xs[i].to_f64();
                by += self.masses[i] * ys[i].to_f64();
            }
            bx /= self.total_mass;
            by /= self.total_mass;
            for i in 0..n {
                let d = (xs[i].to_f64() - bx).hypot(ys[i].to_f64() - by);
                let d_safe = d.max(1e-12);
                let term = if a_plus_1.abs() <= 1e-12 {
                    d_safe.ln()
                } else {
                    d_safe.powf(a_plus_1) / a_plus_1
                };
                u_grav += self.config.gravity * self.masses[i] * term;
            }
        }

        Ok(u_att + u_rep + u_grav)
    }
}

impl<S: Scalar> IterativeSolver<S> for LinLogSolver<S> {
    fn step(&mut self, layout: &mut Layout2d<S>) -> CoreResult<SolverStatus> {
        if layout.len() != self.num_nodes {
            return Err(CoreError::dimension_mismatch(
                self.num_nodes,
                layout.len(),
                "LinLogSolver::step",
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

            if self.iter == 0 && (max_x - min_x) < 1e-12 && (max_y - min_y) < 1e-12 {
                let mut rng = DeterministicRng::from_seed_u64(self.config.seed);
                for i in 0..n {
                    self.x_buf[i] += 1e-4 * (rng.next_f64() - 0.5);
                    self.y_buf[i] += 1e-4 * (rng.next_f64() - 0.5);
                }
            }
        }

        let r_exp = self.config.repulsion_exponent;
        let is_coulomb = (r_exp - (-1.0)).abs() <= 1e-12;

        // 1. Repulsion: m_u * m_v * d^r along (p_u - p_v) / d.
        if self.config.use_barnes_hut {
            if is_coulomb {
                self.quadtree
                    .build_weighted_xy(&self.x_buf, &self.y_buf, &self.masses)?;
                self.quadtree.compute_coulomb_repulsion(
                    self.config.theta,
                    COULOMB_EPS_SQ,
                    &mut self.rep_x,
                    &mut self.rep_y,
                )?;
            } else {
                self.gen_quadtree
                    .build(&self.x_buf, &self.y_buf, &self.masses);
                self.gen_quadtree.compute_general_repulsion(
                    &self.x_buf,
                    &self.y_buf,
                    &self.masses,
                    self.config.theta,
                    r_exp,
                    COULOMB_EPS_SQ,
                    &mut self.rep_x,
                    &mut self.rep_y,
                );
            }
            for i in 0..n {
                let mi = self.masses[i];
                self.fx[i] = mi * self.rep_x[i];
                self.fy[i] = mi * self.rep_y[i];
            }
        } else {
            self.fx.fill(0.0);
            self.fy.fill(0.0);
            let pow_exp = 0.5 * (r_exp - 1.0);
            for i in 0..n {
                let xi = self.x_buf[i];
                let yi = self.y_buf[i];
                let mi = self.masses[i];
                let mut acc_x = 0.0_f64;
                let mut acc_y = 0.0_f64;
                for j in (i + 1)..n {
                    let dx = xi - self.x_buf[j];
                    let dy = yi - self.y_buf[j];
                    let d_sq_reg = dx * dx + dy * dy + COULOMB_EPS_SQ;
                    let mult = if is_coulomb {
                        (mi * self.masses[j]) / d_sq_reg
                    } else {
                        (mi * self.masses[j]) * d_sq_reg.powf(pow_exp)
                    };
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

        // 2. Attraction: w_{uv} * d^a along (p_v - p_u) / d (for a = 0: constant magnitude w_{uv}).
        let a_exp = self.config.attraction_exponent;
        let a_is_zero = a_exp.abs() <= 1e-12;
        let a_is_one = (a_exp - 1.0).abs() <= 1e-12;
        let a_hess_scale = a_exp.abs().max(1.0);
        self.att_hess.fill(0.0);

        for idx in 0..self.edges_u.len() {
            let u = self.edges_u[idx] as usize;
            let v = self.edges_v[idx] as usize;
            let w = self.edges_w[idx];
            if w == 0.0 {
                continue;
            }
            let dx = self.x_buf[v] - self.x_buf[u];
            let dy = self.y_buf[v] - self.y_buf[u];
            let d_sq = dx * dx + dy * dy;
            let (factor, hess) = if a_is_zero {
                if d_sq > 1e-24 {
                    let d = d_sq.sqrt();
                    (w / d, w / d)
                } else {
                    (0.0, 0.0)
                }
            } else if a_is_one {
                (w, w)
            } else if a_exp > 0.0 && d_sq <= 1e-24 {
                (0.0, 0.0)
            } else {
                let d_reg = (d_sq + 1e-24).sqrt();
                let f = w * d_reg.powf(a_exp - 1.0);
                (f, a_hess_scale * f)
            };

            let ax = factor * dx;
            let ay = factor * dy;
            self.fx[u] += ax;
            self.fy[u] += ay;
            self.fx[v] -= ax;
            self.fy[v] -= ay;
            self.att_hess[u] += hess;
            self.att_hess[v] += hess;
        }

        // 3. Mass-barycenter gravity (Noack 2007 §3.6).
        if self.config.gravity > 0.0 && self.total_mass > 0.0 {
            let mut bx = 0.0_f64;
            let mut by = 0.0_f64;
            for i in 0..n {
                bx += self.masses[i] * self.x_buf[i];
                by += self.masses[i] * self.y_buf[i];
            }
            bx /= self.total_mass;
            by /= self.total_mass;

            let g = self.config.gravity;
            for i in 0..n {
                let dx = bx - self.x_buf[i];
                let dy = by - self.y_buf[i];
                let d = dx.hypot(dy);
                if d > 1e-12 {
                    let factor = if a_is_zero {
                        (g * self.masses[i]) / d
                    } else if a_is_one {
                        g * self.masses[i]
                    } else {
                        g * self.masses[i] * d.powf(a_exp - 1.0)
                    };
                    self.fx[i] += factor * dx;
                    self.fy[i] += factor * dy;
                    self.att_hess[i] += a_hess_scale * factor;
                }
            }
        }

        // 4. Adaptive swing-traction speed with local curvature damping so constant-pull a=0
        //    and super-linear a>1 forces converge without oscillation or explosion.
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

        let target_speed = if total_swing > 1e-12 {
            (total_traction / total_swing).clamp(MIN_LINLOG_SPEED, MAX_LINLOG_SPEED)
        } else {
            (self.global_speed * 1.5).min(MAX_LINLOG_SPEED)
        };
        if self.iter > 0 {
            let max_rise = 0.5 * self.global_speed;
            self.global_speed += (target_speed - self.global_speed).min(max_rise);
        } else {
            self.global_speed = target_speed;
        }
        self.global_speed = self.global_speed.clamp(MIN_LINLOG_SPEED, MAX_LINLOG_SPEED);

        // 5. Apply displacements back into layout.
        let speed = self.global_speed;
        let max_disp = self.max_displacement;
        let (xs_mut, ys_mut) = layout.xy_mut();
        for i in 0..n {
            let fxi = self.fx[i];
            let fyi = self.fy[i];
            let local_factor =
                speed / (1.0 + (speed * self.swing_buf[i]).sqrt() + 0.5 * self.att_hess[i]);
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

#[allow(clippy::type_complexity)]
fn extract_canonical_undirected_edges<W: Scalar>(
    graph: &CsrGraph<W>,
) -> CoreResult<(usize, Vec<u32>, Vec<u32>, Vec<f64>)> {
    let n = graph.num_nodes();
    if u32::try_from(n).is_err() {
        return Err(CoreError::invalid_csr("num_nodes exceeds u32::MAX"));
    }
    if let Some(ws) = graph.weights() {
        for &w in ws {
            let wf = w.to_f64();
            if !wf.is_finite() {
                return Err(CoreError::non_finite("linlog::graph_weights"));
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

    if !graph.is_directed() {
        for u in 0..n {
            let u32_u = u as u32;
            let nbrs = graph.neighbors(u);
            let ws = graph.neighbor_weights(u);
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
        }
    }

    Ok((n, edges_u, edges_v, edges_w))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::continuous_fdp::{ForceAtlas2Config, ForceAtlas2Solver};

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

    fn planted_asymmetric_clusters(small_k: usize, large_k: usize) -> CsrGraph<f64> {
        let n = small_k + large_k;
        let mut edges = Vec::new();
        for i in 0..small_k {
            for j in (i + 1)..small_k {
                edges.push((i as u32, j as u32));
            }
        }
        for i in 0..large_k {
            for j in (i + 1)..large_k {
                edges.push(((small_k + i) as u32, (small_k + j) as u32));
            }
        }
        // Single bridge edge connecting the small dense cluster to the large dense cluster.
        edges.push(((small_k - 1) as u32, small_k as u32));
        CsrGraph::from_edges(n, &edges, None, false, false).unwrap()
    }

    fn cluster_compactness_and_separation(
        layout: &Layout2d<f64>,
        small_k: usize,
        large_k: usize,
    ) -> (f64, f64, f64) {
        let mut sx = 0.0;
        let mut sy = 0.0;
        for i in 0..small_k {
            sx += layout.x()[i];
            sy += layout.y()[i];
        }
        sx /= small_k as f64;
        sy /= small_k as f64;

        let mut lx = 0.0;
        let mut ly = 0.0;
        for i in small_k..(small_k + large_k) {
            lx += layout.x()[i];
            ly += layout.y()[i];
        }
        lx /= large_k as f64;
        ly /= large_k as f64;

        let inter = (sx - lx).hypot(sy - ly);

        let mut small_diam = 0.0_f64;
        for i in 0..small_k {
            for j in (i + 1)..small_k {
                let d = (layout.x()[i] - layout.x()[j]).hypot(layout.y()[i] - layout.y()[j]);
                small_diam = small_diam.max(d);
            }
        }
        let mut large_diam = 0.0_f64;
        for i in small_k..(small_k + large_k) {
            for j in (i + 1)..(small_k + large_k) {
                let d = (layout.x()[i] - layout.x()[j]).hypot(layout.y()[i] - layout.y()[j]);
                large_diam = large_diam.max(d);
            }
        }
        (small_diam, large_diam, inter)
    }

    #[test]
    fn test_linlog_config_defaults_and_validation() {
        let cfg = LinLogConfig::default();
        assert_eq!(cfg.iterations, 200);
        assert_eq!(cfg.repulsion_model, LinLogRepulsion::EdgeRepulsion);
        assert_eq!(cfg.attraction_exponent, 0.0);
        assert_eq!(cfg.repulsion_exponent, -1.0);
        assert_eq!(cfg.theta, 1.2);
        assert!(cfg.use_barnes_hut);
        assert_eq!(cfg.gravity, 0.01);
        assert_eq!(cfg.edge_weight_influence, 1.0);
        assert_eq!(cfg.max_displacement, None);
        assert_eq!(cfg.seed, 42);
        cfg.validate().unwrap();

        for bad in [
            LinLogConfig {
                attraction_exponent: -1.0,
                repulsion_exponent: -1.0,
                ..cfg
            },
            LinLogConfig {
                attraction_exponent: -2.0,
                repulsion_exponent: -1.0,
                ..cfg
            },
            LinLogConfig {
                attraction_exponent: f64::NAN,
                ..cfg
            },
            LinLogConfig {
                repulsion_exponent: f64::NEG_INFINITY,
                ..cfg
            },
            LinLogConfig { theta: -0.1, ..cfg },
            LinLogConfig {
                gravity: -0.01,
                ..cfg
            },
            LinLogConfig {
                edge_weight_influence: -1.0,
                ..cfg
            },
            LinLogConfig {
                max_displacement: Some(0.0),
                ..cfg
            },
            LinLogConfig {
                max_displacement: Some(f64::NAN),
                ..cfg
            },
        ] {
            assert!(bad.validate().is_err());
        }
    }

    #[test]
    fn test_linlog_pinhole_collapse_edge_vs_node_and_distinct_from_fa2_linlog() {
        let small_k = 5;
        let large_k = 25;
        let graph = planted_asymmetric_clusters(small_k, large_k);
        let init = seeded_layout::<f64>(graph.num_nodes(), 11);

        // 1. True Noack Edge-Repulsion LinLog exhibits extreme pinhole collapse of the small cluster
        //    relative to inter-cluster separation and the large cluster.
        let mut l_edge = init.clone();
        let mut s_edge = LinLogSolver::new(
            &graph,
            LinLogConfig {
                iterations: 300,
                repulsion_model: LinLogRepulsion::EdgeRepulsion,
                use_barnes_hut: false,
                ..Default::default()
            },
        )
        .unwrap();
        let e0 = s_edge.energy(&l_edge).unwrap();
        assert_eq!(s_edge.run(&mut l_edge).unwrap(), SolverStatus::Completed);
        let e1 = s_edge.energy(&l_edge).unwrap();
        assert!(e1 < e0, "expected LinLog energy decrease: {e0} -> {e1}");

        let (small_d_edge, large_d_edge, inter_edge) =
            cluster_compactness_and_separation(&l_edge, small_k, large_k);
        let pinhole_ratio_edge = inter_edge / small_d_edge.max(1e-12);
        assert!(
            pinhole_ratio_edge > 20.0,
            "expected strong pinhole collapse separation/small_diam > 20, got {pinhole_ratio_edge:.3} (small={small_d_edge:.3}, large={large_d_edge:.3}, inter={inter_edge:.3})"
        );

        // 2. Node-Repulsion LinLog produces distinctly less asymmetric cluster expansion than Edge-Repulsion.
        let mut l_node = init.clone();
        let mut s_node = LinLogSolver::new(
            &graph,
            LinLogConfig {
                iterations: 300,
                repulsion_model: LinLogRepulsion::NodeRepulsion,
                use_barnes_hut: false,
                ..Default::default()
            },
        )
        .unwrap();
        s_node.run(&mut l_node).unwrap();
        let (small_d_node, _large_d_node, inter_node) =
            cluster_compactness_and_separation(&l_node, small_k, large_k);
        assert!(
            inter_edge > inter_node * 2.0,
            "expected EdgeRepulsion inter-cluster distance ({inter_edge:.3}) >> NodeRepulsion ({inter_node:.3})"
        );
        assert!(
            (inter_edge / small_d_edge.max(1e-12)) > (inter_node / small_d_node.max(1e-12)) * 1.5,
            "expected EdgeRepulsion pinhole ratio > 1.5x NodeRepulsion"
        );

        // 3. ForceAtlas2(linlog_mode: true) uses ln(1 + d) attraction + (deg + 1) repulsion,
        //    producing distinctly different cluster compactness/separation than true Noack LinLog.
        //    3a. Compare EdgeRepulsion LinLog against FA2(linlog_mode: true) with matched gravity.
        let mut l_fa2_matched = init.clone();
        ForceAtlas2Solver::new(
            &graph,
            ForceAtlas2Config {
                iterations: 300,
                linlog_mode: true,
                use_barnes_hut: false,
                gravity: 0.01,
                scaling_ratio: 1.0,
                ..Default::default()
            },
        )
        .unwrap()
        .run(&mut l_fa2_matched)
        .unwrap();
        let (small_d_fa2, _large_d_fa2, inter_fa2) =
            cluster_compactness_and_separation(&l_fa2_matched, small_k, large_k);
        let pinhole_ratio_fa2 = inter_fa2 / small_d_fa2.max(1e-12);

        assert!(
            (inter_edge - inter_fa2).abs() / inter_edge.max(1.0) > 0.25,
            "expected distinct separation between Noack LinLog ({inter_edge:.3}) and FA2 linlog_mode ({inter_fa2:.3})"
        );
        assert!(
            pinhole_ratio_edge > pinhole_ratio_fa2 * 1.25,
            "expected Noack LinLog pinhole ratio ({pinhole_ratio_edge:.3}) > 1.25x FA2 linlog_mode ({pinhole_ratio_fa2:.3})"
        );

        //    3b. Compare default FA2(linlog_mode: true) against Noack NodeRepulsion and EdgeRepulsion:
        //        Noack NodeRepulsion collapses dense clusters tightly while FA2(linlog_mode: true)
        //        keeps both intra-cluster diameters larger and inter-cluster ratio much smaller.
        let mut l_fa2_default = init.clone();
        ForceAtlas2Solver::new(
            &graph,
            ForceAtlas2Config {
                iterations: 300,
                linlog_mode: true,
                use_barnes_hut: false,
                ..Default::default()
            },
        )
        .unwrap()
        .run(&mut l_fa2_default)
        .unwrap();
        let (small_d_fa2_def, _large_d_fa2_def, inter_fa2_def) =
            cluster_compactness_and_separation(&l_fa2_default, small_k, large_k);
        let pinhole_ratio_fa2_def = inter_fa2_def / small_d_fa2_def.max(1e-12);
        assert!(
            pinhole_ratio_edge > pinhole_ratio_fa2_def * 3.0,
            "expected Noack LinLog pinhole ratio ({pinhole_ratio_edge:.3}) > 3x default FA2 linlog_mode ({pinhole_ratio_fa2_def:.3})"
        );
        assert!(
            small_d_node < small_d_fa2_def * 0.5,
            "expected NodeRepulsion small cluster diameter ({small_d_node:.3}) < 0.5x default FA2 linlog_mode ({small_d_fa2_def:.3})"
        );
    }

    #[test]
    fn test_linlog_barnes_hut_vs_exact_and_general_polylog_exponents() {
        let graph = planted_asymmetric_clusters(4, 6);
        let init = seeded_layout::<f64>(graph.num_nodes(), 77);

        // 1. Coulomb r = -1.0: ArenaQuadtree2d with theta = 0.0 matches exact O(N^2).
        let mut l_exact = init.clone();
        let mut l_bh0 = init.clone();
        LinLogSolver::new(
            &graph,
            LinLogConfig {
                iterations: 20,
                use_barnes_hut: false,
                ..Default::default()
            },
        )
        .unwrap()
        .run(&mut l_exact)
        .unwrap();
        LinLogSolver::new(
            &graph,
            LinLogConfig {
                iterations: 20,
                use_barnes_hut: true,
                theta: 0.0,
                ..Default::default()
            },
        )
        .unwrap()
        .run(&mut l_bh0)
        .unwrap();
        for i in 0..graph.num_nodes() {
            assert!((l_exact.x()[i] - l_bh0.x()[i]).abs() < 1e-8);
            assert!((l_exact.y()[i] - l_bh0.y()[i]).abs() < 1e-8);
        }

        // 2. General (a=2.0, r=-1.0) 3-PolyLog (FR-like energy) and (a=0.5, r=-1.5):
        //    theta = 0.0 matches exact, and default theta = 1.2 converges cleanly without NaN/explosion.
        for (a_exp, r_exp) in [(2.0, -1.0), (0.5, -1.5)] {
            let mut l_gen_ex = init.clone();
            let mut l_gen_bh0 = init.clone();
            let mut l_gen_bh = init.clone();
            let base_cfg = LinLogConfig {
                iterations: 25,
                attraction_exponent: a_exp,
                repulsion_exponent: r_exp,
                ..Default::default()
            };
            LinLogSolver::new(
                &graph,
                LinLogConfig {
                    use_barnes_hut: false,
                    ..base_cfg
                },
            )
            .unwrap()
            .run(&mut l_gen_ex)
            .unwrap();
            LinLogSolver::new(
                &graph,
                LinLogConfig {
                    use_barnes_hut: true,
                    theta: 0.0,
                    ..base_cfg
                },
            )
            .unwrap()
            .run(&mut l_gen_bh0)
            .unwrap();
            let mut s_bh = LinLogSolver::new(
                &graph,
                LinLogConfig {
                    use_barnes_hut: true,
                    theta: 1.2,
                    ..base_cfg
                },
            )
            .unwrap();
            let e_before = s_bh.energy(&l_gen_bh).unwrap();
            s_bh.run(&mut l_gen_bh).unwrap();
            let e_after = s_bh.energy(&l_gen_bh).unwrap();
            assert!(l_gen_bh.is_all_finite());
            assert!(
                e_after < e_before,
                "expected energy decrease for (a={a_exp}, r={r_exp}): {e_before} -> {e_after}"
            );
            for i in 0..graph.num_nodes() {
                assert!((l_gen_ex.x()[i] - l_gen_bh0.x()[i]).abs() < 1e-8);
                assert!((l_gen_ex.y()[i] - l_gen_bh0.y()[i]).abs() < 1e-8);
            }
        }
    }

    #[test]
    fn test_linlog_step_vs_run_bit_exact_and_edge_cases() {
        let graph = planted_asymmetric_clusters(4, 8);
        let cfg = LinLogConfig {
            iterations: 30,
            ..Default::default()
        };

        // f64 bit-exact step vs run
        let mut l64_r = seeded_layout::<f64>(12, 5);
        let mut l64_s = l64_r.clone();
        let mut s64_r = LinLogSolver::<f64>::new(&graph, cfg).unwrap();
        let mut s64_s = LinLogSolver::<f64>::new(&graph, cfg).unwrap();
        s64_r.run(&mut l64_r).unwrap();
        assert_eq!(
            s64_s.run_steps(&mut l64_s, 11).unwrap(),
            SolverStatus::Running
        );
        while !s64_s.status().is_finished() {
            s64_s.step(&mut l64_s).unwrap();
        }
        for i in 0..12 {
            assert_eq!(l64_r.x()[i].to_bits(), l64_s.x()[i].to_bits());
            assert_eq!(l64_r.y()[i].to_bits(), l64_s.y()[i].to_bits());
        }

        // f32 bit-exact step vs run
        let mut l32_r = seeded_layout::<f32>(12, 5);
        let mut l32_s = l32_r.clone();
        let mut s32_r = LinLogSolver::<f32>::new(&graph, cfg).unwrap();
        let mut s32_s = LinLogSolver::<f32>::new(&graph, cfg).unwrap();
        s32_r.run(&mut l32_r).unwrap();
        while !s32_s.status().is_finished() {
            s32_s.step(&mut l32_s).unwrap();
        }
        for i in 0..12 {
            assert_eq!(l32_r.x()[i].to_bits(), l32_s.x()[i].to_bits());
            assert_eq!(l32_r.y()[i].to_bits(), l32_s.y()[i].to_bits());
        }

        // Degenerate graphs: N=0, N=1, edgeless, collapsed start, mixed connected+isolated, directed reciprocal.
        for n in [0usize, 1, 5] {
            let g = CsrGraph::<f64>::from_edges(n, &[], None, false, false).unwrap();
            let mut l = Layout2d::<f64>::zeros(n);
            let mut s = LinLogSolver::new(&g, cfg).unwrap();
            assert!(s.run(&mut l).unwrap().is_finished());
            assert!(l.is_all_finite());
        }

        let mixed =
            CsrGraph::<f64>::from_edges(5, &[(0, 1), (1, 2), (2, 0)], None, false, false).unwrap();
        let mut l_mixed = Layout2d::<f64>::zeros(5);
        let mut s_mixed = LinLogSolver::new(&mixed, cfg).unwrap();
        assert_eq!(s_mixed.masses()[3], ISOLATED_NODE_MASS);
        assert_eq!(s_mixed.masses()[4], ISOLATED_NODE_MASS);
        assert!(s_mixed.run(&mut l_mixed).unwrap().is_finished());
        assert!(l_mixed.is_all_finite());

        let dir_g = CsrGraph::from_edges(
            4,
            &[(0, 1), (1, 0), (1, 2), (2, 3)],
            Some(&[2.0, 4.0, 1.5, 3.0]),
            true,
            false,
        )
        .unwrap();
        let mut l_dir = seeded_layout::<f64>(4, 9);
        assert!(
            LinLogSolver::new(&dir_g, cfg)
                .unwrap()
                .run(&mut l_dir)
                .unwrap()
                .is_finished()
        );
        assert!(l_dir.is_all_finite());

        // Zero-iteration immediate completion & error paths.
        let s_zero = LinLogSolver::<f64>::new(
            &mixed,
            LinLogConfig {
                iterations: 0,
                ..cfg
            },
        )
        .unwrap();
        assert_eq!(s_zero.status(), SolverStatus::Completed);

        let neg_g = CsrGraph::from_edges(2, &[(0, 1)], Some(&[-1.0]), false, false).unwrap();
        assert!(matches!(
            LinLogSolver::<f64>::new(&neg_g, cfg),
            Err(CoreError::InvalidInput(_))
        ));

        let mut s_err = LinLogSolver::<f64>::new(&mixed, cfg).unwrap();
        let mut wrong = Layout2d::<f64>::zeros(2);
        assert!(matches!(
            s_err.step(&mut wrong),
            Err(CoreError::DimensionMismatch { .. })
        ));
        let mut bad = Layout2d::from_xy(vec![0.0, f64::NAN, 0.0, 0.0, 0.0], vec![0.0; 5]).unwrap();
        assert!(matches!(
            s_err.step(&mut bad),
            Err(CoreError::NonFiniteValue { .. })
        ));
    }
}
