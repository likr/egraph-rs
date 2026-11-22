//! Force-Directed Edge Bundling (`FDEB`) solver with flat coordinate buffers and
//! cooperative `step()`/`run_steps(k)` execution.
//!
//! Implements Holten & Van Wijk (2009), "Force-Directed Edge Bundling for Graph Visualization",
//! taking `&CsrGraph` and `&Layout2d` by reference (avoiding value-drop issues), clamping
//! `acos` arguments to `[-1.0, 1.0]` to eliminate collinear `NaN`s, and guarding zero-length edges.

use std::mem;

use egraph_core::{CoreError, CoreResult, CsrGraph, Layout2d, Scalar, SolverStatus};

const MIN_EDGE_LEN: f64 = 1e-12;
const MIN_SEG_DIST: f64 = 1e-12;

/// Hyperparameters for [`FdebSolver`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FdebConfig {
    /// Number of subdivision cycles (`default: 6`).
    pub cycles: usize,
    /// Initial force step size (`default: 0.1`).
    pub s0: f64,
    /// Initial number of iterations in cycle 0 (`default: 90`).
    pub i0: usize,
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
            i0: 90,
            s_step: 0.5,
            i_step: 2.0 / 3.0,
            spring_k: 0.1,
            minimum_edge_compatibility: 0.6,
        }
    }
}

impl FdebConfig {
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

#[derive(Debug, Clone, Copy)]
struct EdgeEndpoints {
    source: u32,
    target: u32,
    length: f64,
}

#[derive(Debug, Clone, Copy)]
struct EdgePair {
    p: u32,
    q: u32,
    compatibility: f64,
    same_dir: bool,
}

/// Cooperative Force-Directed Edge Bundling (`FDEB`) solver storing contiguous flat edge
/// polyline buffers `(edge_offsets, points_x, points_y)`.
#[derive(Debug, Clone)]
pub struct FdebSolver<S: Scalar = f64> {
    config: FdebConfig,
    num_edges: usize,
    edges: Vec<EdgeEndpoints>,
    edge_pairs: Vec<EdgePair>,
    cycle_iters: Vec<usize>,
    total_iterations: usize,
    current_cycle: usize,
    cycle_iter: usize,
    iter: usize,
    alpha: f64,
    pts_per_edge: usize,
    edge_offsets: Vec<u32>,
    work_x: Vec<f64>,
    work_y: Vec<f64>,
    scratch_x: Vec<f64>,
    scratch_y: Vec<f64>,
    vx: Vec<f64>,
    vy: Vec<f64>,
    points_x: Vec<S>,
    points_y: Vec<S>,
    status: SolverStatus,
}

impl<S: Scalar> FdebSolver<S> {
    /// Constructs an [`FdebSolver`] from a [`CsrGraph`] and node [`Layout2d`] by reference.
    pub fn new<W>(
        graph: &CsrGraph<W>,
        layout: &Layout2d<S>,
        config: FdebConfig,
    ) -> CoreResult<Self> {
        config.validate()?;
        let n = graph.num_nodes();
        if layout.len() != n {
            return Err(CoreError::dimension_mismatch(
                n,
                layout.len(),
                "FdebSolver::new::layout",
            ));
        }
        layout.validate_finite()?;

        let xs = layout.x();
        let ys = layout.y();

        let mut edges = Vec::new();
        let directed = graph.is_directed();
        for u in 0..n {
            let u32_u = u as u32;
            let ux = xs[u].to_f64();
            let uy = ys[u].to_f64();
            for &v in graph.neighbors(u) {
                if directed || u32_u < v {
                    let vx_pos = xs[v as usize].to_f64();
                    let vy_pos = ys[v as usize].to_f64();
                    let length = (vx_pos - ux).hypot(vy_pos - uy);
                    edges.push(EdgeEndpoints {
                        source: u32_u,
                        target: v,
                        length,
                    });
                }
            }
        }

        let num_edges = edges.len();
        let final_pts_per_edge = (1usize << config.cycles) + 1;
        let max_total_pts = num_edges
            .checked_mul(final_pts_per_edge)
            .ok_or_else(|| CoreError::invalid_input("FDEB point buffer size overflow"))?;
        if num_edges > 0 && u32::try_from(max_total_pts).is_err() {
            return Err(CoreError::invalid_input(
                "FDEB total point count exceeds u32::MAX",
            ));
        }

        // Compute per-cycle iteration schedule.
        let mut cycle_iters = Vec::with_capacity(config.cycles);
        let mut cur_iter = config.i0;
        let mut total_iterations = 0usize;
        for _ in 0..config.cycles {
            cycle_iters.push(cur_iter);
            total_iterations = total_iterations.saturating_add(cur_iter);
            cur_iter = ((cur_iter as f64) * config.i_step) as usize;
        }

        // Pre-compute compatible edge pairs.
        let mut edge_pairs = Vec::new();
        for p in 0..num_edges {
            let ep = edges[p];
            if ep.length <= MIN_EDGE_LEN {
                continue;
            }
            let p1x = xs[ep.source as usize].to_f64();
            let p1y = ys[ep.source as usize].to_f64();
            let p2x = xs[ep.target as usize].to_f64();
            let p2y = ys[ep.target as usize].to_f64();

            for (q, &eq) in edges.iter().enumerate().skip(p + 1) {
                if eq.length <= MIN_EDGE_LEN {
                    continue;
                }
                let q1x = xs[eq.source as usize].to_f64();
                let q1y = ys[eq.source as usize].to_f64();
                let q2x = xs[eq.target as usize].to_f64();
                let q2y = ys[eq.target as usize].to_f64();

                let (c_e, cos_theta) = edge_compatibility(
                    p1x, p1y, p2x, p2y, ep.length, q1x, q1y, q2x, q2y, eq.length,
                );
                if c_e >= config.minimum_edge_compatibility {
                    let theta = cos_theta.clamp(-1.0, 1.0).acos();
                    edge_pairs.push(EdgePair {
                        p: p as u32,
                        q: q as u32,
                        compatibility: c_e,
                        same_dir: theta < std::f64::consts::FRAC_PI_2,
                    });
                }
            }
        }

        // Initialize unsubdivided 2-point polylines per edge (source, target).
        let init_pts = num_edges * 2;
        let mut edge_offsets = Vec::with_capacity(num_edges + 1);
        let mut work_x = Vec::with_capacity(max_total_pts);
        let mut work_y = Vec::with_capacity(max_total_pts);
        let mut points_x = Vec::with_capacity(max_total_pts);
        let mut points_y = Vec::with_capacity(max_total_pts);

        edge_offsets.push(0u32);
        for (e_idx, ep) in edges.iter().enumerate() {
            let sx = xs[ep.source as usize];
            let sy = ys[ep.source as usize];
            let tx = xs[ep.target as usize];
            let ty = ys[ep.target as usize];
            work_x.push(sx.to_f64());
            work_y.push(sy.to_f64());
            work_x.push(tx.to_f64());
            work_y.push(ty.to_f64());
            points_x.push(sx);
            points_y.push(sy);
            points_x.push(tx);
            points_y.push(ty);
            edge_offsets.push(((e_idx + 1) * 2) as u32);
        }

        let scratch_x = Vec::with_capacity(max_total_pts);
        let scratch_y = Vec::with_capacity(max_total_pts);
        let vx = vec![0.0; init_pts];
        let vy = vec![0.0; init_pts];

        let status = if total_iterations == 0 {
            SolverStatus::Completed
        } else {
            SolverStatus::Initialized
        };

        Ok(Self {
            config,
            num_edges,
            edges,
            edge_pairs,
            cycle_iters,
            total_iterations,
            current_cycle: 0,
            cycle_iter: 0,
            iter: 0,
            alpha: config.s0,
            pts_per_edge: 2,
            edge_offsets,
            work_x,
            work_y,
            scratch_x,
            scratch_y,
            vx,
            vy,
            points_x,
            points_y,
            status,
        })
    }

    /// Returns the solver configuration.
    #[inline]
    pub fn config(&self) -> &FdebConfig {
        &self.config
    }

    /// Returns the current lifecycle status.
    #[inline]
    pub fn status(&self) -> SolverStatus {
        self.status
    }

    /// Returns the number of iterations executed so far.
    #[inline]
    pub fn iteration(&self) -> usize {
        self.iter
    }

    /// Returns the total scheduled iterations across all subdivision cycles.
    #[inline]
    pub fn total_iterations(&self) -> usize {
        self.total_iterations
    }

    /// Returns `true` if the solver has finished (`Converged` or `Completed`).
    #[inline]
    pub fn is_finished(&self) -> bool {
        self.status.is_finished()
    }

    /// Returns the number of edges being bundled.
    #[inline]
    pub fn num_edges(&self) -> usize {
        self.num_edges
    }

    /// Returns the CSR-style polyline offsets of length `num_edges() + 1`.
    ///
    /// Edge `e`'s bundled polyline occupies indices `edge_offsets()[e]..edge_offsets()[e + 1]`
    /// inside [`FdebSolver::points_x`] and [`FdebSolver::points_y`].
    #[inline]
    pub fn edge_offsets(&self) -> &[u32] {
        &self.edge_offsets
    }

    /// Returns the flat `x` coordinates of all edge control points.
    #[inline]
    pub fn points_x(&self) -> &[S] {
        &self.points_x
    }

    /// Returns the flat `y` coordinates of all edge control points.
    #[inline]
    pub fn points_y(&self) -> &[S] {
        &self.points_y
    }

    /// Returns all edge polyline points packed as interleaved `[x0, y0, x1, y1, ...]`.
    pub fn to_interleaved_points(&self) -> Vec<S> {
        let len = self.points_x.len();
        let mut out = Vec::with_capacity(len * 2);
        for i in 0..len {
            out.push(self.points_x[i]);
            out.push(self.points_y[i]);
        }
        out
    }

    /// Executes a single FDEB force iteration (subdividing edges when entering a new cycle)
    /// and returns the updated [`SolverStatus`].
    pub fn step(&mut self) -> CoreResult<SolverStatus> {
        if self.status.is_finished() {
            return Ok(self.status);
        }

        if self.num_edges == 0 {
            self.status = SolverStatus::Converged;
            return Ok(self.status);
        }

        while self.current_cycle < self.cycle_iters.len()
            && self.cycle_iters[self.current_cycle] == 0
        {
            self.current_cycle += 1;
            self.alpha *= self.config.s_step;
        }

        if self.current_cycle >= self.cycle_iters.len() || self.iter >= self.total_iterations {
            self.status = SolverStatus::Completed;
            return Ok(self.status);
        }

        if self.cycle_iter == 0 {
            self.subdivide_edges();
        }

        self.apply_forces_and_integrate();

        self.iter += 1;
        self.cycle_iter += 1;
        while self.current_cycle < self.cycle_iters.len()
            && self.cycle_iter >= self.cycle_iters[self.current_cycle]
        {
            self.cycle_iter = 0;
            self.current_cycle += 1;
            self.alpha *= self.config.s_step;
        }

        self.status =
            if self.iter >= self.total_iterations || self.current_cycle >= self.cycle_iters.len() {
                SolverStatus::Completed
            } else {
                SolverStatus::Running
            };

        Ok(self.status)
    }

    /// Executes up to `steps` iterations or stops early upon completion.
    pub fn run_steps(&mut self, steps: usize) -> CoreResult<SolverStatus> {
        for _ in 0..steps {
            if self.status.is_finished() {
                break;
            }
            let st = self.step()?;
            if st.is_finished() {
                break;
            }
        }
        Ok(self.status)
    }

    /// Runs the solver to completion.
    pub fn run(&mut self) -> CoreResult<SolverStatus> {
        while !self.status.is_finished() {
            let st = self.step()?;
            if st.is_finished() {
                break;
            }
        }
        Ok(self.status)
    }

    fn subdivide_edges(&mut self) {
        let old_pts = self.pts_per_edge;
        let new_pts = 2 * old_pts - 1;
        let total_new = self.num_edges * new_pts;

        self.scratch_x.clear();
        self.scratch_y.clear();
        self.scratch_x.reserve(total_new);
        self.scratch_y.reserve(total_new);

        for e in 0..self.num_edges {
            let base = e * old_pts;
            for j in 0..(old_pts - 1) {
                let x0 = self.work_x[base + j];
                let y0 = self.work_y[base + j];
                let x1 = self.work_x[base + j + 1];
                let y1 = self.work_y[base + j + 1];
                self.scratch_x.push(x0);
                self.scratch_y.push(y0);
                self.scratch_x.push(0.5 * (x0 + x1));
                self.scratch_y.push(0.5 * (y0 + y1));
            }
            let last_x = self.work_x[base + old_pts - 1];
            let last_y = self.work_y[base + old_pts - 1];
            self.scratch_x.push(last_x);
            self.scratch_y.push(last_y);
        }

        mem::swap(&mut self.work_x, &mut self.scratch_x);
        mem::swap(&mut self.work_y, &mut self.scratch_y);
        self.pts_per_edge = new_pts;

        for (e, offset) in self.edge_offsets.iter_mut().enumerate() {
            *offset = (e * new_pts) as u32;
        }
        self.vx.resize(total_new, 0.0);
        self.vy.resize(total_new, 0.0);
        self.points_x.resize(total_new, S::ZERO);
        self.points_y.resize(total_new, S::ZERO);
    }

    fn apply_forces_and_integrate(&mut self) {
        let m = self.num_edges;
        let pts = self.pts_per_edge;
        let num_mid = pts - 2;
        if num_mid == 0 {
            return;
        }

        self.vx.fill(0.0);
        self.vy.fill(0.0);

        // 1. Spring forces along each subdivided edge.
        let k = self.config.spring_k;
        let num_mid_f = num_mid as f64;
        for (e, ep) in self.edges.iter().enumerate() {
            if ep.length <= MIN_EDGE_LEN {
                continue;
            }
            let kp = k / (num_mid_f * ep.length);
            let base = e * pts;
            for i in 1..=num_mid {
                let p0x = self.work_x[base + i - 1];
                let p0y = self.work_y[base + i - 1];
                let p1x = self.work_x[base + i];
                let p1y = self.work_y[base + i];
                let p2x = self.work_x[base + i + 1];
                let p2y = self.work_y[base + i + 1];
                self.vx[base + i] += kp * (p0x - p1x + p2x - p1x);
                self.vy[base + i] += kp * (p0y - p1y + p2y - p1y);
            }
        }

        // 2. Electrostatic attraction between compatible edge pairs.
        for pair in &self.edge_pairs {
            let base_p = (pair.p as usize) * pts;
            let base_q = (pair.q as usize) * pts;
            let c_e = pair.compatibility;
            for i in 0..num_mid {
                let ip = base_p + 1 + i;
                let iq = if pair.same_dir {
                    base_q + 1 + i
                } else {
                    base_q + num_mid - i
                };
                let dx = self.work_x[iq] - self.work_x[ip];
                let dy = self.work_y[iq] - self.work_y[ip];
                let dist = dx.hypot(dy);
                if dist > MIN_SEG_DIST {
                    let w = c_e / dist;
                    let fx = dx * w;
                    let fy = dy * w;
                    self.vx[ip] += fx;
                    self.vy[ip] += fy;
                    self.vx[iq] -= fx;
                    self.vy[iq] -= fy;
                }
            }
        }

        // 3. Explicit Euler integration of interior control points only (endpoints remain fixed).
        let alpha = self.alpha;
        for e in 0..m {
            let base = e * pts;
            for i in 1..=num_mid {
                let idx = base + i;
                let nx = self.work_x[idx] + alpha * self.vx[idx];
                let ny = self.work_y[idx] + alpha * self.vy[idx];
                if nx.is_finite() {
                    self.work_x[idx] = nx;
                }
                if ny.is_finite() {
                    self.work_y[idx] = ny;
                }
            }
        }

        for i in 0..self.work_x.len() {
            self.points_x[i] = S::from_f64(self.work_x[i]);
            self.points_y[i] = S::from_f64(self.work_y[i]);
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn edge_compatibility(
    p1x: f64,
    p1y: f64,
    p2x: f64,
    p2y: f64,
    p_norm: f64,
    q1x: f64,
    q1y: f64,
    q2x: f64,
    q2y: f64,
    q_norm: f64,
) -> (f64, f64) {
    let pdx = p2x - p1x;
    let pdy = p2y - p1y;
    let qdx = q2x - q1x;
    let qdy = q2y - q1y;

    let dot = pdx * qdx + pdy * qdy;
    let cos_theta = (dot / (p_norm * q_norm)).clamp(-1.0, 1.0);
    let c_a = cos_theta.abs();

    let l_avg = 0.5 * (p_norm + q_norm);
    let l_min = p_norm.min(q_norm);
    let l_max = p_norm.max(q_norm);
    let c_s = 2.0 / (l_avg / l_min + l_max / l_avg);

    let pmx = 0.5 * (p1x + p2x);
    let pmy = 0.5 * (p1y + p2y);
    let qmx = 0.5 * (q1x + q2x);
    let qmy = 0.5 * (q1y + q2y);
    let mid_dist = (qmx - pmx).hypot(qmy - pmy);
    let c_p = l_avg / (l_avg + mid_dist);

    let vp = visibility_compatibility(p1x, p1y, pdx, pdy, p_norm, pmx, pmy, q1x, q1y, q2x, q2y);
    let vq = visibility_compatibility(q1x, q1y, qdx, qdy, q_norm, qmx, qmy, p1x, p1y, p2x, p2y);
    let c_v = vp.min(vq);

    (c_a * c_s * c_p * c_v, cos_theta)
}

#[allow(clippy::too_many_arguments)]
fn visibility_compatibility(
    p1x: f64,
    p1y: f64,
    pdx: f64,
    pdy: f64,
    p_norm: f64,
    pmx: f64,
    pmy: f64,
    q1x: f64,
    q1y: f64,
    q2x: f64,
    q2y: f64,
) -> f64 {
    let p_norm_sq = p_norm * p_norm;
    let r0 = ((q1x - p1x) * pdx + (q1y - p1y) * pdy) / p_norm_sq;
    let r1 = ((q2x - p1x) * pdx + (q2y - p1y) * pdy) / p_norm_sq;

    let i0x = p1x + r0 * pdx;
    let i0y = p1y + r0 * pdy;
    let i1x = p1x + r1 * pdx;
    let i1y = p1y + r1 * pdy;

    let i_dist = (i1x - i0x).hypot(i1y - i0y);
    if i_dist <= MIN_EDGE_LEN {
        return 0.0;
    }
    let imx = 0.5 * (i0x + i1x);
    let imy = 0.5 * (i0y + i1y);
    let m_dist = (imx - pmx).hypot(imy - pmy);
    (1.0 - 2.0 * m_dist / i_dist).max(0.0)
}
