use egraph_core::{CoreError, CoreResult, CsrGraph, Layout2d, Scalar, SolverStatus};

use super::MIN_EDGE_LEN;
use super::compatibility::edge_compatibility;
use super::config::FdebConfig;

#[derive(Debug, Clone, Copy)]
pub(super) struct EdgeEndpoints {
    pub(super) source: u32,
    pub(super) target: u32,
    pub(super) length: f64,
}

#[derive(Debug, Clone, Copy)]
pub(super) struct EdgePair {
    pub(super) p: u32,
    pub(super) q: u32,
    pub(super) compatibility: f64,
    pub(super) same_dir: bool,
}

/// Cooperative Force-Directed Edge Bundling (`FDEB`) solver storing contiguous flat edge
/// polyline buffers `(edge_offsets, points_x, points_y)`.
#[derive(Debug, Clone)]
pub struct FdebSolver<S: Scalar = f64> {
    pub(super) config: FdebConfig,
    pub(super) num_edges: usize,
    pub(super) edges: Vec<EdgeEndpoints>,
    pub(super) edge_pairs: Vec<EdgePair>,
    pub(super) cycle_iters: Vec<usize>,
    pub(super) total_iterations: usize,
    pub(super) current_cycle: usize,
    pub(super) cycle_iter: usize,
    pub(super) iter: usize,
    pub(super) alpha: f64,
    pub(super) pts_per_edge: usize,
    pub(super) edge_offsets: Vec<u32>,
    pub(super) work_x: Vec<f64>,
    pub(super) work_y: Vec<f64>,
    pub(super) scratch_x: Vec<f64>,
    pub(super) scratch_y: Vec<f64>,
    pub(super) vx: Vec<f64>,
    pub(super) vy: Vec<f64>,
    pub(super) points_x: Vec<S>,
    pub(super) points_y: Vec<S>,
    pub(super) status: SolverStatus,
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
}
