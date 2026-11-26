use egraph_core::{CoreError, CoreResult, CsrGraph, Scalar, SolverStatus};

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
    pub(super) num_nodes: usize,
    pub(super) num_edges: usize,
    pub(super) max_total_pts: usize,
    pub(super) initialized_polylines: bool,
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
    pub(super) active_dims: usize,
    pub(super) nd_work: Vec<f64>,
    pub(super) nd_scratch: Vec<f64>,
    pub(super) nd_vel: Vec<f64>,
    pub(super) points_nd: Vec<S>,
    pub(super) status: SolverStatus,
}

impl<S: Scalar> FdebSolver<S> {
    /// Constructs an [`FdebSolver`] from a [`CsrGraph`] and [`FdebConfig`].
    ///
    /// Edge polylines and pairwise compatibilities are lazily initialized on the first
    /// [`egraph_core::IterativeSolver::step`] call against the supplied `Layout2d`.
    pub fn new<W: Scalar>(graph: &CsrGraph<W>, config: FdebConfig) -> CoreResult<Self> {
        config.validate()?;
        let n = graph.num_nodes();

        let mut edges = Vec::new();
        let directed = graph.is_directed();
        for u in 0..n {
            let u32_u = u as u32;
            for &v in graph.neighbors(u) {
                if directed || u32_u < v {
                    edges.push(EdgeEndpoints {
                        source: u32_u,
                        target: v,
                        length: 0.0,
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
        let mut cur_iter = config.steps_per_cycle;
        let mut total_iterations = 0usize;
        for _ in 0..config.cycles {
            cycle_iters.push(cur_iter);
            total_iterations = total_iterations.saturating_add(cur_iter);
            cur_iter = ((cur_iter as f64) * config.i_step) as usize;
        }

        let status = if total_iterations == 0 {
            SolverStatus::Completed
        } else {
            SolverStatus::Initialized
        };

        Ok(Self {
            config,
            num_nodes: n,
            num_edges,
            max_total_pts,
            initialized_polylines: false,
            edges,
            edge_pairs: Vec::new(),
            cycle_iters,
            total_iterations,
            current_cycle: 0,
            cycle_iter: 0,
            iter: 0,
            alpha: config.s0,
            pts_per_edge: 2,
            edge_offsets: Vec::new(),
            work_x: Vec::new(),
            work_y: Vec::new(),
            scratch_x: Vec::new(),
            scratch_y: Vec::new(),
            vx: Vec::new(),
            vy: Vec::new(),
            points_x: Vec::new(),
            points_y: Vec::new(),
            active_dims: 0,
            nd_work: Vec::new(),
            nd_scratch: Vec::new(),
            nd_vel: Vec::new(),
            points_nd: Vec::new(),
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

    /// Returns the scheduled maximum number of iterations.
    #[inline]
    pub fn total_iterations(&self) -> usize {
        self.total_iterations
    }

    /// Returns `true` if the solver has finished (`Converged` or `Completed`).
    #[inline]
    pub fn is_finished(&self) -> bool {
        self.status.is_finished()
    }

    /// Returns the expected number of vertices in [`egraph_core::Layout2d`].
    #[inline]
    pub fn num_nodes(&self) -> usize {
        self.num_nodes
    }

    /// Returns the number of edges being bundled.
    #[inline]
    pub fn num_edges(&self) -> usize {
        self.num_edges
    }

    /// Returns the CSR-style polyline offsets of length `num_edges() + 1` once initialized.
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
