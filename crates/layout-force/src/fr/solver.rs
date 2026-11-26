use egraph_core::{
    CoreError, CoreResult, CsrGraph, DeterministicRng, IterativeSolver, Layout2d, Scalar,
    SolverStatus,
};
use egraph_spatial_trees::ArenaQuadtree2d;

use super::config::{FrBackend, FruchtermanReingoldConfig};
use super::helpers::{GridCell, GridEntry, clamp_cell_coord, extract_canonical_undirected_edges};
use super::nd::FrNdScratch;

pub(super) const COINCIDENT_DIST_SQ: f64 = 1e-24;
pub(super) const COINCIDENT_PERTURB_DIST: f64 = 1e-6;
pub(super) const BH_EPS_SQ: f64 = 1e-12;

/// Cooperative Fruchterman–Reingold (1991) layout solver implementing [`IterativeSolver`].
#[derive(Debug, Clone)]
pub struct FruchtermanReingoldSolver<S: Scalar = f64> {
    pub(super) config: FruchtermanReingoldConfig,
    pub(super) num_nodes: usize,
    pub(super) edges_u: Vec<u32>,
    pub(super) edges_v: Vec<u32>,
    pub(super) edges_w: Vec<f64>,
    pub(super) k: f64,
    initial_temperature: f64,
    pub(super) x_buf: Vec<f64>,
    pub(super) y_buf: Vec<f64>,
    pub(super) disp_x: Vec<f64>,
    pub(super) disp_y: Vec<f64>,
    pub(super) rep_x: Vec<f64>,
    pub(super) rep_y: Vec<f64>,
    pub(super) quadtree: ArenaQuadtree2d<f64>,
    grid_entries: Vec<GridEntry>,
    grid_cells: Vec<GridCell>,
    pub(super) active_dims: usize,
    pub(super) nd_scratch: Option<FrNdScratch>,
    pub(super) rng: DeterministicRng,
    pub(super) iter: usize,
    pub(super) status: SolverStatus,
    _marker: std::marker::PhantomData<S>,
}

impl<S: Scalar> FruchtermanReingoldSolver<S> {
    /// Constructs a [`FruchtermanReingoldSolver`] from a [`CsrGraph`] and [`FruchtermanReingoldConfig`].
    pub fn new<W: Scalar>(
        graph: &CsrGraph<W>,
        config: FruchtermanReingoldConfig,
    ) -> CoreResult<Self> {
        config.validate()?;
        let (n, edges_u, edges_v, edges_w) = extract_canonical_undirected_edges(graph)?;

        let n_eff = n.max(1) as f64;
        let effective_area = config.area.unwrap_or(if let Some(kv) = config.k {
            kv * kv * n_eff
        } else {
            n_eff
        });
        let k = config.k.unwrap_or_else(|| (effective_area / n_eff).sqrt());
        let initial_temperature = config
            .initial_temperature
            .unwrap_or_else(|| 0.1 * effective_area.sqrt());

        let rep_len = if config.backend == FrBackend::BarnesHut {
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
            k,
            initial_temperature,
            x_buf: vec![0.0; n],
            y_buf: vec![0.0; n],
            disp_x: vec![0.0; n],
            disp_y: vec![0.0; n],
            rep_x: vec![0.0; rep_len],
            rep_y: vec![0.0; rep_len],
            quadtree: ArenaQuadtree2d::new(),
            grid_entries: Vec::new(),
            grid_cells: Vec::new(),
            active_dims: 0,
            nd_scratch: None,
            rng: DeterministicRng::from_seed_u64(config.seed),
            iter: 0,
            status,
            _marker: std::marker::PhantomData,
        })
    }

    /// Returns the solver configuration.
    #[inline]
    pub fn config(&self) -> &FruchtermanReingoldConfig {
        &self.config
    }

    /// Returns the expected number of vertices in [`Layout2d`] or [`egraph_core::LayoutNd`].
    #[inline]
    pub fn num_nodes(&self) -> usize {
        self.num_nodes
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
        self.config.iterations
    }

    /// Returns the effective optimal pairwise distance $k$.
    #[inline]
    pub fn optimal_distance(&self) -> f64 {
        self.k
    }

    /// Returns the temperature used for the upcoming (or final) iteration.
    #[inline]
    pub fn current_temperature(&self) -> f64 {
        if self.config.iterations == 0 || self.iter >= self.config.iterations {
            return 0.0;
        }
        let rem = (self.config.iterations - self.iter) as f64;
        self.initial_temperature * (rem / (self.config.iterations as f64))
    }

    fn next_unit_vector(&mut self) -> (f64, f64) {
        let angle = self.rng.next_f64() * std::f64::consts::TAU;
        (angle.cos(), angle.sin())
    }

    fn apply_repulsion_pair(&mut self, i: usize, j: usize, k_sq: f64, cutoff_sq: Option<f64>) {
        let dx = self.x_buf[i] - self.x_buf[j];
        let dy = self.y_buf[i] - self.y_buf[j];
        let d_sq = dx * dx + dy * dy;
        if let Some(limit_sq) = cutoff_sq
            && d_sq >= limit_sq
        {
            return;
        }
        let (fx, fy) = if d_sq > COINCIDENT_DIST_SQ {
            let mult = k_sq / d_sq;
            (mult * dx, mult * dy)
        } else {
            let (ux, uy) = self.next_unit_vector();
            let mag = k_sq / COINCIDENT_PERTURB_DIST;
            (mag * ux, mag * uy)
        };
        self.disp_x[i] += fx;
        self.disp_y[i] += fy;
        self.disp_x[j] -= fx;
        self.disp_y[j] -= fy;
    }

    pub(super) fn compute_grid_repulsion(&mut self, min_x: f64, min_y: f64, k_sq: f64) {
        let n = self.num_nodes;
        let cell_size = 2.0 * self.k;
        let inv_cell = 1.0 / cell_size;
        let cutoff_sq = cell_size * cell_size;

        self.grid_entries.clear();
        self.grid_entries.reserve(n);
        for i in 0..n {
            let gx = ((self.x_buf[i] - min_x) * inv_cell).floor();
            let gy = ((self.y_buf[i] - min_y) * inv_cell).floor();
            let cx = clamp_cell_coord(gx);
            let cy = clamp_cell_coord(gy);
            self.grid_entries.push(GridEntry { cy, cx, node: i });
        }
        self.grid_entries
            .sort_unstable_by_key(|e| (e.cy, e.cx, e.node));

        self.grid_cells.clear();
        let mut idx = 0usize;
        while idx < n {
            let cy = self.grid_entries[idx].cy;
            let cx = self.grid_entries[idx].cx;
            let start = idx;
            idx += 1;
            while idx < n && self.grid_entries[idx].cy == cy && self.grid_entries[idx].cx == cx {
                idx += 1;
            }
            self.grid_cells.push(GridCell {
                cy,
                cx,
                start,
                end: idx,
            });
        }

        const HALF_NEIGHBOR_OFFSETS: [(i64, i64); 4] = [(0, 1), (1, -1), (1, 0), (1, 1)];
        let num_cells = self.grid_cells.len();
        for c_idx in 0..num_cells {
            let cell = self.grid_cells[c_idx];
            // Pairs within the same cell.
            for p in cell.start..cell.end {
                let u = self.grid_entries[p].node;
                for q in (p + 1)..cell.end {
                    let v = self.grid_entries[q].node;
                    self.apply_repulsion_pair(u, v, k_sq, Some(cutoff_sq));
                }
            }

            // Pairs with forward neighboring cells (each pair visited once).
            for &(dy, dx) in &HALF_NEIGHBOR_OFFSETS {
                let target_key = (cell.cy + dy, cell.cx + dx);
                if let Ok(nb_idx) = self
                    .grid_cells
                    .binary_search_by_key(&target_key, |c| (c.cy, c.cx))
                {
                    let nb = self.grid_cells[nb_idx];
                    for p in cell.start..cell.end {
                        let u = self.grid_entries[p].node;
                        for q in nb.start..nb.end {
                            let v = self.grid_entries[q].node;
                            self.apply_repulsion_pair(u, v, k_sq, Some(cutoff_sq));
                        }
                    }
                }
            }
        }
    }
}

impl<S: Scalar> IterativeSolver<S> for FruchtermanReingoldSolver<S> {
    fn step(&mut self, layout: &mut Layout2d<S>) -> CoreResult<SolverStatus> {
        if layout.len() != self.num_nodes {
            return Err(CoreError::dimension_mismatch(
                self.num_nodes,
                layout.len(),
                "FruchtermanReingoldSolver::step",
            ));
        }
        layout.validate_finite()?;
        if self.active_dims == 0 {
            self.active_dims = 2;
        } else if self.active_dims != 2 {
            return Err(CoreError::dimension_mismatch(
                self.active_dims,
                2,
                "FruchtermanReingoldSolver::step::dims",
            ));
        }

        if self.status.is_finished() {
            return Ok(self.status);
        }

        let n = self.num_nodes;
        if n <= 1 {
            self.status = SolverStatus::Converged;
            return Ok(self.status);
        }

        let (min_x, min_y) = {
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
                let spread = 0.5 * self.k;
                min_x = f64::INFINITY;
                min_y = f64::INFINITY;
                for i in 0..n {
                    self.x_buf[i] += spread * (self.rng.next_f64() - 0.5);
                    self.y_buf[i] += spread * (self.rng.next_f64() - 0.5);
                    min_x = min_x.min(self.x_buf[i]);
                    min_y = min_y.min(self.y_buf[i]);
                }
            }
            (min_x, min_y)
        };

        self.disp_x.fill(0.0);
        self.disp_y.fill(0.0);
        let k_sq = self.k * self.k;

        // 1. Repulsion: f_r(d) = k^2 / d.
        match self.config.backend {
            FrBackend::Exact => {
                for i in 0..n {
                    for j in (i + 1)..n {
                        self.apply_repulsion_pair(i, j, k_sq, None);
                    }
                }
            }
            FrBackend::BarnesHut => {
                self.quadtree.build_xy(&self.x_buf, &self.y_buf)?;
                self.quadtree.compute_coulomb_repulsion(
                    self.config.theta,
                    BH_EPS_SQ,
                    &mut self.rep_x,
                    &mut self.rep_y,
                )?;
                for i in 0..n {
                    self.disp_x[i] = k_sq * self.rep_x[i];
                    self.disp_y[i] = k_sq * self.rep_y[i];
                }
            }
            FrBackend::Grid => {
                self.compute_grid_repulsion(min_x, min_y, k_sq);
            }
        }

        // 2. Attraction: f_a(d) = d^2 / k along edges (scaled by edge weight).
        let inv_k = 1.0 / self.k;
        for idx in 0..self.edges_u.len() {
            let u = self.edges_u[idx] as usize;
            let v = self.edges_v[idx] as usize;
            let w = self.edges_w[idx];
            if w == 0.0 {
                continue;
            }
            let dx = self.x_buf[v] - self.x_buf[u];
            let dy = self.y_buf[v] - self.y_buf[u];
            let d = dx.hypot(dy);
            let factor = w * d * inv_k;
            let ax = factor * dx;
            let ay = factor * dy;
            self.disp_x[u] += ax;
            self.disp_y[u] += ay;
            self.disp_x[v] -= ax;
            self.disp_y[v] -= ay;
        }

        // 3. Optional central gravity toward (0, 0).
        if self.config.gravity > 0.0 {
            let g = self.config.gravity;
            for i in 0..n {
                self.disp_x[i] -= g * self.x_buf[i];
                self.disp_y[i] -= g * self.y_buf[i];
            }
        }

        // 4. Temperature clamping and linear cooling update.
        let temp = self.current_temperature();
        let (xs_mut, ys_mut) = layout.xy_mut();
        for i in 0..n {
            let dx = self.disp_x[i];
            let dy = self.disp_y[i];
            let len = dx.hypot(dy);
            if len > 1e-24 && temp > 0.0 {
                let scale = len.min(temp) / len;
                let nx = S::from_f64(self.x_buf[i] + dx * scale);
                let ny = S::from_f64(self.y_buf[i] + dy * scale);
                if nx.is_finite() {
                    xs_mut[i] = nx;
                }
                if ny.is_finite() {
                    ys_mut[i] = ny;
                }
            }
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
