//! Fruchterman–Reingold (1991) force-directed graph layout solver.
//!
//! Implements the classic Fruchterman & Reingold (1991) algorithm with optimal distance
//! $k = \sqrt{\text{area}/N}$, quadratic edge attraction $f_a(d) = d^2/k$, inverse-distance
//! repulsion $f_r(d) = k^2/d$, temperature clamping with linear cooling, and three selectable
//! spatial repulsion backends:
//! - [`FrBackend::Exact`]: $O(N^2)$ all-pairs repulsion.
//! - [`FrBackend::BarnesHut`]: $O(N \log N)$ [`ArenaQuadtree2d`] Coulomb approximation.
//! - [`FrBackend::Grid`]: Fruchterman–Reingold (1991) Fig. 2 $2k \times 2k$ spatial grid with
//!   strict Euclidean cutoff $d < 2k$ ($f_r(d) = 0$ for $d \ge 2k$).

use egraph_core::{
    CoreError, CoreResult, CsrGraph, DeterministicRng, IterativeSolver, Layout2d, Scalar,
    SolverStatus,
};
use egraph_spatial::ArenaQuadtree2d;

const COINCIDENT_DIST_SQ: f64 = 1e-24;
const COINCIDENT_PERTURB_DIST: f64 = 1e-6;
const BH_EPS_SQ: f64 = 1e-12;

/// Spatial repulsion backend for [`FruchtermanReingoldSolver`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum FrBackend {
    /// Exact $O(N^2)$ all-pairs repulsion.
    #[default]
    Exact,
    /// $O(N \log N)$ Barnes-Hut quadtree approximation via [`ArenaQuadtree2d`].
    BarnesHut,
    /// Fruchterman–Reingold (1991) Fig. 2 $2k \times 2k$ grid with strict $d < 2k$ cutoff.
    Grid,
}

/// Hyperparameters for [`FruchtermanReingoldSolver`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FruchtermanReingoldConfig {
    /// Total number of iterations (`default: 200`).
    pub iterations: usize,
    /// Spatial repulsion backend (`default: FrBackend::Exact`).
    pub backend: FrBackend,
    /// Optional optimal pairwise distance $k$ (`None` uses $\sqrt{\text{area} / N}$, `default: None`).
    pub k: Option<f64>,
    /// Optional canvas area (`None` uses $N$ or $k^2 N$ if `k` is specified, `default: None`).
    pub area: Option<f64>,
    /// Optional initial temperature (`None` uses $0.1 \sqrt{\text{area}}$, `default: None`).
    pub initial_temperature: Option<f64>,
    /// Optional central gravity coefficient pulling vertices toward `(0, 0)` (`default: 0.0`).
    pub gravity: f64,
    /// Barnes-Hut opening angle threshold $\theta \ge 0$ (`default: 1.2`).
    pub theta: f64,
    /// Deterministic explicit RNG seed for symmetry breaking (`default: 42`).
    pub seed: u64,
}

impl Default for FruchtermanReingoldConfig {
    fn default() -> Self {
        Self {
            iterations: 200,
            backend: FrBackend::Exact,
            k: None,
            area: None,
            initial_temperature: None,
            gravity: 0.0,
            theta: 1.2,
            seed: 42,
        }
    }
}

impl FruchtermanReingoldConfig {
    /// Validates all Fruchterman–Reingold hyperparameters.
    pub fn validate(&self) -> CoreResult<()> {
        if let Some(k) = self.k
            && (!k.is_finite() || k <= 0.0)
        {
            return Err(CoreError::invalid_input(format!(
                "k must be finite and > 0.0, got {k}"
            )));
        }
        if let Some(area) = self.area
            && (!area.is_finite() || area <= 0.0)
        {
            return Err(CoreError::invalid_input(format!(
                "area must be finite and > 0.0, got {area}"
            )));
        }
        if let Some(t0) = self.initial_temperature
            && (!t0.is_finite() || t0 < 0.0)
        {
            return Err(CoreError::invalid_input(format!(
                "initial_temperature must be finite and >= 0.0, got {t0}"
            )));
        }
        if !self.gravity.is_finite() || self.gravity < 0.0 {
            return Err(CoreError::invalid_input(format!(
                "gravity must be finite and >= 0.0, got {}",
                self.gravity
            )));
        }
        if !self.theta.is_finite() || self.theta < 0.0 {
            return Err(CoreError::invalid_input(format!(
                "theta must be finite and >= 0.0, got {}",
                self.theta
            )));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy)]
struct GridEntry {
    cy: i64,
    cx: i64,
    node: usize,
}

#[derive(Debug, Clone, Copy)]
struct GridCell {
    cy: i64,
    cx: i64,
    start: usize,
    end: usize,
}

/// Cooperative Fruchterman–Reingold (1991) layout solver implementing [`IterativeSolver`].
#[derive(Debug, Clone)]
pub struct FruchtermanReingoldSolver<S: Scalar = f64> {
    config: FruchtermanReingoldConfig,
    num_nodes: usize,
    edges_u: Vec<u32>,
    edges_v: Vec<u32>,
    edges_w: Vec<f64>,
    k: f64,
    initial_temperature: f64,
    x_buf: Vec<f64>,
    y_buf: Vec<f64>,
    disp_x: Vec<f64>,
    disp_y: Vec<f64>,
    rep_x: Vec<f64>,
    rep_y: Vec<f64>,
    quadtree: ArenaQuadtree2d<f64>,
    grid_entries: Vec<GridEntry>,
    grid_cells: Vec<GridCell>,
    rng: DeterministicRng,
    iter: usize,
    status: SolverStatus,
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

    /// Returns the expected number of vertices in [`Layout2d`].
    #[inline]
    pub fn num_nodes(&self) -> usize {
        self.num_nodes
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

    fn compute_grid_repulsion(&mut self, min_x: f64, min_y: f64, k_sq: f64) {
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

#[inline]
fn clamp_cell_coord(v: f64) -> i64 {
    if !v.is_finite() {
        return 0;
    }
    let min_c = -1_000_000_000_000.0_f64;
    let max_c = 1_000_000_000_000.0_f64;
    v.clamp(min_c, max_c) as i64
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
                return Err(CoreError::non_finite("fr::graph_weights"));
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

    fn seeded_layout<S: Scalar>(n: usize, seed: u64) -> Layout2d<S> {
        let mut rng = DeterministicRng::from_seed_u64(seed);
        let mut x = Vec::with_capacity(n);
        let mut y = Vec::with_capacity(n);
        for _ in 0..n {
            x.push(S::from_f64(4.0 * (rng.next_f64() - 0.5)));
            y.push(S::from_f64(4.0 * (rng.next_f64() - 0.5)));
        }
        Layout2d::from_xy(x, y).unwrap()
    }

    #[test]
    fn test_fr_config_defaults_and_validation() {
        let cfg = FruchtermanReingoldConfig::default();
        assert_eq!(cfg.iterations, 200);
        assert_eq!(cfg.backend, FrBackend::Exact);
        assert_eq!(cfg.k, None);
        assert_eq!(cfg.area, None);
        assert_eq!(cfg.initial_temperature, None);
        assert_eq!(cfg.gravity, 0.0);
        assert_eq!(cfg.theta, 1.2);
        assert_eq!(cfg.seed, 42);
        cfg.validate().unwrap();

        for bad in [
            FruchtermanReingoldConfig {
                k: Some(0.0),
                ..cfg
            },
            FruchtermanReingoldConfig {
                k: Some(-1.0),
                ..cfg
            },
            FruchtermanReingoldConfig {
                k: Some(f64::NAN),
                ..cfg
            },
            FruchtermanReingoldConfig {
                area: Some(0.0),
                ..cfg
            },
            FruchtermanReingoldConfig {
                area: Some(f64::INFINITY),
                ..cfg
            },
            FruchtermanReingoldConfig {
                initial_temperature: Some(-0.1),
                ..cfg
            },
            FruchtermanReingoldConfig {
                initial_temperature: Some(f64::NAN),
                ..cfg
            },
            FruchtermanReingoldConfig {
                gravity: -0.1,
                ..cfg
            },
            FruchtermanReingoldConfig { theta: -0.1, ..cfg },
        ] {
            assert!(bad.validate().is_err());
        }
    }

    #[test]
    fn test_fr_backends_exact_barnes_hut_and_grid_cutoff() {
        let graph = grid_graph(4, 4);
        let init = seeded_layout::<f64>(16, 17);

        // 1. BarnesHut with theta=0.0 matches Exact to machine precision.
        let mut l_exact = init.clone();
        let mut l_bh0 = init.clone();
        FruchtermanReingoldSolver::new(
            &graph,
            FruchtermanReingoldConfig {
                iterations: 25,
                backend: FrBackend::Exact,
                ..Default::default()
            },
        )
        .unwrap()
        .run(&mut l_exact)
        .unwrap();
        FruchtermanReingoldSolver::new(
            &graph,
            FruchtermanReingoldConfig {
                iterations: 25,
                backend: FrBackend::BarnesHut,
                theta: 0.0,
                ..Default::default()
            },
        )
        .unwrap()
        .run(&mut l_bh0)
        .unwrap();
        for i in 0..16 {
            assert!((l_exact.x()[i] - l_bh0.x()[i]).abs() < 1e-8);
            assert!((l_exact.y()[i] - l_bh0.y()[i]).abs() < 1e-8);
        }

        // 2. Grid backend enforces strict Euclidean d < 2k cutoff (edgeless graph at d >= 2k stays unmoved,
        //    whereas d < 2k repels).
        let edgeless = CsrGraph::<f64>::from_edges(3, &[], None, false, false).unwrap();
        // Node 0 at (0, 0), Node 1 at (1.5, 0) (d = 1.5 < 2k=2.0), Node 2 at (2.1, 0) (d = 2.1 >= 2k=2.0 from 0, d = 0.6 < 2k from 1).
        // Put Node 2 far away at (10.0, 0) so it is >= 2k from both 0 and 1.
        let mut l_grid_cutoff =
            Layout2d::from_xy(vec![0.0, 1.5, 10.0], vec![0.0, 0.0, 0.0]).unwrap();
        let mut s_grid = FruchtermanReingoldSolver::new(
            &edgeless,
            FruchtermanReingoldConfig {
                iterations: 1,
                backend: FrBackend::Grid,
                k: Some(1.0),
                initial_temperature: Some(10.0),
                ..Default::default()
            },
        )
        .unwrap();
        s_grid.step(&mut l_grid_cutoff).unwrap();
        // Node 0 and Node 1 repel each other (d = 1.5 < 2k = 2.0), Node 2 experiences 0 repulsion (d >= 2k).
        assert!(l_grid_cutoff.x()[0] < 0.0);
        assert!(l_grid_cutoff.x()[1] > 1.5);
        assert_eq!(l_grid_cutoff.x()[2], 10.0);
        assert_eq!(l_grid_cutoff.y()[2], 0.0);

        // Also check diagonal corner inside adjacent grid cell with d >= 2k is strictly ignored:
        // k=1.0 -> 2k=2.0; (0.1, 0.1) in cell (0,0) and (1.9, 1.9) in cell (0,0) or (2.05, 0.05) -> d = sqrt(1.95^2 + 0.05^2) < 2.0 vs (2.1, 2.1) -> d = sqrt(4+4) >= 2.0.
        let mut l_diag = Layout2d::from_xy(vec![0.1, 1.9], vec![0.1, 1.9]).unwrap();
        let edgeless2 = CsrGraph::<f64>::from_edges(2, &[], None, false, false).unwrap();
        let mut s_diag = FruchtermanReingoldSolver::new(
            &edgeless2,
            FruchtermanReingoldConfig {
                iterations: 1,
                backend: FrBackend::Grid,
                k: Some(1.0),
                initial_temperature: Some(10.0),
                ..Default::default()
            },
        )
        .unwrap();
        s_diag.step(&mut l_diag).unwrap();
        // Same cell (0,0), but Euclidean distance = 1.8 * sqrt(2) ~= 2.5456 >= 2.0 -> zero force!
        assert_eq!(l_diag.x()[0], 0.1);
        assert_eq!(l_diag.x()[1], 1.9);

        // 3. All 3 backends produce clean finite layouts on a 4x4 grid.
        for backend in [FrBackend::Exact, FrBackend::BarnesHut, FrBackend::Grid] {
            let mut l = init.clone();
            let mut s = FruchtermanReingoldSolver::new(
                &graph,
                FruchtermanReingoldConfig {
                    iterations: 100,
                    backend,
                    gravity: 0.05,
                    ..Default::default()
                },
            )
            .unwrap();
            assert_eq!(s.run(&mut l).unwrap(), SolverStatus::Completed);
            assert!(l.is_all_finite());
        }
    }

    #[test]
    fn test_fr_step_vs_run_bit_exact_f64_f32_and_edge_cases() {
        let graph = grid_graph(3, 4);
        for backend in [FrBackend::Exact, FrBackend::BarnesHut, FrBackend::Grid] {
            let cfg = FruchtermanReingoldConfig {
                iterations: 20,
                backend,
                gravity: 0.02,
                ..Default::default()
            };

            // f64 bit-exactness
            let mut l64_r = seeded_layout::<f64>(12, 33);
            let mut l64_s = l64_r.clone();
            let mut s64_r = FruchtermanReingoldSolver::<f64>::new(&graph, cfg).unwrap();
            let mut s64_s = FruchtermanReingoldSolver::<f64>::new(&graph, cfg).unwrap();
            s64_r.run(&mut l64_r).unwrap();
            s64_s.run_steps(&mut l64_s, 7).unwrap();
            while !s64_s.status().is_finished() {
                s64_s.step(&mut l64_s).unwrap();
            }
            for i in 0..12 {
                assert_eq!(l64_r.x()[i].to_bits(), l64_s.x()[i].to_bits());
                assert_eq!(l64_r.y()[i].to_bits(), l64_s.y()[i].to_bits());
            }

            // f32 bit-exactness
            let mut l32_r = seeded_layout::<f32>(12, 33);
            let mut l32_s = l32_r.clone();
            let mut s32_r = FruchtermanReingoldSolver::<f32>::new(&graph, cfg).unwrap();
            let mut s32_s = FruchtermanReingoldSolver::<f32>::new(&graph, cfg).unwrap();
            s32_r.run(&mut l32_r).unwrap();
            while !s32_s.status().is_finished() {
                s32_s.step(&mut l32_s).unwrap();
            }
            for i in 0..12 {
                assert_eq!(l32_r.x()[i].to_bits(), l32_s.x()[i].to_bits());
                assert_eq!(l32_r.y()[i].to_bits(), l32_s.y()[i].to_bits());
            }
        }

        // Degenerate graphs: N=0, N=1, collapsed start, extreme coordinate span on Grid (no OOM!),
        // directed reciprocal edges, negative weight rejection, dimension/NaN errors.
        for n in [0usize, 1, 4] {
            let g = CsrGraph::<f64>::from_edges(n, &[], None, false, false).unwrap();
            let mut l = Layout2d::<f64>::zeros(n);
            let mut s =
                FruchtermanReingoldSolver::new(&g, FruchtermanReingoldConfig::default()).unwrap();
            assert!(s.run(&mut l).unwrap().is_finished());
            assert!(l.is_all_finite());
        }

        // Extreme coordinate span with Grid backend must not OOM or panic.
        let far_g = CsrGraph::<f64>::from_edges(2, &[(0, 1)], None, false, false).unwrap();
        let mut far_l = Layout2d::from_xy(vec![-1e9, 1e9], vec![-1e9, 1e9]).unwrap();
        let mut far_s = FruchtermanReingoldSolver::new(
            &far_g,
            FruchtermanReingoldConfig {
                iterations: 5,
                backend: FrBackend::Grid,
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(far_s.run(&mut far_l).unwrap(), SolverStatus::Completed);
        assert!(far_l.is_all_finite());

        let dir_g = CsrGraph::from_edges(
            4,
            &[(0, 1), (1, 0), (1, 2), (2, 3)],
            Some(&[2.0, 4.0, 1.5, 3.0]),
            true,
            false,
        )
        .unwrap();
        let mut l_dir = seeded_layout::<f64>(4, 8);
        assert!(
            FruchtermanReingoldSolver::new(&dir_g, FruchtermanReingoldConfig::default())
                .unwrap()
                .run(&mut l_dir)
                .unwrap()
                .is_finished()
        );
        assert!(l_dir.is_all_finite());

        let s_zero = FruchtermanReingoldSolver::<f64>::new(
            &far_g,
            FruchtermanReingoldConfig {
                iterations: 0,
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(s_zero.status(), SolverStatus::Completed);
        assert_eq!(s_zero.current_temperature(), 0.0);

        let neg_g = CsrGraph::from_edges(2, &[(0, 1)], Some(&[-1.0]), false, false).unwrap();
        assert!(matches!(
            FruchtermanReingoldSolver::<f64>::new(&neg_g, FruchtermanReingoldConfig::default()),
            Err(CoreError::InvalidInput(_))
        ));

        let mut s_err =
            FruchtermanReingoldSolver::<f64>::new(&far_g, FruchtermanReingoldConfig::default())
                .unwrap();
        let mut wrong = Layout2d::<f64>::zeros(5);
        assert!(matches!(
            s_err.step(&mut wrong),
            Err(CoreError::DimensionMismatch { .. })
        ));
        let mut bad = Layout2d::from_xy(vec![0.0, f64::NAN], vec![0.0, 0.0]).unwrap();
        assert!(matches!(
            s_err.step(&mut bad),
            Err(CoreError::NonFiniteValue { .. })
        ));
    }
}
