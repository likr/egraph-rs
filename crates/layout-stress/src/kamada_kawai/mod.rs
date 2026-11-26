//! Pure-Rust Kamada-Kawai (1989) spring layout solver with `|det| <= 1e-12` Hessian
//! singularity guard and bounded inner/outer iteration limits.

#[cfg(test)]
mod tests;

use egraph_core::{
    CoreError, CoreResult, CsrGraph, IterativeSolver, Layout2d, Scalar, SolverStatus,
};
use egraph_linalg::{BfsWorkspace, DijkstraWorkspace};

const HESSIAN_DET_TOL: f64 = 1e-12;
const MIN_SEPARATION: f64 = 1e-12;
const MAX_NEWTON_STEP: f64 = 1e6;

/// Hyperparameters for [`KamadaKawaiSolver`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct KamadaKawaiConfig {
    /// Outer gradient-norm convergence threshold $\varepsilon > 0$ (`default: 1e-2`).
    pub eps: f64,
    /// Maximum Newton-Raphson inner iterations per selected node (`default: 50`).
    pub max_inner_iters: usize,
    /// Optional cap on outer node-selection steps (`None` uses `(10 * n).max(100)`).
    pub max_outer_iters: Option<usize>,
}

impl Default for KamadaKawaiConfig {
    fn default() -> Self {
        Self {
            eps: 1e-2,
            max_inner_iters: 50,
            max_outer_iters: None,
        }
    }
}

impl KamadaKawaiConfig {
    /// Validates Kamada-Kawai hyperparameters.
    pub fn validate(&self) -> CoreResult<()> {
        if !self.eps.is_finite() {
            return Err(CoreError::non_finite("KamadaKawaiConfig::eps"));
        }
        if self.eps <= 0.0 {
            return Err(CoreError::invalid_input(format!(
                "eps must be > 0.0, got {}",
                self.eps
            )));
        }
        if self.max_inner_iters == 0 {
            return Err(CoreError::invalid_input("max_inner_iters must be >= 1"));
        }
        Ok(())
    }
}

/// Cooperative pure-Rust Kamada-Kawai (1989) layout solver implementing [`IterativeSolver`].
///
/// Each [`IterativeSolver::step`] selects the vertex with largest gradient norm and moves it
/// via Newton-Raphson (up to `max_inner_iters`), guarding `|det| <= 1e-12` Hessian
/// singularities with a bounded steepest-descent fallback so degenerate or collinear
/// layouts can never produce `NaN`s or infinite loops.
#[derive(Debug, Clone)]
pub struct KamadaKawaiSolver<S: Scalar = f64> {
    config: KamadaKawaiConfig,
    num_nodes: usize,
    max_outer_iters: usize,
    l: Vec<f64>,
    k: Vec<f64>,
    x_buf: Vec<f64>,
    y_buf: Vec<f64>,
    iter: usize,
    status: SolverStatus,
    _marker: std::marker::PhantomData<S>,
}

impl<S: Scalar> KamadaKawaiSolver<S> {
    /// Constructs a [`KamadaKawaiSolver`] from a [`CsrGraph`] using [`BfsWorkspace`]
    /// (unweighted) or [`DijkstraWorkspace`] (weighted).
    pub fn from_graph<W: Scalar>(
        graph: &CsrGraph<W>,
        config: KamadaKawaiConfig,
    ) -> CoreResult<Self> {
        config.validate()?;
        if let Some(ws) = graph.weights() {
            for &w in ws {
                let wf = w.to_f64();
                if !wf.is_finite() {
                    return Err(CoreError::non_finite(
                        "KamadaKawaiSolver::from_graph::weights",
                    ));
                }
                if wf < 0.0 {
                    return Err(CoreError::invalid_input(
                        "graph edge weights must be non-negative",
                    ));
                }
            }
        }

        let n = graph.num_nodes();
        let total = n
            .checked_mul(n)
            .ok_or_else(|| CoreError::invalid_input("APSP distance matrix overflow"))?;
        let mut dists_flat = vec![f64::INFINITY; total];

        if n > 0 {
            let mut bfs = (!graph.is_weighted()).then(|| BfsWorkspace::new(n));
            let mut dijk = graph.is_weighted().then(|| DijkstraWorkspace::new(n));
            for u in 0..n {
                let row = &mut dists_flat[u * n..(u + 1) * n];
                if let Some(ws) = bfs.as_mut() {
                    ws.bfs_from(graph, u, row)?;
                } else if let Some(ws) = dijk.as_mut() {
                    ws.dijkstra_from(graph, u, row)?;
                }
            }
        }

        Self::from_distance_matrix(n, &dists_flat, config)
    }

    /// Constructs a [`KamadaKawaiSolver`] from a flat row-major `(n, n)` distance matrix.
    ///
    /// Positive `f64::INFINITY` denotes unreachable node pairs (`k_ij = 0.0`); `NaN` and
    /// negative values are rejected.
    pub fn from_distance_matrix(
        n: usize,
        dists_flat: &[f64],
        config: KamadaKawaiConfig,
    ) -> CoreResult<Self> {
        config.validate()?;
        let expected = n
            .checked_mul(n)
            .ok_or_else(|| CoreError::invalid_input("distance matrix dimension overflow"))?;
        if dists_flat.len() != expected {
            return Err(CoreError::dimension_mismatch(
                expected,
                dists_flat.len(),
                "KamadaKawaiSolver::from_distance_matrix",
            ));
        }

        for &d in dists_flat {
            if d.is_nan() {
                return Err(CoreError::non_finite(
                    "KamadaKawaiSolver::from_distance_matrix",
                ));
            }
            if d < 0.0 {
                return Err(CoreError::invalid_input(format!(
                    "distance matrix entries must be >= 0.0, got {d}"
                )));
            }
        }

        let mut l = vec![0.0_f64; expected];
        let mut k = vec![0.0_f64; expected];
        for i in 0..n {
            for j in (i + 1)..n {
                let dij_raw = dists_flat[i * n + j];
                let dji_raw = dists_flat[j * n + i];
                let ok_ij = dij_raw.is_finite() && dij_raw > 0.0;
                let ok_ji = dji_raw.is_finite() && dji_raw > 0.0;
                let d = match (ok_ij, ok_ji) {
                    (true, true) => 0.5 * (dij_raw + dji_raw),
                    (true, false) => dij_raw,
                    (false, true) => dji_raw,
                    (false, false) => f64::INFINITY,
                };
                if d.is_finite() && d > 0.0 {
                    let k_ij = 1.0 / (d * d);
                    if k_ij.is_finite() && k_ij > 0.0 {
                        l[i * n + j] = d;
                        l[j * n + i] = d;
                        k[i * n + j] = k_ij;
                        k[j * n + i] = k_ij;
                    }
                }
            }
        }

        let max_outer_iters = config
            .max_outer_iters
            .unwrap_or_else(|| n.saturating_mul(10).max(100));
        let status = if max_outer_iters == 0 {
            SolverStatus::Completed
        } else {
            SolverStatus::Initialized
        };

        Ok(Self {
            config,
            num_nodes: n,
            max_outer_iters,
            l,
            k,
            x_buf: vec![0.0; n],
            y_buf: vec![0.0; n],
            iter: 0,
            status,
            _marker: std::marker::PhantomData,
        })
    }

    /// Returns the solver configuration.
    #[inline]
    pub fn config(&self) -> &KamadaKawaiConfig {
        &self.config
    }

    /// Returns the expected number of vertices in [`Layout2d`].
    #[inline]
    pub fn num_nodes(&self) -> usize {
        self.num_nodes
    }

    fn select_node(&self) -> Option<(usize, f64)> {
        let n = self.num_nodes;
        let eps_sq = self.config.eps * self.config.eps;
        let mut best_m = 0usize;
        let mut best_delta2 = 0.0_f64;

        for m in 0..n {
            let xm = self.x_buf[m];
            let ym = self.y_buf[m];
            let row = m * n;
            let mut dedx = 0.0_f64;
            let mut dedy = 0.0_f64;
            for i in 0..n {
                if i == m {
                    continue;
                }
                let k_mi = self.k[row + i];
                if k_mi <= 0.0 {
                    continue;
                }
                let dx = xm - self.x_buf[i];
                let dy = ym - self.y_buf[i];
                let d = dx.hypot(dy);
                if d > MIN_SEPARATION && d.is_finite() {
                    let f = k_mi * (1.0 - self.l[row + i] / d);
                    dedx += f * dx;
                    dedy += f * dy;
                }
            }
            let delta2 = dedx * dedx + dedy * dedy;
            if delta2.is_finite() && delta2 > best_delta2 {
                best_delta2 = delta2;
                best_m = m;
            }
        }

        if best_delta2 < eps_sq {
            None
        } else {
            Some((best_m, best_delta2))
        }
    }
}

impl<S: Scalar> IterativeSolver<S> for KamadaKawaiSolver<S> {
    fn step(&mut self, layout: &mut Layout2d<S>) -> CoreResult<SolverStatus> {
        if layout.len() != self.num_nodes {
            return Err(CoreError::dimension_mismatch(
                self.num_nodes,
                layout.len(),
                "KamadaKawaiSolver::step",
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
            for i in 0..n {
                self.x_buf[i] = xs[i].to_f64();
                self.y_buf[i] = ys[i].to_f64();
            }
        }

        let Some((m, _)) = self.select_node() else {
            self.status = SolverStatus::Converged;
            return Ok(self.status);
        };

        let eps_sq = self.config.eps * self.config.eps;
        let row = m * n;
        let mut xm = self.x_buf[m];
        let mut ym = self.y_buf[m];

        for _ in 0..self.config.max_inner_iters {
            let mut hxx = 0.0_f64;
            let mut hyy = 0.0_f64;
            let mut hxy = 0.0_f64;
            let mut dedx = 0.0_f64;
            let mut dedy = 0.0_f64;

            for i in 0..n {
                if i == m {
                    continue;
                }
                let k_mi = self.k[row + i];
                if k_mi <= 0.0 {
                    continue;
                }
                let l_mi = self.l[row + i];
                let dx = xm - self.x_buf[i];
                let dy = ym - self.y_buf[i];
                let d = dx.hypot(dy);
                if d <= MIN_SEPARATION || !d.is_finite() {
                    continue;
                }
                let d3 = d * d * d;
                if d3 > 0.0 && d3.is_finite() {
                    hxx += k_mi * (1.0 - l_mi * dy * dy / d3);
                    hyy += k_mi * (1.0 - l_mi * dx * dx / d3);
                    hxy += k_mi * l_mi * dx * dy / d3;
                }
                let f = k_mi * (1.0 - l_mi / d);
                dedx += f * dx;
                dedy += f * dy;
            }

            let delta2 = dedx * dedx + dedy * dedy;
            if !delta2.is_finite() || delta2 < eps_sq {
                break;
            }

            let det = hxx * hyy - hxy * hxy;
            if !det.is_finite() || det.abs() <= HESSIAN_DET_TOL {
                // Singular or near-singular Hessian: take one bounded steepest-descent fallback step
                // and exit the inner Newton loop cleanly so degenerate configurations never loop.
                let g_norm = delta2.sqrt();
                if g_norm > 0.0 && g_norm.is_finite() {
                    let step_len = (0.1 / g_norm).min(0.25);
                    let cand_x = xm - step_len * dedx;
                    let cand_y = ym - step_len * dedy;
                    if cand_x.is_finite() && cand_y.is_finite() {
                        xm = cand_x;
                        ym = cand_y;
                    }
                }
                break;
            }

            let mut delta_x = (hyy * dedx - hxy * dedy) / det;
            let mut delta_y = (hxx * dedy - hxy * dedx) / det;
            if !delta_x.is_finite() || !delta_y.is_finite() {
                break;
            }
            let step_norm = delta_x.hypot(delta_y);
            if step_norm > MAX_NEWTON_STEP {
                let scale = MAX_NEWTON_STEP / step_norm;
                delta_x *= scale;
                delta_y *= scale;
            }

            let cand_x = xm - delta_x;
            let cand_y = ym - delta_y;
            if !cand_x.is_finite() || !cand_y.is_finite() {
                break;
            }
            xm = cand_x;
            ym = cand_y;

            if step_norm <= self.config.eps * 1e-2 {
                break;
            }
        }

        let new_xm = S::from_f64(xm);
        let new_ym = S::from_f64(ym);
        if new_xm.is_finite() {
            layout.x_mut()[m] = new_xm;
        }
        if new_ym.is_finite() {
            layout.y_mut()[m] = new_ym;
        }

        self.iter += 1;
        self.status = if self.iter >= self.max_outer_iters {
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
        self.max_outer_iters
    }
}
