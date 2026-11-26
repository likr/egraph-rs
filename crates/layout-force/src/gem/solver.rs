use egraph_core::{
    CoreError, CoreResult, CsrGraph, DeterministicRng, IterativeSolver, Layout2d, Scalar,
    SolverStatus,
};

use super::config::GemConfig;

const MIN_DIST_SQ: f64 = 1e-12;
const MIN_NORM: f64 = 1e-18;

/// Cooperative GEM (Frick et al., 1995) layout solver implementing [`IterativeSolver`].
#[derive(Debug, Clone)]
pub struct GemSolver<S: Scalar = f64> {
    config: GemConfig,
    num_nodes: usize,
    max_rounds: usize,
    sigma_r: f64,
    cos_osc_thresh: f64,
    sin_rot_thresh: f64,
    adj_indptr: Vec<usize>,
    adj_indices: Vec<usize>,
    adj_weights: Vec<f64>,
    phi: Vec<f64>,
    x_buf: Vec<f64>,
    y_buf: Vec<f64>,
    imp_x: Vec<f64>,
    imp_y: Vec<f64>,
    dir: Vec<f64>,
    pub(super) temp: Vec<f64>,
    order: Vec<u32>,
    rng: DeterministicRng,
    iter: usize,
    status: SolverStatus,
    _marker: std::marker::PhantomData<S>,
}

impl<S: Scalar> GemSolver<S> {
    /// Constructs a [`GemSolver`] from a [`CsrGraph`] and [`GemConfig`].
    pub fn new<W: Scalar>(graph: &CsrGraph<W>, config: GemConfig) -> CoreResult<Self> {
        config.validate()?;
        let (n, adj_indptr, adj_indices, adj_weights) = extract_undirected_adj(graph)?;

        let max_rounds = config
            .max_rounds
            .unwrap_or_else(|| n.saturating_mul(4).max(1));
        let sigma_r = config
            .rotation_sensitivity
            .unwrap_or_else(|| 1.0 / (2.0 * (n.max(1) as f64)));
        let cos_osc_thresh = (0.5 * config.oscillation_angle).cos();
        let sin_rot_thresh = (0.5 * std::f64::consts::PI + 0.5 * config.rotation_angle).sin();

        let mut phi = vec![1.0_f64; n];
        for (u, phi_u) in phi.iter_mut().enumerate() {
            let deg = adj_indptr[u + 1] - adj_indptr[u];
            *phi_u = 1.0 + 0.5 * (deg as f64);
        }

        let t0 = config.t_init.min(config.t_max);
        let status = if max_rounds == 0 {
            SolverStatus::Completed
        } else {
            SolverStatus::Initialized
        };

        Ok(Self {
            config,
            num_nodes: n,
            max_rounds,
            sigma_r,
            cos_osc_thresh,
            sin_rot_thresh,
            adj_indptr,
            adj_indices,
            adj_weights,
            phi,
            x_buf: vec![0.0; n],
            y_buf: vec![0.0; n],
            imp_x: vec![0.0; n],
            imp_y: vec![0.0; n],
            dir: vec![0.0; n],
            temp: vec![t0; n],
            order: (0..n as u32).collect(),
            rng: DeterministicRng::from_seed_u64(config.seed),
            iter: 0,
            status,
            _marker: std::marker::PhantomData,
        })
    }

    /// Returns the solver configuration.
    #[inline]
    pub fn config(&self) -> &GemConfig {
        &self.config
    }

    /// Returns the expected number of vertices in [`Layout2d`].
    #[inline]
    pub fn num_nodes(&self) -> usize {
        self.num_nodes
    }
}

impl<S: Scalar> IterativeSolver<S> for GemSolver<S> {
    fn step(&mut self, layout: &mut Layout2d<S>) -> CoreResult<SolverStatus> {
        if layout.len() != self.num_nodes {
            return Err(CoreError::dimension_mismatch(
                self.num_nodes,
                layout.len(),
                "GemSolver::step",
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

        if self.iter == 0 {
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

            if (max_x - min_x) < 1e-12 && (max_y - min_y) < 1e-12 {
                let spread = self.config.edge_length;
                for i in 0..n {
                    self.x_buf[i] += spread * (self.rng.next_f64() - 0.5);
                    self.y_buf[i] += spread * (self.rng.next_f64() - 0.5);
                }
            }
        }

        let mut sum_x = 0.0_f64;
        let mut sum_y = 0.0_f64;
        for i in 0..n {
            sum_x += self.x_buf[i];
            sum_y += self.y_buf[i];
        }

        self.rng.shuffle(&mut self.order);

        let n_f = n as f64;
        let e_sq = self.config.edge_length * self.config.edge_length;
        let inv_e_sq = 1.0 / e_sq;
        let gravity = self.config.gravity;
        let perturbation = self.config.perturbation;
        let t_max = self.config.t_max;
        let osc_sens = self.config.oscillation_sensitivity;

        for ord_idx in 0..n {
            let v = self.order[ord_idx] as usize;
            let vx = self.x_buf[v];
            let vy = self.y_buf[v];
            let phi_v = self.phi[v];

            // 1. Gravity toward barycenter.
            let mut px = (sum_x / n_f - vx) * gravity * phi_v;
            let mut py = (sum_y / n_f - vy) * gravity * phi_v;

            // 2. Random symmetry-breaking perturbation.
            if perturbation > 0.0 {
                px += (self.rng.next_f64() * 2.0 - 1.0) * perturbation;
                py += (self.rng.next_f64() * 2.0 - 1.0) * perturbation;
            }

            // 3. All-pairs repulsion.
            for u in 0..n {
                if u == v {
                    continue;
                }
                let dx = vx - self.x_buf[u];
                let dy = vy - self.y_buf[u];
                let d2 = dx * dx + dy * dy;
                if d2 > MIN_DIST_SQ {
                    let scale = e_sq / d2;
                    px += dx * scale;
                    py += dy * scale;
                }
            }

            // 4. Incident edge attraction.
            let adj_start = self.adj_indptr[v];
            let adj_end = self.adj_indptr[v + 1];
            let inv_denom = inv_e_sq / phi_v;
            for edge_pos in adj_start..adj_end {
                let w = self.adj_weights[edge_pos];
                if w <= 0.0 {
                    continue;
                }
                let u = self.adj_indices[edge_pos];
                let dx = vx - self.x_buf[u];
                let dy = vy - self.y_buf[u];
                let d2 = dx * dx + dy * dy;
                let scale = w * d2 * inv_denom;
                px -= dx * scale;
                py -= dy * scale;
            }

            // 5. Position step and local temperature / rotation / oscillation update.
            let p_norm = px.hypot(py);
            let tv = self.temp[v];
            if p_norm > MIN_NORM && tv > 0.0 {
                let step_scale = tv / p_norm;
                let sx = px * step_scale;
                let sy = py * step_scale;

                self.x_buf[v] += sx;
                self.y_buf[v] += sy;
                sum_x += sx;
                sum_y += sy;

                let old_px = self.imp_x[v];
                let old_py = self.imp_y[v];
                let old_norm = old_px.hypot(old_py);
                if old_norm > MIN_NORM {
                    let denom = tv * old_norm;
                    let cos_b = ((sx * old_px + sy * old_py) / denom).clamp(-1.0, 1.0);
                    let sin_b = ((old_px * sy - old_py * sx) / denom).clamp(-1.0, 1.0);

                    if sin_b.abs() > self.sin_rot_thresh {
                        self.dir[v] += self.sigma_r * sin_b.signum();
                    }

                    let mut next_t = tv;
                    if cos_b.abs() > self.cos_osc_thresh {
                        next_t *= (1.0 + osc_sens * cos_b).max(0.0);
                    }
                    next_t = (next_t * (1.0 - self.dir[v].abs()).max(0.0)).min(t_max);
                    self.temp[v] = next_t;
                }

                self.imp_x[v] = sx;
                self.imp_y[v] = sy;
            }
        }

        let (xs_mut, ys_mut) = layout.xy_mut();
        for i in 0..n {
            let nx = S::from_f64(self.x_buf[i]);
            let ny = S::from_f64(self.y_buf[i]);
            if nx.is_finite() {
                xs_mut[i] = nx;
            }
            if ny.is_finite() {
                ys_mut[i] = ny;
            }
        }

        let mean_temp = self.temp.iter().sum::<f64>() / n_f;
        self.iter += 1;
        self.status = if mean_temp <= self.config.t_min {
            SolverStatus::Converged
        } else if self.iter >= self.max_rounds {
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
        self.max_rounds
    }
}

#[allow(clippy::type_complexity)]
fn extract_undirected_adj<W: Scalar>(
    graph: &CsrGraph<W>,
) -> CoreResult<(usize, Vec<usize>, Vec<usize>, Vec<f64>)> {
    let n = graph.num_nodes();
    if u32::try_from(n).is_err() {
        return Err(CoreError::invalid_csr("num_nodes exceeds u32::MAX"));
    }
    if let Some(ws) = graph.weights() {
        for &w in ws {
            let wf = w.to_f64();
            if !wf.is_finite() {
                return Err(CoreError::non_finite("GemSolver::graph_weights"));
            }
            if wf < 0.0 {
                return Err(CoreError::invalid_input(
                    "graph edge weights must be non-negative",
                ));
            }
        }
    }

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
    let mut directed_pairs: Vec<(usize, usize, f64)> = Vec::new();
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
        let ua = a as usize;
        let ub = b as usize;
        directed_pairs.push((ua, ub, merged_w));
        directed_pairs.push((ub, ua, merged_w));
    }
    directed_pairs.sort_unstable_by_key(|x| (x.0, x.1));

    let mut indptr = vec![0usize; n + 1];
    for &(u, _, _) in &directed_pairs {
        indptr[u + 1] += 1;
    }
    for i in 0..n {
        indptr[i + 1] += indptr[i];
    }
    let mut indices = Vec::with_capacity(directed_pairs.len());
    let mut weights = Vec::with_capacity(directed_pairs.len());
    for &(_, v, w) in &directed_pairs {
        indices.push(v);
        weights.push(w);
    }

    Ok((n, indptr, indices, weights))
}
