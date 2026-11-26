#[cfg(doc)]
use egraph_core::IterativeSolver;
use egraph_core::{CoreError, CoreResult, CsrGraph, Layout2d, LayoutNd, Scalar, SolverStatus};
use egraph_spatial_trees::ArenaQuadtree2d;

use super::ISOLATED_NODE_MASS;
use super::config::{LinLogConfig, LinLogRepulsion};
use super::nd::LinLogNdScratch;
use super::quadtree::GenRepulsionQuadtree;

const MAX_LINLOG_DISPLACEMENT: f64 = 50.0;

/// Cooperative true Noack (2007/2009) LinLog / $(a, r)$-PolyLog solver implementing [`IterativeSolver`].
#[derive(Debug, Clone)]
pub struct LinLogSolver<S: Scalar = f64> {
    pub(super) config: LinLogConfig,
    pub(super) num_nodes: usize,
    pub(super) edges_u: Vec<u32>,
    pub(super) edges_v: Vec<u32>,
    pub(super) edges_w: Vec<f64>,
    pub(super) masses: Vec<f64>,
    pub(super) total_mass: f64,
    pub(super) x_buf: Vec<f64>,
    pub(super) y_buf: Vec<f64>,
    pub(super) fx: Vec<f64>,
    pub(super) fy: Vec<f64>,
    pub(super) prev_fx: Vec<f64>,
    pub(super) prev_fy: Vec<f64>,
    pub(super) swing_buf: Vec<f64>,
    pub(super) att_hess: Vec<f64>,
    pub(super) rep_x: Vec<f64>,
    pub(super) rep_y: Vec<f64>,
    pub(super) quadtree: ArenaQuadtree2d<f64>,
    pub(super) gen_quadtree: GenRepulsionQuadtree,
    pub(super) global_speed: f64,
    pub(super) max_displacement: f64,
    pub(super) active_dims: usize,
    pub(super) nd_scratch: Option<LinLogNdScratch>,
    pub(super) iter: usize,
    pub(super) status: SolverStatus,
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
            active_dims: 0,
            nd_scratch: None,
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

    /// Evaluates the exact Noack (2009) $(a, r)$-LinLog energy $U_{(a,r)}(p)$ of an N-D `layout`.
    pub fn energy_nd<T: Scalar>(&self, layout: &LayoutNd<T>) -> CoreResult<f64> {
        if layout.len() != self.num_nodes {
            return Err(CoreError::dimension_mismatch(
                self.num_nodes,
                layout.len(),
                "LinLogSolver::energy_nd",
            ));
        }
        let dims = layout.dims();
        if dims == 0 {
            return Err(CoreError::invalid_input("LayoutNd dims must be >= 1"));
        }
        layout.validate_finite()?;
        let n = self.num_nodes;
        if n <= 1 {
            return Ok(0.0);
        }

        let coords = layout.coords();
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
            let base_u = u * dims;
            let base_v = v * dims;
            let mut d_sq = 0.0_f64;
            for axis in 0..dims {
                let delta = coords[base_u + axis].to_f64() - coords[base_v + axis].to_f64();
                d_sq += delta * delta;
            }
            let d_safe = d_sq.sqrt().max(1e-12);
            let term = if a_plus_1.abs() <= 1e-12 {
                d_safe.ln()
            } else {
                d_safe.powf(a_plus_1) / a_plus_1
            };
            u_att += w * term;
        }

        let mut u_rep = 0.0_f64;
        for i in 0..n {
            let base_i = i * dims;
            let mi = self.masses[i];
            for (j, &mj) in self.masses.iter().enumerate().take(n).skip(i + 1) {
                let base_j = j * dims;
                let mut d_sq = 0.0_f64;
                for axis in 0..dims {
                    let delta = coords[base_i + axis].to_f64() - coords[base_j + axis].to_f64();
                    d_sq += delta * delta;
                }
                let d_safe = d_sq.sqrt().max(1e-12);
                let term = if r_plus_1.abs() <= 1e-12 {
                    d_safe.ln()
                } else {
                    d_safe.powf(r_plus_1) / r_plus_1
                };
                u_rep -= mi * mj * term;
            }
        }

        let mut u_grav = 0.0_f64;
        if self.config.gravity > 0.0 && self.total_mass > 0.0 {
            let mut barycenter = vec![0.0_f64; dims];
            for (i, &m) in self.masses.iter().enumerate().take(n) {
                let base = i * dims;
                for axis in 0..dims {
                    barycenter[axis] += m * coords[base + axis].to_f64();
                }
            }
            for b in &mut barycenter {
                *b /= self.total_mass;
            }
            for i in 0..n {
                let base = i * dims;
                let mut d_sq = 0.0_f64;
                for axis in 0..dims {
                    let delta = coords[base + axis].to_f64() - barycenter[axis];
                    d_sq += delta * delta;
                }
                let d_safe = d_sq.sqrt().max(1e-12);
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
