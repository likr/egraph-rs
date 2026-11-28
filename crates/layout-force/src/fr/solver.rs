use std::marker::PhantomData;

use egraph_core::{
    CoreError, CoreResult, CsrGraph, DeterministicRng, IterativeSolver, Layout2d, RiemannianLayout,
    Scalar, SolverStatus,
};

use crate::backend::DefaultFrBackend;
use crate::graph_edges::{
    extract_canonical_undirected_edges, extract_canonical_undirected_unweighted_edges,
};

use super::config::FruchtermanReingoldConfig;
use super::repulsion::FrRepulsion;

pub(super) const COINCIDENT_DIST_SQ: f64 = 1e-24;
pub(super) const COINCIDENT_PERTURB_DIST: f64 = 1e-6;
pub(super) const BH_EPS_SQ: f64 = 1e-12;
const GRAVITY_ANCHOR_LIMIT: usize = 16;

#[derive(Debug, Clone)]
struct FrScratch {
    dims: usize,
    disp: Vec<f64>,
    log_ij: Vec<f64>,
    log_ji: Vec<f64>,
    delta_buf: Vec<f64>,
}

impl FrScratch {
    fn new(num_nodes: usize, dims: usize) -> CoreResult<Self> {
        let total = num_nodes
            .checked_mul(dims)
            .ok_or_else(|| CoreError::invalid_input("FR displacement buffer size overflow"))?;
        Ok(Self {
            dims,
            disp: vec![0.0; total],
            log_ij: vec![0.0; dims],
            log_ji: vec![0.0; dims],
            delta_buf: vec![0.0; dims],
        })
    }
}

#[derive(Debug, Clone)]
struct FrState {
    config: FruchtermanReingoldConfig,
    num_nodes: usize,
    edges_u: Vec<u32>,
    edges_v: Vec<u32>,
    edges_w: Vec<f64>,
    k: f64,
    initial_temperature: f64,
    locked_tangent_dim: Option<usize>,
    scratch: Option<FrScratch>,
    rng: DeterministicRng,
    iter: usize,
    status: SolverStatus,
}

impl FrState {
    #[inline]
    fn current_temperature(&self) -> f64 {
        if self.config.iterations == 0 || self.iter >= self.config.iterations {
            return 0.0;
        }
        let rem = (self.config.iterations - self.iter) as f64;
        self.initial_temperature * (rem / (self.config.iterations as f64))
    }
}

/// Cooperative Fruchterman–Reingold (1991) layout solver implementing [`IterativeSolver`].
///
/// Parameterized at compile time by target [`RiemannianLayout`] `L` and repulsion backend `B`:
/// - [`Layout2d`](egraph_core::Layout2d) / [`Layout3d`](egraph_core::Layout3d):
///   supports [`Exact`](crate::Exact) (default), [`BarnesHut`](crate::BarnesHut),
///   [`Grid`](crate::Grid), and [`FrBackend`](crate::FrBackend).
/// - [`LayoutNd`](egraph_core::LayoutNd), [`ManifoldLayout2d`](egraph_core::ManifoldLayout2d),
///   and any external [`RiemannianLayout`] (e.g. `SurfaceLayout`): supports
///   [`Exact`](crate::Exact) (default) and rejects [`BarnesHut`](crate::BarnesHut) /
///   [`Grid`](crate::Grid) / [`FrBackend`](crate::FrBackend) at compile time:
///
/// ```compile_fail
/// use egraph_core::{CsrGraph, LayoutNd};
/// use egraph_layout_force::{BarnesHut, FruchtermanReingoldConfig, FruchtermanReingoldSolver};
///
/// let g = CsrGraph::<()>::from_unweighted_edges(3, &[(0, 1), (1, 2)], false, false).unwrap();
/// let _s = FruchtermanReingoldSolver::<LayoutNd<f64>, BarnesHut>::new(&g, FruchtermanReingoldConfig::new());
/// ```
///
/// ```compile_fail
/// use egraph_core::{CsrGraph, ManifoldLayout2d};
/// use egraph_layout_force::{BarnesHut, FruchtermanReingoldConfig, FruchtermanReingoldSolver};
///
/// let g = CsrGraph::<()>::from_unweighted_edges(3, &[(0, 1), (1, 2)], false, false).unwrap();
/// let _s = FruchtermanReingoldSolver::<ManifoldLayout2d<f64>, BarnesHut>::new(&g, FruchtermanReingoldConfig::new());
/// ```
///
/// ```compile_fail
/// use egraph_core::{CsrGraph, LayoutNd};
/// use egraph_layout_force::{FruchtermanReingoldConfig, FruchtermanReingoldSolver, Grid};
///
/// let g = CsrGraph::<()>::from_unweighted_edges(3, &[(0, 1), (1, 2)], false, false).unwrap();
/// let _s = FruchtermanReingoldSolver::<LayoutNd<f64>, Grid>::new(&g, FruchtermanReingoldConfig::new());
/// ```
#[derive(Debug, Clone)]
pub struct FruchtermanReingoldSolver<
    L: RiemannianLayout + DefaultFrBackend = Layout2d<f64>,
    B: FrRepulsion<L> = <L as DefaultFrBackend>::Backend,
> {
    state: FrState,
    backend: B,
    repulsion_state: B::State,
    _layout: PhantomData<fn() -> L>,
}

impl<L, B> FruchtermanReingoldSolver<L, B>
where
    L: RiemannianLayout + DefaultFrBackend,
    B: FrRepulsion<L> + Default,
{
    /// Constructs a [`FruchtermanReingoldSolver`] with the default backend `B::default()`.
    pub fn new<W: Scalar>(
        graph: &CsrGraph<W>,
        config: FruchtermanReingoldConfig,
    ) -> CoreResult<Self> {
        Self::with_backend(graph, config, B::default())
    }

    /// Constructs a [`FruchtermanReingoldSolver`] from an unweighted [`CsrGraph<W>`] (or ignoring
    /// stored edge weights) with `B::default()`, without requiring `W: Scalar`.
    pub fn new_unweighted<W>(
        graph: &CsrGraph<W>,
        config: FruchtermanReingoldConfig,
    ) -> CoreResult<Self> {
        Self::with_backend_unweighted(graph, config, B::default())
    }
}

impl<L, B> FruchtermanReingoldSolver<L, B>
where
    L: RiemannianLayout + DefaultFrBackend,
    B: FrRepulsion<L>,
{
    /// Constructs a [`FruchtermanReingoldSolver`] from a [`CsrGraph`], [`FruchtermanReingoldConfig`],
    /// and explicit `backend`.
    pub fn with_backend<W: Scalar>(
        graph: &CsrGraph<W>,
        config: FruchtermanReingoldConfig,
        backend: B,
    ) -> CoreResult<Self> {
        config.validate()?;
        backend.validate()?;
        let (n, edges_u, edges_v, edges_w, _) =
            extract_canonical_undirected_edges(graph, "FruchtermanReingoldSolver::graph_weights")?;
        Self::from_canonical_edges(n, edges_u, edges_v, edges_w, config, backend)
    }

    /// Constructs a [`FruchtermanReingoldSolver`] from an unweighted [`CsrGraph<W>`] (or ignoring
    /// stored edge weights) and explicit `backend`, without requiring `W: Scalar`.
    pub fn with_backend_unweighted<W>(
        graph: &CsrGraph<W>,
        config: FruchtermanReingoldConfig,
        backend: B,
    ) -> CoreResult<Self> {
        config.validate()?;
        backend.validate()?;
        let (n, edges_u, edges_v, edges_w, _) =
            extract_canonical_undirected_unweighted_edges(graph)?;
        Self::from_canonical_edges(n, edges_u, edges_v, edges_w, config, backend)
    }

    fn from_canonical_edges(
        n: usize,
        edges_u: Vec<u32>,
        edges_v: Vec<u32>,
        edges_w: Vec<f64>,
        config: FruchtermanReingoldConfig,
        backend: B,
    ) -> CoreResult<Self> {
        let repulsion_state = backend.init_state(n)?;
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
        let status = if config.iterations == 0 {
            SolverStatus::Completed
        } else {
            SolverStatus::Initialized
        };
        let seed = config.seed;

        let state = FrState {
            config,
            num_nodes: n,
            edges_u,
            edges_v,
            edges_w,
            k,
            initial_temperature,
            locked_tangent_dim: None,
            scratch: None,
            rng: DeterministicRng::from_seed_u64(seed),
            iter: 0,
            status,
        };

        Ok(Self {
            state,
            backend,
            repulsion_state,
            _layout: PhantomData,
        })
    }

    /// Returns the solver configuration.
    #[inline]
    pub fn config(&self) -> &FruchtermanReingoldConfig {
        &self.state.config
    }

    /// Returns the configured repulsion backend.
    #[inline]
    pub fn backend(&self) -> &B {
        &self.backend
    }

    /// Returns the expected number of vertices in the target layout.
    #[inline]
    pub fn num_nodes(&self) -> usize {
        self.state.num_nodes
    }

    /// Returns the current lifecycle status.
    #[inline]
    pub fn status(&self) -> SolverStatus {
        self.state.status
    }

    /// Returns the number of iterations executed so far.
    #[inline]
    pub fn iteration(&self) -> usize {
        self.state.iter
    }

    /// Returns the scheduled maximum number of iterations.
    #[inline]
    pub fn total_iterations(&self) -> Option<usize> {
        Some(self.state.config.iterations)
    }

    /// Returns the effective optimal pairwise distance $k$.
    #[inline]
    pub fn optimal_distance(&self) -> f64 {
        self.state.k
    }

    /// Returns the temperature used for the upcoming (or final) iteration.
    #[inline]
    pub fn current_temperature(&self) -> f64 {
        self.state.current_temperature()
    }
}

impl<L, B> IterativeSolver for FruchtermanReingoldSolver<L, B>
where
    L: RiemannianLayout + DefaultFrBackend,
    B: FrRepulsion<L>,
{
    type Target = L;

    fn validate_target(&self, target: &Self::Target) -> CoreResult<()> {
        if target.len() != self.state.num_nodes {
            return Err(CoreError::dimension_mismatch(
                self.state.num_nodes,
                target.len(),
                "FruchtermanReingoldSolver::validate_target",
            ));
        }
        let dims = target.tangent_dim();
        if dims == 0 && self.state.num_nodes > 0 {
            return Err(CoreError::invalid_input("tangent_dim must be >= 1"));
        }
        if let Some(prev) = self.state.locked_tangent_dim
            && prev != dims
        {
            return Err(CoreError::dimension_mismatch(
                prev,
                dims,
                "FruchtermanReingoldSolver::validate_target::dims",
            ));
        }
        target.validate_finite()
    }

    fn step(&mut self, target: &mut Self::Target) -> CoreResult<SolverStatus> {
        self.validate_target(target)?;

        if self.state.status.is_finished() {
            return Ok(self.state.status);
        }

        if self.state.num_nodes <= 1 {
            self.state.status = SolverStatus::Converged;
            return Ok(self.state.status);
        }

        let n = self.state.num_nodes;
        let dims = target.tangent_dim();
        if self.state.locked_tangent_dim.is_none() {
            self.state.locked_tangent_dim = Some(dims);
        }

        let mut scratch = match self.state.scratch.take() {
            Some(existing) if existing.dims == dims => existing,
            _ => FrScratch::new(n, dims)?,
        };

        let result = (|| -> CoreResult<()> {
            scratch.disp.fill(0.0);

            // 1. Repulsion via backend B specialized for space L.
            self.backend.accumulate_repulsion(
                &mut self.repulsion_state,
                target,
                self.state.k,
                self.state.iter,
                &mut self.state.rng,
                &mut scratch.disp,
            )?;

            // 2. Geodesic edge attraction: f_a(d) = d^2 / k in tangent space.
            let inv_k = 1.0 / self.state.k;
            for idx in 0..self.state.edges_u.len() {
                let u = self.state.edges_u[idx] as usize;
                let v = self.state.edges_v[idx] as usize;
                let w = self.state.edges_w[idx];
                if w == 0.0 {
                    continue;
                }
                target.log_map_into(u, v, &mut scratch.log_ij)?;
                target.log_map_into(v, u, &mut scratch.log_ji)?;
                let mut d_sq = 0.0_f64;
                for &val in &scratch.log_ij {
                    d_sq += val * val;
                }
                let factor = w * d_sq.sqrt() * inv_k;
                let base_u = u * dims;
                let base_v = v * dims;
                for d in 0..dims {
                    scratch.disp[base_u + d] += factor * scratch.log_ij[d];
                    scratch.disp[base_v + d] += factor * scratch.log_ji[d];
                }
            }

            // 3. O(N) intrinsic Riemannian barycentric gravity when gravity > 0.0.
            if self.state.config.gravity > 0.0 {
                let anchors = n.min(GRAVITY_ANCHOR_LIMIT);
                let scale_g = self.state.config.gravity / (anchors as f64);
                for i in 0..n {
                    let base_i = i * dims;
                    for m in 0..anchors {
                        let anchor = (m * n) / anchors;
                        if anchor == i {
                            continue;
                        }
                        target.log_map_into(i, anchor, &mut scratch.log_ij)?;
                        for d in 0..dims {
                            scratch.disp[base_i + d] += scale_g * scratch.log_ij[d];
                        }
                    }
                }
            }

            // 4. Temperature cooling and exponential map update.
            let temp = self.state.current_temperature();
            for i in 0..n {
                let base = i * dims;
                let mut len_sq = 0.0_f64;
                for d in 0..dims {
                    let v = scratch.disp[base + d];
                    len_sq += v * v;
                }
                let len = len_sq.sqrt();
                if len > 1e-24 && temp > 0.0 {
                    let scale = len.min(temp) / len;
                    for d in 0..dims {
                        scratch.delta_buf[d] = scratch.disp[base + d] * scale;
                    }
                    if let Err(err) = target.exp_map(i, &scratch.delta_buf) {
                        return Err(match err {
                            CoreError::NonFiniteValue { context } => {
                                CoreError::numerical_breakdown(format!(
                                    "FruchtermanReingoldSolver coordinates became non-finite ({context})"
                                ))
                            }
                            other => other,
                        });
                    }
                }
            }
            Ok(())
        })();

        self.state.scratch = Some(scratch);
        if let Err(err) = result {
            if matches!(
                err,
                CoreError::NumericalBreakdown(_)
                    | CoreError::NonFiniteValue { .. }
                    | CoreError::BackendUnavailable(_)
                    | CoreError::DeviceLimitExceeded(_)
            ) {
                self.state.status = SolverStatus::Failed;
            }
            return Err(err);
        }

        self.state.iter += 1;
        self.state.status = if self.state.iter >= self.state.config.iterations {
            SolverStatus::Completed
        } else {
            SolverStatus::Running
        };
        Ok(self.state.status)
    }

    #[inline]
    fn status(&self) -> SolverStatus {
        self.state.status
    }

    #[inline]
    fn iteration(&self) -> usize {
        self.state.iter
    }

    #[inline]
    fn total_iterations(&self) -> Option<usize> {
        Some(self.state.config.iterations)
    }
}
