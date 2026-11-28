use std::fmt;

use egraph_core::{
    CoreError, CoreResult, DeterministicRng, Layout2d, Layout3d, RiemannianLayout, Scalar,
};

use crate::backend::{BarnesHut, Exact, FrBackend, Grid};

use super::barnes_hut::{FrBarnesHut2dState, FrBarnesHut3dState};
use super::grid::{
    FrGrid2dState, FrGrid3dState, compute_grid_repulsion_2d, compute_grid_repulsion_3d,
};
use super::solver::{COINCIDENT_DIST_SQ, COINCIDENT_PERTURB_DIST};

/// Open repulsion backend trait for [`FruchtermanReingoldSolver`](super::FruchtermanReingoldSolver).
///
/// Implementations are provided for:
/// - [`Exact`] across all [`RiemannianLayout`] targets ([`Layout2d`], [`Layout3d`], [`LayoutNd`](egraph_core::LayoutNd),
///   [`ManifoldLayout2d`](egraph_core::ManifoldLayout2d), and external spaces such as `SurfaceLayout`).
/// - [`BarnesHut`], [`Grid`], and [`FrBackend`] strictly on [`Layout2d`] and [`Layout3d`]
///   (passing [`BarnesHut`], [`Grid`], or [`FrBackend`] with [`LayoutNd`](egraph_core::LayoutNd)
///   or [`ManifoldLayout2d`](egraph_core::ManifoldLayout2d) is rejected at compile time).
///
/// External crates (such as `egraph-surface`) may implement this trait for custom spaces and
/// geodesic repulsion backends without modifying `egraph-layout-force`.
pub trait FrRepulsion<L: RiemannianLayout + ?Sized> {
    /// Backend-owned scratch state (e.g. spatial trees or grid buffers; no unused tree fields in
    /// [`Exact`]).
    type State: fmt::Debug + Clone;

    /// Validates backend configuration before constructing a solver.
    fn validate(&self) -> CoreResult<()>;

    /// Initializes backend-specific scratch state for `num_nodes` vertices.
    fn init_state(&self, num_nodes: usize) -> CoreResult<Self::State>;

    /// Accumulates repulsive forces into `disp` (row-major `num_nodes * tangent_dim`).
    fn accumulate_repulsion(
        &self,
        state: &mut Self::State,
        layout: &L,
        k: f64,
        iter: usize,
        rng: &mut DeterministicRng,
        disp: &mut [f64],
    ) -> CoreResult<()>;
}

/// Scratch state for [`Exact`] repulsion in [`FruchtermanReingoldSolver`](super::FruchtermanReingoldSolver).
#[derive(Debug, Clone, Default)]
pub struct FrExactState {
    log_ij: Vec<f64>,
    log_ji: Vec<f64>,
    delta_buf: Vec<f64>,
}

/// Scratch state for [`FrBackend`] in 2D [`FruchtermanReingoldSolver`](super::FruchtermanReingoldSolver).
#[derive(Debug, Clone)]
pub enum FrDynamic2dState {
    Exact(FrExactState),
    BarnesHut(FrBarnesHut2dState),
    Grid(FrGrid2dState),
}

/// Scratch state for [`FrBackend`] in 3D [`FruchtermanReingoldSolver`](super::FruchtermanReingoldSolver).
#[derive(Debug, Clone)]
pub enum FrDynamic3dState {
    Exact(FrExactState),
    BarnesHut(Box<FrBarnesHut3dState>),
    Grid(FrGrid3dState),
}

impl<L: RiemannianLayout> FrRepulsion<L> for Exact {
    type State = FrExactState;

    #[inline]
    fn validate(&self) -> CoreResult<()> {
        Ok(())
    }

    #[inline]
    fn init_state(&self, _num_nodes: usize) -> CoreResult<Self::State> {
        Ok(FrExactState::default())
    }

    fn accumulate_repulsion(
        &self,
        state: &mut Self::State,
        layout: &L,
        k: f64,
        _iter: usize,
        rng: &mut DeterministicRng,
        disp: &mut [f64],
    ) -> CoreResult<()> {
        let n = layout.len();
        let dims = layout.tangent_dim();
        let expected = n
            .checked_mul(dims)
            .ok_or_else(|| CoreError::invalid_input("FR repulsion buffer size overflow"))?;
        if disp.len() != expected {
            return Err(CoreError::dimension_mismatch(
                expected,
                disp.len(),
                "FrRepulsion<Exact>::accumulate_repulsion",
            ));
        }
        if n < 2 {
            return Ok(());
        }

        state.log_ij.resize(dims, 0.0);
        state.log_ji.resize(dims, 0.0);
        state.delta_buf.resize(dims, 0.0);
        let k_sq = k * k;

        for i in 0..n {
            let base_i = i * dims;
            for j in (i + 1)..n {
                let base_j = j * dims;
                layout.log_map_into(i, j, &mut state.log_ij)?;
                layout.log_map_into(j, i, &mut state.log_ji)?;
                let mut d_sq = 0.0_f64;
                for &v in &state.log_ij {
                    d_sq += v * v;
                }
                if d_sq > COINCIDENT_DIST_SQ {
                    let mult = k_sq / d_sq;
                    for d in 0..dims {
                        disp[base_i + d] -= mult * state.log_ij[d];
                        disp[base_j + d] -= mult * state.log_ji[d];
                    }
                } else {
                    let mut norm_sq = 0.0_f64;
                    for slot in &mut state.delta_buf {
                        let c = rng.next_f64() - 0.5;
                        *slot = c;
                        norm_sq += c * c;
                    }
                    let norm = norm_sq.sqrt();
                    let mag = k_sq / COINCIDENT_PERTURB_DIST;
                    if norm > 1e-12 {
                        let scale = mag / norm;
                        for d in 0..dims {
                            let f = scale * state.delta_buf[d];
                            disp[base_i + d] += f;
                            disp[base_j + d] -= f;
                        }
                    } else {
                        disp[base_i] += mag;
                        disp[base_j] -= mag;
                    }
                }
            }
        }
        Ok(())
    }
}

impl<S: Scalar> FrRepulsion<Layout2d<S>> for Grid {
    type State = FrGrid2dState;

    #[inline]
    fn validate(&self) -> CoreResult<()> {
        Ok(())
    }

    fn init_state(&self, num_nodes: usize) -> CoreResult<Self::State> {
        Ok(FrGrid2dState::new(num_nodes))
    }

    fn accumulate_repulsion(
        &self,
        state: &mut Self::State,
        layout: &Layout2d<S>,
        k: f64,
        iter: usize,
        rng: &mut DeterministicRng,
        disp: &mut [f64],
    ) -> CoreResult<()> {
        let n = layout.len();
        let expected = n
            .checked_mul(2)
            .ok_or_else(|| CoreError::invalid_input("FR 2D grid buffer size overflow"))?;
        if disp.len() != expected {
            return Err(CoreError::dimension_mismatch(
                expected,
                disp.len(),
                "FrRepulsion<Grid, Layout2d>::accumulate_repulsion",
            ));
        }
        if n < 2 {
            return Ok(());
        }

        let xs = layout.x();
        let ys = layout.y();
        state.x_buf.resize(n, 0.0);
        state.y_buf.resize(n, 0.0);
        let mut min_x = f64::INFINITY;
        let mut max_x = f64::NEG_INFINITY;
        let mut min_y = f64::INFINITY;
        let mut max_y = f64::NEG_INFINITY;
        for i in 0..n {
            let x = xs[i].to_f64();
            let y = ys[i].to_f64();
            state.x_buf[i] = x;
            state.y_buf[i] = y;
            min_x = min_x.min(x);
            max_x = max_x.max(x);
            min_y = min_y.min(y);
            max_y = max_y.max(y);
        }
        if iter == 0 && (max_x - min_x) < 1e-12 && (max_y - min_y) < 1e-12 {
            let spread = 0.5 * k;
            min_x = f64::INFINITY;
            min_y = f64::INFINITY;
            for i in 0..n {
                state.x_buf[i] += spread * (rng.next_f64() - 0.5);
                state.y_buf[i] += spread * (rng.next_f64() - 0.5);
                min_x = min_x.min(state.x_buf[i]);
                min_y = min_y.min(state.y_buf[i]);
            }
        }
        compute_grid_repulsion_2d(state, min_x, min_y, k, rng, disp);
        Ok(())
    }
}

impl<S: Scalar> FrRepulsion<Layout3d<S>> for Grid {
    type State = FrGrid3dState;

    #[inline]
    fn validate(&self) -> CoreResult<()> {
        Ok(())
    }

    fn init_state(&self, num_nodes: usize) -> CoreResult<Self::State> {
        Ok(FrGrid3dState::new(num_nodes))
    }

    fn accumulate_repulsion(
        &self,
        state: &mut Self::State,
        layout: &Layout3d<S>,
        k: f64,
        iter: usize,
        rng: &mut DeterministicRng,
        disp: &mut [f64],
    ) -> CoreResult<()> {
        let n = layout.len();
        let expected = n
            .checked_mul(3)
            .ok_or_else(|| CoreError::invalid_input("FR 3D grid buffer size overflow"))?;
        if disp.len() != expected {
            return Err(CoreError::dimension_mismatch(
                expected,
                disp.len(),
                "FrRepulsion<Grid, Layout3d>::accumulate_repulsion",
            ));
        }
        if n < 2 {
            return Ok(());
        }

        let xs = layout.x();
        let ys = layout.y();
        let zs = layout.z();
        state.x_buf.resize(n, 0.0);
        state.y_buf.resize(n, 0.0);
        state.z_buf.resize(n, 0.0);
        let mut min_x = f64::INFINITY;
        let mut max_x = f64::NEG_INFINITY;
        let mut min_y = f64::INFINITY;
        let mut max_y = f64::NEG_INFINITY;
        let mut min_z = f64::INFINITY;
        let mut max_z = f64::NEG_INFINITY;
        for i in 0..n {
            let x = xs[i].to_f64();
            let y = ys[i].to_f64();
            let z = zs[i].to_f64();
            state.x_buf[i] = x;
            state.y_buf[i] = y;
            state.z_buf[i] = z;
            min_x = min_x.min(x);
            max_x = max_x.max(x);
            min_y = min_y.min(y);
            max_y = max_y.max(y);
            min_z = min_z.min(z);
            max_z = max_z.max(z);
        }
        if iter == 0
            && (max_x - min_x) < 1e-12
            && (max_y - min_y) < 1e-12
            && (max_z - min_z) < 1e-12
        {
            let spread = 0.5 * k;
            min_x = f64::INFINITY;
            min_y = f64::INFINITY;
            min_z = f64::INFINITY;
            for i in 0..n {
                state.x_buf[i] += spread * (rng.next_f64() - 0.5);
                state.y_buf[i] += spread * (rng.next_f64() - 0.5);
                state.z_buf[i] += spread * (rng.next_f64() - 0.5);
                min_x = min_x.min(state.x_buf[i]);
                min_y = min_y.min(state.y_buf[i]);
                min_z = min_z.min(state.z_buf[i]);
            }
        }
        compute_grid_repulsion_3d(state, min_x, min_y, min_z, k, rng, disp);
        Ok(())
    }
}

impl<S: Scalar> FrRepulsion<Layout2d<S>> for FrBackend {
    type State = FrDynamic2dState;

    #[inline]
    fn validate(&self) -> CoreResult<()> {
        FrBackend::validate(self)
    }

    fn init_state(&self, num_nodes: usize) -> CoreResult<Self::State> {
        match self {
            Self::Exact(ex) => Ok(FrDynamic2dState::Exact(<Exact as FrRepulsion<
                Layout2d<S>,
            >>::init_state(
                ex, num_nodes
            )?)),
            Self::BarnesHut(bh) => Ok(FrDynamic2dState::BarnesHut(<BarnesHut as FrRepulsion<
                Layout2d<S>,
            >>::init_state(
                bh, num_nodes
            )?)),
            Self::Grid(gr) => Ok(FrDynamic2dState::Grid(<Grid as FrRepulsion<
                Layout2d<S>,
            >>::init_state(
                gr, num_nodes
            )?)),
        }
    }

    fn accumulate_repulsion(
        &self,
        state: &mut Self::State,
        layout: &Layout2d<S>,
        k: f64,
        iter: usize,
        rng: &mut DeterministicRng,
        disp: &mut [f64],
    ) -> CoreResult<()> {
        match (self, state) {
            (Self::Exact(ex), FrDynamic2dState::Exact(st)) => {
                ex.accumulate_repulsion(st, layout, k, iter, rng, disp)
            }
            (Self::BarnesHut(bh), FrDynamic2dState::BarnesHut(st)) => {
                bh.accumulate_repulsion(st, layout, k, iter, rng, disp)
            }
            (Self::Grid(gr), FrDynamic2dState::Grid(st)) => {
                gr.accumulate_repulsion(st, layout, k, iter, rng, disp)
            }
            _ => Err(CoreError::invalid_input(
                "FrBackend state mismatch in Fruchterman-Reingold 2D",
            )),
        }
    }
}

impl<S: Scalar> FrRepulsion<Layout3d<S>> for FrBackend {
    type State = FrDynamic3dState;

    #[inline]
    fn validate(&self) -> CoreResult<()> {
        FrBackend::validate(self)
    }

    fn init_state(&self, num_nodes: usize) -> CoreResult<Self::State> {
        match self {
            Self::Exact(ex) => Ok(FrDynamic3dState::Exact(<Exact as FrRepulsion<
                Layout3d<S>,
            >>::init_state(
                ex, num_nodes
            )?)),
            Self::BarnesHut(bh) => Ok(FrDynamic3dState::BarnesHut(Box::new(
                <BarnesHut as FrRepulsion<Layout3d<S>>>::init_state(bh, num_nodes)?,
            ))),
            Self::Grid(gr) => Ok(FrDynamic3dState::Grid(<Grid as FrRepulsion<
                Layout3d<S>,
            >>::init_state(
                gr, num_nodes
            )?)),
        }
    }

    fn accumulate_repulsion(
        &self,
        state: &mut Self::State,
        layout: &Layout3d<S>,
        k: f64,
        iter: usize,
        rng: &mut DeterministicRng,
        disp: &mut [f64],
    ) -> CoreResult<()> {
        match (self, state) {
            (Self::Exact(ex), FrDynamic3dState::Exact(st)) => {
                ex.accumulate_repulsion(st, layout, k, iter, rng, disp)
            }
            (Self::BarnesHut(bh), FrDynamic3dState::BarnesHut(st)) => {
                bh.accumulate_repulsion(st.as_mut(), layout, k, iter, rng, disp)
            }
            (Self::Grid(gr), FrDynamic3dState::Grid(st)) => {
                gr.accumulate_repulsion(st, layout, k, iter, rng, disp)
            }
            _ => Err(CoreError::invalid_input(
                "FrBackend state mismatch in Fruchterman-Reingold 3D",
            )),
        }
    }
}
