use egraph_core::{CoreError, CoreResult, Layout2d, Scalar};
use egraph_spatial_fft::{Fft2dWorkspace, IbFft2d};

use crate::backends::{BarnesHut, Exact, IbFft, IbUmapBackend2d};

use super::config::FieldFftStats;
use super::ib_umap_repulsion::IbUmapRepulsion;
use super::quadtree::FieldQuadtree;
use super::repulsion_impls::check_rep_slice;

const REPULSIVE_DIST_SQ_EPS: f64 = 0.05;

/// Workspace state for [`Exact`] repulsion in [`IbUmapSolver`](super::IbUmapSolver).
#[doc(hidden)]
#[derive(Debug, Clone, Default)]
pub struct ExactIbUmapState;

/// Workspace state for 2D [`BarnesHut`] repulsion in [`IbUmapSolver`](super::IbUmapSolver).
#[doc(hidden)]
#[derive(Debug, Clone)]
pub struct IbUmapBarnesHut2dState<S: Scalar> {
    quadtree: FieldQuadtree,
    rep_x: Vec<S>,
    rep_y: Vec<S>,
}

impl<S: Scalar> IbUmapBarnesHut2dState<S> {
    #[cfg(test)]
    pub(super) fn quadtree_node_capacity(&self) -> usize {
        self.quadtree.nodes.capacity()
    }
}

/// Workspace state for [`IbFft`] repulsion in [`IbUmapSolver`](super::IbUmapSolver).
#[doc(hidden)]
#[derive(Debug, Clone)]
pub struct IbUmapIbFft2dState<S: Scalar> {
    ibfft: IbFft2d<S>,
    fft_ws: Fft2dWorkspace,
    phi_1: Vec<S>,
    rep_x: Vec<S>,
    rep_y: Vec<S>,
    last_fft_stats: Option<FieldFftStats>,
}

#[derive(Debug, Clone)]
enum IbUmapBackend2dStateInner<S: Scalar> {
    IbFft(Box<IbUmapIbFft2dState<S>>),
    BarnesHut(Box<IbUmapBarnesHut2dState<S>>),
    Exact(ExactIbUmapState),
}

/// Workspace state for [`IbUmapBackend2d`] repulsion in [`IbUmapSolver`](super::IbUmapSolver).
#[doc(hidden)]
#[derive(Debug, Clone)]
pub struct IbUmapBackend2dState<S: Scalar>(IbUmapBackend2dStateInner<S>);

impl<S: Scalar> IbUmapRepulsion<Layout2d<S>> for Exact {
    type State = ExactIbUmapState;

    fn validate_config(&self) -> CoreResult<()> {
        Ok(())
    }

    fn init_state(&self, _num_nodes: usize) -> CoreResult<Self::State> {
        Ok(ExactIbUmapState)
    }

    fn compute_repulsion(
        &self,
        _state: &mut Self::State,
        layout: &Layout2d<S>,
        a: f64,
        b: f64,
        c_k: f64,
        rep_forces: &mut [f64],
    ) -> CoreResult<()> {
        let n = layout.len();
        check_rep_slice(n, 2, rep_forces, "Exact::compute_repulsion(IbUmap 2d)")?;
        if n < 2 {
            rep_forces.fill(0.0);
            return Ok(());
        }
        let two_b_ck = 2.0 * b * c_k;
        let xs = layout.x();
        let ys = layout.y();
        for i in 0..n {
            let xi = xs[i].to_f64();
            let yi = ys[i].to_f64();
            let mut fx = 0.0_f64;
            let mut fy = 0.0_f64;
            for j in 0..n {
                if j == i {
                    continue;
                }
                let dx = xi - xs[j].to_f64();
                let dy = yi - ys[j].to_f64();
                let d2 = dx * dx + dy * dy;
                let k = two_b_ck / ((REPULSIVE_DIST_SQ_EPS + d2) * (1.0 + a * d2.powf(b)));
                fx += k * dx;
                fy += k * dy;
            }
            rep_forces[2 * i] = S::from_f64(fx).to_f64();
            rep_forces[2 * i + 1] = S::from_f64(fy).to_f64();
        }
        Ok(())
    }
}

impl<S: Scalar> IbUmapRepulsion<Layout2d<S>> for BarnesHut {
    type State = IbUmapBarnesHut2dState<S>;

    fn validate_config(&self) -> CoreResult<()> {
        self.validate()
    }

    fn init_state(&self, num_nodes: usize) -> CoreResult<Self::State> {
        Ok(IbUmapBarnesHut2dState {
            quadtree: FieldQuadtree::new(),
            rep_x: vec![S::ZERO; num_nodes],
            rep_y: vec![S::ZERO; num_nodes],
        })
    }

    fn compute_repulsion(
        &self,
        state: &mut Self::State,
        layout: &Layout2d<S>,
        a: f64,
        b: f64,
        c_k: f64,
        rep_forces: &mut [f64],
    ) -> CoreResult<()> {
        let n = layout.len();
        check_rep_slice(n, 2, rep_forces, "BarnesHut::compute_repulsion(IbUmap 2d)")?;
        if n < 2 {
            rep_forces.fill(0.0);
            return Ok(());
        }
        if state.rep_x.len() != n {
            state.rep_x.resize(n, S::ZERO);
            state.rep_y.resize(n, S::ZERO);
        }
        let two_b_ck = 2.0 * b * c_k;
        state.quadtree.build_xy(layout.x(), layout.y())?;
        state.quadtree.compute_isotropic_repulsion(
            self.theta,
            |d2| {
                let d2_pos = d2.max(0.0);
                two_b_ck / ((REPULSIVE_DIST_SQ_EPS + d2_pos) * (1.0 + a * d2_pos.powf(b)))
            },
            &mut state.rep_x,
            &mut state.rep_y,
        );
        for i in 0..n {
            rep_forces[2 * i] = state.rep_x[i].to_f64();
            rep_forces[2 * i + 1] = state.rep_y[i].to_f64();
        }
        Ok(())
    }
}

impl<S: Scalar> IbUmapRepulsion<Layout2d<S>> for IbFft {
    type State = IbUmapIbFft2dState<S>;

    fn validate_config(&self) -> CoreResult<()> {
        self.validate()
    }

    fn init_state(&self, num_nodes: usize) -> CoreResult<Self::State> {
        Ok(IbUmapIbFft2dState {
            ibfft: IbFft2d::new(self.ibfft.clone())?,
            fft_ws: Fft2dWorkspace::new(),
            phi_1: vec![S::ZERO; num_nodes],
            rep_x: vec![S::ZERO; num_nodes],
            rep_y: vec![S::ZERO; num_nodes],
            last_fft_stats: None,
        })
    }

    fn compute_repulsion(
        &self,
        state: &mut Self::State,
        layout: &Layout2d<S>,
        a: f64,
        b: f64,
        c_k: f64,
        rep_forces: &mut [f64],
    ) -> CoreResult<()> {
        let n = layout.len();
        check_rep_slice(n, 2, rep_forces, "IbFft::compute_repulsion(IbUmap 2d)")?;
        if n < 2 {
            rep_forces.fill(0.0);
            return Ok(());
        }
        if state.phi_1.len() != n {
            state.phi_1.resize(n, S::ZERO);
            state.rep_x.resize(n, S::ZERO);
            state.rep_y.resize(n, S::ZERO);
        }
        let two_b_ck = 2.0 * b * c_k;
        let before_calls = state.fft_ws.fft_2d_call_count();
        let particle_fwd = state
            .ibfft
            .prepare_xy(&mut state.fft_ws, layout.x(), layout.y())?;
        let soft = state
            .ibfft
            .grid_spec()
            .map_or(0.0, |spec| spec.softening_sq);
        let eps_eff = REPULSIVE_DIST_SQ_EPS + soft;
        state.ibfft.evaluate_isotropic_kernel(
            &mut state.fft_ws,
            |d2| {
                let d2_pos = d2.max(0.0);
                two_b_ck / ((eps_eff + d2_pos) * (1.0 + a * d2_pos.powf(b)))
            },
            &mut state.phi_1,
            &mut state.rep_x,
            &mut state.rep_y,
        )?;
        let total_calls = state
            .fft_ws
            .fft_2d_call_count()
            .saturating_sub(before_calls);
        let spec = state.ibfft.grid_spec();
        state.last_fft_stats = Some(FieldFftStats {
            particle_forward_ffts: particle_fwd,
            kernel_forward_ffts: u64::from(total_calls > particle_fwd),
            inverse_ffts: total_calls.saturating_sub(particle_fwd + 1),
            total_fft_2d_calls: total_calls,
            nx: spec.map_or(0, |s| s.nx),
            ny: spec.map_or(0, |s| s.ny),
        });
        for i in 0..n {
            rep_forces[2 * i] = state.rep_x[i].to_f64();
            rep_forces[2 * i + 1] = state.rep_y[i].to_f64();
        }
        Ok(())
    }

    #[inline]
    fn last_fft_stats<'a>(&self, state: &'a Self::State) -> Option<&'a FieldFftStats> {
        state.last_fft_stats.as_ref()
    }

    #[inline]
    fn total_fft_2d_calls(&self, state: &Self::State) -> u64 {
        state.fft_ws.fft_2d_call_count()
    }
}

impl<S: Scalar> IbUmapRepulsion<Layout2d<S>> for IbUmapBackend2d {
    type State = IbUmapBackend2dState<S>;

    fn validate_config(&self) -> CoreResult<()> {
        self.validate()
    }

    fn init_state(&self, num_nodes: usize) -> CoreResult<Self::State> {
        Ok(IbUmapBackend2dState(match self {
            Self::IbFft(b) => IbUmapBackend2dStateInner::IbFft(Box::new(b.init_state(num_nodes)?)),
            Self::BarnesHut(b) => {
                IbUmapBackend2dStateInner::BarnesHut(Box::new(<BarnesHut as IbUmapRepulsion<
                    Layout2d<S>,
                >>::init_state(
                    b, num_nodes
                )?))
            }
            Self::Exact(b) => {
                IbUmapBackend2dStateInner::Exact(
                    <Exact as IbUmapRepulsion<Layout2d<S>>>::init_state(b, num_nodes)?,
                )
            }
        }))
    }

    fn compute_repulsion(
        &self,
        state: &mut Self::State,
        layout: &Layout2d<S>,
        a: f64,
        b: f64,
        c_k: f64,
        rep_forces: &mut [f64],
    ) -> CoreResult<()> {
        match (self, &mut state.0) {
            (Self::IbFft(backend), IbUmapBackend2dStateInner::IbFft(st)) => {
                backend.compute_repulsion(st, layout, a, b, c_k, rep_forces)
            }
            (Self::BarnesHut(backend), IbUmapBackend2dStateInner::BarnesHut(st)) => {
                backend.compute_repulsion(st.as_mut(), layout, a, b, c_k, rep_forces)
            }
            (Self::Exact(backend), IbUmapBackend2dStateInner::Exact(st)) => {
                backend.compute_repulsion(st, layout, a, b, c_k, rep_forces)
            }
            _ => Err(CoreError::numerical_breakdown(
                "IbUmapBackend2d variant mismatch",
            )),
        }
    }

    #[inline]
    fn last_fft_stats<'a>(&self, state: &'a Self::State) -> Option<&'a FieldFftStats> {
        match (self, &state.0) {
            (Self::IbFft(backend), IbUmapBackend2dStateInner::IbFft(st)) => {
                backend.last_fft_stats(st)
            }
            _ => None,
        }
    }

    #[inline]
    fn total_fft_2d_calls(&self, state: &Self::State) -> u64 {
        match (self, &state.0) {
            (Self::IbFft(backend), IbUmapBackend2dStateInner::IbFft(st)) => {
                backend.total_fft_2d_calls(st)
            }
            _ => 0,
        }
    }
}
