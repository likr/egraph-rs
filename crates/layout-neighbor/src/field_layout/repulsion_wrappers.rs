use egraph_core::{CoreError, CoreResult, Layout2d, Layout3d, Scalar};

use crate::backends::{BarnesHut, Exact, TFdpBackend2d, TFdpBackend3d};

use super::config::FieldFftStats;
use super::repulsion::TFdpRepulsion;
use super::repulsion_impls::{
    ExactTFdpState, TFdpBarnesHut2dState, TFdpBarnesHut3dState, TFdpIbFft2dState,
};

#[derive(Debug, Clone)]
enum TFdpBackend2dStateInner<S: Scalar> {
    IbFft(Box<TFdpIbFft2dState<S>>),
    BarnesHut(Box<TFdpBarnesHut2dState<S>>),
    Exact(ExactTFdpState),
}

/// Workspace state for [`TFdpBackend2d`] repulsion in [`TFdpSolver`](super::TFdpSolver).
#[doc(hidden)]
#[derive(Debug, Clone)]
pub struct TFdpBackend2dState<S: Scalar>(TFdpBackend2dStateInner<S>);

#[derive(Debug, Clone)]
enum TFdpBackend3dStateInner<S: Scalar> {
    BarnesHut(Box<TFdpBarnesHut3dState<S>>),
    Exact(ExactTFdpState),
}

/// Workspace state for [`TFdpBackend3d`] repulsion in [`TFdpSolver`](super::TFdpSolver).
#[doc(hidden)]
#[derive(Debug, Clone)]
pub struct TFdpBackend3dState<S: Scalar>(TFdpBackend3dStateInner<S>);

impl<S: Scalar> TFdpRepulsion<Layout2d<S>> for TFdpBackend2d {
    type State = TFdpBackend2dState<S>;

    fn validate_config(&self) -> CoreResult<()> {
        self.validate()
    }

    fn init_state(&self, num_nodes: usize) -> CoreResult<Self::State> {
        Ok(TFdpBackend2dState(match self {
            Self::IbFft(b) => TFdpBackend2dStateInner::IbFft(Box::new(b.init_state(num_nodes)?)),
            Self::BarnesHut(b) => {
                TFdpBackend2dStateInner::BarnesHut(Box::new(<BarnesHut as TFdpRepulsion<
                    Layout2d<S>,
                >>::init_state(
                    b, num_nodes
                )?))
            }
            Self::Exact(b) => {
                TFdpBackend2dStateInner::Exact(<Exact as TFdpRepulsion<Layout2d<S>>>::init_state(
                    b, num_nodes,
                )?)
            }
        }))
    }

    fn compute_repulsion(
        &self,
        state: &mut Self::State,
        layout: &Layout2d<S>,
        gamma: f64,
        rep_forces: &mut [f64],
    ) -> CoreResult<()> {
        match (self, &mut state.0) {
            (Self::IbFft(b), TFdpBackend2dStateInner::IbFft(st)) => {
                b.compute_repulsion(st, layout, gamma, rep_forces)
            }
            (Self::BarnesHut(b), TFdpBackend2dStateInner::BarnesHut(st)) => {
                b.compute_repulsion(st.as_mut(), layout, gamma, rep_forces)
            }
            (Self::Exact(b), TFdpBackend2dStateInner::Exact(st)) => {
                b.compute_repulsion(st, layout, gamma, rep_forces)
            }
            _ => Err(CoreError::numerical_breakdown(
                "TFdpBackend2d variant mismatch",
            )),
        }
    }

    #[inline]
    fn last_fft_stats<'a>(&self, state: &'a Self::State) -> Option<&'a FieldFftStats> {
        match (self, &state.0) {
            (Self::IbFft(b), TFdpBackend2dStateInner::IbFft(st)) => b.last_fft_stats(st),
            _ => None,
        }
    }

    #[inline]
    fn total_fft_2d_calls(&self, state: &Self::State) -> u64 {
        match (self, &state.0) {
            (Self::IbFft(b), TFdpBackend2dStateInner::IbFft(st)) => b.total_fft_2d_calls(st),
            _ => 0,
        }
    }
}

impl<S: Scalar> TFdpRepulsion<Layout3d<S>> for TFdpBackend3d {
    type State = TFdpBackend3dState<S>;

    fn validate_config(&self) -> CoreResult<()> {
        self.validate()
    }

    fn init_state(&self, num_nodes: usize) -> CoreResult<Self::State> {
        Ok(TFdpBackend3dState(match self {
            Self::BarnesHut(b) => {
                TFdpBackend3dStateInner::BarnesHut(Box::new(<BarnesHut as TFdpRepulsion<
                    Layout3d<S>,
                >>::init_state(
                    b, num_nodes
                )?))
            }
            Self::Exact(b) => {
                TFdpBackend3dStateInner::Exact(<Exact as TFdpRepulsion<Layout3d<S>>>::init_state(
                    b, num_nodes,
                )?)
            }
        }))
    }

    fn compute_repulsion(
        &self,
        state: &mut Self::State,
        layout: &Layout3d<S>,
        gamma: f64,
        rep_forces: &mut [f64],
    ) -> CoreResult<()> {
        match (self, &mut state.0) {
            (Self::BarnesHut(b), TFdpBackend3dStateInner::BarnesHut(st)) => {
                b.compute_repulsion(st.as_mut(), layout, gamma, rep_forces)
            }
            (Self::Exact(b), TFdpBackend3dStateInner::Exact(st)) => {
                b.compute_repulsion(st, layout, gamma, rep_forces)
            }
            _ => Err(CoreError::numerical_breakdown(
                "TFdpBackend3d variant mismatch",
            )),
        }
    }
}
