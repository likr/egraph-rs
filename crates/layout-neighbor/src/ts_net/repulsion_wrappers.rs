use egraph_core::{CoreError, CoreResult, Layout2d, Layout3d, Scalar};
#[cfg(test)]
use egraph_spatial_fft::FitField2d;
use egraph_spatial_fft::FitFieldStats;

use crate::backends::{Auto2d, Auto3d, BarnesHut, Exact, TsNetBackend2d, TsNetBackend3d};

use super::repulsion::TsNetRepulsion;
use super::repulsion_fft::FitFft2dState;
use super::repulsion_impls::{BarnesHut2dState, BarnesHut3dState, ExactTsNetState};

#[derive(Debug, Clone)]
enum Auto2dStateInner<S: Scalar> {
    Exact(ExactTsNetState),
    LinearFft(Box<FitFft2dState<S>>),
}

/// Workspace state for [`Auto2d`] repulsion in [`TsNetSolver`](super::TsNetSolver).
#[doc(hidden)]
#[derive(Debug, Clone)]
pub struct Auto2dState<S: Scalar>(Auto2dStateInner<S>);

impl<S: Scalar> Auto2dState<S> {
    #[cfg(test)]
    pub(super) fn is_exact(&self) -> bool {
        matches!(self.0, Auto2dStateInner::Exact(_))
    }

    #[cfg(test)]
    pub(super) fn fit_field(&self) -> Option<&FitField2d<S>> {
        match &self.0 {
            Auto2dStateInner::Exact(_) => None,
            Auto2dStateInner::LinearFft(st) => Some(&st.fit_field),
        }
    }
}

#[derive(Debug, Clone)]
enum Auto3dStateInner<S: Scalar> {
    Exact(ExactTsNetState),
    BarnesHut(Box<BarnesHut3dState<S>>),
}

/// Workspace state for [`Auto3d`] repulsion in [`TsNetSolver`](super::TsNetSolver).
#[doc(hidden)]
#[derive(Debug, Clone)]
pub struct Auto3dState<S: Scalar>(Auto3dStateInner<S>);

impl<S: Scalar> Auto3dState<S> {
    #[cfg(test)]
    pub(super) fn is_exact(&self) -> bool {
        matches!(self.0, Auto3dStateInner::Exact(_))
    }
}

#[derive(Debug, Clone)]
enum Backend2dStateInner<S: Scalar> {
    Exact(ExactTsNetState),
    BarnesHut(BarnesHut2dState<S>),
    FitFft(FitFft2dState<S>),
    LinearFft(FitFft2dState<S>),
    Auto(Auto2dState<S>),
}

/// Workspace state for [`TsNetBackend2d`] repulsion in [`TsNetSolver`](super::TsNetSolver).
#[doc(hidden)]
#[derive(Debug, Clone)]
pub struct Backend2dState<S: Scalar>(Backend2dStateInner<S>);

#[derive(Debug, Clone)]
enum Backend3dStateInner<S: Scalar> {
    Exact(ExactTsNetState),
    BarnesHut(Box<BarnesHut3dState<S>>),
    Auto(Auto3dState<S>),
}

/// Workspace state for [`TsNetBackend3d`] repulsion in [`TsNetSolver`](super::TsNetSolver).
#[doc(hidden)]
#[derive(Debug, Clone)]
pub struct Backend3dState<S: Scalar>(Backend3dStateInner<S>);

impl<S: Scalar> TsNetRepulsion<Layout2d<S>> for Auto2d {
    type State = Auto2dState<S>;

    fn validate_config(&self) -> CoreResult<()> {
        self.validate()
    }

    fn init_state(
        &self,
        num_nodes: usize,
        epsilon_r: f64,
        any_entropy: bool,
        init_entropy: bool,
    ) -> CoreResult<Self::State> {
        if num_nodes <= self.exact_threshold {
            Ok(Auto2dState(Auto2dStateInner::Exact(
                <Exact as TsNetRepulsion<Layout2d<S>>>::init_state(
                    &Exact,
                    num_nodes,
                    epsilon_r,
                    any_entropy,
                    init_entropy,
                )?,
            )))
        } else {
            Ok(Auto2dState(Auto2dStateInner::LinearFft(Box::new(
                self.linear_fft
                    .init_state(num_nodes, epsilon_r, any_entropy, init_entropy)?,
            ))))
        }
    }

    fn on_stage_enter(
        &self,
        state: &mut Self::State,
        epsilon_r: f64,
        compute_entropy: bool,
    ) -> CoreResult<()> {
        match &mut state.0 {
            Auto2dStateInner::Exact(_) => Ok(()),
            Auto2dStateInner::LinearFft(st) => {
                self.linear_fft
                    .on_stage_enter(st, epsilon_r, compute_entropy)
            }
        }
    }

    fn compute_repulsion(
        &self,
        state: &mut Self::State,
        layout: &Layout2d<S>,
        epsilon_r: f64,
        compute_entropy: bool,
        rep_forces: &mut [f64],
        ent_forces: &mut [f64],
    ) -> CoreResult<()> {
        match &mut state.0 {
            Auto2dStateInner::Exact(st) => Exact.compute_repulsion(
                st,
                layout,
                epsilon_r,
                compute_entropy,
                rep_forces,
                ent_forces,
            ),
            Auto2dStateInner::LinearFft(st) => self.linear_fft.compute_repulsion(
                st,
                layout,
                epsilon_r,
                compute_entropy,
                rep_forces,
                ent_forces,
            ),
        }
    }

    #[inline]
    fn last_fft_stats<'a>(&self, state: &'a Self::State) -> Option<&'a FitFieldStats> {
        match &state.0 {
            Auto2dStateInner::Exact(_) => None,
            Auto2dStateInner::LinearFft(st) => self.linear_fft.last_fft_stats(st),
        }
    }

    #[inline]
    fn total_fft_2d_calls(&self, state: &Self::State) -> u64 {
        match &state.0 {
            Auto2dStateInner::Exact(_) => 0,
            Auto2dStateInner::LinearFft(st) => self.linear_fft.total_fft_2d_calls(st),
        }
    }
}

impl<S: Scalar> TsNetRepulsion<Layout3d<S>> for Auto3d {
    type State = Auto3dState<S>;

    fn validate_config(&self) -> CoreResult<()> {
        self.validate()
    }

    fn init_state(
        &self,
        num_nodes: usize,
        epsilon_r: f64,
        any_entropy: bool,
        init_entropy: bool,
    ) -> CoreResult<Self::State> {
        if num_nodes <= self.exact_threshold {
            Ok(Auto3dState(Auto3dStateInner::Exact(
                <Exact as TsNetRepulsion<Layout3d<S>>>::init_state(
                    &Exact,
                    num_nodes,
                    epsilon_r,
                    any_entropy,
                    init_entropy,
                )?,
            )))
        } else {
            Ok(Auto3dState(Auto3dStateInner::BarnesHut(Box::new(
                <BarnesHut as TsNetRepulsion<Layout3d<S>>>::init_state(
                    &self.barnes_hut,
                    num_nodes,
                    epsilon_r,
                    any_entropy,
                    init_entropy,
                )?,
            ))))
        }
    }

    fn compute_repulsion(
        &self,
        state: &mut Self::State,
        layout: &Layout3d<S>,
        epsilon_r: f64,
        compute_entropy: bool,
        rep_forces: &mut [f64],
        ent_forces: &mut [f64],
    ) -> CoreResult<()> {
        match &mut state.0 {
            Auto3dStateInner::Exact(st) => Exact.compute_repulsion(
                st,
                layout,
                epsilon_r,
                compute_entropy,
                rep_forces,
                ent_forces,
            ),
            Auto3dStateInner::BarnesHut(st) => self.barnes_hut.compute_repulsion(
                st.as_mut(),
                layout,
                epsilon_r,
                compute_entropy,
                rep_forces,
                ent_forces,
            ),
        }
    }
}

impl<S: Scalar> TsNetRepulsion<Layout2d<S>> for TsNetBackend2d {
    type State = Backend2dState<S>;

    fn validate_config(&self) -> CoreResult<()> {
        self.validate()
    }

    fn init_state(
        &self,
        num_nodes: usize,
        epsilon_r: f64,
        any_entropy: bool,
        init_entropy: bool,
    ) -> CoreResult<Self::State> {
        Ok(Backend2dState(match self {
            Self::Exact(b) => {
                Backend2dStateInner::Exact(<Exact as TsNetRepulsion<Layout2d<S>>>::init_state(
                    b,
                    num_nodes,
                    epsilon_r,
                    any_entropy,
                    init_entropy,
                )?)
            }
            Self::BarnesHut(b) => Backend2dStateInner::BarnesHut(<BarnesHut as TsNetRepulsion<
                Layout2d<S>,
            >>::init_state(
                b,
                num_nodes,
                epsilon_r,
                any_entropy,
                init_entropy,
            )?),
            Self::FitFft(b) => Backend2dStateInner::FitFft(b.init_state(
                num_nodes,
                epsilon_r,
                any_entropy,
                init_entropy,
            )?),
            Self::LinearFft(b) => Backend2dStateInner::LinearFft(b.init_state(
                num_nodes,
                epsilon_r,
                any_entropy,
                init_entropy,
            )?),
            Self::Auto(b) => Backend2dStateInner::Auto(b.init_state(
                num_nodes,
                epsilon_r,
                any_entropy,
                init_entropy,
            )?),
        }))
    }

    fn on_stage_enter(
        &self,
        state: &mut Self::State,
        epsilon_r: f64,
        compute_entropy: bool,
    ) -> CoreResult<()> {
        match (self, &mut state.0) {
            (Self::FitFft(b), Backend2dStateInner::FitFft(st)) => {
                b.on_stage_enter(st, epsilon_r, compute_entropy)
            }
            (Self::LinearFft(b), Backend2dStateInner::LinearFft(st)) => {
                b.on_stage_enter(st, epsilon_r, compute_entropy)
            }
            (Self::Auto(b), Backend2dStateInner::Auto(st)) => {
                b.on_stage_enter(st, epsilon_r, compute_entropy)
            }
            _ => Ok(()),
        }
    }

    fn compute_repulsion(
        &self,
        state: &mut Self::State,
        layout: &Layout2d<S>,
        epsilon_r: f64,
        compute_entropy: bool,
        rep_forces: &mut [f64],
        ent_forces: &mut [f64],
    ) -> CoreResult<()> {
        match (self, &mut state.0) {
            (Self::Exact(b), Backend2dStateInner::Exact(st)) => b.compute_repulsion(
                st,
                layout,
                epsilon_r,
                compute_entropy,
                rep_forces,
                ent_forces,
            ),
            (Self::BarnesHut(b), Backend2dStateInner::BarnesHut(st)) => b.compute_repulsion(
                st,
                layout,
                epsilon_r,
                compute_entropy,
                rep_forces,
                ent_forces,
            ),
            (Self::FitFft(b), Backend2dStateInner::FitFft(st)) => b.compute_repulsion(
                st,
                layout,
                epsilon_r,
                compute_entropy,
                rep_forces,
                ent_forces,
            ),
            (Self::LinearFft(b), Backend2dStateInner::LinearFft(st)) => b.compute_repulsion(
                st,
                layout,
                epsilon_r,
                compute_entropy,
                rep_forces,
                ent_forces,
            ),
            (Self::Auto(b), Backend2dStateInner::Auto(st)) => b.compute_repulsion(
                st,
                layout,
                epsilon_r,
                compute_entropy,
                rep_forces,
                ent_forces,
            ),
            _ => Err(CoreError::numerical_breakdown(
                "TsNetBackend2d variant mismatch",
            )),
        }
    }

    #[inline]
    fn last_fft_stats<'a>(&self, state: &'a Self::State) -> Option<&'a FitFieldStats> {
        match (self, &state.0) {
            (Self::FitFft(b), Backend2dStateInner::FitFft(st)) => b.last_fft_stats(st),
            (Self::LinearFft(b), Backend2dStateInner::LinearFft(st)) => b.last_fft_stats(st),
            (Self::Auto(b), Backend2dStateInner::Auto(st)) => b.last_fft_stats(st),
            _ => None,
        }
    }

    #[inline]
    fn total_fft_2d_calls(&self, state: &Self::State) -> u64 {
        match (self, &state.0) {
            (Self::FitFft(b), Backend2dStateInner::FitFft(st)) => b.total_fft_2d_calls(st),
            (Self::LinearFft(b), Backend2dStateInner::LinearFft(st)) => b.total_fft_2d_calls(st),
            (Self::Auto(b), Backend2dStateInner::Auto(st)) => b.total_fft_2d_calls(st),
            _ => 0,
        }
    }
}

impl<S: Scalar> TsNetRepulsion<Layout3d<S>> for TsNetBackend3d {
    type State = Backend3dState<S>;

    fn validate_config(&self) -> CoreResult<()> {
        self.validate()
    }

    fn init_state(
        &self,
        num_nodes: usize,
        epsilon_r: f64,
        any_entropy: bool,
        init_entropy: bool,
    ) -> CoreResult<Self::State> {
        Ok(Backend3dState(match self {
            Self::Exact(b) => {
                Backend3dStateInner::Exact(<Exact as TsNetRepulsion<Layout3d<S>>>::init_state(
                    b,
                    num_nodes,
                    epsilon_r,
                    any_entropy,
                    init_entropy,
                )?)
            }
            Self::BarnesHut(b) => {
                Backend3dStateInner::BarnesHut(Box::new(<BarnesHut as TsNetRepulsion<
                    Layout3d<S>,
                >>::init_state(
                    b,
                    num_nodes,
                    epsilon_r,
                    any_entropy,
                    init_entropy,
                )?))
            }
            Self::Auto(b) => Backend3dStateInner::Auto(b.init_state(
                num_nodes,
                epsilon_r,
                any_entropy,
                init_entropy,
            )?),
        }))
    }

    fn compute_repulsion(
        &self,
        state: &mut Self::State,
        layout: &Layout3d<S>,
        epsilon_r: f64,
        compute_entropy: bool,
        rep_forces: &mut [f64],
        ent_forces: &mut [f64],
    ) -> CoreResult<()> {
        match (self, &mut state.0) {
            (Self::Exact(b), Backend3dStateInner::Exact(st)) => b.compute_repulsion(
                st,
                layout,
                epsilon_r,
                compute_entropy,
                rep_forces,
                ent_forces,
            ),
            (Self::BarnesHut(b), Backend3dStateInner::BarnesHut(st)) => b.compute_repulsion(
                st.as_mut(),
                layout,
                epsilon_r,
                compute_entropy,
                rep_forces,
                ent_forces,
            ),
            (Self::Auto(b), Backend3dStateInner::Auto(st)) => b.compute_repulsion(
                st,
                layout,
                epsilon_r,
                compute_entropy,
                rep_forces,
                ent_forces,
            ),
            _ => Err(CoreError::numerical_breakdown(
                "TsNetBackend3d variant mismatch",
            )),
        }
    }
}
