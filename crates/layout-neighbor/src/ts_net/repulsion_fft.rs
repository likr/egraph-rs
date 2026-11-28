use egraph_core::{CoreResult, Layout2d, Scalar};
use egraph_spatial_fft::{FitField2d, FitFieldConfig, FitFieldOutput, FitFieldStats, IbFftConfig};

use crate::backends::{FitFft, LinearFft};

use super::repulsion::TsNetRepulsion;
use super::repulsion_impls::check_force_slices;

/// Workspace state for [`FitFft`] and [`LinearFft`] repulsion in [`TsNetSolver`](super::TsNetSolver).
#[doc(hidden)]
#[derive(Debug, Clone)]
pub struct FitFft2dState<S: Scalar> {
    ibfft: IbFftConfig,
    pub(super) fit_field: FitField2d<S>,
    fit_out: FitFieldOutput<S>,
    last_fft_stats: Option<FitFieldStats>,
    total_fft_2d_calls: u64,
}

fn init_fit_state<S: Scalar>(
    ibfft: IbFftConfig,
    num_nodes: usize,
    epsilon_r: f64,
    any_entropy: bool,
    init_entropy: bool,
) -> CoreResult<FitFft2dState<S>> {
    let fit_field = FitField2d::new(
        FitFieldConfig::default()
            .with_ibfft(ibfft.clone())
            .with_epsilon_r(epsilon_r)
            .with_compute_entropy(init_entropy),
    )?;
    let mut fit_out = FitFieldOutput::new();
    fit_out.kl_repulsion_x.reserve(num_nodes);
    fit_out.kl_repulsion_y.reserve(num_nodes);
    if any_entropy {
        fit_out.entropy_repulsion_x.reserve(num_nodes);
        fit_out.entropy_repulsion_y.reserve(num_nodes);
    }
    Ok(FitFft2dState {
        ibfft,
        fit_field,
        fit_out,
        last_fft_stats: None,
        total_fft_2d_calls: 0,
    })
}

fn stage_enter_fit_state<S: Scalar>(
    state: &mut FitFft2dState<S>,
    epsilon_r: f64,
    compute_entropy: bool,
) -> CoreResult<()> {
    let cfg = state.fit_field.config();
    if cfg.compute_entropy != compute_entropy || (cfg.epsilon_r - epsilon_r).abs() > 0.0 {
        state.fit_field = FitField2d::new(
            FitFieldConfig::default()
                .with_ibfft(state.ibfft.clone())
                .with_epsilon_r(epsilon_r)
                .with_compute_entropy(compute_entropy),
        )?;
    }
    Ok(())
}

fn evaluate_fit_state<S: Scalar>(
    state: &mut FitFft2dState<S>,
    layout: &Layout2d<S>,
    epsilon_r: f64,
    compute_entropy: bool,
    rep_forces: &mut [f64],
    ent_forces: &mut [f64],
) -> CoreResult<()> {
    let n = layout.len();
    check_force_slices(
        n,
        2,
        compute_entropy,
        rep_forces,
        ent_forces,
        "FitFft::compute_repulsion",
    )?;
    if n < 2 {
        rep_forces.fill(0.0);
        if compute_entropy {
            ent_forces.fill(0.0);
        }
        return Ok(());
    }
    stage_enter_fit_state(state, epsilon_r, compute_entropy)?;
    state.fit_field.evaluate(layout, &mut state.fit_out)?;
    state.last_fft_stats = Some(state.fit_out.stats.clone());
    state.total_fft_2d_calls += state.fit_out.stats.total_fft_2d_calls;

    for i in 0..n {
        rep_forces[2 * i] = state.fit_out.kl_repulsion_x[i].to_f64();
        rep_forces[2 * i + 1] = state.fit_out.kl_repulsion_y[i].to_f64();
    }
    if compute_entropy {
        for i in 0..n {
            ent_forces[2 * i] = state.fit_out.entropy_repulsion_x[i].to_f64();
            ent_forces[2 * i + 1] = state.fit_out.entropy_repulsion_y[i].to_f64();
        }
    }
    Ok(())
}

impl<S: Scalar> TsNetRepulsion<Layout2d<S>> for FitFft {
    type State = FitFft2dState<S>;

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
        init_fit_state(
            self.ibfft.clone(),
            num_nodes,
            epsilon_r,
            any_entropy,
            init_entropy,
        )
    }

    fn on_stage_enter(
        &self,
        state: &mut Self::State,
        epsilon_r: f64,
        compute_entropy: bool,
    ) -> CoreResult<()> {
        stage_enter_fit_state(state, epsilon_r, compute_entropy)
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
        evaluate_fit_state(
            state,
            layout,
            epsilon_r,
            compute_entropy,
            rep_forces,
            ent_forces,
        )
    }

    #[inline]
    fn last_fft_stats<'a>(&self, state: &'a Self::State) -> Option<&'a FitFieldStats> {
        state.last_fft_stats.as_ref()
    }

    #[inline]
    fn total_fft_2d_calls(&self, state: &Self::State) -> u64 {
        state.total_fft_2d_calls
    }
}

impl<S: Scalar> TsNetRepulsion<Layout2d<S>> for LinearFft {
    type State = FitFft2dState<S>;

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
        init_fit_state(
            self.effective_ibfft(),
            num_nodes,
            epsilon_r,
            any_entropy,
            init_entropy,
        )
    }

    fn on_stage_enter(
        &self,
        state: &mut Self::State,
        epsilon_r: f64,
        compute_entropy: bool,
    ) -> CoreResult<()> {
        stage_enter_fit_state(state, epsilon_r, compute_entropy)
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
        evaluate_fit_state(
            state,
            layout,
            epsilon_r,
            compute_entropy,
            rep_forces,
            ent_forces,
        )
    }

    #[inline]
    fn last_fft_stats<'a>(&self, state: &'a Self::State) -> Option<&'a FitFieldStats> {
        state.last_fft_stats.as_ref()
    }

    #[inline]
    fn total_fft_2d_calls(&self, state: &Self::State) -> u64 {
        state.total_fft_2d_calls
    }
}
