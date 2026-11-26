//! Shared inner wrapper and configuration builder for `TsNetSolver`.

use egraph::{
    layout as egraph_layout, neighborhood as egraph_neighborhood, spatial as egraph_spatial,
};
use egraph_layout::{TsNetBackend, TsNetConfig, TsNetSolver};
use egraph_neighborhood::SymmetrizationMode;
use egraph_spatial::{IbFftConfig, InterpolationOrder};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::ffi_convert::{core_error_to_py, normalize_enum_key};
use crate::solver_common::impl_dual_solver;

impl_dual_solver!(TsNetInner, TsNetSolver);

impl TsNetInner {
    pub(super) fn total_fft_2d_calls(&self) -> u64 {
        match self {
            Self::F64(s) => s.total_fft_2d_calls(),
            Self::F32(s) => s.total_fft_2d_calls(),
        }
    }

    pub(super) fn num_nodes(&self) -> usize {
        match self {
            Self::F64(s) => s.num_nodes(),
            Self::F32(s) => s.num_nodes(),
        }
    }

    pub(super) fn stage_index(&self) -> usize {
        match self {
            Self::F64(s) => s.stage_index(),
            Self::F32(s) => s.stage_index(),
        }
    }

    pub(super) fn current_stage_lr(&self) -> f64 {
        match self {
            Self::F64(s) => s.current_stage_lr(),
            Self::F32(s) => s.current_stage_lr(),
        }
    }

    pub(super) fn last_fft_stats_dict<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<Option<Bound<'py, PyDict>>> {
        let stats = match self {
            Self::F64(s) => s.last_fft_stats(),
            Self::F32(s) => s.last_fft_stats(),
        };
        match stats {
            Some(st) => {
                let d = PyDict::new(py);
                d.set_item("particle_forward_ffts", st.particle_forward_ffts)?;
                d.set_item("kernel_forward_ffts", st.kernel_forward_ffts)?;
                d.set_item("inverse_ffts", st.inverse_ffts)?;
                d.set_item("total_fft_2d_calls", st.total_fft_2d_calls)?;
                Ok(Some(d))
            }
            None => Ok(None),
        }
    }
}

pub(super) fn parse_tsnet_backend(raw: &str) -> Result<TsNetBackend, String> {
    match normalize_enum_key(raw).as_str() {
        "exact" => Ok(TsNetBackend::Exact),
        "barneshut" | "bh" => Ok(TsNetBackend::BarnesHut),
        "fft" | "fftinterpolation" | "fit" => Ok(TsNetBackend::FftInterpolation),
        "linearfft" | "l" => Ok(TsNetBackend::LinearFft),
        _ => Err(format!(
            "unknown tsNET backend {raw:?}; expected 'exact', 'barnes_hut', 'fft', or 'linear_fft'"
        )),
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn build_tsnet_config(
    backend: &str,
    perplexity_c: f64,
    min_perplexity: f64,
    max_perplexity: f64,
    power: f64,
    sigma_iters: usize,
    sigma_tolerance: f64,
    row_normalize: bool,
    iterations_stage1: usize,
    iterations_stage2: usize,
    iterations_stage3: usize,
    exaggeration: f64,
    lambda_c_stage1: f64,
    lambda_c_stage2: f64,
    lambda_c_stage3: f64,
    lambda_r_stage3: f64,
    learning_rate: Option<f64>,
    jacobi_precondition: bool,
    momentum: f64,
    epsilon_r: f64,
    theta: f64,
    intervals: usize,
    interpolation_order: usize,
    min_grid_dim: usize,
    max_grid_dim: usize,
    seed: u64,
) -> PyResult<TsNetConfig> {
    let parsed_backend = parse_tsnet_backend(backend).map_err(PyValueError::new_err)?;
    let order = InterpolationOrder::from_order(interpolation_order).map_err(core_error_to_py)?;
    let cfg = TsNetConfig {
        backend: parsed_backend,
        perplexity_c,
        min_perplexity,
        max_perplexity,
        power,
        sigma_iters,
        sigma_tolerance,
        symmetrization: if row_normalize {
            SymmetrizationMode::RowNormalized
        } else {
            SymmetrizationMode::StandardTsne
        },
        iterations_stage1,
        iterations_stage2,
        iterations_stage3,
        exaggeration,
        lambda_c_stage1,
        lambda_c_stage2,
        lambda_c_stage3,
        lambda_r_stage3,
        learning_rate,
        jacobi_precondition,
        momentum,
        epsilon_r,
        theta,
        ibfft: IbFftConfig {
            order,
            intervals,
            min_grid_dim,
            max_grid_dim,
        },
        seed,
    };
    cfg.validate().map_err(core_error_to_py)?;
    Ok(cfg)
}
