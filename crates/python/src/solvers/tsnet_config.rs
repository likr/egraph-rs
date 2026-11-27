//! Shared inner wrapper, `TsNetStage`, and configuration builder for `TsNetSolver`.

use egraph::{
    layout as egraph_layout, neighborhood as egraph_neighborhood, spatial as egraph_spatial,
};
use egraph_layout::{TsNetBackend, TsNetConfig, TsNetSolver, TsNetStage};
use egraph_neighborhood::{SymmetrizationMode, TwoHopSamplingMode};
use egraph_spatial::{IbFftConfig, InterpolationOrder};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::ffi_convert::{StrictF64, StrictUsize, core_error_to_py, normalize_enum_key};
use crate::solver_common::impl_dual_solver;

use super::layout_dispatch::{impl_dual_manifold_solver, impl_dual_nd_solver};

impl_dual_solver!(TsNetInner, TsNetSolver);
impl_dual_nd_solver!(TsNetInner, TsNetSolver);
impl_dual_manifold_solver!(TsNetInner, TsNetSolver);

pub(super) fn tsnet_backend_str(backend: TsNetBackend) -> &'static str {
    match backend {
        TsNetBackend::Exact => "exact",
        TsNetBackend::BarnesHut => "barnes_hut",
        TsNetBackend::FftInterpolation => "fft",
        TsNetBackend::LinearFft => "linear_fft",
        TsNetBackend::Auto => "auto",
    }
}

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

    pub(super) fn resolved_backend(&self) -> &'static str {
        let b = match self {
            Self::F64(s) => s.resolved_backend(),
            Self::F32(s) => s.resolved_backend(),
        };
        tsnet_backend_str(b)
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

/// Optimization stage parameters for `TsNetSolver.from_probabilities_with_stages`.
#[pyclass(name = "TsNetStage", module = "egraph._core", from_py_object)]
#[derive(Clone, Debug)]
pub struct PyTsNetStage {
    pub(super) inner: TsNetStage,
}

#[pymethods]
impl PyTsNetStage {
    /// Construct a `TsNetStage` specifying iterations, exaggeration, and gradient term weights.
    #[new]
    #[pyo3(
        signature = (iterations, *, exaggeration = StrictF64(1.0), lambda_kl = StrictF64(1.0), lambda_c = StrictF64(0.0), lambda_r = StrictF64(0.0)),
        text_signature = "(iterations, *, exaggeration=1.0, lambda_kl=1.0, lambda_c=0.0, lambda_r=0.0)"
    )]
    fn new(
        iterations: StrictUsize,
        exaggeration: StrictF64,
        lambda_kl: StrictF64,
        lambda_c: StrictF64,
        lambda_r: StrictF64,
    ) -> PyResult<Self> {
        let inner = TsNetStage {
            iterations: iterations.0,
            exaggeration: exaggeration.0,
            lambda_kl: lambda_kl.0,
            lambda_c: lambda_c.0,
            lambda_r: lambda_r.0,
        };
        inner.validate().map_err(core_error_to_py)?;
        Ok(Self { inner })
    }

    /// Number of gradient descent iterations in this stage.
    #[getter]
    fn iterations(&self) -> usize {
        self.inner.iterations
    }

    /// Early exaggeration multiplier applied to high-dimensional probabilities.
    #[getter]
    fn exaggeration(&self) -> f64 {
        self.inner.exaggeration
    }

    /// Weight of the KL divergence gradient term.
    #[getter]
    fn lambda_kl(&self) -> f64 {
        self.inner.lambda_kl
    }

    /// Weight of the quadratic origin-compression gradient term.
    #[getter]
    fn lambda_c(&self) -> f64 {
        self.inner.lambda_c
    }

    /// Weight of the repulsive entropy gradient term.
    #[getter]
    fn lambda_r(&self) -> f64 {
        self.inner.lambda_r
    }

    fn __repr__(&self) -> String {
        format!(
            "TsNetStage(iterations={}, exaggeration={}, lambda_kl={}, lambda_c={}, lambda_r={})",
            self.inner.iterations,
            self.inner.exaggeration,
            self.inner.lambda_kl,
            self.inner.lambda_c,
            self.inner.lambda_r
        )
    }
}

pub(super) fn parse_tsnet_backend(raw: &str) -> Result<TsNetBackend, String> {
    match normalize_enum_key(raw).as_str() {
        "exact" => Ok(TsNetBackend::Exact),
        "barneshut" | "bh" => Ok(TsNetBackend::BarnesHut),
        "fft" | "fftinterpolation" | "fit" => Ok(TsNetBackend::FftInterpolation),
        "linearfft" | "l" => Ok(TsNetBackend::LinearFft),
        "auto" => Ok(TsNetBackend::Auto),
        _ => Err(format!(
            "unknown tsNET backend {raw:?}; expected 'exact', 'barnes_hut', 'fft', 'linear_fft', or 'auto'"
        )),
    }
}

pub(super) fn parse_tsnet_sampling_mode(raw: &str) -> Result<TwoHopSamplingMode, String> {
    match normalize_enum_key(raw).as_str() {
        "degreeorderedhubsuppressed" | "degreeordered" | "hubsuppressed" => {
            Ok(TwoHopSamplingMode::DegreeOrderedHubSuppressed)
        }
        "twosteprandomwalk" | "randomwalk" | "rw" => Ok(TwoHopSamplingMode::TwoStepRandomWalk),
        _ => Err(format!(
            "unknown 2-hop sampling mode {raw:?}; expected 'degree_ordered_hub_suppressed' or 'two_step_random_walk'"
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
    learning_rate_divisor: f64,
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
        stage_iterations: [iterations_stage1, iterations_stage2, iterations_stage3],
        exaggeration,
        lambda_c_stage1,
        lambda_c_stage2,
        lambda_c_stage3,
        lambda_r_stage3,
        learning_rate,
        learning_rate_divisor,
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

pub(super) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyTsNetStage>()?;
    Ok(())
}
