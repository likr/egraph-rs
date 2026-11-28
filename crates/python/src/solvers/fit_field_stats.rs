//! Frozen telemetry and stage configuration classes shared by tsNET and t-FDP bindings.

use egraph::layout::{FieldFftStats, TsNetStage};
use egraph::spatial::{FitFieldStats, IbFftConfig, InterpolationOrder};
use pyo3::IntoPyObjectExt;
use pyo3::exceptions::PyKeyError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::errors::core_error_to_py;
use crate::ffi_convert::{StrictF64, StrictUsize};

/// Telemetry counters from the most recent interpolation-based FFT field evaluation.
#[pyclass(
    name = "FitFieldStats",
    module = "egraph._core",
    frozen,
    from_py_object
)]
#[derive(Debug, Clone)]
pub struct PyFitFieldStats {
    pub(crate) particle_forward_ffts: u64,
    pub(crate) kernel_forward_ffts: u64,
    pub(crate) inverse_ffts: u64,
    pub(crate) total_fft_2d_calls: u64,
}

impl PyFitFieldStats {
    pub(crate) fn from_fit(st: &FitFieldStats) -> Self {
        Self {
            particle_forward_ffts: st.particle_forward_ffts,
            kernel_forward_ffts: st.kernel_forward_ffts,
            inverse_ffts: st.inverse_ffts,
            total_fft_2d_calls: st.total_fft_2d_calls,
        }
    }
}

#[pymethods]
impl PyFitFieldStats {
    #[getter]
    fn particle_forward_ffts(&self) -> u64 {
        self.particle_forward_ffts
    }

    #[getter]
    fn kernel_forward_ffts(&self) -> u64 {
        self.kernel_forward_ffts
    }

    #[getter]
    fn inverse_ffts(&self) -> u64 {
        self.inverse_ffts
    }

    #[getter]
    fn total_fft_2d_calls(&self) -> u64 {
        self.total_fft_2d_calls
    }

    fn to_dict(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let d = PyDict::new(py);
        d.set_item("particle_forward_ffts", self.particle_forward_ffts)?;
        d.set_item("kernel_forward_ffts", self.kernel_forward_ffts)?;
        d.set_item("inverse_ffts", self.inverse_ffts)?;
        d.set_item("total_fft_2d_calls", self.total_fft_2d_calls)?;
        Ok(d.unbind())
    }

    fn keys(&self) -> Vec<&'static str> {
        vec![
            "particle_forward_ffts",
            "kernel_forward_ffts",
            "inverse_ffts",
            "total_fft_2d_calls",
        ]
    }

    fn __getitem__(&self, py: Python<'_>, key: &str) -> PyResult<Py<PyAny>> {
        match key {
            "particle_forward_ffts" => self.particle_forward_ffts.into_py_any(py),
            "kernel_forward_ffts" => self.kernel_forward_ffts.into_py_any(py),
            "inverse_ffts" => self.inverse_ffts.into_py_any(py),
            "total_fft_2d_calls" => self.total_fft_2d_calls.into_py_any(py),
            _ => Err(PyKeyError::new_err(key.to_owned())),
        }
    }

    fn __contains__(&self, key: &str) -> bool {
        matches!(
            key,
            "particle_forward_ffts" | "kernel_forward_ffts" | "inverse_ffts" | "total_fft_2d_calls"
        )
    }

    fn __repr__(&self) -> String {
        format!(
            "FitFieldStats(particle_forward_ffts={}, kernel_forward_ffts={}, inverse_ffts={}, total_fft_2d_calls={})",
            self.particle_forward_ffts,
            self.kernel_forward_ffts,
            self.inverse_ffts,
            self.total_fft_2d_calls
        )
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
        let inner = TsNetStage::new()
            .with_iterations(iterations.0)
            .with_exaggeration(exaggeration.0)
            .with_lambda_kl(lambda_kl.0)
            .with_lambda_c(lambda_c.0)
            .with_lambda_r(lambda_r.0);
        inner.validate().map_err(core_error_to_py)?;
        Ok(Self { inner })
    }

    #[getter]
    fn iterations(&self) -> usize {
        self.inner.iterations
    }

    #[getter]
    fn exaggeration(&self) -> f64 {
        self.inner.exaggeration
    }

    #[getter]
    fn lambda_kl(&self) -> f64 {
        self.inner.lambda_kl
    }

    #[getter]
    fn lambda_c(&self) -> f64 {
        self.inner.lambda_c
    }

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

/// Telemetry counters from an interpolation-based FFT field evaluation (`TFdpSolver` / `IbUmapSolver`).
#[pyclass(
    name = "FieldFftStats",
    module = "egraph._core",
    frozen,
    from_py_object
)]
#[derive(Debug, Clone)]
pub struct PyFieldFftStats {
    pub(super) particle_forward_ffts: u64,
    pub(super) kernel_forward_ffts: u64,
    pub(super) inverse_ffts: u64,
    pub(super) total_fft_2d_calls: u64,
    pub(super) nx: usize,
    pub(super) ny: usize,
}

impl PyFieldFftStats {
    pub(super) fn from_core(st: &FieldFftStats) -> Self {
        Self {
            particle_forward_ffts: st.particle_forward_ffts,
            kernel_forward_ffts: st.kernel_forward_ffts,
            inverse_ffts: st.inverse_ffts,
            total_fft_2d_calls: st.total_fft_2d_calls,
            nx: st.nx,
            ny: st.ny,
        }
    }
}

#[pymethods]
impl PyFieldFftStats {
    #[getter]
    fn particle_forward_ffts(&self) -> u64 {
        self.particle_forward_ffts
    }
    #[getter]
    fn kernel_forward_ffts(&self) -> u64 {
        self.kernel_forward_ffts
    }
    #[getter]
    fn inverse_ffts(&self) -> u64 {
        self.inverse_ffts
    }
    #[getter]
    fn total_fft_2d_calls(&self) -> u64 {
        self.total_fft_2d_calls
    }
    #[getter]
    fn nx(&self) -> usize {
        self.nx
    }
    #[getter]
    fn ny(&self) -> usize {
        self.ny
    }

    fn to_dict(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let d = PyDict::new(py);
        d.set_item("particle_forward_ffts", self.particle_forward_ffts)?;
        d.set_item("kernel_forward_ffts", self.kernel_forward_ffts)?;
        d.set_item("inverse_ffts", self.inverse_ffts)?;
        d.set_item("total_fft_2d_calls", self.total_fft_2d_calls)?;
        d.set_item("nx", self.nx)?;
        d.set_item("ny", self.ny)?;
        Ok(d.unbind())
    }

    fn keys(&self) -> Vec<&'static str> {
        vec![
            "particle_forward_ffts",
            "kernel_forward_ffts",
            "inverse_ffts",
            "total_fft_2d_calls",
            "nx",
            "ny",
        ]
    }

    fn __getitem__(&self, py: Python<'_>, key: &str) -> PyResult<Py<PyAny>> {
        match key {
            "particle_forward_ffts" => self.particle_forward_ffts.into_py_any(py),
            "kernel_forward_ffts" => self.kernel_forward_ffts.into_py_any(py),
            "inverse_ffts" => self.inverse_ffts.into_py_any(py),
            "total_fft_2d_calls" => self.total_fft_2d_calls.into_py_any(py),
            "nx" => self.nx.into_py_any(py),
            "ny" => self.ny.into_py_any(py),
            _ => Err(PyKeyError::new_err(key.to_owned())),
        }
    }

    fn __contains__(&self, key: &str) -> bool {
        matches!(
            key,
            "particle_forward_ffts"
                | "kernel_forward_ffts"
                | "inverse_ffts"
                | "total_fft_2d_calls"
                | "nx"
                | "ny"
        )
    }

    fn __repr__(&self) -> String {
        format!(
            "FieldFftStats(total_fft_2d_calls={}, nx={}, ny={})",
            self.total_fft_2d_calls, self.nx, self.ny
        )
    }
}

pub(super) fn build_ibfft_config(
    interpolation_order: usize,
    intervals: usize,
    min_grid_dim: usize,
    max_grid_dim: usize,
) -> PyResult<IbFftConfig> {
    let order = InterpolationOrder::from_order(interpolation_order).map_err(core_error_to_py)?;
    let cfg = IbFftConfig::default()
        .with_order(order)
        .with_intervals(intervals)
        .with_min_grid_dim(min_grid_dim)
        .with_max_grid_dim(max_grid_dim);
    cfg.validate().map_err(core_error_to_py)?;
    Ok(cfg)
}

pub(super) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyTsNetStage>()?;
    m.add_class::<PyFitFieldStats>()?;
    m.add_class::<PyFieldFftStats>()?;
    Ok(())
}
