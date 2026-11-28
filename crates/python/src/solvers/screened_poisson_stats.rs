//! Frozen diagnostics and field evaluation results for Screened Poisson density fields.

use egraph::spatial::{ScreenedPoissonOutput, ScreenedPoissonStats};
use numpy::PyArray1;
use pyo3::exceptions::PyKeyError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

#[derive(Clone)]
pub(super) enum ScreenedPoissonForces {
    F64(Vec<f64>, Vec<f64>),
    F32(Vec<f32>, Vec<f32>),
}

/// Telemetry from a Screened Poisson density field evaluation.
#[pyclass(
    name = "ScreenedPoissonStats",
    module = "egraph._core",
    frozen,
    from_py_object
)]
#[derive(Clone)]
pub struct PyScreenedPoissonStats {
    inner: ScreenedPoissonStats,
}

impl PyScreenedPoissonStats {
    pub(crate) fn from_inner(inner: ScreenedPoissonStats) -> Self {
        Self { inner }
    }

    pub(super) fn fill_dict(&self, bound: &Bound<'_, PyDict>) -> PyResult<()> {
        bound.set_item("energy", self.inner.energy)?;
        bound.set_item("max_density", self.inner.max_density)?;
        bound.set_item("excess_fraction", self.inner.excess_fraction)?;
        bound.set_item("excess_mass", self.inner.excess_mass)?;
        bound.set_item("sigma_used", self.inner.sigma_used)?;
        bound.set_item("kappa_used", self.inner.kappa_used)?;
        bound.set_item("rho_max_used", self.inner.rho_max_used)?;
        bound.set_item("grid_dim", self.inner.grid_dim)?;
        bound.set_item("pad_dim", self.inner.pad_dim)?;
        bound.set_item("fft_2d_calls", self.inner.fft_2d_calls)?;
        bound.set_item("early_exit_no_excess", self.inner.early_exit_no_excess)?;
        Ok(())
    }
}

#[pymethods]
impl PyScreenedPoissonStats {
    #[getter]
    fn energy(&self) -> f64 {
        self.inner.energy
    }

    #[getter]
    fn max_density(&self) -> f64 {
        self.inner.max_density
    }

    #[getter]
    fn excess_fraction(&self) -> f64 {
        self.inner.excess_fraction
    }

    #[getter]
    fn excess_mass(&self) -> f64 {
        self.inner.excess_mass
    }

    #[getter]
    fn sigma_used(&self) -> f64 {
        self.inner.sigma_used
    }

    #[getter]
    fn kappa_used(&self) -> f64 {
        self.inner.kappa_used
    }

    #[getter]
    fn rho_max_used(&self) -> f64 {
        self.inner.rho_max_used
    }

    #[getter]
    fn grid_dim(&self) -> usize {
        self.inner.grid_dim
    }

    #[getter]
    fn pad_dim(&self) -> usize {
        self.inner.pad_dim
    }

    #[getter]
    fn fft_2d_calls(&self) -> u64 {
        self.inner.fft_2d_calls
    }

    #[getter]
    fn early_exit_no_excess(&self) -> bool {
        self.inner.early_exit_no_excess
    }

    #[pyo3(signature = (), text_signature = "($self)")]
    fn to_dict(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let dict = PyDict::new(py);
        self.fill_dict(&dict)?;
        Ok(dict.unbind())
    }

    #[pyo3(signature = (), text_signature = "($self)")]
    fn keys(&self) -> Vec<&'static str> {
        vec![
            "energy",
            "max_density",
            "excess_fraction",
            "excess_mass",
            "sigma_used",
            "kappa_used",
            "rho_max_used",
            "grid_dim",
            "pad_dim",
            "fft_2d_calls",
            "early_exit_no_excess",
        ]
    }

    fn __contains__(&self, key: &str) -> bool {
        matches!(
            key,
            "energy"
                | "max_density"
                | "excess_fraction"
                | "excess_mass"
                | "sigma_used"
                | "kappa_used"
                | "rho_max_used"
                | "grid_dim"
                | "pad_dim"
                | "fft_2d_calls"
                | "early_exit_no_excess"
        )
    }

    fn __getitem__(&self, py: Python<'_>, key: &str) -> PyResult<Py<PyAny>> {
        match key {
            "energy" => Ok(self.inner.energy.into_pyobject(py)?.into_any().unbind()),
            "max_density" => Ok(self
                .inner
                .max_density
                .into_pyobject(py)?
                .into_any()
                .unbind()),
            "excess_fraction" => Ok(self
                .inner
                .excess_fraction
                .into_pyobject(py)?
                .into_any()
                .unbind()),
            "excess_mass" => Ok(self
                .inner
                .excess_mass
                .into_pyobject(py)?
                .into_any()
                .unbind()),
            "sigma_used" => Ok(self.inner.sigma_used.into_pyobject(py)?.into_any().unbind()),
            "kappa_used" => Ok(self.inner.kappa_used.into_pyobject(py)?.into_any().unbind()),
            "rho_max_used" => Ok(self
                .inner
                .rho_max_used
                .into_pyobject(py)?
                .into_any()
                .unbind()),
            "grid_dim" => Ok(self.inner.grid_dim.into_pyobject(py)?.into_any().unbind()),
            "pad_dim" => Ok(self.inner.pad_dim.into_pyobject(py)?.into_any().unbind()),
            "fft_2d_calls" => Ok(self
                .inner
                .fft_2d_calls
                .into_pyobject(py)?
                .into_any()
                .unbind()),
            "early_exit_no_excess" => Ok(self
                .inner
                .early_exit_no_excess
                .into_pyobject(py)?
                .to_owned()
                .into_any()
                .unbind()),
            _ => Err(PyKeyError::new_err(format!(
                "unknown ScreenedPoissonStats key {key:?}"
            ))),
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "ScreenedPoissonStats(energy={:.6}, max_density={:.4}, excess_fraction={:.4})",
            self.inner.energy, self.inner.max_density, self.inner.excess_fraction
        )
    }
}

/// Output forces and telemetry from evaluating `ScreenedPoissonField2d`.
#[pyclass(
    name = "ScreenedPoissonOutput",
    module = "egraph._core",
    frozen,
    from_py_object
)]
#[derive(Clone)]
pub struct PyScreenedPoissonOutput {
    stats: ScreenedPoissonStats,
    forces: ScreenedPoissonForces,
}

impl PyScreenedPoissonOutput {
    pub(super) fn from_f64(out: ScreenedPoissonOutput<f64>) -> Self {
        Self {
            stats: out.stats,
            forces: ScreenedPoissonForces::F64(out.force_x, out.force_y),
        }
    }

    pub(super) fn from_f32(out: ScreenedPoissonOutput<f32>) -> Self {
        Self {
            stats: out.stats,
            forces: ScreenedPoissonForces::F32(out.force_x, out.force_y),
        }
    }
}

#[pymethods]
impl PyScreenedPoissonOutput {
    #[getter]
    fn stats(&self) -> PyScreenedPoissonStats {
        PyScreenedPoissonStats::from_inner(self.stats.clone())
    }

    #[getter]
    fn force_x(&self, py: Python<'_>) -> Py<PyAny> {
        match &self.forces {
            ScreenedPoissonForces::F64(fx, _) => PyArray1::from_slice(py, fx).into_any().unbind(),
            ScreenedPoissonForces::F32(fx, _) => PyArray1::from_slice(py, fx).into_any().unbind(),
        }
    }

    #[getter]
    fn force_y(&self, py: Python<'_>) -> Py<PyAny> {
        match &self.forces {
            ScreenedPoissonForces::F64(_, fy) => PyArray1::from_slice(py, fy).into_any().unbind(),
            ScreenedPoissonForces::F32(_, fy) => PyArray1::from_slice(py, fy).into_any().unbind(),
        }
    }

    #[getter]
    fn energy(&self) -> f64 {
        self.stats.energy
    }

    #[getter]
    fn max_density(&self) -> f64 {
        self.stats.max_density
    }

    #[getter]
    fn excess_fraction(&self) -> f64 {
        self.stats.excess_fraction
    }

    #[getter]
    fn excess_mass(&self) -> f64 {
        self.stats.excess_mass
    }

    #[getter]
    fn sigma_used(&self) -> f64 {
        self.stats.sigma_used
    }

    #[getter]
    fn kappa_used(&self) -> f64 {
        self.stats.kappa_used
    }

    #[getter]
    fn rho_max_used(&self) -> f64 {
        self.stats.rho_max_used
    }

    #[getter]
    fn grid_dim(&self) -> usize {
        self.stats.grid_dim
    }

    #[getter]
    fn pad_dim(&self) -> usize {
        self.stats.pad_dim
    }

    #[getter]
    fn fft_2d_calls(&self) -> u64 {
        self.stats.fft_2d_calls
    }

    #[getter]
    fn early_exit_no_excess(&self) -> bool {
        self.stats.early_exit_no_excess
    }

    #[pyo3(signature = (), text_signature = "($self)")]
    fn to_dict(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let dict = PyDict::new(py);
        PyScreenedPoissonStats::from_inner(self.stats.clone()).fill_dict(&dict)?;
        dict.set_item("force_x", self.force_x(py))?;
        dict.set_item("force_y", self.force_y(py))?;
        Ok(dict.unbind())
    }

    #[pyo3(signature = (), text_signature = "($self)")]
    fn keys(&self) -> Vec<&'static str> {
        let mut keys = PyScreenedPoissonStats::from_inner(self.stats.clone()).keys();
        keys.push("force_x");
        keys.push("force_y");
        keys
    }

    fn __contains__(&self, key: &str) -> bool {
        PyScreenedPoissonStats::from_inner(self.stats.clone()).__contains__(key)
            || matches!(key, "force_x" | "force_y")
    }

    fn __getitem__(&self, py: Python<'_>, key: &str) -> PyResult<Py<PyAny>> {
        match key {
            "force_x" => Ok(self.force_x(py)),
            "force_y" => Ok(self.force_y(py)),
            _ => PyScreenedPoissonStats::from_inner(self.stats.clone()).__getitem__(py, key),
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "ScreenedPoissonOutput(energy={:.6}, max_density={:.4}, excess_fraction={:.4})",
            self.stats.energy, self.stats.max_density, self.stats.excess_fraction
        )
    }
}
