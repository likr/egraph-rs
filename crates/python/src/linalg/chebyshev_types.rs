//! Frozen result class for `project_psd_spectral_filter`.

use pyo3::IntoPyObjectExt;
use pyo3::exceptions::PyKeyError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

const FILTER_PROJ_KEYS: [&str; 3] = ["terms", "delta_psd", "h_min"];

/// Frozen result returned by `project_psd_spectral_filter`.
#[pyclass(
    name = "SpectralFilterProjection",
    module = "egraph._core",
    frozen,
    from_py_object
)]
#[derive(Debug, Clone)]
pub struct PySpectralFilterProjection {
    terms: Vec<(f64, f64)>,
    delta_psd: f64,
    h_min: f64,
}

impl PySpectralFilterProjection {
    pub(super) fn new(terms: Vec<(f64, f64)>, delta_psd: f64, h_min: f64) -> Self {
        Self {
            terms,
            delta_psd,
            h_min,
        }
    }
}

#[pymethods]
impl PySpectralFilterProjection {
    /// Projected PSD spectral filter terms `[(weight, t), ...]`.
    #[getter]
    fn terms(&self) -> Vec<(f64, f64)> {
        self.terms.clone()
    }

    /// Applied PSD shift `delta_psd`.
    #[getter]
    fn delta_psd(&self) -> f64 {
        self.delta_psd
    }

    /// Minimum filter response across `[0, lambda_max]` prior to projection.
    #[getter]
    fn h_min(&self) -> f64 {
        self.h_min
    }

    /// Return projection fields as a Python dictionary.
    fn to_dict(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let dict = PyDict::new(py);
        dict.set_item("terms", self.terms.clone())?;
        dict.set_item("delta_psd", self.delta_psd)?;
        dict.set_item("h_min", self.h_min)?;
        Ok(dict.unbind())
    }

    /// Return ordered field keys for dictionary compatibility.
    fn keys(&self) -> Vec<&'static str> {
        FILTER_PROJ_KEYS.to_vec()
    }

    fn __getitem__(&self, py: Python<'_>, key: &str) -> PyResult<Py<PyAny>> {
        match key {
            "terms" => Ok(self.terms.clone().into_py_any(py)?),
            "delta_psd" => Ok(self.delta_psd.into_py_any(py)?),
            "h_min" => Ok(self.h_min.into_py_any(py)?),
            _ => Err(PyKeyError::new_err(key.to_owned())),
        }
    }

    fn __contains__(&self, key: &str) -> bool {
        FILTER_PROJ_KEYS.contains(&key)
    }

    fn __repr__(&self) -> String {
        format!(
            "SpectralFilterProjection(num_terms={}, delta_psd={:.6e}, h_min={:.6e})",
            self.terms.len(),
            self.delta_psd,
            self.h_min,
        )
    }
}

pub(super) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PySpectralFilterProjection>()?;
    Ok(())
}
