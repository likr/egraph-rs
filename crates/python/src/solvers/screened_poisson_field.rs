//! Screened Poisson unilateral upper-bound 2D density field (`ScreenedPoissonField2d`) bindings.

use egraph::{core as egraph_core, spatial as egraph_spatial};
use egraph_core::CoreError;
use egraph_spatial::{
    DensityFieldKernel, ScreenedPoissonConfig, ScreenedPoissonField2d, ScreenedPoissonOutput,
    ScreenedPoissonStats,
};
use numpy::PyArray1;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::array_extract::extract_f64_1d;
use crate::ffi_convert::{
    ScalarDtype, StrictF64, StrictUsize, core_error_to_py, normalize_enum_key,
};
use crate::layout_buffers::{Layout2dInner, PyLayout2d};

pub(super) fn parse_density_field_kernel(raw: &str) -> Result<DensityFieldKernel, String> {
    match normalize_enum_key(raw).as_str() {
        "screenedpoisson" | "sp" => Ok(DensityFieldKernel::ScreenedPoisson),
        "singlegaussian" | "gaussian" | "sg" => Ok(DensityFieldKernel::SingleGaussian),
        _ => Err(format!(
            "unknown density field kernel {raw:?}; expected 'screened_poisson' or 'single_gaussian'"
        )),
    }
}

fn density_field_kernel_str(kernel: DensityFieldKernel) -> &'static str {
    match kernel {
        DensityFieldKernel::ScreenedPoisson => "screened_poisson",
        DensityFieldKernel::SingleGaussian => "single_gaussian",
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn build_screened_poisson_config(
    kernel: &str,
    grid_size: StrictUsize,
    padding_factor: StrictUsize,
    margin_ratio: StrictF64,
    sigma_ratio: StrictF64,
    sigma: Option<StrictF64>,
    screening_length_ratio: StrictF64,
    kappa: Option<StrictF64>,
    rho_max_factor: StrictF64,
    rho_max: Option<StrictF64>,
    exact_adjoint: bool,
) -> PyResult<ScreenedPoissonConfig> {
    let kernel_enum = parse_density_field_kernel(kernel).map_err(PyValueError::new_err)?;
    let cfg = ScreenedPoissonConfig {
        kernel: kernel_enum,
        grid_size: grid_size.0,
        padding_factor: padding_factor.0,
        margin_ratio: margin_ratio.0,
        sigma_ratio: sigma_ratio.0,
        sigma: sigma.map(|v| v.0),
        screening_length_ratio: screening_length_ratio.0,
        kappa: kappa.map(|v| v.0),
        rho_max_factor: rho_max_factor.0,
        rho_max: rho_max.map(|v| v.0),
        exact_adjoint,
    };
    cfg.validate().map_err(core_error_to_py)?;
    Ok(cfg)
}

pub(super) fn screened_poisson_stats_to_dict(
    py: Python<'_>,
    st: &ScreenedPoissonStats,
) -> PyResult<Py<PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item("energy", st.energy)?;
    dict.set_item("max_density", st.max_density)?;
    dict.set_item("excess_fraction", st.excess_fraction)?;
    dict.set_item("excess_mass", st.excess_mass)?;
    dict.set_item("sigma_used", st.sigma_used)?;
    dict.set_item("kappa_used", st.kappa_used)?;
    dict.set_item("rho_max_used", st.rho_max_used)?;
    dict.set_item("grid_dim", st.grid_dim)?;
    dict.set_item("pad_dim", st.pad_dim)?;
    dict.set_item("fft_2d_calls", st.fft_2d_calls)?;
    dict.set_item("early_exit_no_excess", st.early_exit_no_excess)?;
    Ok(dict.unbind())
}

enum ScreenedPoissonFieldInner {
    F64(ScreenedPoissonField2d<f64>),
    F32(ScreenedPoissonField2d<f32>),
}

enum ScreenedPoissonOutInner {
    F64(ScreenedPoissonOutput<f64>),
    F32(ScreenedPoissonOutput<f32>),
}

fn screened_poisson_output_to_dict(
    py: Python<'_>,
    out: ScreenedPoissonOutInner,
) -> PyResult<Py<PyDict>> {
    match out {
        ScreenedPoissonOutInner::F64(o) => {
            let dict = screened_poisson_stats_to_dict(py, &o.stats)?;
            let bound = dict.bind(py);
            bound.set_item("force_x", PyArray1::from_vec(py, o.force_x))?;
            bound.set_item("force_y", PyArray1::from_vec(py, o.force_y))?;
            Ok(dict)
        }
        ScreenedPoissonOutInner::F32(o) => {
            let dict = screened_poisson_stats_to_dict(py, &o.stats)?;
            let bound = dict.bind(py);
            bound.set_item("force_x", PyArray1::from_vec(py, o.force_x))?;
            bound.set_item("force_y", PyArray1::from_vec(py, o.force_y))?;
            Ok(dict)
        }
    }
}

/// Reusable `f32` R2C-FFT Screened Poisson unilateral upper-bound 2D density field evaluator.
///
/// Releases the Python GIL during evaluation.
#[pyclass(name = "ScreenedPoissonField2d", module = "egraph._core")]
pub struct PyScreenedPoissonField2d {
    inner: ScreenedPoissonFieldInner,
}

#[pymethods]
impl PyScreenedPoissonField2d {
    /// Create a `ScreenedPoissonField2d` evaluator workspace.
    #[new]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(
        signature = (
            *,
            kernel = "screened_poisson",
            grid_size = StrictUsize(2048),
            padding_factor = StrictUsize(2),
            margin_ratio = StrictF64(0.1),
            sigma_ratio = StrictF64(0.015),
            sigma = None,
            screening_length_ratio = StrictF64(0.20),
            kappa = None,
            rho_max_factor = StrictF64(4.0),
            rho_max = None,
            exact_adjoint = true,
            dtype = ScalarDtype::F64
        ),
        text_signature = "(*, kernel='screened_poisson', grid_size=2048, padding_factor=2, margin_ratio=0.1, sigma_ratio=0.015, sigma=None, screening_length_ratio=0.2, kappa=None, rho_max_factor=4.0, rho_max=None, exact_adjoint=True, dtype='float64')"
    )]
    fn new(
        py: Python<'_>,
        kernel: &str,
        grid_size: StrictUsize,
        padding_factor: StrictUsize,
        margin_ratio: StrictF64,
        sigma_ratio: StrictF64,
        sigma: Option<StrictF64>,
        screening_length_ratio: StrictF64,
        kappa: Option<StrictF64>,
        rho_max_factor: StrictF64,
        rho_max: Option<StrictF64>,
        exact_adjoint: bool,
        dtype: ScalarDtype,
    ) -> PyResult<Self> {
        let cfg = build_screened_poisson_config(
            kernel,
            grid_size,
            padding_factor,
            margin_ratio,
            sigma_ratio,
            sigma,
            screening_length_ratio,
            kappa,
            rho_max_factor,
            rho_max,
            exact_adjoint,
        )?;
        let inner = py
            .detach(|| match dtype {
                ScalarDtype::F64 => {
                    ScreenedPoissonField2d::<f64>::new(cfg).map(ScreenedPoissonFieldInner::F64)
                }
                ScalarDtype::F32 => {
                    ScreenedPoissonField2d::<f32>::new(cfg).map(ScreenedPoissonFieldInner::F32)
                }
            })
            .map_err(core_error_to_py)?;
        Ok(Self { inner })
    }

    /// Evaluate unilateral density energy and descent forces over `layout`'s adaptive bounding box.
    #[pyo3(signature = (layout), text_signature = "($self, layout)")]
    fn evaluate(&mut self, py: Python<'_>, layout: &PyLayout2d) -> PyResult<Py<PyDict>> {
        let out = match (&mut self.inner, layout.inner()) {
            (ScreenedPoissonFieldInner::F64(field), Layout2dInner::F64(buf)) => {
                let soa = buf.to_layout2d();
                py.detach(|| {
                    let mut out = ScreenedPoissonOutput::new();
                    field.evaluate(&soa, &mut out)?;
                    Ok::<_, CoreError>(ScreenedPoissonOutInner::F64(out))
                })
                .map_err(core_error_to_py)?
            }
            (ScreenedPoissonFieldInner::F32(field), Layout2dInner::F32(buf)) => {
                let soa = buf.to_layout2d();
                py.detach(|| {
                    let mut out = ScreenedPoissonOutput::new();
                    field.evaluate(&soa, &mut out)?;
                    Ok::<_, CoreError>(ScreenedPoissonOutInner::F32(out))
                })
                .map_err(core_error_to_py)?
            }
            _ => {
                return Err(PyValueError::new_err(format!(
                    "layout dtype '{}' does not match ScreenedPoissonField2d dtype '{}'",
                    layout.dtype(),
                    self.dtype()
                )));
            }
        };
        screened_poisson_output_to_dict(py, out)
    }

    /// Evaluate unilateral density energy and descent forces inside explicit `bounds = [min_x, max_x, min_y, max_y]`.
    #[pyo3(signature = (layout, bounds), text_signature = "($self, layout, bounds)")]
    fn evaluate_in_bounds(
        &mut self,
        py: Python<'_>,
        layout: &PyLayout2d,
        bounds: &Bound<'_, PyAny>,
    ) -> PyResult<Py<PyDict>> {
        let bvec = extract_f64_1d(bounds, "bounds")?;
        if bvec.len() != 4 {
            return Err(PyValueError::new_err(format!(
                "bounds must contain exactly 4 elements [min_x, max_x, min_y, max_y], got {}",
                bvec.len()
            )));
        }
        let b_arr = [bvec[0], bvec[1], bvec[2], bvec[3]];
        let out = match (&mut self.inner, layout.inner()) {
            (ScreenedPoissonFieldInner::F64(field), Layout2dInner::F64(buf)) => {
                let soa = buf.to_layout2d();
                py.detach(|| {
                    let mut out = ScreenedPoissonOutput::new();
                    field.evaluate_in_bounds(&soa, b_arr, &mut out)?;
                    Ok::<_, CoreError>(ScreenedPoissonOutInner::F64(out))
                })
                .map_err(core_error_to_py)?
            }
            (ScreenedPoissonFieldInner::F32(field), Layout2dInner::F32(buf)) => {
                let soa = buf.to_layout2d();
                py.detach(|| {
                    let mut out = ScreenedPoissonOutput::new();
                    field.evaluate_in_bounds(&soa, b_arr, &mut out)?;
                    Ok::<_, CoreError>(ScreenedPoissonOutInner::F32(out))
                })
                .map_err(core_error_to_py)?
            }
            _ => {
                return Err(PyValueError::new_err(format!(
                    "layout dtype '{}' does not match ScreenedPoissonField2d dtype '{}'",
                    layout.dtype(),
                    self.dtype()
                )));
            }
        };
        screened_poisson_output_to_dict(py, out)
    }

    /// Return total heap bytes allocated by the internal `f32` FFT grids and bitmasks.
    #[pyo3(text_signature = "($self)")]
    fn workspace_memory_bytes(&self) -> usize {
        match &self.inner {
            ScreenedPoissonFieldInner::F64(f) => f.workspace_memory_bytes(),
            ScreenedPoissonFieldInner::F32(f) => f.workspace_memory_bytes(),
        }
    }

    /// Active interior grid dimension $G$.
    #[getter]
    fn grid_size(&self) -> usize {
        match &self.inner {
            ScreenedPoissonFieldInner::F64(f) => f.config().grid_size,
            ScreenedPoissonFieldInner::F32(f) => f.config().grid_size,
        }
    }

    /// Zero-padding multiplier (`1` or `2`).
    #[getter]
    fn padding_factor(&self) -> usize {
        match &self.inner {
            ScreenedPoissonFieldInner::F64(f) => f.config().padding_factor,
            ScreenedPoissonFieldInner::F32(f) => f.config().padding_factor,
        }
    }

    /// Density kernel name (`"screened_poisson"` or `"single_gaussian"`).
    #[getter]
    fn kernel(&self) -> &'static str {
        let k = match &self.inner {
            ScreenedPoissonFieldInner::F64(f) => f.config().kernel,
            ScreenedPoissonFieldInner::F32(f) => f.config().kernel,
        };
        density_field_kernel_str(k)
    }

    /// Whether exact variational adjoint masking is enabled.
    #[getter]
    fn exact_adjoint(&self) -> bool {
        match &self.inner {
            ScreenedPoissonFieldInner::F64(f) => f.config().exact_adjoint,
            ScreenedPoissonFieldInner::F32(f) => f.config().exact_adjoint,
        }
    }

    /// Field coordinate precision (`"float64"` or `"float32"`).
    #[getter]
    fn dtype(&self) -> &'static str {
        match &self.inner {
            ScreenedPoissonFieldInner::F64(_) => "float64",
            ScreenedPoissonFieldInner::F32(_) => "float32",
        }
    }

    /// Return a concise string representation of the `ScreenedPoissonField2d`.
    fn __repr__(&self) -> String {
        format!(
            "ScreenedPoissonField2d(kernel='{}', grid_size={}, padding_factor={}, exact_adjoint={}, dtype='{}')",
            self.kernel(),
            self.grid_size(),
            self.padding_factor(),
            self.exact_adjoint(),
            self.dtype()
        )
    }
}
