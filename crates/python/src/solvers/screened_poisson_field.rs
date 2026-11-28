//! Screened Poisson unilateral upper-bound 2D density field (`ScreenedPoissonField2d`) bindings.

use egraph::{core as egraph_core, spatial as egraph_spatial};
use egraph_core::CoreError;
use egraph_spatial::{
    DensityFieldKernel, ScreenedPoissonConfig, ScreenedPoissonField2d, ScreenedPoissonOutput,
};
use numpy::PyArray2;
use numpy::ndarray::Array2;
use pyo3::prelude::*;

use super::screened_poisson_stats::PyScreenedPoissonOutput;
use crate::errors::{dimension_mismatch_err, invalid_input_err};
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
        _ => "screened_poisson",
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
    let kernel_enum = parse_density_field_kernel(kernel).map_err(invalid_input_err)?;
    let cfg = ScreenedPoissonConfig::default()
        .with_kernel(kernel_enum)
        .with_grid_size(grid_size.0)
        .with_padding_factor(padding_factor.0)
        .with_margin_ratio(margin_ratio.0)
        .with_sigma_ratio(sigma_ratio.0)
        .with_sigma(sigma.map(|v| v.0))
        .with_screening_length_ratio(screening_length_ratio.0)
        .with_kappa(kappa.map(|v| v.0))
        .with_rho_max_factor(rho_max_factor.0)
        .with_rho_max(rho_max.map(|v| v.0))
        .with_exact_adjoint(exact_adjoint);
    cfg.validate().map_err(core_error_to_py)?;
    Ok(cfg)
}

enum ScreenedPoissonFieldInner {
    F64(ScreenedPoissonField2d<f64>),
    F32(ScreenedPoissonField2d<f32>),
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
            grid_size = StrictUsize(512),
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
        text_signature = "(*, kernel='screened_poisson', grid_size=512, padding_factor=2, margin_ratio=0.1, sigma_ratio=0.015, sigma=None, screening_length_ratio=0.2, kappa=None, rho_max_factor=4.0, rho_max=None, exact_adjoint=True, dtype='float64')"
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
    fn evaluate(
        &mut self,
        py: Python<'_>,
        layout: &PyLayout2d,
    ) -> PyResult<PyScreenedPoissonOutput> {
        match (&mut self.inner, layout.inner()) {
            (ScreenedPoissonFieldInner::F64(field), Layout2dInner::F64(buf)) => {
                let soa = buf.to_layout2d();
                py.detach(|| {
                    let mut out = ScreenedPoissonOutput::new();
                    field.evaluate(&soa, &mut out)?;
                    Ok::<_, CoreError>(PyScreenedPoissonOutput::from_f64(out))
                })
                .map_err(core_error_to_py)
            }
            (ScreenedPoissonFieldInner::F32(field), Layout2dInner::F32(buf)) => {
                let soa = buf.to_layout2d();
                py.detach(|| {
                    let mut out = ScreenedPoissonOutput::new();
                    field.evaluate(&soa, &mut out)?;
                    Ok::<_, CoreError>(PyScreenedPoissonOutput::from_f32(out))
                })
                .map_err(core_error_to_py)
            }
            _ => Err(dimension_mismatch_err(format!(
                "layout dtype '{}' does not match ScreenedPoissonField2d dtype '{}'",
                layout.dtype(),
                self.dtype()
            ))),
        }
    }

    /// Evaluate unilateral density energy and descent forces inside explicit bounds `(min_x, min_y, max_x, max_y)`.
    #[pyo3(
        signature = (layout, min_x, min_y, max_x, max_y),
        text_signature = "($self, layout, min_x, min_y, max_x, max_y)"
    )]
    fn evaluate_in_bounds(
        &mut self,
        py: Python<'_>,
        layout: &PyLayout2d,
        min_x: StrictF64,
        min_y: StrictF64,
        max_x: StrictF64,
        max_y: StrictF64,
    ) -> PyResult<PyScreenedPoissonOutput> {
        let b_arr = [min_x.0, min_y.0, max_x.0, max_y.0];
        match (&mut self.inner, layout.inner()) {
            (ScreenedPoissonFieldInner::F64(field), Layout2dInner::F64(buf)) => {
                let soa = buf.to_layout2d();
                py.detach(|| {
                    let mut out = ScreenedPoissonOutput::new();
                    field.evaluate_in_bounds(&soa, b_arr, &mut out)?;
                    Ok::<_, CoreError>(PyScreenedPoissonOutput::from_f64(out))
                })
                .map_err(core_error_to_py)
            }
            (ScreenedPoissonFieldInner::F32(field), Layout2dInner::F32(buf)) => {
                let soa = buf.to_layout2d();
                py.detach(|| {
                    let mut out = ScreenedPoissonOutput::new();
                    field.evaluate_in_bounds(&soa, b_arr, &mut out)?;
                    Ok::<_, CoreError>(PyScreenedPoissonOutput::from_f32(out))
                })
                .map_err(core_error_to_py)
            }
            _ => Err(dimension_mismatch_err(format!(
                "layout dtype '{}' does not match ScreenedPoissonField2d dtype '{}'",
                layout.dtype(),
                self.dtype()
            ))),
        }
    }

    /// Return the `(G, G)` empirical density grid from the most recent non-empty evaluation, or `None`.
    #[pyo3(text_signature = "($self)")]
    fn density_grid<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyAny>>> {
        match &self.inner {
            ScreenedPoissonFieldInner::F64(f) => match f.density_grid() {
                Some(grid) => {
                    let dim = f.density_grid_dim();
                    let arr = Array2::from_shape_vec((dim, dim), grid.to_vec())
                        .map_err(|e| invalid_input_err(e.to_string()))?;
                    Ok(Some(PyArray2::from_owned_array(py, arr).into_any()))
                }
                None => Ok(None),
            },
            ScreenedPoissonFieldInner::F32(f) => match f.density_grid() {
                Some(grid) => {
                    let dim = f.density_grid_dim();
                    let arr = Array2::from_shape_vec((dim, dim), grid.to_vec())
                        .map_err(|e| invalid_input_err(e.to_string()))?;
                    Ok(Some(PyArray2::from_owned_array(py, arr).into_any()))
                }
                None => Ok(None),
            },
        }
    }

    /// Interior grid dimension `G` of the populated `density_grid` (`0` before evaluation).
    #[getter]
    fn density_grid_dim(&self) -> usize {
        match &self.inner {
            ScreenedPoissonFieldInner::F64(f) => f.density_grid_dim(),
            ScreenedPoissonFieldInner::F32(f) => f.density_grid_dim(),
        }
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
