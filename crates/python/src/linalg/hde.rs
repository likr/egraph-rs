//! High-Dimensional Embedding (`HDE`) linear-algebra projection bindings.

use egraph::core::{CsrGraph, Layout2d, Layout3d, LayoutGenerator as _, LayoutNd};
use egraph::linalg::{Hde, HdeConfig};
use pyo3::prelude::*;

use crate::csr_graph::PyCsrGraph;
use crate::errors::{
    core_error_to_py, dimension_mismatch_err, invalid_input_err, unsupported_operation_err,
};
use crate::ffi_convert::{ScalarDtype, StrictF64, StrictU64, StrictUsize, extract_u32_1d};
use crate::layout_buffers::PyLayout2d;
use crate::layout_nd::PyLayoutNd;
use crate::layout3d::PyLayout3d;
use crate::solvers::layout_dispatch::SolverTargetKind;

/// Cooperative Harel-Koren High-Dimensional Embedding (`Hde`) layout generator.
#[pyclass(name = "Hde", module = "egraph._core")]
pub struct PyHde {
    graph: CsrGraph<f64>,
    pivots: Option<Vec<u32>>,
    target: SolverTargetKind,
    dtype: ScalarDtype,
    config: HdeConfig,
}

#[pymethods]
impl PyHde {
    /// Construct an `Hde` layout generator over `graph`.
    #[new]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(
        signature = (
            graph,
            *,
            target = "layout2d",
            dimensions = StrictUsize(50),
            output_dims = None,
            seed = StrictU64(42),
            tol = StrictF64(1e-3),
            max_sweeps = StrictUsize(100),
            normalize = true,
            pivots = None,
            dtype = ScalarDtype::F64
        ),
        text_signature = "(graph, *, target='layout2d', dimensions=50, output_dims=None, seed=42, tol=1e-3, max_sweeps=100, normalize=True, pivots=None, dtype='float64')"
    )]
    fn new(
        graph: &PyCsrGraph,
        target: &str,
        dimensions: StrictUsize,
        output_dims: Option<StrictUsize>,
        seed: StrictU64,
        tol: StrictF64,
        max_sweeps: StrictUsize,
        normalize: bool,
        pivots: Option<&Bound<'_, PyAny>>,
        dtype: ScalarDtype,
    ) -> PyResult<Self> {
        let target_kind = SolverTargetKind::from_str(target)?;
        let out_dims = match target_kind {
            SolverTargetKind::Layout2d => {
                if let Some(d) = output_dims
                    && d.0 != 2
                {
                    return Err(dimension_mismatch_err(format!(
                        "Hde with target='layout2d' requires output_dims=2 or None, got {}",
                        d.0
                    )));
                }
                Some(2)
            }
            SolverTargetKind::Layout3d => {
                if let Some(d) = output_dims
                    && d.0 != 3
                {
                    return Err(dimension_mismatch_err(format!(
                        "Hde with target='layout3d' requires output_dims=3 or None, got {}",
                        d.0
                    )));
                }
                Some(3)
            }
            SolverTargetKind::LayoutNd => Some(output_dims.map_or(2, |d| d.0)),
            SolverTargetKind::Manifold2d => {
                return Err(unsupported_operation_err(
                    "Hde does not support target='manifold2d'; expected 'layout2d', 'layout3d', or 'layout_nd'",
                ));
            }
        };
        let pivots_vec = pivots
            .filter(|p| !p.is_none())
            .map(|p| extract_u32_1d(p, "pivots"))
            .transpose()?;
        let config = HdeConfig::new()
            .with_dimensions(dimensions.0)
            .with_output_dims(out_dims)
            .with_seed(seed.0)
            .with_tol(tol.0)
            .with_max_sweeps(max_sweeps.0)
            .with_normalize(normalize);
        Ok(Self {
            graph: graph.inner().into_owned(),
            pivots: pivots_vec,
            target: target_kind,
            dtype,
            config,
        })
    }

    #[getter]
    fn target(&self) -> &'static str {
        self.target.as_str()
    }

    #[getter]
    fn dtype(&self) -> &'static str {
        self.dtype.as_str()
    }

    /// Generate the target layout (`Layout2d`, `Layout3d`, or `LayoutNd`).
    #[pyo3(signature = (), text_signature = "($self)")]
    fn generate<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let cfg = self.config.clone();
        let piv = self.pivots.as_deref();
        let g = &self.graph;
        macro_rules! run_hde {
            ($layout_ty:ty, $wrap:expr) => {{
                let mut generator = Hde::<$layout_ty, _>::from_config(g, cfg);
                if let Some(p) = piv {
                    generator = generator.with_pivots(p);
                }
                let layout = py
                    .detach(|| generator.generate())
                    .map_err(core_error_to_py)?;
                $wrap(layout)
            }};
        }
        match (self.target, self.dtype) {
            (SolverTargetKind::Layout2d, ScalarDtype::F64) => run_hde!(Layout2d<f64>, |l| {
                Py::new(py, PyLayout2d::from_f64_layout(&l)).map(|o| o.into_bound(py).into_any())
            }),
            (SolverTargetKind::Layout2d, ScalarDtype::F32) => run_hde!(Layout2d<f32>, |l| {
                Py::new(py, PyLayout2d::from_f32_layout(&l)).map(|o| o.into_bound(py).into_any())
            }),
            (SolverTargetKind::Layout3d, ScalarDtype::F64) => run_hde!(Layout3d<f64>, |l| {
                Py::new(py, PyLayout3d::from_f64_layout(&l)).map(|o| o.into_bound(py).into_any())
            }),
            (SolverTargetKind::Layout3d, ScalarDtype::F32) => run_hde!(Layout3d<f32>, |l| {
                Py::new(py, PyLayout3d::from_f32_layout(&l)).map(|o| o.into_bound(py).into_any())
            }),
            (SolverTargetKind::LayoutNd, ScalarDtype::F64) => run_hde!(LayoutNd<f64>, |l| {
                Py::new(py, PyLayoutNd::from_f64_nd(l)).map(|o| o.into_bound(py).into_any())
            }),
            (SolverTargetKind::LayoutNd, ScalarDtype::F32) => run_hde!(LayoutNd<f32>, |l| {
                Py::new(py, PyLayoutNd::from_f32_nd(l)).map(|o| o.into_bound(py).into_any())
            }),
            (SolverTargetKind::Manifold2d, _) => Err(unsupported_operation_err(
                "Hde does not support target='manifold2d'",
            )),
        }
    }
}

/// Compute a 2D `Layout2d` via Harel-Koren High-Dimensional Embedding (`HDE`, 2004).
#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(
    signature = (
        graph,
        *,
        dimensions = StrictUsize(50),
        seed = StrictU64(42),
        tol = StrictF64(1e-3),
        max_sweeps = StrictUsize(100),
        normalize = true,
        pivots = None,
        dtype = ScalarDtype::F64
    ),
    text_signature = "(graph, *, dimensions=50, seed=42, tol=1e-3, max_sweeps=100, normalize=True, pivots=None, dtype='float64')"
)]
fn hde(
    py: Python<'_>,
    graph: &PyCsrGraph,
    dimensions: StrictUsize,
    seed: StrictU64,
    tol: StrictF64,
    max_sweeps: StrictUsize,
    normalize: bool,
    pivots: Option<&Bound<'_, PyAny>>,
    dtype: ScalarDtype,
) -> PyResult<PyLayout2d> {
    let pivots_vec = pivots
        .filter(|p| !p.is_none())
        .map(|p| extract_u32_1d(p, "pivots"))
        .transpose()?;
    let cfg = HdeConfig::new()
        .with_dimensions(dimensions.0)
        .with_output_dims(Some(2))
        .with_seed(seed.0)
        .with_tol(tol.0)
        .with_max_sweeps(max_sweeps.0)
        .with_normalize(normalize);
    let g = graph.inner();
    py.detach(|| match (dtype, pivots_vec.as_deref()) {
        (ScalarDtype::F64, None) => Hde::<Layout2d<f64>, _>::from_config(&*g, cfg)
            .generate()
            .map(|l| PyLayout2d::from_f64_layout(&l)),
        (ScalarDtype::F32, None) => Hde::<Layout2d<f32>, _>::from_config(&*g, cfg)
            .generate()
            .map(|l| PyLayout2d::from_f32_layout(&l)),
        (ScalarDtype::F64, Some(p)) => Hde::<Layout2d<f64>, _>::from_config(&*g, cfg)
            .with_pivots(p)
            .generate()
            .map(|l| PyLayout2d::from_f64_layout(&l)),
        (ScalarDtype::F32, Some(p)) => Hde::<Layout2d<f32>, _>::from_config(&*g, cfg)
            .with_pivots(p)
            .generate()
            .map(|l| PyLayout2d::from_f32_layout(&l)),
    })
    .map_err(core_error_to_py)
}

/// Compute an `output_dims`-dimensional `LayoutNd` via Harel-Koren High-Dimensional Embedding (`HDE`, 2004).
#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(
    signature = (
        graph,
        *,
        dimensions = StrictUsize(50),
        output_dims = StrictUsize(2),
        seed = StrictU64(42),
        tol = StrictF64(1e-3),
        max_sweeps = StrictUsize(100),
        normalize = true,
        pivots = None,
        dtype = ScalarDtype::F64
    ),
    text_signature = "(graph, *, dimensions=50, output_dims=2, seed=42, tol=1e-3, max_sweeps=100, normalize=True, pivots=None, dtype='float64')"
)]
fn hde_nd(
    py: Python<'_>,
    graph: &PyCsrGraph,
    dimensions: StrictUsize,
    output_dims: StrictUsize,
    seed: StrictU64,
    tol: StrictF64,
    max_sweeps: StrictUsize,
    normalize: bool,
    pivots: Option<&Bound<'_, PyAny>>,
    dtype: ScalarDtype,
) -> PyResult<PyLayoutNd> {
    if output_dims.0 == 0 {
        return Err(invalid_input_err("output_dims must be >= 1"));
    }
    let pivots_vec = pivots
        .filter(|p| !p.is_none())
        .map(|p| extract_u32_1d(p, "pivots"))
        .transpose()?;
    let cfg = HdeConfig::new()
        .with_dimensions(dimensions.0)
        .with_output_dims(Some(output_dims.0))
        .with_seed(seed.0)
        .with_tol(tol.0)
        .with_max_sweeps(max_sweeps.0)
        .with_normalize(normalize);
    let g = graph.inner();
    py.detach(|| match (dtype, pivots_vec.as_deref()) {
        (ScalarDtype::F64, None) => Hde::<LayoutNd<f64>, _>::from_config(&*g, cfg)
            .generate()
            .map(PyLayoutNd::from_f64_nd),
        (ScalarDtype::F32, None) => Hde::<LayoutNd<f32>, _>::from_config(&*g, cfg)
            .generate()
            .map(PyLayoutNd::from_f32_nd),
        (ScalarDtype::F64, Some(p)) => Hde::<LayoutNd<f64>, _>::from_config(&*g, cfg)
            .with_pivots(p)
            .generate()
            .map(PyLayoutNd::from_f64_nd),
        (ScalarDtype::F32, Some(p)) => Hde::<LayoutNd<f32>, _>::from_config(&*g, cfg)
            .with_pivots(p)
            .generate()
            .map(PyLayoutNd::from_f32_nd),
    })
    .map_err(core_error_to_py)
}

pub(super) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyHde>()?;
    m.add_function(wrap_pyfunction!(hde, m)?)?;
    m.add_function(wrap_pyfunction!(hde_nd, m)?)?;
    Ok(())
}
