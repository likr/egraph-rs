//! `RdMds` and `KorenSpectral` `LayoutGenerator` class bindings.

use egraph::core::{CsrGraph, Layout2d, Layout3d, LayoutGenerator as _, LayoutNd};
use egraph::linalg::{KorenSpectral, KorenSpectralConfig, RdMds, RdMdsConfig};
use pyo3::prelude::*;

use crate::csr_graph::PyCsrGraph;
use crate::errors::{core_error_to_py, dimension_mismatch_err, unsupported_operation_err};
use crate::ffi_convert::{ScalarDtype, StrictF64, StrictU64, StrictUsize};
use crate::layout_buffers::PyLayout2d;
use crate::layout_nd::PyLayoutNd;
use crate::layout3d::PyLayout3d;
use crate::solvers::layout_dispatch::SolverTargetKind;

fn resolve_dims(
    target: SolverTargetKind,
    dims: Option<StrictUsize>,
    cls_name: &str,
) -> PyResult<Option<usize>> {
    match target {
        SolverTargetKind::Layout2d => {
            if let Some(d) = dims
                && d.0 != 2
            {
                return Err(dimension_mismatch_err(format!(
                    "{cls_name} with target='layout2d' requires dims=2 or None, got {}",
                    d.0
                )));
            }
            Ok(Some(2))
        }
        SolverTargetKind::Layout3d => {
            if let Some(d) = dims
                && d.0 != 3
            {
                return Err(dimension_mismatch_err(format!(
                    "{cls_name} with target='layout3d' requires dims=3 or None, got {}",
                    d.0
                )));
            }
            Ok(Some(3))
        }
        SolverTargetKind::LayoutNd => Ok(Some(dims.map_or(2, |d| d.0))),
        SolverTargetKind::Manifold2d => Err(unsupported_operation_err(format!(
            "{cls_name} does not support target='manifold2d'; expected 'layout2d', 'layout3d', or 'layout_nd'"
        ))),
    }
}

/// Cooperative Resistance-Distance Multidimensional Scaling (`RdMds`) layout generator.
#[pyclass(name = "RdMds", module = "egraph._core")]
pub struct PyRdMds {
    graph: CsrGraph<f64>,
    target: SolverTargetKind,
    dtype: ScalarDtype,
    config: RdMdsConfig,
}

#[pymethods]
impl PyRdMds {
    /// Construct an `RdMds` layout generator over `graph`.
    #[new]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(
        signature = (
            graph,
            *,
            target = "layout2d",
            dims = None,
            eig_eps = StrictF64(1e-10),
            tol = StrictF64(1e-6),
            max_iters = StrictUsize(1000),
            seed = StrictU64(42),
            degree_normalize = true,
            target_std = None,
            dtype = ScalarDtype::F64
        ),
        text_signature = "(graph, *, target='layout2d', dims=None, eig_eps=1e-10, tol=1e-6, max_iters=1000, seed=42, degree_normalize=True, target_std=None, dtype='float64')"
    )]
    fn new(
        graph: &PyCsrGraph,
        target: &str,
        dims: Option<StrictUsize>,
        eig_eps: StrictF64,
        tol: StrictF64,
        max_iters: StrictUsize,
        seed: StrictU64,
        degree_normalize: bool,
        target_std: Option<StrictF64>,
        dtype: ScalarDtype,
    ) -> PyResult<Self> {
        let target_kind = SolverTargetKind::from_str(target)?;
        let resolved_dims = resolve_dims(target_kind, dims, "RdMds")?;
        let config = RdMdsConfig::new()
            .with_dims(resolved_dims)
            .with_eig_eps(eig_eps.0)
            .with_tol(tol.0)
            .with_max_iters(max_iters.0)
            .with_seed(seed.0)
            .with_degree_normalize(degree_normalize)
            .with_target_std(target_std.map(|v| v.0));
        Ok(Self {
            graph: graph.inner().into_owned(),
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
        let g = &self.graph;
        match (self.target, self.dtype) {
            (SolverTargetKind::Layout2d, ScalarDtype::F64) => {
                let layout = py
                    .detach(|| RdMds::<Layout2d<f64>, _>::from_config(g, cfg).generate())
                    .map_err(core_error_to_py)?;
                Py::new(py, PyLayout2d::from_f64_layout(&layout))
                    .map(|o| o.into_bound(py).into_any())
            }
            (SolverTargetKind::Layout2d, ScalarDtype::F32) => {
                let layout = py
                    .detach(|| RdMds::<Layout2d<f32>, _>::from_config(g, cfg).generate())
                    .map_err(core_error_to_py)?;
                Py::new(py, PyLayout2d::from_f32_layout(&layout))
                    .map(|o| o.into_bound(py).into_any())
            }
            (SolverTargetKind::Layout3d, ScalarDtype::F64) => {
                let layout = py
                    .detach(|| RdMds::<Layout3d<f64>, _>::from_config(g, cfg).generate())
                    .map_err(core_error_to_py)?;
                Py::new(py, PyLayout3d::from_f64_layout(&layout))
                    .map(|o| o.into_bound(py).into_any())
            }
            (SolverTargetKind::Layout3d, ScalarDtype::F32) => {
                let layout = py
                    .detach(|| RdMds::<Layout3d<f32>, _>::from_config(g, cfg).generate())
                    .map_err(core_error_to_py)?;
                Py::new(py, PyLayout3d::from_f32_layout(&layout))
                    .map(|o| o.into_bound(py).into_any())
            }
            (SolverTargetKind::LayoutNd, ScalarDtype::F64) => {
                let layout = py
                    .detach(|| RdMds::<LayoutNd<f64>, _>::from_config(g, cfg).generate())
                    .map_err(core_error_to_py)?;
                Py::new(py, PyLayoutNd::from_f64_nd(layout)).map(|o| o.into_bound(py).into_any())
            }
            (SolverTargetKind::LayoutNd, ScalarDtype::F32) => {
                let layout = py
                    .detach(|| RdMds::<LayoutNd<f32>, _>::from_config(g, cfg).generate())
                    .map_err(core_error_to_py)?;
                Py::new(py, PyLayoutNd::from_f32_nd(layout)).map(|o| o.into_bound(py).into_any())
            }
            (SolverTargetKind::Manifold2d, _) => Err(unsupported_operation_err(
                "RdMds does not support target='manifold2d'",
            )),
        }
    }
}

/// Cooperative Koren degree-normalized spectral layout generator (`KorenSpectral`).
#[pyclass(name = "KorenSpectral", module = "egraph._core")]
pub struct PyKorenSpectral {
    graph: CsrGraph<f64>,
    target: SolverTargetKind,
    dtype: ScalarDtype,
    config: KorenSpectralConfig,
}

#[pymethods]
impl PyKorenSpectral {
    /// Construct a `KorenSpectral` layout generator over `graph`.
    #[new]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(
        signature = (
            graph,
            *,
            target = "layout2d",
            dims = None,
            target_std = Some(StrictF64(1.0)),
            tol = StrictF64(1e-4),
            max_iters = StrictUsize(300),
            seed = StrictU64(42),
            dtype = ScalarDtype::F64
        ),
        text_signature = "(graph, *, target='layout2d', dims=None, target_std=1.0, tol=1e-4, max_iters=300, seed=42, dtype='float64')"
    )]
    fn new(
        graph: &PyCsrGraph,
        target: &str,
        dims: Option<StrictUsize>,
        target_std: Option<StrictF64>,
        tol: StrictF64,
        max_iters: StrictUsize,
        seed: StrictU64,
        dtype: ScalarDtype,
    ) -> PyResult<Self> {
        let target_kind = SolverTargetKind::from_str(target)?;
        let resolved_dims = resolve_dims(target_kind, dims, "KorenSpectral")?;
        let config = KorenSpectralConfig::new()
            .with_dims(resolved_dims)
            .with_target_std(target_std.map(|v| v.0))
            .with_tol(tol.0)
            .with_max_iters(max_iters.0)
            .with_seed(seed.0);
        Ok(Self {
            graph: graph.inner().into_owned(),
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
        let g = &self.graph;
        match (self.target, self.dtype) {
            (SolverTargetKind::Layout2d, ScalarDtype::F64) => {
                let layout = py
                    .detach(|| KorenSpectral::<Layout2d<f64>, _>::from_config(g, cfg).generate())
                    .map_err(core_error_to_py)?;
                Py::new(py, PyLayout2d::from_f64_layout(&layout))
                    .map(|o| o.into_bound(py).into_any())
            }
            (SolverTargetKind::Layout2d, ScalarDtype::F32) => {
                let layout = py
                    .detach(|| KorenSpectral::<Layout2d<f32>, _>::from_config(g, cfg).generate())
                    .map_err(core_error_to_py)?;
                Py::new(py, PyLayout2d::from_f32_layout(&layout))
                    .map(|o| o.into_bound(py).into_any())
            }
            (SolverTargetKind::Layout3d, ScalarDtype::F64) => {
                let layout = py
                    .detach(|| KorenSpectral::<Layout3d<f64>, _>::from_config(g, cfg).generate())
                    .map_err(core_error_to_py)?;
                Py::new(py, PyLayout3d::from_f64_layout(&layout))
                    .map(|o| o.into_bound(py).into_any())
            }
            (SolverTargetKind::Layout3d, ScalarDtype::F32) => {
                let layout = py
                    .detach(|| KorenSpectral::<Layout3d<f32>, _>::from_config(g, cfg).generate())
                    .map_err(core_error_to_py)?;
                Py::new(py, PyLayout3d::from_f32_layout(&layout))
                    .map(|o| o.into_bound(py).into_any())
            }
            (SolverTargetKind::LayoutNd, ScalarDtype::F64) => {
                let layout = py
                    .detach(|| KorenSpectral::<LayoutNd<f64>, _>::from_config(g, cfg).generate())
                    .map_err(core_error_to_py)?;
                Py::new(py, PyLayoutNd::from_f64_nd(layout)).map(|o| o.into_bound(py).into_any())
            }
            (SolverTargetKind::LayoutNd, ScalarDtype::F32) => {
                let layout = py
                    .detach(|| KorenSpectral::<LayoutNd<f32>, _>::from_config(g, cfg).generate())
                    .map_err(core_error_to_py)?;
                Py::new(py, PyLayoutNd::from_f32_nd(layout)).map(|o| o.into_bound(py).into_any())
            }
            (SolverTargetKind::Manifold2d, _) => Err(unsupported_operation_err(
                "KorenSpectral does not support target='manifold2d'",
            )),
        }
    }
}

pub(super) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyRdMds>()?;
    m.add_class::<PyKorenSpectral>()?;
    Ok(())
}
