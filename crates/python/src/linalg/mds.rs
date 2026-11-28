//! Classical, Pivot, and Representative Multidimensional Scaling (MDS) Python bindings.

use egraph::core::{CsrGraph, Layout2d, Layout3d, LayoutGenerator as _, LayoutNd};
use egraph::linalg::{ClassicalMds, ClassicalMdsConfig};
use pyo3::prelude::*;

use crate::array_extract::extract_f64_2d_raw;
use crate::csr_graph::PyCsrGraph;
use crate::errors::{
    core_error_to_py, dimension_mismatch_err, invalid_input_err, unsupported_operation_err,
};
use crate::ffi_convert::{ScalarDtype, StrictF64, StrictUsize};
use crate::layout_buffers::PyLayout2d;
use crate::layout_nd::PyLayoutNd;
use crate::layout3d::PyLayout3d;
use crate::solvers::layout_dispatch::SolverTargetKind;

pub(super) fn validate_euclidean_target_dims(
    target: SolverTargetKind,
    dims: Option<usize>,
    cls_name: &str,
) -> PyResult<Option<usize>> {
    match target {
        SolverTargetKind::Layout2d => {
            if let Some(d) = dims
                && d != 2
            {
                return Err(dimension_mismatch_err(format!(
                    "{cls_name} with target='layout2d' requires dims=2 or None, got {d}"
                )));
            }
            Ok(Some(2))
        }
        SolverTargetKind::Layout3d => {
            if let Some(d) = dims
                && d != 3
            {
                return Err(dimension_mismatch_err(format!(
                    "{cls_name} with target='layout3d' requires dims=3 or None, got {d}"
                )));
            }
            Ok(Some(3))
        }
        SolverTargetKind::LayoutNd => Ok(Some(dims.unwrap_or(2))),
        SolverTargetKind::Manifold2d => Err(unsupported_operation_err(format!(
            "{cls_name} does not support target='manifold2d'; expected 'layout2d', 'layout3d', or 'layout_nd'"
        ))),
    }
}

pub(super) enum MdsSource {
    Graph(CsrGraph<f64>),
    Distances {
        flat: Vec<f64>,
        num_nodes: usize,
        num_cols: usize,
    },
}

/// Cooperative Classical Multidimensional Scaling (`ClassicalMds`) layout generator.
#[pyclass(name = "ClassicalMds", module = "egraph._core")]
pub struct PyClassicalMds {
    source: MdsSource,
    target: SolverTargetKind,
    dtype: ScalarDtype,
    config: ClassicalMdsConfig,
}

#[pymethods]
impl PyClassicalMds {
    /// Construct a `ClassicalMds` generator over `graph`.
    #[new]
    #[pyo3(
        signature = (graph, *, target = "layout2d", dims = None, eig_eps = StrictF64(1e-12), tol = StrictF64(1e-10), max_sweeps = StrictUsize(64), dtype = ScalarDtype::F64),
        text_signature = "(graph, *, target='layout2d', dims=None, eig_eps=1e-12, tol=1e-10, max_sweeps=64, dtype='float64')"
    )]
    fn new(
        graph: &PyCsrGraph,
        target: &str,
        dims: Option<StrictUsize>,
        eig_eps: StrictF64,
        tol: StrictF64,
        max_sweeps: StrictUsize,
        dtype: ScalarDtype,
    ) -> PyResult<Self> {
        let target_kind = SolverTargetKind::from_str(target)?;
        let resolved_dims =
            validate_euclidean_target_dims(target_kind, dims.map(|d| d.0), "ClassicalMds")?;
        let config = ClassicalMdsConfig::new()
            .with_dims(resolved_dims)
            .with_eig_eps(eig_eps.0)
            .with_tol(tol.0)
            .with_max_sweeps(max_sweeps.0);
        Ok(Self {
            source: MdsSource::Graph(graph.inner().into_owned()),
            target: target_kind,
            dtype,
            config,
        })
    }

    /// Construct a `ClassicalMds` generator from an `(N, N)` distance matrix.
    #[staticmethod]
    #[pyo3(
        signature = (distances, *, target = "layout2d", dims = None, eig_eps = StrictF64(1e-12), tol = StrictF64(1e-10), max_sweeps = StrictUsize(64), dtype = ScalarDtype::F64),
        text_signature = "(distances, *, target='layout2d', dims=None, eig_eps=1e-12, tol=1e-10, max_sweeps=64, dtype='float64')"
    )]
    fn from_distances(
        distances: &Bound<'_, PyAny>,
        target: &str,
        dims: Option<StrictUsize>,
        eig_eps: StrictF64,
        tol: StrictF64,
        max_sweeps: StrictUsize,
        dtype: ScalarDtype,
    ) -> PyResult<Self> {
        let target_kind = SolverTargetKind::from_str(target)?;
        let resolved_dims =
            validate_euclidean_target_dims(target_kind, dims.map(|d| d.0), "ClassicalMds")?;
        let (flat, nrows, ncols) = extract_f64_2d_raw(distances, "distances")?;
        if nrows != ncols {
            return Err(dimension_mismatch_err(format!(
                "distances must be a square (N, N) matrix, got ({nrows}, {ncols})"
            )));
        }
        let config = ClassicalMdsConfig::new()
            .with_dims(resolved_dims)
            .with_eig_eps(eig_eps.0)
            .with_tol(tol.0)
            .with_max_sweeps(max_sweeps.0);
        Ok(Self {
            source: MdsSource::Distances {
                flat,
                num_nodes: nrows,
                num_cols: ncols,
            },
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
        match (self.target, self.dtype, &self.source) {
            (SolverTargetKind::Layout2d, ScalarDtype::F64, MdsSource::Graph(g)) => {
                let layout = py
                    .detach(|| ClassicalMds::<Layout2d<f64>, _>::from_config(g, cfg).generate())
                    .map_err(core_error_to_py)?;
                Py::new(py, PyLayout2d::from_f64_layout(&layout))
                    .map(|o| o.into_bound(py).into_any())
            }
            (SolverTargetKind::Layout2d, ScalarDtype::F32, MdsSource::Graph(g)) => {
                let layout = py
                    .detach(|| ClassicalMds::<Layout2d<f32>, _>::from_config(g, cfg).generate())
                    .map_err(core_error_to_py)?;
                Py::new(py, PyLayout2d::from_f32_layout(&layout))
                    .map(|o| o.into_bound(py).into_any())
            }
            (SolverTargetKind::Layout3d, ScalarDtype::F64, MdsSource::Graph(g)) => {
                let layout = py
                    .detach(|| ClassicalMds::<Layout3d<f64>, _>::from_config(g, cfg).generate())
                    .map_err(core_error_to_py)?;
                Py::new(py, PyLayout3d::from_f64_layout(&layout))
                    .map(|o| o.into_bound(py).into_any())
            }
            (SolverTargetKind::Layout3d, ScalarDtype::F32, MdsSource::Graph(g)) => {
                let layout = py
                    .detach(|| ClassicalMds::<Layout3d<f32>, _>::from_config(g, cfg).generate())
                    .map_err(core_error_to_py)?;
                Py::new(py, PyLayout3d::from_f32_layout(&layout))
                    .map(|o| o.into_bound(py).into_any())
            }
            (SolverTargetKind::LayoutNd, ScalarDtype::F64, MdsSource::Graph(g)) => {
                let layout = py
                    .detach(|| ClassicalMds::<LayoutNd<f64>, _>::from_config(g, cfg).generate())
                    .map_err(core_error_to_py)?;
                Py::new(py, PyLayoutNd::from_f64_nd(layout)).map(|o| o.into_bound(py).into_any())
            }
            (SolverTargetKind::LayoutNd, ScalarDtype::F32, MdsSource::Graph(g)) => {
                let layout = py
                    .detach(|| ClassicalMds::<LayoutNd<f32>, _>::from_config(g, cfg).generate())
                    .map_err(core_error_to_py)?;
                Py::new(py, PyLayoutNd::from_f32_nd(layout)).map(|o| o.into_bound(py).into_any())
            }
            (
                SolverTargetKind::Layout2d,
                ScalarDtype::F64,
                MdsSource::Distances {
                    flat, num_nodes, ..
                },
            ) => {
                let n = *num_nodes;
                let layout = py
                    .detach(|| {
                        ClassicalMds::<Layout2d<f64>, f64>::from_distances(flat, n)
                            .with_config(cfg)
                            .generate()
                    })
                    .map_err(core_error_to_py)?;
                Py::new(py, PyLayout2d::from_f64_layout(&layout))
                    .map(|o| o.into_bound(py).into_any())
            }
            (
                SolverTargetKind::Layout2d,
                ScalarDtype::F32,
                MdsSource::Distances {
                    flat, num_nodes, ..
                },
            ) => {
                let n = *num_nodes;
                let layout = py
                    .detach(|| {
                        ClassicalMds::<Layout2d<f32>, f64>::from_distances(flat, n)
                            .with_config(cfg)
                            .generate()
                    })
                    .map_err(core_error_to_py)?;
                Py::new(py, PyLayout2d::from_f32_layout(&layout))
                    .map(|o| o.into_bound(py).into_any())
            }
            (
                SolverTargetKind::Layout3d,
                ScalarDtype::F64,
                MdsSource::Distances {
                    flat, num_nodes, ..
                },
            ) => {
                let n = *num_nodes;
                let layout = py
                    .detach(|| {
                        ClassicalMds::<Layout3d<f64>, f64>::from_distances(flat, n)
                            .with_config(cfg)
                            .generate()
                    })
                    .map_err(core_error_to_py)?;
                Py::new(py, PyLayout3d::from_f64_layout(&layout))
                    .map(|o| o.into_bound(py).into_any())
            }
            (
                SolverTargetKind::Layout3d,
                ScalarDtype::F32,
                MdsSource::Distances {
                    flat, num_nodes, ..
                },
            ) => {
                let n = *num_nodes;
                let layout = py
                    .detach(|| {
                        ClassicalMds::<Layout3d<f32>, f64>::from_distances(flat, n)
                            .with_config(cfg)
                            .generate()
                    })
                    .map_err(core_error_to_py)?;
                Py::new(py, PyLayout3d::from_f32_layout(&layout))
                    .map(|o| o.into_bound(py).into_any())
            }
            (
                SolverTargetKind::LayoutNd,
                ScalarDtype::F64,
                MdsSource::Distances {
                    flat, num_nodes, ..
                },
            ) => {
                let n = *num_nodes;
                let layout = py
                    .detach(|| {
                        ClassicalMds::<LayoutNd<f64>, f64>::from_distances(flat, n)
                            .with_config(cfg)
                            .generate()
                    })
                    .map_err(core_error_to_py)?;
                Py::new(py, PyLayoutNd::from_f64_nd(layout)).map(|o| o.into_bound(py).into_any())
            }
            (
                SolverTargetKind::LayoutNd,
                ScalarDtype::F32,
                MdsSource::Distances {
                    flat, num_nodes, ..
                },
            ) => {
                let n = *num_nodes;
                let layout = py
                    .detach(|| {
                        ClassicalMds::<LayoutNd<f32>, f64>::from_distances(flat, n)
                            .with_config(cfg)
                            .generate()
                    })
                    .map_err(core_error_to_py)?;
                Py::new(py, PyLayoutNd::from_f32_nd(layout)).map(|o| o.into_bound(py).into_any())
            }
            (SolverTargetKind::Manifold2d, _, _) => Err(unsupported_operation_err(
                "ClassicalMds does not support target='manifold2d'",
            )),
        }
    }
}

/// Compute a `dims`-dimensional `LayoutNd` via zero-guarded Classical MDS on `graph`.
#[pyfunction]
#[pyo3(
    signature = (graph, dims = StrictUsize(2), *, dtype = ScalarDtype::F64),
    text_signature = "(graph, dims=2, *, dtype='float64')"
)]
fn classical_mds(
    py: Python<'_>,
    graph: &PyCsrGraph,
    dims: StrictUsize,
    dtype: ScalarDtype,
) -> PyResult<PyLayoutNd> {
    if dims.0 == 0 {
        return Err(invalid_input_err("dims must be >= 1"));
    }
    let g = graph.inner();
    let cfg = ClassicalMdsConfig::new().with_dims(Some(dims.0));
    py.detach(|| match dtype {
        ScalarDtype::F64 => ClassicalMds::<LayoutNd<f64>, _>::from_config(&*g, cfg)
            .generate()
            .map(PyLayoutNd::from_f64_nd),
        ScalarDtype::F32 => ClassicalMds::<LayoutNd<f32>, _>::from_config(&*g, cfg)
            .generate()
            .map(PyLayoutNd::from_f32_nd),
    })
    .map_err(core_error_to_py)
}

/// Compute a `dims`-dimensional `LayoutNd` via Classical MDS from an `(N, N)` distance matrix.
#[pyfunction]
#[pyo3(
    signature = (distances, dims = StrictUsize(2), *, dtype = ScalarDtype::F64),
    text_signature = "(distances, dims=2, *, dtype='float64')"
)]
fn classical_mds_from_distances(
    py: Python<'_>,
    distances: &Bound<'_, PyAny>,
    dims: StrictUsize,
    dtype: ScalarDtype,
) -> PyResult<PyLayoutNd> {
    if dims.0 == 0 {
        return Err(invalid_input_err("dims must be >= 1"));
    }
    let (flat, nrows, ncols) = extract_f64_2d_raw(distances, "distances")?;
    if nrows != ncols {
        return Err(dimension_mismatch_err(format!(
            "distances must be a square (N, N) matrix, got ({nrows}, {ncols})"
        )));
    }
    let cfg = ClassicalMdsConfig::new().with_dims(Some(dims.0));
    py.detach(|| match dtype {
        ScalarDtype::F64 => ClassicalMds::<LayoutNd<f64>, f64>::from_distances(&flat, nrows)
            .with_config(cfg)
            .generate()
            .map(PyLayoutNd::from_f64_nd),
        ScalarDtype::F32 => ClassicalMds::<LayoutNd<f32>, f64>::from_distances(&flat, nrows)
            .with_config(cfg)
            .generate()
            .map(PyLayoutNd::from_f32_nd),
    })
    .map_err(core_error_to_py)
}

/// Compute a 2D `Layout2d` via zero-guarded Classical MDS from all-pairs shortest paths.
#[pyfunction]
#[pyo3(
    signature = (graph, *, dtype = ScalarDtype::F64),
    text_signature = "(graph, *, dtype='float64')"
)]
fn classical_mds_2d(
    py: Python<'_>,
    graph: &PyCsrGraph,
    dtype: ScalarDtype,
) -> PyResult<PyLayout2d> {
    let g = graph.inner();
    py.detach(|| match dtype {
        ScalarDtype::F64 => ClassicalMds::<Layout2d<f64>, _>::new(&*g)
            .generate()
            .map(|l| PyLayout2d::from_f64_layout(&l)),
        ScalarDtype::F32 => ClassicalMds::<Layout2d<f32>, _>::new(&*g)
            .generate()
            .map(|l| PyLayout2d::from_f32_layout(&l)),
    })
    .map_err(core_error_to_py)
}

pub(super) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyClassicalMds>()?;
    m.add_function(wrap_pyfunction!(classical_mds, m)?)?;
    m.add_function(wrap_pyfunction!(classical_mds_from_distances, m)?)?;
    m.add_function(wrap_pyfunction!(classical_mds_2d, m)?)?;
    Ok(())
}
