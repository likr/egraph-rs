//! PivotMDS (`PivotMds`) and Representative MDS bindings.

use egraph::core::{Layout2d, Layout3d, LayoutGenerator as _, LayoutNd};
use egraph::linalg::{PivotMds, PivotMdsConfig};
use pyo3::prelude::*;

use super::mds::{MdsSource, validate_euclidean_target_dims};
use crate::array_extract::extract_f64_2d_raw;
use crate::csr_graph::PyCsrGraph;
use crate::errors::{core_error_to_py, invalid_input_err, unsupported_operation_err};
use crate::ffi_convert::{ScalarDtype, StrictF64, StrictU64, StrictUsize, extract_u32_1d};
use crate::layout_buffers::PyLayout2d;
use crate::layout_nd::PyLayoutNd;
use crate::layout3d::PyLayout3d;
use crate::solvers::layout_dispatch::SolverTargetKind;

/// Cooperative Pivot Multidimensional Scaling (`PivotMds`) layout generator.
#[pyclass(name = "PivotMds", module = "egraph._core")]
pub struct PyPivotMds {
    source: MdsSource,
    pivots: Option<Vec<u32>>,
    target: SolverTargetKind,
    dtype: ScalarDtype,
    config: PivotMdsConfig,
}

#[pymethods]
impl PyPivotMds {
    /// Construct a `PivotMds` generator over `graph`.
    #[new]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(
        signature = (graph, *, target = "layout2d", dims = None, num_pivots = StrictUsize(50), seed = StrictU64(42), min_degree = StrictUsize(2), target_std = Some(StrictF64(1.0)), pivots = None, dtype = ScalarDtype::F64),
        text_signature = "(graph, *, target='layout2d', dims=None, num_pivots=50, seed=42, min_degree=2, target_std=1.0, pivots=None, dtype='float64')"
    )]
    fn new(
        graph: &PyCsrGraph,
        target: &str,
        dims: Option<StrictUsize>,
        num_pivots: StrictUsize,
        seed: StrictU64,
        min_degree: StrictUsize,
        target_std: Option<StrictF64>,
        pivots: Option<&Bound<'_, PyAny>>,
        dtype: ScalarDtype,
    ) -> PyResult<Self> {
        let target_kind = SolverTargetKind::from_str(target)?;
        let resolved_dims =
            validate_euclidean_target_dims(target_kind, dims.map(|d| d.0), "PivotMds")?;
        let pivots_vec = pivots
            .filter(|p| !p.is_none())
            .map(|p| extract_u32_1d(p, "pivots"))
            .transpose()?;
        let config = PivotMdsConfig::new()
            .with_dims(resolved_dims)
            .with_num_pivots(num_pivots.0)
            .with_min_degree(min_degree.0)
            .with_seed(seed.0)
            .with_target_std(target_std.map(|v| v.0));
        Ok(Self {
            source: MdsSource::Graph(graph.inner().into_owned()),
            pivots: pivots_vec,
            target: target_kind,
            dtype,
            config,
        })
    }

    /// Construct a `PivotMds` generator from an `(N, K)` distance matrix.
    #[staticmethod]
    #[pyo3(
        signature = (distances_nk, *, target = "layout2d", dims = None, target_std = Some(StrictF64(1.0)), seed = StrictU64(42), dtype = ScalarDtype::F64),
        text_signature = "(distances_nk, *, target='layout2d', dims=None, target_std=1.0, seed=42, dtype='float64')"
    )]
    fn from_distances(
        distances_nk: &Bound<'_, PyAny>,
        target: &str,
        dims: Option<StrictUsize>,
        target_std: Option<StrictF64>,
        seed: StrictU64,
        dtype: ScalarDtype,
    ) -> PyResult<Self> {
        let target_kind = SolverTargetKind::from_str(target)?;
        let resolved_dims =
            validate_euclidean_target_dims(target_kind, dims.map(|d| d.0), "PivotMds")?;
        let (flat, n_nodes, n_pivots) = extract_f64_2d_raw(distances_nk, "distances_nk")?;
        let config = PivotMdsConfig::new()
            .with_dims(resolved_dims)
            .with_num_pivots(n_pivots)
            .with_seed(seed.0)
            .with_target_std(target_std.map(|v| v.0));
        Ok(Self {
            source: MdsSource::Distances {
                flat,
                num_nodes: n_nodes,
                num_cols: n_pivots,
            },
            pivots: None,
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
        macro_rules! run_pivot_graph {
            ($layout_ty:ty, $g:expr, $wrap:expr) => {{
                let mut generator = PivotMds::<$layout_ty, _>::from_config($g, cfg);
                if let Some(p) = piv {
                    generator = generator.with_pivots(p);
                }
                let layout = py
                    .detach(|| generator.generate())
                    .map_err(core_error_to_py)?;
                $wrap(layout)
            }};
        }
        macro_rules! run_pivot_dist {
            ($layout_ty:ty, $flat:expr, $n:expr, $k:expr, $wrap:expr) => {{
                let mut generator =
                    PivotMds::<$layout_ty, f64>::from_distances($flat, $n, $k).with_config(cfg);
                let layout = py
                    .detach(|| generator.generate())
                    .map_err(core_error_to_py)?;
                $wrap(layout)
            }};
        }
        match (self.target, self.dtype, &self.source) {
            (SolverTargetKind::Layout2d, ScalarDtype::F64, MdsSource::Graph(g)) => {
                run_pivot_graph!(Layout2d<f64>, g, |l| {
                    Py::new(py, PyLayout2d::from_f64_layout(&l))
                        .map(|o| o.into_bound(py).into_any())
                })
            }
            (SolverTargetKind::Layout2d, ScalarDtype::F32, MdsSource::Graph(g)) => {
                run_pivot_graph!(Layout2d<f32>, g, |l| {
                    Py::new(py, PyLayout2d::from_f32_layout(&l))
                        .map(|o| o.into_bound(py).into_any())
                })
            }
            (SolverTargetKind::Layout3d, ScalarDtype::F64, MdsSource::Graph(g)) => {
                run_pivot_graph!(Layout3d<f64>, g, |l| {
                    Py::new(py, PyLayout3d::from_f64_layout(&l))
                        .map(|o| o.into_bound(py).into_any())
                })
            }
            (SolverTargetKind::Layout3d, ScalarDtype::F32, MdsSource::Graph(g)) => {
                run_pivot_graph!(Layout3d<f32>, g, |l| {
                    Py::new(py, PyLayout3d::from_f32_layout(&l))
                        .map(|o| o.into_bound(py).into_any())
                })
            }
            (SolverTargetKind::LayoutNd, ScalarDtype::F64, MdsSource::Graph(g)) => {
                run_pivot_graph!(LayoutNd<f64>, g, |l| {
                    Py::new(py, PyLayoutNd::from_f64_nd(l)).map(|o| o.into_bound(py).into_any())
                })
            }
            (SolverTargetKind::LayoutNd, ScalarDtype::F32, MdsSource::Graph(g)) => {
                run_pivot_graph!(LayoutNd<f32>, g, |l| {
                    Py::new(py, PyLayoutNd::from_f32_nd(l)).map(|o| o.into_bound(py).into_any())
                })
            }
            (
                SolverTargetKind::Layout2d,
                ScalarDtype::F64,
                MdsSource::Distances {
                    flat,
                    num_nodes,
                    num_cols,
                },
            ) => run_pivot_dist!(Layout2d<f64>, flat, *num_nodes, *num_cols, |l| {
                Py::new(py, PyLayout2d::from_f64_layout(&l)).map(|o| o.into_bound(py).into_any())
            }),
            (
                SolverTargetKind::Layout2d,
                ScalarDtype::F32,
                MdsSource::Distances {
                    flat,
                    num_nodes,
                    num_cols,
                },
            ) => run_pivot_dist!(Layout2d<f32>, flat, *num_nodes, *num_cols, |l| {
                Py::new(py, PyLayout2d::from_f32_layout(&l)).map(|o| o.into_bound(py).into_any())
            }),
            (
                SolverTargetKind::Layout3d,
                ScalarDtype::F64,
                MdsSource::Distances {
                    flat,
                    num_nodes,
                    num_cols,
                },
            ) => run_pivot_dist!(Layout3d<f64>, flat, *num_nodes, *num_cols, |l| {
                Py::new(py, PyLayout3d::from_f64_layout(&l)).map(|o| o.into_bound(py).into_any())
            }),
            (
                SolverTargetKind::Layout3d,
                ScalarDtype::F32,
                MdsSource::Distances {
                    flat,
                    num_nodes,
                    num_cols,
                },
            ) => run_pivot_dist!(Layout3d<f32>, flat, *num_nodes, *num_cols, |l| {
                Py::new(py, PyLayout3d::from_f32_layout(&l)).map(|o| o.into_bound(py).into_any())
            }),
            (
                SolverTargetKind::LayoutNd,
                ScalarDtype::F64,
                MdsSource::Distances {
                    flat,
                    num_nodes,
                    num_cols,
                },
            ) => run_pivot_dist!(LayoutNd<f64>, flat, *num_nodes, *num_cols, |l| {
                Py::new(py, PyLayoutNd::from_f64_nd(l)).map(|o| o.into_bound(py).into_any())
            }),
            (
                SolverTargetKind::LayoutNd,
                ScalarDtype::F32,
                MdsSource::Distances {
                    flat,
                    num_nodes,
                    num_cols,
                },
            ) => run_pivot_dist!(LayoutNd<f32>, flat, *num_nodes, *num_cols, |l| {
                Py::new(py, PyLayoutNd::from_f32_nd(l)).map(|o| o.into_bound(py).into_any())
            }),
            (SolverTargetKind::Manifold2d, _, _) => Err(unsupported_operation_err(
                "PivotMds does not support target='manifold2d'",
            )),
        }
    }
}

/// Compute a 2D `Layout2d` via Core-First MaxMin PivotMDS (or explicit `pivots`).
#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(
    signature = (graph, *, num_pivots = StrictUsize(50), seed = StrictU64(42), min_degree = StrictUsize(2), target_std = Some(StrictF64(1.0)), pivots = None, dtype = ScalarDtype::F64),
    text_signature = "(graph, *, num_pivots=50, seed=42, min_degree=2, target_std=1.0, pivots=None, dtype='float64')"
)]
fn pivot_mds(
    py: Python<'_>,
    graph: &PyCsrGraph,
    num_pivots: StrictUsize,
    seed: StrictU64,
    min_degree: StrictUsize,
    target_std: Option<StrictF64>,
    pivots: Option<&Bound<'_, PyAny>>,
    dtype: ScalarDtype,
) -> PyResult<PyLayout2d> {
    let explicit_pivots = pivots
        .filter(|p| !p.is_none())
        .map(|p| extract_u32_1d(p, "pivots"))
        .transpose()?;
    let g = graph.inner();
    let cfg = PivotMdsConfig::new()
        .with_num_pivots(num_pivots.0)
        .with_min_degree(min_degree.0)
        .with_seed(seed.0)
        .with_target_std(target_std.map(|v| v.0));
    py.detach(|| match (dtype, explicit_pivots.as_deref()) {
        (ScalarDtype::F64, None) => PivotMds::<Layout2d<f64>, _>::from_config(&*g, cfg)
            .generate()
            .map(|l| PyLayout2d::from_f64_layout(&l)),
        (ScalarDtype::F32, None) => PivotMds::<Layout2d<f32>, _>::from_config(&*g, cfg)
            .generate()
            .map(|l| PyLayout2d::from_f32_layout(&l)),
        (ScalarDtype::F64, Some(p)) => PivotMds::<Layout2d<f64>, _>::from_config(&*g, cfg)
            .with_pivots(p)
            .generate()
            .map(|l| PyLayout2d::from_f64_layout(&l)),
        (ScalarDtype::F32, Some(p)) => PivotMds::<Layout2d<f32>, _>::from_config(&*g, cfg)
            .with_pivots(p)
            .generate()
            .map(|l| PyLayout2d::from_f32_layout(&l)),
    })
    .map_err(core_error_to_py)
}

/// Compute a 2D `Layout2d` via Core-First MaxMin PivotMDS with optional isotropic std normalization.
#[pyfunction]
#[pyo3(
    signature = (graph, *, num_pivots = StrictUsize(50), seed = StrictU64(42), min_degree = StrictUsize(2), target_std = Some(StrictF64(1.0)), dtype = ScalarDtype::F64),
    text_signature = "(graph, *, num_pivots=50, seed=42, min_degree=2, target_std=1.0, dtype='float64')"
)]
fn pivot_mds_init(
    py: Python<'_>,
    graph: &PyCsrGraph,
    num_pivots: StrictUsize,
    seed: StrictU64,
    min_degree: StrictUsize,
    target_std: Option<StrictF64>,
    dtype: ScalarDtype,
) -> PyResult<PyLayout2d> {
    pivot_mds(
        py, graph, num_pivots, seed, min_degree, target_std, None, dtype,
    )
}

/// Compute a 2D `Layout2d` via PivotMDS from a precomputed `(N, K)` distance matrix.
#[pyfunction]
#[pyo3(
    signature = (distances_nk, *, target_std = Some(StrictF64(1.0)), seed = StrictU64(42), dtype = ScalarDtype::F64),
    text_signature = "(distances_nk, *, target_std=1.0, seed=42, dtype='float64')"
)]
fn pivot_mds_from_distances(
    py: Python<'_>,
    distances_nk: &Bound<'_, PyAny>,
    target_std: Option<StrictF64>,
    seed: StrictU64,
    dtype: ScalarDtype,
) -> PyResult<PyLayout2d> {
    let (flat, n_nodes, n_pivots) = extract_f64_2d_raw(distances_nk, "distances_nk")?;
    let cfg = PivotMdsConfig::new()
        .with_seed(seed.0)
        .with_target_std(target_std.map(|v| v.0));
    py.detach(|| match dtype {
        ScalarDtype::F64 => {
            PivotMds::<Layout2d<f64>, f64>::from_distances(&flat, n_nodes, n_pivots)
                .with_config(cfg)
                .generate()
                .map(|l| PyLayout2d::from_f64_layout(&l))
        }
        ScalarDtype::F32 => {
            PivotMds::<Layout2d<f32>, f64>::from_distances(&flat, n_nodes, n_pivots)
                .with_config(cfg)
                .generate()
                .map(|l| PyLayout2d::from_f32_layout(&l))
        }
    })
    .map_err(core_error_to_py)
}

/// Compute a `dims`-dimensional `LayoutNd` via Representative/Landmark MDS from an `(N, R)` distance matrix.
#[pyfunction]
#[pyo3(
    signature = (distances_to_reps, dims = StrictUsize(2), *, dtype = ScalarDtype::F64),
    text_signature = "(distances_to_reps, dims=2, *, dtype='float64')"
)]
fn representative_mds(
    py: Python<'_>,
    distances_to_reps: &Bound<'_, PyAny>,
    dims: StrictUsize,
    dtype: ScalarDtype,
) -> PyResult<PyLayoutNd> {
    if dims.0 == 0 {
        return Err(invalid_input_err("dims must be >= 1"));
    }
    let (flat, n_nodes, n_reps) = extract_f64_2d_raw(distances_to_reps, "distances_to_reps")?;
    let cfg = PivotMdsConfig::new()
        .with_dims(Some(dims.0))
        .with_target_std(None);
    py.detach(|| match dtype {
        ScalarDtype::F64 => PivotMds::<LayoutNd<f64>, f64>::from_distances(&flat, n_nodes, n_reps)
            .with_config(cfg)
            .generate()
            .map(PyLayoutNd::from_f64_nd),
        ScalarDtype::F32 => PivotMds::<LayoutNd<f32>, f64>::from_distances(&flat, n_nodes, n_reps)
            .with_config(cfg)
            .generate()
            .map(PyLayoutNd::from_f32_nd),
    })
    .map_err(core_error_to_py)
}

pub(super) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyPivotMds>()?;
    m.add_function(wrap_pyfunction!(pivot_mds, m)?)?;
    m.add_function(wrap_pyfunction!(pivot_mds_init, m)?)?;
    m.add_function(wrap_pyfunction!(pivot_mds_from_distances, m)?)?;
    m.add_function(wrap_pyfunction!(representative_mds, m)?)?;
    Ok(())
}
