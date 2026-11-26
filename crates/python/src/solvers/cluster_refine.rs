//! Cluster-faithful layout refiner (`ClusterRefineSolver`, `refine_cluster_layout`) bindings.

use egraph::layout as egraph_layout;
use egraph_layout::{
    ClusterRefineConfig, ClusterRefineMethod, ClusterRefineSolver,
    refine_cluster_layout as layout_refine_cluster_layout,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::csr_graph::PyCsrGraph;
use crate::ffi_convert::{
    ScalarDtype, StrictF64, StrictU64, StrictUsize, core_error_to_py, extract_u32_1d,
    normalize_enum_key, solver_status_str,
};
use crate::layout_buffers::{Layout2dInner, PyLayout2d};
use crate::solver_common::{
    DualIterativeSolver, SolverAction, SolverLayout, execute_solver_action, extract_solver_layout,
    impl_dual_solver, write_back_solver_layout,
};

pub(super) fn parse_cluster_refine_method(raw: &str) -> Result<ClusterRefineMethod, String> {
    match normalize_enum_key(raw).as_str() {
        "kmeans" | "km" => Ok(ClusterRefineMethod::Kmeans),
        "hac" | "upgma" => Ok(ClusterRefineMethod::Hac),
        _ => Err(format!(
            "unknown cluster refine method {raw:?}; expected 'kmeans' or 'hac'"
        )),
    }
}

impl_dual_solver!(ClusterRefineInner, ClusterRefineSolver);

/// Cooperative cluster-faithful layout refiner (`Cluster-kmeans` and `Cluster-HAC`, Cai et al., 2024).
///
/// Supports both `"float64"` and `"float32"` `Layout2d` while releasing the Python GIL.
#[pyclass(name = "ClusterRefineSolver", module = "egraph._core")]
pub struct PyClusterRefineSolver {
    inner: ClusterRefineInner,
}

#[pymethods]
impl PyClusterRefineSolver {
    /// Construct a `ClusterRefineSolver` from per-node `cluster_labels`.
    #[new]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(
        signature = (
            cluster_labels,
            *,
            method = "kmeans",
            max_rounds = StrictUsize(10),
            kmeans_n_init = StrictUsize(5),
            kmeans_max_iters = StrictUsize(100),
            shift_step = StrictF64(1.0),
            margin = StrictF64(0.1),
            seed = StrictU64(42),
            dtype = ScalarDtype::F64
        ),
        text_signature = "(cluster_labels, *, method='kmeans', max_rounds=10, kmeans_n_init=5, kmeans_max_iters=100, shift_step=1.0, margin=0.1, seed=42, dtype='float64')"
    )]
    fn new(
        py: Python<'_>,
        cluster_labels: &Bound<'_, PyAny>,
        method: &str,
        max_rounds: StrictUsize,
        kmeans_n_init: StrictUsize,
        kmeans_max_iters: StrictUsize,
        shift_step: StrictF64,
        margin: StrictF64,
        seed: StrictU64,
        dtype: ScalarDtype,
    ) -> PyResult<Self> {
        let parsed_method = parse_cluster_refine_method(method).map_err(PyValueError::new_err)?;
        let labels = extract_u32_1d(cluster_labels, "cluster_labels")?;
        let cfg = ClusterRefineConfig {
            method: parsed_method,
            max_rounds: max_rounds.0,
            kmeans_n_init: kmeans_n_init.0,
            kmeans_max_iters: kmeans_max_iters.0,
            shift_step: shift_step.0,
            margin: margin.0,
            seed: seed.0,
        };
        cfg.validate().map_err(core_error_to_py)?;
        let inner = py
            .detach(|| match dtype {
                ScalarDtype::F64 => {
                    ClusterRefineSolver::<f64>::new(labels, cfg).map(ClusterRefineInner::F64)
                }
                ScalarDtype::F32 => {
                    ClusterRefineSolver::<f32>::new(labels, cfg).map(ClusterRefineInner::F32)
                }
            })
            .map_err(core_error_to_py)?;
        Ok(Self { inner })
    }

    /// Construct a `ClusterRefineSolver` using `graph` edges for mean edge length scaling in HAC mode.
    #[staticmethod]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(
        signature = (
            graph,
            cluster_labels,
            *,
            method = "kmeans",
            max_rounds = StrictUsize(10),
            kmeans_n_init = StrictUsize(5),
            kmeans_max_iters = StrictUsize(100),
            shift_step = StrictF64(1.0),
            margin = StrictF64(0.1),
            seed = StrictU64(42),
            dtype = ScalarDtype::F64
        ),
        text_signature = "(graph, cluster_labels, *, method='kmeans', max_rounds=10, kmeans_n_init=5, kmeans_max_iters=100, shift_step=1.0, margin=0.1, seed=42, dtype='float64')"
    )]
    fn from_graph(
        py: Python<'_>,
        graph: &PyCsrGraph,
        cluster_labels: &Bound<'_, PyAny>,
        method: &str,
        max_rounds: StrictUsize,
        kmeans_n_init: StrictUsize,
        kmeans_max_iters: StrictUsize,
        shift_step: StrictF64,
        margin: StrictF64,
        seed: StrictU64,
        dtype: ScalarDtype,
    ) -> PyResult<Self> {
        let parsed_method = parse_cluster_refine_method(method).map_err(PyValueError::new_err)?;
        let labels = extract_u32_1d(cluster_labels, "cluster_labels")?;
        let cfg = ClusterRefineConfig {
            method: parsed_method,
            max_rounds: max_rounds.0,
            kmeans_n_init: kmeans_n_init.0,
            kmeans_max_iters: kmeans_max_iters.0,
            shift_step: shift_step.0,
            margin: margin.0,
            seed: seed.0,
        };
        cfg.validate().map_err(core_error_to_py)?;
        let g = graph.inner();
        let inner = py
            .detach(|| match dtype {
                ScalarDtype::F64 => ClusterRefineSolver::<f64>::from_graph(g, labels, cfg)
                    .map(ClusterRefineInner::F64),
                ScalarDtype::F32 => ClusterRefineSolver::<f32>::from_graph(g, labels, cfg)
                    .map(ClusterRefineInner::F32),
            })
            .map_err(core_error_to_py)?;
        Ok(Self { inner })
    }

    /// Execute one cluster refinement round in place on `layout` while releasing the GIL.
    #[pyo3(signature = (layout), text_signature = "($self, layout)")]
    fn step(&mut self, py: Python<'_>, layout: &Bound<'_, PyLayout2d>) -> PyResult<&'static str> {
        execute_solver_action(&mut self.inner, py, layout, SolverAction::Step)
    }

    /// Execute up to `steps` cluster refinement rounds in place on `layout` while releasing the GIL.
    #[pyo3(signature = (layout, steps), text_signature = "($self, layout, steps)")]
    fn run_steps(
        &mut self,
        py: Python<'_>,
        layout: &Bound<'_, PyLayout2d>,
        steps: StrictUsize,
    ) -> PyResult<&'static str> {
        execute_solver_action(&mut self.inner, py, layout, SolverAction::RunSteps(steps.0))
    }

    /// Run cluster refinement until convergence or `max_rounds` in place on `layout`.
    #[pyo3(signature = (layout), text_signature = "($self, layout)")]
    fn run(&mut self, py: Python<'_>, layout: &Bound<'_, PyLayout2d>) -> PyResult<&'static str> {
        execute_solver_action(&mut self.inner, py, layout, SolverAction::Run)
    }

    /// Number of misplaced vertices shifted in the most recent refinement round.
    #[getter]
    fn last_moved_count(&self) -> usize {
        match &self.inner {
            ClusterRefineInner::F64(s) => s.last_moved_count(),
            ClusterRefineInner::F32(s) => s.last_moved_count(),
        }
    }

    /// Number of distinct ground-truth clusters.
    #[getter]
    fn num_clusters(&self) -> usize {
        match &self.inner {
            ClusterRefineInner::F64(s) => s.num_clusters(),
            ClusterRefineInner::F32(s) => s.num_clusters(),
        }
    }

    /// Current solver status (`"initialized"`, `"running"`, `"converged"`, or `"completed"`).
    #[getter]
    fn status(&self) -> &'static str {
        solver_status_str(self.inner.solver_status())
    }

    /// Number of refinement rounds executed so far.
    #[getter]
    fn iteration(&self) -> usize {
        self.inner.solver_iteration()
    }

    /// Configured maximum number of refinement rounds.
    #[getter]
    fn total_iterations(&self) -> usize {
        self.inner.solver_total_iterations()
    }

    /// Whether the solver has reached `"converged"` or `"completed"`.
    #[getter]
    fn is_finished(&self) -> bool {
        self.inner.solver_status().is_finished()
    }

    /// Expected number of vertices in `Layout2d`.
    #[getter]
    fn num_nodes(&self) -> usize {
        match &self.inner {
            ClusterRefineInner::F64(s) => s.num_nodes(),
            ClusterRefineInner::F32(s) => s.num_nodes(),
        }
    }

    /// Solver scalar precision (`"float64"` or `"float32"`).
    #[getter]
    fn dtype(&self) -> &'static str {
        self.inner.dtype().as_str()
    }

    /// Return a concise string representation of the `ClusterRefineSolver`.
    fn __repr__(&self) -> String {
        format!(
            "ClusterRefineSolver(num_nodes={}, num_clusters={}, last_moved_count={}, status='{}', iteration={}/{}, dtype='{}')",
            self.num_nodes(),
            self.num_clusters(),
            self.last_moved_count(),
            self.status(),
            self.iteration(),
            self.total_iterations(),
            self.dtype()
        )
    }
}

/// Refine `layout` in place to improve geometric cluster faithfulness and return total vertex shifts.
///
/// Releases the Python GIL during computation.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(
    signature = (
        layout,
        cluster_labels,
        *,
        method = "kmeans",
        max_rounds = StrictUsize(10),
        kmeans_n_init = StrictUsize(5),
        kmeans_max_iters = StrictUsize(100),
        shift_step = StrictF64(1.0),
        margin = StrictF64(0.1),
        seed = StrictU64(42)
    ),
    text_signature = "(layout, cluster_labels, *, method='kmeans', max_rounds=10, kmeans_n_init=5, kmeans_max_iters=100, shift_step=1.0, margin=0.1, seed=42)"
)]
fn refine_cluster_layout(
    py: Python<'_>,
    layout: &Bound<'_, PyLayout2d>,
    cluster_labels: &Bound<'_, PyAny>,
    method: &str,
    max_rounds: StrictUsize,
    kmeans_n_init: StrictUsize,
    kmeans_max_iters: StrictUsize,
    shift_step: StrictF64,
    margin: StrictF64,
    seed: StrictU64,
) -> PyResult<usize> {
    let parsed_method = parse_cluster_refine_method(method).map_err(PyValueError::new_err)?;
    let labels = extract_u32_1d(cluster_labels, "cluster_labels")?;
    let cfg = ClusterRefineConfig {
        method: parsed_method,
        max_rounds: max_rounds.0,
        kmeans_n_init: kmeans_n_init.0,
        kmeans_max_iters: kmeans_max_iters.0,
        shift_step: shift_step.0,
        margin: margin.0,
        seed: seed.0,
    };
    cfg.validate().map_err(core_error_to_py)?;
    let target_dtype = match layout.try_borrow()?.inner() {
        Layout2dInner::F64(_) => ScalarDtype::F64,
        Layout2dInner::F32(_) => ScalarDtype::F32,
    };
    let mut soa = extract_solver_layout(layout, target_dtype)?;
    let total_moved = match &mut soa {
        SolverLayout::F64(l64) => py
            .detach(|| layout_refine_cluster_layout(l64, &labels, &cfg))
            .map_err(core_error_to_py)?,
        SolverLayout::F32(l32) => py
            .detach(|| layout_refine_cluster_layout(l32, &labels, &cfg))
            .map_err(core_error_to_py)?,
    };
    write_back_solver_layout(layout, &soa)?;
    Ok(total_moved)
}

pub(super) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyClusterRefineSolver>()?;
    m.add_function(wrap_pyfunction!(refine_cluster_layout, m)?)?;
    Ok(())
}
