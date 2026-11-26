//! Spatial intruder post-processing refiner (`SpatialIntruderRefiner`) bindings.

use egraph::layout as egraph_layout;
use egraph_layout::{SpatialIntruderRefiner, SpatialRefinerConfig, SpatialRefinerStats};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use crate::csr_graph::PyCsrGraph;
use crate::ffi_convert::{
    ScalarDtype, StrictF64, StrictU64, StrictUsize, core_error_to_py, solver_status_str,
};
use crate::layout_buffers::PyLayout2d;
use crate::solver_common::{
    DualIterativeSolver, SolverAction, execute_solver_action, impl_dual_solver,
};

impl_dual_solver!(RefinerInner, SpatialIntruderRefiner);

fn refiner_stats_to_py_dict(py: Python<'_>, stats: &SpatialRefinerStats) -> PyResult<Py<PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item("rounds_executed", stats.rounds_executed)?;
    dict.set_item("n_anchors_total", stats.n_anchors_total)?;
    dict.set_item("n_fn_total", stats.n_fn_total)?;
    dict.set_item("n_fp_total", stats.n_fp_total)?;
    dict.set_item("n_sep_total", stats.n_sep_total)?;
    dict.set_item("n_pairs_total", stats.n_pairs_total)?;
    dict.set_item("initial_scale", stats.initial_scale)?;
    dict.set_item("final_scale", stats.final_scale)?;
    dict.set_item("d_floor", stats.d_floor)?;
    dict.set_item("scale_ratio", stats.scale_ratio)?;
    dict.set_item("used_hub_fallback", stats.used_hub_fallback)?;
    dict.set_item("hinge_fp", stats.hinge_fp)?;

    let rounds_list = PyList::empty(py);
    for r in &stats.round_stats {
        let rd = PyDict::new(py);
        rd.set_item("round", r.round)?;
        rd.set_item("n_anchors", r.n_anchors)?;
        rd.set_item("n_fn", r.n_fn)?;
        rd.set_item("n_fp", r.n_fp)?;
        rd.set_item("n_sep", r.n_sep)?;
        rd.set_item("n_pairs_total", r.n_pairs_total)?;
        rd.set_item("skipped", r.skipped)?;
        rounds_list.append(rd)?;
    }
    dict.set_item("round_stats", rounds_list)?;
    Ok(dict.unbind())
}

/// Cooperative $O(N \log N)$ `StaticKdTree` active-set spatial intruder refiner.
///
/// Detects strictly `> 2`-hop spatial false positives inside each vertex's local spatial neighborhood
/// and resolves them while protecting 1-hop anchors. Releases the Python GIL during stepping.
#[pyclass(name = "SpatialIntruderRefiner", module = "egraph._core")]
pub struct PySpatialIntruderRefiner {
    inner: RefinerInner,
}

#[pymethods]
impl PySpatialIntruderRefiner {
    /// Create a `SpatialIntruderRefiner` precomputing 1+2-hop topology from `graph`.
    #[new]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(
        signature = (
            graph,
            rounds = StrictUsize(4),
            k_cap = StrictUsize(32),
            margin_scale = StrictF64(1.2),
            w_anchor = StrictF64(1.0),
            w_fn = StrictF64(0.0),
            w_fp = StrictF64(0.2),
            fn_shrink = StrictF64(0.88),
            round_decay = StrictF64(0.75),
            t_max = StrictUsize(20),
            eps = StrictF64(0.05),
            hinge_fp = true,
            seed = StrictU64(42),
            *,
            w_anchor2 = StrictF64(0.0),
            k_cap_decay = StrictF64(1.0),
            k_cap_min = StrictUsize(1),
            min_sep_ratio = StrictF64(0.0),
            w_sep = StrictF64(0.0),
            dtype = ScalarDtype::F64
        ),
        text_signature = "(graph, rounds=4, k_cap=32, margin_scale=1.2, w_anchor=1.0, w_fn=0.0, w_fp=0.2, fn_shrink=0.88, round_decay=0.75, t_max=20, eps=0.05, hinge_fp=True, seed=42, *, w_anchor2=0.0, k_cap_decay=1.0, k_cap_min=1, min_sep_ratio=0.0, w_sep=0.0, dtype='float64')"
    )]
    fn new(
        py: Python<'_>,
        graph: &PyCsrGraph,
        rounds: StrictUsize,
        k_cap: StrictUsize,
        margin_scale: StrictF64,
        w_anchor: StrictF64,
        w_fn: StrictF64,
        w_fp: StrictF64,
        fn_shrink: StrictF64,
        round_decay: StrictF64,
        t_max: StrictUsize,
        eps: StrictF64,
        hinge_fp: bool,
        seed: StrictU64,
        w_anchor2: StrictF64,
        k_cap_decay: StrictF64,
        k_cap_min: StrictUsize,
        min_sep_ratio: StrictF64,
        w_sep: StrictF64,
        dtype: ScalarDtype,
    ) -> PyResult<Self> {
        let cfg = SpatialRefinerConfig {
            rounds: rounds.0,
            k_cap: k_cap.0,
            margin_scale: margin_scale.0,
            w_anchor: w_anchor.0,
            w_anchor2: w_anchor2.0,
            w_fn: w_fn.0,
            w_fp: w_fp.0,
            fn_shrink: fn_shrink.0,
            round_decay: round_decay.0,
            k_cap_decay: k_cap_decay.0,
            k_cap_min: k_cap_min.0,
            t_max: t_max.0,
            eps: eps.0,
            hinge_fp,
            min_sep_ratio: min_sep_ratio.0,
            w_sep: w_sep.0,
            seed: seed.0,
        };
        let g = graph.inner();
        let inner = py
            .detach(|| match dtype {
                ScalarDtype::F64 => {
                    SpatialIntruderRefiner::<f64>::new(g, cfg).map(RefinerInner::F64)
                }
                ScalarDtype::F32 => {
                    SpatialIntruderRefiner::<f32>::new(g, cfg).map(RefinerInner::F32)
                }
            })
            .map_err(core_error_to_py)?;
        Ok(Self { inner })
    }

    /// Execute one spatial intruder refinement round in place on `layout` while releasing the GIL.
    #[pyo3(signature = (layout), text_signature = "($self, layout)")]
    fn step(&mut self, py: Python<'_>, layout: &Bound<'_, PyLayout2d>) -> PyResult<&'static str> {
        execute_solver_action(&mut self.inner, py, layout, SolverAction::Step)
    }

    /// Execute up to `steps` refinement rounds in place on `layout` while releasing the GIL.
    #[pyo3(signature = (layout, steps), text_signature = "($self, layout, steps)")]
    fn run_steps(
        &mut self,
        py: Python<'_>,
        layout: &Bound<'_, PyLayout2d>,
        steps: StrictUsize,
    ) -> PyResult<&'static str> {
        execute_solver_action(&mut self.inner, py, layout, SolverAction::RunSteps(steps.0))
    }

    /// Run spatial refinement to completion (or early convergence) in place on `layout` while releasing the GIL.
    #[pyo3(signature = (layout), text_signature = "($self, layout)")]
    fn run(&mut self, py: Python<'_>, layout: &Bound<'_, PyLayout2d>) -> PyResult<&'static str> {
        execute_solver_action(&mut self.inner, py, layout, SolverAction::Run)
    }

    /// Return cumulative and per-round refinement diagnostics as a dict.
    fn stats(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let stats = match &self.inner {
            RefinerInner::F64(r) => r.stats(),
            RefinerInner::F32(r) => r.stats(),
        };
        refiner_stats_to_py_dict(py, stats)
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

    /// Whether the refiner has reached `"converged"` or `"completed"`.
    #[getter]
    fn is_finished(&self) -> bool {
        self.inner.solver_status().is_finished()
    }

    /// Expected number of vertices in `Layout2d`.
    #[getter]
    fn num_nodes(&self) -> usize {
        match &self.inner {
            RefinerInner::F64(r) => r.num_nodes(),
            RefinerInner::F32(r) => r.num_nodes(),
        }
    }

    /// Current dynamic active-set neighborhood cap `k_cap`.
    #[getter]
    fn current_k_cap(&self) -> usize {
        match &self.inner {
            RefinerInner::F64(r) => r.current_k_cap(),
            RefinerInner::F32(r) => r.current_k_cap(),
        }
    }

    /// Update the active-set neighborhood cap `k_cap` for subsequent refinement rounds.
    #[pyo3(signature = (k_cap), text_signature = "($self, k_cap)")]
    fn set_k_cap(&mut self, py: Python<'_>, k_cap: StrictUsize) -> PyResult<()> {
        let cap = k_cap.0;
        py.detach(|| match &mut self.inner {
            RefinerInner::F64(r) => r.set_k_cap(cap),
            RefinerInner::F32(r) => r.set_k_cap(cap),
        })
        .map_err(core_error_to_py)
    }

    /// Refiner scalar precision (`"float64"` or `"float32"`).
    #[getter]
    fn dtype(&self) -> &'static str {
        self.inner.dtype().as_str()
    }

    /// Return a concise string representation of the `SpatialIntruderRefiner`.
    fn __repr__(&self) -> String {
        format!(
            "SpatialIntruderRefiner(num_nodes={}, status='{}', round={}/{}, dtype='{}')",
            self.num_nodes(),
            self.status(),
            self.iteration(),
            self.total_iterations(),
            self.dtype()
        )
    }
}
