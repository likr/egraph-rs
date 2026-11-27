//! Quadrilateral Simmelian backbone extraction and layout (`BackboneSolver`) bindings.

use egraph::layout as egraph_layout;
use egraph_layout::{
    BackboneConfig, BackboneLayoutMode, BackboneResult, BackboneSolver,
    extract_quadrilateral_backbone as layout_extract_quadrilateral_backbone,
};
use numpy::PyArray1;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::csr_graph::PyCsrGraph;
use crate::ffi_convert::{
    ScalarDtype, StrictF64, StrictU64, StrictUsize, core_error_to_py, normalize_enum_key,
    pairs_to_u32_array2, solver_status_str,
};
use crate::layout_buffers::PyLayout2d;
use crate::solver_common::{
    DualIterativeSolver, SolverAction, execute_solver_action, impl_dual_solver,
};

pub(super) fn parse_backbone_layout_mode(raw: &str) -> Result<BackboneLayoutMode, String> {
    match normalize_enum_key(raw).as_str() {
        "stress" | "stressmajorization" | "sm" => Ok(BackboneLayoutMode::StressMajorization),
        "sgd" => Ok(BackboneLayoutMode::Sgd),
        _ => Err(format!(
            "unknown backbone layout_mode {raw:?}; expected 'stress' or 'sgd'"
        )),
    }
}

impl_dual_solver!(BackboneInner, BackboneSolver);

fn backbone_result_to_py_dict(py: Python<'_>, res: BackboneResult) -> PyResult<Py<PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item("umst_edges", pairs_to_u32_array2(py, res.umst_edges)?)?;
    dict.set_item(
        "backbone_edges",
        pairs_to_u32_array2(py, res.backbone_edges)?,
    )?;
    dict.set_item(
        "simmelian_weights",
        PyArray1::from_vec(py, res.simmelian_weights),
    )?;
    dict.set_item("edge_weights", PyArray1::from_vec(py, res.edge_weights))?;
    dict.set_item("backbone_csr", PyCsrGraph::from_csr_graph(res.backbone_csr))?;
    Ok(dict.unbind())
}

/// Extract the Quadrilateral Simmelian Backbone (Nocaj, Ortmann, & Brandes, 2015) from `graph`.
///
/// Returns a dict with `"umst_edges"`, `"backbone_edges"`, `"simmelian_weights"`,
/// `"edge_weights"`, and `"backbone_csr"`. Releases the Python GIL during computation.
#[pyfunction]
#[pyo3(
    signature = (
        graph,
        *,
        retain_ratio = StrictF64(0.2),
        umst_weight_scale = StrictF64(1.0),
        non_backbone_weight_scale = StrictF64(0.0),
        use_weights = true
    ),
    text_signature = "(graph, *, retain_ratio=0.2, umst_weight_scale=1.0, non_backbone_weight_scale=0.0, use_weights=True)"
)]
fn extract_quadrilateral_backbone(
    py: Python<'_>,
    graph: &PyCsrGraph,
    retain_ratio: StrictF64,
    umst_weight_scale: StrictF64,
    non_backbone_weight_scale: StrictF64,
    use_weights: bool,
) -> PyResult<Py<PyDict>> {
    let cfg = BackboneConfig {
        retain_ratio: retain_ratio.0,
        umst_weight_scale: umst_weight_scale.0,
        non_backbone_weight_scale: non_backbone_weight_scale.0,
        use_weights,
        ..BackboneConfig::default()
    };
    cfg.validate().map_err(core_error_to_py)?;
    let g = graph.inner();
    let res = py
        .detach(|| layout_extract_quadrilateral_backbone(&*g, &cfg))
        .map_err(core_error_to_py)?;
    backbone_result_to_py_dict(py, res)
}

/// Cooperative Quadrilateral Simmelian Backbone layout solver (Nocaj et al., 2015).
///
/// Extracts the Simmelian backbone and optimizes node coordinates using `"stress"` or `"sgd"`
/// in either `"float64"` or `"float32"` precision while releasing the Python GIL.
#[pyclass(name = "BackboneSolver", module = "egraph._core")]
pub struct PyBackboneSolver {
    inner: BackboneInner,
}

#[pymethods]
impl PyBackboneSolver {
    /// Construct a `BackboneSolver` by extracting the quadrilateral backbone from `graph`.
    #[new]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(
        signature = (
            graph,
            *,
            retain_ratio = StrictF64(0.2),
            umst_weight_scale = StrictF64(1.0),
            non_backbone_weight_scale = StrictF64(0.0),
            use_weights = true,
            layout_mode = "stress",
            iterations = StrictUsize(100),
            seed = StrictU64(42),
            dtype = ScalarDtype::F64
        ),
        text_signature = "(graph, *, retain_ratio=0.2, umst_weight_scale=1.0, non_backbone_weight_scale=0.0, use_weights=True, layout_mode='stress', iterations=100, seed=42, dtype='float64')"
    )]
    fn new(
        py: Python<'_>,
        graph: &PyCsrGraph,
        retain_ratio: StrictF64,
        umst_weight_scale: StrictF64,
        non_backbone_weight_scale: StrictF64,
        use_weights: bool,
        layout_mode: &str,
        iterations: StrictUsize,
        seed: StrictU64,
        dtype: ScalarDtype,
    ) -> PyResult<Self> {
        let mode = parse_backbone_layout_mode(layout_mode).map_err(PyValueError::new_err)?;
        let cfg = BackboneConfig {
            retain_ratio: retain_ratio.0,
            umst_weight_scale: umst_weight_scale.0,
            non_backbone_weight_scale: non_backbone_weight_scale.0,
            use_weights,
            layout_mode: mode,
            iterations: iterations.0,
            seed: seed.0,
        };
        cfg.validate().map_err(core_error_to_py)?;
        let g = graph.inner();
        let inner = py
            .detach(|| match dtype {
                ScalarDtype::F64 => BackboneSolver::<f64>::new(&*g, cfg).map(BackboneInner::F64),
                ScalarDtype::F32 => BackboneSolver::<f32>::new(&*g, cfg).map(BackboneInner::F32),
            })
            .map_err(core_error_to_py)?;
        Ok(Self { inner })
    }

    /// Execute one backbone layout iteration in place on `layout` while releasing the GIL.
    #[pyo3(signature = (layout), text_signature = "($self, layout)")]
    fn step(&mut self, py: Python<'_>, layout: &Bound<'_, PyLayout2d>) -> PyResult<&'static str> {
        execute_solver_action(&mut self.inner, py, layout, SolverAction::Step)
    }

    /// Execute up to `steps` backbone layout iterations in place on `layout` while releasing the GIL.
    #[pyo3(signature = (layout, steps), text_signature = "($self, layout, steps)")]
    fn run_steps(
        &mut self,
        py: Python<'_>,
        layout: &Bound<'_, PyLayout2d>,
        steps: StrictUsize,
    ) -> PyResult<&'static str> {
        execute_solver_action(&mut self.inner, py, layout, SolverAction::RunSteps(steps.0))
    }

    /// Run backbone layout optimization to completion in place on `layout` while releasing the GIL.
    #[pyo3(signature = (layout), text_signature = "($self, layout)")]
    fn run(&mut self, py: Python<'_>, layout: &Bound<'_, PyLayout2d>) -> PyResult<&'static str> {
        execute_solver_action(&mut self.inner, py, layout, SolverAction::Run)
    }

    /// Return the extracted quadrilateral Simmelian backbone dictionary.
    #[pyo3(signature = (), text_signature = "($self)")]
    fn backbone_result(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let res = match &self.inner {
            BackboneInner::F64(s) => s.backbone_result().clone(),
            BackboneInner::F32(s) => s.backbone_result().clone(),
        };
        backbone_result_to_py_dict(py, res)
    }

    /// Current solver status (`"initialized"`, `"running"`, `"converged"`, or `"completed"`).
    #[getter]
    fn status(&self) -> &'static str {
        solver_status_str(self.inner.solver_status())
    }

    /// Number of iterations executed so far.
    #[getter]
    fn iteration(&self) -> usize {
        self.inner.solver_iteration()
    }

    /// Configured maximum number of layout iterations.
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
            BackboneInner::F64(s) => s.num_nodes(),
            BackboneInner::F32(s) => s.num_nodes(),
        }
    }

    /// Solver scalar precision (`"float64"` or `"float32"`).
    #[getter]
    fn dtype(&self) -> &'static str {
        self.inner.dtype().as_str()
    }

    /// Return a concise string representation of the `BackboneSolver`.
    fn __repr__(&self) -> String {
        format!(
            "BackboneSolver(num_nodes={}, status='{}', iteration={}/{}, dtype='{}')",
            self.num_nodes(),
            self.status(),
            self.iteration(),
            self.total_iterations(),
            self.dtype()
        )
    }
}

pub(super) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyBackboneSolver>()?;
    m.add_function(wrap_pyfunction!(extract_quadrilateral_backbone, m)?)?;
    Ok(())
}
