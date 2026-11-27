//! Group-in-a-Box (`GroupInABoxSolver`) and `squarified_treemap` bindings.

use egraph::layout as egraph_layout;
use egraph_layout::{
    GibBoxLayoutMode, GibIntraLayoutMode, GroupInABoxConfig, GroupInABoxSolver,
    squarified_treemap as layout_squarified_treemap,
};
use numpy::PyArray2;
use numpy::ndarray::Array2;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::array_extract::extract_f64_1d;
use crate::csr_graph::PyCsrGraph;
use crate::ffi_convert::{
    ScalarDtype, StrictF64, StrictU64, StrictUsize, core_error_to_py, extract_optional_weights_1d,
    extract_u32_1d, normalize_enum_key, solver_status_str,
};
use crate::layout_buffers::PyLayout2d;
use crate::solver_common::{
    DualIterativeSolver, SolverAction, execute_solver_action, impl_dual_solver,
};

pub(super) fn parse_gib_box_mode(raw: &str) -> Result<GibBoxLayoutMode, String> {
    match normalize_enum_key(raw).as_str() {
        "treemap" | "squarifiedtreemap" | "tr" | "trgib" => Ok(GibBoxLayoutMode::SquarifiedTreemap),
        "forcedirected" | "fd" | "fdgib" => Ok(GibBoxLayoutMode::ForceDirected),
        _ => Err(format!(
            "unknown Group-in-a-Box box_mode {raw:?}; expected 'treemap' or 'force_directed'"
        )),
    }
}

pub(super) fn parse_gib_intra_mode(raw: &str) -> Result<GibIntraLayoutMode, String> {
    match normalize_enum_key(raw).as_str() {
        "fr" | "fruchtermanreingold" => Ok(GibIntraLayoutMode::Fr),
        "sgdfull" | "sgd" => Ok(GibIntraLayoutMode::SgdFull),
        "yifanhu" | "yf" => Ok(GibIntraLayoutMode::YifanHu),
        _ => Err(format!(
            "unknown Group-in-a-Box intra_mode {raw:?}; expected 'fr', 'sgd_full', or 'yifan_hu'"
        )),
    }
}

impl_dual_solver!(GroupInABoxInner, GroupInABoxSolver);

/// Partition rectangle `[x, x + width] x [y, y + height]` into a squarified treemap (Bruls et al., 2000).
///
/// Returns a `(K, 4)` `float64` NumPy array with columns `[x_min, x_max, y_min, y_max]`.
/// Releases the Python GIL during computation.
#[pyfunction]
#[pyo3(
    signature = (weights, *, width = StrictF64(100.0), height = StrictF64(100.0), x = StrictF64(0.0), y = StrictF64(0.0)),
    text_signature = "(weights, *, width=100.0, height=100.0, x=0.0, y=0.0)"
)]
fn squarified_treemap<'py>(
    py: Python<'py>,
    weights: &Bound<'py, PyAny>,
    width: StrictF64,
    height: StrictF64,
    x: StrictF64,
    y: StrictF64,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let w_vec = extract_f64_1d(weights, "weights")?;
    let rects = py
        .detach(|| layout_squarified_treemap(&w_vec, x.0, y.0, width.0, height.0))
        .map_err(core_error_to_py)?;
    let k = rects.len();
    let mut flat = Vec::with_capacity(k * 4);
    for rect in rects {
        flat.extend_from_slice(&rect);
    }
    let arr =
        Array2::from_shape_vec((k, 4), flat).map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(PyArray2::from_owned_array(py, arr))
}

/// Cooperative Group-in-a-Box (`TR-GIB` / `FD-GIB`, Rodrigues et al., 2011) clustered graph layout solver.
///
/// Partitions the canvas into cluster boxes and runs intra-cluster layouts cooperatively across
/// both `"float64"` and `"float32"` `Layout2d` while releasing the Python GIL.
#[pyclass(name = "GroupInABoxSolver", module = "egraph._core")]
pub struct PyGroupInABoxSolver {
    inner: GroupInABoxInner,
}

#[pymethods]
impl PyGroupInABoxSolver {
    /// Construct a `GroupInABoxSolver` from `graph` and per-node `labels`.
    #[new]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(
        signature = (
            graph,
            labels,
            *,
            box_mode = "treemap",
            intra_mode = "fr",
            total_width = StrictF64(100.0),
            total_height = StrictF64(100.0),
            box_padding = StrictF64(0.05),
            intra_iterations = StrictUsize(100),
            fd_iterations = StrictUsize(100),
            fit_bounds = true,
            cluster_weights = None,
            seed = StrictU64(42),
            dtype = ScalarDtype::F64
        ),
        text_signature = "(graph, labels, *, box_mode='treemap', intra_mode='fr', total_width=100.0, total_height=100.0, box_padding=0.05, intra_iterations=100, fd_iterations=100, fit_bounds=True, cluster_weights=None, seed=42, dtype='float64')"
    )]
    fn new(
        py: Python<'_>,
        graph: &PyCsrGraph,
        labels: &Bound<'_, PyAny>,
        box_mode: &str,
        intra_mode: &str,
        total_width: StrictF64,
        total_height: StrictF64,
        box_padding: StrictF64,
        intra_iterations: StrictUsize,
        fd_iterations: StrictUsize,
        fit_bounds: bool,
        cluster_weights: Option<&Bound<'_, PyAny>>,
        seed: StrictU64,
        dtype: ScalarDtype,
    ) -> PyResult<Self> {
        let parsed_box = parse_gib_box_mode(box_mode).map_err(PyValueError::new_err)?;
        let parsed_intra = parse_gib_intra_mode(intra_mode).map_err(PyValueError::new_err)?;
        let labels_vec = extract_u32_1d(labels, "labels")?;
        let cw_opt = extract_optional_weights_1d(cluster_weights, "cluster_weights")?;
        let cfg = GroupInABoxConfig {
            box_mode: parsed_box,
            intra_mode: parsed_intra,
            total_width: total_width.0,
            total_height: total_height.0,
            box_padding: box_padding.0,
            intra_iterations: intra_iterations.0,
            fd_iterations: fd_iterations.0,
            fit_bounds,
            seed: seed.0,
        };
        cfg.validate().map_err(core_error_to_py)?;
        let g = graph.inner();
        let inner = py
            .detach(|| match (dtype, cw_opt) {
                (ScalarDtype::F64, Some(cw)) => {
                    GroupInABoxSolver::<f64>::with_cluster_weights(&*g, labels_vec, cw, cfg)
                        .map(GroupInABoxInner::F64)
                }
                (ScalarDtype::F32, Some(cw)) => {
                    GroupInABoxSolver::<f32>::with_cluster_weights(&*g, labels_vec, cw, cfg)
                        .map(GroupInABoxInner::F32)
                }
                (ScalarDtype::F64, None) => {
                    GroupInABoxSolver::<f64>::new(&*g, labels_vec, cfg).map(GroupInABoxInner::F64)
                }
                (ScalarDtype::F32, None) => {
                    GroupInABoxSolver::<f32>::new(&*g, labels_vec, cfg).map(GroupInABoxInner::F32)
                }
            })
            .map_err(core_error_to_py)?;
        Ok(Self { inner })
    }

    /// Execute one Group-in-a-Box intra-cluster iteration in place on `layout` while releasing the GIL.
    #[pyo3(signature = (layout), text_signature = "($self, layout)")]
    fn step(&mut self, py: Python<'_>, layout: &Bound<'_, PyLayout2d>) -> PyResult<&'static str> {
        execute_solver_action(&mut self.inner, py, layout, SolverAction::Step)
    }

    /// Execute up to `steps` Group-in-a-Box iterations in place on `layout` while releasing the GIL.
    #[pyo3(signature = (layout, steps), text_signature = "($self, layout, steps)")]
    fn run_steps(
        &mut self,
        py: Python<'_>,
        layout: &Bound<'_, PyLayout2d>,
        steps: StrictUsize,
    ) -> PyResult<&'static str> {
        execute_solver_action(&mut self.inner, py, layout, SolverAction::RunSteps(steps.0))
    }

    /// Run Group-in-a-Box intra-cluster optimization to completion in place on `layout`.
    #[pyo3(signature = (layout), text_signature = "($self, layout)")]
    fn run(&mut self, py: Python<'_>, layout: &Bound<'_, PyLayout2d>) -> PyResult<&'static str> {
        execute_solver_action(&mut self.inner, py, layout, SolverAction::Run)
    }

    /// Return the computed cluster bounding boxes as a list of dictionaries.
    #[pyo3(signature = (), text_signature = "($self)")]
    fn boxes(&self, py: Python<'_>) -> PyResult<Vec<Py<PyDict>>> {
        let slice = match &self.inner {
            GroupInABoxInner::F64(s) => s.boxes(),
            GroupInABoxInner::F32(s) => s.boxes(),
        };
        let mut out = Vec::with_capacity(slice.len());
        for b in slice {
            let dict = PyDict::new(py);
            dict.set_item("cluster_id", b.cluster_id)?;
            dict.set_item("x_min", b.x_min)?;
            dict.set_item("x_max", b.x_max)?;
            dict.set_item("y_min", b.y_min)?;
            dict.set_item("y_max", b.y_max)?;
            dict.set_item("node_count", b.node_count)?;
            dict.set_item("weight", b.weight)?;
            out.push(dict.unbind());
        }
        Ok(out)
    }

    /// Current solver status (`"initialized"`, `"running"`, `"converged"`, or `"completed"`).
    #[getter]
    fn status(&self) -> &'static str {
        solver_status_str(self.inner.solver_status())
    }

    /// Number of intra-cluster iterations executed so far.
    #[getter]
    fn iteration(&self) -> usize {
        self.inner.solver_iteration()
    }

    /// Configured maximum number of intra-cluster iterations.
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
            GroupInABoxInner::F64(s) => s.num_nodes(),
            GroupInABoxInner::F32(s) => s.num_nodes(),
        }
    }

    /// Solver scalar precision (`"float64"` or `"float32"`).
    #[getter]
    fn dtype(&self) -> &'static str {
        self.inner.dtype().as_str()
    }

    /// Return a concise string representation of the `GroupInABoxSolver`.
    fn __repr__(&self) -> String {
        format!(
            "GroupInABoxSolver(num_nodes={}, status='{}', iteration={}/{}, dtype='{}')",
            self.num_nodes(),
            self.status(),
            self.iteration(),
            self.total_iterations(),
            self.dtype()
        )
    }
}

pub(super) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyGroupInABoxSolver>()?;
    m.add_function(wrap_pyfunction!(squarified_treemap, m)?)?;
    Ok(())
}
