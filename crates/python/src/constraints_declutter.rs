//! Twin/near-twin decluttering (`TwinDeclutterer`, `declutter_twins_2d`) and transitive reduction bindings.

use egraph::{core as egraph_core, layout as egraph_layout};
use egraph_core::LayoutTransform;
use egraph_layout::{
    TwinDeclutterConfig, TwinDeclutterer, declutter_twins_2d as layout_declutter_twins_2d,
    transitive_reduction as layout_transitive_reduction,
};
use pyo3::prelude::*;

use crate::csr_graph::PyCsrGraph;
use crate::ffi_convert::{ScalarDtype, StrictF64, StrictUsize, core_error_to_py};
use crate::layout_buffers::{Layout2dInner, PyLayout2d};
use crate::solver_common::{SolverLayout, extract_solver_layout, write_back_solver_layout};

enum TwinDecluttererInner {
    F64(TwinDeclutterer<f64>),
    F32(TwinDeclutterer<f32>),
}

impl TwinDecluttererInner {
    fn dtype(&self) -> ScalarDtype {
        match self {
            Self::F64(_) => ScalarDtype::F64,
            Self::F32(_) => ScalarDtype::F32,
        }
    }
}

fn build_twin_declutter_config(
    eps_rel: StrictF64,
    r_frac: StrictF64,
    r_cap_rel: StrictF64,
    min_pair_rel: StrictF64,
    iterations: StrictUsize,
) -> PyResult<TwinDeclutterConfig> {
    let cfg = TwinDeclutterConfig::new()
        .with_eps_rel(eps_rel.0)
        .with_r_frac(r_frac.0)
        .with_r_cap_rel(r_cap_rel.0)
        .with_min_pair_rel(min_pair_rel.0)
        .with_iterations(iterations.0);
    cfg.validate().map_err(core_error_to_py)?;
    Ok(cfg)
}

/// Cooperative 2D layout transform that resolves exact duplicate and near-twin node overlaps
/// using a spatial hash, cluster-centroid k-d tree clearance queries, and golden-angle Vogel spirals.
#[pyclass(name = "TwinDeclutterer", module = "egraph._core")]
pub struct PyTwinDeclutterer {
    inner: TwinDecluttererInner,
}

#[pymethods]
impl PyTwinDeclutterer {
    /// Construct a `TwinDeclutterer` with optional tolerance and spiral budget overrides.
    #[new]
    #[pyo3(
        signature = (
            *,
            eps_rel = StrictF64(1e-6),
            r_frac = StrictF64(0.35),
            r_cap_rel = StrictF64(3e-3),
            min_pair_rel = StrictF64(5e-7),
            iterations = StrictUsize(4),
            dtype = ScalarDtype::F64
        ),
        text_signature = "(*, eps_rel=1e-6, r_frac=0.35, r_cap_rel=0.003, min_pair_rel=5e-7, iterations=4, dtype='float64')"
    )]
    fn new(
        eps_rel: StrictF64,
        r_frac: StrictF64,
        r_cap_rel: StrictF64,
        min_pair_rel: StrictF64,
        iterations: StrictUsize,
        dtype: ScalarDtype,
    ) -> PyResult<Self> {
        let cfg =
            build_twin_declutter_config(eps_rel, r_frac, r_cap_rel, min_pair_rel, iterations)?;
        let inner = match dtype {
            ScalarDtype::F64 => TwinDeclutterer::<f64>::new(cfg).map(TwinDecluttererInner::F64),
            ScalarDtype::F32 => TwinDeclutterer::<f32>::new(cfg).map(TwinDecluttererInner::F32),
        }
        .map_err(core_error_to_py)?;
        Ok(Self { inner })
    }

    /// Apply twin-decluttering in place on `layout` while releasing the GIL.
    #[pyo3(signature = (layout), text_signature = "($self, layout)")]
    fn apply(&mut self, py: Python<'_>, layout: &Bound<'_, PyLayout2d>) -> PyResult<()> {
        let target_dtype = self.inner.dtype();
        let mut soa = extract_solver_layout(layout, target_dtype)?;
        py.detach(|| match (&mut self.inner, &mut soa) {
            (TwinDecluttererInner::F64(s), SolverLayout::F64(l)) => s.apply(l),
            (TwinDecluttererInner::F32(s), SolverLayout::F32(l)) => s.apply(l),
            _ => unreachable!(),
        })
        .map_err(core_error_to_py)?;
        write_back_solver_layout(layout, &soa)
    }

    /// Number of decluttering passes executed in the most recent `apply()` call.
    #[getter]
    fn last_pass_count(&self) -> usize {
        match &self.inner {
            TwinDecluttererInner::F64(s) => s.last_pass_count(),
            TwinDecluttererInner::F32(s) => s.last_pass_count(),
        }
    }

    /// Scalar precision dtype (`"float64"` or `"float32"`).
    #[getter]
    fn dtype(&self) -> &'static str {
        self.inner.dtype().as_str()
    }

    /// Number of twin clusters detected in the most recent pass.
    #[getter]
    fn last_cluster_count(&self) -> usize {
        match &self.inner {
            TwinDecluttererInner::F64(s) => s.last_cluster_count(),
            TwinDecluttererInner::F32(s) => s.last_cluster_count(),
        }
    }

    /// Number of vertices repositioned in the most recent pass.
    #[getter]
    fn last_moved_count(&self) -> usize {
        match &self.inner {
            TwinDecluttererInner::F64(s) => s.last_moved_count(),
            TwinDecluttererInner::F32(s) => s.last_moved_count(),
        }
    }

    /// Relative spatial-hash merge tolerance.
    #[getter]
    fn eps_rel(&self) -> f64 {
        match &self.inner {
            TwinDecluttererInner::F64(s) => s.config().eps_rel,
            TwinDecluttererInner::F32(s) => s.config().eps_rel,
        }
    }

    /// Fraction of nearest external cluster clearance used as spiral radius budget.
    #[getter]
    fn r_frac(&self) -> f64 {
        match &self.inner {
            TwinDecluttererInner::F64(s) => s.config().r_frac,
            TwinDecluttererInner::F32(s) => s.config().r_frac,
        }
    }

    /// Hard upper cap on cluster spiral radius relative to bounding-box diagonal.
    #[getter]
    fn r_cap_rel(&self) -> f64 {
        match &self.inner {
            TwinDecluttererInner::F64(s) => s.config().r_cap_rel,
            TwinDecluttererInner::F32(s) => s.config().r_cap_rel,
        }
    }

    /// Minimum pairwise separation target relative to bounding-box diagonal.
    #[getter]
    fn min_pair_rel(&self) -> f64 {
        match &self.inner {
            TwinDecluttererInner::F64(s) => s.config().min_pair_rel,
            TwinDecluttererInner::F32(s) => s.config().min_pair_rel,
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "TwinDeclutterer(dtype='{}', last_pass_count={}, last_cluster_count={}, last_moved_count={})",
            self.dtype(),
            self.last_pass_count(),
            self.last_cluster_count(),
            self.last_moved_count(),
        )
    }
}

/// Resolve coincident and near-twin node overlaps in `layout` in place.
///
/// Preserves `layout.dtype` and returns the number of passes executed. Releases the Python GIL.
#[pyfunction]
#[pyo3(
    signature = (
        layout,
        *,
        eps_rel = StrictF64(1e-6),
        r_frac = StrictF64(0.35),
        r_cap_rel = StrictF64(3e-3),
        min_pair_rel = StrictF64(5e-7),
        iterations = StrictUsize(4)
    ),
    text_signature = "(layout, *, eps_rel=1e-6, r_frac=0.35, r_cap_rel=0.003, min_pair_rel=5e-7, iterations=4)"
)]
fn declutter_twins_2d(
    py: Python<'_>,
    layout: &Bound<'_, PyLayout2d>,
    eps_rel: StrictF64,
    r_frac: StrictF64,
    r_cap_rel: StrictF64,
    min_pair_rel: StrictF64,
    iterations: StrictUsize,
) -> PyResult<usize> {
    let cfg = build_twin_declutter_config(eps_rel, r_frac, r_cap_rel, min_pair_rel, iterations)?;
    let target_dtype = match layout.try_borrow()?.inner() {
        Layout2dInner::F64(_) => ScalarDtype::F64,
        Layout2dInner::F32(_) => ScalarDtype::F32,
    };
    let mut soa = extract_solver_layout(layout, target_dtype)?;
    let passes = py
        .detach(|| match &mut soa {
            SolverLayout::F64(l64) => layout_declutter_twins_2d(l64, &cfg),
            SolverLayout::F32(l32) => layout_declutter_twins_2d(l32, &cfg),
        })
        .map_err(core_error_to_py)?;
    write_back_solver_layout(layout, &soa)?;
    Ok(passes)
}

/// Compute the exact transitive reduction of a directed acyclic `CsrGraph`.
///
/// Rejects undirected or cyclic graphs. Releases the Python GIL during computation.
#[pyfunction]
#[pyo3(signature = (graph), text_signature = "(graph)")]
fn transitive_reduction(py: Python<'_>, graph: &PyCsrGraph) -> PyResult<PyCsrGraph> {
    let g = graph.inner();
    let reduced = py
        .detach(|| layout_transitive_reduction(&*g))
        .map_err(core_error_to_py)?;
    PyCsrGraph::from_unit_csr_graph(reduced).map_err(core_error_to_py)
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyTwinDeclutterer>()?;
    m.add_function(wrap_pyfunction!(declutter_twins_2d, m)?)?;
    m.add_function(wrap_pyfunction!(transitive_reduction, m)?)?;
    Ok(())
}
