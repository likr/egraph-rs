//! Graph UMAP (`GuMapSolver`) layout solver bindings.

use egraph::{layout as egraph_layout, neighborhood as egraph_neighborhood};
use egraph_layout::{GuMapConfig, GuMapSolver};
use egraph_neighborhood::DogVaradhanConfig;
use pyo3::prelude::*;

use crate::csr_graph::PyCsrGraph;
use crate::ffi_convert::{
    ScalarDtype, StrictF64, StrictU64, StrictUsize, core_error_to_py, solver_status_str,
};
use crate::layout_buffers::PyLayout2d;
use crate::neighborhood::{PyDogVaradhanLandmarks, PyKnnGraph, PySparseProbabilities};
use crate::solver_common::{
    DualIterativeSolver, SolverAction, execute_solver_action, impl_dual_solver,
};

impl_dual_solver!(GuMapInner, GuMapSolver);

#[allow(clippy::too_many_arguments)]
fn build_gumap_config(
    n_epochs: usize,
    negative_sample_rate: usize,
    learning_rate: f64,
    min_dist: f64,
    spread: f64,
    a: Option<f64>,
    b: Option<f64>,
    sigma_iters: usize,
    sigma_tolerance: f64,
    symmetric_negative: bool,
    seed: u64,
) -> PyResult<GuMapConfig> {
    let cfg = GuMapConfig {
        n_epochs,
        negative_sample_rate,
        learning_rate,
        min_dist,
        spread,
        a,
        b,
        sigma_iters,
        sigma_tolerance,
        symmetric_negative,
        seed,
    };
    cfg.validate().map_err(core_error_to_py)?;
    Ok(cfg)
}

/// Cooperative Graph UMAP (`GuMap`) layout solver with `dist_sq.powf(b)` gradient evaluation.
///
/// Supports both `"float64"` and `"float32"` `Layout2d` and releases the Python GIL during stepping.
#[pyclass(name = "GuMapSolver", module = "egraph._core")]
pub struct PyGuMapSolver {
    inner: GuMapInner,
}

#[pymethods]
impl PyGuMapSolver {
    /// Construct a `GuMapSolver` from a precomputed fuzzy simplicial set `SparseProbabilities`.
    #[staticmethod]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(
        signature = (
            probs,
            *,
            n_epochs = StrictUsize(200),
            negative_sample_rate = StrictUsize(5),
            learning_rate = StrictF64(1.0),
            min_dist = StrictF64(0.1),
            spread = StrictF64(1.0),
            a = None,
            b = None,
            sigma_iters = StrictUsize(64),
            sigma_tolerance = StrictF64(1e-5),
            symmetric_negative = true,
            seed = StrictU64(42),
            dtype = ScalarDtype::F64
        ),
        text_signature = "(probs, *, n_epochs=200, negative_sample_rate=5, learning_rate=1.0, min_dist=0.1, spread=1.0, a=None, b=None, sigma_iters=64, sigma_tolerance=1e-5, symmetric_negative=True, seed=42, dtype='float64')"
    )]
    fn from_fuzzy_set(
        py: Python<'_>,
        probs: &PySparseProbabilities,
        n_epochs: StrictUsize,
        negative_sample_rate: StrictUsize,
        learning_rate: StrictF64,
        min_dist: StrictF64,
        spread: StrictF64,
        a: Option<StrictF64>,
        b: Option<StrictF64>,
        sigma_iters: StrictUsize,
        sigma_tolerance: StrictF64,
        symmetric_negative: bool,
        seed: StrictU64,
        dtype: ScalarDtype,
    ) -> PyResult<Self> {
        let cfg = build_gumap_config(
            n_epochs.0,
            negative_sample_rate.0,
            learning_rate.0,
            min_dist.0,
            spread.0,
            a.map(|v| v.0),
            b.map(|v| v.0),
            sigma_iters.0,
            sigma_tolerance.0,
            symmetric_negative,
            seed.0,
        )?;
        let probs_ref = probs.inner();
        let inner = py
            .detach(|| match dtype {
                ScalarDtype::F64 => {
                    GuMapSolver::<f64>::from_fuzzy_set_ref(probs_ref, cfg).map(GuMapInner::F64)
                }
                ScalarDtype::F32 => {
                    GuMapSolver::<f32>::from_fuzzy_set_ref(probs_ref, cfg).map(GuMapInner::F32)
                }
            })
            .map_err(core_error_to_py)?;
        Ok(Self { inner })
    }

    /// Construct a `GuMapSolver` from a validated `KnnGraph` via `compute_fuzzy_simplicial_set`.
    #[staticmethod]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(
        signature = (
            knn,
            *,
            n_epochs = StrictUsize(200),
            negative_sample_rate = StrictUsize(5),
            learning_rate = StrictF64(1.0),
            min_dist = StrictF64(0.1),
            spread = StrictF64(1.0),
            a = None,
            b = None,
            sigma_iters = StrictUsize(64),
            sigma_tolerance = StrictF64(1e-5),
            symmetric_negative = true,
            seed = StrictU64(42),
            dtype = ScalarDtype::F64
        ),
        text_signature = "(knn, *, n_epochs=200, negative_sample_rate=5, learning_rate=1.0, min_dist=0.1, spread=1.0, a=None, b=None, sigma_iters=64, sigma_tolerance=1e-5, symmetric_negative=True, seed=42, dtype='float64')"
    )]
    fn from_knn(
        py: Python<'_>,
        knn: &PyKnnGraph,
        n_epochs: StrictUsize,
        negative_sample_rate: StrictUsize,
        learning_rate: StrictF64,
        min_dist: StrictF64,
        spread: StrictF64,
        a: Option<StrictF64>,
        b: Option<StrictF64>,
        sigma_iters: StrictUsize,
        sigma_tolerance: StrictF64,
        symmetric_negative: bool,
        seed: StrictU64,
        dtype: ScalarDtype,
    ) -> PyResult<Self> {
        let cfg = build_gumap_config(
            n_epochs.0,
            negative_sample_rate.0,
            learning_rate.0,
            min_dist.0,
            spread.0,
            a.map(|v| v.0),
            b.map(|v| v.0),
            sigma_iters.0,
            sigma_tolerance.0,
            symmetric_negative,
            seed.0,
        )?;
        let knn_ref = knn.inner();
        let inner = py
            .detach(|| match dtype {
                ScalarDtype::F64 => GuMapSolver::<f64>::from_knn(knn_ref, cfg).map(GuMapInner::F64),
                ScalarDtype::F32 => GuMapSolver::<f32>::from_knn(knn_ref, cfg).map(GuMapInner::F32),
            })
            .map_err(core_error_to_py)?;
        Ok(Self { inner })
    }

    /// Construct a `GuMapSolver` from a `CsrGraph` using 1+2-hop priority CSR and regularized DoG Varadhan distances.
    #[staticmethod]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(
        signature = (
            graph,
            max_2hop_per_node = StrictUsize(32),
            *,
            degree = None,
            num_landmarks = StrictUsize(64),
            degree_normalize = true,
            p_exp = StrictF64(0.5),
            eps_psd = StrictF64(1e-4),
            t_coarse = StrictF64(48.0),
            t1 = None,
            t2 = None,
            alpha = None,
            beta_floor = None,
            min_degree = StrictUsize(2),
            jl_dim = StrictUsize(64),
            n_epochs = StrictUsize(200),
            negative_sample_rate = StrictUsize(5),
            learning_rate = StrictF64(1.0),
            min_dist = StrictF64(0.1),
            spread = StrictF64(1.0),
            a = None,
            b = None,
            sigma_iters = StrictUsize(64),
            sigma_tolerance = StrictF64(1e-5),
            symmetric_negative = true,
            seed = StrictU64(42),
            dtype = ScalarDtype::F64
        ),
        text_signature = "(graph, max_2hop_per_node=32, *, degree=None, num_landmarks=64, degree_normalize=True, p_exp=0.5, eps_psd=1e-4, t_coarse=48.0, t1=None, t2=None, alpha=None, beta_floor=None, min_degree=2, jl_dim=64, n_epochs=200, negative_sample_rate=5, learning_rate=1.0, min_dist=0.1, spread=1.0, a=None, b=None, sigma_iters=64, sigma_tolerance=1e-5, symmetric_negative=True, seed=42, dtype='float64')"
    )]
    fn from_graph_dog(
        py: Python<'_>,
        graph: &PyCsrGraph,
        max_2hop_per_node: StrictUsize,
        degree: Option<StrictUsize>,
        num_landmarks: StrictUsize,
        degree_normalize: bool,
        p_exp: StrictF64,
        eps_psd: StrictF64,
        t_coarse: StrictF64,
        t1: Option<StrictF64>,
        t2: Option<StrictF64>,
        alpha: Option<StrictF64>,
        beta_floor: Option<StrictF64>,
        min_degree: StrictUsize,
        jl_dim: StrictUsize,
        n_epochs: StrictUsize,
        negative_sample_rate: StrictUsize,
        learning_rate: StrictF64,
        min_dist: StrictF64,
        spread: StrictF64,
        a: Option<StrictF64>,
        b: Option<StrictF64>,
        sigma_iters: StrictUsize,
        sigma_tolerance: StrictF64,
        symmetric_negative: bool,
        seed: StrictU64,
        dtype: ScalarDtype,
    ) -> PyResult<Self> {
        let cfg = build_gumap_config(
            n_epochs.0,
            negative_sample_rate.0,
            learning_rate.0,
            min_dist.0,
            spread.0,
            a.map(|v| v.0),
            b.map(|v| v.0),
            sigma_iters.0,
            sigma_tolerance.0,
            symmetric_negative,
            seed.0,
        )?;
        let dog_cfg = DogVaradhanConfig {
            t1: t1.map(|v| v.0),
            t2: t2.map(|v| v.0),
            alpha: alpha.map(|v| v.0),
            beta_floor: beta_floor.map(|v| v.0),
            degree: degree.map(|v| v.0),
            n_landmarks: num_landmarks.0,
            min_degree: min_degree.0,
            jl_dim: jl_dim.0,
            eps_psd: eps_psd.0,
            degree_normalize,
            p_exp: p_exp.0,
            t_coarse: t_coarse.0,
            seed: seed.0,
            ..DogVaradhanConfig::default()
        };
        let g = graph.inner();
        let inner = py
            .detach(|| match dtype {
                ScalarDtype::F64 => {
                    GuMapSolver::<f64>::from_graph_dog(g, max_2hop_per_node.0, &dog_cfg, cfg)
                        .map(GuMapInner::F64)
                }
                ScalarDtype::F32 => {
                    GuMapSolver::<f32>::from_graph_dog(g, max_2hop_per_node.0, &dog_cfg, cfg)
                        .map(GuMapInner::F32)
                }
            })
            .map_err(core_error_to_py)?;
        Ok(Self { inner })
    }

    /// Construct a `GuMapSolver` from precomputed `DogVaradhanLandmarks` and 1+2-hop priority CSR.
    #[staticmethod]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(
        signature = (
            graph,
            landmarks,
            max_2hop_per_node = StrictUsize(32),
            *,
            n_epochs = StrictUsize(200),
            negative_sample_rate = StrictUsize(5),
            learning_rate = StrictF64(1.0),
            min_dist = StrictF64(0.1),
            spread = StrictF64(1.0),
            a = None,
            b = None,
            sigma_iters = StrictUsize(64),
            sigma_tolerance = StrictF64(1e-5),
            symmetric_negative = true,
            seed = StrictU64(42),
            dtype = ScalarDtype::F64
        ),
        text_signature = "(graph, landmarks, max_2hop_per_node=32, *, n_epochs=200, negative_sample_rate=5, learning_rate=1.0, min_dist=0.1, spread=1.0, a=None, b=None, sigma_iters=64, sigma_tolerance=1e-5, symmetric_negative=True, seed=42, dtype='float64')"
    )]
    fn from_landmarks(
        py: Python<'_>,
        graph: &PyCsrGraph,
        landmarks: &PyDogVaradhanLandmarks,
        max_2hop_per_node: StrictUsize,
        n_epochs: StrictUsize,
        negative_sample_rate: StrictUsize,
        learning_rate: StrictF64,
        min_dist: StrictF64,
        spread: StrictF64,
        a: Option<StrictF64>,
        b: Option<StrictF64>,
        sigma_iters: StrictUsize,
        sigma_tolerance: StrictF64,
        symmetric_negative: bool,
        seed: StrictU64,
        dtype: ScalarDtype,
    ) -> PyResult<Self> {
        let cfg = build_gumap_config(
            n_epochs.0,
            negative_sample_rate.0,
            learning_rate.0,
            min_dist.0,
            spread.0,
            a.map(|v| v.0),
            b.map(|v| v.0),
            sigma_iters.0,
            sigma_tolerance.0,
            symmetric_negative,
            seed.0,
        )?;
        let g = graph.inner();
        let lm = landmarks.inner();
        let inner = py
            .detach(|| match dtype {
                ScalarDtype::F64 => {
                    GuMapSolver::<f64>::from_landmarks(g, lm, max_2hop_per_node.0, cfg)
                        .map(GuMapInner::F64)
                }
                ScalarDtype::F32 => {
                    GuMapSolver::<f32>::from_landmarks(g, lm, max_2hop_per_node.0, cfg)
                        .map(GuMapInner::F32)
                }
            })
            .map_err(core_error_to_py)?;
        Ok(Self { inner })
    }

    /// Execute a single optimization epoch in place on `layout` while releasing the GIL.
    #[pyo3(signature = (layout), text_signature = "($self, layout)")]
    fn step(&mut self, py: Python<'_>, layout: &Bound<'_, PyLayout2d>) -> PyResult<&'static str> {
        execute_solver_action(&mut self.inner, py, layout, SolverAction::Step)
    }

    /// Execute up to `steps` optimization epochs in place on `layout` while releasing the GIL.
    #[pyo3(signature = (layout, steps), text_signature = "($self, layout, steps)")]
    fn run_steps(
        &mut self,
        py: Python<'_>,
        layout: &Bound<'_, PyLayout2d>,
        steps: StrictUsize,
    ) -> PyResult<&'static str> {
        execute_solver_action(&mut self.inner, py, layout, SolverAction::RunSteps(steps.0))
    }

    /// Run optimization to completion in place on `layout` while releasing the GIL.
    #[pyo3(signature = (layout), text_signature = "($self, layout)")]
    fn run(&mut self, py: Python<'_>, layout: &Bound<'_, PyLayout2d>) -> PyResult<&'static str> {
        execute_solver_action(&mut self.inner, py, layout, SolverAction::Run)
    }

    /// Current solver status (`"initialized"`, `"running"`, `"converged"`, or `"completed"`).
    #[getter]
    fn status(&self) -> &'static str {
        solver_status_str(self.inner.solver_status())
    }

    /// Number of epochs executed so far.
    #[getter]
    fn iteration(&self) -> usize {
        self.inner.solver_iteration()
    }

    /// Scheduled total number of epochs (`n_epochs`).
    #[getter]
    fn total_iterations(&self) -> usize {
        self.inner.solver_total_iterations()
    }

    /// Whether the solver has reached `"converged"` or `"completed"`.
    #[getter]
    fn is_finished(&self) -> bool {
        self.inner.solver_status().is_finished()
    }

    /// Number of stored directed positive probability edges.
    #[getter]
    fn num_edges(&self) -> usize {
        match &self.inner {
            GuMapInner::F64(s) => s.num_edges(),
            GuMapInner::F32(s) => s.num_edges(),
        }
    }

    /// Expected number of vertices in `Layout2d`.
    #[getter]
    fn num_nodes(&self) -> usize {
        match &self.inner {
            GuMapInner::F64(s) => s.num_nodes(),
            GuMapInner::F32(s) => s.num_nodes(),
        }
    }

    /// Resolved low-dimensional curve parameters `(a, b)`.
    #[getter]
    fn ab(&self) -> (f64, f64) {
        match &self.inner {
            GuMapInner::F64(s) => s.ab(),
            GuMapInner::F32(s) => s.ab(),
        }
    }

    /// Solver scalar precision (`"float64"` or `"float32"`).
    #[getter]
    fn dtype(&self) -> &'static str {
        self.inner.dtype().as_str()
    }

    /// Return a concise string representation of the `GuMapSolver`.
    fn __repr__(&self) -> String {
        format!(
            "GuMapSolver(num_nodes={}, num_edges={}, status='{}', iteration={}/{}, dtype='{}')",
            self.num_nodes(),
            self.num_edges(),
            self.status(),
            self.iteration(),
            self.total_iterations(),
            self.dtype()
        )
    }
}
