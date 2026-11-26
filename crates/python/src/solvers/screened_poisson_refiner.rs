//! Screened Poisson unilateral upper-bound density field refiner (`ScreenedPoissonRefiner`).

use egraph::{core as egraph_core, layout as egraph_layout};
use egraph_core::CoreError;
use egraph_layout::{ScreenedPoissonRefiner, ScreenedPoissonRefinerConfig};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::csr_graph::PyCsrGraph;
use crate::ffi_convert::{
    ScalarDtype, StrictF64, StrictU64, StrictUsize, core_error_to_py, extract_u32_1d,
    solver_status_str,
};
use crate::layout_buffers::{Layout2dInner, PyLayout2d};
use crate::solver_common::{
    DualIterativeSolver, SolverAction, SolverLayout, execute_solver_action, extract_solver_layout,
    impl_dual_solver, write_back_solver_layout,
};

use super::screened_poisson_field::{
    build_screened_poisson_config, screened_poisson_stats_to_dict,
};

impl_dual_solver!(ScreenedPoissonRefinerInner, ScreenedPoissonRefiner);

#[allow(clippy::too_many_arguments)]
fn build_screened_poisson_refiner_config(
    kernel: &str,
    grid_size: StrictUsize,
    padding_factor: StrictUsize,
    margin_ratio: StrictF64,
    sigma_ratio: StrictF64,
    sigma: Option<StrictF64>,
    screening_length_ratio: StrictF64,
    kappa: Option<StrictF64>,
    rho_max_factor: StrictF64,
    rho_max: Option<StrictF64>,
    exact_adjoint: bool,
    iterations: StrictUsize,
    step_size: StrictF64,
    momentum: StrictF64,
    max_step_ratio: StrictF64,
    anchor_weight: StrictF64,
    gamma_tether: StrictF64,
    recenter: bool,
    energy_tol: StrictF64,
    seed: StrictU64,
) -> PyResult<ScreenedPoissonRefinerConfig> {
    let field = build_screened_poisson_config(
        kernel,
        grid_size,
        padding_factor,
        margin_ratio,
        sigma_ratio,
        sigma,
        screening_length_ratio,
        kappa,
        rho_max_factor,
        rho_max,
        exact_adjoint,
    )?;
    let cfg = ScreenedPoissonRefinerConfig {
        field,
        iterations: iterations.0,
        step_size: step_size.0,
        momentum: momentum.0,
        max_step_ratio: max_step_ratio.0,
        anchor_weight: anchor_weight.0,
        gamma_tether: gamma_tether.0,
        recenter,
        energy_tol: energy_tol.0,
        seed: seed.0,
    };
    cfg.validate().map_err(core_error_to_py)?;
    Ok(cfg)
}

/// Cooperative Screened Poisson unilateral upper-bound density field refiner.
///
/// Supports `step()`, `run_steps()`, `run()`, and `apply_epoch_step()` with optional 1-hop edge
/// anchors and cluster centroid tethering. Releases the Python GIL during stepping.
#[pyclass(name = "ScreenedPoissonRefiner", module = "egraph._core")]
pub struct PyScreenedPoissonRefiner {
    inner: ScreenedPoissonRefinerInner,
}

#[pymethods]
impl PyScreenedPoissonRefiner {
    /// Create a standalone `ScreenedPoissonRefiner` for `num_nodes` vertices.
    #[new]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(
        signature = (
            num_nodes,
            *,
            kernel = "screened_poisson",
            grid_size = StrictUsize(2048),
            padding_factor = StrictUsize(2),
            margin_ratio = StrictF64(0.1),
            sigma_ratio = StrictF64(0.015),
            sigma = None,
            screening_length_ratio = StrictF64(0.20),
            kappa = None,
            rho_max_factor = StrictF64(4.0),
            rho_max = None,
            exact_adjoint = true,
            iterations = StrictUsize(20),
            step_size = StrictF64(0.25),
            momentum = StrictF64(0.5),
            max_step_ratio = StrictF64(0.05),
            anchor_weight = StrictF64(0.0),
            gamma_tether = StrictF64(0.0),
            recenter = true,
            energy_tol = StrictF64(1e-12),
            seed = StrictU64(42),
            dtype = ScalarDtype::F64
        ),
        text_signature = "(num_nodes, *, kernel='screened_poisson', grid_size=2048, padding_factor=2, margin_ratio=0.1, sigma_ratio=0.015, sigma=None, screening_length_ratio=0.2, kappa=None, rho_max_factor=4.0, rho_max=None, exact_adjoint=True, iterations=20, step_size=0.25, momentum=0.5, max_step_ratio=0.05, anchor_weight=0.0, gamma_tether=0.0, recenter=True, energy_tol=1e-12, seed=42, dtype='float64')"
    )]
    fn new(
        py: Python<'_>,
        num_nodes: StrictUsize,
        kernel: &str,
        grid_size: StrictUsize,
        padding_factor: StrictUsize,
        margin_ratio: StrictF64,
        sigma_ratio: StrictF64,
        sigma: Option<StrictF64>,
        screening_length_ratio: StrictF64,
        kappa: Option<StrictF64>,
        rho_max_factor: StrictF64,
        rho_max: Option<StrictF64>,
        exact_adjoint: bool,
        iterations: StrictUsize,
        step_size: StrictF64,
        momentum: StrictF64,
        max_step_ratio: StrictF64,
        anchor_weight: StrictF64,
        gamma_tether: StrictF64,
        recenter: bool,
        energy_tol: StrictF64,
        seed: StrictU64,
        dtype: ScalarDtype,
    ) -> PyResult<Self> {
        let cfg = build_screened_poisson_refiner_config(
            kernel,
            grid_size,
            padding_factor,
            margin_ratio,
            sigma_ratio,
            sigma,
            screening_length_ratio,
            kappa,
            rho_max_factor,
            rho_max,
            exact_adjoint,
            iterations,
            step_size,
            momentum,
            max_step_ratio,
            anchor_weight,
            gamma_tether,
            recenter,
            energy_tol,
            seed,
        )?;
        let n = num_nodes.0;
        let inner = py
            .detach(|| match dtype {
                ScalarDtype::F64 => {
                    ScreenedPoissonRefiner::<f64>::new(n, cfg).map(ScreenedPoissonRefinerInner::F64)
                }
                ScalarDtype::F32 => {
                    ScreenedPoissonRefiner::<f32>::new(n, cfg).map(ScreenedPoissonRefinerInner::F32)
                }
            })
            .map_err(core_error_to_py)?;
        Ok(Self { inner })
    }

    /// Create a `ScreenedPoissonRefiner` retaining 1-hop edges from `graph` for optional anchors.
    #[staticmethod]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(
        signature = (
            graph,
            *,
            kernel = "screened_poisson",
            grid_size = StrictUsize(2048),
            padding_factor = StrictUsize(2),
            margin_ratio = StrictF64(0.1),
            sigma_ratio = StrictF64(0.015),
            sigma = None,
            screening_length_ratio = StrictF64(0.20),
            kappa = None,
            rho_max_factor = StrictF64(4.0),
            rho_max = None,
            exact_adjoint = true,
            iterations = StrictUsize(20),
            step_size = StrictF64(0.25),
            momentum = StrictF64(0.5),
            max_step_ratio = StrictF64(0.05),
            anchor_weight = StrictF64(0.0),
            gamma_tether = StrictF64(0.0),
            recenter = true,
            energy_tol = StrictF64(1e-12),
            seed = StrictU64(42),
            dtype = ScalarDtype::F64
        ),
        text_signature = "(graph, *, kernel='screened_poisson', grid_size=2048, padding_factor=2, margin_ratio=0.1, sigma_ratio=0.015, sigma=None, screening_length_ratio=0.2, kappa=None, rho_max_factor=4.0, rho_max=None, exact_adjoint=True, iterations=20, step_size=0.25, momentum=0.5, max_step_ratio=0.05, anchor_weight=0.0, gamma_tether=0.0, recenter=True, energy_tol=1e-12, seed=42, dtype='float64')"
    )]
    fn from_graph(
        py: Python<'_>,
        graph: &PyCsrGraph,
        kernel: &str,
        grid_size: StrictUsize,
        padding_factor: StrictUsize,
        margin_ratio: StrictF64,
        sigma_ratio: StrictF64,
        sigma: Option<StrictF64>,
        screening_length_ratio: StrictF64,
        kappa: Option<StrictF64>,
        rho_max_factor: StrictF64,
        rho_max: Option<StrictF64>,
        exact_adjoint: bool,
        iterations: StrictUsize,
        step_size: StrictF64,
        momentum: StrictF64,
        max_step_ratio: StrictF64,
        anchor_weight: StrictF64,
        gamma_tether: StrictF64,
        recenter: bool,
        energy_tol: StrictF64,
        seed: StrictU64,
        dtype: ScalarDtype,
    ) -> PyResult<Self> {
        let cfg = build_screened_poisson_refiner_config(
            kernel,
            grid_size,
            padding_factor,
            margin_ratio,
            sigma_ratio,
            sigma,
            screening_length_ratio,
            kappa,
            rho_max_factor,
            rho_max,
            exact_adjoint,
            iterations,
            step_size,
            momentum,
            max_step_ratio,
            anchor_weight,
            gamma_tether,
            recenter,
            energy_tol,
            seed,
        )?;
        let g = graph.inner();
        let inner = py
            .detach(|| match dtype {
                ScalarDtype::F64 => ScreenedPoissonRefiner::<f64>::with_graph(g, cfg)
                    .map(ScreenedPoissonRefinerInner::F64),
                ScalarDtype::F32 => ScreenedPoissonRefiner::<f32>::with_graph(g, cfg)
                    .map(ScreenedPoissonRefinerInner::F32),
            })
            .map_err(core_error_to_py)?;
        Ok(Self { inner })
    }

    /// Attach or replace a coarse-level cluster centroid tether (`align_clustered_centroids`).
    #[pyo3(signature = (coarse_layout, node_to_coarse), text_signature = "($self, coarse_layout, node_to_coarse)")]
    fn set_coarse_tether(
        &mut self,
        py: Python<'_>,
        coarse_layout: &PyLayout2d,
        node_to_coarse: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        let n2c = extract_u32_1d(node_to_coarse, "node_to_coarse")?;
        match (&mut self.inner, coarse_layout.inner()) {
            (ScreenedPoissonRefinerInner::F64(refiner), Layout2dInner::F64(buf)) => {
                let coarse_soa = buf.to_layout2d();
                py.detach(|| refiner.set_coarse_tether(coarse_soa, n2c))
                    .map_err(core_error_to_py)
            }
            (ScreenedPoissonRefinerInner::F32(refiner), Layout2dInner::F32(buf)) => {
                let coarse_soa = buf.to_layout2d();
                py.detach(|| refiner.set_coarse_tether(coarse_soa, n2c))
                    .map_err(core_error_to_py)
            }
            _ => Err(PyValueError::new_err(format!(
                "coarse_layout dtype '{}' does not match ScreenedPoissonRefiner dtype '{}'",
                coarse_layout.dtype(),
                self.dtype()
            ))),
        }
    }

    /// Execute one density refinement step in place on `layout` while releasing the GIL.
    #[pyo3(signature = (layout), text_signature = "($self, layout)")]
    fn step(&mut self, py: Python<'_>, layout: &Bound<'_, PyLayout2d>) -> PyResult<&'static str> {
        execute_solver_action(&mut self.inner, py, layout, SolverAction::Step)
    }

    /// Execute up to `steps` density refinement steps in place on `layout` while releasing the GIL.
    #[pyo3(signature = (layout, steps), text_signature = "($self, layout, steps)")]
    fn run_steps(
        &mut self,
        py: Python<'_>,
        layout: &Bound<'_, PyLayout2d>,
        steps: StrictUsize,
    ) -> PyResult<&'static str> {
        execute_solver_action(&mut self.inner, py, layout, SolverAction::RunSteps(steps.0))
    }

    /// Run density refinement to completion or early convergence in place on `layout`.
    #[pyo3(signature = (layout), text_signature = "($self, layout)")]
    fn run(&mut self, py: Python<'_>, layout: &Bound<'_, PyLayout2d>) -> PyResult<&'static str> {
        execute_solver_action(&mut self.inner, py, layout, SolverAction::Run)
    }

    /// Execute one decoupled epoch-boundary density step in place on `layout` and return stats.
    #[pyo3(signature = (layout), text_signature = "($self, layout)")]
    fn apply_epoch_step(
        &mut self,
        py: Python<'_>,
        layout: &Bound<'_, PyLayout2d>,
    ) -> PyResult<Py<PyDict>> {
        let target_dtype = self.inner.dtype();
        let mut soa = extract_solver_layout(layout, target_dtype)?;
        let stats = py
            .detach(|| match (&mut self.inner, &mut soa) {
                (ScreenedPoissonRefinerInner::F64(r), SolverLayout::F64(l)) => {
                    r.apply_epoch_step(l)
                }
                (ScreenedPoissonRefinerInner::F32(r), SolverLayout::F32(l)) => {
                    r.apply_epoch_step(l)
                }
                _ => Err(CoreError::invalid_input(
                    "internal solver layout dtype mismatch",
                )),
            })
            .map_err(core_error_to_py)?;
        write_back_solver_layout(layout, &soa)?;
        screened_poisson_stats_to_dict(py, &stats)
    }

    /// Return telemetry from the most recent density field evaluation as a dict.
    #[pyo3(text_signature = "($self)")]
    fn last_stats(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let st = match &self.inner {
            ScreenedPoissonRefinerInner::F64(r) => *r.last_stats(),
            ScreenedPoissonRefinerInner::F32(r) => *r.last_stats(),
        };
        screened_poisson_stats_to_dict(py, &st)
    }

    /// Return total heap bytes allocated by the underlying density field workspace.
    #[pyo3(text_signature = "($self)")]
    fn workspace_memory_bytes(&self) -> usize {
        match &self.inner {
            ScreenedPoissonRefinerInner::F64(r) => r.workspace_memory_bytes(),
            ScreenedPoissonRefinerInner::F32(r) => r.workspace_memory_bytes(),
        }
    }

    /// Most recently evaluated variational density energy.
    #[getter]
    fn last_energy(&self) -> f64 {
        match &self.inner {
            ScreenedPoissonRefinerInner::F64(r) => r.last_energy(),
            ScreenedPoissonRefinerInner::F32(r) => r.last_energy(),
        }
    }

    /// Current solver status (`"initialized"`, `"running"`, `"converged"`, or `"completed"`).
    #[getter]
    fn status(&self) -> &'static str {
        solver_status_str(self.inner.solver_status())
    }

    /// Number of refinement iterations executed so far.
    #[getter]
    fn iteration(&self) -> usize {
        self.inner.solver_iteration()
    }

    /// Configured maximum number of refinement iterations.
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
            ScreenedPoissonRefinerInner::F64(r) => r.num_nodes(),
            ScreenedPoissonRefinerInner::F32(r) => r.num_nodes(),
        }
    }

    /// Refiner scalar precision (`"float64"` or `"float32"`).
    #[getter]
    fn dtype(&self) -> &'static str {
        self.inner.dtype().as_str()
    }

    /// Return a concise string representation of the `ScreenedPoissonRefiner`.
    fn __repr__(&self) -> String {
        format!(
            "ScreenedPoissonRefiner(num_nodes={}, status='{}', iter={}/{}, dtype='{}')",
            self.num_nodes(),
            self.status(),
            self.iteration(),
            self.total_iterations(),
            self.dtype()
        )
    }
}
