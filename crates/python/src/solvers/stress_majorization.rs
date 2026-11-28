//! Cooperative Gansner-Koren-North (2004) Stress Majorization layout solver bindings.

use egraph::core::{Layout2d, Layout3d, LayoutNd, ManifoldKind, ManifoldLayout2d};
use egraph::layout::{
    EuclideanPcg, GeodesicGradient, StressMajorizationConfig, StressMajorizationSolver,
};
use pyo3::prelude::*;

use crate::csr_graph::PyCsrGraph;
use crate::errors::core_error_to_py;
use crate::ffi_convert::{ScalarDtype, StrictF64, StrictUsize, solver_status_str};
use crate::solver_common::format_optional_usize;

use super::distance_extract::extract_distance_matrix_2d;
use super::layout_dispatch::{
    DualTargetSolver, SolverAction, SolverTargetKind, execute_target_action, extract_target_layout,
    validate_target_space,
};
use super::stress_majorization_inner::{
    StressMajorizationInner, StressStrategyChoice, match_stress_inner, match_stress_target,
    resolve_stress_strategy_pair,
};

/// Cooperative Gansner-Koren-North (2004) Stress Majorization layout solver.
#[pyclass(name = "StressMajorizationSolver", module = "egraph._core")]
pub struct PyStressMajorizationSolver {
    inner: StressMajorizationInner,
    locked_manifold: Option<ManifoldKind>,
}

#[pymethods]
impl PyStressMajorizationSolver {
    #[new]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(
        signature = (
            graph,
            *,
            target = SolverTargetKind::Layout2d,
            strategy = None,
            backend = None,
            epsilon = StrictF64(1e-4),
            max_iterations = StrictUsize(100),
            pcg_tolerance = StrictF64(1e-6),
            pcg_max_iterations = None,
            min_norm = StrictF64(1e-4),
            dtype = ScalarDtype::F64
        ),
        text_signature = "(graph, *, target='layout2d', strategy=None, backend=None, epsilon=1e-4, max_iterations=100, pcg_tolerance=1e-6, pcg_max_iterations=None, min_norm=1e-4, dtype='float64')"
    )]
    fn new(
        py: Python<'_>,
        graph: &PyCsrGraph,
        target: SolverTargetKind,
        strategy: Option<&str>,
        backend: Option<&str>,
        epsilon: StrictF64,
        max_iterations: StrictUsize,
        pcg_tolerance: StrictF64,
        pcg_max_iterations: Option<StrictUsize>,
        min_norm: StrictF64,
        dtype: ScalarDtype,
    ) -> PyResult<Self> {
        Self::from_graph(
            py,
            graph,
            target,
            strategy,
            backend,
            epsilon,
            max_iterations,
            pcg_tolerance,
            pcg_max_iterations,
            min_norm,
            dtype,
        )
    }

    #[staticmethod]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(
        signature = (
            graph,
            *,
            target = SolverTargetKind::Layout2d,
            strategy = None,
            backend = None,
            epsilon = StrictF64(1e-4),
            max_iterations = StrictUsize(100),
            pcg_tolerance = StrictF64(1e-6),
            pcg_max_iterations = None,
            min_norm = StrictF64(1e-4),
            dtype = ScalarDtype::F64
        ),
        text_signature = "(graph, *, target='layout2d', strategy=None, backend=None, epsilon=1e-4, max_iterations=100, pcg_tolerance=1e-6, pcg_max_iterations=None, min_norm=1e-4, dtype='float64')"
    )]
    fn from_graph(
        py: Python<'_>,
        graph: &PyCsrGraph,
        target: SolverTargetKind,
        strategy: Option<&str>,
        backend: Option<&str>,
        epsilon: StrictF64,
        max_iterations: StrictUsize,
        pcg_tolerance: StrictF64,
        pcg_max_iterations: Option<StrictUsize>,
        min_norm: StrictF64,
        dtype: ScalarDtype,
    ) -> PyResult<Self> {
        let strat = resolve_stress_strategy_pair(target, strategy, backend)?;
        let cfg = StressMajorizationConfig::new()
            .with_epsilon(epsilon.0)
            .with_max_iterations(max_iterations.0)
            .with_pcg_tolerance(pcg_tolerance.0)
            .with_pcg_max_iterations(pcg_max_iterations.map(|v| v.0))
            .with_min_norm(min_norm.0);
        cfg.validate().map_err(core_error_to_py)?;
        let g = graph.inner();
        let inner = py
            .detach(|| match (target, strat, dtype) {
                (SolverTargetKind::Layout2d, StressStrategyChoice::EuclideanPcg, ScalarDtype::F64) => {
                    StressMajorizationSolver::<Layout2d<f64>, EuclideanPcg>::from_graph_with_strategy(
                        &*g, cfg, EuclideanPcg::new(),
                    ).map(StressMajorizationInner::L2PcgF64)
                }
                (SolverTargetKind::Layout2d, StressStrategyChoice::EuclideanPcg, ScalarDtype::F32) => {
                    StressMajorizationSolver::<Layout2d<f32>, EuclideanPcg>::from_graph_with_strategy(
                        &*g, cfg, EuclideanPcg::new(),
                    ).map(StressMajorizationInner::L2PcgF32)
                }
                (SolverTargetKind::Layout3d, StressStrategyChoice::EuclideanPcg, ScalarDtype::F64) => {
                    StressMajorizationSolver::<Layout3d<f64>, EuclideanPcg>::from_graph_with_strategy(
                        &*g, cfg, EuclideanPcg::new(),
                    ).map(StressMajorizationInner::L3PcgF64)
                }
                (SolverTargetKind::Layout3d, StressStrategyChoice::EuclideanPcg, ScalarDtype::F32) => {
                    StressMajorizationSolver::<Layout3d<f32>, EuclideanPcg>::from_graph_with_strategy(
                        &*g, cfg, EuclideanPcg::new(),
                    ).map(StressMajorizationInner::L3PcgF32)
                }
                (SolverTargetKind::LayoutNd, StressStrategyChoice::EuclideanPcg, ScalarDtype::F64) => {
                    StressMajorizationSolver::<LayoutNd<f64>, EuclideanPcg>::from_graph_with_strategy(
                        &*g, cfg, EuclideanPcg::new(),
                    ).map(StressMajorizationInner::NdPcgF64)
                }
                (SolverTargetKind::LayoutNd, StressStrategyChoice::EuclideanPcg, ScalarDtype::F32) => {
                    StressMajorizationSolver::<LayoutNd<f32>, EuclideanPcg>::from_graph_with_strategy(
                        &*g, cfg, EuclideanPcg::new(),
                    ).map(StressMajorizationInner::NdPcgF32)
                }
                (SolverTargetKind::Layout2d, StressStrategyChoice::GeodesicGradient, ScalarDtype::F64) => {
                    StressMajorizationSolver::<Layout2d<f64>, GeodesicGradient>::from_graph_with_strategy(
                        &*g, cfg, GeodesicGradient::new(),
                    ).map(StressMajorizationInner::L2GeoF64)
                }
                (SolverTargetKind::Layout2d, StressStrategyChoice::GeodesicGradient, ScalarDtype::F32) => {
                    StressMajorizationSolver::<Layout2d<f32>, GeodesicGradient>::from_graph_with_strategy(
                        &*g, cfg, GeodesicGradient::new(),
                    ).map(StressMajorizationInner::L2GeoF32)
                }
                (SolverTargetKind::Layout3d, StressStrategyChoice::GeodesicGradient, ScalarDtype::F64) => {
                    StressMajorizationSolver::<Layout3d<f64>, GeodesicGradient>::from_graph_with_strategy(
                        &*g, cfg, GeodesicGradient::new(),
                    ).map(StressMajorizationInner::L3GeoF64)
                }
                (SolverTargetKind::Layout3d, StressStrategyChoice::GeodesicGradient, ScalarDtype::F32) => {
                    StressMajorizationSolver::<Layout3d<f32>, GeodesicGradient>::from_graph_with_strategy(
                        &*g, cfg, GeodesicGradient::new(),
                    ).map(StressMajorizationInner::L3GeoF32)
                }
                (SolverTargetKind::LayoutNd, StressStrategyChoice::GeodesicGradient, ScalarDtype::F64) => {
                    StressMajorizationSolver::<LayoutNd<f64>, GeodesicGradient>::from_graph_with_strategy(
                        &*g, cfg, GeodesicGradient::new(),
                    ).map(StressMajorizationInner::NdGeoF64)
                }
                (SolverTargetKind::LayoutNd, StressStrategyChoice::GeodesicGradient, ScalarDtype::F32) => {
                    StressMajorizationSolver::<LayoutNd<f32>, GeodesicGradient>::from_graph_with_strategy(
                        &*g, cfg, GeodesicGradient::new(),
                    ).map(StressMajorizationInner::NdGeoF32)
                }
                (SolverTargetKind::Manifold2d, StressStrategyChoice::GeodesicGradient, ScalarDtype::F64) => {
                    StressMajorizationSolver::<ManifoldLayout2d<f64>, GeodesicGradient>::from_graph_with_strategy(
                        &*g, cfg, GeodesicGradient::new(),
                    ).map(StressMajorizationInner::M2GeoF64)
                }
                (SolverTargetKind::Manifold2d, StressStrategyChoice::GeodesicGradient, ScalarDtype::F32) => {
                    StressMajorizationSolver::<ManifoldLayout2d<f32>, GeodesicGradient>::from_graph_with_strategy(
                        &*g, cfg, GeodesicGradient::new(),
                    ).map(StressMajorizationInner::M2GeoF32)
                }
                _ => unreachable!(),
            })
            .map_err(core_error_to_py)?;
        Ok(Self {
            inner,
            locked_manifold: None,
        })
    }

    #[staticmethod]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(
        signature = (
            distances,
            *,
            target = SolverTargetKind::Layout2d,
            strategy = None,
            backend = None,
            epsilon = StrictF64(1e-4),
            max_iterations = StrictUsize(100),
            pcg_tolerance = StrictF64(1e-6),
            pcg_max_iterations = None,
            min_norm = StrictF64(1e-4),
            dtype = ScalarDtype::F64
        ),
        text_signature = "(distances, *, target='layout2d', strategy=None, backend=None, epsilon=1e-4, max_iterations=100, pcg_tolerance=1e-6, pcg_max_iterations=None, min_norm=1e-4, dtype='float64')"
    )]
    fn from_distance_matrix(
        py: Python<'_>,
        distances: &Bound<'_, PyAny>,
        target: SolverTargetKind,
        strategy: Option<&str>,
        backend: Option<&str>,
        epsilon: StrictF64,
        max_iterations: StrictUsize,
        pcg_tolerance: StrictF64,
        pcg_max_iterations: Option<StrictUsize>,
        min_norm: StrictF64,
        dtype: ScalarDtype,
    ) -> PyResult<Self> {
        let strat = resolve_stress_strategy_pair(target, strategy, backend)?;
        let cfg = StressMajorizationConfig::new()
            .with_epsilon(epsilon.0)
            .with_max_iterations(max_iterations.0)
            .with_pcg_tolerance(pcg_tolerance.0)
            .with_pcg_max_iterations(pcg_max_iterations.map(|v| v.0))
            .with_min_norm(min_norm.0);
        cfg.validate().map_err(core_error_to_py)?;
        let (dists_flat, n) = extract_distance_matrix_2d(distances)?;
        let inner = py
            .detach(|| match (target, strat, dtype) {
                (SolverTargetKind::Layout2d, StressStrategyChoice::EuclideanPcg, ScalarDtype::F64) => {
                    StressMajorizationSolver::<Layout2d<f64>, EuclideanPcg>::from_distance_matrix_with_strategy(
                        n, &dists_flat, cfg, EuclideanPcg::new(),
                    ).map(StressMajorizationInner::L2PcgF64)
                }
                (SolverTargetKind::Layout2d, StressStrategyChoice::EuclideanPcg, ScalarDtype::F32) => {
                    StressMajorizationSolver::<Layout2d<f32>, EuclideanPcg>::from_distance_matrix_with_strategy(
                        n, &dists_flat, cfg, EuclideanPcg::new(),
                    ).map(StressMajorizationInner::L2PcgF32)
                }
                (SolverTargetKind::Layout3d, StressStrategyChoice::EuclideanPcg, ScalarDtype::F64) => {
                    StressMajorizationSolver::<Layout3d<f64>, EuclideanPcg>::from_distance_matrix_with_strategy(
                        n, &dists_flat, cfg, EuclideanPcg::new(),
                    ).map(StressMajorizationInner::L3PcgF64)
                }
                (SolverTargetKind::Layout3d, StressStrategyChoice::EuclideanPcg, ScalarDtype::F32) => {
                    StressMajorizationSolver::<Layout3d<f32>, EuclideanPcg>::from_distance_matrix_with_strategy(
                        n, &dists_flat, cfg, EuclideanPcg::new(),
                    ).map(StressMajorizationInner::L3PcgF32)
                }
                (SolverTargetKind::LayoutNd, StressStrategyChoice::EuclideanPcg, ScalarDtype::F64) => {
                    StressMajorizationSolver::<LayoutNd<f64>, EuclideanPcg>::from_distance_matrix_with_strategy(
                        n, &dists_flat, cfg, EuclideanPcg::new(),
                    ).map(StressMajorizationInner::NdPcgF64)
                }
                (SolverTargetKind::LayoutNd, StressStrategyChoice::EuclideanPcg, ScalarDtype::F32) => {
                    StressMajorizationSolver::<LayoutNd<f32>, EuclideanPcg>::from_distance_matrix_with_strategy(
                        n, &dists_flat, cfg, EuclideanPcg::new(),
                    ).map(StressMajorizationInner::NdPcgF32)
                }
                (SolverTargetKind::Layout2d, StressStrategyChoice::GeodesicGradient, ScalarDtype::F64) => {
                    StressMajorizationSolver::<Layout2d<f64>, GeodesicGradient>::from_distance_matrix_with_strategy(
                        n, &dists_flat, cfg, GeodesicGradient::new(),
                    ).map(StressMajorizationInner::L2GeoF64)
                }
                (SolverTargetKind::Layout2d, StressStrategyChoice::GeodesicGradient, ScalarDtype::F32) => {
                    StressMajorizationSolver::<Layout2d<f32>, GeodesicGradient>::from_distance_matrix_with_strategy(
                        n, &dists_flat, cfg, GeodesicGradient::new(),
                    ).map(StressMajorizationInner::L2GeoF32)
                }
                (SolverTargetKind::Layout3d, StressStrategyChoice::GeodesicGradient, ScalarDtype::F64) => {
                    StressMajorizationSolver::<Layout3d<f64>, GeodesicGradient>::from_distance_matrix_with_strategy(
                        n, &dists_flat, cfg, GeodesicGradient::new(),
                    ).map(StressMajorizationInner::L3GeoF64)
                }
                (SolverTargetKind::Layout3d, StressStrategyChoice::GeodesicGradient, ScalarDtype::F32) => {
                    StressMajorizationSolver::<Layout3d<f32>, GeodesicGradient>::from_distance_matrix_with_strategy(
                        n, &dists_flat, cfg, GeodesicGradient::new(),
                    ).map(StressMajorizationInner::L3GeoF32)
                }
                (SolverTargetKind::LayoutNd, StressStrategyChoice::GeodesicGradient, ScalarDtype::F64) => {
                    StressMajorizationSolver::<LayoutNd<f64>, GeodesicGradient>::from_distance_matrix_with_strategy(
                        n, &dists_flat, cfg, GeodesicGradient::new(),
                    ).map(StressMajorizationInner::NdGeoF64)
                }
                (SolverTargetKind::LayoutNd, StressStrategyChoice::GeodesicGradient, ScalarDtype::F32) => {
                    StressMajorizationSolver::<LayoutNd<f32>, GeodesicGradient>::from_distance_matrix_with_strategy(
                        n, &dists_flat, cfg, GeodesicGradient::new(),
                    ).map(StressMajorizationInner::NdGeoF32)
                }
                (SolverTargetKind::Manifold2d, StressStrategyChoice::GeodesicGradient, ScalarDtype::F64) => {
                    StressMajorizationSolver::<ManifoldLayout2d<f64>, GeodesicGradient>::from_distance_matrix_with_strategy(
                        n, &dists_flat, cfg, GeodesicGradient::new(),
                    ).map(StressMajorizationInner::M2GeoF64)
                }
                (SolverTargetKind::Manifold2d, StressStrategyChoice::GeodesicGradient, ScalarDtype::F32) => {
                    StressMajorizationSolver::<ManifoldLayout2d<f32>, GeodesicGradient>::from_distance_matrix_with_strategy(
                        n, &dists_flat, cfg, GeodesicGradient::new(),
                    ).map(StressMajorizationInner::M2GeoF32)
                }
                _ => unreachable!(),
            })
            .map_err(core_error_to_py)?;
        Ok(Self {
            inner,
            locked_manifold: None,
        })
    }

    #[pyo3(signature = (layout), text_signature = "($self, layout)")]
    fn step(&mut self, py: Python<'_>, layout: &Bound<'_, PyAny>) -> PyResult<&'static str> {
        execute_target_action(
            &mut self.inner,
            &mut self.locked_manifold,
            py,
            layout,
            SolverAction::Step,
        )
    }

    #[pyo3(signature = (layout, steps), text_signature = "($self, layout, steps)")]
    fn run_steps(
        &mut self,
        py: Python<'_>,
        layout: &Bound<'_, PyAny>,
        steps: StrictUsize,
    ) -> PyResult<&'static str> {
        execute_target_action(
            &mut self.inner,
            &mut self.locked_manifold,
            py,
            layout,
            SolverAction::RunSteps(steps.0),
        )
    }

    #[pyo3(signature = (layout), text_signature = "($self, layout)")]
    fn run(&mut self, py: Python<'_>, layout: &Bound<'_, PyAny>) -> PyResult<&'static str> {
        execute_target_action(
            &mut self.inner,
            &mut self.locked_manifold,
            py,
            layout,
            SolverAction::Run,
        )
    }

    #[pyo3(signature = (layout), text_signature = "($self, layout)")]
    fn compute_stress(&mut self, py: Python<'_>, layout: &Bound<'_, PyAny>) -> PyResult<f64> {
        validate_target_space(&self.inner, &mut self.locked_manifold, py, layout)?;
        let mut extracted =
            extract_target_layout(layout, self.inner.target_kind(), self.inner.dtype())?;
        py.detach(
            || match_stress_target!(&self.inner, &mut extracted, (s, l) => s.compute_stress(l)),
        )
        .map_err(core_error_to_py)
    }

    #[getter]
    fn target(&self) -> &'static str {
        self.inner.target_kind().as_str()
    }

    #[getter]
    fn last_stress(&self) -> Option<f64> {
        match_stress_inner!(&self.inner, s => s.last_stress())
    }

    #[getter]
    fn status(&self) -> &'static str {
        solver_status_str(self.inner.solver_status())
    }

    #[getter]
    fn iteration(&self) -> usize {
        self.inner.solver_iteration()
    }

    #[getter]
    fn total_iterations(&self) -> Option<usize> {
        self.inner.solver_total_iterations()
    }

    #[getter]
    fn is_finished(&self) -> bool {
        self.inner.solver_status().is_finished()
    }

    #[getter]
    fn num_nodes(&self) -> usize {
        match_stress_inner!(&self.inner, s => s.num_nodes())
    }

    #[getter]
    fn dtype(&self) -> &'static str {
        self.inner.dtype().as_str()
    }

    fn __repr__(&self) -> String {
        format!(
            "StressMajorizationSolver(num_nodes={}, target='{}', status='{}', iteration={}/{}, dtype='{}')",
            self.num_nodes(),
            self.target(),
            self.status(),
            self.iteration(),
            format_optional_usize(self.total_iterations()),
            self.dtype()
        )
    }
}
