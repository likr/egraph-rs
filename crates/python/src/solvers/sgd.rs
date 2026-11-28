//! Python binding for `SgdSolver` across `Layout2d | Layout3d | LayoutNd | ManifoldLayout2d`.

use egraph::core::ManifoldKind;
use egraph::layout::SparseBarrierStats;
use egraph::neighborhood::DogVaradhanStats;
use pyo3::prelude::*;

use crate::csr_graph::PyCsrGraph;
use crate::ffi_convert::{ScalarDtype, StrictF64, StrictU64, StrictUsize, solver_status_str};
use crate::neighborhood::PyDogVaradhanLandmarks;
use crate::solver_common::format_optional_usize;

use super::layout_dispatch::{
    DualTargetSolver, SolverAction, SolverTargetKind, execute_target_action,
};
use super::sgd_barrier::{build_sparse_barrier_dog, build_sparse_barrier_from_landmarks};
use super::sgd_baselines::build_ortmann_sparse_stress_solver;
use super::sgd_core::{
    SgdInner, build_from_pairs, build_from_pairs_with_hinges, build_full, build_omega,
    build_random_pair, build_streaming_negatives,
};
use super::sgd_dispatch::{
    build_from_packed_symmetric, execute_reset_schedule, execute_schedule_window,
};
use super::sgd_stats::{PyDogVaradhanStats, PySparseBarrierStats};

/// Cooperative SoA Stochastic Gradient Descent (SGD) layout solver supporting 3-tier Sparse Barrier
/// SGD, streaming non-edge negative sampling, Ortmann sparse stress, and Riemannian stepping.
#[pyclass(name = "SgdSolver", module = "egraph._core")]
pub struct PySgdSolver {
    inner: SgdInner,
    locked_manifold: Option<ManifoldKind>,
    barrier_stats: Option<SparseBarrierStats>,
    dog_stats: Option<DogVaradhanStats>,
}

impl PySgdSolver {
    pub(super) fn from_inner(inner: SgdInner) -> Self {
        Self {
            inner,
            locked_manifold: None,
            barrier_stats: None,
            dog_stats: None,
        }
    }

    pub(super) fn with_stats(
        inner: SgdInner,
        barrier_stats: Option<SparseBarrierStats>,
        dog_stats: Option<DogVaradhanStats>,
    ) -> Self {
        Self {
            inner,
            locked_manifold: None,
            barrier_stats,
            dog_stats,
        }
    }
}

#[pymethods]
impl PySgdSolver {
    /// Construct a full-APSP stress `SgdSolver` directly from a `CsrGraph`.
    #[new]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(
        signature = (graph, *, target = SolverTargetKind::Layout2d, iterations = StrictUsize(60), eps = StrictF64(0.01), schedule = "exponential", step_scale = StrictF64(1.0), max_mu = None, seed = StrictU64(42), dtype = ScalarDtype::F64),
        text_signature = "(graph, *, target='layout2d', iterations=60, eps=0.01, schedule='exponential', step_scale=1.0, max_mu=None, seed=42, dtype='float64')"
    )]
    fn new(
        py: Python<'_>,
        graph: &PyCsrGraph,
        target: SolverTargetKind,
        iterations: StrictUsize,
        eps: StrictF64,
        schedule: &str,
        step_scale: StrictF64,
        max_mu: Option<StrictF64>,
        seed: StrictU64,
        dtype: ScalarDtype,
    ) -> PyResult<Self> {
        build_full(
            py, graph, target, iterations, eps, schedule, step_scale, max_mu, seed, dtype,
        )
    }

    /// Construct an `SgdSolver` from flat SoA two-sided spring arrays.
    #[staticmethod]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(
        signature = (num_nodes, src, tgt, dij, *, dji = None, wij = None, wji = None, target = SolverTargetKind::Layout2d, iterations = StrictUsize(60), eps = StrictF64(0.01), schedule = "exponential", step_scale = StrictF64(1.0), max_mu = None, seed = StrictU64(42), dtype = ScalarDtype::F64),
        text_signature = "(num_nodes, src, tgt, dij, *, dji=None, wij=None, wji=None, target='layout2d', iterations=60, eps=0.01, schedule='exponential', step_scale=1.0, max_mu=None, seed=42, dtype='float64')"
    )]
    fn from_pairs(
        py: Python<'_>,
        num_nodes: StrictUsize,
        src: &Bound<'_, PyAny>,
        tgt: &Bound<'_, PyAny>,
        dij: &Bound<'_, PyAny>,
        dji: Option<&Bound<'_, PyAny>>,
        wij: Option<&Bound<'_, PyAny>>,
        wji: Option<&Bound<'_, PyAny>>,
        target: SolverTargetKind,
        iterations: StrictUsize,
        eps: StrictF64,
        schedule: &str,
        step_scale: StrictF64,
        max_mu: Option<StrictF64>,
        seed: StrictU64,
        dtype: ScalarDtype,
    ) -> PyResult<Self> {
        build_from_pairs(
            py, num_nodes, src, tgt, dij, dji, wij, wji, target, iterations, eps, schedule,
            step_scale, max_mu, seed, dtype,
        )
    }

    /// Construct an `SgdSolver` combining flat SoA spring pairs and one-sided hinge repulsion pairs.
    #[staticmethod]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(
        signature = (num_nodes, spring_src, spring_tgt, spring_dij, hinge_src, hinge_tgt, hinge_dij, *, spring_dji = None, spring_wij = None, spring_wji = None, hinge_dji = None, hinge_wij = None, hinge_wji = None, target = SolverTargetKind::Layout2d, iterations = StrictUsize(60), eps = StrictF64(0.01), schedule = "exponential", step_scale = StrictF64(1.0), max_mu = None, seed = StrictU64(42), dtype = ScalarDtype::F64),
        text_signature = "(num_nodes, spring_src, spring_tgt, spring_dij, hinge_src, hinge_tgt, hinge_dij, *, spring_dji=None, spring_wij=None, spring_wji=None, hinge_dji=None, hinge_wij=None, hinge_wji=None, target='layout2d', iterations=60, eps=0.01, schedule='exponential', step_scale=1.0, max_mu=None, seed=42, dtype='float64')"
    )]
    fn from_pairs_with_hinges(
        py: Python<'_>,
        num_nodes: StrictUsize,
        spring_src: &Bound<'_, PyAny>,
        spring_tgt: &Bound<'_, PyAny>,
        spring_dij: &Bound<'_, PyAny>,
        hinge_src: &Bound<'_, PyAny>,
        hinge_tgt: &Bound<'_, PyAny>,
        hinge_dij: &Bound<'_, PyAny>,
        spring_dji: Option<&Bound<'_, PyAny>>,
        spring_wij: Option<&Bound<'_, PyAny>>,
        spring_wji: Option<&Bound<'_, PyAny>>,
        hinge_dji: Option<&Bound<'_, PyAny>>,
        hinge_wij: Option<&Bound<'_, PyAny>>,
        hinge_wji: Option<&Bound<'_, PyAny>>,
        target: SolverTargetKind,
        iterations: StrictUsize,
        eps: StrictF64,
        schedule: &str,
        step_scale: StrictF64,
        max_mu: Option<StrictF64>,
        seed: StrictU64,
        dtype: ScalarDtype,
    ) -> PyResult<Self> {
        build_from_pairs_with_hinges(
            py, num_nodes, spring_src, spring_tgt, spring_dij, spring_dji, spring_wij, spring_wji,
            hinge_src, hinge_tgt, hinge_dij, hinge_dji, hinge_wij, hinge_wji, target, iterations,
            eps, schedule, step_scale, max_mu, seed, dtype,
        )
    }

    /// Construct an `SgdSolver` from pre-packed symmetric spring/hinge arrays via `SgdPairBuffer`.
    #[staticmethod]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(
        signature = (num_nodes, src, tgt, dists, weights, is_hinge, *, target = SolverTargetKind::Layout2d, iterations = StrictUsize(60), eps = StrictF64(0.01), schedule = "exponential", step_scale = StrictF64(1.0), max_mu = None, seed = StrictU64(42), dtype = ScalarDtype::F64),
        text_signature = "(num_nodes, src, tgt, dists, weights, is_hinge, *, target='layout2d', iterations=60, eps=0.01, schedule='exponential', step_scale=1.0, max_mu=None, seed=42, dtype='float64')"
    )]
    fn from_packed_symmetric(
        py: Python<'_>,
        num_nodes: StrictUsize,
        src: &Bound<'_, PyAny>,
        tgt: &Bound<'_, PyAny>,
        dists: &Bound<'_, PyAny>,
        weights: &Bound<'_, PyAny>,
        is_hinge: &Bound<'_, PyAny>,
        target: SolverTargetKind,
        iterations: StrictUsize,
        eps: StrictF64,
        schedule: &str,
        step_scale: StrictF64,
        max_mu: Option<StrictF64>,
        seed: StrictU64,
        dtype: ScalarDtype,
    ) -> PyResult<Self> {
        build_from_packed_symmetric(
            py, num_nodes, src, tgt, dists, weights, is_hinge, target, iterations, eps, schedule,
            step_scale, max_mu, seed, dtype,
        )
    }

    /// Construct a full-APSP stress `SgdSolver` directly from a `CsrGraph`.
    #[staticmethod]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(
        signature = (graph, *, target = SolverTargetKind::Layout2d, iterations = StrictUsize(60), eps = StrictF64(0.01), schedule = "exponential", step_scale = StrictF64(1.0), max_mu = None, seed = StrictU64(42), dtype = ScalarDtype::F64),
        text_signature = "(graph, *, target='layout2d', iterations=60, eps=0.01, schedule='exponential', step_scale=1.0, max_mu=None, seed=42, dtype='float64')"
    )]
    fn full(
        py: Python<'_>,
        graph: &PyCsrGraph,
        target: SolverTargetKind,
        iterations: StrictUsize,
        eps: StrictF64,
        schedule: &str,
        step_scale: StrictF64,
        max_mu: Option<StrictF64>,
        seed: StrictU64,
        dtype: ScalarDtype,
    ) -> PyResult<Self> {
        build_full(
            py, graph, target, iterations, eps, schedule, step_scale, max_mu, seed, dtype,
        )
    }

    /// Construct a 3-tier Sparse Barrier `SgdSolver` from precomputed `DogVaradhanLandmarks`.
    #[staticmethod]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(
        signature = (graph, landmarks, pivots = None, *, target = SolverTargetKind::Layout2d, max_2hop_per_node = StrictUsize(32), hinge_repulsion = true, delta_rel = StrictF64(0.25), barrier_sym = "sym_max", omega_1 = StrictF64(1.0), omega_2 = StrictF64(1.0), gamma_non2 = StrictF64(1.0), ht_repulsion = true, min_ratio = StrictF64(0.5), eps_dist = StrictF64(1e-6), neg_per_node = None, num_pivots = StrictUsize(50), core_pivots = true, degree_weight_beta = StrictF64(0.0), iterations = StrictUsize(60), eps = StrictF64(0.01), schedule = "exponential", step_scale = StrictF64(1.0), max_mu = None, seed = StrictU64(42), dtype = ScalarDtype::F64),
        text_signature = "(graph, landmarks, pivots=None, *, target='layout2d', max_2hop_per_node=32, hinge_repulsion=True, delta_rel=0.25, barrier_sym='sym_max', omega_1=1.0, omega_2=1.0, gamma_non2=1.0, ht_repulsion=True, min_ratio=0.5, eps_dist=1e-6, neg_per_node=None, num_pivots=50, core_pivots=True, degree_weight_beta=0.0, iterations=60, eps=0.01, schedule='exponential', step_scale=1.0, max_mu=None, seed=42, dtype='float64')"
    )]
    fn sparse_barrier_from_landmarks(
        py: Python<'_>,
        graph: &PyCsrGraph,
        landmarks: &PyDogVaradhanLandmarks,
        pivots: Option<&Bound<'_, PyAny>>,
        target: SolverTargetKind,
        max_2hop_per_node: StrictUsize,
        hinge_repulsion: bool,
        delta_rel: StrictF64,
        barrier_sym: &str,
        omega_1: StrictF64,
        omega_2: StrictF64,
        gamma_non2: StrictF64,
        ht_repulsion: bool,
        min_ratio: StrictF64,
        eps_dist: StrictF64,
        neg_per_node: Option<StrictUsize>,
        num_pivots: StrictUsize,
        core_pivots: bool,
        degree_weight_beta: StrictF64,
        iterations: StrictUsize,
        eps: StrictF64,
        schedule: &str,
        step_scale: StrictF64,
        max_mu: Option<StrictF64>,
        seed: StrictU64,
        dtype: ScalarDtype,
    ) -> PyResult<Self> {
        build_sparse_barrier_from_landmarks(
            py,
            graph,
            landmarks,
            pivots,
            target,
            max_2hop_per_node,
            hinge_repulsion,
            delta_rel,
            barrier_sym,
            omega_1,
            omega_2,
            gamma_non2,
            ht_repulsion,
            min_ratio,
            eps_dist,
            neg_per_node,
            num_pivots,
            core_pivots,
            degree_weight_beta,
            iterations,
            eps,
            schedule,
            step_scale,
            max_mu,
            seed,
            dtype,
        )
    }

    /// Construct a 3-tier Sparse Barrier `SgdSolver` with integrated Regularized Chebyshev DoG landmarks.
    #[staticmethod]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(
        signature = (graph, *, target = SolverTargetKind::Layout2d, degree = None, num_landmarks = StrictUsize(64), degree_normalize = true, p_exp = StrictF64(0.5), eps_psd = StrictF64(1e-4), t_coarse = StrictF64(48.0), t1 = None, t2 = None, alpha = None, beta_floor = None, min_degree = StrictUsize(2), jl_dim = StrictUsize(64), max_2hop_per_node = StrictUsize(32), hinge_repulsion = true, delta_rel = StrictF64(0.25), barrier_sym = "sym_max", omega_1 = StrictF64(1.0), omega_2 = StrictF64(1.0), gamma_non2 = StrictF64(1.0), ht_repulsion = true, min_ratio = StrictF64(0.5), eps_dist = StrictF64(1e-6), neg_per_node = None, num_pivots = StrictUsize(50), core_pivots = true, degree_weight_beta = StrictF64(0.0), iterations = StrictUsize(60), eps = StrictF64(0.01), schedule = "exponential", step_scale = StrictF64(1.0), max_mu = None, seed = StrictU64(42), dtype = ScalarDtype::F64),
        text_signature = "(graph, *, target='layout2d', degree=None, num_landmarks=64, degree_normalize=True, p_exp=0.5, eps_psd=1e-4, t_coarse=48.0, t1=None, t2=None, alpha=None, beta_floor=None, min_degree=2, jl_dim=64, max_2hop_per_node=32, hinge_repulsion=True, delta_rel=0.25, barrier_sym='sym_max', omega_1=1.0, omega_2=1.0, gamma_non2=1.0, ht_repulsion=True, min_ratio=0.5, eps_dist=1e-6, neg_per_node=None, num_pivots=50, core_pivots=True, degree_weight_beta=0.0, iterations=60, eps=0.01, schedule='exponential', step_scale=1.0, max_mu=None, seed=42, dtype='float64')"
    )]
    fn sparse_barrier_dog(
        py: Python<'_>,
        graph: &PyCsrGraph,
        target: SolverTargetKind,
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
        max_2hop_per_node: StrictUsize,
        hinge_repulsion: bool,
        delta_rel: StrictF64,
        barrier_sym: &str,
        omega_1: StrictF64,
        omega_2: StrictF64,
        gamma_non2: StrictF64,
        ht_repulsion: bool,
        min_ratio: StrictF64,
        eps_dist: StrictF64,
        neg_per_node: Option<StrictUsize>,
        num_pivots: StrictUsize,
        core_pivots: bool,
        degree_weight_beta: StrictF64,
        iterations: StrictUsize,
        eps: StrictF64,
        schedule: &str,
        step_scale: StrictF64,
        max_mu: Option<StrictF64>,
        seed: StrictU64,
        dtype: ScalarDtype,
    ) -> PyResult<Self> {
        build_sparse_barrier_dog(
            py,
            graph,
            target,
            degree,
            num_landmarks,
            degree_normalize,
            p_exp,
            eps_psd,
            t_coarse,
            t1,
            t2,
            alpha,
            beta_floor,
            min_degree,
            jl_dim,
            max_2hop_per_node,
            hinge_repulsion,
            delta_rel,
            barrier_sym,
            omega_1,
            omega_2,
            gamma_non2,
            ht_repulsion,
            min_ratio,
            eps_dist,
            neg_per_node,
            num_pivots,
            core_pivots,
            degree_weight_beta,
            iterations,
            eps,
            schedule,
            step_scale,
            max_mu,
            seed,
            dtype,
        )
    }

    /// Construct a random-pair sparse stress `SgdSolver` directly from a `CsrGraph`.
    #[staticmethod]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(
        signature = (graph, pairs_per_node = StrictUsize(30), *, target = SolverTargetKind::Layout2d, iterations = StrictUsize(60), eps = StrictF64(0.01), schedule = "exponential", step_scale = StrictF64(1.0), max_mu = None, seed = StrictU64(42), dtype = ScalarDtype::F64),
        text_signature = "(graph, pairs_per_node=30, *, target='layout2d', iterations=60, eps=0.01, schedule='exponential', step_scale=1.0, max_mu=None, seed=42, dtype='float64')"
    )]
    fn random_pair(
        py: Python<'_>,
        graph: &PyCsrGraph,
        pairs_per_node: StrictUsize,
        target: SolverTargetKind,
        iterations: StrictUsize,
        eps: StrictF64,
        schedule: &str,
        step_scale: StrictF64,
        max_mu: Option<StrictF64>,
        seed: StrictU64,
        dtype: ScalarDtype,
    ) -> PyResult<Self> {
        build_random_pair(
            py,
            graph,
            pairs_per_node,
            target,
            iterations,
            eps,
            schedule,
            step_scale,
            max_mu,
            seed,
            dtype,
        )
    }

    /// Construct an `omega` sparse stress `SgdSolver` combining graph edges with k-NN pairs in `embedding`.
    #[staticmethod]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(
        signature = (graph, embedding, k = StrictUsize(30), min_dist = StrictF64(1e-3), *, target = SolverTargetKind::Layout2d, iterations = StrictUsize(60), eps = StrictF64(0.01), schedule = "exponential", step_scale = StrictF64(1.0), max_mu = None, seed = StrictU64(42), dtype = ScalarDtype::F64),
        text_signature = "(graph, embedding, k=30, min_dist=0.001, *, target='layout2d', iterations=60, eps=0.01, schedule='exponential', step_scale=1.0, max_mu=None, seed=42, dtype='float64')"
    )]
    fn omega(
        py: Python<'_>,
        graph: &PyCsrGraph,
        embedding: &Bound<'_, PyAny>,
        k: StrictUsize,
        min_dist: StrictF64,
        target: SolverTargetKind,
        iterations: StrictUsize,
        eps: StrictF64,
        schedule: &str,
        step_scale: StrictF64,
        max_mu: Option<StrictF64>,
        seed: StrictU64,
        dtype: ScalarDtype,
    ) -> PyResult<Self> {
        build_omega(
            py, graph, embedding, k, min_dist, target, iterations, eps, schedule, step_scale,
            max_mu, seed, dtype,
        )
    }

    /// Construct a minimal-RAM `SgdSolver` storing only 1-hop edges and streaming non-edge negatives.
    #[staticmethod]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(
        signature = (graph, *, target = SolverTargetKind::Layout2d, negatives_per_node = StrictUsize(5), target_distance = StrictF64(2.0), weight = StrictF64(0.25), hinge_only = true, iterations = StrictUsize(60), eps = StrictF64(0.01), schedule = "exponential", step_scale = StrictF64(1.0), max_mu = None, seed = StrictU64(42), dtype = ScalarDtype::F64),
        text_signature = "(graph, *, target='layout2d', negatives_per_node=5, target_distance=2.0, weight=0.25, hinge_only=True, iterations=60, eps=0.01, schedule='exponential', step_scale=1.0, max_mu=None, seed=42, dtype='float64')"
    )]
    fn streaming_negatives(
        py: Python<'_>,
        graph: &PyCsrGraph,
        target: SolverTargetKind,
        negatives_per_node: StrictUsize,
        target_distance: StrictF64,
        weight: StrictF64,
        hinge_only: bool,
        iterations: StrictUsize,
        eps: StrictF64,
        schedule: &str,
        step_scale: StrictF64,
        max_mu: Option<StrictF64>,
        seed: StrictU64,
        dtype: ScalarDtype,
    ) -> PyResult<Self> {
        build_streaming_negatives(
            py,
            graph,
            target,
            negatives_per_node,
            target_distance,
            weight,
            hinge_only,
            iterations,
            eps,
            schedule,
            step_scale,
            max_mu,
            seed,
            dtype,
        )
    }

    /// Construct an Ortmann et al. (2016) pivot-seeded sparse stress `SgdSolver` from a `CsrGraph`.
    #[staticmethod]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(
        signature = (graph, *, target = SolverTargetKind::Layout2d, num_pivots = StrictUsize(100), pivot_strategy = "kmeans_sp", iterations = StrictUsize(200), eta_max = None, eta_min = None, eps = StrictF64(0.1), seed = StrictU64(42), dtype = ScalarDtype::F64),
        text_signature = "(graph, *, target='layout2d', num_pivots=100, pivot_strategy='kmeans_sp', iterations=200, eta_max=None, eta_min=None, eps=0.1, seed=42, dtype='float64')"
    )]
    fn ortmann_sparse_stress(
        py: Python<'_>,
        graph: &PyCsrGraph,
        target: SolverTargetKind,
        num_pivots: StrictUsize,
        pivot_strategy: &str,
        iterations: StrictUsize,
        eta_max: Option<StrictF64>,
        eta_min: Option<StrictF64>,
        eps: StrictF64,
        seed: StrictU64,
        dtype: ScalarDtype,
    ) -> PyResult<Self> {
        build_ortmann_sparse_stress_solver(
            py,
            graph,
            target,
            num_pivots,
            pivot_strategy,
            iterations,
            eta_max,
            eta_min,
            eps,
            seed,
            dtype,
        )
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

    #[pyo3(
        signature = (*, iterations = StrictUsize(60), eps = StrictF64(0.01), schedule = "exponential", step_scale = StrictF64(1.0), max_mu = None, seed = StrictU64(42)),
        text_signature = "($self, *, iterations=60, eps=0.01, schedule='exponential', step_scale=1.0, max_mu=None, seed=42)"
    )]
    fn reset_schedule(
        &mut self,
        iterations: StrictUsize,
        eps: StrictF64,
        schedule: &str,
        step_scale: StrictF64,
        max_mu: Option<StrictF64>,
        seed: StrictU64,
    ) -> PyResult<()> {
        execute_reset_schedule(
            &mut self.inner,
            iterations,
            eps,
            schedule,
            step_scale,
            max_mu,
            seed,
        )
    }

    #[pyo3(signature = (layout, start_iter, count), text_signature = "($self, layout, start_iter, count)")]
    fn run_schedule_window(
        &mut self,
        py: Python<'_>,
        layout: &Bound<'_, PyAny>,
        start_iter: StrictUsize,
        count: StrictUsize,
    ) -> PyResult<&'static str> {
        execute_schedule_window(
            &mut self.inner,
            &mut self.locked_manifold,
            py,
            layout,
            start_iter,
            count,
        )
    }

    /// Return Sparse Barrier construction diagnostics, or `None`.
    fn stats(&self) -> Option<PySparseBarrierStats> {
        self.barrier_stats
            .clone()
            .map(PySparseBarrierStats::from_core)
    }

    /// Return regularized DoG Varadhan landmark diagnostics, or `None`.
    fn dog_stats(&self) -> Option<PyDogVaradhanStats> {
        self.dog_stats.clone().map(PyDogVaradhanStats::from_core)
    }

    #[getter]
    fn target(&self) -> &'static str {
        self.inner.target_kind().as_str()
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
    fn supports_manifold(&self) -> bool {
        self.inner.supports_manifold()
    }

    #[getter]
    fn num_nodes(&self) -> usize {
        self.inner.num_nodes()
    }

    #[getter]
    fn num_springs(&self) -> usize {
        self.inner.num_springs()
    }

    #[getter]
    fn num_hinges(&self) -> usize {
        self.inner.num_hinges()
    }

    #[getter]
    fn current_eta(&self) -> f64 {
        self.inner.current_eta()
    }

    #[getter]
    fn dtype(&self) -> &'static str {
        self.inner.dtype().as_str()
    }

    fn __repr__(&self) -> String {
        format!(
            "SgdSolver(num_nodes={}, target='{}', springs={}, hinges={}, status='{}', iteration={}/{}, dtype='{}')",
            self.num_nodes(),
            self.target(),
            self.num_springs(),
            self.num_hinges(),
            self.status(),
            self.iteration(),
            format_optional_usize(self.total_iterations()),
            self.dtype()
        )
    }
}
