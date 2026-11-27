//! Thin Python binding dispatch facade for `SgdSolver`.

use egraph::{layout as egraph_layout, neighborhood as egraph_neighborhood};
use egraph_layout::SparseBarrierStats;
use egraph_neighborhood::DogVaradhanStats;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::csr_graph::PyCsrGraph;
use crate::ffi_convert::{ScalarDtype, StrictF64, StrictU64, StrictUsize, solver_status_str};
use crate::layout_buffers::PyLayout2d;
use crate::neighborhood::PyDogVaradhanLandmarks;
use crate::solver_common::{DualIterativeSolver, SolverAction};

use super::coarse_backbone::build_pacmap_graph_with_backbone;
use super::layout_dispatch::execute_any_layout_action;
use super::sgd_barrier::{
    build_sparse_barrier_dog, build_sparse_barrier_from_landmarks, dog_stats_to_py_dict,
    sparse_barrier_stats_to_py_dict,
};
use super::sgd_baselines::{
    build_drgraph_solver, build_graph_cne_solver, build_largevis_solver,
    build_ortmann_sparse_stress_solver,
};
use super::sgd_core::{
    SgdInner, build_from_pairs, build_from_pairs_with_hinges, build_full, build_omega,
    build_random_pair, build_streaming_negatives,
};
use super::sgd_dispatch::{
    build_from_packed_symmetric, execute_reset_schedule, execute_schedule_window,
};
use super::sgd_pacmap::{build_pacmap_graph_solver, build_sgd2_solver, build_snap_t_fdp_solver};

/// Cooperative SoA Stochastic Gradient Descent (SGD) layout solver supporting 3-tier Sparse Barrier
/// SGD, streaming non-edge negative sampling, Euclidean `Layout2d` / `LayoutNd`, and non-Euclidean
/// `ManifoldLayout2d`.
#[pyclass(name = "SgdSolver", module = "egraph._core")]
pub struct PySgdSolver {
    inner: SgdInner,
    barrier_stats: Option<SparseBarrierStats>,
    dog_stats: Option<DogVaradhanStats>,
}

impl PySgdSolver {
    pub(super) fn from_inner(inner: SgdInner) -> Self {
        Self {
            inner,
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
        signature = (graph, *, iterations = StrictUsize(60), eps = StrictF64(0.01), schedule = "exponential", step_scale = StrictF64(1.0), max_mu = None, seed = StrictU64(42), dtype = ScalarDtype::F64),
        text_signature = "(graph, *, iterations=60, eps=0.01, schedule='exponential', step_scale=1.0, max_mu=None, seed=42, dtype='float64')"
    )]
    fn new(
        py: Python<'_>,
        graph: &PyCsrGraph,
        iterations: StrictUsize,
        eps: StrictF64,
        schedule: &str,
        step_scale: StrictF64,
        max_mu: Option<StrictF64>,
        seed: StrictU64,
        dtype: ScalarDtype,
    ) -> PyResult<Self> {
        build_full(
            py, graph, iterations, eps, schedule, step_scale, max_mu, seed, dtype,
        )
    }

    /// Construct an `SgdSolver` from flat SoA two-sided spring arrays.
    #[staticmethod]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(
        signature = (num_nodes, src, tgt, dij, *, dji = None, wij = None, wji = None, iterations = StrictUsize(60), eps = StrictF64(0.01), schedule = "exponential", step_scale = StrictF64(1.0), max_mu = None, seed = StrictU64(42), dtype = ScalarDtype::F64),
        text_signature = "(num_nodes, src, tgt, dij, *, dji=None, wij=None, wji=None, iterations=60, eps=0.01, schedule='exponential', step_scale=1.0, max_mu=None, seed=42, dtype='float64')"
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
        iterations: StrictUsize,
        eps: StrictF64,
        schedule: &str,
        step_scale: StrictF64,
        max_mu: Option<StrictF64>,
        seed: StrictU64,
        dtype: ScalarDtype,
    ) -> PyResult<Self> {
        build_from_pairs(
            py, num_nodes, src, tgt, dij, dji, wij, wji, iterations, eps, schedule, step_scale,
            max_mu, seed, dtype,
        )
    }

    /// Construct an `SgdSolver` combining flat SoA spring pairs and one-sided hinge repulsion pairs.
    #[staticmethod]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(
        signature = (num_nodes, spring_src, spring_tgt, spring_dij, hinge_src, hinge_tgt, hinge_dij, *, spring_dji = None, spring_wij = None, spring_wji = None, hinge_dji = None, hinge_wij = None, hinge_wji = None, iterations = StrictUsize(60), eps = StrictF64(0.01), schedule = "exponential", step_scale = StrictF64(1.0), max_mu = None, seed = StrictU64(42), dtype = ScalarDtype::F64),
        text_signature = "(num_nodes, spring_src, spring_tgt, spring_dij, hinge_src, hinge_tgt, hinge_dij, *, spring_dji=None, spring_wij=None, spring_wji=None, hinge_dji=None, hinge_wij=None, hinge_wji=None, iterations=60, eps=0.01, schedule='exponential', step_scale=1.0, max_mu=None, seed=42, dtype='float64')"
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
            hinge_src, hinge_tgt, hinge_dij, hinge_dji, hinge_wij, hinge_wji, iterations, eps,
            schedule, step_scale, max_mu, seed, dtype,
        )
    }

    /// Construct an `SgdSolver` from pre-packed symmetric spring/hinge arrays.
    #[staticmethod]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(
        signature = (num_nodes, src, tgt, dists, weights, is_hinge, *, iterations = StrictUsize(60), eps = StrictF64(0.01), schedule = "exponential", step_scale = StrictF64(1.0), max_mu = None, seed = StrictU64(42), dtype = ScalarDtype::F64),
        text_signature = "(num_nodes, src, tgt, dists, weights, is_hinge, *, iterations=60, eps=0.01, schedule='exponential', step_scale=1.0, max_mu=None, seed=42, dtype='float64')"
    )]
    fn from_packed_symmetric(
        py: Python<'_>,
        num_nodes: StrictUsize,
        src: &Bound<'_, PyAny>,
        tgt: &Bound<'_, PyAny>,
        dists: &Bound<'_, PyAny>,
        weights: &Bound<'_, PyAny>,
        is_hinge: &Bound<'_, PyAny>,
        iterations: StrictUsize,
        eps: StrictF64,
        schedule: &str,
        step_scale: StrictF64,
        max_mu: Option<StrictF64>,
        seed: StrictU64,
        dtype: ScalarDtype,
    ) -> PyResult<Self> {
        build_from_packed_symmetric(
            py, num_nodes, src, tgt, dists, weights, is_hinge, iterations, eps, schedule,
            step_scale, max_mu, seed, dtype,
        )
    }

    /// Construct a full-APSP stress `SgdSolver` directly from a `CsrGraph`.
    #[staticmethod]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(
        signature = (graph, *, iterations = StrictUsize(60), eps = StrictF64(0.01), schedule = "exponential", step_scale = StrictF64(1.0), max_mu = None, seed = StrictU64(42), dtype = ScalarDtype::F64),
        text_signature = "(graph, *, iterations=60, eps=0.01, schedule='exponential', step_scale=1.0, max_mu=None, seed=42, dtype='float64')"
    )]
    fn full(
        py: Python<'_>,
        graph: &PyCsrGraph,
        iterations: StrictUsize,
        eps: StrictF64,
        schedule: &str,
        step_scale: StrictF64,
        max_mu: Option<StrictF64>,
        seed: StrictU64,
        dtype: ScalarDtype,
    ) -> PyResult<Self> {
        build_full(
            py, graph, iterations, eps, schedule, step_scale, max_mu, seed, dtype,
        )
    }

    /// Construct a 3-tier Sparse Barrier `SgdSolver` from precomputed `DogVaradhanLandmarks`.
    #[staticmethod]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(
        signature = (graph, landmarks, pivots = None, *, max_2hop_per_node = StrictUsize(32), hinge_repulsion = true, delta_rel = StrictF64(0.25), barrier_sym = "sym_max", omega_1 = StrictF64(1.0), omega_2 = StrictF64(1.0), gamma_non2 = StrictF64(1.0), ht_repulsion = true, min_ratio = StrictF64(0.5), eps_dist = StrictF64(1e-6), neg_per_node = None, num_pivots = StrictUsize(50), core_pivots = true, degree_weight_beta = StrictF64(0.0), iterations = StrictUsize(60), eps = StrictF64(0.01), schedule = "exponential", step_scale = StrictF64(1.0), max_mu = None, seed = StrictU64(42), dtype = ScalarDtype::F64),
        text_signature = "(graph, landmarks, pivots=None, *, max_2hop_per_node=32, hinge_repulsion=True, delta_rel=0.25, barrier_sym='sym_max', omega_1=1.0, omega_2=1.0, gamma_non2=1.0, ht_repulsion=True, min_ratio=0.5, eps_dist=1e-6, neg_per_node=None, num_pivots=50, core_pivots=True, degree_weight_beta=0.0, iterations=60, eps=0.01, schedule='exponential', step_scale=1.0, max_mu=None, seed=42, dtype='float64')"
    )]
    fn sparse_barrier_from_landmarks(
        py: Python<'_>,
        graph: &PyCsrGraph,
        landmarks: &PyDogVaradhanLandmarks,
        pivots: Option<&Bound<'_, PyAny>>,
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
        signature = (graph, *, degree = None, num_landmarks = StrictUsize(64), degree_normalize = true, p_exp = StrictF64(0.5), eps_psd = StrictF64(1e-4), t_coarse = StrictF64(48.0), t1 = None, t2 = None, alpha = None, beta_floor = None, min_degree = StrictUsize(2), jl_dim = StrictUsize(64), max_2hop_per_node = StrictUsize(32), hinge_repulsion = true, delta_rel = StrictF64(0.25), barrier_sym = "sym_max", omega_1 = StrictF64(1.0), omega_2 = StrictF64(1.0), gamma_non2 = StrictF64(1.0), ht_repulsion = true, min_ratio = StrictF64(0.5), eps_dist = StrictF64(1e-6), neg_per_node = None, num_pivots = StrictUsize(50), core_pivots = true, degree_weight_beta = StrictF64(0.0), iterations = StrictUsize(60), eps = StrictF64(0.01), schedule = "exponential", step_scale = StrictF64(1.0), max_mu = None, seed = StrictU64(42), dtype = ScalarDtype::F64),
        text_signature = "(graph, *, degree=None, num_landmarks=64, degree_normalize=True, p_exp=0.5, eps_psd=1e-4, t_coarse=48.0, t1=None, t2=None, alpha=None, beta_floor=None, min_degree=2, jl_dim=64, max_2hop_per_node=32, hinge_repulsion=True, delta_rel=0.25, barrier_sym='sym_max', omega_1=1.0, omega_2=1.0, gamma_non2=1.0, ht_repulsion=True, min_ratio=0.5, eps_dist=1e-6, neg_per_node=None, num_pivots=50, core_pivots=True, degree_weight_beta=0.0, iterations=60, eps=0.01, schedule='exponential', step_scale=1.0, max_mu=None, seed=42, dtype='float64')"
    )]
    fn sparse_barrier_dog(
        py: Python<'_>,
        graph: &PyCsrGraph,
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
        signature = (graph, pairs_per_node = StrictUsize(30), *, iterations = StrictUsize(60), eps = StrictF64(0.01), schedule = "exponential", step_scale = StrictF64(1.0), max_mu = None, seed = StrictU64(42), dtype = ScalarDtype::F64),
        text_signature = "(graph, pairs_per_node=30, *, iterations=60, eps=0.01, schedule='exponential', step_scale=1.0, max_mu=None, seed=42, dtype='float64')"
    )]
    fn random_pair(
        py: Python<'_>,
        graph: &PyCsrGraph,
        pairs_per_node: StrictUsize,
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
        signature = (graph, embedding, k = StrictUsize(30), min_dist = StrictF64(1e-3), *, iterations = StrictUsize(60), eps = StrictF64(0.01), schedule = "exponential", step_scale = StrictF64(1.0), max_mu = None, seed = StrictU64(42), dtype = ScalarDtype::F64),
        text_signature = "(graph, embedding, k=30, min_dist=0.001, *, iterations=60, eps=0.01, schedule='exponential', step_scale=1.0, max_mu=None, seed=42, dtype='float64')"
    )]
    fn omega(
        py: Python<'_>,
        graph: &PyCsrGraph,
        embedding: &Bound<'_, PyAny>,
        k: StrictUsize,
        min_dist: StrictF64,
        iterations: StrictUsize,
        eps: StrictF64,
        schedule: &str,
        step_scale: StrictF64,
        max_mu: Option<StrictF64>,
        seed: StrictU64,
        dtype: ScalarDtype,
    ) -> PyResult<Self> {
        build_omega(
            py, graph, embedding, k, min_dist, iterations, eps, schedule, step_scale, max_mu, seed,
            dtype,
        )
    }

    /// Construct a minimal-RAM `SgdSolver` storing only 1-hop edges and streaming non-edge negatives.
    #[staticmethod]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(
        signature = (graph, *, negatives_per_node = StrictUsize(5), target_distance = StrictF64(2.0), weight = StrictF64(0.25), hinge_only = true, iterations = StrictUsize(60), eps = StrictF64(0.01), schedule = "exponential", step_scale = StrictF64(1.0), max_mu = None, seed = StrictU64(42), dtype = ScalarDtype::F64),
        text_signature = "(graph, *, negatives_per_node=5, target_distance=2.0, weight=0.25, hinge_only=True, iterations=60, eps=0.01, schedule='exponential', step_scale=1.0, max_mu=None, seed=42, dtype='float64')"
    )]
    fn streaming_negatives(
        py: Python<'_>,
        graph: &PyCsrGraph,
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

    /// Construct a compact-memory edge-incident negative-sampling `SNAP-tFDP` `SgdSolver`.
    #[staticmethod]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(
        signature = (graph, *, iterations = StrictUsize(200), negatives_per_edge = StrictUsize(3), learning_rate = StrictF64(1.0), alpha = StrictF64(1.0), repulsion_scale = StrictF64(1.0), rho_start = StrictF64(1.0), rho_end = StrictF64(0.1), rho_power = StrictF64(1.0), degree_beta = StrictF64(0.0), bounded_attraction = true, seed = StrictU64(42), dtype = ScalarDtype::F64),
        text_signature = "(graph, *, iterations=200, negatives_per_edge=3, learning_rate=1.0, alpha=1.0, repulsion_scale=1.0, rho_start=1.0, rho_end=0.1, rho_power=1.0, degree_beta=0.0, bounded_attraction=True, seed=42, dtype='float64')"
    )]
    fn snap_t_fdp(
        py: Python<'_>,
        graph: &PyCsrGraph,
        iterations: StrictUsize,
        negatives_per_edge: StrictUsize,
        learning_rate: StrictF64,
        alpha: StrictF64,
        repulsion_scale: StrictF64,
        rho_start: StrictF64,
        rho_end: StrictF64,
        rho_power: StrictF64,
        degree_beta: StrictF64,
        bounded_attraction: bool,
        seed: StrictU64,
        dtype: ScalarDtype,
    ) -> PyResult<Self> {
        build_snap_t_fdp_solver(
            py,
            graph,
            iterations,
            negatives_per_edge,
            learning_rate,
            alpha,
            repulsion_scale,
            rho_start,
            rho_end,
            rho_power,
            degree_beta,
            bounded_attraction,
            seed,
            dtype,
        )
    }

    /// Construct a 3-tier `PaCMAP-Graph` (`NB` / `MN` / `FP`) `SgdSolver`.
    #[staticmethod]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(
        signature = (graph, *, n_neighbors = StrictUsize(10), mn_ratio = StrictF64(0.5), fp_ratio = StrictF64(2.0), phase_iterations = (StrictUsize(100), StrictUsize(100), StrictUsize(250)), learning_rate = StrictF64(1.0), keep_all_1hop = false, max_1hop_per_node = None, degree_weight_beta = StrictF64(0.0), num_landmarks = StrictUsize(16), landmarks = None, seed = StrictU64(42), dtype = ScalarDtype::F64),
        text_signature = "(graph, *, n_neighbors=10, mn_ratio=0.5, fp_ratio=2.0, phase_iterations=(100, 100, 250), learning_rate=1.0, keep_all_1hop=False, max_1hop_per_node=None, degree_weight_beta=0.0, num_landmarks=16, landmarks=None, seed=42, dtype='float64')"
    )]
    fn pacmap_graph(
        py: Python<'_>,
        graph: &PyCsrGraph,
        n_neighbors: StrictUsize,
        mn_ratio: StrictF64,
        fp_ratio: StrictF64,
        phase_iterations: (StrictUsize, StrictUsize, StrictUsize),
        learning_rate: StrictF64,
        keep_all_1hop: bool,
        max_1hop_per_node: Option<StrictUsize>,
        degree_weight_beta: StrictF64,
        num_landmarks: StrictUsize,
        landmarks: Option<&Bound<'_, PyAny>>,
        seed: StrictU64,
        dtype: ScalarDtype,
    ) -> PyResult<Self> {
        build_pacmap_graph_solver(
            py,
            graph,
            n_neighbors,
            mn_ratio,
            fp_ratio,
            phase_iterations,
            learning_rate,
            keep_all_1hop,
            max_1hop_per_node,
            degree_weight_beta,
            num_landmarks,
            landmarks,
            seed,
            dtype,
        )
    }

    /// Construct a 3-tier `PaCMAP-Graph` (`NB` / `MN` / `FP`) `SgdSolver` with external landmarks.
    #[staticmethod]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(
        signature = (graph, landmarks, *, n_neighbors = StrictUsize(10), mn_ratio = StrictF64(0.5), fp_ratio = StrictF64(2.0), phase_iterations = (StrictUsize(100), StrictUsize(100), StrictUsize(250)), learning_rate = StrictF64(1.0), keep_all_1hop = false, max_1hop_per_node = None, degree_weight_beta = StrictF64(0.0), num_landmarks = StrictUsize(16), seed = StrictU64(42), dtype = ScalarDtype::F64),
        text_signature = "(graph, landmarks, *, n_neighbors=10, mn_ratio=0.5, fp_ratio=2.0, phase_iterations=(100, 100, 250), learning_rate=1.0, keep_all_1hop=False, max_1hop_per_node=None, degree_weight_beta=0.0, num_landmarks=16, seed=42, dtype='float64')"
    )]
    fn pacmap_graph_with_landmarks(
        py: Python<'_>,
        graph: &PyCsrGraph,
        landmarks: &Bound<'_, PyAny>,
        n_neighbors: StrictUsize,
        mn_ratio: StrictF64,
        fp_ratio: StrictF64,
        phase_iterations: (StrictUsize, StrictUsize, StrictUsize),
        learning_rate: StrictF64,
        keep_all_1hop: bool,
        max_1hop_per_node: Option<StrictUsize>,
        degree_weight_beta: StrictF64,
        num_landmarks: StrictUsize,
        seed: StrictU64,
        dtype: ScalarDtype,
    ) -> PyResult<Self> {
        build_pacmap_graph_solver(
            py,
            graph,
            n_neighbors,
            mn_ratio,
            fp_ratio,
            phase_iterations,
            learning_rate,
            keep_all_1hop,
            max_1hop_per_node,
            degree_weight_beta,
            num_landmarks,
            Some(landmarks),
            seed,
            dtype,
        )
    }

    /// Construct a 3-tier `PaCMAP-Graph` solver coupled with a coarse Louvain area-reserved backbone.
    #[staticmethod]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(
        signature = (graph, *, n_neighbors = StrictUsize(10), mn_ratio = StrictF64(0.5), fp_ratio = StrictF64(2.0), phase_iterations = (StrictUsize(100), StrictUsize(100), StrictUsize(250)), learning_rate = StrictF64(1.0), keep_all_1hop = false, max_1hop_per_node = None, degree_weight_beta = StrictF64(0.0), num_landmarks = StrictUsize(16), max_coarse_nodes = StrictUsize(10_000), resolution = StrictF64(1.0), resolution_decay = StrictF64(0.5), max_passes = StrictUsize(100), min_modularity_gain = StrictF64(1e-7), max_outer_rounds = StrictUsize(16), edge_length_mode = "conductance_inverse", distance_backend = "rhgh", alpha = StrictF64(0.75), c0 = StrictF64(0.5), kappa = StrictF64(1.0), gamma_w = StrictF64(0.5), gamma_m = StrictF64(0.5), mu_ov = StrictF64(1.0), m_gap = StrictF64(0.1), coarse_pivots = StrictUsize(64), coarse_neg_per_node = StrictUsize(16), coarse_t_max = StrictUsize(80), rhgh_gamma = None, rhgh_p = None, rcf_steps = StrictUsize(2), rcf_eta = StrictF64(0.2), jitter_scale = StrictF64(0.5), gamma_tether = StrictF64(0.5), target_std = Some(StrictF64(1.0)), robust_quantile = StrictF64(0.95), seed = StrictU64(42), dtype = ScalarDtype::F64),
        text_signature = "(graph, *, n_neighbors=10, mn_ratio=0.5, fp_ratio=2.0, phase_iterations=(100, 100, 250), learning_rate=1.0, keep_all_1hop=False, max_1hop_per_node=None, degree_weight_beta=0.0, num_landmarks=16, max_coarse_nodes=10000, resolution=1.0, resolution_decay=0.5, max_passes=100, min_modularity_gain=1e-7, max_outer_rounds=16, edge_length_mode='conductance_inverse', distance_backend='rhgh', alpha=0.75, c0=0.5, kappa=1.0, gamma_w=0.5, gamma_m=0.5, mu_ov=1.0, m_gap=0.1, coarse_pivots=64, coarse_neg_per_node=16, coarse_t_max=80, rhgh_gamma=None, rhgh_p=None, rcf_steps=2, rcf_eta=0.2, jitter_scale=0.5, gamma_tether=0.5, target_std=1.0, robust_quantile=0.95, seed=42, dtype='float64')"
    )]
    fn pacmap_graph_with_backbone(
        py: Python<'_>,
        graph: &PyCsrGraph,
        n_neighbors: StrictUsize,
        mn_ratio: StrictF64,
        fp_ratio: StrictF64,
        phase_iterations: (StrictUsize, StrictUsize, StrictUsize),
        learning_rate: StrictF64,
        keep_all_1hop: bool,
        max_1hop_per_node: Option<StrictUsize>,
        degree_weight_beta: StrictF64,
        num_landmarks: StrictUsize,
        max_coarse_nodes: StrictUsize,
        resolution: StrictF64,
        resolution_decay: StrictF64,
        max_passes: StrictUsize,
        min_modularity_gain: StrictF64,
        max_outer_rounds: StrictUsize,
        edge_length_mode: &str,
        distance_backend: &str,
        alpha: StrictF64,
        c0: StrictF64,
        kappa: StrictF64,
        gamma_w: StrictF64,
        gamma_m: StrictF64,
        mu_ov: StrictF64,
        m_gap: StrictF64,
        coarse_pivots: StrictUsize,
        coarse_neg_per_node: StrictUsize,
        coarse_t_max: StrictUsize,
        rhgh_gamma: Option<StrictF64>,
        rhgh_p: Option<StrictF64>,
        rcf_steps: StrictUsize,
        rcf_eta: StrictF64,
        jitter_scale: StrictF64,
        gamma_tether: StrictF64,
        target_std: Option<StrictF64>,
        robust_quantile: StrictF64,
        seed: StrictU64,
        dtype: ScalarDtype,
    ) -> PyResult<(Self, PyLayout2d, Py<PyDict>)> {
        build_pacmap_graph_with_backbone(
            py,
            graph,
            n_neighbors,
            mn_ratio,
            fp_ratio,
            phase_iterations,
            learning_rate,
            keep_all_1hop,
            max_1hop_per_node,
            degree_weight_beta,
            num_landmarks,
            max_coarse_nodes,
            resolution,
            resolution_decay,
            max_passes,
            min_modularity_gain,
            max_outer_rounds,
            edge_length_mode,
            distance_backend,
            alpha,
            c0,
            kappa,
            gamma_w,
            gamma_m,
            mu_ov,
            m_gap,
            coarse_pivots,
            coarse_neg_per_node,
            coarse_t_max,
            rhgh_gamma,
            rhgh_p,
            rcf_steps,
            rcf_eta,
            jitter_scale,
            gamma_tether,
            target_std,
            robust_quantile,
            seed,
            dtype,
        )
    }

    /// Construct a doubly-stochastic mini-batch diagonal-preconditioned stress SGD `SgdSolver` (distinct from Ahmed et al. 2021 multi-criteria `(SGD)^2`).
    #[staticmethod]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(
        signature = (graph, *, iterations = StrictUsize(60), eps = StrictF64(0.01), step_scale = StrictF64(1.0), batch_size = StrictUsize(4096), damping = StrictF64(1e-4), seed = StrictU64(42), dtype = ScalarDtype::F64),
        text_signature = "(graph, *, iterations=60, eps=0.01, step_scale=1.0, batch_size=4096, damping=1e-4, seed=42, dtype='float64')"
    )]
    fn sgd2(
        py: Python<'_>,
        graph: &PyCsrGraph,
        iterations: StrictUsize,
        eps: StrictF64,
        step_scale: StrictF64,
        batch_size: StrictUsize,
        damping: StrictF64,
        seed: StrictU64,
        dtype: ScalarDtype,
    ) -> PyResult<Self> {
        build_sgd2_solver(
            py, graph, iterations, eps, step_scale, batch_size, damping, seed, dtype,
        )
    }

    /// Construct an Ortmann et al. (2016) pivot-seeded sparse stress `SgdSolver` from a `CsrGraph`.
    #[staticmethod]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(
        signature = (graph, *, num_pivots = StrictUsize(100), pivot_strategy = "kmeans_sp", iterations = StrictUsize(200), eta_max = None, eta_min = None, eps = StrictF64(0.1), seed = StrictU64(42), dtype = ScalarDtype::F64),
        text_signature = "(graph, *, num_pivots=100, pivot_strategy='kmeans_sp', iterations=200, eta_max=None, eta_min=None, eps=0.1, seed=42, dtype='float64')"
    )]
    fn ortmann_sparse_stress(
        py: Python<'_>,
        graph: &PyCsrGraph,
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

    /// Construct a Tang et al. (2016) negative-sampling `LargeVis` `SgdSolver` from a `CsrGraph`.
    #[staticmethod]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(
        signature = (graph, *, iterations = StrictUsize(200), negative_samples = StrictUsize(5), gamma = StrictF64(7.0), initial_lr = StrictF64(1.0), min_lr = StrictF64(1e-4), degree_exponent = StrictF64(0.75), seed = StrictU64(42), dtype = ScalarDtype::F64),
        text_signature = "(graph, *, iterations=200, negative_samples=5, gamma=7.0, initial_lr=1.0, min_lr=1e-4, degree_exponent=0.75, seed=42, dtype='float64')"
    )]
    fn largevis(
        py: Python<'_>,
        graph: &PyCsrGraph,
        iterations: StrictUsize,
        negative_samples: StrictUsize,
        gamma: StrictF64,
        initial_lr: StrictF64,
        min_lr: StrictF64,
        degree_exponent: StrictF64,
        seed: StrictU64,
        dtype: ScalarDtype,
    ) -> PyResult<Self> {
        build_largevis_solver(
            py,
            graph,
            iterations,
            negative_samples,
            gamma,
            initial_lr,
            min_lr,
            degree_exponent,
            seed,
            dtype,
        )
    }

    /// Construct a Zhu et al. (2020) multilevel negative-sampling `DRGraph` `SgdSolver`.
    #[staticmethod]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(
        signature = (graph, *, k_order = StrictUsize(1), negative_samples = StrictUsize(5), b = StrictF64(2.0), gamma_coarse = StrictF64(0.01), gamma_fine = StrictF64(0.1), coarsen_stop_ratio = StrictF64(0.8), iterations = StrictUsize(400), initial_lr = StrictF64(1.0), min_lr = StrictF64(1e-4), seed = StrictU64(42), dtype = ScalarDtype::F64),
        text_signature = "(graph, *, k_order=1, negative_samples=5, b=2.0, gamma_coarse=0.01, gamma_fine=0.1, coarsen_stop_ratio=0.8, iterations=400, initial_lr=1.0, min_lr=1e-4, seed=42, dtype='float64')"
    )]
    fn drgraph(
        py: Python<'_>,
        graph: &PyCsrGraph,
        k_order: StrictUsize,
        negative_samples: StrictUsize,
        b: StrictF64,
        gamma_coarse: StrictF64,
        gamma_fine: StrictF64,
        coarsen_stop_ratio: StrictF64,
        iterations: StrictUsize,
        initial_lr: StrictF64,
        min_lr: StrictF64,
        seed: StrictU64,
        dtype: ScalarDtype,
    ) -> PyResult<Self> {
        build_drgraph_solver(
            py,
            graph,
            k_order,
            negative_samples,
            b,
            gamma_coarse,
            gamma_fine,
            coarsen_stop_ratio,
            iterations,
            initial_lr,
            min_lr,
            seed,
            dtype,
        )
    }

    /// Construct a Bohm et al. (2025) `Graph-CNE` contrastive neighbor embedding `SgdSolver`.
    #[staticmethod]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(
        signature = (graph, *, objective = "infonce", negative_samples = StrictUsize(5), iterations = StrictUsize(400), initial_lr = StrictF64(1.0), min_lr = StrictF64(1e-4), repulsion_weight = StrictF64(1.0), row_normalize = true, seed = StrictU64(42), dtype = ScalarDtype::F64),
        text_signature = "(graph, *, objective='infonce', negative_samples=5, iterations=400, initial_lr=1.0, min_lr=1e-4, repulsion_weight=1.0, row_normalize=True, seed=42, dtype='float64')"
    )]
    fn graph_cne(
        py: Python<'_>,
        graph: &PyCsrGraph,
        objective: &str,
        negative_samples: StrictUsize,
        iterations: StrictUsize,
        initial_lr: StrictF64,
        min_lr: StrictF64,
        repulsion_weight: StrictF64,
        row_normalize: bool,
        seed: StrictU64,
        dtype: ScalarDtype,
    ) -> PyResult<Self> {
        build_graph_cne_solver(
            py,
            graph,
            objective,
            negative_samples,
            iterations,
            initial_lr,
            min_lr,
            repulsion_weight,
            row_normalize,
            seed,
            dtype,
        )
    }

    /// Execute one SGD iteration in place on `layout` (`Layout2d | LayoutNd | ManifoldLayout2d`).
    #[pyo3(signature = (layout), text_signature = "($self, layout)")]
    fn step(&mut self, py: Python<'_>, layout: &Bound<'_, PyAny>) -> PyResult<&'static str> {
        execute_any_layout_action(&mut self.inner, py, layout, SolverAction::Step)
    }

    /// Execute up to `steps` SGD iterations in place on `layout` (`Layout2d | LayoutNd | ManifoldLayout2d`).
    #[pyo3(signature = (layout, steps), text_signature = "($self, layout, steps)")]
    fn run_steps(
        &mut self,
        py: Python<'_>,
        layout: &Bound<'_, PyAny>,
        steps: StrictUsize,
    ) -> PyResult<&'static str> {
        execute_any_layout_action(&mut self.inner, py, layout, SolverAction::RunSteps(steps.0))
    }

    /// Run SGD to completion in place on `layout` (`Layout2d | LayoutNd | ManifoldLayout2d`).
    #[pyo3(signature = (layout), text_signature = "($self, layout)")]
    fn run(&mut self, py: Python<'_>, layout: &Bound<'_, PyAny>) -> PyResult<&'static str> {
        execute_any_layout_action(&mut self.inner, py, layout, SolverAction::Run)
    }

    /// Reset the SGD learning-rate schedule and PRNG seed while retaining prebuilt pair buffers.
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

    /// Execute `count` iterations starting at `start_iter` on a `Layout2d` or `ManifoldLayout2d`.
    #[pyo3(signature = (layout, start_iter, count), text_signature = "($self, layout, start_iter, count)")]
    fn run_schedule_window(
        &mut self,
        py: Python<'_>,
        layout: &Bound<'_, PyAny>,
        start_iter: StrictUsize,
        count: StrictUsize,
    ) -> PyResult<&'static str> {
        execute_schedule_window(&mut self.inner, py, layout, start_iter, count)
    }

    /// Return Sparse Barrier construction diagnostics as a dict, or `None`.
    fn stats(&self, py: Python<'_>) -> PyResult<Option<Py<PyDict>>> {
        self.barrier_stats
            .as_ref()
            .map(|st| sparse_barrier_stats_to_py_dict(py, st))
            .transpose()
    }

    /// Return regularized DoG Varadhan landmark diagnostics as a dict, or `None`.
    fn dog_stats(&self, py: Python<'_>) -> PyResult<Option<Py<PyDict>>> {
        self.dog_stats
            .as_ref()
            .map(|st| dog_stats_to_py_dict(py, st))
            .transpose()
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

    /// Scheduled total number of iterations (`iterations`).
    #[getter]
    fn total_iterations(&self) -> usize {
        self.inner.solver_total_iterations()
    }

    /// Whether the solver has reached `"converged"` or `"completed"`.
    #[getter]
    fn is_finished(&self) -> bool {
        self.inner.solver_status().is_finished()
    }

    /// Whether this solver supports non-Euclidean `ManifoldLayout2d` stepping.
    #[getter]
    fn supports_manifold(&self) -> bool {
        self.inner.supports_manifold()
    }

    /// Expected number of vertices in the layout.
    #[getter]
    fn num_nodes(&self) -> usize {
        self.inner.num_nodes()
    }

    /// Number of two-sided spring pairs stored in SoA buffers.
    #[getter]
    fn num_springs(&self) -> usize {
        self.inner.num_springs()
    }

    /// Number of one-sided hinge repulsion pairs stored in SoA buffers.
    #[getter]
    fn num_hinges(&self) -> usize {
        self.inner.num_hinges()
    }

    /// Learning rate `eta` for the upcoming iteration (`0.0` once finished).
    #[getter]
    fn current_eta(&self) -> f64 {
        self.inner.current_eta()
    }

    /// Solver scalar precision (`"float64"` or `"float32"`).
    #[getter]
    fn dtype(&self) -> &'static str {
        self.inner.dtype().as_str()
    }

    fn __repr__(&self) -> String {
        format!(
            "SgdSolver(num_nodes={}, springs={}, hinges={}, status='{}', iteration={}/{}, dtype='{}')",
            self.num_nodes(),
            self.num_springs(),
            self.num_hinges(),
            self.status(),
            self.iteration(),
            self.total_iterations(),
            self.dtype()
        )
    }
}
