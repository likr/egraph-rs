from __future__ import annotations

from typing import Any, final

from ._core_core import CsrGraph, Layout2d
from ._core_manifold import ManifoldLayout2d
from ._core_neighborhood import DogVaradhanLandmarks
from ._core_types import (
    _DtypeLike,
    _IndexArrayLike,
    _MatrixArrayLike,
    _WeightArrayLike,
)

__all__ = [
    "SgdSolver",
]

@final
class SgdSolver:
    @staticmethod
    def from_pairs(
        num_nodes: int,
        src: _IndexArrayLike,
        tgt: _IndexArrayLike,
        dij: _WeightArrayLike,
        dji: _WeightArrayLike | None = None,
        wij: _WeightArrayLike | None = None,
        wji: _WeightArrayLike | None = None,
        *,
        t_max: int = 60,
        eps: float = 0.01,
        schedule: str = "exponential",
        step_scale: float = 1.0,
        seed: int = 42,
        dtype: _DtypeLike = "float64",
    ) -> SgdSolver: ...
    @staticmethod
    def from_pairs_with_hinges(
        num_nodes: int,
        spring_src: _IndexArrayLike,
        spring_tgt: _IndexArrayLike,
        spring_dij: _WeightArrayLike,
        hinge_src: _IndexArrayLike,
        hinge_tgt: _IndexArrayLike,
        hinge_dij: _WeightArrayLike,
        *,
        spring_dji: _WeightArrayLike | None = None,
        spring_wij: _WeightArrayLike | None = None,
        spring_wji: _WeightArrayLike | None = None,
        hinge_dji: _WeightArrayLike | None = None,
        hinge_wij: _WeightArrayLike | None = None,
        hinge_wji: _WeightArrayLike | None = None,
        t_max: int = 60,
        eps: float = 0.01,
        schedule: str = "exponential",
        step_scale: float = 1.0,
        seed: int = 42,
        dtype: _DtypeLike = "float64",
    ) -> SgdSolver: ...
    @staticmethod
    def full(
        graph: CsrGraph,
        *,
        t_max: int = 60,
        eps: float = 0.01,
        schedule: str = "exponential",
        step_scale: float = 1.0,
        seed: int = 42,
        dtype: _DtypeLike = "float64",
    ) -> SgdSolver: ...
    @staticmethod
    def sparse_barrier_from_landmarks(
        graph: CsrGraph,
        landmarks: DogVaradhanLandmarks,
        pivots: _IndexArrayLike | None = None,
        *,
        max_2hop_per_node: int = 32,
        hinge_repulsion: bool = True,
        delta_rel: float = 0.25,
        barrier_sym: str = "sym_max",
        omega_1: float = 1.0,
        omega_2: float = 1.0,
        gamma_non2: float = 1.0,
        ht_repulsion: bool = True,
        min_ratio: float = 0.5,
        eps_dist: float = 1e-6,
        neg_per_node: int | None = None,
        num_pivots: int = 50,
        core_pivots: bool = True,
        degree_weight_beta: float = 0.0,
        t_max: int = 60,
        eps: float = 0.01,
        schedule: str = "exponential",
        step_scale: float = 1.0,
        seed: int = 42,
        dtype: _DtypeLike = "float64",
    ) -> SgdSolver: ...
    @staticmethod
    def sparse_barrier_dog(
        graph: CsrGraph,
        *,
        degree: int | None = None,
        num_landmarks: int = 64,
        degree_normalize: bool = True,
        p_exp: float = 0.5,
        eps_psd: float = 1e-4,
        t_coarse: float = 48.0,
        t1: float | None = None,
        t2: float | None = None,
        alpha: float | None = None,
        beta_floor: float | None = None,
        min_degree: int = 2,
        jl_dim: int = 64,
        max_2hop_per_node: int = 32,
        hinge_repulsion: bool = True,
        delta_rel: float = 0.25,
        barrier_sym: str = "sym_max",
        omega_1: float = 1.0,
        omega_2: float = 1.0,
        gamma_non2: float = 1.0,
        ht_repulsion: bool = True,
        min_ratio: float = 0.5,
        eps_dist: float = 1e-6,
        neg_per_node: int | None = None,
        num_pivots: int = 50,
        core_pivots: bool = True,
        degree_weight_beta: float = 0.0,
        t_max: int = 60,
        eps: float = 0.01,
        schedule: str = "exponential",
        step_scale: float = 1.0,
        seed: int = 42,
        dtype: _DtypeLike = "float64",
    ) -> SgdSolver: ...
    @staticmethod
    def random_pair(
        graph: CsrGraph,
        pairs_per_node: int = 30,
        *,
        t_max: int = 60,
        eps: float = 0.01,
        schedule: str = "exponential",
        step_scale: float = 1.0,
        seed: int = 42,
        dtype: _DtypeLike = "float64",
    ) -> SgdSolver: ...
    @staticmethod
    def omega(
        graph: CsrGraph,
        embedding: _MatrixArrayLike,
        k: int = 30,
        min_dist: float = 1e-3,
        *,
        t_max: int = 60,
        eps: float = 0.01,
        schedule: str = "exponential",
        step_scale: float = 1.0,
        seed: int = 42,
        dtype: _DtypeLike = "float64",
    ) -> SgdSolver: ...
    @staticmethod
    def streaming_negatives(
        graph: CsrGraph,
        *,
        negatives_per_node: int = 5,
        target_distance: float = 2.0,
        weight: float = 0.25,
        hinge_only: bool = True,
        t_max: int = 60,
        eps: float = 0.01,
        schedule: str = "exponential",
        step_scale: float = 1.0,
        seed: int = 42,
        dtype: _DtypeLike = "float64",
    ) -> SgdSolver: ...
    @staticmethod
    def snap_t_fdp(
        graph: CsrGraph,
        *,
        iterations: int = 200,
        negatives_per_edge: int = 3,
        learning_rate: float = 1.0,
        alpha: float = 1.0,
        repulsion_scale: float = 1.0,
        rho_start: float = 1.0,
        rho_end: float = 0.1,
        rho_power: float = 1.0,
        degree_beta: float = 0.0,
        bounded_attraction: bool = True,
        seed: int = 42,
        dtype: _DtypeLike = "float64",
    ) -> SgdSolver: ...
    @staticmethod
    def pacmap_graph(
        graph: CsrGraph,
        *,
        n_neighbors: int = 10,
        mn_ratio: float = 0.5,
        fp_ratio: float = 2.0,
        phase1_iters: int = 100,
        phase2_iters: int = 100,
        phase3_iters: int = 250,
        learning_rate: float = 1.0,
        keep_all_1hop: bool = False,
        max_1hop_per_node: int | None = None,
        degree_weight_beta: float = 0.0,
        num_landmarks: int = 16,
        landmarks: DogVaradhanLandmarks | _MatrixArrayLike | None = None,
        seed: int = 42,
        dtype: _DtypeLike = "float64",
    ) -> SgdSolver: ...
    @staticmethod
    def pacmap_graph_with_landmarks(
        graph: CsrGraph,
        landmarks: DogVaradhanLandmarks | _MatrixArrayLike,
        *,
        n_neighbors: int = 10,
        mn_ratio: float = 0.5,
        fp_ratio: float = 2.0,
        phase1_iters: int = 100,
        phase2_iters: int = 100,
        phase3_iters: int = 250,
        learning_rate: float = 1.0,
        keep_all_1hop: bool = False,
        max_1hop_per_node: int | None = None,
        degree_weight_beta: float = 0.0,
        num_landmarks: int = 16,
        seed: int = 42,
        dtype: _DtypeLike = "float64",
    ) -> SgdSolver: ...
    @staticmethod
    def pacmap_graph_with_backbone(
        graph: CsrGraph,
        *,
        n_neighbors: int = 10,
        mn_ratio: float = 0.5,
        fp_ratio: float = 2.0,
        phase1_iters: int = 100,
        phase2_iters: int = 100,
        phase3_iters: int = 250,
        learning_rate: float = 1.0,
        keep_all_1hop: bool = False,
        max_1hop_per_node: int | None = None,
        degree_weight_beta: float = 0.0,
        num_landmarks: int = 16,
        max_coarse_nodes: int = 10000,
        resolution: float = 1.0,
        resolution_decay: float = 0.5,
        max_passes: int = 100,
        min_modularity_gain: float = 1e-7,
        max_outer_rounds: int = 16,
        edge_length_mode: str = "conductance_inverse",
        distance_backend: str = "rhgh",
        alpha: float = 0.75,
        c0: float = 0.5,
        kappa: float = 1.0,
        gamma_w: float = 0.5,
        gamma_m: float = 0.5,
        mu_ov: float = 1.0,
        m_gap: float = 0.1,
        coarse_pivots: int = 64,
        coarse_neg_per_node: int = 16,
        coarse_t_max: int = 80,
        rhgh_gamma: float | None = None,
        rhgh_p: float | None = None,
        rcf_steps: int = 2,
        rcf_eta: float = 0.2,
        jitter_scale: float = 0.5,
        gamma_tether: float = 0.5,
        target_std: float | None = 1.0,
        robust_quantile: float = 0.95,
        seed: int = 42,
        dtype: _DtypeLike = "float64",
    ) -> tuple[SgdSolver, Layout2d, dict[str, Any]]: ...
    @staticmethod
    def sgd2(
        graph: CsrGraph,
        *,
        t_max: int = 60,
        eps: float = 0.01,
        step_scale: float = 1.0,
        batch_size: int = 4096,
        damping: float = 1e-4,
        seed: int = 42,
        dtype: _DtypeLike = "float64",
    ) -> SgdSolver: ...
    @staticmethod
    def ortmann_sparse_stress(
        graph: CsrGraph,
        *,
        num_pivots: int = 100,
        pivot_strategy: str = "kmeans_sp",
        iterations: int = 200,
        eta_max: float | None = None,
        eta_min: float | None = None,
        eps: float = 0.1,
        seed: int = 42,
        dtype: _DtypeLike = "float64",
    ) -> SgdSolver: ...
    @staticmethod
    def largevis(
        graph: CsrGraph,
        *,
        iterations: int = 200,
        negative_samples: int = 5,
        gamma: float = 7.0,
        initial_lr: float = 1.0,
        min_lr: float = 1e-4,
        degree_exponent: float = 0.75,
        seed: int = 42,
        dtype: _DtypeLike = "float64",
    ) -> SgdSolver: ...
    @staticmethod
    def drgraph(
        graph: CsrGraph,
        *,
        k_order: int = 1,
        negative_samples: int = 5,
        b: float = 2.0,
        gamma_coarse: float = 0.01,
        gamma_fine: float = 0.1,
        coarsen_stop_ratio: float = 0.8,
        iterations: int = 400,
        initial_lr: float = 1.0,
        min_lr: float = 1e-4,
        seed: int = 42,
        dtype: _DtypeLike = "float64",
    ) -> SgdSolver: ...
    @staticmethod
    def graph_cne(
        graph: CsrGraph,
        *,
        objective: str = "infonce",
        negative_samples: int = 5,
        iterations: int = 400,
        initial_lr: float = 1.0,
        min_lr: float = 1e-4,
        repulsion_weight: float = 1.0,
        row_normalize: bool = True,
        seed: int = 42,
        dtype: _DtypeLike = "float64",
    ) -> SgdSolver: ...
    def step(self, layout: Layout2d) -> str: ...
    def run_steps(self, layout: Layout2d, steps: int) -> str: ...
    def run(self, layout: Layout2d) -> str: ...
    def step_manifold(self, layout: ManifoldLayout2d) -> str: ...
    def run_steps_manifold(self, layout: ManifoldLayout2d, steps: int) -> str: ...
    def run_manifold(self, layout: ManifoldLayout2d) -> str: ...
    def stats(self) -> dict[str, Any] | None: ...
    def dog_stats(self) -> dict[str, Any] | None: ...
    @property
    def status(self) -> str: ...
    @property
    def iteration(self) -> int: ...
    @property
    def total_iterations(self) -> int: ...
    @property
    def is_finished(self) -> bool: ...
    @property
    def num_nodes(self) -> int: ...
    @property
    def num_springs(self) -> int: ...
    @property
    def num_hinges(self) -> int: ...
    @property
    def current_eta(self) -> float: ...
    @property
    def dtype(self) -> str: ...
    def __repr__(self) -> str: ...
