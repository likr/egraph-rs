from __future__ import annotations

from typing import Sequence, final

from ._core_core import CsrGraph, Layout2d, Layout3d, LayoutNd
from ._core_manifold import ManifoldLayout2d
from ._core_neighborhood import DogVaradhanLandmarks
from ._core_solvers_sgd_variants import (
    DogVaradhanStats as DogVaradhanStats,
    DrGraphSolver as DrGraphSolver,
    GraphCneSolver as GraphCneSolver,
    LargeVisSolver as LargeVisSolver,
    PacmapGraphSolver as PacmapGraphSolver,
    Sgd2Solver as Sgd2Solver,
    SnapTFdpSolver as SnapTFdpSolver,
    SparseBarrierStats as SparseBarrierStats,
)
from ._core_types import (
    SolverStatus,
    TargetKind,
    _DtypeLike,
    _IndexArrayLike,
    _MatrixArrayLike,
    _WeightArrayLike,
)

_AnyLayout = Layout2d | Layout3d | LayoutNd | ManifoldLayout2d

__all__ = [
    "DogVaradhanStats",
    "DrGraphSolver",
    "GraphCneSolver",
    "LargeVisSolver",
    "PacmapGraphSolver",
    "Sgd2Solver",
    "SgdSolver",
    "SnapTFdpSolver",
    "SparseBarrierStats",
]

@final
class SgdSolver:
    def __new__(
        cls,
        graph: CsrGraph,
        *,
        target: TargetKind = "layout2d",
        iterations: int = 60,
        eps: float = 0.01,
        schedule: str = "exponential",
        step_scale: float = 1.0,
        max_mu: float | None = None,
        seed: int = 42,
        dtype: _DtypeLike = "float64",
    ) -> SgdSolver: ...
    @staticmethod
    def from_pairs(
        num_nodes: int,
        src: _IndexArrayLike,
        tgt: _IndexArrayLike,
        dij: _WeightArrayLike,
        *,
        dji: _WeightArrayLike | None = None,
        wij: _WeightArrayLike | None = None,
        wji: _WeightArrayLike | None = None,
        target: TargetKind = "layout2d",
        iterations: int = 60,
        eps: float = 0.01,
        schedule: str = "exponential",
        step_scale: float = 1.0,
        max_mu: float | None = None,
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
        target: TargetKind = "layout2d",
        iterations: int = 60,
        eps: float = 0.01,
        schedule: str = "exponential",
        step_scale: float = 1.0,
        max_mu: float | None = None,
        seed: int = 42,
        dtype: _DtypeLike = "float64",
    ) -> SgdSolver: ...
    @staticmethod
    def from_packed_symmetric(
        num_nodes: int,
        src: _IndexArrayLike,
        tgt: _IndexArrayLike,
        dists: _WeightArrayLike,
        weights: _WeightArrayLike,
        is_hinge: Sequence[bool],
        *,
        target: TargetKind = "layout2d",
        iterations: int = 60,
        eps: float = 0.01,
        schedule: str = "exponential",
        step_scale: float = 1.0,
        max_mu: float | None = None,
        seed: int = 42,
        dtype: _DtypeLike = "float64",
    ) -> SgdSolver: ...
    @staticmethod
    def full(
        graph: CsrGraph,
        *,
        target: TargetKind = "layout2d",
        iterations: int = 60,
        eps: float = 0.01,
        schedule: str = "exponential",
        step_scale: float = 1.0,
        max_mu: float | None = None,
        seed: int = 42,
        dtype: _DtypeLike = "float64",
    ) -> SgdSolver: ...
    @staticmethod
    def sparse_barrier_from_landmarks(
        graph: CsrGraph,
        landmarks: DogVaradhanLandmarks,
        pivots: _IndexArrayLike | None = None,
        *,
        target: TargetKind = "layout2d",
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
        iterations: int = 60,
        eps: float = 0.01,
        schedule: str = "exponential",
        step_scale: float = 1.0,
        max_mu: float | None = None,
        seed: int = 42,
        dtype: _DtypeLike = "float64",
    ) -> SgdSolver: ...
    @staticmethod
    def sparse_barrier_dog(
        graph: CsrGraph,
        *,
        target: TargetKind = "layout2d",
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
        iterations: int = 60,
        eps: float = 0.01,
        schedule: str = "exponential",
        step_scale: float = 1.0,
        max_mu: float | None = None,
        seed: int = 42,
        dtype: _DtypeLike = "float64",
    ) -> SgdSolver: ...
    @staticmethod
    def streaming_negatives(
        graph: CsrGraph,
        *,
        target: TargetKind = "layout2d",
        negatives_per_node: int = 5,
        target_distance: float = 2.0,
        weight: float = 0.25,
        hinge_only: bool = True,
        iterations: int = 60,
        eps: float = 0.01,
        schedule: str = "exponential",
        step_scale: float = 1.0,
        max_mu: float | None = None,
        seed: int = 42,
        dtype: _DtypeLike = "float64",
    ) -> SgdSolver: ...
    @staticmethod
    def random_pair(
        graph: CsrGraph,
        pairs_per_node: int = 30,
        *,
        target: TargetKind = "layout2d",
        iterations: int = 60,
        eps: float = 0.01,
        schedule: str = "exponential",
        step_scale: float = 1.0,
        max_mu: float | None = None,
        seed: int = 42,
        dtype: _DtypeLike = "float64",
    ) -> SgdSolver: ...
    @staticmethod
    def omega(
        graph: CsrGraph,
        embedding: _MatrixArrayLike | Layout2d | Layout3d | LayoutNd,
        k: int = 30,
        min_dist: float = 1e-3,
        *,
        target: TargetKind = "layout2d",
        iterations: int = 60,
        eps: float = 0.01,
        schedule: str = "exponential",
        step_scale: float = 1.0,
        max_mu: float | None = None,
        seed: int = 42,
        dtype: _DtypeLike = "float64",
    ) -> SgdSolver: ...
    @staticmethod
    def ortmann_sparse_stress(
        graph: CsrGraph,
        *,
        target: TargetKind = "layout2d",
        num_pivots: int = 100,
        pivot_strategy: str = "kmeans_sp",
        iterations: int = 200,
        eta_max: float | None = None,
        eta_min: float | None = None,
        eps: float = 0.1,
        seed: int = 42,
        dtype: _DtypeLike = "float64",
    ) -> SgdSolver: ...
    def step(self, layout: _AnyLayout) -> SolverStatus: ...
    def run_steps(self, layout: _AnyLayout, steps: int) -> SolverStatus: ...
    def run(self, layout: _AnyLayout) -> SolverStatus: ...
    def reset_schedule(
        self,
        *,
        iterations: int = 60,
        eps: float = 0.01,
        schedule: str = "exponential",
        step_scale: float = 1.0,
        max_mu: float | None = None,
        seed: int = 42,
    ) -> None: ...
    def run_schedule_window(
        self,
        layout: _AnyLayout,
        start_iter: int,
        count: int,
    ) -> SolverStatus: ...
    def stats(self) -> SparseBarrierStats | None: ...
    def dog_stats(self) -> DogVaradhanStats | None: ...
    @property
    def target(self) -> TargetKind: ...
    @property
    def status(self) -> SolverStatus: ...
    @property
    def iteration(self) -> int: ...
    @property
    def total_iterations(self) -> int | None: ...
    @property
    def is_finished(self) -> bool: ...
    @property
    def supports_manifold(self) -> bool: ...
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
