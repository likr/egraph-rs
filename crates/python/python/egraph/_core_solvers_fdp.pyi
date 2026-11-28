from __future__ import annotations

from typing import final

from ._core_core import CsrGraph, Layout2d, Layout3d, LayoutNd
from ._core_neighborhood import DogVaradhanLandmarks, KnnGraph, SparseProbabilities
from ._core_solvers_neighbor import FieldFftStats
from ._core_types import (
    BackendKind,
    EuclideanTargetKind,
    SolverStatus,
    TargetKind,
    _DtypeLike,
    _IndexArrayLike,
    _WeightArrayLike,
)

_EuclidLayout = Layout2d | Layout3d | LayoutNd

__all__ = [
    "IbUmapSolver",
    "TFdpSolver",
]

@final
class IbUmapSolver:
    def __new__(
        cls,
        graph: CsrGraph,
        *,
        target: TargetKind = "layout2d",
        max_2hop_per_node: int = 30,
        degree: int | None = None,
        num_landmarks: int = 64,
        min_degree: int = 2,
        jl_dim: int = 64,
        degree_normalize: bool = True,
        p_exp: float = 0.5,
        eps_psd: float = 1e-4,
        t_coarse: float = 48.0,
        t1: float | None = None,
        t2: float | None = None,
        alpha: float | None = None,
        beta_floor: float | None = None,
        backend: BackendKind = "ib_fft",
        iterations: int = 200,
        learning_rate: float = 1.0,
        min_dist: float = 0.1,
        spread: float = 1.0,
        a: float | None = None,
        b: float | None = None,
        c_k: float = 4.0,
        tau_scale: float = 4.0,
        hub_damping: bool = True,
        theta: float = 0.5,
        interpolation_order: int = 1,
        intervals: int = 25,
        min_grid_dim: int = 64,
        max_grid_dim: int = 512,
        seed: int = 42,
        dtype: _DtypeLike = "float64",
    ) -> IbUmapSolver: ...
    @staticmethod
    def from_knn(
        knn: KnnGraph,
        *,
        target: TargetKind = "layout2d",
        sigma_iters: int = 64,
        sigma_tolerance: float = 1e-5,
        backend: BackendKind = "ib_fft",
        iterations: int = 200,
        learning_rate: float = 1.0,
        min_dist: float = 0.1,
        spread: float = 1.0,
        a: float | None = None,
        b: float | None = None,
        c_k: float = 4.0,
        tau_scale: float = 4.0,
        hub_damping: bool = True,
        theta: float = 0.5,
        interpolation_order: int = 1,
        intervals: int = 25,
        min_grid_dim: int = 64,
        max_grid_dim: int = 512,
        seed: int = 42,
        dtype: _DtypeLike = "float64",
    ) -> IbUmapSolver: ...
    @staticmethod
    def from_fuzzy_set(
        probs: SparseProbabilities,
        *,
        target: TargetKind = "layout2d",
        degrees: _WeightArrayLike | None = None,
        backend: BackendKind = "ib_fft",
        iterations: int = 200,
        learning_rate: float = 1.0,
        min_dist: float = 0.1,
        spread: float = 1.0,
        a: float | None = None,
        b: float | None = None,
        c_k: float = 4.0,
        tau_scale: float = 4.0,
        hub_damping: bool = True,
        theta: float = 0.5,
        interpolation_order: int = 1,
        intervals: int = 25,
        min_grid_dim: int = 64,
        max_grid_dim: int = 512,
        seed: int = 42,
        dtype: _DtypeLike = "float64",
    ) -> IbUmapSolver: ...
    @staticmethod
    def from_landmarks(
        graph: CsrGraph,
        landmarks: DogVaradhanLandmarks,
        max_2hop_per_node: int = 32,
        *,
        target: TargetKind = "layout2d",
        backend: BackendKind = "ib_fft",
        iterations: int = 200,
        learning_rate: float = 1.0,
        min_dist: float = 0.1,
        spread: float = 1.0,
        a: float | None = None,
        b: float | None = None,
        c_k: float = 4.0,
        tau_scale: float = 4.0,
        hub_damping: bool = True,
        theta: float = 0.5,
        interpolation_order: int = 1,
        intervals: int = 25,
        min_grid_dim: int = 64,
        max_grid_dim: int = 512,
        seed: int = 42,
        dtype: _DtypeLike = "float64",
    ) -> IbUmapSolver: ...
    def validate_target(self, layout: Layout2d) -> None: ...
    def step(self, layout: Layout2d) -> SolverStatus: ...
    def run_steps(self, layout: Layout2d, steps: int) -> SolverStatus: ...
    def run(self, layout: Layout2d) -> SolverStatus: ...
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
    def num_nodes(self) -> int: ...
    @property
    def num_edges(self) -> int: ...
    @property
    def current_rho(self) -> float: ...
    @property
    def a(self) -> float: ...
    @property
    def b(self) -> float: ...
    @property
    def ab(self) -> tuple[float, float]: ...
    @property
    def total_fft_2d_calls(self) -> int: ...
    def last_fft_stats(self) -> FieldFftStats | None: ...
    @property
    def dtype(self) -> str: ...
    def __repr__(self) -> str: ...

@final
class TFdpSolver:
    def __new__(
        cls,
        graph: CsrGraph,
        *,
        target: EuclideanTargetKind = "layout2d",
        backend: BackendKind = "auto",
        iterations: int = 200,
        learning_rate: float = 1.0,
        momentum: float = 0.8,
        theta: float = 0.5,
        alpha: float = 1.0,
        gamma: float = 1.0,
        rho_start: float = 1.0,
        rho_end: float = 0.1,
        rho_power: float = 1.0,
        interpolation_order: int = 1,
        intervals: int = 25,
        min_grid_dim: int = 64,
        max_grid_dim: int = 512,
        seed: int = 42,
        dtype: _DtypeLike = "float64",
    ) -> TFdpSolver: ...
    @staticmethod
    def from_knn(
        knn_or_indptr: KnnGraph | _IndexArrayLike,
        indices: _IndexArrayLike | None = None,
        weights: _WeightArrayLike | None = None,
        *,
        target: EuclideanTargetKind = "layout2d",
        backend: BackendKind = "auto",
        iterations: int = 200,
        learning_rate: float = 1.0,
        momentum: float = 0.8,
        theta: float = 0.5,
        alpha: float = 1.0,
        gamma: float = 1.0,
        rho_start: float = 1.0,
        rho_end: float = 0.1,
        rho_power: float = 1.0,
        interpolation_order: int = 1,
        intervals: int = 25,
        min_grid_dim: int = 64,
        max_grid_dim: int = 512,
        seed: int = 42,
        dtype: _DtypeLike = "float64",
    ) -> TFdpSolver: ...
    def validate_target(self, layout: _EuclidLayout) -> None: ...
    def step(self, layout: _EuclidLayout) -> SolverStatus: ...
    def run_steps(self, layout: _EuclidLayout, steps: int) -> SolverStatus: ...
    def run(self, layout: _EuclidLayout) -> SolverStatus: ...
    @property
    def target(self) -> EuclideanTargetKind: ...
    @property
    def status(self) -> SolverStatus: ...
    @property
    def iteration(self) -> int: ...
    @property
    def total_iterations(self) -> int | None: ...
    @property
    def is_finished(self) -> bool: ...
    @property
    def num_nodes(self) -> int: ...
    @property
    def num_edges(self) -> int: ...
    @property
    def current_rho(self) -> float: ...
    @property
    def total_fft_2d_calls(self) -> int: ...
    def last_fft_stats(self) -> FieldFftStats | None: ...
    @property
    def dtype(self) -> str: ...
    def __repr__(self) -> str: ...
