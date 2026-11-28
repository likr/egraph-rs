from __future__ import annotations

from typing import Literal, final

import numpy as np
import numpy.typing as npt

from ._core_core import CsrGraph, Layout2d, Layout3d, LayoutNd
from ._core_manifold import ManifoldLayout2d
from ._core_types import (
    BackendKind,
    EuclideanTargetKind,
    SolverStatus,
    TargetKind,
    _DtypeLike,
    _MatrixArrayLike,
)

_EuclidLayout = Layout2d | Layout3d | LayoutNd
_AnyLayout = Layout2d | Layout3d | LayoutNd | ManifoldLayout2d

__all__ = [
    "GripSolver",
    "KamadaKawaiSolver",
    "MaxentStressSolver",
    "SfcLayout",
    "StressMajorizationSolver",
]

@final
class GripSolver:
    def __new__(
        cls,
        graph: CsrGraph,
        *,
        rounds_min: int = 5,
        rounds_max: int = 30,
        local_neighborhood_size: int = 20,
        repulsion_scale: float = 0.05,
        edge_length: float = 1.0,
        temp_r: float = 0.15,
        temp_s: float = 3.0,
        seed: int = 42,
        dtype: _DtypeLike = "float64",
    ) -> GripSolver: ...
    def step(self, layout: Layout2d) -> SolverStatus: ...
    def run_steps(self, layout: Layout2d, steps: int) -> SolverStatus: ...
    def run(self, layout: Layout2d) -> SolverStatus: ...
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
    def num_levels(self) -> int: ...
    @property
    def dtype(self) -> str: ...
    def __repr__(self) -> str: ...

@final
class KamadaKawaiSolver:
    def __new__(
        cls,
        graph: CsrGraph,
        *,
        target: TargetKind = "layout2d",
        eps: float = 0.01,
        max_inner_iters: int = 50,
        max_iterations: int | None = None,
        dtype: _DtypeLike = "float64",
    ) -> KamadaKawaiSolver: ...
    @staticmethod
    def from_graph(
        graph: CsrGraph,
        *,
        target: TargetKind = "layout2d",
        eps: float = 0.01,
        max_inner_iters: int = 50,
        max_iterations: int | None = None,
        dtype: _DtypeLike = "float64",
    ) -> KamadaKawaiSolver: ...
    @staticmethod
    def from_distance_matrix(
        distances: _MatrixArrayLike,
        *,
        target: TargetKind = "layout2d",
        eps: float = 0.01,
        max_inner_iters: int = 50,
        max_iterations: int | None = None,
        dtype: _DtypeLike = "float64",
    ) -> KamadaKawaiSolver: ...
    def step(self, layout: _AnyLayout) -> SolverStatus: ...
    def run_steps(self, layout: _AnyLayout, steps: int) -> SolverStatus: ...
    def run(self, layout: _AnyLayout) -> SolverStatus: ...
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
    def dtype(self) -> str: ...
    def __repr__(self) -> str: ...

@final
class MaxentStressSolver:
    def __new__(
        cls,
        graph: CsrGraph,
        *,
        target: EuclideanTargetKind = "layout2d",
        backend: BackendKind = "auto",
        q: float | None = None,
        alpha_init: float = 1.0,
        alpha_decay: float = 0.3,
        alpha_min: float = 0.008,
        stages: int = 5,
        steps_per_stage: int = 50,
        tol: float = 1e-3,
        cg_tol: float = 0.1,
        cg_max_iters: int = 10,
        theta: float = 1.2,
        use_barnes_hut: bool | None = None,
        eps_sq: float = 1e-6,
        seed: int = 42,
        dtype: _DtypeLike = "float64",
    ) -> MaxentStressSolver: ...
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
    def current_alpha(self) -> float: ...
    @property
    def effective_q(self) -> float: ...
    @property
    def dtype(self) -> str: ...
    def __repr__(self) -> str: ...

@final
class SfcLayout:
    def __new__(
        cls,
        graph: CsrGraph,
        *,
        curve: str = "hilbert",
        scale: float = 100.0,
        resolution_order: int | None = None,
        seed: int = 42,
        dtype: _DtypeLike = "float64",
    ) -> SfcLayout: ...
    def generate(self) -> Layout2d: ...
    def vertex_order(self) -> npt.NDArray[np.uint32]: ...
    @property
    def num_nodes(self) -> int: ...
    @property
    def dtype(self) -> str: ...
    def __repr__(self) -> str: ...

@final
class StressMajorizationSolver:
    def __new__(
        cls,
        graph: CsrGraph,
        *,
        target: TargetKind = "layout2d",
        strategy: Literal["auto", "euclidean_pcg", "geodesic_gradient"] | None = None,
        backend: Literal["auto", "euclidean_pcg", "geodesic_gradient"] | None = None,
        epsilon: float = 1e-4,
        max_iterations: int = 100,
        pcg_tolerance: float = 1e-6,
        pcg_max_iterations: int | None = None,
        min_norm: float = 1e-4,
        dtype: _DtypeLike = "float64",
    ) -> StressMajorizationSolver: ...
    @staticmethod
    def from_graph(
        graph: CsrGraph,
        *,
        target: TargetKind = "layout2d",
        strategy: Literal["auto", "euclidean_pcg", "geodesic_gradient"] | None = None,
        backend: Literal["auto", "euclidean_pcg", "geodesic_gradient"] | None = None,
        epsilon: float = 1e-4,
        max_iterations: int = 100,
        pcg_tolerance: float = 1e-6,
        pcg_max_iterations: int | None = None,
        min_norm: float = 1e-4,
        dtype: _DtypeLike = "float64",
    ) -> StressMajorizationSolver: ...
    @staticmethod
    def from_distance_matrix(
        distances: _MatrixArrayLike,
        *,
        target: TargetKind = "layout2d",
        strategy: Literal["auto", "euclidean_pcg", "geodesic_gradient"] | None = None,
        backend: Literal["auto", "euclidean_pcg", "geodesic_gradient"] | None = None,
        epsilon: float = 1e-4,
        max_iterations: int = 100,
        pcg_tolerance: float = 1e-6,
        pcg_max_iterations: int | None = None,
        min_norm: float = 1e-4,
        dtype: _DtypeLike = "float64",
    ) -> StressMajorizationSolver: ...
    def step(self, layout: _AnyLayout) -> SolverStatus: ...
    def run_steps(self, layout: _AnyLayout, steps: int) -> SolverStatus: ...
    def run(self, layout: _AnyLayout) -> SolverStatus: ...
    def compute_stress(self, layout: _AnyLayout) -> float: ...
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
    def last_stress(self) -> float | None: ...
    @property
    def dtype(self) -> str: ...
    def __repr__(self) -> str: ...
