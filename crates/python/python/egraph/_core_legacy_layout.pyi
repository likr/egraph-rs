from __future__ import annotations

from typing import Any, Callable, final

import numpy as np
import numpy.typing as npt

from ._core_core import (
    Array1,
    Array2,
    DrawingEuclidean,
    DrawingEuclidean2d,
    DrawingHyperbolic2d,
    DrawingSpherical2d,
    DrawingTorus2d,
    GraphAdapter,
    Rng,
)
from ._core_distance_matrix import DistanceMatrix

__all__ = [
    "AmgCgSolver",
    "CgSolver",
    "ClassicalMds",
    "EigendecompositionResult",
    "Ic0CgSolver",
    "JacobiCgSolver",
    "KamadaKawai",
    "OverwrapRemoval",
    "PivotMds",
    "RdMds",
    "StressMajorization",
]

@final
class AmgCgSolver:
    def __new__(
        cls, max_iterations: int = 100, tolerance: float = 1e-4
    ) -> AmgCgSolver: ...

@final
class CgSolver:
    def __new__(
        cls, max_iterations: int = 100, tolerance: float = 1e-4
    ) -> CgSolver: ...

@final
class ClassicalMds:
    def __new__(cls, graph: GraphAdapter, f: Callable[[int], float]) -> ClassicalMds: ...
    @staticmethod
    def new_with_distance_matrix(d: Any) -> ClassicalMds: ...
    def run(self, d: int) -> DrawingEuclidean: ...
    def run_2d(self) -> DrawingEuclidean2d: ...
    @property
    def eps(self) -> float: ...
    @eps.setter
    def eps(self, value: float) -> None: ...

@final
class EigendecompositionResult:
    @property
    def eigenvectors(self) -> Array2: ...
    @property
    def eigenvalues(self) -> Array1: ...
    @property
    def cg_iterations(self) -> list[int]: ...
    @property
    def power_iterations(self) -> list[int]: ...

@final
class Ic0CgSolver:
    def __new__(
        cls, max_iterations: int = 100, tolerance: float = 1e-4
    ) -> Ic0CgSolver: ...

@final
class JacobiCgSolver:
    def __new__(
        cls, max_iterations: int = 100, tolerance: float = 1e-4
    ) -> JacobiCgSolver: ...

@final
class KamadaKawai:
    def __new__(cls, graph: GraphAdapter, f: Callable[[int], float]) -> KamadaKawai: ...
    def select_node(self, drawing: DrawingEuclidean2d) -> int | None: ...
    def apply_to_node(self, m: int, drawing: DrawingEuclidean2d) -> None: ...
    def run(self, drawing: DrawingEuclidean2d) -> None: ...
    @property
    def eps(self) -> float: ...
    @eps.setter
    def eps(self, value: float) -> None: ...

@final
class OverwrapRemoval:
    def __new__(cls, graph: GraphAdapter, f: Callable[[int], float]) -> OverwrapRemoval: ...
    def apply_with_drawing_euclidean_2d(
        self, drawing: DrawingEuclidean2d
    ) -> None: ...
    def apply_with_drawing_euclidean(self, drawing: DrawingEuclidean) -> None: ...
    def apply_with_drawing_hyperbolic_2d(
        self, drawing: DrawingHyperbolic2d
    ) -> None: ...
    def apply_with_drawing_spherical_2d(
        self, drawing: DrawingSpherical2d
    ) -> None: ...
    def apply_with_drawing_torus_2d(self, drawing: DrawingTorus2d) -> None: ...
    @property
    def strength(self) -> float: ...
    @strength.setter
    def strength(self, value: float) -> None: ...
    @property
    def iterations(self) -> int: ...
    @iterations.setter
    def iterations(self, value: int) -> None: ...
    @property
    def min_distance(self) -> float: ...
    @min_distance.setter
    def min_distance(self, value: float) -> None: ...

@final
class PivotMds:
    def __new__(
        cls, graph: GraphAdapter, f: Callable[[int], float], pivot: list[int]
    ) -> PivotMds: ...
    @staticmethod
    def new_with_distance_matrix(d: Any) -> PivotMds: ...
    def run(self, d: int) -> DrawingEuclidean: ...
    def run_2d(self) -> DrawingEuclidean2d: ...
    @property
    def eps(self) -> float: ...
    @eps.setter
    def eps(self, value: float) -> None: ...

@final
class RdMds:
    def __new__(cls) -> RdMds: ...
    def d(self, d: int) -> RdMds: ...
    def shift(self, shift: float) -> RdMds: ...
    def eigenvalue_max_iterations(
        self, eigenvalue_max_iterations: int
    ) -> RdMds: ...
    def eigenvalue_tolerance(self, eigenvalue_tolerance: float) -> RdMds: ...
    def embedding(
        self,
        graph: GraphAdapter,
        length: Callable[[int], float],
        solver: CgSolver | JacobiCgSolver | Ic0CgSolver | AmgCgSolver,
        rng: Rng,
    ) -> Array2: ...
    def embedding_symmetric_normalized(
        self,
        graph: GraphAdapter,
        length: Callable[[int], float],
        solver: CgSolver | JacobiCgSolver | Ic0CgSolver | AmgCgSolver,
        rng: Rng,
    ) -> Array2: ...
    def embedding_random_walk_normalized(
        self,
        graph: GraphAdapter,
        length: Callable[[int], float],
        solver: CgSolver | JacobiCgSolver | Ic0CgSolver | AmgCgSolver,
        rng: Rng,
    ) -> Array2: ...
    def eigendecomposition(
        self,
        graph: GraphAdapter,
        length: Callable[[int], float],
        solver: CgSolver | JacobiCgSolver | Ic0CgSolver | AmgCgSolver,
        rng: Rng,
    ) -> EigendecompositionResult: ...
    def eigendecomposition_symmetric_normalized(
        self,
        graph: GraphAdapter,
        length: Callable[[int], float],
        solver: CgSolver | JacobiCgSolver | Ic0CgSolver | AmgCgSolver,
        rng: Rng,
    ) -> EigendecompositionResult: ...
    def eigendecomposition_random_walk_normalized(
        self,
        graph: GraphAdapter,
        length: Callable[[int], float],
        solver: CgSolver | JacobiCgSolver | Ic0CgSolver | AmgCgSolver,
        rng: Rng,
    ) -> EigendecompositionResult: ...

@final
class StressMajorization:
    def __new__(
        cls,
        graph: GraphAdapter,
        drawing: DrawingEuclidean2d,
        f: Any = None,
    ) -> StressMajorization: ...
    @classmethod
    def with_distance_matrix(
        cls, drawing: DrawingEuclidean2d, distance_matrix: DistanceMatrix
    ) -> StressMajorization: ...
    def apply(self, drawing: DrawingEuclidean2d) -> float: ...
    def run(self, drawing: DrawingEuclidean2d) -> None: ...
    def update_weight(
        self, f: Callable[[int, int, float, float], float]
    ) -> None: ...
    def update_weight_from_numpy(self, w: npt.NDArray[np.float64]) -> None: ...
    @property
    def epsilon(self) -> float: ...
    @epsilon.setter
    def epsilon(self, value: float) -> None: ...
    @property
    def max_iterations(self) -> int: ...
    @max_iterations.setter
    def max_iterations(self, value: int) -> None: ...
