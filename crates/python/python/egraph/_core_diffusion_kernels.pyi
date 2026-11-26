from __future__ import annotations

from typing import Any, final

from ._core_core import Array1, Array2, GraphAdapter, Rng
from ._core_distance_matrix import Laplacian

__all__ = [
    "BandPassDiffusionKernel",
    "BandPassDiffusionKernelBuilder",
    "DiffusionKernel",
    "LowRankDiffusionKernel",
    "LowRankDiffusionKernelBuilder",
    "LowRankMultiscaleDiffusionKernel",
    "LowRankMultiscaleDiffusionKernelBuilder",
    "MultiscaleDiffusionKernel",
    "PivotedBandPassDiffusionKernel",
    "PivotedBandPassDiffusionKernelBuilder",
    "PivotedDiffusionKernel",
    "PivotedMultiscaleDiffusionKernel",
]

@final
class BandPassDiffusionKernel:
    def __new__(
        cls,
        laplacian: Laplacian,
        terms: list[tuple[float, float]],
        degree: int,
        rng: Rng | None = None,
        lambda_max: float | None = None,
        off_diagonal_floor: float = 0.0,
    ) -> BandPassDiffusionKernel: ...
    @staticmethod
    def new_with_lambda_max(
        laplacian: Laplacian,
        terms: list[tuple[float, float]],
        degree: int,
        lambda_max: float,
        rng: Rng | None = None,
        off_diagonal_floor: float = 0.0,
    ) -> BandPassDiffusionKernel: ...
    @staticmethod
    def dog(
        laplacian: Laplacian,
        t1: float,
        t2: float,
        alpha: float,
        beta_floor: float,
        degree: int,
        rng: Rng | None = None,
        lambda_max: float | None = None,
    ) -> BandPassDiffusionKernel: ...
    @staticmethod
    def multiscale_heat(
        laplacian: Laplacian,
        terms: list[tuple[float, float]],
        degree: int,
        rng: Rng | None = None,
        lambda_max: float | None = None,
        off_diagonal_floor: float = 0.0,
    ) -> BandPassDiffusionKernel: ...
    def get(self, i: int, j: int) -> float: ...
    def n(self) -> int: ...

@final
class BandPassDiffusionKernelBuilder:
    def __new__(
        cls,
        terms: list[tuple[float, float]] | None = None,
        degree: int = 30,
        lambda_max: float | None = None,
        off_diagonal_floor: float = 0.0,
    ) -> BandPassDiffusionKernelBuilder: ...
    @staticmethod
    def dog(
        t1: float,
        t2: float,
        alpha: float,
        beta_floor: float,
        degree: int = 30,
        lambda_max: float | None = None,
    ) -> BandPassDiffusionKernelBuilder: ...
    @staticmethod
    def multiscale_heat(
        terms: list[tuple[float, float]],
        degree: int = 30,
        lambda_max: float | None = None,
        off_diagonal_floor: float = 0.0,
    ) -> BandPassDiffusionKernelBuilder: ...
    def terms(
        self, terms: list[tuple[float, float]]
    ) -> BandPassDiffusionKernelBuilder: ...
    def degree(self, degree: int) -> BandPassDiffusionKernelBuilder: ...
    def lambda_max(
        self, lambda_max: float | None
    ) -> BandPassDiffusionKernelBuilder: ...
    def off_diagonal_floor(
        self, floor: float
    ) -> BandPassDiffusionKernelBuilder: ...
    def build(
        self, laplacian: Laplacian, rng: Rng | None = None
    ) -> BandPassDiffusionKernel: ...

@final
class DiffusionKernel:
    def __new__(
        cls, laplacian: Laplacian, t: float, degree: int, rng: Rng
    ) -> DiffusionKernel: ...
    @staticmethod
    def new_with_lambda_max(
        laplacian: Laplacian, t: float, degree: int, lambda_max: float, rng: Rng
    ) -> DiffusionKernel: ...
    def get(self, i: int, j: int) -> float: ...
    def n(self) -> int: ...

@final
class LowRankDiffusionKernel:
    def __new__(
        cls, t: float, eigenvalues: Array1, eigenvectors: Array2
    ) -> LowRankDiffusionKernel: ...
    def get(self, i: int, j: int) -> float: ...
    def n(self) -> int: ...

@final
class LowRankDiffusionKernelBuilder:
    def __new__(cls) -> LowRankDiffusionKernelBuilder: ...
    def t(self, t: float) -> LowRankDiffusionKernelBuilder: ...
    def rank(self, rank: int) -> LowRankDiffusionKernelBuilder: ...
    def shift(self, shift: float) -> LowRankDiffusionKernelBuilder: ...
    def eigenvalue_max_iterations(
        self, eigenvalue_max_iterations: int
    ) -> LowRankDiffusionKernelBuilder: ...
    def cg_max_iterations(
        self, cg_max_iterations: int
    ) -> LowRankDiffusionKernelBuilder: ...
    def eigenvalue_tolerance(
        self, eigenvalue_tolerance: float
    ) -> LowRankDiffusionKernelBuilder: ...
    def cg_tolerance(self, cg_tolerance: float) -> LowRankDiffusionKernelBuilder: ...
    def build_unnormalized_laplacian(
        self,
        graph: GraphAdapter,
        length: Any = None,
        rng: Rng | None = None,
    ) -> LowRankDiffusionKernel: ...
    def build_symmetric_normalized_laplacian(
        self,
        graph: GraphAdapter,
        length: Any = None,
        rng: Rng | None = None,
    ) -> LowRankDiffusionKernel: ...

@final
class LowRankMultiscaleDiffusionKernel:
    def __new__(
        cls, alpha: float, eigenvalues: Array1, eigenvectors: Array2
    ) -> LowRankMultiscaleDiffusionKernel: ...
    def get(self, i: int, j: int) -> float: ...
    def n(self) -> int: ...

@final
class LowRankMultiscaleDiffusionKernelBuilder:
    def __new__(cls) -> LowRankMultiscaleDiffusionKernelBuilder: ...
    def alpha(self, alpha: float) -> LowRankMultiscaleDiffusionKernelBuilder: ...
    def rank(self, rank: int) -> LowRankMultiscaleDiffusionKernelBuilder: ...
    def shift(self, shift: float) -> LowRankMultiscaleDiffusionKernelBuilder: ...
    def eigenvalue_max_iterations(
        self, eigenvalue_max_iterations: int
    ) -> LowRankMultiscaleDiffusionKernelBuilder: ...
    def cg_max_iterations(
        self, cg_max_iterations: int
    ) -> LowRankMultiscaleDiffusionKernelBuilder: ...
    def eigenvalue_tolerance(
        self, eigenvalue_tolerance: float
    ) -> LowRankMultiscaleDiffusionKernelBuilder: ...
    def cg_tolerance(
        self, cg_tolerance: float
    ) -> LowRankMultiscaleDiffusionKernelBuilder: ...
    def build_unnormalized_laplacian(
        self,
        graph: GraphAdapter,
        length: Any = None,
        rng: Rng | None = None,
    ) -> LowRankMultiscaleDiffusionKernel: ...
    def build_symmetric_normalized_laplacian(
        self,
        graph: GraphAdapter,
        length: Any = None,
        rng: Rng | None = None,
    ) -> LowRankMultiscaleDiffusionKernel: ...

@final
class MultiscaleDiffusionKernel:
    def __new__(cls, laplacian: Laplacian, alpha: float) -> MultiscaleDiffusionKernel: ...
    def get(self, i: int, j: int) -> float: ...
    def n(self) -> int: ...

@final
class PivotedBandPassDiffusionKernel:
    def __new__(
        cls,
        laplacian: Laplacian,
        terms: list[tuple[float, float]],
        degree: int,
        pivots: list[int],
        rng: Rng | None = None,
        num_hutchinson: int | None = ...,
        lambda_max: float | None = None,
        off_diagonal_floor: float = 0.0,
    ) -> PivotedBandPassDiffusionKernel: ...
    @staticmethod
    def new_with_lambda_max(
        laplacian: Laplacian,
        terms: list[tuple[float, float]],
        degree: int,
        pivots: list[int],
        lambda_max: float,
        rng: Rng | None = None,
        num_hutchinson: int | None = ...,
        off_diagonal_floor: float = 0.0,
    ) -> PivotedBandPassDiffusionKernel: ...
    @staticmethod
    def dog(
        laplacian: Laplacian,
        t1: float,
        t2: float,
        alpha: float,
        beta_floor: float,
        degree: int,
        pivots: list[int],
        rng: Rng | None = None,
        num_hutchinson: int | None = ...,
        lambda_max: float | None = None,
    ) -> PivotedBandPassDiffusionKernel: ...
    @staticmethod
    def multiscale_heat(
        laplacian: Laplacian,
        terms: list[tuple[float, float]],
        degree: int,
        pivots: list[int],
        rng: Rng | None = None,
        num_hutchinson: int | None = ...,
        lambda_max: float | None = None,
        off_diagonal_floor: float = 0.0,
    ) -> PivotedBandPassDiffusionKernel: ...
    def pivots(self) -> list[int]: ...
    def get_from_pivot(self, pivot_idx: int, j: int) -> float: ...
    def diagonal(self, j: int) -> float | None: ...
    def n(self) -> int: ...

@final
class PivotedBandPassDiffusionKernelBuilder:
    def __new__(
        cls,
        terms: list[tuple[float, float]] | None = None,
        degree: int = 30,
        pivots: list[int] | None = None,
        lambda_max: float | None = None,
        num_hutchinson: int | None = ...,
        off_diagonal_floor: float = 0.0,
    ) -> PivotedBandPassDiffusionKernelBuilder: ...
    @staticmethod
    def dog(
        t1: float,
        t2: float,
        alpha: float,
        beta_floor: float,
        degree: int = 30,
        pivots: list[int] | None = None,
        lambda_max: float | None = None,
        num_hutchinson: int | None = ...,
    ) -> PivotedBandPassDiffusionKernelBuilder: ...
    @staticmethod
    def multiscale_heat(
        terms: list[tuple[float, float]],
        degree: int = 30,
        pivots: list[int] | None = None,
        lambda_max: float | None = None,
        num_hutchinson: int | None = ...,
        off_diagonal_floor: float = 0.0,
    ) -> PivotedBandPassDiffusionKernelBuilder: ...
    def terms(
        self, terms: list[tuple[float, float]]
    ) -> PivotedBandPassDiffusionKernelBuilder: ...
    def degree(self, degree: int) -> PivotedBandPassDiffusionKernelBuilder: ...
    def pivots(
        self, pivots: list[int]
    ) -> PivotedBandPassDiffusionKernelBuilder: ...
    def lambda_max(
        self, lambda_max: float | None
    ) -> PivotedBandPassDiffusionKernelBuilder: ...
    def num_hutchinson(
        self, num_hutchinson: int | None
    ) -> PivotedBandPassDiffusionKernelBuilder: ...
    def off_diagonal_floor(
        self, floor: float
    ) -> PivotedBandPassDiffusionKernelBuilder: ...
    def build(
        self, laplacian: Laplacian, rng: Rng | None = None
    ) -> PivotedBandPassDiffusionKernel: ...

@final
class PivotedDiffusionKernel:
    def __new__(
        cls,
        laplacian: Laplacian,
        t: float,
        degree: int,
        pivots: list[int],
        rng: Rng,
        num_hutchinson: int | None = None,
    ) -> PivotedDiffusionKernel: ...
    @staticmethod
    def new_with_lambda_max(
        laplacian: Laplacian,
        t: float,
        degree: int,
        pivots: list[int],
        lambda_max: float,
        rng: Rng,
        num_hutchinson: int | None = None,
    ) -> PivotedDiffusionKernel: ...
    def pivots(self) -> list[int]: ...
    def get_from_pivot(self, pivot_idx: int, j: int) -> float: ...
    def diagonal(self, j: int) -> float | None: ...

@final
class PivotedMultiscaleDiffusionKernel:
    def __new__(
        cls, laplacian: Laplacian, alpha: float, pivots: list[int]
    ) -> PivotedMultiscaleDiffusionKernel: ...
    def pivots(self) -> list[int]: ...
    def get_from_pivot(self, pivot_idx: int, j: int) -> float: ...
    def diagonal(self, j: int) -> float | None: ...
