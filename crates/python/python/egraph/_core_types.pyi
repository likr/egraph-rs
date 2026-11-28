from __future__ import annotations

from typing import Any, Literal, Protocol, Sequence, TypedDict, TypeVar, runtime_checkable

import numpy as np
import numpy.typing as npt

_IndexArrayLike = npt.NDArray[np.integer[Any]] | Sequence[int]
_EdgeArrayLike = npt.NDArray[np.integer[Any]] | Sequence[tuple[int, int]] | Sequence[Sequence[int]]
_WeightArrayLike = npt.NDArray[np.floating[Any]] | npt.NDArray[np.integer[Any]] | Sequence[float]
_CoordArrayLike = npt.NDArray[np.floating[Any]] | Sequence[Sequence[float]]
_MatrixArrayLike = npt.NDArray[np.floating[Any]] | npt.NDArray[np.integer[Any]] | Sequence[Sequence[float]]
_VectorArrayLike = (
    npt.NDArray[np.floating[Any]]
    | npt.NDArray[np.integer[Any]]
    | Sequence[float]
    | Sequence[Sequence[float]]
)
_DtypeLike = str | np.dtype[Any] | type[np.floating[Any]]

TargetKind = Literal["layout2d", "layout3d", "layout_nd", "manifold2d"]
EuclideanTargetKind = Literal["layout2d", "layout3d", "layout_nd"]
ManifoldKindLiteral = Literal["hyperbolic", "hyperbolic_lorentz", "spherical", "torus"]
BackendKind = Literal["auto", "exact", "barnes_hut", "grid", "ib_fft", "linear_fft", "fit_fft"]
SolverStatus = Literal["initialized", "running", "converged", "completed", "failed"]

_L_contra = TypeVar("_L_contra", contravariant=True)
_L_co = TypeVar("_L_co", covariant=True)

class CrossingPairDict(TypedDict):
    edge_a: int
    edge_b: int
    u1: int
    v1: int
    u2: int
    v2: int
    angle_rad: float
    cos_sq: float

class TorusCrossingDict(TypedDict):
    edge_a: int
    edge_b: int
    u1: int
    v1: int
    u2: int
    v2: int
    seg_a: list[int]
    seg_b: list[int]
    angle_rad: float
    cos_sq: float

@runtime_checkable
class LayoutStorage(Protocol):
    def __len__(self) -> int: ...
    @property
    def num_nodes(self) -> int: ...
    @property
    def dtype(self) -> str: ...
    def is_empty(self) -> bool: ...
    def is_all_finite(self) -> bool: ...
    def coords(self, copy: bool = False) -> npt.NDArray[Any]: ...

@runtime_checkable
class IterativeSolver(Protocol[_L_contra]):
    @property
    def status(self) -> SolverStatus: ...
    @property
    def iteration(self) -> int: ...
    @property
    def total_iterations(self) -> int | None: ...
    @property
    def is_finished(self) -> bool: ...
    def step(self, layout: _L_contra) -> SolverStatus: ...
    def run_steps(self, layout: _L_contra, steps: int) -> SolverStatus: ...
    def run(self, layout: _L_contra) -> SolverStatus: ...

@runtime_checkable
class LayoutGenerator(Protocol[_L_co]):
    def generate(self) -> _L_co: ...

@runtime_checkable
class LayoutTransform(Protocol[_L_contra]):
    def apply(self, layout: _L_contra) -> None: ...

__all__: list[str] = [
    "BackendKind",
    "CrossingPairDict",
    "EuclideanTargetKind",
    "IterativeSolver",
    "LayoutGenerator",
    "LayoutStorage",
    "LayoutTransform",
    "ManifoldKindLiteral",
    "SolverStatus",
    "TargetKind",
    "TorusCrossingDict",
]
