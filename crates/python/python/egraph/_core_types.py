from __future__ import annotations

from typing import Any, Literal, Protocol, TypedDict, TypeVar, runtime_checkable

import numpy.typing as npt

TargetKind = Literal["layout2d"] | Literal["layout3d"] | Literal["layout_nd"] | Literal["manifold2d"]
EuclideanTargetKind = Literal["layout2d"] | Literal["layout3d"] | Literal["layout_nd"]
ManifoldKindLiteral = Literal["hyperbolic"] | Literal["hyperbolic_lorentz"] | Literal["spherical"] | Literal["torus"]
BackendKind = (
    Literal["auto"]
    | Literal["exact"]
    | Literal["barnes_hut"]
    | Literal["grid"]
    | Literal["ib_fft"]
    | Literal["linear_fft"]
    | Literal["fit_fft"]
)
SolverStatus = Literal["initialized"] | Literal["running"] | Literal["converged"] | Literal["completed"] | Literal["failed"]

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
    def status(self) -> str: ...
    @property
    def iteration(self) -> int: ...
    @property
    def total_iterations(self) -> int | None: ...
    @property
    def is_finished(self) -> bool: ...
    def step(self, layout: _L_contra) -> str: ...
    def run_steps(self, layout: _L_contra, steps: int) -> str: ...
    def run(self, layout: _L_contra) -> str: ...


@runtime_checkable
class LayoutGenerator(Protocol[_L_co]):
    def generate(self) -> _L_co: ...


@runtime_checkable
class LayoutTransform(Protocol[_L_contra]):
    def apply(self, layout: _L_contra) -> None: ...


__all__ = [
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
