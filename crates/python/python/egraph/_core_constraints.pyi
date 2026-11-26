from __future__ import annotations

from typing import Any, Callable, Sequence, final, overload

import numpy as np
import numpy.typing as npt

from ._core_core import CsrGraph, DrawingEuclidean2d, GraphAdapter, Layout2d
from ._core_distance_matrix import DistanceMatrix
from ._core_types import _IndexArrayLike, _WeightArrayLike

__all__ = [
    "Constraint",
    "LongestPath",
    "SeparationConstraint",
    "WeightedEdgeLength",
    "all_sources_bfs",
    "all_sources_dijkstra",
    "assign_layers",
    "cycle_edges",
    "generate_layered_constraints",
    "project_1d",
    "project_clustered_rectangle_no_overlap_2d",
    "project_clustered_rectangle_no_overlap_constraints",
    "project_layout_axis",
    "project_rectangle_no_overlap_2d",
    "project_rectangle_no_overlap_constraints_2d",
    "remove_cycle",
    "remove_cycles",
    "triangulation",
    "warshall_floyd",
]

@final
class Constraint:
    def __new__(cls, left: int, right: int, gap: float) -> Constraint: ...
    @property
    def left(self) -> int: ...
    @property
    def right(self) -> int: ...
    @property
    def gap(self) -> float: ...

@final
class LongestPath:
    def __new__(cls) -> LongestPath: ...
    def assign_layers(self, graph: GraphAdapter) -> dict[int, int]: ...

@final
class SeparationConstraint:
    def __new__(cls, left: int, right: int, gap: float) -> SeparationConstraint: ...
    @property
    def left(self) -> int: ...
    @property
    def right(self) -> int: ...
    @property
    def gap(self) -> float: ...
    def __repr__(self) -> str: ...

@final
class WeightedEdgeLength:
    def __new__(cls, graph: GraphAdapter) -> WeightedEdgeLength: ...
    def __call__(self, edge_index: int) -> int: ...

def all_sources_bfs(
    graph: GraphAdapter, unit_edge_length: float
) -> DistanceMatrix: ...

def all_sources_dijkstra(
    graph: GraphAdapter, f: Any = None
) -> DistanceMatrix: ...

def assign_layers(
    graph: CsrGraph, *, method: str = "longest_path"
) -> npt.NDArray[np.uint32]: ...

@overload
def cycle_edges(graph: GraphAdapter) -> list[tuple[int, int]]: ...

@overload
def cycle_edges(graph: CsrGraph) -> npt.NDArray[np.uint32]: ...

@overload
def generate_layered_constraints(
    graph: GraphAdapter, layers_or_gap: float
) -> list[Constraint]: ...

@overload
def generate_layered_constraints(
    graph: GraphAdapter, *, gap: float
) -> list[Constraint]: ...

@overload
def generate_layered_constraints(
    graph: CsrGraph,
    layers_or_gap: _IndexArrayLike | None = None,
    min_layer_distance: float = ...,
    *,
    layers: _IndexArrayLike | None = None,
    gap: float | None = None,
) -> list[SeparationConstraint]: ...

@overload
def project_1d(
    target: DrawingEuclidean2d, arg2: int, arg3: list[Constraint]
) -> None: ...

@overload
def project_1d(
    target: _WeightArrayLike,
    arg2: Sequence[SeparationConstraint],
    arg3: None = None,
) -> npt.NDArray[Any]: ...

def project_clustered_rectangle_no_overlap_2d(
    layout: Layout2d,
    cluster_ids: _IndexArrayLike,
    half_widths: _WeightArrayLike,
    half_heights: _WeightArrayLike,
) -> None: ...

def project_clustered_rectangle_no_overlap_constraints(
    graph: GraphAdapter,
    drawing: DrawingEuclidean2d,
    cluster_id_fn: Callable[[int], int],
    size_fn: Callable[[int, int], float],
) -> None: ...

def project_layout_axis(
    layout: Layout2d,
    axis: int,
    constraints: Sequence[SeparationConstraint],
) -> None: ...

def project_rectangle_no_overlap_2d(
    layout: Layout2d,
    half_widths: _WeightArrayLike,
    half_heights: _WeightArrayLike,
) -> None: ...

def project_rectangle_no_overlap_constraints_2d(
    drawing: DrawingEuclidean2d, size_fn: Callable[[int, int], float]
) -> None: ...

def remove_cycle(graph: GraphAdapter) -> None: ...

def remove_cycles(graph: CsrGraph) -> CsrGraph: ...

def triangulation(drawing: DrawingEuclidean2d) -> GraphAdapter: ...

def warshall_floyd(
    graph: GraphAdapter, f: Any = None
) -> DistanceMatrix: ...
