from __future__ import annotations

from typing import Any, Sequence, final

import numpy as np
import numpy.typing as npt

from ._core_core import CsrGraph, Layout2d
from ._core_types import _IndexArrayLike, _WeightArrayLike

__all__ = [
    "SeparationConstraint",
    "assign_layers",
    "cycle_edges",
    "generate_layered_constraints",
    "project_1d",
    "project_clustered_rectangle_no_overlap_2d",
    "project_layout_axis",
    "project_rectangle_no_overlap_2d",
    "remove_cycles",
]

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

def assign_layers(
    graph: CsrGraph, *, method: str = "longest_path"
) -> npt.NDArray[np.uint32]: ...

def cycle_edges(graph: CsrGraph) -> npt.NDArray[np.uint32]: ...

def generate_layered_constraints(
    graph: CsrGraph,
    layers: _IndexArrayLike,
    min_layer_distance: float = 1.0,
) -> list[SeparationConstraint]: ...

def project_1d(
    coords: _WeightArrayLike,
    constraints: Sequence[SeparationConstraint],
) -> npt.NDArray[Any]: ...

def project_clustered_rectangle_no_overlap_2d(
    layout: Layout2d,
    cluster_ids: _IndexArrayLike,
    half_widths: _WeightArrayLike,
    half_heights: _WeightArrayLike,
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

def remove_cycles(graph: CsrGraph) -> CsrGraph: ...
