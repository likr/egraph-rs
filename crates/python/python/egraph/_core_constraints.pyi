from __future__ import annotations

from typing import Any, Sequence, final

import numpy as np
import numpy.typing as npt

from ._core_core import CsrGraph, Layout2d, LayoutNd
from ._core_types import _DtypeLike, _IndexArrayLike, _WeightArrayLike

__all__ = [
    "SeparationConstraint",
    "TwinDeclutterer",
    "assign_layers",
    "cycle_edges",
    "declutter_twins_2d",
    "generate_layered_constraints",
    "project_1d",
    "project_clustered_rectangle_no_overlap_2d",
    "project_layout_axis",
    "project_rectangle_no_overlap_2d",
    "remove_cycles",
    "transitive_reduction",
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

@final
class TwinDeclutterer:
    def __new__(
        cls,
        *,
        eps_rel: float = 1e-6,
        r_frac: float = 0.35,
        r_cap_rel: float = 0.003,
        min_pair_rel: float = 5e-7,
        iterations: int = 4,
        dtype: _DtypeLike = "float64",
    ) -> TwinDeclutterer: ...
    def step(self, layout: Layout2d) -> str: ...
    def run_steps(self, layout: Layout2d, steps: int) -> str: ...
    def run(self, layout: Layout2d) -> str: ...
    @property
    def status(self) -> str: ...
    @property
    def iteration(self) -> int: ...
    @property
    def total_iterations(self) -> int: ...
    @property
    def is_finished(self) -> bool: ...
    @property
    def last_cluster_count(self) -> int: ...
    @property
    def last_moved_count(self) -> int: ...
    @property
    def eps_rel(self) -> float: ...
    @property
    def r_frac(self) -> float: ...
    @property
    def r_cap_rel(self) -> float: ...
    @property
    def min_pair_rel(self) -> float: ...
    @property
    def dtype(self) -> str: ...
    def __repr__(self) -> str: ...

def assign_layers(
    graph: CsrGraph, *, method: str = "longest_path"
) -> npt.NDArray[np.uint32]: ...

def cycle_edges(graph: CsrGraph) -> npt.NDArray[np.uint32]: ...

def declutter_twins_2d(
    layout: Layout2d,
    *,
    eps_rel: float = 1e-6,
    r_frac: float = 0.35,
    r_cap_rel: float = 0.003,
    min_pair_rel: float = 5e-7,
    iterations: int = 4,
) -> int: ...

def generate_layered_constraints(
    graph: CsrGraph,
    layers: _IndexArrayLike,
    min_layer_distance: float = 1.0,
) -> list[SeparationConstraint]: ...

def project_1d(
    coords: npt.NDArray[Any],
    constraints: Sequence[SeparationConstraint],
) -> None: ...

def project_clustered_rectangle_no_overlap_2d(
    layout: Layout2d,
    labels: _IndexArrayLike,
    half_widths: _WeightArrayLike,
    half_heights: _WeightArrayLike,
) -> None: ...

def project_layout_axis(
    layout: Layout2d | LayoutNd,
    axis: int,
    constraints: Sequence[SeparationConstraint],
) -> None: ...

def project_rectangle_no_overlap_2d(
    layout: Layout2d,
    half_widths: _WeightArrayLike,
    half_heights: _WeightArrayLike,
) -> None: ...

def remove_cycles(graph: CsrGraph) -> CsrGraph: ...

def transitive_reduction(graph: CsrGraph) -> CsrGraph: ...
