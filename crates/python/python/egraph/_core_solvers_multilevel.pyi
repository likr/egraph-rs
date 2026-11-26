from __future__ import annotations

from typing import Any, final

import numpy as np
import numpy.typing as npt

from ._core_core import CsrGraph, Layout2d
from ._core_types import _DtypeLike, _IndexArrayLike, _WeightArrayLike

__all__ = [
    "BackboneSolver",
    "ClusterRefineSolver",
    "GroupInABoxSolver",
    "coarse_backbone_init",
    "extract_quadrilateral_backbone",
    "refine_cluster_layout",
    "squarified_treemap",
]

@final
class BackboneSolver:
    def __new__(
        cls,
        graph: CsrGraph,
        *,
        retain_ratio: float = 0.2,
        umst_weight_scale: float = 1.0,
        non_backbone_weight_scale: float = 0.0,
        layout_mode: str = "stress",
        iterations: int = 100,
        seed: int = 42,
        dtype: _DtypeLike = "float64",
    ) -> BackboneSolver: ...
    def step(self, layout: Layout2d) -> str: ...
    def run_steps(self, layout: Layout2d, steps: int) -> str: ...
    def run(self, layout: Layout2d) -> str: ...
    def backbone_result(self) -> dict[str, Any]: ...
    @property
    def status(self) -> str: ...
    @property
    def iteration(self) -> int: ...
    @property
    def total_iterations(self) -> int: ...
    @property
    def is_finished(self) -> bool: ...
    @property
    def num_nodes(self) -> int: ...
    @property
    def dtype(self) -> str: ...
    def __repr__(self) -> str: ...

@final
class ClusterRefineSolver:
    def __new__(
        cls,
        cluster_labels: _IndexArrayLike,
        *,
        method: str = "kmeans",
        max_rounds: int = 10,
        kmeans_n_init: int = 5,
        kmeans_max_iters: int = 100,
        shift_step: float = 1.0,
        margin: float = 0.1,
        seed: int = 42,
        dtype: _DtypeLike = "float64",
    ) -> ClusterRefineSolver: ...
    @staticmethod
    def from_graph(
        graph: CsrGraph,
        cluster_labels: _IndexArrayLike,
        *,
        method: str = "kmeans",
        max_rounds: int = 10,
        kmeans_n_init: int = 5,
        kmeans_max_iters: int = 100,
        shift_step: float = 1.0,
        margin: float = 0.1,
        seed: int = 42,
        dtype: _DtypeLike = "float64",
    ) -> ClusterRefineSolver: ...
    def step(self, layout: Layout2d) -> str: ...
    def run_steps(self, layout: Layout2d, steps: int) -> str: ...
    def run(self, layout: Layout2d) -> str: ...
    @property
    def last_moved_count(self) -> int: ...
    @property
    def num_clusters(self) -> int: ...
    @property
    def status(self) -> str: ...
    @property
    def iteration(self) -> int: ...
    @property
    def total_iterations(self) -> int: ...
    @property
    def is_finished(self) -> bool: ...
    @property
    def num_nodes(self) -> int: ...
    @property
    def dtype(self) -> str: ...
    def __repr__(self) -> str: ...

@final
class GroupInABoxSolver:
    def __new__(
        cls,
        graph: CsrGraph,
        cluster_labels: _IndexArrayLike,
        *,
        box_mode: str = "treemap",
        intra_mode: str = "fr",
        total_width: float = 100.0,
        total_height: float = 100.0,
        box_padding: float = 0.05,
        intra_iterations: int = 100,
        fd_iterations: int = 100,
        cluster_weights: _WeightArrayLike | None = None,
        seed: int = 42,
        dtype: _DtypeLike = "float64",
    ) -> GroupInABoxSolver: ...
    def step(self, layout: Layout2d) -> str: ...
    def run_steps(self, layout: Layout2d, steps: int) -> str: ...
    def run(self, layout: Layout2d) -> str: ...
    def boxes(self) -> list[dict[str, Any]]: ...
    @property
    def status(self) -> str: ...
    @property
    def iteration(self) -> int: ...
    @property
    def total_iterations(self) -> int: ...
    @property
    def is_finished(self) -> bool: ...
    @property
    def num_nodes(self) -> int: ...
    @property
    def dtype(self) -> str: ...
    def __repr__(self) -> str: ...

def coarse_backbone_init(
    graph: CsrGraph,
    *,
    max_coarse_nodes: int = 10000,
    resolution: float = 1.0,
    resolution_decay: float = 0.5,
    max_passes: int = 100,
    min_modularity_gain: float = 1e-7,
    max_outer_rounds: int = 16,
    edge_length_mode: str = "conductance_inverse",
    distance_backend: str = "rhgh",
    alpha: float = 0.75,
    c0: float = 0.5,
    kappa: float = 1.0,
    gamma_w: float = 0.5,
    gamma_m: float = 0.5,
    mu_ov: float = 1.0,
    m_gap: float = 0.1,
    coarse_pivots: int = 64,
    coarse_neg_per_node: int = 16,
    coarse_t_max: int = 80,
    rhgh_gamma: float | None = None,
    rhgh_p: float | None = None,
    rcf_steps: int = 2,
    rcf_eta: float = 0.2,
    jitter_scale: float = 0.5,
    gamma_tether: float = 0.5,
    target_std: float | None = 1.0,
    robust_quantile: float = 0.95,
    seed: int = 42,
    dtype: _DtypeLike = "float64",
) -> dict[str, Any]: ...

def extract_quadrilateral_backbone(
    graph: CsrGraph,
    *,
    retain_ratio: float = 0.2,
    umst_weight_scale: float = 1.0,
    non_backbone_weight_scale: float = 0.0,
) -> dict[str, Any]: ...

def refine_cluster_layout(
    layout: Layout2d,
    cluster_labels: _IndexArrayLike,
    *,
    method: str = "kmeans",
    max_rounds: int = 10,
    kmeans_n_init: int = 5,
    kmeans_max_iters: int = 100,
    shift_step: float = 1.0,
    margin: float = 0.1,
    seed: int = 42,
) -> int: ...

def squarified_treemap(
    weights: _WeightArrayLike,
    *,
    width: float = 100.0,
    height: float = 100.0,
    x: float = 0.0,
    y: float = 0.0,
) -> npt.NDArray[np.float64]: ...
