from __future__ import annotations

from typing import Any, final

import numpy as np
import numpy.typing as npt

from ._core_core import CsrGraph, Layout2d
from ._core_solvers_sgd import PacmapGraphSolver
from ._core_types import SolverStatus, _DtypeLike, _IndexArrayLike, _WeightArrayLike

__all__ = [
    "BackboneResult",
    "BackboneSolver",
    "ClusterRefineSolver",
    "CoarseBackbone",
    "CoarseBackboneResult",
    "CoarseBackboneStats",
    "GibBox",
    "GroupInABoxSolver",
    "coarse_backbone_init",
    "extract_quadrilateral_backbone",
    "pacmap_graph_with_backbone",
    "refine_cluster_layout",
    "squarified_treemap",
]

@final
class BackboneResult:
    @property
    def umst_edges(self) -> npt.NDArray[np.uint32]: ...
    @property
    def backbone_edges(self) -> npt.NDArray[np.uint32]: ...
    @property
    def simmelian_weights(self) -> npt.NDArray[np.float64]: ...
    @property
    def edge_weights(self) -> npt.NDArray[np.float64]: ...
    @property
    def backbone_csr(self) -> CsrGraph: ...
    def to_dict(self) -> dict[str, Any]: ...
    def keys(self) -> list[str]: ...
    def __getitem__(self, key: str, /) -> Any: ...
    def __contains__(self, key: object, /) -> bool: ...
    def __repr__(self) -> str: ...

@final
class GibBox:
    @property
    def cluster_id(self) -> int: ...
    @property
    def x_min(self) -> float: ...
    @property
    def x_max(self) -> float: ...
    @property
    def y_min(self) -> float: ...
    @property
    def y_max(self) -> float: ...
    @property
    def node_count(self) -> int: ...
    @property
    def weight(self) -> float: ...
    def to_dict(self) -> dict[str, Any]: ...
    def keys(self) -> list[str]: ...
    def __getitem__(self, key: str, /) -> Any: ...
    def __contains__(self, key: object, /) -> bool: ...
    def __repr__(self) -> str: ...

@final
class CoarseBackboneStats:
    @property
    def num_fine_nodes(self) -> int: ...
    @property
    def num_coarse_nodes(self) -> int: ...
    @property
    def num_coarse_edges(self) -> int: ...
    @property
    def louvain_levels(self) -> int: ...
    @property
    def mean_cluster_size(self) -> float: ...
    @property
    def max_cluster_size(self) -> int: ...
    @property
    def median_l_edge(self) -> float: ...
    @property
    def max_radius(self) -> float: ...
    @property
    def coarse_overlap_pairs_initial(self) -> int: ...
    @property
    def coarse_overlap_pairs_final(self) -> int: ...
    @property
    def scale_factor_applied(self) -> float: ...
    def to_dict(self) -> dict[str, Any]: ...
    def keys(self) -> list[str]: ...
    def __getitem__(self, key: str, /) -> Any: ...
    def __contains__(self, key: object, /) -> bool: ...
    def __repr__(self) -> str: ...

@final
class CoarseBackboneResult:
    @property
    def fine_layout(self) -> Layout2d: ...
    @property
    def coarse_layout(self) -> Layout2d: ...
    @property
    def coarse_graph(self) -> CsrGraph: ...
    @property
    def node_to_coarse(self) -> npt.NDArray[np.uint32]: ...
    @property
    def coarse_sizes(self) -> npt.NDArray[np.uint32]: ...
    @property
    def radii(self) -> npt.NDArray[np.float64]: ...
    @property
    def stats(self) -> CoarseBackboneStats: ...
    def to_dict(self) -> dict[str, Any]: ...
    def keys(self) -> list[str]: ...
    def __getitem__(self, key: str, /) -> Any: ...
    def __contains__(self, key: object, /) -> bool: ...
    def __repr__(self) -> str: ...

@final
class CoarseBackbone:
    def __new__(
        cls,
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
    ) -> CoarseBackbone: ...
    @property
    def num_nodes(self) -> int: ...
    @property
    def dtype(self) -> str: ...
    def generate(self) -> Layout2d: ...
    def __repr__(self) -> str: ...

@final
class BackboneSolver:
    def __new__(
        cls,
        graph: CsrGraph,
        *,
        retain_ratio: float = 0.2,
        umst_weight_scale: float = 1.0,
        non_backbone_weight_scale: float = 0.0,
        use_weights: bool = True,
        layout_mode: str = "stress",
        iterations: int = 100,
        seed: int = 42,
        dtype: _DtypeLike = "float64",
    ) -> BackboneSolver: ...
    def step(self, layout: Layout2d) -> SolverStatus: ...
    def run_steps(self, layout: Layout2d, steps: int) -> SolverStatus: ...
    def run(self, layout: Layout2d) -> SolverStatus: ...
    def backbone_result(self) -> BackboneResult: ...
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
class ClusterRefineSolver:
    def __new__(
        cls,
        labels: _IndexArrayLike,
        *,
        method: str = "kmeans",
        max_iterations: int = 10,
        kmeans_n_init: int = 5,
        kmeans_max_iterations: int = 100,
        shift_step: float = 1.0,
        margin: float = 0.1,
        seed: int = 42,
        dtype: _DtypeLike = "float64",
    ) -> ClusterRefineSolver: ...
    @staticmethod
    def from_graph(
        graph: CsrGraph,
        labels: _IndexArrayLike,
        *,
        method: str = "kmeans",
        max_iterations: int = 10,
        kmeans_n_init: int = 5,
        kmeans_max_iterations: int = 100,
        shift_step: float = 1.0,
        margin: float = 0.1,
        seed: int = 42,
        dtype: _DtypeLike = "float64",
    ) -> ClusterRefineSolver: ...
    def step(self, layout: Layout2d) -> SolverStatus: ...
    def run_steps(self, layout: Layout2d, steps: int) -> SolverStatus: ...
    def run(self, layout: Layout2d) -> SolverStatus: ...
    @property
    def last_moved_count(self) -> int: ...
    @property
    def num_clusters(self) -> int: ...
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
class GroupInABoxSolver:
    def __new__(
        cls,
        graph: CsrGraph,
        labels: _IndexArrayLike,
        *,
        box_mode: str = "treemap",
        intra_mode: str = "fr",
        total_width: float = 100.0,
        total_height: float = 100.0,
        box_padding: float = 0.05,
        intra_iterations: int = 100,
        fd_iterations: int = 100,
        fit_bounds: bool = True,
        cluster_weights: _WeightArrayLike | None = None,
        seed: int = 42,
        dtype: _DtypeLike = "float64",
    ) -> GroupInABoxSolver: ...
    def step(self, layout: Layout2d) -> SolverStatus: ...
    def run_steps(self, layout: Layout2d, steps: int) -> SolverStatus: ...
    def run(self, layout: Layout2d) -> SolverStatus: ...
    def boxes(self) -> list[GibBox]: ...
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
) -> CoarseBackboneResult: ...

def pacmap_graph_with_backbone(
    graph: CsrGraph,
    *,
    n_neighbors: int = 10,
    mn_ratio: float = 0.5,
    fp_ratio: float = 2.0,
    phase_iterations: tuple[int, int, int] = (100, 100, 250),
    learning_rate: float = 1.0,
    keep_all_1hop: bool = False,
    max_1hop_per_node: int | None = None,
    degree_weight_beta: float = 0.0,
    num_landmarks: int = 16,
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
) -> tuple[PacmapGraphSolver, Layout2d, CoarseBackboneResult]: ...

def extract_quadrilateral_backbone(
    graph: CsrGraph,
    *,
    retain_ratio: float = 0.2,
    umst_weight_scale: float = 1.0,
    non_backbone_weight_scale: float = 0.0,
    use_weights: bool = True,
) -> BackboneResult: ...

def refine_cluster_layout(
    layout: Layout2d,
    labels: _IndexArrayLike,
    *,
    graph: CsrGraph | None = None,
    method: str = "kmeans",
    max_iterations: int = 10,
    kmeans_n_init: int = 5,
    kmeans_max_iterations: int = 100,
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
