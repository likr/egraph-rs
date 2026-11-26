from __future__ import annotations

from typing import Any, Callable, final

import numpy as np
import numpy.typing as npt

from ._core_core import CsrGraph, GraphAdapter, Layout2d
from ._core_neighborhood import DogVaradhanLandmarks
from ._core_types import _IndexArrayLike, _MatrixArrayLike, _WeightArrayLike

__all__ = [
    "LabelPropagation",
    "Louvain",
    "SpectralClustering",
    "align_clustered_centroids",
    "coarsen_graph",
    "compute_pivoted_rhgh_landmarks",
    "kmeans_plus_plus",
    "label_propagation",
    "leiden_communities",
    "louvain_coarsen",
    "louvain_communities",
    "project_clustered_layout",
    "project_clustered_layout_with_radii",
    "py_coarsen",
    "spectral_clustering",
]

@final
class LabelPropagation:
    def __new__(cls) -> LabelPropagation: ...
    def detect_communities(self, graph: GraphAdapter) -> dict[int, int]: ...

@final
class Louvain:
    def __new__(cls) -> Louvain: ...
    def detect_communities(self, graph: GraphAdapter) -> dict[int, int]: ...

@final
class SpectralClustering:
    def __new__(cls, k: int) -> SpectralClustering: ...
    def detect_communities(self, graph: GraphAdapter) -> dict[int, int]: ...

def align_clustered_centroids(
    fine_layout: Layout2d,
    coarse_layout: Layout2d,
    node_to_coarse: _IndexArrayLike,
    *,
    gamma_tether: float = 0.5,
) -> None: ...

def coarsen_graph(
    graph: CsrGraph,
    labels: _IndexArrayLike,
) -> tuple[CsrGraph, npt.NDArray[np.uint32], npt.NDArray[np.uint32], int]: ...

def compute_pivoted_rhgh_landmarks(
    graph: CsrGraph,
    *,
    gamma: float | None = None,
    p_exp: float | None = None,
    num_landmarks: int = 64,
    min_degree: int = 2,
    num_probes: int = 64,
    rel_tol: float = 1e-12,
    max_iterations: int = 500,
    normalize_1hop: bool = True,
    seed: int = 42,
) -> tuple[DogVaradhanLandmarks, dict[str, Any]]: ...

def kmeans_plus_plus(
    points: _MatrixArrayLike,
    k: int = 2,
    max_iters: int = 100,
    tol: float = 1e-4,
    n_init: int = 1,
    seed: int = 42,
) -> dict[str, Any]: ...

def label_propagation(
    graph: CsrGraph,
    max_iters: int = 100,
    seed: int = 42,
) -> npt.NDArray[np.uint32]: ...

def leiden_communities(
    graph: CsrGraph,
    *,
    objective: str = "modularity",
    resolution: float = 1.0,
    randomness: float = 0.01,
    max_passes: int = 100,
    min_quality_gain: float = 1e-7,
    seed: int = 42,
) -> dict[str, Any]: ...

def louvain_coarsen(
    graph: CsrGraph,
    *,
    max_coarse_nodes: int = 10000,
    resolution: float = 1.0,
    resolution_decay: float = 0.5,
    max_passes: int = 100,
    min_modularity_gain: float = 1e-7,
    max_outer_rounds: int = 16,
    seed: int = 42,
) -> dict[str, Any]: ...

def louvain_communities(
    graph: CsrGraph,
    max_passes: int = 100,
    resolution: float = 1.0,
    min_modularity_gain: float = 1e-7,
    seed: int = 42,
) -> dict[str, Any]: ...

def project_clustered_layout(
    coarse_layout: Layout2d,
    node_to_coarse: _IndexArrayLike,
    spread_radius: float = 0.1,
    seed: int = 42,
) -> Layout2d: ...

def project_clustered_layout_with_radii(
    coarse_layout: Layout2d,
    node_to_coarse: _IndexArrayLike,
    cluster_radii: _WeightArrayLike,
    *,
    jitter_scale: float = 0.5,
    seed: int = 42,
) -> Layout2d: ...

def py_coarsen(
    graph: GraphAdapter,
    node_group_func: Callable[[int], int],
    node_merge_func: Callable[[list[int]], Any],
    edge_merge_func: Callable[[list[int]], Any],
) -> tuple[GraphAdapter, dict[int, int]]: ...

def spectral_clustering(
    graph: CsrGraph,
    k: int = 2,
    max_eigen_iters: int = 300,
    eigen_tol: float = 1e-4,
    kmeans_max_iters: int = 100,
    kmeans_tol: float = 1e-4,
    seed: int = 42,
) -> npt.NDArray[np.uint32]: ...
