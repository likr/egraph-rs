from __future__ import annotations

from typing import Any, final

import numpy as np
import numpy.typing as npt

from ._core_core import CsrGraph, Layout2d
from ._core_neighborhood import RhghLandmarksResult
from ._core_types import _IndexArrayLike, _MatrixArrayLike, _WeightArrayLike

__all__ = [
    "KMeansResult",
    "LabelPropagationResult",
    "LeidenResult",
    "LouvainCoarsenResult",
    "LouvainResult",
    "SpectralClusteringResult",
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
    "spectral_clustering",
]

@final
class LouvainCoarsenResult:
    @property
    def coarse_graph(self) -> CsrGraph: ...
    @property
    def node_to_coarse(self) -> npt.NDArray[np.uint32]: ...
    @property
    def coarse_sizes(self) -> npt.NDArray[np.uint32]: ...
    @property
    def num_clusters(self) -> int: ...
    @property
    def num_levels(self) -> int: ...
    @property
    def level_num_clusters(self) -> list[int]: ...
    @property
    def level_resolutions(self) -> list[float]: ...
    @property
    def level_modularities(self) -> list[float]: ...
    def to_dict(self) -> dict[str, Any]: ...
    def keys(self) -> list[str]: ...
    def __contains__(self, key: object, /) -> bool: ...
    def __getitem__(self, key: str, /) -> Any: ...
    def __repr__(self) -> str: ...

@final
class LouvainResult:
    @property
    def labels(self) -> npt.NDArray[np.uint32]: ...
    @property
    def num_communities(self) -> int: ...
    @property
    def modularity(self) -> float: ...
    @property
    def passes(self) -> int: ...
    def __repr__(self) -> str: ...

@final
class LeidenResult:
    @property
    def labels(self) -> npt.NDArray[np.uint32]: ...
    @property
    def num_communities(self) -> int: ...
    @property
    def quality(self) -> float: ...
    @property
    def modularity(self) -> float: ...
    @property
    def passes(self) -> int: ...
    def __repr__(self) -> str: ...

@final
class KMeansResult:
    @property
    def labels(self) -> npt.NDArray[np.uint32]: ...
    @property
    def centroids(self) -> npt.NDArray[np.float64]: ...
    @property
    def k(self) -> int: ...
    @property
    def dims(self) -> int: ...
    @property
    def inertia(self) -> float: ...
    @property
    def iterations(self) -> int: ...
    @property
    def converged(self) -> bool: ...
    def __repr__(self) -> str: ...

@final
class LabelPropagationResult:
    @property
    def labels(self) -> npt.NDArray[np.uint32]: ...
    @property
    def num_communities(self) -> int: ...
    @property
    def iterations(self) -> int: ...
    @property
    def converged(self) -> bool: ...
    def __repr__(self) -> str: ...

@final
class SpectralClusteringResult:
    @property
    def labels(self) -> npt.NDArray[np.uint32]: ...
    @property
    def num_clusters(self) -> int: ...
    @property
    def eigen_iterations(self) -> int: ...
    def __repr__(self) -> str: ...

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
) -> RhghLandmarksResult: ...

def kmeans_plus_plus(
    points: _MatrixArrayLike,
    *,
    k: int = 2,
    max_iterations: int = 100,
    tol: float = 1e-4,
    n_init: int = 1,
    seed: int = 42,
) -> KMeansResult: ...

def label_propagation(
    graph: CsrGraph,
    *,
    max_iterations: int = 100,
    seed: int = 42,
) -> LabelPropagationResult: ...

def leiden_communities(
    graph: CsrGraph,
    *,
    objective: str = "modularity",
    resolution: float = 1.0,
    randomness: float = 0.01,
    max_passes: int = 100,
    min_quality_gain: float = 1e-7,
    seed: int = 42,
) -> LeidenResult: ...

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
) -> LouvainCoarsenResult: ...

def louvain_communities(
    graph: CsrGraph,
    *,
    max_passes: int = 100,
    resolution: float = 1.0,
    min_modularity_gain: float = 1e-7,
    seed: int = 42,
) -> LouvainResult: ...

def project_clustered_layout(
    coarse_layout: Layout2d,
    node_to_coarse: _IndexArrayLike,
    *,
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

def spectral_clustering(
    graph: CsrGraph,
    *,
    k: int = 2,
    max_eigen_iterations: int = 300,
    eigen_tol: float = 1e-4,
    kmeans_max_iterations: int = 100,
    kmeans_tol: float = 1e-4,
    seed: int = 42,
) -> SpectralClusteringResult: ...
