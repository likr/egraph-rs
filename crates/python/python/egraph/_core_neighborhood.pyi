from __future__ import annotations

from typing import Any, final

import numpy as np
import numpy.typing as npt

from ._core_core import CsrGraph
from ._core_linalg import CsrMatrix
from ._core_types import (
    _CoordArrayLike,
    _IndexArrayLike,
    _MatrixArrayLike,
    _WeightArrayLike,
)

__all__ = [
    "DogVaradhanLandmarks",
    "KnnGraph",
    "Sparse12HopPairs",
    "SparseProbabilities",
    "apply_relative_distance_floor",
    "build_capped_12hop_knn",
    "build_gumap_12hop_priority_csr",
    "build_hybrid_adaptive_knn",
    "build_pure_2hop_knn",
    "build_rerank_knn",
    "compute_2hop_neighborhood_sizes",
    "compute_fuzzy_simplicial_set",
    "compute_horvitz_thompson_gamma",
    "compute_knn_probabilities",
    "compute_median_2hop_perplexity",
    "compute_nodewise_2hop_perplexities",
    "compute_regularized_dog_varadhan_landmarks",
    "compute_regularized_dog_varadhan_landmarks_from_matrix",
    "extract_1hop_2hop_csr",
    "extract_sparse_1hop_2hop_pairs",
    "find_ab_params",
    "graph_12hop_row_normalized_probabilities",
    "graph_row_normalized_probabilities",
    "partial_bfs_knn",
    "symmetrize_row_normalized",
]

@final
class DogVaradhanLandmarks:
    @property
    def num_nodes(self) -> int: ...
    @property
    def num_landmarks(self) -> int: ...
    def landmarks(self) -> npt.NDArray[np.uint32]: ...
    def table(self) -> npt.NDArray[np.float64]: ...
    def coarse_table(self) -> npt.NDArray[np.float64] | None: ...
    @property
    def t1(self) -> float: ...
    @property
    def t2(self) -> float: ...
    @property
    def alpha(self) -> float: ...
    @property
    def delta_psd(self) -> float: ...
    @property
    def beta_eff(self) -> float: ...
    @property
    def beta_auto(self) -> float: ...
    @property
    def degree(self) -> int: ...
    @property
    def used_coarse_scaffold(self) -> bool: ...
    def alt_distances(
        self, u: _IndexArrayLike, v: _IndexArrayLike
    ) -> npt.NDArray[np.float64]: ...
    def __len__(self) -> int: ...
    def __repr__(self) -> str: ...

@final
class KnnGraph:
    @staticmethod
    def from_csr(
        num_nodes: int,
        indptr: _IndexArrayLike,
        indices: _IndexArrayLike,
        distances: _WeightArrayLike,
    ) -> KnnGraph: ...
    @property
    def num_nodes(self) -> int: ...
    @property
    def nnz(self) -> int: ...
    def indptr(self) -> npt.NDArray[np.uintp]: ...
    def indices(self) -> npt.NDArray[np.uint32]: ...
    def distances(self) -> npt.NDArray[np.float64]: ...
    def __len__(self) -> int: ...
    def __repr__(self) -> str: ...

@final
class Sparse12HopPairs:
    @property
    def num_1hop(self) -> int: ...
    @property
    def num_2hop(self) -> int: ...
    def u1(self) -> npt.NDArray[np.uint32]: ...
    def v1(self) -> npt.NDArray[np.uint32]: ...
    def u2(self) -> npt.NDArray[np.uint32]: ...
    def v2(self) -> npt.NDArray[np.uint32]: ...
    def __repr__(self) -> str: ...

@final
class SparseProbabilities:
    @property
    def num_nodes(self) -> int: ...
    @property
    def nnz(self) -> int: ...
    def indptr(self) -> npt.NDArray[np.uintp]: ...
    def indices(self) -> npt.NDArray[np.uint32]: ...
    def values(self) -> npt.NDArray[np.float64]: ...
    def __len__(self) -> int: ...
    def __repr__(self) -> str: ...

def apply_relative_distance_floor(
    d1: _WeightArrayLike,
    d2: _WeightArrayLike,
    d3: _WeightArrayLike,
    *,
    eps_dist: float = 1e-6,
    min_ratio: float = 0.5,
) -> tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    float,
]: ...

def build_capped_12hop_knn(
    graph: CsrGraph,
    max_2hop_per_node: int = 32,
    *,
    seed: int = 42,
) -> tuple[KnnGraph, npt.NDArray[np.uint32], dict[str, Any]]: ...

def build_gumap_12hop_priority_csr(
    num_nodes: int,
    u1: _IndexArrayLike,
    v1: _IndexArrayLike,
    d1: _WeightArrayLike,
    u2: _IndexArrayLike,
    v2: _IndexArrayLike,
    d2: _WeightArrayLike,
    *,
    tier2_margin: float = 1.1,
) -> KnnGraph: ...

def build_hybrid_adaptive_knn(
    graph: CsrGraph,
    coords: _MatrixArrayLike,
    perplexities: _WeightArrayLike | float,
    *,
    k_factor: float = 3.0,
    k: int | None = None,
    margin_gap: float = 0.1,
    coarse_dim: int = 10,
    rerank_factor: int = 16,
    max_row_k: int | None = None,
) -> tuple[KnnGraph, dict[str, Any]]: ...

def build_pure_2hop_knn(
    graph: CsrGraph,
) -> tuple[KnnGraph, npt.NDArray[np.uint32], dict[str, Any]]: ...

def build_rerank_knn(
    coords: _CoordArrayLike,
    k: int = 15,
    *,
    coarse_dim: int = 10,
    rerank_factor: int = 16,
) -> KnnGraph: ...

def compute_2hop_neighborhood_sizes(graph: CsrGraph) -> npt.NDArray[np.uint32]: ...

def compute_fuzzy_simplicial_set(
    knn: KnnGraph,
    *,
    sigma_iters: int = 64,
    sigma_tolerance: float = 1e-5,
) -> SparseProbabilities: ...

def compute_horvitz_thompson_gamma(
    gamma_non2: float, num_nodes: int, neg_per_node: int
) -> float: ...

def compute_knn_probabilities(
    knn: KnnGraph,
    perplexities: _WeightArrayLike | float,
    *,
    power: float = 2.0,
    sigma_iters: int = 50,
    sigma_tolerance: float = 1e-5,
    row_normalize: bool = False,
) -> SparseProbabilities: ...

def compute_median_2hop_perplexity(
    graph: CsrGraph,
    *,
    c: float = 1.5,
    min_perplexity: float = 15.0,
    max_perplexity: float = 1200.0,
) -> float: ...

def compute_nodewise_2hop_perplexities(
    graph: CsrGraph,
    *,
    c: float = 0.9,
    min_perplexity: float = 5.0,
    max_perplexity: float = 1200.0,
) -> npt.NDArray[np.float64]: ...

def compute_regularized_dog_varadhan_landmarks(
    graph: CsrGraph,
    *,
    degree: int | None = None,
    num_landmarks: int = 64,
    seed: int = 42,
    degree_normalize: bool = True,
    p_exp: float = 0.5,
    eps_psd: float = 1e-4,
    t_coarse: float = 48.0,
    t1: float | None = None,
    t2: float | None = None,
    alpha: float | None = None,
    beta_floor: float | None = None,
) -> DogVaradhanLandmarks: ...

def compute_regularized_dog_varadhan_landmarks_from_matrix(
    graph: CsrGraph,
    matrix: CsrMatrix,
    *,
    lambda_max: float | None = None,
    degree: int | None = None,
    num_landmarks: int = 64,
    min_degree: int = 2,
    jl_dim: int = 64,
    seed: int = 42,
    degree_normalize: bool = True,
    p_exp: float = 0.5,
    eps_psd: float = 1e-4,
    t_coarse: float = 48.0,
    t1: float | None = None,
    t2: float | None = None,
    alpha: float | None = None,
    beta_floor: float | None = None,
) -> DogVaradhanLandmarks: ...

def extract_1hop_2hop_csr(graph: CsrGraph) -> tuple[CsrGraph, CsrGraph]: ...

def extract_sparse_1hop_2hop_pairs(
    graph: CsrGraph,
    max_2hop_per_node: int = 32,
    seed: int = 42,
) -> Sparse12HopPairs: ...

def find_ab_params(min_dist: float = 0.1, spread: float = 1.0) -> tuple[float, float]: ...

def graph_12hop_row_normalized_probabilities(
    graph: CsrGraph,
    w2: float = 0.5,
    max_2hop_per_node: int | None = None,
    *,
    degree_beta: float = 0.0,
    seed: int = 42,
) -> SparseProbabilities: ...

def graph_row_normalized_probabilities(
    graph: CsrGraph,
) -> SparseProbabilities: ...

def partial_bfs_knn(
    graph: CsrGraph,
    k: int = 30,
    *,
    seed: int = 42,
) -> KnnGraph: ...

def symmetrize_row_normalized(
    num_nodes: int,
    indptr: _IndexArrayLike,
    indices: _IndexArrayLike,
    weights: _WeightArrayLike,
) -> SparseProbabilities: ...
