from __future__ import annotations

from typing import Any, Sequence

from ._core_core import CsrGraph, Layout2d
from ._core_types import _IndexArrayLike, _MatrixArrayLike, _WeightArrayLike

__all__ = [
    "adjusted_patch_nig",
    "adjusted_patch_nig_multiscale",
    "calinski_harabasz_score",
    "class_density_measure",
    "cluster_area_faithfulness",
    "cluster_area_faithfulness_detail",
    "cq_hac",
    "cq_kmeans",
    "davies_bouldin_score",
    "distance_consistency",
    "distribution_consistency",
    "landmark_stratified_rta",
    "landmark_stratified_rta_detail",
    "landmark_stratified_rta_from_landmarks",
    "neighborhood_hit",
    "nor_metrics",
    "pairwise_distance_overlap_error",
    "partition_agreement",
    "sepme_gab",
    "sepme_gong",
    "sepme_knng",
    "sepme_rng",
    "silhouette_score",
    "sprawlter_nn",
]

def adjusted_patch_nig(
    layout: Layout2d,
    community_labels: _IndexArrayLike,
    *,
    patch_radius: int = 8,
    canvas_size: int = 2048,
    pad_frac: float = 0.04,
    seed: int = 42,
) -> float: ...

def adjusted_patch_nig_multiscale(
    layout: Layout2d,
    community_labels: _IndexArrayLike,
    *,
    radii: Sequence[int] | None = (2, 8, 32),
    canvas_size: int = 2048,
    pad_frac: float = 0.04,
    seed: int = 42,
) -> dict[str, float]: ...

def calinski_harabasz_score(
    layout: Layout2d,
    labels: _IndexArrayLike,
) -> float: ...

def class_density_measure(
    layout: Layout2d,
    labels: _IndexArrayLike,
    *,
    grid_size: int = 64,
    bandwidth: float | None = None,
) -> tuple[float, float]: ...

def cluster_area_faithfulness(
    layout: Layout2d,
    community_labels: _IndexArrayLike,
    *,
    g_max: int = 128,
    sigma_cells: float = 0.85,
    kappa: float = 10.0,
    tau_frac: float = 0.18,
    boundary_r95: float = 2.0,
    stevens_gamma: float = 1.0,
    node_weights: _WeightArrayLike | None = None,
    cluster_weights: _WeightArrayLike | None = None,
) -> float: ...

def cluster_area_faithfulness_detail(
    layout: Layout2d,
    community_labels: _IndexArrayLike,
    *,
    g_max: int = 128,
    sigma_cells: float = 0.85,
    kappa: float = 10.0,
    tau_frac: float = 0.18,
    boundary_r95: float = 2.0,
    stevens_gamma: float = 1.0,
    node_weights: _WeightArrayLike | None = None,
    cluster_weights: _WeightArrayLike | None = None,
    return_per_cluster: bool = False,
) -> dict[str, Any]: ...

def cq_hac(
    layout: Layout2d,
    community_labels: _IndexArrayLike,
) -> dict[str, float]: ...

def cq_kmeans(
    layout: Layout2d,
    community_labels: _IndexArrayLike,
    *,
    n_init: int = 5,
    max_iters: int = 100,
    seed: int = 42,
) -> dict[str, float]: ...

def davies_bouldin_score(
    layout: Layout2d,
    labels: _IndexArrayLike,
) -> float: ...

def distance_consistency(
    layout: Layout2d,
    labels: _IndexArrayLike,
) -> float: ...

def distribution_consistency(
    layout: Layout2d,
    labels: _IndexArrayLike,
    grid_size: int = 20,
) -> float: ...

def landmark_stratified_rta(
    graph: CsrGraph,
    layout: Layout2d,
    *,
    num_landmarks: int = 64,
    num_triplets: int = 1000000,
    min_degree: int = 2,
    seed: int = 42,
) -> float: ...

def landmark_stratified_rta_detail(
    graph: CsrGraph,
    layout: Layout2d,
    *,
    num_landmarks: int = 64,
    num_triplets: int = 1000000,
    min_degree: int = 2,
    seed: int = 42,
) -> dict[str, Any]: ...

def landmark_stratified_rta_from_landmarks(
    layout: Layout2d,
    landmark_ids: _IndexArrayLike,
    landmark_dists: _IndexArrayLike | _WeightArrayLike | _MatrixArrayLike | Sequence[Sequence[int | float]],
    *,
    num_triplets: int = 1000000,
    seed: int = 42,
) -> dict[str, Any]: ...

def neighborhood_hit(
    layout: Layout2d,
    labels: _IndexArrayLike,
    k: int = 7,
) -> float: ...

def nor_metrics(
    orig_layout: Layout2d,
    adjusted_layout: Layout2d,
    half_widths: _WeightArrayLike,
    half_heights: _WeightArrayLike,
    *,
    graph: CsrGraph | None = None,
) -> dict[str, float]: ...

def pairwise_distance_overlap_error(
    layout: Layout2d,
    labels: _IndexArrayLike,
    num_bins: int = 100,
) -> float: ...

def partition_agreement(
    labels_true: _IndexArrayLike,
    labels_pred: _IndexArrayLike,
) -> dict[str, float]: ...

def sepme_gab(
    layout: Layout2d,
    labels: _IndexArrayLike,
) -> float: ...

def sepme_gong(
    layout: Layout2d,
    labels: _IndexArrayLike,
    gamma: float = 0.35,
) -> float: ...

def sepme_knng(
    layout: Layout2d,
    labels: _IndexArrayLike,
    k: int = 2,
) -> float: ...

def sepme_rng(
    layout: Layout2d,
    labels: _IndexArrayLike,
) -> float: ...

def silhouette_score(
    layout: Layout2d,
    labels: _IndexArrayLike,
) -> float: ...

def sprawlter_nn(
    layout: Layout2d,
    half_widths: _WeightArrayLike,
    half_heights: _WeightArrayLike | None = None,
) -> float: ...
