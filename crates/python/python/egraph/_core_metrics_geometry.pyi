from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence, final

import numpy as np
import numpy.typing as npt

from ._core_core import CsrGraph, Layout2d, LayoutNd
from ._core_manifold import ManifoldLayout2d
from ._core_types import (
    _EdgeArrayLike,
    _IndexArrayLike,
    _MatrixArrayLike,
    _WeightArrayLike,
)

_AnyLayout = Layout2d | LayoutNd | ManifoldLayout2d

__all__ = [
    "MetricReport",
    "all_pairs_bfs",
    "angular_resolution",
    "aspect_ratio",
    "bfs_single_source",
    "crossing_angle",
    "crossing_angle_from_pairs",
    "crossing_edges",
    "crossing_edges_torus",
    "crossing_number",
    "crossing_number_from_pairs",
    "crossing_summary",
    "crosslessness",
    "crosslessness_from_count",
    "dgg_preservation",
    "drng_preservation",
    "edge_angle",
    "evaluate_metrics",
    "gabriel_graph_edges",
    "gabriel_graph_property",
    "geodesic_distortion",
    "ideal_edge_lengths",
    "ideal_edge_lengths_from_distances",
    "ideal_edge_lengths_scaled",
    "ideal_edge_lengths_with_unit",
    "layout_degeneracy",
    "max_crossing_pairs",
    "minimum_angle",
    "minimum_angle_summary",
    "neighborhood_preservation_1hop",
    "neighborhood_preservation_2hop",
    "node_resolution",
    "prism_distortion",
    "prism_distortion_with_edges",
    "relative_neighborhood_edges",
    "sampled_crossing_number",
    "sampled_crosslessness",
    "sampled_np1",
    "sampled_np2",
    "sampled_scale_normalized_stress",
    "sampled_scale_normalized_stress_detail",
    "scale_normalized_stress",
    "scale_normalized_stress_detail",
    "scale_normalized_stress_from_distances",
    "scale_normalized_stress_from_distances_detail",
    "scale_normalized_stress_from_pairs",
    "sweep_crossing_angle",
    "sweep_crossing_number",
    "torus_crossing_number",
]

@final
class MetricReport:
    @property
    def scale_normalized_stress(self) -> float | None: ...
    @property
    def sampled_scale_normalized_stress(self) -> float | None: ...
    @property
    def geodesic_distortion(self) -> float | None: ...
    @property
    def crossing_number(self) -> int | None: ...
    @property
    def crossing_edges(self) -> list[dict[str, Any]] | None: ...
    @property
    def crossing_angle(self) -> float | None: ...
    @property
    def sampled_crossing_number(self) -> float | None: ...
    @property
    def crosslessness(self) -> float | None: ...
    @property
    def sampled_crosslessness(self) -> float | None: ...
    @property
    def torus_crossing_number(self) -> int | None: ...
    @property
    def crossing_edges_torus(self) -> list[dict[str, Any]] | None: ...
    @property
    def neighborhood_preservation_1hop(self) -> float | None: ...
    @property
    def neighborhood_preservation_2hop(self) -> float | None: ...
    @property
    def sampled_np1(self) -> float | None: ...
    @property
    def sampled_np2(self) -> float | None: ...
    @property
    def gabriel_graph_property(self) -> float | None: ...
    @property
    def node_resolution(self) -> float | None: ...
    @property
    def relative_neighborhood_preservation(self) -> float | None: ...
    @property
    def gabriel_graph_preservation(self) -> float | None: ...
    @property
    def aspect_ratio(self) -> float | None: ...
    @property
    def angular_resolution(self) -> float | None: ...
    @property
    def minimum_angle(self) -> float | None: ...
    @property
    def minimum_angle_a1(self) -> float | None: ...
    @property
    def layout_degeneracy(self) -> float | None: ...
    @property
    def ideal_edge_lengths(self) -> float | None: ...
    @property
    def prism_sigma_dist(self) -> float | None: ...
    @property
    def prism_sigma_disp(self) -> float | None: ...
    @property
    def adjusted_patch_nig(self) -> float | None: ...
    @property
    def cluster_area_faithfulness(self) -> float | None: ...
    @property
    def landmark_stratified_rta(self) -> float | None: ...
    @property
    def distance_consistency(self) -> float | None: ...
    @property
    def distribution_consistency(self) -> float | None: ...
    @property
    def class_density_measure(self) -> float | None: ...
    @property
    def histogram_density_measure(self) -> float | None: ...
    @property
    def sepme_knng(self) -> float | None: ...
    @property
    def sepme_rng(self) -> float | None: ...
    @property
    def sepme_gab(self) -> float | None: ...
    @property
    def sepme_gong(self) -> float | None: ...
    @property
    def neighborhood_hit(self) -> float | None: ...
    @property
    def silhouette_score(self) -> float | None: ...
    @property
    def calinski_harabasz_score(self) -> float | None: ...
    @property
    def davies_bouldin_score(self) -> float | None: ...
    @property
    def cq_kmeans_ari(self) -> float | None: ...
    @property
    def cq_kmeans_ami(self) -> float | None: ...
    @property
    def cq_hac_hom(self) -> float | None: ...
    @property
    def cq_hac_ami(self) -> float | None: ...
    @property
    def pairwise_distance_overlap_error(self) -> float | None: ...
    @property
    def sprawlter_nn(self) -> float | None: ...
    @property
    def nor_oo_nni(self) -> float | None: ...
    @property
    def nor_sp_ch_a(self) -> float | None: ...
    @property
    def nor_gs_bb_iar(self) -> float | None: ...
    @property
    def nor_nm_dm_imse(self) -> float | None: ...
    @property
    def nor_el_rsd(self) -> float | None: ...
    def to_dict(self, include_none: bool = False) -> dict[str, Any]: ...
    def keys(self) -> list[str]: ...
    def __getitem__(self, key: str, /) -> Any: ...
    def __contains__(self, key: object, /) -> bool: ...
    def __len__(self) -> int: ...
    def __repr__(self) -> str: ...

def all_pairs_bfs(
    graph: CsrGraph, unit_edge_length: float = 1.0
) -> npt.NDArray[np.float64]: ...

def angular_resolution(graph: CsrGraph, layout: Layout2d) -> float: ...

def aspect_ratio(layout: Layout2d) -> float: ...

def bfs_single_source(
    graph: CsrGraph, source: int, unit_edge_length: float = 1.0
) -> npt.NDArray[np.float64]: ...

def crossing_angle(graph: CsrGraph, layout: Layout2d) -> float: ...

def crossing_angle_from_pairs(pairs: Iterable[Mapping[str, Any]]) -> float: ...

def crossing_edges(graph: CsrGraph, layout: Layout2d) -> list[dict[str, Any]]: ...

def crossing_edges_torus(
    graph: CsrGraph, layout: Layout2d | ManifoldLayout2d
) -> list[dict[str, Any]]: ...

def crossing_number(graph: CsrGraph, layout: Layout2d) -> float: ...

def crossing_number_from_pairs(pairs: Iterable[Mapping[str, Any]]) -> float: ...

def crossing_summary(graph: CsrGraph, layout: Layout2d) -> dict[str, Any]: ...

def crosslessness(graph: CsrGraph, layout: Layout2d) -> float: ...

def crosslessness_from_count(graph: CsrGraph, crossings: float) -> float: ...

def max_crossing_pairs(graph: CsrGraph) -> int: ...

def dgg_preservation(graph: CsrGraph, layout: _AnyLayout) -> float: ...

def drng_preservation(graph: CsrGraph, layout: _AnyLayout) -> float: ...

def edge_angle(x1: float, y1: float, x2: float, y2: float) -> float | None: ...

def evaluate_metrics(
    graph: CsrGraph,
    layout: _AnyLayout,
    targets: Iterable[str],
    *,
    num_pivots: int = 64,
    num_samples: int = 4096,
    seed: int = 42,
    ideal_edge_length: float = 1.0,
    labels: _IndexArrayLike | None = None,
    num_landmarks: int = 64,
    num_triplets: int = 1000000,
    min_degree: int = 2,
    patch_radius: int = 8,
    canvas_size: int = 2048,
    pad_frac: float = 0.04,
    g_max: int = 128,
    sigma_cells: float = 0.85,
    kappa: float = 10.0,
    tau_frac: float = 0.18,
    boundary_r95: float = 2.0,
    stevens_gamma: float = 1.0,
    node_weights: _WeightArrayLike | None = None,
    cluster_weights: _WeightArrayLike | None = None,
    original_layout: Layout2d | None = None,
    sprawlter_half_widths: _WeightArrayLike | None = None,
    sprawlter_half_heights: _WeightArrayLike | None = None,
    sepme_knng_k: int = 2,
    neighborhood_hit_k: int = 7,
    dc_grid_size: int = 20,
    cdm_grid_size: int = 64,
    cdm_bandwidth: float | None = None,
    num_bins: int = 100,
    gong_gamma: float = 0.35,
    kmeans_n_init: int = 5,
    kmeans_max_iterations: int = 100,
    sprawlter_node_radius: float = 1.0,
) -> MetricReport: ...

def gabriel_graph_edges(layout: _AnyLayout) -> npt.NDArray[np.uint32]: ...

def gabriel_graph_property(graph: CsrGraph, layout: _AnyLayout) -> float: ...

def geodesic_distortion(graph: CsrGraph, layout: ManifoldLayout2d) -> float: ...

def ideal_edge_lengths(graph: CsrGraph, layout: _AnyLayout) -> float: ...

def ideal_edge_lengths_with_unit(
    graph: CsrGraph, layout: _AnyLayout, unit_edge_length: float = 1.0
) -> float: ...

def ideal_edge_lengths_scaled(
    graph: CsrGraph, layout: _AnyLayout, unit_edge_length: float = 1.0
) -> float: ...

def ideal_edge_lengths_from_distances(
    graph: CsrGraph, layout: _AnyLayout, distances: _MatrixArrayLike
) -> float: ...

def layout_degeneracy(graph: CsrGraph, layout: _AnyLayout) -> dict[str, Any]: ...

def minimum_angle(graph: CsrGraph, layout: Layout2d) -> float: ...

def minimum_angle_summary(graph: CsrGraph, layout: Layout2d) -> dict[str, float]: ...

def neighborhood_preservation_1hop(
    graph: CsrGraph, layout: _AnyLayout
) -> float: ...

def neighborhood_preservation_2hop(
    graph: CsrGraph, layout: _AnyLayout
) -> float: ...

def node_resolution(layout: _AnyLayout) -> float: ...

def prism_distortion(
    original_layout: Layout2d, adjusted_layout: Layout2d
) -> tuple[float, float]: ...

def prism_distortion_with_edges(
    original_layout: Layout2d,
    adjusted_layout: Layout2d,
    edges: _EdgeArrayLike,
) -> tuple[float, float]: ...

def relative_neighborhood_edges(layout: _AnyLayout) -> npt.NDArray[np.uint32]: ...

def sampled_crossing_number(
    graph: CsrGraph,
    layout: Layout2d,
    *,
    num_samples: int = 4096,
    seed: int = 42,
) -> float: ...

def sampled_crosslessness(
    graph: CsrGraph,
    layout: Layout2d,
    *,
    num_samples: int = 4096,
    seed: int = 42,
) -> float: ...

def sampled_np1(
    graph: CsrGraph,
    layout: _AnyLayout,
    *,
    num_samples: int = 4096,
    seed: int = 42,
) -> float: ...

def sampled_np2(
    graph: CsrGraph,
    layout: _AnyLayout,
    *,
    num_samples: int = 4096,
    seed: int = 42,
) -> float: ...

def sampled_scale_normalized_stress(
    graph: CsrGraph,
    layout: _AnyLayout,
    *,
    num_pivots: int = 64,
    seed: int = 42,
) -> float: ...

def sampled_scale_normalized_stress_detail(
    graph: CsrGraph,
    layout: _AnyLayout,
    *,
    num_pivots: int = 64,
    seed: int = 42,
) -> dict[str, Any]: ...

def scale_normalized_stress(graph: CsrGraph, layout: _AnyLayout) -> float: ...

def scale_normalized_stress_detail(
    graph: CsrGraph, layout: _AnyLayout
) -> dict[str, Any]: ...

def scale_normalized_stress_from_distances(
    layout: _AnyLayout,
    distances: _MatrixArrayLike,
    weights: _MatrixArrayLike | None = None,
) -> float: ...

def scale_normalized_stress_from_distances_detail(
    layout: _AnyLayout,
    distances: _MatrixArrayLike,
    weights: _MatrixArrayLike | None = None,
) -> dict[str, Any]: ...

def scale_normalized_stress_from_pairs(
    layout: _AnyLayout,
    pairs: _MatrixArrayLike | Sequence[tuple[int, int, float, float]],
) -> dict[str, Any]: ...

def sweep_crossing_angle(graph: CsrGraph, layout: Layout2d) -> float: ...

def sweep_crossing_number(graph: CsrGraph, layout: Layout2d) -> float: ...

def torus_crossing_number(
    graph: CsrGraph, layout: Layout2d | ManifoldLayout2d
) -> float: ...
