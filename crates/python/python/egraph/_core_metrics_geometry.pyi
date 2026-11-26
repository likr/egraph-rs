from __future__ import annotations

from typing import Any, Iterable, overload

import numpy as np
import numpy.typing as npt

from ._core_core import CsrGraph, Drawing, DrawingEuclidean2d, GraphAdapter, Layout2d
from ._core_distance_matrix import DistanceMatrix
from ._core_types import (
    _EdgeArrayLike,
    _IndexArrayLike,
    _MatrixArrayLike,
    _WeightArrayLike,
)

__all__ = [
    "all_pairs_bfs",
    "angular_resolution",
    "angular_resolution_v6",
    "aspect_ratio",
    "aspect_ratio_v6",
    "bfs_single_source",
    "crossing_angle",
    "crossing_angle_with_crossing_edges",
    "crossing_edges",
    "crossing_edges_torus",
    "crossing_number",
    "crossing_number_with_crossing_edges",
    "crossing_summary",
    "crosslessness",
    "dgg_preservation",
    "drng_preservation",
    "edge_angle",
    "evaluate_metrics",
    "gabriel_graph_edges",
    "gabriel_graph_property",
    "gabriel_graph_property_v6",
    "ideal_edge_lengths",
    "ideal_edge_lengths_v6",
    "ideal_edge_lengths_with_unit",
    "layout_degeneracy",
    "minimum_angle",
    "minimum_angle_summary",
    "neighborhood_preservation",
    "neighborhood_preservation_1hop",
    "neighborhood_preservation_2hop",
    "neighborhood_preservation_2hop_v6",
    "node_resolution",
    "node_resolution_v6",
    "prism_distortion",
    "prism_distortion_with_edges",
    "relative_neighborhood_edges",
    "sampled_crossing_number",
    "sampled_crosslessness",
    "sampled_np1",
    "sampled_np2",
    "sampled_scale_normalized_stress",
    "scale_normalized_stress",
    "scale_normalized_stress_from_distances",
    "stress",
    "sweep_crossing_angle",
    "sweep_crossing_number",
    "torus_crossing_number",
]

def all_pairs_bfs(
    graph: CsrGraph, unit_edge_length: float = 1.0
) -> npt.NDArray[np.float64]: ...

@overload
def angular_resolution(
    graph: GraphAdapter, drawing: DrawingEuclidean2d
) -> float: ...

@overload
def angular_resolution(graph: CsrGraph, drawing: Layout2d) -> float: ...

def angular_resolution_v6(graph: CsrGraph, layout: Layout2d) -> float: ...

@overload
def aspect_ratio(drawing: DrawingEuclidean2d) -> float: ...

@overload
def aspect_ratio(drawing: Layout2d) -> float: ...

def aspect_ratio_v6(layout: Layout2d) -> float: ...

def bfs_single_source(
    graph: CsrGraph, source: int, unit_edge_length: float = 1.0
) -> npt.NDArray[np.float64]: ...

@overload
def crossing_angle(
    graph: GraphAdapter, drawing: DrawingEuclidean2d
) -> float: ...

@overload
def crossing_angle(graph: CsrGraph, drawing: Layout2d) -> float: ...

def crossing_angle_with_crossing_edges(
    crossing_edges: Any,
) -> float: ...

@overload
def crossing_edges(graph: CsrGraph, drawing: Layout2d) -> list[dict[str, Any]]: ...

@overload
def crossing_edges(graph: GraphAdapter, drawing: Drawing) -> Any: ...

def crossing_edges_torus(graph: CsrGraph, layout: Layout2d) -> list[dict[str, Any]]: ...

@overload
def crossing_number(
    graph: GraphAdapter, drawing: DrawingEuclidean2d
) -> float: ...

@overload
def crossing_number(graph: CsrGraph, drawing: Layout2d) -> float: ...

def crossing_number_with_crossing_edges(
    crossing_edges: Any,
) -> float: ...

def crossing_summary(graph: CsrGraph, layout: Layout2d) -> dict[str, Any]: ...

def crosslessness(graph: CsrGraph, layout: Layout2d) -> float: ...

def dgg_preservation(graph: CsrGraph, layout: Layout2d) -> float: ...

def drng_preservation(graph: CsrGraph, layout: Layout2d) -> float: ...

def edge_angle(x1: float, y1: float, x2: float, y2: float) -> float | None: ...

def evaluate_metrics(
    graph: CsrGraph,
    layout: Layout2d,
    targets: Iterable[str],
    *,
    num_pivots: int = 64,
    num_samples: int = 4096,
    seed: int = 42,
    ideal_edge_length: float = 1.0,
    community_labels: _IndexArrayLike | None = None,
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
    sepme_knng_k: int = 2,
    neighborhood_hit_k: int = 7,
    dc_grid_size: int = 20,
    cdm_grid_size: int = 64,
    cdm_bandwidth: float | None = None,
    num_bins: int = 100,
    gong_gamma: float = 0.35,
    kmeans_n_init: int = 5,
    kmeans_max_iters: int = 100,
    sprawlter_node_radius: float = 1.0,
) -> dict[str, float]: ...

def gabriel_graph_edges(layout: Layout2d) -> npt.NDArray[np.uint32]: ...

@overload
def gabriel_graph_property(
    graph: GraphAdapter, drawing: DrawingEuclidean2d
) -> float: ...

@overload
def gabriel_graph_property(graph: CsrGraph, drawing: Layout2d) -> float: ...

def gabriel_graph_property_v6(graph: CsrGraph, layout: Layout2d) -> float: ...

@overload
def ideal_edge_lengths(
    graph: GraphAdapter,
    drawing: Drawing,
    distance_matrix: DistanceMatrix,
) -> float: ...

@overload
def ideal_edge_lengths(
    graph: CsrGraph,
    drawing: Layout2d,
    distance_matrix: None = None,
) -> float: ...

def ideal_edge_lengths_v6(graph: CsrGraph, layout: Layout2d) -> float: ...

def ideal_edge_lengths_with_unit(
    graph: CsrGraph, layout: Layout2d, unit_edge_length: float = 1.0
) -> float: ...

def layout_degeneracy(graph: CsrGraph, layout: Layout2d) -> dict[str, Any]: ...

def minimum_angle(graph: CsrGraph, layout: Layout2d) -> float: ...

def minimum_angle_summary(graph: CsrGraph, layout: Layout2d) -> dict[str, float]: ...

@overload
def neighborhood_preservation(
    graph: GraphAdapter, drawing: DrawingEuclidean2d
) -> float: ...

@overload
def neighborhood_preservation(graph: CsrGraph, drawing: Layout2d) -> float: ...

def neighborhood_preservation_1hop(graph: CsrGraph, layout: Layout2d) -> float: ...

@overload
def neighborhood_preservation_2hop(
    graph: GraphAdapter, drawing: DrawingEuclidean2d
) -> float: ...

@overload
def neighborhood_preservation_2hop(graph: CsrGraph, drawing: Layout2d) -> float: ...

def neighborhood_preservation_2hop_v6(graph: CsrGraph, layout: Layout2d) -> float: ...

@overload
def node_resolution(drawing: Drawing) -> float: ...

@overload
def node_resolution(drawing: Layout2d) -> float: ...

def node_resolution_v6(layout: Layout2d) -> float: ...

def prism_distortion(
    original_layout: Layout2d, adjusted_layout: Layout2d
) -> tuple[float, float]: ...

def prism_distortion_with_edges(
    original_layout: Layout2d,
    adjusted_layout: Layout2d,
    edges: _EdgeArrayLike,
) -> tuple[float, float]: ...

def relative_neighborhood_edges(layout: Layout2d) -> npt.NDArray[np.uint32]: ...

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
    layout: Layout2d,
    *,
    num_samples: int = 4096,
    seed: int = 42,
) -> float: ...

def sampled_np2(
    graph: CsrGraph,
    layout: Layout2d,
    *,
    num_samples: int = 4096,
    seed: int = 42,
) -> float: ...

def sampled_scale_normalized_stress(
    graph: CsrGraph,
    layout: Layout2d,
    *,
    num_pivots: int = 64,
    seed: int = 42,
) -> float: ...

def scale_normalized_stress(graph: CsrGraph, layout: Layout2d) -> float: ...

def scale_normalized_stress_from_distances(
    layout: Layout2d,
    distances: _MatrixArrayLike,
    weights: _MatrixArrayLike | None = None,
) -> float: ...

def stress(drawing: Drawing, distance_matrix: DistanceMatrix) -> float: ...

def sweep_crossing_angle(graph: CsrGraph, layout: Layout2d) -> float: ...

def sweep_crossing_number(graph: CsrGraph, layout: Layout2d) -> float: ...

def torus_crossing_number(graph: CsrGraph, layout: Layout2d) -> float: ...
