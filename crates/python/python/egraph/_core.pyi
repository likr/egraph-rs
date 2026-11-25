from __future__ import annotations

from typing import Any, Callable, Iterable, Sequence, final, overload
from typing_extensions import disjoint_base

import numpy as np
import numpy.typing as npt

_IndexArrayLike = npt.NDArray[np.integer[Any]] | Sequence[int]
_EdgeArrayLike = npt.NDArray[np.integer[Any]] | Sequence[tuple[int, int]] | Sequence[Sequence[int]]
_WeightArrayLike = npt.NDArray[np.floating[Any]] | npt.NDArray[np.integer[Any]] | Sequence[float]
_CoordArrayLike = npt.NDArray[np.floating[Any]] | Sequence[Sequence[float]]
_MatrixArrayLike = npt.NDArray[np.floating[Any]] | npt.NDArray[np.integer[Any]] | Sequence[Sequence[float]]
_VectorArrayLike = (
    npt.NDArray[np.floating[Any]]
    | npt.NDArray[np.integer[Any]]
    | Sequence[float]
    | Sequence[Sequence[float]]
)
_DtypeLike = str | np.dtype[Any] | type[np.floating[Any]]

__version__: str

__all__ = [
    "__version__",
    "AmgCgSolver",
    "AmgPcgSolver",
    "Array1",
    "Array2",
    "BandPassDiffusionKernel",
    "BandPassDiffusionKernelBuilder",
    "BhTsNet",
    "BhTsNetBuilder",
    "CgSolver",
    "ClassicalMds",
    "Constraint",
    "CsrGraph",
    "CsrMatrix",
    "DiGraph",
    "DiffusionKernel",
    "DistanceMatrix",
    "DogVaradhanLandmarks",
    "Drawing",
    "DrawingEuclidean",
    "DrawingEuclidean2d",
    "DrawingHyperbolic2d",
    "DrawingSpherical2d",
    "DrawingTorus2d",
    "EffectiveResistanceResult",
    "EigendecompositionResult",
    "EmbeddingKernel",
    "EmbeddingKernelBuilder",
    "ExponentialKernel",
    "ExponentialKernelBuilder",
    "FdebSolver",
    "FitTsNet",
    "FitTsNetBuilder",
    "ForceAtlas2Solver",
    "FullSgd",
    "GaussianKernel",
    "GaussianKernelBuilder",
    "Graph",
    "GraphAdapter",
    "GuMap",
    "GuMapBuilder",
    "GuMapSolver",
    "IbUmapSolver",
    "Ic0CgSolver",
    "JacobiCgSolver",
    "KamadaKawai",
    "KamadaKawaiSolver",
    "KernelDistance",
    "KernelDistanceBuilder",
    "KnnGraph",
    "LTsNet",
    "LTsNetBuilder",
    "LabelPropagation",
    "Laplacian",
    "Layout2d",
    "LayoutNd",
    "LongestPath",
    "Louvain",
    "LowRankDiffusionKernel",
    "LowRankDiffusionKernelBuilder",
    "LowRankMultiscaleDiffusionKernel",
    "LowRankMultiscaleDiffusionKernelBuilder",
    "ManifoldLayout2d",
    "MultiscaleDiffusionKernel",
    "NegLogDistance",
    "NegLogDistanceBuilder",
    "NegLogSimDistance",
    "NegLogSimDistanceBuilder",
    "Omega",
    "OverwrapRemoval",
    "PivotMds",
    "PivotedBandPassDiffusionKernel",
    "PivotedBandPassDiffusionKernelBuilder",
    "PivotedDiffusionKernel",
    "PivotedMultiscaleDiffusionKernel",
    "PivotedNegLogDistance",
    "PivotedNegLogDistanceBuilder",
    "PivotedNegLogSimDistance",
    "PivotedNegLogSimDistanceBuilder",
    "PrismOverlapRemoval",
    "PyArray1Iterator",
    "RandomPairSparseSgd",
    "RdMds",
    "ResistanceCurvatureFlowResult",
    "Rng",
    "SchedulerConstant",
    "SchedulerExponential",
    "SchedulerLinear",
    "SchedulerQuadratic",
    "SchedulerReciprocal",
    "ScreenedPoissonField2d",
    "ScreenedPoissonRefiner",
    "SeparationConstraint",
    "Sgd",
    "SgdSolver",
    "Sparse12HopPairs",
    "SparseProbabilities",
    "SparseSgd",
    "SpatialIntruderRefiner",
    "SpectralClustering",
    "StandardLaplacian",
    "StressMajorization",
    "StressMajorizationSolver",
    "SymmetricNormalizedLaplacian",
    "TFdpSolver",
    "TKernel",
    "TKernelBuilder",
    "TsNet",
    "TsNetBuilder",
    "TsNetSolver",
    "WeightedEdgeLength",
    "YifanHuSolver",
    "all_pairs_bfs",
    "all_sources_bfs",
    "all_sources_dijkstra",
    "adjusted_patch_nig",
    "adjusted_patch_nig_multiscale",
    "align_clustered_centroids",
    "angular_resolution",
    "angular_resolution_v6",
    "anisotropic_laplacian",
    "apply_relative_distance_floor",
    "aspect_ratio",
    "aspect_ratio_v6",
    "assign_layers",
    "bfs_single_source",
    "build_capped_12hop_knn",
    "build_gumap_12hop_priority_csr",
    "build_hybrid_adaptive_knn",
    "build_pure_2hop_knn",
    "build_rerank_knn",
    "chebyshev_dog_jl_embedding",
    "chebyshev_dog_jl_embedding_matrix",
    "chebyshev_heat_spmv",
    "chebyshev_heat_spmv_matrix",
    "classical_mds_2d",
    "cluster_area_faithfulness",
    "cluster_area_faithfulness_detail",
    "coarse_backbone_init",
    "coarsen_graph",
    "compute_2hop_neighborhood_sizes",
    "compute_diffusion_embedding",
    "compute_effective_resistances",
    "compute_fuzzy_simplicial_set",
    "compute_horvitz_thompson_gamma",
    "compute_knn_probabilities",
    "compute_median_2hop_perplexity",
    "compute_nodewise_2hop_perplexities",
    "compute_pivoted_rhgh_landmarks",
    "compute_regularized_dog_varadhan_landmarks",
    "compute_regularized_dog_varadhan_landmarks_from_matrix",
    "compute_resistance_embedding",
    "crossing_angle",
    "crossing_angle_with_crossing_edges",
    "crossing_edges",
    "crossing_edges_torus",
    "crossing_number",
    "crossing_number_with_crossing_edges",
    "crossing_summary",
    "crosslessness",
    "cycle_edges",
    "delaunay_to_csr",
    "delaunay_triangulation",
    "dgg_preservation",
    "drng_preservation",
    "edge_angle",
    "evaluate_metrics",
    "extract_1hop_2hop_csr",
    "extract_sparse_1hop_2hop_pairs",
    "find_ab_params",
    "gabriel_graph_edges",
    "gabriel_graph_property",
    "gabriel_graph_property_v6",
    "generate_layered_constraints",
    "graph_12hop_row_normalized_probabilities",
    "graph_row_normalized_probabilities",
    "ideal_edge_lengths",
    "ideal_edge_lengths_v6",
    "ideal_edge_lengths_with_unit",
    "kmeans_plus_plus",
    "koren_spectral_init",
    "label_propagation",
    "landmark_stratified_rta",
    "landmark_stratified_rta_detail",
    "landmark_stratified_rta_from_landmarks",
    "layout_degeneracy",
    "load_dataset",
    "louvain_coarsen",
    "louvain_communities",
    "minimum_angle",
    "minimum_angle_summary",
    "neighborhood_preservation",
    "neighborhood_preservation_1hop",
    "neighborhood_preservation_2hop",
    "neighborhood_preservation_2hop_v6",
    "node_resolution",
    "node_resolution_v6",
    "normalize_layout_2d",
    "parse_csr_csv",
    "partial_bfs_knn",
    "pivot_mds_init",
    "poincare_distance",
    "poincare_mobius_add",
    "prism_distortion",
    "prism_distortion_with_edges",
    "project_1d",
    "project_clustered_layout",
    "project_clustered_layout_with_radii",
    "project_clustered_rectangle_no_overlap_2d",
    "project_clustered_rectangle_no_overlap_constraints",
    "project_layout_axis",
    "project_rectangle_no_overlap_2d",
    "project_rectangle_no_overlap_constraints_2d",
    "py_coarsen",
    "rd_mds",
    "rectangle_overlaps",
    "relative_neighborhood_edges",
    "remove_cycle",
    "remove_cycles",
    "representative_mds",
    "resistance_curvature_flow",
    "robust_normalize_layout_2d",
    "sampled_crossing_number",
    "sampled_crosslessness",
    "sampled_np1",
    "sampled_np2",
    "sampled_scale_normalized_stress",
    "scale_normalized_stress",
    "scale_normalized_stress_from_distances",
    "select_maxmin_sp_pivots",
    "smallest_laplacian_eigenpairs",
    "spectral_clustering",
    "spectral_moment_stats",
    "spherical_distance",
    "stress",
    "sweep_crossing_angle",
    "sweep_crossing_number",
    "symmetrize_row_normalized",
    "torus_crossing_number",
    "torus_distance",
    "triangulation",
    "warshall_floyd",
    "wrap_torus_point",
]

@final
class CsrGraph:
    @staticmethod
    def from_csr(
        indptr: _IndexArrayLike,
        indices: _IndexArrayLike,
        weights: _WeightArrayLike | None = None,
        *,
        directed: bool = False,
        sanitize: bool = True,
    ) -> CsrGraph: ...
    @staticmethod
    def from_edges(
        num_nodes: int,
        edges: _EdgeArrayLike,
        weights: _WeightArrayLike | None = None,
        *,
        directed: bool = False,
        sanitize: bool = True,
    ) -> CsrGraph: ...
    @staticmethod
    def from_edge_list(
        num_nodes: int,
        edges: _EdgeArrayLike,
        weights: _WeightArrayLike | None = None,
        *,
        directed: bool = False,
        sanitize: bool = True,
    ) -> CsrGraph: ...
    @property
    def num_nodes(self) -> int: ...
    @property
    def num_edges(self) -> int: ...
    @property
    def nnz(self) -> int: ...
    @property
    def is_directed(self) -> bool: ...
    @property
    def is_weighted(self) -> bool: ...
    def degree(self, u: int) -> int: ...
    def neighbors(self, u: int) -> npt.NDArray[np.uint32]: ...
    def induced_subgraph(self, nodes: _IndexArrayLike) -> CsrGraph: ...
    def indptr(self) -> npt.NDArray[np.uint32]: ...
    def indices(self) -> npt.NDArray[np.uint32]: ...
    def weights(self) -> npt.NDArray[np.float64] | None: ...
    def largest_connected_component(self) -> tuple[CsrGraph, npt.NDArray[np.uint32]]: ...
    def __repr__(self) -> str: ...

@final
class Layout2d:
    @staticmethod
    def zeros(n: int, dtype: _DtypeLike = "float64") -> Layout2d: ...
    @staticmethod
    def init(n: int, seed: int = 42, dtype: _DtypeLike = "float64") -> Layout2d: ...
    @staticmethod
    def from_coords(coords: _CoordArrayLike) -> Layout2d: ...
    def __len__(self) -> int: ...
    @property
    def num_nodes(self) -> int: ...
    def is_empty(self) -> bool: ...
    def x(self, i: int) -> float: ...
    def y(self, i: int) -> float: ...
    def set_x(self, i: int, v: float) -> None: ...
    def set_y(self, i: int, v: float) -> None: ...
    def normalize_std(self, target_std: float = 1.0) -> None: ...
    @property
    def dtype(self) -> str: ...
    def is_all_finite(self) -> bool: ...
    def copy_coords(self) -> npt.NDArray[Any]: ...
    def coords(self, copy: bool = False) -> npt.NDArray[Any]: ...
    def __repr__(self) -> str: ...

@final
class LayoutNd:
    @staticmethod
    def zeros(n: int, dim: int, dtype: _DtypeLike = "float64") -> LayoutNd: ...
    @staticmethod
    def from_coords(coords: _CoordArrayLike) -> LayoutNd: ...
    def __len__(self) -> int: ...
    def is_empty(self) -> bool: ...
    @property
    def num_nodes(self) -> int: ...
    @property
    def dim(self) -> int: ...
    @property
    def dtype(self) -> str: ...
    def is_all_finite(self) -> bool: ...
    def coords(self, copy: bool = False) -> npt.NDArray[Any]: ...
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
class CsrMatrix:
    def __new__(
        cls,
        num_nodes: int,
        indptr: _IndexArrayLike,
        indices: _IndexArrayLike,
        values: _WeightArrayLike,
    ) -> CsrMatrix: ...
    @staticmethod
    def from_csr(
        num_nodes: int,
        indptr: _IndexArrayLike,
        indices: _IndexArrayLike,
        values: _WeightArrayLike,
    ) -> CsrMatrix: ...
    @property
    def num_nodes(self) -> int: ...
    @property
    def nnz(self) -> int: ...
    def indptr(self) -> npt.NDArray[np.uint32]: ...
    def indices(self) -> npt.NDArray[np.uint32]: ...
    def values(self) -> npt.NDArray[np.float64]: ...
    def diagonal(self) -> npt.NDArray[np.float64]: ...
    def trace(self) -> float: ...
    def gershgorin_spectral_bound(self) -> float: ...
    def estimate_lambda_max(
        self,
        *,
        num_iters: int = 10,
        safety_factor: float = 1.02,
        seed: int = 42,
    ) -> float: ...
    def estimate_lambda_max_lanczos(
        self,
        *,
        num_steps: int = 16,
        safety_factor: float = 1.05,
        seed: int = 42,
    ) -> float: ...
    def spmv(self, x: _WeightArrayLike) -> npt.NDArray[np.float64]: ...
    def scale_and_shift(self, scale: float, shift: float) -> CsrMatrix: ...
    def is_symmetric(self, tol: float = 1e-10) -> bool: ...
    def to_dense(self) -> npt.NDArray[np.float64]: ...
    def __len__(self) -> int: ...
    def __repr__(self) -> str: ...

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

@disjoint_base
class GraphAdapter:
    def add_edges_from_numpy(self, edges: npt.NDArray[np.integer[Any]]) -> None: ...
    def node_count(self) -> int: ...
    def edge_count(self) -> int: ...
    def add_node(self, value: Any) -> int: ...
    def node_weight(self, a: int) -> Any: ...
    def add_edge(self, a: int, b: int, value: Any) -> int: ...
    def edge_weight(self, e: int) -> Any: ...
    def edge_endpoints(self, e: int) -> tuple[int, int]: ...
    def remove_node(self, a: int) -> Any: ...
    def remove_edge(self, e: int) -> Any: ...
    def neighbors(self, a: int) -> list[int]: ...
    def neighbors_directed(self, a: int, dir: int) -> list[int]: ...
    def neighbors_undirected(self, a: int) -> list[int]: ...
    def edges(self, a: int) -> list[Any]: ...
    def contains_edge(self, a: int, b: int) -> bool: ...
    def find_edge(self, a: int, b: int) -> int: ...
    def externals(self, dir: int) -> list[int]: ...
    def node_indices(self) -> list[int]: ...
    def edge_indices(self) -> list[int]: ...
    def map(
        self,
        node_map: Callable[[int, Any], Any],
        edge_map: Callable[[int, Any], Any],
    ) -> GraphAdapter: ...
    def filter_map(
        self,
        node_map: Callable[[int, Any], Any],
        edge_map: Callable[[int, Any], Any],
    ) -> GraphAdapter: ...

@final
class Graph(GraphAdapter):
    def __new__(cls) -> Graph: ...
    @staticmethod
    def from_edges_numpy(
        num_nodes: int, edges: npt.NDArray[np.integer[Any]]
    ) -> Graph: ...

@final
class DiGraph(GraphAdapter):
    def __new__(cls) -> DiGraph: ...
    @staticmethod
    def from_edges_numpy(
        num_nodes: int, edges: npt.NDArray[np.integer[Any]]
    ) -> DiGraph: ...

@disjoint_base
class Drawing: ...

@final
class DrawingEuclidean2d(Drawing):
    def x(self, u: int) -> float | None: ...
    def y(self, u: int) -> float | None: ...
    def set_x(self, u: int, x: float) -> None: ...
    def set_y(self, u: int, y: float) -> None: ...
    def len(self) -> int: ...
    def centralize(self) -> None: ...
    def clamp_region(self, x0: float, y0: float, x1: float, y1: float) -> None: ...
    def edge_segments(
        self, u: int, v: int
    ) -> list[tuple[tuple[float, float], tuple[float, float]]] | None: ...
    @staticmethod
    def initial_placement(graph: GraphAdapter) -> DrawingEuclidean2d: ...
    @staticmethod
    def from_array2(
        graph: GraphAdapter, coordinates: Array2
    ) -> DrawingEuclidean2d: ...
    @staticmethod
    def from_numpy(
        graph: GraphAdapter, coords: npt.NDArray[np.float64]
    ) -> DrawingEuclidean2d: ...
    def to_numpy(self) -> npt.NDArray[np.float64]: ...

@final
class DrawingEuclidean(Drawing):
    def get(self, u: int, d: int) -> float | None: ...
    def set(self, u: int, d: int, value: float) -> None: ...
    def len(self) -> int: ...

@final
class DrawingHyperbolic2d(Drawing):
    def x(self, u: int) -> float | None: ...
    def y(self, u: int) -> float | None: ...
    def set_x(self, u: int, x: float) -> None: ...
    def set_y(self, u: int, y: float) -> None: ...
    def len(self) -> int: ...
    @staticmethod
    def initial_placement(graph: GraphAdapter) -> DrawingHyperbolic2d: ...
    def recenter(self, u: int) -> None: ...

@final
class DrawingSpherical2d(Drawing):
    def lon(self, u: int) -> float | None: ...
    def lat(self, u: int) -> float | None: ...
    def set_lon(self, u: int, value: float) -> None: ...
    def set_lat(self, u: int, value: float) -> None: ...
    def len(self) -> int: ...
    @staticmethod
    def initial_placement(graph: GraphAdapter) -> DrawingSpherical2d: ...

@final
class DrawingTorus2d(Drawing):
    def x(self, u: int) -> float | None: ...
    def y(self, u: int) -> float | None: ...
    def set_x(self, u: int, x: float) -> None: ...
    def set_y(self, u: int, y: float) -> None: ...
    def len(self) -> int: ...
    def is_empty(self) -> bool: ...
    def edge_segments(
        self, u: int, v: int
    ) -> list[tuple[tuple[float, float], tuple[float, float]]] | None: ...
    @staticmethod
    def initial_placement(graph: GraphAdapter) -> DrawingTorus2d: ...

@final
class Rng:
    def __new__(cls) -> Rng: ...
    @classmethod
    def seed_from(cls, seed: int) -> Rng: ...

@final
class PyArray1Iterator:
    def __iter__(self) -> PyArray1Iterator: ...
    def __next__(self) -> float: ...

@final
class Array1:
    def __new__(cls, array: npt.NDArray[np.float64] | None = None) -> Array1: ...
    @classmethod
    def from_numpy(cls, array: npt.NDArray[np.float64]) -> Array1: ...
    def to_numpy(self) -> npt.NDArray[np.float64]: ...
    def __len__(self) -> int: ...
    def __getitem__(self, index: int, /) -> float: ...
    def __setitem__(self, index: int, value: float, /) -> None: ...
    def __repr__(self) -> str: ...
    def __iter__(self) -> PyArray1Iterator: ...
    def to_list(self) -> list[float]: ...

@final
class Array2:
    def __new__(cls, array: npt.NDArray[np.float64] | None = None) -> Array2: ...
    @classmethod
    def from_numpy(cls, array: npt.NDArray[np.float64]) -> Array2: ...
    def to_numpy(self) -> npt.NDArray[np.float64]: ...
    @property
    def shape(self) -> tuple[int, int]: ...
    @property
    def nrows(self) -> int: ...
    @property
    def ncols(self) -> int: ...
    def __getitem__(self, index: tuple[int, int], /) -> float: ...
    def __setitem__(self, index: tuple[int, int], value: float, /) -> None: ...
    def __repr__(self) -> str: ...
    def row(self, index: int) -> Array1: ...
    def column(self, index: int) -> Array1: ...
    def to_list(self) -> list[list[float]]: ...

@final
class DistanceMatrix:
    def shape(self) -> tuple[int, int]: ...
    def pivots(self) -> list[int] | None: ...
    def get(self, u: int, v: int) -> float | None: ...
    def get_by_index(self, i: int, j: int) -> float: ...
    def set(self, u: int, v: int, d: float) -> None: ...
    @staticmethod
    def from_numpy(
        graph: GraphAdapter, array: npt.NDArray[np.float64]
    ) -> DistanceMatrix: ...
    @staticmethod
    def from_pivoted_numpy(
        graph: GraphAdapter,
        pivots: list[int],
        array: npt.NDArray[np.float64],
    ) -> DistanceMatrix: ...
    def to_numpy(self) -> npt.NDArray[np.float64]: ...

@final
class NegLogSimDistance:
    def get(self, u: int, v: int) -> float | None: ...

@final
class NegLogSimDistanceBuilder:
    def __new__(cls) -> NegLogSimDistanceBuilder: ...
    def alpha(self, alpha: float) -> NegLogSimDistanceBuilder: ...
    def beta(self, beta: float) -> NegLogSimDistanceBuilder: ...
    def p(self, p: float) -> NegLogSimDistanceBuilder: ...
    def min_dist(self, min_dist: float) -> NegLogSimDistanceBuilder: ...
    def build(self, graph: GraphAdapter, kernel: Any) -> NegLogSimDistance: ...

@final
class NegLogDistance:
    def get(self, u: int, v: int) -> float | None: ...

@final
class NegLogDistanceBuilder:
    def __new__(cls) -> NegLogDistanceBuilder: ...
    def alpha(self, alpha: float) -> NegLogDistanceBuilder: ...
    def beta(self, beta: float) -> NegLogDistanceBuilder: ...
    def p(self, p: float) -> NegLogDistanceBuilder: ...
    def min_dist(self, min_dist: float) -> NegLogDistanceBuilder: ...
    def build(self, graph: GraphAdapter, kernel: Any) -> NegLogDistance: ...

@final
class PivotedNegLogDistance:
    def get(self, u: int, v: int) -> float | None: ...

@final
class PivotedNegLogDistanceBuilder:
    def __new__(cls) -> PivotedNegLogDistanceBuilder: ...
    def alpha(self, alpha: float) -> PivotedNegLogDistanceBuilder: ...
    def beta(self, beta: float) -> PivotedNegLogDistanceBuilder: ...
    def p(self, p: float) -> PivotedNegLogDistanceBuilder: ...
    def min_dist(self, min_dist: float) -> PivotedNegLogDistanceBuilder: ...
    def build(self, graph: GraphAdapter, kernel: Any) -> PivotedNegLogDistance: ...

@final
class PivotedNegLogSimDistance:
    def __new__(
        cls,
        graph: GraphAdapter,
        kernel: Any,
        epsilon: float = 1e-12,
        scale: float = 1.0,
        power: float = 0.5,
    ) -> PivotedNegLogSimDistance: ...
    def get(self, u: int, v: int) -> float | None: ...
    def get_by_index(self, j: int, pivot_idx: int) -> float: ...
    def shape(self) -> tuple[int, int]: ...
    def pivots(self) -> list[int]: ...

@final
class PivotedNegLogSimDistanceBuilder:
    def __new__(cls) -> PivotedNegLogSimDistanceBuilder: ...
    def epsilon(self, epsilon: float) -> PivotedNegLogSimDistanceBuilder: ...
    def scale(self, scale: float) -> PivotedNegLogSimDistanceBuilder: ...
    def power(self, power: float) -> PivotedNegLogSimDistanceBuilder: ...
    def build(
        self, graph: GraphAdapter, kernel: Any
    ) -> PivotedNegLogSimDistance: ...

@final
class EmbeddingKernel:
    def __new__(cls, graph: GraphAdapter, embedding: Array2) -> EmbeddingKernel: ...
    def get(self, u: int, v: int) -> float | None: ...

@final
class EmbeddingKernelBuilder:
    def __new__(cls) -> EmbeddingKernelBuilder: ...
    def build(self, graph: GraphAdapter, embedding: Array2) -> EmbeddingKernel: ...

@final
class GaussianKernel:
    def __new__(cls, kernel: Any, gamma: float) -> GaussianKernel: ...
    def get(self, u: int, v: int) -> float | None: ...

@final
class GaussianKernelBuilder:
    def __new__(cls, gamma: float = 1.0) -> GaussianKernelBuilder: ...
    def gamma(self, gamma: float) -> GaussianKernelBuilder: ...
    def build(self, kernel: Any) -> GaussianKernel: ...

@final
class ExponentialKernel:
    def __new__(cls, kernel: Any, gamma: float) -> ExponentialKernel: ...
    def get(self, u: int, v: int) -> float | None: ...

@final
class ExponentialKernelBuilder:
    def __new__(cls, gamma: float = 1.0) -> ExponentialKernelBuilder: ...
    def gamma(self, gamma: float) -> ExponentialKernelBuilder: ...
    def build(self, kernel: Any) -> ExponentialKernel: ...

@final
class TKernel:
    def __new__(cls, kernel: Any, dof: float) -> TKernel: ...
    def get(self, u: int, v: int) -> float | None: ...

@final
class TKernelBuilder:
    def __new__(cls, alpha: float = 1.0) -> TKernelBuilder: ...
    def alpha(self, alpha: float) -> TKernelBuilder: ...
    def build(self, kernel: Any) -> TKernel: ...

@final
class KernelDistance:
    def __new__(cls, kernel: Any, min_dist: float = 0.0) -> KernelDistance: ...

@final
class KernelDistanceBuilder:
    def __new__(cls, min_dist: float = 0.0) -> KernelDistanceBuilder: ...
    def min_dist(self, min_dist: float) -> KernelDistanceBuilder: ...
    def build(self, kernel: Any) -> KernelDistance: ...

@final
class Laplacian:
    @staticmethod
    def standard(
        graph: GraphAdapter, length: Any = None
    ) -> Laplacian: ...
    @staticmethod
    def symmetric_normalized(
        graph: GraphAdapter, length: Any = None
    ) -> Laplacian: ...
    def n(self) -> int: ...

@final
class StandardLaplacian:
    @staticmethod
    def build(graph: GraphAdapter, length: Any = None) -> Laplacian: ...

@final
class SymmetricNormalizedLaplacian:
    @staticmethod
    def build(graph: GraphAdapter, length: Any = None) -> Laplacian: ...

@final
class DiffusionKernel:
    def __new__(
        cls, laplacian: Laplacian, t: float, degree: int, rng: Rng
    ) -> DiffusionKernel: ...
    @staticmethod
    def new_with_lambda_max(
        laplacian: Laplacian, t: float, degree: int, lambda_max: float, rng: Rng
    ) -> DiffusionKernel: ...
    def get(self, i: int, j: int) -> float: ...
    def n(self) -> int: ...

@final
class MultiscaleDiffusionKernel:
    def __new__(cls, laplacian: Laplacian, alpha: float) -> MultiscaleDiffusionKernel: ...
    def get(self, i: int, j: int) -> float: ...
    def n(self) -> int: ...

@final
class LowRankDiffusionKernel:
    def __new__(
        cls, t: float, eigenvalues: Array1, eigenvectors: Array2
    ) -> LowRankDiffusionKernel: ...
    def get(self, i: int, j: int) -> float: ...
    def n(self) -> int: ...

@final
class LowRankMultiscaleDiffusionKernel:
    def __new__(
        cls, alpha: float, eigenvalues: Array1, eigenvectors: Array2
    ) -> LowRankMultiscaleDiffusionKernel: ...
    def get(self, i: int, j: int) -> float: ...
    def n(self) -> int: ...

@final
class PivotedDiffusionKernel:
    def __new__(
        cls,
        laplacian: Laplacian,
        t: float,
        degree: int,
        pivots: list[int],
        rng: Rng,
        num_hutchinson: int | None = None,
    ) -> PivotedDiffusionKernel: ...
    @staticmethod
    def new_with_lambda_max(
        laplacian: Laplacian,
        t: float,
        degree: int,
        pivots: list[int],
        lambda_max: float,
        rng: Rng,
        num_hutchinson: int | None = None,
    ) -> PivotedDiffusionKernel: ...
    def pivots(self) -> list[int]: ...
    def get_from_pivot(self, pivot_idx: int, j: int) -> float: ...
    def diagonal(self, j: int) -> float | None: ...

@final
class BandPassDiffusionKernel:
    def __new__(
        cls,
        laplacian: Laplacian,
        terms: list[tuple[float, float]],
        degree: int,
        rng: Rng | None = None,
        lambda_max: float | None = None,
        off_diagonal_floor: float = 0.0,
    ) -> BandPassDiffusionKernel: ...
    @staticmethod
    def new_with_lambda_max(
        laplacian: Laplacian,
        terms: list[tuple[float, float]],
        degree: int,
        lambda_max: float,
        rng: Rng | None = None,
        off_diagonal_floor: float = 0.0,
    ) -> BandPassDiffusionKernel: ...
    @staticmethod
    def dog(
        laplacian: Laplacian,
        t1: float,
        t2: float,
        alpha: float,
        beta_floor: float,
        degree: int,
        rng: Rng | None = None,
        lambda_max: float | None = None,
    ) -> BandPassDiffusionKernel: ...
    @staticmethod
    def multiscale_heat(
        laplacian: Laplacian,
        terms: list[tuple[float, float]],
        degree: int,
        rng: Rng | None = None,
        lambda_max: float | None = None,
        off_diagonal_floor: float = 0.0,
    ) -> BandPassDiffusionKernel: ...
    def get(self, i: int, j: int) -> float: ...
    def n(self) -> int: ...

@final
class BandPassDiffusionKernelBuilder:
    def __new__(
        cls,
        terms: list[tuple[float, float]] | None = None,
        degree: int = 30,
        lambda_max: float | None = None,
        off_diagonal_floor: float = 0.0,
    ) -> BandPassDiffusionKernelBuilder: ...
    @staticmethod
    def dog(
        t1: float,
        t2: float,
        alpha: float,
        beta_floor: float,
        degree: int = 30,
        lambda_max: float | None = None,
    ) -> BandPassDiffusionKernelBuilder: ...
    @staticmethod
    def multiscale_heat(
        terms: list[tuple[float, float]],
        degree: int = 30,
        lambda_max: float | None = None,
        off_diagonal_floor: float = 0.0,
    ) -> BandPassDiffusionKernelBuilder: ...
    def terms(
        self, terms: list[tuple[float, float]]
    ) -> BandPassDiffusionKernelBuilder: ...
    def degree(self, degree: int) -> BandPassDiffusionKernelBuilder: ...
    def lambda_max(
        self, lambda_max: float | None
    ) -> BandPassDiffusionKernelBuilder: ...
    def off_diagonal_floor(
        self, floor: float
    ) -> BandPassDiffusionKernelBuilder: ...
    def build(
        self, laplacian: Laplacian, rng: Rng | None = None
    ) -> BandPassDiffusionKernel: ...

@final
class PivotedBandPassDiffusionKernel:
    def __new__(
        cls,
        laplacian: Laplacian,
        terms: list[tuple[float, float]],
        degree: int,
        pivots: list[int],
        rng: Rng | None = None,
        num_hutchinson: int | None = ...,
        lambda_max: float | None = None,
        off_diagonal_floor: float = 0.0,
    ) -> PivotedBandPassDiffusionKernel: ...
    @staticmethod
    def new_with_lambda_max(
        laplacian: Laplacian,
        terms: list[tuple[float, float]],
        degree: int,
        pivots: list[int],
        lambda_max: float,
        rng: Rng | None = None,
        num_hutchinson: int | None = ...,
        off_diagonal_floor: float = 0.0,
    ) -> PivotedBandPassDiffusionKernel: ...
    @staticmethod
    def dog(
        laplacian: Laplacian,
        t1: float,
        t2: float,
        alpha: float,
        beta_floor: float,
        degree: int,
        pivots: list[int],
        rng: Rng | None = None,
        num_hutchinson: int | None = ...,
        lambda_max: float | None = None,
    ) -> PivotedBandPassDiffusionKernel: ...
    @staticmethod
    def multiscale_heat(
        laplacian: Laplacian,
        terms: list[tuple[float, float]],
        degree: int,
        pivots: list[int],
        rng: Rng | None = None,
        num_hutchinson: int | None = ...,
        lambda_max: float | None = None,
        off_diagonal_floor: float = 0.0,
    ) -> PivotedBandPassDiffusionKernel: ...
    def pivots(self) -> list[int]: ...
    def get_from_pivot(self, pivot_idx: int, j: int) -> float: ...
    def diagonal(self, j: int) -> float | None: ...
    def n(self) -> int: ...

@final
class PivotedBandPassDiffusionKernelBuilder:
    def __new__(
        cls,
        terms: list[tuple[float, float]] | None = None,
        degree: int = 30,
        pivots: list[int] | None = None,
        lambda_max: float | None = None,
        num_hutchinson: int | None = ...,
        off_diagonal_floor: float = 0.0,
    ) -> PivotedBandPassDiffusionKernelBuilder: ...
    @staticmethod
    def dog(
        t1: float,
        t2: float,
        alpha: float,
        beta_floor: float,
        degree: int = 30,
        pivots: list[int] | None = None,
        lambda_max: float | None = None,
        num_hutchinson: int | None = ...,
    ) -> PivotedBandPassDiffusionKernelBuilder: ...
    @staticmethod
    def multiscale_heat(
        terms: list[tuple[float, float]],
        degree: int = 30,
        pivots: list[int] | None = None,
        lambda_max: float | None = None,
        num_hutchinson: int | None = ...,
        off_diagonal_floor: float = 0.0,
    ) -> PivotedBandPassDiffusionKernelBuilder: ...
    def terms(
        self, terms: list[tuple[float, float]]
    ) -> PivotedBandPassDiffusionKernelBuilder: ...
    def degree(self, degree: int) -> PivotedBandPassDiffusionKernelBuilder: ...
    def pivots(
        self, pivots: list[int]
    ) -> PivotedBandPassDiffusionKernelBuilder: ...
    def lambda_max(
        self, lambda_max: float | None
    ) -> PivotedBandPassDiffusionKernelBuilder: ...
    def num_hutchinson(
        self, num_hutchinson: int | None
    ) -> PivotedBandPassDiffusionKernelBuilder: ...
    def off_diagonal_floor(
        self, floor: float
    ) -> PivotedBandPassDiffusionKernelBuilder: ...
    def build(
        self, laplacian: Laplacian, rng: Rng | None = None
    ) -> PivotedBandPassDiffusionKernel: ...

@final
class PivotedMultiscaleDiffusionKernel:
    def __new__(
        cls, laplacian: Laplacian, alpha: float, pivots: list[int]
    ) -> PivotedMultiscaleDiffusionKernel: ...
    def pivots(self) -> list[int]: ...
    def get_from_pivot(self, pivot_idx: int, j: int) -> float: ...
    def diagonal(self, j: int) -> float | None: ...

@final
class LowRankDiffusionKernelBuilder:
    def __new__(cls) -> LowRankDiffusionKernelBuilder: ...
    def t(self, t: float) -> LowRankDiffusionKernelBuilder: ...
    def rank(self, rank: int) -> LowRankDiffusionKernelBuilder: ...
    def shift(self, shift: float) -> LowRankDiffusionKernelBuilder: ...
    def eigenvalue_max_iterations(
        self, eigenvalue_max_iterations: int
    ) -> LowRankDiffusionKernelBuilder: ...
    def cg_max_iterations(
        self, cg_max_iterations: int
    ) -> LowRankDiffusionKernelBuilder: ...
    def eigenvalue_tolerance(
        self, eigenvalue_tolerance: float
    ) -> LowRankDiffusionKernelBuilder: ...
    def cg_tolerance(self, cg_tolerance: float) -> LowRankDiffusionKernelBuilder: ...
    def build_unnormalized_laplacian(
        self,
        graph: GraphAdapter,
        length: Any = None,
        rng: Rng | None = None,
    ) -> LowRankDiffusionKernel: ...
    def build_symmetric_normalized_laplacian(
        self,
        graph: GraphAdapter,
        length: Any = None,
        rng: Rng | None = None,
    ) -> LowRankDiffusionKernel: ...

@final
class LowRankMultiscaleDiffusionKernelBuilder:
    def __new__(cls) -> LowRankMultiscaleDiffusionKernelBuilder: ...
    def alpha(self, alpha: float) -> LowRankMultiscaleDiffusionKernelBuilder: ...
    def rank(self, rank: int) -> LowRankMultiscaleDiffusionKernelBuilder: ...
    def shift(self, shift: float) -> LowRankMultiscaleDiffusionKernelBuilder: ...
    def eigenvalue_max_iterations(
        self, eigenvalue_max_iterations: int
    ) -> LowRankMultiscaleDiffusionKernelBuilder: ...
    def cg_max_iterations(
        self, cg_max_iterations: int
    ) -> LowRankMultiscaleDiffusionKernelBuilder: ...
    def eigenvalue_tolerance(
        self, eigenvalue_tolerance: float
    ) -> LowRankMultiscaleDiffusionKernelBuilder: ...
    def cg_tolerance(
        self, cg_tolerance: float
    ) -> LowRankMultiscaleDiffusionKernelBuilder: ...
    def build_unnormalized_laplacian(
        self,
        graph: GraphAdapter,
        length: Any = None,
        rng: Rng | None = None,
    ) -> LowRankMultiscaleDiffusionKernel: ...
    def build_symmetric_normalized_laplacian(
        self,
        graph: GraphAdapter,
        length: Any = None,
        rng: Rng | None = None,
    ) -> LowRankMultiscaleDiffusionKernel: ...

@final
class ClassicalMds:
    def __new__(cls, graph: GraphAdapter, f: Callable[[int], float]) -> ClassicalMds: ...
    @staticmethod
    def new_with_distance_matrix(d: Any) -> ClassicalMds: ...
    def run(self, d: int) -> DrawingEuclidean: ...
    def run_2d(self) -> DrawingEuclidean2d: ...
    @property
    def eps(self) -> float: ...
    @eps.setter
    def eps(self, value: float) -> None: ...

@final
class PivotMds:
    def __new__(
        cls, graph: GraphAdapter, f: Callable[[int], float], pivot: list[int]
    ) -> PivotMds: ...
    @staticmethod
    def new_with_distance_matrix(d: Any) -> PivotMds: ...
    def run(self, d: int) -> DrawingEuclidean: ...
    def run_2d(self) -> DrawingEuclidean2d: ...
    @property
    def eps(self) -> float: ...
    @eps.setter
    def eps(self, value: float) -> None: ...

@final
class CgSolver:
    def __new__(
        cls, max_iterations: int = 100, tolerance: float = 1e-4
    ) -> CgSolver: ...

@final
class JacobiCgSolver:
    def __new__(
        cls, max_iterations: int = 100, tolerance: float = 1e-4
    ) -> JacobiCgSolver: ...

@final
class Ic0CgSolver:
    def __new__(
        cls, max_iterations: int = 100, tolerance: float = 1e-4
    ) -> Ic0CgSolver: ...

@final
class AmgCgSolver:
    def __new__(
        cls, max_iterations: int = 100, tolerance: float = 1e-4
    ) -> AmgCgSolver: ...

@final
class EigendecompositionResult:
    @property
    def eigenvectors(self) -> Array2: ...
    @property
    def eigenvalues(self) -> Array1: ...
    @property
    def cg_iterations(self) -> list[int]: ...
    @property
    def power_iterations(self) -> list[int]: ...

@final
class RdMds:
    def __new__(cls) -> RdMds: ...
    def d(self, d: int) -> RdMds: ...
    def shift(self, shift: float) -> RdMds: ...
    def eigenvalue_max_iterations(
        self, eigenvalue_max_iterations: int
    ) -> RdMds: ...
    def eigenvalue_tolerance(self, eigenvalue_tolerance: float) -> RdMds: ...
    def embedding(
        self,
        graph: GraphAdapter,
        length: Callable[[int], float],
        solver: CgSolver | JacobiCgSolver | Ic0CgSolver | AmgCgSolver,
        rng: Rng,
    ) -> Array2: ...
    def embedding_symmetric_normalized(
        self,
        graph: GraphAdapter,
        length: Callable[[int], float],
        solver: CgSolver | JacobiCgSolver | Ic0CgSolver | AmgCgSolver,
        rng: Rng,
    ) -> Array2: ...
    def embedding_random_walk_normalized(
        self,
        graph: GraphAdapter,
        length: Callable[[int], float],
        solver: CgSolver | JacobiCgSolver | Ic0CgSolver | AmgCgSolver,
        rng: Rng,
    ) -> Array2: ...
    def eigendecomposition(
        self,
        graph: GraphAdapter,
        length: Callable[[int], float],
        solver: CgSolver | JacobiCgSolver | Ic0CgSolver | AmgCgSolver,
        rng: Rng,
    ) -> EigendecompositionResult: ...
    def eigendecomposition_symmetric_normalized(
        self,
        graph: GraphAdapter,
        length: Callable[[int], float],
        solver: CgSolver | JacobiCgSolver | Ic0CgSolver | AmgCgSolver,
        rng: Rng,
    ) -> EigendecompositionResult: ...
    def eigendecomposition_random_walk_normalized(
        self,
        graph: GraphAdapter,
        length: Callable[[int], float],
        solver: CgSolver | JacobiCgSolver | Ic0CgSolver | AmgCgSolver,
        rng: Rng,
    ) -> EigendecompositionResult: ...

@final
class KamadaKawai:
    def __new__(cls, graph: GraphAdapter, f: Callable[[int], float]) -> KamadaKawai: ...
    def select_node(self, drawing: DrawingEuclidean2d) -> int | None: ...
    def apply_to_node(self, m: int, drawing: DrawingEuclidean2d) -> None: ...
    def run(self, drawing: DrawingEuclidean2d) -> None: ...
    @property
    def eps(self) -> float: ...
    @eps.setter
    def eps(self, value: float) -> None: ...

@final
class OverwrapRemoval:
    def __new__(cls, graph: GraphAdapter, f: Callable[[int], float]) -> OverwrapRemoval: ...
    def apply_with_drawing_euclidean_2d(
        self, drawing: DrawingEuclidean2d
    ) -> None: ...
    def apply_with_drawing_euclidean(self, drawing: DrawingEuclidean) -> None: ...
    def apply_with_drawing_hyperbolic_2d(
        self, drawing: DrawingHyperbolic2d
    ) -> None: ...
    def apply_with_drawing_spherical_2d(
        self, drawing: DrawingSpherical2d
    ) -> None: ...
    def apply_with_drawing_torus_2d(self, drawing: DrawingTorus2d) -> None: ...
    @property
    def strength(self) -> float: ...
    @strength.setter
    def strength(self, value: float) -> None: ...
    @property
    def iterations(self) -> int: ...
    @iterations.setter
    def iterations(self, value: int) -> None: ...
    @property
    def min_distance(self) -> float: ...
    @min_distance.setter
    def min_distance(self, value: float) -> None: ...

@final
class StressMajorization:
    def __new__(
        cls,
        graph: GraphAdapter,
        drawing: DrawingEuclidean2d,
        f: Any = None,
    ) -> StressMajorization: ...
    @classmethod
    def with_distance_matrix(
        cls, drawing: DrawingEuclidean2d, distance_matrix: DistanceMatrix
    ) -> StressMajorization: ...
    def apply(self, drawing: DrawingEuclidean2d) -> float: ...
    def run(self, drawing: DrawingEuclidean2d) -> None: ...
    def update_weight(
        self, f: Callable[[int, int, float, float], float]
    ) -> None: ...
    def update_weight_from_numpy(self, w: npt.NDArray[np.float64]) -> None: ...
    @property
    def epsilon(self) -> float: ...
    @epsilon.setter
    def epsilon(self, value: float) -> None: ...
    @property
    def max_iterations(self) -> int: ...
    @max_iterations.setter
    def max_iterations(self, value: int) -> None: ...

@final
class SchedulerConstant:
    def __new__(
        cls,
        t_max: int,
        eta_min: float | None = None,
        eta_max: float | None = None,
    ) -> SchedulerConstant: ...
    def run(self, f: Callable[[float], Any]) -> None: ...
    def step(self, f: Callable[[float], Any]) -> None: ...
    def is_finished(self) -> bool: ...

@final
class SchedulerLinear:
    def __new__(
        cls,
        t_max: int,
        eta_min: float | None = None,
        eta_max: float | None = None,
    ) -> SchedulerLinear: ...
    def run(self, f: Callable[[float], Any]) -> None: ...
    def step(self, f: Callable[[float], Any]) -> None: ...
    def is_finished(self) -> bool: ...

@final
class SchedulerQuadratic:
    def __new__(
        cls,
        t_max: int,
        eta_min: float | None = None,
        eta_max: float | None = None,
    ) -> SchedulerQuadratic: ...
    def run(self, f: Callable[[float], Any]) -> None: ...
    def step(self, f: Callable[[float], Any]) -> None: ...
    def is_finished(self) -> bool: ...

@final
class SchedulerExponential:
    def __new__(
        cls,
        t_max: int,
        eta_min: float | None = None,
        eta_max: float | None = None,
    ) -> SchedulerExponential: ...
    def run(self, f: Callable[[float], Any]) -> None: ...
    def step(self, f: Callable[[float], Any]) -> None: ...
    def is_finished(self) -> bool: ...

@final
class SchedulerReciprocal:
    def __new__(
        cls,
        t_max: int,
        eta_min: float | None = None,
        eta_max: float | None = None,
    ) -> SchedulerReciprocal: ...
    def run(self, f: Callable[[float], Any]) -> None: ...
    def step(self, f: Callable[[float], Any]) -> None: ...
    def is_finished(self) -> bool: ...

@final
class Sgd:
    def __new__(
        cls,
        node_pairs: list[tuple[int, int, float, float, float, float]],
    ) -> Sgd: ...
    def shuffle(self, rng: Rng) -> None: ...
    def apply(self, drawing: Drawing, eta: float) -> None: ...
    def update_distance(
        self, f: Callable[[int, int, float, float], float]
    ) -> None: ...
    def update_weight(
        self, f: Callable[[int, int, float, float], float]
    ) -> None: ...
    @staticmethod
    def from_numpy_pairs(
        sources: npt.NDArray[np.integer[Any]],
        targets: npt.NDArray[np.integer[Any]],
        distances: npt.NDArray[np.float64],
        weights: npt.NDArray[np.float64],
        n_nodes: int | None = None,
    ) -> Sgd: ...
    @staticmethod
    def from_numpy_hinge_pairs(
        sources: npt.NDArray[np.integer[Any]],
        targets: npt.NDArray[np.integer[Any]],
        distances: npt.NDArray[np.float64],
        weights: npt.NDArray[np.float64],
        n_nodes: int | None = None,
    ) -> Sgd: ...
    @staticmethod
    def from_numpy_pairs_with_hinges(
        sources: npt.NDArray[np.integer[Any]],
        targets: npt.NDArray[np.integer[Any]],
        distances: npt.NDArray[np.float64],
        weights: npt.NDArray[np.float64],
        hinge_sources: npt.NDArray[np.integer[Any]],
        hinge_targets: npt.NDArray[np.integer[Any]],
        hinge_distances: npt.NDArray[np.float64],
        hinge_weights: npt.NDArray[np.float64],
        n_nodes: int | None = None,
    ) -> Sgd: ...
    def add_hinge_pairs_from_numpy(
        self,
        sources: npt.NDArray[np.integer[Any]],
        targets: npt.NDArray[np.integer[Any]],
        distances: npt.NDArray[np.float64],
        weights: npt.NDArray[np.float64],
        n_nodes: int | None = None,
    ) -> None: ...
    def node_pair_count(self) -> int: ...
    def hinge_pair_count(self) -> int: ...
    def update_weight_from_numpy(self, w: npt.NDArray[np.float64]) -> None: ...
    def update_distance_from_numpy(
        self,
        d: npt.NDArray[np.float64],
        w: npt.NDArray[np.float64] | None = None,
    ) -> None: ...
    def scheduler(self, t_max: int, epsilon: float) -> SchedulerExponential: ...
    def scheduler_constant(
        self, t_max: int, epsilon: float
    ) -> SchedulerConstant: ...
    def scheduler_linear(
        self, t_max: int, epsilon: float
    ) -> SchedulerLinear: ...
    def scheduler_exponential(
        self, t_max: int, epsilon: float
    ) -> SchedulerExponential: ...
    def scheduler_quadratic(
        self, t_max: int, epsilon: float
    ) -> SchedulerQuadratic: ...
    def scheduler_reciprocal(
        self, t_max: int, epsilon: float
    ) -> SchedulerReciprocal: ...

@final
class FullSgd:
    def __new__(cls) -> FullSgd: ...
    def build(self, graph: GraphAdapter, f: Any = None) -> Sgd: ...
    def build_with_numpy(
        self,
        n: int,
        d: npt.NDArray[np.float64],
        w: npt.NDArray[np.float64] | None = None,
    ) -> Sgd: ...
    def build_with_distance_matrix(self, d: Any) -> Sgd: ...

@final
class SparseSgd:
    def __new__(cls) -> SparseSgd: ...
    @staticmethod
    def barrier(
        h: int = 64,
        calibrate_one_hop: bool = True,
        max_2hop_per_node: int = 32,
        hinge_repulsion: bool = True,
        delta_rel: float = 0.25,
        barrier_sym: str = "sym_max",
        omega_1: float = 1.0,
        omega_2: float = 1.0,
        gamma_non2: float = 1.0,
        min_ratio: float = 0.5,
        eps_dist: float = 1e-6,
        seed: int = 42,
    ) -> SparseSgd: ...
    def h(self, h: int) -> SparseSgd: ...
    def calibrate_one_hop(self, enabled: bool) -> SparseSgd: ...
    def max_2hop_per_node(self, cap: int) -> SparseSgd: ...
    def hinge_repulsion(self, enabled: bool) -> SparseSgd: ...
    def delta_rel(self, delta_rel: float) -> SparseSgd: ...
    def barrier_sym(self, barrier_sym: str) -> SparseSgd: ...
    def omega_1(self, omega_1: float) -> SparseSgd: ...
    def omega_2(self, omega_2: float) -> SparseSgd: ...
    def gamma_non2(self, gamma_non2: float) -> SparseSgd: ...
    def min_ratio(self, min_ratio: float) -> SparseSgd: ...
    def eps_dist(self, eps_dist: float) -> SparseSgd: ...
    def seed(self, seed: int) -> SparseSgd: ...
    def build(
        self,
        graph: GraphAdapter,
        f: Any = None,
        rng: Rng | None = None,
        *,
        length_fn: Any = None,
    ) -> Sgd: ...
    def new_with_pivot(
        self,
        graph: GraphAdapter,
        f: Any = None,
        pivot: list[int] | None = None,
        *,
        length_fn: Any = None,
    ) -> Sgd: ...
    def build_with_pivot_and_distance_matrix(
        self,
        graph: GraphAdapter,
        f: Any = None,
        pivot: Any = None,
        d: Any = None,
        *,
        length_fn: Any = None,
    ) -> Sgd: ...
    def build_with_pivoted_distance(
        self,
        graph: GraphAdapter,
        f: Any = None,
        d: Any = None,
        *,
        length_fn: Any = None,
    ) -> Sgd: ...
    def build_barrier_from_landmark_tables(
        self,
        graph: GraphAdapter,
        d_ms: npt.NDArray[np.float64],
        d_coarse: npt.NDArray[np.float64] | None = None,
        pivots: _IndexArrayLike | None = None,
        neg_per_node: int | None = None,
    ) -> tuple[Sgd, dict[str, Any]]: ...

@final
class RandomPairSparseSgd:
    def __new__(cls) -> RandomPairSparseSgd: ...
    def k(self, k: int) -> RandomPairSparseSgd: ...
    def build(
        self, graph: GraphAdapter, distance_matrix: Any, rng: Rng
    ) -> Sgd: ...

@final
class Omega:
    def __new__(cls) -> Omega: ...
    def k(self, k: int) -> Omega: ...
    def min_dist(self, min_dist: float) -> Omega: ...
    def build(
        self, graph: GraphAdapter, embedding: Array2, rng: Rng
    ) -> Sgd: ...

@final
class TsNetBuilder:
    def __new__(cls) -> TsNetBuilder: ...
    @staticmethod
    def new_star() -> TsNetBuilder: ...
    def perplexity(self, perplexity: float) -> TsNetBuilder: ...
    def n_epochs(self, n_epochs: int) -> TsNetBuilder: ...
    def initial_lr(self, initial_lr: float) -> TsNetBuilder: ...
    def final_lr(self, final_lr: float) -> TsNetBuilder: ...
    def lr_switch(self, lr_switch: int | None) -> TsNetBuilder: ...
    def learning_rate(self, learning_rate: float) -> TsNetBuilder: ...
    def initial_momentum(self, initial_momentum: float) -> TsNetBuilder: ...
    def final_momentum(self, final_momentum: float) -> TsNetBuilder: ...
    def momentum_switch(self, momentum_switch: int | None) -> TsNetBuilder: ...
    def momentum(self, momentum: float) -> TsNetBuilder: ...
    def initial_l_kl(self, initial_l_kl: float | None) -> TsNetBuilder: ...
    def final_l_kl(self, final_l_kl: float | None) -> TsNetBuilder: ...
    def l_kl_switch(self, l_kl_switch: int | None) -> TsNetBuilder: ...
    def initial_l_c(self, initial_l_c: float | None) -> TsNetBuilder: ...
    def final_l_c(self, final_l_c: float | None) -> TsNetBuilder: ...
    def l_c_switch(self, l_c_switch: int | None) -> TsNetBuilder: ...
    def initial_l_r(self, initial_l_r: float | None) -> TsNetBuilder: ...
    def final_l_r(self, final_l_r: float | None) -> TsNetBuilder: ...
    def l_r_switch(self, l_r_switch: int | None) -> TsNetBuilder: ...
    def r_eps(self, r_eps: float) -> TsNetBuilder: ...
    def sigma_iters(self, sigma_iters: int) -> TsNetBuilder: ...
    def autostop(self, autostop: float) -> TsNetBuilder: ...
    def window_size(self, window_size: int) -> TsNetBuilder: ...
    def build(self) -> TsNet: ...

@final
class TsNet:
    def __new__(cls) -> TsNet: ...
    @staticmethod
    def new_star() -> TsNet: ...
    @staticmethod
    def builder() -> TsNetBuilder: ...
    @staticmethod
    def builder_star() -> TsNetBuilder: ...
    def run(self, drawing: DrawingEuclidean2d, distance_matrix: Any) -> None: ...

@final
class BhTsNetBuilder:
    def __new__(cls) -> BhTsNetBuilder: ...
    def perplexity(self, perplexity: float) -> BhTsNetBuilder: ...
    def theta(self, theta: float) -> BhTsNetBuilder: ...
    def k(self, k: int) -> BhTsNetBuilder: ...
    def iterations_stage1(self, iterations: int) -> BhTsNetBuilder: ...
    def exaggeration(self, exaggeration: float) -> BhTsNetBuilder: ...
    def iterations_stage2(self, iterations: int) -> BhTsNetBuilder: ...
    def lambda_c_stage2(self, lambda_c: float) -> BhTsNetBuilder: ...
    def iterations_stage3(self, iterations: int) -> BhTsNetBuilder: ...
    def lambda_c_stage3(self, lambda_c: float) -> BhTsNetBuilder: ...
    def lambda_r_stage3(self, lambda_r: float) -> BhTsNetBuilder: ...
    def lambda_r(self, lambda_r: float) -> BhTsNetBuilder: ...
    def learning_rate(self, learning_rate: float) -> BhTsNetBuilder: ...
    def momentum(self, momentum: float) -> BhTsNetBuilder: ...
    def power(self, power: float) -> BhTsNetBuilder: ...
    def epsilon_r(self, epsilon_r: float) -> BhTsNetBuilder: ...
    def sigma_iters(self, sigma_iters: int) -> BhTsNetBuilder: ...
    def sigma_tolerance(self, sigma_tolerance: float) -> BhTsNetBuilder: ...
    def build(self) -> BhTsNet: ...

@final
class BhTsNet:
    def __new__(cls) -> BhTsNet: ...
    @staticmethod
    def builder() -> BhTsNetBuilder: ...
    def run(
        self, drawing: DrawingEuclidean2d, graph: GraphAdapter, rng: Rng
    ) -> None: ...
    def run_with_knn(
        self,
        drawing: DrawingEuclidean2d,
        knn_indices: npt.NDArray[np.integer[Any]],
        knn_distances: npt.NDArray[np.float64],
    ) -> None: ...
    def run_with_knn_adaptive(
        self,
        drawing: DrawingEuclidean2d,
        indptr: npt.NDArray[np.integer[Any]],
        indices: npt.NDArray[np.integer[Any]],
        distances: npt.NDArray[np.float64],
        perplexities: npt.NDArray[np.float64],
    ) -> None: ...

@final
class FitTsNetBuilder:
    def __new__(cls) -> FitTsNetBuilder: ...
    def intervals(self, intervals: int) -> FitTsNetBuilder: ...
    def interpolation_points(
        self, interpolation_points: int
    ) -> FitTsNetBuilder: ...
    def perplexity(self, perplexity: float) -> FitTsNetBuilder: ...
    def theta(self, theta: float) -> FitTsNetBuilder: ...
    def k(self, k: int) -> FitTsNetBuilder: ...
    def iterations_stage1(self, iterations: int) -> FitTsNetBuilder: ...
    def exaggeration(self, exaggeration: float) -> FitTsNetBuilder: ...
    def iterations_stage2(self, iterations: int) -> FitTsNetBuilder: ...
    def lambda_c_stage2(self, lambda_c: float) -> FitTsNetBuilder: ...
    def iterations_stage3(self, iterations: int) -> FitTsNetBuilder: ...
    def lambda_c_stage3(self, lambda_c: float) -> FitTsNetBuilder: ...
    def lambda_r_stage3(self, lambda_r: float) -> FitTsNetBuilder: ...
    def lambda_r(self, lambda_r: float) -> FitTsNetBuilder: ...
    def learning_rate(self, learning_rate: float) -> FitTsNetBuilder: ...
    def momentum(self, momentum: float) -> FitTsNetBuilder: ...
    def power(self, power: float) -> FitTsNetBuilder: ...
    def epsilon_r(self, epsilon_r: float) -> FitTsNetBuilder: ...
    def sigma_iters(self, sigma_iters: int) -> FitTsNetBuilder: ...
    def sigma_tolerance(self, sigma_tolerance: float) -> FitTsNetBuilder: ...
    def build(self) -> FitTsNet: ...

@final
class FitTsNet:
    def __new__(cls) -> FitTsNet: ...
    @staticmethod
    def builder() -> FitTsNetBuilder: ...
    def run(
        self, drawing: DrawingEuclidean2d, graph: GraphAdapter, rng: Rng
    ) -> None: ...
    def run_with_knn(
        self,
        drawing: DrawingEuclidean2d,
        knn_indices: npt.NDArray[np.integer[Any]],
        knn_distances: npt.NDArray[np.float64],
    ) -> None: ...
    def run_with_knn_adaptive(
        self,
        drawing: DrawingEuclidean2d,
        indptr: npt.NDArray[np.integer[Any]],
        indices: npt.NDArray[np.integer[Any]],
        distances: npt.NDArray[np.float64],
        perplexities: npt.NDArray[np.float64],
    ) -> None: ...

@final
class LTsNetBuilder:
    def __new__(cls) -> LTsNetBuilder: ...
    def intervals(self, intervals: int) -> LTsNetBuilder: ...
    def interpolation_points(
        self, interpolation_points: int
    ) -> LTsNetBuilder: ...
    def perplexity(self, perplexity: float) -> LTsNetBuilder: ...
    def k(self, k: int) -> LTsNetBuilder: ...
    def iterations_stage1(self, iterations: int) -> LTsNetBuilder: ...
    def exaggeration(self, exaggeration: float) -> LTsNetBuilder: ...
    def iterations_stage2(self, iterations: int) -> LTsNetBuilder: ...
    def lambda_c_stage2(self, lambda_c: float) -> LTsNetBuilder: ...
    def iterations_stage3(self, iterations: int) -> LTsNetBuilder: ...
    def lambda_c_stage3(self, lambda_c: float) -> LTsNetBuilder: ...
    def lambda_r_stage3(self, lambda_r: float) -> LTsNetBuilder: ...
    def lambda_r(self, lambda_r: float) -> LTsNetBuilder: ...
    def learning_rate(self, learning_rate: float) -> LTsNetBuilder: ...
    def momentum(self, momentum: float) -> LTsNetBuilder: ...
    def power(self, power: float) -> LTsNetBuilder: ...
    def epsilon_r(self, epsilon_r: float) -> LTsNetBuilder: ...
    def sigma_iters(self, sigma_iters: int) -> LTsNetBuilder: ...
    def sigma_tolerance(self, sigma_tolerance: float) -> LTsNetBuilder: ...
    def build(self) -> LTsNet: ...

@final
class LTsNet:
    def __new__(cls) -> LTsNet: ...
    @staticmethod
    def builder() -> LTsNetBuilder: ...
    def run(
        self, drawing: DrawingEuclidean2d, graph: GraphAdapter, rng: Rng
    ) -> None: ...
    def run_with_knn(
        self,
        drawing: DrawingEuclidean2d,
        knn_indices: npt.NDArray[np.integer[Any]],
        knn_distances: npt.NDArray[np.float64],
    ) -> None: ...
    def run_with_knn_adaptive(
        self,
        drawing: DrawingEuclidean2d,
        indptr: npt.NDArray[np.integer[Any]],
        indices: npt.NDArray[np.integer[Any]],
        distances: npt.NDArray[np.float64],
        perplexities: npt.NDArray[np.float64],
    ) -> None: ...

@final
class GuMapBuilder:
    def __new__(cls) -> GuMapBuilder: ...
    def k(self, k: int) -> GuMapBuilder: ...
    def min_dist(self, min_dist: float) -> GuMapBuilder: ...
    def spread(self, spread: float) -> GuMapBuilder: ...
    def n_epochs(self, n_epochs: int) -> GuMapBuilder: ...
    def negative_sample_rate(
        self, negative_sample_rate: int
    ) -> GuMapBuilder: ...
    def learning_rate(self, learning_rate: float) -> GuMapBuilder: ...
    def a(self, a: float) -> GuMapBuilder: ...
    def b(self, b: float) -> GuMapBuilder: ...
    def sigma_iters(self, sigma_iters: int) -> GuMapBuilder: ...
    def sigma_tolerance(self, sigma_tolerance: float) -> GuMapBuilder: ...
    def build(self) -> GuMap: ...

@final
class GuMap:
    def __new__(cls) -> GuMap: ...
    @staticmethod
    def builder() -> GuMapBuilder: ...
    def run(
        self, drawing: DrawingEuclidean2d, distance_matrix: Any, rng: Rng
    ) -> None: ...
    def run_with_knn(
        self,
        drawing: DrawingEuclidean2d,
        knn_indices: npt.NDArray[np.integer[Any]],
        knn_distances: npt.NDArray[np.float64],
        rng: Rng,
    ) -> None: ...
    def run_with_knn_csr(
        self,
        drawing: DrawingEuclidean2d,
        indptr: npt.NDArray[np.integer[Any]],
        indices: npt.NDArray[np.integer[Any]],
        distances: npt.NDArray[np.float64],
        rng: Rng,
    ) -> None: ...

@final
class WeightedEdgeLength:
    def __new__(cls, graph: GraphAdapter) -> WeightedEdgeLength: ...
    def __call__(self, edge_index: int) -> int: ...

@final
class LongestPath:
    def __new__(cls) -> LongestPath: ...
    def assign_layers(self, graph: GraphAdapter) -> dict[int, int]: ...

@final
class Constraint:
    def __new__(cls, left: int, right: int, gap: float) -> Constraint: ...
    @property
    def left(self) -> int: ...
    @property
    def right(self) -> int: ...
    @property
    def gap(self) -> float: ...

@final
class Louvain:
    def __new__(cls) -> Louvain: ...
    def detect_communities(self, graph: GraphAdapter) -> dict[int, int]: ...

@final
class LabelPropagation:
    def __new__(cls) -> LabelPropagation: ...
    def detect_communities(self, graph: GraphAdapter) -> dict[int, int]: ...

@final
class SpectralClustering:
    def __new__(cls, k: int) -> SpectralClustering: ...
    def detect_communities(self, graph: GraphAdapter) -> dict[int, int]: ...

def all_sources_bfs(
    graph: GraphAdapter, unit_edge_length: float
) -> DistanceMatrix: ...
def all_sources_dijkstra(
    graph: GraphAdapter, f: Any = None
) -> DistanceMatrix: ...
def warshall_floyd(
    graph: GraphAdapter, f: Any = None
) -> DistanceMatrix: ...
@overload
def cycle_edges(graph: GraphAdapter) -> list[tuple[int, int]]: ...
@overload
def cycle_edges(graph: CsrGraph) -> npt.NDArray[np.uint32]: ...
def remove_cycle(graph: GraphAdapter) -> None: ...
def triangulation(drawing: DrawingEuclidean2d) -> GraphAdapter: ...
@overload
def project_1d(
    target: DrawingEuclidean2d, arg2: int, arg3: list[Constraint]
) -> None: ...
@overload
def project_1d(
    target: _WeightArrayLike,
    arg2: Sequence[SeparationConstraint],
    arg3: None = None,
) -> npt.NDArray[Any]: ...
def project_rectangle_no_overlap_constraints_2d(
    drawing: DrawingEuclidean2d, size_fn: Callable[[int, int], float]
) -> None: ...
@overload
def generate_layered_constraints(
    graph: GraphAdapter, layers_or_gap: float
) -> list[Constraint]: ...
@overload
def generate_layered_constraints(
    graph: GraphAdapter, *, gap: float
) -> list[Constraint]: ...
@overload
def generate_layered_constraints(
    graph: CsrGraph,
    layers_or_gap: _IndexArrayLike | None = None,
    min_layer_distance: float = ...,
    *,
    layers: _IndexArrayLike | None = None,
    gap: float | None = None,
) -> list[SeparationConstraint]: ...
def project_clustered_rectangle_no_overlap_constraints(
    graph: GraphAdapter,
    drawing: DrawingEuclidean2d,
    cluster_id_fn: Callable[[int], int],
    size_fn: Callable[[int, int], float],
) -> None: ...
@overload
def crossing_edges(graph: CsrGraph, drawing: Layout2d) -> list[dict[str, Any]]: ...
@overload
def crossing_edges(graph: GraphAdapter, drawing: Drawing) -> Any: ...
@overload
def angular_resolution(
    graph: GraphAdapter, drawing: DrawingEuclidean2d
) -> float: ...
@overload
def angular_resolution(graph: CsrGraph, drawing: Layout2d) -> float: ...
@overload
def aspect_ratio(drawing: DrawingEuclidean2d) -> float: ...
@overload
def aspect_ratio(drawing: Layout2d) -> float: ...
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
def crossing_number(
    graph: GraphAdapter, drawing: DrawingEuclidean2d
) -> float: ...
@overload
def crossing_number(graph: CsrGraph, drawing: Layout2d) -> float: ...
def crossing_number_with_crossing_edges(
    crossing_edges: Any,
) -> float: ...
@overload
def gabriel_graph_property(
    graph: GraphAdapter, drawing: DrawingEuclidean2d
) -> float: ...
@overload
def gabriel_graph_property(graph: CsrGraph, drawing: Layout2d) -> float: ...
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
@overload
def neighborhood_preservation(
    graph: GraphAdapter, drawing: DrawingEuclidean2d
) -> float: ...
@overload
def neighborhood_preservation(graph: CsrGraph, drawing: Layout2d) -> float: ...
@overload
def neighborhood_preservation_2hop(
    graph: GraphAdapter, drawing: DrawingEuclidean2d
) -> float: ...
@overload
def neighborhood_preservation_2hop(graph: CsrGraph, drawing: Layout2d) -> float: ...
@overload
def node_resolution(drawing: Drawing) -> float: ...
@overload
def node_resolution(drawing: Layout2d) -> float: ...
def stress(drawing: Drawing, distance_matrix: DistanceMatrix) -> float: ...
def scale_normalized_stress(graph: CsrGraph, layout: Layout2d) -> float: ...
def sampled_scale_normalized_stress(
    graph: CsrGraph,
    layout: Layout2d,
    *,
    num_pivots: int = 64,
    seed: int = 42,
) -> float: ...
def sweep_crossing_number(graph: CsrGraph, layout: Layout2d) -> float: ...
def sweep_crossing_angle(graph: CsrGraph, layout: Layout2d) -> float: ...
def sampled_crossing_number(
    graph: CsrGraph,
    layout: Layout2d,
    *,
    num_samples: int = 4096,
    seed: int = 42,
) -> float: ...
def crosslessness(graph: CsrGraph, layout: Layout2d) -> float: ...
def sampled_crosslessness(
    graph: CsrGraph,
    layout: Layout2d,
    *,
    num_samples: int = 4096,
    seed: int = 42,
) -> float: ...
def minimum_angle(graph: CsrGraph, layout: Layout2d) -> float: ...
def minimum_angle_summary(graph: CsrGraph, layout: Layout2d) -> dict[str, float]: ...
def layout_degeneracy(graph: CsrGraph, layout: Layout2d) -> dict[str, Any]: ...
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
    landmark_dists: _IndexArrayLike | Sequence[Sequence[int]],
    *,
    num_triplets: int = 1000000,
    seed: int = 42,
) -> dict[str, Any]: ...
def torus_crossing_number(graph: CsrGraph, layout: Layout2d) -> float: ...
def neighborhood_preservation_1hop(graph: CsrGraph, layout: Layout2d) -> float: ...
def neighborhood_preservation_2hop_v6(graph: CsrGraph, layout: Layout2d) -> float: ...
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
def edge_angle(x1: float, y1: float, x2: float, y2: float) -> float | None: ...
def prism_distortion(
    original_layout: Layout2d, adjusted_layout: Layout2d
) -> tuple[float, float]: ...
def drng_preservation(graph: CsrGraph, layout: Layout2d) -> float: ...
def dgg_preservation(graph: CsrGraph, layout: Layout2d) -> float: ...
def gabriel_graph_property_v6(graph: CsrGraph, layout: Layout2d) -> float: ...
def node_resolution_v6(layout: Layout2d) -> float: ...
def aspect_ratio_v6(layout: Layout2d) -> float: ...
def angular_resolution_v6(graph: CsrGraph, layout: Layout2d) -> float: ...
def ideal_edge_lengths_v6(graph: CsrGraph, layout: Layout2d) -> float: ...
def ideal_edge_lengths_with_unit(
    graph: CsrGraph, layout: Layout2d, unit_edge_length: float = 1.0
) -> float: ...
def crossing_edges_torus(graph: CsrGraph, layout: Layout2d) -> list[dict[str, Any]]: ...
def crossing_summary(graph: CsrGraph, layout: Layout2d) -> dict[str, Any]: ...
def gabriel_graph_edges(layout: Layout2d) -> npt.NDArray[np.uint32]: ...
def relative_neighborhood_edges(layout: Layout2d) -> npt.NDArray[np.uint32]: ...
def all_pairs_bfs(
    graph: CsrGraph, unit_edge_length: float = 1.0
) -> npt.NDArray[np.float64]: ...
def bfs_single_source(
    graph: CsrGraph, source: int, unit_edge_length: float = 1.0
) -> npt.NDArray[np.float64]: ...
def scale_normalized_stress_from_distances(
    layout: Layout2d,
    distances: _MatrixArrayLike,
    weights: _MatrixArrayLike | None = None,
) -> float: ...
def prism_distortion_with_edges(
    original_layout: Layout2d,
    adjusted_layout: Layout2d,
    edges: _EdgeArrayLike,
) -> tuple[float, float]: ...
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
) -> dict[str, float]: ...
def delaunay_triangulation(
    layout: Layout2d,
) -> tuple[npt.NDArray[np.uint32], npt.NDArray[np.uint32]]: ...
def delaunay_to_csr(layout: Layout2d) -> CsrGraph: ...
def rectangle_overlaps(
    layout: Layout2d,
    half_widths: _WeightArrayLike,
    half_heights: _WeightArrayLike,
) -> npt.NDArray[np.uint32]: ...
def load_dataset(name: str) -> CsrGraph: ...
def parse_csr_csv(
    csv: str, *, directed: bool = False, sanitize: bool = True
) -> CsrGraph: ...
def select_maxmin_sp_pivots(
    graph: CsrGraph,
    num_pivots: int = 50,
    *,
    seed: int = 42,
    min_degree: int = 2,
) -> npt.NDArray[np.uint32]: ...
def pivot_mds_init(
    graph: CsrGraph,
    *,
    num_pivots: int = 50,
    seed: int = 42,
    min_degree: int = 2,
    target_std: float | None = 1.0,
    dtype: _DtypeLike = "float64",
) -> Layout2d: ...
def koren_spectral_init(
    graph: CsrGraph,
    *,
    target_std: float | None = 1.0,
    tol: float = 1e-4,
    max_iters: int = 300,
    seed: int = 42,
    dtype: _DtypeLike = "float64",
) -> Layout2d: ...
def smallest_laplacian_eigenpairs(
    graph: CsrGraph,
    n_modes: int = 16,
    *,
    tol: float = 1e-4,
    max_iters: int = 300,
    seed: int = 42,
) -> tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    bool,
]: ...
def spectral_moment_stats(graph: CsrGraph) -> dict[str, float]: ...
def anisotropic_laplacian(
    graph: CsrGraph,
    *,
    alpha: float = 0.5,
    beta: float = 0.0,
    degree_reg: float = 0.0,
    degree_cap: float | None = None,
    reweighted_degree_norm: bool = True,
) -> tuple[CsrMatrix, npt.NDArray[np.float64]]: ...
def chebyshev_heat_spmv(
    graph: CsrGraph,
    vectors: _VectorArrayLike,
    t: float,
    *,
    degree: int | None = None,
    lambda_max: float = 2.0,
) -> npt.NDArray[np.float64]: ...
def chebyshev_heat_spmv_matrix(
    matrix: CsrMatrix,
    vectors: _VectorArrayLike,
    t: float,
    *,
    degree: int | None = None,
    lambda_max: float | None = None,
    lanczos_steps: int = 16,
    safety_factor: float = 1.05,
    seed: int = 42,
) -> npt.NDArray[np.float64]: ...
def chebyshev_dog_jl_embedding(
    graph: CsrGraph,
    *,
    t1: float | None = None,
    t2: float | None = None,
    alpha: float | None = None,
    degree: int = 25,
    jl_dim: int = 64,
    seed: int = 42,
    degree_normalize: bool = True,
    eps_psd: float = 1e-4,
) -> npt.NDArray[np.float64]: ...
def chebyshev_dog_jl_embedding_matrix(
    matrix: CsrMatrix,
    *,
    t1: float | None = None,
    t2: float | None = None,
    alpha: float | None = None,
    degree: int | None = None,
    jl_dim: int = 64,
    seed: int = 42,
    eps_psd: float = 1e-4,
    lambda_max: float | None = None,
    lanczos_steps: int = 16,
    safety_factor: float = 1.05,
    scale_vec: _WeightArrayLike | None = None,
) -> npt.NDArray[np.float64]: ...
def extract_1hop_2hop_csr(graph: CsrGraph) -> tuple[CsrGraph, CsrGraph]: ...
def extract_sparse_1hop_2hop_pairs(
    graph: CsrGraph,
    max_2hop_per_node: int = 32,
    seed: int = 42,
) -> Sparse12HopPairs: ...
def build_pure_2hop_knn(
    graph: CsrGraph,
) -> tuple[KnnGraph, npt.NDArray[np.uint32], dict[str, Any]]: ...
def build_capped_12hop_knn(
    graph: CsrGraph,
    max_2hop_per_node: int = 32,
    *,
    seed: int = 42,
) -> tuple[KnnGraph, npt.NDArray[np.uint32], dict[str, Any]]: ...
def compute_2hop_neighborhood_sizes(graph: CsrGraph) -> npt.NDArray[np.uint32]: ...
def compute_nodewise_2hop_perplexities(
    graph: CsrGraph,
    *,
    c: float = 0.9,
    min_perplexity: float = 5.0,
    max_perplexity: float = 1200.0,
) -> npt.NDArray[np.float64]: ...
def compute_median_2hop_perplexity(
    graph: CsrGraph,
    *,
    c: float = 1.5,
    min_perplexity: float = 15.0,
    max_perplexity: float = 1200.0,
) -> float: ...
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
def partial_bfs_knn(
    graph: CsrGraph,
    k: int = 30,
    *,
    seed: int = 42,
) -> KnnGraph: ...
def compute_knn_probabilities(
    knn: KnnGraph,
    perplexities: _WeightArrayLike | float,
    *,
    power: float = 2.0,
    sigma_iters: int = 50,
    sigma_tolerance: float = 1e-5,
    row_normalize: bool = False,
) -> SparseProbabilities: ...
def compute_fuzzy_simplicial_set(
    knn: KnnGraph,
    *,
    sigma_iters: int = 64,
    sigma_tolerance: float = 1e-5,
) -> SparseProbabilities: ...
def find_ab_params(min_dist: float = 0.1, spread: float = 1.0) -> tuple[float, float]: ...
def compute_horvitz_thompson_gamma(
    gamma_non2: float, num_nodes: int, neg_per_node: int
) -> float: ...
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

@final
class AmgPcgSolver:
    def __new__(
        cls,
        num_nodes: int,
        indptr: _IndexArrayLike,
        indices: _IndexArrayLike,
        values: _WeightArrayLike,
        *,
        rel_tol: float = 1e-8,
        abs_tol: float = 0.0,
        max_iterations: int = 500,
        theta: float = 0.25,
        max_levels: int = 32,
        max_coarse_size: int = 64,
        pre_sweeps: int = 1,
        post_sweeps: int = 1,
        max_agg_size: int = 8,
        diag_eps: float = 1e-14,
        eig_eps: float = 1e-12,
        sym_tol: float = 1e-10,
        seed: int = 42,
        randomize_ties: bool = False,
        near_nullspace: _WeightArrayLike | None = None,
    ) -> AmgPcgSolver: ...
    @staticmethod
    def from_csr(
        num_nodes: int,
        indptr: _IndexArrayLike,
        indices: _IndexArrayLike,
        values: _WeightArrayLike,
        *,
        rel_tol: float = 1e-8,
        abs_tol: float = 0.0,
        max_iterations: int = 500,
        theta: float = 0.25,
        max_levels: int = 32,
        max_coarse_size: int = 64,
        pre_sweeps: int = 1,
        post_sweeps: int = 1,
        max_agg_size: int = 8,
        diag_eps: float = 1e-14,
        eig_eps: float = 1e-12,
        sym_tol: float = 1e-10,
        seed: int = 42,
        randomize_ties: bool = False,
        near_nullspace: _WeightArrayLike | None = None,
    ) -> AmgPcgSolver: ...
    @staticmethod
    def from_csr_matrix(
        matrix: CsrMatrix,
        *,
        rel_tol: float = 1e-8,
        abs_tol: float = 0.0,
        max_iterations: int = 500,
        theta: float = 0.25,
        max_levels: int = 32,
        max_coarse_size: int = 64,
        pre_sweeps: int = 1,
        post_sweeps: int = 1,
        max_agg_size: int = 8,
        diag_eps: float = 1e-14,
        eig_eps: float = 1e-12,
        sym_tol: float = 1e-10,
        seed: int = 42,
        randomize_ties: bool = False,
        near_nullspace: _WeightArrayLike | None = None,
    ) -> AmgPcgSolver: ...
    @staticmethod
    def from_graph_regularized_laplacian(
        graph: CsrGraph,
        gamma: float = 1.0,
        *,
        rel_tol: float = 1e-8,
        abs_tol: float = 0.0,
        max_iterations: int = 500,
        theta: float = 0.25,
        max_levels: int = 32,
        max_coarse_size: int = 64,
        pre_sweeps: int = 1,
        post_sweeps: int = 1,
        max_agg_size: int = 8,
        diag_eps: float = 1e-14,
        eig_eps: float = 1e-12,
        sym_tol: float = 1e-10,
        seed: int = 42,
        randomize_ties: bool = False,
    ) -> AmgPcgSolver: ...
    @staticmethod
    def from_graph_laplacian(
        graph: CsrGraph,
        kind: str = "combinatorial",
        shift: float = 0.0,
        *,
        rel_tol: float = 1e-8,
        abs_tol: float = 0.0,
        max_iterations: int = 500,
        theta: float = 0.25,
        max_levels: int = 32,
        max_coarse_size: int = 64,
        pre_sweeps: int = 1,
        post_sweeps: int = 1,
        max_agg_size: int = 8,
        diag_eps: float = 1e-14,
        eig_eps: float = 1e-12,
        sym_tol: float = 1e-10,
        seed: int = 42,
        randomize_ties: bool = False,
    ) -> AmgPcgSolver: ...
    def solve(
        self,
        b: _WeightArrayLike,
        x0: _WeightArrayLike | None = None,
    ) -> tuple[npt.NDArray[np.float64], dict[str, Any]]: ...
    def solve_batch(
        self,
        b: _MatrixArrayLike,
        x0: _MatrixArrayLike | None = None,
    ) -> tuple[npt.NDArray[np.float64], dict[str, Any]]: ...
    def start_solve(
        self,
        b: _WeightArrayLike,
        x0: _WeightArrayLike | None = None,
    ) -> str: ...
    def step(self) -> str: ...
    def run_steps(self, steps: int) -> str: ...
    def run(self) -> str: ...
    def solution(self) -> npt.NDArray[np.float64]: ...
    @property
    def num_nodes(self) -> int: ...
    @property
    def num_levels(self) -> int: ...
    @property
    def coarsest_size(self) -> int: ...
    @property
    def operator_complexity(self) -> float: ...
    @property
    def max_level_nnz(self) -> int: ...
    @property
    def status(self) -> str: ...
    @property
    def iteration(self) -> int: ...
    @property
    def total_iterations(self) -> int: ...
    @property
    def is_finished(self) -> bool: ...
    @property
    def residual_norm(self) -> float: ...
    @property
    def relative_residual(self) -> float: ...
    @property
    def converged(self) -> bool: ...
    def __repr__(self) -> str: ...

@final
class EffectiveResistanceResult:
    @property
    def num_nodes(self) -> int: ...
    @property
    def num_edges(self) -> int: ...
    @property
    def num_components(self) -> int: ...
    @property
    def foster_sum(self) -> float: ...
    @property
    def expected_foster(self) -> float: ...
    @property
    def foster_rel_err(self) -> float: ...
    @property
    def used_exact(self) -> bool: ...
    @property
    def amg_total_iterations(self) -> int: ...
    @property
    def amg_max_relative_residual(self) -> float: ...
    @property
    def amg_all_converged(self) -> bool: ...
    def edge_u(self) -> npt.NDArray[np.uint32]: ...
    def edge_v(self) -> npt.NDArray[np.uint32]: ...
    def edge_lengths(self) -> npt.NDArray[np.float64]: ...
    def conductances(self) -> npt.NDArray[np.float64]: ...
    def effective_resistances(self) -> npt.NDArray[np.float64]: ...
    def to_dict(self) -> dict[str, Any]: ...
    def __len__(self) -> int: ...
    def __repr__(self) -> str: ...

@final
class ResistanceCurvatureFlowResult:
    @property
    def num_nodes(self) -> int: ...
    @property
    def num_edges(self) -> int: ...
    @property
    def num_components(self) -> int: ...
    @property
    def foster_sum(self) -> float: ...
    @property
    def expected_foster(self) -> float: ...
    @property
    def foster_rel_err(self) -> float: ...
    @property
    def mean_link_curvature(self) -> float: ...
    @property
    def std_link_curvature(self) -> float: ...
    @property
    def min_link_curvature(self) -> float: ...
    @property
    def max_link_curvature(self) -> float: ...
    @property
    def mean_vertex_curvature(self) -> float: ...
    @property
    def min_vertex_curvature(self) -> float: ...
    @property
    def max_vertex_curvature(self) -> float: ...
    @property
    def used_exact(self) -> bool: ...
    @property
    def amg_total_iterations(self) -> int: ...
    @property
    def amg_max_relative_residual(self) -> float: ...
    @property
    def amg_all_converged(self) -> bool: ...
    @property
    def suggested_geometry(self) -> str: ...
    @property
    def suggested_manifold(self) -> str | None: ...
    def edge_u(self) -> npt.NDArray[np.uint32]: ...
    def edge_v(self) -> npt.NDArray[np.uint32]: ...
    def edge_lengths(self) -> npt.NDArray[np.float64]: ...
    def conductances(self) -> npt.NDArray[np.float64]: ...
    def effective_resistances(self) -> npt.NDArray[np.float64]: ...
    def vertex_curvatures(self) -> npt.NDArray[np.float64]: ...
    def link_curvatures(self) -> npt.NDArray[np.float64]: ...
    def step_history(self) -> list[dict[str, Any]]: ...
    def length_graph(self) -> CsrGraph: ...
    def conductance_graph(self) -> CsrGraph: ...
    def to_dict(self) -> dict[str, Any]: ...
    def __len__(self) -> int: ...
    def __repr__(self) -> str: ...

def compute_effective_resistances(
    graph: CsrGraph,
    *,
    jl_dim: int = 64,
    seed: int = 42,
    weights_are_lengths: bool = True,
    exact: bool | None = None,
    exact_max_nodes: int = 64,
    eig_eps: float = 1e-12,
    rel_tol: float = 1e-8,
    max_iterations: int = 500,
) -> EffectiveResistanceResult: ...
def resistance_curvature_flow(
    graph: CsrGraph,
    *,
    steps: int = 2,
    eta: float = 0.2,
    kappa_max: float = 2.0,
    min_edge_length: float = 1e-6,
    jl_dim: int = 64,
    seed: int = 42,
    weights_are_lengths: bool = True,
    exact: bool | None = None,
    exact_max_nodes: int = 64,
    eig_eps: float = 1e-12,
    rel_tol: float = 1e-8,
    max_iterations: int = 500,
    hyperbolic_threshold: float = -0.05,
    spherical_threshold: float = 0.2,
    torus_std_threshold: float = 0.05,
) -> ResistanceCurvatureFlowResult: ...

@final
class ManifoldLayout2d:
    @staticmethod
    def zeros(kind: str, n: int, dtype: _DtypeLike = "float64") -> ManifoldLayout2d: ...
    @staticmethod
    def hyperbolic_poincare(
        n: int, dtype: _DtypeLike = "float64"
    ) -> ManifoldLayout2d: ...
    @staticmethod
    def spherical(n: int, dtype: _DtypeLike = "float64") -> ManifoldLayout2d: ...
    @staticmethod
    def torus_2d(n: int, dtype: _DtypeLike = "float64") -> ManifoldLayout2d: ...
    @staticmethod
    def from_layout(kind: str, layout: Layout2d) -> ManifoldLayout2d: ...
    @staticmethod
    def from_coords(kind: str, coords: _CoordArrayLike) -> ManifoldLayout2d: ...
    @staticmethod
    def random_init(
        kind: str,
        n: int,
        *,
        scale: float = 0.25,
        seed: int = 42,
        dtype: _DtypeLike = "float64",
    ) -> ManifoldLayout2d: ...
    @staticmethod
    def from_layout_scaled(
        kind: str,
        layout: Layout2d,
        *,
        max_radius: float = 0.7,
    ) -> ManifoldLayout2d: ...
    @property
    def kind(self) -> str: ...
    @property
    def manifold(self) -> str: ...
    @property
    def dtype(self) -> str: ...
    def len(self) -> int: ...
    @property
    def num_nodes(self) -> int: ...
    def is_empty(self) -> bool: ...
    def __len__(self) -> int: ...
    def x(self, i: int) -> float: ...
    def y(self, i: int) -> float: ...
    def is_all_finite(self) -> bool: ...
    def set_point(self, index: int, x: float, y: float) -> None: ...
    def coords(self) -> npt.NDArray[Any]: ...
    def to_layout(self) -> Layout2d: ...
    def geodesic_distance(self, i: int, j: int) -> float: ...
    def tangent_delta(self, i: int, j: int) -> tuple[float, float]: ...
    def apply_tangent_step(self, i: int, vx: float, vy: float) -> None: ...
    def project_all(self) -> None: ...
    def poincare_centroid(self) -> tuple[float, float]: ...
    def recenter_poincare(self, cx: float, cy: float) -> None: ...
    def recenter_poincare_at(self, index: int) -> None: ...
    def center_poincare_centroid(self) -> None: ...
    def __repr__(self) -> str: ...

def poincare_distance(x1: float, y1: float, x2: float, y2: float) -> float: ...
def poincare_mobius_add(x1: float, y1: float, x2: float, y2: float) -> tuple[float, float]: ...
def spherical_distance(
    lon1: float, colat1: float, lon2: float, colat2: float
) -> float: ...
def torus_distance(x1: float, y1: float, x2: float, y2: float) -> float: ...
def wrap_torus_point(x: float, y: float) -> tuple[float, float]: ...

@final
class TsNetSolver:
    @staticmethod
    def pure_2hop(
        graph: CsrGraph,
        backend: str = "fft",
        *,
        perplexity_c: float = 0.9,
        min_perplexity: float = 5.0,
        max_perplexity: float = 1200.0,
        power: float = 2.0,
        sigma_iters: int = 50,
        sigma_tolerance: float = 1e-5,
        row_normalize: bool = False,
        iterations_stage1: int = 250,
        iterations_stage2: int = 250,
        iterations_stage3: int = 500,
        exaggeration: float = 4.0,
        lambda_c_stage1: float = 0.1,
        lambda_c_stage2: float = 0.01,
        lambda_c_stage3: float = 0.0,
        lambda_r_stage3: float = 0.0,
        learning_rate: float | None = None,
        momentum: float = 0.8,
        epsilon_r: float = 0.05,
        theta: float = 0.5,
        intervals: int = 25,
        interpolation_order: int = 1,
        min_grid_dim: int = 64,
        max_grid_dim: int = 512,
        jacobi_precondition: bool = True,
        seed: int = 42,
        dtype: _DtypeLike = "float64",
    ) -> TsNetSolver: ...
    @staticmethod
    def from_knn(
        knn: KnnGraph,
        perplexities: _WeightArrayLike | float | None = None,
        uniform_perplexity: float = 15.0,
        *,
        backend: str = "fft",
        power: float = 2.0,
        sigma_iters: int = 50,
        sigma_tolerance: float = 1e-5,
        row_normalize: bool = False,
        iterations_stage1: int = 250,
        iterations_stage2: int = 250,
        iterations_stage3: int = 500,
        exaggeration: float = 4.0,
        lambda_c_stage1: float = 0.1,
        lambda_c_stage2: float = 0.01,
        lambda_c_stage3: float = 0.0,
        lambda_r_stage3: float = 0.0,
        learning_rate: float | None = None,
        momentum: float = 0.8,
        epsilon_r: float = 0.05,
        theta: float = 0.5,
        intervals: int = 25,
        interpolation_order: int = 1,
        min_grid_dim: int = 64,
        max_grid_dim: int = 512,
        jacobi_precondition: bool = True,
        seed: int = 42,
        dtype: _DtypeLike = "float64",
    ) -> TsNetSolver: ...
    @staticmethod
    def from_probabilities(
        probs: SparseProbabilities,
        backend: str = "fft",
        *,
        iterations_stage1: int = 250,
        iterations_stage2: int = 250,
        iterations_stage3: int = 500,
        exaggeration: float = 4.0,
        lambda_c_stage1: float = 0.1,
        lambda_c_stage2: float = 0.01,
        lambda_c_stage3: float = 0.0,
        lambda_r_stage3: float = 0.0,
        learning_rate: float | None = None,
        momentum: float = 0.8,
        epsilon_r: float = 0.05,
        theta: float = 0.5,
        intervals: int = 25,
        interpolation_order: int = 1,
        min_grid_dim: int = 64,
        max_grid_dim: int = 512,
        jacobi_precondition: bool = True,
        seed: int = 42,
        dtype: _DtypeLike = "float64",
    ) -> TsNetSolver: ...
    @staticmethod
    def from_graph_partial_bfs(
        graph: CsrGraph,
        k: int = 15,
        uniform_perplexity: float = 15.0,
        *,
        backend: str = "fft",
        power: float = 2.0,
        sigma_iters: int = 50,
        sigma_tolerance: float = 1e-5,
        row_normalize: bool = False,
        iterations_stage1: int = 250,
        iterations_stage2: int = 250,
        iterations_stage3: int = 500,
        exaggeration: float = 4.0,
        lambda_c_stage1: float = 0.1,
        lambda_c_stage2: float = 0.01,
        lambda_c_stage3: float = 0.0,
        lambda_r_stage3: float = 0.0,
        learning_rate: float | None = None,
        momentum: float = 0.8,
        epsilon_r: float = 0.05,
        theta: float = 0.5,
        intervals: int = 25,
        interpolation_order: int = 1,
        min_grid_dim: int = 64,
        max_grid_dim: int = 512,
        jacobi_precondition: bool = True,
        seed: int = 42,
        dtype: _DtypeLike = "float64",
    ) -> TsNetSolver: ...
    @staticmethod
    def from_graph_row_normalized(
        graph: CsrGraph,
        w2: float = 0.5,
        max_2hop_per_node: int | None = None,
        *,
        degree_beta: float = 0.0,
        backend: str = "fft",
        iterations_stage1: int = 250,
        iterations_stage2: int = 250,
        iterations_stage3: int = 500,
        exaggeration: float = 4.0,
        lambda_c_stage1: float = 0.1,
        lambda_c_stage2: float = 0.01,
        lambda_c_stage3: float = 0.0,
        lambda_r_stage3: float = 0.0,
        learning_rate: float | None = None,
        momentum: float = 0.8,
        epsilon_r: float = 0.05,
        theta: float = 0.5,
        intervals: int = 25,
        interpolation_order: int = 1,
        min_grid_dim: int = 64,
        max_grid_dim: int = 512,
        jacobi_precondition: bool = True,
        seed: int = 42,
        dtype: _DtypeLike = "float64",
    ) -> TsNetSolver: ...
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
    def total_fft_2d_calls(self) -> int: ...
    def last_fft_stats(self) -> dict[str, int] | None: ...
    @property
    def num_nodes(self) -> int: ...
    @property
    def stage_index(self) -> int: ...
    @property
    def current_stage_lr(self) -> float: ...
    @property
    def dtype(self) -> str: ...
    def __repr__(self) -> str: ...

@final
class SgdSolver:
    @staticmethod
    def from_pairs(
        num_nodes: int,
        src: _IndexArrayLike,
        tgt: _IndexArrayLike,
        dij: _WeightArrayLike,
        dji: _WeightArrayLike | None = None,
        wij: _WeightArrayLike | None = None,
        wji: _WeightArrayLike | None = None,
        *,
        t_max: int = 60,
        eps: float = 0.01,
        schedule: str = "exponential",
        step_scale: float = 1.0,
        seed: int = 42,
        dtype: _DtypeLike = "float64",
    ) -> SgdSolver: ...
    @staticmethod
    def from_pairs_with_hinges(
        num_nodes: int,
        spring_src: _IndexArrayLike,
        spring_tgt: _IndexArrayLike,
        spring_dij: _WeightArrayLike,
        spring_dji: _WeightArrayLike | None,
        spring_wij: _WeightArrayLike | None,
        spring_wji: _WeightArrayLike | None,
        hinge_src: _IndexArrayLike,
        hinge_tgt: _IndexArrayLike,
        hinge_dij: _WeightArrayLike,
        hinge_dji: _WeightArrayLike | None = None,
        hinge_wij: _WeightArrayLike | None = None,
        hinge_wji: _WeightArrayLike | None = None,
        *,
        t_max: int = 60,
        eps: float = 0.01,
        schedule: str = "exponential",
        step_scale: float = 1.0,
        seed: int = 42,
        dtype: _DtypeLike = "float64",
    ) -> SgdSolver: ...
    @staticmethod
    def full(
        graph: CsrGraph,
        *,
        t_max: int = 60,
        eps: float = 0.01,
        schedule: str = "exponential",
        step_scale: float = 1.0,
        seed: int = 42,
        dtype: _DtypeLike = "float64",
    ) -> SgdSolver: ...
    @staticmethod
    def sparse_barrier_from_landmarks(
        graph: CsrGraph,
        landmarks: DogVaradhanLandmarks,
        pivots: _IndexArrayLike | None = None,
        *,
        max_2hop_per_node: int = 32,
        hinge_repulsion: bool = True,
        delta_rel: float = 0.25,
        barrier_sym: str = "sym_max",
        omega_1: float = 1.0,
        omega_2: float = 1.0,
        gamma_non2: float = 1.0,
        ht_repulsion: bool = True,
        min_ratio: float = 0.5,
        eps_dist: float = 1e-6,
        neg_per_node: int | None = None,
        num_pivots: int = 50,
        core_pivots: bool = True,
        degree_weight_beta: float = 0.0,
        t_max: int = 60,
        eps: float = 0.01,
        schedule: str = "exponential",
        step_scale: float = 1.0,
        seed: int = 42,
        dtype: _DtypeLike = "float64",
    ) -> SgdSolver: ...
    @staticmethod
    def sparse_barrier_dog(
        graph: CsrGraph,
        *,
        degree: int | None = None,
        num_landmarks: int = 64,
        degree_normalize: bool = True,
        p_exp: float = 0.5,
        eps_psd: float = 1e-4,
        t_coarse: float = 48.0,
        t1: float | None = None,
        t2: float | None = None,
        alpha: float | None = None,
        beta_floor: float | None = None,
        min_degree: int = 2,
        jl_dim: int = 64,
        max_2hop_per_node: int = 32,
        hinge_repulsion: bool = True,
        delta_rel: float = 0.25,
        barrier_sym: str = "sym_max",
        omega_1: float = 1.0,
        omega_2: float = 1.0,
        gamma_non2: float = 1.0,
        ht_repulsion: bool = True,
        min_ratio: float = 0.5,
        eps_dist: float = 1e-6,
        neg_per_node: int | None = None,
        num_pivots: int = 50,
        core_pivots: bool = True,
        degree_weight_beta: float = 0.0,
        t_max: int = 60,
        eps: float = 0.01,
        schedule: str = "exponential",
        step_scale: float = 1.0,
        seed: int = 42,
        dtype: _DtypeLike = "float64",
    ) -> SgdSolver: ...
    @staticmethod
    def random_pair(
        graph: CsrGraph,
        pairs_per_node: int = 30,
        *,
        t_max: int = 60,
        eps: float = 0.01,
        schedule: str = "exponential",
        step_scale: float = 1.0,
        seed: int = 42,
        dtype: _DtypeLike = "float64",
    ) -> SgdSolver: ...
    @staticmethod
    def omega(
        graph: CsrGraph,
        embedding: _MatrixArrayLike,
        k: int = 30,
        min_dist: float = 1e-3,
        *,
        t_max: int = 60,
        eps: float = 0.01,
        schedule: str = "exponential",
        step_scale: float = 1.0,
        seed: int = 42,
        dtype: _DtypeLike = "float64",
    ) -> SgdSolver: ...
    @staticmethod
    def streaming_negatives(
        graph: CsrGraph,
        *,
        negatives_per_node: int = 5,
        target_distance: float = 2.0,
        weight: float = 0.25,
        hinge_only: bool = True,
        t_max: int = 60,
        eps: float = 0.01,
        schedule: str = "exponential",
        step_scale: float = 1.0,
        seed: int = 42,
        dtype: _DtypeLike = "float64",
    ) -> SgdSolver: ...
    @staticmethod
    def snap_t_fdp(
        graph: CsrGraph,
        *,
        iterations: int = 200,
        negatives_per_edge: int = 3,
        learning_rate: float = 1.0,
        alpha: float = 1.0,
        repulsion_scale: float = 1.0,
        rho_start: float = 1.0,
        rho_end: float = 0.1,
        rho_power: float = 1.0,
        degree_beta: float = 0.0,
        bounded_attraction: bool = True,
        seed: int = 42,
        dtype: _DtypeLike = "float64",
    ) -> SgdSolver: ...
    @staticmethod
    def pacmap_graph(
        graph: CsrGraph,
        *,
        n_neighbors: int = 10,
        mn_ratio: float = 0.5,
        fp_ratio: float = 2.0,
        phase1_iters: int = 100,
        phase2_iters: int = 100,
        phase3_iters: int = 250,
        learning_rate: float = 1.0,
        keep_all_1hop: bool = False,
        max_1hop_per_node: int | None = None,
        degree_weight_beta: float = 0.0,
        num_landmarks: int = 16,
        landmarks: DogVaradhanLandmarks | _MatrixArrayLike | None = None,
        seed: int = 42,
        dtype: _DtypeLike = "float64",
    ) -> SgdSolver: ...
    @staticmethod
    def pacmap_graph_with_landmarks(
        graph: CsrGraph,
        landmarks: DogVaradhanLandmarks | _MatrixArrayLike,
        *,
        n_neighbors: int = 10,
        mn_ratio: float = 0.5,
        fp_ratio: float = 2.0,
        phase1_iters: int = 100,
        phase2_iters: int = 100,
        phase3_iters: int = 250,
        learning_rate: float = 1.0,
        keep_all_1hop: bool = False,
        max_1hop_per_node: int | None = None,
        degree_weight_beta: float = 0.0,
        num_landmarks: int = 16,
        seed: int = 42,
        dtype: _DtypeLike = "float64",
    ) -> SgdSolver: ...
    @staticmethod
    def pacmap_graph_with_backbone(
        graph: CsrGraph,
        *,
        n_neighbors: int = 10,
        mn_ratio: float = 0.5,
        fp_ratio: float = 2.0,
        phase1_iters: int = 100,
        phase2_iters: int = 100,
        phase3_iters: int = 250,
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
    ) -> tuple[SgdSolver, Layout2d, dict[str, Any]]: ...
    @staticmethod
    def sgd2(
        graph: CsrGraph,
        *,
        t_max: int = 60,
        eps: float = 0.01,
        step_scale: float = 1.0,
        batch_size: int = 4096,
        damping: float = 1e-4,
        seed: int = 42,
        dtype: _DtypeLike = "float64",
    ) -> SgdSolver: ...
    def step(self, layout: Layout2d) -> str: ...
    def run_steps(self, layout: Layout2d, steps: int) -> str: ...
    def run(self, layout: Layout2d) -> str: ...
    def step_manifold(self, layout: ManifoldLayout2d) -> str: ...
    def run_steps_manifold(self, layout: ManifoldLayout2d, steps: int) -> str: ...
    def run_manifold(self, layout: ManifoldLayout2d) -> str: ...
    def stats(self) -> dict[str, Any] | None: ...
    def dog_stats(self) -> dict[str, Any] | None: ...
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
    def num_springs(self) -> int: ...
    @property
    def num_hinges(self) -> int: ...
    @property
    def current_eta(self) -> float: ...
    @property
    def dtype(self) -> str: ...
    def __repr__(self) -> str: ...

@final
class GuMapSolver:
    @staticmethod
    def from_fuzzy_set(
        probs: SparseProbabilities,
        *,
        n_epochs: int = 200,
        negative_sample_rate: int = 5,
        learning_rate: float = 1.0,
        min_dist: float = 0.1,
        spread: float = 1.0,
        a: float | None = None,
        b: float | None = None,
        sigma_iters: int = 64,
        sigma_tolerance: float = 1e-5,
        symmetric_negative: bool = True,
        seed: int = 42,
        dtype: _DtypeLike = "float64",
    ) -> GuMapSolver: ...
    @staticmethod
    def from_knn(
        knn: KnnGraph,
        *,
        n_epochs: int = 200,
        negative_sample_rate: int = 5,
        learning_rate: float = 1.0,
        min_dist: float = 0.1,
        spread: float = 1.0,
        a: float | None = None,
        b: float | None = None,
        sigma_iters: int = 64,
        sigma_tolerance: float = 1e-5,
        symmetric_negative: bool = True,
        seed: int = 42,
        dtype: _DtypeLike = "float64",
    ) -> GuMapSolver: ...
    @staticmethod
    def from_graph_dog(
        graph: CsrGraph,
        max_2hop_per_node: int = 32,
        *,
        degree: int | None = None,
        num_landmarks: int = 64,
        degree_normalize: bool = True,
        p_exp: float = 0.5,
        eps_psd: float = 1e-4,
        t_coarse: float = 48.0,
        t1: float | None = None,
        t2: float | None = None,
        alpha: float | None = None,
        beta_floor: float | None = None,
        min_degree: int = 2,
        jl_dim: int = 64,
        n_epochs: int = 200,
        negative_sample_rate: int = 5,
        learning_rate: float = 1.0,
        min_dist: float = 0.1,
        spread: float = 1.0,
        a: float | None = None,
        b: float | None = None,
        sigma_iters: int = 64,
        sigma_tolerance: float = 1e-5,
        symmetric_negative: bool = True,
        seed: int = 42,
        dtype: _DtypeLike = "float64",
    ) -> GuMapSolver: ...
    @staticmethod
    def from_landmarks(
        graph: CsrGraph,
        landmarks: DogVaradhanLandmarks,
        max_2hop_per_node: int = 32,
        *,
        n_epochs: int = 200,
        negative_sample_rate: int = 5,
        learning_rate: float = 1.0,
        min_dist: float = 0.1,
        spread: float = 1.0,
        a: float | None = None,
        b: float | None = None,
        sigma_iters: int = 64,
        sigma_tolerance: float = 1e-5,
        symmetric_negative: bool = True,
        seed: int = 42,
        dtype: _DtypeLike = "float64",
    ) -> GuMapSolver: ...
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
    def num_edges(self) -> int: ...
    @property
    def num_nodes(self) -> int: ...
    @property
    def ab(self) -> tuple[float, float]: ...
    @property
    def dtype(self) -> str: ...
    def __repr__(self) -> str: ...

@final
class SpatialIntruderRefiner:
    def __new__(
        cls,
        graph: CsrGraph,
        rounds: int = 4,
        k_cap: int = 32,
        margin_scale: float = 1.2,
        w_anchor: float = 1.0,
        w_fn: float = 0.0,
        w_fp: float = 0.2,
        fn_shrink: float = 0.88,
        round_decay: float = 0.75,
        t_max: int = 20,
        eps: float = 0.05,
        hinge_fp: bool = True,
        seed: int = 42,
        *,
        w_anchor2: float = 0.0,
        k_cap_decay: float = 1.0,
        k_cap_min: int = 1,
        min_sep_ratio: float = 0.0,
        w_sep: float = 0.0,
        dtype: _DtypeLike = "float64",
    ) -> SpatialIntruderRefiner: ...
    def step(self, layout: Layout2d) -> str: ...
    def run_steps(self, layout: Layout2d, steps: int) -> str: ...
    def run(self, layout: Layout2d) -> str: ...
    def stats(self) -> dict[str, Any]: ...
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
    def current_k_cap(self) -> int: ...
    def set_k_cap(self, k_cap: int) -> None: ...
    @property
    def dtype(self) -> str: ...
    def __repr__(self) -> str: ...

@final
class StressMajorizationSolver:
    @staticmethod
    def from_graph(
        graph: CsrGraph,
        *,
        epsilon: float = 1e-4,
        max_iterations: int = 100,
        pcg_tolerance: float = 1e-6,
        pcg_max_iterations: int | None = None,
        min_norm: float = 1e-4,
        dtype: _DtypeLike = "float64",
    ) -> StressMajorizationSolver: ...
    @staticmethod
    def from_distance_matrix(
        distances: _MatrixArrayLike,
        *,
        epsilon: float = 1e-4,
        max_iterations: int = 100,
        pcg_tolerance: float = 1e-6,
        pcg_max_iterations: int | None = None,
        min_norm: float = 1e-4,
        dtype: _DtypeLike = "float64",
    ) -> StressMajorizationSolver: ...
    def step(self, layout: Layout2d) -> str: ...
    def run_steps(self, layout: Layout2d, steps: int) -> str: ...
    def run(self, layout: Layout2d) -> str: ...
    def compute_stress(self, layout: Layout2d) -> float: ...
    @property
    def last_stress(self) -> float | None: ...
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
class KamadaKawaiSolver:
    @staticmethod
    def from_graph(
        graph: CsrGraph,
        *,
        eps: float = 1e-2,
        max_inner_iters: int = 50,
        max_outer_iters: int | None = None,
        dtype: _DtypeLike = "float64",
    ) -> KamadaKawaiSolver: ...
    @staticmethod
    def from_distance_matrix(
        distances: _MatrixArrayLike,
        *,
        eps: float = 1e-2,
        max_inner_iters: int = 50,
        max_outer_iters: int | None = None,
        dtype: _DtypeLike = "float64",
    ) -> KamadaKawaiSolver: ...
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
    def num_nodes(self) -> int: ...
    @property
    def dtype(self) -> str: ...
    def __repr__(self) -> str: ...

def kmeans_plus_plus(
    points: _MatrixArrayLike,
    k: int = 2,
    max_iters: int = 100,
    tol: float = 1e-4,
    n_init: int = 1,
    seed: int = 42,
) -> dict[str, Any]: ...
def spectral_clustering(
    graph: CsrGraph,
    k: int = 2,
    max_eigen_iters: int = 300,
    eigen_tol: float = 1e-4,
    kmeans_max_iters: int = 100,
    kmeans_tol: float = 1e-4,
    seed: int = 42,
) -> npt.NDArray[np.uint32]: ...
def louvain_communities(
    graph: CsrGraph,
    max_passes: int = 100,
    resolution: float = 1.0,
    min_modularity_gain: float = 1e-7,
    seed: int = 42,
) -> dict[str, Any]: ...
def label_propagation(
    graph: CsrGraph,
    max_iters: int = 100,
    seed: int = 42,
) -> npt.NDArray[np.uint32]: ...
def coarsen_graph(
    graph: CsrGraph,
    labels: _IndexArrayLike,
) -> tuple[CsrGraph, npt.NDArray[np.uint32], npt.NDArray[np.uint32], int]: ...
def project_clustered_layout(
    coarse_layout: Layout2d,
    node_to_coarse: _IndexArrayLike,
    spread_radius: float = 0.1,
    seed: int = 42,
) -> Layout2d: ...
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
def project_clustered_layout_with_radii(
    coarse_layout: Layout2d,
    node_to_coarse: _IndexArrayLike,
    cluster_radii: _WeightArrayLike,
    *,
    jitter_scale: float = 0.5,
    seed: int = 42,
) -> Layout2d: ...
def align_clustered_centroids(
    fine_layout: Layout2d,
    coarse_layout: Layout2d,
    node_to_coarse: _IndexArrayLike,
    *,
    gamma_tether: float = 0.5,
) -> None: ...
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
def py_coarsen(
    graph: GraphAdapter,
    node_group_func: Callable[[int], int],
    node_merge_func: Callable[[list[int]], Any],
    edge_merge_func: Callable[[list[int]], Any],
) -> tuple[GraphAdapter, dict[int, int]]: ...

@final
class TFdpSolver:
    def __new__(
        cls,
        graph: CsrGraph,
        *,
        backend: str = "ibfft",
        total_iterations: int = 200,
        learning_rate: float = 1.0,
        momentum: float = 0.8,
        theta: float = 0.5,
        alpha: float = 1.0,
        gamma: float = 1.0,
        rho_start: float = 1.0,
        rho_end: float = 0.1,
        rho_power: float = 1.0,
        ibfft_order: int = 1,
        ibfft_intervals: int = 25,
        ibfft_min_grid_dim: int = 64,
        ibfft_max_grid_dim: int = 512,
        seed: int = 42,
        dtype: _DtypeLike = "float64",
    ) -> TFdpSolver: ...
    @overload
    @staticmethod
    def from_knn(
        knn_or_indptr: KnnGraph,
        indices: None = None,
        weights: None = None,
        *,
        backend: str = "ibfft",
        total_iterations: int = 200,
        learning_rate: float = 1.0,
        momentum: float = 0.8,
        theta: float = 0.5,
        alpha: float = 1.0,
        gamma: float = 1.0,
        rho_start: float = 1.0,
        rho_end: float = 0.1,
        rho_power: float = 1.0,
        ibfft_order: int = 1,
        ibfft_intervals: int = 25,
        ibfft_min_grid_dim: int = 64,
        ibfft_max_grid_dim: int = 512,
        seed: int = 42,
        dtype: _DtypeLike = "float64",
    ) -> TFdpSolver: ...
    @overload
    @staticmethod
    def from_knn(
        knn_or_indptr: _IndexArrayLike,
        indices: _IndexArrayLike,
        weights: _WeightArrayLike | None = None,
        *,
        backend: str = "ibfft",
        total_iterations: int = 200,
        learning_rate: float = 1.0,
        momentum: float = 0.8,
        theta: float = 0.5,
        alpha: float = 1.0,
        gamma: float = 1.0,
        rho_start: float = 1.0,
        rho_end: float = 0.1,
        rho_power: float = 1.0,
        ibfft_order: int = 1,
        ibfft_intervals: int = 25,
        ibfft_min_grid_dim: int = 64,
        ibfft_max_grid_dim: int = 512,
        seed: int = 42,
        dtype: _DtypeLike = "float64",
    ) -> TFdpSolver: ...
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
    def num_nodes(self) -> int: ...
    @property
    def dtype(self) -> str: ...
    @property
    def current_rho(self) -> float: ...
    @property
    def num_edges(self) -> int: ...
    @property
    def total_fft_2d_calls(self) -> int: ...
    def __repr__(self) -> str: ...

@final
class IbUmapSolver:
    def __new__(
        cls,
        graph: CsrGraph,
        *,
        max_2hop_per_node: int = 30,
        degree: int | None = None,
        num_landmarks: int = 64,
        min_degree: int = 2,
        jl_dim: int = 64,
        degree_normalize: bool = True,
        p_exp: float = 0.5,
        eps_psd: float = 1e-4,
        t_coarse: float = 48.0,
        t1: float | None = None,
        t2: float | None = None,
        alpha: float | None = None,
        beta_floor: float | None = None,
        backend: str = "ibfft",
        n_epochs: int = 200,
        learning_rate: float = 1.0,
        min_dist: float = 0.1,
        spread: float = 1.0,
        a: float | None = None,
        b: float | None = None,
        c_k: float = 4.0,
        tau_scale: float = 4.0,
        hub_damping: bool = True,
        theta: float = 0.5,
        ibfft_order: int = 1,
        ibfft_intervals: int = 25,
        ibfft_min_grid_dim: int = 64,
        ibfft_max_grid_dim: int = 512,
        seed: int = 42,
        dtype: _DtypeLike = "float64",
    ) -> IbUmapSolver: ...
    @staticmethod
    def from_landmarks(
        graph: CsrGraph,
        landmarks: DogVaradhanLandmarks,
        max_2hop_per_node: int = 32,
        *,
        backend: str = "ibfft",
        n_epochs: int = 200,
        learning_rate: float = 1.0,
        min_dist: float = 0.1,
        spread: float = 1.0,
        a: float | None = None,
        b: float | None = None,
        c_k: float = 4.0,
        tau_scale: float = 4.0,
        hub_damping: bool = True,
        theta: float = 0.5,
        ibfft_order: int = 1,
        ibfft_intervals: int = 25,
        ibfft_min_grid_dim: int = 64,
        ibfft_max_grid_dim: int = 512,
        seed: int = 42,
        dtype: _DtypeLike = "float64",
    ) -> IbUmapSolver: ...
    @staticmethod
    def from_knn(
        knn: KnnGraph,
        *,
        sigma_iters: int = 64,
        sigma_tol: float = 1e-5,
        backend: str = "ibfft",
        n_epochs: int = 200,
        learning_rate: float = 1.0,
        min_dist: float = 0.1,
        spread: float = 1.0,
        a: float | None = None,
        b: float | None = None,
        c_k: float = 4.0,
        tau_scale: float = 4.0,
        hub_damping: bool = True,
        theta: float = 0.5,
        ibfft_order: int = 1,
        ibfft_intervals: int = 25,
        ibfft_min_grid_dim: int = 64,
        ibfft_max_grid_dim: int = 512,
        seed: int = 42,
        dtype: _DtypeLike = "float64",
    ) -> IbUmapSolver: ...
    @staticmethod
    def from_fuzzy_set(
        probs: SparseProbabilities,
        *,
        degrees: _IndexArrayLike | None = None,
        backend: str = "ibfft",
        n_epochs: int = 200,
        learning_rate: float = 1.0,
        min_dist: float = 0.1,
        spread: float = 1.0,
        a: float | None = None,
        b: float | None = None,
        c_k: float = 4.0,
        tau_scale: float = 4.0,
        hub_damping: bool = True,
        theta: float = 0.5,
        ibfft_order: int = 1,
        ibfft_intervals: int = 25,
        ibfft_min_grid_dim: int = 64,
        ibfft_max_grid_dim: int = 512,
        seed: int = 42,
        dtype: _DtypeLike = "float64",
    ) -> IbUmapSolver: ...
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
    def num_nodes(self) -> int: ...
    @property
    def dtype(self) -> str: ...
    @property
    def current_rho(self) -> float: ...
    @property
    def num_edges(self) -> int: ...
    @property
    def total_fft_2d_calls(self) -> int: ...
    @property
    def a(self) -> float | None: ...
    @property
    def b(self) -> float | None: ...
    @property
    def ab(self) -> tuple[float, float] | None: ...
    def __repr__(self) -> str: ...

@final
class ForceAtlas2Solver:
    def __new__(
        cls,
        graph: CsrGraph,
        *,
        iterations: int = 200,
        theta: float = 1.2,
        use_barnes_hut: bool = True,
        linlog_mode: bool = False,
        strong_gravity: bool = False,
        gravity: float = 1.0,
        scaling_ratio: float = 2.0,
        edge_weight_influence: float = 1.0,
        outbound_attraction_distribution: bool = False,
        jitter_tolerance: float = 1.0,
        max_displacement: float | None = None,
        seed: int = 42,
        dtype: _DtypeLike = "float64",
    ) -> ForceAtlas2Solver: ...
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
    def num_nodes(self) -> int: ...
    @property
    def dtype(self) -> str: ...
    def __repr__(self) -> str: ...

@final
class YifanHuSolver:
    def __new__(
        cls,
        graph: CsrGraph,
        *,
        max_iterations_per_level: int = 200,
        step_size: float = 1.0,
        step_ratio: float = 0.9,
        relative_strength: float = 0.2,
        optimal_distance: float | None = None,
        theta: float = 1.2,
        min_coarsen_size: int = 30,
        min_coarsen_ratio: float = 0.75,
        convergence_tolerance: float = 1e-4,
        seed: int = 42,
        dtype: _DtypeLike = "float64",
    ) -> YifanHuSolver: ...
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
    def num_nodes(self) -> int: ...
    @property
    def num_levels(self) -> int: ...
    @property
    def current_level(self) -> int: ...
    @property
    def dtype(self) -> str: ...
    def __repr__(self) -> str: ...

@final
class PrismOverlapRemoval:
    def __new__(
        cls,
        half_widths: _WeightArrayLike,
        half_heights: _WeightArrayLike,
        *,
        max_iterations: int = 50,
        max_scale_factor: float = 1.5,
        cg_max_iters: int = 80,
        cg_tolerance: float = 1e-3,
        padding: float = 0.0,
        seed: int = 42,
        dtype: _DtypeLike = "float64",
    ) -> PrismOverlapRemoval: ...
    @staticmethod
    def from_radii(
        radii: _WeightArrayLike,
        *,
        max_iterations: int = 50,
        max_scale_factor: float = 1.5,
        cg_max_iters: int = 80,
        cg_tolerance: float = 1e-3,
        padding: float = 0.0,
        seed: int = 42,
        dtype: _DtypeLike = "float64",
    ) -> PrismOverlapRemoval: ...
    def step(self, layout: Layout2d) -> str: ...
    def run_steps(self, layout: Layout2d, steps: int) -> str: ...
    def run(self, layout: Layout2d) -> str: ...
    @property
    def last_overlap_count(self) -> int: ...
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
class FdebSolver:
    def __new__(
        cls,
        graph: CsrGraph,
        layout: Layout2d,
        *,
        cycles: int = 6,
        s0: float = 0.1,
        i0: int = 90,
        s_step: float = 0.5,
        i_step: float = ...,
        spring_k: float = 0.1,
        minimum_edge_compatibility: float = 0.6,
        dtype: _DtypeLike | None = None,
    ) -> FdebSolver: ...
    def step(self) -> str: ...
    def run_steps(self, steps: int) -> str: ...
    def run(self) -> str: ...
    @property
    def status(self) -> str: ...
    @property
    def iteration(self) -> int: ...
    @property
    def total_iterations(self) -> int: ...
    @property
    def is_finished(self) -> bool: ...
    @property
    def num_edges(self) -> int: ...
    @property
    def dtype(self) -> str: ...
    def edge_offsets(self) -> npt.NDArray[np.uint32]: ...
    def points(self) -> npt.NDArray[Any]: ...
    def __repr__(self) -> str: ...

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
def project_clustered_rectangle_no_overlap_2d(
    layout: Layout2d,
    cluster_ids: _IndexArrayLike,
    half_widths: _WeightArrayLike,
    half_heights: _WeightArrayLike,
) -> None: ...
def remove_cycles(graph: CsrGraph) -> CsrGraph: ...
def assign_layers(
    graph: CsrGraph, *, method: str = "longest_path"
) -> npt.NDArray[np.uint32]: ...
def classical_mds_2d(
    graph: CsrGraph, *, dtype: _DtypeLike = "float64"
) -> Layout2d: ...
def rd_mds(
    graph: CsrGraph,
    dims: int = 2,
    *,
    eig_eps: float = 1e-10,
    tol: float = 1e-6,
    max_iters: int = 1000,
    seed: int = 42,
    dtype: _DtypeLike = "float64",
) -> LayoutNd: ...
def normalize_layout_2d(
    layout: Layout2d, *, target_std: float = 1.0, seed: int = 42
) -> None: ...
def robust_normalize_layout_2d(
    layout: Layout2d,
    *,
    target_std: float = 1.0,
    quantile: float = 0.95,
    seed: int = 42,
) -> None: ...
def representative_mds(
    distances_to_reps: _MatrixArrayLike,
    dims: int = 2,
    *,
    dtype: _DtypeLike = "float64",
) -> LayoutNd: ...
def compute_resistance_embedding(
    graph: CsrGraph,
    dim: int = 16,
    *,
    tol: float = 1e-4,
    max_iters: int = 300,
    seed: int = 42,
    eig_eps: float = 1e-10,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]: ...
def compute_diffusion_embedding(
    graph: CsrGraph,
    dim: int = 16,
    *,
    p: float = 0.5,
    t: float | None = None,
    tol: float = 1e-4,
    max_iters: int = 300,
    seed: int = 42,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], float]: ...
def build_rerank_knn(
    coords: _CoordArrayLike,
    k: int = 15,
    *,
    coarse_dim: int = 10,
    rerank_factor: int = 16,
) -> KnnGraph: ...
def graph_row_normalized_probabilities(
    graph: CsrGraph,
) -> SparseProbabilities: ...
def graph_12hop_row_normalized_probabilities(
    graph: CsrGraph,
    w2: float = 0.5,
    max_2hop_per_node: int | None = None,
    *,
    degree_beta: float = 0.0,
    seed: int = 42,
) -> SparseProbabilities: ...
def symmetrize_row_normalized(
    num_nodes: int,
    indptr: _IndexArrayLike,
    indices: _IndexArrayLike,
    weights: _WeightArrayLike,
) -> SparseProbabilities: ...
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

@final
class ScreenedPoissonField2d:
    def __new__(
        cls,
        *,
        kernel: str = "screened_poisson",
        grid_size: int = 2048,
        padding_factor: int = 2,
        margin_ratio: float = 0.1,
        sigma_ratio: float = 0.015,
        sigma: float | None = None,
        screening_length_ratio: float = 0.2,
        kappa: float | None = None,
        rho_max_factor: float = 4.0,
        rho_max: float | None = None,
        exact_adjoint: bool = True,
        dtype: _DtypeLike = "float64",
    ) -> ScreenedPoissonField2d: ...
    def evaluate(self, layout: Layout2d) -> dict[str, Any]: ...
    def evaluate_in_bounds(
        self, layout: Layout2d, bounds: _WeightArrayLike
    ) -> dict[str, Any]: ...
    def workspace_memory_bytes(self) -> int: ...
    @property
    def grid_size(self) -> int: ...
    @property
    def padding_factor(self) -> int: ...
    @property
    def kernel(self) -> str: ...
    @property
    def exact_adjoint(self) -> bool: ...
    @property
    def dtype(self) -> str: ...
    def __repr__(self) -> str: ...

@final
class ScreenedPoissonRefiner:
    def __new__(
        cls,
        num_nodes: int,
        *,
        kernel: str = "screened_poisson",
        grid_size: int = 2048,
        padding_factor: int = 2,
        margin_ratio: float = 0.1,
        sigma_ratio: float = 0.015,
        sigma: float | None = None,
        screening_length_ratio: float = 0.2,
        kappa: float | None = None,
        rho_max_factor: float = 4.0,
        rho_max: float | None = None,
        exact_adjoint: bool = True,
        iterations: int = 20,
        step_size: float = 0.25,
        momentum: float = 0.5,
        max_step_ratio: float = 0.05,
        anchor_weight: float = 0.0,
        gamma_tether: float = 0.0,
        recenter: bool = True,
        energy_tol: float = 1e-12,
        seed: int = 42,
        dtype: _DtypeLike = "float64",
    ) -> ScreenedPoissonRefiner: ...
    @staticmethod
    def from_graph(
        graph: CsrGraph,
        *,
        kernel: str = "screened_poisson",
        grid_size: int = 2048,
        padding_factor: int = 2,
        margin_ratio: float = 0.1,
        sigma_ratio: float = 0.015,
        sigma: float | None = None,
        screening_length_ratio: float = 0.2,
        kappa: float | None = None,
        rho_max_factor: float = 4.0,
        rho_max: float | None = None,
        exact_adjoint: bool = True,
        iterations: int = 20,
        step_size: float = 0.25,
        momentum: float = 0.5,
        max_step_ratio: float = 0.05,
        anchor_weight: float = 0.0,
        gamma_tether: float = 0.0,
        recenter: bool = True,
        energy_tol: float = 1e-12,
        seed: int = 42,
        dtype: _DtypeLike = "float64",
    ) -> ScreenedPoissonRefiner: ...
    def set_coarse_tether(
        self, coarse_layout: Layout2d, node_to_coarse: _IndexArrayLike
    ) -> None: ...
    def step(self, layout: Layout2d) -> str: ...
    def run_steps(self, layout: Layout2d, steps: int) -> str: ...
    def run(self, layout: Layout2d) -> str: ...
    def apply_epoch_step(self, layout: Layout2d) -> dict[str, Any]: ...
    def last_stats(self) -> dict[str, Any]: ...
    def workspace_memory_bytes(self) -> int: ...
    @property
    def last_energy(self) -> float: ...
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
