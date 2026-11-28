from __future__ import annotations

from typing import Sequence, final

import numpy as np
import numpy.typing as npt

from ._core_core import CsrGraph
from ._core_linalg import CsrMatrix, SpectralMomentStats
from ._core_types import _IndexArrayLike, _VectorArrayLike, _WeightArrayLike

__all__ = [
    "ChebyshevDogKernel",
    "DijkstraWorkspace",
    "SpectralFilterProjection",
    "adaptive_chebyshev_degree",
    "adaptive_chebyshev_degree_scaled",
    "all_pairs_dijkstra",
    "build_dog_terms",
    "build_regularized_dog_terms",
    "chebyshev_dog_jl_embedding",
    "chebyshev_dog_jl_embedding_matrix",
    "chebyshev_heat_spmv",
    "chebyshev_heat_spmv_matrix",
    "compute_chebyshev_bandpass_coefficients",
    "dijkstra_single_source",
    "project_psd_spectral_filter",
]

@final
class SpectralFilterProjection:
    @property
    def terms(self) -> list[tuple[float, float]]: ...
    @property
    def delta_psd(self) -> float: ...
    @property
    def h_min(self) -> float: ...
    def to_dict(self) -> dict[str, object]: ...
    def keys(self) -> list[str]: ...
    def __getitem__(self, key: str, /) -> object: ...
    def __contains__(self, key: object, /) -> bool: ...
    def __repr__(self) -> str: ...

@final
class ChebyshevDogKernel:
    def __new__(
        cls,
        graph: CsrGraph,
        *,
        t1: float | None = None,
        t2: float | None = None,
        alpha: float | None = None,
        beta_floor: float | None = None,
        degree: int | None = None,
        n_landmarks: int = 64,
        min_degree: int = 2,
        jl_dim: int = 64,
        eps_psd: float = 1e-4,
        lambda_max: float = 2.0,
        seed: int = 42,
    ) -> ChebyshevDogKernel: ...
    @staticmethod
    def build_with_landmarks(
        graph: CsrGraph,
        landmarks: _IndexArrayLike,
        *,
        t1: float | None = None,
        t2: float | None = None,
        alpha: float | None = None,
        beta_floor: float | None = None,
        degree: int | None = None,
        jl_dim: int = 64,
        eps_psd: float = 1e-4,
        lambda_max: float = 2.0,
        seed: int = 42,
    ) -> ChebyshevDogKernel: ...
    @property
    def num_nodes(self) -> int: ...
    @property
    def num_landmarks(self) -> int: ...
    @property
    def jl_dim(self) -> int: ...
    @property
    def t1(self) -> float: ...
    @property
    def t2(self) -> float: ...
    @property
    def alpha(self) -> float: ...
    @property
    def degree(self) -> int: ...
    @property
    def delta_psd(self) -> float: ...
    @property
    def h_min(self) -> float: ...
    @property
    def beta_auto(self) -> float: ...
    @property
    def beta_eff(self) -> float: ...
    def landmarks(self) -> npt.NDArray[np.uint32]: ...
    def landmark_kernel_matrix(self) -> npt.NDArray[np.float64]: ...
    def diag_estimate(self) -> npt.NDArray[np.float64]: ...
    def jl_embedding(self) -> npt.NDArray[np.float64]: ...
    def moment_stats(self) -> SpectralMomentStats: ...
    def terms_psd(self) -> list[tuple[float, float]]: ...
    def varadhan_landmark_table(
        self,
        *,
        p_exp: float = 0.5,
        degree_normalize: bool = True,
        eps_sim: float = 1e-6,
    ) -> npt.NDArray[np.float64]: ...
    def apply_batch(
        self, l_sym: CsrMatrix, v: _VectorArrayLike
    ) -> npt.NDArray[np.float64]: ...
    def __repr__(self) -> str: ...

@final
class DijkstraWorkspace:
    def __new__(cls, num_nodes: int = 0) -> DijkstraWorkspace: ...
    def ensure_capacity(self, num_nodes: int) -> None: ...
    def dijkstra_from(
        self, graph: CsrGraph, source: int
    ) -> npt.NDArray[np.float64]: ...
    def __repr__(self) -> str: ...

def dijkstra_single_source(
    graph: CsrGraph, source: int
) -> npt.NDArray[np.float64]: ...

def all_pairs_dijkstra(graph: CsrGraph) -> npt.NDArray[np.float64]: ...

def build_dog_terms(
    t1: float, t2: float, alpha: float, *, beta_floor: float = 0.0
) -> list[tuple[float, float]]: ...

def build_regularized_dog_terms(
    sigma_lambda: float, *, beta_floor: float = 0.0
) -> list[tuple[float, float]]: ...

def adaptive_chebyshev_degree(max_t: float) -> int: ...

def adaptive_chebyshev_degree_scaled(
    max_t: float, *, lambda_max: float = 2.0
) -> int: ...

def compute_chebyshev_bandpass_coefficients(
    terms: Sequence[tuple[float, float]],
    *,
    lambda_max: float = 2.0,
    degree: int = 40,
) -> npt.NDArray[np.float64]: ...

def project_psd_spectral_filter(
    terms: Sequence[tuple[float, float]],
    *,
    lambda_max: float = 2.0,
    degree: int = 40,
    eps_psd: float = 1e-4,
) -> SpectralFilterProjection: ...

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
