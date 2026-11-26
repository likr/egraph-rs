from __future__ import annotations

from typing import final

import numpy as np
import numpy.typing as npt

from ._core_core import CsrGraph, Layout2d, LayoutNd
from ._core_types import (
    _DtypeLike,
    _IndexArrayLike,
    _MatrixArrayLike,
    _VectorArrayLike,
    _WeightArrayLike,
)

__all__ = [
    "CsrMatrix",
    "anisotropic_laplacian",
    "chebyshev_dog_jl_embedding",
    "chebyshev_dog_jl_embedding_matrix",
    "chebyshev_heat_spmv",
    "chebyshev_heat_spmv_matrix",
    "classical_mds_2d",
    "compute_diffusion_embedding",
    "compute_resistance_embedding",
    "hde",
    "hde_nd",
    "koren_spectral_init",
    "normalize_layout_2d",
    "pivot_mds_init",
    "rd_mds",
    "representative_mds",
    "robust_normalize_layout_2d",
    "select_maxmin_sp_pivots",
    "smallest_laplacian_eigenpairs",
    "spectral_moment_stats",
]

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

def anisotropic_laplacian(
    graph: CsrGraph,
    *,
    alpha: float = 0.5,
    beta: float = 0.0,
    degree_reg: float = 0.0,
    degree_cap: float | None = None,
    reweighted_degree_norm: bool = True,
) -> tuple[CsrMatrix, npt.NDArray[np.float64]]: ...

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

def classical_mds_2d(
    graph: CsrGraph, *, dtype: _DtypeLike = "float64"
) -> Layout2d: ...

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

def compute_resistance_embedding(
    graph: CsrGraph,
    dim: int = 16,
    *,
    tol: float = 1e-4,
    max_iters: int = 300,
    seed: int = 42,
    eig_eps: float = 1e-10,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]: ...

def hde(
    graph: CsrGraph,
    *,
    dimensions: int = 50,
    seed: int = 42,
    tol: float = 1e-3,
    max_sweeps: int = 100,
    normalize: bool = True,
    pivots: _IndexArrayLike | None = None,
    dtype: _DtypeLike = "float64",
) -> Layout2d: ...

def hde_nd(
    graph: CsrGraph,
    *,
    dimensions: int = 50,
    output_dims: int = 2,
    seed: int = 42,
    tol: float = 1e-3,
    max_sweeps: int = 100,
    normalize: bool = True,
    pivots: _IndexArrayLike | None = None,
    dtype: _DtypeLike = "float64",
) -> LayoutNd: ...

def koren_spectral_init(
    graph: CsrGraph,
    *,
    target_std: float | None = 1.0,
    tol: float = 1e-4,
    max_iters: int = 300,
    seed: int = 42,
    dtype: _DtypeLike = "float64",
) -> Layout2d: ...

def normalize_layout_2d(
    layout: Layout2d, *, target_std: float = 1.0, seed: int = 42
) -> None: ...

def pivot_mds_init(
    graph: CsrGraph,
    *,
    num_pivots: int = 50,
    seed: int = 42,
    min_degree: int = 2,
    target_std: float | None = 1.0,
    dtype: _DtypeLike = "float64",
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

def representative_mds(
    distances_to_reps: _MatrixArrayLike,
    dims: int = 2,
    *,
    dtype: _DtypeLike = "float64",
) -> LayoutNd: ...

def robust_normalize_layout_2d(
    layout: Layout2d,
    *,
    target_std: float = 1.0,
    quantile: float = 0.95,
    seed: int = 42,
) -> None: ...

def select_maxmin_sp_pivots(
    graph: CsrGraph,
    num_pivots: int = 50,
    *,
    seed: int = 42,
    min_degree: int = 2,
) -> npt.NDArray[np.uint32]: ...

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
