from __future__ import annotations

from typing import final

import numpy as np
import numpy.typing as npt

from ._core_core import CsrGraph, Layout2d, Layout3d, LayoutNd
from ._core_types import (
    EuclideanTargetKind,
    _DtypeLike,
    _IndexArrayLike,
    _MatrixArrayLike,
    _WeightArrayLike,
)

__all__ = [
    "ClassicalMds",
    "CsrMatrix",
    "Hde",
    "KorenSpectral",
    "PivotMds",
    "RdMds",
    "SpectralMomentStats",
    "anisotropic_laplacian",
    "classical_mds",
    "classical_mds_2d",
    "classical_mds_from_distances",
    "combinatorial_laplacian",
    "compute_diffusion_embedding",
    "compute_resistance_embedding",
    "hde",
    "hde_nd",
    "koren_spectral_init",
    "laplacian",
    "normalize_layout_2d",
    "pivot_mds",
    "pivot_mds_from_distances",
    "pivot_mds_init",
    "random_walk_laplacian",
    "rd_mds",
    "representative_mds",
    "robust_normalize_layout_2d",
    "select_maxmin_sp_pivots",
    "shifted_normalized_laplacian",
    "smallest_laplacian_eigenpairs",
    "spectral_moment_stats",
    "symmetric_eigen_jacobi",
    "symmetric_normalized_laplacian",
]

@final
class SpectralMomentStats:
    @property
    def sigma_lambda_sq(self) -> float: ...
    @property
    def sigma_lambda(self) -> float: ...
    @property
    def deg_cv(self) -> float: ...
    @property
    def mean_degree(self) -> float: ...
    @property
    def leaf_fraction(self) -> float: ...
    def to_dict(self) -> dict[str, float]: ...
    def keys(self) -> list[str]: ...
    def __getitem__(self, key: str, /) -> float: ...
    def __contains__(self, key: object, /) -> bool: ...
    def __repr__(self) -> str: ...

@final
class ClassicalMds:
    def __new__(
        cls,
        graph: CsrGraph,
        *,
        target: EuclideanTargetKind = "layout2d",
        dims: int | None = None,
        eig_eps: float = 1e-12,
        tol: float = 1e-10,
        max_sweeps: int = 64,
        dtype: _DtypeLike = "float64",
    ) -> ClassicalMds: ...
    @staticmethod
    def from_distances(
        distances: _MatrixArrayLike,
        *,
        target: EuclideanTargetKind = "layout2d",
        dims: int | None = None,
        eig_eps: float = 1e-12,
        tol: float = 1e-10,
        max_sweeps: int = 64,
        dtype: _DtypeLike = "float64",
    ) -> ClassicalMds: ...
    @property
    def target(self) -> EuclideanTargetKind: ...
    @property
    def dtype(self) -> str: ...
    def generate(self) -> Layout2d | Layout3d | LayoutNd: ...
    def __repr__(self) -> str: ...

@final
class PivotMds:
    def __new__(
        cls,
        graph: CsrGraph,
        *,
        target: EuclideanTargetKind = "layout2d",
        dims: int | None = None,
        num_pivots: int = 50,
        seed: int = 42,
        min_degree: int = 2,
        target_std: float | None = 1.0,
        pivots: _IndexArrayLike | None = None,
        dtype: _DtypeLike = "float64",
    ) -> PivotMds: ...
    @staticmethod
    def from_distances(
        distances_nk: _MatrixArrayLike,
        *,
        target: EuclideanTargetKind = "layout2d",
        dims: int | None = None,
        target_std: float | None = 1.0,
        seed: int = 42,
        dtype: _DtypeLike = "float64",
    ) -> PivotMds: ...
    @property
    def target(self) -> EuclideanTargetKind: ...
    @property
    def dtype(self) -> str: ...
    def generate(self) -> Layout2d | Layout3d | LayoutNd: ...
    def __repr__(self) -> str: ...

@final
class Hde:
    def __new__(
        cls,
        graph: CsrGraph,
        *,
        target: EuclideanTargetKind = "layout2d",
        dimensions: int = 50,
        output_dims: int | None = None,
        seed: int = 42,
        tol: float = 1e-3,
        max_sweeps: int = 100,
        normalize: bool = True,
        pivots: _IndexArrayLike | None = None,
        dtype: _DtypeLike = "float64",
    ) -> Hde: ...
    @property
    def target(self) -> EuclideanTargetKind: ...
    @property
    def dtype(self) -> str: ...
    def generate(self) -> Layout2d | Layout3d | LayoutNd: ...
    def __repr__(self) -> str: ...

@final
class RdMds:
    def __new__(
        cls,
        graph: CsrGraph,
        *,
        target: EuclideanTargetKind = "layout2d",
        dims: int | None = None,
        eig_eps: float = 1e-10,
        tol: float = 1e-6,
        max_iters: int = 1000,
        seed: int = 42,
        degree_normalize: bool = True,
        target_std: float | None = None,
        dtype: _DtypeLike = "float64",
    ) -> RdMds: ...
    @property
    def target(self) -> EuclideanTargetKind: ...
    @property
    def dtype(self) -> str: ...
    def generate(self) -> Layout2d | Layout3d | LayoutNd: ...
    def __repr__(self) -> str: ...

@final
class KorenSpectral:
    def __new__(
        cls,
        graph: CsrGraph,
        *,
        target: EuclideanTargetKind = "layout2d",
        dims: int | None = None,
        target_std: float | None = 1.0,
        tol: float = 1e-4,
        max_iters: int = 300,
        seed: int = 42,
        dtype: _DtypeLike = "float64",
    ) -> KorenSpectral: ...
    @property
    def target(self) -> EuclideanTargetKind: ...
    @property
    def dtype(self) -> str: ...
    def generate(self) -> Layout2d | Layout3d | LayoutNd: ...
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

def anisotropic_laplacian(
    graph: CsrGraph,
    *,
    alpha: float = 0.5,
    beta: float = 0.0,
    degree_reg: float = 0.0,
    degree_cap: float | None = None,
    reweighted_degree_norm: bool = True,
) -> tuple[CsrMatrix, npt.NDArray[np.float64]]: ...

def laplacian(
    graph: CsrGraph, kind: str = "symmetric_normalized"
) -> CsrMatrix: ...

def combinatorial_laplacian(graph: CsrGraph) -> CsrMatrix: ...

def symmetric_normalized_laplacian(
    graph: CsrGraph,
) -> tuple[CsrMatrix, npt.NDArray[np.float64]]: ...

def random_walk_laplacian(graph: CsrGraph) -> CsrMatrix: ...

def shifted_normalized_laplacian(
    l_sym_or_graph: CsrMatrix | CsrGraph,
    lambda_max: float = 2.0,
) -> CsrMatrix: ...

def classical_mds(
    graph: CsrGraph,
    dims: int = 2,
    *,
    dtype: _DtypeLike = "float64",
) -> LayoutNd: ...

def classical_mds_2d(
    graph: CsrGraph, *, dtype: _DtypeLike = "float64"
) -> Layout2d: ...

def classical_mds_from_distances(
    distances: _MatrixArrayLike,
    dims: int = 2,
    *,
    dtype: _DtypeLike = "float64",
) -> LayoutNd: ...

def compute_diffusion_embedding(
    graph: CsrGraph,
    dim: int = 16,
    *,
    p: float = 0.5,
    t: float | None = None,
    tol: float = 1e-4,
    max_iterations: int = 300,
    seed: int = 42,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], float]: ...

def compute_resistance_embedding(
    graph: CsrGraph,
    dim: int = 16,
    *,
    tol: float = 1e-4,
    max_iterations: int = 300,
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
    max_iterations: int = 300,
    seed: int = 42,
    dtype: _DtypeLike = "float64",
) -> Layout2d: ...

def normalize_layout_2d(
    layout: Layout2d, *, target_std: float = 1.0, seed: int = 42
) -> None: ...

def pivot_mds(
    graph: CsrGraph,
    *,
    num_pivots: int = 50,
    seed: int = 42,
    min_degree: int = 2,
    target_std: float | None = 1.0,
    pivots: _IndexArrayLike | None = None,
    dtype: _DtypeLike = "float64",
) -> Layout2d: ...

def pivot_mds_init(
    graph: CsrGraph,
    *,
    num_pivots: int = 50,
    seed: int = 42,
    min_degree: int = 2,
    target_std: float | None = 1.0,
    dtype: _DtypeLike = "float64",
) -> Layout2d: ...

def pivot_mds_from_distances(
    distances_nk: _MatrixArrayLike,
    *,
    target_std: float | None = 1.0,
    seed: int = 42,
    dtype: _DtypeLike = "float64",
) -> Layout2d: ...

def rd_mds(
    graph: CsrGraph,
    dims: int = 2,
    *,
    eig_eps: float = 1e-10,
    tol: float = 1e-6,
    max_iterations: int = 1000,
    seed: int = 42,
    degree_normalize: bool = True,
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
    max_iterations: int = 300,
    seed: int = 42,
) -> tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    bool,
]: ...

def spectral_moment_stats(graph: CsrGraph) -> SpectralMomentStats: ...

def symmetric_eigen_jacobi(
    matrix: _MatrixArrayLike,
    *,
    max_sweeps: int = 120,
    tol: float = 1e-12,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]: ...
