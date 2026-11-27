from __future__ import annotations

from typing import Any, final

import numpy as np
import numpy.typing as npt

from ._core_core import CsrGraph
from ._core_linalg import CsrMatrix
from ._core_types import _IndexArrayLike, _MatrixArrayLike, _WeightArrayLike

__all__ = [
    "AmgPcgSolver",
    "EffectiveResistanceResult",
    "ResistanceCurvatureFlowResult",
    "compute_effective_resistances",
    "resistance_curvature_flow",
]

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
    jl_dim: int = 128,
    seed: int = 42,
    weights_are_lengths: bool = True,
    exact: bool | None = None,
    exact_max_nodes: int = 256,
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
    jl_dim: int = 128,
    seed: int = 42,
    weights_are_lengths: bool = True,
    exact: bool | None = None,
    exact_max_nodes: int = 256,
    eig_eps: float = 1e-12,
    rel_tol: float = 1e-8,
    max_iterations: int = 500,
    hyperbolic_threshold: float = -0.05,
    spherical_threshold: float = 0.2,
    torus_std_threshold: float = 0.05,
) -> ResistanceCurvatureFlowResult: ...
