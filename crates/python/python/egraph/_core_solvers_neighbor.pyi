from __future__ import annotations

from typing import Sequence, final

from ._core_core import CsrGraph, Layout2d, Layout3d, LayoutNd
from ._core_manifold import ManifoldLayout2d
from ._core_neighborhood import DogVaradhanLandmarks, KnnGraph, SparseProbabilities
from ._core_types import (
    BackendKind,
    SolverStatus,
    TargetKind,
    _DtypeLike,
    _WeightArrayLike,
)

_AnyLayout = Layout2d | Layout3d | LayoutNd | ManifoldLayout2d

__all__ = [
    "FieldFftStats",
    "FitFieldStats",
    "GuMapSolver",
    "TsNetSolver",
    "TsNetStage",
]

@final
class FitFieldStats:
    @property
    def particle_forward_ffts(self) -> int: ...
    @property
    def kernel_forward_ffts(self) -> int: ...
    @property
    def inverse_ffts(self) -> int: ...
    @property
    def total_fft_2d_calls(self) -> int: ...
    def to_dict(self) -> dict[str, int]: ...
    def keys(self) -> list[str]: ...
    def __getitem__(self, key: str, /) -> int: ...
    def __contains__(self, key: object, /) -> bool: ...
    def __repr__(self) -> str: ...

@final
class FieldFftStats:
    @property
    def particle_forward_ffts(self) -> int: ...
    @property
    def kernel_forward_ffts(self) -> int: ...
    @property
    def inverse_ffts(self) -> int: ...
    @property
    def total_fft_2d_calls(self) -> int: ...
    @property
    def nx(self) -> int: ...
    @property
    def ny(self) -> int: ...
    def to_dict(self) -> dict[str, int]: ...
    def keys(self) -> list[str]: ...
    def __getitem__(self, key: str, /) -> int: ...
    def __contains__(self, key: object, /) -> bool: ...
    def __repr__(self) -> str: ...

@final
class TsNetStage:
    def __new__(
        cls,
        iterations: int,
        *,
        exaggeration: float = 1.0,
        lambda_kl: float = 1.0,
        lambda_c: float = 0.0,
        lambda_r: float = 0.0,
    ) -> TsNetStage: ...
    @property
    def iterations(self) -> int: ...
    @property
    def exaggeration(self) -> float: ...
    @property
    def lambda_kl(self) -> float: ...
    @property
    def lambda_c(self) -> float: ...
    @property
    def lambda_r(self) -> float: ...
    def __repr__(self) -> str: ...

@final
class GuMapSolver:
    def __new__(
        cls,
        graph: CsrGraph,
        max_2hop_per_node: int = 32,
        *,
        target: TargetKind = "layout2d",
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
        iterations: int = 200,
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
        target: TargetKind = "layout2d",
        iterations: int = 200,
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
    def from_fuzzy_set(
        probs: SparseProbabilities,
        *,
        target: TargetKind = "layout2d",
        iterations: int = 200,
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
        target: TargetKind = "layout2d",
        iterations: int = 200,
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
        target: TargetKind = "layout2d",
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
        iterations: int = 200,
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
    def step(self, layout: _AnyLayout) -> SolverStatus: ...
    def run_steps(self, layout: _AnyLayout, steps: int) -> SolverStatus: ...
    def run(self, layout: _AnyLayout) -> SolverStatus: ...
    @property
    def target(self) -> TargetKind: ...
    @property
    def status(self) -> SolverStatus: ...
    @property
    def iteration(self) -> int: ...
    @property
    def total_iterations(self) -> int | None: ...
    @property
    def is_finished(self) -> bool: ...
    @property
    def num_nodes(self) -> int: ...
    @property
    def num_edges(self) -> int: ...
    @property
    def a(self) -> float: ...
    @property
    def b(self) -> float: ...
    @property
    def ab(self) -> tuple[float, float]: ...
    @property
    def dtype(self) -> str: ...
    def __repr__(self) -> str: ...

@final
class TsNetSolver:
    def __new__(
        cls,
        graph: CsrGraph,
        backend: BackendKind = "auto",
        *,
        target: TargetKind = "layout2d",
        perplexity_c: float = 0.9,
        min_perplexity: float = 5.0,
        max_perplexity: float = 1200.0,
        power: float = 2.0,
        sigma_iters: int = 50,
        sigma_tolerance: float = 1e-5,
        row_normalize: bool = False,
        stage_iterations: Sequence[int] | None = None,
        exaggeration: float = 4.0,
        lambda_c_stage1: float = 0.1,
        lambda_c_stage2: float = 0.01,
        lambda_c_stage3: float = 0.0,
        lambda_r_stage3: float = 0.0,
        learning_rate: float | None = None,
        learning_rate_divisor: float = 3.0,
        momentum: float = 0.8,
        epsilon_r: float = 0.05,
        theta: float = 0.5,
        intervals: int = 25,
        interpolation_order: int = 3,
        min_grid_dim: int = 64,
        max_grid_dim: int = 512,
        jacobi_precondition: bool = True,
        seed: int = 42,
        dtype: _DtypeLike = "float64",
    ) -> TsNetSolver: ...
    @staticmethod
    def from_probabilities(
        probs: SparseProbabilities,
        backend: BackendKind = "auto",
        *,
        target: TargetKind = "layout2d",
        stage_iterations: Sequence[int] | None = None,
        exaggeration: float = 4.0,
        lambda_c_stage1: float = 0.1,
        lambda_c_stage2: float = 0.01,
        lambda_c_stage3: float = 0.0,
        lambda_r_stage3: float = 0.0,
        learning_rate: float | None = None,
        learning_rate_divisor: float = 3.0,
        momentum: float = 0.8,
        epsilon_r: float = 0.05,
        theta: float = 0.5,
        intervals: int = 25,
        interpolation_order: int = 3,
        min_grid_dim: int = 64,
        max_grid_dim: int = 512,
        jacobi_precondition: bool = True,
        seed: int = 42,
        dtype: _DtypeLike = "float64",
    ) -> TsNetSolver: ...
    @staticmethod
    def from_probabilities_with_stages(
        probs: SparseProbabilities,
        stages: Sequence[TsNetStage],
        *,
        target: TargetKind = "layout2d",
        backend: BackendKind = "auto",
        learning_rate: float | None = None,
        learning_rate_divisor: float = 3.0,
        jacobi_precondition: bool = True,
        momentum: float = 0.8,
        epsilon_r: float = 0.05,
        theta: float = 0.5,
        intervals: int = 25,
        interpolation_order: int = 3,
        min_grid_dim: int = 64,
        max_grid_dim: int = 512,
        seed: int = 42,
        dtype: _DtypeLike = "float64",
    ) -> TsNetSolver: ...
    @staticmethod
    def from_knn(
        knn: KnnGraph,
        perplexities: _WeightArrayLike | float | None = None,
        uniform_perplexity: float = 15.0,
        *,
        target: TargetKind = "layout2d",
        backend: BackendKind = "auto",
        power: float = 2.0,
        sigma_iters: int = 50,
        sigma_tolerance: float = 1e-5,
        row_normalize: bool = False,
        stage_iterations: Sequence[int] | None = None,
        exaggeration: float = 4.0,
        lambda_c_stage1: float = 0.1,
        lambda_c_stage2: float = 0.01,
        lambda_c_stage3: float = 0.0,
        lambda_r_stage3: float = 0.0,
        learning_rate: float | None = None,
        learning_rate_divisor: float = 3.0,
        momentum: float = 0.8,
        epsilon_r: float = 0.05,
        theta: float = 0.5,
        intervals: int = 25,
        interpolation_order: int = 3,
        min_grid_dim: int = 64,
        max_grid_dim: int = 512,
        jacobi_precondition: bool = True,
        seed: int = 42,
        dtype: _DtypeLike = "float64",
    ) -> TsNetSolver: ...
    @staticmethod
    def pure_2hop(
        graph: CsrGraph,
        backend: BackendKind = "auto",
        *,
        target: TargetKind = "layout2d",
        perplexity_c: float = 0.9,
        min_perplexity: float = 5.0,
        max_perplexity: float = 1200.0,
        power: float = 2.0,
        sigma_iters: int = 50,
        sigma_tolerance: float = 1e-5,
        row_normalize: bool = False,
        stage_iterations: Sequence[int] | None = None,
        exaggeration: float = 4.0,
        lambda_c_stage1: float = 0.1,
        lambda_c_stage2: float = 0.01,
        lambda_c_stage3: float = 0.0,
        lambda_r_stage3: float = 0.0,
        learning_rate: float | None = None,
        learning_rate_divisor: float = 3.0,
        momentum: float = 0.8,
        epsilon_r: float = 0.05,
        theta: float = 0.5,
        intervals: int = 25,
        interpolation_order: int = 3,
        min_grid_dim: int = 64,
        max_grid_dim: int = 512,
        jacobi_precondition: bool = True,
        seed: int = 42,
        dtype: _DtypeLike = "float64",
    ) -> TsNetSolver: ...
    @staticmethod
    def from_graph_partial_bfs(
        graph: CsrGraph,
        *,
        k: int = 15,
        uniform_perplexity: float = 15.0,
        target: TargetKind = "layout2d",
        backend: BackendKind = "auto",
        power: float = 2.0,
        sigma_iters: int = 50,
        sigma_tolerance: float = 1e-5,
        row_normalize: bool = False,
        stage_iterations: Sequence[int] | None = None,
        exaggeration: float = 4.0,
        lambda_c_stage1: float = 0.1,
        lambda_c_stage2: float = 0.01,
        lambda_c_stage3: float = 0.0,
        lambda_r_stage3: float = 0.0,
        learning_rate: float | None = None,
        learning_rate_divisor: float = 3.0,
        momentum: float = 0.8,
        epsilon_r: float = 0.05,
        theta: float = 0.5,
        intervals: int = 25,
        interpolation_order: int = 3,
        min_grid_dim: int = 64,
        max_grid_dim: int = 512,
        jacobi_precondition: bool = True,
        seed: int = 42,
        dtype: _DtypeLike = "float64",
    ) -> TsNetSolver: ...
    @staticmethod
    def from_graph_row_normalized(
        graph: CsrGraph,
        *,
        w2: float = 0.35,
        max_2hop_per_node: int | None = 32,
        degree_beta: float = 0.0,
        mode: str = "degree_ordered_hub_suppressed",
        max_bridge_degree: int | None = None,
        target: TargetKind = "layout2d",
        backend: BackendKind = "auto",
        stage_iterations: Sequence[int] | None = None,
        exaggeration: float = 4.0,
        lambda_c_stage1: float = 0.1,
        lambda_c_stage2: float = 0.01,
        lambda_c_stage3: float = 0.0,
        lambda_r_stage3: float = 0.0,
        learning_rate: float | None = None,
        learning_rate_divisor: float = 3.0,
        momentum: float = 0.8,
        epsilon_r: float = 0.05,
        theta: float = 0.5,
        intervals: int = 25,
        interpolation_order: int = 3,
        min_grid_dim: int = 64,
        max_grid_dim: int = 512,
        jacobi_precondition: bool = True,
        seed: int = 42,
        dtype: _DtypeLike = "float64",
    ) -> TsNetSolver: ...
    def validate_target(self, layout: _AnyLayout) -> None: ...
    def step(self, layout: _AnyLayout) -> SolverStatus: ...
    def run_steps(self, layout: _AnyLayout, steps: int) -> SolverStatus: ...
    def run(self, layout: _AnyLayout) -> SolverStatus: ...
    @property
    def target(self) -> TargetKind: ...
    @property
    def status(self) -> SolverStatus: ...
    @property
    def iteration(self) -> int: ...
    @property
    def total_iterations(self) -> int | None: ...
    @property
    def is_finished(self) -> bool: ...
    @property
    def resolved_backend(self) -> str: ...
    @property
    def total_fft_2d_calls(self) -> int: ...
    def last_fft_stats(self) -> FitFieldStats | None: ...
    @property
    def num_nodes(self) -> int: ...
    @property
    def stage_index(self) -> int: ...
    @property
    def current_stage_lr(self) -> float: ...
    @property
    def dtype(self) -> str: ...
    def __repr__(self) -> str: ...
