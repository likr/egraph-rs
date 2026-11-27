from __future__ import annotations

from typing import Any, final

import numpy as np
import numpy.typing as npt

from ._core_core import CsrGraph, Layout2d
from ._core_types import _DtypeLike, _IndexArrayLike, _WeightArrayLike

__all__ = [
    "ForbidOverlapRemoval",
    "GTreeOverlapRemoval",
    "PfsPrimeOverlapRemoval",
    "PrismOverlapRemoval",
    "ScreenedPoissonField2d",
    "ScreenedPoissonRefiner",
    "SpatialIntruderRefiner",
]

@final
class ForbidOverlapRemoval:
    def __new__(
        cls,
        half_widths: _WeightArrayLike,
        half_heights: _WeightArrayLike,
        *,
        variant: str = "forbid",
        max_scale_iters: int = 20,
        sgd_iterations: int = 30,
        sparse_k: int = 30,
        overlap_weight_k: float = 1.0,
        scale_step: float = 0.1,
        padding: float = 0.0,
        seed: int = 42,
        dtype: _DtypeLike = "float64",
    ) -> ForbidOverlapRemoval: ...
    @staticmethod
    def from_radii(
        radii: _WeightArrayLike,
        *,
        variant: str = "forbid",
        max_scale_iters: int = 20,
        sgd_iterations: int = 30,
        sparse_k: int = 30,
        overlap_weight_k: float = 1.0,
        scale_step: float = 0.1,
        padding: float = 0.0,
        seed: int = 42,
        dtype: _DtypeLike = "float64",
    ) -> ForbidOverlapRemoval: ...
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
class GTreeOverlapRemoval:
    """GTree proximity-MST node overlap removal solver (``compact_t_max > 1.0`` when set)."""

    def __new__(
        cls,
        half_widths: _WeightArrayLike,
        half_heights: _WeightArrayLike,
        *,
        max_iterations: int = 20,
        padding: float = 0.0,
        compact_t_max: float | None = None,
        seed: int = 42,
        dtype: _DtypeLike = "float64",
    ) -> GTreeOverlapRemoval: ...
    @staticmethod
    def from_radii(
        radii: _WeightArrayLike,
        *,
        max_iterations: int = 20,
        padding: float = 0.0,
        compact_t_max: float | None = None,
        seed: int = 42,
        dtype: _DtypeLike = "float64",
    ) -> GTreeOverlapRemoval: ...
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
class PfsPrimeOverlapRemoval:
    def __new__(
        cls,
        half_widths: _WeightArrayLike,
        half_heights: _WeightArrayLike,
        *,
        padding: float = 0.0,
        seed: int = 42,
        dtype: _DtypeLike = "float64",
    ) -> PfsPrimeOverlapRemoval: ...
    @staticmethod
    def from_radii(
        radii: _WeightArrayLike,
        *,
        padding: float = 0.0,
        seed: int = 42,
        dtype: _DtypeLike = "float64",
    ) -> PfsPrimeOverlapRemoval: ...
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
class ScreenedPoissonField2d:
    def __new__(
        cls,
        *,
        kernel: str = "screened_poisson",
        grid_size: int = 512,
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
        self,
        layout: Layout2d,
        min_x: float,
        min_y: float,
        max_x: float,
        max_y: float,
    ) -> dict[str, Any]: ...
    def density_grid(self) -> npt.NDArray[np.float64]: ...
    @property
    def density_grid_dim(self) -> int: ...
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
        grid_size: int = 512,
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
        grid_size: int = 512,
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

@final
class SpatialIntruderRefiner:
    def __new__(
        cls,
        graph: CsrGraph,
        iterations: int = 4,
        k_cap: int = 32,
        margin_scale: float = 1.2,
        w_anchor: float = 1.0,
        w_fn: float = 0.0,
        w_fp: float = 0.2,
        fn_shrink: float = 0.88,
        round_decay: float = 0.75,
        sgd_iterations: int = 20,
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
