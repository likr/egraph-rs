from __future__ import annotations

from typing import Any, Callable, final

import numpy as np
import numpy.typing as npt

from ._core_core import Array2, Drawing, GraphAdapter, Rng
from ._core_types import _IndexArrayLike

__all__ = [
    "FullSgd",
    "Omega",
    "RandomPairSparseSgd",
    "SchedulerConstant",
    "SchedulerExponential",
    "SchedulerLinear",
    "SchedulerQuadratic",
    "SchedulerReciprocal",
    "Sgd",
    "SparseSgd",
]

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
class Omega:
    def __new__(cls) -> Omega: ...
    def k(self, k: int) -> Omega: ...
    def min_dist(self, min_dist: float) -> Omega: ...
    def build(
        self, graph: GraphAdapter, embedding: Array2, rng: Rng
    ) -> Sgd: ...

@final
class RandomPairSparseSgd:
    def __new__(cls) -> RandomPairSparseSgd: ...
    def k(self, k: int) -> RandomPairSparseSgd: ...
    def build(
        self, graph: GraphAdapter, distance_matrix: Any, rng: Rng
    ) -> Sgd: ...

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
