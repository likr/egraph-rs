from __future__ import annotations

from typing import Any, Sequence

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

__all__: list[str] = []
