"""Interoperability helpers between ``egraph.CsrGraph`` / layouts and Python graph ecosystems."""

from __future__ import annotations

import importlib
from collections.abc import Sequence
from typing import Any

import numpy as np

from egraph._core import CsrGraph, Layout2d, LayoutNd, ManifoldLayout2d


def _import_scipy_sparse() -> Any:
    try:
        return importlib.import_module("scipy.sparse")
    except ImportError as exc:
        raise ImportError(
            "scipy is required for SciPy CSR interoperability; install egraph[interop]."
        ) from exc


def _import_networkx() -> Any:
    try:
        return importlib.import_module("networkx")
    except ImportError as exc:
        raise ImportError(
            "networkx is required for NetworkX interoperability; install egraph[interop]."
        ) from exc


def _import_igraph() -> Any:
    try:
        return importlib.import_module("igraph")
    except ImportError as exc:
        raise ImportError(
            "igraph is required for igraph interoperability; install egraph[interop]."
        ) from exc


def from_scipy_csr(
    matrix: Any,
    *,
    directed: bool = False,
    sanitize: bool = True,
) -> CsrGraph:
    """Convert a square SciPy sparse matrix/array into an ``egraph.CsrGraph``."""
    sp = _import_scipy_sparse()
    if not sp.issparse(matrix):
        raise TypeError(
            f"expected a scipy.sparse matrix or array, got {type(matrix).__name__}"
        )

    csr = matrix.tocsr()
    if len(csr.shape) != 2 or csr.shape[0] != csr.shape[1]:
        raise ValueError(
            f"expected a square 2D sparse matrix, got shape {csr.shape}"
        )

    if sanitize:
        csr = csr.copy()
        csr.sum_duplicates()
        csr.eliminate_zeros()

    indptr = np.asarray(csr.indptr, dtype=np.int64)
    indices = np.asarray(csr.indices, dtype=np.int64)
    weights: np.ndarray[Any, np.dtype[np.float64]] | None
    if csr.data.dtype == np.bool_:
        weights = None
    else:
        weights = np.asarray(csr.data, dtype=np.float64)

    return CsrGraph.from_csr(
        indptr,
        indices,
        weights,
        directed=directed,
        sanitize=sanitize,
    )


def to_scipy_csr(graph: CsrGraph) -> Any:
    """Convert an ``egraph.CsrGraph`` to a ``scipy.sparse.csr_matrix``."""
    if not isinstance(graph, CsrGraph):
        raise TypeError(f"expected CsrGraph, got {type(graph).__name__}")
    sp = _import_scipy_sparse()
    n = graph.num_nodes
    indptr = graph.indptr()
    indices = graph.indices()
    w = graph.weights()
    data = w if w is not None else np.ones(len(indices), dtype=np.float64)
    return sp.csr_matrix((data, indices, indptr), shape=(n, n))


def from_networkx(
    graph: Any,
    *,
    weight: str | None = "weight",
    sanitize: bool = True,
) -> tuple[CsrGraph, list[Any]]:
    """Convert a NetworkX ``Graph``/``DiGraph`` to ``(CsrGraph, node_list)``."""
    nx = _import_networkx()
    if not isinstance(graph, nx.Graph):
        raise TypeError(
            f"expected a networkx.Graph or networkx.DiGraph, got {type(graph).__name__}"
        )

    nodes = list(graph.nodes())
    node_to_idx = {node: idx for idx, node in enumerate(nodes)}
    directed = bool(graph.is_directed())

    edges_raw = list(graph.edges(data=True))
    has_weights = weight is not None and any(weight in data for _, _, data in edges_raw)

    edge_pairs: list[tuple[int, int]] = []
    weight_vals: list[float] = []
    for u, v, data in edges_raw:
        edge_pairs.append((node_to_idx[u], node_to_idx[v]))
        if has_weights:
            assert weight is not None
            raw_w = data.get(weight, 1.0)
            if isinstance(raw_w, bool):
                raise TypeError("edge weight attribute must be numeric, not bool")
            weight_vals.append(float(raw_w))

    edges_arr = np.asarray(edge_pairs, dtype=np.uint32).reshape(-1, 2)
    weights_arr = np.asarray(weight_vals, dtype=np.float64) if has_weights else None
    csr = CsrGraph.from_edges(
        len(nodes),
        edges_arr,
        weights_arr,
        directed=directed,
        sanitize=sanitize,
    )
    return csr, nodes


def to_networkx(
    graph: CsrGraph,
    nodes: Sequence[Any] | None = None,
    *,
    layout: Layout2d | LayoutNd | ManifoldLayout2d | None = None,
) -> Any:
    """Convert an ``egraph.CsrGraph`` to a NetworkX ``Graph`` or ``DiGraph``."""
    if not isinstance(graph, CsrGraph):
        raise TypeError(f"expected CsrGraph, got {type(graph).__name__}")
    nx = _import_networkx()

    n = graph.num_nodes
    if nodes is not None:
        node_list = list(nodes)
        if len(node_list) != n:
            raise ValueError(
                f"nodes length ({len(node_list)}) must match graph.num_nodes ({n})"
            )
    else:
        node_list = list(range(n))

    if layout is not None:
        if not isinstance(layout, (Layout2d, LayoutNd, ManifoldLayout2d)):
            raise TypeError(
                f"expected Layout2d, LayoutNd, or ManifoldLayout2d, got {type(layout).__name__}"
            )
        if len(layout) != n:
            raise ValueError(
                f"layout length ({len(layout)}) must match graph.num_nodes ({n})"
            )

    out = nx.DiGraph() if graph.is_directed else nx.Graph()
    if layout is not None:
        coords = layout.coords(copy=True)
        for idx, node in enumerate(node_list):
            out.add_node(node, pos=tuple(float(v) for v in coords[idx]))
    else:
        out.add_nodes_from(node_list)

    indptr = graph.indptr()
    indices = graph.indices()
    weights = graph.weights()
    directed = graph.is_directed

    for u in range(n):
        start = int(indptr[u])
        end = int(indptr[u + 1])
        src_node = node_list[u]
        for idx in range(start, end):
            v = int(indices[idx])
            if not directed and u > v:
                continue
            dst_node = node_list[v]
            if weights is not None:
                out.add_edge(src_node, dst_node, weight=float(weights[idx]))
            else:
                out.add_edge(src_node, dst_node)

    return out


def from_igraph(
    graph: Any,
    *,
    weight: str | None = "weight",
    sanitize: bool = True,
) -> CsrGraph:
    """Convert an ``igraph.Graph`` into an ``egraph.CsrGraph``."""
    ig = _import_igraph()
    if not isinstance(graph, ig.Graph):
        raise TypeError(f"expected an igraph.Graph, got {type(graph).__name__}")

    num_nodes = int(graph.vcount())
    directed = bool(graph.is_directed())
    edge_list = graph.get_edgelist()
    edges_arr = np.asarray(edge_list, dtype=np.uint32).reshape(-1, 2)

    weights_arr: np.ndarray[Any, np.dtype[np.float64]] | None = None
    if weight is not None and weight in graph.es.attributes():
        raw_weights = graph.es[weight]
        parsed_weights: list[float] = []
        for raw_w in raw_weights:
            if raw_w is None:
                parsed_weights.append(1.0)
            elif isinstance(raw_w, bool):
                raise TypeError("edge weight attribute must be numeric, not bool")
            else:
                parsed_weights.append(float(raw_w))
        weights_arr = np.asarray(parsed_weights, dtype=np.float64)

    return CsrGraph.from_edges(
        num_nodes,
        edges_arr,
        weights_arr,
        directed=directed,
        sanitize=sanitize,
    )


def to_igraph(
    graph: CsrGraph,
    nodes: Sequence[Any] | None = None,
    *,
    layout: Layout2d | LayoutNd | ManifoldLayout2d | None = None,
) -> Any:
    """Convert an ``egraph.CsrGraph`` to an ``igraph.Graph``."""
    if not isinstance(graph, CsrGraph):
        raise TypeError(f"expected CsrGraph, got {type(graph).__name__}")
    ig = _import_igraph()

    n = graph.num_nodes
    node_list: list[Any] | None = None
    if nodes is not None:
        node_list = list(nodes)
        if len(node_list) != n:
            raise ValueError(
                f"nodes length ({len(node_list)}) must match graph.num_nodes ({n})"
            )

    if layout is not None:
        if not isinstance(layout, (Layout2d, LayoutNd, ManifoldLayout2d)):
            raise TypeError(
                f"expected Layout2d, LayoutNd, or ManifoldLayout2d, got {type(layout).__name__}"
            )
        if len(layout) != n:
            raise ValueError(
                f"layout length ({len(layout)}) must match graph.num_nodes ({n})"
            )

    indptr = graph.indptr()
    indices = graph.indices()
    weights = graph.weights()
    directed = graph.is_directed

    edge_list: list[tuple[int, int]] = []
    edge_weights: list[float] = []
    for u in range(n):
        start = int(indptr[u])
        end = int(indptr[u + 1])
        for idx in range(start, end):
            v = int(indices[idx])
            if not directed and u > v:
                continue
            edge_list.append((u, v))
            if weights is not None:
                edge_weights.append(float(weights[idx]))

    out = ig.Graph(n=n, edges=edge_list, directed=directed)
    if weights is not None:
        out.es["weight"] = edge_weights
    if node_list is not None:
        out.vs["name"] = node_list
    if layout is not None:
        coords = layout.coords(copy=True)
        out.vs["pos"] = [tuple(float(v) for v in coords[idx]) for idx in range(n)]
    return out


__all__ = [
    "from_igraph",
    "from_networkx",
    "from_scipy_csr",
    "to_igraph",
    "to_networkx",
    "to_scipy_csr",
]
