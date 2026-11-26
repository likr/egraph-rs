"""Shared fixtures, tolerance checks, and Node.js ESM helpers for parity suites."""

from __future__ import annotations

import json
import math
import pathlib
import subprocess
from collections.abc import Callable
from typing import Any


def make_fixtures() -> dict[str, dict[str, Any]]:
    """Build deterministic canonical graph fixtures."""
    barbell_edges = [
        (0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3),
        (3, 4), (4, 5),
        (5, 6), (5, 7), (5, 8), (6, 7), (6, 8), (7, 8),
    ]
    return {
        "path_10": {
            "n": 10,
            "edges": [(i, i + 1) for i in range(9)],
        },
        "cycle_12": {
            "n": 12,
            "edges": [(i, (i + 1) % 12) for i in range(12)],
        },
        "barbell_2comm_9": {
            "n": 9,
            "edges": barbell_edges,
        },
        "star_11": {
            "n": 11,
            "edges": [(0, i) for i in range(1, 11)],
        },
        "disconnected_10": {
            "n": 10,
            "edges": [(0, 1), (1, 2), (2, 0), (4, 5), (5, 6), (7, 8)],
        },
    }


def make_fixture_coords(name: str, n: int) -> list[tuple[float, float]]:
    """Build deterministic, finite, distinct 2D coordinates for a canonical fixture."""
    if name == "path_10":
        return [(float(i), 0.25 * math.sin(0.7 * float(i))) for i in range(n)]
    if name == "cycle_12":
        return [
            (math.cos(2.0 * math.pi * i / n), math.sin(2.0 * math.pi * i / n))
            for i in range(n)
        ]
    if name == "barbell_2comm_9":
        coords: list[tuple[float, float]] = []
        for i in range(n):
            if i < 4:
                ang = i * math.pi / 2.0
                coords.append((-2.5 + 0.6 * math.cos(ang), 0.6 * math.sin(ang)))
            elif i == 4:
                coords.append((0.0, 0.25))
            else:
                ang = (i - 5) * math.pi / 2.0 + 0.35
                coords.append((2.5 + 0.6 * math.cos(ang), 0.6 * math.sin(ang)))
        return coords
    if name == "star_11":
        coords = [(0.0, 0.0)]
        for i in range(1, n):
            ang = 2.0 * math.pi * (i - 1) / max(n - 1, 1)
            coords.append((math.cos(ang), math.sin(ang)))
        return coords
    if name == "disconnected_10":
        return [
            (
                1.5 * float(i % 4) + 0.2 * math.cos(float(i)),
                1.5 * float(i // 4) + 0.2 * math.sin(float(i)),
            )
            for i in range(n)
        ]
    raise ValueError(f"Unknown fixture name: {name}")


def make_similarity_coords(coords: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Apply a deterministic similarity transform (rotation + uniform scale + translation)."""
    theta = 0.7
    scale = 2.5
    c = math.cos(theta)
    s = math.sin(theta)
    return [
        (scale * (c * x - s * y) + 4.0, scale * (s * x + c * y) - 3.0)
        for x, y in coords
    ]


def make_adjusted_coords(coords: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Apply a deterministic similarity transform plus non-rigid perturbation for PRISM checks."""
    theta = 0.4
    scale = 1.35
    c = math.cos(theta)
    s = math.sin(theta)
    return [
        (
            scale * (c * x - s * y) + 2.0 + 0.08 * math.sin(1.3 * float(idx)),
            scale * (s * x + c * y) - 1.5 - 0.06 * math.cos(0.9 * float(idx)),
        )
        for idx, (x, y) in enumerate(coords)
    ]


def make_grid_star_graph_spec() -> tuple[int, list[tuple[int, int]]]:
    """Build deterministic 70-node graph (8x8 grid + bridge + 4-leaf star + isolated node)."""
    edges: list[tuple[int, int]] = []
    for r in range(8):
        for c in range(8):
            u = r * 8 + c
            if c + 1 < 8:
                edges.append((u, u + 1))
            if r + 1 < 8:
                edges.append((u, u + 8))
    edges.append((63, 64))
    for leaf in (65, 66, 67, 68):
        edges.append((64, leaf))
    return 70, edges


def make_grid_edges(rows: int, cols: int) -> list[tuple[int, int]]:
    """Build undirected edges for a rows x cols 2D grid graph."""
    edges: list[tuple[int, int]] = []
    for r in range(rows):
        for c in range(cols):
            u = r * cols + c
            if c + 1 < cols:
                edges.append((u, u + 1))
            if r + 1 < rows:
                edges.append((u, u + cols))
    return edges


def make_barbell_edges(clique_size: int) -> list[tuple[int, int]]:
    """Build a 2-clique barbell graph with a single bridge edge."""
    edges: list[tuple[int, int]] = []
    for off in (0, clique_size):
        for i in range(clique_size):
            for j in range(i + 1, clique_size):
                edges.append((off + i, off + j))
    edges.append((clique_size - 1, clique_size))
    return edges


def make_asymmetric_coords(n: int, scale: float = 1.0) -> list[tuple[float, float]]:
    """Generate deterministic non-symmetric 2D coordinates without gradient ties."""
    return [
        (
            scale * (0.7 * math.cos(0.83 * float(i) + 0.15) + 0.11 * float(i) / max(n, 1)),
            scale * (0.6 * math.sin(1.13 * float(i) - 0.21) - 0.09 * float(i) / max(n, 1)),
        )
        for i in range(n)
    ]


def assert_finite_coords(coords: list[tuple[float, float]], label: str) -> None:
    """Assert that every (x, y) coordinate pair is finite."""
    for idx, (x, y) in enumerate(coords):
        if not (math.isfinite(x) and math.isfinite(y)):
            raise AssertionError(f"{label}: non-finite coord at node {idx}: ({x}, {y})")


def assert_non_panic_error(
    label: str,
    fn: Callable[[], Any],
    expected_types: tuple[type[BaseException], ...] = (ValueError, TypeError, IndexError),
) -> None:
    """Verify invalid inputs raise clean Python exceptions rather than PanicException."""
    caught = False
    try:
        fn()
    except expected_types as exc:
        if "PanicException" in type(exc).__name__:
            raise AssertionError(f"{label} raised PanicException: {exc}") from exc
        caught = True
    except BaseException as exc:
        if "PanicException" in type(exc).__name__:
            raise AssertionError(f"{label} raised PanicException: {exc}") from exc
        raise AssertionError(f"{label} raised unexpected {type(exc).__name__}: {exc}") from exc
    assert caught, f"{label} silently accepted invalid input"


def bind_call(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Callable[[], Any]:
    """Bind positional and keyword arguments into a zero-argument typed callable."""
    def _call() -> Any:
        return fn(*args, **kwargs)
    return _call


def count_rect_overlaps(coords: Any, hw: Any, hh: Any, tol: float = 1e-6) -> int:
    """Count axis-aligned rectangle overlaps in pure Python."""
    n = len(coords)
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            dx = abs(float(coords[i, 0]) - float(coords[j, 0]))
            dy = abs(float(coords[i, 1]) - float(coords[j, 1]))
            if dx < float(hw[i]) + float(hw[j]) - tol and dy < float(hh[i]) + float(hh[j]) - tol:
                count += 1
    return count


def check_array_close(label: str, a: Any, b: Any, tol: float) -> float:
    """Assert two numeric array-like objects have identical shape and max diff < tol."""
    import numpy as np

    arr_a = np.asarray(a, dtype=np.float64).ravel()
    arr_b = np.asarray(b, dtype=np.float64).ravel()
    assert arr_a.shape == arr_b.shape, f"{label}: shape mismatch {arr_a.shape} vs {arr_b.shape}"
    diff = float(np.max(np.abs(arr_a - arr_b))) if arr_a.size else 0.0
    assert diff < tol, f"{label}: diff={diff} >= {tol}"
    return diff


def check_f64_vs_f32_rel(label: str, a64: Any, a32: Any, tol: float = 5e-3) -> float:
    """Assert float64 outputs are finite and match float32 within scale-relative tolerance."""
    import numpy as np

    arr64 = np.asarray(a64, dtype=np.float64).ravel()
    arr32 = np.asarray(a32, dtype=np.float64).ravel()
    assert np.all(np.isfinite(arr64)), f"{label}: non-finite float64 output"
    rel_scale = max(1.0, float(np.max(np.abs(arr64))))
    rel_diff = float(np.max(np.abs(arr64 - arr32))) / rel_scale if arr64.size else 0.0
    assert rel_diff < tol, (
        f"{label}: rel_diff={rel_diff} >= {tol} "
        f"(abs={float(np.max(np.abs(arr64 - arr32)))}, scale={rel_scale})"
    )
    return rel_diff


def resolve_wasm_entries(wasm_pkg: pathlib.Path) -> tuple[str, str, str] | None:
    """Return (node_url, index_url, wasm_path) if index.js exists in wasm_pkg, else None."""
    node_entry = wasm_pkg / "egraph_wasm.node.js"
    index_entry = wasm_pkg / "index.js"
    if not index_entry.exists():
        return None
    node_url = node_entry.resolve().as_uri() if node_entry.exists() else index_entry.resolve().as_uri()
    index_url = index_entry.resolve().as_uri()
    wasm_path = (wasm_pkg / "egraph_wasm_bg.wasm").resolve().as_posix()
    return node_url, index_url, wasm_path


_NODE_PREAMBLE = """
import { readFile } from 'node:fs/promises';
import assert from 'node:assert/strict';

const nodeUrl = process.argv[1];
const indexUrl = process.argv[2];
const wasmPath = process.argv[3];
const spec = JSON.parse(process.argv[4]);

const mod = await import(nodeUrl);
const indexMod = await import(indexUrl);
if (typeof mod.default === 'function') {
  try {
    const probe = mod.CsrGraph.fromCsr(new Uint32Array([0, 1, 2]), new Uint32Array([1, 0]));
    probe.free?.();
  } catch (_e) {
    await mod.default(await readFile(wasmPath));
  }
}

function assertNonPanicError(label, fn) {
  let caught = false;
  try {
    fn();
  } catch (err) {
    if (err instanceof WebAssembly.RuntimeError) {
      throw new Error(`${label} triggered WebAssembly.RuntimeError: ${err}`);
    }
    if (!(err instanceof Error)) {
      throw new Error(`${label} must throw Error`);
    }
    caught = true;
  }
  if (!caught) {
    throw new Error(`${label} unexpectedly succeeded`);
  }
}

function arrEq(a, b) {
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i += 1) {
    if (a[i] !== b[i]) return false;
  }
  return true;
}
"""


def run_node_script(wasm_pkg: pathlib.Path, js_body: str, spec: Any = None) -> None:
    """Execute a Node.js ESM script against wasm_pkg and assert zero exit status."""
    resolved = resolve_wasm_entries(wasm_pkg)
    if resolved is None:
        raise FileNotFoundError(f"Missing WASM index.js in {wasm_pkg}")
    node_url, index_url, wasm_path = resolved
    full_script = _NODE_PREAMBLE + "\n" + js_body
    subprocess.check_call(
        ["node", "--input-type=module", "-e", full_script, node_url, index_url, wasm_path, json.dumps(spec or {})]
    )


def run_node_json(wasm_pkg: pathlib.Path, js_body: str, spec: Any = None) -> dict[str, Any]:
    """Execute a Node.js ESM script against wasm_pkg and parse its final JSON line."""
    resolved = resolve_wasm_entries(wasm_pkg)
    if resolved is None:
        raise FileNotFoundError(f"Missing WASM index.js in {wasm_pkg}")
    node_url, index_url, wasm_path = resolved
    full_script = _NODE_PREAMBLE + "\n" + js_body
    raw_out = subprocess.check_output(
        ["node", "--input-type=module", "-e", full_script, node_url, index_url, wasm_path, json.dumps(spec or {})],
        text=True,
    )
    parsed: dict[str, Any] = json.loads(raw_out.strip().splitlines()[-1])
    return parsed
