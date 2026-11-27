from __future__ import annotations

import unittest
from typing import Any

import egraph as eg
import numpy as np
import numpy.typing as npt


def _to_float_array(
    coords: npt.NDArray[np.float64], dtype_str: str
) -> npt.NDArray[np.floating[Any]]:
    if dtype_str == "float32":
        return coords.astype(np.float32)
    return coords.astype(np.float64)


def _count_rect_overlaps(
    coords: npt.NDArray[np.floating[Any]],
    half_widths: npt.NDArray[np.floating[Any]],
    half_heights: npt.NDArray[np.floating[Any]],
    tol: float = 1e-9,
) -> int:
    n = coords.shape[0]
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            dx = abs(float(coords[i, 0]) - float(coords[j, 0]))
            dy = abs(float(coords[i, 1]) - float(coords[j, 1]))
            req_x = float(half_widths[i]) + float(half_widths[j])
            req_y = float(half_heights[i]) + float(half_heights[j])
            if dx < req_x - tol and dy < req_y - tol:
                count += 1
    return count


class TestConstraintsAndLayering(unittest.TestCase):
    def test_prism_overlap_removal_zero_overlaps(self) -> None:
        rng = np.random.default_rng(77)
        n = 16
        # Dense overlapping initial cluster including coincident points
        base = rng.uniform(-0.2, 0.2, size=(n, 2)).astype(np.float64)
        base[1] = base[0]
        hw = np.full(n, 0.5, dtype=np.float64)
        hh = np.full(n, 0.4, dtype=np.float64)

        for dtype_str in ("float64", "float32"):
            l_run = eg.Layout2d.from_coords(_to_float_array(base, dtype_str))
            l_step = eg.Layout2d.from_coords(_to_float_array(base, dtype_str))
            p_run = eg.PrismOverlapRemoval(hw, hh, max_iterations=50, padding=0.05, seed=42, dtype=dtype_str)
            p_step = eg.PrismOverlapRemoval(hw, hh, max_iterations=50, padding=0.05, seed=42, dtype=dtype_str)
            self.assertEqual(p_run.num_nodes, n)
            self.assertEqual(p_run.dtype, dtype_str)
            self.assertIn("PrismOverlapRemoval", repr(p_run))

            status = p_run.run(l_run)
            self.assertEqual(status, "converged")
            self.assertEqual(p_run.last_overlap_count, 0)
            self.assertTrue(p_run.is_finished)

            p_step.step(l_step)
            p_step.run_steps(l_step, 3)
            p_step.run(l_step)
            np.testing.assert_array_equal(l_run.coords(copy=True), l_step.coords(copy=True))

            tol = 1e-4 if dtype_str == "float32" else 1e-8
            self.assertEqual(_count_rect_overlaps(l_run.coords(copy=True), hw + 0.025, hh + 0.025, tol=tol), 0)

        # from_radii constructor
        radii = np.full(n, 0.35, dtype=np.float64)
        l_rad = eg.Layout2d.from_coords(base)
        pr = eg.PrismOverlapRemoval.from_radii(radii, max_iterations=50, seed=42)
        pr.run(l_rad)
        self.assertEqual(pr.last_overlap_count, 0)
        self.assertEqual(_count_rect_overlaps(l_rad.coords(copy=True), radii, radii), 0)

    def test_separation_constraints_and_overlap_projection(self) -> None:
        c = eg.SeparationConstraint(0, 1, 2.5)
        self.assertEqual(c.left, 0)
        self.assertEqual(c.right, 1)
        self.assertAlmostEqual(c.gap, 2.5)
        self.assertIn("SeparationConstraint", repr(c))

        # project_1d on float64, float32 (in-place -> None); TypeError on read-only / Python sequence
        arr64 = np.array([1.0, 0.0, 0.5], dtype=np.float64)
        eg.project_1d(arr64, [eg.SeparationConstraint(0, 1, 2.0), eg.SeparationConstraint(1, 2, 1.5)])
        self.assertEqual(arr64.dtype, np.float64)
        self.assertGreaterEqual(float(arr64[1] - arr64[0]), 2.0 - 1e-9)
        self.assertGreaterEqual(float(arr64[2] - arr64[1]), 1.5 - 1e-9)

        arr32 = np.array([1.0, 0.0, 0.5], dtype=np.float32)
        eg.project_1d(arr32, [eg.SeparationConstraint(0, 1, 2.0)])
        self.assertEqual(arr32.dtype, np.float32)
        self.assertGreaterEqual(float(arr32[1] - arr32[0]), 2.0 - 1e-6)

        ro = np.array([1.0, 0.0], dtype=np.float64)
        ro.flags.writeable = False
        with self.assertRaises(TypeError):
            eg.project_1d(ro, [eg.SeparationConstraint(0, 1, 3.0)])
        self.assertAlmostEqual(float(ro[0]), 1.0)

        with self.assertRaises(TypeError):
            eg.project_1d([2.0, 0.0], [eg.SeparationConstraint(0, 1, 1.0)])  # type: ignore[arg-type]

        # project_layout_axis + generate_layered_constraints
        dag = eg.CsrGraph.from_edges(4, [(0, 1), (1, 2), (0, 3), (3, 2)], directed=True, sanitize=True)
        layers = eg.assign_layers(dag, method="longest_path")
        layered_cs = eg.generate_layered_constraints(dag, layers, min_layer_distance=2.0)
        self.assertGreaterEqual(len(layered_cs), 4)

        for dtype_str in ("float64", "float32"):
            layout = eg.Layout2d.zeros(4, dtype=dtype_str)
            eg.project_layout_axis(layout, 1, layered_cs)
            ys = layout.coords(copy=True)[:, 1]
            self.assertGreaterEqual(float(ys[2] - ys[0]), 4.0 - 1e-5)

        # project_rectangle_no_overlap_2d
        overlap_coords = np.array([[0.0, 0.0], [0.2, 0.1], [0.1, -0.1], [-0.1, 0.2]], dtype=np.float64)
        hw = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float64)
        hh = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float64)
        for dtype_str in ("float64", "float32"):
            l_rect = eg.Layout2d.from_coords(_to_float_array(overlap_coords, dtype_str))
            eg.project_rectangle_no_overlap_2d(l_rect, hw, hh)
            tol = 1e-5 if dtype_str == "float32" else 1e-9
            self.assertEqual(_count_rect_overlaps(l_rect.coords(copy=True), hw, hh, tol=tol), 0)

        # project_clustered_rectangle_no_overlap_2d preserves intra-cluster relative positions
        c_coords = np.array([[0.0, 0.0], [0.5, 0.0], [0.3, 0.2], [0.8, 0.2]], dtype=np.float64)
        cluster_ids = np.array([0, 0, 1, 1], dtype=np.uint32)
        for dtype_str in ("float64", "float32"):
            l_cl = eg.Layout2d.from_coords(_to_float_array(c_coords, dtype_str))
            eg.project_clustered_rectangle_no_overlap_2d(l_cl, cluster_ids, hw, hh)
            out_c = l_cl.coords(copy=True)
            # Intra-cluster vector (node 1 - node 0) and (node 3 - node 2) preserved rigidly
            np.testing.assert_allclose(out_c[1] - out_c[0], [0.5, 0.0], atol=1e-5)
            np.testing.assert_allclose(out_c[3] - out_c[2], [0.5, 0.0], atol=1e-5)
            # Cluster 0 and Cluster 1 bounding boxes do not overlap
            c0_min_x = min(out_c[0, 0] - 0.5, out_c[1, 0] - 0.5)
            c0_max_x = max(out_c[0, 0] + 0.5, out_c[1, 0] + 0.5)
            c0_min_y = min(out_c[0, 1] - 0.5, out_c[1, 1] - 0.5)
            c0_max_y = max(out_c[0, 1] + 0.5, out_c[1, 1] + 0.5)
            c1_min_x = min(out_c[2, 0] - 0.5, out_c[3, 0] - 0.5)
            c1_max_x = max(out_c[2, 0] + 0.5, out_c[3, 0] + 0.5)
            c1_min_y = min(out_c[2, 1] - 0.5, out_c[3, 1] - 0.5)
            c1_max_y = max(out_c[2, 1] + 0.5, out_c[3, 1] + 0.5)
            disjoint_x = c0_max_x <= c1_min_x + 1e-5 or c1_max_x <= c0_min_x + 1e-5
            disjoint_y = c0_max_y <= c1_min_y + 1e-5 or c1_max_y <= c0_min_y + 1e-5
            self.assertTrue(disjoint_x or disjoint_y)

    def test_layering_cycle_edges_remove_cycles_and_methods(self) -> None:
        cyclic = eg.CsrGraph.from_edges(
            4, [(0, 1), (1, 2), (2, 0), (1, 3)], weights=[1.0, 2.0, 3.0, 4.0], directed=True, sanitize=True
        )
        ce = eg.cycle_edges(cyclic)
        self.assertEqual(ce.dtype, np.uint32)
        self.assertEqual(ce.ndim, 2)
        self.assertEqual(ce.shape[1], 2)
        self.assertGreaterEqual(ce.shape[0], 1)

        dag = eg.remove_cycles(cyclic)
        self.assertTrue(dag.is_directed)
        self.assertEqual(dag.num_nodes, 4)
        ce_after = eg.cycle_edges(dag)
        self.assertEqual(ce_after.shape, (0, 2))
        self.assertEqual(ce_after.dtype, np.uint32)

        # Skip-edge DAG where NetworkSimplex reduces total edge span compared to LongestPath:
        # LongestPath puts 4 at layer 0 (4->3 span=3), whereas NetworkSimplex places 4 at layer 2 (span=1).
        skip_edges = [(0, 1), (1, 2), (2, 3), (4, 3)]
        skip_dag = eg.CsrGraph.from_edges(5, skip_edges, directed=True, sanitize=True)
        lp = eg.assign_layers(skip_dag, method="longest_path")
        ns = eg.assign_layers(skip_dag, method="network_simplex")
        self.assertEqual(lp.dtype, np.uint32)
        self.assertEqual(ns.dtype, np.uint32)
        self.assertEqual(lp.shape, (5,))
        self.assertEqual(ns.shape, (5,))

        for u, v in skip_edges:
            self.assertLess(int(lp[u]), int(lp[v]))
            self.assertLess(int(ns[u]), int(ns[v]))

        lp_span = sum(int(lp[v]) - int(lp[u]) for u, v in skip_edges)
        ns_span = sum(int(ns[v]) - int(ns[u]) for u, v in skip_edges)
        self.assertLess(ns_span, lp_span)


if __name__ == "__main__":
    unittest.main()
