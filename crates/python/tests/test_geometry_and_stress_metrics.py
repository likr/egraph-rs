from __future__ import annotations

import inspect
import math
import unittest
from typing import Any

import egraph as eg
import numpy as np


def _make_grid(rows: int, cols: int) -> tuple[eg.CsrGraph, eg.Layout2d]:
    edges: list[tuple[int, int]] = []
    coords: list[list[float]] = []
    for r in range(rows):
        for c in range(cols):
            idx = r * cols + c
            coords.append([float(c), float(r)])
            if c + 1 < cols:
                edges.append((idx, idx + 1))
            if r + 1 < rows:
                edges.append((idx, idx + cols))
    g = eg.CsrGraph.from_edges(rows * cols, edges, directed=False, sanitize=False)
    return g, eg.Layout2d.from_coords(np.array(coords, dtype=np.float64))


def _make_path(n: int) -> tuple[eg.CsrGraph, eg.Layout2d]:
    edges = [(i, i + 1) for i in range(n - 1)]
    g = eg.CsrGraph.from_edges(n, edges, directed=False, sanitize=False)
    coords = np.array([[float(i), 0.0] for i in range(n)], dtype=np.float64)
    return g, eg.Layout2d.from_coords(coords)


def _make_cycle(n: int) -> tuple[eg.CsrGraph, eg.Layout2d]:
    edges = [(i, (i + 1) % n) for i in range(n)]
    g = eg.CsrGraph.from_edges(n, edges, directed=False, sanitize=False)
    angles = np.linspace(0.0, 2.0 * math.pi, n, endpoint=False)
    coords = np.column_stack([np.cos(angles), np.sin(angles)])
    return g, eg.Layout2d.from_coords(coords)


def _make_barbell(clique_size: int, bridge_len: int) -> tuple[eg.CsrGraph, eg.Layout2d]:
    n = 2 * clique_size + bridge_len
    edges: list[tuple[int, int]] = []
    for i in range(clique_size):
        for j in range(i + 1, clique_size):
            edges.append((i, j))
    offset = clique_size + bridge_len
    for i in range(clique_size):
        for j in range(i + 1, clique_size):
            edges.append((offset + i, offset + j))
    prev = clique_size - 1
    for b in range(bridge_len):
        cur = clique_size + b
        edges.append((prev, cur))
        prev = cur
    edges.append((prev, offset))

    g = eg.CsrGraph.from_edges(n, edges, directed=False, sanitize=False)
    coords: list[list[float]] = []
    radius = 0.6
    for i in range(clique_size):
        ang = i * 2.0 * math.pi / clique_size
        coords.append([-3.0 + radius * math.cos(ang), radius * math.sin(ang)])
    for b in range(bridge_len):
        t = (b + 1) / (bridge_len + 1)
        coords.append([-2.0 + 4.0 * t, 0.0])
    for i in range(clique_size):
        ang = i * 2.0 * math.pi / clique_size
        coords.append([3.0 + radius * math.cos(ang), radius * math.sin(ang)])
    return g, eg.Layout2d.from_coords(np.array(coords, dtype=np.float64))


def _make_karate() -> tuple[eg.CsrGraph, eg.Layout2d]:
    karate_edges = [
        (1, 0), (1, 2), (1, 3), (1, 7), (1, 13), (1, 17), (1, 19), (1, 21), (1, 30),
        (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7), (0, 8), (0, 10), (0, 11),
        (0, 12), (0, 13), (0, 17), (0, 19), (0, 21), (0, 31), (2, 3), (2, 7), (2, 8),
        (2, 9), (2, 13), (2, 27), (2, 28), (2, 32), (3, 7), (3, 12), (3, 13), (4, 6),
        (4, 10), (5, 6), (5, 10), (5, 16), (6, 16), (8, 30), (8, 32), (8, 33), (9, 33),
        (13, 33), (19, 33), (25, 23), (25, 24), (25, 31), (23, 27), (23, 29), (23, 32),
        (23, 33), (24, 27), (24, 31), (27, 33), (28, 31), (28, 33), (29, 26), (29, 32),
        (29, 33), (26, 33), (30, 32), (30, 33), (31, 32), (31, 33), (32, 14), (32, 15),
        (32, 18), (32, 20), (32, 22), (32, 33), (14, 33), (15, 33), (18, 33), (20, 33),
        (22, 33),
    ]
    n = 34
    csr = eg.CsrGraph.from_edges(n, karate_edges, directed=False, sanitize=True)

    # Produce a structured stress-optimized embedding with SgdSolver.full.
    layout = eg.Layout2d.init(n, seed=7)
    eg.SgdSolver.full(csr, iterations=60, seed=7).run(layout)
    return csr, layout


def _scale_layout(layout: eg.Layout2d, factor: float) -> eg.Layout2d:
    arr = layout.coords(copy=True) * factor
    return eg.Layout2d.from_coords(arr)


def _to_f32_layout(layout: eg.Layout2d) -> eg.Layout2d:
    return eg.Layout2d.from_coords(layout.coords(copy=True).astype(np.float32))


class TestGeometryAndStressMetrics(unittest.TestCase):
    def assert_no_panic_raises(
        self,
        expected: tuple[type[BaseException], ...],
        fn: Any,
    ) -> None:
        with self.assertRaises(expected) as ctx:
            fn()
        self.assertNotIn("PanicException", type(ctx.exception).__name__)

    def test_sns_scale_invariance_and_zero_random_inversion(self) -> None:
        fixtures = [
            ("path", *_make_path(16)),
            ("cycle", *_make_cycle(20)),
            ("grid", *_make_grid(5, 5)),
            ("barbell", *_make_barbell(5, 3)),
            ("karate", *_make_karate()),
        ]
        inversions = 0
        total_comparisons = 0

        for name, g, layout in fixtures:
            s1 = eg.scale_normalized_stress(g, layout)
            s001 = eg.scale_normalized_stress(g, _scale_layout(layout, 0.01))
            s100 = eg.scale_normalized_stress(g, _scale_layout(layout, 100.0))
            self.assertAlmostEqual(s001, s1, places=11, msg=f"{name} 0.01x")
            self.assertAlmostEqual(s100, s1, places=11, msg=f"{name} 100x")

            # Verify float32 precision support.
            s_f32 = eg.scale_normalized_stress(g, _to_f32_layout(layout))
            self.assertAlmostEqual(s_f32, s1, places=5, msg=f"{name} f32")

            # Full pivot sample equals exact SNS, and sampled SNS is scale-invariant.
            s_sampled_all = eg.sampled_scale_normalized_stress(
                g, layout, num_pivots=g.num_nodes, seed=42
            )
            self.assertAlmostEqual(s_sampled_all, s1, places=11)
            s_sampled_a = eg.sampled_scale_normalized_stress(
                g, layout, num_pivots=8, seed=42
            )
            s_sampled_b = eg.sampled_scale_normalized_stress(
                g, _scale_layout(layout, 100.0), num_pivots=8, seed=42
            )
            self.assertAlmostEqual(s_sampled_a, s_sampled_b, places=11)

            rng = np.random.default_rng(100)
            for _ in range(10):
                base_rand = rng.uniform(-0.5, 0.5, size=(g.num_nodes, 2))
                for scale in (0.01, 1.0, 100.0):
                    r_layout = eg.Layout2d.from_coords(base_rand * scale)
                    r_sns = eg.scale_normalized_stress(g, r_layout)
                    total_comparisons += 1
                    if s1 >= r_sns:
                        inversions += 1
                    self.assertLess(s1, r_sns, f"{name} scale={scale} inverted")

        self.assertEqual(inversions, 0)
        self.assertEqual(total_comparisons, 5 * 10 * 3)

    def test_core_geometry_and_crossing_metrics_f64_and_f32(self) -> None:
        # Crossed segments at 90 degrees and K5 pentagram.
        g_cross = eg.CsrGraph.from_edges(4, [(0, 1), (2, 3)])
        coords_cross = np.array([[-1.0, 0.0], [1.0, 0.0], [0.0, -1.0], [0.0, 1.0]])

        k5_edges = [(u, v) for u in range(5) for v in range(u + 1, 5)]
        g_k5 = eg.CsrGraph.from_edges(5, k5_edges)
        k5_angles = np.linspace(0.0, 2.0 * math.pi, 5, endpoint=False)
        coords_k5 = np.column_stack([np.cos(k5_angles), np.sin(k5_angles)])

        # Torus crossing wrap fixture.
        coords_torus = np.array([[0.9, 0.1], [0.1, 0.9], [0.9, 0.9], [0.1, 0.1]])

        # 4x4 grid for NP1/NP2/S_RNG(drng)/S_GG(dgg)/Gabriel/node_resolution/aspect_ratio.
        grid_g, grid_l64 = _make_grid(4, 4)

        # Star graph for angular_resolution.
        g_star = eg.CsrGraph.from_edges(4, [(0, 1), (0, 2), (0, 3)])
        coords_star = np.array(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [-0.5, 0.5 * math.sqrt(3.0)],
                [-0.5, -0.5 * math.sqrt(3.0)],
            ]
        )

        # Weighted path for ideal_edge_lengths.
        g_weighted = eg.CsrGraph.from_edges(
            3, [(0, 1), (1, 2)], weights=[2.0, 3.0], sanitize=False
        )
        coords_w = np.array([[0.0, 0.0], [2.0, 0.0], [5.0, 0.0]])

        for np_dtype in (np.float64, np.float32):
            l_cross = eg.Layout2d.from_coords(coords_cross.astype(np_dtype))
            l_k5 = eg.Layout2d.from_coords(coords_k5.astype(np_dtype))
            l_torus = eg.Layout2d.from_coords(coords_torus.astype(np_dtype))
            l_grid = eg.Layout2d.from_coords(grid_l64.coords(copy=True).astype(np_dtype))
            l_star = eg.Layout2d.from_coords(coords_star.astype(np_dtype))
            l_w = eg.Layout2d.from_coords(coords_w.astype(np_dtype))

            self.assertEqual(eg.sweep_crossing_number(g_cross, l_cross), 1.0)
            self.assertAlmostEqual(eg.sweep_crossing_angle(g_cross, l_cross), 0.0, places=6)
            self.assertEqual(eg.sweep_crossing_number(g_k5, l_k5), 5.0)
            self.assertEqual(
                eg.sampled_crossing_number(g_k5, l_k5, num_samples=100, seed=42), 5.0
            )
            self.assertGreater(eg.torus_crossing_number(g_cross, l_torus), 0.0)

            self.assertAlmostEqual(
                eg.neighborhood_preservation_1hop(grid_g, l_grid), 1.0, places=6
            )
            self.assertAlmostEqual(
                eg.neighborhood_preservation_2hop(grid_g, l_grid), 1.0, places=6
            )
            self.assertAlmostEqual(eg.drng_preservation(grid_g, l_grid), 1.0, places=6)
            self.assertAlmostEqual(eg.dgg_preservation(grid_g, l_grid), 1.0, places=6)
            self.assertAlmostEqual(eg.gabriel_graph_property(grid_g, l_grid), 0.0, places=6)
            self.assertGreater(eg.node_resolution(l_grid), 0.0)
            self.assertAlmostEqual(eg.aspect_ratio(l_grid), 1.0, places=6)
            self.assertAlmostEqual(
                eg.angular_resolution(g_star, l_star),
                3.0 * math.exp(-2.0 * math.pi / 3.0),
                places=5,
            )
            self.assertAlmostEqual(eg.ideal_edge_lengths(g_weighted, l_w), 0.0, places=6)

        # Standalone edge_angle helper.
        ang = eg.edge_angle(1.0, 0.0, 0.0, 1.0)
        self.assertIsNotNone(ang)
        assert ang is not None
        self.assertAlmostEqual(ang, math.pi / 2.0, places=12)
        self.assertIsNone(eg.edge_angle(0.0, 0.0, 1.0, 0.0))
        self.assertIsNone(eg.edge_angle(float("nan"), 0.0, 1.0, 0.0))

    def test_prism_distortion_and_delaunay_triangulation(self) -> None:
        orig_pts = np.array(
            [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.5, 1.6]],
            dtype=np.float64,
        )
        theta = 0.7
        c, s = math.cos(theta), math.sin(theta)
        rot = np.array([[c, -s], [s, c]], dtype=np.float64)
        sim_pts = (orig_pts @ rot.T) * 3.5 + np.array([12.0, -4.0])

        orig64 = eg.Layout2d.from_coords(orig_pts)
        sim64 = eg.Layout2d.from_coords(sim_pts)
        sim32 = eg.Layout2d.from_coords(sim_pts.astype(np.float32))

        sig_dist, sig_disp = eg.prism_distortion(orig64, sim64)
        self.assertLess(sig_dist, 1e-12)
        self.assertLess(sig_disp, 1e-12)

        # Mixed float64 / float32 layouts work seamlessly.
        sig_dist_mix, sig_disp_mix = eg.prism_distortion(orig64, sim32)
        self.assertLess(sig_dist_mix, 1e-5)
        self.assertLess(sig_disp_mix, 1e-5)

        warped = eg.Layout2d.from_coords(
            np.array([[0.0, 0.0], [3.0, 0.0], [0.0, 1.0], [1.0, 2.5], [0.5, 2.0]])
        )
        w_dist, w_disp = eg.prism_distortion(orig64, warped)
        self.assertGreater(w_dist, 0.05)
        self.assertGreater(w_disp, 0.01)
        self.assertLessEqual(w_disp, 1.0)

        # Delaunay triangulation on square (f64 and f32).
        sq = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=np.float64)
        for dtype in (np.float64, np.float32):
            edges, tris = eg.delaunay_triangulation(eg.Layout2d.from_coords(sq.astype(dtype)))
            self.assertEqual(edges.dtype, np.uint32)
            self.assertEqual(tris.dtype, np.uint32)
            self.assertEqual(edges.shape, (5, 2))
            self.assertEqual(tris.shape, (2, 3))

        # Collinear points produce (3, 2) edges and (0, 3) triangles without panic.
        collinear = eg.Layout2d.from_coords(
            np.array([[0.0, 1.0], [1.0, 1.0], [2.0, 1.0], [3.0, 1.0]], dtype=np.float32)
        )
        c_edges, c_tris = eg.delaunay_triangulation(collinear)
        self.assertEqual(c_edges.shape, (3, 2))
        self.assertEqual(c_tris.shape, (0, 3))

        # Empty layout.
        e_edges, e_tris = eg.delaunay_triangulation(eg.Layout2d.zeros(0))
        self.assertEqual(e_edges.shape, (0, 2))
        self.assertEqual(e_tris.shape, (0, 3))

    def test_sampled_np1_and_np2(self) -> None:
        g, layout64 = _make_grid(4, 4)
        layout32 = eg.Layout2d.from_coords(layout64.coords(copy=True).astype(np.float32))

        for fn_sampled, fn_exact in (
            (eg.sampled_np1, eg.neighborhood_preservation_1hop),
            (eg.sampled_np2, eg.neighborhood_preservation_2hop),
        ):
            self.assertIn(fn_sampled.__name__, eg.__all__)
            self.assertTrue((fn_sampled.__doc__ or "").strip())
            sig = inspect.signature(fn_sampled)
            self.assertEqual(sig.parameters["num_samples"].default, 4096)
            self.assertEqual(sig.parameters["seed"].default, 42)

            exact = fn_exact(g, layout64)
            full_s = fn_sampled(g, layout64, num_samples=g.num_nodes, seed=7)
            self.assertEqual(full_s, exact)

            sub_a = fn_sampled(g, layout64, num_samples=8, seed=19)
            sub_b = fn_sampled(g, layout64, num_samples=8, seed=19)
            self.assertEqual(sub_a, sub_b)
            self.assertGreaterEqual(sub_a, 0.0)
            self.assertLessEqual(sub_a, 1.0)

            sub_32 = fn_sampled(g, layout32, num_samples=g.num_nodes, seed=7)
            self.assertAlmostEqual(sub_32, fn_exact(g, layout32), places=12)

            self.assert_no_panic_raises(
                (ValueError,), lambda: fn_sampled(g, layout64, num_samples=0)
            )
            self.assert_no_panic_raises(
                (ValueError,), lambda: fn_sampled(g, eg.Layout2d.zeros(4), num_samples=4)
            )
            bad_nan = eg.Layout2d.zeros(g.num_nodes)
            bad_nan.coords(copy=False)[0, 0] = float("nan")
            self.assert_no_panic_raises(
                (ValueError,), lambda: fn_sampled(g, bad_nan, num_samples=4)
            )

    def test_top_level_csr_metrics_and_crossing_pairs(self) -> None:
        edges = [(0, 1), (1, 2), (2, 3), (3, 0), (0, 2), (1, 3)]
        coords = np.array([[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0]], dtype=np.float64)

        csr = eg.CsrGraph.from_edges(4, edges)
        layout64 = eg.Layout2d.from_coords(coords)
        layout32 = eg.Layout2d.from_coords(coords.astype(np.float32))

        for layout in (layout64, layout32):
            pairs = eg.crossing_edges(csr, layout)
            self.assertEqual(eg.crossing_number(csr, layout), 1.0)
            self.assertEqual(eg.crossing_number_from_pairs(pairs), eg.crossing_number(csr, layout))
            self.assertAlmostEqual(
                eg.crossing_angle_from_pairs(pairs), eg.crossing_angle(csr, layout), places=12
            )
            self.assertAlmostEqual(eg.crossing_angle(csr, layout), 0.0, places=12)
            self.assertEqual(
                eg.crossing_number(csr, layout), eg.sweep_crossing_number(csr, layout)
            )
            self.assertAlmostEqual(
                eg.crossing_angle(csr, layout), eg.sweep_crossing_angle(csr, layout), places=12
            )
            self.assertAlmostEqual(eg.neighborhood_preservation_1hop(csr, layout), 1.0, places=6)
            self.assertAlmostEqual(eg.neighborhood_preservation_2hop(csr, layout), 1.0, places=6)
            self.assertAlmostEqual(eg.aspect_ratio(layout), 1.0, places=6)
            self.assertTrue(math.isfinite(eg.angular_resolution(csr, layout)))
            self.assertEqual(eg.node_resolution(layout), 0.0)
            self.assertTrue(math.isfinite(eg.gabriel_graph_property(csr, layout)))
            self.assertGreater(eg.ideal_edge_lengths(csr, layout), 0.0)

        # Non-perpendicular crossing verifies sum cos^2(theta) > 0.0
        g_shear = eg.CsrGraph.from_edges(4, [(0, 1), (2, 3)])
        l_shear = eg.Layout2d.from_coords([[0.0, 0.0], [2.0, 2.0], [0.0, 1.0], [2.0, 1.0]])
        pairs_shear = eg.crossing_edges(g_shear, l_shear)
        self.assertEqual(eg.crossing_number_from_pairs(pairs_shear), 1.0)
        self.assertAlmostEqual(
            eg.crossing_angle_from_pairs(pairs_shear), eg.crossing_angle(g_shear, l_shear), places=12
        )
        self.assertAlmostEqual(eg.crossing_angle(g_shear, l_shear), 0.5, places=12)


if __name__ == "__main__":
    unittest.main()
