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

    # Produce a structured stress-optimized embedding with legacy FullSgd.
    lg = eg.Graph()
    nodes = [lg.add_node(i) for i in range(n)]
    for u, v in karate_edges:
        lg.add_edge(nodes[u], nodes[v], 1.0)
    drawing = eg.DrawingEuclidean2d.initial_placement(lg)
    sgd = eg.FullSgd().build(lg, 1.0)
    rng = eg.Rng.seed_from(7)

    def _step(eta: float) -> None:
        sgd.shuffle(rng)
        sgd.apply(drawing, eta)

    eg.SchedulerExponential(60).run(_step)
    return csr, eg.Layout2d.from_coords(drawing.to_numpy())


def _scale_layout(layout: eg.Layout2d, factor: float) -> eg.Layout2d:
    arr = layout.coords(copy=True) * factor
    return eg.Layout2d.from_coords(arr)


def _to_f32_layout(layout: eg.Layout2d) -> eg.Layout2d:
    return eg.Layout2d.from_coords(layout.coords(copy=True).astype(np.float32))


class TestV6MetricsAndSpatial(unittest.TestCase):
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

    def test_all_m1_metrics_f64_and_f32(self) -> None:
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

        # Weighted path for ideal_edge_lengths_v6.
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
                eg.neighborhood_preservation_2hop_v6(grid_g, l_grid), 1.0, places=6
            )
            self.assertAlmostEqual(eg.drng_preservation(grid_g, l_grid), 1.0, places=6)
            self.assertAlmostEqual(eg.dgg_preservation(grid_g, l_grid), 1.0, places=6)
            self.assertAlmostEqual(eg.gabriel_graph_property_v6(grid_g, l_grid), 0.0, places=6)
            self.assertGreater(eg.node_resolution_v6(l_grid), 0.0)
            self.assertAlmostEqual(eg.aspect_ratio_v6(l_grid), 1.0, places=6)
            self.assertAlmostEqual(
                eg.angular_resolution_v6(g_star, l_star),
                3.0 * math.exp(-2.0 * math.pi / 3.0),
                places=5,
            )
            self.assertAlmostEqual(eg.ideal_edge_lengths_v6(g_weighted, l_w), 0.0, places=6)

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

    def test_selective_evaluate_metrics(self) -> None:
        g, layout = _make_grid(4, 4)

        self.assertEqual(eg.evaluate_metrics(g, layout, []), {})

        subset = eg.evaluate_metrics(g, layout, ["sns", "np2", "drng", "dgg", "np2"])
        self.assertEqual(
            set(subset.keys()),
            {
                "scale_normalized_stress",
                "neighborhood_preservation_2hop",
                "relative_neighborhood_preservation",
                "gabriel_graph_preservation",
            },
        )
        self.assertAlmostEqual(
            subset["scale_normalized_stress"],
            eg.scale_normalized_stress(g, layout),
            places=12,
        )
        self.assertAlmostEqual(
            subset["neighborhood_preservation_2hop"],
            eg.neighborhood_preservation_2hop_v6(g, layout),
            places=12,
        )
        self.assertAlmostEqual(
            subset["relative_neighborhood_preservation"],
            eg.drng_preservation(g, layout),
            places=12,
        )
        self.assertAlmostEqual(
            subset["gabriel_graph_preservation"],
            eg.dgg_preservation(g, layout),
            places=12,
        )

        # Evaluate all 16 targets using mixed alias conventions.
        all_targets = [
            "ScaleNormalizedStress",
            "sampled_sns",
            "crossing-number",
            "sweep_crossing_angle",
            "sampled_crossing_number",
            "np1",
            "neighborhood_preservation_2hop",
            "sampled-np1",
            "sampled_np2",
            "gabriel_graph_property",
            "node_resolution",
            "drng_preservation",
            "dgg_preservation",
            "aspect_ratio",
            "angular_resolution",
            "ideal_edge_lengths",
        ]
        full = eg.evaluate_metrics(g, layout, all_targets, num_pivots=6, num_samples=200, seed=9)
        self.assertEqual(len(full), 16)
        self.assertAlmostEqual(
            full["sampled_scale_normalized_stress"],
            eg.sampled_scale_normalized_stress(g, layout, num_pivots=6, seed=9),
            places=12,
        )
        self.assertEqual(full["crossing_number"], eg.sweep_crossing_number(g, layout))
        self.assertAlmostEqual(full["crossing_angle"], eg.sweep_crossing_angle(g, layout), places=12)
        self.assertAlmostEqual(
            full["sampled_crossing_number"],
            eg.sampled_crossing_number(g, layout, num_samples=200, seed=9),
            places=12,
        )
        self.assertAlmostEqual(
            full["neighborhood_preservation_1hop"],
            eg.neighborhood_preservation_1hop(g, layout),
            places=12,
        )
        self.assertAlmostEqual(
            full["sampled_np1"],
            eg.sampled_np1(g, layout, num_samples=200, seed=9),
            places=12,
        )
        self.assertAlmostEqual(
            full["sampled_np2"],
            eg.sampled_np2(g, layout, num_samples=200, seed=9),
            places=12,
        )
        self.assertAlmostEqual(
            full["gabriel_graph_property"],
            eg.gabriel_graph_property_v6(g, layout),
            places=12,
        )
        self.assertAlmostEqual(full["node_resolution"], eg.node_resolution_v6(layout), places=12)
        self.assertAlmostEqual(full["aspect_ratio"], eg.aspect_ratio_v6(layout), places=12)
        self.assertAlmostEqual(
            full["angular_resolution"], eg.angular_resolution_v6(g, layout), places=12
        )
        self.assertAlmostEqual(
            full["ideal_edge_lengths"], eg.ideal_edge_lengths_v6(g, layout), places=12
        )

        # Invalid sampled parameter only errors when the corresponding target is requested.
        ok = eg.evaluate_metrics(g, layout, ["aspect_ratio"], num_pivots=0)
        self.assertIn("aspect_ratio", ok)
        self.assert_no_panic_raises(
            (ValueError,),
            lambda: eg.evaluate_metrics(g, layout, ["sampled_sns"], num_pivots=0),
        )
        self.assert_no_panic_raises(
            (ValueError,),
            lambda: eg.evaluate_metrics(g, layout, ["sampled_np1"], num_samples=0),
        )
        self.assert_no_panic_raises(
            (ValueError,),
            lambda: eg.evaluate_metrics(g, layout, ["sampled_np2"], num_samples=0),
        )

    def test_sampled_np1_and_np2(self) -> None:
        g, layout64 = _make_grid(4, 4)
        layout32 = eg.Layout2d.from_coords(layout64.coords(copy=True).astype(np.float32))

        for fn_sampled, fn_exact in (
            (eg.sampled_np1, eg.neighborhood_preservation_1hop),
            (eg.sampled_np2, eg.neighborhood_preservation_2hop_v6),
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

    def test_top_level_dual_dispatch_v6_and_legacy(self) -> None:
        # Build matching graphs/layouts in both legacy and v6 representations.
        edges = [(0, 1), (1, 2), (2, 3), (3, 0), (0, 2), (1, 3)]
        coords = np.array([[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0]], dtype=np.float64)

        csr = eg.CsrGraph.from_edges(4, edges)
        layout64 = eg.Layout2d.from_coords(coords)
        layout32 = eg.Layout2d.from_coords(coords.astype(np.float32))

        lg = eg.Graph()
        nodes = [lg.add_node(i) for i in range(4)]
        for u, v in edges:
            lg.add_edge(nodes[u], nodes[v], 1.0)
        drawing = eg.DrawingEuclidean2d.from_numpy(lg, coords)
        dm = eg.DistanceMatrix.from_numpy(lg, np.ones((4, 4), dtype=np.float64))

        for layout in (layout64, layout32):
            self.assertEqual(eg.crossing_number(csr, layout), eg.crossing_number(lg, drawing))
            self.assertAlmostEqual(
                eg.crossing_angle(csr, layout), eg.crossing_angle(lg, drawing), places=5
            )
            self.assertAlmostEqual(
                eg.neighborhood_preservation(csr, layout),
                eg.neighborhood_preservation(lg, drawing),
                places=6,
            )
            self.assertAlmostEqual(
                eg.neighborhood_preservation_2hop(csr, layout),
                eg.neighborhood_preservation_2hop(lg, drawing),
                places=6,
            )
            self.assertAlmostEqual(eg.aspect_ratio(layout), eg.aspect_ratio(drawing), places=5)
            self.assertAlmostEqual(
                eg.angular_resolution(csr, layout),
                eg.angular_resolution(lg, drawing),
                places=5,
            )
            self.assertAlmostEqual(
                eg.node_resolution(layout), eg.node_resolution_v6(layout), places=12
            )
            self.assertEqual(eg.node_resolution(layout), 0.0)
            self.assertTrue(math.isfinite(eg.node_resolution(drawing)))
            self.assertAlmostEqual(
                eg.gabriel_graph_property(csr, layout),
                eg.gabriel_graph_property(lg, drawing),
                places=5,
            )
            self.assertAlmostEqual(
                eg.ideal_edge_lengths(csr, layout),
                eg.ideal_edge_lengths(lg, drawing, dm),
                places=5,
            )

    def test_adversarial_errors_and_signatures(self) -> None:
        g, layout = _make_path(4)
        short_layout = eg.Layout2d.zeros(3)

        # NaN injected via zero-copy NumPy view must raise ValueError cleanly.
        nan_layout = eg.Layout2d.from_coords(layout.coords(copy=True))
        nan_layout.coords(copy=False)[1, 0] = float("nan")

        for fn in (
            eg.scale_normalized_stress,
            eg.sampled_scale_normalized_stress,
            eg.sweep_crossing_number,
            eg.sweep_crossing_angle,
            eg.sampled_crossing_number,
            eg.torus_crossing_number,
            eg.neighborhood_preservation_1hop,
            eg.neighborhood_preservation_2hop_v6,
            eg.drng_preservation,
            eg.dgg_preservation,
            eg.gabriel_graph_property_v6,
            eg.angular_resolution_v6,
            eg.ideal_edge_lengths_v6,
            eg.crossing_number,
            eg.crossing_angle,
            eg.neighborhood_preservation,
            eg.neighborhood_preservation_2hop,
            eg.gabriel_graph_property,
            eg.angular_resolution,
            eg.ideal_edge_lengths,
        ):
            self.assert_no_panic_raises((ValueError,), lambda f=fn: f(g, nan_layout))
            self.assert_no_panic_raises((ValueError,), lambda f=fn: f(g, short_layout))

        for ufn in (
            eg.node_resolution_v6,
            eg.aspect_ratio_v6,
            eg.node_resolution,
            eg.aspect_ratio,
            eg.delaunay_triangulation,
        ):
            self.assert_no_panic_raises((ValueError,), lambda f=ufn: f(nan_layout))

        self.assert_no_panic_raises(
            (ValueError,), lambda: eg.prism_distortion(layout, nan_layout)
        )
        self.assert_no_panic_raises(
            (ValueError,), lambda: eg.prism_distortion(layout, short_layout)
        )
        self.assert_no_panic_raises(
            (ValueError,), lambda: eg.evaluate_metrics(g, nan_layout, ["sns"])
        )
        self.assert_no_panic_raises(
            (ValueError,), lambda: eg.evaluate_metrics(g, short_layout, ["sns"])
        )

        # Invalid target inputs.
        self.assert_no_panic_raises((TypeError,), lambda: eg.evaluate_metrics(g, layout, "sns"))
        self.assert_no_panic_raises(
            (TypeError,), lambda: eg.evaluate_metrics(g, layout, [123])  # type: ignore[list-item]
        )
        self.assert_no_panic_raises(
            (ValueError,), lambda: eg.evaluate_metrics(g, layout, ["unknown_metric"])
        )
        self.assert_no_panic_raises(
            (ValueError,), lambda: eg.evaluate_metrics(g, layout, [""])
        )

        # Invalid sampling parameters.
        self.assert_no_panic_raises(
            (ValueError,), lambda: eg.sampled_scale_normalized_stress(g, layout, num_pivots=0)
        )
        self.assert_no_panic_raises(
            (ValueError,), lambda: eg.sampled_scale_normalized_stress(g, layout, num_pivots=-1)
        )
        self.assert_no_panic_raises(
            (TypeError,), lambda: eg.sampled_scale_normalized_stress(g, layout, num_pivots=True)
        )
        self.assert_no_panic_raises(
            (ValueError,), lambda: eg.sampled_scale_normalized_stress(g, layout, seed=-1)
        )
        self.assert_no_panic_raises(
            (TypeError,), lambda: eg.sampled_scale_normalized_stress(g, layout, seed=True)
        )
        self.assert_no_panic_raises(
            (ValueError,), lambda: eg.sampled_crossing_number(g, layout, num_samples=0)
        )
        self.assert_no_panic_raises(
            (ValueError,), lambda: eg.sampled_crossing_number(g, layout, num_samples=-5)
        )

        # Type mismatches between legacy and v6 objects.
        lg = eg.Graph()
        for i in range(4):
            lg.add_node(i)
        ld = eg.DrawingEuclidean2d.initial_placement(lg)
        dm = eg.DistanceMatrix.from_numpy(lg, np.ones((4, 4), dtype=np.float64))

        self.assert_no_panic_raises((TypeError,), lambda: eg.crossing_number(g, ld))  # type: ignore[call-overload]
        self.assert_no_panic_raises((TypeError,), lambda: eg.crossing_number(lg, layout))  # type: ignore[call-overload]
        self.assert_no_panic_raises((TypeError,), lambda: eg.ideal_edge_lengths(lg, ld))  # type: ignore[call-overload]
        self.assert_no_panic_raises((TypeError,), lambda: eg.ideal_edge_lengths(g, layout, dm))  # type: ignore[call-overload]
        self.assert_no_panic_raises((TypeError,), lambda: eg.aspect_ratio("bad"))  # type: ignore[call-overload]
        self.assert_no_panic_raises((TypeError,), lambda: eg.node_resolution("bad"))  # type: ignore[call-overload]
        self.assert_no_panic_raises((TypeError,), lambda: eg.edge_angle(True, 0.0, 1.0, 0.0))

        # Check inspect.signature defaults and non-empty docstrings.
        sig_sns = inspect.signature(eg.sampled_scale_normalized_stress)
        self.assertEqual(sig_sns.parameters["num_pivots"].default, 64)
        self.assertEqual(sig_sns.parameters["seed"].default, 42)
        sig_cr = inspect.signature(eg.sampled_crossing_number)
        self.assertEqual(sig_cr.parameters["num_samples"].default, 4096)
        self.assertEqual(sig_cr.parameters["seed"].default, 42)
        sig_ev = inspect.signature(eg.evaluate_metrics)
        self.assertEqual(sig_ev.parameters["num_pivots"].default, 64)
        self.assertEqual(sig_ev.parameters["num_samples"].default, 4096)
        self.assertEqual(sig_ev.parameters["seed"].default, 42)

        for name in (
            "scale_normalized_stress",
            "sampled_scale_normalized_stress",
            "sweep_crossing_number",
            "sweep_crossing_angle",
            "sampled_crossing_number",
            "crosslessness",
            "sampled_crosslessness",
            "torus_crossing_number",
            "neighborhood_preservation_1hop",
            "neighborhood_preservation_2hop_v6",
            "edge_angle",
            "prism_distortion",
            "drng_preservation",
            "dgg_preservation",
            "evaluate_metrics",
            "delaunay_triangulation",
            "gabriel_graph_property_v6",
            "node_resolution_v6",
            "aspect_ratio_v6",
            "angular_resolution_v6",
            "minimum_angle",
            "minimum_angle_summary",
            "layout_degeneracy",
            "adjusted_patch_nig",
            "adjusted_patch_nig_multiscale",
            "cluster_area_faithfulness",
            "cluster_area_faithfulness_detail",
            "landmark_stratified_rta",
            "landmark_stratified_rta_detail",
            "landmark_stratified_rta_from_landmarks",
            "ideal_edge_lengths_v6",
        ):
            fn_obj = getattr(eg, name)
            self.assertTrue((fn_obj.__doc__ or "").strip(), f"missing docstring on {name}")

    def test_w3_new_metrics_and_selective_dispatcher(self) -> None:
        # K4 square with crossing diagonals: c = 1, c_max = 3.
        k4 = eg.CsrGraph.from_edges(
            4, [(0, 1), (1, 2), (2, 3), (3, 0), (0, 2), (1, 3)], directed=False, sanitize=True
        )
        sq = eg.Layout2d.from_coords([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
        expected_cl = 1.0 - (1.0 / 3.0) ** 0.5
        self.assertAlmostEqual(eg.crosslessness(k4, sq), expected_cl, places=12)
        self.assertAlmostEqual(
            eg.sampled_crosslessness(k4, sq, num_samples=100, seed=42),
            expected_cl,
            places=12,
        )

        # Minimum angle A1/A2 with zero-length edge divergence.
        star = eg.CsrGraph.from_edges(4, [(0, 1), (0, 2), (0, 3)], directed=False, sanitize=True)
        zero_l = eg.Layout2d.from_coords([[0.0, 0.0], [0.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
        ma = eg.minimum_angle_summary(star, zero_l)
        self.assertGreater(ma["a1"], ma["a2"])
        self.assertAlmostEqual(eg.minimum_angle(star, zero_l), ma["a2"], places=12)

        # Layout degeneracy: -0.0 canonicalization, f32 vs f64 duplicates, zero-length edge.
        l_deg = eg.Layout2d.from_coords(
            [[0.0, -0.0], [-0.0, 0.0], [1.0, 1.0], [1.0 + 1e-12, 1.0]]
        )
        deg = eg.layout_degeneracy(star, l_deg)
        self.assertEqual(deg["num_nodes"], 4)
        self.assertEqual(deg["num_edges"], 3)
        self.assertEqual(deg["duplicate_nodes_f32"], 4)
        self.assertEqual(deg["duplicate_nodes_f64"], 2)
        self.assertEqual(deg["zero_length_edges"], 1)
        self.assertFalse(deg["is_bbox_degenerate"])

        # Adjusted Patch-NIG single-scale & multiscale.
        coords_c = []
        labels_c = []
        for c in range(4):
            cx = float(c * 10)
            for k in range(10):
                coords_c.append([cx + 0.1 * k, 0.05 * (k % 3)])
                labels_c.append(c)
        l_comm = eg.Layout2d.from_coords(coords_c)
        nig = eg.adjusted_patch_nig(l_comm, labels_c, patch_radius=16, canvas_size=2048, seed=42)
        self.assertGreater(nig, 0.9)
        nig_ms = eg.adjusted_patch_nig_multiscale(
            l_comm, labels_c, radii=[8, 16], canvas_size=2048, seed=42
        )
        self.assertAlmostEqual(nig_ms["patch_nig_r16"], nig, places=12)

        # Landmark-Stratified RTA on a 1D path + precomputed landmark table parity.
        path = eg.CsrGraph.from_edges(
            8, [(i, i + 1) for i in range(7)], directed=False, sanitize=True
        )
        path_l = eg.Layout2d.from_coords([[float(i), 0.0] for i in range(8)])
        rta_det = eg.landmark_stratified_rta_detail(
            path, path_l, num_landmarks=4, num_triplets=10000, min_degree=1, seed=42
        )
        self.assertTrue(rta_det["is_exact"])
        self.assertAlmostEqual(rta_det["rta"], 1.0, places=12)
        self.assertAlmostEqual(
            eg.landmark_stratified_rta(
                path, path_l, num_landmarks=4, num_triplets=10000, min_degree=1, seed=42
            ),
            1.0,
            places=12,
        )
        lm_ids = eg.select_maxmin_sp_pivots(path, num_pivots=4, seed=42, min_degree=1)
        apsp = eg.all_pairs_bfs(path, 1.0)
        tbl = np.where(np.isfinite(apsp[lm_ids]), apsp[lm_ids], np.iinfo(np.uint32).max).astype(
            np.uint32
        )
        rta_tbl = eg.landmark_stratified_rta_from_landmarks(
            path_l, lm_ids, tbl, num_triplets=10000, seed=42
        )
        self.assertEqual(rta_tbl, rta_det)

        # Selective evaluate_metrics with new targets + community_labels error guard.
        with self.assertRaises(ValueError):
            eg.evaluate_metrics(k4, sq, ["adjusted_patch_nig"])

        rep = eg.evaluate_metrics(
            k4,
            sq,
            [
                "crosslessness",
                "sampled_crosslessness",
                "minimum_angle",
                "minimum_angle_a1",
                "layout_degeneracy",
                "adjusted_patch_nig",
                "landmark_stratified_rta",
            ],
            community_labels=[0, 0, 1, 1],
            num_landmarks=4,
            min_degree=1,
        )
        self.assertAlmostEqual(rep["crosslessness"], expected_cl, places=12)
        self.assertAlmostEqual(rep["minimum_angle"], eg.minimum_angle(k4, sq), places=12)
        self.assertAlmostEqual(rep["minimum_angle_a1"], eg.minimum_angle_summary(k4, sq)["a1"], places=12)
        self.assertEqual(rep["layout_degeneracy"], 0.0)
        self.assertAlmostEqual(
            rep["adjusted_patch_nig"],
            eg.adjusted_patch_nig(sq, [0, 0, 1, 1], patch_radius=8, canvas_size=2048, seed=42),
            places=12,
        )
        self.assertGreaterEqual(rep["landmark_stratified_rta"], 0.0)

        g_comm = eg.CsrGraph.from_edges(
            40, [(i, i + 1) for i in range(39)], directed=False, sanitize=True
        )
        rep_comm = eg.evaluate_metrics(
            g_comm,
            l_comm,
            ["adjusted_patch_nig"],
            community_labels=labels_c,
            patch_radius=16,
        )
        self.assertAlmostEqual(rep_comm["adjusted_patch_nig"], nig, places=12)

    def test_crosslessness_directed_reciprocal_regression(self) -> None:
        # 1. Single reciprocal pair: c_max == 0 -> crosslessness == 1.0.
        g_pair = eg.CsrGraph.from_edges(2, [(0, 1), (1, 0)], directed=True, sanitize=True)
        l_pair = eg.Layout2d.from_coords([[0.0, 0.0], [1.0, 0.0]])
        self.assertEqual(eg.crosslessness(g_pair, l_pair), 1.0)
        self.assertEqual(eg.sampled_crosslessness(g_pair, l_pair, num_samples=32, seed=42), 1.0)

        # 2. Reciprocal 3-leaf star: adj_pairs > total_pairs, yet c_max == 0 -> crosslessness == 1.0.
        g_star = eg.CsrGraph.from_edges(
            4,
            [(0, 1), (1, 0), (0, 2), (2, 0), (0, 3), (3, 0)],
            directed=True,
            sanitize=True,
        )
        l_star = eg.Layout2d.from_coords([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
        self.assertEqual(eg.crosslessness(g_star, l_star), 1.0)
        self.assertEqual(eg.sampled_crosslessness(g_star, l_star, num_samples=64, seed=42), 1.0)

        # 3. 4-node 6-edge directed graph: adj_pairs (16) > total_pairs (15), c_max = 1, c = 1 -> 0.0.
        g_dir4 = eg.CsrGraph.from_edges(
            4,
            [(0, 1), (0, 3), (1, 3), (3, 0), (3, 1), (3, 2)],
            directed=True,
            sanitize=True,
        )
        l_dir4 = eg.Layout2d.from_coords([[0.0, 0.0], [2.0, 2.0], [0.0, 2.0], [2.0, 0.0]])
        self.assertEqual(eg.sweep_crossing_number(g_dir4, l_dir4), 1.0)
        self.assertEqual(eg.crosslessness(g_dir4, l_dir4), 0.0)
        self.assertEqual(eg.sampled_crosslessness(g_dir4, l_dir4, num_samples=100, seed=42), 0.0)
        rep_dir4 = eg.evaluate_metrics(
            g_dir4,
            l_dir4,
            ["crossing_number", "crosslessness", "sampled_crosslessness"],
            num_samples=100,
            seed=42,
        )
        self.assertEqual(rep_dir4["crossing_number"], 1.0)
        self.assertEqual(rep_dir4["crosslessness"], 0.0)
        self.assertEqual(rep_dir4["sampled_crosslessness"], 0.0)

    def test_cluster_area_faithfulness_and_detail(self) -> None:
        self.assertIn("cluster_area_faithfulness", eg.__all__)
        self.assertIn("cluster_area_faithfulness_detail", eg.__all__)

        # Check inspect.signature defaults across standalone and evaluate_metrics APIs.
        sig_caf = inspect.signature(eg.cluster_area_faithfulness)
        self.assertEqual(sig_caf.parameters["g_max"].default, 128)
        self.assertEqual(sig_caf.parameters["sigma_cells"].default, 0.85)
        self.assertEqual(sig_caf.parameters["kappa"].default, 10.0)
        self.assertEqual(sig_caf.parameters["tau_frac"].default, 0.18)
        self.assertEqual(sig_caf.parameters["boundary_r95"].default, 2.0)
        self.assertEqual(sig_caf.parameters["stevens_gamma"].default, 1.0)

        sig_det = inspect.signature(eg.cluster_area_faithfulness_detail)
        self.assertEqual(sig_det.parameters["g_max"].default, 128)
        self.assertEqual(sig_det.parameters["return_per_cluster"].default, False)

        sig_ev = inspect.signature(eg.evaluate_metrics)
        self.assertEqual(sig_ev.parameters["g_max"].default, 128)
        self.assertEqual(sig_ev.parameters["sigma_cells"].default, 0.85)
        self.assertEqual(sig_ev.parameters["kappa"].default, 10.0)
        self.assertEqual(sig_ev.parameters["tau_frac"].default, 0.18)
        self.assertEqual(sig_ev.parameters["boundary_r95"].default, 2.0)
        self.assertEqual(sig_ev.parameters["stevens_gamma"].default, 1.0)

        # Separated two-cluster fixture (100 nodes) in f64 and f32 + multiple label dtypes.
        rng = np.random.default_rng(7)
        coords = np.array(
            [[float(c) + (0.0 if c < 5 else 2.0), float(r)] for r in range(10) for c in range(10)],
            dtype=np.float64,
        )
        labels_list = [0 if c < 5 else 1 for _r in range(10) for c in range(10)]
        layout64 = eg.Layout2d.from_coords(coords)
        layout32 = eg.Layout2d.from_coords(coords.astype(np.float32))

        caf64 = eg.cluster_area_faithfulness(layout64, labels_list)
        self.assertGreater(caf64, 0.7)
        self.assertLessEqual(caf64, 1.0)

        caf32 = eg.cluster_area_faithfulness(layout32, labels_list)
        self.assertAlmostEqual(caf32, caf64, places=5)

        int_dtypes: tuple[type[np.integer[Any]], ...] = (np.uint32, np.int64, np.int32, np.uint8)
        for dtype in int_dtypes:
            arr = np.array(labels_list, dtype=dtype)
            self.assertAlmostEqual(eg.cluster_area_faithfulness(layout64, arr), caf64, places=12)

        # Detail without and with per-cluster vectors.
        base_keys = {
            "caf",
            "purity_adj",
            "prop_vis",
            "res_density",
            "res_deconv",
            "res_combined",
            "spatial_coherence",
            "in_frac",
            "effective_boundary_r95",
            "g_eff",
            "num_clusters",
            "num_nodes",
        }
        per_cluster_keys = {
            "cluster_ids",
            "counts",
            "p",
            "q",
            "purity",
            "resolution",
            "res_density_per_cluster",
            "res_deconv_per_cluster",
            "caf_per_cluster",
            "coherence",
        }
        det_summary = eg.cluster_area_faithfulness_detail(
            layout64, labels_list, return_per_cluster=False
        )
        self.assertEqual(set(det_summary.keys()), base_keys)
        self.assertEqual(det_summary["caf"], caf64)
        self.assertEqual(det_summary["num_clusters"], 2)
        self.assertEqual(det_summary["num_nodes"], 100)
        self.assertEqual(det_summary["effective_boundary_r95"], 2.0)
        self.assertEqual(det_summary["g_eff"], 12)
        for key in (
            "caf",
            "purity_adj",
            "prop_vis",
            "res_density",
            "res_deconv",
            "res_combined",
            "spatial_coherence",
            "in_frac",
        ):
            val = float(det_summary[key])
            self.assertGreaterEqual(val, 0.0)
            self.assertLessEqual(val, 1.0)

        det_full = eg.cluster_area_faithfulness_detail(
            layout64, labels_list, return_per_cluster=True
        )
        self.assertEqual(set(det_full.keys()), base_keys | per_cluster_keys)
        self.assertEqual(det_full["caf"], caf64)
        for key in per_cluster_keys:
            vec = det_full[key]
            self.assertIsInstance(vec, np.ndarray)
            self.assertEqual(vec.shape, (2,))
        np.testing.assert_array_equal(det_full["cluster_ids"], np.array([0, 1], dtype=np.uint32))
        np.testing.assert_array_equal(det_full["counts"], np.array([50, 50], dtype=np.uintp))
        self.assertAlmostEqual(float(np.sum(det_full["p"])), 1.0, places=12)
        self.assertAlmostEqual(float(np.sum(det_full["q"])), 1.0, places=12)

        # Exact grid-isometry invariances: translation + uniform scaling, 90-deg rotation, reflection.
        coords_scaled_shifted = coords * 7.5 + np.array([150.0, -42.0])
        caf_scaled_shifted = eg.cluster_area_faithfulness(
            eg.Layout2d.from_coords(coords_scaled_shifted), labels_list
        )
        self.assertAlmostEqual(caf_scaled_shifted, caf64, places=12)

        coords_rot90 = np.column_stack([-coords[:, 1], coords[:, 0]])
        caf_rot90 = eg.cluster_area_faithfulness(
            eg.Layout2d.from_coords(coords_rot90), labels_list
        )
        self.assertAlmostEqual(caf_rot90, caf64, places=12)

        # Oblique rotation preserves G-CAF up to raster grid discretization (~1e-4).
        theta = 0.65
        rot = np.array(
            [[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]],
            dtype=np.float64,
        )
        coords_sim = (coords @ rot.T) * 7.5 + np.array([150.0, -42.0])
        caf_sim = eg.cluster_area_faithfulness(eg.Layout2d.from_coords(coords_sim), labels_list)
        self.assertAlmostEqual(caf_sim, caf64, places=3)

        coords_refl = coords.copy()
        coords_refl[:, 1] = -coords_refl[:, 1]
        caf_refl = eg.cluster_area_faithfulness(eg.Layout2d.from_coords(coords_refl), labels_list)
        self.assertAlmostEqual(caf_refl, caf64, places=12)

        perm = rng.permutation(len(labels_list))
        coords_perm = coords[perm]
        labels_perm = [labels_list[i] for i in perm]
        caf_perm = eg.cluster_area_faithfulness(eg.Layout2d.from_coords(coords_perm), labels_perm)
        self.assertAlmostEqual(caf_perm, caf64, places=12)

        relabeled = [100 if lbl == 0 else 200 for lbl in labels_list]
        det_relabeled = eg.cluster_area_faithfulness_detail(
            layout64, relabeled, return_per_cluster=True
        )
        self.assertAlmostEqual(float(det_relabeled["caf"]), caf64, places=12)
        np.testing.assert_array_equal(
            det_relabeled["cluster_ids"], np.array([100, 200], dtype=np.uint32)
        )

    def test_cluster_area_faithfulness_boundaries_and_evaluate_metrics(self) -> None:
        # Degenerate and boundary layouts: N=0, N=1, K=1, collapsed coordinates.
        empty_det = eg.cluster_area_faithfulness_detail(
            eg.Layout2d.zeros(0), [], return_per_cluster=True
        )
        self.assertEqual(empty_det["caf"], 0.0)
        self.assertEqual(empty_det["effective_boundary_r95"], 0.0)
        self.assertEqual(empty_det["g_eff"], 0)
        self.assertEqual(empty_det["num_clusters"], 0)
        self.assertEqual(empty_det["num_nodes"], 0)
        self.assertEqual(len(empty_det["cluster_ids"]), 0)

        one_det = eg.cluster_area_faithfulness_detail(
            eg.Layout2d.from_coords([[1.0, 2.0]]), [5], return_per_cluster=True
        )
        self.assertEqual(one_det["caf"], 0.0)
        self.assertEqual(one_det["effective_boundary_r95"], 0.0)
        self.assertEqual(one_det["g_eff"], 0)
        self.assertEqual(one_det["num_clusters"], 1)
        self.assertEqual(one_det["num_nodes"], 1)

        l4 = eg.Layout2d.from_coords([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        k1_det = eg.cluster_area_faithfulness_detail(l4, [3, 3, 3, 3], return_per_cluster=True)
        self.assertEqual(k1_det["caf"], 0.0)
        self.assertEqual(k1_det["effective_boundary_r95"], 0.0)
        self.assertEqual(k1_det["g_eff"], 0)
        self.assertEqual(k1_det["num_clusters"], 1)
        self.assertEqual(k1_det["num_nodes"], 4)

        collapsed = eg.Layout2d.from_coords([[2.0, -1.0]] * 6)
        self.assertEqual(eg.cluster_area_faithfulness(collapsed, [0, 0, 0, 1, 1, 1]), 0.0)

        # Imbalanced clusters: stevens_gamma and boundary_r95=inf.
        coords_imb = np.vstack(
            [
                np.column_stack([np.linspace(-2.0, -1.5, 20), np.zeros(20)]),
                np.column_stack([np.linspace(1.5, 2.0, 4), np.zeros(4)]),
            ]
        )
        coords_imb[:, 1] = np.sin(np.arange(24) * 0.7) * 0.2
        labels_imb = [0] * 20 + [1] * 4
        l_imb = eg.Layout2d.from_coords(coords_imb)

        det_g05 = eg.cluster_area_faithfulness_detail(
            l_imb, labels_imb, stevens_gamma=0.5, return_per_cluster=True
        )
        det_g20 = eg.cluster_area_faithfulness_detail(
            l_imb, labels_imb, stevens_gamma=2.0, return_per_cluster=True
        )
        self.assertAlmostEqual(float(np.sum(det_g05["p"])), 1.0, places=12)
        self.assertAlmostEqual(float(np.sum(det_g20["p"])), 1.0, places=12)
        self.assertLess(float(det_g05["p"][0]), float(det_g20["p"][0]))

        det_inf = eg.cluster_area_faithfulness_detail(
            l_imb, labels_imb, boundary_r95=float("inf"), return_per_cluster=False
        )
        self.assertAlmostEqual(float(det_inf["in_frac"]), 1.0, places=12)
        self.assertGreaterEqual(float(det_inf["caf"]), 0.0)
        self.assertLessEqual(float(det_inf["caf"]), 1.0)

        # evaluate_metrics integration ("cluster_area_faithfulness" & "gcaf") + lazy validation.
        g_imb = eg.CsrGraph.from_edges(
            24, [(i, i + 1) for i in range(23)], directed=False, sanitize=True
        )
        rep = eg.evaluate_metrics(
            g_imb,
            l_imb,
            ["cluster_area_faithfulness"],
            community_labels=labels_imb,
            g_max=64,
            sigma_cells=0.9,
            kappa=8.0,
            tau_frac=0.2,
            boundary_r95=2.5,
            stevens_gamma=0.7,
        )
        expected_caf = eg.cluster_area_faithfulness(
            l_imb,
            labels_imb,
            g_max=64,
            sigma_cells=0.9,
            kappa=8.0,
            tau_frac=0.2,
            boundary_r95=2.5,
            stevens_gamma=0.7,
        )
        self.assertAlmostEqual(rep["cluster_area_faithfulness"], expected_caf, places=12)

        rep_alias = eg.evaluate_metrics(g_imb, l_imb, ["gcaf"], community_labels=labels_imb)
        self.assertAlmostEqual(
            rep_alias["cluster_area_faithfulness"],
            eg.cluster_area_faithfulness(l_imb, labels_imb),
            places=12,
        )

        # Invalid G-CAF config is lazily ignored unless G-CAF target is requested.
        ok = eg.evaluate_metrics(g_imb, l_imb, ["aspect_ratio"], g_max=4)
        self.assertIn("aspect_ratio", ok)
        self.assert_no_panic_raises(
            (ValueError,),
            lambda: eg.evaluate_metrics(
                g_imb, l_imb, ["cluster_area_faithfulness"], community_labels=labels_imb, g_max=4
            ),
        )
        self.assert_no_panic_raises(
            (ValueError,),
            lambda: eg.evaluate_metrics(g_imb, l_imb, ["cluster_area_faithfulness"]),
        )

        # Adversarial error handling (no panics).
        nan_layout = eg.Layout2d.from_coords(coords_imb.copy())
        nan_layout.coords(copy=False)[0, 0] = float("nan")
        inf_layout = eg.Layout2d.from_coords(coords_imb.copy())
        inf_layout.coords(copy=False)[0, 1] = float("inf")

        self.assert_no_panic_raises(
            (ValueError,), lambda: eg.cluster_area_faithfulness(nan_layout, labels_imb)
        )
        self.assert_no_panic_raises(
            (ValueError,), lambda: eg.cluster_area_faithfulness(inf_layout, labels_imb)
        )
        self.assert_no_panic_raises(
            (ValueError,), lambda: eg.cluster_area_faithfulness(l_imb, labels_imb[:-1])
        )
        self.assert_no_panic_raises(
            (ValueError,), lambda: eg.cluster_area_faithfulness(l_imb, labels_imb, g_max=11)
        )
        self.assert_no_panic_raises(
            (ValueError,), lambda: eg.cluster_area_faithfulness(l_imb, labels_imb, g_max=-1)
        )
        self.assert_no_panic_raises(
            (TypeError,), lambda: eg.cluster_area_faithfulness(l_imb, labels_imb, g_max=True)
        )
        for bad_sigma in (0.0, -0.5, float("nan"), float("inf"), 400.0):
            self.assert_no_panic_raises(
                (ValueError,),
                lambda s=bad_sigma: eg.cluster_area_faithfulness(
                    l_imb, labels_imb, sigma_cells=s
                ),
            )
        for bad_pos in (0.0, -1.0, float("nan"), float("inf")):
            self.assert_no_panic_raises(
                (ValueError,),
                lambda v=bad_pos: eg.cluster_area_faithfulness(l_imb, labels_imb, kappa=v),
            )
            self.assert_no_panic_raises(
                (ValueError,),
                lambda v=bad_pos: eg.cluster_area_faithfulness(l_imb, labels_imb, tau_frac=v),
            )
            self.assert_no_panic_raises(
                (ValueError,),
                lambda v=bad_pos: eg.cluster_area_faithfulness(l_imb, labels_imb, stevens_gamma=v),
            )
        for bad_b in (0.0, -1.0, float("nan"), float("-inf")):
            self.assert_no_panic_raises(
                (ValueError,),
                lambda b=bad_b: eg.cluster_area_faithfulness(l_imb, labels_imb, boundary_r95=b),
            )
        self.assert_no_panic_raises(
            (TypeError,),
            lambda: eg.cluster_area_faithfulness(l_imb, labels_imb, sigma_cells=True),
        )
        self.assert_no_panic_raises(
            (ValueError,),
            lambda: eg.cluster_area_faithfulness(l_imb, [-1] + labels_imb[1:]),
        )
        self.assert_no_panic_raises(
            (TypeError,),
            lambda: eg.cluster_area_faithfulness(l_imb, [True] * len(labels_imb)),
        )
        self.assert_no_panic_raises(
            (ValueError,),
            lambda: eg.cluster_area_faithfulness(
                l_imb, np.zeros((len(labels_imb), 1), dtype=np.uint32)
            ),
        )
        self.assert_no_panic_raises(
            (TypeError,),
            lambda: eg.cluster_area_faithfulness(l_imb, "01" * 12),  # type: ignore[arg-type]
        )

    def test_cluster_area_faithfulness_label_coercions_and_lazy_validation(self) -> None:
        # Gap 6: Untested extract_u32_1d integer/strided/tuple coercions, rejections,
        # bool kwargs, and lazy invalid G-CAF args when G-CAF is not requested.
        coords = np.array(
            [
                [0.0, 0.0],
                [0.2, 0.0],
                [0.0, 0.2],
                [0.2, 0.2],
                [4.0, 0.0],
                [4.2, 0.0],
                [4.0, 0.2],
                [4.2, 0.2],
            ],
            dtype=np.float64,
        )
        layout = eg.Layout2d.from_coords(coords)
        base_labels = [0, 0, 0, 0, 1, 1, 1, 1]
        base_caf = eg.cluster_area_faithfulness(layout, base_labels)
        self.assertGreater(base_caf, 0.2)

        # Valid dtypes, non-contiguous strided slice, and tuple
        for arr in (
            np.array(base_labels, dtype=np.uint16),
            np.array(base_labels, dtype=np.uint64),
            np.array(base_labels, dtype=np.int16),
            np.array(base_labels, dtype=np.int8),
            np.array([0, 99, 0, 99, 0, 99, 0, 99, 1, 99, 1, 99, 1, 99, 1, 99], dtype=np.uint32)[
                ::2
            ],
            tuple(base_labels),
        ):
            val = eg.cluster_area_faithfulness(layout, arr)
            self.assertAlmostEqual(val, base_caf, places=12)

        # ValueError for negative int16 and u32::MAX + 1 in uint64
        self.assert_no_panic_raises(
            (ValueError,),
            lambda: eg.cluster_area_faithfulness(
                layout, np.array([-1, 0, 0, 0, 1, 1, 1, 1], dtype=np.int16)
            ),
        )
        self.assert_no_panic_raises(
            (ValueError,),
            lambda: eg.cluster_area_faithfulness(
                layout, np.array([2**32, 0, 0, 0, 1, 1, 1, 1], dtype=np.uint64)
            ),
        )

        # TypeError for bool ndarray, bytes, and bool kwargs
        self.assert_no_panic_raises(
            (TypeError,),
            lambda: eg.cluster_area_faithfulness(
                layout, np.array([False, False, False, False, True, True, True, True], dtype=bool)
            ),
        )
        self.assert_no_panic_raises(
            (TypeError,),
            lambda: eg.cluster_area_faithfulness(layout, b"\x00\x00\x00\x00\x01\x01\x01\x01"),
        )
        self.assert_no_panic_raises(
            (TypeError,),
            lambda: eg.cluster_area_faithfulness(layout, base_labels, g_max=True),
        )
        self.assert_no_panic_raises(
            (TypeError,),
            lambda: eg.cluster_area_faithfulness(layout, base_labels, sigma_cells=False),
        )
        self.assert_no_panic_raises(
            (TypeError,),
            lambda: eg.cluster_area_faithfulness_detail(
                layout, base_labels, return_per_cluster=1  # type: ignore[arg-type]
            ),
        )

        # Lazy non-GCAF evaluation ignores mismatched community_labels & invalid sigma_cells
        g = eg.CsrGraph.from_edges(
            8, [(i, (i + 1) % 8) for i in range(8)], directed=False, sanitize=True
        )
        sns_only = eg.evaluate_metrics(
            g, layout, ["sns"], community_labels=[0, 1], sigma_cells=-1.0
        )
        self.assertIn("scale_normalized_stress", sns_only)
        self.assertNotIn("cluster_area_faithfulness", sns_only)

    def test_cluster_area_faithfulness_f1_fields_and_g6_weights(self) -> None:
        # F1: finite boundary_r95 clamps g_eff first while preserving effective_boundary_r95.
        rng = np.random.default_rng(123)
        n = 50_000
        coords_big = np.vstack(
            [
                rng.normal(loc=[-1.0, 0.0], scale=0.15, size=(n // 2, 2)),
                rng.normal(loc=[1.0, 0.0], scale=0.15, size=(n - n // 2, 2)),
            ]
        )
        labels_big = [0] * (n // 2) + [1] * (n - n // 2)
        det_2048 = eg.cluster_area_faithfulness_detail(
            eg.Layout2d.from_coords(coords_big),
            labels_big,
            g_max=2048,
            sigma_cells=3.0,
            boundary_r95=2.0,
        )
        self.assertEqual(det_2048["g_eff"], 506)
        self.assertEqual(det_2048["effective_boundary_r95"], 2.0)
        self.assertGreater(float(det_2048["in_frac"]), 0.99)

        # G6: node_weights vs cluster_weights parity, integer/float arrays, stevens_gamma, and errors.
        coords = np.array(
            [
                [0.0, 0.0],
                [0.2, 0.0],
                [0.0, 0.2],
                [0.2, 0.2],
                [4.0, 0.0],
                [4.2, 0.0],
                [4.0, 0.2],
                [4.2, 0.2],
            ],
            dtype=np.float64,
        )
        layout = eg.Layout2d.from_coords(coords)
        labels = [20, 20, 20, 20, 10, 10, 10, 10]  # sorted cluster_ids = [10, 20]
        g = eg.CsrGraph.from_edges(
            8, [(i, (i + 1) % 8) for i in range(8)], directed=False, sanitize=True
        )
        nw = [1.0, 2.0, 1.0, 2.0, 3.0, 3.0, 3.0, 3.0]  # cluster 10 sum=12, cluster 20 sum=6
        cw = [12.0, 6.0]

        caf_nw = eg.cluster_area_faithfulness(layout, labels, node_weights=nw)
        caf_cw = eg.cluster_area_faithfulness(layout, labels, cluster_weights=cw)
        self.assertAlmostEqual(caf_nw, caf_cw, places=12)

        det_nw = eg.cluster_area_faithfulness_detail(
            layout, labels, node_weights=nw, stevens_gamma=2.0, return_per_cluster=True
        )
        det_cw = eg.cluster_area_faithfulness_detail(
            layout, labels, cluster_weights=cw, stevens_gamma=2.0, return_per_cluster=True
        )
        self.assertAlmostEqual(float(det_nw["caf"]), float(det_cw["caf"]), places=12)
        np.testing.assert_allclose(det_nw["p"], np.array([0.8, 0.2]), atol=1e-12)
        np.testing.assert_allclose(det_cw["p"], np.array([0.8, 0.2]), atol=1e-12)

        for w_arr in (
            np.array([12, 6], dtype=np.uint8),
            np.array([12, 6], dtype=np.uint16),
            np.array([12, 6], dtype=np.int8),
            np.array([12, 6], dtype=np.int16),
            np.array([12.0, 6.0], dtype=np.float32),
            np.array([12.0, 6.0], dtype=np.float64),
        ):
            self.assertAlmostEqual(
                eg.cluster_area_faithfulness(layout, labels, cluster_weights=w_arr),
                caf_cw,
                places=6,
            )

        ev_nw = eg.evaluate_metrics(g, layout, ["gcaf"], community_labels=labels, node_weights=nw)
        ev_cw = eg.evaluate_metrics(g, layout, ["gcaf"], community_labels=labels, cluster_weights=cw)
        self.assertAlmostEqual(ev_nw["cluster_area_faithfulness"], caf_nw, places=12)
        self.assertAlmostEqual(ev_cw["cluster_area_faithfulness"], caf_cw, places=12)

        # Lazy validation in evaluate_metrics ignores invalid weight values if gcaf not requested.
        ev_lazy = eg.evaluate_metrics(
            g, layout, ["aspect_ratio"], community_labels=labels, node_weights=[-1.0]
        )
        self.assertIn("aspect_ratio", ev_lazy)

        # ValueError guards: mutual exclusion, mismatch, non-finite, <= 0.
        self.assert_no_panic_raises(
            (ValueError,),
            lambda: eg.cluster_area_faithfulness(
                layout, labels, node_weights=nw, cluster_weights=cw
            ),
        )
        self.assert_no_panic_raises(
            (ValueError,),
            lambda: eg.cluster_area_faithfulness(layout, labels, node_weights=nw[:-1]),
        )
        self.assert_no_panic_raises(
            (ValueError,),
            lambda: eg.cluster_area_faithfulness(layout, labels, cluster_weights=[1.0]),
        )
        for bad_val in (0.0, -1.0, float("nan"), float("inf")):
            self.assert_no_panic_raises(
                (ValueError,),
                lambda v=bad_val: eg.cluster_area_faithfulness(
                    layout, labels, cluster_weights=[v, 1.0]
                ),
            )
            self.assert_no_panic_raises(
                (ValueError,),
                lambda v=bad_val: eg.cluster_area_faithfulness(
                    layout, labels, node_weights=[v] + nw[1:]
                ),
            )

        # TypeError guards: bool, complex, str, dict.
        bad_typed_inputs: tuple[Any, ...] = (
            np.array([True, False], dtype=bool),
            np.array([1 + 0j, 2 + 0j], dtype=np.complex128),
            [True, False],
            "12",
            {0: 1.0, 1: 2.0},
        )
        for bad_typed in bad_typed_inputs:
            self.assert_no_panic_raises(
                (TypeError,),
                lambda bw=bad_typed: eg.cluster_area_faithfulness(
                    layout, labels, cluster_weights=bw
                ),
            )
        self.assert_no_panic_raises(
            (ValueError,),
            lambda: eg.cluster_area_faithfulness(
                layout, labels, cluster_weights=np.ones((2, 1), dtype=np.float64)
            ),
        )

    def test_w1_6_cluster_baselines_and_nor(self) -> None:
        # Well-separated 2-cluster 2D layout (4 points per cluster)
        coords = np.array(
            [
                [0.0, 0.0],
                [0.2, 0.0],
                [0.0, 0.2],
                [0.2, 0.2],
                [10.0, 0.0],
                [10.2, 0.0],
                [10.0, 0.2],
                [10.2, 0.2],
            ],
            dtype=np.float64,
        )
        labels = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.uint32)
        pred_perfect = np.array([10, 10, 10, 10, 20, 20, 20, 20], dtype=np.uint32)
        pred_partial = np.array([0, 0, 0, 1, 1, 1, 1, 0], dtype=np.uint32)

        pa_perf = eg.partition_agreement(labels, pred_perfect)
        self.assertEqual(
            set(pa_perf.keys()),
            {"ari", "ami", "homogeneity", "completeness", "v_measure", "fmi"},
        )
        for key in ("ari", "ami", "homogeneity", "completeness", "v_measure", "fmi"):
            self.assertAlmostEqual(pa_perf[key], 1.0, places=12)

        # Cross-check ARI / FMI / Homogeneity / Completeness / V-measure on partial agreement
        pa_part = eg.partition_agreement(labels, pred_partial)
        self.assertLess(pa_part["ari"], 0.6)
        self.assertGreater(pa_part["fmi"], 0.4)

        for dtype in (np.float64, np.float32):
            lay = eg.Layout2d.from_coords(coords.astype(dtype))
            km = eg.cq_kmeans(lay, labels, n_init=5, max_iters=100, seed=42)
            hac = eg.cq_hac(lay, labels)
            self.assertAlmostEqual(km["ari"], 1.0, places=12)
            self.assertAlmostEqual(km["ami"], 1.0, places=12)
            self.assertAlmostEqual(hac["homogeneity"], 1.0, places=12)
            self.assertAlmostEqual(hac["ami"], 1.0, places=12)

            self.assertAlmostEqual(eg.distance_consistency(lay, labels), 1.0, places=12)
            self.assertAlmostEqual(eg.distribution_consistency(lay, labels, grid_size=20), 1.0, places=12)
            cdm, hdm = eg.class_density_measure(lay, labels, grid_size=32)
            self.assertGreater(cdm, 0.9)
            self.assertAlmostEqual(hdm, 1.0, places=12)

            self.assertAlmostEqual(eg.sepme_knng(lay, labels, k=2), 1.0, places=12)
            self.assertGreater(eg.sepme_rng(lay, labels), 0.7)
            self.assertGreater(eg.sepme_gab(lay, labels), 0.5)
            self.assertGreater(eg.sepme_gong(lay, labels, gamma=0.35), 0.7)
            mixed_labels = np.array([0, 1, 0, 1, 0, 1, 0, 1], dtype=np.uint32)
            self.assertGreater(eg.sepme_rng(lay, labels), eg.sepme_rng(lay, mixed_labels))
            self.assertGreater(eg.sepme_gab(lay, labels), eg.sepme_gab(lay, mixed_labels))
            self.assertGreater(
                eg.sepme_gong(lay, labels, gamma=0.35),
                eg.sepme_gong(lay, mixed_labels, gamma=0.35),
            )
            self.assertAlmostEqual(eg.neighborhood_hit(lay, labels, k=3), 1.0, places=12)

            sil = eg.silhouette_score(lay, labels)
            ch = eg.calinski_harabasz_score(lay, labels)
            db = eg.davies_bouldin_score(lay, labels)
            self.assertGreater(sil, 0.95)
            self.assertGreater(ch, 1000.0)
            self.assertLess(db, 0.1)

            pdoe = eg.pairwise_distance_overlap_error(lay, labels, num_bins=50)
            self.assertGreaterEqual(pdoe, 0.0)
            self.assertLessEqual(pdoe, 1.0)

            hw = np.full(8, 0.25, dtype=np.float64)
            hh = np.full(8, 0.25, dtype=np.float64)
            sp_rect = eg.sprawlter_nn(lay, hw, hh)
            sp_rad = eg.sprawlter_nn(lay, hw)
            self.assertAlmostEqual(sp_rect, sp_rad, places=12)
            self.assertGreater(sp_rect, 0.0)

        # NOR metrics across (f64, f64), (f32, f32), (f64, f32), (f32, f64)
        orig_pts = np.array([[0.0, 0.0], [0.2, 0.0], [2.0, 0.0], [2.0, 2.0]], dtype=np.float64)
        adj_pts = np.array([[0.0, 0.0], [1.2, 0.0], [2.5, 0.0], [2.5, 2.0]], dtype=np.float64)
        hw4 = [0.5, 0.5, 0.5, 0.5]
        hh4 = [0.5, 0.5, 0.5, 0.5]
        g4 = eg.CsrGraph.from_edges(4, [(0, 1), (1, 2), (2, 3)], directed=False, sanitize=True)

        l_orig_64 = eg.Layout2d.from_coords(orig_pts)
        l_adj_64 = eg.Layout2d.from_coords(adj_pts)
        l_orig_32 = eg.Layout2d.from_coords(orig_pts.astype(np.float32))
        l_adj_32 = eg.Layout2d.from_coords(adj_pts.astype(np.float32))

        nor_no_g = eg.nor_metrics(l_orig_64, l_adj_64, hw4, hh4)
        self.assertEqual(
            set(nor_no_g.keys()),
            {"oo_nni", "sp_ch_a", "gs_bb_iar", "nm_dm_imse", "el_rsd"},
        )
        self.assertEqual(nor_no_g["oo_nni"], 0.0)
        self.assertEqual(nor_no_g["el_rsd"], 0.0)

        nor_g = eg.nor_metrics(l_orig_64, l_adj_64, hw4, hh4, graph=g4)
        self.assertGreater(nor_g["el_rsd"], 0.0)
        self.assertGreater(nor_g["sp_ch_a"], 0.0)
        self.assertGreaterEqual(nor_g["gs_bb_iar"], 1.0)
        self.assertGreater(nor_g["nm_dm_imse"], 0.0)

        for lo, la in ((l_orig_32, l_adj_32), (l_orig_64, l_adj_32), (l_orig_32, l_adj_64)):
            nor_mix = eg.nor_metrics(lo, la, hw4, hh4, graph=g4)
            self.assertAlmostEqual(nor_mix["oo_nni"], nor_g["oo_nni"], places=6)
            self.assertAlmostEqual(nor_mix["sp_ch_a"], nor_g["sp_ch_a"], places=5)

        # Degenerate empty / singleton boundaries (zero PanicException)
        empty_lay = eg.Layout2d.zeros(0)
        self.assertEqual(eg.distance_consistency(empty_lay, []), 0.0)
        self.assertEqual(eg.sepme_rng(empty_lay, []), 0.0)
        single_lay = eg.Layout2d.zeros(1)
        self.assertEqual(eg.sepme_gab(single_lay, [0]), 1.0)
        self.assertEqual(eg.silhouette_score(single_lay, [0]), 0.0)

        # Validation errors
        lay64 = eg.Layout2d.from_coords(coords)
        self.assert_no_panic_raises(
            (ValueError,), lambda: eg.partition_agreement([0, 1], [0])
        )
        self.assert_no_panic_raises(
            (ValueError,), lambda: eg.sepme_knng(lay64, labels, k=0)
        )
        self.assert_no_panic_raises(
            (ValueError,), lambda: eg.sepme_gong(lay64, labels, gamma=1.5)
        )
        self.assert_no_panic_raises(
            (ValueError,), lambda: eg.distribution_consistency(lay64, labels, grid_size=0)
        )
        self.assert_no_panic_raises(
            (ValueError,), lambda: eg.sprawlter_nn(lay64, [-1.0] * 8)
        )

    def test_w1_6_evaluate_metrics_targets(self) -> None:
        coords = np.array(
            [
                [0.0, 0.0],
                [0.2, 0.0],
                [0.0, 0.2],
                [0.2, 0.2],
                [8.0, 0.0],
                [8.2, 0.0],
                [8.0, 0.2],
                [8.2, 0.2],
            ],
            dtype=np.float64,
        )
        g = eg.CsrGraph.from_edges(
            8, [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7)], directed=False
        )
        lay = eg.Layout2d.from_coords(coords)
        labels = [0, 0, 0, 0, 1, 1, 1, 1]

        all_w16_targets = [
            "dsc",
            "dc",
            "cdm",
            "hdm",
            "sepme_knng",
            "sepme_rng",
            "sepme_gab",
            "sepme_gong",
            "neighborhood_hit",
            "silhouette",
            "calinski_harabasz",
            "davies_bouldin",
            "cq_kmeans_ari",
            "cq_kmeans_ami",
            "cq_hac_hom",
            "cq_hac_ami",
            "pairwise_distance_overlap_error",
            "sprawlter_nn",
        ]
        res = eg.evaluate_metrics(
            g,
            lay,
            all_w16_targets,
            community_labels=labels,
            sepme_knng_k=2,
            neighborhood_hit_k=3,
            dc_grid_size=20,
            cdm_grid_size=32,
            cdm_bandwidth=0.05,
            num_bins=50,
            gong_gamma=0.35,
            kmeans_n_init=5,
            kmeans_max_iters=100,
            sprawlter_node_radius=0.5,
            seed=42,
        )
        self.assertEqual(set(res.keys()), set(all_w16_targets))
        self.assertAlmostEqual(res["dsc"], eg.distance_consistency(lay, labels), places=12)
        self.assertAlmostEqual(
            res["neighborhood_hit"], eg.neighborhood_hit(lay, labels, k=3), places=12
        )
        self.assertAlmostEqual(
            res["sprawlter_nn"], eg.sprawlter_nn(lay, [0.5] * 8), places=12
        )

        # Alias checks for sepme_knng2 / sepme-knng2 / sepme-knng
        for alias in ("sepme_knng", "sepme-knng", "sepme_knng2", "sepme-knng2"):
            r_alias = eg.evaluate_metrics(g, lay, [alias], community_labels=labels, sepme_knng_k=2)
            self.assertAlmostEqual(r_alias["sepme_knng"], res["sepme_knng"], places=12)

        # Missing community_labels error for community-dependent W1-6 targets
        self.assert_no_panic_raises(
            (ValueError,), lambda: eg.evaluate_metrics(g, lay, ["dsc"])
        )
        # Lazy validation ignores invalid baseline kwargs when unrelated targets are requested
        ok = eg.evaluate_metrics(g, lay, ["aspect_ratio"], sepme_knng_k=0)
        self.assertIn("aspect_ratio", ok)
        self.assert_no_panic_raises(
            (ValueError,),
            lambda: eg.evaluate_metrics(
                g, lay, ["sepme_knng"], community_labels=labels, sepme_knng_k=0
            ),
        )


if __name__ == "__main__":
    unittest.main()
