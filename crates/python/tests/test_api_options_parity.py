from __future__ import annotations

import inspect
import math
import unittest

import egraph as eg
import numpy as np


def _path_graph(n: int = 8) -> eg.CsrGraph:
    edges = [(i, i + 1) for i in range(n - 1)]
    return eg.CsrGraph.from_edges(n, edges, directed=False, sanitize=True)


def _grid_graph(rows: int = 3, cols: int = 3) -> eg.CsrGraph:
    edges: list[tuple[int, int]] = []
    for r in range(rows):
        for c in range(cols):
            u = r * cols + c
            if r + 1 < rows:
                edges.append((u, (r + 1) * cols + c))
            if c + 1 < cols:
                edges.append((u, r * cols + (c + 1)))
    return eg.CsrGraph.from_edges(rows * cols, edges, directed=False, sanitize=True)


class TestCoreLayoutApiParity(unittest.TestCase):
    def test_csr_graph_parity_methods(self) -> None:
        g = eg.CsrGraph.from_edges(
            5, [(0, 1), (1, 2), (2, 0), (3, 4)], weights=[2.0, 3.0, 4.0, 5.0]
        )
        self.assertEqual(g.nnz, 8)
        self.assertTrue(g.is_weighted)
        self.assertEqual(g.degree(0), 2)
        self.assertEqual(g.degree(3), 1)
        np.testing.assert_array_equal(g.neighbors(0), np.array([1, 2], dtype=np.uint32))

        with self.assertRaises(IndexError):
            g.degree(5)
        with self.assertRaises(IndexError):
            g.neighbors(5)

        g_edge_list = eg.CsrGraph.from_edge_list(
            5, [(0, 1), (1, 2), (2, 0), (3, 4)], weights=[2.0, 3.0, 4.0, 5.0]
        )
        np.testing.assert_array_equal(g_edge_list.indptr(), g.indptr())
        np.testing.assert_array_equal(g_edge_list.indices(), g.indices())

        sub = g.induced_subgraph([0, 1, 2])
        self.assertEqual(sub.num_nodes, 3)
        self.assertEqual(sub.num_edges, 3)

        lcc, orig_ids = g.largest_connected_component()
        self.assertEqual(lcc.num_nodes, 3)
        np.testing.assert_array_equal(orig_ids, np.array([0, 1, 2], dtype=np.uint32))

    def test_layout2d_and_layoutnd_parity_f64_and_f32(self) -> None:
        for dtype in ("float64", "float32"):
            empty = eg.Layout2d.zeros(0, dtype=dtype)
            self.assertTrue(empty.is_empty())

            layout = eg.Layout2d.zeros(4, dtype=dtype)
            self.assertEqual(layout.num_nodes, 4)
            self.assertFalse(layout.is_empty())

            l_init = eg.Layout2d.init(5, seed=42, dtype=dtype)
            self.assertEqual(l_init.num_nodes, 5)
            self.assertEqual(l_init.dtype, dtype)
            self.assertTrue(l_init.is_all_finite())
            copied = l_init.copy_coords()
            self.assertEqual(copied.shape, (5, 2))
            copied[0, 0] = 999.0
            self.assertNotAlmostEqual(l_init.x(0), 999.0)
            layout.set_x(1, 3.5)
            layout.set_y(1, -2.5)
            self.assertAlmostEqual(layout.x(1), 3.5, places=5)
            self.assertAlmostEqual(layout.y(1), -2.5, places=5)
            layout.set_x(0, -1.0)
            layout.set_y(0, 1.0)
            layout.normalize_std(target_std=2.0)
            coords = layout.coords(copy=True)
            self.assertTrue(layout.is_all_finite())
            pooled_std = math.sqrt(
                0.5 * (float(np.var(coords[:, 0])) + float(np.var(coords[:, 1])))
            )
            self.assertAlmostEqual(pooled_std, 2.0, places=4)

            nd = eg.LayoutNd.zeros(4, 3, dtype=dtype)
            self.assertEqual(len(nd), 4)
            self.assertEqual(nd.num_nodes, 4)
            self.assertEqual(nd.dim, 3)
            self.assertEqual(nd.dtype, dtype)
            self.assertFalse(nd.is_empty())
            self.assertTrue(nd.is_all_finite())

            view = nd.coords(copy=False)
            view[2, 1] = 7.25
            np.testing.assert_allclose(nd.coords(copy=True)[2, 1], 7.25, rtol=1e-5)

            np_dtype = np.float64 if dtype == "float64" else np.float32
            nd2 = eg.LayoutNd.from_coords(np.ones((3, 5), dtype=np_dtype))
            self.assertEqual(nd2.dim, 5)
            self.assertEqual(nd2.dtype, dtype)

    def test_manifold_layout_and_distances(self) -> None:
        for dtype in ("float64", "float32"):
            hyp = eg.ManifoldLayout2d.hyperbolic_poincare(3, dtype=dtype)
            self.assertEqual(hyp.manifold, "hyperbolic")
            self.assertEqual(hyp.num_nodes, 3)
            self.assertEqual(hyp.dtype, dtype)
            sph = eg.ManifoldLayout2d.spherical(2, dtype=dtype)
            self.assertEqual(sph.manifold, "spherical")
            self.assertEqual(sph.num_nodes, 2)
            tor = eg.ManifoldLayout2d.torus_2d(2, dtype=dtype)
            self.assertEqual(tor.manifold, "torus")
            self.assertEqual(tor.num_nodes, 2)

            m = eg.ManifoldLayout2d.zeros("hyperbolic", 3, dtype=dtype)
            m.set_point(0, 0.1, 0.2)
            m.set_point(1, -0.15, 0.25)
            self.assertTrue(m.is_all_finite())
            self.assertAlmostEqual(m.x(0), 0.1, places=5)
            self.assertAlmostEqual(m.y(0), 0.2, places=5)
            vx, vy = m.tangent_delta(0, 1)
            d_before = m.geodesic_distance(0, 1)
            self.assertGreater(d_before, 0.0)
            m.apply_tangent_step(0, 0.5 * vx, 0.5 * vy)
            d_after = m.geodesic_distance(0, 1)
            self.assertLess(d_after, d_before)

        self.assertGreater(eg.poincare_distance(0.0, 0.0, 0.4, 0.0), 0.4)
        self.assertAlmostEqual(
            eg.spherical_distance(0.0, math.pi / 2, math.pi / 2, math.pi / 2),
            math.pi / 2,
            places=6,
        )
        self.assertAlmostEqual(eg.torus_distance(0.05, 0.1, 0.95, 0.1), 0.1, places=6)
        wx, wy = eg.wrap_torus_point(1.25, -0.25)
        self.assertAlmostEqual(wx, 0.25, places=6)
        self.assertAlmostEqual(wy, 0.75, places=6)


class TestDatasetSpatialMetricsApiParity(unittest.TestCase):
    def test_dataset_loader_and_parse_csr_csv(self) -> None:
        karate = eg.load_dataset("karate")
        self.assertEqual(karate.num_nodes, 34)
        self.assertEqual(karate.num_edges, 78)

        with self.assertRaises(ValueError):
            eg.load_dataset("unknown_dataset_xyz")

        parsed = eg.parse_csr_csv("4,3\n0,1\n1,2\n2,3\n")
        self.assertEqual(parsed.num_nodes, 4)
        self.assertEqual(parsed.num_edges, 3)

    def test_spatial_delaunay_to_csr_and_rectangle_overlaps(self) -> None:
        layout = eg.Layout2d.from_coords([[0.0, 0.0], [1.0, 0.0], [0.5, 1.0], [0.2, 0.2]])
        d_csr = eg.delaunay_to_csr(layout)
        self.assertEqual(d_csr.num_nodes, 4)
        self.assertGreaterEqual(d_csr.num_edges, 5)

        rects = eg.Layout2d.from_coords([[0.0, 0.0], [0.5, 0.0], [5.0, 5.0]])
        overlaps = eg.rectangle_overlaps(rects, [0.4, 0.4, 0.4], [0.4, 0.4, 0.4])
        np.testing.assert_array_equal(overlaps, np.array([[0, 1]], dtype=np.uint32))

    def test_metrics_parity_functions(self) -> None:
        g = eg.CsrGraph.from_edges(4, [(0, 1), (1, 2), (2, 3), (3, 0), (0, 2), (1, 3)])
        layout = eg.Layout2d.from_coords([[0.0, 0.0], [1.0, 1.0], [1.0, 0.0], [0.0, 1.0]])

        crossings = eg.crossing_edges(g, layout)
        self.assertEqual(len(crossings), 1)
        self.assertIn("angle_rad", crossings[0])

        summary = eg.crossing_summary(g, layout)
        self.assertEqual(summary["count"], 1)
        self.assertGreater(summary["min_angle_rad"], 0.0)

        torus_layout = eg.Layout2d.from_coords([[0.1, 0.1], [0.9, 0.9], [0.9, 0.1], [0.1, 0.9]])
        torus_crossings = eg.crossing_edges_torus(g, torus_layout)
        self.assertIsInstance(torus_crossings, list)

        gabriel = eg.gabriel_graph_edges(layout)
        rng_edges = eg.relative_neighborhood_edges(layout)
        self.assertEqual(gabriel.ndim, 2)
        self.assertEqual(gabriel.shape[1], 2)
        self.assertEqual(rng_edges.ndim, 2)

        apsp = eg.all_pairs_bfs(g, unit_edge_length=2.0)
        row0 = eg.bfs_single_source(g, 0, unit_edge_length=2.0)
        np.testing.assert_allclose(apsp[0], row0)

        sns = eg.scale_normalized_stress_from_distances(layout, apsp)
        self.assertTrue(math.isfinite(sns))

        iel = eg.ideal_edge_lengths_with_unit(g, layout, unit_edge_length=1.0)
        self.assertGreaterEqual(iel, 0.0)
        m = eg.evaluate_metrics(
            g, layout, ["ideal_edge_lengths"], ideal_edge_length=2.0
        )
        self.assertGreater(m["ideal_edge_lengths"], 0.0)

        adj = eg.Layout2d.from_coords([[0.0, 0.0], [2.0, 2.0], [2.0, 0.0], [0.0, 2.0]])
        d_len, d_dir = eg.prism_distortion_with_edges(layout, adj, [(0, 1), (1, 2)])
        self.assertTrue(math.isfinite(d_len) and math.isfinite(d_dir))


class TestLinalgNeighborhoodSolverApiParity(unittest.TestCase):
    def test_linalg_and_neighborhood_functions(self) -> None:
        g = _grid_graph(3, 3)

        for dtype in ("float64", "float32"):
            cmds = eg.classical_mds_2d(g, dtype=dtype)
            self.assertEqual(cmds.dtype, dtype)
            self.assertTrue(cmds.is_all_finite())

            rmds = eg.rd_mds(g, dims=3, seed=42, dtype=dtype)
            self.assertEqual(rmds.dim, 3)
            self.assertEqual(rmds.dtype, dtype)

            eg.normalize_layout_2d(cmds, target_std=1.5, seed=42)
            self.assertAlmostEqual(
                float(np.std(cmds.coords(copy=True)[:, 0])), 1.5, places=4
            )

            rep_dist = eg.all_pairs_bfs(g)[:, :4]
            rep_nd = eg.representative_mds(rep_dist, dims=2, dtype=dtype)
            self.assertEqual(rep_nd.num_nodes, 9)
            self.assertEqual(rep_nd.dim, 2)

        res_coords, res_lambdas = eg.compute_resistance_embedding(g, dim=4, seed=42)
        self.assertEqual(res_coords.shape[0], 9)
        self.assertEqual(len(res_lambdas), res_coords.shape[1])

        diff_coords, diff_lambdas, t_used = eg.compute_diffusion_embedding(
            g, dim=4, seed=42
        )
        self.assertEqual(diff_coords.shape[0], 9)
        self.assertEqual(len(diff_lambdas), diff_coords.shape[1])
        self.assertGreater(t_used, 0.0)

        rng = np.random.default_rng(42)
        pts = rng.normal(size=(24, 16))
        knn = eg.build_rerank_knn(pts, k=5, coarse_dim=6, rerank_factor=4)
        self.assertEqual(knn.num_nodes, 24)

        probs = eg.graph_row_normalized_probabilities(g)
        sym = eg.symmetrize_row_normalized(
            g.num_nodes, g.indptr(), g.indices(), np.ones(g.nnz, dtype=np.float64)
        )
        self.assertEqual(probs.num_nodes, 9)
        self.assertEqual(sym.num_nodes, 9)

        knn_prio = eg.build_gumap_12hop_priority_csr(
            4,
            [0, 1, 2],
            [1, 2, 3],
            [1.0, 1.0, 1.0],
            [0, 1],
            [2, 3],
            [1.8, 1.9],
            tier2_margin=1.1,
        )
        self.assertEqual(knn_prio.num_nodes, 4)
        self.assertGreaterEqual(knn_prio.nnz, 3)

        f1, f2, f3, d_floor = eg.apply_relative_distance_floor(
            [1.0, 2.0], [0.1, 0.2], [0.05, 0.05], min_ratio=0.5
        )
        self.assertGreater(d_floor, 0.0)
        self.assertEqual(len(f1), 2)
        self.assertTrue(np.all(f2 >= d_floor - 1e-12))
        self.assertTrue(np.all(f3 >= d_floor - 1e-12))

    def test_solvers_tsnet_tfdp_ibumap_options_and_getters(self) -> None:
        g = _grid_graph(3, 3)

        for dtype in ("float64", "float32"):
            layout = eg.pivot_mds_init(g, num_pivots=5, seed=42, dtype=dtype)

            tsnet = eg.TsNetSolver.from_graph_partial_bfs(
                g,
                k=4,
                uniform_perplexity=3.0,
                backend="fft",
                iterations_stage1=2,
                iterations_stage2=2,
                iterations_stage3=2,
                seed=42,
                dtype=dtype,
            )
            self.assertIsNone(tsnet.last_fft_stats())
            tsnet.step(layout)
            stats = tsnet.last_fft_stats()
            assert stats is not None
            self.assertGreaterEqual(stats["total_fft_2d_calls"], 1)

            tfdp = eg.TFdpSolver(
                g,
                backend="ibfft",
                total_iterations=5,
                ibfft_order=3,
                ibfft_intervals=16,
                ibfft_min_grid_dim=32,
                ibfft_max_grid_dim=128,
                seed=42,
                dtype=dtype,
            )
            self.assertEqual(tfdp.num_edges, g.nnz)
            self.assertEqual(tfdp.total_fft_2d_calls, 0)
            tfdp.step(layout)
            self.assertGreater(tfdp.total_fft_2d_calls, 0)

            ibumap = eg.IbUmapSolver(
                g,
                degree=12,
                num_landmarks=8,
                min_degree=1,
                jl_dim=16,
                degree_normalize=True,
                p_exp=0.5,
                eps_psd=1e-4,
                t_coarse=32.0,
                t1=None,
                t2=None,
                alpha=None,
                beta_floor=None,
                backend="ibfft",
                n_epochs=5,
                ibfft_order=1,
                ibfft_intervals=16,
                ibfft_min_grid_dim=32,
                ibfft_max_grid_dim=128,
                seed=42,
                dtype=dtype,
            )
            self.assertGreater(ibumap.num_edges, 0)
            self.assertIsNotNone(ibumap.a)
            self.assertIsNotNone(ibumap.b)
            self.assertIsNotNone(ibumap.ab)
            ibumap.step(layout)
            self.assertGreater(ibumap.total_fft_2d_calls, 0)

        # Default degree=None alignment across DoG APIs and 64/4096 metric defaults
        self.assertIsNone(inspect.signature(eg.compute_regularized_dog_varadhan_landmarks).parameters["degree"].default)
        self.assertIsNone(inspect.signature(eg.SgdSolver.sparse_barrier_dog).parameters["degree"].default)
        self.assertIsNone(inspect.signature(eg.GuMapSolver.from_graph_dog).parameters["degree"].default)
        self.assertIsNone(inspect.signature(eg.IbUmapSolver).parameters["degree"].default)

        lm_default = eg.compute_regularized_dog_varadhan_landmarks(g)
        self.assertEqual(lm_default.num_nodes, 9)
        sgd_default = eg.SgdSolver.sparse_barrier_dog(g, min_degree=1, jl_dim=8, t_max=3)
        d_stats = sgd_default.dog_stats()
        assert d_stats is not None
        self.assertGreaterEqual(d_stats["degree"], 40)
        gm_default = eg.GuMapSolver.from_graph_dog(g, min_degree=1, jl_dim=8, n_epochs=3)
        self.assertEqual(len(gm_default.ab), 2)
        ib_default = eg.IbUmapSolver(g, min_degree=1, jl_dim=8, backend="exact", n_epochs=3)
        self.assertIsNotNone(ib_default.ab)

        l_m = eg.pivot_mds_init(g, num_pivots=5, seed=42)
        ev_def = eg.evaluate_metrics(g, l_m, ["sampled_sns", "sampled_crossing_number"])
        ev_exp = eg.evaluate_metrics(
            g, l_m, ["sampled_sns", "sampled_crossing_number"], num_pivots=64, num_samples=4096, seed=42
        )
        self.assertEqual(ev_def, ev_exp)


if __name__ == "__main__":
    unittest.main()
