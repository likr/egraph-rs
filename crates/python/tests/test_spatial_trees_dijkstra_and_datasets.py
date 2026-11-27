from __future__ import annotations

import math
import unittest
from collections.abc import Callable
from typing import Any

import egraph as eg
import numpy as np


class TestSpatialTreesDijkstraAndDatasets(unittest.TestCase):
    def assert_no_panic_raises(
        self,
        expected: tuple[type[BaseException], ...],
        fn: Callable[[], Any],
    ) -> None:
        with self.assertRaises(expected) as ctx:
            fn()
        self.assertNotIn("PanicException", type(ctx.exception).__name__)

    def test_arena_octree_3d_repulsion_f64_f32_and_errors(self) -> None:
        coords = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 1.0, 1.0],
            ],
            dtype=np.float64,
        )
        for dtype_str, np_dtype in (("float64", np.float64), ("float32", np.float32)):
            tree = eg.ArenaOctree3d(dtype=dtype_str)
            self.assertEqual(tree.dtype, dtype_str)
            self.assertEqual(len(tree), 0)
            self.assertEqual(tree.point_count, 0)
            self.assertIn("ArenaOctree3d(", repr(tree))

            lay = eg.LayoutNd.from_coords(coords.astype(np_dtype))
            masses = np.ones(5, dtype=np.float64)
            tree.build(lay, masses=masses)
            self.assertEqual(len(tree), 5)
            self.assertEqual(tree.point_count, 5)
            self.assertGreaterEqual(tree.node_count, 1)

            coulomb = tree.compute_coulomb_repulsion(theta=0.0, eps_sq=1e-6)
            self.assertEqual(coulomb.shape, (5, 3))
            self.assertEqual(coulomb.dtype, np_dtype)
            self.assertTrue(np.all(np.isfinite(coulomb)))
            # Zero net force by Newton's 3rd law on exact evaluation (theta=0.0).
            np.testing.assert_allclose(coulomb.sum(axis=0), np.zeros(3), atol=1e-4)

            maxent = tree.compute_maxent_repulsion(q=0.5, theta=0.5)
            kl, kl_z = tree.compute_kl_repulsion(theta=0.5)
            self.assertGreater(kl_z, 0.0)
            ent = tree.compute_entropy_repulsion(theta=0.5, epsilon_r=0.05)
            for out in (maxent, kl, ent):
                self.assertEqual(out.shape, (5, 3))
                self.assertTrue(np.all(np.isfinite(out)))

            tree2 = eg.ArenaOctree3d(dtype=dtype_str)
            tree2.build_xyz(coords[:, 0], coords[:, 1], coords[:, 2])
            np.testing.assert_allclose(
                tree2.compute_coulomb_repulsion(theta=0.0),
                coulomb,
                rtol=1e-5,
                atol=1e-5,
            )
            tree2.clear()
            self.assertEqual(tree2.point_count, 0)

        self.assert_no_panic_raises((ValueError,), lambda: eg.ArenaOctree3d(dtype="int32"))
        bad_tree = eg.ArenaOctree3d()
        self.assert_no_panic_raises(
            (ValueError,), lambda: bad_tree.build(np.zeros((3, 2), dtype=np.float64))
        )
        self.assert_no_panic_raises(
            (ValueError,),
            lambda: bad_tree.build(
                np.zeros((3, 3), dtype=np.float64), masses=np.array([1.0, -1.0, 1.0])
            ),
        )

    def test_dynamic_kdtree_queries_and_validation(self) -> None:
        pts = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 2.0, 0.0],
                [4.0, 4.0, 4.0],
            ],
            dtype=np.float64,
        )
        kd = eg.DynamicKdTree(pts)
        self.assertEqual(kd.dims, 3)
        self.assertEqual(kd.num_nodes, 4)
        self.assertEqual(len(kd), 4)
        self.assertFalse(kd.is_empty())
        self.assertIn("DynamicKdTree(", repr(kd))
        np.testing.assert_allclose(kd.point(1), [1.0, 0.0, 0.0])

        idx_k, dist_k = kd.k_nearest([0.1, 0.0, 0.0], 2)
        self.assertEqual(list(idx_k), [0, 1])
        self.assertAlmostEqual(float(dist_k[0]), 0.01, places=12)

        idx_ex, _ = kd.k_nearest_excluding_self(0, 2)
        self.assertEqual(list(idx_ex), [1, 2])
        nn_res = kd.nearest_excluding_self(0)
        assert nn_res is not None
        self.assertEqual(nn_res[0], 1)
        self.assertAlmostEqual(nn_res[1], 1.0, places=12)

        idx_r, dist_r = kd.within_radius([0.0, 0.0, 0.0], 1.5)
        self.assertEqual(set(int(i) for i in idx_r), {0, 1})
        self.assertTrue(np.all(dist_r <= 2.25 + 1e-12))
        idx_rsq, _ = kd.within_radius_sq([0.0, 0.0, 0.0], 2.25)
        self.assertEqual(set(int(i) for i in idx_rsq), {0, 1})

        # LayoutNd construction and empty/singleton boundary
        kd_lay = eg.DynamicKdTree(eg.LayoutNd.from_coords(pts.astype(np.float32)))
        self.assertEqual(kd_lay.num_nodes, 4)
        kd_empty = eg.DynamicKdTree(np.zeros((0, 2), dtype=np.float64))
        self.assertTrue(kd_empty.is_empty())
        self.assertEqual(len(kd_empty.k_nearest([0.0, 0.0], 3)[0]), 0)
        kd_one = eg.DynamicKdTree(np.array([[1.0, 2.0]], dtype=np.float64))
        self.assertIsNone(kd_one.nearest_excluding_self(0))

        self.assert_no_panic_raises(
            (ValueError,), lambda: eg.DynamicKdTree(np.array([[0.0, float("nan")]]))
        )
        self.assert_no_panic_raises((IndexError,), lambda: kd.point(99))
        self.assert_no_panic_raises((ValueError,), lambda: kd.k_nearest([0.0, 0.0], 1))
        self.assert_no_panic_raises((ValueError,), lambda: kd.within_radius([0.0, 0.0, 0.0], -0.1))

    def test_dijkstra_and_csr_graph_methods(self) -> None:
        g = eg.CsrGraph.from_edges(
            4,
            [(0, 1), (1, 2), (2, 3), (0, 3)],
            weights=[1.5, 2.0, 1.0, 10.0],
            directed=False,
            sanitize=True,
            weight_dtype="float32",
        )
        self.assertEqual(len(g), 4)
        self.assertEqual(g.weight_dtype, "float32")
        self.assertTrue(g.is_connected())
        self.assertTrue(g.has_edge(0, 1))
        self.assertFalse(g.has_edge(0, 2))
        self.assertAlmostEqual(g.edge_weight(0, 1) or -1.0, 1.5, places=6)
        self.assertIsNone(g.edge_weight(0, 2))
        edges = g.edges()
        self.assertEqual(edges.dtype, np.uint32)
        self.assertEqual(edges.shape, (4, 2))

        g_disc = eg.CsrGraph.from_edges(4, [(0, 1)], directed=False, sanitize=True)
        self.assertFalse(g_disc.is_connected())

        d0 = eg.dijkstra_single_source(g, 0)
        self.assertEqual(d0.dtype, np.float64)
        np.testing.assert_allclose(d0, [0.0, 1.5, 3.5, 4.5], atol=1e-5)

        ws = eg.DijkstraWorkspace(2)
        ws.ensure_capacity(4)
        self.assertIn("DijkstraWorkspace(", repr(ws))
        np.testing.assert_allclose(ws.dijkstra_from(g, 0), d0, atol=1e-12)

        ap = eg.all_pairs_dijkstra(g)
        self.assertEqual(ap.shape, (4, 4))
        np.testing.assert_allclose(ap[0], d0, atol=1e-12)
        np.testing.assert_allclose(ap, ap.T, atol=1e-12)

        self.assert_no_panic_raises((IndexError,), lambda: eg.dijkstra_single_source(g, 99))

    def test_datasets_generators_and_interop_exports(self) -> None:
        names = eg.list_datasets()
        self.assertIn("karate", names)
        catalog = eg.dataset_catalog()
        self.assertEqual(len(catalog), len(names))
        info = eg.dataset_info("karate")
        self.assertIsInstance(info, eg.DatasetInfo)
        self.assertEqual(info.name, "karate")
        self.assertEqual(info.num_nodes, 34)
        self.assertGreater(info.num_edges, 0)
        self.assertTrue(len(info.source) > 0)
        self.assertTrue(len(info.description) > 0)
        self.assertIn("DatasetInfo(", repr(info))
        self.assert_no_panic_raises((ValueError,), lambda: eg.dataset_info("nonexistent_dataset"))

        gw = eg.generate_weighted_graph(12, avg_degree=4, min_weight=0.5, max_weight=2.0, seed=7)
        self.assertEqual(gw.num_nodes, 12)
        self.assertTrue(gw.is_weighted)

        dag = eg.generate_layered_dag(4, 3, edge_prob=0.6, seed=7)
        self.assertTrue(dag.is_directed)
        self.assertEqual(dag.num_nodes, 12)
        self.assertEqual(len(eg.cycle_edges(dag)), 0)

        sbm_g, sbm_labels = eg.generate_sbm_graph([6, 6], p_intra=0.8, p_inter=0.05, seed=7)
        self.assertEqual(sbm_g.num_nodes, 12)
        self.assertEqual(sbm_labels.dtype, np.uint32)
        np.testing.assert_array_equal(sbm_labels[:6], np.zeros(6, dtype=np.uint32))
        np.testing.assert_array_equal(sbm_labels[6:], np.ones(6, dtype=np.uint32))

        tor = eg.generate_torus_grid(4, 5, triangulated=True)
        self.assertEqual(tor.num_nodes, 20)
        ico = eg.generate_spherical_icosphere(subdivisions=1)
        self.assertEqual(ico.num_nodes, 42)
        sph_g = eg.generate_spherical_grid(4, 6)
        self.assertEqual(sph_g.num_nodes, 26)
        htree = eg.generate_hyperbolic_tree(branching_factor=2, depth=3)
        self.assertGreater(htree.num_nodes, 7)

        # Interop: to_igraph + N-D / Manifold to_networkx
        ig_g = eg.to_igraph(sbm_g)
        self.assertEqual(ig_g.vcount(), 12)
        self.assertFalse(ig_g.is_directed())

        lay_nd = eg.LayoutNd.init(12, 3, seed=7)
        nx_nd = eg.to_networkx(sbm_g, layout=lay_nd)
        self.assertEqual(len(nx_nd.nodes[0]["pos"]), 3)

        lay_m = eg.ManifoldLayout2d.init("torus", 12, seed=7)
        nx_m = eg.to_networkx(sbm_g, layout=lay_m)
        self.assertEqual(len(nx_m.nodes[0]["pos"]), 2)
        for coord in nx_m.nodes[0]["pos"]:
            self.assertTrue(math.isfinite(coord))


if __name__ == "__main__":
    unittest.main()
