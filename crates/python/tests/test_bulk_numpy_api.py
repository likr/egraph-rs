import unittest
import numpy as np
import egraph as eg


def _build_cycle_graph_manual(n=5):
    g = eg.Graph()
    for _ in range(n):
        g.add_node(None)
    for u in range(n):
        g.add_edge(u, (u + 1) % n, None)
    return g


def _build_path_digraph_manual(n=4):
    g = eg.DiGraph()
    for _ in range(n):
        g.add_node(None)
    for u in range(n - 1):
        g.add_edge(u, u + 1, None)
    return g


def _run_sgd_steps(sgd, drawing, seed=42, t_max=30, eps=0.01):
    rng = eg.Rng.seed_from(seed)
    scheduler = sgd.scheduler(t_max, eps)

    def step(eta):
        sgd.shuffle(rng)
        sgd.apply(drawing, eta)

    scheduler.run(step)
    return drawing.to_numpy()


class TestGraphBulkNumpy(unittest.TestCase):
    def test_graph_from_edges_numpy_dtypes_and_parity(self):
        edges_list = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0), (0, 2)]
        manual = eg.Graph()
        for _ in range(5):
            manual.add_node(None)
        for u, v in edges_list:
            manual.add_edge(u, v, None)
        expected_dm = eg.all_sources_dijkstra(manual, lambda _: 1.0).to_numpy()

        for dtype in (np.uintp, np.int64, np.int32, np.uint32):
            edges = np.array(edges_list, dtype=dtype)
            g = eg.Graph.from_edges_numpy(5, edges)
            self.assertIsInstance(g, eg.Graph)
            self.assertEqual(g.node_count(), 5)
            self.assertEqual(g.edge_count(), len(edges_list))
            for i in range(5):
                self.assertIsNone(g.node_weight(i))
            for e in range(len(edges_list)):
                self.assertIsNone(g.edge_weight(e))
            dm = eg.all_sources_dijkstra(g).to_numpy()
            np.testing.assert_allclose(dm, expected_dm)

    def test_digraph_from_edges_numpy_and_non_contiguous(self):
        base = np.array(
            [
                [0, 1],
                [99, 99],
                [1, 2],
                [88, 88],
                [2, 3],
            ],
            dtype=np.int64,
        )
        sliced = base[::2, :]
        self.assertFalse(sliced.flags["C_CONTIGUOUS"])

        dg = eg.DiGraph.from_edges_numpy(4, sliced)
        self.assertIsInstance(dg, eg.DiGraph)
        self.assertEqual(dg.node_count(), 4)
        self.assertEqual(dg.edge_count(), 3)
        self.assertTrue(dg.contains_edge(0, 1))
        self.assertFalse(dg.contains_edge(1, 0))

        manual = _build_path_digraph_manual(4)
        np.testing.assert_allclose(
            eg.all_sources_dijkstra(dg).to_numpy(),
            eg.all_sources_dijkstra(manual).to_numpy(),
        )

    def test_from_edges_numpy_empty(self):
        empty = np.empty((0, 2), dtype=np.int64)
        g = eg.Graph.from_edges_numpy(4, empty)
        self.assertEqual(g.node_count(), 4)
        self.assertEqual(g.edge_count(), 0)

    def test_add_edges_from_numpy_on_graph_and_digraph(self):
        g = eg.Graph()
        for _ in range(4):
            g.add_node("node")
        g.add_edges_from_numpy(np.array([[0, 1], [1, 2], [2, 3]], dtype=np.int64))
        self.assertEqual(g.edge_count(), 3)
        self.assertEqual(g.node_weight(0), "node")
        self.assertIsNone(g.edge_weight(0))
        g.add_edges_from_numpy(np.array([[3, 0]], dtype=np.uintp))
        self.assertEqual(g.edge_count(), 4)

        dg = eg.DiGraph()
        for _ in range(3):
            dg.add_node(None)
        dg.add_edges_from_numpy(np.array([[0, 1], [1, 2]], dtype=np.int32))
        self.assertEqual(dg.edge_count(), 2)
        self.assertTrue(dg.contains_edge(0, 1))
        self.assertFalse(dg.contains_edge(1, 0))

    def test_graph_bulk_validation_errors(self):
        with self.assertRaises(ValueError):
            eg.Graph.from_edges_numpy(3, np.array([[0, 1, 2]], dtype=np.int64))
        with self.assertRaises(ValueError):
            eg.Graph.from_edges_numpy(3, np.array([[0, 3]], dtype=np.int64))
        with self.assertRaises(ValueError):
            eg.Graph.from_edges_numpy(3, np.array([[-1, 1]], dtype=np.int64))
        with self.assertRaises(TypeError):
            eg.Graph.from_edges_numpy(3, np.array([[0.0, 1.0]], dtype=np.float64))

        g = eg.Graph()
        for _ in range(3):
            g.add_node(None)
        g.add_edge(0, 1, None)
        with self.assertRaises(ValueError):
            g.add_edges_from_numpy(np.array([[1, 2], [2, 5]], dtype=np.int64))
        self.assertEqual(g.edge_count(), 1)


class TestDrawingAndDistanceMatrixNumpy(unittest.TestCase):
    def test_drawing_from_to_numpy_roundtrip_and_non_contiguous(self):
        g = _build_cycle_graph_manual(4)
        coords = np.array([[0.5, -1.0], [2.5, 3.0], [-4.0, 1.25], [1.0, 0.0]], dtype=np.float64)
        drawing = eg.DrawingEuclidean2d.from_numpy(g, coords)
        self.assertEqual(drawing.len(), 4)
        for i in range(4):
            self.assertAlmostEqual(drawing.x(i), coords[i, 0])
            self.assertAlmostEqual(drawing.y(i), coords[i, 1])
        np.testing.assert_allclose(drawing.to_numpy(), coords)

        # Non-contiguous slice
        fortran_coords = np.asfortranarray(coords)
        drawing_f = eg.DrawingEuclidean2d.from_numpy(g, fortran_coords)
        np.testing.assert_allclose(drawing_f.to_numpy(), coords)

        # DiGraph compatibility
        dg = _build_path_digraph_manual(4)
        drawing_dg = eg.DrawingEuclidean2d.from_numpy(dg, coords)
        np.testing.assert_allclose(drawing_dg.to_numpy(), coords)

        with self.assertRaises(ValueError):
            eg.DrawingEuclidean2d.from_numpy(g, np.zeros((4, 3), dtype=np.float64))
        with self.assertRaises(ValueError):
            eg.DrawingEuclidean2d.from_numpy(g, np.zeros((3, 2), dtype=np.float64))

    def test_distance_matrix_full_roundtrip(self):
        g = _build_cycle_graph_manual(4)
        arr = np.array(
            [
                [0.0, 1.0, 2.0, 1.0],
                [1.0, 0.0, 1.0, 2.0],
                [2.0, 1.0, 0.0, 1.0],
                [1.0, 2.0, 1.0, 0.0],
            ],
            dtype=np.float64,
        )
        dm = eg.DistanceMatrix.from_numpy(g, arr)
        self.assertEqual(dm.shape(), (4, 4))
        self.assertIsNone(dm.pivots())
        for u in range(4):
            for v in range(4):
                self.assertAlmostEqual(dm.get(u, v), arr[u, v])
        np.testing.assert_allclose(dm.to_numpy(), arr)

        with self.assertRaises(ValueError):
            eg.DistanceMatrix.from_numpy(g, np.zeros((3, 4), dtype=np.float64))

    def test_distance_matrix_pivoted_roundtrip_and_errors(self):
        g = _build_cycle_graph_manual(5)
        pivots = [1, 3]
        arr = np.array(
            [
                [1.0, 0.0, 1.0, 2.0, 2.0],
                [2.0, 2.0, 1.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        dm = eg.DistanceMatrix.from_pivoted_numpy(g, pivots, arr)
        self.assertEqual(dm.shape(), (2, 5))
        self.assertEqual(dm.pivots(), pivots)
        for row_idx, pivot_node in enumerate(pivots):
            for v in range(5):
                self.assertAlmostEqual(dm.get(pivot_node, v), arr[row_idx, v])
                self.assertAlmostEqual(dm.get_by_index(row_idx, v), arr[row_idx, v])
        np.testing.assert_allclose(dm.to_numpy(), arr)

        with self.assertRaises(ValueError):
            eg.DistanceMatrix.from_pivoted_numpy(g, pivots, arr.T.copy())
        with self.assertRaises(ValueError):
            eg.DistanceMatrix.from_pivoted_numpy(g, [1, 1], arr)
        with self.assertRaises(ValueError):
            eg.DistanceMatrix.from_pivoted_numpy(g, [1, 9], arr)


class TestOptionalEdgeLengthFastPath(unittest.TestCase):
    def test_shortest_path_optional_edge_length(self):
        g = eg.Graph.from_edges_numpy(
            4, np.array([[0, 1], [1, 2], [2, 3], [0, 3]], dtype=np.int64)
        )
        dg = eg.DiGraph.from_edges_numpy(
            4, np.array([[0, 1], [1, 2], [2, 3]], dtype=np.int64)
        )

        for fn in (eg.all_sources_dijkstra, eg.warshall_floyd):
            for graph in (g, dg):
                ref_one = fn(graph, lambda _: 1.0).to_numpy()
                np.testing.assert_allclose(fn(graph).to_numpy(), ref_one)
                np.testing.assert_allclose(fn(graph, None).to_numpy(), ref_one)
                np.testing.assert_allclose(fn(graph, 1.0).to_numpy(), ref_one)

                ref_const = fn(graph, lambda _: 2.5).to_numpy()
                np.testing.assert_allclose(fn(graph, 2.5).to_numpy(), ref_const)
                np.testing.assert_allclose(fn(graph, 2).to_numpy(), fn(graph, lambda _: 2.0).to_numpy())

                ref_cb = fn(graph, lambda e: float(e + 1)).to_numpy()
                self.assertFalse(np.allclose(ref_cb, ref_one))

            with self.assertRaises(TypeError):
                fn(g, True)
            with self.assertRaises(TypeError):
                fn(g, "invalid")

    def test_full_sgd_and_stress_majorization_optional_edge_length(self):
        g = _build_cycle_graph_manual(5)
        coords = np.array(
            [[0.0, 0.0], [1.5, 0.2], [1.2, 1.4], [0.1, 1.8], [-0.8, 0.9]],
            dtype=np.float64,
        )

        builder = eg.FullSgd()
        d_default = _run_sgd_steps(
            builder.build(g), eg.DrawingEuclidean2d.from_numpy(g, coords)
        )
        d_none = _run_sgd_steps(
            builder.build(g, None), eg.DrawingEuclidean2d.from_numpy(g, coords)
        )
        d_one = _run_sgd_steps(
            builder.build(g, 1.0), eg.DrawingEuclidean2d.from_numpy(g, coords)
        )
        d_cb_one = _run_sgd_steps(
            builder.build(g, lambda _: 1.0), eg.DrawingEuclidean2d.from_numpy(g, coords)
        )
        np.testing.assert_allclose(d_default, d_cb_one)
        np.testing.assert_allclose(d_none, d_cb_one)
        np.testing.assert_allclose(d_one, d_cb_one)

        d_two = _run_sgd_steps(
            builder.build(g, 2.0), eg.DrawingEuclidean2d.from_numpy(g, coords)
        )
        d_cb_two = _run_sgd_steps(
            builder.build(g, lambda _: 2.0), eg.DrawingEuclidean2d.from_numpy(g, coords)
        )
        np.testing.assert_allclose(d_two, d_cb_two)

        d_cb_var = _run_sgd_steps(
            builder.build(g, lambda e: 1.0 + 0.5 * e),
            eg.DrawingEuclidean2d.from_numpy(g, coords),
        )
        self.assertFalse(np.allclose(d_cb_var, d_cb_one))

        with self.assertRaises(TypeError):
            builder.build(g, False)

        # StressMajorization.__new__ with optional f
        draw_sm_def = eg.DrawingEuclidean2d.from_numpy(g, coords)
        draw_sm_none = eg.DrawingEuclidean2d.from_numpy(g, coords)
        draw_sm_const = eg.DrawingEuclidean2d.from_numpy(g, coords)
        draw_sm_cb = eg.DrawingEuclidean2d.from_numpy(g, coords)

        sm_def = eg.StressMajorization(g, draw_sm_def)
        sm_none = eg.StressMajorization(g, draw_sm_none, None)
        sm_const = eg.StressMajorization(g, draw_sm_const, 1.0)
        sm_cb = eg.StressMajorization(g, draw_sm_cb, lambda _: 1.0)

        for _ in range(5):
            s_def = sm_def.apply(draw_sm_def)
            s_none = sm_none.apply(draw_sm_none)
            s_const = sm_const.apply(draw_sm_const)
            s_cb = sm_cb.apply(draw_sm_cb)
            self.assertAlmostEqual(s_def, s_cb)
            self.assertAlmostEqual(s_none, s_cb)
            self.assertAlmostEqual(s_const, s_cb)
        np.testing.assert_allclose(draw_sm_def.to_numpy(), draw_sm_cb.to_numpy())
        np.testing.assert_allclose(draw_sm_none.to_numpy(), draw_sm_cb.to_numpy())
        np.testing.assert_allclose(draw_sm_const.to_numpy(), draw_sm_cb.to_numpy())

        draw_sm_2 = eg.DrawingEuclidean2d.from_numpy(g, coords)
        draw_sm_cb_2 = eg.DrawingEuclidean2d.from_numpy(g, coords)
        sm_2 = eg.StressMajorization(g, draw_sm_2, 2.5)
        sm_cb_2 = eg.StressMajorization(g, draw_sm_cb_2, lambda _: 2.5)
        self.assertAlmostEqual(sm_2.apply(draw_sm_2), sm_cb_2.apply(draw_sm_cb_2))
        np.testing.assert_allclose(draw_sm_2.to_numpy(), draw_sm_cb_2.to_numpy())

        with self.assertRaises(TypeError):
            eg.StressMajorization(g, draw_sm_2, True)


class TestBulkSgdAndStressMajorizationNumpy(unittest.TestCase):
    def test_full_sgd_build_with_numpy_and_filtering(self):
        g = _build_cycle_graph_manual(4)
        coords = np.array([[0.0, 0.0], [2.0, 0.1], [2.1, 2.0], [0.2, 2.2]], dtype=np.float64)
        d = np.array(
            [
                [0.0, 1.0, 2.0, np.inf],
                [1.0, 0.0, 1.0, np.nan],
                [2.0, 1.0, 0.0, 1.0],
                [np.inf, np.nan, 1.0, 0.0],
            ],
            dtype=np.float64,
        )
        w = np.array(
            [
                [0.0, 1.0, 0.25, 1.0],
                [1.0, 0.0, 1.0, 1.0],
                [0.25, 1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0, 0.0],
            ],
            dtype=np.float64,
        )
        builder = eg.FullSgd()
        sgd_bulk = builder.build_with_numpy(4, d, w)
        out_bulk = _run_sgd_steps(sgd_bulk, eg.DrawingEuclidean2d.from_numpy(g, coords))
        self.assertTrue(np.all(np.isfinite(out_bulk)))

        # Only surviving finite non-zero pairs in (i < j): (0,1), (0,2), (1,2)
        sgd_pairs = eg.Sgd.from_numpy_pairs(
            np.array([0, 0, 1], dtype=np.int64),
            np.array([1, 2, 2], dtype=np.int64),
            np.array([1.0, 2.0, 1.0], dtype=np.float64),
            np.array([1.0, 0.25, 1.0], dtype=np.float64),
        )
        out_pairs = _run_sgd_steps(sgd_pairs, eg.DrawingEuclidean2d.from_numpy(g, coords))
        np.testing.assert_allclose(out_bulk, out_pairs, atol=1e-12)

        # Default w=None -> w = 1 / d^2
        d_full = np.array(
            [
                [0.0, 1.0, 2.0, 1.0],
                [1.0, 0.0, 1.0, 2.0],
                [2.0, 1.0, 0.0, 1.0],
                [1.0, 2.0, 1.0, 0.0],
            ],
            dtype=np.float64,
        )
        sgd_default_w = builder.build_with_numpy(4, d_full)
        w_explicit = np.zeros_like(d_full)
        mask = d_full > 0
        w_explicit[mask] = 1.0 / (d_full[mask] ** 2)
        sgd_explicit_inv_d2 = builder.build_with_numpy(4, d_full, w_explicit)
        np.testing.assert_allclose(
            _run_sgd_steps(sgd_default_w, eg.DrawingEuclidean2d.from_numpy(g, coords)),
            _run_sgd_steps(sgd_explicit_inv_d2, eg.DrawingEuclidean2d.from_numpy(g, coords)),
            atol=1e-12,
        )

        with self.assertRaises(ValueError):
            builder.build_with_numpy(4, np.zeros((3, 4), dtype=np.float64))
        with self.assertRaises(ValueError):
            builder.build_with_numpy(
                4, d_full, np.zeros((4, 3), dtype=np.float64)
            )

    def test_sgd_from_numpy_pairs_and_bulk_updates(self):
        g = _build_cycle_graph_manual(4)
        coords = np.array([[0.0, 0.0], [1.8, 0.3], [1.7, 1.9], [0.1, 2.1]], dtype=np.float64)
        d0 = eg.all_sources_dijkstra(g).to_numpy()
        w0 = np.zeros_like(d0)
        mask0 = d0 > 0
        w0[mask0] = 1.0 / (d0[mask0] ** 2)

        # uintp and int64 sources/targets
        src = np.array([0, 0, 0, 1, 1, 2], dtype=np.uintp)
        tgt = np.array([1, 2, 3, 2, 3, 3], dtype=np.int64)
        dist_1d = np.array([d0[i, j] for i, j in zip(src, tgt)], dtype=np.float64)
        wght_1d = np.array([w0[i, j] for i, j in zip(src, tgt)], dtype=np.float64)

        sgd_np = eg.Sgd.from_numpy_pairs(src, tgt, dist_1d, wght_1d)
        sgd_cb = eg.Sgd.from_numpy_pairs(src, tgt, dist_1d, wght_1d)

        new_w = np.array(
            [
                [0.0, 2.0, 0.5, 1.5],
                [2.0, 0.0, 2.0, 0.5],
                [0.5, 2.0, 0.0, 2.0],
                [1.5, 0.5, 2.0, 0.0],
            ],
            dtype=np.float64,
        )
        new_d = np.array(
            [
                [0.0, 1.2, 2.4, 1.1],
                [1.2, 0.0, 1.3, 2.1],
                [2.4, 1.3, 0.0, 1.4],
                [1.1, 2.1, 1.4, 0.0],
            ],
            dtype=np.float64,
        )

        sgd_np.update_weight_from_numpy(new_w)
        sgd_cb.update_weight(lambda i, j, _dij, _wij: float(new_w[i, j]))
        np.testing.assert_allclose(
            _run_sgd_steps(sgd_np, eg.DrawingEuclidean2d.from_numpy(g, coords)),
            _run_sgd_steps(sgd_cb, eg.DrawingEuclidean2d.from_numpy(g, coords)),
            atol=1e-12,
        )

        sgd_np.update_distance_from_numpy(new_d, new_w * 0.5)
        sgd_cb.update_distance(lambda i, j, _dij, _wij: float(new_d[i, j]))
        sgd_cb.update_weight(lambda i, j, _dij, _wij: float(new_w[i, j] * 0.5))
        np.testing.assert_allclose(
            _run_sgd_steps(sgd_np, eg.DrawingEuclidean2d.from_numpy(g, coords)),
            _run_sgd_steps(sgd_cb, eg.DrawingEuclidean2d.from_numpy(g, coords)),
            atol=1e-12,
        )

        with self.assertRaises(ValueError):
            eg.Sgd.from_numpy_pairs(
                src[:-1], tgt, dist_1d, wght_1d
            )
        with self.assertRaises(ValueError):
            sgd_np.update_weight_from_numpy(np.zeros((3, 4), dtype=np.float64))
        with self.assertRaises(ValueError):
            sgd_np.update_distance_from_numpy(np.zeros((3, 4), dtype=np.float64))
        with self.assertRaises(ValueError):
            sgd_np.update_distance_from_numpy(new_d, np.zeros((3, 3), dtype=np.float64))

    def test_stress_majorization_update_weight_from_numpy(self):
        g = _build_cycle_graph_manual(5)
        coords = np.array(
            [[0.0, 0.0], [1.4, 0.1], [1.6, 1.5], [0.2, 1.9], [-0.6, 1.0]],
            dtype=np.float64,
        )
        draw_np = eg.DrawingEuclidean2d.from_numpy(g, coords)
        draw_cb = eg.DrawingEuclidean2d.from_numpy(g, coords)

        sm_np = eg.StressMajorization(g, draw_np)
        sm_cb = eg.StressMajorization(g, draw_cb)

        w = np.full((5, 5), 0.25, dtype=np.float64)
        for u in range(5):
            w[u, u] = 0.0
            v = (u + 1) % 5
            w[u, v] = 3.0
            w[v, u] = 3.0

        sm_np.update_weight_from_numpy(w)
        sm_cb.update_weight(lambda i, j, _dij, _wij: float(w[i, j]))

        for _ in range(10):
            s_np = sm_np.apply(draw_np)
            s_cb = sm_cb.apply(draw_cb)
            self.assertAlmostEqual(s_np, s_cb, places=10)
        np.testing.assert_allclose(draw_np.to_numpy(), draw_cb.to_numpy(), atol=1e-10)

        with self.assertRaises(ValueError):
            sm_np.update_weight_from_numpy(np.zeros((4, 5), dtype=np.float64))


if __name__ == "__main__":
    unittest.main()
