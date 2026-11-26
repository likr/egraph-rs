from __future__ import annotations

import importlib
import sys
import unittest
from typing import Any
from unittest.mock import patch

import egraph as eg
import numpy as np


def _sp() -> Any:
    return importlib.import_module("scipy.sparse")


def _nx() -> Any:
    return importlib.import_module("networkx")


def _ig() -> Any:
    return importlib.import_module("igraph")


class TestScipyInterop(unittest.TestCase):
    def test_from_and_to_scipy_csr_roundtrip(self) -> None:
        sp = _sp()
        adj = np.array(
            [
                [0.0, 2.5, 0.0, 1.0],
                [2.5, 0.0, 3.0, 0.0],
                [0.0, 3.0, 0.0, 4.0],
                [1.0, 0.0, 4.0, 0.0],
            ],
            dtype=np.float64,
        )
        mat = sp.csr_matrix(adj)
        g = eg.from_scipy_csr(mat, directed=False, sanitize=True)
        self.assertEqual(g.num_nodes, 4)
        self.assertEqual(g.num_edges, 4)
        self.assertFalse(g.is_directed)
        self.assertTrue(g.is_weighted)

        mat_back = eg.to_scipy_csr(g)
        self.assertTrue(sp.issparse(mat_back))
        np.testing.assert_allclose(mat_back.toarray(), adj)

    def test_from_scipy_other_formats_bool_and_sanitize(self) -> None:
        sp = _sp()
        # Self-loop + duplicate COO + explicit zero should be sanitized cleanly
        coo = sp.coo_matrix(
            ([1.0, 2.0, 5.0, 0.0], ([0, 0, 1, 2], [0, 1, 0, 1])),
            shape=(3, 3),
        )
        g = eg.from_scipy_csr(coo, directed=False, sanitize=True)
        self.assertEqual(g.num_nodes, 3)
        self.assertEqual(g.num_edges, 1)
        w = g.weights()
        assert w is not None
        np.testing.assert_allclose(w, np.array([5.0, 5.0]))

        # Boolean csr_array produces unweighted CsrGraph
        bool_csr = sp.csr_array(
            np.array([[False, True, False], [True, False, True], [False, True, False]])
        )
        g_bool = eg.from_scipy_csr(bool_csr)
        self.assertFalse(g_bool.is_weighted)
        self.assertEqual(g_bool.num_edges, 2)
        back_unweighted = eg.to_scipy_csr(g_bool)
        np.testing.assert_allclose(
            back_unweighted.toarray(),
            np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 0.0]]),
        )

    def test_from_scipy_csr_directed_and_empty_and_errors(self) -> None:
        sp = _sp()
        directed_csc = sp.csc_matrix([[0.0, 3.0], [0.0, 0.0]])
        gd = eg.from_scipy_csr(directed_csc, directed=True, sanitize=True)
        self.assertTrue(gd.is_directed)
        self.assertEqual(gd.num_edges, 1)
        np.testing.assert_array_equal(gd.neighbors(0), np.array([1], dtype=np.uint32))
        np.testing.assert_array_equal(gd.neighbors(1), np.array([], dtype=np.uint32))

        empty = sp.csr_matrix((0, 0), dtype=np.float64)
        g_empty = eg.from_scipy_csr(empty)
        self.assertEqual(g_empty.num_nodes, 0)
        self.assertEqual(g_empty.num_edges, 0)

        with self.assertRaises(TypeError):
            eg.from_scipy_csr(np.eye(3))
        with self.assertRaises(ValueError):
            eg.from_scipy_csr(sp.csr_matrix((2, 3), dtype=np.float64))
        with self.assertRaises(TypeError):
            eg.to_scipy_csr("not_a_graph")  # type: ignore[arg-type]


class TestNetworkxInterop(unittest.TestCase):
    def test_from_and_to_networkx_undirected_with_labels_and_layout(self) -> None:
        nx = _nx()
        G = nx.Graph()
        G.add_edge("alice", "bob", weight=2.0)
        G.add_edge("bob", "carol", weight=3.5)
        G.add_node("dave")

        csr, nodes = eg.from_networkx(G)
        self.assertEqual(nodes, ["alice", "bob", "carol", "dave"])
        self.assertEqual(csr.num_nodes, 4)
        self.assertEqual(csr.num_edges, 2)
        self.assertFalse(csr.is_directed)
        self.assertTrue(csr.is_weighted)

        layout = eg.Layout2d.from_coords([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0], [6.0, 7.0]])
        G_back = eg.to_networkx(csr, nodes, layout=layout)
        self.assertFalse(G_back.is_directed())
        self.assertEqual(list(G_back.nodes()), nodes)
        self.assertEqual(G_back.edges["alice", "bob"]["weight"], 2.0)
        self.assertEqual(G_back.edges["bob", "carol"]["weight"], 3.5)
        self.assertEqual(G_back.nodes["carol"]["pos"], (4.0, 5.0))
        self.assertEqual(G_back.nodes["dave"]["pos"], (6.0, 7.0))

    def test_from_and_to_networkx_directed_and_unweighted(self) -> None:
        nx = _nx()
        D = nx.DiGraph()
        D.add_edges_from([(10, 20), (20, 30)])
        csr, nodes = eg.from_networkx(D, weight=None)
        self.assertEqual(nodes, [10, 20, 30])
        self.assertTrue(csr.is_directed)
        self.assertFalse(csr.is_weighted)
        self.assertEqual(csr.num_edges, 2)

        D_back = eg.to_networkx(csr)
        self.assertTrue(D_back.is_directed())
        self.assertEqual(set(D_back.edges()), {(0, 1), (1, 2)})

        with self.assertRaises(ValueError):
            eg.to_networkx(csr, nodes=[0, 1])
        with self.assertRaises(ValueError):
            eg.to_networkx(csr, layout=eg.Layout2d.zeros(2))
        with self.assertRaises(TypeError):
            eg.from_networkx([[0, 1]])


class TestIgraphInterop(unittest.TestCase):
    def test_from_igraph_undirected_directed_and_weighted(self) -> None:
        ig = _ig()
        g_ig = ig.Graph(n=4, edges=[(0, 1), (1, 2), (2, 3), (3, 0)], directed=False)
        csr = eg.from_igraph(g_ig)
        self.assertEqual(csr.num_nodes, 4)
        self.assertEqual(csr.num_edges, 4)
        self.assertFalse(csr.is_directed)
        self.assertFalse(csr.is_weighted)

        dg_ig = ig.Graph(n=3, edges=[(0, 1), (1, 2)], directed=True)
        dg_ig.es["weight"] = [1.5, None]
        dg_csr = eg.from_igraph(dg_ig)
        self.assertTrue(dg_csr.is_directed)
        self.assertTrue(dg_csr.is_weighted)
        w = dg_csr.weights()
        assert w is not None
        np.testing.assert_allclose(w, np.array([1.5, 1.0]))

        with self.assertRaises(TypeError):
            eg.from_igraph({"edges": []})

    def test_from_igraph_missing_dependency_error(self) -> None:
        real_import = importlib.import_module

        def fake_import(name: str, package: str | None = None) -> Any:
            if name == "igraph":
                raise ImportError("No module named 'igraph'")
            return real_import(name, package)

        with patch.dict(sys.modules, {"igraph": None}):
            with patch("egraph.interop.importlib.import_module", side_effect=fake_import):
                with self.assertRaisesRegex(ImportError, "igraph"):
                    eg.from_igraph(object())


class TestInteropEndToEndWorkflows(unittest.TestCase):
    def test_karate_interop_pipeline_solvers_refiner_and_metrics(self) -> None:
        nx = _nx()
        G = nx.karate_club_graph()
        csr, nodes = eg.from_networkx(G, weight=None)
        self.assertEqual(csr.num_nodes, len(nodes))
        self.assertGreater(csr.num_edges, 70)

        init = eg.pivot_mds_init(csr, num_pivots=16, seed=42)
        tsnet = eg.TsNetSolver.pure_2hop(
            csr,
            backend="exact",
            iterations_stage1=10,
            iterations_stage2=10,
            iterations_stage3=10,
            seed=42,
        )
        layout_tsnet = eg.Layout2d.from_coords(init.coords(copy=True))
        tsnet.run(layout_tsnet)
        self.assertTrue(layout_tsnet.is_all_finite())

        sgd = eg.SgdSolver.sparse_barrier_dog(csr, t_max=10, seed=42)
        layout_sgd = eg.Layout2d.from_coords(init.coords(copy=True))
        sgd.run(layout_sgd)
        self.assertTrue(layout_sgd.is_all_finite())

        gumap = eg.GuMapSolver.from_graph_dog(csr, n_epochs=10, seed=42)
        layout_gumap = eg.Layout2d.from_coords(init.coords(copy=True))
        gumap.run(layout_gumap)
        self.assertTrue(layout_gumap.is_all_finite())

        refiner = eg.SpatialIntruderRefiner(csr, rounds=1, t_max=5, seed=42)
        refiner.run(layout_gumap)
        self.assertTrue(layout_gumap.is_all_finite())

        metrics = eg.evaluate_metrics(
            csr,
            layout_gumap,
            ["neighborhood_preservation_2hop", "scale_normalized_stress"],
        )
        self.assertTrue(np.isfinite(metrics["neighborhood_preservation_2hop"]))
        self.assertTrue(np.isfinite(metrics["scale_normalized_stress"]))

        G_pos = eg.to_networkx(csr, nodes, layout=layout_gumap)
        self.assertIn("pos", G_pos.nodes[nodes[0]])
        sp_mat = eg.to_scipy_csr(csr)
        self.assertEqual(sp_mat.shape, (csr.num_nodes, csr.num_nodes))


if __name__ == "__main__":
    unittest.main()
