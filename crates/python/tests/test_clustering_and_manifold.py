from __future__ import annotations

import inspect
import math
import unittest
from typing import Any, Callable

import egraph as eg
import numpy as np
import numpy.typing as npt


def _make_barbell(clique_size: int, bridge_len: int = 1) -> eg.CsrGraph:
    n = 2 * clique_size + bridge_len - 1
    edges: list[tuple[int, int]] = []
    for i in range(clique_size):
        for j in range(i + 1, clique_size):
            edges.append((i, j))
    offset = clique_size + bridge_len - 1
    for i in range(clique_size):
        for j in range(i + 1, clique_size):
            edges.append((offset + i, offset + j))
    prev = clique_size - 1
    for b in range(bridge_len - 1):
        cur = clique_size + b
        edges.append((prev, cur))
        prev = cur
    edges.append((prev, offset))
    return eg.CsrGraph.from_edges(n, edges, directed=False, sanitize=True)


def _to_float_array(coords: npt.NDArray[np.float64], dtype_str: str) -> npt.NDArray[np.floating[Any]]:
    if dtype_str == "float32":
        return coords.astype(np.float32)
    return coords.astype(np.float64)


class TestClustering(unittest.TestCase):
    def assert_no_panic_raises(
        self,
        expected: tuple[type[BaseException], ...],
        fn: Callable[[], Any],
    ) -> None:
        with self.assertRaises(expected) as ctx:
            fn()
        self.assertNotIn("PanicException", type(ctx.exception).__name__)

    def test_kmeans_plus_plus_determinism_and_validation(self) -> None:
        pts = np.array(
            [
                [10.0, 10.0],
                [10.2, 9.8],
                [9.8, 10.1],
                [-10.0, -10.0],
                [-9.9, -10.2],
                [-10.1, -9.8],
            ],
            dtype=np.float64,
        )
        r1 = eg.kmeans_plus_plus(pts, k=2, max_iterations=100, tol=1e-4, n_init=3, seed=42)
        r2 = eg.kmeans_plus_plus(pts.astype(np.float32), k=2, n_init=3, seed=42)
        labels1 = r1.labels
        centroids1 = r1.centroids
        self.assertEqual(labels1.dtype, np.uint32)
        self.assertEqual(centroids1.dtype, np.float64)
        self.assertEqual(centroids1.shape, (2, 2))
        np.testing.assert_array_equal(labels1, np.array([0, 0, 0, 1, 1, 1], dtype=np.uint32))
        np.testing.assert_array_equal(labels1, r2.labels)
        self.assertTrue(r1.converged)
        self.assertLess(float(r1.inertia), 1.0)
        self.assertGreaterEqual(int(r1.iterations), 1)

        # Python list input and k=1 / k=n boundary
        r_list = eg.kmeans_plus_plus([[0.0, 0.0], [1.0, 0.0]], k=1, seed=42)
        np.testing.assert_array_equal(r_list.labels, np.array([0, 0], dtype=np.uint32))
        r_kn = eg.kmeans_plus_plus([[0.0, 0.0], [1.0, 0.0]], k=2, seed=42)
        np.testing.assert_array_equal(r_kn.labels, np.array([0, 1], dtype=np.uint32))

        # Validation errors
        self.assert_no_panic_raises((ValueError,), lambda: eg.kmeans_plus_plus(pts, k=0))
        self.assert_no_panic_raises((ValueError,), lambda: eg.kmeans_plus_plus(pts, k=10))
        self.assert_no_panic_raises((ValueError,), lambda: eg.kmeans_plus_plus(pts, max_iterations=0))
        self.assert_no_panic_raises((ValueError,), lambda: eg.kmeans_plus_plus(pts, n_init=0))
        self.assert_no_panic_raises((ValueError,), lambda: eg.kmeans_plus_plus(pts, tol=-0.1))
        self.assert_no_panic_raises((ValueError,), lambda: eg.kmeans_plus_plus(np.empty((0, 2)), k=1))
        self.assert_no_panic_raises(
            (ValueError,), lambda: eg.kmeans_plus_plus(np.array([[0.0, float("nan")]]), k=1)
        )
        self.assert_no_panic_raises((TypeError,), lambda: eg.kmeans_plus_plus(pts, k=True))
        self.assert_no_panic_raises((TypeError,), lambda: eg.kmeans_plus_plus("bad"))  # type: ignore[arg-type]

    def test_spectral_clustering_barbell_and_directed_rejection(self) -> None:
        barbell = _make_barbell(6, bridge_len=1)
        res1 = eg.spectral_clustering(barbell, k=2, seed=42)
        res2 = eg.spectral_clustering(barbell, k=2, seed=42)
        l1 = res1.labels
        l2 = res2.labels
        self.assertEqual(res1.num_clusters, 2)
        self.assertEqual(l1.dtype, np.uint32)
        self.assertEqual(l1.shape, (12,))
        np.testing.assert_array_equal(l1, l2)
        np.testing.assert_array_equal(l1[:6], np.zeros(6, dtype=np.uint32))
        np.testing.assert_array_equal(l1[6:], np.ones(6, dtype=np.uint32))

        # Larger barbell (> 48 nodes) exercises Lanczos path
        large_barbell = _make_barbell(30, bridge_len=1)
        l_large = eg.spectral_clustering(large_barbell, k=2, seed=42).labels
        self.assertEqual(l_large.shape, (60,))
        np.testing.assert_array_equal(l_large[:30], np.zeros(30, dtype=np.uint32))
        np.testing.assert_array_equal(l_large[30:], np.ones(30, dtype=np.uint32))

        # Directed graph raises ValueError cleanly
        g_dir = eg.CsrGraph.from_edges(4, [(0, 1), (1, 2), (2, 3)], directed=True, sanitize=True)
        self.assert_no_panic_raises((ValueError,), lambda: eg.spectral_clustering(g_dir, k=2))
        self.assert_no_panic_raises((ValueError,), lambda: eg.spectral_clustering(barbell, k=0))
        self.assert_no_panic_raises((ValueError,), lambda: eg.spectral_clustering(barbell, eigen_tol=0.0))
        self.assert_no_panic_raises((TypeError,), lambda: eg.spectral_clustering(barbell, k=True))

    def test_louvain_and_label_propagation(self) -> None:
        barbell = _make_barbell(6, bridge_len=1)
        louv1 = eg.louvain_communities(barbell, max_passes=100, resolution=1.0, min_modularity_gain=1e-7, seed=42)
        louv2 = eg.louvain_communities(barbell, seed=42)
        self.assertEqual(louv1.num_communities, 2)
        self.assertGreater(float(louv1.modularity), 0.4)
        self.assertGreaterEqual(int(louv1.passes), 1)
        self.assertEqual(louv1.labels.dtype, np.uint32)
        np.testing.assert_array_equal(louv1.labels, np.array([0] * 6 + [1] * 6, dtype=np.uint32))
        np.testing.assert_array_equal(louv1.labels, louv2.labels)

        lp_res1 = eg.label_propagation(barbell, max_iterations=100, seed=42)
        lp_res2 = eg.label_propagation(barbell, max_iterations=100, seed=42)
        lp1 = lp_res1.labels
        lp2 = lp_res2.labels
        self.assertEqual(lp1.dtype, np.uint32)
        np.testing.assert_array_equal(lp1, lp2)
        np.testing.assert_array_equal(lp1[:6], np.zeros(6, dtype=np.uint32))
        np.testing.assert_array_equal(lp1[6:], np.ones(6, dtype=np.uint32))

        g_dir = eg.CsrGraph.from_edges(3, [(0, 1), (1, 2)], directed=True, sanitize=True)
        self.assert_no_panic_raises((ValueError,), lambda: eg.louvain_communities(g_dir))
        self.assert_no_panic_raises((ValueError,), lambda: eg.label_propagation(g_dir))
        self.assert_no_panic_raises((ValueError,), lambda: eg.louvain_communities(barbell, max_passes=0))
        self.assert_no_panic_raises((ValueError,), lambda: eg.louvain_communities(barbell, resolution=-1.0))
        self.assert_no_panic_raises((ValueError,), lambda: eg.label_propagation(barbell, max_iterations=0))

    def test_coarsen_and_project_clustered_layout_f64_and_f32(self) -> None:
        edges = [(0, 1), (1, 2), (0, 2), (3, 4), (4, 5), (3, 5), (0, 3), (2, 4)]
        g = eg.CsrGraph.from_edges(6, edges, directed=False, sanitize=True)
        # Non-contiguous labels are compacted canonically
        coarse_g, n2c, sizes, num_clusters = eg.coarsen_graph(g, [100, 100, 100, 999, 999, 999])
        self.assertEqual(num_clusters, 2)
        self.assertEqual(coarse_g.num_nodes, 2)
        self.assertEqual(coarse_g.num_edges, 1)
        w = coarse_g.weights()
        assert w is not None
        np.testing.assert_allclose(w, [2.0, 2.0])
        self.assertEqual(n2c.dtype, np.uint32)
        self.assertEqual(sizes.dtype, np.uint32)
        np.testing.assert_array_equal(n2c, np.array([0, 0, 0, 1, 1, 1], dtype=np.uint32))
        np.testing.assert_array_equal(sizes, np.array([3, 3], dtype=np.uint32))

        base_coarse = np.array([[-5.0, 1.0], [5.0, -1.0]], dtype=np.float64)
        for dtype_str, np_dtype in (("float64", np.float64), ("float32", np.float32)):
            coarse_coords = _to_float_array(base_coarse, dtype_str)
            cl = eg.Layout2d.from_coords(coarse_coords)
            f1 = eg.project_clustered_layout(cl, n2c, spread_radius=0.5, seed=42)
            f2 = eg.project_clustered_layout(cl, n2c, spread_radius=0.5, seed=42)
            self.assertEqual(f1.dtype, dtype_str)
            self.assertEqual(len(f1), 6)
            self.assertTrue(f1.is_all_finite())
            c1 = f1.coords(copy=True)
            self.assertEqual(c1.dtype, np_dtype)
            np.testing.assert_array_equal(c1, f2.coords(copy=True))
            for i in range(6):
                cid = int(n2c[i])
                dist = math.hypot(float(c1[i, 0] - coarse_coords[cid, 0]), float(c1[i, 1] - coarse_coords[cid, 1]))
                self.assertLessEqual(dist, 0.5 + 1e-5)

            exact = eg.project_clustered_layout(cl, n2c, spread_radius=0.0, seed=42)
            np.testing.assert_allclose(exact.coords(copy=True)[:3], np.tile(coarse_coords[0], (3, 1)))
            np.testing.assert_allclose(exact.coords(copy=True)[3:], np.tile(coarse_coords[1], (3, 1)))

        # Errors
        g_dir = eg.CsrGraph.from_edges(2, [(0, 1)], directed=True, sanitize=True)
        self.assert_no_panic_raises((ValueError,), lambda: eg.coarsen_graph(g_dir, [0, 1]))
        self.assert_no_panic_raises((ValueError,), lambda: eg.coarsen_graph(g, [0, 1]))
        cl2 = eg.Layout2d.from_coords([[0.0, 0.0], [1.0, 1.0]])
        self.assert_no_panic_raises((IndexError,), lambda: eg.project_clustered_layout(cl2, [0, 2]))
        self.assert_no_panic_raises(
            (ValueError,), lambda: eg.project_clustered_layout(cl2, [0, 1], spread_radius=-0.1)
        )
        self.assert_no_panic_raises(
            (ValueError,), lambda: eg.project_clustered_layout(cl2, [0, 1], spread_radius=float("nan"))
        )


class TestManifoldLayout2d(unittest.TestCase):
    def assert_no_panic_raises(
        self,
        expected: tuple[type[BaseException], ...],
        fn: Callable[[], Any],
    ) -> None:
        with self.assertRaises(expected) as ctx:
            fn()
        self.assertNotIn("PanicException", type(ctx.exception).__name__)

    def test_manifold_layout2d_all_manifolds_f64_and_f32(self) -> None:
        for raw_kind, canonical in (
            ("hyperbolic", "hyperbolic"),
            ("hyperbolic_poincare", "hyperbolic"),
            ("poincare", "hyperbolic"),
            ("spherical", "spherical"),
            ("sphere", "spherical"),
            ("torus", "torus"),
            ("torus-2d", "torus"),
        ):
            mz = eg.ManifoldLayout2d.zeros(raw_kind, 4, dtype="float32")
            self.assertEqual(mz.kind, canonical)
            self.assertEqual(mz.dtype, "float32")
            self.assertEqual(len(mz), 4)
            self.assertFalse(mz.is_empty())
            self.assertIn("ManifoldLayout2d(", repr(mz))

        # Hyperbolic Poincare clamping + geodesic distance
        mh = eg.ManifoldLayout2d.from_coords("hyperbolic", [[0.0, 0.0], [3.0, 4.0]])
        ch = mh.coords()
        self.assertEqual(ch.dtype, np.float64)
        self.assertLess(math.hypot(float(ch[1, 0]), float(ch[1, 1])), 1.0)
        mh2 = eg.ManifoldLayout2d.from_coords("hyperbolic", np.array([[0.0, 0.0], [0.5, 0.0]], dtype=np.float32))
        self.assertEqual(mh2.dtype, "float32")
        self.assertAlmostEqual(mh2.geodesic_distance(0, 1), 2.0 * math.atanh(0.5), places=5)

        # Spherical projection + equator quarter-circle distance
        ms = eg.ManifoldLayout2d.from_coords("spherical", [[-0.5, -0.25], [math.pi / 2.0, math.pi / 2.0]])
        cs = ms.coords()
        self.assertTrue(0.0 <= float(cs[0, 0]) < 2.0 * math.pi)
        self.assertTrue(0.0 <= float(cs[0, 1]) <= math.pi)
        ms_eq = eg.ManifoldLayout2d.from_coords(
            "spherical", [[0.2, math.pi / 2.0], [0.2 + math.pi / 2.0, math.pi / 2.0]]
        )
        self.assertAlmostEqual(ms_eq.geodesic_distance(0, 1), math.pi / 2.0, places=10)

        # Torus periodic wrapping + geodesic distance
        l_torus_in = eg.Layout2d.from_coords([[1.95, -0.95], [0.05, 0.95]])
        mt = eg.ManifoldLayout2d.from_layout("torus", l_torus_in)
        mt.project_all()
        ct = mt.coords()
        self.assertTrue(np.all((ct >= 0.0) & (ct < 1.0)))
        self.assertAlmostEqual(mt.geodesic_distance(0, 1), math.hypot(0.1, 0.1), places=10)
        back = mt.to_layout()
        self.assertEqual(back.dtype, "float64")
        np.testing.assert_allclose(back.coords(copy=True), ct)

        # Errors
        self.assert_no_panic_raises((ValueError,), lambda: eg.ManifoldLayout2d.zeros("euclidean", 3))
        self.assert_no_panic_raises((TypeError,), lambda: eg.ManifoldLayout2d.zeros(123, 3))  # type: ignore[arg-type]
        self.assert_no_panic_raises(
            (ValueError,), lambda: eg.ManifoldLayout2d.from_coords("torus", [[0.0, float("nan")]])
        )
        self.assert_no_panic_raises((IndexError,), lambda: mt.geodesic_distance(0, 99))
        self.assert_no_panic_raises((TypeError,), lambda: eg.ManifoldLayout2d.zeros("torus", True))

        # Poincare Mobius centroid centering & safe initialization (f64 and f32)
        ax, ay = eg.poincare_mobius_add(-0.3, 0.4, 0.3, -0.4)
        self.assertLess(math.hypot(ax, ay), 1e-12)
        for dtype_str in ("float64", "float32"):
            mr = eg.ManifoldLayout2d.random_init("hyperbolic", 8, scale=0.25, seed=42, dtype=dtype_str)
            self.assertEqual(mr.dtype, dtype_str)
            mr_coords = mr.coords()
            self.assertTrue(np.all(np.hypot(mr_coords[:, 0], mr_coords[:, 1]) <= 0.25 + 1e-5))

            m_off = eg.ManifoldLayout2d.from_coords(
                "hyperbolic",
                np.array([[0.6, 0.4], [0.7, 0.3], [0.55, 0.5]], dtype=np.float32 if dtype_str == "float32" else np.float64),
            )
            d_before = m_off.geodesic_distance(0, 1)
            m_off.center_poincare_centroid()
            cx, cy = m_off.poincare_centroid()
            self.assertLess(math.hypot(cx, cy), 1e-10 if dtype_str == "float64" else 1e-4)
            self.assertAlmostEqual(m_off.geodesic_distance(0, 1), d_before, places=9 if dtype_str == "float64" else 3)
            m_off.recenter_poincare_at(2)
            self.assertLess(math.hypot(m_off.x(2), m_off.y(2)), 1e-10 if dtype_str == "float64" else 1e-4)

        big_euclid = eg.Layout2d.from_coords([[10.0, -5.0], [20.0, 5.0], [30.0, 15.0]])
        m_scaled = eg.ManifoldLayout2d.from_layout_scaled("hyperbolic", big_euclid, max_radius=0.7)
        r_scaled = np.hypot(m_scaled.coords()[:, 0], m_scaled.coords()[:, 1])
        self.assertAlmostEqual(float(np.max(r_scaled)), 0.7, places=10)

        # Non-Poincare manifolds reject Poincare-only methods
        self.assert_no_panic_raises((ValueError,), lambda: mt.poincare_centroid())
        self.assert_no_panic_raises((ValueError,), lambda: mt.recenter_poincare(0.1, 0.1))
        self.assert_no_panic_raises((ValueError,), lambda: mt.recenter_poincare_at(0))
        self.assert_no_panic_raises((ValueError,), lambda: mt.center_poincare_centroid())

    def test_clustering_and_solver_exports_docstrings_and_signatures(self) -> None:
        exported_names = (
            "ManifoldLayout2d",
            "TsNetSolver",
            "SgdSolver",
            "GuMapSolver",
            "SpatialIntruderRefiner",
            "StressMajorizationSolver",
            "KamadaKawaiSolver",
            "kmeans_plus_plus",
            "spectral_clustering",
            "louvain_communities",
            "label_propagation",
            "coarsen_graph",
            "project_clustered_layout",
        )
        for name in exported_names:
            obj = getattr(eg, name)
            self.assertIs(obj, getattr(eg._core, name))
            self.assertIn(name, eg.__all__)
            self.assertTrue((obj.__doc__ or "").strip(), f"missing docstring on {name}")

        sig_km = inspect.signature(eg.kmeans_plus_plus)
        self.assertEqual(sig_km.parameters["k"].default, 2)
        self.assertEqual(sig_km.parameters["seed"].default, 42)
        sig_ref = inspect.signature(eg.SpatialIntruderRefiner)
        self.assertEqual(sig_ref.parameters["iterations"].default, 4)
        self.assertEqual(sig_ref.parameters["hinge_fp"].default, True)


if __name__ == "__main__":
    unittest.main()
