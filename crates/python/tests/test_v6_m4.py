from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import math
import unittest
from typing import Any, Callable

import egraph as eg
import numpy as np
import numpy.typing as npt


def _make_grid(rows: int, cols: int) -> eg.CsrGraph:
    edges: list[tuple[int, int]] = []
    for r in range(rows):
        for c in range(cols):
            u = r * cols + c
            if c + 1 < cols:
                edges.append((u, u + 1))
            if r + 1 < rows:
                edges.append((u, u + cols))
    return eg.CsrGraph.from_edges(rows * cols, edges, directed=False, sanitize=True)


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


def _seeded_coords(n: int, seed: int = 42, scale: float = 1.0) -> npt.NDArray[np.float64]:
    rng = np.random.default_rng(seed)
    return rng.uniform(-scale, scale, size=(n, 2)).astype(np.float64)


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


class TestV6M4SolversAndPostProcessing(unittest.TestCase):
    def assert_no_panic_raises(
        self,
        expected: tuple[type[BaseException], ...],
        fn: Callable[[], Any],
    ) -> None:
        with self.assertRaises(expected) as ctx:
            fn()
        self.assertNotIn("PanicException", type(ctx.exception).__name__)

    def test_t_fdp_backends_step_vs_run_and_from_knn(self) -> None:
        graph = _make_barbell(5, bridge_len=2)
        n = graph.num_nodes
        base = _seeded_coords(n, seed=11, scale=0.5)

        for dtype_str in ("float64", "float32"):
            for backend in ("ibfft", "barnes_hut", "exact"):
                l_run = eg.Layout2d.from_coords(_to_float_array(base, dtype_str))
                l_step = eg.Layout2d.from_coords(_to_float_array(base, dtype_str))

                s_run = eg.TFdpSolver(
                    graph,
                    backend=backend,
                    total_iterations=12,
                    rho_start=1.5,
                    rho_end=0.2,
                    rho_power=1.0,
                    seed=7,
                    dtype=dtype_str,
                )
                s_step = eg.TFdpSolver(
                    graph,
                    backend=backend,
                    total_iterations=12,
                    rho_start=1.5,
                    rho_end=0.2,
                    rho_power=1.0,
                    seed=7,
                    dtype=dtype_str,
                )
                self.assertEqual(s_run.status, "initialized")
                self.assertEqual(s_run.num_nodes, n)
                self.assertEqual(s_run.total_iterations, 12)
                self.assertEqual(s_run.dtype, dtype_str)
                self.assertAlmostEqual(s_run.current_rho, 1.5, places=6)
                self.assertIn("TFdpSolver", repr(s_run))

                status_run = s_run.run(l_run)
                self.assertEqual(status_run, "completed")
                self.assertTrue(s_run.is_finished)
                self.assertAlmostEqual(s_run.current_rho, 0.2, places=6)

                rhos = [s_step.current_rho]
                for _ in range(5):
                    s_step.step(l_step)
                    rhos.append(s_step.current_rho)
                self.assertEqual(s_step.status, "running")
                self.assertEqual(s_step.iteration, 5)
                s_step.run_steps(l_step, 7)
                self.assertTrue(s_step.is_finished)
                self.assertTrue(all(rhos[i] >= rhos[i + 1] for i in range(len(rhos) - 1)))

                np.testing.assert_array_equal(l_run.coords(copy=True), l_step.coords(copy=True))
                self.assertTrue(l_run.is_all_finite())

        # Both from_knn(KnnGraph) and from_knn(indptr, indices, weights=None)
        knn, _, _ = eg.build_pure_2hop_knn(graph)
        s_knn_obj = eg.TFdpSolver.from_knn(knn, backend="exact", total_iterations=8, seed=42)
        s_knn_raw = eg.TFdpSolver.from_knn(
            knn.indptr(),
            knn.indices(),
            knn.distances(),
            backend="exact",
            total_iterations=8,
            seed=42,
        )
        l1 = eg.Layout2d.from_coords(base)
        l2 = eg.Layout2d.from_coords(base)
        s_knn_obj.run(l1)
        s_knn_raw.run(l2)
        np.testing.assert_array_equal(l1.coords(copy=True), l2.coords(copy=True))

        # from_knn with weights=None
        s_knn_now = eg.TFdpSolver.from_knn(
            graph.indptr(), graph.indices(), None, backend="barnes_hut", total_iterations=6
        )
        l3 = eg.Layout2d.from_coords(base)
        s_knn_now.run(l3)
        self.assertTrue(l3.is_all_finite())

    def test_ib_umap_constructors_and_step_vs_run(self) -> None:
        graph = _make_grid(4, 4)
        n = graph.num_nodes
        base = _seeded_coords(n, seed=19, scale=0.8)

        for dtype_str in ("float64", "float32"):
            for backend in ("ibfft", "barnes_hut", "exact"):
                l_run = eg.Layout2d.from_coords(_to_float_array(base, dtype_str))
                l_step = eg.Layout2d.from_coords(_to_float_array(base, dtype_str))

                s_run = eg.IbUmapSolver(
                    graph,
                    max_2hop_per_node=20,
                    backend=backend,
                    n_epochs=10,
                    seed=123,
                    dtype=dtype_str,
                )
                s_step = eg.IbUmapSolver(
                    graph,
                    max_2hop_per_node=20,
                    backend=backend,
                    n_epochs=10,
                    seed=123,
                    dtype=dtype_str,
                )
                self.assertEqual(s_run.num_nodes, n)
                self.assertEqual(s_run.total_iterations, 10)
                self.assertEqual(s_run.dtype, dtype_str)
                self.assertAlmostEqual(s_run.current_rho, 1.0, places=6)
                self.assertIn("IbUmapSolver", repr(s_run))

                s_run.run(l_run)
                s_step.step(l_step)
                s_step.run_steps(l_step, 4)
                s_step.run(l_step)
                self.assertTrue(s_run.is_finished)
                self.assertTrue(s_step.is_finished)
                np.testing.assert_array_equal(l_run.coords(copy=True), l_step.coords(copy=True))

        knn, _, _ = eg.build_pure_2hop_knn(graph)
        probs = eg.compute_fuzzy_simplicial_set(knn)
        s_knn = eg.IbUmapSolver.from_knn(knn, backend="exact", n_epochs=6, seed=42)
        s_fset = eg.IbUmapSolver.from_fuzzy_set(probs, backend="exact", n_epochs=6, seed=42)
        l_a = eg.Layout2d.from_coords(base)
        l_b = eg.Layout2d.from_coords(base)
        s_knn.run(l_a)
        s_fset.run(l_b)
        np.testing.assert_allclose(l_a.coords(copy=True), l_b.coords(copy=True), rtol=1e-12, atol=1e-12)

    def test_sgd_m4_solvers_and_squared_schedule(self) -> None:
        graph = _make_barbell(5, bridge_len=2)
        n = graph.num_nodes
        base = _seeded_coords(n, seed=31, scale=1.0)

        for dtype_str in ("float64", "float32"):
            # 1. snap_t_fdp
            l_run = eg.Layout2d.from_coords(_to_float_array(base, dtype_str))
            l_step = eg.Layout2d.from_coords(_to_float_array(base, dtype_str))
            s1 = eg.SgdSolver.snap_t_fdp(graph, iterations=12, negatives_per_edge=3, seed=5, dtype=dtype_str)
            s2 = eg.SgdSolver.snap_t_fdp(graph, iterations=12, negatives_per_edge=3, seed=5, dtype=dtype_str)
            self.assertEqual(s1.total_iterations, 12)
            self.assertEqual(s1.dtype, dtype_str)
            self.assertIsNone(s1.stats())
            self.assertIsNone(s1.dog_stats())
            s1.run(l_run)
            s2.step(l_step)
            s2.run_steps(l_step, 5)
            s2.run(l_step)
            np.testing.assert_array_equal(l_run.coords(copy=True), l_step.coords(copy=True))

            # 2. pacmap_graph
            l_run = eg.Layout2d.from_coords(_to_float_array(base, dtype_str))
            l_step = eg.Layout2d.from_coords(_to_float_array(base, dtype_str))
            p1 = eg.SgdSolver.pacmap_graph(
                graph,
                n_neighbors=6,
                phase1_iters=4,
                phase2_iters=4,
                phase3_iters=6,
                seed=9,
                dtype=dtype_str,
            )
            p2 = eg.SgdSolver.pacmap_graph(
                graph,
                n_neighbors=6,
                phase1_iters=4,
                phase2_iters=4,
                phase3_iters=6,
                seed=9,
                dtype=dtype_str,
            )
            self.assertEqual(p1.total_iterations, 14)
            self.assertEqual(p1.dtype, dtype_str)
            p1.run(l_run)
            p2.step(l_step)
            p2.run_steps(l_step, 6)
            p2.run(l_step)
            np.testing.assert_array_equal(l_run.coords(copy=True), l_step.coords(copy=True))

            # 3. sgd2
            l_run = eg.Layout2d.from_coords(_to_float_array(base, dtype_str))
            l_step = eg.Layout2d.from_coords(_to_float_array(base, dtype_str))
            q1 = eg.SgdSolver.sgd2(graph, t_max=10, batch_size=32, seed=13, dtype=dtype_str)
            q2 = eg.SgdSolver.sgd2(graph, t_max=10, batch_size=32, seed=13, dtype=dtype_str)
            self.assertEqual(q1.total_iterations, 10)
            self.assertEqual(q1.dtype, dtype_str)
            q1.run(l_run)
            q2.step(l_step)
            q2.run_steps(l_step, 4)
            q2.run(l_step)
            np.testing.assert_array_equal(l_run.coords(copy=True), l_step.coords(copy=True))

            # 4. "squared" schedule kind on SgdSolver.full
            l_sq1 = eg.Layout2d.from_coords(_to_float_array(base, dtype_str))
            l_sq2 = eg.Layout2d.from_coords(_to_float_array(base, dtype_str))
            sq1 = eg.SgdSolver.full(graph, t_max=10, schedule="squared", seed=17, dtype=dtype_str)
            sq2 = eg.SgdSolver.full(graph, t_max=10, schedule="squared", seed=17, dtype=dtype_str)
            eta0 = sq1.current_eta
            sq1.step(l_sq1)
            eta1 = sq1.current_eta
            self.assertGreater(eta0, eta1)
            sq1.run(l_sq1)
            sq2.run(l_sq2)
            np.testing.assert_array_equal(l_sq1.coords(copy=True), l_sq2.coords(copy=True))

    def test_force_atlas2_and_yifan_hu(self) -> None:
        graph = _make_grid(5, 5)
        n = graph.num_nodes
        base = _seeded_coords(n, seed=41, scale=1.0)

        for dtype_str in ("float64", "float32"):
            for use_bh in (True, False):
                l_run = eg.Layout2d.from_coords(_to_float_array(base, dtype_str))
                l_step = eg.Layout2d.from_coords(_to_float_array(base, dtype_str))
                fa1 = eg.ForceAtlas2Solver(
                    graph,
                    iterations=15,
                    use_barnes_hut=use_bh,
                    linlog_mode=True,
                    strong_gravity=not use_bh,
                    outbound_attraction_distribution=True,
                    seed=7,
                    dtype=dtype_str,
                )
                fa2 = eg.ForceAtlas2Solver(
                    graph,
                    iterations=15,
                    use_barnes_hut=use_bh,
                    linlog_mode=True,
                    strong_gravity=not use_bh,
                    outbound_attraction_distribution=True,
                    seed=7,
                    dtype=dtype_str,
                )
                self.assertEqual(fa1.num_nodes, n)
                self.assertEqual(fa1.total_iterations, 15)
                self.assertEqual(fa1.dtype, dtype_str)
                self.assertIn("ForceAtlas2Solver", repr(fa1))
                fa1.run(l_run)
                fa2.step(l_step)
                fa2.run_steps(l_step, 5)
                fa2.run(l_step)
                np.testing.assert_array_equal(l_run.coords(copy=True), l_step.coords(copy=True))

        # YifanHu multilevel (force >= 2 levels with min_coarsen_size=6 on 25-node grid)
        for dtype_str in ("float64", "float32"):
            l_run = eg.Layout2d.from_coords(_to_float_array(base, dtype_str))
            l_step = eg.Layout2d.from_coords(_to_float_array(base, dtype_str))
            yh1 = eg.YifanHuSolver(
                graph,
                max_iterations_per_level=15,
                min_coarsen_size=6,
                min_coarsen_ratio=0.9,
                seed=11,
                dtype=dtype_str,
            )
            yh2 = eg.YifanHuSolver(
                graph,
                max_iterations_per_level=15,
                min_coarsen_size=6,
                min_coarsen_ratio=0.9,
                seed=11,
                dtype=dtype_str,
            )
            self.assertGreaterEqual(yh1.num_levels, 2)
            self.assertEqual(yh1.current_level, yh1.num_levels - 1)
            self.assertEqual(yh1.num_nodes, n)
            self.assertEqual(yh1.dtype, dtype_str)
            self.assertIn("YifanHuSolver", repr(yh1))

            yh1.run(l_run)
            yh2.step(l_step)
            yh2.run_steps(l_step, 5)
            yh2.run(l_step)
            self.assertTrue(yh1.is_finished)
            self.assertEqual(yh1.current_level, 0)
            np.testing.assert_array_equal(l_run.coords(copy=True), l_step.coords(copy=True))

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

        # project_1d on float64, float32 (in-place + returned), read-only, and list
        arr64 = np.array([1.0, 0.0, 0.5], dtype=np.float64)
        out64 = eg.project_1d(arr64, [eg.SeparationConstraint(0, 1, 2.0), eg.SeparationConstraint(1, 2, 1.5)])
        self.assertEqual(out64.dtype, np.float64)
        np.testing.assert_allclose(arr64, out64)
        self.assertGreaterEqual(float(out64[1] - out64[0]), 2.0 - 1e-9)
        self.assertGreaterEqual(float(out64[2] - out64[1]), 1.5 - 1e-9)

        arr32 = np.array([1.0, 0.0, 0.5], dtype=np.float32)
        out32 = eg.project_1d(arr32, [eg.SeparationConstraint(0, 1, 2.0)])
        self.assertEqual(out32.dtype, np.float32)
        np.testing.assert_allclose(arr32, out32)
        self.assertGreaterEqual(float(out32[1] - out32[0]), 2.0 - 1e-6)

        ro = np.array([1.0, 0.0], dtype=np.float64)
        ro.flags.writeable = False
        out_ro = eg.project_1d(ro, [eg.SeparationConstraint(0, 1, 3.0)])
        self.assertGreaterEqual(float(out_ro[1] - out_ro[0]), 3.0 - 1e-9)
        self.assertAlmostEqual(float(ro[0]), 1.0)

        out_list = eg.project_1d([2.0, 0.0], [eg.SeparationConstraint(0, 1, 1.0)])
        self.assertGreaterEqual(float(out_list[1] - out_list[0]), 1.0 - 1e-9)

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
        # 0 -> 1 -> 2 -> 3, plus 0 -> 4 -> 3 (and 4 has no incoming from 1, wait: 0->4, 4->3 vs 0->1->2->3;
        # actually 0->1->2->3->5 and 0->4->5: LongestPath puts 4 at layer 1 (span 0->4=1, 4->5=3 -> total=4),
        # while NetworkSimplex can place 4 flexibly; or chain 0->1->2->3 with 4->3 and 4->5:
        # LongestPath puts 4 at 0 (4->3 span=3!), NetworkSimplex puts 4 at 2 (4->3 span=1!))
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

    def test_fdeb_solver_step_vs_run_and_polylines(self) -> None:
        graph = _make_grid(3, 3)
        base = np.array(
            [[float(c), float(r)] for r in range(3) for c in range(3)],
            dtype=np.float64,
        )
        for dtype_str in ("float64", "float32"):
            layout = eg.Layout2d.from_coords(_to_float_array(base, dtype_str))
            f_run = eg.FdebSolver(graph, layout, cycles=3, i0=6, dtype=None)
            f_step = eg.FdebSolver(graph, layout, cycles=3, i0=6, dtype=dtype_str)

            self.assertEqual(f_run.dtype, dtype_str)
            self.assertEqual(f_run.num_edges, graph.num_edges)
            self.assertEqual(f_run.status, "initialized")
            self.assertIn("FdebSolver", repr(f_run))

            status = f_run.run()
            self.assertEqual(status, "completed")
            self.assertTrue(f_run.is_finished)

            f_step.step()
            f_step.run_steps(4)
            f_step.run()
            self.assertTrue(f_step.is_finished)

            offsets = f_run.edge_offsets()
            pts_run = f_run.points()
            pts_step = f_step.points()
            self.assertEqual(offsets.dtype, np.uint32)
            self.assertEqual(offsets.shape, (graph.num_edges + 1,))
            self.assertEqual(int(offsets[0]), 0)
            self.assertEqual(int(offsets[-1]), pts_run.shape[0])
            self.assertEqual(pts_run.ndim, 2)
            self.assertEqual(pts_run.shape[1], 2)
            self.assertEqual(pts_run.dtype, np.float32 if dtype_str == "float32" else np.float64)
            self.assertTrue(np.all(np.isfinite(pts_run)))
            np.testing.assert_array_equal(offsets, f_step.edge_offsets())
            np.testing.assert_array_equal(pts_run, pts_step)

    def test_error_handling_and_gil_release(self) -> None:
        graph = _make_grid(3, 3)
        undir = graph
        layout = eg.Layout2d.zeros(9)
        bad_layout = eg.Layout2d.zeros(5)

        # No panics on bad inputs / domain errors
        self.assert_no_panic_raises(
            (ValueError,), lambda: eg.TFdpSolver(graph, backend="invalid")
        )
        self.assert_no_panic_raises(
            (ValueError,), lambda: eg.TFdpSolver(graph, momentum=1.0)
        )
        self.assert_no_panic_raises(
            (TypeError,), lambda: eg.TFdpSolver(graph, total_iterations=True)
        )
        self.assert_no_panic_raises(
            (ValueError,), lambda: eg.TFdpSolver(graph).step(bad_layout)
        )
        self.assert_no_panic_raises(
            (ValueError,), lambda: eg.IbUmapSolver(graph, tau_scale=0.0)
        )
        self.assert_no_panic_raises(
            (ValueError,), lambda: eg.SgdSolver.snap_t_fdp(graph, rho_power=0.0)
        )
        self.assert_no_panic_raises(
            (ValueError,), lambda: eg.SgdSolver.pacmap_graph(graph, n_neighbors=0)
        )
        self.assert_no_panic_raises(
            (ValueError,), lambda: eg.SgdSolver.sgd2(graph, batch_size=0)
        )
        self.assert_no_panic_raises(
            (ValueError,), lambda: eg.ForceAtlas2Solver(graph, jitter_tolerance=0.0)
        )
        self.assert_no_panic_raises(
            (ValueError,), lambda: eg.YifanHuSolver(graph, step_ratio=1.0)
        )
        self.assert_no_panic_raises(
            (ValueError,),
            lambda: eg.PrismOverlapRemoval([1.0, 1.0], [1.0]),
        )
        self.assert_no_panic_raises(
            (ValueError,),
            lambda: eg.PrismOverlapRemoval([1.0], [-0.5]),
        )
        self.assert_no_panic_raises(
            (ValueError,), lambda: eg.SeparationConstraint(0, 1, math.nan)
        )
        self.assert_no_panic_raises(
            (ValueError,),
            lambda: eg.project_layout_axis(layout, 2, []),
        )
        self.assert_no_panic_raises(
            (ValueError,),
            lambda: eg.project_1d([0.0, 1.0], [eg.SeparationConstraint(0, 1, 1.0), eg.SeparationConstraint(1, 0, 1.0)]),
        )
        self.assert_no_panic_raises((ValueError,), lambda: eg.cycle_edges(undir))
        self.assert_no_panic_raises((ValueError,), lambda: eg.remove_cycles(undir))
        self.assert_no_panic_raises((ValueError,), lambda: eg.assign_layers(undir))
        dag = eg.CsrGraph.from_edges(3, [(0, 1), (1, 2)], directed=True)
        self.assert_no_panic_raises(
            (ValueError,), lambda: eg.assign_layers(dag, method="unknown")
        )
        self.assert_no_panic_raises(
            (ValueError,), lambda: eg.FdebSolver(graph, bad_layout)
        )
        self.assert_no_panic_raises(
            (ValueError,), lambda: eg.FdebSolver(graph, layout, cycles=30)
        )

        # GIL release concurrency smoke test
        g_big = _make_grid(6, 6)
        coords_big = _seeded_coords(36, seed=99)

        def run_worker(seed_val: int) -> bool:
            s = eg.ForceAtlas2Solver(g_big, iterations=10, seed=seed_val)
            lay = eg.Layout2d.from_coords(coords_big)
            s.run(lay)
            return lay.is_all_finite()

        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(run_worker, [1, 2]))
        self.assertEqual(results, [True, True])


if __name__ == "__main__":
    unittest.main()
