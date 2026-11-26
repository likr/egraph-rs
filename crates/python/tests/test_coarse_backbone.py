from __future__ import annotations

import math
import unittest

import egraph as eg
import numpy as np


def _three_cluster_chain(big_size: int = 24) -> eg.CsrGraph:
    small = 6
    n = 2 * small + big_size
    edges: list[tuple[int, int]] = []
    for i in range(small):
        for j in range(i + 1, small):
            edges.append((i, j))
    for i in range(big_size):
        for j in range(i + 1, big_size):
            edges.append((small + i, small + j))
    c0 = small + big_size
    for i in range(small):
        for j in range(i + 1, small):
            edges.append((c0 + i, c0 + j))
    edges.append((small - 1, small))
    edges.append((small + big_size - 1, c0))
    return eg.CsrGraph.from_edges(n, edges, directed=False, sanitize=True)


def _grid_graph(rows: int, cols: int) -> eg.CsrGraph:
    edges: list[tuple[int, int]] = []
    for r in range(rows):
        for c in range(cols):
            u = r * cols + c
            if r + 1 < rows:
                edges.append((u, (r + 1) * cols + c))
            if c + 1 < cols:
                edges.append((u, r * cols + (c + 1)))
    return eg.CsrGraph.from_edges(rows * cols, edges, directed=False, sanitize=True)


class TestCoarseBackbone(unittest.TestCase):
    def test_louvain_coarsen_hierarchy_and_disconnected_floor(self) -> None:
        edges: list[tuple[int, int]] = []
        for c in range(4):
            off = c * 5
            for i in range(5):
                for j in range(i + 1, 5):
                    edges.append((off + i, off + j))
        edges.extend([(4, 5), (9, 10), (14, 15)])
        g = eg.CsrGraph.from_edges(20, edges, directed=False, sanitize=True)

        h1 = eg.louvain_coarsen(g, max_coarse_nodes=3, seed=42)
        h2 = eg.louvain_coarsen(g, max_coarse_nodes=3, seed=42)
        self.assertLessEqual(h1["num_clusters"], 3)
        self.assertGreaterEqual(h1["num_levels"], 1)
        np.testing.assert_array_equal(h1["node_to_coarse"], h2["node_to_coarse"])
        np.testing.assert_array_equal(h1["coarse_sizes"], h2["coarse_sizes"])
        self.assertEqual(int(np.sum(h1["coarse_sizes"])), 20)

        # Disconnected 4-triangle graph stops cleanly at 4 components.
        disc_edges: list[tuple[int, int]] = []
        for c in range(4):
            off = c * 3
            disc_edges.extend([(off, off + 1), (off + 1, off + 2), (off, off + 2)])
        disc = eg.CsrGraph.from_edges(12, disc_edges, directed=False, sanitize=True)
        hd = eg.louvain_coarsen(disc, max_coarse_nodes=2, seed=42)
        self.assertEqual(hd["num_clusters"], 4)
        self.assertEqual(hd["coarse_graph"].num_edges, 0)

    def test_centroid_alignment_and_radii_projection_f64_f32(self) -> None:
        for dtype in ("float64", "float32"):
            coarse = eg.Layout2d.from_coords(
                np.array([[5.0, -3.0], [-5.0, 2.0]], dtype=np.float64 if dtype == "float64" else np.float32)
            )
            n2c = np.array([0, 0, 0, 0, 0, 1], dtype=np.uint32)
            fine = eg.Layout2d.from_coords(
                np.array(
                    [[0.0, 0.0], [2.5, 0.25], [5.0, 0.0], [7.5, -0.25], [10.0, 0.0], [0.0, 0.0]],
                    dtype=np.float64 if dtype == "float64" else np.float32,
                )
            )
            eg.align_clustered_centroids(fine, coarse, n2c, gamma_tether=1.0)
            self.assertEqual(fine.dtype, dtype)
            coords = fine.coords(copy=True)
            self.assertAlmostEqual(float(np.mean(coords[:5, 0])), 5.0, places=5)
            self.assertAlmostEqual(float(np.mean(coords[:5, 1])), -3.0, places=5)
            span_x = float(np.max(coords[:5, 0]) - np.min(coords[:5, 0]))
            span_y = float(np.max(coords[:5, 1]) - np.min(coords[:5, 1]))
            self.assertGreater(span_x / span_y, 10.0)

            proj = eg.project_clustered_layout_with_radii(
                coarse, n2c, [2.0, 4.0], jitter_scale=0.5, seed=7
            )
            self.assertEqual(proj.dtype, dtype)
            p_coords = proj.coords(copy=True)
            dists = np.hypot(p_coords[:5, 0] - 5.0, p_coords[:5, 1] + 3.0)
            self.assertTrue(bool(np.all(dists <= 1.0 + 1e-5)))

    def test_pivoted_rhgh_landmarks_and_coarse_backbone(self) -> None:
        path = eg.CsrGraph.from_edges(10, [(i, i + 1) for i in range(9)], directed=False, sanitize=True)
        lm, st = eg.compute_pivoted_rhgh_landmarks(path, num_landmarks=10, min_degree=1, num_probes=32, seed=42)
        self.assertTrue(st["all_converged"])
        d_adj = lm.alt_distances([0, 0], [1, 9])
        self.assertGreater(float(d_adj[1]), float(d_adj[0]))

        chain = _three_cluster_chain(24)
        bb1 = eg.coarse_backbone_init(
            chain,
            max_coarse_nodes=3,
            c0=0.8,
            kappa=1.0,
            mu_ov=2.0,
            m_gap=0.1,
            edge_length_mode="conductance_inverse",
            distance_backend="shortest_path",
            target_std=None,
            seed=42,
        )
        self.assertEqual(bb1["stats"]["num_coarse_nodes"], 3)
        self.assertEqual(bb1["stats"]["coarse_overlap_pairs_final"], 0)
        sizes = bb1["coarse_sizes"]
        b_id = int(np.argmax(sizes))
        ac_ids = [i for i in range(3) if i != b_id]
        cc = bb1["coarse_layout"].coords(copy=True)
        dist_ac = math.hypot(
            float(cc[ac_ids[0], 0] - cc[ac_ids[1], 0]),
            float(cc[ac_ids[0], 1] - cc[ac_ids[1], 1]),
        )
        self.assertGreaterEqual(dist_ac, 2.0 * float(bb1["radii"][b_id]))

        # ResistanceCurvatureFlow + RHGH + float32 determinism check.
        rcf1 = eg.coarse_backbone_init(
            chain,
            max_coarse_nodes=3,
            edge_length_mode="resistance_curvature_flow",
            distance_backend="rhgh",
            dtype="float32",
            seed=42,
        )
        rcf2 = eg.coarse_backbone_init(
            chain,
            max_coarse_nodes=3,
            edge_length_mode="resistance_curvature_flow",
            distance_backend="rhgh",
            dtype="float32",
            seed=42,
        )
        self.assertEqual(rcf1["fine_layout"].dtype, "float32")
        np.testing.assert_array_equal(
            rcf1["fine_layout"].coords(copy=True),
            rcf2["fine_layout"].coords(copy=True),
        )

    def test_degree_weight_beta_and_pacmap_backbone_fem_grid(self) -> None:
        star = eg.CsrGraph.from_edges(17, [(0, i) for i in range(1, 17)], directed=False, sanitize=True)
        sgd_beta = eg.SgdSolver.sparse_barrier_dog(
            star,
            num_landmarks=4,
            min_degree=1,
            jl_dim=8,
            degree_weight_beta=0.5,
            t_max=10,
            seed=42,
        )
        l_star = eg.Layout2d.init(17, seed=42)
        sgd_beta.run(l_star)
        self.assertTrue(l_star.is_all_finite())

        fem = _grid_graph(8, 8)
        solver, layout, bb = eg.SgdSolver.pacmap_graph_with_backbone(
            fem,
            n_neighbors=12,
            phase1_iters=80,
            phase2_iters=80,
            phase3_iters=150,
            num_landmarks=24,
            max_coarse_nodes=16,
            coarse_t_max=60,
            target_std=5.0,
            seed=42,
        )
        self.assertLessEqual(bb["stats"]["num_coarse_nodes"], 16)
        solver.run(layout)
        self.assertTrue(layout.is_all_finite())
        np2 = eg.neighborhood_preservation_2hop(fem, layout)
        self.assertGreater(np2, 0.70)

    def test_adversarial_errors_no_panic(self) -> None:
        g = _grid_graph(3, 3)
        with self.assertRaises(ValueError):
            eg.louvain_coarsen(g, max_coarse_nodes=0)
        with self.assertRaises(ValueError):
            eg.compute_pivoted_rhgh_landmarks(g, gamma=-1.0)
        with self.assertRaises(ValueError):
            eg.coarse_backbone_init(g, edge_length_mode="bad_mode")
        with self.assertRaises(ValueError):
            eg.coarse_backbone_init(g, distance_backend="bad_backend")
        with self.assertRaises(ValueError):
            eg.SgdSolver.sparse_barrier_dog(g, degree_weight_beta=-0.5)
        with self.assertRaises(ValueError):
            eg.SgdSolver.pacmap_graph(g, num_landmarks=0)

    def test_collinear_two_cluster_normalization_and_zero_weight_rcf(self) -> None:
        # Two-clique 20-node graph connected by bridge edge (9, 10).
        edges = (
            [(i, j) for i in range(10) for j in range(i + 1, 10)]
            + [(10 + i, 10 + j) for i in range(10) for j in range(i + 1, 10)]
            + [(9, 10)]
        )
        g = eg.CsrGraph.from_edges(20, edges, directed=False, sanitize=True)
        for dtype, tol in (("float64", 1e-12), ("float32", 1e-5)):
            res = eg.coarse_backbone_init(
                g,
                max_coarse_nodes=2,
                jitter_scale=0.0,
                target_std=1.0,
                dtype=dtype,
                seed=42,
            )
            sc = float(res["stats"]["scale_factor_applied"])
            self.assertGreater(abs(sc - 1.0), 1e-3)
            fl = res["fine_layout"].coords(copy=True)
            cl = res["coarse_layout"].coords(copy=True)
            n2c = np.asarray(res["node_to_coarse"])
            for cid in (0, 1):
                diff = float(np.linalg.norm(fl[n2c == cid].mean(axis=0) - cl[cid]))
                self.assertLess(diff, tol)

        # Explicit 0.0 inter-cluster weight guarded in ResistanceCurvatureFlow mode.
        rcf_edges = (
            [(i, j) for i in range(4) for j in range(i + 1, 4)]
            + [(4 + i, 4 + j) for i in range(4) for j in range(i + 1, 4)]
            + [(8 + i, 8 + j) for i in range(4) for j in range(i + 1, 4)]
            + [(3, 4), (7, 8)]
        )
        rcf_weights = [1.0] * (len(rcf_edges) - 1) + [0.0]
        g_zero = eg.CsrGraph.from_edges(
            12, rcf_edges, weights=rcf_weights, directed=False, sanitize=True
        )
        res_rcf = eg.coarse_backbone_init(
            g_zero,
            max_coarse_nodes=3,
            edge_length_mode="resistance_curvature_flow",
            seed=42,
        )
        self.assertTrue(res_rcf["fine_layout"].is_all_finite())
        self.assertTrue(res_rcf["coarse_layout"].is_all_finite())


if __name__ == "__main__":
    unittest.main()
