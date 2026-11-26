"""Python checks for community detection, backbone, overlap removal, and cluster metrics."""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from typing import Any

from parity.common import (
    assert_non_panic_error,
    make_asymmetric_coords,
    make_barbell_edges,
)


def run_community_baselines_python_checks(eg: Any) -> list[dict[str, Any]]:
    """Run deterministic, f32/f64, step()==run() & adversarial checks for community/backbone/cluster APIs."""
    import numpy as np

    results: list[dict[str, Any]] = []
    t0 = time.perf_counter()

    # Deterministic 10-node 2-clique barbell graph (cliques 0..4 and 5..9 bridged by 4-5)
    g_bb = eg.CsrGraph.from_edges(10, make_barbell_edges(5), directed=False, sanitize=True)
    init10 = np.array(make_asymmetric_coords(10, scale=1.2), dtype=np.float64)

    # 1. Leiden (Modularity & CPM)
    l_mod1 = eg.leiden_communities(g_bb, objective="modularity", resolution=1.0, seed=42)
    l_mod2 = eg.leiden_communities(g_bb, objective="mod", resolution=1.0, seed=42)
    assert np.array_equal(l_mod1["labels"], l_mod2["labels"])
    assert l_mod1["num_communities"] == 2
    assert float(l_mod1["modularity"]) > 0.35
    l_cpm = eg.leiden_communities(g_bb, objective="cpm", resolution=0.3, seed=42)
    assert l_cpm["num_communities"] >= 2
    l_empty = eg.leiden_communities(eg.CsrGraph.from_edges(0, [], directed=False))
    assert l_empty["num_communities"] == 0 and len(l_empty["labels"]) == 0

    for dtype_name, np_dtype in (("float64", np.float64), ("float32", np.float32)):
        init10_t = init10.astype(np_dtype)

        # 2. LinLog, FR, FA2 prevent_overlap
        for rep_model in ("edge_repulsion", "node_repulsion"):
            sr = eg.LinLogSolver(
                g_bb, iterations=10, repulsion_model=rep_model, use_barnes_hut=True, seed=42, dtype=dtype_name
            )
            ss = eg.LinLogSolver(
                g_bb, iterations=10, repulsion_model=rep_model, use_barnes_hut=True, seed=42, dtype=dtype_name
            )
            lr = eg.Layout2d.from_coords(init10_t)
            ls = eg.Layout2d.from_coords(init10_t)
            assert sr.run(lr) == "completed"
            assert math.isfinite(sr.energy(lr))
            assert ss.run_steps(ls, 3) == "running"
            while not ss.is_finished:
                ss.step(ls)
            assert lr.is_all_finite()
            assert np.array_equal(lr.coords(copy=True), ls.coords(copy=True))

        for backend in ("exact", "barnes_hut", "grid"):
            fr = eg.FruchtermanReingoldSolver(
                g_bb, iterations=10, backend=backend, seed=42, dtype=dtype_name
            )
            fs = eg.FruchtermanReingoldSolver(
                g_bb, iterations=10, backend=backend, seed=42, dtype=dtype_name
            )
            lr = eg.Layout2d.from_coords(init10_t)
            ls = eg.Layout2d.from_coords(init10_t)
            assert fr.optimal_distance > 0.0
            assert fr.run(lr) == "completed"
            assert fr.current_temperature == 0.0
            assert fs.run_steps(ls, 3) == "running"
            while not fs.is_finished:
                fs.step(ls)
            assert lr.is_all_finite()
            assert np.array_equal(lr.coords(copy=True), ls.coords(copy=True))

        radii10 = np.full(10, 0.6, dtype=np.float64)
        far = eg.ForceAtlas2Solver(
            g_bb,
            iterations=10,
            prevent_overlap=True,
            default_node_radius=0.5,
            node_radii=radii10,
            seed=42,
            dtype=dtype_name,
        )
        fas = eg.ForceAtlas2Solver(
            g_bb,
            iterations=10,
            prevent_overlap=True,
            default_node_radius=0.5,
            node_radii=radii10,
            seed=42,
            dtype=dtype_name,
        )
        assert far.prevent_overlap is True
        assert np.allclose(far.node_radii, radii10)
        lfar = eg.Layout2d.from_coords(init10_t)
        lfas = eg.Layout2d.from_coords(init10_t)
        assert far.run(lfar) == "completed"
        fas.step(lfas)
        fas.run_steps(lfas, 4)
        fas.run(lfas)
        assert lfar.is_all_finite()
        assert np.array_equal(lfar.coords(copy=True), lfas.coords(copy=True))

        # 3. Quadrilateral Simmelian Backbone
        bb_res = eg.extract_quadrilateral_backbone(g_bb, retain_ratio=0.3)
        assert len(bb_res["backbone_edges"]) >= len(bb_res["umst_edges"])
        for mode in ("stress", "sgd"):
            br = eg.BackboneSolver(
                g_bb, retain_ratio=0.3, layout_mode=mode, iterations=10, seed=42, dtype=dtype_name
            )
            bs = eg.BackboneSolver(
                g_bb, retain_ratio=0.3, layout_mode=mode, iterations=10, seed=42, dtype=dtype_name
            )
            lr = eg.Layout2d.from_coords(init10_t)
            ls = eg.Layout2d.from_coords(init10_t)
            assert br.run(lr) == "completed"
            bs.step(ls)
            bs.run(ls)
            assert lr.is_all_finite()
            assert np.array_equal(lr.coords(copy=True), ls.coords(copy=True))

        # 4. ClusterRefine, squarified_treemap & GroupInABoxSolver
        swapped = np.array(
            [
                [0.0, 0.0],
                [0.2, 0.1],
                [0.1, -0.2],
                [0.3, 0.0],
                [10.0, 0.1],
                [10.0, 0.2],
                [10.2, 0.1],
                [10.1, -0.2],
                [10.3, 0.0],
                [0.0, 0.2],
            ],
            dtype=np_dtype,
        )
        comm_labels = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1], dtype=np.uint32)
        for cr_method in ("kmeans", "hac"):
            cr1 = eg.ClusterRefineSolver.from_graph(
                g_bb, comm_labels, method=cr_method, max_rounds=5, seed=42, dtype=dtype_name
            )
            cr2 = eg.ClusterRefineSolver.from_graph(
                g_bb, comm_labels, method=cr_method, max_rounds=5, seed=42, dtype=dtype_name
            )
            lcr1 = eg.Layout2d.from_coords(swapped)
            lcr2 = eg.Layout2d.from_coords(swapped)
            dsc_before = eg.distance_consistency(lcr1, comm_labels)
            cr1.run(lcr1)
            cr2.step(lcr2)
            cr2.run(lcr2)
            assert np.array_equal(lcr1.coords(copy=True), lcr2.coords(copy=True))
            assert eg.distance_consistency(lcr1, comm_labels) > dsc_before

            lcr_fn = eg.Layout2d.from_coords(swapped)
            moved = eg.refine_cluster_layout(
                lcr_fn, comm_labels, method=cr_method, max_rounds=5, seed=42
            )
            assert moved > 0

        for box_mode in ("treemap", "force_directed"):
            for intra_mode in ("fr", "sgd_full", "yifan_hu"):
                cw = [2.0, 1.0] if box_mode == "treemap" else None
                gr = eg.GroupInABoxSolver(
                    g_bb,
                    comm_labels,
                    box_mode=box_mode,
                    intra_mode=intra_mode,
                    intra_iterations=6,
                    fd_iterations=6,
                    cluster_weights=cw,
                    seed=42,
                    dtype=dtype_name,
                )
                gs = eg.GroupInABoxSolver(
                    g_bb,
                    comm_labels,
                    box_mode=box_mode,
                    intra_mode=intra_mode,
                    intra_iterations=6,
                    fd_iterations=6,
                    cluster_weights=cw,
                    seed=42,
                    dtype=dtype_name,
                )
                lr = eg.Layout2d.from_coords(init10_t)
                ls = eg.Layout2d.from_coords(init10_t)
                assert len(gr.boxes()) == 2
                gr.run(lr)
                gs.step(ls)
                gs.run(ls)
                assert lr.is_all_finite()
                assert np.array_equal(lr.coords(copy=True), ls.coords(copy=True))

        # 5. GTreeOverlapRemoval & ForbidOverlapRemoval
        ov_coords = (init10 * 0.15).astype(np_dtype)
        ov_coords[1] = ov_coords[0]
        hw10 = np.full(10, 0.45, dtype=np.float64)
        hh10 = np.full(10, 0.35, dtype=np.float64)
        gtr = eg.GTreeOverlapRemoval(hw10, hh10, max_iterations=25, padding=0.02, seed=42, dtype=dtype_name)
        gts = eg.GTreeOverlapRemoval(hw10, hh10, max_iterations=25, padding=0.02, seed=42, dtype=dtype_name)
        lgtr = eg.Layout2d.from_coords(ov_coords)
        lgts = eg.Layout2d.from_coords(ov_coords)
        gtr.run(lgtr)
        gts.step(lgts)
        gts.run(lgts)
        assert gtr.last_overlap_count == 0
        assert np.array_equal(lgtr.coords(copy=True), lgts.coords(copy=True))
        assert len(eg.rectangle_overlaps(lgtr, hw10, hh10)) == 0

        for variant in ("forbid", "forbid_prime", "sparse_forbid"):
            fbr = eg.ForbidOverlapRemoval(
                hw10, hh10, variant=variant, max_scale_iters=25, sgd_iterations=30, seed=42, dtype=dtype_name
            )
            fbs = eg.ForbidOverlapRemoval(
                hw10, hh10, variant=variant, max_scale_iters=25, sgd_iterations=30, seed=42, dtype=dtype_name
            )
            lfbr = eg.Layout2d.from_coords(ov_coords)
            lfbs = eg.Layout2d.from_coords(ov_coords)
            fbr.run(lfbr)
            fbs.step(lfbs)
            fbs.run(lfbs)
            assert fbr.last_overlap_count == 0
            assert np.array_equal(lfbr.coords(copy=True), lfbs.coords(copy=True))
            assert len(eg.rectangle_overlaps(lfbr, hw10, hh10)) == 0

        # 6. Cluster Baselines & NOR
        sep_coords = np.array(
            [
                [-5.0, 0.0],
                [-4.8, 0.1],
                [-5.1, -0.1],
                [-4.9, 0.2],
                [-5.0, -0.2],
                [5.0, 0.0],
                [5.2, 0.1],
                [4.9, -0.1],
                [5.1, 0.2],
                [5.0, -0.2],
            ],
            dtype=np_dtype,
        )
        l_sep = eg.Layout2d.from_coords(sep_coords)
        pa = eg.partition_agreement(comm_labels, comm_labels)
        assert abs(pa["ari"] - 1.0) < 1e-12 and abs(pa["fmi"] - 1.0) < 1e-12
        assert abs(eg.cq_kmeans(l_sep, comm_labels, seed=42)["ari"] - 1.0) < 1e-12
        assert abs(eg.cq_hac(l_sep, comm_labels)["ami"] - 1.0) < 1e-12
        assert abs(eg.distance_consistency(l_sep, comm_labels) - 1.0) < 1e-12
        assert eg.distribution_consistency(l_sep, comm_labels, grid_size=16) > 0.9
        cdm, hdm = eg.class_density_measure(l_sep, comm_labels, grid_size=32)
        assert cdm > 0.8 and abs(hdm - 1.0) < 1e-12
        assert abs(eg.sepme_knng(l_sep, comm_labels, k=2) - 1.0) < 1e-12
        assert eg.sepme_rng(l_sep, comm_labels) > 0.7
        assert eg.sepme_gab(l_sep, comm_labels) > 0.5
        assert eg.sepme_gong(l_sep, comm_labels, gamma=0.35) > 0.7
        assert abs(eg.neighborhood_hit(l_sep, comm_labels, k=3) - 1.0) < 1e-12
        assert eg.silhouette_score(l_sep, comm_labels) > 0.9
        assert eg.calinski_harabasz_score(l_sep, comm_labels) > 100.0
        assert eg.davies_bouldin_score(l_sep, comm_labels) < 0.2
        assert 0.0 <= eg.pairwise_distance_overlap_error(l_sep, comm_labels, num_bins=32) <= 1.0
        assert eg.sprawlter_nn(lgtr, hw10, hh10) >= 0.0
        nor_w = eg.nor_metrics(eg.Layout2d.from_coords(ov_coords), lgtr, hw10, hh10, graph=g_bb)
        assert nor_w["el_rsd"] >= 0.0 and math.isfinite(nor_w["oo_nni"])
        nor_wo = eg.nor_metrics(eg.Layout2d.from_coords(ov_coords), lgtr, hw10, hh10)
        assert nor_wo["el_rsd"] == 0.0

        cluster_metric_targets = [
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
        ev_cluster = eg.evaluate_metrics(
            g_bb,
            l_sep,
            cluster_metric_targets,
            community_labels=comm_labels,
            sepme_knng_k=2,
            neighborhood_hit_k=3,
            dc_grid_size=16,
            cdm_grid_size=32,
            num_bins=32,
            sprawlter_node_radius=0.5,
            seed=42,
        )
        assert set(ev_cluster.keys()) == set(cluster_metric_targets)
        ev_alias = eg.evaluate_metrics(
            g_bb, l_sep, ["sepme_knng2", "sepme-knng2"], community_labels=comm_labels, sepme_knng_k=2
        )
        assert abs(ev_alias["sepme_knng"] - ev_cluster["sepme_knng"]) < 1e-12

    rects = eg.squarified_treemap([3.0, 2.0, 1.0], width=100.0, height=50.0, x=0.0, y=0.0)
    assert rects.shape == (3, 4)

    results.append({
        "case": "community_backbone_and_cluster_metrics",
        "status": "pass",
        "elapsed_sec": round(time.perf_counter() - t0, 4),
    })

    # Adversarial community/backbone/cluster checks (never PanicException)
    t1 = time.perf_counter()
    bad_l = eg.Layout2d.zeros(5)
    l10_nan = eg.Layout2d.zeros(10)
    l10_nan.coords(copy=False)[0, 0] = float("nan")
    g_dir = eg.CsrGraph.from_edges(3, [(0, 1), (1, 2)], directed=True)
    community_bad_cases: list[tuple[str, Callable[[], Any]]] = [
        ("leiden_communities/directed", lambda: eg.leiden_communities(g_dir)),
        ("leiden_communities/bad_obj", lambda: eg.leiden_communities(g_bb, objective="bad")),
        ("leiden_communities/zero_passes", lambda: eg.leiden_communities(g_bb, max_passes=0)),
        ("LinLogSolver/bad_repulsion", lambda: eg.LinLogSolver(g_bb, repulsion_model="bad")),
        ("LinLogSolver/dim_mismatch", lambda: eg.LinLogSolver(g_bb).step(bad_l)),
        ("FruchtermanReingoldSolver/bad_backend", lambda: eg.FruchtermanReingoldSolver(g_bb, backend="bad")),
        ("FruchtermanReingoldSolver/dim_mismatch", lambda: eg.FruchtermanReingoldSolver(g_bb).step(bad_l)),
        ("ForceAtlas2Solver/neg_radius", lambda: eg.ForceAtlas2Solver(g_bb, default_node_radius=-1.0)),
        ("ForceAtlas2Solver/radii_len", lambda: eg.ForceAtlas2Solver(g_bb, node_radii=[1.0, 2.0])),
        ("extract_quadrilateral_backbone/bad_ratio", lambda: eg.extract_quadrilateral_backbone(g_bb, retain_ratio=1.5)),
        ("BackboneSolver/bad_mode", lambda: eg.BackboneSolver(g_bb, layout_mode="bad")),
        ("ClusterRefineSolver/bad_method", lambda: eg.ClusterRefineSolver([0, 1], method="bad")),
        ("squarified_treemap/empty", lambda: eg.squarified_treemap([])),
        ("GroupInABoxSolver/bad_box", lambda: eg.GroupInABoxSolver(g_bb, [0] * 10, box_mode="bad")),
        ("GTreeOverlapRemoval/len_mismatch", lambda: eg.GTreeOverlapRemoval([1.0, 1.0], [1.0])),
        ("GTreeOverlapRemoval/nan_layout", lambda: eg.GTreeOverlapRemoval.from_radii([0.5] * 10).step(l10_nan)),
        ("ForbidOverlapRemoval/bad_variant", lambda: eg.ForbidOverlapRemoval([1.0], [1.0], variant="bad")),
        ("ForbidOverlapRemoval/nan_layout", lambda: eg.ForbidOverlapRemoval.from_radii([0.5] * 10).step(l10_nan)),
        ("partition_agreement/len_mismatch", lambda: eg.partition_agreement([0, 1], [0])),
        ("sepme_knng/k0", lambda: eg.sepme_knng(l10_nan, [0] * 10, k=0)),
        ("evaluate_metrics/cluster_metrics_missing_labels", lambda: eg.evaluate_metrics(g_bb, eg.Layout2d.zeros(10), ["dsc"])),
    ]
    for label, fn in community_bad_cases:
        assert_non_panic_error(label, fn)

    results.append({
        "case": "community_backbone_adversarial_non_panic",
        "status": "pass",
        "elapsed_sec": round(time.perf_counter() - t1, 4),
    })
    return results
