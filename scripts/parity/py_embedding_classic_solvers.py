"""Python checks for HDE, SGD presets, MaxentStress, GRIP, GEM, SEL, PFS', and SFC."""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from typing import Any

from parity.common import (
    assert_non_panic_error,
    bind_call,
    make_asymmetric_coords,
    make_barbell_edges,
    make_grid_edges,
    make_grid_star_graph_spec,
)


def run_embedding_classic_solvers_python_checks(eg: Any) -> list[dict[str, Any]]:
    """Run deterministic, f32/f64, step()==run(), invariant & adversarial checks for embedding/classic solvers."""
    import numpy as np

    results: list[dict[str, Any]] = []
    t0 = time.perf_counter()

    # Graph fixtures: asymmetric 70-node graph, 10-node 2-clique barbell, 4x4 grid, 10-node star.
    n_asym, asym_edges = make_grid_star_graph_spec()
    g_asym = eg.CsrGraph.from_edges(n_asym, asym_edges, directed=False, sanitize=True)
    g_bb = eg.CsrGraph.from_edges(10, make_barbell_edges(5), directed=False, sanitize=True)
    g_grid = eg.CsrGraph.from_edges(16, make_grid_edges(4, 4), directed=False, sanitize=True)
    g_star = eg.CsrGraph.from_edges(10, [(0, i) for i in range(1, 10)], directed=False, sanitize=True)

    init10 = np.array(make_asymmetric_coords(10, scale=1.2), dtype=np.float64)
    init_asym = np.array(make_asymmetric_coords(n_asym, scale=1.2), dtype=np.float64)

    for dtype_name, np_dtype in (("float64", np.float64), ("float32", np.float32)):
        init10_t = init10.astype(np_dtype)
        init_asym_t = init_asym.astype(np_dtype)

        # 1. HDE & HDE-ND
        for norm in (True, False):
            lh1 = eg.hde(g_asym, dimensions=16, normalize=norm, seed=42, dtype=dtype_name)
            lh2 = eg.hde(g_asym, dimensions=16, normalize=norm, seed=42, dtype=dtype_name)
            assert lh1.dtype == dtype_name and lh1.is_all_finite()
            ch1 = lh1.coords(copy=True)
            assert ch1.shape == (n_asym, 2) and ch1.dtype == np_dtype
            assert np.array_equal(ch1, lh2.coords(copy=True))

        lhp = eg.hde(g_asym, pivots=[0, 5, 10, 15], seed=42, dtype=dtype_name)
        assert lhp.is_all_finite() and lhp.coords(copy=False).shape == (n_asym, 2)

        for out_dims in (2, 3):
            nd1 = eg.hde_nd(
                g_asym, dimensions=16, output_dims=out_dims, seed=42, dtype=dtype_name
            )
            nd2 = eg.hde_nd(
                g_asym, dimensions=16, output_dims=out_dims, seed=42, dtype=dtype_name
            )
            assert nd1.num_nodes == n_asym and nd1.dims == out_dims and nd1.is_all_finite()
            cnd1 = nd1.coords(copy=True)
            assert cnd1.shape == (n_asym, out_dims) and cnd1.dtype == np_dtype
            assert np.array_equal(cnd1, nd2.coords(copy=True))

        # 2. Ortmann Sparse Stress & LargeVis / DRGraph / Graph-CNE
        sgd_cases: list[tuple[str, Callable[[], Any]]] = [
            (
                "ortmann_kmeans_sp",
                bind_call(
                    eg.SgdSolver.ortmann_sparse_stress,
                    g_bb, num_pivots=4, pivot_strategy="kmeans_sp", iterations=10, seed=42, dtype=dtype_name,
                ),
            ),
            (
                "ortmann_maxmin_random_sp",
                bind_call(
                    eg.SgdSolver.ortmann_sparse_stress,
                    g_bb, num_pivots=4, pivot_strategy="maxmin_random_sp", iterations=10, seed=42, dtype=dtype_name,
                ),
            ),
            (
                "ortmann_maxmin_sp",
                bind_call(
                    eg.SgdSolver.ortmann_sparse_stress,
                    g_bb, num_pivots=4, pivot_strategy="maxmin_sp", iterations=10, seed=42, dtype=dtype_name,
                ),
            ),
            (
                "largevis",
                bind_call(eg.SgdSolver.largevis, g_bb, iterations=10, negative_samples=3, seed=42, dtype=dtype_name),
            ),
            (
                "drgraph_k1",
                bind_call(eg.SgdSolver.drgraph, g_bb, k_order=1, negative_samples=3, iterations=10, seed=42, dtype=dtype_name),
            ),
            (
                "drgraph_k2",
                bind_call(eg.SgdSolver.drgraph, g_bb, k_order=2, negative_samples=3, iterations=10, seed=42, dtype=dtype_name),
            ),
            (
                "graph_cne_infonce",
                bind_call(
                    eg.SgdSolver.graph_cne,
                    g_bb, objective="infonce", negative_samples=3, iterations=10, seed=42, dtype=dtype_name,
                ),
            ),
            (
                "graph_cne_neg_tsne",
                bind_call(
                    eg.SgdSolver.graph_cne,
                    g_bb, objective="neg_tsne", negative_samples=3, iterations=10, seed=42, dtype=dtype_name,
                ),
            ),
        ]
        for label, mk_sgd in sgd_cases:
            sr1 = mk_sgd()
            sr2 = mk_sgd()
            ss = mk_sgd()
            lr1 = eg.Layout2d.from_coords(init10_t)
            lr2 = eg.Layout2d.from_coords(init10_t)
            ls = eg.Layout2d.from_coords(init10_t)
            assert sr1.run(lr1) == "completed", label
            assert sr2.run(lr2) == "completed", label
            ss.step(ls)
            ss.run_steps(ls, 3)
            ss.run(ls)
            assert sr1.is_finished and ss.is_finished and lr1.is_all_finite()
            assert np.array_equal(lr1.coords(copy=True), lr2.coords(copy=True)), label
            assert np.array_equal(lr1.coords(copy=True), ls.coords(copy=True)), label

        # 3. MaxentStressSolver
        for use_bh in (True, False):
            mr1 = eg.MaxentStressSolver(
                g_asym, q=0.5, stages=2, steps_per_stage=4, use_barnes_hut=use_bh, seed=42, dtype=dtype_name
            )
            mr2 = eg.MaxentStressSolver(
                g_asym, q=0.5, stages=2, steps_per_stage=4, use_barnes_hut=use_bh, seed=42, dtype=dtype_name
            )
            ms = eg.MaxentStressSolver(
                g_asym, q=0.5, stages=2, steps_per_stage=4, use_barnes_hut=use_bh, seed=42, dtype=dtype_name
            )
            assert abs(mr1.effective_q - 0.5) < 1e-12
            assert abs(mr1.current_alpha - 1.0) < 1e-12
            lr1 = eg.Layout2d.from_coords(init_asym_t)
            lr2 = eg.Layout2d.from_coords(init_asym_t)
            ls = eg.Layout2d.from_coords(init_asym_t)
            assert mr1.run(lr1) in ("completed", "converged")
            assert mr2.run(lr2) in ("completed", "converged")
            assert mr1.current_alpha < 1.0
            ms.step(ls)
            ms.run_steps(ls, 3)
            ms.run(ls)
            assert mr1.is_finished and ms.is_finished and lr1.is_all_finite()
            assert np.array_equal(lr1.coords(copy=True), lr2.coords(copy=True))
            assert np.array_equal(lr1.coords(copy=True), ls.coords(copy=True))

        # 4. GRIP, GEM, SEL
        gr1 = eg.GripSolver(g_bb, rounds_min=3, rounds_max=8, seed=42, dtype=dtype_name)
        gr2 = eg.GripSolver(g_bb, rounds_min=3, rounds_max=8, seed=42, dtype=dtype_name)
        gs = eg.GripSolver(g_bb, rounds_min=3, rounds_max=8, seed=42, dtype=dtype_name)
        assert gr1.num_levels >= 1
        lgr1 = eg.Layout2d.from_coords(init10_t)
        lgr2 = eg.Layout2d.from_coords(init10_t)
        lgs = eg.Layout2d.from_coords(init10_t)
        gr1.run(lgr1)
        gr2.run(lgr2)
        gs.step(lgs)
        gs.run_steps(lgs, 2)
        gs.run(lgs)
        assert gr1.is_finished and gs.is_finished and lgr1.is_all_finite()
        assert np.array_equal(lgr1.coords(copy=True), lgr2.coords(copy=True))
        assert np.array_equal(lgr1.coords(copy=True), lgs.coords(copy=True))

        gm1 = eg.GemSolver(g_bb, max_iterations=12, seed=42, dtype=dtype_name)
        gm2 = eg.GemSolver(g_bb, max_iterations=12, seed=42, dtype=dtype_name)
        gms = eg.GemSolver(g_bb, max_iterations=12, seed=42, dtype=dtype_name)
        assert eg.GemSolver(g_bb, dtype=dtype_name).total_iterations == min(4 * 10, 400)
        lgm1 = eg.Layout2d.from_coords(init10_t)
        lgm2 = eg.Layout2d.from_coords(init10_t)
        lgms = eg.Layout2d.from_coords(init10_t)
        gm1.run(lgm1)
        gm2.run(lgm2)
        gms.step(lgms)
        gms.run_steps(lgms, 3)
        gms.run(lgms)
        assert gm1.is_finished and gms.is_finished and lgm1.is_all_finite()
        assert np.array_equal(lgm1.coords(copy=True), lgm2.coords(copy=True))
        assert np.array_equal(lgm1.coords(copy=True), lgms.coords(copy=True))

        for mapping in ("sigmoid", "linear", "tanh", "custom_logistic", "binary"):
            for backend in ("fr", "t_fdp"):
                sel1 = eg.SelLayoutSolver(
                    g_bb, mapping=mapping, layout_backend=backend, iterations=8, seed=42, dtype=dtype_name
                )
                sel2 = eg.SelLayoutSolver(
                    g_bb, mapping=mapping, layout_backend=backend, iterations=8, seed=42, dtype=dtype_name
                )
                sels = eg.SelLayoutSolver(
                    g_bb, mapping=mapping, layout_backend=backend, iterations=8, seed=42, dtype=dtype_name
                )
                assert math.isfinite(sel1.se_target)
                mw = sel1.modulated_weights()
                assert mw.dtype == np.float64 and mw.ndim == 1 and np.all(np.isfinite(mw))
                lsel1 = eg.Layout2d.from_coords(init10_t)
                lsel2 = eg.Layout2d.from_coords(init10_t)
                lsels = eg.Layout2d.from_coords(init10_t)
                sel1.run(lsel1)
                sel2.run(lsel2)
                sels.step(lsels)
                sels.run_steps(lsels, 2)
                sels.run(lsels)
                assert sel1.is_finished and sels.is_finished and lsel1.is_all_finite()
                assert np.array_equal(lsel1.coords(copy=True), lsel2.coords(copy=True))
                assert np.array_equal(lsel1.coords(copy=True), lsels.coords(copy=True))

        # 5. PFS' Overlap Removal (0 overlaps & oo_nni == 0)
        ov_coords = (init10 * 0.15).astype(np_dtype)
        ov_coords[1] = ov_coords[0]
        hw10 = np.full(10, 0.45, dtype=np.float64)
        hh10 = np.full(10, 0.35, dtype=np.float64)
        radii10 = np.full(10, 0.50, dtype=np.float64)

        pf1 = eg.PfsPrimeOverlapRemoval(hw10, hh10, padding=0.02, seed=42, dtype=dtype_name)
        pf2 = eg.PfsPrimeOverlapRemoval(hw10, hh10, padding=0.02, seed=42, dtype=dtype_name)
        pfs = eg.PfsPrimeOverlapRemoval(hw10, hh10, padding=0.02, seed=42, dtype=dtype_name)
        lpf1 = eg.Layout2d.from_coords(ov_coords)
        lpf2 = eg.Layout2d.from_coords(ov_coords)
        lpfs = eg.Layout2d.from_coords(ov_coords)
        pf1.run(lpf1)
        pf2.run(lpf2)
        pfs.step(lpfs)
        assert pf1.is_finished and pfs.is_finished
        assert pf1.last_overlap_count == 0
        assert len(eg.rectangle_overlaps(lpf1, hw10, hh10)) == 0
        assert np.array_equal(lpf1.coords(copy=True), lpf2.coords(copy=True))
        assert np.array_equal(lpf1.coords(copy=True), lpfs.coords(copy=True))

        col_coords = np.column_stack([
            np.linspace(-1.0, 1.0, 10, dtype=np.float64),
            np.zeros(10, dtype=np.float64),
        ]).astype(np_dtype)
        l_col_orig = eg.Layout2d.from_coords(col_coords)
        l_col_adj = eg.Layout2d.from_coords(col_coords)
        pfr = eg.PfsPrimeOverlapRemoval.from_radii(radii10, padding=0.0, seed=42, dtype=dtype_name)
        pfr.run(l_col_adj)
        assert pfr.last_overlap_count == 0
        assert len(eg.rectangle_overlaps(l_col_adj, radii10, radii10)) == 0
        nor_pfs = eg.nor_metrics(l_col_orig, l_col_adj, radii10, radii10, graph=g_bb)
        assert abs(float(nor_pfs["oo_nni"])) < 1e-12

        # 6. SFC Layout Solver (0 co-location)
        for curve in ("hilbert", "morton"):
            sfc1 = eg.SfcLayoutSolver(g_bb, curve=curve, scale=80.0, seed=42, dtype=dtype_name)
            sfc2 = eg.SfcLayoutSolver(g_bb, curve=curve, scale=80.0, seed=42, dtype=dtype_name)
            sfcs = eg.SfcLayoutSolver(g_bb, curve=curve, scale=80.0, seed=42, dtype=dtype_name)
            ord1 = sfc1.vertex_order()
            assert ord1.dtype == np.uint32 and sorted(ord1.tolist()) == list(range(10))
            lsfc1 = eg.Layout2d.zeros(10, dtype=dtype_name)
            lsfc2 = eg.Layout2d.zeros(10, dtype=dtype_name)
            lsfcs = eg.Layout2d.zeros(10, dtype=dtype_name)
            sfc1.run(lsfc1)
            sfc2.run(lsfc2)
            sfcs.step(lsfcs)
            assert sfc1.is_finished and sfcs.is_finished and lsfc1.is_all_finite()
            assert np.array_equal(lsfc1.coords(copy=True), lsfc2.coords(copy=True))
            assert np.array_equal(lsfc1.coords(copy=True), lsfcs.coords(copy=True))
            assert eg.layout_degeneracy(g_bb, lsfc1)["duplicate_nodes_f32"] == 0

    # Auto-q invariant on MaxentStressSolver
    assert abs(eg.MaxentStressSolver(g_grid).effective_q - 0.0) < 1e-12
    assert abs(eg.MaxentStressSolver(g_star).effective_q - 0.8) < 1e-12

    # Degenerate graphs across all embedding/classic APIs (N=0, N=1, edgeless, disconnected+isolated, reciprocal directed)
    g_empty = eg.CsrGraph.from_edges(0, [], directed=False)
    g_single = eg.CsrGraph.from_edges(1, [], directed=False)
    g_edgeless = eg.CsrGraph.from_edges(5, [], directed=False)
    g_disc = eg.CsrGraph.from_edges(6, [(0, 1), (1, 2), (0, 2), (3, 4)], directed=False, sanitize=True)
    g_dir_rec = eg.CsrGraph.from_edges(4, [(0, 1), (1, 0), (1, 2), (2, 1), (2, 3)], directed=True, sanitize=False)
    for g_deg in (g_empty, g_single, g_edgeless, g_disc, g_dir_rec):
        m = g_deg.num_nodes
        assert eg.hde(g_deg, dimensions=4, seed=42).is_all_finite()
        assert eg.hde_nd(g_deg, dimensions=4, output_dims=3, seed=42).is_all_finite()
        deg_init = (
            np.column_stack([np.linspace(-1.0, 1.0, m), np.zeros(m)])
            if m > 0
            else np.zeros((0, 2), dtype=np.float64)
        )
        for solver in (
            eg.SgdSolver.ortmann_sparse_stress(g_deg, num_pivots=3, iterations=4, seed=42),
            eg.SgdSolver.largevis(g_deg, iterations=4, seed=42),
            eg.SgdSolver.drgraph(g_deg, iterations=4, seed=42),
            eg.SgdSolver.graph_cne(g_deg, iterations=4, seed=42),
            eg.MaxentStressSolver(g_deg, stages=2, steps_per_stage=3, seed=42),
            eg.GripSolver(g_deg, rounds_min=2, rounds_max=4, seed=42),
            eg.GemSolver(g_deg, max_iterations=4, seed=42),
            eg.SelLayoutSolver(g_deg, iterations=4, seed=42),
            eg.PfsPrimeOverlapRemoval.from_radii(np.full(m, 0.25, dtype=np.float64), seed=42),
            eg.SfcLayoutSolver(g_deg, seed=42),
        ):
            lay = eg.Layout2d.from_coords(deg_init)
            solver.run(lay)
            assert lay.is_all_finite()

    results.append({
        "case": "embedding_and_classic_solvers",
        "status": "pass",
        "elapsed_sec": round(time.perf_counter() - t0, 4),
    })

    # Adversarial invalid-input non-panic checks
    t1 = time.perf_counter()
    bad_lay = eg.Layout2d.zeros(10)
    bad_lay.coords(copy=False)[0, 0] = float("nan")
    wrong_lay = eg.Layout2d.zeros(3)
    hw10_ok = np.full(10, 0.45, dtype=np.float64)
    hh10_ok = np.full(10, 0.35, dtype=np.float64)
    classic_bad_cases: list[tuple[str, Callable[[], Any], tuple[type[BaseException], ...]]] = [
        ("hde/dimensions_0", lambda: eg.hde(g_bb, dimensions=0), (ValueError,)),
        ("hde_nd/output_dims_0", lambda: eg.hde_nd(g_bb, output_dims=0), (ValueError,)),
        ("hde/tol_nan", lambda: eg.hde(g_bb, tol=float("nan")), (ValueError,)),
        ("hde/tol_0", lambda: eg.hde(g_bb, tol=0.0), (ValueError,)),
        ("hde/max_sweeps_0", lambda: eg.hde(g_bb, max_sweeps=0), (ValueError,)),
        ("hde/empty_pivots", lambda: eg.hde(g_bb, pivots=[]), (ValueError,)),
        ("hde/oob_pivot", lambda: eg.hde(g_bb, pivots=[999]), (IndexError,)),
        ("hde/bool_seed", lambda: eg.hde(g_bb, seed=True), (TypeError,)),
        ("ortmann/num_pivots_0", lambda: eg.SgdSolver.ortmann_sparse_stress(g_bb, num_pivots=0), (ValueError,)),
        ("ortmann/bad_strategy", lambda: eg.SgdSolver.ortmann_sparse_stress(g_bb, pivot_strategy="bad"), (ValueError,)),
        ("largevis/neg_gamma", lambda: eg.SgdSolver.largevis(g_bb, gamma=-1.0), (ValueError,)),
        ("largevis/bool_seed", lambda: eg.SgdSolver.largevis(g_bb, seed=True), (TypeError,)),
        ("drgraph/k_order_0", lambda: eg.SgdSolver.drgraph(g_bb, k_order=0), (ValueError,)),
        ("drgraph/bad_ratio", lambda: eg.SgdSolver.drgraph(g_bb, coarsen_stop_ratio=1.5), (ValueError,)),
        ("graph_cne/bad_obj", lambda: eg.SgdSolver.graph_cne(g_bb, objective="bad"), (ValueError,)),
        ("graph_cne/neg_rep", lambda: eg.SgdSolver.graph_cne(g_bb, repulsion_weight=-0.5), (ValueError,)),
        ("largevis/nan_layout", lambda: eg.SgdSolver.largevis(g_bb, iterations=2).run(bad_lay), (ValueError,)),
        ("largevis/wrong_len", lambda: eg.SgdSolver.largevis(g_bb, iterations=2).run(wrong_lay), (ValueError,)),
        ("maxent/q_too_low", lambda: eg.MaxentStressSolver(g_bb, q=-2.0), (ValueError,)),
        ("maxent/q_nan", lambda: eg.MaxentStressSolver(g_bb, q=float("nan")), (ValueError,)),
        ("maxent/alpha_decay_bad", lambda: eg.MaxentStressSolver(g_bb, alpha_decay=1.5), (ValueError,)),
        ("maxent/nan_layout", lambda: eg.MaxentStressSolver(g_bb, stages=1, steps_per_stage=2).run(bad_lay), (ValueError,)),
        ("maxent/wrong_len", lambda: eg.MaxentStressSolver(g_bb, stages=1, steps_per_stage=2).run(wrong_lay), (ValueError,)),
        ("grip/rounds_inverted", lambda: eg.GripSolver(g_bb, rounds_min=10, rounds_max=5), (ValueError,)),
        ("grip/edge_length_0", lambda: eg.GripSolver(g_bb, edge_length=0.0), (ValueError,)),
        ("grip/nan_layout", lambda: eg.GripSolver(g_bb, rounds_min=1, rounds_max=2).run(bad_lay), (ValueError,)),
        ("gem/edge_length_0", lambda: eg.GemSolver(g_bb, edge_length=0.0), (ValueError,)),
        ("gem/gravity_neg", lambda: eg.GemSolver(g_bb, gravity=-1.0), (ValueError,)),
        ("gem/nan_layout", lambda: eg.GemSolver(g_bb, max_iterations=2).run(bad_lay), (ValueError,)),
        ("sel/tree_depth_0", lambda: eg.SelLayoutSolver(g_bb, tree_depth=0), (ValueError,)),
        ("sel/bad_mapping", lambda: eg.SelLayoutSolver(g_bb, mapping="bad"), (ValueError,)),
        ("sel/bad_backend", lambda: eg.SelLayoutSolver(g_bb, layout_backend="bad"), (ValueError,)),
        ("sel/nan_layout", lambda: eg.SelLayoutSolver(g_bb, iterations=2).run(bad_lay), (ValueError,)),
        ("pfs_prime/len_mismatch", lambda: eg.PfsPrimeOverlapRemoval([1.0, 2.0], [1.0]), (ValueError,)),
        ("pfs_prime/neg_hw", lambda: eg.PfsPrimeOverlapRemoval([-1.0], [1.0]), (ValueError,)),
        ("pfs_prime/neg_padding", lambda: eg.PfsPrimeOverlapRemoval([1.0], [1.0], padding=-0.1), (ValueError,)),
        ("pfs_prime/inf_radii", lambda: eg.PfsPrimeOverlapRemoval.from_radii([float("inf")]), (ValueError,)),
        ("pfs_prime/nan_layout", lambda: eg.PfsPrimeOverlapRemoval(hw10_ok, hh10_ok).run(bad_lay), (ValueError,)),
        ("pfs_prime/wrong_len", lambda: eg.PfsPrimeOverlapRemoval(hw10_ok, hh10_ok).run(wrong_lay), (ValueError,)),
        ("sfc/scale_0", lambda: eg.SfcLayoutSolver(g_bb, scale=0.0), (ValueError,)),
        ("sfc/res_order_16", lambda: eg.SfcLayoutSolver(g_bb, resolution_order=16), (ValueError,)),
        ("sfc/bad_curve", lambda: eg.SfcLayoutSolver(g_bb, curve="peano"), (ValueError,)),
        ("sfc/nan_layout", lambda: eg.SfcLayoutSolver(g_bb).run(bad_lay), (ValueError,)),
        ("sfc/wrong_len", lambda: eg.SfcLayoutSolver(g_bb).run(wrong_lay), (ValueError,)),
    ]
    for label, fn, expected in classic_bad_cases:
        assert_non_panic_error(label, fn, expected_types=expected)

    results.append({
        "case": "embedding_classic_adversarial_non_panic",
        "status": "pass",
        "elapsed_sec": round(time.perf_counter() - t1, 4),
    })
    return results
