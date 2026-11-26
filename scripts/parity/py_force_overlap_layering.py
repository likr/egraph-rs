"""Python checks for unified force models, overlap removal, constraints, layering, and FDEB."""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from typing import Any

from parity.common import (
    assert_non_panic_error,
    bind_call,
    count_rect_overlaps,
    make_asymmetric_coords,
    make_grid_edges,
)


def run_force_overlap_layering_python_checks(eg: Any) -> list[dict[str, Any]]:
    """Run deterministic, f32/f64, step()==run() & adversarial checks for force/overlap/layering APIs."""
    import numpy as np

    results: list[dict[str, Any]] = []
    t0 = time.perf_counter()

    g16 = eg.CsrGraph.from_edges(16, make_grid_edges(4, 4), directed=False, sanitize=True)
    init16 = np.array(make_asymmetric_coords(16, scale=0.8), dtype=np.float64)

    g25 = eg.CsrGraph.from_edges(25, make_grid_edges(5, 5), directed=False, sanitize=True)
    init25 = np.array(make_asymmetric_coords(25, scale=1.0), dtype=np.float64)

    knn16, _, _ = eg.build_pure_2hop_knn(g16)
    fuzzy16 = eg.compute_fuzzy_simplicial_set(knn16)
    deg16 = np.diff(g16.indptr()).astype(np.uint32)

    for dtype_name, np_dtype in (("float64", np.float64), ("float32", np.float32)):
        init16_typed = init16.astype(np_dtype)
        init25_typed = init25.astype(np_dtype)

        # 1. TFdpSolver across all 3 backends + from_knn
        for backend in ("ibfft", "barnes_hut", "exact"):
            sr = eg.TFdpSolver(g16, backend=backend, total_iterations=8, seed=42, dtype=dtype_name)
            ss = eg.TFdpSolver(g16, backend=backend, total_iterations=8, seed=42, dtype=dtype_name)
            lr = eg.Layout2d.from_coords(init16_typed)
            ls = eg.Layout2d.from_coords(init16_typed)
            rho0 = sr.current_rho
            assert sr.run(lr) == "completed"
            assert sr.current_rho <= rho0
            assert ss.run_steps(ls, 3) == "running"
            while not ss.is_finished:
                ss.step(ls)
            assert ss.status == "completed"
            assert lr.is_all_finite()
            assert np.array_equal(lr.coords(copy=True), ls.coords(copy=True)), (
                f"TFdpSolver/{backend}/{dtype_name} step != run"
            )

        tk1 = eg.TFdpSolver.from_knn(knn16, backend="exact", total_iterations=6, seed=42, dtype=dtype_name)
        tk2 = eg.TFdpSolver.from_knn(
            knn16.indptr(), knn16.indices(), knn16.distances(),
            backend="exact", total_iterations=6, seed=42, dtype=dtype_name,
        )
        lk1 = eg.Layout2d.from_coords(init16_typed)
        lk2 = eg.Layout2d.from_coords(init16_typed)
        assert tk1.run(lk1) == "completed"
        assert tk2.run(lk2) == "completed"
        assert np.array_equal(lk1.coords(copy=True), lk2.coords(copy=True))

        # 2. IbUmapSolver across all 3 backends + from_knn + from_fuzzy_set
        for backend in ("ibfft", "barnes_hut", "exact"):
            ir = eg.IbUmapSolver(g16, max_2hop_per_node=20, backend=backend, n_epochs=6, seed=42, dtype=dtype_name)
            is_ = eg.IbUmapSolver(g16, max_2hop_per_node=20, backend=backend, n_epochs=6, seed=42, dtype=dtype_name)
            lr = eg.Layout2d.from_coords(init16_typed)
            ls = eg.Layout2d.from_coords(init16_typed)
            assert ir.run(lr) == "completed"
            assert math.isclose(ir.current_rho, 1.0, rel_tol=1e-9)
            assert is_.run_steps(ls, 2) == "running"
            while not is_.is_finished:
                is_.step(ls)
            assert lr.is_all_finite()
            assert np.array_equal(lr.coords(copy=True), ls.coords(copy=True)), (
                f"IbUmapSolver/{backend}/{dtype_name} step != run"
            )

        ik = eg.IbUmapSolver.from_knn(knn16, backend="exact", n_epochs=6, seed=42, dtype=dtype_name)
        ifs = eg.IbUmapSolver.from_fuzzy_set(fuzzy16, degrees=deg16, backend="exact", n_epochs=6, seed=42, dtype=dtype_name)
        lik = eg.Layout2d.from_coords(init16_typed)
        lif = eg.Layout2d.from_coords(init16_typed)
        assert ik.run(lik) == "completed" and lik.is_all_finite()
        assert ifs.run(lif) == "completed" and lif.is_all_finite()

        # 3. SgdSolver force/embedding constructors (snap_t_fdp, pacmap_graph, sgd2, schedule="squared")
        force_sgd_builders: list[tuple[str, Callable[[], Any]]] = [
            ("snap_t_fdp", bind_call(eg.SgdSolver.snap_t_fdp, g16, iterations=8, negatives_per_edge=3, seed=42, dtype=dtype_name)),
            (
                "pacmap_graph",
                bind_call(
                    eg.SgdSolver.pacmap_graph,
                    g16,
                    n_neighbors=5,
                    phase1_iters=3,
                    phase2_iters=3,
                    phase3_iters=4,
                    seed=42,
                    dtype=dtype_name,
                ),
            ),
            ("sgd2", bind_call(eg.SgdSolver.sgd2, g16, t_max=8, batch_size=16, seed=42, dtype=dtype_name)),
            ("squared", bind_call(eg.SgdSolver.full, g16, t_max=8, schedule="squared", seed=42, dtype=dtype_name)),
        ]
        for s_name, mk_s in force_sgd_builders:
            sr = mk_s()
            ss = mk_s()
            lr = eg.Layout2d.from_coords(init16_typed)
            ls = eg.Layout2d.from_coords(init16_typed)
            assert sr.run(lr) == "completed", f"SgdSolver.{s_name} did not complete"
            assert ss.run_steps(ls, 3) == "running"
            while not ss.is_finished:
                ss.step(ls)
            assert lr.is_all_finite()
            assert np.array_equal(lr.coords(copy=True), ls.coords(copy=True)), (
                f"SgdSolver.{s_name}/{dtype_name} step != run"
            )

        # 4. ForceAtlas2Solver (Barnes-Hut & Exact)
        for use_bh in (True, False):
            fr = eg.ForceAtlas2Solver(g16, iterations=10, use_barnes_hut=use_bh, seed=42, dtype=dtype_name)
            fs = eg.ForceAtlas2Solver(g16, iterations=10, use_barnes_hut=use_bh, seed=42, dtype=dtype_name)
            lr = eg.Layout2d.from_coords(init16_typed)
            ls = eg.Layout2d.from_coords(init16_typed)
            assert fr.run(lr) == "completed"
            assert fs.run_steps(ls, 4) == "running"
            while not fs.is_finished:
                fs.step(ls)
            assert lr.is_all_finite()
            assert np.array_equal(lr.coords(copy=True), ls.coords(copy=True)), (
                f"ForceAtlas2Solver/bh={use_bh}/{dtype_name} step != run"
            )

        # 5. YifanHuSolver (multilevel)
        yr = eg.YifanHuSolver(g25, max_iterations_per_level=10, min_coarsen_size=6, min_coarsen_ratio=0.9, seed=42, dtype=dtype_name)
        ys = eg.YifanHuSolver(g25, max_iterations_per_level=10, min_coarsen_size=6, min_coarsen_ratio=0.9, seed=42, dtype=dtype_name)
        assert yr.num_levels >= 2
        lyr = eg.Layout2d.from_coords(init25_typed)
        lys = eg.Layout2d.from_coords(init25_typed)
        assert yr.run(lyr) in ("converged", "completed")
        assert yr.current_level == 0
        ys.step(lys)
        ys.run_steps(lys, 4)
        ys.run(lys)
        assert lyr.is_all_finite()
        assert np.array_equal(lyr.coords(copy=True), lys.coords(copy=True)), (
            f"YifanHuSolver/{dtype_name} step != run"
        )

        # 6. PrismOverlapRemoval (new + from_radii)
        prism_init = (init16 * 0.15).astype(np_dtype)
        hw = np.full(16, 0.5, dtype=np.float64)
        hh = np.full(16, 0.4, dtype=np.float64)
        pr = eg.PrismOverlapRemoval(hw, hh, max_iterations=50, padding=0.05, seed=42, dtype=dtype_name)
        ps = eg.PrismOverlapRemoval(hw, hh, max_iterations=50, padding=0.05, seed=42, dtype=dtype_name)
        lpr = eg.Layout2d.from_coords(prism_init)
        lps = eg.Layout2d.from_coords(prism_init)
        assert pr.run(lpr) == "converged"
        assert pr.last_overlap_count == 0
        ps.step(lps)
        ps.run_steps(lps, 2)
        ps.run(lps)
        assert np.array_equal(lpr.coords(copy=True), lps.coords(copy=True))
        tol_p = 1e-4 if dtype_name == "float32" else 1e-8
        assert count_rect_overlaps(lpr.coords(copy=True), hw + 0.025, hh + 0.025, tol=tol_p) == 0

        pr_rad = eg.PrismOverlapRemoval.from_radii(np.full(16, 0.4, dtype=np.float64), max_iterations=50, seed=42, dtype=dtype_name)
        l_rad = eg.Layout2d.from_coords(prism_init)
        assert pr_rad.run(l_rad) == "converged"
        assert pr_rad.last_overlap_count == 0

        # 7. SeparationConstraint + projection helpers
        arr_1d = np.array([2.0, 0.5, 0.0], dtype=np_dtype)
        out_1d = eg.project_1d(arr_1d, [eg.SeparationConstraint(0, 1, 1.5), eg.SeparationConstraint(1, 2, 2.0)])
        assert out_1d.dtype == np_dtype
        assert float(out_1d[1] - out_1d[0]) >= 1.5 - 1e-5
        assert float(out_1d[2] - out_1d[1]) >= 2.0 - 1e-5

        rect_l = eg.Layout2d.from_coords(np.array([[0.0, 0.0], [0.2, 0.1], [0.1, -0.1], [-0.1, 0.2]], dtype=np_dtype))
        hw4 = np.full(4, 0.5, dtype=np_dtype)
        hh4 = np.full(4, 0.5, dtype=np_dtype)
        eg.project_rectangle_no_overlap_2d(rect_l, hw4, hh4)
        assert count_rect_overlaps(rect_l.coords(copy=True), hw4, hh4, tol=1e-4) == 0

        cl_l = eg.Layout2d.from_coords(np.array([[0.0, 0.0], [0.5, 0.0], [0.3, 0.2], [0.8, 0.2]], dtype=np_dtype))
        eg.project_clustered_rectangle_no_overlap_2d(cl_l, np.array([0, 0, 1, 1], dtype=np.uint32), hw4, hh4)
        cl_c = cl_l.coords(copy=True)
        assert np.allclose(cl_c[1] - cl_c[0], [0.5, 0.0], atol=1e-5)
        assert np.allclose(cl_c[3] - cl_c[2], [0.5, 0.0], atol=1e-5)

        # 8. FdebSolver (step == run, input layout preserved)
        fdeb_base = np.array([[float(c), float(r)] for r in range(4) for c in range(4)], dtype=np_dtype)
        l_fdeb = eg.Layout2d.from_coords(fdeb_base)
        fr_fdeb = eg.FdebSolver(g16, l_fdeb, cycles=2, i0=6, dtype=dtype_name)
        fs_fdeb = eg.FdebSolver(g16, l_fdeb, cycles=2, i0=6, dtype=None)
        assert fs_fdeb.dtype == dtype_name
        assert fr_fdeb.run() == "completed"
        fs_fdeb.step()
        fs_fdeb.run_steps(3)
        fs_fdeb.run()
        off_r = fr_fdeb.edge_offsets()
        pts_r = fr_fdeb.points()
        assert off_r.dtype == np.uint32 and off_r.shape == (g16.num_edges + 1,)
        assert int(off_r[0]) == 0 and int(off_r[-1]) == pts_r.shape[0]
        assert pts_r.dtype == np_dtype and pts_r.shape[1] == 2 and np.all(np.isfinite(pts_r))
        assert np.array_equal(off_r, fs_fdeb.edge_offsets())
        assert np.array_equal(pts_r, fs_fdeb.points())
        assert np.array_equal(l_fdeb.coords(copy=True), fdeb_base)

    # 9. Layering (cycle_edges, remove_cycles, assign_layers) & generate_layered_constraints
    cyc = eg.CsrGraph.from_edges(4, [(0, 1), (1, 2), (2, 0), (1, 3)], directed=True, sanitize=True)
    ce = eg.cycle_edges(cyc)
    assert ce.dtype == np.uint32 and ce.ndim == 2 and ce.shape[1] == 2 and ce.shape[0] >= 1
    dag = eg.remove_cycles(cyc)
    assert dag.is_directed and eg.cycle_edges(dag).shape == (0, 2)

    skip_edges = [(0, 1), (1, 2), (2, 3), (4, 3)]
    skip_dag = eg.CsrGraph.from_edges(5, skip_edges, directed=True, sanitize=True)
    lp = eg.assign_layers(skip_dag, method="longest_path")
    ns = eg.assign_layers(skip_dag, method="network_simplex")
    assert lp.dtype == np.uint32 and ns.dtype == np.uint32
    for u, v in skip_edges:
        assert int(lp[u]) < int(lp[v]) and int(ns[u]) < int(ns[v])
    lp_span = sum(int(lp[v]) - int(lp[u]) for u, v in skip_edges)
    ns_span = sum(int(ns[v]) - int(ns[u]) for u, v in skip_edges)
    assert ns_span < lp_span
    layered_cs = eg.generate_layered_constraints(skip_dag, ns, min_layer_distance=2.0)
    l_lay = eg.Layout2d.zeros(5, dtype="float64")
    eg.project_layout_axis(l_lay, 1, layered_cs)
    ys_lay = l_lay.coords(copy=True)[:, 1]
    for u, v in skip_edges:
        assert float(ys_lay[v] - ys_lay[u]) >= 2.0 - 1e-6

    results.append({
        "case": "force_overlap_and_layering",
        "status": "pass",
        "elapsed_sec": round(time.perf_counter() - t0, 4),
    })

    # 10. Adversarial force/overlap/layering checks (never PanicException)
    t1 = time.perf_counter()
    bad_l = eg.Layout2d.zeros(5)
    l16_nan = eg.Layout2d.zeros(16)
    l16_nan.coords(copy=False)[0, 0] = float("nan")
    force_postproc_bad_cases: list[tuple[str, Callable[[], Any]]] = [
        ("TFdpSolver/bad_backend", lambda: eg.TFdpSolver(g16, backend="bad")),
        ("TFdpSolver/momentum_1", lambda: eg.TFdpSolver(g16, momentum=1.0)),
        ("TFdpSolver/bool_iters", lambda: eg.TFdpSolver(g16, total_iterations=True)),
        ("TFdpSolver/dim_mismatch", lambda: eg.TFdpSolver(g16).step(bad_l)),
        ("TFdpSolver/nan_layout", lambda: eg.TFdpSolver(g16).step(l16_nan)),
        ("TFdpSolver.from_knn/bad_indptr", lambda: eg.TFdpSolver.from_knn([], [])),
        ("IbUmapSolver/zero_tau", lambda: eg.IbUmapSolver(g16, tau_scale=0.0)),
        ("IbUmapSolver/bad_backend", lambda: eg.IbUmapSolver(g16, backend="bad")),
        ("SgdSolver.snap_t_fdp/zero_rho_power", lambda: eg.SgdSolver.snap_t_fdp(g16, rho_power=0.0)),
        ("SgdSolver.pacmap_graph/zero_neighbors", lambda: eg.SgdSolver.pacmap_graph(g16, n_neighbors=0)),
        ("SgdSolver.sgd2/zero_batch", lambda: eg.SgdSolver.sgd2(g16, batch_size=0)),
        ("ForceAtlas2Solver/zero_jitter", lambda: eg.ForceAtlas2Solver(g16, jitter_tolerance=0.0)),
        ("ForceAtlas2Solver/dim_mismatch", lambda: eg.ForceAtlas2Solver(g16).step(bad_l)),
        ("YifanHuSolver/step_ratio_1", lambda: eg.YifanHuSolver(g16, step_ratio=1.0)),
        ("YifanHuSolver/dim_mismatch", lambda: eg.YifanHuSolver(g16).step(bad_l)),
        ("PrismOverlapRemoval/len_mismatch", lambda: eg.PrismOverlapRemoval([1.0, 1.0], [1.0])),
        ("PrismOverlapRemoval/neg_half", lambda: eg.PrismOverlapRemoval([1.0], [-0.5])),
        ("PrismOverlapRemoval/nan_layout", lambda: eg.PrismOverlapRemoval.from_radii([0.5] * 16).step(l16_nan)),
        ("SeparationConstraint/nan_gap", lambda: eg.SeparationConstraint(0, 1, float("nan"))),
        ("project_1d/cycle", lambda: eg.project_1d([0.0, 1.0], [eg.SeparationConstraint(0, 1, 1.0), eg.SeparationConstraint(1, 0, 1.0)])),
        ("project_layout_axis/bad_axis", lambda: eg.project_layout_axis(bad_l, 2, [])),
        ("generate_layered_constraints/len_mismatch", lambda: eg.generate_layered_constraints(dag, [0])),
        ("generate_layered_constraints/neg_dist", lambda: eg.generate_layered_constraints(dag, [0, 1, 2, 3], -1.0)),
        ("project_rectangle_no_overlap_2d/len_mismatch", lambda: eg.project_rectangle_no_overlap_2d(bad_l, [1.0], [1.0])),
        ("project_clustered_rectangle_no_overlap_2d/len_mismatch", lambda: eg.project_clustered_rectangle_no_overlap_2d(bad_l, [0], [1.0] * 5, [1.0] * 5)),
        ("cycle_edges/undirected", lambda: eg.cycle_edges(g16)),
        ("remove_cycles/undirected", lambda: eg.remove_cycles(g16)),
        ("assign_layers/undirected", lambda: eg.assign_layers(g16)),
        ("assign_layers/bad_method", lambda: eg.assign_layers(dag, method="unknown")),
        ("FdebSolver/dim_mismatch", lambda: eg.FdebSolver(g16, bad_l)),
        ("FdebSolver/cycles_too_large", lambda: eg.FdebSolver(g16, eg.Layout2d.zeros(16), cycles=30)),
        ("FdebSolver/nan_layout", lambda: eg.FdebSolver(g16, l16_nan)),
    ]
    for label, fn in force_postproc_bad_cases:
        assert_non_panic_error(label, fn)

    results.append({
        "case": "force_overlap_adversarial_non_panic",
        "status": "pass",
        "elapsed_sec": round(time.perf_counter() - t1, 4),
    })
    return results
