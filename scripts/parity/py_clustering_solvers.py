"""Python checks for clustering, ManifoldLayout2d, and core layout solvers."""

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
)


def run_clustering_solvers_python_checks(eg: Any) -> list[dict[str, Any]]:
    """Run deterministic, f32/f64, step()==run(), manifold & adversarial checks."""
    import numpy as np

    results: list[dict[str, Any]] = []
    t0 = time.perf_counter()

    # 1. Clustering & coarsening (KMeans++, Spectral, Louvain, LPA, Coarsen, Project)
    pts_km = np.array(
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
    km64 = eg.kmeans_plus_plus(pts_km, k=2, max_iters=100, tol=1e-4, n_init=3, seed=42)
    km32 = eg.kmeans_plus_plus(pts_km.astype(np.float32), k=2, max_iters=100, tol=1e-4, n_init=3, seed=42)
    assert np.array_equal(km64["labels"], np.array([0, 0, 0, 1, 1, 1], dtype=np.uint32))
    assert np.array_equal(km64["labels"], km32["labels"])
    assert bool(km64["converged"]) and float(km64["inertia"]) < 1.0

    # 12-node 2-clique barbell for clustering separation checks
    bb12_edges = make_barbell_edges(6)
    g_bb12 = eg.CsrGraph.from_edges(12, bb12_edges, directed=False, sanitize=True)

    sp1 = eg.spectral_clustering(g_bb12, k=2, seed=42)
    sp2 = eg.spectral_clustering(g_bb12, k=2, seed=42)
    assert np.array_equal(sp1, sp2) and np.array_equal(sp1, np.array([0] * 6 + [1] * 6, dtype=np.uint32))

    lv = eg.louvain_communities(g_bb12, max_passes=100, resolution=1.0, min_modularity_gain=1e-7, seed=42)
    assert lv["num_communities"] == 2 and float(lv["modularity"]) > 0.35
    assert np.array_equal(lv["labels"], np.array([0] * 6 + [1] * 6, dtype=np.uint32))

    lpa = eg.label_propagation(g_bb12, max_iters=100, seed=42)
    assert np.array_equal(lpa, np.array([0] * 6 + [1] * 6, dtype=np.uint32))

    coarse_g, n2c, sizes, num_clusters = eg.coarsen_graph(g_bb12, [100] * 6 + [999] * 6)
    assert num_clusters == 2 and coarse_g.num_nodes == 2 and coarse_g.num_edges == 1
    assert np.array_equal(n2c, np.array([0] * 6 + [1] * 6, dtype=np.uint32))
    assert np.array_equal(sizes, np.array([6, 6], dtype=np.uint32))

    coarse_base = np.array([[-2.0, 0.0], [2.0, 0.0]], dtype=np.float64)
    for dtype_name, np_dtype in (("float64", np.float64), ("float32", np.float32)):
        cl = eg.Layout2d.from_coords(coarse_base.astype(np_dtype))
        proj1 = eg.project_clustered_layout(cl, n2c, spread_radius=0.2, seed=42)
        proj2 = eg.project_clustered_layout(cl, n2c, spread_radius=0.2, seed=42)
        assert proj1.dtype == dtype_name and len(proj1) == 12 and proj1.is_all_finite()
        assert np.array_equal(proj1.coords(copy=True), proj2.coords(copy=True))

    # 2. ManifoldLayout2d across all 3 geometries in float64 & float32
    manifold_coords = {
        "hyperbolic": np.array([[0.0, 0.0], [0.5, 0.0], [3.0, 4.0]], dtype=np.float64),
        "spherical": np.array([[-0.3, -0.2], [0.2, math.pi / 2.0], [0.2 + math.pi / 2.0, math.pi / 2.0]], dtype=np.float64),
        "torus": np.array([[1.95, -0.95], [0.05, 0.95], [0.5, 0.5]], dtype=np.float64),
    }
    for kind, base_arr in manifold_coords.items():
        for dtype_name, np_dtype in (("float64", np.float64), ("float32", np.float32)):
            ml = eg.ManifoldLayout2d.from_coords(kind, base_arr.astype(np_dtype))
            assert ml.kind == kind and ml.dtype == dtype_name and ml.len() == 3
            ml.project_all()
            c = ml.coords()
            assert np.all(np.isfinite(c))
            if kind == "hyperbolic":
                assert np.all(c[:, 0] ** 2 + c[:, 1] ** 2 < 1.0)
                assert abs(ml.geodesic_distance(0, 1) - 2.0 * math.atanh(0.5)) < 1e-5
            elif kind == "spherical":
                assert np.all((c[:, 0] >= 0.0) & (c[:, 0] < 2.0 * math.pi))
                assert np.all((c[:, 1] >= 0.0) & (c[:, 1] <= math.pi))
                assert abs(ml.geodesic_distance(1, 2) - math.pi / 2.0) < 1e-5
            else:
                assert np.all((c >= 0.0) & (c < 1.0))
                assert abs(ml.geodesic_distance(0, 1) - math.hypot(0.1, 0.1)) < 1e-5
            back = ml.to_layout()
            ml2 = eg.ManifoldLayout2d.from_layout(kind, back)
            assert np.allclose(ml2.coords(), c)

    # 3. All 6 Euclidean solvers + SgdSolver manifold stepping (step == run bit-exact in f64 & f32)
    g16_edges = make_grid_edges(4, 4)
    g16 = eg.CsrGraph.from_edges(16, g16_edges, directed=False, sanitize=True)
    init16 = np.array(make_asymmetric_coords(16, scale=0.8), dtype=np.float64)
    emb16 = eg.chebyshev_dog_jl_embedding(g16, degree=15, jl_dim=8, seed=42)
    hinge_src = [e[0] for e in g16_edges]
    hinge_tgt = [e[1] for e in g16_edges]
    hinge_ones = [1.0] * len(g16_edges)

    for dtype_name, np_dtype in (("float64", np.float64), ("float32", np.float32)):
        init_typed = init16.astype(np_dtype)

        # TsNetSolver.pure_2hop (all 4 backends)
        for backend in ("fft", "linear_fft", "barnes_hut", "exact"):
            ts_r = eg.TsNetSolver.pure_2hop(
                g16, backend=backend, iterations_stage1=4, iterations_stage2=4, iterations_stage3=4, seed=42, dtype=dtype_name
            )
            ts_s = eg.TsNetSolver.pure_2hop(
                g16, backend=backend, iterations_stage1=4, iterations_stage2=4, iterations_stage3=4, seed=42, dtype=dtype_name
            )
            lr = eg.Layout2d.from_coords(init_typed)
            ls = eg.Layout2d.from_coords(init_typed)
            assert ts_r.run(lr) == "completed"
            assert ts_s.run_steps(ls, 5) == "running"
            while not ts_s.is_finished:
                ts_s.step(ls)
            assert ts_s.status == "completed"
            assert np.array_equal(lr.coords(copy=True), ls.coords(copy=True))

        # SgdSolver variants
        sgd_builders: list[tuple[str, Callable[[], Any]]] = [
            ("full", bind_call(eg.SgdSolver.full, g16, t_max=10, seed=42, dtype=dtype_name)),
            (
                "sparse_barrier_dog",
                bind_call(eg.SgdSolver.sparse_barrier_dog, g16, degree=15, num_landmarks=8, t_max=10, seed=42, dtype=dtype_name),
            ),
            ("random_pair", bind_call(eg.SgdSolver.random_pair, g16, pairs_per_node=4, t_max=10, seed=42, dtype=dtype_name)),
            ("omega", bind_call(eg.SgdSolver.omega, g16, emb16, k=4, t_max=10, seed=42, dtype=dtype_name)),
            (
                "streaming_negatives",
                bind_call(eg.SgdSolver.streaming_negatives, g16, negatives_per_node=3, t_max=10, seed=42, dtype=dtype_name),
            ),
            (
                "from_pairs_with_hinges",
                bind_call(
                    eg.SgdSolver.from_pairs_with_hinges,
                    16,
                    hinge_src,
                    hinge_tgt,
                    hinge_ones,
                    hinge_ones,
                    hinge_ones,
                    hinge_ones,
                    [0, 1],
                    [15, 14],
                    [3.0, 3.0],
                    [3.0, 3.0],
                    [1.0, 1.0],
                    [1.0, 1.0],
                    t_max=10,
                    seed=42,
                    dtype=dtype_name,
                ),
            ),
        ]
        for s_name, mk_solver in sgd_builders:
            sr = mk_solver()
            ss = mk_solver()
            lr = eg.Layout2d.from_coords(init_typed)
            ls = eg.Layout2d.from_coords(init_typed)
            assert sr.run(lr) == "completed", f"SgdSolver.{s_name} did not complete"
            assert ss.run_steps(ls, 4) == "running"
            while not ss.is_finished:
                ss.step(ls)
            assert np.array_equal(lr.coords(copy=True), ls.coords(copy=True)), f"SgdSolver.{s_name} step != run"

        # SgdSolver manifold stepping (all 3 manifolds)
        for kind in ("hyperbolic", "spherical", "torus"):
            mr = eg.SgdSolver.full(g16, t_max=8, seed=42, dtype=dtype_name)
            ms = eg.SgdSolver.full(g16, t_max=8, seed=42, dtype=dtype_name)
            mlr = eg.ManifoldLayout2d.from_coords(kind, (init16 * 0.25).astype(np_dtype))
            mls = eg.ManifoldLayout2d.from_coords(kind, (init16 * 0.25).astype(np_dtype))
            assert mr.run_manifold(mlr) == "completed"
            assert ms.run_steps_manifold(mls, 3) == "running"
            while not ms.is_finished:
                ms.step_manifold(mls)
            assert np.array_equal(mlr.coords(), mls.coords())

        # GuMapSolver.from_graph_dog
        gmr = eg.GuMapSolver.from_graph_dog(g16, max_2hop_per_node=16, degree=15, num_landmarks=8, n_epochs=15, seed=42, dtype=dtype_name)
        gms = eg.GuMapSolver.from_graph_dog(g16, max_2hop_per_node=16, degree=15, num_landmarks=8, n_epochs=15, seed=42, dtype=dtype_name)
        glr = eg.Layout2d.from_coords(init_typed)
        gls = eg.Layout2d.from_coords(init_typed)
        assert gmr.run(glr) == "completed"
        assert gms.run_steps(gls, 5) == "running"
        while not gms.is_finished:
            gms.step(gls)
        assert np.array_equal(glr.coords(copy=True), gls.coords(copy=True))

        # SpatialIntruderRefiner on folded 3x3 grid
        g9 = eg.CsrGraph.from_edges(9, make_grid_edges(3, 3), directed=False, sanitize=True)
        folded9 = np.array([[float(c), float(r)] for r in range(3) for c in range(3)], dtype=np.float64)
        folded9[8] = np.array([0.08, 0.08], dtype=np.float64)
        rf_r = eg.SpatialIntruderRefiner(g9, rounds=3, k_cap=32, t_max=10, seed=42, dtype=dtype_name)
        rf_s = eg.SpatialIntruderRefiner(g9, rounds=3, k_cap=32, t_max=10, seed=42, dtype=dtype_name)
        rlr = eg.Layout2d.from_coords(folded9.astype(np_dtype))
        rls = eg.Layout2d.from_coords(folded9.astype(np_dtype))
        assert rf_r.run(rlr) in ("converged", "completed")
        rf_s.step(rls)
        rf_s.run(rls)
        assert np.array_equal(rlr.coords(copy=True), rls.coords(copy=True))
        assert rf_r.stats()["n_fp_total"] > 0

        # StressMajorizationSolver.from_graph
        sm_r = eg.StressMajorizationSolver.from_graph(g16, max_iterations=15, epsilon=1e-6, dtype=dtype_name)
        sm_s = eg.StressMajorizationSolver.from_graph(g16, max_iterations=15, epsilon=1e-6, dtype=dtype_name)
        slr = eg.Layout2d.from_coords(init_typed)
        sls = eg.Layout2d.from_coords(init_typed)
        assert sm_r.run(slr) in ("converged", "completed")
        sm_s.run_steps(sls, 4)
        while not sm_s.is_finished:
            sm_s.step(sls)
        assert np.array_equal(slr.coords(copy=True), sls.coords(copy=True))

        # KamadaKawaiSolver.from_graph
        kk_r = eg.KamadaKawaiSolver.from_graph(g9, eps=1e-3, max_inner_iters=40, max_outer_iters=20, dtype=dtype_name)
        kk_s = eg.KamadaKawaiSolver.from_graph(g9, eps=1e-3, max_inner_iters=40, max_outer_iters=20, dtype=dtype_name)
        init9 = np.array(make_asymmetric_coords(9, scale=1.2), dtype=np.float64).astype(np_dtype)
        klr = eg.Layout2d.from_coords(init9)
        kls = eg.Layout2d.from_coords(init9)
        assert kk_r.run(klr) in ("converged", "completed")
        kk_s.run_steps(kls, 4)
        while not kk_s.is_finished:
            kk_s.step(kls)
        assert np.array_equal(klr.coords(copy=True), kls.coords(copy=True))

    results.append({
        "case": "clustering_manifold_and_solvers",
        "status": "pass",
        "elapsed_sec": round(time.perf_counter() - t0, 4),
    })

    # 4. Adversarial clustering & solver checks (never PanicException)
    t1 = time.perf_counter()
    g_dir = eg.CsrGraph.from_edges(3, [(0, 1), (1, 2)], directed=True, sanitize=True)
    l_nan = eg.Layout2d.zeros(12)
    l_nan.coords(copy=False)[0, 0] = float("nan")
    sm_chk = eg.StressMajorizationSolver.from_graph(g_bb12, max_iterations=5)
    kk_chk = eg.KamadaKawaiSolver.from_graph(g_bb12, max_outer_iters=5)
    solver_bad_cases: list[tuple[str, Callable[[], Any]]] = [
        ("kmeans_plus_plus/k_zero", lambda: eg.kmeans_plus_plus(pts_km, k=0)),
        ("kmeans_plus_plus/k_gt_n", lambda: eg.kmeans_plus_plus(pts_km, k=10)),
        ("spectral_clustering/directed", lambda: eg.spectral_clustering(g_dir, k=2)),
        ("louvain_communities/directed", lambda: eg.louvain_communities(g_dir)),
        ("label_propagation/directed", lambda: eg.label_propagation(g_dir)),
        ("coarsen_graph/directed", lambda: eg.coarsen_graph(g_dir, [0, 1, 1])),
        ("project_clustered_layout/oob", lambda: eg.project_clustered_layout(eg.Layout2d.zeros(2), [0, 2])),
        ("ManifoldLayout2d/bad_kind", lambda: eg.ManifoldLayout2d.zeros("euclidean", 3)),
        ("ManifoldLayout2d/nan", lambda: eg.ManifoldLayout2d.from_coords("torus", [[float("nan"), 0.0]])),
        ("TsNetSolver/bad_backend", lambda: eg.TsNetSolver.pure_2hop(g16, backend="bad")),
        ("SgdSolver/bad_schedule", lambda: eg.SgdSolver.full(g16, schedule="cosine")),
        ("SgdSolver/oob_pair", lambda: eg.SgdSolver.from_pairs(2, [0], [5], [1.0])),
        ("GuMapSolver/neg_min_dist", lambda: eg.GuMapSolver.from_graph_dog(g16, min_dist=-0.1)),
        ("SpatialIntruderRefiner/k_cap_zero", lambda: eg.SpatialIntruderRefiner(g16, k_cap=0)),
        ("StressMajorizationSolver/nan_layout", lambda: sm_chk.step(l_nan)),
        ("KamadaKawaiSolver/dim_mismatch", lambda: kk_chk.step(eg.Layout2d.zeros(3))),
    ]
    for label, fn in solver_bad_cases:
        assert_non_panic_error(label, fn)

    results.append({
        "case": "clustering_solvers_adversarial_non_panic",
        "status": "pass",
        "elapsed_sec": round(time.perf_counter() - t1, 4),
    })
    return results
