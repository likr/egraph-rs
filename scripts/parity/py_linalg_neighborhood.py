"""Python checks for linear algebra, spectral, AMG-PCG, and neighborhood APIs."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

from parity.common import (
    assert_non_panic_error,
    make_fixtures,
    make_grid_star_graph_spec,
)


def run_linalg_neighborhood_python_checks(eg: Any) -> list[dict[str, Any]]:
    """Run deterministic, f32/f64, edge-case & adversarial checks for linalg/neighborhood APIs."""
    import numpy as np

    results: list[dict[str, Any]] = []
    t0 = time.perf_counter()
    n, edges = make_grid_star_graph_spec()
    g = eg.CsrGraph.from_edges(n, edges, directed=False, sanitize=True)

    # 1. Pivots, PivotMDS (f64/f32, target_std=1.0/None), Koren (f64/f32, target_std=1.0/None)
    piv1 = eg.select_maxmin_sp_pivots(g, 12, seed=42, min_degree=2)
    piv2 = eg.select_maxmin_sp_pivots(g, 12, seed=42, min_degree=2)
    assert piv1.dtype == np.uint32 and piv1.shape == (12,)
    assert np.array_equal(piv1, piv2) and int(piv1.max()) < n

    for dtype_name, np_dtype in (("float64", np.float64), ("float32", np.float32)):
        lp = eg.pivot_mds_init(
            g, num_pivots=12, seed=42, min_degree=2, target_std=1.0, dtype=dtype_name
        )
        assert lp.dtype == dtype_name and lp.is_all_finite()
        cp = lp.coords(copy=True)
        assert cp.dtype == np_dtype and cp.shape == (n, 2)
        assert abs(float(np.std(cp)) - 1.0) < 1e-4

        lk = eg.koren_spectral_init(
            g, target_std=1.0, tol=1e-4, max_iterations=300, seed=42, dtype=dtype_name
        )
        assert lk.dtype == dtype_name and lk.is_all_finite()
        ck = lk.coords(copy=True)
        assert ck.dtype == np_dtype and ck.shape == (n, 2)
        assert abs(float(np.std(ck)) - 1.0) < 1e-4

    assert eg.pivot_mds_init(g, num_pivots=12, target_std=None).is_all_finite()
    assert eg.koren_spectral_init(g, target_std=None).is_all_finite()

    # Empty & singleton graph edge cases
    for empty_n in (0, 1):
        ge = eg.CsrGraph.from_edges(empty_n, [])
        assert len(eg.select_maxmin_sp_pivots(ge, 4)) == empty_n
        assert len(eg.pivot_mds_init(ge, num_pivots=4, dtype="float32")) == empty_n
        assert len(eg.koren_spectral_init(ge, dtype="float64")) == empty_n
        lm_e = eg.compute_regularized_dog_varadhan_landmarks(ge)
        assert lm_e.num_nodes == empty_n
        assert len(lm_e.alt_distances([0] if empty_n == 1 else [], [0] if empty_n == 1 else [])) == empty_n

    # 2. Eigenpairs (both Lanczos path on n=70 and dense fallback on barbell_2comm_9) & moment stats
    lams, vecs, inv_d, used_dense = eg.smallest_laplacian_eigenpairs(
        g, n_modes=4, tol=1e-4, max_iterations=300, seed=42
    )
    assert used_dense is False
    assert lams.shape == (4,) and vecs.shape == (n, 4) and inv_d.shape == (n,)
    assert np.allclose(vecs.T @ vecs, np.eye(4), atol=1e-6)
    assert float(inv_d[69]) == 0.0

    bb_spec = make_fixtures()["barbell_2comm_9"]
    g_bb = eg.CsrGraph.from_edges(bb_spec["n"], bb_spec["edges"], directed=False, sanitize=True)
    _, _, _, bb_dense = eg.smallest_laplacian_eigenpairs(g_bb, n_modes=4)
    assert bb_dense is True

    stats = eg.spectral_moment_stats(g)
    assert stats["sigma_lambda_sq"] > 0.0 and stats["sigma_lambda"] > 0.0
    assert stats["leaf_fraction"] > 0.0 and stats["mean_degree"] > 0.0

    # 3. Chebyshev heat SpMV (1D & 2D) and DoG JL embedding
    deg_vec = np.where(inv_d > 0.0, 1.0 / np.maximum(inv_d, 1e-300), 0.0)
    heat1d = eg.chebyshev_heat_spmv(g, deg_vec, 2.5)
    assert heat1d.shape == (n,) and np.allclose(heat1d, deg_vec, atol=1e-9)
    mat2d = np.column_stack([deg_vec, np.arange(n, dtype=np.float64) / float(n)])
    heat2d = eg.chebyshev_heat_spmv(g, mat2d, 2.5, degree=20, lambda_max=2.0)
    assert heat2d.shape == (n, 2) and np.all(np.isfinite(heat2d))

    jl_emb = eg.chebyshev_dog_jl_embedding(
        g, degree=25, jl_dim=12, seed=42, degree_normalize=True, eps_psd=1e-4
    )
    jl_plain = eg.chebyshev_dog_jl_embedding(
        g, degree=25, jl_dim=12, seed=42, degree_normalize=False, eps_psd=1e-4
    )
    assert jl_emb.shape == (n, 12)
    assert np.allclose(jl_emb, jl_plain * inv_d[:, None], atol=1e-12)

    # 4. 1+2-hop CSR, pure 2-hop kNN, perplexities, hybrid kNN, partial BFS kNN
    hop1, hop2 = eg.extract_1hop_2hop_csr(g)
    assert hop1.num_nodes == n and hop2.num_nodes == n
    assert hop1.num_edges == g.num_edges

    pure_knn, k2_deg, pure_stats = eg.build_pure_2hop_knn(g)
    assert pure_knn.indptr().dtype == np.uint32
    sizes = eg.compute_2hop_neighborhood_sizes(g)
    assert np.array_equal(k2_deg, sizes)
    assert int(k2_deg[69]) == 0 and pure_stats["nnz"] == pure_knn.nnz
    knn_rt = eg.KnnGraph.from_csr(
        pure_knn.num_nodes, pure_knn.indptr(), pure_knn.indices(), pure_knn.distances()
    )
    assert len(knn_rt) == n and knn_rt.nnz == pure_knn.nnz and knn_rt.indptr().dtype == np.uint32
    assert np.array_equal(knn_rt.indices(), pure_knn.indices())
    assert np.allclose(knn_rt.distances(), pure_knn.distances())

    node_perps = eg.compute_nodewise_2hop_perplexities(g)
    med_perp = eg.compute_median_2hop_perplexity(g)
    assert node_perps.shape == (n,) and np.all(node_perps > 0.0) and med_perp > 0.0

    hybrid_knn, hybrid_stats = eg.build_hybrid_adaptive_knn(g, jl_emb, node_perps)
    assert hybrid_knn.num_nodes == n
    assert hybrid_stats["n_truncated_1hop"] == 0
    assert hybrid_stats["n_filled_residual"] + hybrid_stats["n_pure_2hop_rows"] == n

    bfs_knn = eg.partial_bfs_knn(g, k=6, seed=42)
    assert bfs_knn.num_nodes == n and bfs_knn.nnz > 0

    # 5. Probabilities, fuzzy simplicial set, find_ab_params, Horvitz-Thompson, DoG Varadhan
    probs = eg.compute_knn_probabilities(hybrid_knn, node_perps)
    probs_scalar = eg.compute_knn_probabilities(hybrid_knn, float(med_perp))
    assert probs.num_nodes == n and abs(float(np.sum(probs.values())) - 1.0) < 1e-10
    assert abs(float(np.sum(probs_scalar.values())) - 1.0) < 1e-10

    fuzzy = eg.compute_fuzzy_simplicial_set(hybrid_knn)
    fvals = fuzzy.values()
    assert fuzzy.num_nodes == n and np.all((fvals > 1e-7) & (fvals <= 1.0))

    a, b = eg.find_ab_params(0.1, 1.0)
    assert abs(a - 1.577) < 0.02 and abs(b - 0.895) < 0.02
    assert abs(eg.compute_horvitz_thompson_gamma(1.0, 1600, 16) - 10.0) < 1e-12

    lm = eg.compute_regularized_dog_varadhan_landmarks(
        g, degree=25, num_landmarks=16, seed=42, t_coarse=48.0
    )
    assert lm.num_nodes == n and lm.num_landmarks == 16 and lm.used_coarse_scaffold is True
    assert lm.table().shape == (n, 16)
    coarse_tbl = lm.coarse_table()
    assert coarse_tbl is not None and coarse_tbl.shape == (n, 16)
    alt = lm.alt_distances([0, 63, 64, 69], [63, 0, 68, 69])
    assert alt.shape == (4,) and alt[3] == 0.0 and abs(alt[0] - alt[1]) < 1e-12

    lm_no_coarse = eg.compute_regularized_dog_varadhan_landmarks(
        g, degree=25, num_landmarks=8, seed=42, t_coarse=0.0
    )
    assert lm_no_coarse.used_coarse_scaffold is False and lm_no_coarse.coarse_table() is None

    # 5b. AmgPcgSolver (SPD, singular/isolated, cooperative stepping, batch)
    amg_reg = eg.AmgPcgSolver.from_graph_regularized_laplacian(g, gamma=10.0)
    b_amg = np.cos(0.17 * np.arange(n, dtype=np.float64))
    x_amg, amg_info = amg_reg.solve(b_amg)
    assert amg_info["converged"] and amg_info["relative_residual"] <= 1e-8
    amg_step = eg.AmgPcgSolver.from_graph_regularized_laplacian(g, gamma=10.0)
    amg_step.start_solve(b_amg)
    if not amg_step.is_finished:
        amg_step.step()
        amg_step.run()
    assert np.array_equal(amg_step.solution(), x_amg)

    b_amg_batch = np.column_stack([b_amg, -2.0 * b_amg])
    x_amg_batch, batch_info = amg_reg.solve_batch(b_amg_batch)
    assert batch_info["all_converged"] and np.allclose(x_amg_batch[:, 0], x_amg, atol=1e-12)

    amg_sing = eg.AmgPcgSolver.from_graph_laplacian(g, kind="combinatorial", shift=0.0)
    b_sing = np.zeros(n, dtype=np.float64)
    b_sing[0] = 1.0
    b_sing[64] = -1.0
    x_sing, sing_info = amg_sing.solve(b_sing)
    assert sing_info["converged"] and float(x_sing[69]) == 0.0

    results.append({
        "case": "linalg_and_neighborhood",
        "status": "pass",
        "elapsed_sec": round(time.perf_counter() - t0, 4),
    })

    # 6. Adversarial linalg/neighborhood checks (must raise ValueError/TypeError/IndexError, never PanicException)
    t1 = time.perf_counter()
    bad_coords = jl_emb.copy()
    bad_coords[0, 0] = float("nan")
    linalg_bad_cases: list[tuple[str, Callable[[], Any]]] = [
        ("KnnGraph.from_csr/bad_indptr", lambda: eg.KnnGraph.from_csr(2, [0, 1], [1], [1.0])),
        ("KnnGraph.from_csr/neg_dist", lambda: eg.KnnGraph.from_csr(2, [0, 1, 1], [1], [-1.0])),
        ("KnnGraph.from_csr/nan_dist", lambda: eg.KnnGraph.from_csr(2, [0, 1, 1], [1], [float("nan")])),
        ("KnnGraph.from_csr/oob_idx", lambda: eg.KnnGraph.from_csr(2, [0, 1, 1], [9], [1.0])),
        ("KnnGraph.from_csr/bool_nodes", lambda: eg.KnnGraph.from_csr(True, [0, 1], [0], [1.0])),
        ("pivot_mds_init/zero_pivots", lambda: eg.pivot_mds_init(g, num_pivots=0)),
        ("pivot_mds_init/bad_dtype", lambda: eg.pivot_mds_init(g, dtype="int32")),
        ("koren_spectral_init/zero_tol", lambda: eg.koren_spectral_init(g, tol=0.0)),
        ("koren_spectral_init/zero_iters", lambda: eg.koren_spectral_init(g, max_iterations=0)),
        ("smallest_laplacian_eigenpairs/zero_modes", lambda: eg.smallest_laplacian_eigenpairs(g, 0)),
        ("chebyshev_heat_spmv/len_mismatch", lambda: eg.chebyshev_heat_spmv(g, np.ones(n - 1), 1.0)),
        ("chebyshev_heat_spmv/nan_vec", lambda: eg.chebyshev_heat_spmv(g, np.full(n, float("nan")), 1.0)),
        ("chebyshev_heat_spmv/neg_t", lambda: eg.chebyshev_heat_spmv(g, deg_vec, -1.0)),
        ("chebyshev_dog_jl_embedding/zero_deg", lambda: eg.chebyshev_dog_jl_embedding(g, degree=0)),
        ("chebyshev_dog_jl_embedding/zero_dim", lambda: eg.chebyshev_dog_jl_embedding(g, jl_dim=0)),
        ("compute_nodewise_2hop_perplexities/bad_c", lambda: eg.compute_nodewise_2hop_perplexities(g, c=0.0)),
        (
            "compute_median_2hop_perplexity/min_gt_max",
            lambda: eg.compute_median_2hop_perplexity(g, min_perplexity=30.0, max_perplexity=10.0),
        ),
        ("build_hybrid_adaptive_knn/nan_coords", lambda: eg.build_hybrid_adaptive_knn(g, bad_coords, node_perps)),
        ("build_hybrid_adaptive_knn/len_perps", lambda: eg.build_hybrid_adaptive_knn(g, jl_emb, node_perps[:-1])),
        ("compute_knn_probabilities/nan_perp", lambda: eg.compute_knn_probabilities(hybrid_knn, float("nan"))),
        ("compute_fuzzy_simplicial_set/zero_iters", lambda: eg.compute_fuzzy_simplicial_set(hybrid_knn, sigma_iters=0)),
        ("find_ab_params/neg_min_dist", lambda: eg.find_ab_params(-0.1, 1.0)),
        ("compute_horvitz_thompson_gamma/neg_gamma", lambda: eg.compute_horvitz_thompson_gamma(-1.0, 10, 4)),
        ("DogVaradhanLandmarks.alt_distances/oob", lambda: lm.alt_distances([n + 5], [0])),
        ("DogVaradhanLandmarks.alt_distances/mismatch", lambda: lm.alt_distances([0, 1], [0])),
        ("compute_regularized_dog_varadhan_landmarks/bad_beta", lambda: eg.compute_regularized_dog_varadhan_landmarks(g, beta_floor=1.0)),
        ("AmgPcgSolver/asym", lambda: eg.AmgPcgSolver(2, [0, 2, 4], [0, 1, 0, 1], [2.0, -1.0, -2.0, 2.0])),
        ("AmgPcgSolver/neg_diag", lambda: eg.AmgPcgSolver(2, [0, 2, 4], [0, 1, 0, 1], [-1.0, 0.0, 0.0, 2.0])),
        ("AmgPcgSolver/solve_nan", lambda: amg_reg.solve(np.full(n, float("nan")))),
        ("AmgPcgSolver/solve_dim", lambda: amg_reg.solve(np.ones(n - 1))),
    ]
    for label, fn in linalg_bad_cases:
        assert_non_panic_error(label, fn)

    results.append({
        "case": "linalg_neighborhood_adversarial_non_panic",
        "status": "pass",
        "elapsed_sec": round(time.perf_counter() - t1, 4),
    })
    return results
