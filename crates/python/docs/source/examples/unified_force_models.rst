Unified Force & Multilevel Models (t-FDP, ibUMAP, SNAP-tFDP, PaCMAP, FA2, Yifan Hu)
======================================================================================

This example demonstrates the unified force-directed and sampling embedding solvers introduced in v6, all sharing the cooperative ``step()`` / ``run_steps()`` / ``run()`` interface over ``CsrGraph`` and ``Layout2d``.

``TFdpSolver``, ``IbUmapSolver``, and ``SnapTFdpSolver``
--------------------------------------------------------

.. testcode:: python

    import egraph as eg

    edges = [(i, (i + 1) % 12) for i in range(12)] + [(0, 6), (3, 9)]
    graph = eg.CsrGraph.from_edges(12, edges)

    # Bounded Student-t FDP (ibFFT backend)
    l_tfdp = eg.pivot_mds_init(graph, num_pivots=6, seed=42)
    eg.TFdpSolver(graph, backend="ib_fft", iterations=30, seed=42).run(l_tfdp)
    assert l_tfdp.is_all_finite()

    # Interpolation-Based UMAP (ibUMAP)
    l_ibumap = eg.pivot_mds_init(graph, num_pivots=6, seed=42)
    eg.IbUmapSolver(graph, backend="ib_fft", iterations=30, seed=42).run(l_ibumap)
    assert l_ibumap.is_all_finite()

    # SNAP-tFDP (edge-incident negative sampling, low peak memory)
    l_snap = eg.pivot_mds_init(graph, num_pivots=6, seed=42, dtype="float32")
    eg.SnapTFdpSolver(graph, iterations=30, seed=42, dtype="float32").run(l_snap)
    assert l_snap.is_all_finite()

    print("t-FDP, ibUMAP, and SNAP-tFDP complete")

.. testoutput:: python

    t-FDP, ibUMAP, and SNAP-tFDP complete

``PacmapGraphSolver``, ``ForceAtlas2Solver``, and ``YifanHuSolver``
-------------------------------------------------------------------

.. testcode:: python

    import egraph as eg

    edges = [(i, (i + 1) % 12) for i in range(12)] + [(0, 6), (3, 9)]
    graph = eg.CsrGraph.from_edges(12, edges)

    # PaCMAP-Graph (3-phase near/mid/far pair schedule)
    l_pacmap = eg.pivot_mds_init(graph, num_pivots=6, seed=42)
    eg.PacmapGraphSolver(
        graph, phase_iterations=(10, 10, 15), seed=42
    ).run(l_pacmap)
    assert l_pacmap.is_all_finite()

    # ForceAtlas2 (Barnes-Hut)
    l_fa2 = eg.pivot_mds_init(graph, num_pivots=6, seed=42)
    eg.ForceAtlas2Solver(graph, backend="barnes_hut", iterations=30, seed=42).run(l_fa2)
    assert l_fa2.is_all_finite()

    # Multilevel Yifan Hu (EC+MIVS coarsening)
    l_hu = eg.pivot_mds_init(graph, num_pivots=6, seed=42)
    eg.YifanHuSolver(graph, max_iterations_per_level=20, seed=42).run(l_hu)
    assert l_hu.is_all_finite()

    print("PaCMAP-Graph, ForceAtlas2, and Yifan Hu complete")

.. testoutput:: python

    PaCMAP-Graph, ForceAtlas2, and Yifan Hu complete
