Track A: Pure 1+2-Hop & Row-Normalized Adaptive tsNET (``TsNetSolver``)
=======================================================================

This example demonstrates Track A using ``TsNetSolver.pure_2hop``, ``TsNetSolver.from_graph_row_normalized`` (with ``TwoStepRandomWalk`` affinity and ``backend="auto"``), normalized PivotMDS initialization (``pivot_mds_init``), and selective quality metric evaluation (``evaluate_metrics``).

Basic ``TsNetSolver.pure_2hop`` Workflow
----------------------------------------

.. testcode:: python

    import egraph as eg

    # Build a 3x4 grid graph
    edges = []
    for r in range(3):
        for c in range(4):
            u = r * 4 + c
            if c + 1 < 4:
                edges.append((u, u + 1))
            if r + 1 < 3:
                edges.append((u, u + 4))

    graph = eg.CsrGraph.from_edges(12, edges)
    layout = eg.pivot_mds_init(graph, num_pivots=6, seed=42)

    solver = eg.TsNetSolver.pure_2hop(
        graph,
        backend="auto",
        perplexity_c=0.9,
        stage_iterations=(25, 25, 40),
        seed=42,
    )
    assert solver.resolved_backend == "exact"
    status = solver.run(layout)
    assert solver.is_finished
    assert layout.is_all_finite()

    metrics = eg.evaluate_metrics(graph, layout, ["sns", "np1", "np2", "crossing_number"])
    assert 0.0 <= metrics["scale_normalized_stress"] <= 1.0
    print("Track A status:", status)

.. testoutput:: python

    Track A status: completed

Row-Normalized Affinity (``TsNetSolver.from_graph_row_normalized``)
-------------------------------------------------------------------

``TsNetSolver.from_graph_row_normalized`` builds direct row-normalized $A_1 + w_2 A_2$ affinity (default ``mode="two_step_random_walk"``) without Gaussian binary-search perplexity tuning:

.. testcode:: python

    import egraph as eg

    graph = eg.load_dataset("karate")
    layout = eg.pivot_mds_init(graph, num_pivots=16, seed=42)
    rn_solver = eg.TsNetSolver.from_graph_row_normalized(
        graph,
        w2=0.5,
        mode="two_step_random_walk",
        backend="auto",
        stage_iterations=(15, 15, 20),
        seed=42,
    )
    rn_solver.run(layout)
    assert layout.is_all_finite()
    print("Row-normalized tsNET backend:", rn_solver.resolved_backend)

.. testoutput:: python

    Row-normalized tsNET backend: exact

Custom $k$-NN Input (``TsNetSolver.from_knn``)
-----------------------------------------------

You can also construct a ``TsNetSolver`` from pure 1+2-hop or hybrid adaptive $k$-NN neighborhoods:

.. testcode:: python

    import egraph as eg

    graph = eg.CsrGraph.from_edges(8, [(i, (i + 1) % 8) for i in range(8)] + [(0, 4), (2, 6)])
    knn, _, _ = eg.build_pure_2hop_knn(graph)
    perplexities = eg.compute_nodewise_2hop_perplexities(graph, c=0.9)

    layout = eg.koren_spectral_init(graph, seed=42)
    solver = eg.TsNetSolver.from_knn(
        knn,
        perplexities=perplexities,
        backend="barnes_hut",
        stage_iterations=(15, 15, 20),
        seed=42,
    )
    solver.run(layout)
    assert layout.is_all_finite()
    print("Custom k-NN tsNET complete")

.. testoutput:: python

    Custom k-NN tsNET complete
