Track B: Regularized DoG Sparse Barrier SGD, GuMap & Spatial Refinement
=======================================================================

This example demonstrates Track B: regularized Chebyshev Difference-of-Gaussians (DoG) 3-tier Sparse Barrier SGD (``SgdSolver.sparse_barrier_dog``), ``GuMapSolver.from_graph_dog``, and 2D ``StaticKdTree`` post-layout refinement (``SpatialIntruderRefiner``).

``SgdSolver.sparse_barrier_dog`` + ``SpatialIntruderRefiner``
-------------------------------------------------------------

.. testcode:: python

    import egraph as eg

    # Build a 4x4 grid graph
    edges = []
    for r in range(4):
        for c in range(4):
            u = r * 4 + c
            if c + 1 < 4:
                edges.append((u, u + 1))
            if r + 1 < 4:
                edges.append((u, u + 4))

    graph = eg.CsrGraph.from_edges(16, edges)
    layout = eg.pivot_mds_init(graph, num_pivots=8, seed=42)

    # Track B Sparse Barrier SGD
    sgd = eg.SgdSolver.sparse_barrier_dog(graph, t_max=40, seed=42)
    sgd.run(layout)
    stats = sgd.dog_stats()
    assert stats is not None and stats["n_landmarks"] > 0

    # 2D StaticKdTree spatial intruder refinement
    refiner = eg.SpatialIntruderRefiner(graph, rounds=2, seed=42)
    refiner.run(layout)
    ref_stats = refiner.stats()
    assert layout.is_all_finite() and "rounds_executed" in ref_stats
    print("Track B + SpatialIntruderRefiner complete")

.. testoutput:: python

    Track B + SpatialIntruderRefiner complete

``GuMapSolver.from_graph_dog``
------------------------------

.. testcode:: python

    import egraph as eg

    graph = eg.CsrGraph.from_edges(10, [(i, (i + 1) % 10) for i in range(10)] + [(0, 5)])
    layout = eg.pivot_mds_init(graph, num_pivots=6, seed=42)

    gumap = eg.GuMapSolver.from_graph_dog(graph, n_epochs=40, seed=42)
    gumap.run(layout)
    assert layout.is_all_finite()
    print("GuMap complete, ab len:", len(gumap.ab))

.. testoutput:: python

    GuMap complete, ab len: 2
